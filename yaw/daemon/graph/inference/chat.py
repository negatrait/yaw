"""Native DSPy LM and explicit local chat policy with SQLite response caching."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
import functools
import hashlib
import json
import os
import re
import sqlite3
from typing import Any

import dspy
from filelock import FileLock
from openai import DefaultHttpxClient, OpenAI

from .models import BASE_URL, CHAT_MODELS, DEFAULT_EXTRACTION_CHAT, DEFAULT_QA_CHAT
from .prompts import (
    PROMPT_JSON_TEMPLATE,
    best_dspy_prompt,
    ner_conditioned_re_system,
    ner_system,
    rag_qa_system,
)


RECIPE_POLICY_TAG = {
    "flm": "think-off:prompt-v3:require-stop",
    "ryzenai-llm": "fold-system:prompt-v4:require-stop",
    "llamacpp": "gbnf-v3:require-stop",
}

TIGHTENED_QA_SYSTEM = (
    f"{rag_qa_system} "
    "For 'Who', 'What', 'When', or 'Where' questions, state only the exact entity or fact. "
    "For 'How' or 'Why' questions, state both the key condition/means and the resulting outcome from the passages."
)
QA_TURN_REMINDER = (
    "Based on the passages above, reason step by step and conclude on a new line "
    "with 'Answer: ' followed only by the concise final answer."
)
SYSTEM_PROMPT_TIGHTENINGS = {
    rag_qa_system: TIGHTENED_QA_SYSTEM,
}


def _tightened_openie_schemas():
    ner_schema = deepcopy(PROMPT_JSON_TEMPLATE["ner"])
    ner_schema["properties"]["named_entities"]["items"]["minLength"] = 1
    triple_schema = deepcopy(PROMPT_JSON_TEMPLATE["triples"])
    triple_schema["properties"]["triples"]["items"]["items"]["minLength"] = 1
    return {
        ner_system: {
            "type": "json_schema",
            "json_schema": {"name": "ner", "schema": ner_schema, "strict": True},
        },
        ner_conditioned_re_system: {
            "type": "json_schema",
            "json_schema": {"name": "triples", "schema": triple_schema, "strict": True},
        },
    }


OPENIE_JSON_SCHEMAS = _tightened_openie_schemas()
DSPY_FILTER_SYSTEM = best_dspy_prompt["prog"]["system"]
RERANK_GBNF_GRAMMAR = r"""root ::= "[[ ## fact_after_filter ## ]]\n" fact-obj "\n\n[[ ## completed ## ]]"
fact-obj ::= "{\"fact\": [" fact-list "]}"
fact-list ::= (triple (", " triple)*)?
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""
QA_GBNF_GRAMMAR = r"""root ::= thought "\nAnswer: " answer
thought ::= char+
answer ::= char+
char ::= [^\r\n\x00-\x1F]
"""


def _gbnf_literal(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _build_rerank_grammar(messages, max_facts: int = 4) -> str:
    if not messages or messages[-1].get("role") != "user":
        return RERANK_GBNF_GRAMMAR
    content = messages[-1].get("content", "")
    marker = "[[ ## fact_before_filter ## ]]\n"
    if marker not in content:
        return RERANK_GBNF_GRAMMAR
    after_marker = content.split(marker, 1)[1]
    raw_json = after_marker.split("\n\n", 1)[0].strip()
    try:
        payload = json.loads(raw_json)
        facts = payload["fact"]
        if not isinstance(facts, list):
            return RERANK_GBNF_GRAMMAR
        candidates = []
        for item in facts:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 3
                or not all(isinstance(x, str) for x in item)
            ):
                return RERANK_GBNF_GRAMMAR
            serialized = json.dumps(list(item), ensure_ascii=False)
            if serialized not in candidates:
                candidates.append(serialized)
    except (KeyError, TypeError, ValueError):
        return RERANK_GBNF_GRAMMAR

    if not candidates:
        return (
            'root ::= "[[ ## fact_after_filter ## ]]\\n" '
            '"{\\"fact\\": []}" '
            '"\\n\\n[[ ## completed ## ]]"\n'
        )

    limit = min(max_facts, len(candidates))
    seq = "triple"
    for _ in range(limit - 1):
        seq = f'triple (", " {seq})?'
    triple_rule = " | ".join(_gbnf_literal(c) for c in candidates)
    return (
        'root ::= "[[ ## fact_after_filter ## ]]\\n" fact-obj "\\n\\n[[ ## completed ## ]]"\n'
        'fact-obj ::= "{\\"fact\\": [" fact-list "]}"\n'
        f"fact-list ::= ({seq})?\n"
        f"triple ::= {triple_rule}\n"
    )


def _normalize_qa_response(response: str, *, allow_missing_delimiter: bool = False) -> str:
    normalized = re.sub(r"\*{1,2}Answer\*{0,2}:\*{0,2}", "Answer:", response)
    normalized = re.sub(r"\bAnswer\s+:", "Answer:", normalized)
    if "Answer:" not in normalized:
        if not allow_missing_delimiter:
            raise ValueError("QA response omitted required 'Answer:' delimiter.")
        if not normalized.strip():
            raise ValueError("QA response omitted required 'Answer:' content.")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", normalized.strip()) if s.strip()]
        return f"{normalized.rstrip()}\nAnswer: {sentences[-1]}"
    parts = normalized.split("Answer:")
    if not parts[-1].strip():
        raise ValueError("QA response omitted required 'Answer:' content.")
    if len(parts) > 2:
        normalized = "Answer -".join(parts[:-1]) + "Answer:" + parts[-1]
    return normalized


def _repair_truncated_openie_json(response: str, *, is_triples: bool) -> str:
    try:
        json.loads(response)
        return response
    except ValueError:
        pass
    depth, in_str, esc, last_end = 0, False, False, -1
    for i, ch in enumerate(response):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if not is_triples and depth == 2:
                    last_end = i + 1
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            if is_triples and depth == 3 and ch == "]":
                last_end = i + 1
            depth -= 1
    if last_end == -1:
        return '{"triples": []}' if is_triples else '{"named_entities": []}'
    return response[:last_end] + "]}"


def chat_identity(model):
    checkpoint, recipe = CHAT_MODELS[model]
    policy = RECIPE_POLICY_TAG[recipe]
    return f"chat-v1:{model}:{checkpoint}:{recipe}:{policy}"


def create_dspy_lm(
    model_name: str = DEFAULT_EXTRACTION_CHAT,
    *,
    base_url: str = BASE_URL,
    api_key: str = "lemonade",
    provider: str = "openai",
    role: str = "extraction",
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    num_retries: int = 0,
    **kwargs: Any,
) -> dspy.LM:
    """Create a native dspy.LM configured for Lemonade (openai/...) or Ollama (ollama_chat/...).

    Applies recipe-aware and role-aware defaults for our 3-slot hardware stack:
    - Qwen3-4B-Instruct-2507-GGUF (llamacpp on Vulkan GPU, 8192-16384 ctx): supports GBNF, temp=0.2, freq_penalty=0.5.
    - Mistral-7B-Instruct-v0.3-Hybrid (ryzenai-llm on NPU/OGA, 32786 ctx): NO GBNF, orchestrator for RLM.
    - gemma4-it-12b-FLM (flm): extra_body={"think": False}.
    """
    clean_name = model_name
    for prefix in ("openai/", "ollama_chat/", "ollama/"):
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix) :]
            break

    profile = CHAT_MODELS.get(clean_name)
    recipe = profile.recipe if profile is not None else None

    if temperature is None:
        if role == "extraction" and profile is not None:
            temperature = profile.extraction_temperature
        elif role in ("qa", "rlm") and profile is not None and profile.qa_temperature > 0:
            temperature = profile.qa_temperature
        else:
            temperature = 0.0

    if max_tokens is None:
        # Qwen3-4B runs with an 8192-16384 context window; Mistral-7B Hybrid runs with 32786.
        max_tokens = 4096 if role == "extraction" else 2048

    lm_kwargs: dict[str, Any] = dict(kwargs)
    if role == "extraction" and profile is not None and profile.frequency_penalty > 0:
        lm_kwargs.setdefault("frequency_penalty", profile.frequency_penalty)
    if recipe == "flm":
        lm_kwargs.setdefault("extra_body", {"think": False})

    if provider == "ollama" or model_name.startswith(("ollama_chat/", "ollama/")):
        litellm_model = (
            model_name
            if model_name.startswith(("ollama_chat/", "ollama/"))
            else f"ollama_chat/{clean_name}"
        )
        lm = dspy.LM(
            model=litellm_model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            api_base=base_url,
            **lm_kwargs,
        )
    else:
        litellm_model = model_name if "/" in model_name else f"openai/{clean_name}"
        lm = dspy.LM(
            model=litellm_model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            api_base=base_url,
            api_key=api_key,
            **lm_kwargs,
        )

    lm._yaw_recipe = recipe
    lm._yaw_supports_gbnf = recipe == "llamacpp"
    return lm


def create_dspy_rlm_pair(
    orchestrator_model: str = DEFAULT_QA_CHAT,
    sub_model: str = DEFAULT_EXTRACTION_CHAT,
    *,
    base_url: str = BASE_URL,
    api_key: str = "lemonade",
    provider: str = "openai",
    cache: bool = True,
    num_retries: int = 0,
) -> tuple[dspy.LM, dspy.LM]:
    """Create the heterogeneous (Mistral NPU Orchestrator, Qwen GPU GBNF Worker) LM pair.

    - `orchestrator_lm` (`Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI NPU/OGA): runs outer `dspy.RLM`
      Python REPL orchestration without GBNF (`_yaw_supports_gbnf=False`).
    - `sub_lm` (`Qwen3-4B-Instruct-2507-GGUF` on `llama.cpp` Vulkan GPU): runs inner `llm_query` and
      GBNF-constrained `qwen_*` extraction/reranking/CoT worker calls (`_yaw_supports_gbnf=True`).
    """
    orchestrator_lm = create_dspy_lm(
        orchestrator_model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        role="rlm",
        cache=cache,
        num_retries=num_retries,
    )
    sub_lm = create_dspy_lm(
        sub_model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        role="extraction",
        cache=cache,
        num_retries=num_retries,
    )
    return orchestrator_lm, sub_lm


@dataclass
class _LLMConfig:
    llm_name: str
    llm_base_url: str | None = None
    generate_params: dict[str, Any] = field(default_factory=dict)


def _normalize_generation_token_params(
    base_params: dict[str, Any],
    call_kwargs: dict[str, Any],
    target_token_key: str,
) -> dict[str, Any]:
    params = deepcopy(base_params)
    params.update(call_kwargs)
    token_val = None
    for key in ("max_new_tokens", "max_tokens", "max_completion_tokens"):
        if key in params:
            val = params.pop(key)
            if val is not None:
                token_val = val
    if token_val is not None:
        params[target_token_key] = token_val
    return params


def cache_response(func):
    """SQLite-backed deterministic response cache using FileLock for concurrent safety."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        global_config = getattr(self, "global_config", None)
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = (
            getattr(self, "llm_config", None).generate_params
            if hasattr(self, "llm_config") and self.llm_config is not None
            else {}
        )
        key_data = _normalize_generation_token_params(gen_params, kwargs, "_max_tokens")
        key_data["messages"] = messages
        key_data["_cache_schema"] = 3
        key_data["_provider_class"] = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        key_data["_endpoint"] = getattr(global_config, "azure_endpoint", None) or getattr(
            global_config, "llm_base_url", None
        )
        key_data["_azure_api_version"] = getattr(global_config, "azure_api_version", None)
        key_data["_azure_chat_deployment"] = getattr(global_config, "azure_chat_deployment", None)
        if key_data.get("store") is True:
            message, metadata = func(self, *args, **kwargs)
            return message, metadata, False

        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        database_lock_file = self.cache_file_name + ".lock"
        key_lock_file = f"{self.cache_file_name}.{key_hash[:2]}.lock"
        with FileLock(key_lock_file):
            with FileLock(database_lock_file):
                with sqlite3.connect(self.cache_file_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS cache (
                            key TEXT PRIMARY KEY,
                            message TEXT,
                            metadata TEXT
                        )
                        """
                    )
                    cursor.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                    row = cursor.fetchone()
            if row is not None:
                message, metadata_str = row
                return message, json.loads(metadata_str), True

            message, metadata = func(self, *args, **kwargs)

            with FileLock(database_lock_file):
                with sqlite3.connect(self.cache_file_name) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                        (key_hash, message, json.dumps(metadata)),
                    )

        return message, metadata, False

    return wrapper


class CacheOpenAI:
    """Self-contained OpenAI-compatible chat client with SQLite caching (zero hipporag imports)."""

    @classmethod
    def from_experiment_config(cls, global_config) -> "CacheOpenAI":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(
            cache_dir=cache_dir,
            global_config=global_config,
            max_retries=getattr(global_config, "max_retry_attempts", 0),
        )

    def __init__(
        self,
        cache_dir: str,
        global_config: Any,
        cache_filename: str | None = None,
        high_throughput: bool = True,
        **kwargs: Any,
    ) -> None:
        self.cache_dir = cache_dir
        self.global_config = global_config
        self.llm_name = global_config.llm_name
        self.llm_base_url = getattr(global_config, "llm_base_url", None) or BASE_URL

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self.max_retries = kwargs.get("max_retries", 2)
        self.request_model_name = self.llm_name
        self._init_llm_config()

        api_key_env = getattr(global_config, "llm_api_key_env", None)
        api_key = (os.environ.get(api_key_env) if api_key_env else None) or "lemonade"
        client = DefaultHttpxClient(timeout=5 * 60) if high_throughput else None
        try:
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=self.llm_base_url,
                http_client=client,
                max_retries=self.max_retries,
            )
        except Exception:
            if client is not None:
                client.close()
            raise

    def _init_llm_config(self) -> None:
        generate_params: dict[str, Any] = {
            "model": self.request_model_name,
        }
        max_new_tokens = getattr(self.global_config, "max_new_tokens", None)
        if max_new_tokens is not None:
            generate_params["max_completion_tokens"] = max_new_tokens
        seed = getattr(self.global_config, "seed", None)
        if seed is not None:
            generate_params["seed"] = seed
        temperature = getattr(self.global_config, "temperature", None)
        if temperature is not None:
            generate_params["temperature"] = temperature
        self.llm_config = _LLMConfig(
            llm_name=self.global_config.llm_name,
            llm_base_url=self.global_config.llm_base_url,
            generate_params=generate_params,
        )

    @cache_response
    def infer(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict[str, Any], bool]:
        params = _normalize_generation_token_params(
            self.llm_config.generate_params, kwargs, "max_tokens"
        )
        params["messages"] = messages

        response = self.openai_client.chat.completions.create(**params)
        if len(response.choices) != 1:
            raise ValueError(
                f"Expected exactly one OpenAI choice, received {len(response.choices)}."
            )
        choice = response.choices[0]
        response_message = choice.message.content
        if not isinstance(response_message, str) or not response_message:
            refusal = getattr(choice.message, "refusal", None)
            detail = f" Refusal: {refusal}" if refusal else ""
            raise ValueError(f"OpenAI response did not contain non-empty text.{detail}")
        usage = response.usage
        if usage is None:
            raise ValueError(
                "OpenAI response omitted usage; cannot account for this request safely."
            )
        total_tokens = getattr(usage, "total_tokens", None)
        metadata: dict[str, Any] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else usage.prompt_tokens + usage.completion_tokens
            ),
            "finish_reason": choice.finish_reason,
        }
        for key, value in (
            ("response_id", getattr(response, "id", None)),
            ("model", getattr(response, "model", None)),
            ("request_id", getattr(response, "_request_id", None)),
        ):
            if value is not None:
                metadata[key] = value
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if cached_tokens is not None:
            metadata["cached_tokens"] = cached_tokens
        if reasoning_tokens is not None:
            metadata["reasoning_tokens"] = reasoning_tokens

        return response_message, metadata

    def close(self) -> None:
        self.openai_client.close()


class LocalChat(CacheOpenAI):
    @classmethod
    def from_experiment_config(cls, global_config, model_name=None):
        config = replace(global_config, llm_name=model_name) if model_name else global_config
        return super().from_experiment_config(config)

    def _init_llm_config(self):
        super()._init_llm_config()
        self.calls = []
        self.errors = []
        if CHAT_MODELS[self.llm_name][1] == "flm":
            # FLM's documented request field, not llama.cpp chat_template_kwargs.
            self.llm_config.generate_params["extra_body"] = {"think": False}

    def infer(self, messages, **kwargs):
        recipe = CHAT_MODELS[self.llm_name][1]
        raw_system = (
            messages[0].get("content")
            if messages and messages[0].get("role") == "system"
            else None
        )
        if raw_system in SYSTEM_PROMPT_TIGHTENINGS:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_TIGHTENINGS[raw_system]},
                *messages[1:],
            ]
        profile = CHAT_MODELS.get(self.llm_name)
        if recipe == "ryzenai-llm":
            if profile is not None and getattr(profile, "qa_temperature", 0.0) > 0:
                current_temp = kwargs.get("temperature", getattr(self.global_config, "temperature", 0.0))
                if current_temp < profile.qa_temperature:
                    kwargs = {**kwargs, "temperature": profile.qa_temperature}
            if (len(messages) >= 2
                    and messages[0].get("role") == "system" and messages[1].get("role") == "user"):
                # Ryzen AI's Mistral template injects system into the final [INST] turn after
                # few-shot turns; folding it into the first user turn preserves one-shot formatting.
                messages = [
                    {"role": "user", "content": messages[0]["content"].strip() + "\n\n" + messages[1]["content"]},
                    *messages[2:],
                ]
            if (raw_system == rag_qa_system and len(messages) >= 2
                    and messages[-1].get("role") == "user"):
                user_text = messages[-1].get("content", "")
                if QA_TURN_REMINDER not in user_text and "\nQuestion: " in user_text:
                    head, tail = user_text.rsplit("\nQuestion: ", 1)
                    messages = [
                        *messages[:-1],
                        {"role": "user", "content": f"{head}\n{QA_TURN_REMINDER}\nQuestion: {tail}"},
                    ]
        elif recipe == "llamacpp" and raw_system is not None:
            if raw_system in OPENIE_JSON_SCHEMAS and "response_format" not in kwargs:
                kwargs = {**kwargs, "response_format": OPENIE_JSON_SCHEMAS[raw_system]}
                profile = CHAT_MODELS.get(self.llm_name)
                ext_temp = getattr(profile, "extraction_temperature", 0.2)
                current_temp = kwargs.get("temperature", getattr(self.global_config, "temperature", 0.0))
                if current_temp < ext_temp:
                    kwargs = {**kwargs, "temperature": ext_temp}
                if raw_system == ner_conditioned_re_system:
                    freq_pen = getattr(profile, "frequency_penalty", 0.5)
                    if freq_pen > 0 and "frequency_penalty" not in kwargs:
                        kwargs = {**kwargs, "frequency_penalty": freq_pen}
            elif raw_system == DSPY_FILTER_SYSTEM and "extra_body" not in kwargs:
                kwargs = {**kwargs, "extra_body": {"grammar": _build_rerank_grammar(messages)}}
            elif raw_system == rag_qa_system and "extra_body" not in kwargs:
                kwargs = {**kwargs, "extra_body": {"grammar": QA_GBNF_GRAMMAR}}
        try:
            result = super().infer(messages, **kwargs)
            text = result[0]
            meta = result[1]
            finish_reason = meta.get("finish_reason")
            if recipe == "ryzenai-llm" and isinstance(text, str) and r"\_" in text:
                # Mistral v0.3 occasionally escapes underscores as \_ inside JSON keys.
                text = text.replace(r"\_", "_")
            if raw_system in OPENIE_JSON_SCHEMAS and finish_reason == "length" and isinstance(text, str):
                text = _repair_truncated_openie_json(text, is_triples=(raw_system == ner_conditioned_re_system))
            if raw_system == rag_qa_system and isinstance(text, str):
                text = _normalize_qa_response(text, allow_missing_delimiter=(recipe != "llamacpp"))
            if text is not result[0]:
                result = (text, *result[1:])
            allowed_finish = ("stop", "length") if raw_system in OPENIE_JSON_SCHEMAS else ("stop",)
            if finish_reason not in allowed_finish:
                raise ValueError(f"Chat response did not finish normally: {finish_reason}")
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            raise
        self.calls.append({
            "finish_reason": meta.get("finish_reason"),
            "cache_hit": bool(result[2]),
            "prompt_tokens": int(meta.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(meta.get("completion_tokens", 0) or 0),
            "total_tokens": int(meta.get("total_tokens", 0) or 0),
        })
        return result

    def summary(self):
        return {
            "model": self.llm_name,
            "recipe": CHAT_MODELS[self.llm_name][1],
            "calls": len(self.calls),
            "cache_hits": sum(1 for c in self.calls if c["cache_hit"]),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "errors": list(self.errors),
        }
