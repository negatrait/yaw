"""Explicit local chat policy, using HippoRAG's client and cache."""

from copy import deepcopy
from dataclasses import replace
import json
import re

from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.prompts.filter_default_prompt import best_dspy_prompt
from hipporag.prompts.templates.ner import ner_system
from hipporag.prompts.templates.rag_qa_musique import rag_qa_system
from hipporag.prompts.templates.triple_extraction import ner_conditioned_re_system
from hipporag.utils.llm_utils import PROMPT_JSON_TEMPLATE

from .models import CHAT_MODELS

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


def chat_identity(model):
    checkpoint, recipe = CHAT_MODELS[model]
    policy = RECIPE_POLICY_TAG[recipe]
    return f"chat-v1:{model}:{checkpoint}:{recipe}:{policy}"


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
            if recipe == "ryzenai-llm" and isinstance(text, str) and r"\_" in text:
                # Mistral v0.3 occasionally escapes underscores as \_ inside JSON keys.
                text = text.replace(r"\_", "_")
            if raw_system == rag_qa_system and isinstance(text, str):
                text = _normalize_qa_response(text, allow_missing_delimiter=(recipe != "llamacpp"))
            if text is not result[0]:
                result = (text, *result[1:])
            meta = result[1]
            if meta.get("finish_reason") != "stop":
                raise ValueError(f"Chat response did not finish normally: {meta.get('finish_reason')}")
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



