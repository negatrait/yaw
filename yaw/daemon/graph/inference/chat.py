import logging
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import dspy

from yaw.core.config import scale_context_budget

from .grammars import lm_supports_gbnf
from .models import BASE_URL, CHAT_MODELS, DEFAULT_EXTRACTION_CHAT, DEFAULT_QA_CHAT

for _logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
if hasattr(dspy, "disable_litellm_logging"):
    dspy.disable_litellm_logging()

_YAW_SLEEP_SECONDS: ContextVar[float] = ContextVar("yaw_sleep_seconds", default=0.0)


class _UsageAccountingCache:
    """Delegate storage/keys to DSPy while replaying logical inference cost.

    DSPy intentionally records zero physical usage on hits. YAW's optimization
    metric needs the original cost instead, so cached candidates do not get an
    artificial advantage. LM history retains DSPy's zero physical usage.
    """

    def __init__(self, backend: Any, account_cached_usage: bool = True) -> None:
        self.backend = backend
        self.account_cached_usage = account_cached_usage
        self.cache_hits = 0
        self.cached_tokens = 0
        # FanoutCache constructs connections on this thread. Do not transfer
        # those handles to short-lived DSPy evaluation/daemon worker lifetimes.
        self.close()

    def close(self) -> None:
        """Close this thread's disk handles; persisted entries remain available."""
        disk_cache = getattr(self.backend, "disk_cache", None)
        close = getattr(disk_cache, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    @staticmethod
    def _is_lm_request(request: dict[str, Any]) -> bool:
        return str(request.get("_fn_identifier", "")).startswith(("dspy.clients.lm.", "dspy.clients.lm15."))

    @staticmethod
    def _ignored(ignored: list[str] | None) -> list[str]:
        # DSPy excludes endpoints by default; local servers may serve different
        # weights under the same model name. Retain endpoints in YAW cache keys.
        return [key for key in (ignored or ()) if key not in {"api_base", "base_url"}]

    def _usage_key(self, request: dict[str, Any], ignored: list[str] | None) -> dict[str, str]:
        return {"yaw_logical_usage_v1": self.backend.cache_key(request, ignored)}

    def get(
        self,
        request: dict[str, Any],
        ignored_args_for_cache_key: list[str] | None = None,
    ) -> Any:
        try:
            return self._get(request, ignored_args_for_cache_key)
        finally:
            self.close()

    def _get(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None) -> Any:
        ignored_args_for_cache_key = self._ignored(ignored_args_for_cache_key)
        record = self.backend.get(request, ignored_args_for_cache_key)
        if record is None or not self.account_cached_usage or not self._is_lm_request(request):
            return record
        if isinstance(record, dict) and "_dspy_format" in record:
            usage = record.get("usage")
        else:
            # Legacy SDK responses have their usage cleared by DSPy's get().
            # New writes retain a small usage companion in the same disk cache.
            metadata = self.backend.get(self._usage_key(request, ignored_args_for_cache_key))
            if metadata is None:
                return None  # Unverifiable old usage: recompute instead of scoring it as zero.
            usage = metadata.get("usage")
        model = request.get("model") or request.get("request", {}).get("model")
        if not isinstance(usage, Mapping) or not isinstance(model, str):
            return None
        tracker = dspy.settings.usage_tracker
        if tracker is not None:
            tracker.add_usage(model, dict(usage))
        self.cache_hits += 1
        self.cached_tokens += int(
            usage.get("total_tokens")
            or (
                (usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                + (usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            )
        )
        return record

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: list[str] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        try:
            self._put(request, value, ignored_args_for_cache_key, enable_memory_cache)
        finally:
            self.close()

    def _put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: list[str] | None,
        enable_memory_cache: bool,
    ) -> None:
        ignored_args_for_cache_key = self._ignored(ignored_args_for_cache_key)
        self.backend.put(request, value, ignored_args_for_cache_key, enable_memory_cache)
        if not self.account_cached_usage or not self._is_lm_request(request):
            return
        if not (isinstance(value, dict) and "_dspy_format" in value):
            usage = getattr(value, "usage", None)
            if usage is None and isinstance(value, Mapping):
                usage = value.get("usage")
            self.backend.put(
                self._usage_key(request, ignored_args_for_cache_key),
                {"usage": dict(usage or {})},
                enable_memory_cache=enable_memory_cache,
            )


def get_yaw_sleep_seconds() -> float:
    """Return the currently configured sleep duration injected between inference calls."""
    return _YAW_SLEEP_SECONDS.get()


def set_yaw_sleep_seconds(seconds: float) -> None:
    """Set the sleep duration injected between inference calls (0.0 to disable)."""
    _YAW_SLEEP_SECONDS.set(max(0.0, float(seconds or 0.0)))


__all__ = [
    "YawDSPyLM",
    "configure_yaw_dspy_cache",
    "create_dspy_lm",
    "create_dspy_rlm_pair",
    "get_yaw_sleep_seconds",
    "lm_supports_gbnf",
    "set_yaw_sleep_seconds",
]


_THINKING_CHANNEL_RE = re.compile(r"<\|channel\>thought.*?<channel\|>", re.DOTALL)


def strip_thinking_channel(text: str) -> str:
    """Strip Gemma 4 / thinking control blocks from model completions."""
    if "<|channel>thought" in text:
        text = _THINKING_CHANNEL_RE.sub("", text)
    return text.lstrip()


class YawDSPyLM(dspy.LM):
    """Native dspy.LM wrapper preserving recipe metadata across copies without overriding BaseLM.forward."""

    _yaw_recipe: str | None = None
    _yaw_context_window_tokens: int | None = None
    _yaw_supports_gbnf: bool = False

    def __call__(self, prompt: Any = None, *, messages: Any = None, **kwargs: Any) -> Any:
        res = super().__call__(prompt=prompt, messages=messages, **kwargs)
        if isinstance(res, list):
            res = [strip_thinking_channel(item) if isinstance(item, str) else item for item in res]
        elif isinstance(res, str):
            res = strip_thinking_channel(res)
        sleep_sec = get_yaw_sleep_seconds()
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return res

    def copy(self, **kwargs: Any) -> YawDSPyLM:
        copied = super().copy(**kwargs)
        if not isinstance(copied, YawDSPyLM):
            raise TypeError("DSPy returned an incompatible LM copy")
        copied._yaw_context_window_tokens = getattr(self, "_yaw_context_window_tokens", None)
        copied._yaw_recipe = getattr(self, "_yaw_recipe", None)
        copied._yaw_supports_gbnf = getattr(self, "_yaw_supports_gbnf", False)
        return copied


def configure_yaw_dspy_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    enable_disk_cache: bool = True,
    enable_memory_cache: bool = True,
    disk_size_limit_bytes: int = 1_073_741_824,
    memory_max_entries: int = 100_000,
    restrict_pickle: bool = True,
    safe_types: list[type[Any]] | None = None,
    account_cached_usage: bool = False,
) -> None:
    """Configure workspace-isolated DSPy cache via dspy.configure_cache with restrict_pickle=True by default."""
    resolved_dir: str | None = None
    if cache_dir is not None:
        p = Path(cache_dir)
        if enable_disk_cache:
            p.mkdir(parents=True, exist_ok=True)
        resolved_dir = str(p)
    elif enable_disk_cache:
        p = Path(".yaw") / "dspy_cache"
        p.mkdir(parents=True, exist_ok=True)
        resolved_dir = str(p)
    kwargs: dict[str, Any] = {
        "enable_disk_cache": bool(enable_disk_cache),
        "enable_memory_cache": bool(enable_memory_cache),
        "disk_size_limit_bytes": int(disk_size_limit_bytes),
        "memory_max_entries": int(memory_max_entries),
        "restrict_pickle": bool(restrict_pickle),
        "safe_types": safe_types,
    }
    if resolved_dir is not None:
        kwargs["disk_cache_dir"] = resolved_dir
    old_cache = getattr(dspy, "cache", None)
    old_disk = getattr(old_cache, "disk_cache", None) if old_cache is not None else None
    close_fn = getattr(old_disk, "close", None)
    if callable(close_fn):
        close_fn()
    dspy.configure_cache(**kwargs)
    dspy.cache = _UsageAccountingCache(dspy.cache, account_cached_usage=account_cached_usage)


def create_dspy_lm(
    model_name: str = DEFAULT_EXTRACTION_CHAT,
    *,
    base_url: str = BASE_URL,
    api_key: str = "lemonade",
    provider: str = "openai",
    role: str = "extraction",
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    context_window_tokens: int | None = None,
    cache: bool = True,
    timeout: float = 240.0,
    **kwargs: Any,
) -> YawDSPyLM:
    """Create a native dspy.LM configured for Lemonade (openai/...) or Ollama (ollama_chat/...).

    Applies recipe-aware and role-aware defaults for our local hardware stack:
    - gemma-4-E4B-it-qat-q4_0-gguf-Q4_0 (llamacpp GPU): default chat/reflection model;
      direct/worker roles support GBNF, while RLM orchestrators generate plain code.
      Standardized Gemma 4 sampling: temperature=1.0, top_p=0.95, top_k=64.
    - Qwen3-4B-Instruct-2507-GGUF (llamacpp on Vulkan GPU, 16384 ctx): supports GBNF for direct/worker
      roles (`extraction`, `qa`, `reflection_worker`) with temp=0.35, and runs
      unconstrained Python REPL code generation (`_yaw_supports_gbnf=False`) for RLM orchestrator roles
      (`rlm`, `reflection`).
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
        if role == "reflection":
            temperature = profile.reflection_temperature if profile is not None else 0.8
        elif role == "reflection_worker":
            temperature = profile.reflection_worker_temperature if profile is not None else 0.7
        elif role == "extraction" and profile is not None:
            temperature = profile.extraction_temperature
        elif role in ("qa", "rlm") and profile is not None and profile.qa_temperature > 0:
            temperature = profile.qa_temperature
        else:
            temperature = profile.default_temperature if profile is not None else 0.35

    if top_p is None and profile is not None and profile.top_p is not None:
        top_p = profile.top_p
    if top_k is None and profile is not None and profile.top_k is not None:
        top_k = profile.top_k

    context_tokens = (
        context_window_tokens if context_window_tokens is not None else profile.expected_context if profile else None
    )
    if max_tokens is None:
        max_tokens = (
            scale_context_budget(4096, context_tokens)
            if context_tokens is not None and (context_window_tokens is not None or context_tokens >= 65536)
            else 4096
        )
    if context_tokens is not None and not 0 < max_tokens < context_tokens:
        raise ValueError("Require 0 < max_tokens < context_window_tokens")

    lm_kwargs: dict[str, Any] = dict(kwargs)
    if top_p is not None:
        lm_kwargs.setdefault("top_p", top_p)
    if top_k is not None:
        lm_kwargs.setdefault("top_k", top_k)
        extra = dict(lm_kwargs.get("extra_body") or {})
        extra.setdefault("top_k", top_k)
        lm_kwargs["extra_body"] = extra
    if recipe == "flm":
        extra = dict(lm_kwargs.get("extra_body") or {})
        extra.setdefault("think", False)
        lm_kwargs["extra_body"] = extra

    if provider == "ollama" or model_name.startswith(("ollama_chat/", "ollama/")):
        litellm_model = (
            model_name if model_name.startswith(("ollama_chat/", "ollama/")) else f"ollama_chat/{clean_name}"
        )
        lm = YawDSPyLM(
            model=litellm_model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            api_base=base_url,
            timeout=timeout,
            **lm_kwargs,
        )
    else:
        litellm_model = model_name if "/" in model_name else f"openai/{clean_name}"
        lm = YawDSPyLM(
            model=litellm_model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            api_base=base_url,
            api_key=api_key,
            timeout=timeout,
            **lm_kwargs,
        )

    lm._yaw_recipe = recipe
    lm._yaw_context_window_tokens = context_tokens
    lm._yaw_supports_gbnf = recipe == "llamacpp" and role not in ("rlm", "reflection")
    return lm


def create_dspy_rlm_pair(
    orchestrator_model: str = DEFAULT_QA_CHAT,
    sub_model: str = DEFAULT_EXTRACTION_CHAT,
    *,
    base_url: str = BASE_URL,
    api_key: str = "lemonade",
    provider: str = "openai",
    reflection: bool = False,
    cache: bool = True,
    context_window_tokens: int | None = None,
    max_tokens: int | None = None,
) -> tuple[YawDSPyLM, YawDSPyLM]:
    """Create the `(orchestrator_lm, sub_lm)` DSPy pair on the configured chat models.

    - `orchestrator_lm` (`role="rlm"` or `"reflection"`): runs `dspy.RLM` Python REPL code
      generation without GBNF (`_yaw_supports_gbnf=False`).
    - `sub_lm` (`role="extraction"` or `"reflection_worker"`): runs inner `llm_query`,
      `synthesize_instruction`, and GBNF-constrained `llm_*` worker calls (`_yaw_supports_gbnf=True`).
    - When `reflection=True`, configures `orchestrator_lm` with `role="reflection"` (`temp=0.8`, `max_tokens=4096`)
      and `sub_lm` with `role="reflection_worker"`; sampling follows each model profile.
    """
    orchestrator_lm = create_dspy_lm(
        orchestrator_model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        role="reflection" if reflection else "rlm",
        cache=cache,
        context_window_tokens=context_window_tokens,
        max_tokens=max_tokens,
    )
    sub_lm = create_dspy_lm(
        sub_model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        role="reflection_worker" if reflection else "extraction",
        cache=cache,
        context_window_tokens=context_window_tokens,
        max_tokens=max_tokens,
    )
    return orchestrator_lm, sub_lm


def extract_completion_logprob_confidence(
    response_or_history_entry: Any,
    span_text: str | None = None,
) -> float | None:
    """Extract length-normalized token logprob probability confidence in (0, 1].

    If token logprobs are absent or span_text is not found, returns None (strict honest unknown).
    """
    if response_or_history_entry is None:
        return None

    # Unwrap DSPy history entry or wrapper structures
    curr = response_or_history_entry
    if isinstance(curr, Mapping):
        if "response" in curr:
            curr = curr["response"]
        elif "choices" not in curr and "logprobs" not in curr:
            # Check for possible nested response under outputs/raw
            curr = curr.get("raw") or curr.get("output") or curr
    elif hasattr(curr, "response"):
        curr = curr.response

    # Extract choice
    if isinstance(curr, Mapping):
        choices = curr.get("choices")
        choice = choices[0] if isinstance(choices, Sequence) and choices else curr
    elif hasattr(curr, "choices"):
        choices = getattr(curr, "choices", None)
        choice = choices[0] if isinstance(choices, Sequence) and choices else curr
    else:
        choice = curr

    # Extract logprobs structure from choice or parent
    logprobs_obj = None
    if isinstance(choice, Mapping):
        logprobs_obj = choice.get("logprobs")
    elif hasattr(choice, "logprobs"):
        logprobs_obj = getattr(choice, "logprobs", None)

    if logprobs_obj is None and hasattr(curr, "logprobs"):
        logprobs_obj = getattr(curr, "logprobs", None)

    if logprobs_obj is None:
        return None

    # Parse tokens and logprobs into pairs (token_str, logprob_val)
    token_pairs: list[tuple[str, float]] = []

    # Format 1: Chat Completion logprobs with "content" list
    content_list = None
    if isinstance(logprobs_obj, Mapping):
        content_list = logprobs_obj.get("content")
    elif hasattr(logprobs_obj, "content"):
        content_list = getattr(logprobs_obj, "content", None)

    if isinstance(content_list, Sequence):
        for item in content_list:
            t = None
            lp = None
            if isinstance(item, Mapping):
                t = item.get("token")
                lp = item.get("logprob")
            elif hasattr(item, "token") and hasattr(item, "logprob"):
                t = item.token
                lp = item.logprob
            if isinstance(t, str) and isinstance(lp, (int, float)):
                token_pairs.append((t, float(lp)))

    # Format 2: Completion logprobs with "tokens" and "token_logprobs"
    if not token_pairs:
        tokens = None
        tlps = None
        if isinstance(logprobs_obj, Mapping):
            tokens = logprobs_obj.get("tokens")
            tlps = logprobs_obj.get("token_logprobs")
        elif hasattr(logprobs_obj, "tokens") and hasattr(logprobs_obj, "token_logprobs"):
            tokens = getattr(logprobs_obj, "tokens", None)
            tlps = getattr(logprobs_obj, "token_logprobs", None)

        if isinstance(tokens, Sequence) and isinstance(tlps, Sequence) and len(tokens) == len(tlps):
            for t, lp in zip(tokens, tlps, strict=False):
                if isinstance(t, str) and isinstance(lp, (int, float)):
                    token_pairs.append((t, float(lp)))

    # Format 3: Direct sequence of token logprob items
    if not token_pairs and isinstance(logprobs_obj, Sequence):
        for item in logprobs_obj:
            t = None
            lp = None
            if isinstance(item, Mapping):
                t = item.get("token")
                lp = item.get("logprob")
            elif hasattr(item, "token") and hasattr(item, "logprob"):
                t = item.token
                lp = item.logprob
            if isinstance(t, str) and isinstance(lp, (int, float)):
                token_pairs.append((t, float(lp)))

    if not token_pairs:
        return None

    # Handle span matching if requested
    if span_text is not None:
        trimmed_span = span_text.strip()
        if not trimmed_span:
            return None
        full_text = "".join(t for t, _ in token_pairs)
        idx = full_text.find(trimmed_span)
        if idx == -1:
            idx = full_text.lower().find(trimmed_span.lower())
        if idx == -1:
            return None

        span_start = idx
        span_end = idx + len(trimmed_span)
        matched_lps: list[float] = []
        char_pos = 0
        for tok, lp in token_pairs:
            tok_start = char_pos
            tok_end = char_pos + len(tok)
            char_pos = tok_end
            if max(tok_start, span_start) < min(tok_end, span_end):
                matched_lps.append(lp)
        if not matched_lps:
            return None
        target_lps = matched_lps
    else:
        target_lps = [lp for _, lp in token_pairs]

    mean_lp = sum(target_lps) / len(target_lps)
    prob = math.exp(mean_lp)
    return float(min(1.0, max(1e-6, prob)))
