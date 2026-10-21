import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import dspy

from .grammars import lm_supports_gbnf
from .models import BASE_URL, CHAT_MODELS, DEFAULT_EXTRACTION_CHAT, DEFAULT_QA_CHAT

for _logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
if hasattr(dspy, "disable_litellm_logging"):
    dspy.disable_litellm_logging()

_YAW_SLEEP_SECONDS: ContextVar[float] = ContextVar("yaw_sleep_seconds", default=0.0)


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


class YawDSPyLM(dspy.LM):
    """Native dspy.LM wrapper preserving recipe metadata across copies without overriding BaseLM.forward."""

    def __call__(self, prompt: Any = None, *, messages: Any = None, **kwargs: Any) -> Any:
        if (
            getattr(self, "_yaw_recipe", None) == "ryzenai-llm"
            and isinstance(messages, list)
            and len(messages) >= 2
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
            and isinstance(messages[1], dict)
            and messages[1].get("role") == "user"
        ):
            messages = [
                {
                    "role": "user",
                    "content": str(messages[0].get("content", "")).strip()
                    + "\n\n"
                    + str(messages[1].get("content", "")),
                },
                *messages[2:],
            ]
        res = super().__call__(prompt=prompt, messages=messages, **kwargs)
        sleep_sec = get_yaw_sleep_seconds()
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return res

    def copy(self, **kwargs: Any) -> Any:
        copied = super().copy(**kwargs)
        setattr(copied, "_yaw_recipe", getattr(self, "_yaw_recipe", None))
        setattr(copied, "_yaw_supports_gbnf", getattr(self, "_yaw_supports_gbnf", False))
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
) -> None:
    """Configure workspace-isolated DSPy cache via dspy.configure_cache with restrict_pickle=True by default."""
    resolved_dir: str | None = None
    if cache_dir is not None:
        p = Path(cache_dir)
        if enable_disk_cache:
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
    num_retries: int = 1,
    timeout: float = 240.0,
    **kwargs: Any,
) -> dspy.LM:
    """Create a native dspy.LM configured for Lemonade (openai/...) or Ollama (ollama_chat/...).

    Applies recipe-aware and role-aware defaults for our local hardware stack:
    - Qwen3-4B-Instruct-2507-GGUF (llamacpp on Vulkan GPU, 16384 ctx): supports GBNF for direct/worker
      roles (`extraction`, `qa`, `reflection_worker`) with temp=0.2, freq_penalty=0.5, and runs
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
            temperature = 0.0

    if max_tokens is None:
        # Per-completion output token cap (total context window is 16384 on Qwen in Lemonade).
        max_tokens = 4096

    lm_kwargs: dict[str, Any] = dict(kwargs)
    if role == "extraction" and profile is not None and profile.frequency_penalty > 0:
        lm_kwargs.setdefault("frequency_penalty", profile.frequency_penalty)
    elif role == "reflection_worker":
        freq = profile.reflection_frequency_penalty if profile is not None else 0.3
        if freq > 0:
            lm_kwargs.setdefault("frequency_penalty", freq)
    if recipe == "flm":
        lm_kwargs.setdefault("extra_body", {"think": False})

    if provider == "ollama" or model_name.startswith(("ollama_chat/", "ollama/")):
        litellm_model = (
            model_name
            if model_name.startswith(("ollama_chat/", "ollama/"))
            else f"ollama_chat/{clean_name}"
        )
        lm = YawDSPyLM(
            model=litellm_model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
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
            num_retries=num_retries,
            api_base=base_url,
            api_key=api_key,
            timeout=timeout,
            **lm_kwargs,
        )

    setattr(lm, "_yaw_recipe", recipe)
    setattr(
        lm, "_yaw_supports_gbnf", (recipe == "llamacpp") and (role not in ("rlm", "reflection"))
    )
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
    num_retries: int = 0,
) -> tuple[dspy.LM, dspy.LM]:
    """Create the `(orchestrator_lm, sub_lm)` DSPy LM pair on `Qwen3-4B-Instruct-2507-GGUF`.

    - `orchestrator_lm` (`role="rlm"` or `"reflection"`): runs `dspy.RLM` Python REPL code
      generation without GBNF (`_yaw_supports_gbnf=False`).
    - `sub_lm` (`role="extraction"` or `"reflection_worker"`): runs inner `llm_query`,
      `synthesize_instruction`, and GBNF-constrained `qwen_*` worker calls (`_yaw_supports_gbnf=True`).
    - When `reflection=True`, configures `orchestrator_lm` with `role="reflection"` (`temp=0.8`, `max_tokens=4096`)
      and `sub_lm` with `role="reflection_worker"` (`temp=0.7`, `max_tokens=4096`, `freq_penalty=0.3`).
    """
    orchestrator_lm = create_dspy_lm(
        orchestrator_model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        role="reflection" if reflection else "rlm",
        cache=cache,
        num_retries=num_retries,
    )
    sub_lm = create_dspy_lm(
        sub_model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        role="reflection_worker" if reflection else "extraction",
        cache=cache,
        num_retries=num_retries,
    )
    return orchestrator_lm, sub_lm
