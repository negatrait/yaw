import logging
from typing import Any

import dspy

from .grammars import lm_supports_gbnf
from .models import BASE_URL, CHAT_MODELS, DEFAULT_EXTRACTION_CHAT, DEFAULT_QA_CHAT

for _logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

__all__ = [
    "YawDSPyLM",
    "create_dspy_lm",
    "create_dspy_rlm_pair",
    "lm_supports_gbnf",
]


class YawDSPyLM(dspy.LM):
    """Native dspy.LM wrapper providing recipe-aware fixes (e.g. Ryzen AI Jinja system folding)."""

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
        return super().__call__(prompt=prompt, messages=messages, **kwargs)

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
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
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    def copy(self, **kwargs: Any) -> Any:
        copied = super().copy(**kwargs)
        setattr(copied, "_yaw_recipe", getattr(self, "_yaw_recipe", None))
        setattr(copied, "_yaw_supports_gbnf", getattr(self, "_yaw_supports_gbnf", False))
        return copied


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
    setattr(lm, "_yaw_supports_gbnf", (recipe == "llamacpp") and (role not in ("rlm", "reflection")))
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
