"""Native DSPy LM and heterogeneous RLM model pair factories for local inference."""

from typing import Any

import dspy

from .grammars import lm_supports_gbnf
from .models import BASE_URL, CHAT_MODELS, DEFAULT_EXTRACTION_CHAT, DEFAULT_QA_CHAT

__all__ = [
    "create_dspy_lm",
    "create_dspy_rlm_pair",
    "lm_supports_gbnf",
]


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
