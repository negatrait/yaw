"""Declared model profiles and local inference defaults; remote weights are not attested."""

from dataclasses import dataclass

from yaw.core.config import DEFAULT_CONTEXT_WINDOW_TOKENS


@dataclass(frozen=True)
class ChatProfile:
    checkpoint: str
    recipe: str
    role: str = "both"  # "extraction", "qa", or "both"
    default_temperature: float = 0.35
    extraction_temperature: float = 0.35
    qa_temperature: float = 0.35
    reflection_temperature: float = 0.8
    reflection_worker_temperature: float = 0.7
    top_p: float | None = None
    top_k: int | None = None
    expected_device: str | None = None
    expected_context: int | None = None


CHAT_MODELS = {
    "Qwen3-4B-Instruct-2507-GGUF": ChatProfile(
        checkpoint="unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        recipe="llamacpp",
        role="both",
        default_temperature=0.35,
        extraction_temperature=0.35,
        qa_temperature=0.35,
        expected_device="gpu",
        expected_context=16384,
    ),
    "gemma-4-e4b": ChatProfile(
        checkpoint="unsloth/gemma-4-E4B-it-qat-q4_0-gguf-Q4_0:Q4_K_M",
        recipe="llamacpp",
        role="both",
        default_temperature=1.0,
        extraction_temperature=1.0,
        qa_temperature=1.0,
        reflection_temperature=1.0,
        reflection_worker_temperature=1.0,
        top_p=0.95,
        top_k=64,
        expected_device="gpu",
        expected_context=DEFAULT_CONTEXT_WINDOW_TOKENS,
    ),
}
DEFAULT_EXTRACTION_CHAT = "gemma-4-e4b"
DEFAULT_QA_CHAT = "gemma-4-e4b"
BASE_URL = "http://127.0.0.1:8888/v1"


@dataclass(frozen=True)
class ServingModel:
    """Active capacity and server-reported deployment identity for one model alias."""

    alias: str
    context_tokens: int
    checkpoint: str


def resolve_serving_models(base_url: str, api_key: str, aliases: set[str]) -> dict[str, ServingModel]:
    """Resolve configured aliases from authenticated native OpenAI model metadata."""
    from openai import OpenAI

    with OpenAI(base_url=base_url, api_key=api_key) as client:
        models = {model.id: model for model in client.models.list().data}
    resolved: dict[str, ServingModel] = {}
    for alias in aliases:
        model = models.get(alias)
        if model is None:
            raise ValueError(f"Serving model {alias!r} is missing from models.list")
        metadata = model.model_extra or {}
        context = metadata.get("context_length")
        if metadata.get("loaded") is not True:
            raise ValueError(f"Serving model {alias!r} is not loaded")
        if type(context) is not int or context < 2:
            raise ValueError(f"Serving model {alias!r} has invalid active context_length")
        checkpoint = metadata.get("alias_for")
        quant = metadata.get("quant")
        if not isinstance(checkpoint, str) or not checkpoint:
            if not isinstance(quant, str) or not quant:
                raise ValueError(f"Serving model {alias!r} lacks checkpoint identity")
            checkpoint = f"{alias}:{quant}"
        elif isinstance(quant, str) and quant and not checkpoint.endswith(f":{quant}"):
            checkpoint = f"{checkpoint}:{quant}"
        resolved[alias] = ServingModel(alias, context, checkpoint)
    return resolved
