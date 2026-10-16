"""Explicit model selection and read-only Lemonade identity checks."""

import hashlib
import json
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ChatProfile:
    checkpoint: str
    recipe: str
    role: str = "both"  # "extraction", "qa", or "both"
    default_temperature: float = 0.2
    extraction_temperature: float = 0.2
    qa_temperature: float = 0.2
    frequency_penalty: float = 0.0
    expected_device: str | None = None
    expected_context: int | None = None

    def __iter__(self):
        yield self.checkpoint
        yield self.recipe

    def __getitem__(self, index):
        if index == 0:
            return self.checkpoint
        if index == 1:
            return self.recipe
        raise IndexError("ChatProfile tuple index out of range (expected 0 or 1)")


CHAT_MODELS = {
    "Phi-4-mini-instruct-GGUF": ChatProfile(
        checkpoint="unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf",
        recipe="llamacpp",
        role="extraction",
        default_temperature=0.2,
        extraction_temperature=0.2,
        qa_temperature=0.2,
        frequency_penalty=0.5,
        expected_device="gpu",
        expected_context=16384,
    ),
    "Llama-3.2-3B-Instruct-GGUF": ChatProfile(
        checkpoint="unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf",
        recipe="llamacpp",
        role="extraction",
        default_temperature=0.2,
        extraction_temperature=0.2,
        qa_temperature=0.2,
        frequency_penalty=0.5,
        expected_device="gpu",
        expected_context=16384,
    ),
    "Qwen3-4B-Instruct-2507-GGUF": ChatProfile(
        checkpoint="unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        recipe="llamacpp",
        role="extraction",
        default_temperature=0.2,
        extraction_temperature=0.2,
        qa_temperature=0.2,
        frequency_penalty=0.5,
        expected_device="gpu",
        expected_context=16384,
    ),
    "Mistral-7B-Instruct-v0.3-Hybrid": ChatProfile(
        checkpoint="amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-1.7-hybrid",
        recipe="ryzenai-llm",
        role="qa",
        default_temperature=0.2,
        extraction_temperature=0.2,
        qa_temperature=0.2,
        frequency_penalty=0.0,
        expected_device="npu",
        expected_context=32786,
    ),
    "gemma4-it-12b-FLM": ChatProfile(
        checkpoint="gemma4-it:12b",
        recipe="flm",
        role="both",
        default_temperature=0.0,
        extraction_temperature=0.0,
        qa_temperature=0.0,
        frequency_penalty=0.0,
    ),
    "qwen3.5-9b-FLM": ChatProfile(
        checkpoint="qwen3.5:9b",
        recipe="flm",
        role="both",
        default_temperature=0.0,
        extraction_temperature=0.0,
        qa_temperature=0.0,
        frequency_penalty=0.0,
    ),
}
DEFAULT_EXTRACTION_CHAT = "Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_QA_CHAT = "Mistral-7B-Instruct-v0.3-Hybrid"
DEFAULT_CHAT = DEFAULT_QA_CHAT
BASE_URL = "http://localhost:13305/api/v1"
DEFAULT_BASE_URL = BASE_URL


def get_json(base_url, route):
    with urlopen(base_url.rstrip("/") + route, timeout=30) as response:
        return json.load(response)


def canonical(name):
    return name.removeprefix("user.").removeprefix("builtin.")


def select_model(models, name):
    matches = [model for model in models if canonical(
        model.get("id", model.get("model_name", ""))
    ) == canonical(name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one registered model for {name}.")
    return matches[0]


def check_registry(models, profile, chat_model=None, qa_model=None):
    selections = [(profile.model, profile.checkpoint, "llamacpp", "embeddings", "embeddings")]
    if chat_model is not None:
        checkpoint, recipe = CHAT_MODELS[chat_model]
        selections.append((chat_model, checkpoint, recipe, "chat", "chat"))
    if qa_model is not None:
        checkpoint, recipe = CHAT_MODELS[qa_model]
        selections.append((qa_model, checkpoint, recipe, "chat", "qa_chat"))
    result = {}
    for name, checkpoint, recipe, label, key in selections:
        model = select_model(models, name)
        if (model.get("checkpoint") != checkpoint or model.get("recipe") != recipe
                or label not in model.get("labels", [])):
            raise ValueError(f"{name}: checkpoint/recipe/role differs from its declared profile.")
        if not model.get("downloaded"):
            raise ValueError(f"{name} is not downloaded. Install it explicitly in Lemonade first.")
        result[key] = model
    return result


def preflight(profile, chat_model=None, qa_model=None, base_url=BASE_URL):
    return check_registry(
        get_json(base_url, "/models?show_all=true")["data"], profile, chat_model, qa_model,
    )


def embedding_evidence(profile, base_url=BASE_URL):
    health = get_json(base_url, "/health")
    loaded_models = health.get("all_models_loaded", [])
    if not any(
        canonical(m.get("id", m.get("model_name", ""))) == canonical(profile.model)
        for m in loaded_models
    ):
        # Warm / auto-load the configured embedding model into residency if Lemonade evicted it.
        probe_req = Request(
            base_url.rstrip("/") + "/embeddings",
            data=json.dumps({"model": profile.model, "input": "probe"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(probe_req, timeout=30):
                pass
        except Exception:
            pass
        health = get_json(base_url, "/health")
        loaded_models = health.get("all_models_loaded", [])

    model = select_model(loaded_models, profile.model)
    if (model.get("checkpoint") != profile.checkpoint or model.get("recipe") != "llamacpp"
            or model.get("backend_health") != "ready"):
        raise ValueError("Loaded embedding backend differs from the selected profile.")
    command = model.get("launch_command", [])
    if "--pooling" not in command or command[command.index("--pooling") + 1] != profile.pooling:
        raise ValueError(f"{profile.model} requires explicit --pooling {profile.pooling} in Lemonade.")
    # Resolve the actual local weights, not just the mutable registry alias.
    if "-m" not in command:
        raise ValueError("Cannot identify the loaded embedding weights.")
    weights_path = command[command.index("-m") + 1]
    identity = hashlib.sha256(
        (profile.index_identity + "\n" + weights_path).encode("utf-8")
    ).hexdigest()
    return {"version": health.get("version"), "embedding_backend": model,
            "resolved_index_identity": "yaw-loaded-embedding-v2:" + identity}


def chat_evidence(model_name, base_url=BASE_URL):
    profile = CHAT_MODELS[model_name]
    checkpoint, recipe = profile.checkpoint, profile.recipe
    health = get_json(base_url, "/health")
    model = select_model(health.get("all_models_loaded", []), model_name)
    if (model.get("checkpoint") != checkpoint or model.get("recipe") != recipe
            or model.get("backend_health") != "ready"):
        raise ValueError(f"Loaded chat backend for {model_name} differs from its declared profile.")
    if profile.expected_device is not None and model.get("device") != profile.expected_device:
        raise ValueError(f"{model_name} expected device {profile.expected_device}, got {model.get('device')}.")
    if recipe == "llamacpp":
        command = model.get("launch_command", [])
        if "-dio" in command or "--direct-io" in command:
            raise ValueError(f"{model_name} must not use -dio/--direct-io on Windows CUDA.")
        if "--cache-ram" not in command or command[command.index("--cache-ram") + 1] != "0":
            raise ValueError(
                f"{model_name} requires explicit --cache-ram 0 in Lemonade to prevent host RAM prompt-cache growth."
            )
    elif recipe == "ryzenai-llm":
        if model.get("device") != "npu":
            raise ValueError(f"{model_name} requires device 'npu' in Lemonade.")
        ctx_size = model.get("recipe_options", {}).get("ctx_size")
        if profile.expected_context and ctx_size and ctx_size < profile.expected_context:
            raise ValueError(f"{model_name} requires ctx_size >= {profile.expected_context}, got {ctx_size}.")
    return model
