"""Explicit model selection and read-only Lemonade identity checks."""

import hashlib
import json
from urllib.request import urlopen

CHAT_MODELS = {
    "Phi-4-mini-instruct-GGUF": (
        "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf", "llamacpp",
    ),
    "Llama-3.2-3B-Instruct-GGUF": (
        "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf", "llamacpp",
    ),
    "Qwen3-4B-Instruct-2507-GGUF": (
        "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "llamacpp",
    ),
    "Mistral-7B-Instruct-v0.3-Hybrid": (
        "amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-1.7-hybrid", "ryzenai-llm",
    ),
    "gemma4-it-12b-FLM": ("gemma4-it:12b", "flm"),
    "qwen3.5-9b-FLM": ("qwen3.5:9b", "flm"),
}
DEFAULT_EXTRACTION_CHAT = "Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_QA_CHAT = "Mistral-7B-Instruct-v0.3-Hybrid"
DEFAULT_CHAT = DEFAULT_QA_CHAT
BASE_URL = "http://localhost:13305/api/v1"


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
    model = select_model(health.get("all_models_loaded", []), profile.model)
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
    checkpoint, recipe = CHAT_MODELS[model_name]
    health = get_json(base_url, "/health")
    model = select_model(health.get("all_models_loaded", []), model_name)
    if (model.get("checkpoint") != checkpoint or model.get("recipe") != recipe
            or model.get("backend_health") != "ready"):
        raise ValueError(f"Loaded chat backend for {model_name} differs from its declared profile.")
    return model
