"""Read-only registry and serving identity checks over real loopback HTTP."""

import asyncio
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TypedDict

import numpy as np
import pytest

from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter, create_dspy_embedder, resolve_embedding_profile
from yaw.daemon.graph.inference.models import (
    CHAT_MODELS,
    ChatProfile,
    chat_evidence,
    embedding_evidence,
    preflight,
    select_model,
)


class RegistryState(TypedDict):
    loaded: list[dict[str, Any]]
    registry: list[dict[str, Any]]
    after_probe: list[dict[str, Any]]
    probe_status: int
    requests: list[tuple[str, Any]]


@pytest.fixture
def registry_server():
    state: RegistryState = {"loaded": [], "registry": [], "after_probe": [], "probe_status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(("GET", self.path))
            payload = (
                {"data": state["registry"]}
                if self.path.startswith("/models")
                else {"version": "test", "all_models_loaded": state["loaded"]}
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(("POST", body))
            state["loaded"] = state["after_probe"]
            self.send_response(state["probe_status"])
            self.end_headers()
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "model": body["model"],
                        "data": [
                            {"object": "embedding", "index": i, "embedding": [3.0, 4.0]} for i in range(len(inputs))
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def embedding_entry(profile):
    return {
        "id": "user." + profile.model,
        "checkpoint": profile.checkpoint,
        "recipe": "llamacpp",
        "backend_health": "ready",
        "launch_command": ["server", "--pooling", profile.pooling, "-m", "weights.gguf"],
        "labels": ["embeddings"],
        "downloaded": True,
    }


def test_chat_profile_tuple_contract_and_ambiguous_aliases():
    profile = ChatProfile("weights", "flm")
    assert tuple(profile) == ("weights", "flm")
    assert profile[0] == "weights"
    assert profile[1] == "flm"
    for index in (-1, 2):
        with pytest.raises(IndexError, match="tuple index"):
            profile[index]
    for entries in ([], [{"id": "user.model"}, {"model_name": "builtin.model"}]):
        with pytest.raises(ValueError, match="exactly one"):
            select_model(entries, "model")


def test_registry_preflight_fetches_exact_registered_embedding(registry_server):
    state, base_url = registry_server
    profile = resolve_embedding_profile()
    state["registry"] = [embedding_entry(profile)]
    result = preflight(profile, base_url=base_url + "/")
    assert result["embeddings"]["checkpoint"] == profile.checkpoint
    assert state["requests"] == [("GET", "/models?show_all=true")]


@pytest.mark.parametrize("probe_status", [200, 503])
def test_embedding_residency_probe_rechecks_health_and_weight_identity(registry_server, probe_status):
    state, base_url = registry_server
    profile = resolve_embedding_profile()
    entry = embedding_entry(profile)
    state["after_probe"] = [entry]
    state["probe_status"] = probe_status
    result = embedding_evidence(profile, base_url)
    assert result["embedding_backend"] == entry
    assert result["version"] == "test"
    identity = result["resolved_index_identity"]
    assert identity.startswith("yaw-loaded-embedding-v2:")
    assert state["requests"] == [
        ("GET", "/health"),
        ("POST", {"model": profile.model, "input": "probe"}),
        ("GET", "/health"),
    ]
    entry["launch_command"][-1] = "other-quant.gguf"
    assert embedding_evidence(profile, base_url)["resolved_index_identity"] != identity
    assert len(state["requests"]) == 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint": "wrong"}, "differs"),
        ({"recipe": "wrong"}, "differs"),
        ({"backend_health": "loading"}, "differs"),
        ({"launch_command": ["server", "-m", "weights"]}, "explicit --pooling"),
        ({"launch_command": ["server", "--pooling", "last", "-m", "weights"]}, "explicit --pooling"),
        ({"launch_command": ["server", "--pooling", "mean"]}, "identify.*weights"),
    ],
)
def test_loaded_embedding_contract_rejects_incompatible_backend(registry_server, change, message):
    state, base_url = registry_server
    profile = resolve_embedding_profile()
    state["loaded"] = [{**embedding_entry(profile), **change}]
    with pytest.raises(ValueError, match=message):
        embedding_evidence(profile, base_url)
    assert state["requests"] == [("GET", "/health")]


def test_chat_identity_uses_http_health_and_rejects_unready_backend(registry_server):
    state, base_url = registry_server
    name = "gemma4-it-12b-FLM"
    profile = CHAT_MODELS[name]
    entry = {"id": name, "checkpoint": profile.checkpoint, "recipe": profile.recipe, "backend_health": "ready"}
    state["loaded"] = [entry]
    assert chat_evidence(name, base_url) == entry
    entry["backend_health"] = "loading"
    with pytest.raises(ValueError, match="differs"):
        chat_evidence(name, base_url)


@pytest.mark.parametrize(
    ("device", "message"),
    [("npu", None), ("gpu", "expected device npu")],
)
def test_registered_npu_profile_checks_expected_device(registry_server, monkeypatch, device, message):
    state, base_url = registry_server
    name = "test-flm-npu-profile"
    profile = ChatProfile("test-weights", "flm", expected_device="npu")
    monkeypatch.setitem(CHAT_MODELS, name, profile)
    entry = {
        "id": name,
        "checkpoint": profile.checkpoint,
        "recipe": profile.recipe,
        "backend_health": "ready",
        "device": device,
    }
    state["loaded"] = [entry]
    if message:
        with pytest.raises(ValueError, match=message):
            chat_evidence(name, base_url)
    else:
        assert chat_evidence(name, base_url) == entry


def test_native_async_embedder_uses_real_http_provider(registry_server):
    state, base_url = registry_server
    profile = replace(resolve_embedding_profile(), dimensions=2)
    embedder = create_dspy_embedder(profile, base_url=base_url + "/v1", caching=False)
    adapter = DSPyEmbeddingAdapter(profile, embedder)
    vectors = asyncio.run(adapter.acall(["first", "second"]))
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.6, 0.8]])
    assert state["requests"][-1][0] == "POST"
    assert state["requests"][-1][1]["input"] == [profile.document_prefix + "first", profile.document_prefix + "second"]
