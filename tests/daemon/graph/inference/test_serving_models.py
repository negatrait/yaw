"""Active model capability resolution over authenticated HTTP."""

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from yaw.daemon.graph.inference.models import resolve_serving_models


@pytest.fixture
def models_server():
    entries = [
        {"id": "chat", "loaded": True, "context_length": 131072, "alias_for": "vendor/chat:Q4"},
        {
            "id": "embed",
            "loaded": True,
            "context_length": 8192,
            "max_context_length": 38656,
            "native_context_length": 262144,
            "alias_for": "vendor/embed:Q6",
        },
    ]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            payload = {
                "object": "list",
                "data": [dict(item, object="model", created=0, owned_by="fixture") for item in entries],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("X-Yaw-Embedding-Deployment")))
            if self.path == "/v1/chat/completions":
                text = str(entries[0]["alias_for"])
                payload = {
                    "id": "chat",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
                return
            if self.path == "/v1/completions":
                text = str(entries[0]["alias_for"])
                payload = {
                    "id": "chat",
                    "object": "text_completion",
                    "model": body["model"],
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
                return
            values = [3.0, 4.0] if str(entries[1]["alias_for"]).endswith("Q6") else [4.0, 3.0]
            payload = {
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": i, "embedding": values} for i in range(len(body["input"]))],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield entries, calls, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_active_limits_and_checkpoint_identity(models_server):
    entries, calls, base = models_server
    models = resolve_serving_models(base, "fixture-key", {"chat", "embed"})
    assert calls == [("/v1/models", "Bearer fixture-key")]
    assert models["chat"].context_tokens == 131072
    assert models["embed"].context_tokens == 8192
    assert models["embed"].checkpoint == "vendor/embed:Q6"
    entries[1]["alias_for"] = "vendor/embed:Q8"
    assert resolve_serving_models(base, "fixture-key", {"embed"})["embed"].checkpoint == "vendor/embed:Q8"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda entries: entries.clear(), "missing"),
        (lambda entries: entries[0].update(loaded=False), "not loaded"),
        (lambda entries: entries[0].pop("context_length"), "context_length"),
        (lambda entries: entries[0].pop("alias_for"), "checkpoint identity"),
    ],
)
def test_rejects_missing_unloaded_or_incomplete_metadata(models_server, mutation, match):
    entries, _, base = models_server
    mutation(entries)
    with pytest.raises(ValueError, match=match):
        resolve_serving_models(base, "fixture-key", {"chat"})


def test_runtime_binds_active_context_and_embedding_identity(models_server, tmp_path):
    from yaw.core import YawConfig
    from yaw.daemon.runtime import YawRuntime

    entries, calls, base = models_server
    entries[0]["id"] = "gemma-4-e4b"
    entries[1]["id"] = "embeddinggemma-2"
    config = YawConfig(workspace_root=tmp_path, base_url=base, api_key="fixture-key", max_output_tokens=128)
    runtime = YawRuntime(config)
    program, embedder, engine = runtime._create_bound_engine(config)
    try:
        assert runtime.config.context_window_tokens == 131072
        assert program.context_window_tokens == 131072
        assert program.orchestrator_lm._yaw_context_window_tokens == 131072
        assert embedder.profile.context_tokens == 8192
        assert embedder.profile.checkpoint == "vendor/embed:Q6"
        embedder.profile = replace(embedder.profile, dimensions=2)
        first = embedder.batch_encode("same input")
        embedder.batch_encode("same input")
        assert sum(path == "/v1/embeddings" for path, _ in calls) == 1
        previous = embedder.index_identity
    finally:
        engine.close()
    entries[1]["alias_for"] = "vendor/embed:Q8"
    _, updated, engine = runtime._create_bound_engine(runtime.config)
    try:
        updated.profile = replace(updated.profile, dimensions=2)
        assert updated.index_identity != previous
        second = updated.batch_encode("same input")
        assert sum(path == "/v1/embeddings" for path, _ in calls) == 2
        assert not np.array_equal(first, second)
    finally:
        engine.close()


def test_runtime_rejects_output_budget_exceeding_active_capacity(models_server, tmp_path):
    from yaw.core import YawConfig, YawRuntimeError
    from yaw.daemon.runtime import YawRuntime

    entries, _, base = models_server
    entries[0].update(id="gemma-4-e4b", context_length=1024)
    entries[1]["id"] = "embeddinggemma-2"
    config = YawConfig(workspace_root=tmp_path, base_url=base, api_key="fixture-key", max_output_tokens=2048)
    with pytest.raises(YawRuntimeError, match="max_output_tokens"):
        YawRuntime(config)._create_bound_engine(config)


def test_runtime_scales_automatic_budgets_from_active_context_and_preserves_intent(models_server, tmp_path):
    from yaw.core import YawConfig
    from yaw.daemon.runtime import YawRuntime

    entries, _, base = models_server
    entries[0].update(id="gemma-4-e4b", context_length=8192)
    entries[1]["id"] = "embeddinggemma-2"
    config = YawConfig(workspace_root=tmp_path, base_url=base, api_key="fixture-key")
    assert config.max_output_tokens is None and config.rlm_max_output_chars is None
    runtime = YawRuntime(config)
    program, _, engine = runtime._create_bound_engine(config)
    try:
        assert runtime.config.context_window_tokens == 8192
        assert runtime.config.max_output_tokens is None and runtime.config.rlm_max_output_chars is None
        assert runtime.config.for_optimization().max_output_tokens is None
        assert runtime.config.for_benchmark("fixture", "1234").rlm_max_output_chars is None
        assert program.max_output_tokens is None
        assert program.shacl_healer_rlm.max_output_chars == 1250
        assert engine.config.max_output_tokens is None
        assert program.rlm_sub_lm.kwargs["max_tokens"] is None
    finally:
        engine.close()


def test_runtime_chat_deployment_invalidates_native_and_graph_caches(models_server, tmp_path):
    import dspy

    from yaw.core import YawConfig
    from yaw.daemon.graph.inference.diagnostics import label_judge_fingerprint
    from yaw.daemon.graph.inference.optimizer import openie_fingerprint
    from yaw.daemon.runtime import YawRuntime

    entries, calls, base = models_server
    entries[0]["id"] = "gemma-4-e4b"
    entries[1]["id"] = "embeddinggemma-2"
    config = YawConfig(workspace_root=tmp_path, base_url=base, api_key="fixture-key")
    runtime = YawRuntime(config)
    program, _, engine = runtime._create_bound_engine(config)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    try:
        first_openie = openie_fingerprint(program)
        first_judge = label_judge_fingerprint(program)
        assert program.rlm_sub_lm("same prompt") == ["vendor/chat:Q4"]
        assert program.rlm_sub_lm("same prompt") == ["vendor/chat:Q4"]
        assert sum(path == "/v1/completions" for path, _ in calls) == 1
    finally:
        engine.close()
    entries[0]["alias_for"] = "vendor/chat:Q8"
    updated, _, engine = runtime._create_bound_engine(runtime.config)
    try:
        assert openie_fingerprint(updated) != first_openie
        assert label_judge_fingerprint(updated) != first_judge
        assert updated.rlm_sub_lm("same prompt") == ["vendor/chat:Q8"]
        assert sum(path == "/v1/completions" for path, _ in calls) == 2
    finally:
        engine.close()


def test_native_chat_route_cache_tracks_deployment(models_server, tmp_path):
    import dspy

    from yaw.core import YawConfig
    from yaw.daemon.runtime import YawRuntime

    entries, calls, base = models_server
    entries[1]["id"] = "embeddinggemma-2"
    config = YawConfig(
        workspace_root=tmp_path, base_url=base, api_key="fixture-key", extraction_model="chat", qa_model="chat"
    )
    runtime = YawRuntime(config)
    program, _, engine = runtime._create_bound_engine(config)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    try:
        assert program.orchestrator_lm("same prompt") == ["vendor/chat:Q4"]
        assert program.orchestrator_lm("same prompt") == ["vendor/chat:Q4"]
        assert sum(path == "/v1/chat/completions" for path, _ in calls) == 1
    finally:
        engine.close()
    entries[0]["alias_for"] = "vendor/chat:Q8"
    updated, _, engine = runtime._create_bound_engine(runtime.config)
    try:
        assert updated.orchestrator_lm("same prompt") == ["vendor/chat:Q8"]
        assert sum(path == "/v1/chat/completions" for path, _ in calls) == 2
    finally:
        engine.close()
