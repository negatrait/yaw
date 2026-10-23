"""Configured model routing and context safeguards over real loopback requests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import pytest

from yaw.core.config import YawConfig, context_char_limit, load_config
from yaw.core.types import YawRuntimeError
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, HippoRAGProgram, create_dspy_lm
from yaw.daemon.graph.inference.embedding import MODEL, resolve_embedding_profile


@pytest.fixture
def provider():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if self.path.endswith("/embeddings"):
                response = {
                    "model": request["model"],
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * 767}
                        for i, _ in enumerate(request["input"])
                    ],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                }
            else:
                response = {
                    "id": "budget",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {"named_entities": [["Aster", "A project."]]}
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                }
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_default_and_explicit_embedding_routes_are_independent(tmp_path, provider):
    url, requests = provider
    default = YawConfig(workspace_root=tmp_path, base_url=url)
    assert default.embedding_model == MODEL == "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0"
    DSPyEmbeddingAdapter.from_config(default).batch_encode(["Aster"])
    assert requests[-1]["model"] == MODEL
    explicit = YawConfig(
        workspace_root=tmp_path, base_url=url, embedding_model="embeddinggemma-300m-GGUF"
    )
    DSPyEmbeddingAdapter.from_config(explicit).batch_encode(["Aster"])
    assert requests[-1]["model"] == explicit.embedding_model
    assert (
        resolve_embedding_profile(MODEL).index_identity
        != resolve_embedding_profile(explicit.embedding_model).index_identity
    )
    with pytest.raises(KeyError):
        resolve_embedding_profile("embeddiggemma-300m-qat-q8_0-GGUF-Q8_0")


def test_serving_context_changes_real_extraction_truncation(provider):
    url, requests = provider
    fingerprints = []
    passage = "Aster is a project. " + "x" * 30000 + "TAIL_RETAINED_AT_64K"
    for context_tokens in (32768, 65536):
        lm = create_dspy_lm(
            "budget-test", base_url=url, cache=False, context_window_tokens=context_tokens
        )
        task = HippoRAGProgram(
            enable_gbnf=False,
            include_demos=False,
            orchestrator_lm=lm,
            rlm_sub_lm=lm,
            context_window_tokens=context_tokens,
        )
        with dspy.context(lm=lm, disable_history=True):
            assert task.extract_entities(passage) == ("Aster",)
        prompt = str(requests[-1]["messages"])
        assert ("TAIL_RETAINED_AT_64K" in prompt) == (context_tokens == 65536)
        assert "context_window_tokens" not in requests[-1]  # local budget, not an API parameter
        fingerprints.append(YawGraphEngine._openie_fingerprint(task))
    assert fingerprints[0] != fingerprints[1]
    assert context_char_limit(24000, 32768, 4096) == 24000
    assert context_char_limit(24000, 65536, 8192) == 48000


def test_configured_limits_override_old_compiled_deployment_settings(tmp_path):
    old = HippoRAGProgram(enable_gbnf=False)
    old.pareto_frontier_states = [old.dump_state()]
    saved = tmp_path / "compiled.json"
    old.save(str(saved))
    config = YawConfig(
        workspace_root=tmp_path,
        compiled_program_path=saved,
        context_window_tokens=65536,
        max_output_tokens=8192,
        rlm_max_iters=14,
        rlm_max_llm_calls=25,
        rlm_max_output_chars=18000,
    )
    program = HippoRAGProgram.from_config(config)
    assert program.context_window_tokens == 65536
    assert program.max_output_tokens == 8192
    assert program.orchestrator_lm.kwargs["max_tokens"] == 8192
    assert program.orchestrator_lm.copy()._yaw_context_window_tokens == 65536
    assert program.shacl_healer_rlm.max_iters == 14
    assert program.shacl_healer_rlm.max_llm_calls == 25
    assert program.shacl_healer_rlm.max_output_chars == 18000
    assert config.max_context_chars == 48000
    assert program.pareto_ensemble.programs[0].context_window_tokens == 65536
    assert program.pareto_frontier_states[0]["yaw_surface"]["context_window_tokens"] == 65536
    assert config.for_optimization().context_window_tokens == 65536
    assert config.config_fingerprint() != YawConfig(workspace_root=tmp_path).config_fingerprint()


def test_repo_config_and_invalid_budgets(tmp_path):
    root = Path(__file__).resolve().parents[4]
    config = load_config(root / "yaw.toml")
    assert config.embedding_model == MODEL
    assert config.context_window_tokens == 65536
    with pytest.raises(YawRuntimeError):
        YawConfig(workspace_root=tmp_path, context_window_tokens=4096)
    with pytest.raises(YawRuntimeError):
        YawConfig(workspace_root=tmp_path, rlm_max_iters=0)
