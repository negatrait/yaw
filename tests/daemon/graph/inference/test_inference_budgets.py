"""Configured model routing and context safeguards over real loopback requests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import pytest

from yaw.core.config import YawConfig, context_char_limit, load_config, scale_context_budget
from yaw.core.types import YawRuntimeError
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    HippoRAGProgram,
    RLMInstructionProposer,
    create_dspy_lm,
    openie_fingerprint,
)
from yaw.daemon.graph.inference.embedding import MODEL, resolve_embedding_profile
from yaw.daemon.graph.inference.models import CHAT_MODELS


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
                                    {"named_entities": [["Aster", "A project."]], "repairs": [], "decisions": []}
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

        def log_message(self, format: str, *args: object) -> None:
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
    explicit = YawConfig(workspace_root=tmp_path, base_url=url, embedding_model="embeddinggemma-300m-GGUF")
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
        lm = create_dspy_lm("budget-test", base_url=url, cache=False, context_window_tokens=context_tokens)
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
        fingerprints.append(openie_fingerprint(task))
    assert fingerprints[0] != fingerprints[1]
    assert context_char_limit(24000, 32768, 4096) == 24000
    assert context_char_limit(24000, 65536, 8192) == 48000


@pytest.mark.parametrize("context_tokens", [65536, 131072])
def test_configured_limits_override_old_compiled_deployment_settings(tmp_path, context_tokens):
    old = HippoRAGProgram(enable_gbnf=False, context_window_tokens=32768, max_output_tokens=4096)
    saved = tmp_path / "compiled.json"
    old.save(str(saved))
    config = YawConfig(
        workspace_root=tmp_path,
        compiled_program_path=saved,
        context_window_tokens=context_tokens,
        max_output_tokens=8192,
        rlm_max_iters=14,
        rlm_max_llm_calls=25,
        rlm_max_output_chars=18000,
    )
    program = HippoRAGProgram.from_config(config)
    assert program.context_window_tokens == context_tokens
    assert program.max_output_tokens == 8192
    assert program.orchestrator_lm.kwargs["max_tokens"] == 8192
    assert program.orchestrator_lm.copy()._yaw_context_window_tokens == context_tokens
    assert program.shacl_healer_rlm.max_iters == 14
    assert program.shacl_healer_rlm.max_llm_calls == 25
    assert program.shacl_healer_rlm.max_output_chars == 18000
    assert config.max_context_chars == context_char_limit(24000, context_tokens, 8192)
    assert config.for_optimization().context_window_tokens == context_tokens
    assert config.config_fingerprint() != YawConfig(workspace_root=tmp_path).config_fingerprint()


def test_repo_config_and_invalid_budgets(tmp_path):
    root = Path(__file__).resolve().parents[4]
    config = load_config(root / "yaw.toml")
    assert config.embedding_model == MODEL
    assert config.context_window_tokens == 131072
    assert (config.max_output_tokens, config.reflection_input_tokens, config.rlm_max_output_chars) == (
        8192,
        32768,
        20000,
    )
    with pytest.raises(YawRuntimeError):
        YawConfig(workspace_root=tmp_path, context_window_tokens=4096, max_output_tokens=4096)
    with pytest.raises(YawRuntimeError):
        YawConfig(workspace_root=tmp_path, rlm_max_iters=0)


@pytest.mark.parametrize("context", [32768, 65536, 131072, 262144])
def test_automatic_budgets_follow_context_and_explicit_overrides_remain_fixed(tmp_path, context):
    config = YawConfig(workspace_root=tmp_path, context_window_tokens=context)
    assert config.max_output_tokens == context // 16
    assert config.reflection_input_tokens == context // 4
    assert config.rlm_max_output_chars == scale_context_budget(10000, context)
    assert config.max_context_chars == context_char_limit(24000, context, context // 16)
    assert config.rlm_max_iters == 10 and config.rlm_max_llm_calls == 20
    assert config.for_optimization().reflection_input_tokens == config.reflection_input_tokens
    pinned = YawConfig(
        workspace_root=tmp_path,
        context_window_tokens=context,
        max_output_tokens=1024,
        reflection_input_tokens=2048,
        rlm_max_output_chars=4000,
        max_context_chars=8000,
    )
    assert (pinned.max_output_tokens, pinned.reflection_input_tokens, pinned.rlm_max_output_chars) == (1024, 2048, 4000)
    assert pinned.max_context_chars == 8000


def test_gemma_profile_config_and_standalone_program_share_128k_defaults(tmp_path):
    config = YawConfig(workspace_root=tmp_path)
    program = HippoRAGProgram()
    lm = create_dspy_lm("Gemma-4-E4B-it-GGUF")
    proposer = RLMInstructionProposer(worker_lm=lm, orchestrator_lm=lm)
    assert CHAT_MODELS[config.qa_model].expected_context == config.context_window_tokens == 131072
    assert program.context_window_tokens == lm._yaw_context_window_tokens == 131072
    assert program.max_output_tokens == lm.kwargs["max_tokens"] == config.max_output_tokens == 8192
    assert program.shacl_healer_rlm.max_output_chars == config.rlm_max_output_chars == 20000
    assert proposer.reflection_input_tokens == config.reflection_input_tokens == 32768
    assert proposer.inner_rlm.max_output_chars == 20000
    assert proposer.max_instruction_chars == 3000
    assert proposer.max_field_chars == 1200
    assert program._context_char_limit(12000) > 50000
    assert program.rlm_sub_lm is None  # construction never loads a model


def test_standalone_program_infers_context_and_output_capacity_from_bound_lm(provider):
    url, _requests = provider
    lm = create_dspy_lm("budget-test", base_url=url, context_window_tokens=65536)
    program = HippoRAGProgram(orchestrator_lm=lm)
    assert program.context_window_tokens == 65536
    assert program.max_output_tokens == 4096
    assert program.shacl_healer_rlm.max_output_chars == 10000


def test_toml_context_changes_recompute_unpinned_budgets(tmp_path):
    path = tmp_path / "yaw.toml"
    for context in (65536, 131072):
        path.write_text(f"[inference]\ncontext_window_tokens = {context}\n", encoding="utf-8")
        config = load_config(path)
        assert config.max_output_tokens == context // 16
        assert config.reflection_input_tokens == context // 4
    assert config.rlm_max_output_chars == 20000


def test_hot_reload_does_not_restore_stale_context_or_completion_caps():
    old = HippoRAGProgram(context_window_tokens=65536, max_output_tokens=4096)
    current = HippoRAGProgram(context_window_tokens=131072, max_output_tokens=8192)
    current.load_state(old.dump_state())
    assert current.context_window_tokens == 131072
    assert current.max_output_tokens == 8192
    assert current.shacl_healer_rlm.max_output_chars == 20000


def test_128k_retains_extraction_and_shacl_context_lost_at_64k(provider):
    url, requests = provider
    passage = "Aster is a project. " + "x" * 70000 + "TAIL_FOR_128K"
    focus = "Aster is a project. " + "x" * 40000 + "FOCUS_FOR_128K"
    for context in (65536, 131072):
        lm = create_dspy_lm("budget-test", base_url=url, cache=False, context_window_tokens=context)
        program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm, context_window_tokens=context)
        program.extract_entities(passage)
        assert ("TAIL_FOR_128K" in str(requests[-1]["messages"])) == (context == 131072)
        assert requests[-1].get("max_completion_tokens", requests[-1].get("max_tokens")) == context // 16
        program.propose_shacl_repairs(
            '[{"focus_node": "Aster", "source_shape": "ConceptShape"}]', focus_context=focus, mode="direct"
        )
        assert ("FOCUS_FOR_128K" in str(requests[-1]["messages"])) == (context == 131072)


def test_standalone_engine_derives_hydration_budget_from_program(tmp_path, provider):
    url, _requests = provider
    config = YawConfig(workspace_root=tmp_path, base_url=url)
    program = HippoRAGProgram.from_config(config)
    engine = YawGraphEngine(
        db_path=tmp_path / "context.db",
        workspace_root=tmp_path,
        program=program,
        embedder=DSPyEmbeddingAdapter.from_config(config),
    )
    try:
        assert engine.max_context_chars == config.max_context_chars
        assert engine.max_context_chars > 100000
    finally:
        engine.close()


def test_structured_prompt_batches_scale_without_splitting_json_items(provider):
    url, requests = provider
    pairs = [{"pair_id": f"PAIR_{index}", "left": "Aster", "right": "Project"} for index in range(25)]
    for context in (65536, 131072):
        lm = create_dspy_lm("budget-test", base_url=url, cache=False, context_window_tokens=context)
        program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
        program.judge_synonyms(pairs)
        assert ("PAIR_24" in str(requests[-1]["messages"])) == (context == 131072)
    large_pairs = [{"pair_id": str(index), "definition": "x" * 20000} for index in range(10)]
    program.judge_synonyms(large_pairs)
    content = requests[-1]["messages"][-1]["content"]
    serialized = content.split("[[ ## candidate_pairs ## ]]\n", 1)[1].split("\n\nRespond", 1)[0]
    assert len(serialized) <= program._context_char_limit(12000)
    assert len(json.loads(serialized)) == 2
