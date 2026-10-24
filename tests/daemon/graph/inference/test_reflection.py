"""Native DSPy reflection over real HTTP, SQLite and process boundaries."""

import gc
import json
import subprocess
import sys
import threading
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest

from yaw.daemon.graph.candidate_cache import CandidatePayloadCodec, CandidateStageCache
from yaw.daemon.graph.inference.chat import create_dspy_lm
from yaw.daemon.graph.inference.program import RLMInstructionProposer
from yaw.daemon.graph.inference.reflection import (
    ReflectionBudgetError,
    ReflectionLM,
    ServingTokenCounter,
    deterministic_evidence,
    reflection_examples,
    resolve_evidence,
)
from yaw.daemon.graph.storage import CandidateCacheStore, open_managed_yaw_db


@pytest.fixture
def provider(request: pytest.FixtureRequest):
    calls = []
    context_tokens = getattr(request, "param", 65536)
    weights = "C:/models/reflection.gguf"

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            assert isinstance(self.server, ThreadingHTTPServer)
            if self.path.endswith("/health"):
                self.reply(
                    {
                        "all_models_loaded": [
                            {
                                "id": model,
                                "backend_url": f"http://127.0.0.1:{self.server.server_port}",
                                "launch_command": ["server", "-m", weights],
                            }
                            for model in ("reflection-model", "test-runtime-lm")
                        ]
                    }
                )
            elif self.path == "/props":
                self.reply(
                    {
                        "model_path": weights,
                        "chat_template": "test-template",
                        "default_generation_settings": {"n_ctx": context_tokens},
                    }
                )
            else:
                self.send_error(404)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/apply-template":
                self.reply({"prompt": json.dumps(request, ensure_ascii=False)})
                return
            if self.path == "/tokenize":
                # A deliberately simple loopback serving tokenizer. Tests verify
                # that callers use its counts, not a local character heuristic.
                self.reply({"tokens": list(range(len(request["content"].encode()) // 3))})
                return
            calls.append(request)
            prompt = str(request["messages"])
            if "variables_info" in prompt and "RLM_FAILURE" in prompt:
                self.send_error(500, "Selector unavailable")
                return
            if "variables_info" in prompt and "code" in prompt and "INVALID_REF" not in prompt:
                result = {
                    "reasoning": "Inspect the end of the original feedback.",
                    "code": "SUBMIT(selection={'diagnosis': 'Use exact evidence', "
                    "'references': [{'example_id': examples_with_feedback[0]"
                    "['evidence_id'], 'path': []}]})",
                }
            elif "variables_info" in prompt and "INVALID_REF" in prompt:
                result = {
                    "reasoning": "Select a reference.",
                    "code": "SUBMIT(selection={'diagnosis':'test', 'references':"
                    "[{'example_id':'invented','path':[]}]})",
                }
            elif "shortened_instruction" in prompt:
                result = {"shortened_instruction": "Use source evidence; preserve the output schema."}
            elif "OVERSIZED_DRAFT" in prompt:
                result = {"new_instruction": "x" * 20000}
            elif "LONG_DRAFT" in prompt:
                result = {"new_instruction": "x" * 2000}
            else:
                result = {"new_instruction": "Use source evidence; preserve the output schema."}
            self.reply(
                {
                    "id": "reflection",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(result)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            thread.join(5)


def proposer(url, **kwargs):
    lm = create_dspy_lm("reflection-model", base_url=url, api_key="test", cache=False)
    counter = ServingTokenCounter(url, lm.model)
    return RLMInstructionProposer(
        worker_lm=lm,
        orchestrator_lm=lm,
        token_counter=counter,
        include_demos=False,
        enable_gbnf=False,
        **kwargs,
    )


def test_exact_evidence_and_training_answers():
    examples = reflection_examples(
        [
            {
                "Inputs": {"passage": "x" * 10000 + "needle"},
                "Generated Outputs": {"answer": "wrong"},
                "Feedback": "Expected=['Turing']",
            }
        ]
    )
    selection = {
        "references": [
            {
                "example_id": examples[0]["evidence_id"],
                "path": ["Inputs", "passage"],
                "start": 10000,
                "end": 10006,
            }
        ]
    }
    assert resolve_evidence(examples, selection)[0]["evidence"] == "needle"
    assert "Turing" in resolve_evidence(examples, deterministic_evidence(examples))[0]["evidence"]["Feedback"]
    selection["references"][0]["end"] = 10007
    with pytest.raises(ValueError, match="span"):
        resolve_evidence(examples, selection)
    selection["references"][0]["example_id"] = "invented"
    with pytest.raises(KeyError):
        resolve_evidence(examples, selection)


def test_snapshot_hydrates_source_excerpts_without_copying_corpus(tmp_path):
    source = "Original corpus passage. " * 100
    codec = CandidatePayloadCodec(lambda: {"source-identity": source})
    snapshot = {"examples": [{"Inputs": {"passage": source}, "evidence": source[150:300]}]}
    encoded = codec.encode(snapshot)
    assert source[150:300] not in encoded
    db = open_managed_yaw_db(tmp_path / "snapshot.db")
    try:
        with db.transaction() as conn:
            conn.execute("CREATE TABLE snapshots(value TEXT)")
            conn.execute("INSERT INTO snapshots VALUES (?)", (encoded,))
            stored = conn.execute("SELECT value FROM snapshots").fetchone()[0]
        assert codec.decode(stored) == snapshot
        with pytest.raises(KeyError):
            CandidatePayloadCodec(lambda: {}).decode(stored)
    finally:
        db.close()


def test_native_writer_and_complete_budget(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    result = prop(
        component_name="qa.predict",
        current_instruction="Use evidence.",
        examples_with_feedback=[{"Feedback": "Expected=['Turing']"}],
    )
    assert "source evidence" in result.new_instruction
    assert "Turing" in str(calls[-1])
    assert "qa.predict" in str(calls[-1])
    assert prop.qwen_compressor.signature.output_fields.keys() == {"shortened_instruction"}
    prop.reflection_input_tokens = 10
    prop.worker_lm = prop.worker_lm.target
    prop.orchestrator_lm = prop.orchestrator_lm.target
    before = len(calls)
    with pytest.raises(ReflectionBudgetError):
        prop(component_name="qa.predict", current_instruction="Use evidence.")
    assert len(calls) == before


def test_optional_evidence_removed_before_inference(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False, reflection_input_tokens=1800)
    prop(
        component_name="qa.predict",
        current_instruction="Use evidence.",
        examples_with_feedback=[{"Feedback": "x" * 50000}],
    )
    assert "x" * 1000 not in str(calls[-1])
    assert "Use evidence." in str(calls[-1])


def test_native_compression_uses_its_actual_signature(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    result = prop(component_name="qa.predict", current_instruction="LONG_DRAFT")
    assert result.new_instruction == "Use source evidence; preserve the output schema."
    assert len(calls) == 2
    assert "shortened_instruction" in str(calls[-1])


def test_compression_and_demos_cannot_bypass_complete_request_budget(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False, reflection_input_tokens=1800)
    with pytest.raises(ReflectionBudgetError):
        prop(component_name="qa.predict", current_instruction="OVERSIZED_DRAFT")
    assert len(calls) == 1  # The oversized compression request never reaches inference.
    prop.qwen_proposer.demos = [
        dspy.Example(
            current_instruction="Use evidence.",
            examples_with_feedback=[{"Feedback": "x" * 50000}],
            length_limit="1500",
            new_instruction="Use evidence.",
        )
    ]
    with pytest.raises(ReflectionBudgetError):
        prop(component_name="qa.predict", current_instruction="Use evidence.")
    assert len(calls) == 1


def test_output_reserve_and_copy(provider):
    url, calls = provider
    prop = proposer(url)
    lm = ReflectionLM(prop.worker_lm, prop.token_counter, 16384, 1000, 800)
    with pytest.raises(ReflectionBudgetError):
        lm.copy(max_tokens=800)(messages=[{"role": "user", "content": "x" * 1000}])
    assert not calls


def test_rlm_selects_original_evidence_in_real_sandbox(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False, rlm_max_iters=2)
    outer_trace = []
    with dspy.context(trace=outer_trace):
        result = prop.propose_for_component(
            "qa.predict",
            "Use evidence.",
            [{"Inputs": {"question": "Who?"}, "Feedback": "Expected=['Turing']"}],
        )
    assert outer_trace == []
    assert "source evidence" in result
    assert prop.telemetry[-1]["fallback"] is None
    assert 2 <= len(calls) <= 3
    assert "Turing" in str(calls[-1])


@pytest.mark.parametrize("instruction", ["RLM_FAILURE", "INVALID_REF"])
def test_failed_selector_uses_original_evidence_with_native_writer(provider, instruction):
    url, calls = provider
    prop = proposer(url, enable_simba=False, rlm_max_iters=2)
    result = prop.propose_for_component(
        "qa.predict",
        instruction,
        [{"Inputs": {"question": "Who?"}, "Feedback": "Expected=['Turing']"}],
    )
    assert "source evidence" in result
    assert prop.telemetry[-1]["fallback"] is not None
    assert "Turing" in str(calls[-1])


def test_missing_tokenizer_and_legacy_state_rejected():
    prop = RLMInstructionProposer()
    with pytest.raises(ReflectionBudgetError, match="tokenizer"):
        prop(component_name="qa.predict")
    state = prop.dump_simba_state()
    state["version"] = 1
    with pytest.raises(ValueError, match="Unsupported"):
        prop.load_simba_state(state)


@pytest.mark.parametrize("interrupt_after_proposal", [False, True])
def test_proposal_checkpoint_survives_fresh_process(tmp_path, provider, interrupt_after_proposal):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    with db.transaction() as conn:
        store = CandidateCacheStore(conn)
    cache = CandidateStageCache(store, db, "reflection", lambda: "test")
    prop.checkpoint_get = lambda key: json.loads(cache.get((key,))) if cache.get((key,)) else None
    prop.checkpoint_put = lambda key, value: cache.__setitem__((key,), json.dumps(value))
    candidate = {"qa.predict": "Use evidence."}
    data = {"qa.predict": [{"Inputs": {"question": "Who?"}, "Feedback": "Expected Turing"}]}
    # Select and persist evidence through the actual DSPy sandbox first.
    prop.select_evidence("qa.predict", candidate["qa.predict"], data["qa.predict"])
    evidence_calls = len(calls)
    if interrupt_after_proposal:
        save = prop.checkpoint_put

        def interrupted(key, value):
            save(key, value)
            if key.startswith("proposal:"):
                raise RuntimeError("Power lost after durable proposal")

        prop.checkpoint_put = interrupted
        with pytest.raises(RuntimeError, match="Power lost"):
            prop(candidate, data, ["qa.predict"])
        result = {"qa.predict": "Use source evidence; preserve the output schema."}
    else:
        result = prop(candidate, data, ["qa.predict"])
    count = len(calls)
    assert count == evidence_calls + 1
    db.close()
    script = """
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('reflection_tests',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
p=module.proposer(sys.argv[2],enable_simba=False)
db=module.open_managed_yaw_db(Path(sys.argv[3]))
with db.transaction() as conn: store=module.CandidateCacheStore(conn)
cache=module.CandidateStageCache(store,db,'reflection',lambda:'test')
p.checkpoint_get=lambda key: json.loads(cache.get((key,))) if cache.get((key,)) else None
try:
 candidate={'qa.predict':'Use evidence.'}
 data={'qa.predict':[{'Inputs':{'question':'Who?'},'Feedback':'Expected Turing'}]}
 print(json.dumps(p(candidate,data,['qa.predict'])))
finally: db.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, __file__, url, str(tmp_path / "yaw.db")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == result
    assert len(calls) == count


def test_each_reflection_role_requires_its_own_resident_tokenizer(provider):
    url, calls = provider
    prop = proposer(url)
    prop.orchestrator_lm = create_dspy_lm("unloaded-selector", base_url=url, api_key="test")
    with pytest.raises(ReflectionBudgetError, match="not resident"):
        prop.preflight()
    assert calls == []


def test_repeated_native_proposals_release_inputs_and_histories(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)

    class TraceRecord(dict):
        pass

    retained = []
    outer_trace = []
    with dspy.context(trace=outer_trace, disable_history=True):
        for index in range(12):
            record = TraceRecord(Feedback=f"Expected training answer {index}: " + "x" * 50000)
            retained.append(weakref.ref(record))
            result = prop(
                component_name="qa.predict",
                current_instruction="Use evidence.",
                examples_with_feedback=[record],
            )
            assert result.new_instruction
            del record, result
            calls.clear()  # Provider inspection is test instrumentation, not application retention.
    gc.collect()
    assert all(reference() is None for reference in retained)
    assert outer_trace == []
    assert prop.worker_lm.history == prop.worker_lm.target.history == []
    assert prop.qwen_proposer.traces == prop.qwen_compressor.traces == []
    assert prop.worker_module.sub_telemetry == []


def test_completed_request_invalidates_when_learned_writer_changes(tmp_path, provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    try:
        with db.transaction() as conn:
            store = CandidateCacheStore(conn)
        cache = CandidateStageCache(store, db, "reflection", lambda: "test")

        # CandidateStageCache uses tuple keys consistently on disk.
        def get(key):
            raw = cache.get((key,))
            return json.loads(raw) if raw is not None else None

        prop.checkpoint_get = get
        prop.checkpoint_put = lambda key, value: cache.__setitem__((key,), json.dumps(value))
        candidate = {"qa.predict": "Use evidence."}
        data = {"qa.predict": [{"Inputs": {"question": "Who?"}, "Feedback": "Expected Turing"}]}
        first = prop(candidate, data, ["qa.predict"])
        completed_calls = len(calls)
        assert prop(candidate, data, ["qa.predict"]) == first
        assert len(calls) == completed_calls
        prop.qwen_proposer.signature = prop.qwen_proposer.signature.with_instructions(
            prop.qwen_proposer.signature.instructions + "\nPreserve exact source names in task outputs."
        )
        assert prop(candidate, data, ["qa.predict"]) == first
        assert len(calls) == completed_calls + 1  # Evidence is reused; the new writer runs once.
        assert "Preserve exact source names in task outputs." in str(calls[-1])
        assert prop(candidate, data, ["qa.predict"]) == first
        assert len(calls) == completed_calls + 1
    finally:
        db.close()


def test_budget_includes_default_lm_template_options(provider):
    url, calls = provider
    prop = proposer(url)
    prop.worker_lm.kwargs["extra_body"] = {"chat_template_kwargs": {"tool_description": "x" * 50000}}
    with pytest.raises(ReflectionBudgetError, match="input limit"):
        prop(component_name="qa.predict", current_instruction="Use evidence.")
    assert calls == []


@pytest.mark.parametrize("provider", [65536, 131072], indirect=True)
def test_default_reflection_budgets_follow_verified_serving_capacity(provider):
    url, calls = provider
    counter = ServingTokenCounter(url, "reflection-model")
    context = counter.context_tokens
    lm = create_dspy_lm("reflection-model", base_url=url, cache=False, context_window_tokens=context)
    guarded = ReflectionLM(lm, counter, context_tokens=context)
    assert guarded.context_tokens == context
    assert guarded.input_tokens == context // 4
    assert guarded.output_tokens == guarded.kwargs["max_tokens"] == context // 16
    prop = RLMInstructionProposer(orchestrator_lm=lm, worker_lm=lm, token_counter=counter)
    prop.preflight()
    assert prop.reflection_input_tokens == prop.worker_lm.input_tokens == context // 4
    assert prop.max_instruction_chars == 1500 * context // 65536
    with pytest.raises(ReflectionBudgetError, match="input limit"):
        guarded.count([{"role": "user", "content": "x" * (3 * (context // 4) + 100)}])
    assert not calls
