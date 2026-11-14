"""Native SIMBA/task inference over HTTP, with real SQLite resume and disk caches."""

import json
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import pytest
from embedding_fixture import IdentifiedEmbeddingCallable

from daemon.graph.test_indexing_overlap import adapter as overlap_adapter
from daemon.graph.test_indexing_overlap import provider as indexing_provider
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    RLMInstructionProposer,
    YawProgram,
    compute_qa_em_f1,
    compute_task_quality_score,
    create_dspy_lm,
)
from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter
from yaw.daemon.graph.inference.optimizer import configure_simba_task_learning
from yaw.daemon.indexing import bind_native_indexing

__all__ = ["indexing_provider"]

GOOD = "FULL_NAME: Answer with the full name supported by the context."
BAD = "Ignore the evidence and always answer Unknown."
BASE = "Answer with a surname."


@pytest.fixture
def provider():
    calls = []
    drafts = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert isinstance(self.server, ThreadingHTTPServer)
            value = {
                "all_models_loaded": [
                    {
                        "id": "worker-model",
                        "backend_url": f"http://127.0.0.1:{self.server.server_port}",
                        "launch_command": ["server", "-m", "C:/worker.gguf"],
                    }
                ]
            }
            if self.path == "/props":
                value = {
                    "model_path": "C:/worker.gguf",
                    "chat_template": "test",
                    "default_generation_settings": {"n_ctx": 65536},
                }
            self.reply(value)

        def reply(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            nonlocal drafts
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/apply-template":
                self.reply({"prompt": json.dumps(request)})
                return
            if self.path == "/tokenize":
                self.reply({"tokens": list(range(len(request["content"]) // 4))})
                return
            prompt = str(request["messages"])
            calls.append(request)
            if "task-model" in request["model"]:
                answer = "Alan Turing" if "FULL_NAME" in prompt else ("Unknown" if BAD in prompt else "Turing")
                result = {"plan": "Use the provided evidence.", "answer": answer}
            elif "module_advice" in prompt and "discussion" in prompt:
                result = {
                    "discussion": "Full names improve measured task answers.",
                    "module_advice": {"llm_proposer": "LEARNED_RULE: Use FULL_NAME."},
                }
            else:
                result = {
                    "new_instruction": GOOD
                    if (
                        "LEARNED_RULE" in prompt
                        or drafts < 2
                        or any(m["role"] == "assistant" for m in request["messages"])
                    )
                    else BAD
                }
                drafts += 1
            response = {
                "id": "simba-test",
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
            yield f"http://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            thread.join(timeout=5)


def task_metric(gold, pred):
    em, f1 = compute_qa_em_f1(pred.answer, gold.gold_answers)
    return float(compute_task_quality_score(em, em=em, f1=f1, tokens=25).score)


def make_learning(root, url, *, resume=False, answer="Alan Turing"):
    task_lm = create_dspy_lm("task-model", base_url=url, api_key="test", cache=False)
    worker_lm = create_dspy_lm("worker-model", base_url=url, api_key="test", cache=False)
    task = YawProgram(enable_gbnf=False, orchestrator_lm=task_lm, rlm_sub_lm=task_lm)
    task.qa.predict.signature = task.qa.predict.signature.with_instructions(BASE)
    # Context-only calibration exercises real QA without embedding/retrieval I/O.
    engine = bind_native_indexing(
        YawGraphEngine(
            root / "yaw.db",
            root,
            task,
            DSPyEmbeddingAdapter(
                embedder=dspy.Embedder(IdentifiedEmbeddingCallable(lambda texts: [], "simba-empty-fixture"))
            ),
        )
    )
    proposer = RLMInstructionProposer(
        worker_lm=worker_lm,
        orchestrator_lm=worker_lm,
        enable_gbnf=False,
    )
    train = [
        dspy.Example(
            context="Alan Turing designed ACE.",
            question=f"Who designed ACE? {i}",
            gold_answers=[answer],
        ).with_inputs("context", "question")
        for i in range(3)
    ]
    try:
        with dspy.context(lm=None):
            configure_simba_task_learning(
                proposer,
                task,
                train,
                task_metric,
                hooks=engine._build_optimization_hooks(),
                save_path=root / "compiled.json",
                resume=resume,
            )
    except Exception:
        engine.close()
        raise
    return engine, proposer


def example():
    return dspy.Example(
        task_candidate={"qa.predict": BASE},
        component_name="qa.predict",
        current_instruction=BASE,
        examples_with_feedback=[{"Inputs": {"question": "Who?"}, "Feedback": "Incomplete answer"}],
    ).with_inputs("component_name", "current_instruction", "examples_with_feedback")


def test_task_reward_disk_cache_and_isolated_training_queries(tmp_path, provider):
    url, calls = provider
    engine, proposer = make_learning(tmp_path, url)
    try:
        ex = example()
        good = proposer._score_task_improvement(ex, dspy.Prediction(new_instruction=GOOD))
        bad = proposer._score_task_improvement(ex, dspy.Prediction(new_instruction=BAD))
        same = proposer._score_task_improvement(ex, dspy.Prediction(new_instruction=BASE))
        # Partial answer (BASE: surname only, F1>0) scores higher than wrong answer (BAD: F1=0).
        assert good > same == 0.5 > bad
        assert len(calls) == 6  # paired task evaluations; subsequent repeats must reuse the cache
        assert all("task-model" in call["model"] for call in calls)
        assert all("Who designed ACE? 2" not in str(call) for call in calls)
        proposer._score_task_improvement(ex, dspy.Prediction(new_instruction=GOOD))
        assert len(calls) == 6
        assert "task_candidate" not in ex.inputs()
        correct_baseline = dspy.Example(
            **{**ex.toDict(), "task_candidate": {"qa.predict": GOOD}, "current_instruction": GOOD}
        ).with_inputs("component_name", "current_instruction", "examples_with_feedback")
        assert proposer._score_task_improvement(correct_baseline, dspy.Prediction(new_instruction=BAD)) < 0.5
    finally:
        engine.close()


def test_native_simba_learning_survives_process_restart(tmp_path, provider):
    url, calls = provider
    engine, proposer = make_learning(tmp_path, url)
    try:
        proposer._simba_examples = [example(), example()]
        before = proposer.llm_proposer.signature.instructions
        assert proposer.step_simba()
        assert proposer._simba_steps == 1
        assert any(call.get("temperature") == 1.0 and "worker-model" in call["model"] for call in calls)
        assert proposer.llm_proposer.signature.instructions != before or proposer.llm_proposer.demos
        # Persist demos on both predictors as well as the genuinely learned rule.
        proposer.llm_compressor.demos = [
            dspy.Example(shortened_instruction=GOOD, instruction=GOOD, length_limit="1500")
        ]
        proposer.save_simba_state(proposer.dump_simba_state())
        learned = proposer.dump_simba_state()
    finally:
        engine.close()

    before_restart = len(calls)
    script = """
import json, runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[2]))
helpers = runpy.run_path(sys.argv[1])
engine, proposer = helpers['make_learning'](Path(sys.argv[2]), sys.argv[3], resume=True)
try:
    state = proposer.dump_simba_state()
    score = proposer._score_task_improvement(helpers['example'](),
        helpers['dspy'].Prediction(new_instruction=helpers['GOOD']))
    assert score > 0.5
    assert proposer.worker_module.llm_proposer is proposer.llm_proposer
    proposal = proposer(component_name='qa.predict', current_instruction=helpers['BASE'],
        examples_with_feedback=[{'Feedback': 'Incomplete answer'}])
    assert proposal.new_instruction == helpers['GOOD']
    print('RESTORED=' + json.dumps(state, sort_keys=True))
finally:
    engine.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(tmp_path), url],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    restored = json.loads(
        next(line.removeprefix("RESTORED=") for line in result.stdout.splitlines() if line.startswith("RESTORED="))
    )
    assert restored == learned
    assert len(calls) == before_restart + 1  # only the next worker call; task scores are cached
    assert "worker-model" in calls[-1]["model"]


def test_resume_rejects_changed_task_labels(tmp_path, provider):
    url, _calls = provider
    engine, proposer = make_learning(tmp_path, url)
    proposer.save_simba_state(proposer.dump_simba_state())
    engine.close()
    with pytest.raises(ValueError, match="does not match"):
        make_learning(tmp_path, url, resume=True, answer="Different gold")


def test_failed_sqlite_snapshot_preserves_previous_complete_state(tmp_path, provider):
    url, _calls = provider
    engine, proposer = make_learning(tmp_path, url)
    try:
        proposer.save_simba_state(proposer.dump_simba_state())
        before = engine.conn.execute("SELECT value FROM meta WHERE key LIKE 'simba_worker:%'").fetchone()[0]
        engine.conn.execute("PRAGMA query_only=ON")
        proposer._simba_steps = 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            proposer.save_simba_state(proposer.dump_simba_state())
        after = engine.conn.execute("SELECT value FROM meta WHERE key LIKE 'simba_worker:%'").fetchone()[0]
        assert after == before
    finally:
        engine.close()


def test_no_format_reward_without_task_scorer():
    proposer = RLMInstructionProposer()
    proposer._simba_examples = [example(), example()]
    assert not proposer.step_simba()


def test_active_request_survives_cache_eviction_with_atomic_worker_state(tmp_path, provider):
    url, _calls = provider
    engine, proposer = make_learning(tmp_path, url)
    try:
        proposer.checkpoint_put("request:one", {"phase": "collecting", "state": proposer.dump_simba_state()})
        proposer.checkpoint_put("evidence:one", {"references": ["original-training-record"]})
        proposer._simba_steps = 1
        proposer.checkpoint_put("request:one", {"phase": "learned", "state": proposer.dump_simba_state()})
        with engine.managed_conn.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key LIKE 'reflection_checkpoint:%'")
    finally:
        engine.close()
    engine, restored = make_learning(tmp_path, url, resume=True)
    try:
        assert restored.dump_simba_state()["steps"] == 1
        assert restored.checkpoint_get("request:one")["phase"] == "learned"
        assert restored.checkpoint_get("evidence:one") == {"references": ["original-training-record"]}
    finally:
        engine.close()


def test_nonfinite_reward_is_observable_and_never_successful():
    proposer = RLMInstructionProposer(task_scorer=lambda _: float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        proposer._score_task_improvement(example(), dspy.Prediction(new_instruction=GOOD))
    assert proposer._simba_reward_errors == 1


def test_native_simba_failed_task_evaluation_keeps_durable_worker(tmp_path, provider, caplog):
    url, _calls = provider
    engine, proposer = make_learning(tmp_path, url)
    try:
        proposer._simba_examples = [example(), example()]
        before = proposer.dump_simba_state()
        proposer.save_simba_state(before)

        def failed_score(_candidate):
            raise RuntimeError("task provider unavailable")

        proposer.task_scorer = failed_score
        with pytest.raises(RuntimeError, match="SIMBA task evaluations failed"):
            proposer.step_simba()
        assert proposer.dump_simba_state() == before
    finally:
        engine.close()
    engine, restored = make_learning(tmp_path, url, resume=True)
    try:
        assert restored.dump_simba_state() == before
    finally:
        engine.close()


def test_prediction_usage_is_scoped_and_counts_all_native_events():
    from dspy.utils.usage_tracker import track_usage

    from yaw.daemon.graph.inference.optimizer import _prediction_lm_usage

    # More than 20 earlier model events belong to a different prediction.
    with track_usage() as prior:
        for _ in range(27):
            prior.add_usage("model", {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    with track_usage() as current:
        for _ in range(25):
            current.add_usage("model", {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    prediction = dspy.Prediction(answer="current")
    prediction.set_lm_usage(current.get_total_tokens())
    assert _prediction_lm_usage(prediction) == (125, True)
    assert _prediction_lm_usage(dspy.Prediction(answer="retrieval only")) == (0, False)
    assert prior.get_total_tokens()["model"]["total_tokens"] == 324


def test_current_prediction_usage_with_fresh_candidate_indexing(tmp_path, indexing_provider):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.inference.optimizer import _prediction_lm_usage

    url, state = indexing_provider
    state["delay"] = 0
    source = "Ada studies Algebra."
    (tmp_path / "source.txt").write_text(source, encoding="utf-8")
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False)
    program = YawProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    engine = bind_native_indexing(YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, overlap_adapter(url)))
    try:
        engine._opt_chunks_pool = [(ChunkPointer("source", "source.txt", 1, 1), source)]
        engine._in_optimization = True

        def retrieval(question, program, *, item_docs=None, **kwargs):
            engine._ensure_candidate_chunks_indexed(program, item_docs)
            return source, (source,), (1.0,), ("source",)

        engine.program.set_retrieval_hook(retrieval)
        with dspy.context(track_usage=True):
            prediction = engine.program(question="Who studies Algebra?", item_docs=[source], retrieve_only=True)
        usage = _prediction_lm_usage(prediction)
        stats = engine._query_rlm_stats_for_eval([source])
        assert sum(kind == "chat" for kind, _, _ in state["intervals"]) == 3
        assert usage == (0, True)
        assert stats["relation_judge_tokens"] == 15
        assert "openie_tokens" not in stats
    finally:
        engine.close()
