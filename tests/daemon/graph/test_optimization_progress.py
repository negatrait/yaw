"""Native GEPA callbacks and real checkpoint files, without inference calls."""

import json

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.callbacks import CompositeCallback, notify_callbacks
from gepa.core.state import GEPAState, ValsetEvaluation

from yaw.daemon.graph.inference import GepaProgress


def test_native_frontier_displacement_and_checkpoint_restore(tmp_path):
    state = GEPAState(
        {"qa": "seed"},
        ValsetEvaluation(
            outputs_by_val_id={0: "a", 1: "b"},
            scores_by_val_id={0: 0.2, 1: 0.8},
            objective_scores_by_val_id={0: {"quality": 0.2}, 1: {"quality": 0.8}},
        ),
        frontier_type="hybrid",
    )
    # Counters are initialized by GEPA's runner after construction.
    state.total_num_evals = 2
    state.num_full_ds_evals = 1
    progress = GepaProgress(lambda: None)
    progress.restore_state(state, 2)
    assert progress.data["frontier"] == [0]
    state.update_state_with_new_program(
        [0],
        {"qa": "better"},
        ValsetEvaluation(
            outputs_by_val_id={0: "a", 1: "b"},
            scores_by_val_id={0: 0.9, 1: 0.9},
            objective_scores_by_val_id={0: {"quality": 0.9}, 1: {"quality": 0.9}},
        ),
        None,
        2,
    )
    progress.restore_state(state, 2)
    assert progress.data["frontier"] == [1]
    assert progress.data["frontier_history"][-1]["removed"] == [0]
    assert progress.data["validation"][0]["score"] == 0.5
    state.save(str(tmp_path))
    restored = GepaProgress(lambda: None)
    restored.restore_state(GEPAState.load(str(tmp_path)), 2)
    assert restored.data["frontier"] == [1]
    assert restored.data["validation"] == progress.data["validation"]
    for i in range(300):
        notify_callbacks(
            [restored], "on_pareto_front_updated", {"iteration": i, "new_front": [1], "displaced_candidates": []}
        )
    assert len(restored.data["frontier_history"]) == 250
    assert restored.data["frontier_history"][-1]["event"] == 301


def test_real_gepa_loop_publishes_scores_coverage_and_phases(tmp_path):
    audit = tmp_path / "audit.json"
    snapshots = []

    def publish():
        payload = progress.snapshot()
        audit.write_text(json.dumps(payload), encoding="utf-8")
        snapshots.append(json.loads(audit.read_text(encoding="utf-8")))

    progress = GepaProgress(publish)

    class ArithmeticTask:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            score = float(candidate["quality"])
            for _ in batch:
                progress.metric_evaluated()
                publish()
            return EvaluationBatch(
                outputs=[score] * len(batch),
                scores=[score] * len(batch),
                trajectories=[{"score": score}] * len(batch) if capture_traces else None,
                objective_scores=[{"quality": score}] * len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {"quality": [{"feedback": "Increase the task score"}]}

    result = gepa.optimize(
        seed_candidate={"quality": "0.0"},
        trainset=[0, 1],
        valset=[2, 3],
        adapter=ArithmeticTask(),
        custom_candidate_proposer=lambda candidate, reflective_dataset, components_to_update: {"quality": "0.8"},
        reflection_minibatch_size=2,
        max_metric_calls=10,
        frontier_type="hybrid",
        run_dir=str(tmp_path / "checkpoint"),
        callbacks=[CompositeCallback([progress])],
    )
    saved = json.loads(audit.read_text(encoding="utf-8"))
    assert saved["phase"] == "optimization complete"
    assert saved["frontier"] == sorted(
        {i for group in GEPAState.load(str(tmp_path / "checkpoint")).get_pareto_front_mapping().values() for i in group}
    )
    assert saved["validation"][0]["score"] == 0.0  # no smoothing floor
    assert saved["validation"][1]["score"] == result.val_aggregate_scores[1] == 0.8
    assert all(r["evaluated"] == r["total"] == 2 for r in saved["validation"])
    assert any(s["phase"] == "reflecting / generating instructions" for s in snapshots)
    assert any(s["phase"] == "minibatch evaluation" and s["phase_evaluations"] == 1 for s in snapshots)
    assert any(h["removed"] == [0] for h in saved["frontier_history"])


def test_evaluate_program_uses_canonical_recall_and_gold_matching(tmp_path, monkeypatch):
    import dspy

    import yaw.daemon.graph.inference.optimizer as optimizer_mod
    from yaw.daemon.graph.inference import (
        GraphOptimizationHooks,
        HippoRAGProgram,
        evaluate_program,
        evolve_program_prompts,
    )

    assert "doc_matches_gold" not in optimizer_mod.__all__

    long_gold = (
        "Title: Quantum Telemetry\n"
        "The primary oscillator stabilizes at 42.5 MHz during cryogenic phase lock "
        "when the secondary superconducting coil is energized."
    )
    cropped_chunk = (
        "Title: Quantum Telemetry\n"
        "The primary oscillator stabilizes at 42.5 MHz during cryogenic phase lock\n"
        "[... AST section cropped; 1 more lines ...]"
    )
    short_false_positive = "Title: Quantum"

    class DummyEvalProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.retrieve_k = lambda q, k=200: (["chunk-short", "chunk-cropped"], [0.9, 0.8])
            self.chunk_hydrator = lambda cids: [
                short_false_positive if cid == "chunk-short" else cropped_chunk for cid in cids
            ]

        def forward(self, question: str):
            return dspy.Prediction(
                answer="42.5 MHz",
                context=cropped_chunk,
                retrieved_chunk_ids=["chunk-short", "chunk-cropped"],
                retrieved_passages=[short_false_positive, cropped_chunk],
                rationale="From Quantum Telemetry oscillator stabilization.",
            )

    examples = [
        dspy.Example(
            question="At what frequency does the primary oscillator stabilize?",
            answer="42.5 MHz",
            gold_docs=[long_gold],
        ).with_inputs("question")
    ]
    res = evaluate_program(DummyEvalProgram(), examples)
    assert len(res.results) == 1
    _, pred_out, score_out = res.results[0]
    assert score_out == 1.0
    assert pred_out.em == 1.0
    assert pred_out.f1 == 1.0
    assert pred_out.answer_passage_match == 1.0

    # Also verify evolve_program_prompts._evaluate_prediction uses compute_retrieval_recalls & doc_matches_gold
    monkeypatch.setattr(optimizer_mod.RLMInstructionProposer, "preflight", lambda self: None)

    class FakeMIPROv2:
        def __init__(self, **kwargs):
            self.metric = kwargs["metric"]

        def compile(self, student, **kwargs):
            for ex in kwargs.get("trainset", ()):
                pred = dspy.Prediction(
                    answer="42.5 MHz",
                    context=cropped_chunk,
                    retrieved_chunk_ids=["chunk-short", "chunk-cropped"],
                    retrieved_passages=[short_false_positive, cropped_chunk],
                )
                self.metric(ex, pred)
            return student

    monkeypatch.setattr(dspy, "MIPROv2", FakeMIPROv2)

    dummy_lm = dspy.LM("openai/dummy", api_base="http://127.0.0.1:9/v1", api_key="none")
    prog = HippoRAGProgram(orchestrator_lm=dummy_lm, rlm_sub_lm=dummy_lm)
    hooks = GraphOptimizationHooks(
        config=None,
        dspy_cache_dir=None,
        get_meta=lambda k: None,
        set_meta=lambda k, v: None,
        commit_active_checkpoint=None,
        encode_payload=lambda v: b"",
        decode_payload=lambda b: {},
        open_checkpoint_cache=lambda ctx: {},
        ensure_candidate_indexed=lambda p: None,
        reset_sandbox_graph=lambda: None,
        index_chunks=lambda c: None,
        candidate_cache_namespace=lambda: {},
        candidate_cache_sources=lambda: [],
        prepare_optimization_pools=None,
        hydrate_top_chunks=lambda cids, p: {
            cid: (short_false_positive if cid == "chunk-short" else cropped_chunk) for cid in cids
        },
        query_rlm_stats=lambda docs: {},
        graph_rlm_stats=lambda: {},
        last_hook_program=lambda: prog,
        persist_trails=lambda p: None,
        bind_program_hooks=lambda p: None,
        finalize_compiled_program=lambda p: None,
        rehydrate_sandbox_after_optimization=lambda p: None,
    )
    save_path = tmp_path / "compiled.json"
    evolve_program_prompts(
        prog,
        examples,
        hooks=hooks,
        mode="miprov2",
        save_path=save_path,
        reflection_lm=dummy_lm,
        reflection_worker_lm=dummy_lm,
        reflection_token_counter=lambda text: len(text.split()),
    )
    audit_path = tmp_path / "compiled_graph_rlm_audit.json"
    assert audit_path.exists()
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(audit_data["eval_records"]) == 1
    rec = audit_data["eval_records"][0]
    # chunk-short (<60 chars prefix) must not match at rank 1, while chunk-cropped matches at rank 2:
    assert rec["r1"] == 0.0
    assert rec["r2"] == 1.0
    assert rec["r5"] == 1.0
    assert rec["em"] == 1.0
    assert rec["f1"] == 1.0
