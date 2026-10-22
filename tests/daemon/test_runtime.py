"""Unmocked integration tests for YawRuntime, FileLock, 32-slot queue, crash-resume, CLI, and optimization."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import pytest

from yaw.core import (
    OpenIERecord,
    YawConfig,
    YawRuntimeError,
    load_config,
)
from yaw.daemon.cli import build_parser, inspect_status, main
from yaw.daemon.graph.inference import (
    HippoRAGProgram,
    compute_sizecoding_pareto_score,
    resolve_embedding_profile,
)
from yaw.daemon.graph.storage import get_meta, get_openie_checkpoints, open_yaw_db
from yaw.daemon.runtime import YawRuntime


class DeterministicTestLM(dspy.LM):
    """Deterministic in-process dspy.LM for offline daemon runtime tests."""

    def __init__(self) -> None:
        super().__init__(model="openai/test-runtime-lm")
        self.fail_online: bool = False

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if self.fail_online:
            raise ConnectionError(
                "Inference server at http://localhost:13305/api/v1 is unreachable"
            )
        req_str = str(prompt or messages or kwargs)
        if dspy.settings.usage_tracker is not None:
            dspy.settings.usage_tracker.add_usage(
                self.model,
                {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            )
        if (
            "JudgeSynonyms" in req_str
            or "candidate_pairs_json" in req_str
            or "candidate_pairs" in req_str
        ):
            return ['{"decisions": []}']
        if (
            "ProposeSHACLRepairs" in req_str
            or "HealSHACLViolations" in req_str
            or "violations_json" in req_str
        ):
            return ['{"repairs": []}']
        if (
            "DefineMissingEntities" in req_str
            or "missing_entities" in req_str
            or "entity_definitions" in req_str
        ):
            if "Turing" in req_str:
                return [
                    '{"entity_definitions": [["Alan Turing", "English mathematician and cryptanalyst"], ["Bletchley Park", "British wartime codebreaking center"]]}'
                ]
            if "Lovelace" in req_str:
                return [
                    '{"entity_definitions": [["Ada Lovelace", "English mathematician"], ["Analytical Engine", "Early mechanical general-purpose computer"]]}'
                ]
            return [
                '{"entity_definitions": [["University of Southampton", "Public research university in Southampton"], ["1862", "Founding year of the Hartley Institution"]]}'
            ]
        if "FilterTriples" in req_str or "fact_before_filter" in req_str:
            if "Turing" in req_str:
                return ['{"fact": [["Alan Turing", "worked at", "Bletchley Park"]]}']
            if "Lovelace" in req_str:
                return ['{"fact": [["Ada Lovelace", "programmed", "Analytical Engine"]]}']
            return ['{"fact": [["University of Southampton", "founded in", "1862"]]}']
        if "ExtractTriples" in req_str or '"triples"' in req_str or "`triples`" in req_str:
            if "Turing" in req_str:
                return [
                    '{"triples": [{"sub": "Alan Turing", "pred": "worked at", "obj": "Bletchley Park", "def": "Alan Turing worked at Bletchley Park."}]}'
                ]
            if "Lovelace" in req_str:
                return [
                    '{"triples": [{"sub": "Ada Lovelace", "pred": "programmed", "obj": "Analytical Engine", "def": "Ada Lovelace programmed the Analytical Engine."}]}'
                ]
            return [
                '{"triples": [{"sub": "University of Southampton", "pred": "founded in", "obj": "1862", "def": "University of Southampton was founded in 1862."}]}'
            ]
        if "ExtractEntities" in req_str or "named_entities" in req_str:
            if "Turing" in req_str:
                return [
                    '{"named_entities": [["Alan Turing", "English mathematician and cryptanalyst"], ["Bletchley Park", "British wartime codebreaking center"]]}'
                ]
            if "Lovelace" in req_str:
                return [
                    '{"named_entities": [["Ada Lovelace", "English mathematician"], ["Analytical Engine", "Early mechanical general-purpose computer"]]}'
                ]
            return [
                '{"named_entities": [["University of Southampton", "Public research university in Southampton"], ["1862", "Founding year of the Hartley Institution"]]}'
            ]
        if "ExtractQueryIntent" in req_str or "query_intent" in req_str:
            return ['{"query_intent": "Alan Turing worked at Bletchley Park"}']
        return ['{"reasoning": "Derived from retrieved passages.", "answer": "Bletchley Park"}']


class DeterministicTestEmbedder:
    """Deterministic L2-normalized 16-D embedder with EmbeddingProfile prefix support."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.profile = resolve_embedding_profile("embeddinggemma-300m-GGUF", "distinct-titled")
        self.fail_online: bool = False

    def update_prefixes(
        self,
        *,
        query_prefix: str | None = None,
        fact_query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        from dataclasses import replace

        updates: dict[str, Any] = {}
        if query_prefix is not None:
            updates["query_prefix"] = query_prefix
        if fact_query_prefix is not None:
            updates["fact_query_prefix"] = fact_query_prefix
        if document_prefix is not None:
            updates["document_prefix"] = document_prefix
        if updates:
            self.profile = replace(self.profile, **updates)

    def encode(self, texts: Sequence[str], instruction: str = "") -> list[np.ndarray]:
        if self.fail_online:
            raise ConnectionError("Embedding server unreachable")
        vecs: list[np.ndarray] = []
        for t in texts:
            tl = t.lower()
            v = np.zeros(self.dim, dtype=np.float32)
            if "turing" in tl or "bletchley" in tl:
                v[0] = 1.0
            elif "lovelace" in tl or "analytical" in tl:
                v[1] = 1.0
            elif "southampton" in tl or "1862" in tl:
                v[2] = 1.0
            else:
                digest = hashlib.md5(tl.encode("utf-8")).digest()
                for i in range(self.dim):
                    v[i] = float(digest[i % len(digest)]) / 255.0 + 0.01
            norm = float(np.linalg.norm(v))
            vecs.append(v / norm if norm > 0 else v)
        return vecs


def _make_test_config(tmp_path: Path, *, queue_maxsize: int = 32) -> tuple[YawConfig, Path]:
    toml_path = tmp_path / "yaw.toml"
    toml_content = f"""
[paths]
corpus_root = "corpus"
db_path = ".yaw/yaw.db"
lock_path = ".yaw/yaw.db.lock"
state_dir = ".yaw/coco_state"
log_path = ".yaw/yaw.log"
status_path = ".yaw/status.json"
compiled_program_path = ".yaw/compiled_program.json"

[ingestion]
included_patterns = ["*.md", "*.txt"]
excluded_patterns = [".*", "*/.*", "__pycache__/*"]
chunk_size = 1200
chunk_overlap = 100
min_chunk_size = 20
poll_interval_seconds = 0.2

[inference]
base_url = "http://localhost:13305/api/v1"
enable_gbnf = false

[graph]
linking_top_k = 4
retrieval_top_k = 3
embed_batch_size = 16

[runtime]
queue_maxsize = {queue_maxsize}
http_port = 13306
"""
    toml_path.write_text(toml_content.strip() + "\n", encoding="utf-8")
    cfg = load_config(toml_path)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True, exist_ok=True)
    return cfg, toml_path


def _make_program_and_embedder() -> tuple[
    HippoRAGProgram, Any, DeterministicTestLM, DeterministicTestEmbedder
]:
    lm = DeterministicTestLM()
    dspy.settings.configure(lm=lm)
    prog = HippoRAGProgram(
        enable_gbnf=False,
        orchestrator_lm=lm,
        rlm_sub_lm=lm,
    )
    emb = DeterministicTestEmbedder(dim=16)
    return prog, emb, lm, emb


def test_load_config_and_fingerprint(tmp_path: Path) -> None:
    """Verify stdlib tomllib loader, relative path resolution, and deterministic config fingerprinting."""
    repo_cfg = load_config(Path(__file__).resolve().parents[2] / "yaw.toml")
    assert repo_cfg.queue_maxsize == 32
    assert repo_cfg.embedding_model == "embeddinggemma-300m-GGUF"
    assert repo_cfg.extraction_model == "Qwen3-4B-Instruct-2507-GGUF"
    assert repo_cfg.qa_model == "Qwen3-4B-Instruct-2507-GGUF"
    assert repo_cfg.config_fingerprint().startswith("yaw-cfg-v1:")

    cfg1, toml_path = _make_test_config(tmp_path)
    cfg2 = load_config(toml_path)
    assert cfg1.config_fingerprint() == cfg2.config_fingerprint()
    assert cfg1.corpus_root == (tmp_path / "corpus").resolve()
    assert cfg1.db_path == (tmp_path / ".yaw" / "yaw.db").resolve()

    with pytest.raises(YawRuntimeError) as exc_info:
        load_config(tmp_path / "nonexistent.toml")
    assert exc_info.value.code == "INVALID_INPUT"


def test_optimize_resume_parser_aliases() -> None:
    parser = build_parser()
    for command in (["dspy", "optimize"], ["optimize"]):
        assert parser.parse_args([*command, "--resume"]).resume
        for flag in ("--db", "--db-path"):
            args = parser.parse_args([*command, flag, "latest"])
            assert args.db_path == "latest"


def test_cli_resume_without_a_run_fails_before_dataset_or_provider_access(tmp_path: Path) -> None:
    cfg, toml_path = _make_test_config(tmp_path)
    errors = io.StringIO()
    with redirect_stderr(errors):
        code = main(["dspy", "optimize", "--config", str(toml_path), "--resume"])
    assert code == 1
    assert "No optimization run exists" in errors.getvalue()
    assert cfg.db_path is not None and not cfg.db_path.exists()


def test_cli_resume_real_gepa_checkpoint_and_custom_output(tmp_path: Path) -> None:
    """Real GEPA/DSPy + SQLite + CocoIndex; deterministic LM tests orchestration only."""
    from gepa.core.state import GEPAState, ValsetEvaluation

    cfg, toml_path = _make_test_config(tmp_path)
    dataset = tmp_path / "resume_dataset.json"
    records = [
        {
            "id": f"q{i}", "question": f"Where did Alan Turing work ({i})?",
            "answer": ["Bletchley Park"],
            "paragraphs": [{
                "title": "Alan Turing",
                "text": "Alan Turing worked at Bletchley Park as a wartime codebreaker.",
                "is_supporting": True,
            }],
        }
        for i in range(2)
    ]
    dataset.write_text(json.dumps(records), encoding="utf-8")
    custom_output = tmp_path / "my_compiled.json"
    common = [
        "dspy", "optimize", "--config", str(toml_path), "--dataset", str(dataset),
        "--mode", "gepa", "--max-queries", "2", "--train-queries", "1",
        "--num-trials", "1",
    ]

    def run_cli(extra: list[str], *, bare: bool = False) -> tuple[int, str, str]:
        prog, emb, _, _ = _make_program_and_embedder()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            invocation = ["dspy", "optimize", "--config", str(toml_path)] if bare else common
            code = main([*invocation, *extra], program=prog, embedder=emb)
        return code, out.getvalue(), err.getvalue()

    rc, output, errors = run_cli(["--output", str(custom_output)])
    assert rc == 0, errors
    original = json.loads(output)
    db_path = Path(original["sandbox_db_path"])
    telemetry = tmp_path / "my_compiled_telemetry"
    checkpoint = telemetry / "gepa_state.bin"
    assert checkpoint.is_file()
    state = GEPAState.load(str(telemetry))
    # Persist a superior candidate using GEPA's real state API. A restart must
    # return it, rather than deleting the checkpoint and returning the seed.
    candidate = dict(state.program_candidates[0])
    candidate["qa.predict"] += "\nRESUMED_WINNER: ground every answer in evidence."
    state.update_state_with_new_program(
        [0], candidate,
        ValsetEvaluation(
            outputs_by_val_id={0: dspy.Prediction(answer="Bletchley Park")},
            scores_by_val_id={0: 1.0},
            objective_scores_by_val_id={0: {"f1": 1.0}},
        ),
        str(telemetry), 1,
    )
    state.save(str(telemetry), use_cloudpickle=True)
    custom_output.unlink()  # unfinished run need not have a final program
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
        initial_openie = conn.execute("SELECT * FROM openie_chunks").fetchall()
    for selection in (["--resume"], ["--db-path", "latest"]):
        rc, output, errors = run_cli(selection, bare=True)
        assert rc == 0, errors
        resumed = json.loads(output)
        assert resumed["sandbox_db_path"] == original["sandbox_db_path"]
        assert custom_output.is_file()
        assert "RESUMED_WINNER" in custom_output.read_text(encoding="utf-8")
        loaded = GEPAState.load(str(telemetry))
        assert loaded.program_candidates == state.program_candidates
        assert loaded.total_num_evals == state.total_num_evals
        pools = json.loads((tmp_path / "my_compiled_pools.json").read_text())
        assert pools["corpus_chunk_ids"]
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT * FROM openie_chunks").fetchall() == initial_openie

    # Same length, different questions: reject before replacing the corpus or
    # touching the checkpoint, preserving both for inspection or recovery.
    original_checkpoint = checkpoint.read_bytes()
    assert cfg.corpus_root is not None
    corpus_path = cfg.corpus_root / ".yaw_optimize_slice.json"
    original_corpus = corpus_path.read_bytes()
    records[0]["question"] = "A changed training question"
    dataset.write_text(json.dumps(records), encoding="utf-8")
    rc, _, errors = run_cli(["--resume"], bare=True)
    assert rc == 1
    assert "Cannot resume" in errors
    assert checkpoint.read_bytes() == original_checkpoint
    assert corpus_path.read_bytes() == original_corpus


def test_resume_preserves_unverifiable_legacy_checkpoint(tmp_path: Path) -> None:
    cfg, _ = _make_test_config(tmp_path)
    prog, emb, _, _ = _make_program_and_embedder()
    target = tmp_path / "legacy.json"
    checkpoint = tmp_path / "legacy_telemetry" / "gepa_state.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"legacy checkpoint must survive")
    runtime = YawRuntime(cfg, program=prog, embedder=emb)
    runtime.start(sync_on_start=False)
    try:
        examples = [dspy.Example(question="Test?").with_inputs("question")]
        with pytest.raises(YawRuntimeError, match="legacy GEPA checkpoint"):
            runtime.optimize(examples, mode="gepa", save_path=target, num_trials=1)
        assert checkpoint.read_bytes() == b"legacy checkpoint must survive"
    finally:
        runtime.stop()


def test_second_runtime_fails_fast_on_lock(tmp_path: Path) -> None:
    """D1-A5: A second YawRuntime targeting the same yaw.db.lock fails fast with BUSY."""
    cfg, _ = _make_test_config(tmp_path)
    prog1, emb1, _, _ = _make_program_and_embedder()
    prog2, emb2, _, _ = _make_program_and_embedder()

    rt1 = YawRuntime(cfg, program=prog1, embedder=emb1)
    rt2 = YawRuntime(cfg, program=prog2, embedder=emb2)

    rt1.start(sync_on_start=False)
    try:
        assert rt1.status().lifecycle == "ready"
        with pytest.raises(YawRuntimeError) as exc_info:
            rt2.start(sync_on_start=False)
        assert exc_info.value.code == "BUSY"
    finally:
        rt1.stop()

    # Once rt1 stops and releases yaw.db.lock, rt2 can start cleanly
    rt2.start(sync_on_start=False)
    try:
        assert rt2.status().lifecycle == "ready"
    finally:
        rt2.stop()


def test_bounded_32_slot_queue_overflow_and_non_blocking_status(tmp_path: Path) -> None:
    """D1-A5 & E2-A3: 32-slot bounded queue overflows with BUSY on the 33rd queued item while status() stays <50ms."""
    cfg, _ = _make_test_config(tmp_path, queue_maxsize=32)
    prog, emb, _, _ = _make_program_and_embedder()
    rt = YawRuntime(cfg, program=prog, embedder=emb)
    rt.start(sync_on_start=False)

    gate = threading.Event()
    worker_entered = threading.Event()
    progress_stages: list[str] = []

    def _blocking_task() -> str:
        worker_entered.set()
        gate.wait(timeout=5.0)
        return "done"

    try:
        # 1. Occupy the worker thread with a slow operation
        first_fut = rt.submit("slow_op", _blocking_task, progress_callback=progress_stages.append)
        assert worker_entered.wait(timeout=2.0)

        # 2. Fill all 32 slots in the bounded FIFO queue
        queued_futures = [rt.submit("noop", lambda: "ok") for _ in range(32)]
        assert len(queued_futures) == 32

        # 3. 33rd submission while queue is full must immediately raise BUSY
        with pytest.raises(YawRuntimeError) as exc_info:
            rt.submit("overflow", lambda: "overflow")
        assert exc_info.value.code == "BUSY"

        # 4. Non-blocking cached status() and stats() must return in <50ms while worker is blocked
        t0 = time.perf_counter()
        status = rt.status()
        stats = rt.stats()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0
        assert status.queue_depth == 32
        assert status.active_op == "slow_op"
        assert stats.config_fingerprint == cfg.config_fingerprint()

        # 5. Release worker and verify all queued tasks complete
        gate.set()
        assert first_fut.result(timeout=5.0) == "done"
        for f in queued_futures:
            assert f.result(timeout=5.0) == "ok"
        assert progress_stages == ["queued", "started", "ready"]
    finally:
        gate.set()
        rt.stop()


def test_stage1_crash_resume_does_not_reextract_completed_chunks(tmp_path: Path) -> None:
    """D1-A3 & D1-A5: Crash mid-OpenIE persists completed chunks and pending_sync_delta; restart resumes without re-extracting."""
    cfg, _ = _make_test_config(tmp_path)
    assert cfg.corpus_root is not None
    assert cfg.db_path is not None

    (cfg.corpus_root / "01_turing.md").write_text(
        "# Alan Turing\nAlan Turing worked at Bletchley Park during World War II.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "02_lovelace.md").write_text(
        "# Ada Lovelace\nAda Lovelace programmed the Analytical Engine in London.\n",
        encoding="utf-8",
    )

    prog1, emb1, _, _ = _make_program_and_embedder()
    extracted_counts: dict[str, int] = {}
    orig_extract1 = prog1.extract_openie
    should_crash_on_second = [True]

    def _InstrumentedExtract(chunk_id: str, passage: str, **kw: Any) -> OpenIERecord:
        extracted_counts[chunk_id] = extracted_counts.get(chunk_id, 0) + 1
        if should_crash_on_second[0] and len(extracted_counts) == 2:
            raise RuntimeError("Simulated mid-batch Stage 1 OpenIE crash on chunk 2")
        return orig_extract1(chunk_id, passage, **kw)

    setattr(prog1, "extract_openie", _InstrumentedExtract)

    rt1 = YawRuntime(cfg, program=prog1, embedder=emb1)
    rt1.start(sync_on_start=False)
    try:
        with pytest.raises(YawRuntimeError) as exc_info:
            rt1.sync_now()
        assert exc_info.value.code == "MODEL_ERROR"
        assert len(extracted_counts) == 2
    finally:
        rt1.stop()

    # Verify chunk 1 was checkpointed to openie_chunks and pending_sync_delta was saved in yaw.db meta
    first_chunk_id = next(iter(extracted_counts.keys()))
    conn = open_yaw_db(cfg.db_path)
    try:
        saved_openie = get_openie_checkpoints(conn, list(extracted_counts.keys()))
        assert first_chunk_id in saved_openie
        assert get_meta(conn, "pending_sync_delta") is not None
    finally:
        conn.close()

    # Restart a new YawRuntime with the crash cleared: chunk 1 must NOT be re-extracted!
    should_crash_on_second[0] = False
    prog2, emb2, _, _ = _make_program_and_embedder()
    orig_extract2 = prog2.extract_openie

    def _InstrumentedExtract2(chunk_id: str, passage: str, **kw: Any) -> OpenIERecord:
        extracted_counts[chunk_id] = extracted_counts.get(chunk_id, 0) + 1
        return orig_extract2(chunk_id, passage, **kw)

    setattr(prog2, "extract_openie", _InstrumentedExtract2)

    rt2 = YawRuntime(cfg, program=prog2, embedder=emb2)
    rt2.start(sync_on_start=True)
    try:
        # Chunk 1 was extracted once in run 1 and 0 times in run 2 (count == 1)
        assert extracted_counts[first_chunk_id] == 1
        assert rt2.stats().chunks == 2
        assert rt2.engine is not None
        assert rt2.engine.get_meta("pending_sync_delta") is None

        hits = rt2.retrieve("Where did Alan Turing work?", top_k=2)
        assert len(hits) == 2
        assert any("Bletchley Park" in h.text for h in hits)
    finally:
        rt2.stop()


def test_inference_outage_degradation_and_recovery(tmp_path: Path) -> None:
    """D1-A5: Inference outage sets provider_ready=False and lifecycle='degraded' without crashing; recovery restores ready."""
    cfg, _ = _make_test_config(tmp_path)
    assert cfg.corpus_root is not None
    (cfg.corpus_root / "turing.md").write_text(
        "# Alan Turing\nAlan Turing worked at Bletchley Park.\n",
        encoding="utf-8",
    )

    prog, emb, lm, raw_emb = _make_program_and_embedder()
    rt = YawRuntime(cfg, program=prog, embedder=emb)
    rt.start(sync_on_start=True)
    try:
        assert rt.status().lifecycle == "ready"
        assert rt.status().provider_ready is True

        # Trigger inference outage on LM
        lm.fail_online = True
        with pytest.raises(YawRuntimeError) as exc_info:
            rt.rag_qa("Where did Alan Turing work?")
        assert exc_info.value.code == "MODEL_ERROR"

        degraded_status = rt.status()
        assert degraded_status.lifecycle == "degraded"
        assert degraded_status.provider_ready is False
        assert degraded_status.last_error is not None
        assert "unreachable" in degraded_status.last_error

        # Restore inference server: daemon recovers automatically on next request
        lm.fail_online = False
        raw_emb.fail_online = False
        qa_res = rt.rag_qa("Where did Alan Turing work?")
        assert qa_res["answer"] == "Bletchley Park"
        assert len(qa_res["evidence"]) >= 1

        recovered_status = rt.status()
        assert recovered_status.lifecycle == "ready"
        assert recovered_status.provider_ready is True
        assert recovered_status.last_error is None
    finally:
        rt.stop()


def test_optimize_and_reload_compiled_program_with_composite_fitness(tmp_path: Path) -> None:
    """D1-A4: Verify composite fitness formula (Recall@1..20 * EM * F1) / tokens and full-surface persistence."""
    # 1. Verify exact composite fitness calculation in compute_sizecoding_pareto_score
    pred_metric = compute_sizecoding_pareto_score(
        0.8,
        recall_at_1=0.5,
        recall_at_2=0.6,
        recall_at_10=1.0,
        recall_at_20=1.0,
        em=1.0,
        f1=0.9,
        tokens=50,
    )
    expected_quality = 0.5 * 0.6 * 0.8 * 1.0 * 1.0 * 1.0 * 0.9
    expected_pareto_raw = expected_quality / 50
    assert pytest.approx(pred_metric.quality_numerator) == expected_quality
    assert pytest.approx(pred_metric.pareto_raw) == expected_pareto_raw
    assert "em" in pred_metric.objective_scores
    assert "f1" in pred_metric.objective_scores
    assert "recall_at_1" in pred_metric.objective_scores
    assert "recall_at_20" in pred_metric.objective_scores

    # 2. Verify full-surface optimization and persistence via YawRuntime.optimize()
    cfg, _ = _make_test_config(tmp_path)
    assert cfg.corpus_root is not None
    (cfg.corpus_root / "turing.md").write_text(
        "# Alan Turing\nAlan Turing worked at Bletchley Park.\n",
        encoding="utf-8",
    )

    prog, emb, _, _ = _make_program_and_embedder()
    # Mutate full optimizable surface: instructions, EmbeddingGemma prefixes, and GBNF bounds
    prog.embedding_query_prefix = "task: custom optimized query | query: "
    prog.embedding_fact_prefix = "task: custom optimized fact | query: "
    prog.embedding_document_prefix = "title: custom | text: "
    prog.gbnf_max_entities = 6
    prog.gbnf_max_triples = 5
    prog.gbnf_max_facts = 3
    prog.gbnf_max_reasoning_chars = 160
    prog.ner.signature = prog.ner.signature.with_instructions("Optimized NER surface instruction.")

    class StubGEPA:
        def __init__(
            self,
            metric: Any,
            reflection_lm: Any = None,
            instruction_proposer: Any = None,
            auto: str = "light",
            track_stats: bool = False,
            track_best_outputs: bool = False,
            log_dir: str | None = None,
            **kwargs: Any,
        ) -> None:
            self.metric = metric
            self.reflection_lm = reflection_lm
            self.instruction_proposer = instruction_proposer
            self.auto = auto
            self.track_stats = track_stats
            self.track_best_outputs = track_best_outputs
            self.log_dir = log_dir

        def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
            assert self.reflection_lm is not None
            assert self.instruction_proposer is not None
            assert callable(self.instruction_proposer)
            assert self.track_stats is True
            assert self.track_best_outputs is True
            pred = student(question=trainset[0].question)
            score_pred = self.metric(trainset[0], pred, pred_name="qa")
            assert score_pred.score > 0.0
            return student

    orig_gepa = getattr(dspy, "GEPA", None)
    setattr(dspy, "GEPA", StubGEPA)
    try:
        rt1 = YawRuntime(cfg, program=prog, embedder=emb)
        rt1.start(sync_on_start=True)
        try:
            trainset = [
                dspy.Example(
                    question="Where did Alan Turing work?",
                    gold_answers=["Bletchley Park"],
                    gold_docs=["Alan Turing worked at Bletchley Park."],
                ).with_inputs("question")
            ]
            saved_path = rt1.optimize(trainset, mode="gepa")
            assert saved_path.exists()
        finally:
            rt1.stop()
    finally:
        if orig_gepa is not None:
            setattr(dspy, "GEPA", orig_gepa)

    # 3. Start a brand-new YawRuntime and verify compiled_program.json restores all 5 surfaces
    fresh_prog, fresh_emb, _, _ = _make_program_and_embedder()
    rt2 = YawRuntime(cfg, program=fresh_prog, embedder=fresh_emb)
    rt2.start(sync_on_start=False)
    try:
        assert rt2.engine is not None
        loaded_p = rt2.engine.program
        assert loaded_p.ner.signature.instructions == "Optimized NER surface instruction."
        assert loaded_p.gbnf_max_entities == 6
        assert loaded_p.gbnf_max_triples == 5
        assert loaded_p.gbnf_max_facts == 3
        assert loaded_p.gbnf_max_reasoning_chars == 160
        assert rt2.engine.embedder.profile.query_prefix == "task: custom optimized query | query: "
        assert (
            rt2.engine.embedder.profile.fact_query_prefix == "task: custom optimized fact | query: "
        )
        assert rt2.engine.embedder.profile.document_prefix == "title: custom | text: "
    finally:
        rt2.stop()


def test_cli_subcommands_sync_status_rebuild_and_stop(tmp_path: Path) -> None:
    """Verify CLI subcommands sync, status, rebuild, and stop with --config yaw.toml."""
    cfg, toml_path = _make_test_config(tmp_path)
    assert cfg.corpus_root is not None
    (cfg.corpus_root / "note.md").write_text(
        "# University of Southampton\nThe University of Southampton was founded in 1862.\n",
        encoding="utf-8",
    )

    prog, emb, _, _ = _make_program_and_embedder()

    # 1. CLI graph build (standalone incremental sync)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["graph", "build", "--config", str(toml_path)], program=prog, embedder=emb)
    assert rc == 0
    sync_out = json.loads(buf.getvalue())
    assert sync_out["added_chunks"] == 1
    assert sync_out["stats"]["chunks"] == 1

    # 2. CLI daemon status (offline inspection of yaw.db)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["daemon", "status", "--config", str(toml_path)])
    assert rc == 0
    status_out = json.loads(buf.getvalue())
    assert status_out["status"]["lifecycle"] == "stopped"
    assert status_out["stats"]["chunks"] == 1

    # 3. CLI graph rebuild
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["graph", "rebuild", "--config", str(toml_path)], program=prog, embedder=emb)
    assert rc == 0
    rebuild_out = json.loads(buf.getvalue())
    assert rebuild_out["rebuilt"] is True
    assert rebuild_out["stats"]["chunks"] == 1

    # 4. Foreground daemon run + CLI stop from another thread
    run_thread = threading.Thread(
        target=lambda: main(
            ["daemon", "run", "--config", str(toml_path)], program=prog, embedder=emb
        ),
        daemon=True,
    )
    run_thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        live = inspect_status(cfg)
        if live["status"]["lifecycle"] == "ready":
            break
        time.sleep(0.05)

    assert inspect_status(cfg)["status"]["lifecycle"] == "ready"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc_stop = main(["daemon", "stop", "--config", str(toml_path)])
    assert rc_stop == 0
    stop_out = json.loads(buf.getvalue())
    assert stop_out["stopped"] is True
    run_thread.join(timeout=5.0)
    assert not run_thread.is_alive()


def test_cli_optimize_ergonomics_and_anti_overfitting_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `yaw dspy optimize` CLI ergonomics: benchmark slice parsing, train/val split, telemetry paths, and Tier 2 anti-overfitting guard."""
    cfg, toml_path = _make_test_config(tmp_path)
    assert cfg.corpus_root is not None
    prog, emb, _, _ = _make_program_and_embedder()

    # Write a 6-query benchmark-format dataset JSON (with paragraphs + is_supporting)
    bench_dataset_path = tmp_path / "tier1_slice.json"
    bench_samples = [
        {
            "id": f"q-{i}",
            "question": f"Where did Scientist {i} work?",
            "answer": [f"Institute {i}"],
            "paragraphs": [
                {
                    "title": f"Scientist {i}",
                    "text": f"Scientist {i} worked at Institute {i}.",
                    "is_supporting": True,
                },
                {
                    "title": f"Distractor {i}",
                    "text": f"Distractor {i} is unrelated.",
                    "is_supporting": False,
                },
            ],
        }
        for i in range(6)
    ]
    bench_dataset_path.write_text(json.dumps(bench_samples), encoding="utf-8")

    captured_gepa: dict[str, Any] = {}
    opt_slice_file = cfg.corpus_root / ".yaw_optimize_slice.json"

    class CaptureGEPA:
        def __init__(self, metric: Any, **kwargs: Any) -> None:
            captured_gepa["metric"] = metric
            captured_gepa.update(kwargs)

        def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
            captured_gepa["trainset_len"] = len(trainset)
            captured_gepa["valset_len"] = len(valset) if valset is not None else 0
            # Verify .yaw_optimize_slice.json stays on disk with indent=2 during GEPA evolution
            assert opt_slice_file.exists()
            raw_slice_text = opt_slice_file.read_text(encoding="utf-8")
            assert "\n  {\n" in raw_slice_text
            pred = student(question=trainset[0].question)
            assert pred.context.strip() != ""
            proposer = captured_gepa.get("instruction_proposer")
            if proposer is not None and hasattr(proposer, "telemetry"):
                proposer.telemetry.append(
                    {
                        "component_name": "qa.predict",
                        "previous_instruction": "old",
                        "new_instruction": "new",
                        "num_feedback_examples": len(trainset),
                    }
                )
            return student

    monkeypatch.setattr(dspy, "GEPA", CaptureGEPA, raising=False)

    out_compiled = tmp_path / "custom_compiled.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "dspy",
                "optimize",
                "--config",
                str(toml_path),
                "--dataset",
                str(bench_dataset_path),
                "--mode",
                "gepa",
                "--max-queries",
                "6",
                "--train-queries",
                "4",
                "--auto",
                "light",
                "--output",
                str(out_compiled),
            ],
            program=prog,
            embedder=emb,
        )
    assert rc == 0
    assert opt_slice_file.exists()
    opt_out = json.loads(buf.getvalue())
    assert opt_out["optimized"] is True
    assert opt_out["mode"] == "gepa"
    assert opt_out["auto"] == "light"
    assert opt_out["train_queries"] == 4
    assert opt_out["val_queries"] == 2
    assert Path(opt_out["compiled_program_path"]).exists()
    assert Path(opt_out["proposer_telemetry_path"]).exists()
    assert opt_out["stats"]["chunks"] >= 6
    assert captured_gepa["trainset_len"] == 4
    assert captured_gepa["valset_len"] == 2
    assert captured_gepa["auto"] == "light"

    # Anti-Overfitting Rule: Reject full Tier 2 benchmark optimization (--max-queries > 50)
    err_buf = io.StringIO()
    with redirect_stderr(err_buf):
        rc_overfit = main(
            [
                "dspy",
                "optimize",
                "--config",
                str(toml_path),
                "--dataset",
                "2wikimultihopqa",
                "--max-queries",
                "100",
            ],
            program=prog,
            embedder=emb,
        )
    assert rc_overfit == 1
    err_payload = json.loads(err_buf.getvalue())
    assert err_payload["error"] == "INVALID_INPUT"


def test_cli_optimize_multi_dataset_interleaved_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-dataset optimization: array of datasets, interleaved train/val splits, and budget allocation."""
    _, toml_path = _make_test_config(tmp_path)
    prog, emb, _, _ = _make_program_and_embedder()

    # Create 3 synthetic datasets (each with 10 questions)
    ds_paths: list[Path] = []
    for d_name in ["alpha", "beta", "gamma"]:
        ds_file = tmp_path / f"ds_{d_name}.json"
        samples = [
            {
                "id": f"{d_name}-{i}",
                "question": f"Question {d_name} {i}?",
                "answer": [f"Answer {d_name} {i}"],
                "paragraphs": [
                    {
                        "title": f"Doc {d_name} {i}",
                        "text": f"Content for {d_name} {i}.",
                        "is_supporting": True,
                    }
                ],
            }
            for i in range(10)
        ]
        ds_file.write_text(json.dumps(samples), encoding="utf-8")
        ds_paths.append(ds_file)

    captured_gepa: dict[str, Any] = {}

    class CaptureGEPA:
        def __init__(self, metric: Any, **kwargs: Any) -> None:
            captured_gepa["metric"] = metric
            captured_gepa.update(kwargs)

        def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
            captured_gepa["trainset"] = trainset
            captured_gepa["valset"] = valset
            return student

    monkeypatch.setattr(dspy, "GEPA", CaptureGEPA, raising=False)

    out_compiled = tmp_path / "multi_compiled.json"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "dspy",
                "optimize",
                "--config",
                str(toml_path),
                "--dataset",
                str(ds_paths[0]),
                str(ds_paths[1]),
                str(ds_paths[2]),
                "--mode",
                "gepa",
                "--max-queries",
                "10",
                "--train-queries",
                "7",
                "--auto",
                "light",
                "--output",
                str(out_compiled),
            ],
            program=prog,
            embedder=emb,
        )
    assert rc == 0
    opt_out = json.loads(buf.getvalue())
    assert opt_out["optimized"] is True
    assert opt_out["auto"] == "light"
    # 3 datasets * 7 train queries = 21 training queries (satisfies DSPy 20 minimum!)
    assert opt_out["train_queries"] == 21
    # 3 datasets * 3 val queries = 9 validation queries
    assert opt_out["val_queries"] == 9
    assert len(opt_out["datasets"]) == 3

    trainset = captured_gepa["trainset"]
    assert len(trainset) == 21
    # Verify interleaving: first 3 examples come from alpha, beta, gamma respectively
    assert "alpha" in trainset[0].question
    assert "beta" in trainset[1].question
    assert "gamma" in trainset[2].question

    valset = captured_gepa["valset"]
    assert len(valset) == 9
    assert "alpha" in valset[0].question
    assert "beta" in valset[1].question
    assert "gamma" in valset[2].question


def test_cli_status_reports_active_indexing_progress(tmp_path: Path) -> None:
    """Verify yaw status inspects openie checkpoints and surfaces active indexing progress."""
    from yaw.core import OpenIERecord
    from yaw.daemon.cli import inspect_status
    from yaw.daemon.graph.storage import open_yaw_db, save_openie_checkpoints

    cfg, _ = _make_test_config(tmp_path)
    assert cfg.db_path is not None
    conn = open_yaw_db(cfg.db_path)
    try:
        # Seed 3 dummy openie records
        records = [
            OpenIERecord(chunk_id=f"chunk-{i}", entities=("E",), triples=(("A", "r", "B"),))
            for i in range(3)
        ]
        save_openie_checkpoints(conn, records)
    finally:
        conn.close()

    # Offline inspect
    offline_status = inspect_status(cfg)
    assert offline_status["stats"]["openie_checkpoints"] == 3
    assert offline_status["status"]["lifecycle"] == "stopped"

    # With daemon locked, inspect_status recognizes indexing in progress
    prog, emb, _, _ = _make_program_and_embedder()
    rt = YawRuntime(cfg, program=prog, embedder=emb)
    rt.start(sync_on_start=False)
    try:
        live_status = inspect_status(cfg)
        assert live_status["stats"]["openie_checkpoints"] == 3
        assert "indexing" in live_status["status"]["active_op"]
        assert live_status["status"]["sync_state"] == "syncing"
    finally:
        rt.stop()


def test_cli_config_get_and_set(tmp_path: Path) -> None:
    """Verify `yaw config get` and `yaw config set` read/write yaw.toml safely with validation."""
    _, toml_path = _make_test_config(tmp_path)

    # 1. Get specific key
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config", "get", "optimization.auto", "--config", str(toml_path)])
    assert rc == 0
    val_out = json.loads(buf.getvalue())
    assert val_out["optimization.auto"] == "light"

    # 2. Set valid key
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config", "set", "optimization.auto", "medium", "--config", str(toml_path)])
    assert rc == 0
    set_out = json.loads(buf.getvalue())
    assert set_out["updated"] is True
    assert set_out["key"] == "optimization.auto"
    assert set_out["value"] == "medium"

    # Verify get reflects updated value
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config", "get", "optimization.auto", "--config", str(toml_path)])
    assert rc == 0
    assert json.loads(buf.getvalue())["optimization.auto"] == "medium"

    # 3. Setting invalid value fails with INVALID_INPUT and preserves yaw.toml integrity
    err_buf = io.StringIO()
    with redirect_stderr(err_buf):
        rc_err = main(
            ["config", "set", "optimization.auto", "super_heavy", "--config", str(toml_path)]
        )
    assert rc_err == 1
    err_payload = json.loads(err_buf.getvalue())
    assert err_payload["error"] == "INVALID_INPUT"

    # File still has 'medium'
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["config", "get", "optimization.auto", "--config", str(toml_path)])
    assert rc == 0
    assert json.loads(buf.getvalue())["optimization.auto"] == "medium"


def test_cli_dspy_load_program_hot_reload_live_and_staged(tmp_path: Path) -> None:
    """Verify `yaw dspy load-program <file>` stages offline and hot-reloads live daemon in-memory."""
    cfg, toml_path = _make_test_config(tmp_path)
    prog, emb, _, _ = _make_program_and_embedder()

    # Create dummy candidate program JSON using prog.save()
    cand1 = tmp_path / "cand1.json"
    prog.embedding_query_prefix = "task: cand1 query | query: "
    prog.save(str(cand1), save_program=False)

    # 1. Offline promotion (daemon stopped)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["dspy", "load-program", str(cand1), "--config", str(toml_path)])
    assert rc == 0
    load_out = json.loads(buf.getvalue())
    assert load_out["reloaded"] is False
    assert load_out["staged"] is True
    assert cfg.compiled_program_path is not None
    assert Path(cfg.compiled_program_path).exists()

    # 2. Boot daemon: verifies it loads staged program
    rt = YawRuntime(cfg, program=HippoRAGProgram.from_config(cfg), embedder=emb)
    rt.start(sync_on_start=False)
    try:
        assert rt.engine is not None
        assert rt.engine.embedder.profile.query_prefix == "task: cand1 query | query: "

        # Create cand2 with different instruction/prefix
        cand2 = tmp_path / "cand2.json"
        prog.embedding_query_prefix = "task: cand2 hot reloaded | query: "
        prog.save(str(cand2), save_program=False)

        # 3. Live hot-reload via CLI while daemon is running
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = main(["dspy", "load-program", str(cand2), "--config", str(toml_path)])
        assert rc2 == 0
        load_out2 = json.loads(buf2.getvalue())
        assert load_out2["reloaded"] is True
        assert load_out2["pid"] == os.getpid()

        # Verify engine reloaded in-memory without restarting
        assert rt.engine.embedder.profile.query_prefix == "task: cand2 hot reloaded | query: "
    finally:
        rt.stop()


def test_cli_dspy_optimize_sandbox_isolation_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `yaw dspy optimize` runs in an isolated sandbox concurrently with production daemon."""
    from yaw.daemon.cli import is_daemon_locked

    cfg, toml_path = _make_test_config(tmp_path)
    prog, emb, _, _ = _make_program_and_embedder()

    # Start persistent production daemon holding yaw.db.lock
    prod_rt = YawRuntime(cfg, program=prog, embedder=emb)
    prod_rt.start(sync_on_start=False)
    try:
        assert is_daemon_locked(cfg) is True

        # Write small synthetic dataset
        bench_ds = tmp_path / "bench.json"
        bench_ds.write_text(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "question": "Question 1?",
                        "answer": ["Answer 1"],
                        "paragraphs": [
                            {"title": "Doc 1", "text": "Content 1.", "is_supporting": True}
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

        class FastGEPA:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def compile(self, student: Any, *args: Any, **kwargs: Any) -> Any:
                return student

        monkeypatch.setattr(dspy, "GEPA", FastGEPA, raising=False)
        monkeypatch.setattr(dspy, "MIPROv2", FastGEPA, raising=False)

        # Run optimization concurrently while production daemon holds yaw.db.lock
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(
                [
                    "dspy",
                    "optimize",
                    "--config",
                    str(toml_path),
                    "--dataset",
                    str(bench_ds),
                    "--max-queries",
                    "1",
                    "--train-queries",
                    "1",
                    "--auto",
                    "light",
                ],
                program=prog,
                embedder=emb,
            )
        assert rc == 0
        opt_out = json.loads(buf.getvalue())
        assert opt_out["optimized"] is True
        assert "yaw_optimize_" in opt_out["sandbox_db_path"]
        assert Path(opt_out["compiled_program_path"]).exists()

        # Production database was NOT touched or overwritten
        assert cfg.db_path is not None
        assert Path(cfg.db_path).exists()
        # Production compiled_program.json was NOT overwritten by random optimization run
        assert cfg.compiled_program_path is not None
        assert not Path(cfg.compiled_program_path).exists()
    finally:
        prod_rt.stop()


def test_cli_graph_lint_and_turtle_over_read_only_snapshot_while_daemon_locked(
    tmp_path: Path,
) -> None:
    """Verify `yaw graph lint [--turtle]` runs over a read-only WAL snapshot while daemon holds yaw.db.lock."""
    from yaw.daemon.cli import is_daemon_locked

    cfg, toml_path = _make_test_config(tmp_path)
    prog, emb, _, _ = _make_program_and_embedder()

    prod_rt = YawRuntime(cfg, program=prog, embedder=emb)
    prod_rt.start(sync_on_start=True)
    try:
        assert is_daemon_locked(cfg) is True

        # 1. JSON SHACL report over read-only WAL snapshot while daemon holds yaw.db.lock
        buf_json = io.StringIO()
        with redirect_stdout(buf_json):
            rc_json = main(["graph", "lint", "--config", str(toml_path)])
        assert rc_json == 0
        report = json.loads(buf_json.getvalue())
        assert report["conforms"] is True
        assert report["violation_count"] == 0

        # 2. W3C Turtle report over read-only WAL snapshot while daemon holds yaw.db.lock
        buf_ttl = io.StringIO()
        with redirect_stdout(buf_ttl):
            rc_ttl = main(["graph", "lint", "--config", str(toml_path), "--turtle"])
        assert rc_ttl == 0
        ttl_out = buf_ttl.getvalue()
        assert "@prefix skos: <http://www.w3.org/2004/02/skos/core#> ." in ttl_out
        assert "sh:conforms true" in ttl_out
    finally:
        prod_rt.stop()
