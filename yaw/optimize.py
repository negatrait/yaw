"""Offline DSPy GEPA + bare dspy.ReAct and Optuna hyperparameter optimization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import string
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import optuna
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback
from filelock import FileLock, Timeout
from numpy.typing import NDArray

from yaw.core import (
    YawConfig,
    YawRuntimeError,
    format_query_for_embedding,
    parse_dataset_list,
    resolve_coco_state_root,
    write_text_atomic,
)
from yaw.datasets import (
    DATASET_REGISTRY,
    DATASET_SOURCE_FIELDS,
    DEFAULT_PARITY_DATASETS,
    TIER2_CORPORA,
    DatasetBundle,
    build_benchmark_slice,
    doc_matches_gold,
    document_identity,
    document_text,
    load_dataset_bundle,
    load_dataset_files,
    resolve_dataset_bundle,
    resolve_dataset_name,
)
from yaw.pipeline import close_pipeline, make_dspy_embed_fn, start_pipeline, update_pipeline_once
from yaw.qa import build_qa_agent, iter_tool_observations, make_qa_tools
from yaw.retrieval import build_vector_search, hydrate_ranked_passages, open_readonly_db, search_vectors

_NORM_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_SPAN_HEADER_RE = re.compile(
    r"\[(?:\d+\]\s*\[)?(?P<path>[^\]|:\s]+?\.[A-Za-z0-9]+):(?P<s1>\d+)-(?P<s2>\d+)(?:@\d+-\d+)?(?:\s*\|[^\]]*)?\]"
)
_ABSTAIN_PHRASES = ("insufficient evidence", "cannot determine", "not enough information", "no evidence")
_INLINE_CITATION_RE = re.compile(r"\[chunk_id=(\d+)\]")
_OBSERVED_CITATION_RE = re.compile(r"^\[chunk_id=(\d+)\].*score=", re.MULTILINE)


def normalize_qa_answer(text: str) -> str:
    """SQuAD / HotpotQA standard answer normalization."""
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_art = _NORM_ARTICLES_RE.sub(" ", no_punc)
    return " ".join(no_art.split())


def compute_em(prediction: str, gold_answers: Sequence[str]) -> float:
    """Maximum normalized Exact Match across reference answers."""
    if not gold_answers:
        return 0.0
    norm_pred = normalize_qa_answer(prediction)
    return 1.0 if any(norm_pred == normalize_qa_answer(g) for g in gold_answers if g) else 0.0


def compute_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Maximum token-level F1 score across reference answers."""
    if not gold_answers:
        return 0.0
    pred_toks = normalize_qa_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_toks = normalize_qa_answer(gold).split()
        if not pred_toks and not gold_toks:
            return 1.0
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_toks)
        rec = num_same / len(gold_toks)
        best = max(best, 2.0 * prec * rec / (prec + rec))
    return best


def compute_recall_at_k(
    retrieved_docs: Sequence[str],
    gold_docs: Sequence[str] | None,
    k: int,
) -> float | None:
    """Fraction of gold documents matched within the top-k hydrated passages."""
    if gold_docs is None:
        return None
    if not gold_docs:
        return 1.0
    top = retrieved_docs[: max(0, k)]
    hits = sum(1 for gd in gold_docs if any(doc_matches_gold(rd, gd) for rd in top))
    return float(hits) / float(len(gold_docs))


def compute_ndcg_at_k(
    retrieved_docs: Sequence[str],
    gold_docs: Sequence[str] | None,
    k: int,
) -> float | None:
    """Binary-relevance Normalized Discounted Cumulative Gain at rank k."""
    if gold_docs is None or not gold_docs or k <= 0:
        return None
    remaining = list(gold_docs)
    dcg = 0.0
    for rank, rd in enumerate(retrieved_docs[:k], start=1):
        matched_idx = next((i for i, gd in enumerate(remaining) if doc_matches_gold(rd, gd)), None)
        if matched_idx is not None:
            remaining.pop(matched_idx)
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(len(gold_docs), k)
    idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def _collect_tool_evidence(pred: Any, pred_trace: DSPyTrace | None) -> tuple[list[str], int]:
    """Extract observed passage blocks and tool step count from ReAct trajectory or DSPy trace."""
    blocks: list[str] = []
    tool_calls = 0
    history = getattr(pred, "history", None)
    if history is not None:
        messages = history.get("messages", ()) if isinstance(history, Mapping) else history.messages
        blocks = list(iter_tool_observations(messages))
        return blocks, len(blocks)
    traj = getattr(pred, "trajectory", None)
    if isinstance(traj, Mapping):
        idx = 0
        while f"tool_name_{idx}" in traj or f"observation_{idx}" in traj:
            t_name = str(traj.get(f"tool_name_{idx}", ""))
            if t_name and t_name != "finish":
                tool_calls += 1
            obs = str(traj.get(f"observation_{idx}", "")).strip()
            if obs:
                blocks.append(obs)
            idx += 1
    if not blocks and pred_trace:
        for _, inputs, outputs in pred_trace:
            if isinstance(inputs, Mapping) and "trajectory" in inputs and isinstance(inputs["trajectory"], str):
                blocks.append(inputs["trajectory"])
            out_t = getattr(outputs, "trajectory", None)
            if isinstance(out_t, Mapping):
                for key, val in out_t.items():
                    if str(key).startswith("observation_") and val:
                        blocks.append(str(val))
    return blocks, tool_calls


def qa_gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: DSPyTrace | None = None,
    pred_name: str | None = None,
    pred_trace: DSPyTrace | None = None,
) -> ScoreWithFeedback:
    """5-argument GEPA metric returning ScoreWithFeedback with diagnostic grounding critique."""
    _ = (trace, pred_name)
    raw_ans = str(getattr(pred, "answer", "") or "").strip()
    raw_cites = getattr(pred, "citations", None) or []
    cites = [str(c).strip() for c in raw_cites if str(c).strip()] if isinstance(raw_cites, list) else []

    gold_answers_raw = getattr(gold, "gold_answers", None)
    if gold_answers_raw is None:
        single = getattr(gold, "answer", "")
        gold_answers = [str(single)] if single else []
    else:
        gold_answers = [str(a) for a in gold_answers_raw if str(a)]
    gold_docs_raw = getattr(gold, "gold_docs", None)
    gold_docs = [str(d) for d in gold_docs_raw] if isinstance(gold_docs_raw, list) else None

    answer = _INLINE_CITATION_RE.sub("", raw_ans).strip()
    f1 = compute_f1(answer, gold_answers)
    em = compute_em(answer, gold_answers)
    obs_blocks, tool_calls = _collect_tool_evidence(pred, pred_trace)
    seen_spans = {
        f"{m.group('path')}:{m.group('s1')}-{m.group('s2')}"
        for blk in obs_blocks
        for m in _SPAN_HEADER_RE.finditer(blk)
    }

    abstained = any(p in raw_ans.lower() for p in _ABSTAIN_PHRASES)
    valid_cites = [
        c
        for c in cites
        if any(c == sp or c.startswith(f"{sp}@") or sp.startswith(c.split("@", 1)[0]) for sp in seen_spans)
    ]
    if inline_cites := list(dict.fromkeys(_INLINE_CITATION_RE.findall(raw_ans))):
        observed_ids = {cid for block in obs_blocks for cid in _OBSERVED_CITATION_RE.findall(block)}
        cites = inline_cites
        valid_cites = [cid for cid in cites if cid in observed_ids]
    if abstained:
        cite_score = 1.0 if not cites else 0.5
    elif not cites:
        cite_score = 0.0
    else:
        cite_score = len(valid_cites) / len(cites)

    ev_recall = compute_recall_at_k(obs_blocks, gold_docs, max(len(obs_blocks), 1)) if gold_docs else None
    if ev_recall is not None:
        score = 0.55 * f1 + 0.15 * em + 0.15 * cite_score + 0.15 * ev_recall
    else:
        score = 0.65 * f1 + 0.20 * em + 0.15 * cite_score

    notes: list[str] = [f"F1={f1:.3f}, EM={em:.1f}, citation_grounding={cite_score:.2f}, tool_calls={tool_calls}."]
    if f1 < 0.8 and gold_answers:
        notes.append(f"Expected concise answer matching {gold_answers[0]!r}, got {raw_ans!r}.")
    if not abstained and cite_score < 1.0:
        notes.append("Cite passages observed in tool outputs using their inline chunk IDs.")
    if ev_recall is not None and ev_recall < 1.0:
        notes.append(
            f"Evidence coverage={ev_recall:.2f}; issue focused follow-up `search_passages` or `expand_neighbors` calls."
        )
    return ScoreWithFeedback(score=float(score), feedback=" ".join(notes))


class ProposeInstruction(dspy.Signature):
    """Refine a DSPy predictor instruction using execution traces and diagnostic feedback."""

    current_instruction: str = dspy.InputField(desc="Current system instruction for the target predictor.")
    reflective_dataset_json: str = dspy.InputField(desc="JSON traces with inputs, outputs, and diagnostic feedback.")
    improved_instruction: str = dspy.OutputField(desc="Improved concise instruction addressing failure modes.")


def make_react_instruction_proposer(reflection_lm: dspy.LM) -> Callable[..., dict[str, str]]:
    """Build a bare dspy.ReAct GEPA instruction proposer with tools for inspecting reflective traces."""

    def propose_component(current: str, rows: Sequence[Mapping[str, Any]]) -> str:
        def count_examples() -> str:
            """Return the number of reflective examples available for this component."""
            return f"{len(rows)} reflective trace(s) available."

        def read_failure_case(index: int = 0) -> str:
            """Read a specific reflective trace entry by zero-based index."""
            if not rows:
                return "No reflective traces recorded."
            clamped = max(0, min(int(index), len(rows) - 1))
            return json.dumps(rows[clamped], ensure_ascii=False, default=str)[:3000]

        proposer = dspy.ReAct(ProposeInstruction, tools=[count_examples, read_failure_case], max_iters=3)
        summary = json.dumps(rows[:6], ensure_ascii=False, default=str)[:6000]
        with dspy.context(lm=reflection_lm):
            res = proposer(current_instruction=current, reflective_dataset_json=summary)
        return str(getattr(res, "improved_instruction", "") or "").strip() or current

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: Sequence[str],
    ) -> dict[str, str]:
        return {
            comp: propose_component(candidate.get(comp, ""), reflective_dataset.get(comp, ()))
            for comp in components_to_update
        }

    return propose


@dataclass(frozen=True, slots=True)
class _DatasetSpecRef:
    canonical: str
    bundle: DatasetBundle | None


def _resolve_dataset_refs(ws: Path, names: Sequence[str]) -> list[_DatasetSpecRef]:
    out: list[_DatasetSpecRef] = []
    for raw in names:
        path = Path(raw)
        if path.is_dir() or path.is_file():
            bundle = resolve_dataset_bundle(path)
            out.append(_DatasetSpecRef(bundle.dataset, bundle))
            continue
        canonical = resolve_dataset_name(raw)
        root = ws / "outputs" / "datasets" / canonical
        bundle = load_dataset_bundle(root) if (root / "manifest.json").is_file() else None
        out.append(_DatasetSpecRef(canonical, bundle))
    return out


def _load_slice(
    ws: Path, ref: _DatasetSpecRef, *, max_queries: int, query_offset: int, index_queries: int | None = None
) -> tuple[list[str], list[str], list[list[str]] | None, list[list[str]], Sequence[Mapping[str, Any]], str, str]:
    if ref.bundle is not None:
        samples, doc_rows = ref.bundle.read_inputs()
        corpus = [dict(r, idx=d.document_id) for d, r in zip(ref.bundle.documents, doc_rows, strict=True)]
        t_key, b_key = ref.bundle.title_key, ref.bundle.body_key
    else:
        raw_dir = ws / "data" / "benchmarks"
        q_file, c_file = DATASET_REGISTRY[ref.canonical]
        if not ((raw_dir / q_file).is_file() and (raw_dir / c_file).is_file()):
            raise YawRuntimeError(
                "INVALID_INPUT", f"Dataset {ref.canonical!r} not found under outputs/datasets or {raw_dir}."
            )
        samples, corpus = load_dataset_files(ref.canonical, dataset_dir=raw_dir, project_root=ws)
        fields = DATASET_SOURCE_FIELDS.get(ref.canonical, {})
        t_key, b_key = str(fields.get("title_key", "title")), str(fields.get("body_key", "text"))

    doc_ids, questions, gold_docs, gold_answers = build_benchmark_slice(
        ref.canonical, samples, corpus, max_queries=max_queries, query_offset=query_offset, index_queries=index_queries
    )
    if len(questions) < 2:
        raise YawRuntimeError("INVALID_INPUT", f"Dataset {ref.canonical!r} yielded < 2 questions.")
    return doc_ids, questions, gold_docs, gold_answers, corpus, t_key, b_key


def _guard_optimization_slice(refs: Sequence[_DatasetSpecRef], offset: int) -> None:
    if offset < 10 and any(ref.canonical in TIER2_CORPORA or ref.canonical in DEFAULT_PARITY_DATASETS for ref in refs):
        raise YawRuntimeError("INVALID_INPUT", "query_offset < 10 overlaps held-out evaluation.")


def _stage_slice_documents(
    stage_dir: Path, slices: Sequence[tuple[str, Sequence[str], Sequence[Mapping[str, Any]], str, str]]
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    desired: set[Path] = set()
    for canonical, doc_ids, corpus, title_key, body_key in slices:
        by_id = {document_identity(row): row for row in corpus}
        for did in doc_ids:
            if (row := by_id.get(did)) is None:
                continue
            rendered = document_text(row, title_key, body_key)
            digest = hashlib.sha256(f"{canonical}\0{did}".encode()).hexdigest()[:16]
            target = (stage_dir / f"{canonical}_{digest}.txt").resolve()
            desired.add(target)
            if not target.exists() or target.read_bytes() != rendered.encode("utf-8"):
                write_text_atomic(target, rendered, newline="")
    for stale in stage_dir.iterdir():
        if stale.is_file() and stale.resolve() not in desired:
            stale.unlink(missing_ok=True)


def _unlink_sqlite_artifacts(db_file: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{db_file}{suffix}").unlink(missing_ok=True)


def _index_staged_slices(
    cfg: YawConfig,
    stage_dir: Path,
    db_file: Path,
    embed_fn: Callable[[list[str]], NDArray[np.float32]],
    slices: Sequence[tuple[str, Sequence[str], Sequence[Mapping[str, Any]], str, str]],
    *,
    fresh: bool,
    extra_wipe: Sequence[Path] = (),
) -> None:
    if fresh:
        if cfg.state_dir is not None and resolve_coco_state_root(cfg.state_dir).exists():
            drop_session = start_pipeline(
                replace(cfg, corpus_root=stage_dir), live=False, embed_fn=embed_fn, vector_dim=1
            )
            try:
                drop_session.app.drop_blocking()
            finally:
                close_pipeline(drop_session)
        _unlink_sqlite_artifacts(db_file)
        for p in (stage_dir, *extra_wipe):
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink(missing_ok=True)

    _stage_slice_documents(stage_dir, slices)
    idx_cfg = replace(cfg, db_path=db_file, corpus_root=stage_dir)
    session = start_pipeline(idx_cfg, embed_fn=embed_fn, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)


def _make_lm(cfg: YawConfig, *, temperature: float) -> dspy.LM:
    return dspy.LM(
        cfg.qa_model if "/" in cfg.qa_model else f"{cfg.provider}/{cfg.qa_model}",
        model_type="chat",
        api_base=cfg.base_url,
        api_key=cfg.api_key or "yaw",
        temperature=temperature,
        max_tokens=cfg.max_output_tokens,
        num_retries=3,
        cache=False,
    )


def optimize_optuna(
    cfg: YawConfig,
    *,
    datasets: str | None = None,
    trials: int | None = None,
    max_queries: int | None = None,
    query_offset: int | None = None,
    fresh: bool = False,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
) -> dict[str, Any]:
    """Tune retrieval hyperparameters (hybrid_alpha, ppr_alpha, seed_top_k) via Optuna TPE."""
    opt_cfg = cfg.for_optimization(".yaw/optuna_index.db")
    assert opt_cfg.db_path is not None and opt_cfg.state_dir is not None
    ws = opt_cfg.workspace_root
    opt_dir = opt_cfg.db_path.parent
    opt_dir.mkdir(parents=True, exist_ok=True)
    study_db = opt_dir / "optuna_study.db"
    status_path = opt_dir / "optuna_status.json"
    params_path = opt_dir / "optuna_best_params.json"
    stage_dir = opt_dir / "optuna_corpus"

    ds_names = parse_dataset_list(datasets) if datasets is not None else list(opt_cfg.optimize_datasets)
    refs = _resolve_dataset_refs(ws, ds_names)
    eff_max_q = int(max_queries if max_queries is not None else opt_cfg.optimize_max_queries)
    eff_offset = int(query_offset if query_offset is not None else opt_cfg.optimize_query_offset)
    _guard_optimization_slice(refs, eff_offset)
    eff_trials = max(1, int(trials if trials is not None else 64))

    loaded = [_load_slice(ws, r, max_queries=eff_max_q, query_offset=eff_offset) for r in refs]
    staged = [
        (ref.canonical, doc_ids, corpus, title_key, body_key)
        for ref, (doc_ids, _, _, _, corpus, title_key, body_key) in zip(refs, loaded, strict=True)
    ]
    active_embed = embed_fn or make_dspy_embed_fn(opt_cfg)

    write_text_atomic(
        status_path,
        json.dumps({"state": "running", "pid": os.getpid(), "started_at": time.time()}, indent=2) + "\n",
    )
    _index_staged_slices(
        opt_cfg,
        stage_dir,
        opt_cfg.db_path,
        active_embed,
        staged,
        fresh=fresh,
        extra_wipe=(study_db, params_path),
    )

    conn = open_readonly_db(opt_cfg.db_path)
    try:
        vs = build_vector_search(conn)
        eval_items: list[tuple[str, NDArray[np.float32], list[str]]] = []
        for _, questions, gold_docs, _, _, _, _ in loaded:
            if not gold_docs:
                continue
            formatted_qs = [format_query_for_embedding(q, task_mode=opt_cfg.embedding_task_mode) for q in questions]
            q_vecs = np.asarray(active_embed(formatted_qs), dtype=np.float32)
            for idx, (q, gd) in enumerate(zip(questions, gold_docs, strict=True)):
                if gd:
                    eval_items.append((q, q_vecs[idx], gd))
        if not eval_items:
            raise YawRuntimeError("INVALID_INPUT", "Selected Optuna datasets contain no gold retrieval documents.")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage_url = f"sqlite:///{study_db.resolve().as_posix()}"
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name="yaw_retrieval_ppr",
            storage=storage_url,
            direction="maximize",
            load_if_exists=not fresh,
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            h_alpha = trial.suggest_float("hybrid_alpha", 0.1, 0.95)
            p_alpha = trial.suggest_float("ppr_alpha", 0.0, 0.95)
            s_top_k = trial.suggest_int("seed_top_k", 5, 40)
            scores: list[float] = []
            for q_text, q_vec, gd in eval_items:
                ranked = search_vectors(
                    vs,
                    q_vec,
                    query_text=q_text,
                    conn=conn,
                    top_k=10,
                    seed_top_k=s_top_k,
                    hybrid_alpha=h_alpha,
                    ppr_alpha=p_alpha,
                )
                hydrated = hydrate_ranked_passages(vs, ranked, stage_dir, top_k=5)
                texts = [hp.text for hp in hydrated if not hp.stale_on_disk]
                r_k = compute_recall_at_k(texts, gd, 5) or 0.0
                n_k = compute_ndcg_at_k(texts, gd, 5) or 0.0
                scores.append(0.6 * r_k + 0.4 * n_k)
            return float(np.mean(scores)) if scores else 0.0

        study.optimize(objective, n_trials=eff_trials)
        payload = {
            "best_score": float(study.best_value),
            "best_params": dict(study.best_params),
            "total_trials": len(study.trials),
            "datasets": [r.canonical for r in refs],
        }
        write_text_atomic(params_path, json.dumps(payload, indent=2) + "\n")
        write_text_atomic(
            status_path,
            json.dumps({"state": "completed", "completed_at": time.time(), **payload}, indent=2) + "\n",
        )
        return payload
    finally:
        conn.close()


def optimize_dspy(
    cfg: YawConfig,
    *,
    datasets: str | None = None,
    auto: str | None = None,
    max_metric_calls: int | None = None,
    max_queries: int | None = None,
    query_offset: int | None = None,
    threads: int | None = None,
    fresh: bool = False,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    student_lm: Any = None,
    reflection_lm: Any = None,
    instruction_proposer: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compile the bare dspy.ReAct QA agent with dspy.GEPA and persist program + audit artifacts."""
    if auto is not None and max_metric_calls is not None:
        raise YawRuntimeError("INVALID_INPUT", "auto and max_metric_calls are mutually exclusive.")
    if threads is not None and threads < 1:
        raise YawRuntimeError("INVALID_INPUT", "threads must be positive.")
    opt_cfg = cfg.for_optimization()
    assert opt_cfg.lock_path is not None and opt_cfg.db_path is not None
    opt_cfg.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(opt_cfg.lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise YawRuntimeError("BUSY", f"Optimization lock is already held: {opt_cfg.lock_path}") from exc
    try:
        return _compile_dspy(
            opt_cfg,
            datasets=datasets,
            auto=auto,
            max_metric_calls=max_metric_calls,
            max_queries=max_queries,
            query_offset=query_offset,
            threads=threads,
            fresh=fresh,
            embed_fn=embed_fn,
            student_lm=student_lm,
            reflection_lm=reflection_lm,
            instruction_proposer=instruction_proposer,
        )
    except KeyboardInterrupt:
        write_text_atomic(
            opt_cfg.db_path.parent / "dspy_status.json",
            json.dumps({"state": "interrupted", "pid": None}, indent=2) + "\n",
        )
        raise
    finally:
        lock.release()


def _compile_dspy(
    opt_cfg: YawConfig,
    *,
    datasets: str | None,
    auto: str | None,
    max_metric_calls: int | None,
    max_queries: int | None,
    query_offset: int | None,
    threads: int | None,
    fresh: bool,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None,
    student_lm: Any,
    reflection_lm: Any,
    instruction_proposer: Callable[..., dict[str, str]] | None,
) -> dict[str, Any]:
    assert opt_cfg.db_path is not None and opt_cfg.state_dir is not None
    ws = opt_cfg.workspace_root
    opt_dir = opt_cfg.db_path.parent
    opt_dir.mkdir(parents=True, exist_ok=True)
    prog_path = opt_dir / "qa_program.json"
    status_path = opt_dir / "dspy_status.json"
    audit_path = opt_dir / "dspy_gepa_audit.json"
    telem_dir = opt_dir / "gepa_logs"
    stage_dir = opt_dir / "dspy_corpus"

    ds_names = parse_dataset_list(datasets) if datasets is not None else list(opt_cfg.optimize_datasets)
    refs = _resolve_dataset_refs(ws, ds_names)
    eff_max_q = int(max_queries if max_queries is not None else opt_cfg.optimize_max_queries)
    eff_offset = int(query_offset if query_offset is not None else opt_cfg.optimize_query_offset)
    eff_threads = threads if threads is not None else 1
    _guard_optimization_slice(refs, eff_offset)

    loaded = [_load_slice(ws, r, max_queries=eff_max_q, query_offset=eff_offset) for r in refs]
    staged = [
        (ref.canonical, doc_ids, corpus, title_key, body_key)
        for ref, (doc_ids, _, _, _, corpus, title_key, body_key) in zip(refs, loaded, strict=True)
    ]
    active_embed = embed_fn or make_dspy_embed_fn(opt_cfg)

    write_text_atomic(
        status_path,
        json.dumps({"state": "running", "pid": os.getpid(), "started_at": time.time()}, indent=2) + "\n",
    )
    _index_staged_slices(
        opt_cfg,
        stage_dir,
        opt_cfg.db_path,
        active_embed,
        staged,
        fresh=fresh,
        extra_wipe=(prog_path, audit_path, telem_dir),
    )
    telem_dir.mkdir(parents=True, exist_ok=True)

    trainset: list[dspy.Example] = []
    valset: list[dspy.Example] = []
    for _, questions, gold_docs, gold_answers, _, _, _ in loaded:
        examples: list[dspy.Example] = []
        for idx, q in enumerate(questions):
            ga = gold_answers[idx] if idx < len(gold_answers) else []
            gd = gold_docs[idx] if gold_docs is not None and idx < len(gold_docs) else None
            ex = dspy.Example(
                question=q,
                answer=ga[0] if ga else "",
                gold_answers=ga,
                gold_docs=gd,
            ).with_inputs("question")
            examples.append(ex)
        split_at = max(1, int(round(len(examples) * 0.7)))
        if split_at >= len(examples):
            split_at = len(examples) - 1
        trainset.extend(examples[:split_at])
        valset.extend(examples[split_at:])

    s_lm = student_lm or _make_lm(opt_cfg, temperature=1.0)
    r_lm = reflection_lm or _make_lm(opt_cfg, temperature=0.7)
    proposer = instruction_proposer or make_react_instruction_proposer(r_lm)

    gepa_kwargs: dict[str, Any] = {
        "metric": qa_gepa_metric,
        "reflection_lm": r_lm,
        "instruction_proposer": proposer,
        "num_threads": eff_threads,
        "log_dir": str(telem_dir),
        "track_stats": True,
        "seed": 42,
    }
    if max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = int(max_metric_calls)
    else:
        eff_auto = auto if auto is not None else opt_cfg.optimize_auto
        gepa_kwargs["auto"] = eff_auto

    conn = open_readonly_db(opt_cfg.db_path)
    try:
        vs = build_vector_search(conn)
        tools = make_qa_tools(vs, stage_dir, active_embed, conn, task_mode=opt_cfg.embedding_task_mode)
        student = build_qa_agent(tools, max_iters=opt_cfg.react_max_iters)
        optimizer = dspy.GEPA(**gepa_kwargs)
        with dspy.context(lm=s_lm):
            compiled = optimizer.compile(student, trainset=trainset, valset=valset)

        compiled.save(str(prog_path))
        res_stats = getattr(compiled, "detailed_results", None)
        val_scores = list(getattr(res_stats, "val_aggregate_scores", ())) if res_stats is not None else []
        best_score = float(max(val_scores)) if val_scores else 0.0
        audit = {
            "program_path": str(prog_path),
            "datasets": [r.canonical for r in refs],
            "train_size": len(trainset),
            "val_size": len(valset),
            "num_candidates": len(val_scores),
            "best_val_score": best_score,
            "val_aggregate_scores": [float(s) for s in val_scores],
        }
        write_text_atomic(audit_path, json.dumps(audit, indent=2) + "\n")
        write_text_atomic(
            status_path,
            json.dumps({"state": "completed", "completed_at": time.time(), **audit}, indent=2) + "\n",
        )
        return audit
    finally:
        conn.close()


def evaluate_optimization(
    cfg: YawConfig,
    *,
    dataset: str = "2wikimultihopqa",
    max_queries: int = 20,
    query_offset: int = 1000,
    fresh: bool = False,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    lm: Any = None,
) -> dict[str, Any]:
    """Evaluate baseline vs compiled QA program on a held-out benchmark slice."""
    opt_cfg = cfg.for_optimization()
    assert opt_cfg.db_path is not None
    ws = opt_cfg.workspace_root
    opt_dir = opt_cfg.db_path.parent
    prog_path = opt_dir / "qa_program.json"
    if not prog_path.is_file():
        raise YawRuntimeError("OPTIMIZATION_NOT_RUN", f"Compiled program missing at {prog_path}")

    eval_db = opt_dir / "eval_index.db"
    eval_state = opt_dir / "eval_state"
    eval_stage = opt_dir / "eval_corpus"
    eval_out = opt_dir / "eval_held_out.json"
    eval_cfg = replace(opt_cfg, db_path=eval_db, state_dir=eval_state, corpus_root=eval_stage)

    ref = _resolve_dataset_refs(ws, [dataset])[0]
    doc_ids, questions, gold_docs, gold_answers, corpus, t_key, b_key = _load_slice(
        ws, ref, max_queries=max_queries, query_offset=query_offset
    )
    active_embed = embed_fn or make_dspy_embed_fn(eval_cfg)
    _index_staged_slices(
        eval_cfg,
        eval_stage,
        eval_db,
        active_embed,
        [(ref.canonical, doc_ids, corpus, t_key, b_key)],
        fresh=fresh,
        extra_wipe=(eval_out,),
    )

    active_lm = lm or _make_lm(eval_cfg, temperature=1.0)
    conn = open_readonly_db(eval_db)
    try:
        vs = build_vector_search(conn)
        tools = make_qa_tools(vs, eval_stage, active_embed, conn, task_mode=eval_cfg.embedding_task_mode)
        baseline_agent = build_qa_agent(tools, max_iters=eval_cfg.react_max_iters)
        compiled_agent = build_qa_agent(tools, artifact_path=prog_path, max_iters=eval_cfg.react_max_iters)

        def _score_agent(agent: dspy.Module) -> dict[str, float]:
            ems: list[float] = []
            f1s: list[float] = []
            comp: list[float] = []
            with dspy.context(lm=active_lm):
                for idx, q in enumerate(questions):
                    ga = gold_answers[idx] if idx < len(gold_answers) else []
                    gd = gold_docs[idx] if gold_docs is not None and idx < len(gold_docs) else None
                    ex = dspy.Example(question=q, gold_answers=ga, gold_docs=gd).with_inputs("question")
                    pred = agent(question=q)
                    ans = str(getattr(pred, "answer", "") or "")
                    ems.append(compute_em(ans, ga))
                    f1s.append(compute_f1(ans, ga))
                    comp.append(float(qa_gepa_metric(ex, pred).score))
            return {
                "em": float(np.mean(ems)) if ems else 0.0,
                "f1": float(np.mean(f1s)) if f1s else 0.0,
                "composite": float(np.mean(comp)) if comp else 0.0,
            }

        base_m = _score_agent(baseline_agent)
        comp_m = _score_agent(compiled_agent)
        report = {
            "dataset": ref.canonical,
            "num_questions": len(questions),
            "baseline": base_m,
            "compiled": comp_m,
            "delta_f1": comp_m["f1"] - base_m["f1"],
            "delta_composite": comp_m["composite"] - base_m["composite"],
        }
        write_text_atomic(eval_out, json.dumps(report, indent=2) + "\n")
        return report
    finally:
        conn.close()
