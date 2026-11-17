"""Bare-ReAct GEPA instruction optimization and lazy Optuna TPE retrieval tuning."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import dspy
import filelock
import numpy as np
from dspy.teleprompt.gepa.gepa import ProposalFn
from numpy.typing import NDArray

from yaw.core import (
    RetrievalParams,
    YawConfig,
    YawRuntimeError,
    format_query_for_embedding,
    get_logger,
    parse_dataset_list,
    resolve_coco_state_root,
    write_retrieval_section,
    write_text_atomic,
)
from yaw.datasets import (
    DATASET_REGISTRY,
    DEFAULT_PARITY_DATASETS,
    TIER2_CORPORA,
    DatasetBundle,
    build_benchmark_slice,
    compute_qa_em_f1,
    doc_matches_gold,
    document_identity,
    load_dataset_bundle,
    load_dataset_files,
    resolve_dataset_name,
    score_retrieval_recalls,
)
from yaw.pipeline import close_pipeline, make_dspy_embed_fn, start_pipeline, update_pipeline_once
from yaw.qa import build_qa_agent, make_qa_tools
from yaw.retrieval import (
    VectorSearch,
    build_vector_search,
    hydrate_ranked_passages,
    open_readonly_db,
    query_external_hints,
    search_vectors,
)
from yaw.runtime import create_lm

if TYPE_CHECKING:
    from dspy.teleprompt.gepa.gepa import GEPAFeedbackMetric

_CHUNK_ID_RE = re.compile(r"\[chunk_id=(\d+)\]")
_PASSAGE_HDR_RE = re.compile(r"\[chunk_id=\d+\]\s+\([^\n]+:\d+-\d+\)\s+score=[-+]?\d+(?:\.\d+)?:\n")
_VALID_AUTO: dict[str, Literal["light", "medium", "heavy"]] = {"light": "light", "medium": "medium", "heavy": "heavy"}
_HELD_OUT_GUARDED = TIER2_CORPORA | frozenset(DEFAULT_PARITY_DATASETS)
_RETRIEVAL_K_LIST = (1, 2, 5, 10, 20)


class ProposeInstruction(dspy.Signature):
    """Inspect stripped reflective failures via tools and propose an improved instruction."""

    component_name: str = dspy.InputField(desc="Predictor component name")
    current_instruction: str = dspy.InputField(desc="Existing instruction text")
    num_examples: int = dspy.InputField(desc="Count of reflective examples")
    length_guidance: str = dspy.InputField(desc="Length guidance for new_instruction")
    new_instruction: str = dspy.OutputField(desc="Improved instruction")


def _validate_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise YawRuntimeError("INVALID_INPUT", f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _propose_component(
    comp: str,
    current_instruction: str,
    stripped: Sequence[dict[str, Any]],
    *,
    lm: dspy.LM,
    char_limit: int | None,
    iter_limit: int,
    lm_sem: threading.Semaphore | None,
) -> str:
    def list_examples() -> str:
        """List 0-based reflective examples with single-line question and feedback previews."""
        return "\n".join(
            f"[{idx}] question={' '.join(str(inp.get('question', '') if isinstance(inp, Mapping) else '').split())!r} "
            f"| feedback={' '.join(str(item.get('Feedback', '')).split())!r}"
            for idx, item in enumerate(stripped)
            for inp in (item.get("Inputs"),)
        )

    def inspect_example(idx: int, field: str = "") -> str:
        """Inspect a full stripped reflective example by 0-based index and optional field name."""
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"idx must be an integer, got {idx!r}")
        if idx < 0 or idx >= len(stripped):
            raise IndexError(f"idx {idx} out of range for {len(stripped)} example(s)")
        item = stripped[idx]
        if not (req := field.strip() if isinstance(field, str) else ""):
            return json.dumps(item, ensure_ascii=False, indent=2, default=str)
        inp = item.get("Inputs")
        if req in item:
            val = item[req]
        elif isinstance(inp, Mapping) and req in inp:
            val = inp[req]
        else:
            raise KeyError(f"Unknown field {field!r}")
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, indent=2, default=str)

    proposer = dspy.ReAct(ProposeInstruction, tools=[list_examples, inspect_example], max_iters=iter_limit)
    guidance = f"<= {char_limit} chars" if char_limit is not None else "no hard character limit"
    with lm_sem or contextlib.nullcontext(), dspy.context(lm=lm):
        pred = proposer(
            component_name=comp,
            current_instruction=current_instruction,
            num_examples=len(stripped),
            length_guidance=guidance,
        )

    term = str(getattr(pred, "termination_reason", "") or "")
    if term not in ("submit", "forced_submit"):
        raise ValueError(f"Unexpected instruction proposal termination: {term!r}")
    raw_instr = getattr(pred, "new_instruction", None)
    if not isinstance(raw_instr, str) or not (cleaned := raw_instr.strip()):
        raise ValueError("Instruction proposer produced an empty instruction.")
    if char_limit is not None and len(cleaned) > char_limit:
        raise ValueError(f"Proposed instruction length {len(cleaned)} exceeds max_chars={char_limit}.")
    return cleaned


def make_react_instruction_proposer(
    *,
    lm: dspy.LM,
    max_chars: int | None = None,
    max_iters: int = 6,
    lm_sem: threading.Semaphore | None = None,
) -> ProposalFn:
    """Build a bare dspy.ReAct GEPA ProposalFn with tool-stripped reflective example inspection."""
    char_limit = _validate_int(max_chars, "max_chars", 1) if max_chars is not None else None
    iter_limit = _validate_int(max_iters, "max_iters", 1)

    def _propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for comp in components_to_update:
            if not (raw_examples := reflective_dataset.get(comp)):
                out[comp] = candidate.get(comp, "")
                continue
            stripped = [
                {
                    **ex,
                    **(
                        {"Inputs": {k: v for k, v in ex["Inputs"].items() if k != "tools"}}
                        if isinstance(ex.get("Inputs"), Mapping)
                        else {}
                    ),
                }
                for ex in raw_examples
            ]
            out[comp] = _propose_component(
                comp,
                candidate.get(comp, ""),
                stripped,
                lm=lm,
                char_limit=char_limit,
                iter_limit=iter_limit,
                lm_sem=lm_sem,
            )
        return out

    return _propose


def _collect_tool_evidence(pred: Any, trace: Any) -> tuple[set[int], list[str]]:
    messages: list[Any] = []
    hist_msgs = getattr(getattr(pred, "history", None), "messages", None)
    if isinstance(hist_msgs, Sequence) and not isinstance(hist_msgs, (str, bytes)):
        messages.extend(hist_msgs)
    elif isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
        for step in trace:
            if (
                isinstance(step, tuple)
                and len(step) >= 2
                and isinstance(step[1], Mapping)
                and isinstance(sm := getattr(step[1].get("history"), "messages", None), Sequence)
                and not isinstance(sm, (str, bytes))
            ):
                messages.extend(sm)

    seen_ids: set[int] = set()
    seen_texts: list[str] = []
    for msg in messages:
        if not isinstance(msg, Mapping) or (tc := msg.get("tool_calls")) is None:
            continue
        calls = (tc.get("tool_calls") if isinstance(tc, Mapping) else getattr(tc, "tool_calls", None)) or ()
        tcr = tc.get("tool_call_results") if isinstance(tc, Mapping) else getattr(tc, "tool_call_results", None)
        results = (
            tcr.get("tool_call_results") if isinstance(tcr, Mapping) else getattr(tcr, "tool_call_results", None)
        ) or ()
        for call, res in zip(calls, results, strict=False):
            if (call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")) == "submit":
                continue
            is_err = bool(res.get("is_error")) if isinstance(res, Mapping) else bool(getattr(res, "is_error", False))
            val = res.get("value") if isinstance(res, Mapping) else getattr(res, "value", None)
            if not is_err and isinstance(val, str):
                seen_ids.update(int(m.group(1)) for m in _CHUNK_ID_RE.finditer(val))
                seen_texts.extend(clean for block in _PASSAGE_HDR_RE.split(val) if (clean := block.strip()))
    return seen_ids, list(dict.fromkeys(seen_texts))


def qa_gepa_metric(
    gold: dspy.Example,
    pred: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """Score ReAct QA prediction with 0.5*(EM+F1) and structured grounding feedback."""
    _ = (pred_name, pred_trace)
    raw_ans = pred.get("answer") if isinstance(pred, Mapping) else getattr(pred, "answer", None)
    pred_answer = raw_ans.strip() if isinstance(raw_ans, str) else (str(raw_ans).strip() if raw_ans is not None else "")
    raw_gold = gold.get("gold_answers") if isinstance(gold, Mapping) else getattr(gold, "gold_answers", None)
    gold_answers = (
        [raw_gold]
        if isinstance(raw_gold, str)
        else ([str(a) for a in raw_gold] if isinstance(raw_gold, Sequence) else [])
    )

    em, f1 = compute_qa_em_f1(pred_answer, gold_answers)
    score = 0.5 * em + 0.5 * f1
    inline_ids = [int(m.group(1)) for m in _CHUNK_ID_RE.finditer(pred_answer)]
    seen_ids, seen_texts = _collect_tool_evidence(pred, trace)
    term = str(getattr(pred, "termination_reason", "") or "none") if pred is not None else "none"

    parts = [
        f"EM={em:.2f}, F1={f1:.2f}, termination_reason={term}, "
        f"inline_citations={len(inline_ids)}, tool_seen_chunks={len(seen_ids)}"
    ]
    if score < 1.0:
        raw_docs = gold.get("gold_docs") if isinstance(gold, Mapping) else getattr(gold, "gold_docs", None)
        gold_docs = (
            [str(d) for d in raw_docs] if isinstance(raw_docs, Sequence) and not isinstance(raw_docs, str) else []
        )
        missed = [
            (doc.splitlines()[0].strip()[:100] or doc[:80])
            for doc in gold_docs
            if not any(doc_matches_gold(txt, doc) for txt in seen_texts)
        ]
        if missed:
            parts.append(f"missed_gold_titles={missed}")
        elif gold_docs:
            parts.append("hint=gold passages retrieved; refine exact answer extraction")
        else:
            parts.append("hint=refine search_passages/lexical_grep entity terms")
        if ungrounded := [cid for cid in dict.fromkeys(inline_ids) if cid not in seen_ids]:
            parts.append(f"ungrounded_inline_ids={ungrounded}")
        elif not inline_ids and pred_answer:
            parts.append("hint=cite supporting [chunk_id=<int>] inline")
        if term == "forced_submit":
            parts.append("hint=hit max_iters before submit; search concisely")
    return dspy.Prediction(score=float(score), feedback="; ".join(parts))


@dataclass(frozen=True, slots=True)
class _DatasetSpecRef:
    canonical: str
    bundle: DatasetBundle | None


def _resolve_workspace_path(ws: Path, raw_path: Path | str) -> Path:
    return p.resolve() if (p := Path(raw_path)).is_absolute() else (ws / p).resolve()


def _write_json(path: Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_bundle_or_error(bundle_dir: Path) -> DatasetBundle:
    try:
        return load_dataset_bundle(bundle_dir)
    except (ValueError, OSError) as exc:
        raise YawRuntimeError("INVALID_INPUT", f"Invalid dataset bundle {bundle_dir}: {exc}") from exc


def _resolve_refs_and_bounds(
    ws: Path,
    ds_names: Sequence[str],
    query_offset: int,
    max_queries: int,
    train_queries: int,
    *,
    allow_held_out_overlap: bool = False,
    max_guarded_queries: int | None = None,
) -> tuple[list[_DatasetSpecRef], int, int, int]:
    if not ds_names:
        raise YawRuntimeError("INVALID_INPUT", "At least one dataset must be specified.")
    refs: list[_DatasetSpecRef] = []
    for raw in ds_names:
        cand = Path(raw)
        if not cand.is_dir() and not cand.is_absolute() and (ws / cand).is_dir():
            cand = ws / cand
        if cand.name == "manifest.json" and cand.is_file():
            cand = cand.parent
        if cand.is_dir():
            bundle = _load_bundle_or_error(cand)
            refs.append(_DatasetSpecRef(canonical=bundle.dataset.strip().lower(), bundle=bundle))
            continue
        try:
            canon = resolve_dataset_name(raw)
        except ValueError as exc:
            raise YawRuntimeError("INVALID_INPUT", str(exc)) from exc
        bdir = ws / "outputs" / "datasets" / canon
        refs.append(
            _DatasetSpecRef(
                canonical=canon, bundle=_load_bundle_or_error(bdir) if (bdir / "manifest.json").is_file() else None
            )
        )

    eff_offset = _validate_int(query_offset, "query_offset", 0)
    eff_max_q = _validate_int(max_queries, "max_queries", 2)
    eff_train_q = _validate_int(train_queries, "train_queries", 1)
    if eff_train_q >= eff_max_q:
        raise YawRuntimeError("INVALID_INPUT", "Require 1 <= train_queries < max_queries.")
    if any(r.canonical in _HELD_OUT_GUARDED for r in refs):
        if not allow_held_out_overlap and eff_offset < 10:
            raise YawRuntimeError("INVALID_INPUT", "query_offset < 10 overlaps held-out evaluation on parity/tier-2.")
        if max_guarded_queries is not None and eff_max_q > max_guarded_queries:
            raise YawRuntimeError(
                "INVALID_INPUT", f"Held-out guard forbids max_queries > {max_guarded_queries} on parity/tier-2 corpora."
            )
    return refs, eff_offset, eff_max_q, eff_train_q


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
        t_key, b_key = "title", "text"

    doc_ids, questions, gold_docs, gold_answers = build_benchmark_slice(
        ref.canonical, samples, corpus, max_queries=max_queries, query_offset=query_offset, index_queries=index_queries
    )
    if len(questions) < 2:
        raise YawRuntimeError("INVALID_INPUT", f"Dataset {ref.canonical!r} yielded < 2 questions.")
    return doc_ids, questions, gold_docs, gold_answers, corpus, t_key, b_key


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
            body = str(row.get(body_key) if body_key in row else row.get("text", row.get("content", "")))
            rendered = (
                json.dumps({"title": str(row.get(title_key, "")), "text": body}, ensure_ascii=False, indent=2) + "\n"
            )
            digest = hashlib.sha256(f"{canonical}\0{did}".encode()).hexdigest()[:16]
            target = (stage_dir / f"{canonical}_{digest}.json").resolve()
            desired.add(target)
            if not target.exists() or target.read_bytes() != rendered.encode("utf-8"):
                write_text_atomic(target, rendered, newline="")
    for stale in stage_dir.glob("*.json"):
        if stale.resolve() not in desired:
            stale.unlink(missing_ok=True)


def _acquire_exclusive_lock(lock_path: Path, label: str) -> filelock.FileLock:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout as exc:
        raise YawRuntimeError("BUSY", f"{label} lock {lock_path} is already held.") from exc
    return lock


def _unlink_sqlite_artifacts(db_file: Path) -> None:
    for suffix in ("", "-wal", "-wal2", "-shm"):
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
    session = start_pipeline(replace(cfg, corpus_root=stage_dir), live=False, embed_fn=embed_fn)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)


def _train_split_index(n_items: int, eff_train: int, eff_max: int) -> int:
    return eff_train if n_items > eff_train else max(1, min(n_items - 1, (n_items * eff_train) // eff_max))


def _write_gepa_reports(
    *,
    compiled: dspy.Module,
    refs: Sequence[_DatasetSpecRef],
    prog_path: Path,
    eval_path: Path,
    audit_path: Path,
    eff_offset: int,
    n_train: int,
    n_val: int,
) -> tuple[float, int]:
    detailed = getattr(compiled, "detailed_results", None)
    val_scores = [float(s) for s in (getattr(detailed, "val_aggregate_scores", None) or [])]
    val_subscores = list(getattr(detailed, "val_subscores", None) or [])
    best_idx = int(getattr(detailed, "best_idx", 0)) if val_scores else 0
    best_score = max(val_scores, default=0.0)
    num_trials = int(getattr(detailed, "total_metric_calls", None) or len(val_scores))
    best_sub = (
        val_subscores[best_idx]
        if best_idx < len(val_subscores) and isinstance(val_subscores[best_idx], Mapping)
        else {}
    )

    _write_json(
        eval_path,
        {
            "datasets": [r.canonical for r in refs],
            "query_offset": eff_offset,
            "train_queries": n_train,
            "val_queries": n_val,
            "best_idx": best_idx,
            "best_score": float(best_score),
            "trials": int(num_trials),
            "val_aggregate_scores": val_scores,
            "best_val_subscores": {str(k): float(v) for k, v in best_sub.items()},
        },
    )

    cand_rows, eval_rows = [], []
    for idx, sc in enumerate(val_scores):
        sub = val_subscores[idx] if idx < len(val_subscores) and isinstance(val_subscores[idx], Mapping) else {}
        vals = [float(v) for v in sub.values()]
        cand_rows.append(
            {
                "cand": idx + 1,
                "n": max(1, len(vals)),
                "score": sc,
                "score_min": min(vals, default=sc),
                "score_max": max(vals, default=sc),
                "val_score": sc,
                "gepa_idx": idx,
                "is_best": idx == best_idx,
                "optimizer": "gepa",
            }
        )
        eval_rows.extend({"cand": idx + 1, "eval": ei, "score": vs_i} for ei, vs_i in enumerate(vals, start=1))

    _write_json(
        audit_path,
        {
            "program_path": str(prog_path),
            "best_idx": best_idx,
            "best_score": float(best_score),
            "total_metric_calls": int(num_trials),
            "num_full_val_evals": int(getattr(detailed, "num_full_val_evals", 0) or 0),
            "best_instructions": {name: pred.signature.instructions for name, pred in compiled.named_predictors()},
            "candidate_records": cand_rows,
            "eval_records": eval_rows,
        },
    )
    return float(best_score), int(num_trials)


def optimize_dspy(
    cfg: YawConfig,
    *,
    datasets: Sequence[str] | str | None = None,
    max_queries: int | None = None,
    train_queries: int | None = None,
    query_offset: int | None = None,
    auto: str | None = None,
    max_metric_calls: int | None = None,
    output_path: Path | str | None = None,
    db_path: Path | str | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Compile bare dspy.ReAct QA instructions with GEPA inside an isolated optimization sandbox."""
    ws = Path(cfg.workspace_root).resolve()
    ds_names = parse_dataset_list(datasets) if datasets is not None else list(cfg.optimize_datasets)
    raw_offset = cfg.optimize_query_offset if query_offset is None else query_offset
    refs, eff_offset, eff_max_q, eff_train_q = _resolve_refs_and_bounds(
        ws,
        ds_names,
        raw_offset,
        cfg.optimize_max_queries if max_queries is None else max_queries,
        cfg.optimize_train_queries if train_queries is None else train_queries,
        max_guarded_queries=50,
    )
    if auto is not None and max_metric_calls is not None:
        raise YawRuntimeError("INVALID_INPUT", "auto and max_metric_calls are mutually exclusive.")
    gepa_auto: Literal["light", "medium", "heavy"] | None = None
    gepa_max_calls: int | None = None
    if max_metric_calls is not None:
        gepa_max_calls = _validate_int(max_metric_calls, "max_metric_calls", 1)
    elif (gepa_auto := _VALID_AUTO.get(str(cfg.optimize_auto if auto is None else auto).strip().lower())) is None:
        raise YawRuntimeError("INVALID_INPUT", f"Unsupported auto budget: {auto!r}.")

    opt_cfg = cfg.for_optimization(db_path)
    if not all(
        (opt_cfg.db_path, opt_cfg.lock_path, opt_cfg.status_path, opt_cfg.compiled_program_path, opt_cfg.state_dir)
    ):
        raise YawRuntimeError("INVALID_INPUT", "Optimization config is missing required sandbox paths.")
    assert (
        opt_cfg.db_path
        and opt_cfg.lock_path
        and opt_cfg.status_path
        and opt_cfg.compiled_program_path
        and opt_cfg.state_dir
    )
    opt_db = Path(opt_cfg.db_path).resolve()
    status_path = Path(opt_cfg.status_path).resolve()
    prog_path = (
        _resolve_workspace_path(ws, output_path)
        if output_path is not None
        else Path(opt_cfg.compiled_program_path).resolve()
    )
    eval_path = prog_path.with_name(f"{prog_path.stem}_eval.json")
    audit_path = prog_path.with_name(f"{prog_path.stem}_audit.json")
    telem_dir = prog_path.with_name(f"{prog_path.stem}_telemetry")
    stage_dir = opt_db.parent / f"{opt_db.stem}_corpus"
    _ = resolve_coco_state_root(opt_cfg.state_dir)

    file_lock = _acquire_exclusive_lock(Path(opt_cfg.lock_path).resolve(), "Optimization")

    def _write_status(lifecycle: str, active_op: str | None, *, num_chunks: int = 0, err: str | None = None) -> None:
        body: dict[str, Any] = {
            "lifecycle": lifecycle,
            "active_op": active_op,
            "pid": os.getpid() if lifecycle != "stopped" else None,
            "datasets": [r.canonical for r in refs],
            "num_chunks": num_chunks,
            "last_error": err,
            "updated_at": time.time(),
        }
        body["status"] = dict(body)
        _write_json(status_path, body)

    last_err: str | None = None
    indexed_chunks = 0
    try:
        embed_fn = make_dspy_embed_fn(opt_cfg)
        trainset, valset, staged_slices = [], [], []
        for ref in refs:
            doc_ids, questions, gold_docs, gold_answers, corpus, t_key, b_key = _load_slice(
                ws, ref, max_queries=eff_max_q, query_offset=eff_offset
            )
            ds_ex = [
                dspy.Example(
                    question=questions[i],
                    gold_answers=gold_answers[i],
                    gold_docs=gold_docs[i] if gold_docs is not None else [],
                ).with_inputs("question")
                for i in range(len(questions))
            ]
            split_idx = _train_split_index(len(ds_ex), eff_train_q, eff_max_q)
            trainset.extend(ds_ex[:split_idx])
            valset.extend(ds_ex[split_idx:])
            staged_slices.append((ref.canonical, doc_ids, corpus, t_key, b_key))

        _write_status("starting", "indexing")
        _index_staged_slices(
            opt_cfg,
            stage_dir,
            opt_db,
            embed_fn,
            staged_slices,
            fresh=fresh,
            extra_wipe=(status_path, prog_path, audit_path, eval_path, telem_dir),
        )

        with contextlib.closing(open_readonly_db(opt_db)) as conn:
            vs = build_vector_search(conn, opt_cfg.retrieval)
            if vs.size == 0:
                raise YawRuntimeError("EMPTY_INDEX", "Optimization slice produced zero indexed chunks.")
            indexed_chunks = vs.size
            _write_status("running", "gepa_compile", num_chunks=indexed_chunks)

            agent = build_qa_agent(
                make_qa_tools(vs, stage_dir, embed_fn, conn, task_mode=opt_cfg.embedding_task_mode),
                max_iters=opt_cfg.react_max_iters,
            )
            lm = create_lm(opt_cfg)
            telem_dir.mkdir(parents=True, exist_ok=True)
            optimizer = dspy.GEPA(
                metric=cast("GEPAFeedbackMetric", qa_gepa_metric),
                reflection_lm=lm,
                instruction_proposer=make_react_instruction_proposer(lm=lm, lm_sem=threading.Semaphore(1)),
                reflection_minibatch_size=min(10, len(trainset)),
                candidate_selection_strategy="pareto",
                add_format_failure_as_feedback=True,
                num_threads=1,
                log_dir=str(telem_dir),
                track_stats=True,
                auto=gepa_auto,
                max_metric_calls=gepa_max_calls,
            )
            with dspy.context(lm=lm):
                compiled = optimizer.compile(student=agent, trainset=trainset, valset=valset)
            prog_path.parent.mkdir(parents=True, exist_ok=True)
            compiled.save(str(prog_path), save_program=False)

        best_score, num_trials = _write_gepa_reports(
            compiled=compiled,
            refs=refs,
            prog_path=prog_path,
            eval_path=eval_path,
            audit_path=audit_path,
            eff_offset=eff_offset,
            n_train=len(trainset),
            n_val=len(valset),
        )
        return {
            "program_path": str(prog_path),
            "eval_path": str(eval_path),
            "audit_path": str(audit_path),
            "best_score": best_score,
            "trials": num_trials,
        }
    except BaseException as exc:
        last_err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            _write_status("stopped", None, num_chunks=indexed_chunks, err=last_err)
        except OSError:
            if last_err is None:
                raise
        finally:
            if file_lock.is_locked:
                file_lock.release()


@dataclass(frozen=True, slots=True)
class _OptunaSliceData:
    canonical: str
    corpus_dir: Path
    vs: VectorSearch
    questions: tuple[str, ...]
    q_vecs: NDArray[np.float32]
    hints: tuple[tuple[dict[int, float], dict[int, float]], ...]
    gold_docs: tuple[tuple[str, ...], ...]
    train_split: int


def _eval_optuna_slices(
    slices: Sequence[_OptunaSliceData], params: RetrievalParams
) -> tuple[dict[str, float], float, int, int]:
    pool_k = max(20 * 4, 50)
    train_scores, val_r5s, cycle_count, seed_deficit = [], [], 0, 0

    for item in slices:
        retrieved: list[list[str]] = []
        for idx, query in enumerate(item.questions):
            fts_s, vec_s = item.hints[idx]
            trace = search_vectors(
                item.vs, item.q_vecs[idx], query, top_k=pool_k, fts_scores=fts_s, vec_scores=vec_s, params=params
            )
            if trace.num_particles < 1 or not trace.ranked_chunks:
                seed_deficit += 1
            route_graph: dict[int, set[int]] = {}
            for branch in trace.routes:
                for step_a, step_b in zip(branch, branch[1:], strict=False):
                    route_graph.setdefault(step_b.chunk_id, set()).add(step_a.chunk_id)
            try:
                list(TopologicalSorter(route_graph).static_order())
            except CycleError:
                cycle_count += 1
            retrieved.append(
                [hp.text for hp in hydrate_ranked_passages(item.vs, trace.ranked_chunks, item.corpus_dir, top_k=20)]
            )

        n_tr = item.train_split
        tr_means, _ = score_retrieval_recalls(item.gold_docs[:n_tr], retrieved[:n_tr], k_list=_RETRIEVAL_K_LIST)
        va_means, _ = score_retrieval_recalls(item.gold_docs[n_tr:], retrieved[n_tr:], k_list=(5,))
        train_scores.append(tr_means)
        val_r5s.append(float(va_means["Recall@5"]))

    pooled_train = {
        f"Recall@{k}": sum(m[f"Recall@{k}"] for m in train_scores) / len(train_scores) for k in _RETRIEVAL_K_LIST
    }
    return pooled_train, sum(val_r5s) / len(val_r5s), cycle_count, seed_deficit


def optimize_optuna(
    cfg: YawConfig,
    *,
    datasets: Sequence[str] | str | None = None,
    query_offset: int = 10,
    max_queries: int = 20,
    train_queries: int = 12,
    index_queries: int = 20,
    trials: int = 64,
    study_db: Path | str = ".yaw/optimize_optuna.db",
    fresh: bool = False,
    write_config: bool = False,
    allow_eval_overlap: bool = False,
    output_path: Path | str | None = None,
    toml_path: Path | str | None = None,
) -> dict[str, Any]:
    """Tune the 5 dimensionless RetrievalParams via 5-objective TPE with feasibility constraints."""
    import optuna
    from optuna.importance import PedAnovaImportanceEvaluator

    ws = Path(cfg.workspace_root).resolve()
    ds_names = parse_dataset_list(datasets) if datasets is not None else list(DEFAULT_PARITY_DATASETS)
    refs, eff_offset, eff_max_q, eff_train_q = _resolve_refs_and_bounds(
        ws, ds_names, query_offset, max_queries, train_queries, allow_held_out_overlap=allow_eval_overlap
    )
    eff_index_q = _validate_int(index_queries, "index_queries", 1)
    eff_trials = _validate_int(trials, "trials", 1)

    study_path = _resolve_workspace_path(ws, study_db)
    file_lock = _acquire_exclusive_lock(Path(f"{study_path}.lock"), "Optuna study")
    storage = None
    try:
        if fresh:
            _unlink_sqlite_artifacts(study_path)

        embed_fn = make_dspy_embed_fn(cfg)
        state_root = resolve_coco_state_root(cfg.state_dir or (study_path.parent / "coco_lmdb"))
        slice_items: list[_OptunaSliceData] = []
        for ref in refs:
            doc_ids, questions, gold_docs, _, corpus, t_key, b_key = _load_slice(
                ws, ref, max_queries=eff_max_q, query_offset=eff_offset, index_queries=eff_index_q
            )
            if gold_docs is None:
                raise YawRuntimeError("INVALID_INPUT", f"Dataset {ref.canonical!r} has no gold retrieval passages.")
            safe_ds = re.sub(r"[^a-zA-Z0-9_.-]+", "_", ref.canonical).strip("_") or "dataset"
            ds_dir = study_path.parent / f"optuna_{safe_ds}"
            ds_corpus, ds_db = ds_dir / "corpus", ds_dir / "yaw.db"
            ds_cfg = replace(
                cfg,
                corpus_root=ds_corpus,
                db_path=ds_db,
                lock_path=ds_dir / "yaw.db.lock",
                state_dir=state_root,
                status_path=ds_dir / "status.json",
                poll_interval_seconds=0.0,
            )
            _index_staged_slices(
                ds_cfg, ds_corpus, ds_db, embed_fn, [(ref.canonical, doc_ids, corpus, t_key, b_key)], fresh=fresh
            )
            q_vecs = np.asarray(
                embed_fn([format_query_for_embedding(q, cfg.embedding_task_mode) for q in questions]), dtype=np.float32
            )
            with contextlib.closing(open_readonly_db(ds_db)) as conn:
                vs = build_vector_search(conn, cfg.retrieval)
                if vs.size == 0:
                    raise YawRuntimeError("EMPTY_INDEX", f"Dataset slice {ref.canonical!r} indexed zero chunks.")
                hints = tuple(query_external_hints(conn, q, q_vecs[i]) for i, q in enumerate(questions))

            slice_items.append(
                _OptunaSliceData(
                    ref.canonical,
                    ds_corpus,
                    vs,
                    tuple(questions),
                    q_vecs,
                    hints,
                    tuple(tuple(g) for g in gold_docs),
                    _train_split_index(len(questions), eff_train_q, eff_max_q),
                )
            )

        _, baseline_val_r5, _, _ = _eval_optuna_slices(slice_items, cfg.retrieval)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_path.as_posix()}")
        study = optuna.create_study(
            study_name="yaw_retrieval",
            directions=["maximize"] * 5,
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, constant_liar=True),
            storage=storage,
            load_if_exists=not fresh,
        )

        def _objective(trial: optuna.Trial) -> tuple[float, float, float, float, float]:
            params = RetrievalParams(
                peclet=trial.suggest_float("peclet", 0.20, 0.95),
                alpha_sprt=trial.suggest_float("alpha_sprt", 0.005, 0.25),
                redundancy_curvature=trial.suggest_float("redundancy_curvature", 0.10, 1.50),
                coalition_coupling=trial.suggest_float("coalition_coupling", 0.0, 0.90),
                external_trust=trial.suggest_float("external_trust", 0.05, 0.80),
            )
            tr_recalls, val_r5, cycle_cnt, seed_def = _eval_optuna_slices(slice_items, params)
            for cname, cval in (
                ("cycle_count", float(cycle_cnt)),
                ("seed_deficit", float(seed_def)),
                ("val_regression", max(0.0, baseline_val_r5 - val_r5)),
            ):
                trial.set_constraint(cname, cval)
            trial.set_user_attr("val_recall_5", float(val_r5))
            return (
                float(tr_recalls["Recall@1"]),
                float(tr_recalls["Recall@2"]),
                float(tr_recalls["Recall@5"]),
                float(tr_recalls["Recall@10"]),
                float(tr_recalls["Recall@20"]),
            )

        study.optimize(_objective, n_trials=eff_trials)
        complete = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values and len(t.values) == 5
        ]
        if not complete:
            raise YawRuntimeError("MODEL_ERROR", "Optuna study produced zero completed trials.")
        feasible = [t for t in complete if len(t.constraints) == 3 and all(v <= 0.0 for v in t.constraints.values())]
        if not feasible:
            raise YawRuntimeError("MODEL_ERROR", "Optuna study produced zero feasible trials.")

        best_trial = max(
            feasible,
            key=lambda t: (
                float((t.values or (0.0,) * 5)[2]),
                float((t.values or (0.0,) * 5)[4]),
                float((t.values or (0.0,) * 5)[3]),
                float((t.values or (0.0,) * 5)[1]),
                float((t.values or (0.0,) * 5)[0]),
                -t.number,
            ),
        )
        best_params = {str(k): float(v) for k, v in best_trial.params.items()}
        best_values = {f"Recall@{k}": float(v) for k, v in zip(_RETRIEVAL_K_LIST, best_trial.values or (), strict=True)}

        importances: dict[str, float] | None = None
        distinct_r5 = {round(float(t.values[2]), 6) for t in complete if t.values}
        if len(complete) >= 4 and len(distinct_r5) >= 2:
            raw_imp = PedAnovaImportanceEvaluator().evaluate(
                study, target=lambda t: -float(t.values[2]) if t.values else 0.0
            )
            importances = {str(k): float(v) for k, v in raw_imp.items()}
        else:
            get_logger("yaw.optimize").info(
                "Skipping PedAnova importance evaluation (complete=%d, distinct_recall5=%d).",
                len(complete),
                len(distinct_r5),
            )

        if write_config:
            write_retrieval_section(
                best_params,
                path=_resolve_workspace_path(ws, toml_path) if toml_path is not None else None,
                workspace_override=ws,
            )

        result: dict[str, Any] = {
            "best_params": best_params,
            "best_values": best_values,
            "importances": importances,
            "study_db": str(study_path),
            "trials": len(complete),
        }
        if output_path is not None:
            _write_json(
                _resolve_workspace_path(ws, output_path),
                {**result, "baseline_val_recall_5": float(baseline_val_r5)},
            )
        return result
    finally:
        if storage is not None:
            storage.engine.dispose()
        if file_lock.is_locked:
            file_lock.release()
