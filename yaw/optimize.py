"""Offline DSPy GEPA + bare dspy.ReAct prompt optimization."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback
from filelock import FileLock, Timeout

from yaw.core import (
    EmbedFn,
    YawConfig,
    YawRuntimeError,
    parse_dataset_list,
    redact_secrets,
    resolve_coco_state_root,
    write_text_atomic,
)
from yaw.datasets import (
    DEFAULT_PARITY_DATASETS,
    TIER2_CORPORA,
    DatasetBundle,
    build_benchmark_slice,
    compute_qa_em_f1,
    load_dataset_bundle,
    resolve_dataset_bundle,
    resolve_dataset_name,
    resolve_slice_corpus_scope,
    score_retrieval_recalls,
)
from yaw.pipeline import close_pipeline, drop_pipeline, make_dspy_embed_fn, start_pipeline, update_pipeline_once
from yaw.qa import (
    ABSTAIN_REPLY,
    build_qa_agent,
    collect_tool_history,
    create_lm,
    extract_inline_chunk_ids,
    make_qa_tools,
    strip_inline_citations,
)
from yaw.retrieval import build_retrieval_index, open_readonly_db


def qa_gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: DSPyTrace | None = None,
    pred_name: str | None = None,
    pred_trace: DSPyTrace | None = None,
) -> ScoreWithFeedback:
    """5-argument GEPA metric returning ScoreWithFeedback with diagnostic grounding critique."""
    _ = (trace, pred_name, pred_trace)
    raw_ans = str(getattr(pred, "answer", "") or "").strip()

    gold_answers_raw = getattr(gold, "gold_answers", None)
    if gold_answers_raw is None:
        single = getattr(gold, "answer", None)
        gold_answers = [s] if single is not None and (s := str(single).strip()) else []
    elif isinstance(gold_answers_raw, str):
        gold_answers = [s] if (s := gold_answers_raw.strip()) else []
    else:
        gold_answers = [s for a in gold_answers_raw if a is not None and (s := str(a).strip())]
    gold_docs_raw = getattr(gold, "gold_docs", None)
    gold_docs = (
        [s for d in gold_docs_raw if d is not None and (s := str(d).strip())]
        if isinstance(gold_docs_raw, Sequence) and not isinstance(gold_docs_raw, (str, bytes))
        else None
    )

    answer = strip_inline_citations(raw_ans)
    em, f1 = compute_qa_em_f1(answer, gold_answers)
    obs_blocks, tool_calls, seen_chunks = collect_tool_history(pred)

    abstained = ABSTAIN_REPLY.lower() in raw_ans.lower()
    cites = extract_inline_chunk_ids(raw_ans)
    valid_cites = [cid for cid in cites if cid in seen_chunks]
    if abstained:
        cite_score = 1.0 if not cites else 0.5
    elif not cites:
        cite_score = 0.0
    else:
        cite_score = len(valid_cites) / len(cites)

    if gold_docs is not None and bool(gold_docs):
        k_ev = max(len(obs_blocks), 1)
        ev_recall = score_retrieval_recalls([gold_docs], [obs_blocks], k_list=(k_ev,))[0][f"Recall@{k_ev}"]
        score = 0.55 * f1 + 0.15 * em + 0.15 * cite_score + 0.15 * ev_recall
    else:
        ev_recall = None
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
    """Inspect reflective traces via tools and refine the predictor instruction to fix failure modes.

    Produce a general, transferable instruction without copying example-specific facts,
    respecting the specified length guidance.
    """

    component_name: str = dspy.InputField(desc="Name of the DSPy predictor component being updated.")
    current_instruction: str = dspy.InputField(desc="Current system instruction for the target predictor.")
    num_examples: int = dspy.InputField(desc="Number of reflective trace examples available via tools.")
    length_guidance: str = dspy.InputField(desc="Maximum length or conciseness guidance for the instruction.")
    improved_instruction: str = dspy.OutputField(desc="Improved concise instruction addressing failure modes.")


def make_react_instruction_proposer(
    reflection_lm: dspy.LM,
    *,
    max_chars: int | None = None,
    max_iters: int = 6,
) -> Callable[..., dict[str, str]]:
    """Build a bare dspy.ReAct GEPA instruction proposer with tools for inspecting reflective traces."""
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1):
        raise ValueError("max_chars must be None or a positive integer.")
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer.")

    length_guidance = (
        f"Must be at most {max_chars} characters."
        if max_chars is not None
        else "No hard character limit; be concise and clear."
    )

    def propose_component(comp: str, current: str, rows: Sequence[Mapping[str, Any]]) -> str:
        cleaned_rows: list[dict[str, Any]] = []
        for ex in rows:
            row = dict(ex)
            inputs = row.get("Inputs")
            if isinstance(inputs, Mapping):
                row["Inputs"] = {k: v for k, v in inputs.items() if k != "tools"}
            cleaned_rows.append(row)

        def list_examples() -> str:
            """List available reflective example indices, keys, scores, and question/feedback overview."""
            overview: list[dict[str, Any]] = []
            for idx, row in enumerate(cleaned_rows):
                entry: dict[str, Any] = {"index": idx, "fields": list(row.keys())}
                inputs = row.get("Inputs")
                if isinstance(inputs, Mapping):
                    entry["input_keys"] = list(inputs.keys())
                    if "question" in inputs:
                        entry["question"] = inputs["question"]
                outputs = row.get("Generated Outputs")
                if isinstance(outputs, Mapping):
                    entry["output_keys"] = list(outputs.keys())
                for score_key in ("score", "Score"):
                    if score_key in row:
                        entry["score"] = row[score_key]
                        break
                for fb_key in ("Feedback", "feedback"):
                    if fb_key in row:
                        entry["feedback"] = row[fb_key]
                        break
                overview.append(entry)
            return json.dumps(overview, ensure_ascii=False, default=str)

        def inspect_example(index: int, field: str = "") -> str:
            """Return the full uncropped JSON for one reflective example or a single top-level field."""
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("index must be an integer.")
            if not isinstance(field, str):
                raise TypeError("field must be a string.")
            if index < 0 or index >= len(cleaned_rows):
                raise IndexError(f"Example index {index} out of range for {len(cleaned_rows)} examples.")
            row = cleaned_rows[index]
            if not field:
                return json.dumps(row, ensure_ascii=False, default=str)
            if field not in row:
                raise KeyError(f"Unknown field {field!r}.")
            val = row[field]
            return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)

        proposer = dspy.ReAct(ProposeInstruction, tools=[list_examples, inspect_example], max_iters=max_iters)
        proposer.react.update_config(tool_choice="required")
        with dspy.context(lm=reflection_lm):
            res = proposer(
                component_name=comp,
                current_instruction=current,
                num_examples=len(cleaned_rows),
                length_guidance=length_guidance,
            )
        termination_reason = getattr(res, "termination_reason", None)
        if termination_reason is not None and termination_reason not in ("submit", "forced_submit"):
            raise ValueError(f"Instruction proposer did not submit (termination_reason={termination_reason!r}).")
        improved = str(getattr(res, "improved_instruction", "") or "").strip()
        if not improved:
            raise ValueError("Instruction proposer returned an empty instruction.")
        if max_chars is not None and len(improved) > max_chars:
            raise ValueError(f"Proposed instruction length {len(improved)} exceeds max_chars={max_chars}.")
        return improved

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: Sequence[str],
    ) -> dict[str, str]:
        return {
            comp: propose_component(comp, candidate[comp], rows)
            for comp in components_to_update
            if comp in candidate and (rows := reflective_dataset.get(comp)) is not None and bool(rows)
        }

    return propose


def _require_opt_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise YawRuntimeError("INVALID_INPUT", f"Configuration {label} must not be empty.")
    return path


@dataclass(frozen=True, slots=True)
class _DatasetSpecRef:
    canonical: str
    bundle: DatasetBundle


def _guard_optimization_slice(canonicals: Sequence[str], offset: int) -> None:
    if offset < 10 and any(c in TIER2_CORPORA or c in DEFAULT_PARITY_DATASETS for c in canonicals):
        raise YawRuntimeError(
            "INVALID_INPUT",
            "query_offset < 10 overlaps held-out evaluation; pass --query-offset 10 "
            "or set [optimization].query_offset >= 10 in yaw.toml.",
        )


def _resolve_dataset_refs(ws: Path, names: Sequence[str], *, query_offset: int) -> list[_DatasetSpecRef]:
    if not names:
        raise YawRuntimeError("INVALID_INPUT", "At least one dataset must be specified for optimization.")
    specs: list[tuple[str, DatasetBundle | Path]] = []
    for raw in names:
        path = Path(raw)
        if path.is_dir() or path.is_file():
            bundle = resolve_dataset_bundle(path)
            specs.append((bundle.dataset, bundle))
        else:
            canonical = resolve_dataset_name(raw)
            specs.append((canonical, ws / "outputs" / "datasets" / canonical))
    _guard_optimization_slice([c for c, _ in specs], query_offset)
    out: list[_DatasetSpecRef] = []
    for canonical, src in specs:
        if isinstance(src, DatasetBundle):
            out.append(_DatasetSpecRef(canonical, src))
        elif (src / "manifest.json").is_file():
            out.append(_DatasetSpecRef(canonical, load_dataset_bundle(src)))
        else:
            raise YawRuntimeError(
                "INVALID_INPUT",
                f"Prepared dataset bundle not found for {canonical!r}; "
                f"run python -m scripts.download_datasets --dataset {canonical}.",
            )
    return out


def _load_slice(
    ref: _DatasetSpecRef, *, max_queries: int, query_offset: int
) -> tuple[list[str], list[str], list[list[str]] | None, list[list[str]]]:
    samples, corpus = ref.bundle.read_inputs()
    doc_ids, questions, gold_docs, gold_answers = build_benchmark_slice(
        ref.canonical, samples, corpus, max_queries=max_queries, query_offset=query_offset
    )
    if len(questions) < 2:
        raise YawRuntimeError("INVALID_INPUT", f"Dataset {ref.canonical!r} yielded < 2 questions.")
    return doc_ids, questions, gold_docs, gold_answers


def _unlink_sqlite_artifacts(db_file: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{db_file}{suffix}").unlink(missing_ok=True)


def _index_bundle_slices(
    cfg: YawConfig,
    corpus_root: Path,
    slice_patterns: tuple[str, ...],
    db_file: Path,
    embed_fn: EmbedFn,
    *,
    fresh: bool,
    extra_wipe: Sequence[Path] = (),
) -> None:
    idx_cfg = replace(cfg, db_path=db_file, corpus_root=corpus_root, included_patterns=slice_patterns)
    if fresh:
        if cfg.state_dir is not None and resolve_coco_state_root(cfg.state_dir).exists():
            drop_session = start_pipeline(idx_cfg, live=False, embed_fn=embed_fn)
            try:
                drop_pipeline(drop_session)
            finally:
                close_pipeline(drop_session)
        _unlink_sqlite_artifacts(db_file)
        for p in extra_wipe:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink(missing_ok=True)

    session = start_pipeline(idx_cfg, embed_fn=embed_fn, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)


@contextlib.contextmanager
def _optimization_status(status_path: Path, api_key: str) -> Generator[None]:
    write_text_atomic(
        status_path,
        json.dumps({"state": "running", "pid": os.getpid(), "started_at": time.time()}, indent=2) + "\n",
    )
    try:
        yield
    except KeyboardInterrupt:
        write_text_atomic(status_path, json.dumps({"state": "interrupted", "pid": None}, indent=2) + "\n")
        raise
    except Exception as exc:
        err = redact_secrets(f"{type(exc).__name__}: {exc}", api_key)
        write_text_atomic(status_path, json.dumps({"state": "failed", "pid": None, "error": err}, indent=2) + "\n")
        raise


def _require_int_param(val: int | None, default: int, *, name: str, min_value: int = 1) -> int:
    if val is None:
        return default
    if isinstance(val, bool) or not isinstance(val, int) or val < min_value:
        raise YawRuntimeError("INVALID_INPUT", f"{name} must be a non-bool integer >= {min_value}.")
    return val


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
    embed_fn: EmbedFn | None = None,
    student_lm: Any = None,
    reflection_lm: Any = None,
    instruction_proposer: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compile the bare dspy.ReAct QA agent with dspy.GEPA and persist program + audit artifacts."""
    if auto is not None and max_metric_calls is not None:
        raise YawRuntimeError("INVALID_INPUT", "auto and max_metric_calls are mutually exclusive.")
    eff_threads = _require_int_param(threads, 1, name="threads", min_value=1)
    eff_max_calls = (
        _require_int_param(max_metric_calls, 1, name="max_metric_calls", min_value=1)
        if max_metric_calls is not None
        else None
    )
    eff_max_q = _require_int_param(max_queries, cfg.optimize_max_queries, name="max_queries", min_value=1)
    eff_offset = _require_int_param(query_offset, cfg.optimize_query_offset, name="query_offset", min_value=0)
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
            max_metric_calls=eff_max_calls,
            max_queries=eff_max_q,
            query_offset=eff_offset,
            threads=eff_threads,
            fresh=fresh,
            embed_fn=embed_fn,
            student_lm=student_lm,
            reflection_lm=reflection_lm,
            instruction_proposer=instruction_proposer,
        )
    finally:
        lock.release()


def _compile_dspy(
    opt_cfg: YawConfig,
    *,
    datasets: str | None,
    auto: str | None,
    max_metric_calls: int | None,
    max_queries: int,
    query_offset: int,
    threads: int,
    fresh: bool,
    embed_fn: EmbedFn | None,
    student_lm: Any,
    reflection_lm: Any,
    instruction_proposer: Callable[..., dict[str, str]] | None,
) -> dict[str, Any]:
    db_path = _require_opt_path(opt_cfg.db_path, "dspy db_path")
    prog_path = _require_opt_path(opt_cfg.compiled_program_path, "compiled_program_path")
    status_path = _require_opt_path(opt_cfg.status_path, "dspy status_path")
    ws = opt_cfg.workspace_root
    opt_dir = db_path.parent
    opt_dir.mkdir(parents=True, exist_ok=True)
    audit_path = opt_dir / "dspy_gepa_audit.json"
    telem_dir = opt_dir / "gepa_logs"

    ds_names = parse_dataset_list(datasets) if datasets is not None else list(opt_cfg.optimize_datasets)
    refs = _resolve_dataset_refs(ws, ds_names, query_offset=query_offset)
    loaded = [_load_slice(r, max_queries=max_queries, query_offset=query_offset) for r in refs]
    corpus_root, slice_patterns = resolve_slice_corpus_scope(
        [(ref.bundle, doc_ids) for ref, (doc_ids, _, _, _) in zip(refs, loaded, strict=True)]
    )
    active_embed = embed_fn or make_dspy_embed_fn(opt_cfg)

    with _optimization_status(status_path, opt_cfg.api_key):
        _index_bundle_slices(
            opt_cfg,
            corpus_root,
            slice_patterns,
            db_path,
            active_embed,
            fresh=fresh,
            extra_wipe=(prog_path, audit_path, telem_dir),
        )
        telem_dir.mkdir(parents=True, exist_ok=True)

        trainset: list[dspy.Example] = []
        valset: list[dspy.Example] = []
        for _, questions, gold_docs, gold_answers in loaded:
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

        s_lm = student_lm or create_lm(opt_cfg, temperature=1.0)
        r_lm = reflection_lm or create_lm(opt_cfg, temperature=1.0)
        proposer = instruction_proposer or make_react_instruction_proposer(r_lm, max_iters=opt_cfg.react_max_iters)

        gepa_kwargs: dict[str, Any] = {
            "metric": qa_gepa_metric,
            "reflection_lm": r_lm,
            "instruction_proposer": proposer,
            "num_threads": threads,
            "log_dir": str(telem_dir),
            "track_stats": True,
            "seed": 42,
        }
        if max_metric_calls is not None:
            gepa_kwargs["max_metric_calls"] = max_metric_calls
        else:
            eff_auto = auto if auto is not None else opt_cfg.optimize_auto
            gepa_kwargs["auto"] = eff_auto

        with contextlib.closing(open_readonly_db(db_path)) as init_conn:
            index = build_retrieval_index(
                init_conn,
                expected_embedding_model=opt_cfg.embedding_model,
                expected_task_mode=opt_cfg.embedding_task_mode,
            )
        tools = make_qa_tools(
            index,
            corpus_root,
            active_embed,
            db_path,
            task_mode=opt_cfg.embedding_task_mode,
        )
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
