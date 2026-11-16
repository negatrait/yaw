"""Persistent single-study Optuna TPESampler tuner for SubStateManifold hyperparameters."""

import argparse
import hashlib
import json
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna

from yaw.core import ChunkPointer, YawConfig, load_config, parse_dataset_list, write_manifold_section, write_text_atomic
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.hmm import (
    EvidenceHop,
    ManifoldHyperparams,
    SubStateManifold,
    build_substate_manifold,
    run_substate_hmm,
)
from yaw.daemon.graph.inference import (
    PASSAGE_INSTRUCTION,
    DSPyEmbeddingAdapter,
    YawProgram,
    compute_retrieval_recalls,
    create_dspy_lm,
)
from yaw.daemon.graph.storage import (
    get_chunk_pointers,
    hydrate_budgeted_passages,
    open_readonly_yaw_db,
    search_chunk_vectors_knn,
    search_chunks_fts,
)
from yaw.daemon.ingestion import build_benchmark_slice, resolve_dataset_bundle
from yaw.daemon.runtime import YawRuntime

DEFAULT_TUNE_DATASETS: tuple[str, ...] = ("2wikimultihopqa", "musique", "hotpotqa", "popqa")
DEFAULT_STUDY_DB = ".yaw/optuna_manifold.db"
DEFAULT_BEST_JSON = ".yaw/optuna_manifold_best.json"
DEFAULT_STUDY_NAME = "manifold_motpe"

_MIN_DISJOINT_QUERY_OFFSET = 10
_FTS_K, _VEC_K, _QUERY_POOL_K = 50, 200, 50
_EVAL_KS: tuple[int, ...] = (1, 2, 5, 10, 20)
_DEFAULT_HP = ManifoldHyperparams()


@dataclass(frozen=True, slots=True)
class _ParamSpec:
    name: str
    kind: Literal["int", "float"]
    low: float
    high: float
    default: int | float


_PARAM_SPECS: tuple[_ParamSpec, ...] = tuple(
    _ParamSpec(name, "float", lo, hi, getattr(_DEFAULT_HP, name))
    for name, lo, hi in (
        ("peclet", 0.20, 0.95),
        ("alpha_sprt", 0.005, 0.25),
        ("redundancy_curvature", 0.10, 1.50),
        ("coalition_coupling", 0.0, 0.90),
        ("external_trust", 0.05, 0.80),
    )
)


@dataclass(slots=True)
class _PreparedDataset:
    dataset: str
    bundle_root: Path
    ptr_by_cid: dict[str, ChunkPointer]
    queries: list[str]
    gold_docs: list[list[str]]
    query_vecs: np.ndarray
    fts_hits_by_q: list[dict[str, float]]
    vec_hits_by_q: list[dict[str, float]]
    train_queries: int
    max_context_chars: int | None
    max_ast_expansion_hops: int
    default_manifold: SubStateManifold
    recall_memo: dict[Any, tuple[float, ...]]


def _db_has_manifold(db_path: Path, min_chunks: int) -> bool:
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return False
    conn = open_readonly_yaw_db(db_path, timeout=2.0)
    try:
        sql = "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        if not {"chunks", "chunk_vec"}.issubset({str(r[0]) for r in conn.execute(sql)}):
            return False
        n_c = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return n_c >= min_chunks and int(conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]) == n_c
    finally:
        conn.close()


def _route_has_cycle(routes: tuple[tuple[EvidenceHop, ...], ...], fallback: tuple[EvidenceHop, ...]) -> bool:
    preds: dict[str, set[str]] = {}
    for br in routes or ((fallback,) if fallback else ()):
        for u, v in zip(br, br[1:], strict=False):
            preds.setdefault(v.chunk_id, set()).add(u.chunk_id)
            preds.setdefault(u.chunk_id, set())
    try:
        if preds:
            list(TopologicalSorter(preds).static_order())
        return False
    except CycleError:
        return True


def _eval_split(
    datasets: Sequence[_PreparedDataset], manifolds: Sequence[SubStateManifold], params: ManifoldHyperparams, split: str
) -> dict[str, Any]:
    ds_recalls: list[np.ndarray] = []
    cycles, seeds, queries = 0, 0, 0
    max_k, pool_lim = max(_EVAL_KS), max(max(_EVAL_KS) * 2, 40)
    for ds, man in zip(datasets, manifolds, strict=True):
        q_idx_range = range(ds.train_queries) if split == "train" else range(ds.train_queries, len(ds.queries))
        q_rec = np.zeros(len(_EVAL_KS), dtype=np.float64)
        for qi in q_idx_range:
            queries += 1
            res = run_substate_hmm(
                man,
                ds.queries[qi],
                ds.query_vecs[qi],
                top_k=_QUERY_POOL_K,
                fts_chunk_scores=ds.fts_hits_by_q[qi],
                vec_chunk_scores=ds.vec_hits_by_q[qi],
                params=params,
            )
            seeds += int(bool(res.route or res.ranked_chunks))
            cycles += int(_route_has_cycle(res.routes, res.route))
            ptrs: list[tuple[ChunkPointer, float]] = []
            seen_sp: set[tuple[str, int, int]] = set()
            for cid, sc in res.ranked_chunks:
                if (ptr := ds.ptr_by_cid.get(cid)) and (
                    sk := (ptr.rel_posix_path, ptr.start_line, ptr.end_line)
                ) not in seen_sp:
                    seen_sp.add(sk)
                    ptrs.append((ptr, sc))
                    if len(ptrs) >= pool_lim:
                        break
            mkey = (qi, tuple((p.chunk_id, round(float(s), 9)) for p, s in ptrs))
            if (cached := ds.recall_memo.get(mkey)) is None:
                hyd = hydrate_budgeted_passages(
                    ptrs,
                    corpus_root=ds.bundle_root,
                    max_context_chars=ds.max_context_chars,
                    max_expansion_hops=ds.max_ast_expansion_hops,
                )
                seen_t, texts = set(), []
                for hp in hyd:
                    if (nt := hp.text.strip()) and nt not in seen_t:
                        seen_t.add(nt)
                        texts.append(hp.text)
                        if len(texts) >= max_k:
                            break
                rec = compute_retrieval_recalls(ds.gold_docs[qi], texts, k_list=_EVAL_KS)
                cached = tuple(float(rec.get(k, 0.0)) for k in _EVAL_KS)
                ds.recall_memo[mkey] = cached
            q_rec += np.asarray(cached, dtype=np.float64)
        ds_recalls.append(q_rec / max(1, len(q_idx_range)))
    mean_r = np.mean(ds_recalls, axis=0) if ds_recalls else np.zeros(len(_EVAL_KS), dtype=np.float64)
    out: dict[str, Any] = {f"r{k}": float(v) for k, v in zip(_EVAL_KS, mean_r, strict=True)}
    out.update({"combined": float(np.mean(mean_r)), "cycles": cycles, "seeds": seeds, "queries": queries})
    return out


def _minimal_overrides(params: ManifoldHyperparams) -> dict[str, int | float]:
    return {
        f.name: int(v) if type(getattr(_DEFAULT_HP, f.name)) is int else float(v)
        for f in fields(ManifoldHyperparams)
        if (v := getattr(params, f.name)) != getattr(_DEFAULT_HP, f.name)
    }


def _is_feasible(trial: optuna.trial.FrozenTrial) -> bool:
    return bool(trial.constraints) and all(float(v) <= 0.0 for v in trial.constraints.values())


def _ped_anova_importances(study: optuna.Study) -> dict[str, float]:
    zeros = {s.name: 0.0 for s in _PARAM_SPECS}
    done = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values and len(t.values) == len(_EVAL_KS)
    ]
    targets = [-float(sum(float(x) for x in v)) / len(_EVAL_KS) for t in done if (v := t.values) is not None]
    if len(done) < 2 or float(np.std(targets)) <= 1e-12:
        return zeros
    if not any(
        len(vs) == len(done) and float(np.std(vs)) > 1e-12
        for s in _PARAM_SPECS
        if (vs := [float(t.params[s.name]) for t in done if s.name in t.params])
    ):
        return zeros
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            raw = optuna.importance.get_param_importances(
                study,
                evaluator=optuna.importance.PedAnovaImportanceEvaluator(),
                target=lambda t: -float(sum(float(x) for x in t.values)) / len(_EVAL_KS) if t.values else 0.0,
            )
        return {s.name: float(raw.get(s.name, 0.0)) for s in _PARAM_SPECS}
    except ValueError, RuntimeError, ZeroDivisionError:
        return zeros


def _select_candidate(
    study: optuna.Study, baseline_train: Mapping[str, Any], baseline_val: Mapping[str, Any]
) -> tuple[str, ManifoldHyperparams, dict[str, Any], dict[str, Any]]:
    best = (
        round(0.40 * float(baseline_train["combined"]) + 0.60 * float(baseline_val["combined"]), 12),
        round(float(baseline_val["combined"]), 12),
        round(float(baseline_val["r5"]), 12),
        0,
        "defaults",
        _DEFAULT_HP,
        dict(baseline_train),
        dict(baseline_val),
    )
    seen: set[tuple[tuple[str, int | float], ...]] = {()}
    for tr in [*study.best_trials, *study.trials]:
        if tr.state != optuna.trial.TrialState.COMPLETE or not _is_feasible(tr):
            continue
        tm, vm = tr.user_attrs.get("train_metrics"), tr.user_attrs.get("val_metrics")
        if not isinstance(tm, Mapping) or not isinstance(vm, Mapping):
            continue
        if int(vm.get("cycles", 0)) > 0 or int(vm.get("seeds", 0)) < int(vm.get("queries", 0)):
            continue
        if any(float(vm[f"r{k}"]) < float(baseline_val[f"r{k}"]) - 1e-12 for k in _EVAL_KS):
            continue
        params = ManifoldHyperparams.from_overrides(tr.params)
        if (key := tuple(sorted((ov := _minimal_overrides(params)).items()))) in seen:
            continue
        seen.add(key)
        cand = (
            round(0.40 * float(tm["combined"]) + 0.60 * float(vm["combined"]), 12),
            round(float(vm["combined"]), 12),
            round(float(vm["r5"]), 12),
            -len(ov),
            "optuna_pareto",
            params,
            dict(tm),
            dict(vm),
        )
        if cand[:4] > best[:4]:
            best = cand
    src, params, tm, vm = best[4], best[5], best[6], best[7]
    if src == "defaults" or not _minimal_overrides(params):
        src = "baseline_fallback"
        sys.stderr.write(
            "[tune-manifold] WARNING: baseline_fallback — no trial cleared all horizons; retaining defaults.\n"
        )
        sys.stderr.flush()
    return src, params, tm, vm


def _prepare_datasets(
    cfg: YawConfig,
    *,
    datasets: Sequence[str],
    query_offset: int,
    max_queries: int,
    train_queries: int,
    index_queries: int,
    index_root: Path | None,
    fresh_indexes: bool,
    program: YawProgram | None,
    embedder: DSPyEmbeddingAdapter | None,
) -> list[_PreparedDataset]:
    assert cfg.workspace_root is not None
    ws_root = Path(cfg.workspace_root).resolve()
    active_prog = program or YawProgram(
        orchestrator_lm=create_dspy_lm(
            model_name=cfg.qa_model,
            diagnostic_dir=ws_root / "outputs" / "inference_failures",
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            provider=cfg.provider,
            context_window_tokens=cfg.context_window_tokens,
            max_tokens=cfg.max_output_tokens,
            role="qa",
        )
    )
    active_emb = embedder if embedder is not None else DSPyEmbeddingAdapter.from_config(cfg)
    m_cfg = replace(cfg, retriever_mode="manifold")
    prepared: list[_PreparedDataset] = []
    for ds_arg in datasets:
        bundle = resolve_dataset_bundle(ds_arg, workspace_root=ws_root)
        samples, doc_rows = bundle.read_inputs()
        c_rows = [dict(r, idx=e.document_id) for e, r in zip(bundle.documents, doc_rows, strict=True)]
        docs, queries, gold_docs, _ = build_benchmark_slice(
            bundle.dataset,
            samples,
            c_rows,
            max_queries=max_queries,
            query_offset=query_offset,
            full_corpus=False,
            index_queries=index_queries,
        )
        if gold_docs is None:
            raise ValueError(f"Dataset {bundle.dataset!r} does not define gold_docs for retrieval tuning.")
        if len(queries) <= train_queries:
            raise ValueError(f"Dataset {bundle.dataset!r} produced {len(queries)} queries; need > {train_queries}.")
        meta = {
            "dataset": bundle.dataset,
            "docs": len(docs),
            "first_doc": docs[0][:64] if docs else "",
            "queries": len(queries),
            "offset": query_offset,
            "full_corpus": False,
        }
        shash = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        db_path = Path(m_cfg.for_benchmark(bundle.dataset, shash, index_root=index_root, fresh=False).db_path or "")
        if fresh_indexes or not _db_has_manifold(db_path, len(docs)):
            YawRuntime(m_cfg, program=program, embedder=embedder, auto_poll=False).benchmark(
                datasets=str(bundle.root),
                max_queries=max_queries,
                index_queries=index_queries,
                query_offset=query_offset,
                mode="retrieval",
                retriever="manifold",
                index_root=index_root,
                fresh=fresh_indexes,
            )
        engine = YawGraphEngine(
            db_path=db_path,
            workspace_root=bundle.root,
            program=active_prog,
            embedder=active_emb,
            retriever_mode="manifold",
            manifold_params=_DEFAULT_HP,
            max_context_chars=cfg.max_context_chars,
            max_ast_expansion_hops=cfg.max_ast_expansion_hops,
            ast_context_hops=cfg.ast_context_hops,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            min_chunk_size=cfg.min_chunk_size,
        )
        try:
            c_data = engine.load_manifold_corpus()
            ptr_map = get_chunk_pointers(engine.conn, [p.chunk_id for p in c_data.chunk_pointers])
            q_vecs = np.asarray(
                active_emb.batch_encode(queries, instruction=PASSAGE_INSTRUCTION, stage="query_passage_embed"),
                dtype=np.float32,
            )
            fts_hits = [dict(search_chunks_fts(engine.conn, q, top_k=_FTS_K)) for q in queries]
            vec_hits = [
                dict(search_chunk_vectors_knn(engine.conn, q_vecs[i], top_k=_VEC_K)) for i in range(len(queries))
            ]
        finally:
            engine.close()
        man = build_substate_manifold(
            c_data.chunk_pointers, c_data.chunk_texts, c_data.chunk_vectors, params=_DEFAULT_HP
        )
        prepared.append(
            _PreparedDataset(
                bundle.dataset,
                bundle.root,
                ptr_map,
                queries,
                gold_docs,
                q_vecs,
                fts_hits,
                vec_hits,
                train_queries,
                cfg.max_context_chars,
                cfg.max_ast_expansion_hops,
                man,
                {},
            )
        )
    return prepared


def tune_manifold(
    config: YawConfig,
    *,
    datasets: Sequence[str] | str | None = None,
    query_offset: int = 10,
    max_queries: int = 20,
    train_queries: int = 12,
    index_queries: int = 20,
    trials: int = 64,
    study_db: Path | str = DEFAULT_STUDY_DB,
    study_name: str = DEFAULT_STUDY_NAME,
    seed: int = 42,
    fresh: bool = False,
    fresh_indexes: bool = False,
    apply: bool = False,
    allow_eval_overlap: bool = False,
    output_path: Path | str | None = None,
    config_path: Path | str | None = None,
    index_root: Path | str | None = None,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> dict[str, Any]:
    """Run persistent single-study Optuna TPESampler tuning across all 5 Recall@{1,2,5,10,20} horizons."""
    if query_offset < _MIN_DISJOINT_QUERY_OFFSET and not allow_eval_overlap:
        raise ValueError(
            f"query_offset={query_offset} overlaps held-out evaluation range [0, 10); pass allow_eval_overlap=True."
        )
    if max_queries < 2 or not 0 < train_queries < max_queries:
        raise ValueError(f"Expected 0 < train_queries < max_queries >= 2, got {train_queries}/{max_queries}")
    if index_queries < max_queries or trials < 0:
        raise ValueError("Expected index_queries >= max_queries and trials >= 0.")

    raw_ds = parse_dataset_list(datasets)
    selected_datasets = tuple(raw_ds) if raw_ds else DEFAULT_TUNE_DATASETS
    assert config.workspace_root is not None
    ws_root = Path(config.workspace_root).resolve()
    resolved_db = (Path(study_db) if Path(study_db).is_absolute() else ws_root / study_db).resolve()
    resolved_db.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        for p in (resolved_db, Path(f"{resolved_db}-wal"), Path(f"{resolved_db}-shm")):
            p.unlink(missing_ok=True)

    prepared = _prepare_datasets(
        config,
        datasets=selected_datasets,
        query_offset=query_offset,
        max_queries=max_queries,
        train_queries=train_queries,
        index_queries=index_queries,
        index_root=Path(index_root) if index_root else None,
        fresh_indexes=fresh_indexes,
        program=program,
        embedder=embedder,
    )
    default_mans = [ds.default_manifold for ds in prepared]
    baseline_train = _eval_split(prepared, default_mans, _DEFAULT_HP, "train")
    baseline_val = _eval_split(prepared, default_mans, _DEFAULT_HP, "val")
    sys.stderr.write(
        f"[tune-manifold] baseline train={baseline_train['combined']:.4f} "
        f"val={baseline_val['combined']:.4f} (datasets={len(prepared)})\n"
    )
    sys.stderr.flush()

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{resolved_db.as_posix()}")
    prev_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=not fresh,
            directions=["maximize"] * len(_EVAL_KS),
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True),
        )

        def _objective(trial: optuna.trial.Trial) -> tuple[float, float, float, float, float]:
            sampled = {s.name: trial.suggest_float(s.name, float(s.low), float(s.high)) for s in _PARAM_SPECS}
            params = ManifoldHyperparams.from_overrides(sampled)
            train_m = _eval_split(prepared, default_mans, params, "train")
            val_m = _eval_split(prepared, default_mans, params, "val")
            constraints = {
                "cycles": float(train_m["cycles"] + val_m["cycles"]),
                "seed_deficit": float((train_m["queries"] - train_m["seeds"]) + (val_m["queries"] - val_m["seeds"])),
                **{f"val_r{k}": float(baseline_val[f"r{k}"]) - float(val_m[f"r{k}"]) - 1e-12 for k in _EVAL_KS},
            }
            for ck, cv in constraints.items():
                trial.set_constraint(ck, cv)
            trial.set_user_attr("train_metrics", train_m)
            trial.set_user_attr("val_metrics", val_m)
            trial.set_user_attr("overrides", _minimal_overrides(params))
            r_str = "/".join(f"{train_m[f'r{k}']:.3f}" for k in _EVAL_KS)
            sys.stderr.write(
                f"[tune-manifold] trial={trial.number} train_r1..20={r_str} "
                f"val={val_m['combined']:.4f} feasible={all(v <= 0.0 for v in constraints.values())}\n"
            )
            sys.stderr.flush()
            tr = train_m
            return float(tr["r1"]), float(tr["r2"]), float(tr["r5"]), float(tr["r10"]), float(tr["r20"])

        if not study.trials and trials > 0:
            study.enqueue_trial({s.name: s.default for s in _PARAM_SPECS})
        done_before = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if (rem := max(0, trials - done_before)) > 0:
            study.optimize(_objective, n_trials=rem, show_progress_bar=False)

        done_n = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        feas_n = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and _is_feasible(t))
        total_n, pareto_n = len(study.trials), len(study.best_trials)
        importances = _ped_anova_importances(study)
        win_src, win_params, win_train, win_val = _select_candidate(study, baseline_train, baseline_val)
    finally:
        storage.engine.dispose()
        optuna.logging.set_verbosity(prev_verbosity)

    best_ov = _minimal_overrides(win_params)
    app_p = write_manifold_section(best_ov, path=config_path, workspace_override=ws_root) if apply else None
    best_json = (ws_root / DEFAULT_BEST_JSON).resolve()
    payload: dict[str, Any] = {
        "datasets": [ds.dataset for ds in prepared],
        "query_offset": query_offset,
        "max_queries": max_queries,
        "train_queries": train_queries,
        "val_queries": max_queries - train_queries,
        "index_queries": index_queries,
        "study_db": str(resolved_db),
        "study_name": study_name,
        "applied": bool(apply),
        "applied_config_path": str(app_p) if app_p else None,
        "best_source": win_src,
        "best_overrides": best_ov,
        "baseline": {"train": baseline_train, "val": baseline_val},
        "study": {
            "trials_total": total_n,
            "completed_trials": done_n,
            "feasible_trials": feas_n,
            "pareto_count": pareto_n,
            "specs": [asdict(s) for s in _PARAM_SPECS],
            "param_importances": importances,
        },
        "selected": {
            "selected_source": win_src,
            "train": win_train,
            "val": win_val,
            "selection_score": float(0.40 * float(win_train["combined"]) + 0.60 * float(win_val["combined"])),
        },
    }
    serialized = json.dumps(payload, indent=2) + "\n"
    write_text_atomic(best_json, serialized)
    if output_path is not None:
        out_p = (Path(output_path) if Path(output_path).is_absolute() else ws_root / output_path).resolve()
        if out_p != best_json:
            write_text_atomic(out_p, serialized)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Construct standalone CLI parser for single-study SubStateManifold Optuna tuning."""
    p = argparse.ArgumentParser(description="Tune SubStateManifold hyperparameters.", allow_abbrev=False)
    p.add_argument("--config", type=str, default=None, help="Path to yaw.toml")
    p.add_argument("--dataset", nargs="*", default=None, help="Datasets to tune on")
    p.add_argument("--query-offset", type=int, default=10, help="Starting query offset")
    p.add_argument("--max-queries", type=int, default=20, help="Total queries per dataset")
    p.add_argument("--train-queries", type=int, default=12, help="Training queries per dataset")
    p.add_argument("--index-queries", type=int, default=20, help="Queries pooled for corpus indexing")
    p.add_argument("--trials", type=int, default=64, help="Optuna TPESampler trial budget")
    p.add_argument("--study-db", type=str, default=DEFAULT_STUDY_DB, help="SQLite study DB path")
    p.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME, help="Optuna study name prefix")
    p.add_argument("--seed", type=int, default=42, help="Deterministic sampler seed")
    p.add_argument("--fresh", action="store_true", help="Reset Optuna study database")
    p.add_argument("--fresh-indexes", action="store_true", help="Rebuild cached manifold indexes")
    p.add_argument("--apply", action="store_true", help="Persist winning overrides to [manifold]")
    p.add_argument("--allow-eval-overlap", action="store_true", help="Allow query_offset < 10")
    p.add_argument("--output", type=str, default=None, help="Path to write tuning summary JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for scripts/tune_manifold.py."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg_p = Path(args.config).resolve() if args.config else None
    result = tune_manifold(
        load_config(cfg_p),
        datasets=args.dataset,
        query_offset=args.query_offset,
        max_queries=args.max_queries,
        train_queries=args.train_queries,
        index_queries=args.index_queries,
        trials=args.trials,
        study_db=args.study_db,
        study_name=args.study_name,
        seed=args.seed,
        fresh=args.fresh,
        fresh_indexes=args.fresh_indexes,
        apply=args.apply,
        allow_eval_overlap=args.allow_eval_overlap,
        output_path=args.output,
        config_path=cfg_p,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


__all__ = [
    "DEFAULT_BEST_JSON",
    "DEFAULT_STUDY_DB",
    "DEFAULT_STUDY_NAME",
    "DEFAULT_TUNE_DATASETS",
    "build_parser",
    "main",
    "tune_manifold",
]
