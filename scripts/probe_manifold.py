"""Read-only diagnostic probe for cached pure-chunk SubStateManifold indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from yaw.core import (
    ChunkPointer,
    YawConfig,
    load_config,
    parse_dataset_list,
    write_text_atomic,
)
from yaw.daemon.graph.hmm import (
    ManifoldHyperparams,
    SubStateManifold,
    SubStateResult,
    build_substate_manifold,
    derive_manifold_telemetry,
    run_substate_hmm,
)
from yaw.daemon.graph.inference import (
    PASSAGE_INSTRUCTION,
    DSPyEmbeddingAdapter,
    compute_retrieval_recalls,
    configure_yaw_dspy_cache,
    doc_matches_gold,
)
from yaw.daemon.graph.storage import (
    get_chunk_pointers,
    hydrate_budgeted_passages,
    load_manifold_chunks,
    make_chunk_hydrator,
    open_readonly_yaw_db,
    search_chunk_vectors_knn,
    search_chunks_fts,
)
from yaw.daemon.ingestion import build_benchmark_slice, resolve_dataset_bundle

K_LIST: tuple[int, ...] = (1, 2, 5, 10, 20)
FTS_SEED_POOL_K = 50
VEC_SEED_POOL_K = 200
TOP_K_EVAL = 20
HMM_POOL_K = max(TOP_K_EVAL * 4, 50)
DEFAULT_POOL_LIMIT = 40
DEFAULT_DATASETS: tuple[str, ...] = ("hotpotqa", "2wikimultihopqa", "musique", "popqa")


def _compute_slice_hash(
    dataset: str,
    docs: list[str],
    queries: list[str],
    *,
    offset: int,
    full_corpus: bool,
) -> str:
    payload = {
        "dataset": dataset,
        "docs": len(docs),
        "first_doc": docs[0][:64] if docs else "",
        "queries": len(queries),
        "offset": offset,
        "full_corpus": full_corpus,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _format_hydrated_passage(ptr: ChunkPointer, text: str) -> str:
    if not ptr.title:
        return text
    if text.startswith(f"{ptr.title}\n") or text == ptr.title:
        return text
    return f"{ptr.title}\n{text}"


def _hydrate_ranked_passages(
    conn: Any,
    ranked_chunks: list[tuple[str, float]],
    *,
    corpus_root: Path,
    cfg: YawConfig,
    top_k: int,
    pool_limit: int,
) -> list[str]:
    if not ranked_chunks or top_k <= 0:
        return []
    ranked_cids = [cid for cid, _ in ranked_chunks]
    pointers = get_chunk_pointers(conn, ranked_cids)
    score_by_cid = dict(ranked_chunks)

    ptrs_with_scores: list[tuple[ChunkPointer, float]] = []
    seen_spans: set[tuple[str, int, int]] = set()
    eff_pool = max(pool_limit, top_k * 2, 10)
    for cid in ranked_cids:
        ptr = pointers.get(cid)
        if ptr is None:
            continue
        span_key = (ptr.rel_posix_path, ptr.start_line, ptr.end_line)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        ptrs_with_scores.append((ptr, score_by_cid[cid]))
        if len(ptrs_with_scores) >= eff_pool:
            break

    hydrated = hydrate_budgeted_passages(
        ptrs_with_scores,
        corpus_root=corpus_root,
        max_context_chars=cfg.max_context_chars,
        max_expansion_hops=cfg.max_ast_expansion_hops,
    )
    deduped_texts: list[str] = []
    seen_texts: set[str] = set()
    for hp in hydrated:
        formatted = _format_hydrated_passage(hp.pointer, hp.text)
        norm_text = formatted.strip()
        if not norm_text or norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)
        deduped_texts.append(formatted)
        if len(deduped_texts) >= top_k:
            break
    return deduped_texts


def _summarize_recalls(per_query_recalls: list[dict[int, float]]) -> dict[str, float]:
    n = len(per_query_recalls)
    if n == 0:
        return {f"Recall@{k}": 0.0 for k in K_LIST}
    return {f"Recall@{k}": round(sum(r[k] for r in per_query_recalls) / n, 6) for k in K_LIST}


def _analyze_manifold_structure(manifold: SubStateManifold) -> dict[str, Any]:
    n_chunks = len(manifold.chunk_ids)
    dim = int(manifold.u_chunk_raw.shape[1]) if manifold.u_chunk_raw.ndim == 2 else 0
    ast_siblings_count = sum(int(sibs.size) for sibs in manifold.ast_siblings)
    csls = manifold.csls_chunk
    return {
        "n_chunks": n_chunks,
        "dim": dim,
        "abtt_removed_dims": int(manifold.abtt_removed_dims),
        "rmt_lambda_plus": round(float(manifold.rmt_lambda_plus), 6),
        "spacy_vocab_size": len(manifold.vocab_index),
        "ast_siblings_count": ast_siblings_count,
        "csls_mean": round(float(np.mean(csls)), 6) if csls.size else 0.0,
        "csls_p95": round(float(np.quantile(csls, 0.95)), 6) if csls.size else 0.0,
    }


def _gold_chunk_groups(
    gold_docs_for_query: list[str],
    formatted_chunks: list[str],
) -> list[list[int]]:
    groups: list[list[int]] = []
    for gdoc in dict.fromkeys(gold_docs_for_query):
        if not gdoc.strip():
            continue
        matched = [ci for ci, ctext in enumerate(formatted_chunks) if doc_matches_gold(ctext, gdoc)]
        groups.append(matched)
    return groups


def _count_gold_gt32(
    ranked_chunk_ids: list[str],
    chunk_id_to_idx: dict[str, int],
    gold_groups: list[list[int]],
) -> tuple[int, int]:
    top32 = {chunk_id_to_idx[cid] for cid in ranked_chunk_ids[:32] if cid in chunk_id_to_idx}
    matched_groups = [grp for grp in gold_groups if grp]
    if not matched_groups:
        return 0, 0
    gt32_docs = sum(1 for grp in matched_groups if not any(ci in top32 for ci in grp))
    return gt32_docs, int(gt32_docs > 0)


def _gold_set_hit_fraction(candidate_indices: set[int], gold_groups: list[list[int]]) -> float:
    matched_groups = [grp for grp in gold_groups if grp]
    if not matched_groups:
        return 0.0
    hits = sum(1 for grp in matched_groups if any(ci in candidate_indices for ci in grp))
    return hits / float(len(matched_groups))


def _evaluate_variant(
    *,
    conn: Any,
    manifold: SubStateManifold,
    params: ManifoldHyperparams,
    queries: list[str],
    qvecs: np.ndarray,
    gold_docs: list[list[str]],
    fts_list: list[dict[str, float]],
    vec_list: list[dict[str, float]],
    gold_groups_by_q: list[list[list[int]]],
    chunk_id_to_idx: dict[str, int],
    corpus_root: Path,
    cfg: YawConfig,
    pool_limit: int,
    num_hops: int | None = None,
) -> dict[str, Any]:
    recalls: list[dict[int, float]] = []
    latencies_ms: list[float] = []
    hw_tail_all_zero = 0
    hw_means_accum: list[list[float]] = []
    seed_reach_rates: list[float] = []
    hop0_reach_rates: list[float] = []
    hop1_to_h_reach_rates: list[float] = []
    final_gold_doc_gt32 = 0
    final_query_any_gt32 = 0
    hop0_post_gold_doc_gt32 = 0
    hop0_post_query_any_gt32 = 0
    matched_gold_doc_total = 0

    for q, qv, gd, fts, vec, g_groups in zip(
        queries, qvecs, gold_docs, fts_list, vec_list, gold_groups_by_q, strict=True
    ):
        t0 = time.perf_counter()
        res: SubStateResult = run_substate_hmm(
            manifold,
            q,
            qv,
            top_k=HMM_POOL_K,
            fts_chunk_scores=fts,
            vec_chunk_scores=vec,
            num_hops=num_hops,
            params=params,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        ret_texts = _hydrate_ranked_passages(
            conn,
            list(res.ranked_chunks),
            corpus_root=corpus_root,
            cfg=cfg,
            top_k=TOP_K_EVAL,
            pool_limit=pool_limit,
        )
        recalls.append(compute_retrieval_recalls(gd, ret_texts, k_list=K_LIST))

        hw = [float(w) for w in res.hop_weights]
        for h_i, w in enumerate(hw):
            while len(hw_means_accum) <= h_i:
                hw_means_accum.append([])
            hw_means_accum[h_i].append(w)
        tail = hw[1:]
        if tail and all(w == 0.0 for w in tail):
            hw_tail_all_zero += 1

        seed_indices = {chunk_id_to_idx[cid] for cid in list(fts) + list(vec) if cid in chunk_id_to_idx}
        seed_reach_rates.append(_gold_set_hit_fraction(seed_indices, g_groups))

        hop0_indices: set[int] = set()
        if res.hop_posteriors:
            hop0_vals = np.asarray(res.hop_posteriors[0], dtype=np.float64)
            hop0_order = np.argsort(-hop0_vals, kind="mergesort")
            hop0_indices = {int(ci) for ci in hop0_order[:TOP_K_EVAL]}
            hop0_ranked_cids = [manifold.chunk_ids[int(ci)] for ci in hop0_order]
            h0_doc_gt32, h0_q_gt32 = _count_gold_gt32(hop0_ranked_cids, chunk_id_to_idx, g_groups)
            hop0_post_gold_doc_gt32 += h0_doc_gt32
            hop0_post_query_any_gt32 += h0_q_gt32
        hop0_reach_rates.append(_gold_set_hit_fraction(hop0_indices, g_groups))

        later_indices: set[int] = set(hop0_indices)
        for h_idx, h_post in enumerate(res.hop_posteriors[1:], start=1):
            if h_idx < len(hw) and hw[h_idx] <= 0.0:
                continue
            vals = np.asarray(h_post, dtype=np.float64)
            if float(vals.max(initial=0.0)) <= 0.0:
                continue
            top_later = np.argsort(-vals, kind="mergesort")[:TOP_K_EVAL]
            later_indices.update(int(ci) for ci in top_later if vals[int(ci)] > 0.0)
        hop1_to_h_reach_rates.append(_gold_set_hit_fraction(later_indices, g_groups))

        matched_gold_doc_total += sum(1 for grp in g_groups if grp)
        f_doc_gt32, f_q_gt32 = _count_gold_gt32(
            [cid for cid, _ in res.ranked_chunks],
            chunk_id_to_idx,
            g_groups,
        )
        final_gold_doc_gt32 += f_doc_gt32
        final_query_any_gt32 += f_q_gt32

    n_q = len(queries)
    return {
        "recalls": _summarize_recalls(recalls),
        "hop_weights_mean": [round(float(np.mean(vals)), 6) for vals in hw_means_accum],
        "hw_tail_all_zero_queries": hw_tail_all_zero,
        "silent_hop_rate": round(hw_tail_all_zero / float(n_q), 6) if n_q else 0.0,
        "seed_gold_reach_rate": round(float(np.mean(seed_reach_rates)), 6) if seed_reach_rates else 0.0,
        "hop0_gold_reach_rate": round(float(np.mean(hop0_reach_rates)), 6) if hop0_reach_rates else 0.0,
        "hop1_to_h_gold_reach_rate": round(float(np.mean(hop1_to_h_reach_rates)), 6) if hop1_to_h_reach_rates else 0.0,
        "matched_gold_doc_total": matched_gold_doc_total,
        "gold_doc_gt32_final": final_gold_doc_gt32,
        "query_any_gold_gt32_final": final_query_any_gt32,
        "gold_doc_gt32_hop0_posterior": hop0_post_gold_doc_gt32,
        "query_any_gold_gt32_hop0_posterior": hop0_post_query_any_gt32,
        "latency_ms": {
            "mean_ms": round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
            "median_ms": round(float(np.median(latencies_ms)), 3) if latencies_ms else 0.0,
            "p95_ms": round(float(np.quantile(latencies_ms, 0.95)), 3) if latencies_ms else 0.0,
        },
    }


def _probe_slice(
    *,
    dataset: str,
    samples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    corpus_root: Path,
    cfg: YawConfig,
    embedder: DSPyEmbeddingAdapter,
    index_root: Path,
    query_offset: int,
    max_queries: int,
    index_queries: int,
    pool_limit: int,
    default_params: ManifoldHyperparams,
    active_params: ManifoldHyperparams,
) -> dict[str, Any]:
    docs, queries, gold_docs, _ = build_benchmark_slice(
        dataset,
        samples,
        corpus,
        max_queries=max_queries,
        query_offset=query_offset,
        full_corpus=False,
        index_queries=index_queries,
    )
    if gold_docs is None:
        raise ValueError(f"Dataset {dataset!r} did not provide gold_docs.")
    slice_hash = _compute_slice_hash(
        dataset,
        docs,
        queries,
        offset=query_offset,
        full_corpus=False,
    )
    index_dir = index_root / f"{dataset}_{slice_hash}_manifold"
    db_path = index_dir / "yaw.db"
    cache_dir = index_dir / "dspy_cache"
    if not db_path.exists():
        raise FileNotFoundError(f"Cached manifold index not found: {db_path}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cached dspy_cache not found: {cache_dir}")

    configure_yaw_dspy_cache(cache_dir, restrict_pickle=True, account_cached_usage=True)
    qvecs = embedder.batch_encode(queries, instruction=PASSAGE_INSTRUCTION, stage="query_passage_embed")

    conn = open_readonly_yaw_db(db_path)
    try:
        chunk_ptrs, chunk_vecs = load_manifold_chunks(conn)
        hydrator = make_chunk_hydrator(conn, corpus_root)
        chunk_texts = {ptr.chunk_id: hydrator(ptr.chunk_id) for ptr in chunk_ptrs}
        formatted_chunks = [_format_hydrated_passage(ptr, chunk_texts[ptr.chunk_id]) for ptr in chunk_ptrs]
        chunk_id_to_idx = {ptr.chunk_id: ci for ci, ptr in enumerate(chunk_ptrs)}

        manifold_default = build_substate_manifold(
            chunk_ptrs,
            chunk_texts,
            chunk_vecs,
            params=default_params,
        )
        same_params = active_params == default_params
        manifold_active = (
            manifold_default
            if same_params
            else build_substate_manifold(
                chunk_ptrs,
                chunk_texts,
                chunk_vecs,
                params=active_params,
            )
        )

        fts_list = [dict(search_chunks_fts(conn, q, top_k=FTS_SEED_POOL_K)) for q in queries]
        vec_list = [dict(search_chunk_vectors_knn(conn, qvecs[i], top_k=VEC_SEED_POOL_K)) for i in range(len(queries))]

        n_c = len(chunk_ptrs)
        fts_pos_counts = [sum(1 for sc in fts.values() if sc > 0.0) for fts in fts_list]
        vec_pos_counts = [sum(1 for sc in vec.values() if sc > 0.0) for vec in vec_list]
        vec_hit_recalls: list[dict[int, float]] = []
        gold_groups_by_q: list[list[list[int]]] = []
        for gd, vec in zip(gold_docs, vec_list, strict=True):
            ranked_vec = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)
            ret_vec = _hydrate_ranked_passages(
                conn,
                ranked_vec,
                corpus_root=corpus_root,
                cfg=cfg,
                top_k=TOP_K_EVAL,
                pool_limit=pool_limit,
            )
            vec_hit_recalls.append(compute_retrieval_recalls(gd, ret_vec, k_list=K_LIST))
            gold_groups_by_q.append(_gold_chunk_groups(gd, formatted_chunks))

        default_res = _evaluate_variant(
            conn=conn,
            manifold=manifold_default,
            params=default_params,
            queries=queries,
            qvecs=qvecs,
            gold_docs=gold_docs,
            fts_list=fts_list,
            vec_list=vec_list,
            gold_groups_by_q=gold_groups_by_q,
            chunk_id_to_idx=chunk_id_to_idx,
            corpus_root=corpus_root,
            cfg=cfg,
            pool_limit=pool_limit,
        )
        default_hop0_res = _evaluate_variant(
            conn=conn,
            manifold=manifold_default,
            params=default_params,
            queries=queries,
            qvecs=qvecs,
            gold_docs=gold_docs,
            fts_list=fts_list,
            vec_list=vec_list,
            gold_groups_by_q=gold_groups_by_q,
            chunk_id_to_idx=chunk_id_to_idx,
            corpus_root=corpus_root,
            cfg=cfg,
            pool_limit=pool_limit,
            num_hops=1,
        )
        if same_params:
            active_res = default_res
            active_hop0_res = default_hop0_res
        else:
            active_res = _evaluate_variant(
                conn=conn,
                manifold=manifold_active,
                params=active_params,
                queries=queries,
                qvecs=qvecs,
                gold_docs=gold_docs,
                fts_list=fts_list,
                vec_list=vec_list,
                gold_groups_by_q=gold_groups_by_q,
                chunk_id_to_idx=chunk_id_to_idx,
                corpus_root=corpus_root,
                cfg=cfg,
                pool_limit=pool_limit,
            )
            active_hop0_res = _evaluate_variant(
                conn=conn,
                manifold=manifold_active,
                params=active_params,
                queries=queries,
                qvecs=qvecs,
                gold_docs=gold_docs,
                fts_list=fts_list,
                vec_list=vec_list,
                gold_groups_by_q=gold_groups_by_q,
                chunk_id_to_idx=chunk_id_to_idx,
                corpus_root=corpus_root,
                cfg=cfg,
                pool_limit=pool_limit,
                num_hops=1,
            )

        return {
            "dataset": dataset,
            "query_offset": query_offset,
            "max_queries": max_queries,
            "index_queries": index_queries,
            "slice_hash": slice_hash,
            "doc_count": len(docs),
            "query_count": len(queries),
            "db_path": str(db_path),
            "seed_saturation": {
                "chunk_count": n_c,
                "fts_pos_mean": round(float(np.mean(fts_pos_counts)), 4) if fts_pos_counts else 0.0,
                "fts_pos_max": max(fts_pos_counts, default=0),
                "vec_pos_mean": round(float(np.mean(vec_pos_counts)), 4) if vec_pos_counts else 0.0,
                "vec_pos_max": max(vec_pos_counts, default=0),
            },
            "structure": {
                "default": _analyze_manifold_structure(manifold_default),
                "active_config": _analyze_manifold_structure(manifold_active),
            },
            "variants": {
                "default": default_res,
                "active_config": active_res,
                "default_hop0_only": default_hop0_res,
                "active_hop0_only": active_hop0_res,
                "vec_hits": {"recalls": _summarize_recalls(vec_hit_recalls)},
            },
        }
    finally:
        conn.close()


def _aggregate_slice(datasets_report: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ds_names = list(datasets_report.keys())
    if not ds_names:
        return {}
    variant_names = list(datasets_report[ds_names[0]]["variants"].keys())
    macro_variants: dict[str, Any] = {}
    for vname in variant_names:
        recall_keys = list(datasets_report[ds_names[0]]["variants"][vname]["recalls"].keys())
        recalls_avg = {
            rk: round(sum(datasets_report[ds]["variants"][vname]["recalls"][rk] for ds in ds_names) / len(ds_names), 6)
            for rk in recall_keys
        }
        entry: dict[str, Any] = {"recalls": recalls_avg}
        if "silent_hop_rate" in datasets_report[ds_names[0]]["variants"][vname]:
            entry["silent_hop_rate"] = round(
                sum(datasets_report[ds]["variants"][vname]["silent_hop_rate"] for ds in ds_names) / len(ds_names),
                6,
            )
            entry["mean_ms"] = round(
                sum(datasets_report[ds]["variants"][vname]["latency_ms"]["mean_ms"] for ds in ds_names) / len(ds_names),
                3,
            )
        macro_variants[vname] = entry
    popqa_silent = (
        datasets_report["popqa"]["variants"]["active_config"]["silent_hop_rate"] if "popqa" in datasets_report else None
    )
    return {
        "query_count": sum(datasets_report[ds]["query_count"] for ds in ds_names),
        "popqa_silent_hop_rate": popqa_silent,
        "variants": macro_variants,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only diagnostic probe for pure-chunk SubStateManifold indexes.")
    parser.add_argument("--dataset", default="all", help="Comma-separated datasets or 'all'.")
    parser.add_argument("--offsets", nargs="+", type=int, default=[10, 0], help="Slice query offsets to probe.")
    parser.add_argument("--query-offset", type=int, default=None, help="Single query offset override.")
    parser.add_argument("--max-queries", type=int, default=None, help="Shared max_queries override.")
    parser.add_argument("--max-queries-10", type=int, default=20, help="max_queries for offset=10.")
    parser.add_argument("--max-queries-0", type=int, default=10, help="max_queries for offset=0.")
    parser.add_argument("--index-queries", type=int, default=None, help="Optional explicit index_queries override.")
    parser.add_argument("--pool-limit", type=int, default=DEFAULT_POOL_LIMIT, help="Chunk hydration pool limit.")
    parser.add_argument("--index-root", type=Path, default=Path("outputs/benchmarks/indexes"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/probe_manifold_chunk_only.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    datasets = list(DEFAULT_DATASETS) if args.dataset == "all" else parse_dataset_list(args.dataset)
    offsets = [int(args.query_offset)] if args.query_offset is not None else [int(o) for o in args.offsets]

    default_params = ManifoldHyperparams()
    active_params = ManifoldHyperparams.from_overrides(dict(cfg.manifold_overrides))
    embedder = DSPyEmbeddingAdapter.from_config(cfg)

    slice_specs: list[tuple[int, int, int]] = []
    for off in offsets:
        if args.max_queries is not None:
            mq = int(args.max_queries)
        elif off == 10:
            mq = int(args.max_queries_10)
        elif off == 0:
            mq = int(args.max_queries_0)
        else:
            raise ValueError(f"Specify --max-queries for non-standard offset {off}.")
        iq = int(args.index_queries) if args.index_queries is not None else mq
        slice_specs.append((off, mq, iq))

    reports_by_offset: dict[str, dict[str, dict[str, Any]]] = {str(off): {} for off, _, _ in slice_specs}
    for ds in datasets:
        bundle = resolve_dataset_bundle(ds, workspace_root=cfg.workspace_root)
        samples, doc_rows = bundle.read_inputs()
        corpus = [dict(row, idx=entry.document_id) for entry, row in zip(bundle.documents, doc_rows, strict=True)]
        for off, mq, iq in slice_specs:
            reports_by_offset[str(off)][ds] = _probe_slice(
                dataset=ds,
                samples=samples,
                corpus=corpus,
                corpus_root=bundle.root,
                cfg=cfg,
                embedder=embedder,
                index_root=args.index_root,
                query_offset=off,
                max_queries=mq,
                index_queries=iq,
                pool_limit=int(args.pool_limit),
                default_params=default_params,
                active_params=active_params,
            )

    slices_output: dict[str, Any] = {}
    for off, mq, iq in slice_specs:
        off_key = str(off)
        ds_reports = reports_by_offset[off_key]
        slices_output[off_key] = {
            "query_offset": off,
            "max_queries": mq,
            "index_queries": iq,
            "macro": _aggregate_slice(ds_reports),
            "datasets": ds_reports,
        }

    payload = {
        "schema_version": 2,
        "k_list": list(K_LIST),
        "fts_seed_pool_k": FTS_SEED_POOL_K,
        "vec_seed_pool_k": VEC_SEED_POOL_K,
        "top_k_eval": TOP_K_EVAL,
        "hmm_pool_k": HMM_POOL_K,
        "pool_limit": int(args.pool_limit),
        "manifold_overrides": dict(cfg.manifold_overrides),
        "default_params": asdict(default_params),
        "default_telemetry": asdict(derive_manifold_telemetry(default_params)),
        "active_params": asdict(active_params),
        "active_telemetry": asdict(derive_manifold_telemetry(active_params)),
        "slices": slices_output,
    }
    write_text_atomic(args.output, json.dumps(payload, indent=2) + "\n")
    for off_key, sdata in slices_output.items():
        macro = sdata["macro"]
        print(f"slice offset={off_key} (q={macro['query_count']}):")
        for vname, vdata in macro["variants"].items():
            print(f"  {vname}: {vdata['recalls']}")


if __name__ == "__main__":
    main()
