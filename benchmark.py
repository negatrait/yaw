"""Live Native YAW VectorSearch & ReAct QA benchmark runner.

Tier 0 (sample):
  uv run --locked python benchmark.py --dataset sample --mode all
Tier 1 (calibration slice with pooled candidate passages):
  uv run --locked python benchmark.py --dataset musique --max-queries 10
  uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 10
  uv run --locked python benchmark.py --dataset hotpotqa --max-queries 10
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import dspy
import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation import (  # noqa: E402
    QAExactMatch,
    QAF1Score,
    RetrievalRecall,
    check_benchmark_integrity,
)
from yaw.core import (  # noqa: E402
    EmbedFn,
    YawConfig,
    configure_logging,
    format_doc_for_embedding,
    format_query_for_embedding,
    load_config,
    redact_secrets,
    write_text_atomic,
)
from yaw.datasets import (  # noqa: E402
    DATASET_REGISTRY,
    DATASET_SOURCE_FIELDS,
    DATASETS,
    PINNED_COMMIT,
    build_benchmark_slice,
    document_identity,
    document_text,
    get_gold_answers,
    get_gold_docs,
    resolve_dataset_bundle,
    resolve_slice_corpus_scope,
)
from yaw.datasets import (  # noqa: E402
    load_dataset_files as _datasets_load_dataset_files,
)
from yaw.pipeline import (  # noqa: E402
    close_pipeline,
    make_dspy_embed_fn,
    start_pipeline,
    update_pipeline_once,
)
from yaw.qa import (  # noqa: E402
    ask_question,
    build_qa_agent,
    create_lm,
    make_qa_tools,
    strip_inline_citations,
)
from yaw.retrieval import (  # noqa: E402
    build_vector_search,
    hydrate_ranked_passages,
    open_readonly_db,
    resolve_retrieval_params,
    search_vectors,
)

__all__ = [
    "BASELINES",
    "DATASET_REGISTRY",
    "DATASETS",
    "PINNED_COMMIT",
    "PROJECT_ROOT",
    "QA_TOP_K",
    "RETRIEVAL_TOP_K",
    "benchmark_config_fingerprint",
    "build_benchmark_slice",
    "check_benchmark_integrity",
    "clean_qa_answer",
    "get_gold_answers",
    "get_gold_docs",
    "load_dataset_files",
    "main",
    "print_benchmark_summary",
    "run_yaw_benchmark",
    "verify_embeddings",
]

BASELINES: dict[str, dict[str, Any]] = json.loads(
    (PROJECT_ROOT / "data" / "benchmarks" / "baselines.json").read_text(encoding="utf-8")
)["baselines"]

QA_TOP_K = 5
RETRIEVAL_TOP_K = 20
_EMBEDDING_TASK_MODES: tuple[str, ...] = ("search", "titled", "qa", "distinct", "distinct-titled", "fact")


def benchmark_config_fingerprint(cfg: YawConfig) -> str:
    """Return a deterministic SHA-256 fingerprint over live index and runtime config fields."""
    payload = {
        "corpus_root": str(cfg.corpus_root) if cfg.corpus_root is not None else None,
        "included_patterns": list(cfg.included_patterns),
        "excluded_patterns": list(cfg.excluded_patterns),
        "chunk_size": int(cfg.chunk_size),
        "chunk_overlap": int(cfg.chunk_overlap),
        "min_chunk_size": int(cfg.min_chunk_size),
        "embedding_model": cfg.embedding_model,
        "embedding_task_mode": cfg.embedding_task_mode,
        "qa_model": cfg.qa_model,
        "max_output_tokens": cfg.max_output_tokens,
        "react_max_iters": int(cfg.react_max_iters),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def clean_qa_answer(raw_ans: str) -> str:
    """Normalize extracted QA answer by stripping conversational prefixes and enclosing quotes."""
    ans = str(raw_ans or "").strip()
    if "Answer:" in ans:
        ans = ans.split("Answer:")[-1].strip()
    if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
        ans = ans[1:-1].strip()
    return ans


def load_dataset_files(
    dataset_name: str,
    dataset_dir: Path | None = None,
    *,
    max_queries: int | None = None,
    include_corpus: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load dataset and (when needed) corpus JSON files from local cache."""
    return _datasets_load_dataset_files(
        dataset_name,
        dataset_dir,
        max_queries=max_queries,
        include_corpus=include_corpus,
        project_root=PROJECT_ROOT,
    )


def _encode_normalized(embed_fn: EmbedFn, texts: list[str]) -> NDArray[np.float32]:
    arr = np.asarray(embed_fn(texts), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.all(np.isfinite(arr)):
        raise RuntimeError("Embedding output contains zero-norm or non-finite vectors")
    return (arr / norms).astype(np.float32, copy=False)


def _split_doc_for_probe(doc: str) -> tuple[str, str]:
    if "\n" in doc:
        title, body = doc.split("\n", 1)
        return title.strip(), body
    return "", doc


def verify_embeddings(
    embed_fn: EmbedFn,
    queries: Sequence[str],
    docs: Sequence[str],
    *,
    task_mode: str = "distinct-titled",
) -> list[dict[str, Any]]:
    """Five repeats of batch, individual, and reversed inputs across all active formatted roles."""
    probe_queries = [str(q) for q in queries[:3]]
    probe_docs = [str(d) for d in docs[:3]]
    roles: list[tuple[str, list[str]]] = []
    if probe_queries:
        roles.append(("query", [format_query_for_embedding(q, task_mode=task_mode) for q in probe_queries]))
        if task_mode in ("distinct", "distinct-titled"):
            roles.append(("fact_query", [format_query_for_embedding(q, task_mode="fact") for q in probe_queries]))
    if probe_docs:
        untitle_inputs = [
            format_doc_for_embedding(
                doc.split("\n", 1)[-1].replace("\n", " "),
                title="",
                task_mode="search",
            )
            for doc in probe_docs
        ]
        roles.append(("document", untitle_inputs))
        if task_mode in ("titled", "distinct-titled"):
            titled_inputs = [
                format_doc_for_embedding(body, title=title, task_mode=task_mode)
                for title, body in (_split_doc_for_probe(d) for d in probe_docs)
            ]
            roles.append(("titled_document", titled_inputs))

    probes: list[dict[str, Any]] = []
    for role, formatted in roles:
        observations: list[NDArray[np.float32]] = []
        for _ in range(5):
            observations.extend(
                [
                    _encode_normalized(embed_fn, formatted),
                    np.concatenate([_encode_normalized(embed_fn, [text]) for text in formatted], axis=0),
                    _encode_normalized(embed_fn, formatted[::-1])[::-1],
                ]
            )
        samples = np.stack(observations)
        cosine = np.einsum("rid,sid->irs", samples, samples)
        minimum = float(cosine.min())
        probes.append(
            {
                "role": role,
                "inputs": formatted,
                "repeats": 5,
                "dimensions": int(samples.shape[-1]),
                "min_pairwise_cosine": minimum,
            }
        )
        if minimum < 0.999:
            raise RuntimeError(f"{role} embeddings are unstable: min cosine {minimum:.6f} < 0.999")
    return probes


def print_benchmark_summary(
    dataset: str,
    num_docs: int,
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    print(f"\n{'=' * 72}")
    print(f"=== Benchmark Summary [NATIVE YAW ENGINE]: {dataset} ({len(records)} queries, {num_docs} passages) ===")
    print(f"{'=' * 72}")
    print("Native YAW Retrieval:   ", metrics["hipporag"]["retrieval"])
    print("Standard DPR Retrieval: ", metrics["standard_dpr"]["retrieval"])
    if metrics["hipporag"]["qa"] is not None:
        print("Native YAW QA:          ", metrics["hipporag"]["qa"])
    print("Delta (YAW - DPR):      ", metrics["delta_vs_dpr"])
    print("Recognition Memory:     ", metrics["recognition_memory"])
    if "monkeypatched_hipporag_baseline" in metrics:
        print(
            "Target 1 (Monkeypatched Baseline):",
            metrics["monkeypatched_hipporag_baseline"],
        )
    if "paper_baselines" in metrics and "hipporag2_llama3.3_70b_nvembed_v2" in metrics["paper_baselines"]:
        print(
            "Target 2 (Paper Reference arXiv:2502.14802):",
            metrics["paper_baselines"]["hipporag2_llama3.3_70b_nvembed_v2"],
        )
    print(f"{'=' * 72}\n", flush=True)


def run_yaw_benchmark(
    *,
    dataset: str,
    document_ids: Sequence[str],
    queries: Sequence[str],
    gold_docs: list[list[str]] | None,
    gold_answers: list[list[str]],
    config: YawConfig,
    include_qa: bool = True,
    compiled_program: str | Path | None = None,
    index_dir: Path,
    report: dict[str, Any],
    embed_fn: EmbedFn | None = None,
    lm: Any = None,
) -> dict[str, Any]:
    """Execute live benchmark directly over prepared v2 Markdown bundles, VectorSearch, and ReAct QA."""
    bundle = resolve_dataset_bundle(dataset, workspace_root=PROJECT_ROOT)
    selected_ids = list(dict.fromkeys(document_ids)) if document_ids else [d.document_id for d in bundle.documents]
    corpus_dir, slice_patterns = resolve_slice_corpus_scope([(bundle, selected_ids)])
    num_docs = len(selected_ids)

    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "yaw.db"

    index_start = perf_counter()
    active_embed = embed_fn or make_dspy_embed_fn(config)
    idx_cfg = replace(
        config,
        db_path=db_path,
        corpus_root=corpus_dir,
        included_patterns=slice_patterns,
        state_dir=index_dir / "coco_lmdb",
        lock_path=index_dir / "yaw.db.lock",
    )

    session = start_pipeline(idx_cfg, embed_fn=active_embed, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)

    report["indexing_seconds"] = round(perf_counter() - index_start, 2)

    with contextlib.closing(open_readonly_db(db_path)) as conn:
        vs = build_vector_search(conn)
        num_chunks = vs.size
        has_chunks = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
        num_indexed_docs = (
            int(conn.execute("SELECT COUNT(DISTINCT rel_path) FROM chunks").fetchone()[0]) if has_chunks else 0
        )
        report["graph"] = {
            "num_passage_nodes": num_chunks,
            "num_indexed_docs": num_indexed_docs,
            "num_entity_nodes": len(vs.entity_keys),
            "num_edges": int(vs.entity_to_chunk.nnz),
        }

        search_params = resolve_retrieval_params(config)
        eval_start = perf_counter()
        records: list[dict[str, Any]] = []
        all_retrieved_passages: list[list[str]] = []
        all_dpr_passages: list[list[str]] = []
        retrieved_ids_list: list[list[int]] = []
        qa_answers: list[str] = []
        timing_records: list[tuple[float, float]] = []

        if queries:
            formatted_qs = [format_query_for_embedding(q, task_mode=config.embedding_task_mode) for q in queries]
            q_vecs = _encode_normalized(active_embed, formatted_qs)
        else:
            q_vecs = np.zeros((0, 0), dtype=np.float32)

        agent = None
        qa_lm = None
        if include_qa:
            qa_lm = lm or create_lm(config)
            tools = make_qa_tools(
                vs,
                corpus_dir,
                active_embed,
                conn,
                task_mode=config.embedding_task_mode,
                search_params=search_params,
            )
            compiled_path = Path(compiled_program) if compiled_program else None
            agent = build_qa_agent(tools, max_iters=config.react_max_iters, artifact_path=compiled_path)

        pool_k = max(RETRIEVAL_TOP_K * 4, 50)
        total_queries = len(queries)
        for q_idx, query in enumerate(queries):
            q_unit = q_vecs[q_idx]
            t_q0 = perf_counter()
            ranked = search_vectors(vs, q_unit, query_text=query, conn=conn, top_k=pool_k, **search_params)
            hydrated = hydrate_ranked_passages(vs, ranked, corpus_dir, top_k=RETRIEVAL_TOP_K)
            retrieval_elapsed = perf_counter() - t_q0

            if vs.size > 0:
                dpr_scores = (vs.embeddings @ q_unit).astype(np.float32, copy=False)
                dpr_k = min(vs.size, pool_k)
                if dpr_k < vs.size:
                    top_pos = np.argpartition(-dpr_scores, dpr_k - 1)[:dpr_k]
                    top_pos = top_pos[np.argsort(-dpr_scores[top_pos], kind="stable")]
                else:
                    top_pos = np.argsort(-dpr_scores, kind="stable")
                dpr_ranked = [(vs.chunk_ids[int(i)], float(dpr_scores[int(i)])) for i in top_pos]
                dpr_hydrated = hydrate_ranked_passages(vs, dpr_ranked, corpus_dir, top_k=RETRIEVAL_TOP_K)
            else:
                dpr_hydrated = ()

            retrieved_passages = [hp.text for hp in hydrated if not hp.stale_on_disk]
            dpr_passages = [hp.text for hp in dpr_hydrated if not hp.stale_on_disk]
            retrieved_ids = [int(hp.pointer.chunk_id) for hp in hydrated]
            all_retrieved_passages.append(retrieved_passages)
            all_dpr_passages.append(dpr_passages)
            retrieved_ids_list.append(retrieved_ids)

            qa_answer = ""
            qa_elapsed = 0.0
            if include_qa and agent is not None and qa_lm is not None:
                t_qa0 = perf_counter()
                with dspy.context(lm=qa_lm):
                    qa_res = ask_question(query, vs=vs, corpus_root=corpus_dir, agent=agent)
                qa_answer = clean_qa_answer(strip_inline_citations(qa_res.answer))
                qa_elapsed = perf_counter() - t_qa0
                qa_answers.append(qa_answer)

            timing_records.append((retrieval_elapsed, qa_elapsed))
            if (q_idx + 1) % 5 == 0 or (q_idx + 1) == total_queries:
                print(
                    f"[Native YAW] [{dataset}] Query {q_idx + 1}/{total_queries} completed "
                    f"(retrieval={retrieval_elapsed:.2f}s, qa={qa_elapsed:.2f}s)",
                    flush=True,
                )

    if gold_docs is not None:
        gold_docs_norm = [[d.strip() for d in gd] for gd in gold_docs]
        ret_docs_norm = [[p.strip() for p in rp] for rp in all_retrieved_passages]
        dpr_docs_norm = [[p.strip() for p in dp] for dp in all_dpr_passages]

        _, ret_per_query = RetrievalRecall().calculate_metric_scores(
            gold_docs=gold_docs_norm,
            retrieved_docs=ret_docs_norm,
            k_list=[1, 2, 5, 10, 20],
        )
        _, dpr_per_query = RetrievalRecall().calculate_metric_scores(
            gold_docs=gold_docs_norm,
            retrieved_docs=dpr_docs_norm,
            k_list=[1, 2, 5, 10, 20],
        )
        recalls_1 = [round(float(r["Recall@1"]), 4) for r in ret_per_query]
        recalls_2 = [round(float(r["Recall@2"]), 4) for r in ret_per_query]
        recalls_5 = [round(float(r["Recall@5"]), 4) for r in ret_per_query]
        recalls_10 = [round(float(r["Recall@10"]), 4) for r in ret_per_query]
        recalls_20 = [round(float(r["Recall@20"]), 4) for r in ret_per_query]
        dpr_recalls_1 = [round(float(r["Recall@1"]), 4) for r in dpr_per_query]
        dpr_recalls_2 = [round(float(r["Recall@2"]), 4) for r in dpr_per_query]
        dpr_recalls_5 = [round(float(r["Recall@5"]), 4) for r in dpr_per_query]
        dpr_recalls_10 = [round(float(r["Recall@10"]), 4) for r in dpr_per_query]
        dpr_recalls_20 = [round(float(r["Recall@20"]), 4) for r in dpr_per_query]
    else:
        recalls_1 = recalls_2 = recalls_5 = recalls_10 = recalls_20 = []
        dpr_recalls_1 = dpr_recalls_2 = dpr_recalls_5 = dpr_recalls_10 = dpr_recalls_20 = []

    qa_metrics: dict[str, float] | None = None
    em_list: list[float] = []
    f1_list: list[float] = []
    if include_qa:
        em_overall, per_query_em_raw = QAExactMatch().calculate_metric_scores(gold_answers, qa_answers)
        f1_overall, per_query_f1_raw = QAF1Score().calculate_metric_scores(gold_answers, qa_answers)
        em_list = [float(item["ExactMatch"]) for item in per_query_em_raw]
        f1_list = [float(item["F1"]) for item in per_query_f1_raw]
        qa_metrics = {
            "ExactMatch": round(float(em_overall["ExactMatch"]), 4),
            "F1": round(float(f1_overall["F1"]), 4),
        }

    for q_idx, query in enumerate(queries):
        retrieval_elapsed, qa_elapsed = timing_records[q_idx]
        records.append(
            {
                "query_index": q_idx,
                "question": query,
                "answer": qa_answers[q_idx] if include_qa else "",
                "gold_answers": gold_answers[q_idx],
                "gold_docs": gold_docs[q_idx] if gold_docs is not None else None,
                "qa_evidence": all_retrieved_passages[q_idx][:QA_TOP_K],
                "dpr_evidence": all_dpr_passages[q_idx][:QA_TOP_K],
                "recall": (
                    {
                        "@1": recalls_1[q_idx],
                        "@2": recalls_2[q_idx],
                        "@5": recalls_5[q_idx],
                        "@10": recalls_10[q_idx],
                        "@20": recalls_20[q_idx],
                    }
                    if gold_docs is not None
                    else None
                ),
                "dpr_recall": (
                    {
                        "@1": dpr_recalls_1[q_idx],
                        "@2": dpr_recalls_2[q_idx],
                        "@5": dpr_recalls_5[q_idx],
                        "@10": dpr_recalls_10[q_idx],
                        "@20": dpr_recalls_20[q_idx],
                    }
                    if gold_docs is not None
                    else None
                ),
                "retrieved_chunk_ids": retrieved_ids_list[q_idx],
                "exact_match": em_list[q_idx] if include_qa else None,
                "f1": f1_list[q_idx] if include_qa else None,
                "retrieval_seconds": round(retrieval_elapsed, 4),
                "qa_seconds": round(qa_elapsed, 4),
            }
        )

    report["evaluation_seconds"] = round(perf_counter() - eval_start, 2)
    active_queries = sum(1 for ids in retrieved_ids_list if ids)
    fallback_count = len(records) - active_queries
    if gold_docs is not None:
        hipporag_ret: dict[str, Any] = {
            "Recall@1": round(float(np.mean(recalls_1)), 4) if recalls_1 else 0.0,
            "Recall@2": round(float(np.mean(recalls_2)), 4) if recalls_2 else 0.0,
            "Recall@5": round(float(np.mean(recalls_5)), 4) if recalls_5 else 0.0,
            "Recall@10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0.0,
            "Recall@20": round(float(np.mean(recalls_20)), 4) if recalls_20 else 0.0,
        }
        dpr_ret: dict[str, Any] = {
            "Recall@1": round(float(np.mean(dpr_recalls_1)), 4) if dpr_recalls_1 else 0.0,
            "Recall@2": round(float(np.mean(dpr_recalls_2)), 4) if dpr_recalls_2 else 0.0,
            "Recall@5": round(float(np.mean(dpr_recalls_5)), 4) if dpr_recalls_5 else 0.0,
            "Recall@10": round(float(np.mean(dpr_recalls_10)), 4) if dpr_recalls_10 else 0.0,
            "Recall@20": round(float(np.mean(dpr_recalls_20)), 4) if dpr_recalls_20 else 0.0,
        }
        delta_vs_dpr: dict[str, Any] = {
            "Recall@1": round(hipporag_ret["Recall@1"] - dpr_ret["Recall@1"], 4),
            "Recall@2": round(hipporag_ret["Recall@2"] - dpr_ret["Recall@2"], 4),
            "Recall@5": round(hipporag_ret["Recall@5"] - dpr_ret["Recall@5"], 4),
            "Recall@10": round(hipporag_ret["Recall@10"] - dpr_ret["Recall@10"], 4),
            "Recall@20": round(hipporag_ret["Recall@20"] - dpr_ret["Recall@20"], 4),
        }
    else:
        hipporag_ret = {}
        dpr_ret = {}
        delta_vs_dpr = {}

    metrics: dict[str, Any] = {
        "hipporag": {
            "retrieval": hipporag_ret,
            "qa": qa_metrics,
        },
        "standard_dpr": {
            "retrieval": dpr_ret,
            "qa": None,
        },
        "delta_vs_dpr": delta_vs_dpr,
        "recognition_memory": {
            "queries_with_graph_seeds": active_queries,
            "dpr_fallback_count": fallback_count,
            "dpr_fallback_rate": round(fallback_count / len(records), 4) if records else 0.0,
        },
    }
    ds_baselines = BASELINES.get(dataset, {})
    mp = ds_baselines.get("monkeypatched_hipporag")
    if isinstance(mp, dict):
        metrics["monkeypatched_hipporag_baseline"] = mp
        metrics["delta_vs_monkeypatched_hipporag"] = {
            "Recall@1": (
                round(hipporag_ret["Recall@1"] - float(mp["Recall@1"]), 4)
                if "Recall@1" in hipporag_ret and mp.get("Recall@1") is not None
                else None
            ),
            "Recall@2": (
                round(hipporag_ret["Recall@2"] - float(mp["Recall@2"]), 4)
                if "Recall@2" in hipporag_ret and mp.get("Recall@2") is not None
                else None
            ),
            "Recall@5": (
                round(hipporag_ret["Recall@5"] - float(mp["Recall@5"]), 4)
                if "Recall@5" in hipporag_ret and mp.get("Recall@5") is not None
                else None
            ),
            "ExactMatch": (
                round(float(qa_metrics["ExactMatch"]) - float(mp["ExactMatch"]), 4)
                if qa_metrics is not None and mp.get("ExactMatch") is not None
                else None
            ),
            "F1": (
                round(float(qa_metrics["F1"]) - float(mp["F1"]), 4)
                if qa_metrics is not None and mp.get("F1") is not None
                else None
            ),
        }
    paper_baselines = {k: v for k, v in ds_baselines.items() if k != "monkeypatched_hipporag"}
    if paper_baselines:
        metrics["paper_baselines"] = paper_baselines

    integrity_errors = check_benchmark_integrity(
        records,
        report["graph"],
        expected_doc_count=num_docs,
        include_qa=include_qa,
    )
    report["integrity"] = {"passed": len(integrity_errors) == 0, "errors": integrity_errors}
    report["evaluation_metrics"] = metrics
    report["records"] = records
    print_benchmark_summary(dataset, num_docs, records, metrics)
    return report


def main(
    dataset: str = "sample",
    max_queries: int | None = None,
    index_queries: int | None = None,
    full_corpus: bool = False,
    mode: str = "eval",
    query_offset: int = 0,
    compiled_program: str | None = None,
    qa_model: str | None = None,
    embedding_model: str | None = None,
    embedding_task_mode: str | None = None,
    fresh_index: bool = False,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, workspace_override=PROJECT_ROOT if config_path is None else None)
    config = replace(
        config,
        embedding_model=embedding_model if embedding_model is not None else config.embedding_model,
        qa_model=qa_model if qa_model is not None else config.qa_model,
        embedding_task_mode=embedding_task_mode if embedding_task_mode is not None else config.embedding_task_mode,
    )
    include_qa = mode not in ("embeddings", "retrieval")

    total_needed = None
    if max_queries is not None:
        effective_index_q = index_queries if index_queries is not None else int(max_queries * 1.4)
        pool_needed = max(
            max_queries,
            effective_index_q * 2 if dataset == "popqa" else effective_index_q,
        )
        total_needed = pool_needed + query_offset

    samples, corpus = load_dataset_files(
        dataset,
        max_queries=total_needed,
        include_corpus=True,
    )
    docs, queries, gold_docs, gold_answers = build_benchmark_slice(
        dataset,
        samples,
        corpus,
        max_queries=max_queries,
        query_offset=query_offset,
        full_corpus=full_corpus,
        index_queries=index_queries,
    )
    fields = DATASET_SOURCE_FIELDS.get(dataset, {})
    id_key = fields.get("id_key", "idx")
    title_key = str(fields.get("title_key", "title"))
    body_key = str(fields.get("body_key", "text"))
    by_id = {document_identity(row, id_key=id_key, title_key=title_key, body_key=body_key): row for row in corpus}
    selected_corpus = [by_id[did] for did in docs if did in by_id]
    embedding_probe_docs = [document_text(row, title_key, body_key) for row in selected_corpus[:3]]
    del samples, corpus, by_id
    gc.collect()
    num_docs = len(docs)

    started = perf_counter()
    root = PROJECT_ROOT / "outputs" / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"run-{dataset}-", dir=root))

    slice_hash = hashlib.sha256("\n".join(docs).encode("utf-8")).hexdigest()[:10]
    cfg_fingerprint = benchmark_config_fingerprint(config)
    id_hash = hashlib.sha256(
        f"yaw|{config.embedding_model}|{config.embedding_task_mode}|{cfg_fingerprint}|{config.provider}|{config.base_url}".encode()
    ).hexdigest()[:10]
    index_dir = run_dir / "index" if fresh_index else root / "indexes" / f"{dataset}_d{num_docs}_{slice_hash}_{id_hash}"

    configure_logging(log_file=run_dir / "run.log", max_lines=500)
    report: dict[str, Any] = {
        "status": "running",
        "dataset": dataset,
        "engine": "yaw",
        "query_offset": query_offset,
        "num_queries": len(queries),
        "index_queries": index_queries,
        "num_docs": num_docs,
        "full_corpus": full_corpus,
        "embedding_model": config.embedding_model,
        "embedding_task_mode": config.embedding_task_mode,
        "qa_model": config.qa_model if include_qa else None,
        "max_output_tokens": config.max_output_tokens,
        "react_max_iters": config.react_max_iters,
        "config_fingerprint": cfg_fingerprint,
        "base_url": config.base_url,
        "index_dir": str(index_dir),
        "run_dir": str(run_dir),
        "mode": mode,
    }
    print(
        f"\n[Native YAW] Benchmark run: {run_dir} (dataset={dataset}, queries={len(queries)}, docs={num_docs})",
        flush=True,
    )
    try:
        active_embed = make_dspy_embed_fn(config)
        if mode in ("embeddings", "all"):
            report["embedding_probes"] = verify_embeddings(
                active_embed,
                queries,
                embedding_probe_docs,
                task_mode=config.embedding_task_mode,
            )
            if mode == "embeddings":
                report["status"] = "passed"
                return report
        result = run_yaw_benchmark(
            dataset=dataset,
            document_ids=docs,
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            config=config,
            include_qa=include_qa,
            compiled_program=compiled_program,
            index_dir=index_dir,
            report=report,
            embed_fn=active_embed,
        )
        report["status"] = "passed" if result["integrity"]["passed"] else "failed"
        if not result["integrity"]["passed"]:
            raise RuntimeError(f"Benchmark integrity failed: {result['integrity']['errors']}")
        return result
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = redact_secrets(f"{type(error).__name__}: {error}", config.api_key)
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 2)
        write_text_atomic(
            run_dir / "report.json",
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            newline="\n",
        )
        print(f"\n{report['status'].upper()}: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="sample")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Inference and ingestion settings (default: yaw.toml).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Number of queries to evaluate (pools candidate passages up to --index-queries unless --full-corpus).",
    )
    parser.add_argument(
        "--index-queries",
        type=int,
        default=None,
        help=(
            "Number of queries from which to pool candidate passages for indexing (defaults to int(max_queries * 1.4))."
        ),
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Index the full dataset corpus even when --max-queries is set.",
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "retrieval", "all", "embeddings"),
        default="eval",
        help="Run full benchmark ('eval'), retrieval-only ('retrieval'), include probes ('all'), or probes only.",
    )
    parser.add_argument("--qa-model", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--embedding-task-mode", choices=_EMBEDDING_TASK_MODES, default=None)
    parser.add_argument(
        "--fresh-index",
        action="store_true",
        help="Force indexing in a fresh disposable directory instead of reusing outputs/benchmarks/indexes.",
    )
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="Starting index offset into dataset queries for evaluation.",
    )
    parser.add_argument(
        "--compiled-program",
        type=str,
        default=None,
        help="Path to load compiled DSPy QA ReAct state.",
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        config_path=args.config,
        max_queries=args.max_queries,
        index_queries=args.index_queries,
        full_corpus=args.full_corpus,
        mode=args.mode,
        query_offset=args.query_offset,
        compiled_program=args.compiled_program,
        qa_model=args.qa_model,
        embedding_model=args.embedding_model,
        embedding_task_mode=args.embedding_task_mode,
        fresh_index=args.fresh_index,
    )
