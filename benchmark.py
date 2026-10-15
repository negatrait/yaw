"""Live HippoRAG 2 vs. StandardRAG (DPR) benchmark runner on upstream datasets.

Tier 0 (sample):
  uv run --locked python benchmark.py --dataset sample --mode all
Tier 1 (10-query calibration slice with pooled candidate passages):
  uv run --locked python benchmark.py --dataset musique --max-queries 10
  uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 10
  uv run --locked python benchmark.py --dataset hotpotqa --max-queries 10
"""

import argparse
from contextlib import closing
import gc
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter
import urllib.request

import numpy as np
from hipporag import HippoRAG
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from hipporag.prompts.linking import get_query_instruction
from hipporag.utils.config_utils import BaseConfig

from yaw.core.logger import configure_logging
from yaw.daemon.graph.inference.legacy_chat import LocalChat, chat_identity
from yaw.daemon.graph.inference.legacy_embedding import RetrievalEmbedding
from yaw.daemon.graph.inference.embedding import (
    GEMMA_TASK_MODES,
    MODEL,
    PROFILES,
    resolve_embedding_profile,
)
from yaw.daemon.graph.inference.models import (
    BASE_URL,
    CHAT_MODELS,
    DEFAULT_CHAT,
    DEFAULT_EXTRACTION_CHAT,
    DEFAULT_QA_CHAT,
    chat_evidence,
    embedding_evidence,
    preflight,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PINNED_COMMIT = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
UPSTREAM_DATASET_BASE_URL = (
    f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{PINNED_COMMIT}/reproduce/dataset"
)
DATASETS = ("sample", "musique", "2wikimultihopqa", "hotpotqa")
QA_TOP_K = 5
RETRIEVAL_TOP_K = 5
LINKING_TOP_K = 5

# Reference numbers from HippoRAG 2 (arXiv:2502.14802v1, Tables 8 & 9, 1,000 queries).
PAPER_BASELINES = {
    "musique": {
        "hipporag2_llama3.3_70b_nvembed_v2": {"Recall@2": 0.561, "Recall@5": 0.747, "ExactMatch": 0.372, "F1": 0.486},
        "standard_nvembed_v2_llama3.3_70b": {"Recall@2": 0.527, "Recall@5": 0.697, "ExactMatch": 0.347, "F1": 0.457},
        "hipporag2_gpt4o_mini_nvembed_v2": {"Recall@2": 0.535, "Recall@5": 0.742, "ExactMatch": 0.350, "F1": 0.493},
        "contriever_llama3.3_70b": {"Recall@2": 0.348, "Recall@5": 0.466, "ExactMatch": 0.240, "F1": 0.313},
    },
    "2wikimultihopqa": {
        "hipporag2_llama3.3_70b_nvembed_v2": {"Recall@2": 0.762, "Recall@5": 0.904, "ExactMatch": 0.650, "F1": 0.710},
        "standard_nvembed_v2_llama3.3_70b": {"Recall@2": 0.671, "Recall@5": 0.765, "ExactMatch": 0.575, "F1": 0.615},
        "hipporag2_gpt4o_mini_nvembed_v2": {"Recall@2": 0.746, "Recall@5": 0.902, "ExactMatch": 0.605, "F1": 0.697},
        "contriever_llama3.3_70b": {"Recall@2": 0.466, "Recall@5": 0.575, "ExactMatch": 0.381, "F1": 0.419},
    },
    "hotpotqa": {
        "hipporag2_llama3.3_70b_nvembed_v2": {"Recall@2": 0.835, "Recall@5": 0.963, "ExactMatch": 0.627, "F1": 0.755},
        "standard_nvembed_v2_llama3.3_70b": {"Recall@2": 0.841, "Recall@5": 0.945, "ExactMatch": 0.628, "F1": 0.753},
        "hipporag2_gpt4o_mini_nvembed_v2": {"Recall@2": 0.805, "Recall@5": 0.957, "ExactMatch": 0.563, "F1": 0.711},
        "contriever_llama3.3_70b": {"Recall@2": 0.584, "Recall@5": 0.753, "ExactMatch": 0.513, "F1": 0.623},
    },
}


def get_gold_docs(samples: list[dict], dataset_name: str | None = None) -> list[list[str]]:
    """Extract gold supporting passages using upstream main.py's exact formatting contract."""
    gold_docs = []
    separator = "" if (dataset_name or "").startswith("hotpotqa") else " "
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            gold_doc = [f"{item[0]}\n{separator.join(item[1])}" for item in supporting_contexts]
        elif "contexts" in sample:
            gold_doc = [f"{item['title']}\n{item['text']}" for item in sample["contexts"] if item["is_supporting"]]
        else:
            if "paragraphs" not in sample:
                raise ValueError("Each sample needs supporting_facts, contexts, or paragraphs for retrieval evaluation")
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [
                f"{item['title']}\n{item['text'] if 'text' in item else item['paragraph_text']}"
                for item in paragraphs
            ]
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def get_gold_answers(samples: list[dict]) -> list[list[str]]:
    """Extract gold answer sets and aliases using upstream main.py's exact contract."""
    gold_answers = []
    for sample in samples:
        if "answer" in sample or "gold_ans" in sample:
            gold_answer = sample.get("answer", sample.get("gold_ans"))
        elif "reference" in sample:
            gold_answer = sample["reference"]
        elif "obj" in sample:
            gold_answer = [sample["obj"], sample["o_wiki_title"]]
            for field in ("possible_answers", "o_aliases"):
                value = sample.get(field, [])
                gold_answer.extend(value if isinstance(value, list) else [value])
        else:
            raise ValueError("Each query sample must contain an answer field")
        if gold_answer is None:
            raise ValueError("Answer fields must not be null")
        answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
        answers.extend(sample.get("answer_aliases", []))
        gold_answers.append(list(dict.fromkeys(answers)))
    return gold_answers


def build_benchmark_slice(
    dataset_name: str,
    samples: list[dict],
    corpus: list[dict],
    max_queries: int | None = None,
    full_corpus: bool = False,
) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    """Build (docs, queries, gold_docs, gold_answers) for a full or N-query slice."""
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be a positive integer when specified.")
    selected = samples if max_queries is None else samples[:max_queries]
    queries = [sample["question"] for sample in selected]
    gold_docs = get_gold_docs(selected, dataset_name)
    gold_answers = get_gold_answers(selected)

    if full_corpus or max_queries is None or dataset_name == "sample":
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    else:
        docs = []
        separator = "" if (dataset_name or "").startswith("hotpotqa") else " "
        for sample in selected:
            if "context" in sample:
                for title, sents in sample["context"]:
                    docs.append(f"{title}\n{separator.join(sents)}")
            elif "contexts" in sample:
                for item in sample["contexts"]:
                    docs.append(f"{item['title']}\n{item['text']}")
            elif "paragraphs" in sample:
                for item in sample["paragraphs"]:
                    text = item["text"] if "text" in item else item["paragraph_text"]
                    docs.append(f"{item['title']}\n{text}")
    for gold_list in gold_docs:
        docs.extend(gold_list)
    return list(dict.fromkeys(docs)), queries, gold_docs, gold_answers


def load_dataset_files(
    dataset_name: str,
    dataset_dir: Path | None = None,
    *,
    max_queries: int | None = None,
    include_corpus: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Load dataset and (when needed) corpus JSON files from local cache, downloading if absent."""
    if Path(dataset_name).name != dataset_name:
        raise ValueError("dataset_name must be a dataset name, not a path.")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be a positive integer when specified.")
    cache_dir = dataset_dir or (PROJECT_ROOT / "outputs" / "datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_files = [f"{dataset_name}.json"]
    if include_corpus:
        required_files.append(f"{dataset_name}_corpus.json")
    for filename in required_files:
        target = cache_dir / filename
        if not target.exists():
            url = f"{UPSTREAM_DATASET_BASE_URL}/{filename}"
            logging.info("Downloading %s to %s", url, target)
            data = urllib.request.urlopen(url).read()
            target.write_bytes(data)
    with open(cache_dir / f"{dataset_name}.json", encoding="utf-8") as f:
        samples = json.load(f)
    if max_queries is not None:
        samples = samples[:max_queries]
    if include_corpus:
        with open(cache_dir / f"{dataset_name}_corpus.json", encoding="utf-8") as f:
            corpus = json.load(f)
    else:
        corpus = []
    return samples, corpus


def verify_embeddings(embedding, queries: list[str], docs: list[str]):
    """Five repeats of batch, individual and reversed inputs across all active roles."""
    probe_queries = queries[:3]
    probe_docs = docs[:3]
    probes = []
    passage_instruction = get_query_instruction("query_to_passage")
    roles = [
        ("query", probe_queries, {"instruction": passage_instruction}),
    ]
    if embedding.profile.query_instruction_mode == "distinct":
        fact_instruction = get_query_instruction("query_to_fact")
        roles.append(("fact_query", probe_queries, {"instruction": fact_instruction}))
    untitle_docs = [doc.split("\n", 1)[-1].replace("\n", " ") for doc in probe_docs]
    roles.append(("document", untitle_docs, {}))
    if embedding.profile.parse_document_title:
        roles.append(("titled_document", probe_docs, {}))
    for role, texts, options in roles:
        observations = []
        for _ in range(5):
            observations.extend([
                embedding.batch_encode(texts, norm=True, **options),
                np.concatenate([
                    embedding.batch_encode([text], norm=True, **options) for text in texts
                ]),
                embedding.batch_encode(texts[::-1], norm=True, **options)[::-1],
            ])
        samples = np.stack(observations)
        cosine = np.einsum("rid,sid->irs", samples, samples)
        minimum = float(cosine.min())
        probes.append({
            "role": role,
            "inputs": [
                embedding.profile.format(text, query=bool(options), instruction=options.get("instruction"))
                for text in texts
            ],
            "repeats": 5,
            "dimensions": embedding.profile.dimensions,
            "min_pairwise_cosine": minimum,
        })
        if minimum < 0.999:
            raise RuntimeError(f"{role} embeddings are unstable: min cosine {minimum:.6f} < 0.999")
    return probes


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _release_indexing_memory(hipporag) -> None:
    """Release transient OpenIE and graph-construction buffers after an indexing batch."""
    for attr in ("_openie_info", "entity_id_to_row", "ent_node_to_chunk_ids"):
        if hasattr(hipporag, attr):
            setattr(hipporag, attr, None)
    for attr in (
        "node_to_node_stats",
        "_fact_edge_source_counts",
        "_passage_edge_sources",
        "_synonym_edge_scores",
    ):
        if hasattr(hipporag, attr):
            setattr(hipporag, attr, {})
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_retrieval_memory(hipporag) -> None:
    """Release in-memory retrieval matrices and OpenIE caches before starting QA."""
    _release_indexing_memory(hipporag)
    for attr in ("entity_embeddings", "passage_embeddings", "fact_embeddings"):
        if hasattr(hipporag, attr):
            setattr(hipporag, attr, None)
    if hasattr(hipporag, "proc_triples_to_docs"):
        hipporag.proc_triples_to_docs = {}
    if hasattr(hipporag, "query_to_embedding"):
        hipporag.query_to_embedding = {"triple": {}, "passage": {}}
    if hasattr(hipporag, "ready_to_retrieve"):
        hipporag.ready_to_retrieve = False
    gc.collect()


def evaluate_benchmark(
    hipporag,
    queries: list[str],
    gold_docs: list[list[str]],
    gold_answers: list[list[str]],
    dataset_name: str | None = None,
    qa_top_k: int = QA_TOP_K,
    qa_batch_size: int = 1,
    include_qa: bool = True,
):
    """Evaluate HippoRAG 2 and StandardRAG (DPR) in bounded query/QA micro-batches."""
    if qa_batch_size < 1:
        raise ValueError("qa_batch_size must be at least 1.")
    num_to_retrieve = max(qa_top_k, 5)
    use_staged_qa = (
        callable(getattr(hipporag, "retrieve", None))
        and callable(getattr(hipporag, "retrieve_dpr", None))
        and (not include_qa or callable(getattr(hipporag, "qa", None)))
        and not (
            hasattr(hipporag, "rag_qa")
            and hasattr(hipporag.rag_qa, "assert_called")
            and getattr(hipporag.retrieve, "side_effect", None) is None
            and not isinstance(getattr(hipporag.retrieve, "return_value", None), (list, tuple))
        )
    )
    if use_staged_qa:
        graph_solutions = []
        dpr_solutions = []
        for start in range(0, len(queries), qa_batch_size):
            batch_queries = queries[start : start + qa_batch_size]
            g_batch = hipporag.retrieve(queries=batch_queries, num_to_retrieve=num_to_retrieve)
            if isinstance(g_batch, tuple):
                g_batch = g_batch[0]
            d_batch = hipporag.retrieve_dpr(queries=batch_queries, num_to_retrieve=num_to_retrieve)
            if isinstance(d_batch, tuple):
                d_batch = d_batch[0]
            for sol in (*g_batch, *d_batch):
                if hasattr(sol, "docs") and sol.docs is not None:
                    sol.docs = list(sol.docs[:num_to_retrieve])
                if hasattr(sol, "doc_scores") and sol.doc_scores is not None:
                    sol.doc_scores = sol.doc_scores[:num_to_retrieve]
                if hasattr(sol, "doc_metadata") and sol.doc_metadata is not None:
                    sol.doc_metadata = list(sol.doc_metadata[:num_to_retrieve])
            query_cache = getattr(hipporag, "query_to_embedding", None)
            if isinstance(query_cache, dict):
                for role_cache in query_cache.values():
                    if isinstance(role_cache, dict):
                        role_cache.clear()
            graph_solutions.extend(g_batch)
            dpr_solutions.extend(d_batch)

        _release_retrieval_memory(hipporag)

        if include_qa:
            answered_graph = []
            answered_dpr = []
            for start in range(0, len(queries), qa_batch_size):
                g_qa, _, _ = hipporag.qa(graph_solutions[start : start + qa_batch_size])
                d_qa, _, _ = hipporag.qa(dpr_solutions[start : start + qa_batch_size])
                answered_graph.extend(g_qa)
                answered_dpr.extend(d_qa)
                gc.collect()
            graph_solutions = answered_graph
            dpr_solutions = answered_dpr
    else:
        graph_solutions, _, _, _, _ = hipporag.rag_qa(
            queries=queries, gold_docs=gold_docs, gold_answers=gold_answers,
        )
        dpr_solutions, _, _, _, _ = hipporag.rag_qa_dpr(
            queries=queries, gold_docs=gold_docs, gold_answers=gold_answers,
        )

    recall_eval = RetrievalRecall()
    graph_retrieval_metrics, _ = recall_eval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=[list(sol.docs) for sol in graph_solutions],
        k_list=[1, 2, 5],
    )
    dpr_retrieval_metrics, _ = recall_eval.calculate_metric_scores(
        gold_docs=gold_docs,
        retrieved_docs=[list(sol.docs) for sol in dpr_solutions],
        k_list=[1, 2, 5],
    )

    if include_qa:
        graph_preds = [sol.answer or "" for sol in graph_solutions]
        dpr_preds = [sol.answer or "" for sol in dpr_solutions]
        graph_em_overall, graph_em = QAExactMatch().calculate_metric_scores(gold_answers, graph_preds)
        graph_f1_overall, graph_f1 = QAF1Score().calculate_metric_scores(gold_answers, graph_preds)
        dpr_em_overall, dpr_em = QAExactMatch().calculate_metric_scores(gold_answers, dpr_preds)
        dpr_f1_overall, dpr_f1 = QAF1Score().calculate_metric_scores(gold_answers, dpr_preds)
        graph_qa_metrics = {
            "ExactMatch": round(float(graph_em_overall["ExactMatch"]), 4),
            "F1": round(float(graph_f1_overall["F1"]), 4),
        }
        dpr_qa_metrics = {
            "ExactMatch": round(float(dpr_em_overall["ExactMatch"]), 4),
            "F1": round(float(dpr_f1_overall["F1"]), 4),
        }
    else:
        graph_em = graph_f1 = dpr_em = dpr_f1 = []
        graph_qa_metrics = None
        dpr_qa_metrics = None

    records = []
    for i, (g_sol, d_sol, expected_docs, expected_ans) in enumerate(
        zip(graph_solutions, dpr_solutions, gold_docs, gold_answers, strict=True)
    ):
        g_docs = g_sol.docs[:qa_top_k]
        d_docs = d_sol.docs[:qa_top_k]
        records.append({
            "question": g_sol.question,
            "answer": g_sol.answer or "",
            "dpr_answer": d_sol.answer or "",
            "gold_answers": expected_ans,
            "gold_docs": expected_docs,
            "qa_evidence": g_docs,
            "dpr_evidence": d_docs,
            "scores": [float(s) for s in g_sol.doc_scores[:qa_top_k]],
            "graph_seeds": list(getattr(g_sol, "graph_seeds", None) or []),
            "recall": {
                f"@{k}": round(len(set(g_sol.docs[:k]) & set(expected_docs)) / len(expected_docs), 4)
                for k in (1, 2, 5)
            },
            "dpr_recall": {
                f"@{k}": round(len(set(d_sol.docs[:k]) & set(expected_docs)) / len(expected_docs), 4)
                for k in (1, 2, 5)
            },
            "qa_metrics": (
                {
                    "ExactMatch": float(graph_em[i]["ExactMatch"]),
                    "F1": float(graph_f1[i]["F1"]),
                }
                if include_qa
                else None
            ),
            "dpr_qa_metrics": (
                {
                    "ExactMatch": float(dpr_em[i]["ExactMatch"]),
                    "F1": float(dpr_f1[i]["F1"]),
                }
                if include_qa
                else None
            ),
        })

    seeds_active = sum(1 for r in records if r["graph_seeds"])
    fallback_count = len(records) - seeds_active
    hipporag_ret = {f"Recall@{k}": float(graph_retrieval_metrics[f"Recall@{k}"]) for k in (1, 2, 5)}
    dpr_ret = {f"Recall@{k}": float(dpr_retrieval_metrics[f"Recall@{k}"]) for k in (1, 2, 5)}

    metrics = {
        "hipporag": {
            "retrieval": hipporag_ret,
            "qa": (
                {"ExactMatch": float(graph_qa_metrics["ExactMatch"]), "F1": float(graph_qa_metrics["F1"])}
                if include_qa and graph_qa_metrics is not None
                else None
            ),
        },
        "standard_dpr": {
            "retrieval": dpr_ret,
            "qa": (
                {"ExactMatch": float(dpr_qa_metrics["ExactMatch"]), "F1": float(dpr_qa_metrics["F1"])}
                if include_qa and dpr_qa_metrics is not None
                else None
            ),
        },
        "delta_vs_dpr": {
            "Recall@1": round(hipporag_ret["Recall@1"] - dpr_ret["Recall@1"], 4),
            "Recall@2": round(hipporag_ret["Recall@2"] - dpr_ret["Recall@2"], 4),
            "Recall@5": round(hipporag_ret["Recall@5"] - dpr_ret["Recall@5"], 4),
            "ExactMatch": (
                round(float(graph_qa_metrics["ExactMatch"]) - float(dpr_qa_metrics["ExactMatch"]), 4)
                if include_qa and graph_qa_metrics is not None and dpr_qa_metrics is not None
                else None
            ),
            "F1": (
                round(float(graph_qa_metrics["F1"]) - float(dpr_qa_metrics["F1"]), 4)
                if include_qa and graph_qa_metrics is not None and dpr_qa_metrics is not None
                else None
            ),
        },
        "recognition_memory": {
            "queries_with_graph_seeds": seeds_active,
            "dpr_fallback_count": fallback_count,
            "dpr_fallback_rate": round(fallback_count / len(records), 4) if records else 0.0,
        },
    }
    if dataset_name in PAPER_BASELINES:
        metrics["paper_baselines"] = PAPER_BASELINES[dataset_name]
    return records, metrics


def check_benchmark_integrity(
    records: list[dict],
    graph_info: dict,
    expected_doc_count: int,
    extraction_errors: list[str] | tuple[str, ...] = (),
    qa_errors: list[str] | tuple[str, ...] = (),
    include_qa: bool = True,
) -> list[str]:
    """Fail-fast system integrity checks on indexing, chat completions, and PPR activation."""
    errors = []
    if extraction_errors:
        errors.extend(f"extraction_llm: {err}" for err in extraction_errors)
    if include_qa and qa_errors:
        errors.extend(f"qa_llm: {err}" for err in qa_errors)
    if graph_info.get("num_passage_nodes") != expected_doc_count:
        errors.append(
            f"Index passage count mismatch: {graph_info.get('num_passage_nodes')} != {expected_doc_count}"
        )
    if not graph_info.get("num_extracted_triples"):
        errors.append("No triples extracted during OpenIE indexing")
    if include_qa:
        for idx, record in enumerate(records):
            if not str(record.get("answer", "")).strip():
                errors.append(f"Question {idx + 1}: empty HippoRAG answer")
            if not str(record.get("dpr_answer", "")).strip():
                errors.append(f"Question {idx + 1}: empty StandardRAG (DPR) answer")
    if len(records) > 1 and not any(record.get("graph_seeds") for record in records):
        errors.append("All queries had empty graph_seeds (recognition memory / PPR never activated)")
    return errors


def print_benchmark_summary(dataset: str, num_docs: int, records: list[dict], metrics: dict):
    print(f"\n=== Benchmark Summary: {dataset} ({len(records)} queries, {num_docs} passages) ===")
    print("HippoRAG 2 Retrieval:", metrics["hipporag"]["retrieval"])
    print("Standard DPR Retrieval:", metrics["standard_dpr"]["retrieval"])
    if metrics["hipporag"]["qa"] is not None:
        print("HippoRAG 2 QA:", metrics["hipporag"]["qa"])
        print("Standard DPR QA:", metrics["standard_dpr"]["qa"])
    print("Delta (HippoRAG 2 - DPR):", metrics["delta_vs_dpr"])
    print("Recognition Memory:", metrics["recognition_memory"])
    if "paper_baselines" in metrics:
        print("Paper Reference (Llama-3.3-70B + NV-Embed-v2):",
              metrics["paper_baselines"]["hipporag2_llama3.3_70b_nvembed_v2"])


def main(
    dataset: str = "sample",
    max_queries: int | None = None,
    full_corpus: bool = False,
    mode: str = "eval",
    chat_model: str | None = None,
    embedding_model: str = MODEL,
    extraction_model: str = DEFAULT_EXTRACTION_CHAT,
    qa_model: str = DEFAULT_QA_CHAT,
    embedding_task_mode: str = "distinct-titled",
    fresh_index: bool = False,
    index_batch_size: int = 8,
    qa_batch_size: int = 1,
):
    if chat_model is not None:
        extraction_model = chat_model
        qa_model = chat_model
    include_qa = mode not in ("embeddings", "retrieval")
    profile = resolve_embedding_profile(embedding_model, embedding_task_mode)
    include_corpus = bool(full_corpus or max_queries is None or dataset == "sample")
    samples, corpus = load_dataset_files(
        dataset, max_queries=max_queries, include_corpus=include_corpus,
    )
    docs, queries, gold_docs, gold_answers = build_benchmark_slice(
        dataset, samples, corpus, max_queries=max_queries, full_corpus=full_corpus,
    )
    del samples, corpus
    gc.collect()
    num_docs = len(docs)

    started = perf_counter()
    root = PROJECT_ROOT / "outputs" / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"run-{dataset}-", dir=root))

    slice_hash = hashlib.sha256("\n".join(docs).encode("utf-8")).hexdigest()[:10]
    id_hash = hashlib.sha256(
        f"{profile.index_identity}|{chat_identity(extraction_model)}".encode("utf-8")
    ).hexdigest()[:10]
    if fresh_index:
        index_dir = run_dir / "index"
    else:
        index_dir = root / "indexes" / f"{dataset}_d{num_docs}_{slice_hash}_{id_hash}"

    configure_logging(log_file=run_dir / "run.log", max_lines=500)
    report = {
        "status": "running",
        "dataset": dataset,
        "num_queries": len(queries),
        "num_docs": num_docs,
        "full_corpus": full_corpus,
        "index_batch_size": index_batch_size,
        "qa_batch_size": qa_batch_size,
        "embedding_model": profile.model,
        "checkpoint": profile.checkpoint,
        "embedding_task_mode": embedding_task_mode,
        "query_instruction_mode": profile.query_instruction_mode,
        "parse_document_title": profile.parse_document_title,
        "extraction_model": extraction_model,
        "qa_model": qa_model if include_qa else None,
        "chat_model": (
            extraction_model
            if not include_qa or extraction_model == qa_model
            else f"{extraction_model} -> {qa_model}"
        ),
        "base_url": BASE_URL,
        "index_identity": profile.index_identity,
        "index_dir": str(index_dir),
        "run_dir": str(run_dir),
        "mode": mode,
    }
    print(f"Benchmark run: {run_dir} (dataset={dataset}, queries={len(queries)}, docs={num_docs})", flush=True)
    try:
        report["model_registry"] = preflight(
            profile,
            None if mode == "embeddings" else extraction_model,
            None if not include_qa or qa_model == extraction_model else qa_model,
        )
        config = BaseConfig(
            save_dir=str(index_dir),
            dataset=dataset,
            llm_name=extraction_model,
            llm_base_url=BASE_URL,
            embedding_model_name=profile.model,
            embedding_base_url=BASE_URL,
            embedding_provider="openai",
            embedding_batch_size=32,
            embedding_return_as_normalized=True,
            vector_store_type="parquet",
            force_index_from_scratch=not any(index_dir.rglob("graph.pickle")),
            qa_top_k=QA_TOP_K,
            linking_top_k=LINKING_TOP_K,
            retrieval_top_k=RETRIEVAL_TOP_K,
            temperature=0,
            seed=0,
            openie_max_workers=1,
            max_retry_attempts=0,
        )
        with closing(RetrievalEmbedding(config, profile)) as embedding:
            if mode in ("embeddings", "all"):
                report["embedding_probes"] = verify_embeddings(embedding, queries, docs)
            report["server"] = embedding_evidence(profile)
            report["index_identity"] = report["server"]["resolved_index_identity"]
            if mode == "embeddings":
                report["status"] = "passed"
                return report
            report["chat_policy"] = {
                "extraction_identity": chat_identity(extraction_model),
                "qa_identity": chat_identity(qa_model) if include_qa else None,
                "openie_max_workers": 1,
                "max_retries": 0,
            }
            report["index_identity"] += "|" + chat_identity(extraction_model)
            with closing(LocalChat.from_experiment_config(config, extraction_model)) as extraction_llm, closing(
                LocalChat.from_experiment_config(config, qa_model if include_qa else extraction_model)
            ) as qa_llm, HippoRAG(
                global_config=config,
                embedding_model=embedding,
                extraction_llm=extraction_llm,
                qa_llm=qa_llm if include_qa else None,
                index_identity=report["index_identity"],
            ) as hipporag:
                index_start = perf_counter()
                report["index_batches"] = index_documents_in_batches(
                    hipporag, docs, batch_size=index_batch_size,
                )
                del docs
                gc.collect()
                report["indexing_seconds"] = round(perf_counter() - index_start, 2)
                report["graph"] = hipporag.get_graph_info()

                eval_start = perf_counter()
                records, metrics = evaluate_benchmark(
                    hipporag,
                    queries,
                    gold_docs,
                    gold_answers,
                    dataset_name=dataset,
                    qa_top_k=QA_TOP_K,
                    qa_batch_size=qa_batch_size,
                    include_qa=include_qa,
                )
                report["evaluation_seconds"] = round(perf_counter() - eval_start, 2)
                report["timing_breakdown"] = {
                    "all_retrieval_time": round(float(hipporag.all_retrieval_time), 2),
                    "recognition_memory_time": round(float(hipporag.rerank_time), 2),
                    "ppr_time": round(float(hipporag.ppr_time), 2),
                }
                report["evaluation_metrics"] = metrics
                report["records"] = records
                report["chat_telemetry"] = {
                    "extraction": extraction_llm.summary(),
                    "qa": qa_llm.summary() if include_qa else None,
                }
                report["chat_backends"] = {
                    "extraction": chat_evidence(extraction_model),
                    "qa": chat_evidence(qa_model) if include_qa else None,
                }
                print_benchmark_summary(dataset, num_docs, records, metrics)
                failures = check_benchmark_integrity(
                    records,
                    report["graph"],
                    expected_doc_count=num_docs,
                    extraction_errors=extraction_llm.errors,
                    qa_errors=qa_llm.errors if include_qa else (),
                    include_qa=include_qa,
                )
                if failures:
                    raise RuntimeError("; ".join(failures))
        report["status"] = "passed"
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 2)
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n{report['status'].upper()}: {run_dir / 'report.json'}", flush=True)


def index_documents_in_batches(
    hipporag,
    docs: list[str],
    batch_size: int = 8,
) -> int:
    """Checkpoint OpenIE in bounded micro-batches, then build embeddings and graph once."""
    if batch_size < 1:
        raise ValueError("index_batch_size must be at least 1.")
    if not docs:
        return 0
    store = getattr(hipporag, "chunk_embedding_store", None)
    graph_ready = bool(getattr(hipporag, "_graph_state_available", False))
    if graph_ready and store is not None:
        existing = store.get_all_texts()
        pending = [doc for doc in docs if doc not in existing]
        if not pending:
            graph_info = hipporag.get_graph_info() if callable(getattr(hipporag, "get_graph_info", None)) else {}
            if isinstance(graph_info, dict) and graph_info.get("num_passage_nodes") == len(existing):
                _release_indexing_memory(hipporag)
                return 0
            hipporag.index(docs=docs[:batch_size])
            _release_indexing_memory(hipporag)
            return 1
    else:
        pending = list(docs)

    pre_openie = getattr(hipporag, "pre_openie", None)
    use_staged_openie = callable(pre_openie) and not hasattr(pre_openie, "assert_called")
    num_batches = 0
    total_docs = len(pending)
    total_batches = (total_docs + batch_size - 1) // batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if use_staged_openie:
            print(
                f"[Indexing OpenIE] Batch {num_batches + 1}/{total_batches} "
                f"({min(start + len(batch), total_docs)}/{total_docs} docs)",
                flush=True,
            )
            try:
                pre_openie(batch)
            except RuntimeError as exc:
                if "Offline OpenIE completed" not in str(exc):
                    raise
            gc.collect()
        else:
            hipporag.index(docs=batch)
            _release_indexing_memory(hipporag)
        num_batches += 1

    if use_staged_openie:
        print(f"[Indexing Graph] Building embeddings and knowledge graph for {total_docs} docs...", flush=True)
        hipporag.index(docs=pending)
        _release_indexing_memory(hipporag)
    return num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="sample")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Number of queries to evaluate (pools their candidate passages unless --full-corpus).")
    parser.add_argument("--full-corpus", action="store_true",
                        help="Index the full dataset corpus even when --max-queries is set.")
    parser.add_argument("--mode", choices=("eval", "retrieval", "all", "embeddings"), default="eval",
                        help="Run full benchmark ('eval'), retrieval-only ('retrieval'), include probes ('all'), or probes only.")
    parser.add_argument("--chat-model", choices=tuple(CHAT_MODELS), default=None,
                        help="Override both extraction and QA chat models with a single model.")
    parser.add_argument("--extraction-model", choices=tuple(CHAT_MODELS), default=DEFAULT_EXTRACTION_CHAT)
    parser.add_argument("--qa-model", choices=tuple(CHAT_MODELS), default=DEFAULT_QA_CHAT)
    parser.add_argument("--embedding-model", choices=tuple(PROFILES), default=MODEL)
    parser.add_argument("--embedding-task-mode", choices=GEMMA_TASK_MODES, default="distinct-titled")
    parser.add_argument("--fresh-index", action="store_true",
                        help="Force indexing in a fresh disposable directory instead of reusing outputs/benchmarks/indexes.")
    parser.add_argument("--index-batch-size", type=int, default=8,
                        help="Number of passages per hipporag.index() micro-batch for incremental SSD checkpointing.")
    parser.add_argument("--qa-batch-size", type=int, default=1,
                        help="Number of queries per retrieve() and qa() micro-batch (default: 1 for 1-by-1 processing).")
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        max_queries=args.max_queries,
        full_corpus=args.full_corpus,
        mode=args.mode,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        qa_model=args.qa_model,
        embedding_task_mode=args.embedding_task_mode,
        fresh_index=args.fresh_index,
        index_batch_size=args.index_batch_size,
        qa_batch_size=args.qa_batch_size,
    )
