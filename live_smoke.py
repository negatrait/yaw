"""Fail-fast live indexing, graph retrieval and QA evaluation on Lemonade.

Run all modes: uv run --locked python live_smoke.py
Evaluation only: uv run --locked python live_smoke.py --mode eval
Each run creates a new index and report under outputs/live-smoke/.
Model installation/configuration is explicit; see README.
"""

import argparse
from contextlib import closing
import json
import logging
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from hipporag import HippoRAG
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.prompts.linking import get_query_instruction
from hipporag.utils.config_utils import BaseConfig

from yaw.chat import LocalChat, chat_identity
from yaw.embedding import (
    GEMMA_TASK_MODES,
    MODEL,
    PROFILES,
    RetrievalEmbedding,
    resolve_embedding_profile,
)
from yaw.models import (
    BASE_URL,
    CHAT_MODELS,
    DEFAULT_CHAT,
    DEFAULT_EXTRACTION_CHAT,
    DEFAULT_QA_CHAT,
    chat_evidence,
    embedding_evidence,
    preflight,
)

CHAT_MODEL = DEFAULT_CHAT
QA_TOP_K = 5
DOCS = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is bom in Minsk.",
    "Montebello is a part of Rockland County.",
]
QUERIES = [
    "What is George Rankin's occupation?",
    "How did Cinderella reach her happy ending?",
    "What county is Erik Hort's birthplace a part of?",
]
GOLD_DOCS = [[DOCS[1]], DOCS[3:6], [DOCS[6], DOCS[8]]]
# Defined from source evidence, independently of generated predictions.
GOLD_ANSWERS = [
    ["Politician"],
    ["The glass slipper fit Cinderella, reuniting her with the prince."],
    ["Rockland County"],
]


def verify_embeddings(embedding):
    """Five repeats of batch, individual and reversed inputs, for all active roles."""
    probes = []
    passage_instruction = get_query_instruction("query_to_passage")
    roles = [
        ("query", QUERIES, {"instruction": passage_instruction}),
    ]
    if embedding.profile.query_instruction_mode == "distinct":
        fact_instruction = get_query_instruction("query_to_fact")
        roles.append(("fact_query", QUERIES, {"instruction": fact_instruction}))
    roles.append(("document", [DOCS[1], DOCS[3], DOCS[8]], {}))
    if embedding.profile.parse_document_title:
        roles.append((
            "titled_document",
            [f"George Rankin\n{DOCS[1]}", f"Cinderella\n{DOCS[3]}", f"Montebello\n{DOCS[8]}"],
            {},
        ))
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
        # Compare every pair of contexts/repeats for each input, not just adjacent calls.
        cosine = np.einsum("rid,sid->irs", samples, samples)
        minimum = float(cosine.min())
        probes.append({
            "role": role,
            "inputs": [
                embedding.profile.format(text, query=bool(options), instruction=options.get("instruction"))
                for text in texts
            ],
            "repeats": 5, "dimensions": embedding.profile.dimensions, "min_pairwise_cosine": minimum,
        })
        if minimum < 0.999:
            raise RuntimeError(f"{role} embeddings are unstable: min cosine {minimum:.6f} < 0.999")
    return probes


def record_results(results):
    records = []
    for result, expected_docs in zip(results, GOLD_DOCS, strict=True):
        evidence = result.docs[:QA_TOP_K]
        answer = result.answer or ""
        records.append({
            "question": result.question, "answer": answer,
            "qa_evidence": evidence,
            "scores": [float(score) for score in result.doc_scores[:QA_TOP_K]],
            "graph_seeds": result.graph_seeds,
            "recall": {
                f"@{k}": len(set(result.docs[:k]) & set(expected_docs)) / len(expected_docs)
                for k in (1, 2, 5)
            },
        })
    return records


def check_results(records):
    """Fixture checks are explicit, modest checks, not a general semantic judge."""
    errors = []
    if records[0]["qa_evidence"][0] != DOCS[1]:
        errors.append("George Rankin's evidence must rank first")
    for i, required in enumerate(GOLD_DOCS):
        if not set(required).issubset(records[i]["qa_evidence"]):
            errors.append(f"Question {i + 1}: missing evidence inside QA top {QA_TOP_K}")
    for i, record in enumerate(records):
        if "graph_seeds" in record and not record["graph_seeds"]:
            errors.append(f"Question {i + 1}: empty graph_seeds (rerank/PPR fell back to DPR)")
        answer = record["answer"].lower()
        if not answer.strip() or "unknown" in answer:
            errors.append(f"Question {i + 1}: empty/Unknown answer")
        expected_terms = (("politician",), ("slipper", "prince"), ("rockland county",))[i]
        if not all(term in answer for term in expected_terms):
            errors.append(f"Question {i + 1}: expected answer concepts absent")
    return errors


def evaluate(hipporag):
    """Use upstream gold-document recall (graph and DPR) and QA metrics, including per-query EM/F1."""
    results, _, _, retrieval_metrics, qa_metrics = hipporag.rag_qa(
        queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=GOLD_ANSWERS,
    )
    _, dpr_retrieval_metrics = hipporag.retrieve_dpr(
        queries=QUERIES, num_to_retrieve=QA_TOP_K, gold_docs=GOLD_DOCS,
    )
    records = record_results(results)
    predictions = [row["answer"] for row in records]
    _, exact_matches = QAExactMatch().calculate_metric_scores(GOLD_ANSWERS, predictions)
    _, f1_scores = QAF1Score().calculate_metric_scores(GOLD_ANSWERS, predictions)
    for i, row in enumerate(records):
        row.update(
            gold_docs=GOLD_DOCS[i], gold_answers=GOLD_ANSWERS[i],
            qa_metrics={"ExactMatch": float(exact_matches[i]["ExactMatch"]),
                        "F1": float(f1_scores[i]["F1"])},
        )
    # Above five, upstream's default recall cutoffs exceed the actual retrieval window.
    metrics = {
        "retrieval": {f"Recall@{k}": retrieval_metrics[f"Recall@{k}"] for k in (1, 2, 5)},
        "dpr_retrieval": {f"Recall@{k}": dpr_retrieval_metrics[f"Recall@{k}"] for k in (1, 2, 5)},
        "qa": qa_metrics,
    }
    return records, metrics


def print_evaluation(records, metrics):
    print("\n=== Evaluation: gold documents and gold answers ===")
    for row in records:
        print(f"\nQ: {row['question']}\nPrediction: {row['answer']}")
        print("Gold answers:", row["gold_answers"])
        print("Gold documents:", row["gold_docs"])
        print("Retrieval recall:", row["recall"])
        print("Answer scores:", row["qa_metrics"])
    print("\nAggregate graph retrieval:", metrics["retrieval"])
    print("Aggregate DPR retrieval:", metrics["dpr_retrieval"])
    print("Aggregate answer scores:", metrics["qa"])
    print("F1 measures normalized answer-token overlap; a correct paraphrase can score below 1.")


def main(
    mode="all",
    chat_model=None,
    embedding_model=MODEL,
    extraction_model=DEFAULT_EXTRACTION_CHAT,
    qa_model=DEFAULT_QA_CHAT,
    embedding_task_mode="search",
):
    if chat_model is not None:
        extraction_model = chat_model
        qa_model = chat_model
    profile = resolve_embedding_profile(embedding_model, embedding_task_mode)
    started = perf_counter()
    root = Path(__file__).resolve().parent / "outputs" / "live-smoke"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
    )
    report = {
        "status": "running", "embedding_model": profile.model, "checkpoint": profile.checkpoint,
        "embedding_task_mode": embedding_task_mode,
        "query_instruction_mode": profile.query_instruction_mode,
        "extraction_model": extraction_model, "qa_model": qa_model,
        "chat_model": qa_model if extraction_model == qa_model else f"{extraction_model} -> {qa_model}",
        "base_url": BASE_URL, "index_identity": profile.index_identity, "qa_top_k": QA_TOP_K,
        "run_dir": str(run_dir), "mode": mode,
    }
    print(f"Fresh index: {run_dir}", flush=True)
    try:
        report["model_registry"] = preflight(
            profile,
            None if mode == "embeddings" else extraction_model,
            None if mode == "embeddings" or qa_model == extraction_model else qa_model,
        )
        config = BaseConfig(
            save_dir=str(run_dir / "index"),
            llm_name=extraction_model, llm_base_url=BASE_URL,
            embedding_model_name=profile.model, embedding_base_url=BASE_URL,
            embedding_provider="openai", embedding_batch_size=32,
            embedding_return_as_normalized=True, vector_store_type="parquet",
            qa_top_k=QA_TOP_K, retrieval_top_k=QA_TOP_K,
            temperature=0, seed=0, openie_max_workers=1, max_retry_attempts=0,
        )
        # HippoRAG does not own/close injected clients.
        with closing(RetrievalEmbedding(config, profile)) as embedding:
            report["embedding_probes"] = verify_embeddings(embedding)
            report["server"] = embedding_evidence(profile)
            report["index_identity"] = report["server"]["resolved_index_identity"]
            if mode == "embeddings":
                report["status"] = "passed"
                return
            report["chat_policy"] = {
                "extraction_identity": chat_identity(extraction_model),
                "qa_identity": chat_identity(qa_model),
                "openie_max_workers": 1, "max_retries": 0,
            }
            # Index artifacts depend on embeddings and OpenIE extraction_llm, not qa_llm.
            report["index_identity"] += "|" + chat_identity(extraction_model)
            with closing(LocalChat.from_experiment_config(config, extraction_model)) as extraction_llm, closing(
                LocalChat.from_experiment_config(config, qa_model)
            ) as qa_llm, HippoRAG(
                global_config=config, embedding_model=embedding,
                extraction_llm=extraction_llm, qa_llm=qa_llm,
                index_identity=report["index_identity"],
            ) as hipporag:
                hipporag.index(docs=DOCS)
                report["graph"] = hipporag.get_graph_info()
                result_modes = []
                if mode in ("smoke", "all"):
                    retrieved = hipporag.retrieve(queries=QUERIES, num_to_retrieve=QA_TOP_K)
                    report["separate"] = record_results(hipporag.rag_qa(retrieved)[0])
                    report["combined"] = record_results(hipporag.rag_qa(queries=QUERIES)[0])
                    result_modes.extend(("separate", "combined"))
                if mode in ("eval", "all"):
                    report["eval"], report["evaluation_metrics"] = evaluate(hipporag)
                    report["upstream_retrieval_metrics"] = report["evaluation_metrics"]["retrieval"]
                    report["upstream_dpr_retrieval_metrics"] = report["evaluation_metrics"]["dpr_retrieval"]
                    report["upstream_qa_metrics"] = report["evaluation_metrics"]["qa"]
                    print_evaluation(report["eval"], report["evaluation_metrics"])
                    result_modes.append("eval")
                report["chat_telemetry"] = {
                    "extraction": extraction_llm.summary(),
                    "qa": qa_llm.summary(),
                }
                report["chat_backends"] = {
                    "extraction": chat_evidence(extraction_model),
                    "qa": chat_evidence(qa_model),
                }
                failures = []
                if extraction_llm.errors:
                    failures.extend(f"extraction_llm: {err}" for err in extraction_llm.errors)
                if qa_llm.errors:
                    failures.extend(f"qa_llm: {err}" for err in qa_llm.errors)
                for result_mode in result_modes:
                    failures.extend(f"{result_mode}: {error}" for error in check_results(report[result_mode]))
                    for row in report[result_mode]:
                        print(f"\n{result_mode}: {row['question']}\nAnswer: {row['answer']}")
                        print("QA evidence:", row["qa_evidence"])
                        print("Graph seeds:", row["graph_seeds"])
                if report["graph"]["num_passage_nodes"] != len(DOCS):
                    failures.append("Index passage count mismatch")
                if not report["graph"]["num_extracted_triples"]:
                    failures.append("No triples extracted")
                if failures:
                    raise RuntimeError("; ".join(failures))
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 2)
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8",
        )
        print(f"\n{report['status'].upper()}: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "eval", "all", "embeddings"), default="all",
                        help="Run QA smoke, gold evaluation, both (default), or only embedding probes.")
    parser.add_argument("--chat-model", choices=tuple(CHAT_MODELS), default=None,
                        help="Override both extraction and QA chat models with a single model.")
    parser.add_argument("--extraction-model", choices=tuple(CHAT_MODELS), default=DEFAULT_EXTRACTION_CHAT,
                        help="Chat model for OpenIE NER, triple extraction, and recognition memory.")
    parser.add_argument("--qa-model", choices=tuple(CHAT_MODELS), default=DEFAULT_QA_CHAT,
                        help="Chat model for final QA reading comprehension.")
    parser.add_argument("--embedding-model", choices=tuple(PROFILES), default=MODEL)
    parser.add_argument("--embedding-task-mode", choices=GEMMA_TASK_MODES, default="search")
    args = parser.parse_args()
    main(
        args.mode, args.chat_model, args.embedding_model,
        args.extraction_model, args.qa_model, args.embedding_task_mode,
    )

