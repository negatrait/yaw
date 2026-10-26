"""Self-contained evaluation metrics for retrieval and QA benchmarks.

Migrated from upstream hipporag into scripts/ to eliminate external
benchmarking dependencies while maintaining strict numerical parity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from yaw.daemon.graph.inference import (
    compute_qa_em_f1,
    doc_matches_gold,
    normalize_answer,
)

__all__ = [
    "QAExactMatch",
    "QAF1Score",
    "RetrievalRecall",
    "normalize_answer",
]


class RetrievalRecall:
    """Calculates Recall@k for retrieved passages (or AST islands) against gold passages."""

    metric_name: str = "retrieval_recall"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    @staticmethod
    def _doc_matches_gold(retrieved_doc: str, gold_doc: str) -> bool:
        return doc_matches_gold(retrieved_doc, gold_doc)

    def calculate_metric_scores(
        self,
        gold_docs: list[list[str]],
        retrieved_docs: list[list[str]],
        k_list: list[int] | tuple[int, ...] = (1, 5, 10, 20),
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        if len(gold_docs) != len(retrieved_docs):
            raise ValueError("gold_docs and retrieved_docs must have the same length.")
        sorted_k = sorted(set(k_list))
        if not sorted_k or any(k < 1 for k in sorted_k):
            raise ValueError("k_list must contain at least one positive integer.")

        example_eval_results: list[dict[str, float]] = []
        pooled_eval_results: dict[str, float] = {f"Recall@{k}": 0.0 for k in sorted_k}

        for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs, strict=False):
            example_eval_result = {f"Recall@{k}": 0.0 for k in sorted_k}
            unique_gold = list(dict.fromkeys(example_gold_docs))
            for k in sorted_k:
                top_k_docs = example_retrieved_docs[:k]
                if unique_gold:
                    matched_gold = sum(1 for g in unique_gold if any(self._doc_matches_gold(r, g) for r in top_k_docs))
                    example_eval_result[f"Recall@{k}"] = matched_gold / len(unique_gold)
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0

            example_eval_results.append(example_eval_result)
            for k in sorted_k:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        num_examples = len(gold_docs)
        if num_examples == 0:
            return pooled_eval_results, example_eval_results

        for k in sorted_k:
            pooled_eval_results[f"Recall@{k}"] /= num_examples

        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results


class QAExactMatch:
    """Calculates Exact Match (EM) score across multiple reference answers."""

    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    def calculate_metric_scores(
        self,
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        aggregation_fn: Callable[[list[float]], float] = np.max,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        if len(gold_answers) != len(predicted_answers):
            raise ValueError("gold_answers and predicted_answers must have the same length.")

        example_eval_results: list[dict[str, float]] = []
        total_em = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers, strict=False):
            em_scores = [compute_qa_em_f1(predicted, [gold])[0] for gold in gold_list]
            aggregated_em = float(aggregation_fn(em_scores)) if em_scores else 0.0
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}
        return pooled_eval_results, example_eval_results


class QAF1Score:
    """Calculates token-level F1 score across multiple reference answers."""

    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    def calculate_metric_scores(
        self,
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        aggregation_fn: Callable[[list[float]], float] = np.max,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        if len(gold_answers) != len(predicted_answers):
            raise ValueError("gold_answers and predicted_answers must have the same length.")

        example_eval_results: list[dict[str, float]] = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers, strict=False):
            f1_scores = [compute_qa_em_f1(predicted, [gold])[1] for gold in gold_list]
            aggregated_f1 = float(aggregation_fn(f1_scores)) if f1_scores else 0.0
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}
        return pooled_eval_results, example_eval_results
