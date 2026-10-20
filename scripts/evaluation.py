"""Self-contained evaluation metrics for retrieval and QA benchmarks.

Migrated from upstream hipporag into scripts/ to eliminate external
benchmarking dependencies while maintaining strict numerical parity.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np


def normalize_answer(answer: str) -> str:
    """Normalize a given string for exact match and F1 comparison.

    Transformations:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles ("a", "an", "the")
    4. Collapse multiple whitespace characters into single space
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


class RetrievalRecall:
    """Calculates Recall@k for retrieved passages (or AST islands) against gold passages."""

    metric_name: str = "retrieval_recall"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    @staticmethod
    def _doc_matches_gold(retrieved_doc: str, gold_doc: str) -> bool:
        if retrieved_doc == gold_doc:
            return True
        clean_ret = retrieved_doc.split("\n[... AST section cropped", 1)[0].strip()
        clean_gold = gold_doc.strip()
        if not clean_ret or not clean_gold:
            return False
        if clean_ret == clean_gold:
            return True

        # Strip standard titles/prefixes for content comparison
        for pfx in ("Wikipedia Title: ", "Title: "):
            if clean_ret.startswith(pfx):
                clean_ret = clean_ret[len(pfx):].strip()
            if clean_gold.startswith(pfx):
                clean_gold = clean_gold[len(pfx):].strip()

        if clean_ret == clean_gold:
            return True

        ws_ret = " ".join(clean_ret.split())
        ws_gold = " ".join(clean_gold.split())
        if ws_ret == ws_gold:
            return True

        # Check containment in both directions
        if len(ws_ret) >= 20 and (ws_ret in ws_gold or ws_gold in ws_ret):
            return True

        # Check body-only without title line
        if "\n" in clean_ret:
            body_only = " ".join(clean_ret.split("\n", 1)[1].split())
            if len(body_only) >= 20 and (body_only in ws_gold or ws_gold in body_only):
                return True

        return False

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
                    matched_gold = sum(
                        1
                        for g in unique_gold
                        if any(self._doc_matches_gold(r, g) for r in top_k_docs)
                    )
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
            norm_pred = normalize_answer(predicted)
            em_scores = [1.0 if normalize_answer(gold) == norm_pred else 0.0 for gold in gold_list]
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

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results: list[dict[str, float]] = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers, strict=False):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = float(aggregation_fn(f1_scores)) if f1_scores else 0.0
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}
        return pooled_eval_results, example_eval_results


QUERY_INSTRUCTIONS: dict[str, str] = {
    "ner_to_node": "Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.",
    "query_to_node": "Given a question, retrieve relevant phrases that are mentioned in this question.",
    "query_to_fact": "Given a question, retrieve relevant triplet facts that matches this question.",
    "query_to_sentence": "Given a question, retrieve relevant sentences that best answer the question.",
    "query_to_passage": "Given a question, retrieve relevant documents that best answer the question.",
}

DEFAULT_QUERY_INSTRUCTION = "Given a question, retrieve relevant documents that best answer the question."


def get_query_instruction(linking_method: str) -> str:
    """Retrieve embedding instruction for the specified query linking mode."""
    return QUERY_INSTRUCTIONS.get(linking_method, DEFAULT_QUERY_INSTRUCTION)
