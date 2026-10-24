"""Exact numerical contracts for the benchmark's production metrics."""

import numpy as np
import pytest

from scripts.evaluation import (
    DEFAULT_QUERY_INSTRUCTION,
    QUERY_INSTRUCTIONS,
    QAExactMatch,
    QAF1Score,
    RetrievalRecall,
    get_query_instruction,
    normalize_answer,
)


@pytest.mark.parametrize(
    ("retrieved", "gold", "matched"),
    [
        ("exact", "exact", 1.0),
        ("", "nonempty", 0.0),
        ("nonempty", "", 0.0),
        (" text \n", "text", 1.0),
        ("Wikipedia Title: content", "Title: content", 1.0),
        ("many    words", "many words", 1.0),
        ("a sufficiently long source passage", "a sufficiently long source passage with more context", 1.0),
        (
            "Retrieved heading\nShared body with sufficiently detailed content",
            "Gold heading\nShared body with sufficiently detailed content",
            1.0,
        ),
        ("Heading\nshort", "unrelated", 0.0),
        ("Heading\na longer body with nothing in common", "a separate unrelated passage entirely", 0.0),
        ("short", "other", 0.0),
        ("content\n[... AST section cropped due to context window limitations ...]", "content", 1.0),
    ],
)
def test_retrieval_matching_is_normalized_without_counting_unrelated_passages(retrieved, gold, matched):
    pooled, examples = RetrievalRecall().calculate_metric_scores([[gold]], [[retrieved]], (1,))
    assert pooled == {"Recall@1": matched}
    assert examples == [{"Recall@1": matched}]


def test_retrieval_deduplicates_gold_orders_ranks_and_averages_examples():
    metric = RetrievalRecall(global_config={"test": True})
    pooled, individual = metric.calculate_metric_scores([["one", "one", "two"], []], [["one", "two"], []], [2, 1, 2])
    assert pooled == {"Recall@1": 0.25, "Recall@2": 0.5}
    assert individual == [{"Recall@1": 0.5, "Recall@2": 1.0}, {"Recall@1": 0.0, "Recall@2": 0.0}]
    assert metric.calculate_metric_scores([], [], [1]) == ({"Recall@1": 0.0}, [])
    with pytest.raises(ValueError, match="same length"):
        metric.calculate_metric_scores([["one"]], [])
    for ranks in ([], [0], [-1, 1]):
        with pytest.raises(ValueError, match="positive integer"):
            metric.calculate_metric_scores([], [], ranks)


def test_qa_aliases_aggregation_and_empty_examples_have_exact_scores():
    assert normalize_answer(" THE,   Brown fox! ") == "brown fox"
    gold = [["the brown fox", "brown fox"], ["one two"], []]
    predicted = ["Brown fox!", "one three", "anything"]
    exact = QAExactMatch()
    f1 = QAF1Score()
    pooled, cases = exact.calculate_metric_scores(gold, predicted)
    assert cases == [{"ExactMatch": 1.0}, {"ExactMatch": 0.0}, {"ExactMatch": 0.0}]
    assert pooled["ExactMatch"] == pytest.approx(1 / 3)
    pooled_f1, cases_f1 = f1.calculate_metric_scores(gold, predicted)
    assert cases_f1 == [{"F1": 1.0}, {"F1": 0.5}, {"F1": 0.0}]
    assert pooled_f1["F1"] == pytest.approx(0.5)
    assert exact.calculate_metric_scores([["same", "different"]], ["same"], np.mean)[0]["ExactMatch"] == 0.5
    assert f1.calculate_metric_scores([["same", "different"]], ["same"], np.mean)[0]["F1"] == 0.5
    assert f1.calculate_metric_scores([["", "no overlap"]], [""])[0] == {"F1": 0.0}
    assert exact.calculate_metric_scores([], []) == ({"ExactMatch": 0.0}, [])
    assert f1.calculate_metric_scores([], []) == ({"F1": 0.0}, [])
    for metric in (exact, f1):
        with pytest.raises(ValueError, match="same length"):
            metric.calculate_metric_scores([["one"]], [])


def test_embedding_query_instruction_modes_and_fallback_are_explicit():
    assert all(get_query_instruction(key) == value for key, value in QUERY_INSTRUCTIONS.items())
    assert get_query_instruction("unknown-mode") == DEFAULT_QUERY_INSTRUCTION
