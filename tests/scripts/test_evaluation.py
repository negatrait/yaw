"""Exact numerical contracts for the benchmark's production metrics."""

import numpy as np
import pytest

from scripts.evaluation import (
    QAExactMatch,
    QAF1Score,
    RetrievalRecall,
    check_benchmark_integrity,
    evaluate_benchmark_gate,
)
from yaw.core import format_doc_for_embedding, format_query_for_embedding


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
        (
            "Header\nfirst sufficient span here\n[... lines 2-4 omitted ...]\nfinal sufficient span here",
            "Header\nfirst sufficient span here\nomitted middle text\nfinal sufficient span here",
            1.0,
        ),
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
    assert exact.calculate_metric_scores([["same", "different"]], ["same"], np.min)[0]["ExactMatch"] == 0.0
    assert f1.calculate_metric_scores([["same", "different"]], ["same"], np.min)[0]["F1"] == 0.0
    assert f1.calculate_metric_scores([["", "no overlap"]], [""])[0] == {"F1": 0.0}
    assert exact.calculate_metric_scores([], []) == ({"ExactMatch": 0.0}, [])
    assert f1.calculate_metric_scores([], []) == ({"F1": 0.0}, [])
    for metric in (exact, f1):
        with pytest.raises(ValueError, match="same length"):
            metric.calculate_metric_scores([["one"]], [])


@pytest.mark.parametrize(
    ("mode", "expected_query_prefix", "expected_doc_prefix"),
    [
        ("search", "task: search result | query: ", "title: none | text: "),
        ("titled", "task: search result | query: ", "title: Doc | text: "),
        ("qa", "task: question answering | query: ", "title: none | text: "),
        ("distinct", "task: question answering | query: ", "title: none | text: "),
        ("distinct-titled", "task: question answering | query: ", "title: Doc | text: "),
        ("fact", "task: fact checking | query: ", "title: none | text: "),
    ],
)
def test_embedding_query_and_doc_formatting_modes_and_invalid_mode(
    mode: str, expected_query_prefix: str, expected_doc_prefix: str
) -> None:
    assert format_query_for_embedding("who?", task_mode=mode) == expected_query_prefix + "who?"
    assert format_doc_for_embedding("body", title="Doc", task_mode=mode) == expected_doc_prefix + "body"
    assert format_doc_for_embedding("body", title="", task_mode=mode) == "title: none | text: body"


def test_embedding_formatting_rejects_invalid_mode_and_double_prefixing() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding task_mode"):
        format_query_for_embedding("who?", task_mode="unknown-mode")
    with pytest.raises(ValueError, match="Unsupported embedding task_mode"):
        format_doc_for_embedding("body", title="Doc", task_mode="unknown-mode")
    with pytest.raises(ValueError, match="already prefixed"):
        format_query_for_embedding("task: question answering | query: who?")
    with pytest.raises(ValueError, match="already prefixed"):
        format_doc_for_embedding("title: none | text: body")


def test_check_benchmark_integrity_clean_slate_rules() -> None:
    assert (
        check_benchmark_integrity(
            [{"answer": "Ada", "retrieved_chunk_ids": [1]}, {"answer": "Grace", "retrieved_chunk_ids": []}],
            {"num_passage_nodes": 3},
            expected_doc_count=3,
        )
        == []
    )
    assert (
        check_benchmark_integrity(
            [{"answer": "Ada", "graph_seeds": ["seed"]}, {"answer": "Grace", "graph_seeds": []}],
            {"num_passage_nodes": 3},
            expected_doc_count=3,
        )
        == []
    )
    errs = check_benchmark_integrity(
        [{"answer": "", "retrieved_chunk_ids": []}, {"answer": "Grace", "retrieved_chunk_ids": []}],
        {"num_passage_nodes": 0},
        expected_doc_count=2,
        extraction_errors=["idx failed"],
        qa_errors=["qa failed"],
        include_qa=True,
    )
    assert len(errs) == 6
    assert any("extraction_llm: idx failed" in e for e in errs)
    assert any("qa_llm: qa failed" in e for e in errs)
    assert any("mismatch" in e for e in errs)
    assert any("No chunks indexed" in e for e in errs)
    assert any("Question 1: empty QA answer" in e for e in errs)
    assert any("empty retrieval results" in e for e in errs)

    # Partial document drops cannot hide behind multi-chunk passage counts
    drop_errs = check_benchmark_integrity(
        [{"answer": "Ada", "retrieved_chunk_ids": [1]}],
        {"num_passage_nodes": 5, "num_indexed_docs": 2},
        expected_doc_count=3,
    )
    assert drop_errs == ["Index passage count mismatch: 2 != 3"]


def test_evaluate_benchmark_gate_preserves_null_qa_and_fails_fast_in_eval_mode() -> None:
    report = {
        "evaluation_metrics": {
            "hipporag": {
                "retrieval": {"Recall@1": 0.6, "Recall@2": 0.75, "Recall@5": 0.85},
                "qa": None,
            }
        }
    }
    spec = {
        "datasets": {
            "popqa": {
                "monkeypatched_hipporag": {
                    "Recall@1": 0.5,
                    "Recall@2": 0.7,
                    "Recall@5": 0.8,
                    "ExactMatch": 0.7,
                    "F1": 0.75,
                }
            }
        }
    }

    ret_gate = evaluate_benchmark_gate({"popqa": report}, spec, mode="retrieval")
    assert ret_gate["overall_gate_passed"] is True
    ret_ds = ret_gate["datasets"]["popqa"]
    assert ret_ds["yaw_hipporag"]["ExactMatch"] is None
    assert ret_ds["yaw_hipporag"]["F1"] is None
    assert ret_ds["delta_yaw_vs_hipporag"]["ExactMatch"] is None
    assert ret_ds["delta_yaw_vs_hipporag"]["F1"] is None

    eval_gate = evaluate_benchmark_gate({"popqa": report}, spec, mode="eval")
    assert eval_gate["overall_gate_passed"] is False
    eval_ds = eval_gate["datasets"]["popqa"]
    assert eval_ds["gate_passed"] is False
    assert eval_ds["yaw_hipporag"]["ExactMatch"] is None
    assert eval_ds["yaw_hipporag"]["F1"] is None
    assert eval_ds["delta_yaw_vs_hipporag"]["ExactMatch"] is None
    assert eval_ds["delta_yaw_vs_hipporag"]["F1"] is None
    assert "Missing QA F1 metric in eval mode" in eval_ds["failure_reasons"]
    assert "Missing QA ExactMatch metric in eval mode" in eval_ds["failure_reasons"]
