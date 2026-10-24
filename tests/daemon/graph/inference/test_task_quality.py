"""Quality-only GEPA rewards, verified with native DSPy metrics and adapter."""

import math

import dspy
import pytest
from dspy.evaluate.metrics import F1
from dspy.teleprompt.gepa.gepa_utils import _extract_score_and_objective_scores

from yaw.daemon.graph.inference import compute_qa_em_f1, compute_task_quality_score
from yaw.daemon.graph.inference.reflection import deterministic_evidence, reflection_examples, resolve_evidence


def test_all_seven_quality_metrics_have_equal_arithmetic_weight():
    result = compute_task_quality_score(
        0.6,
        recall_at_1=0.2,
        recall_at_2=0.4,
        recall_at_10=0.8,
        recall_at_20=1.0,
        em=0.25,
        f1=0.75,
        tokens=250_000,
    )
    expected = math.fsum([0.2, 0.4, 0.6, 0.8, 1.0, 0.25, 0.75]) / 7
    score, objectives = _extract_score_and_objective_scores(result)
    assert score == pytest.approx(expected)
    assert set(objectives) == {"recall_at_1", "recall_at_2", "recall_at_5", "recall_at_10", "recall_at_20", "em", "f1"}
    assert "Task quality arithmetic mean" in result.feedback


@pytest.mark.parametrize("tokens", [0, 1, 30_000, 10**9])
def test_tokens_repairs_and_integrity_do_not_change_reward_or_pareto_frontier(tokens):
    baseline = compute_task_quality_score(0.8, em=0.5, f1=0.7, tokens=10)
    expensive = compute_task_quality_score(
        0.8,
        em=0.5,
        f1=0.7,
        tokens=tokens,
        synonym_judge_tokens=tokens,
        shacl_healer_tokens=tokens,
        shacl_conformance=0.0,
        shacl_violations_initial=100,
        shacl_violations_remaining=50,
        missing_entity_defs_count=40,
        missing_entity_defs_tokens=tokens,
    )
    assert expensive.score == baseline.score
    assert expensive.objective_scores == baseline.objective_scores
    assert expensive.tokens == tokens
    assert expensive.shacl_conforms == 0.0


@pytest.mark.parametrize(
    "zero_key", ["recall_at_1", "recall_at_2", "recall_at_5", "recall_at_10", "recall_at_20", "em", "f1"]
)
def test_zero_components_do_not_collapse_positive_metrics(zero_key):
    values = dict.fromkeys(
        ["recall_at_1", "recall_at_2", "recall_at_5", "recall_at_10", "recall_at_20", "em", "f1"], 1.0
    )
    values[zero_key] = 0.0
    result = compute_task_quality_score(**values, tokens=0)
    assert result.score == pytest.approx(6.0 / 7.0)
    assert result.objective_scores[zero_key] == 0.0


def test_em_and_f1_zero_with_positive_recall_has_positive_score():
    result = compute_task_quality_score(
        0.5,
        recall_at_1=0.2,
        recall_at_2=0.4,
        recall_at_10=0.8,
        recall_at_20=1.0,
        em=0.0,
        f1=0.0,
        tokens=0,
    )
    assert result.score > 0.0
    assert result.score == pytest.approx(math.fsum([0.2, 0.4, 0.5, 0.8, 1.0, 0.0, 0.0]) / 7)
    all_zero = compute_task_quality_score(
        0.0,
        recall_at_1=0.0,
        recall_at_2=0.0,
        recall_at_10=0.0,
        recall_at_20=0.0,
        em=0.0,
        f1=0.0,
        tokens=0,
    )
    assert all_zero.score == 0.0


def test_qa_only_and_retrieval_only_do_not_invent_objectives():
    qa = compute_task_quality_score(None, em=0.5, f1=0.8, tokens=0)
    assert qa.score == pytest.approx((0.5 + 0.8) / 2)
    assert qa.objective_scores == {"em": 0.5, "f1": 0.8}
    retrieval = compute_task_quality_score(
        0.6,
        recall_at_1=0.2,
        recall_at_2=0.4,
        recall_at_10=0.8,
        recall_at_20=1.0,
        tokens=0,
    )
    assert retrieval.score == pytest.approx((0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5)
    assert "em" not in retrieval.objective_scores
    with pytest.raises(ValueError, match="at least one labeled metric"):
        compute_task_quality_score(None, tokens=0)


@pytest.mark.parametrize(
    ("prediction", "answers"),
    [
        ("the Eiffel Tower", ["Eiffel Tower", "Louvre"]),
        ("Alan Turing was a mathematician", ["Alan Turing"]),
        ("red blue", ["red green", "red blue yellow"]),
        ("red red blue", ["red blue blue"]),
        ("cathedral", ["cat"]),
        ("", [""]),
    ],
)
def test_qa_scores_match_dspy_without_substring_or_threshold_shortcuts(prediction, answers):
    em, f1 = compute_qa_em_f1(prediction, answers)
    assert em == float(
        dspy.evaluate.answer_exact_match(dspy.Example(answer=answers), dspy.Prediction(answer=prediction))
    )
    assert f1 == pytest.approx(F1(prediction, answers))


def test_partial_answer_is_not_promoted_to_exact_match():
    em, f1 = compute_qa_em_f1("Alan Turing was a mathematician", ["Alan Turing"])
    assert em == 0.0
    assert 0.0 < f1 < 1.0
    assert compute_qa_em_f1("Eiffel Tower", ["the Eiffel Tower"]) == (1.0, 1.0)


def test_reflection_fallback_reads_quality_feedback_to_select_failures_and_contrasts():
    examples = reflection_examples(
        [
            {"Feedback": compute_task_quality_score(value, tokens=0).feedback}
            for value in (0.9, 0.2, 0.8, 0.4, 0.1, 0.6)
        ]
    )
    resolved = resolve_evidence(examples, deterministic_evidence(examples))
    feedbacks = [item["evidence"]["Feedback"] for item in resolved]
    for actual, expected in zip(feedbacks, (0.1, 0.2, 0.4, 0.9, 0.8), strict=True):
        assert f"Task quality arithmetic mean={expected:.6f}" in actual
