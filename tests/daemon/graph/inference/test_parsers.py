"""Tests for decision parsers and task quality score with the 3-stage cascade."""

import json

import pytest

from yaw.core import VALUE_BEARING_SHACL_ACTIONS
from yaw.daemon.graph.inference.parsers import (
    compute_task_quality_score,
    parse_hierarchy_decisions,
    parse_label_decisions,
    parse_related_decisions,
    parse_shacl_repairs,
    shacl_repair_actions,
)


def test_parse_label_decisions_truth_table() -> None:
    raw = {
        "decisions": [
            {
                "pair_id": "eq|dup",
                "reasoning": "Identical concept across passages",
                "equivalent": True,
                "definition": "Unified canonical concept definition.",
            },
            {"pair_id": "a|b", "reasoning": "Standard synonym", "equivalent": False, "alt_label": True},
            {
                "pair_id": "c|d",
                "reasoning": "Typo in source",
                "equivalent": False,
                "alt_label": False,
                "hidden_label": True,
            },
            {
                "pair_id": "g|h",
                "reasoning": "Distinct concepts",
                "equivalent": False,
                "alt_label": False,
                "hidden_label": False,
            },
        ]
    }
    decisions = parse_label_decisions(raw)
    assert len(decisions) == 4
    assert decisions[0]["equivalent"] is True
    assert decisions[0]["definition"] == "Unified canonical concept definition."
    assert decisions[0]["relation"] == "skos:altLabel"
    assert decisions[1]["equivalent"] is False
    assert decisions[1]["alt_label"] is True
    assert decisions[1]["hidden_label"] is False
    assert decisions[1]["relation"] == "skos:altLabel"
    assert decisions[2]["relation"] == "skos:hiddenLabel"
    assert decisions[3]["relation"] is None


def test_parse_label_decisions_json_string() -> None:
    payload = json.dumps(
        {"decisions": [{"pair_id": "x|y", "reasoning": "Acronym variant", "equivalent": False, "alt_label": True}]}
    )
    decisions = parse_label_decisions(payload)
    assert len(decisions) == 1
    assert decisions[0]["pair_id"] == "x|y"
    assert decisions[0]["relation"] == "skos:altLabel"


def test_parse_label_decisions_skips_invalid() -> None:
    raw = [
        {"pair_id": "invalid", "reasoning": "bad id", "equivalent": False, "alt_label": True},
        {"pair_id": "a|a", "reasoning": "self loop", "equivalent": False, "alt_label": True},
        {"pair_id": "a|b", "reasoning": "", "equivalent": False, "alt_label": True},
        {"pair_id": "a|b", "reasoning": "missing def for equiv", "equivalent": True, "definition": ""},
        "not a dict",
        {
            "pair_id": "valid|ok",
            "reasoning": "valid reasoning",
            "equivalent": False,
            "alt_label": False,
            "hidden_label": True,
        },
    ]
    decisions = parse_label_decisions(raw)
    assert len(decisions) == 1
    assert decisions[0]["pair_id"] == "valid|ok"
    assert decisions[0]["relation"] == "skos:hiddenLabel"


def test_parse_hierarchy_decisions_truth_table() -> None:
    raw = {
        "decisions": [
            {"pair_id": "a|b", "reasoning": "A is subtype of B", "broader": True, "narrower": False},
            {"pair_id": "c|d", "reasoning": "D is subtype of C", "broader": False, "narrower": True},
            {"pair_id": "e|f", "reasoning": "Bidirectional cycle", "broader": True, "narrower": True},
            {"pair_id": "g|h", "reasoning": "No taxonomic link", "broader": False, "narrower": False},
        ]
    }
    decisions = parse_hierarchy_decisions(raw)
    assert len(decisions) == 4
    # (True, False) -> broader, a_to_b
    assert decisions[0]["relation"] == "skos:broader"
    assert decisions[0]["source_direction"] == "a_to_b"

    # (False, True) -> broader, b_to_a (SKOS S25 inverse)
    assert decisions[1]["relation"] == "skos:broader"
    assert decisions[1]["source_direction"] == "b_to_a"

    # (True, True) -> related (SKOS S27 degrade)
    assert decisions[2]["relation"] == "skos:related"
    assert decisions[2]["source_direction"] is None

    # (False, False) -> None (advances to Stage 3)
    assert decisions[3]["relation"] is None
    assert decisions[3]["source_direction"] is None


def test_parse_related_decisions_truth_table() -> None:
    raw = {
        "decisions": [
            {"pair_id": "a|b", "reasoning": "Domain interaction", "related": True},
            {"pair_id": "c|d", "reasoning": "Completely disjoint", "related": False},
        ]
    }
    decisions = parse_related_decisions(raw)
    assert len(decisions) == 2
    assert decisions[0]["relation"] == "skos:related"
    assert decisions[1]["relation"] == "owl:differentFrom"


def test_compute_task_quality_score_stage_tokens() -> None:
    pred = compute_task_quality_score(
        0.8,
        tokens=150,
        label_judge_tokens=42,
        hierarchy_judge_tokens=30,
        related_judge_tokens=15,
    )
    assert pred.label_judge_tokens == 42
    assert pred.hierarchy_judge_tokens == 30
    assert pred.related_judge_tokens == 15
    assert not hasattr(pred, "synonym_judge_tokens")


def test_stage_invariants_label_judge() -> None:
    import dspy

    from yaw.daemon.graph.inference.invariants import validate_stage_invariants

    inputs = {
        "candidate_pairs": json.dumps(
            {
                "format": "synonym-evidence-v2",
                "pairs": [{"pair_id": "a|b"}],
            }
        )
    }
    pred_pass = dspy.Prediction(
        decisions=[{"pair_id": "a|b", "reasoning": "Valid reasoning", "equivalent": False, "alt_label": True}]
    )
    score, feedback = validate_stage_invariants("label_judge", inputs, pred_pass)
    assert score == 1.0
    assert "PASS [label_judge]" in feedback

    pred_equiv_pass = dspy.Prediction(
        decisions=[
            {
                "pair_id": "a|b",
                "reasoning": "Same concept",
                "equivalent": True,
                "definition": "Unified concept definition.",
            }
        ]
    )
    score_eq, feedback_eq = validate_stage_invariants("label_judge", inputs, pred_equiv_pass)
    assert score_eq == 1.0
    assert "PASS [label_judge]" in feedback_eq

    # Empty reasoning fails
    pred_empty_reason = dspy.Prediction(
        decisions=[{"pair_id": "a|b", "reasoning": "", "equivalent": False, "alt_label": True}]
    )
    score_bad, feedback_bad = validate_stage_invariants("label_judge", inputs, pred_empty_reason)
    assert score_bad < 1.0
    assert "Empty reasoning string" in feedback_bad

    # Missing boolean field fails when alt_label=False and hidden_label is omitted
    pred_missing_bool = dspy.Prediction(
        decisions=[{"pair_id": "a|b", "reasoning": "Valid reasoning", "equivalent": False, "alt_label": False}]
    )
    score_bad2, feedback_bad2 = validate_stage_invariants("label_judge", inputs, pred_missing_bool)
    assert score_bad2 < 1.0
    assert "Missing boolean field 'hidden_label'" in feedback_bad2

    # Conflicting alt_label=True and hidden_label=True fails
    pred_conflict = dspy.Prediction(
        decisions=[
            {
                "pair_id": "a|b",
                "reasoning": "Conflict",
                "equivalent": False,
                "alt_label": True,
                "hidden_label": True,
            }
        ]
    )
    score_bad3, feedback_bad3 = validate_stage_invariants("label_judge", inputs, pred_conflict)
    assert score_bad3 < 1.0
    assert "Conflicting alt_label=True and hidden_label=True" in feedback_bad3


def test_stage_invariants_hierarchy_and_related_judges() -> None:
    import dspy

    from yaw.daemon.graph.inference.invariants import validate_stage_invariants

    inputs_h = {
        "candidate_pairs": json.dumps(
            {
                "format": "synonym-evidence-v2",
                "pairs": [{"pair_id": "cat|feline"}],
            }
        )
    }
    pred_h = dspy.Prediction(
        decisions=[{"pair_id": "cat|feline", "reasoning": "Cat is subtype", "broader": True, "narrower": False}]
    )
    score, feedback = validate_stage_invariants("hierarchy_judge", inputs_h, pred_h)
    assert score == 1.0

    inputs_r = {
        "candidate_pairs": json.dumps(
            {
                "format": "synonym-evidence-v2",
                "pairs": [{"pair_id": "cat|dog"}],
            }
        )
    }
    pred_r = dspy.Prediction(decisions=[{"pair_id": "cat|dog", "reasoning": "Co-occurring pets", "related": True}])
    score_r, feedback_r = validate_stage_invariants("related_judge", inputs_r, pred_r)
    assert score_r == 1.0


def test_shacl_repair_actions_maps_scalar_and_resource_label_shapes() -> None:
    violations = [
        {"focus_node": "entity-scalar", "shape": "ConceptShape/noScalarLiterals", "result_path": "skos:prefLabel"},
        {"focus_node": "entity-res-shape", "shape": "ConceptShape/resourceLabel", "result_path": "skos:prefLabel"},
        {
            "focus_node": "entity-res-source",
            "shape": "OtherShape",
            "source_shape": "ConceptShape/resourceLabel",
            "result_path": "skos:prefLabel",
        },
        {"focus_node": "fact-def", "shape": "StatementShape/closedWorld", "result_path": "skos:definition"},
    ]
    actions = shacl_repair_actions(violations)
    assert actions["entity-scalar"] == {"set_pref_label"}
    assert actions["entity-res-shape"] == {"set_pref_label"}
    assert actions["entity-res-source"] == {"set_pref_label"}
    assert actions["fact-def"] == {"set_definition"}


def test_parse_shacl_repairs_abstention_coverage_and_lexical_validation() -> None:
    allowed = {"entity-1": {"set_pref_label"}, "fact-1": {"set_definition"}}

    # 1. Empty abstention is permitted and returns []
    assert parse_shacl_repairs({"repairs": []}, allowed_actions=allowed) == []

    # 2. Non-empty partial repair list missing a focus node raises ValueError
    with pytest.raises(ValueError, match="Incomplete SHACL repairs"):
        parse_shacl_repairs(
            {
                "repairs": [
                    {"focus_node": "entity-1", "action": "set_pref_label", "value": "HTTP 429 Too Many Requests"}
                ]
            },
            allowed_actions=allowed,
        )

    # 3. Empty or whitespace-only value on value-bearing actions raises ValueError
    for action in (*VALUE_BEARING_SHACL_ACTIONS, "ground_fact_source", "set_fact_source"):
        for empty_val in ("", "   "):
            with pytest.raises(ValueError):
                parse_shacl_repairs([{"focus_node": "node-1", "action": action, "value": empty_val}])

    # 4. Scalar or non-resource labels on set_pref_label raise ValueError
    for invalid_label in ("429", "2024-01-01", "100%", "---"):
        with pytest.raises(ValueError):
            parse_shacl_repairs(
                [{"focus_node": "entity-1", "action": "set_pref_label", "value": invalid_label}],
                allowed_actions={"entity-1": {"set_pref_label"}},
            )

    # 5. Valid lexical label on set_pref_label is accepted
    parsed = parse_shacl_repairs(
        {
            "repairs": [
                {"focus_node": "entity-1", "action": "set_pref_label", "value": "HTTP 429 Too Many Requests"},
                {
                    "focus_node": "fact-1",
                    "action": "set_definition",
                    "value": "The gateway rate-limits excess client traffic with HTTP 429.",
                },
            ]
        },
        allowed_actions=allowed,
    )
    assert parsed == [
        {"focus_node": "entity-1", "action": "set_pref_label", "value": "HTTP 429 Too Many Requests"},
        {
            "focus_node": "fact-1",
            "action": "set_definition",
            "value": "The gateway rate-limits excess client traffic with HTTP 429.",
        },
    ]
