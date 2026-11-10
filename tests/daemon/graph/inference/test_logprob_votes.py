"""Measured decision-token confidence, with no completion-likelihood proxy."""

import json
import math

import pytest

from daemon.graph.synonym_protocol import response_logprobs
from yaw.daemon.graph.inference.confidence import extraction_logprob_votes, judge_logprob_votes


def test_votes_ignore_reasoning_syntax_and_escaped_field_names():
    rows = [
        {"reasoning": 'Å ball: "related": false, "pair_id": "fake"', "related": True, "pair_id": "a|b"},
        {"pair_id": "b|c", "reasoning": "Different senses.", "related": False},
    ]
    metadata = response_logprobs(json.dumps({"decisions": rows}, ensure_ascii=False), 0.64)
    for token in metadata["content"]:
        if token["token"] not in ("true", "false"):
            token["logprob"] = -100
    decisions = [{"pair_id": row["pair_id"], "related": row["related"]} for row in reversed(rows)]
    judge_logprob_votes(metadata, decisions)
    assert [row["llm_prob"] for row in decisions] == pytest.approx([0.64, 0.64])


def test_joint_boolean_pieces_then_log_opinion_pool():
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","broader":true,"narrower":false}]}')
    tokens = metadata["content"]
    index = next(i for i, token in enumerate(tokens) if token["token"] == "true")
    tokens[index : index + 1] = [
        {"token": "tr", "bytes": list(b"tr"), "logprob": math.log(0.5)},
        {"token": "ue", "bytes": list(b"ue"), "logprob": math.log(0.8)},
    ]
    decisions = [{"pair_id": "a|b", "broader": True, "narrower": False}]
    judge_logprob_votes(metadata, decisions)
    assert decisions[0]["llm_prob"] == pytest.approx(math.sqrt(0.5 * 0.8 * 0.81))


@pytest.mark.parametrize("metadata", [None, {"content": []}, {"content": [{"bytes": None}]}])
def test_unavailable_metadata_is_none(metadata):
    decisions = [{"pair_id": "a|b"}]
    judge_logprob_votes(metadata, decisions)
    assert decisions[0]["llm_prob"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.1])
def test_invalid_decision_logprob_fails(value):
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","related":true}]}')
    next(token for token in metadata["content"] if token["token"] == "true")["logprob"] = value
    with pytest.raises(ValueError, match="Invalid decision token"):
        judge_logprob_votes(metadata, [{"pair_id": "a|b", "related": True}])


def test_unmeasured_value_is_none_and_mismatched_pair_fails():
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","related":true}]}')
    next(token for token in metadata["content"] if token["token"] == "true")["logprob"] = None
    decisions = [{"pair_id": "a|b", "related": True}]
    judge_logprob_votes(metadata, decisions)
    assert decisions[0]["llm_prob"] is None
    with pytest.raises(ValueError, match="do not cover"):
        judge_logprob_votes(metadata, [{"pair_id": "other"}])


def test_metadata_must_match_boolean_values_and_field_coverage():
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","related":false}]}')
    with pytest.raises(ValueError, match="do not match"):
        judge_logprob_votes(metadata, [{"pair_id": "a|b", "related": True}])
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","broader":true}]}')
    with pytest.raises(ValueError, match="do not match"):
        judge_logprob_votes(metadata, [{"pair_id": "a|b", "broader": True, "narrower": False}])


@pytest.mark.parametrize(
    "field,records",
    [
        ("named_entities", [["Å ball", 'A label containing "quotes" and {braces}.'], ["Port", "A destination."]]),
        ("triples", [{"sub": ["Orion", "A service."], "pred": "reports to", "obj": "Anchor", "def": "Orion reports."}]),
        ("triples", []),
    ],
)
def test_extraction_record_alignment_and_none(field, records):
    metadata = response_logprobs(json.dumps({field: records}, ensure_ascii=False), 0.64)
    assert extraction_logprob_votes(metadata, field, records) == pytest.approx([0.64] * len(records))
    assert extraction_logprob_votes(None, field, records) == (None,) * len(records)
    with pytest.raises(ValueError, match="do not"):
        extraction_logprob_votes(metadata, field, ["different output"])


def test_extraction_value_length_normalization_ignores_keys_and_syntax():
    raw = '{"named_entities":[["Orion","A longer definition"]]}'
    metadata = response_logprobs(raw, 0.81)
    tokens = metadata["content"]
    index = next(i for i, token in enumerate(tokens) if token["token"] == '"A longer definition"')
    pieces = ('"', "A ", "longer ", "definition", '"')
    tokens[index : index + 1] = [
        {"token": piece, "bytes": list(piece.encode()), "logprob": math.log(0.81) if piece != '"' else -0.001}
        for piece in pieces
    ]
    assert extraction_logprob_votes(metadata, "named_entities", [["Orion", "A longer definition"]]) == (
        pytest.approx([0.81])
    )


def test_duplicate_extraction_fields_are_not_scored_as_a_single_observation():
    metadata = response_logprobs('{"triples":[{"sub":"Other","sub":"Orion"}]}')
    with pytest.raises(ValueError, match="Duplicate extraction field"):
        extraction_logprob_votes(metadata, "triples", [{"sub": "Orion"}])
