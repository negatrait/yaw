"""Measured decision-token confidence, with no completion-likelihood proxy."""

import json
import math
from dataclasses import replace

import pytest
from dspy.lm15 import TokenLogprob

from daemon.graph.synonym_protocol import response_logprobs
from yaw.daemon.graph.inference.confidence import extraction_logprob_votes, judge_logprob_votes


def test_votes_ignore_reasoning_syntax_and_escaped_field_names():
    rows = [
        {"reasoning": 'Å ball: "related": false, "pair_id": "fake"', "related": True, "pair_id": "a|b"},
        {"pair_id": "b|c", "reasoning": "Different senses.", "related": False},
    ]
    metadata = response_logprobs(json.dumps({"decisions": rows}, ensure_ascii=False), 0.64)
    metadata = tuple(
        replace(token, logprob=-100) if token.token not in ("true", "false") else token for token in metadata
    )
    decisions = [{"pair_id": row["pair_id"], "related": row["related"]} for row in reversed(rows)]
    judge_logprob_votes(metadata, decisions)
    assert [row["llm_prob"] for row in decisions] == pytest.approx([0.64, 0.64])


def test_joint_boolean_pieces_then_log_opinion_pool():
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","broader":true,"narrower":false}]}')
    tokens = list(metadata)
    index = next(i for i, token in enumerate(tokens) if token.token == "true")
    tokens[index : index + 1] = [
        TokenLogprob("tr", math.log(0.5), tuple(b"tr")),
        TokenLogprob("ue", math.log(0.8), tuple(b"ue")),
    ]
    decisions = [{"pair_id": "a|b", "broader": True, "narrower": False}]
    judge_logprob_votes(tuple(tokens), decisions)
    assert decisions[0]["llm_prob"] == pytest.approx(math.sqrt(0.5 * 0.8 * 0.81))


@pytest.mark.parametrize("metadata", [None, (), (TokenLogprob("x", -1.0, None),)])
def test_unavailable_metadata_is_none(metadata):
    decisions = [{"pair_id": "a|b"}]
    judge_logprob_votes(metadata, decisions)
    assert decisions[0]["llm_prob"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.1])
def test_invalid_decision_logprob_fails(value):
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","related":true}]}')
    metadata = tuple(replace(token, logprob=value) if token.token == "true" else token for token in metadata)
    with pytest.raises(ValueError, match="Invalid decision token"):
        judge_logprob_votes(metadata, [{"pair_id": "a|b", "related": True}])


def test_unmeasured_value_is_none_and_mismatched_pair_fails():
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","related":true}]}')
    metadata = tuple(replace(token, bytes=None) if token.token == "true" else token for token in metadata)
    decisions = [{"pair_id": "a|b", "related": True}]
    judge_logprob_votes(metadata, decisions)
    assert decisions[0]["llm_prob"] is None
    metadata = response_logprobs('{"decisions":[{"pair_id":"a|b","related":true}]}')
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
    tokens = list(metadata)
    index = next(i for i, token in enumerate(tokens) if token.token == '"A longer definition"')
    pieces = ('"', "A ", "longer ", "definition", '"')
    tokens[index : index + 1] = [
        TokenLogprob(piece, math.log(0.81) if piece != '"' else -0.001, tuple(piece.encode())) for piece in pieces
    ]
    assert extraction_logprob_votes(tuple(tokens), "named_entities", [["Orion", "A longer definition"]]) == (
        pytest.approx([0.81])
    )


def test_duplicate_extraction_fields_are_not_scored_as_a_single_observation():
    metadata = response_logprobs('{"triples":[{"sub":"Other","sub":"Orion"}]}')
    with pytest.raises(ValueError, match="Duplicate extraction field"):
        extraction_logprob_votes(metadata, "triples", [{"sub": "Orion"}])


def test_relation_confidence_measures_only_selected_value_and_validates_it():
    raw = '{"decisions":[{"pair_id":"a|b","reasoning":"explanation","relation":"skos:related"}]}'
    tokens = list(response_logprobs(raw))
    index = next(i for i, token in enumerate(tokens) if token.token == '"skos:related"')
    tokens[index : index + 1] = [
        TokenLogprob('"', -20, (34,)),
        TokenLogprob("skos:", math.log(0.5), tuple(b"skos:")),
        TokenLogprob("related", math.log(0.8), tuple(b"related")),
        TokenLogprob('"', -20, (34,)),
    ]
    decisions = [{"pair_id": "a|b", "relation": "skos:related"}]
    judge_logprob_votes(tuple(tokens), decisions)
    assert decisions[0]["llm_prob"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="do not match"):
        judge_logprob_votes(tuple(tokens), [{"pair_id": "a|b", "relation": "skos:broader"}])
