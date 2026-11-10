"""Malformed model responses, Unicode senses and graph-coordinate invariants."""

import json

import dspy
import numpy as np
import pytest

from yaw.core import StructuredTriple
from yaw.daemon.graph.inference.parsers import (
    coerce_generated_facts,
    compute_qa_em_f1,
    extract_entity_definitions,
    extract_structured_triples,
    match_facts_to_candidate_indices,
    parse_label_decisions,
    parse_shacl_repairs,
    text_processing,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"subject": [" Ada ", "Mathematician"], "predicate": "wrote", "object": ["Note G"]},
            StructuredTriple("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "Ada wrote Note G."),
        ),
        (
            ("Ada", "Mathematician", "wrote", "Note G", "Algorithm"),
            StructuredTriple("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "Ada wrote Note G."),
        ),
        (
            ("Ada", "", "wrote", "Note G", ""),
            StructuredTriple("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "Ada wrote Note G."),
        ),
        (
            ("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "An explicit statement."),
            StructuredTriple("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "An explicit statement."),
        ),
        (
            (["Ada"], "wrote", ["Note G", None], None),
            StructuredTriple("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "Ada wrote Note G."),
        ),
        (
            {"sub": "Ada", "pred": "wrote", "obj": "Note G", "subjDef": "Mathematician", "def": None},
            StructuredTriple("Ada", "Mathematician", "wrote", "Note G", "Algorithm", "Ada wrote Note G."),
        ),
    ],
)
def test_structured_interchange_forms_preserve_endpoints_and_definitions(payload, expected):
    result = extract_structured_triples([payload], entity_definitions={"Ada": "Mathematician", "Note G": "Algorithm"})
    assert result == (expected,)
    assert extract_structured_triples([expected, expected]) == (expected,)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        42,
        {},
        {"sub": "A", "obj": "B"},
        {"sub": 1, "pred": "p", "obj": "B"},
        {"sub": "A", "pred": "p", "obj": 1},
        {"sub": "A", "pred": "", "obj": "B"},
        ["", "", "p", "B", ""],
        ["A", "", "", "B", ""],
        ["A", "p", ""],
        ["short"],
    ],
)
def test_invalid_triples_do_not_enter_the_graph(payload):
    assert extract_structured_triples([payload]) == ()


@pytest.mark.parametrize("payload", [[None, "p", "B"], ["A", None, "B"], ["A", "p", None]])
def test_missing_sequence_endpoints_match_missing_dictionary_endpoint_validation(payload):
    assert extract_structured_triples([payload]) == ()


@pytest.mark.parametrize(
    "payload",
    [
        ([None], "p", "B"),
        ("A", "p", [None]),
        {"sub": [None], "pred": "p", "obj": "B"},
        {"sub": "A", "pred": "p", "obj": [None]},
    ],
)
def test_null_inline_names_are_not_materialized_as_named_concepts(payload):
    assert extract_structured_triples([payload]) == ()


def test_null_definitions_are_missing_and_can_be_filled_from_context():
    assert extract_entity_definitions({"named_entities": [{"name": "A", "definition": None}]}) == (("A", ""),)
    result = extract_structured_triples(
        [{"sub": "A", "pred": "p", "obj": "B", "subjDef": None, "objDef": None}],
        entity_definitions={"A": "Alpha", "B": "Beta"},
    )
    assert result[0].subj_def == "Alpha"
    assert result[0].obj_def == "Beta"


def test_same_label_different_senses_do_not_collapse_into_one_concept():
    definitions = json.loads(
        '[["Mercury", "A planet"], ["mercury", "An element"], ["Mercury", "A PLANET"], ["", ""], ["short"]]'
    )
    extracted = extract_entity_definitions(definitions)
    assert extracted == (("Mercury", "A planet"), ("mercury", "An element"), ("short", ""))
    result = extract_structured_triples([("Mercury", "orbits", "Sun")], entity_definitions=definitions)
    assert len(result) == 1
    assert result[0].subj_def == "A planet"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('noise {bad} {"other": []} {"entity_definitions": [["Ada", "Mathematician"]]}', (("Ada", "Mathematician"),)),
        ('```json\n[["Ada", "Mathematician"]]\n```', (("Ada", "Mathematician"),)),
        ('```\n["Ada"]\n```', (("Ada", ""),)),
        ({"entity_definitions": [["Ada", "Mathematician"]]}, (("Ada", "Mathematician"),)),
        ({"named_entities": "wrong type"}, ()),
        (None, ()),
    ],
)
def test_entity_response_envelopes_and_truncation(payload, expected):
    assert extract_entity_definitions(payload) == expected
    for bad in ('{"entity_definitions": [["Ada", "Mathematician"], ["unfinished', "[42]", "not JSON"):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            extract_entity_definitions(bad)


def test_entity_definitions_fill_missing_data_and_preserve_homographs():
    payload = [
        "Ada",
        ["Ada", "Mathematician"],
        ["Ada", None],
        ["Ada", "MATHEMATICIAN"],
        {"name": "Ada", "definition": "A different person"},
        ["Unicode", "İ"],
        ["Unicode", "i\u0307"],
        "",
        42,
    ]
    assert extract_entity_definitions(payload) == (
        ("Ada", "Mathematician"),
        ("Ada", "A different person"),
        ("Unicode", "i\u0307"),
    )


@pytest.mark.parametrize("payload", [42, None])
def test_broken_structured_response_envelopes_produce_no_triples(payload):
    assert extract_structured_triples(payload) == ()
    for bad in ("", "not JSON", '{"triples": "wrong type"}'):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            extract_structured_triples(bad)


@pytest.mark.parametrize(
    "payload",
    ['[["Ada", "wrote", "Note G"]]', '{"triples": [["Ada", "wrote", "Note G"]]}'],
)
def test_bare_and_wrapped_structured_triples_are_equivalent(payload):
    assert extract_structured_triples(payload)[0].as_tuple() == ("Ada", "wrote", "Note G")


@pytest.mark.parametrize("payload", ["not JSON", '{"triples": "wrong type"}'])
def test_plain_triple_parser_rejects_invalid_response_shapes(payload):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract_structured_triples(payload)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('[[ ## fact_after_filter ## ]]\n[["A", "p", "B"]]', [["A", "p", "B"]]),
        ("[[ ## fact_after_filter ## ]]\n{'fact': [('A', 'p', 'B'), ['short']]}", [["A", "p", "B"]]),
        ('[[ ## fact_after_filter ## ]]\n{"fact":"wrong type"}', []),
        ("[[ ## fact_after_filter ## ]]\nnot JSON", []),
        ('[["A", "p", "B"], ["short"]]', [["A", "p", "B"]]),
        ('{"fact":[["A", "p", "B"]]}', [["A", "p", "B"]]),
        ("not JSON", []),
    ],
)
def test_recognition_memory_accepts_only_three_coordinate_facts(payload, expected):
    assert coerce_generated_facts(payload) == expected


@pytest.mark.parametrize("envelope", ["prediction", "dictionary", "list", "json"])
def test_fact_envelopes_preserve_actual_candidates(envelope):
    facts = [["A", "p", "B"], ["short"], [1, 2, 3]]
    payload = {
        "prediction": dspy.Prediction(fact=facts),
        "dictionary": {"fact": facts},
        "list": facts,
        "json": json.dumps({"fact": facts}),
    }[envelope]
    assert coerce_generated_facts(payload) == [["A", "p", "B"], ["1", "2", "3"]]
    assert coerce_generated_facts(None) == []
    assert coerce_generated_facts({"fact": "wrong type"}) == []


def test_entity_definitions_mapping_and_deduplication():
    assert extract_entity_definitions({"Ada Lovelace": "English mathematician", "Engine": None}) == (
        ("Ada Lovelace", "English mathematician"),
        ("Engine", ""),
    )
    assert extract_entity_definitions(
        [("Ada Lovelace", ""), ("Ada Lovelace", "English mathematician")],
    ) == (("Ada Lovelace", "English mathematician"),)


def test_candidate_matching_retains_original_order_indices_with_numpy_string_values():
    candidates = [(np.str_("A"), "p", "B"), ("C", "p", "D")]
    assert match_facts_to_candidate_indices(candidates, [["A", "p", "B"]], max_facts=None) == [0]
    assert match_facts_to_candidate_indices(candidates, [["C", "p", "D"], ["C", "p", "D"]]) == [1]
    assert match_facts_to_candidate_indices([], [["A", "p", "B"]]) == []


def test_empty_answer_metrics_and_triple_filtering():
    assert compute_qa_em_f1("", [""]) == (1.0, 0.0)
    assert compute_qa_em_f1("A", []) == (0.0, 0.0)
    assert compute_qa_em_f1("A B", ["B C"]) == (0.0, pytest.approx(2 / 3))
    assert text_processing(json.loads("12")) == "12"
    assert [
        list(t.as_tuple())
        for t in extract_structured_triples([{}, ["short"], StructuredTriple("A", pred="p", obj="B")])
    ] == [["A", "p", "B"]]


@pytest.mark.parametrize("name", ['A"B', "A\\B"])
def test_truncation_repair_preserves_escaped_characters_in_completed_names(name):
    truncated = json.dumps({"named_entities": [name]})[:-2] + ', "unfinished'
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract_entity_definitions(truncated)


def test_wrong_nested_response_envelopes_and_missing_filter_output_are_not_facts():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract_structured_triples('{"other":[["A"]], "unfinished')
    assert extract_structured_triples({"triples": [["A", "p", "B"]]})[0].as_tuple() == ("A", "p", "B")
    assert match_facts_to_candidate_indices([("A", "p", "B")], coerce_generated_facts("No structured response")) == []


def test_invalid_label_items_cannot_displace_a_later_valid_decision():
    raw = [
        None,
        {"pair_id": "", "reasoning": "valid", "equivalent": True, "definition": "Valid def", "hidden_label": False},
        {"pair_id": "0", "reasoning": "valid", "equivalent": True, "definition": "Valid def", "hidden_label": False},
        {"pair_id": "a|b", "reasoning": "", "equivalent": True, "definition": "Valid def", "hidden_label": False},
        {"pair_id": "a|b", "reasoning": "valid", "equivalent": True, "definition": "Valid def", "hidden_label": False},
    ]
    assert parse_label_decisions(raw) == [
        {
            "pair_id": "a|b",
            "reasoning": "valid",
            "equivalent": True,
            "alt_label": True,
            "hidden_label": False,
            "relation": "skos:altLabel",
            "definition": "Valid def",
        }
    ]


@pytest.mark.parametrize("envelope", ["prediction", "dictionary", "json", "bare-json", "list"])
def test_synonym_and_shacl_decisions_reject_unknown_actions(envelope):
    decisions = [
        {"pair_id": "a|b", "reasoning": "valid", "equivalent": True, "definition": "Valid def", "hidden_label": False},
        {"pair_id": "c|d", "reasoning": "valid", "equivalent": True, "definition": "Valid def", "hidden_label": True},
        {"pair_id": "e", "reasoning": "invalid pair_id"},
        "invalid",
    ]
    repairs = [
        {"focus_node": "A", "action": "set_definition", "target": "A concept"},
        {"focus_node": "A", "action": "unknown"},
        "invalid",
    ]

    def wrap(field, value):
        return {
            "prediction": dspy.Prediction(**{field: value}),
            "dictionary": {field: value},
            "json": json.dumps({field: value}),
            "bare-json": json.dumps(value),
            "list": value,
        }[envelope]

    assert parse_label_decisions(wrap("decisions", decisions)) == [
        {
            "pair_id": "a|b",
            "reasoning": "valid",
            "equivalent": True,
            "alt_label": True,
            "hidden_label": False,
            "relation": "skos:altLabel",
            "definition": "Valid def",
        },
        {
            "pair_id": "c|d",
            "reasoning": "valid",
            "equivalent": True,
            "alt_label": False,
            "hidden_label": True,
            "relation": "skos:hiddenLabel",
            "definition": "Valid def",
        },
    ]
    with pytest.raises(ValueError, match="Invalid SHACL repair"):
        parse_shacl_repairs(wrap("repairs", repairs))
    assert parse_shacl_repairs(wrap("repairs", repairs[:1])) == [
        {"focus_node": "A", "action": "set_definition", "value": "A concept"},
    ]
    with pytest.raises(json.JSONDecodeError):
        parse_label_decisions("not JSON")
    with pytest.raises(json.JSONDecodeError):
        parse_shacl_repairs("not JSON")
