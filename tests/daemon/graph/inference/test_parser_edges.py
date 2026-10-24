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
    extract_entity_nodes,
    extract_structured_triples,
    extract_triples,
    filter_invalid_triples,
    filter_reranked_facts,
    flatten_facts,
    match_facts_to_candidate_indices,
    parse_filter_facts,
    parse_shacl_repairs,
    parse_synonym_decisions,
    strip_mistral_bracket_prefix,
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
    result = extract_structured_triples([("Mercury", "differs from", "Mercury")], entity_definitions=definitions)
    assert len(result) == 1
    assert result[0].subj_def == "A planet"
    assert result[0].obj_def == "An element"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('noise {bad} {"other": []} {"entity_definitions": [["Ada", "Mathematician"]]}', (("Ada", "Mathematician"),)),
        ('```json\n[["Ada", "Mathematician"]]\n```', (("Ada", "Mathematician"),)),
        ('```\n["Ada"]\n```', (("Ada", ""),)),
        ('{"entity_definitions": [["Ada", "Mathematician"], ["unfinished', (("Ada", "Mathematician"),)),
        ({"entity_definitions": [["Ada", "Mathematician"]]}, (("Ada", "Mathematician"),)),
        ({"named_entities": "wrong type"}, ()),
        ("[42]", ()),
        ("not JSON", ()),
        (None, ()),
    ],
)
def test_entity_response_envelopes_and_truncation(payload, expected):
    assert extract_entity_definitions(payload) == expected


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


@pytest.mark.parametrize("payload", ["", "not JSON", '{"triples": "wrong type"}', 42, None])
def test_broken_structured_response_envelopes_produce_no_triples(payload):
    assert extract_structured_triples(payload) == ()


@pytest.mark.parametrize(
    "payload",
    ['[["Ada", "wrote", "Note G"]]', '{"triples": [["Ada", "wrote", "Note G"]]}'],
)
def test_bare_and_wrapped_structured_triples_are_equivalent(payload):
    assert extract_structured_triples(payload)[0].as_tuple() == ("Ada", "wrote", "Note G")


@pytest.mark.parametrize("payload", ["not JSON", '{"triples": "wrong type"}'])
def test_plain_triple_parser_rejects_invalid_response_shapes(payload):
    assert extract_triples(payload) == ()


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
    assert parse_filter_facts(payload) == expected


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


def test_graph_coordinate_extraction_deduplicates_and_rejects_empty_or_self_edges():
    triples = [(" A! ", "links to", "B"), ("A", "links to", "B"), ("A", "p", "A"), ("", "p", "B"), ("short",)]
    nodes = extract_entity_nodes(triples)
    assert [name for _, name in nodes] == ["a", "b"]
    facts = flatten_facts(triples)
    assert len(facts) == 1
    _, subject_id, predicate, object_id, raw = facts[0]
    assert (subject_id, object_id) == (nodes[0][0], nodes[1][0])
    assert predicate == "links to"
    assert json.loads(raw) == ["a", "links to", "b"]


def test_candidate_matching_retains_original_order_indices_with_numpy_string_values():
    candidates = [(np.str_("A"), "p", "B"), ("C", "p", "D")]
    assert match_facts_to_candidate_indices(candidates, [["A", "p", "B"]], max_facts=None) == [0]
    assert match_facts_to_candidate_indices(candidates, [["C", "p", "D"], ["C", "p", "D"]]) == [1]
    assert match_facts_to_candidate_indices([], [["A", "p", "B"]]) == []


def test_empty_answer_metrics_and_legacy_bracket_cleanup():
    assert compute_qa_em_f1("", [""]) == (1.0, 0.0)
    assert compute_qa_em_f1("A", []) == (0.0, 0.0)
    assert compute_qa_em_f1("A B", ["B C"]) == (0.0, pytest.approx(2 / 3))
    assert strip_mistral_bracket_prefix("]\n\nAnswer") == "Answer"
    assert text_processing(json.loads("12")) == "12"
    assert filter_invalid_triples([{}, ["short"], StructuredTriple("A", pred="p", obj="B")]) == [["A", "p", "B"]]


@pytest.mark.parametrize("name", ['A"B', "A\\B"])
def test_truncation_repair_preserves_escaped_characters_in_completed_names(name):
    truncated = json.dumps({"named_entities": [name]})[:-2] + ', "unfinished'
    assert extract_entity_definitions(truncated) == ((name, ""),)


def test_wrong_nested_response_envelopes_and_missing_filter_output_are_not_facts():
    assert extract_structured_triples('{"other":[["A"]], "unfinished') == ()
    assert extract_structured_triples({"triples": [["A", "p", "B"]]})[0].as_tuple() == ("A", "p", "B")
    assert filter_reranked_facts([("A", "p", "B")], "No structured response") == []


def test_invalid_synonym_items_cannot_displace_a_later_valid_decision():
    raw = [
        None,
        {"pair_id": "", "relation": "related"},
        {"pair_id": "0", "relation": "unknown"},
        {"pair_id": "1", "relation": "related"},
    ]
    assert parse_synonym_decisions(raw) == [{"pair_id": "1", "relation": "skos:related", "canonical": "A"}]


@pytest.mark.parametrize("envelope", ["prediction", "dictionary", "json", "bare-json", "list"])
def test_synonym_and_shacl_decisions_reject_unknown_actions(envelope):
    decisions = [
        {"pair_id": "0", "relation": "related"},
        {"pair_id": "1", "relation": "differentFrom"},
        {"pair_id": "2", "relation": "unknown"},
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

    assert parse_synonym_decisions(wrap("decisions", decisions), num_pairs=1) == [
        {"pair_id": "0", "relation": "skos:related", "canonical": "A"},
    ]
    assert parse_shacl_repairs(wrap("repairs", repairs)) == [
        {"focus_node": "A", "action": "set_definition", "value": "A concept"},
    ]
    assert parse_synonym_decisions("not JSON") == []
    assert parse_shacl_repairs("not JSON") == []
