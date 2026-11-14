"""Tests for pure-JSON GBNF grammars of the 3-stage W3C semantic cascade."""

import pytest

from yaw.daemon.graph.inference.grammars import (
    DSPY_NER_GBNF_GRAMMAR,
    DSPY_SHACL_REPAIR_GBNF_GRAMMAR,
    DSPY_TRIPLES_GBNF_GRAMMAR,
    build_dspy_label_judge_grammar,
    build_dspy_relation_judge_grammar,
    build_dspy_shacl_repair_grammar,
    build_dspy_triples_grammar,
)


def test_label_judge_grammar_structure() -> None:
    grammar = build_dspy_label_judge_grammar()
    assert 'root ::= "{\\"decisions\\": [" decision-list "]}"' in grammar
    assert 'decision ::= "{\\"pair_id\\": " string ", \\"reasoning\\": " reasoning-str' in grammar
    assert ', \\"equivalent\\": " equiv-branch' in grammar
    assert (
        'equiv-branch ::= "true, \\"definition\\": " reasoning-str ", \\"hidden_label\\": " boolean | "false"'
        in grammar
    )
    assert 'reasoning-str ::= "\\"" char+ "\\""' in grammar
    assert 'boolean ::= "true" | "false"' in grammar


def test_label_judge_grammar_bounded_decisions() -> None:
    grammar = build_dspy_label_judge_grammar(max_decisions=2)
    assert 'decision (", " decision)?' in grammar


def test_label_judge_grammar_pair_ids() -> None:
    pairs = ["apple|fruit", "tree|plant"]
    grammar = build_dspy_label_judge_grammar(pair_ids=pairs)
    assert 'decision-0 ::= "{\\"pair_id\\": \\"apple|fruit\\", \\"reasoning\\": "' in grammar
    assert 'decision-1 ::= "{\\"pair_id\\": \\"tree|plant\\", \\"reasoning\\": "' in grammar
    assert 'root ::= "{\\"decisions\\": [" decision-0 ", " decision-1 "]}"' in grammar


def test_label_judge_grammar_invalid_pair_ids() -> None:
    with pytest.raises(ValueError, match="nonempty, unique"):
        build_dspy_label_judge_grammar(pair_ids=[])

    with pytest.raises(ValueError, match="nonempty, unique"):
        build_dspy_label_judge_grammar(pair_ids=["a|b", "a|b"])

    with pytest.raises(ValueError, match="Invalid synonym pair ID"):
        build_dspy_label_judge_grammar(pair_ids=["invalid"])

    with pytest.raises(ValueError, match="Invalid synonym pair ID"):
        build_dspy_label_judge_grammar(pair_ids=["same|same"])


def test_relation_judge_grammar_choices_and_pair_constraints() -> None:
    grammar = build_dspy_relation_judge_grammar(pair_ids=["a|b", "c|d"], associated_pair_ids=["a|b"])
    lines = [line for line in grammar.splitlines() if line.startswith("decision-")]
    assert all(value in lines[0] for value in ("skos:broader", "skos:narrower", "skos:related"))
    assert "owl:differentFrom" not in lines[0]
    assert "owl:differentFrom" in lines[1]
    with pytest.raises(ValueError):
        build_dspy_relation_judge_grammar(pair_ids=["invalid"])


def test_custom_reasoning_chars() -> None:
    g100 = build_dspy_label_judge_grammar(max_reasoning_chars=100)
    assert 'reasoning-str ::= "\\"" char{1,100} "\\""' in g100

    gunlimited = build_dspy_label_judge_grammar(max_reasoning_chars=None)
    assert 'reasoning-str ::= "\\"" char+ "\\""' in gunlimited


def test_adapter_resolves_3_stage_grammars() -> None:
    from yaw.daemon.graph.inference.adapter import YawGBNFAdapter
    from yaw.daemon.graph.inference.prompts import (
        CompressInstructionSignature,
        JudgeLabels,
        JudgeRelations,
        ProposeInstructionSignature,
    )

    adapter = YawGBNFAdapter(enable_gbnf=True)
    g_label = adapter.resolve_grammar(JudgeLabels)
    assert g_label is not None
    assert "hidden_label" in g_label

    g_rel = adapter.resolve_grammar(JudgeRelations)
    assert g_rel is not None
    assert "skos:broader" in g_rel and "owl:differentFrom" in g_rel

    g_prop = adapter.resolve_grammar(ProposeInstructionSignature)
    assert g_prop is not None
    assert "new_instruction" in g_prop

    g_comp = adapter.resolve_grammar(CompressInstructionSignature)
    assert g_comp is not None
    assert "shortened_instruction" in g_comp
    assert "new_instruction" not in g_comp


def _parse_gbnf_rules(grammar: str) -> dict[str, str]:
    rules: dict[str, str] = {}
    for line in grammar.splitlines():
        name, body = line.split(" ::= ", 1)
        rules[name] = body
    return rules


def test_shacl_repair_grammar_lexical_label_branch_and_abstention() -> None:
    base_rules = _parse_gbnf_rules(DSPY_SHACL_REPAIR_GBNF_GRAMMAR)
    assert base_rules["label"] == '"\\"" non-letter* letter char* "\\""'
    assert base_rules["letter"] == "[a-zA-Z\\x80-\\U0010FFFF]"
    assert base_rules["non-letter"] == "[\\x20\\x21\\x23-\\x40\\x5B\\x5D-\\x60\\x7B-\\x7E]"

    grammar = build_dspy_shacl_repair_grammar(
        allowed_actions={"entity-1": {"set_pref_label"}, "fact-1": {"set_definition"}}
    )
    rules = _parse_gbnf_rules(grammar)
    assert rules["repair-list"] == '"[]" | "[" repair-0 (", " repair-0)* ", " repair-1 (", " repair-1)* "]"'
    assert rules["repair-0"] == (
        '"{\\"focus_node\\": " "\\"entity-1\\"" ", \\"action\\": " ("\\"set_pref_label\\"") ", \\"value\\": " label "}"'
    )
    assert rules["repair-1"] == (
        '"{\\"focus_node\\": " "\\"fact-1\\"" ", \\"action\\": " ("\\"set_definition\\"") ", \\"value\\": " string "}"'
    )
    assert rules["label"] == '"\\"" non-letter* letter char* "\\""'
    assert rules["letter"] == "[a-zA-Z\\x80-\\U0010FFFF]"
    assert rules["non-letter"] == "[\\x20\\x21\\x23-\\x40\\x5B\\x5D-\\x60\\x7B-\\x7E]"


def test_ner_and_triples_grammars_leave_output_unbounded_by_default() -> None:
    ner_rules = _parse_gbnf_rules(DSPY_NER_GBNF_GRAMMAR)
    assert ner_rules["entity-list"] == '"[]" | "[" entity-pair (", " entity-pair)* "]"'
    assert ner_rules["string"] == '"\\"" char+ "\\""'
    assert ner_rules["label"] == '"\\"" non-letter* letter char* "\\""'

    open_triples = _parse_gbnf_rules(DSPY_TRIPLES_GBNF_GRAMMAR)
    assert open_triples["triple-list"] == '"[]" | "[" triple (", " triple)* "]"'
    assert open_triples["triple"] == '"{\\"sub\\": " resource-slot ", \\"pred\\": " pred-obj ", \\"def\\": " string "}"'
    assert open_triples["pred-obj"] == 'resource-pred ", \\"obj\\": " resource-slot | literal-statement'
    assert open_triples["resource-slot"] == 'string | "[" string ", " string "]"'
    assert open_triples["string"] == '"\\"" char+ "\\""'

    pinned_triples = _parse_gbnf_rules(build_dspy_triples_grammar(None, [["HubSpot", "CRM platform."]]))
    assert pinned_triples["triple-list"] == '"[]" | "[" triple (", " triple)* "]"'
    assert pinned_triples["string"] == '"\\"" char+ "\\""'


def test_triples_grammar_normalizes_empty_definitions_to_prevent_reflexive_statements() -> None:
    # Service Mesh has definition, service mesh does not; both normalize to the same sense
    grammar = build_dspy_triples_grammar(
        known_entities=[
            ["Service Mesh", "Infrastructure layer for microservices."],
            "service mesh",
        ]
    )
    # Neither should have an objects rule linking to the other
    assert "objects-0" not in grammar
    assert "objects-1" not in grammar
    assert 'subject-0 ::= endpoint-0 ", \\"pred\\": " literal-statement' in grammar
    assert 'subject-1 ::= endpoint-1 ", \\"pred\\": " literal-statement' in grammar


def test_triples_grammar_ignores_empty_definition_pairs_for_ambiguity() -> None:
    grammar = build_dspy_triples_grammar(
        known_entities=[
            ["Service Mesh", "Infrastructure layer for microservices."],
            ["Service Mesh", ""],
        ]
    )
    assert 'endpoint-0 ::= "\\"Service Mesh\\""' in grammar
    assert "endpoint-1" not in grammar
