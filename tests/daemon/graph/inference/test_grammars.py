"""Tests for pure-JSON GBNF grammars of the 3-stage W3C semantic cascade."""

import pytest

from yaw.daemon.graph.inference.grammars import (
    build_dspy_hierarchy_judge_grammar,
    build_dspy_label_judge_grammar,
    build_dspy_related_judge_grammar,
)


def test_label_judge_grammar_structure() -> None:
    grammar = build_dspy_label_judge_grammar()
    assert 'root ::= "{\\"decisions\\": [" decision-list "]}"' in grammar
    assert 'decision ::= "{\\"pair_id\\": " string ", \\"reasoning\\": " reasoning-str' in grammar
    assert ', \\"equivalent\\": " equiv-branch' in grammar
    assert (
        'equiv-branch ::= "true, \\"definition\\": " reasoning-str | "false, \\"alt_label\\": " alt-branch' in grammar
    )
    assert 'alt-branch ::= "true" | "false, \\"hidden_label\\": " boolean' in grammar
    assert 'reasoning-str ::= "\\"" char{1,240} "\\""' in grammar
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


def test_hierarchy_judge_grammar_structure() -> None:
    grammar = build_dspy_hierarchy_judge_grammar()
    assert 'root ::= "{\\"decisions\\": [" decision-list "]}"' in grammar
    assert 'decision ::= "{\\"pair_id\\": " string ", \\"reasoning\\": " reasoning-str' in grammar
    assert ', \\"broader\\": " boolean' in grammar
    assert ', \\"narrower\\": " boolean' in grammar
    assert 'reasoning-str ::= "\\"" char{1,240} "\\""' in grammar
    assert 'boolean ::= "true" | "false"' in grammar


def test_hierarchy_judge_grammar_pair_ids() -> None:
    pairs = ["car|vehicle"]
    grammar = build_dspy_hierarchy_judge_grammar(pair_ids=pairs)
    assert 'decision-0 ::= "{\\"pair_id\\": \\"car|vehicle\\", \\"reasoning\\": "' in grammar
    assert ', \\"broader\\": " boolean' in grammar
    assert ', \\"narrower\\": " boolean' in grammar


def test_hierarchy_judge_grammar_invalid_pair_ids() -> None:
    with pytest.raises(ValueError, match="nonempty, unique"):
        build_dspy_hierarchy_judge_grammar(pair_ids=[])

    with pytest.raises(ValueError, match="Invalid synonym pair ID"):
        build_dspy_hierarchy_judge_grammar(pair_ids=["single"])


def test_related_judge_grammar_structure() -> None:
    grammar = build_dspy_related_judge_grammar()
    assert 'root ::= "{\\"decisions\\": [" decision-list "]}"' in grammar
    assert 'decision ::= "{\\"pair_id\\": " string ", \\"reasoning\\": " reasoning-str' in grammar
    assert ', \\"related\\": " boolean' in grammar
    assert 'reasoning-str ::= "\\"" char{1,240} "\\""' in grammar
    assert 'boolean ::= "true" | "false"' in grammar


def test_related_judge_grammar_pair_ids() -> None:
    pairs = ["doctor|hospital"]
    grammar = build_dspy_related_judge_grammar(pair_ids=pairs)
    assert 'decision-0 ::= "{\\"pair_id\\": \\"doctor|hospital\\", \\"reasoning\\": "' in grammar
    assert ', \\"related\\": " boolean' in grammar


def test_custom_reasoning_chars() -> None:
    g100 = build_dspy_label_judge_grammar(max_reasoning_chars=100)
    assert 'reasoning-str ::= "\\"" char{1,100} "\\""' in g100

    gunlimited = build_dspy_label_judge_grammar(max_reasoning_chars=None)
    assert 'reasoning-str ::= "\\"" char+ "\\""' in gunlimited


def test_adapter_resolves_3_stage_grammars() -> None:
    from yaw.daemon.graph.inference.adapter import YawGBNFAdapter
    from yaw.daemon.graph.inference.prompts import (
        JudgeHierarchy,
        JudgeLabels,
        JudgeRelated,
    )

    adapter = YawGBNFAdapter(enable_gbnf=True)
    g_label = adapter.resolve_grammar(JudgeLabels)
    assert g_label is not None
    assert "alt_label" in g_label

    g_hier = adapter.resolve_grammar(JudgeHierarchy)
    assert g_hier is not None
    assert "broader" in g_hier

    g_rel = adapter.resolve_grammar(JudgeRelated)
    assert g_rel is not None
    assert "related" in g_rel


def test_concept_scheme_defs_grammar_structure() -> None:
    from yaw.daemon.graph.inference.grammars import (
        DSPY_CONCEPT_SCHEME_DEFS_GBNF_GRAMMAR,
        build_dspy_concept_scheme_defs_grammar,
    )

    grammar = build_dspy_concept_scheme_defs_grammar()
    assert grammar == DSPY_CONCEPT_SCHEME_DEFS_GBNF_GRAMMAR
    assert 'root ::= "{\\"entity_definitions\\": [" scheme-list "]}"' in grammar
    assert '\\"scheme_id\\": " string' in grammar
    assert '\\"pref_label\\": " string' in grammar
    assert '\\"definition\\": " string' in grammar
    assert '\\"history_note\\": " string' in grammar


def test_concept_scheme_defs_grammar_pinned_ids() -> None:
    from yaw.daemon.graph.inference.grammars import build_dspy_concept_scheme_defs_grammar

    sids = ["scheme-048a06e0", "scheme-1f627afd"]
    grammar = build_dspy_concept_scheme_defs_grammar(sids)
    assert 'root ::= "{\\"entity_definitions\\": [" scheme-item-0 ", " scheme-item-1 "]}"' in grammar
    assert 'scheme-item-0 ::= "{\\"scheme_id\\": " "\\"scheme-048a06e0\\""' in grammar
    assert 'scheme-item-1 ::= "{\\"scheme_id\\": " "\\"scheme-1f627afd\\""' in grammar
    assert ', \\"pref_label\\": " string' in grammar
    assert ', \\"definition\\": " string' in grammar
    assert ', \\"history_note\\": " string "}"' in grammar
