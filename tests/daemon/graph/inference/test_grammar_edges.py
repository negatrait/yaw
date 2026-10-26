"""Grammar fallback and task quality contracts with native DSPy objects."""

import json

import dspy

from yaw.daemon.graph.inference.grammars import (
    DSPY_RERANK_GBNF_GRAMMAR,
    build_dspy_rerank_grammar,
    build_dspy_triples_grammar,
    lm_supports_gbnf,
)
from yaw.daemon.graph.inference.parsers import compute_task_quality_score


def test_dynamic_grammar_rejects_malformed_candidates_and_ignores_invalid_known_names():
    assert build_dspy_rerank_grammar([]) == 'root ::= "{\\"fact\\": []}"\n'
    for payload in ("[null]", '[["a", "b"]]', '[["a", 2, "b"]]'):
        candidates = json.loads(payload)
        assert build_dspy_rerank_grammar(candidates) == DSPY_RERANK_GBNF_GRAMMAR
    assert build_dspy_triples_grammar(2, [None, [], 3, "Known"]) == build_dspy_triples_grammar(2, ["Known"])


def test_multirank_retrieval_score_does_not_invent_qa_objectives():
    base = compute_task_quality_score(0.8, tokens=100, recall_at_1=0.5)
    assert base.em is None and base.f1 is None
    assert "em" not in base.objective_scores and "f1" not in base.objective_scores
    assert base.objective_scores["recall_at_1"] == 0.5
    costly = compute_task_quality_score(
        0.8,
        tokens=100,
        recall_at_1=0.5,
        shacl_violations_initial=50,
        shacl_violations_remaining=20,
    )
    assert costly.gm_good == base.gm_good
    assert costly.score == base.score
    assert costly.objective_scores == base.objective_scores


def test_absent_default_lm_keeps_grammar_supported():
    with dspy.context(lm=None):
        assert lm_supports_gbnf() is True


def test_native_lm_recipe_metadata_disables_unsupported_grammar():
    class RecipeLM(dspy.LM):
        _yaw_recipe: str

    lm = RecipeLM("openai/test-model", cache=False)
    lm._yaw_recipe = "flm"
    assert lm_supports_gbnf(lm) is False
