"""Adversarial sparse HMM boundaries and clean retired-API rejection."""

import numpy as np
import pytest

from yaw.daemon.graph.ppr import build_transition_matrix, rank_passages_by_hmm, run_hmm


def test_empty_graph_has_empty_scores_and_passages():
    graph = build_transition_matrix([], [], [])
    assert graph.transition_t.shape == (0, 0)
    assert run_hmm(graph, {}).size == 0
    assert rank_passages_by_hmm(graph, np.array([])) == []


def test_non_entity_edges_and_unknown_nodes_do_not_create_hidden_states():
    graph = build_transition_matrix(
        ["a", "a", "b"],
        ["a", "c", "c"],
        [
            ("a", "b", "passage", 0),
            ("b", "c", "passage", 0),
            ("a", "absent", "passage", 0.8),
            ("a", "a", "passage", 1),
        ],
    )
    assert graph.node_ids == ("a", "b", "c")
    assert graph.transition_t.nnz == 0
    np.testing.assert_array_equal(run_hmm(graph, {"absent": 1}), [0, 0, 0])


def test_direct_passage_seeds_are_retained_without_entity_states():
    graph = build_transition_matrix(["a", "b"], [], [])
    seed = np.array([6.0, 2.0])
    scores = run_hmm(graph, seed)
    np.testing.assert_array_equal(scores, [0.75, 0.25])
    np.testing.assert_array_equal(seed, [6.0, 2.0])
    assert rank_passages_by_hmm(graph, scores) == [("a", 0.75), ("b", 0.25)]
    with pytest.raises(ValueError, match="Reset vector size"):
        run_hmm(graph, np.array([1.0]))
    with pytest.raises(ValueError, match="score size"):
        rank_passages_by_hmm(graph, np.array([]))


@pytest.mark.parametrize(
    "parameter,value", [("activation", "linear"), ("activation_param", 1), ("dual_weight", 0.5), ("renormalize", True)]
)
def test_retired_ppr_parameters_are_rejected(parameter, value):
    graph = build_transition_matrix([], [], [])
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_hmm(graph, {}, **{parameter: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_operator_parameters_fail_before_inference(value):
    with pytest.raises(ValueError, match="beta"):
        build_transition_matrix([], [], [], beta=value)
    graph = build_transition_matrix([], [], [])
    with pytest.raises(ValueError, match="damping"):
        run_hmm(graph, {}, damping=value)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_seed_evidence_is_rejected(value):
    graph = build_transition_matrix(["a"], [], [])
    with pytest.raises(ValueError, match="Seed evidence"):
        run_hmm(graph, {"a": value})
    with pytest.raises(ValueError, match="Seed evidence"):
        run_hmm(graph, np.array([value]))


def test_confidence_and_bounds_fail_fast():
    with pytest.raises(ValueError, match="confidence"):
        build_transition_matrix([], ["e"], [], node_priors={"e": float("nan")})
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError, match="weights"):
            build_transition_matrix(["c"], ["e"], [("e", "c", "passage", weight)])
    graph = build_transition_matrix([], [], [])
    with pytest.raises(ValueError, match="max_iter"):
        run_hmm(graph, {}, max_iter=-1)
    with pytest.raises(ValueError, match="max_iter"):
        run_hmm(graph, {}, tol=float("nan"))


def test_retrieval_version_invalidates_query_and_reward_but_not_openie(monkeypatch):
    from yaw.daemon.graph.inference import optimizer
    from yaw.daemon.graph.inference.program import YawProgram

    program = YawProgram()
    before = (optimizer.candidate_fingerprint(program), optimizer.query_retrieval_fingerprint(program))
    openie = optimizer.openie_fingerprint(program)
    monkeypatch.setattr(optimizer, "HMM_RETRIEVAL_VERSION", "future-retrieval-contract")
    after = (optimizer.candidate_fingerprint(program), optimizer.query_retrieval_fingerprint(program))
    assert all(first != second for first, second in zip(before, after, strict=True))
    assert optimizer.openie_fingerprint(program) == openie
