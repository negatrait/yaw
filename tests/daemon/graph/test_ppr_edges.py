"""Sparse propagation boundary contracts using real matrices."""

import numpy as np
import pytest

from yaw.daemon.graph.ppr import (
    apply_graph_activation,
    build_transition_matrix,
    rank_passages_by_ppr,
    run_ppr,
)


def test_empty_graph_has_no_operator_or_ranked_passages():
    graph = build_transition_matrix([], [], [])
    assert graph.transition_t.shape == (0, 0)
    assert graph.dangling_mask.size == 0
    scores = run_ppr(graph, {})
    assert scores.size == 0
    assert rank_passages_by_ppr(graph, scores) == []


def test_invalid_edges_do_not_connect_or_duplicate_nodes():
    graph = build_transition_matrix(
        ["a", "a", "b"],
        ["a", "c", "c"],
        [("a", "b", 0.0), ("b", "c", -1.0), ("a", "absent", 2.0), ("a", "a", 1.0)],
    )
    assert graph.node_ids == ("a", "b", "c")
    assert graph.transition_t.nnz == 0
    assert graph.dangling_mask.all()
    # With no usable seed, dangling mass stays uniformly distributed.
    np.testing.assert_allclose(run_ppr(graph, {"absent": 5.0, "a": 0.0}, activation="linear"), [1 / 3] * 3)


def test_general_degree_normalization_matches_dense_definition():
    graph = build_transition_matrix(["a", "b"], ["c"], [("a", "b", 2.0), ("b", "c", 1.0)], beta=0.3)
    adjacency = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    degrees = np.array([2.0, 3.0, 1.0])
    expected = degrees[:, None] ** -0.3 * adjacency * degrees[None, :] ** -0.7
    np.testing.assert_allclose(graph.transition_t.toarray(), expected)


def test_array_reset_is_copied_normalized_and_shape_checked():
    graph = build_transition_matrix(["a", "b"], [], [("a", "b", 1.0)], beta=0.0)
    reset = np.array([6.0, 2.0])
    scores = run_ppr(graph, reset, damping=0.0, activation="linear")
    np.testing.assert_allclose(scores, [0.75, 0.25])
    np.testing.assert_array_equal(reset, [6.0, 2.0])
    with pytest.raises(ValueError, match="Reset vector size 1 does not match graph nodes 2"):
        run_ppr(graph, np.array([1.0]))
    assert rank_passages_by_ppr(graph, scores, top_k=0) == []
    assert rank_passages_by_ppr(graph, np.array([])) == []


def test_nonlinear_renormalization_preserves_mass_and_zero_fixed_point():
    graph = build_transition_matrix(["a", "b"], [], [("a", "b", 1.0)], beta=0.0)
    scores = run_ppr(graph, {"a": 1.0}, renormalize=True, dual_weight=None)
    assert np.isclose(scores.sum(), 1.0)
    zero = run_ppr(
        graph,
        {"a": 1.0},
        activation="threshold_relu",
        activation_param=2.0,
        renormalize=True,
        dual_weight=None,
    )
    np.testing.assert_array_equal(zero, [0.0, 0.0])
    with pytest.raises(ValueError, match="Unknown graph activation"):
        apply_graph_activation(np.array([0.1]), "unsupported")
