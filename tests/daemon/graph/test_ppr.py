"""Independent sparse state/emission HMM contracts (H1); no PPR conservation assumptions."""

import time

import numpy as np
import pytest

from yaw.core import RelationTraversalPolicy
from yaw.daemon.graph.ppr import build_transition_matrix, min_max_normalize, rank_passages_by_hmm, run_hmm


def directed_chain(beta=0.0, priors=None):
    return build_transition_matrix(
        ["near", "far", "island"],
        ["a", "b", "unrelated"],
        [
            ("a", "near", "passage", 1.0),
            ("b", "far", "passage", 0.8),
            ("unrelated", "island", "passage", 1.0),
            ("a", "b", "fact", 1.0),
        ],
        directed=True,
        beta=beta,
        node_priors=priors,
    )


def test_dense_feature_normalization():
    np.testing.assert_array_equal(min_max_normalize(np.array([])), [])
    np.testing.assert_array_equal(min_max_normalize(np.array([5.0, 5.0])), [1, 1])
    np.testing.assert_allclose(min_max_normalize(np.array([2, 4, 10])), [0, 0.25, 1])


def test_entity_transitions_and_passage_emissions_are_separate():
    graph = directed_chain()
    np.testing.assert_array_equal(graph.hmm.emissions.toarray(), np.diag([1.0, 0.8, 1.0]))
    np.testing.assert_array_equal(graph.hmm.forward.toarray(), [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    assert graph.hmm.forward.shape == (3, 3)
    assert graph.hmm.emissions.shape == (3, 3)
    # Passage degree has no part in entity normalization.
    more = build_transition_matrix(
        ["one", "two"],
        ["a", "b"],
        [("a", "one", "passage", 0.5), ("a", "two", "passage", 0.9), ("a", "b", "fact", 1)],
        directed=True,
        beta=0,
    )
    assert more.hmm.forward[1, 0] == 1


def test_independent_typed_relations_pool_before_entity_normalization():
    graph = build_transition_matrix(
        [],
        ["a", "b", "c"],
        [("a", "b", "fact", 0.8), ("a", "b", "skos:related", 0.6), ("a", "c", "fact", 0.2)],
        directed=True,
        beta=0,
    )
    # Independent 0.8 and 0.6 yield 0.92; the second outgoing state has potential 0.2.
    assert graph.hmm.forward[1, 0] == pytest.approx(0.92 / 1.12)
    assert graph.hmm.forward[2, 0] == pytest.approx(0.2 / 1.12)


def test_reciprocal_passage_rows_are_one_observation():
    graph = build_transition_matrix(["c"], ["e"], [("c", "e", "passage", 0.6), ("e", "c", "passage", 0.6)])
    assert graph.hmm.emissions[0, 0] == 0.6


def test_filter_matches_analytic_directed_chain_and_emits_independent_layer():
    graph = directed_chain()
    scores = run_hmm(graph, {"a": 1}, damping=0.5, tol=1e-12)
    # Stationary restart/dangling equations: b=d*a and a+b=1, hence a=2/3, b=1/3.
    np.testing.assert_allclose(scores[3:], [2 / 3, 1 / 3, 0], atol=1e-12)
    np.testing.assert_allclose(scores[:3], [2 / 3, 0.8 / 3, 0], atol=1e-12)
    assert scores.sum() > 1  # Separate passage and hidden-state layers.
    assert rank_passages_by_hmm(graph, scores, 3) == [("near", scores[0]), ("far", scores[1])]


def test_beta_normalizes_only_the_entity_potential_against_dense_definition():
    adjacency = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    degree = np.array([2.0, 1.0, 1.0])
    for beta in (0, 0.3, 0.5, 1):
        graph = build_transition_matrix(
            ["passage"],
            ["hub", "leaf1", "leaf2"],
            [("passage", "hub", "passage", 0.7), ("hub", "leaf1", "fact", 1), ("hub", "leaf2", "fact", 1)],
            beta=beta,
        )
        expected = degree[:, None] ** -beta * adjacency * degree[None, :] ** (beta - 1)
        np.testing.assert_allclose(graph.hmm.forward.toarray(), expected)
        assert graph.hmm.emissions[0, 0] == 0.7


def test_damping_and_iteration_bound_are_active_not_floored_or_capped():
    graph = directed_chain()
    for damping in (0.0, 0.1, 0.5, 0.7):
        scores = run_hmm(graph, {"a": 1}, damping=damping, tol=1e-12)
        assert scores[4] == pytest.approx(damping / (1 + damping), abs=1e-12)
    np.testing.assert_array_equal(run_hmm(graph, {"a": 1}, max_iter=0)[3:], [1, 0, 0])
    np.testing.assert_allclose(run_hmm(graph, {"a": 1}, max_iter=1)[3:], [0.5, 0.5, 0])


def test_priors_remain_observation_likelihoods_when_degrees_cancel_edge_scale():
    graph = directed_chain(priors={"a": None, "b": 0.1})
    neutral = directed_chain(priors={"a": None, "b": None})
    # Structural transitions are identical; source confidence is retained independently.
    np.testing.assert_array_equal(graph.hmm.forward.toarray(), neutral.hmm.forward.toarray())
    measured = run_hmm(graph, {"a": 1})
    unmeasured = run_hmm(neutral, {"a": 1})
    assert 0 < measured[4] < unmeasured[4]
    blocked = run_hmm(directed_chain(priors={"b": 0.0}), {"a": 1})
    assert blocked[4] == 0 and blocked[1] == 0


def test_seedless_and_disconnected_islands_receive_no_evidence():
    graph = directed_chain()
    np.testing.assert_array_equal(run_hmm(graph, {}), np.zeros(6))
    assert rank_passages_by_hmm(graph, run_hmm(graph, {"absent": 1})) == []
    scores = run_hmm(graph, {"a": 1})
    assert scores[2] == 0 and scores[5] == 0
    small = build_transition_matrix(
        ["near", "far"],
        ["a", "b"],
        [("a", "near", "passage", 1), ("b", "far", "passage", 0.8), ("a", "b", "fact", 1)],
        directed=True,
        beta=0,
    )
    np.testing.assert_array_equal(run_hmm(small, {"a": 1}), scores[[0, 1, 3, 4]])


def test_distinctness_neither_transitions_nor_inhibits_relevance():
    graph = directed_chain()
    distinct = build_transition_matrix(
        graph.chunk_ids,
        ["a", "b", "unrelated"],
        [
            ("a", "near", "passage", 1),
            ("b", "far", "passage", 0.8),
            ("unrelated", "island", "passage", 1),
            ("a", "b", "fact", 1),
            ("a", "unrelated", "owl:differentFrom", 0.99),
        ],
        directed=True,
        beta=0,
    )
    np.testing.assert_array_equal(run_hmm(graph, {"a": 1}), run_hmm(distinct, {"a": 1}))


def test_relation_policy_direction_and_noisy_or_passage_emission():
    graph = build_transition_matrix(
        ["both"],
        ["child", "parent"],
        [("both", "child", "passage", 0.8), ("both", "parent", "passage", 0.6), ("child", "parent", "skos:broader", 1)],
        policy=RelationTraversalPolicy(skos_broader_up=1, skos_broader_down=0),
    )
    assert graph.hmm.forward[1, 0] > 0 and graph.hmm.forward[0, 1] == 0
    scores = run_hmm(graph, {"child": 1}, damping=0.5)
    assert scores[0] == pytest.approx(1 - (1 - 0.8 * scores[1]) * (1 - 0.6 * scores[2]))


def test_sparse_inference_under_existing_five_ms_gate():
    chunks = [f"c{i}" for i in range(200)]
    entities = [f"e{i}" for i in range(800)]
    edges = [(chunks[i], entities[(i * 3 + j) % 800], "passage", 1.0) for i in range(200) for j in range(3)]
    edges += [(entities[i], entities[(i + 1) % 800], "fact", 1.0) for i in range(800)]
    graph = build_transition_matrix(chunks, entities, edges)
    seed = {entities[10]: 0.8, entities[25]: 0.5, chunks[0]: 0.05}
    run_hmm(graph, seed)
    started = time.perf_counter()
    scores = run_hmm(graph, seed)
    elapsed = (time.perf_counter() - started) * 1000
    assert np.all(np.isfinite(scores)) and np.all((scores >= 0) & (scores <= 1))
    assert elapsed < 5.0, f"Existing <5ms gate exceeded: {elapsed:.2f}ms"


def test_symmetric_copies_pool_once_before_directional_policy():
    policy = RelationTraversalPolicy(fact_forward=0.5, fact_reverse=0, skos_related=0.5)
    edges = [
        ("a", "b", "skos:related", 0.6),
        ("b", "a", "skos:related", 0.7),
        ("a", "b", "fact", 0.8),
        ("a", "b", "fact", 0.5),
        ("a", "c", "fact", 0.6),
    ]
    graph = build_transition_matrix([], ["a", "b", "c"], edges, beta=0, policy=policy)
    # Noisy-OR fact=.9, related=.7. Scale each by .5, then pool:
    # 1 - (1-.45)(1-.35) = .6425; the competing a->c potential is .3.
    assert graph.hmm.forward[1, 0] == pytest.approx(0.6425 / 0.9425)
    assert graph.hmm.forward[2, 0] == pytest.approx(0.3 / 0.9425)
    directed = build_transition_matrix([], ["a", "b"], edges[:2], beta=0, directed=True)
    # Direct mode retains both explicitly supplied orientations without expansion.
    np.testing.assert_array_equal(directed.hmm.forward.toarray(), [[0, 1], [1, 0]])


def test_passage_seed_projection_activates_bidirectional_filter():
    graph = build_transition_matrix(
        ["c1", "c2"],
        ["a", "b"],
        [("a", "c1", "passage", 1), ("b", "c2", "passage", 1), ("a", "b", "fact", 1)],
        beta=0,
        directed=True,
    )
    # q=(.8,.2); one forward step=(.48,.52), backward=(.82,.18).
    # Normalize their elementwise product: (.3936,.0936)/.4872.
    result = run_hmm(graph, {"c1": 0.8, "c2": 0.2}, damping=0.5, max_iter=1)
    np.testing.assert_allclose(result[2:], np.array([0.3936, 0.0936]) / 0.4872)
    np.testing.assert_allclose(result[:2], [1 - (1 - result[2]) * 0.2, 1 - (1 - result[3]) * 0.8])


def test_real_sqlite_topology_keeps_strongest_symmetric_copy(tmp_path):
    import sqlite3

    from yaw.daemon.graph.storage.graph_ops import load_graph_topology

    with sqlite3.connect(tmp_path / "topology.db") as conn:
        conn.executescript("""
            CREATE TABLE chunks(chunk_id TEXT);
            CREATE TABLE entities(entity_id TEXT, confidence REAL);
            CREATE TABLE edges(src_id TEXT, dst_id TEXT, kind TEXT, weight REAL);
            CREATE TABLE facts(fact_id TEXT, subj_id TEXT, obj_id TEXT, obj_kind TEXT, confidence REAL);
            CREATE TABLE fact_sources(fact_id TEXT, atomic_confidence REAL);
            INSERT INTO entities VALUES ('a', NULL), ('b', NULL), ('c', NULL);
            INSERT INTO edges VALUES ('a', 'b', 'skos:related', .6),
                                     ('b', 'a', 'skos:related', .7), ('a', 'c', 'fact', .3);
        """)
        chunks, entities, edges, priors = load_graph_topology(conn)
    assert ("a", "b", "skos:related", 0.7) in edges
    assert len([e for e in edges if e[2] == "skos:related"]) == 1
    graph = build_transition_matrix(chunks, entities, edges, beta=0, node_priors=priors)
    assert graph.hmm.forward[1, 0] == pytest.approx(0.7)
    assert graph.hmm.forward[2, 0] == pytest.approx(0.3)


def test_real_sqlite_topology_rejects_negative_edge(tmp_path):
    import sqlite3

    from yaw.daemon.graph.storage.graph_ops import load_graph_topology

    with sqlite3.connect(tmp_path / "invalid-topology.db") as conn:
        conn.executescript("""
            CREATE TABLE chunks(chunk_id TEXT);
            CREATE TABLE entities(entity_id TEXT, confidence REAL);
            CREATE TABLE edges(src_id TEXT, dst_id TEXT, kind TEXT, weight REAL);
            INSERT INTO entities VALUES ('a', NULL), ('b', NULL);
            INSERT INTO edges VALUES ('a', 'b', 'skos:related', -.1),
                                     ('b', 'a', 'skos:related', .7);
        """)
        with pytest.raises(ValueError, match="nonnegative"):
            load_graph_topology(conn)
