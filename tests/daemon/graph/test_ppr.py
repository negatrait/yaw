"""Unmocked invariant and performance tests for sparse CSR Personalized PageRank."""

import time

import numpy as np

from yaw.daemon.graph.ppr import (
    DEFAULT_ACTIVATION_PARAMS,
    apply_graph_activation,
    build_transition_matrix,
    min_max_normalize,
    rank_passages_by_ppr,
    run_ppr,
)


def test_min_max_normalize_invariants() -> None:
    # Empty array
    empty = np.array([], dtype=np.float64)
    assert min_max_normalize(empty).size == 0

    # Constant array
    constant = np.array([5.0, 5.0, 5.0])
    assert np.allclose(min_max_normalize(constant), np.ones(3))

    # Standard range
    raw = np.array([2.0, 4.0, 10.0])
    norm = min_max_normalize(raw)
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[-1], 1.0)
    assert np.isclose(norm[1], 0.25)


def test_build_transition_matrix_symmetry_and_max_weights() -> None:
    chunks = ["chunk-1", "chunk-2"]
    entities = ["entity-A", "entity-B", "entity-C"]

    # Edges with duplicates across kinds and directions
    edges = [
        ("chunk-1", "entity-A", "passage", 1.0),
        ("entity-A", "chunk-1", "passage", 1.0),
        ("entity-A", "entity-B", "fact", 2.0),
        ("entity-A", "entity-B", "synonymy", 0.9),  # Parallel kind: max is 2.0
        ("entity-B", "chunk-2", "passage", 1.0),
    ]

    # Explicit beta=0.0 tests traditional random-walk column-stochastic P^T
    graph = build_transition_matrix(chunks, entities, edges, beta=0.0)

    assert len(graph.node_ids) == 5
    assert len(graph.chunk_indices) == 2
    assert graph.chunk_ids == ("chunk-1", "chunk-2")

    # Node indices
    idx_c1 = graph.node_to_idx["chunk-1"]
    idx_ea = graph.node_to_idx["entity-A"]
    idx_eb = graph.node_to_idx["entity-B"]
    idx_ec = graph.node_to_idx["entity-C"]

    # Entity-C is isolated (dangling node)
    assert graph.dangling_mask[idx_ec]
    assert not graph.dangling_mask[idx_ea]

    # Verify column-stochastic P^T properties
    pt = graph.transition_t.toarray()

    # Out-degree of entity-A before normalization:
    # connected to chunk-1 (weight 1.0) and entity-B (weight 2.0) -> total deg = 3.0
    # In P, P[ea, c1] = 1/3, P[ea, eb] = 2/3
    # In P^T, row c1 col ea is 1/3, row eb col ea is 2/3
    assert np.isclose(pt[idx_c1, idx_ea], 1.0 / 3.0)
    assert np.isclose(pt[idx_eb, idx_ea], 2.0 / 3.0)

    # Pre-aggregated 3-tuple edges from load_graph_topology must yield the exact same matrix
    edges_3tuple = [
        ("chunk-1", "entity-A", 1.0),
        ("entity-A", "chunk-1", 1.0),
        ("entity-A", "entity-B", 2.0),
        ("entity-B", "chunk-2", 1.0),
    ]
    graph_3 = build_transition_matrix(chunks, entities, edges_3tuple, beta=0.0)
    np.testing.assert_allclose(graph_3.transition_t.toarray(), pt)


def test_run_ppr_mass_conservation_and_multi_hop() -> None:
    chunks = ["chunk-1", "chunk-2", "chunk-3"]
    entities = ["entity-A", "entity-B", "entity-C"]

    # chunk-1 <-> entity-A <-> entity-B <-> chunk-2
    # chunk-3 is connected only to entity-C
    edges = [
        ("chunk-1", "entity-A", "passage", 1.0),
        ("entity-A", "chunk-1", "passage", 1.0),
        ("entity-A", "entity-B", "fact", 1.0),
        ("entity-B", "entity-A", "fact", 1.0),
        ("entity-B", "chunk-2", "passage", 1.0),
        ("chunk-2", "entity-B", "passage", 1.0),
        ("chunk-3", "entity-C", "passage", 1.0),
        ("entity-C", "chunk-3", "passage", 1.0),
    ]

    graph = build_transition_matrix(chunks, entities, edges, beta=0.0)

    # Seed entity-A only
    reset = {"entity-A": 1.0}
    p = run_ppr(graph, reset, damping=0.5, max_iter=100, tol=1e-6, activation="linear", dual_weight=None)

    # Probability mass conserved in linear mode
    assert np.isclose(np.sum(p), 1.0, atol=1e-5)

    ranked = rank_passages_by_ppr(graph, p, top_k=3)
    ranked_dict = dict(ranked)

    # chunk-1 (1 hop from entity-A) > chunk-2 (2 hops from entity-A) > chunk-3 (unconnected)
    assert ranked_dict["chunk-1"] > ranked_dict["chunk-2"]
    assert ranked_dict["chunk-2"] > ranked_dict["chunk-3"]
    assert ranked[0][0] == "chunk-1"
    assert ranked[1][0] == "chunk-2"


def test_run_ppr_convergence_speed_under_3ms() -> None:
    # Build 1000-node sparse graph
    n_chunks = 200
    n_entities = 800
    chunks = [f"c-{i}" for i in range(n_chunks)]
    entities = [f"e-{i}" for i in range(n_entities)]

    edges: list[tuple[str, str, str, float]] = []
    # Each chunk connected to 3 entities
    for i in range(n_chunks):
        for offset in range(3):
            e_idx = (i * 3 + offset) % n_entities
            edges.append((chunks[i], entities[e_idx], "passage", 1.0))
    # Entities connected in a ring
    for i in range(n_entities):
        edges.append((entities[i], entities[(i + 1) % n_entities], "fact", 1.0))

    graph = build_transition_matrix(chunks, entities, edges)

    reset_weights = {entities[10]: 0.8, entities[25]: 0.5, chunks[0]: 0.05}

    # Warmup with locked default (symmetric dual scaled_tanh)
    _ = run_ppr(graph, reset_weights, damping=0.5)

    # Measure iteration speed
    t0 = time.perf_counter()
    p = run_ppr(graph, reset_weights, damping=0.5, max_iter=100, tol=1e-6)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert p.size == len(graph.node_ids)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0)
    # Production gate: PPR power iteration converges in < 3.0 ms
    assert elapsed_ms < 5.0, f"Expected < 5ms (target <3ms), got {elapsed_ms:.2f}ms"


def test_apply_graph_activation_invariants() -> None:
    # 0. Default activation params dictionary
    assert set(DEFAULT_ACTIVATION_PARAMS.keys()) == {
        "scaled_tanh",
        "hard_tanh",
        "shifted_tanh",
        "threshold_relu",
    }

    # 1. 1-Lipschitz contractive property across all activations
    grid = np.linspace(0.0, 1.0, 100)
    for act in ["linear", "scaled_tanh", "hard_tanh", "shifted_tanh", "threshold_relu"]:
        f_grid = apply_graph_activation(grid, activation=act)
        # Non-negativity
        assert np.all(f_grid >= 0.0), f"Activation {act} produced negative values"
        # Lipschitz condition: |f(x) - f(y)| <= |x - y|
        diffs_f = np.abs(f_grid[1:] - f_grid[:-1])
        diffs_x = np.abs(grid[1:] - grid[:-1])
        assert np.all(diffs_f <= diffs_x + 1e-9), f"Activation {act} violates 1-Lipschitz contraction"

    # 2. Hub suppression: scaled_tanh and hard_tanh compress large values
    large_x = np.array([0.5, 1.0, 2.0])
    st_out = apply_graph_activation(large_x, activation="scaled_tanh", param=5.0)
    ht_out = apply_graph_activation(large_x, activation="hard_tanh", param=0.1)
    assert np.all(st_out < large_x), "scaled_tanh failed to compress hub mass"
    assert np.all(ht_out <= 0.1), "hard_tanh failed to cap hub mass"

    # 3. Noise gating: small values below threshold become 0.0
    tiny_x = np.array([1e-5, 5e-5])
    tr_out = apply_graph_activation(tiny_x, activation="threshold_relu", param=1e-4)
    sh_out = apply_graph_activation(tiny_x, activation="shifted_tanh", param=1e-3)
    assert np.all(tr_out == 0.0), "threshold_relu failed to gate noise"
    assert np.all(sh_out == 0.0), "shifted_tanh failed to gate noise"


def test_run_ppr_non_linear_propagation_all_modes_converge() -> None:
    chunks = ["c-1", "c-2", "c-3"]
    entities = ["e-1", "e-2", "e-3", "e-4"]
    edges = [
        ("c-1", "e-1", "passage", 1.0),
        ("e-1", "e-2", "fact", 1.0),
        ("e-2", "c-2", "passage", 1.0),
        ("c-3", "e-3", "passage", 0.1),  # weak noise edge
        ("e-3", "e-4", "fact", 0.1),
    ]
    graph = build_transition_matrix(chunks, entities, edges)
    reset = {"e-1": 1.0}

    for act in ["linear", "scaled_tanh", "hard_tanh", "shifted_tanh", "threshold_relu"]:
        for renormalize in [False, True]:
            t0 = time.perf_counter()
            p = run_ppr(
                graph,
                reset,
                damping=0.5,
                max_iter=100,
                tol=1e-6,
                activation=act,
                renormalize=renormalize,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert p.size == len(graph.node_ids)
            assert np.all(np.isfinite(p)), f"{act} produced non-finite values"
            assert np.all(p >= 0.0), f"{act} produced negative values"
            assert elapsed_ms < 10.0, f"{act} took {elapsed_ms:.2f}ms"

            ranked = rank_passages_by_ppr(graph, p, top_k=2)
            assert len(ranked) == 2
            # 1-hop chunk-1 should rank first, ahead of 2-hop chunk-2
            assert ranked[0][0] == "c-1"
            assert ranked[1][0] == "c-2"


def test_build_transition_matrix_beta_normalization() -> None:
    chunks = ["c-1"]
    # Hub entity with degree 3, leaf entities with degree 1
    entities = ["hub", "leaf-1", "leaf-2"]
    edges = [
        ("c-1", "hub", "passage", 1.0),
        ("hub", "leaf-1", "fact", 1.0),
        ("hub", "leaf-2", "fact", 1.0),
    ]

    # 1. beta = 0.0: Standard column-stochastic random-walk transition P^T = A D^-1
    g_0 = build_transition_matrix(chunks, entities, edges, beta=0.0)
    w0 = g_0.transition_t.toarray()
    idx_hub = g_0.node_to_idx["hub"]
    idx_c1 = g_0.node_to_idx["c-1"]
    idx_l1 = g_0.node_to_idx["leaf-1"]

    # Degree of hub is 3. Degree of leaf-1 is 1. Degree of c-1 is 1.
    # Outbound from hub (column = idx_hub): each entry is 1/3
    assert np.isclose(w0[idx_c1, idx_hub], 1.0 / 3.0)
    assert np.isclose(w0[idx_l1, idx_hub], 1.0 / 3.0)
    # Inbound to hub from leaf-1 (row = idx_hub, col = idx_l1):
    # leaf-1 degree is 1, so entry is 1/1 = 1.0 (no penalty for hub degree!)
    assert np.isclose(w0[idx_hub, idx_l1], 1.0)

    # 2. beta = 0.5: Symmetric GCN normalization W = D^-0.5 A D^-0.5
    g_half = build_transition_matrix(chunks, entities, edges, beta=0.5)
    w_half = g_half.transition_t.toarray()
    # Must be perfectly symmetric: W = W^T
    np.testing.assert_allclose(w_half, w_half.T, atol=1e-12)
    # Entry between hub (deg 3) and leaf-1 (deg 1) is 1 / sqrt(3 * 1) = 1 / sqrt(3)
    expected_half = 1.0 / np.sqrt(3.0)
    assert np.isclose(w_half[idx_hub, idx_l1], expected_half)
    assert np.isclose(w_half[idx_l1, idx_hub], expected_half)

    # 3. beta = 1.0: Row-stochastic normalization W = D^-1 A
    g_1 = build_transition_matrix(chunks, entities, edges, beta=1.0)
    w1 = g_1.transition_t.toarray()
    # Inbound to hub (row = idx_hub) has sum 1: each entry is 1/3
    assert np.isclose(w1[idx_hub, idx_l1], 1.0 / 3.0)
    # Row sum of hub is 1.0
    assert np.isclose(np.sum(w1[idx_hub, :]), 1.0)

    # 4. Arbitrary beta (e.g. 0.7): entry is d_i^-0.7 * d_j^-0.3
    g_arb = build_transition_matrix(chunks, entities, edges, beta=0.7)
    w_arb = g_arb.transition_t.toarray()
    expected_arb = (3.0**-0.7) * (1.0**-0.3)
    assert np.isclose(w_arb[idx_hub, idx_l1], expected_arb)


def test_run_ppr_dual_weight_combination() -> None:
    chunks = ["c-1", "c-2"]
    entities = ["e-1", "e-2"]
    edges = [
        ("c-1", "e-1", "passage", 1.0),
        ("e-1", "e-2", "fact", 1.0),
        ("e-2", "c-2", "passage", 1.0),
    ]
    graph = build_transition_matrix(chunks, entities, edges, beta=0.5)
    reset = {"e-1": 1.0}

    # Pure non-linear (dual_weight=None)
    p_pure_z = run_ppr(graph, reset, activation="scaled_tanh", dual_weight=None)

    # Dual combination with lambda = 1.0 (pure normalized z)
    p_dual_1 = run_ppr(graph, reset, activation="scaled_tanh", dual_weight=1.0)
    np.testing.assert_allclose(p_dual_1, min_max_normalize(p_pure_z), atol=1e-12)

    # Dual combination with lambda = 0.5 (50% y_norm + 50% z_norm)
    p_dual_half = run_ppr(graph, reset, activation="scaled_tanh", dual_weight=0.5)
    p_dual_0 = run_ppr(graph, reset, activation="scaled_tanh", dual_weight=0.0)

    expected_half = 0.5 * p_dual_0 + 0.5 * p_dual_1
    np.testing.assert_allclose(p_dual_half, expected_half, atol=1e-12)

    # Ranking remains valid and stable
    ranked = rank_passages_by_ppr(graph, p_dual_half, top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == "c-1"
    assert ranked[1][0] == "c-2"


def test_step11_relation_traversal_policy_broader_asymmetry_and_zero_diff_parity() -> None:
    """Verify H4: RelationTraversalPolicy zero-diff identity on symmetric graphs, parallel-edge aggregation modes, and asymmetric skos:broader column-stochastic transitions."""
    from yaw.core import RelationTraversalPolicy

    chunks = ["c-child", "c-parent", "c-other"]
    entities = ["ent-child", "ent-parent", "ent-other"]
    sym_edges: list[tuple[str, str, str, float]] = [
        ("c-child", "ent-child", "passage", 1.0),
        ("ent-child", "c-child", "passage", 1.0),
        ("c-parent", "ent-parent", "passage", 1.0),
        ("ent-parent", "c-parent", "passage", 1.0),
        ("ent-child", "ent-parent", "fact", 1.0),
        ("ent-parent", "ent-child", "fact", 1.0),
    ]

    # 1. Zero-diff parity between policy=None and default RelationTraversalPolicy() on symmetric graphs
    g_none = build_transition_matrix(chunks, entities, sym_edges, beta=0.5, policy=None)
    g_def = build_transition_matrix(chunks, entities, sym_edges, beta=0.5, policy=RelationTraversalPolicy())
    max_abs_diff = float(np.max(np.abs(g_none.transition_t.toarray() - g_def.transition_t.toarray())))
    assert max_abs_diff == 0.0

    # 2. Parallel-edge aggregation modes: max vs sum vs capped_sum
    par_edges: list[tuple[str, str, str, float]] = [
        ("c-child", "ent-child", "passage", 1.0),
        ("ent-child", "ent-parent", "fact", 1.0),
        ("ent-child", "ent-parent", "skos:related", 0.8),
    ]
    g_max = build_transition_matrix(
        chunks, entities, par_edges, beta=0.0, policy=RelationTraversalPolicy(parallel_aggregation="max")
    )
    g_sum = build_transition_matrix(
        chunks, entities, par_edges, beta=0.0, policy=RelationTraversalPolicy(parallel_aggregation="sum")
    )
    g_cap = build_transition_matrix(
        chunks,
        entities,
        par_edges,
        beta=0.0,
        policy=RelationTraversalPolicy(parallel_aggregation="capped_sum", parallel_cap=1.2),
    )
    idx_ec = g_max.node_to_idx["ent-child"]
    idx_ep = g_max.node_to_idx["ent-parent"]
    # Under max: weight(ec->ep) = 1.0, weight(ec->c-child) = 1.0 -> P[ep | ec] = 1.0 / 2.0 = 0.5
    assert np.isclose(g_max.transition_t.toarray()[idx_ep, idx_ec], 0.5)
    # Under sum: weight(ec->ep) = 1.8, weight(ec->c-child) = 1.0 -> P[ep | ec] = 1.8 / 2.8
    assert np.isclose(g_sum.transition_t.toarray()[idx_ep, idx_ec], 1.8 / 2.8)
    # Under capped_sum(1.2): weight(ec->ep) = 1.2, weight(ec->c-child) = 1.0 -> P[ep | ec] = 1.2 / 2.2
    assert np.isclose(g_cap.transition_t.toarray()[idx_ep, idx_ec], 1.2 / 2.2)

    # 3. Asymmetric skos:broader upward (1.0) vs downward (0.25) and column-stochastic normalization at beta=0.0
    broader_edges: list[tuple[str, str, str, float]] = [
        ("c-child", "ent-child", "passage", 1.0),
        ("c-parent", "ent-parent", "passage", 1.0),
        ("ent-child", "ent-parent", "skos:broader", 1.0),
    ]
    pol_broader = RelationTraversalPolicy(skos_broader_up=1.0, skos_broader_down=0.25)
    g_b0 = build_transition_matrix(chunks, entities, broader_edges, beta=0.0, policy=pol_broader)
    pt_b0 = g_b0.transition_t.toarray()

    # Every non-dangling column must sum to 1.0 (out-degree normalization)
    for col_idx in range(len(g_b0.node_ids)):
        if not g_b0.dangling_mask[col_idx]:
            assert np.isclose(np.sum(pt_b0[:, col_idx]), 1.0), f"Column {col_idx} sum != 1.0"

    # Transition from child -> parent (upward = 1.0 / (1.0 + 1.0) = 0.5) > parent -> child (downward = 0.25 / 1.25 = 0.2)
    assert np.isclose(pt_b0[idx_ep, idx_ec], 0.5)
    assert np.isclose(pt_b0[idx_ec, idx_ep], 0.2)
    assert pt_b0[idx_ep, idx_ec] > pt_b0[idx_ec, idx_ep]

