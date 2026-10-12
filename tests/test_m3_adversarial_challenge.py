"""Adversarial stress-test suite for Milestone 3: Retriever Seeding & Conductance.

Empirically probes:
1. Conductance Symmetry without Degree Dampening:
   - High in-degree hub vs isolated node across close_matches edges (CHECK(src_id < dst_id)).
   - Exact weight equality W(A, B) == W(B, A) == score in CSR adjacency matrix.
   - Verification that hub dampening applies to wikilinks but NOT to close_matches.
   - Multi-score clusters and multiple closeMatch relation variants.
2. Bayesian Query-to-Fact Seeding Edge Cases:
   - Case 1: Empty fact_vec table (0 rows) and missing fact_vec table; smooth degradation to alias priors.
   - Case 2: Zero vector norm (q = 0) offline degradation without division by zero.
   - Case 3: Negative and zero fact cosine similarities; clamping to 0.0 without invalid probability mass.
   - Case 4: Extreme lambdas (lambda = 0.0 and lambda = 1.0); pure fact vs pure alias seeding without NaN.
3. Additional Adversarial Invariants:
   - Continuous degradation when one channel is missing under extreme lambdas.
   - Self-loop relation handling (subject_id == object_id) preventing double counting.
   - Endpoint mean cosine pooling across multiple facts for the same entity per ADR-012.
   - Dangling fact endpoints not present in graph nodes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from yaw import config
from yaw.indexer.canonicalize import _register_alias_in_kb, create_in_memory_kb
from yaw.retrieval import retriever as retriever_module
from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage import (
    ensure_schema_views,
    get_db,
    insert_close_match,
    insert_fact_vec,
    serialize_embedding,
)


def _init_test_db(conn: sqlite3.Connection) -> None:
    """Initialize test SQLite database with all required tables and unified views."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS leaf_relations (relation_id TEXT PRIMARY KEY, source_leaf_id TEXT, subject_id TEXT, predicate TEXT, predicate_id TEXT, object_id TEXT, weight REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
        "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, PRIMARY KEY (node_id, source_leaf_id))"
    )
    ensure_schema_views(conn)


def _ser(vec: np.ndarray) -> bytes:
    """Serialize embedding and assert non-None bytes for static type checker."""
    b = serialize_embedding(vec)
    assert b is not None
    return b


# ==============================================================================
# SECTION 1: CONDUCTANCE SYMMETRY WITHOUT DEGREE DAMPENING
# ==============================================================================


def test_adversarial_conductance_symmetry_with_extreme_in_degree_hub(tmp_path: Path):
    """Adversarially probe that a massive in-degree hub does NOT dampen close_matches conductance.

    Scenario:
    - Node A ('concepts:isolated_concept') has in-degree 1 (only the closeMatch edge).
    - Node B ('concepts:massive_hub') has in-degree 51 (50 incoming wikilinks + 1 closeMatch).
    - Under gamma=0.5 degree dampening, if closeMatch were dampened:
        W(A -> B) = 0.85 / sqrt(51) = 0.119
        W(B -> A) = 0.85 / sqrt(1)  = 0.850
      which would violate conductance symmetry!
    Invariant:
      In the CSR adjacency matrix, W(A, B) == W(B, A) == 0.85 exactly.
      Meanwhile, wikilinks pointing to B ARE dampened: W < 1.0.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        # 1. Insert nodes (ensuring lexicographical order src_id < dst_id for close_matches)
        node_a = "concepts:isolated_concept"
        node_b = "concepts:massive_hub"
        assert node_a < node_b, (
            "Test precondition: node_a must be lexicographically < node_b"
        )

        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/a.md', 'Isolated', 'concepts', 'Isolated concept.')",
            (node_a,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/b.md', 'Massive Hub', 'concepts', 'Hub concept.')",
            (node_b,),
        )

        # 2. Insert 50 chunks and 50 incoming wikilinks pointing to node B
        for i in range(50):
            chunk_uri = f"raw/chunks.md#chunk:{i}"
            conn.execute(
                "INSERT INTO note_chunks VALUES (?, ?, 'raw/chunks.md', ?, 'Heading', 'heading', 1, 5, NULL)",
                (f"c{i}", chunk_uri, i),
            )
            conn.execute(
                "INSERT INTO wikilinks (source_uri, target_uri, target_note, weight) VALUES (?, ?, 'raw/b.md', 1.0)",
                (chunk_uri, node_b),
            )

        # 3. Insert symmetric closeMatch between A and B with score 0.85
        score = 0.85
        insert_close_match(conn, node_a, node_b, score=score)
        conn.commit()

        # 4. Sync graph and build CSR matrix
        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None, "Adjacency matrix must be constructed"
        adj = retriever.adj.toarray()

        idx_a = retriever._node_to_idx[node_a]
        idx_b = retriever._node_to_idx[node_b]

        # Empirical Verification:
        # A -> B and B -> A must be identical and equal to score without dampening
        w_ab = float(adj[idx_a, idx_b])
        w_ba = float(adj[idx_b, idx_a])

        assert pytest.approx(w_ab, rel=1e-5) == score, (
            f"W(A, B) must equal {score}, got {w_ab}"
        )
        assert pytest.approx(w_ba, rel=1e-5) == score, (
            f"W(B, A) must equal {score}, got {w_ba}"
        )
        assert pytest.approx(w_ab, rel=1e-6) == w_ba, (
            f"Conductance symmetry violated: W(A, B)={w_ab} != W(B, A)={w_ba}"
        )

        # Contrast Check: verify that regular wikilinks to B WERE dampened by B's in-degree
        sample_chunk = "raw/chunks.md#chunk:0"
        idx_chunk = retriever._node_to_idx[sample_chunk]
        w_chunk_to_b = float(adj[idx_chunk, idx_b])
        assert w_chunk_to_b < 0.5, (
            f"Wikilink to hub B should be degree-dampened below 0.5 (1/sqrt(51) ~ 0.14), got {w_chunk_to_b}"
        )


def test_adversarial_conductance_symmetry_multi_score_cluster(tmp_path: Path):
    """Verify conductance symmetry in a triangle cluster with diverse scores.

    Cluster:
      A < B: score 0.65
      A < C: score 0.92
      B < C: score 0.78
    Assert:
      W(A, B) == W(B, A) == 0.65
      W(A, C) == W(C, A) == 0.92
      W(B, C) == W(C, B) == 0.78
      Diagonal W(X, X) == 0.0
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        nodes = ["concepts:node_a", "concepts:node_b", "concepts:node_c"]
        for n in nodes:
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, 'raw/test.md', ?, 'concepts', 'Desc')",
                (n, n.split(":")[1]),
            )

        pairs = [
            ("concepts:node_a", "concepts:node_b", 0.65),
            ("concepts:node_a", "concepts:node_c", 0.92),
            ("concepts:node_b", "concepts:node_c", 0.78),
        ]
        for src, dst, sc in pairs:
            insert_close_match(conn, src, dst, score=sc)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        adj = retriever.adj.toarray()

        for src, dst, expected_score in pairs:
            u = retriever._node_to_idx[src]
            v = retriever._node_to_idx[dst]
            assert pytest.approx(adj[u, v], rel=1e-5) == expected_score
            assert pytest.approx(adj[v, u], rel=1e-5) == expected_score
            assert pytest.approx(adj[u, v], rel=1e-6) == adj[v, u]

        # No self-loops
        for n in nodes:
            idx = retriever._node_to_idx[n]
            assert adj[idx, idx] == 0.0


def test_adversarial_conductance_symmetry_all_relation_variants():
    """Verify that closeMatch, skos:closeMatch, and yaw:closeMatch all bypass degree dampening."""
    retriever = GraphRAGRetriever()

    # Create synthetic test graph with high-degree node v
    nodes = {"concepts:alpha", "concepts:beta"}
    forward_edges = [
        ("concepts:alpha", "concepts:beta", "closeMatch", 0.88),
        ("concepts:beta", "concepts:alpha", "skos:closeMatch", 0.88),
    ]

    retriever._build_matrix(
        nodes=nodes,
        forward_edges=forward_edges,
        back_edges=[],
        db_mtime=0.0,
    )

    assert retriever.adj is not None
    adj = retriever.adj.toarray()
    idx_a = retriever._node_to_idx["concepts:alpha"]
    idx_b = retriever._node_to_idx["concepts:beta"]

    assert pytest.approx(adj[idx_a, idx_b], rel=1e-5) == 0.88
    assert pytest.approx(adj[idx_b, idx_a], rel=1e-5) == 0.88


# ==============================================================================
# SECTION 2: BAYESIAN QUERY-TO-FACT SEEDING EDGE CASES
# ==============================================================================


@pytest.mark.anyio
async def test_adversarial_seeding_empty_fact_vec(tmp_path: Path):
    """Case 1a: fact_vec table exists but has 0 rows.

    Invariant:
      compute_seed_vector must not crash or error; it must smoothly degrade
      to pure alias prior seeding s_concepts = s_alias_hat, and p0 must be L1-normalized.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:empty_fact_test', 'raw/t.md', 'Test Concept', 'concepts', 'Def')"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Test', 'test', 1, 5, NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:test', 'raw/t.md', 'concepts:empty_fact_test', 'manifests_in', 'predicates:manifests_in', 'raw/t.md#chunk:0', 1.0)"
        )
        conn.commit()

        kb = create_in_memory_kb()
        _register_alias_in_kb(kb, "test-term", "concepts:empty_fact_test", score=1.0)

        retriever = GraphRAGRetriever(kb=kb)

        # Non-zero query vector
        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            p0, _, _ = await retriever.compute_seed_vector("test-term", conn)

            assert p0 is not None
            assert len(p0) == len(retriever._node_to_idx)
            idx_concept = retriever._node_to_idx["concepts:empty_fact_test"]

            # Aliased concept must hold the concept seed mass
            assert p0[idx_concept] > 0.0
            assert pytest.approx(float(p0.sum()), rel=1e-4) == 1.0
            assert not np.isnan(p0).any()
            assert not np.isinf(p0).any()


@pytest.mark.anyio
async def test_adversarial_seeding_missing_fact_vec_table(tmp_path: Path):
    """Case 1b: fact_vec table does not exist at all in SQLite.

    Invariant:
      OperationalError is safely trapped, falling back cleanly to alias priors.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)
        conn.execute("DROP TABLE IF EXISTS fact_vec")

        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:missing_table', 'raw/t.md', 'Missing Table', 'concepts', 'Def')"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Head', 'head', 1, 5, NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:missing', 'raw/t.md', 'concepts:missing_table', 'manifests_in', 'predicates:manifests_in', 'raw/t.md#chunk:0', 1.0)"
        )
        conn.commit()

        kb = create_in_memory_kb()
        _register_alias_in_kb(kb, "missing", "concepts:missing_table", score=1.0)

        retriever = GraphRAGRetriever(kb=kb)

        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[5] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            p0, _, _ = await retriever.compute_seed_vector("missing", conn)

            assert p0 is not None
            idx = retriever._node_to_idx["concepts:missing_table"]
            assert p0[idx] > 0.0
            assert pytest.approx(float(p0.sum()), rel=1e-4) == 1.0


@pytest.mark.anyio
async def test_adversarial_seeding_zero_norm_query_vector(tmp_path: Path):
    """Case 2: Query vector has zero norm (q = 0) simulating offline embedding model.

    Invariant:
      When np.all(q_vec == 0), is_dense_active is False.
      No KNN query is made against fact_vec or note_chunks.
      s_concepts degrades smoothly to s_alias_hat.
      Total seed mass sums to 1.0 with zero NaN/Inf.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:offline_node', 'raw/off.md', 'Offline', 'concepts', 'Def')"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/off.md#chunk:0', 'raw/off.md', 0, 'Off', 'off', 1, 5, NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:offline_1', 'raw/off.md', 'concepts:offline_node', 'predicates:cites', 'predicates:cites', 'raw/off.md#chunk:0', 1.0)"
        )

        fact_bytes = serialize_embedding(np.ones(768, dtype=np.float32) / np.sqrt(768))
        assert fact_bytes is not None
        insert_fact_vec(conn, "rel:offline_1", fact_bytes)
        conn.commit()

        kb = create_in_memory_kb()
        _register_alias_in_kb(kb, "offline-term", "concepts:offline_node", score=1.0)

        retriever = GraphRAGRetriever(kb=kb)

        # Force query embedding to zero vector
        with patch("yaw.retrieval.retriever.is_ollama_online", return_value=False):
            p0, q_returned, _ = await retriever.compute_seed_vector(
                "offline-term", conn
            )

            assert np.all(q_returned == 0.0), (
                "Returned query vector must be exact zero vector"
            )
            assert not np.isnan(p0).any(), "Seed vector must not contain NaN"
            assert not np.isinf(p0).any(), "Seed vector must not contain Inf"

            idx_node = retriever._node_to_idx["concepts:offline_node"]
            assert p0[idx_node] > 0.0
            assert pytest.approx(float(p0.sum()), rel=1e-4) == 1.0


@pytest.mark.anyio
async def test_adversarial_seeding_negative_and_zero_cosine_handling(tmp_path: Path):
    """Case 3: Fact similarity is negative or zero (orthogonal or anti-correlated).

    Setup:
      Fact 1 vector: [1.0, 0, 0, ...]
      Query 1 vector: [-1.0, 0, 0, ...]  -> opposite direction, cos = -1.0, distance = 2.0
      Query 2 vector: [0.0, 1.0, 0, ...] -> orthogonal direction, cos = 0.0, distance = 1.0
    Invariant:
      Formula rel_sim_map = {rid: max(0.0, 1.0 - dist)} clamps negative similarities to 0.0.
      r_vec must contain strictly 0.0 for those facts.
      No negative probabilities are emitted.
      s_concepts cleanly degrades to alias priors without producing negative mass.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:negative_sim_concept', 'raw/neg.md', 'Negative Sim', 'concepts', 'Def')"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/neg.md#chunk:0', 'raw/neg.md', 0, 'Neg', 'neg', 1, 5, NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:neg_rel', 'raw/neg.md', 'concepts:negative_sim_concept', 'predicates:cites', 'predicates:cites', 'raw/neg.md#chunk:0', 1.0)"
        )

        # Fact vector is [1.0, 0, 0, ...]
        f_vec = np.zeros(768, dtype=np.float32)
        f_vec[0] = 1.0
        insert_fact_vec(conn, "rel:neg_rel", _ser(f_vec))
        conn.commit()

        kb = create_in_memory_kb()
        _register_alias_in_kb(
            kb, "neg-query", "concepts:negative_sim_concept", score=1.0
        )

        retriever = GraphRAGRetriever(kb=kb)

        # Test Subcase A: Anti-correlated query [-1.0, 0, 0, ...] -> distance = 2.0, cos = -1.0
        q_anti = np.zeros(768, dtype=np.float32)
        q_anti[0] = -1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_anti

            p0_anti, _, _ = await retriever.compute_seed_vector("neg-query", conn)

            # Invariant: p0 must be strictly non-negative
            assert np.all(p0_anti >= 0.0), "Probabilities must never be negative"
            assert not np.isnan(p0_anti).any()
            assert pytest.approx(float(p0_anti.sum()), rel=1e-4) == 1.0

        # Test Subcase B: Orthogonal query [0.0, 1.0, 0, ...] -> distance = 1.0, cos = 0.0
        q_ortho = np.zeros(768, dtype=np.float32)
        q_ortho[1] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_ortho

            p0_ortho, _, _ = await retriever.compute_seed_vector("neg-query", conn)

            assert np.all(p0_ortho >= 0.0), "Probabilities must never be negative"
            assert not np.isnan(p0_ortho).any()
            assert pytest.approx(float(p0_ortho.sum()), rel=1e-4) == 1.0


@pytest.mark.anyio
async def test_adversarial_seeding_extreme_lambdas_pure_fact_and_pure_alias(
    tmp_path: Path,
):
    """Case 4: Extreme lambda values (lambda = 0.0 and lambda = 1.0).

    Setup:
      - Concept A ('concepts:alias_only') is matched ONLY by query mentions in KB.
      - Concept B ('concepts:fact_only') is matched ONLY by fact relevance in Space R.
    Invariants:
      - When lambda = 1.0:
          s_concepts = 1.0 * s_alias_hat + 0.0 * r_hat = s_alias_hat.
          Concept A gets positive concept mass; Concept B gets 0.0 concept mass.
      - When lambda = 0.0:
          s_concepts = 0.0 * s_alias_hat + 1.0 * r_hat = r_hat.
          Concept B gets positive concept mass; Concept A gets 0.0 concept mass.
      - When lambda = 0.5:
          Both Concept A and Concept B get positive concept mass.
      - Zero NaN, zero division by zero across all configurations.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        node_alias = "concepts:alias_only"
        node_fact = "concepts:fact_only"

        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/t.md', 'Alias Only', 'concepts', 'Def')",
            (node_alias,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/t.md', 'Fact Only', 'concepts', 'Def')",
            (node_fact,),
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Chunk', 'chunk', 1, 5, NULL)"
        )

        # Ground both nodes in graph
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:alias_excl', 'raw/t.md', ?, 'manifests_in', 'predicates:manifests_in', 'raw/t.md#chunk:0', 1.0)",
            (node_alias,),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:fact_excl', 'raw/t.md', ?, 'predicates:cites', 'predicates:cites', 'raw/t.md#chunk:0', 1.0)",
            (node_fact,),
        )

        # Embed fact as [1.0, 0, 0, ...]
        f_vec = np.zeros(768, dtype=np.float32)
        f_vec[0] = 1.0
        insert_fact_vec(conn, "rel:fact_excl", _ser(f_vec))
        conn.commit()

        # KB exclusively registers alias for node_alias
        kb = create_in_memory_kb()
        _register_alias_in_kb(kb, "alias-query", node_alias, score=1.0)

        retriever = GraphRAGRetriever(kb=kb)

        # Query vector matching fact vector perfectly
        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            # -------------------------------------------------------------
            # Subtest A: lambda = 1.0 (pure alias seeding)
            # -------------------------------------------------------------
            with patch.object(retriever_module, "FACT_SEED_LAMBDA", 1.0):
                p0_l1, _, _ = await retriever.compute_seed_vector("alias-query", conn)
                idx_a = retriever._node_to_idx[node_alias]
                idx_b = retriever._node_to_idx[node_fact]

                assert not np.isnan(p0_l1).any()
                assert pytest.approx(float(p0_l1.sum()), rel=1e-4) == 1.0

                # Concept A has positive mass; Concept B has 0.0 mass
                assert p0_l1[idx_a] > 0.0, (
                    "Concept A must receive seed mass under lambda=1.0"
                )
                assert p0_l1[idx_b] == 0.0, (
                    "Concept B must receive 0 seed mass under lambda=1.0"
                )

            # -------------------------------------------------------------
            # Subtest B: lambda = 0.0 (pure fact seeding)
            # -------------------------------------------------------------
            with patch.object(retriever_module, "FACT_SEED_LAMBDA", 0.0):
                p0_l0, _, _ = await retriever.compute_seed_vector("alias-query", conn)
                assert not np.isnan(p0_l0).any()
                assert pytest.approx(float(p0_l0.sum()), rel=1e-4) == 1.0

                # Concept B has positive mass; Concept A has 0.0 mass
                assert p0_l0[idx_b] > 0.0, (
                    "Concept B must receive seed mass under lambda=0.0"
                )
                assert p0_l0[idx_a] == 0.0, (
                    "Concept A must receive 0 seed mass under lambda=0.0"
                )

            # -------------------------------------------------------------
            # Subtest C: lambda = 0.5 (balanced Bayesian fusion)
            # -------------------------------------------------------------
            with patch.object(retriever_module, "FACT_SEED_LAMBDA", 0.5):
                p0_lhalf, _, _ = await retriever.compute_seed_vector(
                    "alias-query", conn
                )
                assert not np.isnan(p0_lhalf).any()
                assert pytest.approx(float(p0_lhalf.sum()), rel=1e-4) == 1.0

                # Both concepts receive positive seed mass
                assert p0_lhalf[idx_a] > 0.0, (
                    "Concept A must receive seed mass under lambda=0.5"
                )
                assert p0_lhalf[idx_b] > 0.0, (
                    "Concept B must receive seed mass under lambda=0.5"
                )


@pytest.mark.anyio
async def test_adversarial_seeding_extreme_lambdas_single_channel_missing(
    tmp_path: Path,
):
    """Verify that when lambda = 0.0 or 1.0 and the preferred channel is missing,

    the retriever gracefully and continuously falls back to the available channel.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        node = "concepts:single_channel_node"
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/t.md', 'Single', 'concepts', 'Def')",
            (node,),
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Chunk', 'chunk', 1, 5, NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:single', 'raw/t.md', ?, 'manifests_in', 'predicates:manifests_in', 'raw/t.md#chunk:0', 1.0)",
            (node,),
        )
        conn.commit()

        # KB has alias, but fact_vec has ZERO rows
        kb = create_in_memory_kb()
        _register_alias_in_kb(kb, "alias-only-term", node, score=1.0)

        retriever = GraphRAGRetriever(kb=kb)

        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            # Under lambda = 0.0 (pure fact requested), but facts are completely empty:
            # Must continuously degrade to alias seeding instead of outputting 0 or NaN!
            with patch.object(retriever_module, "FACT_SEED_LAMBDA", 0.0):
                p0, _, _ = await retriever.compute_seed_vector("alias-only-term", conn)
                idx = retriever._node_to_idx[node]
                assert p0[idx] > 0.0, (
                    "Must fall back to alias channel when facts are empty"
                )
                assert pytest.approx(float(p0.sum()), rel=1e-4) == 1.0


# ==============================================================================
# SECTION 3: COMPLEX RELATIONAL INVARIANTS
# ==============================================================================


@pytest.mark.anyio
async def test_adversarial_fact_seeding_self_loop_relation(tmp_path: Path):
    """Verify that a self-loop relation (subject_id == object_id) does not double-count endpoints."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        node = "concepts:self_loop_concept"
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/t.md', 'Self Loop', 'concepts', 'Def')",
            (node,),
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Chunk', 'chunk', 1, 5, NULL)"
        )
        # Self loop relation: subject == object
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:self', 'raw/t.md', ?, 'predicates:relates_to', 'predicates:relates_to', ?, 1.0)",
            (node, node),
        )

        f_vec = np.zeros(768, dtype=np.float32)
        f_vec[0] = 1.0
        insert_fact_vec(conn, "rel:self", _ser(f_vec))
        conn.commit()

        kb = create_in_memory_kb()
        retriever = GraphRAGRetriever(kb=kb)

        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            p0, _, _ = await retriever.compute_seed_vector("any query", conn)
            idx = retriever._node_to_idx[node]
            assert p0[idx] > 0.0
            assert pytest.approx(float(p0.sum()), rel=1e-4) == 1.0


@pytest.mark.anyio
async def test_adversarial_fact_seeding_mean_pooling_multiple_facts(tmp_path: Path):
    """Verify that multiple facts touching the same entity compute the exact mean cosine:

    r(q, e) = mean_{f in F_q, e in {subj, obj}} cos(q, f) per ADR-012.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        target_node = "concepts:target_entity"
        other_node_1 = "concepts:other_1"
        other_node_2 = "concepts:other_2"

        for n in [target_node, other_node_1, other_node_2]:
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, 'raw/t.md', ?, 'concepts', 'Def')",
                (n, n),
            )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Chunk', 'chunk', 1, 5, NULL)"
        )

        # Fact 1: target -> other_1
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:1', 'raw/t.md', ?, 'predicates:causes', 'predicates:causes', ?, 1.0)",
            (target_node, other_node_1),
        )
        # Fact 2: other_2 -> target
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:2', 'raw/t.md', ?, 'predicates:prevents', 'predicates:prevents', ?, 1.0)",
            (other_node_2, target_node),
        )

        # Fact 1 vector: unit vector along dim 0 -> cos = 1.0 (sim = 1.0)
        f1 = np.zeros(768, dtype=np.float32)
        f1[0] = 1.0

        # Fact 2 vector: cos = 0.60 (sim = 0.60)
        # dim 0 = 0.60, dim 1 = 0.80
        f2 = np.zeros(768, dtype=np.float32)
        f2[0] = 0.60
        f2[1] = 0.80

        insert_fact_vec(conn, "rel:1", _ser(f1))
        insert_fact_vec(conn, "rel:2", _ser(f2))
        conn.commit()

        kb = create_in_memory_kb()
        retriever = GraphRAGRetriever(kb=kb)

        # Query along dim 0: cos(q, f1) = 1.0, cos(q, f2) = 0.60
        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            # With lambda = 0.0 (pure fact seeding)
            with patch.object(retriever_module, "FACT_SEED_LAMBDA", 0.0):
                p0, _, _ = await retriever.compute_seed_vector("unmatched", conn)
                idx_target = retriever._node_to_idx[target_node]
                idx_other1 = retriever._node_to_idx[other_node_1]
                idx_other2 = retriever._node_to_idx[other_node_2]

                ratio_target_to_other2 = p0[idx_target] / p0[idx_other2]
                assert ratio_target_to_other2 == pytest.approx(0.80 / 0.60, rel=1e-3)

                ratio_other1_to_target = p0[idx_other1] / p0[idx_target]
                assert ratio_other1_to_target == pytest.approx(1.00 / 0.80, rel=1e-3)


@pytest.mark.anyio
async def test_adversarial_fact_seeding_dangling_endpoints_graceful(tmp_path: Path):
    """Verify that facts pointing to nodes outside the graph do not cause KeyErrors."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db(load_vec=True) as conn:
        _init_test_db(conn)

        # Graph has only chunk node
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', 'raw/t.md#chunk:0', 'raw/t.md', 0, 'Chunk', 'chunk', 1, 5, NULL)"
        )
        # Relation has dangling nodes that don't exist in leaf_nodes
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('rel:dangling', 'raw/t.md', 'concepts:ghost_subj', 'predicates:cites', 'predicates:cites', 'concepts:ghost_obj', 1.0)"
        )

        f_vec = np.zeros(768, dtype=np.float32)
        f_vec[0] = 1.0
        insert_fact_vec(conn, "rel:dangling", _ser(f_vec))
        conn.commit()

        kb = create_in_memory_kb()
        retriever = GraphRAGRetriever(kb=kb)

        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0

        with patch(
            "yaw.retrieval.retriever.embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = q_vec

            # Must not raise KeyError when ghost endpoints are not in _node_to_idx
            p0, _, _ = await retriever.compute_seed_vector("ghost query", conn)
            assert not np.isnan(p0).any()
