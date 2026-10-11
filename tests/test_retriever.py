"""Unit tests for SciPy CSR GraphRAG retriever, directional conductance, and dual-layer formatting."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import scipy.sparse as sp

from yaw import config
from yaw.nlp.embeddings import deserialize_embedding
from yaw.retrieval.retriever import (
    GraphRAGRetriever,
    Retriever,
    _is_chunk,
    _is_concept,
)
from yaw.storage.db import ensure_schema_views


def test_concept_and_chunk_node_classification():
    """Verify prefix-based identification of ontological concepts and note chunks across all 9 namespaces."""
    for ns in (
        "entities",
        "concepts",
        "predicates",
        "invariants",
        "decisions",
        "claims",
        "processes",
        "sources",
        "contexts",
    ):
        assert _is_concept(f"{ns}:sample-node")
        assert not _is_chunk(f"{ns}:sample-node")

    assert not _is_concept("raw/physics.md#chunk:0")
    assert not _is_concept("raw/biology.md")
    assert not _is_concept("unknown:node")

    assert _is_chunk("raw/physics.md#chunk:0")
    assert _is_chunk("raw/notes/quantum.md#chunk:5")
    assert not _is_chunk("concepts:uncertainty-principle")
    assert not _is_chunk("entities:heisenberg")


def test_csr_matrix_directional_conductance_and_hub_dampening(tmp_path):
    """Verify CSR transition matrix construction, row-stochasticity, and directional conductance."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )

        # 1. Downward edge: concept -> chunk (dir_w = 1.0)
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:quantum', 'manifests_in', 'raw/physics.md#chunk:0', 'raw/physics.md', 1.0)"
        )
        # 2. Upward edge: chunk -> concept (dir_w = 0.2)
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('raw/physics.md#chunk:0', 'mentions', 'concepts:quantum', 'raw/physics.md', 1.0)"
        )
        # 3. Peer edge: concept -> concept (dir_w = 1.0)
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:quantum', 'subClassOf', 'concepts:physics', 'raw/physics.md', 1.0)"
        )
        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever._P is not None
        assert isinstance(retriever._P, sp.csr_matrix)
        assert len(retriever._idx_to_node) == 3

        P = retriever._P.toarray()
        c_quant_idx = retriever._node_to_idx["concepts:quantum"]
        c_phys_idx = retriever._node_to_idx["concepts:physics"]
        chunk_idx = retriever._node_to_idx["raw/physics.md#chunk:0"]

        # Non-dangling rows must be row-stochastic (sum to 1.0)
        assert pytest.approx(P[c_quant_idx].sum(), rel=1e-5) == 1.0
        assert pytest.approx(P[chunk_idx].sum(), rel=1e-5) == 1.0

        # Row for chunk only transitions to concept (prob = 1.0)
        assert pytest.approx(P[chunk_idx, c_quant_idx], rel=1e-5) == 1.0

        # Row for concept: down to chunk (dir_w = 1.0) vs peer to concept (dir_w = 1.0)
        assert P[c_quant_idx, chunk_idx] > 0.0
        assert P[c_quant_idx, c_phys_idx] > 0.0


def test_power_iteration_convergence():
    """Verify PPR power iteration converges and preserves probability conservation."""
    retriever = GraphRAGRetriever()
    n = 4
    # Create a cyclic stochastic transition matrix
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    retriever._P = sp.csr_matrix(adj)
    retriever._dangling_mask = np.zeros(n, dtype=bool)

    p0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    p_star = retriever.power_iteration(p0, alpha=0.85, max_iter=50)

    assert len(p_star) == n
    assert pytest.approx(float(np.sum(p_star)), rel=1e-4) == 1.0
    # Highest mass at seed
    assert p_star[0] > p_star[2]


@pytest.mark.anyio
async def test_dual_layer_retrieval_rendering(tmp_path):
    """Verify retrieve() returns formatted Ontological Context and Grounded Evidence with disk hydration."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "heisenberg.md"
    note_content = "# Heisenberg\nWerner Heisenberg discovered the uncertainty principle in quantum physics."
    note_path.write_text(note_content, encoding="utf-8")

    dummy_vec = np.ones(768, dtype=np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        chunk_uri = "raw/heisenberg.md#chunk:0"
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/heisenberg.md', 0, 'Heisenberg', 'heisenberg', 1, 2, ?)",
            (chunk_uri, dummy_vec.tobytes()),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:uncertainty-principle', 'raw/heisenberg.md', 'Uncertainty Principle', 'concepts', 'Fundamental limit of precision.', ?)",
            (dummy_vec.tobytes(),),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:uncertainty-principle', 'discovered_by', 'entities:heisenberg', 'raw/heisenberg.md', 1.0)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:uncertainty-principle', 'manifests_in', 'raw/heisenberg.md#chunk:0', 'raw/heisenberg.md', 1.0)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/heisenberg.md', 'Heisenberg', ?)",
            (chunk_uri, note_content),
        )
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    with patch("yaw.nlp.embeddings.embed_query", new_callable=AsyncMock) as mock_qv:
        mock_qv.return_value = dummy_vec

        result = await retriever.retrieve("uncertainty principle", top_k=1)

        assert "# Result: [1]" in result
        assert "## Ontological Context" in result
        assert "## Evidence Chunk" in result
        assert "concepts:uncertainty-principle" in result
        assert "discovered_by" in result
        assert "raw/heisenberg.md:1-2#chunk:0" in result
        assert "Werner Heisenberg discovered the uncertainty principle" in result


def test_deserialize_embedding_all_variants():
    """Verify SSOT deserialize_embedding handles all expected types and failure modes."""
    # 1. None
    assert deserialize_embedding(None) is None

    # 2. Empty string / whitespace
    assert deserialize_embedding("") is None
    assert deserialize_embedding("   \n\t  ") is None

    # 3. JSON text string
    json_str = "[0.1, -0.2, 0.35]"
    res = deserialize_embedding(json_str)
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, np.array([0.1, -0.2, 0.35], dtype=np.float32))

    # 4. Binary BLOB (bytes and memoryview)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b_data = arr.tobytes()
    res_b = deserialize_embedding(b_data)
    assert isinstance(res_b, np.ndarray)
    np.testing.assert_allclose(res_b, arr)

    res_mv = deserialize_embedding(memoryview(b_data))
    assert isinstance(res_mv, np.ndarray)
    np.testing.assert_allclose(res_mv, arr)

    assert deserialize_embedding(b"") is None

    # 5. np.ndarray
    res_arr = deserialize_embedding(arr)
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.dtype == np.float32

    # 6. list and tuple
    res_list = deserialize_embedding([0.5, 0.25])
    assert isinstance(res_list, np.ndarray)
    assert res_list.dtype == np.float32

    res_tuple = deserialize_embedding((0.5, 0.25))
    assert isinstance(res_tuple, np.ndarray)
    assert res_tuple.dtype == np.float32

    # 7. Invalid JSON string raises ValueError
    with pytest.raises(ValueError, match="Failed to parse embedding JSON string"):
        deserialize_embedding("[0.1, 0.2, unclosed")

    # 8. Unsupported type raises TypeError
    with pytest.raises(TypeError, match="Unsupported embedding type"):
        deserialize_embedding(12345)


@pytest.mark.anyio
async def test_dual_layer_retrieval_with_json_text_embeddings(tmp_path):
    """Verify GraphRAGRetriever succeeds when SQLite stores embeddings as JSON text (Cocoindex format)."""
    import json

    vault_dir = tmp_path / "vault"
    raw_dir = vault_dir / "raw"
    cache_dir = vault_dir / ".cache"
    raw_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    note_path = raw_dir / "planck.md"
    note_content = "Max Planck originated quantum theory in 1900."
    note_path.write_text(note_content, encoding="utf-8")

    config.set_vault(vault_dir)

    db_path = cache_dir / "yaw.db"
    dummy_vec = np.ones(768, dtype=np.float32)
    norm = float(np.linalg.norm(dummy_vec))
    dummy_vec /= norm
    json_vec_str = json.dumps(dummy_vec.tolist())

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding TEXT)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, "
            "embedding TEXT, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute("CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT)")
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        chunk_uri = "raw/planck.md#chunk:0"
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/planck.md', 0, 'Planck', 'planck', 1, 2, ?)",
            (chunk_uri, json_vec_str),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:quantum-hypothesis', 'raw/planck.md', 'Quantum Hypothesis', 'concepts', 'Energy quanta.', ?)",
            (json_vec_str,),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:quantum-hypothesis', 'manifests_in', 'raw/planck.md#chunk:0', 'raw/planck.md', 1.0)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/planck.md', 'Planck', ?)",
            (chunk_uri, note_content),
        )
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    with patch("yaw.nlp.embeddings.embed_query", new_callable=AsyncMock) as mock_qv:
        mock_qv.return_value = dummy_vec

        result = await retriever.retrieve("quantum hypothesis", top_k=1)
        assert "# Result: [1]" in result
        assert "## Ontological Context" in result
        assert "## Evidence Chunk" in result
        assert "concepts:quantum-hypothesis" in result
        assert "Max Planck originated quantum theory" in result


def test_retriever_reciprocal_wikilinks_coexistence_with_ontological_edges(tmp_path):
    """Verify reciprocal wikilinks coexist seamlessly with ontological concept/chunk edges."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )

        # 1. Downward concept -> chunk
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:gravity', 'manifests_in', 'raw/relativity.md#chunk:0', 'raw/relativity.md', 1.0)"
        )
        # 2. Wikilinks between chunks: relativity <-> quantum
        conn.execute(
            "INSERT INTO wikilinks VALUES ('raw/relativity.md#chunk:0', 'raw/quantum.md#chunk:0', 'quantum', 1.0)"
        )
        conn.execute(
            "INSERT INTO wikilinks VALUES ('raw/quantum.md#chunk:0', 'raw/relativity.md#chunk:0', 'relativity', 1.0)"
        )

        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        adj = retriever.adj.toarray()
        node_to_idx = retriever._node_to_idx

        u_rel = node_to_idx["raw/relativity.md#chunk:0"]
        u_q = node_to_idx["raw/quantum.md#chunk:0"]
        u_grav = node_to_idx["concepts:gravity"]

        # Incoming to quantum (in-degree 1): 1.0 (fwd) + 0.5 (rev) = 1.5
        assert pytest.approx(float(adj[u_rel, u_q]), abs=1e-5) == 1.5

        # Incoming to relativity (in-degree 2 from gravity + quantum): (1.0 + 0.5) / sqrt(2)
        assert pytest.approx(float(adj[u_q, u_rel]), abs=1e-5) == 1.5 / (2.0**0.5)

        # Downward concept to chunk is present with dir_w = 1.0 / sqrt(2)
        assert pytest.approx(float(adj[u_grav, u_rel]), abs=1e-5) == 1.0 / (2.0**0.5)

        # Row-stochastic property verified across all non-dangling rows
        assert retriever._P is not None
        P = retriever._P.toarray()
        assert pytest.approx(float(P[u_rel].sum()), rel=1e-5) == 1.0
        assert pytest.approx(float(P[u_q].sum()), rel=1e-5) == 1.0
        assert pytest.approx(float(P[u_grav].sum()), rel=1e-5) == 1.0


@pytest.mark.anyio
async def test_retriever_offline_lexical_fallback(tmp_path):
    """Verify offline fallback correctly resolves chunks via entity_uri and hydrates lexical evidence."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "test.md"
    note_content = "# Test Document\nThis note contains search query content that should be retrieved offline."
    note_path.write_text(note_content, encoding="utf-8")

    chunk_uri = "raw/test.md#chunk:0"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/test.md', 0, 'Test Document', 'test-document', 1, 2, NULL)",
            (chunk_uri,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:test', 'raw/test.md', 'Test Concept', 'concepts', 'Concept for offline fallback test.', NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:test', 'manifests_in', ?, 'raw/test.md', 1.0)",
            (chunk_uri,),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/test.md', 'Test Document', ?)",
            (chunk_uri, note_content),
        )
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    with patch("yaw.retrieval.retriever.is_ollama_online", return_value=False):
        result = await retriever.retrieve("search query", top_k=1)

    assert "_No ground-truth passages found._" not in result
    assert (
        "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold."
        not in result
    )
    assert "search query content that should be retrieved offline" in result
    assert "raw/test.md:1-2#chunk:0" in result


@pytest.mark.anyio
async def test_retriever_concept_diffusion_reaches_unseeded_chunk(tmp_path):
    """Verify PPR diffuses probability mass from an FTS-seeded concept to an unseeded chunk via manifests_in."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Note text deliberately does NOT contain the query term "photosynthesis"
    note_path = raw_dir / "chloroplast.md"
    note_content = (
        "# Cellular Energy\n"
        "Chloroplasts capture photons from sunlight to convert carbon dioxide into glucose."
    )
    note_path.write_text(note_content, encoding="utf-8")

    chunk_uri = "raw/chloroplast.md#chunk:0"
    concept_uri = "concepts:photosynthesis"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/chloroplast.md', 0, 'Cellular Energy', 'cellular-energy', 1, 2, NULL)",
            (chunk_uri,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/chloroplast.md', 'Photosynthesis', 'concepts', 'Biological synthesis of organic compounds.', NULL)",
            (concept_uri,),
        )
        # Downward edge: concept -> chunk
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'manifests_in', ?, 'raw/chloroplast.md', 1.0)",
            (concept_uri, chunk_uri),
        )
        # FTS table only indexes the concept, NOT the chunk
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/chloroplast.md', 'Photosynthesis', 'Biological synthesis of organic compounds using light.')",
            (concept_uri,),
        )
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    with patch("yaw.retrieval.retriever.is_ollama_online", return_value=False):
        result = await retriever.retrieve("photosynthesis", top_k=1)

    assert "# Result: [1]" in result
    assert "## Ontological Context" in result
    assert "concepts:photosynthesis" in result
    assert "## Evidence Chunk" in result
    assert "raw/chloroplast.md:1-2#chunk:0" in result
    assert "Chloroplasts capture photons" in result
    assert "_No ground-truth passages found._" not in result
    assert (
        "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold."
        not in result
    )


@pytest.mark.anyio
async def test_retriever_bridging_subgraph_filters_unlinked_distractor_concepts(
    tmp_path,
):
    """Verify bridging subgraph filters out unlinked distractor concepts even if high-degree."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "relativity.md"
    note_content = (
        "# Special Relativity\n"
        "Albert Einstein formulated special relativity describing inertial reference frames."
    )
    note_path.write_text(note_content, encoding="utf-8")

    chunk_uri = "raw/relativity.md#chunk:0"
    grounded_concept = "concepts:special-relativity"
    distractor_concept = "concepts:unrelated-distractor"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/relativity.md', 0, 'Special Relativity', 'special-relativity', 1, 2, NULL)",
            (chunk_uri,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/relativity.md', 'Special Relativity', 'concepts', 'Theory of space and time.', NULL)",
            (grounded_concept,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/distractor.md', 'Unrelated Distractor', 'concepts', 'High-degree distractor concept.', NULL)",
            (distractor_concept,),
        )
        # Grounded concept has manifests_in pointing to the winning chunk
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'manifests_in', ?, 'raw/relativity.md', 1.0)",
            (grounded_concept, chunk_uri),
        )

        # Distractor concept forms a high-degree hub with other distractor nodes
        for i in range(5):
            other_node = f"concepts:hub-peer-{i}"
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, 'raw/distractor.md', 'Hub Peer', 'concepts', 'Peer node.', NULL)",
                (other_node,),
            )
            conn.execute(
                "INSERT INTO leaf_relations VALUES (?, 'relates_to', ?, 'raw/distractor.md', 1.0)",
                (other_node, distractor_concept),
            )
            conn.execute(
                "INSERT INTO leaf_relations VALUES (?, 'relates_to', ?, 'raw/distractor.md', 1.0)",
                (distractor_concept, other_node),
            )

        # Connect chunk to distractor with generic relation so it receives diffusion mass
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'generic_rel', ?, 'raw/relativity.md', 1.0)",
            (chunk_uri, distractor_concept),
        )

        # FTS indexes the chunk for query
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/relativity.md', 'Special Relativity', ?)",
            (chunk_uri, note_content),
        )
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    with patch("yaw.retrieval.retriever.is_ollama_online", return_value=False):
        result = await retriever.retrieve("inertial reference frames", top_k=1)

    assert "# Result: [1]" in result
    assert "## Ontological Context" in result
    assert "concepts:special-relativity" in result
    assert "concepts:unrelated-distractor" not in result
    assert "raw/relativity.md:1-2#chunk:0" in result


@pytest.mark.anyio
async def test_retriever_zero_chunks_guidance_and_global_concepts(tmp_path):
    """Verify K=0 falls back to global top concepts and renders recommended source research guidance."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    concept_uri = "concepts:general-relativity"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/relativity.md', 'General Relativity', 'concepts', 'Geometric theory of gravitation.', NULL)",
            (concept_uri,),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'extends', 'concepts:special-relativity', 'raw/relativity.md', 1.0)",
            (concept_uri,),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/relativity.md', 'General Relativity', 'Geometric theory of gravitation Einstein.')",
            (concept_uri,),
        )
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    with patch("yaw.retrieval.retriever.is_ollama_online", return_value=False):
        result = await retriever.retrieve("general relativity", top_k=2)

    assert "# Result: 0" in result
    assert "## Ontological Context" in result
    assert "concepts:general-relativity" in result
    assert "extends" in result
    assert "## Evidence Chunk" in result
    assert (
        "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold. Further source research and ingestion via ingest_raw is recommended to grow the corpus._"
        in result
    )


def test_render_ontological_context_unified_views(tmp_path):
    """Verify _render_ontological_context reads from unified_edges and unified_nodes views."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        ensure_schema_views(conn)

        conn.execute(
            "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding) VALUES ('invariants:purity', 'raw/rules.md', 'Purity', 'invariants', 'Vault contains only markdown.', NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding) VALUES ('decisions:sqlite', 'raw/rules.md', 'SQLite Storage', 'decisions', 'SQLite is the SSOT.', NULL)"
        )
        conn.execute(
            "INSERT INTO leaf_relations (subject_id, predicate, object_id, source_leaf_id, weight) VALUES ('decisions:sqlite', 'enforces', 'invariants:purity', 'raw/rules.md', 1.0)"
        )
        conn.commit()

        retriever = GraphRAGRetriever()
        lines = retriever._render_ontological_context(
            conn, ["decisions:sqlite", "invariants:purity"]
        )

        assert any(
            "`decisions:sqlite` [decisions]: SQLite is the SSOT." in line
            for line in lines
        )
        assert any(
            "`invariants:purity` [invariants]: Vault contains only markdown." in line
            for line in lines
        )
        assert any(
            "`decisions:sqlite` --[enforces]--> `invariants:purity`" in line
            for line in lines
        )


@pytest.mark.anyio
async def test_retriever_cold_start_and_zero_matches(tmp_path):
    """Test GraphRAGRetriever against empty db, shouldn't crash."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        from yaw.storage.db import (
            apply_pragmas,
            ensure_schema_tables,
            ensure_schema_views,
        )

        apply_pragmas(conn)
        ensure_schema_tables(conn)
        ensure_schema_views(conn)
        conn.commit()

    retriever = GraphRAGRetriever()

    # Should not raise exception (e.g. ZeroDivisionError or IndexError)
    result = await retriever.retrieve("__nonexistent_query_xyz_98765__", top_k=5)

    assert result is not None
    assert (
        "Result: 0" in result
        or "Result: [" in result
        or result == ""
        or "Zero relevant evidence chunks" in result
    )


def test_retriever_extract_fts_terms_strips_stopwords() -> None:
    """Verify Retriever._extract_fts_terms strips stopwords and punctuation."""
    terms = Retriever._extract_fts_terms("what is a qubit")
    assert terms == ["qubit"]


def test_create_canonical_node_guards_predicates(tmp_path: Path) -> None:
    """Verify _create_canonical_node guards against predicate namespace insertion into leaf_nodes."""
    from yaw.indexer.canonicalize import _create_canonical_node
    from yaw.storage.db import apply_pragmas, ensure_schema_tables

    db_path = tmp_path / "test_predicate_guard.db"
    conn = sqlite3.connect(str(db_path))
    try:
        apply_pragmas(conn)
        ensure_schema_tables(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )

        # Predicate namespace should be ignored
        _create_canonical_node(
            conn=conn,
            surface_id="causes",
            name="causes",
            namespace="predicates",
            source_leaf_id="raw/doc.md#0",
        )
        row = conn.execute(
            "SELECT * FROM leaf_nodes WHERE node_id = 'causes'"
        ).fetchone()
        assert row is None

        # surface_id starting with predicates: should be ignored
        _create_canonical_node(
            conn=conn,
            surface_id="predicates:relates_to",
            name="relates_to",
            namespace="concept",
            source_leaf_id="raw/doc.md#0",
        )
        row2 = conn.execute(
            "SELECT * FROM leaf_nodes WHERE node_id = 'predicates:relates_to'"
        ).fetchone()
        assert row2 is None

        # Valid entity should be inserted with fallback provenance if source_leaf_id is None
        _create_canonical_node(
            conn=conn,
            surface_id="concept:superposition",
            name="superposition",
            namespace="concept",
            source_leaf_id=None,
        )
        row3 = conn.execute(
            "SELECT node_id, source_leaf_id FROM leaf_nodes WHERE node_id = 'concept:superposition'"
        ).fetchone()
        assert row3 is not None
        assert row3[0] == "concept:superposition"
        assert row3[1] == "system:canonical_induction"
    finally:
        conn.close()


def test_taxonomic_subsumption_conductance_asymmetry_wp516(tmp_path: Path):
    """Verify WP 5.1.6: SubClassOf/PartOf upward abstraction is damped to 0.2 while downward is 1.0."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        from yaw.storage.db import ensure_schema_tables

        ensure_schema_tables(conn)
        # Child --[subClassOf]--> Parent
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:quantum', 'subClassOf', 'concepts:physics', 'raw/doc.md', 1.0)"
        )
        # Sibling anchor so rows have alternatives to demonstrate relative weights
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:quantum', 'manifests_in', 'raw/doc.md#chunk:0', 'raw/doc.md', 1.0)"
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('concepts:physics', 'manifests_in', 'raw/doc.md#chunk:1', 'raw/doc.md', 1.0)"
        )
        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever._P is not None
        P = retriever._P.toarray()
        idx_child = retriever._node_to_idx["concepts:quantum"]
        idx_parent = retriever._node_to_idx["concepts:physics"]

        # Upward abstraction (Child -> Parent via subClassOf) has dir_w = 0.2
        # Downward subsumption (Parent -> Child via synthesized hasSubClass) has dir_w = 1.0
        # In-degree for quantum = 1 (synthesized back-edge does not increase forward in-degree), for physics = 1.
        # Transition Parent -> Child (dir_w=1.0) must be 5x Child -> Parent (dir_w=0.2)
        prob_parent_to_child = P[idx_parent, idx_child]
        prob_child_to_parent = P[idx_child, idx_parent]

        assert prob_parent_to_child > prob_child_to_parent
        assert (
            pytest.approx(prob_parent_to_child / prob_child_to_parent, rel=1e-3) == 3.0
        )


@pytest.mark.anyio
async def test_daemon_mcp_tool_stubs_and_schema_wp518(tmp_path: Path):
    """Verify WP 5.1.8: FastMCP tool stubs and ontology schema contracts."""
    from yaw.server.daemon import (
        compile_wiki,
        get_ontology_schema,
        ingest_raw,
        lint_wiki,
    )

    orig_db = config.DB_PATH
    config.DB_PATH = tmp_path / "nonexistent.db"
    try:
        schema = get_ontology_schema()
        assert len(schema["base_predicates"]) == 14
        assert len(schema["namespaces"]) == 9
        assert "subClassOf" in schema["base_predicates"]
        assert "concepts" in schema["namespaces"]

        comp = await compile_wiki()
        assert comp["status"] == "compiled"
        assert comp["clusters_bisected"] == 0

        lint = await lint_wiki()
        assert lint["status"] == "clean"
        assert lint["violations"] == []

        ing = await ingest_raw("raw/test.md")
        assert ing["status"] == "ingested"
        assert ing["file_path"] == "raw/test.md"
    finally:
        config.DB_PATH = orig_db


def test_bayesian_evidence_chunk_scoring_fusion(tmp_path: Path):
    """Verify ADR-010 Bayesian evidence fusion chunk rescoring: Score(c) = (sum P(e|a)*p*[e] + p*[c]) * sim(q,c)."""
    from yaw.indexer.canonicalize import create_in_memory_kb
    from yaw.retrieval.retriever import GraphRAGRetriever
    from yaw.storage.db import (
        apply_pragmas,
        ensure_schema_tables,
        ensure_schema_views,
        serialize_embedding,
    )

    retriever = GraphRAGRetriever()
    kb = create_in_memory_kb()
    # Register entity and alias with prior probability 0.8
    kb.add_entity(
        entity="concepts:entropy", freq=1, entity_vector=np.zeros(300, dtype=np.float32)
    )
    kb.add_alias(
        alias="shannon-entropy", entities=["concepts:entropy"], probabilities=[0.8]
    )
    retriever.kb = kb

    # Set up node-to-index mapping for graph stationary distribution p*
    retriever._node_to_idx = {
        "concepts:entropy": 0,
        "raw/note.md#chunk:0": 1,
        "raw/note.md#chunk:1": 2,
    }
    p_star = np.array([0.50, 0.30, 0.10], dtype=np.float32)

    db_path = tmp_path / "bayesian_test.db"
    with sqlite3.connect(db_path) as conn:
        apply_pragmas(conn)
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB)"
        )
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        ensure_schema_tables(conn)
        ensure_schema_views(conn)

        q_vec = np.zeros(300, dtype=np.float32)
        q_vec[0] = 0.90
        chunk_emb = np.zeros(300, dtype=np.float32)
        chunk_emb[0] = 1.0
        emb_blob = serialize_embedding(chunk_emb)

        conn.execute(
            "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding) VALUES ('concepts:entropy', 'doc1.md', 'entropy', 'concepts', 'Information measure', NULL)"
        )
        # chunk:0 manifests concepts:entropy
        conn.execute(
            "INSERT INTO leaf_relations (subject_id, predicate, object_id, source_leaf_id, weight) VALUES ('concepts:entropy', 'manifests_in', 'raw/note.md#chunk:0', 'raw/note.md', 1.0)"
        )
        # chunk:0 has text mention "shannon-entropy" in heading
        conn.execute(
            "INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal, heading, heading_slug, line_start, line_end, embedding) VALUES ('c0', 'raw/note.md#chunk:0', 'raw/note.md', 0, 'Shannon-Entropy Study', 'shannon-entropy-study', 1, 10, ?)",
            (emb_blob,),
        )
        conn.execute(
            "INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal, heading, heading_slug, line_start, line_end, embedding) VALUES ('c1', 'raw/note.md#chunk:1', 'raw/note.md', 1, 'General Overview', 'general-overview', 11, 20, ?)",
            (emb_blob,),
        )
        conn.commit()

        top_chunk_pool = [
            ("raw/note.md#chunk:0", 1, 0.30),
            ("raw/note.md#chunk:1", 2, 0.10),
        ]

        # Score chunk candidates using Bayesian evidence fusion with query mentions
        scored_candidates = retriever._score_chunk_candidates(
            conn=conn,
            top_chunk_pool=top_chunk_pool,
            p_star=p_star,
            q_vec=q_vec,
            top_k=2,
            query_mentions=["shannon-entropy", "entropy"],
        )

        scores = {c["entity_uri"]: c["final_score"] for c in scored_candidates}
        assert "raw/note.md#chunk:0" in scores
        assert "raw/note.md#chunk:1" in scores
        # chunk:0 has concept entity: Prior(0.8) * p*[concepts:entropy](0.50) + p*[chunk:0](0.30) = 0.40 + 0.30 = 0.70
        # Multiplied by vec_sim(0.90) = 0.63
        # chunk:1 has no entities: p*[chunk:1](0.10) * vec_sim(0.90) = 0.09
        assert scores["raw/note.md#chunk:0"] > scores["raw/note.md#chunk:1"]
        assert (
            pytest.approx(scores["raw/note.md#chunk:0"], rel=1e-2)
            == (0.8 * 0.50 + 0.30) * 0.90
        )
        assert pytest.approx(scores["raw/note.md#chunk:1"], rel=1e-2) == 0.10 * 0.90


@pytest.mark.anyio
async def test_canopy_seeding_with_in_memory_kb_priors(tmp_path: Path):
    """Verify canopy seeding blends candidate priors continuously without branch failure."""
    import scipy.sparse as sp

    from yaw.indexer.canonicalize import create_in_memory_kb
    from yaw.retrieval.retriever import GraphRAGRetriever

    retriever = GraphRAGRetriever()
    kb = create_in_memory_kb()
    kb.add_entity(
        entity="concepts:gravity", freq=1, entity_vector=np.zeros(300, dtype=np.float32)
    )
    kb.add_alias(
        alias="gravitation", entities=["concepts:gravity"], probabilities=[0.95]
    )
    retriever.kb = kb

    retriever._node_to_idx = {"concepts:gravity": 0}
    retriever._idx_to_node = ["concepts:gravity"]
    retriever._P = sp.eye(1, format="csr")
    try:
        retriever._cached_db_mtime = config.DB_PATH.stat().st_mtime
    except OSError:
        retriever._cached_db_mtime = 0.0

    db_path = tmp_path / "seeding_test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (entity_uri TEXT, rel_path TEXT, heading TEXT, line_start INT, line_end INT, embedding BLOB)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB)"
        )
        conn.execute(
            "CREATE TABLE aliases (alias_id TEXT, kind TEXT, canonical_id TEXT, method TEXT, score REAL, status TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        # Query mentioning alias 'gravitation'
        p0, _q_vec, _cand_rows = await retriever.compute_seed_vector(
            query="gravitation and curvature",
            conn=conn,
            terms=["gravitation", "curvature"],
        )

        assert p0.shape == (1,)
        # concepts:gravity must be seeded with non-zero probability mass derived from KB alias prior
        assert p0[0] > 0.0
        assert pytest.approx(p0.sum(), rel=1e-5) == 1.0


@pytest.mark.anyio
async def test_daemon_kb_wiring_and_retriever_synchronization(tmp_path: Path):
    """Verify FastMCP daemon initializes persistent KB and syncs across lifecycle."""
    from yaw.server.daemon import (
        compile_wiki,
        get_daemon_kb,
        get_retriever,
        reset_daemon_kb,
    )

    reset_daemon_kb()
    kb1 = get_daemon_kb()
    assert kb1 is not None

    # Singleton persistence
    kb2 = get_daemon_kb()
    assert kb1 is kb2

    # Retriever instance receives KB
    retriever = get_retriever(kb=kb1)
    assert retriever.kb is kb1

    # compile_wiki refreshes cache and preserves KB
    comp = await compile_wiki(vault_dir=str(tmp_path))
    assert comp["status"] == "compiled"
    reset_daemon_kb()


@pytest.mark.anyio
async def test_polysemous_topological_disambiguation_end_to_end(tmp_path: Path):
    """Verify polysemous mentions disambiguate via graph diffusion without runtime LLM calls."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Note 1: Machine Learning Transformer
    ml_path = raw_dir / "ml_transformer.md"
    ml_content = (
        "# Machine Learning Transformer\n"
        "Self-attention mechanism powers neural network transformer architecture for language understanding."
    )
    ml_path.write_text(ml_content, encoding="utf-8")

    # Note 2: Electrical Transformer
    ee_path = raw_dir / "power_transformer.md"
    ee_content = (
        "# Power Transformer\n"
        "Electrical transformers transfer energy between circuits through electromagnetic induction."
    )
    ee_path.write_text(ee_content, encoding="utf-8")

    chunk_ml = "raw/ml_transformer.md:1-2#chunk:0"
    chunk_ee = "raw/power_transformer.md:1-2#chunk:0"
    concept_ml = "concepts:transformer-neural-network"
    concept_ee = "concepts:electrical-transformer"
    concept_attn = "concepts:attention-mechanism"

    from yaw.storage.db import (
        ensure_schema_tables,
        ensure_schema_views,
        insert_alias,
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )
        ensure_schema_tables(conn)

        # Insert chunks
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c_ml', ?, 'raw/ml_transformer.md', 0, 'Machine Learning Transformer', 'ml-transformer', 1, 2, NULL)",
            (chunk_ml,),
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c_ee', ?, 'raw/power_transformer.md', 0, 'Power Transformer', 'power-transformer', 1, 2, NULL)",
            (chunk_ee,),
        )

        # Insert leaf nodes
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/ml_transformer.md', 'Transformer NN', 'concepts', 'Attention-based deep learning architecture.', NULL)",
            (concept_ml,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/power_transformer.md', 'Electrical Transformer', 'concepts', 'Passive electromagnetic induction device.', NULL)",
            (concept_ee,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/ml_transformer.md', 'Attention Mechanism', 'concepts', 'Contextual token weight vector.', NULL)",
            (concept_attn,),
        )

        # Aliases table: 'transformer' has calibrated candidate distribution (0.50 ML, 0.50 EE)
        insert_alias(
            conn,
            "transformer",
            concept_ml,
            kind="entity",
            method="ontology",
            score=0.50,
        )
        insert_alias(
            conn, "attention", concept_attn, kind="entity", method="ontology", score=1.0
        )

        # Relations: ML cluster connects concept_ml with concept_attn and chunk_ml
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'manifests_in', ?, 'raw/ml_transformer.md', 1.0)",
            (concept_ml, chunk_ml),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'manifests_in', ?, 'raw/ml_transformer.md', 1.0)",
            (concept_attn, chunk_ml),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'dependsOn', ?, 'raw/ml_transformer.md', 1.0)",
            (concept_ml, concept_attn),
        )

        # EE cluster connects concept_ee with chunk_ee
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'manifests_in', ?, 'raw/power_transformer.md', 1.0)",
            (concept_ee, chunk_ee),
        )

        # FTS5 indexing
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/ml_transformer.md', 'Machine Learning Transformer', 'Self-attention mechanism powers neural network transformer architecture.')",
            (chunk_ml,),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, 'raw/power_transformer.md', 'Power Transformer', 'Electrical transformers transfer energy between circuits.')",
            (chunk_ee,),
        )
        ensure_schema_views(conn)
        conn.commit()

        # Mount KB with polysemous candidate distribution
        from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

        kb = create_in_memory_kb()
        for cid in (concept_ml, concept_ee, concept_attn):
            kb.add_entity(
                entity=cid, freq=1, entity_vector=np.zeros(300, dtype=np.float32)
            )
        kb.add_alias(
            alias="transformer",
            entities=[concept_ml, concept_ee],
            probabilities=[0.50, 0.50],
        )
        sync_kb_from_db(kb, conn)

    retriever = GraphRAGRetriever(kb=kb)

    with patch("yaw.retrieval.retriever.is_ollama_online", return_value=False):
        # Query combining polysemous 'transformer' with contextual cue 'attention'
        result = await retriever.retrieve("transformer attention", top_k=1)

    assert "# Result: [1]" in result
    assert "## Ontological Context" in result
    # Contextual resonance amplifies the ML transformer concept
    assert "concepts:transformer-neural-network" in result
    assert "## Evidence Chunk" in result
    # ML chunk wins decisively over the electrical transformer chunk
    assert "raw/ml_transformer.md:1-2#chunk:0" in result
    assert "Self-attention mechanism" in result
    evidence_section = result.split("## Evidence Chunk")[1]
    assert "power_transformer.md" not in evidence_section


def test_bayesian_evidence_chunk_scoring_large_candidate_pool_scaling(tmp_path: Path):
    """Attack _score_chunk_candidates with K > 100 candidate chunks and skewed prior distributions."""
    from yaw.indexer.canonicalize import create_in_memory_kb
    from yaw.retrieval.retriever import GraphRAGRetriever
    from yaw.storage.db import (
        apply_pragmas,
        ensure_schema_tables,
        ensure_schema_views,
    )

    retriever = GraphRAGRetriever()
    kb = create_in_memory_kb()
    kb.add_entity(
        entity="concepts:target-concept",
        freq=1,
        entity_vector=np.zeros(300, dtype=np.float32),
    )
    kb.add_alias(
        alias="target-alias", entities=["concepts:target-concept"], probabilities=[0.85]
    )
    retriever.kb = kb

    num_chunks = 120
    node_to_idx = {"concepts:target-concept": 0}
    for i in range(num_chunks):
        node_to_idx[f"raw/note_{i}.md#chunk:0"] = i + 1

    retriever._node_to_idx = node_to_idx
    retriever._idx_to_node = sorted(node_to_idx.keys())

    # Simulated stationary PPR distribution
    p_star = np.zeros(num_chunks + 1, dtype=np.float32)
    p_star[node_to_idx["concepts:target-concept"]] = 0.40
    for i in range(num_chunks):
        p_star[node_to_idx[f"raw/note_{i}.md#chunk:0"]] = 0.01 + (0.001 * i)

    db_path = tmp_path / "large_pool_test.db"
    with sqlite3.connect(db_path) as conn:
        apply_pragmas(conn)
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB)"
        )
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        ensure_schema_tables(conn)
        ensure_schema_views(conn)

        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:target-concept', 'seed.md', 'target', 'concepts', 'Target concept definition', NULL)"
        )

        for i in range(num_chunks):
            uri = f"raw/note_{i}.md#chunk:0"
            conn.execute(
                "INSERT INTO note_chunks VALUES (?, ?, ?, 0, 'Heading', 'heading', 1, 10, NULL)",
                (f"c_{i}", uri, f"raw/note_{i}.md"),
            )
            # Chunks 100-119 manifest the target concept
            if i >= 100:
                conn.execute(
                    "INSERT INTO leaf_relations (subject_id, predicate, object_id, source_leaf_id, weight) VALUES ('concepts:target-concept', 'manifests_in', ?, 'seed.md', 1.0)",
                    (uri,),
                )
        conn.commit()

        top_chunk_pool = [
            (
                f"raw/note_{i}.md#chunk:0",
                node_to_idx[f"raw/note_{i}.md#chunk:0"],
                float(p_star[node_to_idx[f"raw/note_{i}.md#chunk:0"]]),
            )
            for i in range(num_chunks)
        ]

        q_vec = np.zeros(300, dtype=np.float32)  # offline embedding

        scored = retriever._score_chunk_candidates(
            conn=conn,
            top_chunk_pool=top_chunk_pool,
            p_star=p_star,
            q_vec=q_vec,
            top_k=5,
            query_mentions=["target-alias"],
        )

        assert len(scored) == 5
        # Top scoring chunks must be from chunks 100-119 that manifest the target concept with compound Bayesian prior
        top_uris = [c["entity_uri"] for c in scored]
        assert all(int(u.split("_")[1].split(".")[0]) >= 100 for u in top_uris)
        # Winning score equals prior(0.85) * p*[concept](0.40) + p*[chunk]
        assert scored[0]["final_score"] > 0.85 * 0.40


@pytest.mark.anyio
async def test_zero_fallback_kb_initialization_error_early(tmp_path: Path):
    """Verify retriever raises explicit RuntimeError early if KB cannot be initialized, enforcing zero fallbacks."""
    from yaw.retrieval.retriever import GraphRAGRetriever

    retriever = GraphRAGRetriever(kb=None)
    fake_conn = sqlite3.connect(":memory:")
    fake_conn.close()  # closed connection cannot be queried or synchronized

    with pytest.raises(
        RuntimeError, match="Mandatory InMemoryLookupKB could not be initialized"
    ):
        await retriever.compute_seed_vector("query", fake_conn)
