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
    db_path = tmp_path / ".cache" / "vectors.db"
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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
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
    db_path = tmp_path / ".cache" / "vectors.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "heisenberg.md"
    note_content = "# Heisenberg\nWerner Heisenberg discovered the uncertainty principle in quantum physics."
    note_path.write_text(note_content, encoding="utf-8")
    b_content = note_content.encode("utf-8")

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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        chunk_uri = "raw/heisenberg.md#chunk:0"
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/heisenberg.md', 0, 'Heisenberg', 'heisenberg', 0, ?, 1, 2, ?)",
            (chunk_uri, len(b_content), dummy_vec.tobytes()),
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
    b_content = note_content.encode("utf-8")

    config.set_vault(vault_dir)

    db_path = cache_dir / "vectors.db"
    dummy_vec = np.ones(768, dtype=np.float32)
    norm = float(np.linalg.norm(dummy_vec))
    dummy_vec /= norm
    json_vec_str = json.dumps(dummy_vec.tolist())

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
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
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/planck.md', 0, 'Planck', 'planck', 0, ?, 1, 2, ?)",
            (chunk_uri, len(b_content), json_vec_str),
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
    db_path = tmp_path / ".cache" / "vectors.db"
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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
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
    db_path = tmp_path / ".cache" / "vectors.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "test.md"
    note_content = "# Test Document\nThis note contains search query content that should be retrieved offline."
    note_path.write_text(note_content, encoding="utf-8")
    b_content = note_content.encode("utf-8")

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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/test.md', 0, 'Test Document', 'test-document', 0, ?, 1, 2, NULL)",
            (chunk_uri, len(b_content)),
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
    db_path = tmp_path / ".cache" / "vectors.db"
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
    b_content = note_content.encode("utf-8")

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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/chloroplast.md', 0, 'Cellular Energy', 'cellular-energy', 0, ?, 1, 2, NULL)",
            (chunk_uri, len(b_content)),
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
    db_path = tmp_path / ".cache" / "vectors.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "relativity.md"
    note_content = (
        "# Special Relativity\n"
        "Albert Einstein formulated special relativity describing inertial reference frames."
    )
    note_path.write_text(note_content, encoding="utf-8")
    b_content = note_content.encode("utf-8")

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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c0', ?, 'raw/relativity.md', 0, 'Special Relativity', 'special-relativity', 0, ?, 1, 2, NULL)",
            (chunk_uri, len(b_content)),
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
    db_path = tmp_path / ".cache" / "vectors.db"
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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
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
    db_path = tmp_path / ".cache" / "vectors.db"
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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
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
    db_path = tmp_path / ".cache" / "vectors.db"
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
    db_path = tmp_path / ".cache" / "vectors.db"
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
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
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
