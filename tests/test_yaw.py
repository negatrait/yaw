"""Tests for yaw.py FastMCP server, find_seed tool, and summed probability retrieval pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastmcp import Client

from yaw import config
from yaw.indexer.pipeline import (
    _node_id,
    _relation_id,
    build_entity_uri,
    clean_edge_target,
    extract_title,
)
from yaw.nlp.client import call_ollama_json, is_ollama_online
from yaw.nlp.embeddings import (
    compute_similarity_matrix,
    format_document,
    format_prompt,
    format_proposition,
)
from yaw.server.daemon import compute_seed_candidates, hydrate_disk_slice, mcp


@pytest.mark.anyio
async def test_mcp_server_metadata_and_tools():
    """Verify FastMCP server name, instructions, and exposed tool definitions."""
    assert mcp.name == "yaw-mcp"
    assert (
        mcp.instructions
        == "Local Ontology GraphRAG continuous associative search & seed generation engine."
    )
    tool = await mcp.get_tool("find_seed")
    assert tool is not None
    assert "query" in tool.parameters["properties"]
    assert "top_k" in tool.parameters["properties"]

    qw_tool = await mcp.get_tool("query_wiki")
    assert qw_tool is not None
    assert "query" in qw_tool.parameters["properties"]
    assert "top_k" in qw_tool.parameters["properties"]

    rd_tool = await mcp.get_tool("read_document")
    assert rd_tool is not None
    assert "path" in rd_tool.parameters["properties"]
    assert "byte_offset" in rd_tool.parameters["properties"]
    assert "byte_length" in rd_tool.parameters["properties"]

    os_tool = await mcp.get_tool("get_ontology_schema")
    assert os_tool is not None


@pytest.mark.anyio
async def test_find_seed_fastmcp_client_integration():
    """Verify find_seed invocation via FastMCP in-memory Client pointing to sample-llm-wiki."""
    vault_path = Path(__file__).resolve().parent.parent / "sample-llm-wiki"
    config.set_vault(vault_path)
    with patch(
        "yaw.server.daemon.compute_seed_candidates", new_callable=AsyncMock
    ) as mock_csc:
        mock_csc.return_value = [
            {
                "entity_uri": "raw/test-material/test-document.md#chunk:0",
                "kind": "chunk",
                "rel_path": "raw/test-material/test-document.md",
                "heading": "Testing Intent",
                "byte_offset": 0,
                "byte_length": 58,
                "hybrid_prob": 0.85,
                "rerank_score": 0.92,
                "text": "Allways test the real source code, never do soft mocking.",
            }
        ]
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            assert "find_seed" in tool_names
            assert "query_wiki" in tool_names
            assert "read_document" in tool_names
            assert "get_ontology_schema" in tool_names

            res = await client.call_tool(
                "find_seed",
                {"query": "testing source code", "top_k": 1},
            )
            assert res.data is not None
            assert isinstance(res.data, list)
            assert len(res.data) == 1
            seed = res.data[0]
            assert isinstance(seed, dict)

            # Contract: Exact 9 keys
            expected_keys = {
                "entity_uri",
                "kind",
                "rel_path",
                "heading",
                "byte_offset",
                "byte_length",
                "hybrid_prob",
                "rerank_score",
                "text",
            }
            assert set(seed.keys()) == expected_keys

            # Exact field values and types
            assert seed["entity_uri"] == "raw/test-material/test-document.md#chunk:0"
            assert seed["kind"] == "chunk"
            assert seed["rel_path"] == "raw/test-material/test-document.md"
            assert seed["heading"] == "Testing Intent"
            assert isinstance(seed["byte_offset"], int)
            assert seed["byte_offset"] == 0
            assert isinstance(seed["byte_length"], int)
            assert seed["byte_length"] == 58
            assert isinstance(seed["hybrid_prob"], float)
            assert seed["hybrid_prob"] == 0.85
            assert isinstance(seed["rerank_score"], float)
            assert seed["rerank_score"] == 0.92
            assert (
                seed["text"]
                == "Allways test the real source code, never do soft mocking."
            )

            # Test get_ontology_schema
            res_schema = await client.call_tool("get_ontology_schema", {})
            assert res_schema.data is not None
            assert "namespaces" in str(res_schema.data)

            # Test read_document
            res_read = await client.call_tool(
                "read_document",
                {
                    "path": "raw/test-material/test-document.md",
                    "byte_offset": 0,
                    "byte_length": 15,
                },
            )
            assert res_read.data is not None

            # Test query_wiki
            with patch(
                "yaw.retrieval.retriever.GraphRAGRetriever.retrieve",
                new_callable=AsyncMock,
            ) as mock_ret:
                mock_ret.return_value = (
                    "# Ontological Context\n- Active Concepts: `concepts:testing`\n\n"
                    "# Retrieved Ground-Truth Evidence\n### 1. Testing Intent\n"
                )
                res_qw = await client.call_tool(
                    "query_wiki",
                    {"query": "testing intent", "top_k": 1},
                )
                assert res_qw.data is not None
                assert "Ontological Context" in res_qw.data


@pytest.mark.anyio
async def test_find_seed_pipeline_math_and_summed_probability(tmp_path):
    """Verify FTS + Vector sum-to-1 normalization, summed probability, and disk hydration."""
    config.set_vault(tmp_path)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note1 = raw_dir / "note1.md"
    note1_text = "# Physics\nWerner Heisenberg formulated the uncertainty principle in quantum mechanics."
    note1.write_text(note1_text, encoding="utf-8")

    note2 = raw_dir / "note2.md"
    note2_text = (
        "# Biology\nDNA structure was discovered by Watson and Crick in Cambridge."
    )
    note2.write_text(note2_text, encoding="utf-8")

    db_path = cache_dir / "vectors.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_meta (rel_path TEXT PRIMARY KEY, title TEXT, category TEXT, modified_at REAL)"
        )
        conn.execute(
            "INSERT INTO note_meta VALUES ('raw/note1.md', 'Physics', 'raw', 1.0)"
        )
        conn.execute(
            "INSERT INTO note_meta VALUES ('raw/note2.md', 'Biology', 'raw', 1.0)"
        )

        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        b1 = note1_text.encode("utf-8")
        b2 = note2_text.encode("utf-8")
        dummy_vec1 = np.ones(768, dtype=np.float32).tobytes()
        dummy_vec2 = np.zeros(768, dtype=np.float32).tobytes()

        conn.execute(
            "INSERT INTO note_chunks VALUES ('c1', 'raw/note1.md#chunk:0', 'raw/note1.md', 0, 'Physics', 'physics', 0, ?, 1, 2, ?)",
            (len(b1), dummy_vec1),
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c2', 'raw/note2.md#chunk:0', 'raw/note2.md', 0, 'Biology', 'biology', 0, ?, 1, 2, ?)",
            (len(b2), dummy_vec2),
        )

        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('raw/note1.md#chunk:0', 'raw/note1.md', 'Physics', ?)",
            (note1_text,),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('raw/note2.md#chunk:0', 'raw/note2.md', 'Biology', ?)",
            (note2_text,),
        )
        conn.commit()

    with (
        patch("yaw.retrieval.retriever._query_vectors") as mock_qvc,
        patch("yaw.nlp.embeddings.embed_query", new_callable=AsyncMock) as mock_eq,
        patch("yaw.nlp.embeddings.embed_texts", new_callable=AsyncMock) as mock_embed,
    ):
        mock_eq.return_value = np.array([1.0, 0.0], dtype=np.float32)
        mock_qvc.return_value = [
            {
                "entity_uri": "raw/note1.md#chunk:0",
                "rel_path": "raw/note1.md",
                "heading": "Physics",
                "byte_offset": 0,
                "byte_length": len(b1),
                "sim": 0.90,
            },
            {
                "entity_uri": "raw/note2.md#chunk:0",
                "rel_path": "raw/note2.md",
                "heading": "Biology",
                "byte_offset": 0,
                "byte_length": len(b2),
                "sim": 0.10,
            },
        ]
        mock_embed.return_value = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        seeds = await compute_seed_candidates("Heisenberg uncertainty", top_k=2)
        assert len(seeds) == 2

        s1 = seeds[0]
        assert s1["entity_uri"] == "raw/note1.md#chunk:0"
        assert s1["text"] == note1_text
        assert pytest.approx(s1["hybrid_prob"], rel=1e-2) == 1.9


@pytest.mark.anyio
async def test_unified_vector_and_taxonomy_candidate_retrieval(tmp_path):
    """Verify that compute_seed_candidates searches and hydrates BOTH note_chunks AND leaf_nodes."""
    config.set_vault(tmp_path)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_file = raw_dir / "quantum.md"
    note_text = "# Quantum Mechanics\nWerner Heisenberg developed matrix mechanics."
    note_file.write_text(note_text, encoding="utf-8")
    b_text = note_text.encode("utf-8")

    db_path = cache_dir / "vectors.db"
    dummy_vec = np.ones(768, dtype=np.float32).tobytes()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, byte_offset INTEGER, byte_length INTEGER, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c1', 'raw/quantum.md#chunk:0', 'raw/quantum.md', 0, 'Quantum', 'quantum', 0, ?, 1, 2, ?)",
            (len(b_text), dummy_vec),
        )

        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concepts:matrix-mechanics', 'raw/quantum.md', 'Matrix Mechanics', 'concepts', 'Formulation of quantum theory using matrices.', ?)",
            (dummy_vec,),
        )

        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('raw/quantum.md#chunk:0', 'raw/quantum.md', 'Quantum', ?)",
            (note_text,),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('concepts:matrix-mechanics', 'raw/quantum.md', 'Matrix Mechanics', 'Matrix Mechanics: Formulation of quantum theory using matrices.')",
        )
        conn.commit()

    with (
        patch("yaw.nlp.embeddings.embed_query", new_callable=AsyncMock) as mock_eq,
        patch("yaw.nlp.embeddings.embed_texts", new_callable=AsyncMock) as mock_embed,
    ):
        mock_eq.return_value = np.ones(768, dtype=np.float32)
        mock_embed.return_value = [
            np.ones(768, dtype=np.float32),
            np.ones(768, dtype=np.float32),
        ]

        seeds = await compute_seed_candidates("matrix mechanics", top_k=2)
        assert len(seeds) == 2

        kinds = {s["kind"] for s in seeds}
        assert "chunk" in kinds
        assert "taxonomy" in kinds

        tax_seed = next(s for s in seeds if s["kind"] == "taxonomy")
        assert tax_seed["entity_uri"] == "concepts:matrix-mechanics"
        assert "Matrix Mechanics (concepts)" in tax_seed["heading"]
        assert "Formulation of quantum theory" in tax_seed["text"]

        chunk_seed = next(s for s in seeds if s["kind"] == "chunk")
        assert chunk_seed["entity_uri"] == "raw/quantum.md#chunk:0"
        assert chunk_seed["text"] == note_text


def test_hydrate_disk_slice_edge_cases(tmp_path):
    """Verify hydrate_disk_slice handles missing files, empty reads, and valid byte offsets."""
    old_root = config.VAULT_ROOT
    try:
        config.set_vault(tmp_path)
        # Missing file
        assert hydrate_disk_slice("missing.md", 0, 100) == ""

        # Valid file
        test_file = tmp_path / "sample.md"
        content = "Hello, zero-copy GraphRAG!"
        test_file.write_text(content, encoding="utf-8")

        # Read slice
        slice_text = hydrate_disk_slice("sample.md", 7, 9)
        assert slice_text == "zero-copy"

        # Out of bounds / seek beyond EOF
        assert hydrate_disk_slice("sample.md", 1000, 10) == ""
    finally:
        config.set_vault(old_root)


def test_nlp_client_online_check_and_json_call():
    """Verify is_ollama_online and call_ollama_json behavior with mock endpoints."""
    import urllib.error

    # Test online returns bool
    online = is_ollama_online("http://127.0.0.1:9")
    assert online is False

    # Test call_ollama_json raises URLError on offline host
    with pytest.raises((urllib.error.URLError, RuntimeError)):
        call_ollama_json("/api/version", host="http://127.0.0.1:9", timeout_s=0.1)


def test_embeddings_formatting_and_proposition():
    """Verify prompt formatting, document formatting, and proposition assembly."""
    prop = format_proposition("Albert Einstein", "developed", "General Relativity")
    assert prop == "Albert Einstein developed General Relativity."

    doc_none = format_document("Sample body text", title=None)
    assert doc_none == "title: none | text: Sample body text"

    doc_titled = format_document("Sample body text", title="Physics Notes")
    assert doc_titled == "title: Physics Notes | text: Sample body text"

    prompt_ret = format_prompt("quantum decoherence", "Retrieval-query")
    assert prompt_ret == "task: search result | query: quantum decoherence"

    prompt_custom = format_prompt("test", "UnknownTask")
    assert prompt_custom == "test"


def test_compute_similarity_matrix():
    """Verify compute_similarity_matrix produces correct dot product gram matrix."""
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    mat = compute_similarity_matrix([v1, v2])
    assert mat.shape == (2, 2)
    assert float(mat[0, 0]) == 1.0
    assert float(mat[0, 1]) == 0.0
    assert float(mat[1, 0]) == 0.0
    assert float(mat[1, 1]) == 1.0

    empty_mat = compute_similarity_matrix([])
    assert empty_mat.shape == (0, 0)


def test_pipeline_uri_and_slug_utilities():
    """Verify build_entity_uri, extract_title, clean_edge_target, _node_id, and _relation_id."""
    uri = build_entity_uri("raw/test.md", 3)
    assert uri == "raw/test.md#chunk:3"

    title = extract_title("heisenberg.md", "# Heisenberg Uncertainty\nBody text")
    assert title == "Heisenberg Uncertainty"

    fallback_title = extract_title("sub/my-note.md", "No header text here")
    assert fallback_title == "My Note"

    target = clean_edge_target("raw/note1.md", "note2", {"raw/note2.md"})
    assert target == "raw/note2.md"
    assert clean_edge_target("raw/note1.md", "note1", {"raw/note1.md"}) is None
    assert (
        clean_edge_target("raw/note1.md", "https://example.com", {"raw/note2.md"})
        is None
    )

    nid = _node_id("General Relativity", "theories")
    assert nid == "theories:general-relativity"

    rel_id = _relation_id(
        "raw/physics.md",
        "General Relativity",
        "published_in",
        "Annalen der Physik",
    )
    assert rel_id == "raw/physics.md#general-relativity:published-in:annalen-der-physik"
