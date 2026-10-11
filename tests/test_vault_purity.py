"""Epic 2 verification: Vault purity, canonical entity URIs, composite PK, and bidirectional graph topology."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from yaw import config
from yaw.indexer.pipeline import sync_vault
from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage.db import ensure_schema_views, get_db

ENGINE_TERMS = ("fastmcp", "sqlite_vec", "pytest", "spacy_llm")
ENGINE_SPEC_FILES = (
    "cocoindex-neo4j.md",
    "cocoindex.md",
    "efficient-ppr-algorithms.md",
    "karpathy-llm-wiki.md",
    "llm-wiki-fts-vector-graph-playbook.md",
    "llm-wiki-spec-addendum.md",
    "llm-wiki-spec-v2.md",
    "ontological-roots.md",
    "power-pagerank.md",
    "scipy-graphrag-extensions.md",
    "spacy-llm-api.md",
    "spacy-llm-docs.md",
    "yaw-architecture-plan.md",
)
CANONICAL_URI_RE = re.compile(r"^[a-z_]+:[a-z0-9_-]+$")
CHUNK_URI_RE = re.compile(r"^.+#chunk:\d+$")


def test_invariant_1_filesystem_purity():
    """Verify sample-llm-wiki/raw has zero engine specs and all 13 specs exist in docs/."""
    vault_root = Path("sample-llm-wiki")
    raw_dir = vault_root / "raw"
    mcp_dir = raw_dir / "llm-wiki-mcp"
    assert not mcp_dir.exists(), "sample-llm-wiki/raw/llm-wiki-mcp must be deleted"

    # Verify no markdown files in raw/ match any engine spec filenames
    raw_md_files = [f.name for f in raw_dir.rglob("*.md")]
    for spec in ENGINE_SPEC_FILES:
        assert spec not in raw_md_files, (
            f"Engine specification {spec} still exists in sample-llm-wiki/raw/"
        )

    # Verify all 13 canonical specifications exist under docs/
    docs_dir = Path("docs")
    for spec in ENGINE_SPEC_FILES:
        spec_path = docs_dir / spec
        assert spec_path.is_file(), (
            f"Engine specification {spec} must exist under docs/"
        )


def test_invariant_2_and_3_vault_rebuild_cleanliness_and_topology():
    """Rebuild sample-llm-wiki with sync_vault and verify zero engine terms, canonical URIs, and edge topology."""
    vault_dir = Path("sample-llm-wiki").resolve()
    sync_vault(vault_dir=vault_dir, force=True, verbose=False)

    db_path = vault_dir / ".cache" / "yaw.db"
    assert db_path.is_file(), f"Database not found at {db_path}"

    with sqlite3.connect(db_path) as conn:
        ensure_schema_views(conn)

        # 1. Zero records matching engine development terms in leaf_nodes
        nodes = conn.execute(
            "SELECT node_id, name, definition FROM leaf_nodes"
        ).fetchall()
        assert len(nodes) > 0, "leaf_nodes must contain indexed domain concepts"

        for node_id, name, definition in nodes:
            combined = f"{node_id} {name} {definition}".lower()
            for term in ENGINE_TERMS:
                assert term not in combined, (
                    f"Engine development term '{term}' found in leaf_nodes: {node_id} ({name})"
                )

        # 2. Canonical regex validation: all node_id match ^[a-z_]+:[a-z0-9_-]+$
        for node_id, _, _ in nodes:
            assert CANONICAL_URI_RE.match(node_id), (
                f"Non-canonical node_id '{node_id}' does not match ^[a-z_]+:[a-z0-9_-]+$"
            )
            assert "#" not in node_id, f"node_id contains path separator '#': {node_id}"

        # 3. Invariant 3: Bidirectional chunk-concept edges in unified_edges
        manifests_in_rows = conn.execute(
            "SELECT src_id, dst_id, weight FROM unified_edges WHERE rel_type = 'manifests_in'"
        ).fetchall()
        assert len(manifests_in_rows) > 0, (
            "unified_edges must contain manifests_in grounding edges"
        )

        for src_id, dst_id, weight in manifests_in_rows:
            assert CANONICAL_URI_RE.match(src_id), (
                f"manifests_in src_id must be canonical concept URI: {src_id}"
            )
            assert CHUNK_URI_RE.match(dst_id), (
                f"manifests_in dst_id must be chunk entity URI: {dst_id}"
            )
            assert abs(weight - 1.0) < 1e-4, (
                f"manifests_in edge weight must be 1.0, got {weight}"
            )

        mentions_rows = conn.execute(
            "SELECT src_id, dst_id, weight FROM unified_edges WHERE rel_type = 'mentions'"
        ).fetchall()
        assert len(mentions_rows) > 0, (
            "unified_edges must contain mentions abstraction edges"
        )

        for src_id, dst_id, weight in mentions_rows:
            assert CHUNK_URI_RE.match(src_id), (
                f"mentions src_id must be chunk entity URI: {src_id}"
            )
            assert CANONICAL_URI_RE.match(dst_id), (
                f"mentions dst_id must be canonical concept URI: {dst_id}"
            )
            assert abs(weight - 0.2) < 1e-4, (
                f"mentions edge weight must be 0.2, got {weight}"
            )


def test_invariant_4_composite_primary_key_multi_document_persistence(tmp_path):
    """Verify composite PRIMARY KEY (node_id, source_leaf_id) allows same concept across notes and unified_nodes deduplicates."""
    db_path = tmp_path / "test_composite.db"
    dummy_vec = np.ones(768, dtype=np.float32).tobytes()

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE leaf_nodes (
                node_id TEXT NOT NULL,
                source_leaf_id TEXT NOT NULL,
                name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                definition TEXT NOT NULL,
                embedding BLOB,
                PRIMARY KEY (node_id, source_leaf_id)
            )
            """)
        conn.execute("""
            CREATE TABLE note_chunks (
                entity_uri TEXT PRIMARY KEY,
                heading TEXT,
                rel_path TEXT
            )
            """)
        ensure_schema_views(conn)

        concept_id = "concepts:transformer"
        # Note 1 defines concept
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/note1.md', 'Transformer', 'concepts', 'Neural network architecture.', ?)",
            (concept_id, dummy_vec),
        )
        # Note 2 mentions same concept without PK collision
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/note2.md', 'Transformer', 'concepts', 'Neural network architecture.', ?)",
            (concept_id, dummy_vec),
        )
        conn.commit()

        # Both records persist in leaf_nodes
        persisted = conn.execute(
            "SELECT node_id, source_leaf_id FROM leaf_nodes WHERE node_id = ?",
            (concept_id,),
        ).fetchall()
        assert len(persisted) == 2
        assert {p[1] for p in persisted} == {"raw/note1.md", "raw/note2.md"}

        # Duplicate (node_id, source_leaf_id) collision MUST raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, 'raw/note1.md', 'Transformer', 'concepts', 'Duplicate note1.', ?)",
                (concept_id, dummy_vec),
            )

        # unified_nodes view yields exactly 1 distinct canonical node
        canonical_nodes = conn.execute(
            "SELECT node_id, name, namespace FROM unified_nodes WHERE node_id = ?",
            (concept_id,),
        ).fetchall()
        assert len(canonical_nodes) == 1, (
            f"unified_nodes must deduplicate concept_id across documents, found {len(canonical_nodes)}"
        )
        assert canonical_nodes[0][0] == concept_id


@pytest.mark.anyio
async def test_invariant_5_sample_vault_dual_epistemic_retrieval():
    """Invariant 5: Verify dual epistemic retrieval contract (ontological context + grounded evidence)

    against real indexed data without tautological cushion mocks or exception masking.
    """
    vault_dir = Path("sample-llm-wiki").resolve()
    db_path = vault_dir / ".cache" / "yaw.db"
    if not db_path.is_file():
        sync_vault(vault_dir=vault_dir, force=False, verbose=False)

    old_root = config.VAULT_ROOT
    try:
        config.set_vault(vault_dir)
        retriever = GraphRAGRetriever()

        # Step 1: Internal contract verification — compute_seed_vector must populate
        # cand_chunks with properly deserialized float32 ndarrays, preventing silent masking.
        conn = get_db(load_vec=True)
        try:
            _p0, _q_vec, cand_chunks = await retriever.compute_seed_vector(
                "embedding gemma model", conn
            )
            assert len(cand_chunks) > 0, (
                "cand_chunks must be non-empty when querying indexed vault; JSON deserialization must not fail"
            )
            for c in cand_chunks:
                emb = c["embedding"]
                assert isinstance(emb, np.ndarray), (
                    f"Chunk embedding must be np.ndarray, got {type(emb)}"
                )
                assert emb.shape == (768,), (
                    f"Chunk embedding shape must be (768,), got {emb.shape}"
                )
                assert emb.dtype == np.float32, (
                    f"Chunk embedding dtype must be float32, got {emb.dtype}"
                )
                assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-2, (
                    f"Chunk embedding must be unit-normalized, got {np.linalg.norm(emb)}"
                )
        finally:
            conn.close()

        # Step 2: End-to-end dual epistemic retrieval contract verification
        result = await retriever.retrieve("embedding gemma model", top_k=2)
        assert isinstance(result, str)
        assert len(result) > 0

        # Layer 1: Ontological Context
        assert "## Ontological Context" in result, (
            "Result must contain '## Ontological Context' section"
        )
        assert "- `" in result, "Ontological Context must display active concepts"
        assert "_No indexed knowledge available._" not in result, (
            "Ontological Context must not report missing knowledge"
        )

        # Layer 2: Grounded Evidence
        assert "## Evidence Chunk" in result, (
            "Result must contain '## Evidence Chunk' section"
        )
        assert "raw/" in result, "Grounded Evidence must cite raw markdown source paths"
        assert "_No ground-truth passages found._" not in result, (
            "Retrieval must not fall back to empty evidence placeholder"
        )
        assert (
            "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold."
            not in result
        ), "Retrieval must not fall back to empty evidence placeholder"

        # Domain term presence: retrieved passages must be relevant to EmbeddingGemma
        result_lower = result.lower()
        assert "gemma" in result_lower or "embedding" in result_lower, (
            "Retrieved passages must contain query-relevant domain content"
        )

    finally:
        config.set_vault(old_root)
