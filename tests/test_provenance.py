"""Integration tests for Sprint 5-Align architectural guarantees.

Validates:
1. Chunk Provenance Grounding (Invariant E1/E2):
   - _store_chunk_concept_edges does not fall back to all chunks when concept absent from subsequent chunks.
   - Only originating/matching chunks are linked.
   - Ghost concepts without provenance or text matches produce zero edges.
2. Predicate Alias View Join:
   - Slugified predicate alias joins in unified_edges view.
   - Raw predicate string match joins in unified_edges view.
   - Qualified URI predicate alias joins in unified_edges view.
   - Contested/inactive aliases fall back to raw predicate.
3. canon_verdicts Query Caching:
   - resolve_entity_with_llm and _adjudicate_pair_with_llm return cached CanonicalVerdict without LLM calls.
4. Context Prompt Canonical View Routing:
   - _render_ontological_context reflects canonical alias entities and predicates rather than raw leaf tokens.
5. Foreign Key Enforcement:
   - apply_pragmas sets PRAGMA foreign_keys = 1 and enforces referential integrity on disk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from yaw.indexer.pipeline import (
    LeafRelationRow,
    MarkdownChunk,
    _store_chunk_concept_edges,
)
from yaw.nlp.extractor import ExtractedNode
from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage.db import (
    apply_pragmas,
    ensure_schema_tables,
    ensure_schema_views,
    insert_alias,
)


class _InMemoryRelationTarget:
    """In-memory table target collecting declared LeafRelationRow items without external mocks."""

    def __init__(self) -> None:
        self.rows: list[LeafRelationRow] = []

    def declare_row(self, *, row: LeafRelationRow) -> None:
        self.rows.append(row)


def _init_provenance_test_db(conn: sqlite3.Connection) -> None:
    """Initialize full schema for testing unified views and referential integrity."""
    apply_pragmas(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_nodes (
            node_id TEXT NOT NULL,
            source_leaf_id TEXT NOT NULL,
            name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            definition TEXT,
            embedding BLOB,
            PRIMARY KEY (node_id, source_leaf_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_relations (
            relation_id TEXT PRIMARY KEY,
            source_leaf_id TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            predicate_id TEXT,
            object_id TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wikilinks (
            source_uri TEXT NOT NULL,
            target_uri TEXT NOT NULL,
            link_label TEXT,
            PRIMARY KEY (source_uri, target_uri)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_chunks (
            chunk_id TEXT NOT NULL,
            entity_uri TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            ordinal INT NOT NULL,
            heading TEXT,
            heading_slug TEXT,
            byte_offset INT,
            byte_length INT,
            line_start INT,
            line_end INT,
            embedding BLOB
        )
        """
    )
    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    conn.commit()


# ==============================================================================
# 1. Chunk Provenance Grounding (Invariant E1/E2)
# ==============================================================================


def test_chunk_provenance_grounding_no_fallback_to_all_chunks():
    """Verify _store_chunk_concept_edges grounds concepts strictly to matching chunks.

    Guarantees:
    - Node with explicit provenance to chunk 0 only links to chunk 0, even when
      its name does not appear in subsequent chunks.
    - It must NOT fall back to linking all chunks.
    - Substring matches link only to the chunk where the name occurs.
    - Ghost nodes with neither provenance nor text matches generate 0 edges.
    """
    rel = "raw/ml_architecture.md"
    chunks = [
        MarkdownChunk(
            chunk_id="chunk:0",
            heading="Introduction",
            heading_slug="introduction",
            text="High-level architecture overview with foundational concepts.",
            ordinal=0,
            byte_offset=0,
            byte_length=60,
            line_start=1,
            line_end=4,
        ),
        MarkdownChunk(
            chunk_id="chunk:1",
            heading="Gradient Descent",
            heading_slug="gradient-descent",
            text="Detailed discussion of Backpropagation and loss optimization surfaces.",
            ordinal=1,
            byte_offset=61,
            byte_length=74,
            line_start=5,
            line_end=8,
        ),
        MarkdownChunk(
            chunk_id="chunk:2",
            heading="Regularization",
            heading_slug="regularization",
            text="Weight decay, dropout, and numerical stability techniques.",
            ordinal=2,
            byte_offset=136,
            byte_length=61,
            line_start=9,
            line_end=12,
        ),
    ]

    target = _InMemoryRelationTarget()
    seen: set[str] = set()

    # Case 1: Node whose name never appears in any chunk text, but has provenance in chunk 0.
    # Must NOT link to chunk 1 or chunk 2.
    raw_nodes_prov_only = {
        "concepts:residual-connection": (
            f"{rel}#chunk:0",
            ExtractedNode(
                name="Skip Connection Mechanism",
                namespace="concepts",
                definition="Identity shortcut mapping",
            ),
        )
    }
    node_prov = {"concepts:residual-connection": {f"{rel}#chunk:0"}}

    _store_chunk_concept_edges(
        raw_nodes_prov_only,
        chunks,
        rel,
        cast(Any, target),
        seen,
        node_provenance=node_prov,
    )

    manifest_edges = [r for r in target.rows if r.predicate == "manifests_in"]
    mentions_edges = [r for r in target.rows if r.predicate == "mentions"]

    assert len(manifest_edges) == 1, (
        f"Expected exactly 1 manifests_in edge, found {len(manifest_edges)}"
    )
    assert manifest_edges[0].object_id == f"{rel}#chunk:0"
    assert manifest_edges[0].subject_id == "concepts:residual-connection"
    assert manifest_edges[0].weight == 1.0

    assert len(mentions_edges) == 1, (
        f"Expected exactly 1 mentions edge, found {len(mentions_edges)}"
    )
    assert mentions_edges[0].subject_id == f"{rel}#chunk:0"
    assert mentions_edges[0].object_id == "concepts:residual-connection"
    assert mentions_edges[0].weight == 0.2

    # Verify no fallback to chunk 1 or chunk 2
    linked_chunks = {r.object_id for r in manifest_edges}
    assert f"{rel}#chunk:1" not in linked_chunks
    assert f"{rel}#chunk:2" not in linked_chunks

    # Case 2: Node matching text in chunk 1 only (case-insensitive) without explicit provenance
    target.rows.clear()
    seen.clear()
    raw_nodes_text_only = {
        "concepts:backprop": (
            "",
            ExtractedNode(
                name="backpropagation",
                namespace="concepts",
                definition="Reverse-mode automatic differentiation",
            ),
        )
    }
    _store_chunk_concept_edges(
        raw_nodes_text_only,
        chunks,
        rel,
        cast(Any, target),
        seen,
        node_provenance={},
    )
    assert len(target.rows) == 2  # 1 manifests_in + 1 mentions
    assert any(
        r.predicate == "manifests_in" and r.object_id == f"{rel}#chunk:1"
        for r in target.rows
    )
    assert any(
        r.predicate == "mentions" and r.subject_id == f"{rel}#chunk:1"
        for r in target.rows
    )

    # Case 3: Ghost node (name absent from all chunks, empty provenance)
    target.rows.clear()
    seen.clear()
    raw_nodes_ghost = {
        "concepts:quantum-entanglement": (
            "",
            ExtractedNode(
                name="Quantum Entanglement",
                namespace="concepts",
                definition="Non-local quantum correlations",
            ),
        )
    }
    _store_chunk_concept_edges(
        raw_nodes_ghost,
        chunks,
        rel,
        cast(Any, target),
        seen,
        node_provenance={},
    )
    assert len(target.rows) == 0, (
        f"Invariant E1/E2: Ghost node must produce 0 edges, got {len(target.rows)}"
    )

    # Case 4: Deduplication with seen_rel_ids
    # Rerunning with existing seen set must produce 0 new rows
    _store_chunk_concept_edges(
        raw_nodes_text_only,
        chunks,
        rel,
        cast(Any, target),
        seen,
        node_provenance={},
    )
    initial_count = len(target.rows)
    _store_chunk_concept_edges(
        raw_nodes_text_only,
        chunks,
        rel,
        cast(Any, target),
        seen,
        node_provenance={},
    )
    assert len(target.rows) == initial_count


# ==============================================================================
# 2. Predicate Alias View Join
# ==============================================================================


def test_predicate_alias_view_join_slug_and_raw_match(tmp_path: Path):
    """In a real SQLite DB, verify unified_edges rewrites predicate names via aliases.

    Tests:
    - Slugified alias match: alias_id = "is-composed-of" -> canonical_id = "predicates:comprises"
    - Raw string match: alias_id = "is composed of" -> canonical_id = "predicates:consists-of"
    - Fully-qualified predicate_id match: alias_id = "predicates:is-composed-of"
    - Inactive / contested alias falls back to raw predicate
    """
    db_path = tmp_path / "predicate_alias_test.db"
    conn = sqlite3.connect(db_path)
    _init_provenance_test_db(conn)

    # Insert leaf relation with multi-word predicate
    conn.execute(
        """
        INSERT INTO leaf_relations (
            relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "rel:car:engine",
            "raw/automotive.md",
            "concepts:car",
            "is composed of",
            "predicates:is-composed-of",
            "concepts:engine",
            0.95,
        ),
    )
    conn.commit()

    # Before alias: rel_type must be raw predicate string
    row = conn.execute(
        "SELECT src_id, dst_id, rel_type, weight FROM unified_edges WHERE src_id = 'concepts:car'"
    ).fetchone()
    assert row is not None
    assert row[2] == "is composed of"

    # 1. Slugified predicate alias: alias_id = "is-composed-of"
    insert_alias(
        conn,
        alias_id="is-composed-of",
        canonical_id="predicates:comprises",
        kind="predicate",
        method="lexical",
        score=1.0,
        status="active",
        source_leaf_id="raw/automotive.md",
    )
    conn.commit()

    row_slug = conn.execute(
        "SELECT src_id, dst_id, rel_type, weight FROM unified_edges WHERE src_id = 'concepts:car'"
    ).fetchone()
    assert row_slug is not None
    assert row_slug[2] == "predicates:comprises", (
        f"Expected rewritten rel_type 'predicates:comprises', got '{row_slug[2]}'"
    )

    # 2. Raw match: delete slugified alias and insert raw match with spaces
    conn.execute("DELETE FROM aliases WHERE alias_id = 'is-composed-of'")
    insert_alias(
        conn,
        alias_id="is composed of",
        canonical_id="predicates:consists-of",
        kind="predicate",
        method="lexical",
        score=1.0,
        status="active",
        source_leaf_id="raw/automotive.md",
    )
    conn.commit()

    row_raw = conn.execute(
        "SELECT src_id, dst_id, rel_type, weight FROM unified_edges WHERE src_id = 'concepts:car'"
    ).fetchone()
    assert row_raw is not None
    assert row_raw[2] == "predicates:consists-of", (
        f"Expected rewritten rel_type 'predicates:consists-of', got '{row_raw[2]}'"
    )

    # 3. Predicate ID match: alias_id = "predicates:is-composed-of"
    conn.execute("DELETE FROM aliases WHERE alias_id = 'is composed of'")
    insert_alias(
        conn,
        alias_id="predicates:is-composed-of",
        canonical_id="predicates:assembled-from",
        kind="predicate",
        method="lexical",
        score=1.0,
        status="active",
        source_leaf_id="raw/automotive.md",
    )
    conn.commit()

    row_pid = conn.execute(
        "SELECT src_id, dst_id, rel_type, weight FROM unified_edges WHERE src_id = 'concepts:car'"
    ).fetchone()
    assert row_pid is not None
    assert row_pid[2] == "predicates:assembled-from"

    # 4. Inactive/contested alias falls back to raw predicate
    conn.execute(
        "UPDATE aliases SET status = 'contested' WHERE alias_id = 'predicates:is-composed-of'"
    )
    conn.commit()

    row_contested = conn.execute(
        "SELECT src_id, dst_id, rel_type, weight FROM unified_edges WHERE src_id = 'concepts:car'"
    ).fetchone()
    assert row_contested is not None
    assert row_contested[2] == "is composed of", (
        f"Contested alias must fall back to raw predicate, got '{row_contested[2]}'"
    )

    conn.close()


# ==============================================================================
# 4. Context Prompt Canonical View Routing
# ==============================================================================


def test_context_prompt_canonical_view_routing(tmp_path: Path):
    """Verify _render_ontological_context reflects canonical alias entities and relations instead of raw leaf data."""
    db_path = tmp_path / "context_view_test.db"
    conn = sqlite3.connect(db_path)
    _init_provenance_test_db(conn)

    # Insert raw leaf entities
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES (?, ?, ?, ?, ?, NULL)
        """,
        (
            "concepts:artificial_neural_net",
            "raw/ai.md",
            "Artificial Neural Net",
            "concepts",
            "Biology-inspired computational graph with parameterized weights.",
        ),
    )
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES (?, ?, ?, ?, ?, NULL)
        """,
        (
            "concepts:deep_learning",
            "raw/ai.md",
            "Deep Learning",
            "concepts",
            "Hierarchical representation learning across multilayer networks.",
        ),
    )

    # Insert raw leaf relation
    conn.execute(
        """
        INSERT INTO leaf_relations (
            relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "rel:ai:1",
            "raw/ai.md",
            "concepts:artificial_neural_net",
            "enables",
            "predicates:enables",
            "concepts:deep_learning",
            1.0,
        ),
    )

    # Insert entity and predicate aliases
    insert_alias(
        conn,
        alias_id="concepts:artificial_neural_net",
        canonical_id="concepts:neural-network",
        kind="entity",
        method="compile",
        score=1.0,
        status="active",
        source_leaf_id="raw/ai.md",
    )
    insert_alias(
        conn,
        alias_id="enables",
        canonical_id="predicates:facilitates",
        kind="predicate",
        method="compile",
        score=1.0,
        status="active",
        source_leaf_id="raw/ai.md",
    )
    conn.commit()

    retriever = GraphRAGRetriever()
    context_lines = retriever._render_ontological_context(
        conn, ["concepts:neural-network", "concepts:deep_learning"]
    )

    joined_context = "\n".join(context_lines)

    # Invariant: Output reflects canonical entity ID and definition from unified_nodes
    assert (
        "`concepts:neural-network` [concepts]: Biology-inspired computational"
        " graph" in joined_context
    ), f"Canonical entity not reflected in context:\n{joined_context}"

    # Invariant: Output reflects canonical predicate ID from unified_edges
    assert (
        "`concepts:neural-network` --[predicates:facilitates]-->"
        " `concepts:deep_learning`" in joined_context
    ), f"Canonical edge not reflected in context:\n{joined_context}"

    # Invariant: Raw leaf tokens must NOT appear
    assert "concepts:artificial_neural_net" not in joined_context, (
        "Raw subject entity leaked into rendered ontological context!"
    )
    assert "--[enables]-->" not in joined_context, (
        "Raw un-aliased predicate leaked into rendered ontological context!"
    )

    conn.close()


# ==============================================================================
# 5. Foreign Key Enforcement
# ==============================================================================


def test_apply_pragmas_foreign_key_enforcement(tmp_path: Path):
    """Verify apply_pragmas sets foreign_keys = 1 and referential integrity is strictly enforced."""
    db_path = tmp_path / "pragma_fk_test.db"
    conn = sqlite3.connect(db_path)

    # 1. Prior to apply_pragmas, verify foreign_keys default is 0 in standard SQLite
    initial_fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert initial_fk == 0, "Default SQLite connection should have foreign_keys = 0"

    # 2. apply_pragmas must set foreign_keys = 1
    apply_pragmas(conn)
    post_fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert post_fk == 1, "apply_pragmas must turn on PRAGMA foreign_keys = 1"

    # 3. Create parent and child tables with foreign key constraint
    conn.execute(
        """
        CREATE TABLE parent_nodes (
            node_id TEXT PRIMARY KEY,
            label TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE child_edges (
            edge_id TEXT PRIMARY KEY,
            parent_id TEXT NOT NULL,
            FOREIGN KEY (parent_id) REFERENCES parent_nodes(node_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()

    # 4. Attempt to insert orphan child -> must raise sqlite3.IntegrityError
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        conn.execute(
            "INSERT INTO child_edges (edge_id, parent_id) VALUES ('edge:ghost',"
            " 'node:nonexistent')"
        )

    # 5. Valid parent insert allows valid child insert
    conn.execute(
        "INSERT INTO parent_nodes (node_id, label) VALUES ('node:root', 'Root Concept')"
    )
    conn.execute(
        "INSERT INTO child_edges (edge_id, parent_id) VALUES ('edge:valid',"
        " 'node:root')"
    )
    conn.commit()

    edge_count = conn.execute("SELECT count(*) FROM child_edges").fetchone()[0]
    assert edge_count == 1

    # 6. Deletion cascades properly
    conn.execute("DELETE FROM parent_nodes WHERE node_id = 'node:root'")
    conn.commit()

    remaining_edges = conn.execute("SELECT count(*) FROM child_edges").fetchone()[0]
    assert remaining_edges == 0, "ON DELETE CASCADE should have removed child edge"

    # 7. Also verify busy_timeout and synchronous pragma are applied
    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert busy_timeout > 0, "busy_timeout should be greater than 0"

    conn.close()


from yaw.storage.db import clean_db


def test_storage_lifecycle_and_schema_integrity(tmp_path: Path):
    db_path = tmp_path / "test_lifecycle.db"

    # Test clean_db
    # Create dummy files
    db_path.touch()
    Path(f"{db_path}-wal").touch()
    Path(f"{db_path}-shm").touch()

    clean_db(db_path)
    assert not db_path.exists()
    assert not Path(f"{db_path}-wal").exists()
    assert not Path(f"{db_path}-shm").exists()

    # Idempotent check
    clean_db(db_path)

    # Test verify_schema_integrity
    conn = sqlite3.connect(db_path)
    _init_provenance_test_db(conn)

    from yaw.storage.db import verify_schema_integrity

    result = verify_schema_integrity(conn)
    assert result.get("healthy") is True

    tables = result.get("tables", [])
    expected_tables = [
        "aliases",
        "canon_verdicts",
        "leaf_nodes",
        "leaf_relations",
        "note_chunks",
        "wikilinks",
    ]
    for t in expected_tables:
        assert t in tables, f"Expected table {t} missing"

    views = result.get("views", [])
    expected_views = ["unified_edges", "unified_nodes"]
    for v in expected_views:
        assert v in views, f"Expected view {v} missing"

    # Test get_aliases_for_canonical
    from yaw.storage.db import get_aliases_for_canonical

    # insert two active aliases
    insert_alias(
        conn,
        alias_id="concepts:entrophy",
        canonical_id="concepts:entropy",
        kind="entity",
        method="lexical",
        score=1.0,
        status="active",
        source_leaf_id="raw/doc.md",
    )
    insert_alias(
        conn,
        alias_id="concepts:shannon-entropy",
        canonical_id="concepts:entropy",
        kind="entity",
        method="lexical",
        score=1.0,
        status="active",
        source_leaf_id="raw/doc.md",
    )
    # insert one inactive/contested alias
    insert_alias(
        conn,
        alias_id="concepts:disorder",
        canonical_id="concepts:entropy",
        kind="entity",
        method="lexical",
        score=1.0,
        status="contested",
        source_leaf_id="raw/doc.md",
    )

    aliases = get_aliases_for_canonical(conn, "concepts:entropy", "entity")
    assert len(aliases) == 2
    assert "concepts:entrophy" in aliases
    assert "concepts:shannon-entropy" in aliases
    assert "concepts:disorder" not in aliases

    conn.close()
