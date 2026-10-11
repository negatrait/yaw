"""Epic 7.6 (Sprint 7F): Compile-Time Closed-World SHACL Validation and Pre-Compiled Lint Caching.

Verifies:
1. compile_wiki runs fast closed-world SHACL (<200ms) and commits violations to SQLite compiled_violations.
2. lint_wiki serves from compiled_violations cache in <10ms with zero pyshacl calls when force=False.
3. lint_wiki scoped query filtering returns localized contradictions via indexed SQL lookup.
4. lint_wiki force=True triggers live re-validation and updates compiled_violations cache.
5. Dynamic ontology ingestion indexes classes into query_fts and leaf_nodes.embedding.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from yaw import config
from yaw.ontology.loader import ingest_ontology
from yaw.ontology.shacl import ValidationReport
from yaw.server.daemon import compile_wiki, lint_wiki
from yaw.storage.db import (
    ensure_schema_views,
    fetch_compiled_violations,
    insert_leaf_node,
)


def _seed_node(
    conn: sqlite3.Connection,
    node_id: str,
    name: str,
    namespace: str,
    definition: str,
    source_leaf_id: str = "raw/note1.md#chunk:0",
) -> None:
    insert_leaf_node(
        conn=conn,
        node_id=node_id,
        name=name,
        namespace=namespace,
        definition=definition,
        source_leaf_id=source_leaf_id,
    )
    ensure_schema_views(conn)


def _seed_relation(
    conn: sqlite3.Connection,
    rel_id: str,
    src_id: str,
    predicate: str,
    dst_id: str,
    weight: float = 1.0,
    source_leaf_id: str = "raw/note1.md#chunk:0",
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO leaf_relations (
            relation_id, source_leaf_id, subject_id, predicate, object_id, weight
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (rel_id, source_leaf_id, src_id, predicate, dst_id, weight),
    )
    ensure_schema_views(conn)


# ==============================================================================
# 1. Compile-Time Fast SHACL Hook & Persistence
# ==============================================================================


@pytest.mark.anyio
async def test_compile_wiki_runs_fast_shacl_and_persists_violations(
    test_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed contradictory relations (A causes B and A prevents B), call compile_wiki(),

    assert compiled_violations contains 2 records and execution completes in <250 ms.
    """
    db_row = test_db.execute("PRAGMA database_list").fetchone()
    db_file = Path(db_row[2])
    monkeypatch.setattr(config, "DB_PATH", db_file)

    _seed_node(
        test_db,
        node_id="claims:alpha",
        name="Claim Alpha",
        namespace="claims",
        definition="Claim Alpha proposition.",
    )
    _seed_node(
        test_db,
        node_id="concepts:beta",
        name="Outcome Beta",
        namespace="concepts",
        definition="Outcome Beta concept.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_cause",
        src_id="claims:alpha",
        predicate="causes",
        dst_id="concepts:beta",
    )
    _seed_relation(
        test_db,
        rel_id="rel_prevent",
        src_id="claims:alpha",
        predicate="prevents",
        dst_id="concepts:beta",
    )
    test_db.commit()

    t0 = time.perf_counter()
    result = await compile_wiki(vault_dir=str(db_file))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert result["status"] == "compiled"
    assert result["violations_count"] == 2
    assert elapsed_ms < 250.0

    violations = fetch_compiled_violations(test_db)
    assert len(violations) == 2

    paths = {v["result_path"] for v in violations}
    assert "http://yaw.local/predicates/causes" in paths
    assert "http://yaw.local/predicates/prevents" in paths
    assert all("alpha" in v["focus_node"] for v in violations)


# ==============================================================================
# 2. Instantaneous O(1) Cache Reads (<10ms, Zero pySHACL Calls)
# ==============================================================================


@pytest.mark.anyio
async def test_lint_wiki_serves_from_sqlite_cache_in_sub_10ms(
    test_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert lint_wiki() queries the SQLite cache in <10ms with zero pyshacl.validate calls when force=False."""
    db_row = test_db.execute("PRAGMA database_list").fetchone()
    db_file = Path(db_row[2])
    monkeypatch.setattr(config, "DB_PATH", db_file)

    _seed_node(
        test_db,
        node_id="claims:alpha",
        name="Alpha",
        namespace="claims",
        definition="Definition.",
    )
    _seed_node(
        test_db,
        node_id="concepts:beta",
        name="Beta",
        namespace="concepts",
        definition="Definition.",
    )
    _seed_relation(test_db, "r1", "claims:alpha", "causes", "concepts:beta")
    _seed_relation(test_db, "r2", "claims:alpha", "prevents", "concepts:beta")
    test_db.commit()

    # Pre-compile to populate cache
    await compile_wiki(vault_dir=str(db_file))

    # Spy on pyshacl.validate to assert ZERO invocations during cached lint_wiki
    import pyshacl

    pyshacl_call_count = 0
    orig_validate = pyshacl.validate

    def _forbidden_validate(*args: Any, **kwargs: Any) -> Any:
        nonlocal pyshacl_call_count
        pyshacl_call_count += 1
        return orig_validate(*args, **kwargs)

    monkeypatch.setattr(pyshacl, "validate", _forbidden_validate)

    # Warm-up pass
    await lint_wiki(vault_dir=str(db_file), force=False)

    t0 = time.perf_counter()
    report: ValidationReport = await lint_wiki(vault_dir=str(db_file), force=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert pyshacl_call_count == 0, (
        f"pyshacl.validate was called {pyshacl_call_count} times during cached lint_wiki!"
    )
    assert elapsed_ms < 10.0, (
        f"Cached lint_wiki took {elapsed_ms:.2f} ms, expected < 10 ms"
    )
    assert len(report) == 2
    assert report.conforms is False
    assert report["status"] == "violations"


# ==============================================================================
# 3. Scoped Query Filtering
# ==============================================================================


@pytest.mark.anyio
async def test_lint_wiki_scoped_query_filtering(
    test_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert lint_wiki(query='target_concept') returns only violations where focus_node or message matches target."""
    db_row = test_db.execute("PRAGMA database_list").fetchone()
    db_file = Path(db_row[2])
    monkeypatch.setattr(config, "DB_PATH", db_file)

    # Insert two distinct contradictions: one on alpha, one on gamma
    _seed_node(
        test_db, "claims:target-concept", "Target", "claims", "Target definition."
    )
    _seed_node(
        test_db, "concepts:outcome-a", "Outcome A", "concepts", "Outcome A definition."
    )
    _seed_relation(
        test_db, "r_t1", "claims:target-concept", "causes", "concepts:outcome-a"
    )
    _seed_relation(
        test_db, "r_t2", "claims:target-concept", "prevents", "concepts:outcome-a"
    )

    _seed_node(test_db, "claims:other-concept", "Other", "claims", "Other definition.")
    _seed_node(
        test_db, "concepts:outcome-b", "Outcome B", "concepts", "Outcome B definition."
    )
    _seed_relation(
        test_db, "r_o1", "claims:other-concept", "causes", "concepts:outcome-b"
    )
    _seed_relation(
        test_db, "r_o2", "claims:other-concept", "prevents", "concepts:outcome-b"
    )
    test_db.commit()

    await compile_wiki(vault_dir=str(db_file))

    # All violations should contain 4 records
    all_report = await lint_wiki(vault_dir=str(db_file))
    assert len(all_report) == 4

    # Scoped query for target-concept should return only the 2 target records
    target_report = await lint_wiki(query="target-concept", vault_dir=str(db_file))
    assert len(target_report) == 2
    assert all("target-concept" in v["focus_node"] for v in target_report)
    assert not any("other-concept" in v["focus_node"] for v in target_report)

    # Query for nonexistent concept should return clean report
    empty_report = await lint_wiki(query="nonexistent_slug", vault_dir=str(db_file))
    assert len(empty_report) == 0
    assert empty_report.conforms is True
    assert empty_report["status"] == "clean"


# ==============================================================================
# 4. Live Re-validation on Force=True
# ==============================================================================


@pytest.mark.anyio
async def test_lint_wiki_force_revalidation_updates_cache(
    test_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert lint_wiki(force=True) triggers live validation and refreshes the cache."""
    db_row = test_db.execute("PRAGMA database_list").fetchone()
    db_file = Path(db_row[2])
    monkeypatch.setattr(config, "DB_PATH", db_file)

    # Seed contradictory relations
    _seed_node(test_db, "claims:node1", "N1", "claims", "N1 def.")
    _seed_node(test_db, "concepts:node2", "N2", "concepts", "N2 def.")
    _seed_relation(test_db, "rel1", "claims:node1", "causes", "concepts:node2")
    _seed_relation(test_db, "rel2", "claims:node1", "prevents", "concepts:node2")
    test_db.commit()

    # Pre-compile
    await compile_wiki(vault_dir=str(db_file))
    initial_violations = fetch_compiled_violations(test_db)
    assert len(initial_violations) == 2

    # Resolve contradiction by deleting the preventing edge
    test_db.execute("DELETE FROM leaf_relations WHERE relation_id = 'rel2'")
    ensure_schema_views(test_db)
    test_db.commit()

    # Default cached lint_wiki still reflects stale cache (2 violations) until compilation or force
    stale_report = await lint_wiki(vault_dir=str(db_file), force=False)
    assert len(stale_report) == 2

    # Force re-validation bypasses cache and updates SQLite compiled_violations
    fresh_report = await lint_wiki(vault_dir=str(db_file), force=True, fast=True)
    assert len(fresh_report) == 0
    assert fresh_report.conforms is True

    # Cache in SQLite is now updated to clean
    updated_cached = fetch_compiled_violations(test_db)
    assert len(updated_cached) == 0


# ==============================================================================
# 5. Dynamic Ontology Ingestion Indexes FTS5 & Vectors
# ==============================================================================


def test_dynamic_ontology_ingestion_inserts_fts_and_vectors(
    test_db: sqlite3.Connection,
) -> None:
    """Assert ingest_ontology indexes classes into query_fts and leaf_nodes.embedding."""
    sample_ttl = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix bio: <http://yaw.local/bio/> .

    <http://yaw.local/bio/> a owl:Ontology .

    bio:Organism a owl:Class ;
        rdfs:label "Organism"@en ;
        skos:definition "An individual living entity exhibiting the properties of life."@en .

    bio:Plant a owl:Class ;
        rdfs:label "Plant"@en ;
        rdfs:subClassOf bio:Organism ;
        skos:definition "A multicellular eukaryotic organism of the kingdom Plantae."@en .
    """

    res = ingest_ontology(source=sample_ttl, conn=test_db, ontology_id="bio")
    assert res["status"] == "success"
    assert res["classes_ingested"] >= 2

    # Verify query_fts contains indexed classes (contentless FTS5 indexes heading/content)
    organism_hits = test_db.execute(
        "SELECT entity_uri FROM query_fts WHERE query_fts MATCH 'Organism'"
    ).fetchall()
    assert len(organism_hits) >= 1
    assert any("Organism" in r[0] for r in organism_hits)

    plant_hits = test_db.execute(
        "SELECT entity_uri FROM query_fts WHERE query_fts MATCH 'Plant'"
    ).fetchall()
    assert len(plant_hits) >= 1
    assert any("Plant" in r[0] for r in plant_hits)

    fts_rows = test_db.execute(
        "SELECT entity_uri FROM query_fts WHERE entity_uri LIKE 'bio:%'"
    ).fetchall()
    assert len(fts_rows) >= 2

    # Verify leaf_nodes contains stored vector embeddings
    leaf_rows = test_db.execute(
        "SELECT node_id, name, embedding FROM leaf_nodes WHERE node_id LIKE 'bio:%'"
    ).fetchall()
    assert len(leaf_rows) >= 2
    for row in leaf_rows:
        assert row[2] is not None, (
            f"Node {row[0]} is missing vector embedding BLOB in leaf_nodes"
        )
        assert len(row[2]) > 0
