"""Milestone 1 (Phase 2) Adversarial Challenge Test Suite.

Adversarially probes:
1. Dynamic ontology ingestion idempotency (3x sequential ingestion, query_fts constant, entity_vec constant, leaf_nodes 0).
2. entity_vec and fact_vec dimensional enforcement (300d rejected with OperationalError, 768d clean insertion).
3. SHACL ambiguous identity detection (1 target clean, >= 2 unlinked targets trigger sh:Warning, linked targets clean).
"""

from __future__ import annotations

import sqlite3
import struct
import time

import pytest

from yaw.ontology.loader import ingest_ontology
from yaw.ontology.shacl import validate_wiki_graph, validate_wiki_graph_fast
from yaw.storage.db import (
    ensure_schema_views,
    insert_close_match,
    insert_entity_vec,
    insert_fact_vec,
    insert_leaf_node,
)

SAMPLE_BIO_TTL = """
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

bio:hasCellType a owl:ObjectProperty ;
    rdfs:label "has cell type"@en ;
    rdfs:comment "Relates an organism to its cellular composition."@en .
"""

SAMPLE_DICO_TTL = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dico: <http://yaw.local/dico/> .

<http://yaw.local/dico/> a owl:Ontology .

dico:BuildingElement a owl:Class ;
    rdfs:label "Building Element"@en ;
    skos:definition "A constituent part of a building."@en .

dico:Wall a owl:Class ;
    rdfs:subClassOf dico:BuildingElement ;
    rdfs:label "Wall"@en ;
    skos:definition "A vertical structure defining an area."@en .
"""


# ==============================================================================
# Challenge 1: Dynamic Ontology Ingestion Idempotency & Separation
# ==============================================================================


def test_dynamic_ontology_ingestion_idempotency_3x(test_db: sqlite3.Connection) -> None:
    """Ingest the same ontology 3 times in a row into SQLite.

    Assert:
    - query_fts row count remains constant across all 3 passes.
    - entity_vec retains exactly the right class count (2 classes).
    - leaf_nodes has 0 rows (Invariant 1: dynamic ontologies never pollute raw note extractions).
    - dynamic_classes has exactly 2 rows (no duplicated rows).
    """
    counts_history: list[dict[str, int]] = []

    for pass_idx in range(1, 4):
        res = ingest_ontology(source=SAMPLE_BIO_TTL, conn=test_db, ontology_id="bio")
        assert res["status"] == "success", f"Ingestion pass {pass_idx} failed"
        assert res["classes_ingested"] == 2

        fts_cnt = test_db.execute(
            "SELECT COUNT(*) FROM query_fts WHERE rel_path = ?",
            ("dynamic_ontology:bio",),
        ).fetchone()[0]

        vec_cnt = test_db.execute(
            "SELECT COUNT(*) FROM entity_vec WHERE partition_key = ?",
            ("bio",),
        ).fetchone()[0]

        leaf_cnt = test_db.execute(
            "SELECT COUNT(*) FROM leaf_nodes WHERE node_id LIKE 'bio:%'"
        ).fetchone()[0]

        class_cnt = test_db.execute(
            "SELECT COUNT(*) FROM dynamic_classes WHERE ontology_id = 'bio'"
        ).fetchone()[0]

        counts_history.append(
            {
                "pass": pass_idx,
                "fts": fts_cnt,
                "vec": vec_cnt,
                "leaf": leaf_cnt,
                "classes": class_cnt,
            }
        )

    # Invariants verification
    first = counts_history[0]
    assert first["fts"] == 3, (
        f"Expected 3 FTS rows (2 classes + 1 prop), got {first['fts']}"
    )
    assert first["vec"] == 2, f"Expected 2 entity_vec rows, got {first['vec']}"
    assert first["leaf"] == 0, f"Expected 0 leaf_nodes rows, got {first['leaf']}"
    assert first["classes"] == 2, (
        f"Expected 2 dynamic_classes rows, got {first['classes']}"
    )

    for entry in counts_history[1:]:
        assert entry["fts"] == first["fts"], (
            f"FTS count mutated on pass {entry['pass']}: {entry['fts']} vs {first['fts']}"
        )
        assert entry["vec"] == first["vec"], (
            f"entity_vec count mutated on pass {entry['pass']}: {entry['vec']} vs {first['vec']}"
        )
        assert entry["leaf"] == 0, (
            f"leaf_nodes non-zero on pass {entry['pass']}: {entry['leaf']}"
        )
        assert entry["classes"] == first["classes"], (
            f"dynamic_classes count mutated on pass {entry['pass']}: {entry['classes']} vs {first['classes']}"
        )


def test_interleaved_multi_ontology_ingestion_isolation(
    test_db: sqlite3.Connection,
) -> None:
    """Interleaved ingestion of different ontologies does not corrupt partitions or FTS."""
    # 1. Ingest bio
    ingest_ontology(source=SAMPLE_BIO_TTL, conn=test_db, ontology_id="bio")
    # 2. Ingest dico
    ingest_ontology(source=SAMPLE_DICO_TTL, conn=test_db, ontology_id="dico")
    # 3. Re-ingest bio
    ingest_ontology(source=SAMPLE_BIO_TTL, conn=test_db, ontology_id="bio")

    # Verify partition keys in entity_vec
    bio_vec = test_db.execute(
        "SELECT COUNT(*) FROM entity_vec WHERE partition_key = 'bio'"
    ).fetchone()[0]
    dico_vec = test_db.execute(
        "SELECT COUNT(*) FROM entity_vec WHERE partition_key = 'dico'"
    ).fetchone()[0]
    assert bio_vec == 2
    assert dico_vec == 2

    # Verify query_fts isolation
    bio_fts = test_db.execute(
        "SELECT COUNT(*) FROM query_fts WHERE rel_path = 'dynamic_ontology:bio'"
    ).fetchone()[0]
    dico_fts = test_db.execute(
        "SELECT COUNT(*) FROM query_fts WHERE rel_path = 'dynamic_ontology:dico'"
    ).fetchone()[0]
    assert bio_fts == 3
    assert dico_fts == 2

    # Total leaf_nodes remains 0
    bio_leaf = test_db.execute(
        "SELECT COUNT(*) FROM leaf_nodes WHERE node_id LIKE 'bio:%' OR node_id LIKE 'dico:%'"
    ).fetchone()[0]
    assert bio_leaf == 0


# ==============================================================================
# Challenge 2: entity_vec & fact_vec Dimensional Enforcement
# ==============================================================================


def test_entity_vec_and_fact_vec_dimensional_enforcement(
    test_db: sqlite3.Connection,
) -> None:
    """entity_vec and fact_vec reject non-768d vectors with sqlite3.OperationalError."""
    # 300d vectors (e.g. legacy spaCy or glove vectors) MUST be rejected by sqlite-vec
    vec_300d = struct.pack("300f", *([0.05] * 300))

    with pytest.raises(sqlite3.OperationalError):
        insert_entity_vec(test_db, "entities:test-300d", "entities", vec_300d)

    with pytest.raises(sqlite3.OperationalError):
        insert_fact_vec(test_db, "rel:test-300d", vec_300d)

    # Other arbitrary non-768 dimensions MUST also be rejected
    for bad_dim in (64, 128, 384, 512, 1024, 1536):
        bad_vec = struct.pack(f"{bad_dim}f", *([0.01] * bad_dim))
        with pytest.raises(sqlite3.OperationalError):
            insert_entity_vec(test_db, f"entities:test-{bad_dim}d", "entities", bad_vec)
        with pytest.raises(sqlite3.OperationalError):
            insert_fact_vec(test_db, f"rel:test-{bad_dim}d", bad_vec)

    # 768d vector MUST insert cleanly into both tables
    vec_768d = struct.pack("768f", *([0.01] * 768))
    insert_entity_vec(test_db, "entities:test-768d", "entities", vec_768d)
    insert_fact_vec(test_db, "rel:test-768d", vec_768d)

    # Verify retrieval from entity_vec
    row_ent = test_db.execute(
        "SELECT entity_id, partition_key, length(embedding) FROM entity_vec WHERE entity_id = 'entities:test-768d'"
    ).fetchone()
    assert row_ent is not None
    assert row_ent[0] == "entities:test-768d"
    assert row_ent[1] == "entities"
    assert row_ent[2] == 768 * 4  # 768 float32 values = 3072 bytes

    # Verify retrieval from fact_vec
    row_fact = test_db.execute(
        "SELECT relation_id, length(embedding) FROM fact_vec WHERE relation_id = 'rel:test-768d'"
    ).fetchone()
    assert row_fact is not None
    assert row_fact[0] == "rel:test-768d"
    assert row_fact[1] == 768 * 4


# ==============================================================================
# Challenge 3: SHACL Ambiguous Identity Detection (AmbiguousIdentityShape)
# ==============================================================================


def _seed_node(
    conn: sqlite3.Connection, node_id: str, name: str, definition: str
) -> None:
    insert_leaf_node(
        conn=conn,
        node_id=node_id,
        name=name,
        namespace=node_id.split(":")[0],
        definition=definition,
        source_leaf_id="raw/note.md#chunk:0",
    )


def test_shacl_single_close_match_conforms(test_db: sqlite3.Connection) -> None:
    """Assert that 1 closeMatch target does NOT trigger AmbiguousIdentity warning."""
    _seed_node(test_db, "entities:A", "Entity A", "Definition for entity A.")
    _seed_node(test_db, "entities:B", "Entity B", "Definition for entity B.")
    insert_close_match(test_db, "entities:A", "entities:B", 0.88)
    ensure_schema_views(test_db)

    # 1. Full validate_wiki_graph
    _conforms, violations = validate_wiki_graph(test_db)
    ambig_violations = [
        v
        for v in violations
        if "ambiguous identity" in v.get("message", "").lower()
        or "close matches" in v.get("message", "").lower()
        or "AmbiguousIdentityShape" in v.get("shape_id", "")
    ]
    assert len(ambig_violations) == 0, (
        f"Unexpected ambiguous identity warning for 1 target: {ambig_violations}"
    )

    # 2. Fast validate_wiki_graph_fast
    _fast_conforms, fast_report = validate_wiki_graph_fast(test_db)
    fast_ambig = [
        v
        for v in fast_report
        if "ambiguous identity" in v.get("message", "").lower()
        or "close matches" in v.get("message", "").lower()
        or "AmbiguousIdentityShape" in v.get("shape_id", "")
    ]
    assert len(fast_ambig) == 0, (
        f"Fast SHACL emitted unexpected ambiguous warning for 1 target: {fast_ambig}"
    )


def test_shacl_multiple_unlinked_close_matches_trigger_warning(
    test_db: sqlite3.Connection,
) -> None:
    """Assert that >= 2 unlinked closeMatch targets DO trigger sh:Warning with ambiguous identity message."""
    _seed_node(test_db, "entities:focus", "Focus Node", "Definition for focus entity.")
    _seed_node(test_db, "entities:target1", "Target One", "Definition for target 1.")
    _seed_node(test_db, "entities:target2", "Target Two", "Definition for target 2.")

    # focus has closeMatch to target1 and target2, but target1 and target2 are UNLINKED
    insert_close_match(test_db, "entities:focus", "entities:target1", 0.86)
    insert_close_match(test_db, "entities:focus", "entities:target2", 0.84)
    ensure_schema_views(test_db)

    # 1. Full validate_wiki_graph
    _conforms, violations = validate_wiki_graph(test_db)
    ambig_violations = [
        v
        for v in violations
        if "ambiguous identity" in v.get("message", "").lower()
        or "close matches" in v.get("message", "").lower()
        or "AmbiguousIdentityShape" in v.get("shape_id", "")
    ]
    assert len(ambig_violations) >= 1, (
        f"Expected ambiguous identity warning for >= 2 unlinked targets, got: {violations}"
    )
    for v in ambig_violations:
        assert v["severity"] == "Warning", (
            f"Ambiguous identity must be sh:Warning, got {v['severity']}"
        )
        assert (
            "ambiguous identity" in v["message"].lower()
            or "close matches" in v["message"].lower()
        )
        assert "focus" in v["focus_node"] or "focus" in v.get("value", "")

    # 2. Fast validate_wiki_graph_fast
    start_t = time.perf_counter()
    _fast_conforms, fast_report = validate_wiki_graph_fast(test_db)
    fast_elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    fast_ambig = [
        v
        for v in fast_report
        if "ambiguous identity" in v.get("message", "").lower()
        or "close matches" in v.get("message", "").lower()
        or "AmbiguousIdentityShape" in v.get("shape_id", "")
    ]
    assert len(fast_ambig) >= 1, (
        f"Fast SHACL failed to detect ambiguous identity: {fast_report}"
    )
    assert fast_ambig[0]["severity"] == "Warning"
    assert fast_elapsed_ms < 250.0, (
        f"Fast validation exceeded latency budget: {fast_elapsed_ms:.2f} ms"
    )


def test_shacl_triangular_linked_close_matches_do_not_warn(
    test_db: sqlite3.Connection,
) -> None:
    """Assert that >= 2 closeMatch targets that ARE linked to each other (triangular clique) do NOT trigger warning."""
    _seed_node(test_db, "entities:A", "Node A", "Definition A.")
    _seed_node(test_db, "entities:B", "Node B", "Definition B.")
    _seed_node(test_db, "entities:C", "Node C", "Definition C.")

    # A <-> B, A <-> C, AND B <-> C (consistent equivalence cluster)
    insert_close_match(test_db, "entities:A", "entities:B", 0.88)
    insert_close_match(test_db, "entities:A", "entities:C", 0.85)
    insert_close_match(test_db, "entities:B", "entities:C", 0.87)
    ensure_schema_views(test_db)

    _conforms, violations = validate_wiki_graph(test_db)
    ambig_violations = [
        v
        for v in violations
        if "ambiguous identity" in v.get("message", "").lower()
        or "close matches" in v.get("message", "").lower()
        or "AmbiguousIdentityShape" in v.get("shape_id", "")
    ]
    assert len(ambig_violations) == 0, (
        f"Triangular linked targets should not trigger ambiguous warning, but got: {ambig_violations}"
    )


def test_shacl_mixed_intra_vault_and_cross_scheme_close_matches(
    test_db: sqlite3.Connection,
) -> None:
    """Assert that mixed yaw:closeMatch and skos:closeMatch unlinked targets trigger warning."""
    _seed_node(test_db, "entities:internal", "Internal Node", "Internal definition.")
    _seed_node(test_db, "entities:peer", "Peer Internal", "Peer definition.")
    _seed_node(
        test_db, "external:foreign", "External Concept", "External concept definition."
    )

    # internal <-> peer (yaw:closeMatch) and internal <-> foreign (skos:closeMatch)
    insert_close_match(test_db, "entities:internal", "entities:peer", 0.85)
    insert_close_match(test_db, "entities:internal", "external:foreign", 0.82)
    ensure_schema_views(test_db)

    _conforms, violations = validate_wiki_graph(test_db)
    ambig_violations = [
        v
        for v in violations
        if "ambiguous identity" in v.get("message", "").lower()
        or "close matches" in v.get("message", "").lower()
        or "AmbiguousIdentityShape" in v.get("shape_id", "")
    ]
    assert len(ambig_violations) >= 1, (
        f"Mixed unlinked targets should trigger ambiguous identity warning, got: {violations}"
    )
    assert any(v["severity"] == "Warning" for v in ambig_violations)
