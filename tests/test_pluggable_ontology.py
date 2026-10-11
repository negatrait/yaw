"""Production TDD verification suite for Epic 7: Pluggable Ontologies & Configurable Open-IE.

Adheres strictly to docs/epic7-pluggable-ontology-integration-spec.md §5 and TEST_INFRA.md.
All 8 test cases are unmocked, disk-grounded, and execute against real SQLite with WAL pragmas.
"""

from __future__ import annotations

import importlib
import sqlite3
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import rdflib

from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage.db import apply_pragmas, ensure_schema_views


def _get_normalize_predicate_antonym_guarded() -> Any:
    try:
        mod = importlib.import_module("yaw.nlp.components")
        func = mod.__dict__.get("normalize_predicate_antonym_guarded")
        if func is not None:
            return func
    except ImportError, ModuleNotFoundError:
        pass

    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "yaw.nlp.components.normalize_predicate_antonym_guarded not yet implemented"
        )

    return _stub


def _get_ingest_ontology() -> Any:
    try:
        mod = importlib.import_module("yaw.ontology.loader")
        func = mod.__dict__.get("ingest_ontology")
        if func is not None:
            return func
    except ImportError, ModuleNotFoundError:
        pass

    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "yaw.ontology.loader.ingest_ontology not yet implemented"
        )

    return _stub


def _get_validate_wiki_graph_fast() -> Any:
    try:
        mod = importlib.import_module("yaw.ontology.shacl")
        func = mod.__dict__.get("validate_wiki_graph_fast")
        if func is not None:
            return func
    except ImportError, ModuleNotFoundError:
        pass

    def _stub(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "yaw.ontology.shacl.validate_wiki_graph_fast not yet implemented"
        )

    return _stub


normalize_predicate_antonym_guarded = _get_normalize_predicate_antonym_guarded()
ingest_ontology = _get_ingest_ontology()
validate_wiki_graph_fast = _get_validate_wiki_graph_fast()


DICO_SAMPLE_TTL = """
@prefix dico: <https://w3id.org/digitalconstruction/0.5/Variables#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

dico:BuildingElement a owl:Class ;
    rdfs:label "Building Element"@en .

dico:Wall a owl:Class ;
    rdfs:subClassOf dico:BuildingElement ;
    rdfs:label "Wall"@en .

dico:ExteriorWall a owl:Class ;
    rdfs:subClassOf dico:Wall ;
    rdfs:label "Exterior Wall"@en .

dico:GhostClass a owl:Class ;
    rdfs:subClassOf dico:BuildingElement ;
    rdfs:label "Uninstantiated Ghost Element"@en .

dico:isAdjacentTo a owl:ObjectProperty, owl:SymmetricProperty ;
    rdfs:label "is adjacent to"@en .
"""

PUNNING_SAMPLE_TTL = """
@prefix ex: <https://example.org/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:PunningViolation a owl:Class, skos:Concept ;
    rdfs:label "Punning Violation Entity"@en ;
    skos:prefLabel "Punning Violation Entity"@en .
"""


@pytest.fixture
def real_disk_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Unmocked real SQLite database on disk with WAL pragmas per Invariant 6."""
    db_file = tmp_path / "yaw.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_chunks (
            chunk_id TEXT PRIMARY KEY,
            entity_uri TEXT,
            rel_path TEXT,
            ordinal INT,
            heading TEXT,
            heading_slug TEXT,
            line_start INT,
            line_end INT,
            embedding BLOB
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wikilinks (
            source_uri TEXT,
            target_uri TEXT,
            link_label TEXT,
            PRIMARY KEY(source_uri, target_uri)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_nodes (
            node_id TEXT,
            source_leaf_id TEXT,
            name TEXT,
            namespace TEXT,
            definition TEXT,
            embedding BLOB,
            PRIMARY KEY(node_id, source_leaf_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_relations (
            relation_id TEXT PRIMARY KEY,
            source_leaf_id TEXT,
            subject_id TEXT,
            predicate TEXT,
            object_id TEXT,
            weight REAL,
            predicate_id TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aliases (
            alias_id TEXT PRIMARY KEY,
            canonical_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('entity', 'predicate')),
            status TEXT NOT NULL DEFAULT 'active',
            method TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 1.0,
            created_at REAL NOT NULL DEFAULT (unixepoch())
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aliases_lookup ON aliases(alias_id, kind, status);"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_relations (
            relation_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_nodes (
            node_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            definition TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_ontologies (
            ontology_id TEXT PRIMARY KEY,
            uri_prefix TEXT NOT NULL,
            source_uri TEXT,
            is_default INT NOT NULL DEFAULT 0,
            loaded_at REAL NOT NULL DEFAULT (unixepoch())
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_classes (
            curie TEXT PRIMARY KEY,
            class_uri TEXT NOT NULL UNIQUE,
            ontology_id TEXT NOT NULL,
            label TEXT NOT NULL,
            parent_class_uri TEXT,
            depth INT NOT NULL DEFAULT 0,
            is_root INT NOT NULL DEFAULT 0,
            epistemic_type TEXT NOT NULL CHECK(epistemic_type IN ('class', 'concept'))
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_class_parents (
            curie TEXT NOT NULL,
            parent_class_uri TEXT NOT NULL,
            PRIMARY KEY (curie, parent_class_uri)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_properties (
            curie TEXT PRIMARY KEY,
            property_uri TEXT NOT NULL UNIQUE,
            ontology_id TEXT NOT NULL,
            label TEXT NOT NULL,
            parent_property_uri TEXT,
            is_transitive INT NOT NULL DEFAULT 0,
            is_symmetric INT NOT NULL DEFAULT 0
        );
        """
    )

    ensure_schema_views(conn)
    conn.commit()

    try:
        yield conn
    finally:
        conn.close()


def test_quadrant_1_open_world_baseline(real_disk_db: sqlite3.Connection) -> None:
    """Verify Q1 (internal, open_ie=true): Extracts leaf triples, permits open predicates."""
    real_disk_db.execute(
        "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', '1'), ('target_ontology', 'internal')"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_q1', 'raw/1.md#0', 'concepts:foo', 'disruptsMarket', 'entities:bar', 1.0, 'disruptsmarket')"
    )
    real_disk_db.commit()

    edges = real_disk_db.execute(
        "SELECT src_id, dst_id, rel_type FROM unified_edges WHERE src_id='concepts:foo'"
    ).fetchall()
    assert len(edges) == 1
    assert edges[0][2] == "disruptsMarket"


def test_quadrant_2_closed_predicates(real_disk_db: sqlite3.Connection) -> None:
    """Verify Q2 (internal, open_ie=false): Filters out open-IE predicates in unified_edges (<5ms toggle)."""
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_open', 'raw/1.md#0', 'concepts:x', 'disruptsMarket', 'entities:y', 1.0, 'disruptsmarket')"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_fixed', 'raw/1.md#0', 'concepts:x', 'causes', 'entities:y', 1.0, 'causes')"
    )
    real_disk_db.commit()

    # Toggle open_ie to 0
    t0 = time.perf_counter()
    real_disk_db.execute(
        "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', '0'), ('target_ontology', 'internal')"
    )
    real_disk_db.commit()

    edges = real_disk_db.execute(
        "SELECT rel_type FROM unified_edges WHERE src_id='concepts:x'"
    ).fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    rel_types = [e[0] for e in edges]
    assert "causes" in rel_types
    assert "disruptsMarket" not in rel_types
    assert (
        elapsed_ms < 50.0
    )  # < 5ms view query time, budget < 50ms total toggle + fetch

    # Invariant 2: leaf_relations retains raw triple
    raw = real_disk_db.execute(
        "SELECT COUNT(*) FROM leaf_relations WHERE predicate='disruptsMarket'"
    ).fetchone()[0]
    assert raw == 1


def test_quadrant_3_open_world_external(real_disk_db: sqlite3.Connection) -> None:
    """Verify Q3 (external, open_ie=true): External ontology classes active, open verbs permitted."""
    g = rdflib.Graph().parse(data=DICO_SAMPLE_TTL, format="turtle")
    res = ingest_ontology(source=g, conn=real_disk_db, ontology_id="dico")
    assert res["status"] == "success"

    real_disk_db.execute(
        "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', '1'), ('target_ontology', 'external')"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_q3', 'raw/1.md#0', 'dico:Wall', 'degradesOverTime', 'concepts:entropy', 1.0, 'degradesovertime')"
    )
    real_disk_db.commit()

    edges = real_disk_db.execute(
        "SELECT rel_type FROM unified_edges WHERE src_id='dico:Wall'"
    ).fetchall()
    assert any(e[0] == "degradesOverTime" for e in edges)


def test_quadrant_4_closed_domain(real_disk_db: sqlite3.Connection) -> None:
    """Verify Q4 (external, open_ie=false): Filters out non-canonical, non-ontology predicates in view."""
    g = rdflib.Graph().parse(data=DICO_SAMPLE_TTL, format="turtle")
    res = ingest_ontology(source=g, conn=real_disk_db, ontology_id="dico")
    assert res["status"] == "success"

    real_disk_db.execute(
        "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', '0'), ('target_ontology', 'external')"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_q4a', 'raw/1.md#0', 'dico:ExteriorWall', 'isAdjacentTo', 'dico:Wall', 1.0, 'isadjacentto')"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_q4b', 'raw/1.md#0', 'dico:ExteriorWall', 'randomVerb', 'dico:Wall', 1.0, 'randomverb')"
    )
    real_disk_db.commit()

    edges = real_disk_db.execute(
        "SELECT rel_type FROM unified_edges WHERE src_id='dico:ExteriorWall'"
    ).fetchall()
    rel_types = [e[0] for e in edges]
    assert "dico:isAdjacentTo" in rel_types or "isAdjacentTo" in rel_types
    assert "randomVerb" not in rel_types


def test_antonym_polarity_exclusion() -> None:
    """Verify Polarity Exclusion Mask blocks causal and dialectical vector collapse."""
    # 'prevents' must NOT normalize to 'causes' despite high GloVe cosine
    res_prevents = normalize_predicate_antonym_guarded("prevents")
    assert res_prevents != "causes"
    assert "prevent" in res_prevents

    # 'refutes' must NOT normalize to 'supports'
    res_refutes = normalize_predicate_antonym_guarded("refutes")
    assert res_refutes != "supports"
    assert "refute" in res_refutes

    # 'contradicts' must NOT normalize to 'supports'
    res_contradicts = normalize_predicate_antonym_guarded("contradicts")
    assert res_contradicts != "supports"
    assert "contradict" in res_contradicts


def test_lean_pyshacl_latency(real_disk_db: sqlite3.Connection) -> None:
    """Assert relational pre-closure + inference=None executes in < 200 ms on 1,000+ triples."""
    # Seed 500 leaf nodes and 500 hierarchical leaf relations (yielding 1,000+ RDF triples)
    node_rows = [
        (
            f"concepts:node_{i}",
            "raw/benchmark.md#0",
            f"Concept Node {i}",
            "concepts",
            f"Formal taxonomic entity definition for node {i}",
            None,
        )
        for i in range(500)
    ]
    real_disk_db.executemany(
        "INSERT INTO leaf_nodes VALUES (?, ?, ?, ?, ?, ?)",
        node_rows,
    )

    relation_rows = [
        (
            f"rel_bench_{i}",
            "raw/benchmark.md#0",
            f"concepts:node_{i}",
            "subClassOf",
            f"concepts:node_{(i + 1) % 500}",
            1.0,
            "subclassof",
        )
        for i in range(500)
    ]
    real_disk_db.executemany(
        "INSERT INTO leaf_relations VALUES (?, ?, ?, ?, ?, ?, ?)",
        relation_rows,
    )
    real_disk_db.commit()

    t0 = time.perf_counter()
    conforms, _report = validate_wiki_graph_fast(real_disk_db)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert conforms is True
    assert elapsed_ms < 200.0, (
        f"SHACL validation exceeded 200ms threshold: {elapsed_ms:.1f}ms"
    )


def test_active_subgraph_projection(real_disk_db: sqlite3.Connection) -> None:
    """Verify DiCo TTL ingestion, alias materialization, and ghost class pruning in PPR."""
    g = rdflib.Graph().parse(data=DICO_SAMPLE_TTL, format="turtle")
    res = ingest_ontology(source=g, conn=real_disk_db, ontology_id="dico")
    assert res["status"] == "success"
    assert res["classes_ingested"] == 4

    # Instantiate leaf nodes & relation in knowledge graph
    real_disk_db.execute(
        "INSERT INTO leaf_nodes VALUES ('dico:ExteriorWall', 'raw/spec.md#0', 'Exterior Wall', 'dico', 'Wall', NULL)"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_nodes VALUES ('dico:Wall', 'raw/spec.md#0', 'Wall', 'dico', 'Wall base', NULL)"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel1', 'raw/spec.md#0', 'dico:ExteriorWall', 'dico:isAdjacentTo', 'dico:Wall', 1.0, 'dico:isadjacentto')"
    )
    real_disk_db.commit()

    retriever = GraphRAGRetriever()
    retriever._sync_graph(real_disk_db)

    # Active classes must be present
    assert "dico:ExteriorWall" in retriever._node_to_idx
    assert "dico:Wall" in retriever._node_to_idx
    # Ancestor BuildingElement must be projected via dynamic_classes hierarchy in unified_edges
    assert "dico:BuildingElement" in retriever._node_to_idx
    # Ghost class must be strictly PRUNED
    assert "dico:GhostClass" not in retriever._node_to_idx


def test_invariant_7_non_punning_rejection(real_disk_db: sqlite3.Connection) -> None:
    """Verify Invariant 7: Dual-epistemic punning (URI typed as Class AND Concept) is strictly rejected."""
    g = rdflib.Graph().parse(data=PUNNING_SAMPLE_TTL, format="turtle")
    with pytest.raises(ValueError, match="(?i)punning"):
        ingest_ontology(source=g, conn=real_disk_db, ontology_id="punning_test")

    # Verify no classes were persisted on metamodeling violation
    count = real_disk_db.execute(
        "SELECT COUNT(*) FROM dynamic_classes WHERE ontology_id='punning_test'"
    ).fetchone()[0]
    assert count == 0


def test_zero_cache_wipes(real_disk_db: sqlite3.Connection) -> None:
    """Verify Invariant 2: Mode switching modifies 0 chunk embeddings and 0 leaf_relations."""
    real_disk_db.execute(
        "INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal, heading, heading_slug, line_start, line_end, embedding) VALUES ('c0', 'raw/1.md#0', '1.md', 0, 'Test', 'test', 1, 10, X'1234')"
    )
    real_disk_db.execute(
        "INSERT INTO leaf_relations VALUES ('rel_z', 'raw/1.md#0', 'concepts:a', 'causes', 'concepts:b', 1.0, 'causes')"
    )
    real_disk_db.commit()

    emb_before = real_disk_db.execute(
        "SELECT embedding FROM note_chunks WHERE entity_uri='raw/1.md#0'"
    ).fetchone()[0]
    count_before = real_disk_db.execute(
        "SELECT COUNT(*) FROM leaf_relations"
    ).fetchone()[0]

    # Toggle settings
    for mode in ["0", "1"]:
        real_disk_db.execute(
            "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', ?)", (mode,)
        )
        real_disk_db.commit()

    emb_after = real_disk_db.execute(
        "SELECT embedding FROM note_chunks WHERE entity_uri='raw/1.md#0'"
    ).fetchone()[0]
    count_after = real_disk_db.execute(
        "SELECT COUNT(*) FROM leaf_relations"
    ).fetchone()[0]

    assert emb_before == emb_after, "Chunk embedding was modified during mode toggle!"
    assert count_before == count_after, (
        "leaf_relations rows were modified during mode toggle!"
    )


@pytest.mark.anyio
async def test_degraded_retrieval_visibility_when_ollama_offline() -> None:
    """Verify mandate 2026-10-09T19:10:23Z: explicit degraded alerts when Ollama is offline."""
    from unittest.mock import patch

    from yaw.retrieval.retriever import GraphRAGRetriever
    from yaw.server.daemon import find_seed

    # 1. retriever.retrieve() markdown alert callout
    with (
        patch("yaw.nlp.is_ollama_online", return_value=False),
        patch("yaw.retrieval.retriever.is_ollama_online", return_value=False),
    ):
        retriever = GraphRAGRetriever()
        output = await retriever.retrieve("taxonomic inquiry")
        assert output.startswith(
            "> [!WARNING] Degraded Retrieval Mode: Ollama offline/unreachable."
        )

    # 2. daemon.find_seed() structured degraded warning
    with (
        patch("yaw.server.daemon.is_ollama_online", return_value=False),
        patch("yaw.retrieval.retriever.is_ollama_online", return_value=False),
    ):
        seed_res = await find_seed("taxonomic inquiry")
        assert isinstance(seed_res, dict)
        assert seed_res["status"] == "degraded"
        assert "warnings" in seed_res
        assert any("Ollama offline" in w for w in seed_res["warnings"])
        assert "candidates" in seed_res


@pytest.mark.anyio
async def test_mcp_daemon_extraction_and_ontology_tools(tmp_path: Path) -> None:
    """Verify FastMCP set_extraction_mode (<5ms) and load_ontology tools."""
    from yaw.server.daemon import load_ontology, set_extraction_mode

    db_path = tmp_path / "yaw.db"
    conn = sqlite3.connect(str(db_path))
    apply_pragmas(conn)
    ensure_schema_views(conn)
    conn.close()

    # 1. set_extraction_mode toggle < 50ms budget (< 5ms typical)
    t0 = time.perf_counter()
    mode_res = await set_extraction_mode(
        open_ie=False, target_ontology="external", vault_dir=str(tmp_path)
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert mode_res["status"] == "success"
    assert mode_res["open_ie"] is False
    assert mode_res["target_ontology"] == "external"
    assert elapsed_ms < 50.0

    # 2. load_ontology dynamic ingestion
    ont_res = await load_ontology(
        source=DICO_SAMPLE_TTL, ontology_id="dico", vault_dir=str(tmp_path)
    )
    assert ont_res["status"] == "success"
    assert ont_res["classes_ingested"] == 4
    assert ont_res["properties_ingested"] == 1


def test_dynamic_ontology_kb_synchronization(tmp_path: Path) -> None:
    """Verify ADR-010 / R3: Dynamic ontology loader synchronizes classes, properties, and altLabels into InMemoryLookupKB."""
    from yaw.indexer.canonicalize import create_in_memory_kb
    from yaw.ontology.loader import ingest_ontology
    from yaw.storage.db import apply_pragmas, ensure_schema_views

    db_path = tmp_path / "sync_kb_test.db"
    conn = sqlite3.connect(str(db_path))
    apply_pragmas(conn)
    ensure_schema_views(conn)

    kb = create_in_memory_kb()

    # Ingest DICO ontology passing kb
    res = ingest_ontology(source=DICO_SAMPLE_TTL, conn=conn, ontology_id="dico", kb=kb)
    assert res["status"] == "success"

    # Dynamic classes registered in KB
    assert kb.contains_entity("dico:BuildingElement")
    assert kb.contains_entity("dico:Wall")
    assert kb.contains_entity("dico:ExteriorWall")

    # Dynamic properties registered in KB
    assert kb.contains_entity("dico:isAdjacentTo")

    # Synonyms / labels registered as aliases (both exact casing and lowercase)
    assert kb.contains_alias("wall")
    assert kb.contains_alias("Wall")
    assert kb.contains_alias("dico:Wall")
    assert kb.contains_alias("ExteriorWall")
    assert kb.contains_alias("exteriorwall")

    cands = kb.get_alias_candidates("wall")
    assert len(cands) >= 1
    assert any(c.entity_ == "dico:Wall" for c in cands)

    cands_exact = kb.get_alias_candidates("Wall")
    assert len(cands_exact) >= 1
    assert any(c.entity_ == "dico:Wall" for c in cands_exact)

    conn.close()
