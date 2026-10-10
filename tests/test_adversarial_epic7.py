"""Production adversarial stress test suite for Epic 7 (teamwork_preview_challenger_e7_2).

Evaluates:
1. pySHACL Performance Stress Test (1k, 2k, 5k relational triples).
2. Active Subgraph Projection Stress Test (1k external ghost classes pruning & PageRank mass conservation).
3. Degraded Retrieval Visibility (offline Ollama markdown alert callout & daemon structured warning payload).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rdflib

from yaw.ontology.loader import ingest_ontology
from yaw.ontology.shacl import validate_wiki_graph_fast
from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.server.daemon import find_seed
from yaw.storage.db import apply_pragmas, ensure_schema_views


def _build_test_sqlite_db(db_path: Path) -> sqlite3.Connection:
    """Initialize real SQLite database on disk with complete schema and views."""
    conn = sqlite3.connect(str(db_path))
    apply_pragmas(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_chunks (
            entity_uri TEXT PRIMARY KEY,
            rel_path TEXT,
            heading TEXT,
            byte_offset INT,
            byte_length INT,
            line_start INT,
            line_end INT,
            ordinal INT,
            chunk_id TEXT,
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
    return conn


@pytest.mark.parametrize("num_triples", [1000, 2000, 5000])
def test_challenge_1_pyshacl_latency_benchmark(
    tmp_path: Path, num_triples: int
) -> None:
    """pySHACL Performance Stress Test:

    Construct synthetic SQLite databases with 1,000, 2,000, and 5,000 relational triples.
    Evaluate validate_wiki_graph_fast() across 5 repeated runs.
    Records latency distributions for empirical audit.
    """
    db_file = tmp_path / f"pyshacl_{num_triples}.db"
    conn = _build_test_sqlite_db(db_file)

    # Populate leaf nodes
    node_rows = [
        (
            f"concepts:node_{i}",
            "raw/stress.md#0",
            f"Concept Node {i}",
            "concepts",
            f"Formal taxonomic definition for concept node {i}",
            None,
        )
        for i in range(num_triples)
    ]
    conn.executemany("INSERT INTO leaf_nodes VALUES (?, ?, ?, ?, ?, ?)", node_rows)

    # Populate hierarchical relational triples in leaf_relations
    rel_rows = [
        (
            f"rel_{i}",
            "raw/stress.md#0",
            f"concepts:node_{i}",
            "subClassOf",
            f"concepts:node_{i // 2 if i > 0 else 0}",
            1.0,
            "subclassof",
        )
        for i in range(num_triples)
    ]
    conn.executemany(
        "INSERT INTO leaf_relations VALUES (?, ?, ?, ?, ?, ?, ?)", rel_rows
    )
    conn.commit()

    latencies_ms: list[float] = []
    conforms_list: list[bool] = []

    # Warm-up pass to populate shapes and schema caches
    validate_wiki_graph_fast(conn)

    for _ in range(5):
        t0 = time.perf_counter()
        conforms, _report = validate_wiki_graph_fast(conn)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        conforms_list.append(conforms)

    conn.close()

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    # Ensure validity
    assert all(conforms_list), f"Validation failed conformance on {num_triples} triples"

    print(
        f"\n[EMPIRICAL LATENCY REPORT] N={num_triples} triples: "
        f"avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms, runs={[round(x, 2) for x in latencies_ms]}"
    )

    # For 1,000 triples: SLA < 200 ms strictly holds
    if num_triples == 1000:
        assert avg_ms < 200.0, f"N=1000 exceeded 200ms latency budget: {avg_ms:.2f}ms"


def test_challenge_2_active_subgraph_projection_stress(tmp_path: Path) -> None:
    """Active Subgraph Projection Stress Test:

    1. Ingest external ontology with 1,000 uninstantiated classes.
    2. Verify retriever._node_to_idx contains strictly 0 uninstantiated external ghost classes.
    3. Verify seed PageRank mass does not dilute into inactive nodes.
    """
    db_file = tmp_path / "subgraph_stress.db"
    conn = _build_test_sqlite_db(db_file)

    # 1. Synthesize Turtle ontology with 1,000 ghost classes
    ttl_lines = [
        "@prefix ext: <https://example.org/stress-ontology#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
        "ext:RootSuperClass a owl:Class ;",
        '    rdfs:label "Root Super Class"@en .',
        "",
        "ext:ActiveInstantiated a owl:Class ;",
        "    rdfs:subClassOf ext:RootSuperClass ;",
        '    rdfs:label "Active Instantiated"@en .',
        "",
    ]
    for i in range(1000):
        ttl_lines.extend(
            [
                f"ext:GhostClass_{i} a owl:Class ;",
                "    rdfs:subClassOf ext:RootSuperClass ;",
                f'    rdfs:label "Ghost Class {i}"@en .',
                "",
            ]
        )

    ext_ttl = "\n".join(ttl_lines)
    g = rdflib.Graph().parse(data=ext_ttl, format="turtle")
    res = ingest_ontology(source=g, conn=conn, ontology_id="ext")
    assert res["status"] == "success"
    assert res["classes_ingested"] == 1002

    # 2. Instantiate ONLY ext:ActiveInstantiated in the wiki knowledge graph
    conn.execute(
        "INSERT INTO leaf_nodes VALUES ('ext:ActiveInstantiated', 'raw/page.md#0', 'Active Instantiated', 'ext', 'Definition', NULL)"
    )
    conn.execute(
        "INSERT INTO leaf_nodes VALUES ('concepts:OtherLeaf', 'raw/page.md#0', 'Other Leaf', 'concepts', 'Other concept', NULL)"
    )
    conn.execute(
        "INSERT INTO leaf_relations VALUES ('rel_inst', 'raw/page.md#0', 'ext:ActiveInstantiated', 'partOf', 'concepts:OtherLeaf', 1.0, 'partof')"
    )
    conn.commit()

    # 3. Build Retriever and sync graph
    retriever = GraphRAGRetriever()
    retriever._sync_graph(conn)

    # Invariant: Active classes and their direct ancestors to root are retained
    assert "ext:ActiveInstantiated" in retriever._node_to_idx
    assert "concepts:OtherLeaf" in retriever._node_to_idx
    assert "ext:RootSuperClass" in retriever._node_to_idx

    # Invariant: Check that strictly 0 of the 1,000 ghost classes appear in _node_to_idx
    ghost_present = [
        f"ext:GhostClass_{i}"
        for i in range(1000)
        if f"ext:GhostClass_{i}" in retriever._node_to_idx
    ]
    assert len(ghost_present) == 0, (
        f"Found {len(ghost_present)} uninstantiated ghost classes in retriever._node_to_idx: {ghost_present[:5]}"
    )

    # 4. Stress Test Seed PageRank Mass:
    # Ensure seed mass does not dilute into inactive nodes
    p0 = np.zeros(len(retriever._idx_to_node), dtype=np.float32)
    active_idx = retriever._node_to_idx["ext:ActiveInstantiated"]
    p0[active_idx] = 1.0

    p_star = retriever.power_iteration(p0, alpha=0.85, max_iter=30)

    # Mass conservation: sum of probabilities must be ~1.0 strictly across the active subgraph
    total_mass = float(np.sum(p_star))
    assert np.isclose(total_mass, 1.0, atol=1e-4), (
        f"PageRank mass leaked: sum={total_mass}"
    )

    # Verify that ghost classes received exactly 0 PageRank mass (they are not in the graph)
    for i in range(1000):
        ghost_curie = f"ext:GhostClass_{i}"
        assert ghost_curie not in retriever._node_to_idx

    conn.close()


@pytest.mark.anyio
async def test_challenge_3_degraded_retrieval_visibility_stress() -> None:
    """Degraded Retrieval Visibility Stress Test:

    1. Simulate offline Ollama state at query time.
    2. Assert retriever.retrieve() prepends '> [!WARNING] Degraded Retrieval Mode:' markdown alert callout.
    3. Assert daemon.find_seed() returns structured warning payload.
    """
    # 1. retriever.retrieve() markdown alert callout
    with (
        patch("yaw.nlp.is_ollama_online", return_value=False),
        patch("yaw.retrieval.retriever.is_ollama_online", return_value=False),
    ):
        retriever = GraphRAGRetriever()
        result_markdown = await retriever.retrieve("Architecture Overview")

        expected_callout = (
            "> [!WARNING] Degraded Retrieval Mode: Ollama offline/unreachable."
        )
        assert result_markdown.startswith(expected_callout), (
            f"Expected markdown to start with '{expected_callout}', but got: {result_markdown[:100]}"
        )
        assert "lexical FTS5 BM25 and taxonomic graph conductance" in result_markdown

    # 2. daemon.find_seed() structured degraded warning
    with (
        patch("yaw.server.daemon.is_ollama_online", return_value=False),
        patch("yaw.retrieval.retriever.is_ollama_online", return_value=False),
    ):
        seed_payload = await find_seed("Architecture Overview")
        assert isinstance(seed_payload, dict), (
            f"Expected dict payload, got {type(seed_payload)}"
        )
        assert seed_payload.get("status") == "degraded", (
            f"Expected status 'degraded', got {seed_payload.get('status')}"
        )
        assert "warnings" in seed_payload, "Missing 'warnings' key in payload"
        assert len(seed_payload["warnings"]) > 0
        assert any("Ollama offline" in w for w in seed_payload["warnings"])
        assert "candidates" in seed_payload
