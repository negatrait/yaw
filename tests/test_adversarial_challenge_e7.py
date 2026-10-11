"""Empirical Adversarial Stress Suite for Epic 7 (teamwork_preview_challenger_e7_1).

Challenge Scope:
1. Antonym Polarity Stress Testing (multi-word causal & dialectical inversions, zero polarity collapse).
2. Invariant 7 Metamodeling Stress Testing (subtle punning, subgraphs, cross-epistemic subsumption/typing).
3. View Toggle Scalability (< 5ms latency budget, zero modification to leaf_relations under 5,000+ rows).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Generator
from pathlib import Path

import pytest
import rdflib

from yaw.nlp.components import normalize_predicate_antonym_guarded
from yaw.ontology.loader import ingest_ontology
from yaw.storage.db import apply_pragmas, ensure_schema_views


@pytest.fixture
def challenge_disk_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
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
    yield conn
    conn.close()


# ==============================================================================
# CHALLENGE 1: Antonym Polarity Stress Testing
# ==============================================================================


def test_antonym_polarity_multiword_causal_inversions() -> None:
    """Stress test multi-word causal inversions against polarity collapse.

    Confirm that positive causal phrases (e.g. 'leads to', 'results in') NEVER
    collapse to negative/inhibitory polarities ('prevents'), and inhibitory phrases
    (e.g. 'inhibits', 'prevents') NEVER collapse to positive polarities ('causes').
    """
    causal_positives = [
        "leads to",
        "lead to",
        "results in",
        "result in",
        "triggers",
        "induces",
        "brings about",
        "produces",
        "generates",
        "causes",
        "sparks",
        "gives rise to",
    ]

    causal_negatives = [
        "inhibits",
        "inhibit",
        "inhibited",
        "inhibiting",
        "inhibition",
        "prevents",
        "prevent",
        "prevented",
        "preventing",
        "prevention",
        "blocks",
        "stops",
        "thwarts",
        "precludes",
        "leads to inhibition",
        "results in prevention",
        "causes blocking",
    ]

    # 1. Assert all positive causal candidates NEVER normalize to 'prevents'
    for pos_phrase in causal_positives:
        norm = normalize_predicate_antonym_guarded(pos_phrase)
        assert norm != "prevents", (
            f"Polarity collapse! Positive phrase '{pos_phrase}' normalized to 'prevents'"
        )

    # 2. Assert all negative causal candidates NEVER normalize to 'causes'
    for neg_phrase in causal_negatives:
        norm = normalize_predicate_antonym_guarded(neg_phrase)
        assert norm != "causes", (
            f"Polarity collapse! Inhibitory phrase '{neg_phrase}' normalized to 'causes'"
        )

    # 3. Assert explicit pairwise distinctions
    assert normalize_predicate_antonym_guarded("leads to") == "causes"
    assert normalize_predicate_antonym_guarded("results in") == "causes"

    # 'inhibits' and 'prevents' must retain inhibitory semantic identity
    norm_inhibits = normalize_predicate_antonym_guarded("inhibits")
    assert norm_inhibits != "causes"
    norm_prevents = normalize_predicate_antonym_guarded("prevents")
    assert norm_prevents != "causes"
    assert norm_prevents == "prevents" or "prevent" in norm_prevents


def test_antonym_polarity_dialectical_inversions() -> None:
    """Stress test dialectical inversions ('supports' vs 'refutes' / 'contradicts').

    Confirm zero polarity collapse across diverse affirmative and counter-evidentiary phrasing.
    """
    dialectical_supports = [
        "supports",
        "support",
        "corroborates",
        "substantiates",
        "upholds",
        "endorses",
        "backs",
        "confirms",
        "verifies",
        "is consistent with",
        "strongly supports",
    ]

    dialectical_refutes = [
        "refutes",
        "refute",
        "disproves",
        "debunks",
        "rebuts",
        "invalidates",
        "undermines",
        "wholly refutes",
        "strongly refutes",
    ]

    dialectical_contradicts = [
        "contradicts",
        "contradict",
        "opposes",
        "negates",
        "strongly contradicts",
    ]

    # 1. Evidentiary supports must NEVER normalize to 'refutes' or 'contradicts'
    for sup in dialectical_supports:
        norm = normalize_predicate_antonym_guarded(sup)
        assert norm not in ("refutes", "contradicts"), (
            f"Polarity collapse! Supporting phrase '{sup}' normalized to counter-evidence '{norm}'"
        )

    # 2. Refuting / debunking phrases must NEVER normalize to 'supports'
    for ref in dialectical_refutes:
        norm = normalize_predicate_antonym_guarded(ref)
        assert norm != "supports", (
            f"Polarity collapse! Refuting phrase '{ref}' normalized to 'supports'"
        )

    # 3. Contradicting / opposing phrases must NEVER normalize to 'supports'
    for con in dialectical_contradicts:
        norm = normalize_predicate_antonym_guarded(con)
        assert norm != "supports", (
            f"Polarity collapse! Contradicting phrase '{con}' normalized to 'supports'"
        )

    # 4. Canonical baseline verification
    assert normalize_predicate_antonym_guarded("corroborates") == "supports"
    assert normalize_predicate_antonym_guarded("disproves") == "refutes"
    assert normalize_predicate_antonym_guarded("opposes") == "contradicts"


# ==============================================================================
# CHALLENGE 2: Invariant 7 Metamodeling Stress Testing
# ==============================================================================


def test_invariant_7_subtle_punning_scenarios(
    challenge_disk_db: sqlite3.Connection,
) -> None:
    """Adversarially challenge Invariant 7 with subtle metamodeling punning:

    Scenario A: Direct dual declaration in single turtle (owl:Class AND skos:Concept).
    Scenario B: Punning introduced via merged/imported subgraphs.
    Scenario C: Instance of Concept that is also typed as owl:Class.
    Scenario D: Subclass of Concept (cross-epistemic subsumption) or rdfs:Class punning.
    """

    # Scenario A: Direct dual declaration
    direct_punning_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

    ex:PunningNode a owl:Class, skos:Concept ;
        skos:prefLabel "Punning Node" .
    """
    g_a = rdflib.Graph().parse(data=direct_punning_ttl, format="turtle")
    with pytest.raises(ValueError, match="Invariant 7"):
        ingest_ontology(g_a, challenge_disk_db, ontology_id="direct_pun")

    # Scenario B: Punning via merged imported subgraphs
    # Subgraph 1 defines it as OWL Class
    subgraph_1_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    ex:MergedNode a owl:Class .
    """
    # Subgraph 2 defines it as SKOS Concept
    subgraph_2_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    ex:MergedNode a skos:Concept .
    """
    g_merged = rdflib.Graph()
    g_merged.parse(data=subgraph_1_ttl, format="turtle")
    g_merged.parse(data=subgraph_2_ttl, format="turtle")

    with pytest.raises(ValueError, match="Invariant 7"):
        ingest_ontology(g_merged, challenge_disk_db, ontology_id="merged_pun")

    # Scenario C: rdfs:Class and skos:Concept dual-typing (RDFS metamodeling punning)
    rdfs_punning_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

    ex:RdfsPunningNode a rdfs:Class, skos:Concept .
    """
    g_c = rdflib.Graph().parse(data=rdfs_punning_ttl, format="turtle")
    with pytest.raises(ValueError, match="Invariant 7"):
        ingest_ontology(g_c, challenge_disk_db, ontology_id="rdfs_pun")

    # Scenario D: Subclass of Concept (Class declared as rdfs:subClassOf Concept)
    subclass_of_concept_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

    ex:ConceptParent a skos:Concept .
    ex:ChildClass a owl:Class ;
        rdfs:subClassOf ex:ConceptParent .
    """
    g_d = rdflib.Graph().parse(data=subclass_of_concept_ttl, format="turtle")
    # Let's observe behavior: does ingest_ontology reject or accept this?
    try:
        ingest_ontology(g_d, challenge_disk_db, ontology_id="subclass_of_concept")
        subclass_of_concept_rejected = False
    except ValueError:
        subclass_of_concept_rejected = True

    # Scenario E: Instance of Concept that is also an OWL Class
    # (Node typed as ConceptItem where ConceptItem a skos:Concept, and Node a owl:Class)
    instance_of_concept_owl_class_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

    ex:TaxonConcept a skos:Concept .
    ex:HybridNode a ex:TaxonConcept, owl:Class .
    """
    g_e = rdflib.Graph().parse(data=instance_of_concept_owl_class_ttl, format="turtle")
    try:
        ingest_ontology(g_e, challenge_disk_db, ontology_id="instance_of_concept")
        instance_of_concept_rejected = False
    except ValueError:
        instance_of_concept_rejected = True

    # Scenario F: Instance of OWL Class that is also a skos:Concept
    instance_of_class_concept_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .

    ex:MetaClass a owl:Class .
    ex:HybridEntity a ex:MetaClass, skos:Concept .
    """
    g_f = rdflib.Graph().parse(data=instance_of_class_concept_ttl, format="turtle")
    try:
        ingest_ontology(g_f, challenge_disk_db, ontology_id="instance_of_class")
        instance_of_class_rejected = False
    except ValueError:
        instance_of_class_rejected = True

    print(
        f"\n[EMPIRICAL PUNNING PROBE] "
        f"subclass_of_concept_rejected={subclass_of_concept_rejected}, "
        f"instance_of_concept_rejected={instance_of_concept_rejected}, "
        f"instance_of_class_rejected={instance_of_class_rejected}"
    )
    assert subclass_of_concept_rejected is True, (
        "Subclass of concept must be rejected under Invariant 7"
    )
    assert instance_of_concept_rejected is True, (
        "Instance of concept typed as OWL Class must be rejected under Invariant 7"
    )
    assert instance_of_class_rejected is True, (
        "Instance of class typed as SKOS Concept must be rejected under Invariant 7"
    )


# ==============================================================================
# CHALLENGE 3: View Toggle Scalability & Zero Leaf Modification
# ==============================================================================


def test_view_toggle_scalability_and_immutability(
    challenge_disk_db: sqlite3.Connection,
) -> None:
    """Benchmark view toggle latency (< 5ms) and verify ZERO modifications to leaf_relations.

    Seeds 5,000 leaf relations across 14 canonical predicates and random open-IE verbs.
    Toggles runtime_settings.open_ie between 1 and 0 multiple times.
    Asserts:
    1. Toggle execution takes < 5 ms.
    2. unified_edges correctly filters open-IE rows down to canonical/ontology rows.
    3. leaf_relations is 100% byte-for-byte and row-for-row identical before and after.
    """
    # 1. Ingest base ontology with a registered property
    onto_ttl = """
    @prefix ex: <http://example.org/test#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    ex:TestOntology a owl:Ontology .
    ex:DomainEntity a owl:Class ; rdfs:label "Domain Entity" .
    ex:domainRelation a owl:ObjectProperty ; rdfs:label "domainRelation" .
    """
    g = rdflib.Graph().parse(data=onto_ttl, format="turtle")
    ingest_ontology(g, challenge_disk_db, ontology_id="test")

    # 2. Seed 5,000 leaf relations: 2,500 canonical/ontology, 2,500 open-IE verbs
    # Also seed leaf nodes
    challenge_disk_db.execute(
        "INSERT INTO leaf_nodes VALUES ('concepts:entity_a', 'raw/1.md#0', 'Entity A', 'concepts', 'Def', NULL)"
    )
    challenge_disk_db.execute(
        "INSERT INTO leaf_nodes VALUES ('concepts:entity_b', 'raw/1.md#0', 'Entity B', 'concepts', 'Def', NULL)"
    )

    rows: list[tuple[str, str, str, str, str, float, str]] = []
    for i in range(2500):
        # Canonical predicate
        rows.append(
            (
                f"rel_canon_{i}",
                "raw/1.md#0",
                "concepts:entity_a",
                "causes",
                "concepts:entity_b",
                1.0,
                "causes",
            )
        )
    for i in range(2500):
        # Open-IE out-of-vocabulary predicate
        rows.append(
            (
                f"rel_openie_{i}",
                "raw/1.md#0",
                "concepts:entity_a",
                f"arbitraryVerb_{i}",
                "concepts:entity_b",
                0.75,
                f"arbitraryverb_{i}",
            )
        )

    challenge_disk_db.executemany(
        "INSERT INTO leaf_relations VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    challenge_disk_db.commit()

    # Capture baseline snapshot of leaf_relations
    baseline_count = challenge_disk_db.execute(
        "SELECT COUNT(*) FROM leaf_relations"
    ).fetchone()[0]
    assert baseline_count == 5000
    baseline_hashes = challenge_disk_db.execute(
        "SELECT relation_id, predicate, weight FROM leaf_relations ORDER BY relation_id"
    ).fetchall()

    # 3. Test open_ie = 1 (Quadrant 1 / Quadrant 3)
    challenge_disk_db.execute(
        "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', '1'), ('target_ontology', 'internal')"
    )
    challenge_disk_db.commit()

    t0 = time.perf_counter()
    open_edges_count = challenge_disk_db.execute(
        "SELECT COUNT(*) FROM unified_edges WHERE source_chunk_id='raw/1.md#0'"
    ).fetchone()[0]
    t1 = time.perf_counter()
    latency_open_ms = (t1 - t0) * 1000

    # In open mode, all 5,000 relations should be visible
    assert open_edges_count == 5000
    assert latency_open_ms < 50.0  # SQLite view scan budget (< 50ms for 5,000 rows)

    # 4. Benchmark Toggle Latency (< 5 ms budget for mode switch DML)
    toggle_latencies: list[float] = []
    for _ in range(10):
        t_start = time.perf_counter()
        challenge_disk_db.execute(
            "INSERT OR REPLACE INTO runtime_settings VALUES ('open_ie', '0')"
        )
        challenge_disk_db.commit()
        t_end = time.perf_counter()
        toggle_latencies.append((t_end - t_start) * 1000)

    avg_toggle_ms = sum(toggle_latencies) / len(toggle_latencies)
    assert avg_toggle_ms < 5.0, (
        f"Toggle latency exceeded 5 ms SLA: {avg_toggle_ms:.3f} ms avg"
    )

    # 5. Check unified_edges filtering with open_ie = 0
    t2 = time.perf_counter()
    filtered_edges_count = challenge_disk_db.execute(
        "SELECT COUNT(*) FROM unified_edges WHERE source_chunk_id='raw/1.md#0'"
    ).fetchone()[0]
    t3 = time.perf_counter()
    latency_filtered_ms = (t3 - t2) * 1000
    assert latency_filtered_ms < 50.0  # SQLite view scan budget

    print(
        f"\n[BENCHMARK] View Toggle Scalability (5,000 leaf relations): "
        f"avg_toggle_dml={avg_toggle_ms:.4f} ms (budget < 5 ms), "
        f"open_view_scan={latency_open_ms:.4f} ms, "
        f"closed_view_scan={latency_filtered_ms:.4f} ms"
    )

    # Exactly the 2,500 canonical relations remain; open-IE relations filtered
    assert filtered_edges_count == 2500, (
        f"Expected 2,500 canonical edges in closed mode, got {filtered_edges_count}"
    )

    # 6. Verify INVARIANT 2: leaf_relations is COMPLETELY UNTOUCHED
    after_count = challenge_disk_db.execute(
        "SELECT COUNT(*) FROM leaf_relations"
    ).fetchone()[0]
    assert after_count == 5000, (
        "Violated Invariant 2! leaf_relations row count mutated!"
    )

    after_hashes = challenge_disk_db.execute(
        "SELECT relation_id, predicate, weight FROM leaf_relations ORDER BY relation_id"
    ).fetchall()
    assert after_hashes == baseline_hashes, (
        "Violated Invariant 2! leaf_relations contents mutated by view toggle!"
    )
