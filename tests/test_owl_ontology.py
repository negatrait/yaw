"""Comprehensive un-mocked test suite for Epic 6.2: OWL 2 RL & SKOS Taxonomic Subsumption Inference.

Verifies:
- R1: W3C Vocabulary & Scheme Alignment:
    - Static Turtle ontology loading via rdflib (src/yaw/ontology.ttl).
    - 8 universal node root classes (owl:Class, skos:Concept, rdfs:subClassOf owl:Thing, skos:topConceptOf yaw:VaultScheme).
    - 1 universal predicate property root (predicates:Root as owl:ObjectProperty, NOT owl:Class).
    - Canonical concept scheme (yaw:VaultScheme as skos:ConceptScheme with 8 top concepts).
    - Seed transitive properties (predicates:subClassOf, predicates:partOf, predicates:instanceOf).
    - build_skos_concept_scheme(conn) mapping SQLite unified_nodes & aliases to SKOS concepts, prefLabel,
      altLabel, definition, rdfs:subClassOf, skos:broader/narrower.
- R2: Deterministic Transitive Closure Engine:
    - get_subclass_closure(node_id, conn, include_self=False):
        - Multi-hop linear hierarchy (A -> B -> C -> D).
        - Diamond hierarchy (A -> B, A -> C, B -> D, C -> D).
        - Root node queries (returns empty set).
        - Leaf queries.
        - include_self=True parameter.
        - Support for bare and CURIE-prefixed predicates (subClassOf, predicates:subClassOf, rdfs:subClassOf).
        - Cycle-safe execution (does not hang or crash on cyclic data).
    - get_broader_closure(node_id, conn, include_self=False):
        - Multi-hop SKOS broader hierarchy with CURIEs (broader, skos:broader, predicates:broader).
        - Cycle-safe execution.
    - compute_transitive_closure_csr(adj_matrix, max_depth=None):
        - SciPy CSR matrix powers producing boolean reachability.
        - Parity with recursive SQLite CTE on identical graph topology.
        - max_depth hop limiting and boundary conditions (empty, 1x1, cyclic).
    - Invariant: Zero external DL reasoners (no JVM, HermiT, Pellet, or external daemons).
- R3: Reversible Identity & Distinction Mapping:
    - get_same_as_triples(conn):
        - Active aliases generate bidirectional owl:sameAs triples.
        - Inactive or contested aliases are excluded.
        - Instant reversibility when status changes from active to contested.
    - get_different_from_triples(conn):
        - Confirmed distinct canon_verdicts generate bidirectional owl:differentFrom triples.
        - Non-distinct (match/pending) verdicts are excluded.
        - Deduplication and empty table handling.
- R4: Subsumption DAG Cycle Prevention & Conductance Integration:
    - validate_subsumption_dag(edges, raise_on_cycle=True):
        - Valid DAG returns cleanly.
        - 2-node cycles, 3-node cycles, and self-loops raise TaxonomicCycleError (subclass of ValueError).
        - raise_on_cycle=False returns detected cycle paths without raising.
        - Disconnected components with cycles properly flagged.
    - GraphRAGRetriever conductance:
        - Asymmetric weights: 1.0 downward broadcast (hasSubClass, narrower), 0.2 upward damping (subClassOf, broader).
        - Support for CURIE prefixes (rdfs:subClassOf, predicates:subClassOf, skos:broader, predicates:broader, skos:narrower).
        - Real retriever test with test_db verifying CSR conductance weights.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rdflib
import scipy.sparse as sp
from rdflib import OWL, RDF, RDFS, SKOS, Literal, URIRef

try:
    _owl = importlib.import_module("yaw.ontology.owl")
    TaxonomicCycleError = _owl.TaxonomicCycleError
    build_skos_concept_scheme = _owl.build_skos_concept_scheme
    compute_transitive_closure_csr = _owl.compute_transitive_closure_csr
    get_broader_closure = _owl.get_broader_closure
    get_ontology_graph = _owl.get_ontology_graph
    get_same_as_triples = _owl.get_same_as_triples
    get_subclass_closure = _owl.get_subclass_closure
    validate_subsumption_dag = _owl.validate_subsumption_dag
    _MODULE_MISSING = False
except ImportError, ModuleNotFoundError:
    _MODULE_MISSING = True

    class _MissingModule:
        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError("No module named 'yaw.ontology'")

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError("No module named 'yaw.ontology'")

    _stub = _MissingModule()
    TaxonomicCycleError = ValueError  # type: ignore[assignment,misc]
    build_skos_concept_scheme = _stub  # type: ignore[assignment]
    compute_transitive_closure_csr = _stub  # type: ignore[assignment]
    get_broader_closure = _stub  # type: ignore[assignment]
    get_ontology_graph = _stub  # type: ignore[assignment]
    get_same_as_triples = _stub  # type: ignore[assignment]
    get_subclass_closure = _stub  # type: ignore[assignment]
    validate_subsumption_dag = _stub  # type: ignore[assignment]

pytestmark = pytest.mark.xfail(
    condition=_MODULE_MISSING,
    reason="Epic 6.2 Phase 1 RED: yaw.ontology not yet implemented",
    raises=(ModuleNotFoundError, ImportError, AssertionError),
)

from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage.rdf import ROOT_NAMESPACES, YAW_PREFIX

# Canonical Universal Roots defined in ADR-005 and docs/rdf-owl-shacl.md
EXPECTED_NODE_ROOT_NAMESPACES = (
    "concepts",
    "entities",
    "claims",
    "invariants",
    "decisions",
    "processes",
    "sources",
    "contexts",
)
VAULT_SCHEME_URI = URIRef(f"{YAW_PREFIX}VaultScheme")


# ==============================================================================
# Helper Utilities for Real SQLite Test Data Insertion
# ==============================================================================


def _insert_leaf_node(
    conn: sqlite3.Connection,
    node_id: str,
    name: str,
    namespace: str,
    definition: str | None = None,
    source_leaf_id: str = "raw/test.md",
) -> None:
    """Insert a leaf node record into real SQLite database."""
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES (?, ?, ?, ?, ?)
        """,
        (node_id, source_leaf_id, name, namespace, definition),
    )


def _insert_leaf_relation(
    conn: sqlite3.Connection,
    relation_id: str,
    subject_id: str,
    predicate: str,
    object_id: str,
    weight: float = 1.0,
    source_leaf_id: str = "raw/test.md",
) -> None:
    """Insert a leaf relation record into real SQLite database."""
    conn.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id, weight)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (relation_id, source_leaf_id, subject_id, predicate, object_id, weight),
    )


def _insert_alias(
    conn: sqlite3.Connection,
    alias_id: str,
    canonical_id: str,
    kind: str = "entity",
    method: str = "alias",
) -> None:
    """Insert an alias record into real SQLite database."""
    conn.execute(
        """
        INSERT INTO aliases (alias_id, kind, canonical_id, method, score)
        VALUES (?, ?, ?, ?, 1.0)
        """,
        (alias_id, kind, canonical_id, method),
    )


def _insert_canon_verdict(
    conn: sqlite3.Connection,
    surface_id: str,
    candidate_id: str,
    verdict: str,
    kind: str = "entity",
    phase: str = "compile",
    decided_at: float = 1700000000.0,
) -> None:
    """Insert a canon verdict record into real SQLite database."""
    try:
        conn.execute(
            """
            INSERT INTO canon_verdicts (surface_id, kind, candidate_id, cosine, lexical, verdict, phase, decided_at)
            VALUES (?, ?, ?, 0.9, 0.9, ?, ?, ?)
            """,
            (surface_id, kind, candidate_id, verdict, phase, decided_at),
        )
    except sqlite3.IntegrityError:
        alt_phase = "ingest" if phase == "compile" else "compile"
        conn.execute(
            """
            INSERT INTO canon_verdicts (surface_id, kind, candidate_id, cosine, lexical, verdict, phase, decided_at)
            VALUES (?, ?, ?, 0.9, 0.9, ?, ?, ?)
            """,
            (surface_id, kind, candidate_id, verdict, alt_phase, decided_at),
        )


# ==============================================================================
# R1: W3C Vocabulary & Concept Scheme Alignment Tests
# ==============================================================================


def test_static_ontology_turtle_roots_and_scheme() -> None:
    """Verify src/yaw/ontology.ttl declares 8 node roots, 1 predicate root, and yaw:VaultScheme."""
    ttl_path = Path("src/yaw/ontology.ttl")
    assert ttl_path.is_file(), f"Ontology Turtle file not found at {ttl_path}"

    # Load via get_ontology_graph() and verify parity with direct rdflib parse
    g = get_ontology_graph()
    assert isinstance(g, rdflib.Graph)

    direct_g = rdflib.Graph()
    direct_g.parse(ttl_path, format="turtle")
    assert len(g) == len(direct_g), (
        "Graph loaded via get_ontology_graph() differs from direct Turtle parse"
    )

    # Invariant: 8 Node Roots must be owl:Class, skos:Concept, rdfs:subClassOf owl:Thing, in yaw:VaultScheme
    for ns_name in EXPECTED_NODE_ROOT_NAMESPACES:
        root_uri = URIRef(f"{YAW_PREFIX}{ns_name}/Root")
        assert (root_uri, RDF.type, OWL.Class) in g, (
            f"{root_uri} must be declared as owl:Class"
        )
        assert (root_uri, RDF.type, SKOS.Concept) in g, (
            f"{root_uri} must be declared as skos:Concept"
        )
        assert (root_uri, RDFS.subClassOf, OWL.Thing) in g, (
            f"{root_uri} must be rdfs:subClassOf owl:Thing"
        )
        assert (root_uri, SKOS.topConceptOf, VAULT_SCHEME_URI) in g, (
            f"{root_uri} must be skos:topConceptOf yaw:VaultScheme"
        )
        assert (root_uri, SKOS.inScheme, VAULT_SCHEME_URI) in g, (
            f"{root_uri} must be skos:inScheme yaw:VaultScheme"
        )

    # Invariant: 1 Universal Predicate Root must be owl:ObjectProperty, explicitly NOT owl:Class or owl:Thing
    pred_root_uri = URIRef(f"{YAW_PREFIX}predicates/Root")
    assert (pred_root_uri, RDF.type, OWL.ObjectProperty) in g, (
        "predicates:Root must be declared as owl:ObjectProperty"
    )
    assert (pred_root_uri, RDF.type, OWL.Class) not in g, (
        "predicates:Root must NOT be declared as owl:Class"
    )
    assert (pred_root_uri, RDFS.subClassOf, OWL.Thing) not in g, (
        "predicates:Root must NOT be a subclass of owl:Thing"
    )

    # Invariant: yaw:VaultScheme must be skos:ConceptScheme with 8 top concepts
    assert (VAULT_SCHEME_URI, RDF.type, SKOS.ConceptScheme) in g, (
        "yaw:VaultScheme must be declared as skos:ConceptScheme"
    )
    top_concepts = set(g.objects(VAULT_SCHEME_URI, SKOS.hasTopConcept))
    for ns_name in EXPECTED_NODE_ROOT_NAMESPACES:
        root_uri = URIRef(f"{YAW_PREFIX}{ns_name}/Root")
        assert root_uri in top_concepts, (
            f"yaw:VaultScheme missing top concept {root_uri}"
        )
    assert len(top_concepts) == len(EXPECTED_NODE_ROOT_NAMESPACES)

    # Invariant: Seed transitive properties
    subclass_pred = URIRef(f"{YAW_PREFIX}predicates/subClassOf")
    assert (subclass_pred, RDF.type, OWL.TransitiveProperty) in g, (
        "predicates:subClassOf must be an owl:TransitiveProperty"
    )
    assert (subclass_pred, RDFS.subPropertyOf, RDFS.subClassOf) in g, (
        "predicates:subClassOf must be rdfs:subPropertyOf rdfs:subClassOf"
    )
    assert (subclass_pred, RDFS.subPropertyOf, pred_root_uri) in g, (
        "predicates:subClassOf must be rdfs:subPropertyOf predicates:Root"
    )

    partof_pred = URIRef(f"{YAW_PREFIX}predicates/partOf")
    assert (partof_pred, RDF.type, OWL.TransitiveProperty) in g, (
        "predicates:partOf must be an owl:TransitiveProperty"
    )
    assert (partof_pred, RDFS.subPropertyOf, pred_root_uri) in g, (
        "predicates:partOf must be rdfs:subPropertyOf predicates:Root"
    )


def test_build_skos_concept_scheme_nodes_and_literals(
    test_db: sqlite3.Connection,
) -> None:
    """Verify build_skos_concept_scheme projects SQLite unified_nodes into SKOS concepts and labels."""
    _insert_leaf_node(
        test_db,
        node_id="concepts:transformer",
        name="Transformer Architecture",
        namespace="concepts",
        definition="Self-attention neural sequence model.",
    )
    _insert_leaf_node(
        test_db,
        node_id="entities:alan-turing",
        name="Alan Turing",
        namespace="entities",
        definition="English mathematician and computer scientist.",
    )
    _insert_leaf_node(
        test_db,
        node_id="claims:p-equals-np",
        name="P equals NP",
        namespace="claims",
        definition=None,  # Edge case: NULL definition
    )
    test_db.commit()

    g = build_skos_concept_scheme(test_db)
    assert isinstance(g, rdflib.Graph)

    transformer_uri = URIRef(f"{YAW_PREFIX}concepts/transformer")
    turing_uri = URIRef(f"{YAW_PREFIX}entities/alan-turing")
    claim_uri = URIRef(f"{YAW_PREFIX}claims/p-equals-np")

    # Invariant: Type mapping
    assert (transformer_uri, RDF.type, SKOS.Concept) in g
    assert (transformer_uri, RDF.type, ROOT_NAMESPACES["concepts"]["Root"]) in g
    assert (turing_uri, RDF.type, SKOS.Concept) in g
    assert (turing_uri, RDF.type, ROOT_NAMESPACES["entities"]["Root"]) in g

    # Invariant: Labels and definitions
    assert (transformer_uri, SKOS.prefLabel, Literal("Transformer Architecture")) in g
    assert (
        transformer_uri,
        SKOS.definition,
        Literal("Self-attention neural sequence model."),
    ) in g
    assert (turing_uri, SKOS.prefLabel, Literal("Alan Turing")) in g

    # Edge case: NULL definition must NOT emit empty literal
    assert (claim_uri, SKOS.prefLabel, Literal("P equals NP")) in g
    assert list(g.objects(claim_uri, SKOS.definition)) == []


def test_build_skos_concept_scheme_aliases_alt_label(
    test_db: sqlite3.Connection,
) -> None:
    """Verify active aliases project into skos:altLabel while contested ones are excluded."""
    _insert_leaf_node(
        test_db,
        node_id="concepts:transformer",
        name="Transformer Architecture",
        namespace="concepts",
        definition="Attention model.",
    )
    # Active aliases
    _insert_alias(
        test_db,
        alias_id="concepts:attention-model",
        canonical_id="concepts:transformer",
    )
    _insert_alias(
        test_db,
        alias_id="concepts:xformer",
        canonical_id="concepts:transformer",
    )
    # Close match soft link (not a hard alias merge)
    from yaw.storage.db import insert_close_match

    insert_close_match(
        test_db,
        src_id="concepts:recurrent-net",
        dst_id="concepts:transformer",
        score=0.82,
    )
    test_db.commit()

    g = build_skos_concept_scheme(test_db)
    assert isinstance(g, rdflib.Graph)
    transformer_uri = URIRef(f"{YAW_PREFIX}concepts/transformer")

    alt_labels = {str(obj) for obj in g.objects(transformer_uri, SKOS.altLabel)}
    # Active aliases must appear as altLabel (either formatted name or alias slug)
    assert any(
        "attention-model" in lbl or "attention" in lbl.lower() for lbl in alt_labels
    ), f"Active alias attention-model missing in altLabels: {alt_labels}"
    assert any("xformer" in lbl for lbl in alt_labels), (
        f"Active alias xformer missing in altLabels: {alt_labels}"
    )

    # Soft close match must NOT appear as skos:altLabel
    assert not any("recurrent-net" in lbl for lbl in alt_labels), (
        "Close match recurrent-net must not be emitted as skos:altLabel"
    )


def test_build_skos_concept_scheme_taxonomic_relations(
    test_db: sqlite3.Connection,
) -> None:
    """Verify subsumption edges map to rdfs:subClassOf, skos:broader, and reciprocal skos:narrower."""
    _insert_leaf_node(test_db, "concepts:bert", "BERT", "concepts")
    _insert_leaf_node(test_db, "concepts:transformer", "Transformer", "concepts")
    _insert_leaf_node(test_db, "concepts:neural-net", "Neural Network", "concepts")

    _insert_leaf_relation(
        test_db, "r1", "concepts:bert", "subClassOf", "concepts:transformer"
    )
    _insert_leaf_relation(
        test_db, "r2", "concepts:transformer", "skos:broader", "concepts:neural-net"
    )
    test_db.commit()

    g = build_skos_concept_scheme(test_db)
    assert isinstance(g, rdflib.Graph)
    bert_uri = URIRef(f"{YAW_PREFIX}concepts/bert")
    transformer_uri = URIRef(f"{YAW_PREFIX}concepts/transformer")
    neural_net_uri = URIRef(f"{YAW_PREFIX}concepts/neural-net")

    # rdfs:subClassOf
    assert (bert_uri, RDFS.subClassOf, transformer_uri) in g
    # skos:broader
    assert (transformer_uri, SKOS.broader, neural_net_uri) in g
    # reciprocal skos:narrower
    assert (neural_net_uri, SKOS.narrower, transformer_uri) in g


# ==============================================================================
# R2: Deterministic Transitive Closure Engine Tests
# ==============================================================================


def test_subclass_closure_multi_hop_linear(test_db: sqlite3.Connection) -> None:
    """Verify recursive CTE computes exact multi-hop subclass closure over linear chain A->B->C->D."""
    # A subClassOf B subClassOf C subClassOf D
    _insert_leaf_relation(test_db, "r1", "concepts:a", "subClassOf", "concepts:b")
    _insert_leaf_relation(test_db, "r2", "concepts:b", "subClassOf", "concepts:c")
    _insert_leaf_relation(test_db, "r3", "concepts:c", "subClassOf", "concepts:d")
    test_db.commit()

    # Query A: ancestors = {B, C, D}
    closure_a = get_subclass_closure("concepts:a", test_db, include_self=False)
    assert closure_a == {"concepts:b", "concepts:c", "concepts:d"}

    # Query A with include_self=True: {A, B, C, D}
    closure_a_self = get_subclass_closure("concepts:a", test_db, include_self=True)
    assert closure_a_self == {"concepts:a", "concepts:b", "concepts:c", "concepts:d"}

    # Query B: {C, D}
    assert get_subclass_closure("concepts:b", test_db) == {"concepts:c", "concepts:d"}

    # Query C: {D}
    assert get_subclass_closure("concepts:c", test_db) == {"concepts:d"}

    # Query D (root node): empty set
    assert get_subclass_closure("concepts:d", test_db) == set()
    assert get_subclass_closure("concepts:d", test_db, include_self=True) == {
        "concepts:d"
    }


def test_subclass_closure_diamond_hierarchy(test_db: sqlite3.Connection) -> None:
    """Verify recursive CTE handles diamond inheritance without duplication (A->B, A->C, B->D, C->D)."""
    _insert_leaf_relation(test_db, "r1", "concepts:a", "subClassOf", "concepts:b")
    _insert_leaf_relation(test_db, "r2", "concepts:a", "subClassOf", "concepts:c")
    _insert_leaf_relation(test_db, "r3", "concepts:b", "subClassOf", "concepts:d")
    _insert_leaf_relation(test_db, "r4", "concepts:c", "subClassOf", "concepts:d")
    test_db.commit()

    closure_a = get_subclass_closure("concepts:a", test_db)
    assert closure_a == {"concepts:b", "concepts:c", "concepts:d"}
    assert len(closure_a) == 3

    assert get_subclass_closure("concepts:b", test_db) == {"concepts:d"}
    assert get_subclass_closure("concepts:c", test_db) == {"concepts:d"}


def test_subclass_closure_predicates_curie_variants(
    test_db: sqlite3.Connection,
) -> None:
    """Verify recursive CTE supports bare 'subClassOf', 'predicates:subClassOf', and 'rdfs:subClassOf'."""
    _insert_leaf_relation(test_db, "r1", "concepts:x", "subClassOf", "concepts:y")
    _insert_leaf_relation(
        test_db, "r2", "concepts:y", "predicates:subClassOf", "concepts:z"
    )
    _insert_leaf_relation(test_db, "r3", "concepts:z", "rdfs:subClassOf", "concepts:w")
    test_db.commit()

    closure_x = get_subclass_closure("concepts:x", test_db)
    assert closure_x == {"concepts:y", "concepts:z", "concepts:w"}


def test_subclass_closure_root_leaf_oov_and_empty_db(
    test_db: sqlite3.Connection,
) -> None:
    """Verify get_subclass_closure handles empty DB, out-of-vocabulary terms, and isolated nodes."""
    # Completely empty DB
    assert (
        get_subclass_closure("concepts:unknown", test_db, include_self=False) == set()
    )
    assert get_subclass_closure("concepts:unknown", test_db, include_self=True) == {
        "concepts:unknown"
    }

    # Isolated node with no incoming/outgoing relations
    _insert_leaf_node(test_db, "concepts:island", "Island", "concepts")
    test_db.commit()

    assert get_subclass_closure("concepts:island", test_db, include_self=False) == set()
    assert get_subclass_closure("concepts:island", test_db, include_self=True) == {
        "concepts:island"
    }


def test_subclass_closure_cycle_safe_execution(test_db: sqlite3.Connection) -> None:
    """Adversarial trap: verify corrupted cyclic data (A->B->C->A) does NOT hang or overflow recursive CTE."""
    _insert_leaf_relation(test_db, "r1", "concepts:a", "subClassOf", "concepts:b")
    _insert_leaf_relation(test_db, "r2", "concepts:b", "subClassOf", "concepts:c")
    _insert_leaf_relation(test_db, "r3", "concepts:c", "subClassOf", "concepts:a")
    test_db.commit()

    # Must terminate in finite time (< 1s) and return cycle members without crashing
    closure = get_subclass_closure("concepts:a", test_db, include_self=False)
    assert closure == {"concepts:b", "concepts:c"}

    closure_self = get_subclass_closure("concepts:a", test_db, include_self=True)
    assert closure_self == {"concepts:a", "concepts:b", "concepts:c"}


def test_broader_closure_multi_hop_and_curies(test_db: sqlite3.Connection) -> None:
    """Verify get_broader_closure traverses SKOS broader relations across bare and CURIE variants."""
    _insert_leaf_relation(
        test_db, "r1", "concepts:term_1", "broader", "concepts:term_2"
    )
    _insert_leaf_relation(
        test_db, "r2", "concepts:term_2", "skos:broader", "concepts:term_3"
    )
    _insert_leaf_relation(
        test_db, "r3", "concepts:term_3", "predicates:broader", "concepts:term_4"
    )
    test_db.commit()

    closure = get_broader_closure("concepts:term_1", test_db, include_self=False)
    assert closure == {"concepts:term_2", "concepts:term_3", "concepts:term_4"}

    closure_self = get_broader_closure("concepts:term_1", test_db, include_self=True)
    assert closure_self == {
        "concepts:term_1",
        "concepts:term_2",
        "concepts:term_3",
        "concepts:term_4",
    }


def test_broader_closure_cycle_safe_execution(test_db: sqlite3.Connection) -> None:
    """Adversarial trap: verify cyclic broader relations terminate cleanly."""
    _insert_leaf_relation(test_db, "r1", "concepts:x", "skos:broader", "concepts:y")
    _insert_leaf_relation(test_db, "r2", "concepts:y", "skos:broader", "concepts:x")
    test_db.commit()

    closure = get_broader_closure("concepts:x", test_db, include_self=False)
    assert closure == {"concepts:y"}


def test_compute_transitive_closure_csr_linear_and_diamond() -> None:
    """Verify compute_transitive_closure_csr produces boolean reachability over SciPy CSR matrices."""
    # 5-node linear DAG: 0 -> 1 -> 2 -> 3 -> 4
    n = 5
    row_ind = [0, 1, 2, 3]
    col_ind = [1, 2, 3, 4]
    data = [1.0, 1.0, 1.0, 1.0]
    adj = sp.csr_matrix((data, (row_ind, col_ind)), shape=(n, n), dtype=np.float32)

    closure = compute_transitive_closure_csr(adj)
    assert isinstance(closure, sp.csr_matrix)
    closure_arr = closure.toarray()

    # Row 0 reaches 1, 2, 3, 4
    np.testing.assert_array_equal(closure_arr[0], [0, 1, 1, 1, 1])
    # Row 1 reaches 2, 3, 4
    np.testing.assert_array_equal(closure_arr[1], [0, 0, 1, 1, 1])
    # Row 2 reaches 3, 4
    np.testing.assert_array_equal(closure_arr[2], [0, 0, 0, 1, 1])
    # Row 3 reaches 4
    np.testing.assert_array_equal(closure_arr[3], [0, 0, 0, 0, 1])
    # Row 4 reaches nothing
    np.testing.assert_array_equal(closure_arr[4], [0, 0, 0, 0, 0])

    # 4-node diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    d_rows = [0, 0, 1, 2]
    d_cols = [1, 2, 3, 3]
    d_data = [1.0, 1.0, 1.0, 1.0]
    diamond_adj = sp.csr_matrix(
        (d_data, (d_rows, d_cols)), shape=(4, 4), dtype=np.float32
    )

    d_closure = compute_transitive_closure_csr(diamond_adj).toarray()
    np.testing.assert_array_equal(d_closure[0], [0, 1, 1, 1])
    np.testing.assert_array_equal(d_closure[1], [0, 0, 0, 1])
    np.testing.assert_array_equal(d_closure[2], [0, 0, 0, 1])
    np.testing.assert_array_equal(d_closure[3], [0, 0, 0, 0])


def test_compute_transitive_closure_csr_parity_with_sql_cte(
    test_db: sqlite3.Connection,
) -> None:
    """Verify 100% mathematical reachability parity between SciPy CSR powers and SQLite recursive CTE."""
    # Build a complex branching hierarchy with 6 nodes
    nodes = [
        "concepts:n0",
        "concepts:n1",
        "concepts:n2",
        "concepts:n3",
        "concepts:n4",
        "concepts:n5",
    ]
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edges = [
        ("concepts:n0", "concepts:n1"),
        ("concepts:n0", "concepts:n2"),
        ("concepts:n1", "concepts:n3"),
        ("concepts:n2", "concepts:n4"),
        ("concepts:n3", "concepts:n5"),
        ("concepts:n4", "concepts:n5"),
    ]
    for i, (src, dst) in enumerate(edges):
        _insert_leaf_relation(test_db, f"rel_{i}", src, "subClassOf", dst)
    test_db.commit()

    # 1. SQL CTE closures for all nodes
    cte_closures = {
        n: get_subclass_closure(n, test_db, include_self=False) for n in nodes
    }

    # 2. SciPy CSR closure
    rows = [node_to_idx[src] for src, _ in edges]
    cols = [node_to_idx[dst] for _, dst in edges]
    adj = sp.csr_matrix(
        ([1.0] * len(edges), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    csr_closure = compute_transitive_closure_csr(adj).toarray()

    # 3. Assert mathematical equality
    for n in nodes:
        u = node_to_idx[n]
        reachable_indices = np.where(csr_closure[u] > 0)[0]
        csr_reached_nodes = {nodes[idx] for idx in reachable_indices}
        assert csr_reached_nodes == cte_closures[n], (
            f"Discrepancy at node {n}: CTE got {cte_closures[n]} vs CSR got {csr_reached_nodes}"
        )


def test_compute_transitive_closure_csr_max_depth_and_boundary_conditions() -> None:
    """Verify compute_transitive_closure_csr supports max_depth limiting and handles matrix boundaries."""
    # Linear chain of 4 nodes: 0 -> 1 -> 2 -> 3
    adj = sp.csr_matrix(([1.0, 1.0, 1.0], ([0, 1, 2], [1, 2, 3])), shape=(4, 4))

    # max_depth = 1: reachability equals immediate adjacency
    c1 = compute_transitive_closure_csr(adj, max_depth=1).toarray()
    np.testing.assert_array_equal(c1, adj.toarray())

    # max_depth = 2: reaches up to 2 hops
    c2 = compute_transitive_closure_csr(adj, max_depth=2).toarray()
    np.testing.assert_array_equal(
        c2[0], [0, 1, 1, 0]
    )  # 0 reaches 1 and 2, but not 3 yet
    np.testing.assert_array_equal(c2[1], [0, 0, 1, 1])

    # Boundaries: empty 0x0 matrix
    empty_adj = sp.csr_matrix((0, 0), dtype=np.float32)
    empty_c = compute_transitive_closure_csr(empty_adj)
    assert empty_c.shape == (0, 0)

    # 1x1 zero matrix
    single_adj = sp.csr_matrix((1, 1), dtype=np.float32)
    single_c = compute_transitive_closure_csr(single_adj)
    assert single_c.shape == (1, 1)
    assert single_c[0, 0] == 0

    # Cyclic matrix 0 -> 1 -> 0 converges without infinite loop
    cyclic_adj = sp.csr_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2))
    cyclic_c = compute_transitive_closure_csr(cyclic_adj).toarray()
    assert cyclic_c[0, 1] == 1
    assert cyclic_c[1, 0] == 1


def test_zero_dl_reasoners_invariant() -> None:
    """Verify that zero external DL tableaux reasoners (HermiT, Pellet, FaCT++) are imported or used."""
    forbidden_modules = (
        "owlready2.reasoning",
        "pyreasoner",
        "jpype",
        "pyjnius",
    )
    for mod in forbidden_modules:
        assert mod not in sys.modules, (
            f"Forbidden external DL reasoner module loaded: {mod}"
        )


# ==============================================================================
# R3: Reversible Identity & Distinction Mapping Tests
# ==============================================================================


def test_get_same_as_triples_active_aliases(test_db: sqlite3.Connection) -> None:
    """Verify aliases project into bidirectional owl:sameAs triples."""
    _insert_alias(test_db, "entities:albert", "entities:albert-einstein")
    _insert_alias(test_db, "entities:marie", "entities:marie-curie")
    test_db.commit()

    triples = get_same_as_triples(test_db)
    assert isinstance(triples, list)
    assert len(triples) == 4, (
        f"Expected 4 bidirectional triples for 2 aliases, got {len(triples)}"
    )

    # Check bidirectional pairs
    pairs = {(s, o) for s, p, o in triples if "sameAs" in p}
    assert ("entities:albert", "entities:albert-einstein") in pairs or (
        f"{YAW_PREFIX}entities/albert",
        f"{YAW_PREFIX}entities/albert-einstein",
    ) in pairs
    assert ("entities:albert-einstein", "entities:albert") in pairs or (
        f"{YAW_PREFIX}entities/albert-einstein",
        f"{YAW_PREFIX}entities/albert",
    ) in pairs
    assert ("entities:marie", "entities:marie-curie") in pairs or (
        f"{YAW_PREFIX}entities/marie",
        f"{YAW_PREFIX}entities/marie-curie",
    ) in pairs


def test_get_same_as_triples_excludes_contested_and_reversibility(
    test_db: sqlite3.Connection,
) -> None:
    """Verify close matches are excluded from sameAs and alias deletion immediately revokes equivalence."""
    from yaw.storage.db import delete_alias, insert_close_match

    _insert_alias(test_db, "entities:isaac", "entities:isaac-newton")
    insert_close_match(test_db, "entities:leibniz", "entities:isaac-newton", 0.82)
    test_db.commit()

    triples = get_same_as_triples(test_db)
    pairs = {(s, o) for s, p, o in triples}

    # Leibniz must NOT be equated to Newton via owl:sameAs
    assert not any("leibniz" in s or "leibniz" in o for s, o in pairs), (
        "Close match 'leibniz' must not generate owl:sameAs triples"
    )

    # Reversibility test: delete alias on isaac
    delete_alias(test_db, "entities:isaac", "entity")
    test_db.commit()

    triples_revoked = get_same_as_triples(test_db)
    assert len(triples_revoked) == 0, (
        "Deleting alias must instantly remove all owl:sameAs triples"
    )


def test_canon_verdicts_do_not_project_different_from_triples(
    test_db: sqlite3.Connection,
) -> None:
    """Verify ADR-012 §5: distinct canon verdicts do NOT project into owl:differentFrom triples under OWA."""
    _insert_canon_verdict(
        test_db, "entities:heisenberg", "entities:bohr", verdict="distinct"
    )
    _insert_canon_verdict(
        test_db, "entities:dirac", "entities:fermi", verdict="distinct"
    )
    # Non-distinct verdicts
    _insert_canon_verdict(
        test_db, "entities:feynman", "entities:richard-feynman", verdict="match"
    )
    _insert_canon_verdict(
        test_db, "entities:schroedinger", "entities:erwin", verdict="match"
    )
    test_db.commit()

    # Verify get_different_from_triples is deleted from yaw.ontology
    import yaw.ontology as ont

    assert not hasattr(ont, "get_different_from_triples")

    # Verify project_knowledge_graph emits zero owl:differentFrom triples
    from yaw.storage.rdf import project_knowledge_graph

    g = project_knowledge_graph(test_db)
    diff_triples = list(g.triples((None, OWL.differentFrom, None)))
    assert len(diff_triples) == 0


# ==============================================================================
# R4: Subsumption DAG Cycle Prevention & Conductance Integration Tests
# ==============================================================================


def test_validate_subsumption_dag_valid_dag_passes() -> None:
    """Verify validate_subsumption_dag returns cleanly with empty cycle list on valid acyclic trees."""
    valid_edges = [
        ("concepts:cat", "concepts:mammal"),
        ("concepts:dog", "concepts:mammal"),
        ("concepts:mammal", "concepts:animal"),
        ("concepts:animal", "concepts:living_thing"),
    ]
    cycles = validate_subsumption_dag(valid_edges, raise_on_cycle=True)
    assert cycles == [] or cycles is True or cycles is None

    # Empty edge set
    assert validate_subsumption_dag([], raise_on_cycle=True) in ([], True, None)


def test_validate_subsumption_dag_2_node_cycle_raises_taxonomic_cycle_error() -> None:
    """Verify 2-node cycle (A->B->A) raises TaxonomicCycleError (subclass of ValueError)."""
    cyclic_edges = [
        ("concepts:a", "concepts:b"),
        ("concepts:b", "concepts:a"),
    ]
    # Invariant: Must subclass ValueError
    assert issubclass(TaxonomicCycleError, ValueError)

    with pytest.raises(TaxonomicCycleError) as exc_info:
        validate_subsumption_dag(cyclic_edges, raise_on_cycle=True)
    err_str = str(exc_info.value)
    assert "concepts:a" in err_str and "concepts:b" in err_str

    # With raise_on_cycle=False, does not raise; returns detected cycle path
    cycles = validate_subsumption_dag(cyclic_edges, raise_on_cycle=False)
    assert isinstance(cycles, list)
    assert len(cycles) > 0


def test_validate_subsumption_dag_multi_hop_cycle_and_self_loops() -> None:
    """Verify 3-node cycle (X->Y->Z->X) and self-loop (S->S) raise TaxonomicCycleError."""
    tri_cycle = [
        ("concepts:x", "concepts:y"),
        ("concepts:y", "concepts:z"),
        ("concepts:z", "concepts:x"),
    ]
    with pytest.raises(TaxonomicCycleError) as exc_info:
        validate_subsumption_dag(tri_cycle, raise_on_cycle=True)
    assert "concepts:x" in str(exc_info.value)

    self_loop = [("concepts:self_term", "concepts:self_term")]
    with pytest.raises(TaxonomicCycleError):
        validate_subsumption_dag(self_loop, raise_on_cycle=True)


def test_validate_subsumption_dag_disconnected_components_with_one_cycle() -> None:
    """Verify cycle detection catches cycles in disconnected subgraphs without being masked by valid trees."""
    edges = [
        # Valid component 1
        ("concepts:tree1_a", "concepts:tree1_b"),
        ("concepts:tree1_b", "concepts:tree1_c"),
        # Cyclic component 2
        ("concepts:cycle_1", "concepts:cycle_2"),
        ("concepts:cycle_2", "concepts:cycle_3"),
        ("concepts:cycle_3", "concepts:cycle_1"),
    ]
    with pytest.raises(TaxonomicCycleError) as exc_info:
        validate_subsumption_dag(edges, raise_on_cycle=True)
    assert "concepts:cycle" in str(exc_info.value)


def test_retriever_subsumption_conductance_asymmetric_ratio(
    test_db: sqlite3.Connection,
) -> None:
    """Verify GraphRAGRetriever assigns asymmetric weights (1.0 downward, 0.2 upward, ratio 5.0) on subClassOf."""
    _insert_leaf_node(test_db, "concepts:transformer", "Transformer", "concepts")
    _insert_leaf_node(test_db, "concepts:neural-net", "Neural Net", "concepts")

    # Directed subsumption: transformer is a subClassOf neural-net
    _insert_leaf_relation(
        test_db, "r1", "concepts:transformer", "subClassOf", "concepts:neural-net"
    )
    test_db.commit()

    retriever = GraphRAGRetriever()
    retriever._sync_graph(test_db)

    assert retriever._adj is not None
    assert isinstance(retriever._adj, sp.csr_matrix)

    u_child = retriever._node_to_idx["concepts:transformer"]
    u_parent = retriever._node_to_idx["concepts:neural-net"]

    # Invariant: Downward parent -> child conductance (hasSubClass) = 1.0 (before hub damping)
    # Upward child -> parent damping (subClassOf) = 0.2 (before hub damping)
    # For an isolated pair with 1 edge, in-degrees are 1.0, so w = dir_w / (1.0^0.5) = dir_w
    w_down = retriever._adj[u_parent, u_child]
    w_up = retriever._adj[u_child, u_parent]

    assert pytest.approx(w_down, rel=1e-4) == 1.0, (
        f"Expected downward weight 1.0, got {w_down}"
    )
    assert pytest.approx(w_up, rel=1e-4) == 0.2, (
        f"Expected upward weight 0.2, got {w_up}"
    )
    assert pytest.approx(w_down / w_up, rel=1e-4) == 5.0, (
        "Downward-to-upward ratio must be exactly 5.0"
    )


def test_retriever_conductance_curie_prefixes_and_skos_broader_narrower(
    test_db: sqlite3.Connection,
) -> None:
    """Verify GraphRAGRetriever normalizes CURIE prefixes (predicates:, rdfs:, skos:) and broader/narrower."""
    _insert_leaf_node(test_db, "concepts:cnn", "CNN", "concepts")
    _insert_leaf_node(test_db, "concepts:rnn", "RNN", "concepts")
    _insert_leaf_node(test_db, "concepts:deep-learning", "Deep Learning", "concepts")
    _insert_leaf_node(test_db, "concepts:bert", "BERT", "concepts")
    _insert_leaf_node(test_db, "concepts:transformer", "Transformer", "concepts")
    _insert_leaf_node(test_db, "concepts:roberta", "RoBERTa", "concepts")

    # 1. predicates:subClassOf
    _insert_leaf_relation(
        test_db, "r1", "concepts:cnn", "predicates:subClassOf", "concepts:deep-learning"
    )
    # 2. rdfs:subClassOf
    _insert_leaf_relation(
        test_db, "r2", "concepts:rnn", "rdfs:subClassOf", "concepts:deep-learning"
    )
    # 3. skos:broader (bert is narrower than transformer)
    _insert_leaf_relation(
        test_db, "r3", "concepts:bert", "skos:broader", "concepts:transformer"
    )
    # 4. skos:narrower (transformer has narrower concept roberta)
    _insert_leaf_relation(
        test_db, "r4", "concepts:transformer", "skos:narrower", "concepts:roberta"
    )
    test_db.commit()

    retriever = GraphRAGRetriever()
    retriever._sync_graph(test_db)

    assert retriever._adj is not None
    assert isinstance(retriever._adj, sp.csr_matrix)

    idx = retriever._node_to_idx

    # Upward edges must be dampened to 0.2
    # cnn -> deep-learning (predicates:subClassOf)
    assert retriever._adj[idx["concepts:cnn"], idx["concepts:deep-learning"]] < 0.5
    # rnn -> deep-learning (rdfs:subClassOf)
    assert retriever._adj[idx["concepts:rnn"], idx["concepts:deep-learning"]] < 0.5
    # bert -> transformer (skos:broader)
    assert retriever._adj[idx["concepts:bert"], idx["concepts:transformer"]] < 0.5
    # roberta -> transformer (synthesized broader from narrower)
    assert retriever._adj[idx["concepts:roberta"], idx["concepts:transformer"]] < 0.5

    # Downward edges must broadcast at 1.0 (before hub damping)
    # deep-learning -> cnn (synthesized hasSubClass)
    assert retriever._adj[idx["concepts:deep-learning"], idx["concepts:cnn"]] > 0.5
    # deep-learning -> rnn (synthesized hasSubClass)
    assert retriever._adj[idx["concepts:deep-learning"], idx["concepts:rnn"]] > 0.5
    # transformer -> bert (synthesized narrower from broader)
    assert retriever._adj[idx["concepts:transformer"], idx["concepts:bert"]] > 0.5
    # transformer -> roberta (skos:narrower)
    assert retriever._adj[idx["concepts:transformer"], idx["concepts:roberta"]] > 0.5
