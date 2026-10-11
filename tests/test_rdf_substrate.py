"""Unmocked real invariant and integration tests for RDF substrate and quad store integration.

Validates W3C RDF triple and named quad projection from SQLite views via in-process rdflib.
Covers:
- Group 1: to_rdflib_graph construction, namespace bindings, node typing, and labels.
- Group 2: Named quad provenance context, chunk URI retention, and context preservation.
- Group 3: Canonical CURIE mapping, predicate whitespace sanitization, and alias resolution.
- Group 4: Serialization formats (turtle, json-ld, nt, nquads), aliases, and case-insensitivity.
- Group 5: Round-trip parsing, statement equivalence, and quad context fidelity.
- Group 6: Corner cases, boundary defenses, missing views, and multi-note deduplication.
- Group 7: UTF-8 multilingual, international scripts, and mathematical symbol integrity.
"""

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from rdflib import BNode, ConjunctiveGraph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, SKOS

from yaw.storage.db import (
    apply_pragmas,
    ensure_schema_tables,
    ensure_schema_views,
    insert_alias,
)
from yaw.storage.rdf import (
    ROOT_NAMESPACES,
    YAW,
    YAW_PREFIX,
    chunk_to_context_uri,
    curie_to_uri,
    export_graph,
    to_rdflib_graph,
)

# Filter rdflib 7.6.0 ConjunctiveGraph deprecation warnings across this module
pytestmark = pytest.mark.filterwarnings(
    "ignore:ConjunctiveGraph is deprecated:DeprecationWarning"
)


def _parse_quads_without_warnings(data: str, format: str) -> ConjunctiveGraph:
    """Helper to parse RDF serialization into a ConjunctiveGraph with deprecation warnings filtered."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cg = ConjunctiveGraph()
        cg.parse(data=data, format=format)
        return cg


def _create_base_tables(conn: sqlite3.Connection) -> None:
    """Initialize raw base tables required by unified_edges and unified_nodes views."""
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
            line_start INT,
            line_end INT,
            embedding BLOB
        )
        """
    )


@pytest.fixture
def populated_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Create an isolated, fully populated SQLite database with sample domain records."""
    db_file = tmp_path / "populated_rdf.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    # 1. Chunks
    conn.execute(
        """
        INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal, heading, heading_slug, line_start, line_end)
        VALUES
        ('c0', 'raw/transformers.md#chunk:0', 'raw/transformers.md', 0, 'Self-Attention Mechanism', 'self-attention-mechanism', 1, 10),
        ('c1', 'raw/transformers.md#chunk:1', 'raw/transformers.md', 1, 'Multi-Head Projection', 'multi-head-projection', 11, 20)
        """
    )

    # 2. Leaf Nodes
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES
        ('concepts:transformer', 'raw/transformers.md', 'Transformer', 'concepts', 'Attention-based neural architecture.'),
        ('concepts:attention', 'raw/transformers.md', 'Attention', 'concepts', 'Sequence alignment mechanism.'),
        ('entities:vaswani', 'raw/transformers.md', 'Ashish Vaswani', 'entities', 'Primary author of the Transformer paper.')
        """
    )

    # 3. Leaf Relations (including bidirectional grounding & abstraction)
    conn.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id, weight)
        VALUES
        ('rel:1', 'raw/transformers.md#chunk:0', 'concepts:transformer', 'uses', 'concepts:attention', 1.0),
        ('rel:2', 'raw/transformers.md#chunk:0', 'concepts:transformer', 'manifests_in', 'raw/transformers.md#chunk:0', 1.0),
        ('rel:3', 'raw/transformers.md#chunk:1', 'raw/transformers.md#chunk:1', 'mentions', 'concepts:attention', 0.2),
        ('rel:4', 'raw/transformers.md#chunk:0', 'entities:vaswani', 'invented', 'concepts:transformer', 1.0)
        """
    )

    # 4. Wikilinks
    conn.execute(
        """
        INSERT INTO wikilinks (source_uri, target_uri, link_label)
        VALUES
        ('raw/transformers.md#chunk:0', 'raw/attention.md#chunk:0', 'Attention')
        """
    )

    # 5. Compiled Nodes & Relations
    conn.execute(
        """
        INSERT INTO compiled_nodes (node_id, name, namespace, definition, compiled_at)
        VALUES
        ('concepts:neural-network', 'Neural Network', 'concepts', 'Connectionist computational model.', 1700000000.0)
        """
    )
    conn.execute(
        """
        INSERT INTO compiled_relations (relation_id, subject_id, predicate, object_id, weight, source_rule, compiled_at)
        VALUES
        ('crel:1', 'concepts:transformer', 'subclass_of', 'concepts:neural-network', 1.0, 'rule:compile', 1700000000.0)
        """
    )

    # 6. Active Aliases
    insert_alias(
        conn,
        alias_id="concepts:self-attention",
        kind="entity",
        canonical_id="concepts:attention",
        method="lexical",
    )
    insert_alias(
        conn,
        alias_id="uses",
        kind="predicate",
        canonical_id="predicates:utilizes",
        method="lexical",
    )

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def empty_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Create an isolated, empty SQLite database with valid tables and views but zero rows."""
    db_file = tmp_path / "empty_rdf.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    conn.commit()
    yield conn
    conn.close()


# ==============================================================================
# Group 1: to_rdflib_graph Construction & Namespace Bindings
# ==============================================================================


def test_to_rdflib_graph_returns_conjunctive_graph(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify to_rdflib_graph returns a populated rdflib ConjunctiveGraph."""
    cg = to_rdflib_graph(populated_db)
    assert isinstance(cg, ConjunctiveGraph)
    quads = list(cg.quads())
    assert len(quads) > 0
    # 6 nodes * 4 triples each (24) + 6 edge quads = 30 total quads
    assert len(quads) == 30


def test_standard_and_root_namespace_bindings(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify all W3C and universal root namespaces are bound in the graph."""
    cg = to_rdflib_graph(populated_db)
    bound = dict(cg.namespaces())

    # W3C bindings
    assert str(bound["yaw"]) == str(YAW)
    assert str(bound["rdf"]) == str(RDF)
    assert str(bound["rdfs"]) == str(RDFS)
    assert str(bound["owl"]) == str(OWL)
    assert str(bound["sh"]) == str(SH)
    assert str(bound["skos"]) == str(SKOS)

    # Universal root bindings
    for prefix, expected_ns in ROOT_NAMESPACES.items():
        assert prefix in bound, f"Namespace prefix '{prefix}' must be bound"
        assert str(bound[prefix]) == str(expected_ns)


def test_node_namespace_type_mapping_concepts_entities_chunks(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify node namespace typing produces correct Root classes."""
    cg = to_rdflib_graph(populated_db)

    # Concept node typing
    transformer_uri = URIRef(f"{YAW_PREFIX}concepts/transformer")
    concept_root = ROOT_NAMESPACES["concepts"]["Root"]
    assert (transformer_uri, RDF.type, concept_root) in cg

    # Entity node typing
    vaswani_uri = URIRef(f"{YAW_PREFIX}entities/vaswani")
    entity_root = ROOT_NAMESPACES["entities"]["Root"]
    assert (vaswani_uri, RDF.type, entity_root) in cg

    # Chunk node typing -> mapped to sources:Root
    chunk_uri = URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0")
    sources_root = ROOT_NAMESPACES["sources"]["Root"]
    assert (chunk_uri, RDF.type, sources_root) in cg


def test_node_labels_and_definitions_mapping(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify RDFS.label, SKOS.prefLabel, and SKOS.definition literals."""
    cg = to_rdflib_graph(populated_db)

    transformer_uri = URIRef(f"{YAW_PREFIX}concepts/transformer")
    labels = list(cg.objects(transformer_uri, RDFS.label))
    pref_labels = list(cg.objects(transformer_uri, SKOS.prefLabel))
    definitions = list(cg.objects(transformer_uri, SKOS.definition))

    assert Literal("Transformer") in labels
    assert Literal("Transformer") in pref_labels
    assert Literal("Attention-based neural architecture.") in definitions

    # Note chunk label and definition (rel_path)
    chunk_uri = URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0")
    chunk_labels = list(cg.objects(chunk_uri, RDFS.label))
    chunk_defs = list(cg.objects(chunk_uri, SKOS.definition))
    assert Literal("Self-Attention Mechanism") in chunk_labels
    assert Literal("raw/transformers.md") in chunk_defs


def test_nodes_with_null_and_whitespace_attributes_emit_no_empty_literals(
    tmp_path: Path,
) -> None:
    """Verify nodes with NULL or whitespace-only name/definition omit empty literals."""
    db_file = tmp_path / "null_attr.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES
        ('concepts:null-node', 'note.md', '', 'concepts', NULL),
        ('concepts:ws-node', 'note.md', '   ', 'concepts', '  \t\n  ')
        """
    )
    conn.commit()

    cg = to_rdflib_graph(conn)
    for slug in ("null-node", "ws-node"):
        uri = URIRef(f"{YAW_PREFIX}concepts/{slug}")
        assert list(cg.objects(uri, RDFS.label)) == []
        assert list(cg.objects(uri, SKOS.prefLabel)) == []
        assert list(cg.objects(uri, SKOS.definition)) == []
        # Type triple must still exist
        assert (uri, RDF.type, ROOT_NAMESPACES["concepts"]["Root"]) in cg

    conn.close()


# ==============================================================================
# Group 2: Named Quad Provenance Context & Retention
# ==============================================================================


def test_named_quad_provenance_chunk_uri_preservation(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify quads are partitioned into named graph contexts reflecting source chunks."""
    cg = to_rdflib_graph(populated_db)

    ctx_chunk0 = URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0")
    ctx_chunk1 = URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:1")

    context_ids = {str(c.identifier) for c in cg.contexts()}
    assert str(ctx_chunk0) in context_ids
    assert str(ctx_chunk1) in context_ids

    # Check edges in chunk:0
    quads_c0 = list(cg.get_context(ctx_chunk0).triples((None, None, None)))
    assert len(quads_c0) == 4  # rel:1, rel:2, rel:4, wikilink


def test_named_quad_grounding_and_abstraction_contexts(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify bidirectional grounding (manifests_in) and abstraction (mentions) retain context."""
    cg = to_rdflib_graph(populated_db)

    ctx_chunk0 = URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0")
    ctx_chunk1 = URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:1")

    transformer = URIRef(f"{YAW_PREFIX}concepts/transformer")
    attention = URIRef(f"{YAW_PREFIX}concepts/attention")
    manifests_in = URIRef(f"{YAW_PREFIX}predicates/manifests_in")
    mentions = URIRef(f"{YAW_PREFIX}predicates/mentions")

    # Downward grounding edge in chunk:0
    c0_triples = list(cg.get_context(ctx_chunk0).triples((None, None, None)))
    assert (transformer, manifests_in, ctx_chunk0) in c0_triples

    # Upward abstraction edge in chunk:1
    c1_triples = list(cg.get_context(ctx_chunk1).triples((None, None, None)))
    assert (ctx_chunk1, mentions, attention) in c1_triples


def test_compiled_relation_context_uri_mapping(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify compiled relation quads land in contexts/compiled:{id} named graph."""
    cg = to_rdflib_graph(populated_db)

    expected_ctx = URIRef(f"{YAW_PREFIX}contexts/compiled:crel:1")
    ctx_triples = list(cg.get_context(expected_ctx).triples((None, None, None)))
    assert len(ctx_triples) == 1

    s, p, o = ctx_triples[0]
    assert s == URIRef(f"{YAW_PREFIX}concepts/transformer")
    assert p == URIRef(f"{YAW_PREFIX}predicates/subclass_of")
    assert o == URIRef(f"{YAW_PREFIX}concepts/neural-network")


def test_nquads_quad_structure_and_named_context_line_format(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify serialized N-Quads output conforms to standard 4-term format with provenance."""
    nq_text = export_graph(populated_db, "nquads")
    assert isinstance(nq_text, str)
    lines = [line.strip() for line in nq_text.strip().split("\n") if line.strip()]
    assert len(lines) == 30

    # Find the line representing the grounding edge
    grounding_found = False
    for line in lines:
        if "manifests_in" in line:
            grounding_found = True
            # Expecting 4 components ending with '.'
            match = re.match(
                r"^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s+\.$", line
            )
            assert match is not None, f"Grounding quad line malformed: {line}"
            sub, pred, obj, ctx = match.groups()
            assert sub == f"{YAW_PREFIX}concepts/transformer"
            assert pred == f"{YAW_PREFIX}predicates/manifests_in"
            assert obj == f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0"
            assert ctx == f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0"

    assert grounding_found, "Grounding edge quad not found in N-Quads export"


def test_nquads_roundtrip_quad_context_retention(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify parsing serialized N-Quads into a fresh ConjunctiveGraph restores named contexts."""
    nq_text = export_graph(populated_db, "nquads")
    g2 = _parse_quads_without_warnings(data=nq_text, format="nquads")

    assert len(list(g2.quads())) == 30

    recovered_named_contexts = {
        str(ctx.identifier)
        for _, _, _, ctx in g2.quads()
        if ctx is not None and not isinstance(ctx.identifier, BNode)
    }

    assert (
        f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0" in recovered_named_contexts
    )
    assert (
        f"{YAW_PREFIX}sources/raw/transformers.md#chunk:1" in recovered_named_contexts
    )
    assert f"{YAW_PREFIX}contexts/compiled:crel:1" in recovered_named_contexts


def test_jsonld_roundtrip_named_graph_retention(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify JSON-LD serialization preserves named contexts and parses cleanly."""
    jsonld_text = export_graph(populated_db, "json-ld")
    g2 = _parse_quads_without_warnings(data=jsonld_text, format="json-ld")

    assert len(list(g2.quads())) == 30
    recovered_contexts = {
        str(ctx.identifier)
        for _, _, _, ctx in g2.quads()
        if ctx is not None and not isinstance(ctx.identifier, BNode)
    }
    assert f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0" in recovered_contexts


# ==============================================================================
# Group 3: Canonical CURIE & URI Mapping & Sanitization
# ==============================================================================


@pytest.mark.parametrize(
    ("input_curie", "expected_uri"),
    [
        (None, f"{YAW_PREFIX}unknown"),
        ("", f"{YAW_PREFIX}unknown"),
        ("   ", f"{YAW_PREFIX}unknown"),
        ("http://example.org/entity", "http://example.org/entity"),
        ("https://example.org/path/resource", "https://example.org/path/resource"),
        ("urn:isbn:0-486-27557-4", "urn:isbn:0-486-27557-4"),
        ("yaw:concepts:transformer", f"{YAW_PREFIX}concepts/transformer"),
        ("concepts:transformer", f"{YAW_PREFIX}concepts/transformer"),
        ("entities:vaswani", f"{YAW_PREFIX}entities/vaswani"),
        ("predicates:manifests_in", f"{YAW_PREFIX}predicates/manifests_in"),
        ("claims:superposition-valid", f"{YAW_PREFIX}claims/superposition-valid"),
        ("invariants:immutability", f"{YAW_PREFIX}invariants/immutability"),
        ("decisions:adr-008", f"{YAW_PREFIX}decisions/adr-008"),
        ("processes:indexing", f"{YAW_PREFIX}processes/indexing"),
        ("custom_ns:foo", f"{YAW_PREFIX}custom_ns/foo"),
        ("raw/doc.md#chunk:0", f"{YAW_PREFIX}sources/raw/doc.md#chunk:0"),
        ("yaw:sources/raw/doc.md#chunk:0", f"{YAW_PREFIX}sources/raw/doc.md#chunk:0"),
        ("notes/attention.md", f"{YAW_PREFIX}sources/notes/attention.md"),
        ("/leading/slash.md", f"{YAW_PREFIX}sources/leading/slash.md"),
        ("plain-concept", f"{YAW_PREFIX}concepts/plain-concept"),
        ("plain concept", f"{YAW_PREFIX}concepts/plain-concept"),
    ],
)
def test_curie_to_uri_standard_patterns(
    input_curie: str | None, expected_uri: str
) -> None:
    """Verify curie_to_uri mapping across URIs, CURIEs, chunk paths, and plain strings."""
    assert curie_to_uri(input_curie) == URIRef(expected_uri)


def test_curie_to_uri_w3c_namespaces() -> None:
    """Verify curie_to_uri maps W3C standard prefixes to official W3C URIRefs."""
    assert curie_to_uri("rdf:type") == RDF.type
    assert curie_to_uri("rdfs:label") == RDFS.label
    assert curie_to_uri("owl:Class") == OWL.Class
    assert curie_to_uri("skos:prefLabel") == SKOS.prefLabel
    assert curie_to_uri("sh:NodeShape") == SH.NodeShape


def test_curie_to_uri_slug_whitespace_sanitization() -> None:
    """Verify whitespace in CURIE slugs is sanitized to hyphens."""
    assert curie_to_uri("concepts:neural network architecture") == URIRef(
        f"{YAW_PREFIX}concepts/neural-network-architecture"
    )
    assert curie_to_uri("predicates:is part of") == URIRef(
        f"{YAW_PREFIX}predicates/is-part-of"
    )


@pytest.mark.parametrize(
    ("input_chunk_id", "expected_context"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("http://example.org/ctx", URIRef("http://example.org/ctx")),
        ("compiled:crel:123", URIRef(f"{YAW_PREFIX}contexts/compiled:crel:123")),
        (
            "raw/transformers.md#chunk:0",
            URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0"),
        ),
        (
            "yaw:sources/raw/transformers.md#chunk:0",
            URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0"),
        ),
        (
            "/raw/transformers.md#chunk:0",
            URIRef(f"{YAW_PREFIX}sources/raw/transformers.md#chunk:0"),
        ),
    ],
)
def test_chunk_to_context_uri_patterns(
    input_chunk_id: str | None, expected_context: URIRef | None
) -> None:
    """Verify chunk_to_context_uri converts chunk IDs into canonical named graph URIs."""
    assert chunk_to_context_uri(input_chunk_id) == expected_context


def test_predicate_space_sanitization_in_graph(tmp_path: Path) -> None:
    """Verify predicates with whitespace are sanitized into hyphenated URIRefs."""
    db_file = tmp_path / "spaces.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    conn.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id)
        VALUES
        ('r1', 'raw/test.md#chunk:0', 'concepts:a', 'relate to', 'concepts:b'),
        ('r2', 'raw/test.md#chunk:0', 'concepts:b', 'is part of', 'concepts:c')
        """
    )
    conn.commit()

    cg = to_rdflib_graph(conn)
    p1 = URIRef(f"{YAW_PREFIX}predicates/relate-to")
    p2 = URIRef(f"{YAW_PREFIX}predicates/is-part-of")

    assert (
        URIRef(f"{YAW_PREFIX}concepts/a"),
        p1,
        URIRef(f"{YAW_PREFIX}concepts/b"),
    ) in cg
    assert (
        URIRef(f"{YAW_PREFIX}concepts/b"),
        p2,
        URIRef(f"{YAW_PREFIX}concepts/c"),
    ) in cg
    conn.close()


def test_alias_resolution_in_curie_projection(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify active aliases in aliases table re-route node and predicate URIs to canonical IDs."""
    cg = to_rdflib_graph(populated_db)

    # In populated_db, 'uses' alias points to 'predicates:utilizes'
    # 'concepts:transformer' -uses-> 'concepts:attention' must be projected as utilizes
    transformer = URIRef(f"{YAW_PREFIX}concepts/transformer")
    utilizes = URIRef(f"{YAW_PREFIX}predicates/utilizes")
    uses = URIRef(f"{YAW_PREFIX}predicates/uses")
    attention = URIRef(f"{YAW_PREFIX}concepts/attention")

    assert (transformer, utilizes, attention) in cg
    assert (transformer, uses, attention) not in cg


# ==============================================================================
# Group 4: Serialization Formats & Aliasing
# ==============================================================================


def test_export_graph_all_canonical_formats(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify export_graph produces non-empty output for all 4 canonical formats."""
    for fmt in ("turtle", "json-ld", "nt", "nquads"):
        output = export_graph(populated_db, fmt)
        assert isinstance(output, str)
        assert len(output.strip()) > 0


@pytest.mark.parametrize(
    ("alias", "canonical_target"),
    [
        ("ttl", "turtle"),
        ("jsonld", "json-ld"),
        ("ntriples", "nt"),
        ("n-triples", "nt"),
        ("nq", "nquads"),
        ("n-quads", "nquads"),
    ],
)
def test_export_graph_format_aliases(
    populated_db: sqlite3.Connection, alias: str, canonical_target: str
) -> None:
    """Verify all recognized format aliases produce valid serializations matching canonicals."""
    alias_output = export_graph(populated_db, alias)
    canonical_output = export_graph(populated_db, canonical_target)

    # Both must parse to equivalent quad count
    g_alias = _parse_quads_without_warnings(data=alias_output, format=canonical_target)
    g_canonical = _parse_quads_without_warnings(
        data=canonical_output, format=canonical_target
    )
    assert len(list(g_alias.quads())) == len(list(g_canonical.quads()))


@pytest.mark.parametrize(
    "case_variant",
    [
        "TURTLE",
        "Turtle",
        "TTL",
        "ttl",
        "JSON-LD",
        "Json-Ld",
        "jsonld",
        "JSONLD",
        "NT",
        "ntriples",
        "N-TRIPLES",
        "NQUADS",
        "NQuads",
        "nq",
        "NQ",
    ],
)
def test_export_graph_case_insensitivity(
    populated_db: sqlite3.Connection, case_variant: str
) -> None:
    """Verify format parameter handling is case-insensitive and whitespace-tolerant."""
    output = export_graph(populated_db, f"  {case_variant}  ")
    assert isinstance(output, str)
    assert len(output.strip()) > 0


@pytest.mark.parametrize("invalid_fmt", ["xml", "rdf/xml", "trig", "csv", "", "   "])
def test_export_graph_invalid_format_raises_value_error(
    populated_db: sqlite3.Connection, invalid_fmt: str
) -> None:
    """Verify unsupported format strings raise ValueError with descriptive text."""
    with pytest.raises(ValueError, match="Invalid format|Unsupported RDF format"):
        export_graph(populated_db, invalid_fmt)


@pytest.mark.parametrize("non_string", [None, 123, 45.6, [], {}])
def test_export_graph_invalid_types_raise_value_error(
    populated_db: sqlite3.Connection, non_string: Any
) -> None:
    """Verify non-string format arguments raise ValueError."""
    with pytest.raises(ValueError, match="Invalid format argument"):
        export_graph(populated_db, format=non_string)


def test_turtle_prefix_declarations_in_output(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify Turtle serialization contains @prefix declarations for standard namespaces."""
    ttl = export_graph(populated_db, "turtle")
    assert "@prefix concepts: <http://yaw.local/concepts/>" in ttl
    assert "@prefix predicates: <http://yaw.local/predicates/>" in ttl
    assert "@prefix sources: <http://yaw.local/sources/>" in ttl
    assert "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>" in ttl
    assert "@prefix skos: <http://www.w3.org/2004/02/skos/core#>" in ttl


def test_jsonld_valid_structure(populated_db: sqlite3.Connection) -> None:
    """Verify JSON-LD output is valid JSON containing @context and graph entries."""
    json_text = export_graph(populated_db, "json-ld")
    data = json.loads(json_text)
    assert isinstance(data, (dict, list))
    if isinstance(data, dict):
        assert "@graph" in data or "@context" in data
    elif isinstance(data, list):
        assert len(data) > 0


# ==============================================================================
# Group 5: Round-Trip Parsing & Statement Equivalence
# ==============================================================================


def test_turtle_roundtrip_parsing_preserves_all_triples(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify Turtle export round-trips all triples without statement loss."""
    cg_original = to_rdflib_graph(populated_db)
    ttl_text = export_graph(populated_db, "turtle")

    g2 = _parse_quads_without_warnings(data=ttl_text, format="turtle")
    orig_triples = set(cg_original.triples((None, None, None)))
    g2_triples = set(g2.triples((None, None, None)))

    assert orig_triples == g2_triples


def test_ntriples_roundtrip_parsing_preserves_all_triples(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify N-Triples export round-trips all triples identically."""
    cg_original = to_rdflib_graph(populated_db)
    nt_text = export_graph(populated_db, "nt")

    g2 = _parse_quads_without_warnings(data=nt_text, format="nt")
    orig_triples = set(cg_original.triples((None, None, None)))
    g2_triples = set(g2.triples((None, None, None)))

    assert orig_triples == g2_triples


def test_nquads_roundtrip_parsing_preserves_all_quads_and_contexts(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify N-Quads round-trip preserves quad count and provenance graph identities."""
    cg_original = to_rdflib_graph(populated_db)
    nq_text = export_graph(populated_db, "nquads")

    g2 = _parse_quads_without_warnings(data=nq_text, format="nquads")
    assert len(list(g2.quads())) == len(list(cg_original.quads()))

    orig_contexts: set[str] = set()
    for _, _, _, ctx in cg_original.quads():
        if ctx is not None and not isinstance(ctx.identifier, BNode):
            orig_contexts.add(str(ctx.identifier))
    g2_contexts: set[str] = set()
    for _, _, _, ctx in g2.quads():
        if ctx is not None and not isinstance(ctx.identifier, BNode):
            g2_contexts.add(str(ctx.identifier))
    assert orig_contexts == g2_contexts


def test_jsonld_roundtrip_parsing_preserves_all_statements(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify JSON-LD round-trip preserves all triples."""
    cg_original = to_rdflib_graph(populated_db)
    jsonld_text = export_graph(populated_db, "json-ld")

    g2 = _parse_quads_without_warnings(data=jsonld_text, format="json-ld")
    orig_triples = set(cg_original.triples((None, None, None)))
    g2_triples = set(g2.triples((None, None, None)))

    assert orig_triples == g2_triples


# ==============================================================================
# Group 6: Corner Cases, Error Handling & Boundary Defense
# ==============================================================================


def test_export_graph_empty_database_returns_valid_empty_serialization(
    empty_db: sqlite3.Connection,
) -> None:
    """Verify empty database returns empty valid serializations across all 4 formats."""
    cg = to_rdflib_graph(empty_db)
    assert len(list(cg.quads())) == 0

    for fmt in ("turtle", "json-ld", "nt", "nquads"):
        output = export_graph(empty_db, fmt)
        g2 = _parse_quads_without_warnings(data=output, format=fmt)
        assert len(list(g2.quads())) == 0


def test_to_rdflib_graph_missing_views_raises_operational_error(
    tmp_path: Path,
) -> None:
    """Verify calling to_rdflib_graph without required unified views raises OperationalError."""
    db_file = tmp_path / "noviews.db"
    conn = sqlite3.connect(str(db_file))
    # No views created

    with pytest.raises(
        sqlite3.OperationalError, match="Missing required unified views"
    ):
        to_rdflib_graph(conn)

    with pytest.raises(
        sqlite3.OperationalError, match="Missing required unified views"
    ):
        export_graph(conn, "turtle")

    conn.close()


def test_to_rdflib_graph_closed_connection_raises_error(
    populated_db: sqlite3.Connection,
) -> None:
    """Verify closed connection raises sqlite3.ProgrammingError or OperationalError."""
    populated_db.close()
    with pytest.raises(sqlite3.Error):
        to_rdflib_graph(populated_db)


def test_multi_note_composite_pk_nodes_deduplicated_in_rdf(
    tmp_path: Path,
) -> None:
    """Verify duplicate concept mentions across multiple notes are deduplicated by unified_nodes."""
    db_file = tmp_path / "dedup.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    # Note 1 and Note 2 both mention 'concepts:entropy'
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES
        ('concepts:entropy', 'note1.md', 'Entropy', 'concepts', 'Measure of uncertainty.'),
        ('concepts:entropy', 'note2.md', 'Entropy', 'concepts', 'Measure of uncertainty.')
        """
    )
    conn.commit()

    cg = to_rdflib_graph(conn)
    entropy_uri = URIRef(f"{YAW_PREFIX}concepts/entropy")

    # Should have exactly 1 type triple, 1 rdfs:label, 1 skos:prefLabel, 1 skos:definition
    types = list(cg.objects(entropy_uri, RDF.type))
    labels = list(cg.objects(entropy_uri, RDFS.label))
    defs = list(cg.objects(entropy_uri, SKOS.definition))

    assert len(types) == 1
    assert len(labels) == 1
    assert len(defs) == 1
    conn.close()


def test_wikilink_without_chunk_hash_mapping(tmp_path: Path) -> None:
    """Verify plain wikilink target paths without chunk hashes map to sources namespace."""
    db_file = tmp_path / "wikilinks.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    conn.execute(
        """
        INSERT INTO wikilinks (source_uri, target_uri, link_label)
        VALUES
        ('raw/quantum.md#chunk:0', 'notes/relativity.md', 'Relativity')
        """
    )
    conn.commit()

    cg = to_rdflib_graph(conn)
    s = URIRef(f"{YAW_PREFIX}sources/raw/quantum.md#chunk:0")
    p = URIRef(f"{YAW_PREFIX}predicates/wikilink")
    o = URIRef(f"{YAW_PREFIX}sources/notes/relativity.md")

    assert (s, p, o) in cg
    conn.close()


# ==============================================================================
# Group 7: UTF-8, Multilingual & Mathematical Symbols
# ==============================================================================


def test_unicode_cjk_cyrillic_accented_rdf_serialization(tmp_path: Path) -> None:
    """Verify CJK, Cyrillic, accented Latin, and German umlauts round-trip across all formats."""
    db_file = tmp_path / "multilingual.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    conn.execute(
        """
        INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal, heading)
        VALUES
        ('c0', 'raw/quantum.md#chunk:0', 'raw/quantum.md', 0, '量子计算与力学')
        """
    )
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES
        ('concepts:量子计算', 'raw/quantum.md', '量子计算', 'concepts', '利用量子叠加和纠缠特性的计算技术。'),
        ('concepts:квантовая-механика', 'raw/quantum.md', 'Квантовая Механика', 'concepts', 'Фундаментальная физическая теория.'),
        ('concepts:café-façade', 'raw/quantum.md', 'Café & Façade', 'concepts', 'Élégance française.'),
        ('concepts:schrödinger', 'raw/quantum.md', 'Erwin Schrödinger', 'concepts', 'Österreichischer Physiker.')
        """
    )
    conn.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id)
        VALUES
        ('r1', 'raw/quantum.md#chunk:0', 'concepts:量子计算', '基于', 'concepts:квантовая-механика')
        """
    )
    conn.commit()

    for fmt in ("turtle", "json-ld", "nt", "nquads"):
        output = export_graph(conn, fmt)
        g2 = _parse_quads_without_warnings(data=output, format=fmt)

        cjk_uri = URIRef(f"{YAW_PREFIX}concepts/量子计算")
        cyrillic_uri = URIRef(f"{YAW_PREFIX}concepts/квантовая-механика")
        pred_uri = URIRef(f"{YAW_PREFIX}predicates/基于")

        # Verify literal preservation
        assert Literal("利用量子叠加和纠缠特性的计算技术。") in g2.objects(
            cjk_uri, SKOS.definition
        )
        assert Literal("Фундаментальная физическая теория.") in g2.objects(
            cyrillic_uri, SKOS.definition
        )
        assert Literal("Élégance française.") in g2.objects(
            URIRef(f"{YAW_PREFIX}concepts/café-façade"), SKOS.definition
        )
        assert Literal("Österreichischer Physiker.") in g2.objects(
            URIRef(f"{YAW_PREFIX}concepts/schrödinger"), SKOS.definition
        )
        assert (cjk_uri, pred_uri, cyrillic_uri) in g2

    conn.close()


def test_mathematical_symbols_and_special_characters_preservation(
    tmp_path: Path,
) -> None:
    """Verify mathematical formulas, Greek characters, and symbols round-trip cleanly."""
    db_file = tmp_path / "math_symbols.db"
    conn = sqlite3.connect(str(db_file))
    apply_pragmas(conn)
    _create_base_tables(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    math_def = (
        "iℏ ∂Ψ/∂t = ĤΨ where ∫ |Ψ|² dV = 1, ∑ |c_i|² = 1, λ ∈ {λ₁, λ₂}, and [x̂, p̂] = iℏ"
    )
    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES
        ('claims:schroedinger-equation', 'raw/physics.md', 'Schrödinger Wave Equation', 'claims', ?)
        """,
        (math_def,),
    )
    conn.commit()

    for fmt in ("turtle", "json-ld", "nt", "nquads"):
        output = export_graph(conn, fmt)
        g2 = _parse_quads_without_warnings(data=output, format=fmt)

        claim_uri = URIRef(f"{YAW_PREFIX}claims/schroedinger-equation")
        defs = list(g2.objects(claim_uri, SKOS.definition))
        assert len(defs) == 1
        assert str(defs[0]) == math_def

    conn.close()
