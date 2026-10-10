"""Epic 6.3 (Sprint 6C): Closed-World SHACL Wiki Linter verification suite.

Verifies declarative W3C SHACL shape validation over SQLite unified views
via in-process pyshacl with OWL 2 RL inference and zero external graph daemons.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest
import rdflib
from rdflib import RDF, SH

from yaw import config
from yaw.ontology.shacl import (
    ValidationReport,
    load_shapes_graph,
    parse_validation_report,
    validate_wiki_graph,
)
from yaw.server.daemon import lint_wiki
from yaw.storage.db import (
    ensure_schema_tables,
    ensure_schema_views,
    insert_alias,
    insert_leaf_node,
)

YAW_PREFIX = "http://yaw.local/"


def _seed_clean_node(
    conn: sqlite3.Connection,
    node_id: str,
    name: str,
    namespace: str,
    definition: str,
    source_leaf_id: str = "raw/note1.md#chunk:0",
) -> None:
    """Helper to insert a valid node and ensure unified views are updated."""
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
    """Helper to insert a relation into leaf_relations and refresh views."""
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
# R1: Declarative SHACL Shape Specifications
# ==============================================================================


def test_shapes_file_exists_and_declares_shapes() -> None:
    """Verify wiki_shapes.ttl exists, parses cleanly, and defines required shapes."""
    shapes_path = Path("src/yaw/ontology/shapes/wiki_shapes.ttl")
    assert shapes_path.is_file(), f"SHACL shapes file not found at {shapes_path}"

    g = load_shapes_graph()
    assert isinstance(g, rdflib.Graph)
    assert len(g) > 0

    shape_subjects = set(g.subjects(RDF.type, SH.NodeShape))
    shape_uris = {str(s) for s in shape_subjects}

    assert any("ConceptShape" in uri for uri in shape_uris), (
        f"Missing ConceptShape in declared shapes: {shape_uris}"
    )
    assert any("RelationShape" in uri for uri in shape_uris), (
        f"Missing RelationShape in declared shapes: {shape_uris}"
    )
    assert any("DialecticalIntegrityShape" in uri for uri in shape_uris), (
        f"Missing DialecticalIntegrityShape in declared shapes: {shape_uris}"
    )


# ==============================================================================
# R2 & R3: Closed-World Validation Invariants (Real SQLite)
# ==============================================================================


def test_clean_knowledge_graph_conforms(test_db: sqlite3.Connection) -> None:
    """Verify that a compliant SQLite database passes closed-world SHACL validation."""
    _seed_clean_node(
        test_db,
        node_id="concepts:thermodynamics",
        name="Thermodynamics",
        namespace="concepts",
        definition="The branch of physical science that deals with the relations between heat and other forms of energy.",
    )
    _seed_clean_node(
        test_db,
        node_id="concepts:entropy",
        name="Entropy",
        namespace="concepts",
        definition="A thermodynamic quantity representing the unavailability of a system's thermal energy for conversion into mechanical work.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_1",
        src_id="concepts:entropy",
        predicate="partOf",
        dst_id="concepts:thermodynamics",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is True
    assert violations == []


def test_detect_missing_definition(test_db: sqlite3.Connection) -> None:
    """Verify that a concept without a definition triggers a SHACL violation."""
    # Insert node with empty/missing definition
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:undefined-concept', 'raw/test.md#chunk:0', 'Undefined Concept', 'concepts', '')
        """
    )
    ensure_schema_views(test_db)

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    assert len(violations) >= 1

    matched = [
        v
        for v in violations
        if "undefined-concept" in v["focus_node"]
        and ("definition" in v["message"].lower() or "definition" in v["path"].lower())
    ]
    assert len(matched) >= 1
    assert matched[0]["severity"] in ("Violation", "Warning")


def test_detect_definition_exceeding_word_limit(test_db: sqlite3.Connection) -> None:
    """Verify that a concept with > 50 words triggers a word limit violation."""
    long_definition = " ".join([f"term{i}" for i in range(55)])
    _seed_clean_node(
        test_db,
        node_id="concepts:verbose-concept",
        name="Verbose Concept",
        namespace="concepts",
        definition=long_definition,
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    assert len(violations) >= 1

    matched = [
        v
        for v in violations
        if "verbose-concept" in v["focus_node"] and "50 words" in v["message"].lower()
    ]
    assert len(matched) >= 1
    assert matched[0]["constraint"] in (
        "PatternConstraintComponent",
        "SPARQLConstraintComponent",
    )


def test_definition_within_word_limit_passes(test_db: sqlite3.Connection) -> None:
    """Verify that a concept with exactly 50 words conforms without violation."""
    exact_50_words = " ".join([f"word{i}" for i in range(config.DEFINITION_MAX_WORDS)])
    _seed_clean_node(
        test_db,
        node_id="concepts:exact-limit-concept",
        name="Exact Limit Concept",
        namespace="concepts",
        definition=exact_50_words,
    )

    _conforms, violations = validate_wiki_graph(test_db)
    word_limit_violations = [
        v
        for v in violations
        if "exact-limit-concept" in v["focus_node"]
        and "50 words" in v["message"].lower()
    ]
    assert word_limit_violations == []


def test_detect_invalid_namespace_syntax(test_db: sqlite3.Connection) -> None:
    """Verify that a node with unregistered namespace URI triggers a syntax violation."""
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('unregistered_ns:bad-concept', 'raw/test.md#chunk:0', 'Bad Node', 'unregistered_ns', 'Valid concise definition.')
        """
    )
    ensure_schema_views(test_db)

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "bad-concept" in v["focus_node"] or "unregistered_ns" in v["focus_node"]
    ]
    assert len(matched) >= 1


def test_detect_dangling_relation(test_db: sqlite3.Connection) -> None:
    """Verify that an edge pointing to a non-existent target is flagged as dangling."""
    _seed_clean_node(
        test_db,
        node_id="concepts:valid-source",
        name="Valid Source",
        namespace="concepts",
        definition="A valid source concept definition.",
    )
    # Target 'concepts:ghost-target' is NEVER inserted into unified_nodes
    _seed_relation(
        test_db,
        rel_id="rel_dangling",
        src_id="concepts:valid-source",
        predicate="partOf",
        dst_id="concepts:ghost-target",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "ghost-target" in v["value"] or "ghost-target" in v["focus_node"]
    ]
    assert len(matched) >= 1
    assert any("dangling" in v["message"].lower() for v in matched)


def test_detect_dialectical_contradiction_causes_prevents(
    test_db: sqlite3.Connection,
) -> None:
    """Verify that contradictory relations (causes vs prevents) trigger DialecticalIntegrityShape."""
    _seed_clean_node(
        test_db,
        node_id="claims:action-a",
        name="Action A",
        namespace="claims",
        definition="First proposition assertion.",
    )
    _seed_clean_node(
        test_db,
        node_id="concepts:outcome-b",
        name="Outcome B",
        namespace="concepts",
        definition="Resulting outcome concept.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_cause",
        src_id="claims:action-a",
        predicate="causes",
        dst_id="concepts:outcome-b",
    )
    _seed_relation(
        test_db,
        rel_id="rel_prevent",
        src_id="claims:action-a",
        predicate="prevents",
        dst_id="concepts:outcome-b",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "action-a" in v["focus_node"]
        and (
            "causes" in v["message"].lower() or "contradiction" in v["message"].lower()
        )
    ]
    assert len(matched) >= 1


def test_detect_dialectical_contradiction_supports_contradicts(
    test_db: sqlite3.Connection,
) -> None:
    """Verify that contradictory relations (supports vs contradicts) trigger DialecticalIntegrityShape."""
    _seed_clean_node(
        test_db,
        node_id="claims:claim-1",
        name="Claim 1",
        namespace="claims",
        definition="First claim assertion.",
    )
    _seed_clean_node(
        test_db,
        node_id="claims:claim-2",
        name="Claim 2",
        namespace="claims",
        definition="Second claim assertion.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_sup",
        src_id="claims:claim-1",
        predicate="supports",
        dst_id="claims:claim-2",
    )
    _seed_relation(
        test_db,
        rel_id="rel_con",
        src_id="claims:claim-1",
        predicate="contradicts",
        dst_id="claims:claim-2",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "claim-1" in v["focus_node"]
        and ("supports" in v["message"].lower() or "contradict" in v["message"].lower())
    ]
    assert len(matched) >= 1


def test_detect_contested_alias_merge(test_db: sqlite3.Connection) -> None:
    """Verify that an alias with status='contested' is flagged in validation."""
    ensure_schema_tables(test_db)
    _seed_clean_node(
        test_db,
        node_id="entities:canonical-target",
        name="Canonical Target",
        namespace="entities",
        definition="Canonical entity target.",
    )
    insert_alias(
        conn=test_db,
        alias_id="entities:disputed-surface",
        canonical_id="entities:canonical-target",
        kind="entity",
        status="contested",
    )
    ensure_schema_views(test_db)

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "disputed-surface" in v["focus_node"] or "disputed-surface" in v["value"]
    ]
    assert len(matched) >= 1
    assert any("contested" in v["message"].lower() for v in matched)


def test_validation_report_diagnostic_dictionary_keys(
    test_db: sqlite3.Connection,
) -> None:
    """Verify that each violation dictionary conforms strictly to the diagnostic schema."""
    _seed_clean_node(
        test_db,
        node_id="concepts:verbose-test",
        name="Verbose Test",
        namespace="concepts",
        definition=" ".join([f"word{i}" for i in range(55)]),
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    assert len(violations) >= 1

    required_keys = {"focus_node", "path", "severity", "constraint", "message", "value"}
    for v in violations:
        assert isinstance(v, dict)
        assert required_keys.issubset(v.keys())
        assert isinstance(v["focus_node"], str)
        assert isinstance(v["severity"], str)
        assert isinstance(v["constraint"], str)
        assert isinstance(v["message"], str)


def test_query_filtering_by_namespace(test_db: sqlite3.Connection) -> None:
    """Verify that query filtering by namespace isolates relevant violations."""
    # Add a concept violation and an entity contested violation
    _seed_clean_node(
        test_db,
        node_id="concepts:verbose-query",
        name="Verbose Query",
        namespace="concepts",
        definition=" ".join([f"word{i}" for i in range(55)]),
    )
    _seed_clean_node(
        test_db,
        node_id="entities:canon-ent",
        name="Canon Entity",
        namespace="entities",
        definition="Valid definition for canonical entity.",
    )
    insert_alias(
        conn=test_db,
        alias_id="entities:contested-ent",
        canonical_id="entities:canon-ent",
        kind="entity",
        status="contested",
    )
    ensure_schema_views(test_db)

    _c_all, all_v = validate_wiki_graph(test_db)
    assert len(all_v) >= 2

    # Filter strictly for concepts
    _c_concepts, concept_v = validate_wiki_graph(test_db, query="concepts")
    assert len(concept_v) >= 1
    assert all(
        "concepts" in v["focus_node"] or "concepts" in v["path"] for v in concept_v
    )
    assert not any("contested-ent" in v["focus_node"] for v in concept_v)

    # Filter strictly for entities
    _c_entities, entity_v = validate_wiki_graph(test_db, query="entities")
    assert len(entity_v) >= 1
    assert all(
        "entities" in v["focus_node"] or "entities" in v["value"] for v in entity_v
    )
    assert not any("verbose-query" in v["focus_node"] for v in entity_v)


def test_query_filtering_by_keyword(test_db: sqlite3.Connection) -> None:
    """Verify that query filtering by keyword narrows violations accurately."""
    _seed_clean_node(
        test_db,
        node_id="concepts:verbose-alpha",
        name="Alpha",
        namespace="concepts",
        definition=" ".join([f"word{i}" for i in range(55)]),
    )
    _seed_clean_node(
        test_db,
        node_id="concepts:verbose-beta",
        name="Beta",
        namespace="concepts",
        definition=" ".join([f"word{i}" for i in range(55)]),
    )

    _c, alpha_v = validate_wiki_graph(test_db, query="alpha")
    assert len(alpha_v) >= 1
    assert all("alpha" in v["focus_node"].lower() for v in alpha_v)
    assert not any("beta" in v["focus_node"].lower() for v in alpha_v)


@pytest.mark.anyio
async def test_lint_wiki_fastmcp_tool_invocation(
    test_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify asynchronous execution of @mcp.tool() async def lint_wiki()."""
    db_row = test_db.execute("PRAGMA database_list").fetchone()
    db_file = Path(db_row[2])
    _seed_clean_node(
        test_db,
        node_id="concepts:tool-node",
        name="Tool Node",
        namespace="concepts",
        definition=" ".join([f"w{i}" for i in range(55)]),
    )
    test_db.commit()

    monkeypatch.setattr(config, "DB_PATH", db_file)

    report: ValidationReport = await lint_wiki()
    assert isinstance(report, list)
    assert len(report) >= 1
    assert "focus_node" in report[0]

    # Check backwards-compatible dict access on the returned ValidationReport
    assert report["status"] == "violations"
    assert isinstance(report["violations"], list)

    # Check query filtering via FastMCP tool
    filtered: ValidationReport = await lint_wiki(query="nonexistent_keyword")
    assert len(filtered) == 0
    assert filtered["status"] == "clean"


def test_zero_external_daemon_processes(test_db: sqlite3.Connection) -> None:
    """Verify that validation runs strictly in-process with zero child daemons or external servers."""
    _seed_clean_node(
        test_db,
        node_id="concepts:inprocess-node",
        name="Inprocess Node",
        namespace="concepts",
        definition="A valid in-process node definition for architecture testing.",
    )

    # Snapshot process tree / ports before
    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is True
    assert violations == []

    # Verify no neo4j, blazegraph, or external java daemons were spawned
    pgrep_res = subprocess.run(
        ["pgrep", "-f", "neo4j|blazegraph|fuseki"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pgrep_res.returncode != 0, "External graph daemon detected running!"


def test_empty_database_conforms(test_db: sqlite3.Connection) -> None:
    """Verify that an empty database conforms with zero violations."""
    ensure_schema_views(test_db)
    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is True
    assert len(violations) == 0


def test_note_chunks_with_rel_path_conforms(test_db: sqlite3.Connection) -> None:
    """Verify that indexed note chunks in note_chunks do not trigger false positive URI violations."""
    test_db.execute(
        """
        INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal, heading)
        VALUES ('c1', 'raw/quantum.md#chunk:0', 'raw/quantum.md', 0, 'Quantum Mechanics')
        """
    )
    ensure_schema_views(test_db)

    _conforms, violations = validate_wiki_graph(test_db)
    chunk_uri_violations = [
        v
        for v in violations
        if "quantum.md" in v.get("focus_node", "")
        and "canonical 9-namespace" in v.get("message", "").lower()
    ]
    assert chunk_uri_violations == [], (
        f"False positive URI violation on note chunk: {chunk_uri_violations}"
    )


def test_canonical_uri_with_dots_conforms(test_db: sqlite3.Connection) -> None:
    """Verify that canonical concept URIs containing version numbers or dots conform."""
    _seed_clean_node(
        test_db,
        node_id="concepts:python-3.14",
        name="Python 3.14",
        namespace="concepts",
        definition="The 3.14 release of the Python programming language.",
    )
    _conforms, violations = validate_wiki_graph(test_db)
    dot_violations = [
        v
        for v in violations
        if "python-3.14" in v.get("focus_node", "")
        and "canonical 9-namespace" in v.get("message", "").lower()
    ]
    assert dot_violations == [], (
        f"Concept with dot in slug triggered false positive URI violation: {dot_violations}"
    )


def test_unregistered_namespace_subject_flagged(test_db: sqlite3.Connection) -> None:
    """Verify that a relation whose subject has an unregistered namespace is flagged."""
    _seed_clean_node(
        test_db,
        node_id="concepts:valid-target",
        name="Valid Target",
        namespace="concepts",
        definition="A valid target node definition.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_bad_subj",
        src_id="unregistered_ns:bad-source",
        predicate="partOf",
        dst_id="concepts:valid-target",
    )
    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "bad-source" in v.get("focus_node", "")
        or "unregistered_ns" in v.get("focus_node", "")
    ]
    assert len(matched) >= 1, (
        f"Unregistered namespace subject was not flagged: {violations}"
    )


def test_unregistered_namespace_predicate_flagged(test_db: sqlite3.Connection) -> None:
    """Verify that a relation whose predicate has an unregistered namespace is flagged."""
    _seed_clean_node(
        test_db,
        node_id="concepts:subj-node",
        name="Subject Node",
        namespace="concepts",
        definition="Subject node definition.",
    )
    _seed_clean_node(
        test_db,
        node_id="concepts:obj-node",
        name="Object Node",
        namespace="concepts",
        definition="Object node definition.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_bad_pred",
        src_id="concepts:subj-node",
        predicate="bad_ns:illegal_rel",
        dst_id="concepts:obj-node",
    )
    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "bad_ns" in v.get("path", "") or "illegal_rel" in v.get("path", "")
    ]
    assert len(matched) >= 1, f"Unregistered predicate was not flagged: {violations}"


def test_query_filtering_edge_cases(test_db: sqlite3.Connection) -> None:
    """Verify query filtering handles Unicode, regex meta-characters, and whitespace."""
    _seed_clean_node(
        test_db,
        node_id="concepts:quantum-spin",
        name="Quantum Spin",
        namespace="concepts",
        definition=" ".join(
            [f"term{i}" for i in range(55)]
        ),  # triggers word limit violation
    )
    ensure_schema_views(test_db)

    # Whitespace-only query should return all violations
    _c1, v_ws = validate_wiki_graph(test_db, query="   \t\n  ")
    assert len(v_ws) >= 1

    # Regex special characters like [ or * should not crash
    _c2, v_bracket = validate_wiki_graph(test_db, query="[")
    assert isinstance(v_bracket, ValidationReport)

    _c3, v_star = validate_wiki_graph(test_db, query="*")
    assert len(v_star) == len(v_ws)

    # Unicode query with no match should return clean report
    _c4, v_unicode = validate_wiki_graph(test_db, query="量子力学")
    assert len(v_unicode) == 0
    assert v_unicode.status == "clean"


def test_multiple_dialectical_conflicts_simultaneous(
    test_db: sqlite3.Connection,
) -> None:
    """Verify that multiple simultaneous dialectical conflicts and contested merges are all flagged."""
    ensure_schema_tables(test_db)
    _seed_clean_node(test_db, "claims:c1", "Claim 1", "claims", "Claim 1 def.")
    _seed_clean_node(test_db, "claims:c2", "Claim 2", "claims", "Claim 2 def.")
    _seed_clean_node(test_db, "claims:c3", "Claim 3", "claims", "Claim 3 def.")
    _seed_clean_node(test_db, "entities:canon1", "Canon 1", "entities", "Canon 1 def.")

    # Causes vs prevents on c1 -> c2
    _seed_relation(test_db, "r_cause", "claims:c1", "causes", "claims:c2")
    _seed_relation(test_db, "r_prev", "claims:c1", "prevents", "claims:c2")

    # Supports vs contradicts on c2 -> c3
    _seed_relation(test_db, "r_sup", "claims:c2", "supports", "claims:c3")
    _seed_relation(test_db, "r_con", "claims:c2", "contradicts", "claims:c3")

    # Contested alias
    insert_alias(
        conn=test_db,
        alias_id="entities:disputed1",
        canonical_id="entities:canon1",
        kind="entity",
        status="contested",
    )
    ensure_schema_views(test_db)

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    assert len(violations) >= 3

    has_causes_prevents = any("causes" in v["message"].lower() for v in violations)
    has_supports_contradicts = any(
        "supports" in v["message"].lower() for v in violations
    )
    has_contested = any("contested" in v["message"].lower() for v in violations)

    assert has_causes_prevents, f"Missing causes/prevents violation: {violations}"
    assert has_supports_contradicts, (
        f"Missing supports/contradicts violation: {violations}"
    )
    assert has_contested, f"Missing contested alias violation: {violations}"


def test_canonical_uri_with_colon_in_slug_conforms(test_db: sqlite3.Connection) -> None:
    """Verify that canonical concept and entity URIs with colons in slugs (e.g. RFCs, Wikidata) conform."""
    _seed_clean_node(
        test_db,
        node_id="concepts:rfc:9110",
        name="RFC 9110",
        namespace="concepts",
        definition="HTTP Semantics standard specification defined in RFC 9110.",
    )
    _seed_clean_node(
        test_db,
        node_id="entities:wikidata:Q42",
        name="Douglas Adams",
        namespace="entities",
        definition="English author and humorist, best known for The Hitchhiker's Guide to the Galaxy.",
    )
    conforms, violations = validate_wiki_graph(test_db)
    uri_violations = [
        v
        for v in violations
        if "canonical 9-namespace" in v.get("message", "").lower()
        and (
            "rfc:9110" in v.get("focus_node", "")
            or "wikidata:Q42" in v.get("focus_node", "")
        )
    ]
    assert uri_violations == [], (
        f"Colons in CURIE slugs triggered false positive URI violations: {uri_violations}"
    )
    assert conforms is True


def test_detect_dialectical_contradiction_supports_refutes(
    test_db: sqlite3.Connection,
) -> None:
    """Verify that simultaneous supports and refutes on identical target triggers DialecticalIntegrityShape."""
    _seed_clean_node(
        test_db,
        node_id="claims:thesis-a",
        name="Thesis A",
        namespace="claims",
        definition="First thesis claim assertion.",
    )
    _seed_clean_node(
        test_db,
        node_id="claims:thesis-b",
        name="Thesis B",
        namespace="claims",
        definition="Second thesis claim assertion.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_sup_ab",
        src_id="claims:thesis-a",
        predicate="supports",
        dst_id="claims:thesis-b",
    )
    _seed_relation(
        test_db,
        rel_id="rel_ref_ab",
        src_id="claims:thesis-a",
        predicate="refutes",
        dst_id="claims:thesis-b",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "thesis-a" in v["focus_node"]
        and (
            "refutes" in v["message"].lower() or "contradiction" in v["message"].lower()
        )
    ]
    assert len(matched) >= 1, f"Simultaneous supports/refutes not flagged: {violations}"


def test_detect_dialectical_reciprocal_contradiction(
    test_db: sqlite3.Connection,
) -> None:
    """Verify that reciprocal contradictions (c1 contradicts c2 while c2 supports c1) are flagged."""
    _seed_clean_node(
        test_db,
        node_id="claims:prop-x",
        name="Proposition X",
        namespace="claims",
        definition="Proposition X assertion.",
    )
    _seed_clean_node(
        test_db,
        node_id="claims:prop-y",
        name="Proposition Y",
        namespace="claims",
        definition="Proposition Y assertion.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_contra_xy",
        src_id="claims:prop-x",
        predicate="contradicts",
        dst_id="claims:prop-y",
    )
    _seed_relation(
        test_db,
        rel_id="rel_sup_yx",
        src_id="claims:prop-y",
        predicate="supports",
        dst_id="claims:prop-x",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "contradiction" in v["message"].lower() or "supports" in v["message"].lower()
    ]
    assert len(matched) >= 1, f"Reciprocal contradiction not flagged: {violations}"


def test_parse_validation_report_none_and_slash_uris() -> None:
    """Verify parse_validation_report gracefully handles None input and slash-delimited URIs."""
    assert parse_validation_report(None) == []

    g = rdflib.Graph()
    res_bnode = rdflib.BNode()
    g.add((res_bnode, RDF.type, SH.ValidationResult))
    g.add((res_bnode, SH.focusNode, rdflib.URIRef("http://yaw.local/concepts/foo")))
    g.add(
        (
            res_bnode,
            SH.resultSeverity,
            rdflib.URIRef("http://example.org/custom/SeverityLevel"),
        )
    )
    g.add(
        (
            res_bnode,
            SH.sourceConstraintComponent,
            rdflib.URIRef("http://example.org/constraints/CustomRule"),
        )
    )
    g.add((res_bnode, SH.resultMessage, rdflib.Literal("Custom failure message")))

    parsed = parse_validation_report(g)
    assert len(parsed) == 1
    assert parsed[0]["severity"] == "SeverityLevel"
    assert parsed[0]["constraint"] == "CustomRule"
    assert parsed[0]["message"] == "Custom failure message"


def test_validation_report_conforms_and_dialectical_properties(
    test_db: sqlite3.Connection,
) -> None:
    """Verify ValidationReport exposes conforms and dialectical_contradictions accessors."""
    ensure_schema_views(test_db)
    conforms, report = validate_wiki_graph(test_db)
    assert conforms is True
    assert report.conforms is True
    assert report["conforms"] is True
    assert report.dialectical_contradictions == 0
    assert report["dialectical_contradictions"] == 0
    assert "conforms" in report
    assert "dialectical_contradictions" in report

    # Seed dialectical contradiction
    _seed_clean_node(test_db, "claims:d1", "D1", "claims", "Claim D1.")
    _seed_clean_node(test_db, "claims:d2", "D2", "claims", "Claim D2.")
    _seed_relation(test_db, "r_cause_d", "claims:d1", "causes", "claims:d2")
    _seed_relation(test_db, "r_prev_d", "claims:d1", "prevents", "claims:d2")

    _c, report_bad = validate_wiki_graph(test_db)
    assert report_bad.conforms is False
    assert report_bad["conforms"] is False
    assert report_bad.dialectical_contradictions >= 1
    assert report_bad["dialectical_contradictions"] >= 1
    assert report_bad.get("dialectical_contradictions") >= 1


def test_definition_with_unicode_whitespace_and_newlines(
    test_db: sqlite3.Connection,
) -> None:
    """Verify definitions containing non-breaking spaces and newlines are evaluated accurately."""
    valid_def = "First\u00a0sentence\u00a0here.\nSecond\u00a0sentence\u00a0continues.\nThird line."
    _seed_clean_node(
        test_db,
        node_id="concepts:unicode-ws-ok",
        name="Unicode WS OK",
        namespace="concepts",
        definition=valid_def,
    )
    _conforms, violations = validate_wiki_graph(test_db)
    matched = [v for v in violations if "unicode-ws-ok" in v.get("focus_node", "")]
    assert matched == [], f"Valid multiline/unicode definition was flagged: {matched}"

    # Now seed a >50 word definition separated by unicode non-breaking spaces and newlines
    words_55 = (
        "\u00a0".join([f"term{i}" for i in range(25)])
        + "\n"
        + "\u00a0".join([f"word{i}" for i in range(30)])
    )
    _seed_clean_node(
        test_db,
        node_id="concepts:unicode-ws-bad",
        name="Unicode WS Bad",
        namespace="concepts",
        definition=words_55,
    )
    _conforms2, violations2 = validate_wiki_graph(test_db)
    matched_bad = [
        v for v in violations2 if "unicode-ws-bad" in v.get("focus_node", "")
    ]
    assert len(matched_bad) >= 1, (
        "Exceeding definition with unicode whitespace was not flagged!"
    )


@pytest.mark.anyio
async def test_lint_wiki_with_direct_db_path(test_db: sqlite3.Connection) -> None:
    """Verify lint_wiki handles vault_dir pointing directly to a SQLite database file."""
    db_row = test_db.execute("PRAGMA database_list").fetchone()
    db_file = Path(db_row[2])
    _seed_clean_node(
        test_db,
        node_id="concepts:direct-db-test",
        name="Direct DB Test",
        namespace="concepts",
        definition="Valid concept definition for direct DB path testing.",
    )
    test_db.commit()

    report: ValidationReport = await lint_wiki(vault_dir=str(db_file))
    assert isinstance(report, ValidationReport)
    assert report.status == "clean"
    assert report.conforms is True


def test_detect_reciprocal_supports_refutes_contradiction(
    test_db: sqlite3.Connection,
) -> None:
    """Verify reciprocal dialectical contradiction when c1 supports c2 while c2 refutes c1."""
    _seed_clean_node(
        test_db,
        node_id="claims:recip-a",
        name="Reciprocal A",
        namespace="claims",
        definition="Claim Reciprocal A definition.",
    )
    _seed_clean_node(
        test_db,
        node_id="claims:recip-b",
        name="Reciprocal B",
        namespace="claims",
        definition="Claim Reciprocal B definition.",
    )
    _seed_relation(
        test_db,
        rel_id="rel_recip_sup",
        src_id="claims:recip-a",
        predicate="supports",
        dst_id="claims:recip-b",
    )
    _seed_relation(
        test_db,
        rel_id="rel_recip_ref",
        src_id="claims:recip-b",
        predicate="refutes",
        dst_id="claims:recip-a",
    )

    conforms, violations = validate_wiki_graph(test_db)
    assert conforms is False
    matched = [
        v
        for v in violations
        if "contradiction" in v["message"].lower()
        or ("supports" in v["message"].lower() and "refutes" in v["message"].lower())
    ]
    assert len(matched) >= 1, (
        f"Reciprocal supports vs refutes contradiction not flagged: {violations}"
    )


def test_validation_report_to_dict_keys_values_items() -> None:
    """Verify ValidationReport exposes complete dict-like methods (to_dict, keys, values, items)."""
    item = {
        "focus_node": "http://yaw.local/claims/foo",
        "path": "http://yaw.local/predicates/causes",
        "severity": "Violation",
        "constraint": "SPARQLConstraintComponent",
        "message": "Dialectical contradiction: causes and prevents",
        "value": "http://yaw.local/claims/bar",
    }
    report = ValidationReport([item], conforms=False)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["conforms"] is False
    assert d["status"] == "violations"
    assert d["dialectical_contradictions"] == 1
    assert "status" in set(report.keys())
    assert "conforms" in set(report.keys())
    assert len(report.values()) == 6
    assert any(k == "status" and v == "violations" for k, v in report.items())


def test_load_shapes_graph_caching_and_reload() -> None:
    """Verify load_shapes_graph caches template, returns distinct isolated copies, and honors reload."""
    g1 = load_shapes_graph()
    g2 = load_shapes_graph()
    assert g1 is not g2, "Cached shapes graph should return isolated clones"
    assert len(g1) == len(g2)

    # Mutating g1 does not affect g2 or cache
    g1.add(
        (
            rdflib.URIRef("http://example.org/test_sub"),
            rdflib.RDF.type,
            rdflib.URIRef("http://example.org/test_obj"),
        )
    )
    assert len(g1) == len(g2) + 1

    g3 = load_shapes_graph(reload=True)
    assert len(g3) == len(g2)


def test_validate_wiki_graph_empty_and_whitespace_query_equivalence(
    test_db: sqlite3.Connection,
) -> None:
    """Verify validate_wiki_graph handles empty string, whitespace, and star queries identically to None."""
    _seed_clean_node(
        test_db,
        node_id="concepts:query-equiv-clean",
        name="Clean Concept",
        namespace="concepts",
        definition="A valid concept definition for query equivalence testing.",
    )
    c_none, r_none = validate_wiki_graph(test_db, query=None)
    c_empty, r_empty = validate_wiki_graph(test_db, query="")
    c_spaces, r_spaces = validate_wiki_graph(test_db, query="   ")
    c_star, r_star = validate_wiki_graph(test_db, query="*")

    assert c_none is True and c_empty is True and c_spaces is True and c_star is True
    assert len(r_none) == len(r_empty) == len(r_spaces) == len(r_star) == 0


def test_validate_wiki_graph_none_conn_raises_value_error() -> None:
    """Verify validate_wiki_graph and build_validation_data_graph raise ValueError on None conn."""
    from yaw.ontology.shacl import build_validation_data_graph

    bad_conn: Any = None
    with pytest.raises(ValueError, match="A valid open sqlite3.Connection is required"):
        validate_wiki_graph(bad_conn)

    with pytest.raises(ValueError, match="A valid open sqlite3.Connection is required"):
        build_validation_data_graph(bad_conn)
