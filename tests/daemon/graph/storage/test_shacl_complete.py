"""Comprehensive test suite for the complete 11 closed-world SHACL shapes and deterministic auto-healers."""

from pathlib import Path

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import (
    apply_shacl_repairs,
    get_facts_with_meta_by_ids,
    inspect_focus_node,
    lint_graph_shacl,
    open_yaw_db,
    serialize_graph_turtle,
    upsert_chunk_pointers,
    upsert_entities,
    upsert_entity_labels,
    upsert_fact_sources,
    upsert_facts,
)

ALL_11_SHACL_SHAPES = {
    "ConceptShape/prefLabel",
    "ConceptShape/definition",
    "ConceptShape/S13-disjointLabels",
    "ConceptShape/homographDifferentFrom",
    "StatementShape/closedWorld",
    "EdgeShape/closedWorld",
    "ConceptShape/broaderCycle",
    "ConceptShape/S27-disjointBroaderRelated",
    "ConceptShape/noScalarLiterals",
    "ConceptShape/lexicalSubsumption",
    "ConceptShape/siblingNotBroader",
    "ProvenanceShape/factSources",
}


def test_shacl_new_shapes_detection(tmp_path: Path) -> None:
    """Verify that all 11 closed-world SHACL shapes (including noScalarLiterals, lexicalSubsumption,
    siblingNotBroader, and factSources) flag violations when lint_graph_shacl(conn, auto_heal=False) runs."""
    conn = open_yaw_db(tmp_path / "test_shacl_detect.db")
    try:
        upsert_chunk_pointers(
            conn,
            [
                (ChunkPointer("c1", "doc.md", 1, 5, "Doc 1"), "System documentation."),
                (ChunkPointer("c2", "doc2.md", 1, 5, "Doc 2"), "Additional documentation."),
            ],
        )

        upsert_entities(
            conn,
            [
                ("ent-sys", "system", "Core platform system"),
                ("ent-no-pref", "orphan concept", "Concept missing skos:prefLabel"),
                ("ent-multi-pref", "primary label", "Concept with multiple skos:prefLabels"),
                ("ent-empty-def", "undocumented module", ""),
                ("ent-s13", "caching layer", "Distributed cache"),
                ("ent-hom-a", "mercury", "Planet closest to the Sun"),
                ("ent-hom-b", "mercury", "Heavy liquid metal element"),
                ("ent-2cyc-a", "distributed consensus", "Consensus protocols"),
                ("ent-2cyc-b", "state machine replication", "Replicated state machines"),
                ("ent-cyc-a", "cryptanalysis", "Analyzing cryptographic systems"),
                ("ent-cyc-b", "discrete mathematics", "Study of discrete structures"),
                ("ent-cyc-c", "number theory", "Study of integer properties"),
                ("ent-s27-leaf", "elliptic curve cryptography", "Public-key cryptography on algebraic curves"),
                ("ent-s27-mid", "asymmetric cryptography", "Public-key cryptographic algorithms"),
                ("ent-s27-root", "applied mathematics", "Mathematical methods in science and engineering"),
                ("ent-date", "2026-10-25", "A calendar date"),
                ("ent-lit-port", "8080", "Listen port literal"),
                ("ent-sec-eng", "security engineering", "Specialized engineering"),
                ("ent-eng", "engineering", "Broad engineering"),
                ("ent-v1", "python 3.10", "Python version 3.10"),
                ("ent-v2", "python 3.11", "Python version 3.11"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("system", "ent-sys", "skos:prefLabel", "system", "c1"),
                ("undocumented module", "ent-empty-def", "skos:prefLabel", "undocumented module", "c1"),
                ("mercury", "ent-hom-a", "skos:prefLabel", "mercury", "c1"),
                ("mercury", "ent-hom-b", "skos:prefLabel", "mercury", "c2"),
                ("distributed consensus", "ent-2cyc-a", "skos:prefLabel", "distributed consensus", "c1"),
                ("state machine replication", "ent-2cyc-b", "skos:prefLabel", "state machine replication", "c1"),
                ("cryptanalysis", "ent-cyc-a", "skos:prefLabel", "cryptanalysis", "c1"),
                ("discrete mathematics", "ent-cyc-b", "skos:prefLabel", "discrete mathematics", "c1"),
                ("number theory", "ent-cyc-c", "skos:prefLabel", "number theory", "c1"),
                (
                    "elliptic curve cryptography",
                    "ent-s27-leaf",
                    "skos:prefLabel",
                    "elliptic curve cryptography",
                    "c1",
                ),
                ("asymmetric cryptography", "ent-s27-mid", "skos:prefLabel", "asymmetric cryptography", "c1"),
                ("applied mathematics", "ent-s27-root", "skos:prefLabel", "applied mathematics", "c1"),
                ("2026-10-25", "ent-date", "skos:prefLabel", "2026-10-25", "c1"),
                ("8080", "ent-lit-port", "skos:prefLabel", "8080", "c1"),
                ("security engineering", "ent-sec-eng", "skos:prefLabel", "security engineering", "c1"),
                ("engineering", "ent-eng", "skos:prefLabel", "engineering", "c1"),
                ("python 3.10", "ent-v1", "skos:prefLabel", "python 3.10", "c1"),
                ("python 3.11", "ent-v2", "skos:prefLabel", "python 3.11", "c1"),
            ],
        )

        # 1. ConceptShape/prefLabel:
        # - ent-no-pref has only an altLabel (no prefLabel)
        # - ent-multi-pref has 2 distinct prefLabel norms inserted via raw SQL
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('orphan alias', 'ent-no-pref', 'skos:altLabel', 'orphan alias', 'c1')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('primary label', 'ent-multi-pref', 'skos:prefLabel', 'primary label', 'c1')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('secondary label', 'ent-multi-pref', 'skos:prefLabel', 'secondary label', 'c2')"
        )

        # 3. ConceptShape/S13-disjointLabels: same label_norm with both prefLabel and altLabel across chunks
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('caching layer', 'ent-s13', 'skos:prefLabel', 'caching layer', 'c1')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('caching layer', 'ent-s13', 'skos:altLabel', 'caching layer', 'c2')"
        )

        # 6. EdgeShape/closedWorld:
        # - dangling endpoint (ent-sys -> ent-dangling)
        # - skos:related + owl:differentFrom collision on non-homographs (ent-sys | ent-s13)
        # - asymmetric skos:related (ent-sys -> ent-eng without reverse)
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-dangling', 'fact', 1.0)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-s13', 'ent-sys', 'skos:related', 0.6)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-s13', 'skos:related', 0.6)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-s13', 'ent-sys', 'owl:differentFrom', 0.0)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-s13', 'owl:differentFrom', 0.0)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-eng', 'skos:related', 0.7)"
        )

        # 7. ConceptShape/broaderCycle (2-cycle and 3-cycle) & ConceptShape/S27-disjointBroaderRelated:
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-2cyc-a', 'ent-2cyc-b', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-2cyc-b', 'ent-2cyc-a', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-a', 'ent-cyc-b', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-b', 'ent-cyc-c', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-c', 'ent-cyc-a', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) "
            "VALUES ('ent-s27-leaf', 'ent-s27-mid', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) "
            "VALUES ('ent-s27-mid', 'ent-s27-root', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) "
            "VALUES ('ent-s27-leaf', 'ent-s27-root', 'skos:related', 0.5)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) "
            "VALUES ('ent-s27-root', 'ent-s27-leaf', 'skos:related', 0.5)"
        )

        # 8. ConceptShape/noScalarLiterals:
        # - pure literal node (ent-lit-port) with illegal skos:broader and skos:related edges
        # - scalar entity (ent-date) with obj_kind='entity'
        upsert_facts(
            conn,
            [
                (
                    "fact-port",
                    "ent-sys",
                    "listens on port",
                    "ent-lit-port",
                    '["system", "listens on port", "8080"]',
                    "System listens on port 8080.",
                    "literal",
                    "8080",
                    "rdfs:Literal",
                    "",
                ),
            ],
        )
        upsert_fact_sources(conn, [("fact-port", "c1")])
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('c1', 'ent-lit-port', 'passage', 1.0)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-lit-port', 'fact', 0.5)")
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-lit-port', 'ent-sys', 'skos:broader', 0.5)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-lit-port', 'ent-sys', 'skos:related', 0.5)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-lit-port', 'skos:related', 0.5)"
        )

        # 9. ConceptShape/lexicalSubsumption: raw inverted broader edge (engineering -> security engineering)
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-eng', 'ent-sec-eng', 'skos:broader', 0.8)"
        )

        # 10. ConceptShape/siblingNotBroader: raw version siblings connected with skos:broader
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-v1', 'ent-v2', 'skos:broader', 0.5)"
        )

        # 5. StatementShape/closedWorld (dangling endpoint + empty definition) &
        # 11. ProvenanceShape/factSources (fact without entry in fact_sources)
        upsert_facts(
            conn,
            [
                (
                    "fact-dangling",
                    "ent-sys",
                    "depends on",
                    "ent-missing-target",
                    '["system", "depends on", "missing"]',
                    "System depends on missing target.",
                    "entity",
                    None,
                    "",
                    "",
                ),
                (
                    "fact-empty-def",
                    "ent-sys",
                    "uses",
                    "ent-s13",
                    '["system", "uses", "caching layer"]',
                    "",
                    "entity",
                    None,
                    "",
                    "",
                ),
                (
                    "fact-orphan-1",
                    "ent-sec-eng",
                    "part of",
                    "ent-eng",
                    '["security engineering", "part of", "engineering"]',
                    "Security engineering is a branch of engineering.",
                    "entity",
                    None,
                    "",
                    "",
                ),
            ],
        )
        upsert_fact_sources(conn, [("fact-dangling", "c1"), ("fact-empty-def", "c1")])
        # Note: we deliberately DO NOT call upsert_fact_sources for fact-orphan-1

        violations = lint_graph_shacl(conn, auto_heal=False)
        shapes = {v.shape for v in violations}
        assert ALL_11_SHACL_SHAPES.issubset(shapes), f"Missing shapes: {ALL_11_SHACL_SHAPES - shapes}"

        focus_by_shape: dict[str, set[str]] = {}
        for v in violations:
            focus_by_shape.setdefault(v.shape, set()).add(v.focus_node)

        assert {"ent-no-pref", "ent-multi-pref"}.issubset(focus_by_shape["ConceptShape/prefLabel"])
        assert "ent-empty-def" in focus_by_shape["ConceptShape/definition"]
        assert "ent-s13" in focus_by_shape["ConceptShape/S13-disjointLabels"]
        assert "ent-hom-a|ent-hom-b" in focus_by_shape["ConceptShape/homographDifferentFrom"]
        assert {"fact-dangling", "fact-empty-def"}.issubset(focus_by_shape["StatementShape/closedWorld"])
        assert {
            "ent-sys->ent-dangling",
            "ent-s13|ent-sys",
            "ent-sys->ent-eng",
        }.issubset(focus_by_shape["EdgeShape/closedWorld"])
        assert {
            "ent-2cyc-a",
            "ent-2cyc-b",
            "ent-cyc-a",
            "ent-cyc-b",
            "ent-cyc-c",
        }.issubset(focus_by_shape["ConceptShape/broaderCycle"])
        assert "ent-s27-leaf|ent-s27-root" in focus_by_shape["ConceptShape/S27-disjointBroaderRelated"]
        assert {"ent-date", "ent-lit-port"}.issubset(focus_by_shape["ConceptShape/noScalarLiterals"])
        assert "ent-eng->ent-sec-eng" in focus_by_shape["ConceptShape/lexicalSubsumption"]
        assert "ent-v1->ent-v2" in focus_by_shape["ConceptShape/siblingNotBroader"]
        assert "fact-orphan-1" in focus_by_shape["ProvenanceShape/factSources"]

    finally:
        conn.close()


def test_shacl_auto_healing_complete(tmp_path: Path) -> None:
    """Verify that lint_graph_shacl(conn, auto_heal=True) deterministically heals all 11 SHACL shapes."""
    conn = open_yaw_db(tmp_path / "test_shacl_heal.db")
    try:
        upsert_chunk_pointers(
            conn,
            [
                (ChunkPointer("c1", "doc.md", 1, 5, "Doc 1"), "Documentation on versions and dates."),
                (ChunkPointer("c2", "doc2.md", 1, 5, "Doc 2"), "Documentation on homographs and caching."),
            ],
        )
        upsert_entities(
            conn,
            [
                ("ent-sys", "system", "A software system"),
                ("ent-no-pref", "orphan concept", "Concept with only altLabel"),
                ("ent-multi-pref", "primary label", "Concept with two prefLabels"),
                ("ent-empty-def", "undocumented service", ""),
                ("ent-s13", "caching layer", "Distributed cache"),
                ("ent-hom-a", "mercury", "Planet Mercury"),
                ("ent-hom-b", "mercury", "Element Hg"),
                ("ent-2cyc-a", "distributed consensus", "Consensus protocols"),
                ("ent-2cyc-b", "state machine replication", "Replicated state machines"),
                ("ent-cyc-a", "cryptanalysis", "Cryptographic analysis"),
                ("ent-cyc-b", "discrete mathematics", "Discrete math"),
                ("ent-cyc-c", "number theory", "Number theory"),
                ("ent-date", "2026-10-25", "A date string"),
                ("ent-lit-port", "8080", "Listen port literal"),
                ("ent-eng", "engineering", "Base engineering concept"),
                ("ent-sec-eng", "security engineering", "Specialized engineering concept"),
                ("ent-v1", "http 2", "HTTP version 2"),
                ("ent-v2", "http 3", "HTTP version 3"),
                ("ent-es-a", "edge grounded alpha", "Grounded only via edge_sources"),
                ("ent-es-b", "edge grounded beta", "Grounded only via edge_sources"),
                ("ent-ungrounded-1", "ghost alpha", "Ungrounded node A"),
                ("ent-ungrounded-2", "ghost beta", "Ungrounded node B"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("system", "ent-sys", "skos:prefLabel", "system", "c1"),
                ("undocumented service", "ent-empty-def", "skos:prefLabel", "undocumented service", "c1"),
                ("mercury", "ent-hom-a", "skos:prefLabel", "mercury", "c1"),
                ("mercury", "ent-hom-b", "skos:prefLabel", "mercury", "c2"),
                ("distributed consensus", "ent-2cyc-a", "skos:prefLabel", "distributed consensus", "c1"),
                ("state machine replication", "ent-2cyc-b", "skos:prefLabel", "state machine replication", "c1"),
                ("cryptanalysis", "ent-cyc-a", "skos:prefLabel", "cryptanalysis", "c1"),
                ("discrete mathematics", "ent-cyc-b", "skos:prefLabel", "discrete mathematics", "c1"),
                ("number theory", "ent-cyc-c", "skos:prefLabel", "number theory", "c1"),
                ("2026-10-25", "ent-date", "skos:prefLabel", "2026-10-25", "c1"),
                ("8080", "ent-lit-port", "skos:prefLabel", "8080", "c1"),
                ("engineering", "ent-eng", "skos:prefLabel", "engineering", "c1"),
                ("security engineering", "ent-sec-eng", "skos:prefLabel", "security engineering", "c1"),
                ("http 2", "ent-v1", "skos:prefLabel", "http 2", "c1"),
                ("http 3", "ent-v2", "skos:prefLabel", "http 3", "c1"),
                ("edge grounded alpha", "ent-es-a", "skos:prefLabel", "edge grounded alpha", "__healed__"),
                ("edge grounded beta", "ent-es-b", "skos:prefLabel", "edge grounded beta", "__healed__"),
                ("ghost alpha", "ent-ungrounded-1", "skos:prefLabel", "ghost alpha", "__healed__"),
                ("ghost beta", "ent-ungrounded-2", "skos:prefLabel", "ghost beta", "__healed__"),
            ],
        )

        # Seed Shape 1 (prefLabel) & Shape 3 (S13-disjointLabels) violations via raw SQL
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('orphan alias', 'ent-no-pref', 'skos:altLabel', 'orphan alias', 'c1')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('primary label', 'ent-multi-pref', 'skos:prefLabel', 'primary label', 'c1')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('secondary label', 'ent-multi-pref', 'skos:prefLabel', 'secondary label', 'c2')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('caching layer', 'ent-s13', 'skos:prefLabel', 'caching layer', 'c1')"
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('caching layer', 'ent-s13', 'skos:altLabel', 'caching layer', 'c2')"
        )

        # Seed Shape 6 (EdgeShape/closedWorld): dangling edge, related+differentFrom conflict, asymmetric skos:related
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-missing', 'fact', 1.0)")
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-s13', 'ent-sys', 'skos:related', 0.6)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-s13', 'skos:related', 0.6)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-s13', 'ent-sys', 'owl:differentFrom', 0.0)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-s13', 'owl:differentFrom', 0.0)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-multi-pref', 'skos:related', 0.7)"
        )

        # Seed Shape 7 (broaderCycle 2-cycle & 3-cycle + S27 transitive broader/related conflict)
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-2cyc-a', 'ent-2cyc-b', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-2cyc-b', 'ent-2cyc-a', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-a', 'ent-cyc-b', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-b', 'ent-cyc-c', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-c', 'ent-cyc-a', 'skos:broader', 0.8)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-a', 'ent-cyc-c', 'skos:related', 0.5)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-cyc-c', 'ent-cyc-a', 'skos:related', 0.5)"
        )

        # Seed Shape 9 (lexicalSubsumption) & Shape 10 (siblingNotBroader) via raw SQL
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-eng', 'ent-sec-eng', 'skos:broader', 0.9)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-v1', 'ent-v2', 'skos:broader', 0.5)"
        )

        # Ground ent-es-a -> ent-es-b via edge_sources in chunk 'c2' so ProvenanceShape/factSources can recover 'c2'
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-es-a', 'ent-es-b', 'fact', 1.0)")
        conn.execute(
            "INSERT INTO edge_sources (src_id, dst_id, kind, chunk_id) VALUES ('ent-es-a', 'ent-es-b', 'fact', 'c2')"
        )

        # Seed Shape 8 (noScalarLiterals):
        # (a) pure literal node 'ent-lit-port' with 0.5 fact edge, passage edge, and illegal skos:broader & skos:related
        # (b) orphan scalar concept 'ent-date' referenced as obj_kind='entity'
        upsert_facts(
            conn,
            [
                (
                    "fact-port-1",
                    "ent-sys",
                    "listens on port",
                    "ent-lit-port",
                    '["system", "listens on port", "8080"]',
                    "The system listens on port 8080.",
                    "literal",
                    "8080",
                    "rdfs:Literal",
                    "",
                ),
                (
                    "fact-date-1",
                    "ent-sys",
                    "released on",
                    "ent-date",
                    '["system", "released on", "2026-10-25"]',
                    "The system was released on 2026-10-25.",
                    "entity",
                    None,
                    "",
                    "",
                ),
                # Shape 5 (StatementShape/closedWorld): dangling endpoint & empty fact definition
                (
                    "fact-dangling-1",
                    "ent-sys",
                    "calls",
                    "ent-nonexistent",
                    '["system", "calls", "nonexistent"]',
                    "System calls nonexistent.",
                    "entity",
                    None,
                    "",
                    "",
                ),
                (
                    "fact-empty-def-1",
                    "ent-sys",
                    "caches in",
                    "ent-s13",
                    '["system", "caches in", "caching layer"]',
                    "",
                    "entity",
                    None,
                    "",
                    "",
                ),
                # Shape 11 (ProvenanceShape/factSources):
                # (a) recoverable fact whose endpoints have chunk 'c1' in entity_labels
                (
                    "fact-no-src",
                    "ent-sec-eng",
                    "relates to",
                    "ent-eng",
                    '["security engineering", "relates to", "engineering"]',
                    "Security engineering relates to engineering.",
                    "entity",
                    None,
                    "",
                    "",
                ),
                # (b) recoverable fact whose endpoints have chunk 'c2' only in edge_sources
                (
                    "fact-no-src-es",
                    "ent-es-a",
                    "communicates with",
                    "ent-es-b",
                    '["edge grounded alpha", "communicates with", "edge grounded beta"]',
                    "Edge grounded alpha communicates with edge grounded beta.",
                    "entity",
                    None,
                    "",
                    "",
                ),
                # (c) true orphan fact whose endpoints have no valid chunk in `chunks`
                (
                    "fact-truly-orphan",
                    "ent-ungrounded-1",
                    "links to",
                    "ent-ungrounded-2",
                    '["ghost alpha", "links to", "ghost beta"]',
                    "Ghost alpha links to ghost beta.",
                    "entity",
                    None,
                    "",
                    "",
                ),
            ],
        )
        upsert_fact_sources(
            conn,
            [
                ("fact-port-1", "c1"),
                ("fact-date-1", "c1"),
                ("fact-dangling-1", "c1"),
                ("fact-empty-def-1", "c1"),
            ],
        )
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('c1', 'ent-lit-port', 'passage', 1.0)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-lit-port', 'fact', 0.5)")
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-lit-port', 'ent-sys', 'skos:broader', 0.6)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-lit-port', 'ent-sys', 'skos:related', 0.6)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-lit-port', 'skos:related', 0.6)"
        )

        # Before auto-heal, all 11 SHACL shapes flag violations
        pre_violations = lint_graph_shacl(conn, auto_heal=False)
        pre_shapes = {v.shape for v in pre_violations}
        assert ALL_11_SHACL_SHAPES.issubset(pre_shapes), f"Missing pre-heal shapes: {ALL_11_SHACL_SHAPES - pre_shapes}"

        # Run deterministic auto-heal
        remaining = lint_graph_shacl(conn, auto_heal=True)
        assert remaining == [], f"Unexpected remaining violations: {remaining}"

        # 1. ConceptShape/prefLabel healed: ent-no-pref promoted, ent-multi-pref demoted extra prefLabel
        no_pref_kinds = [
            str(r[0])
            for r in conn.execute(
                "SELECT label_kind FROM entity_labels WHERE entity_id = 'ent-no-pref'"
            ).fetchall()
        ]
        assert no_pref_kinds == ["skos:prefLabel"]
        multi_pref_rows = {
            str(r[0]): str(r[1])
            for r in conn.execute(
                "SELECT label_norm, label_kind FROM entity_labels WHERE entity_id = 'ent-multi-pref'"
            ).fetchall()
        }
        assert multi_pref_rows["primary label"] == "skos:prefLabel"
        assert multi_pref_rows["secondary label"] == "skos:altLabel"

        # 2. ConceptShape/definition healed: ent-empty-def has a synthesized non-empty definition
        empty_def_row = conn.execute(
            "SELECT definition FROM entities WHERE entity_id = 'ent-empty-def'"
        ).fetchone()
        assert empty_def_row is not None and str(empty_def_row[0]).strip() != ""

        # 3. ConceptShape/S13-disjointLabels healed: ent-s13 has only skos:prefLabel for 'caching layer'
        s13_kinds = {
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT label_kind FROM entity_labels "
                "WHERE entity_id = 'ent-s13' AND label_norm = 'caching layer'"
            ).fetchall()
        }
        assert s13_kinds == {"skos:prefLabel"}

        # 4. ConceptShape/homographDifferentFrom healed: symmetric owl:differentFrom added between ent-hom-a & ent-hom-b
        hom_diff = {
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                "SELECT src_id, dst_id FROM edges WHERE kind = 'owl:differentFrom' "
                "AND src_id IN ('ent-hom-a', 'ent-hom-b')"
            ).fetchall()
        }
        assert hom_diff == {("ent-hom-a", "ent-hom-b"), ("ent-hom-b", "ent-hom-a")}

        # 5. StatementShape/closedWorld healed: dangling fact deleted, empty fact definition synthesized
        assert conn.execute("SELECT 1 FROM facts WHERE fact_id = 'fact-dangling-1'").fetchone() is None
        f_def_row = conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-empty-def-1'").fetchone()
        assert f_def_row is not None and str(f_def_row[0]) == "system caches in caching layer."

        # 6. EdgeShape/closedWorld healed: dangling edge deleted, related+differentFrom conflict resolved,
        # asymmetric skos:related symmetrized
        assert (
            conn.execute("SELECT 1 FROM edges WHERE src_id = 'ent-sys' AND dst_id = 'ent-missing'").fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM edges WHERE src_id = 'ent-s13' AND dst_id = 'ent-sys' AND kind = 'skos:related'"
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM edges WHERE src_id = 'ent-multi-pref' AND dst_id = 'ent-sys' AND kind = 'skos:related'"
            ).fetchone()
            is not None
        )

        # 7. ConceptShape/broaderCycle & S27 healed: 2-cycle and 3-cycle broken, conflicting skos:related removed
        cyc2_broader = {
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader' "
                "AND src_id IN ('ent-2cyc-a', 'ent-2cyc-b')"
            ).fetchall()
        }
        assert len(cyc2_broader) == 1
        cyc_broader = {
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader' "
                "AND src_id IN ('ent-cyc-a', 'ent-cyc-b', 'ent-cyc-c')"
            ).fetchall()
        }
        assert len(cyc_broader) == 2
        assert (
            conn.execute(
                "SELECT 1 FROM edges WHERE kind = 'skos:related' "
                "AND src_id IN ('ent-cyc-a', 'ent-cyc-c') AND dst_id IN ('ent-cyc-a', 'ent-cyc-c')"
            ).fetchone()
            is None
        )

        # 8a. Pure literal node (ent-lit-port) is preserved in entities with its 0.5 fact edge,
        # while its illegal skos:broader and skos:related edges were stripped
        port_ent = conn.execute("SELECT entity_id FROM entities WHERE entity_id = 'ent-lit-port'").fetchone()
        assert port_ent is not None
        port_fact_edge = conn.execute(
            "SELECT weight FROM edges WHERE src_id = 'ent-sys' AND dst_id = 'ent-lit-port' AND kind = 'fact'"
        ).fetchone()
        assert port_fact_edge is not None and abs(float(port_fact_edge[0]) - 0.5) < 1e-6
        port_skos_edges = conn.execute(
            "SELECT 1 FROM edges WHERE (src_id = 'ent-lit-port' OR dst_id = 'ent-lit-port') "
            "AND kind IN ('skos:broader', 'skos:related')"
        ).fetchone()
        assert port_skos_edges is None

        # 8b. Orphan scalar entity (ent-date) was removed from entities and fact-date-1 converted to literal
        date_ent = conn.execute("SELECT entity_id FROM entities WHERE entity_id = 'ent-date'").fetchone()
        assert date_ent is None
        fact_meta = get_facts_with_meta_by_ids(conn, ["fact-date-1"])
        assert fact_meta["fact-date-1"]["obj_kind"] == "literal"
        assert fact_meta["fact-date-1"]["obj_id"] is None
        assert fact_meta["fact-date-1"]["obj_literal_lex"] == "2026-10-25"

        # 9. Inverted broader edge was inverted: security engineering -> engineering
        eng_broader = conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader' "
            "AND src_id IN ('ent-eng', 'ent-sec-eng')"
        ).fetchall()
        assert len(eng_broader) == 1
        assert (str(eng_broader[0][0]), str(eng_broader[0][1])) == ("ent-sec-eng", "ent-eng")

        # 10. Sibling broader edge was replaced with symmetric skos:related
        related_edges = conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related' AND src_id IN ('ent-v1', 'ent-v2')"
        ).fetchall()
        assert len(related_edges) == 2
        related_pairs = {(str(r[0]), str(r[1])) for r in related_edges}
        assert related_pairs == {("ent-v1", "ent-v2"), ("ent-v2", "ent-v1")}

        # 11. ProvenanceShape/factSources:
        # (a) fact-no-src recovered chunk 'c1' from entity_labels
        fs_row = conn.execute("SELECT chunk_id FROM fact_sources WHERE fact_id = 'fact-no-src'").fetchone()
        assert fs_row is not None
        assert str(fs_row[0]) == "c1"
        # (b) fact-no-src-es recovered chunk 'c2' from edge_sources
        fs_es_row = conn.execute("SELECT chunk_id FROM fact_sources WHERE fact_id = 'fact-no-src-es'").fetchone()
        assert fs_es_row is not None
        assert str(fs_es_row[0]) == "c2"
        # (c) fact-truly-orphan had no valid chunk in `chunks` so it was deleted (never given '__healed__')
        assert conn.execute("SELECT 1 FROM facts WHERE fact_id = 'fact-truly-orphan'").fetchone() is None
        assert conn.execute("SELECT 1 FROM fact_sources WHERE chunk_id = '__healed__'").fetchone() is None

        # 12. Turtle serialization conforms
        ttl = serialize_graph_turtle(conn)
        assert "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ." in ttl
        assert "sh:conforms true" in ttl
        assert '"2026-10-25"' in ttl

    finally:
        conn.close()


def test_serialize_graph_turtle_literal_datatypes_and_lang_tags(tmp_path: Path) -> None:
    """Verify serialize_graph_turtle formatting for xsd:date CURIE (no angle brackets),
    full IRI (^^<http://...>), language-tagged literal (@en), and plain rdfs:Literal."""
    conn = open_yaw_db(tmp_path / "test_turtle_literals.db")
    try:
        upsert_chunk_pointers(
            conn,
            [(ChunkPointer("c1", "doc.md", 1, 5, "Doc"), "Literal formatting test.")],
        )
        upsert_entities(conn, [("ent-sys", "system", "Core system")])
        upsert_entity_labels(conn, [("system", "ent-sys", "skos:prefLabel", "system", "c1")])
        upsert_facts(
            conn,
            [
                (
                    "fact-xsd-date",
                    "ent-sys",
                    "launched on",
                    None,
                    '["system", "launched on", "2026-10-25"]',
                    "System launched on 2026-10-25.",
                    "literal",
                    "2026-10-25",
                    "xsd:date",
                    "",
                ),
                (
                    "fact-full-iri",
                    "ent-sys",
                    "has custom metric",
                    None,
                    '["system", "has custom metric", "42.5"]',
                    "System has custom metric 42.5.",
                    "literal",
                    "42.5",
                    "http://example.org/datatypes#millisecond",
                    "",
                ),
                (
                    "fact-lang-en",
                    "ent-sys",
                    "has motto",
                    None,
                    '["system", "has motto", "fast graph retrieval"]',
                    "System motto in English.",
                    "literal",
                    "fast graph retrieval",
                    "rdfs:Literal",
                    "en",
                ),
                (
                    "fact-plain-lit",
                    "ent-sys",
                    "listens on port",
                    None,
                    '["system", "listens on port", "8080"]',
                    "System listens on port 8080.",
                    "literal",
                    "8080",
                    "rdfs:Literal",
                    "",
                ),
            ],
        )
        upsert_fact_sources(
            conn,
            [
                ("fact-xsd-date", "c1"),
                ("fact-full-iri", "c1"),
                ("fact-lang-en", "c1"),
                ("fact-plain-lit", "c1"),
            ],
        )

        assert lint_graph_shacl(conn, auto_heal=False) == []
        ttl = serialize_graph_turtle(conn)
        assert 'rdf:object "2026-10-25"^^xsd:date' in ttl
        assert '^^<xsd:date>' not in ttl
        assert 'rdf:object "42.5"^^<http://example.org/datatypes#millisecond>' in ttl
        assert 'rdf:object "fast graph retrieval"@en' in ttl
        assert 'rdf:object "8080" ;' in ttl
        assert "^^rdfs:Literal" not in ttl
        assert "sh:conforms true" in ttl
    finally:
        conn.close()


def test_heal_8_scalar_entity_with_passage_edge_retains_obj_id_and_updates_fact_edge_weight(tmp_path: Path) -> None:
    """Verify Heal 8 converts obj_kind='entity' to 'literal' while retaining obj_id and updating
    the fact edge weight from 1.0 to 0.5 when the scalar node is grounded by a passage edge."""
    conn = open_yaw_db(tmp_path / "test_heal8_passage_scalar.db")
    try:
        upsert_chunk_pointers(
            conn,
            [(ChunkPointer("c1", "doc.md", 1, 5, "Doc"), "System released on 2026-10-25.")],
        )
        upsert_entities(
            conn,
            [
                ("ent-sys", "System", "Core system"),
                ("ent-date", "2026-10-25", "Release date"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("system", "ent-sys", "skos:prefLabel", "System", "c1"),
                ("2026-10-25", "ent-date", "skos:prefLabel", "2026-10-25", "c1"),
            ],
        )
        upsert_facts(
            conn,
            [
                (
                    "fact-date-1",
                    "ent-sys",
                    "released on",
                    "ent-date",
                    '["System", "released on", "2026-10-25"]',
                    "System released on 2026-10-25.",
                    "entity",
                    None,
                    "",
                    "",
                )
            ],
        )
        upsert_fact_sources(conn, [("fact-date-1", "c1")])
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('c1', 'ent-date', 'passage', 1.0)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-sys', 'ent-date', 'fact', 1.0)")
        conn.execute(
            "INSERT INTO edge_sources (src_id, dst_id, kind, chunk_id) VALUES ('ent-sys', 'ent-date', 'fact', 'c1')"
        )

        pre_v = lint_graph_shacl(conn, auto_heal=False)
        assert any(v.shape == "ConceptShape/noScalarLiterals" and v.focus_node == "ent-date" for v in pre_v)

        rem_v = lint_graph_shacl(conn, auto_heal=True)
        assert rem_v == []

        f_meta = get_facts_with_meta_by_ids(conn, ["fact-date-1"])["fact-date-1"]
        assert f_meta["obj_kind"] == "literal"
        assert f_meta["obj_id"] == "ent-date"
        assert f_meta["obj_literal_lex"] == "2026-10-25"

        edge_row = conn.execute(
            "SELECT weight FROM edges WHERE src_id = 'ent-sys' AND dst_id = 'ent-date' AND kind = 'fact'"
        ).fetchone()
        assert edge_row is not None
        assert abs(float(edge_row[0]) - 0.5) < 1e-6
    finally:
        conn.close()


def test_punctuated_surface_name_repairs_and_label_priority(tmp_path: Path) -> None:
    """Verify punctuated surface names ('Apple Inc.', 'Python 3.10') preserve canonical
    text_processing norms across upsert_entity_labels, apply_shacl_repairs, Heal 1, and inspect_focus_node."""
    conn = open_yaw_db(tmp_path / "yaw_punct_repairs.db")
    try:
        upsert_chunk_pointers(
            conn,
            [(ChunkPointer("c1", "doc.md", 1, 5, "Doc"), "Apple Inc. and Python 3.10.")],
        )

        # 1. upsert_entity_labels: punctuated raw norm ('apple inc.') followed by
        # canonical text_processing norm ('apple inc')
        upsert_entities(
            conn,
            [
                ("entity-apple", "Apple Inc.", "Technology company."),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("apple inc.", "entity-apple", "skos:prefLabel", "Apple Inc.", "c1"),
                ("apple inc", "entity-apple", "skos:prefLabel", "Apple Inc", "c1"),
            ],
        )
        assert lint_graph_shacl(conn, auto_heal=False) == []
        apple_kinds = {
            str(r[0]): str(r[1])
            for r in conn.execute(
                "SELECT label_norm, label_kind FROM entity_labels WHERE entity_id = 'entity-apple'"
            ).fetchall()
        }
        assert apple_kinds["apple inc"] == "skos:prefLabel"
        assert apple_kinds["apple inc."] == "skos:altLabel"

        # 2. inspect_focus_node resolves 'apple inc' even though entities.name is 'Apple Inc.'
        info = inspect_focus_node(conn, "apple inc")
        assert info["entity"] is not None
        assert info["entity"]["entity_id"] == "entity-apple"
        assert info["entity"]["name"] == "Apple Inc."

        # 3. apply_shacl_repairs demote_to_alt_label keeps canonical 'apple inc' over 'aardvark'
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
            "VALUES ('aardvark', 'entity-apple', 'skos:prefLabel', 'Aardvark', 'c1')"
        )
        applied = apply_shacl_repairs(
            conn,
            [{"focus_node": "entity-apple", "action": "demote_to_alt_label", "target_node": "", "value": ""}],
        )
        assert applied == 1
        assert lint_graph_shacl(conn, auto_heal=False) == []

        # 4. Heal 1 promotes 'python 3.10' over alphabetically earlier 'a_alias' and preserves 'Python 3.10'
        upsert_entities(
            conn,
            [
                ("entity-py", "Python 3.10", "Programming language release."),
            ],
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) VALUES "
            "('a_alias', 'entity-py', 'skos:altLabel', 'a_alias', 'c1'), "
            "('python 3.10', 'entity-py', 'skos:altLabel', 'Python 3.10', 'c1')"
        )
        assert lint_graph_shacl(conn, auto_heal=True) == []
        py_name = conn.execute("SELECT name FROM entities WHERE entity_id = 'entity-py'").fetchone()
        assert py_name is not None and str(py_name[0]) == "Python 3.10"
        py_pref = conn.execute(
            "SELECT label_norm FROM entity_labels WHERE entity_id = 'entity-py' AND label_kind = 'skos:prefLabel'"
        ).fetchall()
        assert [str(r[0]) for r in py_pref] == ["python 3.10"]
    finally:
        conn.close()
