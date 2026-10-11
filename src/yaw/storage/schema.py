"""SQLite database schema, view definitions, and schema integrity validation."""

from __future__ import annotations

import sqlite3
from typing import Any

__all__ = [
    "UNIFIED_EDGES_VIEW_TEMPLATE",
    "UNIFIED_NODES_VIEW_SQL",
    "ensure_schema_tables",
    "ensure_schema_views",
    "verify_schema_integrity",
]


def ensure_schema_tables(conn: sqlite3.Connection) -> None:
    """Create aliases, canon_verdicts, compiled_nodes, compiled_relations, and dynamic ontology catalog tables idempotently."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aliases (
            alias_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('entity', 'predicate')),
            canonical_id TEXT NOT NULL,
            method TEXT NOT NULL CHECK(method IN ('lexical', 'auto', 'llm', 'compile', 'exact', 'alias', 'new', 'ontology')),
            score REAL NOT NULL DEFAULT 1.0 CHECK(score >= 0.0 AND score <= 1.0),
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'contested')),
            source_leaf_id TEXT,
            resolved_at REAL DEFAULT (unixepoch()),
            evidence TEXT,
            PRIMARY KEY (alias_id, kind)
        )
        """
    )
    # Idempotent migration for existing aliases table lacking 'ontology' method
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='aliases'"
    )
    row = cursor.fetchone()
    if row and row[0] and "'ontology'" not in row[0]:
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.execute("DROP TABLE IF EXISTS aliases_new;")
        conn.execute(
            """
            CREATE TABLE aliases_new (
                alias_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('entity', 'predicate')),
                canonical_id TEXT NOT NULL,
                method TEXT NOT NULL CHECK(method IN ('lexical', 'auto', 'llm', 'compile', 'exact', 'alias', 'new', 'ontology')),
                score REAL NOT NULL DEFAULT 1.0 CHECK(score >= 0.0 AND score <= 1.0),
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'contested')),
                source_leaf_id TEXT,
                resolved_at REAL DEFAULT (unixepoch()),
                evidence TEXT,
                PRIMARY KEY (alias_id, kind)
            )
            """
        )
        cols = [r[1] for r in conn.execute("PRAGMA table_info(aliases)").fetchall()]
        common_cols = [
            c
            for c in cols
            if c
            in (
                "alias_id",
                "kind",
                "canonical_id",
                "method",
                "score",
                "status",
                "source_leaf_id",
                "resolved_at",
                "evidence",
            )
        ]
        col_str = ", ".join(common_cols)
        conn.execute(
            f"INSERT INTO aliases_new ({col_str}) SELECT {col_str} FROM aliases"
        )
        conn.execute("DROP VIEW IF EXISTS unified_edges")
        conn.execute("DROP VIEW IF EXISTS unified_nodes")
        conn.execute("DROP TABLE aliases")
        conn.execute("ALTER TABLE aliases_new RENAME TO aliases")
        conn.execute("PRAGMA foreign_keys=ON;")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON aliases(canonical_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_aliases_lookup ON aliases(alias_id, kind, status)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS canon_verdicts (
            surface_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            cosine REAL NOT NULL,
            lexical REAL NOT NULL,
            verdict TEXT NOT NULL CHECK(verdict IN ('match', 'distinct')),
            phase TEXT NOT NULL CHECK(phase IN ('ingest', 'compile')),
            decided_at REAL NOT NULL,
            rationale TEXT,
            PRIMARY KEY (surface_id, kind, candidate_id, phase)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_verdicts_lookup ON canon_verdicts(surface_id, kind, phase)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_nodes (
            node_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            definition TEXT NOT NULL DEFAULT '',
            parent_hub_id TEXT,
            compiled_at REAL NOT NULL DEFAULT (unixepoch())
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compiled_nodes_ns ON compiled_nodes(namespace)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_relations (
            relation_id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            source_rule TEXT,
            compiled_at REAL NOT NULL DEFAULT (unixepoch())
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compiled_rel_sub ON compiled_relations(subject_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compiled_rel_obj ON compiled_relations(object_id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_ontologies (
            ontology_id TEXT PRIMARY KEY,
            uri_prefix TEXT NOT NULL,
            source_uri TEXT,
            is_default INT NOT NULL DEFAULT 0,
            loaded_at REAL NOT NULL DEFAULT (unixepoch())
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_classes (
            curie TEXT PRIMARY KEY,
            class_uri TEXT NOT NULL UNIQUE,
            ontology_id TEXT NOT NULL,
            label TEXT NOT NULL,
            definition TEXT DEFAULT '',
            parent_class_uri TEXT,
            depth INT NOT NULL DEFAULT 0,
            is_root INT NOT NULL DEFAULT 0,
            epistemic_type TEXT NOT NULL DEFAULT 'class' CHECK(epistemic_type IN ('class', 'concept'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dynamic_classes_uri ON dynamic_classes(class_uri)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dynamic_classes_parent ON dynamic_classes(parent_class_uri)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_class_parents (
            curie TEXT NOT NULL,
            parent_class_uri TEXT NOT NULL,
            PRIMARY KEY (curie, parent_class_uri)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dynamic_class_parents_parent ON dynamic_class_parents(parent_class_uri)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_properties (
            curie TEXT PRIMARY KEY,
            property_uri TEXT NOT NULL UNIQUE,
            ontology_id TEXT NOT NULL,
            label TEXT NOT NULL,
            definition TEXT DEFAULT '',
            parent_property_uri TEXT,
            is_transitive INT NOT NULL DEFAULT 0,
            is_symmetric INT NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dynamic_properties_uri ON dynamic_properties(property_uri)"
    )

    dc_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(dynamic_classes)").fetchall()
    }
    if dc_cols and "definition" not in dc_cols:
        conn.execute(
            "ALTER TABLE dynamic_classes ADD COLUMN definition TEXT DEFAULT ''"
        )

    dp_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(dynamic_properties)").fetchall()
    }
    if dp_cols and "definition" not in dp_cols:
        conn.execute(
            "ALTER TABLE dynamic_properties ADD COLUMN definition TEXT DEFAULT ''"
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO runtime_settings (key, value) VALUES ('open_ie', '1')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO runtime_settings (key, value) VALUES ('target_ontology', 'internal')"
    )


UNIFIED_EDGES_VIEW_TEMPLATE = """
CREATE VIEW unified_edges AS
SELECT
    COALESCE(a_src.canonical_id, w.source_uri) AS src_id,
    COALESCE(a_dst.canonical_id, w.target_uri) AS dst_id,
    'wikilink' AS rel_type,
    1.0 AS weight,
    w.source_uri AS source_chunk_id
FROM wikilinks w
LEFT JOIN aliases a_src
    ON a_src.alias_id = w.source_uri
    AND a_src.kind = 'entity'
    AND a_src.status = 'active'
LEFT JOIN aliases a_dst
    ON a_dst.alias_id = w.target_uri
    AND a_dst.kind = 'entity'
    AND a_dst.status = 'active'
UNION ALL
SELECT
    COALESCE(a_sub.canonical_id, lr.subject_id) AS src_id,
    COALESCE(a_obj.canonical_id, lr.object_id) AS dst_id,
    COALESCE(a_pred.canonical_id, lr.predicate) AS rel_type,
    {weight_col},
    lr.source_leaf_id AS source_chunk_id
FROM leaf_relations lr
LEFT JOIN aliases a_sub
    ON a_sub.alias_id = lr.subject_id
    AND a_sub.kind = 'entity'
    AND a_sub.status = 'active'
LEFT JOIN aliases a_obj
    ON a_obj.alias_id = lr.object_id
    AND a_obj.kind = 'entity'
    AND a_obj.status = 'active'
LEFT JOIN aliases a_pred
    ON (a_pred.alias_id = lr.predicate_id
        OR a_pred.alias_id = lr.predicate 
        OR a_pred.alias_id = replace(lower(lr.predicate), ' ', '-'))
    AND a_pred.kind = 'predicate'
    AND a_pred.status = 'active'
WHERE (
    (SELECT COALESCE(value, '1') FROM runtime_settings WHERE key = 'open_ie') = '1'
    OR a_pred.canonical_id IS NOT NULL
    OR lr.predicate IN (
        'partOf', 'subClassOf', 'manifestsIn', 'mentions',
        'causes', 'prevents', 'supports', 'refutes', 'contradicts',
        'supersedes', 'produces', 'derivedFrom', 'cites', 'instanceOf',
        'manifests_in', 'defined_in', 'sameAs', 'aliasOf', 'depends_on',
        'precedes', 'applies_to', 'asserts'
    )
)
UNION ALL
SELECT
    COALESCE(a_sub.canonical_id, cr.subject_id) AS src_id,
    COALESCE(a_obj.canonical_id, cr.object_id) AS dst_id,
    COALESCE(a_pred.canonical_id, cr.predicate) AS rel_type,
    cr.weight AS weight,
    'compiled:' || cr.relation_id AS source_chunk_id
FROM compiled_relations cr
LEFT JOIN aliases a_sub
    ON a_sub.alias_id = cr.subject_id
    AND a_sub.kind = 'entity'
    AND a_sub.status = 'active'
LEFT JOIN aliases a_obj
    ON a_obj.alias_id = cr.object_id
    AND a_obj.kind = 'entity'
    AND a_obj.status = 'active'
LEFT JOIN aliases a_pred
    ON (a_pred.alias_id = cr.predicate 
        OR a_pred.alias_id = replace(lower(cr.predicate), ' ', '-'))
    AND a_pred.kind = 'predicate'
    AND a_pred.status = 'active'
UNION ALL
SELECT
    dc.curie AS src_id,
    COALESCE(p.curie, dc.parent_class_uri) AS dst_id,
    CASE WHEN dc.epistemic_type = 'concept' THEN 'broader' ELSE 'subClassOf' END AS rel_type,
    1.0 AS weight,
    'dynamic_ontology:' || dc.ontology_id AS source_chunk_id
FROM dynamic_classes dc
LEFT JOIN dynamic_classes p ON p.class_uri = dc.parent_class_uri OR p.curie = dc.parent_class_uri
WHERE dc.parent_class_uri IS NOT NULL
UNION ALL
SELECT
    dp.curie AS src_id,
    COALESCE(p.curie, dp.parent_property_uri) AS dst_id,
    'subPropertyOf' AS rel_type,
    1.0 AS weight,
    'dynamic_ontology:' || dp.ontology_id AS source_chunk_id
FROM dynamic_properties dp
LEFT JOIN dynamic_properties p ON p.property_uri = dp.parent_property_uri OR p.curie = dp.parent_property_uri
WHERE dp.parent_property_uri IS NOT NULL
"""

UNIFIED_NODES_VIEW_SQL = """
CREATE VIEW unified_nodes AS
SELECT
    entity_uri AS node_id,
    heading AS name,
    'chunk' AS namespace,
    rel_path AS definition
FROM note_chunks
UNION ALL
SELECT DISTINCT
    COALESCE(a.canonical_id, ln.node_id) AS node_id,
    ln.name,
    CASE
        WHEN a.canonical_id IS NOT NULL AND instr(a.canonical_id, ':') > 0
        THEN substr(a.canonical_id, 1, instr(a.canonical_id, ':') - 1)
        ELSE ln.namespace
    END AS namespace,
    ln.definition
FROM leaf_nodes ln
LEFT JOIN aliases a
    ON a.alias_id = ln.node_id
    AND a.kind = 'entity'
    AND a.status = 'active'
UNION
SELECT
    node_id,
    name,
    namespace,
    definition
FROM compiled_nodes
UNION ALL
SELECT
    curie AS node_id,
    label AS name,
    ontology_id AS namespace,
    COALESCE(NULLIF(definition, ''), label) AS definition
FROM dynamic_classes
"""


def ensure_schema_views(conn: sqlite3.Connection) -> None:
    """Ensure aliases tables exist, then create/replace unified_edges and unified_nodes views."""
    ensure_schema_tables(conn)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    if "wikilinks" in tables and "leaf_relations" in tables:
        conn.execute("DROP VIEW IF EXISTS unified_edges")
        lr_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(leaf_relations)").fetchall()
        }
        if "predicate_id" not in lr_cols:
            conn.execute("ALTER TABLE leaf_relations ADD COLUMN predicate_id TEXT")
        weight_col = "lr.weight AS weight" if "weight" in lr_cols else "1.0 AS weight"
        conn.execute(UNIFIED_EDGES_VIEW_TEMPLATE.format(weight_col=weight_col))

    if "note_chunks" in tables:
        nc_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(note_chunks)").fetchall()
        }
        migrated_nc = False
        if "byte_offset" in nc_cols:
            conn.execute("ALTER TABLE note_chunks DROP COLUMN byte_offset;")
            migrated_nc = True
        if "byte_length" in nc_cols:
            conn.execute("ALTER TABLE note_chunks DROP COLUMN byte_length;")
            migrated_nc = True
        if migrated_nc:
            try:
                conn.execute("VACUUM;")
            except sqlite3.OperationalError:
                pass

    if "note_chunks" in tables and "leaf_nodes" in tables:
        conn.execute("DROP VIEW IF EXISTS unified_nodes")
        conn.execute(UNIFIED_NODES_VIEW_SQL)


def verify_schema_integrity(conn: sqlite3.Connection) -> dict[str, Any]:
    """Validate existence and consistency of tables, indexes, and views."""
    master_rows = conn.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view', 'index')"
    ).fetchall()
    tables = {r[1] for r in master_rows if r[0] == "table"}
    views = {r[1] for r in master_rows if r[0] == "view"}
    indexes = {r[1] for r in master_rows if r[0] == "index"}

    required_tables = ["aliases", "canon_verdicts"]
    required_views = ["unified_edges", "unified_nodes"]

    missing_tables = [t for t in required_tables if t not in tables]
    missing_views = [v for v in required_views if v not in views]

    counts: dict[str, int] = {}
    for t in (
        "aliases",
        "canon_verdicts",
        "leaf_nodes",
        "leaf_relations",
        "wikilinks",
        "note_chunks",
    ):
        if t in tables:
            counts[t] = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]

    integrity = "ok"
    try:
        row = conn.execute("PRAGMA integrity_check(1)").fetchone()
        integrity = row[0] if row else "unknown"
    except sqlite3.Error as e:
        integrity = str(e)

    healthy = (len(missing_tables) == 0) and (integrity == "ok")

    return {
        "healthy": healthy,
        "integrity": integrity,
        "tables": sorted(tables),
        "views": sorted(views),
        "indexes": sorted(indexes),
        "counts": counts,
        "missing_tables": missing_tables,
        "missing_views": missing_views,
    }
