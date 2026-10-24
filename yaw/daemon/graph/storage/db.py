"""Unified yaw.db SQLite connection lifecycle, schema DDL, and metadata key-value store."""

import json
import sqlite3
import time
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

import sqlite_vec
from cocoindex.connectors.sqlite import ManagedConnection
from cocoindex.connectors.sqlite import connect as coco_sqlite_connect

SQL_BATCH_SIZE = 500
type YawConnection = sqlite3.Connection

_CREATE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        rel_posix_path TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        ast_island_start INTEGER NOT NULL DEFAULT 0,
        ast_island_end INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(rel_posix_path)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_path_end ON chunks(rel_posix_path, end_line)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        chunk_id UNINDEXED,
        title,
        body,
        content='',
        contentless_delete=1,
        contentless_unindexed=1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS openie_chunks (
        chunk_id TEXT PRIMARY KEY,
        entities_json TEXT NOT NULL,
        triples_json TEXT NOT NULL,
        structured_triples_json TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        definition TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
    """
    CREATE TABLE IF NOT EXISTS entity_labels (
        label_norm TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        label_kind TEXT NOT NULL CHECK (label_kind IN ('skos:prefLabel', 'skos:altLabel', 'skos:hiddenLabel')),
        raw_label TEXT NOT NULL DEFAULT '',
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (label_norm, entity_id, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entity_labels_norm ON entity_labels(label_norm)",
    "CREATE INDEX IF NOT EXISTS idx_entity_labels_entity ON entity_labels(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_entity_labels_chunk ON entity_labels(chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS facts (
        fact_id TEXT PRIMARY KEY,
        subj_id TEXT NOT NULL,
        pred TEXT NOT NULL,
        obj_kind TEXT NOT NULL DEFAULT 'entity' CHECK (obj_kind IN ('entity', 'literal')),
        obj_id TEXT,
        obj_literal_lex TEXT,
        obj_datatype TEXT NOT NULL DEFAULT '',
        obj_lang TEXT NOT NULL DEFAULT '',
        raw_triple_json TEXT NOT NULL,
        definition TEXT NOT NULL DEFAULT '',
        CHECK (
            (obj_kind = 'entity' AND obj_id IS NOT NULL AND TRIM(obj_id) != '' AND obj_literal_lex IS NULL)
            OR (obj_kind = 'literal' AND obj_id IS NULL AND obj_literal_lex IS NOT NULL AND TRIM(obj_literal_lex) != '')
        )
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_subj ON facts(subj_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_obj ON facts(obj_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_pair ON facts(subj_id, obj_id)",
    """
    CREATE TABLE IF NOT EXISTS fact_sources (
        fact_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (fact_id, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_fact_sources_chunk ON fact_sources(chunk_id)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
        fact_id UNINDEXED,
        text,
        content='',
        contentless_delete=1,
        contentless_unindexed=1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('passage', 'fact', 'skos:related', 'skos:broader', 'owl:differentFrom')),
        weight REAL NOT NULL,
        PRIMARY KEY (src_id, dst_id, kind)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, kind)",
    """
    CREATE TABLE IF NOT EXISTS edge_sources (
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (src_id, dst_id, kind, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edge_sources_chunk ON edge_sources(chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS chunk_vec (
        chunk_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_vec (
        entity_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_vec (
        fact_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
    )
    """,
)


def batched(items: Sequence[str], size: int = SQL_BATCH_SIZE) -> Iterator[Sequence[str]]:
    """Yield slices of items with at most size elements per SQLite query batch."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


@contextmanager
def write_tx(conn: sqlite3.Connection) -> Generator[None]:
    """Acquire an immediate write transaction lock or join an active outer transaction."""
    owns_tx = not conn.in_transaction
    if owns_tx:
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        if owns_tx and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        if owns_tx and conn.in_transaction:
            conn.execute("COMMIT")


def _configure_connection_pragmas(
    conn: sqlite3.Connection,
    *,
    enable_wal: bool = False,
    enable_foreign_keys: bool = False,
    busy_timeout_ms: int = 30000,
) -> None:
    conn.row_factory = sqlite3.Row
    if enable_wal:
        conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    if enable_foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)};")


def open_managed_yaw_db(db_path: Path | str) -> ManagedConnection:
    """Open or create a unified yaw.db via CocoIndex ManagedConnection with FIFO RWLock and sqlite-vec."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    managed = coco_sqlite_connect(db_str, timeout=30.0, load_vec=True)
    _configure_connection_pragmas(managed._conn)
    with managed.transaction() as conn:
        initialize_schema(conn)
    return managed


def open_yaw_db(db_path: Path | str) -> sqlite3.Connection:
    """Open or create a unified yaw.db SQLite connection with WAL and sqlite-vec loaded."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_str, timeout=30.0, check_same_thread=False)
    conn.isolation_level = None
    _configure_connection_pragmas(
        conn,
        enable_wal=(db_str != ":memory:"),
        enable_foreign_keys=True,
    )

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    initialize_schema(conn)
    return conn


def open_readonly_yaw_db(
    db_path: Path | str,
    *,
    timeout: float = 30.0,
    backoff: Sequence[float] = (),
) -> sqlite3.Connection:
    """Open a read-only WAL snapshot connection ('file:...?mode=ro') with busy_timeout and sqlite-vec configured."""
    resolved = Path(db_path).resolve()
    busy_ms = max(0, int(timeout * 1000))
    for i in range(len(backoff) + 1):
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                f"file:{resolved.as_posix()}?mode=ro",
                uri=True,
                timeout=timeout,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {busy_ms};")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            return conn
        except sqlite3.Error:
            if conn is not None:
                with suppress(sqlite3.Error):
                    conn.close()
            if i >= len(backoff):
                raise
            time.sleep(backoff[i])
    raise sqlite3.OperationalError(f"Failed to open read-only SQLite database: {resolved}")


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Initialize all unified schema tables and indices in yaw.db with in-place migrations."""
    with write_tx(conn):
        for stmt in _CREATE_SCHEMA_STATEMENTS:
            conn.execute(stmt)

        # Auto-migrate legacy content_hash column on chunks if present
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "content_hash" in chunk_cols:
            conn.execute("ALTER TABLE chunks DROP COLUMN content_hash")
        if "ast_island_start" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN ast_island_start INTEGER NOT NULL DEFAULT 0")
        if "ast_island_end" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN ast_island_end INTEGER NOT NULL DEFAULT 0")

        # Auto-migrate openie_chunks.structured_triples_json
        openie_cols = {row[1] for row in conn.execute("PRAGMA table_info(openie_chunks)").fetchall()}
        if "structured_triples_json" not in openie_cols:
            conn.execute("ALTER TABLE openie_chunks ADD COLUMN structured_triples_json TEXT NOT NULL DEFAULT '[]'")

        # Auto-migrate entities.definition and drop legacy UNIQUE constraint on entities.name
        entity_cols = {row[1] for row in conn.execute("PRAGMA table_info(entities)").fetchall()}
        if "definition" not in entity_cols:
            conn.execute("ALTER TABLE entities ADD COLUMN definition TEXT NOT NULL DEFAULT ''")
        entity_ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'entities'"
        ).fetchone()
        if entity_ddl_row is not None and "UNIQUE" in str(entity_ddl_row[0]).upper():
            conn.execute(
                """
                CREATE TABLE _entities_migrated (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    definition TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "INSERT INTO _entities_migrated (entity_id, name, definition) SELECT entity_id, name, "
                "definition FROM entities"
            )
            conn.execute("DROP TABLE entities")
            conn.execute("ALTER TABLE _entities_migrated RENAME TO entities")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")

        # Auto-migrate facts table to support mutually exclusive entity vs literal objects
        fact_cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "definition" not in fact_cols:
            conn.execute("ALTER TABLE facts ADD COLUMN definition TEXT NOT NULL DEFAULT ''")
        facts_ddl_row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'facts'").fetchone()
        if facts_ddl_row is not None and (
            "obj_kind" not in fact_cols or "obj_literal_lex" not in str(facts_ddl_row[0])
        ):
            conn.execute(
                """
                CREATE TABLE _facts_migrated (
                    fact_id TEXT PRIMARY KEY,
                    subj_id TEXT NOT NULL,
                    pred TEXT NOT NULL,
                    obj_kind TEXT NOT NULL DEFAULT 'entity' CHECK (obj_kind IN ('entity', 'literal')),
                    obj_id TEXT,
                    obj_literal_lex TEXT,
                    obj_datatype TEXT NOT NULL DEFAULT '',
                    obj_lang TEXT NOT NULL DEFAULT '',
                    raw_triple_json TEXT NOT NULL,
                    definition TEXT NOT NULL DEFAULT '',
                    CHECK (
                        (obj_kind = 'entity' AND obj_id IS NOT NULL AND TRIM(obj_id) != '' AND obj_literal_lex IS NULL)
                        OR (
                            obj_kind = 'literal'
                            AND obj_id IS NULL
                            AND obj_literal_lex IS NOT NULL
                            AND TRIM(obj_literal_lex) != ''
                        )
                    )
                )
                """
            )
            conn.execute(
                """
                INSERT INTO _facts_migrated (
                    fact_id, subj_id, pred, obj_kind, obj_id, obj_literal_lex,
                    obj_datatype, obj_lang, raw_triple_json, definition
                )
                SELECT
                    fact_id, subj_id, pred, 'entity', obj_id, NULL,
                    '', '', raw_triple_json, COALESCE(definition, '')
                FROM facts
                WHERE obj_id IS NOT NULL
                """
            )
            conn.execute("DROP TABLE facts")
            conn.execute("ALTER TABLE _facts_migrated RENAME TO facts")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subj ON facts(subj_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_obj ON facts(obj_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_pair ON facts(subj_id, obj_id)")

        # Auto-migrate edges CHECK constraint and legacy 'synonymy' kind -> 'skos:related', plus 'skos:broader'
        edges_ddl_row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'edges'").fetchone()
        if edges_ddl_row is not None and (
            "skos:related" not in str(edges_ddl_row[0]) or "skos:broader" not in str(edges_ddl_row[0])
        ):
            conn.execute(
                """
                CREATE TABLE _edges_migrated (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (
                        kind IN ('passage', 'fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
                    ),
                    weight REAL NOT NULL,
                    PRIMARY KEY (src_id, dst_id, kind)
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO _edges_migrated (src_id, dst_id, kind, weight)
                SELECT
                    src_id,
                    dst_id,
                    CASE WHEN kind = 'synonymy' THEN 'skos:related' ELSE kind END,
                    weight
                FROM edges
                WHERE kind IN ('passage', 'fact', 'synonymy', 'skos:related', 'skos:broader', 'owl:differentFrom')
                """
            )
            conn.execute("DROP TABLE edges")
            conn.execute("ALTER TABLE _edges_migrated RENAME TO edges")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, kind)")

        # Auto-backfill fact_sources from edge_sources for legacy entity-object facts
        fact_sources_cnt = int(conn.execute("SELECT COUNT(*) FROM fact_sources").fetchone()[0])
        if fact_sources_cnt == 0:
            conn.execute(
                """
                INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id)
                SELECT f.fact_id, es.chunk_id
                FROM facts f
                INNER JOIN edge_sources es
                  ON es.kind = 'fact' AND es.src_id = f.subj_id AND es.dst_id = f.obj_id
                WHERE f.obj_kind = 'entity' AND f.obj_id IS NOT NULL
                """
            )

        # Auto-backfill entity_labels for any legacy entities without labels
        missing_label_rows = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el WHERE el.entity_id = e.entity_id
            )
            """
        ).fetchall()
        if missing_label_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                VALUES (?, ?, 'skos:prefLabel', ?, '__legacy__')
                """,
                ((str(r[1]).strip().lower(), str(r[0]), str(r[1])) for r in missing_label_rows),
            )

        # Auto-backfill fact_fts if facts exist but fact_fts is empty
        fact_count = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        if fact_count > 0:
            fts_count = int(conn.execute("SELECT COUNT(*) FROM fact_fts").fetchone()[0])
            if fts_count == 0:
                rows = conn.execute("SELECT fact_id, pred, raw_triple_json FROM facts").fetchall()
                fts_items: list[tuple[str, str]] = []
                for row in rows:
                    fid = str(row[0])
                    pred = str(row[1])
                    try:
                        items = json.loads(row[2])
                        fts_items.append((fid, f"{items[0]} {pred} {items[2]}"))
                    except Exception:
                        fts_items.append((fid, pred))
                if fts_items:
                    conn.executemany("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)", fts_items)


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Retrieve a metadata value by key from the meta table, or None if not present."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row is not None else None


def set_meta(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    """Upsert or delete (when value is None) a metadata entry in the meta table."""
    with write_tx(conn):
        if value is None:
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            conn.execute(
                """
                INSERT INTO meta (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
