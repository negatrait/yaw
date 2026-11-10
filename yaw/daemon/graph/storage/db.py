"""Unified yaw.db SQLite connection lifecycle, schema DDL, and metadata key-value store."""

import re
import sqlite3
import time
import uuid
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

import cocoindex as coco
import sqlite_vec
from cocoindex.connectors.sqlite import ManagedConnection
from cocoindex.connectors.sqlite import connect as coco_sqlite_connect

from yaw.core import resolve_coco_state_root as resolve_coco_state_root

SQL_BATCH_SIZE = 500
DEFAULT_VEC_DIMENSION = 768
_VEC0_PK_COLUMNS: dict[str, str] = {
    "chunk_vec": "chunk_id",
    "entity_vec": "entity_id",
    "fact_vec": "fact_id",
}
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_VEC0_DIM_RE = re.compile(r"USING\s+vec0\s*\(.*?\bfloat\[(\d+)\]", re.IGNORECASE | re.DOTALL)
GRAPH_DB = coco.ContextKey[ManagedConnection]("yaw/graph_db")

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
    CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        name TEXT NOT NULL COLLATE NOCASE,
        definition TEXT NOT NULL DEFAULT '',
        confidence REAL
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
        confidence REAL,
        CHECK (
            (obj_kind = 'entity' AND obj_id IS NOT NULL AND TRIM(obj_id) != '' AND obj_literal_lex IS NULL)
            OR (
                obj_kind = 'literal'
                AND (obj_id IS NULL OR (TRIM(obj_id) != '' AND obj_id != subj_id))
                AND obj_literal_lex IS NOT NULL
                AND TRIM(obj_literal_lex) != ''
            )
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
        llm_prob REAL,
        vec_coherence REAL,
        atomic_confidence REAL,
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
        kind TEXT NOT NULL CHECK (
            kind IN ('passage', 'fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
        ),
        weight REAL NOT NULL,
        reasoning TEXT NOT NULL DEFAULT '',
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
        llm_prob REAL,
        vec_coherence REAL,
        atomic_confidence REAL,
        PRIMARY KEY (src_id, dst_id, kind, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edge_sources_chunk ON edge_sources(chunk_id)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
        chunk_id TEXT PRIMARY KEY,
        embedding float[768] distance_metric=cosine
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS entity_vec USING vec0(
        entity_id TEXT PRIMARY KEY,
        embedding float[768] distance_metric=cosine
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fact_vec USING vec0(
        fact_id TEXT PRIMARY KEY,
        embedding float[768] distance_metric=cosine
    )
    """,
)


def _validate_identifier(value: str) -> str:
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"Invalid SQLite identifier: {value!r}")
    return value


def vec0_table_dimension(conn: sqlite3.Connection, table_name: str) -> int | None:
    """Return the float[N] dimension of an existing vec0 virtual table, or None if not a vec0 table."""
    _validate_identifier(table_name)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if row is None or not row[0]:
        return None
    match = _VEC0_DIM_RE.search(str(row[0]))
    return int(match.group(1)) if match is not None else None


def ensure_vec0_table(
    conn: sqlite3.Connection,
    table_name: str,
    pk_col: str,
    dim: int | None,
    *,
    column: str = "embedding",
    metric: str = "cosine",
) -> None:
    """Ensure a sqlite-vec vec0 virtual table exists with the requested primary key and dimension."""
    _validate_identifier(table_name)
    _validate_identifier(pk_col)
    _validate_identifier(column)
    if metric != "cosine":
        raise ValueError(f"Unsupported vec0 distance metric: {metric!r}")
    if dim is not None and int(dim) <= 0:
        raise ValueError(f"Invalid vec0 dimension: {dim}")

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if row is not None and row[0]:
        normalized = " ".join(str(row[0]).split())
        if "USING VEC0(" in normalized.upper():
            if dim is None:
                return
            if (
                f"{pk_col} TEXT PRIMARY KEY" in normalized
                and f"{column} float[{int(dim)}] distance_metric={metric}" in normalized
            ):
                return
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    target_dim = int(dim) if dim is not None else DEFAULT_VEC_DIMENSION
    conn.execute(
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table_name}" USING vec0('
        f"{pk_col} TEXT PRIMARY KEY, {column} float[{target_dim}] distance_metric={metric})"
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
    conn.execute("PRAGMA cache_size = -65536;")
    conn.execute("PRAGMA mmap_size = 268435456;")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)};")


def open_managed_yaw_db(db_path: Path | str) -> ManagedConnection:
    """Open or create a unified yaw.db via CocoIndex ManagedConnection with FIFO RWLock and sqlite-vec."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    managed = coco_sqlite_connect(db_str, timeout=30.0, load_vec=True)
    try:
        validate_graph_schema(managed._conn)
    except ValueError:
        managed.close()
        raise
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
    try:
        validate_graph_schema(conn)
    except ValueError:
        conn.close()
        raise
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
            _configure_connection_pragmas(conn, busy_timeout_ms=busy_ms)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            validate_graph_schema(conn)
            return conn
        except ValueError:
            if conn is not None:
                conn.close()
            raise
        except sqlite3.Error:
            if conn is not None:
                with suppress(sqlite3.Error):
                    conn.close()
            if i >= len(backoff):
                raise
            time.sleep(backoff[i])
    raise sqlite3.OperationalError(f"Failed to open read-only SQLite database: {resolved}")


def validate_graph_schema(conn: sqlite3.Connection) -> None:
    """Reject retired graph stores before any persistent database change."""
    mention_columns = {row[1] for row in conn.execute("PRAGMA table_info(entity_mentions)")}
    if mention_columns and "llm_prob" not in mention_columns:
        raise ValueError("Source confidence schema changed; preserve this database and rebuild the index from source")
    retired = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
        ("concept_schemes", "concept_scheme_members"),
    ).fetchone()
    if retired is not None:
        raise ValueError("Obsolete graph schema: preserve this database and build a fresh scheme-free index")


def resolve_db_identity(conn: sqlite3.Connection, db_path: Path | str | None = None) -> str:
    """Return stable '<canonical_db_path>:<db_nonce>' stored in meta['db_identity']
    so recreated SQLite files at the same path never collide with stale LMDB target state.
    """
    has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
    if has_meta is not None:
        row = conn.execute("SELECT value FROM meta WHERE key = 'db_identity'").fetchone()
        if row is not None and str(row[0]).strip():
            return str(row[0])
    if db_path is not None and str(db_path) != ":memory:":
        canonical_path = Path(db_path).resolve().as_posix()
    else:
        db_rows = conn.execute("PRAGMA database_list").fetchall()
        main_file = next((str(r[2]) for r in db_rows if str(r[1]) == "main" and r[2]), "")
        canonical_path = Path(main_file).resolve().as_posix() if main_file and main_file != ":memory:" else ":memory:"
    identity = f"{canonical_path}:{uuid.uuid4().hex[:12]}"
    with write_tx(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('db_identity', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (identity,),
        )
    return identity


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Initialize all unified schema tables and indices in yaw.db."""
    validate_graph_schema(conn)
    with write_tx(conn):
        for stmt in _CREATE_SCHEMA_STATEMENTS:
            conn.execute(stmt)
        resolve_db_identity(conn)


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
