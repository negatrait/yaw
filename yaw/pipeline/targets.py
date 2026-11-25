"""SQLite chunk/FTS5/artifact schemas, CocoIndex target handlers, and reconciliation."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, override

import cocoindex as coco
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import sqlite

from yaw.core import require_corpus_dir
from yaw.retrieval import compute_db_fingerprints, compute_index_artifacts

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FTS_MODE = "chunk_fts_u61_vocab"
_RABITQ_COLS = frozenset(
    {
        "id",
        "n_chunks",
        "dim",
        "embedding_model",
        "task_mode",
        "vec_sha256",
        "source_sha256",
        "n_words",
        "codes_blob",
        "ip_norm_blob",
        "mean_blob",
        "d_eff_shannon",
    }
)
_CITATION_GRAPH_COLS = frozenset({"id", "n_chunks", "vec_sha256", "source_sha256", "doc_titles_blob", "graph_blob"})


def validate_identifier(name: str) -> str:
    """Validate that `name` is a safe unquoted SQLite identifier."""
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQLite identifier: {name!r}")
    return name


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """Row schema for the plain SQLite `chunks` table storing IEEE float16 embeddings."""

    chunk_id: int
    embedding_f16: bytes
    rel_path: str
    source_hash: str
    start_line: int
    end_line: int
    char_start: int
    char_end: int
    ast_island_start: int
    ast_island_end: int
    title: str


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """Deterministic file chunk ready for batched embedding and target declaration."""

    chunk_id: int
    rel_path: str
    source_hash: str
    start_line: int
    end_line: int
    char_start: int
    char_end: int
    ast_island_start: int
    ast_island_end: int
    title: str
    body: str


@dataclass(slots=True)
class SinkMutationTracker:
    """Thread-safe monotonic counter incremented strictly on actual SQLite target writes."""

    seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self) -> int:
        """Increment and return the target mutation sequence counter."""
        with self._lock:
            self.seq += 1
            return self.seq


SINK_TRACKER = coco.ContextKey[SinkMutationTracker]("yaw/pipeline/sink_tracker")


def _bump_sink_tracker(context_provider: coco.ContextProvider) -> None:
    try:
        tracker = context_provider.get(SINK_TRACKER)
    except KeyError:
        return
    tracker.bump()


class FtsTableKey(NamedTuple):
    """CocoIndex stable key identifying `(db_key, table_name)` for contentless FTS5 targets."""

    db_key: str
    table_name: str


@dataclass(frozen=True)
class _FtsTableAction:
    key: FtsTableKey
    table_name: str
    action: statediff.DiffAction | None
    desired_mode: str | coco.NonExistenceType


@dataclass(frozen=True)
class _FtsRowAction:
    chunk_id: int
    title: str | None
    body: str | None


class _FtsRowHandler(coco.TargetHandler[tuple[str, str], bytes]):
    """Declarative target handler reconciling contentless FTS5 rows in `chunk_fts`."""

    tracks_value_fingerprint = True

    def __init__(self, conn_mgr: sqlite.ManagedConnection, table_name: str) -> None:
        self._conn_mgr = conn_mgr
        self._table_name = validate_identifier(table_name)
        self._sink = coco.TargetActionSink[_FtsRowAction].from_fn(self._apply)

    def _apply(self, context_provider: coco.ContextProvider, actions: Sequence[_FtsRowAction], /) -> None:
        if not actions:
            return
        deletes = [(act.chunk_id,) for act in actions]
        inserts = [
            (act.chunk_id, act.title, act.body) for act in actions if act.title is not None and act.body is not None
        ]
        with self._conn_mgr.transaction() as conn:
            conn.executemany(f'DELETE FROM "{self._table_name}" WHERE rowid = ?', deletes)
            if inserts:
                conn.executemany(
                    f'INSERT INTO "{self._table_name}"(rowid, title, body) VALUES (?, ?, ?)',
                    inserts,
                )
        _bump_sink_tracker(context_provider)

    @override
    def reconcile(
        self,
        key: coco.StableKey,
        desired: tuple[str, str] | coco.NonExistenceType,
        prev_possible_records: Collection[bytes],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[_FtsRowAction, bytes] | None:
        if not isinstance(key, tuple) or not key or not isinstance(key[0], int):
            raise TypeError(f"Expected (chunk_id,) int key, got {key!r}")
        row_id = key[0]
        if coco.is_non_existence(desired):
            if not prev_possible_records and not prev_may_be_missing:
                return None
            return coco.TargetReconcileOutput[_FtsRowAction, bytes, None](
                action=_FtsRowAction(row_id, None, None),
                sink=self._sink,
                tracking_record=coco.NON_EXISTENCE,
            )
        fp = fingerprint_object(desired)
        if not prev_may_be_missing and prev_possible_records and all(p == fp for p in prev_possible_records):
            return None
        return coco.TargetReconcileOutput[_FtsRowAction, bytes, None](
            action=_FtsRowAction(row_id, desired[0], desired[1]),
            sink=self._sink,
            tracking_record=fp,
        )


def _fts_row_handler_factory(context_provider: coco.ContextProvider, action: _FtsTableAction) -> _FtsRowHandler:
    conn_mgr = context_provider.get(action.key.db_key, sqlite.ManagedConnection)
    return _FtsRowHandler(conn_mgr, validate_identifier(action.table_name))


def _apply_fts_tables(context_provider: coco.ContextProvider, actions: Sequence[_FtsTableAction], /) -> None:
    by_db: dict[str, list[_FtsTableAction]] = {}
    for act in actions:
        by_db.setdefault(act.key.db_key, []).append(act)
    for db_key, db_actions in by_db.items():
        conn_mgr = context_provider.get(db_key, sqlite.ManagedConnection)
        did_mutate = False
        with conn_mgr.transaction() as conn:
            for act in db_actions:
                table = validate_identifier(act.table_name)
                if act.action in ("delete", "replace"):
                    conn.execute('DROP TABLE IF EXISTS "chunk_vocab"')
                    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                    did_mutate = True
                if coco.is_non_existence(act.desired_mode):
                    continue
                if act.action in ("insert", "upsert", "replace"):
                    conn.execute(
                        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table}" '
                        "USING fts5(title, body, content='', contentless_delete=1, "
                        "tokenize='unicode61 remove_diacritics 1')"
                    )
                    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS \"chunk_vocab\" USING fts5vocab({table}, 'row')")
                    did_mutate = True
        if did_mutate:
            _bump_sink_tracker(context_provider)


_FTS_TABLE_SINK = coco.TargetActionSink[_FtsTableAction].from_fn(
    _apply_fts_tables,
    child_handler=_fts_row_handler_factory,
)


class _FtsTableHandler(coco.TargetHandler[str, str, _FtsRowHandler]):
    @override
    def reconcile(
        self,
        key: coco.StableKey,
        desired: str | coco.NonExistenceType,
        prev_possible_records: Collection[str],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[_FtsTableAction, str, _FtsRowHandler] | None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Expected (db_key, table_name) key, got {key!r}")
        t_key = FtsTableKey(str(key[0]), validate_identifier(str(key[1])))
        action = statediff.diff(statediff.TrackingRecordTransition(desired, prev_possible_records, prev_may_be_missing))
        invalid: Literal["destructive"] | None = "destructive" if action in ("delete", "replace") else None
        return coco.TargetReconcileOutput[_FtsTableAction, str, _FtsRowHandler](
            action=_FtsTableAction(t_key, t_key.table_name, action, desired),
            sink=_FTS_TABLE_SINK,
            tracking_record=desired,
            child_invalidation=invalid,
        )


FTS_TABLES = coco.register_root_target_states_provider("yaw/pipeline/fts5", _FtsTableHandler())


def open_managed_db(db_path: Path | str) -> sqlite.ManagedConnection:
    """Open CocoIndex ManagedConnection with WAL configured and no vector extensions."""
    target = Path(db_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn_mgr = sqlite.connect(target, timeout=30.0, load_vec=False)
    with conn_mgr.readonly() as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn_mgr


def _create_index_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS "rabitq_index" (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            n_chunks INTEGER NOT NULL,
            dim INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            task_mode TEXT NOT NULL,
            vec_sha256 TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            n_words INTEGER NOT NULL,
            codes_blob BLOB NOT NULL,
            ip_norm_blob BLOB NOT NULL,
            mean_blob BLOB NOT NULL,
            d_eff_shannon REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS "citation_graph" (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            n_chunks INTEGER NOT NULL,
            vec_sha256 TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            doc_titles_blob BLOB NOT NULL,
            graph_blob BLOB NOT NULL
        )
        """
    )


def reconcile_index_tables(
    conn_mgr: sqlite.ManagedConnection,
    *,
    embedding_model: str,
    task_mode: str,
    corpus_root: Path | str,
) -> bool:
    """Optimize FTS5 and materialize single-row `rabitq_index` + `citation_graph` when chunks change."""
    resolved_root = require_corpus_dir(corpus_root)
    with conn_mgr.readonly() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            return False
        n, dim, vec_sha, src_sha, rows = compute_db_fingerprints(conn)
        r_cols = {str(r[1]) for r in conn.execute('PRAGMA table_info("rabitq_index")').fetchall()}
        g_cols = {str(r[1]) for r in conn.execute('PRAGMA table_info("citation_graph")').fetchall()}
        schema_ok = _RABITQ_COLS.issubset(r_cols) and _CITATION_GRAPH_COLS.issubset(g_cols)
        if schema_ok:
            if n == 0:
                has_r = conn.execute('SELECT 1 FROM "rabitq_index" WHERE id = 1').fetchone() is not None
                has_g = conn.execute('SELECT 1 FROM "citation_graph" WHERE id = 1').fetchone() is not None
                if not has_r and not has_g:
                    return False
            else:
                r_row = conn.execute(
                    "SELECT n_chunks, dim, embedding_model, task_mode, vec_sha256, source_sha256, "
                    'length(codes_blob), length(ip_norm_blob), length(mean_blob) FROM "rabitq_index" WHERE id = 1'
                ).fetchone()
                g_row = conn.execute(
                    "SELECT n_chunks, vec_sha256, source_sha256, length(doc_titles_blob), length(graph_blob) "
                    'FROM "citation_graph" WHERE id = 1'
                ).fetchone()
                if (
                    r_row is not None
                    and g_row is not None
                    and tuple(r_row[:6]) == (n, dim, embedding_model, task_mode, vec_sha, src_sha)
                    and tuple(g_row[:3]) == (n, vec_sha, src_sha)
                    and min(int(r_row[6]), int(r_row[7]), int(r_row[8]), int(g_row[3]), int(g_row[4])) > 0
                ):
                    return False

    if n == 0:
        with conn_mgr.transaction() as conn:
            _create_index_tables(conn)
            r_del = conn.execute('DELETE FROM "rabitq_index" WHERE id = 1').rowcount
            g_del = conn.execute('DELETE FROM "citation_graph" WHERE id = 1').rowcount
        return bool((r_del or 0) + (g_del or 0))

    artifacts = compute_index_artifacts(rows, resolved_root)
    with conn_mgr.transaction() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_fts'").fetchone() is not None:
            conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
        cur_r_cols = {str(r[1]) for r in conn.execute('PRAGMA table_info("rabitq_index")').fetchall()}
        if cur_r_cols and not _RABITQ_COLS.issubset(cur_r_cols):
            conn.execute('DROP TABLE IF EXISTS "rabitq_index"')
        cur_g_cols = {str(r[1]) for r in conn.execute('PRAGMA table_info("citation_graph")').fetchall()}
        if cur_g_cols and not _CITATION_GRAPH_COLS.issubset(cur_g_cols):
            conn.execute('DROP TABLE IF EXISTS "citation_graph"')
        _create_index_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO "rabitq_index" (
                id, n_chunks, dim, embedding_model, task_mode, vec_sha256, source_sha256,
                n_words, codes_blob, ip_norm_blob, mean_blob, d_eff_shannon
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifacts.n_chunks,
                artifacts.dim,
                embedding_model,
                task_mode,
                artifacts.vec_sha256,
                artifacts.source_sha256,
                artifacts.n_words,
                artifacts.codes_blob,
                artifacts.ip_norm_blob,
                artifacts.mean_blob,
                artifacts.d_eff_shannon,
            ),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO "citation_graph" (
                id, n_chunks, vec_sha256, source_sha256, doc_titles_blob, graph_blob
            ) VALUES (1, ?, ?, ?, ?, ?)
            """,
            (
                artifacts.n_chunks,
                artifacts.vec_sha256,
                artifacts.source_sha256,
                artifacts.doc_titles_blob,
                artifacts.graph_blob,
            ),
        )
    return True


def list_user_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return top-level logical tables in yaw.db excluding FTS5 and SQLite internal shadow tables."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    logical: list[str] = []
    shadow_prefixes = ("chunk_fts_", "sqlite_")
    for name, _ in rows:
        name_s = str(name)
        if not name_s.startswith(shadow_prefixes):
            logical.append(name_s)
    return tuple(logical)
