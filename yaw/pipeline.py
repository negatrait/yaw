"""Declarative live CocoIndex v1 pipeline indexing IEEE float16 chunks, unicode61 FTS5, and SoA artifacts."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import sqlite3
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import cocoindex as coco
import dspy
import numpy as np
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.code import CodeSource
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from numpy.typing import NDArray

from yaw.core import (
    YawConfig,
    YawRuntimeError,
    compute_chunk_int_id,
    format_doc_for_embedding,
    inclusive_end_line,
    resolve_coco_state_root,
)
from yaw.retrieval import (
    compute_db_fingerprints,
    compute_index_artifacts,
    normalize_fts_unicode61,
)

CORPUS_ROOT = coco.ContextKey[Path]("yaw/pipeline/corpus_root")
SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/pipeline/sqlite_db")

_SPLITTER = RecursiveSplitter()
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FTS_MODE = "chunk_fts_u61_vocab"
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


def _validate_identifier(name: str) -> str:
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQLite identifier: {name!r}")
    return name


class _DspyBatchEmbedder:
    """Native DSPy batching with one shared HTTP admission gate for indexing and queries."""

    def __init__(self, cfg: YawConfig) -> None:
        kwargs: dict[str, Any] = {"api_base": cfg.base_url, "api_key": cfg.api_key or "yaw"}
        self._embedder = dspy.Embedder(
            f"{cfg.provider}/{cfg.embedding_model}",
            batch_size=cfg.embed_batch_size,
            caching=False,
            **kwargs,
        )
        self._sem = threading.BoundedSemaphore(cfg.embed_max_in_flight)

    @staticmethod
    def _normalize_batch(raw: Any) -> NDArray[np.float32]:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 1 or not np.all(np.isfinite(arr)):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced invalid or non-finite embedding array.")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced zero-norm embedding vector.")
        return (arr / norms).astype(np.float32, copy=False)

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._sem:
            raw = self._embedder(texts)
        return self._normalize_batch(raw)


class _PipelineEmbedder:
    """CocoIndex-native batching adapter over a batch embed callable."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], NDArray[np.float32]],
        *,
        memo_key: object = None,
        batch_size: int,
        max_in_flight: int,
    ) -> None:
        self._embed_fn = embed_fn
        self._memo_key = memo_key
        self._dim: int | None = None
        self._dim_lock = threading.Lock()
        self.embed = coco.fn.as_async(
            batching=True,
            max_batch_size=max(1, batch_size),
            max_in_flight=max(1, max_in_flight),
        )(self._embed_batch)

    async def invoke(self, texts: list[str]) -> NDArray[np.float32]:
        raw = await asyncio.to_thread(self._embed_fn, texts)
        return np.asarray(raw, dtype=np.float32)

    def __coco_memo_key__(self) -> object:
        return self._memo_key

    async def _embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if not texts:
            return []
        arr = np.ascontiguousarray(await self.invoke(texts), dtype=np.float32)
        if arr.ndim == 1 and len(texts) == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
            if len(texts) > 1:
                raise coco.RetryWithSmallerBatch()
            raise YawRuntimeError("MODEL_ERROR", f"Expected ({len(texts)}, dim) embedding array, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced non-finite embedding values.")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced zero-norm embedding vector.")
        unit = (arr / norms).astype(np.float32, copy=False)
        with self._dim_lock:
            if self._dim is None:
                self._dim = int(unit.shape[1])
            elif self._dim != int(unit.shape[1]):
                raise YawRuntimeError("MODEL_ERROR", f"Embedding dimension changed: {self._dim} != {unit.shape[1]}")
        return [unit[i] for i in range(len(texts))]


EMBEDDER = coco.ContextKey[_PipelineEmbedder]("yaw/pipeline/embedder", detect_change=True)


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


class _FtsTableKey(NamedTuple):
    db_key: str
    table_name: str


@dataclass(frozen=True)
class _FtsTableAction:
    key: _FtsTableKey
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
        self._table_name = _validate_identifier(table_name)
        self._sink = coco.TargetActionSink[_FtsRowAction].from_fn(self._apply)

    def _apply(self, _context_provider: coco.ContextProvider, actions: Sequence[_FtsRowAction], /) -> None:
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
    return _FtsRowHandler(conn_mgr, _validate_identifier(action.table_name))


def _apply_fts_tables(context_provider: coco.ContextProvider, actions: Sequence[_FtsTableAction], /) -> None:
    by_db: dict[str, list[_FtsTableAction]] = {}
    for act in actions:
        by_db.setdefault(act.key.db_key, []).append(act)
    for db_key, db_actions in by_db.items():
        conn_mgr = context_provider.get(db_key, sqlite.ManagedConnection)
        with conn_mgr.transaction() as conn:
            for act in db_actions:
                table = _validate_identifier(act.table_name)
                if act.action in ("delete", "replace"):
                    conn.execute('DROP TABLE IF EXISTS "chunk_vocab"')
                    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                if coco.is_non_existence(act.desired_mode):
                    continue
                if act.action in ("insert", "upsert", "replace"):
                    conn.execute(
                        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table}" '
                        "USING fts5(title, body, content='', contentless_delete=1, "
                        "tokenize='unicode61 remove_diacritics 1')"
                    )
                    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS \"chunk_vocab\" USING fts5vocab({table}, 'row')")


_FTS_TABLE_SINK = coco.TargetActionSink[_FtsTableAction].from_fn(
    _apply_fts_tables,
    child_handler=_fts_row_handler_factory,
)


class _FtsTableHandler(coco.TargetHandler[str, str, _FtsRowHandler]):
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
        t_key = _FtsTableKey(str(key[0]), _validate_identifier(str(key[1])))
        action = statediff.diff(statediff.TrackingRecordTransition(desired, prev_possible_records, prev_may_be_missing))
        invalid: Literal["destructive"] | None = "destructive" if action in ("delete", "replace") else None
        return coco.TargetReconcileOutput[_FtsTableAction, str, _FtsRowHandler](
            action=_FtsTableAction(t_key, t_key.table_name, action, desired),
            sink=_FTS_TABLE_SINK,
            tracking_record=desired,
            child_invalidation=invalid,
        )


_FTS_TABLES = coco.register_root_target_states_provider("yaw/pipeline/fts5", _FtsTableHandler())


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


def make_dspy_embed_fn(cfg: YawConfig) -> Callable[[list[str]], NDArray[np.float32]]:
    """Build a native DSPy batch embedder preserving each input unchanged."""
    return _DspyBatchEmbedder(cfg)


def chunk_file_content(
    rel_path: str,
    raw_text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_size: int = 100,
) -> tuple[SourceChunk, ...]:
    """Use one native CocoIndex parse and preserve its exact source spans."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ()
    lines = text.splitlines()
    language = detect_code_language(filename=rel_path)
    source = CodeSource(text, language=language)
    chunks = _SPLITTER.split(source, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size)
    out: list[SourceChunk] = []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        size = len(chunk.text.encode("utf-8"))
        if size > chunk_size:
            raise ValueError(
                f"{rel_path}:{chunk.start.line}: native chunk is {size} bytes, exceeding chunk_size={chunk_size}"
            )
        sl = int(chunk.start.line)
        el = inclusive_end_line(sl, int(chunk.end.line), int(chunk.end.column))
        cs, ce = int(chunk.start.char_offset), int(chunk.end.char_offset)
        source_hash = hashlib.sha256("\n".join(lines[sl - 1 : el]).encode("utf-8")).hexdigest()
        cid = compute_chunk_int_id(rel_path, sl, el, cs, ce, source_hash)
        # Empty until native AST title/scope metadata is exposed uniformly.
        out.append(SourceChunk(cid, rel_path, source_hash, sl, el, cs, ce, 1, len(lines), "", chunk.text))
    return tuple(out)


# Keyed strictly by content (title, body) so line shifts or file moves reuse CocoIndex LMDB memos.
@coco.fn(memo=True, version=4)
async def _compute_chunk_embedding(
    title_body: tuple[str, str],
    task_mode: str,
    embedder: _PipelineEmbedder,
) -> bytes:
    title, body = title_body
    formatted = format_doc_for_embedding(body, title=title, task_mode=task_mode)
    vec = await embedder.embed(formatted)
    return np.asarray(vec, dtype="<f2").tobytes()


@coco.fn(memo=True, version=6)
async def _process_file(
    file: FileLike[Any],
    chunks_target: sqlite.TableTarget[ChunkRow],
    fts_target: coco.TargetStateProvider[tuple[str, str] | coco.NonExistenceType, None],
    embedder: _PipelineEmbedder,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    task_mode: str,
) -> int:
    rel_path = file.file_path.as_posix()
    raw_text = await file.read_text()
    pieces = chunk_file_content(
        rel_path,
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    if not pieces:
        return 0
    embeddings_f16 = await coco.map(_compute_chunk_embedding, [(p.title, p.body) for p in pieces], task_mode, embedder)
    for piece, emb_f16 in zip(pieces, embeddings_f16, strict=True):
        chunks_target.declare_row(
            row=ChunkRow(
                chunk_id=piece.chunk_id,
                embedding_f16=bytes(emb_f16),
                rel_path=piece.rel_path,
                source_hash=piece.source_hash,
                start_line=piece.start_line,
                end_line=piece.end_line,
                char_start=piece.char_start,
                char_end=piece.char_end,
                ast_island_start=piece.ast_island_start,
                ast_island_end=piece.ast_island_end,
                title=piece.title,
            )
        )
        coco.declare_target_state(
            fts_target.target_state(
                (piece.chunk_id,),
                (normalize_fts_unicode61(piece.title), normalize_fts_unicode61(piece.body)),
            )
        )
    return len(pieces)


@coco.fn
async def _indexing_main(
    included_patterns: tuple[str, ...],
    excluded_patterns: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    task_mode: str,
    live: bool,
) -> None:
    embedder = coco.use_context(EMBEDDER)
    schema = await sqlite.TableSchema[ChunkRow].from_class(ChunkRow, primary_key=["chunk_id"])
    chunks_target = await sqlite.mount_table_target(SQLITE_DB, "chunks", schema)
    fts_target = await coco.mount_target(_FTS_TABLES.target_state(_FtsTableKey(SQLITE_DB.key, "chunk_fts"), _FTS_MODE))
    matcher = PatternFilePathMatcher(
        included_patterns=list(included_patterns),
        excluded_patterns=list(excluded_patterns),
    )
    files = localfs.walk_dir(CORPUS_ROOT, recursive=True, path_matcher=matcher, live=live)
    mount_handle = await coco.mount_each(
        _process_file,
        files.items(),
        chunks_target,
        fts_target,
        embedder,
        chunk_size,
        chunk_overlap,
        min_chunk_size,
        task_mode,
    )
    if not live:
        await mount_handle.ready()


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
    corpus_root: Path | None = None,
) -> bool:
    """Optimize FTS5 and materialize single-row `rabitq_index` + `citation_graph` when chunks change."""
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

    artifacts = compute_index_artifacts(rows, corpus_root)
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


@dataclass(slots=True)
class PipelineSession:
    """Handle bundle owning a CocoIndex Environment, worker thread/loop, and SQLite connection."""

    db_path: Path
    state_dir: Path
    corpus_root: Path
    embedding_model: str
    task_mode: str
    env: coco.Environment
    app: coco.App[Any, Any]
    conn_mgr: sqlite.ManagedConnection
    live: bool
    ready_event: threading.Event
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    stop_async: asyncio.Event
    error: BaseException | None = None


async def _watch_pipeline(
    session: PipelineSession,
    handle: coco.UpdateHandle[Any],
    stop_async: asyncio.Event,
) -> None:
    snaps: AsyncIterator[Any] | None = None
    had_errors = 0
    try:
        snaps = handle.watch()
        async for snap in snaps:
            if session.error is not None:
                session.ready_event.set()
                stop_async.set()
                break
            if snap.stats.total.num_errors:
                had_errors = snap.stats.total.num_errors
                if snap.status.name == "READY":
                    break
                continue
            if snap.status.name == "READY":
                await asyncio.to_thread(
                    reconcile_index_tables,
                    session.conn_mgr,
                    embedding_model=session.embedding_model,
                    task_mode=session.task_mode,
                    corpus_root=session.corpus_root,
                )
                session.ready_event.set()
        if had_errors and session.error is None:
            session.error = YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"CocoIndex live update reported {had_errors} component error(s).",
            )
            session.ready_event.set()
            stop_async.set()
    except asyncio.CancelledError:
        return
    except BaseException as exc:
        if session.error is None:
            session.error = exc
        session.ready_event.set()
        stop_async.set()
    finally:
        if isinstance(snaps, AsyncGenerator):
            await snaps.aclose()


def start_pipeline(
    cfg: YawConfig,
    *,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    live: bool = True,
    app_name: str | None = None,
) -> PipelineSession:
    """Start the declarative CocoIndex indexing session (live watcher or catch-up app)."""
    assert cfg.db_path is not None and cfg.state_dir is not None and cfg.corpus_root is not None
    db_path = Path(cfg.db_path).resolve()
    corpus_root = Path(cfg.corpus_root).resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    state_dir = resolve_coco_state_root(cfg.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    active_embed = embed_fn or make_dspy_embed_fn(cfg)
    conn_mgr = open_managed_db(db_path)

    os.environ.setdefault("COCOINDEX_BANNER", "false")
    ready_event = threading.Event()
    init_event = threading.Event()
    err_lock = threading.Lock()
    holder: dict[str, Any] = {}

    def _on_component_error(exc: BaseException, _ctx: coco.ExceptionContext) -> None:
        with err_lock:
            session = holder.get("session")
            if session is not None and session.error is None:
                session.error = exc
        ready_event.set()
        raise exc

    base_memo: tuple[object, ...] = (cfg.provider, cfg.embedding_model, cfg.base_url, cfg.embedding_task_mode)
    if isinstance(active_embed, _DspyBatchEmbedder):
        memo_key: object = (*base_memo, "dspy")
    else:
        custom_id = (
            getattr(active_embed, "__module__", ""),
            getattr(active_embed, "__qualname__", type(active_embed).__qualname__),
        )
        memo_key = (*base_memo, "custom", custom_id)
    embedder = _PipelineEmbedder(
        active_embed,
        memo_key=memo_key,
        batch_size=cfg.embed_batch_size,
        max_in_flight=cfg.embed_max_in_flight,
    )
    resolved_app = app_name or f"yaw_{hashlib.sha256(db_path.as_posix().encode('utf-8')).hexdigest()[:12]}"

    async def _session_worker() -> None:
        loop = asyncio.get_running_loop()
        stop_async = asyncio.Event()
        env: coco.Environment | None = None
        try:
            ctx = coco.ContextProvider()
            ctx.provide(CORPUS_ROOT, corpus_root)
            ctx.provide(SQLITE_DB, conn_mgr)
            ctx.provide(EMBEDDER, embedder)
            env = coco.Environment(
                settings=coco.Settings(db_path=state_dir),
                context_provider=ctx,
                event_loop=loop,
                exception_handler=_on_component_error,
            )
            app = coco.App(
                coco.AppConfig(name=resolved_app, environment=env),
                _indexing_main,
                tuple(cfg.included_patterns),
                tuple(cfg.excluded_patterns),
                cfg.chunk_size,
                cfg.chunk_overlap,
                cfg.min_chunk_size,
                cfg.embedding_task_mode,
                live,
            )
            session = PipelineSession(
                db_path=db_path,
                state_dir=state_dir,
                corpus_root=corpus_root,
                embedding_model=cfg.embedding_model,
                task_mode=cfg.embedding_task_mode,
                env=env,
                app=app,
                conn_mgr=conn_mgr,
                live=live,
                ready_event=ready_event,
                thread=threading.current_thread(),
                loop=loop,
                stop_async=stop_async,
            )
            holder["session"] = session
            init_event.set()
        except BaseException as exc:
            holder["init_error"] = exc
            if env is not None:
                env.close()
            init_event.set()
            return

        if not live:
            await stop_async.wait()
            return

        handle = app.update(live=True)
        watch_task = asyncio.create_task(_watch_pipeline(session, handle, stop_async))
        stop_task = asyncio.create_task(stop_async.wait())
        _, pending = await asyncio.wait({watch_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        results = await asyncio.gather(watch_task, stop_task, return_exceptions=True)
        for res in results:
            if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
                if session.error is None:
                    session.error = res
                session.ready_event.set()
        cur = asyncio.current_task()
        other = [t for t in asyncio.all_tasks(loop) if t is not cur and not t.done()]
        for t in other:
            t.cancel()
        if other:
            await asyncio.gather(*other, return_exceptions=True)
        try:
            await asyncio.wait_for(handle.result(), timeout=5.0)
        except asyncio.CancelledError, TimeoutError:
            pass
        except BaseException as exc:
            if session.error is None:
                session.error = exc
            session.ready_event.set()
        del handle

    thread = threading.Thread(
        target=lambda: asyncio.run(_session_worker()),
        daemon=True,
        name="yaw-cocoindex-session",
    )
    try:
        thread.start()
    except Exception:
        conn_mgr.close()
        raise
    init_event.wait()
    if (init_err := holder.get("init_error")) is not None:
        thread.join(timeout=5.0)
        conn_mgr.close()
        raise init_err
    return cast(PipelineSession, holder["session"])


def update_pipeline_once(session: PipelineSession) -> None:
    """Execute one catch-up CocoIndex update synchronously and reconcile persisted index tables."""
    if session.error is not None:
        raise session.error
    session.app.update_blocking(live=False)
    if session.error is not None:
        raise session.error
    reconcile_index_tables(
        session.conn_mgr,
        embedding_model=session.embedding_model,
        task_mode=session.task_mode,
        corpus_root=session.corpus_root,
    )
    session.ready_event.set()


def wait_pipeline_ready(session: PipelineSession, timeout: float = 30.0) -> bool:
    """Block until initial CocoIndex catch-up and reconciliation finish; re-raises live watcher failures."""
    ready = session.ready_event.wait(timeout=timeout)
    if session.error is not None:
        raise session.error
    return ready


def close_pipeline(session: PipelineSession) -> None:
    """Stop the CocoIndex worker thread/loop cleanly and close Environment and SQLite handles."""
    try:
        if not session.loop.is_closed() and session.loop.is_running():
            with contextlib.suppress(RuntimeError):
                session.loop.call_soon_threadsafe(session.stop_async.set)
        if session.thread.is_alive() and session.thread is not threading.current_thread():
            session.thread.join(timeout=10.0)
            if session.thread.is_alive():
                raise YawRuntimeError("SHUTTING_DOWN", "Timed out waiting for CocoIndex session thread to stop.")
    finally:
        try:
            session.env.close()
        finally:
            session.conn_mgr.close()


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
