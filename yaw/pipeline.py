"""Declarative live CocoIndex v1 pipeline indexing chunks into SQLite vec0 and FTS5."""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import AsyncGenerator, Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import cocoindex as coco
import dspy
import numpy as np
import sqlite_vec
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from numpy.typing import NDArray

from yaw.core import (
    YawConfig,
    YawRuntimeError,
    compute_chunk_int_id,
    format_doc_for_embedding,
    inclusive_end_line,
    json_document_fields,
    render_document_text,
    resolve_coco_state_root,
)
from yaw.retrieval import extract_text_features

SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/pipeline/sqlite_db")
EMBED_FN = coco.ContextKey[Callable[[list[str]], NDArray[np.float32]]]("yaw/pipeline/embed_fn")
INDEXING_ERROR_SINK = coco.ContextKey[Callable[[BaseException], None]]("yaw/pipeline/indexing_error_sink")

_CHUNK_AUXILIARY_COLS: list[str] = [
    "rel_path",
    "source_hash",
    "start_line",
    "end_line",
    "char_start",
    "char_end",
    "ast_island_start",
    "ast_island_end",
    "title",
    "tokens_json",
    "entities_json",
]
_SPLITTER = RecursiveSplitter()
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s+(.*?))?(?:\s+#+\s*)?$")
_SETEXT_RE = re.compile(r"^ {0,3}(?:=+|-+)\s*$")
_VEC0_DIM_RE = re.compile(r"\bfloat\[(\d+)\]", re.IGNORECASE)
_EMBED_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class ChunkVecRow:
    """Row schema for the single sqlite-vec `chunks` virtual table."""

    chunk_id: int
    embedding: NDArray[np.float32]
    rel_path: str
    source_hash: str
    start_line: int
    end_line: int
    char_start: int
    char_end: int
    ast_island_start: int
    ast_island_end: int
    title: str
    tokens_json: str
    entities_json: str


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """Deterministic file chunk ready for batched feature extraction and target declaration."""

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
        self._table_name = table_name
        self._sink = coco.TargetActionSink[_FtsRowAction].from_fn(self._apply)

    def _apply(self, context_provider: coco.ContextProvider, actions: Sequence[_FtsRowAction], /) -> None:
        if not actions:
            return
        try:
            with self._conn_mgr.transaction() as conn:
                for act in actions:
                    conn.execute(f'DELETE FROM "{self._table_name}" WHERE rowid = ?', (act.chunk_id,))
                    if act.title is not None and act.body is not None:
                        conn.execute(
                            f'INSERT INTO "{self._table_name}"(rowid, title, body) VALUES (?, ?, ?)',
                            (act.chunk_id, act.title, act.body),
                        )
        except BaseException as exc:
            context_provider.get(INDEXING_ERROR_SINK)(exc)
            raise

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
    return _FtsRowHandler(context_provider.get(SQLITE_DB), action.table_name)


def _apply_fts_tables(context_provider: coco.ContextProvider, actions: Sequence[_FtsTableAction], /) -> None:
    try:
        conn_mgr = context_provider.get(SQLITE_DB)
        with conn_mgr.transaction() as conn:
            for act in actions:
                if act.action in ("delete", "replace"):
                    conn.execute(f'DROP TABLE IF EXISTS "{act.table_name}"')
                if act.action in ("insert", "upsert", "replace"):
                    conn.execute(
                        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{act.table_name}" '
                        "USING fts5(title, body, content='', contentless_delete=1)"
                    )
    except BaseException as exc:
        context_provider.get(INDEXING_ERROR_SINK)(exc)
        raise


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
        t_key = _FtsTableKey(str(key[0]), str(key[1]))
        action = statediff.diff(statediff.TrackingRecordTransition(desired, prev_possible_records, prev_may_be_missing))
        invalid: Literal["destructive"] | None = "destructive" if action in ("delete", "replace") else None
        return coco.TargetReconcileOutput[_FtsTableAction, str, _FtsRowHandler](
            action=_FtsTableAction(t_key, t_key.table_name, action),
            sink=_FTS_TABLE_SINK,
            tracking_record=desired,
            child_invalidation=invalid,
        )


_FTS_TABLES = coco.register_root_target_states_provider("yaw/pipeline/fts5", _FtsTableHandler())


def open_managed_db(db_path: Path | str) -> sqlite.ManagedConnection:
    """Open CocoIndex ManagedConnection with sqlite-vec loaded and WAL configured in autocommit."""
    target = Path(db_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn_mgr = sqlite.connect(target, timeout=30.0, load_vec=True)
    with conn_mgr.readonly() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
    return conn_mgr


def make_dspy_embed_fn(cfg: YawConfig) -> Callable[[list[str]], NDArray[np.float32]]:
    """Build a bounded, unit-L2-normalizing batch embedder backed by bare dspy.Embedder."""
    embedder = dspy.Embedder(
        f"{cfg.provider}/{cfg.embedding_model}",
        api_base=cfg.base_url,
        api_key=cfg.api_key,
        batch_size=min(_EMBED_BATCH_SIZE, max(1, cfg.embed_batch_size)),
        caching=False,
    )
    sem = threading.BoundedSemaphore(2)

    def _embed(texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        parts: list[NDArray[np.float32]] = []
        for idx in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[idx : idx + _EMBED_BATCH_SIZE]
            with sem:
                raw = embedder(batch)
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            parts.append((arr / norms).astype(np.float32, copy=False))
        return np.vstack(parts) if len(parts) > 1 else parts[0]

    return _embed


def _detect_existing_vec_dim(conn_mgr: sqlite.ManagedConnection) -> int | None:
    with conn_mgr.readonly() as conn:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
    if row is None or not row[0]:
        return None
    m = _VEC0_DIM_RE.search(str(row[0]))
    return int(m.group(1)) if m is not None else None


def _markdown_headings(lines: list[str]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    in_fence: tuple[str, int] | None = None
    prev: tuple[int, str] | None = None
    for line_no, raw in enumerate(lines, start=1):
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", raw)
        if fence is not None:
            if in_fence is None:
                in_fence = (fence[1][0], len(fence[1]))
            elif fence[1][0] == in_fence[0] and len(fence[1]) >= in_fence[1] and not fence[2].strip():
                in_fence = None
            prev = None
            continue
        if in_fence is not None:
            continue
        if (m_atx := _ATX_HEADING_RE.match(raw)) is not None:
            headings.append((line_no, (m_atx.group(1) or "").strip()))
            prev = None
        elif prev is not None and prev[0] == line_no - 1 and _SETEXT_RE.match(raw) is not None:
            headings.append(prev)
            prev = None
        else:
            prev = (line_no, raw.strip()) if raw.strip() else None
    return headings


def _heading_bounds(
    headings: Sequence[tuple[int, str]], line_no: int, total_lines: int, fallback: str
) -> tuple[str, int, int]:
    if not headings:
        return fallback, 1, max(1, total_lines)
    idx = bisect.bisect_right([h[0] for h in headings], line_no) - 1
    if idx < 0:
        return fallback, 1, max(1, headings[0][0] - 1)
    end = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else max(1, total_lines)
    return headings[idx][1] or fallback, headings[idx][0], end


def _line_slice_sha256(lines: Sequence[str], start_line: int, end_line: int) -> str:
    payload = "\n".join(lines[start_line - 1 : end_line])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_file_content(
    rel_path: str,
    raw_text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
    min_chunk_size: int = 100,
) -> tuple[SourceChunk, ...]:
    """Split raw disk text into 1-indexed deterministic line/char chunks with source hashes."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ()
    lines = text.splitlines()
    total_lines = max(1, len(lines))
    suffix = Path(rel_path).suffix.lower()
    fallback_title = Path(rel_path).stem
    out: list[SourceChunk] = []

    if suffix in (".json", ".jsonl", ".ndjson"):
        units: list[tuple[str, str, int, int, str]] = []
        if suffix in (".jsonl", ".ndjson"):
            for line_no, raw_line in enumerate(lines, start=1):
                if raw_line.strip():
                    t, b = json_document_fields(json.loads(raw_line), fallback_title)
                    units.append((t, b, line_no, line_no, _line_slice_sha256(lines, line_no, line_no)))
        else:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                t, b = json_document_fields(parsed, fallback_title)
                units.append((t, b, 1, total_lines, _line_slice_sha256(lines, 1, total_lines)))
            elif isinstance(parsed, list):
                s_hash = _line_slice_sha256(lines, 1, total_lines)
                for item in parsed:
                    t, b = json_document_fields(item, fallback_title)
                    units.append((t, b, 1, total_lines, s_hash))
        for t, b, sl, el, s_hash in units:
            rendered = render_document_text(t, b)
            if not rendered.strip():
                continue
            if len(rendered) <= chunk_size:
                cid = compute_chunk_int_id(rel_path, sl, el, 0, 0, s_hash)
                out.append(SourceChunk(cid, rel_path, s_hash, sl, el, 0, 0, sl, el, t, rendered))
            else:
                eff = max(min_chunk_size, chunk_size - len(t) - 1) if t else chunk_size
                parts = _SPLITTER.split(
                    b,
                    chunk_size=eff,
                    chunk_overlap=min(chunk_overlap, max(0, eff - 1)),
                    min_chunk_size=min(min_chunk_size, max(1, len(b.strip()))),
                    language="markdown",
                )
                for c in parts:
                    cs, ce = c.start.char_offset, c.end.char_offset
                    body_text = render_document_text(t, c.text)
                    cid = compute_chunk_int_id(rel_path, sl, el, cs, ce, s_hash)
                    out.append(SourceChunk(cid, rel_path, s_hash, sl, el, cs, ce, sl, el, t, body_text))
        return tuple({c.chunk_id: c for c in out}.values())

    lang = detect_code_language(filename=rel_path)
    headings = _markdown_headings(lines) if lang in (None, "markdown") else []
    split_chunks = _SPLITTER.split(
        text,
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, max(0, chunk_size - 1)),
        min_chunk_size=min(min_chunk_size, max(1, len(text.strip()))),
        language=lang,
    )
    same_span_counts: dict[tuple[int, int], int] = {}
    raw_pieces = []
    for c in split_chunks:
        sl = max(1, int(c.start.line))
        el = min(total_lines, max(sl, inclusive_end_line(sl, int(c.end.line), int(c.end.column))))
        same_span_counts[(sl, el)] = same_span_counts.get((sl, el), 0) + 1
        raw_pieces.append((c, sl, el))

    for c, sl, el in raw_pieces:
        if same_span_counts[(sl, el)] > 1:
            cs, ce = int(c.start.char_offset), int(c.end.char_offset)
        else:
            cs, ce = 0, 0
        s_hash = _line_slice_sha256(lines, sl, el)
        title, ast_s, ast_e = _heading_bounds(headings, sl, total_lines, fallback_title)
        cid = compute_chunk_int_id(rel_path, sl, el, cs, ce, s_hash)
        out.append(SourceChunk(cid, rel_path, s_hash, sl, el, cs, ce, ast_s, ast_e, title, c.text))
    return tuple({c.chunk_id: c for c in out}.values())


@coco.fn.as_async(batching=True, max_batch_size=_EMBED_BATCH_SIZE)
async def _embed_formatted_texts(
    items: list[tuple[Callable[[list[str]], NDArray[np.float32]], str]],
) -> list[bytes]:
    by_fn: dict[int, tuple[Callable[[list[str]], NDArray[np.float32]], list[tuple[int, str]]]] = {}
    for idx, (fn, text) in enumerate(items):
        bucket = by_fn.get(id(fn))
        if bucket is None:
            by_fn[id(fn)] = (fn, [(idx, text)])
        else:
            bucket[1].append((idx, text))
    out: dict[int, bytes] = {}

    def _run() -> None:
        for fn, pairs in by_fn.values():
            arr = fn([text for _, text in pairs])
            if arr.ndim != 2 or arr.shape[0] != len(pairs):
                raise ValueError(f"Expected ({len(pairs)}, dim) embedding array, got {arr.shape}")
            for row_i, (orig_i, _) in enumerate(pairs):
                out[orig_i] = arr[row_i].astype("<f4", copy=False).tobytes()

    await asyncio.to_thread(_run)
    return [out[i] for i in range(len(items))]


@coco.fn(memo=True, version=1)
async def _compute_chunk_features(
    chunk_id: int,
    title: str,
    body: str,
    task_mode: str,
) -> tuple[bytes, str, str]:
    _ = chunk_id
    formatted = format_doc_for_embedding(body, title=title, task_mode=task_mode)
    embed_fn = coco.use_context(EMBED_FN)
    vec_bytes = await _embed_formatted_texts((embed_fn, formatted))

    def _nlp_features() -> tuple[str, str]:
        terms, ents = extract_text_features(title, body)
        return json.dumps(terms, ensure_ascii=False), json.dumps(list(ents), ensure_ascii=False)

    tokens_json, entities_json = await asyncio.to_thread(_nlp_features)
    return vec_bytes, tokens_json, entities_json


@coco.fn(memo=True, version=1)
async def _process_file(
    file: FileLike[Any],
    corpus_root: Path,
    chunks_target: sqlite.TableTarget[ChunkVecRow],
    fts_target: coco.TargetStateProvider[tuple[str, str] | coco.NonExistenceType, None],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    task_mode: str,
) -> int:
    try:
        rel_path = Path(file.file_path.path).resolve().relative_to(corpus_root).as_posix()
        raw_text = await file.read_text()
        pieces = chunk_file_content(
            rel_path,
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        count = 0
        for start in range(0, len(pieces), _EMBED_BATCH_SIZE):
            batch = pieces[start : start + _EMBED_BATCH_SIZE]
            feats = await asyncio.gather(
                *(_compute_chunk_features(c.chunk_id, c.title, c.body, task_mode) for c in batch)
            )
            for piece, (vec_bytes, tokens_json, entities_json) in zip(batch, feats, strict=True):
                vec = np.frombuffer(vec_bytes, dtype="<f4").copy()
                chunks_target.declare_row(
                    row=ChunkVecRow(
                        chunk_id=piece.chunk_id,
                        embedding=vec,
                        rel_path=piece.rel_path,
                        source_hash=piece.source_hash,
                        start_line=piece.start_line,
                        end_line=piece.end_line,
                        char_start=piece.char_start,
                        char_end=piece.char_end,
                        ast_island_start=piece.ast_island_start,
                        ast_island_end=piece.ast_island_end,
                        title=piece.title,
                        tokens_json=tokens_json,
                        entities_json=entities_json,
                    )
                )
                coco.declare_target_state(fts_target.target_state((piece.chunk_id,), (piece.title, piece.body)))
                count += 1
        return count
    except BaseException as exc:
        coco.use_context(INDEXING_ERROR_SINK)(exc)
        raise


@coco.fn
async def _indexing_main(
    corpus_root: Path,
    vector_dim: int,
    included_patterns: tuple[str, ...],
    excluded_patterns: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    task_mode: str,
    live: bool,
) -> None:
    try:
        schema = await sqlite.TableSchema[ChunkVecRow].from_class(
            ChunkVecRow,
            primary_key=["chunk_id"],
            column_overrides={
                "embedding": sqlite.SqliteType(
                    f"float[{vector_dim}] distance_metric=cosine",
                    sqlite_vec.serialize_float32,
                )
            },
        )
        chunks_target = await sqlite.mount_table_target(
            SQLITE_DB,
            "chunks",
            schema,
            virtual_table_def=sqlite.Vec0TableDef(auxiliary_columns=list(_CHUNK_AUXILIARY_COLS)),
        )
        fts_target = await coco.mount_target(
            _FTS_TABLES.target_state(_FtsTableKey(SQLITE_DB.key, "chunk_fts"), "chunk_fts")
        )

        async def _mount_file(file: localfs.File, /) -> int:
            return await _process_file(
                file, corpus_root, chunks_target, fts_target, chunk_size, chunk_overlap, min_chunk_size, task_mode
            )

        matcher = PatternFilePathMatcher(
            included_patterns=list(included_patterns),
            excluded_patterns=list(excluded_patterns),
        )
        files = localfs.walk_dir(corpus_root, recursive=True, path_matcher=matcher, live=live)
        await coco.mount_each(_mount_file, files.items())
    except BaseException as exc:
        coco.use_context(INDEXING_ERROR_SINK)(exc)
        raise


@dataclass(slots=True)
class PipelineSession:
    """Handle bundle owning a CocoIndex Environment, SQLite ManagedConnection, and live watcher."""

    db_path: Path
    state_dir: Path
    corpus_root: Path
    env: coco.Environment
    app: coco.App[Any, Any]
    conn_mgr: sqlite.ManagedConnection
    live: bool
    ready_event: threading.Event
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None
    watch_task: asyncio.Task[None] | None = None
    error: BaseException | None = None


def _run_live_loop(session: PipelineSession) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session.loop = loop

    async def _watch() -> None:
        handle = session.app.update(live=True)
        snaps = handle.watch()
        try:
            async for snap in snaps:
                if session.error is not None:
                    session.ready_event.set()
                    break
                if snap.stats.total.num_errors:
                    if session.error is None:
                        session.error = YawRuntimeError(
                            "CONSISTENCY_ERROR",
                            f"CocoIndex live update reported {snap.stats.total.num_errors} component error(s).",
                        )
                    session.ready_event.set()
                    break
                if snap.status.name == "READY":
                    session.ready_event.set()
        finally:
            if isinstance(snaps, AsyncGenerator):
                await snaps.aclose()

    task = loop.create_task(_watch())
    session.watch_task = task
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    except BaseException as exc:
        session.error = exc
        session.ready_event.set()
    finally:
        loop.close()


def start_pipeline(
    cfg: YawConfig,
    *,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    vector_dim: int | None = None,
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
    try:
        dim = vector_dim or _detect_existing_vec_dim(conn_mgr)
        if dim is None:
            probe = np.asarray(active_embed(["dimension probe"]), dtype=np.float32)
            if probe.ndim != 2 or probe.shape[1] < 1:
                raise ValueError(f"Invalid embedding dimension from probe shape {probe.shape}")
            dim = int(probe.shape[1])

        os.environ.setdefault("COCOINDEX_BANNER", "false")
        ready_event = threading.Event()
        err_lock = threading.Lock()
        session_holder: list[PipelineSession] = []

        def _record_error(exc: BaseException) -> None:
            with err_lock:
                if session_holder and session_holder[0].error is None:
                    session_holder[0].error = exc
                ready_event.set()

        ctx = coco.ContextProvider()
        ctx.provide(SQLITE_DB, conn_mgr)
        ctx.provide(EMBED_FN, active_embed)
        ctx.provide(INDEXING_ERROR_SINK, _record_error)
        env = coco.Environment(settings=coco.Settings(db_path=state_dir), context_provider=ctx)
        resolved_app = app_name or f"yaw_{hashlib.sha256(db_path.as_posix().encode('utf-8')).hexdigest()[:12]}"
        app = coco.App(
            coco.AppConfig(name=resolved_app, environment=env, max_inflight_components=4),
            _indexing_main,
            corpus_root,
            dim,
            tuple(cfg.included_patterns),
            tuple(cfg.excluded_patterns),
            cfg.chunk_size,
            cfg.chunk_overlap,
            cfg.min_chunk_size,
            cfg.embedding_task_mode,
            live,
        )
    except Exception:
        conn_mgr.close()
        raise

    session = PipelineSession(
        db_path=db_path,
        state_dir=state_dir,
        corpus_root=corpus_root,
        env=env,
        app=app,
        conn_mgr=conn_mgr,
        live=live,
        ready_event=ready_event,
    )
    session_holder.append(session)
    if live:
        thread = threading.Thread(target=_run_live_loop, args=(session,), daemon=True, name="yaw-coco-live")
        session.thread = thread
        thread.start()
    return session


def update_pipeline_once(session: PipelineSession) -> None:
    """Execute one catch-up CocoIndex update synchronously (for non-live batch sessions)."""
    if session.error is not None:
        raise session.error

    async def _run_once() -> int:
        handle = session.app.update(live=False)
        await handle.result()
        stats = handle.stats()
        return stats.total.num_errors if stats is not None else 0

    num_errors = asyncio.run(_run_once())
    if session.error is not None:
        raise session.error
    if num_errors:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"CocoIndex update reported {num_errors} component error(s).")
    session.ready_event.set()


def wait_pipeline_ready(session: PipelineSession, timeout: float = 30.0) -> bool:
    """Block until initial CocoIndex catch-up finishes; re-raises live watcher failures."""
    ready = session.ready_event.wait(timeout=timeout)
    if session.error is not None:
        raise session.error
    return ready


def close_pipeline(session: PipelineSession) -> None:
    """Stop any live watcher loop cleanly and close CocoIndex Environment and SQLite handles."""
    if session.loop is not None and session.watch_task is not None:
        with contextlib.suppress(RuntimeError):
            session.loop.call_soon_threadsafe(session.watch_task.cancel)
    if session.thread is not None and session.thread.is_alive():
        session.thread.join(timeout=5.0)
    try:
        session.env.close()
    finally:
        session.conn_mgr.close()


def list_user_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return top-level logical tables in yaw.db excluding sqlite-vec and fts5 internal shadow tables."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    logical: list[str] = []
    shadow_prefixes = ("chunks_", "chunk_fts_", "sqlite_")
    for name, _ in rows:
        name_s = str(name)
        if not name_s.startswith(shadow_prefixes):
            logical.append(name_s)
    return tuple(logical)
