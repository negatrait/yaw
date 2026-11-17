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
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import cocoindex as coco
import dspy
import numpy as np
import sqlite_vec
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.text import CustomLanguageConfig, RecursiveSplitter, detect_code_language
from cocoindex.resources import schema as coco_schema
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

CORPUS_ROOT = coco.ContextKey[Path]("yaw/pipeline/corpus_root")
SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/pipeline/sqlite_db")

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
# Configure hierarchical prose separators for non-Tree-sitter formats; AST languages use Tree-sitter.
_SPLITTER = RecursiveSplitter(
    custom_languages=[
        CustomLanguageConfig(
            language_name="prose",
            separators_regex=[r"\n\n+", r"[.?!]+\s+", r"\n", r"\s+"],
        )
    ]
)
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s+(.*?))?(?:\s+#+\s*)?$")
_SETEXT_RE = re.compile(r"^ {0,3}(?:=+|-+)\s*$")
_VEC0_DIM_RE = re.compile(r"\bfloat\[(\d+)\]", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EMBED_BATCH_SIZE = 64
_MAX_EMBED_BATCH_CHARS = 8192
_MAX_EMBED_IN_FLIGHT = 4
_MAX_INFLIGHT_COMPONENTS = 16


def _validate_identifier(name: str) -> str:
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQLite identifier: {name!r}")
    return name


class _DspyBatchEmbedder:
    """Byte-budgeted, unit-L2-normalizing batch embedder backed by bare dspy.Embedder."""

    def __init__(self, cfg: YawConfig) -> None:
        self._max_items = min(_EMBED_BATCH_SIZE, max(1, int(cfg.embed_batch_size)))
        self._max_item_chars = max(4096, int(cfg.chunk_size) * 4)
        self._max_batch_chars = max(self._max_item_chars, _MAX_EMBED_BATCH_CHARS)
        self._embedder = dspy.Embedder(
            f"{cfg.provider}/{cfg.embedding_model}",
            api_base=cfg.base_url,
            api_key=cfg.api_key or "yaw",
            batch_size=self._max_items,
            caching=False,
        )
        self._sem = threading.BoundedSemaphore(_MAX_EMBED_IN_FLIGHT)
        self._async_sem: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None

    def _get_async_sem(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._async_sem is None or self._async_sem[0] is not loop:
            self._async_sem = (loop, asyncio.Semaphore(_MAX_EMBED_IN_FLIGHT))
        return self._async_sem[1]

    def _pack_batches(self, texts: list[str]) -> list[list[str]]:
        for idx, item in enumerate(texts):
            if len(item) > self._max_item_chars:
                raise YawRuntimeError(
                    "INVALID_INPUT",
                    f"Embedding input[{idx}] length {len(item)} chars exceeds limit {self._max_item_chars}",
                )
        # Pack sub-batches by both item count and character budget to prevent oversized API payloads.
        batches: list[list[str]] = []
        cur_batch: list[str] = []
        cur_chars = 0
        for item in texts:
            if cur_batch and (len(cur_batch) >= self._max_items or cur_chars + len(item) > self._max_batch_chars):
                batches.append(cur_batch)
                cur_batch = []
                cur_chars = 0
            cur_batch.append(item)
            cur_chars += len(item)
        if cur_batch:
            batches.append(cur_batch)
        return batches

    @staticmethod
    def _normalize_batch(raw: Any) -> NDArray[np.float32]:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
        return (arr / norms).astype(np.float32, copy=False)

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        parts: list[NDArray[np.float32]] = []
        for batch in self._pack_batches(texts):
            with self._sem:
                raw = self._embedder(batch, batch_size=len(batch))
            parts.append(self._normalize_batch(raw))
        return np.vstack(parts) if len(parts) > 1 else parts[0]

    async def acall(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # Use DSPy's native async path directly on the CocoIndex loop; fall back to threadpool only for sync stubs.
        acall_fn = getattr(self._embedder, "acall", None)
        if not callable(acall_fn):
            return await asyncio.to_thread(self, texts)
        sem = self._get_async_sem()
        parts: list[NDArray[np.float32]] = []
        for batch in self._pack_batches(texts):
            async with sem:
                raw = await acall_fn(batch, batch_size=len(batch))
            parts.append(self._normalize_batch(raw))
        return np.vstack(parts) if len(parts) > 1 else parts[0]


# Bound in-flight CocoIndex batches natively so partial batches do not stall behind saturated worker threads.
# Module-level function keeps `texts` as the sole batched arg and `embedder` as the hashable batcher key.
@coco.fn.as_async(batching=True, max_batch_size=_EMBED_BATCH_SIZE, max_in_flight=_MAX_EMBED_IN_FLIGHT)
async def _embed_batch(texts: list[str], embedder: _PipelineEmbedder) -> list[NDArray[np.float32]]:
    if not texts:
        return []
    try:
        arr = np.ascontiguousarray(await embedder.invoke(texts), dtype="<f4")
    except Exception as exc:
        # Split failed multi-item batches so payload/token caps or one bad chunk never poison healthy peers.
        if len(texts) > 1:
            raise coco.RetryWithSmallerBatch() from exc
        raise
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        if len(texts) > 1:
            raise coco.RetryWithSmallerBatch()
        raise ValueError(f"Expected ({len(texts)}, dim) embedding array, got {arr.shape}")
    embedder.record_dim(int(arr.shape[1]))
    return [arr[i] for i in range(len(texts))]


class _PipelineEmbedder(coco_schema.VectorSchemaProvider):
    """CocoIndex-native batching and VectorSchemaProvider adapter over a batch embed callable."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], NDArray[np.float32]],
        *,
        vector_dim: int | None = None,
        memo_key: object = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._dim = vector_dim
        self._memo_key = memo_key
        self._dim_lock = asyncio.Lock()

    async def invoke(self, texts: list[str]) -> NDArray[np.float32]:
        raw = (
            await self._embed_fn.acall(texts)
            if isinstance(self._embed_fn, _DspyBatchEmbedder)
            else await asyncio.to_thread(self._embed_fn, texts)
        )
        return np.asarray(raw, dtype=np.float32)

    def record_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
        elif self._dim != dim:
            raise ValueError(f"Inconsistent embedding dimension: expected {self._dim}, got {dim}")

    async def _resolve_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        async with self._dim_lock:
            if self._dim is None:
                probe = await self.invoke(["dimension probe"])
                if probe.ndim != 2 or probe.shape[1] < 1:
                    raise ValueError(f"Invalid embedding dimension from probe shape {probe.shape}")
                self._dim = int(probe.shape[1])
            return self._dim

    async def __coco_vector_schema__(self) -> coco_schema.VectorSchema:
        dim = await self._resolve_dim()
        return coco_schema.VectorSchema(dtype=np.dtype(np.float32), size=dim)

    def __coco_memo_key__(self) -> object:
        return self._memo_key

    async def embed(self, text: str) -> NDArray[np.float32]:
        return await _embed_batch(text, self)


EMBEDDER = coco.ContextKey[_PipelineEmbedder]("yaw/pipeline/embedder", detect_change=True)


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
        self._table_name = _validate_identifier(table_name)
        self._sink = coco.TargetActionSink[_FtsRowAction].from_fn(self._apply)

    def _apply(self, _context_provider: coco.ContextProvider, actions: Sequence[_FtsRowAction], /) -> None:
        if not actions:
            return
        deletes = [(act.chunk_id,) for act in actions]
        inserts = [
            (act.chunk_id, act.title, act.body) for act in actions if act.title is not None and act.body is not None
        ]
        # Batch FTS5 deletions and insertions with executemany inside one CocoIndex transaction.
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
                    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                if act.action in ("insert", "upsert", "replace"):
                    conn.execute(
                        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table}" '
                        "USING fts5(title, body, content='', contentless_delete=1)"
                    )


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
            action=_FtsTableAction(t_key, t_key.table_name, action),
            sink=_FTS_TABLE_SINK,
            tracking_record=desired,
            child_invalidation=invalid,
        )


_FTS_TABLES = coco.register_root_target_states_provider("yaw/pipeline/fts5", _FtsTableHandler())


def open_managed_db(db_path: Path | str) -> sqlite.ManagedConnection:
    """Open CocoIndex ManagedConnection with sqlite-vec loaded and WAL configured."""
    target = Path(db_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn_mgr = sqlite.connect(target, timeout=30.0, load_vec=True)
    # ManagedConnection uses autocommit internally; readonly() acquires its lock without BEGIN,
    # which SQLite forbids for PRAGMA journal_mode and PRAGMA synchronous.
    with conn_mgr.readonly() as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn_mgr


def make_dspy_embed_fn(cfg: YawConfig) -> Callable[[list[str]], NDArray[np.float32]]:
    """Build a byte-budgeted, unit-L2-normalizing batch embedder backed by bare dspy.Embedder."""
    return _DspyBatchEmbedder(cfg)


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


def _subsplit_by_bytes(text: str, max_bytes: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    """Enforce a hard UTF-8 byte cap on any oversized leaf node returned by RecursiveSplitter."""
    if not text.strip():
        return []
    limit = max(4, int(max_bytes))
    if len(text.encode("utf-8")) <= limit:
        return [(0, len(text), text)]
    out: list[tuple[int, int, str]] = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(n, pos + limit)
        while end > pos and len(text[pos:end].encode("utf-8")) > limit:
            byte_len = len(text[pos:end].encode("utf-8"))
            shrink = max(1, (end - pos) * limit // byte_len)
            end = pos + shrink if shrink < (end - pos) else end - 1
        if end == pos:
            end = pos + 1
        piece = text[pos:end]
        if piece.strip():
            out.append((pos, end, piece))
        if end >= n:
            break
        step = max(1, (end - pos) - min(max(0, overlap_chars), max(0, (end - pos) - 1)))
        pos += step
    return out


def chunk_file_content(
    rel_path: str,
    raw_text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
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
        units: list[tuple[str, str, int, int, str, str]] = []
        if suffix in (".jsonl", ".ndjson"):
            for line_no, raw_line in enumerate(lines, start=1):
                if raw_line.strip():
                    t, b = json_document_fields(json.loads(raw_line), fallback_title)
                    units.append((t, b, line_no, line_no, _line_slice_sha256(lines, line_no, line_no), rel_path))
        else:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                t, b = json_document_fields(parsed, fallback_title)
                units.append((t, b, 1, total_lines, _line_slice_sha256(lines, 1, total_lines), rel_path))
            elif isinstance(parsed, list):
                s_hash = _line_slice_sha256(lines, 1, total_lines)
                for item_idx, item in enumerate(parsed):
                    t, b = json_document_fields(item, fallback_title)
                    # Disambiguate array items in chunk_id computation while preserving rel_path for hydration.
                    id_key = rel_path if item_idx == 0 else f"{rel_path}#{item_idx}"
                    units.append((t, b, 1, total_lines, s_hash, id_key))
        for t, b, sl, el, s_hash, id_key in units:
            rendered = render_document_text(t, b)
            if not rendered.strip():
                continue
            if len(rendered.encode("utf-8")) <= chunk_size:
                cid = compute_chunk_int_id(id_key, sl, el, 0, 0, s_hash)
                out.append(SourceChunk(cid, rel_path, s_hash, sl, el, 0, 0, sl, el, t, rendered))
            else:
                title_bytes = len(t.encode("utf-8")) + 1 if t else 0
                eff = max(min_chunk_size, chunk_size - title_bytes) if title_bytes < chunk_size else chunk_size
                parts = _SPLITTER.split(
                    b,
                    chunk_size=eff,
                    chunk_overlap=min(chunk_overlap, max(0, eff - 1)),
                    min_chunk_size=min(min_chunk_size, max(1, len(b.strip()))),
                    language="prose",
                )
                for c in parts:
                    base_cs = int(c.start.char_offset)
                    for sub_cs, sub_ce, sub_text in _subsplit_by_bytes(c.text, eff, chunk_overlap):
                        cs, ce = base_cs + sub_cs, base_cs + sub_ce
                        body_text = render_document_text(t, sub_text)
                        cid = compute_chunk_int_id(id_key, sl, el, cs, ce, s_hash)
                        out.append(SourceChunk(cid, rel_path, s_hash, sl, el, cs, ce, sl, el, t, body_text))
        return tuple({c.chunk_id: c for c in out}.values())

    lang = detect_code_language(filename=rel_path) or "prose"
    headings = _markdown_headings(lines) if lang in ("prose", "markdown") else []
    split_chunks = _SPLITTER.split(
        text,
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, max(0, chunk_size - 1)),
        min_chunk_size=min(min_chunk_size, max(1, len(text.strip()))),
        language=lang,
    )
    same_span_counts: dict[tuple[int, int], int] = {}
    raw_pieces: list[tuple[int, int, int, int, str]] = []
    for c in split_chunks:
        if not c.text.strip():
            continue
        sl = max(1, int(c.start.line))
        el = min(total_lines, max(sl, inclusive_end_line(sl, int(c.end.line), int(c.end.column))))
        base_cs = int(c.start.char_offset)
        for sub_cs, sub_ce, sub_text in _subsplit_by_bytes(c.text, chunk_size, chunk_overlap):
            raw_pieces.append((sl, el, base_cs + sub_cs, base_cs + sub_ce, sub_text))
            same_span_counts[(sl, el)] = same_span_counts.get((sl, el), 0) + 1

    for sl, el, piece_cs, piece_ce, piece_text in raw_pieces:
        cs, ce = (piece_cs, piece_ce) if same_span_counts[(sl, el)] > 1 else (0, 0)
        s_hash = _line_slice_sha256(lines, sl, el)
        title, ast_s, ast_e = _heading_bounds(headings, sl, total_lines, fallback_title)
        cid = compute_chunk_int_id(rel_path, sl, el, cs, ce, s_hash)
        out.append(SourceChunk(cid, rel_path, s_hash, sl, el, cs, ce, ast_s, ast_e, title, piece_text))
    return tuple({c.chunk_id: c for c in out}.values())


# Keyed strictly by content (title, body) so line shifts or file moves reuse CocoIndex LMDB memos.
@coco.fn(memo=True, version=2)
async def _compute_chunk_features(
    title_body: tuple[str, str],
    task_mode: str,
    embedder: _PipelineEmbedder,
) -> tuple[NDArray[np.float32], str, str]:
    title, body = title_body
    formatted = format_doc_for_embedding(body, title=title, task_mode=task_mode)

    def _nlp_features() -> tuple[str, str]:
        terms, ents = extract_text_features(title, body)
        return json.dumps(terms, ensure_ascii=False), json.dumps(list(ents), ensure_ascii=False)

    # Overlap async embedding HTTP I/O with CPU spaCy feature extraction.
    vec, (tokens_json, entities_json) = await asyncio.gather(
        embedder.embed(formatted),
        asyncio.to_thread(_nlp_features),
    )
    return vec, tokens_json, entities_json


@coco.fn(memo=True, version=3)
async def _process_file(
    file: FileLike[Any],
    chunks_target: sqlite.TableTarget[ChunkVecRow],
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
    feats = await coco.map(_compute_chunk_features, [(p.title, p.body) for p in pieces], task_mode, embedder)
    for piece, (vec, tokens_json, entities_json) in zip(pieces, feats, strict=True):
        vec_arr = np.asarray(vec, dtype=np.float32)
        chunks_target.declare_row(
            row=ChunkVecRow(
                chunk_id=piece.chunk_id,
                embedding=vec_arr,
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
    vector_schema = await embedder.__coco_vector_schema__()
    vector_dim = vector_schema.size
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


@dataclass(slots=True)
class PipelineSession:
    """Handle bundle owning a CocoIndex Environment, worker thread/loop, and SQLite connection."""

    db_path: Path
    state_dir: Path
    corpus_root: Path
    env: coco.Environment
    app: coco.App[Any, Any]
    conn_mgr: sqlite.ManagedConnection
    live: bool
    ready_event: threading.Event
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    stop_async: asyncio.Event
    watch_handle: coco.UpdateHandle[Any] | None = None
    error: BaseException | None = None


async def _watch_pipeline(session: PipelineSession, stop_async: asyncio.Event) -> None:
    handle = session.app.update(live=True)
    session.watch_handle = handle
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
        known_dim = vector_dim or _detect_existing_vec_dim(conn_mgr)
    except Exception:
        conn_mgr.close()
        raise

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
    embedder = _PipelineEmbedder(active_embed, vector_dim=known_dim, memo_key=memo_key)
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
                coco.AppConfig(name=resolved_app, environment=env, max_inflight_components=_MAX_INFLIGHT_COMPONENTS),
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

        watch_task = asyncio.create_task(_watch_pipeline(session, stop_async))
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
    """Execute one catch-up CocoIndex update synchronously (for non-live batch sessions)."""
    if session.error is not None:
        raise session.error
    session.app.update_blocking(live=False)
    if session.error is not None:
        raise session.error
    session.ready_event.set()


def wait_pipeline_ready(session: PipelineSession, timeout: float = 30.0) -> bool:
    """Block until initial CocoIndex catch-up finishes; re-raises live watcher failures."""
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
    """Return top-level logical tables in yaw.db excluding sqlite-vec and fts5 internal shadow tables."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    logical: list[str] = []
    shadow_prefixes = ("chunks_", "chunk_fts_", "sqlite_")
    for name, _ in rows:
        name_s = str(name)
        if not name_s.startswith(shadow_prefixes):
            logical.append(name_s)
    return tuple(logical)
