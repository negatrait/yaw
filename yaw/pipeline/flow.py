"""Declarative CocoIndex v1 chunking, memoized embedding, and file processing flow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.code import CodeSource
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.core import compute_chunk_int_id, format_doc_for_embedding, inclusive_end_line
from yaw.retrieval import normalize_fts_unicode61

from .embedder import EMBEDDER, PipelineEmbedder
from .targets import FTS_MODE, FTS_TABLES, ChunkRow, FtsTableKey, SourceChunk

CORPUS_ROOT = coco.ContextKey[Path]("yaw/pipeline/corpus_root")
SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/pipeline/sqlite_db")

_SPLITTER = RecursiveSplitter()


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
    embedder: PipelineEmbedder,
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
    embedder: PipelineEmbedder,
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
async def indexing_main(
    included_patterns: tuple[str, ...],
    excluded_patterns: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    task_mode: str,
    live: bool,
) -> None:
    """Declarative CocoIndex main flow mounting SQLite chunks and FTS5 targets."""
    embedder = coco.use_context(EMBEDDER)
    schema = await sqlite.TableSchema[ChunkRow].from_class(ChunkRow, primary_key=["chunk_id"])
    chunks_target = await sqlite.mount_table_target(SQLITE_DB, "chunks", schema)
    fts_target = await coco.mount_target(FTS_TABLES.target_state(FtsTableKey(SQLITE_DB.key, "chunk_fts"), FTS_MODE))
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
