"""Declarative CocoIndex v1 chunking, memoized embedding, and file processing flow."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectorkits.fingerprint import fingerprint_str
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.code import CodeMatch, CodeSource, render_match
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk, TextPosition
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.core import (
    compute_chunk_int_id,
    format_doc_for_embedding,
    inclusive_end_line,
    markdown_heading_level,
    normalize_ast_heading,
    split_lf_lines,
)
from yaw.retrieval import normalize_fts_unicode61

from .embedder import EMBEDDER, PipelineEmbedder
from .targets import FTS_MODE, FTS_TABLES, ChunkRow, FtsTableKey, SourceChunk

CORPUS_ROOT = coco.ContextKey[Path]("yaw/pipeline/corpus_root")
SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/pipeline/sqlite_db")

_SPLITTER = RecursiveSplitter()
# CocoIndex 1.2.0 SourceView.segments labels all frames uniformly as kind='frame' without AST node_type;
# CodePattern rejects Markdown and typed/annotated defs like '-> int:'.
_CONTROL_FLOW_FRAME_RE = re.compile(
    r"^(?:async\s+(?:for|with)|else\s+if|if|elif|else|for|while|try|except|finally|"
    r"with|match|case|switch|do|catch|loop)\b"
)
_MAX_ISLAND_LINES = 9_999


def _compute_line_positions(lines: Sequence[str]) -> list[tuple[TextPosition, TextPosition] | None]:
    positions: list[tuple[TextPosition, TextPosition] | None] = []
    byte_off = 0
    char_off = 0
    for line_no, raw_ln in enumerate(lines, start=1):
        stripped = raw_ln.lstrip()
        line_bytes = len(raw_ln.encode("utf-8"))
        line_chars = len(raw_ln)
        if not stripped:
            positions.append(None)
        else:
            lead_str = raw_ln[: len(raw_ln) - len(stripped)]
            lead_bytes = len(lead_str.encode("utf-8"))
            lead_chars = len(lead_str)
            start_pos = TextPosition(
                byte_offset=byte_off + lead_bytes,
                char_offset=char_off + lead_chars,
                line=line_no,
                column=lead_chars + 1,
            )
            end_pos = TextPosition(
                byte_offset=byte_off + line_bytes,
                char_offset=char_off + line_chars,
                line=line_no,
                column=line_chars + 1,
            )
            positions.append((start_pos, end_pos))
        byte_off += line_bytes + 1
        char_off += line_chars + 1
    return positions


def _resolve_chunk_anchor(
    lines: Sequence[str],
    frames: Sequence[int],
    sl: int,
    *,
    is_markdown: bool,
) -> tuple[int, str] | None:
    if is_markdown:
        headings = [(ln, lvl) for ln in frames if (lvl := markdown_heading_level(lines, ln)) is not None]
        if not headings:
            return None
        use_sub = len(headings) >= 2 and headings[0][1] == 1 and headings[0][0] < sl
        chosen_ln = headings[1][0] if use_sub else headings[0][0]
        title = normalize_ast_heading(lines[chosen_ln - 1])
        return (chosen_ln, title) if title else None
    for ln in reversed(frames):
        stripped = lines[ln - 1].lstrip()
        if not stripped or _CONTROL_FLOW_FRAME_RE.match(stripped):
            continue
        if title := normalize_ast_heading(lines[ln - 1]):
            return ln, title
    return None


def _query_frame_lines(source: CodeSource, span: tuple[TextPosition, TextPosition]) -> list[int]:
    match = CodeMatch(kind="probe", chunks=[Chunk(text="", start=span[0], end=span[1])], captures={})
    view = render_match(source, match)
    return [
        int(seg.start.line)
        for seg in view.segments
        if seg.kind == "frame" and seg.start.byte_offset < seg.end.byte_offset
    ]


def _has_inner_frame(lines: Sequence[str], frames: Sequence[int], sl: int, *, is_markdown: bool) -> bool:
    if is_markdown:
        return any(ln >= sl and markdown_heading_level(lines, ln) is not None for ln in frames)
    return any(f >= sl for f in frames)


def _probe_markdown_candidates(
    lines: Sequence[str],
    line_spans: Sequence[tuple[TextPosition, TextPosition] | None],
    source: CodeSource,
    non_empty: Sequence[int],
    sl: int,
) -> list[int] | None:
    for pos in range(len(non_empty) - 1):
        if markdown_heading_level(lines, non_empty[pos]) is None:
            continue
        if (span := line_spans[non_empty[pos + 1] - 1]) is None:
            continue
        fl = _query_frame_lines(source, span)
        if _has_inner_frame(lines, fl, sl, is_markdown=True):
            return fl
    return None


def _probe_code_candidates(
    lines: Sequence[str],
    line_spans: Sequence[tuple[TextPosition, TextPosition] | None],
    source: CodeSource,
    non_empty: Sequence[int],
    sl: int,
) -> list[int] | None:
    saw_step = False
    prev_indent = len(lines[non_empty[0] - 1]) - len(lines[non_empty[0] - 1].lstrip())
    for ln in non_empty[1:]:
        raw_ln = lines[ln - 1]
        indent = len(raw_ln) - len(raw_ln.lstrip())
        is_step = indent > prev_indent
        prev_indent = indent
        if not is_step:
            continue
        saw_step = True
        if (span := line_spans[ln - 1]) is None:
            continue
        fl = _query_frame_lines(source, span)
        if _has_inner_frame(lines, fl, sl, is_markdown=False):
            return fl
    if not saw_step and len(non_empty) > 1 and (tail_span := line_spans[non_empty[-1] - 1]) is not None:
        tail_fl = _query_frame_lines(source, tail_span)
        if _has_inner_frame(lines, tail_fl, sl, is_markdown=False):
            return tail_fl
    return None


def _probe_opening_frames(
    lines: Sequence[str],
    line_spans: Sequence[tuple[TextPosition, TextPosition] | None],
    source: CodeSource,
    non_empty: Sequence[int],
    sl: int,
    *,
    is_markdown: bool,
) -> list[int]:
    if not non_empty:
        return []
    span0 = line_spans[non_empty[0] - 1]
    first_frames = _query_frame_lines(source, span0) if span0 is not None else []
    if _has_inner_frame(lines, first_frames, sl, is_markdown=is_markdown):
        return first_frames
    if is_markdown:
        return _probe_markdown_candidates(lines, line_spans, source, non_empty, sl) or first_frames
    return _probe_code_candidates(lines, line_spans, source, non_empty, sl) or first_frames


def _probe_tail_or_next_frames(
    lines: Sequence[str],
    line_spans: Sequence[tuple[TextPosition, TextPosition] | None],
    source: CodeSource,
    sl: int,
    el: int,
    *,
    is_markdown: bool,
) -> list[int] | None:
    nxt_ln = next((ln for ln in range(el + 1, len(lines) + 1) if line_spans[ln - 1] is not None), None)
    if nxt_ln is not None and (nxt_span := line_spans[nxt_ln - 1]) is not None:
        nxt_frames = _query_frame_lines(source, nxt_span)
        bounded = [f for f in nxt_frames if f <= el]
        if _has_inner_frame(lines, bounded, sl, is_markdown=is_markdown):
            return bounded
    return None


def _probe_chunk_anchor(
    lines: Sequence[str],
    line_spans: Sequence[tuple[TextPosition, TextPosition] | None],
    source: CodeSource,
    sl: int,
    el: int,
    *,
    is_markdown: bool,
) -> tuple[int, str] | None:
    non_empty = [ln for ln in range(sl, el + 1) if line_spans[ln - 1] is not None]
    picked = _probe_opening_frames(lines, line_spans, source, non_empty, sl, is_markdown=is_markdown)
    has_inner = _has_inner_frame(lines, picked, sl, is_markdown=is_markdown)
    if (
        not has_inner
        and (extra_frames := _probe_tail_or_next_frames(lines, line_spans, source, sl, el, is_markdown=is_markdown))
        is not None
    ):
        picked = extra_frames
        has_inner = True
    if (
        is_markdown
        and not has_inner
        and len(non_empty) == 1
        and markdown_heading_level(lines, non_empty[0]) is not None
        and (title := normalize_ast_heading(lines[non_empty[0] - 1]))
    ):
        return non_empty[0], title
    return _resolve_chunk_anchor(lines, picked, sl, is_markdown=is_markdown)


def _emit_bounded_island(
    out: list[tuple[int, int, str]],
    spans: Sequence[tuple[int, int]],
    start_idx: int,
    end_idx: int,
    group_start: int,
    ceil_ln: int,
    title: str,
) -> None:
    if ceil_ln - group_start + 1 <= _MAX_ISLAND_LINES:
        for k in range(start_idx, end_idx):
            out[k] = (group_start, ceil_ln, title)
        return
    k = start_idx
    win_start = group_start
    while k < end_idx:
        sl, el = spans[k]
        if el - win_start + 1 > _MAX_ISLAND_LINES:
            win_start = sl
        win_limit = min(ceil_ln, max(el, win_start + _MAX_ISLAND_LINES - 1))
        j = k + 1
        while j < end_idx and spans[j][1] <= win_limit:
            j += 1
        win_end = max(spans[m][1] for m in range(k, j)) if j < end_idx else win_limit
        for m in range(k, j):
            out[m] = (win_start, win_end, title)
        k = j


def _assign_island_bounds(
    spans: Sequence[tuple[int, int]],
    anchors: Sequence[tuple[int, str] | None],
    lines: Sequence[str],
) -> list[tuple[int, int, str]]:
    n = len(spans)
    out: list[tuple[int, int, str]] = [(*spans[i], "") for i in range(n)]
    idx = 0
    while idx < n:
        anchor = anchors[idx]
        if anchor is None:
            idx += 1
            continue
        end_idx = idx + 1
        while end_idx < n and anchors[end_idx] == anchor:
            end_idx += 1
        anchor_line, title = anchor
        first_sl = min(spans[k][0] for k in range(idx, end_idx))
        min_allowed = min(first_sl, spans[idx - 1][1] + 1 if idx > 0 else 1)
        group_start = max(min_allowed, min(anchor_line, first_sl))
        while group_start < first_sl and not lines[group_start - 1].strip():
            group_start += 1
        max_el = max(spans[k][1] for k in range(idx, end_idx))
        ceil_ln = max(max_el, spans[end_idx][0] - 1) if end_idx < n else len(lines)
        while ceil_ln > max_el and not lines[ceil_ln - 1].strip():
            ceil_ln -= 1
        _emit_bounded_island(out, spans, idx, end_idx, group_start, ceil_ln, title)
        idx = end_idx
    return out


def chunk_file_content(
    rel_path: str,
    raw_text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_size: int = 100,
) -> tuple[SourceChunk, ...]:
    """Use one native CocoIndex AST parse and populate AST heading titles, islands, and source fingerprints."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return ()
    lines = split_lf_lines(text)
    language = detect_code_language(filename=rel_path)
    source = CodeSource(text, language=language)
    chunks = _SPLITTER.split(source, chunk_size, min_chunk_size=min_chunk_size, chunk_overlap=chunk_overlap)
    kept: list[tuple[int, int, int, int, str]] = []
    for chunk in chunks:
        cs, ce = int(chunk.start.char_offset), int(chunk.end.char_offset)
        body = text[cs:ce]
        if not body.strip():
            continue
        size = len(body.encode("utf-8"))
        if size > chunk_size:
            raise ValueError(
                f"{rel_path}:{chunk.start.line}: native chunk is {size} bytes, exceeding chunk_size={chunk_size}"
            )
        sl = int(chunk.start.line)
        el = inclusive_end_line(sl, int(chunk.end.line), int(chunk.end.column))
        kept.append((sl, el, cs, ce, body))
    if not kept:
        return ()

    if language is not None:
        line_spans = _compute_line_positions(lines)
        is_md = language == "markdown"
        anchors = [_probe_chunk_anchor(lines, line_spans, source, sl, el, is_markdown=is_md) for sl, el, *_ in kept]
    else:
        anchors = [None] * len(kept)

    islands = _assign_island_bounds([(sl, el) for sl, el, *_ in kept], anchors, lines)
    out: list[SourceChunk] = []
    for (sl, el, cs, ce, body), (is_s, is_e, title) in zip(kept, islands, strict=True):
        source_hash = fingerprint_str("\n".join(lines[sl - 1 : el])).hex()
        cid = compute_chunk_int_id(rel_path, sl, el, cs, ce, source_hash)
        out.append(SourceChunk(cid, rel_path, source_hash, sl, el, cs, ce, is_s, is_e, title, body))
    return tuple(out)


# Keyed strictly by content (title, body) so line shifts or file moves reuse CocoIndex LMDB memos.
@coco.fn(memo=True, version=5)
async def _compute_chunk_embedding(
    title_body: tuple[str, str],
    task_mode: str,
    embedder: PipelineEmbedder,
) -> bytes:
    title, body = title_body
    formatted = format_doc_for_embedding(body, title=title, task_mode=task_mode)
    vec = await embedder.embed(formatted)
    return np.asarray(vec, dtype="<f2").tobytes()


@coco.fn(memo=True, version=7)
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
