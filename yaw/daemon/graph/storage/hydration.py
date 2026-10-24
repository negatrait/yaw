"""1-Indexed disk line hydration with CocoIndex AST island bounds and graceful context dehydration."""

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cocoindex.ops.code import CodeSource
from cocoindex.ops.text import RecursiveSplitter, detect_code_language

from yaw.core import (
    ChunkPointer,
    HydratedPassage,
    compute_chunk_id,
    resolve_corpus_path,
)

_MAX_CHUNK_LINE_SPAN = 5000
_DEFAULT_MAX_ISLAND_CHARS = 4000
_DEFAULT_MAX_CONTEXT_CHARS = 32000
_AST_CROP_NOTICE = "\n[... AST section cropped due to context window limitations ...]"
_HYDRATION_SPLITTER = RecursiveSplitter()


@dataclass(slots=True)
class MutableSpan:
    """Working span used during priority-ordered 1-hop AST window expansion."""

    chunk_id: str
    rel_posix_path: str
    start_line: int
    end_line: int
    title: str
    score: float
    ast_island_start: int
    ast_island_end: int
    hit_cids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.hit_cids is None:
            self.hit_cids = [self.chunk_id] if self.chunk_id else []

    @property
    def ast_start(self) -> int:
        return self.ast_island_start

    @ast_start.setter
    def ast_start(self, val: int) -> None:
        self.ast_island_start = val

    @property
    def ast_end(self) -> int:
        return self.ast_island_end

    @ast_end.setter
    def ast_end(self, val: int) -> None:
        self.ast_island_end = val


@dataclass(slots=True, frozen=True)
class CoalescedSpan:
    """Merged, non-overlapping 1-indexed line span ready for single-pass disk hydration."""

    rel_posix_path: str
    start_line: int
    end_line: int
    title: str
    max_score: float
    hit_chunk_ids: tuple[str, ...]


def coalesce_spans(spans: Sequence[MutableSpan]) -> list[CoalescedSpan]:
    """Merge overlapping or contiguous line spans within each file so no line is duplicated."""
    if not spans:
        return []

    by_file: dict[str, list[MutableSpan]] = {}
    for s in spans:
        by_file.setdefault(s.rel_posix_path, []).append(s)

    coalesced: list[CoalescedSpan] = []
    for rel_path, file_spans in by_file.items():
        sorted_spans = sorted(file_spans, key=lambda s: (s.start_line, -s.end_line))
        cur = sorted_spans[0]
        cur_s, cur_e = cur.start_line, cur.end_line
        cur_title = cur.title
        cur_cids = list(cur.hit_cids or [cur.chunk_id])
        cur_max_score = cur.score
        cur_ast_s = cur.ast_island_start
        cur_ast_e = cur.ast_island_end

        for nxt in sorted_spans[1:]:
            same_title = not cur_title or not nxt.title or cur_title == nxt.title
            same_ast_island = cur_ast_s > 0 and cur_ast_s == nxt.ast_island_start and cur_ast_e == nxt.ast_island_end
            in_same_island = (
                (cur_ast_s == 0 or nxt.ast_island_start == 0 or cur_ast_s == nxt.ast_island_start)
                and (cur_ast_e == 0 or nxt.start_line <= cur_ast_e)
                and (nxt.ast_island_start == 0 or cur_e >= nxt.ast_island_start - 1)
            )
            is_contiguous = nxt.start_line <= cur_e + 1
            if same_title and (same_ast_island or (in_same_island and is_contiguous)):
                cur_e = max(cur_e, nxt.end_line)
                if not cur_title and nxt.title:
                    cur_title = nxt.title
                cur_cids.extend(nxt.hit_cids or [nxt.chunk_id])
                cur_max_score = max(cur_max_score, nxt.score)
                cur_ast_s = (
                    min(cur_ast_s, nxt.ast_island_start)
                    if cur_ast_s and nxt.ast_island_start
                    else (cur_ast_s or nxt.ast_island_start)
                )
                cur_ast_e = max(cur_ast_e, nxt.ast_island_end)
            else:
                coalesced.append(
                    CoalescedSpan(
                        rel_posix_path=rel_path,
                        start_line=cur_s,
                        end_line=cur_e,
                        title=cur_title,
                        max_score=cur_max_score,
                        hit_chunk_ids=tuple(dict.fromkeys(cur_cids)),
                    )
                )
                cur_s, cur_e = nxt.start_line, nxt.end_line
                cur_title = nxt.title
                cur_cids = list(nxt.hit_cids or [nxt.chunk_id])
                cur_max_score = nxt.score
                cur_ast_s = nxt.ast_island_start
                cur_ast_e = nxt.ast_island_end

        coalesced.append(
            CoalescedSpan(
                rel_posix_path=rel_path,
                start_line=cur_s,
                end_line=cur_e,
                title=cur_title,
                max_score=cur_max_score,
                hit_chunk_ids=tuple(dict.fromkeys(cur_cids)),
            )
        )

    coalesced.sort(key=lambda cs: cs.max_score, reverse=True)
    return coalesced


coalesce_chunk_spans = coalesce_spans


def normalize_json_passage_text(raw_text: str, default_title: str = "") -> str:
    """Normalize raw JSON passage lines hydrated from disk into clean, readable text."""
    stripped = raw_text.strip()
    if not stripped:
        return ""
    if stripped.endswith(",") and not stripped.startswith('"'):
        stripped = stripped[:-1].rstrip()

    try:
        data = json.loads(stripped)
    except Exception:
        try:
            data = json.loads("{" + stripped.rstrip(",") + "}")
        except Exception:
            return (
                f"{default_title}\n{stripped}"
                if default_title and not (stripped.startswith(f"{default_title}\n") or stripped == default_title)
                else stripped
            )

    if isinstance(data, list):
        if len(data) >= 2 and isinstance(data[0], str) and isinstance(data[1], list):
            title = data[0].strip()
            sents = [str(s) for s in data[1] if s is not None]
            sep = "" if any(s.endswith((".", "!", "?", " ")) for s in sents[:-1]) else " "
            body = sep.join(sents).strip()
            return f"{title}\n{body}" if title else body
        if len(data) == 1 and isinstance(data[0], str):
            return data[0].strip()
        return " ".join(str(x) for x in data)

    if isinstance(data, dict):
        title = str(data.get("title") or data.get("name") or default_title).strip()
        body = data.get("text") or data.get("paragraph_text") or data.get("contents")
        if body is not None:
            body_str = str(body).strip()
            return (
                f"{title}\n{body_str}"
                if title and not (body_str.startswith(f"{title}\n") or body_str == title)
                else body_str
            )
        items = [f"{k}: {v}" for k, v in data.items() if v is not None]
        body_str = "\n".join(items)
        return (
            f"{title}\n{body_str}"
            if title and not (body_str.startswith(f"{title}\n") or body_str == title)
            else body_str
        )

    if isinstance(data, str):
        return data.strip()

    return stripped


def _detect_hydration_language(rel_posix_path: str) -> str:
    lang = detect_code_language(filename=rel_posix_path) if rel_posix_path else None
    if lang is None or lang == "json":
        return "markdown"
    return lang


def dehydrate_passage_ast(
    text: str,
    max_chars: int,
    *,
    title: str = "",
    rel_posix_path: str = "",
) -> str:
    """Gracefully dehydrate a passage to fit max_chars at CocoIndex AST boundaries with a crop notice."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    notice = _AST_CROP_NOTICE
    if max_chars <= len(notice) + 40:
        return text[: max(1, max_chars - len(notice))].rstrip() + notice

    budget = max_chars - len(notice)
    header_prefix = ""
    body_text = text

    if "\n" in text:
        first_line, rest = text.split("\n", 1)
        fl_strip = first_line.strip()
        clean_title = title.strip()
        if (
            fl_strip.lower().startswith(("wikipedia title:", "title:"))
            or fl_strip.startswith("#")
            or (clean_title and fl_strip == clean_title)
        ) and len(first_line) + 1 < budget // 2:
            header_prefix = first_line + "\n"
            body_text = rest

    body_budget = max(40, budget - len(header_prefix))
    lang = _detect_hydration_language(rel_posix_path)
    body_bytes = body_text.encode("utf-8")

    best_body = ""
    for step_size in (min(body_budget, 600), max(40, body_budget // 3), 80):
        try:
            src = CodeSource(body_text, language=lang)
            ast_chunks = _HYDRATION_SPLITTER.split(
                src,
                chunk_size=max(20, step_size),
                chunk_overlap=0,
                min_chunk_size=1,
            )
        except Exception:
            ast_chunks = []

        if ast_chunks:
            last_valid_end_byte = 0
            for ch in ast_chunks:
                cand = body_bytes[: ch.end.byte_offset].decode("utf-8", errors="replace")
                if len(cand) <= body_budget:
                    last_valid_end_byte = ch.end.byte_offset
                else:
                    break
            if last_valid_end_byte > 0:
                best_body = body_bytes[:last_valid_end_byte].decode("utf-8", errors="replace").rstrip()
                break

    if not best_body:
        # Fallback to last paragraph or sentence boundary within body_budget
        window = body_text[:body_budget]
        for sep in ("\n\n", "\n", ". ", "; ", " "):
            idx = window.rfind(sep)
            if idx >= body_budget // 3:
                best_body = window[: idx + (1 if sep == ". " else 0)].rstrip()
                break
        if not best_body:
            best_body = window.rstrip()

    return f"{header_prefix}{best_body}{notice}"


_CODE_SOURCE_CACHE: dict[str, tuple[float, CodeSource, list[str]]] = {}


def get_cached_code_source(target_path: Path) -> tuple[CodeSource, list[str]] | None:
    """Retrieve or construct a cached CocoIndex CodeSource, invalidating on file mtime changes."""
    try:
        mtime = target_path.stat().st_mtime
    except OSError:
        return None
    path_key = str(target_path)
    cached = _CODE_SOURCE_CACHE.get(path_key)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]
    try:
        text = target_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    src = CodeSource(text)
    lines = src.text.splitlines(keepends=True)
    _CODE_SOURCE_CACHE[path_key] = (mtime, src, lines)
    return src, lines


def read_corpus_lines(
    target_path: Path,
    start_line: int,
    end_line: int,
) -> str | None:
    """Read a 1-indexed [start_line, end_line] slice from a corpus file using CocoIndex CodeSource caching.

    Validates mtime on each call, avoiding repeated open/close disk thrashing.
    Returns the newline-stripped content string, or None if the file is missing/unreadable or the slice is empty.
    """
    if not target_path.is_file():
        return None
    try:
        cached = get_cached_code_source(target_path)
        if cached is None:
            return None
        _, lines = cached
        if start_line < 1 or start_line - 1 >= len(lines):
            return None
        raw_text = "".join(lines[start_line - 1 : end_line])
    except OSError, UnicodeDecodeError:
        return None

    if not raw_text:
        return None
    return raw_text.rstrip("\r\n")


def hydrate_chunk_pointer(
    pointer: ChunkPointer,
    corpus_root: Path,
    score: float = 0.0,
    *,
    max_chars: int = _DEFAULT_MAX_ISLAND_CHARS,
) -> HydratedPassage:
    """Hydrate a 1-indexed [start_line, end_line] AST island from disk, gracefully dehydrating if oversized."""
    if pointer.start_line < 1 or pointer.end_line < pointer.start_line:
        raise ValueError(f"Invalid line span [{pointer.start_line}, {pointer.end_line}] for chunk {pointer.chunk_id}")
    line_count = pointer.end_line - pointer.start_line + 1
    if line_count > _MAX_CHUNK_LINE_SPAN:
        raise ValueError(f"Line span {line_count} exceeds maximum {_MAX_CHUNK_LINE_SPAN} for chunk {pointer.chunk_id}")

    target_path = resolve_corpus_path(corpus_root, pointer.rel_posix_path)
    text = read_corpus_lines(target_path, pointer.start_line, pointer.end_line)
    if text is None:
        return HydratedPassage(pointer=pointer, text="", score=score, stale_on_disk=True)

    if pointer.rel_posix_path.lower().endswith((".json", ".jsonl")):
        text = normalize_json_passage_text(text, default_title=pointer.title)
    if max_chars > 0 and len(text) > max_chars:
        text = dehydrate_passage_ast(
            text,
            max_chars,
            title=pointer.title,
            rel_posix_path=pointer.rel_posix_path,
        )
    return HydratedPassage(
        pointer=pointer,
        text=text,
        score=score,
        stale_on_disk=False,
    )


def hydrate_passages(
    pointers_with_scores: Sequence[tuple[ChunkPointer, float]],
    corpus_root: Path,
    *,
    max_island_chars: int = _DEFAULT_MAX_ISLAND_CHARS,
    max_total_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> tuple[HydratedPassage, ...]:
    """Batch-hydrate ChunkPointers into AST-bounded HydratedPassages with graceful context window dehydration."""
    hydrated: list[HydratedPassage] = []
    used_chars = 0
    for ptr, score in pointers_with_scores:
        hp = hydrate_chunk_pointer(ptr, corpus_root, score, max_chars=max_island_chars)
        if hp.stale_on_disk or not hp.text:
            hydrated.append(hp)
            continue
        if max_total_chars > 0:
            remaining = max_total_chars - used_chars
            if remaining < 240:
                break
            if len(hp.text) > remaining:
                dehydrated_text = dehydrate_passage_ast(
                    hp.text,
                    remaining,
                    title=ptr.title,
                    rel_posix_path=ptr.rel_posix_path,
                )
                hp = HydratedPassage(
                    pointer=hp.pointer,
                    text=dehydrated_text,
                    score=hp.score,
                    stale_on_disk=False,
                )
        used_chars += len(hp.text) + 2
        hydrated.append(hp)
    return tuple(hydrated)


def make_chunk_hydrator(
    conn: sqlite3.Connection,
    corpus_root: Path,
) -> Callable[[str], str]:
    """Create a callable `hydrate_chunk(chunk_id: str) -> str` tool for dspy.RLM passage inspection."""
    from yaw.daemon.graph.storage.db import get_chunk_pointer

    def hydrate_chunk(chunk_id: str) -> str:
        pointer = get_chunk_pointer(conn, chunk_id)
        if pointer is None:
            return f"Error: Chunk '{chunk_id}' not found in database."
        passage = hydrate_chunk_pointer(pointer, corpus_root)
        if passage.stale_on_disk:
            return f"Error: Chunk '{chunk_id}' file is missing or unreadable on disk."
        return passage.text

    return hydrate_chunk


def _hydrate_spans(
    spans: Sequence[CoalescedSpan],
    corpus_root: Path,
) -> list[HydratedPassage]:
    passages: list[HydratedPassage] = []
    for span in spans:
        cid = (
            span.hit_chunk_ids[0]
            if span.hit_chunk_ids
            else compute_chunk_id(span.rel_posix_path, span.start_line, span.end_line, span.title)
        )
        ptr = ChunkPointer(
            chunk_id=cid,
            rel_posix_path=span.rel_posix_path,
            start_line=span.start_line,
            end_line=span.end_line,
            title=span.title,
            ast_island_start=span.start_line,
            ast_island_end=span.end_line,
        )
        target_path = resolve_corpus_path(corpus_root, span.rel_posix_path)
        text = read_corpus_lines(target_path, span.start_line, span.end_line)
        if text is None:
            passages.append(HydratedPassage(pointer=ptr, text="", score=span.max_score, stale_on_disk=True))
            continue
        if span.rel_posix_path.lower().endswith((".json", ".jsonl")):
            text = normalize_json_passage_text(text, default_title=span.title)
        passages.append(HydratedPassage(pointer=ptr, text=text, score=span.max_score, stale_on_disk=False))
    return passages


def hydrate_budgeted_passages(
    pointers_with_scores: Sequence[tuple[ChunkPointer, float]],
    corpus_root: str | Path,
    *,
    max_context_chars: int = 24000,
    max_expansion_hops: int = 1,
) -> list[HydratedPassage]:
    """Hydrate retrieved chunks via greedy priority AST-expansion:
    1. Sort chunks by relevance descending (FIFO queue).
    2. Pick the most relevant remaining chunk.
    3. Expand ONCE to immediate neighbors (max_expansion_hops=1, non-recursive) respecting AST boundaries \
if headroom permits.
    4. Deduplicate / coalesce contiguous spans (MaxP score inheritance).
    5. Inviolate Hit Invariant: retrieved hit chunks are never dropped.
"""
    if not pointers_with_scores:
        return []

    root_path = Path(corpus_root)

    # 1. Sort chunks by relevance descending
    sorted_hits = sorted(pointers_with_scores, key=lambda x: x[1], reverse=True)

    # Initialize active spans from raw hits (Inviolate Hit Invariant)
    active_spans: list[MutableSpan] = []
    for ptr, score in sorted_hits:
        ast_s = ptr.ast_island_start if ptr.ast_island_start > 0 else ptr.start_line
        ast_e = ptr.ast_island_end if ptr.ast_island_end > 0 else ptr.end_line
        active_spans.append(
            MutableSpan(
                chunk_id=ptr.chunk_id,
                rel_posix_path=ptr.rel_posix_path,
                start_line=ptr.start_line,
                end_line=ptr.end_line,
                title=ptr.title,
                score=score,
                ast_island_start=min(ast_s, ptr.start_line),
                ast_island_end=max(ast_e, ptr.end_line),
                hit_cids=[ptr.chunk_id],
            )
        )

    # 2. Greedily expand most relevant candidates ONCE (non-recursive) while headroom permits
    def estimate_total_chars(spans: list[MutableSpan]) -> int:
        return sum((s.end_line - s.start_line + 1) * 80 for s in spans)

    current_len = estimate_total_chars(active_spans)

    if max_expansion_hops >= 1:
        for span in active_spans:
            if current_len >= max_context_chars:
                break
            # Expand span ONCE outward to immediate neighbor lines (±10 lines) bounded by AST
            new_start = max(span.ast_island_start, span.start_line - 10)
            new_end = min(span.ast_island_end, span.end_line + 10)
            added_lines = (span.start_line - new_start) + (new_end - span.end_line)
            added_chars = added_lines * 80
            if current_len + added_chars <= max_context_chars:
                span.start_line = new_start
                span.end_line = new_end
                current_len += added_chars

    # 3. Deduplicate and coalesce overlapping / contiguous spans
    final_spans = coalesce_spans(active_spans)

    # 4. Hydrate exact line spans directly from disk
    return _hydrate_spans(final_spans, root_path)
