"""1-Indexed disk line hydration with CocoIndex AST island bounds and graceful context dehydration."""

import json
import re
import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from cocoindex.ops.code import CodeMatch, CodeSource, render_match
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk, TextPosition

from yaw.core import (
    ChunkPointer,
    HydratedPassage,
    compute_chunk_id,
    json_document_fields,
    render_document_text,
    resolve_corpus_path,
)

from .chunks import get_chunk_pointer

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


def _is_json_corpus_path(rel_posix_path: str) -> bool:
    return rel_posix_path.lower().endswith((".json", ".jsonl", ".ndjson"))


def coalesce_spans(spans: Sequence[MutableSpan]) -> list[CoalescedSpan]:
    """Merge overlapping or contiguous line spans within each file so no line is duplicated."""
    if not spans:
        return []

    by_file: dict[str, list[MutableSpan]] = {}
    for s in spans:
        by_file.setdefault(s.rel_posix_path, []).append(s)

    coalesced: list[CoalescedSpan] = []
    for rel_path, file_spans in by_file.items():
        is_json_file = _is_json_corpus_path(rel_path)
        sorted_spans = sorted(file_spans, key=lambda s: (s.start_line, -s.end_line))
        cur = sorted_spans[0]
        cur_s, cur_e = cur.start_line, cur.end_line
        cur_title = cur.title
        cur_cids = list(cur.hit_cids or [cur.chunk_id])
        cur_max_score = cur.score
        cur_ast_s = cur.ast_island_start
        cur_ast_e = cur.ast_island_end

        for nxt in sorted_spans[1:]:
            if is_json_file:
                should_merge = cur_s == nxt.start_line and cur_e == nxt.end_line
            else:
                same_title = not cur_title or not nxt.title or cur_title == nxt.title
                same_ast_island = (
                    cur_ast_s > 0 and cur_ast_s == nxt.ast_island_start and cur_ast_e == nxt.ast_island_end
                )
                in_same_island = (
                    (cur_ast_s == 0 or nxt.ast_island_start == 0 or cur_ast_s == nxt.ast_island_start)
                    and (cur_ast_e == 0 or nxt.start_line <= cur_ast_e)
                    and (nxt.ast_island_start == 0 or cur_e >= nxt.ast_island_start - 1)
                )
                is_contiguous = nxt.start_line <= cur_e + 1
                should_merge = same_title and (same_ast_island or (in_same_island and is_contiguous))
            if should_merge:
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


def _decode_json_passage_fields(raw_text: str, default_title: str = "") -> tuple[str, str]:
    raw = raw_text.strip()
    if not raw:
        return "", ""
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(raw)
    remainder = raw[end:].lstrip()
    if isinstance(data, str) and remainder.startswith(":"):
        fields = {}
        cursor = 0
        while cursor < len(raw):
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            if cursor == len(raw):
                break
            key, cursor = decoder.raw_decode(raw, cursor)
            if not isinstance(key, str):
                raise ValueError("Projected JSON property key must be a string")
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            if cursor == len(raw) or raw[cursor] != ":":
                raise ValueError("Projected JSON property requires a colon")
            cursor += 1
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            fields[key], cursor = decoder.raw_decode(raw, cursor)
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            if cursor < len(raw):
                if raw[cursor] != ",":
                    raise ValueError("Invalid projected JSON property separator")
                cursor += 1
        data = fields
    elif remainder not in ("", ","):
        raise ValueError("Unexpected content after JSON source value")
    return json_document_fields(data, default_title)


def normalize_json_passage_text(raw_text: str, default_title: str = "") -> str:
    """Decode a document object or an explicitly projected property sequence.

    Source-aligned properties may retain their JSON separator commas. This is
    a defined projection grammar, not malformed-document recovery.
    """
    title, body = _decode_json_passage_fields(raw_text, default_title)
    if not title and not body:
        return ""
    return render_document_text(title, body)


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
    # All splitter attempts share CocoIndex's lazy parsed AST.
    src = CodeSource(body_text, language=lang)
    for step_size in (min(body_budget, 600), max(40, body_budget // 3), 80):
        ast_chunks = _HYDRATION_SPLITTER.split(
            src,
            chunk_size=max(20, step_size),
            chunk_overlap=0,
            min_chunk_size=1,
        )
        if not ast_chunks and body_text.strip():
            raise ValueError(f"{rel_posix_path}: splitter produced no chunks during hydration")

        if ast_chunks:
            last_valid_end_byte = 0
            for ch in ast_chunks:
                cand = body_bytes[: ch.end.byte_offset].decode("utf-8")
                if len(cand) <= body_budget:
                    last_valid_end_byte = ch.end.byte_offset
                else:
                    break
            if last_valid_end_byte > 0:
                best_body = body_bytes[:last_valid_end_byte].decode("utf-8").rstrip()
                break

    if not best_body:
        # Valid AST chunks may all exceed the budget (for example a long word).
        # Crop that valid text at a lexical boundary; parser failures propagate.
        window = body_text[:body_budget]
        for sep in ("\n\n", "\n", ". ", "; ", " "):
            idx = window.rfind(sep)
            if idx >= body_budget // 3:
                best_body = window[: idx + (1 if sep == ". " else 0)].rstrip()
                break
        if not best_body:
            best_body = window.rstrip()

    return f"{header_prefix}{best_body}{notice}"


_CODE_SOURCE_CACHE: OrderedDict[str, tuple[tuple[int, int, int], CodeSource, list[str]]] = OrderedDict()
_HYDRATED_TEXT_CACHE: OrderedDict[tuple[object, ...], str] = OrderedDict()
_HYDRATION_CACHE_LOCK = RLock()
_HYDRATED_CACHE_CHARS = 1_000_000


def _source_revision(target_path: Path) -> tuple[int, int, int]:
    st = target_path.stat()
    return st.st_mtime_ns, st.st_ctime_ns, st.st_size


def get_cached_code_source(target_path: Path) -> tuple[CodeSource, list[str]] | None:
    """Retrieve or construct a cached CocoIndex CodeSource, invalidating on file mtime changes."""
    try:
        revision = _source_revision(target_path)
    except OSError:
        return None
    path_key = str(target_path.resolve())
    with _HYDRATION_CACHE_LOCK:
        cached = _CODE_SOURCE_CACHE.get(path_key)
        if cached is not None and cached[0] == revision:
            _CODE_SOURCE_CACHE.move_to_end(path_key)
            return cached[1], cached[2]
        _CODE_SOURCE_CACHE.pop(path_key, None)
    try:
        text = target_path.read_text(encoding="utf-8")
    except OSError:
        return None
    src = CodeSource(text)
    lines = src.text.splitlines(keepends=True)
    with _HYDRATION_CACHE_LOCK:
        _CODE_SOURCE_CACHE[path_key] = (revision, src, lines)
        # Bound retained source text as well as file count. Large single files
        # are usable but not retained after the current hydration operation.
        while len(_CODE_SOURCE_CACHE) > 4 or sum(v[0][2] for v in _CODE_SOURCE_CACHE.values()) > 32_000_000:
            _CODE_SOURCE_CACHE.popitem(last=False)
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
    except OSError:
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
    try:
        revision = _source_revision(target_path)
    except OSError:
        return HydratedPassage(pointer=pointer, text="", score=score, stale_on_disk=True)
    cache_key = (str(target_path.resolve()), revision, pointer.start_line, pointer.end_line, pointer.title, max_chars)
    with _HYDRATION_CACHE_LOCK:
        cached_text = _HYDRATED_TEXT_CACHE.get(cache_key)
        if cached_text is not None:
            _HYDRATED_TEXT_CACHE.move_to_end(cache_key)
            return HydratedPassage(pointer=pointer, text=cached_text, score=score, stale_on_disk=False)
    text = read_corpus_lines(target_path, pointer.start_line, pointer.end_line)
    if text is None:
        return HydratedPassage(pointer=pointer, text="", score=score, stale_on_disk=True)

    if _is_json_corpus_path(pointer.rel_posix_path):
        text = normalize_json_passage_text(text, default_title=pointer.title)
    if max_chars > 0 and len(text) > max_chars:
        text = dehydrate_passage_ast(
            text,
            max_chars,
            title=pointer.title,
            rel_posix_path=pointer.rel_posix_path,
        )
    # Cache only bounded rendered text, never pointers, scores or entire ASTs.
    # Re-check the revision to avoid storing content read during a source edit.
    if len(text) <= _HYDRATED_CACHE_CHARS:
        try:
            unchanged = _source_revision(target_path) == revision
        except OSError:
            unchanged = False
        if unchanged:
            with _HYDRATION_CACHE_LOCK:
                _HYDRATED_TEXT_CACHE[cache_key] = text
                while (
                    len(_HYDRATED_TEXT_CACHE) > 256
                    or sum(map(len, _HYDRATED_TEXT_CACHE.values())) > _HYDRATED_CACHE_CHARS
                ):
                    _HYDRATED_TEXT_CACHE.popitem(last=False)
    return HydratedPassage(
        pointer=pointer,
        text=text,
        score=score,
        stale_on_disk=False,
    )


def resolve_chunk_text(
    pointer: ChunkPointer,
    corpus_root: Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> str:
    """Resolve the exact chunk text for a ChunkPointer (including split JSON/JSONL sub-chunks)."""
    hydrated = hydrate_chunk_pointer(pointer, corpus_root)
    if hydrated.stale_on_disk or not hydrated.text.strip():
        raise ValueError(f"Failed to hydrate source chunk {pointer.chunk_id} from {pointer.rel_posix_path}")
    if _is_json_corpus_path(pointer.rel_posix_path):
        full_unbounded = hydrate_chunk_pointer(pointer, corpus_root, max_chars=0)
        if (
            not full_unbounded.stale_on_disk
            and compute_chunk_id(pointer.rel_posix_path, pointer.start_line, pointer.end_line, full_unbounded.text)
            != pointer.chunk_id
        ):
            target_path = resolve_corpus_path(corpus_root, pointer.rel_posix_path)
            raw_slice = read_corpus_lines(target_path, pointer.start_line, pointer.end_line)
            if raw_slice is not None:
                title, body = _decode_json_passage_fields(raw_slice, default_title=pointer.title)
                if body.strip():
                    c_size = max(1, int(chunk_size))
                    c_overlap = max(0, min(int(chunk_overlap), c_size - 1))
                    m_size = max(1, min(int(min_chunk_size), c_size))
                    effective_size = max(m_size, c_size - len(title) - 1) if title else c_size
                    for ch in _HYDRATION_SPLITTER.split(
                        CodeSource(body, language="markdown"),
                        chunk_size=effective_size,
                        chunk_overlap=min(c_overlap, max(0, effective_size - 1)),
                        min_chunk_size=min(m_size, max(1, len(body.strip()))),
                    ):
                        cand_text = render_document_text(title, ch.text)
                        if (
                            compute_chunk_id(pointer.rel_posix_path, pointer.start_line, pointer.end_line, cand_text)
                            == pointer.chunk_id
                        ):
                            return cand_text
    return hydrated.text


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


_ATX_HEADING_LEVEL_RE = re.compile(r"^ {0,3}(#{1,6})(?:\s+(.*?))?(?:\s+#+\s*)?$")


def resolve_one_hop_upward_ast_bounds(
    target_path: Path,
    pointer: ChunkPointer,
    *,
    hops: int = 1,
) -> tuple[int, int, str]:
    """Expand `hops` levels upwards in the document AST (unless already at the root or `hops <= 0`).

    Returns `(parent_start_line, parent_end_line, parent_heading_title)`.
    - If `hops <= 0`, returns `(pointer.start_line, pointer.end_line, pointer.title)`.
    - If `pointer` is already at the root (`start_line <= 1` and `ast_island_start <= 1`),
      returns `(1, max(end_line, ast_island_end), pointer.title)`.
    - If `pointer.start_line > pointer.ast_island_start`, 1 hop upwards in the AST is the
      enclosing AST island `[ast_island_start, ast_island_end]`.
    - If `pointer.start_line == pointer.ast_island_start > 1`, climbs one level up the
      CocoIndex Tree-sitter AST frame / Markdown ATX heading hierarchy toward the document root.
    """
    if hops <= 0:
        return pointer.start_line, pointer.end_line, pointer.title
    if _is_json_corpus_path(pointer.rel_posix_path):
        ast_s = max(1, pointer.ast_island_start if pointer.ast_island_start > 0 else pointer.start_line)
        ast_e = max(pointer.end_line, pointer.ast_island_end if pointer.ast_island_end > 0 else pointer.end_line)
        return min(ast_s, pointer.start_line), max(ast_e, pointer.end_line), pointer.title
    curr_ptr = pointer
    parent_s, parent_e, parent_title = pointer.start_line, pointer.end_line, pointer.title
    for _ in range(hops):
        ast_s = max(1, curr_ptr.ast_island_start if curr_ptr.ast_island_start > 0 else curr_ptr.start_line)
        ast_e = max(curr_ptr.end_line, curr_ptr.ast_island_end if curr_ptr.ast_island_end > 0 else curr_ptr.end_line)
        if curr_ptr.start_line <= 1 and ast_s <= 1:
            return 1, ast_e, curr_ptr.title

        cached = get_cached_code_source(target_path) if target_path.is_file() else None
        if cached is None:
            parent_s = ast_s if curr_ptr.start_line > ast_s else 1
            return parent_s, ast_e, curr_ptr.title

        _, lines = cached
        total_lines = len(lines)
        ast_e = min(total_lines, max(ast_e, curr_ptr.end_line))

        if curr_ptr.start_line > ast_s:
            h_line = lines[ast_s - 1].strip() if 1 <= ast_s <= total_lines else ""
            m_h = _ATX_HEADING_LEVEL_RE.match(h_line)
            p_title = (m_h.group(2) or "").strip() if m_h else (curr_ptr.title or h_line[:80])
            parent_s, parent_e, parent_title = ast_s, ast_e, p_title
        else:
            curr_line_text = lines[ast_s - 1].rstrip("\r\n") if 1 <= ast_s <= total_lines else ""
            m_curr = _ATX_HEADING_LEVEL_RE.match(curr_line_text)
            resolved_step = False
            if m_curr is not None:
                curr_level = len(m_curr.group(1))
                for idx in range(ast_s - 2, -1, -1):
                    m_prev = _ATX_HEADING_LEVEL_RE.match(lines[idx].rstrip("\r\n"))
                    if m_prev is not None and len(m_prev.group(1)) < curr_level:
                        parent_s, parent_e, parent_title = idx + 1, ast_e, (m_prev.group(2) or "").strip()
                        resolved_step = True
                        break
                if not resolved_step:
                    parent_s, parent_e, parent_title = 1, ast_e, curr_ptr.title
                    resolved_step = True

            if not resolved_step:
                lang = detect_code_language(filename=curr_ptr.rel_posix_path)
                if lang is not None:
                    full_text = "".join(lines)
                    lang_src = CodeSource(full_text, language=lang)
                    byte_start = sum(len(line.encode("utf-8")) for line in lines[: ast_s - 1])
                    char_start = sum(map(len, lines[: ast_s - 1]))
                    anchor = lines[ast_s - 1].rstrip("\r\n")
                    anchor_match = CodeMatch(
                        kind="scope_anchor",
                        chunks=[
                            Chunk(
                                text=anchor,
                                start=TextPosition(byte_start, char_start, ast_s, 1),
                                end=TextPosition(
                                    byte_start + len(anchor.encode("utf-8")),
                                    char_start + len(anchor),
                                    ast_s,
                                    len(anchor) + 1,
                                ),
                            )
                        ],
                        captures={},
                    )
                    view = render_match(lang_src, anchor_match)
                    enclosing_frames = [
                        seg.start.line
                        for seg in view.segments
                        if seg.kind == "frame" and seg.summary is None and seg.start.line < ast_s
                    ]
                    if enclosing_frames:
                        parent_line = max(enclosing_frames)
                        p_text = lines[parent_line - 1].strip().rstrip("{:").strip()
                        parent_s, parent_e, parent_title = parent_line, ast_e, p_text or curr_ptr.title
                        resolved_step = True
            if not resolved_step:
                parent_s, parent_e, parent_title = 1, ast_e, curr_ptr.title

        if parent_s <= 1:
            break
        curr_ptr = ChunkPointer(
            chunk_id=curr_ptr.chunk_id,
            rel_posix_path=curr_ptr.rel_posix_path,
            start_line=parent_s,
            end_line=parent_e,
            title=parent_title,
            ast_island_start=parent_s,
            ast_island_end=parent_e,
        )
    return parent_s, parent_e, parent_title


def hydrate_ast_parent_context(
    pointer: ChunkPointer,
    corpus_root: Path,
    *,
    hops: int = 0,
    max_chars: int = 1600,
) -> str:
    """Hydrate `hops`-level upward parent AST header/preamble lines preceding `pointer.start_line`.

    Defaults to `hops=0` (returns `""` unless `hops > 0` and `pointer.start_line > 1`).
    """
    if hops <= 0 or pointer.start_line <= 1 or _is_json_corpus_path(pointer.rel_posix_path):
        return ""
    target_path = resolve_corpus_path(corpus_root, pointer.rel_posix_path)
    if not target_path.is_file():
        return ""
    parent_s, _parent_e, _parent_title = resolve_one_hop_upward_ast_bounds(target_path, pointer, hops=hops)
    preamble_end = pointer.start_line - 1
    if preamble_end < parent_s:
        return ""

    parts: list[str] = []
    if parent_s > 1:
        root_end = min(parent_s - 1, 8)
        root_head = read_corpus_lines(target_path, 1, root_end)
        if root_head and root_head.strip():
            parts.append(root_head.strip())
    parent_head = read_corpus_lines(target_path, parent_s, min(preamble_end, parent_s + 14))
    if parent_head and parent_head.strip():
        parts.append(parent_head.strip())

    combined = "\n...\n".join(parts)
    if not combined:
        return ""
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + " ..."
    return combined


def make_chunk_hydrator(
    conn: sqlite3.Connection,
    corpus_root: Path,
) -> Callable[[str], str]:
    """Create a callable `hydrate_chunk(chunk_id: str) -> str` tool for dspy.RLM passage inspection."""

    def hydrate_chunk(chunk_id: str) -> str:
        pointer = get_chunk_pointer(conn, chunk_id)
        if pointer is None:
            return f"Error: Chunk '{chunk_id}' not found in database."
        passage = hydrate_chunk_pointer(pointer, corpus_root)
        if passage.stale_on_disk:
            return f"Error: Chunk '{chunk_id}' file is missing or unreadable on disk."
        return passage.text

    return hydrate_chunk


def make_ast_island_hydrator(
    conn: sqlite3.Connection,
    corpus_root: Path,
    *,
    max_chars: int = _DEFAULT_MAX_ISLAND_CHARS,
) -> Callable[[str], str]:
    """Create a callable `hydrate_ast_island(chunk_id: str) -> str` expanding 1 hop upwards in the document AST."""

    def hydrate_ast_island(chunk_id: str) -> str:
        pointer = get_chunk_pointer(conn, chunk_id)
        if pointer is None:
            return f"Error: Chunk '{chunk_id}' not found in database."
        target_path = resolve_corpus_path(corpus_root, pointer.rel_posix_path)
        parent_s, parent_e, parent_title = resolve_one_hop_upward_ast_bounds(target_path, pointer)
        expanded_ptr = ChunkPointer(
            chunk_id=pointer.chunk_id,
            rel_posix_path=pointer.rel_posix_path,
            start_line=min(parent_s, pointer.start_line),
            end_line=max(parent_e, pointer.end_line),
            title=parent_title or pointer.title,
            ast_island_start=min(parent_s, pointer.ast_island_start or pointer.start_line),
            ast_island_end=max(parent_e, pointer.ast_island_end or pointer.end_line),
        )
        passage = hydrate_chunk_pointer(expanded_ptr, corpus_root, max_chars=max_chars)
        if passage.stale_on_disk:
            return f"Error: Chunk '{chunk_id}' file is missing or unreadable on disk."
        header = (
            f"# Document: {pointer.rel_posix_path} | AST Section: {expanded_ptr.title or '(root)'} "
            f"(Lines {expanded_ptr.start_line}–{expanded_ptr.end_line}, "
            f"Focus Chunk Lines {pointer.start_line}–{pointer.end_line})\n"
        )
        return f"{header}{passage.text}"

    return hydrate_ast_island


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
        if _is_json_corpus_path(span.rel_posix_path):
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


def format_subgraph_for_qa(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str],
    seed_facts: Sequence[tuple[str, str, str]] = (),
    *,
    max_subgraph_chars: int = 6000,
) -> str:
    """Format the closed-world SKOS/OWL/RDF subgraph induced by `chunk_ids` and `seed_facts` for QA grounding."""
    if not chunk_ids and not seed_facts:
        return ""

    active_entity_ids: set[str] = set()
    seed_triples_set: set[tuple[str, str, str]] = set()

    for item in seed_facts:
        if len(item) >= 3:
            s, p, o = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
            seed_triples_set.add((s, p, o))
            active_entity_ids.add(s)
            active_entity_ids.add(o)

    chunk_list = list(dict.fromkeys(chunk_ids))
    chunk_facts: list[tuple[str, str, str, str, str]] = []
    if chunk_list:
        ph = ",".join("?" for _ in chunk_list)
        fact_rows = conn.execute(
            f"""
            SELECT f.fact_id, f.subj_id, f.pred, f.obj_kind, f.obj_id, f.obj_literal_lex, COALESCE(f.definition, '')
            FROM facts f
            JOIN fact_sources fs ON f.fact_id = fs.fact_id
            WHERE fs.chunk_id IN ({ph})
            """,
            chunk_list,
        ).fetchall()
        for fid, subj, pred, obj_kind, obj_id, obj_lex, definition in fact_rows:
            obj_val = str(obj_id) if obj_kind == "entity" and obj_id else str(obj_lex or "")
            s_str, p_str, o_str = str(subj), str(pred), obj_val
            chunk_facts.append((s_str, p_str, o_str, str(definition or "").strip(), str(fid)))
            active_entity_ids.add(s_str)
            if obj_kind == "entity" and obj_id:
                active_entity_ids.add(str(obj_id))

        ent_rows = conn.execute(
            f"SELECT DISTINCT entity_id FROM entity_labels WHERE chunk_id IN ({ph})",
            chunk_list,
        ).fetchall()
        for (eid,) in ent_rows:
            active_entity_ids.add(str(eid))

    if active_entity_ids:
        ph_ent = ",".join("?" for _ in active_entity_ids)
        mapped_rows = conn.execute(
            f"SELECT entity_id, name FROM entities WHERE entity_id IN ({ph_ent}) OR name IN ({ph_ent})",
            list(active_entity_ids) * 2,
        ).fetchall()
        for eid, _ in mapped_rows:
            active_entity_ids.add(str(eid))

    entity_names: dict[str, str] = {}
    entity_defs: dict[str, str] = {}
    if active_entity_ids:
        ph_act = ",".join("?" for _ in active_entity_ids)
        for eid, name, defn in conn.execute(
            f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({ph_act})",
            list(active_entity_ids),
        ).fetchall():
            entity_names[str(eid)] = str(name).strip() or str(eid)
            if defn:
                entity_defs[str(eid)] = str(defn).strip()

    # SECTION 1: Knowledge Graph Statements (rdf:Statement)
    statement_lines: list[str] = []
    seen_statements: set[tuple[str, str, str]] = set()

    for s, p, o in seed_triples_set:
        fact_def_row = conn.execute(
            """
            SELECT definition FROM facts
            WHERE subj_id = ? AND pred = ? AND (obj_id = ? OR obj_literal_lex = ?)
            LIMIT 1
            """,
            (s, p, o, o),
        ).fetchone()
        f_def = str(fact_def_row[0] or "").strip() if fact_def_row is not None and fact_def_row[0] else ""
        seen_statements.add((s, p, o))
        if f_def:
            statement_lines.append(f"- [{s}, {p}, {o}]: {f_def}")
        else:
            statement_lines.append(f"- [{s}, {p}, {o}]")

    for s, p, o, f_def, _ in chunk_facts:
        f_subj = entity_names.get(s, s)
        f_obj = entity_names.get(o, o)
        if (f_subj, p, f_obj) not in seen_statements:
            seen_statements.add((f_subj, p, f_obj))
            if f_def:
                statement_lines.append(f"- [{f_subj}, {p}, {f_obj}]: {f_def}")
            else:
                statement_lines.append(f"- [{f_subj}, {p}, {f_obj}]")

    # SECTION 2: Taxonomy & Disambiguation Trace (SKOS / OWL)
    trace_lines: list[str] = []
    if active_entity_ids:
        ph_act = ",".join("?" for _ in active_entity_ids)
        edge_rows = conn.execute(
            f"""
            SELECT src_id, dst_id, kind, weight, COALESCE(reasoning, '')
            FROM edges
            WHERE kind IN ('owl:differentFrom', 'skos:broader', 'skos:related')
              AND (src_id IN ({ph_act}) OR dst_id IN ({ph_act}))
            """,
            list(active_entity_ids) * 2,
        ).fetchall()

        # Resolve names and definitions for 1-hop counterpart entities from other chunks
        edge_endpoint_ids = {str(r[0]) for r in edge_rows} | {str(r[1]) for r in edge_rows}
        missing_endpoint_ids = edge_endpoint_ids - set(entity_names.keys())
        if missing_endpoint_ids:
            ph_miss = ",".join("?" for _ in missing_endpoint_ids)
            for eid, name, defn in conn.execute(
                f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({ph_miss})",
                list(missing_endpoint_ids),
            ).fetchall():
                entity_names[str(eid)] = str(name).strip() or str(eid)
                if defn:
                    entity_defs[str(eid)] = str(defn).strip()

        kind_prio = {"owl:differentFrom": 0, "skos:broader": 1, "skos:related": 2}
        seen_pairs: set[tuple[str, str, str]] = set()

        sorted_edge_rows = sorted(
            edge_rows,
            key=lambda r: (
                0 if (str(r[0]) in active_entity_ids and str(r[1]) in active_entity_ids) else 1,
                kind_prio.get(str(r[2]), 9),
            ),
        )

        for src, dst, kind, _, reasoning in sorted_edge_rows:
            src_s, dst_s, kind_s, reas_s = str(src), str(dst), str(kind), str(reasoning or "").strip()
            canon_pair = (
                (min(src_s, dst_s), max(src_s, dst_s), kind_s) if kind_s != "skos:broader" else (src_s, dst_s, kind_s)
            )
            if canon_pair in seen_pairs:
                continue
            seen_pairs.add(canon_pair)

            lbl_a = entity_names.get(src_s, src_s)
            lbl_b = entity_names.get(dst_s, dst_s)

            if kind_s == "owl:differentFrom":
                def_a = entity_defs.get(src_s)
                if not def_a:
                    r = conn.execute("SELECT definition FROM entities WHERE entity_id = ?", (src_s,)).fetchone()
                    def_a = str(r[0] or "").strip() if r is not None and r[0] else ""
                def_b = entity_defs.get(dst_s)
                if not def_b:
                    r = conn.execute("SELECT definition FROM entities WHERE entity_id = ?", (dst_s,)).fetchone()
                    def_b = str(r[0] or "").strip() if r is not None and r[0] else ""

                pieces: list[str] = [f"- [DISJOINT] {lbl_a} owl:differentFrom {lbl_b}"]
                if reas_s:
                    pieces.append(f" — {reas_s}")
                if def_a or def_b:
                    pieces.append(f" ({lbl_a}: {def_a or 'N/A'} | {lbl_b}: {def_b or 'N/A'})")
                trace_lines.append("".join(pieces))
            elif kind_s == "skos:broader":
                if reas_s:
                    trace_lines.append(f"- [HIERARCHY] {lbl_a} skos:broader {lbl_b} — {reas_s}")
                else:
                    trace_lines.append(f"- [HIERARCHY] {lbl_a} skos:broader {lbl_b}")
            elif kind_s == "skos:related":
                if reas_s:
                    trace_lines.append(f"- [RELATED] {lbl_a} skos:related {lbl_b} — {reas_s}")
                else:
                    trace_lines.append(f"- [RELATED] {lbl_a} skos:related {lbl_b}")

    # SECTION 3: Concept Definitions & Aliases (skos:Concept)
    concept_lines: list[str] = []
    if active_entity_ids:
        ph_act = ",".join("?" for _ in active_entity_ids)
        concept_entities = conn.execute(
            f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({ph_act})",
            list(active_entity_ids),
        ).fetchall()

        labels_by_ent: dict[str, dict[str, list[str]]] = {}
        for eid, lbl, lkind in conn.execute(
            f"SELECT entity_id, raw_label, label_kind FROM entity_labels WHERE entity_id IN ({ph_act})",
            list(active_entity_ids),
        ).fetchall():
            labels_by_ent.setdefault(str(eid), {}).setdefault(str(lkind), []).append(str(lbl))

        sorted_concepts = sorted(
            concept_entities,
            key=lambda c: (
                0 if str(c[0]) in {s for s, _, _ in seed_triples_set} | {o for _, _, o in seed_triples_set} else 1,
                0 if str(c[2]).strip() else 1,
                str(c[1]),
            ),
        )

        for eid, name, defn in sorted_concepts:
            eid_s = str(eid)
            pref_name = str(name).strip() or eid_s
            def_str = str(defn or "").strip()
            ent_labels = labels_by_ent.get(eid_s, {})
            alt_labels = sorted({lbl for lbl in ent_labels.get("skos:altLabel", ()) if lbl != pref_name})
            hid_labels = sorted({lbl for lbl in ent_labels.get("skos:hiddenLabel", ()) if lbl != pref_name})

            alias_parts: list[str] = []
            if alt_labels:
                alias_parts.append(f"altLabel: {', '.join(alt_labels)}")
            if hid_labels:
                alias_parts.append(f"hiddenLabel: {', '.join(hid_labels)}")

            alias_clause = f" ({'; '.join(alias_parts)})" if alias_parts else ""
            if def_str:
                concept_lines.append(f"- **{pref_name}**{alias_clause}: {def_str}")
            elif alias_clause:
                concept_lines.append(f"- **{pref_name}**{alias_clause}")

    sections: list[str] = []
    if statement_lines:
        sections.append("### Knowledge Graph Statements (rdf:Statement)\n" + "\n".join(statement_lines))
    if trace_lines:
        sections.append("### Taxonomy & Disambiguation Trace (SKOS / OWL)\n" + "\n".join(trace_lines))
    if concept_lines:
        sections.append("### Concept Definitions & Aliases (skos:Concept)\n" + "\n".join(concept_lines))

    if not sections:
        return ""

    subgraph_text = "\n\n".join(sections)
    if len(subgraph_text) > max_subgraph_chars:
        truncated = subgraph_text[:max_subgraph_chars]
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            truncated = truncated[:last_nl]
        return truncated.rstrip()
    return subgraph_text
