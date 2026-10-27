"""Textbook CocoIndex v1 incremental line-pointer and native Tree-sitter AST ingestion pipeline."""

import bisect
import hashlib
import inspect
import json
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import localfs
from cocoindex.ops.code import CodeMatch, CodePattern, CodeSource, render_match
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk, TextPosition
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.core import (
    DEFAULT_EXCLUDED_PATTERNS,
    DEFAULT_INCLUDED_PATTERNS,
    ChunkPointer,
    SyncDelta,
    compute_chunk_id,
    inclusive_end_line,
    json_document_fields,
    render_document_text,
)

_SPLITTER = RecursiveSplitter()
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s+(.*?))?(?:\s+#+\s*)?$")
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)\s*$")
_PLAIN_TITLE_LINE_RE = re.compile(r"^\s*(?:title\s*:\s*(.+)|#+\s*(.+))\s*$", re.IGNORECASE)

_HTML_HEADING_PATTERNS: tuple[CodePattern, ...] = (
    CodePattern(r"<title>\TEXT</title>", language="html", min_len=1),
    *(CodePattern(rf"<h{lvl}>\TEXT</h{lvl}>", language="html", min_len=1) for lvl in range(1, 7)),
)
_TOML_SECTION_PATTERNS: tuple[CodePattern, ...] = (
    CodePattern(r"[\SECTION]", language="toml", min_len=1),
    CodePattern(r"[[\SECTION]]", language="toml", min_len=1),
)
_YAML_PAIR_PATTERN = CodePattern(r"\KEY: \VAL", language="yaml", min_len=1)
_JSON_PAIR_PATTERN = CodePattern(r"\Q\KEY\Q: \VAL", language="json", min_len=1)

_ChunkValue = tuple[ChunkPointer, str]
_ChunkAction = tuple[str, _ChunkValue | None]

_SINK_KEY = coco.ContextKey[list[_ChunkAction]]("yaw_chunk_sink")
_CORPUS_ROOT_KEY = coco.ContextKey[Path]("yaw_corpus_root")
_COMPILER_KEY = coco.ContextKey[Callable[[SyncDelta], None] | None]("yaw_delta_compiler")
_COMPILER_ERROR_KEY = coco.ContextKey[list[Exception]]("yaw_delta_compiler_errors")


class _ChunkTargetHandler(coco.TargetHandler[_ChunkValue, bytes]):
    """CocoIndex target handler managing incremental chunk lifecycle via statediff."""

    def __init__(self) -> None:
        self._sink = coco.TargetActionSink.from_fn(self._apply_actions)

    def _apply_actions(self, context_provider: coco.ContextProvider, actions: Sequence[_ChunkAction]) -> None:
        compiler = context_provider.get(_COMPILER_KEY)
        if compiler is not None and actions:
            added_chunks: list[tuple[ChunkPointer, str]] = []
            deleted_chunk_ids: list[str] = []
            for chunk_id, value in actions:
                if value is not None:
                    ptr, chunk_text = value
                    added_chunks.append((ptr, chunk_text))
                else:
                    deleted_chunk_ids.append(chunk_id)
            delta = SyncDelta(
                added_chunks=tuple(added_chunks),
                deleted_chunk_ids=tuple(deleted_chunk_ids),
            )
            try:
                compiler(delta)
            except Exception as exc:
                context_provider.get(_COMPILER_ERROR_KEY).append(exc)
                raise
        context_provider.get(_SINK_KEY).extend(actions)

    def reconcile(
        self,
        key: coco.StableKey,
        desired_target_state: _ChunkValue | coco.NonExistenceType,
        prev_possible_records: Collection[bytes],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[Any, Any, Any] | None:
        desired_fp: bytes | coco.NonExistenceType = (
            coco.NON_EXISTENCE
            if coco.is_non_existence(desired_target_state)
            else fingerprint_object(desired_target_state)
        )
        action = statediff.diff(
            statediff.TrackingRecordTransition(
                desired=desired_fp,
                prev=prev_possible_records,
                prev_may_be_missing=prev_may_be_missing,
            )
        )
        if action is None:
            return None

        chunk_id = str(key)
        if action == "delete":
            return coco.TargetReconcileOutput(
                action=(chunk_id, None),
                sink=self._sink,
                tracking_record=coco.NON_EXISTENCE,
            )

        assert not coco.is_non_existence(desired_target_state)
        assert not coco.is_non_existence(desired_fp)
        return coco.TargetReconcileOutput(
            action=(chunk_id, desired_target_state),
            sink=self._sink,
            tracking_record=desired_fp,
        )


_CHUNK_HANDLER = _ChunkTargetHandler()
_CHUNK_PROVIDER = coco.register_root_target_states_provider("yaw/corpus/chunk", _CHUNK_HANDLER)


def _declare_chunk(key: coco.StableKey, value: _ChunkValue) -> None:
    target_state = _CHUNK_PROVIDER.target_state(key, value)
    coco.declare_target_state(target_state)


def detect_corpus_language(rel_posix_path: str) -> str | None:
    """Detect Tree-sitter language name from file path via CocoIndex, including .jsonl."""
    lang = detect_code_language(filename=rel_posix_path)
    if lang is None and Path(rel_posix_path).suffix.lower() in (".jsonl", ".ndjson"):
        return "json"
    return lang


def _extract_markdown_ast_headings(source: str | CodeSource) -> list[tuple[int, str]]:
    """Extract ATX and Setext headings via CocoIndex Rust AST while ignoring fenced code blocks."""
    raw_text = source.text if isinstance(source, CodeSource) else source.replace("\r\n", "\n")
    if not raw_text.strip():
        return []

    headings_by_line: dict[int, str] = {}
    in_fence_marker: tuple[str, int] | None = None
    prev_non_fenced_line: tuple[int, str] | None = None

    for line_idx, line in enumerate(raw_text.splitlines(keepends=True), start=1):
        stripped_line = line.rstrip("\r\n")
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", stripped_line)
        is_fence_delimiter = False
        if fence is not None:
            is_fence_delimiter = True
            if in_fence_marker is None:
                in_fence_marker = (fence[1][0], len(fence[1]))
            elif fence[1][0] == in_fence_marker[0] and len(fence[1]) >= in_fence_marker[1] and not fence[2].strip():
                in_fence_marker = None

        in_ast_code_fence = in_fence_marker is not None or is_fence_delimiter
        if not in_ast_code_fence:
            m_line_atx = _ATX_HEADING_RE.match(stripped_line)
            if m_line_atx is not None:
                if line_idx not in headings_by_line:
                    headings_by_line[line_idx] = (m_line_atx.group(1) or "").strip()
                prev_non_fenced_line = None
            elif (
                _SETEXT_UNDERLINE_RE.match(stripped_line) is not None
                and prev_non_fenced_line is not None
                and prev_non_fenced_line[0] == line_idx - 1
            ):
                p_idx, p_text = prev_non_fenced_line
                if p_idx not in headings_by_line:
                    headings_by_line[p_idx] = p_text
                prev_non_fenced_line = None
            elif stripped_line.strip():
                prev_non_fenced_line = (line_idx, stripped_line.strip())
            else:
                prev_non_fenced_line = None
        else:
            prev_non_fenced_line = None

    return sorted(headings_by_line.items(), key=lambda h: h[0])


def extract_ast_headings(
    source: str | CodeSource,
    language: str | None = None,
) -> list[tuple[int, str]]:
    """Extract sorted [(1_indexed_start_line, title)] AST headings/scopes across all languages."""
    if isinstance(source, CodeSource):
        clean_text = source.text
        lang = language or source.language
        code_src: CodeSource | None = source
    else:
        clean_text = source.replace("\r\n", "\n")
        lang = language
        code_src = None

    if not clean_text.strip():
        return []

    if lang == "markdown":
        return _extract_markdown_ast_headings(code_src if code_src is not None else clean_text)

    if lang is None:
        first_line = clean_text.splitlines()[0] if clean_text.splitlines() else ""
        m_plain = _PLAIN_TITLE_LINE_RE.match(first_line)
        if m_plain is not None:
            title_val = (m_plain.group(1) or m_plain.group(2) or "").strip()
            if title_val:
                return [(1, title_val)]
        return []

    if code_src is None or code_src.language != lang:
        code_src = CodeSource(clean_text, language=lang)

    headings_by_line: dict[int, str] = {}
    if lang == "html":
        for pat in _HTML_HEADING_PATTERNS:
            for m in pat.match_source(code_src):
                cap = m.captures.get("TEXT")
                if cap:
                    line_no = cap[0].start.line
                    txt = cap[0].text.strip()
                    if txt and line_no not in headings_by_line:
                        headings_by_line[line_no] = txt
        return sorted(headings_by_line.items(), key=lambda h: h[0])

    if lang == "toml":
        for pat in _TOML_SECTION_PATTERNS:
            for m in pat.match_source(code_src):
                cap = m.captures.get("SECTION")
                if cap:
                    line_no = cap[0].start.line
                    txt = cap[0].text.strip()
                    if txt and line_no not in headings_by_line:
                        headings_by_line[line_no] = txt
        return sorted(headings_by_line.items(), key=lambda h: h[0])

    if lang == "yaml":
        for m in _YAML_PAIR_PATTERN.match_source(code_src):
            cap = m.captures.get("KEY")
            if cap and cap[0].start.column == 1:
                line_no = cap[0].start.line
                txt = cap[0].text.strip().strip("\"'")
                if txt and line_no not in headings_by_line:
                    headings_by_line[line_no] = txt
        return sorted(headings_by_line.items(), key=lambda h: h[0])

    if lang == "json":
        for m in _JSON_PAIR_PATTERN.match_source(code_src):
            k_cap = m.captures.get("KEY")
            v_cap = m.captures.get("VAL")
            if k_cap and v_cap and k_cap[0].text == "title":
                raw_v = v_cap[0].text.strip()
                if len(raw_v) >= 2 and raw_v.startswith('"') and raw_v.endswith('"'):
                    line_no = m.chunks[0].start.line
                    headings_by_line[line_no] = json.loads(raw_v)
        return sorted(headings_by_line.items(), key=lambda h: h[0])

    # General Tree-sitter programming languages: extract enclosing AST scope frames
    src_text = code_src.text
    pos = 0
    char_pos = 0
    ranges = []
    previous_indent = -1
    previous_text = ""
    for line_number, line in enumerate(clean_text.splitlines(keepends=True), start=1):
        b_len = len(line.encode("utf-8"))
        content_len = len(line.rstrip("\r\n").encode("utf-8"))
        indent = len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())
        if content_len > 0 and (
            indent > previous_indent or previous_text.endswith(("{", ":")) or line.lstrip().startswith("{")
        ):
            anchor = line.rstrip("\r\n")
            ranges.append(
                CodeMatch(
                    kind="scope_anchor",
                    chunks=[
                        Chunk(
                            text=anchor,
                            start=TextPosition(pos, char_pos, line_number, 1),
                            end=TextPosition(pos + content_len, char_pos + len(anchor), line_number, len(anchor) + 1),
                        )
                    ],
                    captures={},
                )
            )
        if line.strip():
            previous_indent = indent
            previous_text = line.rstrip()
        pos += b_len
        char_pos += len(line)
    # Rendering returns frames around the ranges' common envelope. Combining
    # unrelated scopes loses their headers; inspect indentation anchors instead
    # of every body line, sharing the same native AST across all inspections.
    for anchor_match in ranges:
        view = render_match(code_src, anchor_match)
        for seg in view.segments:
            if seg.kind == "frame" and seg.summary is None and seg.start.line not in headings_by_line:
                text = src_text[seg.start.char_offset : seg.end.char_offset].strip()
                if text:
                    headings_by_line[seg.start.line] = text.splitlines()[0].strip().rstrip("{:").strip()

    return sorted(headings_by_line.items(), key=lambda h: h[0])


def extract_markdown_headings(content: str | CodeSource) -> list[tuple[int, str]]:
    """Extract sorted [(1_indexed_start_line, title)] Markdown headings using CocoIndex Rust AST."""
    return extract_ast_headings(content, language="markdown")


def resolve_heading_for_line(
    headings: Sequence[tuple[int, str]],
    start_line: int,
    fallback_title: str = "",
) -> str:
    """Find the nearest heading on or before start_line using O(log H) binary search."""
    if not headings:
        return fallback_title
    line_keys = [h[0] for h in headings]
    idx = bisect.bisect_right(line_keys, start_line)
    if idx == 0:
        return fallback_title
    heading_title = headings[idx - 1][1]
    return heading_title if heading_title else fallback_title


def extract_chunk_title(content: str, start_line: int, rel_posix_path: str) -> str:
    """Extract closest preceding AST heading/scope or fallback to file stem."""
    lang = detect_corpus_language(rel_posix_path) or "markdown"
    headings = extract_ast_headings(content, language=lang)
    fallback_title = Path(rel_posix_path).stem
    return resolve_heading_for_line(headings, start_line, fallback_title)


@dataclass(frozen=True, slots=True)
class ProjectedRecord:
    title: str
    body: str
    start_line: int
    end_line: int


def should_keep_json_key_path(
    key: str,
    dotted_path: str,
    whitelist: Sequence[str] = ("*",),
    blacklist: Sequence[str] = (),
) -> bool:
    if any(fnmatchcase(dotted_path, p) or fnmatchcase(key, p) for p in blacklist):
        return False
    return not whitelist or any(fnmatchcase(dotted_path, p) or fnmatchcase(key, p) for p in whitelist)


def project_json_records(
    content: str | CodeSource,
    rel_posix_path: str = "record.json",
    *,
    whitelist: Sequence[str] = ("*",),
    blacklist: Sequence[str] = (),
) -> list[ProjectedRecord]:
    """Strict document objects/JSONL with source-aligned, hydratable projection."""
    src = content if isinstance(content, CodeSource) else CodeSource(content, language="json")
    text = src.text
    whitelist, blacklist = tuple(whitelist), tuple(blacklist)
    if not text.strip():
        return []
    decoder = json.JSONDecoder()
    objects = []
    if Path(rel_posix_path).suffix.lower() in (".jsonl", ".ndjson"):
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.strip():
                data = json.loads(line)
                objects.append((data, offset, offset + len(line.rstrip("\r\n"))))
            offset += len(line)
    else:
        data = json.loads(text)  # Malformed input must not become an empty corpus.
        if isinstance(data, list):
            cursor = text.index("[") + 1
            for item in data:
                while text[cursor].isspace() or text[cursor] == ",":
                    cursor += 1
                _, end = decoder.raw_decode(text, cursor)
                objects.append((item, cursor, end))
                cursor = end
        else:
            start = len(text) - len(text.lstrip())
            objects.append((data, start, len(text.rstrip())))
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    pair_matches = (
        sorted(
            _JSON_PAIR_PATTERN.match_source(src),
            key=lambda m: (m.chunks[0].start.byte_offset, -m.chunks[0].end.byte_offset),
        )
        if blacklist or whitelist not in ((), ("*",))
        else []
    )
    pair_starts = [match.chunks[0].start.char_offset for match in pair_matches]
    records = []

    def safe_lines(start_char: int, end_char: int, key_path: str) -> tuple[int, int]:
        sl = bisect.bisect_right(line_starts, start_char)
        el = bisect.bisect_right(line_starts, max(start_char, end_char - 1))
        before = text[line_starts[sl - 1] : start_char].strip()
        line_end = line_starts[el] if el < len(line_starts) else len(text)
        after = text[end_char:line_end].strip()
        if before or after not in ("", ","):
            raise ValueError(f"{rel_posix_path}:{sl}: unsafe JSON projection/source span for {key_path!r}")
        return sl, el

    for obj, start, end in objects:
        if not isinstance(obj, dict):
            raise ValueError(f"{rel_posix_path}: expected document object, got {type(obj).__name__}")
        if not obj:
            continue
        title = obj.get("title", "")
        if not isinstance(title, str):
            raise ValueError(f"{rel_posix_path}: title must be a string")
        if not should_keep_json_key_path("title", "title", whitelist, blacklist) or any(
            fnmatchcase(title, p) for p in blacklist
        ):
            title = ""
        s_line, e_line = safe_lines(start, end, "document")
        if not pair_matches:
            if blacklist or whitelist not in ((), ("*",)):
                continue
            _, body = json_document_fields(obj)
            if body or title:
                records.append(ProjectedRecord(title, body, s_line, e_line))
            continue
        kept = []
        containers = []
        nested_spans = set()
        excluded = []
        stack = []
        first_pair = bisect.bisect_left(pair_starts, start)
        last_pair = bisect.bisect_left(pair_starts, end)
        for pair_index in range(first_pair, last_pair):
            match = pair_matches[pair_index]
            chunk = match.chunks[0]
            if chunk.start.char_offset < start or chunk.end.char_offset > end:
                continue
            while stack and stack[-1][1] <= chunk.start.byte_offset:
                stack.pop()
            key = json.loads('"' + match.captures["KEY"][0].text + '"')
            value = match.captures["VAL"][0]
            dotted = ".".join([*(entry[0] for entry in stack), key])
            parsed = json.loads(value.text)
            inherited = any(entry[2] for entry in stack)
            rejected = inherited or any(fnmatchcase(dotted, p) or fnmatchcase(key, p) for p in blacklist)
            if isinstance(parsed, str) and any(fnmatchcase(parsed, p) for p in blacklist):
                rejected = True
            stack.append((key, chunk.end.byte_offset, rejected))
            if rejected:
                excluded.append(
                    (chunk.start.line, inclusive_end_line(chunk.start.line, chunk.end.line, chunk.end.column))
                )
                continue
            if (isinstance(parsed, dict) and parsed) or (
                isinstance(parsed, list) and any(isinstance(v, (dict, list)) for v in parsed)
            ):
                if len(stack) == 1:
                    containers.append((key, parsed, chunk))
                continue
            if should_keep_json_key_path(key, dotted, whitelist, blacklist):
                kept.append((dotted, parsed, chunk))
                if len(stack) > 1:
                    nested_spans.add((chunk.start.byte_offset, chunk.end.byte_offset))
            else:
                excluded.append(
                    (chunk.start.line, inclusive_end_line(chunk.start.line, chunk.end.line, chunk.end.column))
                )
        if not excluded:
            _, body = json_document_fields(obj)
            if title or body:
                records.append(ProjectedRecord(title, body, s_line, e_line))
            continue
        safe_containers = [
            (key, value, chunk)
            for key, value, chunk in containers
            if not any(a <= chunk.end.line and b >= chunk.start.line for a, b in excluded)
            and any(chunk.start.char_offset < leaf.start.char_offset < chunk.end.char_offset for _, _, leaf in kept)
        ]
        kept = safe_containers + [
            (key, value, chunk)
            for key, value, chunk in kept
            if not any(
                parent.start.char_offset < chunk.start.char_offset < parent.end.char_offset
                for _, _, parent in safe_containers
            )
        ]
        for key, value, chunk in kept:
            if key == "title":
                continue
            if (chunk.start.byte_offset, chunk.end.byte_offset) in nested_spans:
                raise ValueError(
                    f"{rel_posix_path}:{chunk.start.line}: unsafe JSON projection for {key!r}; "
                    "its enclosing object cannot be hydrated without excluded content"
                )
            sl, el = safe_lines(chunk.start.char_offset, chunk.end.char_offset, key)
            if any(a <= el and b >= sl for a, b in excluded):
                raise ValueError(
                    f"{rel_posix_path}:{sl}: unsafe JSON projection for {key!r}; excluded data shares source lines"
                )
            # A projected property has an explicit decoder contract in hydration;
            # it is never repaired by guessing braces or returning raw JSON.
            body = (
                value
                if key in ("text", "content") and isinstance(value, str)
                else json.dumps({key: value}, ensure_ascii=False)
            )
            records.append(ProjectedRecord(title, body, sl, el))
    return records


def resolve_ast_island_bounds(headings: Sequence[tuple[int, str]], line: int, total_lines: int) -> tuple[int, int]:
    index = bisect.bisect_right([h[0] for h in headings], line) - 1
    if index < 0:
        return 1, headings[0][0] - 1 if headings else max(1, total_lines)
    return headings[index][0], headings[index + 1][0] - 1 if index + 1 < len(headings) else total_lines


def chunk_source_text(
    content: str,
    rel_posix_path: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
    json_keys_whitelist: Sequence[str] = ("*",),
    json_keys_blacklist: Sequence[str] = (),
) -> tuple[tuple[ChunkPointer, str], ...]:
    text = content.replace("\r\n", "\n")
    if not text.strip():
        return ()
    language = detect_corpus_language(rel_posix_path)
    source = CodeSource(text, language=language)
    records = (
        project_json_records(source, rel_posix_path, whitelist=json_keys_whitelist, blacklist=json_keys_blacklist)
        if language == "json"
        else None
    )
    headings = [] if records is not None else extract_ast_headings(source, language)
    result = []
    units = records if records is not None else [ProjectedRecord("", text, 1, max(1, len(text.splitlines())))]
    for record in units:
        body = record.body
        full = render_document_text(record.title, body)
        if records is not None and len(full) <= chunk_size:
            pieces = [(full, record.start_line, record.end_line)]
        else:
            unit_source = CodeSource(body, language="markdown") if records is not None else source
            effective_size = max(min_chunk_size, chunk_size - len(record.title) - 1) if record.title else chunk_size
            chunks = _SPLITTER.split(
                unit_source,
                chunk_size=effective_size,
                chunk_overlap=min(chunk_overlap, max(0, effective_size - 1)),
                min_chunk_size=min(min_chunk_size, max(1, len(body.strip()))),
            )
            if not chunks:
                raise ValueError(f"{rel_posix_path}: splitter produced no chunks for nonempty content")
            pieces = [
                (render_document_text(record.title, c.text), record.start_line, record.end_line)
                if records is not None
                else (c.text, c.start.line, inclusive_end_line(c.start.line, c.end.line, c.end.column))
                for c in chunks
            ]
        for body, sl, el in pieces:
            title = (
                record.title
                if records is not None
                else resolve_heading_for_line(headings, sl, Path(rel_posix_path).stem)
            )
            ast_s, ast_e = (
                (record.start_line, record.end_line)
                if records is not None
                else resolve_ast_island_bounds(headings, sl, len(text.splitlines()))
            )
            ptr = ChunkPointer(
                compute_chunk_id(rel_posix_path, sl, el, body), rel_posix_path, sl, el, title, ast_s, ast_e
            )
            result.append((ptr, body))
    return tuple(dict((p.chunk_id, (p, t)) for p, t in result).values())


def chunk_json_text(
    content: str,
    rel_posix_path: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
    whitelist: Sequence[str] = ("*",),
    blacklist: Sequence[str] = (),
) -> tuple[tuple[ChunkPointer, str], ...]:
    return chunk_source_text(
        content,
        rel_posix_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        json_keys_whitelist=whitelist,
        json_keys_blacklist=blacklist,
    )


def chunk_markdown_text(
    content: str, rel_posix_path: str, *, chunk_size: int = 1200, chunk_overlap: int = 100, min_chunk_size: int = 50
) -> tuple[tuple[ChunkPointer, str], ...]:
    return chunk_source_text(
        content, rel_posix_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size
    )


@coco.fn(
    memo=True,
    version=3,
    deps={
        "parser_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract_source": hashlib.sha256(Path(inspect.getfile(json_document_fields)).read_bytes()).hexdigest(),
        "cocoindex_version": version("cocoindex"),
    },
)
async def process_corpus_file(
    file: FileLike,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    json_keys_whitelist: tuple[str, ...],
    json_keys_blacklist: tuple[str, ...],
) -> tuple[tuple[ChunkPointer, str], ...]:
    if file.file_path.path.is_absolute():
        raise ValueError("CocoIndex source paths must be relative")
    chunks = chunk_source_text(
        await file.read_text(),
        file.file_path.path.as_posix(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        json_keys_whitelist=json_keys_whitelist,
        json_keys_blacklist=json_keys_blacklist,
    )
    for ptr, text in chunks:
        _declare_chunk(ptr.chunk_id, (ptr, text))
    return chunks


@coco.fn
async def mount_corpus_file(
    item: tuple[coco.StableKey, FileLike],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    json_keys_whitelist: tuple[str, ...],
    json_keys_blacklist: tuple[str, ...],
) -> tuple[tuple[ChunkPointer, str], ...]:
    key, file = item
    return await coco.use_mount(
        coco.component_subpath("file", key),
        process_corpus_file,
        file,
        chunk_size,
        chunk_overlap,
        min_chunk_size,
        json_keys_whitelist,
        json_keys_blacklist,
    )


@coco.fn
async def corpus_app_main(
    included_patterns: tuple[str, ...],
    excluded_patterns: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    json_keys_whitelist: tuple[str, ...],
    json_keys_blacklist: tuple[str, ...],
) -> tuple[tuple[ChunkPointer, str], ...]:
    files = localfs.walk_dir(
        _CORPUS_ROOT_KEY,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=list(included_patterns), excluded_patterns=list(excluded_patterns)
        ),
    )
    # Materialize enumeration before target reconciliation: an unreadable/partial
    # directory scan must never be mistaken for removal of unvisited sources.
    items = [item async for item in files.items()]
    results = await coco.map(
        mount_corpus_file,
        items,
        chunk_size,
        chunk_overlap,
        min_chunk_size,
        json_keys_whitelist,
        json_keys_blacklist,
    )
    return tuple(chunk for chunks in results for chunk in chunks)


class CorpusIngestor:
    """Persistent native CocoIndex targets plus complete cached component results."""

    def __init__(
        self,
        corpus_root: Path | str,
        state_dir: Path | str,
        *,
        included_patterns: Sequence[str] = DEFAULT_INCLUDED_PATTERNS,
        excluded_patterns: Sequence[str] = DEFAULT_EXCLUDED_PATTERNS,
        json_keys_whitelist: Sequence[str] = ("*",),
        json_keys_blacklist: Sequence[str] = (),
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        on_delta: Callable[[SyncDelta], None] | None = None,
    ) -> None:
        self.corpus_root = Path(corpus_root).resolve()
        if not self.corpus_root.is_dir():
            raise ValueError(f"Corpus root does not exist: {self.corpus_root}")
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._on_delta = on_delta
        self._actions_list = []
        self._compiler_errors = []
        self._snapshot = None
        cp = coco.ContextProvider()
        cp.provide(_SINK_KEY, self._actions_list)
        cp.provide(_CORPUS_ROOT_KEY, self.corpus_root)
        cp.provide(_COMPILER_KEY, on_delta)
        cp.provide(_COMPILER_ERROR_KEY, self._compiler_errors)
        self._env = coco.Environment(settings=coco.Settings.from_env(self.state_dir / "coco_lmdb"), context_provider=cp)
        self._app = coco.App(
            coco.AppConfig(name="yaw_corpus", environment=self._env, max_inflight_components=4),
            corpus_app_main,
            included_patterns=tuple(included_patterns),
            excluded_patterns=tuple(excluded_patterns),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            json_keys_whitelist=tuple(json_keys_whitelist),
            json_keys_blacklist=tuple(json_keys_blacklist),
        )

    @property
    def on_delta(self) -> Callable[[SyncDelta], None] | None:
        return self._on_delta

    def poll(self) -> SyncDelta:
        self._actions_list.clear()
        self._compiler_errors.clear()
        self._snapshot = None
        result = self._app.update_blocking()
        if self._compiler_errors:
            raise self._compiler_errors[0]
        if not isinstance(result, tuple):
            raise TypeError("CocoIndex corpus app must return a complete chunk tuple")
        self._snapshot = result
        return SyncDelta(
            tuple(v for _, v in self._actions_list if v is not None),
            tuple(k for k, v in self._actions_list if v is None),
        )

    def snapshot(self) -> tuple[tuple[ChunkPointer, str], ...]:
        if self._snapshot is None:
            raise RuntimeError("A successful poll is required before requesting a snapshot")
        return self._snapshot

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> CorpusIngestor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def sync_corpus_once(corpus_root: Path | str, state_dir: Path | str, **kwargs: Any) -> SyncDelta:
    with CorpusIngestor(corpus_root, state_dir, **kwargs) as ingestor:
        return ingestor.poll()
