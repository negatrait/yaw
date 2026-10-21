"""Textbook CocoIndex v1 incremental line-pointer and native Tree-sitter AST ingestion pipeline."""

import bisect
import json
import re
from collections.abc import Callable, Collection, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex._internal import core as _coco_core
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import localfs
from cocoindex.ops.code import CodePattern, CodeSource
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.core import (
    ChunkPointer,
    SyncDelta,
    compute_chunk_id,
    inclusive_end_line,
)

_DEFAULT_INCLUDED_PATTERNS: tuple[str, ...] = (
    "*.md",
    "*.mdx",
    "*.txt",
    "*.rst",
    "*.adoc",
    "*.org",
    "*.py",
    "*.pyi",
    "*.rs",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.mjs",
    "*.cjs",
    "*.go",
    "*.java",
    "*.kt",
    "*.kts",
    "*.c",
    "*.h",
    "*.cpp",
    "*.hpp",
    "*.cc",
    "*.cs",
    "*.rb",
    "*.php",
    "*.swift",
    "*.scala",
    "*.lua",
    "*.r",
    "*.R",
    "*.sh",
    "*.bash",
    "*.zsh",
    "*.sql",
    "*.graphql",
    "*.proto",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.json",
    "*.jsonl",
    "*.xml",
    "*.html",
    "*.css",
    "*.scss",
    "*.tex",
    "*.bib",
    "*.pas",
    "*.dpr",
    "*.f90",
    "*.sol",
    "*.dtd",
)
_DEFAULT_EXCLUDED_PATTERNS: tuple[str, ...] = (".*", "*/.*", "__pycache__/*")

_SPLITTER = RecursiveSplitter()
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:\s+(.*?))?(?:\s+#+\s*)?$")
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)\s*$")
_PLAIN_TITLE_LINE_RE = re.compile(r"^\s*(?:title\s*:\s*(.+)|#+\s*(.+))\s*$", re.IGNORECASE)
_LVEVAL_SPLIT_RE = re.compile(r"\n*###\s*Passage\s*\d+\s*[:\n]\s*")

_HTML_HEADING_PATTERNS: tuple[CodePattern, ...] = (
    CodePattern(r"<title>\TEXT</title>", language="html", min_len=1),
    *(CodePattern(rf"<h{lvl}>\TEXT</h{lvl}>", language="html", min_len=1) for lvl in range(1, 7)),
)
_TOML_SECTION_PATTERNS: tuple[CodePattern, ...] = (
    CodePattern(r"[\SECTION]", language="toml", min_len=1),
    CodePattern(r"[[\SECTION]]", language="toml", min_len=1),
)
_YAML_PAIR_PATTERN = CodePattern(r"\KEY: \VAL", language="yaml", min_len=1)
_JSON_OBJ_PATTERN = CodePattern(r"{\(PAIRS*\)}", language="json", min_len=1)
_JSON_PAIR_PATTERN = CodePattern(r"\Q\KEY\Q: \VAL", language="json", min_len=1)
_JSON_TUPLE_PASSAGE_PATTERN = CodePattern(r"[\Q\TITLE\Q, [\(SENTS*\)]]", language="json", min_len=1)

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

    def _apply_actions(
        self, context_provider: coco.ContextProvider, actions: Sequence[_ChunkAction]
    ) -> None:
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

    if isinstance(source, CodeSource) and source.language == "markdown" and raw_text.endswith("\n"):
        md_src = source
    else:
        padded = raw_text if raw_text.endswith("\n") else raw_text + "\n"
        md_src = CodeSource(padded, language="markdown")

    src_text = md_src.text
    headings_by_line: dict[int, str] = {}
    pos = 0
    in_fence_marker: str | None = None
    prev_non_fenced_line: tuple[int, str] | None = None

    for line_idx, line in enumerate(raw_text.splitlines(keepends=True), start=1):
        b_len = len(line.encode("utf-8"))
        end_pos = pos + b_len
        stripped_line = line.rstrip("\r\n")
        lstripped = stripped_line.lstrip()

        fence_prefix: str | None = None
        if lstripped.startswith("```"):
            fence_prefix = "```"
        elif lstripped.startswith("~~~"):
            fence_prefix = "~~~"

        is_fence_delimiter = False
        if fence_prefix is not None:
            is_fence_delimiter = True
            if in_fence_marker is None:
                in_fence_marker = fence_prefix
            elif lstripped.startswith(in_fence_marker):
                in_fence_marker = None

        view = _coco_core.render_ranges(md_src._src, [(pos, end_pos)])
        in_ast_code_fence = in_fence_marker is not None or is_fence_delimiter
        for seg in view.segments:
            if seg.kind == "frame" and seg.summary is None:
                seg_text = src_text[seg.start_char_offset : seg.end_char_offset]
                seg_lines = seg_text.splitlines()
                first_line = seg_lines[0].strip() if seg_lines else ""
                if first_line.startswith(("```", "~~~")):
                    in_ast_code_fence = True
                elif seg.start_line not in headings_by_line:
                    m_atx = _ATX_HEADING_RE.match(first_line)
                    if m_atx is not None:
                        headings_by_line[seg.start_line] = (m_atx.group(1) or "").strip()
                    elif (
                        len(seg_lines) >= 2 and _SETEXT_UNDERLINE_RE.match(seg_lines[1]) is not None
                    ):
                        headings_by_line[seg.start_line] = first_line

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

        pos = end_pos

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
        try:
            code_src = CodeSource(clean_text, language=lang)
        except ValueError:
            return []

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
                    headings_by_line[line_no] = _unescape_json_str(raw_v[1:-1])
        return sorted(headings_by_line.items(), key=lambda h: h[0])

    # General Tree-sitter programming languages: extract enclosing AST scope frames
    src_text = code_src.text
    pos = 0
    for line in clean_text.splitlines(keepends=True):
        b_len = len(line.encode("utf-8"))
        content_len = len(line.rstrip("\r\n").encode("utf-8"))
        if content_len > 0:
            view = _coco_core.render_ranges(code_src._src, [(pos, pos + content_len)])
            for seg in view.segments:
                if (
                    seg.kind == "frame"
                    and seg.summary is None
                    and seg.start_line not in headings_by_line
                ):
                    seg_text = src_text[seg.start_char_offset : seg.end_char_offset].strip()
                    first_sig = seg_text.splitlines()[0].strip() if seg_text else ""
                    first_sig = first_sig.rstrip("{:").strip()
                    if first_sig:
                        headings_by_line[seg.start_line] = first_sig
        pos += b_len

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


def _unescape_json_str(raw: str) -> str:
    try:
        val = json.loads(f'"{raw}"')
        return str(val)
    except json.JSONDecodeError:
        return raw


def should_keep_json_key_path(
    key: str,
    dotted_path: str,
    whitelist: Sequence[str] = ("*",),
    blacklist: Sequence[str] = (),
) -> bool:
    """Evaluate formal JSON key projection rule pi_{W, B} with blacklist dominance."""
    if any(fnmatchcase(dotted_path, pat) or fnmatchcase(key, pat) for pat in blacklist):
        return False
    if not whitelist:
        return True
    return any(fnmatchcase(dotted_path, pat) or fnmatchcase(key, pat) for pat in whitelist)


def project_json_records(
    content: str | CodeSource,
    rel_posix_path: str = "record.json",
    *,
    whitelist: Sequence[str] = ("*",),
    blacklist: Sequence[str] = (),
) -> list[tuple[str, str, int, int]]:
    """Project JSON/JSONL records via CocoIndex Tree-sitter AST key filtering into (title, body, s_line, e_line) tuples."""
    if isinstance(content, CodeSource):
        clean_text = content.text
        src = content
    else:
        clean_text = content.replace("\r\n", "\n")
        if not clean_text.strip():
            return []
        non_empty = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]
        if (
            len(non_empty) > 1
            and not non_empty[0].startswith("[")
            and all(ln.startswith("{") and ln.endswith(("}", "},")) for ln in non_empty[:2])
        ):
            wrapped = "[\n" + ",\n".join(ln.rstrip(",") for ln in non_empty) + "\n]"
            src = CodeSource(wrapped, language="json")
        else:
            src = CodeSource(clean_text, language="json")

    obj_spans = sorted(
        (
            (m.chunks[0].start.byte_offset, m.chunks[0].end.byte_offset)
            for m in _JSON_OBJ_PATTERN.match_source(src)
        ),
        key=lambda x: (x[0], -x[1]),
    )
    if not obj_spans:
        return []

    top_objs: list[tuple[int, int]] = []
    for o_s, o_e in obj_spans:
        if not top_objs or o_s >= top_objs[-1][1]:
            top_objs.append((o_s, o_e))

    all_pairs = sorted(
        _JSON_PAIR_PATTERN.match_source(src),
        key=lambda m: (m.chunks[0].start.byte_offset, -m.chunks[0].end.byte_offset),
    )
    all_tuples = sorted(
        _JSON_TUPLE_PASSAGE_PATTERN.match_source(src),
        key=lambda m: m.chunks[0].start.byte_offset,
    )

    records: list[tuple[str, str, int, int]] = []
    clean_bytes = clean_text.encode("utf-8")

    def line_of_byte(b_off: int) -> int:
        return clean_bytes[:b_off].decode("utf-8", errors="replace").count("\n") + 1

    pair_idx = 0
    num_pairs = len(all_pairs)
    obj_idx = 0
    num_objs = len(obj_spans)
    tuple_idx = 0
    num_tuples = len(all_tuples)
    is_hotpot = "hotpot" in rel_posix_path.lower()

    for o_s, o_e in top_objs:
        while pair_idx < num_pairs and all_pairs[pair_idx].chunks[0].start.byte_offset < o_s:
            pair_idx += 1
        obj_pairs = []
        p_cursor = pair_idx
        while p_cursor < num_pairs and all_pairs[p_cursor].chunks[0].start.byte_offset < o_e:
            if all_pairs[p_cursor].chunks[0].end.byte_offset <= o_e:
                obj_pairs.append(all_pairs[p_cursor])
            p_cursor += 1
        pair_idx = p_cursor

        while obj_idx < num_objs and obj_spans[obj_idx][0] < o_s:
            obj_idx += 1
        nested_objs: list[tuple[int, int]] = []
        o_cursor = obj_idx
        while o_cursor < num_objs and obj_spans[o_cursor][0] < o_e:
            sub_s, sub_e = obj_spans[o_cursor]
            if sub_e <= o_e and (sub_s > o_s or sub_e < o_e):
                nested_objs.append((sub_s, sub_e))
            o_cursor += 1

        while tuple_idx < num_tuples and all_tuples[tuple_idx].chunks[0].start.byte_offset < o_s:
            tuple_idx += 1
        obj_tuples = []
        t_cursor = tuple_idx
        while t_cursor < num_tuples and all_tuples[t_cursor].chunks[0].start.byte_offset < o_e:
            if all_tuples[t_cursor].chunks[0].end.byte_offset <= o_e:
                obj_tuples.append(all_tuples[t_cursor])
            t_cursor += 1
        tuple_idx = t_cursor

        stack: list[tuple[str, int]] = []
        blacklisted_spans: list[tuple[int, int]] = []
        kept_pairs: list[tuple[str, str, int, int, int, int, str]] = []
        all_pair_spans: list[tuple[int, int]] = []

        for m in obj_pairs:
            p_s = m.chunks[0].start.byte_offset
            p_e = m.chunks[0].end.byte_offset
            all_pair_spans.append((p_s, p_e))
            while stack and stack[-1][1] <= p_s:
                stack.pop()
            if any(b_s <= p_s and p_e <= b_e for b_s, b_e in blacklisted_spans):
                continue

            key = _unescape_json_str(m.captures["KEY"][0].text)
            dotted = ".".join([*(k for k, _ in stack), key])
            val_chunk = m.captures["VAL"][0]
            v_s = val_chunk.start.byte_offset
            v_e = val_chunk.end.byte_offset
            v_raw = val_chunk.text.strip()

            is_bl = any(fnmatchcase(dotted, pat) or fnmatchcase(key, pat) for pat in blacklist)
            if (
                not is_bl
                and blacklist
                and len(v_raw) >= 2
                and v_raw.startswith('"')
                and v_raw.endswith('"')
            ):
                v_unesc = _unescape_json_str(v_raw[1:-1])
                if any(fnmatchcase(v_unesc, pat) for pat in blacklist):
                    is_bl = True

            if is_bl:
                blacklisted_spans.append((p_s, p_e))
                continue

            stack.append((key, p_e))
            is_wl = not whitelist or any(
                fnmatchcase(dotted, pat) or fnmatchcase(key, pat) or pat.startswith(f"{dotted}.")
                for pat in whitelist
            )
            if is_wl:
                kept_pairs.append((key, dotted, p_s, p_e, v_s, v_e, v_raw))
            else:
                blacklisted_spans.append((p_s, p_e))

        if not kept_pairs:
            continue

        extracted: list[tuple[str, str, int, int]] = []

        # 1. Nested passage objects (e.g., paragraphs, contexts, gold_docs)
        for sub_s, sub_e in nested_objs:
            if any(b_s <= sub_s and sub_e <= b_e for b_s, b_e in blacklisted_spans):
                continue
            direct_str: dict[str, str] = {}
            parent_dotted = ""
            for k, d, p_s, p_e, _vs, _ve, v_raw in kept_pairs:
                if sub_s < p_s and p_e < sub_e:
                    if any(
                        sub_s < in_s and in_e < sub_e and in_s < p_s and p_e < in_e
                        for in_s, in_e in nested_objs
                    ):
                        continue
                    if len(v_raw) >= 2 and v_raw.startswith('"') and v_raw.endswith('"'):
                        direct_str[k] = _unescape_json_str(v_raw[1:-1])
                        if "." in d:
                            parent_dotted = d.rsplit(".", 1)[0]
            if parent_dotted == "document.summary":
                continue
            sub_body = (
                direct_str.get("text")
                or direct_str.get("paragraph_text")
                or direct_str.get("contents")
            )
            if sub_body is not None:
                sub_title = direct_str.get("title", "")
                s_line = line_of_byte(sub_s)
                e_line = line_of_byte(sub_e)
                extracted.append((sub_title, sub_body, s_line, e_line))

        # 2. 2-Tuple passages [title, [sents...]] under surviving "context"
        if not extracted and obj_tuples:
            ctx_spans = [
                (v_s, v_e) for k, _d, _ps, _pe, v_s, v_e, _vr in kept_pairs if k == "context"
            ]
            sep = "" if is_hotpot else " "
            seen_tuple_spans: set[tuple[int, int]] = set()
            for c_vs, c_ve in ctx_spans:
                for tm in obj_tuples:
                    ts = tm.chunks[0].start.byte_offset
                    te = tm.chunks[0].end.byte_offset
                    if (ts, te) in seen_tuple_spans:
                        continue
                    if (
                        c_vs <= ts
                        and te <= c_ve
                        and not any(b_s <= ts and te <= b_e for b_s, b_e in blacklisted_spans)
                    ):
                        seen_tuple_spans.add((ts, te))
                        try:
                            parsed_tuple = json.loads(tm.chunks[0].text)
                        except json.JSONDecodeError:
                            continue
                        if (
                            isinstance(parsed_tuple, list)
                            and len(parsed_tuple) >= 2
                            and isinstance(parsed_tuple[0], str)
                            and isinstance(parsed_tuple[1], list)
                        ):
                            t_title = parsed_tuple[0]
                            sents = [str(s) for s in parsed_tuple[1] if isinstance(s, str)]
                            t_body = sep.join(sents)
                            if t_title or t_body:
                                s_line = tm.chunks[0].start.line
                                e_line = inclusive_end_line(
                                    tm.chunks[0].start.line,
                                    tm.chunks[0].end.line,
                                    tm.chunks[0].end.column,
                                )
                                extracted.append((t_title, t_body, s_line, e_line))

        # 3. String "context" (lveval ### Passage N) or nested "document" (narrativeqa)
        dotted_str_map: dict[str, str] = {}
        for _k, d, _ps, _pe, _vs, _ve, v_raw in kept_pairs:
            if len(v_raw) >= 2 and v_raw.startswith('"') and v_raw.endswith('"'):
                dotted_str_map[d] = _unescape_json_str(v_raw[1:-1])

        if not extracted:
            s_line = line_of_byte(o_s)
            e_line = line_of_byte(o_e)
            if "context" in dotted_str_map and "### Passage" in dotted_str_map["context"]:
                parts = [
                    p.strip()
                    for p in _LVEVAL_SPLIT_RE.split(dotted_str_map["context"])
                    if p.strip()
                ]
                for p in parts:
                    extracted.append(("", p, s_line, e_line))
            elif "document.summary.text" in dotted_str_map or "document.text" in dotted_str_map:
                doc_title = (
                    dotted_str_map.get("document.summary.title")
                    or dotted_str_map.get("document.title")
                    or ""
                )
                doc_body = (
                    dotted_str_map.get("document.summary.text")
                    or dotted_str_map.get("document.text")
                    or ""
                )
                if doc_body:
                    extracted.append((doc_title, doc_body, s_line, e_line))

        # 4. Direct top-level passage or general/corporate JSON record fallback
        if not extracted:
            leaf_kept = [
                (k, d, p_s, p_e, v_s, v_e, v_raw)
                for k, d, p_s, p_e, v_s, v_e, v_raw in kept_pairs
                if not any(v_s <= p2_s and p2_e <= v_e for p2_s, p2_e in all_pair_spans)
            ]
            if leaf_kept:
                body_kept = [p for p in leaf_kept if p[1] != "title"] or leaf_kept
                s_line = line_of_byte(min(p[2] for p in body_kept))
                e_line = line_of_byte(max(p[3] for p in body_kept))
                leaf_keys = {d for _k, d, *_ in leaf_kept}
                passage_keys = {"title", "text", "paragraph_text", "contents", "body", "content"}
                if leaf_keys <= passage_keys:
                    rec_title = dotted_str_map.get("title", "")
                    rec_body = (
                        dotted_str_map.get("text")
                        or dotted_str_map.get("paragraph_text")
                        or dotted_str_map.get("contents")
                        or dotted_str_map.get("body")
                        or dotted_str_map.get("content")
                        or ""
                    )
                    if rec_body or rec_title:
                        extracted.append((rec_title, rec_body, s_line, e_line))
                else:
                    rec_title = (
                        dotted_str_map.get("title")
                        or dotted_str_map.get("name")
                        or dotted_str_map.get("service")
                        or Path(rel_posix_path).stem
                    )
                    surviving_dict: dict[str, Any] = {}
                    for _k, d, _ps, _pe, _vs, _ve, v_raw in leaf_kept:
                        if d == "title" and rec_title == dotted_str_map.get("title"):
                            continue
                        try:
                            surviving_dict[d] = json.loads(v_raw)
                        except json.JSONDecodeError:
                            surviving_dict[d] = v_raw
                    if surviving_dict:
                        rec_body = json.dumps(surviving_dict, ensure_ascii=False)
                        extracted.append((rec_title, rec_body, s_line, e_line))
                    elif rec_title:
                        extracted.append((rec_title, rec_title, s_line, e_line))

        records.extend(extracted)

    return records


def chunk_json_text(
    content: str,
    rel_posix_path: str,
    *,
    corpus_root: Path | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
    max_island_chars: int = 3600,
    whitelist: Sequence[str] = ("*",),
    blacklist: Sequence[str] = (),
) -> tuple[tuple[ChunkPointer, str], ...]:
    """Project JSON/JSONL records via AST filtering and map line coordinates directly to the source file."""
    records = project_json_records(
        content,
        rel_posix_path,
        whitelist=whitelist,
        blacklist=blacklist,
    )
    unique_records = list(dict.fromkeys(records))
    if not unique_records:
        return ()

    results: list[tuple[ChunkPointer, str]] = []
    seen_cids: set[str] = set()

    for item in unique_records:
        if len(item) == 4:
            title_i, body_i, rec_start, rec_end = item
        else:
            title_i, body_i = item[0], item[1]
            rec_start, rec_end = 1, max(1, content.count("\n") + 1)

        clean_title = title_i.replace("\r\n", "\n").replace("\n", " ").strip()
        clean_body = body_i.replace("\r\n", "\n").rstrip("\n")
        passage_text = (
            f"{clean_title}\n{clean_body}"
            if clean_title
            and not (clean_body.startswith(f"{clean_title}\n") or clean_body == clean_title)
            else clean_body
        )

        if len(passage_text) <= chunk_size:
            cid = compute_chunk_id(rel_posix_path, rec_start, rec_end, passage_text)
            if cid not in seen_cids:
                seen_cids.add(cid)
                ptr = ChunkPointer(
                    chunk_id=cid,
                    rel_posix_path=rel_posix_path,
                    start_line=rec_start,
                    end_line=rec_end,
                    title=clean_title,
                    ast_island_start=rec_start,
                    ast_island_end=rec_end,
                )
                results.append((ptr, passage_text))
            continue

        rec_src = CodeSource(passage_text, language="markdown")
        sub_chunks = _SPLITTER.split(
            rec_src,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min(min_chunk_size, max(1, len(passage_text.strip()))),
        )
        if not sub_chunks:
            cid = compute_chunk_id(rel_posix_path, rec_start, rec_end, passage_text)
            if cid not in seen_cids:
                seen_cids.add(cid)
                ptr = ChunkPointer(
                    chunk_id=cid,
                    rel_posix_path=rel_posix_path,
                    start_line=rec_start,
                    end_line=rec_end,
                    title=clean_title,
                    ast_island_start=rec_start,
                    ast_island_end=rec_end,
                )
                results.append((ptr, passage_text))
            continue

        for sc in sub_chunks:
            sc_text = sc.text
            if (
                clean_title
                and not sc_text.startswith(f"{clean_title}\n")
                and sc_text.strip() != clean_title
            ):
                sc_text = f"{clean_title}\n{sc_text.lstrip()}"
            sub_s = max(rec_start, rec_start + sc.start.line - 1)
            sub_e = min(
                rec_end,
                rec_start + inclusive_end_line(sc.start.line, sc.end.line, sc.end.column) - 1,
            )
            if sub_e < sub_s:
                sub_e = sub_s
            cid = compute_chunk_id(rel_posix_path, sub_s, sub_e, sc_text)
            if cid not in seen_cids:
                seen_cids.add(cid)
                ptr = ChunkPointer(
                    chunk_id=cid,
                    rel_posix_path=rel_posix_path,
                    start_line=sub_s,
                    end_line=sub_e,
                    title=clean_title,
                    ast_island_start=rec_start,
                    ast_island_end=rec_end,
                )
                results.append((ptr, sc_text))

    return tuple(results)


def resolve_ast_island_bounds(
    headings: Sequence[tuple[int, str]],
    line: int,
    total_lines: int,
) -> tuple[int, int]:
    """Resolve enclosing AST section line bounds [ast_start, ast_end] for a line."""
    if not headings:
        return 1, max(1, total_lines)
    ast_start = 1
    ast_end = total_lines
    for i, (h_line, _) in enumerate(headings):
        if h_line <= line:
            ast_start = h_line
            if i + 1 < len(headings):
                ast_end = max(ast_start, headings[i + 1][0] - 1)
            else:
                ast_end = total_lines
        else:
            break
    return ast_start, ast_end


def chunk_source_text(
    content: str,
    rel_posix_path: str,
    *,
    corpus_root: Path | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
    max_island_chars: int = 3600,
    json_keys_whitelist: Sequence[str] = ("*",),
    json_keys_blacklist: Sequence[str] = (),
) -> tuple[tuple[ChunkPointer, str], ...]:
    """Single-parse CocoIndex Tree-sitter AST splitting and scope/heading resolution across all languages."""
    clean_text = content.replace("\r\n", "\n")
    if not clean_text.strip():
        return ()

    lang = detect_corpus_language(rel_posix_path)
    if lang == "json":
        return chunk_json_text(
            clean_text,
            rel_posix_path,
            corpus_root=corpus_root,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_island_chars=max_island_chars,
            whitelist=json_keys_whitelist,
            blacklist=json_keys_blacklist,
        )

    fallback_title = Path(rel_posix_path).stem
    code_src: CodeSource | None = None
    if lang is not None:
        try:
            code_src = CodeSource(clean_text, language=lang)
        except ValueError:
            code_src = None

    headings = extract_ast_headings(
        code_src if code_src is not None else clean_text,
        language=lang,
    )

    effective_min = min(min_chunk_size, len(clean_text.strip()))
    if code_src is not None:
        chunks = _SPLITTER.split(
            code_src,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=effective_min,
        )
    else:
        chunks = _SPLITTER.split(
            clean_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=effective_min,
            language=lang,
        )

    total_file_lines = max(1, clean_text.count("\n") + 1)
    if not chunks:
        lines = clean_text.splitlines()
        end_line = max(1, len(lines))
        title = resolve_heading_for_line(headings, 1, fallback_title)
        ast_s, ast_e = resolve_ast_island_bounds(headings, 1, total_file_lines)
        cid = compute_chunk_id(rel_posix_path, 1, end_line, clean_text)
        ptr = ChunkPointer(
            chunk_id=cid,
            rel_posix_path=rel_posix_path,
            start_line=1,
            end_line=end_line,
            title=title,
            ast_island_start=ast_s,
            ast_island_end=ast_e,
        )
        return ((ptr, clean_text),)

    results: list[tuple[ChunkPointer, str]] = []
    seen_cids: set[str] = set()
    for c in chunks:
        start_line = c.start.line
        end_line = inclusive_end_line(c.start.line, c.end.line, c.end.column)
        title = resolve_heading_for_line(headings, start_line, fallback_title)
        ast_s, ast_e = resolve_ast_island_bounds(headings, start_line, total_file_lines)
        cid = compute_chunk_id(rel_posix_path, start_line, end_line, c.text)
        if cid not in seen_cids:
            seen_cids.add(cid)
            ptr = ChunkPointer(
                chunk_id=cid,
                rel_posix_path=rel_posix_path,
                start_line=start_line,
                end_line=end_line,
                title=title,
                ast_island_start=ast_s,
                ast_island_end=ast_e,
            )
            results.append((ptr, c.text))

    return tuple(results)


def chunk_markdown_text(
    content: str,
    rel_posix_path: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> tuple[tuple[ChunkPointer, str], ...]:
    """Pure syntax-aware text splitting returning 1-indexed ChunkPointers and passage texts."""
    return chunk_source_text(
        content,
        rel_posix_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )


@coco.fn(memo=True)
async def process_corpus_file(
    file: FileLike,
    corpus_root_str: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    json_keys_whitelist: tuple[str, ...],
    json_keys_blacklist: tuple[str, ...],
) -> None:
    """Memoized per-file processing component declaring chunks as target state."""
    if file.file_path.path.is_absolute():
        raise ValueError(
            f"Expected relative FilePath from localfs.walk_dir, got absolute path: {file.file_path.path}"
        )
    rel_posix_path = file.file_path.path.as_posix()
    text = await file.read_text()

    chunks = chunk_source_text(
        text,
        rel_posix_path,
        corpus_root=Path(corpus_root_str),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        json_keys_whitelist=json_keys_whitelist,
        json_keys_blacklist=json_keys_blacklist,
    )
    for ptr, chunk_text in chunks:
        _declare_chunk(ptr.chunk_id, (ptr, chunk_text))


@coco.fn
async def corpus_app_main(
    corpus_root_str: str,
    included_patterns: tuple[str, ...],
    excluded_patterns: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    json_keys_whitelist: tuple[str, ...],
    json_keys_blacklist: tuple[str, ...],
) -> None:
    """Top-level CocoIndex application function walking corpus_root and mounting files."""
    files = localfs.walk_dir(
        _CORPUS_ROOT_KEY,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=list(included_patterns),
            excluded_patterns=list(excluded_patterns),
        ),
    )
    await coco.mount_each(
        process_corpus_file,
        files.items(),
        corpus_root_str,
        chunk_size,
        chunk_overlap,
        min_chunk_size,
        json_keys_whitelist,
        json_keys_blacklist,
    )


class CorpusIngestor:
    """Incremental line-pointer corpus ingestor backed by CocoIndex v1."""

    def __init__(
        self,
        corpus_root: Path | str,
        state_dir: Path | str,
        *,
        included_patterns: Sequence[str] = _DEFAULT_INCLUDED_PATTERNS,
        excluded_patterns: Sequence[str] = _DEFAULT_EXCLUDED_PATTERNS,
        json_keys_whitelist: Sequence[str] = ("*",),
        json_keys_blacklist: Sequence[str] = (),
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        on_delta: Callable[[SyncDelta], None] | None = None,
    ) -> None:
        self.corpus_root = Path(corpus_root).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.corpus_root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.included_patterns = tuple(included_patterns)
        self.excluded_patterns = tuple(excluded_patterns)
        self.json_keys_whitelist = tuple(json_keys_whitelist)
        self.json_keys_blacklist = tuple(json_keys_blacklist)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._on_delta = on_delta

        self._actions_list: list[_ChunkAction] = []
        self._compiler_errors: list[Exception] = []
        self._cp = coco.ContextProvider()
        self._cp.provide(_SINK_KEY, self._actions_list)
        self._cp.provide(_CORPUS_ROOT_KEY, self.corpus_root)
        self._cp.provide(_COMPILER_KEY, self._on_delta)
        self._cp.provide(_COMPILER_ERROR_KEY, self._compiler_errors)

        lmdb_path = self.state_dir / "coco_lmdb"
        self._env = coco.Environment(
            settings=coco.Settings.from_env(lmdb_path),
            context_provider=self._cp,
        )
        self._app = coco.App(
            coco.AppConfig(name="yaw_corpus", environment=self._env),
            corpus_app_main,
            corpus_root_str=self.corpus_root.as_posix(),
            included_patterns=self.included_patterns,
            excluded_patterns=self.excluded_patterns,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            json_keys_whitelist=self.json_keys_whitelist,
            json_keys_blacklist=self.json_keys_blacklist,
        )

    def set_delta_compiler(self, compiler: Callable[[SyncDelta], None] | None) -> None:
        """Bind or update the target sink delta compiler callback."""
        self._on_delta = compiler
        self._cp.provide(_COMPILER_KEY, compiler)

    def close(self) -> None:
        """Close the underlying CocoIndex environment and release LMDB resources."""
        self._env.close()

    def __enter__(self) -> CorpusIngestor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def poll(self) -> SyncDelta:
        """Run incremental CocoIndex update and return coalesced batch SyncDelta."""
        self._actions_list.clear()
        self._compiler_errors.clear()
        self._app.update_blocking()
        if self._compiler_errors:
            raise self._compiler_errors[0]

        added_chunks: list[tuple[ChunkPointer, str]] = []
        deleted_chunk_ids: list[str] = []

        for chunk_id, value in self._actions_list:
            if value is not None:
                ptr, chunk_text = value
                added_chunks.append((ptr, chunk_text))
            else:
                deleted_chunk_ids.append(chunk_id)

        return SyncDelta(
            added_chunks=tuple(added_chunks),
            deleted_chunk_ids=tuple(deleted_chunk_ids),
        )


def sync_corpus_once(
    corpus_root: Path | str,
    state_dir: Path | str,
    **kwargs: Any,
) -> SyncDelta:
    """One-shot helper to initialize a CorpusIngestor and poll for changes once."""
    with CorpusIngestor(corpus_root, state_dir, **kwargs) as ingestor:
        return ingestor.poll()
