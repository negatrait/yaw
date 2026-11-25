"""Canonical typed contracts and pure deterministic helpers for Yaw."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

type EmbedFn = Callable[[list[str]], Any]

ErrorCode = Literal[
    "BUSY",
    "CONSISTENCY_ERROR",
    "EMPTY_INDEX",
    "INDEX_NOT_READY",
    "INVALID_INPUT",
    "MODEL_ERROR",
    "NOT_READY",
    "SHUTTING_DOWN",
]


class YawRuntimeError(RuntimeError):
    """Structured runtime failure carrying a stable machine-readable error code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code: ErrorCode = code
        self.detail: str = message


_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")
_URL_USERINFO_RE = re.compile(r"(https?://)([^:/@\s]+):([^@/\s]+)@")
_API_KEY_KV_RE = re.compile(r"(?i)(api_key\s*[=:]\s*)([^\s,;\"']+)")


def redact_secrets(text: str, api_key: str = "") -> str:
    """Mask configured API keys, Bearer tokens, OpenAI keys, and URL credentials."""
    if not text:
        return text
    cleaned = text
    if api_key and len(api_key) >= 4:
        cleaned = cleaned.replace(api_key, "[REDACTED]")
    cleaned = _BEARER_RE.sub("Bearer [REDACTED]", cleaned)
    cleaned = _SK_RE.sub("[REDACTED]", cleaned)
    cleaned = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", cleaned)
    return _API_KEY_KV_RE.sub(r"\1[REDACTED]", cleaned)


def canonical_chunk_span_key(
    rel_path: str, start_line: int, end_line: int, char_start: int = 0, char_end: int = 0, source_hash: str = ""
) -> str:
    """Build the canonical provenance span key for deterministic chunk identity."""
    norm_path = rel_path.replace("\\", "/")
    base = f"{norm_path}:{int(start_line)}:{int(end_line)}"
    if char_start or char_end:
        base = f"{base}:{int(char_start)}:{int(char_end)}"
    return f"{base}:{source_hash}" if source_hash else base


def compute_chunk_int_id(
    rel_path: str, start_line: int, end_line: int, char_start: int = 0, char_end: int = 0, source_hash: str = ""
) -> int:
    """Compute positive signed 63-bit integer primary key from canonical chunk span."""
    key = canonical_chunk_span_key(rel_path, start_line, end_line, char_start, char_end, source_hash)
    raw = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
    return raw if raw != 0 else 1


def inclusive_end_line(start_line: int, end_line: int, end_column: int) -> int:
    """Convert half-open splitter end line/column into 1-indexed inclusive end line."""
    if end_column == 1 and end_line > start_line:
        return end_line - 1
    return max(start_line, end_line)


def split_lf_lines(text: str) -> list[str]:
    """Split normalized LF text strictly on `\\n` to stay 1-to-1 with byte/line numbers."""
    if not text:
        return []
    return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


DASH_RE = re.compile(r"[\u2010-\u2015\u2212]")
_LEADING_ATX_RE = re.compile(r"^#+\s*")
_CLOSING_ATX_RE = re.compile(r"(?:^|\s+)#+$")
_TITLE_TRAILING_SEP_RE = re.compile(r"(?:\s+-+|\s*[:{]+)+\s*$")
_TITLE_TAG_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s+(.+)$")


def normalize_ast_heading(raw_line: str) -> str:
    """Normalize a Markdown heading or code declaration header into a clean title."""
    first = next((ln for ln in split_lf_lines(raw_line) if ln.strip()), "")
    head = " ".join(first.split())
    if not head:
        return ""
    if head.startswith("#"):
        head = _CLOSING_ATX_RE.sub("", _LEADING_ATX_RE.sub("", head)).strip()
    head = DASH_RE.sub("-", head)
    if m := _TITLE_TAG_PREFIX_RE.match(head):
        raw_grp = m.group(1)
        if isinstance(raw_grp, str) and (stripped := raw_grp.strip()):
            head = stripped
    head = _TITLE_TRAILING_SEP_RE.sub("", head)
    return " ".join(head.split())


def advance_markdown_fence(raw_line: str, fence: tuple[str, int]) -> tuple[tuple[str, int], bool]:
    """Advance CommonMark fenced code block state for `raw_line`, returning `(next_fence, is_delimiter)`."""
    stripped = raw_line.lstrip(" ")
    if raw_line.startswith("\t") or len(raw_line) - len(stripped) > 3:
        return fence, False
    f_ch, f_len = fence
    if not f_ch and stripped.startswith(("```", "~~~")):
        ch = stripped[0]
        return (ch, len(stripped) - len(stripped.lstrip(ch))), True
    if f_ch and stripped.startswith(f_ch * f_len) and not stripped.strip(f_ch + " \t"):
        return ("", 0), True
    return fence, False


def leading_frontmatter_end(lines: Sequence[str], max_el: int | None = None) -> int:
    """Return the 0-based line index immediately after leading YAML frontmatter, or 0 if absent."""
    limit = len(lines) if max_el is None else min(len(lines), max(0, int(max_el)))
    first_idx = next((i for i in range(limit) if lines[i].strip()), None)
    if (
        first_idx is None
        or lines[first_idx].strip() != "---"
        or first_idx + 1 >= limit
        or not lines[first_idx + 1].strip()
    ):
        return 0
    for j in range(first_idx + 1, limit):
        s = lines[j].strip()
        if s in ("---", "..."):
            return j + 1
        if s and not (lines[j][0].isspace() or s.startswith(("#", "-")) or ":" in s):
            return 0
    return 0


def _is_setext_underline(raw_line: str, ch: str) -> bool:
    stripped = raw_line.lstrip(" ")
    if raw_line.startswith("\t") or len(raw_line) - len(stripped) > 3:
        return False
    u = stripped.strip()
    return bool(u) and set(u) == {ch}


def _is_paragraph_interrupter(raw_line: str) -> bool:
    stripped = raw_line.lstrip(" ")
    if raw_line.startswith("\t") or len(raw_line) - len(stripped) > 3:
        return False
    if stripped.startswith(("```", "~~~", ">")):
        return True
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if 1 <= hashes <= 6 and (len(stripped) == hashes or stripped[hashes] in (" ", "\t")):
        return True
    u = stripped.strip()
    if u in ("-", "*", "+") or (len(stripped) >= 2 and stripped[0] in ("-", "*", "+") and stripped[1] in (" ", "\t")):
        return True
    no_ws = u.replace(" ", "").replace("\t", "")
    return len(no_ws) >= 3 and len(set(no_ws)) == 1 and no_ws[0] in ("-", "*", "_")


def markdown_heading_level(lines: Sequence[str], frame_line: int) -> int | None:
    """Return CommonMark ATX (1..6) or Setext (1..2) heading level for 1-indexed `frame_line`, or None."""
    if frame_line < 1 or frame_line > len(lines):
        return None
    raw = lines[frame_line - 1]
    stripped = raw.lstrip(" ")
    if not raw.strip() or raw.startswith("\t") or len(raw) - len(stripped) > 3:
        return None
    if stripped.startswith(("```", "~~~", ">")):
        return None
    if frame_line <= leading_frontmatter_end(lines):
        return None
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if 1 <= hashes <= 6 and (len(stripped) == hashes or stripped[hashes] in (" ", "\t")):
        return hashes
    if stripped.startswith(("-", "*", "+", "#")) or frame_line >= len(lines):
        return None
    no_ws0 = stripped.strip().replace(" ", "").replace("\t", "")
    if len(no_ws0) >= 3 and len(set(no_ws0)) == 1 and no_ws0[0] == "_":
        return None
    for idx in range(frame_line, min(len(lines), frame_line + 3)):
        cand = lines[idx]
        if not cand.strip():
            break
        if _is_setext_underline(cand, "="):
            return 1
        if _is_setext_underline(cand, "-"):
            return 2
        if _is_paragraph_interrupter(cand):
            break
    return None


def resolve_corpus_path(corpus_root: Path | str, rel_path: str) -> Path:
    """Resolve `rel_path` strictly inside `corpus_root`, rejecting traversal or absolute paths."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("Relative corpus path must be a non-empty string.")
    norm = rel_path.replace("\\", "/")
    raw_rel = Path(norm)
    if raw_rel.is_absolute() or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        raise ValueError(f"Absolute path not allowed: {rel_path!r}")
    if ".." in raw_rel.parts:
        raise ValueError(f"Parent traversal not allowed: {rel_path!r}")
    root = Path(corpus_root).resolve()
    resolved = (root / raw_rel).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes corpus root: {rel_path!r}") from exc
    return resolved


def require_corpus_dir(corpus_root: Path | str) -> Path:
    """Validate that `corpus_root` names an existing directory, resolving it or raising CONSISTENCY_ERROR."""
    if not isinstance(corpus_root, (Path, str)) or not str(corpus_root).strip():
        raise YawRuntimeError("CONSISTENCY_ERROR", "Valid corpus_root directory is required.")
    resolved = Path(corpus_root).resolve()
    if not resolved.is_dir():
        raise YawRuntimeError("CONSISTENCY_ERROR", f"corpus_root does not exist or is not a directory: {resolved}")
    return resolved


WIKI_REL_PREFIX = "wiki:"


def is_wiki_rel_path(rel_path: str) -> bool:
    """Return whether `rel_path` carries the canonical `wiki:` provenance prefix."""
    return isinstance(rel_path, str) and rel_path.startswith(WIKI_REL_PREFIX)


def make_wiki_rel_path(rel_path: str) -> str:
    """Normalize `\\` to `/` and prefix a validated relative wiki path with `wiki:`."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("Relative wiki path must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in rel_path):
        raise ValueError(f"Control characters not allowed in wiki path: {rel_path!r}")
    norm = rel_path.replace("\\", "/")
    if norm.startswith(WIKI_REL_PREFIX):
        raise ValueError(f"Wiki path is already prefixed: {rel_path!r}")
    raw_rel = Path(norm)
    if raw_rel.is_absolute() or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
        raise ValueError(f"Absolute path not allowed: {rel_path!r}")
    if ".." in norm.split("/") or not raw_rel.parts or norm.endswith("/"):
        raise ValueError(f"Invalid relative wiki path: {rel_path!r}")
    return f"{WIKI_REL_PREFIX}{norm}"


def strip_wiki_rel_path(rel_path: str) -> str:
    """Strip the `wiki:` prefix from a wiki relative path, raising ValueError if invalid."""
    if not isinstance(rel_path, str) or not rel_path.startswith(WIKI_REL_PREFIX):
        raise ValueError(f"Expected {WIKI_REL_PREFIX!r}-prefixed path, got {rel_path!r}")
    suffix = rel_path[len(WIKI_REL_PREFIX) :]
    if not suffix.strip():
        raise ValueError(f"Wiki path suffix must not be empty: {rel_path!r}")
    return suffix


def resolve_source_path(rel_path: str, corpus_root: Path | str, wiki_root: Path | str | None = None) -> Path:
    """Resolve a corpus or `wiki:`-prefixed relative path against its configured root."""
    if is_wiki_rel_path(rel_path):
        if wiki_root is None:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"Cannot resolve wiki path {rel_path!r} when wiki_root is not configured.",
            )
        return resolve_corpus_path(wiki_root, strip_wiki_rel_path(rel_path))
    return resolve_corpus_path(corpus_root, rel_path)


@dataclass(frozen=True, slots=True)
class ChunkPointer:
    """Zero-copy provenance pointer referencing a disk-backed source span."""

    chunk_id: int
    rel_path: str
    source_hash: str
    start_line: int
    end_line: int
    char_start: int = 0
    char_end: int = 0
    ast_island_start: int = 1
    ast_island_end: int = 1
    title: str = ""

    @property
    def rel_posix_path(self) -> str:
        return self.rel_path.replace("\\", "/")

    @property
    def provenance(self) -> Literal["corpus", "wiki"]:
        return "wiki" if is_wiki_rel_path(self.rel_posix_path) else "corpus"

    @property
    def span_label(self) -> str:
        base = f"{self.rel_posix_path}:{self.start_line}-{self.end_line}"
        return f"{base}@{self.char_start}-{self.char_end}" if (self.char_start or self.char_end) else base


@dataclass(frozen=True, slots=True)
class HydratedPassage:
    """Disk-hydrated passage paired with its provenance pointer and retrieval score."""

    pointer: ChunkPointer
    text: str
    score: float
    stale_on_disk: bool = False
    hydrated_start_line: int | None = None
    hydrated_end_line: int | None = None

    @property
    def resolved_start_line(self) -> int:
        return self.pointer.start_line if self.hydrated_start_line is None else self.hydrated_start_line

    @property
    def resolved_end_line(self) -> int:
        return self.pointer.end_line if self.hydrated_end_line is None else self.hydrated_end_line


@dataclass(frozen=True, slots=True)
class QAResult:
    """Grounded question-answering output with provenance citations."""

    question: str
    answer: str
    citations: tuple[str, ...] = ()
    passages: tuple[HydratedPassage, ...] = ()
    latency_ms: float = 0.0
    turns: int = 0


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    """Non-blocking daemon lifecycle, sync state, and chunk count snapshot."""

    lifecycle: str = "stopped"
    provider_ready: bool = False
    sync_state: str = "idle"
    queue_depth: int = 0
    active_op: str | None = None
    last_error: str | None = None
    last_sync_at: float | None = None
    pid: int | None = None
    num_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ALREADY_PREFIXED_DOC_RE = re.compile(r"^title: (?:none|[^\r\n]+?) \| text: ")
_MAX_EMBED_TITLE_CHARS = 256
_QUERY_PREFIXES: dict[str, str] = {
    "search": "task: search result | query: ",
    "titled": "task: search result | query: ",
    "qa": "task: question answering | query: ",
    "distinct": "task: question answering | query: ",
    "distinct-titled": "task: question answering | query: ",
    "fact": "task: fact checking | query: ",
}
_DOCUMENT_PREFIX = "title: none | text: "


def render_document_text(title: str, body: str) -> str:
    """Render benchmark dataset fields once, preserving an already present title line."""
    return (
        f"{title}\n{body}" if title and body != title and not body.startswith((title + "\n", title + "\r\n")) else body
    )


def format_query_for_embedding(query: str, task_mode: str = "distinct-titled") -> str:
    """Format a query string using the EmbeddingGemma task prefix configured for Yaw."""
    if not isinstance(query, str):
        raise TypeError("Embedding query must be a string.")
    if task_mode not in _QUERY_PREFIXES:
        raise ValueError(f"Unsupported embedding task_mode: {task_mode!r}")
    if any(query.startswith(prefix) for prefix in set(_QUERY_PREFIXES.values())):
        raise ValueError("Pass raw query text, not an already prefixed embedding input.")
    return _QUERY_PREFIXES[task_mode] + query


def format_doc_for_embedding(text: str, title: str = "", task_mode: str = "distinct-titled") -> str:
    """Wrap passage `text` with an explicit known `title`, or `title: none` when unknown."""
    if not isinstance(text, str):
        raise TypeError("Embedding document text must be a string.")
    if task_mode not in _QUERY_PREFIXES:
        raise ValueError(f"Unsupported embedding task_mode: {task_mode!r}")
    if text.startswith(_DOCUMENT_PREFIX) or _ALREADY_PREFIXED_DOC_RE.match(text):
        raise ValueError("Pass raw document text, not an already prefixed embedding input.")
    if task_mode in ("titled", "distinct-titled") and title:
        clean_title = " ".join(title.split()).replace(" | text: ", " - text: ")
        if clean_title and len(clean_title) <= _MAX_EMBED_TITLE_CHARS:
            return f"title: {clean_title} | text: {text}"
    return _DOCUMENT_PREFIX + text


__all__ = [
    "ChunkPointer",
    "DASH_RE",
    "DaemonStatus",
    "EmbedFn",
    "ErrorCode",
    "HydratedPassage",
    "QAResult",
    "WIKI_REL_PREFIX",
    "YawRuntimeError",
    "advance_markdown_fence",
    "canonical_chunk_span_key",
    "compute_chunk_int_id",
    "format_doc_for_embedding",
    "format_query_for_embedding",
    "inclusive_end_line",
    "is_wiki_rel_path",
    "leading_frontmatter_end",
    "make_wiki_rel_path",
    "markdown_heading_level",
    "normalize_ast_heading",
    "redact_secrets",
    "render_document_text",
    "require_corpus_dir",
    "resolve_corpus_path",
    "resolve_source_path",
    "split_lf_lines",
    "strip_wiki_rel_path",
]
