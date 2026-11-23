"""Canonical typed contracts and pure deterministic helpers for Yaw."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
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
    "DaemonStatus",
    "EmbedFn",
    "ErrorCode",
    "HydratedPassage",
    "QAResult",
    "YawRuntimeError",
    "canonical_chunk_span_key",
    "compute_chunk_int_id",
    "format_doc_for_embedding",
    "format_query_for_embedding",
    "inclusive_end_line",
    "redact_secrets",
    "render_document_text",
    "resolve_corpus_path",
]
