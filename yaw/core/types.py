"""Frozen, zero-dependency data contracts and pure coordinate/ID helpers for YAW."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ErrorCode = Literal[
    "INVALID_INPUT",
    "NOT_READY",
    "EMPTY_INDEX",
    "BUSY",
    "MODEL_ERROR",
    "CONSISTENCY_ERROR",
    "SHUTTING_DOWN",
]


class YawRuntimeError(RuntimeError):
    """Structured runtime error carrying a canonical ErrorCode for CLI and MCP transport."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code: ErrorCode = code
        self.message: str = message


@dataclass(slots=True, frozen=True)
class ChunkPointer:
    """1-indexed line span pointer to an immutable markdown/text chunk on disk."""

    chunk_id: str
    rel_posix_path: str
    start_line: int
    end_line: int
    title: str = ""
    ast_island_start: int = 0
    ast_island_end: int = 0


@dataclass(slots=True, frozen=True)
class HydratedPassage:
    """A passage hydrated on demand from disk with optional stale detection."""

    pointer: ChunkPointer
    text: str
    score: float
    stale_on_disk: bool = False


@dataclass(slots=True, frozen=True)
class StructuredTriple:
    """An OpenIE relation triple enriched with inline subject, object, and statement definitions."""

    subj: str
    subj_def: str = ""
    pred: str = ""
    obj: str = ""
    obj_def: str = ""
    spo_def: str = ""

    @property
    def subject(self) -> str:
        return self.subj

    @property
    def predicate(self) -> str:
        return self.pred

    @property
    def object(self) -> str:
        return self.obj

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subj, self.pred, self.obj)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class OpenIERecord:
    """NER entities and relation triples extracted from a single chunk."""

    chunk_id: str
    entities: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]
    structured_triples: tuple[StructuredTriple, ...] = ()
    entity_definitions: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True, frozen=True)
class SHACLViolation:
    """Closed-world SHACL validation report item on the SQLite knowledge graph."""

    shape: str
    focus_node: str
    result_path: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Retrieved top-k hydrated passages and optional generated answer."""

    question: str
    passages: tuple[HydratedPassage, ...]
    answer: str | None = None


@dataclass(slots=True, frozen=True)
class SyncDelta:
    """Coalesced file additions and deletions emitted by the ingestion pipeline."""

    added_chunks: tuple[tuple[ChunkPointer, str], ...] = ()
    deleted_chunk_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class IndexStats:
    """Cached non-blocking graph index statistics and configuration fingerprint."""

    chunks: int = 0
    entities: int = 0
    facts: int = 0
    edges: int = 0
    config_fingerprint: str = ""
    openie_checkpoints: int = 0
    entity_labels: int = 0
    shacl_conforms: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DaemonStatus:
    """Cached non-blocking daemon lifecycle, provider health, and worker queue snapshot."""

    lifecycle: str = "stopped"
    provider_ready: bool = False
    sync_state: str = "idle"
    queue_depth: int = 0
    active_op: str | None = None
    last_error: str | None = None
    last_sync_at: float | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute the MD5 hash of the given content string and optionally prepend a prefix."""
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def compute_chunk_id(rel_posix_path: str, start_line: int, end_line: int, text: str) -> str:
    """Compute deterministic canonical chunk ID from relative path, line span, and slice text."""
    payload = f"{rel_posix_path}:{start_line}:{end_line}:{text}"
    return compute_mdhash_id(payload, prefix="chunk-")


def inclusive_end_line(start_line: int, end_line: int, end_column: int) -> int:
    """Convert half-open coordinate (end_line, end_column) to 1-indexed inclusive end line."""
    if end_column == 1 and end_line > start_line:
        return end_line - 1
    return end_line


def resolve_corpus_path(corpus_root: Path, rel_posix_path: str) -> Path:
    """Resolve a relative POSIX path against a corpus root, rejecting path traversals."""
    root_resolved = corpus_root.resolve()
    target_path = (root_resolved / rel_posix_path).resolve()
    if not target_path.is_relative_to(root_resolved):
        raise ValueError(f"Path traversal detected: {rel_posix_path!r} resolves outside corpus root {corpus_root}")
    return target_path


def to_rel_posix_path(corpus_root: Path, file_path: Path | str) -> str:
    """Convert an absolute or relative file path to a relative POSIX path under corpus_root."""
    root_resolved = corpus_root.resolve()
    target_path = Path(file_path).resolve()
    if not target_path.is_relative_to(root_resolved):
        raise ValueError(f"File path {file_path!r} resolves outside corpus root {corpus_root}")
    return target_path.relative_to(root_resolved).as_posix()
