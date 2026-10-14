"""Frozen, zero-dependency data contracts for YAW."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ChunkPointer:
    """1-indexed line span pointer to an immutable markdown/text chunk on disk."""

    chunk_id: str
    rel_posix_path: str
    start_line: int
    end_line: int
    title: str
    content_hash: str


@dataclass(slots=True, frozen=True)
class HydratedPassage:
    """A passage hydrated on demand from disk with optional stale detection."""

    pointer: ChunkPointer
    text: str
    score: float
    stale_on_disk: bool = False


@dataclass(slots=True, frozen=True)
class OpenIERecord:
    """NER entities and relation triples extracted from a single chunk."""

    chunk_id: str
    entities: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]


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
