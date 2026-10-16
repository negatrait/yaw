"""1-Indexed disk line hydration with path traversal guard and line span limits."""

import itertools
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

from yaw.core import ChunkPointer, HydratedPassage, resolve_corpus_path

_MAX_CHUNK_LINE_SPAN = 5000


def hydrate_chunk_pointer(
    pointer: ChunkPointer,
    corpus_root: Path,
    score: float = 0.0,
) -> HydratedPassage:
    """Hydrate a 1-indexed [start_line, end_line] text slice from disk for a ChunkPointer."""
    if pointer.start_line < 1 or pointer.end_line < pointer.start_line:
        raise ValueError(
            f"Invalid line span [{pointer.start_line}, {pointer.end_line}] for chunk {pointer.chunk_id}"
        )
    line_count = pointer.end_line - pointer.start_line + 1
    if line_count > _MAX_CHUNK_LINE_SPAN:
        raise ValueError(
            f"Line span {line_count} exceeds maximum {_MAX_CHUNK_LINE_SPAN} for chunk {pointer.chunk_id}"
        )

    target_path = resolve_corpus_path(corpus_root, pointer.rel_posix_path)

    if not target_path.is_file():
        return HydratedPassage(pointer=pointer, text="", score=score, stale_on_disk=True)

    try:
        with target_path.open("r", encoding="utf-8", errors="replace") as f:
            raw_text = "".join(itertools.islice(f, pointer.start_line - 1, pointer.end_line))
    except OSError:
        return HydratedPassage(pointer=pointer, text="", score=score, stale_on_disk=True)

    if not raw_text:
        return HydratedPassage(pointer=pointer, text="", score=score, stale_on_disk=True)

    text = raw_text.rstrip("\r\n")
    return HydratedPassage(
        pointer=pointer,
        text=text,
        score=score,
        stale_on_disk=False,
    )


def hydrate_passages(
    pointers_with_scores: Sequence[tuple[ChunkPointer, float]],
    corpus_root: Path,
) -> tuple[HydratedPassage, ...]:
    """Batch-hydrate multiple ChunkPointers into HydratedPassages."""
    return tuple(
        hydrate_chunk_pointer(ptr, corpus_root, score)
        for ptr, score in pointers_with_scores
    )


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
