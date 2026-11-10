"""SQLite chunk metadata and passage pointer CRUD operations."""

import sqlite3
from collections.abc import Sequence

from yaw.core import ChunkPointer

from .db import batched, write_tx


def row_to_chunk_pointer(row: sqlite3.Row) -> ChunkPointer:
    """Convert an SQLite Row from the chunks table to a ChunkPointer dataclass."""
    keys = row.keys()
    ast_s = int(row["ast_island_start"]) if "ast_island_start" in keys else 0
    ast_e = int(row["ast_island_end"]) if "ast_island_end" in keys else 0
    return ChunkPointer(
        chunk_id=str(row["chunk_id"]),
        rel_posix_path=str(row["rel_posix_path"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        title=str(row["title"] or ""),
        ast_island_start=ast_s,
        ast_island_end=ast_e,
    )


def get_chunk_pointer(conn: sqlite3.Connection, chunk_id: str) -> ChunkPointer | None:
    """Retrieve ChunkPointer metadata for a single chunk_id."""
    row = conn.execute(
        "SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, "
        "ast_island_end FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    return row_to_chunk_pointer(row) if row is not None else None


def get_chunk_pointers(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str],
) -> dict[str, ChunkPointer]:
    """Batch retrieve ChunkPointer metadata for given chunk_ids."""
    if not chunk_ids:
        return {}
    results: dict[str, ChunkPointer] = {}
    for batch in batched(chunk_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end
            FROM chunks WHERE chunk_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            results[str(row["chunk_id"])] = row_to_chunk_pointer(row)
    return results


def upsert_chunk_pointers(
    conn: sqlite3.Connection,
    chunks: Sequence[tuple[ChunkPointer, str]],
) -> None:
    """Bulk upsert ChunkPointer metadata into chunks and index passage text in contentless chunk_fts."""
    if not chunks:
        return
    with write_tx(conn):
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    p.chunk_id,
                    p.rel_posix_path,
                    p.start_line,
                    p.end_line,
                    p.title,
                    p.ast_island_start,
                    p.ast_island_end,
                )
                for p, _ in chunks
            ),
        )
        conn.executemany(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            ((p.chunk_id,) for p, _ in chunks),
        )
        conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, title, body) VALUES (?, ?, ?)",
            ((p.chunk_id, p.title, body) for p, body in chunks),
        )
