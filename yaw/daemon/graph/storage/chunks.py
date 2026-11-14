"""SQLite chunk metadata and passage pointer CRUD operations."""

import sqlite3
from collections.abc import Sequence

from yaw.core import ChunkPointer

from .db import batched, write_tx


def row_to_chunk_pointer(row: sqlite3.Row) -> ChunkPointer:
    """Convert an SQLite Row from the chunks table to a ChunkPointer dataclass."""
    try:
        ast_s = int(row["ast_island_start"])
    except IndexError, KeyError:
        ast_s = 0
    try:
        ast_e = int(row["ast_island_end"])
    except IndexError, KeyError:
        ast_e = 0
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
        "SELECT chunk_id,rel_posix_path,start_line,end_line,title,ast_island_start,ast_island_end "
        "FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return ChunkPointer(
        str(row[0]),
        str(row[1]),
        int(row[2]),
        int(row[3]),
        str(row[4] or ""),
        int(row[5]),
        int(row[6]),
    )


def get_chunk_pointers(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str],
) -> dict[str, ChunkPointer]:
    """Batch retrieve ChunkPointer metadata for given chunk_ids."""
    if not chunk_ids:
        return {}
    results: dict[str, ChunkPointer] = {}
    for batch in batched(chunk_ids):
        for cid, path, sl, el, title, ast_s, ast_e in conn.execute(
            f"SELECT chunk_id,rel_posix_path,start_line,end_line,title,ast_island_start,ast_island_end "
            f"FROM chunks WHERE chunk_id IN({('?,' * len(batch))[:-1]})",
            batch,
        ):
            cid = str(cid)
            results[cid] = ChunkPointer(
                cid,
                str(path),
                int(sl),
                int(el),
                str(title or ""),
                int(ast_s),
                int(ast_e),
            )
    return results


def upsert_chunk_pointers(
    conn: sqlite3.Connection,
    chunks: Sequence[tuple[ChunkPointer, str]],
) -> None:
    """Bulk upsert ChunkPointer metadata into chunks and index passage text in contentless chunk_fts."""
    if not chunks:
        return
    meta = []
    dels = []
    fts = []
    for p, body in chunks:
        cid = p.chunk_id
        title = p.title
        meta.append(
            (
                cid,
                p.rel_posix_path,
                p.start_line,
                p.end_line,
                title,
                p.ast_island_start,
                p.ast_island_end,
            )
        )
        dels.append((cid,))
        fts.append((cid, title, body))
    with write_tx(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO chunks("
            "chunk_id,rel_posix_path,start_line,end_line,title,ast_island_start,ast_island_end"
            ")VALUES(?,?,?,?,?,?,?)",
            meta,
        )
        conn.executemany("DELETE FROM chunk_fts WHERE chunk_id=?", dels)
        conn.executemany("INSERT INTO chunk_fts(chunk_id,title,body)VALUES(?,?,?)", fts)
