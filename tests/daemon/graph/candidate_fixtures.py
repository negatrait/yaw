"""Explicit source-vector inputs for storage-only candidate fixtures."""

import asyncio
import sqlite3

from yaw.core import text_processing
from yaw.daemon.graph.storage.evidence import evidence_schemas
from yaw.daemon.graph.storage.managed_sqlite import create_table


def seed_candidate_vectors(conn: sqlite3.Connection) -> None:
    schema = next(s for s in asyncio.run(evidence_schemas()) if s.name == "entity_candidate_vectors")
    create_table(conn, schema)
    rows = conn.execute(
        "SELECT e.entity_id,e.name,e.definition,v.embedding FROM entities e "
        "LEFT JOIN entity_vec v ON e.entity_id=v.entity_id WHERE e.entity_id!=''"
    ).fetchall()
    conn.executemany(
        "INSERT OR REPLACE INTO entity_candidate_vectors VALUES (?,?,?,?,?)",
        [
            (
                eid,
                eid,
                str(text_processing(name)),
                definition.strip().casefold(),
                blob if blob is not None else bytes(8),
            )
            for eid, name, definition, blob in rows
        ],
    )
