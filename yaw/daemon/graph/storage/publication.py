"""Prepared graph rows and persistence-only application within a caller transaction."""

import sqlite3
from dataclasses import dataclass

import numpy as np

from yaw.core import ChunkPointer

from .graph_ops import (
    upsert_chunk_pointers,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_fact_sources,
    upsert_facts,
)
from .vectors import upsert_chunk_vectors, upsert_entity_vectors, upsert_fact_vectors

type FactRow = tuple[str, str, str, str | None, str, str, str, str | None, str | None, str | None]
type VectorRow = tuple[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class PreparedGraphRows:
    chunks: tuple[tuple[ChunkPointer, str], ...]
    entities: tuple[tuple[str, str, str], ...]
    entity_labels: tuple[tuple[str, str, str, str, str], ...]
    facts: tuple[FactRow, ...]
    fact_sources: tuple[tuple[str, str], ...]
    edges: tuple[tuple[str, str, str, float, str], ...]
    chunk_vectors: tuple[VectorRow, ...]
    entity_vectors: tuple[VectorRow, ...]
    fact_vectors: tuple[VectorRow, ...]


def apply_prepared_graph_rows(conn: sqlite3.Connection, rows: PreparedGraphRows) -> None:
    """Apply prepared rows and FTS updates; never perform inference or commit."""
    if not conn.in_transaction:
        raise RuntimeError("Graph publication requires an active transaction")
    upsert_chunk_vectors(conn, rows.chunk_vectors)
    upsert_chunk_pointers(conn, rows.chunks)
    upsert_entities(conn, rows.entities)
    upsert_entity_labels(conn, rows.entity_labels)
    upsert_facts(conn, rows.facts)
    upsert_fact_sources(conn, rows.fact_sources)
    upsert_entity_vectors(conn, rows.entity_vectors)
    upsert_fact_vectors(conn, rows.fact_vectors)
    upsert_edges_with_sources(conn, rows.edges)
