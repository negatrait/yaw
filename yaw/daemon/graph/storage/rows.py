"""Typed desired graph rows; final schemas belong to CocoIndex targets."""

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from yaw.core import ChunkPointer

from .evidence import EVIDENCE_INDEXES, EntityMention, FactAssertion
from .managed_sqlite import mount_table
from .shacl import lint_graph_shacl


class ChunkRow(NamedTuple):
    chunk_id: str
    rel_posix_path: str
    start_line: int
    end_line: int
    title: str
    ast_island_start: int
    ast_island_end: int


class EntityRow(NamedTuple):
    entity_id: str
    name: str
    definition: str
    confidence: float | None


class LabelRow(NamedTuple):
    label_norm: str
    entity_id: str
    label_kind: str
    raw_label: str
    chunk_id: str


class FactRow(NamedTuple):
    fact_id: str
    subj_id: str
    pred: str
    obj_kind: str
    obj_id: str | None
    obj_literal_lex: str | None
    obj_datatype: str
    obj_lang: str
    raw_triple_json: str
    definition: str
    confidence: float | None


class FactSourceRow(NamedTuple):
    fact_id: str
    chunk_id: str
    llm_prob: float | None
    vec_coherence: float | None
    atomic_confidence: float | None


class EdgeRow(NamedTuple):
    src_id: str
    dst_id: str
    kind: str
    weight: float
    reasoning: str


class EdgeSourceRow(NamedTuple):
    src_id: str
    dst_id: str
    kind: str
    chunk_id: str
    llm_prob: float | None
    vec_coherence: float | None
    atomic_confidence: float | None


class ChunkVector(NamedTuple):
    chunk_id: str
    embedding: bytes


class EntityVector(NamedTuple):
    entity_id: str
    embedding: bytes


class FactVector(NamedTuple):
    fact_id: str
    embedding: bytes


class ChunkText(NamedTuple):
    chunk_id: str
    title: str
    body: str


class FactText(NamedTuple):
    fact_id: str
    text: str


class MetaRow(NamedTuple):
    key: str
    value: str


@dataclass(frozen=True)
class DesiredTable:
    name: str
    row_type: type[Any]
    primary_key: tuple[str, ...]
    rows: tuple[Any, ...]
    fts_unindexed: tuple[str, ...] | None = None


GRAPH_TABLES = (
    ("chunks", ChunkRow, ("chunk_id",)),
    ("entities", EntityRow, ("entity_id",)),
    ("entity_labels", LabelRow, ("label_norm", "entity_id", "chunk_id")),
    ("facts", FactRow, ("fact_id",)),
    ("fact_sources", FactSourceRow, ("fact_id", "chunk_id")),
    ("edges", EdgeRow, ("src_id", "dst_id", "kind")),
    ("edge_sources", EdgeSourceRow, ("src_id", "dst_id", "kind", "chunk_id")),
    ("chunk_vec", ChunkVector, ("chunk_id",)),
    ("entity_vec", EntityVector, ("entity_id",)),
    ("fact_vec", FactVector, ("fact_id",)),
    ("entity_mentions", EntityMention, ("mention_id",)),
    ("fact_assertions", FactAssertion, ("assertion_id",)),
)


def graph_tables(
    conn: sqlite3.Connection, chunks: Sequence[tuple[ChunkPointer, str]], fingerprints: Sequence[tuple[str, str]]
) -> tuple[DesiredTable, ...]:
    tables = []
    for name, row_type, key in GRAPH_TABLES:
        columns = ",".join(row_type.__annotations__)
        values = tuple(row_type(*row) for row in conn.execute(f"SELECT {columns} FROM {name} ORDER BY {','.join(key)}"))
        tables.append(DesiredTable(name, row_type, key, values))
    tables.append(
        DesiredTable(
            "chunk_fts",
            ChunkText,
            ("chunk_id",),
            tuple(ChunkText(pointer.chunk_id, pointer.title, text) for pointer, text in chunks),
            ("chunk_id",),
        )
    )
    facts = []
    for fid, predicate, raw in conn.execute("SELECT fact_id,pred,raw_triple_json FROM facts ORDER BY fact_id"):
        subject, _, obj = json.loads(raw)
        facts.append(FactText(fid, f"{subject} {predicate} {obj}"))
    tables.append(DesiredTable("fact_fts", FactText, ("fact_id",), tuple(facts), ("fact_id",)))
    tables.append(
        DesiredTable("meta", MetaRow, ("key",), (MetaRow("graph_input_fingerprints", json.dumps(tuple(fingerprints))),))
    )
    return tuple(tables)


def empty_graph_tables() -> tuple[DesiredTable, ...]:
    return (
        *(DesiredTable(name, row_type, key, ()) for name, row_type, key in GRAPH_TABLES),
        DesiredTable("chunk_fts", ChunkText, ("chunk_id",), (), ("chunk_id",)),
        DesiredTable("fact_fts", FactText, ("fact_id",), (), ("fact_id",)),
        DesiredTable("meta", MetaRow, ("key",), (MetaRow("graph_input_fingerprints", "[]"),)),
    )


async def declare_tables(tables: tuple[DesiredTable, ...]) -> None:
    targets = [
        await mount_table(
            table.name,
            table.row_type,
            list(table.primary_key),
            fts_unindexed=table.fts_unindexed,
            indexes=EVIDENCE_INDEXES.get(table.name, ()),
        )
        for table in tables
    ]
    for target, table in zip(targets, tables, strict=True):
        for row in table.rows:
            target.declare_row(row=row)


def validate_graph_state(conn: sqlite3.Connection) -> None:
    violations = lint_graph_shacl(conn)
    if violations:
        raise ValueError(f"Graph publication rejected SHACL violations: {violations[:3]}")
    if conn.execute(
        "SELECT 1 FROM entity_mentions m LEFT JOIN entities e ON e.entity_id=m.entity_id "
        "LEFT JOIN chunks c ON c.chunk_id=m.chunk_id WHERE e.entity_id IS NULL OR c.chunk_id IS NULL LIMIT 1"
    ).fetchone():
        raise ValueError("Dangling source mention binding")
    if (
        conn.execute(
            "SELECT 1 FROM entity_mentions GROUP BY sense_id HAVING COUNT(DISTINCT entity_id)>1 LIMIT 1"
        ).fetchone()
        or conn.execute(
            "SELECT 1 FROM entity_mentions GROUP BY entity_id HAVING entity_id!=MIN(sense_id) LIMIT 1"
        ).fetchone()
    ):
        raise ValueError("Noncanonical source mention binding")
    if conn.execute(
        "SELECT 1 FROM fact_assertions a LEFT JOIN entity_mentions s ON s.mention_id=a.subject_mention_id "
        "LEFT JOIN entity_mentions o ON o.mention_id=a.object_mention_id LEFT JOIN facts f ON f.fact_id=a.fact_id "
        "LEFT JOIN chunks c ON c.chunk_id=a.chunk_id "
        "WHERE c.chunk_id IS NULL OR s.mention_id IS NULL "
        "OR a.chunk_id!=s.chunk_id OR a.source_revision!=s.source_revision "
        "OR (a.object_mention_id IS NOT NULL AND (o.mention_id IS NULL "
        "OR a.chunk_id!=o.chunk_id OR a.source_revision!=o.source_revision)) "
        "OR (a.fact_id IS NOT NULL AND f.fact_id IS NULL) LIMIT 1"
    ).fetchone():
        raise ValueError("Invalid source assertion binding")
