"""Typed desired graph rows; final schemas belong to CocoIndex targets."""

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import cocoindex as coco

from yaw.core import ChunkPointer

from .evidence import EVIDENCE_INDEXES, EntityMention, FactAssertion
from .managed_sqlite import DELETE_ROW, VecSpec, mount_table
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
    vec: VecSpec | None = None
    deleted_keys: tuple[tuple[Any, ...], ...] = ()
    is_delta: bool = False


_VEC_TABLE_NAMES = frozenset({"chunk_vec", "entity_vec", "fact_vec"})

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


def _row_pk(row: Any, key: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(getattr(row, col) for col in key)


def snapshot_table_rows(conn: sqlite3.Connection) -> dict[str, dict[tuple[Any, ...], Any]]:
    """Snapshot seeded rows in a staging connection before delta mutations."""
    existing_tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    snap: dict[str, dict[tuple[Any, ...], Any]] = {}
    for name, row_type, key in GRAPH_TABLES:
        if name not in existing_tables:
            snap[name] = {}
            continue
        columns = ",".join(row_type.__annotations__)
        rows = (row_type(*row) for row in conn.execute(f"SELECT {columns} FROM {name} ORDER BY {','.join(key)}"))
        snap[name] = {_row_pk(r, key): r for r in rows}
    return snap


def graph_tables(
    conn: sqlite3.Connection,
    chunks: Sequence[tuple[ChunkPointer, str]],
    fingerprints: Sequence[tuple[str, str]],
    *,
    seeded_rows_by_table: Mapping[str, Mapping[tuple[Any, ...], Any]] | None = None,
    db_identity: str = "",
) -> tuple[DesiredTable, ...]:
    is_delta = seeded_rows_by_table is not None
    tables = []
    deleted_chunk_fts: tuple[tuple[Any, ...], ...] = ()
    for name, row_type, key in GRAPH_TABLES:
        columns = ",".join(row_type.__annotations__)
        values = tuple(row_type(*row) for row in conn.execute(f"SELECT {columns} FROM {name} ORDER BY {','.join(key)}"))
        vec = (
            VecSpec(
                "embedding",
                len(values[0].embedding) // 4
                if values and isinstance(values[0], (ChunkVector, EntityVector, FactVector))
                else None,
            )
            if name in _VEC_TABLE_NAMES
            else None
        )
        if seeded_rows_by_table is None:
            tables.append(DesiredTable(name, row_type, key, values, vec=vec))
        else:
            seeded = seeded_rows_by_table.get(name, {})
            current_map = {_row_pk(r, key): r for r in values}
            changed_rows = tuple(r for pk, r in current_map.items() if seeded.get(pk) != r)
            deleted_keys = tuple(sorted(pk for pk in seeded if pk not in current_map))
            if name == "chunks":
                deleted_chunk_fts = deleted_keys
            tables.append(
                DesiredTable(name, row_type, key, changed_rows, vec=vec, deleted_keys=deleted_keys, is_delta=True)
            )
    tables.append(
        DesiredTable(
            "chunk_fts",
            ChunkText,
            ("chunk_id",),
            tuple(ChunkText(pointer.chunk_id, pointer.title, text) for pointer, text in chunks),
            ("chunk_id",),
            deleted_keys=deleted_chunk_fts,
            is_delta=is_delta,
        )
    )
    facts = []
    for fid, predicate, raw in conn.execute("SELECT fact_id,pred,raw_triple_json FROM facts ORDER BY fact_id"):
        subject, _, obj = json.loads(raw)
        facts.append(FactText(fid, f"{subject} {predicate} {obj}"))
    if seeded_rows_by_table is None:
        tables.append(DesiredTable("fact_fts", FactText, ("fact_id",), tuple(facts), ("fact_id",)))
    else:
        seeded_facts = seeded_rows_by_table.get("facts", {})
        seeded_fact_fts: dict[tuple[Any, ...], FactText] = {}
        for pk, f_row in seeded_facts.items():
            s_lbl, _, o_lbl = json.loads(f_row.raw_triple_json)
            seeded_fact_fts[pk] = FactText(f_row.fact_id, f"{s_lbl} {f_row.pred} {o_lbl}")
        current_fact_fts = {(r.fact_id,): r for r in facts}
        changed_fact_fts = tuple(r for pk, r in current_fact_fts.items() if seeded_fact_fts.get(pk) != r)
        deleted_fact_fts = tuple(sorted(pk for pk in seeded_fact_fts if pk not in current_fact_fts))
        tables.append(
            DesiredTable(
                "fact_fts",
                FactText,
                ("fact_id",),
                changed_fact_fts,
                ("fact_id",),
                deleted_keys=deleted_fact_fts,
                is_delta=True,
            )
        )
    meta_rows: list[MetaRow] = [MetaRow("graph_input_fingerprints", json.dumps(tuple(fingerprints)))]
    if db_identity:
        meta_rows.append(MetaRow("db_identity", db_identity))
    tables.append(DesiredTable("meta", MetaRow, ("key",), tuple(meta_rows)))
    return tuple(tables)


def empty_delta_graph_tables(
    fingerprints: Sequence[tuple[str, str]],
    *,
    db_identity: str = "",
) -> tuple[DesiredTable, ...]:
    meta_rows: list[MetaRow] = [MetaRow("graph_input_fingerprints", json.dumps(tuple(fingerprints)))]
    if db_identity:
        meta_rows.append(MetaRow("db_identity", db_identity))
    return (
        *(
            DesiredTable(
                name,
                row_type,
                key,
                (),
                vec=VecSpec("embedding", None) if name in _VEC_TABLE_NAMES else None,
                is_delta=True,
            )
            for name, row_type, key in GRAPH_TABLES
        ),
        DesiredTable("chunk_fts", ChunkText, ("chunk_id",), (), ("chunk_id",), is_delta=True),
        DesiredTable("fact_fts", FactText, ("fact_id",), (), ("fact_id",), is_delta=True),
        DesiredTable("meta", MetaRow, ("key",), tuple(meta_rows)),
    )


def empty_graph_tables(
    *,
    seeded_rows_by_table: Mapping[str, Mapping[tuple[Any, ...], Any]] | None = None,
    db_identity: str = "",
) -> tuple[DesiredTable, ...]:
    meta_rows: list[MetaRow] = [MetaRow("graph_input_fingerprints", "[]")]
    if db_identity:
        meta_rows.append(MetaRow("db_identity", db_identity))
    deleted_chunks = tuple(sorted(seeded_rows_by_table.get("chunks", {}))) if seeded_rows_by_table is not None else ()
    deleted_facts = tuple(sorted(seeded_rows_by_table.get("facts", {}))) if seeded_rows_by_table is not None else ()
    return (
        *(
            DesiredTable(
                name,
                row_type,
                key,
                (),
                vec=VecSpec("embedding", None) if name in _VEC_TABLE_NAMES else None,
                deleted_keys=(
                    tuple(sorted(seeded_rows_by_table.get(name, {}))) if seeded_rows_by_table is not None else ()
                ),
            )
            for name, row_type, key in GRAPH_TABLES
        ),
        DesiredTable("chunk_fts", ChunkText, ("chunk_id",), (), ("chunk_id",), deleted_keys=deleted_chunks),
        DesiredTable("fact_fts", FactText, ("fact_id",), (), ("fact_id",), deleted_keys=deleted_facts),
        DesiredTable("meta", MetaRow, ("key",), tuple(meta_rows)),
    )


async def declare_tables(tables: tuple[DesiredTable, ...], *, db_identity: str = "") -> None:
    shared_vec_dim = next(
        (
            (len(t.rows[0].embedding) // 4 if t.rows else (t.vec.dim if t.vec is not None else None))
            for t in tables
            if (t.vec is not None or t.name in _VEC_TABLE_NAMES)
            and (t.rows or (t.vec is not None and t.vec.dim is not None))
        ),
        None,
    )
    targets = [
        await mount_table(
            table.name,
            table.row_type,
            list(table.primary_key),
            fts_unindexed=table.fts_unindexed,
            indexes=EVIDENCE_INDEXES.get(table.name, ()),
            db_identity=db_identity,
            vec=VecSpec(
                column=table.vec.column if table.vec is not None else "embedding",
                dim=(len(table.rows[0].embedding) // 4)
                if table.rows
                else (table.vec.dim if table.vec is not None and table.vec.dim is not None else shared_vec_dim),
                distance_metric=table.vec.distance_metric if table.vec is not None else "cosine",
            )
            if (table.vec is not None or table.name in _VEC_TABLE_NAMES)
            else None,
            is_delta=table.is_delta,
        )
        for table in tables
    ]
    for target, table in zip(targets, tables, strict=True):
        for row in table.rows:
            target.declare_row(row=row)
        for key in table.deleted_keys:
            coco.declare_target_state(target._provider.target_state(key, DELETE_ROW))
        if table.name == "meta" and db_identity and not any(row.key == "db_identity" for row in table.rows):
            target.declare_row(row=MetaRow("db_identity", db_identity))


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
