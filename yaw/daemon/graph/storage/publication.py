"""Prepared graph rows and persistence-only application within a caller transaction."""

import json
import sqlite3
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors.sqlite import ManagedConnection

from yaw.core import ChunkPointer

from .chunks import upsert_chunk_pointers
from .graph_ops import (
    delete_chunks,
    upsert_concept_scheme_members,
    upsert_concept_schemes,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_fact_sources,
    upsert_facts,
)
from .shacl import lint_graph_shacl
from .vectors import upsert_chunk_vectors, upsert_entity_vectors, upsert_fact_vectors

type FactRow = tuple[str, str, str, str | None, str, str, str, str | None, str | None, str | None]
type VectorRow = tuple[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class PreparedGraphRows:
    chunks: tuple[tuple[ChunkPointer, str], ...]
    entities: tuple[tuple[str, str, str], ...] | tuple[tuple[str, str, str, float | None], ...]
    entity_labels: tuple[tuple[str, str, str, str, str], ...]
    facts: tuple[FactRow, ...]
    fact_sources: tuple[tuple[str, str], ...] | tuple[tuple[str, str, float | None, float | None, float | None], ...]
    edges: tuple[tuple[str, str, str, float, str], ...]
    chunk_vectors: tuple[VectorRow, ...]
    entity_vectors: tuple[VectorRow, ...]
    fact_vectors: tuple[VectorRow, ...]
    concept_schemes: tuple[tuple[str, str, str, str, float | None], ...] = ()
    concept_scheme_members: tuple[tuple[str, str, int], ...] = ()
    input_fingerprints: tuple[tuple[str, str], ...] = ()


def clear_concept_schemes(conn: sqlite3.Connection) -> None:
    """Remove derived taxonomy before replacing it from current community membership."""
    if not conn.in_transaction:
        raise RuntimeError("Taxonomy replacement requires an active transaction")
    conn.execute("DELETE FROM edge_sources WHERE kind='skos:topConceptOf'")
    conn.execute("DELETE FROM edges WHERE kind='skos:topConceptOf'")
    conn.execute("DELETE FROM entity_vec WHERE entity_id IN (SELECT scheme_id FROM concept_schemes)")
    conn.execute("DELETE FROM concept_scheme_members")
    conn.execute("DELETE FROM concept_schemes")


def apply_prepared_graph_rows(conn: sqlite3.Connection, rows: PreparedGraphRows) -> None:
    """Apply prepared rows and FTS updates; never perform inference or commit."""
    if not conn.in_transaction:
        raise RuntimeError("Graph publication requires an active transaction")
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('graph_input_fingerprints', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (json.dumps(rows.input_fingerprints),),
    )
    upsert_chunk_vectors(conn, rows.chunk_vectors)
    upsert_chunk_pointers(conn, rows.chunks)
    upsert_entities(conn, rows.entities)
    upsert_entity_labels(conn, rows.entity_labels)
    upsert_facts(conn, rows.facts)
    upsert_fact_sources(conn, rows.fact_sources)
    upsert_entity_vectors(conn, rows.entity_vectors)
    upsert_fact_vectors(conn, rows.fact_vectors)
    upsert_edges_with_sources(conn, rows.edges)
    if rows.concept_schemes:
        upsert_concept_schemes(conn, rows.concept_schemes)
    if rows.concept_scheme_members:
        upsert_concept_scheme_members(conn, rows.concept_scheme_members)


GRAPH_DB = coco.ContextKey[ManagedConnection]("yaw/graph_db")


@dataclass(frozen=True, slots=True)
class GraphTrackingRecord:
    fingerprint: bytes
    chunks: tuple[tuple[str, bytes], ...]
    global_edges: tuple[tuple[str, str, str], ...]


def contribution_fingerprints(rows: PreparedGraphRows) -> tuple[tuple[str, bytes], ...]:
    """Fingerprint source-owned rows using the graph's existing provenance."""
    contributions: dict[str, list[Any]] = {pointer.chunk_id: [pointer, text] for pointer, text in rows.chunks}
    if len(contributions) != len(rows.chunks):
        raise ValueError("Graph declaration contains duplicate chunk identities")
    entities = {row[0]: row for row in rows.entities}
    entity_vectors = dict(rows.entity_vectors)
    facts = {row[0]: row for row in rows.facts}
    fact_vectors = dict(rows.fact_vectors)
    for chunk_id, vector in rows.chunk_vectors:
        contributions[chunk_id].append(("vector", vector))
    for label in rows.entity_labels:
        chunk_id = label[4]
        if chunk_id in contributions:
            contributions[chunk_id].append(("label", label, entities.get(label[1]), entity_vectors.get(label[1])))
    for fact_source_row in rows.fact_sources:
        fact_id, chunk_id = str(fact_source_row[0]), str(fact_source_row[1])
        if chunk_id in contributions:
            contributions[chunk_id].append(("fact", facts[fact_id], fact_vectors.get(fact_id)))
    for edge in rows.edges:
        if edge[4] in contributions:
            contributions[edge[4]].append(("edge", edge))
    return tuple((chunk_id, fingerprint_object(payload)) for chunk_id, payload in sorted(contributions.items()))


@dataclass(frozen=True, slots=True)
class GraphAction:
    rows: PreparedGraphRows | None
    removed_chunks: tuple[str, ...]
    removed_global_edges: tuple[tuple[str, str, str], ...]


class GraphTargetHandler(coco.TargetHandler[PreparedGraphRows, GraphTrackingRecord]):
    """One complete prepared graph declaration, atomically reconciled by CocoIndex.

    Tracking records contain fingerprints and ownership IDs, never vectors or
    source text. The graph planner supplies all desired rows, including survivors
    of removed source contributions. No inference runs in this sink.
    """

    def __init__(self) -> None:
        self.sink = coco.TargetActionSink.from_fn(self.apply_actions)

    def apply_actions(self, context: coco.ContextProvider, actions: Sequence[GraphAction]) -> None:
        with context.get(GRAPH_DB).transaction() as conn:
            for action in actions:
                for src, dst, kind in action.removed_global_edges:
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id=? AND dst_id=? AND kind=? AND chunk_id=''",
                        (src, dst, kind),
                    )
                    conn.execute(
                        "DELETE FROM edges WHERE src_id=? AND dst_id=? AND kind=? AND NOT EXISTS "
                        "(SELECT 1 FROM edge_sources s WHERE s.src_id=edges.src_id "
                        "AND s.dst_id=edges.dst_id AND s.kind=edges.kind)",
                        (src, dst, kind),
                    )
                if action.removed_chunks:
                    delete_chunks(conn, action.removed_chunks)
                if action.rows is not None:
                    clear_concept_schemes(conn)
                    apply_prepared_graph_rows(conn, action.rows)
                else:
                    conn.execute("DELETE FROM meta WHERE key='graph_input_fingerprints'")
            violations = lint_graph_shacl(conn)
            if violations:
                raise ValueError(f"Graph publication rejected {len(violations)} SHACL violations: {violations[:3]}")

    def reconcile(
        self,
        key: coco.StableKey,
        desired: PreparedGraphRows | coco.NonExistenceType,
        previous: Collection[GraphTrackingRecord],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[Any, Any, Any] | None:
        if str(key) != "graph":
            raise ValueError("The atomic graph target requires the single 'graph' key")
        record = (
            coco.NON_EXISTENCE
            if coco.is_non_existence(desired)
            else GraphTrackingRecord(
                fingerprint_object(desired),
                contribution_fingerprints(desired),
                tuple(sorted({(src, dst, kind) for src, dst, kind, _, cid in desired.edges if not cid})),
            )
        )
        change = statediff.diff(
            statediff.TrackingRecordTransition(
                desired=record,
                prev=previous,
                prev_may_be_missing=prev_may_be_missing,
            )
        )
        if change is None:
            return None
        previous_chunks = {chunk for item in previous for chunk, _ in item.chunks}
        previous_global_edges = {edge for item in previous for edge in item.global_edges}
        if coco.is_non_existence(desired):
            action = GraphAction(None, tuple(sorted(previous_chunks)), tuple(sorted(previous_global_edges)))
        else:
            assert not coco.is_non_existence(record)
            desired_chunks = dict(record.chunks)
            replaced = {
                chunk
                for item in previous
                for chunk, fingerprint in item.chunks
                if desired_chunks.get(chunk) != fingerprint
            }
            action = GraphAction(
                desired,
                tuple(sorted(replaced)),
                tuple(sorted(previous_global_edges - set(record.global_edges))),
            )
        return coco.TargetReconcileOutput(action=action, sink=self.sink, tracking_record=record)


GRAPH_TARGET = coco.register_root_target_states_provider("yaw/graph/publication", GraphTargetHandler())


def declare_graph(rows: PreparedGraphRows) -> None:
    coco.declare_target_state(GRAPH_TARGET.target_state("graph", rows))
