"""CG-GRAPH-1 capability gate and real sqlite-vec vec0 virtual table invariants."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cocoindex as coco
import numpy as np
import pytest
from daemon.graph.candidate_fixtures import seed_candidate_vectors

from yaw.daemon.graph.storage import (
    find_entity_candidates,
    open_managed_yaw_db,
    open_yaw_db,
    search_chunk_vectors_knn,
    search_entity_vectors_knn,
    search_fact_vectors_knn,
    serialize_f32_vector,
    upsert_chunk_vectors,
    upsert_entity_vectors,
    upsert_fact_vectors,
)
from yaw.daemon.graph.storage.db import GRAPH_DB, ensure_vec0_table, vec0_table_dimension
from yaw.daemon.graph.storage.evidence import CandidateVector, EntityMention, FactAssertion, fact_id, sense_id
from yaw.daemon.graph.storage.graph_ops import rekey_facts, upsert_entities, upsert_fact_sources, upsert_facts
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE
from yaw.daemon.graph.storage.rows import (
    ChunkRow,
    ChunkText,
    ChunkVector,
    DesiredTable,
    EdgeRow,
    EdgeSourceRow,
    EntityRow,
    EntityVector,
    FactRow,
    FactSourceRow,
    FactText,
    FactVector,
    LabelRow,
    MetaRow,
    declare_tables,
    empty_graph_tables,
    validate_graph_state,
)


def _exact_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


@dataclass
class _DesiredGraphSource:
    tables: tuple[DesiredTable, ...]


@coco.fn
async def _publish_desired_graph(source: _DesiredGraphSource) -> None:
    await declare_tables(source.tables)


def _build_valid_desired_tables(
    *,
    chunk_id: str,
    chunk_vec: np.ndarray,
    entity_a_def: str,
    entity_a_vec: np.ndarray,
    entity_b_vec: np.ndarray,
    fact_vec: np.ndarray,
    revision: str,
) -> tuple[DesiredTable, ...]:
    ent_a = sense_id("Ada Lovelace", "first computer programmer")
    ent_b = sense_id("Analytical Engine", "mechanical computer")
    fid = fact_id(ent_a, "wrote notes on", "entity", ent_b)
    raw_triple = json.dumps(["Ada Lovelace", "wrote notes on", "Analytical Engine"])
    rows_by_table = {
        "chunks": (ChunkRow(chunk_id, "notes.md", 1, 4, "Notes", 0, 0),),
        "entities": (
            EntityRow(ent_a, "Ada Lovelace", entity_a_def, 0.95),
            EntityRow(ent_b, "Analytical Engine", "mechanical computer", 0.90),
        ),
        "entity_labels": (
            LabelRow("ada lovelace", ent_a, "skos:prefLabel", "Ada Lovelace", chunk_id),
            LabelRow("analytical engine", ent_b, "skos:prefLabel", "Analytical Engine", chunk_id),
        ),
        "facts": (
            FactRow(
                fid,
                ent_a,
                "wrote notes on",
                "entity",
                ent_b,
                None,
                "",
                "",
                raw_triple,
                "Ada Lovelace wrote notes on the Analytical Engine.",
                0.92,
            ),
        ),
        "fact_sources": (FactSourceRow(fid, chunk_id, 0.9, 0.94, 0.92),),
        "edges": (
            EdgeRow(ent_a, chunk_id, "passage", 1.0, ""),
            EdgeRow(ent_b, chunk_id, "passage", 1.0, ""),
            EdgeRow(ent_a, ent_b, "fact", 0.92, ""),
        ),
        "edge_sources": (
            EdgeSourceRow(ent_a, chunk_id, "passage", chunk_id, None, None, 1.0),
            EdgeSourceRow(ent_b, chunk_id, "passage", chunk_id, None, None, 1.0),
            EdgeSourceRow(ent_a, ent_b, "fact", chunk_id, 0.9, 0.94, 0.92),
        ),
        "chunk_vec": (ChunkVector(chunk_id, serialize_f32_vector(chunk_vec)),),
        "entity_vec": (
            EntityVector(ent_a, serialize_f32_vector(entity_a_vec)),
            EntityVector(ent_b, serialize_f32_vector(entity_b_vec)),
        ),
        "entity_candidate_vectors": (
            CandidateVector(
                ent_a,
                ent_a,
                "ada lovelace",
                "first computer programmer",
                serialize_f32_vector(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
            ),
            CandidateVector(
                ent_b,
                ent_b,
                "analytical engine",
                "mechanical computer",
                serialize_f32_vector(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
            ),
        ),
        "fact_vec": (FactVector(fid, serialize_f32_vector(fact_vec)),),
        "entity_mentions": (
            EntityMention(
                f"{chunk_id}:m1",
                chunk_id,
                revision,
                ent_a,
                "Ada Lovelace",
                "first computer programmer",
                "first computer programmer",
                ent_a,
            ),
            EntityMention(
                f"{chunk_id}:m2",
                chunk_id,
                revision,
                ent_b,
                "Analytical Engine",
                "mechanical computer",
                "mechanical computer",
                ent_b,
            ),
        ),
        "fact_assertions": (
            FactAssertion(
                f"{chunk_id}:a1",
                chunk_id,
                revision,
                f"{chunk_id}:m1",
                f"{chunk_id}:m2",
                raw_triple,
                fid,
            ),
        ),
        "chunk_fts": (ChunkText(chunk_id, "Notes", f"Ada Lovelace wrote notes in {revision}."),),
        "fact_fts": (FactText(fid, f"Ada Lovelace wrote notes on Analytical Engine {revision}"),),
        "meta": (MetaRow("graph_input_fingerprints", json.dumps([[chunk_id, revision]])),),
    }
    return tuple(replace(table, rows=rows_by_table.get(table.name, table.rows)) for table in empty_graph_tables())


def test_vec0_managed_transaction_upsert_rollback_and_knn_parity(tmp_path: Path) -> None:
    """CG-GRAPH-1: verify sqlite-vec vec0 DDL, DatabaseSink upserts/deletes/rollback, rekey_facts, and KNN parity."""
    db_path = tmp_path / "yaw.db"
    managed = open_managed_yaw_db(db_path)

    v_c1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_c1_updated = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
    v_ea = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_ea_updated = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    v_eb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v_f1 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    v_f1_updated = np.array([0.0, 0.6, 0.8, 0.0], dtype=np.float32)

    source = _DesiredGraphSource(
        _build_valid_desired_tables(
            chunk_id="chunk-1",
            chunk_vec=v_c1,
            entity_a_def="first computer programmer",
            entity_a_vec=v_ea,
            entity_b_vec=v_eb,
            fact_vec=v_f1,
            revision="rev_1",
        )
    )
    context = coco.ContextProvider()
    context.provide(GRAPH_DB, managed)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
    env = coco.Environment(settings=coco.Settings(db_path=tmp_path / "coco_state"), context_provider=context)
    app = coco.App(coco.AppConfig(name="cg_graph_1", environment=env), _publish_desired_graph, source)

    try:
        # 1. Initial publication creates 4-D vec0 virtual tables and commits all 15 tables
        app.update_blocking()
        with managed.readonly() as conn:
            for vec_table in ("chunk_vec", "entity_vec", "fact_vec"):
                ddl_row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (vec_table,),
                ).fetchone()
                assert ddl_row is not None
                assert "USING vec0" in str(ddl_row[0])
                assert vec0_table_dimension(conn, vec_table) == 4

            assert search_chunk_vectors_knn(conn, v_c1, top_k=1)[0][0] == "chunk-1"
            assert search_chunk_vectors_knn(conn, v_c1, top_k=1)[0][1] == pytest.approx(1.0, abs=1e-6)

        # 2. DatabaseSink DELETE+INSERT upsert updates vec0 embeddings in place without ON CONFLICT errors
        source.tables = _build_valid_desired_tables(
            chunk_id="chunk-1",
            chunk_vec=v_c1_updated,
            entity_a_def="first computer programmer",
            entity_a_vec=v_ea_updated,
            entity_b_vec=v_eb,
            fact_vec=v_f1_updated,
            revision="rev_2",
        )
        app.update_blocking()
        with managed.readonly() as conn:
            c_hit = search_chunk_vectors_knn(conn, v_c1_updated, top_k=1)[0]
            assert c_hit[0] == "chunk-1"
            assert c_hit[1] == pytest.approx(1.0, abs=1e-6)
            snap_committed = {
                table.name: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {table.name}"))
                for table in empty_graph_tables()
            }

        # 3. Invalid SHACL state (empty entity definition) rolls back relational, vec0, and fts5 mutations atomically
        v_bad = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        source.tables = _build_valid_desired_tables(
            chunk_id="chunk-1",
            chunk_vec=v_bad,
            entity_a_def="",  # violates ConceptShape/definition
            entity_a_vec=v_bad,
            entity_b_vec=v_bad,
            fact_vec=v_bad,
            revision="rev_bad",
        )
        with pytest.raises(ValueError, match="Graph publication rejected SHACL violations"):
            app.update_blocking()

        with managed.readonly() as conn:
            snap_after_abort = {
                table.name: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {table.name}"))
                for table in empty_graph_tables()
            }
            assert snap_after_abort == snap_committed
            assert search_chunk_vectors_knn(conn, v_c1_updated, top_k=1)[0][1] == pytest.approx(1.0, abs=1e-6)
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'rev_2'").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'rev_bad'").fetchone()[0] == 0

        # 4. Deletion to empty_graph_tables() preserves vec0 table dimension and deletes all rows cleanly
        source.tables = empty_graph_tables()
        app.update_blocking()
        with managed.readonly() as conn:
            for vec_table in ("chunk_vec", "entity_vec", "fact_vec"):
                assert conn.execute(f"SELECT COUNT(*) FROM {vec_table}").fetchone()[0] == 0
                assert vec0_table_dimension(conn, vec_table) == 4
            assert search_chunk_vectors_knn(conn, v_c1, top_k=5) == []
            assert search_entity_vectors_knn(conn, v_ea, top_k=5) == []
            assert search_fact_vectors_knn(conn, v_f1, top_k=5) == []
            changes_before = conn.total_changes

        # No-op update on empty_graph_tables executes 0 SQLite writes
        app.update_blocking()
        with managed.readonly() as conn:
            assert conn.total_changes == changes_before
    finally:
        env.close()
        managed.close()

    # 5. Verify rekey_facts on vec0 and exact cosine parity for _search_knn and find_entity_candidates
    knn_conn = open_yaw_db(tmp_path / "knn_parity.db")
    try:
        rng = np.random.default_rng(42)
        dim = 16
        raw_vecs = rng.normal(size=(8, dim)).astype(np.float32)
        unit_vecs = [v / float(np.linalg.norm(v)) for v in raw_vecs]

        chunk_items = [(f"c-{i}", unit_vecs[i]) for i in range(8)]
        # Include a duplicate key in entity_items to verify last-write-wins deduplication on vec0
        entity_items = [(f"e-{i}", unit_vecs[i]) for i in range(8)] + [("e-0", unit_vecs[0])]
        fact_items = [(f"f-{i}", unit_vecs[i]) for i in range(8)]

        upsert_chunk_vectors(knn_conn, chunk_items)
        upsert_entity_vectors(knn_conn, entity_items)
        upsert_fact_vectors(knn_conn, fact_items)

        query = rng.normal(size=(dim,)).astype(np.float32)
        query = query / float(np.linalg.norm(query))

        expected_order = sorted(
            ((i, _exact_cosine_similarity(query, unit_vecs[i])) for i in range(8)),
            key=lambda item: (-item[1], f"c-{item[0]}"),
        )[:5]

        for prefix, search_fn in (
            ("c", search_chunk_vectors_knn),
            ("e", search_entity_vectors_knn),
            ("f", search_fact_vectors_knn),
        ):
            hits = search_fn(knn_conn, query, top_k=5)
            assert [h[0] for h in hits] == [f"{prefix}-{i}" for i, _ in expected_order]
            for (_, actual_sim), (_, exp_sim) in zip(hits, expected_order, strict=True):
                assert actual_sim == pytest.approx(exp_sim, abs=1e-6)

        # Deterministic tie-breaking on identical vectors (distance == 0.0 -> sorted by id ASC)
        tie_conn = open_yaw_db(tmp_path / "tie_knn.db")
        try:
            tie_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            upsert_entity_vectors(tie_conn, [("ent-c", tie_vec), ("ent-a", tie_vec), ("ent-b", tie_vec)])
            tie_hits = search_entity_vectors_knn(tie_conn, tie_vec, top_k=3)
            assert [h[0] for h in tie_hits] == ["ent-a", "ent-b", "ent-c"]
            for _, sim in tie_hits:
                assert sim == pytest.approx(1.0, abs=1e-6)

            # Verify find_entity_candidates parity with exact cosine reference
            v_x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            v_y = np.array([0.92, float(np.sqrt(1.0 - 0.92**2)), 0.0, 0.0], dtype=np.float32)
            v_z = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            upsert_entities(tie_conn, [("x", "X", "def x"), ("y", "Y", "def y"), ("z", "Z", "def z")])
            upsert_entity_vectors(tie_conn, [("x", v_x), ("y", v_y), ("z", v_z)])
            # Clean out ent-a/b/c so only x, y, z remain
            tie_conn.execute("DELETE FROM entity_vec WHERE entity_id IN ('ent-a', 'ent-b', 'ent-c')")
            seed_candidate_vectors(tie_conn)
            cands = find_entity_candidates(tie_conn, dense_threshold=0.8, lexical_threshold=2.0)
            assert len(cands) == 1
            assert (cands[0][0], cands[0][1]) == ("x", "y")
            assert cands[0][2] == pytest.approx(_exact_cosine_similarity(v_x, v_y), abs=1e-6)

            # Verify rekey_facts on fact_vec (collision takes min(bytes(old_vec), bytes(new_vec)) via DELETE+INSERT)
            canon_fid = fact_id("x", "rel", "entity", "y")
            upsert_facts(
                tie_conn,
                [
                    ("old-fact", "x", "rel", "y", '["X","rel","Y"]', "def old"),
                    (canon_fid, "x", "rel", "y", '["X","rel","Y"]', "def canon"),
                ],
            )
            upsert_fact_sources(tie_conn, [("old-fact", "c1", 0.8, 0.8, 0.8), (canon_fid, "c2", 0.9, 0.9, 0.9)])
            vec_old = serialize_f32_vector(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
            vec_new = serialize_f32_vector(np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32))
            upsert_fact_vectors(
                tie_conn,
                [
                    ("old-fact", np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)),
                    (canon_fid, np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)),
                ],
            )
            redirects = rekey_facts(tie_conn, ["old-fact"])
            assert redirects == (("old-fact", canon_fid),)
            surviving_rows = tie_conn.execute("SELECT fact_id, embedding FROM fact_vec").fetchall()
            assert len(surviving_rows) == 1
            assert str(surviving_rows[0][0]) == canon_fid
            assert bytes(surviving_rows[0][1]) == min(vec_old, vec_new)

            # Negative boundary traps on ensure_vec0_table
            with pytest.raises(ValueError, match="Invalid SQLite identifier"):
                ensure_vec0_table(tie_conn, "bad-name", "id", 4)
            with pytest.raises(ValueError, match="Unsupported vec0 distance metric"):
                ensure_vec0_table(tie_conn, "fact_vec", "fact_id", 4, metric="l2")
        finally:
            tie_conn.close()
    finally:
        knn_conn.close()
