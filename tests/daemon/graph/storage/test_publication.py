"""Prepared graph publication is atomic across relational, vector and FTS rows."""

import json
from dataclasses import replace
from pathlib import Path

import cocoindex as coco
import numpy as np
import pytest
from daemon.graph.test_indexing_overlap import provider as provider

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import PreparedGraphRows, apply_prepared_graph_rows, open_managed_yaw_db
from yaw.daemon.graph.storage.publication import GRAPH_DB, GraphAction, GraphTargetHandler, declare_graph


@coco.fn
async def graph_fixture(rows: PreparedGraphRows | None) -> None:
    if rows is not None:
        declare_graph(rows)


def test_graph_sink_rejects_invalid_replacement_atomically(tmp_path):
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    rows = PreparedGraphRows(
        chunks=((ChunkPointer("c", "source.md", 1, 1), "Valid evidence"),),
        entities=(("e", "Entity", "A grounded definition"),),
        entity_labels=(("entity", "e", "skos:prefLabel", "Entity", "c"),),
        facts=(),
        fact_sources=(),
        edges=(("e", "c", "passage", 1.0, "c"),),
        chunk_vectors=(("c", np.array([1, 0], dtype=np.float32)),),
        entity_vectors=(),
        fact_vectors=(),
        input_fingerprints=(("c", "committed"),),
    )
    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    handler = GraphTargetHandler()
    try:
        handler.apply_actions(context, [GraphAction(rows, (), ())])
        invalid = replace(rows, entities=(("e", "Entity", ""),), input_fingerprints=(("c", "uncommitted"),))
        with pytest.raises(ValueError, match="Graph publication rejected"):
            handler.apply_actions(context, [GraphAction(invalid, ("c",), ())])
        with db.readonly() as conn:
            assert (
                conn.execute("SELECT definition FROM entities WHERE entity_id='e'").fetchone()[0]
                == "A grounded definition"
            )
            assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 1
            assert json.loads(
                conn.execute("SELECT value FROM meta WHERE key='graph_input_fingerprints'").fetchone()[0]
            ) == [["c", "committed"]]
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'Valid'").fetchone()[0] == 1
        handler.apply_actions(context, [GraphAction(rows, ("c",), ())])
    finally:
        db.close()


@pytest.mark.parametrize("shared_source", [False, True])
def test_native_target_withdraws_global_assertion_without_removing_sources(tmp_path, shared_source):
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    rows = PreparedGraphRows(
        chunks=((ChunkPointer("c", "source.md", 1, 1), "Evidence"),),
        entities=(("a", "A", "First entity"), ("b", "B", "Second entity")),
        entity_labels=(("a", "a", "skos:prefLabel", "A", "c"), ("b", "b", "skos:prefLabel", "B", "c")),
        facts=(),
        fact_sources=(),
        edges=(("a", "b", "skos:related", 1.0, "c"), ("b", "a", "skos:related", 1.0, "c")) if shared_source else (),
        chunk_vectors=(),
        entity_vectors=(),
        fact_vectors=(),
    )
    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    try:
        for desired in (
            replace(
                rows, edges=(*rows.edges, ("a", "b", "skos:related", 1.0, ""), ("b", "a", "skos:related", 1.0, ""))
            ),
            rows,
        ):
            env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
            try:
                app = coco.App(coco.AppConfig(name="global_edges", environment=env), graph_fixture, desired)
                app.update_blocking()
                app.update_blocking()
            finally:
                env.close()
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2 * int(shared_source)
            assert conn.execute("SELECT COUNT(*) FROM edge_sources WHERE chunk_id=''").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM edge_sources WHERE chunk_id='c'").fetchone()[0] == 2 * int(
                shared_source
            )
    finally:
        db.close()


def test_native_graph_target_reconciles_replacement_and_removal(tmp_path: Path) -> None:
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    try:
        for chunk_id, entity_id in (("first", "old"), ("first", "new"), ("second", "new"), (None, None)):
            rows = (
                None
                if chunk_id is None
                else PreparedGraphRows(
                    chunks=((ChunkPointer(chunk_id, "source.md", 1, 1), "Grounded evidence"),),
                    entities=((entity_id, entity_id, "Grounded entity definition"),),
                    entity_labels=((entity_id, entity_id, "skos:prefLabel", entity_id, chunk_id),),
                    facts=(),
                    fact_sources=(),
                    edges=((entity_id, chunk_id, "passage", 1.0, chunk_id),),
                    chunk_vectors=((chunk_id, np.array([1, 0], dtype=np.float32)),),
                    entity_vectors=(),
                    fact_vectors=(),
                )
            )
            context = coco.ContextProvider()
            context.provide(GRAPH_DB, db)
            env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
            app = coco.App(coco.AppConfig(name="graph", environment=env), graph_fixture, rows)
            try:
                app.update_blocking()
                app.update_blocking()
                with db.readonly() as conn:
                    expected = [] if chunk_id is None else [(chunk_id,)]
                    assert list(map(tuple, conn.execute("SELECT chunk_id FROM chunks"))) == expected
                    assert list(map(tuple, conn.execute("SELECT chunk_id FROM chunk_vec"))) == expected
                    assert conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == len(expected)
                    expected_entities = [] if entity_id is None else [(entity_id,)]
                    assert list(map(tuple, conn.execute("SELECT entity_id FROM entities"))) == expected_entities
            finally:
                env.close()
    finally:
        db.close()


def test_publication_requires_transaction_and_rolls_back_all_tables(tmp_path: Path) -> None:
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    rows = PreparedGraphRows(
        chunks=((ChunkPointer("c", "source.md", 1, 1, "Algebra"), "Ada studies algebra."),),
        entities=(("e", "Ada", "A mathematician"),),
        entity_labels=(("ada", "e", "skos:prefLabel", "Ada", "c"),),
        facts=(),
        fact_sources=(),
        edges=(("e", "c", "passage", 1.0, "c"),),
        chunk_vectors=(("c", np.array([1, 0], dtype=np.float32)),),
        entity_vectors=(("e", np.array([0, 1], dtype=np.float32)),),
        fact_vectors=(),
    )
    tables = ("chunks", "entities", "entity_labels", "edges", "edge_sources", "chunk_vec", "entity_vec", "chunk_fts")
    try:
        with db.readonly() as conn, pytest.raises(RuntimeError, match="active transaction"):
            apply_prepared_graph_rows(conn, rows)
        with pytest.raises(RuntimeError, match="abort publication"), db.transaction() as conn:
            apply_prepared_graph_rows(conn, rows)
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'algebra'").fetchone()[0] == 1
            raise RuntimeError("abort publication")
        with db.readonly() as conn:
            assert {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables
            } == dict.fromkeys(tables, 0)
        with db.transaction() as conn:
            apply_prepared_graph_rows(conn, rows)
        with db.readonly() as conn:
            before = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}
        with db.transaction() as conn:
            apply_prepared_graph_rows(conn, rows)
        with db.readonly() as conn:
            assert {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables} == before
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'algebra'").fetchone()[0] == 1
    finally:
        db.close()


def test_plan_graph_publication_keeps_yaw_db_outside_transaction_and_preserves_shared_elements(
    tmp_path: Path, provider
) -> None:
    import asyncio

    from yaw.core import OpenIERecord
    from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, YawProgram, create_dspy_lm
    from yaw.daemon.graph.inference.components import (
        EMBEDDING_ENDPOINT,
        OPENIE_ENDPOINT,
        EmbeddingEndpoint,
        OpenIEEndpoint,
    )
    from yaw.daemon.graph.planner import (
        GraphPlanSettings,
        plan_graph_publication,
    )

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.md").write_text("# Section A\nAda Lovelace wrote the analytical engine notes.\n", encoding="utf-8")
    (corpus_dir / "b.md").write_text("# Section B\nAda Lovelace collaborated with Charles Babbage.\n", encoding="utf-8")

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    in_tx_observations: list[bool] = []

    def fake_embed_fn(texts: list[str], **kwargs: object) -> np.ndarray:
        in_tx_observations.append(db._conn.in_transaction)
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for idx, text in enumerate(texts):
            out[idx, idx % 768] = 1.0
            out[idx, (sum(ord(ch) for ch in text) % 768)] += 0.5
        return out

    adapter = DSPyEmbeddingAdapter(embedder=fake_embed_fn)

    url, state = provider
    state["delay"] = 0.01
    lm = create_dspy_lm("publication", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)

    chunk_a = (
        ChunkPointer("chunk_a", "a.md", 1, 2, "Section A"),
        OpenIERecord(
            chunk_id="chunk_a",
            entities=[
                ("Ada Lovelace", "first computer programmer"),
                ("Analytical Engine", "mechanical general-purpose computer"),
            ],
            triples=[("Ada Lovelace", "wrote notes on", "Analytical Engine")],
        ),
    )
    chunk_b = (
        ChunkPointer("chunk_b", "b.md", 1, 2, "Section B"),
        OpenIERecord(
            chunk_id="chunk_b",
            entities=[
                ("Ada Lovelace", "first computer programmer"),
                ("Analytical Engine", "mechanical general-purpose computer"),
            ],
            triples=[("Ada Lovelace", "wrote notes on", "Analytical Engine")],
        ),
    )

    @coco.fn
    async def planned_graph_app(
        settings: GraphPlanSettings,
        chunk_rows: tuple[tuple[ChunkPointer, OpenIERecord], ...],
    ) -> None:
        plan = await plan_graph_publication(settings, chunk_rows)
        declare_graph(plan.rows)

    settings = GraphPlanSettings(
        corpus_root=corpus_dir,
        program_identity="prog_v1",
        embedding_identity=adapter.profile.index_identity,
        synonymy_top_k=0,
    )
    try:
        for active_rows in ((chunk_a, chunk_b), (chunk_a,)):
            context = coco.ContextProvider()
            context.provide(GRAPH_DB, db)
            context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(adapter, asyncio.Semaphore(4)))
            context.provide(
                OPENIE_ENDPOINT,
                OpenIEEndpoint(program, lm, "prog_v1", asyncio.Semaphore(1)),
            )
            env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "plan_state"), context_provider=context)
            app = coco.App(
                coco.AppConfig(name="planned_graph", environment=env),
                planned_graph_app,
                settings,
                active_rows,
            )
            try:
                app.update_blocking()
            finally:
                env.close()

        assert in_tx_observations and not any(in_tx_observations)
        with db.readonly() as conn:
            assert [r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall()] == ["chunk_a"]
            assert [r[0] for r in conn.execute("SELECT chunk_id FROM chunk_vec").fetchall()] == ["chunk_a"]
            assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
            assert (
                conn.execute("SELECT COUNT(*) FROM entity_vec WHERE entity_id NOT LIKE 'scheme-%'").fetchone()[0] == 2
            )
            assert conn.execute("SELECT COUNT(*) FROM entity_vec").fetchone()[0] == 3
            assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM fact_vec").fetchone()[0] == 1
            assert [r[0] for r in conn.execute("SELECT chunk_id FROM fact_sources").fetchall()] == ["chunk_a"]
    finally:
        db.close()
