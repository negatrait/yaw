"""Native typed graph targets publish and reconcile one coherent state."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cocoindex as coco
import numpy as np
import pytest
from daemon.graph.test_indexing_overlap import provider as provider

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import open_managed_yaw_db, serialize_f32_vector
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE
from yaw.daemon.graph.storage.rows import (
    ChunkRow,
    ChunkText,
    ChunkVector,
    DesiredTable,
    EdgeRow,
    EdgeSourceRow,
    EntityRow,
    LabelRow,
    MetaRow,
    declare_tables,
    empty_graph_tables,
    validate_graph_state,
)


def desired_graph(
    chunk="c", entity="e", text="Valid evidence", definition="A grounded definition", revision="committed"
):
    rows = {
        "chunks": (ChunkRow(chunk, "source.md", 1, 1, "Evidence", 0, 0),),
        "entities": (EntityRow(entity, "Entity", definition, None),),
        "entity_labels": (LabelRow("entity", entity, "skos:prefLabel", "Entity", chunk),),
        "edges": (EdgeRow(entity, chunk, "passage", 1.0, ""),),
        "edge_sources": (EdgeSourceRow(entity, chunk, "passage", chunk, None, None, 1.0),),
        "chunk_vec": (ChunkVector(chunk, serialize_f32_vector(np.array([1, 0], dtype=np.float32))),),
        "chunk_fts": (ChunkText(chunk, "Evidence", text),),
        "meta": (MetaRow("graph_input_fingerprints", json.dumps([[chunk, revision]])),),
    }
    return tuple(replace(table, rows=rows.get(table.name, table.rows)) for table in empty_graph_tables())


@dataclass
class DesiredGraph:
    tables: tuple[DesiredTable, ...]


@coco.fn
async def graph_fixture(source: DesiredGraph) -> None:
    await declare_tables(source.tables)


@pytest.fixture
def graph_db(tmp_path):
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def publication(tmp_path, graph_db):
    source = DesiredGraph(desired_graph())
    context = coco.ContextProvider()
    context.provide(GRAPH_DB, graph_db)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
    env = coco.Environment(settings=coco.Settings(db_path=tmp_path / "state"), context_provider=context)
    app = coco.App(coco.AppConfig(name="graph", environment=env), graph_fixture, source)
    try:
        yield source, app
    finally:
        env.close()


def graph_snapshot(db):
    with db.readonly() as conn:
        return {
            table.name: sorted(tuple(row) for row in conn.execute(f"SELECT * FROM {table.name}"))
            for table in empty_graph_tables()
        }


def test_graph_sink_rejects_invalid_replacement_atomically(graph_db, publication):
    source, app = publication
    app.update_blocking()
    before = graph_snapshot(graph_db)
    source.tables = desired_graph(chunk="replacement", text="Rejected replacement", definition="", revision="rejected")
    with pytest.raises(ValueError, match="Graph publication rejected"):
        app.update_blocking()
    assert graph_snapshot(graph_db) == before
    with graph_db.readonly() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'Valid'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'Rejected'").fetchone()[0] == 0
    source.tables = desired_graph(chunk="replacement", text="Accepted replacement", revision="accepted")
    app.update_blocking()
    with graph_db.readonly() as conn:
        assert json.loads(
            conn.execute("SELECT value FROM meta WHERE key='graph_input_fingerprints'").fetchone()[0]
        ) == [["replacement", "accepted"]]
        assert conn.execute("SELECT chunk_id FROM chunk_vec").fetchone()[0] == "replacement"
        assert conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH 'Accepted'").fetchone()[0] == 1


@pytest.mark.parametrize("shared_source", [False, True])
def test_native_target_withdraws_global_assertion_without_removing_sources(graph_db, publication, shared_source):
    source, app = publication
    for has_global in (True, False):
        edges = (EdgeRow("a", "b", "skos:related", 1.0, ""), EdgeRow("b", "a", "skos:related", 1.0, ""))
        rows = {
            "entities": (EntityRow("a", "A", "First entity", None), EntityRow("b", "B", "Second entity", None)),
            "entity_labels": (
                LabelRow("a", "a", "skos:prefLabel", "A", "c"),
                LabelRow("b", "b", "skos:prefLabel", "B", "c"),
            ),
            "edges": edges if has_global or shared_source else (),
            "edge_sources": tuple(
                EdgeSourceRow(edge.src_id, edge.dst_id, edge.kind, chunk, None, None, 1.0)
                for edge in edges
                for chunk in (("",) if has_global else ()) + (("c",) if shared_source else ())
            ),
        }
        source.tables = tuple(replace(table, rows=rows.get(table.name, table.rows)) for table in desired_graph())
        app.update_blocking()
    with graph_db.readonly() as conn:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 2 * int(shared_source)
        assert conn.execute("SELECT COUNT(*) FROM edge_sources WHERE chunk_id=''").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM edge_sources WHERE chunk_id='c'").fetchone()[0] == 2 * int(
            shared_source
        )


def test_native_graph_target_reconciles_replacement_removal_and_noop(graph_db, publication):
    source, app = publication
    for chunk_id, entity_id in (("first", "old"), ("first", "new"), ("second", "new"), (None, None)):
        source.tables = empty_graph_tables() if chunk_id is None else desired_graph(chunk_id, entity_id)
        app.update_blocking()
        before = graph_snapshot(graph_db)
        with graph_db.readonly() as conn:
            changes = conn.total_changes
        app.update_blocking()
        assert graph_snapshot(graph_db) == before
        with graph_db.readonly() as conn:
            assert conn.total_changes == changes
            expected = [] if chunk_id is None else [(chunk_id,)]
            assert list(map(tuple, conn.execute("SELECT chunk_id FROM chunks"))) == expected
            assert list(map(tuple, conn.execute("SELECT chunk_id FROM chunk_vec"))) == expected
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == len(expected)
            expected_entities = [] if entity_id is None else [(entity_id,)]
            assert list(map(tuple, conn.execute("SELECT entity_id FROM entities"))) == expected_entities


def test_plan_graph_publication_keeps_yaw_db_outside_transaction_and_preserves_shared_elements(
    tmp_path: Path, provider
) -> None:
    import asyncio
    import hashlib
    import json
    from dataclasses import asdict

    from yaw.core import OpenIERecord, StructuredTriple
    from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, YawProgram, create_dspy_lm
    from yaw.daemon.graph.inference.components import (
        EMBEDDING_ENDPOINT,
        OPENIE_ENDPOINT,
        ChunkInference,
        ChunkProcessOutput,
        EmbeddingEndpoint,
        OpenIEEndpoint,
    )
    from yaw.daemon.graph.planner import (
        GraphPlanSettings,
        plan_graph_publication,
    )

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    text_a = "# Section A\nAda Lovelace wrote the analytical engine notes."
    text_b = "# Section B\nAda Lovelace wrote the analytical engine notes."
    (corpus_dir / "a.md").write_text(text_a + "\n", encoding="utf-8")
    (corpus_dir / "b.md").write_text(text_b + "\n", encoding="utf-8")

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    in_tx_observations: list[bool] = []

    def fake_embed_fn(texts: list[str], **kwargs: object) -> np.ndarray:
        with db.readonly() as conn:
            in_tx_observations.append(conn.in_transaction)
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for idx, text in enumerate(texts):
            out[idx, idx % 768] = 1.0
            out[idx, (sum(ord(ch) for ch in text) % 768)] += 0.5
        return out

    adapter = DSPyEmbeddingAdapter(embedder=fake_embed_fn)

    url, state = provider
    state["delay"] = 0.01
    lm = create_dspy_lm("publication", base_url=url + "/v1", cache=False, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)

    rec_a = OpenIERecord(
        chunk_id="chunk_a",
        entities=("Ada Lovelace", "Analytical Engine"),
        entity_definitions=(
            ("Ada Lovelace", "first computer programmer"),
            ("Analytical Engine", "mechanical general-purpose computer"),
        ),
        triples=(("Ada Lovelace", "wrote notes on", "Analytical Engine"),),
        structured_triples=(
            StructuredTriple(
                subj="Ada Lovelace",
                subj_def="first computer programmer",
                pred="wrote notes on",
                obj="Analytical Engine",
                obj_def="mechanical general-purpose computer",
                spo_def="Ada Lovelace wrote the analytical engine notes.",
            ),
        ),
    )
    rec_b = OpenIERecord(
        chunk_id="chunk_b",
        entities=("Ada Lovelace", "Analytical Engine"),
        entity_definitions=(
            ("Ada Lovelace", "first computer programmer"),
            ("Analytical Engine", "mechanical general-purpose computer"),
        ),
        triples=(("Ada Lovelace", "wrote notes on", "Analytical Engine"),),
        structured_triples=(
            StructuredTriple(
                subj="Ada Lovelace",
                subj_def="first computer programmer",
                pred="wrote notes on",
                obj="Analytical Engine",
                obj_def="mechanical general-purpose computer",
                spo_def="Ada Lovelace wrote the analytical engine notes.",
            ),
        ),
    )
    vec_bytes = np.ones((768,), dtype="<f4").tobytes()
    chunk_a = ChunkProcessOutput(
        pointer=ChunkPointer("chunk_a", "a.md", 1, 2, "Section A"),
        text=text_a,
        inference=ChunkInference(
            chunk_id="chunk_a",
            source_hash=hashlib.sha256(text_a.encode()).hexdigest(),
            program_identity="prog_v1",
            embedding_identity=adapter.profile.index_identity,
            extraction_json=json.dumps(asdict(rec_a)),
            traces_json="[]",
        ),
        embedding=vec_bytes,
    )
    chunk_b = ChunkProcessOutput(
        pointer=ChunkPointer("chunk_b", "b.md", 1, 2, "Section B"),
        text=text_b,
        inference=ChunkInference(
            chunk_id="chunk_b",
            source_hash=hashlib.sha256(text_b.encode()).hexdigest(),
            program_identity="prog_v1",
            embedding_identity=adapter.profile.index_identity,
            extraction_json=json.dumps(asdict(rec_b)),
            traces_json="[]",
        ),
        embedding=vec_bytes,
    )

    @coco.fn
    async def planned_graph_app(
        settings: GraphPlanSettings,
        chunk_outputs: tuple[ChunkProcessOutput, ...],
    ) -> None:
        plan = await plan_graph_publication(settings, chunk_outputs)
        await declare_tables(plan.tables)

    settings = GraphPlanSettings(
        corpus_root=corpus_dir,
        program_identity="prog_v1",
        embedding_identity=adapter.profile.index_identity,
        candidate_dense_threshold=2.0,
        candidate_lexical_threshold=2.0,
    )
    try:
        for active_rows in ((chunk_a, chunk_b), (chunk_a,)):
            context = coco.ContextProvider()
            context.provide(GRAPH_DB, db)
            context.provide(VALIDATE_SQLITE, validate_graph_state)
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
            assert conn.execute("SELECT COUNT(*) FROM entity_vec").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM fact_vec").fetchone()[0] == 1
            assert [r[0] for r in conn.execute("SELECT chunk_id FROM fact_sources").fetchall()] == ["chunk_a"]
            assert conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == 1
    finally:
        db.close()


def test_delta_declare_tables_preserves_survivors_and_applies_deletions(tmp_path: Path) -> None:
    from yaw.daemon.graph.storage.rows import DesiredTable, MetaRow

    db = open_managed_yaw_db(tmp_path / "delta_yaw.db")

    class StateHolder:
        tables: tuple[DesiredTable, ...] = ()

    holder = StateHolder()

    @coco.fn
    async def delta_app(h: StateHolder) -> None:
        await declare_tables(h.tables)

    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, lambda _conn: None)
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "delta_state"), context_provider=context)
    app = coco.App(coco.AppConfig(name="delta_test", environment=env), delta_app, holder)
    try:
        # Run 1 (is_delta=False): declare k1 and k2
        holder.tables = (
            DesiredTable(
                "meta",
                MetaRow,
                ("key",),
                (MetaRow("k1", "v1"), MetaRow("k2", "v2")),
                is_delta=False,
            ),
        )
        app.update_blocking()
        with db.readonly() as conn:
            rows = dict(conn.execute("SELECT key, value FROM meta WHERE key IN ('k1', 'k2', 'k3')").fetchall())
            assert rows == {"k1": "v1", "k2": "v2"}

        # Run 2 (is_delta=True): omit k1 (survivor), delete k2, insert k3
        holder.tables = (
            DesiredTable(
                "meta",
                MetaRow,
                ("key",),
                (MetaRow("k3", "v3"),),
                deleted_keys=(("k2",),),
                is_delta=True,
            ),
        )
        app.update_blocking()
        with db.readonly() as conn:
            rows = dict(conn.execute("SELECT key, value FROM meta WHERE key IN ('k1', 'k2', 'k3')").fetchall())
            assert rows == {"k1": "v1", "k3": "v3"}
    finally:
        env.close()
        db.close()
