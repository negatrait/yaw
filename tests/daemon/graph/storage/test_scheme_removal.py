"""Real storage and retrieval invariants for the scheme-free cutover."""

import hashlib
import sqlite3
from dataclasses import replace

import cocoindex as coco
import numpy as np
import pytest

from yaw.daemon.graph.ppr import build_transition_matrix
from yaw.daemon.graph.storage import (
    load_graph_topology,
    open_managed_yaw_db,
    open_readonly_yaw_db,
    open_yaw_db,
    serialize_f32_vector,
    serialize_graph_turtle,
)
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE
from yaw.daemon.graph.storage.rows import (
    ChunkRow,
    ChunkText,
    ChunkVector,
    EdgeRow,
    EdgeSourceRow,
    EntityRow,
    EntityVector,
    LabelRow,
    declare_tables,
    empty_graph_tables,
    validate_graph_state,
)


@coco.fn
async def declare_shared_entity_graph():
    vector = serialize_f32_vector(np.array([1, 0], dtype=np.float32))
    rows = {
        "chunks": tuple(ChunkRow(cid, f"{cid}.md", 1, 1, "", 0, 0) for cid in ("c1", "c2")),
        "entities": (EntityRow("ada", "Ada", "A mathematician.", None),),
        "entity_labels": tuple(LabelRow("ada", "ada", "skos:prefLabel", "Ada", cid) for cid in ("c1", "c2")),
        "edges": tuple(EdgeRow("ada", cid, "passage", 0.9, "") for cid in ("c1", "c2")),
        "edge_sources": tuple(EdgeSourceRow("ada", cid, "passage", cid, None, None, 0.9) for cid in ("c1", "c2")),
        "chunk_vec": tuple(ChunkVector(cid, vector) for cid in ("c1", "c2")),
        "entity_vec": (EntityVector("ada", vector),),
        "chunk_fts": tuple(ChunkText(cid, "", "Ada wrote notes.") for cid in ("c1", "c2")),
    }
    await declare_tables(tuple(replace(table, rows=rows.get(table.name, table.rows)) for table in empty_graph_tables()))


def test_scheme_free_targets_keep_shared_entity_connectivity(tmp_path):
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
    env = coco.Environment(settings=coco.Settings(db_path=tmp_path / "native-state"), context_provider=context)
    try:
        app = coco.App(coco.AppConfig(name="scheme_free", environment=env), declare_shared_entity_graph)
        app.update_blocking()
        with db.readonly() as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert not {"concept_schemes", "concept_scheme_members"} & names
        with db.readonly() as conn:
            chunk_ids, entity_ids, edges, priors = load_graph_topology(conn)
            turtle = serialize_graph_turtle(conn)
        assert chunk_ids == ("c1", "c2")
        assert entity_ids == ("ada",)
        graph = build_transition_matrix(chunk_ids, entity_ids, edges, node_priors=priors)
        assert graph.transition_t.nnz == 4
        assert "skos:ConceptScheme" not in turtle
        assert "skos:topConceptOf" not in turtle
        assert "skos:inScheme" not in turtle
        assert ":ada" in turtle
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            conn.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", ("ada", "x", "skos:topConceptOf", 1.0, ""))
    finally:
        env.close()
        db.close()


@pytest.mark.parametrize("opener", [open_yaw_db, open_managed_yaw_db, open_readonly_yaw_db])
def test_retired_database_is_rejected_without_modifying_it(tmp_path, opener):
    path = tmp_path / "experiment.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE concept_schemes (scheme_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO concept_schemes VALUES ('saved-experiment')")
    before = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match="preserve this database"):
        opener(path)
    assert hashlib.sha256(path.read_bytes()).digest() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM concept_schemes").fetchall() == [("saved-experiment",)]
