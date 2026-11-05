"""Real storage and retrieval invariants for the scheme-free cutover."""

import hashlib
import sqlite3

import numpy as np
import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.ppr import build_transition_matrix
from yaw.daemon.graph.storage import (
    PreparedGraphRows,
    apply_prepared_graph_rows,
    load_graph_topology,
    open_managed_yaw_db,
    open_readonly_yaw_db,
    open_yaw_db,
    serialize_graph_turtle,
)


def test_scheme_free_targets_keep_shared_entity_connectivity(tmp_path):
    conn = open_yaw_db(tmp_path / "yaw.db")
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert not {"concept_schemes", "concept_scheme_members"} & names
        vector = np.array([1, 0], dtype=np.float32)
        rows = PreparedGraphRows(
            chunks=tuple((ChunkPointer(cid, f"{cid}.md", 1, 1), "Ada wrote notes.") for cid in ("c1", "c2")),
            entities=(("ada", "Ada", "A mathematician."),),
            entity_labels=tuple(("ada", "ada", "skos:prefLabel", "Ada", cid) for cid in ("c1", "c2")),
            facts=(),
            fact_sources=(),
            edges=tuple(("ada", cid, "passage", 0.9, cid) for cid in ("c1", "c2")),
            chunk_vectors=(("c1", vector), ("c2", vector)),
            entity_vectors=(("ada", vector),),
            fact_vectors=(),
        )
        with conn:
            conn.execute("BEGIN")
            apply_prepared_graph_rows(conn, rows)
        chunk_ids, entity_ids, edges, priors = load_graph_topology(conn)
        assert chunk_ids == ("c1", "c2")
        assert entity_ids == ("ada",)
        graph = build_transition_matrix(chunk_ids, entity_ids, edges, node_priors=priors)
        assert graph.transition_t.nnz == 4
        turtle = serialize_graph_turtle(conn)
        assert "skos:ConceptScheme" not in turtle
        assert "skos:topConceptOf" not in turtle
        assert "skos:inScheme" not in turtle
        assert ":ada" in turtle
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", ("ada", "x", "skos:topConceptOf", 1.0, ""))
    finally:
        conn.close()


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
