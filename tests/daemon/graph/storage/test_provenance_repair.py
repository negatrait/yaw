"""Source-bound provenance repair preserves Bayesian observations and raw evidence."""

import asyncio
from contextlib import closing

import numpy as np
import pytest

from yaw.core import ChunkPointer, OpenIERecord, StructuredTriple
from yaw.daemon.graph.storage import (
    lint_graph_shacl,
    merge_entities,
    open_yaw_db,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_entities,
    upsert_entity_labels,
    upsert_fact_sources,
    upsert_facts,
)
from yaw.daemon.graph.storage.evidence import (
    bind_merged_evidence,
    build_evidence,
    evidence_schemas,
    fact_id,
    refresh_source_confidences,
    restore_asserted_fact_sources,
    upsert_evidence,
)
from yaw.daemon.graph.storage.managed_sqlite import create_table


@pytest.fixture
def evidence_db(tmp_path):
    with closing(open_yaw_db(tmp_path / "evidence.db")) as conn:
        for schema in asyncio.run(evidence_schemas()):
            create_table(conn, schema)
        upsert_entities(conn, [("entity-a", "Orion", "A service."), ("entity-b", "Anchor", "A destination.")])
        triples = (
            StructuredTriple(
                subj="Orion",
                pred="reports to",
                obj="Anchor",
                spo_def="Orion reports to Anchor.",
                subj_def="A service.",
                obj_def="A destination.",
            ),
            StructuredTriple(
                subj="Orion",
                pred="port",
                obj="8080",
                spo_def="Orion listens on port 8080.",
                subj_def="A service.",
                obj_kind="literal",
                obj_datatype="xsd:integer",
            ),
        )
        for index in range(2):
            pointer = ChunkPointer(f"chunk-{index}", "source.md", index + 1, index + 1)
            upsert_chunk_pointers(conn, [(pointer, "Orion reports to Anchor and listens on port 8080.")])
            upsert_entity_labels(
                conn,
                [
                    ("orion", "entity-a", "skos:prefLabel", "Orion", pointer.chunk_id),
                    ("anchor", "entity-b", "skos:prefLabel", "Anchor", pointer.chunk_id),
                ],
            )
            bindings = {
                (pointer.chunk_id, "orion", "a service."): ("entity-a", "A service."),
                (pointer.chunk_id, "anchor", "a destination."): ("entity-b", "A destination."),
            }
            record = OpenIERecord(
                pointer.chunk_id,
                ("Orion", "Anchor"),
                (),
                triples,
                (("Orion", "A service."), ("Anchor", "A destination.")),
            )
            mentions, assertions = build_evidence(pointer, "Source evidence.", "fixed", record, bindings)
            upsert_evidence(conn, mentions, assertions)
        upsert_facts(
            conn,
            [
                (
                    fact_id("entity-a", "reports to", "entity", "entity-b"),
                    "entity-a",
                    "reports to",
                    "entity-b",
                    '["Orion","reports to","Anchor"]',
                    "Orion reports to Anchor.",
                ),
                (
                    fact_id("entity-a", "port", "literal", "8080", "xsd:integer"),
                    "entity-a",
                    "port",
                    None,
                    '["Orion","port","8080"]',
                    "Orion listens on port 8080.",
                    "literal",
                    "8080",
                    "xsd:integer",
                ),
            ],
        )
        conn.execute("UPDATE facts SET confidence=0.91")
        yield conn


def test_complete_partial_and_duplicate_evidence_preserves_scores(evidence_db):
    conn = evidence_db
    fid = conn.execute("SELECT fact_id FROM facts ORDER BY fact_id").fetchone()[0]
    upsert_fact_sources(conn, [(fid, "chunk-0", 0.8, 0.9, 0.85)])
    conn.execute(
        "INSERT INTO fact_assertions SELECT assertion_id||'-duplicate',chunk_id,source_revision,"
        "subject_mention_id,object_mention_id,raw_json,fact_id FROM fact_assertions"
    )
    raw = [tuple(row) for row in conn.execute("SELECT * FROM fact_assertions ORDER BY assertion_id")]
    assert any(v.shape == "ProvenanceShape/factSources" for v in lint_graph_shacl(conn))
    assert restore_asserted_fact_sources(conn) == 3
    assert lint_graph_shacl(conn) == []
    assert tuple(
        conn.execute(
            "SELECT llm_prob,vec_coherence,atomic_confidence FROM fact_sources WHERE fact_id=? AND chunk_id='chunk-0'",
            (fid,),
        ).fetchone()
    ) == (0.8, 0.9, 0.85)
    assert conn.execute("SELECT COUNT(*) FROM fact_sources WHERE atomic_confidence IS NULL").fetchone()[0] == 3
    assert [row[0] for row in conn.execute("SELECT confidence FROM facts")] == [0.91, 0.91]
    assert [tuple(row) for row in conn.execute("SELECT * FROM fact_assertions ORDER BY assertion_id")] == raw
    assert restore_asserted_fact_sources(conn) == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "UPDATE fact_assertions SET fact_id=NULL",
        "DELETE FROM chunks",
        "DELETE FROM entity_mentions",
        "UPDATE entity_mentions SET source_revision='other'",
        "UPDATE entity_mentions SET chunk_id='other'",
        "UPDATE entity_mentions SET entity_id='other'",
        "UPDATE fact_assertions SET subject_mention_id='missing'",
        "UPDATE fact_assertions SET object_mention_id='missing'",
    ],
)
def test_unsupported_provenance_stays_unresolved(evidence_db, corruption):
    evidence_db.execute(corruption)
    assert restore_asserted_fact_sources(evidence_db) == 0
    assert sum(v.shape == "ProvenanceShape/factSources" for v in lint_graph_shacl(evidence_db)) == 2


def test_rebound_assertions_restore_current_fact_ids(evidence_db):
    conn = evidence_db
    upsert_entities(conn, [("entity-0", "Orion service", "The same service.")])
    bind_merged_evidence(conn, merge_entities(conn, "entity-a", "entity-0"))
    assert restore_asserted_fact_sources(conn) == 4
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM fact_sources fs JOIN facts f USING(fact_id) WHERE f.subj_id='entity-0'"
        ).fetchone()[0]
        == 4
    )
    assert restore_asserted_fact_sources(conn) == 0


@pytest.mark.parametrize("invalid", ["revision", "chunk", "subject", "object", "missing-object"])
def test_invalid_assertion_cannot_overwrite_supported_confidence(evidence_db, invalid):
    conn = evidence_db
    vector = np.array([1.0, 0.0], dtype=np.float32)
    upsert_chunk_vectors(conn, [("chunk-0", vector), ("chunk-1", vector)])
    conn.executemany(
        "INSERT INTO entity_candidate_vectors VALUES (?,?,?,?,?)",
        [
            (sid, eid, name, definition, vector.tobytes())
            for sid, eid, name, definition in conn.execute(
                "SELECT DISTINCT sense_id,entity_id,name,definition FROM entity_mentions"
            )
        ],
    )
    conn.execute("UPDATE fact_assertions SET raw_json=json_set(raw_json,'$.llm_prob',0.16)")
    restore_asserted_fact_sources(conn)
    refresh_source_confidences(conn)
    expected = [tuple(row) for row in conn.execute("SELECT * FROM fact_sources ORDER BY fact_id,chunk_id")]
    assertion = conn.execute(
        "SELECT * FROM fact_assertions WHERE object_mention_id IS NOT NULL AND chunk_id='chunk-0' LIMIT 1"
    ).fetchone()
    values = list(assertion)
    values[0] += "-invalid"
    if invalid == "revision":
        values[2] = "other"
    elif invalid == "chunk":
        values[1] = "chunk-1"
    elif invalid == "subject":
        values[3] = values[4]
    elif invalid == "object":
        values[4] = values[3]
    else:
        values[4] = "missing"
    conn.execute("INSERT INTO fact_assertions VALUES (?,?,?,?,?,?,?)", values)
    conn.execute(
        "UPDATE fact_assertions SET raw_json=json_set(raw_json,'$.llm_prob',0.99) WHERE assertion_id=?", (values[0],)
    )
    refresh_source_confidences(conn)
    assert [tuple(row) for row in conn.execute("SELECT * FROM fact_sources ORDER BY fact_id,chunk_id")] == expected
