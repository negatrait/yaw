"""Canonical identities converge without rewriting source-owned extraction evidence."""

import asyncio
import json
import struct
from contextlib import closing
from itertools import permutations

import pytest

from yaw.core import ChunkPointer, OpenIERecord, StructuredTriple
from yaw.daemon.graph.planner import materialize_entities_and_facts
from yaw.daemon.graph.storage import (
    merge_entities,
    open_yaw_db,
    upsert_chunk_pointers,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_fact_sources,
    upsert_facts,
)
from yaw.daemon.graph.storage.evidence import (
    bind_merged_evidence,
    build_evidence,
    canonicalize_fact_text,
    evidence_schemas,
    fact_id,
    sense_id,
    upsert_evidence,
)
from yaw.daemon.graph.storage.managed_sqlite import create_table


def snapshot(conn):
    tables = (
        "entities",
        "entity_labels",
        "facts",
        "fact_sources",
        "edges",
        "edge_sources",
        "entity_mentions",
        "fact_assertions",
        "fact_vec",
    )
    return {table: sorted(tuple(row) for row in conn.execute(f"SELECT * FROM {table}")) for table in tables}


def test_incremental_alias_permutations_preserve_raw_evidence_and_canonical_fact_ids(tmp_path):
    definitions = ("An Orion service.", "The Orion service for telemetry.", "Orion collects telemetry events.")
    expected_id = min(sense_id("Orion", definition) for definition in definitions)
    expected = None
    schemas = asyncio.run(evidence_schemas())
    for order in permutations(range(3)):
        for reverse in (False, True):
            with closing(open_yaw_db(tmp_path / f"{'-'.join(map(str, order))}-{reverse}.db")) as conn:
                for schema in schemas:
                    create_table(conn, schema)
                canonical = None
                raw_mentions, raw_assertions = {}, {}
                for index in order:
                    definition = definitions[index]
                    pointer = ChunkPointer(f"chunk-{index}", "document.md", index + 1, index + 1, "Document")
                    triples = (
                        StructuredTriple(
                            subj="Orion",
                            pred="Released On",
                            obj="2026-11-05",
                            obj_kind="literal",
                            obj_datatype="xsd:date",
                            subj_def=definition,
                            spo_def="Release date.",
                        ),
                        StructuredTriple(
                            subj="Orion",
                            pred="display Name",
                            obj="Orion",
                            obj_kind="literal",
                            obj_datatype="rdf:langString",
                            obj_lang="en",
                            subj_def=definition,
                            spo_def="English display name.",
                        ),
                        StructuredTriple(
                            subj="Orion",
                            pred="reports To",
                            obj="Anchor",
                            subj_def=definition,
                            obj_def="The Anchor service.",
                            spo_def="Orion reports to Anchor.",
                        ),
                        StructuredTriple(
                            subj="Anchor",
                            pred="receives From",
                            obj="Orion",
                            subj_def="The Anchor service.",
                            obj_def=definition,
                            spo_def="Anchor receives from Orion.",
                        ),
                    )
                    record = OpenIERecord(
                        pointer.chunk_id,
                        ("Orion", "Anchor"),
                        (),
                        triples,
                        (("Orion", definition), ("Anchor", "The Anchor service.")),
                    )
                    batch = materialize_entities_and_facts(
                        [(pointer.chunk_id, list(triples), list(record.entity_definitions))],
                        {"orion": "Orion", "anchor": "Anchor"},
                        {(pointer.chunk_id, "orion"): "Orion", (pointer.chunk_id, "anchor"): "Anchor"},
                        {},
                    )
                    text = f"Evidence {index}"
                    upsert_chunk_pointers(conn, [(pointer, text)])
                    upsert_entities(conn, batch.entity_nodes)
                    upsert_entity_labels(conn, batch.entity_labels_to_upsert)
                    upsert_facts(
                        conn,
                        [
                            (fid, sid, pred, oid, raw, definition, kind, lexical, datatype, language)
                            for (
                                fid,
                                sid,
                                pred,
                                oid,
                                raw,
                                definition,
                                _sd,
                                _od,
                                kind,
                                lexical,
                                datatype,
                                language,
                            ) in batch.fact_nodes
                        ],
                    )
                    upsert_fact_sources(
                        conn, [(fid, pointer.chunk_id, 0.8, 0.9, 0.85) for fid, _ in batch.fact_sources_batch]
                    )
                    conn.executemany(
                        "INSERT OR REPLACE INTO fact_vec VALUES (?,?)",
                        ((row[0], struct.pack("ff", 1.0, 0.0)) for row in batch.fact_nodes),
                    )
                    upsert_edges_with_sources(conn, batch.edges_with_sources)
                    mentions, assertions = build_evidence(pointer, text, "fixed-program", record, batch.bindings)
                    upsert_evidence(conn, mentions, assertions)
                    for mention in mentions:
                        raw_mentions[mention.mention_id] = tuple(
                            conn.execute(
                                "SELECT chunk_id,source_revision,sense_id,name,definition,resolved_definition "
                                "FROM entity_mentions WHERE mention_id=?",
                                (mention.mention_id,),
                            ).fetchone()
                        )
                    for assertion in assertions:
                        raw_assertions[assertion.assertion_id] = assertion.raw_json
                    incoming = sense_id("Orion", definitions[index])
                    if canonical is not None:
                        src, dst = (incoming, canonical) if reverse else (canonical, incoming)
                        merged = merge_entities(conn, src, dst)
                        bind_merged_evidence(conn, merged)
                        canonical = merged.canonical_id
                    else:
                        canonical = incoming

                canonicalize_fact_text(conn)
                assert canonical == expected_id
                facts = conn.execute(
                    "SELECT fact_id,subj_id,pred,obj_kind,obj_id,obj_literal_lex,obj_datatype,obj_lang FROM facts"
                ).fetchall()
                assert len(facts) == 4
                for fid, subject, predicate, kind, obj, lexical, datatype, language in facts:
                    assert fid == fact_id(
                        subject, predicate, kind, obj if kind == "entity" else lexical, datatype, language
                    )
                assert conn.execute("SELECT COUNT(*) FROM fact_sources").fetchone()[0] == 12
                assert dict(conn.execute("SELECT fact_id,embedding FROM fact_vec")) == {
                    row[0]: struct.pack("ff", 1.0, 0.0) for row in facts
                }
                assert all(
                    tuple(row) == (0.8, 0.9, 0.85)
                    for row in conn.execute("SELECT llm_prob,vec_coherence,atomic_confidence FROM fact_sources")
                )
                assert dict(conn.execute("SELECT assertion_id,raw_json FROM fact_assertions")) == raw_assertions
                assert {
                    row[0]: tuple(row[1:])
                    for row in conn.execute(
                        "SELECT mention_id,chunk_id,source_revision,sense_id,name,definition,resolved_definition "
                        "FROM entity_mentions"
                    )
                } == raw_mentions
                assert (
                    conn.execute(
                        "SELECT COUNT(*) FROM fact_assertions a LEFT JOIN facts f ON f.fact_id=a.fact_id "
                        "WHERE f.fact_id IS NULL"
                    ).fetchone()[0]
                    == 0
                )
                assert conn.execute("SELECT COUNT(*) FROM fact_fts").fetchone()[0] == 4
                state = snapshot(conn)
                if expected is None:
                    expected = state
                assert state == expected


def test_merge_prunes_collapsed_fact_but_retains_raw_assertion(tmp_path):
    with closing(open_yaw_db(tmp_path / "collapsed.db")) as conn:
        for schema in asyncio.run(evidence_schemas()):
            create_table(conn, schema)
        upsert_entities(conn, [("entity-a", "A", "First sense"), ("entity-b", "B", "Second sense")])
        fid = fact_id("entity-b", "same As", "entity", "entity-a")
        raw = json.dumps({"subj": "B", "pred": "same As", "obj": "A"})
        upsert_facts(conn, [(fid, "entity-b", "same As", "entity-a", '["B","same As","A"]', "Source statement")])
        conn.execute(
            "INSERT INTO fact_assertions VALUES ('assertion','chunk','revision','subject','object',?,?)", (raw, fid)
        )
        merged = merge_entities(conn, "entity-a", "entity-b")
        bind_merged_evidence(conn, merged)
        assert merged.canonical_id == "entity-a"
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
        assert tuple(conn.execute("SELECT raw_json,fact_id FROM fact_assertions").fetchone()) == (raw, None)
        before = snapshot(conn)
        with pytest.raises(ValueError, match="both resources"):
            merge_entities(conn, "entity-missing", "entity-a")
        assert snapshot(conn) == before
