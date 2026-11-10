"""Measured NER/triple provenance through native publication, merges and withdrawals."""

import json
import math
import sqlite3
from dataclasses import replace

import pytest

from daemon.graph.test_evidence_revisions import native_corpus, publish
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer, OpenIERecord, StructuredTriple, bayesian_noisy_or, pool_confidences
from yaw.daemon.graph.storage import load_graph_topology
from yaw.daemon.graph.storage.db import validate_graph_schema
from yaw.daemon.graph.storage.evidence import build_evidence


def source_vector(text):
    if "destination" in text.lower():
        return [0.0, 1.0, 0.0, 0.0]
    if "service" in text.lower():
        return [1.0, 0.0, 0.0, 0.0]
    return [1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("missing", [False, True])
def test_existing_extraction_calls_publish_measured_or_none_confidence(tmp_path, provider, missing):
    url, state = provider
    state.update(delay=0, omit_logprobs=missing)
    state["logprob_probability"] = lambda output: 0.64 if "named_entities" in output else 0.81
    state["embedding_vector"] = source_vector
    text = "Orion reports to Anchor."
    pointer = ChunkPointer("source", "source.md", 1, 1)
    (tmp_path / pointer.rel_posix_path).write_text(text, encoding="utf-8")
    state["extractions"] = {
        text: {
            "named_entities": [["Orion", "A service."], ["Anchor", "A destination."]],
            "triples": [{"sub": "Orion", "pred": "reports to", "obj": "Anchor", "def": "Orion reports to Anchor."}],
        }
    }
    with native_corpus(tmp_path, tmp_path / "index", url) as (db, env, app):
        publish(env, app, [(pointer, text)])
        assert len(state["extraction_requests"]) == 2
        assert state["extraction_logprobs"] == [True, True]
        assert state.get("synonym_requests", []) == []
        with db.readonly() as conn:
            entity_probabilities = [row[0] for row in conn.execute("SELECT confidence FROM entities")]
            fact = conn.execute("SELECT confidence FROM facts").fetchone()[0]
            source = tuple(conn.execute("SELECT llm_prob,vec_coherence,atomic_confidence FROM fact_sources").fetchone())
            raw = json.loads(conn.execute("SELECT raw_json FROM fact_assertions").fetchone()[0])
            record = json.loads(conn.execute("SELECT extraction_json FROM chunk_inference").fetchone()[0])
            if missing:
                assert entity_probabilities == pytest.approx([math.sqrt(0.5)] * 2)
                assert fact == pytest.approx(math.sqrt(0.5))
                assert source[0] is None
                assert source[1:] == pytest.approx([math.sqrt(0.5)] * 2)
                assert raw["llm_prob"] is None
                assert all(p is None for _, _, p in record["entity_probabilities"])
            else:
                assert entity_probabilities == pytest.approx([pool_confidences(0.64, math.sqrt(0.5))] * 2)
                assert fact == pytest.approx(pool_confidences(0.81, math.sqrt(0.5)))
                assert source[0] == pytest.approx(0.81)
                assert source[1] == pytest.approx(math.sqrt(0.5))
                assert source[2] == pytest.approx(pool_confidences(0.81, source[1]))
                assert raw["llm_prob"] == pytest.approx(0.81)
                assert [p for _, _, p in record["entity_probabilities"]] == pytest.approx([0.64, 0.64])
            changes = conn.total_changes
        calls = len(state["intervals"])
        publish(env, app, [(pointer, text)])
        assert len(state["intervals"]) == calls
        assert conn.total_changes == changes


def test_measurements_survive_merging_and_withdraw_with_their_source(tmp_path, provider):
    url, state = provider
    state.update(delay=0, relation="skos:altLabel")
    state["embedding_vector"] = source_vector

    def probability(output):
        if "named_entities" in output:
            return 0.64 if "North" in json.dumps(output) else (0.9 if state.get("reintroduced") else 0.81)
        if "triples" in output:
            return 0.49 if "North" in json.dumps(output) else (0.25 if state.get("reintroduced") else 0.36)
        return 0.81

    state["logprob_probability"] = probability
    state["extractions"] = {}
    chunks = []
    for region in ("North", "South"):
        name = f"Orion {region}"
        text = f"{name} routes through Anchor."
        pointer = ChunkPointer(region, f"{region}.md", 1, 1)
        (tmp_path / pointer.rel_posix_path).write_text(text, encoding="utf-8")
        chunks.append((pointer, text))
        state["extractions"][text] = {
            "named_entities": [
                [name, f"The telemetry routing service described by the {region} regional account."],
                ["Anchor", "The destination for telemetry routing."],
            ],
            "triples": [
                {"sub": name, "pred": "routes through", "obj": "Anchor", "def": f"{name} sends telemetry to Anchor."}
            ],
        }
    with native_corpus(tmp_path, tmp_path / "index", url) as (db, env, app):
        for step, included in enumerate((chunks, chunks[:1], chunks)):
            state["reintroduced"] = step == 2
            publish(env, app, included)
            with db.readonly() as conn:
                assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
                sources = [row[0] for row in conn.execute("SELECT llm_prob FROM fact_sources ORDER BY chunk_id")]
                expected_sources = [0.49] if len(included) == 1 else [0.49, 0.25 if step == 2 else 0.36]
                assert sorted(sources) == pytest.approx(sorted(expected_sources))
                assert conn.execute("SELECT confidence FROM facts").fetchone()[0] == pytest.approx(
                    bayesian_noisy_or([pool_confidences(p, math.sqrt(0.5)) for p in expected_sources])
                )
                expected_entity = bayesian_noisy_or(
                    [
                        pool_confidences(p, math.sqrt(0.5))
                        for p in ([0.64] if len(included) == 1 else [0.64, 0.9 if step == 2 else 0.81])
                    ]
                )
                assert [row[0] for row in conn.execute("SELECT confidence FROM entities")] == pytest.approx(
                    [expected_entity, expected_entity]
                )
                assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == len(included)
            # Two calls per normal source appearance; withdrawal itself performs no extraction.
            assert len(state["extraction_requests"]) == (4, 4, 6)[step]


def test_probability_metadata_does_not_fabricate_a_new_assertion_identity():
    pointer = ChunkPointer("source", "source.md", 1, 1)
    triple = StructuredTriple(
        subj="Orion",
        subj_def="A service.",
        pred="reports to",
        obj="Anchor",
        obj_def="A destination.",
        spo_def="Orion reports to Anchor.",
        llm_prob=0.2,
    )
    record = OpenIERecord(
        "source", ("Orion", "Anchor"), (), (triple,), (("Orion", "A service."), ("Anchor", "A destination."))
    )
    bindings = {
        ("source", "orion", "a service."): ("a", "A service."),
        ("source", "anchor", "a destination."): ("b", "A destination."),
    }
    _, first = build_evidence(pointer, "Source.", "program", record, bindings)
    _, second = build_evidence(
        pointer, "Source.", "program", replace(record, structured_triples=(replace(triple, llm_prob=0.9),)), bindings
    )
    assert first[0].assertion_id == second[0].assertion_id
    assert json.loads(first[0].raw_json)["llm_prob"] == 0.2
    assert json.loads(second[0].raw_json)["llm_prob"] == 0.9


def test_withdrawing_model_votes_keeps_semantic_votes_and_removes_stale_weights(tmp_path, provider):
    url, state = provider
    state.update(delay=0, relation="skos:altLabel")
    state["embedding_vector"] = source_vector
    state["extractions"] = {}
    chunks = []
    for region in ("North", "South"):
        text = f"Orion {region} reports to Anchor."
        pointer = ChunkPointer(region, f"{region}.md", 1, 1)
        (tmp_path / pointer.rel_posix_path).write_text(text, encoding="utf-8")
        chunks.append((pointer, text))
        state["extractions"][text] = {
            "named_entities": [
                [f"Orion {region}", f"A telemetry service described by the {region} office."],
                ["Anchor", "A destination."],
            ],
            "triples": [
                {
                    "sub": f"Orion {region}",
                    "pred": "reports to",
                    "obj": "Anchor",
                    "def": "The service reports to Anchor.",
                }
            ],
        }
    with native_corpus(tmp_path, tmp_path / "index", url) as (db, env, app):
        publish(env, app, chunks[:1])
        state["omit_logprobs"] = True
        publish(env, app, chunks)
        with db.readonly() as conn:
            assert conn.execute("SELECT confidence FROM facts").fetchone()[0] is not None
        publish(env, app, chunks[1:])
        with db.readonly() as conn:
            assert conn.execute("SELECT confidence FROM facts").fetchone()[0] == pytest.approx(math.sqrt(0.5))
            assert all(row[0] == pytest.approx(math.sqrt(0.5)) for row in conn.execute("SELECT weight FROM edges"))
            assert all(row[0] is None for row in conn.execute("SELECT llm_prob FROM edge_sources"))
            assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == 1
            assert load_graph_topology(conn)[2]


def test_legacy_mention_probability_schema_requires_source_rebuild(tmp_path):
    with sqlite3.connect(tmp_path / "legacy.db") as conn:
        conn.execute(
            "CREATE TABLE entity_mentions (mention_id TEXT PRIMARY KEY,chunk_id TEXT NOT NULL,"
            "source_revision TEXT NOT NULL,sense_id TEXT NOT NULL,name TEXT NOT NULL,definition TEXT NOT NULL,"
            "resolved_definition TEXT NOT NULL,entity_id TEXT NOT NULL)"
        )
        with pytest.raises(ValueError, match="rebuild the index from source"):
            validate_graph_schema(conn)
