"""Fixed evidence through real native targets, SQLite, vectors and loopback HTTP."""

import json
import time
from collections import Counter, defaultdict

import pytest

from daemon.graph.test_indexing_overlap import adapter
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import YawProgram, create_dspy_lm
from yaw.daemon.indexing import bind_native_indexing


def frozen_document(root):
    """One 128-chunk document: two topics, homographs, literals and a final bridge."""
    definitions = {
        "Ada": "A mathematician studying algebra.",
        "Algebra": "A mathematical discipline concerning symbolic structures.",
        "Euler": "A mathematician studying graphs.",
        "Graph theory": "A mathematical discipline concerning graphs.",
        "Mercury": "A planet orbiting the Sun.",
        "Sun": "The star at the center of the solar system.",
        "Conductivity": "The ability of a material to carry electric current.",
    }
    chunks = []
    extractions = {}
    for index in range(128):
        if index == 63:
            subject, predicate, obj = "Mercury", "orbits", "Sun"
        elif index == 64:
            subject, predicate, obj = "Mercury", "has property", "Conductivity"
        elif index == 127:
            subject, predicate, obj = "Algebra", "supports", "Graph theory"
        elif index % 2:
            subject, predicate, obj = "Euler", "studies", "Graph theory"
        else:
            subject, predicate, obj = "Ada", "studies", "Algebra"
        statement = f"{subject} {predicate} {obj}."
        text = f"Evidence {index}: {statement} Observation number: {index}."
        pairs = [[subject, definitions[subject]], [obj, definitions[obj]]]
        if index == 64:
            pairs[0][1] = "A metallic chemical element that conducts electric current."
        extractions[text] = {
            "named_entities": pairs,
            "triples": [
                {"sub": subject, "pred": [predicate, "owl:ObjectProperty"], "obj": obj, "def": statement},
                {
                    "sub": subject,
                    "pred": ["has observation number", "owl:DatatypeProperty"],
                    "obj": {"value": str(index), "datatype": "xsd:integer", "lang": ""},
                    "def": f"{subject} has observation number {index}.",
                },
            ],
        }
        chunks.append((ChunkPointer(f"c{index:03}", "document.md", index + 1, index + 1), text))
    (root / "document.md").write_text("\n".join(text for _, text in chunks) + "\n", encoding="utf-8")
    return chunks, extractions


def canonical_graph(conn):
    """Compare stored identities/support, not graph isomorphism or aggregate counts."""
    queries = {
        "entities": "SELECT entity_id, name, definition, confidence FROM entities ORDER BY entity_id",
        "labels": "SELECT * FROM entity_labels ORDER BY entity_id, label_norm, chunk_id",
        "mentions": "SELECT * FROM entity_mentions ORDER BY mention_id",
        "assertions": "SELECT * FROM fact_assertions ORDER BY assertion_id",
        "facts": (
            "SELECT fact_id, subj_id, pred, obj_kind, obj_id, obj_literal_lex, obj_datatype, obj_lang, "
            "raw_triple_json, definition, confidence FROM facts ORDER BY fact_id"
        ),
        "fact_sources": "SELECT * FROM fact_sources ORDER BY fact_id, chunk_id",
        "edges": "SELECT src_id, dst_id, kind, weight, reasoning FROM edges ORDER BY src_id, dst_id, kind",
        "edge_sources": "SELECT * FROM edge_sources ORDER BY src_id, dst_id, kind, chunk_id",
        "chunks": "SELECT * FROM chunks ORDER BY chunk_id",
        "chunk_fts": "SELECT chunk_id FROM chunk_fts ORDER BY chunk_id",
        "fact_fts": "SELECT fact_id FROM fact_fts ORDER BY fact_id",
        "extractions": "SELECT chunk_id, extraction_json FROM chunk_inference ORDER BY chunk_id",
        "chunk_vectors": "SELECT * FROM chunk_vec ORDER BY chunk_id",
        "entity_vectors": "SELECT * FROM entity_vec ORDER BY entity_id",
        "fact_vectors": "SELECT * FROM fact_vec ORDER BY fact_id",
    }
    return {name: conn.execute(sql).fetchall() for name, sql in queries.items()}


def connected(conn, left, right):
    adjacency = defaultdict(set)
    for source, target in conn.execute(
        "SELECT src_id, dst_id FROM edges WHERE kind != 'owl:differentFrom' AND weight > 0"
    ):
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen = {left}
    pending = [left]
    while pending:
        for neighbor in adjacency[pending.pop()] - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    return right in seen


def assert_same_graph(actual, expected):
    assert actual.keys() == expected.keys()
    for table, reference_rows in expected.items():
        rows = actual[table]
        assert len(rows) == len(reference_rows), f"{table}: {len(rows)} != {len(reference_rows)}"
        for row, reference in zip(rows, reference_rows, strict=True):
            for value, reference_value in zip(row, reference, strict=True):
                if isinstance(reference_value, float):
                    assert value == pytest.approx(reference_value, abs=1e-6, rel=0), table
                else:
                    assert value == reference_value, table


def fixture_vector(text):
    """Controlled geometry: the Mercury senses are distant, not perfect matches."""
    text = text.casefold()
    if "planet orbiting" in text or "mercury orbits sun" in text:
        return [1.0, 0.0, 0.0, 0.0]
    if "metallic chemical element" in text or "mercury has property conductivity" in text:
        return [0.0, 1.0, 0.0, 0.0]
    return [0.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("restart", [False, True], ids=["same-process", "restart-between-batches"])
def test_fixed_evidence_64_plus_64_equals_128(tmp_path, provider, restart, record_property):
    url, state = provider
    state["delay"] = 0
    state["relation"] = "owl:differentFrom"
    state["embedding_vector"] = fixture_vector
    chunks, state["extractions"] = frozen_document(tmp_path)
    results = {}
    timings = {}

    def make_engine(arm, batch_size):
        lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=512)
        program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
        return bind_native_indexing(
            YawGraphEngine(tmp_path / arm / "yaw.db", tmp_path, program, adapter(url)), batch_size=batch_size
        )

    for arm, batch_size in (("incremental", 64), ("reference", 128)):
        requests_before = len(state.get("extraction_requests", []))
        engine = make_engine(arm, batch_size)
        started = time.perf_counter()
        try:
            if arm == "incremental":
                engine.index_chunks(chunks[:64])
                ids = dict(engine.conn.execute("SELECT name, entity_id FROM entities"))
                assert not connected(engine.conn, ids["Ada"], ids["Euler"])
                assert engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 64
                if restart:
                    engine.close()
                    engine = make_engine(arm, batch_size)
            # The frozen API declares a complete snapshot, including after restart.
            engine.index_chunks(chunks)
            timings[arm] = time.perf_counter() - started
            counts = Counter(field for field, _ in state["extraction_requests"][requests_before:])
            assert counts == {"named_entities": 128, "triples": 128}
            assert engine.lint_graph()["conforms"]
            ids = dict(engine.conn.execute("SELECT name, entity_id FROM entities"))
            assert connected(engine.conn, ids["Ada"], ids["Euler"])
            assert engine.conn.execute("SELECT COUNT(*) FROM entities WHERE name = 'Mercury'").fetchone()[0] == 2
            assert engine.conn.execute("SELECT COUNT(*) FROM facts WHERE obj_kind = 'literal'").fetchone()[0] == 128
            assert engine.conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0] == 256
            assert (
                engine.conn.execute("SELECT COUNT(*) FROM fact_assertions WHERE fact_id IS NOT NULL").fetchone()[0]
                == 256
            )
            assert not connected(engine.conn, ids["Ada"], ids["Sun"])
            results[arm] = canonical_graph(engine.conn)
            previous_calls = len(state["intervals"])
            previous_changes = engine.conn.total_changes
            engine.index_chunks(chunks)
            assert len(state["intervals"]) == previous_calls
            assert engine.conn.total_changes == previous_changes
        finally:
            engine.close()
    record_property("loopback_fixture_seconds", json.dumps(timings, sort_keys=True))
    record_property(
        "graph_row_counts", json.dumps({arm: {k: len(v) for k, v in rows.items()} for arm, rows in results.items()})
    )
    assert_same_graph(results["incremental"], results["reference"])


def test_incremental_publication_without_memory_clone_or_full_survivor_scan(tmp_path, provider):
    import re

    import yaw.daemon.graph.planner as planner_mod

    assert not hasattr(planner_mod, "_".join(("clone", "graph", "to", "memory")))
    assert not hasattr(planner_mod, "_".join(("load", "active", "chunk", "inferences")))

    url, state = provider
    state["delay"] = 0
    state["relation"] = "owl:differentFrom"
    state["embedding_vector"] = fixture_vector
    chunks, state["extractions"] = frozen_document(tmp_path)

    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(
        YawGraphEngine(tmp_path / "inc_bounded" / "yaw.db", tmp_path, program, adapter(url)),
        batch_size=64,
    )
    try:
        engine.index_chunks(chunks[:64])
        embeds_after_b1 = len(state.get("embedding_inputs", []))
        readonly_selects: list[str] = []

        def trace_sql(stmt: str) -> None:
            norm = " ".join(stmt.strip().split())
            if not engine.conn.in_transaction and norm.upper().startswith("SELECT "):
                readonly_selects.append(norm)

        engine.conn.set_trace_callback(trace_sql)
        try:
            engine.index_chunks(chunks)
        finally:
            engine.conn.set_trace_callback(None)

        unbounded_pattern = re.compile(
            r"^SELECT\b(?!.*\bWHERE\b).*\bFROM\s+\"?(entities|facts|edges|chunk_vec|entity_vec|fact_vec|entity_mentions|fact_assertions)\"?\b",
            re.IGNORECASE,
        )
        unbounded_hits = [sql for sql in readonly_selects if unbounded_pattern.search(sql)]
        assert unbounded_hits == [], f"Unexpected unbounded full-table SELECTs on ro_conn: {unbounded_hits}"

        b2_embeds = state.get("embedding_inputs", [])[embeds_after_b1:]
        # Batch 2 must not re-embed any Batch 1 chunk (Evidence 0..63)
        reembedded_b1_chunks = [inp for inp in b2_embeds if any(f"Evidence {i}:" in inp for i in range(64))]
        assert reembedded_b1_chunks == []
        assert engine.lint_graph()["conforms"]
    finally:
        engine.close()
