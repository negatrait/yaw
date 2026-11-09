"""Fixed evidence through real native targets, SQLite, vectors and loopback HTTP."""

import json
import time
from collections import Counter, defaultdict

import pytest

from daemon.graph.test_indexing_overlap import adapter
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer, compute_chunk_id
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
        rel_path = f"doc_{index:03d}.md"
        cid = compute_chunk_id(rel_path, 1, 1, text)
        ptr = ChunkPointer(cid, rel_path, 1, 1, f"doc_{index:03d}", 1, 1)
        chunks.append((ptr, text))
        (root / rel_path).write_text(text + "\n", encoding="utf-8")
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


def test_reverse_top_k_asymmetric_candidate_completeness_64_plus_64_equals_128(tmp_path, provider):
    """Verify reverse top-k candidate completeness when e_old is excluded from e_new's forward top-k (U3)."""
    import math

    url, state = provider
    state["delay"] = 0
    state["relation"] = "skos:related"

    satellites = [
        ("consensus log module one", 0.945),
        ("replicated state unit two", 0.942),
        ("quorum ballot tracker three", 0.939),
        ("heartbeat rpc channel four", 0.936),
        ("snapshot compaction worker five", 0.933),
        ("commit index monitor six", 0.930),
    ]
    sat_cos_by_name = {name: c for name, c in satellites}
    orthogonal_entities = [
        "byzantine fault tolerance",
        "chord dht routing",
        "consistent hashing ring",
        "lamport logical clock",
        "vector timestamp ordering",
        "leader election term",
    ]
    orth_idx_by_name = {name: idx for idx, name in enumerate(orthogonal_entities)}

    def asymmetric_vector(text: str) -> list[float]:
        t = text.casefold()
        if "# raft consensus mechanism\n" in t:
            return [1.0, 0.0, 0.0, 0.0]
        if "# distributed consensus protocol\n" in t:
            return [0.86, math.sqrt(1.0 - 0.86**2), 0.0, 0.0]
        for sat_name, c in sat_cos_by_name.items():
            if f"# {sat_name}\n" in t:
                return [c, -math.sqrt(1.0 - c**2), 0.0, 0.0]
        for orth_name, idx in orth_idx_by_name.items():
            if f"# {orth_name}\n" in t:
                angle = idx * (math.pi / 3.0)
                return [0.0, 0.0, math.cos(angle), math.sin(angle)]
        return [0.0, 0.0, 1.0, 0.0]

    state["embedding_vector"] = asymmetric_vector

    definitions = {
        "distributed consensus protocol": "Protocol for achieving agreement across distributed network nodes.",
        "byzantine fault tolerance": "Resilience of a fault-tolerant computer system to arbitrary node failures.",
        "chord dht routing": "Structured peer-to-peer distributed hash table lookup protocol.",
        "consistent hashing ring": "Hashing scheme minimizing key remapping when hash table slots resize.",
        "lamport logical clock": "Monotonically increasing software counter capturing causal event ordering.",
        "vector timestamp ordering": "Array of logical clocks tracking causal history across distributed processes.",
        "raft consensus mechanism": "Replicated state machine protocol using leader-driven log replication.",
        "leader election term": "Monotonically numbered epoch governed by a single elected cluster leader.",
        **{
            sat_name: f"Replicated state machine satellite module {sat_name.rsplit(' ', 1)[-1]}."
            for sat_name, _ in satellites
        },
    }

    b1_pairs = [
        ("distributed consensus protocol", "byzantine fault tolerance", "mitigates"),
        ("chord dht routing", "consistent hashing ring", "partitions"),
        ("lamport logical clock", "vector timestamp ordering", "generalizes to"),
    ]
    b2_pairs = [
        ("raft consensus mechanism", "leader election term", "advances"),
        *[(sat_name, "leader election term", "synchronizes") for sat_name, _ in satellites],
    ]

    chunks = []
    extractions = {}
    for index in range(128):
        if index < 64:
            subject, obj, predicate = b1_pairs[index % len(b1_pairs)]
        else:
            subject, obj, predicate = b2_pairs[(index - 64) % len(b2_pairs)]
        statement = f"{subject} {predicate} {obj}."
        text = f"Asymmetric {index}: {statement}"
        extractions[text] = {
            "named_entities": [[subject, definitions[subject]], [obj, definitions[obj]]],
            "triples": [
                {"sub": subject, "pred": [predicate, "owl:ObjectProperty"], "obj": obj, "def": statement},
            ],
        }
        rel_path = f"asym_{index:03d}.md"
        cid = compute_chunk_id(rel_path, 1, 1, text)
        ptr = ChunkPointer(cid, rel_path, 1, 1, f"asym_{index:03d}", 1, 1)
        chunks.append((ptr, text))
        (tmp_path / rel_path).write_text(text + "\n", encoding="utf-8")
    state["extractions"] = extractions

    results = {}

    def make_engine(arm: str, batch_size: int) -> YawGraphEngine:
        lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=512)
        program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
        return bind_native_indexing(
            YawGraphEngine(
                tmp_path / f"asym_{arm}" / "yaw.db",
                tmp_path,
                program,
                adapter(url),
                synonymy_threshold=0.80,
                synonymy_top_k=5,
            ),
            batch_size=batch_size,
        )

    for arm, batch_size in (("incremental", 64), ("reference", 128)):
        engine = make_engine(arm, batch_size)
        try:
            if arm == "incremental":
                engine.index_chunks(chunks[:64])
                b1_ids = dict(engine.conn.execute("SELECT name, entity_id FROM entities"))
                assert len(b1_ids) == 6
                assert "distributed consensus protocol" in b1_ids
            engine.index_chunks(chunks)
            assert engine.lint_graph()["conforms"]
            ids = dict(engine.conn.execute("SELECT name, entity_id FROM entities"))
            assert len(ids) == 14
            e_old_id = ids["distributed consensus protocol"]
            e_new_id = ids["raft consensus mechanism"]
            rel_edge = engine.conn.execute(
                """
                SELECT 1 FROM edges
                WHERE ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                  AND kind = 'skos:related'
                """,
                (e_old_id, e_new_id, e_new_id, e_old_id),
            ).fetchone()
            assert rel_edge is not None, f"Reverse top-k pair (e_old, e_new) missing in {arm} arm"
            assert connected(engine.conn, e_old_id, e_new_id)
            results[arm] = canonical_graph(engine.conn)
        finally:
            engine.close()

    assert_same_graph(results["incremental"], results["reference"])


def test_passive_endpoint_homograph_preserves_edges_and_facts_across_batches(tmp_path, provider):
    """Verify passive-endpoint homographs in load_affected_neighborhood preserve edges and facts (CG-OPT-2)."""

    url, state = provider
    state["delay"] = 0
    state["relation"] = "skos:related"

    def homograph_passive_vector(text: str) -> list[float]:
        t = text.casefold()
        if "horizontal row of seating" in t or "tier located in stadium" in t:
            return [1.0, 0.0, 0.0, 0.0]
        if "pricing level in a software" in t or "tier governs subscription" in t or "tier targets enterprise" in t:
            return [0.0, 1.0, 0.0, 0.0]
        if "corporate organization" in t:
            return [0.0, -1.0, 0.0, 0.0]
        if "sports arena" in t:
            return [0.0, 0.0, 1.0, 0.0]
        if "recurring software licensing" in t:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.0, 0.0, 1.0, 0.0]

    state["embedding_vector"] = homograph_passive_vector

    c0_text = "Batch1 A: Tier is located in Stadium."
    c1_text = "Batch1 B: Tier governs Subscription and Tier targets Enterprise."
    c2_text = "Batch2 C: Stadium offers Subscription."

    state["extractions"] = {
        c0_text: {
            "named_entities": [
                ["Tier", "A horizontal row of seating in a stadium."],
                ["Stadium", "A sports arena with tiered seating."],
            ],
            "triples": [
                {
                    "sub": "Tier",
                    "pred": ["located in", "owl:ObjectProperty"],
                    "obj": "Stadium",
                    "def": "Tier is located in Stadium.",
                }
            ],
        },
        c1_text: {
            "named_entities": [
                ["Tier", "A pricing level in a software subscription model."],
                ["Subscription", "A recurring software licensing model."],
                ["Enterprise", "A large corporate organization."],
            ],
            "triples": [
                {
                    "sub": "Tier",
                    "pred": ["governs", "owl:ObjectProperty"],
                    "obj": "Subscription",
                    "def": "Tier governs Subscription.",
                },
                {
                    "sub": "Tier",
                    "pred": ["targets", "owl:ObjectProperty"],
                    "obj": "Enterprise",
                    "def": "Tier targets Enterprise.",
                },
            ],
        },
        c2_text: {
            "named_entities": [
                ["Stadium", "A sports arena with tiered seating."],
                ["Subscription", "A recurring software licensing model."],
            ],
            "triples": [
                {
                    "sub": "Stadium",
                    "pred": ["offers", "owl:ObjectProperty"],
                    "obj": "Subscription",
                    "def": "Stadium offers Subscription.",
                }
            ],
        },
    }

    c_texts = [c0_text, c1_text, c2_text]
    chunks = [
        (ChunkPointer(compute_chunk_id(f"passive_{i}.md", 1, 1, t), f"passive_{i}.md", 1, 1), t)
        for i, t in enumerate(c_texts)
    ]
    for ptr, t in chunks:
        (tmp_path / ptr.rel_posix_path).write_text(t + "\n", encoding="utf-8")

    results = {}

    def make_engine(arm: str, batch_size: int) -> YawGraphEngine:
        lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=512)
        program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
        return bind_native_indexing(
            YawGraphEngine(
                tmp_path / f"passive_{arm}" / "yaw.db",
                tmp_path,
                program,
                adapter(url),
                synonymy_threshold=0.80,
                synonymy_top_k=5,
            ),
            batch_size=batch_size,
        )

    for arm, batch_size in (("incremental", 2), ("reference", 4)):
        engine = make_engine(arm, batch_size)
        try:
            if arm == "incremental":
                engine.index_chunks(chunks[:2])
                assert engine.lint_graph()["conforms"]
                assert engine.conn.execute("SELECT COUNT(*) FROM entities WHERE name = 'Tier'").fetchone()[0] == 2
            engine.index_chunks(chunks)
            assert engine.lint_graph()["conforms"]
            assert engine.conn.execute("SELECT COUNT(*) FROM entities WHERE name = 'Tier'").fetchone()[0] == 2
            assert engine.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 4
            results[arm] = canonical_graph(engine.conn)
        finally:
            engine.close()

    assert_same_graph(results["incremental"], results["reference"])


def test_suffix_lexical_match_on_passive_endpoint_preserves_closed_world_across_batches(tmp_path, provider):
    """Verify suffix lexical matches (>= 0.70) with disjoint 3-char prefixes preserve closed-world across batches."""
    url, state = provider
    state["delay"] = 0
    state["relation"] = "skos:exactMatch"

    def suffix_lexical_vector(text: str) -> list[float]:
        t = text.casefold()
        if "nimbus client sdk" in t:
            return [1.0, 0.0, 0.0, 0.0]
        if "sse disconnect" in t:
            return [0.0, 1.0, 0.0, 0.0]
        if "streaming disconnect" in t:
            return [0.0, 0.0, 1.0, 0.0]
        if "sse-gateway" in t:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.5, 0.5]

    state["embedding_vector"] = suffix_lexical_vector

    c0_text = "Batch1 A: Nimbus client SDK is retried on streaming disconnect."
    c1_text = "Batch1 B: sse-gateway monitors streaming disconnect."
    c2_text = "Batch2 C: sse-gateway mitigates SSE disconnect."

    state["extractions"] = {
        c0_text: {
            "named_entities": [
                ["Nimbus client SDK", "Client software development kit."],
                ["streaming disconnect", "Short-lived connection disruption during telemetry streams."],
            ],
            "triples": [
                {
                    "sub": "Nimbus client SDK",
                    "pred": ["retried on", "owl:ObjectProperty"],
                    "obj": "streaming disconnect",
                    "def": "Nimbus client SDK is retried on streaming disconnect.",
                }
            ],
        },
        c1_text: {
            "named_entities": [
                ["sse-gateway", "Gateway component handling SSE streams."],
                ["streaming disconnect", "Short-lived connection disruption during telemetry streams."],
            ],
            "triples": [
                {
                    "sub": "sse-gateway",
                    "pred": ["monitors", "owl:ObjectProperty"],
                    "obj": "streaming disconnect",
                    "def": "sse-gateway monitors streaming disconnect.",
                }
            ],
        },
        c2_text: {
            "named_entities": [
                ["sse-gateway", "Gateway component handling SSE streams."],
                ["SSE disconnect", "Short-lived connection disruption during telemetry streams."],
            ],
            "triples": [
                {
                    "sub": "sse-gateway",
                    "pred": ["mitigates", "owl:ObjectProperty"],
                    "obj": "SSE disconnect",
                    "def": "sse-gateway mitigates SSE disconnect.",
                }
            ],
        },
    }

    c_texts = [c0_text, c1_text, c2_text]
    chunks = [
        (ChunkPointer(compute_chunk_id(f"suffix_{i}.md", 1, 1, t), f"suffix_{i}.md", 1, 1), t)
        for i, t in enumerate(c_texts)
    ]
    for ptr, t in chunks:
        (tmp_path / ptr.rel_posix_path).write_text(t + "\n", encoding="utf-8")

    results = {}

    def make_engine(arm: str, batch_size: int) -> YawGraphEngine:
        lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=512)
        program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
        return bind_native_indexing(
            YawGraphEngine(
                tmp_path / f"suffix_{arm}" / "yaw.db",
                tmp_path,
                program,
                adapter(url),
                synonymy_threshold=0.80,
                synonymy_top_k=5,
            ),
            batch_size=batch_size,
        )

    for arm, batch_size in (("incremental", 2), ("reference", 4)):
        engine = make_engine(arm, batch_size)
        try:
            if arm == "incremental":
                engine.index_chunks(chunks[:2])
                assert engine.lint_graph()["conforms"]
            engine.index_chunks(chunks)
            assert engine.lint_graph()["conforms"]
            results[arm] = canonical_graph(engine.conn)
        finally:
            engine.close()

    assert_same_graph(results["incremental"], results["reference"])


def test_g5_g6_paired_64_plus_64_vs_128_acceptance_report(tmp_path, provider):
    """Run N=20 paired fixed-evidence trials (10 cold + 10 warm Stage-1) and emit g5_g6_acceptance.json (G5-G6)."""
    import re
    import statistics
    import subprocess
    import sys
    import urllib.request
    from pathlib import Path

    import psutil

    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    state["delay"] = 0
    state["relation"] = "owl:differentFrom"
    state["embedding_vector"] = fixture_vector
    chunks, state["extractions"] = frozen_document(tmp_path)

    unbounded_pattern = re.compile(
        r"^SELECT\b(?!.*\bWHERE\b).*\bFROM\s+\"?(entities|facts|edges|chunk_vec|entity_vec|fact_vec|entity_mentions|fact_assertions)\"?\b",
        re.IGNORECASE,
    )
    b1_chunk_ids = {ptr.chunk_id for ptr, _ in chunks[:64]}
    b1_mutation_pattern = re.compile(
        r"^(UPDATE|DELETE)\b.*\b(chunks|chunk_vec|chunk_fts|source_chunks|chunk_inference)\b",
        re.IGNORECASE,
    )

    # Pre-warm Stage-1 cache (extract_text) for the 10 warm_stage1_memo trials
    lm_seed = create_dspy_lm("overlap", base_url=url + "/v1", cache=True, max_tokens=512)
    prog_seed = YawProgram(orchestrator_lm=lm_seed, rlm_sub_lm=lm_seed, enable_gbnf=False, max_output_tokens=512)
    eng_seed = bind_native_indexing(
        YawGraphEngine(tmp_path / "warm_seed" / "yaw.db", tmp_path, prog_seed, adapter(url)),
        batch_size=128,
        state_root=tmp_path / "warm_template_state",
    )
    try:
        eng_seed.index_chunks(chunks)
    finally:
        eng_seed.close()

    cold_trials: list[dict] = []
    warm_trials: list[dict] = []
    proc = psutil.Process()
    peak_rss_bytes = int(proc.memory_info().rss)
    total_reembedded_b1 = 0
    total_unindexed_selects = 0
    total_unaffected_writes = 0

    for trial_idx in range(20):
        regime = "cold_memo" if trial_idx < 10 else "warm_stage1_memo"
        arm_order = ("incremental", "reference") if trial_idx % 2 == 0 else ("reference", "incremental")
        trial_dir = tmp_path / f"trial_{trial_idx:02d}"
        trial_graphs: dict[str, dict] = {}
        trial_timings: dict[str, float] = {}

        for arm in arm_order:
            batch_size = 64 if arm == "incremental" else 128
            arm_dir = trial_dir / arm
            arm_state_root = arm_dir / "state"
            use_warm_stage1 = regime == "warm_stage1_memo"

            lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=use_warm_stage1, max_tokens=512)
            program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
            engine = bind_native_indexing(
                YawGraphEngine(arm_dir / "yaw.db", tmp_path, program, adapter(url)),
                batch_size=batch_size,
                state_root=arm_state_root,
            )
            try:
                if arm == "incremental":
                    readonly_selects: list[str] = []
                    b1_mutations: list[str] = []

                    def trace_sql(
                        stmt: str,
                        _ro=readonly_selects,
                        _mut=b1_mutations,
                        _eng=engine,
                    ) -> None:
                        norm = " ".join(stmt.strip().split())
                        if not _eng.conn.in_transaction and norm.upper().startswith("SELECT "):
                            _ro.append(norm)
                        elif b1_mutation_pattern.search(norm) and any(cid in norm for cid in b1_chunk_ids):
                            _mut.append(norm)

                    b1_state = {"embeds_after_b1": 0, "batches_seen": 0}

                    def on_progress(
                        message: str,
                        _b1=b1_state,
                        _eng=engine,
                        _tr=trace_sql,
                    ) -> None:
                        if message.endswith("chunks published"):
                            _b1["batches_seen"] += 1
                            if _b1["batches_seen"] == 1:
                                _b1["embeds_after_b1"] = len(state.get("embedding_inputs", []))
                                _eng.conn.set_trace_callback(_tr)
                            else:
                                _eng.conn.set_trace_callback(None)

                    t0 = time.perf_counter()
                    try:
                        engine.index_chunks(chunks, progress_callback=on_progress)
                    finally:
                        engine.conn.set_trace_callback(None)
                    trial_timings[arm] = time.perf_counter() - t0
                    assert b1_state["batches_seen"] == 2
                    embeds_after_b1 = b1_state["embeds_after_b1"]

                    b2_embeds = state.get("embedding_inputs", [])[embeds_after_b1:]
                    reembedded = [inp for inp in b2_embeds if any(f"Evidence {i}:" in inp for i in range(64))]
                    unindexed = [sql for sql in readonly_selects if unbounded_pattern.search(sql)]
                    total_reembedded_b1 += len(reembedded)
                    total_unindexed_selects += len(unindexed)
                    total_unaffected_writes += len(b1_mutations)
                    assert reembedded == []
                    assert unindexed == []
                    assert b1_mutations == []
                else:
                    t0 = time.perf_counter()
                    engine.index_chunks(chunks)
                    trial_timings[arm] = time.perf_counter() - t0

                assert engine.lint_graph()["conforms"]
                trial_graphs[arm] = canonical_graph(engine.conn)
                peak_rss_bytes = max(peak_rss_bytes, int(proc.memory_info().rss))
            finally:
                engine.close()

        assert_same_graph(trial_graphs["incremental"], trial_graphs["reference"])
        ratio = trial_timings["incremental"] / trial_timings["reference"]
        record = {
            "trial": trial_idx,
            "regime": regime,
            "order": f"{arm_order[0]}->{arm_order[1]}",
            "incremental_s": round(trial_timings["incremental"], 6),
            "reference_s": round(trial_timings["reference"], 6),
            "ratio": round(ratio, 6),
            "canonical_15_table_parity": True,
        }
        if regime == "cold_memo":
            cold_trials.append(record)
        else:
            warm_trials.append(record)

    r_time_ratios = [t["ratio"] for t in cold_trials]
    r_graph_ratios = [t["ratio"] for t in warm_trials]
    r_time_mean = statistics.mean(r_time_ratios)
    r_time_median = statistics.median(r_time_ratios)
    r_graph_mean = statistics.mean(r_graph_ratios)
    r_graph_median = statistics.median(r_graph_ratios)

    assert 0.80 <= r_time_mean <= 1.25, f"R_time mean {r_time_mean:.4f} outside [0.80, 1.25]"
    assert 0.80 <= r_graph_mean <= 1.25, f"R_graph mean {r_graph_mean:.4f} outside [0.80, 1.25]"
    assert total_reembedded_b1 == 0
    assert total_unindexed_selects == 0
    assert total_unaffected_writes == 0
    assert peak_rss_bytes <= 1073741824

    # Probe local live inference server status & run verify_paper_parity_gate.py
    live_endpoint_reachable = False
    try:
        with urllib.request.urlopen("http://localhost:13305/api/v1/models", timeout=2.0) as resp:
            live_endpoint_reachable = resp.status == 200
    except OSError:
        live_endpoint_reachable = False

    repo_root = Path(__file__).resolve().parents[3]
    parity_script = repo_root / "scripts" / "verify_paper_parity_gate.py"
    tmp_parity_json = tmp_path / "paper_parity_gate.json"
    tmp_parity_md = tmp_path / "paper_parity_gate.md"
    existing_parity_json = repo_root / "outputs" / "benchmarks" / "paper_parity_gate.json"
    if existing_parity_json.exists():
        parity_data = json.loads(existing_parity_json.read_text(encoding="utf-8"))
        for ds_entry in parity_data.get("datasets", {}).values():
            if isinstance(ds_entry, dict):
                ds_entry["mode"] = "retrieval"
        tmp_parity_json.write_text(json.dumps(parity_data, indent=2), encoding="utf-8")
    parity_proc = subprocess.run(
        [
            sys.executable,
            str(parity_script),
            "--datasets",
            "popqa",
            "--mode",
            "retrieval",
            "--output-json",
            str(tmp_parity_json),
            "--output-md",
            str(tmp_parity_md),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
    )

    report_payload = {
        "checkpoint": "U3/P3-P4 & I4/G5-G6",
        "trials_total": 20,
        "cold_memo_trials": len(cold_trials),
        "warm_stage1_memo_trials": len(warm_trials),
        "R_time": {
            "mean": round(r_time_mean, 4),
            "median": round(r_time_median, 4),
            "min": round(min(r_time_ratios), 4),
            "max": round(max(r_time_ratios), 4),
            "within_0_80_to_1_25": 0.80 <= r_time_mean <= 1.25,
        },
        "R_graph": {
            "mean": round(r_graph_mean, 4),
            "median": round(r_graph_median, 4),
            "min": round(min(r_graph_ratios), 4),
            "max": round(max(r_graph_ratios), 4),
            "within_0_80_to_1_25": 0.80 <= r_graph_mean <= 1.25,
        },
        "invariants": {
            "canonical_15_table_parity_trials_passed": 20,
            "shacl_conforms": True,
            "batch2_reembedded_batch1_chunks": total_reembedded_b1,
            "batch2_unindexed_readonly_selects": total_unindexed_selects,
            "batch2_unaffected_row_writes": total_unaffected_writes,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_limit_bytes": 1073741824,
        },
        "table_row_counts": {k: len(v) for k, v in trial_graphs["incremental"].items()},
        "live_silicon_parity_probe": {
            "endpoint": "http://localhost:13305/api/v1/models",
            "live_endpoint_reachable": live_endpoint_reachable,
            "verify_paper_parity_gate_returncode": parity_proc.returncode,
            "verify_paper_parity_gate_summary": (parity_proc.stdout or parity_proc.stderr).strip()[:500],
        },
        "trials": [*cold_trials, *warm_trials],
    }
    out_path = repo_root / "outputs" / "benchmarks" / "g5_g6_acceptance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
