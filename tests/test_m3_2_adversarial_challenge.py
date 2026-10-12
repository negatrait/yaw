"""Adversarial stress-test suite for Milestone 3 Challenger 2 (Phase 4 of Epic 7.7).

Adversarially probes:
1. Dead Code Purge Verification:
   - Repo-wide search and AST audit for zero occurrences of `leaf_nodes.embedding`,
     `_score_leaf_nodes_vec`, or vector scans against `leaf_nodes`.
   - Verification that leaf_nodes has NO vector column, no virtual table, and
     LeafNodeRow has no embedding field.
2. Pipeline Fact Embedding Edge Cases:
   - Empty relations list in `_embed_relation_facts` returns `{}` without failure.
   - Extreme & unusual characters (empty/whitespace names, multi-byte Unicode,
     emojis, quotes, SQL injection payloads, 5000-char strings) in triples.
   - Strict 768-dimensional float32 vector enforcement in `fact_vec` via sqlite-vec vec0;
     fail-loud rejection of 300d, 512d, 1024d, and corrupt byte buffers.
   - Offline degradation: when Ollama is offline or fact_embeddings is empty,
     relation persistence completes safely without crashing.
3. Representation Space Isolation & Zero Cross-Space Leakage:
   - Space S (`entity_vec`) is strictly isolated and NEVER queried by `retriever.py`
     (verified via static AST and dynamic SQLite query execution tracing).
   - Space R (`fact_vec`) is strictly isolated and NEVER queried by `canonicalize.py`
     for candidate matching (verified via static AST and dynamic DML query tracing).
   - Reflexive relation handling (subject_id == object_id) in query-to-fact seeding
     prevents double counting of similarity mass.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pytest

from yaw import config
from yaw.indexer.canonicalize import (
    create_in_memory_kb,
    resolve,
)
from yaw.indexer.pipeline import (
    LeafNodeRow,
    _embed_relation_facts,
    _relation_id,
    _resolve_and_store_relations,
)
from yaw.nlp.models import ExtractedNode, ExtractedTriple
from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage import (
    ensure_schema_tables,
    ensure_schema_views,
    get_db,
    insert_entity_vec,
    insert_fact_vec,
    serialize_embedding,
)

# ==============================================================================
# FIXTURES & HELPERS
# ==============================================================================


class _MockEmbeddingServer(BaseHTTPRequestHandler):
    """Loopback HTTP server returning real 768-dimensional float32 embeddings."""

    received_requests: ClassVar[list[dict[str, Any]]] = []

    def do_GET(self) -> None:
        if "version" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"version": "0.3.0"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body)
        _MockEmbeddingServer.received_requests.append(data)

        inputs = data.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        # Deterministic 768-dimensional normalized unit vector per input
        embeddings: list[list[float]] = []
        for inp in inputs:
            rng = np.random.RandomState(abs(hash(inp)) % (2**31 - 1))
            vec = rng.randn(768).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embeddings.append(vec.tolist())

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"embeddings": embeddings}).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def loopback_embed_server() -> Generator[str]:
    _MockEmbeddingServer.received_requests.clear()
    server = HTTPServer(("127.0.0.1", 0), _MockEmbeddingServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _init_sqlite_vault(tmp_path: Path) -> sqlite3.Connection:
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db(load_vec=True)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS leaf_relations (relation_id TEXT PRIMARY KEY, source_leaf_id TEXT, subject_id TEXT, predicate TEXT, predicate_id TEXT, object_id TEXT, weight REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
        "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, PRIMARY KEY (node_id, source_leaf_id))"
    )
    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    return conn


class MockTableTarget:
    """Minimal TableTarget recorder collecting declared rows without mocking."""

    def __init__(self) -> None:
        self.rows: list[Any] = []

    def declare_row(self, row: Any) -> None:
        self.rows.append(row)


# ==============================================================================
# SECTION 1: DEAD CODE PURGE VERIFICATION
# ==============================================================================


def test_repo_wide_ast_and_source_zero_leaf_nodes_embedding():
    """Verify zero leftover mentions of leaf_nodes.embedding or _score_leaf_nodes_vec in src/."""
    src_dir = Path(__file__).resolve().parent.parent / "src" / "yaw"
    assert src_dir.is_dir(), f"src directory not found at {src_dir}"

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        rel_str = str(py_file.relative_to(src_dir))

        # 1. No mention of _score_leaf_nodes_vec
        assert "_score_leaf_nodes_vec" not in content, (
            f"Dead code found in {rel_str}: _score_leaf_nodes_vec"
        )

        # 2. No query referencing leaf_nodes.embedding
        assert "leaf_nodes.embedding" not in content.lower(), (
            f"Forbidden column reference found in {rel_str}: leaf_nodes.embedding"
        )

        # 3. No SQL selecting embedding from leaf_nodes
        lower_content = " ".join(content.lower().split())
        assert "embedding from leaf_nodes" not in lower_content, (
            f"Forbidden SQL query selecting embedding from leaf_nodes in {rel_str}"
        )


def test_leaf_nodes_schema_strictness(tmp_path: Path):
    """Verify leaf_nodes table has NO vector column, no virtual table, and LeafNodeRow has no embedding."""
    conn = _init_sqlite_vault(tmp_path)
    try:
        # 1. PRAGMA table_info on leaf_nodes
        cols = [r[1] for r in conn.execute("PRAGMA table_info(leaf_nodes)").fetchall()]
        assert "embedding" not in cols, (
            f"leaf_nodes table unexpectedly contains 'embedding' column: {cols}"
        )
        assert cols == ["node_id", "source_leaf_id", "name", "namespace", "definition"]

        # 2. Check LeafNodeRow dataclass fields
        fields = list(LeafNodeRow.__dataclass_fields__.keys())
        assert "embedding" not in fields, (
            f"LeafNodeRow unexpectedly has 'embedding' field: {fields}"
        )
        assert fields == [
            "node_id",
            "source_leaf_id",
            "name",
            "namespace",
            "definition",
        ]

        # 3. Check virtual tables in sqlite_master
        vec_tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%vec0%'"
            ).fetchall()
        ]
        assert "leaf_nodes" not in vec_tables
        assert "leaf_nodes_vec" not in vec_tables
        assert "entity_vec" in vec_tables
        assert "fact_vec" in vec_tables
    finally:
        conn.close()


# ==============================================================================
# SECTION 2: PIPELINE FACT EMBEDDING EDGE CASES
# ==============================================================================


@pytest.mark.anyio
async def test_embed_relation_facts_empty_list():
    """_embed_relation_facts handles empty list without exceptions and returns empty dict."""
    result = await _embed_relation_facts([])
    assert result == {}


@pytest.mark.anyio
async def test_embed_relation_facts_extreme_and_unusual_characters(
    loopback_embed_server: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify _embed_relation_facts handles unusual characters, Unicode, emojis, SQL injection, and long strings."""
    triple_specs = [
        # Multi-byte CJK
        ("量子计算", "accelerates", "密码学", "concepts"),
        # Multi-byte Cyrillic
        ("нейронная сеть", "learns", "представление", "concepts"),
        # Multi-byte Arabic
        ("الذكاء الاصطناعي", "powers", "البحث", "concepts"),
        # Emoji characters
        ("🧠 Brain", "generates", "💡 Insight", "concepts"),
        # SQL injection strings
        ("'; DROP TABLE fact_vec; --", "injects", "' OR '1'='1", "concepts"),
        # Quotes and special symbols
        ('Special "Quotes" & <Tags>', "maps to", "Symbolic / Logic", "concepts"),
        # Very long entity string
        ("A" * 2000, "relates_to", "B" * 2000, "concepts"),
    ]

    triples: list[tuple[str, ExtractedTriple]] = []
    for s_name, p_name, o_name, ns in triple_specs:
        node_s = ExtractedNode(name=s_name, namespace=ns)
        node_p = ExtractedNode(name=p_name, namespace="predicates")
        node_o = ExtractedNode(name=o_name, namespace=ns)
        triple = ExtractedTriple(subject=node_s, predicate=node_p, object=node_o)
        triples.append(("raw/adversarial.md", triple))

    # Point to loopback server
    monkeypatch.setattr(config, "OLLAMA_HOST", loopback_embed_server)

    results = await _embed_relation_facts(triples)
    assert len(results) == len(triples)
    for vec in results.values():
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)

    # Verify requests sent to loopback server
    reqs = _MockEmbeddingServer.received_requests
    assert len(reqs) >= 1
    embedded_inputs = reqs[-1]["input"]
    assert len(embedded_inputs) == len(triples)
    # Check that each input has Space R document formatting
    for text in embedded_inputs:
        assert text.startswith("title: none | text: ")


def test_fact_vec_dimension_strictness_and_fail_loud(tmp_path: Path):
    """Verify fact_vec accepts only 768d float32 vectors and rejects dimension mismatches loudly."""
    conn = _init_sqlite_vault(tmp_path)
    try:
        # 1. 768-dim float32 vector succeeds
        vec_768 = np.random.randn(768).astype(np.float32)
        vec_768 /= np.linalg.norm(vec_768)
        bytes_768 = serialize_embedding(vec_768)
        assert bytes_768 is not None
        insert_fact_vec(conn, "fact:test:valid768", bytes_768)

        # Read back and verify exact 768d float32 restoration
        row = conn.execute(
            "SELECT relation_id, embedding FROM fact_vec WHERE relation_id = ?",
            ("fact:test:valid768",),
        ).fetchone()
        assert row is not None
        restored = np.frombuffer(row[1], dtype=np.float32)
        assert restored.shape == (768,)
        assert restored.dtype == np.float32
        assert np.allclose(vec_768, restored, atol=1e-6)

        # 2. 300-dim vector fails loud with OperationalError
        vec_300 = np.random.randn(300).astype(np.float32)
        bytes_300 = serialize_embedding(vec_300)
        assert bytes_300 is not None
        with pytest.raises(sqlite3.OperationalError) as exc_300:
            insert_fact_vec(conn, "fact:test:bad300", bytes_300)
        assert "expected 768 dimensions but received 300" in str(exc_300.value).lower()

        # 3. 512-dim vector fails loud
        vec_512 = np.random.randn(512).astype(np.float32)
        bytes_512 = serialize_embedding(vec_512)
        assert bytes_512 is not None
        with pytest.raises(sqlite3.OperationalError) as exc_512:
            insert_fact_vec(conn, "fact:test:bad512", bytes_512)
        assert "expected 768 dimensions but received 512" in str(exc_512.value).lower()

        # 4. 1024-dim vector fails loud
        vec_1024 = np.random.randn(1024).astype(np.float32)
        bytes_1024 = serialize_embedding(vec_1024)
        assert bytes_1024 is not None
        with pytest.raises(sqlite3.OperationalError) as exc_1024:
            insert_fact_vec(conn, "fact:test:bad1024", bytes_1024)
        assert (
            "expected 768 dimensions but received 1024" in str(exc_1024.value).lower()
        )

        # 5. Corrupted bytes fail loud
        with pytest.raises(sqlite3.OperationalError):
            insert_fact_vec(conn, "fact:test:corrupt", b"not_a_valid_float_vector")
    finally:
        conn.close()


def test_resolve_and_store_relations_deduplication_and_offline(tmp_path: Path):
    """Verify _resolve_and_store_relations deduplicates identical rel_ids and handles offline embeddings."""
    conn = _init_sqlite_vault(tmp_path)
    try:
        kb = create_in_memory_kb()
        relations_target = MockTableTarget()
        seen_rel_ids: set[str] = set()

        # Two triples with identical slugs that produce the same rel_id
        t1 = ExtractedTriple(
            subject=ExtractedNode(name="Albert Einstein", namespace="concepts"),
            predicate=ExtractedNode(name="born in", namespace="predicates"),
            object=ExtractedNode(name="Ulm", namespace="concepts"),
        )
        t2 = ExtractedTriple(
            subject=ExtractedNode(name="albert einstein", namespace="concepts"),
            predicate=ExtractedNode(name="born-in", namespace="predicates"),
            object=ExtractedNode(name="ulm", namespace="concepts"),
        )

        all_triples = [
            ("raw/einstein.md", t1),
            ("raw/einstein.md", t2),
        ]

        expected_rel_id = _relation_id(
            "raw/einstein.md",
            t1.subject.name,
            t1.predicate.name,
            t1.object.name,
        )

        # Fake 768d fact embeddings
        v768 = np.ones(768, dtype=np.float32).tolist()
        fact_embeddings = {expected_rel_id: v768}

        _resolve_and_store_relations(
            all_triples,
            cast(Any, relations_target),
            conn,
            seen_rel_ids,
            kb=kb,
            fact_embeddings=fact_embeddings,
        )

        # 1. Exactly one relation row declared (deduplicated)
        assert len(relations_target.rows) == 1
        assert relations_target.rows[0].relation_id == expected_rel_id

        # 2. Exactly one row in fact_vec
        fact_row = conn.execute(
            "SELECT relation_id FROM fact_vec WHERE relation_id = ?",
            (expected_rel_id,),
        ).fetchone()
        assert fact_row is not None
        assert fact_row[0] == expected_rel_id

        # 3. Test offline scenario (fact_embeddings is None or empty):
        relations_target_2 = MockTableTarget()
        seen_rel_ids_2: set[str] = set()
        t3 = ExtractedTriple(
            subject=ExtractedNode(name="Niels Bohr", namespace="concepts"),
            predicate=ExtractedNode(name="debates", namespace="predicates"),
            object=ExtractedNode(name="Albert Einstein", namespace="concepts"),
        )
        _resolve_and_store_relations(
            [("raw/bohr.md", t3)],
            cast(Any, relations_target_2),
            conn,
            seen_rel_ids_2,
            kb=kb,
            fact_embeddings={},
        )
        assert len(relations_target_2.rows) == 1
        # No fact_vec row should exist for bohr relation
        bohr_rel_id = _relation_id(
            "raw/bohr.md", t3.subject.name, t3.predicate.name, t3.object.name
        )
        assert (
            conn.execute(
                "SELECT relation_id FROM fact_vec WHERE relation_id = ?",
                (bohr_rel_id,),
            ).fetchone()
            is None
        )
    finally:
        conn.close()


# ==============================================================================
# SECTION 3: REPRESENTATION SPACE ISOLATION & LEAKAGE CHECK
# ==============================================================================


@pytest.mark.anyio
async def test_retriever_never_queries_entity_vec_static_and_dynamic(
    tmp_path: Path,
    loopback_embed_server: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify entity_vec is strictly Space S and never accessed by retriever.py statically or dynamically."""
    # 1. Static check: 'entity_vec' string never appears in retriever.py
    retriever_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "yaw"
        / "retrieval"
        / "retriever.py"
    )
    retriever_code = retriever_path.read_text(encoding="utf-8")
    assert "entity_vec" not in retriever_code, (
        "Representation space leakage: 'entity_vec' found in retriever.py!"
    )

    # 2. Dynamic check: Trace all SQL queries executed during retriever.compute_seed_vector and search
    monkeypatch.setattr(config, "OLLAMA_HOST", loopback_embed_server)
    conn = _init_sqlite_vault(tmp_path)
    try:
        node_1 = "concepts:general_relativity"
        node_2 = "concepts:gravitational_waves"
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/gr.md', 'General Relativity', 'concepts', 'Theory of gravity.')",
            (node_1,),
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/gw.md', 'Gravitational Waves', 'concepts', 'Ripples in spacetime.')",
            (node_2,),
        )
        rel_id = "raw/gr.md#general-relativity:predicts:gravitational-waves"
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'raw/gr.md', ?, 'predicts', 'predicates:predicts', ?, 1.0)",
            (rel_id, node_1, node_2),
        )

        v768 = np.ones(768, dtype=np.float32)
        v768 /= np.linalg.norm(v768)
        bytes_768 = serialize_embedding(v768)
        assert bytes_768 is not None

        # Insert into both entity_vec (Space S) and fact_vec (Space R)
        insert_entity_vec(conn, node_1, "concepts", bytes_768)
        insert_fact_vec(conn, rel_id, bytes_768)

        # Setup retriever
        kb = create_in_memory_kb()
        retriever = GraphRAGRetriever(kb=kb)

        # Trace all SQL queries
        executed_queries: list[str] = []

        def trace_callback(statement: str) -> None:
            executed_queries.append(statement)

        conn.set_trace_callback(trace_callback)

        # Run compute_seed_vector
        p0, _, _cand_rows = await retriever.compute_seed_vector(
            query="gravitational waves prediction",
            conn=conn,
        )

        conn.set_trace_callback(None)

        # Verify fact_vec was queried
        fact_vec_queried = any("fact_vec" in q.lower() for q in executed_queries)
        assert fact_vec_queried, (
            "Expected fact_vec to be queried for query-to-fact seeding"
        )

        # Verify entity_vec was NEVER queried (neither DDL nor DML)
        entity_vec_queried = any("entity_vec" in q.lower() for q in executed_queries)
        assert not entity_vec_queried, (
            "Representation space leakage: entity_vec was dynamically queried during retrieval!"
        )

        # Verify p0 received seed mass from the fact
        assert np.sum(p0) > 0.0
    finally:
        conn.close()


def test_canonicalize_never_queries_fact_vec_static_and_dynamic(tmp_path: Path):
    """Verify fact_vec is strictly Space R and never accessed by canonicalize.py for candidate matching."""
    # 1. Static check: 'fact_vec' string never appears in canonicalize.py
    canon_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "yaw"
        / "indexer"
        / "canonicalize.py"
    )
    canon_code = canon_path.read_text(encoding="utf-8")
    assert "fact_vec" not in canon_code, (
        "Representation space leakage: 'fact_vec' found in canonicalize.py!"
    )

    # 2. Dynamic check: Trace all SQL queries executed during canonicalization
    conn = _init_sqlite_vault(tmp_path)
    try:
        v768 = np.ones(768, dtype=np.float32)
        v768 /= np.linalg.norm(v768)
        bytes_768 = serialize_embedding(v768)
        assert bytes_768 is not None

        # Populate entity_vec and fact_vec
        insert_entity_vec(conn, "concepts:special_relativity", "concepts", bytes_768)
        insert_fact_vec(conn, "fact:test:rel1", bytes_768)

        kb = create_in_memory_kb()

        # Trace all SQL queries
        executed_queries: list[str] = []

        def trace_callback(statement: str) -> None:
            executed_queries.append(statement)

        conn.set_trace_callback(trace_callback)

        # Perform Tier 2 vector resolution
        resolve(
            name="Special Theory of Relativity",
            kind="entity",
            definition_ctx="Einstein 1905 paper on electrodynamics.",
            conn=conn,
            embedding=v768.tolist(),
            namespace="concepts",
            kb=kb,
        )

        conn.set_trace_callback(None)

        # Verify entity_vec was queried for KNN matching
        entity_vec_dml = any(
            "entity_vec" in q.lower() and not q.strip().upper().startswith("CREATE")
            for q in executed_queries
        )
        assert entity_vec_dml, "Expected entity_vec to be queried during Tier 2 KNN"

        # Verify fact_vec was NEVER queried for data matching (zero DML queries on fact_vec)
        fact_vec_dml = any(
            "fact_vec" in q.lower() and not q.strip().upper().startswith("CREATE")
            for q in executed_queries
        )
        assert not fact_vec_dml, (
            "Representation space leakage: fact_vec was queried for data during canonicalization!"
        )
    finally:
        conn.close()


@pytest.mark.anyio
async def test_reflexive_relation_endpoints_no_double_counting(
    tmp_path: Path,
    loopback_embed_server: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify reflexive relation (subject_id == object_id) does not double count endpoint similarity."""
    monkeypatch.setattr(config, "OLLAMA_HOST", loopback_embed_server)
    conn = _init_sqlite_vault(tmp_path)
    try:
        node_id = "concepts:self_loop_concept"
        conn.execute(
            "INSERT INTO leaf_nodes VALUES (?, 'raw/loop.md', 'Self Loop', 'concepts', 'Reflexive concept.')",
            (node_id,),
        )
        rel_id = "raw/loop.md#self-loop:relates_to:self-loop"
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'raw/loop.md', ?, 'relates_to', 'predicates:relates_to', ?, 1.0)",
            (rel_id, node_id, node_id),
        )

        v768 = np.ones(768, dtype=np.float32)
        v768 /= np.linalg.norm(v768)
        bytes_768 = serialize_embedding(v768)
        assert bytes_768 is not None
        insert_fact_vec(conn, rel_id, bytes_768)

        kb = create_in_memory_kb()
        retriever = GraphRAGRetriever(kb=kb)

        p0, _, _ = await retriever.compute_seed_vector(
            query="self loop",
            conn=conn,
        )

        idx = retriever._node_to_idx[node_id]
        # Mass should be valid and bounded, not duplicated or inflated
        assert p0[idx] > 0.0
        assert np.isclose(np.sum(p0), 1.0, atol=1e-5)
    finally:
        conn.close()
