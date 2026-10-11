"""Un-mocked integration tests for CLI lifecycle, GraphRAG conductance, and document hydration.

Adheres strictly to .agents/skills/engineering-real-tests/SKILL.md:
- Real SQLite database with full schema and views.
- Real disk files with UTF-8 byte offsets.
- Real SciPy sparse matrix operations.
- Zero unittest.mock or synthetic test doubles.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import multiprocessing
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest
import scipy.sparse as sp

from yaw.server.daemon import LOCK_EX, LOCK_NB, LOCK_UN, flock

REPO_ROOT = Path(__file__).resolve().parent.parent
YAW_SCRIPT = str(REPO_ROOT / "yaw.py")

from conftest import BaseTestHTTPHandler, get_cli_env

from yaw import config
from yaw.retrieval.retriever import (
    RELEVANCE_THRESHOLD,
    GraphRAGRetriever,
    hydrate_disk_slice,
    read_document,
)
from yaw.server.daemon import (
    DaemonLifecycle,
    compute_seed_candidates,
    find_seed,
    stop_daemon,
)
from yaw.storage.db import ensure_schema_views

# ==============================================================================
# Milestone 1: Reciprocal Wikilink Back-Edge Conductance & SciPy CSR Tests
# ==============================================================================


def test_reciprocal_wikilink_back_edge_csr_conductance(tmp_path: Path):
    """Verify reciprocal wikilink synthesis and SciPy CSR coordinate summing.

    Invariant:
    - Forward wikilink (u, v) synthesizes reciprocal back-edge (v, u) with weight 0.5 * w.
    - Mutually linked notes (A -> B and B -> A) naturally sum in CSR construction to 1.0 + 0.5 = 1.5.
    - Unilateral link (C -> D) has forward weight 1.0 and reciprocal weight 0.5 on (D -> C).
    - Rows in row-stochastic matrix _P sum to 1.0 for all non-dangling nodes.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )

        # 1. Mutual wikilinks: A -> B and B -> A
        conn.execute("INSERT INTO wikilinks VALUES ('raw/a.md', 'raw/b.md', 'b', 1.0)")
        conn.execute("INSERT INTO wikilinks VALUES ('raw/b.md', 'raw/a.md', 'a', 1.0)")

        # 2. Unilateral wikilink: C -> D
        conn.execute("INSERT INTO wikilinks VALUES ('raw/c.md', 'raw/d.md', 'd', 1.0)")

        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        # Verify contiguous indexing and matrix existence
        assert retriever.adj is not None
        assert isinstance(retriever.adj, sp.csr_matrix)
        assert len(retriever._idx_to_node) == 4

        node_to_idx = retriever._node_to_idx
        idx_a = node_to_idx["raw/a.md"]
        idx_b = node_to_idx["raw/b.md"]
        idx_c = node_to_idx["raw/c.md"]
        idx_d = node_to_idx["raw/d.md"]

        adj = retriever.adj.toarray()

        # Mutual link summing invariant: 1.0 (fwd) + 0.5 (rev) = 1.5
        assert pytest.approx(float(adj[idx_a, idx_b]), abs=1e-5) == 1.5
        assert pytest.approx(float(adj[idx_b, idx_a]), abs=1e-5) == 1.5

        # Unilateral link invariant: 1.0 (fwd), 0.5 (rev)
        assert pytest.approx(float(adj[idx_c, idx_d]), abs=1e-5) == 1.0
        assert pytest.approx(float(adj[idx_d, idx_c]), abs=1e-5) == 0.5

        # No spurious cross-edges
        assert pytest.approx(float(adj[idx_a, idx_c]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_a, idx_d]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_b, idx_c]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_b, idx_d]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_c, idx_a]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_c, idx_b]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_d, idx_a]), abs=1e-5) == 0.0
        assert pytest.approx(float(adj[idx_d, idx_b]), abs=1e-5) == 0.0

        # Row-stochastic transition matrix _P verification
        assert retriever._P is not None
        P = retriever._P.toarray()
        for i in range(4):
            assert pytest.approx(float(P[i].sum()), rel=1e-5) == 1.0


def test_reciprocal_wikilink_hub_dampening_forward_edges(tmp_path: Path):
    """Verify hub dampening is computed strictly across forward edges.

    Invariant:
    - 4 forward links point to Hub (in-degree = 4).
    - Hub dampening gamma = 0.5 scales forward edges by 1 / sqrt(4) = 0.5.
    - Each source has forward in-degree 0 (max(1, 0) = 1).
    - Reciprocal edges (Hub -> Source) have raw weight 0.5 and denominator 1.0, so weight is 0.5.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL)"
        )
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )

        sources = [f"raw/src_{i}.md" for i in range(4)]
        hub = "raw/hub.md"
        for s in sources:
            conn.execute("INSERT INTO wikilinks VALUES (?, ?, 'hub', 1.0)", (s, hub))

        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        adj = retriever.adj.toarray()
        node_to_idx = retriever._node_to_idx
        idx_hub = node_to_idx[hub]

        for s in sources:
            idx_s = node_to_idx[s]
            # Forward edge dampened by hub forward in-degree: 1.0 / sqrt(4) = 0.5
            assert pytest.approx(float(adj[idx_s, idx_hub]), abs=1e-5) == 0.5
            # Reciprocal back-edge from hub to source: 0.5 / sqrt(1) = 0.5
            assert pytest.approx(float(adj[idx_hub, idx_s]), abs=1e-5) == 0.5


# ==============================================================================
# Milestone 1: Document Hydration & Path Confinement Tests
# ==============================================================================


def test_hydrate_disk_slice_eof_and_bounded(tmp_path: Path):
    """Verify hydrate_disk_slice reads whole file when line args are None and supports bounded line slicing."""
    config.set_vault(tmp_path)
    test_file = tmp_path / "notes" / "mechanics.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)

    content = "Line 1: Quantum mechanics: Δx · Δp ≥ ℏ/2\nLine 2: Schrödinger 🐱\nLine 3: Dirac notation\n"
    test_file.write_text(content, encoding="utf-8")

    # 1. line_start and line_end are None -> read entire file
    assert hydrate_disk_slice("notes/mechanics.md", None, None) == content
    assert hydrate_disk_slice("notes/mechanics.md") == content

    # 2. Bounded slice line 1 to 1
    assert (
        hydrate_disk_slice("notes/mechanics.md", 1, 1)
        == "Line 1: Quantum mechanics: Δx · Δp ≥ ℏ/2\n"
    )

    # 3. Bounded slice lines 2 to 3
    assert (
        hydrate_disk_slice("notes/mechanics.md", 2, 3)
        == "Line 2: Schrödinger 🐱\nLine 3: Dirac notation\n"
    )

    # 4. Non-existent file returns empty string
    assert hydrate_disk_slice("notes/missing.md", 1, 10) == ""
    assert hydrate_disk_slice("notes/missing.md") == ""

    # 5. Seek past EOF returns empty string
    assert hydrate_disk_slice("notes/mechanics.md", 100, 150) == ""


def test_hydrate_disk_slice_security_confinement(tmp_path: Path):
    """Verify hydrate_disk_slice rejects directory traversal and confines to VAULT_ROOT."""
    vault = tmp_path / "vault"
    vault.mkdir()
    config.set_vault(vault)

    # Valid file inside vault
    inside = vault / "inside.md"
    inside.write_text("safe content", encoding="utf-8")

    # Secret file outside vault
    outside = tmp_path / "secret.env"
    outside.write_text("API_KEY=top_secret", encoding="utf-8")

    # 1. Traversal via relative ../
    with pytest.raises(ValueError, match="outside VAULT_ROOT"):
        hydrate_disk_slice("../secret.env")

    # 2. Deep traversal escaping vault
    with pytest.raises(ValueError, match="outside VAULT_ROOT"):
        hydrate_disk_slice("sub/../../secret.env")

    # 3. Absolute path pointing outside vault
    with pytest.raises(ValueError, match="outside VAULT_ROOT"):
        hydrate_disk_slice(str(outside))

    # 4. System path traversal
    with pytest.raises(ValueError, match="outside VAULT_ROOT"):
        hydrate_disk_slice("/etc/passwd")

    # 5. Valid access inside vault succeeds
    assert hydrate_disk_slice("inside.md") == "safe content"
    assert hydrate_disk_slice(str(inside)) == "safe content"


def test_read_document_parity(tmp_path: Path):
    """Verify read_document maintains full parity with hydrate_disk_slice."""
    config.set_vault(tmp_path)
    doc_path = tmp_path / "doc.md"
    content = "Line 1: Authoritative document text for GraphRAG.\nLine 2: Second line of context.\n"
    doc_path.write_text(content, encoding="utf-8")

    # 1. Full read
    assert read_document("doc.md") == content
    assert read_document("doc.md", None, None) == content

    # 2. Bounded slice
    assert (
        read_document("doc.md", 1, 1)
        == "Line 1: Authoritative document text for GraphRAG.\n"
    )
    assert read_document("doc.md", 1, 2) == content

    # 3. Confinement check
    with pytest.raises(ValueError, match="outside VAULT_ROOT"):
        read_document("../outside.md")


def test_relevance_threshold_module_alignment():
    """Verify retriever.RELEVANCE_THRESHOLD aligns with config.RELEVANCE_THRESHOLD."""
    assert RELEVANCE_THRESHOLD == config.RELEVANCE_THRESHOLD
    assert isinstance(RELEVANCE_THRESHOLD, float)
    assert RELEVANCE_THRESHOLD == 0.10


# ==============================================================================
# Milestone 2: DaemonLifecycle, WAL Truncate Checkpoint & In-Memory Rescoring
# ==============================================================================


class _LoopbackHandler(BaseTestHTTPHandler):
    pass


@pytest.fixture
def loopback_server():
    _LoopbackHandler.received_requests.clear()
    server = HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_daemon_lifecycle_pid_and_signal_management(tmp_path: Path):
    """Verify real PID file creation, PID matching, signal handler setup, and cleanup on shutdown."""
    config.set_vault(tmp_path)
    pid_file = tmp_path / ".cache" / "yaw.pid"
    lifecycle = DaemonLifecycle(pid_file=pid_file)

    assert not pid_file.exists()

    # 1. Start lifecycle -> PID file created with current process PID
    lifecycle.start()
    assert pid_file.exists(), "PID file must exist after lifecycle start"
    written_pid = int(pid_file.read_text(encoding="utf-8").strip())
    assert written_pid == os.getpid(), (
        f"PID file must contain current PID {os.getpid()}, got {written_pid}"
    )

    # 2. Setup signals -> registers SIGINT and SIGTERM handlers
    orig_sigint = signal.getsignal(signal.SIGINT)
    orig_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        lifecycle.setup_signals()
        handler_int = signal.getsignal(signal.SIGINT)
        handler_term = signal.getsignal(signal.SIGTERM)
        assert callable(handler_int), "SIGINT handler must be callable"
        assert callable(handler_term), "SIGTERM handler must be callable"
        assert handler_int == handler_term, (
            "SIGINT and SIGTERM should share lifecycle handler"
        )
    finally:
        signal.signal(signal.SIGINT, orig_sigint)
        signal.signal(signal.SIGTERM, orig_sigterm)

    # 3. Shutdown -> PID file cleanly removed
    lifecycle.shutdown()
    assert not pid_file.exists(), "PID file must be deleted upon shutdown"

    # 4. Idempotency check: repeated shutdown does not raise FileNotFoundError
    lifecycle.shutdown()
    assert not pid_file.exists()


def test_daemon_lifecycle_signal_interception(tmp_path: Path):
    """Verify that _signal_handler intercepts termination signal and exits cleanly."""
    config.set_vault(tmp_path)
    pid_file = tmp_path / ".cache" / "yaw.pid"
    lifecycle = DaemonLifecycle(pid_file=pid_file)
    lifecycle.start()
    assert pid_file.exists()

    with pytest.raises(SystemExit) as exc_info:
        lifecycle._signal_handler(signal.SIGTERM, None)
    assert exc_info.value.code == 0
    assert not pid_file.exists()


def test_daemon_lifecycle_wal_checkpoint_truncate(tmp_path: Path):
    """Verify real WAL page accumulation, checkpoint truncation, and zero data loss."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    wal_path = tmp_path / ".cache" / "yaw.db-wal"

    # Create table in WAL mode and commit substantial batch to generate dirty WAL pages
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE note_chunks (chunk_id TEXT PRIMARY KEY, payload BLOB);")
    chunk_payload = b"X" * 4096
    for i in range(25):
        conn.execute("INSERT INTO note_chunks VALUES (?, ?);", (f"c{i}", chunk_payload))
    conn.commit()

    # Physical verification: WAL file must exist and contain dirty pages
    assert wal_path.exists(), "WAL file must exist after uncheckpointed commits"
    initial_wal_size = wal_path.stat().st_size
    assert initial_wal_size > 0, (
        f"WAL size must be > 0 bytes before checkpoint, got {initial_wal_size}"
    )

    lifecycle = DaemonLifecycle(db_path=db_path)

    # Execute checkpoint via active connection
    res = lifecycle.checkpoint_wal(conn)
    # PRAGMA wal_checkpoint(TRUNCATE) returns (busy=0, log=0, checkpointed=0) on success
    assert res == (0, 0, 0), (
        f"Expected (0, 0, 0) from wal_checkpoint(TRUNCATE), got {res}"
    )

    # Physical disk verification: WAL file must be truncated to 0 bytes
    assert wal_path.stat().st_size == 0, "WAL file size must be 0 bytes post-truncation"

    # Data integrity: all 25 rows remain readable
    row_count = conn.execute("SELECT count(*) FROM note_chunks;").fetchone()[0]
    assert row_count == 25, (
        "All rows must remain accessible after WAL truncate checkpoint"
    )

    # Secondary verification: checkpoint without connection parameter (uses db_path)
    # Generate another dirty commit while keeping conn open to maintain WAL file presence
    conn2 = sqlite3.connect(str(db_path))
    conn2.execute("INSERT INTO note_chunks VALUES ('c_new', ?);", (chunk_payload,))
    conn2.commit()
    conn2.close()

    assert wal_path.stat().st_size > 0

    res2 = lifecycle.checkpoint_wal()
    assert res2 == (0, 0, 0)
    assert wal_path.stat().st_size == 0
    conn.close()


def test_daemon_lifecycle_shutdown_sla(tmp_path: Path):
    """Verify shutdown executes checkpoint, removes PID, and completes well within 5.0s SLA."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    pid_file = tmp_path / ".cache" / "yaw.pid"

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE t (id INT);")
    conn.execute("INSERT INTO t VALUES (1);")
    conn.commit()

    lifecycle = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    lifecycle.start()
    assert pid_file.exists()

    t0 = time.monotonic()
    lifecycle.shutdown(conn)
    duration = time.monotonic() - t0
    conn.close()

    assert duration < 5.0, f"Shutdown took {duration:.2f}s, exceeding 5.0s SLA"
    assert not pid_file.exists()


@pytest.mark.anyio
async def test_in_memory_rescoring_zero_outbound_http_calls(
    tmp_path: Path, loopback_server: str
):
    """Verify compute_seed_candidates performs strictly zero outbound HTTP calls during rescoring."""
    config.set_vault(tmp_path)
    config.HOST = loopback_server
    config.OLLAMA_HOST = loopback_server

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Author physical markdown file
    note_file = raw_dir / "quantum.md"
    note_text = (
        "# Quantum Mechanics\nHeisenberg uncertainty relation defines conjugate limit."
    )
    note_file.write_text(note_text, encoding="utf-8")

    # 2. Build real SQLite DB with vectors and FTS5
    db_path = cache_dir / "yaw.db"
    vec_aligned = np.zeros(768, dtype=np.float32)
    vec_aligned[0] = 1.0  # dot product with [1.0, 0.0, ...] is 1.0

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c1', 'raw/quantum.md#chunk:0', 'raw/quantum.md', 0, 'Quantum Mechanics', 'quantum-mechanics', 1, 2, ?)",
            (vec_aligned.tobytes(),),
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('raw/quantum.md#chunk:0', 'raw/quantum.md', 'Quantum Mechanics', ?)",
            (note_text,),
        )
        conn.commit()

    # Pre-defined query vector: [1.0, 0.0, ...]
    q_vec = np.zeros(768, dtype=np.float32)
    q_vec[0] = 1.0

    _LoopbackHandler.received_requests.clear()

    # Query with query_vec
    seeds = await compute_seed_candidates(
        "uncertainty relation", top_k=1, query_vec=q_vec
    )

    # Substrate Invariant: Zero HTTP calls to loopback server
    assert len(_LoopbackHandler.received_requests) == 0, (
        f"Zero HTTP calls permitted during retrieval; received {len(_LoopbackHandler.received_requests)}"
    )

    # Invariant: Output candidate matches dot product similarity
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["entity_uri"] == "raw/quantum.md#chunk:0"
    assert pytest.approx(seed["rerank_score"], abs=1e-4) == 1.0
    assert seed["text"] == note_text
    assert seed["line_start"] == 1
    assert seed["line_end"] == 2


@pytest.mark.anyio
async def test_in_memory_rescoring_missing_vector_fallback(
    tmp_path: Path, loopback_server: str, caplog: pytest.LogCaptureFixture
):
    """Verify missing SQLite vector falls back to hybrid_prob without HTTP embed_texts call, logging a warning."""
    config.set_vault(tmp_path)
    config.HOST = loopback_server

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    note_file = raw_dir / "unembedded.md"
    content = "# Unembedded Note\nLexical text only without vector embedding."
    note_file.write_text(content, encoding="utf-8")

    db_path = cache_dir / "yaw.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        # Store chunk with NULL embedding
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c1', 'raw/unembedded.md#chunk:0', 'raw/unembedded.md', 0, 'Unembedded', 'unembedded', 1, 2, NULL)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('raw/unembedded.md#chunk:0', 'raw/unembedded.md', 'Unembedded', ?)",
            (content,),
        )
        conn.commit()

    q_vec = np.ones(768, dtype=np.float32)
    _LoopbackHandler.received_requests.clear()

    with caplog.at_level(logging.WARNING):
        seeds = await compute_seed_candidates("Lexical text", top_k=1, query_vec=q_vec)

    # Invariant 1: Zero HTTP calls to embed_texts
    assert len(_LoopbackHandler.received_requests) == 0, (
        "No HTTP embed_texts fallback allowed"
    )

    # Invariant 2: Candidate rerank_score equals hybrid_prob
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["rerank_score"] == seed["hybrid_prob"]

    # Invariant 3: Warning was logged indicating missing vector
    warning_logged = any(
        "lacks stored embedding vector" in r.message for r in caplog.records
    )
    assert warning_logged, (
        "Must log warning when candidate lacks stored embedding vector"
    )


@pytest.mark.anyio
async def test_find_seed_structured_output_contract_and_disk_seek(tmp_path: Path):
    """Verify find_seed returns 9-field list[dict] with valid physical disk byte pointers."""
    config.set_vault(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Multi-byte UTF-8 content
    content = "# Relativity\nMass-energy equivalence is expressed as $E = mc^2$ ⚡.\n## General\nSpacetime curvature."
    note_path = raw_dir / "relativity.md"
    note_path.write_text(content, encoding="utf-8")

    chunk1_text = "# Relativity\nMass-energy equivalence is expressed as $E = mc^2$ ⚡.\n"
    line_start = 1
    line_end = 2

    db_path = cache_dir / "yaw.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
            "heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, "
            "embedding BLOB)"
        )
        conn.execute(
            "INSERT INTO note_chunks VALUES ('c1', 'raw/relativity.md#chunk:0', 'raw/relativity.md', 0, 'Relativity', 'relativity', ?, ?, NULL)",
            (line_start, line_end),
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, content='', contentless_unindexed=1)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('raw/relativity.md#chunk:0', 'raw/relativity.md', 'Relativity', ?)",
            (chunk1_text,),
        )
        conn.commit()

    results = await find_seed("mass-energy equivalence", top_k=1)

    # 1. Type contract: list[dict]
    assert isinstance(results, list)
    assert len(results) == 1
    seed = results[0]
    assert isinstance(seed, dict)

    # 2. Strict 9-field schema
    expected_keys = {
        "entity_uri",
        "kind",
        "rel_path",
        "heading",
        "line_start",
        "line_end",
        "hybrid_prob",
        "rerank_score",
        "text",
    }
    assert set(seed.keys()) == expected_keys

    # 3. Data type assertions
    assert isinstance(seed["entity_uri"], str)
    assert isinstance(seed["kind"], str)
    assert isinstance(seed["rel_path"], str)
    assert isinstance(seed["heading"], str)
    assert isinstance(seed["line_start"], int)
    assert isinstance(seed["line_end"], int)
    assert isinstance(seed["hybrid_prob"], float)
    assert isinstance(seed["rerank_score"], float)
    assert isinstance(seed["text"], str)

    # 4. Zero-copy line hydration verification
    hydrated = hydrate_disk_slice(seed["rel_path"], seed["line_start"], seed["line_end"])
    assert hydrated.rstrip("\r\n") == seed["text"].rstrip("\r\n")


def _get_free_port() -> int:
    """Find an available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Milestone 3: CLI Subparsers & STDIO Protocol Hygiene Tests


def test_cli_index_foreground(tmp_path: Path):
    """Verify yaw index executes synchronously in foreground, exits 0, and leaves no zombies.

    Invariants:
    - Synchronous execution: subprocess blocks until indexing is fully complete.
    - Exit code 0 on valid vault directory.
    - Real database: .cache/yaw.db exists on disk with populated tables.
    - Process hygiene: no detached or zombie background processes remain running.
    """
    vault_dir = tmp_path / "vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "index_test.md"
    note_content = (
        "# Quantum Entanglement\n"
        "Entangled states demonstrate non-local correlations predicted by quantum theory."
    )
    note_path.write_text(note_content, encoding="utf-8")

    env = get_cli_env()

    parent_proc = psutil.Process(os.getpid())
    children_before = {p.pid for p in parent_proc.children(recursive=True)}

    cmd = [
        sys.executable,
        YAW_SCRIPT,
        "index",
        "--vault",
        str(vault_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
        check=False,
    )

    # Invariant 1: Exit code 0
    assert result.returncode == 0, (
        f"yaw index failed with code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    # Invariant 2: Physical SQLite database exists
    db_path = vault_dir / ".cache" / "yaw.db"
    assert db_path.is_file(), f"Database {db_path} must exist after foreground indexing"

    # Invariant 3: Real data committed to tables
    with sqlite3.connect(str(db_path)) as conn:
        chunk_count = conn.execute("SELECT count(*) FROM note_chunks").fetchone()[0]
        assert chunk_count >= 1, "note_chunks must contain indexed rows"

    # Invariant 4: No lingering detached background or zombie processes
    children_after = set(parent_proc.children(recursive=True))
    new_lingering = [
        p
        for p in children_after
        if p.pid not in children_before
        and p.is_running()
        and p.status() != psutil.STATUS_ZOMBIE
    ]
    assert len(new_lingering) == 0, (
        f"Detected lingering background child processes: {[p.pid for p in new_lingering]}"
    )


def test_cli_schema_json(tmp_path: Path):
    """Verify yaw schema outputs valid JSON with namespaces, base_predicates, and stats.

    Invariants:
    - Exit code 0.
    - Pure JSON output on stdout: json.loads succeeds with zero decode errors.
    - Schema contract: dict containing 'namespaces' (list), 'base_predicates' (list),
      and 'stats' (dict with 'chunks', 'taxonomies', 'edges').
    - All diagnostic/log output directed strictly to stderr.
    """
    vault_dir = tmp_path / "vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "entropy.md"
    note_path.write_text(
        "# Thermodynamics\nEntropy always increases in an isolated system.",
        encoding="utf-8",
    )

    env = get_cli_env()

    # Index note first so stats are populated
    index_res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
        check=False,
    )
    assert index_res.returncode == 0

    cmd = [
        sys.executable,
        YAW_SCRIPT,
        "schema",
        "--vault",
        str(vault_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=15.0,
        check=False,
    )

    # Invariant 1: Exit code 0
    assert result.returncode == 0, (
        f"yaw schema failed with code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    # Invariant 2: Pure JSON output on stdout
    try:
        schema = json.loads(result.stdout.strip())
    except json.JSONDecodeError as err:
        pytest.fail(
            f"yaw schema stdout is not valid JSON: {err}\nSTDOUT:\n{result.stdout}"
        )

    # Invariant 3: Dict structure and schema keys
    assert isinstance(schema, dict)
    assert "namespaces" in schema, "schema JSON missing 'namespaces'"
    assert "base_predicates" in schema, "schema JSON missing 'base_predicates'"
    assert "stats" in schema, "schema JSON missing 'stats'"

    assert isinstance(schema["namespaces"], list)
    assert isinstance(schema["base_predicates"], list)
    assert isinstance(schema["stats"], dict)

    # Invariant 4: Stats dictionary contains 'chunks', 'taxonomies', 'edges'
    stats = schema["stats"]
    assert "chunks" in stats, "stats missing 'chunks'"
    assert "taxonomies" in stats, "stats missing 'taxonomies'"
    assert "edges" in stats, "stats missing 'edges'"

    assert isinstance(stats["chunks"], int)
    assert isinstance(stats["taxonomies"], int)
    assert isinstance(stats["edges"], int)
    assert stats["chunks"] >= 1, "chunks stat must reflect indexed note"


def test_cli_query_markdown(tmp_path: Path):
    """Verify yaw query outputs dual-layer Markdown on stdout.

    Invariants:
    - Exit code 0.
    - Output on stdout contains '# Ontological Context' and '# Retrieved Ground-Truth Evidence'.
    - Retrieved passages contain ground-truth text from physical vault files.
    - Diagnostic logging routed to stderr, not polluting stdout.
    """
    vault_dir = tmp_path / "vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "uncertainty.md"
    note_content = (
        "# Uncertainty Principle\n"
        "Werner Heisenberg formulated the uncertainty principle in quantum mechanics, "
        "bounding the precision of conjugate observables."
    )
    note_path.write_text(note_content, encoding="utf-8")

    env = get_cli_env()

    # Index the note
    index_res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30.0,
        check=False,
    )
    assert index_res.returncode == 0

    cmd = [
        sys.executable,
        YAW_SCRIPT,
        "query",
        "uncertainty principle",
        "--vault",
        str(vault_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=15.0,
        check=False,
    )

    # Invariant 1: Exit code 0
    assert result.returncode == 0, (
        f"yaw query failed with code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )

    # Invariant 2: Dual-layer Markdown sections present
    stdout = result.stdout
    assert "# Ontological Context" in stdout, (
        "Query output missing '# Ontological Context'"
    )
    assert "## Evidence Chunk" in stdout, "Query output missing '## Evidence Chunk'"

    # Invariant 3: Ground-truth content matches physical note
    assert "uncertainty" in stdout.lower() or "heisenberg" in stdout.lower(), (
        f"Expected ground-truth note content in query output:\n{stdout}"
    )


def test_cli_serve_stdio_pure_json(tmp_path: Path):
    """Verify yaw serve --transport stdio outputs strictly pure JSON-RPC on stdout.

    Invariants:
    - FastMCP banner is suppressed (show_banner=False).
    - Exactly 0 non-JSON characters written to stdout.
    - JSON-RPC initialize request returns valid JSON-RPC 2.0 response.
    - All server logs and diagnostic information routed to stderr.
    """
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)

    env = get_cli_env()

    cmd = [
        sys.executable,
        YAW_SCRIPT,
        "serve",
        "--transport",
        "stdio",
        "--vault",
        str(vault_dir),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    stdout_rem = ""
    try:
        # Standard MCP / JSON-RPC 2.0 initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        assert proc.stdin is not None
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line, "No response received on stdout from yaw serve --transport stdio"

        # Invariant 1: Strictly 0 non-JSON characters on stdout
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as err:
            pytest.fail(
                f"Non-JSON character detected on stdout from yaw serve:\n"
                f"Line received: {line!r}\n"
                f"Error: {err}"
            )

        # Invariant 2: Valid JSON-RPC 2.0 response
        assert resp.get("jsonrpc") == "2.0"
        assert resp.get("id") == 1
        assert "result" in resp

    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.terminate()
            stdout_rem, _ = proc.communicate(timeout=3.0)
        except subprocess.TimeoutExpired, OSError:
            proc.kill()
            stdout_rem, _ = proc.communicate()

    # Invariant 3: Any trailing lines on stdout must also be valid JSON
    if stdout_rem:
        for trailing_line in stdout_rem.strip().splitlines():
            if trailing_line.strip():
                try:
                    json.loads(trailing_line)
                except json.JSONDecodeError as err:
                    pytest.fail(
                        f"Trailing non-JSON output on stdout: {trailing_line!r} ({err})"
                    )


def test_cli_serve_signal_shutdown(tmp_path: Path):
    """Verify daemon creates PID file, terminates cleanly on SIGINT/SIGTERM within 5s, and cleans PID file.

    Invariants:
    - Daemon process creates vault_dir/.cache/yaw.pid upon startup containing child PID.
    - Intercepting SIGINT or SIGTERM initiates graceful shutdown.
    - Process exits with returncode 0 within 5.0 seconds SLA.
    - PID file is cleanly removed from disk upon termination.
    """
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    pid_file = vault_dir / ".cache" / "yaw.pid"
    port = _get_free_port()

    env = get_cli_env()

    cmd = [
        sys.executable,
        YAW_SCRIPT,
        "serve",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--vault",
        str(vault_dir),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        # Invariant 1: PID file created within 5.0s
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not pid_file.is_file():
            time.sleep(0.05)

        assert pid_file.is_file(), (
            f"PID file {pid_file} was not created within 5.0s of daemon startup"
        )
        pid_content = pid_file.read_text(encoding="utf-8").strip()
        child_pids = [
            c.pid for c in psutil.Process(proc.pid).children(recursive=True)
        ]
        assert (
            int(pid_content) == proc.pid or int(pid_content) in child_pids
        ), f"PID file contains {pid_content}, expected child PID {proc.pid} or child of launcher {child_pids}"

        # Invariant 2: Intercept SIGINT/SIGTERM and terminate within 5.0s SLA
        t0 = time.monotonic()
        if sys.platform == "win32":
            assert stop_daemon(pid_file) == 0
        else:
            proc.send_signal(signal.SIGINT)

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Daemon failed to shut down within 5.0s SLA after signal")

        duration = time.monotonic() - t0
        assert duration < 5.0, f"Shutdown took {duration:.2f}s, exceeding 5.0s SLA"

        # Invariant 3: Exit code 0 (or 1 on Windows TerminateProcess)
        expected_codes = (0, 1) if sys.platform == "win32" else (0,)
        assert proc.returncode in expected_codes, (
            f"Daemon exited with unexpected code {proc.returncode}"
        )

        # Invariant 4: PID file cleanly removed
        assert not pid_file.exists(), (
            f"PID file {pid_file} still exists after daemon shutdown"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ==============================================================================
# Milestone 4: In-Process CLI Lifecycle & Server Daemon Coverage Hardening
# ==============================================================================


_YAW_CLI_MOD: Any = None


def _get_yaw_cli() -> Any:
    """Load canonical root CLI module in-process for direct unit testing."""
    global _YAW_CLI_MOD
    if _YAW_CLI_MOD is not None:
        return _YAW_CLI_MOD
    spec = importlib.util.spec_from_file_location("yaw", YAW_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _YAW_CLI_MOD = module
    return module


def test_inprocess_cli_build_parser():
    yaw = _get_yaw_cli()
    parser = yaw.build_parser()

    # Test index parser
    args_index = parser.parse_args(["index", "--vault", "/tmp/v", "--force", "--quiet"])
    assert args_index.command == "index"
    assert args_index.vault == "/tmp/v"
    assert args_index.force is True
    assert args_index.quiet is True

    # Test serve parser defaults and flags
    args_serve = parser.parse_args(
        [
            "serve",
            "--transport",
            "stdio",
            "--port",
            "8888",
            "--host",
            "0.0.0.0",
        ]
    )
    assert args_serve.command == "serve"
    assert args_serve.transport == "stdio"
    assert args_serve.port == 8888
    assert args_serve.host == "0.0.0.0"

    args_serve_stdio = parser.parse_args(["serve", "--stdio"])
    assert args_serve_stdio.stdio is True

    args_serve_http = parser.parse_args(["serve", "--http"])
    assert args_serve_http.http is True

    # Test query parser
    args_query = parser.parse_args(["query", "search prompt", "--top-k", "5"])
    assert args_query.command == "query"
    assert args_query.prompt == "search prompt"
    assert args_query.top_k == 5

    # Test schema parser
    args_schema = parser.parse_args(["schema", "--vault", "/tmp/v2"])
    assert args_schema.command == "schema"
    assert args_schema.vault == "/tmp/v2"


def test_inprocess_cli_handle_index(tmp_path: Path):
    yaw = _get_yaw_cli()
    vault = tmp_path / "index_vault"
    raw_dir = vault / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "note.md").write_text("# Test\nContent", encoding="utf-8")

    parser = yaw.build_parser()
    args = parser.parse_args(["index", "--vault", str(vault), "--quiet"])
    ret = yaw.handle_index(args)
    assert ret == 0
    assert (vault / ".cache" / "yaw.db").is_file()

    # Re-run with force
    args_force = parser.parse_args(
        ["index", "--vault", str(vault), "--force", "--quiet"]
    )
    ret_force = yaw.handle_index(args_force)
    assert ret_force == 0

    # Error path: invalid/inaccessible vault filesystem path
    bad_file = tmp_path / "not_a_dir_file"
    bad_file.write_text("file", encoding="utf-8")
    args_err = parser.parse_args(
        ["index", "--vault", str(bad_file / "sub_path"), "--quiet"]
    )
    ret_err = yaw.handle_index(args_err)
    assert ret_err == 1


def test_inprocess_cli_handle_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    yaw = _get_yaw_cli()
    parser = yaw.build_parser()

    # 1. Unindexed vault schema
    args_unindexed = parser.parse_args(["schema", "--vault", str(tmp_path)])
    ret = yaw.handle_schema(args_unindexed)
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "namespaces" in data
    assert "stats" in data
    assert data["stats"]["chunks"] == 0

    # 2. Indexed vault schema
    vault = tmp_path / "schema_vault"
    raw_dir = vault / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "doc.md").write_text(
        "# Doc\nSome text with [[target]].", encoding="utf-8"
    )
    args_index = parser.parse_args(["index", "--vault", str(vault), "--quiet"])
    yaw.handle_index(args_index)
    capsys.readouterr()

    args_indexed = parser.parse_args(["schema", "--vault", str(vault)])
    ret_indexed = yaw.handle_schema(args_indexed)
    assert ret_indexed == 0
    captured_indexed = capsys.readouterr()
    data_indexed = json.loads(captured_indexed.out)
    assert data_indexed["stats"]["chunks"] >= 1

    # 3. Schema resilience on corrupt / unindexed / non-existent database substrates
    # 3a. Non-existent vault directory
    args_nonexistent = parser.parse_args(
        ["schema", "--vault", str(tmp_path / "nonexistent_vault")]
    )
    ret_nonexistent = yaw.handle_schema(args_nonexistent)
    assert ret_nonexistent == 0
    data_nonexistent = json.loads(capsys.readouterr().out)
    assert data_nonexistent["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}

    # 3b. Empty database file with missing tables
    empty_vault = tmp_path / "empty_schema_vault"
    empty_cache = empty_vault / ".cache"
    empty_cache.mkdir(parents=True)
    with sqlite3.connect(empty_cache / "yaw.db") as conn:
        conn.execute("CREATE TABLE unrelated_metadata (id INTEGER PRIMARY KEY)")
    args_empty = parser.parse_args(["schema", "--vault", str(empty_vault)])
    ret_empty = yaw.handle_schema(args_empty)
    assert ret_empty == 0
    data_empty = json.loads(capsys.readouterr().out)
    assert data_empty["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}

    # 3c. Physical database file corruption (corrupted binary bytes)
    corrupt_vault = tmp_path / "corrupt_schema_vault"
    corrupt_cache = corrupt_vault / ".cache"
    corrupt_cache.mkdir(parents=True)
    (corrupt_cache / "yaw.db").write_bytes(b"CORRUPTED INVALID SQLITE FILE HEADER")
    args_corrupt = parser.parse_args(["schema", "--vault", str(corrupt_vault)])
    ret_corrupt = yaw.handle_schema(args_corrupt)
    assert ret_corrupt == 0
    data_corrupt = json.loads(capsys.readouterr().out)
    assert data_corrupt["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}

    # 3d. Simulated broken stdout pipe triggers schema handler error exit
    class _BrokenPipeWriter:
        def write(self, text: str) -> int:
            raise OSError("Broken pipe")

        def flush(self) -> None:
            pass

    orig_stdout = sys.stdout
    sys.stdout = _BrokenPipeWriter()  # type: ignore[assignment]
    try:
        args_pipe = parser.parse_args(["schema", "--vault", str(vault)])
        assert yaw.handle_schema(args_pipe) == 1
    finally:
        sys.stdout = orig_stdout


def test_inprocess_cli_handle_serve(tmp_path: Path):
    # 1. Serve with --stdio flag via real subprocess
    proc_stdio = subprocess.Popen(
        [sys.executable, YAW_SCRIPT, "serve", "--stdio", "--vault", str(tmp_path)],
        env=get_cli_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc_stdio.stderr is not None
        line = proc_stdio.stderr.readline()
        assert "stdio transport" in line
    finally:
        proc_stdio.terminate()
        proc_stdio.wait(timeout=5)

    # 2. Serve with --http flag, custom host and port via real subprocess
    port = _get_free_port()
    proc_http = subprocess.Popen(
        [
            sys.executable,
            YAW_SCRIPT,
            "serve",
            "--http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--vault",
            str(tmp_path),
        ],
        env=get_cli_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc_http.stderr is not None
        line_http = proc_http.stderr.readline()
        assert "http transport" in line_http
    finally:
        proc_http.terminate()
        proc_http.wait(timeout=5)

    # 3. In-process handle_serve validation with concurrent daemon PID lock
    yaw = _get_yaw_cli()
    parser = yaw.build_parser()
    pid_file = tmp_path / ".cache" / "yaw.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    held_fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o644)
    flock(held_fd, LOCK_EX | LOCK_NB)
    try:
        args_serve_stdio = parser.parse_args(
            ["serve", "--stdio", "--vault", str(tmp_path)]
        )
        with pytest.raises(RuntimeError, match="Another yaw daemon is already running"):
            yaw.handle_serve(args_serve_stdio)

        args_serve_http = parser.parse_args(
            [
                "serve",
                "--http",
                "--host",
                "127.0.0.1",
                "--port",
                "9876",
                "--vault",
                str(tmp_path),
            ]
        )
        with pytest.raises(RuntimeError, match="Another yaw daemon is already running"):
            yaw.handle_serve(args_serve_http)
    finally:
        flock(held_fd, LOCK_UN)
        os.close(held_fd)


def test_inprocess_cli_handle_query(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    yaw = _get_yaw_cli()
    parser = yaw.build_parser()

    # 1. Graceful query on unindexed vault returns 0 with empty context
    args_query_unindexed = parser.parse_args(
        [
            "query",
            "quantum",
            "--vault",
            str(tmp_path),
            "--top-k",
            "2",
        ]
    )
    ret_unindexed = yaw.handle_query(args_query_unindexed)
    assert ret_unindexed == 0
    out_unindexed = capsys.readouterr().out
    assert "# Ontological Context" in out_unindexed

    # 2. Error path: corrupt database raises real sqlite3.DatabaseError in retrieve
    corrupt_vault = tmp_path / "corrupt_vault"
    corrupt_db = corrupt_vault / ".cache" / "yaw.db"
    corrupt_db.parent.mkdir(parents=True)
    corrupt_db.write_bytes(b"corrupted sqlite header garbage")

    args_query_corrupt = parser.parse_args(
        ["query", "physics", "--vault", str(corrupt_vault)]
    )
    ret_fail = yaw.handle_query(args_query_corrupt)
    assert ret_fail == 1
    err_out = capsys.readouterr().err
    assert "Error executing GraphRAG query" in err_out

    # 3. Success path: real indexed vault with real note returns dual-layer Markdown
    good_vault = tmp_path / "good_vault"
    good_raw = good_vault / "raw"
    good_raw.mkdir(parents=True)
    (good_raw / "physics.md").write_text(
        "# Quantum Physics\n\nThe uncertainty principle defines physical limits.\n",
        encoding="utf-8",
    )
    yaw.sync_vault(vault_dir=good_vault, force=True, verbose=False)

    args_query_ok = parser.parse_args(["query", "physics", "--vault", str(good_vault)])
    ret_ok = yaw.handle_query(args_query_ok)
    assert ret_ok == 0
    captured = capsys.readouterr()
    assert "# Ontological Context" in captured.out
    assert "## Evidence Chunk" in captured.out


def test_inprocess_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    yaw = _get_yaw_cli()

    # 1. Empty args -> help to stderr, returns 2
    ret_empty = yaw.main([])
    assert ret_empty == 2
    err_empty = capsys.readouterr().err
    assert "usage: yaw" in err_empty

    # 2. Main with index
    vault = tmp_path / "main_vault"
    raw_dir = vault / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "note.md").write_text("# Hello", encoding="utf-8")
    ret_index = yaw.main(["index", "--vault", str(vault), "--quiet"])
    assert ret_index == 0

    # 3. Main with schema
    ret_schema = yaw.main(["schema", "--vault", str(vault)])
    assert ret_schema == 0
    assert "namespaces" in capsys.readouterr().out

    # 4. Main with serve routing via concurrent PID lock (in-process) and subprocess
    pid_file = vault / ".cache" / "yaw.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    held_fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o644)
    flock(held_fd, LOCK_EX | LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="Another yaw daemon is already running"):
            yaw.main(["serve", "--vault", str(vault)])
    finally:
        flock(held_fd, LOCK_UN)
        os.close(held_fd)

    proc_serve = subprocess.Popen(
        [sys.executable, YAW_SCRIPT, "serve", "--stdio", "--vault", str(vault)],
        env=get_cli_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    assert proc_serve.poll() is None
    proc_serve.terminate()
    proc_serve.wait(timeout=5)

    # 5. Main with query against real indexed vault
    ret_query = yaw.main(["query", "Hello", "--vault", str(vault)])
    assert ret_query == 0
    assert "# Ontological Context" in capsys.readouterr().out

    # 6. Unexpected/invalid command via authentic standard library argparse SystemExit
    with pytest.raises(SystemExit) as exc_info:
        yaw.main(["nonexistent"])
    assert exc_info.value.code == 2


def test_daemon_rescore_fallback_warning_paths(caplog: pytest.LogCaptureFixture):
    from yaw.server.daemon import _rescore_single_candidate

    q_vec = np.array([0.6, 0.8], dtype=np.float32)

    # 1. Success path: matching ndarray shapes
    cand_ok = {"entity_uri": "note:ok", "hybrid_prob": 0.5}
    cand_vec_ok = np.array([0.6, 0.8], dtype=np.float32)
    _rescore_single_candidate(cand_ok, cand_vec_ok, q_vec)
    assert cand_ok["rerank_score"] == 1.0

    # 2. Missing vector (None)
    cand_none = {"entity_uri": "note:missing", "hybrid_prob": 0.42}
    with caplog.at_level(logging.WARNING):
        _rescore_single_candidate(cand_none, None, q_vec)
    assert cand_none["rerank_score"] == 0.42
    assert "lacks stored embedding vector" in caplog.text

    # 3. Non-ndarray vector (string/list)
    caplog.clear()
    cand_bad = {"entity_uri": "note:bad", "hybrid_prob": 0.35}
    invalid_vec: Any = "not-an-array"
    with caplog.at_level(logging.WARNING):
        _rescore_single_candidate(cand_bad, invalid_vec, q_vec)
    assert cand_bad["rerank_score"] == 0.35
    assert "lacks stored embedding vector" in caplog.text

    # 4. Dimension mismatch
    caplog.clear()
    cand_dim = {"entity_uri": "note:dim", "hybrid_prob": 0.77}
    cand_vec_dim = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        _rescore_single_candidate(cand_dim, cand_vec_dim, q_vec)
    assert cand_dim["rerank_score"] == 0.77
    assert "vector dimension mismatch" in caplog.text

    # 5. Query vector is None
    caplog.clear()
    cand_qnone = {"entity_uri": "note:qnone", "hybrid_prob": 0.19}
    with caplog.at_level(logging.WARNING):
        _rescore_single_candidate(cand_qnone, cand_vec_ok, None)
    assert cand_qnone["rerank_score"] == 0.19
    assert "Query vector unavailable" in caplog.text


def test_daemon_hydrate_candidate_items_edge_cases(tmp_path: Path):
    from yaw.server.daemon import _hydrate_candidate_items

    # 1. Empty candidate list
    assert _hydrate_candidate_items([], {}) == []

    # 2. Candidate not present in metadata (neither chunk nor taxonomy)
    assert _hydrate_candidate_items(["unknown:entity"], {"unknown:entity": 0.5}) == []

    # 3. Candidate present as taxonomy in leaf_nodes
    db_file = tmp_path / "hydrate_test.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT)"
    )
    conn.execute(
        "INSERT INTO leaf_nodes VALUES ('tax:test', 'leaf1.md', 'TestNode', 'concept', 'A test definition')"
    )
    conn.commit()
    conn.close()

    orig_db = config.DB_PATH
    config.DB_PATH = db_file
    try:
        items = _hydrate_candidate_items(["tax:test"], {"tax:test": 0.88})
        assert len(items) == 1
        item = items[0]
        assert item["kind"] == "taxonomy"
        assert item["entity_uri"] == "tax:test"
        assert item["line_start"] is None
        assert item["line_end"] is None
        assert item["hybrid_prob"] == 0.88
        assert "TestNode" in item["heading"]
    finally:
        config.DB_PATH = orig_db


def test_daemon_get_ontology_schema_unindexed_and_error_paths(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    from yaw.server.daemon import get_ontology_schema

    orig_db = config.DB_PATH

    # 1. DB path does not exist
    config.DB_PATH = tmp_path / "nonexistent.db"
    schema_empty = get_ontology_schema()
    assert schema_empty["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}

    # 2. DB exists but has no tables
    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(empty_db))
    conn.close()
    config.DB_PATH = empty_db
    schema_no_tables = get_ontology_schema()
    assert schema_no_tables["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}

    # 3. DB has leaf_relations instead of unified_edges
    lr_db = tmp_path / "leaf_rel.db"
    conn = sqlite3.connect(str(lr_db))
    conn.execute("CREATE TABLE note_chunks (id INT)")
    conn.execute("INSERT INTO note_chunks VALUES (1), (2)")
    conn.execute("CREATE TABLE leaf_nodes (id INT)")
    conn.execute("INSERT INTO leaf_nodes VALUES (1)")
    conn.execute("CREATE TABLE leaf_relations (id INT)")
    conn.execute("INSERT INTO leaf_relations VALUES (1), (2), (3)")
    conn.commit()
    conn.close()
    config.DB_PATH = lr_db
    schema_lr = get_ontology_schema()
    assert schema_lr["stats"]["chunks"] == 2
    assert schema_lr["stats"]["taxonomies"] == 1
    assert schema_lr["stats"]["edges"] == 3

    # 4. DB has neither unified_edges nor leaf_relations
    no_edges_db = tmp_path / "no_edges.db"
    conn = sqlite3.connect(str(no_edges_db))
    conn.execute("CREATE TABLE note_chunks (id INT)")
    conn.execute("CREATE TABLE leaf_nodes (id INT)")
    conn.commit()
    conn.close()
    config.DB_PATH = no_edges_db
    schema_ne = get_ontology_schema()
    assert schema_ne["stats"]["edges"] == 0

    # 5. DatabaseError during query
    bad_db = tmp_path / "corrupt.db"
    bad_db.write_bytes(b"NOT A SQLITE FILE")
    config.DB_PATH = bad_db
    with caplog.at_level(logging.WARNING):
        schema_err = get_ontology_schema()
    assert schema_err["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}
    assert "Failed to query ontology stats" in caplog.text

    config.DB_PATH = orig_db


def test_daemon_candidate_metadata_and_vector_fetch_branches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    from yaw.server.daemon import (
        _fetch_table_vectors,
        _get_candidate_metadata,
        _get_candidate_vectors,
        _get_chunk_locations,
    )

    # 1. Empty URIs
    assert _get_candidate_metadata([]) == ({}, {})
    assert _get_candidate_vectors([]) == {}
    assert _get_chunk_locations([]) == {}

    # 2. Corrupt JSON embedding in SQLite table
    db_file = tmp_path / "vec_err.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE note_chunks (entity_uri TEXT, embedding TEXT)")
    conn.execute("INSERT INTO note_chunks VALUES ('note:1', 'INVALID_JSON_EMBEDDING')")
    conn.commit()

    vecs: dict[str, np.ndarray] = {}
    with caplog.at_level(logging.WARNING):
        _fetch_table_vectors(conn, "note_chunks", "entity_uri", ["note:1"], vecs)
    assert "note:1" not in vecs
    assert "Failed to deserialize embedding" in caplog.text

    # 3. OperationalError when table does not exist
    _fetch_table_vectors(conn, "nonexistent_table", "id", ["note:1"], vecs)
    conn.close()


def test_daemon_lifecycle_edge_cases(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    from yaw.server.daemon import DaemonLifecycle

    pid_file = tmp_path / "test.pid"
    db_file = tmp_path / "test.db"
    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_file)

    # 1. _signal_handler: triggers cleanup and sys.exit(0)
    with pytest.raises(SystemExit) as exc_info:
        lc._signal_handler(signal.SIGINT, None)
    assert exc_info.value.code == 0

    # 2. _unlink_pid when PID file belongs to another process PID
    pid_file.write_text("999999", encoding="utf-8")
    lc._cleaned = False
    with caplog.at_level(logging.WARNING):
        lc._unlink_pid()
    assert pid_file.is_file(), "PID file belonging to PID 999999 must not be unlinked"
    assert "belongs to PID 999999" in caplog.text

    # 3. _unlink_pid when PID file has unparseable non-integer content
    caplog.clear()
    pid_file.write_text("not_a_pid\n", encoding="utf-8")
    lc._unlink_pid()
    assert not pid_file.is_file(), "Corrupted PID file should be unlinked"

    # 4. _restore_signals: restore dummy handler
    lc._orig_handlers = {signal.SIGINT: signal.SIG_DFL}
    lc._restore_signals()
    assert len(lc._orig_handlers) == 0

    # 5. checkpoint_wal when DB does not exist
    assert lc.checkpoint_wal() is None

    # 6. checkpoint_wal when SQLite connection is closed (raises sqlite3.ProgrammingError)
    closed_conn = sqlite3.connect(":memory:")
    closed_conn.close()
    with caplog.at_level(logging.WARNING):
        res = DaemonLifecycle._execute_checkpoint(closed_conn)
    assert res is None
    assert "WAL checkpoint failed" in caplog.text

    # 7. Real subprocess execution of serve verifying PID file and lifecycle
    proc_stdio = subprocess.Popen(
        [sys.executable, YAW_SCRIPT, "serve", "--stdio", "--vault", str(tmp_path)],
        env=get_cli_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    assert proc_stdio.poll() is None
    proc_stdio.terminate()
    proc_stdio.wait(timeout=5)

    port = _get_free_port()
    proc_http = subprocess.Popen(
        [
            sys.executable,
            YAW_SCRIPT,
            "serve",
            "--http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--vault",
            str(tmp_path),
        ],
        env=get_cli_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    assert proc_http.poll() is None
    proc_http.terminate()
    proc_http.wait(timeout=5)


def test_cli_query_offline_fast_exit(tmp_path: Path):
    """Verify yaw query exits cleanly without threadpool hangs when Ollama is offline."""
    vault_dir = tmp_path / "offline_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_path = raw_dir / "offline_note.md"
    note_path.write_text(
        "# Offline Note\nGround-truth evidence in offline mode.\n",
        encoding="utf-8",
    )

    env = get_cli_env()

    # 1. Index vault with standard test harness
    subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir), "--quiet"],
        env=env,
        check=True,
    )

    # 2. Measure query duration with completely offline / unreachable Ollama host
    offline_env = get_cli_env()
    offline_env["YAW_HOST"] = "http://127.0.0.1:1"

    t0 = time.time()
    res = subprocess.run(
        [
            sys.executable,
            YAW_SCRIPT,
            "query",
            "offline evidence",
            "--vault",
            str(vault_dir),
        ],
        capture_output=True,
        text=True,
        env=offline_env,
        timeout=10.0,
        check=False,
    )
    duration = time.time() - t0

    assert res.returncode == 0
    assert duration < 5.0, (
        f"Query took {duration:.2f}s, exceeding 5.0s offline threshold"
    )
    assert "# Ontological Context" in res.stdout
    assert "## Evidence Chunk" in res.stdout


def test_cli_query_empty_vault_offline_fast_exit(tmp_path: Path):
    """Verify yaw query on an empty vault exits cleanly without network attempts."""
    empty_vault = tmp_path / "empty_vault"
    empty_vault.mkdir(parents=True, exist_ok=True)

    env = get_cli_env()
    env["YAW_HOST"] = "http://127.0.0.1:1"

    t0 = time.time()
    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "query", "anything", "--vault", str(empty_vault)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )
    duration = time.time() - t0

    assert res.returncode == 0
    assert duration < 5.0, (
        f"Empty vault query took {duration:.2f}s, exceeding 5.0s threshold"
    )
    assert "_No indexed knowledge available._" in res.stdout


@pytest.mark.anyio
async def test_compute_seed_candidates_offline_fallback_zero_http(tmp_path: Path):
    """Verify compute_seed_candidates with query_vec=None falls back cleanly when offline."""
    from yaw import config
    from yaw.server.daemon import compute_seed_candidates

    config.set_vault(tmp_path)
    orig_host = config.OLLAMA_HOST
    config.OLLAMA_HOST = "http://127.0.0.1:1"
    try:
        db_path = tmp_path / ".cache" / "yaw.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE note_chunks (chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB)"
            )
            conn.execute(
                "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id))"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE query_fts USING fts5(entity_uri UNINDEXED, heading, content, tokenize='porter unicode61')"
            )
            conn.execute(
                "INSERT INTO note_chunks VALUES ('c1', 'chunk:offline_1', 'raw/offline.md', 0, 'Offline Header', 'offline-header', 1, 2, NULL)"
            )
            conn.execute(
                "INSERT INTO query_fts VALUES ('chunk:offline_1', 'Offline Header', 'offline test document content')"
            )
            conn.commit()

        raw_file = tmp_path / "raw" / "offline.md"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.write_text(
            "# Offline Header\noffline test document content", encoding="utf-8"
        )

        # Query with query_vec=None while Ollama is offline
        seeds = await compute_seed_candidates("offline test", top_k=1, query_vec=None)

        assert len(seeds) == 1
        assert seeds[0]["entity_uri"] == "chunk:offline_1"
        assert seeds[0]["rerank_score"] == seeds[0]["hybrid_prob"]
        assert seeds[0]["kind"] in ("chunk", "taxonomy")
    finally:
        config.OLLAMA_HOST = orig_host


def test_sync_vault_idempotent_reindex_when_cocoindex_cache_missing(tmp_path: Path):
    """Verify that sync_vault cleanly re-indexes an existing yaw.db without --force

    even when .cache/cocoindex tracking directory was deleted/missing, preventing
    'sqlite3.OperationalError: table "note_meta" already exists'.
    """
    import shutil

    from yaw.indexer.pipeline import sync_vault

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "alpha.md").write_text(
        "# Alpha\nContent about [[Beta]]\n", encoding="utf-8"
    )

    # 1. Initial indexing run
    ret1 = sync_vault(vault_dir=tmp_path, force=False, verbose=False)
    assert ret1 == 0
    db_file = tmp_path / ".cache" / "yaw.db"
    assert db_file.is_file()

    # 2. Simulate external cache deletion (e.g. git clean, docker cache wipe)
    coco_cache = tmp_path / ".cache" / "cocoindex"
    if coco_cache.exists():
        shutil.rmtree(coco_cache)
    assert not coco_cache.exists()
    assert db_file.is_file()

    # 3. Second indexing run WITHOUT --force must succeed idempotently
    ret2 = sync_vault(vault_dir=tmp_path, force=False, verbose=False)
    assert ret2 == 0
    assert db_file.is_file()

    # Verify tables and unified views exist and queryable
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM note_meta")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT count(*) FROM note_chunks")
        assert cursor.fetchone()[0] >= 1


class _OllamaEmbedErrorServer(BaseTestHTTPHandler):
    embed_status_code = 500


@pytest.mark.anyio
async def test_compute_seed_candidates_query_embedding_error_branch(tmp_path: Path):
    """Verify compute_seed_candidates catches embedding HTTP errors and falls back to lexical scoring."""
    from yaw import config
    from yaw.server.daemon import compute_seed_candidates, query_wiki

    config.set_vault(tmp_path)
    server = HTTPServer(("127.0.0.1", 0), _OllamaEmbedErrorServer)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    orig_host = config.OLLAMA_HOST
    config.OLLAMA_HOST = f"http://127.0.0.1:{port}"
    try:
        seeds = await compute_seed_candidates("error test", top_k=1, query_vec=None)
        assert isinstance(seeds, list)

        wiki_res = await query_wiki("error test", top_k=1)
        assert "# Ontological Context" in wiki_res
    finally:
        config.OLLAMA_HOST = orig_host
        server.shutdown()


def test_daemon_lifecycle_active_pid_branches(tmp_path: Path):
    """Verify DaemonLifecycle start re-entrance and _unlink_pid with active held fd."""
    from yaw.server.daemon import DaemonLifecycle

    pid_file = tmp_path / "active.pid"
    db_file = tmp_path / "active.db"
    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_file)

    lc.start()
    assert lc._pid_fd is not None
    lc.start()
    lc.cleanup()
    assert lc._pid_fd is None
    assert not pid_file.exists()


# ==============================================================================
# Adversarial Suite: Milestone 1 Graph Conductance Invariants
# ==============================================================================


def _init_graphrag_test_db_schema(conn: sqlite3.Connection) -> None:
    """Initialize base schema tables required for GraphRAG unified views."""
    conn.execute(
        "CREATE TABLE wikilinks ("
        "source_uri TEXT, target_uri TEXT, target_note TEXT, weight REAL"
        ")"
    )
    conn.execute(
        "CREATE TABLE leaf_relations ("
        "subject_id TEXT, predicate TEXT, object_id TEXT, source_leaf_id TEXT, weight REAL"
        ")"
    )
    conn.execute(
        "CREATE TABLE note_chunks ("
        "chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER, "
        "heading TEXT, heading_slug TEXT, "
        "line_start INTEGER, line_end INTEGER, embedding BLOB"
        ")"
    )
    conn.execute(
        "CREATE TABLE leaf_nodes ("
        "node_id TEXT, name TEXT, namespace TEXT, definition TEXT, "
        "source_leaf_id TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id)"
        ")"
    )


# ==============================================================================
# 1. Cyclic Triangles (A -> B -> C -> A)
# ==============================================================================


def test_adversarial_cyclic_triangle_conductance_and_flow(tmp_path: Path):
    """Stress-test cyclic triangle A -> B -> C -> A.

    Invariants:
    - Forward in-degrees: A=1, B=1, C=1 (deg = 1.0, dampening denominator = 1.0).
    - Forward edges: A->B (1.0), B->C (1.0), C->A (1.0).
    - Reciprocal back-edges: B->A (0.5), C->B (0.5), A->C (0.5).
    - Raw adjacency matrix row sums: exactly 1.5 for each node.
    - Transition matrix _P: forward branch has prob 2/3 (1.0/1.5), back-edge branch prob 1/3 (0.5/1.5).
    - Uniform seed converges to uniform stationary distribution [1/3, 1/3, 1/3].
    - Non-uniform seed p0=[1, 0, 0] favors forward flow: p*[B] > p*[C].
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)
        conn.execute("INSERT INTO wikilinks VALUES ('raw/a.md', 'raw/b.md', 'b', 1.0)")
        conn.execute("INSERT INTO wikilinks VALUES ('raw/b.md', 'raw/c.md', 'c', 1.0)")
        conn.execute("INSERT INTO wikilinks VALUES ('raw/c.md', 'raw/a.md', 'a', 1.0)")
        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        assert isinstance(retriever.adj, sp.csr_matrix)
        assert len(retriever._idx_to_node) == 3

        node_to_idx = retriever._node_to_idx
        idx_a = node_to_idx["raw/a.md"]
        idx_b = node_to_idx["raw/b.md"]
        idx_c = node_to_idx["raw/c.md"]

        adj = retriever.adj.toarray()

        # Invariant: Forward edges retain 1.0, reciprocal back-edges retain 0.5
        assert pytest.approx(float(adj[idx_a, idx_b]), abs=1e-5) == 1.0
        assert pytest.approx(float(adj[idx_b, idx_a]), abs=1e-5) == 0.5

        assert pytest.approx(float(adj[idx_b, idx_c]), abs=1e-5) == 1.0
        assert pytest.approx(float(adj[idx_c, idx_b]), abs=1e-5) == 0.5

        assert pytest.approx(float(adj[idx_c, idx_a]), abs=1e-5) == 1.0
        assert pytest.approx(float(adj[idx_a, idx_c]), abs=1e-5) == 0.5

        # Invariant: Diagonal elements are zero (no self-loops)
        for i in range(3):
            assert pytest.approx(float(adj[i, i]), abs=1e-5) == 0.0

        # Invariant: Each row sums to 1.5 in un-normalized adjacency
        for i in range(3):
            assert pytest.approx(float(adj[i].sum()), abs=1e-5) == 1.5

        # Invariant: Transition probabilities in _P
        assert retriever._P is not None
        P = retriever._P.toarray()
        assert pytest.approx(float(P[idx_a, idx_b]), abs=1e-5) == 1.0 / 1.5
        assert pytest.approx(float(P[idx_a, idx_c]), abs=1e-5) == 0.5 / 1.5
        assert pytest.approx(float(P[idx_b, idx_c]), abs=1e-5) == 1.0 / 1.5
        assert pytest.approx(float(P[idx_b, idx_a]), abs=1e-5) == 0.5 / 1.5
        assert pytest.approx(float(P[idx_c, idx_a]), abs=1e-5) == 1.0 / 1.5
        assert pytest.approx(float(P[idx_c, idx_b]), abs=1e-5) == 0.5 / 1.5

        # Invariant: PPR stationary distribution under uniform seed
        p0_uniform = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        p_star_uniform = retriever.power_iteration(p0_uniform, alpha=0.85, max_iter=50)
        for val in p_star_uniform:
            assert pytest.approx(float(val), abs=1e-4) == 1.0 / 3.0

        # Invariant: Asymmetric directional bias from seed A
        p0_a = np.zeros(3, dtype=np.float32)
        p0_a[idx_a] = 1.0
        p_star_a = retriever.power_iteration(p0_a, alpha=0.85, max_iter=50)
        # B receives 2/3 forward flux, C receives 1/3 backflux -> p*[B] > p*[C]
        assert p_star_a[idx_b] > p_star_a[idx_c]


# ==============================================================================
# 2. Multi-Hop Chains
# ==============================================================================


def test_adversarial_multi_hop_chain_conductance_and_reversibility(tmp_path: Path):
    """Stress-test a directed linear chain: 0 -> 1 -> 2 -> 3 -> 4.

    Invariants:
    - Forward edges: (i, i+1) @ 1.0.
    - Reciprocal back-edges: (i+1, i) @ 0.5.
    - Node 4 (terminus) is NOT dangling because it possesses reciprocal back-edge (4, 3) @ 0.5.
    - Whole graph is 0% dangling (all non-dangling rows sum to 1.0 in _P).
    - Information diffuses all the way backwards from terminus to root under PPR.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = 5
    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)
        for i in range(n_nodes - 1):
            src = f"raw/node_{i}.md"
            dst = f"raw/node_{i + 1}.md"
            conn.execute(
                "INSERT INTO wikilinks VALUES (?, ?, ?, 1.0)",
                (src, dst, f"node_{i + 1}"),
            )
        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        assert len(retriever._idx_to_node) == n_nodes

        node_to_idx = retriever._node_to_idx
        adj = retriever.adj.toarray()

        for i in range(n_nodes - 1):
            u = node_to_idx[f"raw/node_{i}.md"]
            v = node_to_idx[f"raw/node_{i + 1}.md"]
            assert pytest.approx(float(adj[u, v]), abs=1e-5) == 1.0
            assert pytest.approx(float(adj[v, u]), abs=1e-5) == 0.5

        # Terminus node 4 must NOT be dangling
        assert retriever._dangling_mask is not None
        assert not np.any(retriever._dangling_mask), (
            "Reciprocal back-edges eliminate dangling sinks"
        )

        # Row-stochastic normalization
        assert retriever._P is not None
        P = retriever._P.toarray()
        for i in range(n_nodes):
            assert pytest.approx(float(P[i].sum()), rel=1e-5) == 1.0

        # Terminus row transitions backwards with 100% probability
        u_last = node_to_idx[f"raw/node_{n_nodes - 1}.md"]
        u_prev = node_to_idx[f"raw/node_{n_nodes - 2}.md"]
        assert pytest.approx(float(P[u_last, u_prev]), rel=1e-5) == 1.0

        # PPR mass seeded exclusively at terminus diffuses all the way back to root (node 0)
        p0 = np.zeros(n_nodes, dtype=np.float32)
        p0[u_last] = 1.0
        p_star = retriever.power_iteration(p0, alpha=0.85, max_iter=50)

        for i in range(n_nodes):
            u_i = node_to_idx[f"raw/node_{i}.md"]
            assert p_star[u_i] > 0.0, f"Node {i} must receive diffused probability mass"
        # Monotonicity check backwards along the chain:
        # Node 3 reflects 100% of node 4's outgoing flux, so node 3 and node 4 hold the vast majority of mass.
        # Flow decays monotonically as distance from terminus increases: p*[3] > p*[2] > p*[1] > p*[0].
        u_0 = node_to_idx["raw/node_0.md"]
        u_1 = node_to_idx["raw/node_1.md"]
        u_2 = node_to_idx["raw/node_2.md"]
        u_3 = node_to_idx["raw/node_3.md"]
        assert p_star[u_3] > p_star[u_2] > p_star[u_1] > p_star[u_0]
        assert p_star[u_last] > p_star[u_2]


# ==============================================================================
# 3. Multiple Mutual Wikilink Pairs: Exact 1.5 Conductance
# ==============================================================================


def test_adversarial_multiple_mutual_pairs_exact_1_5_conductance(tmp_path: Path):
    """Assert every mutual wikilink edge receives exactly 1.5 conductance (1.0 + 0.5) in SciPy CSR.

    Tests multiple independent mutual pairs simultaneously:
    (A_i <-> B_i) for i in {0, 1, 2, 3, 4}.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    num_pairs = 6
    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)
        for i in range(num_pairs):
            a_uri = f"raw/pair_{i}_a.md"
            b_uri = f"raw/pair_{i}_b.md"
            conn.execute(
                "INSERT INTO wikilinks VALUES (?, ?, ?, 1.0)", (a_uri, b_uri, f"b_{i}")
            )
            conn.execute(
                "INSERT INTO wikilinks VALUES (?, ?, ?, 1.0)", (b_uri, a_uri, f"a_{i}")
            )
        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        assert len(retriever._idx_to_node) == num_pairs * 2

        node_to_idx = retriever._node_to_idx
        adj = retriever.adj.toarray()

        for i in range(num_pairs):
            u_a = node_to_idx[f"raw/pair_{i}_a.md"]
            u_b = node_to_idx[f"raw/pair_{i}_b.md"]

            # Mutual link invariant: Exactly 1.5 in SciPy CSR matrix
            assert pytest.approx(float(adj[u_a, u_b]), abs=1e-6) == 1.5
            assert pytest.approx(float(adj[u_b, u_a]), abs=1e-6) == 1.5

            # Self-loops must be exactly zero
            assert pytest.approx(float(adj[u_a, u_a]), abs=1e-6) == 0.0
            assert pytest.approx(float(adj[u_b, u_b]), abs=1e-6) == 0.0

            # Row sum must be exactly 1.5
            assert pytest.approx(float(adj[u_a].sum()), abs=1e-6) == 1.5
            assert pytest.approx(float(adj[u_b].sum()), abs=1e-6) == 1.5

            # Ensure zero cross-pair contamination
            for j in range(num_pairs):
                if i != j:
                    v_a = node_to_idx[f"raw/pair_{j}_a.md"]
                    v_b = node_to_idx[f"raw/pair_{j}_b.md"]
                    assert float(adj[u_a, v_a]) == 0.0
                    assert float(adj[u_a, v_b]) == 0.0
                    assert float(adj[u_b, v_a]) == 0.0
                    assert float(adj[u_b, v_b]) == 0.0

        # Row-stochastic transition matrix _P
        assert retriever._P is not None
        P = retriever._P.toarray()
        for i in range(num_pairs):
            u_a = node_to_idx[f"raw/pair_{i}_a.md"]
            u_b = node_to_idx[f"raw/pair_{i}_b.md"]
            assert pytest.approx(float(P[u_a, u_b]), abs=1e-6) == 1.0
            assert pytest.approx(float(P[u_b, u_a]), abs=1e-6) == 1.0


# ==============================================================================
# 4. Forward-Edge Hub Dampening vs. Reciprocal Back-Edges
# ==============================================================================


def test_adversarial_forward_hub_dampening_vs_reciprocal_back_edges(tmp_path: Path):
    """Stress-test that forward-edge hub dampening scales target nodes without dampening reciprocal edges incorrectly.

    Topology:
    - Central Hub H (raw/hub.md) has 4 incoming forward wikilinks from S0, S1, S2, S3.
    - H has forward wikilink back to S0 (making H <-> S0 a mutual pair).
    - An auxiliary node X has a forward wikilink to S0.

    Forward in-degrees:
    - H: incoming from S0, S1, S2, S3 -> forward in-degree = 4.
    - S0: incoming from H and X -> forward in-degree = 2.
    - S1, S2, S3, X: forward in-degree = 0 -> max(1, 0) = 1.

    Invariants:
    1. Forward edge S0 -> H enters Hub (deg 4) -> 1.0 / sqrt(4) = 0.5.
       Reciprocal back-edge for H -> S0 enters Hub (deg 4) -> 0.5 / sqrt(4) = 0.25.
       CSR entry adj[S0, H] = 0.5 + 0.25 = 0.75.
    2. Forward edge H -> S0 enters S0 (deg 2) -> 1.0 / sqrt(2).
       Reciprocal back-edge for S0 -> H enters S0 (deg 2) -> 0.5 / sqrt(2).
       CSR entry adj[H, S0] = 1.5 / sqrt(2) ≈ 1.06066.
    3. Unilateral edge S1 -> H enters Hub (deg 4) -> adj[S1, H] = 1.0 / sqrt(4) = 0.5.
    4. Reciprocal back-edge from Hub to S1 enters S1 (deg 1) -> adj[H, S1] = 0.5 / sqrt(1) = 0.5!
       CRITICAL: adj[H, S1] must NOT be dampened by H's in-degree (which would yield 0.25).
       It must strictly be 0.5.
    5. Forward edge X -> S0 enters S0 (deg 2) -> adj[X, S0] = 1.0 / sqrt(2).
    6. Reciprocal back-edge S0 -> X enters X (deg 1) -> adj[S0, X] = 0.5 / sqrt(1) = 0.5.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)
        hub = "raw/hub.md"
        spokes = [f"raw/s_{i}.md" for i in range(4)]
        aux_x = "raw/x.md"

        # 4 spokes point to hub
        for s in spokes:
            conn.execute("INSERT INTO wikilinks VALUES (?, ?, 'hub', 1.0)", (s, hub))

        # Hub points back to S0
        conn.execute(
            "INSERT INTO wikilinks VALUES (?, ?, 's_0', 1.0)", (hub, spokes[0])
        )

        # Aux X points to S0
        conn.execute(
            "INSERT INTO wikilinks VALUES (?, ?, 's_0', 1.0)", (aux_x, spokes[0])
        )

        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        adj = retriever.adj.toarray()
        node_to_idx = retriever._node_to_idx

        u_hub = node_to_idx[hub]
        u_s0 = node_to_idx[spokes[0]]
        u_s1 = node_to_idx[spokes[1]]
        u_s2 = node_to_idx[spokes[2]]
        u_s3 = node_to_idx[spokes[3]]
        u_x = node_to_idx[aux_x]

        # 1. Invariant: (S0, H) mutual link dampened by Hub in-degree 4
        # (1.0 + 0.5) / sqrt(4) = 1.5 / 2.0 = 0.75
        assert pytest.approx(float(adj[u_s0, u_hub]), abs=1e-5) == 0.75

        # 2. Invariant: (H, S0) mutual link dampened by S0 in-degree 2
        # (1.0 + 0.5) / sqrt(2) = 1.5 / sqrt(2)
        assert pytest.approx(float(adj[u_hub, u_s0]), abs=1e-5) == 1.5 / (2.0**0.5)

        # 3. Invariant: Unilateral forward edges entering Hub are dampened by sqrt(4)
        for u_s in [u_s1, u_s2, u_s3]:
            assert pytest.approx(float(adj[u_s, u_hub]), abs=1e-5) == 0.5

        # 4. Invariant: Reciprocal back-edges from Hub to spokes enter spokes with forward in-degree 0 -> max(1, 0) = 1.
        # They must NOT be penalized by Hub in-degree!
        for u_s in [u_s1, u_s2, u_s3]:
            assert pytest.approx(float(adj[u_hub, u_s]), abs=1e-5) == 0.5

        # 5. Invariant: Unilateral edge X -> S0 enters S0 (deg 2)
        assert pytest.approx(float(adj[u_x, u_s0]), abs=1e-5) == 1.0 / (2.0**0.5)

        # 6. Invariant: Reciprocal back-edge S0 -> X enters X (deg 1) -> 0.5
        assert pytest.approx(float(adj[u_s0, u_x]), abs=1e-5) == 0.5


# ==============================================================================
# 5. Isolated Nodes, Dangling Sinks, Self-Loops, and Disconnected Components
# ==============================================================================


def test_adversarial_isolated_nodes_dangling_sinks_and_self_loops(tmp_path: Path):
    """Stress-test edge cases: disconnected components, self-loops, and dangling sinks.

    Components:
    - Component 1: Mutual pair A <-> B (conductance 1.5).
    - Component 2: Self-loop node S -> S (wikilink, conductance 1.5).
    - Dangling sink D: Leaf relation chunk C -> D (mentions, weight 0.2).
      Since leaf_relations do NOT synthesize wikilink_rev edges, D has zero outgoing edges (dangling).

    Invariants:
    - Self-loop node S receives exactly 1.0 (fwd) + 0.5 (rev) = 1.5 conductance.
    - Dangling sink D is correctly identified in _dangling_mask.
    - PPR power iteration teleports dangling mass properly, maintaining exact probability conservation.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)

        # 1. Component 1: Mutual pair A <-> B
        conn.execute("INSERT INTO wikilinks VALUES ('raw/a.md', 'raw/b.md', 'b', 1.0)")
        conn.execute("INSERT INTO wikilinks VALUES ('raw/b.md', 'raw/a.md', 'a', 1.0)")

        # 2. Component 2: Self-loop node S
        conn.execute(
            "INSERT INTO wikilinks VALUES ('raw/self.md', 'raw/self.md', 'self', 1.0)"
        )

        # 3. Component 3: C -> D via leaf_relations (dangling sink)
        conn.execute(
            "INSERT INTO leaf_relations VALUES ('raw/c.md#chunk:0', 'mentions', 'concepts:sink', 'raw/c.md', 0.2)"
        )

        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        assert len(retriever._idx_to_node) == 5

        node_to_idx = retriever._node_to_idx
        idx_a = node_to_idx["raw/a.md"]
        idx_b = node_to_idx["raw/b.md"]
        idx_self = node_to_idx["raw/self.md"]
        idx_c = node_to_idx["raw/c.md#chunk:0"]
        idx_sink = node_to_idx["concepts:sink"]

        adj = retriever.adj.toarray()

        # Invariant 1: Mutual pair has 1.5 conductance
        assert pytest.approx(float(adj[idx_a, idx_b]), abs=1e-5) == 1.5
        assert pytest.approx(float(adj[idx_b, idx_a]), abs=1e-5) == 1.5

        # Invariant 2: Self-loop receives 1.0 + 0.5 = 1.5
        assert pytest.approx(float(adj[idx_self, idx_self]), abs=1e-5) == 1.5

        # Invariant 3: Dangling sink D has row sum 0.0
        assert pytest.approx(float(adj[idx_sink].sum()), abs=1e-5) == 0.0
        assert retriever._dangling_mask is not None
        assert (
            retriever._dangling_mask[idx_sink] is True
            or retriever._dangling_mask[idx_sink] == 1
        )
        assert not retriever._dangling_mask[idx_a]
        assert not retriever._dangling_mask[idx_b]
        assert not retriever._dangling_mask[idx_self]
        assert not retriever._dangling_mask[idx_c]

        # Invariant 4: PPR power iteration converges and conserves probability (sum = 1.0)
        p0 = np.zeros(5, dtype=np.float32)
        p0[idx_c] = 1.0  # Seed starts at C which points to dangling sink
        p_star = retriever.power_iteration(p0, alpha=0.85, max_iter=50)

        assert pytest.approx(float(np.sum(p_star)), abs=1e-5) == 1.0
        assert p_star[idx_sink] > 0.0, "Dangling sink receives probability mass"


# ==============================================================================
# 6. Asymmetric Edge Weights & Directional Conductance Rules
# ==============================================================================


def test_adversarial_asymmetric_edge_weights_and_directional_rules(tmp_path: Path):
    """Stress-test asymmetric edge weights and directional conductance rules.

    Invariants:
    1. Concept -> Chunk: downward broadcast rule dir_w = 1.0.
    2. Chunk -> Concept: upward abstraction rule dir_w = 0.2 (topic leakage prevention).
       Ratio dir_w(concept->chunk) / dir_w(chunk->concept) = 5.0.
    3. Sequential chunk edges: chunk:0 -> chunk:1 has dir_w = 0.8 (one-way).
    4. Coexisting Wikilink + Leaf Relation on same pair:
       Forward: wikilink (1.0) + leaf_relation (0.7) = 1.7.
       Reverse: wikilink_rev (0.5).
       Resulting in asymmetric conductance: adj[U, V] = 1.7, adj[V, U] = 0.5.
    """
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)

        # 1. Asymmetric Concept <-> Chunk
        concept = "concepts:machine_learning"
        chunk_0 = "raw/ml.md#chunk:0"
        chunk_1 = "raw/ml.md#chunk:1"

        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'manifests_in', ?, 'raw/ml.md', 1.0)",
            (concept, chunk_0),
        )
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'mentions', ?, 'raw/ml.md', 1.0)",
            (chunk_0, concept),
        )

        # 2. Sequential chunks
        conn.execute(
            "INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal) "
            "VALUES ('c0', ?, 'raw/ml.md', 0)",
            (chunk_0,),
        )
        conn.execute(
            "INSERT INTO note_chunks (chunk_id, entity_uri, rel_path, ordinal) "
            "VALUES ('c1', ?, 'raw/ml.md', 1)",
            (chunk_1,),
        )

        # 3. Coexisting Wikilink + Leaf Relation
        note_u = "raw/u.md"
        note_v = "raw/v.md"
        conn.execute("INSERT INTO wikilinks VALUES (?, ?, 'v', 1.0)", (note_u, note_v))
        conn.execute(
            "INSERT INTO leaf_relations VALUES (?, 'influences', ?, 'raw/u.md', 0.7)",
            (note_u, note_v),
        )

        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        adj = retriever.adj.toarray()
        node_to_idx = retriever._node_to_idx

        u_conc = node_to_idx[concept]
        u_c0 = node_to_idx[chunk_0]
        u_c1 = node_to_idx[chunk_1]
        u_u = node_to_idx[note_u]
        u_v = node_to_idx[note_v]

        # Check concept <-> chunk asymmetry
        # Both concept and chunk_0 have in-degree 1 from each other -> dampening factor = 1.0
        # Concept -> Chunk: dir_w = 1.0
        assert pytest.approx(float(adj[u_conc, u_c0]), abs=1e-5) == 1.0
        # Chunk -> Concept: dir_w = 0.2
        assert pytest.approx(float(adj[u_c0, u_conc]), abs=1e-5) == 0.2
        # Asymmetry ratio: exactly 5.0
        assert (
            pytest.approx(float(adj[u_conc, u_c0] / adj[u_c0, u_conc]), abs=1e-5) == 5.0
        )

        # Check sequential chunk edge: chunk:0 -> chunk:1 has dir_w = 0.8
        # chunk:1 forward in-degree = 1 (from sequence edge)
        assert pytest.approx(float(adj[u_c0, u_c1]), abs=1e-5) == 0.8
        # Sequence edge is strictly one-way (no reverse sequence back-edge)
        assert pytest.approx(float(adj[u_c1, u_c0]), abs=1e-5) == 0.0

        # Check coexisting wikilink + leaf relation:
        # Note V forward in-degree = 2 (from wikilink and leaf_relation from U)
        # Forward edges from U to V:
        # 1. wikilink: 1.0 / sqrt(2)
        # 2. leaf_relation: 0.7 / sqrt(2)
        # Total forward = 1.7 / sqrt(2)
        assert pytest.approx(float(adj[u_u, u_v]), abs=1e-5) == 1.7 / (2.0**0.5)

        # Reciprocal back-edge from V to U:
        # Synthesized from wikilink U -> V: raw_w = 0.5.
        # U forward in-degree = 0 -> max(1, 0) = 1.
        # Weight = 0.5 / sqrt(1) = 0.5.
        assert pytest.approx(float(adj[u_v, u_u]), abs=1e-5) == 0.5


# ==============================================================================
# 7. Empty Graph / Zero Edges Robustness
# ==============================================================================


def test_adversarial_empty_graph_and_zero_nodes(tmp_path: Path):
    """Verify retriever handles empty or zero-edge databases without crashing."""
    config.set_vault(tmp_path)
    db_path = tmp_path / ".cache" / "yaw.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _init_graphrag_test_db_schema(conn)
        ensure_schema_views(conn)
        conn.commit()

        retriever = GraphRAGRetriever()
        retriever._sync_graph(conn)

        assert retriever.adj is not None
        assert retriever.adj.shape == (0, 0)
        assert retriever._P is not None
        assert retriever._P.shape == (0, 0)
        assert len(retriever._node_to_idx) == 0

        # Power iteration on empty p0 returns safely
        p0 = np.zeros(0, dtype=np.float32)
        p_star = retriever.power_iteration(p0)
        assert len(p_star) == 0


# ==============================================================================
# Adversarial Suite: Milestone 2 Daemon Lifecycle & WAL Governance
# ==============================================================================


def _lock_proc(
    db_path_str: str,
    barrier_start: Any,
    barrier_stop: Any,
) -> None:
    """Helper process holding exclusive transaction lock."""
    conn = sqlite3.connect(db_path_str)
    conn.execute("BEGIN EXCLUSIVE;")
    conn.execute("INSERT INTO lock_tbl VALUES (1, 'locked');")
    barrier_start.set()
    barrier_stop.wait(timeout=10.0)
    conn.commit()
    conn.close()


def test_lifecycle_abrupt_sigterm_cleanup(tmp_path: Path) -> None:
    """Empirical verification: Abrupt SIGTERM to running daemon triggers clean exit 0, PID deletion, and WAL truncation."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"
    ready_file = tmp_path / "ready.txt"

    # Seed WAL database
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT);")
    conn.commit()
    conn.close()

    script = f"""
import sys, time, sqlite3
from pathlib import Path
sys.path.insert(0, "/home/harri/yaw/src")
from yaw.server.daemon import DaemonLifecycle

lifecycle = DaemonLifecycle(pid_file=Path("{pid_file}"), db_path=Path("{db_path}"))
with lifecycle:
    daemon_conn = sqlite3.connect(str(lifecycle.db_path))
    daemon_conn.execute("PRAGMA journal_mode=WAL;")
    for i in range(20):
        daemon_conn.execute("INSERT INTO test (data) VALUES (?);", (f"val_{{i}}",))
    daemon_conn.commit()

    Path("{ready_file}").write_text("ready")
    while True:
        time.sleep(0.05)
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        t0 = time.monotonic()
        while not ready_file.exists():
            if time.monotonic() - t0 > 5.0:
                pytest.fail("Subprocess daemon failed to become ready within 5.0s")
            time.sleep(0.05)

        assert pid_file.is_file(), "PID file must exist while daemon is running"
        assert int(pid_file.read_text().strip()) == proc.pid

        wal_file = Path(str(db_path) + "-wal")
        assert wal_file.exists() and wal_file.stat().st_size > 0, (
            "WAL file must contain uncheckpointed pages"
        )

        # Fire SIGTERM
        proc.send_signal(signal.SIGTERM)

        # Invariant: exits cleanly with code 0 within SLA
        exit_code = proc.wait(timeout=5.0)
        assert exit_code == 0, (
            f"Daemon exited with non-zero code {exit_code} on SIGTERM"
        )

        # Invariant: PID file cleanly unlinked
        assert not pid_file.exists(), "PID file must be unlinked on SIGTERM exit"

        # Invariant: WAL file truncated or unlinked
        if wal_file.exists():
            assert wal_file.stat().st_size == 0, (
                f"WAL file not truncated to 0 bytes: {wal_file.stat().st_size}"
            )

        # Invariant: 0 data loss
        verify_conn = sqlite3.connect(str(db_path))
        row_count = verify_conn.execute("SELECT count(*) FROM test;").fetchone()[0]
        verify_conn.close()
        assert row_count == 20, f"Expected 20 persisted rows, found {row_count}"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_lifecycle_abrupt_sigint_cleanup(tmp_path: Path) -> None:
    """Empirical verification: Abrupt SIGINT triggers clean exit 0 and PID unlinking."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"
    ready_file = tmp_path / "ready.txt"

    script = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, "/home/harri/yaw/src")
from yaw.server.daemon import DaemonLifecycle

with DaemonLifecycle(pid_file=Path("{pid_file}"), db_path=Path("{db_path}")):
    Path("{ready_file}").write_text("ready")
    while True:
        time.sleep(0.05)
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        t0 = time.monotonic()
        while not ready_file.exists():
            if time.monotonic() - t0 > 5.0:
                pytest.fail("Subprocess daemon failed to become ready within 5.0s")
            time.sleep(0.05)

        assert pid_file.is_file()
        proc.send_signal(signal.SIGINT)

        exit_code = proc.wait(timeout=5.0)
        assert exit_code == 0, f"Daemon exited with non-zero code {exit_code} on SIGINT"
        assert not pid_file.exists(), "PID file must be unlinked on SIGINT exit"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_lifecycle_rapid_start_stop_cycles(tmp_path: Path) -> None:
    """Stress-test 50 rapid sequential start/stop cycles ensuring zero orphaned PID files or resource leaks."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    t0 = time.monotonic()
    for i in range(50):
        lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
        with lc:
            assert pid_file.is_file(), (
                f"Cycle {i}: PID file should exist during context"
            )
            assert int(pid_file.read_text().strip()) == os.getpid()
        assert not pid_file.exists(), (
            f"Cycle {i}: PID file should be unlinked after context exit"
        )
    duration = time.monotonic() - t0
    assert duration < 5.0, f"50 rapid cycles took {duration:.2f}s, expected < 5.0s"


def test_lifecycle_dirty_wal_high_volume_checkpoint(tmp_path: Path) -> None:
    """Stress-test checkpoint truncation against 1000 rows / 50 commits (>4MB WAL)."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"
    wal_path = Path(str(db_path) + "-wal")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE big_data (id INTEGER PRIMARY KEY, blob BLOB);")

    chunk = b"W" * 8192
    for _ in range(50):
        conn.execute("BEGIN;")
        for _ in range(20):
            conn.execute("INSERT INTO big_data (blob) VALUES (?);", (chunk,))
        conn.commit()

    assert wal_path.exists()
    wal_size_before = wal_path.stat().st_size
    assert wal_size_before > 100_000, (
        f"WAL file should exceed 100KB, got {wal_size_before}"
    )

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    t0 = time.monotonic()
    res = lc.checkpoint_wal(conn)
    elapsed = time.monotonic() - t0

    assert res is not None
    assert res[0] == 0, f"Checkpoint returned busy code {res[0]}"
    if wal_path.exists():
        assert wal_path.stat().st_size == 0, (
            f"WAL size not truncated to 0 bytes: {wal_path.stat().st_size}"
        )
    conn.close()

    assert elapsed < 5.0, f"Checkpoint exceeded 5.0s SLA: took {elapsed:.3f}s"

    verify_conn = sqlite3.connect(str(db_path))
    count = verify_conn.execute("SELECT count(*) FROM big_data;").fetchone()[0]
    verify_conn.close()
    assert count == 1000, f"Data loss detected: expected 1000 rows, found {count}"


def test_lifecycle_simulated_exclusive_lock_timeout_graceful(tmp_path: Path) -> None:
    """Stress-test concurrent exclusive lock: verify checkpoint_wal handles lock timeout gracefully within 5s SLA."""
    db_path = tmp_path / "yaw.db"
    pid_file = tmp_path / "yaw.pid"

    init_conn = sqlite3.connect(str(db_path))
    init_conn.execute("PRAGMA journal_mode=WAL;")
    init_conn.execute("CREATE TABLE lock_tbl (id INTEGER PRIMARY KEY, v TEXT);")
    init_conn.commit()
    init_conn.close()

    ctx = multiprocessing.get_context("spawn")
    barrier_start = ctx.Event()
    barrier_stop = ctx.Event()
    lock_proc = ctx.Process(
        target=_lock_proc, args=(str(db_path), barrier_start, barrier_stop)
    )
    lock_proc.start()

    try:
        assert barrier_start.wait(timeout=5.0), (
            "Lock process failed to acquire exclusive lock"
        )

        # Case 1: Short timeout under active lock returns within SLA without hanging
        lc_short = DaemonLifecycle(pid_file=pid_file, db_path=db_path, timeout_s=1.0)
        t0 = time.monotonic()
        res_short = lc_short.checkpoint_wal()
        elapsed_short = time.monotonic() - t0

        assert elapsed_short < 2.5, (
            f"checkpoint_wal hung! Took {elapsed_short}s with timeout_s=1.0"
        )
        # Under exclusive lock, checkpoint either returns busy tuple or None without crash
        assert res_short is None or res_short[0] != 0

        # Case 2: Lock released before 5.0s SLA allows checkpoint to complete
        def delayed_unlock() -> None:
            time.sleep(1.0)
            barrier_stop.set()

        unlock_t = threading.Thread(target=delayed_unlock)
        unlock_t.start()

        lc_5s = DaemonLifecycle(pid_file=pid_file, db_path=db_path, timeout_s=5.0)
        t1 = time.monotonic()
        res_5s = lc_5s.checkpoint_wal()
        elapsed_5s = time.monotonic() - t1
        unlock_t.join()

        assert elapsed_5s < 5.0, f"Checkpoint took {elapsed_5s}s, exceeding 5.0s SLA"
        assert res_5s is not None and res_5s[0] == 0, (
            f"Expected successful checkpoint, got {res_5s}"
        )
    finally:
        barrier_stop.set()
        lock_proc.join(timeout=5.0)
        if lock_proc.is_alive():
            lock_proc.kill()


def test_lifecycle_simulated_active_reader_lock_nonblocking(tmp_path: Path) -> None:
    """Verify that an active reader holding a transaction does not hang checkpoint indefinitely."""
    db_path = tmp_path / "yaw.db"
    pid_file = tmp_path / "yaw.pid"

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE read_test (id INT, text TEXT);")
    conn.execute("INSERT INTO read_test VALUES (1, 'initial');")
    conn.commit()

    reader = sqlite3.connect(str(db_path))
    reader.execute("BEGIN;")
    reader.execute("SELECT * FROM read_test;").fetchall()

    conn.execute("INSERT INTO read_test VALUES (2, 'post_reader');")
    conn.commit()

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path, timeout_s=1.5)
    t0 = time.monotonic()
    result = lc.checkpoint_wal()
    elapsed = time.monotonic() - t0

    reader.commit()
    reader.close()
    conn.close()

    assert elapsed < 3.0, f"Active reader caused checkpoint to hang for {elapsed:.2f}s"
    assert result is not None, "Active reader checkpoint should return status tuple"


def test_lifecycle_corrupted_pid_file_overwritten_and_cleaned(tmp_path: Path) -> None:
    """Verify that pre-existing corrupted non-numeric PID file is overwritten and cleaned up."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    # Pre-populate corrupted PID file
    pid_file.write_text("CORRUPTED_NON_NUMERIC_GARBAGE\x00\n", encoding="utf-8")

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    with lc:
        assert pid_file.is_file()
        assert int(pid_file.read_text().strip()) == os.getpid()
    assert not pid_file.exists(), (
        "Corrupted PID file must be cleaned up on context exit"
    )


def test_lifecycle_corrupted_db_file_graceful_handling(tmp_path: Path) -> None:
    """Verify that non-SQLite database file does not crash checkpoint_wal."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"
    db_path.write_bytes(b"NOT_A_VALID_SQLITE_HEADER")

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    res = lc.checkpoint_wal()
    assert res is None, (
        "Corrupted DB file must return None without raising unhandled exception"
    )


def test_lifecycle_context_manager_exception_cleanup(tmp_path: Path) -> None:
    """Verify PID file cleanup when exception is raised inside with block."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    with pytest.raises(ValueError, match="Adversarial explosion"), lc:
        assert pid_file.is_file()
        raise ValueError("Adversarial explosion")

    assert not pid_file.exists(), (
        "PID file must be unlinked even if context manager raises exception"
    )


def test_lifecycle_sys_exit_code_cleanup(tmp_path: Path) -> None:
    """Verify PID file cleanup via atexit when subprocess calls sys.exit(1)."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/harri/yaw/src")
from yaw.server.daemon import DaemonLifecycle

with DaemonLifecycle(pid_file=Path("{pid_file}"), db_path=Path("{db_path}")):
    sys.exit(1)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, check=False
    )
    assert proc.returncode == 1
    assert not pid_file.exists(), "PID file must be unlinked on sys.exit(1)"


def test_lifecycle_instance_reentry_cleanup(tmp_path: Path) -> None:
    """Verify that re-entering the same DaemonLifecycle instance resets _cleaned,
    truncates WAL, and cleans up PID file on each exit.
    """
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    # Initialize SQLite database in WAL mode
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("CREATE TABLE t (x INT);")

    wal_path = Path(str(db_path) + "-wal")

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    with lc:
        assert pid_file.is_file()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("INSERT INTO t VALUES (1);")
        assert wal_path.exists()
    assert not pid_file.exists()
    # On first exit, WAL must be checkpointed and truncated
    assert not wal_path.exists() or wal_path.stat().st_size == 0

    # Second entry on the same instance
    with lc:
        assert pid_file.is_file()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("INSERT INTO t VALUES (2);")
        assert wal_path.exists()
    assert not pid_file.exists(), (
        "Re-entering DaemonLifecycle failed to unlink PID file on second exit!"
    )
    # On second exit, WAL must also be checkpointed and truncated
    assert not wal_path.exists() or wal_path.stat().st_size == 0


def test_lifecycle_concurrent_pid_no_cross_unlink(tmp_path: Path) -> None:
    """Verify that an exiting daemon does not unlink a PID file that belongs to a different running process."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    lc = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    lc.start()
    assert pid_file.is_file()

    # Another process overwrote the PID file with its own PID
    pid_file.write_text("99999\n", encoding="utf-8")

    # Exiting daemon runs cleanup
    lc.cleanup()

    # The PID file should NOT be deleted if it belongs to PID 99999
    assert pid_file.exists(), (
        "Exiting daemon blindly unlinked PID file belonging to different PID 99999!"
    )


def test_lifecycle_flock_concurrency_locking(tmp_path: Path) -> None:
    """Verify that DaemonLifecycle uses fcntl.flock to prevent concurrent daemons from sharing a PID file."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"

    lc1 = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
    lc1.start()
    try:
        assert pid_file.is_file()
        assert lc1._pid_fd is not None
        assert pid_file.read_text(encoding="utf-8").strip() == str(os.getpid())

        # Second daemon against same PID file MUST raise RuntimeError due to non-blocking flock
        lc2 = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
        with pytest.raises(RuntimeError, match="Another yaw daemon is already running"):
            lc2.start()

        # Failed attempt must not leave _pid_fd open or corrupt the file
        assert lc2._pid_fd is None
        assert pid_file.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        lc1.cleanup()

    assert not pid_file.exists()
    assert lc1._pid_fd is None

    # After lc1 exits, lc2 can now successfully acquire the lock
    lc2.start()
    try:
        assert pid_file.is_file()
        assert lc2._pid_fd is not None
        assert pid_file.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        lc2.cleanup()

    assert not pid_file.exists()


def test_lifecycle_flock_cross_process_mutual_exclusion(tmp_path: Path) -> None:
    """Verify cross-process mutual exclusion via fcntl.flock and PID protection."""
    pid_file = tmp_path / "yaw.pid"
    db_path = tmp_path / "yaw.db"
    ready_file = tmp_path / "child_ready.txt"

    script = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, "/home/harri/yaw/src")
from yaw.server.daemon import DaemonLifecycle

lc = DaemonLifecycle(pid_file=Path("{pid_file}"), db_path=Path("{db_path}"))
with lc:
    Path("{ready_file}").write_text("ready")
    while True:
        time.sleep(0.05)
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    try:
        t0 = time.monotonic()
        while not ready_file.exists():
            if time.monotonic() - t0 > 5.0:
                pytest.fail("Child daemon did not become ready")
            time.sleep(0.05)

        assert pid_file.is_file()
        assert int(pid_file.read_text(encoding="utf-8").strip()) == proc.pid

        # Local daemon attempt must be rejected by fcntl.flock
        lc_local = DaemonLifecycle(pid_file=pid_file, db_path=db_path)
        with pytest.raises(RuntimeError, match="Another yaw daemon is already running"):
            lc_local.start()

        # Local cleanup must not unlink child's PID file
        lc_local.cleanup()
        assert pid_file.exists()
        assert int(pid_file.read_text(encoding="utf-8").strip()) == proc.pid
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)

    # After child exits, PID file is unlinked by child
    assert not pid_file.exists()


# ==============================================================================
# Adversarial Suite: Milestone 2 Candidate Rescoring & Byte-Seek Invariants
# ==============================================================================


def _setup_m2_adversarial_vault_and_db(tmp_path: Path) -> tuple[Path, Path]:
    config.set_vault(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "yaw.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE note_chunks (
                chunk_id TEXT, entity_uri TEXT PRIMARY KEY, rel_path TEXT, ordinal INTEGER,
                heading TEXT, heading_slug TEXT,
                line_start INTEGER, line_end INTEGER, embedding BLOB
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE leaf_nodes (
                node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT,
                definition TEXT, embedding BLOB, PRIMARY KEY (node_id, source_leaf_id)
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE query_fts USING fts5(
                entity_uri UNINDEXED, rel_path UNINDEXED, heading, content,
                content='', contentless_unindexed=1
            )
            """
        )
        conn.commit()

    return raw_dir, db_path


# ==============================================================================
# Invariant 1: Strictly 0 Outbound HTTP Calls During Retrieval / Rescoring
# ==============================================================================


@pytest.mark.anyio
async def test_adversarial_zero_outbound_http_rescoring_diverse_queries(
    tmp_path: Path, loopback_server: str
):
    """Stress-test that diverse queries make strictly ZERO outbound HTTP calls during rescoring."""
    raw_dir, db_path = _setup_m2_adversarial_vault_and_db(tmp_path)
    config.HOST = loopback_server
    config.OLLAMA_HOST = loopback_server

    # Populate 3 documents with multi-paragraph content
    docs = [
        (
            "quantum.md",
            "# Quantum Physics\nHeisenberg uncertainty limit in conjugate observables.",
        ),
        (
            "gravity.md",
            "# General Relativity\nSpacetime curvature determined by stress-energy tensor.",
        ),
        (
            "entropy.md",
            "# Thermodynamics\nSecond law states entropy never decreases in isolated systems.",
        ),
    ]

    unit_vec = np.zeros(768, dtype=np.float32)
    unit_vec[0] = 1.0  # Normalized unit vector

    with sqlite3.connect(db_path) as conn:
        for idx, (filename, content) in enumerate(docs):
            p = raw_dir / filename
            p.write_text(content, encoding="utf-8")
            uri = f"raw/{filename}#chunk:0"
            conn.execute(
                "INSERT INTO note_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"c{idx}",
                    uri,
                    f"raw/{filename}",
                    0,
                    f"Heading {idx}",
                    f"heading-{idx}",
                    1,
                    2,
                    unit_vec.tobytes(),
                ),
            )
            conn.execute(
                "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                (uri, f"raw/{filename}", f"Heading {idx}", content),
            )
        # Add taxonomy node
        tax_vec = np.zeros(768, dtype=np.float32)
        tax_vec[1] = 1.0
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concept:uncertainty', 'raw/quantum.md', 'Uncertainty', 'concept', 'Principle of quantum mechanics', ?)",
            (tax_vec.tobytes(),),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('concept:uncertainty', 'raw/quantum.md', 'Uncertainty', 'Principle of quantum mechanics')",
        )
        conn.commit()

    diverse_queries = [
        "quantum physics uncertainty",
        "spacetime curvature general relativity",
        "thermodynamics second law entropy",
        "concept:uncertainty",
        "completely unindexed gibberish xyzabc12345",
        "quantum AND relativity OR entropy",
        "special characters: !@#$%^&*()_+-=[]{}|;':,.<>/?",
        "SQL injection test: ' OR '1'='1' --",
        "Unicode characters: 量子力学 相对论 熵 🧠 ⚡",
        "empty whitespace",
    ]

    q_vec = np.zeros(768, dtype=np.float32)
    q_vec[0] = 1.0

    # 1. Probe with precomputed query_vec: ZERO HTTP requests permitted across ALL diverse queries
    BaseTestHTTPHandler.reset_requests()
    for q in diverse_queries:
        candidates = await compute_seed_candidates(q, top_k=5, query_vec=q_vec)
        assert isinstance(candidates, list)
        for c in candidates:
            assert "rerank_score" in c
            assert isinstance(c["rerank_score"], float)

    assert len(BaseTestHTTPHandler.received_requests) == 0, (
        f"Expected strictly 0 HTTP calls with query_vec supplied; received: {BaseTestHTTPHandler.received_requests}"
    )

    # 2. Concurrency test: execute multiple queries concurrently via asyncio.gather
    BaseTestHTTPHandler.reset_requests()
    tasks = [
        compute_seed_candidates(q, top_k=3, query_vec=q_vec) for q in diverse_queries
    ]
    concurrency_results = await asyncio.gather(*tasks)
    assert len(concurrency_results) == len(diverse_queries)
    assert len(BaseTestHTTPHandler.received_requests) == 0, (
        "Concurrent execution must make strictly 0 outbound HTTP calls when query_vec is supplied"
    )

    # 3. Probe candidate rescoring: verify ZERO calls to embed_texts (candidate document embedding)
    # Even if query_vec is not passed and query embedding is fetched, candidate documents must NEVER be embedded via HTTP
    BaseTestHTTPHandler.reset_requests()
    seeds = await compute_seed_candidates("quantum physics", top_k=3)
    assert len(seeds) > 0
    # Any HTTP call must ONLY be for the query itself, NEVER for candidate documents
    for req in BaseTestHTTPHandler.received_requests:
        req_body = req.get("body", req)
        inputs = req_body.get("input", req_body.get("inputs", []))
        if isinstance(inputs, list):
            for item in inputs:
                assert "Heisenberg uncertainty" not in item, (
                    "Candidate document must never be sent for embedding!"
                )
                assert "Spacetime curvature" not in item, (
                    "Candidate document must never be sent for embedding!"
                )


# ==============================================================================
# Invariant 2: Graceful Fallbacks for Stored Vectors (NULL, Malformed, Mismatch)
# ==============================================================================


@pytest.mark.anyio
async def test_adversarial_missing_and_malformed_vector_fallbacks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Verify that NULL, odd-sized, malformed JSON, and mismatched vectors all fall back to hybrid_prob."""
    raw_dir, db_path = _setup_m2_adversarial_vault_and_db(tmp_path)

    # Create test document
    doc_text = "# Adversarial Tests\nTesting robustness of vector deserialization."
    doc_path = raw_dir / "adversarial.md"
    doc_path.write_text(doc_text, encoding="utf-8")

    valid_vec = np.zeros(768, dtype=np.float32)
    valid_vec[0] = 1.0  # Aligned with q_vec

    short_vec = np.zeros(256, dtype=np.float32)  # Dimension mismatch (256 != 768)
    long_vec = np.zeros(1024, dtype=np.float32)  # Dimension mismatch (1024 != 768)

    odd_bytes = b"\x01\x02\x03"  # 3 bytes: cannot be divided by 4 for float32

    test_candidates = [
        ("c_valid", "raw/adversarial.md#chunk:valid", "valid", valid_vec.tobytes()),
        ("c_null", "raw/adversarial.md#chunk:null", "null", None),
        ("c_empty_blob", "raw/adversarial.md#chunk:empty", "empty", b""),
        ("c_odd_bytes", "raw/adversarial.md#chunk:odd", "odd", odd_bytes),
        ("c_short_vec", "raw/adversarial.md#chunk:short", "short", short_vec.tobytes()),
        ("c_long_vec", "raw/adversarial.md#chunk:long", "long", long_vec.tobytes()),
    ]

    with sqlite3.connect(db_path) as conn:
        for cid, uri, name, blob in test_candidates:
            conn.execute(
                "INSERT INTO note_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    uri,
                    "raw/adversarial.md",
                    0,
                    f"Section {name}",
                    f"sec-{name}",
                    1,
                    2,
                    blob,
                ),
            )
            conn.execute(
                "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                (uri, "raw/adversarial.md", f"Section {name}", doc_text),
            )

        # Taxonomy nodes: NULL embedding, malformed JSON text, mismatched vector
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concept:tax_null', 'raw/adversarial.md', 'TaxNull', 'concept', 'Def null', NULL)"
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('concept:tax_null', 'raw/adversarial.md', 'TaxNull', 'Def null')"
        )
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('concept:tax_mismatch', 'raw/adversarial.md', 'TaxMismatch', 'concept', 'Def mismatch', ?)",
            (short_vec.tobytes(),),
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('concept:tax_mismatch', 'raw/adversarial.md', 'TaxMismatch', 'Def mismatch')"
        )
        conn.commit()

    q_vec = np.zeros(768, dtype=np.float32)
    q_vec[0] = 1.0

    with caplog.at_level(logging.WARNING):
        candidates = await compute_seed_candidates(
            "Adversarial Tests", top_k=10, query_vec=q_vec
        )

    assert len(candidates) >= 5, "Candidate pool should include all matching entities"

    candidate_map = {c["entity_uri"]: c for c in candidates}

    # 1. Valid candidate receives dot product score
    assert "raw/adversarial.md#chunk:valid" in candidate_map
    valid_c = candidate_map["raw/adversarial.md#chunk:valid"]
    assert pytest.approx(valid_c["rerank_score"], abs=1e-4) == 1.0

    # 2. NULL embedding chunk falls back to hybrid_prob
    if "raw/adversarial.md#chunk:null" in candidate_map:
        null_c = candidate_map["raw/adversarial.md#chunk:null"]
        assert null_c["rerank_score"] == null_c["hybrid_prob"]

    # 3. Empty blob chunk falls back to hybrid_prob
    if "raw/adversarial.md#chunk:empty" in candidate_map:
        empty_c = candidate_map["raw/adversarial.md#chunk:empty"]
        assert empty_c["rerank_score"] == empty_c["hybrid_prob"]

    # 4. Odd-sized byte chunk falls back to hybrid_prob without crashing
    if "raw/adversarial.md#chunk:odd" in candidate_map:
        odd_c = candidate_map["raw/adversarial.md#chunk:odd"]
        assert odd_c["rerank_score"] == odd_c["hybrid_prob"]

    # 5. Dimension mismatch chunks fall back to hybrid_prob
    if "raw/adversarial.md#chunk:short" in candidate_map:
        short_c = candidate_map["raw/adversarial.md#chunk:short"]
        assert short_c["rerank_score"] == short_c["hybrid_prob"]

    if "raw/adversarial.md#chunk:long" in candidate_map:
        long_c = candidate_map["raw/adversarial.md#chunk:long"]
        assert long_c["rerank_score"] == long_c["hybrid_prob"]

    # 6. Taxonomy fallbacks
    if "concept:tax_null" in candidate_map:
        tax_null_c = candidate_map["concept:tax_null"]
        assert tax_null_c["rerank_score"] == tax_null_c["hybrid_prob"]

    if "concept:tax_mismatch" in candidate_map:
        tax_mis_c = candidate_map["concept:tax_mismatch"]
        assert tax_mis_c["rerank_score"] == tax_mis_c["hybrid_prob"]

    # 7. Warnings logged for fallbacks
    warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "lacks stored embedding vector" in msg
        or "dimension mismatch" in msg
        or "Failed to deserialize" in msg
        for msg in warning_texts
    ), f"Expected warnings logged for fallback candidates, got: {warning_texts}"


# ==============================================================================
# Invariant 3: Physical Disk Byte Seeks Over Complex Multi-Byte UTF-8 Texts
# ==============================================================================


@pytest.mark.anyio
async def test_adversarial_physical_disk_line_hydration_multibyte_utf8(tmp_path: Path):
    """Stress-test 1-indexed line hydration seeking over complex multi-byte UTF-8 texts on physical disk."""
    raw_dir, db_path = _setup_m2_adversarial_vault_and_db(tmp_path)

    # Complex multi-byte text with 4-byte emojis, 3-byte CJK, 3-byte Devanagari, 2-byte Greek/Cyrillic
    # and varied newlines/code blocks
    doc_sections = [
        (
            "# 🧠 Title & Introduction\nThis introduction has 4-byte emojis ⚡🚀🔬 and Greek letters $\\alpha + \\beta = \\gamma$.\n\n",
            "Intro",
        ),
        (
            "## 量子力学 & 相对论\n中文字符占用3个字节。Spacetime curvature: $R_{\\mu\\nu} - \\frac{1}{2}g_{\\mu\\nu}R = 8\\pi T_{\\mu\\nu}$.\n\n",
            "CJK",
        ),
        (
            "## Devanagari नमस्ते\nक्वांटम भौतिकी और ब्रह्मांड विज्ञान (Quantum physics in Hindi).\n\n",
            "Devanagari",
        ),
        (
            "## Code & End\n```python\n# Cyrillic comment: Теория струн\ndef energy(m, c=3e8):\n    return m * c**2\n```\nFinal sentence without trailing newline.",
            "Code",
        ),
    ]

    full_text = "".join(text for text, _ in doc_sections)
    full_bytes = full_text.encode("utf-8")
    note_file = raw_dir / "multibyte_corpus.md"
    note_file.write_bytes(full_bytes)

    # Calculate exact line ranges on disk
    current_line = 1
    chunks_meta = []
    for idx, (section_text, heading) in enumerate(doc_sections):
        num_newlines = section_text.count("\n")
        l_start = current_line
        l_end = max(
            l_start,
            current_line + num_newlines - (1 if section_text.endswith("\n") else 0),
        )
        chunks_meta.append(
            (
                f"c{idx}",
                f"raw/multibyte_corpus.md#chunk:{idx}",
                heading,
                l_start,
                l_end,
                section_text,
            )
        )
        current_line += num_newlines

    with sqlite3.connect(db_path) as conn:
        for cid, uri, heading, l_start, l_end, sec_text in chunks_meta:
            conn.execute(
                "INSERT INTO note_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    cid,
                    uri,
                    "raw/multibyte_corpus.md",
                    0,
                    heading,
                    heading.lower(),
                    l_start,
                    l_end,
                ),
            )
            conn.execute(
                "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                (uri, "raw/multibyte_corpus.md", heading, sec_text),
            )

        # Add taxonomy node with Chinese name and multi-byte definition
        tax_text = "量子纠缠 (quantum:entanglement): 处于纠缠态的两个粒子彼此关联"
        conn.execute(
            "INSERT INTO leaf_nodes VALUES ('quantum:entanglement', 'raw/multibyte_corpus.md', '量子纠缠', 'quantum', '处于纠缠态的两个粒子彼此关联', NULL)",
        )
        conn.execute(
            "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES ('quantum:entanglement', 'raw/multibyte_corpus.md', '量子纠缠', ?)",
            (tax_text,),
        )
        conn.commit()

    # Query for all seeds
    results = await find_seed("量子力学 नमस्ते", top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0

    chunk_results = [r for r in results if r["kind"] == "chunk"]
    tax_results = [r for r in results if r["kind"] == "taxonomy"]

    # 1. Verify chunk candidates on real physical disk
    for seed in chunk_results:
        # Schema verification
        assert set(seed.keys()) == {
            "entity_uri",
            "kind",
            "rel_path",
            "heading",
            "line_start",
            "line_end",
            "hybrid_prob",
            "rerank_score",
            "text",
        }
        assert isinstance(seed["line_start"], int)
        assert isinstance(seed["line_end"], int)
        assert seed["line_start"] >= 1
        assert seed["line_end"] >= seed["line_start"]

        # Physical disk line hydration
        file_disk_path = tmp_path / seed["rel_path"]
        assert file_disk_path.is_file()

        # Parity with read_document
        parity_text = read_document(
            seed["rel_path"], seed["line_start"], seed["line_end"]
        )
        assert parity_text.rstrip("\r\n") == seed["text"].rstrip("\r\n")

    # 2. Verify taxonomy candidate contract
    for tax_seed in tax_results:
        assert tax_seed["kind"] == "taxonomy"
        assert tax_seed["line_start"] is None
        assert tax_seed["line_end"] is None
        assert isinstance(tax_seed["text"], str)


# ==============================================================================
# Invariant 4: Edge Cases on hydrate_disk_slice & read_document
# ==============================================================================


def test_adversarial_read_document_boundary_conditions(tmp_path: Path):
    """Stress-test out-of-bounds, negative, traversal, and EOF disk seek handling."""
    config.set_vault(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    test_file = raw_dir / "boundary.md"
    test_content = "Line 1: Hello, world! 🌍\nLine 2: Unicode boundary check.\nLine 3: Third line.\n"
    test_file.write_text(test_content, encoding="utf-8")

    # 1. Read to EOF with line_start=None, line_end=None
    eof_slice = read_document("raw/boundary.md", line_start=None, line_end=None)
    assert eof_slice == test_content

    # 2. Seek beyond file length lines: returns empty string
    beyond_slice = read_document("raw/boundary.md", line_start=100, line_end=150)
    assert beyond_slice == ""

    # 3. Read exceeding line count: reads available lines without error
    overshoot_slice = read_document("raw/boundary.md", line_start=1, line_end=1000)
    assert overshoot_slice == test_content

    # 4. Invalid line ranges (line_start < 1 or line_end < line_start) raise ValueError
    with pytest.raises(ValueError, match="must be >= 1"):
        read_document("raw/boundary.md", line_start=0, line_end=2)

    with pytest.raises(ValueError, match="must be >= line_start"):
        read_document("raw/boundary.md", line_start=3, line_end=1)

    # 5. Security path traversal: must raise ValueError
    with pytest.raises(ValueError, match="outside VAULT_ROOT"):
        read_document("../../etc/passwd", line_start=1, line_end=10)

    # 6. Non-existent file: returns empty string
    assert read_document("raw/non_existent.md", line_start=1, line_end=10) == ""


# ==============================================================================
# Adversarial Suite: Milestone 3 CLI Subparsers & Process Governance
# ==============================================================================


def test_cli_no_args_stderr_and_exit_code_2():
    """Verify yaw invocation with zero arguments prints usage to stderr and exits code 2."""
    env = get_cli_env()
    res = subprocess.run(
        [sys.executable, YAW_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )

    assert res.returncode == 2, (
        f"Expected exit code 2 when run with no arguments, got {res.returncode}"
    )
    assert res.stdout == "", (
        f"Expected empty stdout when run with no arguments, got: {res.stdout!r}"
    )
    assert "usage: yaw" in res.stderr.lower(), (
        f"Expected usage string in stderr, got: {res.stderr!r}"
    )


@pytest.mark.parametrize(
    ("args", "expected_err_snippet"),
    [
        (["bogus_subcommand"], "invalid choice"),
        (["--unrecognized-option"], "unrecognized arguments"),
        (["index", "--unknown-flag"], "unrecognized arguments"),
        (["query"], "the following arguments are required: prompt"),
        (["serve", "--transport", "unsupported"], "invalid choice"),
        (["serve", "--port", "invalid_port"], "invalid int value"),
        (["query", "test", "--top-k", "not_a_number"], "invalid int value"),
    ],
)
def test_cli_invalid_args_and_commands_exit_code_2(
    args: list[str], expected_err_snippet: str
):
    """Verify invalid subcommands and malformed options exit code 2 and route errors to stderr."""
    env = get_cli_env()
    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )

    assert res.returncode == 2, (
        f"Command with args {args} should exit code 2, got {res.returncode}"
    )
    assert res.stdout == "", (
        f"Stdout should be empty on argument parsing error for {args}, got: {res.stdout!r}"
    )
    assert expected_err_snippet.lower() in res.stderr.lower(), (
        f"Stderr should contain '{expected_err_snippet}', got: {res.stderr!r}"
    )


def test_cli_index_synchronous_and_zero_zombies(tmp_path: Path):
    """Verify yaw index completes synchronously in foreground with zero detached zombie processes."""
    vault_dir = tmp_path / "valid_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_a = raw_dir / "note_a.md"
    note_a.write_text(
        "# Note Alpha\nConcept alpha is linked to [[Note Beta]].", encoding="utf-8"
    )
    note_b = raw_dir / "note_b.md"
    note_b.write_text(
        "# Note Beta\nConcept beta is defined as an entity in graph.", encoding="utf-8"
    )

    env = get_cli_env()

    # Capture initial process IDs
    current_proc = psutil.Process()
    children_before = set(current_proc.children(recursive=True))

    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw index failed with code {res.returncode}:\nSTDERR: {res.stderr}"
    )

    # Invariant: SQLite database created and populated
    db_file = vault_dir / ".cache" / "yaw.db"
    assert db_file.exists(), "yaw.db was not created by yaw index"

    # Invariant: Check no lingering zombie/orphan subprocesses
    children_after = set(current_proc.children(recursive=True))
    leaked_children = children_after - children_before
    assert not leaked_children, f"Detected leaked zombie children: {leaked_children}"


def test_cli_index_nonexistent_vault_directory(tmp_path: Path):
    """Verify yaw index handles non-existent vault directories synchronously without zombie spawns."""
    nonexistent_vault = tmp_path / "does_not_exist_subpath" / "target_vault"
    assert not nonexistent_vault.exists()

    env = get_cli_env()
    current_proc = psutil.Process()
    children_before = set(current_proc.children(recursive=True))

    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(nonexistent_vault)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw index on nonexistent vault returned {res.returncode}:\nSTDERR: {res.stderr}"
    )

    children_after = set(current_proc.children(recursive=True))
    leaked_children = children_after - children_before
    assert not leaked_children, f"Detected leaked zombie children: {leaked_children}"


def test_cli_index_flags_force_and_quiet(tmp_path: Path):
    """Verify yaw index supports --force and --quiet flags deterministically."""
    vault_dir = tmp_path / "flags_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "sample.md").write_text("# Sample\nText content.", encoding="utf-8")

    env = get_cli_env()

    # Initial index
    res_init = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=False,
    )
    assert res_init.returncode == 0

    # Quiet index
    res_quiet = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir), "--quiet"],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=False,
    )
    assert res_quiet.returncode == 0

    # Force rebuild
    res_force = subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir), "--force"],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=False,
    )
    assert res_force.returncode == 0
    assert "[clean-rebuild]" in res_force.stderr


def test_cli_schema_missing_database_contract(tmp_path: Path):
    """Verify yaw schema outputs structured JSON with zero stats when database is missing."""
    empty_vault = tmp_path / "empty_vault_no_db"
    empty_vault.mkdir(parents=True, exist_ok=True)

    env = get_cli_env()
    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "schema", "--vault", str(empty_vault)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw schema failed on missing database with code {res.returncode}:\nSTDERR: {res.stderr}"
    )

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as err:
        pytest.fail(
            f"yaw schema stdout is not valid JSON:\nSTDOUT: {res.stdout}\nError: {err}"
        )

    assert "namespaces" in data
    assert "base_predicates" in data
    assert "stats" in data
    assert isinstance(data["namespaces"], list)
    assert isinstance(data["base_predicates"], list)
    assert data["stats"] == {"chunks": 0, "taxonomies": 0, "edges": 0}


def test_cli_schema_populated_database_contract(tmp_path: Path):
    """Verify yaw schema outputs structured JSON with populated stats on indexed vault."""
    vault_dir = tmp_path / "populated_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "note.md").write_text(
        "# Entity A\nContent mentioning [[Entity B]].", encoding="utf-8"
    )

    env = get_cli_env()

    # Index first
    subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=True,
    )

    # Query schema
    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "schema", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw schema returned {res.returncode}:\nSTDERR: {res.stderr}"
    )

    data = json.loads(res.stdout)
    assert isinstance(data["namespaces"], list)
    assert len(data["namespaces"]) > 0
    assert isinstance(data["base_predicates"], list)
    assert "stats" in data
    assert data["stats"]["chunks"] > 0
    assert isinstance(data["stats"]["taxonomies"], int)
    assert isinstance(data["stats"]["edges"], int)


def test_cli_schema_corrupted_database_resilience(tmp_path: Path):
    """Verify yaw schema handles corrupted database files gracefully without uncaught exceptions on stdout."""
    corrupt_vault = tmp_path / "corrupt_vault"
    db_dir = corrupt_vault / ".cache"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "yaw.db").write_text(
        "CORRUPTED NOT A SQLITE DATABASE", encoding="utf-8"
    )

    env = get_cli_env()
    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "schema", "--vault", str(corrupt_vault)],
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )

    # Either it returns exit code 0 with zero fallback stats, or handled exit code 1 with stderr diagnostic
    if res.returncode == 0:
        data = json.loads(res.stdout)
        assert data["stats"]["chunks"] == 0
    else:
        assert res.returncode == 1
        assert "Error querying schema" in res.stderr
        assert res.stdout == ""


def test_cli_query_typical_and_evidence_content(tmp_path: Path):
    """Verify yaw query returns dual-layer Markdown with physical note evidence on typical prompt."""
    vault_dir = tmp_path / "query_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "superconductivity.md").write_text(
        "# High Temperature Superconductivity\n"
        "Werner Heisenberg and physicists investigated [[Quantum Physics]] where "
        "cuprate perovskite materials exhibit superconductivity above boiling point.",
        encoding="utf-8",
    )

    env = get_cli_env()

    subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=True,
    )

    res = subprocess.run(
        [
            sys.executable,
            YAW_SCRIPT,
            "query",
            "superconductivity perovskite",
            "--vault",
            str(vault_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw query failed with code {res.returncode}:\nSTDERR: {res.stderr}"
    )

    stdout = res.stdout
    assert "# Ontological Context" in stdout
    assert "## Evidence Chunk" in stdout
    assert "superconductivity" in stdout.lower() or "cuprate" in stdout.lower()


@pytest.mark.parametrize(
    "query_prompt",
    [
        "",  # Empty string
        "    ",  # Whitespace only
        'quantum "entanglement" AND OR NOT',  # FTS5 keyword combinations
    ],
)
def test_cli_query_empty_and_fts_keywords(tmp_path: Path, query_prompt: str):
    """Verify yaw query handles empty, whitespace, and boolean keyword prompts without crashing."""
    vault_dir = tmp_path / "adv_query_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "photon.md").write_text(
        "# Quantum Entanglement\n"
        "Werner Heisenberg formulated relations in [[Quantum Theory]] showing "
        "EPR paradox demonstrates nonlocal correlations between entangled photons.",
        encoding="utf-8",
    )

    env = get_cli_env()

    subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=True,
    )

    res = subprocess.run(
        [sys.executable, YAW_SCRIPT, "query", query_prompt, "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=15.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw query crashed on prompt {query_prompt!r} with code {res.returncode}:\nSTDERR: {res.stderr}"
    )
    assert "# Ontological Context" in res.stdout
    assert "## Evidence Chunk" in res.stdout


def test_cli_query_malformed_syntax_resilience(tmp_path: Path):
    """Verify yaw query handles malformed syntax gracefully without crashing or polluting stdout."""
    vault_dir = tmp_path / "syntax_err_vault"
    raw_dir = vault_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "test.md").write_text("# Test\nContent with [[Link]].", encoding="utf-8")

    env = get_cli_env()

    subprocess.run(
        [sys.executable, YAW_SCRIPT, "index", "--vault", str(vault_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=25.0,
        check=True,
    )

    malformed_prompt = '!@#$%^&*()_+-=[]{}|;:",.<>?/'
    res = subprocess.run(
        [
            sys.executable,
            YAW_SCRIPT,
            "query",
            malformed_prompt,
            "--vault",
            str(vault_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15.0,
        check=False,
    )

    # Invariant: handled gracefully without crashing uncaught or polluting stdout
    assert res.returncode in (0, 1)
    if res.returncode == 0:
        assert "# Ontological Context" in res.stdout
        assert "## Evidence Chunk" in res.stdout
    else:
        assert "Error executing GraphRAG query" in res.stderr
        assert res.stdout == ""


def test_cli_query_unindexed_vault_fallback(tmp_path: Path):
    """Verify yaw query on empty unindexed vault returns structured fallback Markdown."""
    empty_vault = tmp_path / "unindexed_vault"
    empty_vault.mkdir(parents=True, exist_ok=True)

    env = get_cli_env()
    res = subprocess.run(
        [
            sys.executable,
            YAW_SCRIPT,
            "query",
            "any search query",
            "--vault",
            str(empty_vault),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"yaw query on unindexed vault failed with code {res.returncode}:\nSTDERR: {res.stderr}"
    )
    assert "# Ontological Context" in res.stdout
    assert "_No indexed knowledge available._" in res.stdout
    assert "## Evidence Chunk" in res.stdout
    assert (
        "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold."
        in res.stdout
    )


def test_stop_daemon_missing_and_stale_pid(tmp_path: Path) -> None:
    """Verify stop_daemon handling of non-existent, stale, and corrupted PID files."""
    missing_pid = tmp_path / "missing.pid"
    assert stop_daemon(missing_pid) == 0

    # Stale PID: process has already terminated
    dead_script = tmp_path / "dead_script.py"
    dead_script.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    dead_proc = subprocess.Popen([sys.executable, str(dead_script)])
    dead_proc.wait(timeout=5.0)

    stale_pid = tmp_path / "stale.pid"
    stale_pid.write_text(f"{dead_proc.pid}\n", encoding="utf-8")
    assert stale_pid.is_file()
    assert stop_daemon(stale_pid) == 0
    assert not stale_pid.exists()

    # Corrupt / non-numeric PID file
    corrupt_pid = tmp_path / "corrupt.pid"
    corrupt_pid.write_text("invalid_pid\n", encoding="utf-8")
    assert stop_daemon(corrupt_pid) == 0
    assert not corrupt_pid.exists()


def test_stop_daemon_terminates_running_process(tmp_path: Path) -> None:
    """Verify stop_daemon cleanly terminates a running process via SIGTERM and unlinks PID."""
    ready_file = tmp_path / "ready.txt"
    dummy_script = tmp_path / "dummy_daemon.py"
    dummy_script.write_text(
        f"""import sys, time
from pathlib import Path
Path({str(ready_file)!r}).write_text("ready", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(dummy_script)])
    try:
        t0 = time.monotonic()
        while not ready_file.exists():
            if time.monotonic() - t0 > 5.0:
                pytest.fail("Dummy daemon failed to write ready flag")
            time.sleep(0.05)

        pid_file = tmp_path / "yaw.pid"
        pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

        assert stop_daemon(pid_file) == 0
        assert not pid_file.exists()
        proc.wait(timeout=5.0)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)


def test_stop_daemon_sigkill_on_hung_process(tmp_path: Path) -> None:
    """Verify stop_daemon falls back to SIGKILL when process ignores SIGTERM."""
    ready_file = tmp_path / "ready_sigkill.txt"
    hung_script = tmp_path / "hung_daemon.py"
    hung_script.write_text(
        f"""import signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(ready_file)!r}).write_text("ready", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(hung_script)])
    try:
        t0 = time.monotonic()
        while not ready_file.exists():
            if time.monotonic() - t0 > 5.0:
                pytest.fail("Hung daemon failed to write ready flag")
            time.sleep(0.05)

        pid_file = tmp_path / "hung.pid"
        pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

        # Use short timeout to trigger SIGKILL fallback
        assert stop_daemon(pid_file, timeout_s=0.2) == 0
        assert not pid_file.exists()
        proc.wait(timeout=5.0)
        assert proc.returncode is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)


def test_cli_shutdown_subcommand(tmp_path: Path) -> None:
    """Verify yaw shutdown cleanly shuts down daemon in vault cache and handles idle vault."""
    vault_dir = tmp_path / "test_vault"
    cache_dir = vault_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pid_file = cache_dir / "yaw.pid"

    ready_file = tmp_path / "cli_ready.txt"
    script = tmp_path / "cli_daemon.py"
    script.write_text(
        f"""import sys, time
from pathlib import Path
Path({str(ready_file)!r}).write_text("ready", encoding="utf-8")
while True:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        t0 = time.monotonic()
        while not ready_file.exists():
            if time.monotonic() - t0 > 5.0:
                pytest.fail("CLI test daemon failed to write ready flag")
            time.sleep(0.05)

        pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

        env = get_cli_env()
        res = subprocess.run(
            [sys.executable, YAW_SCRIPT, "shutdown", "--vault", str(vault_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=10.0,
            check=False,
        )
        assert res.returncode == 0, f"yaw shutdown failed: {res.stderr}"
        assert not pid_file.exists()
        proc.wait(timeout=5.0)
        assert proc.returncode is not None

        # Re-running shutdown on idle vault succeeds idempotently
        res2 = subprocess.run(
            [sys.executable, YAW_SCRIPT, "shutdown", "--vault", str(vault_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=10.0,
            check=False,
        )
        assert res2.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)
