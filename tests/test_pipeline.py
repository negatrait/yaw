"""Unmocked integration tests for declarative CocoIndex sqlite-vec + FTS5 pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlite_vec
from numpy.typing import NDArray

from yaw.core import YawConfig, format_query_for_embedding
from yaw.pipeline import (
    close_pipeline,
    list_user_tables,
    make_dspy_embed_fn,
    start_pipeline,
    update_pipeline_once,
    wait_pipeline_ready,
)


def _hash_embed(texts: list[str], *, dim: int = 16) -> NDArray[np.float32]:
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = np.frombuffer(digest * ((dim * 4 // len(digest)) + 1), dtype=np.uint8)[:dim].astype(np.float32) - 127.5
        norm = max(float(np.linalg.norm(vals)), 1e-12)
        rows[row_idx] = vals / norm
    return rows


def _make_cfg(root: Path, *, chunk_size: int = 300, chunk_overlap: int = 40, min_chunk_size: int = 20) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )


def _open_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def test_declarative_two_table_schema_edit_and_delete(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    corpus = cfg.corpus_root
    assert corpus is not None and cfg.db_path is not None
    corpus.mkdir(parents=True)
    doc_a = corpus / "alpha.md"
    doc_a.write_text("# Alpha System\nCobalt batteries power lunar rovers reliably.\n", encoding="utf-8")
    doc_b = corpus / "beta.json"
    doc_b.write_text(
        json.dumps({"title": "Beta Protocol", "text": "Zephyr turbine achieves 92% efficiency in arctic tests."}),
        encoding="utf-8",
    )

    session = start_pipeline(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=16), vector_dim=16, live=False)
    try:
        update_pipeline_once(session)
        with _open_ro(cfg.db_path) as conn:
            assert list_user_tables(conn) == ("chunk_fts", "chunks")
            ddl = str(conn.execute("SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0])
            assert "distance_metric=cosine" in ddl
            rows = conn.execute(
                "SELECT chunk_id, rel_path, source_hash, start_line, end_line, title, tokens_json, entities_json "
                "FROM chunks ORDER BY rel_path"
            ).fetchall()
            assert len(rows) == 2
            fts_count = conn.execute("SELECT count(*) FROM chunk_fts").fetchone()[0]
            assert fts_count == 2

            # Verify FTS5 BM25 query and vec0 cosine KNN query work on the same integer chunk_id
            fts_hits = conn.execute("SELECT rowid, title FROM chunk_fts WHERE chunk_fts MATCH 'lunar'").fetchall()
            assert len(fts_hits) == 1
            alpha_cid = rows[0][0]
            assert fts_hits[0][0] == alpha_cid
            q_vec = _hash_embed([format_query_for_embedding("lunar rovers", "distinct-titled")], dim=16)[0]
            knn_hits = conn.execute(
                "SELECT chunk_id, distance FROM chunks WHERE embedding MATCH ? AND k = 2",
                (sqlite_vec.serialize_float32(q_vec.tolist()),),
            ).fetchall()
            assert len(knn_hits) == 2
            assert {r[0] for r in rows} == {h[0] for h in knn_hits}

        # Mutate doc_a and unlink doc_b -> assert clean reconciliation
        doc_a.write_text("# Alpha System\nTitanium thrusters propel orbital tugs smoothly.\n", encoding="utf-8")
        doc_b.unlink()
        update_pipeline_once(session)

        with _open_ro(cfg.db_path) as conn:
            rows_after = conn.execute("SELECT chunk_id, rel_path FROM chunks").fetchall()
            assert len(rows_after) == 1
            assert rows_after[0][1] == "alpha.md"
            assert conn.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 1
            assert conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'lunar'").fetchall() == []
            thruster_hits = conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'thrusters'").fetchall()
            assert [r[0] for r in thruster_hits] == [rows_after[0][0]]
    finally:
        close_pipeline(session)


def test_microbatch_64_plus_64_equals_128_invariance(tmp_path: Path) -> None:
    root_a = tmp_path / "batch_64_64"
    root_b = tmp_path / "batch_128"
    cfg_a = _make_cfg(root_a, chunk_size=500, chunk_overlap=0, min_chunk_size=10)
    cfg_b = _make_cfg(root_b, chunk_size=500, chunk_overlap=0, min_chunk_size=10)
    assert cfg_a.corpus_root is not None and cfg_b.corpus_root is not None
    assert cfg_a.db_path is not None and cfg_b.db_path is not None
    cfg_a.corpus_root.mkdir(parents=True)
    cfg_b.corpus_root.mkdir(parents=True)

    batch_sizes_seen: list[int] = []

    def tracked_embed(texts: list[str]) -> NDArray[np.float32]:
        batch_sizes_seen.append(len(texts))
        return _hash_embed(texts, dim=12)

    docs = [
        (f"doc_{i:03d}.md", f"# Note {i}\nEntity{i} collaborates with Laboratory{i} in sector {i % 7}.\n")
        for i in range(128)
    ]
    for name, body in docs[:64]:
        (cfg_a.corpus_root / name).write_text(body, encoding="utf-8")
    for name, body in docs:
        (cfg_b.corpus_root / name).write_text(body, encoding="utf-8")

    sess_a = start_pipeline(cfg_a, embed_fn=tracked_embed, vector_dim=12, live=False)
    try:
        update_pipeline_once(sess_a)
        for name, body in docs[64:]:
            (cfg_a.corpus_root / name).write_text(body, encoding="utf-8")
        update_pipeline_once(sess_a)
    finally:
        close_pipeline(sess_a)

    sess_b = start_pipeline(cfg_b, embed_fn=tracked_embed, vector_dim=12, live=False)
    try:
        update_pipeline_once(sess_b)
    finally:
        close_pipeline(sess_b)

    assert max(batch_sizes_seen) <= 64

    with _open_ro(cfg_a.db_path) as conn_a, _open_ro(cfg_b.db_path) as conn_b:
        sel = (
            "SELECT chunk_id, embedding, rel_path, source_hash, start_line, end_line, "
            "char_start, char_end, ast_island_start, ast_island_end, title, tokens_json, entities_json "
            "FROM chunks ORDER BY chunk_id"
        )
        rows_a = conn_a.execute(sel).fetchall()
        rows_b = conn_b.execute(sel).fetchall()
        assert len(rows_a) == 128
        assert rows_a == rows_b

        fts_a = conn_a.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'collaborates' ORDER BY rowid"
        ).fetchall()
        fts_b = conn_b.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'collaborates' ORDER BY rowid"
        ).fetchall()
        assert len(fts_a) == 128
        assert fts_a == fts_b


def test_live_mode_reconciles_disk_changes_and_allows_concurrent_wal_reads(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "seed.md").write_text("# Seed\nInitial quartz oscillator specification.\n", encoding="utf-8")

    session = start_pipeline(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), vector_dim=8, live=True)
    try:
        assert wait_pipeline_ready(session, timeout=20.0)
        with _open_ro(cfg.db_path) as reader:
            assert reader.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
            v0 = reader.execute("PRAGMA data_version").fetchone()[0]

            # Add live file while WAL reader remains open and polls concurrently
            (cfg.corpus_root / "live_added.md").write_text(
                "# Live\nSuperconducting niobium cavity resonance test.\n",
                encoding="utf-8",
            )
            deadline = time.monotonic() + 15.0
            seen_two = False
            while time.monotonic() < deadline:
                cnt = reader.execute("SELECT count(*) FROM chunks").fetchone()[0]
                fts_hit = reader.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'niobium'").fetchall()
                if cnt == 2 and len(fts_hit) == 1:
                    seen_two = True
                    break
                time.sleep(0.1)
            assert seen_two
            v1 = reader.execute("PRAGMA data_version").fetchone()[0]
            assert v1 >= v0
    finally:
        close_pipeline(session)


def test_loopback_dspy_embedder_pipeline(tmp_path: Path) -> None:
    seen_inputs: list[str] = []

    class _EmbedHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_in = payload["input"]
            texts = [raw_in] if isinstance(raw_in, str) else [str(x) for x in raw_in]
            seen_inputs.extend(texts)
            vecs = _hash_embed(texts, dim=10) * 3.0  # Unnormalized to verify pipeline normalizes to L2=1
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vecs[i].tolist()} for i in range(len(texts))
                    ],
                    "model": payload.get("model", "embeddinggemma-2"),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbedHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    try:
        cfg = _make_cfg(tmp_path)
        cfg = YawConfig(
            workspace_root=cfg.workspace_root,
            corpus_root=cfg.corpus_root,
            db_path=cfg.db_path,
            lock_path=cfg.lock_path,
            state_dir=cfg.state_dir,
            log_path=cfg.log_path,
            status_path=cfg.status_path,
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="test-token",
            provider="openai",
            embedding_model="embeddinggemma-2",
            embedding_task_mode="distinct-titled",
        )
        assert cfg.corpus_root is not None and cfg.db_path is not None
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "note.md").write_text("# Photonic Core\nOptical waveguide routing.\n", encoding="utf-8")

        embed_fn = make_dspy_embed_fn(cfg)
        session = start_pipeline(cfg, embed_fn=embed_fn, vector_dim=10, live=False)
        try:
            update_pipeline_once(session)
            assert any(s.startswith("title: Photonic Core | text: ") for s in seen_inputs)
            with _open_ro(cfg.db_path) as conn:
                blob = conn.execute("SELECT embedding FROM chunks").fetchone()[0]
                vec = np.frombuffer(blob, dtype=np.float32)
                assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
        finally:
            close_pipeline(session)
    finally:
        server.shutdown()
        server.server_close()


def test_batch_pipeline_propagates_component_failures(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "alpha.md").write_text("# Alpha\nFailing embedder batch check.\n", encoding="utf-8")

    def failing_embed(texts: list[str]) -> NDArray[np.float32]:
        _ = texts
        raise RuntimeError("Embedder exploded in batch mode")

    session = start_pipeline(cfg, embed_fn=failing_embed, vector_dim=4, live=False)
    try:
        with pytest.raises(RuntimeError, match="Embedder exploded in batch mode"):
            update_pipeline_once(session)
    finally:
        close_pipeline(session)


def test_live_pipeline_propagates_component_failures(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "alpha.md").write_text("# Alpha\nFailing embedder live initial check.\n", encoding="utf-8")

    def failing_embed(texts: list[str]) -> NDArray[np.float32]:
        _ = texts
        raise RuntimeError("Model server is down 503")

    session = start_pipeline(cfg, embed_fn=failing_embed, vector_dim=4, live=True)
    try:
        with pytest.raises(RuntimeError, match="Model server is down 503"):
            wait_pipeline_ready(session, timeout=10.0)
    finally:
        close_pipeline(session)


def test_live_pipeline_propagates_incremental_failures_after_ready(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "alpha.md").write_text("# Alpha\nInitial healthy file.\n", encoding="utf-8")
    calls = 0

    def flaky_embed(texts: list[str]) -> NDArray[np.float32]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("Mid-stream failure after ready")
        return _hash_embed(texts, dim=4)

    session = start_pipeline(cfg, embed_fn=flaky_embed, vector_dim=4, live=True)
    try:
        assert wait_pipeline_ready(session, timeout=15.0)
        (cfg.corpus_root / "beta.md").write_text("# Beta\nTriggers second embed call.\n", encoding="utf-8")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and session.error is None:
            time.sleep(0.05)
        with pytest.raises(RuntimeError, match="Mid-stream failure after ready"):
            wait_pipeline_ready(session, timeout=1.0)
    finally:
        close_pipeline(session)
