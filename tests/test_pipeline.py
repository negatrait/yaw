"""Unmocked integration tests for declarative CocoIndex sqlite-vec + FTS5 pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlite_vec
from numpy.typing import NDArray

import yaw.pipeline as yaw_pipeline
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


def test_pipeline_uses_native_exception_handler_without_error_sink() -> None:
    assert "INDEXING_ERROR_SINK" not in vars(yaw_pipeline)
    assert not hasattr(yaw_pipeline, "INDEXING_ERROR_SINK")
    for exported in (
        "CORPUS_ROOT",
        "SQLITE_DB",
        "EMBED_FN",
        "PipelineSession",
        "open_managed_db",
        "make_dspy_embed_fn",
        "chunk_file_content",
        "start_pipeline",
        "update_pipeline_once",
        "wait_pipeline_ready",
        "close_pipeline",
        "list_user_tables",
    ):
        assert hasattr(yaw_pipeline, exported), f"Missing export {exported}"
    src = Path(yaw_pipeline.__file__).read_text(encoding="utf-8")
    assert "INDEXING_ERROR_SINK" not in src
    assert "_loop_runner" not in src
    assert "cocoindex._internal" not in src
    assert "exception_handler=" in src


def test_live_pipeline_clean_shutdown_no_task_or_context_leaks(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_cfg(tmp_path / "inproc")
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "seed.md").write_text("# Seed\nInitial content.\n", encoding="utf-8")
    capfd.readouterr()

    session = start_pipeline(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), vector_dim=8, live=True)
    try:
        assert wait_pipeline_ready(session, timeout=20.0)
        churn = cfg.corpus_root / "churn.md"
        churn.write_text("# Churn\nCreated right before shutdown.\n", encoding="utf-8")
        churn.write_text("# Churn\nUpdated right before shutdown.\n", encoding="utf-8")
    finally:
        close_pipeline(session)
    gc.collect()
    captured = capfd.readouterr()
    for bad in ("Failed to clean up pending stage token", "Task was destroyed", "different Context"):
        assert bad not in captured.err

    sub_root = (tmp_path / "subproc").as_posix()
    script = (
        "import gc\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from yaw.core import YawConfig\n"
        "from yaw.pipeline import close_pipeline, start_pipeline, wait_pipeline_ready\n"
        f"root = Path({sub_root!r})\n"
        "cfg = YawConfig(\n"
        "    workspace_root=root,\n"
        "    corpus_root=root / 'corpus',\n"
        "    db_path=root / '.yaw' / 'yaw.db',\n"
        "    lock_path=root / '.yaw' / 'yaw.db.lock',\n"
        "    state_dir=root / '.yaw' / 'coco_lmdb',\n"
        "    log_path=root / '.yaw' / 'yaw.log',\n"
        "    status_path=root / '.yaw' / 'status.json',\n"
        ")\n"
        "assert cfg.corpus_root is not None\n"
        "cfg.corpus_root.mkdir(parents=True)\n"
        "(cfg.corpus_root / 'seed.md').write_text('# Seed\\nInitial.\\n', encoding='utf-8')\n"
        "def emb(ts):\n"
        "    return np.ones((len(ts), 8), dtype=np.float32) / np.sqrt(8.0)\n"
        "sess = start_pipeline(cfg, embed_fn=emb, vector_dim=8, live=True)\n"
        "try:\n"
        "    assert wait_pipeline_ready(sess, timeout=20.0)\n"
        "    churn = cfg.corpus_root / 'churn.md'\n"
        "    churn.write_text('# Churn\\nCreated before close.\\n', encoding='utf-8')\n"
        "    churn.write_text('# Churn\\nUpdated before close.\\n', encoding='utf-8')\n"
        "finally:\n"
        "    close_pipeline(sess)\n"
        "gc.collect()\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    for bad in ("Failed to clean up pending stage token", "Task was destroyed", "different Context"):
        assert bad not in proc.stderr


def test_concurrent_live_sessions_isolated_on_close(tmp_path: Path) -> None:
    cfg_a = _make_cfg(tmp_path / "sess_a")
    cfg_b = _make_cfg(tmp_path / "sess_b")
    assert cfg_a.corpus_root is not None and cfg_b.corpus_root is not None
    assert cfg_b.db_path is not None
    cfg_a.corpus_root.mkdir(parents=True)
    cfg_b.corpus_root.mkdir(parents=True)
    (cfg_a.corpus_root / "a1.md").write_text("# Session A\nAlpha session seed.\n", encoding="utf-8")
    (cfg_b.corpus_root / "b1.md").write_text("# Session B\nBeta session seed.\n", encoding="utf-8")

    sa = start_pipeline(cfg_a, embed_fn=lambda ts: _hash_embed(ts, dim=8), vector_dim=8, live=True)
    sb = start_pipeline(cfg_b, embed_fn=lambda ts: _hash_embed(ts, dim=8), vector_dim=8, live=True)
    try:
        assert wait_pipeline_ready(sa, timeout=20.0)
        assert wait_pipeline_ready(sb, timeout=20.0)
        assert sa.loop is not sb.loop
        assert sa.env.event_loop is not sb.env.event_loop

        close_pipeline(sa)
        close_pipeline(sa)  # Idempotent close check

        assert sb.thread.is_alive()
        assert sb.env.event_loop.is_running()

        (cfg_b.corpus_root / "b2.md").write_text("# Session B2\nContinued live indexing zircon.\n", encoding="utf-8")
        with _open_ro(cfg_b.db_path) as reader:
            deadline = time.monotonic() + 15.0
            indexed = False
            while time.monotonic() < deadline:
                cnt = reader.execute("SELECT count(*) FROM chunks").fetchone()[0]
                fts = reader.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'zircon'").fetchall()
                if cnt == 2 and len(fts) == 1:
                    indexed = True
                    break
                time.sleep(0.05)
            assert indexed
    finally:
        close_pipeline(sa)
        close_pipeline(sb)


def test_live_pipeline_inside_asyncio_run_uses_dedicated_loop(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "async_caller")
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "seed.md").write_text("# Async Caller\nDedicated event loop test.\n", encoding="utf-8")

    async def _exercise() -> float:
        caller_loop = asyncio.get_running_loop()
        sess = start_pipeline(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), vector_dim=8, live=True)
        try:
            assert sess.env.event_loop is not caller_loop
            assert wait_pipeline_ready(sess, timeout=20.0)
        finally:
            t0 = time.monotonic()
            close_pipeline(sess)
            elapsed = time.monotonic() - t0
            close_pipeline(sess)  # Idempotent close check
        return elapsed

    elapsed = asyncio.run(_exercise())
    assert elapsed < 3.0


def test_memo_hit_and_cross_file_batching(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=500, chunk_overlap=0, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    sub_dir = cfg.corpus_root / "sub" / "nested"
    sub_dir.mkdir(parents=True)
    for i in range(9):
        (cfg.corpus_root / f"doc_{i:02d}.md").write_text(
            f"# Note {i}\nCross-file batching content {i}.\n", encoding="utf-8"
        )
    (sub_dir / "deep.md").write_text("# Deep\nCross-file batching content nested.\n", encoding="utf-8")

    batch_sizes: list[int] = []

    def tracked_embed(texts: list[str]) -> NDArray[np.float32]:
        batch_sizes.append(len(texts))
        time.sleep(0.05)
        return _hash_embed(texts, dim=8)

    sess = start_pipeline(cfg, embed_fn=tracked_embed, vector_dim=8, live=False)
    try:
        update_pipeline_once(sess)
        assert sum(batch_sizes) == 10
        assert len(batch_sizes) <= 3
        assert max(batch_sizes) <= 64

        with _open_ro(cfg.db_path) as conn:
            sel = "SELECT chunk_id, embedding, rel_path FROM chunks ORDER BY chunk_id"
            rows_before = conn.execute(sel).fetchall()
            fts_before = conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'batching' ORDER BY rowid"
            ).fetchall()
            assert len(rows_before) == 10
            assert len(fts_before) == 10
            assert "sub/nested/deep.md" in {r[2] for r in rows_before}

        calls_after_first = len(batch_sizes)
        update_pipeline_once(sess)
        assert len(batch_sizes) == calls_after_first  # 0 new embedder calls on unchanged memo hit

        with _open_ro(cfg.db_path) as conn:
            rows_after = conn.execute(sel).fetchall()
            fts_after = conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'batching' ORDER BY rowid"
            ).fetchall()
            assert rows_after == rows_before
            assert fts_after == fts_before
    finally:
        close_pipeline(sess)


def _make_http_cfg(root: Path, *, port: int, embedding_model: str, min_chunk_size: int = 10) -> YawConfig:
    base = _make_cfg(root, min_chunk_size=min_chunk_size)
    return YawConfig(
        workspace_root=base.workspace_root,
        corpus_root=base.corpus_root,
        db_path=base.db_path,
        lock_path=base.lock_path,
        state_dir=base.state_dir,
        log_path=base.log_path,
        status_path=base.status_path,
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        min_chunk_size=base.min_chunk_size,
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="test-token",
        provider="openai",
        embedding_model=embedding_model,
        embedding_task_mode="distinct-titled",
    )


@contextlib.contextmanager
def _loopback_embed_server(
    handler_fn: Callable[[str, list[str]], NDArray[np.float32]],
) -> Generator[int]:
    class _EmbedHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_in = payload["input"]
            texts = [raw_in] if isinstance(raw_in, str) else [str(x) for x in raw_in]
            model = str(payload.get("model", ""))
            vecs = handler_fn(model, texts)
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vecs[i].tolist()} for i in range(len(texts))
                    ],
                    "model": model,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbedHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_model_change_invalidates_memo_cache(tmp_path: Path) -> None:
    seen_calls: list[tuple[str, list[str]]] = []

    def _by_model(model: str, texts: list[str]) -> NDArray[np.float32]:
        seen_calls.append((model, list(texts)))
        vec = np.array([0.0, 1.0, 0.0, 0.0] if "model-b" in model else [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.tile(vec, (len(texts), 1))

    with _loopback_embed_server(_by_model) as port:
        cfg_a = _make_http_cfg(tmp_path, port=port, embedding_model="model-a")
        assert cfg_a.corpus_root is not None and cfg_a.db_path is not None
        cfg_a.corpus_root.mkdir(parents=True)
        (cfg_a.corpus_root / "alpha.md").write_text(
            "# Alpha Note\nSignal routing across photonic waveguide couplers stays deterministic.\n",
            encoding="utf-8",
        )

        s1 = start_pipeline(cfg_a, embed_fn=make_dspy_embed_fn(cfg_a), live=False)
        try:
            update_pipeline_once(s1)
        finally:
            close_pipeline(s1)

        calls_a = [c for c in seen_calls if "model-a" in c[0] and any("waveguide" in t for t in c[1])]
        assert len(calls_a) == 1
        with _open_ro(cfg_a.db_path) as conn:
            blob_a = conn.execute("SELECT embedding FROM chunks").fetchone()[0]
            assert np.allclose(np.frombuffer(blob_a, dtype=np.float32), [1.0, 0.0, 0.0, 0.0], atol=1e-5)

        count_before_b = len(seen_calls)
        cfg_b = _make_http_cfg(tmp_path, port=port, embedding_model="model-b")
        s2 = start_pipeline(cfg_b, embed_fn=make_dspy_embed_fn(cfg_b), live=False)
        try:
            update_pipeline_once(s2)
        finally:
            close_pipeline(s2)

        calls_b = [c for c in seen_calls[count_before_b:] if "model-b" in c[0] and any("waveguide" in t for t in c[1])]
        assert len(calls_b) == 1
        with _open_ro(cfg_b.db_path) as conn:
            blob_b = conn.execute("SELECT embedding FROM chunks").fetchone()[0]
            assert np.allclose(np.frombuffer(blob_b, dtype=np.float32), [0.0, 1.0, 0.0, 0.0], atol=1e-5)


def test_restart_memo_hit_with_make_dspy_embed_fn(tmp_path: Path) -> None:
    http_calls: list[list[str]] = []

    def _deterministic_embed(model: str, texts: list[str]) -> NDArray[np.float32]:
        _ = model
        http_calls.append(list(texts))
        return _hash_embed(texts, dim=8)

    with _loopback_embed_server(_deterministic_embed) as port:
        cfg = _make_http_cfg(tmp_path, port=port, embedding_model="model-a")
        assert cfg.corpus_root is not None and cfg.db_path is not None
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "alpha.md").write_text(
            "# Alpha Note\nIncremental memo hits must survive restarts across new DSPy embedders.\n",
            encoding="utf-8",
        )

        s1 = start_pipeline(cfg, embed_fn=make_dspy_embed_fn(cfg), live=False)
        try:
            update_pipeline_once(s1)
        finally:
            close_pipeline(s1)

        calls_after_first = len(http_calls)
        assert calls_after_first > 0
        with _open_ro(cfg.db_path) as conn:
            rows_first = conn.execute("SELECT chunk_id, rel_path, embedding FROM chunks ORDER BY chunk_id").fetchall()
            fts_first = conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'memo' ORDER BY rowid"
            ).fetchall()
            assert rows_first and fts_first

        # Fresh start_pipeline + make_dspy_embed_fn on existing schema (_detect_existing_vec_dim -> 8)
        s2 = start_pipeline(cfg, embed_fn=make_dspy_embed_fn(cfg), live=False)
        try:
            update_pipeline_once(s2)
        finally:
            close_pipeline(s2)

        assert len(http_calls) == calls_after_first
        with _open_ro(cfg.db_path) as conn:
            rows_second = conn.execute("SELECT chunk_id, rel_path, embedding FROM chunks ORDER BY chunk_id").fetchall()
            fts_second = conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'memo' ORDER BY rowid"
            ).fetchall()
            assert rows_second == rows_first
            assert fts_second == fts_first
