"""Unmocked integration tests for declarative CocoIndex 5-table SQLite, unicode61 FTS5, and SoA RaBitQ pipeline."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gc
import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

import numpy as np
import pytest
from cocoindex.ops.code import CodeSource
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from numpy.typing import NDArray

import yaw.pipeline as yaw_pipeline
from yaw.core import ChunkPointer, YawConfig, YawRuntimeError, format_query_for_embedding, load_config
from yaw.pipeline import (
    SinkMutationTracker,
    chunk_file_content,
    close_pipeline,
    drop_pipeline,
    list_user_tables,
    make_dspy_embed_fn,
    open_pipeline_lifespan,
    reconcile_index_tables,
    start_pipeline,
    update_pipeline_once,
)
from yaw.retrieval import hydrate_passage, open_readonly_db


def _hash_embed(texts: list[str], *, dim: int = 16) -> NDArray[np.float32]:
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = np.frombuffer(digest * ((dim * 4 // len(digest)) + 1), dtype=np.uint8)[:dim].astype(np.float32) - 127.5
        norm = max(float(np.linalg.norm(vals)), 1e-12)
        rows[row_idx] = vals / norm
    return rows


def _make_cfg(
    root: Path,
    *,
    chunk_size: int = 300,
    chunk_overlap: int = 40,
    min_chunk_size: int = 20,
    embed_batch_size: int | None = None,
) -> YawConfig:
    cfg = YawConfig(
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
    return cfg if embed_batch_size is None else replace(cfg, embed_batch_size=embed_batch_size)


def _open_ro(db_path: Path) -> sqlite3.Connection:
    return open_readonly_db(db_path)


def test_declarative_five_table_schema_edit_and_delete(tmp_path: Path) -> None:
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

    session = start_pipeline(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=16), live=False)
    try:
        update_pipeline_once(session)
        with _open_ro(cfg.db_path) as conn:
            assert list_user_tables(conn) == (
                "chunk_fts",
                "chunk_vocab",
                "chunks",
                "citation_graph",
                "rabitq_index",
            )
            all_names = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
            assert "yaw_stage1" not in all_names
            ddl = str(conn.execute("SELECT sql FROM sqlite_master WHERE name='chunks'").fetchone()[0])
            assert "vec0" not in ddl.lower()
            cols = {str(r[1]) for r in conn.execute('PRAGMA table_info("chunks")').fetchall()}
            assert cols == {
                "chunk_id",
                "embedding_f16",
                "rel_path",
                "source_hash",
                "start_line",
                "end_line",
                "char_start",
                "char_end",
                "ast_island_start",
                "ast_island_end",
                "title",
            }
            assert "tokens_json" not in cols and "entities_json" not in cols

            fts_ddl = str(conn.execute("SELECT sql FROM sqlite_master WHERE name='chunk_fts'").fetchone()[0])
            assert "unicode61 remove_diacritics 1" in fts_ddl

            rows = conn.execute(
                "SELECT chunk_id, embedding_f16, rel_path, source_hash, start_line, end_line, title "
                "FROM chunks ORDER BY rel_path"
            ).fetchall()
            assert len(rows) == 2
            assert conn.execute("SELECT count(*) FROM chunk_fts").fetchone()[0] == 2

            # Verify IEEE <f2> float16 LE blob round-trip matches embedded unit vector
            raw_a = doc_a.read_text(encoding="utf-8")
            expected_f16 = _hash_embed([f"title: none | text: {raw_a.strip()}"], dim=16)[0].astype("<f2").tobytes()
            assert bytes(rows[0][1]) == expected_f16
            vec_a = np.frombuffer(bytes(rows[0][1]), dtype="<f2").astype(np.float32)
            assert vec_a.shape == (16,) and np.all(np.isfinite(vec_a))
            assert abs(float(np.linalg.norm(vec_a)) - 1.0) < 5e-3

            # Verify FTS5 + fts5vocab + single-row rabitq_index & citation_graph
            fts_hits = conn.execute("SELECT rowid, title FROM chunk_fts WHERE chunk_fts MATCH 'lunar'").fetchall()
            assert [r[0] for r in fts_hits] == [rows[0][0]]
            vocab_row = conn.execute("SELECT doc FROM chunk_vocab WHERE term = 'lunar'").fetchone()
            assert vocab_row is not None and int(vocab_row[0]) == 1

            r_row = conn.execute(
                "SELECT n_chunks, dim, embedding_model, task_mode, n_words, "
                "length(codes_blob), length(ip_norm_blob), length(mean_blob), d_eff_shannon "
                "FROM rabitq_index WHERE id = 1"
            ).fetchone()
            assert r_row is not None
            assert tuple(r_row[:5]) == (2, 16, cfg.embedding_model, cfg.embedding_task_mode, 1)
            assert tuple(r_row[5:8]) == (8 * 2, 4 * 2, 16 * 4)
            assert float(r_row[8]) >= 2.0

            g_row = conn.execute(
                "SELECT n_chunks, length(doc_titles_blob), length(graph_blob) FROM citation_graph WHERE id = 1"
            ).fetchone()
            assert g_row is not None and g_row[0] == 2 and g_row[1] > 0 and g_row[2] > 0

        # Idempotent reconciliation emits False when chunks have not changed
        assert (
            reconcile_index_tables(
                session.conn_mgr,
                embedding_model=cfg.embedding_model,
                task_mode=cfg.embedding_task_mode,
                corpus_root=corpus,
            )
            is False
        )

        # Mutate doc_a and unlink doc_b -> assert clean reconciliation across all 5 tables
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
            assert conn.execute("SELECT n_chunks FROM rabitq_index WHERE id = 1").fetchone()[0] == 1
            assert conn.execute("SELECT n_chunks FROM citation_graph WHERE id = 1").fetchone()[0] == 1

        # Deleting remaining file clears chunks, rabitq_index, and citation_graph cleanly
        doc_a.unlink()
        update_pipeline_once(session)
        with _open_ro(cfg.db_path) as conn:
            assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM rabitq_index").fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM citation_graph").fetchone()[0] == 0
    finally:
        close_pipeline(session)


def test_microbatch_64_plus_64_equals_128_invariance(tmp_path: Path) -> None:
    root_a = tmp_path / "batch_64_64"
    root_b = tmp_path / "batch_128"
    cfg_a = _make_cfg(root_a, chunk_size=500, chunk_overlap=0, min_chunk_size=10, embed_batch_size=64)
    cfg_b = _make_cfg(root_b, chunk_size=500, chunk_overlap=0, min_chunk_size=10, embed_batch_size=64)
    assert cfg_a.corpus_root is not None and cfg_b.corpus_root is not None
    assert cfg_a.db_path is not None and cfg_b.db_path is not None
    cfg_a.corpus_root.mkdir(parents=True)
    cfg_b.corpus_root.mkdir(parents=True)

    batch_sizes_seen: list[int] = []

    def tracked_embed(texts: list[str]) -> NDArray[np.float32]:
        batch_sizes_seen.append(len(texts))
        return _hash_embed(texts, dim=70)

    docs = [
        (f"doc_{i:03d}.md", f"# Note {i}\nEntity{i} collaborates with Laboratory{i} in sector {i % 7}.\n")
        for i in range(128)
    ]
    for name, body in docs[:64]:
        (cfg_a.corpus_root / name).write_text(body, encoding="utf-8")
    for name, body in docs:
        (cfg_b.corpus_root / name).write_text(body, encoding="utf-8")

    sess_a = start_pipeline(cfg_a, embed_fn=tracked_embed, live=False)
    try:
        update_pipeline_once(sess_a)
        for name, body in docs[64:]:
            (cfg_a.corpus_root / name).write_text(body, encoding="utf-8")
        update_pipeline_once(sess_a)
    finally:
        close_pipeline(sess_a)

    sess_b = start_pipeline(cfg_b, embed_fn=tracked_embed, live=False)
    try:
        update_pipeline_once(sess_b)
    finally:
        close_pipeline(sess_b)

    assert max(batch_sizes_seen) <= cfg_a.embed_batch_size

    with _open_ro(cfg_a.db_path) as conn_a, _open_ro(cfg_b.db_path) as conn_b:
        sel = (
            "SELECT chunk_id, embedding_f16, rel_path, source_hash, start_line, end_line, "
            "char_start, char_end, ast_island_start, ast_island_end, title "
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

        vocab_a = conn_a.execute("SELECT term, doc, cnt FROM chunk_vocab ORDER BY term").fetchall()
        vocab_b = conn_b.execute("SELECT term, doc, cnt FROM chunk_vocab ORDER BY term").fetchall()
        assert vocab_a == vocab_b

        rab_a = conn_a.execute("SELECT * FROM rabitq_index").fetchall()
        rab_b = conn_b.execute("SELECT * FROM rabitq_index").fetchall()
        assert len(rab_a) == 1 and rab_a == rab_b
        assert rab_a[0][1] == 128 and rab_a[0][2] == 70 and rab_a[0][7] == 2  # n_words == (70 + 63) // 64
        assert len(bytes(rab_a[0][8])) == 2 * 8 * 128

        cg_a = conn_a.execute("SELECT * FROM citation_graph").fetchall()
        cg_b = conn_b.execute("SELECT * FROM citation_graph").fetchall()
        assert len(cg_a) == 1 and cg_a == cg_b


def test_live_mode_reconciles_disk_changes_and_allows_concurrent_wal_reads(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    corpus_root, db_path = cfg.corpus_root, cfg.db_path
    corpus_root.mkdir(parents=True)
    (corpus_root / "seed.md").write_text("# Seed\nInitial quartz oscillator specification.\n", encoding="utf-8")

    async def _run() -> None:
        async with open_pipeline_lifespan(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), live=True) as session:
            assert session.ready_event is not None
            await asyncio.wait_for(session.ready_event.wait(), timeout=20.0)
            with _open_ro(db_path) as reader:
                assert reader.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
                v0 = reader.execute("PRAGMA data_version").fetchone()[0]

                (corpus_root / "live_added.md").write_text(
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
                    await asyncio.sleep(0.05)
                assert seen_two
                v1 = reader.execute("PRAGMA data_version").fetchone()[0]
                assert v1 >= v0

    asyncio.run(_run())


def test_loopback_dspy_embedder_pipeline(tmp_path: Path) -> None:
    seen_inputs: list[str] = []

    class _EmbedHandler(BaseHTTPRequestHandler):
        @override
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
        session = start_pipeline(cfg, embed_fn=embed_fn, live=False)
        try:
            update_pipeline_once(session)
            assert any(s.startswith("title: none | text: # Photonic Core\n") for s in seen_inputs)
            with _open_ro(cfg.db_path) as conn:
                blob = conn.execute("SELECT embedding_f16 FROM chunks").fetchone()[0]
                vec = np.frombuffer(blob, dtype="<f2").astype(np.float32)
                assert abs(float(np.linalg.norm(vec)) - 1.0) < 5e-3
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

    session = start_pipeline(cfg, embed_fn=failing_embed, live=False)
    try:
        with pytest.raises(RuntimeError, match="Embedder exploded in batch mode"):
            update_pipeline_once(session)
    finally:
        close_pipeline(session)


def test_live_pipeline_propagates_component_failures(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "alpha.md").write_text("# Alpha\nFailing embedder live initial check.\n", encoding="utf-8")

    def failing_embed(texts: list[str]) -> NDArray[np.float32]:
        _ = texts
        raise RuntimeError("Model server is down 503")

    async def _run() -> None:
        err_evt = asyncio.Event()
        errors: list[Exception] = []

        def _on_err(exc: Exception) -> None:
            errors.append(exc)
            err_evt.set()

        async with open_pipeline_lifespan(cfg, embed_fn=failing_embed, live=True, on_error=_on_err) as session:
            assert session.ready_event is not None
            await asyncio.wait_for(session.ready_event.wait(), timeout=10.0)
            await asyncio.wait_for(err_evt.wait(), timeout=2.0)
            assert session.error is not None
            assert "Model server is down 503" in str(session.error)
            assert errors and "Model server is down 503" in str(errors[0])

    capfd.readouterr()
    asyncio.run(_run())
    time.sleep(0.05)
    captured = capfd.readouterr()
    assert "Failed to clean up pending stage token" not in captured.err
    assert "environment is closed" not in captured.err


def test_live_pipeline_propagates_incremental_failures_after_ready(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    corpus_root = cfg.corpus_root
    corpus_root.mkdir(parents=True)
    (corpus_root / "alpha.md").write_text("# Alpha\nInitial healthy file.\n", encoding="utf-8")
    calls = 0

    def flaky_embed(texts: list[str]) -> NDArray[np.float32]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("Mid-stream failure after ready")
        return _hash_embed(texts, dim=4)

    async def _run() -> None:
        err_evt = asyncio.Event()

        async with open_pipeline_lifespan(
            cfg, embed_fn=flaky_embed, live=True, on_error=lambda _exc: err_evt.set()
        ) as session:
            assert session.ready_event is not None
            await asyncio.wait_for(session.ready_event.wait(), timeout=15.0)
            (corpus_root / "beta.md").write_text("# Beta\nTriggers second embed call.\n", encoding="utf-8")
            await asyncio.wait_for(err_evt.wait(), timeout=10.0)
            assert session.error is not None
            assert "Mid-stream failure after ready" in str(session.error)

    capfd.readouterr()
    asyncio.run(_run())
    time.sleep(0.05)
    captured = capfd.readouterr()
    assert "Failed to clean up pending stage token" not in captured.err
    assert "environment is closed" not in captured.err


def test_pipeline_uses_native_exception_handler_without_error_sink() -> None:
    assert "INDEXING_ERROR_SINK" not in vars(yaw_pipeline)
    assert not hasattr(yaw_pipeline, "INDEXING_ERROR_SINK")
    assert not hasattr(yaw_pipeline, "wait_pipeline_ready")
    for exported in (
        "CORPUS_ROOT",
        "SQLITE_DB",
        "EMBEDDER",
        "SINK_TRACKER",
        "SinkMutationTracker",
        "DspyBatchEmbedder",
        "PipelineSession",
        "open_managed_db",
        "open_pipeline_lifespan",
        "make_dspy_embed_fn",
        "chunk_file_content",
        "reconcile_index_tables",
        "start_pipeline",
        "update_pipeline_once",
        "drop_pipeline",
        "close_pipeline",
        "list_user_tables",
    ):
        assert hasattr(yaw_pipeline, exported), f"Missing export {exported}"
    assert yaw_pipeline.__file__ is not None
    sources = [
        py_file.read_text(encoding="utf-8") for py_file in sorted(Path(yaw_pipeline.__file__).parent.rglob("*.py"))
    ]
    for forbidden in ("INDEXING_ERROR_SINK", "_loop_runner", "cocoindex._internal", "yaw-cocoindex-session"):
        assert all(forbidden not in s for s in sources)
    assert all("update_blocking" not in s and "drop_blocking" not in s for s in sources)
    assert any("exception_handler=" in s for s in sources)


def test_live_pipeline_clean_shutdown_no_task_or_context_leaks(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_cfg(tmp_path / "inproc")
    assert cfg.corpus_root is not None
    corpus_root = cfg.corpus_root
    corpus_root.mkdir(parents=True)
    (corpus_root / "seed.md").write_text("# Seed\nInitial content.\n", encoding="utf-8")
    capfd.readouterr()

    async def _inproc() -> None:
        async with open_pipeline_lifespan(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), live=True) as session:
            assert session.ready_event is not None
            await asyncio.wait_for(session.ready_event.wait(), timeout=20.0)
            churn = corpus_root / "churn.md"
            churn.write_text("# Churn\nCreated right before shutdown.\n", encoding="utf-8")
            churn.write_text("# Churn\nUpdated right before shutdown.\n", encoding="utf-8")

    asyncio.run(_inproc())
    gc.collect()
    captured = capfd.readouterr()
    for bad in ("Failed to clean up pending stage token", "Task was destroyed", "different Context"):
        assert bad not in captured.err

    sub_root = (tmp_path / "subproc").as_posix()
    script = (
        "import asyncio, gc\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from yaw.core import YawConfig\n"
        "from yaw.pipeline import open_pipeline_lifespan\n"
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
        "async def run():\n"
        "    async with open_pipeline_lifespan(cfg, embed_fn=emb, live=True) as sess:\n"
        "        await asyncio.wait_for(sess.ready_event.wait(), timeout=20.0)\n"
        "        churn = cfg.corpus_root / 'churn.md'\n"
        "        churn.write_text('# Churn\\nCreated before close.\\n', encoding='utf-8')\n"
        "        churn.write_text('# Churn\\nUpdated before close.\\n', encoding='utf-8')\n"
        "asyncio.run(run())\n"
        "gc.collect()\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    for bad in ("Failed to clean up pending stage token", "Task was destroyed", "different Context"):
        assert bad not in proc.stderr


def test_concurrent_live_sessions_isolated_on_close(tmp_path: Path) -> None:
    cfg_a = _make_cfg(tmp_path / "sess_a")
    cfg_b = _make_cfg(tmp_path / "sess_b")
    assert cfg_a.corpus_root is not None and cfg_b.corpus_root is not None and cfg_b.db_path is not None
    cfg_a.corpus_root.mkdir(parents=True)
    cfg_b.corpus_root.mkdir(parents=True)
    (cfg_a.corpus_root / "a1.md").write_text("# Session A\nAlpha session seed.\n", encoding="utf-8")
    (cfg_b.corpus_root / "b1.md").write_text("# Session B\nBeta session seed.\n", encoding="utf-8")

    ready_a = threading.Event()
    ready_b = threading.Event()
    stop_a = threading.Event()
    stop_b = threading.Event()
    loops: dict[str, asyncio.AbstractEventLoop] = {}

    def _run_sess(name: str, cfg: YawConfig, ready_ev: threading.Event, stop_ev: threading.Event) -> None:
        async def _inner() -> None:
            async with open_pipeline_lifespan(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), live=True) as sess:
                assert sess.env is not None and sess.ready_event is not None
                loops[name] = sess.env.event_loop
                await asyncio.wait_for(sess.ready_event.wait(), timeout=20.0)
                ready_ev.set()
                while not stop_ev.is_set():
                    await asyncio.sleep(0.05)

        asyncio.run(_inner())

    ta = threading.Thread(target=_run_sess, args=("a", cfg_a, ready_a, stop_a), daemon=True)
    tb = threading.Thread(target=_run_sess, args=("b", cfg_b, ready_b, stop_b), daemon=True)
    ta.start()
    tb.start()
    try:
        assert ready_a.wait(timeout=20.0) and ready_b.wait(timeout=20.0)
        assert loops["a"] is not loops["b"]
        stop_a.set()
        ta.join(timeout=10.0)
        assert not ta.is_alive() and tb.is_alive()
        assert loops["b"].is_running()

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
        stop_a.set()
        stop_b.set()
        ta.join(timeout=5.0)
        tb.join(timeout=5.0)


def test_live_pipeline_runs_on_caller_loop_preserves_bystander_and_tracks_real_mutations(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "shared_loop")
    assert cfg.corpus_root is not None
    corpus_root = cfg.corpus_root
    corpus_root.mkdir(parents=True)
    (corpus_root / "seed.md").write_text("# Seed\nInitial content.\n", encoding="utf-8")

    with pytest.raises(YawRuntimeError) as exc_live:
        start_pipeline(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), live=True)
    assert exc_live.value.code == "INVALID_INPUT"

    reconciled_events: list[tuple[bool, bool]] = []

    async def _on_reconciled(changed: bool, is_initial: bool) -> None:
        reconciled_events.append((changed, is_initial))

    async def _exercise() -> float:
        caller_loop = asyncio.get_running_loop()
        bystander_stop = asyncio.Event()

        async def _bystander() -> str:
            await bystander_stop.wait()
            return "intact"

        bystander_task = asyncio.create_task(_bystander())
        try:
            async with open_pipeline_lifespan(
                cfg,
                embed_fn=lambda ts: _hash_embed(ts, dim=8),
                live=True,
                on_reconciled=_on_reconciled,
            ) as sess:
                assert sess.env is not None and sess.env.event_loop is caller_loop
                assert sess.ready_event is not None and sess.sink_tracker is not None
                assert isinstance(sess.sink_tracker, SinkMutationTracker)
                await asyncio.wait_for(sess.ready_event.wait(), timeout=20.0)
                seq_after_init = sess.sink_tracker.seq
                events_after_init = len(reconciled_events)

                # Empty and excluded files must not advance sink tracker or trigger reconciliation
                (corpus_root / "empty.md").write_text("   \n", encoding="utf-8")
                (corpus_root / "ignored.bin").write_bytes(b"\x00\x01\x02")
                await asyncio.sleep(0.4)
                assert sess.sink_tracker.seq == seq_after_init
                assert len(reconciled_events) == events_after_init

                # Real create/edit/delete advance sink_tracker strictly
                target = corpus_root / "mutated.md"
                target.write_text("# Alpha\nFirst mutation line.\n", encoding="utf-8")
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline and len(reconciled_events) <= events_after_init:
                    await asyncio.sleep(0.05)
                assert len(reconciled_events) > events_after_init
                assert sess.sink_tracker.seq > seq_after_init
                t0 = time.monotonic()
            elapsed = time.monotonic() - t0
            assert not bystander_task.done()
            bystander_stop.set()
            assert await bystander_task == "intact"
            return elapsed
        finally:
            bystander_stop.set()
            if not bystander_task.done():
                bystander_task.cancel()
                await asyncio.gather(bystander_task, return_exceptions=True)

    elapsed = asyncio.run(_exercise())
    assert elapsed < 3.0


def test_live_pipeline_watch_non_worker_error_notifies_and_unblocks_ready_event(tmp_path: Path) -> None:
    class _CustomWatchError(Exception):
        pass

    class _SimulatedRustPanic(BaseException):
        pass

    for label, exc_cls in (("custom_exc", _CustomWatchError), ("rust_panic", _SimulatedRustPanic)):
        cfg = _make_cfg(tmp_path / label)
        assert cfg.corpus_root is not None
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "seed.md").write_text("# Seed\nWatch pipeline error check.\n", encoding="utf-8")

        async def _run(cur_cfg: YawConfig = cfg, err_type: type[BaseException] = exc_cls) -> None:
            errors: list[Exception] = []

            async def _boom_reconciled(_changed: bool, _is_initial: bool) -> None:
                raise err_type("watch consumer exploded")

            async with open_pipeline_lifespan(
                cur_cfg,
                embed_fn=lambda ts: _hash_embed(ts, dim=8),
                live=True,
                on_reconciled=_boom_reconciled,
                on_error=errors.append,
            ) as sess:
                assert sess.ready_event is not None
                await asyncio.wait_for(sess.ready_event.wait(), timeout=15.0)
                assert isinstance(sess.error, RuntimeError)
                assert isinstance(sess.error.__cause__, err_type)
                assert len(errors) == 1
                assert errors[0] is sess.error

        asyncio.run(_run())


def test_dspy_batch_embedder_acall_batches_errors_and_drop_pipeline(tmp_path: Path) -> None:
    seen_calls: list[list[str]] = []

    class _EmbedHandler(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_in = payload["input"]
            texts = [raw_in] if isinstance(raw_in, str) else [str(x) for x in raw_in]
            seen_calls.append(list(texts))
            vecs = (_hash_embed(texts, dim=8) * 2.0).astype(np.float32)
            # Emit reversed index order + base64 payload for "b64" inputs to verify dspy.Embedder handling
            items = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": (
                        base64.b64encode(vecs[i].astype("<f4").tobytes()).decode("ascii")
                        if "b64" in texts[i]
                        else vecs[i].tolist()
                    ),
                }
                for i in reversed(range(len(texts)))
            ]
            body = json.dumps(
                {
                    "object": "list",
                    "data": items,
                    "model": str(payload.get("model", "")),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with _loopback_embed_server_handler(_EmbedHandler) as port:
        cfg = replace(
            _make_http_cfg(tmp_path / "acall_ok", port=port, embedding_model="acall-model"),
            embed_batch_size=2,
        )
        embedder = make_dspy_embed_fn(cfg)

        async def _check_acall() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
            empty = await embedder.acall([])
            assert empty.shape == (0, 0)
            v1 = await embedder.acall(["one", "two", "b64-three"])
            v2 = await embedder.acall(["four"])
            return v1, v2

        v1, v2 = asyncio.run(_check_acall())
        assert v1.shape == (3, 8)
        assert v2.shape == (1, 8)
        assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)
        assert np.allclose(v1, _hash_embed(["one", "two", "b64-three"], dim=8), atol=1e-5)
        assert np.allclose(v2, embedder(["four"]), atol=1e-5)
        assert seen_calls == [["one", "two"], ["b64-three"], ["four"], ["four"]]

        assert cfg.corpus_root is not None and cfg.db_path is not None
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "note.md").write_text("# Note\nDrop pipeline check.\n", encoding="utf-8")
        sess = start_pipeline(cfg, embed_fn=embedder, live=False)
        try:
            update_pipeline_once(sess)
            with _open_ro(cfg.db_path) as conn:
                assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
            drop_pipeline(sess)
            with _open_ro(cfg.db_path) as conn:
                remaining = set(list_user_tables(conn))
                assert "chunks" not in remaining and "chunk_fts" not in remaining
        finally:
            close_pipeline(sess)

    # HTTP 500 fails fast (max_retries=0 -> 1 attempt) and symmetrically across __call__ and acall
    err_attempts = 0

    class _ErrHandler(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            nonlocal err_attempts
            err_attempts += 1
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(length)
            body = json.dumps({"error": {"message": "upstream fail"}}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with _loopback_embed_server_handler(_ErrHandler) as err_port:
        err_cfg = _make_http_cfg(tmp_path / "acall_err", port=err_port, embedding_model="err-model")
        err_emb = make_dspy_embed_fn(err_cfg)

        with pytest.raises(Exception) as sync_exc:
            err_emb(["boom"])
        assert type(sync_exc.value).__name__ == "InternalServerError"
        assert err_attempts == 1

        async def _check_err() -> None:
            with pytest.raises(Exception) as async_exc:
                await err_emb.acall(["boom"])
            assert type(async_exc.value).__name__ == "InternalServerError"

        asyncio.run(_check_err())
        assert err_attempts == 2


@contextlib.contextmanager
def _loopback_embed_server_handler(handler_cls: type[BaseHTTPRequestHandler]) -> Generator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_memo_hit_and_cross_file_batching(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=500, chunk_overlap=0, min_chunk_size=10, embed_batch_size=16)
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

    sess = start_pipeline(cfg, embed_fn=tracked_embed, live=False)
    try:
        update_pipeline_once(sess)
        assert sum(batch_sizes) == 10
        assert max(batch_sizes) > 1
        assert len(batch_sizes) <= 3
        assert max(batch_sizes) <= cfg.embed_batch_size

        with _open_ro(cfg.db_path) as conn:
            sel = "SELECT chunk_id, embedding_f16, rel_path FROM chunks ORDER BY chunk_id"
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
        @override
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


@pytest.mark.parametrize("asynchronous", [False, True])
def test_embedding_requests_obey_configured_batch_size(tmp_path: Path, asynchronous: bool) -> None:
    requests: list[list[str]] = []

    def receive(_model: str, texts: list[str]) -> NDArray[np.float32]:
        requests.append(texts)
        return _hash_embed(texts, dim=8)

    # Same shape as optimizer query pre-embedding: distinct inputs, not document chunks.
    texts = [format_query_for_embedding(f"Question {i}: " + "routing " * 35) for i in range(16)]
    with _loopback_embed_server(receive) as port:
        cfg = replace(
            _make_http_cfg(tmp_path, port=port, embedding_model="budget-model"),
            chunk_size=1000,
            embed_batch_size=3,
        )
        embed = make_dspy_embed_fn(cfg)
        vectors = asyncio.run(asyncio.to_thread(embed, texts)) if asynchronous else embed(texts)

    assert [text for batch in requests for text in batch] == texts
    assert all(len(batch) <= cfg.embed_batch_size for batch in requests)
    assert np.allclose(vectors, _hash_embed(texts, dim=8))


@pytest.mark.parametrize("suffix", ["txt", "md", "json", "py"])
def test_document_chunks_obey_byte_budget(suffix: str) -> None:
    text = "# Heading\n" + 'Long prose, café 漢字 🧠 and code: {"key": "value"}. ' * 350
    chunks = chunk_file_content(f"sample.{suffix}", text, chunk_size=1000, chunk_overlap=150)
    assert len(chunks) > 1
    assert all(0 < len(chunk.body.encode("utf-8")) <= 1000 for chunk in chunks)


@pytest.mark.parametrize("suffix", ["md", "json"])
def test_native_ast_is_the_only_chunk_boundary_authority(tmp_path: Path, suffix: str) -> None:
    body = "Résumé 漢字 🧠 with a literal \\n and quoted evidence. " * 90
    text = (
        f"# Incident\n\n{body}\n\n```sh\n# not a title\nshow counters\n```\n\n## Recovery\nDone.\n"
        if suffix == "md"
        else json.dumps({"title": "Incident", "text": body}, ensure_ascii=False, indent=2)
    )
    name = f"source.{suffix}"
    (tmp_path / name).write_text(text, encoding="utf-8")
    native = RecursiveSplitter().split(
        CodeSource(text, language=detect_code_language(filename=name)),
        chunk_size=1000,
        chunk_overlap=150,
        min_chunk_size=100,
    )
    chunks = chunk_file_content(name, text)
    assert [chunk.body for chunk in chunks] == [chunk.text for chunk in native if chunk.text.strip()]
    assert all(chunk.title == "" for chunk in chunks)
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.body
        pointer = ChunkPointer(
            chunk_id=chunk.chunk_id,
            rel_path=name,
            source_hash=chunk.source_hash,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        hydrated = hydrate_passage(pointer, tmp_path)
        assert not hydrated.stale_on_disk
        assert hydrated.text == chunk.body


def test_indivisible_native_chunk_fails_instead_of_manual_resplitting() -> None:
    with pytest.raises(ValueError, match="native chunk is 5000 bytes, exceeding chunk_size=1000"):
        chunk_file_content("unbroken.txt", "x" * 5000)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_batching_preserves_long_queries_and_ignores_chunk_size(tmp_path: Path, asynchronous: bool) -> None:
    requests: list[list[str]] = []

    def receive(_model: str, texts: list[str]) -> NDArray[np.float32]:
        requests.append(texts)
        return _hash_embed(texts, dim=8)

    texts = ["a" * 350, "b" * 350, format_query_for_embedding("long query " * 500), "tail"]
    with _loopback_embed_server(receive) as port:
        base = replace(
            _make_http_cfg(tmp_path, port=port, embedding_model="budget-model"),
            embed_batch_size=2,
        )
        for chunk_size in (100, 5000):
            requests.clear()
            embed = make_dspy_embed_fn(replace(base, chunk_size=chunk_size))
            vectors = asyncio.run(asyncio.to_thread(embed, texts)) if asynchronous else embed(texts)
            assert requests == [texts[:2], texts[2:]]
            assert np.allclose(vectors, _hash_embed(texts, dim=8))


@pytest.mark.parametrize("field", ["embed_batch_size", "embed_max_in_flight"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_embedding_budget_fails_at_configuration(tmp_path: Path, field: str, value: int) -> None:
    with pytest.raises(YawRuntimeError, match=field):
        replace(_make_cfg(tmp_path), **{field: value})


def test_retired_character_budget_fails_explicitly(tmp_path: Path) -> None:
    config = tmp_path / "yaw.toml"
    config.write_text("[graph]\nembed_batch_chars = 2048\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError, match="embed_batch_chars is retired"):
        load_config(config)


def test_embedding_settings_have_one_authoritative_section(tmp_path: Path) -> None:
    config = tmp_path / "yaw.toml"
    config.write_text("[ingestion]\nembed_batch_size = 3\n[inference]\nembed_max_in_flight = 1\n", encoding="utf-8")
    cfg = load_config(config)
    assert cfg.embed_batch_size == 3
    assert cfg.embed_max_in_flight == 1
    for key in ("embed_batch_size", "embed_max_in_flight"):
        config.write_text(f"[graph]\n{key} = 4\n", encoding="utf-8")
        with pytest.raises(YawRuntimeError, match="belongs in"):
            load_config(config)


@pytest.mark.parametrize("batch_size", [1, 3, 70])
def test_native_pipeline_batch_size_is_configured(tmp_path: Path, batch_size: int) -> None:
    requests: list[list[str]] = []

    def receive(_model: str, texts: list[str]) -> NDArray[np.float32]:
        requests.append(texts)
        return _hash_embed(texts, dim=8)

    with _loopback_embed_server(receive) as port:
        cfg = replace(
            _make_http_cfg(tmp_path, port=port, embedding_model="batch-model"),
            chunk_size=1000,
            embed_batch_size=batch_size,
        )
        assert cfg.corpus_root is not None and cfg.db_path is not None
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "long.txt").write_text("\n\n".join(f"Paragraph {i}: " + "routing " * 90 for i in range(80)))
        pieces = chunk_file_content(
            "long.txt",
            (cfg.corpus_root / "long.txt").read_text(),
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            min_chunk_size=cfg.min_chunk_size,
        )
        session = start_pipeline(cfg, embed_fn=make_dspy_embed_fn(cfg), live=False)
        try:
            update_pipeline_once(session)
            with _open_ro(cfg.db_path) as conn:
                assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == len(pieces)
        finally:
            close_pipeline(session)
    assert sum(map(len, requests)) == len(pieces)
    assert max(map(len, requests)) <= batch_size
    assert all(len(text.encode("utf-8")) <= 1100 for batch in requests for text in batch)
    if batch_size > 1:
        assert max(map(len, requests)) > 1
    if batch_size > 64:
        assert max(map(len, requests)) > 64


def test_batch_size_changes_only_http_grouping_not_chunks(tmp_path: Path) -> None:
    requests: list[list[str]] = []

    def receive(_model: str, texts: list[str]) -> NDArray[np.float32]:
        requests.append(list(texts))
        return _hash_embed(texts, dim=8)

    payloads: list[Counter[str]] = []
    snapshots: list[list[Any]] = []
    with _loopback_embed_server(receive) as port:
        for batch_size in (1, 16):
            requests.clear()
            cfg = replace(
                _make_http_cfg(tmp_path / str(batch_size), port=port, embedding_model="grouping-model"),
                chunk_size=1000,
                embed_batch_size=batch_size,
            )
            assert cfg.corpus_root is not None and cfg.db_path is not None
            cfg.corpus_root.mkdir(parents=True)
            for title in ("Interviewer workload", "Proxy timer skew"):
                body = "\n\n".join(
                    f"Section {i}: " + "Distinct evidence with café, 漢字 and 🧠. " * 20 for i in range(24)
                )
                (cfg.corpus_root / f"{title}.md").write_text(f"# {title}\n\n{body}", encoding="utf-8")
            session = start_pipeline(cfg, embed_fn=make_dspy_embed_fn(cfg), live=False)
            try:
                update_pipeline_once(session)
                with _open_ro(cfg.db_path) as conn:
                    snapshots.append(conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall())
            finally:
                close_pipeline(session)
            payloads.append(Counter(text for batch in requests for text in batch))
            assert max(map(len, requests)) <= batch_size
            if batch_size == 16:
                assert max(map(len, requests)) > 1
    # Exact on-wire text (including title/prefix), IDs, spans, features and vectors.
    assert sum(payloads[0].values()) > 16
    assert payloads[0] == payloads[1]
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("max_in_flight", [1, 2])
def test_http_concurrency_is_shared_by_queries_and_ingestion(tmp_path: Path, max_in_flight: int) -> None:
    lock = threading.Lock()
    saturated = threading.Event()
    release = threading.Event()
    active = peak = 0

    def receive(_model: str, texts: list[str]) -> NDArray[np.float32]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active >= max_in_flight:
                saturated.set()
        try:
            assert release.wait(timeout=10)
            return _hash_embed(texts, dim=8)
        finally:
            with lock:
                active -= 1

    with _loopback_embed_server(receive) as port:
        cfg = replace(
            _make_http_cfg(tmp_path, port=port, embedding_model="concurrency-model"),
            embed_batch_size=2,
            embed_max_in_flight=max_in_flight,
        )
        assert cfg.corpus_root is not None
        cfg.corpus_root.mkdir(parents=True)
        for i in range(8):
            (cfg.corpus_root / f"note{i}.txt").write_text(f"Routing example {i}.\n", encoding="utf-8")
        embed = make_dspy_embed_fn(cfg)
        session = start_pipeline(cfg, embed_fn=embed, live=False)
        try:
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(update_pipeline_once, session)]
                futures.extend(pool.submit(embed, [format_query_for_embedding(f"Question {i}")]) for i in range(4))
                try:
                    assert saturated.wait(timeout=10)
                finally:
                    release.set()
                for future in futures:
                    future.result(timeout=20)
        finally:
            release.set()
            close_pipeline(session)
    assert peak == max_in_flight


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
            blob_a = conn.execute("SELECT embedding_f16 FROM chunks").fetchone()[0]
            assert np.allclose(np.frombuffer(blob_a, dtype="<f2").astype(np.float32), [1.0, 0.0, 0.0, 0.0], atol=1e-3)
            assert conn.execute("SELECT embedding_model FROM rabitq_index WHERE id = 1").fetchone()[0] == "model-a"

        count_before_b = len(seen_calls)
        cfg_b = _make_http_cfg(tmp_path, port=port, embedding_model="model-b")
        s2 = start_pipeline(cfg_b, embed_fn=make_dspy_embed_fn(cfg_b), live=False)
        try:
            update_pipeline_once(s2)
        finally:
            close_pipeline(s2)

        calls_b = [c for c in seen_calls[count_before_b:] if "model-b" in c[0] and any("waveguide" in t for t in c[1])]
        assert len(calls_b) == 1
        assert cfg_b.db_path is not None
        with _open_ro(cfg_b.db_path) as conn:
            blob_b = conn.execute("SELECT embedding_f16 FROM chunks").fetchone()[0]
            assert np.allclose(np.frombuffer(blob_b, dtype="<f2").astype(np.float32), [0.0, 1.0, 0.0, 0.0], atol=1e-3)
            assert conn.execute("SELECT embedding_model FROM rabitq_index WHERE id = 1").fetchone()[0] == "model-b"


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
            rows_first = conn.execute(
                "SELECT chunk_id, rel_path, embedding_f16 FROM chunks ORDER BY chunk_id"
            ).fetchall()
            fts_first = conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'memo' ORDER BY rowid"
            ).fetchall()
            assert rows_first and fts_first

        # Fresh start_pipeline + make_dspy_embed_fn on existing 5-table schema
        s2 = start_pipeline(cfg, embed_fn=make_dspy_embed_fn(cfg), live=False)
        try:
            update_pipeline_once(s2)
        finally:
            close_pipeline(s2)

        assert len(http_calls) == calls_after_first
        with _open_ro(cfg.db_path) as conn:
            rows_second = conn.execute(
                "SELECT chunk_id, rel_path, embedding_f16 FROM chunks ORDER BY chunk_id"
            ).fetchall()
            fts_second = conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'memo' ORDER BY rowid"
            ).fetchall()
            assert rows_second == rows_first
            assert fts_second == fts_first


def test_live_pipeline_shutdown_joins_inflight_reconcile_before_closing_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path / "join_reconcile")
    assert cfg.corpus_root is not None and cfg.db_path is not None
    corpus_root = cfg.corpus_root
    corpus_root.mkdir(parents=True)
    (corpus_root / "seed.md").write_text("# Seed\nInitial quartz oscillator specification.\n", encoding="utf-8")

    import yaw.pipeline.session as pipeline_session

    real_reconcile = pipeline_session.reconcile_index_tables
    reconcile_entered = threading.Event()
    release_reconcile = threading.Event()
    reconcile_finished = threading.Event()
    reconcile_errors: list[BaseException] = []
    call_count = 0

    def _blocking_reconcile(
        conn_mgr: Any,
        *,
        embedding_model: str,
        task_mode: str,
        corpus_root: Path | None = None,
    ) -> bool:
        nonlocal call_count
        call_count += 1
        current = call_count
        if current == 1:
            reconcile_entered.set()
            assert release_reconcile.wait(timeout=10.0)
        try:
            return real_reconcile(
                conn_mgr,
                embedding_model=embedding_model,
                task_mode=task_mode,
                corpus_root=corpus_root,
            )
        except BaseException as exc:
            reconcile_errors.append(exc)
            raise
        finally:
            if current == 1:
                reconcile_finished.set()

    monkeypatch.setattr(pipeline_session, "reconcile_index_tables", _blocking_reconcile)

    async def _run() -> None:
        cm = open_pipeline_lifespan(cfg, embed_fn=lambda ts: _hash_embed(ts, dim=8), live=True)
        _sess = await cm.__aenter__()
        assert await asyncio.to_thread(reconcile_entered.wait, 15.0)

        exit_task = asyncio.create_task(cm.__aexit__(None, None, None))
        await asyncio.sleep(0.2)
        assert not exit_task.done(), "open_pipeline_lifespan closed env/conn_mgr before worker reconcile finished"
        assert not reconcile_finished.is_set()

        release_reconcile.set()
        await asyncio.wait_for(exit_task, timeout=10.0)
        assert reconcile_finished.is_set()

    try:
        asyncio.run(_run())
    finally:
        release_reconcile.set()
    assert reconcile_errors == []
    with _open_ro(cfg.db_path) as conn:
        assert conn.execute("SELECT n_chunks FROM rabitq_index WHERE id = 1").fetchone()[0] == 1
