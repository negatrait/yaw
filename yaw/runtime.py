"""Plain zero-wrapper YawRuntime lifecycle, live WAL data_version swaps, and query entrypoints."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
import filelock
import numpy as np

from yaw.core import (
    DaemonStatus,
    EmbedFn,
    HydratedPassage,
    QAResult,
    YawConfig,
    YawRuntimeError,
    configure_logging,
    format_query_for_embedding,
    get_logger,
    redact_secrets,
    write_text_atomic,
)
from yaw.pipeline import (
    PipelineSession,
    close_pipeline,
    make_dspy_embed_fn,
    start_pipeline,
    update_pipeline_once,
    wait_pipeline_ready,
)
from yaw.qa import ask_question, build_qa_agent, create_lm, make_qa_tools
from yaw.retrieval import (
    VectorSearch,
    build_vector_search,
    hydrate_ranked_passages,
    open_readonly_db,
    resolve_retrieval_params,
    search_vectors,
)

_kernel32: Any = None
if os.name == "nt":
    import ctypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    _kernel32.CreateEventW.restype = ctypes.c_void_p
    _kernel32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    _kernel32.OpenEventW.restype = ctypes.c_void_p
    _kernel32.SetEvent.argtypes = [ctypes.c_void_p]
    _kernel32.SetEvent.restype = ctypes.c_int
    _kernel32.ResetEvent.argtypes = [ctypes.c_void_p]
    _kernel32.ResetEvent.restype = ctypes.c_int
    _kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int


@dataclass(slots=True)
class YawRuntime:
    """Mutable runtime state owning lock, live CocoIndex pipeline, semaphores, and VectorSearch."""

    config: YawConfig
    pipeline: PipelineSession
    lm: dspy.LM
    embed_fn: EmbedFn
    lm_sem: threading.Semaphore
    query_sem: threading.BoundedSemaphore
    index_lock: threading.RLock
    vector_search: VectorSearch
    index_data_version: int
    lifecycle: str
    last_error: str | None
    last_sync_at: float | None
    file_lock: filelock.FileLock
    stop_event: threading.Event
    win32_stop_handle: int | None = field(default=None, repr=False)
    win32_waiter_thread: threading.Thread | None = field(default=None, repr=False)
    ready_thread: threading.Thread | None = field(default=None, repr=False)
    probe_conn: sqlite3.Connection | None = None
    probe_lock: threading.Lock = field(default_factory=threading.Lock)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    active_queries: int = 0
    active_op: str | None = None


def win32_stop_event_name(lock_path: Path | str) -> str:
    """Return the deterministic Local kernel stop-event name for lock_path."""
    resolved = str(Path(lock_path).resolve()).replace("\\", "/").lower()
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"Local\\yaw-stop-{digest}"


def signal_stop_event(lock_path: Path | str) -> bool:
    """Signal an active Win32 Local stop event for lock_path if open."""
    if os.name != "nt" or _kernel32 is None:
        return False
    handle = _kernel32.OpenEventW(0x0002, 0, win32_stop_event_name(lock_path))
    if not handle:
        return False
    try:
        return bool(_kernel32.SetEvent(handle))
    finally:
        _kernel32.CloseHandle(handle)


def _start_win32_stop_event(rt: YawRuntime) -> None:
    if os.name != "nt" or _kernel32 is None or rt.config.lock_path is None:
        return
    handle = _kernel32.CreateEventW(None, 1, 0, win32_stop_event_name(rt.config.lock_path))
    if not handle:
        return
    _kernel32.ResetEvent(handle)
    rt.win32_stop_handle = int(handle)

    def _wait_kernel_event(ev_handle: int) -> None:
        if _kernel32.WaitForSingleObject(ev_handle, 0xFFFFFFFF) == 0:
            rt.stop_event.set()

    thread = threading.Thread(
        target=_wait_kernel_event,
        args=(int(handle),),
        name="yaw-win32-stop-waiter",
        daemon=True,
    )
    rt.win32_waiter_thread = thread
    thread.start()


def _close_win32_stop_event(rt: YawRuntime) -> None:
    if os.name != "nt" or _kernel32 is None:
        return
    handle = rt.win32_stop_handle
    rt.win32_stop_handle = None
    if handle:
        with contextlib.suppress(OSError):
            _kernel32.SetEvent(handle)
    waiter = rt.win32_waiter_thread
    if waiter is not None and waiter.is_alive() and waiter is not threading.current_thread():
        waiter.join(timeout=1.0)
    rt.win32_waiter_thread = None
    if handle:
        with contextlib.suppress(OSError):
            _kernel32.CloseHandle(handle)


def require_path(path: Path | None, label: str) -> Path:
    """Resolve a required configuration path or raise INVALID_INPUT when omitted."""
    if path is None:
        raise YawRuntimeError("INVALID_INPUT", f"require_path(): YawConfig.{label} must not be None.")
    return Path(path).resolve()


def _read_data_version(rt: YawRuntime) -> int:
    with rt.probe_lock:
        if rt.probe_conn is None:
            raise YawRuntimeError("SHUTTING_DOWN", "Runtime database probe is closed.")
        row = rt.probe_conn.execute("PRAGMA data_version").fetchone()
        return int(row[0]) if row else 0


def get_vector_search(rt: YawRuntime, conn: sqlite3.Connection | None = None) -> VectorSearch:
    """Return current VectorSearch, rebuilding outside index_lock when PRAGMA data_version changes."""
    cur_ver = _read_data_version(rt)
    with rt.index_lock:
        if rt.index_data_version >= 0 and cur_ver == rt.index_data_version:
            return rt.vector_search

    db_path = require_path(rt.config.db_path, "db_path")
    with contextlib.nullcontext(conn) if conn is not None else contextlib.closing(open_readonly_db(db_path)) as ro_conn:
        if ro_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            return rt.vector_search
        new_vs = build_vector_search(ro_conn)

    updated = False
    with rt.index_lock:
        if cur_ver >= rt.index_data_version:
            rt.vector_search = new_vs
            rt.index_data_version = cur_ver
            updated = True
        current_vs = rt.vector_search
    if updated:
        now = time.time()
        with rt.state_lock:
            rt.last_sync_at = now
    return current_vs


def start_runtime(
    cfg: YawConfig,
    *,
    live: bool = True,
    wait_for_ready: bool = True,
    embed_fn: EmbedFn | None = None,
    lm: dspy.LM | None = None,
) -> YawRuntime:
    """Acquire exclusive yaw.db.lock, start CocoIndex pipeline, and optionally block until ready."""
    lock_path = require_path(cfg.lock_path, "lock_path")
    db_path = require_path(cfg.db_path, "db_path")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    file_lock = filelock.FileLock(str(lock_path), timeout=0)
    try:
        file_lock.acquire(timeout=0)
    except filelock.Timeout as exc:
        raise YawRuntimeError("BUSY", f"Lock {lock_path} is already held by another YAW instance.") from exc

    pipeline: PipelineSession | None = None
    probe_conn: sqlite3.Connection | None = None
    rt: YawRuntime | None = None
    try:
        if cfg.log_path is not None:
            configure_logging(cfg.log_path)
        active_embed = embed_fn or make_dspy_embed_fn(cfg)
        active_lm = lm or create_lm(cfg)
        pipeline = start_pipeline(cfg, embed_fn=active_embed, live=live)
        probe_conn = open_readonly_db(db_path)
        probe_conn.isolation_level = None
        probe_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()

        rt = YawRuntime(
            config=cfg,
            pipeline=pipeline,
            lm=active_lm,
            embed_fn=active_embed,
            lm_sem=threading.Semaphore(1),
            query_sem=threading.BoundedSemaphore(cfg.queue_maxsize),
            index_lock=threading.RLock(),
            vector_search=build_vector_search(probe_conn),
            index_data_version=-1,
            lifecycle="starting",
            last_error=None,
            last_sync_at=None,
            file_lock=file_lock,
            stop_event=threading.Event(),
            probe_conn=probe_conn,
        )
        _start_win32_stop_event(rt)
        get_status(rt)

        if wait_for_ready:
            if not wait_ready(rt, timeout=30.0):
                raise YawRuntimeError("NOT_READY", "Timed out waiting for initial indexing readiness.")
        elif live:

            def _await_initial_ready() -> None:
                if rt is None:
                    return
                rt.pipeline.ready_event.wait()
                if not rt.stop_event.is_set() and rt.lifecycle == "starting":
                    try:
                        wait_ready(rt, timeout=0.0)
                    except Exception as exc:
                        msg = redact_secrets(f"{type(exc).__name__}: {exc}", rt.config.api_key)
                        with rt.state_lock:
                            rt.last_error = msg
                            rt.lifecycle = "error"
                        rt.pipeline.ready_event.set()
                        get_status(rt)

            rt.ready_thread = threading.Thread(target=_await_initial_ready, name="yaw-ready-waiter", daemon=True)
            rt.ready_thread.start()
        return rt
    except BaseException:
        if rt is not None:
            close_runtime(rt)
        else:
            if probe_conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    probe_conn.close()
            if pipeline is not None:
                close_pipeline(pipeline)
            if file_lock.is_locked:
                file_lock.release()
        raise


def wait_ready(rt: YawRuntime, timeout: float = 30.0) -> bool:
    """Wait for initial indexing catch-up, build VectorSearch, and mark runtime ready."""
    try:
        if not rt.pipeline.live and not rt.pipeline.ready_event.is_set():
            update_pipeline_once(rt.pipeline)
            ready = True
        else:
            ready = wait_pipeline_ready(rt.pipeline, timeout=timeout)
    except BaseException as exc:
        if not rt.stop_event.is_set():
            msg = redact_secrets(f"{type(exc).__name__}: {exc}", rt.config.api_key)
            with rt.state_lock:
                rt.last_error = msg
            get_status(rt)
        raise

    if not ready or rt.stop_event.is_set():
        return False

    get_vector_search(rt)
    now = time.time()
    with rt.state_lock:
        if rt.lifecycle in ("stopping", "stopped") or rt.stop_event.is_set():
            return False
        if rt.lifecycle == "starting":
            rt.lifecycle = "ready"
        rt.last_sync_at = now
        rt.last_error = None
    get_status(rt)
    return True


def _validate_text_input(raw: str, label: str) -> str:
    if not isinstance(raw, str):
        raise YawRuntimeError("INVALID_INPUT", f"_validate_text_input(): {label} must be a string.")
    cleaned = raw.strip()
    if not cleaned:
        raise YawRuntimeError("INVALID_INPUT", f"_validate_text_input(): {label} must not be empty.")
    return cleaned


def _sync_pipeline_error(rt: YawRuntime) -> tuple[str, str | None]:
    pipe_err = rt.pipeline.error
    pipe_num_errors = getattr(rt.pipeline, "num_errors", 0)
    err_msg: str | None = None
    if pipe_err is not None:
        err_msg = f"{type(pipe_err).__name__}: {pipe_err}"
    elif pipe_num_errors and pipe_num_errors > 0:
        err_msg = f"Pipeline reported {pipe_num_errors} component error(s)."
    with rt.state_lock:
        if err_msg is not None:
            if not rt.last_error or pipe_err is not None:
                rt.last_error = err_msg
            if rt.lifecycle not in ("stopping", "stopped"):
                rt.lifecycle = "error"
        return rt.lifecycle, rt.last_error


def _ensure_ready(rt: YawRuntime) -> None:
    lc, _ = _sync_pipeline_error(rt)
    if rt.stop_event.is_set() or lc in ("stopping", "stopped"):
        raise YawRuntimeError("SHUTTING_DOWN", "Daemon runtime is stopping or stopped.")
    if lc == "error":
        get_status(rt)
    if lc != "ready":
        raise YawRuntimeError("NOT_READY", f"Daemon runtime is not ready (lifecycle={lc}).")


@contextlib.contextmanager
def _admit_query(rt: YawRuntime, op_name: str) -> Generator[None]:
    if not rt.query_sem.acquire(blocking=False):
        raise YawRuntimeError("BUSY", f"Runtime query capacity ({rt.config.queue_maxsize}) saturated.")
    with rt.state_lock:
        rt.active_queries += 1
        rt.active_op = op_name
    try:
        yield
    finally:
        with rt.state_lock:
            rt.active_queries = max(0, rt.active_queries - 1)
            if rt.active_queries == 0:
                rt.active_op = None
        rt.query_sem.release()


def _validate_recall_args(query: str, top_k: int) -> str:
    q = _validate_text_input(query, "Query")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not (1 <= top_k <= 20):
        raise YawRuntimeError("INVALID_INPUT", "recall(): top_k must be an integer between 1 and 20.")
    return q


def read_db_chunk_count(db_path: Path) -> int:
    """Return indexed chunk count from a read-only SQLite DB, or 0 if uninitialized."""
    if not db_path.is_file():
        return 0
    with contextlib.closing(open_readonly_db(db_path)) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            return 0
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0


@contextlib.contextmanager
def _open_indexed_search(cfg: YawConfig) -> Generator[tuple[sqlite3.Connection, VectorSearch, Path]]:
    db_path = require_path(cfg.db_path, "db_path")
    if not db_path.is_file():
        raise YawRuntimeError("EMPTY_INDEX", f"Database not found: {db_path}")
    corpus_root = require_path(cfg.corpus_root, "corpus_root")
    with contextlib.closing(open_readonly_db(db_path)) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            raise YawRuntimeError("EMPTY_INDEX", f"Index table 'chunks' not initialized in {db_path}")
        yield conn, build_vector_search(conn), corpus_root


def _run_recall(
    cfg: YawConfig,
    conn: sqlite3.Connection,
    vs: VectorSearch,
    corpus_root: Path,
    embed_fn: EmbedFn | None,
    query: str,
    top_k: int,
) -> tuple[HydratedPassage, ...]:
    params = resolve_retrieval_params(cfg)
    if vs.size == 0:
        return ()
    active_embed = embed_fn or make_dspy_embed_fn(cfg)
    formatted = format_query_for_embedding(query, cfg.embedding_task_mode)
    raw_emb = np.asarray(active_embed([formatted]), dtype=np.float32)
    q_vec = raw_emb[0] if raw_emb.ndim == 2 else raw_emb.reshape(-1)
    ranked = search_vectors(vs, q_vec, query_text=query, conn=conn, top_k=max(top_k * 4, 50), **params)
    return hydrate_ranked_passages(vs, ranked, corpus_root, top_k=top_k)


def recall(rt: YawRuntime, query: str, *, top_k: int = 5) -> tuple[HydratedPassage, ...]:
    """Execute zero-LLM hybrid VectorSearch retrieval with MaxP disk hydration."""
    q = _validate_recall_args(query, top_k)
    _ensure_ready(rt)

    db_path = require_path(rt.config.db_path, "db_path")
    corpus_root = require_path(rt.config.corpus_root, "corpus_root")
    with _admit_query(rt, "recall"), contextlib.closing(open_readonly_db(db_path)) as conn:
        vs = get_vector_search(rt, conn)
        return _run_recall(rt.config, conn, vs, corpus_root, rt.embed_fn, q, top_k)


def recall_once(cfg: YawConfig, query: str, *, top_k: int = 5) -> tuple[HydratedPassage, ...]:
    """Execute standalone read-only zero-LLM recall without acquiring the daemon lock."""
    q = _validate_recall_args(query, top_k)
    with _open_indexed_search(cfg) as (conn, vs, corpus_root):
        return _run_recall(cfg, conn, vs, corpus_root, None, q, top_k)


def resolve_program_path(cfg: YawConfig) -> Path | None:
    """Resolve the configured compiled program or the optimizer's artifact."""
    if cfg.compiled_program_path is not None and Path(cfg.compiled_program_path).is_file():
        return Path(cfg.compiled_program_path)
    fallback = require_path(cfg.for_optimization().compiled_program_path, "optimization compiled_program_path")
    return fallback if fallback.is_file() else None


def serialize_passage(rank: int, hp: HydratedPassage) -> dict[str, Any]:
    """Serialize a hydrated passage into the canonical JSON payload shared by MCP and CLI."""
    ptr = hp.pointer
    digest = hashlib.sha256(hp.text.encode("utf-8")).hexdigest()
    return {
        "rank": rank,
        "chunk_id": int(ptr.chunk_id),
        "text": hp.text,
        "score": float(hp.score),
        "stale_on_disk": bool(hp.stale_on_disk),
        "content_hash": f"sha256:{digest}",
        "source": {
            "rel_posix_path": ptr.rel_posix_path,
            "start_line": int(ptr.start_line),
            "end_line": int(ptr.end_line),
            "title": ptr.title,
        },
    }


def _run_qa(
    cfg: YawConfig,
    conn: sqlite3.Connection,
    vs: VectorSearch,
    corpus_root: Path,
    embed_fn: EmbedFn,
    lm: dspy.LM,
    question: str,
    *,
    lm_lock: AbstractContextManager[Any] | None = None,
) -> QAResult:
    if vs.size == 0:
        raise YawRuntimeError("EMPTY_INDEX", "Cannot answer question on an empty corpus index.")
    search_params = resolve_retrieval_params(cfg)
    tools = make_qa_tools(
        vs, corpus_root, embed_fn, conn, task_mode=cfg.embedding_task_mode, search_params=search_params
    )
    agent = build_qa_agent(tools, max_iters=cfg.react_max_iters, artifact_path=resolve_program_path(cfg))
    lock_ctx = lm_lock if lm_lock is not None else contextlib.nullcontext()
    with lock_ctx, dspy.context(lm=lm):
        return ask_question(question, corpus_root=corpus_root, vs=vs, agent=agent)


def ask(rt: YawRuntime, question: str) -> QAResult:
    """Execute bounded bare dspy.ReAct QA over indexed corpus passages with citation verification."""
    q = _validate_text_input(question, "Question")
    _ensure_ready(rt)

    db_path = require_path(rt.config.db_path, "db_path")
    corpus_root = require_path(rt.config.corpus_root, "corpus_root")
    with _admit_query(rt, "qa"), contextlib.closing(open_readonly_db(db_path)) as conn:
        vs = get_vector_search(rt, conn)
        try:
            result = _run_qa(rt.config, conn, vs, corpus_root, rt.embed_fn, rt.lm, q, lm_lock=rt.lm_sem)
        except YawRuntimeError as exc:
            if exc.code in ("MODEL_ERROR", "CONSISTENCY_ERROR"):
                with rt.state_lock:
                    rt.last_error = redact_secrets(str(exc), rt.config.api_key)
            raise
        with rt.state_lock:
            rt.last_error = None
        return result


def ask_once(cfg: YawConfig, question: str) -> QAResult:
    """Execute standalone read-only ReAct QA without acquiring the daemon lock."""
    q = _validate_text_input(question, "Question")
    with _open_indexed_search(cfg) as (conn, vs, corpus_root):
        return _run_qa(cfg, conn, vs, corpus_root, make_dspy_embed_fn(cfg), create_lm(cfg), q)


def get_status(rt: YawRuntime) -> DaemonStatus:
    """Return lock-free redacted DaemonStatus snapshot and persist status.json atomically."""
    num_chunks = rt.vector_search.size
    lc, raw_err = _sync_pipeline_error(rt)
    with rt.state_lock:
        queue_depth = rt.active_queries
        active_op = rt.active_op
        last_sync_at = rt.last_sync_at
    redacted_err = redact_secrets(raw_err, rt.config.api_key) if raw_err else None
    sync_state = "error" if redacted_err else ("syncing" if lc == "starting" else "idle")
    status = DaemonStatus(
        lifecycle=lc,
        provider_ready=lc in ("starting", "ready") and not rt.stop_event.is_set(),
        sync_state=sync_state,
        queue_depth=queue_depth,
        active_op=active_op,
        last_error=redacted_err,
        last_sync_at=last_sync_at,
        pid=os.getpid() if lc != "stopped" else None,
        num_chunks=num_chunks,
    )
    if rt.config.status_path is not None:
        try:
            write_text_atomic(rt.config.status_path, json.dumps(status.to_dict(), indent=2))
        except OSError as exc:
            get_logger("yaw.runtime").error("Failed to persist status.json: %s", exc)
    return status


def close_runtime(rt: YawRuntime) -> None:
    """Stop runtime cleanly, close CocoIndex/SQLite probe handles, and release yaw.db.lock."""
    rt.stop_event.set()
    with rt.state_lock:
        if rt.lifecycle == "stopped":
            _close_win32_stop_event(rt)
            if rt.file_lock.is_locked:
                rt.file_lock.release()
            return
        rt.lifecycle = "stopping"
    rt.pipeline.ready_event.set()
    if rt.ready_thread is not None and rt.ready_thread.is_alive() and rt.ready_thread is not threading.current_thread():
        rt.ready_thread.join(timeout=2.0)
    rt.ready_thread = None
    _close_win32_stop_event(rt)
    try:
        with rt.probe_lock:
            if rt.probe_conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    rt.probe_conn.close()
                rt.probe_conn = None
        close_pipeline(rt.pipeline)
    finally:
        with rt.state_lock:
            rt.lifecycle = "stopped"
            rt.active_op = None
        get_status(rt)
        if rt.file_lock.is_locked:
            rt.file_lock.release()
