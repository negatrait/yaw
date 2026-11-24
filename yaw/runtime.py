"""Plain zero-wrapper YawRuntime lifecycle, async lifespan orchestration, and query entrypoints."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
import filelock
import numpy as np
from numpy.typing import NDArray

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
    DspyBatchEmbedder,
    PipelineSession,
    close_pipeline,
    make_dspy_embed_fn,
    open_pipeline_lifespan,
    start_pipeline,
    update_pipeline_once,
)
from yaw.qa import ask_question, build_qa_agent, create_lm, make_qa_tools
from yaw.retrieval import (
    RetrievalIndex,
    SearchResult,
    build_retrieval_index,
    empty_retrieval_index,
    hydrate_ranked_passages,
    open_readonly_db,
    search_index,
)
from yaw.retrieval import (
    build_recall_payload as build_recall_payload,
)
from yaw.retrieval import (
    dump_recall_json as dump_recall_json,
)
from yaw.retrieval import (
    format_recall_text as format_recall_text,
)


@dataclass(slots=True)
class YawRuntime:
    """Mutable runtime state owning lock, CocoIndex pipeline, semaphores, and RetrievalIndex."""

    config: YawConfig
    pipeline: PipelineSession
    lm: dspy.LM
    embed_fn: EmbedFn
    lm_sem: threading.Semaphore
    query_sem: threading.BoundedSemaphore
    index_lock: threading.RLock
    retrieval_index: RetrievalIndex
    lifecycle: str
    last_error: str | None
    last_sync_at: float | None
    file_lock: filelock.FileLock
    stop_event: threading.Event
    owns_file_lock: bool = True
    index_generation: int = 0
    ready_evt: asyncio.Event | None = None
    async_loop: asyncio.AbstractEventLoop | None = None
    async_query_sem: asyncio.Semaphore | None = None
    async_embed_sem: asyncio.Semaphore | None = None
    async_lm_sem: asyncio.Semaphore | None = None
    embed_uses_async: bool = False
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    active_queries: int = 0
    active_op: str | None = None


def require_path(path: Path | None, label: str) -> Path:
    """Resolve a required configuration path or raise INVALID_INPUT when omitted."""
    if path is None:
        raise YawRuntimeError("INVALID_INPUT", f"require_path(): YawConfig.{label} must not be None.")
    return Path(path).resolve()


def control_token_path(cfg: YawConfig) -> Path:
    """Return the path of the daemon HTTP control token beside yaw.db.lock."""
    return require_path(cfg.lock_path, "lock_path").parent / "control.token"


def _acquire_file_lock(cfg: YawConfig) -> filelock.FileLock:
    lock_path = require_path(cfg.lock_path, "lock_path")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout as exc:
        raise YawRuntimeError("BUSY", f"Lock {lock_path} is already held by another YAW instance.") from exc
    return lock


@contextlib.contextmanager
def acquire_daemon_control(cfg: YawConfig) -> Generator[tuple[filelock.FileLock, str]]:
    """Acquire exclusive yaw.db.lock first, then write/cleanup 0o600 .yaw/control.token under lock."""
    file_lock = _acquire_file_lock(cfg)
    tok_path = control_token_path(cfg)
    token = secrets.token_hex(32)
    tmp_path = tok_path.with_name(f".{tok_path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, token.encode("ascii"))
        finally:
            os.close(fd)
        tmp_path.replace(tok_path)
        yield file_lock, token
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            tok_path.unlink(missing_ok=True)
        if file_lock.is_locked:
            file_lock.release()


def sync_embed_concurrency_limit(embed_max_in_flight: int) -> int:
    """Clamp sync embed_fn thread offload concurrency to leave >= 4 default-executor workers free."""
    cpu_count = os.process_cpu_count() or 1
    default_pool_cap = min(32, cpu_count + 4)
    return max(1, min(int(embed_max_in_flight), default_pool_cap - 4))


def get_retrieval_index(rt: YawRuntime) -> RetrievalIndex:
    """Return the active immutable RetrievalIndex snapshot under short RLock."""
    with rt.index_lock:
        return rt.retrieval_index


def reload_retrieval_index(rt: YawRuntime, conn: sqlite3.Connection | None = None) -> RetrievalIndex:
    """Rebuild RetrievalIndex from ?mode=ro WAL outside index_lock and atomically swap it in."""
    db_path = require_path(rt.config.db_path, "db_path")
    with contextlib.nullcontext(conn) if conn is not None else contextlib.closing(open_readonly_db(db_path)) as ro_conn:
        try:
            new_idx = build_retrieval_index(
                ro_conn,
                expected_embedding_model=rt.config.embedding_model,
                expected_task_mode=rt.config.embedding_task_mode,
            )
        except YawRuntimeError as exc:
            if exc.code == "INDEX_NOT_READY":
                with rt.index_lock:
                    if rt.pipeline.live and rt.index_generation > 0:
                        return rt.retrieval_index
            raise

    with rt.index_lock:
        rt.retrieval_index = new_idx
        rt.index_generation += 1
        current_idx = rt.retrieval_index
    now = time.time()
    with rt.state_lock:
        rt.last_sync_at = now
    return current_idx


def _build_runtime_instance(
    cfg: YawConfig,
    pipeline: PipelineSession,
    lm: dspy.LM,
    embed_fn: EmbedFn,
    file_lock: filelock.FileLock,
    *,
    owns_file_lock: bool,
    ready_evt: asyncio.Event | None = None,
    async_loop: asyncio.AbstractEventLoop | None = None,
) -> YawRuntime:
    uses_async = isinstance(embed_fn, DspyBatchEmbedder)
    embed_cap = cfg.embed_max_in_flight if uses_async else sync_embed_concurrency_limit(cfg.embed_max_in_flight)
    return YawRuntime(
        config=cfg,
        pipeline=pipeline,
        lm=lm,
        embed_fn=embed_fn,
        lm_sem=threading.Semaphore(1),
        query_sem=threading.BoundedSemaphore(cfg.queue_maxsize),
        index_lock=threading.RLock(),
        retrieval_index=empty_retrieval_index(),
        lifecycle="starting",
        last_error=None,
        last_sync_at=None,
        file_lock=file_lock,
        stop_event=threading.Event(),
        owns_file_lock=owns_file_lock,
        ready_evt=ready_evt,
        async_loop=async_loop,
        async_query_sem=asyncio.Semaphore(cfg.queue_maxsize) if async_loop is not None else None,
        async_embed_sem=asyncio.Semaphore(embed_cap) if async_loop is not None else None,
        async_lm_sem=asyncio.Semaphore(1) if async_loop is not None else None,
        embed_uses_async=uses_async,
    )


def _record_runtime_error(rt: YawRuntime, exc: BaseException) -> None:
    msg = redact_secrets(f"{type(exc).__name__}: {exc}", rt.config.api_key)
    with rt.state_lock:
        rt.last_error = msg
        if rt.lifecycle not in ("stopping", "stopped"):
            rt.lifecycle = "error"


def _mark_ready(rt: YawRuntime) -> None:
    now = time.time()
    with rt.state_lock:
        if rt.lifecycle in ("stopping", "stopped") or rt.stop_event.is_set():
            return
        if rt.lifecycle == "starting":
            rt.lifecycle = "ready"
        rt.last_sync_at = now
        rt.last_error = None


@contextlib.asynccontextmanager
async def runtime_lifespan(
    cfg: YawConfig,
    *,
    live: bool = True,
    wait_for_ready: bool = False,
    embed_fn: EmbedFn | None = None,
    lm: dspy.LM | None = None,
    file_lock: filelock.FileLock | None = None,
    ready_evt: asyncio.Event | None = None,
) -> AsyncGenerator[YawRuntime]:
    """Run unified CocoIndex + YawRuntime lifecycle directly on the caller's asyncio event loop."""
    owns_file_lock = file_lock is None
    active_lock = file_lock if file_lock is not None else _acquire_file_lock(cfg)
    evt = ready_evt if ready_evt is not None else asyncio.Event()
    loop = asyncio.get_running_loop()
    rt_box: dict[str, YawRuntime] = {}
    active_embed: EmbedFn | None = None

    def _on_error(exc: Exception) -> None:
        if (active_rt := rt_box.get("rt")) is None or active_rt.stop_event.is_set():
            evt.set()
            return
        _record_runtime_error(active_rt, exc)
        get_status(active_rt)
        evt.set()

    async def _on_reconciled(changed: bool, is_initial: bool) -> None:
        if (active_rt := rt_box.get("rt")) is None or active_rt.stop_event.is_set():
            return
        if changed or is_initial:
            await asyncio.to_thread(reload_retrieval_index, active_rt)
        if is_initial:
            _mark_ready(active_rt)
        await asyncio.to_thread(get_status, active_rt)
        evt.set()

    try:
        if cfg.log_path is not None:
            configure_logging(cfg.log_path)
        active_embed = embed_fn or make_dspy_embed_fn(cfg)
        active_lm = lm or create_lm(cfg)
        async with open_pipeline_lifespan(
            cfg,
            embed_fn=active_embed,
            live=live,
            on_reconciled=_on_reconciled,
            on_error=_on_error,
        ) as pipeline:
            rt = _build_runtime_instance(
                cfg,
                pipeline,
                active_lm,
                active_embed,
                active_lock,
                owns_file_lock=owns_file_lock,
                ready_evt=evt,
                async_loop=loop,
            )
            rt_box["rt"] = rt
            if not live and rt.lifecycle == "starting":
                await asyncio.to_thread(reload_retrieval_index, rt)
                _mark_ready(rt)
                await asyncio.to_thread(get_status, rt)
                evt.set()
            else:
                await asyncio.to_thread(get_status, rt)
                if wait_for_ready and not await wait_ready_async(rt, timeout=30.0):
                    raise YawRuntimeError("NOT_READY", "Timed out waiting for initial indexing readiness.")
            try:
                yield rt
            finally:
                rt.stop_event.set()
                with rt.state_lock:
                    if rt.lifecycle != "stopped":
                        rt.lifecycle = "stopping"
                evt.set()
    finally:
        try:
            if isinstance(active_embed, DspyBatchEmbedder):
                await active_embed.aclose()
        finally:
            try:
                if (active_rt := rt_box.get("rt")) is not None:
                    with active_rt.state_lock:
                        active_rt.lifecycle = "stopped"
                        active_rt.active_op = None
                    await asyncio.to_thread(get_status, active_rt)
                evt.set()
            finally:
                if owns_file_lock and active_lock.is_locked:
                    active_lock.release()


def start_runtime(
    cfg: YawConfig,
    *,
    live: bool = False,
    wait_for_ready: bool = True,
    embed_fn: EmbedFn | None = None,
    lm: dspy.LM | None = None,
    file_lock: filelock.FileLock | None = None,
) -> YawRuntime:
    """Acquire yaw.db.lock (when not pre-acquired) and start catch-up YawRuntime (live=False only)."""
    if live:
        raise YawRuntimeError(
            "INVALID_INPUT",
            "start_runtime() supports live=False only; use runtime_lifespan() for live mode.",
        )
    owns_file_lock = file_lock is None
    active_lock = file_lock if file_lock is not None else _acquire_file_lock(cfg)
    pipeline: PipelineSession | None = None
    rt: YawRuntime | None = None
    try:
        if cfg.log_path is not None:
            configure_logging(cfg.log_path)
        active_embed = embed_fn or make_dspy_embed_fn(cfg)
        active_lm = lm or create_lm(cfg)
        pipeline = start_pipeline(cfg, embed_fn=active_embed, live=False)
        rt = _build_runtime_instance(
            cfg,
            pipeline,
            active_lm,
            active_embed,
            active_lock,
            owns_file_lock=owns_file_lock,
        )
        get_status(rt)
        if wait_for_ready and not wait_ready(rt, timeout=30.0):
            raise YawRuntimeError("NOT_READY", "Timed out waiting for initial indexing readiness.")
        return rt
    except BaseException:
        if rt is not None:
            close_runtime(rt)
        else:
            if pipeline is not None:
                close_pipeline(pipeline)
            if owns_file_lock and active_lock.is_locked:
                active_lock.release()
        raise


def wait_ready(rt: YawRuntime, timeout: float = 30.0) -> bool:
    """Execute initial catch-up indexing synchronously, load RetrievalIndex, and mark runtime ready."""
    _ = timeout
    if rt.pipeline.live:
        raise YawRuntimeError("INVALID_INPUT", "wait_ready() supports catch-up mode only; use wait_ready_async().")
    try:
        if rt.lifecycle == "starting":
            update_pipeline_once(rt.pipeline)
    except BaseException as exc:
        if not rt.stop_event.is_set():
            _record_runtime_error(rt, exc)
            get_status(rt)
        raise
    if rt.stop_event.is_set():
        return False
    reload_retrieval_index(rt)
    _mark_ready(rt)
    get_status(rt)
    return rt.lifecycle == "ready"


async def wait_ready_async(rt: YawRuntime, timeout: float = 30.0) -> bool:
    """Await initial indexing readiness on the running asyncio loop."""
    if not rt.pipeline.live:
        if rt.lifecycle == "starting":
            await asyncio.to_thread(wait_ready, rt, timeout)
        return rt.lifecycle == "ready"
    evt = rt.ready_evt or rt.pipeline.ready_event
    if evt is not None:
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
        except TimeoutError:
            return False
    lc, _ = _sync_pipeline_error(rt)
    if rt.pipeline.error is not None:
        raise rt.pipeline.error
    return lc == "ready" and not rt.stop_event.is_set()


def _validate_text_input(raw: str, label: str) -> str:
    if not isinstance(raw, str):
        raise YawRuntimeError("INVALID_INPUT", f"_validate_text_input(): {label} must be a string.")
    cleaned = raw.strip()
    if not cleaned:
        raise YawRuntimeError("INVALID_INPUT", f"_validate_text_input(): {label} must not be empty.")
    return cleaned


def _sync_pipeline_error(rt: YawRuntime) -> tuple[str, str | None]:
    pipe_err = rt.pipeline.error
    err_msg = (
        redact_secrets(f"{type(pipe_err).__name__}: {pipe_err}", rt.config.api_key) if pipe_err is not None else None
    )
    with rt.state_lock:
        if err_msg is not None:
            if rt.last_error is None or not bool(rt.last_error) or rt.lifecycle not in ("error", "stopping", "stopped"):
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


def _ensure_async_primitives(rt: YawRuntime) -> tuple[asyncio.Semaphore, asyncio.Semaphore, asyncio.Semaphore]:
    loop = asyncio.get_running_loop()
    if rt.async_loop is not loop or rt.async_query_sem is None or rt.async_embed_sem is None or rt.async_lm_sem is None:
        embed_cap = (
            rt.config.embed_max_in_flight
            if rt.embed_uses_async
            else sync_embed_concurrency_limit(rt.config.embed_max_in_flight)
        )
        rt.async_loop = loop
        rt.async_query_sem = asyncio.Semaphore(rt.config.queue_maxsize)
        rt.async_embed_sem = asyncio.Semaphore(embed_cap)
        rt.async_lm_sem = asyncio.Semaphore(1)
    return rt.async_query_sem, rt.async_embed_sem, rt.async_lm_sem


@contextlib.asynccontextmanager
async def _admit_query_async(rt: YawRuntime, op_name: str) -> AsyncGenerator[None]:
    q_sem, _, _ = _ensure_async_primitives(rt)
    with rt.state_lock:
        if rt.active_queries >= rt.config.queue_maxsize or q_sem.locked():
            raise YawRuntimeError("BUSY", f"Runtime query capacity ({rt.config.queue_maxsize}) saturated.")
        rt.active_queries += 1
        rt.active_op = op_name
    try:
        await q_sem.acquire()
    except BaseException:
        with rt.state_lock:
            rt.active_queries = max(0, rt.active_queries - 1)
            if rt.active_queries == 0:
                rt.active_op = None
        raise
    try:
        yield
    finally:
        q_sem.release()
        with rt.state_lock:
            rt.active_queries = max(0, rt.active_queries - 1)
            if rt.active_queries == 0:
                rt.active_op = None


@contextlib.contextmanager
def _admit_query(rt: YawRuntime, op_name: str) -> Generator[None]:
    if not rt.query_sem.acquire(blocking=False):
        raise YawRuntimeError("BUSY", f"Runtime query capacity ({rt.config.queue_maxsize}) saturated.")
    with rt.state_lock:
        if rt.active_queries >= rt.config.queue_maxsize:
            rt.query_sem.release()
            raise YawRuntimeError("BUSY", f"Runtime query capacity ({rt.config.queue_maxsize}) saturated.")
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
def _open_indexed_search(cfg: YawConfig) -> Generator[tuple[sqlite3.Connection, RetrievalIndex, Path]]:
    db_path = require_path(cfg.db_path, "db_path")
    if not db_path.is_file():
        raise YawRuntimeError("EMPTY_INDEX", f"Database not found: {db_path}")
    corpus_root = require_path(cfg.corpus_root, "corpus_root")
    with contextlib.closing(open_readonly_db(db_path)) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            raise YawRuntimeError("EMPTY_INDEX", f"Index table 'chunks' not initialized in {db_path}")
        index = build_retrieval_index(
            conn,
            expected_embedding_model=cfg.embedding_model,
            expected_task_mode=cfg.embedding_task_mode,
        )
        yield conn, index, corpus_root


def _search_and_hydrate_with_result(
    db_path: Path,
    index: RetrievalIndex,
    corpus_root: Path,
    q_vec: NDArray[np.float32],
    query: str,
    top_k: int,
) -> tuple[SearchResult, tuple[HydratedPassage, ...]]:
    with contextlib.closing(open_readonly_db(db_path)) as conn:
        res = search_index(index, q_vec, query_text=query, conn=conn, top_k=top_k, pool_k=max(top_k * 4, 50))
    return res, hydrate_ranked_passages(index, res.ranked, corpus_root, top_k=top_k)


def _run_recall_with_result(
    cfg: YawConfig,
    conn: sqlite3.Connection,
    index: RetrievalIndex,
    corpus_root: Path,
    embed_fn: EmbedFn | None,
    query: str,
    top_k: int,
) -> tuple[SearchResult | None, tuple[HydratedPassage, ...]]:
    if index.size == 0:
        return None, ()
    active_embed = embed_fn or make_dspy_embed_fn(cfg)
    formatted = format_query_for_embedding(query, cfg.embedding_task_mode)
    raw_emb = np.asarray(active_embed([formatted]), dtype=np.float32)
    q_vec = raw_emb[0] if raw_emb.ndim == 2 else raw_emb.reshape(-1)
    res = search_index(index, q_vec, query_text=query, conn=conn, top_k=top_k, pool_k=max(top_k * 4, 50))
    return res, hydrate_ranked_passages(index, res.ranked, corpus_root, top_k=top_k)


async def _arecall_core(
    rt: YawRuntime,
    query: str,
    *,
    top_k: int = 5,
) -> tuple[RetrievalIndex, SearchResult | None, tuple[HydratedPassage, ...]]:
    q = _validate_recall_args(query, top_k)
    _ensure_ready(rt)
    db_path = require_path(rt.config.db_path, "db_path")
    corpus_root = require_path(rt.config.corpus_root, "corpus_root")
    async with _admit_query_async(rt, "recall"):
        index = get_retrieval_index(rt)
        if index.size == 0:
            return index, None, ()
        _, embed_sem, _ = _ensure_async_primitives(rt)
        formatted = format_query_for_embedding(q, rt.config.embedding_task_mode)
        async with embed_sem:
            if isinstance(rt.embed_fn, DspyBatchEmbedder):
                raw_emb = await rt.embed_fn.acall([formatted])
            else:
                raw_emb = np.asarray(await asyncio.to_thread(rt.embed_fn, [formatted]), dtype=np.float32)
        q_vec = raw_emb[0] if raw_emb.ndim == 2 else raw_emb.reshape(-1)
        res, passages = await asyncio.to_thread(
            _search_and_hydrate_with_result, db_path, index, corpus_root, q_vec, q, top_k
        )
        return index, res, passages


async def arecall(rt: YawRuntime, query: str, *, top_k: int = 5) -> tuple[HydratedPassage, ...]:
    """Execute zero-LLM hybrid RetrievalIndex search asynchronously using DspyBatchEmbedder.acall or clamped offload."""
    _, _, passages = await _arecall_core(rt, query, top_k=top_k)
    return passages


async def arecall_payload(rt: YawRuntime, query: str, *, top_k: int = 5) -> dict[str, Any]:
    """Execute asynchronous zero-LLM recall and assemble the canonical sparse slip payload."""
    index, res, passages = await _arecall_core(rt, query, top_k=top_k)
    return build_recall_payload(passages, index=index, search_result=res)


def recall(rt: YawRuntime, query: str, *, top_k: int = 5) -> tuple[HydratedPassage, ...]:
    """Execute zero-LLM hybrid RetrievalIndex search synchronously with MaxP disk hydration."""
    q = _validate_recall_args(query, top_k)
    _ensure_ready(rt)
    db_path = require_path(rt.config.db_path, "db_path")
    corpus_root = require_path(rt.config.corpus_root, "corpus_root")
    with _admit_query(rt, "recall"), contextlib.closing(open_readonly_db(db_path)) as conn:
        index = get_retrieval_index(rt)
        _, passages = _run_recall_with_result(rt.config, conn, index, corpus_root, rt.embed_fn, q, top_k)
        return passages


def recall_once(cfg: YawConfig, query: str, *, top_k: int = 5) -> tuple[HydratedPassage, ...]:
    """Execute standalone read-only zero-LLM recall without acquiring the daemon lock."""
    q = _validate_recall_args(query, top_k)
    with _open_indexed_search(cfg) as (conn, index, corpus_root):
        _, passages = _run_recall_with_result(cfg, conn, index, corpus_root, None, q, top_k)
        return passages


def recall_once_payload(cfg: YawConfig, query: str, *, top_k: int = 5) -> dict[str, Any]:
    """Execute standalone read-only zero-LLM recall and return the canonical sparse slip payload."""
    q = _validate_recall_args(query, top_k)
    with _open_indexed_search(cfg) as (conn, index, corpus_root):
        res, passages = _run_recall_with_result(cfg, conn, index, corpus_root, None, q, top_k)
        return build_recall_payload(passages, index=index, search_result=res)


def resolve_program_path(cfg: YawConfig) -> Path | None:
    """Resolve the configured compiled program or the optimizer's artifact."""
    if cfg.compiled_program_path is not None and Path(cfg.compiled_program_path).is_file():
        return Path(cfg.compiled_program_path)
    fallback = require_path(cfg.for_optimization().compiled_program_path, "optimization compiled_program_path")
    return fallback if fallback.is_file() else None


def _run_qa(
    cfg: YawConfig,
    conn: sqlite3.Connection,
    index: RetrievalIndex,
    corpus_root: Path,
    embed_fn: EmbedFn,
    lm: dspy.LM,
    question: str,
    *,
    lm_lock: AbstractContextManager[Any] | None = None,
) -> QAResult:
    if index.size == 0:
        raise YawRuntimeError("EMPTY_INDEX", "Cannot answer question on an empty corpus index.")
    tools = make_qa_tools(index, corpus_root, embed_fn, conn, task_mode=cfg.embedding_task_mode)
    agent = build_qa_agent(tools, max_iters=cfg.react_max_iters, artifact_path=resolve_program_path(cfg))
    lock_ctx = lm_lock if lm_lock is not None else contextlib.nullcontext()
    with lock_ctx, dspy.context(lm=lm):
        return ask_question(question, corpus_root=corpus_root, index=index, agent=agent)


def _ask_with_index(rt: YawRuntime, db_path: Path, corpus_root: Path, index: RetrievalIndex, q: str) -> QAResult:
    with contextlib.closing(open_readonly_db(db_path)) as conn:
        try:
            result = _run_qa(rt.config, conn, index, corpus_root, rt.embed_fn, rt.lm, q, lm_lock=rt.lm_sem)
        except YawRuntimeError as exc:
            if exc.code in ("MODEL_ERROR", "CONSISTENCY_ERROR"):
                with rt.state_lock:
                    rt.last_error = redact_secrets(str(exc), rt.config.api_key)
            raise
    with rt.state_lock:
        rt.last_error = None
    return result


async def aask(rt: YawRuntime, question: str) -> QAResult:
    """Execute bounded bare dspy.ReAct QA asynchronously serialized under async_lm_sem."""
    q = _validate_text_input(question, "Question")
    _ensure_ready(rt)
    db_path = require_path(rt.config.db_path, "db_path")
    corpus_root = require_path(rt.config.corpus_root, "corpus_root")
    async with _admit_query_async(rt, "qa"):
        index = get_retrieval_index(rt)
        _, _, lm_sem = _ensure_async_primitives(rt)
        async with lm_sem:
            return await asyncio.to_thread(_ask_with_index, rt, db_path, corpus_root, index, q)


def ask(rt: YawRuntime, question: str) -> QAResult:
    """Execute bounded bare dspy.ReAct QA synchronously over indexed corpus passages."""
    q = _validate_text_input(question, "Question")
    _ensure_ready(rt)
    db_path = require_path(rt.config.db_path, "db_path")
    corpus_root = require_path(rt.config.corpus_root, "corpus_root")
    with _admit_query(rt, "qa"):
        index = get_retrieval_index(rt)
        return _ask_with_index(rt, db_path, corpus_root, index, q)


def ask_once(cfg: YawConfig, question: str) -> QAResult:
    """Execute standalone read-only ReAct QA without acquiring the daemon lock."""
    q = _validate_text_input(question, "Question")
    with _open_indexed_search(cfg) as (conn, index, corpus_root):
        return _run_qa(cfg, conn, index, corpus_root, make_dspy_embed_fn(cfg), create_lm(cfg), q)


def snapshot_status(rt: YawRuntime) -> DaemonStatus:
    """Return lock-free redacted DaemonStatus snapshot without disk I/O."""
    num_chunks = rt.retrieval_index.size
    lc, raw_err = _sync_pipeline_error(rt)
    with rt.state_lock:
        queue_depth = rt.active_queries
        active_op = rt.active_op
        last_sync_at = rt.last_sync_at
    redacted_err = redact_secrets(raw_err, rt.config.api_key) if raw_err is not None and bool(raw_err) else None
    sync_state = (
        "error" if redacted_err is not None and bool(redacted_err) else ("syncing" if lc == "starting" else "idle")
    )
    return DaemonStatus(
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


def get_status(rt: YawRuntime) -> DaemonStatus:
    """Return lock-free redacted DaemonStatus snapshot and persist status.json atomically."""
    status = snapshot_status(rt)
    if rt.config.status_path is not None:
        try:
            write_text_atomic(rt.config.status_path, json.dumps(status.to_dict(), indent=2))
        except OSError as exc:
            get_logger("yaw.runtime").error("Failed to persist status.json: %s", exc)
    return status


def close_runtime(rt: YawRuntime) -> None:
    """Stop catch-up runtime cleanly, close CocoIndex/SQLite handles, and release owned lock."""
    rt.stop_event.set()
    with rt.state_lock:
        if rt.lifecycle == "stopped":
            if rt.owns_file_lock and rt.file_lock.is_locked:
                rt.file_lock.release()
            return
        rt.lifecycle = "stopping"
    try:
        close_pipeline(rt.pipeline)
    finally:
        with rt.state_lock:
            rt.lifecycle = "stopped"
            rt.active_op = None
        get_status(rt)
        if rt.owns_file_lock and rt.file_lock.is_locked:
            rt.file_lock.release()
