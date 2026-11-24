"""Pipeline session lifecycle, worker loop orchestration, and fail-fast error mapping."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import sqlite3
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cocoindex as coco
import numpy as np
from cocoindex.connectors import sqlite
from dspy.utils.exceptions import DSPyError
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError, resolve_coco_state_root

from .embedder import EMBEDDER, DspyBatchEmbedder, PipelineEmbedder, make_dspy_embed_fn
from .flow import CORPUS_ROOT, SQLITE_DB, indexing_main
from .targets import open_managed_db, reconcile_index_tables

WORKER_ERROR_TYPES: tuple[type[Exception], ...] = (
    YawRuntimeError,
    sqlite3.Error,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    LookupError,
    ArithmeticError,
    AttributeError,
    TimeoutError,
    DSPyError,
)


def _normalize_worker_error(exc: BaseException) -> Exception:
    if isinstance(exc, WORKER_ERROR_TYPES):
        return exc
    detail = (
        f"{type(exc).__name__}: {exc} ({'; '.join(f'{type(sub).__name__}: {sub}' for sub in exc.exceptions)})"
        if isinstance(exc, BaseExceptionGroup)
        else f"{type(exc).__name__}: {exc}"
    )
    err = RuntimeError(detail)
    err.__cause__ = exc
    return err


@dataclass(slots=True)
class PipelineSession:
    """Handle bundle owning a CocoIndex Environment, worker thread/loop, and SQLite connection."""

    db_path: Path
    state_dir: Path
    corpus_root: Path
    embedding_model: str
    task_mode: str
    env: coco.Environment
    app: coco.App[Any, Any]
    conn_mgr: sqlite.ManagedConnection
    live: bool
    ready_event: threading.Event
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    stop_async: asyncio.Event
    error: Exception | None = None


async def _watch_pipeline(
    session: PipelineSession,
    handle: coco.UpdateHandle[Any],
    stop_async: asyncio.Event,
) -> None:
    snaps: AsyncIterator[Any] | None = None
    had_errors = 0
    try:
        snaps = handle.watch()
        async for snap in snaps:
            if session.error is not None:
                break
            if snap.stats.total.num_errors:
                had_errors = snap.stats.total.num_errors
                if snap.status.name == "READY":
                    break
                continue
            if snap.status.name == "READY":
                await asyncio.to_thread(
                    reconcile_index_tables,
                    session.conn_mgr,
                    embedding_model=session.embedding_model,
                    task_mode=session.task_mode,
                    corpus_root=session.corpus_root,
                )
                session.ready_event.set()
        if had_errors and session.error is None:
            session.error = YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"CocoIndex live update reported {had_errors} component error(s).",
            )
    finally:
        try:
            if session.error is not None:
                session.ready_event.set()
                stop_async.set()
        finally:
            if isinstance(snaps, AsyncGenerator):
                await snaps.aclose()


def _build_embedder_memo_key(
    cfg: YawConfig,
    active_embed: Callable[[list[str]], NDArray[np.float32]],
) -> object:
    base_memo: tuple[object, ...] = (cfg.provider, cfg.embedding_model, cfg.base_url, cfg.embedding_task_mode)
    if isinstance(active_embed, DspyBatchEmbedder):
        return (*base_memo, "dspy")
    custom_id = (
        getattr(active_embed, "__module__", ""),
        getattr(active_embed, "__qualname__", type(active_embed).__qualname__),
    )
    return (*base_memo, "custom", custom_id)


def _make_component_error_handler(
    holder: dict[str, Any],
    err_lock: threading.Lock,
    ready_event: threading.Event,
) -> Callable[[BaseException, coco.ExceptionContext], None]:
    def _on_component_error(exc: BaseException, _ctx: coco.ExceptionContext) -> None:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise exc
        with err_lock:
            session = holder.get("session")
            if session is not None and session.error is None:
                session.error = _normalize_worker_error(exc)
        ready_event.set()
        raise exc

    return _on_component_error


async def _init_pipeline_session(
    *,
    db_path: Path,
    state_dir: Path,
    corpus_root: Path,
    cfg: YawConfig,
    conn_mgr: sqlite.ManagedConnection,
    embedder: PipelineEmbedder,
    resolved_app: str,
    live: bool,
    ready_event: threading.Event,
    stop_async: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
    on_component_error: Callable[[BaseException, coco.ExceptionContext], None],
) -> PipelineSession:
    ctx = coco.ContextProvider()
    ctx.provide(CORPUS_ROOT, corpus_root)
    ctx.provide(SQLITE_DB, conn_mgr)
    ctx.provide(EMBEDDER, embedder)
    env = coco.Environment(
        settings=coco.Settings(db_path=state_dir),
        context_provider=ctx,
        event_loop=loop,
        exception_handler=on_component_error,
    )
    success = False
    try:
        app = coco.App(
            coco.AppConfig(name=resolved_app, environment=env),
            indexing_main,
            tuple(cfg.included_patterns),
            tuple(cfg.excluded_patterns),
            cfg.chunk_size,
            cfg.chunk_overlap,
            cfg.min_chunk_size,
            cfg.embedding_task_mode,
            live,
        )
        session = PipelineSession(
            db_path=db_path,
            state_dir=state_dir,
            corpus_root=corpus_root,
            embedding_model=cfg.embedding_model,
            task_mode=cfg.embedding_task_mode,
            env=env,
            app=app,
            conn_mgr=conn_mgr,
            live=live,
            ready_event=ready_event,
            thread=threading.current_thread(),
            loop=loop,
            stop_async=stop_async,
        )
        success = True
        return session
    finally:
        if not success:
            env.close()


def _record_async_results(session: PipelineSession, results: Sequence[object]) -> None:
    for res in results:
        if isinstance(res, (KeyboardInterrupt, SystemExit)):
            raise res
        if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
            if session.error is None:
                session.error = _normalize_worker_error(res)
            session.ready_event.set()


async def _run_live_session(
    session: PipelineSession,
    stop_async: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
    cur: asyncio.Task[Any] | None,
) -> None:
    handle = session.app.update(live=True)
    watch_task = asyncio.create_task(_watch_pipeline(session, handle, stop_async))
    stop_task = asyncio.create_task(stop_async.wait())
    _, pending = await asyncio.wait({watch_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    results = await asyncio.gather(watch_task, stop_task, return_exceptions=True)
    _record_async_results(session, results)
    other = [t for t in asyncio.all_tasks(loop) if t is not cur and not t.done()]
    for t in other:
        t.cancel()
    if other:
        await asyncio.gather(*other, return_exceptions=True)
    (shutdown_res,) = await asyncio.gather(asyncio.wait_for(handle.result(), timeout=5.0), return_exceptions=True)
    if isinstance(shutdown_res, (KeyboardInterrupt, SystemExit)) or (
        isinstance(shutdown_res, asyncio.CancelledError) and cur is not None and cur.cancelling() > 0
    ):
        raise shutdown_res
    if isinstance(shutdown_res, BaseException) and not isinstance(shutdown_res, (TimeoutError, asyncio.CancelledError)):
        if session.error is None:
            session.error = _normalize_worker_error(shutdown_res)
        session.ready_event.set()
    del handle


async def _session_worker(
    *,
    db_path: Path,
    state_dir: Path,
    corpus_root: Path,
    cfg: YawConfig,
    conn_mgr: sqlite.ManagedConnection,
    embedder: PipelineEmbedder,
    resolved_app: str,
    live: bool,
    ready_event: threading.Event,
    init_event: threading.Event,
    holder: dict[str, Any],
    on_component_error: Callable[[BaseException, coco.ExceptionContext], None],
) -> None:
    loop = asyncio.get_running_loop()
    stop_async = asyncio.Event()
    (init_res,) = await asyncio.gather(
        _init_pipeline_session(
            db_path=db_path,
            state_dir=state_dir,
            corpus_root=corpus_root,
            cfg=cfg,
            conn_mgr=conn_mgr,
            embedder=embedder,
            resolved_app=resolved_app,
            live=live,
            ready_event=ready_event,
            stop_async=stop_async,
            loop=loop,
            on_component_error=on_component_error,
        ),
        return_exceptions=True,
    )
    cur = asyncio.current_task()
    if isinstance(init_res, BaseException):
        is_fatal = isinstance(init_res, (KeyboardInterrupt, SystemExit)) or (
            isinstance(init_res, asyncio.CancelledError) and cur is not None and cur.cancelling() > 0
        )
        holder["init_error"] = init_res if is_fatal else _normalize_worker_error(init_res)
        init_event.set()
        if is_fatal:
            raise init_res
        return
    session = init_res
    holder["session"] = session
    init_event.set()

    if not live:
        await stop_async.wait()
        return
    await _run_live_session(session, stop_async, loop, cur)


def start_pipeline(
    cfg: YawConfig,
    *,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    live: bool = True,
    app_name: str | None = None,
) -> PipelineSession:
    """Start the declarative CocoIndex indexing session (live watcher or catch-up app)."""
    assert cfg.db_path is not None and cfg.state_dir is not None and cfg.corpus_root is not None
    db_path = Path(cfg.db_path).resolve()
    corpus_root = Path(cfg.corpus_root).resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    state_dir = resolve_coco_state_root(cfg.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    active_embed = embed_fn or make_dspy_embed_fn(cfg)
    conn_mgr = open_managed_db(db_path)

    os.environ.setdefault("COCOINDEX_BANNER", "false")
    ready_event = threading.Event()
    init_event = threading.Event()
    err_lock = threading.Lock()
    holder: dict[str, Any] = {}
    on_component_error = _make_component_error_handler(holder, err_lock, ready_event)

    embedder = PipelineEmbedder(
        active_embed,
        memo_key=_build_embedder_memo_key(cfg, active_embed),
        batch_size=cfg.embed_batch_size,
        max_in_flight=cfg.embed_max_in_flight,
    )
    resolved_app = app_name or f"yaw_{hashlib.sha256(db_path.as_posix().encode('utf-8')).hexdigest()[:12]}"

    thread = threading.Thread(
        target=lambda: asyncio.run(
            _session_worker(
                db_path=db_path,
                state_dir=state_dir,
                corpus_root=corpus_root,
                cfg=cfg,
                conn_mgr=conn_mgr,
                embedder=embedder,
                resolved_app=resolved_app,
                live=live,
                ready_event=ready_event,
                init_event=init_event,
                holder=holder,
                on_component_error=on_component_error,
            )
        ),
        daemon=True,
        name="yaw-cocoindex-session",
    )
    try:
        thread.start()
    except Exception:
        conn_mgr.close()
        raise
    init_event.wait()
    if (init_err := holder.get("init_error")) is not None:
        thread.join(timeout=5.0)
        conn_mgr.close()
        raise init_err
    return cast(PipelineSession, holder["session"])


def update_pipeline_once(session: PipelineSession) -> None:
    """Execute one catch-up CocoIndex update synchronously and reconcile persisted index tables."""
    if session.error is not None:
        raise session.error
    session.app.update_blocking(live=False)
    if session.error is not None:
        raise session.error
    reconcile_index_tables(
        session.conn_mgr,
        embedding_model=session.embedding_model,
        task_mode=session.task_mode,
        corpus_root=session.corpus_root,
    )
    session.ready_event.set()


def wait_pipeline_ready(session: PipelineSession, timeout: float = 30.0) -> bool:
    """Block until initial CocoIndex catch-up and reconciliation finish; re-raises live watcher failures."""
    ready = session.ready_event.wait(timeout=timeout)
    if session.error is not None:
        raise session.error
    return ready


def close_pipeline(session: PipelineSession) -> None:
    """Stop the CocoIndex worker thread/loop cleanly and close Environment and SQLite handles."""
    try:
        if not session.loop.is_closed() and session.loop.is_running():
            with contextlib.suppress(RuntimeError):
                session.loop.call_soon_threadsafe(session.stop_async.set)
        if session.thread.is_alive() and session.thread is not threading.current_thread():
            session.thread.join(timeout=10.0)
            if session.thread.is_alive():
                raise YawRuntimeError("SHUTTING_DOWN", "Timed out waiting for CocoIndex session thread to stop.")
    finally:
        try:
            session.env.close()
        finally:
            session.conn_mgr.close()
