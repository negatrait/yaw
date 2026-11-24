"""Pipeline session lifecycle, shared-loop CocoIndex lifespan, and fail-fast error mapping."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import sqlite3
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectors import sqlite
from dspy.utils.exceptions import DSPyError
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError, resolve_coco_state_root

from .embedder import EMBEDDER, DspyBatchEmbedder, PipelineEmbedder, make_dspy_embed_fn
from .flow import CORPUS_ROOT, SQLITE_DB, indexing_main
from .targets import SINK_TRACKER, SinkMutationTracker, open_managed_db, reconcile_index_tables

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
    """Handle bundle owning CocoIndex environment/app handles, embedder, and SQLite connection."""

    db_path: Path
    state_dir: Path
    corpus_root: Path
    embedding_model: str
    task_mode: str
    cfg: YawConfig
    resolved_app: str
    embedder: PipelineEmbedder
    conn_mgr: sqlite.ManagedConnection
    live: bool
    env: coco.Environment | None = None
    app: coco.App[Any, Any] | None = None
    sink_tracker: SinkMutationTracker | None = None
    ready_event: asyncio.Event | None = None
    error: Exception | None = None


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


def _deliver_component_error(
    norm: Exception,
    loop: asyncio.AbstractEventLoop,
    ready_event: asyncio.Event | None,
    on_error: Callable[[Exception], None] | None,
) -> None:
    def _notify() -> None:
        if on_error is not None:
            on_error(norm)
        if ready_event is not None:
            ready_event.set()

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _notify()
    elif not loop.is_closed() and loop.is_running():
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(_notify)


def _make_component_error_handler(
    holder: dict[str, Any],
    err_lock: threading.Lock,
    loop: asyncio.AbstractEventLoop,
    ready_event: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> Callable[[BaseException, coco.ExceptionContext], None]:
    def _on_component_error(exc: BaseException, _ctx: coco.ExceptionContext) -> None:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise exc
        norm = _normalize_worker_error(exc)
        with err_lock:
            session = holder.get("session")
            if session is not None and session.error is None:
                session.error = norm
            elif holder.get("error") is None:
                holder["error"] = norm
        _deliver_component_error(norm, loop, ready_event, on_error)
        raise exc

    return _on_component_error


def _build_coco_env_and_app(
    *,
    state_dir: Path,
    corpus_root: Path,
    conn_mgr: sqlite.ManagedConnection,
    embedder: PipelineEmbedder,
    cfg: YawConfig,
    resolved_app: str,
    live: bool,
    loop: asyncio.AbstractEventLoop,
    on_component_error: Callable[[BaseException, coco.ExceptionContext], None],
) -> tuple[coco.Environment, coco.App[Any, Any], SinkMutationTracker]:
    tracker = SinkMutationTracker()
    ctx = coco.ContextProvider()
    ctx.provide(CORPUS_ROOT, corpus_root)
    ctx.provide(SQLITE_DB, conn_mgr)
    ctx.provide(EMBEDDER, embedder)
    ctx.provide(SINK_TRACKER, tracker)
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
        success = True
        return env, app, tracker
    finally:
        if not success:
            env.close()


async def _to_thread_joined[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


async def _reconcile_until_settled(
    session: PipelineSession,
    tracker: SinkMutationTracker,
    last_reconciled_seq: int,
    *,
    initial: bool,
    stop_requested: list[bool],
) -> tuple[bool, int]:
    changed_any = False
    pending_initial = initial
    while (pending_initial or tracker.seq != last_reconciled_seq) and not stop_requested[0]:
        seq_snapshot = tracker.seq
        changed = await _to_thread_joined(
            reconcile_index_tables,
            session.conn_mgr,
            embedding_model=session.embedding_model,
            task_mode=session.task_mode,
            corpus_root=session.corpus_root,
        )
        changed_any = changed_any or changed
        last_reconciled_seq = seq_snapshot
        pending_initial = False
    return changed_any, last_reconciled_seq


async def _reconcile_snapshot_step(
    session: PipelineSession,
    tracker: SinkMutationTracker,
    last_reconciled_seq: int,
    *,
    is_initial: bool,
    stop_requested: list[bool],
    on_reconciled: Callable[[bool, bool], Awaitable[None]] | None,
) -> int:
    changed, next_seq = await _reconcile_until_settled(
        session,
        tracker,
        last_reconciled_seq,
        initial=is_initial,
        stop_requested=stop_requested,
    )
    if not stop_requested[0] and on_reconciled is not None:
        await on_reconciled(changed, is_initial)
    if not stop_requested[0] and session.ready_event is not None:
        session.ready_event.set()
    return next_seq


async def _consume_watch_snapshots(
    session: PipelineSession,
    snaps: AsyncIterator[Any],
    tracker: SinkMutationTracker,
    inflight_reconcile: list[asyncio.Task[int] | None],
    stop_requested: list[bool],
    on_reconciled: Callable[[bool, bool], Awaitable[None]] | None,
) -> int:
    initial_ready_done = False
    last_reconciled_seq = 0
    had_errors = 0
    async for snap in snaps:
        if stop_requested[0] or session.error is not None:
            break
        if snap.stats.total.num_errors:
            had_errors = int(snap.stats.total.num_errors)
            if snap.status.name == "READY":
                break
            continue
        if snap.status.name == "READY" and snap.stats.total.num_in_progress == 0:
            is_initial = not initial_ready_done
            if is_initial or tracker.seq != last_reconciled_seq:
                task = asyncio.create_task(
                    _reconcile_snapshot_step(
                        session,
                        tracker,
                        last_reconciled_seq,
                        is_initial=is_initial,
                        stop_requested=stop_requested,
                        on_reconciled=on_reconciled,
                    )
                )
                inflight_reconcile[0] = task
                try:
                    last_reconciled_seq = await asyncio.shield(task)
                    inflight_reconcile[0] = None
                except BaseException as exc:
                    if not isinstance(exc, asyncio.CancelledError) and task.done():
                        inflight_reconcile[0] = None
                    raise
                initial_ready_done = True
    return had_errors


def _finalize_watch_result(
    session: PipelineSession,
    watch_res: int | BaseException,
    cur: asyncio.Task[Any] | None,
) -> None:
    if isinstance(watch_res, (KeyboardInterrupt, SystemExit)) or (
        isinstance(watch_res, asyncio.CancelledError) and cur is not None and cur.cancelling() > 0
    ):
        raise watch_res
    if isinstance(watch_res, BaseException):
        if session.error is None:
            session.error = _normalize_worker_error(watch_res)
    elif watch_res and session.error is None:
        session.error = YawRuntimeError(
            "CONSISTENCY_ERROR",
            f"CocoIndex live update reported {int(watch_res)} component error(s).",
        )


async def _watch_pipeline(
    session: PipelineSession,
    handle: coco.UpdateHandle[Any],
    tracker: SinkMutationTracker,
    *,
    inflight_reconcile: list[asyncio.Task[int] | None],
    stop_requested: list[bool],
    on_reconciled: Callable[[bool, bool], Awaitable[None]] | None,
    on_error: Callable[[Exception], None] | None,
) -> None:
    snaps_box: list[AsyncIterator[Any]] = []
    cur = asyncio.current_task()

    async def _run_watch() -> int:
        snaps = handle.watch()
        snaps_box.append(snaps)
        return await _consume_watch_snapshots(
            session,
            snaps,
            tracker,
            inflight_reconcile,
            stop_requested,
            on_reconciled,
        )

    try:
        (watch_res,) = await asyncio.gather(_run_watch(), return_exceptions=True)
        _finalize_watch_result(session, watch_res, cur)
    finally:
        try:
            if session.error is not None:
                if on_error is not None:
                    on_error(session.error)
                if session.ready_event is not None:
                    session.ready_event.set()
        finally:
            if snaps_box and isinstance(snaps_box[0], AsyncGenerator):
                await snaps_box[0].aclose()


def _record_async_results(
    session: PipelineSession,
    results: Sequence[object],
    on_error: Callable[[Exception], None] | None,
) -> None:
    for res in results:
        if isinstance(res, (KeyboardInterrupt, SystemExit)):
            raise res
        if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
            if session.error is None:
                session.error = _normalize_worker_error(res)
            err = session.error
            if on_error is not None:
                on_error(err)
            if session.ready_event is not None:
                session.ready_event.set()


async def _stop_live_handle(
    session: PipelineSession,
    handle: coco.UpdateHandle[Any],
    watch_task: asyncio.Task[None],
    loop: asyncio.AbstractEventLoop,
    on_error: Callable[[Exception], None] | None,
    *,
    inflight_reconcile: list[asyncio.Task[int] | None],
    stop_requested: list[bool],
) -> None:
    stop_requested[0] = True
    watch_task.cancel()
    results = await asyncio.gather(watch_task, return_exceptions=True)
    _record_async_results(session, results, on_error)
    pending_reconcile = inflight_reconcile[0]
    if pending_reconcile is not None:
        inflight_reconcile[0] = None
        reconcile_results = await asyncio.gather(pending_reconcile, return_exceptions=True)
        _record_async_results(session, reconcile_results, on_error)
    cur = asyncio.current_task()
    live_tasks = [
        t
        for t in asyncio.all_tasks(loop)
        if t is not cur and not t.done() and getattr(t.get_coro(), "__qualname__", None) == "_process_live_wrapper"
    ]
    for task in live_tasks:
        task.cancel()
    if live_tasks:
        await asyncio.gather(*live_tasks, return_exceptions=True)
    (shutdown_res,) = await asyncio.gather(asyncio.wait_for(handle.result(), timeout=2.0), return_exceptions=True)
    if isinstance(shutdown_res, (KeyboardInterrupt, SystemExit)) or (
        isinstance(shutdown_res, asyncio.CancelledError) and cur is not None and cur.cancelling() > 0
    ):
        raise shutdown_res
    if isinstance(shutdown_res, BaseException) and not isinstance(shutdown_res, (TimeoutError, asyncio.CancelledError)):
        if session.error is None:
            session.error = _normalize_worker_error(shutdown_res)
        err = session.error
        if on_error is not None:
            on_error(err)
        if session.ready_event is not None:
            session.ready_event.set()


def _prepare_session_paths(
    cfg: YawConfig,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None,
    app_name: str | None,
) -> tuple[Path, Path, Path, sqlite.ManagedConnection, PipelineEmbedder, str]:
    assert cfg.db_path is not None and cfg.state_dir is not None and cfg.corpus_root is not None
    db_path = Path(cfg.db_path).resolve()
    corpus_root = Path(cfg.corpus_root).resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    state_dir = resolve_coco_state_root(cfg.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COCOINDEX_BANNER", "false")
    active_embed = embed_fn or make_dspy_embed_fn(cfg)
    conn_mgr = open_managed_db(db_path)
    embedder = PipelineEmbedder(
        active_embed,
        memo_key=_build_embedder_memo_key(cfg, active_embed),
        batch_size=cfg.embed_batch_size,
        max_in_flight=cfg.embed_max_in_flight,
    )
    resolved_app = app_name or f"yaw_{hashlib.sha256(db_path.as_posix().encode('utf-8')).hexdigest()[:12]}"
    return db_path, state_dir, corpus_root, conn_mgr, embedder, resolved_app


@contextlib.asynccontextmanager
async def open_pipeline_lifespan(
    cfg: YawConfig,
    *,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    live: bool = True,
    app_name: str | None = None,
    on_reconciled: Callable[[bool, bool], Awaitable[None]] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> AsyncGenerator[PipelineSession]:
    """Run declarative CocoIndex lifecycle directly on the caller's asyncio event loop."""
    loop = asyncio.get_running_loop()
    db_path, state_dir, corpus_root, conn_mgr, embedder, resolved_app = _prepare_session_paths(cfg, embed_fn, app_name)
    ready_event = asyncio.Event()
    err_lock = threading.Lock()
    holder: dict[str, Any] = {"session": None, "error": None}
    error_notified = False

    def _notify_once(exc: Exception) -> None:
        nonlocal error_notified
        if error_notified:
            return
        error_notified = True
        if on_error is not None:
            on_error(exc)

    on_err = _make_component_error_handler(holder, err_lock, loop, ready_event, _notify_once)
    env: coco.Environment | None = None
    try:
        env, app, tracker = _build_coco_env_and_app(
            state_dir=state_dir,
            corpus_root=corpus_root,
            conn_mgr=conn_mgr,
            embedder=embedder,
            cfg=cfg,
            resolved_app=resolved_app,
            live=live,
            loop=loop,
            on_component_error=on_err,
        )
        session = PipelineSession(
            db_path=db_path,
            state_dir=state_dir,
            corpus_root=corpus_root,
            embedding_model=cfg.embedding_model,
            task_mode=cfg.embedding_task_mode,
            cfg=cfg,
            resolved_app=resolved_app,
            embedder=embedder,
            conn_mgr=conn_mgr,
            live=live,
            env=env,
            app=app,
            sink_tracker=tracker,
            ready_event=ready_event,
        )
        with err_lock:
            holder["session"] = session
            if holder["error"] is not None and session.error is None:
                session.error = holder["error"]
        if live:
            handle = app.update(live=True)
            inflight_reconcile: list[asyncio.Task[int] | None] = [None]
            stop_requested = [False]
            watch_task = asyncio.create_task(
                _watch_pipeline(
                    session,
                    handle,
                    tracker,
                    inflight_reconcile=inflight_reconcile,
                    stop_requested=stop_requested,
                    on_reconciled=on_reconciled,
                    on_error=_notify_once,
                )
            )
            try:
                yield session
            finally:
                await _stop_live_handle(
                    session,
                    handle,
                    watch_task,
                    loop,
                    _notify_once,
                    inflight_reconcile=inflight_reconcile,
                    stop_requested=stop_requested,
                )
        else:
            await app.update(live=False).result()
            changed = await _to_thread_joined(
                reconcile_index_tables,
                conn_mgr,
                embedding_model=cfg.embedding_model,
                task_mode=cfg.embedding_task_mode,
                corpus_root=corpus_root,
            )
            if on_reconciled is not None:
                await on_reconciled(changed, True)
            ready_event.set()
            yield session
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            conn_mgr.close()


def start_pipeline(
    cfg: YawConfig,
    *,
    embed_fn: Callable[[list[str]], NDArray[np.float32]] | None = None,
    live: bool = False,
    app_name: str | None = None,
) -> PipelineSession:
    """Create a thread-free catch-up CocoIndex PipelineSession (live=False only)."""
    if live:
        raise YawRuntimeError(
            "INVALID_INPUT",
            "start_pipeline() supports live=False only; use open_pipeline_lifespan() for live pipelines.",
        )
    db_path, state_dir, corpus_root, conn_mgr, embedder, resolved_app = _prepare_session_paths(cfg, embed_fn, app_name)
    return PipelineSession(
        db_path=db_path,
        state_dir=state_dir,
        corpus_root=corpus_root,
        embedding_model=cfg.embedding_model,
        task_mode=cfg.embedding_task_mode,
        cfg=cfg,
        resolved_app=resolved_app,
        embedder=embedder,
        conn_mgr=conn_mgr,
        live=False,
    )


async def _run_catchup_action(session: PipelineSession, *, drop: bool) -> None:
    loop = asyncio.get_running_loop()
    err_lock = threading.Lock()
    holder: dict[str, Any] = {"session": session, "error": None}
    on_err = _make_component_error_handler(holder, err_lock, loop)
    env, app, _ = _build_coco_env_and_app(
        state_dir=session.state_dir,
        corpus_root=session.corpus_root,
        conn_mgr=session.conn_mgr,
        embedder=session.embedder,
        cfg=session.cfg,
        resolved_app=session.resolved_app,
        live=False,
        loop=loop,
        on_component_error=on_err,
    )
    try:
        if drop:
            await app.drop()
        else:
            await app.update(live=False).result()
    finally:
        env.close()


def update_pipeline_once(session: PipelineSession) -> None:
    """Execute one catch-up CocoIndex update synchronously and reconcile persisted index tables."""
    if session.error is not None:
        raise session.error
    asyncio.run(_run_catchup_action(session, drop=False))
    if session.error is not None:
        raise session.error
    reconcile_index_tables(
        session.conn_mgr,
        embedding_model=session.embedding_model,
        task_mode=session.task_mode,
        corpus_root=session.corpus_root,
    )


def drop_pipeline(session: PipelineSession) -> None:
    """Drop all CocoIndex target states and tracking memos for a catch-up session."""
    asyncio.run(_run_catchup_action(session, drop=True))


def close_pipeline(session: PipelineSession) -> None:
    """Close any open CocoIndex Environment and SQLite ManagedConnection."""
    try:
        if session.env is not None:
            session.env.close()
            session.env = None
    finally:
        session.conn_mgr.close()
