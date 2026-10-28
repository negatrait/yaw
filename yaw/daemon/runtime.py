"""Single-owner persistent daemon runtime with FileLock, 32-slot bounded queue, and crash-resilient CocoIndex sync."""

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar

import filelock

from yaw.core import (
    ChunkPointer,
    DaemonStatus,
    HydratedPassage,
    IndexStats,
    SyncDelta,
    YawConfig,
    YawRuntimeError,
    configure_logging,
    write_text_atomic,
)
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    HippoRAGProgram,
    check_benchmark_integrity,
    compute_qa_em_f1,
    compute_retrieval_recalls,
    create_dspy_rlm_pair,
    evaluate_benchmark_gate,
)
from yaw.daemon.ingestion import (
    DEFAULT_PARITY_DATASETS,
    MONKEYPATCHED_HIPPORAG_BASELINES,
    PAPER_BASELINES,
    CorpusIngestor,
    DatasetBundle,
    build_benchmark_slice,
    resolve_dataset_bundle,
    snapshot_dataset_bundle,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

_PENDING_DELTA_META_KEY = "pending_sync_delta"
_CONFIG_FP_META_KEY = "config_fingerprint"

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


def _win32_stop_event_name(lock_path: Path | str) -> str:
    resolved = str(Path(lock_path).resolve()).replace("\\", "/").lower()
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return f"Local\\yaw-stop-{digest}"


def is_file_locked(lock_path: Path | str | None) -> bool:
    """Return True if `lock_path` exists and is currently held by a FileLock owner."""
    if lock_path is None:
        return False
    lock_p = Path(lock_path)
    if not lock_p.exists():
        return False
    lock = filelock.FileLock(str(lock_p), timeout=0)
    try:
        lock.acquire(timeout=0)
        lock.release()
        return False
    except filelock.Timeout:
        return True


@dataclass(slots=True)
class _WorkItem:
    op_name: str
    fn: Callable[[], Any]
    future: Future[Any]
    progress_callback: Callable[[str], None] | None = None


def _serialize_sync_delta(delta: SyncDelta) -> str:
    return json.dumps(
        {
            "added_chunks": [[asdict(ptr), text] for ptr, text in delta.added_chunks],
            "deleted_chunk_ids": list(delta.deleted_chunk_ids),
        },
        ensure_ascii=False,
    )


def _deserialize_sync_delta(raw: str) -> SyncDelta:
    data = json.loads(raw)
    added = tuple((ChunkPointer(**ptr_dict), str(text)) for ptr_dict, text in data.get("added_chunks", ()))
    deleted = tuple(str(cid) for cid in data.get("deleted_chunk_ids", ()))
    return SyncDelta(added_chunks=added, deleted_chunk_ids=deleted)


def validate_query_input(text: str) -> str:
    """Validate and trim a query/question input (non-empty, <= 1024 UTF-8 bytes)."""
    if not isinstance(text, str):
        raise YawRuntimeError("INVALID_INPUT", "Query must be a string.")
    cleaned = text.strip()
    if not cleaned:
        raise YawRuntimeError("INVALID_INPUT", "Query must not be empty.")
    if len(cleaned.encode("utf-8")) > 1024:
        raise YawRuntimeError("INVALID_INPUT", "Query exceeds maximum length of 1024 UTF-8 bytes.")
    return cleaned


class YawRuntime:
    """Single-owner persistent daemon runtime managing yaw.db, CocoIndex, and bounded worker execution."""

    _active_lock: ClassVar[threading.Lock] = threading.Lock()
    _active_instances: ClassVar[dict[Path, YawRuntime]] = {}

    def __init__(
        self,
        config: YawConfig,
        *,
        program: HippoRAGProgram | None = None,
        embedder: DSPyEmbeddingAdapter | None = None,
        auto_poll: bool = False,
    ) -> None:
        self.config = config
        self._custom_program = program
        self._custom_embedder = embedder
        self.auto_poll = auto_poll

        assert self.config.lock_path is not None
        self.config.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = filelock.FileLock(str(self.config.lock_path), timeout=0)

        self._state_lock = threading.Lock()
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue(maxsize=self.config.queue_maxsize)
        self._stop_event = threading.Event()
        self._has_started = False
        self._worker_thread: threading.Thread | None = None
        self._poller_thread: threading.Thread | None = None
        self._win32_stop_handle: int | None = None
        self._win32_waiter_thread: threading.Thread | None = None

        self._pending_sync_future: Future[SyncDelta] | None = None
        self._pending_scan_id: str | None = None

        self.ingestor: CorpusIngestor | None = None
        self.engine: YawGraphEngine | None = None

        self._stats = IndexStats(config_fingerprint=self.config.config_fingerprint())
        self._status = DaemonStatus(
            lifecycle="stopped",
            provider_ready=False,
            sync_state="idle",
            queue_depth=0,
            active_op=None,
            last_error=None,
            last_sync_at=None,
            pid=None,
        )

    @classmethod
    def signal_local_stop(cls, lock_path: Path) -> bool:
        """Signal an in-process YawRuntime or Windows Named Kernel Event holding lock_path to stop gracefully."""
        signaled = False
        with cls._active_lock:
            rt = cls._active_instances.get(lock_path.resolve())
        if rt is not None:
            rt.request_stop()
            signaled = True
        if os.name == "nt":
            event_name = _win32_stop_event_name(lock_path)
            handle = _kernel32.OpenEventW(0x0002, 0, event_name)
            if handle:
                try:
                    if _kernel32.SetEvent(handle):
                        signaled = True
                finally:
                    _kernel32.CloseHandle(handle)
        return signaled

    def request_stop(self) -> None:
        """Signal the runtime stop event and wake the worker loop."""
        self._stop_event.set()
        if os.name == "nt" and self._win32_stop_handle:
            with contextlib.suppress(OSError):
                _kernel32.SetEvent(self._win32_stop_handle)
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the runtime is signaled to stop or timeout expires."""
        return self._stop_event.wait(timeout=timeout)

    def _open_components(self) -> None:
        assert self.config.corpus_root is not None
        assert self.config.state_dir is not None
        assert self.config.db_path is not None

        if self._custom_program is not None:
            program = self._custom_program
            if self.config.compiled_program_path and Path(self.config.compiled_program_path).exists():
                program.load(str(self.config.compiled_program_path))
        else:
            program = HippoRAGProgram.from_config(self.config)

        embedder = (
            self._custom_embedder
            if self._custom_embedder is not None
            else DSPyEmbeddingAdapter.from_config(self.config)
        )

        assert self.config.max_context_chars is not None
        self.engine = YawGraphEngine(
            db_path=self.config.db_path,
            workspace_root=self.config.corpus_root,
            program=program,
            embedder=embedder,
            linking_top_k=self.config.linking_top_k,
            retrieval_top_k=self.config.retrieval_top_k,
            damping=self.config.damping,
            synonymy_threshold=self.config.synonymy_threshold,
            synonymy_top_k=self.config.synonymy_top_k,
            embed_batch_size=self.config.embed_batch_size,
            max_context_chars=self.config.max_context_chars,
            max_ast_expansion_hops=self.config.max_ast_expansion_hops,
            ast_context_hops=self.config.ast_context_hops,
        )
        self.engine.config = self.config
        self.ingestor = CorpusIngestor(
            corpus_root=self.config.corpus_root,
            state_dir=self.config.state_dir,
            included_patterns=self.config.included_patterns,
            excluded_patterns=self.config.excluded_patterns,
            json_keys_whitelist=self.config.json_keys_whitelist,
            json_keys_blacklist=self.config.json_keys_blacklist,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            on_delta=self._apply_delta,
        )
        fp = self.config.config_fingerprint()
        self.engine.set_meta(_CONFIG_FP_META_KEY, fp)
        stats = self.engine.get_stats(fp)
        with self._state_lock:
            self._stats = stats

    def _close_components(self) -> None:
        if self.ingestor is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self.ingestor.close()
            self.ingestor = None
        if self.engine is not None:
            with contextlib.suppress(OSError, sqlite3.Error, RuntimeError):
                self.engine.close()
            self.engine = None

    def _update_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            current = self._status.to_dict()
            current.update(kwargs)
            current["queue_depth"] = self._queue.qsize()
            self._status = DaemonStatus(**current)
            snapshot = {
                "status": self._status.to_dict(),
                "stats": self._stats.to_dict(),
            }
        self._persist_status_file(snapshot)

    def _persist_status_file(self, snapshot: dict[str, Any]) -> None:
        if self.config.status_path is None:
            return
        status_path = Path(self.config.status_path)
        for _ in range(5):
            try:
                write_text_atomic(status_path, json.dumps(snapshot, indent=2))
                break
            except OSError:
                time.sleep(0.01)

    def _start_win32_stop_event(self) -> None:
        if os.name != "nt" or self.config.lock_path is None:
            return
        event_name = _win32_stop_event_name(self.config.lock_path)
        handle = _kernel32.CreateEventW(None, 1, 0, event_name)
        if not handle:
            return
        _kernel32.ResetEvent(handle)
        self._win32_stop_handle = int(handle)

        def _wait_kernel_event(ev_handle: int) -> None:
            res = _kernel32.WaitForSingleObject(ev_handle, 0xFFFFFFFF)
            if res == 0 and not self._stop_event.is_set():
                self.request_stop()

        self._win32_waiter_thread = threading.Thread(
            target=_wait_kernel_event,
            args=(int(handle),),
            name="yaw-win32-stop-waiter",
            daemon=True,
        )
        self._win32_waiter_thread.start()

    def _close_win32_stop_event(self) -> None:
        if os.name != "nt":
            return
        handle = self._win32_stop_handle
        self._win32_stop_handle = None
        if handle:
            with contextlib.suppress(OSError):
                _kernel32.SetEvent(handle)
        if (
            self._win32_waiter_thread is not None
            and self._win32_waiter_thread.is_alive()
            and self._win32_waiter_thread is not threading.current_thread()
        ):
            self._win32_waiter_thread.join(timeout=1.0)
        self._win32_waiter_thread = None
        if handle:
            with contextlib.suppress(OSError):
                _kernel32.CloseHandle(handle)

    def start(self, *, sync_on_start: bool = True) -> None:
        """Acquire exclusive yaw.db.lock, open CocoIndex + YawGraphEngine, and start worker thread."""
        with self._state_lock:
            if self._status.lifecycle in ("starting", "ready", "degraded"):
                return

        try:
            self._file_lock.acquire(timeout=0.2)
        except filelock.Timeout as exc:
            raise YawRuntimeError(
                "BUSY",
                f"Database lock {self.config.lock_path} is already held by another YAW daemon.",
            ) from exc

        assert self.config.lock_path is not None
        with self._active_lock:
            self._active_instances[self.config.lock_path.resolve()] = self

        try:
            if self._has_started:
                self._stop_event.clear()
            self._has_started = True
            self._start_win32_stop_event()
            if self.config.log_path is not None:
                configure_logging(self.config.log_path)
            self._update_state(
                lifecycle="starting",
                provider_ready=True,
                sync_state="idle",
                active_op=None,
                last_error=None,
                pid=os.getpid(),
            )
            self._open_components()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="yaw-runtime-worker",
                daemon=True,
            )
            self._worker_thread.start()

            has_pending = self.engine is not None and self.engine.get_meta(_PENDING_DELTA_META_KEY) is not None
            if (sync_on_start or has_pending) and not self._stop_event.is_set():
                try:
                    self.sync_now()
                except YawRuntimeError as err:
                    if err.code != "SHUTTING_DOWN" or not self._stop_event.is_set():
                        raise
            elif not self._stop_event.is_set():
                self._update_state(lifecycle="ready")

            if self._stop_event.is_set():
                self.stop()
                return

            if self.auto_poll and self.config.poll_interval_seconds > 0 and not self._stop_event.is_set():
                self._poller_thread = threading.Thread(
                    target=self._poll_loop,
                    name="yaw-runtime-poller",
                    daemon=True,
                )
                self._poller_thread.start()
        except Exception:
            self._close_win32_stop_event()
            if self.config.lock_path is not None:
                with self._active_lock:
                    self._active_instances.pop(self.config.lock_path.resolve(), None)
            self._close_components()
            if self._file_lock.is_locked:
                self._file_lock.release()
            raise

    def stop(self) -> None:
        """Gracefully stop poller and worker threads, close databases, and release yaw.db.lock."""
        self._stop_event.set()
        if self.config.lock_path is not None:
            with self._active_lock:
                self._active_instances.pop(self.config.lock_path.resolve(), None)

        with self._state_lock:
            if self._status.lifecycle == "stopped":
                self._close_win32_stop_event()
                if self._file_lock.is_locked:
                    self._file_lock.release()
                return
            self._status = DaemonStatus(
                lifecycle="stopping",
                provider_ready=self._status.provider_ready,
                sync_state="idle",
                queue_depth=self._queue.qsize(),
                active_op=self._status.active_op,
                last_error=self._status.last_error,
                last_sync_at=self._status.last_sync_at,
                pid=self._status.pid,
            )

        self._stop_event.set()
        self._close_win32_stop_event()

        if self._poller_thread is not None and self._poller_thread.is_alive():
            self._poller_thread.join(timeout=2.0)
        self._poller_thread = None

        # Cancel queued items that have not started yet
        while True:
            try:
                queued_item = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued_item is not None and not queued_item.future.done():
                queued_item.future.set_exception(
                    YawRuntimeError("SHUTTING_DOWN", "Daemon stopped before task execution.")
                )
            self._queue.task_done()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(None)
            self._worker_thread.join(timeout=5.0)
        self._worker_thread = None

        self._close_components()
        self._update_state(
            lifecycle="stopped",
            provider_ready=False,
            sync_state="idle",
            active_op=None,
            pid=None,
        )
        if self._file_lock.is_locked:
            self._file_lock.release()

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> YawRuntime:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def status(self) -> DaemonStatus:
        """Return cached DaemonStatus immediately without blocking on the worker thread."""
        with self._state_lock:
            return DaemonStatus(
                lifecycle=self._status.lifecycle,
                provider_ready=self._status.provider_ready,
                sync_state=self._status.sync_state,
                queue_depth=self._queue.qsize(),
                active_op=self._status.active_op,
                last_error=self._status.last_error,
                last_sync_at=self._status.last_sync_at,
                pid=self._status.pid,
            )

    def stats(self) -> IndexStats:
        """Return cached IndexStats immediately without blocking on the worker thread."""
        with self._state_lock:
            return self._stats

    def submit(
        self,
        op_name: str,
        fn: Callable[[], T],
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Future[T]:
        """Enqueue a task on the 32-slot bounded worker queue or fail fast with BUSY."""
        with self._state_lock:
            if self._status.lifecycle in ("stopping", "stopped"):
                raise YawRuntimeError("SHUTTING_DOWN", "Daemon runtime is not running.")
            if self._status.lifecycle == "starting" and op_name != "sync":
                raise YawRuntimeError("NOT_READY", "Daemon runtime is still starting.")

        fut: Future[T] = Future()
        item = _WorkItem(
            op_name=op_name,
            fn=fn,
            future=fut,
            progress_callback=progress_callback,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            raise YawRuntimeError(
                "BUSY",
                f"Runtime worker queue ({self.config.queue_maxsize} slots) is full.",
            ) from exc

        if progress_callback is not None:
            progress_callback("queued")
        self._update_state()
        return fut

    def _poll_loop(self) -> None:
        poll_interval = max(0.05, float(self.config.poll_interval_seconds))
        last_sync_check = time.monotonic()
        assert self.config.workspace_root is not None
        yaw_dir = (self.config.workspace_root / ".yaw").resolve()
        signal_file = yaw_dir / "reload.signal"
        result_file = yaw_dir / "reload.result"

        while not self._stop_event.is_set():
            if signal_file.exists():
                with contextlib.suppress(Exception):
                    raw = signal_file.read_text(encoding="utf-8")
                    data = json.loads(raw) if raw.strip() else {}
                    prog_path = data.get("program_path")
                    req_id = data.get("req_id")
                    if prog_path:
                        try:
                            res = self.reload_program(prog_path)
                            res["req_id"] = req_id
                            result_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
                        except Exception as exc:
                            err_payload = {
                                "reloaded": False,
                                "req_id": req_id,
                                "error": str(exc),
                                "pid": os.getpid(),
                            }
                            result_file.write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
                        finally:
                            with contextlib.suppress(OSError):
                                signal_file.unlink(missing_ok=True)

            now = time.monotonic()
            if now - last_sync_check >= poll_interval:
                last_sync_check = now
                with contextlib.suppress(YawRuntimeError):
                    self.request_sync()

            if self._stop_event.wait(0.2):
                break

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            if not item.future.set_running_or_notify_cancel():
                self._queue.task_done()
                self._update_state()
                continue

            sync_state = "syncing" if item.op_name in ("sync", "rebuild") else self._status.sync_state
            self._update_state(active_op=item.op_name, sync_state=sync_state)
            if item.progress_callback is not None:
                item.progress_callback("started")

            try:
                result = item.fn()
            except YawRuntimeError as err:
                if err.code == "MODEL_ERROR":
                    self._update_state(
                        lifecycle="degraded",
                        provider_ready=False,
                        sync_state="error" if item.op_name == "sync" else "idle",
                        active_op=None,
                        last_error=err.message[:300],
                    )
                else:
                    self._update_state(active_op=None, sync_state="idle")
                item.future.set_exception(err)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                err_code: Literal["CONSISTENCY_ERROR", "MODEL_ERROR"] = (
                    "CONSISTENCY_ERROR" if isinstance(exc, sqlite3.DatabaseError) else "MODEL_ERROR"
                )
                wrapped = YawRuntimeError(err_code, msg)
                self._update_state(
                    lifecycle="degraded" if err_code == "MODEL_ERROR" else self._status.lifecycle,
                    provider_ready=err_code != "MODEL_ERROR",
                    sync_state="error" if item.op_name == "sync" else "idle",
                    active_op=None,
                    last_error=msg[:300],
                )
                item.future.set_exception(wrapped)
            else:
                if item.progress_callback is not None:
                    item.progress_callback("ready")
                now_ts = time.time() if item.op_name in ("sync", "rebuild") else self._status.last_sync_at
                self._update_state(
                    lifecycle="ready",
                    provider_ready=True,
                    sync_state="idle",
                    active_op=None,
                    last_error=None,
                    last_sync_at=now_ts,
                )
                item.future.set_result(result)
            finally:
                self._queue.task_done()

        while True:
            try:
                queued_item = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued_item is not None and not queued_item.future.done():
                queued_item.future.set_exception(
                    YawRuntimeError("SHUTTING_DOWN", "Daemon is shutting down before queued operation could start.")
                )
            self._queue.task_done()

    def _apply_delta(
        self,
        delta: SyncDelta,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        assert self.engine is not None
        if not delta.deleted_chunk_ids and not delta.added_chunks:
            return
        cb = progress_callback
        if cb is None:

            def _default_sync_progress(msg: str) -> None:
                self._update_state(active_op=msg, sync_state="syncing")

            cb = _default_sync_progress
        self.engine.set_meta(_PENDING_DELTA_META_KEY, _serialize_sync_delta(delta))
        if delta.deleted_chunk_ids:
            self.engine.delete_chunks(delta.deleted_chunk_ids)
        if delta.added_chunks:
            self.engine.index_chunks(delta.added_chunks, progress_callback=cb)
        self.engine.set_meta(_PENDING_DELTA_META_KEY, None)

    def _execute_sync_worker(self) -> SyncDelta:
        if self.ingestor is None or self.engine is None:
            raise YawRuntimeError("NOT_READY", "Engine or ingestor is not initialized.")

        def _sync_progress(msg: str) -> None:
            self._update_state(active_op=msg, sync_state="syncing")

        # 1. Replay any uncommitted pending delta before polling CocoIndex
        raw_pending = self.engine.get_meta(_PENDING_DELTA_META_KEY)
        if raw_pending:
            pending_delta = _deserialize_sync_delta(raw_pending)
            self._apply_delta(pending_delta, progress_callback=_sync_progress)
        else:
            pending_delta = None

        # 2. Poll CocoIndex for subsequent corpus changes (compiled inside target sink before LMDB commit)
        _sync_progress("polling CocoIndex for corpus changes")
        polled_delta = self.ingestor.poll()
        if getattr(self.ingestor, "on_delta", getattr(self.ingestor, "_on_delta", None)) is None:
            self._apply_delta(polled_delta, progress_callback=_sync_progress)

        new_stats = self.engine.get_stats(self.config.config_fingerprint())
        with self._state_lock:
            self._stats = new_stats

        if polled_delta.added_chunks or polled_delta.deleted_chunk_ids or pending_delta is None:
            return polled_delta
        return pending_delta

    def request_sync(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Request an asynchronous coalesced CocoIndex sync on the worker thread."""
        with self._state_lock:
            if self._pending_sync_future is not None and not self._pending_sync_future.done():
                return {
                    "scan_id": self._pending_scan_id,
                    "status": "already_pending",
                }
            scan_id = f"scan-{uuid.uuid4().hex[:8]}"
            self._pending_scan_id = scan_id

        fut = self.submit("sync", self._execute_sync_worker, progress_callback=progress_callback)
        with self._state_lock:
            self._pending_sync_future = fut
        return {
            "scan_id": scan_id,
            "status": "accepted",
        }

    def sync_now(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> SyncDelta:
        """Run or await a CocoIndex incremental sync and return the applied SyncDelta."""
        with self._state_lock:
            existing_fut = (
                self._pending_sync_future
                if self._pending_sync_future is not None and not self._pending_sync_future.done()
                else None
            )
        if existing_fut is not None:
            return existing_fut.result()

        fut = self.submit("sync", self._execute_sync_worker, progress_callback=progress_callback)
        with self._state_lock:
            self._pending_sync_future = fut
        return fut.result()

    def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Future[list[HydratedPassage]]:
        """Validate query and enqueue hybrid PPR + 1-indexed disk line hydration."""
        cleaned = validate_query_input(query)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            raise YawRuntimeError("INVALID_INPUT", "top_k must be an integer between 1 and 20.")

        def _run_retrieve() -> list[HydratedPassage]:
            if self.engine is None:
                raise YawRuntimeError("NOT_READY", "Graph engine is not initialized.")
            pred = self.engine.ensemble(question=cleaned, retrieve_only=True, k=top_k)
            cids = tuple(str(c) for c in getattr(pred, "retrieved_chunk_ids", ()))
            scores = tuple(float(s) for s in getattr(pred, "retrieved_scores", ()))
            hits = self.engine.hydrate_ranked_chunks(list(zip(cids, scores, strict=False)), top_k)
            return [
                HydratedPassage(pointer=ptr, text=text, score=score, stale_on_disk=False) for ptr, score, text in hits
            ]

        return self.submit("retrieve", _run_retrieve, progress_callback=progress_callback)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[HydratedPassage]:
        """Synchronously retrieve top-k hydrated passages via the bounded worker queue."""
        return self.retrieve_async(query, top_k=top_k, progress_callback=progress_callback).result()

    def rag_qa_async(
        self,
        question: str,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Future[dict[str, Any]]:
        """Validate question and enqueue multi-hop HippoRAG 2 QA on the worker queue."""
        cleaned = validate_query_input(question)
        if self.stats().chunks == 0:
            raise YawRuntimeError("EMPTY_INDEX", "Cannot execute rag_qa on an empty graph index.")

        def _run_qa() -> dict[str, Any]:
            if self.engine is None:
                raise YawRuntimeError("NOT_READY", "Graph engine is not initialized.")
            if self.engine.get_stats().chunks == 0:
                raise YawRuntimeError("EMPTY_INDEX", "Cannot execute rag_qa on an empty graph index.")
            pred = self.engine.ensemble(question=cleaned)
            ans = str(getattr(pred, "answer", "")).strip()
            reasoning = str(getattr(pred, "reasoning", "")).strip()
            top_n = min(5, self.config.retrieval_top_k)
            passages_text = tuple(getattr(pred, "retrieved_passages", ()))[:top_n]
            scores = tuple(getattr(pred, "retrieved_scores", ()))[:top_n]
            pids = tuple(getattr(pred, "retrieved_chunk_ids", ()))[:top_n]
            ptrs_map = self.engine.resolve_chunk_pointers(pids)
            passages = [
                {
                    "rank": idx + 1,
                    "text": str(text),
                    "score": float(scores[idx]) if idx < len(scores) else 0.0,
                    "chunk_id": str(cid),
                    "stale_on_disk": False,
                    "source": {
                        "rel_posix_path": ptrs_map[cid].rel_posix_path if cid in ptrs_map else "",
                        "start_line": ptrs_map[cid].start_line if cid in ptrs_map else 0,
                        "end_line": ptrs_map[cid].end_line if cid in ptrs_map else 0,
                        "title": ptrs_map[cid].title if cid in ptrs_map else "",
                    },
                }
                for idx, (cid, text) in enumerate(zip(pids, passages_text, strict=False))
            ]
            return {
                "question": cleaned,
                "answer": ans,
                "reasoning": reasoning,
                "evidence": passages,
                "last_sync_at": self.status().last_sync_at,
            }

        return self.submit("rag_qa", _run_qa, progress_callback=progress_callback)

    def rag_qa(
        self,
        question: str,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Synchronously execute multi-hop HippoRAG 2 QA via the bounded worker queue."""
        return self.rag_qa_async(question, progress_callback=progress_callback).result()

    def rebuild(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> IndexStats:
        """Wipe yaw.db and CocoIndex state on the worker thread and perform a full re-sync."""

        def _run_rebuild() -> IndexStats:
            self._close_components()
            if self.config.db_path is not None:
                db_p = Path(self.config.db_path)
                for suffix in ("", "-wal", "-shm"):
                    target = Path(str(db_p) + suffix)
                    with contextlib.suppress(OSError):
                        target.unlink(missing_ok=True)
            if self.config.state_dir is not None:
                state_p = Path(self.config.state_dir)
                if state_p.exists():
                    shutil.rmtree(state_p, ignore_errors=True)

            self._open_components()
            self._execute_sync_worker()
            return self.stats()

        return self.submit("rebuild", _run_rebuild, progress_callback=progress_callback).result()

    def optimize(
        self,
        trainset: Sequence[Any],
        valset: Sequence[Any] | None = None,
        *,
        mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid",
        save_path: Path | str | None = None,
        reflection_lm: Any = None,
        reflection_worker_lm: Any = None,
        auto: Literal["light", "medium", "heavy"] | None = None,
        num_candidates: int | None = None,
        num_trials: int | None = None,
        sleep_seconds: float | None = None,
        max_bootstrapped_demos: int | None = None,
        max_labeled_demos: int | None = None,
        corpus_docs: Sequence[str] | None = None,
        dataset_bundles: Sequence[DatasetBundle] | None = None,
        progress_callback: Callable[[str], None] | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> Path:
        """Compile DSPy prompts and surface bounds on the worker thread and save artifact."""
        target_path = Path(save_path or self.config.compiled_program_path or "compiled_program.json").resolve()
        resolved_auto = auto or self.config.optimize_auto or "light"

        def _run_optimize() -> Path:
            if self.engine is None:
                raise YawRuntimeError("NOT_READY", "Graph engine is not initialized.")

            self.validate_optimization_run(target_path, trainset, valset, corpus_docs, mode, run_options)

            def _progress(msg: str) -> None:
                self._update_state(active_op=msg)
                if progress_callback is not None:
                    progress_callback(msg)

            opt_chunks = []
            if dataset_bundles:
                _progress("indexing: loading complete persistent corpus snapshot")
                assert self.config.state_dir is not None
                opt_chunks = []
                roots = {bundle.root.parent for bundle in dataset_bundles}
                if len(roots) != 1:
                    raise ValueError("Dataset bundles must share one dataset directory")
                self.engine.workspace_root = next(iter(roots))
                for bundle in dataset_bundles:
                    opt_chunks.extend(
                        snapshot_dataset_bundle(
                            bundle,
                            Path(self.config.state_dir) / "optimize_coco" / bundle.dataset,
                            chunk_size=self.config.chunk_size,
                            chunk_overlap=self.config.chunk_overlap,
                            min_chunk_size=self.config.min_chunk_size,
                        )
                    )
                new_stats = self.engine.get_stats(self.config.config_fingerprint())
                with self._state_lock:
                    self._stats = new_stats

            eff_ref_lm = reflection_lm
            eff_worker_lm = reflection_worker_lm
            if eff_ref_lm is None and self._custom_program is None:
                _progress("optimizing: configuring reflection LM pair")
                eff_ref_lm, eff_worker_lm = create_dspy_rlm_pair(
                    orchestrator_model=self.config.qa_model,
                    sub_model=self.config.extraction_model,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    provider=self.config.provider,
                    reflection=True,
                    context_window_tokens=self.config.context_window_tokens,
                    max_tokens=self.config.max_output_tokens,
                )

            _progress(f"optimizing: preparing {mode} (auto={resolved_auto}); validation indexing precedes trials")
            eff_sleep = (
                sleep_seconds if sleep_seconds is not None else getattr(self.config, "optimize_sleep_seconds", 0.0)
            )
            self.engine.evolve_prompts(
                trainset=trainset,
                valset=valset,
                mode=mode,
                save_path=target_path,
                reflection_lm=eff_ref_lm,
                reflection_worker_lm=eff_worker_lm,
                auto=resolved_auto,
                num_candidates=num_candidates,
                num_trials=num_trials,
                progress_callback=_progress,
                opt_chunks=opt_chunks,
                sleep_seconds=eff_sleep,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
            final_stats = self.engine.get_stats(self.config.config_fingerprint())
            with self._state_lock:
                self._stats = final_stats
            return target_path

        return self.submit("optimize", _run_optimize, progress_callback=progress_callback).result()

    def validate_optimization_run(
        self,
        target_path: Path,
        trainset: Sequence[Any],
        valset: Sequence[Any] | None,
        corpus_docs: Sequence[str] | None,
        mode: str,
        run_options: dict[str, Any] | None = None,
    ) -> None:
        """Bind checkpoints to their ordered data, inference settings and scoring code.

        Called on the writer thread, before replacing corpus sources or indexing.
        Budgets may be increased when continuing an existing search.
        """
        assert self.engine is not None
        settings = {
            k: v
            for k, v in asdict(self.config).items()
            if not k.endswith(("_path", "_dir", "_root"))
            and not k.startswith(("optimize_", "http_"))
            and k not in {"api_key", "poll_interval_seconds", "queue_maxsize", "indexing_overlap"}
        }
        settings.update(
            max_bootstrapped_demos=self.config.optimize_max_bootstrapped_demos,
            max_labeled_demos=self.config.optimize_max_labeled_demos,
        )
        daemon_dir = Path(__file__).parent
        source_paths = sorted((daemon_dir / "graph").rglob("*.py"))
        source_paths.append(daemon_dir / "ingestion" / "corpus.py")
        identity = {
            "version": 1,
            "metric": "task-quality-arithmetic-mean-v2",
            "mode": mode,
            "settings": settings,
            "data_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "train": [ex.toDict() for ex in trainset],
                        "val": [ex.toDict() for ex in (valset or trainset)],
                        "corpus": list(corpus_docs or ()),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "code_sha256": hashlib.sha256(
                b"".join(
                    p.relative_to(daemon_dir).as_posix().encode("utf-8") + b"\0" + p.read_bytes() for p in source_paths
                )
            ).hexdigest(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("dspy", "gepa", "cocoindex", "sqlite-vec", "scipy")
            },
        }
        # TOML tuples become JSON lists on disk; compare the canonical form.
        identity = json.loads(json.dumps(identity))
        conn = self.engine.conn
        row = conn.execute("SELECT value FROM meta WHERE key = 'optimization_run'").fetchone()
        if row is not None:
            saved = json.loads(row[0])
            saved_id = saved.get("identity")
            invariant_keys = ("version", "mode", "settings", "data_sha256", "packages")
            if (
                saved.get("output_path") != str(target_path)
                or not isinstance(saved_id, dict)
                or any(saved_id.get(k) != identity[k] for k in invariant_keys)
            ):
                raise YawRuntimeError(
                    "INVALID_INPUT",
                    "Cannot resume: optimization data, settings, code, packages or output path "
                    "changed. Start a fresh run without --resume/--db-path; "
                    "old state is preserved.",
                )
            metric_upgraded = saved_id.get("metric") != identity["metric"]
            code_upgraded = saved_id.get("code_sha256") != identity["code_sha256"]
            if metric_upgraded:
                checkpoint = target_path.parent / f"{target_path.stem}_telemetry" / "gepa_state.bin"
                checkpoint.unlink(missing_ok=True)
            if metric_upgraded or code_upgraded or run_options is not None:
                saved["identity"] = identity
                if run_options is not None:
                    saved["cli_options"] = run_options
                with self.engine.managed_conn.transaction() as tx:
                    if metric_upgraded or code_upgraded:
                        tx.execute("DELETE FROM meta WHERE key LIKE 'simba_worker:%'")
                    if metric_upgraded:
                        tx.execute("DELETE FROM candidate_stage_cache WHERE stage = 'simba_task_score'")
                    tx.execute(
                        "UPDATE meta SET value = ? WHERE key = 'optimization_run'",
                        (json.dumps(saved),),
                    )
            return
        checkpoint = target_path.parent / f"{target_path.stem}_telemetry" / "gepa_state.bin"
        if checkpoint.exists():
            raise YawRuntimeError(
                "INVALID_INPUT",
                "Cannot verify this legacy GEPA checkpoint: no run identity was recorded. "
                "Start a fresh run; the existing checkpoint is preserved.",
            )
        with self.engine.managed_conn.transaction() as tx:
            tx.execute(
                "INSERT INTO meta(key, value) VALUES ('optimization_run', ?)",
                (
                    json.dumps(
                        {
                            "output_path": str(target_path),
                            "identity": identity,
                            "cli_options": run_options or {},
                        }
                    ),
                ),
            )

    def reload_program(self, program_path: Path | str) -> dict[str, Any]:
        """Hot-reload a compiled DSPy program into the running engine in-memory."""
        p = Path(program_path).resolve()
        if not p.exists() or not p.is_file():
            raise YawRuntimeError("INVALID_INPUT", f"Compiled program file not found: {p}")

        def _run_reload() -> dict[str, Any]:
            if self.engine is None:
                raise YawRuntimeError("NOT_READY", "Graph engine is not initialized.")
            self.engine.program.load(str(p))
            self.engine.refresh_program_bindings()
            self._update_state()
            return {
                "reloaded": True,
                "program_path": str(p),
                "pid": os.getpid(),
                "timestamp": time.time(),
            }

        return self.submit("reload_program", _run_reload).result()

    def benchmark(
        self,
        datasets: Sequence[str] | str | None = None,
        *,
        gate_path: Path | str | None = None,
        max_queries: int | None = None,
        index_queries: int | None = None,
        query_offset: int = 0,
        full_corpus: bool = False,
        mode: Literal["eval", "retrieval"] = "eval",
        compiled_program: Path | str | None = None,
        compiled_program_path: Path | str | None = None,
        output_path: Path | str | None = None,
        fresh: bool = False,
        index_root: Path | str | None = None,
        target: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run isolated per-dataset HippoRAG 2 + DPR benchmark evaluation and optional gate verification."""
        gate_spec: dict[str, Any] | None = None
        if gate_path is not None:
            gp = Path(gate_path).resolve()
            if not gp.exists() or not gp.is_file():
                raise YawRuntimeError("INVALID_INPUT", f"Benchmark gate file not found: {gp}")
            loaded_gate = json.loads(gp.read_text(encoding="utf-8"))
            if not isinstance(loaded_gate, dict):
                raise YawRuntimeError("INVALID_INPUT", f"Invalid benchmark gate JSON in {gp}")
            gate_spec = loaded_gate

        effective_mode: Literal["eval", "retrieval"] = (
            "retrieval"
            if str(mode or (gate_spec.get("mode", "eval") if gate_spec else "eval")).lower() == "retrieval"
            else "eval"
        )

        raw_datasets: list[str] = []
        if datasets is not None:
            if isinstance(datasets, str):
                raw_datasets = [p.strip() for p in datasets.split(",") if p.strip()]
            else:
                for item in datasets:
                    raw_datasets.extend(p.strip() for p in str(item).split(",") if p.strip())
        if not raw_datasets:
            if gate_spec and isinstance(gate_spec.get("datasets"), dict) and gate_spec["datasets"]:
                raw_datasets = list(gate_spec["datasets"].keys())
            else:
                raw_datasets = list(DEFAULT_PARITY_DATASETS)

        if full_corpus:
            eff_max_q: int | None = None
            eff_idx_q: int | None = None
        else:
            eff_max_q = (
                max_queries
                if max_queries is not None
                else (int(gate_spec["queries_per_dataset"]) if gate_spec and "queries_per_dataset" in gate_spec else 25)
            )
            eff_idx_q = (
                index_queries
                if index_queries is not None
                else (
                    int(gate_spec["index_queries"])
                    if gate_spec and "index_queries" in gate_spec
                    else max(eff_max_q, 35)
                )
            )

        dataset_reports: dict[str, dict[str, Any]] = {}
        for ds_arg in raw_datasets:
            bundle = resolve_dataset_bundle(ds_arg, workspace_root=self.config.workspace_root)
            canonical_ds = bundle.dataset
            samples, document_rows = bundle.read_inputs()
            corpus = [
                dict(row, idx=entry.document_id) for entry, row in zip(bundle.documents, document_rows, strict=True)
            ]

            docs, queries, gold_docs, gold_answers = build_benchmark_slice(
                canonical_ds,
                samples,
                corpus,
                max_queries=eff_max_q,
                query_offset=query_offset,
                full_corpus=full_corpus,
                index_queries=eff_idx_q,
            )

            slice_hash = hashlib.sha256(
                json.dumps(
                    {
                        "dataset": canonical_ds,
                        "docs": len(docs),
                        "first_doc": docs[0][:64] if docs else "",
                        "queries": len(queries),
                        "offset": query_offset,
                        "full_corpus": full_corpus,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:8]

            bench_cfg = self.config.for_benchmark(
                canonical_ds,
                slice_hash,
                index_root=Path(index_root) if index_root else None,
                fresh=fresh,
            )
            assert bench_cfg.corpus_root is not None
            assert bench_cfg.state_dir is not None
            assert bench_cfg.db_path is not None
            assert bench_cfg.max_context_chars is not None

            if progress_callback is not None:
                progress_callback(f"benchmarking {canonical_ds}: indexing {len(docs)} docs")

            effective_corpus_root = bundle.root.parent
            chunks = snapshot_dataset_bundle(
                bundle,
                bench_cfg.state_dir,
                document_ids=docs,
                chunk_size=bench_cfg.chunk_size,
                chunk_overlap=bench_cfg.chunk_overlap,
                min_chunk_size=1,
            )

            prog = self._custom_program if self._custom_program is not None else HippoRAGProgram.from_config(bench_cfg)
            comp_p = compiled_program or compiled_program_path or bench_cfg.compiled_program_path
            if comp_p and Path(comp_p).exists() and hasattr(prog, "load"):
                prog.load(str(comp_p))

            emb = (
                self._custom_embedder
                if self._custom_embedder is not None
                else DSPyEmbeddingAdapter.from_config(bench_cfg)
            )

            bench_engine = YawGraphEngine(
                db_path=bench_cfg.db_path,
                workspace_root=effective_corpus_root,
                program=prog,
                embedder=emb,
                linking_top_k=bench_cfg.linking_top_k,
                retrieval_top_k=bench_cfg.retrieval_top_k,
                damping=bench_cfg.damping,
                synonymy_threshold=bench_cfg.synonymy_threshold,
                synonymy_top_k=bench_cfg.synonymy_top_k,
                embed_batch_size=bench_cfg.embed_batch_size,
                max_context_chars=bench_cfg.max_context_chars,
                max_ast_expansion_hops=bench_cfg.max_ast_expansion_hops,
                ast_context_hops=bench_cfg.ast_context_hops,
            )
            try:
                existing_cids = {str(r[0]) for r in bench_engine.conn.execute("SELECT chunk_id FROM chunks").fetchall()}
                unindexed = [(ptr, txt) for ptr, txt in chunks if ptr.chunk_id not in existing_cids]
                index_batches = (
                    bench_engine.index_chunks(unindexed)
                    if unindexed
                    else int(bench_engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                )

                if progress_callback is not None:
                    progress_callback(f"benchmarking {canonical_ds}: evaluating {len(queries)} queries")

                top_k = max(bench_cfg.retrieval_top_k, 5)
                hippo_r1, hippo_r2, hippo_r5 = 0.0, 0.0, 0.0
                dpr_r1, dpr_r2, dpr_r5 = 0.0, 0.0, 0.0
                hippo_em, hippo_f1 = 0.0, 0.0
                dpr_em, dpr_f1 = 0.0, 0.0
                seeds_count = 0
                bench_records: list[dict[str, Any]] = []

                for q_idx, q in enumerate(queries):
                    g_docs = gold_docs[q_idx] if q_idx < len(gold_docs) else []
                    g_ans = gold_answers[q_idx] if q_idx < len(gold_answers) else []
                    ens_pred = bench_engine.ensemble(
                        question=q,
                        retrieve_only=(effective_mode != "eval"),
                        k=top_k,
                    )
                    ens_cids = tuple(str(c) for c in getattr(ens_pred, "retrieved_chunk_ids", ()))
                    ens_scores = tuple(float(s) for s in getattr(ens_pred, "retrieved_scores", ()))
                    hits = bench_engine.hydrate_ranked_chunks(
                        list(zip(ens_cids, ens_scores, strict=False)),
                        top_k,
                    )
                    seeds = list(getattr(bench_engine, "_last_query_seeds", ()))
                    dpr_hits = bench_engine.retrieve_dpr(q, top_k=top_k)
                    if seeds:
                        seeds_count += 1
                    ret_texts = [t for _, _, t in hits]
                    dpr_texts = [t for _, _, t in dpr_hits]

                    h_rec = compute_retrieval_recalls(g_docs, ret_texts, k_list=(1, 2, 5))
                    d_rec = compute_retrieval_recalls(g_docs, dpr_texts, k_list=(1, 2, 5))
                    hippo_r1 += h_rec.get(1, 0.0)
                    hippo_r2 += h_rec.get(2, 0.0)
                    hippo_r5 += h_rec.get(5, 0.0)
                    dpr_r1 += d_rec.get(1, 0.0)
                    dpr_r2 += d_rec.get(2, 0.0)
                    dpr_r5 += d_rec.get(5, 0.0)

                    pred_ans = ""
                    dpr_ans: str | None = None
                    dpr_qa_metrics: dict[str, float] | None = None
                    if effective_mode == "eval":
                        pred_ans = str(getattr(ens_pred, "answer", "") or "").strip()
                        em_val, f1_val = compute_qa_em_f1(pred_ans, g_ans)
                        hippo_em += em_val
                        hippo_f1 += f1_val

                        dpr_ctx_str = bench_engine.format_qa_context(dpr_hits[:5])
                        dpr_pred = (
                            prog.answer_question(context=dpr_ctx_str, question=q)
                            if hasattr(prog, "answer_question")
                            else prog(question=q)
                        )
                        dpr_ans = str(getattr(dpr_pred, "answer", dpr_pred) or "").strip()
                        d_em_val, d_f1_val = compute_qa_em_f1(dpr_ans, g_ans)
                        dpr_em += d_em_val
                        dpr_f1 += d_f1_val
                        dpr_qa_metrics = {"ExactMatch": d_em_val, "F1": d_f1_val}

                    bench_records.append(
                        {
                            "question": q,
                            "graph_seeds": seeds,
                            "answer": pred_ans,
                            "dpr_answer": dpr_ans,
                            "dpr_qa_metrics": dpr_qa_metrics,
                        }
                    )

                n_q = max(len(queries), 1)
                fallback_count = len(queries) - seeds_count
                num_facts = int(bench_engine.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
                graph_info = {
                    "num_passage_nodes": int(bench_engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
                    "num_entity_nodes": int(bench_engine.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]),
                    "num_fact_nodes": num_facts,
                    "num_extracted_triples": num_facts,
                    "num_edges": int(bench_engine.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
                }
                expected_doc_count = (
                    int(graph_info["num_passage_nodes"])
                    if int(graph_info["num_passage_nodes"]) >= len(docs) and len(docs) > 0
                    else len(docs)
                )
                integrity_errors = check_benchmark_integrity(
                    bench_records,
                    graph_info,
                    expected_doc_count=expected_doc_count,
                    include_qa=(effective_mode == "eval"),
                )

                dataset_reports[canonical_ds] = {
                    "dataset": canonical_ds,
                    "mode": effective_mode,
                    "queries": len(queries),
                    "docs": len(docs),
                    "index_dir": str(bench_cfg.workspace_root),
                    "index_batches": index_batches,
                    "graph": graph_info,
                    "integrity": {"passed": len(integrity_errors) == 0, "errors": integrity_errors},
                    "evaluation_metrics": {
                        "hipporag": {
                            "retrieval": {
                                "Recall@1": round(hippo_r1 / n_q, 4),
                                "Recall@2": round(hippo_r2 / n_q, 4),
                                "Recall@5": round(hippo_r5 / n_q, 4),
                            },
                            "qa": (
                                {
                                    "ExactMatch": round(hippo_em / n_q, 4),
                                    "F1": round(hippo_f1 / n_q, 4),
                                }
                                if effective_mode == "eval"
                                else {}
                            ),
                        },
                        "standard_dpr": {
                            "retrieval": {
                                "Recall@1": round(dpr_r1 / n_q, 4),
                                "Recall@2": round(dpr_r2 / n_q, 4),
                                "Recall@5": round(dpr_r5 / n_q, 4),
                            },
                            "qa": (
                                {
                                    "ExactMatch": round(dpr_em / n_q, 4),
                                    "F1": round(dpr_f1 / n_q, 4),
                                }
                                if effective_mode == "eval"
                                else {}
                            ),
                        },
                        "recognition_memory": {
                            "queries_with_graph_seeds": seeds_count,
                            "dpr_fallback_count": fallback_count,
                            "dpr_fallback_rate": round(fallback_count / n_q, 4),
                        },
                    },
                }
            finally:
                bench_engine.close()

        effective_gate_spec = gate_spec or {
            "mode": effective_mode,
            "per_dataset_tolerance": 0.0,
            "require_mean_recall5_parity": True,
            "require_mean_f1_parity": effective_mode == "eval",
            "require_mean_em_parity": effective_mode == "eval",
            "datasets": {
                ds_key: {
                    "monkeypatched_hipporag": MONKEYPATCHED_HIPPORAG_BASELINES.get(ds_key, {}),
                    "paper_70b": PAPER_BASELINES.get(ds_key, {}),
                }
                for ds_key in dataset_reports
            },
        }
        gate_eval = evaluate_benchmark_gate(
            dataset_reports,
            effective_gate_spec,
            mode=effective_mode,
            target=target,
        )

        result_payload: dict[str, Any] = {
            "status": "passed" if gate_eval["overall_gate_passed"] else "failed",
            "mode": effective_mode,
            "overall_gate_passed": gate_eval["overall_gate_passed"],
            "overall_gate_status": gate_eval["overall_gate_status"],
            "datasets": dataset_reports,
            "gate": gate_eval,
        }
        if output_path is not None:
            write_text_atomic(Path(output_path).resolve(), json.dumps(result_payload, indent=2))
        return result_payload


__all__ = [
    "YawRuntime",
    "is_file_locked",
    "validate_query_input",
]
