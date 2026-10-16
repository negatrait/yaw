"""Single-owner persistent daemon runtime with FileLock, 32-slot bounded queue, and crash-resilient CocoIndex sync."""

import contextlib
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
)
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    HippoRAGProgram,
    create_dspy_rlm_pair,
)
from yaw.daemon.ingestion import CorpusIngestor

logger = logging.getLogger(__name__)
T = TypeVar("T")

_PENDING_DELTA_META_KEY = "pending_sync_delta"
_CONFIG_FP_META_KEY = "config_fingerprint"


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
    added = tuple(
        (ChunkPointer(**ptr_dict), str(text))
        for ptr_dict, text in data.get("added_chunks", ())
    )
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
    _active_instances: ClassVar[dict[Path, "YawRuntime"]] = {}

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
        self._worker_thread: threading.Thread | None = None
        self._poller_thread: threading.Thread | None = None

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
        """Signal an in-process YawRuntime holding lock_path to stop gracefully."""
        with cls._active_lock:
            rt = cls._active_instances.get(lock_path.resolve())
        if rt is not None:
            rt.request_stop()
            return True
        return False

    def request_stop(self) -> None:
        """Signal the runtime stop event and wake the worker loop."""
        self._stop_event.set()
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
            if (
                self.config.compiled_program_path
                and Path(self.config.compiled_program_path).exists()
            ):
                program.load(str(self.config.compiled_program_path))
        else:
            program = HippoRAGProgram.from_config(self.config)

        embedder = (
            self._custom_embedder
            if self._custom_embedder is not None
            else DSPyEmbeddingAdapter.from_config(self.config)
        )

        self.ingestor = CorpusIngestor(
            corpus_root=self.config.corpus_root,
            state_dir=self.config.state_dir,
            included_patterns=self.config.included_patterns,
            excluded_patterns=self.config.excluded_patterns,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
        )
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
        with contextlib.suppress(OSError):
            status_path = Path(self.config.status_path)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = status_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            tmp_path.replace(status_path)

    def start(self, *, sync_on_start: bool = True) -> None:
        """Acquire exclusive yaw.db.lock, open CocoIndex + YawGraphEngine, and start worker thread."""
        with self._state_lock:
            if self._status.lifecycle in ("starting", "ready", "degraded"):
                return

        try:
            self._file_lock.acquire(timeout=0)
        except filelock.Timeout as exc:
            raise YawRuntimeError(
                "BUSY",
                f"Database lock {self.config.lock_path} is already held by another YAW daemon.",
            ) from exc

        assert self.config.lock_path is not None
        with self._active_lock:
            self._active_instances[self.config.lock_path.resolve()] = self

        try:
            if self.config.log_path is not None:
                configure_logging(self.config.log_path)
            self._stop_event.clear()
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
            self._update_state(lifecycle="ready")

            has_pending = (
                self.engine is not None
                and self.engine.get_meta(_PENDING_DELTA_META_KEY) is not None
            )
            if sync_on_start or has_pending:
                self.sync_now()

            if self.auto_poll and self.config.poll_interval_seconds > 0:
                self._poller_thread = threading.Thread(
                    target=self._poll_loop,
                    name="yaw-runtime-poller",
                    daemon=True,
                )
                self._poller_thread.start()
        except Exception:
            if self.config.lock_path is not None:
                with self._active_lock:
                    self._active_instances.pop(self.config.lock_path.resolve(), None)
            self._close_components()
            if self._file_lock.is_locked:
                self._file_lock.release()
            raise

    def stop(self) -> None:
        """Gracefully stop poller and worker threads, close databases, and release yaw.db.lock."""
        if self.config.lock_path is not None:
            with self._active_lock:
                self._active_instances.pop(self.config.lock_path.resolve(), None)

        with self._state_lock:
            if self._status.lifecycle == "stopped":
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

    def __enter__(self) -> "YawRuntime":
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
            if self._status.lifecycle == "starting":
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
        interval = max(0.05, float(self.config.poll_interval_seconds))
        while not self._stop_event.wait(interval):
            with contextlib.suppress(YawRuntimeError):
                self.request_sync()

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
                    "CONSISTENCY_ERROR"
                    if isinstance(exc, sqlite3.DatabaseError)
                    else "MODEL_ERROR"
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

    def _apply_delta(self, delta: SyncDelta) -> None:
        assert self.engine is not None
        if not delta.deleted_chunk_ids and not delta.added_chunks:
            return
        self.engine.set_meta(_PENDING_DELTA_META_KEY, _serialize_sync_delta(delta))
        if delta.deleted_chunk_ids:
            self.engine.delete_chunks(delta.deleted_chunk_ids)
        if delta.added_chunks:
            self.engine.index_chunks(delta.added_chunks)
        self.engine.set_meta(_PENDING_DELTA_META_KEY, None)

    def _execute_sync_worker(self) -> SyncDelta:
        if self.ingestor is None or self.engine is None:
            raise YawRuntimeError("NOT_READY", "Engine or ingestor is not initialized.")

        # 1. Replay any uncommitted pending delta before polling CocoIndex
        raw_pending = self.engine.get_meta(_PENDING_DELTA_META_KEY)
        if raw_pending:
            pending_delta = _deserialize_sync_delta(raw_pending)
            self._apply_delta(pending_delta)
        else:
            pending_delta = None

        # 2. Poll CocoIndex for subsequent corpus changes and apply
        polled_delta = self.ingestor.poll()
        self._apply_delta(polled_delta)

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
            hits = self.engine.retrieve(cleaned, top_k=top_k)
            return [
                HydratedPassage(pointer=ptr, text=text, score=score, stale_on_disk=False)
                for ptr, score, text in hits
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
        return self.retrieve_async(
            query, top_k=top_k, progress_callback=progress_callback
        ).result()

    def rag_qa_async(
        self,
        question: str,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Future[dict[str, Any]]:
        """Validate question and enqueue multi-hop HippoRAG 2 QA on the worker queue."""
        cleaned = validate_query_input(question)
        if self.stats().chunks == 0:
            raise YawRuntimeError(
                "EMPTY_INDEX", "Cannot execute rag_qa on an empty graph index."
            )

        def _run_qa() -> dict[str, Any]:
            if self.engine is None:
                raise YawRuntimeError("NOT_READY", "Graph engine is not initialized.")
            if self.engine.get_stats().chunks == 0:
                raise YawRuntimeError(
                    "EMPTY_INDEX", "Cannot execute rag_qa on an empty graph index."
                )
            hits = self.engine.retrieve(cleaned, top_k=self.config.retrieval_top_k)
            formatted_context = "\n\n".join(
                f"Wikipedia Title: {p.title}\n{text}" if p.title else text
                for p, _, text in hits
            )
            pred = self.engine.program.answer_question(
                context=formatted_context, question=cleaned
            )
            ans = str(getattr(pred, "answer", "")).strip()
            reasoning = str(getattr(pred, "reasoning", "")).strip()
            passages = [
                {
                    "rank": idx + 1,
                    "text": text,
                    "score": score,
                    "chunk_id": ptr.chunk_id,
                    "stale_on_disk": False,
                    "source": {
                        "rel_posix_path": ptr.rel_posix_path,
                        "start_line": ptr.start_line,
                        "end_line": ptr.end_line,
                        "title": ptr.title,
                    },
                }
                for idx, (ptr, score, text) in enumerate(hits[:5])
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
        max_metric_calls: int | None = None,
        corpus_docs: Sequence[str] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        """Compile DSPy prompts and surface bounds on the worker thread and save artifact."""
        target_path = Path(
            save_path or self.config.compiled_program_path or "compiled_program.json"
        ).resolve()

        def _run_optimize() -> Path:
            if self.engine is None:
                raise YawRuntimeError("NOT_READY", "Graph engine is not initialized.")
            if corpus_docs:
                self.engine.index(corpus_docs)
                new_stats = self.engine.get_stats(self.config.config_fingerprint())
                with self._state_lock:
                    self._stats = new_stats

            eff_ref_lm = reflection_lm
            eff_worker_lm = reflection_worker_lm
            if eff_ref_lm is None and self._custom_program is None:
                eff_ref_lm, eff_worker_lm = create_dspy_rlm_pair(
                    orchestrator_model=self.config.qa_model,
                    sub_model=self.config.extraction_model,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    provider=self.config.provider,
                    reflection=True,
                )

            self.engine.evolve_prompts(
                trainset=trainset,
                valset=valset,
                mode=mode,
                save_path=target_path,
                reflection_lm=eff_ref_lm,
                reflection_worker_lm=eff_worker_lm,
                max_metric_calls=max_metric_calls,
            )
            return target_path

        return self.submit("optimize", _run_optimize, progress_callback=progress_callback).result()


__all__ = [
    "YawRuntime",
    "validate_query_input",
]
