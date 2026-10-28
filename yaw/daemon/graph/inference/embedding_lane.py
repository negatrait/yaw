"""One bounded embedding worker; all cache and telemetry publication stays with its owner."""

import importlib
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Any

import numpy as np

from .embedding import DSPyEmbeddingAdapter, EmbeddingResult


class EmbeddingLane:
    def __init__(
        self,
        adapter: DSPyEmbeddingAdapter,
        texts: Sequence[str],
        *,
        batch_size: int,
        lookup: Callable[[str], Any],
        store: Callable[[str, np.ndarray], None],
    ) -> None:
        self.adapter = adapter
        self.texts = tuple(texts)
        self.batch_size = batch_size
        self.store = store
        self.vectors: dict[str, np.ndarray] = {}
        self.missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = lookup(text)
            if cached is None:
                self.missing.append(text)
            else:
                self.vectors[text] = cached
        self.cache_hits = len(self.vectors)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yaw-chunk-embedding")
        self.future: Future[EmbeddingResult] | None = None
        self.pending: tuple[str, ...] = ()
        self.offset = 0
        self.started = time.perf_counter()
        self.join_seconds = 0.0
        self.provider_seconds = 0.0
        self.intervals: list[tuple[float, float]] = []
        self.stop = Event()

    def __enter__(self) -> EmbeddingLane:
        # OpenAI lazily imports mutually dependent resource modules on first use.
        # Resolve them on the owner before chat and embeddings race in a fresh process.
        importlib.import_module("openai.resources")
        self._schedule()
        return self

    def _schedule(self) -> None:
        if self.offset >= len(self.missing):
            return
        self.pending = tuple(self.missing[self.offset : self.offset + self.batch_size])
        self.offset += len(self.pending)
        request = self.adapter.prepare_request(self.pending, batch_size=self.batch_size, stage="chunk_embed")
        self.future = self.pool.submit(self.adapter.execute_request, request, self.stop)

    def drain(self, *, wait: bool = False, schedule: bool = True) -> None:
        """Consume completed work only between OpenIE calls, outside their accounting slices."""
        while self.future is not None and (wait or self.future.done()):
            future, self.future = self.future, None
            started = time.perf_counter()
            try:
                result = future.result()
            finally:
                self.join_seconds += time.perf_counter() - started
            for text, vector in zip(self.pending, result.vectors, strict=True):
                self.store(text, vector)
                self.vectors[text] = vector
            self.adapter.publish_result(result)
            self.provider_seconds += float(result.telemetry["duration_seconds"])
            self.intervals.append((result.telemetry["started_at"], result.telemetry["completed_at"]))
            self.pending = ()
            if schedule:
                self._schedule()

    def finish(self) -> list[np.ndarray]:
        self.drain(wait=True)
        return [self.vectors[text] for text in self.texts]

    def overlap_seconds(self, start: float, end: float) -> float:
        return sum(max(0.0, min(end, b) - max(start, a)) for a, b in self.intervals)

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if exc is not None:
            self.stop.set()
        try:
            # Do not schedule more work after either lane fails. Valid in-flight
            # results remain reusable even when the graph transaction never starts.
            self.drain(wait=True, schedule=False)
        except Exception as drain_error:
            if exc is None:
                raise
            exc.add_note(f"Embedding lane also failed: {drain_error}")
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.vectors.clear()
            self.missing.clear()
