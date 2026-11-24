"""DSPy batch embedder and CocoIndex batching adapter."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import cocoindex as coco
import dspy
import numpy as np
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError


class DspyBatchEmbedder:
    """Native DSPy batching with one shared HTTP admission gate for indexing and queries."""

    def __init__(self, cfg: YawConfig) -> None:
        kwargs: dict[str, Any] = {"api_base": cfg.base_url, "api_key": cfg.api_key or "yaw"}
        self._embedder = dspy.Embedder(
            f"{cfg.provider}/{cfg.embedding_model}",
            batch_size=cfg.embed_batch_size,
            caching=False,
            **kwargs,
        )
        self._sem = threading.BoundedSemaphore(cfg.embed_max_in_flight)

    @staticmethod
    def _normalize_batch(raw: Any) -> NDArray[np.float32]:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 1 or not np.all(np.isfinite(arr)):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced invalid or non-finite embedding array.")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced zero-norm embedding vector.")
        return (arr / norms).astype(np.float32, copy=False)

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._sem:
            raw = self._embedder(texts)
        return self._normalize_batch(raw)


def make_dspy_embed_fn(cfg: YawConfig) -> DspyBatchEmbedder:
    """Build a native DSPy batch embedder preserving each input unchanged."""
    return DspyBatchEmbedder(cfg)


class PipelineEmbedder:
    """CocoIndex-native batching adapter over a batch embed callable."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], NDArray[np.float32]],
        *,
        memo_key: object = None,
        batch_size: int,
        max_in_flight: int,
    ) -> None:
        self._embed_fn = embed_fn
        self._memo_key = memo_key
        self._dim: int | None = None
        self._dim_lock = threading.Lock()
        self.embed = coco.fn.as_async(
            batching=True,
            max_batch_size=max(1, batch_size),
            max_in_flight=max(1, max_in_flight),
        )(self._embed_batch)

    async def invoke(self, texts: list[str]) -> NDArray[np.float32]:
        raw = await asyncio.to_thread(self._embed_fn, texts)
        return np.asarray(raw, dtype=np.float32)

    def __coco_memo_key__(self) -> object:
        return self._memo_key

    async def _embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if not texts:
            return []
        arr = np.ascontiguousarray(await self.invoke(texts), dtype=np.float32)
        if arr.ndim == 1 and len(texts) == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
            if len(texts) > 1:
                raise coco.RetryWithSmallerBatch()
            raise YawRuntimeError("MODEL_ERROR", f"Expected ({len(texts)}, dim) embedding array, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced non-finite embedding values.")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise YawRuntimeError("MODEL_ERROR", "Embedder produced zero-norm embedding vector.")
        unit = (arr / norms).astype(np.float32, copy=False)
        with self._dim_lock:
            if self._dim is None:
                self._dim = int(unit.shape[1])
            elif self._dim != int(unit.shape[1]):
                raise YawRuntimeError("MODEL_ERROR", f"Embedding dimension changed: {self._dim} != {unit.shape[1]}")
        return [unit[i] for i in range(len(texts))]


EMBEDDER = coco.ContextKey[PipelineEmbedder]("yaw/pipeline/embedder", detect_change=True)
