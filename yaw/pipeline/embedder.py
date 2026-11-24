"""DSPy batch embedder and CocoIndex batching adapter."""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import Callable
from typing import Any

import cocoindex as coco
import dspy
import numpy as np
import openai
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError


class DspyBatchEmbedder:
    """Native DSPy batching with sync HTTP gate and persistent pooled async acall."""

    def __init__(self, cfg: YawConfig) -> None:
        self._model_provider = cfg.provider
        self._model_name = cfg.embedding_model
        self._base_url = cfg.base_url
        self._api_key = cfg.api_key or "yaw"
        self._batch_size = max(1, int(cfg.embed_batch_size))
        model_str = f"{self._model_provider}/{self._model_name}"
        self._embed_model = model_str.split("/", 1)[-1]
        kwargs: dict[str, Any] = {
            "api_base": self._base_url,
            "api_key": self._api_key,
            "max_retries": 0,
        }
        self._embedder = dspy.Embedder(
            model_str,
            batch_size=self._batch_size,
            caching=False,
            **kwargs,
        )
        self._sem = threading.BoundedSemaphore(cfg.embed_max_in_flight)
        self._async_client: openai.AsyncOpenAI | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

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

    def _get_async_client(self) -> openai.AsyncOpenAI:
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop or self._async_loop.is_closed():
            self._async_client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                max_retries=0,
            )
            self._async_loop = loop
        return self._async_client

    @staticmethod
    def _decode_embeddings_response(resp: Any, expected_count: int) -> list[Any]:
        vectors: list[Any] = [
            np.frombuffer(base64.b64decode(item.embedding), dtype="<f4").tolist()
            if isinstance(item.embedding, str)
            else list(item.embedding)
            for item in sorted(resp.data, key=lambda d: d.index)
        ]
        if len(vectors) != expected_count:
            raise YawRuntimeError(
                "MODEL_ERROR",
                f"Expected {expected_count} embedding vectors, got {len(vectors)}.",
            )
        return vectors

    def __call__(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._sem:
            raw = self._embedder(texts)
        return self._normalize_batch(raw)

    async def acall(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts asynchronously via pooled AsyncOpenAI (or dspy.Embedder fallback)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model_provider != "openai":
            raw = await self._embedder.acall(texts)
            return self._normalize_batch(raw)
        client = self._get_async_client()
        vectors: list[Any] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = await client.embeddings.create(
                model=self._embed_model,
                input=batch,
                encoding_format="float",
            )
            vectors.extend(self._decode_embeddings_response(resp, len(batch)))
        return self._normalize_batch(vectors)

    async def aclose(self) -> None:
        """Close the pooled AsyncOpenAI client if bound to the active running loop."""
        client = self._async_client
        bound_loop = self._async_loop
        self._async_client = None
        self._async_loop = None
        if client is None:
            return
        running_loop = asyncio.get_running_loop()
        if bound_loop is None or (bound_loop is running_loop and not bound_loop.is_closed()):
            await client.close()


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
