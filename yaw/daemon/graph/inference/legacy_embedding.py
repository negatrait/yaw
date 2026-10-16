"""Legacy RetrievalEmbedding adapter for pinned upstream HippoRAG baseline."""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import DefaultHttpxClient, OpenAI

from .embedding import PROFILES, QUERY_INSTRUCTIONS
from .models import DEFAULT_BASE_URL


@dataclass
class _EmbeddingConfig:
    embedding_model_name: str
    norm: bool = True
    encode_params: dict[str, Any] = field(default_factory=lambda: {"batch_size": 32})


class RetrievalEmbedding:
    """Self-contained OpenAI-compatible embedding client with task prefixes and strict validation.

    Preserves full interface compatibility for the pinned upstream `hipporag` baseline in `benchmark.py`.
    """

    __module__ = "yaw.embedding"

    query_instruction_mode = "shared"

    def __init__(self, global_config, profile=None):
        self.global_config = global_config
        self.profile = profile or PROFILES[global_config.embedding_model_name]
        if global_config.embedding_model_name != self.profile.model:
            raise ValueError("Embedding config model must match the selected profile.")
        self.embedding_model_name = self.profile.model
        self.request_model_name = self.embedding_model_name
        self.query_instruction_mode = self.profile.query_instruction_mode
        self.last_usage = None
        self._instruction_warning_emitted = False

        batch_size = getattr(global_config, "embedding_batch_size", 32)
        norm_default = getattr(global_config, "embedding_return_as_normalized", True)
        self.embedding_config = _EmbeddingConfig(
            embedding_model_name=self.embedding_model_name,
            norm=norm_default,
            encode_params={"batch_size": batch_size},
        )

        timeout = getattr(global_config, "embedding_request_timeout", 120.0)
        max_retries = getattr(global_config, "max_retry_attempts", 0)
        base_url = getattr(global_config, "embedding_base_url", None) or DEFAULT_BASE_URL
        api_key_env = getattr(global_config, "embedding_api_key_env", None)
        api_key = (os.environ.get(api_key_env) if api_key_env else None) or "lemonade"

        http_client = DefaultHttpxClient(timeout=timeout)
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
                max_retries=max_retries,
            )
        except Exception:
            http_client.close()
            raise

    def _normalize_embeddings(self, embeddings: np.ndarray, normalize: bool | None = None) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if normalize is None:
            normalize = self.embedding_config.norm
        if not isinstance(normalize, bool):
            raise TypeError("Embedding normalization must be a boolean.")
        if not normalize:
            return embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot normalize a zero embedding vector.")
        return embeddings / norms

    def get_query_doc_scores(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        return np.dot(query_vec, doc_vecs.T)

    def batch_encode(self, texts, **kwargs):
        unsupported = set(kwargs) - {"batch_size", "instruction", "norm"}
        if unsupported:
            raise TypeError(f"Unsupported OpenAI embedding options: {', '.join(sorted(unsupported))}.")
        instruction = kwargs.pop("instruction", None)
        if instruction not in (None, "") and instruction not in QUERY_INSTRUCTIONS:
            raise ValueError(f"Unsupported retrieval instruction: {instruction!r}")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("OpenAI embedding input cannot be empty.")
        prepared = [
            self.profile.format(t, query=bool(instruction), instruction=instruction)
            for t in texts
        ]
        batch_size = kwargs.get("batch_size", self.embedding_config.encode_params["batch_size"])
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("OpenAI embedding batch_size must be a positive integer.")

        if len(prepared) <= batch_size:
            results = self.encode(prepared)
        else:
            results = []
            usage_totals = {"prompt_tokens": 0, "total_tokens": 0}
            try:
                for i in range(0, len(prepared), batch_size):
                    batch = prepared[i : i + batch_size]
                    results.append(self.encode(batch))
                    if self.last_usage is not None:
                        usage_totals["prompt_tokens"] += self.last_usage.get("prompt_tokens", 0)
                        usage_totals["total_tokens"] += self.last_usage.get("total_tokens", 0)
                    self.last_usage = {**usage_totals, "complete": False}
            except Exception:
                failed_usage = self.last_usage or {}
                for token_key in ("prompt_tokens", "total_tokens"):
                    token_count = failed_usage.get(token_key)
                    if isinstance(token_count, int):
                        usage_totals[token_key] += token_count
                self.last_usage = {**usage_totals, "complete": False}
                if failed_usage.get("usage_unknown") or not any(
                    key in failed_usage for key in ("prompt_tokens", "total_tokens")
                ):
                    self.last_usage["usage_unknown"] = True
                raise
            results = np.concatenate(results)
            self.last_usage = usage_totals

        return self._normalize_embeddings(results, normalize=kwargs.get("norm"))

    def encode(self, texts):
        """Send prepared inputs unchanged; reject malformed/partial responses."""
        if not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError("Expected a nonempty list of prepared text strings.")
        self.last_usage = {"usage_unknown": True, "complete": False}
        response = self.client.embeddings.create(
            input=texts, model=self.request_model_name, encoding_format="float",
        )
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        total = getattr(usage, "total_tokens", None)
        if (not isinstance(prompt, int) or prompt < 0
                or (total is not None and (not isinstance(total, int) or total < 0))):
            raise ValueError("Embedding response omitted or supplied invalid token usage.")
        self.last_usage = {"prompt_tokens": prompt, "total_tokens": total if total is not None else prompt}
        try:
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(texts))):
                raise ValueError("Embedding response indices do not match the input rows.")
            vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
            if (vectors.shape != (len(texts), self.profile.dimensions)
                    or not np.isfinite(vectors).all()
                    or np.any(np.linalg.norm(vectors, axis=1) == 0)):
                raise ValueError(f"Expected finite, nonzero {self.profile.dimensions}-dimensional vectors.")
        except Exception:
            self.last_usage["complete"] = False
            raise
        return vectors

    def close(self) -> None:
        self.client.close()


__all__ = [
    "RetrievalEmbedding",
]
