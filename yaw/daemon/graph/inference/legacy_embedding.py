"""Legacy RetrievalEmbedding adapter for pinned upstream HippoRAG baseline."""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import DefaultHttpxClient, OpenAI

from .embedding import (
    PROFILES,
    QUERY_INSTRUCTIONS,
    TITLE_LINE_RE,
    batch_texts_by_budget,
    is_batch_overflow_error,
    pool_subchunk_embeddings,
    split_text_gracefully,
)
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

        batch_size = kwargs.get("batch_size", self.embedding_config.encode_params["batch_size"])
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("OpenAI embedding batch_size must be a positive integer.")

        normalize = kwargs.get("norm", True)
        max_chunk_chars = 3000
        max_batch_chars = 3500

        # 1. Gracefully divide texts exceeding max_chunk_chars and format with profile
        text_piece_map: list[list[int]] = []
        flat_pieces: list[str] = []
        piece_weights: list[int] = []

        is_query = bool(instruction)
        for t in texts:
            if len(t) <= max_chunk_chars:
                formatted = self.profile.format(t, query=is_query, instruction=instruction)
                text_piece_map.append([len(flat_pieces)])
                flat_pieces.append(formatted)
                piece_weights.append(max(1, len(t)))
            else:
                title = ""
                body = t
                if not is_query and self.profile.parse_document_title and "\n" in t:
                    first_line, rest = t.split("\n", 1)
                    match = TITLE_LINE_RE.match(first_line.strip())
                    if match and match.group(1).strip() and rest.strip():
                        title = match.group(1).strip()
                        body = rest.strip()

                sub_chunks = split_text_gracefully(body, max_chars=max_chunk_chars)
                indices: list[int] = []
                for sc in sub_chunks:
                    indices.append(len(flat_pieces))
                    if is_query:
                        formatted = self.profile.format(sc, query=True, instruction=instruction)
                    elif title:
                        formatted = f"title: {title} | text: {sc}"
                    else:
                        formatted = self.profile.format(sc, query=False)
                    flat_pieces.append(formatted)
                    piece_weights.append(max(1, len(sc)))
                text_piece_map.append(indices)

        # 2. Partition flat_pieces into micro-batches bounded by character budget and count
        micro_batches = batch_texts_by_budget(
            flat_pieces,
            max_batch_chars=max_batch_chars,
            max_batch_size=batch_size,
        )

        # 3. Execute micro-batches with encode()
        flat_vectors: list[np.ndarray | None] = [None] * len(flat_pieces)
        usage_totals = {"prompt_tokens": 0, "total_tokens": 0}
        for mb in micro_batches:
            mb_texts = [flat_pieces[i] for i in mb]
            vecs = self.encode(mb_texts)
            if self.last_usage is not None:
                usage_totals["prompt_tokens"] += self.last_usage.get("prompt_tokens", 0)
                usage_totals["total_tokens"] += self.last_usage.get("total_tokens", 0)
            for local_idx, global_idx in enumerate(mb):
                flat_vectors[global_idx] = vecs[local_idx]

        self.last_usage = usage_totals

        # 4. Pool sub-chunks back into document embeddings via length-weighted average
        doc_vectors: list[np.ndarray] = []
        for indices in text_piece_map:
            sub_vecs = np.asarray(
                [flat_vectors[i] for i in indices if flat_vectors[i] is not None],
                dtype=np.float32,
            )
            weights = [piece_weights[i] for i in indices]
            doc_vec = pool_subchunk_embeddings(sub_vecs, weights, norm=normalize)
            doc_vectors.append(doc_vec)

        results = np.stack(doc_vectors)
        return self._normalize_embeddings(results, normalize=normalize)

    def encode(self, texts):
        """Send prepared inputs unchanged; reject malformed/partial responses."""
        if not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError("Expected a nonempty list of prepared text strings.")
        self.last_usage = {"usage_unknown": True, "complete": False}
        try:
            response = self.client.embeddings.create(
                input=texts, model=self.request_model_name, encoding_format="float",
            )
        except Exception as exc:
            if not is_batch_overflow_error(exc):
                raise
            if len(texts) > 1:
                mid = len(texts) // 2
                left = self.encode(texts[:mid])
                left_usage = self.last_usage or {}
                right = self.encode(texts[mid:])
                right_usage = self.last_usage or {}
                self.last_usage = {
                    "prompt_tokens": left_usage.get("prompt_tokens", 0) + right_usage.get("prompt_tokens", 0),
                    "total_tokens": left_usage.get("total_tokens", 0) + right_usage.get("total_tokens", 0),
                }
                return np.concatenate([left, right], axis=0)

            # Single text overflow: divide gracefully into 2 sub-chunks and pool without truncation
            single = texts[0]
            sub_chunks = split_text_gracefully(single, max_chars=max(256, len(single) // 2))
            if len(sub_chunks) <= 1:
                raise
            parts = []
            usage_totals = {"prompt_tokens": 0, "total_tokens": 0}
            for sc in sub_chunks:
                parts.append(self.encode([sc]))
                if self.last_usage:
                    usage_totals["prompt_tokens"] += self.last_usage.get("prompt_tokens", 0)
                    usage_totals["total_tokens"] += self.last_usage.get("total_tokens", 0)
            self.last_usage = usage_totals
            sub_vecs = np.concatenate(parts, axis=0)
            weights = [max(1, len(s)) for s in sub_chunks]
            pooled = pool_subchunk_embeddings(sub_vecs, weights, norm=True)
            return pooled.reshape(1, -1)

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
