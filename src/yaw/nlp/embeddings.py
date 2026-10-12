"""EmbeddingGemma vector embeddings, task prompt prefixes, and bi-encoder similarity."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yaw.nlp.client import (
    call_ollama_json,
    get_embedding_model,
    get_ollama_host,
    get_relevance_threshold,
)
from yaw.nlp.templates import PROMPT_PREFIXES
from yaw.nlp.utils import is_valid_entity_text
from yaw.utils import format_proposition

__all__ = [
    "PROMPT_PREFIXES",
    "compute_similarity",
    "compute_similarity_matrix",
    "deserialize_embedding",
    "embed_query",
    "embed_texts",
    "embed_texts_sync",
    "format_document",
    "format_prompt",
    "format_proposition",
    "format_space_r_doc",
    "format_space_r_query",
    "format_space_s",
    "is_valid_entity_text",
    "rerank_candidates",
]


def format_document(text: str, title: str | None = None) -> str:
    """Format document text with EmbeddingGemma document prompt."""
    t = (title or "").strip()
    return f"title: {t} | text: {text}" if t else f"title: none | text: {text}"


def format_space_s(name: str, definition: str = "") -> str:
    """Format entity mention for symmetric similarity (Space S)."""
    d = (definition or "").strip()
    n = (name or "").strip()
    return (
        f"task: sentence similarity | query: {n}: {d}"
        if d
        else f"task: sentence similarity | query: {n}"
    )


def format_space_r_doc(text: str, title: str | None = None) -> str:
    """Format document or fact text for asymmetric retrieval (Space R)."""
    return format_document(text, title)


def format_space_r_query(query: str) -> str:
    """Format search query for asymmetric retrieval (Space R)."""
    prefix = "task: search result | query: "
    if query.startswith(prefix):
        return query
    return f"{prefix}{query}"


def format_prompt(text: str, task: str = "Retrieval-query") -> str:
    """Format text with EmbeddingGemma task prefix."""
    prefix = PROMPT_PREFIXES.get(task)
    return f"{prefix}{text}" if prefix else text


def embed_texts_sync(
    texts: list[str],
    host: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
) -> list[NDArray[np.float32]]:
    """Batch embed texts via Ollama /api/embed endpoint, returning unit-normalized float32 arrays."""
    if not texts:
        return []
    target_host = host or get_ollama_host()
    target_model = model or get_embedding_model()

    data = call_ollama_json(
        "/api/embed",
        payload={"model": target_model, "input": texts},
        host=target_host,
        timeout_s=timeout_s,
    )
    raw_embs = data.get("embeddings", [])

    result: list[NDArray[np.float32]] = []
    for raw in raw_embs:
        arr = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr = arr / norm
        result.append(arr)
    return result


async def embed_texts(
    texts: list[str],
    host: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
) -> list[NDArray[np.float32]]:
    """Async wrapper for batch embedding."""
    return await asyncio.to_thread(embed_texts_sync, texts, host, model, timeout_s)


async def embed_query(
    query: str,
    host: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
) -> NDArray[np.float32]:
    """Embed a search query in Space R query space."""
    formatted = format_space_r_query(query)
    vecs = await embed_texts([formatted], host=host, model=model, timeout_s=timeout_s)
    return vecs[0]


def compute_similarity(
    q_vec: NDArray[np.float32], doc_vecs: list[NDArray[np.float32]]
) -> list[float]:
    """Compute cosine similarity between normalized query vector and doc vectors via dot product."""
    if not doc_vecs:
        return []
    mat = np.stack(doc_vecs)
    scores = np.dot(mat, q_vec)
    return [round(float(s), 4) for s in scores]


async def rerank_candidates(
    query: str, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rerank candidates using EmbeddingGemma Retrieval-query prompt (pure bi-encoder cosine similarity).

    ARCH-PERF-002: In-memory rescoring only. Zero outbound HTTP requests during candidate rescoring.
    """
    if not candidates:
        return []

    q_vec = await embed_query(query)
    doc_vecs: list[NDArray[np.float32]] = []
    for c in candidates:
        v = c.get("vector")
        if isinstance(v, (bytes, bytearray)):
            v = np.frombuffer(v, dtype=np.float32)
        elif isinstance(v, list):
            v = np.asarray(v, dtype=np.float32)
        if isinstance(v, np.ndarray) and v.shape == q_vec.shape:
            doc_vecs.append(v)
        else:
            doc_vecs.append(np.zeros_like(q_vec))
    similarities = compute_similarity(q_vec, doc_vecs)

    for idx, sim in enumerate(similarities):
        score_val = round(float(sim), 4)
        candidates[idx].setdefault("channels", {})["rerank_score"] = score_val
        g_prob = candidates[idx].get("graph_prob", 1.0)
        candidates[idx]["score"] = round(score_val * g_prob, 4)

    rel_thresh = get_relevance_threshold()
    if any((c["channels"]["rerank_score"] or 0.0) >= rel_thresh for c in candidates):
        candidates = [
            c
            for c in candidates
            if (c["channels"]["rerank_score"] or 0.0) >= rel_thresh
        ]

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def compute_similarity_matrix(vecs: Any) -> np.ndarray:
    """Compute pairwise cosine similarity matrix via dot product."""
    if len(vecs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    mat = np.asarray(vecs, dtype=np.float32)
    return np.dot(mat, mat.T)


def _from_buffer(raw: bytes | memoryview) -> NDArray[np.float32] | None:
    """Deserialize raw float32 buffer from SQLite BLOB column."""
    if not len(raw):
        return None
    return np.frombuffer(raw, dtype=np.float32)


def _from_str(s: str) -> NDArray[np.float32] | None:
    """Deserialize JSON text string from Cocoindex SQLite TEXT column."""
    trimmed = s.strip()
    if not trimmed:
        return None
    try:
        parsed = json.loads(trimmed)
    except (json.JSONDecodeError, ValueError) as err:
        raise ValueError(f"Failed to parse embedding JSON string: {err}") from err
    if parsed is None:
        return None
    return np.asarray(parsed, dtype=np.float32)


_parse_embedding_str = _from_str


def _from_sequence(raw: list[Any] | tuple[Any, ...]) -> NDArray[np.float32]:
    """Convert sequence of floats into float32 array."""
    return np.asarray(raw, dtype=np.float32)


def _from_array(raw: np.ndarray) -> NDArray[np.float32]:
    """Normalize numpy array to float32 without copying if possible."""
    return raw.astype(np.float32, copy=False)


_EMBEDDING_CONVERTERS: tuple[
    tuple[type | tuple[type, ...], Callable[[Any], NDArray[np.float32] | None]],
    ...,
] = (
    (np.ndarray, _from_array),
    ((bytes, memoryview), _from_buffer),
    (str, _from_str),
    ((list, tuple), _from_sequence),
)


def deserialize_embedding(raw: Any) -> NDArray[np.float32] | None:
    """Deserialize vector embedding from JSON text string, binary buffer, list, or ndarray.

    Supports:
    - None: returns None
    - str: JSON-encoded float array from Cocoindex SQLite TEXT column
    - bytes / memoryview: raw float32 buffer from SQLite BLOB column
    - np.ndarray: pre-parsed NumPy array
    - list / tuple: Python sequence of floats
    """
    if raw is None:
        return None
    for types, handler in _EMBEDDING_CONVERTERS:
        if isinstance(raw, types):
            return handler(raw)
    raise TypeError(
        f"Unsupported embedding type for deserialization: {type(raw).__name__}"
    )
