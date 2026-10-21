"""Native DSPy and OpenAI/Ollama embedding adapter with explicit model-specific retrieval contracts."""

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import dspy
import numpy as np

from yaw.core.config import YawConfig

FACT_INSTRUCTION = "Given a question, retrieve relevant triplet facts that matches this question."
PASSAGE_INSTRUCTION = "Given a question, retrieve relevant documents that best answer the question."
QUERY_INSTRUCTIONS = {FACT_INSTRUCTION, PASSAGE_INSTRUCTION}
TITLE_LINE_RE = re.compile(r"^(?:#{1,6}\s+|(?:wikipedia\s+)?title:\s*)(.+)$", re.IGNORECASE)
DEFAULT_BASE_URL = "http://localhost:13305/api/v1"


@dataclass(frozen=True)
class EmbeddingProfile:
    model: str
    checkpoint: str
    dimensions: int
    context_tokens: int
    pooling: str
    query_prefix: str
    document_prefix: str
    fact_query_prefix: str | None = None
    parse_document_title: bool = False

    @property
    def query_instruction_mode(self):
        if self.fact_query_prefix and self.fact_query_prefix != self.query_prefix:
            return "distinct"
        return "shared"

    @property
    def index_identity(self):
        payload = asdict(self)
        if self.fact_query_prefix is None:
            payload.pop("fact_query_prefix", None)
        if not self.parse_document_title:
            payload.pop("parse_document_title", None)
        contract = {"adapter": "retrieval-prompts-v3", "normalization": "l2", **payload}
        return (
            "yaw-embedding-v2:"
            + hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
        )

    def format(self, text, *, query, instruction=None, title: str | None = None):
        if not isinstance(text, str):
            raise TypeError("Embedding inputs must all be strings.")
        prefixes = tuple(
            p for p in (self.query_prefix, self.fact_query_prefix, self.document_prefix) if p
        )
        if text.startswith(prefixes) or (
            (self.document_prefix or self.parse_document_title)
            and text.startswith("title: ")
            and " | text: " in text
        ):
            raise ValueError("Pass raw text, not an already prefixed embedding input.")
        if not query:
            if self.parse_document_title:
                if "\n" in text:
                    first_line, body = text.split("\n", 1)
                    match = TITLE_LINE_RE.match(first_line.strip())
                    if match and match.group(1).strip() and body.strip():
                        return f"title: {match.group(1).strip()} | text: {body.strip()}"
                if title and title.strip():
                    clean_title = title.strip()
                    if text.startswith(f"{clean_title}\n"):
                        body = text[len(clean_title) + 1 :].strip()
                    else:
                        body = text.strip()
                    if body:
                        return f"title: {clean_title} | text: {body}"
            return self.document_prefix + text
        if instruction == FACT_INSTRUCTION and self.fact_query_prefix:
            return self.fact_query_prefix + text
        # Preserve the model's instruction separator, including Qwen's newline.
        return self.query_prefix + text


QUERY_PREFIX = "task: search result | query: "
QA_QUERY_PREFIX = "task: question answering | query: "
FACT_CHECK_PREFIX = "task: fact checking | query: "
DOCUMENT_PREFIX = "title: none | text: "
GEMMA_TASK_MODES = ("search", "qa", "fact", "distinct", "titled", "distinct-titled")
PROFILES = {
    "embeddinggemma-300m-GGUF": EmbeddingProfile(
        "embeddinggemma-300m-GGUF",
        "ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf",
        768,
        2048,
        "mean",
        QUERY_PREFIX,
        DOCUMENT_PREFIX,
    ),
    "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0": EmbeddingProfile(
        "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0",
        "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF:Q8_0",
        768,
        2048,
        "mean",
        QUERY_PREFIX,
        DOCUMENT_PREFIX,
    ),
    "Qwen3-Embedding-0.6B-GGUF": EmbeddingProfile(
        "Qwen3-Embedding-0.6B-GGUF",
        "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
        1024,
        32768,
        "last",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        "",
    ),
}
MODEL = "embeddinggemma-300m-GGUF"


def resolve_embedding_profile(model=MODEL, task_mode="search"):
    profile = PROFILES[model]
    if task_mode == "search":
        return profile
    if not model.startswith("embeddinggemma"):
        raise ValueError(f"Task mode {task_mode!r} is only supported for EmbeddingGemma profiles.")
    if task_mode == "qa":
        return replace(profile, query_prefix=QA_QUERY_PREFIX)
    if task_mode == "fact":
        return replace(profile, query_prefix=FACT_CHECK_PREFIX)
    if task_mode == "distinct":
        return replace(profile, query_prefix=QA_QUERY_PREFIX, fact_query_prefix=FACT_CHECK_PREFIX)
    if task_mode == "titled":
        return replace(profile, parse_document_title=True)
    if task_mode == "distinct-titled":
        return replace(
            profile,
            query_prefix=QA_QUERY_PREFIX,
            fact_query_prefix=FACT_CHECK_PREFIX,
            parse_document_title=True,
        )
    raise ValueError(f"Unsupported embedding task mode: {task_mode!r}")


def create_dspy_embedder(
    model_or_profile: EmbeddingProfile | str | Callable = MODEL,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "lemonade",
    provider: str = "openai",
    batch_size: int = 32,
    caching: bool = False,
    **kwargs: Any,
) -> dspy.Embedder:
    """Create a native dspy.Embedder for Lemonade (openai/...) or Ollama (ollama/...)."""
    if callable(model_or_profile):
        return dspy.Embedder(model_or_profile, batch_size=batch_size, caching=caching, **kwargs)

    model_name = (
        model_or_profile.model
        if isinstance(model_or_profile, EmbeddingProfile)
        else str(model_or_profile)
    )
    embedder_cls: Any = dspy.Embedder
    if provider == "ollama" or model_name.startswith("ollama/"):
        litellm_model = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"
        return embedder_cls(
            litellm_model,
            batch_size=batch_size,
            caching=caching,
            api_base=base_url,
            **kwargs,
        )

    litellm_model = model_name if "/" in model_name else f"openai/{model_name}"
    return embedder_cls(
        litellm_model,
        batch_size=batch_size,
        caching=caching,
        api_base=base_url,
        api_key=api_key,
        encoding_format="float",
        **kwargs,
    )


def is_batch_overflow_error(exc: Exception) -> bool:
    """Detect embedding server token limit or physical batch size rejection."""
    msg = str(exc).lower()
    return (
        "too large to process" in msg
        or "physical batch size" in msg
        or "context size" in msg
        or "maximum context length" in msg
        or "token limit" in msg
        or "context window" in msg
        or "too many tokens" in msg
        or ("exceeds" in msg and "tokens" in msg)
    )


_is_batch_overflow_error = is_batch_overflow_error


def split_text_gracefully(
    text: str,
    max_chars: int = 3000,
    overlap_chars: int = 100,
) -> list[str]:
    """Gracefully divide a long text into sub-chunks without truncation.

    Snaps split boundaries to paragraph (\\n\\n), newline (\\n), sentence (. ! ?),
    or word boundaries (space), ensuring 100% of the original text is covered.
    """
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    total_len = len(text)

    while start < total_len:
        if total_len - start <= max_chars:
            chunks.append(text[start:])
            break

        target_end = min(start + max_chars, total_len)
        min_split = start + max(1, max_chars // 3)
        window = text[min_split:target_end]
        split_pos = -1

        # Priority 1: Paragraph break \n\n
        idx = window.rfind("\n\n")
        if idx != -1:
            split_pos = min_split + idx + 2
        else:
            # Priority 2: Line break \n
            idx = window.rfind("\n")
            if idx != -1:
                split_pos = min_split + idx + 1
            else:
                # Priority 3: Sentence boundary (. ! ?)
                sentence_match = list(re.finditer(r"[.!?]\s+", window))
                if sentence_match:
                    split_pos = min_split + sentence_match[-1].end()
                else:
                    # Priority 4: Clause or phrase boundary (; ,)
                    clause_match = list(re.finditer(r"[,;:]\s+", window))
                    if clause_match:
                        split_pos = min_split + clause_match[-1].end()
                    else:
                        # Priority 5: Word boundary
                        idx = window.rfind(" ")
                        split_pos = min_split + idx + 1 if idx != -1 else target_end

        chunk = text[start:split_pos]
        if chunk.strip():
            chunks.append(chunk)

        advance = (
            max(1, split_pos - overlap_chars)
            if overlap_chars > 0 and split_pos < total_len
            else split_pos
        )
        start = max(start + 1, advance)

    return chunks if chunks else [text]


def pool_subchunk_embeddings(
    vectors: np.ndarray,
    weights: Sequence[int] | np.ndarray,
    *,
    norm: bool = True,
) -> np.ndarray:
    """Combine sub-chunk embedding vectors into a single document vector via length-weighted pooling.

    Args:
        vectors: Array of shape (K, dim) containing sub-chunk embeddings.
        weights: Sequence of positive weights (typically character counts of sub-chunks).
        norm: Whether to apply L2 normalization to the pooled vector.

    Returns:
        1D array of shape (dim,) representing the whole document.
    """
    if len(vectors) == 1:
        vec = np.asarray(vectors[0], dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)
        total_w = float(w.sum())
        w = w / total_w if total_w > 0 else np.full_like(w, 1.0 / len(w))
        vec = np.sum(vectors * w[:, None], axis=0)

    if norm:
        v_norm = float(np.linalg.norm(vec))
        vec = vec / max(v_norm, 1e-12)
    return vec


def batch_texts_by_budget(
    texts: Sequence[str],
    *,
    max_batch_chars: int = 3500,
    max_batch_size: int = 16,
) -> list[list[int]]:
    """Partition flat texts into index micro-batches bounded by character budget and count."""
    micro_batches: list[list[int]] = []
    current_batch: list[int] = []
    current_chars = 0

    for idx, t in enumerate(texts):
        t_len = len(t)
        if current_batch and (
            current_chars + t_len > max_batch_chars or len(current_batch) >= max_batch_size
        ):
            micro_batches.append(current_batch)
            current_batch = [idx]
            current_chars = t_len
        else:
            current_batch.append(idx)
            current_chars += t_len

    if current_batch:
        micro_batches.append(current_batch)
    return micro_batches


def _repack_overflow_batch(texts: list[str], context_tokens: int) -> list[list[str]]:
    """Gracefully split texts without truncation after embedding server rejection."""
    char_budget = max(512, int(context_tokens * 1.5))
    if len(texts) == 1:
        sub_chunks = split_text_gracefully(texts[0], max_chars=max(256, char_budget // 2))
        return [[sc] for sc in sub_chunks]

    packed: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for t in texts:
        if len(t) > char_budget:
            if current:
                packed.append(current)
                current = []
                current_chars = 0
            for sc in split_text_gracefully(t, max_chars=char_budget):
                packed.append([sc])
            continue

        if current and current_chars + len(t) > char_budget:
            packed.append(current)
            current = [t]
            current_chars = len(t)
        else:
            current.append(t)
            current_chars += len(t)
    if current:
        packed.append(current)

    if len(packed) > 1:
        return packed

    mid = len(texts) // 2
    return [texts[:mid], texts[mid:]]


class DSPyEmbeddingAdapter(dspy.Embedder):
    """Profile-aware retrieval embedding adapter subclassing native `dspy.Embedder`."""

    def __init__(
        self,
        profile: EmbeddingProfile | None = None,
        embedder: dspy.Embedder | Any = None,
        *,
        task_mode: str = "distinct-titled",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "lemonade",
        provider: str = "openai",
        batch_size: int = 32,
        caching: bool = False,
    ) -> None:
        self.task_mode = task_mode
        self.profile = profile or resolve_embedding_profile(MODEL, task_mode)
        base_embedder = embedder or create_dspy_embedder(
            self.profile,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            batch_size=batch_size,
            caching=caching,
        )
        if isinstance(base_embedder, dspy.Embedder):
            super().__init__(
                base_embedder.model,
                batch_size=base_embedder.batch_size,
                caching=base_embedder.caching,
                **getattr(base_embedder, "default_kwargs", {}),
            )
            self.embedder: Any = base_embedder
        else:
            super().__init__(
                base_embedder,
                batch_size=batch_size,
                caching=caching,
            )
            self.embedder = base_embedder
        self.telemetry: list[dict[str, Any]] = []
        self._telemetry_hook: Callable[[dict[str, Any]], None] | None = None

    def bind_telemetry_hook(self, hook: Callable[[dict[str, Any]], None] | None) -> None:
        """Bind a callback invoked with each structured embedding telemetry entry."""
        self._telemetry_hook = hook

    def reset_telemetry(self) -> None:
        """Clear accumulated embedding telemetry entries."""
        self.telemetry.clear()

    @classmethod
    def from_config(cls, config: YawConfig) -> DSPyEmbeddingAdapter:
        """Construct a profile-aware DSPyEmbeddingAdapter from a YawConfig."""
        profile = resolve_embedding_profile(config.embedding_model, config.embedding_task_mode)
        return cls(
            profile=profile,
            task_mode=config.embedding_task_mode,
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            batch_size=config.embed_batch_size,
        )

    def _encode_prepared(
        self,
        prepared: list[str],
        call_kwargs: dict[str, Any],
    ) -> np.ndarray:
        try:
            if isinstance(self.embedder, dspy.Embedder):
                raw = dspy.Embedder.__call__(self, prepared, **call_kwargs)
            else:
                raw = self.embedder(prepared, **call_kwargs)
            vectors = np.asarray(raw, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            from .chat import get_yaw_sleep_seconds

            sleep_sec = get_yaw_sleep_seconds()
            if sleep_sec > 0:
                import time

                time.sleep(sleep_sec)
            return vectors
        except Exception as exc:
            if not is_batch_overflow_error(exc):
                raise
            if len(prepared) > 1:
                mid = len(prepared) // 2
                left = self._encode_prepared(prepared[:mid], {**call_kwargs, "batch_size": mid})
                right = self._encode_prepared(
                    prepared[mid:], {**call_kwargs, "batch_size": len(prepared) - mid}
                )
                return np.concatenate([left, right], axis=0)

            # Single item overflow: divide gracefully into 2+ sub-chunks and pool without truncation
            single = prepared[0]
            sub_chunks = split_text_gracefully(single, max_chars=max(256, len(single) // 2))
            if len(sub_chunks) <= 1:
                raise
            sub_vecs = self._encode_prepared(
                sub_chunks, {**call_kwargs, "batch_size": len(sub_chunks)}
            )
            weights = [max(1, len(s)) for s in sub_chunks]
            pooled = pool_subchunk_embeddings(sub_vecs, weights, norm=True)
            return pooled.reshape(1, -1)

    async def _aencode_prepared(
        self,
        prepared: list[str],
        call_kwargs: dict[str, Any],
    ) -> np.ndarray:
        try:
            if isinstance(self.embedder, dspy.Embedder) and isinstance(self.model, str):
                raw = await dspy.Embedder.acall(self, prepared, **call_kwargs)
            else:
                target_fn = (
                    (lambda items, **kw: dspy.Embedder.__call__(self, items, **kw))
                    if isinstance(self.embedder, dspy.Embedder)
                    else self.embedder
                )
                async_fn = cast(
                    Callable[..., Awaitable[Any]],
                    dspy.asyncify(cast(Any, target_fn)),
                )
                raw = await async_fn(prepared, **call_kwargs)
            vectors = np.asarray(raw, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            from .chat import get_yaw_sleep_seconds

            sleep_sec = get_yaw_sleep_seconds()
            if sleep_sec > 0:
                import time

                time.sleep(sleep_sec)
            return vectors
        except Exception as exc:
            if not is_batch_overflow_error(exc):
                raise
            if len(prepared) > 1:
                mid = len(prepared) // 2
                left = await self._aencode_prepared(
                    prepared[:mid], {**call_kwargs, "batch_size": mid}
                )
                right = await self._aencode_prepared(
                    prepared[mid:], {**call_kwargs, "batch_size": len(prepared) - mid}
                )
                return np.concatenate([left, right], axis=0)

            single = prepared[0]
            sub_chunks = split_text_gracefully(single, max_chars=max(256, len(single) // 2))
            if len(sub_chunks) <= 1:
                raise
            sub_vecs = await self._aencode_prepared(
                sub_chunks, {**call_kwargs, "batch_size": len(sub_chunks)}
            )
            weights = [max(1, len(s)) for s in sub_chunks]
            pooled = pool_subchunk_embeddings(sub_vecs, weights, norm=True)
            return pooled.reshape(1, -1)

    def _prepare_pieces_and_batches(
        self,
        texts: list[str] | str,
        *,
        instruction: str | None = None,
        batch_size: int | None = None,
    ) -> tuple[list[str], bool, list[list[int]], list[str], list[int], list[list[int]]]:
        if instruction not in (None, "") and instruction not in QUERY_INSTRUCTIONS:
            raise ValueError(f"Unsupported retrieval instruction: {instruction!r}")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("Embedding input cannot be empty.")

        max_chunk_chars = 3000
        max_batch_chars = 3500
        effective_max_batch_size = (
            batch_size if (isinstance(batch_size, int) and batch_size >= 1) else 16
        )

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

        micro_batches = batch_texts_by_budget(
            flat_pieces,
            max_batch_chars=max_batch_chars,
            max_batch_size=effective_max_batch_size,
        )
        return texts, is_query, text_piece_map, flat_pieces, piece_weights, micro_batches

    def _finalize_vectors_and_telemetry(
        self,
        texts: list[str],
        text_piece_map: list[list[int]],
        flat_pieces: list[str],
        piece_weights: list[int],
        micro_batches: list[list[int]],
        flat_vectors: list[np.ndarray | None],
        *,
        is_query: bool,
        instruction: str | None,
        norm: bool,
        stage: str | None,
    ) -> np.ndarray:
        doc_vectors: list[np.ndarray] = []
        for indices in text_piece_map:
            sub_vecs = np.asarray(
                [flat_vectors[i] for i in indices if flat_vectors[i] is not None],
                dtype=np.float32,
            )
            weights = [piece_weights[i] for i in indices]
            doc_vec = pool_subchunk_embeddings(sub_vecs, weights, norm=norm)
            doc_vectors.append(doc_vec)

        vectors = np.stack(doc_vectors)
        if (
            vectors.shape != (len(texts), self.profile.dimensions)
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)
        ):
            raise ValueError(
                f"Expected finite, nonzero {self.profile.dimensions}-dimensional vectors."
            )

        if is_query:
            active_prefix = (
                self.profile.fact_query_prefix
                if (instruction == FACT_INSTRUCTION and self.profile.fact_query_prefix)
                else self.profile.query_prefix
            )
            resolved_stage = stage or (
                "query_fact_embed" if instruction == FACT_INSTRUCTION else "query_passage_embed"
            )
        else:
            active_prefix = self.profile.document_prefix
            resolved_stage = stage or "document_embed"

        total_chars = sum(len(p) for p in flat_pieces)
        approx_tokens = max(1, (total_chars + 3) // 4)
        entry: dict[str, Any] = {
            "stage": resolved_stage,
            "predictor": "embedder",
            "model": self.profile.model,
            "mode": "embed",
            "task_mode": self.task_mode,
            "query": is_query,
            "instruction": instruction or "",
            "active_prefix": active_prefix,
            "parse_document_title": bool(self.profile.parse_document_title),
            "num_inputs": len(texts),
            "num_pieces": len(flat_pieces),
            "num_batches": len(micro_batches),
            "total_chars": total_chars,
            "tokens_used": approx_tokens,
            "usage": {
                self.profile.model: {
                    "prompt_tokens": approx_tokens,
                    "completion_tokens": 0,
                    "total_tokens": approx_tokens,
                }
            },
            "inputs_summary": (
                f"num_inputs={len(texts)}, num_pieces={len(flat_pieces)}, "
                f"prefix={active_prefix!r}, sample={flat_pieces[0][:120]!r}"
            ),
        }
        self.telemetry.append(entry)
        if self._telemetry_hook is not None:
            self._telemetry_hook(entry)

        return vectors

    def batch_encode(
        self,
        texts: list[str] | str,
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        (
            resolved_texts,
            is_query,
            text_piece_map,
            flat_pieces,
            piece_weights,
            micro_batches,
        ) = self._prepare_pieces_and_batches(texts, instruction=instruction, batch_size=batch_size)

        flat_vectors: list[np.ndarray | None] = [None] * len(flat_pieces)
        for mb in micro_batches:
            mb_texts = [flat_pieces[i] for i in mb]
            call_kwargs = {"batch_size": len(mb_texts)}
            vecs = self._encode_prepared(mb_texts, call_kwargs)
            for local_idx, global_idx in enumerate(mb):
                flat_vectors[global_idx] = vecs[local_idx]

        return self._finalize_vectors_and_telemetry(
            resolved_texts,
            text_piece_map,
            flat_pieces,
            piece_weights,
            micro_batches,
            flat_vectors,
            is_query=is_query,
            instruction=instruction,
            norm=norm,
            stage=stage,
        )

    async def abatch_encode(
        self,
        texts: list[str] | str,
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        """Asynchronously encode micro-batches concurrently using anyio task groups and dspy.asyncify."""
        import anyio

        (
            resolved_texts,
            is_query,
            text_piece_map,
            flat_pieces,
            piece_weights,
            micro_batches,
        ) = self._prepare_pieces_and_batches(texts, instruction=instruction, batch_size=batch_size)

        flat_vectors: list[np.ndarray | None] = [None] * len(flat_pieces)

        async def _run_micro_batch(mb: list[int]) -> None:
            mb_texts = [flat_pieces[i] for i in mb]
            call_kwargs = {"batch_size": len(mb_texts)}
            vecs = await self._aencode_prepared(mb_texts, call_kwargs)
            for local_idx, global_idx in enumerate(mb):
                flat_vectors[global_idx] = vecs[local_idx]

        if len(micro_batches) == 1:
            await _run_micro_batch(micro_batches[0])
        else:
            async with anyio.create_task_group() as tg:
                for mb in micro_batches:
                    tg.start_soon(_run_micro_batch, mb)

        return self._finalize_vectors_and_telemetry(
            resolved_texts,
            text_piece_map,
            flat_pieces,
            piece_weights,
            micro_batches,
            flat_vectors,
            is_query=is_query,
            instruction=instruction,
            norm=norm,
            stage=stage,
        )

    def __call__(
        self,
        inputs: str | list[str],
        batch_size: int | None = None,
        caching: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """DSPy Embedder entrypoint delegating to profile-aware batch_encode."""
        _ = caching
        instruction = kwargs.pop("instruction", None)
        norm = bool(kwargs.pop("norm", True))
        stage = kwargs.pop("stage", None)
        return self.batch_encode(
            inputs,
            instruction=instruction if isinstance(instruction, str) else None,
            norm=norm,
            batch_size=batch_size,
            stage=stage if isinstance(stage, str) else None,
        )

    async def acall(
        self,
        inputs: str | list[str],
        batch_size: int | None = None,
        caching: bool | None = None,
        *,
        instruction: str | None = None,
        norm: bool = True,
        stage: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Async DSPy Embedder entrypoint delegating to profile-aware abatch_encode."""
        _ = (caching, kwargs)
        return await self.abatch_encode(
            inputs,
            instruction=instruction,
            norm=norm,
            batch_size=batch_size,
            stage=stage,
        )

    def forward(
        self,
        texts: list[str] | str,
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        """Forward pass delegating to batch_encode."""
        return self.batch_encode(
            texts,
            instruction=instruction,
            norm=norm,
            batch_size=batch_size,
            stage=stage,
        )

    def as_async(self) -> Callable[..., Awaitable[np.ndarray]]:
        """Return a thread-context-preserving async callable via dspy.asyncify(self)."""
        return cast(
            Callable[..., Awaitable[np.ndarray]],
            dspy.asyncify(cast(Any, self)),
        )

    def encode(
        self,
        texts: list[str] | str | Sequence[str],
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        """Alias for batch_encode matching sentence-transformers and DSPy embedder interfaces."""
        if not isinstance(texts, (str, list)):
            texts = list(texts)
        return self.batch_encode(
            texts,
            instruction=instruction,
            norm=norm,
            batch_size=batch_size,
            stage=stage,
        )

    def update_prefixes(
        self,
        *,
        query_prefix: str | None = None,
        fact_query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        """Apply optimized EmbeddingGemma task prefixes to the active EmbeddingProfile."""
        updates: dict[str, Any] = {}
        if query_prefix is not None:
            updates["query_prefix"] = query_prefix
        if fact_query_prefix is not None:
            updates["fact_query_prefix"] = fact_query_prefix
        if document_prefix is not None:
            updates["document_prefix"] = document_prefix
        if updates:
            self.profile = replace(self.profile, **updates)


__all__ = [
    "FACT_INSTRUCTION",
    "PASSAGE_INSTRUCTION",
    "QUERY_INSTRUCTIONS",
    "EmbeddingProfile",
    "PROFILES",
    "MODEL",
    "GEMMA_TASK_MODES",
    "resolve_embedding_profile",
    "create_dspy_embedder",
    "DSPyEmbeddingAdapter",
    "is_batch_overflow_error",
    "split_text_gracefully",
    "pool_subchunk_embeddings",
    "batch_texts_by_budget",
]
