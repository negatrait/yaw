"""Native DSPy embedding adapter with model-specific retrieval contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from threading import BoundedSemaphore
from typing import Any, cast

import dspy
import numpy as np

from yaw.core.config import DEFAULT_EMBEDDING_MODEL, YawConfig
from yaw.core.types import ChunkPointer

from .diagnostics import DiagnosticHistory, trim_diagnostics
from .models import BASE_URL

FACT_INSTRUCTION = "Given a question, retrieve relevant triplet facts that matches this question."
PASSAGE_INSTRUCTION = "Given a question, retrieve relevant documents that best answer the question."
TITLE_LINE_RE = re.compile(r"^(?:#{1,6}\s+|(?:wikipedia\s+)?title:\s*)(.+)$", re.IGNORECASE)


def _hash_contract(prefix: str, contract: dict[str, Any], *, ensure_ascii: bool = True) -> str:
    return (
        prefix
        + hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=ensure_ascii).encode("utf-8")).hexdigest()
    )


def _extract_title_body(text: str) -> tuple[str, str] | None:
    if "\n" in text:
        first_line, body = text.split("\n", 1)
        if match := TITLE_LINE_RE.match(first_line.strip()):
            title = match.group(1).strip()
            body = body.strip()
            if title and body:
                return title, body
    return None


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
    def query_instruction_mode(self) -> str:
        if self.fact_query_prefix and self.fact_query_prefix != self.query_prefix:
            return "distinct"
        return "shared"

    @property
    def index_contract(self) -> EmbeddingProfile:
        """Exclude query-only prefixes from document/entity/fact vector work."""
        return replace(self, query_prefix="", fact_query_prefix=None)

    @property
    def index_identity(self) -> str:
        contract: dict[str, Any] = {
            "adapter": "retrieval-prompts-v5-unsplit",
            "checkpoint": self.checkpoint,
            "context_tokens": self.context_tokens,
            "dimensions": self.dimensions,
            "document_prefix": self.document_prefix,
            "model": self.model,
            "normalization": "l2",
            "pooling": self.pooling,
        }
        if self.parse_document_title:
            contract["parse_document_title"] = True
        return _hash_contract("yaw-embedding-v2:", contract)

    def format(self, text: str, *, query: bool, instruction: str | None = None, title: str | None = None) -> str:
        if not isinstance(text, str):
            raise TypeError("Embedding inputs must all be strings.")
        qp = self.query_prefix
        fqp = self.fact_query_prefix
        dp = self.document_prefix
        if (
            (qp and text.startswith(qp))
            or (fqp and text.startswith(fqp))
            or (dp and text.startswith(dp))
            or ((dp or self.parse_document_title) and text.startswith("title: ") and " | text: " in text)
        ):
            raise ValueError("Pass raw text, not an already prefixed embedding input.")
        if not query:
            if self.parse_document_title:
                if (parsed := _extract_title_body(text)) is not None:
                    return f"title: {parsed[0]} | text: {parsed[1]}"
                if title and (clean_title := title.strip()):
                    body = text[len(clean_title) + 1 :].strip() if text.startswith(f"{clean_title}\n") else text.strip()
                    if body:
                        return f"title: {clean_title} | text: {body}"
            return dp + text
        if instruction == FACT_INSTRUCTION and fqp:
            return fqp + text
        # Preserve the model's instruction separator, including Qwen's newline.
        return qp + text


QUERY_PREFIX = "task: search result | query: "
QA_QUERY_PREFIX = "task: question answering | query: "
FACT_CHECK_PREFIX = "task: fact checking | query: "
DOCUMENT_PREFIX = "title: none | text: "
GEMMA_TASK_MODES = ("search", "qa", "fact", "distinct", "titled", "distinct-titled")
_GEMMA_TASK_OVERRIDES: dict[str, dict[str, Any]] = {
    "qa": {"query_prefix": QA_QUERY_PREFIX},
    "fact": {"query_prefix": FACT_CHECK_PREFIX},
    "distinct": {"query_prefix": QA_QUERY_PREFIX, "fact_query_prefix": FACT_CHECK_PREFIX},
    "titled": {"parse_document_title": True},
    "distinct-titled": {
        "query_prefix": QA_QUERY_PREFIX,
        "fact_query_prefix": FACT_CHECK_PREFIX,
        "parse_document_title": True,
    },
}
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
    "embeddinggemma-2": EmbeddingProfile(
        "embeddinggemma-2",
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
MODEL = DEFAULT_EMBEDDING_MODEL


def resolve_embedding_profile(model: str = MODEL, task_mode: str = "search") -> EmbeddingProfile:
    profile = PROFILES[model]
    if task_mode == "search":
        return profile
    if not model.startswith("embeddinggemma"):
        raise ValueError(f"Task mode {task_mode!r} is only supported for EmbeddingGemma profiles.")
    overrides = _GEMMA_TASK_OVERRIDES.get(task_mode)
    if overrides is None:
        raise ValueError(f"Unsupported embedding task mode: {task_mode!r}")
    return replace(profile, **overrides)


def create_dspy_embedder(
    model_or_profile: EmbeddingProfile | str | Callable = MODEL,
    *,
    base_url: str = BASE_URL,
    api_key: str = "local",
    provider: str = "openai",
    batch_size: int = 32,
    caching: bool = True,
) -> dspy.Embedder:
    """Construct DSPy's native OpenAI-compatible or callable embedder."""
    if callable(model_or_profile):
        return dspy.Embedder(model_or_profile, batch_size=batch_size, caching=caching)
    if provider != "openai":
        raise ValueError("Embedding requires an OpenAI-compatible provider")
    name = model_or_profile.model if isinstance(model_or_profile, EmbeddingProfile) else model_or_profile
    return dspy.Embedder(
        f"openai/{name.removeprefix('openai/')}",
        batch_size=batch_size,
        caching=caching,
        base_url=cast(Any, base_url),
        api_key=cast(Any, api_key),
    )


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: tuple[str, ...]
    profile: EmbeddingProfile
    task_mode: str
    batch_size: int
    max_batch_chars: int
    index_identity: str
    instruction: str = ""
    stage: str = "chunk_embed"


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    telemetry: dict[str, Any]


class DSPyEmbeddingAdapter:
    """Profile-aware retrieval adapter using one native DSPy embedder."""

    def __init__(
        self,
        profile: EmbeddingProfile | None = None,
        embedder: dspy.Embedder | None = None,
        *,
        task_mode: str = "distinct-titled",
        base_url: str = BASE_URL,
        api_key: str = "local",
        provider: str = "openai",
        batch_size: int = 32,
        caching: bool = True,
        max_batch_chars: int = 3500,
    ) -> None:
        if batch_size < 1 or max_batch_chars < 1:
            raise ValueError("Embedding batch size and character budget must be positive.")
        self.max_batch_chars = max_batch_chars
        self.task_mode = task_mode
        self.provider = provider
        self.profile = profile or resolve_embedding_profile(MODEL, task_mode)
        self.embedder = embedder or create_dspy_embedder(
            self.profile,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            batch_size=batch_size,
            caching=caching,
        )
        if not isinstance(self.embedder, dspy.Embedder):
            raise TypeError("embedder must be a native dspy.Embedder")
        if callable(self.embedder.model):
            identity = getattr(self.embedder.model, "index_identity", None)
            if not identity:
                raise ValueError("Callable embedding models must declare index_identity")
            self.base_url = str(identity)
        else:
            name = self.profile.model
            expected_model = f"openai/{name.removeprefix('openai/')}"
            if self.embedder.model != expected_model:
                raise ValueError("Embedding model does not match the configured profile")
            self.base_url = str(self.embedder.default_kwargs.get("base_url", base_url)).rstrip("/")
        self.batch_size = self.embedder.batch_size
        self._admission = BoundedSemaphore(2)
        self.telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self._telemetry_hook: Callable[[dict[str, Any]], None] | None = None

    @property
    def index_identity(self) -> str:
        """Identify the configured document-vector contract, without claiming remote weight verification."""
        return _hash_contract(
            "yaw-embedding-deployment-v1:",
            {
                "profile": self.profile.index_identity,
                "provider": self.provider,
                "base_url": self.base_url.rstrip("/"),
                "max_batch_chars": self.max_batch_chars,
            },
            ensure_ascii=False,
        )

    def bind_telemetry_hook(self, hook: Callable[[dict[str, Any]], None] | None) -> None:
        """Bind a callback invoked with each structured embedding telemetry entry."""
        self._telemetry_hook = hook

    def prepare_request(self, texts: Sequence[str], *, batch_size: int, stage: str) -> EmbeddingRequest:
        """Capture an immutable embedding contract on the engine owner thread."""
        return EmbeddingRequest(
            tuple(texts),
            self.profile.index_contract,
            self.task_mode,
            batch_size,
            self.max_batch_chars,
            self.index_identity,
            stage=stage,
        )

    def publish_result(self, result: EmbeddingResult) -> np.ndarray:
        """Publish validated inference telemetry on the engine owner thread."""
        self.telemetry.append(result.telemetry)
        trim_diagnostics(self)
        if self._telemetry_hook is not None:
            self._telemetry_hook(result.telemetry)
        return result.vectors

    async def aexecute_request(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute on the native async provider without publishing shared state."""
        if (self.index_identity, self.task_mode, self.max_batch_chars) != (
            request.index_identity,
            request.task_mode,
            request.max_batch_chars,
        ):
            raise RuntimeError("Embedding settings changed during an indexing batch")
        started = time.perf_counter()
        batches = self._prepare_batches(request.texts, instruction=request.instruction, batch_size=request.batch_size)
        vectors = [await self._aencode_prepared(batch, {"batch_size": len(batch)}) for batch in batches]
        result = self._build_embedding_result(
            batches, vectors, instruction=request.instruction, norm=True, stage=request.stage
        )
        completed = time.perf_counter()
        telemetry = result.telemetry
        telemetry["duration_seconds"] = completed - started
        telemetry["started_at"] = started
        telemetry["completed_at"] = completed
        telemetry["batch_sizes"] = [len(batch) for batch in batches]
        return result

    @classmethod
    def from_config(cls, config: YawConfig) -> DSPyEmbeddingAdapter:
        """Construct a profile-aware DSPyEmbeddingAdapter from a YawConfig."""
        return cls(
            profile=resolve_embedding_profile(config.embedding_model, config.embedding_task_mode),
            task_mode=config.embedding_task_mode,
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            batch_size=config.embed_batch_size,
            max_batch_chars=config.embed_batch_chars,
        )

    def _encode_prepared(self, prepared: list[str], call_kwargs: dict[str, Any]) -> np.ndarray:
        with self._admission:
            return np.asarray(self.embedder(prepared, **call_kwargs), dtype=np.float32)

    async def _aencode_prepared(self, prepared: list[str], call_kwargs: dict[str, Any]) -> np.ndarray:
        acquire = asyncio.create_task(asyncio.to_thread(self._admission.acquire))
        try:
            await asyncio.shield(acquire)
        except asyncio.CancelledError:
            # A worker waiting for a permit cannot be cancelled. Return it as
            # soon as the worker acquires it, even if this task has exited.
            acquire.add_done_callback(lambda _: self._admission.release())
            raise
        try:
            return np.asarray(await self.embedder.acall(prepared, **call_kwargs), dtype=np.float32)
        finally:
            self._admission.release()

    def _prepare_batches(
        self,
        texts: list[str] | str | Sequence[str],
        *,
        instruction: str | None = None,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        if instruction not in (None, "", FACT_INSTRUCTION, PASSAGE_INSTRUCTION):
            raise ValueError(f"Unsupported retrieval instruction: {instruction!r}")
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = list(texts)
        if not texts:
            raise ValueError("Embedding input cannot be empty.")

        requested_batch_size = batch_size if batch_size is not None else self.batch_size
        if requested_batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        effective_max_batch_size = min(requested_batch_size, 64)
        is_query = bool(instruction)
        formatted = [self.profile.format(t, query=is_query, instruction=instruction) for t in texts]
        batches: list[list[str]] = []
        current: list[str] = []
        chars = 0
        for text in formatted:
            if current and (len(current) >= effective_max_batch_size or chars + len(text) > self.max_batch_chars):
                batches.append(current)
                current, chars = [], 0
            current.append(text)
            chars += len(text)
        if current:
            batches.append(current)
        return batches

    def _build_embedding_result(
        self,
        batches: list[list[str]],
        outputs: list[np.ndarray],
        *,
        instruction: str | None,
        norm: bool,
        stage: str | None,
    ) -> EmbeddingResult:
        dims = self.profile.dimensions
        for batch, output in zip(batches, outputs, strict=True):
            if output.shape != (len(batch), dims) or not np.isfinite(output).all():
                raise ValueError("Embedding provider returned an invalid vector count, dimensions or values")
        vectors = np.concatenate(outputs, axis=0)
        formatted_inputs = [text for batch in batches for text in batch]
        is_query = bool(instruction)
        profile = self.profile
        dims = profile.dimensions
        num_inputs = len(formatted_inputs)
        if (
            vectors.shape != (num_inputs, dims)
            or not np.isfinite(vectors).all()
            or np.any((norms := np.linalg.norm(vectors.astype(np.float64), axis=1)) == 0)
        ):
            raise ValueError(f"Expected finite, nonzero {dims}-dimensional vectors.")
        if norm:
            vectors = (vectors / norms[:, None]).astype(np.float32)

        if is_query:
            is_fact = instruction == FACT_INSTRUCTION
            active_prefix = (
                profile.fact_query_prefix if (is_fact and profile.fact_query_prefix) else profile.query_prefix
            )
            resolved_stage = stage or ("query_fact_embed" if is_fact else "query_passage_embed")
        else:
            active_prefix = profile.document_prefix
            resolved_stage = stage or "document_embed"

        total_chars = sum(map(len, formatted_inputs))
        model_name = profile.model
        entry: dict[str, Any] = {
            "stage": resolved_stage,
            "predictor": "embedder",
            "model": model_name,
            "mode": "embed",
            "task_mode": self.task_mode,
            "query": is_query,
            "instruction": instruction or "",
            "active_prefix": active_prefix,
            "parse_document_title": bool(profile.parse_document_title),
            "num_inputs": num_inputs,
            "num_batches": len(batches),
            "total_chars": total_chars,
            # Native Embedder returns vectors, without provider token usage.
            "tokens_used": None,
            "usage": {},
            "inputs_summary": (
                f"num_inputs={num_inputs}, prefix={active_prefix!r}, sample={formatted_inputs[0][:120]!r}"
            ),
        }
        return EmbeddingResult(vectors, entry)

    def batch_encode(
        self,
        texts: list[str] | str | Sequence[str],
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        batches = self._prepare_batches(texts, instruction=instruction, batch_size=batch_size)
        outputs = [self._encode_prepared(batch, {"batch_size": len(batch)}) for batch in batches]
        return self.publish_result(
            self._build_embedding_result(batches, outputs, instruction=instruction, norm=norm, stage=stage)
        )

    async def abatch_encode(
        self,
        texts: list[str] | str | Sequence[str],
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        batches = self._prepare_batches(texts, instruction=instruction, batch_size=batch_size)
        outputs = [await self._aencode_prepared(batch, {"batch_size": len(batch)}) for batch in batches]
        return self.publish_result(
            self._build_embedding_result(batches, outputs, instruction=instruction, norm=norm, stage=stage)
        )

    __call__ = batch_encode
    acall = abatch_encode

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

    def format_chunk_for_embedding(self, ptr: ChunkPointer, text: str) -> str:
        """Ensure AST-resolved ChunkPointer.title is passed as a Title: line for title-aware embedders."""
        parse_title = bool(getattr(getattr(self, "profile", None), "parse_document_title", False))
        clean_title = ptr.title.strip() if ptr.title else ""
        if not parse_title or not clean_title:
            return text
        nl = text.find("\n")
        first_line = (text[:nl] if nl != -1 else text).strip()
        if first_line.startswith("#") or first_line.lower().startswith(("title:", "wikipedia title:")):
            return text
        if text.startswith(f"{clean_title}\n"):
            body = text[len(clean_title) + 1 :]
            return f"Title: {clean_title}\n{body}" if body.strip() else text
        return f"Title: {clean_title}\n{text}"

    @staticmethod
    def format_entity_for_embedding(name: str, definition: str) -> str:
        """Format an entity name and contextual definition for embedding."""
        clean_def = definition.strip()
        return f"# {name}\n{clean_def}" if clean_def else name

    @staticmethod
    def format_fact_for_embedding(
        subj: str,
        pred: str,
        obj: str,
        definition: str = "",
        subj_def: str = "",
        obj_def: str = "",
    ) -> str:
        """Format a subject-predicate-object fact and contextual statement definition for embedding."""
        base = f"{subj} {pred} {obj}"
        s_d = definition.strip()
        if s_d and s_d.lower().rstrip(".") != base.lower().rstrip("."):
            return f"{base}. {s_d}"
        return base


__all__ = [
    "FACT_INSTRUCTION",
    "PASSAGE_INSTRUCTION",
    "EmbeddingProfile",
    "PROFILES",
    "MODEL",
    "GEMMA_TASK_MODES",
    "resolve_embedding_profile",
    "create_dspy_embedder",
    "DSPyEmbeddingAdapter",
]
