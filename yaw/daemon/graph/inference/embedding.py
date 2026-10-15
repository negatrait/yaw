"""Native DSPy and OpenAI/Ollama embedding adapter with explicit model-specific retrieval contracts."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
import re
from typing import Any, Callable

import dspy
import numpy as np

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
        return "yaw-embedding-v2:" + hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def format(self, text, *, query, instruction=None):
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
            if self.parse_document_title and "\n" in text:
                first_line, body = text.split("\n", 1)
                match = TITLE_LINE_RE.match(first_line.strip())
                if match and match.group(1).strip() and body.strip():
                    return f"title: {match.group(1).strip()} | text: {body.strip()}"
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
        768, 2048, "mean", QUERY_PREFIX, DOCUMENT_PREFIX,
    ),
    "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0": EmbeddingProfile(
        "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0",
        "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF:Q8_0",
        768, 2048, "mean", QUERY_PREFIX, DOCUMENT_PREFIX,
    ),
    "Qwen3-Embedding-0.6B-GGUF": EmbeddingProfile(
        "Qwen3-Embedding-0.6B-GGUF",
        "Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf",
        1024, 32768, "last",
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
    if provider == "ollama" or model_name.startswith("ollama/"):
        litellm_model = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"
        return dspy.Embedder(
            litellm_model,
            batch_size=batch_size,
            caching=caching,
            api_base=base_url,
            **kwargs,
        )

    litellm_model = model_name if "/" in model_name else f"openai/{model_name}"
    return dspy.Embedder(
        litellm_model,
        batch_size=batch_size,
        caching=caching,
        api_base=base_url,
        api_key=api_key,
        encoding_format="float",
        **kwargs,
    )


class DSPyEmbeddingAdapter:
    """Profile-aware retrieval embedding adapter wrapping a native dspy.Embedder."""

    def __init__(
        self,
        profile: EmbeddingProfile | None = None,
        embedder: dspy.Embedder | None = None,
        *,
        task_mode: str = "distinct-titled",
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "lemonade",
        provider: str = "openai",
        batch_size: int = 32,
        caching: bool = False,
    ) -> None:
        self.profile = profile or resolve_embedding_profile(MODEL, task_mode)
        self.embedder = embedder or create_dspy_embedder(
            self.profile,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            batch_size=batch_size,
            caching=caching,
        )

    def batch_encode(
        self,
        texts: list[str] | str,
        *,
        instruction: str | None = None,
        norm: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if instruction not in (None, "") and instruction not in QUERY_INSTRUCTIONS:
            raise ValueError(f"Unsupported retrieval instruction: {instruction!r}")
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("Embedding input cannot be empty.")
        prepared = [
            self.profile.format(t, query=bool(instruction), instruction=instruction)
            for t in texts
        ]
        call_kwargs: dict[str, Any] = {}
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("batch_size must be a positive integer.")
            call_kwargs["batch_size"] = batch_size
        raw = self.embedder(prepared, **call_kwargs)
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if (
            vectors.shape != (len(prepared), self.profile.dimensions)
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)
        ):
            raise ValueError(
                f"Expected finite, nonzero {self.profile.dimensions}-dimensional vectors."
            )
        if not norm:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


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
    "RetrievalEmbedding",
]


def __getattr__(name: str):
    if name in ("RetrievalEmbedding", "_EmbeddingConfig"):
        from .legacy_embedding import RetrievalEmbedding, _EmbeddingConfig

        mapping = {
            "RetrievalEmbedding": RetrievalEmbedding,
            "_EmbeddingConfig": _EmbeddingConfig,
        }
        val = mapping[name]
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
