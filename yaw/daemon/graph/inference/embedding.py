"""Native DSPy and OpenAI/Ollama embedding adapter with explicit model-specific retrieval contracts."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
import re
from typing import Any, Callable

import dspy
import numpy as np
from openai import DefaultHttpxClient, OpenAI

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


@dataclass
class _EmbeddingConfig:
    embedding_model_name: str
    norm: bool = True
    encode_params: dict[str, Any] = field(default_factory=lambda: {"batch_size": 32})


class RetrievalEmbedding:
    """Self-contained OpenAI-compatible embedding client with task prefixes and strict validation.

    Zero runtime dependency on upstream `hipporag`, while preserving full interface
    compatibility for both `yaw` and `benchmark.py`.
    """

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
                    usage_totals["prompt_tokens"] += self.last_usage["prompt_tokens"]
                    usage_totals["total_tokens"] += self.last_usage["total_tokens"]
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
