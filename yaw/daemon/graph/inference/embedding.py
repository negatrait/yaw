"""One OpenAI embedding adapter with explicit model-specific retrieval contracts."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re

import numpy as np
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.prompts.linking import get_query_instruction

FACT_INSTRUCTION = get_query_instruction("query_to_fact")
PASSAGE_INSTRUCTION = get_query_instruction("query_to_passage")
QUERY_INSTRUCTIONS = {FACT_INSTRUCTION, PASSAGE_INSTRUCTION}
TITLE_LINE_RE = re.compile(r"^(?:#{1,6}\s+|(?:wikipedia\s+)?title:\s*)(.+)$", re.IGNORECASE)


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


class RetrievalEmbedding(OpenAIEmbeddingModel):
    """Keep upstream SDK, batching, usage aggregation and normalization.

    encode is intentionally local: upstream encode flattens newlines, corrupting
    Qwen's task separator. Its ordering/usage guarantees are retained and tested.
    No global patch, auto-detection, fallback model or dimension truncation.
    """

    query_instruction_mode = "shared"

    def __init__(self, global_config, profile=None):
        self.profile = profile or PROFILES[global_config.embedding_model_name]
        if global_config.embedding_model_name != self.profile.model:
            raise ValueError("Embedding config model must match the selected profile.")
        self.query_instruction_mode = self.profile.query_instruction_mode
        super().__init__(global_config)

    def batch_encode(self, texts, **kwargs):
        instruction = kwargs.pop("instruction", None)
        if instruction not in (None, "") and instruction not in QUERY_INSTRUCTIONS:
            raise ValueError(f"Unsupported retrieval instruction: {instruction!r}")
        if isinstance(texts, str):
            texts = [texts]
        prepared = [
            self.profile.format(t, query=bool(instruction), instruction=instruction)
            for t in texts
        ]
        return super().batch_encode(prepared, **kwargs)

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
