"""Inference leaf package: Native DSPy HippoRAG 2 engine, GBNF grammars, and legacy baseline adapters."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    # Native DSPy LM & Heterogeneous RLM
    "create_dspy_lm": ".chat",
    "create_dspy_rlm_pair": ".chat",
    "lm_supports_gbnf": ".grammars",
    # Native DSPy Embedder & Profiles
    "create_dspy_embedder": ".embedding",
    "DSPyEmbeddingAdapter": ".embedding",
    "resolve_embedding_profile": ".embedding",
    "EmbeddingProfile": ".embedding",
    "PROFILES": ".embedding",
    "MODEL": ".embedding",
    "GEMMA_TASK_MODES": ".embedding",
    # Native DSPy Program & Signatures
    "HippoRAGProgram": ".program",
    "ExtractEntities": ".prompts",
    "ExtractTriples": ".prompts",
    "FilterFacts": ".prompts",
    "AnswerQuestion": ".prompts",
    "FactList": ".prompts",
    # Pure-JSON Context-Free Grammars & Sizecoding Metric
    "compute_sizecoding_pareto_score": ".grammars",
    "build_dspy_ner_grammar": ".grammars",
    "build_dspy_triples_grammar": ".grammars",
    "build_dspy_rerank_grammar": ".grammars",
    "build_dspy_qa_cot_grammar": ".grammars",
    "DSPY_NER_GBNF_GRAMMAR": ".grammars",
    "DSPY_TRIPLES_GBNF_GRAMMAR": ".grammars",
    "DSPY_RERANK_GBNF_GRAMMAR": ".grammars",
    "DSPY_QA_COT_GBNF_GRAMMAR": ".grammars",
    # Pinned Upstream Baseline Adapters
    "LocalChat": ".legacy_chat",
    "CacheOpenAI": ".legacy_chat",
    "chat_identity": ".legacy_chat",
    "RetrievalEmbedding": ".legacy_embedding",
    # Models & Constants
    "CHAT_MODELS": ".models",
    "BASE_URL": ".models",
    "DEFAULT_EXTRACTION_CHAT": ".models",
    "DEFAULT_QA_CHAT": ".models",
}

__all__ = sorted(_PUBLIC_API.keys())


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_API:
        mod_name = _PUBLIC_API[name]
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
