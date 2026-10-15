"""Inference leaf package: Native DSPy HippoRAG 2 engine, GBNF grammars, and legacy baseline adapters."""

import importlib
from typing import Any

__all__ = [
    # Native DSPy LM & Heterogeneous RLM
    "create_dspy_lm",
    "create_dspy_rlm_pair",
    "lm_supports_gbnf",
    # Native DSPy Embedder & Profiles
    "create_dspy_embedder",
    "DSPyEmbeddingAdapter",
    "resolve_embedding_profile",
    "EmbeddingProfile",
    "PROFILES",
    "MODEL",
    # Native DSPy Program & Signatures
    "HippoRAGProgram",
    "ExtractEntities",
    "ExtractTriples",
    "FilterFacts",
    "AnswerQuestion",
    "FactList",
    # Pure-JSON Context-Free Grammars & Sizecoding Metric
    "compute_sizecoding_pareto_score",
    "build_dspy_ner_grammar",
    "build_dspy_triples_grammar",
    "build_dspy_rerank_grammar",
    "build_dspy_qa_cot_grammar",
    "DSPY_NER_GBNF_GRAMMAR",
    "DSPY_TRIPLES_GBNF_GRAMMAR",
    "DSPY_RERANK_GBNF_GRAMMAR",
    "DSPY_QA_COT_GBNF_GRAMMAR",
    # Legacy Adapters (for upstream baseline parity)
    "LocalChat",
    "CacheOpenAI",
    "chat_identity",
    "RetrievalEmbedding",
    # Models & Constants
    "CHAT_MODELS",
    "BASE_URL",
    "DEFAULT_EXTRACTION_CHAT",
    "DEFAULT_QA_CHAT",
]

_MODULE_MAP = {
    "create_dspy_lm": ".chat",
    "create_dspy_rlm_pair": ".chat",
    "lm_supports_gbnf": ".grammars",
    "create_dspy_embedder": ".embedding",
    "DSPyEmbeddingAdapter": ".embedding",
    "resolve_embedding_profile": ".embedding",
    "EmbeddingProfile": ".embedding",
    "PROFILES": ".embedding",
    "MODEL": ".embedding",
    "HippoRAGProgram": ".program",
    "ExtractEntities": ".prompts",
    "ExtractTriples": ".prompts",
    "FilterFacts": ".prompts",
    "AnswerQuestion": ".prompts",
    "FactList": ".prompts",
    "compute_sizecoding_pareto_score": ".grammars",
    "build_dspy_ner_grammar": ".grammars",
    "build_dspy_triples_grammar": ".grammars",
    "build_dspy_rerank_grammar": ".grammars",
    "build_dspy_qa_cot_grammar": ".grammars",
    "DSPY_NER_GBNF_GRAMMAR": ".grammars",
    "DSPY_TRIPLES_GBNF_GRAMMAR": ".grammars",
    "DSPY_RERANK_GBNF_GRAMMAR": ".grammars",
    "DSPY_QA_COT_GBNF_GRAMMAR": ".grammars",
    "LocalChat": ".legacy_chat",
    "CacheOpenAI": ".legacy_chat",
    "chat_identity": ".legacy_chat",
    "RetrievalEmbedding": ".legacy_embedding",
    "CHAT_MODELS": ".models",
    "BASE_URL": ".models",
    "DEFAULT_EXTRACTION_CHAT": ".models",
    "DEFAULT_QA_CHAT": ".models",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_MAP:
        mod_name = _MODULE_MAP[name]
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
