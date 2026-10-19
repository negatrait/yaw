"""Inference leaf package: Native DSPy HippoRAG 2 engine, GBNF grammars, and legacy baseline adapters."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    # Native DSPy LM & Heterogeneous RLM
    "create_dspy_lm": ".chat",
    "create_dspy_rlm_pair": ".chat",
    "configure_yaw_dspy_cache": ".chat",
    "lm_supports_gbnf": ".grammars",
    # Native DSPy Embedder & Profiles
    "create_dspy_embedder": ".embedding",
    "DSPyEmbeddingAdapter": ".embedding",
    "resolve_embedding_profile": ".embedding",
    "EmbeddingProfile": ".embedding",
    "PROFILES": ".embedding",
    "MODEL": ".embedding",
    "GEMMA_TASK_MODES": ".embedding",
    "FACT_INSTRUCTION": ".embedding",
    "PASSAGE_INSTRUCTION": ".embedding",
    # Stateless Parsers & Hash Utilities
    "compute_mdhash_id": ".parsers",
    "compute_qa_em_f1": ".parsers",
    "text_processing": ".parsers",
    "extract_entity_definitions": ".parsers",
    "extract_entity_nodes": ".parsers",
    "extract_structured_triples": ".parsers",
    "flatten_facts": ".parsers",
    "filter_invalid_triples": ".parsers",
    "parse_synonym_decisions": ".parsers",
    "parse_shacl_repairs": ".parsers",
    # Native DSPy Program & 12 Canonical Signatures
    "HippoRAGProgram": ".program",
    "RLMInstructionProposer": ".program",
    "YawGBNFAdapter": ".program",
    "YawPassage": ".program",
    "YawRM": ".program",
    "YawRetriever": ".program",
    "YawAssertionError": ".program",
    "yaw_assert": ".program",
    "yaw_suggest": ".program",
    "validate_stage_invariants": ".program",
    "wrap_stage_with_refine": ".program",
    "ParetoProgramEnsemble": ".program",
    "is_abstaining_answer": ".program",
    "ExtractEntities": ".prompts",
    "ExtractQueryIntent": ".prompts",
    "ExtractTriples": ".prompts",
    "DefineMissingEntities": ".prompts",
    "JudgeSynonyms": ".prompts",
    "ProposeSHACLRepairs": ".prompts",
    "HealSHACLViolations": ".prompts",
    "FilterTriples": ".prompts",
    "AnswerQuestion": ".prompts",
    "ReflectiveInnerDistillerSignature": ".prompts",
    "ProposeInstructionSignature": ".prompts",
    "CompressInstructionSignature": ".prompts",
    "FactList": ".prompts",
    # Pure-JSON Context-Free Grammars & Sizecoding Metric
    "compute_sizecoding_pareto_score": ".grammars",
    "build_dspy_ner_grammar": ".grammars",
    "build_dspy_triples_grammar": ".grammars",
    "build_dspy_missing_entity_defs_grammar": ".grammars",
    "build_dspy_synonym_judge_grammar": ".grammars",
    "build_dspy_synonym_judgment_grammar": ".grammars",
    "build_dspy_shacl_repair_grammar": ".grammars",
    "build_dspy_rerank_grammar": ".grammars",
    "build_dspy_qa_grammar": ".grammars",
    "build_dspy_instruction_proposal_grammar": ".grammars",
    "build_dspy_query_intent_grammar": ".grammars",
    "DSPY_NER_GBNF_GRAMMAR": ".grammars",
    "DSPY_TRIPLES_GBNF_GRAMMAR": ".grammars",
    "DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR": ".grammars",
    "DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR": ".grammars",
    "DSPY_SYNONYM_JUDGMENT_GBNF_GRAMMAR": ".grammars",
    "DSPY_SHACL_REPAIR_GBNF_GRAMMAR": ".grammars",
    "DSPY_RERANK_GBNF_GRAMMAR": ".grammars",
    "DSPY_QA_GBNF_GRAMMAR": ".grammars",
    "DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR": ".grammars",
    "DSPY_QUERY_INTENT_GBNF_GRAMMAR": ".grammars",
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
