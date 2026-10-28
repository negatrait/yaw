"""Inference leaf package: Native DSPy HippoRAG 2 engine, GBNF grammars, and legacy baseline adapters."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    "ActiveReflectionCheckpoint": ".reflection",
    "ActiveTraceComponentSelector": ".optimizer",
    "GepaProgress": ".optimizer",
    "GraphOptimizationHooks": ".optimizer",
    "candidate_fingerprint": ".optimizer",
    "checkpoint_object": ".reflection",
    "compile_pareto_ensemble": ".optimizer",
    "configure_simba_task_learning": ".optimizer",
    "embed_fingerprint": ".optimizer",
    "evaluate_program": ".optimizer",
    "evolve_program_prompts": ".optimizer",
    "extract_named_trace_entries": ".optimizer",
    "extraction_fingerprint": ".optimizer",
    "openie_fingerprint": ".optimizer",
    "promote_compiled_program": ".optimizer",
    "prune_mismatching_demos": ".optimizer",
    "query_retrieval_fingerprint": ".optimizer",
    "reduce_pareto_predictions": ".optimizer",
    "rrf_fuse_passage_lists": ".optimizer",
    "select_pareto_frontier_programs": ".optimizer",
    "serialize_predictor_surface": ".optimizer",
    "shacl_fingerprint": ".optimizer",
    "sync_pareto_telemetry_siblings": ".optimizer",
    "synonym_fingerprint": ".optimizer",
    "validate_active_checkpoint": ".reflection",
    "validate_request_checkpoint": ".reflection",
    "validate_simba_state": ".reflection",
    # Native DSPy LM & Heterogeneous RLM
    "create_dspy_lm": ".chat",
    "create_dspy_rlm_pair": ".chat",
    "configure_yaw_dspy_cache": ".chat",
    "set_yaw_sleep_seconds": ".chat",
    "trim_diagnostics": ".diagnostics",
    "lm_supports_gbnf": ".grammars",
    # Native DSPy Embedder & Profiles
    "create_dspy_embedder": ".embedding",
    "DSPyEmbeddingAdapter": ".embedding",
    "resolve_embedding_profile": ".embedding",
    "EmbeddingProfile": ".embedding",
    "EmbeddingRequest": ".embedding",
    "EmbeddingResult": ".embedding",
    "EmbeddingLane": ".embedding_lane",
    "PROFILES": ".embedding",
    "MODEL": ".embedding",
    "GEMMA_TASK_MODES": ".embedding",
    "FACT_INSTRUCTION": ".embedding",
    "PASSAGE_INSTRUCTION": ".embedding",
    # Stateless Parsers, Metrics & Hash Utilities
    "VALID_SYNONYM_RELATIONS": ".parsers",
    "VALID_SHACL_ACTIONS": ".parsers",
    "VALUE_BEARING_SHACL_ACTIONS": ".parsers",
    "check_benchmark_integrity": ".parsers",
    "compute_mdhash_id": ".parsers",
    "compute_qa_em_f1": ".parsers",
    "compute_task_quality_score": ".parsers",
    "detect_scalar_literal": ".parsers",
    "doc_matches_gold": ".parsers",
    "evaluate_benchmark_gate": ".parsers",
    "compute_retrieval_recalls": ".parsers",
    "compute_answer_passage_match": ".parsers",
    "text_processing": ".parsers",
    "extract_entity_definitions": ".parsers",
    "extract_structured_triples": ".parsers",
    "filter_invalid_triples": ".parsers",
    "normalize_answer": ".parsers",
    "parse_synonym_decisions": ".parsers",
    "parse_shacl_repairs": ".parsers",
    # Native DSPy Program & 12 Canonical Signatures
    "HippoRAGProgram": ".program",
    "RLMInstructionProposer": ".proposer",
    "ServingTokenCounter": ".reflection",
    "YawGBNFAdapter": ".adapter",
    "YawPassage": ".retriever",
    "YawRM": ".retriever",
    "YawRetriever": ".retriever",
    "validate_stage_invariants": ".invariants",
    "is_abstaining_answer": ".invariants",
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
    # Pure-JSON Context-Free Grammars
    "build_dspy_ner_grammar": ".grammars",
    "build_dspy_triples_grammar": ".grammars",
    "build_dspy_missing_entity_defs_grammar": ".grammars",
    "build_dspy_synonym_judge_grammar": ".grammars",
    "build_dspy_shacl_repair_grammar": ".grammars",
    "build_dspy_rerank_grammar": ".grammars",
    "build_dspy_qa_grammar": ".grammars",
    "build_dspy_instruction_proposal_grammar": ".grammars",
    "build_dspy_query_intent_grammar": ".grammars",
    "DSPY_NER_GBNF_GRAMMAR": ".grammars",
    "DSPY_TRIPLES_GBNF_GRAMMAR": ".grammars",
    "DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR": ".grammars",
    "DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR": ".grammars",
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
