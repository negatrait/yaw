"""Core package: Frozen data contracts, configuration, pure coordinate/ID helpers, and logging utilities."""

import importlib
from typing import Any

_PUBLIC_API: dict[str, str] = {
    "ChunkPointer": ".types",
    "DEFAULT_EXCLUDED_PATTERNS": ".config",
    "DEFAULT_INCLUDED_PATTERNS": ".config",
    "DaemonStatus": ".types",
    "ErrorCode": ".types",
    "FIFOLineRotatingHandler": ".logger",
    "FatalGraphPublicationError": ".types",
    "HydratedPassage": ".types",
    "IndexStats": ".types",
    "LABEL_KIND_PRIORITY": ".constants",
    "OpenIERecord": ".types",
    "RelationTraversalPolicy": ".types",
    "RetrievalResult": ".types",
    "SCHEMA_PLACEHOLDER_DEFS": ".constants",
    "SEMANTIC_EDGE_KINDS": ".constants",
    "SHACLViolation": ".types",
    "SKOS_LABEL_KINDS": ".constants",
    "SYNONYM_RELATION_ALIASES": ".constants",
    "StructuredTriple": ".types",
    "SynonymEvidenceBatch": ".types",
    "VALID_SHACL_ACTIONS": ".constants",
    "VALID_SYNONYM_RELATIONS": ".constants",
    "VALUE_BEARING_SHACL_ACTIONS": ".constants",
    "YawConfig": ".config",
    "YawRuntimeError": ".types",
    "bayesian_noisy_or": ".types",
    "clean_schema_definition": ".constants",
    "clean_surface_label": ".types",
    "compute_chunk_id": ".types",
    "compute_mdhash_id": ".types",
    "configure_logging": ".logger",
    "detect_scalar_literal": ".types",
    "get_config_value": ".config",
    "get_logger": ".logger",
    "inclusive_end_line": ".types",
    "is_irreflexive_statement": ".types",
    "is_resource_label": ".types",
    "is_scalar_label": ".types",
    "json_document_fields": ".types",
    "load_config": ".config",
    "parse_dataset_list": ".config",
    "pool_confidences": ".types",
    "resolve_corpus_path": ".types",
    "render_document_text": ".types",
    "set_config_value": ".config",
    "text_processing": ".types",
    "to_rel_posix_path": ".types",
    "write_text_atomic": ".config",
    "write_via_temp_atomic": ".config",
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
