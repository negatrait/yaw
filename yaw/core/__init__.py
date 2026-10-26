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
    "HydratedPassage": ".types",
    "IndexStats": ".types",
    "OpenIERecord": ".types",
    "RelationTraversalPolicy": ".types",
    "RetrievalResult": ".types",
    "SHACLViolation": ".types",
    "StructuredTriple": ".types",
    "SyncDelta": ".types",
    "YawConfig": ".config",
    "YawRuntimeError": ".types",
    "check_lexical_subsumption_inversion": ".types",
    "compute_chunk_id": ".types",
    "compute_mdhash_id": ".types",
    "configure_logging": ".logger",
    "detect_scalar_literal": ".types",
    "get_config_value": ".config",
    "inclusive_end_line": ".types",
    "is_acronym_pair": ".types",
    "is_version_or_ordinal_sibling": ".types",
    "load_config": ".config",
    "resolve_corpus_path": ".types",
    "set_config_value": ".config",
    "text_processing": ".types",
    "to_rel_posix_path": ".types",
    "write_text_atomic": ".config",
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
