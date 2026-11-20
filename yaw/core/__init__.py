"""Core package: Frozen data contracts, configuration, pure coordinate/ID helpers, and logging utilities."""

import importlib
from typing import Any

_PUBLIC_API: dict[str, str] = {
    "ChunkPointer": ".types",
    "DEFAULT_EXCLUDED_PATTERNS": ".config",
    "DEFAULT_INCLUDED_PATTERNS": ".config",
    "DaemonStatus": ".types",
    "EmbedFn": ".types",
    "ErrorCode": ".types",
    "FIFOLineRotatingHandler": ".logger",
    "HydratedPassage": ".types",
    "QAResult": ".types",
    "YawConfig": ".config",
    "YawRuntimeError": ".types",
    "compute_chunk_int_id": ".types",
    "configure_logging": ".logger",
    "format_doc_for_embedding": ".types",
    "format_query_for_embedding": ".types",
    "get_config_value": ".config",
    "get_logger": ".logger",
    "inclusive_end_line": ".types",
    "load_config": ".config",
    "normalize_entity": ".types",
    "parse_dataset_list": ".config",
    "redact_secrets": ".types",
    "render_document_text": ".types",
    "resolve_coco_state_root": ".config",
    "resolve_corpus_path": ".types",
    "set_config_value": ".config",
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
