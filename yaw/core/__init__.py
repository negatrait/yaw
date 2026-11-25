"""Core package: Frozen data contracts, configuration, pure coordinate/ID helpers, and logging utilities."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import (
        DEFAULT_EXCLUDED_PATTERNS,
        DEFAULT_INCLUDED_PATTERNS,
        YawConfig,
        get_config_value,
        load_config,
        parse_dataset_list,
        resolve_coco_state_root,
        set_config_value,
        write_text_atomic,
        write_via_temp_atomic,
    )
    from .logger import FIFOLineRotatingHandler, configure_logging, get_logger
    from .types import (
        DASH_RE,
        WIKI_REL_PREFIX,
        ChunkPointer,
        DaemonStatus,
        EmbedFn,
        ErrorCode,
        HydratedPassage,
        QAResult,
        YawRuntimeError,
        advance_markdown_fence,
        compute_chunk_int_id,
        format_doc_for_embedding,
        format_query_for_embedding,
        inclusive_end_line,
        is_wiki_rel_path,
        leading_frontmatter_end,
        make_wiki_rel_path,
        markdown_heading_level,
        normalize_ast_heading,
        redact_secrets,
        render_document_text,
        require_corpus_dir,
        resolve_corpus_path,
        resolve_source_path,
        split_lf_lines,
        strip_wiki_rel_path,
    )

_PUBLIC_API: dict[str, str] = {
    "ChunkPointer": ".types",
    "DASH_RE": ".types",
    "DEFAULT_EXCLUDED_PATTERNS": ".config",
    "DEFAULT_INCLUDED_PATTERNS": ".config",
    "DaemonStatus": ".types",
    "EmbedFn": ".types",
    "ErrorCode": ".types",
    "FIFOLineRotatingHandler": ".logger",
    "HydratedPassage": ".types",
    "QAResult": ".types",
    "WIKI_REL_PREFIX": ".types",
    "YawConfig": ".config",
    "YawRuntimeError": ".types",
    "advance_markdown_fence": ".types",
    "compute_chunk_int_id": ".types",
    "configure_logging": ".logger",
    "format_doc_for_embedding": ".types",
    "format_query_for_embedding": ".types",
    "get_config_value": ".config",
    "get_logger": ".logger",
    "inclusive_end_line": ".types",
    "is_wiki_rel_path": ".types",
    "leading_frontmatter_end": ".types",
    "load_config": ".config",
    "make_wiki_rel_path": ".types",
    "markdown_heading_level": ".types",
    "normalize_ast_heading": ".types",
    "parse_dataset_list": ".config",
    "redact_secrets": ".types",
    "render_document_text": ".types",
    "require_corpus_dir": ".types",
    "resolve_coco_state_root": ".config",
    "resolve_corpus_path": ".types",
    "resolve_source_path": ".types",
    "set_config_value": ".config",
    "split_lf_lines": ".types",
    "strip_wiki_rel_path": ".types",
    "write_text_atomic": ".config",
    "write_via_temp_atomic": ".config",
}

__all__ = [
    "ChunkPointer",
    "DASH_RE",
    "DEFAULT_EXCLUDED_PATTERNS",
    "DEFAULT_INCLUDED_PATTERNS",
    "DaemonStatus",
    "EmbedFn",
    "ErrorCode",
    "FIFOLineRotatingHandler",
    "HydratedPassage",
    "QAResult",
    "WIKI_REL_PREFIX",
    "YawConfig",
    "YawRuntimeError",
    "advance_markdown_fence",
    "compute_chunk_int_id",
    "configure_logging",
    "format_doc_for_embedding",
    "format_query_for_embedding",
    "get_config_value",
    "get_logger",
    "inclusive_end_line",
    "is_wiki_rel_path",
    "leading_frontmatter_end",
    "load_config",
    "make_wiki_rel_path",
    "markdown_heading_level",
    "normalize_ast_heading",
    "parse_dataset_list",
    "redact_secrets",
    "render_document_text",
    "require_corpus_dir",
    "resolve_coco_state_root",
    "resolve_corpus_path",
    "resolve_source_path",
    "set_config_value",
    "split_lf_lines",
    "strip_wiki_rel_path",
    "write_text_atomic",
    "write_via_temp_atomic",
]


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
