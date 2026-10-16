"""Ingestion leaf package: Native CocoIndex v1 incremental line-pointer ingestion."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    "CorpusIngestor": ".corpus",
    "chunk_markdown_text": ".corpus",
    "sync_corpus_once": ".corpus",
    "extract_chunk_title": ".corpus",
    "extract_markdown_headings": ".corpus",
    "resolve_heading_for_line": ".corpus",
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
    return list(__all__)
