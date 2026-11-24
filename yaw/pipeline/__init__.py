"""Declarative live CocoIndex v1 pipeline indexing IEEE float16 chunks, unicode61 FTS5, and SoA artifacts."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .embedder import EMBEDDER, DspyBatchEmbedder, make_dspy_embed_fn
    from .flow import CORPUS_ROOT, SQLITE_DB, chunk_file_content
    from .session import (
        WORKER_ERROR_TYPES,
        PipelineSession,
        close_pipeline,
        start_pipeline,
        update_pipeline_once,
        wait_pipeline_ready,
    )
    from .targets import ChunkRow, SourceChunk, list_user_tables, open_managed_db, reconcile_index_tables

_PUBLIC_API: dict[str, str] = {
    "CORPUS_ROOT": ".flow",
    "ChunkRow": ".targets",
    "DspyBatchEmbedder": ".embedder",
    "EMBEDDER": ".embedder",
    "PipelineSession": ".session",
    "SQLITE_DB": ".flow",
    "SourceChunk": ".targets",
    "WORKER_ERROR_TYPES": ".session",
    "chunk_file_content": ".flow",
    "close_pipeline": ".session",
    "list_user_tables": ".targets",
    "make_dspy_embed_fn": ".embedder",
    "open_managed_db": ".targets",
    "reconcile_index_tables": ".targets",
    "start_pipeline": ".session",
    "update_pipeline_once": ".session",
    "wait_pipeline_ready": ".session",
}

__all__ = [
    "CORPUS_ROOT",
    "ChunkRow",
    "DspyBatchEmbedder",
    "EMBEDDER",
    "PipelineSession",
    "SQLITE_DB",
    "SourceChunk",
    "WORKER_ERROR_TYPES",
    "chunk_file_content",
    "close_pipeline",
    "list_user_tables",
    "make_dspy_embed_fn",
    "open_managed_db",
    "reconcile_index_tables",
    "start_pipeline",
    "update_pipeline_once",
    "wait_pipeline_ready",
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
