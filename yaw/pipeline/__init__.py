"""Declarative live CocoIndex v1 pipeline indexing IEEE float16 chunks, unicode61 FTS5, and SoA artifacts."""

import importlib
import os
from typing import TYPE_CHECKING, Any

os.environ.setdefault("COCOINDEX_DISABLE_USAGE_TRACKING", "1")

if TYPE_CHECKING:
    from .embedder import EMBEDDER, DspyBatchEmbedder, make_dspy_embed_fn
    from .flow import CORPUS_ROOT, SQLITE_DB, WIKI_ROOT, build_wiki_path_matcher, chunk_file_content
    from .session import (
        WORKER_ERROR_TYPES,
        PipelineSession,
        aupdate_pipeline_once,
        close_pipeline,
        drop_pipeline,
        open_pipeline_lifespan,
        start_pipeline,
        update_pipeline_once,
    )
    from .targets import (
        SINK_TRACKER,
        ChunkRow,
        SinkMutationTracker,
        SourceChunk,
        list_user_tables,
        open_managed_db,
        reconcile_index_tables,
    )

_PUBLIC_API: dict[str, str] = {
    "CORPUS_ROOT": ".flow",
    "ChunkRow": ".targets",
    "DspyBatchEmbedder": ".embedder",
    "EMBEDDER": ".embedder",
    "PipelineSession": ".session",
    "SINK_TRACKER": ".targets",
    "SQLITE_DB": ".flow",
    "SinkMutationTracker": ".targets",
    "SourceChunk": ".targets",
    "WIKI_ROOT": ".flow",
    "WORKER_ERROR_TYPES": ".session",
    "aupdate_pipeline_once": ".session",
    "build_wiki_path_matcher": ".flow",
    "chunk_file_content": ".flow",
    "close_pipeline": ".session",
    "drop_pipeline": ".session",
    "list_user_tables": ".targets",
    "make_dspy_embed_fn": ".embedder",
    "open_managed_db": ".targets",
    "open_pipeline_lifespan": ".session",
    "reconcile_index_tables": ".targets",
    "start_pipeline": ".session",
    "update_pipeline_once": ".session",
}

__all__ = [
    "CORPUS_ROOT",
    "ChunkRow",
    "DspyBatchEmbedder",
    "EMBEDDER",
    "PipelineSession",
    "SINK_TRACKER",
    "SQLITE_DB",
    "SinkMutationTracker",
    "SourceChunk",
    "WIKI_ROOT",
    "WORKER_ERROR_TYPES",
    "aupdate_pipeline_once",
    "build_wiki_path_matcher",
    "chunk_file_content",
    "close_pipeline",
    "drop_pipeline",
    "list_user_tables",
    "make_dspy_embed_fn",
    "open_managed_db",
    "open_pipeline_lifespan",
    "reconcile_index_tables",
    "start_pipeline",
    "update_pipeline_once",
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
