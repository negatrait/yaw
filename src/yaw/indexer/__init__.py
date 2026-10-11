"""Indexer domain: Single-pass Cocoindex vault indexing and chunk offsets."""

from __future__ import annotations

from yaw.indexer.canonicalize import (
    AliasStatus,
    MatchMethod,
    Resolution,
    ResolutionResult,
    SurfaceIdentity,
    TierEnum,
    create_in_memory_kb,
    resolve,
    sync_kb_from_db,
)
from yaw.indexer.pipeline import (
    LeafNodeRow,
    LeafRelationRow,
    MarkdownChunk,
    NoteChunkRow,
    NoteEdgeRow,
    NoteMetaRow,
    build_entity_uri,
    chunk_markdown,
    classify_freeform_llm,
    clean_edge_target,
    extract_title,
    extract_wikilinks,
    process_vault_raw,
    sync_vault,
)

__all__ = [
    "AliasStatus",
    "LeafNodeRow",
    "LeafRelationRow",
    "MarkdownChunk",
    "MatchMethod",
    "NoteChunkRow",
    "NoteEdgeRow",
    "NoteMetaRow",
    "Resolution",
    "ResolutionResult",
    "SurfaceIdentity",
    "TierEnum",
    "build_entity_uri",
    "chunk_markdown",
    "classify_freeform_llm",
    "clean_edge_target",
    "create_in_memory_kb",
    "extract_title",
    "extract_wikilinks",
    "process_vault_raw",
    "resolve",
    "sync_kb_from_db",
    "sync_vault",
]
