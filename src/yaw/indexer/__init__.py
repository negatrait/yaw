"""Indexer domain: Single-pass Cocoindex vault indexing and chunk offsets."""

from __future__ import annotations

from yaw.indexer.canonicalize import (
    AliasStatus,
    CanonicalVerdict,
    MatchMethod,
    Resolution,
    ResolutionResult,
    SurfaceIdentity,
    TierEnum,
    resolve,
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
    "CanonicalVerdict",
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
    "extract_title",
    "extract_wikilinks",
    "process_vault_raw",
    "resolve",
    "sync_vault",
]
