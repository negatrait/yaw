"""Native SQLite source contributions, owned by CocoIndex file components."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.daemon.ingestion.corpus import chunk_source_text

SOURCE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/source_db")
SOURCE_ROOT = coco.ContextKey[Path]("yaw/source_root", detect_change=True)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """Source evidence, distinct from the subsequently published graph."""

    chunk_id: str
    rel_posix_path: str
    source_hash: str
    start_line: int
    end_line: int
    title: str
    ast_island_start: int
    ast_island_end: int
    text: str


@dataclass(frozen=True, slots=True)
class SourceSettings:
    included_patterns: tuple[str, ...]
    excluded_patterns: tuple[str, ...]
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    json_keys_whitelist: tuple[str, ...] = ("*",)
    json_keys_blacklist: tuple[str, ...] = ()


@coco.fn(memo=True)
async def declare_source_file(
    file: FileLike, table: sqlite.TableTarget[SourceChunk], settings: SourceSettings
) -> tuple[SourceChunk, ...]:
    """Failed parsing declares nothing; CocoIndex retains the previous targets."""
    path = file.file_path.path
    if path.is_absolute():
        raise ValueError("Source contribution paths must be relative")
    content = await file.read_text()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    chunks = chunk_source_text(
        content,
        path.as_posix(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
        json_keys_whitelist=settings.json_keys_whitelist,
        json_keys_blacklist=settings.json_keys_blacklist,
    )
    contributions = tuple(
        SourceChunk(
            pointer.chunk_id,
            pointer.rel_posix_path,
            digest,
            pointer.start_line,
            pointer.end_line,
            pointer.title,
            pointer.ast_island_start,
            pointer.ast_island_end,
            text,
        )
        for pointer, text in chunks
    )
    for contribution in contributions:
        table.declare_row(row=contribution)
    return contributions


@coco.fn
async def source_app_main(settings: SourceSettings) -> None:
    """The same native file tree supports catch-up and live application runs."""
    table = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceChunk].from_class(SourceChunk, primary_key=["chunk_id"]),
    )
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        live=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=list(settings.included_patterns),
            excluded_patterns=list(settings.excluded_patterns),
        ),
    )
    await coco.mount_each(declare_source_file, files.items(), table, settings)
