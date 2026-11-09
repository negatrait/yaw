"""Native SQLite source contributions, owned by CocoIndex file components."""

import hashlib
import inspect
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path, PurePath, PurePosixPath

import cocoindex as coco
from cocoindex.connectors import sqlite
from cocoindex.resources.file import FileLike, FilePathMatcher, PatternFilePathMatcher

from yaw.core import json_document_fields
from yaw.daemon.ingestion.corpus import chunk_source_text

SOURCE_DB = coco.ContextKey[sqlite.ManagedConnection]("yaw/source_db")
SOURCE_ROOT = coco.ContextKey[Path]("yaw/source_root", detect_change=True)


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Persistent source pointer; corpus text is not duplicated in this target."""

    chunk_id: str
    rel_posix_path: str
    source_hash: str
    start_line: int
    end_line: int
    title: str
    ast_island_start: int
    ast_island_end: int


@dataclass(frozen=True, slots=True)
class SourceSettings:
    included_patterns: tuple[str, ...]
    excluded_patterns: tuple[str, ...]
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    json_keys_whitelist: tuple[str, ...] = ("*",)
    json_keys_blacklist: tuple[str, ...] = ()
    included_files: tuple[str, ...] | None = None


class _SelectedFilesPathMatcher(FilePathMatcher):
    def __init__(self, pattern_matcher: PatternFilePathMatcher, included_files: tuple[str, ...]) -> None:
        self._pattern_matcher = pattern_matcher
        self._included_files = frozenset(included_files)
        self._included_dirs = frozenset(
            parent.as_posix() for rel in self._included_files for parent in PurePosixPath(rel).parents
        )

    def is_dir_included(self, path: PurePath) -> bool:
        return path.as_posix() in self._included_dirs and self._pattern_matcher.is_dir_included(path)

    def is_file_included(self, path: PurePath) -> bool:
        return path.as_posix() in self._included_files and self._pattern_matcher.is_file_included(path)


def build_source_path_matcher(settings: SourceSettings) -> FilePathMatcher:
    pattern_matcher = PatternFilePathMatcher(
        included_patterns=list(settings.included_patterns),
        excluded_patterns=list(settings.excluded_patterns),
    )
    if settings.included_files is None:
        return pattern_matcher
    return _SelectedFilesPathMatcher(pattern_matcher, settings.included_files)


def normalize_chunking_settings(settings: SourceSettings) -> SourceSettings:
    """Strip path-selection filters so file chunking memos depend only on chunking parameters."""
    if not settings.included_patterns and not settings.excluded_patterns and settings.included_files is None:
        return settings
    return replace(settings, included_patterns=(), excluded_patterns=(), included_files=None)


@coco.fn(
    memo=True,
    version=2,
    deps={
        "parser_source": hashlib.sha256(Path(inspect.getfile(chunk_source_text)).read_bytes()).hexdigest(),
        "contract_source": hashlib.sha256(Path(inspect.getfile(json_document_fields)).read_bytes()).hexdigest(),
        "cocoindex_version": version("cocoindex"),
    },
)
async def chunk_file_memo(file: FileLike, settings: SourceSettings) -> tuple[tuple[SourceMetadata, str], ...]:
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
    return tuple(
        (
            SourceMetadata(
                pointer.chunk_id,
                pointer.rel_posix_path,
                digest,
                pointer.start_line,
                pointer.end_line,
                pointer.title,
                pointer.ast_island_start,
                pointer.ast_island_end,
            ),
            text,
        )
        for pointer, text in chunks
    )


@coco.fn
async def declare_source_file(
    file: FileLike, table: sqlite.TableTarget[SourceMetadata], settings: SourceSettings
) -> tuple[tuple[SourceMetadata, str], ...]:
    """Failed parsing declares nothing; CocoIndex retains the previous targets."""
    contributions = await chunk_file_memo(file, normalize_chunking_settings(settings))
    for meta, _ in contributions:
        table.declare_row(row=meta)
    return contributions
