"""Textbook CocoIndex v1 incremental line-pointer ingestion pipeline."""

import bisect
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectorkits import statediff
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import localfs
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from markdown_it import MarkdownIt

from yaw.core import (
    ChunkPointer,
    SyncDelta,
    compute_chunk_id,
    inclusive_end_line,
)

_MD_PARSER = MarkdownIt("commonmark")
_DEFAULT_INCLUDED_PATTERNS: tuple[str, ...] = ("*.md", "*.txt")
_DEFAULT_EXCLUDED_PATTERNS: tuple[str, ...] = (".*", "*/.*", "__pycache__/*")

_ChunkValue = tuple[ChunkPointer, str]
_ChunkAction = tuple[str, _ChunkValue | None]

_SINK_KEY = coco.ContextKey[list[_ChunkAction]]("yaw_chunk_sink")
_CORPUS_ROOT_KEY = coco.ContextKey[Path]("yaw_corpus_root")
_COMPILER_KEY = coco.ContextKey[Callable[[SyncDelta], None] | None]("yaw_delta_compiler")
_COMPILER_ERROR_KEY = coco.ContextKey[list[Exception]]("yaw_delta_compiler_errors")


class _ChunkTargetHandler(coco.TargetHandler[_ChunkValue, bytes]):
    """CocoIndex target handler managing incremental chunk lifecycle via statediff."""

    def __init__(self) -> None:
        self._sink = coco.TargetActionSink.from_fn(self._apply_actions)

    def _apply_actions(
        self, context_provider: coco.ContextProvider, actions: Sequence[_ChunkAction]
    ) -> None:
        compiler = context_provider.get(_COMPILER_KEY)
        if compiler is not None and actions:
            added_chunks: list[tuple[ChunkPointer, str]] = []
            deleted_chunk_ids: list[str] = []
            for chunk_id, value in actions:
                if value is not None:
                    ptr, chunk_text = value
                    added_chunks.append((ptr, chunk_text))
                else:
                    deleted_chunk_ids.append(chunk_id)
            delta = SyncDelta(
                added_chunks=tuple(added_chunks),
                deleted_chunk_ids=tuple(deleted_chunk_ids),
            )
            try:
                compiler(delta)
            except Exception as exc:
                context_provider.get(_COMPILER_ERROR_KEY).append(exc)
                raise
        context_provider.get(_SINK_KEY).extend(actions)

    def reconcile(
        self,
        key: coco.StableKey,
        desired_target_state: _ChunkValue | coco.NonExistenceType,
        prev_possible_records: Collection[bytes],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[Any, Any, Any] | None:
        desired_fp: bytes | coco.NonExistenceType = (
            coco.NON_EXISTENCE
            if coco.is_non_existence(desired_target_state)
            else fingerprint_object(desired_target_state)
        )
        action = statediff.diff(
            statediff.TrackingRecordTransition(
                desired=desired_fp,
                prev=prev_possible_records,
                prev_may_be_missing=prev_may_be_missing,
            )
        )
        if action is None:
            return None

        chunk_id = str(key)
        if action == "delete":
            return coco.TargetReconcileOutput(
                action=(chunk_id, None),
                sink=self._sink,
                tracking_record=coco.NON_EXISTENCE,
            )

        assert not coco.is_non_existence(desired_target_state)
        assert not coco.is_non_existence(desired_fp)
        return coco.TargetReconcileOutput(
            action=(chunk_id, desired_target_state),
            sink=self._sink,
            tracking_record=desired_fp,
        )


_CHUNK_HANDLER = _ChunkTargetHandler()
_CHUNK_PROVIDER = coco.register_root_target_states_provider("yaw/corpus/chunk", _CHUNK_HANDLER)


def _declare_chunk(key: coco.StableKey, value: _ChunkValue) -> None:
    target_state = _CHUNK_PROVIDER.target_state(key, value)
    coco.declare_target_state(target_state)


def extract_markdown_headings(content: str) -> list[tuple[int, str]]:
    """Parse CommonMark tokens and return sorted [(1_indexed_start_line, title)] headings.

    Ignores '#' comments inside fenced code blocks and supports ATX and Setext headings.
    """
    clean_text = content.replace("\r\n", "\n")
    tokens = _MD_PARSER.parse(clean_text)
    headings: list[tuple[int, str]] = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.map:
            start_line_1_indexed = token.map[0] + 1
            title_text = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title_text = tokens[i + 1].content.strip()
            headings.append((start_line_1_indexed, title_text))
    headings.sort(key=lambda h: h[0])
    return headings


def resolve_heading_for_line(
    headings: Sequence[tuple[int, str]],
    start_line: int,
    fallback_title: str = "",
) -> str:
    """Find the nearest heading on or before start_line using O(log H) binary search."""
    if not headings:
        return fallback_title
    line_keys = [h[0] for h in headings]
    idx = bisect.bisect_right(line_keys, start_line)
    if idx == 0:
        return fallback_title
    heading_title = headings[idx - 1][1]
    return heading_title if heading_title else fallback_title


def extract_chunk_title(content: str, start_line: int, rel_posix_path: str) -> str:
    """Extract closest preceding Markdown heading or fallback to file stem."""
    headings = extract_markdown_headings(content)
    fallback_title = Path(rel_posix_path).stem
    return resolve_heading_for_line(headings, start_line, fallback_title)


def chunk_markdown_text(
    content: str,
    rel_posix_path: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> tuple[tuple[ChunkPointer, str], ...]:
    """Pure syntax-aware text splitting returning 1-indexed ChunkPointers and passage texts."""
    clean_text = content.replace("\r\n", "\n")
    if not clean_text.strip():
        return ()

    headings = extract_markdown_headings(clean_text)
    fallback_title = Path(rel_posix_path).stem

    lang = detect_code_language(filename=rel_posix_path)
    splitter = RecursiveSplitter()
    chunks = splitter.split(
        clean_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min(min_chunk_size, len(clean_text.strip())),
        language=lang,
    )

    if not chunks:
        lines = clean_text.splitlines()
        end_line = max(1, len(lines))
        title = resolve_heading_for_line(headings, 1, fallback_title)
        cid = compute_chunk_id(rel_posix_path, 1, end_line, clean_text)
        ptr = ChunkPointer(
            chunk_id=cid,
            rel_posix_path=rel_posix_path,
            start_line=1,
            end_line=end_line,
            title=title,
        )
        return ((ptr, clean_text),)

    results: list[tuple[ChunkPointer, str]] = []
    for c in chunks:
        start_line = c.start.line
        end_line = inclusive_end_line(c.start.line, c.end.line, c.end.column)
        title = resolve_heading_for_line(headings, start_line, fallback_title)
        cid = compute_chunk_id(rel_posix_path, start_line, end_line, c.text)
        ptr = ChunkPointer(
            chunk_id=cid,
            rel_posix_path=rel_posix_path,
            start_line=start_line,
            end_line=end_line,
            title=title,
        )
        results.append((ptr, c.text))

    return tuple(results)


@coco.fn(memo=True)
async def process_corpus_file(
    file: FileLike,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> None:
    """Memoized per-file processing component declaring chunks as target state."""
    if file.file_path.path.is_absolute():
        raise ValueError(
            f"Expected relative FilePath from localfs.walk_dir, got absolute path: {file.file_path.path}"
        )
    rel_posix_path = file.file_path.path.as_posix()
    text = await file.read_text()

    chunks = chunk_markdown_text(
        text,
        rel_posix_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    for ptr, chunk_text in chunks:
        _declare_chunk(ptr.chunk_id, (ptr, chunk_text))


@coco.fn
async def corpus_app_main(
    included_patterns: tuple[str, ...],
    excluded_patterns: tuple[str, ...],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> None:
    """Top-level CocoIndex application function walking corpus_root and mounting files."""
    files = localfs.walk_dir(
        _CORPUS_ROOT_KEY,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=list(included_patterns),
            excluded_patterns=list(excluded_patterns),
        ),
    )
    await coco.mount_each(
        process_corpus_file,
        files.items(),
        chunk_size,
        chunk_overlap,
        min_chunk_size,
    )


class CorpusIngestor:
    """Incremental line-pointer corpus ingestor backed by CocoIndex v1."""

    def __init__(
        self,
        corpus_root: Path | str,
        state_dir: Path | str,
        *,
        included_patterns: Sequence[str] = _DEFAULT_INCLUDED_PATTERNS,
        excluded_patterns: Sequence[str] = _DEFAULT_EXCLUDED_PATTERNS,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        on_delta: Callable[[SyncDelta], None] | None = None,
    ) -> None:
        self.corpus_root = Path(corpus_root).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.corpus_root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.included_patterns = tuple(included_patterns)
        self.excluded_patterns = tuple(excluded_patterns)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._on_delta = on_delta

        self._actions_list: list[_ChunkAction] = []
        self._compiler_errors: list[Exception] = []
        self._cp = coco.ContextProvider()
        self._cp.provide(_SINK_KEY, self._actions_list)
        self._cp.provide(_CORPUS_ROOT_KEY, self.corpus_root)
        self._cp.provide(_COMPILER_KEY, self._on_delta)
        self._cp.provide(_COMPILER_ERROR_KEY, self._compiler_errors)

        lmdb_path = self.state_dir / "coco_lmdb"
        self._env = coco.Environment(
            settings=coco.Settings.from_env(lmdb_path),
            context_provider=self._cp,
        )
        self._app = coco.App(
            coco.AppConfig(name="yaw_corpus", environment=self._env),
            corpus_app_main,
            included_patterns=self.included_patterns,
            excluded_patterns=self.excluded_patterns,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
        )

    def set_delta_compiler(self, compiler: Callable[[SyncDelta], None] | None) -> None:
        """Bind or update the target sink delta compiler callback."""
        self._on_delta = compiler
        self._cp.provide(_COMPILER_KEY, compiler)

    def close(self) -> None:
        """Close the underlying CocoIndex environment and release LMDB resources."""
        self._env.close()

    def __enter__(self) -> "CorpusIngestor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def poll(self) -> SyncDelta:
        """Run incremental CocoIndex update and return coalesced batch SyncDelta."""
        self._actions_list.clear()
        self._compiler_errors.clear()
        self._app.update_blocking()
        if self._compiler_errors:
            raise self._compiler_errors[0]

        added_chunks: list[tuple[ChunkPointer, str]] = []
        deleted_chunk_ids: list[str] = []

        for chunk_id, value in self._actions_list:
            if value is not None:
                ptr, chunk_text = value
                added_chunks.append((ptr, chunk_text))
            else:
                deleted_chunk_ids.append(chunk_id)

        return SyncDelta(
            added_chunks=tuple(added_chunks),
            deleted_chunk_ids=tuple(deleted_chunk_ids),
        )


def sync_corpus_once(
    corpus_root: Path | str,
    state_dir: Path | str,
    **kwargs: Any,
) -> SyncDelta:
    """One-shot helper to initialize a CorpusIngestor and poll for changes once."""
    with CorpusIngestor(corpus_root, state_dir, **kwargs) as ingestor:
        return ingestor.poll()

