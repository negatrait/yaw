"""Zero-copy disk hydration, SHA-256 freshness verification, and MaxP span deduplication."""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yaw.core import ChunkPointer, HydratedPassage, YawRuntimeError, resolve_corpus_path

from .types import CACHE_LOCK, CACHE_MAX_ENTRIES, RetrievalIndex

_MAX_HYDRATE_SPAN = 10_000
_TITLE_TRAILING_SEP_RE = re.compile(r"\s+[-–—:]+\s*$")
_TITLE_TAG_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s+(.+)$")

type _FileCache = dict[str, tuple[Path, str, list[str]] | None]


def extract_markdown_h1(raw_head: str) -> str:
    """Extract normalized title text from a leading Markdown # Heading line."""
    first = next((ln.strip() for ln in raw_head.splitlines() if ln.strip()), "")
    if not first.startswith("#"):
        return ""
    title = _TITLE_TRAILING_SEP_RE.sub("", first.lstrip("#").strip()).strip()
    if m := _TITLE_TAG_PREFIX_RE.match(title):
        raw_grp = m.group(1)
        if isinstance(raw_grp, str) and (stripped := raw_grp.strip()):
            title = stripped
    return title


def pack_doc_titles_blob(doc_titles: Mapping[str, str]) -> bytes:
    """Serialize and zlib-compress the relative-path to title mapping."""
    payload = json.dumps(dict(doc_titles), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return zlib.compress(payload, level=6)


def build_chunk_pointers(
    rows: Sequence[Any],
) -> tuple[list[int], list[ChunkPointer], list[str], NDArray[np.uint32]]:
    """Build ordered chunk IDs, ChunkPointers, distinct POSIX paths, and path indices."""
    chunk_ids: list[int] = []
    pointers: list[ChunkPointer] = []
    distinct_paths: list[str] = []
    path_to_idx: dict[str, int] = {}
    path_idx_arr = np.zeros(len(rows), dtype=np.uint32)
    for pos, r in enumerate(rows):
        cid = int(r[0])
        rel = str(r[2]).replace("\\", "/")
        p_idx = path_to_idx.get(rel)
        if p_idx is None:
            p_idx = len(distinct_paths)
            path_to_idx[rel] = p_idx
            distinct_paths.append(rel)
        path_idx_arr[pos] = p_idx
        chunk_ids.append(cid)
        pointers.append(
            ChunkPointer(
                chunk_id=cid,
                rel_path=str(r[2]),
                source_hash=str(r[3]),
                start_line=int(r[4]),
                end_line=int(r[5]),
                char_start=int(r[6]),
                char_end=int(r[7]),
                ast_island_start=int(r[8]),
                ast_island_end=int(r[9]),
                title=str(r[10]),
            )
        )
    return chunk_ids, pointers, distinct_paths, path_idx_arr


def _validate_max_chars(max_chars: int | None) -> int | None:
    if max_chars is None:
        return None
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError(f"max_chars must be a non-negative int or None, got {max_chars!r}")
    return max_chars


def _crop_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and max_chars >= 120:
        half = max(20, (max_chars - 48) // 2)
        head: list[str] = []
        used = 0
        for ln in lines:
            if head and used + len(ln) + 1 > half:
                break
            head.append(ln)
            used += len(ln) + 1
        tail: list[str] = []
        used_t = 0
        for ln in reversed(lines[len(head) :]):
            if tail and used_t + len(ln) + 1 > half:
                break
            tail.append(ln)
            used_t += len(ln) + 1
        tail.reverse()
        omitted_start = len(head) + 1
        omitted_end = len(lines) - len(tail)
        if omitted_end >= omitted_start and head and tail:
            marker = f"[... lines {omitted_start}-{omitted_end} omitted ...]"
            candidate = "\n".join([*head, marker, *tail])
            if len(candidate) <= max_chars:
                return candidate
    marker_c = "\n[... AST section cropped ...]\n"
    if max_chars <= len(marker_c) + 8:
        return text[:max_chars]
    rem = max_chars - len(marker_c)
    left = rem // 2
    right = rem - left
    return f"{text[:left]}{marker_c}{text[-right:]}"


def _resolve_cached_path(
    corpus_root: Path,
    rel_posix_path: str,
    path_cache: dict[tuple[Path, str], Path] | None,
) -> Path:
    if path_cache is None:
        return resolve_corpus_path(corpus_root, rel_posix_path)
    p_key = (corpus_root, rel_posix_path)
    with CACHE_LOCK:
        cached = path_cache.get(p_key)
    if cached is not None:
        return cached
    target = resolve_corpus_path(corpus_root, rel_posix_path)
    with CACHE_LOCK:
        path_cache[p_key] = target
        while len(path_cache) > CACHE_MAX_ENTRIES:
            path_cache.pop(next(iter(path_cache)))
    return target


def _load_corpus_file(
    corpus_root: Path,
    rel_posix_path: str,
    file_cache: _FileCache | None,
    path_cache: dict[tuple[Path, str], Path] | None,
) -> tuple[Path, str, list[str]] | None:
    if file_cache is not None and rel_posix_path in file_cache:
        return file_cache[rel_posix_path]
    target = _resolve_cached_path(corpus_root, rel_posix_path, path_cache)
    if not target.is_file():
        loaded: tuple[Path, str, list[str]] | None = None
    else:
        try:
            text = target.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, UnicodeDecodeError) as exc:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR", f"Invalid UTF-8 or unreadable source file {target}: {exc}"
            ) from exc
        loaded = (target, text, text.splitlines())
    if file_cache is not None:
        file_cache[rel_posix_path] = loaded
    return loaded


def _render_hydrated_slice(
    ptr: ChunkPointer,
    text: str,
    lines: list[str],
    raw_slice: str,
    include_island: bool,
    full_span: bool,
    limit: int | None,
) -> str:
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if include_island and 0 < ptr.ast_island_start <= ptr.ast_island_end:
        es, ee = min(sl, int(ptr.ast_island_start)), min(len(lines), max(el, int(ptr.ast_island_end)))
        rendered = "\n".join(lines[es - 1 : ee])
    elif not full_span and ptr.char_end > ptr.char_start:
        rendered = text[ptr.char_start : ptr.char_end]
    else:
        rendered = raw_slice
    if (
        ptr.title
        and not rendered.lstrip().startswith("#")
        and not rendered.startswith(f"{ptr.title}\n")
        and rendered != ptr.title
    ):
        rendered = f"{ptr.title}\n{rendered}"
    if limit is not None and limit > 0:
        rendered = _crop_text(rendered, limit)
    return rendered


def hydrate_one(
    corpus_root: Path,
    ptr: ChunkPointer,
    score: float,
    max_chars: int | None,
    include_island: bool,
    full_span: bool,
    *,
    strict: bool = False,
    file_cache: _FileCache | None = None,
    path_cache: dict[tuple[Path, str], Path] | None = None,
) -> HydratedPassage:
    """Hydrate one verified slice from disk, enforcing line bounds and SHA-256 freshness."""
    limit = _validate_max_chars(max_chars)
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if sl < 1 or el < sl or (el - sl + 1) > _MAX_HYDRATE_SPAN:
        raise ValueError(f"Invalid hydration line range [{sl}, {el}] for {ptr.rel_posix_path}")

    loaded = _load_corpus_file(corpus_root, ptr.rel_posix_path, file_cache, path_cache)
    if loaded is None:
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Missing corpus source file: {ptr.rel_posix_path}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    target, text, lines = loaded
    if el > len(lines):
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Stale line range [{sl}, {el}] in {target}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)
    raw_slice = "\n".join(lines[sl - 1 : el])
    if ptr.source_hash and hashlib.sha256(raw_slice.encode("utf-8")).hexdigest() != ptr.source_hash:
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"SHA-256 drift detected on disk for {target}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    rendered = _render_hydrated_slice(ptr, text, lines, raw_slice, include_island, full_span, limit)
    return HydratedPassage(ptr, rendered, float(score), stale_on_disk=False)


def hydrate_passage(
    ptr: ChunkPointer,
    corpus_root: Path,
    *,
    score: float = 0.0,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> HydratedPassage:
    """Hydrate a single ChunkPointer directly from disk, verifying SHA-256 freshness."""
    return hydrate_one(corpus_root, ptr, score, max_chars, include_island, full_span, strict=False)


def dedup_ranked_positions(
    idx: RetrievalIndex,
    ranked_chunks: Sequence[tuple[int, float]],
    *,
    limit: int | None = None,
) -> tuple[tuple[int, int, float], ...]:
    """Apply MaxP 5-tuple span deduplication without opening or reading files on disk."""
    if not ranked_chunks or (limit is not None and limit <= 0):
        return ()
    out: list[tuple[int, int, float]] = []
    seen_spans: set[tuple[str, int, int, int, int]] = set()
    for cid, score in ranked_chunks:
        cid_int = int(cid)
        if (pos := idx.id_to_pos.get(cid_int)) is None:
            continue
        ptr = idx.pointers[pos]
        span = (ptr.rel_posix_path, int(ptr.start_line), int(ptr.end_line), int(ptr.char_start), int(ptr.char_end))
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append((pos, cid_int, float(score)))
        if limit is not None and len(out) >= limit:
            break
    return tuple(out)


def hydrate_ranked_passages(
    idx: RetrievalIndex,
    ranked_chunks: Sequence[tuple[int, float]],
    corpus_root: Path,
    *,
    top_k: int = 5,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> tuple[HydratedPassage, ...]:
    """Apply MaxP span deduplication and hydrate up to `top_k` passages from disk."""
    limit = _validate_max_chars(max_chars)
    deduped = dedup_ranked_positions(idx, ranked_chunks, limit=top_k)
    if not deduped:
        return ()
    file_cache: _FileCache = {}
    return tuple(
        hydrate_one(
            corpus_root,
            idx.pointers[pos],
            score,
            limit,
            include_island,
            full_span,
            strict=False,
            file_cache=file_cache,
            path_cache=idx.resolved_path_cache,
        )
        for pos, _cid, score in deduped
    )


def format_evidence_route(passages: Sequence[HydratedPassage]) -> str:
    """Format hydrated passages with explicit provenance headers for recall and ReAct output."""
    blocks: list[str] = []
    for idx, hp in enumerate(passages, start=1):
        if hp.stale_on_disk:
            blocks.append(
                f"[{idx}] [STALE: {hp.pointer.span_label} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]"
            )
        else:
            blocks.append(
                f"[{idx}] [{hp.pointer.span_label} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]\n{hp.text}"
            )
    return "\n\n".join(blocks)


def hydrate_corpus_chunks(
    pointers: Sequence[ChunkPointer],
    corpus_root: Path,
    *,
    extract_titles_for: Collection[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Hydrate all corpus chunks in one disk-read pass per distinct relative file path."""
    n = len(pointers)
    if n == 0:
        return [], {}
    c_root = corpus_root if corpus_root.is_absolute() else corpus_root.resolve()
    title_targets = set(extract_titles_for)
    by_rel: dict[str, list[int]] = {}
    for pos, ptr in enumerate(pointers):
        by_rel.setdefault(ptr.rel_posix_path, []).append(pos)

    texts = [""] * n
    titles: dict[str, str] = {}

    def _read_group(item: tuple[str, list[int]]) -> tuple[list[tuple[int, str]], str, str]:
        rel, pos_list = item
        local_cache: _FileCache = {}
        chunk_texts = [
            (
                pos,
                hydrate_one(
                    c_root,
                    pointers[pos],
                    1.0,
                    None,
                    False,
                    False,
                    strict=True,
                    file_cache=local_cache,
                ).text,
            )
            for pos in pos_list
        ]
        title = ""
        if rel in title_targets and (loaded := local_cache.get(rel)) is not None:
            first_line = next((ln for ln in loaded[2] if ln.strip()), "")
            if first_line:
                title = extract_markdown_h1(first_line)
        return chunk_texts, rel, title

    for grp_res, rel, title in map(_read_group, by_rel.items()):
        for pos, txt in grp_res:
            texts[pos] = txt
        if title:
            titles[rel] = title
    return texts, titles
