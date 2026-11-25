"""Zero-copy disk hydration, CocoIndex fingerprint freshness verification, and MaxP island deduplication."""

from __future__ import annotations

import json
import zlib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from cocoindex.connectorkits.fingerprint import fingerprint_str
from numpy.typing import NDArray

from yaw.core import (
    DASH_RE,
    ChunkPointer,
    HydratedPassage,
    YawRuntimeError,
    advance_markdown_fence,
    is_wiki_rel_path,
    leading_frontmatter_end,
    markdown_heading_level,
    normalize_ast_heading,
    resolve_source_path,
    split_lf_lines,
)

from .types import CACHE_LOCK, CACHE_MAX_ENTRIES, RetrievalIndex

_MAX_HYDRATE_SPAN = 10_000
_MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdx")

type _FileCache = dict[str, tuple[Path, str, list[str]] | None]


def is_markdown_path(rel_path: str) -> bool:
    """Return True when `rel_path` has a recognized Markdown file suffix."""
    return rel_path.lower().endswith(_MARKDOWN_SUFFIXES)


def resolve_hydrated_line_span(
    ptr: ChunkPointer,
    *,
    include_island: bool,
    total_lines: int | None = None,
) -> tuple[int, int]:
    """Resolve effective 1-indexed inclusive `[es, ee]` hydration lines without widening sentinel `(1, 1)`."""
    sl, el = int(ptr.start_line), int(ptr.end_line)
    is_, ie_ = int(ptr.ast_island_start), int(ptr.ast_island_end)
    if not include_island or is_ <= 0 or ie_ < is_ or (is_, ie_) == (1, 1):
        return sl, el
    es, ee = min(sl, is_), max(el, ie_)
    if total_lines is not None:
        max_ln = max(1, int(total_lines))
        es = max(1, min(es, max_ln))
        ee = max(es, min(ee, max_ln))
    max_span = max(el - sl + 1, _MAX_HYDRATE_SPAN - 1)
    if ee - es + 1 > max_span:
        ee = min(ee, max(el, es + max_span - 1))
        if ee - es + 1 > max_span:
            es = max(1, ee - max_span + 1)
    return es, ee


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
    lines = split_lf_lines(text)
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
    *,
    wiki_root: Path | None = None,
) -> Path:
    if path_cache is None:
        return resolve_source_path(rel_posix_path, corpus_root, wiki_root)
    cache_root = wiki_root if (wiki_root is not None and is_wiki_rel_path(rel_posix_path)) else corpus_root
    p_key = (cache_root, rel_posix_path)
    with CACHE_LOCK:
        cached = path_cache.get(p_key)
    if cached is not None:
        return cached
    target = resolve_source_path(rel_posix_path, corpus_root, wiki_root)
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
    *,
    wiki_root: Path | None = None,
) -> tuple[Path, str, list[str]] | None:
    if file_cache is not None and rel_posix_path in file_cache:
        return file_cache[rel_posix_path]
    target = _resolve_cached_path(corpus_root, rel_posix_path, path_cache, wiki_root=wiki_root)
    if not target.is_file():
        loaded: tuple[Path, str, list[str]] | None = None
    else:
        try:
            text = target.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, UnicodeDecodeError) as exc:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR", f"Invalid UTF-8 or unreadable source file {target}: {exc}"
            ) from exc
        loaded = (target, text, split_lf_lines(text))
    if file_cache is not None:
        file_cache[rel_posix_path] = loaded
    return loaded


def _render_hydrated_slice(
    ptr: ChunkPointer,
    text: str,
    lines: list[str],
    raw_slice: str,
    es: int,
    ee: int,
    full_span: bool,
    limit: int | None,
    *,
    prepend_title: bool = True,
) -> str:
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if (es, ee) != (sl, el):
        rendered = "\n".join(lines[es - 1 : ee])
    elif not full_span and ptr.char_end > ptr.char_start:
        rendered = text[ptr.char_start : ptr.char_end]
    else:
        rendered = raw_slice
    if prepend_title and ptr.title:
        head_span = max(1, el - es + 1)
        rendered_lines = split_lf_lines(rendered)
        has_opening = (
            rendered == ptr.title
            or rendered.startswith(f"{ptr.title}\n")
            or any(normalize_ast_heading(ln) == ptr.title for ln in rendered_lines[:head_span])
        )
        if not has_opening:
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
    prepend_title: bool = True,
    file_cache: _FileCache | None = None,
    path_cache: dict[tuple[Path, str], Path] | None = None,
    wiki_root: Path | None = None,
) -> HydratedPassage:
    """Hydrate one verified slice from disk, enforcing line bounds and fingerprint freshness."""
    limit = _validate_max_chars(max_chars)
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if sl < 1 or el < sl or (el - sl + 1) > _MAX_HYDRATE_SPAN:
        raise ValueError(f"Invalid hydration line range [{sl}, {el}] for {ptr.rel_posix_path}")

    loaded = _load_corpus_file(corpus_root, ptr.rel_posix_path, file_cache, path_cache, wiki_root=wiki_root)
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
    if ptr.source_hash and fingerprint_str(raw_slice).hex() != ptr.source_hash:
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Source fingerprint drift detected on disk for {target}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    es, ee = resolve_hydrated_line_span(ptr, include_island=include_island, total_lines=len(lines))
    rendered = _render_hydrated_slice(
        ptr,
        text,
        lines,
        raw_slice,
        es,
        ee,
        full_span,
        limit,
        prepend_title=prepend_title,
    )
    return HydratedPassage(
        ptr,
        rendered,
        float(score),
        stale_on_disk=False,
        hydrated_start_line=es,
        hydrated_end_line=ee,
    )


def hydrate_passage(
    ptr: ChunkPointer,
    corpus_root: Path,
    *,
    score: float = 0.0,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
    wiki_root: Path | None = None,
) -> HydratedPassage:
    """Hydrate a single ChunkPointer directly from disk, verifying source fingerprint freshness."""
    return hydrate_one(
        corpus_root,
        ptr,
        score,
        max_chars,
        include_island,
        full_span,
        strict=False,
        wiki_root=wiki_root,
    )


def verify_source_fingerprint(
    corpus_root: Path | str,
    rel_path: str,
    start_line: int,
    end_line: int,
    expected_hash: str,
    *,
    wiki_root: Path | str | None = None,
) -> tuple[bool, str]:
    """Verify the CocoIndex fingerprint_str hex digest for a 1-indexed source line range on disk."""
    if isinstance(start_line, bool) or not isinstance(start_line, int):
        raise ValueError(f"start_line must be an int >= 1, got {start_line!r}")
    if isinstance(end_line, bool) or not isinstance(end_line, int):
        raise ValueError(f"end_line must be an int >= start_line, got {end_line!r}")
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"Invalid line range [{start_line}, {end_line}] for {rel_path!r}")
    target = resolve_source_path(rel_path, corpus_root, wiki_root)
    if not target.is_file():
        return False, ""
    try:
        text = target.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Invalid UTF-8 or unreadable source file {target}: {exc}") from exc
    lines = split_lf_lines(text)
    if end_line > len(lines):
        return False, ""
    actual = fingerprint_str("\n".join(lines[start_line - 1 : end_line])).hex()
    return actual == expected_hash, actual


def dedup_ranked_positions(
    idx: RetrievalIndex,
    ranked_chunks: Sequence[tuple[int, float]],
    *,
    limit: int | None = None,
    include_island: bool = False,
) -> tuple[tuple[int, int, float], ...]:
    """Apply MaxP span (or expanded AST island when `include_island=True`) deduplication with zero disk opens."""
    if not ranked_chunks or (limit is not None and limit <= 0):
        return ()
    out: list[tuple[int, int, float]] = []
    seen_spans: set[tuple[Any, ...]] = set()
    for cid, score in ranked_chunks:
        cid_int = int(cid)
        if (pos := idx.id_to_pos.get(cid_int)) is None:
            continue
        ptr = idx.pointers[pos]
        sl, el = int(ptr.start_line), int(ptr.end_line)
        if include_island:
            es, ee = resolve_hydrated_line_span(ptr, include_island=True)
            span: tuple[Any, ...] = (
                (ptr.rel_posix_path, es, ee)
                if es < ee
                else (ptr.rel_posix_path, sl, el, int(ptr.char_start), int(ptr.char_end))
            )
        else:
            span = (ptr.rel_posix_path, sl, el, int(ptr.char_start), int(ptr.char_end))
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
    include_island: bool = True,
    full_span: bool = False,
    wiki_root: Path | None = None,
) -> tuple[HydratedPassage, ...]:
    """Apply MaxP span/island deduplication and hydrate up to `top_k` passages from disk."""
    limit = _validate_max_chars(max_chars)
    deduped = dedup_ranked_positions(idx, ranked_chunks, limit=top_k, include_island=include_island)
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
            wiki_root=wiki_root,
        )
        for pos, _cid, score in deduped
    )


def format_evidence_route(passages: Sequence[HydratedPassage]) -> str:
    """Format hydrated passages with explicit provenance headers for recall and ReAct output."""
    blocks: list[str] = []
    for idx, hp in enumerate(passages, start=1):
        span = f"{hp.pointer.rel_posix_path}:{hp.resolved_start_line}-{hp.resolved_end_line}"
        if hp.stale_on_disk:
            blocks.append(f"[{idx}] [STALE: {span} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]")
        else:
            blocks.append(f"[{idx}] [{span} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]\n{hp.text}")
    return "\n\n".join(blocks)


def _collect_h1_by_line(lines: Sequence[str], max_el: int) -> dict[int, str]:
    start_idx = leading_frontmatter_end(lines, max_el)
    fence = ("", 0)
    h1_by_line: dict[int, str] = {}
    for idx in range(start_idx, max_el):
        fence, is_delim = advance_markdown_fence(lines[idx], fence)
        if (
            not is_delim
            and not fence[0]
            and markdown_heading_level(lines, idx + 1) == 1
            and (h1 := normalize_ast_heading(lines[idx]))
        ):
            h1_by_line[idx + 1] = h1
    return h1_by_line


def _resolve_chunk_doc_h1(lines: Sequence[str], ptrs: Sequence[ChunkPointer]) -> str:
    if not ptrs or not lines:
        return ""
    ordered = sorted(ptrs, key=lambda p: (int(p.start_line), int(p.char_start)))
    max_el = min(len(lines), max(int(p.end_line) for p in ordered))
    h1_by_line = _collect_h1_by_line(lines, max_el)
    has_titled_chunk = any(p.title.strip() for p in ordered)
    for ptr in ordered:
        t_ptr = DASH_RE.sub("-", ptr.title.strip())
        if has_titled_chunk and not t_ptr:
            continue
        sl = max(1, int(ptr.start_line))
        el = min(max_el, int(ptr.end_line))
        hits = [h1_by_line[ln] for ln in range(sl, el + 1) if ln in h1_by_line]
        if hits:
            return t_ptr if t_ptr in hits else hits[0]
    return ""


def hydrate_corpus_chunks(
    pointers: Sequence[ChunkPointer],
    corpus_root: Path,
    *,
    extract_titles_for: Collection[str] = (),
    wiki_root: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Hydrate all corpus chunks in one disk-read pass per distinct relative file path."""
    n = len(pointers)
    if n == 0:
        return [], {}
    c_root = corpus_root if corpus_root.is_absolute() else corpus_root.resolve()
    w_root = None if wiki_root is None else (wiki_root if wiki_root.is_absolute() else wiki_root.resolve())
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
                    prepend_title=False,
                    file_cache=local_cache,
                    wiki_root=w_root,
                ).text,
            )
            for pos in pos_list
        ]
        title = ""
        if rel in title_targets and (loaded := local_cache.get(rel)) is not None:
            title = _resolve_chunk_doc_h1(loaded[2], [pointers[pos] for pos in pos_list])
        return chunk_texts, rel, title

    for grp_res, rel, title in map(_read_group, by_rel.items()):
        for pos, txt in grp_res:
            texts[pos] = txt
        if title:
            titles[rel] = title
    return texts, titles
