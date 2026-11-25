"""Markdown librarian wiki card parsing, rendering, slug validation, and atomic persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from yaw.core import (
    YawRuntimeError,
    advance_markdown_fence,
    markdown_heading_level,
    normalize_ast_heading,
    resolve_corpus_path,
    split_lf_lines,
    write_text_atomic,
)


@dataclass(frozen=True, slots=True)
class WikiCardSource:
    """Verified corpus or wiki source span cited by a wiki card."""

    rel_path: str
    start_line: int
    end_line: int
    source_hash: str


@dataclass(frozen=True, slots=True)
class WikiCard:
    """Parsed Markdown wiki card with YAML frontmatter metadata and H1 title."""

    slug: str
    rel_path: str
    title: str
    body: str
    sources: tuple[WikiCardSource, ...]
    stale: bool = False
    kind: str = "note"
    updated_at: str = ""


def validate_wiki_slug(slug: str) -> str:
    """Validate and normalize a relative wiki card slug without the `.md` suffix."""
    if not isinstance(slug, str) or not slug.strip():
        raise ValueError("Wiki card slug must be a non-empty string.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in slug):
        raise ValueError(f"Control characters not allowed in wiki slug: {slug!r}")
    norm = slug.strip().replace("\\", "/")
    if norm.startswith("/") or norm.endswith("/") or ":" in norm:
        raise ValueError(f"Invalid wiki card slug path: {slug!r}")
    if norm.lower().endswith(".tmp"):
        raise ValueError(f"Temporary suffix '.tmp' not allowed in wiki slug: {slug!r}")
    if norm.lower().endswith(".md"):
        norm = norm[:-3]
    if not norm or norm.endswith("/"):
        raise ValueError(f"Wiki card slug must not be empty: {slug!r}")
    parts = norm.split("/")
    if any(not part or part in {".", ".."} or part.startswith(".") or part.endswith(".tmp") for part in parts):
        raise ValueError(f"Forbidden path segment in wiki slug: {slug!r}")
    return "/".join(parts)


def _resolve_h1_at(lines: Sequence[str], idx: int) -> tuple[str, int] | None:
    if markdown_heading_level(lines, idx + 1) != 1:
        return None
    if lines[idx].lstrip(" ").startswith("#"):
        title = normalize_ast_heading(lines[idx])
        return (title, idx + 1) if title else None
    for u_idx in range(idx + 1, min(len(lines), idx + 4)):
        cand = lines[u_idx]
        u = cand.strip()
        if not u:
            break
        if not cand.startswith("\t") and len(cand) - len(cand.lstrip(" ")) <= 3 and set(u) == {"="}:
            title = normalize_ast_heading(" ".join(lines[idx:u_idx]))
            return (title, u_idx + 1) if title else None
    return None


def _strip_matching_leading_h1(body: str, clean_title: str) -> str:
    norm_body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not norm_body:
        return ""
    lines = split_lf_lines(norm_body)
    if (resolved := _resolve_h1_at(lines, 0)) is not None and resolved[0] == clean_title:
        return "\n".join(lines[resolved[1] :]).strip()
    return norm_body


def render_wiki_card(
    title: str,
    body: str,
    *,
    sources: Sequence[WikiCardSource] = (),
    stale: bool = False,
    kind: str = "note",
    updated_at: str | None = None,
) -> str:
    """Render a wiki card with YAML frontmatter metadata and a canonical H1 heading."""
    if not isinstance(title, str) or not (clean_title := normalize_ast_heading(title)):
        raise ValueError("Wiki card title must be a non-empty string.")
    if not isinstance(body, str):
        raise TypeError("Wiki card body must be a string.")
    if not isinstance(kind, str) or not (clean_kind := kind.strip()):
        raise ValueError("Wiki card kind must be a non-empty string.")
    ts = (
        updated_at.strip()
        if updated_at is not None
        else datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    meta: dict[str, Any] = {
        "kind": clean_kind,
        "stale": bool(stale),
        "updated_at": ts,
        "sources": [
            {
                "rel_path": src.rel_path,
                "start_line": int(src.start_line),
                "end_line": int(src.end_line),
                "source_hash": src.source_hash,
            }
            for src in sources
        ],
    }
    frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip("\n")
    body_rest = _strip_matching_leading_h1(body, clean_title)
    md_section = f"# {clean_title}\n\n{body_rest}\n" if body_rest else f"# {clean_title}\n"
    return f"---\n{frontmatter}\n---\n\n{md_section}"


def _split_card_frontmatter(text: str, rel_path: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end_idx = text.find("\n---\n", 4)
    if end_idx != -1:
        raw_yaml = text[4:end_idx]
        md_body = text[end_idx + 5 :]
    elif text.rstrip("\n").endswith("\n---"):
        end_eof = text.rfind("\n---")
        raw_yaml = text[4:end_eof]
        md_body = ""
    else:
        raise ValueError(f"Wiki card {rel_path!r} is missing closing YAML frontmatter '---'.")
    loaded: object = yaml.safe_load(raw_yaml)
    if loaded is None:
        return {}, md_body
    if not isinstance(loaded, dict):
        raise ValueError(f"Wiki card {rel_path!r} YAML frontmatter must be a mapping.")
    if "title" in loaded:
        raise ValueError(f"Wiki card {rel_path!r} must store title as Markdown H1, not in YAML frontmatter.")
    return {str(k): v for k, v in loaded.items()}, md_body


def _parse_card_sources(raw_sources: object, rel_path: str) -> tuple[WikiCardSource, ...]:
    if raw_sources is None:
        return ()
    if not isinstance(raw_sources, list):
        raise ValueError(f"Wiki card {rel_path!r} frontmatter 'sources' must be a list.")
    parsed: list[WikiCardSource] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            raise ValueError(f"Wiki card {rel_path!r} source entry must be a mapping.")
        src_rel = item.get("rel_path")
        sl = item.get("start_line")
        el = item.get("end_line")
        sh = item.get("source_hash")
        if not isinstance(src_rel, str) or not src_rel.strip():
            raise ValueError(f"Wiki card {rel_path!r} source missing valid 'rel_path'.")
        if isinstance(sl, bool) or not isinstance(sl, int) or sl < 1:
            raise ValueError(f"Wiki card {rel_path!r} source has invalid 'start_line': {sl!r}")
        if isinstance(el, bool) or not isinstance(el, int) or el < sl:
            raise ValueError(f"Wiki card {rel_path!r} source has invalid 'end_line': {el!r}")
        if not isinstance(sh, str):
            raise ValueError(f"Wiki card {rel_path!r} source has invalid 'source_hash': {sh!r}")
        parsed.append(WikiCardSource(rel_path=src_rel, start_line=sl, end_line=el, source_hash=sh))
    return tuple(parsed)


def _extract_card_h1(md_body: str, rel_path: str) -> tuple[str, str]:
    lines = split_lf_lines(md_body)
    fence = ("", 0)
    for idx, line in enumerate(lines):
        fence, is_delim = advance_markdown_fence(line, fence)
        if is_delim or fence[0]:
            continue
        if (resolved := _resolve_h1_at(lines, idx)) is not None:
            title, drop_end = resolved
            body = "\n".join([*lines[:idx], *lines[drop_end:]]).strip()
            return title, body
    raise ValueError(f"Wiki card {rel_path!r} is missing an H1 heading outside fenced code blocks.")


def parse_wiki_card(card_path: Path, wiki_root: Path) -> WikiCard:
    """Read and validate a wiki card Markdown file under `wiki_root`."""
    root = Path(wiki_root).resolve()
    target = Path(card_path).resolve()
    rel_path = target.relative_to(root).as_posix()
    slug = validate_wiki_slug(rel_path)
    try:
        text = target.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Invalid UTF-8 or unreadable wiki card {target}: {exc}") from exc

    meta, md_body = _split_card_frontmatter(text, rel_path)
    raw_stale = meta.get("stale", False)
    if not isinstance(raw_stale, bool):
        raise ValueError(f"Wiki card {rel_path!r} frontmatter 'stale' must be a boolean.")
    raw_kind = meta.get("kind", "note")
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise ValueError(f"Wiki card {rel_path!r} frontmatter 'kind' must be a non-empty string.")
    raw_updated = meta.get("updated_at", "")
    if not isinstance(raw_updated, str):
        raise ValueError(f"Wiki card {rel_path!r} frontmatter 'updated_at' must be a string.")
    sources = _parse_card_sources(meta.get("sources"), rel_path)
    title, body = _extract_card_h1(md_body, rel_path)
    return WikiCard(
        slug=slug,
        rel_path=rel_path,
        title=title,
        body=body,
        sources=sources,
        stale=raw_stale,
        kind=raw_kind.strip(),
        updated_at=raw_updated,
    )


def write_wiki_card(
    wiki_root: Path | str,
    slug: str,
    title: str,
    body: str,
    *,
    sources: Sequence[WikiCardSource] = (),
    stale: bool = False,
    kind: str = "note",
    updated_at: str | None = None,
) -> tuple[Path, bool]:
    """Write a wiki card atomically unless an identical semantic card already exists on disk."""
    clean_slug = validate_wiki_slug(slug)
    clean_title = normalize_ast_heading(title)
    if not clean_title:
        raise ValueError("Wiki card title must be a non-empty string.")
    clean_kind = kind.strip()
    if not clean_kind:
        raise ValueError("Wiki card kind must be a non-empty string.")
    root = Path(wiki_root).resolve()
    target = resolve_corpus_path(root, f"{clean_slug}.md")
    norm_body = _strip_matching_leading_h1(body, clean_title)
    norm_sources = tuple(sources)
    bool_stale = bool(stale)

    if target.is_file():
        try:
            existing = parse_wiki_card(target, root)
        except ValueError, yaml.YAMLError:
            existing = None
        if (
            existing is not None
            and (existing.title, existing.body, existing.sources, existing.stale, existing.kind)
            == (clean_title, norm_body, norm_sources, bool_stale, clean_kind)
            and (updated_at is None or existing.updated_at == updated_at.strip())
        ):
            return target, False

    rendered = render_wiki_card(
        clean_title,
        norm_body,
        sources=norm_sources,
        stale=bool_stale,
        kind=clean_kind,
        updated_at=updated_at,
    )
    write_text_atomic(target, rendered, newline="\n")
    return target, True


def is_ignored_wiki_rel(rel_parts: Sequence[str]) -> bool:
    """Return True when a relative wiki path contains hidden `.*`, `.tmp`, or `.tmp.md` segments."""
    return any(part.startswith(".") or part.endswith((".tmp", ".tmp.md")) for part in rel_parts)


def list_wiki_cards(wiki_root: Path | str) -> list[dict[str, Any]]:
    """List parsed metadata for all active `.md` wiki cards under `wiki_root`."""
    root = Path(wiki_root).resolve()
    if not root.exists():
        return []
    if not root.is_dir():
        raise YawRuntimeError("CONSISTENCY_ERROR", f"wiki_root is not a directory: {root}")
    cards: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        rel_parts = path.relative_to(root).parts
        if is_ignored_wiki_rel(rel_parts):
            continue
        try:
            card = parse_wiki_card(path, root)
        except (ValueError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Invalid wiki card {path}: {exc}") from exc
        cards.append(
            {
                "slug": card.slug,
                "rel_path": card.rel_path,
                "title": card.title,
                "kind": card.kind,
                "stale": card.stale,
                "updated_at": card.updated_at,
                "sources": [asdict(src) for src in card.sources],
            }
        )
    return cards


def read_wiki_card(wiki_root: Path | str, slug: str) -> str:
    """Read raw LF-normalized Markdown content for an existing wiki card slug."""
    clean_slug = validate_wiki_slug(slug)
    root = Path(wiki_root).resolve()
    target = resolve_corpus_path(root, f"{clean_slug}.md")
    if not target.is_file():
        raise YawRuntimeError("INVALID_INPUT", f"Wiki card not found: {clean_slug!r}")
    try:
        return target.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Invalid UTF-8 or unreadable wiki card {target}: {exc}") from exc
