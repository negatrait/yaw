"""Strictly read-only physical fingerprint and frontmatter linter for `wiki_root`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from yaw.core import YawConfig, YawRuntimeError, require_corpus_dir
from yaw.retrieval import verify_source_fingerprint

from .cards import is_ignored_wiki_rel, parse_wiki_card


@dataclass(frozen=True, slots=True)
class LintIssue:
    """Single wiki lint finding (error or non-fatal warning)."""

    severity: Literal["error", "warning"]
    code: str
    card_rel_path: str
    detail: str


@dataclass(frozen=True, slots=True)
class LintReport:
    """Aggregate read-only wiki lint report."""

    wiki_root: str | None
    cards_checked: int
    stale_cards: tuple[str, ...]
    issues: tuple[LintIssue, ...]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lint_single_card(
    card_path: Path,
    wiki_root: Path,
    corpus_root: Path,
    issues: list[LintIssue],
) -> bool:
    card_rel = card_path.relative_to(wiki_root).as_posix()
    try:
        card = parse_wiki_card(card_path, wiki_root)
    except (ValueError, yaml.YAMLError) as exc:
        issues.append(LintIssue("error", "INVALID_CARD", card_rel, str(exc)))
        return False

    if (corpus_root / card_rel).is_file():
        issues.append(
            LintIssue(
                "warning",
                "REL_PATH_COLLISION",
                card_rel,
                f"Wiki card {card_rel!r} collides with corpus file at the same relative path.",
            )
        )

    is_stale = False
    if card.stale:
        issues.append(LintIssue("error", "CARD_MARKED_STALE", card_rel, "Card frontmatter sets stale: true."))
        is_stale = True

    for src in card.sources:
        try:
            matched, actual = verify_source_fingerprint(
                corpus_root,
                src.rel_path,
                src.start_line,
                src.end_line,
                src.source_hash,
                wiki_root=wiki_root,
            )
        except ValueError as exc:
            issues.append(LintIssue("error", "STALE_SOURCE", card_rel, str(exc)))
            is_stale = True
            continue
        if not matched:
            issues.append(
                LintIssue(
                    "error",
                    "STALE_SOURCE",
                    card_rel,
                    f"Source {src.rel_path}:{src.start_line}-{src.end_line} failed fingerprint verification "
                    f"(expected={src.source_hash}, actual={actual or 'missing'}).",
                )
            )
            is_stale = True

    return is_stale


def lint_wiki(cfg: YawConfig) -> LintReport:
    """Verify wiki cards against physical disk fingerprints without mutating any files or SQLite state."""
    if cfg.wiki_root is None:
        return LintReport(wiki_root=None, cards_checked=0, stale_cards=(), issues=(), ok=True)

    wiki_root = Path(cfg.wiki_root).resolve()
    if not wiki_root.exists():
        return LintReport(wiki_root=str(wiki_root), cards_checked=0, stale_cards=(), issues=(), ok=True)
    if not wiki_root.is_dir():
        raise YawRuntimeError("CONSISTENCY_ERROR", f"wiki_root does not exist or is not a directory: {wiki_root}")

    card_paths = [
        path for path in sorted(wiki_root.rglob("*.md")) if not is_ignored_wiki_rel(path.relative_to(wiki_root).parts)
    ]
    if not card_paths:
        return LintReport(wiki_root=str(wiki_root), cards_checked=0, stale_cards=(), issues=(), ok=True)

    assert cfg.corpus_root is not None
    corpus_root = require_corpus_dir(cfg.corpus_root)
    issues: list[LintIssue] = []
    stale_cards = [
        card_path.relative_to(wiki_root).as_posix()
        for card_path in card_paths
        if _lint_single_card(card_path, wiki_root, corpus_root, issues)
    ]

    ok = not any(issue.severity == "error" for issue in issues)
    return LintReport(
        wiki_root=str(wiki_root),
        cards_checked=len(card_paths),
        stale_cards=tuple(stale_cards),
        issues=tuple(issues),
        ok=ok,
    )
