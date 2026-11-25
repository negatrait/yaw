"""Librarian wiki card management, physical fingerprint linting, and contradiction scouting."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cards import (
        WikiCard,
        WikiCardSource,
        is_ignored_wiki_rel,
        list_wiki_cards,
        parse_wiki_card,
        read_wiki_card,
        render_wiki_card,
        validate_wiki_slug,
        write_wiki_card,
    )
    from .lint import LintIssue, LintReport, lint_wiki
    from .scout import ScoutCardSignature, is_fresh_passage, should_queue_scout, synthesize_scout_card

_PUBLIC_API: dict[str, str] = {
    "LintIssue": ".lint",
    "LintReport": ".lint",
    "ScoutCardSignature": ".scout",
    "WikiCard": ".cards",
    "WikiCardSource": ".cards",
    "is_fresh_passage": ".scout",
    "is_ignored_wiki_rel": ".cards",
    "lint_wiki": ".lint",
    "list_wiki_cards": ".cards",
    "parse_wiki_card": ".cards",
    "read_wiki_card": ".cards",
    "render_wiki_card": ".cards",
    "should_queue_scout": ".scout",
    "synthesize_scout_card": ".scout",
    "validate_wiki_slug": ".cards",
    "write_wiki_card": ".cards",
}

__all__ = [
    "LintIssue",
    "LintReport",
    "ScoutCardSignature",
    "WikiCard",
    "WikiCardSource",
    "is_fresh_passage",
    "is_ignored_wiki_rel",
    "lint_wiki",
    "list_wiki_cards",
    "parse_wiki_card",
    "read_wiki_card",
    "render_wiki_card",
    "should_queue_scout",
    "synthesize_scout_card",
    "validate_wiki_slug",
    "write_wiki_card",
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
