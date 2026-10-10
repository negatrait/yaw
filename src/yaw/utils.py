"""General purpose text and string normalization utilities."""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "format_proposition",
    "normalize_token",
    "slugify",
]


def slugify(text: str) -> str:
    """Deterministic lowercase hyphenated slug."""
    normalized = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    )
    return "-".join(re.findall(r"[a-z0-9]+", normalized.casefold()))


def normalize_token(name: str) -> str:
    """Whitespace-collapsed casefolded token string."""
    return " ".join(name.casefold().split())


def format_proposition(subject: str, predicate: str, obj: str) -> str:
    """Format triple into clean declarative proposition."""
    return f"{subject.strip()} {predicate.strip()} {obj.strip()}."
