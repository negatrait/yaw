"""Text and token normalization utilities."""

from __future__ import annotations

from yaw.utils import normalize_token, slugify

__all__ = [
    "is_valid_entity_text",
    "normalize_token",
    "slugify",
]


_STRIP_PUNCT_WS = " \t\n\r\"'.,:;!?(){}[]<>"
_CODE_SYNTAX_CHARS = frozenset(
    {"}", "{", ")", "(", "]", "[", "\\", '"', "'", "=", ";", "\n", "\t", "`"}
)


def is_valid_entity_text(text: str) -> bool:
    """Validate whether entity text represents a genuine human/domain entity phrase."""
    s = text.strip(_STRIP_PUNCT_WS)
    if len(s) < 2 or len(s) > 100:
        return False
    if any(c in _CODE_SYNTAX_CHARS for c in s):
        return False
    if s.replace(" ", "").isdigit():
        return False
    return any(c.isalnum() for c in s)
