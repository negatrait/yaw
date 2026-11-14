"""Stable identities for native callable embedding fixtures."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class IdentifiedEmbeddingCallable:
    fn: Callable[..., Any]
    index_identity: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)
