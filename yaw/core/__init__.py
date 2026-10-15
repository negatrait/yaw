"""Core package: Frozen data contracts and logging utilities."""

from .logger import FIFOLineRotatingHandler, configure_logging
from .types import (
    ChunkPointer,
    HydratedPassage,
    OpenIERecord,
    RetrievalResult,
    SyncDelta,
)

__all__ = [
    "ChunkPointer",
    "HydratedPassage",
    "OpenIERecord",
    "RetrievalResult",
    "SyncDelta",
    "FIFOLineRotatingHandler",
    "configure_logging",
]
