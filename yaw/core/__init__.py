"""Core package: Frozen data contracts, configuration, pure coordinate/ID helpers, and logging utilities."""

from .config import YawConfig, load_config
from .logger import FIFOLineRotatingHandler, configure_logging
from .types import (
    ChunkPointer,
    DaemonStatus,
    ErrorCode,
    HydratedPassage,
    IndexStats,
    OpenIERecord,
    RetrievalResult,
    SyncDelta,
    YawRuntimeError,
    compute_chunk_id,
    compute_mdhash_id,
    inclusive_end_line,
    resolve_corpus_path,
    to_rel_posix_path,
)

__all__ = [
    "ChunkPointer",
    "DaemonStatus",
    "ErrorCode",
    "FIFOLineRotatingHandler",
    "HydratedPassage",
    "IndexStats",
    "OpenIERecord",
    "RetrievalResult",
    "SyncDelta",
    "YawConfig",
    "YawRuntimeError",
    "compute_chunk_id",
    "compute_mdhash_id",
    "configure_logging",
    "inclusive_end_line",
    "load_config",
    "resolve_corpus_path",
    "to_rel_posix_path",
]
