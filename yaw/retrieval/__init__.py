"""1-bit SoA RaBitQ + tangent-steered cyclic citation diffusion retrieval and zero-copy disk hydration."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .db import compute_db_fingerprints, normalize_fts_unicode61, open_readonly_db
    from .graph import compute_index_artifacts
    from .hydration import (
        dedup_ranked_positions,
        format_evidence_route,
        hydrate_passage,
        hydrate_ranked_passages,
        resolve_hydrated_line_span,
        verify_source_fingerprint,
    )
    from .index import build_retrieval_index
    from .lexical import lexical_search
    from .search import expand_citation_neighbors, search_index
    from .slip import (
        NO_MATCHES_TEXT,
        build_recall_payload,
        dump_recall_json,
        format_recall_text,
        sanitize_slip_label,
    )
    from .types import (
        CitationGraph,
        DiffusionOperators,
        IndexArtifacts,
        RetrievalIndex,
        SearchResult,
        empty_retrieval_index,
    )

_PUBLIC_API: dict[str, str] = {
    "NO_MATCHES_TEXT": ".slip",
    "CitationGraph": ".types",
    "DiffusionOperators": ".types",
    "IndexArtifacts": ".types",
    "RetrievalIndex": ".types",
    "SearchResult": ".types",
    "build_recall_payload": ".slip",
    "build_retrieval_index": ".index",
    "compute_db_fingerprints": ".db",
    "compute_index_artifacts": ".graph",
    "dedup_ranked_positions": ".hydration",
    "dump_recall_json": ".slip",
    "empty_retrieval_index": ".types",
    "expand_citation_neighbors": ".search",
    "format_evidence_route": ".hydration",
    "format_recall_text": ".slip",
    "hydrate_passage": ".hydration",
    "hydrate_ranked_passages": ".hydration",
    "lexical_search": ".lexical",
    "normalize_fts_unicode61": ".db",
    "open_readonly_db": ".db",
    "resolve_hydrated_line_span": ".hydration",
    "sanitize_slip_label": ".slip",
    "search_index": ".search",
    "verify_source_fingerprint": ".hydration",
}

__all__ = [
    "NO_MATCHES_TEXT",
    "CitationGraph",
    "DiffusionOperators",
    "IndexArtifacts",
    "RetrievalIndex",
    "SearchResult",
    "build_recall_payload",
    "build_retrieval_index",
    "compute_db_fingerprints",
    "compute_index_artifacts",
    "dedup_ranked_positions",
    "dump_recall_json",
    "empty_retrieval_index",
    "expand_citation_neighbors",
    "format_evidence_route",
    "format_recall_text",
    "hydrate_passage",
    "hydrate_ranked_passages",
    "lexical_search",
    "normalize_fts_unicode61",
    "open_readonly_db",
    "resolve_hydrated_line_span",
    "sanitize_slip_label",
    "search_index",
    "verify_source_fingerprint",
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
