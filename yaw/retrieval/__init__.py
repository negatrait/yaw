"""1-bit SoA RaBitQ + tangent-steered cyclic citation diffusion retrieval and zero-copy disk hydration."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .db import compute_db_fingerprints, normalize_fts_unicode61, open_readonly_db
    from .graph import compute_index_artifacts
    from .hydration import format_evidence_route, hydrate_passage, hydrate_ranked_passages
    from .index import build_retrieval_index
    from .lexical import lexical_search
    from .search import expand_citation_neighbors, search_index
    from .types import (
        CitationGraph,
        DiffusionOperators,
        IndexArtifacts,
        RetrievalIndex,
        SearchResult,
        empty_retrieval_index,
    )

_PUBLIC_API: dict[str, str] = {
    "CitationGraph": ".types",
    "DiffusionOperators": ".types",
    "IndexArtifacts": ".types",
    "RetrievalIndex": ".types",
    "SearchResult": ".types",
    "build_retrieval_index": ".index",
    "compute_db_fingerprints": ".db",
    "compute_index_artifacts": ".graph",
    "empty_retrieval_index": ".types",
    "expand_citation_neighbors": ".search",
    "format_evidence_route": ".hydration",
    "hydrate_passage": ".hydration",
    "hydrate_ranked_passages": ".hydration",
    "lexical_search": ".lexical",
    "normalize_fts_unicode61": ".db",
    "open_readonly_db": ".db",
    "search_index": ".search",
}

__all__ = [
    "CitationGraph",
    "DiffusionOperators",
    "IndexArtifacts",
    "RetrievalIndex",
    "SearchResult",
    "build_retrieval_index",
    "compute_db_fingerprints",
    "compute_index_artifacts",
    "empty_retrieval_index",
    "expand_citation_neighbors",
    "format_evidence_route",
    "hydrate_passage",
    "hydrate_ranked_passages",
    "lexical_search",
    "normalize_fts_unicode61",
    "open_readonly_db",
    "search_index",
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
