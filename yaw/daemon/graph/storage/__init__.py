"""Storage leaf package: Unified yaw.db SQLite engine, sqlite-vec KNN, and line hydration."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    # Database lifecycle & schema
    "open_yaw_db": ".db",
    "initialize_schema": ".db",
    # Vector serialization & bounded SQL cosine KNN
    "serialize_f32_vector": ".db",
    "deserialize_f32_vector": ".db",
    "upsert_chunk_vectors": ".db",
    "upsert_entity_vectors": ".db",
    "upsert_fact_vectors": ".db",
    "search_chunk_vectors_knn": ".db",
    "search_entity_vectors_knn": ".db",
    "search_fact_vectors_knn": ".db",
    "get_entity_vectors": ".db",
    "find_entity_synonym_candidates": ".db",
    # Chunk metadata & Contentless FTS5 lexical search
    "upsert_chunk_pointers": ".db",
    "get_chunk_pointer": ".db",
    "get_chunk_pointers": ".db",
    "get_max_end_line": ".db",
    "search_chunks_fts": ".db",
    "search_facts_fts": ".db",
    # OpenIE checkpoint cache
    "get_openie_checkpoints": ".db",
    "save_openie_checkpoints": ".db",
    # Graph nodes, edges & provenance
    "upsert_entities": ".db",
    "upsert_facts": ".db",
    "upsert_edges_with_sources": ".db",
    "upsert_synonymy_edges": ".db",
    "get_existing_ids": ".db",
    "get_facts_by_ids": ".db",
    "get_entity_doc_frequencies": ".db",
    "get_graph_counts": ".db",
    "get_meta": ".db",
    "set_meta": ".db",
    "load_graph_topology": ".db",
    # Atomic cascading chunk deletion
    "delete_chunks": ".db",
    # 1-Indexed disk line hydration
    "hydrate_chunk_pointer": ".hydration",
    "hydrate_passages": ".hydration",
    "make_chunk_hydrator": ".hydration",
}

__all__ = sorted(_PUBLIC_API.keys())


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
