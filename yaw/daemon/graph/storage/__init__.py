"""Storage leaf package: Unified yaw.db SQLite engine, sqlite-vec KNN, and line hydration."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    "CandidateCacheStore": ".candidate_cache",
    "CandidatePayloadCodec": ".candidate_cache",
    "CandidateStageCache": ".candidate_cache",
    # Database lifecycle & schema
    "ManagedConnection": ".db",
    "YawConnection": ".db",
    "open_managed_yaw_db": ".db",
    "open_yaw_db": ".db",
    "open_readonly_yaw_db": ".db",
    "initialize_schema": ".db",
    "get_meta": ".db",
    "set_meta": ".db",
    # Vector serialization, bounded SQL cosine KNN & Contentless FTS5 lexical search
    "serialize_f32_vector": ".vectors",
    "deserialize_f32_vector": ".vectors",
    "upsert_chunk_vectors": ".vectors",
    "upsert_entity_vectors": ".vectors",
    "upsert_fact_vectors": ".vectors",
    "search_chunk_vectors_knn": ".vectors",
    "search_entity_vectors_knn": ".vectors",
    "search_fact_vectors_knn": ".vectors",
    "get_entity_vectors": ".vectors",
    "search_chunks_fts": ".vectors",
    "search_facts_fts": ".vectors",
    # Chunk metadata, OpenIE checkpoints, SKOS/RDF/OWL graph CRUD & candidate mining
    "find_entity_synonym_candidates": ".graph_ops",
    "find_entity_disambiguation_candidates": ".graph_ops",
    "upsert_chunk_pointers": ".graph_ops",
    "get_chunk_pointer": ".graph_ops",
    "get_chunk_pointers": ".graph_ops",
    "get_openie_checkpoints": ".graph_ops",
    "save_openie_checkpoints": ".graph_ops",
    "upsert_entities": ".graph_ops",
    "upsert_entity_labels": ".graph_ops",
    "get_entities_by_label_norm": ".graph_ops",
    "resolve_entity_labels": ".graph_ops",
    "merge_entities": ".graph_ops",
    "upsert_facts": ".graph_ops",
    "upsert_fact_sources": ".graph_ops",
    "upsert_edges_with_sources": ".graph_ops",
    "upsert_semantic_edges": ".graph_ops",
    "upsert_synonymy_edges": ".graph_ops",
    "get_existing_ids": ".graph_ops",
    "get_facts_by_ids": ".graph_ops",
    "get_facts_with_meta_by_ids": ".graph_ops",
    "get_entity_doc_frequencies": ".graph_ops",
    "get_entity_metadata_batch": ".graph_ops",
    "get_graph_counts": ".graph_ops",
    "load_graph_topology": ".graph_ops",
    "delete_chunks": ".graph_ops",
    # Closed-world SQL SHACL linter, repair execution & W3C Turtle serializer
    "inspect_focus_node": ".shacl",
    "format_entity_skos_markdown": ".shacl",
    "format_pair_disambiguation_dossier": ".shacl",
    "lint_graph_shacl": ".shacl",
    "apply_shacl_repairs": ".shacl",
    "serialize_graph_turtle": ".shacl",
    # 1-Indexed disk line hydration & AST dehydration
    "CoalescedSpan": ".hydration",
    "MutableSpan": ".hydration",
    "coalesce_chunk_spans": ".hydration",
    "coalesce_spans": ".hydration",
    "dehydrate_passage_ast": ".hydration",
    "hydrate_ast_parent_context": ".hydration",
    "hydrate_budgeted_passages": ".hydration",
    "hydrate_chunk_pointer": ".hydration",
    "hydrate_passages": ".hydration",
    "make_ast_island_hydrator": ".hydration",
    "make_chunk_hydrator": ".hydration",
    "resolve_one_hop_upward_ast_bounds": ".hydration",
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
