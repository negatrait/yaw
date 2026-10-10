"""Validation module facade re-exporting SHACL validation functions and classes."""

from __future__ import annotations

from yaw.ontology.shacl import (
    ValidationReport,
    build_validation_data_graph,
    filter_violations,
    generate_dynamic_whitelist_shape,
    load_fast_shapes_graph,
    load_shapes_graph,
    parse_validation_report,
    precompute_rdfs_closure,
    validate_wiki_graph,
    validate_wiki_graph_fast,
)

__all__ = [
    "ValidationReport",
    "build_validation_data_graph",
    "filter_violations",
    "generate_dynamic_whitelist_shape",
    "load_fast_shapes_graph",
    "load_shapes_graph",
    "parse_validation_report",
    "precompute_rdfs_closure",
    "validate_wiki_graph",
    "validate_wiki_graph_fast",
]
