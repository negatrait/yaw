"""Server domain: FastMCP daemon and retrieval tools."""

from __future__ import annotations

from yaw.server.daemon import (
    DaemonLifecycle,
    compile_wiki,
    compute_seed_candidates,
    find_seed,
    get_ontology_schema,
    hydrate_disk_slice,
    ingest_raw,
    lint_wiki,
    mcp,
    query_wiki,
    run_daemon,
    stop_daemon,
)

__all__ = [
    "DaemonLifecycle",
    "compile_wiki",
    "compute_seed_candidates",
    "find_seed",
    "get_ontology_schema",
    "hydrate_disk_slice",
    "ingest_raw",
    "lint_wiki",
    "mcp",
    "query_wiki",
    "run_daemon",
    "stop_daemon",
]
