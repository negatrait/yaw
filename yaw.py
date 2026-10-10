#!/usr/bin/env python3
"""Canonical CLI entrypoint delegator for YAW."""

from __future__ import annotations

import sys
from pathlib import Path

_root_dir = str(Path(__file__).resolve().parent)
_src_dir = str(Path(__file__).resolve().parent / "src")
while _root_dir in sys.path:
    sys.path.remove(_root_dir)
while "" in sys.path:
    sys.path.remove("")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_self_entrypoint = sys.modules.get("yaw")
if _self_entrypoint is not None and not hasattr(_self_entrypoint, "__path__"):
    del sys.modules["yaw"]
try:
    from yaw.cli import (
        build_parser,
        find_seed,
        get_ontology_schema,
        handle_index,
        handle_query,
        handle_schema,
        handle_serve,
        handle_shutdown,
        hydrate_disk_slice,
        main,
        mcp,
        run_daemon,
        stop_daemon,
        sync_vault,
    )
finally:
    if _self_entrypoint is not None and not hasattr(_self_entrypoint, "__path__"):
        sys.modules["yaw"] = _self_entrypoint

__all__ = [
    "build_parser",
    "find_seed",
    "get_ontology_schema",
    "handle_index",
    "handle_query",
    "handle_schema",
    "handle_serve",
    "handle_shutdown",
    "hydrate_disk_slice",
    "main",
    "mcp",
    "run_daemon",
    "stop_daemon",
    "sync_vault",
]

if __name__ == "__main__":
    sys.exit(main())
