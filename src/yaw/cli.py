"""Canonical CLI entrypoint and command handlers for YAW."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import urllib.error
from pathlib import Path

from yaw import config
from yaw.indexer import sync_vault
from yaw.retrieval import GraphRAGRetriever
from yaw.server import (
    find_seed,
    get_ontology_schema,
    hydrate_disk_slice,
    mcp,
    run_daemon,
    stop_daemon,
)

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


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with standard subparsers for index, serve, query, and schema."""
    parser = argparse.ArgumentParser(
        prog="yaw",
        description="YAW: Local Ontology GraphRAG continuous associative search & seed generation engine.",
    )
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--vault", "-w", type=str, default=None, help="Root directory of the wiki vault"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # index
    p_index = subparsers.add_parser(
        "index",
        parents=[common_parser],
        help="Sync vault index synchronously in the foreground",
    )
    p_index.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild index database from scratch",
    )
    p_index.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose indexing logs",
    )

    # serve
    p_serve = subparsers.add_parser(
        "serve",
        parents=[common_parser],
        help="Run FastMCP daemon",
    )
    p_serve.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="FastMCP transport protocol (default: http)",
    )
    p_serve.add_argument(
        "--stdio",
        action="store_true",
        help="Run FastMCP server with STDIO transport (alias for --transport stdio)",
    )
    p_serve.add_argument(
        "--http",
        action="store_true",
        help="Run FastMCP server with HTTP transport (alias for --transport http)",
    )
    p_serve.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"HTTP host (default: {config.HTTP_HOST})",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"HTTP port (default: {config.HTTP_PORT})",
    )

    # query
    p_query = subparsers.add_parser(
        "query",
        parents=[common_parser],
        help="Execute GraphRAG query and output dual-layer Markdown",
    )
    p_query.add_argument("prompt", type=str, help="Search query prompt")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Max chunks/seeds to return (default: 3)",
    )

    # schema
    subparsers.add_parser(
        "schema",
        parents=[common_parser],
        help="Inspect ontology schema and graph statistics as JSON",
    )

    # shutdown
    subparsers.add_parser(
        "shutdown",
        parents=[common_parser],
        help="Cleanly shut down a running FastMCP daemon",
    )

    return parser


def handle_index(args: argparse.Namespace) -> int:
    """Execute synchronous foreground indexing."""
    vault_dir = Path(args.vault).resolve() if args.vault else None
    if vault_dir:
        config.set_vault(vault_dir)
    try:
        sync_vault(vault_dir=vault_dir, force=args.force, verbose=not args.quiet)
        return 0
    except (sqlite3.DatabaseError, OSError, RuntimeError) as err:
        sys.stderr.write(f"Error indexing vault: {err}\n")
        return 1


def handle_serve(args: argparse.Namespace) -> int:
    """Start FastMCP daemon with STDIO protocol hygiene."""
    vault_dir = Path(args.vault).resolve() if args.vault else None
    if vault_dir:
        config.set_vault(vault_dir)
    if args.port:
        config.set_port(args.port)

    transport = "stdio" if getattr(args, "stdio", False) else args.transport
    if getattr(args, "http", False):
        transport = "http"

    host = args.host or config.HTTP_HOST
    port = args.port or config.HTTP_PORT

    print(
        f"Starting YAW FastMCP daemon ({transport} transport)...",
        file=sys.stderr,
        flush=True,
    )
    run_daemon(transport=transport, host=host, port=port)
    return 0


def handle_shutdown(args: argparse.Namespace) -> int:
    """Cleanly shut down running FastMCP daemon."""
    vault_dir = Path(args.vault).resolve() if args.vault else None
    if vault_dir:
        config.set_vault(vault_dir)
    return stop_daemon()


def handle_query(args: argparse.Namespace) -> int:
    """Execute GraphRAG query and output dual-layer Markdown to sys.stdout."""
    vault_dir = Path(args.vault).resolve() if args.vault else None
    if vault_dir:
        config.set_vault(vault_dir)

    try:
        retriever = GraphRAGRetriever()
        markdown = asyncio.run(retriever.retrieve(args.prompt, top_k=args.top_k))
        sys.stdout.write(markdown + ("\n" if not markdown.endswith("\n") else ""))
        return 0
    except (
        sqlite3.DatabaseError,
        OSError,
        ValueError,
        RuntimeError,
        urllib.error.URLError,
    ) as err:
        sys.stderr.write(f"Error executing GraphRAG query: {err}\n")
        return 1


def handle_schema(args: argparse.Namespace) -> int:
    """Output structured ontology schema JSON to sys.stdout."""
    vault_dir = Path(args.vault).resolve() if args.vault else None
    if vault_dir:
        config.set_vault(vault_dir)

    try:
        schema = get_ontology_schema()
        sys.stdout.write(json.dumps(schema, indent=2) + "\n")
        return 0
    except (sqlite3.DatabaseError, OSError) as err:
        sys.stderr.write(f"Error querying schema: {err}\n")
        return 1


def main(args_list: list[str] | None = None) -> int:
    """Main CLI entrypoint routing commands to handlers."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    if not args.command:
        parser.print_help(sys.stderr)
        return 2

    if args.command == "index":
        return handle_index(args)
    if args.command == "serve":
        return handle_serve(args)
    if args.command == "shutdown":
        return handle_shutdown(args)
    if args.command == "query":
        return handle_query(args)
    return handle_schema(args)


if __name__ == "__main__":
    sys.exit(main())
