"""Canonical CLI entrypoint for the YAW persistent HippoRAG 2 daemon.

Grammar:
  yaw daemon {start|stop|status|run} [--config <path>]
  yaw graph {build|rebuild|lint} [--config <path>] [--turtle]
  yaw dspy {optimize|load-program|benchmark} [--flags]
  yaw config {get|set} [--flags]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy

from yaw.core import (
    DaemonStatus,
    IndexStats,
    YawConfig,
    YawRuntimeError,
    get_config_value,
    load_config,
    parse_dataset_list,
    set_config_value,
    write_text_atomic,
)
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, YawProgram, promote_compiled_program
from yaw.daemon.graph.storage import (
    get_graph_counts,
    lint_graph_shacl,
    open_readonly_yaw_db,
    open_yaw_db,
    serialize_graph_turtle,
)
from yaw.daemon.ingestion import load_dataset_splits, resolve_dataset_bundle
from yaw.daemon.runtime import YawRuntime, is_file_locked

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build hierarchical argument parser for `yaw <domain> <action> --<flags>`."""
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument(
        "--config",
        "-c",
        dest="config",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to yaw.toml configuration file (default: ./yaw.toml)",
    )

    parser = argparse.ArgumentParser(
        prog="yaw",
        description="YAW: Native HippoRAG 2 & CocoIndex v1 persistent daemon.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        type=str,
        default=None,
        help="Path to yaw.toml configuration file (default: ./yaw.toml)",
    )

    domain_subparsers = parser.add_subparsers(dest="domain", help="Domain to operate on")

    # -------------------------------------------------------------------------
    # 1. yaw daemon {start|stop|status|run}
    # -------------------------------------------------------------------------
    p_daemon = domain_subparsers.add_parser("daemon", help="Daemon lifecycle commands", allow_abbrev=False)
    daemon_subs = p_daemon.add_subparsers(dest="action", help="Daemon action")

    daemon_subs.add_parser("start", parents=[common], help="Start persistent daemon in background", allow_abbrev=False)

    daemon_subs.add_parser("stop", parents=[common], help="Gracefully stop a running daemon", allow_abbrev=False)
    daemon_subs.add_parser(
        "status",
        parents=[common],
        help="Inspect daemon status, graph index statistics, and active optimizations as JSON",
        allow_abbrev=False,
    )
    daemon_subs.add_parser(
        "run",
        parents=[common],
        help="Run persistent daemon in the foreground until stopped",
        allow_abbrev=False,
    )

    # -------------------------------------------------------------------------
    # 2. yaw graph {build|rebuild|lint}
    # -------------------------------------------------------------------------
    p_graph = domain_subparsers.add_parser(
        "graph", help="Standalone graph and corpus index commands", allow_abbrev=False
    )
    graph_subs = p_graph.add_subparsers(dest="action", help="Graph action")

    graph_subs.add_parser(
        "build",
        parents=[common],
        help="Run standalone incremental CocoIndex corpus sync and update yaw.db",
        allow_abbrev=False,
    )
    graph_subs.add_parser(
        "rebuild",
        parents=[common],
        help="Wipe yaw.db and CocoIndex state and perform a full rebuild from scratch",
        allow_abbrev=False,
    )
    p_lint = graph_subs.add_parser(
        "lint",
        parents=[common],
        help="Run closed-world SHACL graph lint or emit W3C Turtle (RDF/SKOS/OWL/SHACL)",
        allow_abbrev=False,
    )
    p_lint.add_argument(
        "--turtle",
        action="store_true",
        help="Emit W3C Turtle (RDF/SKOS/OWL/SHACL) serialization instead of JSON",
    )
    p_lint.add_argument(
        "--db-path",
        default=None,
        help="Inspect a specific SQLite DB or 'latest' optimization sandbox instead of config.db_path",
    )

    # -------------------------------------------------------------------------
    # 3. yaw dspy {optimize|load-program|benchmark}
    # -------------------------------------------------------------------------
    p_dspy = domain_subparsers.add_parser(
        "dspy", help="DSPy prompt optimization, benchmark, and program management", allow_abbrev=False
    )
    dspy_subs = p_dspy.add_subparsers(dest="action", help="DSPy action")

    p_opt = dspy_subs.add_parser(
        "optimize",
        parents=[common],
        help="Compile and evolve DSPy prompts in an isolated sandbox and save candidate JSON",
        allow_abbrev=False,
    )
    p_opt.add_argument(
        "--auto",
        dest="auto",
        choices=["light", "medium", "heavy"],
        default=None,
        help="GEPA/MIPROv2 evaluation budget mode (defaults to yaw.toml [optimization].auto, then 'light')",
    )
    p_opt.add_argument(
        "--mode",
        dest="mode",
        choices=["miprov2", "gepa", "hybrid"],
        default=None,
        help="Prompt optimization mode (defaults to yaw.toml [optimization].mode, then 'hybrid')",
    )
    p_opt.add_argument(
        "--dataset",
        dest="dataset",
        nargs="*",
        default=None,
        help=(
            "One or more dataset names (enterprise_rag_bench, sample, 2wikimultihopqa, musique, hotpotqa) "
            "or path to JSON training file (defaults to yaw.toml [optimization].datasets)"
        ),
    )
    p_opt.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries to load (defaults to yaw.toml [optimization].max_queries, max 50 on Tier 2 corpora)",
    )
    p_opt.add_argument(
        "--train-queries",
        type=int,
        default=None,
        help="Number of queries for training [0:K] (defaults to yaw.toml [optimization].train_queries)",
    )
    p_opt.add_argument(
        "--query-offset",
        type=int,
        default=None,
        help="Starting query offset in dataset (defaults to yaw.toml [optimization].query_offset)",
    )
    p_opt.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Override output path for candidate compiled_program JSON",
    )
    p_opt.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Explicit number of optimization trials (sets auto to None for fast smoke tests)",
    )
    p_opt.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Explicit number of instruction/fewshot candidates per predictor",
    )
    p_opt.add_argument(
        "--sleep",
        dest="sleep_seconds",
        type=float,
        default=None,
        help="Sleep pause in seconds between LLM/embedding calls to manage thermal duty cycles "
        "(defaults to yaw.toml [optimization].sleep_seconds, default 0.0)",
    )
    p_opt.add_argument(
        "--max-bootstrapped-demos",
        dest="max_bootstrapped_demos",
        type=int,
        default=None,
        help="Maximum bootstrapped demonstrations for MIPROv2 (defaults to yaw.toml "
        "[optimization].max_bootstrapped_demos, 0 to disable)",
    )
    p_opt.add_argument(
        "--max-labeled-demos",
        dest="max_labeled_demos",
        type=int,
        default=None,
        help="Maximum labeled demonstrations for MIPROv2 (defaults to yaw.toml "
        "[optimization].max_labeled_demos, 0 to disable)",
    )
    p_opt.add_argument(
        "--db-path",
        dest="db_path",
        type=str,
        default=None,
        help="Override sandbox database path (default: .yaw/optimize_yaw.db)",
    )
    p_opt.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Wipe existing optimize_yaw.db and optimize_* sidecars before starting from scratch "
            "(default: resume existing run)"
        ),
    )

    p_load = dspy_subs.add_parser(
        "load-program",
        parents=[common],
        help="Promote a compiled DSPy program JSON into production and hot-reload in-memory",
        allow_abbrev=False,
    )
    p_load.add_argument(
        "file",
        type=str,
        help="Path to compiled DSPy program JSON file to promote and load",
    )

    p_bench = dspy_subs.add_parser(
        "benchmark",
        parents=[common],
        help="Run HippoRAG 2 + DPR benchmark evaluation and optional parity gate",
        allow_abbrev=False,
    )
    p_bench.add_argument(
        "--dataset",
        dest="dataset",
        nargs="*",
        default=None,
        help="One or more dataset names or JSON files to benchmark",
    )
    p_bench.add_argument(
        "--gate",
        dest="gate",
        nargs="?",
        const="data/benchmarks/baselines.json",
        default=None,
        help="Evaluate against data/benchmarks/baselines.json (or optional path to gate JSON file)",
    )
    p_bench.add_argument(
        "--target",
        dest="target",
        type=str,
        default=None,
        help="Primary baseline target key in gate specification (default: monkeypatched_hipporag)",
    )
    p_bench.add_argument(
        "--max-queries",
        dest="max_queries",
        type=int,
        default=None,
        help="Maximum queries to evaluate per dataset (default: 25 or gate spec)",
    )
    p_bench.add_argument(
        "--index-queries",
        dest="index_queries",
        type=int,
        default=None,
        help="Queries pooled for distractor passage indexing (default: 35 or gate spec)",
    )
    p_bench.add_argument(
        "--query-offset",
        dest="query_offset",
        type=int,
        default=0,
        help="Starting query offset in dataset (default: 0)",
    )
    p_bench.add_argument(
        "--full-corpus",
        dest="full_corpus",
        action="store_true",
        help="Index the full dataset corpus instead of an N-query slice",
    )
    p_bench.add_argument(
        "--mode",
        dest="mode",
        choices=["eval", "retrieval"],
        default="eval",
        help="Benchmark mode: full QA+retrieval ('eval') or retrieval-only ('retrieval')",
    )
    p_bench.add_argument(
        "--compiled-program",
        dest="compiled_program",
        type=str,
        default=None,
        help="Optional path to compiled DSPy program JSON to load before evaluation",
    )
    p_bench.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Optional path to write full benchmark report JSON",
    )
    p_bench.add_argument(
        "--fresh",
        dest="fresh",
        action="store_true",
        help="Wipe existing benchmark sandbox before indexing",
    )

    # -------------------------------------------------------------------------
    # 4. yaw config {get|set}
    # -------------------------------------------------------------------------
    p_cfg = domain_subparsers.add_parser("config", help="Configuration management", allow_abbrev=False)
    cfg_subs = p_cfg.add_subparsers(dest="action", help="Config action")

    p_get = cfg_subs.add_parser("get", parents=[common], help="Read configuration value(s)", allow_abbrev=False)
    p_get.add_argument(
        "key",
        nargs="?",
        default=None,
        help="Configuration key to read (e.g. 'optimization.auto' or 'qa_model'). Omit to dump all.",
    )

    p_set = cfg_subs.add_parser("set", parents=[common], help="Update a configuration value", allow_abbrev=False)
    p_set.add_argument(
        "key",
        type=str,
        help="Configuration key to update (e.g. 'optimization.auto' or 'qa_model')",
    )
    p_set.add_argument(
        "value",
        type=str,
        help="New value for configuration key (int, float, bool, JSON array, or string)",
    )

    return parser


def is_daemon_locked(config: YawConfig) -> bool:
    """Return True if another process currently holds yaw.db.lock."""
    return is_file_locked(config.lock_path)


def _nonnegative_status_count(value: Any, fallback: int = 0) -> int:
    return value if type(value) is int and value >= 0 else fallback


def _read_db_summary(conn: sqlite3.Connection) -> tuple[int, int, int, int, int, int, list[Any]]:
    chunks, entities, facts, edges, openie = get_graph_counts(conn)
    entity_labels = 0
    violations: list[Any] = []
    with contextlib.suppress(sqlite3.Error):
        row = conn.execute("SELECT COUNT(*) FROM entity_labels").fetchone()
        entity_labels = int(row[0]) if row else 0
        violations = lint_graph_shacl(conn)
    return chunks, entities, facts, edges, openie, entity_labels, violations


def _inspect_active_optimization(config: YawConfig) -> dict[str, Any] | None:
    """Detect if any sandbox optimize_yaw*.db.lock is currently locked and read its live progress."""
    if not config.lock_path:
        return None
    base_dir = config.lock_path.parent
    if not base_dir.exists():
        return None

    lock_candidates = [base_dir / "optimize_yaw.db.lock", *sorted(base_dir.glob("optimize_yaw_*.db.lock"))]
    for lock_file in dict.fromkeys(lock_candidates):
        if lock_file.exists() and is_file_locked(lock_file):
            raw_name = lock_file.name
            stem = raw_name.removesuffix(".lock").removesuffix(".db")
            suffix = stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
            status_file = base_dir / (f"optimize_status_{suffix}.json" if suffix else "optimize_status.json")
            opt_db = base_dir / f"{stem}.db"

            opt_chunks = opt_entities = opt_facts = opt_edges = openie = 0
            if opt_db.exists():
                with (
                    contextlib.suppress(sqlite3.Error),
                    contextlib.closing(open_readonly_yaw_db(opt_db, timeout=1.0)) as conn,
                ):
                    opt_chunks, opt_entities, opt_facts, opt_edges, openie, _, _ = _read_db_summary(conn)

            snap: dict[str, Any] = {}
            if status_file.exists():
                with contextlib.suppress(OSError, json.JSONDecodeError, ValueError, TypeError):
                    parsed = json.loads(status_file.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        snap = parsed

            stats = snap.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}
            status = snap.get("status", {})
            if not isinstance(status, dict):
                status = {}
            active_op = status.get("active_op")
            if not isinstance(active_op, str) or not active_op:
                active_op = "optimizing"

            return {
                "active": True,
                "datestamp": suffix or "canonical",
                "sandbox_db": str(opt_db),
                "chunks": max(opt_chunks, _nonnegative_status_count(stats.get("chunks"))),
                "entities": max(opt_entities, _nonnegative_status_count(stats.get("entities"))),
                "facts": max(opt_facts, _nonnegative_status_count(stats.get("facts"))),
                "edges": max(opt_edges, _nonnegative_status_count(stats.get("edges"))),
                "openie_checkpoints": max(openie, _nonnegative_status_count(stats.get("openie_checkpoints"))),
                "active_op": active_op,
                "status": status,
            }
    return None


def inspect_status(config: YawConfig) -> dict[str, Any]:
    """Read live daemon status if running, offline yaw.db stats if stopped, and active optimization progress."""
    running = is_daemon_locked(config)
    chunks = entities = facts = edges = openie = entity_labels = 0
    shacl_conforms = True
    if config.db_path and Path(config.db_path).exists():
        with (
            contextlib.suppress(sqlite3.Error),
            contextlib.closing(open_readonly_yaw_db(config.db_path, timeout=1.0)) as conn,
        ):
            chunks, entities, facts, edges, openie, entity_labels, violations = _read_db_summary(conn)
            shacl_conforms = len(violations) == 0

    opt_info = _inspect_active_optimization(config)

    if running and config.status_path and Path(config.status_path).exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            data = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and isinstance(data.get("status"), dict)
                and isinstance(data.get("stats"), dict)
                and data["status"].get("lifecycle") in ("stopped", "starting", "ready", "stopping")
            ):
                data["stats"]["openie_checkpoints"] = max(
                    openie, _nonnegative_status_count(data["stats"].get("openie_checkpoints"))
                )
                data["stats"]["chunks"] = _nonnegative_status_count(data["stats"].get("chunks"), chunks)
                if not isinstance(data["status"].get("active_op"), (str, type(None))):
                    data["status"]["active_op"] = None
                data["stats"].setdefault("entity_labels", entity_labels)
                data["stats"].setdefault("shacl_conforms", shacl_conforms)
                if not data["status"].get("active_op") and openie > data["stats"].get("chunks", 0):
                    data["status"]["active_op"] = f"indexing: {openie} openie passages processed"
                    data["status"]["sync_state"] = "syncing"
                if opt_info:
                    data["active_optimization"] = opt_info
                return data

    active_op = None
    sync_state = "idle"
    if running and openie > chunks:
        active_op = f"indexing: {openie} openie passages processed"
        sync_state = "syncing"
    elif opt_info:
        active_op = f"optimization active in sandbox: {opt_info['active_op']}"

    status = DaemonStatus(
        lifecycle="ready" if running else "stopped",
        provider_ready=running,
        sync_state=sync_state,
        queue_depth=0,
        active_op=active_op,
    )
    stats = IndexStats(
        chunks=chunks,
        entities=entities,
        facts=facts,
        edges=edges,
        config_fingerprint=config.config_fingerprint(),
        openie_checkpoints=openie,
        entity_labels=entity_labels,
        shacl_conforms=shacl_conforms,
    )
    payload = {
        "status": status.to_dict(),
        "stats": stats.to_dict(),
    }
    if opt_info:
        payload["active_optimization"] = opt_info
    return payload


def _load_dataset_splits(
    dataset_args: Sequence[str] | str | None,
    *,
    workspace_root: Path | None = None,
    max_queries: int = 20,
    train_queries: int = 7,
    query_offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example] | None, list[str]]:
    import dspy

    def _dspy_example_factory(*, _inputs: tuple[str, ...] = ("question", "item_docs"), **kwargs: Any) -> dspy.Example:
        return dspy.Example(**kwargs).with_inputs(*_inputs)

    return load_dataset_splits(
        dataset_args,
        workspace_root=workspace_root,
        max_queries=max_queries,
        train_queries=train_queries,
        query_offset=query_offset,
        example_factory=_dspy_example_factory,
    )


def handle_daemon_run(
    config: YawConfig,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
    max_seconds: float | None = None,
) -> int:
    """Run the persistent daemon in the foreground until stopped."""
    runtime = YawRuntime(
        config,
        program=program,
        embedder=embedder,
        auto_poll=True,
    )

    def _sig_handler(_signum: int, _frame: Any) -> None:
        runtime.request_stop()

    signals_to_catch = [signal.SIGINT, signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signals_to_catch.append(sigbreak)

    prev_handlers: dict[int, Any] = {}
    with contextlib.suppress(ValueError):
        for sig in signals_to_catch:
            prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _sig_handler)

    try:
        runtime.start(sync_on_start=True)
        runtime.wait(timeout=max_seconds)
    finally:
        runtime.stop()
        with contextlib.suppress(ValueError):
            for sig, prev in prev_handlers.items():
                signal.signal(sig, prev)
    return 0


def handle_daemon_start(config: YawConfig, config_path: Path | None) -> int:
    """Spawn detached background daemon and wait for lock/status readiness."""
    if is_daemon_locked(config):
        payload = inspect_status(config)
        payload["already_running"] = True
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    cmd = [sys.executable, "-m", "yaw", "daemon", "run"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    popen_kwargs: dict[str, Any] = {
        "cwd": str(config.workspace_root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

    def _detach(p: subprocess.Popen[Any]) -> None:
        handle = getattr(p, "_handle", None)
        if handle is not None and hasattr(handle, "Close"):
            with contextlib.suppress(OSError):
                handle.Close()
        p.returncode = 0

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if is_daemon_locked(config):
            started_pid = proc.pid
            _detach(proc)
            payload = inspect_status(config)
            payload["started_pid"] = started_pid
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    if proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.terminate()
            proc.wait(timeout=5)
    raise YawRuntimeError("NOT_READY", "Background daemon failed to acquire lock before timeout.")


def handle_daemon_stop(config: YawConfig) -> int:
    """Signal a running daemon to stop and wait for yaw.db.lock release."""
    if not is_daemon_locked(config):
        sys.stdout.write(json.dumps({"stopped": True, "was_running": False}, indent=2) + "\n")
        return 0

    assert config.lock_path is not None
    lock_p = Path(config.lock_path)
    signaled_local = YawRuntime.signal_local_stop(lock_p)

    pid: int | None = None
    if not signaled_local and config.status_path is not None:
        status_p = Path(config.status_path)
        if status_p.exists():
            for _ in range(3):
                try:
                    snap = json.loads(status_p.read_text(encoding="utf-8"))
                    raw_pid = snap.get("status", {}).get("pid")
                    if isinstance(raw_pid, int):
                        pid = raw_pid
                        break
                except OSError, json.JSONDecodeError:
                    pass
                time.sleep(0.01)

    if pid is not None and pid != os.getpid() and os.name != "nt":
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_daemon_locked(config):
            sys.stdout.write(json.dumps({"stopped": True, "was_running": True}, indent=2) + "\n")
            return 0
        YawRuntime.signal_local_stop(lock_p)
        if os.name != "nt" and pid is None and config.status_path is not None:
            status_p = Path(config.status_path)
            if status_p.exists():
                with contextlib.suppress(OSError, json.JSONDecodeError):
                    snap = json.loads(status_p.read_text(encoding="utf-8"))
                    raw_pid = snap.get("status", {}).get("pid")
                    if isinstance(raw_pid, int) and raw_pid != os.getpid():
                        pid = raw_pid
                        os.kill(pid, signal.SIGTERM)
        time.sleep(0.05)

    if pid is not None and pid != os.getpid() and os.name != "nt":
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        kill_deadline = time.monotonic() + 5.0
        while time.monotonic() < kill_deadline:
            if not is_daemon_locked(config):
                break
            time.sleep(0.05)

    stopped = not is_daemon_locked(config)
    sys.stdout.write(json.dumps({"stopped": stopped, "was_running": True}, indent=2) + "\n")
    return 0 if stopped else 1


def handle_graph_build(
    config: YawConfig,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Execute a standalone incremental CocoIndex corpus sync and graph index update."""
    runtime = YawRuntime(config, program=program, embedder=embedder, auto_poll=False)
    runtime.start(sync_on_start=False)
    try:
        delta = runtime.sync_now()
        payload = {
            "added_chunks": len(delta.added_chunks),
            "deleted_chunks": len(delta.deleted_chunk_ids),
            "stats": runtime.stats().to_dict(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    finally:
        runtime.stop()


def handle_graph_rebuild(
    config: YawConfig,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Rebuild yaw.db and CocoIndex state from scratch."""
    runtime = YawRuntime(config, program=program, embedder=embedder, auto_poll=False)
    runtime.start(sync_on_start=False)
    try:
        stats = runtime.rebuild()
        payload = {
            "rebuilt": True,
            "stats": stats.to_dict(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    finally:
        runtime.stop()


def handle_graph_lint(
    config: YawConfig,
    *,
    as_turtle: bool = False,
    db_path: str | Path | None = None,
) -> int:
    """Validate closed-world SHACL graph shapes or emit W3C Turtle over a read-only WAL snapshot."""
    target_cfg = config.for_optimization(db_override=db_path) if db_path is not None else config
    db_file = Path(target_cfg.db_path).resolve() if target_cfg.db_path else None
    if db_file is not None and db_file.exists() and db_file.stat().st_size > 0:
        conn = open_readonly_yaw_db(db_file, timeout=5.0)
    else:
        conn = open_yaw_db(":memory:")

    try:
        chunks, entities, facts, edges, openie, entity_labels, violations = _read_db_summary(conn)

        if as_turtle:
            ttl = serialize_graph_turtle(conn, violations=violations)
            if not ttl.endswith("\n"):
                ttl += "\n"
            sys.stdout.write(ttl)
        else:
            edge_counts = {
                str(r[0]): int(r[1])
                for r in conn.execute("SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY kind").fetchall()
            }
            label_counts = {
                str(r[0]): int(r[1])
                for r in conn.execute(
                    "SELECT label_kind, COUNT(*) FROM entity_labels GROUP BY label_kind ORDER BY label_kind"
                ).fetchall()
            }
            payload = {
                "db_path": str(db_file) if db_file is not None else ":memory:",
                "conforms": len(violations) == 0,
                "violation_count": len(violations),
                "violations": [v.to_dict() for v in violations],
                "stats": {
                    "chunks": chunks,
                    "entities": entities,
                    "facts": facts,
                    "edges": edges,
                    "openie_checkpoints": openie,
                    "entity_labels": entity_labels,
                },
                "edge_counts": edge_counts,
                "label_counts": label_counts,
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    finally:
        conn.close()


def _clean_optimization_artifacts(opt_config: YawConfig, out_override: str | None = None) -> None:
    """Wipe canonical optimization SQLite DB and sidecar artifacts (preserving coco_lmdb)."""
    assert opt_config.db_path is not None
    assert opt_config.lock_path is not None
    assert opt_config.status_path is not None
    assert opt_config.compiled_program_path is not None
    if is_file_locked(Path(opt_config.lock_path)):
        raise YawRuntimeError(
            "LOCK_CONFLICT",
            f"Optimization database is currently locked: {opt_config.db_path}",
        )
    db_p = Path(opt_config.db_path)
    base_dir = db_p.parent
    suffix = db_p.stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
    for ext in ("", "-wal", "-shm", ".lock"):
        Path(str(db_p) + ext).unlink(missing_ok=True)
    Path(opt_config.status_path).unlink(missing_ok=True)
    telemetry_file = base_dir / (f"optimize_telemetry_{suffix}.json" if suffix else "optimize_telemetry.json")
    telemetry_file.unlink(missing_ok=True)
    frame_dir = base_dir / (f"optimize_kg_frames_{suffix}" if suffix else "optimize_kg_frames")
    shutil.rmtree(frame_dir, ignore_errors=True)
    out_path = Path(out_override).resolve() if out_override else Path(opt_config.compiled_program_path)
    stem = out_path.stem
    parent = out_path.parent
    for candidate in (
        out_path,
        parent / f"{stem}_audit.json",
        parent / f"{stem}_eval.json",
        parent / f"{stem}_pools.json",
        parent / f"{stem}_pareto_manifest.json",
    ):
        candidate.unlink(missing_ok=True)
    for member in parent.glob(f"{stem}_pareto_*.json"):
        member.unlink(missing_ok=True)
    shutil.rmtree(parent / f"{stem}_telemetry", ignore_errors=True)


def handle_dspy_optimize(
    config: YawConfig,
    args: argparse.Namespace,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Compile DSPy prompts in an isolated sandbox and save candidate JSON."""
    raw_db = getattr(args, "db_path", None)
    fresh = bool(getattr(args, "fresh", False))
    opt_config = config.for_optimization(db_override=raw_db)
    out_override = getattr(args, "output", None)
    assert opt_config.db_path is not None
    if fresh:
        _clean_optimization_artifacts(opt_config, out_override)
    elif opt_config.db_path.is_file() and opt_config.db_path.stat().st_size > 0:
        # Read metadata before loading datasets, so default resume recovers
        # the original slice and budget even when they were CLI overrides.
        with contextlib.closing(open_readonly_yaw_db(opt_config.db_path, timeout=5.0)) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'optimization_run'").fetchone()
        if row is not None:
            saved_run = json.loads(row[0])
            saved_output = Path(saved_run["output_path"])
            if out_override and Path(out_override).resolve() != saved_output:
                raise YawRuntimeError("INVALID_INPUT", f"Resume must use the original output path: {saved_output}")
            out_override = str(saved_output)
            for key, value in saved_run.get("cli_options", {}).items():
                if getattr(args, key, None) is None:
                    setattr(args, key, value)
        logger.info("Resuming optimization run from %s", opt_config.db_path)

    raw_dataset = getattr(args, "dataset", None)
    if raw_dataset is not None:
        dataset_args = parse_dataset_list(
            raw_dataset if isinstance(raw_dataset, (str, Sequence)) else [str(raw_dataset)]
        )
    elif config.optimize_datasets:
        dataset_args = list(config.optimize_datasets)
    else:
        dataset_args = ["2wikimultihopqa"]

    raw_max_q = getattr(args, "max_queries", None)
    max_queries = int(raw_max_q if raw_max_q is not None else (config.optimize_max_queries or 20))

    raw_offset = getattr(args, "query_offset", None)
    query_offset = int(raw_offset if raw_offset is not None else (config.optimize_query_offset or 0))

    raw_train_q = getattr(args, "train_queries", None)
    train_queries = int(raw_train_q if raw_train_q is not None else (config.optimize_train_queries or 7))

    raw_auto = getattr(args, "auto", None)
    auto = raw_auto or config.optimize_auto or "light"

    raw_trials = getattr(args, "num_trials", None)
    num_trials = int(raw_trials) if raw_trials is not None else None

    raw_candidates = getattr(args, "num_candidates", None)
    num_candidates = int(raw_candidates) if raw_candidates is not None else None

    raw_mode = getattr(args, "mode", None)
    mode = raw_mode or config.optimize_mode or "hybrid"

    trainset, valset, corpus_docs = _load_dataset_splits(
        dataset_args,
        workspace_root=config.workspace_root,
        max_queries=max_queries,
        train_queries=train_queries,
        query_offset=query_offset,
    )
    dataset_bundles = [resolve_dataset_bundle(arg, workspace_root=config.workspace_root) for arg in dataset_args]

    # Resolution order for sandbox: always isolate prompt compilation
    import dspy

    dspy.settings.configure(provide_traceback=True)
    raw_sleep = getattr(args, "sleep_seconds", None)
    sleep_seconds = float(raw_sleep) if raw_sleep is not None else float(config.optimize_sleep_seconds or 0.0)
    object.__setattr__(opt_config, "optimize_sleep_seconds", sleep_seconds)

    raw_bootstrapped = getattr(args, "max_bootstrapped_demos", None)
    max_bootstrapped_demos = (
        int(raw_bootstrapped) if raw_bootstrapped is not None else int(config.optimize_max_bootstrapped_demos or 0)
    )
    raw_labeled = getattr(args, "max_labeled_demos", None)
    max_labeled_demos = int(raw_labeled) if raw_labeled is not None else int(config.optimize_max_labeled_demos or 0)
    object.__setattr__(opt_config, "optimize_max_bootstrapped_demos", max_bootstrapped_demos)
    object.__setattr__(opt_config, "optimize_max_labeled_demos", max_labeled_demos)

    runtime = YawRuntime(opt_config, program=program, embedder=embedder, auto_poll=False)
    runtime.start(sync_on_start=False)

    try:
        saved_path = runtime.optimize(
            trainset=trainset,
            valset=valset,
            mode=mode,
            save_path=out_override or opt_config.compiled_program_path,
            auto=auto,
            num_candidates=num_candidates,
            num_trials=num_trials,
            sleep_seconds=sleep_seconds,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            corpus_docs=corpus_docs,
            dataset_bundles=dataset_bundles,
            run_options={
                "dataset": dataset_args,
                "max_queries": max_queries,
                "train_queries": train_queries,
                "query_offset": query_offset,
                "mode": mode,
                "auto": auto,
                "num_trials": num_trials,
                "num_candidates": num_candidates,
                "sleep_seconds": sleep_seconds,
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
            },
        )
        graph_rlm_audit = saved_path.parent / f"{saved_path.stem}_audit.json"
        gepa_telemetry_dir = saved_path.parent / f"{saved_path.stem}_telemetry"
        val_eval_file = saved_path.parent / f"{saved_path.stem}_eval.json"
        eval_summary: dict[str, Any] | None = None
        if val_eval_file.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError, ValueError, TypeError):
                raw_eval = json.loads(val_eval_file.read_text(encoding="utf-8"))
                if isinstance(raw_eval, list) and raw_eval:
                    dict_items = [it for it in raw_eval if isinstance(it, dict)]
                    f1_scores = [float(it.get("f1", it.get("score", 0.0))) for it in dict_items]
                    em_scores = [float(it.get("em", 0.0)) for it in dict_items]
                    apm_scores = [float(it.get("answer_passage_match", 0.0)) for it in dict_items]
                    eval_summary = {
                        "num_evaluated": len(raw_eval),
                        "mean_f1": (sum(f1_scores) / len(f1_scores) if f1_scores else 0.0),
                        "mean_em": (sum(em_scores) / len(em_scores) if em_scores else 0.0),
                        "answer_passage_match_rate": (sum(apm_scores) / len(apm_scores) if apm_scores else 0.0),
                    }
                    for k_val in (1, 2, 5, 10, 20):
                        rk_key = f"recall_at_{k_val}"
                        rk_scores = [float(it[rk_key]) for it in dict_items if rk_key in it]
                        if rk_scores:
                            eval_summary[f"mean_recall_at_{k_val}"] = sum(rk_scores) / len(rk_scores)
        payload: dict[str, Any] = {
            "optimized": True,
            "mode": mode,
            "auto": auto,
            "dataset": dataset_args[0] if len(dataset_args) == 1 else dataset_args,
            "datasets": dataset_args,
            "train_queries": len(trainset),
            "val_queries": len(valset) if valset is not None else 0,
            "compiled_program_path": str(saved_path),
            "val_eval_path": str(val_eval_file) if val_eval_file.exists() else None,
            "eval_summary": eval_summary,
            "sandbox_db_path": str(opt_config.db_path),
            "graph_rlm_audit_path": (str(graph_rlm_audit) if graph_rlm_audit.exists() else None),
            "gepa_telemetry_dir": (str(gepa_telemetry_dir) if gepa_telemetry_dir.exists() else None),
            "stats": runtime.stats().to_dict(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    finally:
        runtime.stop()


def handle_dspy_load_program(
    config: YawConfig,
    source_file: str | Path,
) -> int:
    """Validate program file, promote to config.compiled_program_path, and hot-reload in-memory."""
    src = Path(source_file).resolve()
    if not src.exists() or not src.is_file():
        raise YawRuntimeError("INVALID_INPUT", f"Compiled program file not found: {src}")

    try:
        content = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError("Root JSON must be a dictionary object")
    except Exception as exc:
        raise YawRuntimeError("INVALID_INPUT", f"Invalid compiled program JSON in {src}: {exc}") from exc

    target_path = (
        Path(config.compiled_program_path).resolve()
        if config.compiled_program_path
        else (config.workspace_root / ".yaw" / "compiled_program.json").resolve()
    )
    promote_compiled_program(src, target_path)

    if is_daemon_locked(config):
        assert config.lock_path is not None
        with YawRuntime._active_lock:
            local_rt = YawRuntime._active_instances.get(config.lock_path.resolve())

        if local_rt is not None:
            res = local_rt.reload_program(target_path)
            res["source_path"] = str(src)
            sys.stdout.write(json.dumps(res, indent=2) + "\n")
            return 0

        yaw_dir = config.workspace_root / ".yaw"
        yaw_dir.mkdir(parents=True, exist_ok=True)
        signal_file = yaw_dir / "reload.signal"
        result_file = yaw_dir / "reload.result"
        with contextlib.suppress(OSError):
            result_file.unlink(missing_ok=True)

        req_id = str(uuid.uuid4())
        payload = {
            "program_path": str(target_path),
            "source_path": str(src),
            "req_id": req_id,
        }
        write_text_atomic(signal_file, json.dumps(payload, indent=2))

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if result_file.exists():
                try:
                    res = json.loads(result_file.read_text(encoding="utf-8"))
                    if res.get("req_id") == req_id:
                        sys.stdout.write(json.dumps(res, indent=2) + "\n")
                        return 0 if res.get("reloaded", True) else 1
                except OSError, json.JSONDecodeError, ValueError, TypeError:
                    pass
            time.sleep(0.05)

        sys.stdout.write(
            json.dumps(
                {
                    "reloaded": False,
                    "staged": True,
                    "program_path": str(target_path),
                    "source_path": str(src),
                    "message": "Signal written; daemon did not acknowledge before timeout. Program staged for next "
                    "startup.",
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    sys.stdout.write(
        json.dumps(
            {
                "reloaded": False,
                "staged": True,
                "compiled_program_path": str(target_path),
                "source_path": str(src),
                "message": "Daemon is not running. Program promoted to compiled_program_path and will load on next "
                "daemon start.",
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def handle_dspy_benchmark(
    config: YawConfig,
    args: argparse.Namespace,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Execute per-dataset HippoRAG 2 + DPR benchmark evaluation and optional gate verification."""
    raw_gate = getattr(args, "gate", None)
    gate_path: Path | str | None = None
    if isinstance(raw_gate, str):
        cand_path = Path(raw_gate)
        if cand_path.exists():
            gate_path = cand_path
        elif raw_gate in ("data/benchmarks/baselines.json", "baselines.json"):
            default_gate = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "baselines.json"
            gate_path = default_gate if default_gate.exists() else raw_gate
        else:
            gate_path = raw_gate

    runtime = YawRuntime(config, program=program, embedder=embedder, auto_poll=False)
    res = runtime.benchmark(
        datasets=getattr(args, "dataset", None),
        gate_path=gate_path,
        max_queries=getattr(args, "max_queries", None),
        index_queries=getattr(args, "index_queries", None),
        query_offset=int(getattr(args, "query_offset", 0) or 0),
        full_corpus=bool(getattr(args, "full_corpus", False)),
        mode=getattr(args, "mode", "eval") or "eval",
        compiled_program=getattr(args, "compiled_program", None),
        output_path=getattr(args, "output", None),
        fresh=bool(getattr(args, "fresh", False)),
        target=getattr(args, "target", None),
    )
    sys.stdout.write(json.dumps(res, indent=2) + "\n")
    return 0 if res.get("overall_gate_passed", True) else 1


def handle_config_get(
    config: YawConfig,
    key: str | None = None,
    config_path: Path | None = None,
) -> int:
    """Read a specific key or dump entire configuration as JSON."""
    val = get_config_value(
        path=config_path,
        key=key,
        workspace_override=config.workspace_root,
    )
    payload = {key: val} if key is not None else val
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def handle_config_set(
    config: YawConfig,
    key: str,
    value_raw: str,
    config_path: Path | None = None,
) -> int:
    """Safely update a configuration setting in yaw.toml and verify validity."""
    resolved_key, parsed_val = set_config_value(
        key=key,
        raw_value=value_raw,
        path=config_path,
        workspace_override=config.workspace_root,
    )
    payload = {
        "updated": True,
        "key": resolved_key,
        "value": parsed_val,
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Main CLI dispatcher returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    domain = getattr(args, "domain", None)
    action = getattr(args, "action", None)

    if not domain or not action:
        parser.print_help(sys.stderr)
        return 2

    cfg_path = Path(args.config).resolve() if getattr(args, "config", None) else None
    try:
        config = load_config(cfg_path)

        # 1. yaw daemon {start|stop|status|run}
        if domain == "daemon":
            if action == "status":
                sys.stdout.write(json.dumps(inspect_status(config), indent=2) + "\n")
                return 0
            if action == "run":
                return handle_daemon_run(config, program=program, embedder=embedder)
            if action == "start":
                return handle_daemon_start(config, cfg_path)
            if action == "stop":
                return handle_daemon_stop(config)

        # 2. yaw graph {build|rebuild|lint}
        if domain == "graph":
            if action == "build":
                return handle_graph_build(config, program=program, embedder=embedder)
            if action == "rebuild":
                return handle_graph_rebuild(config, program=program, embedder=embedder)
            if action == "lint":
                return handle_graph_lint(
                    config,
                    as_turtle=bool(getattr(args, "turtle", False)),
                    db_path=getattr(args, "db_path", None),
                )

        # 3. yaw dspy {optimize|load-program|benchmark}
        if domain == "dspy":
            if action == "optimize":
                return handle_dspy_optimize(config, args, program=program, embedder=embedder)
            if action == "load-program":
                return handle_dspy_load_program(config, args.file)
            if action == "benchmark":
                return handle_dspy_benchmark(config, args, program=program, embedder=embedder)

        # 4. yaw config {get|set}
        if domain == "config":
            if action == "get":
                return handle_config_get(config, key=args.key, config_path=cfg_path)
            if action == "set":
                return handle_config_set(config, key=args.key, value_raw=args.value, config_path=cfg_path)

        parser.print_help(sys.stderr)
        return 2
    except YawRuntimeError as err:
        sys.stderr.write(json.dumps({"error": err.code, "message": err.message}) + "\n")
        return 1
    except (ValueError, FileNotFoundError) as err:
        sys.stderr.write(json.dumps({"error": "INVALID_INPUT", "message": str(err)}) + "\n")
        return 1


def cli_entrypoint() -> None:
    """Console script entrypoint for `yaw`."""
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "cli_entrypoint",
    "handle_config_get",
    "handle_config_set",
    "handle_daemon_run",
    "handle_daemon_start",
    "handle_daemon_stop",
    "handle_dspy_benchmark",
    "handle_dspy_load_program",
    "handle_dspy_optimize",
    "handle_graph_build",
    "handle_graph_lint",
    "handle_graph_rebuild",
    "inspect_status",
    "is_daemon_locked",
    "main",
]
