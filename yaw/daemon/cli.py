"""Canonical CLI entrypoint for the YAW persistent HippoRAG 2 daemon.

Grammar:
  yaw daemon {start|stop|status|run} [--config <path>]
  yaw graph {build|rebuild|lint} [--config <path>] [--turtle]
  yaw dspy {optimize|load-program|benchmark} [--flags]
  yaw config {get|set} [--flags]
"""

from __future__ import annotations

import argparse
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

import contextlib

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


def _emit(payload: Any, code: int = 0) -> int:
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return code


def _read_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError, TypeError:
            pass
    return None


def _run_rt(config: YawConfig, program: YawProgram | None, embedder: DSPyEmbeddingAdapter | None, fn: Any) -> int:
    rt = YawRuntime(config, program=program, embedder=embedder, auto_poll=False)
    rt.start(sync_on_start=False)
    try:
        return _emit(fn(rt))
    finally:
        rt.stop()


def build_parser() -> argparse.ArgumentParser:
    """Build hierarchical argument parser for `yaw <domain> <action> --<flags>`."""
    ap = argparse.ArgumentParser
    cfg_help = "Path to yaw.toml configuration file (default: ./yaw.toml)"
    common = ap(add_help=False, allow_abbrev=False)
    common.add_argument("--config", "-c", dest="config", type=str, default=argparse.SUPPRESS, help=cfg_help)
    parser = ap(prog="yaw", description="YAW: Native HippoRAG 2 & CocoIndex v1 persistent daemon.", allow_abbrev=False)
    parser.add_argument("--config", "-c", dest="config", type=str, default=None, help=cfg_help)
    dom_subs = parser.add_subparsers(dest="domain", help="Domain to operate on")

    tmap = {"int": int, "float": float, "str": str}
    specs = (
        (
            "daemon",
            "Daemon lifecycle commands",
            "Daemon action",
            (
                ("start", "Start persistent daemon in background", ()),
                ("stop", "Gracefully stop a running daemon", ()),
                ("status", "Inspect daemon status, graph index statistics, and active optimizations as JSON", ()),
                ("run", "Run persistent daemon in the foreground until stopped", ()),
            ),
        ),
        (
            "graph",
            "Standalone graph and corpus index commands",
            "Graph action",
            (
                ("build", "Run standalone incremental CocoIndex corpus sync and update yaw.db", ()),
                ("rebuild", "Wipe yaw.db and CocoIndex state and perform a full rebuild from scratch", ()),
                (
                    "lint",
                    "Run closed-world SHACL graph lint or emit W3C Turtle (RDF/SKOS/OWL/SHACL)",
                    (
                        (
                            "--turtle",
                            (
                                ("action", "store_true"),
                                ("help", "Emit W3C Turtle (RDF/SKOS/OWL/SHACL) serialization instead of JSON"),
                            ),
                        ),
                        (
                            "--db-path",
                            (
                                ("default", None),
                                (
                                    "help",
                                    (
                                        "Inspect a specific SQLite DB or 'latest' optimization sandbox instea"
                                        "d of config.db_path"
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        (
            "dspy",
            "DSPy prompt optimization, benchmark, and program management",
            "DSPy action",
            (
                (
                    "optimize",
                    "Compile and evolve DSPy prompts in an isolated sandbox and save candidate JSON",
                    (
                        (
                            "--auto",
                            (
                                ("choices", ("light", "medium", "heavy")),
                                (
                                    "help",
                                    (
                                        "GEPA/MIPROv2 evaluation budget mode (defaults to yaw.toml [optimizat"
                                        "ion].auto, then 'light')"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--mode",
                            (
                                ("choices", ("miprov2", "gepa", "hybrid")),
                                (
                                    "help",
                                    (
                                        "Prompt optimization mode (defaults to yaw.toml [optimization].mode, "
                                        "then 'hybrid')"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--dataset",
                            (
                                ("nargs", "*"),
                                (
                                    "help",
                                    (
                                        "One or more dataset names (enterprise_rag_bench, sample, 2wikimultih"
                                        "opqa, musique, hotpotqa) or path to JSON training file (defaults to "
                                        "yaw.toml [optimization].datasets)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--max-queries",
                            (
                                ("type", "int"),
                                (
                                    "help",
                                    (
                                        "Maximum queries to load (defaults to yaw.toml [optimization].max_que"
                                        "ries, max 50 on Tier 2 corpora)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--train-queries",
                            (
                                ("type", "int"),
                                (
                                    "help",
                                    (
                                        "Number of queries for training [0:K] (defaults to yaw.toml [optimiza"
                                        "tion].train_queries)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--query-offset",
                            (
                                ("type", "int"),
                                (
                                    "help",
                                    (
                                        "Starting query offset in dataset (defaults to yaw.toml [optimization"
                                        "].query_offset)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--output",
                            (("type", "str"), ("help", "Override output path for candidate compiled_program JSON")),
                        ),
                        (
                            "--num-trials",
                            (
                                ("type", "int"),
                                (
                                    "help",
                                    "Explicit number of optimization trials (sets auto to None for fast smoke tests)",
                                ),
                            ),
                        ),
                        (
                            "--num-candidates",
                            (
                                ("type", "int"),
                                ("help", "Explicit number of instruction/fewshot candidates per predictor"),
                            ),
                        ),
                        (
                            "--sleep",
                            (
                                ("dest", "sleep_seconds"),
                                ("type", "float"),
                                (
                                    "help",
                                    (
                                        "Sleep pause in seconds between LLM/embedding calls to manage thermal"
                                        " duty cycles (defaults to yaw.toml [optimization].sleep_seconds, def"
                                        "ault 0.0)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--max-bootstrapped-demos",
                            (
                                ("type", "int"),
                                (
                                    "help",
                                    (
                                        "Maximum bootstrapped demonstrations for MIPROv2 (defaults to yaw.tom"
                                        "l [optimization].max_bootstrapped_demos, 0 to disable)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--max-labeled-demos",
                            (
                                ("type", "int"),
                                (
                                    "help",
                                    (
                                        "Maximum labeled demonstrations for MIPROv2 (defaults to yaw.toml [op"
                                        "timization].max_labeled_demos, 0 to disable)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--db-path",
                            (
                                ("type", "str"),
                                ("help", "Override sandbox database path (default: .yaw/optimize_yaw.db)"),
                            ),
                        ),
                        (
                            "--fresh",
                            (
                                ("action", "store_true"),
                                (
                                    "help",
                                    (
                                        "Wipe existing optimize_yaw.db and optimize_* sidecars before startin"
                                        "g from scratch (default: resume existing run)"
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                (
                    "load-program",
                    "Promote a compiled DSPy program JSON into production and hot-reload in-memory",
                    (
                        (
                            "file",
                            (("type", "str"), ("help", "Path to compiled DSPy program JSON file to promote and load")),
                        ),
                    ),
                ),
                (
                    "benchmark",
                    "Run HippoRAG 2 + DPR benchmark evaluation and optional parity gate",
                    (
                        (
                            "--dataset",
                            (("nargs", "*"), ("help", "One or more dataset names or JSON files to benchmark")),
                        ),
                        (
                            "--gate",
                            (
                                ("nargs", "?"),
                                ("const", "data/benchmarks/baselines.json"),
                                (
                                    "help",
                                    (
                                        "Evaluate against data/benchmarks/baselines.json (or optional path to"
                                        " gate JSON file)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--target",
                            (
                                ("type", "str"),
                                (
                                    "help",
                                    (
                                        "Primary baseline target key in gate specification (default: monkeypa"
                                        "tched_hipporag)"
                                    ),
                                ),
                            ),
                        ),
                        (
                            "--max-queries",
                            (
                                ("type", "int"),
                                ("help", "Maximum queries to evaluate per dataset (default: 25 or gate spec)"),
                            ),
                        ),
                        (
                            "--index-queries",
                            (
                                ("type", "int"),
                                ("help", "Queries pooled for distractor passage indexing (default: 35 or gate spec)"),
                            ),
                        ),
                        (
                            "--query-offset",
                            (
                                ("type", "int"),
                                ("default", 0),
                                ("help", "Starting query offset in dataset (default: 0)"),
                            ),
                        ),
                        (
                            "--full-corpus",
                            (
                                ("action", "store_true"),
                                ("help", "Index the full dataset corpus instead of an N-query slice"),
                            ),
                        ),
                        (
                            "--mode",
                            (
                                ("choices", ("eval", "retrieval")),
                                ("default", "eval"),
                                ("help", "Benchmark mode: full QA+retrieval ('eval') or retrieval-only ('retrieval')"),
                            ),
                        ),
                        (
                            "--compiled-program",
                            (
                                ("type", "str"),
                                ("help", "Optional path to compiled DSPy program JSON to load before evaluation"),
                            ),
                        ),
                        ("--output", (("type", "str"), ("help", "Optional path to write full benchmark report JSON"))),
                        (
                            "--fresh",
                            (("action", "store_true"), ("help", "Wipe existing benchmark sandbox before indexing")),
                        ),
                    ),
                ),
            ),
        ),
        (
            "config",
            "Configuration management",
            "Config action",
            (
                (
                    "get",
                    "Read configuration value(s)",
                    (
                        (
                            "key",
                            (
                                ("nargs", "?"),
                                ("default", None),
                                (
                                    "help",
                                    (
                                        "Configuration key to read (e.g. 'optimization.auto' or 'qa_model'). "
                                        "Omit to dump all."
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                (
                    "set",
                    "Update a configuration value",
                    (
                        (
                            "key",
                            (
                                ("type", "str"),
                                ("help", "Configuration key to update (e.g. 'optimization.auto' or 'qa_model')"),
                            ),
                        ),
                        (
                            "value",
                            (
                                ("type", "str"),
                                ("help", "New value for configuration key (int, float, bool, JSON array, or string)"),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    parents = [common]
    for dom, dom_help, act_help, actions in specs:
        sub = dom_subs.add_parser(dom, help=dom_help, allow_abbrev=False).add_subparsers(dest="action", help=act_help)
        for act, act_h, args in actions:
            p = sub.add_parser(act, parents=parents, help=act_h, allow_abbrev=False)
            for flag, kw_items in args:
                kw: dict[str, Any] = dict(kw_items)
                if "type" in kw:
                    kw["type"] = tmap[kw["type"]]
                if "choices" in kw:
                    kw["choices"] = list(kw["choices"])
                p.add_argument(flag, **kw)
    return parser


def is_daemon_locked(config: YawConfig) -> bool:
    """Return True if another process currently holds yaw.db.lock."""
    return is_file_locked(config.lock_path)


def _nonnegative_status_count(value: Any, fallback: int = 0) -> int:
    return value if type(value) is int and value >= 0 else fallback


def _read_db_summary(conn: sqlite3.Connection) -> tuple[int, int, int, int, int, int, list[Any]]:
    chunks, entities, facts, edges, openie = get_graph_counts(conn)
    entity_labels, violations = 0, []
    try:
        row = conn.execute("SELECT COUNT(*) FROM entity_labels").fetchone()
        entity_labels = int(row[0]) if row else 0
        violations = lint_graph_shacl(conn)
    except sqlite3.Error:
        pass
    return chunks, entities, facts, edges, openie, entity_labels, violations


def _read_db_summary_path(db_path: Path | str) -> tuple[int, int, int, int, int, int, list[Any]]:
    try:
        conn = open_readonly_yaw_db(db_path, timeout=1.0)
        try:
            return _read_db_summary(conn)
        finally:
            conn.close()
    except sqlite3.Error:
        return 0, 0, 0, 0, 0, 0, []


def _inspect_active_optimization(config: YawConfig) -> dict[str, Any] | None:
    """Detect if any sandbox optimize_yaw*.db.lock is currently locked and read its live progress."""
    if not config.lock_path or not (base_dir := config.lock_path.parent).exists():
        return None
    for lock_file in (base_dir / "optimize_yaw.db.lock", *sorted(base_dir.glob("optimize_yaw_*.db.lock"))):
        if not (lock_file.exists() and is_file_locked(lock_file)):
            continue
        stem = lock_file.name[:-8]
        suffix = stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
        opt_db = base_dir / f"{stem}.db"
        opt_chunks, opt_entities, opt_facts, opt_edges, openie, _, _ = (
            _read_db_summary_path(opt_db) if opt_db.exists() else (0, 0, 0, 0, 0, 0, [])
        )
        raw = _read_json(base_dir / (f"optimize_status_{suffix}.json" if suffix else "optimize_status.json"))
        snap = raw if isinstance(raw, dict) else {}
        stats = s if isinstance((s := snap.get("stats")), dict) else {}
        status = s if isinstance((s := snap.get("status")), dict) else {}
        return {
            "active": True,
            "datestamp": suffix or "canonical",
            "sandbox_db": str(opt_db),
            "chunks": max(opt_chunks, _nonnegative_status_count(stats.get("chunks"))),
            "entities": max(opt_entities, _nonnegative_status_count(stats.get("entities"))),
            "facts": max(opt_facts, _nonnegative_status_count(stats.get("facts"))),
            "edges": max(opt_edges, _nonnegative_status_count(stats.get("edges"))),
            "openie_checkpoints": max(openie, _nonnegative_status_count(stats.get("openie_checkpoints"))),
            "active_op": op if isinstance((op := status.get("active_op")), str) and op else "optimizing",
            "status": status,
        }
    return None


def inspect_status(config: YawConfig) -> dict[str, Any]:
    """Read live daemon status if running, offline yaw.db stats if stopped, and active optimization progress."""
    running = is_daemon_locked(config)
    chunks, entities, facts, edges, openie, entity_labels, violations = (
        _read_db_summary_path(config.db_path)
        if config.db_path and Path(config.db_path).exists()
        else (0, 0, 0, 0, 0, 0, [])
    )
    shacl_conforms = not violations
    opt_info = _inspect_active_optimization(config)

    if (
        running
        and config.status_path
        and isinstance((data := _read_json(Path(config.status_path))), dict)
        and (
            isinstance((st := data.get("status")), dict)
            and isinstance((ds := data.get("stats")), dict)
            and st.get("lifecycle") in ("stopped", "starting", "ready", "stopping")
        )
    ):
        ds["openie_checkpoints"] = max(openie, _nonnegative_status_count(ds.get("openie_checkpoints")))
        ds["chunks"] = _nonnegative_status_count(ds.get("chunks"), chunks)
        if not isinstance(st.get("active_op"), (str, type(None))):
            st["active_op"] = None
        ds.setdefault("entity_labels", entity_labels)
        ds.setdefault("shacl_conforms", shacl_conforms)
        if not st.get("active_op") and openie > ds.get("chunks", 0):
            st["active_op"], st["sync_state"] = f"indexing: {openie} openie passages processed", "syncing"
        if opt_info:
            data["active_optimization"] = opt_info
        return data

    active_op, sync_state = None, "idle"
    if running and openie > chunks:
        active_op, sync_state = f"indexing: {openie} openie passages processed", "syncing"
    elif opt_info:
        active_op = f"optimization active in sandbox: {opt_info['active_op']}"

    payload: dict[str, Any] = {
        "status": DaemonStatus(
            lifecycle="ready" if running else "stopped",
            provider_ready=running,
            sync_state=sync_state,
            queue_depth=0,
            active_op=active_op,
        ).to_dict(),
        "stats": IndexStats(
            chunks=chunks,
            entities=entities,
            facts=facts,
            edges=edges,
            config_fingerprint=config.config_fingerprint(),
            openie_checkpoints=openie,
            entity_labels=entity_labels,
            shacl_conforms=shacl_conforms,
        ).to_dict(),
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

    return load_dataset_splits(
        dataset_args,
        workspace_root=workspace_root,
        max_queries=max_queries,
        train_queries=train_queries,
        query_offset=query_offset,
        example_factory=lambda *, _inputs=("question", "item_docs"), **kw: dspy.Example(**kw).with_inputs(*_inputs),
    )


def handle_daemon_run(
    config: YawConfig,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
    max_seconds: float | None = None,
) -> int:
    """Run the persistent daemon in the foreground until stopped."""
    runtime = YawRuntime(config, program=program, embedder=embedder, auto_poll=True)
    sigs = (signal.SIGINT, signal.SIGTERM, *((b,) if (b := getattr(signal, "SIGBREAK", None)) is not None else ()))

    def handler(_s, _f):
        return runtime.request_stop()

    prev: dict[int, Any] = {}
    try:
        for sig in sigs:
            prev[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
    except ValueError:
        pass
    try:
        runtime.start(sync_on_start=True)
        runtime.wait(timeout=max_seconds)
    finally:
        runtime.stop()
        try:
            for sig, h in prev.items():
                signal.signal(sig, h)
        except ValueError:
            pass
    return 0


def handle_daemon_start(config: YawConfig, config_path: Path | None) -> int:
    """Spawn detached background daemon and wait for lock/status readiness."""
    if is_daemon_locked(config):
        return _emit({**inspect_status(config), "already_running": True})

    cmd = [
        sys.executable,
        "-m",
        "yaw",
        "daemon",
        "run",
        *(["--config", str(config_path)] if config_path is not None else ()),
    ]
    extra: dict[str, Any] = (
        {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        }
        if os.name == "nt"
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(config.workspace_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **extra,
    )

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if is_daemon_locked(config):
            started_pid = proc.pid
            if (handle := getattr(proc, "_handle", None)) is not None and hasattr(handle, "Close"):
                with contextlib.suppress(OSError):
                    handle.Close()
            proc.returncode = 0
            return _emit({**inspect_status(config), "started_pid": started_pid})
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except OSError:
            pass
    raise YawRuntimeError("NOT_READY", "Background daemon failed to acquire lock before timeout.")


def handle_daemon_stop(config: YawConfig) -> int:
    """Signal a running daemon to stop and wait for yaw.db.lock release."""
    if not is_daemon_locked(config):
        return _emit({"stopped": True, "was_running": False})

    assert config.lock_path is not None
    lock_p = Path(config.lock_path)
    status_p = Path(config.status_path) if config.status_path is not None else None
    my_pid, is_posix = os.getpid(), os.name != "nt"

    def _get_pid() -> int | None:
        if status_p is not None and isinstance((snap := _read_json(status_p)), dict):
            raw = snap.get("status", {}).get("pid")
            return raw if isinstance(raw, int) else None
        return None

    pid: int | None = None
    if not YawRuntime.signal_local_stop(lock_p) and status_p is not None and status_p.exists():
        for _ in range(3):
            if (pid := _get_pid()) is not None:
                break
            time.sleep(0.01)

    if is_posix and pid is not None and pid != my_pid:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_daemon_locked(config):
            return _emit({"stopped": True, "was_running": True})
        YawRuntime.signal_local_stop(lock_p)
        if is_posix and pid is None and (cand := _get_pid()) is not None and cand != my_pid:
            pid = cand
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
        time.sleep(0.05)

    if is_posix and pid is not None and pid != my_pid:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        kill_deadline = time.monotonic() + 5.0
        while time.monotonic() < kill_deadline and is_daemon_locked(config):
            time.sleep(0.05)

    stopped = not is_daemon_locked(config)
    return _emit({"stopped": stopped, "was_running": True}, 0 if stopped else 1)


def handle_graph_build(
    config: YawConfig, *, program: YawProgram | None = None, embedder: DSPyEmbeddingAdapter | None = None
) -> int:
    """Execute a standalone incremental CocoIndex corpus sync and graph index update."""
    return _run_rt(
        config,
        program,
        embedder,
        lambda rt: {
            "added_chunks": len((d := rt.sync_now()).added_chunks),
            "deleted_chunks": len(d.deleted_chunk_ids),
            "stats": rt.stats().to_dict(),
        },
    )


def handle_graph_rebuild(
    config: YawConfig, *, program: YawProgram | None = None, embedder: DSPyEmbeddingAdapter | None = None
) -> int:
    """Rebuild yaw.db and CocoIndex state from scratch."""
    return _run_rt(config, program, embedder, lambda rt: {"rebuilt": True, "stats": rt.rebuild().to_dict()})


def handle_graph_lint(config: YawConfig, *, as_turtle: bool = False, db_path: str | Path | None = None) -> int:
    """Validate closed-world SHACL graph shapes or emit W3C Turtle over a read-only WAL snapshot."""
    target_cfg = config.for_optimization(db_override=db_path) if db_path is not None else config
    db_file = Path(target_cfg.db_path).resolve() if target_cfg.db_path else None
    conn = (
        open_readonly_yaw_db(db_file, timeout=5.0)
        if db_file is not None and db_file.exists() and db_file.stat().st_size > 0
        else open_yaw_db(":memory:")
    )
    try:
        chunks, entities, facts, edges, openie, entity_labels, violations = _read_db_summary(conn)
        if as_turtle:
            ttl = serialize_graph_turtle(conn, violations=violations)
            sys.stdout.write(ttl if ttl.endswith("\n") else ttl + "\n")
            return 0
        return _emit(
            {
                "db_path": str(db_file) if db_file is not None else ":memory:",
                "conforms": not violations,
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
                "edge_counts": {
                    str(r[0]): int(r[1])
                    for r in conn.execute("SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY kind")
                },
                "label_counts": {
                    str(r[0]): int(r[1])
                    for r in conn.execute(
                        "SELECT label_kind, COUNT(*) FROM entity_labels GROUP BY label_kind ORDER BY label_kind"
                    )
                },
            }
        )
    finally:
        conn.close()


def _clean_optimization_artifacts(opt_config: YawConfig, out_override: str | None = None) -> None:
    """Wipe canonical optimization SQLite DB and sidecar artifacts (preserving coco_lmdb)."""
    assert (
        opt_config.db_path is not None
        and opt_config.lock_path is not None
        and opt_config.status_path is not None
        and opt_config.compiled_program_path is not None
    )
    if is_file_locked(Path(opt_config.lock_path)):
        raise YawRuntimeError("LOCK_CONFLICT", f"Optimization database is currently locked: {opt_config.db_path}")
    db_p = Path(opt_config.db_path)
    base_dir = db_p.parent
    suffix = db_p.stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
    tail = f"_{suffix}" if suffix else ""
    for ext in ("", "-wal", "-shm", ".lock"):
        Path(f"{db_p}{ext}").unlink(missing_ok=True)
    Path(opt_config.status_path).unlink(missing_ok=True)
    (base_dir / f"optimize_telemetry{tail}.json").unlink(missing_ok=True)
    shutil.rmtree(base_dir / f"optimize_kg_frames{tail}", ignore_errors=True)
    shutil.rmtree(base_dir / f"optimize_coco{tail}", ignore_errors=True)
    out_path = Path(out_override).resolve() if out_override else Path(opt_config.compiled_program_path)
    stem, parent = out_path.stem, out_path.parent
    for sfx in ("", "_audit", "_eval", "_pools", "_pareto_manifest"):
        (out_path if not sfx else parent / f"{stem}{sfx}.json").unlink(missing_ok=True)
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
    opt_config = config.for_optimization(db_override=getattr(args, "db_path", None))
    out_override = getattr(args, "output", None)
    assert opt_config.db_path is not None
    if getattr(args, "fresh", False):
        _clean_optimization_artifacts(opt_config, out_override)
    elif opt_config.db_path.is_file() and opt_config.db_path.stat().st_size > 0:
        conn = open_readonly_yaw_db(opt_config.db_path, timeout=5.0)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'optimization_run'").fetchone()
        finally:
            conn.close()
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

    raw_ds = getattr(args, "dataset", None)
    dataset_args = (
        parse_dataset_list(raw_ds if isinstance(raw_ds, (str, Sequence)) else [str(raw_ds)])
        if raw_ds is not None
        else (list(config.optimize_datasets) if config.optimize_datasets else ["2wikimultihopqa"])
    )
    opts: dict[str, Any] = {"dataset": dataset_args}
    for k, cast, fb in (
        ("max_queries", int, config.optimize_max_queries or 20),
        ("train_queries", int, config.optimize_train_queries or 7),
        ("query_offset", int, config.optimize_query_offset or 0),
        ("mode", str, config.optimize_mode or "hybrid"),
        ("auto", str, config.optimize_auto or "light"),
        ("num_trials", int, None),
        ("num_candidates", int, None),
        ("sleep_seconds", float, float(config.optimize_sleep_seconds or 0.0)),
        ("max_bootstrapped_demos", int, config.optimize_max_bootstrapped_demos),
        ("max_labeled_demos", int, config.optimize_max_labeled_demos),
    ):
        v = getattr(args, k, None)
        opts[k] = (
            cast(fb)
            if (v is None or (k in ("mode", "auto") and not v)) and fb is not None
            else (cast(v) if v is not None else None)
        )

    trainset, valset, corpus_docs = _load_dataset_splits(
        dataset_args,
        workspace_root=config.workspace_root,
        max_queries=opts["max_queries"],
        train_queries=opts["train_queries"],
        query_offset=opts["query_offset"],
    )
    dataset_bundles = [resolve_dataset_bundle(arg, workspace_root=config.workspace_root) for arg in dataset_args]

    import dspy

    dspy.settings.configure(provide_traceback=True)
    object.__setattr__(opt_config, "optimize_sleep_seconds", opts["sleep_seconds"])
    object.__setattr__(opt_config, "optimize_max_bootstrapped_demos", opts["max_bootstrapped_demos"])
    object.__setattr__(opt_config, "optimize_max_labeled_demos", opts["max_labeled_demos"])

    def _run(rt: YawRuntime) -> dict[str, Any]:
        saved_path = rt.optimize(
            trainset=trainset,
            valset=valset,
            mode=opts["mode"],
            save_path=out_override or opt_config.compiled_program_path,
            auto=opts["auto"],
            num_candidates=opts["num_candidates"],
            num_trials=opts["num_trials"],
            sleep_seconds=opts["sleep_seconds"],
            max_bootstrapped_demos=opts["max_bootstrapped_demos"],
            max_labeled_demos=opts["max_labeled_demos"],
            corpus_docs=corpus_docs,
            dataset_bundles=dataset_bundles,
            progress_callback=lambda msg: print(msg, file=sys.stderr, flush=True),
            run_options=opts,
        )
        parent, stem = saved_path.parent, saved_path.stem
        graph_rlm_audit, gepa_telemetry_dir, val_eval_file = (
            parent / f"{stem}_audit.json",
            parent / f"{stem}_telemetry",
            parent / f"{stem}_eval.json",
        )
        eval_summary: dict[str, Any] | None = None
        if isinstance((raw_eval := _read_json(val_eval_file)), list) and raw_eval:
            items = [it for it in raw_eval if isinstance(it, dict)]
            n = len(items)
            eval_summary = {
                "num_evaluated": len(raw_eval),
                "mean_f1": sum([float(it.get("f1", it.get("score", 0.0))) for it in items]) / n if n else 0.0,
                "mean_em": sum([float(it.get("em", 0.0)) for it in items]) / n if n else 0.0,
                "answer_passage_match_rate": sum([float(it.get("answer_passage_match", 0.0)) for it in items]) / n
                if n
                else 0.0,
            }
            for rk_k, out_k in (
                ("recall_at_1", "mean_recall_at_1"),
                ("recall_at_2", "mean_recall_at_2"),
                ("recall_at_5", "mean_recall_at_5"),
                ("recall_at_10", "mean_recall_at_10"),
                ("recall_at_20", "mean_recall_at_20"),
            ):
                if vals := [float(it[rk_k]) for it in items if rk_k in it]:
                    eval_summary[out_k] = sum(vals) / len(vals)
        return {
            "optimized": True,
            "mode": opts["mode"],
            "auto": opts["auto"],
            "dataset": dataset_args[0] if len(dataset_args) == 1 else dataset_args,
            "datasets": dataset_args,
            "train_queries": len(trainset),
            "val_queries": len(valset) if valset is not None else 0,
            "compiled_program_path": str(saved_path),
            "val_eval_path": str(val_eval_file) if val_eval_file.exists() else None,
            "eval_summary": eval_summary,
            "sandbox_db_path": str(opt_config.db_path),
            "graph_rlm_audit_path": str(graph_rlm_audit) if graph_rlm_audit.exists() else None,
            "gepa_telemetry_dir": str(gepa_telemetry_dir) if gepa_telemetry_dir.exists() else None,
            "stats": rt.stats().to_dict(),
        }

    return _run_rt(opt_config, program, embedder, _run)


def handle_dspy_load_program(config: YawConfig, source_file: str | Path) -> int:
    """Validate program file, promote to config.compiled_program_path, and hot-reload in-memory."""
    src = Path(source_file).resolve()
    if not src.exists() or not src.is_file():
        raise YawRuntimeError("INVALID_INPUT", f"Compiled program file not found: {src}")
    try:
        if not isinstance(json.loads(src.read_text(encoding="utf-8")), dict):
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
            return _emit({**local_rt.reload_program(target_path), "source_path": str(src)})

        yaw_dir = config.workspace_root / ".yaw"
        yaw_dir.mkdir(parents=True, exist_ok=True)
        signal_file, result_file = yaw_dir / "reload.signal", yaw_dir / "reload.result"
        with contextlib.suppress(OSError):
            result_file.unlink(missing_ok=True)
        req_id = str(uuid.uuid4())
        write_text_atomic(
            signal_file,
            json.dumps({"program_path": str(target_path), "source_path": str(src), "req_id": req_id}, indent=2),
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if isinstance((res := _read_json(result_file)), dict) and res.get("req_id") == req_id:
                return _emit(res, 0 if res.get("reloaded", True) else 1)
            time.sleep(0.05)
        return _emit(
            {
                "reloaded": False,
                "staged": True,
                "program_path": str(target_path),
                "source_path": str(src),
                ("message"): (
                    "Signal written; daemon did not acknowledge before timeout. Program staged for next startup."
                ),
            }
        )

    return _emit(
        {
            "reloaded": False,
            "staged": True,
            "compiled_program_path": str(target_path),
            "source_path": str(src),
            ("message"): (
                "Daemon is not running. Program promoted to compiled_program_path and will load on next daemon start."
            ),
        }
    )


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
        if (cand := Path(raw_gate)).exists():
            gate_path = cand
        elif (
            raw_gate in ("data/benchmarks/baselines.json", "baselines.json")
            and (
                default_gate := Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "baselines.json"
            ).exists()
        ):
            gate_path = default_gate
        else:
            gate_path = raw_gate

    res = YawRuntime(config, program=program, embedder=embedder, auto_poll=False).benchmark(
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
    return _emit(res, 0 if res.get("overall_gate_passed", True) else 1)


def handle_config_get(config: YawConfig, key: str | None = None, config_path: Path | None = None) -> int:
    """Read a specific key or dump entire configuration as JSON."""
    val = get_config_value(path=config_path, key=key, workspace_override=config.workspace_root)
    return _emit({key: val} if key is not None else val)


def handle_config_set(config: YawConfig, key: str, value_raw: str, config_path: Path | None = None) -> int:
    """Safely update a configuration setting in yaw.toml and verify validity."""
    resolved_key, parsed_val = set_config_value(
        key=key, raw_value=value_raw, path=config_path, workspace_override=config.workspace_root
    )
    return _emit({"updated": True, "key": resolved_key, "value": parsed_val})


def main(
    argv: Sequence[str] | None = None,
    *,
    program: YawProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Main CLI dispatcher returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    domain, action = getattr(args, "domain", None), getattr(args, "action", None)
    if not domain or not action:
        parser.print_help(sys.stderr)
        return 2

    cfg_path = Path(args.config).resolve() if getattr(args, "config", None) else None
    try:
        config = load_config(cfg_path)
        if domain == "daemon":
            if action == "status":
                return _emit(inspect_status(config))
            if action == "run":
                return handle_daemon_run(config, program=program, embedder=embedder)
            if action == "start":
                return handle_daemon_start(config, cfg_path)
            if action == "stop":
                return handle_daemon_stop(config)
        elif domain == "graph":
            if action == "build":
                return handle_graph_build(config, program=program, embedder=embedder)
            if action == "rebuild":
                return handle_graph_rebuild(config, program=program, embedder=embedder)
            if action == "lint":
                return handle_graph_lint(
                    config, as_turtle=bool(getattr(args, "turtle", False)), db_path=getattr(args, "db_path", None)
                )
        elif domain == "dspy":
            if action == "optimize":
                return handle_dspy_optimize(config, args, program=program, embedder=embedder)
            if action == "load-program":
                return handle_dspy_load_program(config, args.file)
            if action == "benchmark":
                return handle_dspy_benchmark(config, args, program=program, embedder=embedder)
        elif domain == "config":
            if action == "get":
                return handle_config_get(config, key=args.key, config_path=cfg_path)
            if action == "set":
                return handle_config_set(config, key=args.key, value_raw=args.value, config_path=cfg_path)
        parser.print_help(sys.stderr)
        return 2
    except (YawRuntimeError, ValueError, FileNotFoundError) as err:
        code, msg = (err.code, err.message) if isinstance(err, YawRuntimeError) else ("INVALID_INPUT", str(err))
        sys.stderr.write(json.dumps({"error": code, "message": msg}) + "\n")
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
