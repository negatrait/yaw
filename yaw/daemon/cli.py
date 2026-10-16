"""Canonical CLI entrypoint for the YAW persistent HippoRAG 2 daemon."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dspy
import filelock

from yaw.core import (
    DaemonStatus,
    IndexStats,
    YawConfig,
    YawRuntimeError,
    load_config,
)
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, HippoRAGProgram
from yaw.daemon.graph.storage import get_graph_counts, open_yaw_db
from yaw.daemon.runtime import YawRuntime


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for `python -m yaw {run|start|status|sync|stop|rebuild|optimize} --config yaw.toml`."""
    common = argparse.ArgumentParser(add_help=False)
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
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        type=str,
        default=None,
        help="Path to yaw.toml configuration file (default: ./yaw.toml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    subparsers.add_parser(
        "run",
        parents=[common],
        help="Run the persistent YAW daemon in the foreground",
    )
    subparsers.add_parser(
        "start",
        parents=[common],
        help="Start the persistent YAW daemon in the background",
    )
    subparsers.add_parser(
        "status",
        parents=[common],
        help="Inspect daemon status and graph index statistics as JSON",
    )
    subparsers.add_parser(
        "sync",
        parents=[common],
        help="Run an incremental CocoIndex corpus sync and update yaw.db",
    )
    subparsers.add_parser(
        "stop",
        parents=[common],
        help="Gracefully stop a running YAW daemon",
    )
    subparsers.add_parser(
        "rebuild",
        parents=[common],
        help="Rebuild yaw.db and CocoIndex state from scratch",
    )

    p_opt = subparsers.add_parser(
        "optimize",
        parents=[common],
        help="Compile and evolve DSPy prompts (MIPROv2 / GEPA) and save compiled_program.json",
    )
    p_opt.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to JSON list of {question, gold_answers, gold_docs} training examples",
    )
    p_opt.add_argument(
        "--mode",
        choices=["miprov2", "gepa", "hybrid"],
        default="hybrid",
        help="Prompt optimization mode (default: hybrid)",
    )
    p_opt.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path for compiled_program.json",
    )

    return parser


def is_daemon_locked(config: YawConfig) -> bool:
    """Return True if another process currently holds yaw.db.lock."""
    assert config.lock_path is not None
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(config.lock_path), timeout=0)
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout:
        return True
    else:
        lock.release()
        return False


def inspect_status(config: YawConfig) -> dict[str, Any]:
    """Read live daemon status if running, or offline yaw.db statistics if stopped."""
    running = is_daemon_locked(config)
    if running and config.status_path and Path(config.status_path).exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            data = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "status" in data and "stats" in data:
                return data

    chunks = entities = facts = edges = 0
    if config.db_path and Path(config.db_path).exists():
        conn = open_yaw_db(config.db_path)
        try:
            chunks, entities, facts, edges = get_graph_counts(conn)
        finally:
            conn.close()

    status = DaemonStatus(
        lifecycle="ready" if running else "stopped",
        provider_ready=running,
        sync_state="idle",
        queue_depth=0,
    )
    stats = IndexStats(
        chunks=chunks,
        entities=entities,
        facts=facts,
        edges=edges,
        config_fingerprint=config.config_fingerprint(),
    )
    return {
        "status": status.to_dict(),
        "stats": stats.to_dict(),
    }


def _load_training_examples(dataset_path: str | None) -> list[dspy.Example]:
    if dataset_path is not None:
        raw_items = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        return [
            dspy.Example(
                question=str(item["question"]),
                gold_answers=list(item.get("gold_answers", item.get("answer", []))),
                gold_docs=list(item.get("gold_docs", [])),
            ).with_inputs("question")
            for item in raw_items
        ]

    return [
        dspy.Example(
            question="When was the University of Southampton founded?",
            gold_answers=["1862"],
            gold_docs=[],
        ).with_inputs("question")
    ]


def handle_run(
    config: YawConfig,
    *,
    program: HippoRAGProgram | None = None,
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
    runtime.start(sync_on_start=True)

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
        runtime.wait(timeout=max_seconds)
    finally:
        runtime.stop()
        with contextlib.suppress(ValueError):
            for sig, prev in prev_handlers.items():
                signal.signal(sig, prev)
    return 0


def handle_start(config: YawConfig, config_path: Path | None) -> int:
    """Spawn detached background daemon and wait for lock/status readiness."""
    if is_daemon_locked(config):
        payload = inspect_status(config)
        payload["already_running"] = True
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    cmd = [sys.executable, "-m", "yaw", "run"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    popen_kwargs: dict[str, Any] = {
        "cwd": str(config.workspace_root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if is_daemon_locked(config):
            payload = inspect_status(config)
            payload["started_pid"] = proc.pid
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    raise YawRuntimeError("NOT_READY", "Background daemon failed to acquire lock before timeout.")


def handle_stop(config: YawConfig) -> int:
    """Signal a running daemon to stop and wait for yaw.db.lock release."""
    if not is_daemon_locked(config):
        sys.stdout.write(json.dumps({"stopped": True, "was_running": False}, indent=2) + "\n")
        return 0

    assert config.lock_path is not None
    signaled_local = YawRuntime.signal_local_stop(config.lock_path)

    pid: int | None = None
    if not signaled_local and config.status_path is not None:
        status_p = Path(config.status_path)
        if status_p.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                snap = json.loads(status_p.read_text(encoding="utf-8"))
                raw_pid = snap.get("status", {}).get("pid")
                if isinstance(raw_pid, int):
                    pid = raw_pid

    if pid is not None and pid != os.getpid():
        polite_sig = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)
        with contextlib.suppress(OSError):
            os.kill(pid, polite_sig)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_daemon_locked(config):
            sys.stdout.write(json.dumps({"stopped": True, "was_running": True}, indent=2) + "\n")
            return 0
        time.sleep(0.05)

    if pid is not None and pid != os.getpid():
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


def handle_sync(
    config: YawConfig,
    *,
    program: HippoRAGProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Execute a synchronous incremental CocoIndex sync."""
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


def handle_rebuild(
    config: YawConfig,
    *,
    program: HippoRAGProgram | None = None,
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


def handle_optimize(
    config: YawConfig,
    args: argparse.Namespace,
    *,
    program: HippoRAGProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Compile DSPy prompts via MIPROv2/GEPA and save compiled_program.json."""
    trainset = _load_training_examples(getattr(args, "dataset", None))
    mode = getattr(args, "mode", "hybrid")
    out_override = getattr(args, "output", None)

    runtime = YawRuntime(config, program=program, embedder=embedder, auto_poll=False)
    runtime.start(sync_on_start=True)
    try:
        saved_path = runtime.optimize(
            trainset=trainset,
            mode=mode,
            save_path=out_override,
        )
        payload = {
            "optimized": True,
            "mode": mode,
            "compiled_program_path": str(saved_path),
            "stats": runtime.stats().to_dict(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    finally:
        runtime.stop()


def main(
    argv: Sequence[str] | None = None,
    *,
    program: HippoRAGProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Main CLI dispatcher returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help(sys.stderr)
        return 2

    cfg_path = Path(args.config).resolve() if getattr(args, "config", None) else None
    try:
        config = load_config(cfg_path)
        if args.command == "status":
            sys.stdout.write(json.dumps(inspect_status(config), indent=2) + "\n")
            return 0
        if args.command == "run":
            return handle_run(config, program=program, embedder=embedder)
        if args.command == "start":
            return handle_start(config, cfg_path)
        if args.command == "stop":
            return handle_stop(config)
        if args.command == "sync":
            return handle_sync(config, program=program, embedder=embedder)
        if args.command == "rebuild":
            return handle_rebuild(config, program=program, embedder=embedder)
        if args.command == "optimize":
            return handle_optimize(config, args, program=program, embedder=embedder)
        return 2
    except YawRuntimeError as err:
        sys.stderr.write(json.dumps({"error": err.code, "message": err.message}) + "\n")
        return 1


__all__ = [
    "build_parser",
    "handle_optimize",
    "handle_rebuild",
    "handle_run",
    "handle_start",
    "handle_stop",
    "handle_sync",
    "inspect_status",
    "is_daemon_locked",
    "main",
]
