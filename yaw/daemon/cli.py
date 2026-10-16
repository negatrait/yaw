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
        "--datasets",
        dest="dataset",
        nargs="*",
        default=None,
        help=(
            "One or more Tier 1 dataset names (sample, 2wikimultihopqa, musique, hotpotqa) "
            "or path to a JSON list of training examples (defaults to yaw.toml [optimization].datasets)"
        ),
    )
    p_opt.add_argument(
        "--mode",
        "--evolve",
        dest="mode",
        choices=["miprov2", "gepa", "hybrid"],
        default=None,
        help="Prompt optimization mode (defaults to yaw.toml [optimization].mode)",
    )
    p_opt.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries to load from dataset slice (defaults to yaw.toml [optimization].max_queries, max 50 on Tier 2 corpora)",
    )
    p_opt.add_argument(
        "--train-queries",
        type=int,
        default=None,
        help="Number of queries for training [0:K], leaving [K:max_queries] for validation (defaults to yaw.toml [optimization].train_queries)",
    )
    p_opt.add_argument(
        "--query-offset",
        type=int,
        default=None,
        help="Starting query offset in the dataset (defaults to yaw.toml [optimization].query_offset)",
    )
    p_opt.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Optional explicit GEPA evaluation budget (defaults to yaw.toml [optimization].max_metric_calls)",
    )
    p_opt.add_argument(
        "--output",
        "--compiled-program",
        dest="output",
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
    chunks = entities = facts = edges = openie = 0
    if config.db_path and Path(config.db_path).exists():
        conn = open_yaw_db(config.db_path)
        try:
            chunks, entities, facts, edges, openie = get_graph_counts(conn)
        finally:
            conn.close()

    if running and config.status_path and Path(config.status_path).exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            data = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "status" in data and "stats" in data:
                data["stats"]["openie_checkpoints"] = max(
                    openie, data["stats"].get("openie_checkpoints", 0)
                )
                if not data["status"].get("active_op") and openie > data["stats"].get("chunks", 0):
                    data["status"]["active_op"] = f"indexing: {openie} openie passages processed"
                    data["status"]["sync_state"] = "syncing"
                return data

    status = DaemonStatus(
        lifecycle="ready" if running else "stopped",
        provider_ready=running,
        sync_state="syncing" if (running and openie > chunks) else "idle",
        queue_depth=0,
        active_op=f"indexing: {openie} openie passages processed" if (running and openie > chunks) else None,
    )
    stats = IndexStats(
        chunks=chunks,
        entities=entities,
        facts=facts,
        edges=edges,
        config_fingerprint=config.config_fingerprint(),
        openie_checkpoints=openie,
    )
    return {
        "status": status.to_dict(),
        "stats": stats.to_dict(),
    }


_TIER2_CORPORA = frozenset({"2wikimultihopqa", "musique", "hotpotqa"})
_KNOWN_DATASETS = frozenset({"sample", "2wikimultihopqa", "musique", "hotpotqa"})


def _extract_item_gold_answers(item: dict[str, Any]) -> list[str]:
    if "gold_answers" in item:
        raw = item["gold_answers"]
    elif "answer" in item or "gold_ans" in item:
        raw = item.get("answer", item.get("gold_ans"))
    elif "reference" in item:
        raw = item["reference"]
    else:
        raw = []
    ans = [raw] if isinstance(raw, str) else [str(a) for a in (raw or ())]
    ans.extend(str(a) for a in item.get("answer_aliases", ()))
    return list(dict.fromkeys(ans))


def _extract_item_docs_and_gold(
    item: dict[str, Any],
    dataset_name: str = "",
) -> tuple[list[str], list[str]]:
    if "gold_docs" in item and "paragraphs" not in item and "context" not in item and "contexts" not in item:
        g_docs = [str(d) for d in item.get("gold_docs", ())]
        return g_docs, g_docs

    separator = "" if dataset_name.startswith("hotpotqa") else " "
    all_docs: list[str] = []
    gold_docs: list[str] = []

    if "supporting_facts" in item and "context" in item:
        gold_titles = { sf[0] for sf in item["supporting_facts"] }
        for title, sents in item["context"]:
            passage = f"{title}\n{separator.join(sents)}"
            all_docs.append(passage)
            if title in gold_titles:
                gold_docs.append(passage)
    elif "contexts" in item:
        for ctx in item["contexts"]:
            passage = f"{ctx['title']}\n{ctx['text']}"
            all_docs.append(passage)
            if ctx.get("is_supporting", True):
                gold_docs.append(passage)
    elif "paragraphs" in item:
        for p in item["paragraphs"]:
            text = p["text"] if "text" in p else p.get("paragraph_text", "")
            passage = f"{p['title']}\n{text}"
            all_docs.append(passage)
            if p.get("is_supporting", True):
                gold_docs.append(passage)

    return list(dict.fromkeys(all_docs)), list(dict.fromkeys(gold_docs))


def _load_single_dataset_examples(
    dataset_name: str,
    *,
    workspace_root: Path | None = None,
    max_queries: int = 10,
    query_offset: int = 0,
) -> tuple[list[dspy.Example], list[str]]:
    ds_clean = dataset_name.strip()
    ds_lower = Path(ds_clean).stem.lower()
    if (ds_clean in _TIER2_CORPORA or ds_lower in _TIER2_CORPORA) and max_queries > 50:
        raise YawRuntimeError(
            "INVALID_INPUT",
            f"Anti-overfitting policy forbids prompt evolution on >50 queries of Tier 2 dataset '{ds_clean}' (got max_queries={max_queries}).",
        )

    candidate_path = Path(ds_clean)
    if not candidate_path.exists() and ds_clean in _KNOWN_DATASETS:
        roots = [Path.cwd()]
        if workspace_root is not None:
            roots.insert(0, workspace_root)
        for root in roots:
            cached = root / "outputs" / "datasets" / f"{ds_clean}.json"
            if cached.exists():
                candidate_path = cached
                break

    if not candidate_path.exists():
        raise YawRuntimeError("INVALID_INPUT", f"Training dataset not found: {ds_clean}")

    raw_items = json.loads(candidate_path.read_text(encoding="utf-8"))
    start = max(0, int(query_offset))
    sliced = list(raw_items)[start : start + max_queries]
    examples: list[dspy.Example] = []
    corpus_docs: list[str] = []
    for item in sliced:
        item_docs, gold_docs = _extract_item_docs_and_gold(item, ds_lower)
        corpus_docs.extend(item_docs)
        examples.append(
            dspy.Example(
                question=str(item["question"]),
                gold_answers=_extract_item_gold_answers(item),
                gold_docs=gold_docs,
            ).with_inputs("question")
        )
    return examples, list(dict.fromkeys(corpus_docs))


def _load_dataset_splits(
    dataset_args: Sequence[str] | str | None,
    *,
    workspace_root: Path | None = None,
    max_queries: int = 20,
    train_queries: int = 7,
    query_offset: int = 0,
) -> tuple[list[dspy.Example], list[dspy.Example] | None, list[str]]:
    if max_queries < 1:
        raise YawRuntimeError("INVALID_INPUT", "max_queries must be >= 1.")

    if not dataset_args:
        return (
            [
                dspy.Example(
                    question="When was the University of Southampton founded?",
                    gold_answers=["1862"],
                    gold_docs=[],
                ).with_inputs("question")
            ],
            None,
            [],
        )

    if isinstance(dataset_args, str):
        ds_list = [p.strip() for p in dataset_args.split(",") if p.strip()]
    else:
        ds_list = []
        for item in dataset_args:
            for sub in str(item).split(","):
                if sub.strip():
                    ds_list.append(sub.strip())

    train_per_ds: list[list[dspy.Example]] = []
    val_per_ds: list[list[dspy.Example]] = []
    all_corpus_docs: list[str] = []

    for ds_name in ds_list:
        d_examples, d_docs = _load_single_dataset_examples(
            ds_name,
            workspace_root=workspace_root,
            max_queries=max_queries,
            query_offset=query_offset,
        )
        all_corpus_docs.extend(d_docs)
        if 0 < train_queries < len(d_examples):
            train_per_ds.append(d_examples[:train_queries])
            val_per_ds.append(d_examples[train_queries:])
        else:
            train_per_ds.append(d_examples)
            val_per_ds.append([])

    # Interleave training examples across datasets for balanced mini-batches
    trainset: list[dspy.Example] = []
    max_t = max((len(t) for t in train_per_ds), default=0)
    for i in range(max_t):
        for t in train_per_ds:
            if i < len(t):
                trainset.append(t[i])

    # Interleave validation examples across datasets
    valset_list: list[dspy.Example] = []
    max_v = max((len(v) for v in val_per_ds), default=0)
    for i in range(max_v):
        for v in val_per_ds:
            if i < len(v):
                valset_list.append(v[i])

    valset = valset_list if valset_list else None
    return trainset, valset, list(dict.fromkeys(all_corpus_docs))


def _load_training_examples(
    dataset_arg: Sequence[str] | str | None,
    *,
    workspace_root: Path | None = None,
    max_queries: int = 10,
    query_offset: int = 0,
) -> tuple[list[dspy.Example], list[str]]:
    trainset, valset, corpus_docs = _load_dataset_splits(
        dataset_arg,
        workspace_root=workspace_root,
        max_queries=max_queries,
        train_queries=max_queries,
        query_offset=query_offset,
    )
    all_examples = list(trainset)
    if valset:
        all_examples.extend(valset)
    return all_examples, corpus_docs


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
    raw_dataset = getattr(args, "dataset", None)
    if raw_dataset is not None:
        if isinstance(raw_dataset, str):
            dataset_args = [p.strip() for p in raw_dataset.split(",") if p.strip()]
        elif isinstance(raw_dataset, (list, tuple)):
            dataset_args = []
            for item in raw_dataset:
                for sub in str(item).split(","):
                    if sub.strip():
                        dataset_args.append(sub.strip())
        else:
            dataset_args = [str(raw_dataset)]
    else:
        dataset_args = list(config.optimize_datasets)

    raw_max_q = getattr(args, "max_queries", None)
    max_queries = int(raw_max_q if raw_max_q is not None else config.optimize_max_queries)

    raw_offset = getattr(args, "query_offset", None)
    query_offset = int(raw_offset if raw_offset is not None else config.optimize_query_offset)

    raw_train_q = getattr(args, "train_queries", None)
    train_queries = int(
        raw_train_q if raw_train_q is not None else config.optimize_train_queries
    )

    raw_budget = getattr(args, "max_metric_calls", None)
    max_metric_calls = (
        int(raw_budget)
        if raw_budget is not None
        else config.optimize_max_metric_calls
    )

    trainset, valset, corpus_docs = _load_dataset_splits(
        dataset_args,
        workspace_root=config.workspace_root,
        max_queries=max_queries,
        train_queries=train_queries,
        query_offset=query_offset,
    )

    raw_mode = getattr(args, "mode", None)
    mode = raw_mode if raw_mode is not None else config.optimize_mode
    out_override = getattr(args, "output", None)

    runtime = YawRuntime(config, program=program, embedder=embedder, auto_poll=False)
    runtime.start(sync_on_start=True)
    try:
        saved_path = runtime.optimize(
            trainset=trainset,
            valset=valset,
            mode=mode,
            save_path=out_override,
            max_metric_calls=max_metric_calls,
            corpus_docs=corpus_docs,
        )
        proposer_telemetry = saved_path.parent / f"{saved_path.stem}_proposer_telemetry.json"
        gepa_telemetry_dir = saved_path.parent / f"{saved_path.stem}_telemetry"
        payload: dict[str, Any] = {
            "optimized": True,
            "mode": mode,
            "dataset": dataset_args[0] if len(dataset_args) == 1 else dataset_args,
            "datasets": dataset_args,
            "train_queries": len(trainset),
            "val_queries": len(valset) if valset is not None else 0,
            "compiled_program_path": str(saved_path),
            "proposer_telemetry_path": (
                str(proposer_telemetry) if proposer_telemetry.exists() else None
            ),
            "gepa_telemetry_dir": (
                str(gepa_telemetry_dir) if gepa_telemetry_dir.exists() else None
            ),
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


def cli_entrypoint() -> None:
    """Console script entrypoint for `yaw`."""
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "cli_entrypoint",
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
