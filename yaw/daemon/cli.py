"""Canonical CLI entrypoint for the YAW persistent HippoRAG 2 daemon.

Grammar:
  yaw daemon {start|stop|status|run} [--config <path>]
  yaw graph {build|rebuild|lint} [--config <path>] [--turtle]
  yaw dspy {optimize|load-program} [--flags]
  yaw config {get|set} [--flags]
"""

import argparse
import contextlib
import json
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
from typing import Any

import dspy
import filelock

from yaw.core import (
    DaemonStatus,
    IndexStats,
    YawConfig,
    YawRuntimeError,
    get_config_value,
    load_config,
    set_config_value,
)
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, HippoRAGProgram
from yaw.daemon.graph.storage import (
    get_graph_counts,
    lint_graph_shacl,
    open_yaw_db,
    serialize_graph_turtle,
)
from yaw.daemon.runtime import YawRuntime


def build_parser() -> argparse.ArgumentParser:
    """Build hierarchical argument parser for `yaw <domain> <action> --<flags>`."""
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

    domain_subparsers = parser.add_subparsers(dest="domain", help="Domain to operate on")

    # -------------------------------------------------------------------------
    # 1. yaw daemon {start|stop|status|run}
    # -------------------------------------------------------------------------
    p_daemon = domain_subparsers.add_parser("daemon", help="Daemon lifecycle commands")
    daemon_subs = p_daemon.add_subparsers(dest="action", help="Daemon action")

    daemon_subs.add_parser(
        "start", parents=[common], help="Start persistent daemon in background"
    )

    daemon_subs.add_parser("stop", parents=[common], help="Gracefully stop a running daemon")
    daemon_subs.add_parser(
        "status",
        parents=[common],
        help="Inspect daemon status, graph index statistics, and active optimizations as JSON",
    )
    daemon_subs.add_parser(
        "run",
        parents=[common],
        help="Run persistent daemon in the foreground until stopped",
    )

    # -------------------------------------------------------------------------
    # 2. yaw graph {build|rebuild|lint}
    # -------------------------------------------------------------------------
    p_graph = domain_subparsers.add_parser(
        "graph", help="Standalone graph and corpus index commands"
    )
    graph_subs = p_graph.add_subparsers(dest="action", help="Graph action")

    graph_subs.add_parser(
        "build",
        parents=[common],
        help="Run standalone incremental CocoIndex corpus sync and update yaw.db",
    )
    graph_subs.add_parser(
        "rebuild",
        parents=[common],
        help="Wipe yaw.db and CocoIndex state and perform a full rebuild from scratch",
    )
    p_lint = graph_subs.add_parser(
        "lint",
        parents=[common],
        help="Run closed-world SHACL graph lint or emit W3C Turtle (RDF/SKOS/OWL/SHACL)",
    )
    p_lint.add_argument(
        "--turtle",
        action="store_true",
        help="Emit W3C Turtle (RDF/SKOS/OWL/SHACL) serialization instead of JSON",
    )

    # -------------------------------------------------------------------------
    # 3. yaw dspy {optimize|load-program} + top-level aliases {optimize|load-program}
    # -------------------------------------------------------------------------
    p_opt_common = argparse.ArgumentParser(add_help=False)
    p_opt_common.add_argument(
        "--auto",
        dest="auto",
        choices=["light", "medium", "heavy"],
        default=None,
        help="GEPA/MIPROv2 evaluation budget mode (defaults to yaw.toml [optimization].auto, then 'light')",
    )
    p_opt_common.add_argument(
        "--mode",
        dest="mode",
        choices=["miprov2", "gepa", "hybrid"],
        default=None,
        help="Prompt optimization mode (defaults to yaw.toml [optimization].mode, then 'hybrid')",
    )
    p_opt_common.add_argument(
        "--dataset",
        dest="dataset",
        nargs="*",
        default=None,
        help=(
            "One or more dataset names (enterprise_rag_bench, sample, 2wikimultihopqa, musique, hotpotqa) "
            "or path to JSON training file (defaults to yaw.toml [optimization].datasets)"
        ),
    )
    p_opt_common.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries to load (defaults to yaw.toml [optimization].max_queries, max 50 on Tier 2 corpora)",
    )
    p_opt_common.add_argument(
        "--train-queries",
        type=int,
        default=None,
        help="Number of queries for training [0:K] (defaults to yaw.toml [optimization].train_queries)",
    )
    p_opt_common.add_argument(
        "--query-offset",
        type=int,
        default=None,
        help="Starting query offset in dataset (defaults to yaw.toml [optimization].query_offset)",
    )
    p_opt_common.add_argument(
        "--output",
        dest="output",
        type=str,
        default=None,
        help="Override output path for candidate compiled_program JSON",
    )
    p_opt_common.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Explicit number of optimization trials (sets auto to None for fast smoke tests)",
    )
    p_opt_common.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Explicit number of instruction/fewshot candidates per predictor",
    )
    p_opt_common.add_argument(
        "--sleep",
        dest="sleep_seconds",
        type=float,
        default=None,
        help="Sleep pause in seconds between LLM/embedding calls to manage thermal duty cycles (defaults to yaw.toml [optimization].sleep_seconds, default 0.0)",
    )
    p_opt_common.add_argument(
        "--db",
        dest="db_path",
        type=str,
        default=None,
        help="Path to existing sandbox database (or 'latest') to reuse (defaults to seeding from latest sandbox)",
    )

    p_dspy = domain_subparsers.add_parser(
        "dspy", help="DSPy prompt optimization and program management"
    )
    dspy_subs = p_dspy.add_subparsers(dest="action", help="DSPy action")

    dspy_subs.add_parser(
        "optimize",
        parents=[common, p_opt_common],
        help="Compile and evolve DSPy prompts in an isolated sandbox and save candidate JSON",
    )

    domain_subparsers.add_parser(
        "optimize",
        parents=[common, p_opt_common],
        help="Alias for `yaw dspy optimize`: compile and evolve DSPy prompts in an isolated sandbox",
    )

    p_load_common = argparse.ArgumentParser(add_help=False)
    p_load_common.add_argument(
        "file",
        type=str,
        help="Path to compiled DSPy program JSON file to promote and load",
    )

    dspy_subs.add_parser(
        "load-program",
        parents=[common, p_load_common],
        help="Promote a compiled DSPy program JSON into production and hot-reload in-memory",
    )

    domain_subparsers.add_parser(
        "load-program",
        parents=[common, p_load_common],
        help="Alias for `yaw dspy load-program`: promote a compiled DSPy program JSON into production",
    )

    # -------------------------------------------------------------------------
    # 4. yaw config {get|set}
    # -------------------------------------------------------------------------
    p_cfg = domain_subparsers.add_parser("config", help="Configuration management")
    cfg_subs = p_cfg.add_subparsers(dest="action", help="Config action")

    p_get = cfg_subs.add_parser("get", parents=[common], help="Read configuration value(s)")
    p_get.add_argument(
        "key",
        nargs="?",
        default=None,
        help="Configuration key to read (e.g. 'optimization.auto' or 'qa_model'). Omit to dump all.",
    )

    p_set = cfg_subs.add_parser("set", parents=[common], help="Update a configuration value")
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


def _inspect_active_optimization(config: YawConfig) -> dict[str, Any] | None:
    """Detect if any sandbox yaw_optimize_*.db.lock is currently locked and read its live progress."""
    if not config.lock_path:
        return None
    base_dir = config.lock_path.parent
    if not base_dir.exists():
        return None

    for lock_file in base_dir.glob("yaw_optimize_*.db.lock"):
        fl = filelock.FileLock(str(lock_file), timeout=0)
        is_held = False
        try:
            fl.acquire(timeout=0)
            fl.release()
        except filelock.Timeout:
            is_held = True

        if is_held:
            raw_name = lock_file.name
            # Format: yaw_optimize_<datestamp>.db.lock
            stem = raw_name.removesuffix(".lock").removesuffix(".db")
            ds = stem.removeprefix("yaw_optimize_")
            status_file = base_dir / f"status_optimize_{ds}.json"
            opt_db = base_dir / f"yaw_optimize_{ds}.db"

            openie = 0
            if opt_db.exists():
                with contextlib.suppress(sqlite3.Error):
                    db_uri = f"file:{opt_db.resolve().as_posix()}?mode=ro"
                    conn = sqlite3.connect(db_uri, uri=True, timeout=1.0)
                    try:
                        _, _, _, _, openie = get_graph_counts(conn)
                    finally:
                        conn.close()

            snap: dict[str, Any] = {}
            if status_file.exists():
                with contextlib.suppress(Exception):
                    snap = json.loads(status_file.read_text(encoding="utf-8"))

            return {
                "active": True,
                "datestamp": ds,
                "sandbox_db": str(opt_db),
                "openie_checkpoints": max(
                    openie, snap.get("stats", {}).get("openie_checkpoints", 0)
                ),
                "active_op": snap.get("status", {}).get("active_op", "optimizing"),
                "status": snap.get("status", {}),
            }
    return None


def inspect_status(config: YawConfig) -> dict[str, Any]:
    """Read live daemon status if running, offline yaw.db stats if stopped, and active optimization progress."""
    running = is_daemon_locked(config)
    chunks = entities = facts = edges = openie = entity_labels = 0
    shacl_conforms = True
    if config.db_path and Path(config.db_path).exists():
        with contextlib.suppress(sqlite3.Error):
            db_uri = f"file:{Path(config.db_path).resolve().as_posix()}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True, timeout=1.0)
            try:
                chunks, entities, facts, edges, openie = get_graph_counts(conn)
                with contextlib.suppress(sqlite3.Error):
                    row = conn.execute("SELECT COUNT(*) FROM entity_labels").fetchone()
                    entity_labels = int(row[0]) if row else 0
                    shacl_conforms = len(lint_graph_shacl(conn, auto_heal=False)) == 0
            finally:
                conn.close()

    opt_info = _inspect_active_optimization(config)

    if running and config.status_path and Path(config.status_path).exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            data = json.loads(Path(config.status_path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "status" in data and "stats" in data:
                data["stats"]["openie_checkpoints"] = max(
                    openie, data["stats"].get("openie_checkpoints", 0)
                )
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


_TIER2_CORPORA = frozenset({"2wikimultihopqa", "musique", "hotpotqa"})
_KNOWN_DATASETS = frozenset(
    {
        "sample",
        "2wikimultihopqa",
        "musique",
        "hotpotqa",
        "enterprise_rag_bench",
        "enterprise",
    }
)
_DATASET_ALIASES = {"enterprise": "enterprise_rag_bench"}


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
    ans.extend(str(a) for a in item.get("answer_facts", ()))
    return list(dict.fromkeys(ans))


def _extract_item_docs_and_gold(
    item: dict[str, Any],
    dataset_name: str = "",
) -> tuple[list[str], list[str]]:
    if (
        "gold_docs" in item
        and "paragraphs" not in item
        and "context" not in item
        and "contexts" not in item
    ):
        g_docs = [str(d) for d in item.get("gold_docs", ())]
        return g_docs, g_docs

    separator = "" if dataset_name.startswith("hotpotqa") else " "
    all_docs: list[str] = []
    gold_docs: list[str] = []

    if "supporting_facts" in item and "context" in item:
        gold_titles = {sf[0] for sf in item["supporting_facts"]}
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
        resolved_ds = _DATASET_ALIASES.get(ds_clean, ds_clean)
        roots = [Path.cwd()]
        if workspace_root is not None:
            roots.insert(0, workspace_root)
        for root in roots:
            cached = root / "outputs" / "datasets" / f"{resolved_ds}.json"
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
                item_docs=item_docs,
            ).with_inputs("question", "item_docs")
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
            val_per_ds.append(list(d_examples))

    trainset: list[dspy.Example] = []
    max_t = max((len(t) for t in train_per_ds), default=0)
    for i in range(max_t):
        for t in train_per_ds:
            if i < len(t):
                trainset.append(t[i])

    valset_list: list[dspy.Example] = []
    max_v = max((len(v) for v in val_per_ds), default=0)
    for i in range(max_v):
        for v in val_per_ds:
            if i < len(v):
                valset_list.append(v[i])

    valset = valset_list if valset_list else None
    return trainset, valset, list(dict.fromkeys(all_corpus_docs))


def handle_daemon_run(
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


def handle_daemon_stop(config: YawConfig) -> int:
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


def handle_graph_build(
    config: YawConfig,
    *,
    program: HippoRAGProgram | None = None,
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


def handle_graph_lint(
    config: YawConfig,
    *,
    as_turtle: bool = False,
) -> int:
    """Validate closed-world SHACL graph shapes or emit W3C Turtle over a read-only WAL snapshot."""
    db_file = Path(config.db_path).resolve() if config.db_path else None
    if db_file is not None and db_file.exists() and db_file.stat().st_size > 0:
        db_uri = f"file:{db_file.as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
    else:
        conn = open_yaw_db(":memory:")

    try:
        try:
            violations = lint_graph_shacl(conn, auto_heal=False)
        except sqlite3.OperationalError:
            conn.close()
            conn = open_yaw_db(":memory:")
            violations = lint_graph_shacl(conn, auto_heal=False)

        if as_turtle:
            ttl = serialize_graph_turtle(conn, violations=violations)
            if not ttl.endswith("\n"):
                ttl += "\n"
            sys.stdout.write(ttl)
        else:
            payload = {
                "conforms": len(violations) == 0,
                "violation_count": len(violations),
                "violations": [v.to_dict() for v in violations],
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
    finally:
        conn.close()


def handle_dspy_optimize(
    config: YawConfig,
    args: argparse.Namespace,
    *,
    program: HippoRAGProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Compile DSPy prompts in an isolated sandbox and save candidate JSON."""
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
    elif config.optimize_datasets:
        dataset_args = list(config.optimize_datasets)
    else:
        dataset_args = ["2wikimultihopqa"]

    raw_max_q = getattr(args, "max_queries", None)
    max_queries = int(raw_max_q if raw_max_q is not None else (config.optimize_max_queries or 20))

    raw_offset = getattr(args, "query_offset", None)
    query_offset = int(
        raw_offset if raw_offset is not None else (config.optimize_query_offset or 0)
    )

    raw_train_q = getattr(args, "train_queries", None)
    train_queries = int(
        raw_train_q if raw_train_q is not None else (config.optimize_train_queries or 7)
    )

    raw_auto = getattr(args, "auto", None)
    auto = raw_auto or config.optimize_auto or "light"

    raw_trials = getattr(args, "num_trials", None)
    num_trials = int(raw_trials) if raw_trials is not None else None

    raw_candidates = getattr(args, "num_candidates", None)
    num_candidates = int(raw_candidates) if raw_candidates is not None else None

    raw_mode = getattr(args, "mode", None)
    mode = raw_mode or config.optimize_mode or "hybrid"

    out_override = getattr(args, "output", None)

    trainset, valset, corpus_docs = _load_dataset_splits(
        dataset_args,
        workspace_root=config.workspace_root,
        max_queries=max_queries,
        train_queries=train_queries,
        query_offset=query_offset,
    )

    # Resolution order for sandbox: always isolate prompt compilation
    raw_db = getattr(args, "db_path", None)
    opt_config = config.for_optimization(db_override=raw_db)
    raw_sleep = getattr(args, "sleep_seconds", None)
    sleep_seconds = float(raw_sleep) if raw_sleep is not None else float(config.optimize_sleep_seconds or 0.0)
    object.__setattr__(opt_config, "optimize_sleep_seconds", sleep_seconds)

    runtime = YawRuntime(opt_config, program=program, embedder=embedder, auto_poll=False)
    runtime.start(sync_on_start=False)

    def _cli_progress(msg: str) -> None:
        print(f"\r[yaw optimize] {msg}\x1b[K", end="", file=sys.stderr, flush=True)

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
            corpus_docs=corpus_docs,
            progress_callback=_cli_progress,
        )
        print(file=sys.stderr)  # Move to next line before JSON output
        proposer_telemetry = saved_path.parent / f"{saved_path.stem}_proposer_telemetry.json"
        graph_rlm_audit = saved_path.parent / f"{saved_path.stem}_graph_rlm_audit.json"
        gepa_telemetry_dir = saved_path.parent / f"{saved_path.stem}_telemetry"
        val_eval_file = saved_path.parent / f"{saved_path.stem}_eval.json"
        eval_summary: dict[str, Any] | None = None
        if val_eval_file.exists():
            with contextlib.suppress(Exception):
                raw_eval = json.loads(val_eval_file.read_text(encoding="utf-8"))
                if isinstance(raw_eval, list) and raw_eval:
                    f1_scores = [
                        float(it.get("f1", it.get("score", 0.0)))
                        for it in raw_eval
                        if isinstance(it, dict)
                    ]
                    em_scores = [
                        float(it.get("em", 0.0))
                        for it in raw_eval
                        if isinstance(it, dict)
                    ]
                    apm_scores = [
                        float(it.get("answer_passage_match", 0.0))
                        for it in raw_eval
                        if isinstance(it, dict)
                    ]
                    eval_summary = {
                        "num_evaluated": len(raw_eval),
                        "mean_f1": (
                            sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                        ),
                        "mean_em": (
                            sum(em_scores) / len(em_scores) if em_scores else 0.0
                        ),
                        "answer_passage_match_rate": (
                            sum(apm_scores) / len(apm_scores) if apm_scores else 0.0
                        ),
                    }
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
            "proposer_telemetry_path": (
                str(proposer_telemetry) if proposer_telemetry.exists() else None
            ),
            "graph_rlm_audit_path": (
                str(graph_rlm_audit) if graph_rlm_audit.exists() else None
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
        raise YawRuntimeError(
            "INVALID_INPUT", f"Invalid compiled program JSON in {src}: {exc}"
        ) from exc

    target_path = (
        Path(config.compiled_program_path).resolve()
        if config.compiled_program_path
        else (config.workspace_root / ".yaw" / "compiled_program.json").resolve()
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if src != target_path:
        tmp_target = target_path.with_suffix(f".tmp.{os.getpid()}")
        shutil.copy2(src, tmp_target)
        tmp_target.replace(target_path)

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
        signal_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if result_file.exists():
                try:
                    res = json.loads(result_file.read_text(encoding="utf-8"))
                    if res.get("req_id") == req_id:
                        sys.stdout.write(json.dumps(res, indent=2) + "\n")
                        return 0 if res.get("reloaded", True) else 1
                except Exception:
                    pass
            time.sleep(0.05)

        sys.stdout.write(
            json.dumps(
                {
                    "reloaded": False,
                    "staged": True,
                    "program_path": str(target_path),
                    "source_path": str(src),
                    "message": "Signal written; daemon did not acknowledge before timeout. Program staged for next startup.",
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
                "message": "Daemon is not running. Program promoted to compiled_program_path and will load on next daemon start.",
            },
            indent=2,
        )
        + "\n"
    )
    return 0


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
    program: HippoRAGProgram | None = None,
    embedder: DSPyEmbeddingAdapter | None = None,
) -> int:
    """Main CLI dispatcher returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    domain = getattr(args, "domain", None)
    action = getattr(args, "action", None)
    if domain == "optimize":
        domain = "dspy"
        action = "optimize"
    elif domain == "load-program":
        domain = "dspy"
        action = "load-program"

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
                return handle_graph_lint(config, as_turtle=bool(getattr(args, "turtle", False)))

        # 3. yaw dspy {optimize|load-program}
        if domain == "dspy":
            if action == "optimize":
                return handle_dspy_optimize(config, args, program=program, embedder=embedder)
            if action == "load-program":
                return handle_dspy_load_program(config, args.file)

        # 4. yaw config {get|set}
        if domain == "config":
            if action == "get":
                return handle_config_get(config, key=args.key, config_path=cfg_path)
            if action == "set":
                return handle_config_set(
                    config, key=args.key, value_raw=args.value, config_path=cfg_path
                )

        parser.print_help(sys.stderr)
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
    "handle_config_get",
    "handle_config_set",
    "handle_daemon_run",
    "handle_daemon_start",
    "handle_daemon_stop",
    "handle_dspy_load_program",
    "handle_dspy_optimize",
    "handle_graph_build",
    "handle_graph_lint",
    "handle_graph_rebuild",
    "inspect_status",
    "is_daemon_locked",
    "main",
]
