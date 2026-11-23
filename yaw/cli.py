"""Unified clean-slate YAW CLI for daemon lifecycle, recall, QA, optimization, and config."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import Any

import filelock

from yaw.core import (
    DaemonStatus,
    YawConfig,
    YawRuntimeError,
    get_config_value,
    load_config,
    redact_secrets,
    set_config_value,
    write_text_atomic,
)
from yaw.mcp_server import serve_http
from yaw.runtime import (
    ask_once,
    read_db_chunk_count,
    recall_once,
    require_path,
    serialize_passage,
    signal_stop_event,
)

_kernel32: Any = None
_console_handlers: list[Any] = []
if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    _kernel32.GetExitCodeProcess.restype = ctypes.c_int
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int


def _is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt" and _kernel32 is not None:
        handle = _kernel32.OpenProcess(0x1000, 0, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_uint32()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == 259
        finally:
            _kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_text_retry(path: Path, *, retries: int = 5) -> str:
    for attempt in range(max(1, retries)):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt == max(1, retries) - 1:
                raise
            time.sleep(0.01 * (2**attempt))
    return ""


def _read_pid_file(pid_path: Path) -> int | None:
    if not pid_path.is_file():
        return None
    raw = _read_text_retry(pid_path).strip()
    if not raw:
        raise ValueError(f"Empty PID file: {pid_path}")
    pid = int(raw)
    if pid <= 0:
        raise ValueError(f"Invalid PID in {pid_path}: {pid}")
    return pid


def _read_status_file(status_path: Path) -> dict[str, Any]:
    if not status_path.is_file():
        return {}
    data = json.loads(_read_text_retry(status_path))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid status payload in {status_path}")
    return data


def _resolve_pid(pid_path: Path, status_data: dict[str, Any]) -> int | None:
    if (pid := _read_pid_file(pid_path)) is not None:
        return pid
    raw_pid = status_data.get("pid")
    return raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0 else None


def _is_lock_held(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    probe = filelock.FileLock(str(lock_path), timeout=0)
    try:
        probe.acquire(timeout=0)
    except filelock.Timeout:
        return True
    probe.release()
    return False


def _terminate_child(proc: subprocess.Popen[Any], lock_path: Path) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        signal_stop_event(lock_path)
    else:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)


def _format_passages_text(items: Sequence[dict[str, Any]]) -> str:
    if not items:
        return "No matching passages."
    blocks = []
    for item in items:
        src = item["source"]
        body = "[STALE ON DISK]" if item["stale_on_disk"] else item["text"]
        head = (
            f"[{item['rank']}] chunk_id={item['chunk_id']} "
            f"({src['rel_posix_path']}:{src['start_line']}-{src['end_line']}) score={item['score']:.3f}"
        )
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def _daemon_status(cfg: YawConfig) -> dict[str, Any]:
    state_dir = require_path(cfg.state_dir, "state_dir")
    status_path = require_path(cfg.status_path, "status_path")
    db_path = require_path(cfg.db_path, "db_path")
    status_data = _read_status_file(status_path)
    pid = _resolve_pid(state_dir / "daemon.pid", status_data)
    num_chunks = read_db_chunk_count(db_path)
    raw_err = status_data.get("last_error")
    err = redact_secrets(str(raw_err), cfg.api_key) if raw_err else None
    raw_sync = status_data.get("last_sync_at")
    last_sync_at = float(raw_sync) if isinstance(raw_sync, (int, float)) and not isinstance(raw_sync, bool) else None
    if not _is_pid_alive(pid):
        return DaemonStatus(
            lifecycle="stopped",
            provider_ready=False,
            sync_state="idle",
            queue_depth=0,
            active_op=None,
            last_error=err,
            last_sync_at=last_sync_at,
            pid=None,
            num_chunks=num_chunks,
        ).to_dict()
    if not status_data:
        return DaemonStatus(
            lifecycle="starting",
            provider_ready=False,
            sync_state="syncing",
            pid=pid,
            num_chunks=num_chunks,
        ).to_dict()
    return DaemonStatus(
        lifecycle=str(status_data.get("lifecycle", "starting")),
        provider_ready=bool(status_data.get("provider_ready", False)),
        sync_state=str(status_data.get("sync_state", "syncing")),
        queue_depth=int(status_data.get("queue_depth", 0)),
        active_op=status_data.get("active_op"),
        last_error=err,
        last_sync_at=last_sync_at,
        pid=pid,
        num_chunks=num_chunks,
    ).to_dict()


def _daemon_run(cfg: YawConfig) -> dict[str, Any]:
    state_dir = require_path(cfg.state_dir, "state_dir")
    lock_path = require_path(cfg.lock_path, "lock_path")
    if _is_lock_held(lock_path):
        raise YawRuntimeError("BUSY", "Daemon is already running.")
    pid_path = state_dir / "daemon.pid"
    my_pid = str(os.getpid())
    write_text_atomic(pid_path, f"{my_pid}\n")
    try:
        serve_http(cfg)
    finally:
        if pid_path.is_file():
            with contextlib.suppress(OSError):
                if pid_path.read_text(encoding="utf-8").strip() == my_pid:
                    pid_path.unlink(missing_ok=True)
    return {"lifecycle": "stopped"}


def _daemon_start(cfg: YawConfig, config_path: Path | None) -> dict[str, Any]:
    state_dir = require_path(cfg.state_dir, "state_dir")
    lock_path = require_path(cfg.lock_path, "lock_path")
    status_path = require_path(cfg.status_path, "status_path")
    pid_path = state_dir / "daemon.pid"
    if _is_pid_alive(_resolve_pid(pid_path, _read_status_file(status_path))):
        raise YawRuntimeError("BUSY", "Daemon is already running.")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    probe = filelock.FileLock(str(lock_path), timeout=0)
    try:
        probe.acquire(timeout=0)
    except filelock.Timeout as exc:
        raise YawRuntimeError("BUSY", "Daemon is already running.") from exc
    finally:
        if probe.is_locked:
            probe.release()

    cmd = [sys.executable, "-m", "yaw", "daemon", "run"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path.resolve())])
    pid_path.unlink(missing_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = state_dir / "daemon.stderr.log"
    popen_kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    else:
        popen_kwargs["start_new_session"] = True
    with stderr_log.open("wb") as stderr_fh:
        proc = subprocess.Popen(cmd, stderr=stderr_fh, **popen_kwargs)

    raw_host = cfg.http_host[1:-1] if cfg.http_host.startswith("[") and cfg.http_host.endswith("]") else cfg.http_host
    url_host = f"[{raw_host}]" if ":" in raw_host else raw_host
    readyz_url = f"http://{url_host}:{cfg.http_port}/readyz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 30.0
    try:
        while time.monotonic() < deadline:
            if (rc := proc.poll()) is not None:
                err_tail = (
                    stderr_log.read_text(encoding="utf-8", errors="replace").strip() if stderr_log.is_file() else ""
                )
                raise YawRuntimeError("NOT_READY", f"Daemon exited prematurely (code={rc}): {err_tail or 'no stderr'}")
            if pid_path.is_file() and status_path.is_file():
                child_pid = _read_pid_file(pid_path)
                st = _read_status_file(status_path)
                if child_pid is not None and st.get("pid") == child_pid and _is_pid_alive(child_pid):
                    if st.get("lifecycle") == "error":
                        msg = redact_secrets(str(st.get("last_error") or "Daemon startup failed."), cfg.api_key)
                        raise YawRuntimeError("NOT_READY", msg)
                    if st.get("lifecycle") == "ready":
                        try:
                            with opener.open(readyz_url, timeout=1.0) as resp:
                                if int(resp.status) == 200:
                                    return _daemon_status(cfg)
                        except urllib.error.URLError, OSError, TimeoutError:
                            pass
            time.sleep(0.05)
        raise YawRuntimeError("NOT_READY", "Timed out waiting for daemon readiness.")
    except BaseException:
        _terminate_child(proc, lock_path)
        raise


def _daemon_stop(cfg: YawConfig) -> dict[str, Any]:
    state_dir = require_path(cfg.state_dir, "state_dir")
    lock_path = require_path(cfg.lock_path, "lock_path")
    status_path = require_path(cfg.status_path, "status_path")
    pid_path = state_dir / "daemon.pid"
    pid = _resolve_pid(pid_path, _read_status_file(status_path))
    pid_alive = _is_pid_alive(pid)
    if not pid_alive and not _is_lock_held(lock_path):
        raise YawRuntimeError("NOT_READY", "Daemon is not running.")

    signaled = False
    if os.name == "nt":
        signaled = signal_stop_event(lock_path)
    elif pid is not None and pid_alive:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
            signaled = True

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        dead = (pid is None) or (not _is_pid_alive(pid))
        if dead and not _is_lock_held(lock_path):
            pid_path.unlink(missing_ok=True)
            return {"lifecycle": "stopped"}
        if os.name == "nt" and not signaled:
            signaled = signal_stop_event(lock_path)
        time.sleep(0.05)
    raise YawRuntimeError("BUSY", "Timed out waiting for daemon shutdown.")


def _cmd_recall(cfg: YawConfig, query: str, top_k: int) -> dict[str, Any]:
    passages = recall_once(cfg, query, top_k=top_k)
    return {
        "query": query.strip(),
        "top_k": top_k,
        "results": [serialize_passage(i, hp) for i, hp in enumerate(passages, 1)],
    }


def _cmd_qa(cfg: YawConfig, question: str) -> dict[str, Any]:
    res = ask_once(cfg, question)
    return {
        "question": res.question,
        "answer": res.answer,
        "turns": int(res.turns),
        "citations": [serialize_passage(i, hp) for i, hp in enumerate(res.passages, start=1)],
    }


def _redact_config_value(key: str | None, val: Any) -> Any:
    if isinstance(val, dict):
        return {k: ("[REDACTED]" if k == "api_key" and v else v) for k, v in val.items()}
    if key is not None and key.strip().split(".")[-1] == "api_key" and val:
        return "[REDACTED]"
    return val


def _cmd_config(args: argparse.Namespace, config_path: Path | None) -> tuple[Any, str]:
    if args.config_cmd == "show":
        data = _redact_config_value(None, get_config_value(config_path))
        return data, "\n".join(f"{k} = {v}" for k, v in data.items())
    if args.config_cmd == "get":
        val = _redact_config_value(args.key, get_config_value(config_path, args.key))
        if args.key is None:
            return val, "\n".join(f"{k} = {v}" for k, v in val.items())
        return {"key": args.key, "value": val}, (val if isinstance(val, str) else json.dumps(val))
    resolved_key, parsed_val = set_config_value(args.key, args.value, config_path)
    display_val = _redact_config_value(resolved_key, parsed_val)
    return {"key": resolved_key, "value": display_val}, f"{resolved_key} = {display_val}"


def _emit(payload: Any, *, json_out: bool, text: str) -> None:
    print(json.dumps(payload, indent=2) if json_out else text)


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, metavar="PATH", help="Path to yaw.toml")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Emit JSON output")

    parser = argparse.ArgumentParser(prog="yaw", parents=[common], description="YAW GraphRAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_daemon = sub.add_parser("daemon", parents=[common], help="Manage the YAW HTTP daemon")
    sub_daemon = p_daemon.add_subparsers(dest="daemon_cmd", required=True)
    for name in ("run", "start", "stop", "status"):
        sub_daemon.add_parser(name, parents=[common], help=f"{name.capitalize()} daemon")

    p_recall = sub.add_parser("recall", parents=[common], help="Run read-only zero-LLM passage recall")
    p_recall.add_argument("query", help="Search query")
    p_recall.add_argument("--top-k", type=int, default=5, help="Maximum passages to return (1-20)")

    p_qa = sub.add_parser("qa", parents=[common], help="Run ReAct multi-hop question answering")
    p_qa.add_argument("question", help="Question to answer")

    p_opt = sub.add_parser("optimize", parents=[common], help="Optimize QA prompts with GEPA")
    sub_opt = p_opt.add_subparsers(dest="optimize_cmd", required=True)
    p_dspy = sub_opt.add_parser("dspy", parents=[common], help="Compile bare ReAct QA instructions with GEPA")
    p_dspy.add_argument("--dataset", nargs="+", default=None, dest="datasets")
    p_dspy.add_argument("--max-queries", type=int, default=None)
    p_dspy.add_argument("--query-offset", type=int, default=None)
    p_dspy.add_argument("--auto", choices=("light", "medium", "heavy"), default=None)
    p_dspy.add_argument("--max-metric-calls", type=int, default=None)
    p_dspy.add_argument("--threads", type=int, default=None)
    p_dspy.add_argument("--fresh", action="store_true", default=False)

    p_cfg = sub.add_parser("config", parents=[common], help="Inspect or update yaw.toml settings")
    sub_cfg = p_cfg.add_subparsers(dest="config_cmd", required=True)
    sub_cfg.add_parser("show", parents=[common], help="Show resolved configuration")
    p_get = sub_cfg.add_parser("get", parents=[common], help="Read a configuration key")
    p_get.add_argument("key", nargs="?", default=None)
    p_set = sub_cfg.add_parser("set", parents=[common], help="Set a configuration key")
    p_set.add_argument("key")
    p_set.add_argument("value")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, dispatch commands, and map errors to structured exit codes."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2

    raw_cfg = getattr(args, "config", None)
    config_path = Path(raw_cfg) if raw_cfg is not None else None
    json_out = bool(getattr(args, "json", False))
    api_key = ""
    pending_secret = ""
    if (
        args.command == "config"
        and getattr(args, "config_cmd", None) == "set"
        and str(getattr(args, "key", "")).strip().split(".")[-1] == "api_key"
    ):
        pending_secret = str(getattr(args, "value", ""))

    try:
        with contextlib.redirect_stdout(sys.stderr) if json_out else contextlib.nullcontext():
            if args.command == "config":
                payload, text = _cmd_config(args, config_path)
            else:
                cfg = load_config(config_path)
                api_key = cfg.api_key
                if args.command == "daemon":
                    if args.daemon_cmd == "run":
                        payload = _daemon_run(cfg)
                    elif args.daemon_cmd == "start":
                        payload = _daemon_start(cfg, config_path)
                    elif args.daemon_cmd == "stop":
                        payload = _daemon_stop(cfg)
                    else:
                        payload = _daemon_status(cfg)
                    text = json.dumps(payload, indent=2)
                elif args.command == "recall":
                    payload = _cmd_recall(cfg, args.query, args.top_k)
                    text = _format_passages_text(payload["results"])
                elif args.command == "qa":
                    payload = _cmd_qa(cfg, args.question)
                    cites = _format_passages_text(payload["citations"]) if payload["citations"] else ""
                    text = f"{payload['answer']}\n\n{cites}" if cites else payload["answer"]
                else:
                    from yaw.optimize import optimize_dspy

                    payload = optimize_dspy(
                        cfg,
                        datasets=",".join(args.datasets) if args.datasets else None,
                        max_queries=args.max_queries,
                        query_offset=args.query_offset,
                        auto=args.auto,
                        max_metric_calls=args.max_metric_calls,
                        threads=args.threads,
                        fresh=args.fresh,
                    )
                    text = json.dumps(payload, indent=2)
        _emit(payload, json_out=json_out, text=text)
        return 0
    except (YawRuntimeError, ValueError, KeyError, FileNotFoundError, OSError) as exc:
        if args.command == "config" and not api_key:
            try:
                api_key = load_config(config_path).api_key
            except YawRuntimeError, ValueError, FileNotFoundError, OSError:
                api_key = ""
        msg = redact_secrets(str(exc), api_key)
        if pending_secret:
            msg = redact_secrets(msg, pending_secret)
            if (escaped := repr(pending_secret)[1:-1]) != pending_secret:
                msg = redact_secrets(msg, escaped)
        sys.stderr.write((json.dumps({"error": msg}) if json_out else msg) + "\n")
        return 1


def cli_entrypoint() -> None:
    """Run the CLI with graceful interruption and a repeated-signal escape hatch."""
    interrupted_signal: int | None = None
    console_signal: int | None = None

    def interrupt(signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted_signal
        if interrupted_signal is not None:
            os._exit(128 + signum)
        interrupted_signal = signum
        if console_signal is None:
            os.write(2, b"Interrupt received; signal again to force exit.\n")
        raise KeyboardInterrupt

    signals = (signal.SIGINT, signal.SIGTERM)
    if (sigbreak := getattr(signal, "SIGBREAK", None)) is not None:
        signals += (sigbreak,)
    previous = {sig: signal.signal(sig, interrupt) for sig in signals}
    if os.name == "nt":
        handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32)

        def console_interrupt(event: int) -> bool:
            nonlocal console_signal
            if event not in (0, 1):
                return False
            signum = signal.SIGINT if event == 0 else signal.SIGBREAK
            if console_signal is not None or interrupted_signal is not None:
                os._exit(128 + signum)
            console_signal = signum
            os.write(2, b"Interrupt received; signal again to force exit.\n")
            # Let CPython receive the first console event and unwind when its wait returns.
            return False

        handler = handler_type(console_interrupt)
        _kernel32.SetConsoleCtrlHandler.argtypes = [handler_type, ctypes.c_int]
        _kernel32.SetConsoleCtrlHandler.restype = ctypes.c_int
        if not _kernel32.SetConsoleCtrlHandler(handler, 1):
            raise ctypes.WinError()
        _console_handlers.append(handler)
    try:
        code = main()
    except KeyboardInterrupt:
        code = 128 + (interrupted_signal or signal.SIGINT)
        sys.stderr.write("Interrupted.\n")
    finally:
        # Keep the escape hatch installed while Python waits for worker threads on exit.
        if interrupted_signal is None and console_signal is None:
            while _console_handlers:
                active_handler = _console_handlers.pop()
                if not _kernel32.SetConsoleCtrlHandler(active_handler, 0):
                    raise ctypes.WinError()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    raise SystemExit(code)
