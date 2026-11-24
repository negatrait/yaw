"""Unified clean-slate YAW CLI for daemon lifecycle, recall, QA, optimization, and config."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import http.client
import json
import os
import select
import selectors
import signal
import subprocess
import sys
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
)
from yaw.mcp_server import serve_http
from yaw.runtime import (
    ask_once,
    build_recall_payload,
    control_token_path,
    dump_recall_json,
    format_recall_text,
    read_db_chunk_count,
    recall_once_payload,
    require_path,
)

_kernel32: Any = None
_console_handlers: list[Any] = []
if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int


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


def _http_host_for_connect(cfg: YawConfig) -> str:
    raw = cfg.http_host.strip()
    return raw[1:-1] if raw.startswith("[") and raw.endswith("]") else raw


def _http_host_header(cfg: YawConfig) -> str:
    host = _http_host_for_connect(cfg)
    return f"[{host}]:{cfg.http_port}" if ":" in host else f"{host}:{cfg.http_port}"


def _http_json(
    cfg: YawConfig,
    method: str,
    path: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    req_headers = {"Host": _http_host_header(cfg), "Connection": "close"}
    if headers is not None:
        req_headers.update(headers)
    conn = http.client.HTTPConnection(_http_host_for_connect(cfg), cfg.http_port, timeout=timeout)
    try:
        conn.request(method, path, headers=req_headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        data = json.loads(raw) if raw.strip() else {}
        return int(resp.status), data if isinstance(data, dict) else {}
    finally:
        conn.close()


def _open_kqueue_wait_handle(pid: int) -> tuple[str, Any] | None:
    kqueue_fn = getattr(select, "kqueue", None)
    kevent_fn = getattr(select, "kevent", None)
    kq_filter = getattr(select, "KQ_FILTER_PROC", None)
    kq_note_exit = getattr(select, "KQ_NOTE_EXIT", None)
    if kqueue_fn is None or kevent_fn is None or kq_filter is None or kq_note_exit is None:
        return None
    kq = kqueue_fn()
    try:
        flags = int(getattr(select, "KQ_ADD", 0)) | int(getattr(select, "KQ_ONESHOT", 0))
        ev = kevent_fn(int(pid), filter=kq_filter, flags=flags, fflags=kq_note_exit)
        kq.control([ev], 0, 0)
        return ("kqueue", kq)
    except ProcessLookupError:
        kq.close()
        return ("exited", 0)
    except OSError:
        kq.close()
        return None


def _open_process_wait_handle(pid: int | None) -> tuple[str, Any] | None:
    if pid is None or pid <= 0:
        return None
    if os.name == "nt" and _kernel32 is not None:
        handle = _kernel32.OpenProcess(0x00100000, 0, int(pid))
        if not handle:
            if ctypes.get_last_error() == 87:
                return ("exited", 0)
            return None
        return ("win32", int(handle))
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is not None:
        try:
            return ("pidfd", int(pidfd_open(int(pid), 0)))
        except ProcessLookupError:
            return ("exited", 0)
        except OSError:
            pass
    if (kq_handle := _open_kqueue_wait_handle(int(pid))) is not None:
        return kq_handle
    return ("pid", int(pid))


def _close_process_wait_handle(handle_info: tuple[str, Any] | None) -> None:
    if handle_info is None:
        return
    kind, val = handle_info
    if kind == "win32" and _kernel32 is not None:
        with contextlib.suppress(OSError):
            _kernel32.CloseHandle(val)
    elif kind == "pidfd":
        with contextlib.suppress(OSError):
            os.close(val)
    elif kind == "kqueue":
        with contextlib.suppress(OSError):
            val.close()


def _probe_pid_exited(pid: int) -> bool:
    if os.name == "nt":
        if _kernel32 is None:
            return False
        handle = _kernel32.OpenProcess(0x00100000, 0, int(pid))
        if not handle:
            return ctypes.get_last_error() == 87
        try:
            return int(_kernel32.WaitForSingleObject(handle, 0)) == 0
        finally:
            with contextlib.suppress(OSError):
                _kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_process_handle(handle_info: tuple[str, Any] | None, *, timeout: float) -> bool:
    if handle_info is None:
        return True
    kind, val = handle_info
    if kind == "exited":
        return True
    if kind == "win32" and _kernel32 is not None:
        try:
            res = int(_kernel32.WaitForSingleObject(val, int(max(0.0, timeout) * 1000)))
            return res == 0
        finally:
            with contextlib.suppress(OSError):
                _kernel32.CloseHandle(val)
    if kind == "pidfd":
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(val, selectors.EVENT_READ)
                return bool(selector.select(timeout=max(0.0, timeout)))
        finally:
            with contextlib.suppress(OSError):
                os.close(val)
    if kind == "kqueue":
        try:
            return bool(val.control(None, 1, max(0.0, timeout)))
        finally:
            with contextlib.suppress(OSError):
                val.close()
    return _probe_pid_exited(int(val))


def _wait_lock_release(lock_path: Path, *, timeout: float) -> None:
    if not _is_lock_held(lock_path):
        return
    probe = filelock.FileLock(str(lock_path), timeout=timeout)
    try:
        probe.acquire(timeout=timeout)
    except filelock.Timeout as exc:
        raise YawRuntimeError("BUSY", "Timed out waiting for daemon lock release.") from exc
    probe.release()


def _read_status_file(status_path: Path) -> dict[str, Any]:
    if not status_path.is_file():
        return {}
    data = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid status payload in {status_path}")
    return data


def _redact_status_payload(data: dict[str, Any], api_key: str) -> dict[str, Any]:
    raw_err = data.get("last_error")
    err = redact_secrets(str(raw_err), api_key) if raw_err else None
    raw_sync = data.get("last_sync_at")
    last_sync_at = float(raw_sync) if isinstance(raw_sync, (int, float)) and not isinstance(raw_sync, bool) else None
    raw_pid = data.get("pid")
    pid = int(raw_pid) if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0 else None
    return DaemonStatus(
        lifecycle=str(data.get("lifecycle", "starting")),
        provider_ready=bool(data.get("provider_ready", False)),
        sync_state=str(data.get("sync_state", "syncing")),
        queue_depth=int(data.get("queue_depth", 0)),
        active_op=data.get("active_op"),
        last_error=err,
        last_sync_at=last_sync_at,
        pid=pid,
        num_chunks=int(data.get("num_chunks", 0)),
    ).to_dict()


def _normalize_stopped_status(cfg: YawConfig) -> dict[str, Any]:
    status_path = require_path(cfg.status_path, "status_path")
    db_path = require_path(cfg.db_path, "db_path")
    status_data = _read_status_file(status_path)
    num_chunks = read_db_chunk_count(db_path)
    raw_err = status_data.get("last_error")
    err = redact_secrets(str(raw_err), cfg.api_key) if raw_err else None
    raw_sync = status_data.get("last_sync_at")
    last_sync_at = float(raw_sync) if isinstance(raw_sync, (int, float)) and not isinstance(raw_sync, bool) else None
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


def _daemon_status(cfg: YawConfig) -> dict[str, Any]:
    lock_path = require_path(cfg.lock_path, "lock_path")
    if not _is_lock_held(lock_path):
        return _normalize_stopped_status(cfg)
    try:
        code, data = _http_json(cfg, "GET", "/status", timeout=2.0)
        if code == 200 and data:
            return _redact_status_payload(data, cfg.api_key)
    except OSError, http.client.HTTPException, json.JSONDecodeError:
        pass
    if control_token_path(cfg).is_file():
        db_path = require_path(cfg.db_path, "db_path")
        return DaemonStatus(
            lifecycle="starting",
            provider_ready=False,
            sync_state="syncing",
            pid=None,
            num_chunks=read_db_chunk_count(db_path),
        ).to_dict()
    return _normalize_stopped_status(cfg)


def _daemon_run(cfg: YawConfig) -> dict[str, Any]:
    notify_stdout = os.environ.get("YAW_DAEMON_NOTIFY_STDOUT") == "1"
    serve_http(cfg, notify_stdout=notify_stdout)
    return {"lifecycle": "stopped"}


def _spawn_daemon_process(cfg: YawConfig, config_path: Path | None) -> tuple[subprocess.Popen[str], Path]:
    state_dir = require_path(cfg.state_dir, "state_dir")
    cmd = [sys.executable, "-m", "yaw", "daemon", "run"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path.resolve())])
    env = os.environ.copy()
    env["YAW_DAEMON_NOTIFY_STDOUT"] = "1"
    state_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = state_dir / "daemon.stderr.log"
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    else:
        popen_kwargs["start_new_session"] = True
    with stderr_log.open("wb") as stderr_fh:
        proc = subprocess.Popen(cmd, stderr=stderr_fh, **popen_kwargs)
    return proc, stderr_log


def _await_listening_sentinel(proc: subprocess.Popen[str], stderr_log: Path) -> None:
    saw_listening = False
    try:
        if proc.stdout is not None:
            for raw_line in proc.stdout:
                if raw_line.strip() == "YAW_LISTENING":
                    saw_listening = True
                    break
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    if saw_listening:
        return
    try:
        rc = proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait(timeout=2.0)
    err_tail = stderr_log.read_text(encoding="utf-8", errors="replace").strip() if stderr_log.is_file() else ""
    code = "BUSY" if "[BUSY]" in err_tail else "NOT_READY"
    raise YawRuntimeError(code, f"Daemon exited before listening (code={rc}): {err_tail or 'no stderr'}")


def _rollback_started_daemon(cfg: YawConfig, proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    pid: int | None = None
    with contextlib.suppress(OSError, http.client.HTTPException, json.JSONDecodeError):
        code, st = _http_json(cfg, "GET", "/status", timeout=2.0)
        if code == 200:
            raw_pid = st.get("pid")
            if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0:
                pid = raw_pid
    stopped = False
    if pid is not None:
        tok_path = control_token_path(cfg)
        token = tok_path.read_text(encoding="ascii").strip() if tok_path.is_file() else ""
        wait_handle = _open_process_wait_handle(pid)
        if token:
            with contextlib.suppress(OSError, http.client.HTTPException, json.JSONDecodeError):
                stop_code, _ = _http_json(
                    cfg,
                    "POST",
                    "/_internal/stop",
                    timeout=5.0,
                    headers={"X-Yaw-Control-Token": token},
                )
                stopped = stop_code == 200
        if stopped and wait_handle is not None and wait_handle[0] != "pid":
            if not _wait_process_handle(wait_handle, timeout=5.0):
                stopped = False
        else:
            _close_process_wait_handle(wait_handle)
            if stopped and proc.poll() is None:
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    stopped = False
    if not stopped and proc.poll() is None:
        if pid is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)
        if proc.poll() is None:
            proc.kill()
    if proc.poll() is None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)


def _startup_failure_detail(cfg: YawConfig, stderr_log: Path) -> str:
    with contextlib.suppress(OSError, http.client.HTTPException, json.JSONDecodeError):
        st_code, st = _http_json(cfg, "GET", "/status", timeout=2.0)
        if st_code == 200 and st.get("last_error"):
            return redact_secrets(str(st["last_error"]), cfg.api_key)
    err_tail = stderr_log.read_text(encoding="utf-8", errors="replace").strip() if stderr_log.is_file() else ""
    return redact_secrets(err_tail or "Daemon readiness failed.", cfg.api_key)


def _read_startup_probe_failure(stderr_log: Path, exc: BaseException, api_key: str) -> str:
    err_tail = stderr_log.read_text(encoding="utf-8", errors="replace").strip() if stderr_log.is_file() else ""
    probe_err = f"{type(exc).__name__}: {exc}"
    detail = f"{err_tail} ({probe_err})" if err_tail else f"Daemon readiness probe failed ({probe_err})"
    return redact_secrets(detail, api_key)


def _await_daemon_ready(cfg: YawConfig, stderr_log: Path) -> dict[str, Any]:
    code, ready_body = _http_json(cfg, "GET", "/readyz?wait=30", timeout=35.0)
    if code != 200:
        raise YawRuntimeError("NOT_READY", _startup_failure_detail(cfg, stderr_log) or str(ready_body))
    st_code, st_body = _http_json(cfg, "GET", "/status", timeout=5.0)
    if st_code != 200 or not st_body:
        raise YawRuntimeError("NOT_READY", "Daemon status endpoint did not return a status payload.")
    return _redact_status_payload(st_body, cfg.api_key)


def _daemon_start(cfg: YawConfig, config_path: Path | None) -> dict[str, Any]:
    lock_path = require_path(cfg.lock_path, "lock_path")
    if _is_lock_held(lock_path):
        raise YawRuntimeError("BUSY", "Daemon is already running.")
    proc, stderr_log = _spawn_daemon_process(cfg, config_path)
    _await_listening_sentinel(proc, stderr_log)
    try:
        return _await_daemon_ready(cfg, stderr_log)
    except BaseException as exc:
        _rollback_started_daemon(cfg, proc)
        if isinstance(exc, (OSError, http.client.HTTPException, json.JSONDecodeError)):
            raise YawRuntimeError(
                "NOT_READY",
                _read_startup_probe_failure(stderr_log, exc, cfg.api_key),
            ) from exc
        raise


def _daemon_stop(cfg: YawConfig) -> dict[str, Any]:
    lock_path = require_path(cfg.lock_path, "lock_path")
    if not _is_lock_held(lock_path):
        raise YawRuntimeError("NOT_READY", "Daemon is not running.")
    tok_path = control_token_path(cfg)
    if not tok_path.is_file():
        raise YawRuntimeError("NOT_READY", "Daemon control token not found.")
    token = tok_path.read_text(encoding="ascii").strip()
    if not token:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Daemon control token is empty.")

    try:
        st_code, st_body = _http_json(cfg, "GET", "/status", timeout=5.0)
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise YawRuntimeError("NOT_READY", f"Failed to query daemon status before stop: {exc}") from exc
    if st_code != 200:
        raise YawRuntimeError("NOT_READY", f"Daemon /status returned HTTP {st_code}.")
    raw_pid = st_body.get("pid")
    pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0 else None
    wait_handle = _open_process_wait_handle(pid)
    waited = False
    try:
        try:
            stop_code, _ = _http_json(
                cfg,
                "POST",
                "/_internal/stop",
                timeout=5.0,
                headers={"X-Yaw-Control-Token": token},
            )
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            raise YawRuntimeError("BUSY", f"Failed to send HTTP stop request: {exc}") from exc
        if stop_code == 403:
            raise YawRuntimeError("CONSISTENCY_ERROR", "Daemon rejected control token.")
        if stop_code != 200:
            raise YawRuntimeError("BUSY", f"Daemon /_internal/stop returned HTTP {stop_code}.")
        waited = True
        if wait_handle is not None and wait_handle[0] == "pid":
            _wait_lock_release(lock_path, timeout=10.0)
        elif not _wait_process_handle(wait_handle, timeout=10.0):
            raise YawRuntimeError("BUSY", "Timed out waiting for daemon process exit.")
    finally:
        if not waited:
            _close_process_wait_handle(wait_handle)

    _wait_lock_release(lock_path, timeout=2.0)
    return {"lifecycle": "stopped"}


def _cmd_recall(cfg: YawConfig, query: str, top_k: int) -> dict[str, Any]:
    return recall_once_payload(cfg, query, top_k=top_k)


def _cmd_qa(cfg: YawConfig, question: str) -> dict[str, Any]:
    res = ask_once(cfg, question)
    return {
        "question": res.question,
        "answer": res.answer,
        "turns": int(res.turns),
        "citations": build_recall_payload(res.passages)["results"],
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
    print(dump_recall_json(payload, indent=2) if json_out else text)


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


def _dispatch_daemon(args: argparse.Namespace, cfg: YawConfig, config_path: Path | None) -> dict[str, Any]:
    if args.daemon_cmd == "run":
        return _daemon_run(cfg)
    if args.daemon_cmd == "start":
        return _daemon_start(cfg, config_path)
    if args.daemon_cmd == "stop":
        return _daemon_stop(cfg)
    return _daemon_status(cfg)


def _dispatch_command(args: argparse.Namespace, config_path: Path | None, err_ctx: dict[str, str]) -> tuple[Any, str]:
    if args.command == "config":
        payload, text = _cmd_config(args, config_path)
        return payload, text
    cfg = load_config(config_path)
    err_ctx["api_key"] = cfg.api_key
    if args.command == "daemon":
        payload = _dispatch_daemon(args, cfg, config_path)
        return payload, json.dumps(payload, indent=2)
    if args.command == "recall":
        payload = _cmd_recall(cfg, args.query, args.top_k)
        return payload, format_recall_text(payload)
    if args.command == "qa":
        payload = _cmd_qa(cfg, args.question)
        cites = format_recall_text({"results": payload["citations"]}) if payload["citations"] else ""
        text = f"{payload['answer']}\n\n{cites}" if cites else payload["answer"]
        return payload, text

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
    return payload, json.dumps(payload, indent=2)


def _extract_pending_secret(args: argparse.Namespace) -> str:
    if (
        args.command == "config"
        and getattr(args, "config_cmd", None) == "set"
        and str(getattr(args, "key", "")).strip().split(".")[-1] == "api_key"
    ):
        return str(getattr(args, "value", ""))
    return ""


def _resolve_error_api_key(args: argparse.Namespace, config_path: Path | None, err_ctx: dict[str, str]) -> str:
    if err_ctx["api_key"]:
        return err_ctx["api_key"]
    if args.command == "config":
        try:
            return load_config(config_path).api_key
        except YawRuntimeError, ValueError, FileNotFoundError, OSError:
            return ""
    return ""


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
    pending_secret = _extract_pending_secret(args)
    err_ctx = {"api_key": ""}

    try:
        with contextlib.redirect_stdout(sys.stderr) if json_out else contextlib.nullcontext():
            payload, text = _dispatch_command(args, config_path, err_ctx)
        _emit(payload, json_out=json_out, text=text)
        return 0
    except (YawRuntimeError, ValueError, KeyError, FileNotFoundError, OSError, http.client.HTTPException) as exc:
        api_key = _resolve_error_api_key(args, config_path, err_ctx)
        msg = redact_secrets(str(exc), api_key)
        if pending_secret:
            msg = redact_secrets(msg, pending_secret)
            if (escaped := repr(pending_secret)[1:-1]) != pending_secret:
                msg = redact_secrets(msg, escaped)
        sys.stderr.write((json.dumps({"error": msg}) if json_out else msg) + "\n")
        return 1


def _install_win32_console_handler(interrupted_signals: list[int], console_signals: list[int]) -> None:
    if os.name != "nt" or _kernel32 is None:
        return
    handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32)

    def console_interrupt(event: int) -> bool:
        if event not in (0, 1):
            return False
        signum = signal.SIGINT if event == 0 else signal.SIGBREAK
        if console_signals or interrupted_signals:
            os._exit(128 + signum)
        console_signals.append(signum)
        os.write(2, b"Interrupt received; signal again to force exit.\n")
        return False

    handler = handler_type(console_interrupt)
    _kernel32.SetConsoleCtrlHandler.argtypes = [handler_type, ctypes.c_int]
    _kernel32.SetConsoleCtrlHandler.restype = ctypes.c_int
    if not _kernel32.SetConsoleCtrlHandler(handler, 1):
        raise ctypes.WinError()
    _console_handlers.append(handler)


def cli_entrypoint() -> None:
    """Run the CLI with graceful interruption and a repeated-signal escape hatch."""
    interrupted_signals: list[int] = []
    console_signals: list[int] = []

    def interrupt(signum: int, _frame: FrameType | None) -> None:
        if interrupted_signals:
            os._exit(128 + signum)
        interrupted_signals.append(signum)
        if not console_signals:
            os.write(2, b"Interrupt received; signal again to force exit.\n")
        raise KeyboardInterrupt

    signals = (signal.SIGINT, signal.SIGTERM)
    if (sigbreak := getattr(signal, "SIGBREAK", None)) is not None:
        signals += (sigbreak,)
    previous = {sig: signal.signal(sig, interrupt) for sig in signals}
    _install_win32_console_handler(interrupted_signals, console_signals)
    try:
        code = main()
    except KeyboardInterrupt:
        last_sig = interrupted_signals[-1] if interrupted_signals else signal.SIGINT
        code = 128 + last_sig
        sys.stderr.write("Interrupted.\n")
    finally:
        if not interrupted_signals and not console_signals:
            while _console_handlers:
                active_handler = _console_handlers.pop()
                if not _kernel32.SetConsoleCtrlHandler(active_handler, 0):
                    raise ctypes.WinError()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    raise SystemExit(code)
