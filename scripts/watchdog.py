"""Clean-slate telemetry monitor for the 14-file YAW stack and optimization artifacts."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from itertools import accumulate
from pathlib import Path
from typing import Any

import filelock
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import psutil

from yaw.core import YawRuntimeError, configure_logging, get_logger, load_config, write_text_atomic
from yaw.retrieval import open_readonly_db

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = WORKSPACE_ROOT / "yaw.toml"
_DEFAULT_CONFIG_PATH = _CONFIG_PATH
_PROMETHEUS_TPS = "llamacpp:predicted_tokens_seconds"
logger = get_logger("yaw.watchdog")
plt.rcParams.update({"axes.grid": True, "axes.labelcolor": "#e2e8f0", "grid.color": "#1e293b", "svg.fonttype": "none"})


def _resolve_artifact_dir() -> Path:
    if custom := os.environ.get("YAW_ARTIFACT_DIR"):
        return Path(custom)
    brain = Path.home() / ".gemini" / "antigravity" / "brain"
    with contextlib.suppress(OSError):
        if brain.is_dir() and (reps := list(brain.glob("*/benchmark_mobile_report.md"))):
            return max(reps, key=lambda p: p.stat().st_mtime).parent
    return WORKSPACE_ROOT / "outputs" / "dashboard"


ARTIFACT_DIR = _resolve_artifact_dir()


def _nvidia_smi(query: str) -> list[float]:
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0, check=False)
        if res.returncode == 0 and res.stdout.strip():
            return [float(x.strip()) for x in res.stdout.strip().splitlines()[0].split(",")]
    return []


def get_gpu_temperature() -> float | None:
    """Return GPU temperature in Celsius via nvidia-smi, or None if unavailable."""
    vals = _nvidia_smi("temperature.gpu")
    return vals[0] if vals else None


def get_llm_server_metrics(base_url: str, api_key: str) -> dict[str, Any]:
    """Read generation throughput from the llama.cpp ``/metrics`` Prometheus endpoint."""
    root = base_url.rstrip("/").removesuffix("/v1")
    req = urllib.request.Request(f"{root}/metrics", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except OSError, http.client.HTTPException:
        return {"server_up": False}
    gauges: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and len(parts := line.split()) == 2:
            with contextlib.suppress(ValueError):
                gauges[parts[0]] = float(parts[1])
    tps = gauges.get(_PROMETHEUS_TPS)
    return {"server_up": True, "llm_tps": None if tps is None else round(tps, 1)}


def get_pipeline_resource_metrics() -> dict[str, Any]:
    """Sum RSS of unsloth/llama-server/yaw processes and query GPU VRAM usage."""
    rss = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if any(tok in name or tok in cmd for tok in ("unsloth", "llama-server", "yaw")):
                rss += int(proc.memory_info().rss)
        except psutil.Error, OSError:
            continue
    gpu = _nvidia_smi("memory.used,memory.total")
    vram_used, vram_tot = gpu[:2] if len(gpu) >= 2 else (None, None)
    return {
        "pipeline_ram_gb": round(rss / 1024**3, 2),
        "pipeline_vram_gb": round(vram_used / 1024.0, 2) if vram_used is not None else None,
        "total_vram_gb": round(vram_tot / 1024.0, 2) if vram_tot is not None else None,
    }


def is_file_locked(path: Path) -> bool:
    """Return True when ``path`` exists and its OS FileLock is actively held."""
    if not path.is_file():
        return False
    lock = filelock.FileLock(str(path), timeout=0)
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout, OSError:
        return True
    lock.release()
    return False


def _read_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.debug("Skipping torn/unreadable JSON %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _inspect_sqlite_db(db_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(db_path), "exists": db_path.is_file(), "chunks": None, "tables": []}
    try:
        with contextlib.closing(open_readonly_db(db_path)) as conn:
            raw = [str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            info["tables"] = [t for t in raw if not t.startswith(("chunks_", "chunk_fts_", "sqlite_"))]
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            info["chunks"] = int(row[0]) if row else 0
    except (YawRuntimeError, sqlite3.Error) as exc:
        if db_path.is_file() and "no such table" not in str(exc).lower():
            info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def get_active_snapshot(workspace_root: Path | None = None, *, include_llm_metrics: bool = True) -> dict[str, Any]:
    """Collect a read-only telemetry snapshot from configured SQLite DBs, JSON artifacts, locks, and hardware."""
    ws = Path(workspace_root or WORKSPACE_ROOT).resolve()
    cfg_path = _CONFIG_PATH if (workspace_root is None and _CONFIG_PATH != _DEFAULT_CONFIG_PATH) else ws / "yaw.toml"
    cfg = None
    llm_metrics: dict[str, Any] | None = None
    if cfg_path.is_file():
        try:
            cfg = load_config(cfg_path)
            if include_llm_metrics:
                llm_metrics = get_llm_server_metrics(cfg.base_url, cfg.api_key)
        except (YawRuntimeError, OSError) as exc:
            logger.warning("Failed to load config %s: %s", cfg_path, exc)
            if include_llm_metrics:
                llm_metrics = {"server_up": False}
    base_dir = cfg.db_path.parent if cfg and cfg.db_path else ws / ".yaw"
    specs = (
        ("yaw.db", "status", cfg, "status.json"),
        ("optimize_yaw.db", "dspy_status", cfg.for_optimization() if cfg else None, "dspy_status.json"),
        (
            "optuna_index.db",
            "optuna_status",
            cfg.for_optimization(base_dir / "optuna_index.db") if cfg else None,
            "optuna_status.json",
        ),
    )
    dbs, locks, statuses = {}, {}, {}
    for key, st_key, sub, st_name in specs:
        db_p = (sub.db_path if sub else None) or base_dir / key
        dbs[key] = _inspect_sqlite_db(db_p)
        locks[f"{key}.lock"] = is_file_locked((sub.lock_path if sub else None) or Path(f"{db_p}.lock"))
        statuses[st_key] = _read_json_safe((sub.status_path if sub else None) or base_dir / st_name)
    return {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "workspace_root": str(ws),
        "databases": dbs,
        "locks": locks,
        **statuses,
        "dspy_gepa_audit": _read_json_safe(base_dir / "dspy_gepa_audit.json"),
        "optuna_best_params": _read_json_safe(base_dir / "optuna_best_params.json"),
        "llm_metrics": llm_metrics,
        "resources": get_pipeline_resource_metrics(),
        "gpu_temp_c": get_gpu_temperature(),
    }


def _finite_float(val: Any) -> float | None:
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val) else None


def _extract_scores(snapshot: dict[str, Any]) -> tuple[list[float], float | None]:
    for key in ("dspy_gepa_audit", "dspy_status"):
        src = snapshot.get(key)
        raw = src.get("val_aggregate_scores") if isinstance(src, dict) else None
        if isinstance(raw, list) and (scores := [f for v in raw if (f := _finite_float(v)) is not None]):
            return scores, max(scores)
    for key in ("dspy_gepa_audit", "dspy_status", "optuna_best_params", "optuna_status"):
        field = "best_val_score" if key.startswith("dspy") else "best_score"
        if isinstance(src := snapshot.get(key), dict) and (best := _finite_float(src.get(field))) is not None:
            return [], best
    return [], None


def generate_score_svg(dest: Path, snapshot: dict[str, Any]) -> None:
    """Render an atomic SVG plotting GEPA validation scores or a clean waiting placeholder."""
    scores, best_score = _extract_scores(snapshot)
    fig, ax = plt.subplots(figsize=(8.0, 4.0), facecolor="#070b14", subplot_kw={"facecolor": "#090f1d"})
    try:
        ax.tick_params(colors="#94a3b8")
        if scores:
            xs = list(range(1, len(scores) + 1))
            ax.plot(xs, scores, marker="o", color="#38bdf8", linewidth=1.8, label="Candidate val score")
            ax.plot(xs, list(accumulate(scores, max)), linestyle="--", color="#10b981", linewidth=1.5, label="Best")
            ticks = xs if len(xs) <= 16 else xs[:: max(1, len(xs) // 8)]
            ax.set(xticks=ticks, xlabel="Candidate", ylabel="Validation Score")
            ax.set_title(f"DSPy GEPA Score Progress (best={max(scores):.4f})", color="#f8fafc")
        elif best_score is not None:
            ax.axhline(best_score, linestyle="--", color="#10b981", linewidth=1.6)
            ax.plot([1], [best_score], marker="o", markersize=9, color="#38bdf8", label=f"Best: {best_score:.4f}")
            ax.set(xticks=[1], ylabel="Score")
            ax.set_title(f"Optimization Best Score ({best_score:.4f})", color="#f8fafc")
        else:
            ax.text(0.5, 0.5, "Waiting for optimization scores...", ha="center", va="center", color="#94a3b8")
            ax.set(xticks=[], yticks=[])
            ax.set_title("YAW Optimization Score Progress", color="#f8fafc")
        if plotted := (scores or ([best_score] if best_score is not None else [])):
            if min(plotted) >= 0.0 and max(plotted) <= 1.0:
                ax.set_ylim(-0.02, 1.05)
            ax.legend(loc="lower right", frameon=False, labelcolor="#f8fafc")
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
        write_text_atomic(dest, buf.getvalue())
    finally:
        plt.close(fig)


def _fmt(val: Any, spec: str) -> str:
    return format(val, spec) if isinstance(val, (int, float)) and not isinstance(val, bool) else "n/a"


def generate_report_markdown(snapshot: dict[str, Any]) -> str:
    """Format a concise mobile Markdown status report from a snapshot."""
    dbs, locks = snapshot.get("databases") or {}, snapshot.get("locks") or {}
    st, dspy_st = snapshot.get("status") or {}, snapshot.get("dspy_status") or {}
    audit, opt_st = snapshot.get("dspy_gepa_audit") or {}, snapshot.get("optuna_status") or {}
    opt_best, llm = snapshot.get("optuna_best_params") or {}, snapshot.get("llm_metrics") or {}
    res, temp = snapshot.get("resources") or {}, snapshot.get("gpu_temp_c")

    def _db(name: str) -> str:
        info = dbs.get(name) or {}
        tables = ", ".join(info.get("tables") or ()) or "none"
        lk = "locked" if locks.get(f"{name}.lock") else "unlocked"
        return f"`{_fmt(info.get('chunks'), ',')}` chunks • tables: `{tables}` • `{lk}`"

    d_best = _fmt(audit.get("best_val_score", dspy_st.get("best_val_score")), ".4f")
    d_cands = audit.get("num_candidates", len(audit.get("val_aggregate_scores") or ()))
    o_best = _fmt(opt_best.get("best_score", opt_st.get("best_score")), ".4f")
    o_trials = _fmt(opt_best.get("total_trials", opt_st.get("total_trials")), ",")
    llm_st = "up" if llm.get("server_up") else ("down" if "server_up" in llm else "n/a")
    tps = f"{_fmt(llm.get('llm_tps'), '.1f')} tok/s" if llm.get("llm_tps") is not None else "n/a"
    ram = f"{_fmt(res.get('pipeline_ram_gb'), '.2f')} GB"
    vram = f"{_fmt(res.get('pipeline_vram_gb'), '.2f')}/{_fmt(res.get('total_vram_gb'), '.2f')} GB"
    lc, sync, pid = st.get("lifecycle", "stopped"), st.get("sync_state", "idle"), st.get("pid") or "none"
    rows = [
        ("Daemon", f"`{lc}` • `{sync}` • pid `{pid}`"),
        *[(n, _db(n)) for n in ("yaw.db", "optimize_yaw.db", "optuna_index.db")],
        ("DSPy GEPA", f"`{dspy_st.get('state', 'idle')}` • cands `{d_cands or 0}` • best `{d_best}`"),
        ("Optuna", f"`{opt_st.get('state', 'idle')}` • trials `{o_trials}` • `{o_best}`"),
        ("LLM / GPU", f"server `{llm_st}` (`{tps}`) • GPU `{_fmt(temp, '.0f')}°C` • RAM `{ram}` • VRAM `{vram}`"),
    ]
    table = "\n".join(f"| **{k}** | {v} |" for k, v in rows)
    return (
        f"# YAW Live Monitor\n\n*Updated*: `{snapshot.get('timestamp', 'n/a')}`\n\n"
        f"| Component | Status |\n| :--- | :--- |\n{table}\n\n![Score Progress](score_progress.svg)\n"
    )


def check_thermal_guard(
    crisis_temp: float = 82.0, target_temp: float = 72.0, max_pause_sec: float = 300.0, *, poll_sec: float = 10.0
) -> bool:
    """Pause monitoring briefly when GPU temperature meets or exceeds ``crisis_temp``."""
    if (temp := get_gpu_temperature()) is None or temp < crisis_temp:
        return False
    logger.warning("GPU temperature %.1f°C >= %.1f°C; pausing watchdog", temp, crisis_temp)
    deadline = time.monotonic() + max(0.0, max_pause_sec)
    while time.monotonic() < deadline:
        time.sleep(min(poll_sec, max(0.0, deadline - time.monotonic())))
        if (cur := get_gpu_temperature()) is None or cur <= target_temp:
            break
    return True


def run_once(
    *, artifact_dir: Path | None = None, workspace_root: Path | None = None, include_llm_metrics: bool = True
) -> dict[str, Any]:
    """Capture snapshot and atomically write ``score_progress.svg`` and ``benchmark_mobile_report.md``."""
    out_dir = Path(artifact_dir) if artifact_dir is not None else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = get_active_snapshot(workspace_root=workspace_root, include_llm_metrics=include_llm_metrics)
    generate_score_svg(out_dir / "score_progress.svg", snapshot)
    write_text_atomic(out_dir / "benchmark_mobile_report.md", generate_report_markdown(snapshot))
    return snapshot


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for single-shot or periodic YAW watchdog reporting."""
    ap = argparse.ArgumentParser(description="YAW live telemetry monitor")
    ap.add_argument("--watch", type=int, default=None, metavar="SECONDS", help="Poll interval in seconds")
    ap.add_argument("--once", action="store_true", help="Run once and exit (default)")
    ap.add_argument("--artifact-dir", type=Path, default=None, help="Output directory for report and SVG")
    ap.add_argument("--workspace", type=Path, default=None, help="Workspace root containing .yaw/ and yaw.toml")
    args = ap.parse_args(argv)
    if args.watch is not None and args.watch < 1:
        ap.error("--watch must be >= 1")
    handlers = configure_logging(log_file=Path(args.workspace or WORKSPACE_ROOT).resolve() / ".yaw" / "yaw.log")

    def _tick() -> None:
        snap = run_once(artifact_dir=args.artifact_dir, workspace_root=args.workspace)
        chunks = ((snap.get("databases") or {}).get("yaw.db") or {}).get("chunks")
        lc = (snap.get("status") or {}).get("lifecycle", "stopped")
        logger.info("Snapshot updated: daemon=%s, yaw_chunks=%s", lc, chunks)

    try:
        if args.once or args.watch is None:
            _tick()
            return 0
        while True:
            try:
                _tick()
            except (YawRuntimeError, OSError, sqlite3.Error) as exc:
                logger.warning("Watchdog cycle failed: %s", exc)
            for h in handlers:
                h.flush()
            check_thermal_guard()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    finally:
        for h in handlers:
            h.flush()


if __name__ == "__main__":
    sys.exit(main())
