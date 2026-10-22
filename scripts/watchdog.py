"""Deterministic live dashboard generator for benchmark_mobile_report.md and SVG visualizations."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import psutil
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(
    r"C:\Users\Harri.Ahokas\.gemini\antigravity\brain\3061c87c-9ec6-4ff5-b756-80b05e1c4342"
)
# Report figure order (Fig 1..5). Figs 3-5 are indexed by optimizer candidate, not wall-clock.
FIG_FILES = (
    "knowledge_graph_extraction_dashboard.svg",
    "knowledge_graph_2d_projection.png",
    "candidate_score_progress.svg",
    "candidate_quality_metrics.svg",
    "pareto_frontier_progress.svg",
)
# Superseded wall-clock figures; removed once so stale images cannot be mistaken for live data.
LEGACY_FIG_FILES = (
    "knowledge_graph_2d_projection.svg",  # 5 MB; replaced by a compact PNG for mobile
    "candidate_pareto_metric_vs_time.svg",
    "benchmark_optimization_progress.svg",
    "recall_em_f1_vs_time.svg",
    "quality_metrics_vs_time.svg",
)

plt.rcParams.update(
    {
        "figure.facecolor": "#070b14",
        "savefig.facecolor": "#070b14",
        "savefig.format": "svg",
        "svg.fonttype": "none",
        "axes.facecolor": "#090f1d",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "axes.grid": True,
        "grid.color": "#1e293b",
        "grid.linestyle": "--",
        "grid.alpha": 0.65,
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#f8fafc",
        "legend.frameon": False,
        "font.sans-serif": ["Segoe UI", "Roboto", "DejaVu Sans", "Arial"],
    }
)

_CACHE: dict[str, Any] = {}


def _nvidia_smi(query: str) -> list[float]:
    with contextlib.suppress(Exception):
        res = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return [float(x) for x in res.stdout.strip().splitlines()[0].split(",")]
    return []


def get_gpu_temperature() -> float | None:
    vals = _nvidia_smi("temperature.gpu")
    return vals[0] if vals else None


@dataclass(frozen=True)
class PoolTelemetry:
    """Progress figures with a real source; any value without one stays None (rendered "n/a")."""

    openie_cnt: int  # OpenIE-extracted chunks in the sandbox DB (openie_chunks)
    train_queries: int | None  # from the engine's <stem>_pools.json manifest
    val_queries: int | None
    gepa_val_queries: int | None
    val_done: int | None  # manifest val chunk ids ∩ openie_chunks
    val_pool: int | None
    train_done: int | None
    train_pool: int | None
    corpus_done: int | None
    corpus_pool: int | None
    elapsed_hours: float  # wall-clock since the sandbox DB was created
    velocity: float | None  # chunks/min over the last measured window (telemetry samples)
    sec_per_chunk: float | None
    val_eta_min: float | None  # only while upfront val indexing is still running

    @classmethod
    def from_sandbox(cls, sandbox: dict[str, Any]) -> PoolTelemetry:
        cnt = int(sandbox.get("counts", {}).get("openie_chunks", 0))
        pools = sandbox.get("pools") or {}
        indexed: set[str] = sandbox.get("indexed_chunk_ids") or set()

        def _split(key: str) -> tuple[int | None, int | None]:
            ids = pools.get(key)
            if not isinstance(ids, list):
                return None, None
            return len(indexed.intersection(ids)), len(ids)

        val_done, val_pool = _split("val_chunk_ids")
        train_done, train_pool = _split("train_chunk_ids")
        corpus_done, corpus_pool = _split("corpus_chunk_ids")

        samples = sandbox.get("timeseries_samples", [])
        el_sec = max(
            float(sandbox.get("elapsed_seconds", 0.0)),
            float(samples[-1]["time_sec"]) if samples else 0.0,
        )
        vel: float | None = None
        if len(samples) >= 2:
            t_last, p_last = float(samples[-1]["time_sec"]), int(samples[-1]["passages"])
            first = next(s for s in samples if float(s["time_sec"]) >= t_last - 1200.0)
            span = t_last - float(first["time_sec"])
            if span >= 120.0:
                vel = max(0, p_last - int(first["passages"])) / (span / 60.0)
        val_eta = None
        if vel and val_pool is not None and val_done is not None and val_done < val_pool:
            val_eta = (val_pool - val_done) / vel

        def _opt_int(key: str) -> int | None:
            v = pools.get(key)
            return int(v) if isinstance(v, int) else None

        return cls(
            openie_cnt=cnt,
            train_queries=_opt_int("train_queries"),
            val_queries=_opt_int("val_queries"),
            gepa_val_queries=_opt_int("gepa_val_queries"),
            val_done=val_done,
            val_pool=val_pool,
            train_done=train_done,
            train_pool=train_pool,
            corpus_done=corpus_done,
            corpus_pool=corpus_pool,
            elapsed_hours=el_sec / 3600.0,
            velocity=vel,
            sec_per_chunk=(60.0 / vel) if vel else None,
            val_eta_min=val_eta,
        )

    @staticmethod
    def frac(done: int | None, pool: int | None) -> str:
        if done is None or not pool:
            return "n/a"
        return f"{done}/{pool} ({done / pool * 100.0:.1f}%)"

    @staticmethod
    def duration(minutes: float | None) -> str:
        if minutes is None:
            return "n/a"
        return f"{int(minutes // 60)}h {int(minutes % 60)}m"

    def badge(
        self, n_cands: int, n_evals: int, active: bool, active_op: str = ""
    ) -> tuple[str, str]:
        op_suffix = f" • {active_op}" if active_op and active else ""
        if n_cands:
            state = "OPTIMIZER RUNNING" if active else "OPTIMIZER STOPPED"
            col = "#10b981" if active else "#f59e0b"
            return f"{state}: {n_cands} CANDIDATES ({n_evals} EVALS){op_suffix}"[:96], col
        if active:
            return f"OPTIMIZER RUNNING{op_suffix}"[:96], "#10b981"
        return f"OPTIMIZER STOPPED • {self.openie_cnt} CHUNKS INDEXED", "#f59e0b"


def _pareto_2d(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    s = sorted(records, key=lambda r: (float(r["s_per_it"]), -float(r["gm_good"])))
    q = np.array([float(r["gm_good"]) for r in s])
    return [
        r
        for r, k in zip(
            s, np.concatenate(([True], q[1:] > np.maximum.accumulate(q)[:-1])), strict=True
        )
        if k
    ]


def get_lemonade_server_metrics() -> dict[str, Any]:
    try:
        req = urllib.request.Request(
            "http://localhost:13305/metrics", headers={"Authorization": "Bearer lemonade"}
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            m = {
                p[0]: float(p[1])
                for line in resp.read().decode("utf-8", errors="ignore").splitlines()
                if not line.startswith("#") and len(p := line.split()) == 2
            }

        def pick(sub: str) -> float:
            return next((v for k, v in m.items() if "Qwen3-4B" in k and sub in k), 0.0)

        return {
            "server_up": True,
            "qwen_tps": round(pick("tokens_per_second"), 1),
            "qwen_ttft_sec": round(pick("time_to_first_token"), 3),
            "qwen_latest_prompt_tokens": int(pick("model_input_tokens{")),
            "qwen_latest_output_tokens": int(pick("model_output_tokens{")),
            "cpu_usage_pct": round(m.get("lemonade_cpu_usage_percent", 0.0), 1),
            "memory_used_gb": round(m.get("lemonade_memory_used_gb", 0.0), 1),
            "requests_total": int(m.get("lemonade_requests_total", 0)),
            "prompt_tokens_total": int(m.get("lemonade_llamacpp_prompt_tokens_total", 0)),
            "tokens_predicted_total": int(m.get("lemonade_llamacpp_tokens_predicted_total", 0)),
        }
    except Exception:
        return {"server_up": False}


def get_pipeline_resource_metrics() -> dict[str, Any]:
    procs: list[tuple[str, psutil.Process, Any]] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        with contextlib.suppress(psutil.Error, OSError):
            n, cmd = (p.info["name"] or "").lower(), " ".join(p.info["cmdline"] or []).lower()
            role = (
                "lemonade"
                if "lemonade" in n or "lemonade-server" in cmd
                else "llama"
                if "llama-server" in n
                else "yaw"
                if "yaw" in cmd
                else None
            )
            if role:
                procs.append((role, p, p.cpu_times()))

    t1 = time.time()
    time.sleep(0.25)
    dt, ncpu = max(1e-4, time.time() - t1), psutil.cpu_count() or 1
    ram, cpu = Counter(), Counter()
    for role, p, ct1 in procs:
        with contextlib.suppress(psutil.Error, OSError):
            ram[role] += p.memory_info().rss
            ct2 = p.cpu_times()
            cpu[role] += ((ct2.user - ct1.user + ct2.system - ct1.system) / dt) / ncpu * 100.0

    gpu = _nvidia_smi("memory.used,memory.total,utilization.gpu")
    vram_used, vram_tot, gpu_util = gpu[:3] if len(gpu) >= 3 else (None, None, None)
    vm = psutil.virtual_memory()
    return {
        "pipeline_ram_gb": round(sum(ram.values()) / 1024**3, 2),
        "llama_ram_gb": round(ram["llama"] / 1024**3, 2),
        "yaw_ram_gb": round(ram["yaw"] / 1024**3, 2),
        "lemonade_ram_mb": round(ram["lemonade"] / 1024**2, 1),
        "pipeline_vram_gb": round(vram_used / 1024.0, 2) if vram_used is not None else None,
        "total_vram_gb": round(vram_tot / 1024.0, 2) if vram_tot is not None else None,
        "pipeline_cpu_pct": round(sum(cpu.values()), 1),
        "llama_cpu_pct": round(cpu["llama"], 1),
        "yaw_cpu_pct": round(cpu["yaw"], 1),
        "lemonade_cpu_pct": round(cpu["lemonade"], 1),
        "gpu_util_pct": round(gpu_util, 0) if gpu_util is not None else None,
        "host_ram_used_gb": round(vm.used / 1024**3, 1),
        "host_ram_total_gb": round(vm.total / 1024**3, 1),
        "host_cpu_pct": round(psutil.cpu_percent(), 1),
    }


def get_empirical_telemetry_timeseries(
    start_dt: datetime,
    current_elapsed_sec: float,
    current_passages: int,
    current_entities: int,
    current_triples: int,
    db_stem: str = "sidecar",
) -> list[dict[str, Any]]:
    t_file = WORKSPACE_ROOT / ".yaw" / f"{db_stem}_telemetry.json"
    cps: list[dict[str, Any]] = []
    if t_file.exists():
        with contextlib.suppress(Exception):
            cps = json.loads(t_file.read_text(encoding="utf-8"))

    if cps:
        current_elapsed_sec = max(current_elapsed_sec, float(cps[-1].get("time_sec", 0.0)) + 1.0)
    if current_passages > 0 and (
        not cps
        or cps[-1].get("passages") != current_passages
        or abs(current_elapsed_sec - cps[-1].get("time_sec", 0)) >= 45
    ):
        cps.append(
            {
                "time_sec": round(current_elapsed_sec, 2),
                "passages": current_passages,
                "entities": current_entities,
                "triples": current_triples,
            }
        )

    seen, deduped, max_p = set(), [], 0
    for cp in sorted(cps, key=lambda x: (float(x.get("time_sec", 0)), int(x.get("passages", 0)))):
        p_val = int(cp.get("passages", 0))
        key = (round(float(cp.get("time_sec", 0)), 0), p_val)
        if p_val >= max_p - 5 and key not in seen:
            max_p, _ = max(max_p, p_val), seen.add(key)
            deduped.append(cp)

    with contextlib.suppress(Exception):
        t_file.write_text(json.dumps(deduped, indent=2), encoding="utf-8")

    # Interval rate/latency between consecutive real samples only; no seeded defaults.
    samples: list[dict[str, Any]] = []
    prev: tuple[float, int] | None = None
    for i, cp in enumerate(deduped):
        t_s, p_c = float(cp["time_sec"]), int(cp["passages"])
        vel = lat = None
        if prev is not None and t_s - prev[0] > 10:
            dt, dp = t_s - prev[0], max(0, p_c - prev[1])
            vel = round(dp / dt * 60.0, 2)
            lat = round(dt / dp, 1) if dp else None
        samples.append(
            {
                "index": i + 1,
                "time_sec": t_s,
                "time_hours": t_s / 3600.0,
                "passages": p_c,
                "entities": cp["entities"],
                "triples": cp["triples"],
                "rolling_velocity": vel,
                "rolling_latency": lat,
            }
        )
        prev = (t_s, p_c)
    return samples


def _open_ro_sqlite(
    db_path: Path, backoff: tuple[float, ...] = (1.0, 2.0, 3.0)
) -> sqlite3.Connection | None:
    # Never use `immutable=1`: the optimizer writes this DB concurrently (rollback journal), and
    # immutable readers skip locking/journal checks and serve stale or torn pages.
    for i in range(len(backoff) + 1):
        try:
            conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=30.0)
            conn.execute("PRAGMA busy_timeout = 30000;")
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            return conn
        except sqlite3.Error:
            pass
        if i < len(backoff):
            time.sleep(backoff[i])
    return None


def _read_status(status_file: Path) -> dict[str, Any]:
    with contextlib.suppress(OSError, ValueError):
        obj = json.loads(status_file.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("status"), dict):
            return obj["status"]
    return {}


def _pid_alive(pid: Any) -> bool:
    with contextlib.suppress(TypeError, ValueError, psutil.Error):
        return int(pid) > 0 and psutil.pid_exists(int(pid))
    return False


def get_active_sandbox_info() -> dict[str, Any]:
    yaw_dir = WORKSPACE_ROOT / ".yaw"
    dbs = (
        sorted(yaw_dir.glob("yaw_optimize_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        if yaw_dir.exists()
        else []
    )
    if not dbs:
        return {}
    latest_db, sz = dbs[0], dbs[0].stat().st_size
    ds_str = latest_db.stem.removeprefix("yaw_optimize_")
    status = _read_status(yaw_dir / f"status_optimize_{ds_str}.json")
    active_op = str(status.get("active_op") or "")
    optimizer_alive = _pid_alive(status.get("pid")) and status.get("lifecycle") not in (
        "stopped",
        "failed",
    )

    def _parse_ts(p: Path) -> datetime | None:
        with contextlib.suppress(Exception):
            return datetime.strptime(p.stem[13:28], "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        return None

    pools_obj = _read_audit(_pools_path(ds_str))

    start_dt = _parse_ts(latest_db) or datetime.now(UTC)
    info: dict[str, Any] = {
        "db_name": latest_db.name,
        "db_path": str(latest_db),
        "datestamp": ds_str,
        "active_op": active_op,
        "optimizer_alive": optimizer_alive,
        "pools": pools_obj,
        "db_size_mb": round(sz / 1024**2, 2),
        "mtime": datetime.fromtimestamp(latest_db.stat().st_mtime, tz=UTC).isoformat(),
        "start_time": start_dt.isoformat(),
        "elapsed_seconds": max(1.0, (datetime.now(UTC) - start_dt).total_seconds()),
        "lemonade_metrics": get_lemonade_server_metrics(),
        "pipeline_resources": get_pipeline_resource_metrics(),
        # Last good DB snapshot for THIS run only (never another run's numbers).
        **_CACHE.get(str(latest_db), {}),
    }

    conn = _open_ro_sqlite(latest_db)
    if not conn:
        return info

    try:
        counts = {}
        for tbl in ("chunks", "entities", "facts", "edges", "openie_chunks"):
            with contextlib.suppress(sqlite3.Error):
                counts[tbl] = (conn.execute(f"SELECT count(*) FROM {tbl}").fetchone() or [0])[0]
        diff_from_rows: list[tuple[str, str, str, str, str, str]] = []
        related_rows: list[tuple[str, str]] = []
        alt_label_cnt = 0
        with contextlib.suppress(sqlite3.Error):
            diff_from_rows = [
                (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]))
                for r in conn.execute(
                    """
                    SELECT a.entity_id, a.name, COALESCE(a.definition, ''),
                           b.entity_id, b.name, COALESCE(b.definition, '')
                    FROM edges e
                    JOIN entities a ON a.entity_id = e.src_id
                    JOIN entities b ON b.entity_id = e.dst_id
                    WHERE e.kind = 'owl:differentFrom' AND e.src_id < e.dst_id
                    """
                ).fetchall()
            ]
            related_rows = [
                (str(r[0]), str(r[1]))
                for r in conn.execute(
                    """
                    SELECT a.name, b.name
                    FROM edges e
                    JOIN entities a ON a.entity_id = e.src_id
                    JOIN entities b ON b.entity_id = e.dst_id
                    WHERE e.kind = 'skos:related' AND e.src_id < e.dst_id
                    """
                ).fetchall()
            ]
            alt_label_cnt = int(
                (
                    conn.execute(
                        "SELECT COUNT(DISTINCT label_norm || '|' || entity_id) "
                        "FROM entity_labels "
                        "WHERE label_kind IN ('skos:altLabel', 'skos:hiddenLabel')"
                    ).fetchone()
                    or [0]
                )[0]
            )
            lbl_counts = dict(
                conn.execute(
                    "SELECT entity_id, COUNT(DISTINCT label_norm) "
                    "FROM entity_labels GROUP BY entity_id"
                ).fetchall()
            )
            entity_details = {
                str(r[0]): (str(r[1]), str(r[2]), max(1, int(lbl_counts.get(r[0], 1))))
                for r in conn.execute(
                    "SELECT entity_id, name, COALESCE(definition, '') FROM entities"
                ).fetchall()
            }
            graph_edges = [
                (str(r[0]), str(r[1]), str(r[2]))
                for r in conn.execute(
                    """
                    SELECT src_id, dst_id, kind
                    FROM edges
                    WHERE kind IN ('fact', 'skos:related', 'owl:differentFrom')
                      AND src_id < dst_id
                    """
                ).fetchall()
            ]
        rows = conn.execute(
            "SELECT chunk_id, entities_json, triples_json, structured_triples_json "
            "FROM openie_chunks ORDER BY rowid ASC"
        ).fetchall()
        conn.close()

        cum_e = cum_t = 0
        uniq_e, uniq_edges, preds, recent, raw_counts, triples_raw = (
            set(),
            set(),
            Counter(),
            [],
            Counter(),
            [],
        )
        for idx, (cid, ej, tj, _stj) in enumerate(rows):
            ents = json.loads(ej or "[]") if ej else []
            trips = json.loads(tj or "[]") if tj else []
            cum_e, cum_t = cum_e + len(ents), cum_t + len(trips)
            for e in ents:
                if name := str(e[0] if isinstance(e, (list, tuple)) else e).strip():
                    uniq_e.add(name)
                    raw_counts[name] += 1
            for t in trips:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    s, p, o = (
                        str(t[0][0] if isinstance(t[0], (list, tuple)) else t[0]).strip(),
                        str(t[1]).strip(),
                        str(t[2][0] if isinstance(t[2], (list, tuple)) else t[2]).strip(),
                    )
                    if p:
                        preds[p] += 1
                    if s and o and s != o:
                        uniq_edges.add(tuple(sorted((s, o))))
                    if s and p and o:
                        triples_raw.append((s, p, o, str(cid)))
                        raw_counts[s] += 1
                        raw_counts[o] += 1
            if idx >= len(rows) - 4:
                recent.append(
                    {"chunk_id": cid, "entity_count": len(ents), "triple_count": len(trips)}
                )

        info["counts"] = counts
        info["indexed_chunk_ids"] = {str(r[0]) for r in rows}
        info["kg_metrics"] = {
            "total_entities": cum_e,
            "unique_entities": len(uniq_e),
            "avg_entities_per_chunk": round(cum_e / max(1, len(rows)), 1),
            "total_triples": cum_t,
            "unique_edges": len(uniq_edges),
            "avg_triples_per_chunk": round(cum_t / max(1, len(rows)), 1),
            "unique_predicates": len(preds),
            "top_predicates": preds.most_common(6),
            "recent_extractions": recent,
            "different_from_pairs": len(diff_from_rows),
            "related_pairs": len(related_rows),
            "synonym_merges": alt_label_cnt,
        }
        info["graph_cache"] = (raw_counts, triples_raw, diff_from_rows, related_rows)
        info["kg_graph"] = {
            "entities": entity_details,
            "edges": graph_edges,
            "diff_from_rows": diff_from_rows,
            "related_rows": related_rows,
        }
        info["timeseries_samples"] = get_empirical_telemetry_timeseries(
            start_dt, info["elapsed_seconds"], len(rows), cum_e, cum_t, latest_db.stem
        )
        _CACHE.clear()
        _CACHE[str(latest_db)] = {
            k: info[k]
            for k in (
                "counts",
                "indexed_chunk_ids",
                "kg_metrics",
                "graph_cache",
                "kg_graph",
                "timeseries_samples",
            )
        }
    except Exception as exc:
        info["error"] = str(exc)
    return info


_AUDIT_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_CAND_METRICS = ("r1", "r2", "r5", "r10", "r20", "em", "f1")


def _audit_path(datestamp: str) -> Path:
    return WORKSPACE_ROOT / ".yaw" / f"compiled_program_optimize_{datestamp}_graph_rlm_audit.json"


def _pools_path(datestamp: str) -> Path:
    """Split/pool manifest the engine writes as soon as the optimization pools are built."""
    return WORKSPACE_ROOT / ".yaw" / f"compiled_program_optimize_{datestamp}_pools.json"


def _read_audit(path: Path) -> dict[str, Any]:
    """Parse the run's audit once per (mtime, size); keep the last good snapshot on torn reads."""
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    cached = _AUDIT_CACHE.get(str(path))
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return cached[1] if cached is not None else {}
    if not isinstance(obj, dict):
        return cached[1] if cached is not None else {}
    _AUDIT_CACHE[str(path)] = (key, obj)
    return obj


def _geo_quality(row: dict[str, Any]) -> float:
    vals = np.array([max(0.05, float(row.get(k, 0.0))) for k in _CAND_METRICS])
    return round(float(np.exp(np.log(vals).mean())), 4)


def load_candidate_records(sandbox: dict[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    """Return (per-candidate rows, total evals, legacy) for the ACTIVE run only.

    Rows come from the engine's `candidate_records` (one row per distinct predictor-surface
    fingerprint, numbered in first-evaluation order). Audits written before candidate tagging
    only carry per-eval rows; those are returned one-per-eval with `legacy=True` so the x-axis
    is labelled honestly. No other run's files and no synthesized metrics are ever used.
    """
    ds = sandbox.get("datestamp")
    if not ds:
        return [], 0, False
    obj = _read_audit(_audit_path(str(ds)))
    evals = [r for r in (obj.get("eval_records") or []) if isinstance(r, dict)]
    raw = [r for r in (obj.get("candidate_records") or []) if isinstance(r, dict)]
    legacy = not raw and bool(evals)
    if legacy:
        raw = [dict(r, cand=i + 1, n=1) for i, r in enumerate(evals)]
    n_evals = sum(int(r.get("n", 1)) for r in raw)

    rows: list[dict[str, Any]] = []
    for r in sorted(raw, key=lambda x: int(x.get("cand", 0))):
        sc = float(r.get("score", 0.0))
        val = r.get("val_score")
        row: dict[str, Any] = {
            "cand": int(r.get("cand", len(rows) + 1)),
            "n": int(r.get("n", 1)),
            "score": sc,
            "score_min": float(r.get("score_min", sc)),
            "score_max": float(r.get("score_max", sc)),
            "val_score": float(val) if val is not None else None,
            "gepa_idx": r.get("gepa_idx"),
            "is_best": bool(r.get("is_best", False)),
            "optimizer": str(r.get("optimizer", "")),
            "s_per_it": float(r.get("s_per_it", 0.0)),
            "tokens": float(r.get("tokens", 0.0)),
            **{k: float(r.get(k, 0.0)) for k in _CAND_METRICS},
        }
        # Valset score (when GEPA ran one) is the comparable number; minibatch means are noisy.
        row["eff_score"] = row["val_score"] if row["val_score"] is not None else sc
        row["gm_good"] = _geo_quality(row)
        rows.append(row)
    best = 0.0
    for row in rows:
        best = max(best, row["eff_score"])
        row["best_so_far"] = best
    return rows, n_evals, legacy


def read_config_info() -> dict[str, Any]:
    with contextlib.suppress(Exception):
        import tomllib

        return tomllib.loads((WORKSPACE_ROOT / "yaw.toml").read_text("utf-8")).get(
            "optimization", {}
        )
    return {
        "mode": "hybrid",
        "datasets": ["enterprise_rag_bench"],
        "max_queries": 250,
        "train_queries": 150,
        "auto": "light",
    }


def _atomic_write(path: Path, write: Any) -> None:
    """Render to a unique temp file and rename with retries to withstand Windows file watchers."""
    tmp = path.with_name(f".{path.stem}.{os.getpid()}_{time.time_ns()}.tmp{path.suffix}")
    try:
        write(tmp)
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 5:
                    write(path)
                    return
                time.sleep(0.04 * (2**attempt))
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _save(fig: plt.Figure, *paths: Path) -> None:
    try:
        for p in paths:
            if p.suffix == ".png":
                _atomic_write(p, lambda t: fig.savefig(t, format="png", dpi=110))
            else:
                _atomic_write(p, fig.savefig)
    finally:
        plt.close(fig)


def generate_kg_dashboard_svg(dest: Path, sandbox: dict[str, Any], cfg: dict[str, Any]) -> None:
    pt, kg, lem, samples = (
        PoolTelemetry.from_sandbox(sandbox),
        sandbox.get("kg_metrics", {}),
        sandbox.get("lemonade_metrics", {}),
        sandbox.get("timeseries_samples", []),
    )
    ent, trip = int(kg.get("total_entities", 0)), int(kg.get("total_triples", 0))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 1.0], "hspace": 0.26},
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.09, right=0.90)
    ax1_r, ax2_r = ax1.twinx(), ax2.twinx()
    fig.suptitle(
        "Fig. 1: Passage Indexing (OpenIE) & Throughput vs. Wall-Clock Time",
        x=0.04,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    rate = f"{pt.velocity:.2f}/m" if pt.velocity is not None else "n/a"
    tps = f"{lem['qwen_tps']:.0f} tok/s" if lem.get("server_up") else "LLM metrics n/a"
    fig.text(
        0.04,
        0.91,
        f"Val chunks: {pt.frac(pt.val_done, pt.val_pool)} | "
        f"Train chunks: {pt.frac(pt.train_done, pt.train_pool)} | "
        f"Indexed: {pt.openie_cnt} | Rate (20 min): {rate} | {tps} | "
        f"Val ETA: {pt.duration(pt.val_eta_min)}",
        fontsize=9,
        color="#38bdf8",
        fontweight="bold",
    )

    # Origin = sandbox creation (0 chunks); every other point is a recorded sample.
    t_pts = [0.0] + [float(s["time_hours"]) for s in samples]
    ax1.plot(
        t_pts,
        [0.0] + [float(s["passages"]) for s in samples],
        color="#38bdf8",
        lw=1.0,
        marker="o",
        ms=3,
        label=f"Indexed chunks ({pt.openie_cnt})",
    )
    proj_end = pt.elapsed_hours
    if pt.val_eta_min is not None and pt.val_pool is not None and pt.val_done is not None:
        # Projection uses the measured 20-minute rate; it only covers the upfront val pool.
        proj_end = pt.elapsed_hours + pt.val_eta_min / 60.0
        ax1.plot(
            [pt.elapsed_hours, proj_end],
            [pt.openie_cnt, pt.openie_cnt + (pt.val_pool - pt.val_done)],
            color="#38bdf8",
            lw=1.0,
            ls="--",
            label="Val-pool projection",
        )
    y_top = max(10.0, float(pt.openie_cnt))
    for pool, col, lbl in (
        (pt.val_pool, "#10b981", "Val pool"),
        (pt.corpus_pool, "#f59e0b", "Train + val pool"),
    ):
        if pool:
            ax1.axhline(pool, color=col, ls="--", lw=1.0, label=f"{lbl} ({pool})")
            y_top = max(y_top, float(pool))
    ax1_r.plot(
        t_pts,
        [0.0] + [float(s["entities"]) for s in samples],
        color="#818cf8",
        lw=1.5,
        label=f"Entity mentions ({ent:,})",
    )
    ax1_r.plot(
        t_pts,
        [0.0] + [float(s["triples"]) for s in samples],
        color="#f43f5e",
        lw=1.5,
        label=f"Triples ({trip:,})",
    )
    ax1.set_xlim(0.0, max(0.1, pt.elapsed_hours, proj_end) * 1.06)
    ax1.set_ylim(0.0, y_top * 1.15)
    ax1_r.set_ylim(0.0, max(float(max(ent, trip)) * 1.25, 10.0))
    ax1_r.grid(False)
    ax1.set_ylabel("Chunks", color="#38bdf8", fontweight="bold")
    ax1_r.set_ylabel("Entity mentions & triples", color="#818cf8", fontweight="bold")
    ax1.legend(
        ax1.get_legend_handles_labels()[0] + ax1_r.get_legend_handles_labels()[0],
        ax1.get_legend_handles_labels()[1] + ax1_r.get_legend_handles_labels()[1],
        loc="upper left",
        ncol=3,
        fontsize=8,
    )

    rv = [(float(s["time_hours"]), s["rolling_velocity"]) for s in samples]
    rv = [(t, v) for t, v in rv if v is not None]
    rl = [(float(s["time_hours"]), s["rolling_latency"]) for s in samples]
    rl = [(t, v) for t, v in rl if v is not None]
    if rv:
        ax2.plot(
            *zip(*rv, strict=True),
            color="#10b981",
            lw=1.0,
            marker="o",
            ms=2.5,
            label="Interval rate (chunks/min)",
        )
        ax2.set_ylim(0.0, max(1.0, max(v for _, v in rv) * 1.25))
    if rl:
        ax2_r.plot(
            *zip(*rl, strict=True),
            color="#fbbf24",
            lw=1.0,
            ls="--",
            marker="o",
            ms=2.5,
            label="Interval latency (s/chunk)",
        )
        ax2_r.set_ylim(0.0, max(1.0, max(v for _, v in rl) * 1.25))
    if not rv and not rl:
        ax2.text(
            0.5,
            0.5,
            "Not enough samples yet",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            color="#94a3b8",
        )
    ax2_r.grid(False)
    ax2.set_ylabel("Rate (/min)", color="#10b981", fontweight="bold")
    ax2_r.set_ylabel("Latency (s)", color="#fbbf24", fontweight="bold")
    ax2.set_xlabel("Wall-clock time since sandbox creation (hours) →", fontweight="bold")
    handles = ax2.get_legend_handles_labels()[0] + ax2_r.get_legend_handles_labels()[0]
    if handles:
        ax2.legend(
            handles,
            ax2.get_legend_handles_labels()[1] + ax2_r.get_legend_handles_labels()[1],
            loc="upper left",
            ncol=2,
            fontsize=8,
        )
    _save(fig, dest)


def generate_graph_projection_svg(
    dest: Path, sandbox: dict[str, Any], badge: tuple[str, str]
) -> dict[str, Any]:
    kg_g = sandbox.get("kg_graph")
    if not kg_g:
        _placeholder_svg(
            dest,
            "Fig. 2: Knowledge Graph Network & Disambiguation",
            badge,
            f"No extracted entities yet in {sandbox.get('db_name', 'the active sandbox')}",
        )
        return {}

    entities: dict[str, tuple[str, str, int]] = kg_g.get("entities", {})
    all_edges: list[tuple[str, str, str]] = kg_g.get("edges", [])
    diff_from_rows: list[tuple[str, str, str, str, str, str]] = kg_g.get("diff_from_rows", [])

    if not entities and not all_edges and not diff_from_rows:
        _placeholder_svg(
            dest,
            "Fig. 2: Knowledge Graph Network & Disambiguation",
            badge,
            f"No extracted entities yet in {sandbox.get('db_name', 'the active sandbox')}",
        )
        return {}

    # Identify all owl:differentFrom and skos:related entities
    diff_nodes: set[str] = {u for u, v, k in all_edges if k == "owl:differentFrom"} | {
        v for u, v, k in all_edges if k == "owl:differentFrom"
    }
    rel_nodes: set[str] = {u for u, v, k in all_edges if k == "skos:related"} | {
        v for u, v, k in all_edges if k == "skos:related"
    }
    base_nodes = diff_nodes | rel_nodes

    # Include all 1-hop connected rdf:Statement (fact) entities
    inc_fact: dict[str, set[str]] = defaultdict(set)
    for u, v, k in all_edges:
        if k == "fact":
            if u in base_nodes:
                inc_fact[u].add(v)
            if v in base_nodes:
                inc_fact[v].add(u)

    selected: set[str] = set(base_nodes)
    for nbrs in inc_fact.values():
        selected.update(nbrs)

    nodes = sorted(
        selected,
        key=lambda x: (x in diff_nodes, x in rel_nodes, len(inc_fact[x])),
        reverse=True,
    )
    if not nodes:
        _placeholder_svg(
            dest,
            "Fig. 2: Knowledge Graph Network & Disambiguation",
            badge,
            f"No connected entities yet in {sandbox.get('db_name', 'the active sandbox')}",
        )
        return {}

    idx = {eid: i for i, eid in enumerate(nodes)}

    # Segment lists
    diff_segs = [
        [idx[u], idx[v]]
        for u, v, k in all_edges
        if k == "owl:differentFrom" and u in idx and v in idx
    ]
    rel_segs = [
        [idx[u], idx[v]]
        for u, v, k in all_edges
        if k == "skos:related" and u in idx and v in idx
    ]
    fact_segs = [
        [idx[u], idx[v]]
        for u, v, k in all_edges
        if k == "fact" and u in idx and v in idx
    ]
    active_edge_arr = (
        np.array(diff_segs + rel_segs + fact_segs, dtype=int)
        if (diff_segs or rel_segs or fact_segs)
        else np.zeros((0, 2), dtype=int)
    )

    n_cnt = len(nodes)
    rng = np.random.default_rng(42)

    if n_cnt <= 250:
        ang = np.linspace(0, 2 * math.pi, n_cnt, endpoint=False)
        rad = 220.0 + rng.standard_normal(n_cnt) * 30.0
        pos = np.column_stack([np.cos(ang) * rad, np.sin(ang) * rad])
        vel = np.zeros((n_cnt, 2))
        for _ in range(80):
            diff = pos[:, None, :] - pos[None, :, :]
            d2 = np.sum(diff**2, axis=-1) + 25.0
            f_rep = 8500.0 / (d2 * np.sqrt(d2))
            np.fill_diagonal(f_rep, 0.0)
            f = np.sum(diff * f_rep[:, :, None], axis=1) - pos * 0.012
            if len(active_edge_arr):
                de = pos[active_edge_arr[:, 1]] - pos[active_edge_arr[:, 0]]
                dl = np.linalg.norm(de, axis=1, keepdims=True) + 1e-3
                fe = (dl - 35.0) * 0.035 * (de / dl)
                np.add.at(f, active_edge_arr[:, 0], fe)
                np.add.at(f, active_edge_arr[:, 1], -fe)
            vel = (vel + f) * 0.80
            sp = np.linalg.norm(vel, axis=1, keepdims=True)
            vel = np.where(sp > 8.0, vel / sp * 8.0, vel)
            pos += vel
    else:
        import scipy.sparse as sp
        import scipy.sparse.csgraph as csg

        if len(active_edge_arr):
            u_e = active_edge_arr[:, 0]
            v_e = active_edge_arr[:, 1]
            adj = sp.coo_matrix(
                (np.ones(len(u_e) * 2), (np.concatenate([u_e, v_e]), np.concatenate([v_e, u_e]))),
                shape=(n_cnt, n_cnt),
            ).tocsr()
            n_comp, comp_labels = csg.connected_components(adj, directed=False)
        else:
            n_comp, comp_labels = n_cnt, np.arange(n_cnt)

        pos = np.zeros((n_cnt, 2))
        comp_nodes = [[] for _ in range(n_comp)]
        for i, c in enumerate(comp_labels):
            comp_nodes[c].append(i)
        comp_nodes.sort(key=len, reverse=True)

        for c_idx, comp in enumerate(comp_nodes):
            c_len = len(comp)
            phi = c_idx * 2.399963229728653
            radius = 16.0 * math.sqrt(c_idx + 1) * 3.2
            center = np.array([math.cos(phi) * radius, math.sin(phi) * radius])

            if c_len == 1:
                pos[comp[0]] = center
            elif c_len == 2:
                pos[comp[0]] = center + np.array([-14.0, 0.0])
                pos[comp[1]] = center + np.array([14.0, 0.0])
            else:
                angles = np.linspace(0, 2 * math.pi, c_len, endpoint=False)
                r_c = 14.0 + math.sqrt(c_len) * 7.5
                pos[comp] = (
                    center
                    + np.column_stack([np.cos(angles) * r_c, np.sin(angles) * r_c])
                    + rng.standard_normal((c_len, 2)) * 2.5
                )

        deg_arr = np.maximum(np.bincount(active_edge_arr.ravel(), minlength=n_cnt), 1)[:, None]
        for _ in range(40):
            if len(active_edge_arr):
                de = pos[active_edge_arr[:, 1]] - pos[active_edge_arr[:, 0]]
                dl = np.linalg.norm(de, axis=1, keepdims=True) + 1e-2
                fe = (dl - 26.0) * 0.35 * (de / dl)
                step = np.zeros_like(pos)
                np.add.at(step, active_edge_arr[:, 0], fe)
                np.add.at(step, active_edge_arr[:, 1], -fe)
                pos += step / deg_arr  # degree-normalised: hubs cannot blow up
            sample_idx = rng.choice(n_cnt, size=min(64, n_cnt), replace=False)
            diff_s = pos[:, None, :] - pos[sample_idx][None, :, :]
            d2_s = np.sum(diff_s**2, axis=-1) + 40.0
            f_rep = np.sum(diff_s * (6000.0 / (d2_s * np.sqrt(d2_s)))[:, :, None], axis=1)
            pos += np.clip(f_rep * 0.04, -8.0, 8.0) - pos * 0.003

    fig, ax = plt.subplots(figsize=(10.2, 7.8))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.06)

    fig.suptitle(
        "Fig. 2: Knowledge Graph Network & Disambiguation",
        x=0.04,
        y=0.96,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.915,
        f"{n_cnt} Entities ({len(diff_nodes)} owl:differentFrom, {len(rel_nodes)} skos:related, "
        f"{n_cnt - len(base_nodes)} 1-Hop rdf:Statement) • {len(fact_segs)} Fact Edges • "
        f"{len(rel_segs)} skos:related • {len(diff_segs)} owl:differentFrom (\u2260)",
        fontsize=9.5,
        color="#38bdf8",
        fontweight="bold",
    )

    # 1. Fact Triple Edges (solid slate)
    if fact_segs:
        ax.add_collection(
            LineCollection(
                [[pos[u], pos[v]] for u, v in fact_segs],
                colors="#475569",
                linewidths=1,
                linestyles="solid",
                alpha=0.8,
                zorder=1,
            )
        )

    # 2. skos:related Edges (purple / violet)
    if rel_segs:
        ax.add_collection(
            LineCollection(
                [[pos[u], pos[v]] for u, v in rel_segs],
                colors="#c084fc",
                linewidths=1,
                linestyles="solid",
                alpha=0.8,
                zorder=2,
            )
        )

    # 3. owl:differentFrom Edges (crimson)
    if diff_segs:
        ax.add_collection(
            LineCollection(
                [[pos[u], pos[v]] for u, v in diff_segs],
                colors="#f43f5e",
                linewidths=1,
                linestyles="solid",
                alpha=0.8,
                zorder=3,
            )
        )

    # Node Sizing: strictly based on skos:*Label count
    sizes, colors, edgecolors, lws = [], [], [], []
    base_sz = 6.0 if n_cnt > 2000 else (8.0 if n_cnt > 500 else 10.0)
    for eid in nodes:
        name, defn, skos_cnt = entities.get(eid, (eid, "", 1))
        # Base size scaled with skos:*Label count (synonyms & lexical variants)
        s = min(300.0, base_sz + 9.9 * (skos_cnt - 1))
        sizes.append(s)
        if eid in diff_nodes:
            colors.append("#f43f5e")  # rose/crimson for disambiguation endpoints
            edgecolors.append("#ffffff")
            lws.append(0.5)
        elif eid in rel_nodes:
            colors.append("#a855f7")  # violet for skos:related context nodes
            edgecolors.append("#c084fc")
            lws.append(0.5)
        elif skos_cnt > 1:
            colors.append("#38bdf8")  # sky blue for multi-label entities
            edgecolors.append("#0ea5e9")
            lws.append(0.5)
        else:
            colors.append("#64748b")  # slate for standard single-label entities
            edgecolors.append("#1e293b")
            lws.append(0.5)

    ax.scatter(
        pos[:, 0], pos[:, 1], s=sizes, c=colors, edgecolors=edgecolors, linewidths=lws, zorder=4
    )

    # Entity Name Labels:
    # Show entity names exclusively for both ends of owl:differentFrom disambiguation;
    # other entity names are suppressed per user request.
    diff_peer_map: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for ea, na, da, eb, nb, db in diff_from_rows:
        diff_peer_map[ea].append((eb, nb, db))
        diff_peer_map[eb].append((ea, na, da))

    for eid in diff_nodes:
        if eid not in idx:
            continue
        i = idx[eid]
        name, defn, _ = entities.get(eid, (eid, "", 1))
        # Could well be possible to just show the name to save space
#        peers = diff_peer_map.get(eid, [])
#        has_homograph = any(
#            p_name.strip().lower() == name.strip().lower() for _, p_name, _ in peers
#        )
#        if has_homograph:
#            words = [w for w in defn.replace(";", " ").replace(".", " ").split() if len(w) > 2]
#            hint = " ".join(words[:2]) if words else eid[-4:]
#            disp_label = f"{name} [{hint}]"
#        else:
        disp_label = name

        ax.text(
            pos[i, 0],
            pos[i, 1] + 4.0,
            f" {disp_label[:30]}",
            fontsize=7,
            fontweight="normal",
            color="#ffffff",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="#070b14", ec="#475569", lw=0.5, alpha=0.90),
            zorder=5,
        )

    # Legend explaining edge types and node sizing
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#f43f5e",
            lw=1.0,
            ls="-",
            label=f"owl:differentFrom ({len(diff_segs)} active \u2260)",
        ),
        Line2D(
            [0],
            [0],
            color="#c084fc",
            lw=1.0,
            ls="-",
            label=f"skos:related ({len(rel_segs)} links)",
        ),
        Line2D(
            [0],
            [0],
            color="#475569",
            lw=1.0,
            ls="-",
            label=f"rdf:Statement ({len(fact_segs)} edges)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#f43f5e",
            markeredgecolor="w",
            markersize=8,
            label="owl:differentFrom assertions",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#818cf8",
            markeredgecolor="#0f172a",
            markersize=6,
            label="Node size ~ skos:*Label count",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.88,
        facecolor="#090f1d",
        edgecolor="#334155",
        fontsize=7.8,
        labelcolor="#e2e8f0",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, dest)
    return {
        "nodes_rendered": n_cnt,
        "diff_nodes": len(diff_nodes),
        "active_edges": len(active_edge_arr),
        "diff_edges": len(diff_segs),
        "related_edges": len(rel_segs),
        "fact_edges": len(fact_segs),
    }


def _placeholder_svg(dest: Path, title: str, badge: tuple[str, str], message: str) -> None:
    """Always overwrite the figure so an empty run never leaves a previous run's image behind."""
    fig = plt.figure(figsize=(9.4, 4.2))
    fig.suptitle(title, x=0.04, y=0.93, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.04, 0.84, badge[0], fontsize=9.5, color=badge[1], fontweight="bold")
    fig.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, color="#94a3b8")
    _save(fig, dest)


def _header(fig: plt.Figure, title: str, badge: tuple[str, str], y: float = 0.96) -> None:
    fig.suptitle(title, x=0.04, y=y, ha="left", fontsize=13, fontweight="bold")
    # Keep a constant ~0.32in gap between title and badge regardless of figure height.
    fig.text(
        0.04,
        y - 0.32 / fig.get_figheight(),
        badge[0],
        fontsize=9.5,
        color=badge[1],
        fontweight="bold",
    )


def _candidate_xaxis(ax: plt.Axes, cands: list[dict[str, Any]], legacy: bool) -> None:
    """Integer candidate axis with fixed padding so 1..N candidates never autoscale oddly."""
    hi = max(int(c["cand"]) for c in cands)
    ax.set_xlim(0.4, hi + 0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=14, min_n_ticks=1))
    ax.set_xlabel(
        "Evaluation # (legacy audit without candidate tags) →"
        if legacy
        else "Candidate # (order of first evaluation) →",
        fontweight="bold",
    )


def _no_candidates_message(sandbox: dict[str, Any]) -> str:
    if not sandbox.get("datestamp"):
        return "No optimization sandbox found in .yaw/"
    pt = PoolTelemetry.from_sandbox(sandbox)
    op = sandbox.get("active_op") or "in progress"
    db = sandbox.get("db_name")
    if not sandbox.get("optimizer_alive"):
        return f"No candidate evaluations recorded for {db}\n(optimizer stopped)"
    prog = f"Upfront Indexing: {pt.frac(pt.val_done, pt.val_pool)} validation chunks"
    return f"{prog} • Candidate evolution begins once indexed\nActive: {op}"


def generate_candidate_score_svg(
    dest: Path,
    cands: list[dict[str, Any]],
    legacy: bool,
    badge: tuple[str, str],
    sandbox: dict[str, Any],
) -> None:
    title = "Fig. 3: Candidate Score & Evaluation Cost per Candidate"
    if not cands:
        _placeholder_svg(dest, title, badge, _no_candidates_message(sandbox))
        return
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.14},
    )
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.09, right=0.97)
    _header(fig, title, badge)

    x = np.array([c["cand"] for c in cands], dtype=float)
    sc = np.array([c["score"] for c in cands]) * 100.0
    lo = np.array([c["score_min"] for c in cands]) * 100.0
    hi = np.array([c["score_max"] for c in cands]) * 100.0
    best = np.array([c["best_so_far"] for c in cands]) * 100.0

    ax1.step(x, best, where="post", color="#10b981", lw=1.0, label=f"Best so far ({best[-1]:.1f}%)")
    ax1.errorbar(
        x,
        sc,
        yerr=np.vstack([np.clip(sc - lo, 0, None), np.clip(hi - sc, 0, None)]),
        fmt="o",
        ms=5,
        color="#38bdf8",
        ecolor="#38bdf8",
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.9,
        label="Mean score (bar = min–max over evals)",
    )
    val = [(c["cand"], c["val_score"] * 100.0) for c in cands if c["val_score"] is not None]
    if val:
        ax1.scatter(
            *zip(*val, strict=True),
            marker="D",
            s=48,
            facecolors="none",
            edgecolors="#fbbf24",
            lw=1.0,
            zorder=4,
            label="GEPA valset score",
        )
    if best_c := [c for c in cands if c["is_best"]]:
        ax1.scatter(
            [c["cand"] for c in best_c],
            [c["eff_score"] * 100.0 for c in best_c],
            marker="*",
            s=150,
            color="#f472b6",
            zorder=5,
            label="GEPA best program",
        )
    top = max(float(hi.max()), float(best.max()), max((v for _, v in val), default=0.0))
    ax1.set_ylim(0.0, min(105.0, max(10.0, top * 1.2 + 2.0)))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax1.set_ylabel("Score (%)", color="#10b981", fontweight="bold")
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    lat = np.array([c["s_per_it"] for c in cands])
    ax2.bar(x, lat, width=0.6, color="#f43f5e", alpha=0.75, label="Mean latency per eval")
    ax2.set_ylim(0.0, max(1.0, float(lat.max()) * 1.3))
    if len(cands) <= 40:
        for c, li in zip(cands, lat, strict=True):
            ax2.text(
                c["cand"], li, f"n={c['n']}", ha="center", va="bottom", fontsize=7, color="#cbd5e1"
            )
    ax2.set_ylabel("Latency (s / eval)", color="#f43f5e", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    _candidate_xaxis(ax2, cands, legacy)
    _save(fig, dest)


def generate_candidate_quality_svg(
    dest: Path,
    cands: list[dict[str, Any]],
    legacy: bool,
    badge: tuple[str, str],
    sandbox: dict[str, Any],
) -> None:
    title = "Fig. 4: Retrieval Recall@k & QA Quality (EM, F1) per Candidate"
    if not cands:
        _placeholder_svg(dest, title, badge, _no_candidates_message(sandbox))
        return
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.22},
    )
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.09, right=0.97)
    _header(fig, title, badge)
    x = [c["cand"] for c in cands]
    ms = 4 if len(cands) <= 60 else 0
    for k, col in (
        ("r20", "#14b8a6"),
        ("r10", "#06b6d4"),
        ("r5", "#10b981"),
        ("r2", "#818cf8"),
        ("r1", "#38bdf8"),
    ):
        ax1.plot(
            x,
            [c[k] * 100.0 for c in cands],
            color=col,
            lw=1.0,
            marker="o",
            ms=ms,
            label=f"R@{k[1:]}",
        )
    for ax in (ax1, ax2):
        ax.set_ylim(0.0, 105.0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        # Set background color of the top plot area
        ax.set_facecolor("#0a0a0a")
        # Add horizontal grid lines at the major y-axis marks (0%, 25%, 50%, 75%, 100%)
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="#3b3b3b")
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.5, color="#1f1f1f")
        ax.set_axisbelow(True)  # Keeps grid lines behind your plotted data lines
    ax1.set_ylabel("Mean Recall", color="#10b981", fontweight="bold")
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, fontsize=8)
    ax2.plot(
        x,
        [c["f1"] * 100.0 for c in cands],
        color="#f43f5e",
        lw=1.0,
        marker="o",
        ms=ms,
        label="QA F1",
    )
    ax2.plot(
        x, [c["em"] * 100.0 for c in cands], color="#fbbf24", lw=1.0, marker="s", ms=ms, label="EM"
    )
    ax2.set_ylabel("Mean QA Score", color="#f43f5e", fontweight="bold")
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8)
    _candidate_xaxis(ax2, cands, legacy)
    _save(fig, dest)


def generate_pareto_svg(
    dest: Path,
    cands: list[dict[str, Any]],
    legacy: bool,
    badge: tuple[str, str],
    sandbox: dict[str, Any],
) -> None:
    title = "Fig. 5: Quality–Latency Pareto Frontier across Candidates"
    if not cands:
        _placeholder_svg(dest, title, badge, _no_candidates_message(sandbox))
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 5.6), gridspec_kw={"wspace": 0.32})
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.97)
    _header(fig, title, badge, y=0.95)
    nd = _pareto_2d(cands)
    lat = np.array([c["s_per_it"] for c in cands])
    gm = np.array([c["gm_good"] for c in cands])
    sc = ax1.scatter(
        lat,
        gm,
        c=[c["cand"] for c in cands],
        cmap="viridis",
        vmin=1,
        vmax=max(2, len(cands)),
        s=30,
        alpha=0.85,
        zorder=3,
    )
    ax1.plot(
        [r["s_per_it"] for r in nd],
        [r["gm_good"] for r in nd],
        color="#10b981",
        lw=1.0,
        marker="o",
        zorder=4,
        label=f"Pareto front ({len(nd)})",
    )
    cb = fig.colorbar(sc, ax=ax1, pad=0.02, fraction=0.05)
    cb.set_label("Evaluation #" if legacy else "Candidate #", color="#e2e8f0")
    cb.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xlim(0.0, max(1.0, float(lat.max()) * 1.15))
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Latency τ (s / eval) →")
    ax1.set_ylabel("GM quality (R@k, EM, F1) ↑", color="#38bdf8", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)

    arch = [len(_pareto_2d(cands[: k + 1])) for k in range(len(cands))]
    ax2.step(
        [c["cand"] for c in cands],
        arch,
        where="post",
        color="#10b981",
        lw=1.0,
        label=f"Frontier size ({arch[-1]})",
    )
    ax2.set_ylim(0.0, max(arch) + 1.0)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel("Archive size ↑", color="#10b981", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    _candidate_xaxis(ax2, cands, legacy)
    _save(fig, dest)


def generate_report_markdown(
    dest: Path,
    sandbox: dict[str, Any],
    cands: list[dict[str, Any]],
    n_evals: int,
    legacy: bool,
    badge: tuple[str, str],
) -> None:
    """Every value below is read from the run's DB, its engine manifests, or live probes."""
    pt, kg, lem, res = (
        PoolTelemetry.from_sandbox(sandbox),
        sandbox.get("kg_metrics", {}),
        sandbox.get("lemonade_metrics", {}),
        sandbox.get("pipeline_resources", {}),
    )
    counts = sandbox.get("counts", {})
    art = ARTIFACT_DIR.as_posix()
    gpu_t = get_gpu_temperature()

    def _v(x: Any, fmt: str = "{}") -> str:
        return "n/a" if x is None else fmt.format(x)

    split = (
        f"Train **`{pt.train_queries}`** queries • Val **`{pt.val_queries}`** queries "
        f"(GEPA valset `{_v(pt.gepa_val_queries)}`)"
        if pt.train_queries is not None
        else "n/a (engine has not published the split manifest yet)"
    )
    rows = [
        ("Dataset Split", split),
        (
            "Indexing (OpenIE)",
            f"Val chunks **`{pt.frac(pt.val_done, pt.val_pool)}`** • "
            f"Train chunks **`{pt.frac(pt.train_done, pt.train_pool)}`** • "
            f"Indexed **`{pt.openie_cnt}`** of `{_v(pt.corpus_pool)}` train+val chunks • "
            f"Val ETA `{pt.duration(pt.val_eta_min)}`",
        ),
    ]
    if cands:
        last = cands[-1]
        unit = "evals (legacy, untagged)" if legacy else "candidates"
        rows.append(
            (
                "Optimizer Candidates",
                f"**`{len(cands)}` {unit}** (`{n_evals}` evals) • "
                f"Latest `#{last['cand']}` (n=`{last['n']}`): "
                f"R@5 `{last['r5'] * 100:.0f}%`, F1 `{last['f1'] * 100:.1f}%`",
            )
        )
        # Progress is judged on full-valset F1 / R@5, not the composite score or 10-example
        # minibatch means. Only candidates GEPA scored on the whole valset qualify.
        full = [c for c in cands if c.get("val_score") is not None]
        if full:
            seed = full[0]
            best_f1 = max(full, key=lambda c: c["f1"])
            best_r5 = max(full, key=lambda c: c["r5"])
            rows.append(
                (
                    "Full-Valset Front",
                    f"`{len(full)}` fully scored • Best F1 **`#{best_f1['cand']}` "
                    f"`{best_f1['f1'] * 100:.1f}%`** ({(best_f1['f1'] - seed['f1']) * 100:+.1f}pt "
                    f"vs seed) • Best R@5 **`#{best_r5['cand']}` `{best_r5['r5'] * 100:.0f}%`** "
                    f"({(best_r5['r5'] - seed['r5']) * 100:+.0f}pt vs seed)",
                )
            )
    rows += [
        (
            "Throughput",
            f"**`{_v(pt.velocity, '{:.2f}')}` chunks/min** (last 20 min, "
            f"`{_v(pt.sec_per_chunk, '{:.1f}')}` s/chunk) • LLM "
            f"**`{_v(lem.get('qwen_tps') if lem.get('server_up') else None, '{:.1f}')}` tok/s**",
        ),
        (
            "GPU / Memory",
            f"GPU **`{_v(gpu_t, '{:.0f}°C')}`** • Pipeline RAM "
            f"**`{_v(res.get('pipeline_ram_gb'), '{:.2f} GB')}`** • VRAM "
            f"**`{_v(res.get('pipeline_vram_gb'), '{:.2f}')}` / "
            f"`{_v(res.get('total_vram_gb'), '{:.2f}')}` GB**",
        ),
        (
            "Graph (sandbox DB)",
            f"**`{counts.get('entities', 0):,}` entities** • "
            f"**`{counts.get('facts', 0):,}` facts** • "
            f"`{kg.get('synonym_merges', 0)}` alt/hidden labels • "
            f"`{kg.get('related_pairs', 0)}` `skos:related` • "
            f"`{kg.get('different_from_pairs', 0)}` `owl:differentFrom`",
        ),
    ]
    table = "\n".join(f"| **{k}** | {v} |" for k, v in rows)

    kg_g = sandbox.get("kg_graph", {})
    diff_from_rows = kg_g.get("diff_from_rows", [])
    assertions_md = ""
    if diff_from_rows:
        # Mobile-friendly: a narrow bullet list instead of a wide multi-column table.
        items = []
        for _ea, na, da, _eb, nb, db in diff_from_rows[:10]:
            da_s = da.strip()[:110] or "n/a"
            db_s = db.strip()[:110] or "n/a"
            items.append(f"- **`{na}`** ≠ **`{nb}`**\n  - A: *{da_s}*\n  - B: *{db_s}*")
        total_pairs = len(diff_from_rows)
        trunc = (
            f"\n*(Showing 10 of {total_pairs} active disambiguation pairs)*\n"
            if total_pairs > 10
            else "\n"
        )
        assertions_md = (
            "### Graph Assertions for Intervention\n"
            "*If an `owl:differentFrom` assertion below is false, it is an intervention "
            "candidate in `yaw.db`.*\n\n" + "\n".join(items) + f"\n{trunc}\n"
        )

    # Small image files referenced by path (no inline base64: that made the report multi-MB and
    # unrenderable on mobile). Fig. 2 is a compact PNG; the others are small SVGs.
    fig_titles = [
        "Fig. 1: Extraction & Throughput",
        "Fig. 2: Knowledge Graph & Disambiguation",
        "Fig. 3: Candidate Score",
        "Fig. 4: Retrieval Quality",
        "Fig. 5: Pareto Frontier",
    ]
    figs = "\n\n".join(
        f"![{t}](/{art}/{name})" for t, name in zip(fig_titles, FIG_FILES, strict=True)
    )
    md = (
        f"# Live Optimization Telemetry (`{sandbox.get('db_name', 'no sandbox')}`)\n\n"
        f"*Updated*: `{datetime.now(UTC):%Y-%m-%d %H:%M:%S UTC}` • *State*: **`{badge[0]}`**\n\n"
        "| Metric | Live Value |\n| :--- | :--- |\n"
        f"{table}\n\n"
        f"{assertions_md}"
        f"{figs}\n"
    )
    _atomic_write(dest, lambda p: p.write_text(md, encoding="utf-8"))


@contextlib.contextmanager
def _suspended_llama(reason: str):
    targets = [
        p
        for p in psutil.process_iter(["pid", "name"])
        if "llama-server" in (p.info.get("name") or "").lower()
    ]
    if targets:
        ts = f"{datetime.now(UTC):%H:%M:%S}"
        print(
            f"[{ts}] Suspending {len(targets)} llama-server process(es) ({reason})...",
            flush=True,
        )
        for p in targets:
            with contextlib.suppress(psutil.Error, OSError):
                p.suspend()
    try:
        yield bool(targets)
    finally:
        for p in targets:
            with contextlib.suppress(psutil.Error, OSError):
                p.resume()


def check_thermal_guard(
    crisis_temp: float = 82.0, target_temp: float = 72.0, max_pause_sec: int = 300
) -> bool:
    if (temp := get_gpu_temperature()) is None or temp < crisis_temp:
        return False
    with _suspended_llama(f"GPU {temp:.1f}°C >= {crisis_temp:.1f}°C") as active:
        start = time.time()
        while active and time.time() - start < max_pause_sec:
            time.sleep(10.0)
            if (cur := get_gpu_temperature()) is None or cur <= target_temp:
                break
    return True


def cool_inference_servers(cool_seconds: int) -> None:
    if cool_seconds > 0:
        with _suspended_llama(f"{cool_seconds}s cooldown"):
            time.sleep(cool_seconds)


def run_once() -> tuple[dict[str, Any], list[dict[str, Any]], int, tuple[str, str]]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sb = get_active_sandbox_info()
    cfg = read_config_info()
    cands, n_evals, legacy = load_candidate_records(sb)
    badge = PoolTelemetry.from_sandbox(sb).badge(
        len(cands), n_evals, bool(sb.get("optimizer_alive")), sb.get("active_op", "")
    )
    jobs = (
        (FIG_FILES[0], lambda d: generate_kg_dashboard_svg(d, sb, cfg)),
        (FIG_FILES[1], lambda d: generate_graph_projection_svg(d, sb, badge)),
        (FIG_FILES[2], lambda d: generate_candidate_score_svg(d, cands, legacy, badge, sb)),
        (FIG_FILES[3], lambda d: generate_candidate_quality_svg(d, cands, legacy, badge, sb)),
        (FIG_FILES[4], lambda d: generate_pareto_svg(d, cands, legacy, badge, sb)),
    )
    for name, job in jobs:
        # Isolate figures: one failing renderer must not leave the others stale.
        try:
            job(ARTIFACT_DIR / name)
        except Exception as exc:  # noqa: BLE001 - surfaced in the log; the watch loop must survive
            plt.close("all")
            print(
                f"[{datetime.now(UTC):%H:%M:%S}] {name} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
    generate_report_markdown(
        ARTIFACT_DIR / "benchmark_mobile_report.md", sb, cands, n_evals, legacy, badge
    )
    return sb, cands, n_evals, badge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=None)
    ap.add_argument("--cool-seconds", type=int, default=0)
    ap.add_argument("--thermal-guard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crisis-temp", type=float, default=82.0)
    ap.add_argument("--target-temp", type=float, default=72.0)
    args = ap.parse_args()

    for name in LEGACY_FIG_FILES:
        with contextlib.suppress(OSError):
            (ARTIFACT_DIR / name).unlink(missing_ok=True)

    while True:
        try:
            sb, cands, n_evals, badge = run_once()
            pt = PoolTelemetry.from_sandbox(sb)
            cand_info = ""
            if cands:
                full = [c for c in cands if c.get("val_score") is not None]
                last = cands[-1]
                best_txt = ""
                if full:
                    best = max(full, key=lambda c: c["f1"])
                    best_txt = f", best full-val F1 #{best['cand']} {best['f1'] * 100:.1f}%"
                cand_info = (
                    f" | Candidates: {len(cands)} ({n_evals} evals){best_txt}, latest "
                    f"#{last['cand']} R@5 {last['r5'] * 100:.0f}% F1 {last['f1'] * 100:.0f}%"
                )
            pct_str = f"{pt.openie_cnt / pt.corpus_pool * 100:.1f}%" if pt.corpus_pool else "n/a"
            print(
                f"[{datetime.now(UTC):%H:%M:%S}] [{badge[0]}] "
                f"Passages: {pt.openie_cnt}/{pt.corpus_pool or 'n/a'} ({pct_str}){cand_info}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - the watch loop must survive transient failures
            print(f"[{datetime.now(UTC):%H:%M:%S}] Error in run_once: {exc}", flush=True)
        if args.thermal_guard:
            with contextlib.suppress(Exception):
                check_thermal_guard(args.crisis_temp, args.target_temp)
        if args.watch is None:
            return 0
        cool_inference_servers(args.cool_seconds)
        time.sleep(max(5, args.watch))


if __name__ == "__main__":
    sys.exit(main())
