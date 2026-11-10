"""Deterministic live dashboard generator for benchmark_mobile_report.md and SVG visualizations."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
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
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from yaw.core.config import write_text_atomic, write_via_temp_atomic  # noqa: E402
from yaw.core.logger import configure_logging, get_logger  # noqa: E402
from yaw.daemon.graph.storage import get_graph_counts, open_readonly_yaw_db  # noqa: E402
from yaw.daemon.runtime import is_file_locked  # noqa: E402

logger = get_logger("yaw.watchdog")


def _resolve_artifact_dir() -> Path:
    if custom := os.environ.get("YAW_ARTIFACT_DIR"):
        return Path(custom)
    try:
        brain = Path.home() / ".gemini" / "antigravity" / "brain"
        if brain.exists():
            reports = sorted(
                brain.glob("*/benchmark_mobile_report.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if reports:
                return reports[0].parent
    except Exception:
        pass
    return WORKSPACE_ROOT / "outputs" / "dashboard"


ARTIFACT_DIR = _resolve_artifact_dir()
# Report figure order (Fig 1..5). Fig. 3 includes evaluation and candidate axes.
FIG_FILES = (
    "knowledge_graph_extraction_dashboard.svg",
    "knowledge_graph_2d_projection.png",
    "candidate_score_progress.svg",
    "candidate_quality_metrics.svg",
    "pareto_frontier_progress.svg",
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

    openie_cnt: int  # OpenIE-extracted chunks in the sandbox DB (chunk_inference)
    train_queries: int | None  # from the engine's <stem>_pools.json manifest
    val_queries: int | None
    gepa_val_queries: int | None
    val_done: int | None  # manifest val chunk ids ∩ chunk_inference
    val_pool: int | None
    train_done: int | None
    train_pool: int | None
    train_val_done: int | None
    train_val_pool: int | None
    background_done: int | None  # corpus chunks not in train or val query gold sets
    background_pool: int | None
    corpus_done: int | None
    corpus_pool: int | None
    elapsed_hours: float  # wall-clock since the sandbox DB was created
    velocity: float | None  # chunks/min over the last measured window (telemetry samples)
    sec_per_chunk: float | None
    val_eta_min: float | None  # only while upfront val indexing is still running
    corpus_eta_min: float | None  # ETA to index remaining corpus chunks

    @classmethod
    def from_sandbox(cls, sandbox: dict[str, Any]) -> PoolTelemetry:
        cnt = int(sandbox.get("counts", {}).get("chunk_inference", 0))
        pools = sandbox.get("pools") or {}
        indexed: set[str] = sandbox.get("indexed_chunk_ids") or set()

        def _split(key: str) -> tuple[int | None, int | None]:
            ids = pools.get(key)
            if not isinstance(ids, list):
                return None, None
            return len(indexed.intersection(ids)), len(ids)

        val_done, val_pool = _split("val_chunk_ids")
        train_done, train_pool = _split("train_chunk_ids")
        val_ids = pools.get("val_chunk_ids")
        train_ids = pools.get("train_chunk_ids")
        if isinstance(val_ids, list) and isinstance(train_ids, list) and (val_ids or train_ids):
            train_val_union = set(val_ids) | set(train_ids)
            train_val_done, train_val_pool = len(indexed.intersection(train_val_union)), len(train_val_union)
        else:
            train_val_done, train_val_pool = None, None

        corpus_ids = pools.get("corpus_chunk_ids")
        if isinstance(corpus_ids, list) and corpus_ids:
            corpus_done, corpus_pool = len(indexed.intersection(set(corpus_ids))), len(corpus_ids)
        elif train_val_pool is not None:
            corpus_done, corpus_pool = train_val_done, train_val_pool
        else:
            corpus_done, corpus_pool = _split("corpus_chunk_ids")
        if corpus_done is None:
            corpus_done = cnt

        background_done = None
        background_pool = None
        if corpus_pool is not None and train_val_pool is not None:
            background_pool = max(0, corpus_pool - train_val_pool)
            background_done = max(0, cnt - (train_val_done or 0))

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
        corpus_eta = None
        if vel and corpus_pool is not None and cnt < corpus_pool:
            corpus_eta = (corpus_pool - cnt) / vel

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
            train_val_done=train_val_done,
            train_val_pool=train_val_pool,
            background_done=background_done,
            background_pool=background_pool,
            corpus_done=corpus_done,
            corpus_pool=corpus_pool,
            elapsed_hours=el_sec / 3600.0,
            velocity=vel,
            sec_per_chunk=(60.0 / vel) if vel else None,
            val_eta_min=val_eta,
            corpus_eta_min=corpus_eta,
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

    def badge(self, n_cands: int, n_evals: int, active: bool, active_op: str = "") -> tuple[str, str]:
        op_suffix = f" • {active_op}" if active_op and active else ""
        if n_cands:
            state = "OPTIMIZER RUNNING" if active else "OPTIMIZER STOPPED"
            col = "#10b981" if active else "#f59e0b"
            return f"{state}: {n_cands} CANDIDATES ({n_evals} EVALS){op_suffix}"[:96], col
        if active:
            return f"OPTIMIZER RUNNING{op_suffix}"[:96], "#10b981"
        return f"OPTIMIZER STOPPED • {self.openie_cnt} CHUNKS INDEXED", "#f59e0b"


def get_lemonade_server_metrics() -> dict[str, Any]:
    try:
        req = urllib.request.Request("http://localhost:13305/metrics", headers={"Authorization": "Bearer lemonade"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            m = {
                p[0]: float(p[1])
                for line in resp.read().decode("utf-8", errors="ignore").splitlines()
                if not line.startswith("#") and len(p := line.split()) == 2
            }
        tps = next(
            (v for pref in ("Gemma", "Qwen3-4B", "") for k, v in m.items() if pref in k and "tokens_per_second" in k),
            0.0,
        )
        return {"server_up": True, "llm_tps": round(tps, 1)}
    except Exception:
        return {"server_up": False}


def get_pipeline_resource_metrics() -> dict[str, Any]:
    rss = 0
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        with contextlib.suppress(psutil.Error, OSError):
            n, cmd = (p.info["name"] or "").lower(), " ".join(p.info["cmdline"] or []).lower()
            if any(k in n or k in cmd for k in ("lemonade", "llama-server", "yaw")):
                rss += p.memory_info().rss
    gpu = _nvidia_smi("memory.used,memory.total")
    vram_used, vram_tot = gpu[:2] if len(gpu) >= 2 else (None, None)
    return {
        "pipeline_ram_gb": round(rss / 1024**3, 2),
        "pipeline_vram_gb": round(vram_used / 1024.0, 2) if vram_used is not None else None,
        "total_vram_gb": round(vram_tot / 1024.0, 2) if vram_tot is not None else None,
    }


def get_empirical_telemetry_timeseries(
    start_dt: datetime,
    current_elapsed_sec: float,
    current_passages: int,
    current_entities: int,
    current_triples: int,
    db_stem: str = "",
) -> list[dict[str, Any]]:
    suffix = db_stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
    if suffix == "canonical":
        suffix = ""
    t_file = WORKSPACE_ROOT / ".yaw" / (f"optimize_telemetry_{suffix}.json" if suffix else "optimize_telemetry.json")
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
        write_text_atomic(t_file, json.dumps(deduped, indent=2))

    # Interval rate/latency between consecutive real samples only; no seeded defaults.
    samples: list[dict[str, Any]] = []
    prev: tuple[float, int, int] | None = None
    for i, cp in enumerate(deduped):
        t_s, p_c, e_c = float(cp["time_sec"]), int(cp["passages"]), int(cp.get("entities", 0))
        vel = lat = None
        if prev is not None and t_s - prev[0] > 10:
            dt, dp = t_s - prev[0], max(0, p_c - prev[1])
            vel = round(dp / dt * 60.0, 2)
            lat = round(dt / dp, 1) if dp else None
        if prev is not None and p_c - prev[1] > 0:
            epc: float | None = round(max(0, e_c - prev[2]) / (p_c - prev[1]), 1)
        elif p_c > 0:
            epc = round(e_c / p_c, 1)
        else:
            epc = None
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
                "entities_per_chunk": epc,
            }
        )
        prev = (t_s, p_c, e_c)
    return samples


def _open_ro_sqlite(db_path: Path, backoff: tuple[float, ...] = (1.0, 2.0, 3.0)) -> sqlite3.Connection | None:
    with contextlib.suppress(sqlite3.Error):
        return open_readonly_yaw_db(db_path, backoff=backoff)
    return None


def _read_status(status_file: Path) -> dict[str, Any]:
    with contextlib.suppress(OSError, ValueError):
        obj = json.loads(status_file.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("status"), dict):
            return obj["status"]
    return {}


def get_active_sandbox_info() -> dict[str, Any]:
    yaw_dir = WORKSPACE_ROOT / ".yaw"
    seen_paths: set[Path] = set()
    candidates: list[Path] = []
    if yaw_dir.exists():
        for p in [yaw_dir / "optimize_yaw.db", *yaw_dir.glob("optimize_yaw_*.db")]:
            if p.is_file() and (rp := p.resolve()) not in seen_paths:
                seen_paths.add(rp)
                candidates.append(p)
    dbs = sorted(
        candidates,
        key=lambda p: (is_file_locked(Path(f"{p}.lock")), p.stat().st_mtime, p.name),
        reverse=True,
    )
    if not dbs:
        return {}
    latest_db, st = dbs[0], dbs[0].stat()
    sz = st.st_size
    suffix = latest_db.stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
    run_tag = suffix or "canonical"
    status = _read_status(yaw_dir / (f"optimize_status_{suffix}.json" if suffix else "optimize_status.json"))
    active_op = str(status.get("active_op") or "")
    optimizer_alive = is_file_locked(Path(f"{latest_db}.lock")) and status.get("lifecycle") not in (
        "stopped",
        "failed",
    )

    pools_obj = _read_audit(_pools_path(run_tag))
    start_dt = datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime), tz=UTC)
    conn = _open_ro_sqlite(latest_db)
    if conn is not None:
        with contextlib.suppress(sqlite3.Error, ValueError, TypeError):
            row = conn.execute("SELECT value FROM meta WHERE key = 'optimization_run'").fetchone()
            if row and row[0]:
                meta_run = json.loads(row[0])
                if isinstance(meta_run, dict) and meta_run.get("started_at"):
                    parsed_dt = datetime.fromisoformat(str(meta_run["started_at"]))
                    start_dt = parsed_dt if parsed_dt.tzinfo else parsed_dt.replace(tzinfo=UTC)

    info: dict[str, Any] = {
        "db_name": latest_db.name,
        "db_path": str(latest_db),
        "datestamp": run_tag,
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

    if not conn:
        return info

    try:
        counts: dict[str, int] = {}
        with contextlib.suppress(sqlite3.Error):
            counts = dict(
                zip(
                    ("chunks", "entities", "facts", "edges", "chunk_inference"),
                    get_graph_counts(conn),
                    strict=True,
                )
            )
        diff_from_rows: list[tuple[str, str, str, str, str, str]] = []
        related_assertions: list[tuple[str, str, str, str, str, str]] = []
        broader_assertions: list[tuple[str, str, str, str, str, str]] = []
        entity_details: dict[str, tuple[str, str, int]] = {}
        graph_edges: list[tuple[str, str, str]] = []
        alt_label_cnt = 0
        largest_label_info: dict[str, Any] = {}
        with contextlib.suppress(sqlite3.Error):
            diff_from_rows = recent_relation_assertions(conn, "owl:differentFrom", limit=100000)
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
                    "SELECT entity_id, COUNT(DISTINCT label_norm) FROM entity_labels GROUP BY entity_id"
                ).fetchall()
            )
            entity_details = {
                str(r[0]): (str(r[1]), str(r[2]), max(1, int(lbl_counts.get(r[0], 1))))
                for r in conn.execute("SELECT entity_id, name, COALESCE(definition, '') FROM entities").fetchall()
            }
            largest_label_row = conn.execute(
                """
                SELECT e.entity_id, e.name, COUNT(DISTINCT el.label_norm) as lbl_cnt
                FROM entities e
                JOIN entity_labels el ON el.entity_id = e.entity_id
                GROUP BY e.entity_id, e.name
                ORDER BY lbl_cnt DESC, e.entity_id ASC
                LIMIT 1
                """
            ).fetchone()
            if largest_label_row:
                top_eid, top_name, top_cnt = (
                    str(largest_label_row[0]),
                    str(largest_label_row[1]),
                    int(largest_label_row[2]),
                )
                labels_for_node = [
                    (str(r[0]), str(r[1]))
                    for r in conn.execute(
                        "SELECT label_kind, raw_label FROM entity_labels WHERE entity_id = ? "
                        "ORDER BY label_kind ASC, raw_label ASC",
                        (top_eid,),
                    ).fetchall()
                ]
                largest_label_info = {
                    "entity_id": top_eid,
                    "name": top_name,
                    "count": top_cnt,
                    "labels": labels_for_node,
                }
            graph_edges = [
                (str(r[0]), str(r[1]), str(r[2]))
                for r in conn.execute(
                    """
                    SELECT src_id, dst_id, kind
                    FROM edges
                    WHERE kind IN ('fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
                      AND (kind IN ('fact', 'skos:broader') OR src_id < dst_id)
                    """
                ).fetchall()
            ]
            related_assertions = recent_relation_assertions(conn, "skos:related")
            broader_assertions = recent_relation_assertions(conn, "skos:broader", limit=5)

        indexed_ids: set[str] = set()
        cum_e = cum_t = 0
        with contextlib.suppress(sqlite3.Error):
            for cid, ext_raw in conn.execute("SELECT chunk_id, extraction_json FROM chunk_inference"):
                indexed_ids.add(str(cid))
                ext = json.loads(ext_raw or "{}")
                cum_e += len(ext.get("entities") or ())
                cum_t += len(ext.get("triples") or ())

        info["counts"] = counts
        info["indexed_chunk_ids"] = indexed_ids
        info["kg_metrics"] = {
            "total_entities": cum_e,
            "total_triples": cum_t,
            "different_from_pairs": len(diff_from_rows),
            "related_pairs": sum(
                1 for u, v, k in graph_edges if k == "skos:related" and u in entity_details and v in entity_details
            ),
            "broader_pairs": sum(
                1 for u, v, k in graph_edges if k == "skos:broader" and u in entity_details and v in entity_details
            ),
            "synonym_merges": alt_label_cnt,
        }
        info["kg_graph"] = {
            "entities": entity_details,
            "edges": graph_edges,
            "diff_from_rows": diff_from_rows,
            "related_assertion_rows": related_assertions,
            "broader_assertion_rows": broader_assertions,
            "largest_label_node": largest_label_info,
        }
        info["timeseries_samples"] = get_empirical_telemetry_timeseries(
            start_dt, info["elapsed_seconds"], len(indexed_ids), cum_e, cum_t, suffix
        )
        _CACHE.clear()
        _CACHE[str(latest_db)] = {
            k: info[k]
            for k in (
                "counts",
                "indexed_chunk_ids",
                "kg_metrics",
                "kg_graph",
                "timeseries_samples",
            )
        }
    except Exception as exc:
        info["error"] = str(exc)
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.close()
    return info


_AUDIT_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_CAND_METRICS = ("r1", "r2", "r5", "r10", "r20", "em", "f1")


def _audit_path(tag: str) -> Path:
    name = "optimize_program_audit.json" if tag in {"", "canonical"} else f"optimize_program_{tag}_audit.json"
    return WORKSPACE_ROOT / ".yaw" / name


def _pools_path(tag: str) -> Path:
    """Split/pool manifest the engine writes as soon as the optimization pools are built."""
    name = "optimize_program_pools.json" if tag in {"", "canonical"} else f"optimize_program_{tag}_pools.json"
    return WORKSPACE_ROOT / ".yaw" / name


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


def load_candidate_records(sandbox: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Return (per-candidate rows, total evals) for the ACTIVE run only."""
    ds = sandbox.get("datestamp")
    if not ds:
        return [], 0
    obj = _read_audit(_audit_path(str(ds)))
    evals = [r for r in (obj.get("eval_records") or []) if isinstance(r, dict)]
    raw = [r for r in (obj.get("candidate_records") or []) if isinstance(r, dict)]
    n_evals = sum(int(r.get("n", 1)) for r in raw)
    evaluations_by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, evaluation in enumerate(evals):
        candidate = evaluation.get("cand")
        if candidate is None:
            continue
        evaluations_by_candidate[int(candidate)].append(
            {
                "eval": int(evaluation.get("eval", i + 1)),
                "score": float(evaluation.get("score", 0.0)),
            }
        )

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
            "evaluations": evaluations_by_candidate[int(r.get("cand", len(rows) + 1))],
            **{k: float(r.get(k, 0.0)) for k in _CAND_METRICS},
        }
        # Valset score (when GEPA ran one) is the comparable number; minibatch means are noisy.
        row["eff_score"] = row["val_score"] if row["val_score"] is not None else sc
        rows.append(row)
    best = 0.0
    for row in rows:
        best = max(best, row["eff_score"])
        row["best_so_far"] = best
    return rows, n_evals


def _save(fig: plt.Figure, *paths: Path) -> None:
    try:
        for p in paths:
            if p.suffix == ".png":
                write_via_temp_atomic(p, lambda t: fig.savefig(t, format="png", dpi=110))
            else:
                write_via_temp_atomic(p, fig.savefig)
    finally:
        plt.close(fig)


def generate_kg_dashboard_svg(dest: Path, sandbox: dict[str, Any]) -> None:
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
    tps = f"{lem['llm_tps']:.0f} tok/s" if lem.get("server_up") else "LLM metrics n/a"
    eta_str = pt.duration(pt.corpus_eta_min if pt.corpus_eta_min is not None else pt.val_eta_min)
    fig.text(
        0.04,
        0.91,
        f"Corpus: {pt.frac(pt.openie_cnt, pt.corpus_pool)} | "
        f"Train: {pt.frac(pt.train_done, pt.train_pool)} | "
        f"Val: {pt.frac(pt.val_done, pt.val_pool)} | "
        f"Rate (20 min): {rate} | {tps} | "
        f"ETA: {eta_str}",
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
    if pt.val_eta_min is not None and pt.val_pool is not None and pt.val_done is not None and pt.val_done < pt.val_pool:
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
    elif pt.corpus_eta_min is not None and pt.corpus_pool is not None and pt.openie_cnt < pt.corpus_pool:
        proj_end = pt.elapsed_hours + pt.corpus_eta_min / 60.0
        ax1.plot(
            [pt.elapsed_hours, proj_end],
            [pt.openie_cnt, pt.corpus_pool],
            color="#38bdf8",
            lw=1.0,
            ls="--",
            label="Corpus projection",
        )
    y_top = max(10.0, float(pt.openie_cnt))
    seen_pools: set[int] = set()
    for pool, col, lbl in (
        (pt.val_pool, "#10b981", "Val pool"),
        (pt.train_val_pool, "#f59e0b", "Train + val pool"),
        (pt.corpus_pool, "#64748b", "Full corpus"),
    ):
        if pool and pool not in seen_pools:
            seen_pools.add(pool)
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

    rl = [(float(s["time_hours"]), s.get("rolling_latency")) for s in samples]
    rl = [(t, v) for t, v in rl if v is not None]
    epc = [(float(s["time_hours"]), s.get("entities_per_chunk")) for s in samples]
    epc = [(t, v) for t, v in epc if v is not None]
    if rl:
        ax2.plot(
            *zip(*rl, strict=True),
            color="#10b981",
            lw=1.0,
            marker="o",
            ms=2.5,
            label="Throughput (s/chunk)",
        )
        ax2.set_ylim(0.0, max(1.0, max(v for _, v in rl) * 1.25))
    if epc:
        ax2_r.plot(
            *zip(*epc, strict=True),
            color="#fbbf24",
            lw=1.0,
            ls="--",
            marker="o",
            ms=2.5,
            label="Extraction (entities/chunk)",
        )
        ax2_r.set_ylim(0.0, max(1.0, max(v for _, v in epc) * 1.25))
    if not rl and not epc:
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
    ax2.set_ylabel("Throughput (s/chunk)", color="#10b981", fontweight="bold")
    ax2_r.set_ylabel("Entities / chunk", color="#fbbf24", fontweight="bold")
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


def largest_one_hop_neighborhood(edges: list[tuple[str, str, str]]) -> set[str]:
    """Largest induced component one hop from semantic/taxonomic endpoints."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    seeds: set[str] = set()
    for u, v, kind in edges:
        if kind not in {"fact", "skos:related", "owl:differentFrom", "skos:broader"} or u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
        if kind in {"skos:related", "owl:differentFrom", "skos:broader"}:
            seeds.update((u, v))
    selected = seeds | {node for seed in seeds for node in adjacency[seed]}
    remaining = set(selected)
    largest: set[str] = set()
    # Sorted starts give deterministic tie-breaking for equally large components.
    for start in sorted(selected):
        if start not in remaining:
            continue
        component: set[str] = set()
        pending = [start]
        remaining.remove(start)
        while pending:
            node = pending.pop()
            component.add(node)
            neighbors = adjacency[node] & remaining
            remaining.difference_update(neighbors)
            pending.extend(neighbors)
        if len(component) > len(largest):
            largest = component
    return largest


def recent_relation_assertions(
    conn: sqlite3.Connection, relation: str, limit: int = 10
) -> list[tuple[str, str, str, str, str, str]]:
    """Newest active canonical pairs by insertion rowid; reassertion times aren't stored."""
    order_filter = "AND e.src_id < e.dst_id" if relation != "skos:broader" else ""
    return [
        (str(left_id), str(left_name), str(left_def), str(right_id), str(right_name), str(right_def))
        for left_id, left_name, left_def, right_id, right_name, right_def in conn.execute(
            "SELECT a.entity_id, a.name, COALESCE(a.definition, ''), "
            "b.entity_id, b.name, COALESCE(b.definition, '') "
            "FROM edges e JOIN entities a ON a.entity_id = e.src_id "
            "JOIN entities b ON b.entity_id = e.dst_id "
            f"WHERE e.kind = ? {order_filter} ORDER BY e.rowid DESC LIMIT ?",
            (relation, limit),
        )
    ]


def generate_graph_projection_svg(
    dest: Path, sandbox: dict[str, Any], badge: tuple[str, str], *, layout_iterations: int = 500
) -> dict[str, Any]:
    if layout_iterations < 1:
        raise ValueError("layout_iterations must be positive")
    kg_g = sandbox.get("kg_graph") or {}
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

    selected = largest_one_hop_neighborhood(all_edges)
    # Counts and colors describe only the connected neighborhood actually shown.
    all_edges_tail = [(u, v, k) for u, v, k in all_edges if u in selected and v in selected][-1000:]
    if len(all_edges_tail) == 1000:
        selected = {u for u, v, _ in all_edges_tail} | {v for u, v, _ in all_edges_tail}
    diff_nodes: set[str] = {u for u, v, k in all_edges_tail if k == "owl:differentFrom"} | {
        v for u, v, k in all_edges_tail if k == "owl:differentFrom"
    }
    rel_nodes: set[str] = {u for u, v, k in all_edges_tail if k == "skos:related"} | {
        v for u, v, k in all_edges_tail if k == "skos:related"
    }
    broader_nodes: set[str] = {v for u, v, k in all_edges_tail if k == "skos:broader"}
    base_nodes = diff_nodes | rel_nodes | broader_nodes

    # Include all 1-hop connected rdf:Statement (fact) entities
    inc_fact: dict[str, set[str]] = defaultdict(set)
    for u, v, k in all_edges_tail:
        if k == "fact":
            if u in base_nodes:
                inc_fact[u].add(v)
            if v in base_nodes:
                inc_fact[v].add(u)

    nodes = sorted(
        selected,
        key=lambda x: (
            x in diff_nodes,
            x in broader_nodes,
            x in rel_nodes,
            len(inc_fact[x]),
        ),
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
    diff_segs = [[idx[u], idx[v]] for u, v, k in all_edges_tail if k == "owl:differentFrom" and u in idx and v in idx]
    rel_segs = [[idx[u], idx[v]] for u, v, k in all_edges_tail if k == "skos:related" and u in idx and v in idx]
    fact_segs = [[idx[u], idx[v]] for u, v, k in all_edges_tail if k == "fact" and u in idx and v in idx]
    broader_segs = [[idx[u], idx[v]] for u, v, k in all_edges_tail if k == "skos:broader" and u in idx and v in idx]
    active_edge_arr = (
        np.array(diff_segs + rel_segs + fact_segs + broader_segs, dtype=int)
        if (diff_segs or rel_segs or fact_segs or broader_segs)
        else np.zeros((0, 2), dtype=int)
    )

    n_cnt = len(nodes)
    rng = np.random.default_rng(42)

    if n_cnt <= 250:
        ang = np.linspace(0, 2 * math.pi, n_cnt, endpoint=False)
        rad = 220.0 + rng.standard_normal(n_cnt) * 30.0
        pos = np.column_stack([np.cos(ang) * rad, np.sin(ang) * rad])
        vel = np.zeros((n_cnt, 2))
        for _ in range(layout_iterations):
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
        for _ in range(layout_iterations):
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
        0.91,
        f"{n_cnt} entities • Largest 1-hop component • "
        f"{len(diff_nodes)} ≠ endpoints • {len(broader_nodes)} broader concepts\n"
        f"Edges: {len(fact_segs)} fact • {len(rel_segs)} skos:related • "
        f"{len(broader_segs)} skos:broader • {len(diff_segs)} owl:differentFrom",
        fontsize=9,
        va="top",
        color="#38bdf8",
        fontweight="bold",
    )

    for segs, col, lw, alpha, z in (
        (fact_segs, "#475569", 1, 0.8, 1),
        (rel_segs, "#c084fc", 1, 0.8, 2),
        (broader_segs, "#eab308", 1.2, 0.85, 2),
        (diff_segs, "#f43f5e", 1, 0.8, 3),
    ):
        if segs:
            ax.add_collection(
                LineCollection(
                    [[pos[u], pos[v]] for u, v in segs],
                    colors=col,
                    linewidths=lw,
                    linestyles="solid",
                    alpha=alpha,
                    zorder=z,
                )
            )

    # Compute node cardinality (degree of active edges in the view)
    deg_map: dict[str, int] = defaultdict(int)
    for u, v, _k in all_edges_tail:
        if u in idx and v in idx:
            deg_map[u] += 1
            deg_map[v] += 1

    deg_vals = [deg_map.get(eid, 1) for eid in nodes]
    min_deg = min(deg_vals) if deg_vals else 1
    max_deg = max(deg_vals) if deg_vals else 1
    span_deg = max(1.0, float(max_deg - min_deg))

    # Cardinality colormap: violet -> cyan -> lime (yellowish green)
    cardinality_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cardinality_vcl",
        ["#8b5cf6", "#06b6d4", "#84cc16", "#a3e635"],
        N=256,
    )

    # Node Sizing: strictly based on skos:*Label count
    sizes, colors, edgecolors, lws = [], [], [], []
    base_sz = 6.0 if n_cnt > 2000 else (8.0 if n_cnt > 500 else 10.0)
    for eid in nodes:
        name, defn, skos_cnt = entities.get(eid, (eid, "", 1))
        # Base size scaled with skos:*Label count (synonyms & lexical variants)
        s = min(300.0, base_sz + 12.0 * (skos_cnt - 1))
        sizes.append(s)

        deg = deg_map.get(eid, 1)
        t_card = math.log1p(deg - min_deg) / math.log1p(span_deg) if span_deg > 0 else 0.0
        card_rgba = cardinality_cmap(np.clip(t_card, 0.0, 1.0))
        card_hex = matplotlib.colors.to_hex(card_rgba)

        if eid in diff_nodes:
            colors.append("#f43f5e")  # rose/crimson for disambiguation endpoints
            edgecolors.append("#ffffff")
            lws.append(0.6)
        elif eid in broader_nodes:
            colors.append("#fef08a")  # light warm shade of yellow for broader nodes
            edgecolors.append("#eab308")  # warm golden amber edge
            lws.append(0.8)
        else:
            colors.append(card_hex)  # change in hue for change in cardinality (violet -> cyan -> lime)
            edgecolors.append("#0f172a")
            lws.append(0.5)

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=sizes,
        c=colors,
        edgecolors=edgecolors,
        linewidths=lws,
        zorder=4,
    )

    # Entity Name Labels:
    # Show entity names exclusively for both ends of owl:differentFrom disambiguation;
    # other entity names are suppressed per user request.
    label_nodes: set[str] = set()
    label_pairs = 0
    for ea, _na, _da, eb, _nb, _db in diff_from_rows:
        if ea in idx and eb in idx:
            label_nodes.update((ea, eb))
            label_pairs += 1
            if label_pairs == 10:
                break
    for eid in sorted(label_nodes):
        if eid not in idx:
            continue
        i = idx[eid]
        disp_label = entities.get(eid, (eid, "", 1))[0]
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
        *(
            Line2D([0], [0], color=c, lw=lw, ls="-", label=lbl)
            for c, lw, lbl in (
                ("#f43f5e", 1.0, f"owl:differentFrom ({len(diff_segs)} active \u2260)"),
                ("#eab308", 1.2, f"skos:broader ({len(broader_segs)} hierarchy)"),
                ("#c084fc", 1.0, f"skos:related ({len(rel_segs)} links)"),
                ("#475569", 1.0, f"Fact relationships ({len(fact_segs)} edges)"),
            )
        ),
        *(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=fc, markeredgecolor=ec, markersize=ms, label=lbl)
            for fc, ec, ms, lbl in (
                ("#fef08a", "#eab308", 7, f"skos:broader concept ({len(broader_nodes)} warm yellow)"),
                ("#8b5cf6", "#a3e635", 6, "Cardinality: violet (low) \u2192 lime (high)"),
                ("#f43f5e", "w", 8, "owl:differentFrom assertions"),
                ("#818cf8", "#0f172a", 6, "Node size ~ skos:*Label count"),
            )
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
    if dest.suffix == ".png":
        from PIL import Image

        tag = str(sandbox.get("datestamp") or "")
        if sandbox.get("frame_dir"):
            frame_dir = Path(sandbox["frame_dir"])
        elif tag in {"", "canonical"}:
            frame_dir = WORKSPACE_ROOT / ".yaw" / "optimize_kg_frames"
        else:
            frame_dir = WORKSPACE_ROOT / ".yaw" / f"optimize_kg_frames_{tag}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        state_key = (
            n_cnt,
            len(fact_segs),
            len(rel_segs),
            len(broader_segs),
            len(diff_segs),
            int(sandbox.get("counts", {}).get("chunk_inference", 0)),
        )
        frame_name = (
            f"frame_{state_key[5]:05d}_{state_key[0]:05d}_{state_key[1]:05d}_"
            f"{state_key[2]:05d}_{state_key[3]:05d}_{state_key[4]:05d}.png"
        )
        frame_path = frame_dir / frame_name
        try:
            if not frame_path.exists():
                write_via_temp_atomic(frame_path, lambda t: fig.savefig(t, format="png", dpi=110))
            for stale_f in frame_dir.glob(f"frame_{state_key[5]:05d}_*.png"):
                if stale_f != frame_path:
                    with contextlib.suppress(OSError):
                        stale_f.unlink(missing_ok=True)
        finally:
            plt.close(fig)
        frame_files = sorted(frame_dir.glob("frame_*.png"))
        if len(frame_files) > 24:
            for old_f in frame_files[:-24]:
                with contextlib.suppress(OSError):
                    old_f.unlink(missing_ok=True)
            frame_files = frame_files[-24:]
        frames: list[Image.Image] = []
        for fp in frame_files:
            with Image.open(fp) as im:
                frames.append(im.convert("RGBA"))
        if len(frames) >= 2:
            durations = [450] * (len(frames) - 1) + [1800]
            write_via_temp_atomic(
                dest,
                lambda t: frames[0].save(
                    t,
                    format="PNG",
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=0,
                ),
            )
        elif frames:
            write_via_temp_atomic(dest, lambda t: frames[0].save(t, format="PNG"))
    else:
        _save(fig, dest)
    return {
        "nodes_rendered": n_cnt,
        "diff_nodes": len(diff_nodes),
        "active_edges": len(active_edge_arr),
        "diff_edges": len(diff_segs),
        "related_edges": len(rel_segs),
        "broader_edges": len(broader_segs),
        "fact_edges": len(fact_segs),
        "layout_iterations": layout_iterations,
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


def _create_dark_figure(
    figsize: tuple[float, float],
    title: str,
    badge: tuple[str, str],
    *,
    nrows: int = 2,
    sharex: bool = False,
    gridspec_kw: dict[str, Any] | None = None,
    adjust_kw: dict[str, float] | None = None,
    header_y: float = 0.96,
) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        sharex=sharex,
        gridspec_kw=gridspec_kw,
    )
    if adjust_kw:
        fig.subplots_adjust(**adjust_kw)
    _header(fig, title, badge, y=header_y)
    return fig, axes


def _candidate_xaxis(ax: plt.Axes, cands: list[dict[str, Any]]) -> None:
    """Integer candidate axis with fixed padding so 1..N candidates never autoscale oddly."""
    hi = max(int(c["cand"]) for c in cands)
    ax.set_xlim(0.4, hi + 0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1) if hi < 14 else ticker.MaxNLocator(integer=True, nbins=14))
    ax.set_xlabel("Candidate # (order of first evaluation) →", fontweight="bold")


def _no_candidates_message(sandbox: dict[str, Any]) -> str:
    if not sandbox.get("datestamp"):
        return "No optimization sandbox found in .yaw/"
    pt = PoolTelemetry.from_sandbox(sandbox)
    op = sandbox.get("active_op") or "in progress"
    db = sandbox.get("db_name")
    if not sandbox.get("optimizer_alive"):
        return f"No candidate evaluations recorded for {db}\n(optimizer stopped)"
    prog = f"Corpus Indexing: {pt.frac(pt.openie_cnt, pt.corpus_pool)} (Val: {pt.frac(pt.val_done, pt.val_pool)})"
    return f"{prog} • Candidate evolution begins once indexed\nActive: {op}"


def generate_candidate_score_svg(
    dest: Path,
    cands: list[dict[str, Any]],
    badge: tuple[str, str],
    sandbox: dict[str, Any],
) -> None:
    title = "Fig. 3: Individual Evaluations, Candidate Means & Cost"
    if not cands:
        _placeholder_svg(dest, title, badge, _no_candidates_message(sandbox))
        return
    fig, (ax_eval, ax1, ax2) = _create_dark_figure(
        (9.4, 9.4),
        title,
        badge,
        nrows=3,
        gridspec_kw={"height_ratios": [1.35, 1.45, 1.0], "hspace": 0.48},
        adjust_kw={"top": 0.88, "bottom": 0.08, "left": 0.09, "right": 0.97},
    )

    evaluations = sorted(
        [(e["eval"], e["score"], c["cand"], c["score"]) for c in cands for e in c.get("evaluations", [])],
        key=lambda e: e[0],
    )
    if evaluations:
        eval_x = [e[0] for e in evaluations]
        eval_scores = [e[1] * 100.0 for e in evaluations]
        ax_eval.plot(
            eval_x, eval_scores, "o-", color="#a78bfa", ms=3, lw=0.6, alpha=0.75, label="Individual evaluation score"
        )
        # Disconnected segments keep distinct candidates' means visually separate.
        for candidate in cands:
            candidate_evals = sorted(candidate.get("evaluations", []), key=lambda e: e["eval"])
            if candidate_evals:
                ax_eval.plot(
                    [e["eval"] for e in candidate_evals],
                    [candidate["score"] * 100.0] * len(candidate_evals),
                    "_--",
                    color="#38bdf8",
                    lw=1.2,
                    ms=5,
                    label="Current candidate mean (all its evals)" if not ax_eval.lines[1:] else None,
                )
        latest = evaluations[-1]
        ax_eval.set_title(
            f"Latest: eval #{latest[0]} · candidate #{latest[2]} · score {latest[1]:.1%} · mean {latest[3]:.1%}",
            fontsize=9,
            color="#cbd5e1",
        )
        ax_eval.set_xlim(min(eval_x) - 0.5, max(eval_x) + 0.5)
        ax_eval.xaxis.set_major_locator(
            ticker.MultipleLocator(1)
            if (max(eval_x) - min(eval_x)) < 14
            else ticker.MaxNLocator(integer=True, nbins=14)
        )
        ax_eval.legend(loc="upper left", fontsize=8)
        ax_eval.set_xlabel(f"Evaluation # → (recent {len(evaluations)} recorded; older points may be evicted)")
    else:
        ax_eval.text(
            0.5,
            0.5,
            "Individual evaluation scores unavailable in this audit",
            transform=ax_eval.transAxes,
            ha="center",
            color="#cbd5e1",
        )
        ax_eval.set_xlabel("Evaluation # →")
        ax_eval.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_eval.set_ylim(-3, 105)
    ax_eval.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax_eval.set_ylabel("Score (%)")

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
    _candidate_xaxis(ax1, cands)

    lat = np.array([c["s_per_it"] for c in cands])
    ax2.bar(x, lat, width=0.6, color="#f43f5e", alpha=0.75, label="Mean latency per eval")
    ax2.set_ylim(0.0, max(1.0, float(lat.max()) * 1.3))
    if len(cands) <= 40:
        for c, li in zip(cands, lat, strict=True):
            ax2.text(c["cand"], li, f"n={c['n']}", ha="center", va="bottom", fontsize=7, color="#cbd5e1")
    ax2.set_ylabel("Latency (s / eval)", color="#f43f5e", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    _candidate_xaxis(ax2, cands)
    _save(fig, dest)


def generate_candidate_quality_svg(
    dest: Path,
    cands: list[dict[str, Any]],
    badge: tuple[str, str],
    sandbox: dict[str, Any],
) -> None:
    title = "Fig. 4: Retrieval Recall@k & QA Quality (EM, F1) per Candidate"
    if not cands:
        _placeholder_svg(dest, title, badge, _no_candidates_message(sandbox))
        return
    fig, (ax1, ax2) = _create_dark_figure(
        (9.4, 6.4),
        title,
        badge,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.22},
        adjust_kw={"top": 0.84, "bottom": 0.10, "left": 0.09, "right": 0.97},
    )
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
    ax2.plot(x, [c["em"] * 100.0 for c in cands], color="#fbbf24", lw=1.0, marker="s", ms=ms, label="EM")
    ax2.set_ylabel("Mean QA Score", color="#f43f5e", fontweight="bold")
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8)
    _candidate_xaxis(ax2, cands)
    _save(fig, dest)


def generate_pareto_svg(
    dest: Path,
    cands: list[dict[str, Any]],
    badge: tuple[str, str],
    sandbox: dict[str, Any],
) -> None:
    title = "Fig. 5: GEPA Validation & Native Frontier Progress"
    if not sandbox:
        _placeholder_svg(dest, title, badge, _no_candidates_message(sandbox))
        return
    ds = sandbox.get("datestamp")
    progress = _read_audit(_audit_path(str(ds))).get("gepa_progress") if ds else None
    if not isinstance(progress, dict) or progress.get("version") != 1:
        _placeholder_svg(
            dest,
            title,
            badge,
            "Native GEPA telemetry unavailable in this audit.\n"
            "Requires an optimizer started with progress telemetry.\n"
            "Candidate averages cannot reconstruct GEPA's frontier.",
        )
        return
    fig, (ax1, ax2) = _create_dark_figure(
        (9.4, 8.4),
        title,
        badge,
        nrows=2,
        gridspec_kw={"hspace": 0.5},
        adjust_kw={"top": 0.77, "bottom": 0.12, "left": 0.09, "right": 0.97},
        header_y=0.95,
    )
    observed = int(progress.get("phase_evaluations", 0))
    expected = progress.get("phase_expected")
    count = f"{observed}/{expected}" if expected is not None else str(observed)
    activity = "RUNNING" if sandbox.get("optimizer_alive") else "STOPPED / last recorded phase"
    fig.text(
        0.09,
        0.84,
        f"{activity}: {progress.get('phase', 'unknown')} · iteration {progress.get('iteration', '?')}\n"
        f"Metric evaluations since phase began: {count} (cache hits may bypass metrics)",
        fontsize=9,
        color="#cbd5e1",
    )
    rows = sorted(progress.get("validation", []), key=lambda r: r["candidate_idx"])
    full = [r for r in rows if r["total"] > 0 and r["evaluated"] == r["total"]]
    partial = [r for r in rows if r not in full]
    if full:
        x = [r["candidate_idx"] for r in full]
        y = [r["score"] * 100 for r in full]
        ax1.plot(x, y, "o", ms=5, color="#38bdf8", label="Full validation score")
        ax1.step(
            x,
            np.maximum.accumulate(y),
            where="post",
            color="#10b981",
            label=f"Best full-validation score ({max(y):.1f}%)",
        )
        seed = next((r for r in full if r["candidate_idx"] == 0), None)
        if seed is not None:
            ax1.axhline(seed["score"] * 100, color="#fbbf24", ls=":", label=f"Seed baseline ({seed['score']:.1%})")
    if partial:
        ax1.scatter(
            [r["candidate_idx"] for r in partial],
            [r["score"] * 100 for r in partial],
            marker="x",
            color="#f59e0b",
            label="Partial validation (not comparable)",
        )
    for row in rows[-20:]:
        ax1.annotate(
            f"{row['evaluated']}/{row['total']}",
            (row["candidate_idx"], row["score"] * 100),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#cbd5e1",
        )
    if rows:
        ax1.legend(loc="upper left", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "Awaiting native validation results", transform=ax1.transAxes, ha="center", color="#cbd5e1")
    ax1.set_ylim(-3, 110)
    ax1.set_ylabel("GEPA validation score (%)")
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax1.set_xlabel("Native GEPA candidate index (seed = 0); labels = validation coverage")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    history = progress.get("frontier_history", [])
    front = progress.get("frontier")
    if front is None:
        ax2.text(
            0.5,
            0.5,
            "Native frontier membership not yet available",
            transform=ax2.transAxes,
            ha="center",
            color="#cbd5e1",
        )
    else:
        for key, marker, color, label in (
            ("members", "s", "#38bdf8", "Retained"),
            ("added", "o", "#10b981", "Added"),
            ("removed", "x", "#f43f5e", "Displaced"),
        ):
            points = [(h["event"], idx) for h in history for idx in h[key]]
            if points:
                ax2.scatter(*zip(*points, strict=True), marker=marker, color=color, s=22, label=label)
        if ax2.collections:
            ax2.legend(loc="upper left", fontsize=8, ncol=3)
        shown = ", ".join(str(i) for i in front[:20]) or "empty"
        suffix = f" … ({len(front)} total)" if len(front) > 20 else ""
        ax2.set_title(f"Current native frontier: {shown}{suffix}", fontsize=9, color="#cbd5e1")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel("Native GEPA candidate index")
    ax2.set_xlabel("Native frontier observation # (latest 250 since process start; first = baseline)")
    _save(fig, dest)


def _truncate_def(text: str, limit: int = 120) -> str:
    s = " ".join(text.strip().split())
    if not s:
        return "n/a"
    if len(s) <= limit:
        return s
    cut = s[: max(1, limit - 1)].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{cut or s[: max(1, limit - 1)]}…"


def generate_report_markdown(
    dest: Path,
    sandbox: dict[str, Any],
    cands: list[dict[str, Any]],
    n_evals: int,
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
            f"Corpus **`{pt.frac(pt.openie_cnt, pt.corpus_pool)}`** "
            f"(Train gold **`{pt.frac(pt.train_done, pt.train_pool)}`** • "
            f"Val gold **`{pt.frac(pt.val_done, pt.val_pool)}`** • "
            f"Background **`{pt.frac(pt.background_done, pt.background_pool)}`**) • "
            f"ETA `{pt.duration(pt.corpus_eta_min if pt.corpus_eta_min is not None else pt.val_eta_min)}`",
        ),
    ]
    if cands:
        last = cands[-1]
        rows.append(
            (
                "Optimizer Candidates",
                f"**`{len(cands)}` candidates** (`{n_evals}` evals) • "
                f"Latest `#{last['cand']}` (n=`{last['n']}`): "
                f"R@5 `{last['r5'] * 100:.0f}%`, F1 `{last['f1'] * 100:.1f}%`",
            )
        )
        # Progress is judged on full-valset F1 / R@5, not the composite score or 10-example
        # minibatch means. Only candidates GEPA scored on the whole valset qualify.
        full = [c for c in cands if c.get("val_score") is not None]
        if full:
            seed = full[0]
            best = max(full, key=lambda c: c["val_score"])
            rows.append(
                (
                    "Full Validation",
                    f"`{len(full)}` fully scored • Best composite score **`#{best['cand']}` "
                    f"`{best['val_score']:.4f}`** "
                    f"({best['val_score'] - seed['val_score']:+.4f} vs seed). "
                    "Candidate F1/recall averages also include minibatch evaluations.",
                )
            )
    rows += [
        (
            "Throughput",
            f"**`{_v(pt.velocity, '{:.2f}')}` chunks/min** (last 20 min, "
            f"`{_v(pt.sec_per_chunk, '{:.1f}')}` s/chunk) • LLM "
            f"**`{_v(lem.get('llm_tps') if lem.get('server_up') else None, '{:.1f}')}` tok/s**",
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
            f"`{kg.get('broader_pairs', 0)}` `skos:broader` • "
            f"`{kg.get('related_pairs', 0)}` `skos:related` • "
            f"`{kg.get('different_from_pairs', 0)}` `owl:differentFrom`",
        ),
    ]
    table = "\n".join(f"| **{k}** | {v} |" for k, v in rows)

    kg_g = sandbox.get("kg_graph", {})
    diff_from_rows = kg_g.get("diff_from_rows", [])
    related_assertions = kg_g.get("related_assertion_rows", [])
    broader_assertions = kg_g.get("broader_assertion_rows", [])
    largest_label_node = kg_g.get("largest_label_node")
    assertions_md = ""
    if diff_from_rows:
        # Mobile-friendly: a narrow bullet list instead of a wide multi-column table.
        items = []
        for _ea, na, da, _eb, nb, db in diff_from_rows[:10]:
            da_s = _truncate_def(da)
            db_s = _truncate_def(db)
            items.append(f"- **`{na}`** ≠ **`{nb}`**\n  - A: *{da_s}*\n  - B: *{db_s}*")
        total_pairs = len(diff_from_rows)
        trunc = f"\n*(Showing 10 of {total_pairs} active disambiguation pairs)*\n" if total_pairs > 10 else "\n"
        assertions_md = (
            "### Graph Assertions for Intervention\n"
            "*If an `owl:differentFrom` assertion below is false, it is an intervention "
            "candidate in `yaw.db`.*\n\n" + "\n".join(items) + f"\n{trunc}\n"
        )
    if broader_assertions:
        broader_items = []
        for _ea, na, da, _eb, nb, db in broader_assertions[:5]:
            da_s = _truncate_def(da)
            db_s = _truncate_def(db)
            broader_items.append(
                f"- **`{na}`** (narrower) → **`{nb}`** (broader)\n  - Narrower: *{da_s}*\n  - Broader: *{db_s}*"
            )
        assertions_md += (
            "### Most Recent `skos:broader` Hierarchy Assertions\n\n"
            "*Top 5 active taxonomy pairs (narrower concept → broader concept), newest insertion first.*\n\n"
            + "\n".join(broader_items)
            + "\n\n"
        )
    if related_assertions:
        related_items = []
        for _ea, na, da, _eb, nb, db in related_assertions:
            related_items.append(
                f"- **`{na}`** ↔ **`{nb}`**\n  - A: *{_truncate_def(da)}*\n  - B: *{_truncate_def(db)}*"
            )
        assertions_md += (
            "### Most Recent `skos:related` Assertions\n\n"
            "*Up to 10 active pairs, newest insertion first. Reassertion times are not stored. "
            "Lists cover the sandbox; Fig. 2 shows its largest seeded neighborhood.*\n\n"
            + "\n".join(related_items)
            + "\n\n"
        )
    if largest_label_node and largest_label_node.get("labels"):
        top_name = largest_label_node["name"]
        top_cnt = largest_label_node["count"]
        by_kind: dict[str, list[str]] = defaultdict(list)
        for k_label, l_label in largest_label_node["labels"]:
            by_kind[k_label].append(l_label)
        label_lines = [
            f"  - `{k_label}`: {', '.join(f'`{raw}`' for raw in sorted(set(vals)))}"
            for k_label in ("skos:prefLabel", "skos:altLabel", "skos:hiddenLabel", *list(by_kind))
            if (vals := by_kind.pop(k_label, None))
        ]
        assertions_md += (
            "### Audit Trail: Largest Concept by SKOS Labels\n\n"
            f"**`{top_name}`** ({top_cnt} total unique labels):\n" + "\n".join(label_lines) + "\n\n"
        )
    # Small image files referenced by path (no inline base64: that made the report multi-MB and
    # unrenderable on mobile). Fig. 2 is a compact PNG; the others are small SVGs.
    fig_titles = [
        "Fig. 1: Extraction & Throughput",
        "Fig. 2: Knowledge Graph & Disambiguation",
        "Fig. 3: Candidate Score",
        "Fig. 4: Retrieval Quality",
        "Fig. 5: GEPA Validation & Frontier",
    ]
    figures = []
    for index, (title, name) in enumerate(zip(fig_titles, FIG_FILES, strict=True)):
        figures.append(f"![{title}](/{art}/{name})")
        if index == 1 and assertions_md:
            figures.append(assertions_md)
    figs = "\n\n".join(figures)
    md = (
        f"# Live Optimization Telemetry (`{sandbox.get('db_name', 'no sandbox')}`)\n\n"
        f"*Updated*: `{datetime.now(UTC):%Y-%m-%d %H:%M:%S UTC}` • *State*: **`{badge[0]}`**\n\n"
        "| Metric | Live Value |\n| :--- | :--- |\n"
        f"{table}\n\n"
        f"{figs}\n"
    )
    write_text_atomic(dest, md)


def _find_workload_processes(*, include_optimizer: bool = False) -> list[psutil.Process]:
    """Discover llama-server and optionally yaw optimizer processes to suspend/resume in top-down order."""
    my_pids = {os.getpid()}
    with contextlib.suppress(psutil.Error, OSError):
        my_pids.add(psutil.Process().parent().pid)

    targets: list[psutil.Process] = []
    seen_pids: set[int] = set()

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.pid in my_pids or p.pid in seen_pids:
            continue
        try:
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or []).lower()
        except psutil.Error, OSError:
            continue

        if "llama-server" in name:
            targets.append(p)
            seen_pids.add(p.pid)
        elif include_optimizer and any(k in name for k in ("yaw", "python")):
            if ("yaw" in cmd and "optimize" in cmd) and "watchdog" not in cmd:
                targets.append(p)
                seen_pids.add(p.pid)
                with contextlib.suppress(psutil.Error, OSError):
                    for child in p.children(recursive=True):
                        if child.pid not in seen_pids and child.pid not in my_pids:
                            targets.append(child)
                            seen_pids.add(child.pid)

    def _proc_depth(proc: psutil.Process) -> int:
        try:
            return len(proc.parents())
        except psutil.Error, OSError:
            return 0

    # Suspend deeper child processes first, resume in reverse
    targets.sort(key=_proc_depth, reverse=True)
    return targets


@contextlib.contextmanager
def _suspended_workload(reason: str, *, include_optimizer: bool = False):
    targets = _find_workload_processes(include_optimizer=include_optimizer)
    if targets:
        logger.info(
            "Suspending %d process(es) [include_optimizer=%s] (%s)...",
            len(targets),
            include_optimizer,
            reason,
        )
        for p in targets:
            with contextlib.suppress(psutil.Error, OSError):
                p.suspend()
    try:
        yield bool(targets)
    finally:
        # Resume in reverse order (parents before children)
        for p in reversed(targets):
            with contextlib.suppress(psutil.Error, OSError):
                p.resume()


def _suspended_llama(reason: str):
    """Backwards-compatible wrapper suspending llama-server inference instances."""
    return _suspended_workload(reason, include_optimizer=False)


def is_within_schedule_window(now_time: dt_time, start_str: str, end_str: str) -> bool:
    """Check if current time falls within [start, end) supporting overnight wrap-around."""
    t_start = dt_time.fromisoformat(start_str)
    t_end = dt_time.fromisoformat(end_str)
    if t_start <= t_end:
        return t_start <= now_time < t_end
    return now_time >= t_start or now_time < t_end


def seconds_until_schedule_end(now_dt: datetime, end_str: str) -> float:
    """Calculate remaining seconds until the schedule window ends."""
    t_end = dt_time.fromisoformat(end_str)
    end_dt = datetime.combine(now_dt.date(), t_end, tzinfo=now_dt.tzinfo)
    if end_dt <= now_dt:
        end_dt += timedelta(days=1)
    return max(0.0, (end_dt - now_dt).total_seconds())


def check_schedule_guard(
    day_suspend_start: str = "07:00",
    day_suspend_end: str = "21:00",
    *,
    poll_interval_sec: float = 30.0,
) -> bool:
    """If current local time is inside the suspend window, pause optimizer and llama-server until window ends."""
    now_local = datetime.now().astimezone()
    if not is_within_schedule_window(now_local.time(), day_suspend_start, day_suspend_end):
        return False

    rem = seconds_until_schedule_end(now_local, day_suspend_end)
    rem_h = rem / 3600.0
    reason = (
        f"Daytime schedule window {day_suspend_start}..{day_suspend_end} "
        f"(resuming in {rem_h:.2f}h at {day_suspend_end})"
    )
    with _suspended_workload(reason, include_optimizer=True) as active:
        while active:
            time.sleep(poll_interval_sec)
            now_check = datetime.now().astimezone()
            if not is_within_schedule_window(now_check.time(), day_suspend_start, day_suspend_end):
                logger.info(
                    "Exiting daytime schedule suspend window at %s (window %s..%s). Resuming workload.",
                    now_check.strftime("%H:%M:%S"),
                    day_suspend_start,
                    day_suspend_end,
                )
                break
    return True


def check_thermal_guard(crisis_temp: float = 82.0, target_temp: float = 72.0, max_pause_sec: int = 300) -> bool:
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


def run_once(
    *,
    layout_iterations: int = 500,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, tuple[str, str]]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sb = get_active_sandbox_info()
    cands, n_evals = load_candidate_records(sb)
    badge = PoolTelemetry.from_sandbox(sb).badge(
        len(cands), n_evals, bool(sb.get("optimizer_alive")), sb.get("active_op", "")
    )
    jobs = (
        (FIG_FILES[0], lambda d: generate_kg_dashboard_svg(d, sb)),
        (
            FIG_FILES[1],
            lambda d: generate_graph_projection_svg(d, sb, badge, layout_iterations=layout_iterations),
        ),
        (FIG_FILES[2], lambda d: generate_candidate_score_svg(d, cands, badge, sb)),
        (FIG_FILES[3], lambda d: generate_candidate_quality_svg(d, cands, badge, sb)),
        (FIG_FILES[4], lambda d: generate_pareto_svg(d, cands, badge, sb)),
    )
    for name, job in jobs:
        # Isolate figures: one failing renderer must not leave the others stale.
        try:
            job(ARTIFACT_DIR / name)
        except Exception as exc:  # noqa: BLE001 - surfaced in the log; the watch loop must survive
            plt.close("all")
            logger.warning("%s failed: %s: %s", name, type(exc).__name__, exc)
    generate_report_markdown(ARTIFACT_DIR / "benchmark_mobile_report.md", sb, cands, n_evals, badge)

    # Mirror artifacts so both the active brain artifact directory and repo dashboard stay in sync
    mirror_dirs: list[Path] = []
    default_dir = WORKSPACE_ROOT / "outputs" / "dashboard"
    if ARTIFACT_DIR.resolve() != default_dir.resolve():
        mirror_dirs.append(default_dir)
    try:
        brain = Path.home() / ".gemini" / "antigravity" / "brain"
        if brain.exists():
            for r in brain.glob("*/benchmark_mobile_report.md"):
                bp = r.parent
                if bp.resolve() != ARTIFACT_DIR.resolve() and bp not in mirror_dirs:
                    mirror_dirs.append(bp)
    except Exception:
        pass

    for md in mirror_dirs:
        with contextlib.suppress(Exception):
            md.mkdir(parents=True, exist_ok=True)
            for fname in FIG_FILES:
                src_f = ARTIFACT_DIR / fname
                if src_f.exists():
                    shutil.copy2(src_f, md / fname)
            rep_src = ARTIFACT_DIR / "benchmark_mobile_report.md"
            if rep_src.exists():
                content = rep_src.read_text(encoding="utf-8")
                old_prefix = f"/{ARTIFACT_DIR.as_posix()}/"
                new_prefix = f"/{md.as_posix()}/"
                if old_prefix != new_prefix:
                    content = content.replace(old_prefix, new_prefix)
                write_text_atomic(md / "benchmark_mobile_report.md", content)

    return sb, cands, n_evals, badge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=None)
    ap.add_argument(
        "--layout-iterations",
        type=int,
        default=500,
        help="Force-layout relaxation rounds for Fig. 2 (default: 500)",
    )
    ap.add_argument("--cool-seconds", type=int, default=0)
    ap.add_argument("--thermal-guard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crisis-temp", type=float, default=82.0)
    ap.add_argument("--target-temp", type=float, default=72.0)
    ap.add_argument(
        "--schedule-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable daytime workload suspension (default: False)",
    )
    ap.add_argument(
        "--day-suspend-start",
        type=str,
        default="07:00",
        help="Local time (HH:MM) to begin suspending optimizer and llama-server (default: 07:00)",
    )
    ap.add_argument(
        "--day-suspend-end",
        type=str,
        default="21:00",
        help="Local time (HH:MM) to resume optimizer and llama-server (default: 21:00)",
    )
    args = ap.parse_args()
    if args.layout_iterations < 1:
        ap.error("--layout-iterations must be positive")
    configure_logging("INFO", log_file_path=WORKSPACE_ROOT / ".yaw" / "yaw.log")

    while True:
        try:
            sb, cands, n_evals, badge = run_once(layout_iterations=args.layout_iterations)
            pt = PoolTelemetry.from_sandbox(sb)
            cand_info = ""
            if cands:
                full = [c for c in cands if c.get("val_score") is not None]
                last = cands[-1]
                best_txt = ""
                if full:
                    best = max(full, key=lambda c: c["val_score"])
                    best_txt = f", best full-val score #{best['cand']} {best['val_score']:.4f}"
                cand_info = (
                    f" | Candidates: {len(cands)} ({n_evals} evals){best_txt}, latest "
                    f"#{last['cand']} R@5 {last['r5'] * 100:.0f}% F1 {last['f1'] * 100:.0f}%"
                )
            pct_str = f"{pt.openie_cnt / pt.corpus_pool * 100:.1f}%" if pt.corpus_pool else "n/a"
            logger.info(
                "[%s] Passages: %d/%s (%s)%s",
                badge[0],
                pt.openie_cnt,
                pt.corpus_pool or "n/a",
                pct_str,
                cand_info,
            )
        except Exception as exc:  # noqa: BLE001 - the watch loop must survive transient failures
            logger.warning("Error in run_once: %s", exc)
        if args.thermal_guard:
            with contextlib.suppress(Exception):
                check_thermal_guard(args.crisis_temp, args.target_temp)
        if args.schedule_guard:
            with contextlib.suppress(Exception):
                check_schedule_guard(args.day_suspend_start, args.day_suspend_end)
        if args.watch is None:
            return 0
        cool_inference_servers(args.cool_seconds)
        time.sleep(max(5, args.watch))


if __name__ == "__main__":
    sys.exit(main())
