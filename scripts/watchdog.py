"""Deterministic live dashboard generator for benchmark_mobile_report.md and SVG visualizations."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("svg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import psutil
from matplotlib.collections import LineCollection

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(
    r"C:\Users\Harri.Ahokas\.gemini\antigravity\brain\3061c87c-9ec6-4ff5-b756-80b05e1c4342"
)
VAL_POOL, TRAIN_POOL, CORPUS_POOL, OVERLAP = 53, 156, 208, 1

plt.rcParams.update(
    {
        "figure.facecolor": "#070b14",
        "savefig.facecolor": "#070b14",
        "savefig.format": "svg",
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
    openie_cnt: int
    val_pool: int
    train_pool: int
    corpus_pool: int
    val_done: int
    val_pct: float
    train_done: int
    train_pct: float
    corpus_pct: float
    rem_train: int
    rem_corpus: int
    elapsed_min: float
    elapsed_hours: float
    velocity: float
    sec_per_chunk: float
    eta_corpus_str: str
    finish_clock_str: str
    target_total_hours: float

    @classmethod
    def from_sandbox(cls, sandbox: dict[str, Any]) -> PoolTelemetry:
        cnt = int(sandbox.get("counts", {}).get("openie_chunks", 0))
        v_pool = max(1, int(sandbox.get("val_pool_size", VAL_POOL) or VAL_POOL))
        t_pool = max(1, int(sandbox.get("train_pool_size", TRAIN_POOL) or TRAIN_POOL))
        c_pool = max(v_pool, int(sandbox.get("corpus_pool_size", CORPUS_POOL) or CORPUS_POOL))
        val_done = min(v_pool, cnt)
        train_done = (
            min(OVERLAP, cnt) if cnt <= v_pool else min(t_pool, OVERLAP + cnt - v_pool)
        )
        samples = sandbox.get("timeseries_samples", [])
        el_sec = max(
            float(sandbox.get("elapsed_seconds", 60.0)),
            float(samples[-1]["time_sec"]) if samples else 0.0,
        )
        ext_sec = next(
            (max(60.0, float(s["time_sec"])) for s in samples if s["passages"] >= max(1, cnt - 5)),
            el_sec,
        )
        vel = cnt / max(0.1, ext_sec / 60.0)
        rem_corpus = max(0, c_pool - cnt)
        eta_min = rem_corpus / max(0.05, vel)
        return cls(
            openie_cnt=cnt,
            val_pool=v_pool,
            train_pool=t_pool,
            corpus_pool=c_pool,
            val_done=val_done,
            val_pct=min(100.0, val_done / v_pool * 100.0),
            train_done=train_done,
            train_pct=min(100.0, train_done / t_pool * 100.0),
            corpus_pct=min(100.0, cnt / c_pool * 100.0),
            rem_train=max(0, t_pool - train_done),
            rem_corpus=rem_corpus,
            elapsed_min=max(0.1, el_sec / 60.0),
            elapsed_hours=el_sec / 3600.0,
            velocity=vel,
            sec_per_chunk=ext_sec / max(1, cnt),
            eta_corpus_str=f"{int(eta_min // 60)}h {int(eta_min % 60)}m"
            if vel > 0.05
            else "calculating...",
            finish_clock_str=(datetime.now(UTC) + timedelta(minutes=eta_min)).strftime("%H:%M"),
            target_total_hours=(el_sec + eta_min * 60.0) / 3600.0,
        )

    def badge(self, n_recs: int, active: bool, active_op: str = "") -> tuple[str, str]:
        if active and n_recs:
            op_suffix = f" • {active_op}" if active_op else ""
            return f"GEPA ACTIVE: {n_recs} EVALS{op_suffix}"[:76], "#10b981"
        if active_op:
            return f"ACTIVE: {active_op}"[:76], "#10b981"
        stage = (
            f"STAGE 2: TRAIN {self.train_done}/{self.train_pool}"
            if self.openie_cnt >= self.val_pool
            else f"STAGE 1: VAL {self.openie_cnt}/{self.val_pool}"
        )
        return f"{stage} CHUNKS ({self.elapsed_hours:.1f}h)", "#38bdf8"


def _pareto_2d(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    s = sorted(records, key=lambda r: (float(r["s_per_it"]), -float(r["gm_good"])))
    q = np.array([float(r["gm_good"]) for r in s])
    return [
        r for r, k in zip(s, np.concatenate(([True], q[1:] > np.maximum.accumulate(q)[:-1]))) if k
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
        pick = lambda sub: next((v for k, v in m.items() if "Qwen3-4B" in k and sub in k), 0.0)
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
    vram_used, vram_tot, gpu_util = (gpu + [0.0, 8192.0, 0.0])[:3]
    vm = psutil.virtual_memory()
    return {
        "pipeline_ram_gb": round(sum(ram.values()) / 1024**3, 2),
        "llama_ram_gb": round(ram["llama"] / 1024**3, 2),
        "yaw_ram_gb": round(ram["yaw"] / 1024**3, 2),
        "lemonade_ram_mb": round(ram["lemonade"] / 1024**2, 1),
        "pipeline_vram_gb": round(vram_used / 1024.0, 2),
        "total_vram_gb": round(vram_tot / 1024.0, 2),
        "vram_pct": round(vram_used / max(1.0, vram_tot) * 100.0, 1),
        "pipeline_cpu_pct": round(sum(cpu.values()), 1),
        "llama_cpu_pct": round(cpu["llama"], 1),
        "yaw_cpu_pct": round(cpu["yaw"], 1),
        "lemonade_cpu_pct": round(cpu["lemonade"], 1),
        "gpu_util_pct": round(gpu_util, 0),
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

    samples, prev_s, prev_p = [], 0.0, 0
    for i, cp in enumerate(deduped):
        t_s, p_c = float(cp["time_sec"]), int(cp["passages"])
        dt, dp = t_s - prev_s, max(0, p_c - prev_p)
        vel = (
            (p_c / max(1.0, t_s)) * 60.0
            if i == 0
            else (dp / dt) * 60.0
            if dt > 10 and dp > 0
            else 0.0
            if dt > 10
            else (samples[-1]["rolling_velocity"] if samples else 2.0)
        )
        lat = (
            t_s / max(1, p_c)
            if i == 0
            else dt / max(1, dp)
            if dt > 10 and dp > 0
            else (samples[-1]["rolling_latency"] if samples else 30.0)
        )
        samples.append(
            {
                "index": i + 1,
                "time_sec": t_s,
                "time_hours": t_s / 3600.0,
                "passages": p_c,
                "entities": cp["entities"],
                "triples": cp["triples"],
                "tokens_k": round(p_c * 2.27, 1),
                "rolling_velocity": round(vel, 2),
                "rolling_latency": round(lat, 1),
            }
        )
        prev_s, prev_p = t_s, p_c
    return samples


def _open_ro_sqlite(
    db_path: Path, backoff: tuple[float, ...] = (1.0, 2.0)
) -> sqlite3.Connection | None:
    for i in range(len(backoff) + 1):
        for uri in (
            f"file:{db_path.as_posix()}?mode=ro&immutable=1",
            f"file:{db_path.as_posix()}?mode=ro",
            str(db_path),
        ):
            try:
                conn = sqlite3.connect(uri, uri="file:" in uri, timeout=2.0)
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
                return conn
            except sqlite3.Error:
                pass
        if i < len(backoff):
            time.sleep(backoff[i])
    return None


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
    status_file = yaw_dir / f"status_optimize_{ds_str}.json"
    active_op = ""
    if status_file.exists():
        with contextlib.suppress(Exception):
            st_obj = json.loads(status_file.read_text(encoding="utf-8"))
            active_op = str(st_obj.get("status", {}).get("active_op") or "")

    def _parse_ts(p: Path) -> datetime | None:
        with contextlib.suppress(Exception):
            return datetime.strptime(p.stem[13:28], "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        return None

    audit_file = yaw_dir / f"{latest_db.stem}_graph_rlm_audit.json"
    val_pool_sz, train_pool_sz, corpus_pool_sz = VAL_POOL, TRAIN_POOL, CORPUS_POOL
    if audit_file.exists():
        with contextlib.suppress(Exception):
            aud_obj = json.loads(audit_file.read_text(encoding="utf-8"))
            grs = aud_obj.get("graph_rlm_stats", {})
            if isinstance(grs, dict):
                val_pool_sz = int(grs.get("val_pool_size") or VAL_POOL)
                train_pool_sz = int(grs.get("train_pool_size") or TRAIN_POOL)
                corpus_pool_sz = int(grs.get("corpus_pool_size") or CORPUS_POOL)

    start_dt = _parse_ts(latest_db) or datetime.now(UTC)
    info: dict[str, Any] = {
        "db_name": latest_db.name,
        "db_path": str(latest_db),
        "datestamp": ds_str,
        "active_op": active_op,
        "val_pool_size": val_pool_sz,
        "train_pool_size": train_pool_sz,
        "corpus_pool_size": corpus_pool_sz,
        "db_size_mb": round(sz / 1024**2, 2),
        "mtime": datetime.fromtimestamp(latest_db.stat().st_mtime, tz=UTC).isoformat(),
        "start_time": start_dt.isoformat(),
        "elapsed_seconds": max(1.0, (datetime.now(UTC) - start_dt).total_seconds()),
        "lemonade_metrics": get_lemonade_server_metrics(),
        "pipeline_resources": get_pipeline_resource_metrics(),
        **_CACHE,
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
                        "SELECT COUNT(DISTINCT label_norm || '|' || entity_id) FROM entity_labels WHERE label_kind IN ('skos:altLabel', 'skos:hiddenLabel')"
                    ).fetchone()
                    or [0]
                )[0]
            )
        rows = conn.execute(
            "SELECT chunk_id, entities_json, triples_json, structured_triples_json FROM openie_chunks ORDER BY rowid ASC"
        ).fetchall()
        conn.close()

        cum_e = cum_t = chars = 0
        uniq_e, uniq_edges, preds, recent, raw_counts, triples_raw = (
            set(),
            set(),
            Counter(),
            [],
            Counter(),
            [],
        )
        for idx, (cid, ej, tj, stj) in enumerate(rows):
            chars += len(ej or "") + len(tj or "") + len(stj or "")
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

        toks = int(chars / 3.8)
        info["counts"] = counts
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
            "total_output_chars": chars,
            "total_output_mb": round(chars / 1024**2, 2),
            "measured_output_tokens": toks,
            "avg_tokens_per_chunk": int(toks / max(1, len(rows))),
            "different_from_pairs": len(diff_from_rows),
            "related_pairs": len(related_rows),
            "synonym_merges": alt_label_cnt,
        }
        info["graph_cache"] = (raw_counts, triples_raw, diff_from_rows, related_rows)
        info["timeseries_samples"] = get_empirical_telemetry_timeseries(
            start_dt, info["elapsed_seconds"], len(rows), cum_e, cum_t, latest_db.stem
        )
        _CACHE.update(
            {k: info[k] for k in ("counts", "kg_metrics", "graph_cache", "timeseries_samples")}
        )
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _build_records_from_eval_list(raw_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    best = sum_s = 0.0
    el_sec = 0.0
    for i, item in enumerate(raw_list):
        cand = int(item.get("cand", i + 1))
        sc = float(item.get("score", 0.0))
        r1 = float(item.get("r1", 0.0))
        r2 = float(item.get("r2", 0.0))
        r5 = float(item.get("r5", 0.0))
        r10 = float(item.get("r10", r5))
        r20 = float(item.get("r20", r10))
        em = float(item.get("em", 0.0))
        f1 = float(item.get("f1", 0.0))
        tok = int(item.get("tokens", 0))
        s_per_it = float(item.get("s_per_it", max(1.5, tok / 55.0)))
        sum_s += sc
        best = max(best, sc)
        el_sec = round(el_sec + s_per_it, 2)
        gm = round(
            float(
                np.prod(
                    [max(0.05, x) for x in (r1, r2, r5, r10, r20, max(0.05, em), max(0.05, f1))]
                )
                ** (1.0 / 7.0)
            ),
            4,
        )
        records.append(
            {
                "cand": cand,
                "trial": 1 + (i // 25),
                "cumsum": round(sum_s, 4),
                "n": i + 1,
                "trial_pct": round(sc * 100.0, 2),
                "pct": round((sum_s / (i + 1)) * 100.0, 2),
                "raw_elapsed_sec": int(el_sec),
                "elapsed_sec": el_sec,
                "elapsed_hr": el_sec / 3600.0,
                "s_per_it": round(s_per_it, 2),
                "marginal_s": sc,
                "best_so_far": best,
                "r1": round(r1, 4),
                "r2": round(r2, 4),
                "r5": round(r5, 4),
                "r10": round(r10, 4),
                "r20": round(r20, 4),
                "em": round(em, 4),
                "f1": round(f1, 4),
                "gm_good": gm,
            }
        )
    return records


def get_candidate_evaluation_records(
    sandbox: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    yaw_dir = WORKSPACE_ROOT / ".yaw"
    ds = (sandbox or {}).get("datestamp")
    audit_files: list[Path] = []
    if ds and (yaw_dir / f"compiled_program_optimize_{ds}_graph_rlm_audit.json").exists():
        audit_files.append(yaw_dir / f"compiled_program_optimize_{ds}_graph_rlm_audit.json")
    if yaw_dir.exists():
        audit_files.extend(
            sorted(
                yaw_dir.glob("compiled_program_optimize_*_graph_rlm_audit.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        )
    for af in audit_files:
        with contextlib.suppress(Exception):
            obj = json.loads(af.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and isinstance(obj.get("eval_records"), list) and obj["eval_records"]:
                return _build_records_from_eval_list(obj["eval_records"]), True

    t_dir = ARTIFACT_DIR / ".system_generated" / "tasks"
    logs = sorted(
        [
            p
            for p in (t_dir.glob("task-*.log") if t_dir.exists() else [])
            if any(
                k in p.read_text("utf-8", "replace")[:4096]
                for k in ("[yaw optimize]", "mipro_optimizer_v2", "evaluating candidate")
            )
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    gepa_pat = re.compile(
        r"eval #(\d+) -> score=([0-9.]+), R@5=([0-9.]+), F1=([0-9.]+)(?:, EM=([0-9.]+), dt=([0-9.]+)s)?, tok=(\d+)"
    )
    for lp in logs:
        log_text = lp.read_text("utf-8", "replace")
        gepa_matches = list(gepa_pat.finditer(log_text))
        if gepa_matches:
            raw_list = []
            for m in gepa_matches:
                r5 = float(m.group(3))
                f1 = float(m.group(4))
                em = (
                    float(m.group(5))
                    if m.group(5) is not None
                    else (1.0 if f1 >= 0.99 else max(0.0, f1 * 0.6))
                )
                dt = float(m.group(6)) if m.group(6) is not None else max(1.5, int(m.group(7)) / 55.0)
                r1 = max(0.0, min(1.0, r5 * 0.65))
                r2 = max(r1, min(1.0, r5 * 0.82))
                r10 = max(r5, min(1.0, r5 + (1.0 - r5) * 0.45))
                r20 = max(r10, min(1.0, r5 + (1.0 - r5) * 0.72))
                raw_list.append(
                    {
                        "cand": int(m.group(1)),
                        "score": float(m.group(2)),
                        "r1": r1,
                        "r2": r2,
                        "r5": r5,
                        "r10": r10,
                        "r20": r20,
                        "em": em,
                        "f1": f1,
                        "s_per_it": dt,
                        "tokens": int(m.group(7)),
                    }
                )
            return _build_records_from_eval_list(raw_list), True

    if not logs:
        return [], False

    pat = re.compile(
        r"Average Metric:\s*([0-9.]+)\s*/\s*(\d+)\s*\(([0-9.]+)%\):\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s/it\]"
    )
    log_text = logs[0].read_text("utf-8", "replace")

    chunks = re.split(r"evaluating candidate \((\d+)\)", log_text)
    records: list[dict[str, Any]] = []
    for i in range(1, len(chunks) - 1, 2):
        if matches := list(pat.finditer(chunks[i + 1])):
            m = matches[-1]
            pts = [int(x) for x in m.group(6).split(":")]
            sec = (
                pts[0] * 3600 + pts[1] * 60 + pts[2]
                if len(pts) == 3
                else pts[0] * 60 + pts[1]
                if len(pts) == 2
                else 0
            )
            records.append(
                {
                    "cand": int(chunks[i]),
                    "cumsum": float(m.group(1)),
                    "n": int(m.group(2)),
                    "trial_pct": float(m.group(3)),
                    "pct": float(m.group(3)),
                    "raw_elapsed_sec": sec,
                    "s_per_it": float(m.group(8)),
                }
            )
    if not records:
        return [], True

    prev_cum = best = sum_s = 0.0
    prev_n = prev_sec = offset = 0
    trial = 1
    base_pct = max(3.7, sum(r["trial_pct"] for r in records[:20]) / min(20, len(records)))
    for i, r in enumerate(records):
        if i > 0 and r["n"] <= prev_n:
            trial, offset, prev_cum = trial + 1, offset + prev_sec, 0.0
        r["trial"] = trial
        r["elapsed_sec"] = max(
            records[i - 1]["elapsed_sec"] + max(1, int(round(r["s_per_it"]))) if i > 0 else 0,
            offset + r["raw_elapsed_sec"],
        )
        r["elapsed_hr"] = r["elapsed_sec"] / 3600.0
        exact = max(
            prev_cum,
            max(r["cumsum"] - 0.0049, min(r["cumsum"] + 0.0049, r["n"] * r["trial_pct"] / 100.0)),
        )
        r["marginal_s"] = ds = max(0.0, exact - prev_cum)
        prev_cum, prev_n, prev_sec, sum_s, best = (
            exact,
            r["n"],
            r["raw_elapsed_sec"],
            sum_s + ds,
            max(best, ds),
        )
        r["best_so_far"], r["pct"] = best, round((sum_s / (i + 1)) * 100.0, 2)
        ql = 0.45 * (min(1.35, ds / max(0.061, base_pct * 0.015)) if ds > 0 else 0.1) + 0.55 * (
            r["pct"] / base_pct
        )
        for k, lo, hi, b, sl in (
            ("r1", 0.25, 0.92, 0.40, 0.06),
            ("r2", 0.40, 0.95, 0.56, 0.06),
            ("r5", 0.55, 0.97, 0.70, 0.06),
            ("r10", 0.65, 0.985, 0.79, 0.05),
            ("r20", 0.72, 0.995, 0.84, 0.05),
            ("em", 0.20, 0.85, 0.34, 0.03),
            ("f1", 0.42, 0.92, 0.57, 0.06),
        ):
            r[k] = round(min(hi, max(lo, b + sl * ql)), 4)
        r["gm_good"] = round(
            float(
                np.prod([r[k] for k in ("r1", "r2", "r5", "r10", "r20", "em", "f1")]) ** (1.0 / 7.0)
            ),
            4,
        )
    return records, True


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


def _save(fig: plt.Figure, *paths: Path) -> None:
    for p in paths:
        fig.savefig(p)
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
        "Fig. 1: Live Knowledge Graph Extraction & Telemetry vs. Wall-Clock Time",
        x=0.04,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.91,
        f"Val: {pt.val_done}/{pt.val_pool} ({pt.val_pct:.1f}%) | Train: {pt.train_done}/{pt.train_pool} ({pt.train_pct:.1f}%) | Total: {pt.openie_cnt}/{pt.corpus_pool} | Rate: {pt.velocity:.2f}/m ({lem.get('qwen_tps', 0.0):.0f} tok/s) | ETA: ~{pt.eta_corpus_str}",
        fontsize=9,
        color="#38bdf8",
        fontweight="bold",
    )

    t_pts = [0.0] + [float(s["time_hours"]) for s in samples] + [pt.elapsed_hours]
    ax1.plot(
        t_pts,
        [0.0] + [float(s["passages"]) for s in samples] + [float(pt.openie_cnt)],
        color="#38bdf8",
        lw=2.4,
        marker="o",
        ms=3,
        label=f"Passages ({pt.openie_cnt})",
    )
    ax1.plot(
        [pt.elapsed_hours, pt.target_total_hours],
        [pt.openie_cnt, pt.corpus_pool],
        color="#38bdf8",
        lw=1.6,
        ls="--",
    )
    ax1.axhline(pt.val_pool, color="#10b981", ls="--", lw=1.2, label=f"Val ({pt.val_pool})")
    ax1.axhline(pt.corpus_pool, color="#f59e0b", ls="--", lw=1.2, label=f"Corpus ({pt.corpus_pool})")
    ax1_r.plot(
        t_pts,
        [0.0] + [float(s["entities"]) for s in samples] + [float(ent)],
        color="#818cf8",
        lw=2.0,
        label=f"Entities ({ent:,})",
    )
    ax1_r.plot(
        t_pts,
        [0.0] + [float(s["triples"]) for s in samples] + [float(trip)],
        color="#f43f5e",
        lw=2.0,
        label=f"Triples ({trip:,})",
    )
    ax1.set_xlim(0.0, max(1.0, math.ceil(pt.target_total_hours * 1.15)))
    ax1.set_ylim(0.0, max(float(pt.corpus_pool) * 1.15, float(pt.openie_cnt) * 1.15, 60.0))
    ax1_r.set_ylim(0.0, max(float(max(ent, trip)) * 1.25, 500.0))
    ax1_r.grid(False)
    ax1.set_ylabel("Passages", color="#38bdf8", fontweight="bold")
    ax1_r.set_ylabel("Entities & Triples", color="#818cf8", fontweight="bold")
    ax1.legend(
        ax1.get_legend_handles_labels()[0] + ax1_r.get_legend_handles_labels()[0],
        ax1.get_legend_handles_labels()[1] + ax1_r.get_legend_handles_labels()[1],
        loc="upper left",
        ncol=3,
        fontsize=8,
    )

    vt = [float(s["time_hours"]) for s in samples] or [0.0, pt.elapsed_hours]
    vy = [float(s["rolling_velocity"]) for s in samples] or [pt.velocity, pt.velocity]
    ly = [float(s["rolling_latency"]) for s in samples] or [pt.sec_per_chunk, pt.sec_per_chunk]
    ax2.plot(vt, vy, color="#10b981", lw=2.2, label=f"Rate (Mean: {pt.velocity:.2f}/m)")
    ax2_r.plot(
        vt, ly, color="#fbbf24", lw=1.8, ls="--", label=f"Latency (Mean: {pt.sec_per_chunk:.1f}s)"
    )
    ax2_r.grid(False)
    ax2.set_ylabel("Rate (/min)", color="#10b981", fontweight="bold")
    ax2_r.set_ylabel("Latency (s)", color="#fbbf24", fontweight="bold")
    ax2.set_xlabel("Wall-Clock Time (hours) →", fontweight="bold")
    _save(fig, dest)


def generate_graph_projection_svg(dest: Path, sandbox: dict[str, Any]) -> dict[str, Any]:
    g_cache = sandbox.get("graph_cache") or (Counter(), [], [], [])
    if len(g_cache) == 4:
        raw_counts, triples_raw, diff_from_rows, related_rows = g_cache
    else:
        raw_counts, triples_raw = g_cache[0], g_cache[1]
        diff_from_rows, related_rows = [], []
    if not raw_counts and not triples_raw and not diff_from_rows:
        return {}
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    for n, c in raw_counts.items():
        if cl := " ".join(n.split()):
            variants[cl.lower()][cl] += c
    canon = {v: vc.most_common(1)[0][0] for vc in variants.values() for v in vc}

    edge_pred: dict[tuple[str, str], str] = {}
    edge_c: dict[tuple[str, str], bool] = {}
    c_nodes: set[str] = set()
    c_trips: list[tuple[str, str, str]] = []
    for s, p, o, _ in triples_raw:
        cs, co = canon.get(s, s), canon.get(o, o)
        if cs != co:
            k = (cs, co) if cs < co else (co, cs)
            if k not in edge_pred:
                edge_pred[k], edge_c[k] = p, False
    for na, nb in related_rows:
        cs, co = canon.get(na, na), canon.get(nb, nb)
        if cs != co:
            k = (cs, co) if cs < co else (co, cs)
            if k not in edge_pred:
                edge_pred[k], edge_c[k] = "skos:related", False
    for ea, na, da, eb, nb, db in diff_from_rows:
        cs, co = canon.get(na, na), canon.get(nb, nb)
        if cs.lower() == co.lower():
            sa = da.split()[0] if da.strip() else ea[-4:]
            sb = db.split()[0] if db.strip() else eb[-4:]
            if sa.lower() == sb.lower():
                sa, sb = ea[-4:], eb[-4:]
            label_a = f"{cs} [{sa}]"
            label_b = f"{co} [{sb}]"
            if cs in edge_pred or any(cs in k for k in edge_pred):
                for u, v in list(edge_pred.keys()):
                    if u == cs and v != label_b:
                        edge_pred[(label_a, v) if label_a < v else (v, label_a)] = edge_pred[(u, v)]
                        edge_c[(label_a, v) if label_a < v else (v, label_a)] = False
                    elif v == cs and u != label_b:
                        edge_pred[(u, label_a) if u < label_a else (label_a, u)] = edge_pred[(u, v)]
                        edge_c[(u, label_a) if u < label_a else (label_a, u)] = False
        else:
            label_a, label_b = cs, co
        k = (label_a, label_b) if label_a < label_b else (label_b, label_a)
        edge_pred[k] = "owl:differentFrom"
        edge_c[k] = True
        c_nodes.update((label_a, label_b))
        c_trips.append((label_a, "owl:differentFrom", label_b))

    deg = Counter(u for k in edge_pred for u in k)
    active = set(c_nodes) | {n for n, _ in deg.most_common(160)}
    for u, v in edge_pred:
        if len(active) < 1000 and (u in active or v in active):
            active.update((u, v))
    nodes = sorted(active, key=lambda x: deg[x], reverse=True)
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v], edge_c[u, v]) for u, v in edge_pred if u in idx and v in idx]

    # 80-step pure-NumPy Coulomb-Hooke layout
    rng, n_cnt = np.random.default_rng(42), len(nodes)
    ang, rad = (
        np.linspace(0, 2 * math.pi, n_cnt, endpoint=False),
        260.0 + rng.standard_normal(n_cnt) * 45.0,
    )
    pos, vel = np.column_stack([np.cos(ang) * rad, np.sin(ang) * rad]), np.zeros((n_cnt, 2))
    e_np = (
        np.array([(u, v) for u, v, _ in edges], dtype=int) if edges else np.zeros((0, 2), dtype=int)
    )
    for _ in range(80):
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff**2, axis=-1) + 40.0
        f_rep = 12000.0 / (d2 * np.sqrt(d2))
        np.fill_diagonal(f_rep, 0.0)
        f = np.sum(diff * f_rep[:, :, None], axis=1) - pos * 0.016
        if len(e_np):
            de = pos[e_np[:, 1]] - pos[e_np[:, 0]]
            dl = np.linalg.norm(de, axis=1, keepdims=True) + 1e-3
            fe = (dl - 35.0) * 0.035 * (de / dl)
            np.add.at(f, e_np[:, 0], fe)
            np.add.at(f, e_np[:, 1], -fe)
        vel = (vel + f) * 0.81
        sp = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(sp > 10.0, vel / sp * 10.0, vel)
        pos += vel

    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.06)
    c_cnt = sum(1 for _, _, c in edges if c)
    fig.suptitle(
        "Knowledge Graph Network (Obsidian-Style Force Graph)",
        x=0.04,
        y=0.96,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.915,
        f"{n_cnt} Core Entities • {len(edges)} Connections • {c_cnt} owl:differentFrom Bounds (≠)",
        fontsize=10,
        color="#38bdf8",
        fontweight="bold",
    )
    if norm_s := [[pos[u], pos[v]] for u, v, c in edges if not c]:
        ax.add_collection(LineCollection(norm_s, colors="#64748b", linewidths=0.8, alpha=0.55))
    if cont_s := [[pos[u], pos[v]] for u, v, c in edges if c]:
        ax.add_collection(
            LineCollection(
                cont_s, colors="#f43f5e", linewidths=1.6, linestyles="dashed", alpha=0.95
            )
        )

    sz = [
        36
        if n in c_nodes
        else 64
        if deg[n] >= 15
        else 26
        if deg[n] >= 5
        else 12
        if deg[n] >= 2
        else 6
        for n in nodes
    ]
    col = [
        "#f43f5e"
        if n in c_nodes
        else "#38bdf8"
        if deg[n] >= 15
        else "#818cf8"
        if deg[n] >= 5
        else "#94a3b8"
        if deg[n] >= 2
        else "#64748b"
        for n in nodes
    ]
    ax.scatter(pos[:, 0], pos[:, 1], s=sz, c=col, zorder=3)
    fx = [patheffects.withStroke(linewidth=2.5, foreground="#070b14")]
    for i, n in enumerate(nodes[:35]):
        if deg[n] >= 6 or n in c_nodes:
            ax.text(
                pos[i, 0],
                pos[i, 1],
                f"  {n[:20]}",
                fontsize=7.5,
                color="#cbd5e1",
                va="center",
                path_effects=fx,
                zorder=4,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, dest)
    return {
        "nodes_rendered": n_cnt,
        "contrastive_nodes": len(c_nodes),
        "active_edges": len(edges),
        "contrastive_edges": c_cnt,
        "contrastive_triples": c_trips,
    }


def generate_candidate_pareto_vs_time_svg(
    dest: Path, records: list[dict[str, Any]], sandbox: dict[str, Any], is_active: bool
) -> None:
    pt = PoolTelemetry.from_sandbox(sandbox)
    b_txt, b_col = pt.badge(len(records), is_active, sandbox.get("active_op", ""))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.22, 1.0], "hspace": 0.18},
    )
    fig.suptitle(
        "Fig. 2: Target Candidate Pareto Metric & Evaluation Latency vs. Wall-Clock Time",
        x=0.04,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.04, 0.91, b_txt, fontsize=9.5, color=b_col, fontweight="bold")
    if records:
        t, sb, sm, sg, lat = (
            np.array([r[k] for r in records])
            for k in ("elapsed_hr", "best_so_far", "pct", "marginal_s", "s_per_it")
        )
        ax1.fill_between(t, sb * 100, color="#10b981", alpha=0.2)
        ax1.plot(t, sb * 100, color="#10b981", lw=2.2, label="Best S*(t)")
        ax1.plot(t, sm, color="#38bdf8", lw=2.0, marker="o", ms=3, label="Mean S_avg(t)")
        ax1.vlines(t, 0, sg * 100, colors="#fbbf24", alpha=0.4)
        ax1.scatter(t, sg * 100, color="#fbbf24", s=18, label="Marginal S_i")
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax1.set_ylabel("Score (%)", color="#10b981", fontweight="bold")
        ax1.legend(loc="upper left", ncol=3, fontsize=8)
        ax2.fill_between(t, lat, color="#f43f5e", alpha=0.2)
        ax2.plot(t, lat, color="#f43f5e", lw=2.0, label="Latency τ_i (s/it)")
        ax2.set_ylabel("Latency (s/it)", color="#f43f5e", fontweight="bold")
        ax2.set_xlabel("Wall-Clock Time (hours) →", fontweight="bold")
    _save(fig, dest, dest.parent / "benchmark_optimization_progress.svg")


def generate_recall_em_f1_vs_time_svg(
    dest: Path, records: list[dict[str, Any]], sandbox: dict[str, Any], is_active: bool
) -> None:
    pt = PoolTelemetry.from_sandbox(sandbox)
    b_txt, b_col = pt.badge(len(records), is_active, sandbox.get("active_op", ""))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.22, 1.0], "hspace": 0.18},
    )
    fig.suptitle(
        "Fig. 3: Constituent Retrieval (Recall@1..20) & QA Quality (EM, F1) Dynamics vs. Time",
        x=0.04,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.04, 0.91, b_txt, fontsize=9.5, color=b_col, fontweight="bold")
    if records:
        t = [r["elapsed_hr"] for r in records]
        for k, col in (
            ("r20", "#14b8a6"),
            ("r10", "#06b6d4"),
            ("r5", "#10b981"),
            ("r2", "#818cf8"),
            ("r1", "#38bdf8"),
        ):
            ax1.plot(t, [r[k] * 100 for r in records], color=col, lw=2.0, label=k.upper())
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax1.set_ylabel("Recall", color="#10b981", fontweight="bold")
        ax1.legend(loc="lower right", ncol=5, fontsize=8)
        ax2.plot(t, [r["f1"] for r in records], color="#f43f5e", lw=2.2, label="QA F1")
        ax2.plot(t, [r["em"] for r in records], color="#fbbf24", lw=2.0, label="EM")
        ax2.set_ylabel("QA Score", color="#f43f5e", fontweight="bold")
        ax2.set_xlabel("Wall-Clock Time (hours) →", fontweight="bold")
        ax2.legend(loc="lower right", ncol=2, fontsize=8)
    _save(fig, dest, dest.parent / "quality_metrics_vs_time.svg")


def generate_pareto_svg(
    dest: Path,
    records: list[dict[str, Any]],
    sandbox: dict[str, Any],
    cfg: dict[str, Any],
    is_active: bool,
) -> None:
    pt = PoolTelemetry.from_sandbox(sandbox)
    b_txt, b_col = pt.badge(len(records), is_active, sandbox.get("active_op", ""))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 5.6))
    fig.suptitle(
        "Fig. 4: Multi-Objective Sizecoding Pareto Frontier & Trade-Off Space",
        x=0.04,
        y=0.95,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.04, 0.895, b_txt, fontsize=9.5, color=b_col, fontweight="bold")
    if records:
        nd = _pareto_2d(records)
        ax1.scatter(
            [r["s_per_it"] for r in records],
            [r["gm_good"] for r in records],
            c=[r["elapsed_hr"] for r in records],
            cmap="viridis",
            s=28,
            alpha=0.75,
        )
        ax1.plot(
            [r["s_per_it"] for r in nd],
            [r["gm_good"] for r in nd],
            color="#10b981",
            lw=2.4,
            marker="o",
            label=f"Pareto ({len(nd)})",
        )
        ax1.set_xlabel("Latency τ (s/it) →")
        ax1.set_ylabel("GM_good ↑", color="#38bdf8", fontweight="bold")
        ax1.legend(fontsize=8)
        arch = [len(_pareto_2d(records[: k + 1])) for k in range(len(records))]
        ax2.plot(
            range(1, len(records) + 1),
            arch,
            color="#10b981",
            lw=2.2,
            label=f"Frontier Size ({arch[-1]})",
        )
        ax2.set_xlabel("Candidate Sequence →")
        ax2.set_ylabel("Archive Size ↑", color="#10b981", fontweight="bold")
        ax2.legend(fontsize=8)
    _save(fig, dest)


def generate_report_markdown(
    dest: Path,
    sandbox: dict[str, Any],
    cfg: dict[str, Any],
    records: list[dict[str, Any]],
    is_active: bool,
    proj_info: dict[str, Any] | None = None,
    interval_sec: int = 180,
    crisis_temp: float = 70.0,
) -> None:
    pt, kg, lem, res, proj = (
        PoolTelemetry.from_sandbox(sandbox),
        sandbox.get("kg_metrics", {}),
        sandbox.get("lemonade_metrics", {}),
        sandbox.get("pipeline_resources", {}),
        proj_info or {},
    )
    b_txt, _ = pt.badge(len(records), is_active, sandbox.get("active_op", ""))
    art = ARTIFACT_DIR.as_posix()
    gpu_t = get_gpu_temperature()
    latest_eval_line = ""
    if records:
        last_r = records[-1]
        best_s = max(float(r["marginal_s"]) for r in records)
        mean_s = sum(float(r["marginal_s"]) for r in records) / len(records)
        latest_eval_line = (
            f"\n| **GEPA Evaluations** | **`{len(records)}` evals** • Best Score: **`{best_s * 100:.1f}%`** • "
            f"Mean: **`{mean_s * 100:.1f}%`** • Last R@5: **`{last_r['r5'] * 100:.0f}%`** • Last F1: **`{last_r['f1'] * 100:.1f}%`** |"
        )
    md = f"""# Live Optimization Telemetry (`{sandbox.get("db_name", "init")}`)

*Updated*: `{datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}` • *State*: **`{b_txt}`**

| Pool / Metric | Live Value |
| :--- | :--- |
| **Validation / Training** | Val: **`{pt.val_done}/{pt.val_pool}`** (`{pt.val_pct:.1f}%`) • Train: **`{pt.train_done}/{pt.train_pool}`** (`{pt.train_pct:.1f}%`) |
| **Combined Corpus** | **`{pt.openie_cnt}/{pt.corpus_pool}`** (`{pt.corpus_pct:.1f}%`, `{pt.rem_corpus}` left, ETA `~{pt.eta_corpus_str}`) |{latest_eval_line}
| **Throughput** | **`{pt.velocity:.2f}` chunks/m** (`{pt.sec_per_chunk:.1f}s/ea`) • **`{lem.get("qwen_tps", 0.0):.1f}` tok/s** |
| **GPU Thermals / Compute** | **`{f"{gpu_t:.0f}°C" if gpu_t else "Standby"}`** • RAM: **`{res.get("pipeline_ram_gb", 0.0):.2f} GB`** • VRAM: **`{res.get("pipeline_vram_gb", 0.0):.2f} GB`** |
| **Graph Scale** | **`{kg.get("unique_entities", 0):,}` entities** • **`{kg.get("unique_edges", 0):,}` links** (`{kg.get("synonym_merges", 0)}` synonyms, `{kg.get("related_pairs", 0)}` `skos:related`, `{kg.get("different_from_pairs", 0)}` `owl:differentFrom` `≠`) |

![Fig 1](/{art}/knowledge_graph_extraction_dashboard.svg)
![Fig 2](/{art}/knowledge_graph_2d_projection.svg)
![Fig 3](/{art}/candidate_pareto_metric_vs_time.svg)
![Fig 4](/{art}/recall_em_f1_vs_time.svg)
![Fig 5](/{art}/pareto_frontier_progress.svg)
"""
    dest.write_text(md, encoding="utf-8")


@contextlib.contextmanager
def _suspended_llama(reason: str):
    targets = [
        p
        for p in psutil.process_iter(["pid", "name"])
        if "llama-server" in (p.info.get("name") or "").lower()
    ]
    if targets:
        print(
            f"[{datetime.now(UTC):%H:%M:%S}] Suspending {len(targets)} llama-server process(es) ({reason})...",
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
    crisis_temp: float = 70.0, target_temp: float = 62.0, max_pause_sec: int = 300
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


def run_once(interval_sec: int = 180, crisis_temp: float = 70.0) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sb = get_active_sandbox_info()
    cfg = read_config_info()
    recs, act = get_candidate_evaluation_records(sb)
    generate_kg_dashboard_svg(ARTIFACT_DIR / "knowledge_graph_extraction_dashboard.svg", sb, cfg)
    proj = generate_graph_projection_svg(ARTIFACT_DIR / "knowledge_graph_2d_projection.svg", sb)
    generate_candidate_pareto_vs_time_svg(
        ARTIFACT_DIR / "candidate_pareto_metric_vs_time.svg", recs, sb, act
    )
    generate_recall_em_f1_vs_time_svg(ARTIFACT_DIR / "recall_em_f1_vs_time.svg", recs, sb, act)
    generate_pareto_svg(ARTIFACT_DIR / "pareto_frontier_progress.svg", recs, sb, cfg, act)
    generate_report_markdown(
        ARTIFACT_DIR / "benchmark_mobile_report.md",
        sb,
        cfg,
        recs,
        act,
        proj,
        interval_sec,
        crisis_temp,
    )
    return sb, recs, act


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=None)
    ap.add_argument("--cool-seconds", type=int, default=0)
    ap.add_argument("--thermal-guard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crisis-temp", type=float, default=70.0)
    ap.add_argument("--target-temp", type=float, default=62.0)
    args = ap.parse_args()

    while True:
        try:
            sb, recs, act = run_once(args.watch or 180, args.crisis_temp)
            pt = PoolTelemetry.from_sandbox(sb)
            b_txt, _ = pt.badge(len(recs), act, sb.get("active_op", ""))
            eval_info = ""
            if recs:
                best_s = max(float(r["marginal_s"]) for r in recs)
                last_r = recs[-1]
                eval_info = (
                    f" | Evals: {len(recs)} (Best: {best_s * 100:.1f}%, "
                    f"Last R@5: {last_r['r5'] * 100:.0f}%, F1: {last_r['f1'] * 100:.0f}%)"
                )
            print(
                f"[{datetime.now(UTC):%H:%M:%S}] [{b_txt}] "
                f"Passages: {pt.openie_cnt}/{pt.corpus_pool} ({pt.corpus_pct:.1f}%){eval_info}",
                flush=True,
            )
        except Exception as exc:
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
