"""Edge-case and resilience tests for the clean-slate YAW watchdog monitor."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from xml.etree import ElementTree

import filelock
import pytest

import scripts.watchdog as watchdog


def test_empty_workspace_handles_missing_dbs_locks_and_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "dashboard"
    snap = watchdog.run_once(artifact_dir=out_dir, workspace_root=tmp_path)

    for db_name in ("yaw.db", "optimize_yaw.db", "optuna_index.db"):
        info = snap["databases"][db_name]
        assert info["exists"] is False
        assert info["chunks"] is None
        assert info["tables"] == []
        assert "error" not in info

    assert snap["locks"] == {
        "yaw.db.lock": False,
        "optimize_yaw.db.lock": False,
        "optuna_index.db.lock": False,
    }
    assert not any((tmp_path / ".yaw").glob("*.lock")) if (tmp_path / ".yaw").exists() else True
    assert snap["llm_metrics"] is None
    for key in ("status", "dspy_status", "dspy_gepa_audit", "optuna_status", "optuna_best_params"):
        assert snap[key] is None

    svg_root = ElementTree.fromstring((out_dir / "score_progress.svg").read_text(encoding="utf-8"))
    assert "Waiting for optimization scores..." in " ".join(svg_root.itertext())
    assert not watchdog.plt.get_fignums()

    md_text = (out_dir / "benchmark_mobile_report.md").read_text(encoding="utf-8")
    assert "![Score Progress](score_progress.svg)" in md_text
    assert "`n/a` chunks" in md_text


def test_sqlite_missing_chunks_table_and_corrupt_db_file(tmp_path: Path) -> None:
    yaw_dir = tmp_path / ".yaw"
    yaw_dir.mkdir(parents=True)

    with closing(sqlite3.connect(str(yaw_dir / "yaw.db"))) as conn:
        conn.execute("CREATE TABLE other_meta (id INTEGER PRIMARY KEY)")
        conn.commit()
    (yaw_dir / "optimize_yaw.db").write_bytes(b"corrupt sqlite bytes")

    snap = watchdog.get_active_snapshot(tmp_path, include_llm_metrics=False)
    yaw_info = snap["databases"]["yaw.db"]
    assert yaw_info["exists"] is True
    assert yaw_info["tables"] == ["other_meta"]
    assert yaw_info["chunks"] is None
    assert "error" not in yaw_info

    opt_info = snap["databases"]["optimize_yaw.db"]
    assert opt_info["exists"] is True
    assert opt_info["chunks"] is None
    assert opt_info["tables"] == []
    assert "error" in opt_info


def test_torn_invalid_and_nondict_json_resilience(tmp_path: Path) -> None:
    yaw_dir = tmp_path / ".yaw"
    yaw_dir.mkdir(parents=True)

    (yaw_dir / "status.json").write_text('{"lifecycle": "ready"', encoding="utf-8")
    (yaw_dir / "dspy_status.json").write_text("[1, 2, 3]", encoding="utf-8")
    (yaw_dir / "dspy_gepa_audit.json").write_bytes(b"\xff\xfe{")
    (yaw_dir / "optuna_status.json").write_text('"scalar"', encoding="utf-8")
    (yaw_dir / "optuna_best_params.json").write_text(
        json.dumps({"best_score": 0.9125, "total_trials": 8}), encoding="utf-8"
    )

    snap = watchdog.get_active_snapshot(tmp_path, include_llm_metrics=False)
    assert snap["status"] is None
    assert snap["dspy_status"] is None
    assert snap["dspy_gepa_audit"] is None
    assert snap["optuna_status"] is None
    assert snap["optuna_best_params"] == {"best_score": 0.9125, "total_trials": 8}

    svg_path = tmp_path / "optuna_best.svg"
    watchdog.generate_score_svg(svg_path, snap)
    svg_text = " ".join(ElementTree.fromstring(svg_path.read_text(encoding="utf-8")).itertext())
    assert "0.9125" in svg_text
    assert not watchdog.plt.get_fignums()


def test_score_svg_filters_invalid_values_and_closes_figure_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap = {
        "dspy_gepa_audit": {
            "val_aggregate_scores": [True, "bad", float("nan"), float("inf"), 0.5, 0.75],
        }
    }
    svg_path = tmp_path / "filtered.svg"
    watchdog.generate_score_svg(svg_path, snap)
    svg_text = " ".join(ElementTree.fromstring(svg_path.read_text(encoding="utf-8")).itertext())
    assert "0.7500" in svg_text
    assert not watchdog.plt.get_fignums()

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(watchdog, "write_text_atomic", _boom)
    with pytest.raises(OSError, match="disk full"):
        watchdog.generate_score_svg(tmp_path / "fail.svg", snap)
    assert not watchdog.plt.get_fignums()


def test_file_lock_lifecycle_and_config_error_resilience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_lock = tmp_path / "missing.lock"
    assert watchdog.is_file_locked(missing_lock) is False
    assert not missing_lock.exists()

    lock_path = tmp_path / "active.lock"
    lock_path.write_text("", encoding="utf-8")
    lock = filelock.FileLock(str(lock_path), timeout=0)
    lock.acquire()
    try:
        assert watchdog.is_file_locked(lock_path) is True
    finally:
        lock.release()
    assert watchdog.is_file_locked(lock_path) is False

    (tmp_path / "yaw.toml").write_text("[invalid toml", encoding="utf-8")
    snap_bad_cfg = watchdog.get_active_snapshot(tmp_path)
    assert snap_bad_cfg["llm_metrics"] == {"server_up": False}

    (tmp_path / "yaw.toml").write_text('[inference]\nbase_url = "http://127.0.0.1:9/v1"\n', encoding="utf-8")
    snap_down = watchdog.get_active_snapshot(tmp_path)
    assert snap_down["llm_metrics"] == {"server_up": False}

    custom_dir = tmp_path / "custom_art"
    monkeypatch.setenv("YAW_ARTIFACT_DIR", str(custom_dir))
    assert watchdog._resolve_artifact_dir() == custom_dir

    with pytest.raises(SystemExit) as exc_info:
        watchdog.main(["--watch", "0"])
    assert exc_info.value.code == 2
