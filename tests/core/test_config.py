"""Unit tests for YawConfig contract, TOML loading, optimization sandbox derivation, and get/set."""

import os
from pathlib import Path
from typing import Any, cast

import pytest

from yaw.core import YawConfig, YawRuntimeError, get_config_value, set_config_value


def test_yaw_config_defaults() -> None:
    cfg = YawConfig()
    assert cfg.optimize_auto == "light"
    assert not hasattr(cfg, "optimize_max_metric_calls")
    assert cfg.optimize_mode == "hybrid"
    assert cfg.retrieval_top_k == 10


def test_yaw_config_validation() -> None:
    with pytest.raises(YawRuntimeError) as exc_info:
        YawConfig(optimize_auto=cast(Any, "invalid"))
    assert exc_info.value.code == "INVALID_INPUT"


def test_yaw_config_for_optimization(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    opt = cfg.for_optimization("20261016_120000")

    assert opt.db_path is not None
    assert opt.lock_path is not None
    assert opt.status_path is not None
    assert opt.state_dir is not None
    assert opt.compiled_program_path is not None

    assert opt.db_path.name == "yaw_optimize_20261016_120000.db"
    assert opt.lock_path.name == "yaw_optimize_20261016_120000.db.lock"
    assert opt.status_path.name == "status_optimize_20261016_120000.json"
    assert opt.state_dir.name == "coco_state_optimize_20261016_120000"
    assert opt.compiled_program_path.name == "compiled_program_optimize_20261016_120000.json"
    assert opt.poll_interval_seconds == 0.0

    # Model and retrieval parameters preserved
    assert opt.embedding_model == cfg.embedding_model
    assert opt.qa_model == cfg.qa_model
    assert opt.retrieval_top_k == cfg.retrieval_top_k
    assert opt.optimize_auto == cfg.optimize_auto


def test_optimization_latest_is_newest_run_not_largest_or_last_written(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    old = cfg.for_optimization("20261021_120000")
    new = cfg.for_optimization("20261022_120000")
    assert old.db_path is not None and new.db_path is not None
    old.db_path.parent.mkdir(parents=True)
    old.db_path.write_bytes(b"old" * 100)
    new.db_path.write_bytes(b"new")
    os.utime(old.db_path, (2_000_000_000, 2_000_000_000))
    os.utime(new.db_path, (1_000_000_000, 1_000_000_000))
    (old.db_path.parent / "yaw_optimize_z_backup.db").write_bytes(b"backup")
    assert cfg.for_optimization(db_override="latest") == new
    assert cfg.for_optimization(db_override=new.db_path) == new


def test_optimization_resume_missing_database_never_creates_a_run(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    for selected in ("latest", "missing.db"):
        with pytest.raises(YawRuntimeError, match="No optimization run|does not exist"):
            cfg.for_optimization(db_override=selected)
    assert not (tmp_path / ".yaw").exists()


def test_optimization_custom_database_has_stable_paths(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    database = tmp_path / "custom" / "experiment.db"
    database.parent.mkdir()
    database.touch()
    first = cfg.for_optimization(db_override="custom/experiment.db")
    second = cfg.for_optimization("20300101_000000", db_override=database)
    assert first == second
    assert first.lock_path == database.with_suffix(".db.lock")
    assert first.compiled_program_path == (database.parent / "compiled_program_optimize_experiment.json")


def test_yaw_config_for_benchmark(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    idx_root = tmp_path / "bench_idx"
    bench = cfg.for_benchmark("2wikimultihopqa", "a1b2c3d4", index_root=idx_root, fresh=False)

    assert bench.workspace_root == idx_root / "2wikimultihopqa_a1b2c3d4"
    assert bench.corpus_root == bench.workspace_root
    assert bench.db_path == bench.workspace_root / "yaw.db"
    assert bench.lock_path == bench.workspace_root / "yaw.db.lock"
    assert bench.status_path == bench.workspace_root / "status.json"
    assert bench.state_dir == bench.workspace_root / "coco_state"
    assert bench.log_path == bench.workspace_root / "yaw.log"
    assert bench.poll_interval_seconds == 0.0
    assert "title" in bench.json_keys_whitelist
    assert "paragraphs.*" in bench.json_keys_whitelist
    assert "question*" in bench.json_keys_blacklist
    assert "answer*" in bench.json_keys_blacklist

    # Verify fresh=True cleans existing sandbox dir
    marker = bench.workspace_root / "stale.txt"
    marker.write_text("stale", encoding="utf-8")
    bench_fresh = cfg.for_benchmark("2wikimultihopqa", "a1b2c3d4", index_root=idx_root, fresh=True)
    assert bench_fresh.workspace_root == bench.workspace_root
    assert not marker.exists()


def test_config_get_and_set_values(tmp_path: Path) -> None:
    toml_file = tmp_path / "yaw.toml"
    toml_file.write_text(
        """[optimization]
auto = "light"
mode = "gepa"

[inference]
qa_model = "Qwen3-4B-Instruct-2507-GGUF"
""",
        encoding="utf-8",
    )

    # 1. get specific keys
    auto_val = get_config_value(path=toml_file, key="optimization.auto", workspace_override=tmp_path)
    assert auto_val == "light"

    # Alias check
    auto_alias = get_config_value(path=toml_file, key="auto", workspace_override=tmp_path)
    assert auto_alias == "light"

    # 2. set key
    key, val = set_config_value(
        key="optimization.auto",
        raw_value="heavy",
        path=toml_file,
        workspace_override=tmp_path,
    )
    assert key == "optimization.auto"
    assert val == "heavy"

    # Verify updated
    assert get_config_value(path=toml_file, key="optimization.auto", workspace_override=tmp_path) == "heavy"

    # 3. get all keys
    all_cfg = get_config_value(path=toml_file, workspace_override=tmp_path)
    assert isinstance(all_cfg, dict)
    assert all_cfg["optimize_auto"] == "heavy"


def test_yaw_config_json_keys_and_patterns(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    assert cfg.json_keys_whitelist == ("*",)
    assert cfg.json_keys_blacklist == ()
    for ext in (
        "*.md",
        "*.txt",
        "*.json",
        "*.jsonl",
        "*.py",
        "*.rs",
        "*.ts",
        "*.html",
        "*.toml",
        "*.yaml",
    ):
        assert ext in cfg.included_patterns

    fp_default = cfg.config_fingerprint()
    cfg_filtered = YawConfig(
        workspace_root=tmp_path,
        json_keys_whitelist=cast(Any, "title,text,paragraphs.*"),
        json_keys_blacklist=("question*", "answer*", "Bearer*"),
    )
    assert cfg_filtered.json_keys_whitelist == ("title", "text", "paragraphs.*")
    assert cfg_filtered.json_keys_blacklist == ("question*", "answer*", "Bearer*")
    assert cfg_filtered.config_fingerprint() != fp_default


def test_default_patterns_and_write_text_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from yaw.core import (
        DEFAULT_EXCLUDED_PATTERNS,
        DEFAULT_INCLUDED_PATTERNS,
        write_text_atomic,
    )

    cfg = YawConfig(workspace_root=tmp_path)
    assert cfg.included_patterns == DEFAULT_INCLUDED_PATTERNS
    assert cfg.excluded_patterns == DEFAULT_EXCLUDED_PATTERNS
    assert "*.md" in DEFAULT_INCLUDED_PATTERNS
    assert "__pycache__/*" in DEFAULT_EXCLUDED_PATTERNS

    target = tmp_path / "nested" / "atomic.json"
    write_text_atomic(target, '{"status": "ok"}\n')
    assert target.read_text(encoding="utf-8") == '{"status": "ok"}\n'
    write_text_atomic(str(target), '{"status": "updated"}\n', retries=2)
    assert target.read_text(encoding="utf-8") == '{"status": "updated"}\n'
    assert not list(target.parent.glob("*.tmp"))

    real_replace = os.replace
    attempts = 0

    def flaky_replace(src: Any, dst: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("Simulated transient Windows file lock")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    write_text_atomic(target, '{"status": "recovered"}\n', retries=2)
    assert attempts == 2
    assert target.read_text(encoding="utf-8") == '{"status": "recovered"}\n'

    def always_locked_replace(src: Any, dst: Any) -> None:
        raise PermissionError("Simulated persistent Windows file lock")

    monkeypatch.setattr(os, "replace", always_locked_replace)
    write_text_atomic(target, '{"status": "fallback"}\n', retries=2)
    assert target.read_text(encoding="utf-8") == '{"status": "fallback"}\n'
    assert not list(target.parent.glob("*.tmp"))
