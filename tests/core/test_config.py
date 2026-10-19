"""Unit tests for YawConfig contract, TOML loading, optimization sandbox derivation, and get/set."""

from pathlib import Path
from typing import Any, cast

import pytest

from yaw.core import YawConfig, YawRuntimeError, get_config_value, set_config_value


def test_yaw_config_defaults() -> None:
    cfg = YawConfig()
    assert cfg.optimize_auto == "light"
    assert not hasattr(cfg, "optimize_max_metric_calls")
    assert cfg.optimize_mode == "hybrid"


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
    for ext in ("*.md", "*.txt", "*.json", "*.jsonl", "*.py", "*.rs", "*.ts", "*.html", "*.toml", "*.yaml"):
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
