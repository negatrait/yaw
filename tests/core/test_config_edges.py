"""Real TOML round trips, rejected edits and strict section/validation boundaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from yaw.core import YawConfig, YawRuntimeError, get_config_value, load_config, set_config_value


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap"),
        ({"chunk_size": 100, "chunk_overlap": -1}, "chunk_overlap"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"http_port": 70000}, "http_port"),
    ],
)
def test_invalid_yaw_config_boundaries_are_rejected(kwargs, message):
    with pytest.raises(YawRuntimeError, match=message) as exc_info:
        YawConfig(**kwargs)
    assert exc_info.value.code == "INVALID_INPUT"


def test_loading_defaults_and_explicit_workspace_selection(tmp_path):
    assert load_config(workspace_override=tmp_path).workspace_root == tmp_path
    path = tmp_path / "yaw.toml"
    path.write_text('[paths]\nworkspace_root = "nested"\n', encoding="utf-8")
    assert load_config(workspace_override=tmp_path).workspace_root == tmp_path
    assert load_config(path).workspace_root == tmp_path / "nested"
    path.write_text("[runtime\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError, match="Invalid TOML"):
        load_config(path)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [('"alpha, beta, "', ("alpha", "beta")), ('["alpha", " beta "]', ("alpha", "beta"))],
)
def test_dataset_forms_load_into_stable_tuples(tmp_path, literal, expected):
    path = tmp_path / "yaw.toml"
    path.write_text(f"[optimization]\ndatasets = {literal}\n", encoding="utf-8")
    assert load_config(path).optimize_datasets == expected


@pytest.mark.parametrize(
    "bad_toml",
    [
        "[optimization]\ndatasets = 42\n",
        "[optimization]\ndatasets = [1, 2]\n",
        "[optimization]\ndatasets = []\n",
        '[optimization]\ndatasets = ""\n',
        '[optimization]\ndatasets = "  , "\n',
        "[ingestion]\nincluded_patterns = 123\n",
        "[ingestion]\nincluded_patterns = [1]\n",
        "[paths]\ndb_path = 123\n",
        "[paths]\nworkspace_root = 123\n",
        "[inference]\nqa_model = 123\n",
        "[inference]\nembedding_task_mode = [1]\n",
    ],
)
def test_non_string_and_non_path_toml_values_fail_fast(tmp_path, bad_toml):
    path = tmp_path / "yaw.toml"
    path.write_text(bad_toml, encoding="utf-8")
    with pytest.raises(YawRuntimeError) as exc_info:
        load_config(path)
    assert exc_info.value.code == "INVALID_INPUT"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"db_path": 123},
        {"workspace_root": 123},
        {"workspace_root": True},
        {"workspace_root": "   "},
        {"qa_model": 123},
        {"embedding_task_mode": [1]},
        {"included_patterns": 123},
        {"included_patterns": [1]},
        {"optimize_datasets": 42},
        {"optimize_datasets": [1, 2]},
        {"optimize_datasets": ()},
        {"optimize_datasets": []},
        {"optimize_datasets": "   "},
    ],
)
def test_yaw_config_rejects_non_string_and_invalid_path_types(kwargs):
    with pytest.raises(YawRuntimeError) as exc_info:
        YawConfig(**cast(dict[str, Any], kwargs))
    assert exc_info.value.code == "INVALID_INPUT"


def test_constructor_normalizes_comma_separated_filters_and_dataset_names(tmp_path):
    values = json.loads(
        '{"optimize_datasets": "a, b, ", "included_patterns": "*.md, *.txt", "excluded_patterns": "*.tmp, dist/*"}'
    )
    config = YawConfig(workspace_root=tmp_path, **values)
    assert config.optimize_datasets == ("a", "b")
    assert config.included_patterns == ("*.md", "*.txt")
    assert config.excluded_patterns == ("*.tmp", "dist/*")


@pytest.mark.parametrize(
    ("key", "literal", "expected"),
    [
        ("ingestion.chunk_overlap", "120", 120),
        ("inference.max_output_tokens", "2048", 2048),
        ("inference.max_output_tokens", "null", None),
        ("runtime.http_port", "13307", 13307),
        ("inference.qa_model", "'model with spaces'", "model with spaces"),
        ("inference.qa_model", "[invalid JSON]", "[invalid JSON]"),
        ("inference.qa_model", "bare-model-alias", "bare-model-alias"),
        ("ingestion.included_patterns", '["*.md", "*.txt"]', ["*.md", "*.txt"]),
    ],
)
def test_atomic_scalar_and_array_updates_roundtrip_real_toml(tmp_path, key, literal, expected):
    path = tmp_path / "yaw.toml"
    path.write_text("# Keep this comment\n[runtime]\nqueue_maxsize = 4\n", encoding="utf-8")
    returned_key, value = set_config_value(key, literal, path)
    assert returned_key == key
    assert value == expected
    assert get_config_value(path, key) == expected
    assert path.read_text(encoding="utf-8").startswith("# Keep this comment\n")
    assert not list(tmp_path.glob("*.tmp.*"))


def test_inserted_settings_preserve_section_boundaries_and_aliases(tmp_path):
    path = tmp_path / "yaw.toml"
    path.write_text('[runtime]\n# Existing section\n[optimization]\noptimize_auto = "light"\n', encoding="utf-8")
    assert set_config_value("queue_maxsize", "7", path) == ("runtime.queue_maxsize", 7)
    text = path.read_text(encoding="utf-8")
    assert text.index("queue_maxsize = 7") < text.index("[optimization]")
    assert set_config_value("optimize_auto", '"medium"', path) == ("optimization.auto", "medium")
    assert get_config_value(path, "auto") == "medium"
    assert get_config_value(path, "optimization.auto") == "medium"
    assert "optimize_auto =" not in path.read_text(encoding="utf-8")


def test_unknown_and_wrong_section_edits_fail_without_altering_source(tmp_path):
    path = tmp_path / "yaw.toml"
    original = b"[runtime]\nqueue_maxsize = 4\n"
    path.write_bytes(original)
    for key in ("unknown", "runtime.unknown", "runtime.chunk_size", "graph.linking_top_k"):
        with pytest.raises(YawRuntimeError):
            set_config_value(key, "42", path)
        assert path.read_bytes() == original
    with pytest.raises(YawRuntimeError, match="Unknown"):
        get_config_value(path, "runtime.unknown")
    with pytest.raises(YawRuntimeError, match="not found"):
        set_config_value("queue_maxsize", "4", tmp_path / "absent.toml")
    with pytest.raises(YawRuntimeError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_empty_last_section_can_receive_a_setting_and_plain_keys_return_native_values(tmp_path):
    path = tmp_path / "yaw.toml"
    path.write_text("[runtime]\n# Insert below this comment\n", encoding="utf-8")
    assert set_config_value("queue_maxsize", "8", path) == ("runtime.queue_maxsize", 8)
    assert get_config_value(path, "queue_maxsize") == 8
    assert get_config_value(path, "workspace_root") == str(tmp_path)
    assert get_config_value(path, "optimize_datasets") == ["enterprise_rag_bench"]


@pytest.mark.parametrize(
    ("key", "literal"),
    [
        ("optimize_datasets", '["popqa", true]'),
        ("optimize_datasets", '["popqa", 1.5]'),
        ("included_patterns", "[1, 2]"),
        ("optimize_datasets", "[]"),
        ("optimize_datasets", '""'),
    ],
)
def test_set_config_value_rejects_non_string_and_empty_sequence_literals_without_mutating_file(
    tmp_path, key: str, literal: str
) -> None:
    path = tmp_path / "yaw.toml"
    original = b'[optimization]\ndatasets = ["enterprise_rag_bench"]\n'
    path.write_bytes(original)
    with pytest.raises(YawRuntimeError) as exc_info:
        set_config_value(key, literal, path)
    assert exc_info.value.code == "INVALID_INPUT"
    assert path.read_bytes() == original


def test_set_config_value_replaces_multiline_toml_array_cleanly(tmp_path) -> None:
    src_toml = Path(__file__).resolve().parents[2] / "yaw.toml"
    copy_path = tmp_path / "yaw.toml"
    shutil.copyfile(src_toml, copy_path)
    assert "included_patterns = [\n" in copy_path.read_text(encoding="utf-8")

    updated_key, updated_val = set_config_value("included_patterns", '["*.md", "*.py"]', copy_path)
    assert updated_key == "ingestion.included_patterns"
    assert updated_val == ["*.md", "*.py"]
    loaded = load_config(copy_path)
    assert loaded.included_patterns == ("*.md", "*.py")
    assert loaded.excluded_patterns == (".*", "*/.*", "__pycache__/*")
    assert loaded.chunk_size == 1000

    bracket_edge = tmp_path / "edge.toml"
    bracket_edge.write_text(
        '[ingestion]\nincluded_patterns = [\n  "foo]bar",\n  # comment with ]\n  "*.txt",\n]\nchunk_size = 500\n',
        encoding="utf-8",
    )
    set_config_value("included_patterns", '["*.md"]', bracket_edge)
    edge_cfg = load_config(bracket_edge)
    assert edge_cfg.included_patterns == ("*.md",)
    assert edge_cfg.chunk_size == 500
