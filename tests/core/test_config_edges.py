"""Real TOML round trips, rejected edits and deployment-capacity boundaries."""

import json

import pytest

from yaw.core import YawConfig, YawRuntimeError, get_config_value, load_config, set_config_value
from yaw.core.config import context_char_limit


@pytest.mark.parametrize("output", [0, -1, 32, 33])
def test_character_budget_rejects_impossible_context_allowances(output):
    with pytest.raises(ValueError, match="Require"):
        context_char_limit(100, 32, output)


def test_invalid_completion_capacity_is_rejected_by_the_configuration():
    with pytest.raises(YawRuntimeError, match="max_output_tokens"):
        YawConfig(context_window_tokens=4096, max_output_tokens=4096)


def test_loading_defaults_and_explicit_workspace_selection(tmp_path):
    assert load_config(workspace_override=tmp_path).workspace_root == tmp_path
    path = tmp_path / "yaw.toml"
    path.write_text('[paths]\nworkspace_root = "nested"\n', encoding="utf-8")
    assert load_config(workspace_override=tmp_path).workspace_root == tmp_path
    assert load_config(path).workspace_root == tmp_path / "nested"
    path.write_text("[graph\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError, match="Invalid TOML"):
        load_config(path)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [('"alpha, beta, "', ("alpha", "beta")), ('["alpha", " beta "]', ("alpha", "beta")), ("42", ("42",))],
)
def test_dataset_forms_load_into_stable_tuples(tmp_path, literal, expected):
    path = tmp_path / "yaw.toml"
    path.write_text(f"[optimization]\ndatasets = {literal}\n", encoding="utf-8")
    assert load_config(path).optimize_datasets == expected


def test_constructor_normalizes_comma_separated_filters_and_dataset_names(tmp_path):
    values = json.loads(
        '{"optimize_datasets":"a, b, ", "json_keys_whitelist":"title, text","json_keys_blacklist":"answer, question"}'
    )
    config = YawConfig(workspace_root=tmp_path, **values)
    assert config.optimize_datasets == ("a", "b")
    assert config.json_keys_whitelist == ("title", "text")
    assert config.json_keys_blacklist == ("answer", "question")


@pytest.mark.parametrize(
    ("key", "literal", "expected"),
    [
        ("inference.enable_gbnf", "false", False),
        ("runtime.indexing_overlap", "true", True),
        ("inference.enable_gbnf", "TRUE", True),
        ("optimization.sleep_seconds", "1.25", 1.25),
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


def test_inserted_settings_preserve_section_boundaries_and_legacy_aliases(tmp_path):
    path = tmp_path / "yaw.toml"
    path.write_text('[runtime]\n# Existing section\n[optimization]\noptimize_mode = "gepa"\n', encoding="utf-8")
    assert set_config_value("queue_maxsize", "7", path) == ("runtime.queue_maxsize", 7)
    text = path.read_text(encoding="utf-8")
    assert text.index("queue_maxsize = 7") < text.index("[optimization]")
    assert set_config_value("optimize_mode", '"hybrid"', path) == ("optimization.mode", "hybrid")
    assert get_config_value(path, "mode") == "hybrid"
    assert get_config_value(path, "optimization.mode") == "hybrid"
    assert "optimize_mode =" not in path.read_text(encoding="utf-8")


def test_unknown_configuration_edits_fail_without_altering_source(tmp_path):
    path = tmp_path / "yaw.toml"
    original = b"[runtime]\nqueue_maxsize = 4\n"
    path.write_bytes(original)
    for key in ("unknown", "runtime.unknown"):
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
    assert get_config_value(path, "optimize_datasets") == ["2wikimultihopqa"]
