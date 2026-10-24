"""Configuration and logging failures against real files and native contracts."""

import importlib
import json
import logging
import runpy
import sys

import pytest

from yaw.core import SHACLViolation, StructuredTriple, YawConfig, YawRuntimeError, load_config
from yaw.core.logger import FIFOLineRotatingHandler, SuppressThirdPartyInfoFilter


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("runtime", "queue_maxsize", "0", "queue_maxsize"),
        ("ingestion", "chunk_size", "0", "chunk_size"),
        ("inference", "max_context_chars", "0", "max_context_chars"),
        ("graph", "max_context_chars", "0", "max_context_chars"),
        ("inference", "rlm_max_llm_calls", "0", "rlm_max_llm_calls"),
        ("inference", "rlm_max_output_chars", "0", "rlm_max_output_chars"),
        ("inference", "reflection_input_tokens", "0", "reflection_input_tokens"),
        ("optimization", "mode", '"unknown"', "optimize_mode"),
        ("optimization", "auto", '"unknown"', "optimize_auto"),
    ],
)
def test_invalid_toml_settings_fail_before_runtime_start(tmp_path, section, key, value, message):
    path = tmp_path / "yaw.toml"
    path.write_text(f"[{section}]\n{key} = {value}\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError, match=message) as error:
        load_config(path)
    assert error.value.code == "INVALID_INPUT"


def test_native_data_contract_aliases_and_json():
    triple = StructuredTriple("Ada", "A mathematician", "wrote", "Note G", "An algorithm", "Ada wrote Note G.")
    assert (triple.subject, triple.predicate, triple.object) == triple.as_tuple() == ("Ada", "wrote", "Note G")
    assert json.loads(json.dumps(triple.to_dict())) == {
        "subj": "Ada",
        "subj_def": "A mathematician",
        "pred": "wrote",
        "obj": "Note G",
        "obj_def": "An algorithm",
        "spo_def": "Ada wrote Note G.",
    }
    violation = SHACLViolation("ConceptShape", "Ada", "skos:definition", "Missing definition")
    assert violation.to_dict() == {
        "shape": "ConceptShape",
        "focus_node": "Ada",
        "result_path": "skos:definition",
        "message": "Missing definition",
    }


def test_empty_and_unwritable_fifo_paths_preserve_the_log_buffer(tmp_path):
    path = tmp_path / "fifo.log"
    handler = FIFOLineRotatingHandler(path)
    try:
        handler.flush()
        assert not path.exists()
        # A directory at the atomic replacement target is a real write failure.
        path.mkdir()
        handler.emit(logging.LogRecord("yaw.test", logging.ERROR, __file__, 1, "Retain this record", (), None))
        assert path.is_dir()
        path.rmdir()
        handler.flush()
        assert path.read_text(encoding="utf-8") == "Retain this record\n"
        handler.flush()
        assert path.read_text(encoding="utf-8") == "Retain this record\n"
    finally:
        handler.close()


def test_fifo_reports_unformattable_records_without_dropping_existing_lines(tmp_path, capsys):
    path = tmp_path / "fifo.log"
    handler = FIFOLineRotatingHandler(path, flush_interval=1)
    try:
        handler.emit(logging.LogRecord("yaw.test", logging.INFO, __file__, 1, "Good", (), None))
        handler.emit(logging.LogRecord("yaw.test", logging.INFO, __file__, 2, "%d", ("invalid",), None))
        assert path.read_text(encoding="utf-8") == "Good\n"
        assert "Logging error" in capsys.readouterr().err
    finally:
        handler.close()


def test_third_party_info_is_filtered_but_warnings_and_yaw_logs_remain():
    filtering = SuppressThirdPartyInfoFilter()
    informational = logging.LogRecord("httpx.transport", logging.INFO, __file__, 1, "HTTP trace", (), None)
    litellm_info = logging.LogRecord("litellm.router", logging.INFO, __file__, 2, "Router trace", (), None)
    real_warning = logging.LogRecord("litellm.router", logging.WARNING, __file__, 3, "Provider warning", (), None)
    real_error = logging.LogRecord("httpx.transport", logging.ERROR, __file__, 4, "Provider disconnected", (), None)
    yaw_info = logging.LogRecord("yaw.daemon", logging.INFO, __file__, 5, "Daemon started", (), None)
    assert not filtering.filter(informational)
    assert not filtering.filter(litellm_info)
    assert filtering.filter(real_warning)
    assert filtering.filter(real_error)
    assert filtering.filter(yaw_info)


def test_all_leaf_facades_advertise_and_resolve_their_public_exports():
    for name in ("yaw.core", "yaw.daemon.graph.inference", "yaw.daemon.graph.storage", "yaw.daemon.ingestion"):
        module = importlib.import_module(name)
        names = dir(module)
        assert names
        exports = module.__all__
        assert set(exports).issubset(names)
        assert all(not value.startswith("_") for value in exports)
        for symbol in exports:
            assert getattr(module, symbol) is not None


def test_python_module_entrypoint_exits_successfully_for_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["yaw", "--help"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("yaw", run_name="__main__", alter_sys=True)
    assert exited.value.code == 0
    assert "daemon" in capsys.readouterr().out


def test_dynamic_config_normalization_retains_defaults_for_missing_sequences(tmp_path):
    raw = json.loads('{"optimize_datasets": null, "json_keys_whitelist": null, "json_keys_blacklist": null}')
    config = YawConfig(workspace_root=tmp_path, **raw)
    assert config.optimize_datasets == ("2wikimultihopqa",)
    assert config.json_keys_whitelist == ("*",)
    assert config.json_keys_blacklist == ()
