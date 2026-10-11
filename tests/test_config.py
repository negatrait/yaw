"""Tests for configuration loader and constants (mirrors src/yaw/config.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
import spacy.util

from yaw import config
from yaw.server.daemon import get_ontology_schema


def test_config_ssot_defaults() -> None:
    """Verify authoritative configuration constants and public API in yaw.config."""
    assert config.SPACY_MODEL == "en_core_web_lg"
    assert config.CANON_SIM_THRESHOLD == 0.80
    assert config.SPACY_CACHE_DIR == Path(".cache/spacy_llm")
    assert "SPACY_MODEL" in config.__all__
    assert "CANON_SIM_THRESHOLD" in config.__all__
    assert "SPACY_CACHE_DIR" in config.__all__
    assert "get_spacy_model_name" in config.__all__


def test_config_base_predicates_ssot_purges_is_a() -> None:
    """Verify config.BASE_PREDICATES contains the 14 ADR-008 predicates and purges is_a."""
    assert len(config.BASE_PREDICATES) == 14
    assert "is_a" not in config.BASE_PREDICATES
    expected_predicates = {
        "partOf",
        "subClassOf",
        "manifestsIn",
        "mentions",
        "causes",
        "prevents",
        "supports",
        "refutes",
        "contradicts",
        "supersedes",
        "produces",
        "derivedFrom",
        "cites",
        "instanceOf",
    }
    assert set(config.BASE_PREDICATES) == expected_predicates

    schema = get_ontology_schema()
    assert "is_a" not in schema["base_predicates"]
    assert len(schema["base_predicates"]) == 14


def test_get_spacy_model_name_installed_model() -> None:
    """Explicitly requesting an installed model returns it immediately."""
    assert config.get_spacy_model_name("en_core_web_lg") == "en_core_web_lg"
    assert config.get_spacy_model_name() == "en_core_web_lg"


def test_get_spacy_model_name_uninstalled_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RuntimeError is raised when requested model is not installed."""
    installed_packages: set[str] = set()

    def mock_is_package(name: str) -> bool:
        return name in installed_packages

    monkeypatch.setattr(spacy.util, "is_package", mock_is_package)

    # Case A: target is installed
    installed_packages = {"en_core_web_lg"}
    assert config.get_spacy_model_name("en_core_web_lg") == "en_core_web_lg"

    # Case B: target missing -> raises RuntimeError per FOUNDATIONAL.md
    installed_packages = {"en_core_web_md"}
    with pytest.raises(RuntimeError, match="Authoritative spaCy model"):
        config.get_spacy_model_name("en_core_web_lg")

    # Case C: None installed at all -> raises RuntimeError
    installed_packages = set()
    with pytest.raises(RuntimeError, match="Authoritative spaCy model"):
        config.get_spacy_model_name("en_core_web_lg")


def test_get_spacy_model_name_edge_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test unusual and degenerate inputs to get_spacy_model_name raise RuntimeError."""
    # Falsy inputs fall back to SPACY_MODEL default ("en_core_web_lg")
    assert config.get_spacy_model_name("") == "en_core_web_lg"
    assert config.get_spacy_model_name(None) == "en_core_web_lg"

    # Non-empty uninstalled models raise RuntimeError
    for invalid in (
        "   ",
        "!@#$%^&*()",
        "DROP TABLE models;--",
        "модель_рус",
        "nonexistent",
    ):
        with pytest.raises(RuntimeError, match="Authoritative spaCy model"):
            config.get_spacy_model_name(invalid)

    monkeypatch.setattr(config, "SPACY_MODEL", "nonexistent_custom_model")
    with pytest.raises(RuntimeError, match="Authoritative spaCy model"):
        config.get_spacy_model_name(None)


def test_get_spacy_model_name_resilience(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RuntimeError when spacy.util raises AttributeError or is missing."""
    import sys

    class CrashingUtil:
        def is_package(self, _name: str) -> bool:
            raise AttributeError("Simulated util AttributeError")

    monkeypatch.setattr(spacy, "util", CrashingUtil())
    with pytest.raises(RuntimeError, match="Authoritative spaCy model"):
        config.get_spacy_model_name()

    monkeypatch.setitem(sys.modules, "spacy.util", None)
    with pytest.raises(RuntimeError, match="Authoritative spaCy model"):
        config.get_spacy_model_name()


def test_set_vault_and_port(tmp_path: Path) -> None:
    """Verify set_vault updates paths and set_port updates server port."""
    vault = tmp_path / "custom_vault"
    vault.mkdir()
    config.set_vault(vault)
    assert config.VAULT_ROOT == vault.resolve()
    assert config.DB_PATH == (vault / ".cache" / "yaw.db").resolve()

    config.set_port(9999)
    assert config.HTTP_PORT == 9999
