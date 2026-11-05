"""Shared verification settings; every collected test runs by default."""

import os
from pathlib import Path

import pytest

# Python 3.14's default sysmon core cannot retain testmon's per-test contexts.
os.environ.setdefault("COVERAGE_CORE", "ctrace")

# Prevent LiteLLM from making outbound network calls to GitHub for model prices
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_TELEMETRY", "False")

# Use repository-cached Deno runtime directory when available
workspace_deno = Path(__file__).resolve().parents[1] / ".yaw" / "deno_dir"
if workspace_deno.exists():
    os.environ.setdefault("DENO_DIR", str(workspace_deno))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="Compatibility option; resident-model tests always run")
