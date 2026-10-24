"""Shared verification settings; every collected test runs by default."""

import os

import pytest

# Python 3.14's default sysmon core cannot retain testmon's per-test contexts.
os.environ.setdefault("COVERAGE_CORE", "ctrace")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="Compatibility option; resident-model tests always run")
