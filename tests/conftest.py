"""Keep routine verification independent of resident inference services."""

import os

import pytest

# Python 3.14's default sysmon core cannot retain testmon's per-test contexts.
os.environ.setdefault("COVERAGE_CORE", "ctrace")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="Run resident-model inference tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="Resident-model test; explicitly enable with --live")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip)
