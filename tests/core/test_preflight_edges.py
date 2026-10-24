"""Real tool output, timeout cleanup and durable preflight-report contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from yaw.core.preflight import main, run_check, typecheck, write_report


def test_run_check_retains_unicode_output_and_nonzero_exit(tmp_path, capsys):
    command = [sys.executable, "-c", "print('Aster — Åä'); raise SystemExit(3)"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = run_check("failure", command, tmp_path, tmp_path, env, 10, verbose=True)
    assert result["status"] == "failed"
    assert result["exit_code"] == 3
    assert "Aster — Åä" in Path(result["log"]).read_text(encoding="utf-8")
    assert "Aster — Åä" in capsys.readouterr().out


def test_run_check_reports_an_unlaunchable_executable(tmp_path):
    result = run_check("absent", [str(tmp_path / "missing.exe")], tmp_path, tmp_path, dict(os.environ), 1)
    assert result["status"] == "error"
    assert result["exit_code"] is None
    assert result["error"]


def test_run_check_timeout_is_reported_and_owned_process_finishes(tmp_path):
    command = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(3)"]
    result = run_check("timeout", command, tmp_path, tmp_path, dict(os.environ), 0.5)
    assert result["status"] == "timeout"
    assert result["exit_code"] is not None


def test_run_check_emits_a_real_heartbeat_for_a_long_running_child(tmp_path, capsys):
    command = [sys.executable, "-c", "import time; time.sleep(30.2)"]
    result = run_check("heartbeat", command, tmp_path, tmp_path, dict(os.environ), 0)
    assert result["status"] == "passed"
    assert "[heartbeat] Running" in capsys.readouterr().out


def test_run_check_force_stops_an_owned_child_ignoring_interrupts(tmp_path):
    marker = tmp_path / "owned-child.pid"
    script = (
        "import os,signal,sys,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "if hasattr(signal, 'SIGBREAK'): signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    child_pid = None
    try:
        result = run_check(
            "stubborn", [sys.executable, "-c", script, str(marker)], tmp_path, tmp_path, dict(os.environ), 5
        )
        child_pid = int(marker.read_text(encoding="utf-8"))
        assert result["status"] == "timeout"
        assert result["exit_code"] is not None
        assert not psutil.pid_exists(child_pid)
    finally:
        # Only this test's positively identified child is eligible for cleanup.
        if child_pid is not None:
            try:
                child = psutil.Process(child_pid)
                if str(marker) in child.cmdline():
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


def test_report_keeps_failure_evidence_and_explicit_coverage_limit(tmp_path):
    report = {
        "mode": "incremental",
        "checks": [{"name": "pytest", "status": "failed", "exit_code": 1, "seconds": 1.2}],
        "diagnostics": {"ruff": {"count": 1, "rules": {"F821": 1}}},
        "tests": {"tests": 1, "failures": 1, "errors": 0, "skipped": 0},
        "failed_tests": ["sample.test_failure"],
        "coverage_note": "No complete-suite coverage was measured.",
        "report_errors": ["Missing coverage data"],
    }
    write_report(tmp_path, report)
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == report
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "sample.test_failure" in text
    assert "No complete-suite coverage was measured." in text
    assert "Missing coverage data" in text


def test_preflight_rejects_invalid_timeout_and_missing_project(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exited:
        main(["--timeout", "-1"])
    assert exited.value.code == 2
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exited:
        main([])
    assert exited.value.code == 2


def test_typecheck_entrypoint_uses_the_real_typechecker_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["typecheck", "--help"])
    assert typecheck() == 0


@pytest.mark.parametrize("mode", ["live", "no-coverage", "no-junit", "malformed-junit", "collection-only"])
def test_preflight_accounts_for_actual_tool_reports(tmp_path, monkeypatch, mode):
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n', encoding="utf-8")
    (tmp_path / "yaw").mkdir()
    (tmp_path / "yaw" / "measured.py").write_text("value = 7\n", encoding="utf-8")
    (tmp_path / "test_value.py").write_text(
        'import runpy\n\ndef test_value():\n    assert runpy.run_path("yaw/measured.py")["value"] == 7\n',
        encoding="utf-8",
    )
    conftest = 'def pytest_addoption(parser):\n    parser.addoption("--live", action="store_true")\n'
    if mode == "malformed-junit":
        conftest = (
            "import os\nfrom pathlib import Path\nimport pytest\n"
            "@pytest.hookimpl(trylast=True)\n"
            "def pytest_terminal_summary():\n"
            '    target = Path(os.environ["COVERAGE_FILE"]).with_name("pytest.xml")\n'
            '    target.write_text("invalid XML", encoding="utf-8")\n'
        )
    (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "reports"
    arguments = ["--output-dir", str(output), "--timeout", "0"]
    if mode == "live":
        arguments.extend(["--live", "--verbose"])
    elif mode == "no-coverage":
        arguments.extend(["--", "--no-cov"])
    elif mode == "no-junit":
        arguments.extend(["--", f"--junitxml={tmp_path / 'elsewhere.xml'}"])
    elif mode == "collection-only":
        arguments.extend(["--", "--collect-only", "--no-cov"])
    result = main(arguments)
    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    report = json.loads(Path(pointer["summary"]).read_text(encoding="utf-8"))
    if mode == "live":
        assert result == 0
        assert report["live"] is True
        assert report["tests"]["tests"] == 1
    elif mode == "collection-only":
        assert result == 0
        assert report["tests"]["tests"] == 0
        assert "coverage_note" in report
    else:
        assert result == 1
        assert report["report_errors"]
        if mode == "malformed-junit":
            assert "Cannot read test/coverage report" in report["report_errors"][0]


def test_interrupted_preflight_persists_a_partial_report_in_an_owned_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
    ready = tmp_path / "pytest-ready.pid"
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n', encoding="utf-8")
    (tmp_path / "yaw").mkdir()
    (tmp_path / "yaw" / "measured.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "test_wait.py").write_text(
        "import os,time\nfrom pathlib import Path\n\ndef test_wait():\n"
        f"    Path({str(ready)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "    time.sleep(45)\n",
        encoding="utf-8",
    )
    script = """
import signal,sys,threading,time
from pathlib import Path
from yaw.core.preflight import main
ready=Path(sys.argv[1])
def interrupt_when_test_runs():
    deadline=time.monotonic()+20
    while not ready.exists():
        if time.monotonic()>deadline:
            raise RuntimeError('Owned child never reached its test')
        time.sleep(0.01)
    signal.raise_signal(signal.SIGINT)
threading.Thread(target=interrupt_when_test_runs,daemon=True).start()
raise SystemExit(main(['--output-dir',sys.argv[2],'--timeout','90']))
"""
    output = tmp_path / "reports"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(ready), str(output)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(root), "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            encoding="utf-8",
            timeout=90,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
        report = json.loads(Path(pointer["summary"]).read_text(encoding="utf-8"))
        assert report["checks"][-1]["name"] == "pytest"
        assert report["checks"][-1]["status"] == "interrupted"
    finally:
        if ready.exists():
            try:
                child = psutil.Process(int(ready.read_text(encoding="utf-8")))
                if Path(child.cwd()) == tmp_path and "pytest" in " ".join(child.cmdline()):
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
