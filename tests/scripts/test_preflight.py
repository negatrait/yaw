"""Exercise the real toolchain and durable reports in disposable projects."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_preflight_reports_all_failures_and_reuses_testmon(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n', encoding="utf-8")
    (tmp_path / "yaw").mkdir()
    (tmp_path / "yaw" / "measured.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "benchmark.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts" / "measured.py").write_text("value = 3\n", encoding="utf-8")
    subject = tmp_path / "subject.py"
    subject.write_text(
        'def unused():\n    return undefined_name\n\ndef value() -> int:\n    return "bad"\n',
        encoding="utf-8",
    )
    test = tmp_path / "test_subject.py"
    test.write_text(
        "import runpy\nfrom subject import value\ndef test_value():\n"
        '    assert value() == "bad"\n'
        '    assert runpy.run_path("yaw/measured.py")["value"] == 1\n'
        '    assert runpy.run_path("benchmark.py")["value"] == 2\n'
        '    assert runpy.run_path("scripts/measured.py")["value"] == 3\n',
        encoding="utf-8",
    )
    output = tmp_path / "reports"
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONIOENCODING": "utf-8"}

    def invoke(*args):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scripts.preflight import main; raise SystemExit(main())",
                "--output-dir",
                str(output),
                *args,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            encoding="utf-8",
            timeout=90,
        )
        pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
        report = json.loads(Path(pointer["summary"]).read_text(encoding="utf-8"))
        return completed, report

    completed, report = invoke("--full")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert [check["status"] for check in report["checks"]] == [
        "passed",
        "failed",
        "failed",
        "passed",
    ]
    assert report["tests"] == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    assert "coverage" in report
    assert report["coverage"]["covered_lines"] >= 1
    pointer = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    coverage = json.loads(Path(pointer["summary"]).with_name("coverage.json").read_text(encoding="utf-8"))
    measured = {name.replace("\\", "/") for name in coverage["files"]}
    assert {"yaw/measured.py", "benchmark.py", "scripts/measured.py"} <= measured
    assert all(
        data["summary"]["covered_lines"] == 1
        for name, data in coverage["files"].items()
        if name.replace("\\", "/") in {"yaw/measured.py", "benchmark.py", "scripts/measured.py"}
    )
    assert (output / ".testmondata").exists()

    subject.write_text("def value() -> int:\n    return 7\n", encoding="utf-8")
    test.write_text(
        "import runpy\nfrom subject import value\n\n\ndef test_value():\n"
        "    assert value() == 7\n"
        '    assert runpy.run_path("yaw/measured.py")["value"] == 1\n'
        '    assert runpy.run_path("benchmark.py")["value"] == 2\n'
        '    assert runpy.run_path("scripts/measured.py")["value"] == 3\n',
        encoding="utf-8",
    )
    completed, report = invoke()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert report["mode"] == "incremental"
    assert report["tests"]["tests"] == 1
    assert all(check["status"] == "passed" for check in report["checks"])

    completed, report = invoke()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert report["tests"]["tests"] == 0  # Unchanged successful tests are genuinely deselected.
