"""Serial development checks with durable, honest reports and incremental tests."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree


def run_check(
    name: str,
    command: list[str],
    root: Path,
    output: Path,
    env: dict[str, str],
    timeout: float,
    verbose: bool = False,
) -> dict:
    """Stream a child to its own log; keep checking after ordinary failures."""
    started = time.monotonic()
    log = output / f"{name}.log"
    result = {"name": name, "command": command, "log": str(log)}
    print(f"\n[{name}] Starting", flush=True)
    try:
        with log.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            def stream() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    if verbose:
                        print(line, end="", flush=True)

            reader = threading.Thread(target=stream, daemon=True)
            reader.start()
            try:
                deadline = started + timeout if timeout else None
                while True:
                    interval = min(30, max(0.01, deadline - time.monotonic())) if deadline else 30
                    try:
                        result["exit_code"] = process.wait(timeout=interval)
                        break
                    except subprocess.TimeoutExpired:
                        if deadline and time.monotonic() >= deadline:
                            raise
                        print(f"[{name}] Running ({time.monotonic() - started:.0f}s)", flush=True)
                result["status"] = "passed" if result["exit_code"] == 0 else "failed"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "timeout"
                if process.poll() is None:
                    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                result["exit_code"] = process.returncode
            finally:
                reader.join(timeout=10)
    except OSError as exc:
        result.update(status="error", exit_code=None, error=str(exc))
    result["seconds"] = round(time.monotonic() - started, 3)
    print(f"[{name}] {result['status']} ({result['seconds']:.1f}s)", flush=True)
    return result


def write_report(output: Path, report: dict) -> None:
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Preflight",
        "",
        f"Mode: {report['mode']}. Tests run serially.",
        "",
        "| Check | Status | Exit | Seconds |",
        "| --- | --- | ---: | ---: |",
    ]
    for result in report["checks"]:
        lines.append(f"| {result['name']} | {result['status']} | {result.get('exit_code')} | {result['seconds']:.1f} |")
    for name, diagnostics in report.get("diagnostics", {}).items():
        lines += [
            "",
            f"{name}: {diagnostics['count']} diagnostics.",
            f"Rules: `{json.dumps(diagnostics['rules'])}`",
        ]
    lines += [
        "",
        "Coverage describes only tests executed in this invocation. Incremental",
        "coverage is not a full-suite coverage result. Python subprocess coverage",
        "is enabled by the project .coveragerc. Use --full for a fresh complete test pass.",
        "",
    ]
    if "tests" in report:
        lines.append(f"Pytest: `{json.dumps(report['tests'])}`\n")
    if report.get("failed_tests"):
        lines += ["Failed tests:", *[f"- {name}" for name in report["failed_tests"]], ""]
    if "coverage" in report:
        lines.append(f"Coverage.py: `{json.dumps(report['coverage'])}`\n")
    if "coverage_note" in report:
        lines.append(report["coverage_note"] + "\n")
    if report.get("report_errors"):
        lines += ["Report errors:", *[f"- {error}" for error in report["report_errors"]], ""]
    lines += ["Artifacts: format.log, ty.log, ruff.log, pytest.log, pytest.xml, coverage.json.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    # Windows redirected consoles otherwise default to cp1252 and can crash
    # the output reader on Unicode diagnostics, leaving the child pipe blocked.
    configure_output = getattr(sys.stdout, "reconfigure", None)
    if callable(configure_output):
        configure_output(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Run all tests, updating testmon data")
    parser.add_argument("--verbose", action="store_true", help="Stream complete tool diagnostics")
    parser.add_argument("--live", action="store_true", help="Compatibility option; resident-model tests always run")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/preflight"))
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds per check; 0 disables")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Pytest arguments after --")
    args = parser.parse_args(argv)
    if args.timeout < 0:
        parser.error("--timeout must be non-negative")
    root = Path.cwd()
    if not (root / "pyproject.toml").is_file():
        parser.error("Run preflight from the repository root")
    base = args.output_dir.resolve()
    output = base / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}")
    output.mkdir(parents=True)
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
        "COVERAGE_CORE": "ctrace",
        "COVERAGE_FILE": str(output / ".coverage"),
        "TESTMON_DATAFILE": str(base / ".testmondata"),
    }
    # coverage.py sysmon cannot supply testmon's per-test dynamic contexts.
    # pytest-cov and testmon explicitly cooperate to share coverage.py line data.
    lock = root / "uv.lock"
    identity = hashlib.sha256(lock.read_bytes() if lock.exists() else b"").hexdigest()[:16]
    mode = "full" if args.full else "incremental"
    report = {
        "mode": mode,
        "live": True,
        "checks": [],
        "python": sys.version,
        "versions": {
            name: version(name) for name in ("ty", "ruff", "pytest", "pytest-testmon", "coverage", "pytest-cov")
        },
        "lock_identity": identity,
    }
    extra = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    report["pytest_args"] = extra
    pytest = [
        sys.executable,
        "-m",
        "pytest",
        "--testmon",
        "--cov=yaw",
        "--cov=benchmark",
        "--cov=scripts",
        "--cov-report=term",
        f"--cov-report=json:{output / 'coverage.json'}",
        f"--testmon-env=preflight-{identity}-all-tests",
        "-ra",
        "--tb=short",
        "--durations=10",
        f"--junitxml={output / 'pytest.xml'}",
    ]
    if args.full:
        pytest.append("--testmon-noselect")
    if args.live:
        pytest.append("--live")
    pytest.extend(extra)
    commands = [
        ("format", [sys.executable, "-m", "ruff", "format", "."]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "--fix", "."]),
        ("ty", [sys.executable, "-m", "ty", "check", "--fix", "--output-format", "concise"]),
        ("pytest", pytest),
    ]
    print(f"Preflight: {mode}; reports in {output}", flush=True)
    for name, command in commands:
        result = run_check(name, command, root, output, env, args.timeout, args.verbose)
        report["checks"].append(result)
        write_report(output, report)
        if result["status"] == "interrupted":
            break
    report["diagnostics"] = {}
    for name, pattern in (
        ("ruff", r"^([A-Z]+\d+) "),
        ("ty", r"^.+?:\d+:\d+: (?:error|warning)\[([^]]+)\]"),
    ):
        log = output / f"{name}.log"
        if log.exists():
            counts = Counter(re.findall(pattern, log.read_text(encoding="utf-8"), re.MULTILINE))
            report["diagnostics"][name] = {"count": sum(counts.values()), "rules": dict(counts)}
    report["report_errors"] = []
    test_count = None
    try:
        junit = output / "pytest.xml"
        if junit.exists():
            tree = ElementTree.parse(junit).getroot()
            cases = list(tree.iter("testcase"))
            # Some plugin combinations inflate suite header counts. Actual case
            # records agree with pytest's terminal totals and remain auditable.
            test_counts = {
                "tests": len(cases),
                "failures": sum(test.find("failure") is not None for test in cases),
                "errors": sum(test.find("error") is not None for test in cases),
                "skipped": sum(test.find("skipped") is not None for test in cases),
            }
            report["tests"] = test_counts
            test_count = test_counts["tests"]
            report["failed_tests"] = [
                f"{test.get('classname', '')}.{test.get('name', '')}"
                for test in cases
                if test.find("failure") is not None or test.find("error") is not None
            ]
        else:
            report["report_errors"].append("Pytest did not produce its JUnit report")
        coverage = output / "coverage.json"
        if coverage.exists():
            report["coverage"] = json.loads(coverage.read_text(encoding="utf-8"))["totals"]
        elif test_count == 0 and any(
            check["name"] == "pytest" and check["status"] == "passed" for check in report["checks"]
        ):
            report["coverage_note"] = "No tests selected; coverage was not measured in this run."
        else:
            report["report_errors"].append("Coverage.py did not produce its JSON report")
    except (OSError, ElementTree.ParseError, json.JSONDecodeError, KeyError, ValueError) as exc:
        report["report_errors"].append(f"Cannot read test/coverage report: {exc}")
    write_report(output, report)
    (base / "latest.json").write_text(
        json.dumps({"report": str(output / "report.md"), "summary": str(output / "summary.json")}) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport: {output / 'report.md'}", flush=True)
    for name, diagnostics in report["diagnostics"].items():
        print(f"{name}: {diagnostics['count']} remaining diagnostics", flush=True)
    if "tests" in report:
        print(f"Pytest: {json.dumps(report['tests'])}", flush=True)
    clean = all(check["status"] == "passed" for check in report["checks"])
    return 0 if clean and not report["report_errors"] else 1


def typecheck() -> int:
    """Project command for ty with fixes enabled; ty has no persistent fix setting."""
    return subprocess.call([sys.executable, "-m", "ty", "check", "--fix", *sys.argv[1:]])
