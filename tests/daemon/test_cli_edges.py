"""CLI status and reload protocols using real locks, SQLite and signal files."""

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import psutil
import pytest
from filelock import FileLock

from yaw.core import YawConfig, YawRuntimeError, load_config
from yaw.daemon.cli import (
    handle_daemon_start,
    handle_daemon_stop,
    handle_dspy_load_program,
    handle_graph_lint,
    inspect_status,
    main,
)
from yaw.daemon.graph.inference import HippoRAGProgram
from yaw.daemon.graph.storage import open_yaw_db


def configured_workspace(tmp_path) -> YawConfig:
    config = YawConfig(workspace_root=tmp_path)
    assert config.lock_path is not None
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    return config


def seed_openie(path, count):
    with closing(open_yaw_db(path)) as conn:
        conn.executemany(
            "INSERT INTO openie_chunks(chunk_id,entities_json,triples_json) VALUES (?,'[]','[]')",
            [(f"source-{i}",) for i in range(count)],
        )


def test_status_reads_only_held_optimizer_locks_and_merges_durable_progress(tmp_path):
    config = configured_workspace(tmp_path)
    assert config.lock_path is not None and config.status_path is not None
    folder = config.lock_path.parent
    free = FileLock(str(folder / "yaw_optimize_unused.db.lock"))
    with free:
        pass
    seed_openie(config.db_path, 2)
    seed_openie(folder / "yaw_optimize_test.db", 3)
    (folder / "status_optimize_test.json").write_text(
        json.dumps(
            {
                "status": {"active_op": "reflecting"},
                "stats": {"openie_checkpoints": 4},
            }
        ),
        encoding="utf-8",
    )
    with FileLock(str(folder / "yaw_optimize_test.db.lock")):
        stopped = inspect_status(config)
        assert stopped["status"]["lifecycle"] == "stopped"
        assert stopped["active_optimization"]["datestamp"] == "test"
        assert stopped["active_optimization"]["openie_checkpoints"] == 4
        assert stopped["status"]["active_op"] == "optimization active in sandbox: reflecting"
        config.status_path.write_text(
            json.dumps(
                {
                    "status": {"lifecycle": "ready", "active_op": None},
                    "stats": {"chunks": 0},
                }
            ),
            encoding="utf-8",
        )
        with FileLock(str(config.lock_path)):
            running = inspect_status(config)
        assert running["status"]["active_op"] == "indexing: 2 openie passages processed"
        assert running["status"]["sync_state"] == "syncing"
        assert running["stats"]["openie_checkpoints"] == 2
        assert running["stats"]["shacl_conforms"] is True
        assert running["active_optimization"]["active_op"] == "reflecting"


@pytest.mark.parametrize(
    "content",
    [
        "broken JSON",
        "[]",
        '{"status":[],"stats":[]}',
        '{"status":{"active_op":7},"stats":{"openie_checkpoints":"many"}}',
    ],
)
def test_status_survives_corrupt_optional_status_and_database_files(tmp_path, content):
    config = configured_workspace(tmp_path)
    assert config.lock_path is not None and config.db_path is not None and config.status_path is not None
    config.db_path.write_bytes(b"not SQLite")
    config.status_path.write_text(content, encoding="utf-8")
    with FileLock(str(config.lock_path)), FileLock(str(config.lock_path.parent / "yaw_optimize_broken.db.lock")):
        (config.lock_path.parent / "status_optimize_broken.json").write_text(content, encoding="utf-8")
        status = inspect_status(config)
    assert status["status"]["lifecycle"] == "ready"
    assert status["active_optimization"]["active_op"] == "optimizing"
    assert status["active_optimization"]["openie_checkpoints"] == 0


def test_start_of_already_locked_daemon_and_stop_of_stopped_daemon_are_idempotent(tmp_path, capsys):
    config = configured_workspace(tmp_path)
    with FileLock(str(config.lock_path)):
        assert handle_daemon_start(config, None) == 0
    assert json.loads(capsys.readouterr().out)["already_running"] is True
    assert handle_daemon_stop(config) == 0
    assert json.loads(capsys.readouterr().out) == {"stopped": True, "was_running": False}


@pytest.mark.parametrize("mode", ["success", "rejected", "timeout"])
def test_remote_reload_correlates_actual_signal_acknowledgements(tmp_path, capsys, mode):
    config = configured_workspace(tmp_path)
    assert config.compiled_program_path is not None
    source = tmp_path / "candidate.json"
    HippoRAGProgram(include_demos=False).save(str(source))
    folder = tmp_path / ".yaw"
    signal_file = folder / "reload.signal"
    result_file = folder / "reload.result"
    result_file.write_text('{"req_id":"stale-before-request"}', encoding="utf-8")

    def acknowledge():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if signal_file.exists():
                try:
                    request = json.loads(signal_file.read_text(encoding="utf-8"))
                except OSError, json.JSONDecodeError:
                    time.sleep(0.01)
                    continue
                result_file.write_text('{"req_id":"other-request"}', encoding="utf-8")
                time.sleep(0.15)
                result_file.write_text("incomplete JSON", encoding="utf-8")
                time.sleep(0.15)
                result_file.write_text(
                    json.dumps(
                        {
                            "req_id": request["req_id"],
                            "reloaded": mode == "success",
                        }
                    ),
                    encoding="utf-8",
                )
                return request
            time.sleep(0.01)
        raise AssertionError("The CLI did not write its reload request")

    with FileLock(str(config.lock_path)), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(acknowledge) if mode != "timeout" else None
        result = handle_dspy_load_program(config, source)
        response = json.loads(capsys.readouterr().out)
        if future is not None:
            request = future.result(timeout=5)
            assert response["req_id"] == request["req_id"]
            assert response["reloaded"] is (mode == "success")
            assert result == (0 if mode == "success" else 1)
        else:
            assert result == 0 and response["staged"] is True and response["reloaded"] is False
    assert config.compiled_program_path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("content", ["not JSON", "[]"])
def test_program_load_rejects_invalid_json_before_promoting_files(tmp_path, content):
    config = configured_workspace(tmp_path)
    assert config.compiled_program_path is not None
    source = tmp_path / "candidate.json"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(YawRuntimeError, match="Invalid compiled program JSON"):
        handle_dspy_load_program(config, source)
    assert not config.compiled_program_path.exists()


def test_graph_lint_of_missing_or_uninitialized_database_uses_empty_schema(tmp_path, capsys):
    config = configured_workspace(tmp_path)
    assert handle_graph_lint(config) == 0
    assert json.loads(capsys.readouterr().out)["conforms"] is True
    with closing(open_yaw_db(config.db_path)) as conn:
        conn.execute("DROP TABLE entities")
    assert handle_graph_lint(config, as_turtle=True) == 0
    assert "@prefix" in capsys.readouterr().out


def test_cli_requires_a_complete_subcommand(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize("bad_count", [-1, "pending", True, [1]])
def test_daemon_status_normalizes_bad_counters_against_sqlite(tmp_path, bad_count):
    config = configured_workspace(tmp_path)
    assert config.lock_path is not None and config.status_path is not None
    seed_openie(config.db_path, 2)
    config.status_path.write_text(
        json.dumps(
            {
                "status": {"lifecycle": "ready", "active_op": 7},
                "stats": {"chunks": bad_count, "openie_checkpoints": bad_count},
            }
        ),
        encoding="utf-8",
    )
    with FileLock(str(config.lock_path)):
        result = inspect_status(config)
    assert result["stats"]["chunks"] == 0
    assert result["stats"]["openie_checkpoints"] == 2
    assert result["status"]["active_op"] == "indexing: 2 openie passages processed"


def test_detached_empty_daemon_starts_and_stops_without_leaking_process_handles(tmp_path, capsys):
    config_file = tmp_path / "yaw.toml"
    config_file.write_text(
        f"[workspace]\nworkspace_root={json.dumps(tmp_path.as_posix())}\n"
        '[inference]\nbase_url="http://127.0.0.1:9/v1"\nenable_gbnf=false\n'
        '[ingestion]\nincluded_patterns=["*.txt"]\npoll_interval_seconds=0.1\n',
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.corpus_root is not None and config.status_path is not None
    config.corpus_root.mkdir(parents=True)
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always", ResourceWarning)
            assert handle_daemon_start(config, config_file) == 0
            started = json.loads(capsys.readouterr().out)
            assert started["started_pid"] > 0
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    status = json.loads(config.status_path.read_text(encoding="utf-8"))
                except OSError, json.JSONDecodeError:
                    time.sleep(0.05)
                    continue
                if status["status"]["lifecycle"] == "ready":
                    break
                time.sleep(0.05)
            else:
                pytest.fail("Owned empty daemon did not reach readiness")
            assert handle_daemon_stop(config) == 0
            assert json.loads(capsys.readouterr().out)["stopped"] is True
            assert inspect_status(config)["status"]["lifecycle"] == "stopped"
        assert not [notice for notice in notices if issubclass(notice.category, ResourceWarning)]
    finally:
        # Match both the unique config path and workspace before touching any PID.
        for process in psutil.process_iter(["cmdline"]):
            try:
                if str(config_file) in (process.info["cmdline"] or []) and process.cwd() == str(tmp_path):
                    process.terminate()
                    process.wait(timeout=5)
            except psutil.NoSuchProcess:
                continue


def test_cli_canonical_dspy_benchmark_handler_and_config_set_no_abbrev():
    from yaw.daemon import cli

    assert callable(cli.handle_dspy_benchmark)
    assert not hasattr(cli, "handle_benchmark")
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["config", "set", "--unknown-abbreviated-flag", "inference.enable_gbnf", "false"])
