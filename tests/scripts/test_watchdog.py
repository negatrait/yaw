"""Graph selection and assertion trails over real SQLite and report files."""

import sqlite3
from contextlib import closing
from datetime import UTC

from daemon.native_inference import publish_inference

from scripts import watchdog
from scripts.watchdog import (
    generate_graph_projection_svg,
    generate_report_markdown,
    largest_one_hop_neighborhood,
    recent_relation_assertions,
)
from yaw.core import OpenIERecord


def test_watchdog_reads_native_artifacts_and_empty_withdrawal(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(watchdog, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "ARTIFACT_DIR", tmp_path / "dashboard")
    path = tmp_path / ".yaw" / "optimize_yaw.db"
    records = [
        OpenIERecord(chunk_id=cid, entities=("Mercury", "Sun"), triples=(("Mercury", "orbits", "Sun"),))
        for cid in ("first", "second")
    ]
    publish_inference(path, records)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('optimization_run', ?)",
            (json.dumps({"started_at": "2026-11-08T08:00:00+00:00"}),),
        )
        conn.commit()
    populated = watchdog.get_active_sandbox_info()
    assert "error" not in populated
    assert populated["datestamp"] == "canonical"
    assert populated["start_time"] == "2026-11-08T08:00:00+00:00"
    assert populated["counts"]["chunk_inference"] == 2
    assert "openie_chunks" not in populated["counts"]
    assert populated["indexed_chunk_ids"] == {"first", "second"}
    assert populated["kg_metrics"]["total_entities"] == 4
    assert (tmp_path / ".yaw" / "optimize_telemetry.json").is_file()

    publish_inference(path, [])
    empty = watchdog.get_active_sandbox_info()
    assert "error" not in empty
    assert empty["counts"]["chunk_inference"] == 0
    assert empty["indexed_chunk_ids"] == set()


def test_largest_neighborhood_keeps_both_seeds_and_only_one_hop():
    edges = [
        ("a", "b", "owl:differentFrom"),
        ("b", "c", "fact"),
        ("c", "d", "fact"),
        ("x", "y", "skos:related"),
        ("x", "z", "fact"),
        ("y", "w", "fact"),
        ("w", "outside", "fact"),
        ("lonely", "other", "fact"),
    ]
    assert largest_one_hop_neighborhood(edges) == {"x", "y", "z", "w"}
    assert largest_one_hop_neighborhood(edges + [("c", "z", "fact")]) == {
        "a",
        "b",
        "c",
        "x",
        "y",
        "z",
        "w",
    }
    assert largest_one_hop_neighborhood([("a", "b", "fact")]) == set()
    assert largest_one_hop_neighborhood([]) == set()


def test_component_ties_are_deterministic_and_unknown_edges_do_not_expand():
    edges = [
        ("x", "y", "skos:related"),
        ("a", "b", "owl:differentFrom"),
        ("a", "outside", "passage"),
    ]
    assert largest_one_hop_neighborhood(edges) == {"a", "b"}
    assert largest_one_hop_neighborhood(list(reversed(edges))) == {"a", "b"}


def test_recent_related_assertions_are_active_canonical_pairs(tmp_path):
    with closing(sqlite3.connect(tmp_path / "assertions.db")) as conn:
        conn.executescript(
            "CREATE TABLE entities(entity_id TEXT PRIMARY KEY, name TEXT, definition TEXT);"
            "CREATE TABLE edges(src_id TEXT, dst_id TEXT, kind TEXT, "
            "PRIMARY KEY(src_id, dst_id, kind));"
        )
        for i in range(12):
            a, b = f"a{i:02}", f"b{i:02}"
            conn.executemany(
                "INSERT INTO entities VALUES (?, ?, ?)",
                [(a, a, "Definition A"), (b, b, "Definition B")],
            )
            conn.executemany("INSERT INTO edges VALUES (?, ?, 'skos:related')", [(a, b), (b, a)])
        rows = recent_relation_assertions(conn, "skos:related")
        assert len(rows) == 10
        assert [row[0] for row in rows] == [f"a{i:02}" for i in range(11, 1, -1)]
        conn.execute("DELETE FROM edges WHERE src_id IN ('a11', 'b11')")
        assert recent_relation_assertions(conn, "skos:related")[0][0] == "a10"
        assert recent_relation_assertions(conn, "owl:differentFrom") == []

    report = tmp_path / "report.md"
    graph = {
        "related_assertion_rows": rows,
        "diff_from_rows": [("d1", "Ambiguous A", "First sense", "d2", "Ambiguous B", "Other")],
    }
    candidates = [
        {"cand": 1, "n": 100, "val_score": 0.4, "f1": 0.8, "r5": 0.9},
        {"cand": 2, "n": 110, "val_score": 0.5, "f1": 0.1, "r5": 0.8},
    ]
    generate_report_markdown(report, {"kg_graph": graph}, candidates, 210, ("TEST", "#38bdf8"))
    text = report.read_text(encoding="utf-8")
    assert text.index("![Fig. 2:") < text.index("### Graph Assertions")
    assert text.index("### Graph Assertions") < text.index("### Most Recent `skos:related`")
    assert text.index("### Most Recent `skos:related`") < text.index("![Fig. 3:")
    assert text.index("**`a11`**") < text.index("**`a10`**")
    assert "**`a00`**" not in text
    assert "Best composite score **`#2` `0.5000`**" in text
    assert "Best F1" not in text


def test_projection_renders_only_largest_seeded_component(tmp_path):
    edges = [("a", "b", "owl:differentFrom"), ("b", "c", "fact"), ("x", "y", "skos:related")]
    sandbox = {"kg_graph": {"edges": edges, "entities": {n: (n, "Definition", 1) for n in "abcxy"}}}
    output = tmp_path / "graph.svg"
    result = generate_graph_projection_svg(output, sandbox, ("TEST", "#38bdf8"), layout_iterations=200)
    assert result["nodes_rendered"] == 3
    assert result["diff_edges"] == 1
    assert result["related_edges"] == 0
    assert result["layout_iterations"] == 200
    assert "Largest 1-hop component" in output.read_text(encoding="utf-8")


def test_watchdog_preserves_directed_fact_edges_and_word_boundary_truncation(tmp_path, monkeypatch):
    import json
    import os

    from yaw.daemon.graph.storage import open_yaw_db

    monkeypatch.setattr(watchdog, "WORKSPACE_ROOT", tmp_path)
    yaw_dir = tmp_path / ".yaw"
    yaw_dir.mkdir(parents=True, exist_ok=True)
    db_path = yaw_dir / "optimize_yaw.db"
    status_path = yaw_dir / "optimize_status.json"
    pools_path = yaw_dir / "optimize_program_pools.json"
    status_path.write_text(
        json.dumps({"status": {"lifecycle": "ready", "pid": os.getpid(), "active_op": "indexing: stale"}}),
        encoding="utf-8",
    )
    pools_path.write_text(
        json.dumps({"train_queries": 7, "val_queries": 13, "val_chunk_ids": ["c1"]}),
        encoding="utf-8",
    )
    with closing(open_yaw_db(db_path)) as conn:
        conn.executemany(
            "INSERT INTO entities(entity_id, name, definition) VALUES (?, ?, ?)",
            [
                ("entity-z", "Zeta", "A very long definition that should truncate cleanly on a word boundary " * 3),
                ("entity-a", "Alpha", "Short definition"),
                ("entity-m", "Mu", "Connected only via directed fact where src_id > dst_id"),
            ],
        )
        conn.executemany(
            "INSERT INTO edges(src_id, dst_id, kind, weight) VALUES (?, ?, ?, 1.0)",
            [
                ("entity-a", "entity-z", "owl:differentFrom"),
                ("entity-z", "entity-a", "owl:differentFrom"),
                ("entity-z", "entity-m", "fact"),
            ],
        )
        conn.commit()

    info = watchdog.get_active_sandbox_info()
    assert info["datestamp"] == "canonical"
    assert info["optimizer_alive"] is False
    assert info["pools"]["train_queries"] == 7
    assert ("entity-z", "entity-m", "fact") in info["kg_graph"]["edges"]
    assert largest_one_hop_neighborhood(info["kg_graph"]["edges"]) == {"entity-a", "entity-z", "entity-m"}

    report = tmp_path / "report.md"
    generate_report_markdown(report, info, [], 0, ("TEST", "#38bdf8"))
    report_text = report.read_text(encoding="utf-8")
    assert any(tok in report_text for tok in ("boundary…", "cleanly…", "truncate…", "should…"))


def test_open_ro_sqlite_backoff_and_dark_figure_candidate_svgs(tmp_path):
    import threading
    import time

    from scripts.watchdog import (
        _open_ro_sqlite,
        generate_candidate_quality_svg,
        generate_candidate_score_svg,
    )
    from yaw.daemon.graph.storage import get_graph_counts, open_readonly_yaw_db, open_yaw_db

    missing_db = tmp_path / "delayed.db"
    assert _open_ro_sqlite(missing_db, backoff=()) is None

    def _create_delayed():
        time.sleep(0.05)
        with closing(open_yaw_db(missing_db)):
            pass

    writer = threading.Thread(target=_create_delayed, daemon=True)
    writer.start()
    ro_conn = open_readonly_yaw_db(missing_db, timeout=1.0, backoff=(0.04, 0.08, 0.12))
    try:
        assert get_graph_counts(ro_conn) == (0, 0, 0, 0, 0)
    finally:
        ro_conn.close()
        writer.join(timeout=5.0)
        assert not writer.is_alive()

    cands = [
        {
            "cand": 1,
            "n": 4,
            "score": 0.65,
            "score_min": 0.55,
            "score_max": 0.75,
            "val_score": 0.68,
            "eff_score": 0.68,
            "best_so_far": 0.68,
            "is_best": True,
            "s_per_it": 1.2,
            "r1": 0.5,
            "r2": 0.6,
            "r5": 0.75,
            "r10": 0.8,
            "r20": 0.85,
            "em": 0.6,
            "f1": 0.7,
            "evaluations": [{"eval": 1, "score": 0.65}],
        }
    ]
    score_svg = tmp_path / "score.svg"
    qual_svg = tmp_path / "qual.svg"
    generate_candidate_score_svg(score_svg, cands, ("RUNNING", "#10b981"), {})
    generate_candidate_quality_svg(qual_svg, cands, ("RUNNING", "#10b981"), {})
    assert "<svg" in score_svg.read_text(encoding="utf-8")
    assert "<svg" in qual_svg.read_text(encoding="utf-8")


def test_kg_dashboard_throughput_entities_per_chunk_and_apng_projection(tmp_path, monkeypatch):
    from PIL import Image

    from scripts.watchdog import generate_kg_dashboard_svg

    monkeypatch.setattr(watchdog, "WORKSPACE_ROOT", tmp_path)
    kg_svg = tmp_path / "kg_dashboard.svg"
    sandbox_fig1 = {
        "counts": {"chunk_inference": 10},
        "kg_metrics": {"total_entities": 60, "total_triples": 45},
        "timeseries_samples": [
            {
                "time_sec": 360.0,
                "time_hours": 0.1,
                "passages": 5,
                "entities": 30,
                "triples": 20,
                "rolling_latency": 28.5,
                "entities_per_chunk": 6.0,
            },
            {
                "time_sec": 720.0,
                "time_hours": 0.2,
                "passages": 10,
                "entities": 60,
                "triples": 45,
                "rolling_latency": 24.0,
                "entities_per_chunk": 6.0,
            },
        ],
    }
    generate_kg_dashboard_svg(kg_svg, sandbox_fig1)
    svg_text = kg_svg.read_text(encoding="utf-8")
    assert "Throughput (s/chunk)" in svg_text
    assert "Extraction (entities/chunk)" in svg_text
    assert "Entities / chunk" in svg_text

    apng_dest = tmp_path / "knowledge_graph_2d_projection.png"
    sandbox_step1 = {
        "datestamp": "canonical",
        "counts": {"chunk_inference": 1},
        "kg_graph": {
            "edges": [("a", "b", "skos:related")],
            "entities": {"a": ("Alpha", "Def A", 1), "b": ("Beta", "Def B", 1)},
        },
    }
    sandbox_step2 = {
        "datestamp": "canonical",
        "counts": {"chunk_inference": 2},
        "kg_graph": {
            "edges": [("a", "b", "skos:related"), ("b", "c", "fact")],
            "entities": {
                "a": ("Alpha", "Def A", 1),
                "b": ("Beta", "Def B", 1),
                "c": ("Gamma", "Def C", 1),
            },
        },
    }
    generate_graph_projection_svg(apng_dest, sandbox_step1, ("STEP 1", "#38bdf8"), layout_iterations=30)
    generate_graph_projection_svg(apng_dest, sandbox_step2, ("STEP 2", "#38bdf8"), layout_iterations=30)
    assert (tmp_path / ".yaw" / "optimize_kg_frames").is_dir()
    with Image.open(apng_dest) as img:
        assert getattr(img, "is_animated", False) is True
        assert getattr(img, "n_frames", 1) == 2


def test_schedule_guard_window_and_seconds_until_end():
    from datetime import datetime as dt
    from datetime import time as dt_time

    # Day window: 07:00 to 21:00
    assert watchdog.is_within_schedule_window(dt_time(12, 0), "07:00", "21:00") is True
    assert watchdog.is_within_schedule_window(dt_time(7, 0), "07:00", "21:00") is True
    assert watchdog.is_within_schedule_window(dt_time(6, 59, 59), "07:00", "21:00") is False
    assert watchdog.is_within_schedule_window(dt_time(21, 0), "07:00", "21:00") is False
    assert watchdog.is_within_schedule_window(dt_time(23, 0), "07:00", "21:00") is False

    # Overnight window: 21:00 to 07:00
    assert watchdog.is_within_schedule_window(dt_time(22, 0), "21:00", "07:00") is True
    assert watchdog.is_within_schedule_window(dt_time(2, 0), "21:00", "07:00") is True
    assert watchdog.is_within_schedule_window(dt_time(7, 0), "21:00", "07:00") is False
    assert watchdog.is_within_schedule_window(dt_time(12, 0), "21:00", "07:00") is False

    # Seconds until end
    t_day = dt(2026, 11, 10, 10, 0, tzinfo=UTC)
    assert watchdog.seconds_until_schedule_end(t_day, "21:00") == 11 * 3600.0

    t_night = dt(2026, 11, 10, 23, 0, tzinfo=UTC)
    assert watchdog.seconds_until_schedule_end(t_night, "07:00") == 8 * 3600.0


def test_suspended_workload_context_manager(monkeypatch):
    calls = []

    class DummyProc:
        def __init__(self, pid: int, depth: int):
            self.pid = pid
            self._depth = depth

        def parents(self):
            return [0] * self._depth

        def suspend(self):
            calls.append(("suspend", self.pid))

        def resume(self):
            calls.append(("resume", self.pid))

    p_parent = DummyProc(100, 1)
    p_child = DummyProc(101, 2)

    monkeypatch.setattr(watchdog, "_find_workload_processes", lambda **kwargs: [p_child, p_parent])

    with watchdog._suspended_workload("unit test", include_optimizer=True) as active:
        assert active is True
        assert calls == [("suspend", 101), ("suspend", 100)]

    # Resume must be in reverse order (parent before child)
    assert calls == [
        ("suspend", 101),
        ("suspend", 100),
        ("resume", 100),
        ("resume", 101),
    ]


def test_check_schedule_guard_active_and_inactive(monkeypatch):
    resumed = []

    class DummyProc:
        def suspend(self):
            pass

        def resume(self):
            resumed.append(True)

    monkeypatch.setattr(watchdog, "_find_workload_processes", lambda **kwargs: [DummyProc()])

    # When outside the window: returns False immediately without sleeping or suspending
    from datetime import datetime as dt

    # 22:00 is outside 07:00..21:00
    outside_dt = dt(2026, 11, 10, 22, 0, tzinfo=UTC)

    class MockDatetimeOutside(dt):
        @classmethod
        def now(cls, tz=None):
            return outside_dt

    monkeypatch.setattr(watchdog, "datetime", MockDatetimeOutside)
    assert watchdog.check_schedule_guard("07:00", "21:00") is False
    assert not resumed

    # When inside window: suspends, loops, and exits when window passes
    inside_times = [
        dt(2026, 11, 10, 12, 0, tzinfo=UTC),  # initial
        dt(2026, 11, 10, 21, 1, tzinfo=UTC),  # check after sleep
    ]

    class MockDatetimeInside(dt):
        @classmethod
        def now(cls, tz=None):
            if inside_times:
                return inside_times.pop(0)
            return dt(2026, 11, 10, 22, 0, tzinfo=UTC)

    monkeypatch.setattr(watchdog, "datetime", MockDatetimeInside)
    monkeypatch.setattr(watchdog.time, "sleep", lambda sec: None)

    assert watchdog.check_schedule_guard("07:00", "21:00", poll_interval_sec=0.01) is True
    assert resumed == [True]
