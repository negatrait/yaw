"""Native dashboard parsing and rendering against real audit and SVG files."""

import json
from xml.etree import ElementTree

import pytest

import scripts.watchdog as watchdog


def audit_file(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(watchdog, "WORKSPACE_ROOT", tmp_path)
    folder = tmp_path / ".yaw"
    folder.mkdir(exist_ok=True)
    path = folder / "compiled_program_optimize_test_graph_rlm_audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_candidate_audit_keeps_run_identity_and_last_good_snapshot_on_torn_reads(tmp_path, monkeypatch):
    path = audit_file(
        tmp_path,
        monkeypatch,
        {
            "candidate_records": [
                {"cand": 2, "n": 4, "score": 0.7, "val_score": 0.5, "is_best": True},
                {"cand": 1, "n": 2, "score": 0.6},
                None,
            ]
        },
    )
    assert watchdog.load_candidate_records({}) == ([], 0, False)
    assert watchdog.load_candidate_records({"datestamp": "missing"}) == ([], 0, False)
    rows, evaluations, legacy = watchdog.load_candidate_records({"datestamp": "test"})
    assert evaluations == 6 and legacy is False
    assert [row["cand"] for row in rows] == [1, 2]
    assert [row["eff_score"] for row in rows] == [0.6, 0.5]
    assert [row["best_so_far"] for row in rows] == [0.6, 0.6]
    assert watchdog.load_candidate_records({"datestamp": "test"}) == (rows, evaluations, legacy)
    for malformed in ("invalid JSON", "[]"):
        path.write_text(malformed, encoding="utf-8")
        assert watchdog.load_candidate_records({"datestamp": "test"}) == (rows, evaluations, legacy)
    path.write_text(json.dumps({"eval_records": [{"score": 0.4}, "bad", {"score": 0.8}]}), encoding="utf-8")
    restored, evaluations, legacy = watchdog.load_candidate_records({"datestamp": "test"})
    assert evaluations == 2 and legacy is True
    assert [row["cand"] for row in restored] == [1, 2]
    assert [row["score"] for row in restored] == [0.4, 0.8]


def test_pool_telemetry_uses_observed_progress_and_labels_missing_values_honestly():
    empty = watchdog.PoolTelemetry.from_sandbox({})
    assert empty.velocity is None and empty.val_eta_min is None
    assert empty.frac(None, None) == "n/a"
    assert empty.frac(1, 0) == "n/a"
    assert empty.duration(None) == "n/a"
    assert empty.duration(125.5) == "2h 5m"
    assert "STOPPED" in empty.badge(0, 0, False)[0]
    assert "RUNNING" in empty.badge(0, 0, True, "Indexing")[0]
    assert "STOPPED" in empty.badge(2, 10, False)[0]
    assert len(empty.badge(2, 10, True, "x" * 100)[0]) <= 96
    telemetry = watchdog.PoolTelemetry.from_sandbox(
        {
            "counts": {"openie_chunks": 11},
            "indexed_chunk_ids": {"a", "b"},
            "pools": {"val_chunk_ids": ["a", "c", "d"], "train_chunk_ids": ["b"], "train_queries": 150},
            "timeseries_samples": [{"time_sec": 0, "passages": 1}, {"time_sec": 600, "passages": 11}],
        }
    )
    assert telemetry.val_done == 1 and telemetry.val_pool == 3
    assert telemetry.train_done == 1 and telemetry.train_pool == 1
    assert telemetry.train_queries == 150 and telemetry.val_queries is None
    assert telemetry.velocity == 1.0 and telemetry.sec_per_chunk == 60.0
    assert telemetry.val_eta_min == 2.0


def test_individual_evaluations_update_within_the_same_candidate(tmp_path, monkeypatch):
    payload = {
        "candidate_records": [
            {"cand": 1, "n": 20, "score": 0.45},
            {"cand": 2, "n": 2, "score": 0.5},
        ],
        # A bounded audit tail need not start at evaluation 1 or contain every candidate.
        "eval_records": [
            {"eval": 21, "cand": 2, "score": 0.2},
            {"eval": 22, "cand": 2, "score": 0.8},
        ],
    }
    audit = audit_file(tmp_path, monkeypatch, payload)
    rows, total, legacy = watchdog.load_candidate_records({"datestamp": "test"})
    assert total == 22 and not legacy
    assert rows[0]["evaluations"] == []
    assert rows[1]["evaluations"] == [
        {"eval": 21, "score": 0.2},
        {"eval": 22, "score": 0.8},
    ]
    path = tmp_path / "scores.svg"
    watchdog.generate_candidate_score_svg(path, rows, legacy, ("TEST", "#38bdf8"), {})
    text = " ".join(ElementTree.parse(path).getroot().itertext())
    assert "eval #22" in text and "candidate #2" in text
    assert "score 80.0%" in text and "mean 50.0%" in text
    assert "Individual evaluation score" in text
    assert "Current candidate mean (all its evals)" in text
    assert "recent 2 recorded" in text

    payload["candidate_records"][1].update(n=3, score=0.4)
    payload["eval_records"].append({"eval": 23, "cand": 2, "score": 0.2})
    audit.write_text(json.dumps(payload), encoding="utf-8")
    rows, total, legacy = watchdog.load_candidate_records({"datestamp": "test"})
    assert len(rows) == 2 and total == 23
    watchdog.generate_candidate_score_svg(path, rows, legacy, ("TEST", "#38bdf8"), {})
    text = " ".join(ElementTree.parse(path).getroot().itertext())
    assert "eval #23" in text and "score 20.0%" in text and "mean 40.0%" in text
    assert "recent 3 recorded" in text
    assert not watchdog.plt.get_fignums()


def test_native_gepa_chart_uses_validation_coverage_and_exact_membership(tmp_path, monkeypatch):
    payload = {
        "gepa_progress": {
            "version": 1,
            "phase": "reflecting / generating instructions",
            "iteration": 3,
            "phase_evaluations": 0,
            "phase_expected": None,
            "validation": [
                {"candidate_idx": 0, "score": 0.0, "evaluated": 10, "total": 10},
                {"candidate_idx": 1, "score": 0.4, "evaluated": 10, "total": 10},
                {"candidate_idx": 2, "score": 0.9, "evaluated": 2, "total": 10},
            ],
            "frontier": [1, 2],
            "frontier_history": [
                {"event": 1, "members": [0], "added": [0], "removed": []},
                {"event": 2, "members": [1, 2], "added": [1, 2], "removed": [0]},
            ],
        }
    }
    audit = audit_file(tmp_path, monkeypatch, payload)
    path = tmp_path / "native.svg"
    sandbox = {"datestamp": "test", "optimizer_alive": True}
    # No dashboard candidate aggregates are needed, including after checkpoint restoration.
    watchdog.generate_pareto_svg(path, [], False, ("TEST", "#38bdf8"), sandbox)
    text = " ".join(ElementTree.parse(path).getroot().itertext())
    for expected in (
        "Seed baseline (0.0%)",
        "Best full-validation score (40.0%)",
        "Partial validation (not comparable)",
        "10/10",
        "2/10",
        "Current native frontier: 1, 2",
        "Displaced",
        "reflecting / generating instructions",
        "iteration 3",
    ):
        assert expected in text
    assert "Latency" not in text and "GM quality" not in text and "Archive size" not in text
    payload["gepa_progress"].update(phase="minibatch evaluation", phase_evaluations=2, phase_expected=10)
    audit.write_text(json.dumps(payload), encoding="utf-8")
    watchdog.generate_pareto_svg(path, [], False, ("TEST", "#38bdf8"), sandbox)
    text = " ".join(ElementTree.parse(path).getroot().itertext())
    assert "Metric evaluations since phase began: 2/10" in text
    assert not watchdog.plt.get_fignums()


def test_gepa_chart_does_not_invent_native_frontier_for_older_audits(tmp_path, monkeypatch):
    audit_file(tmp_path, monkeypatch, {"candidate_records": [{"cand": 1, "n": 10, "score": 0.9}]})
    path = tmp_path / "old.svg"
    watchdog.generate_pareto_svg(path, [], False, ("TEST", "#38bdf8"), {"datestamp": "test"})
    text = " ".join(ElementTree.parse(path).getroot().itertext())
    assert "Native GEPA telemetry unavailable" in text
    assert "cannot reconstruct GEPA's frontier" in text


@pytest.mark.parametrize("legacy", [False, True])
def test_candidate_plots_render_real_audit_metrics_and_close_figures(tmp_path, monkeypatch, legacy):
    candidates = [
        {
            "cand": 1,
            "n": 2,
            "score": 0.3,
            "score_min": 0.1,
            "score_max": 0.5,
            "val_score": 0.4,
            "is_best": True,
            "s_per_it": 2.0,
            "r1": 0.2,
            "r2": 0.3,
            "r5": 0.4,
            "r10": 0.5,
            "r20": 0.6,
            "em": 0.2,
            "f1": 0.3,
        },
        {"cand": 2, "n": 3, "score": 0.6, "s_per_it": 3.0},
        {"cand": 3, "n": 1, "score": 0.2, "s_per_it": 5.0},
    ]
    audit_file(tmp_path, monkeypatch, {"candidate_records": candidates})
    rows, _, _ = watchdog.load_candidate_records({"datestamp": "test"})
    for render in (
        watchdog.generate_candidate_score_svg,
        watchdog.generate_candidate_quality_svg,
        watchdog.generate_pareto_svg,
    ):
        path = tmp_path / f"{render.__name__}.svg"
        render(path, rows, legacy, ("TEST", "#38bdf8"), {})
        root = ElementTree.parse(path).getroot()
        text = " ".join(root.itertext())
        assert "Fig." in text and "TEST" in text
        assert root.findall(".//{http://www.w3.org/2000/svg}path")
    assert not watchdog.plt.get_fignums()


@pytest.mark.parametrize("sandbox", [{}, {"db_name": "run.db"}, {"db_name": "run.db", "optimizer_alive": True}])
def test_missing_candidates_render_honest_status_placeholders(tmp_path, sandbox):
    for render in (
        watchdog.generate_candidate_score_svg,
        watchdog.generate_candidate_quality_svg,
        watchdog.generate_pareto_svg,
    ):
        path = tmp_path / f"{render.__name__}.svg"
        render(path, [], False, ("TEST", "#38bdf8"), sandbox)
        text = " ".join(ElementTree.parse(path).getroot().itertext())
        assert "No optimization sandbox" in text if not sandbox else "candidate" in text.lower()


@pytest.mark.parametrize("populated", [False, True])
def test_indexing_dashboard_renders_measured_throughput_and_empty_state(tmp_path, populated):
    sandbox = {}
    if populated:
        sandbox = {
            "counts": {"openie_chunks": 11},
            "kg_metrics": {"total_entities": 20, "total_triples": 15},
            "indexed_chunk_ids": {"a"},
            "pools": {"val_chunk_ids": ["a", "b"], "corpus_chunk_ids": ["a", "b", "c"]},
            "lemonade_metrics": {"server_up": True, "qwen_tps": 20.0},
            "timeseries_samples": [
                {
                    "time_sec": 0,
                    "time_hours": 0,
                    "passages": 1,
                    "entities": 2,
                    "triples": 1,
                    "rolling_velocity": None,
                    "rolling_latency": None,
                },
                {
                    "time_sec": 600,
                    "time_hours": 1 / 6,
                    "passages": 11,
                    "entities": 20,
                    "triples": 15,
                    "rolling_velocity": 1.0,
                    "rolling_latency": 60.0,
                },
            ],
        }
    path = tmp_path / "indexing.svg"
    watchdog.generate_kg_dashboard_svg(path, sandbox, {})
    text = " ".join(ElementTree.parse(path).getroot().itertext())
    assert "Fig. 1" in text
    assert "Val-pool projection" in text if populated else "Not enough samples yet" in text


def test_config_reader_uses_real_toml_and_explicit_missing_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "WORKSPACE_ROOT", tmp_path)
    assert watchdog.read_config_info()["mode"] == "hybrid"
    config = tmp_path / "yaw.toml"
    config.write_text('[optimization]\nmode="gepa"\n', encoding="utf-8")
    assert watchdog.read_config_info() == {"mode": "gepa"}
    config.write_text("invalid = [", encoding="utf-8")
    assert watchdog.read_config_info()["mode"] == "hybrid"
