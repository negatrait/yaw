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
