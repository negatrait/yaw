"""Parity reports must preserve gate decisions and emit parseable comparison figures."""

from xml.etree import ElementTree

import pytest

from scripts.verify_paper_parity_gate import (
    aggregate_gate_summary,
    compare_engine_reports,
    format_gate_html,
    generate_latency_svg,
    generate_quality_svg,
    parse_args,
)


def gate_payload(*, dataset="musique", exact_match=0.8, zero_time=False):
    baseline = {
        "engine": "hipporag",
        "status": "passed",
        "num_queries": 2,
        "num_docs": 3,
        "retrieval": {"Recall@1": 0.5, "Recall@2": 0.7, "Recall@5": 0.8},
        "qa": {"ExactMatch": 0.8, "F1": 0.8},
        "indexing_seconds": 10.0,
        "evaluation_seconds": 20.0,
    }
    yaw = {
        **baseline,
        "engine": "yaw",
        "indexing_seconds": 0.0 if zero_time else 5.0,
        "evaluation_seconds": 0.0 if zero_time else 10.0,
        "qa": {"ExactMatch": exact_match, "F1": 0.8},
        "standard_dpr": {"Recall@5": 0.4},
    }
    entry = compare_engine_reports(dataset, baseline, yaw, queries_per_dataset=2, index_queries=3)
    return {
        "config": {"datasets": [dataset, "pending"]},
        "datasets": {dataset: entry, "pending": {"status": "pending"}},
        "summary": aggregate_gate_summary({dataset: entry}),
    }


@pytest.mark.parametrize("zero_time", [False, True])
def test_latency_svg_is_well_formed_and_shows_both_engine_measurements(zero_time):
    svg = generate_latency_svg(gate_payload(zero_time=zero_time))
    root = ElementTree.fromstring(svg)
    text = " ".join(root.itertext())
    assert "YAW" in text and "Hippo" in text
    assert "Indexing (s)" in text and "Total Run (s)" in text
    assert "1.0× OVERALL" in text if zero_time else "2.0× OVERALL" in text
    assert len(root.findall(".//{http://www.w3.org/2000/svg}rect[@fill='url(#hippoGrad2)']")) >= 4


def test_quality_chart_preserves_exact_match_gate_failure():
    payload = gate_payload(exact_match=0.1)
    assert payload["datasets"]["musique"]["gate_status"] == "FAIL"
    root = ElementTree.fromstring(generate_quality_svg(payload))
    text = " ".join(root.itertext())
    assert "FAIL (ΔR@5=" in text
    assert "Paper " in text and "DPR " in text


@pytest.mark.parametrize(("exact_match", "status"), [(0.8, "PASS"), (0.1, "FAIL")])
def test_legacy_quality_chart_without_saved_decision_uses_all_available_gate_metrics(exact_match, status):
    payload = gate_payload(exact_match=exact_match)
    del payload["datasets"]["musique"]["gate_passed"]
    text = " ".join(ElementTree.fromstring(generate_quality_svg(payload)).itertext())
    assert f"{status} (ΔR@5=" in text


def test_report_charts_escape_dataset_names_and_resolve_unlisted_completed_entries():
    payload = gate_payload(dataset='source<&"name')
    payload["config"]["datasets"] = ["pending"]
    for render in (generate_quality_svg, generate_latency_svg):
        root = ElementTree.fromstring(render(payload))
        assert 'source<&"name' in " ".join(root.itertext())
    assert ElementTree.fromstring(generate_quality_svg({})).tag.endswith("svg")
    assert ElementTree.fromstring(generate_latency_svg({})).tag.endswith("svg")


def test_html_embeds_native_charts_and_preserves_completed_and_pending_rows():
    payload = gate_payload()
    quality, latency = generate_quality_svg(payload), generate_latency_svg(payload)
    html = format_gate_html(payload, quality, latency)
    assert html.count("<svg ") == 2
    assert "Pending" in html and "musique" in html
    assert "2.0× FASTER" in html
    assert quality in html and latency in html


def test_parity_cli_parses_explicit_budget_and_rejects_unknown_mode():
    args = parse_args(
        ["--datasets", "musique", "--queries-per-dataset", "12", "--index-queries", "15", "--mode", "retrieval"]
    )
    assert args.datasets == ["musique"] and args.queries_per_dataset == 12 and args.index_queries == 15
    with pytest.raises(SystemExit) as exited:
        parse_args(["--mode", "invalid"])
    assert exited.value.code == 2


def test_statistical_parity_gate_computes_confidence_and_enforces_strict_zero_tolerance():
    from yaw.daemon.graph.inference import evaluate_benchmark_gate

    payload = gate_payload(exact_match=0.8)
    ds_entry = payload["datasets"]["musique"]
    assert "confidence" in ds_entry
    assert set(ds_entry["confidence"].keys()) == {"Recall@5", "ExactMatch", "F1"}
    r5_conf = ds_entry["confidence"]["Recall@5"]
    assert r5_conf["n"] == 2
    assert r5_conf["parity_confidence"] == pytest.approx(0.5)
    assert r5_conf["diff_confidence"] == pytest.approx(0.0)
    assert len(r5_conf["ci_95"]) == 2
    assert "confidence" in payload["summary"]
    assert set(payload["summary"]["confidence"].keys()) == {"Recall@5", "ExactMatch", "F1"}

    # Strict parity: -0.01 delta must FAIL (no -0.02 tolerance)
    slight_drop = gate_payload(exact_match=0.79)
    assert slight_drop["datasets"]["musique"]["gate_passed"] is False
    assert slight_drop["summary"]["overall_gate_passed"] is False

    # evaluate_benchmark_gate handles rep["queries"] as a list of query dicts
    gate_res = evaluate_benchmark_gate(
        {
            "musique": {
                "queries": [{"q": "1"}, {"q": "2"}, {"q": "3"}, {"q": "4"}],
                "retrieval": {"Recall@1": 0.5, "Recall@2": 0.7, "Recall@5": 0.9},
                "qa": {"ExactMatch": 0.85, "F1": 0.9},
            }
        },
        {
            "datasets": {
                "musique": {
                    "monkeypatched_hipporag": {
                        "Recall@1": 0.5,
                        "Recall@2": 0.7,
                        "Recall@5": 0.8,
                        "ExactMatch": 0.8,
                        "F1": 0.8,
                    }
                }
            }
        },
    )
    assert gate_res["overall_gate_passed"] is True
    assert gate_res["per_dataset_tolerance"] == 0.0
    assert gate_res["datasets"]["musique"]["queries"] == 4
    assert gate_res["datasets"]["musique"]["confidence"]["Recall@5"]["n"] == 4
    assert gate_res["datasets"]["musique"]["confidence"]["Recall@5"]["parity_confidence"] > 0.5
    assert gate_res["macro_means"]["confidence"]["Recall@5"]["n"] == 4

    # Mixed-case dataset names and partial-None QA metrics across datasets
    from scripts.verify_paper_parity_gate import _render_svg_card_frame

    frame_lines = _render_svg_card_frame("Title", "Sub", [(0, "yawGrad2", "YAW")], 340, "2")
    assert any("bgGrad2" in line for line in frame_lines)

    pop_comp = compare_engine_reports(
        "PopQA",
        {
            "engine": "hipporag",
            "num_queries": 4,
            "num_docs": 8,
            "retrieval": {"Recall@1": 0.5, "Recall@2": 0.7, "Recall@5": 0.8},
            "qa": {"ExactMatch": 0.7, "F1": 0.75},
        },
        {
            "engine": "yaw",
            "num_queries": 4,
            "num_docs": 8,
            "retrieval": {"Recall@1": 0.6, "Recall@2": 0.75, "Recall@5": 0.85},
            "qa": {"ExactMatch": 0.8, "F1": 0.85},
        },
        queries_per_dataset=4,
        index_queries=6,
    )
    assert pop_comp["dataset"] == "popqa"
    assert pop_comp["gate_passed"] is True

    custom_ret_only = compare_engine_reports(
        "MyCustomDS",
        {
            "engine": "hipporag",
            "num_queries": 4,
            "num_docs": 8,
            "retrieval": {"Recall@1": 0.4, "Recall@2": 0.6, "Recall@5": 0.8},
            "qa": {"ExactMatch": None, "F1": None},
        },
        {
            "engine": "yaw",
            "num_queries": 4,
            "num_docs": 8,
            "retrieval": {"Recall@1": 0.5, "Recall@2": 0.7, "Recall@5": 0.9},
            "qa": {"ExactMatch": None, "F1": None},
        },
        queries_per_dataset=4,
        index_queries=6,
    )
    assert custom_ret_only["dataset"] == "MyCustomDS"
    assert custom_ret_only["delta_yaw_vs_hipporag"]["ExactMatch"] is None
    assert custom_ret_only["delta_yaw_vs_paper"]["ExactMatch"] is None

    mixed_summary = aggregate_gate_summary({"popqa": pop_comp, "MyCustomDS": custom_ret_only})
    assert mixed_summary["hipporag_mean"]["ExactMatch"] == pytest.approx(0.7)
    assert mixed_summary["yaw_mean"]["ExactMatch"] == pytest.approx(0.8)
    assert mixed_summary["delta_mean"]["ExactMatch"] == pytest.approx(0.1)
    assert mixed_summary["overall_gate_passed"] is True
