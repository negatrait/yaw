"""Graph selection and assertion trails over real SQLite and report files."""

import sqlite3
from contextlib import closing

from scripts.watchdog import (
    generate_graph_projection_svg,
    generate_report_markdown,
    largest_one_hop_neighborhood,
    recent_relation_assertions,
)


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
    generate_report_markdown(report, {"kg_graph": graph}, candidates, 210, False, ("TEST", "#38bdf8"))
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
    generate_candidate_score_svg(score_svg, cands, False, ("RUNNING", "#10b981"), {})
    generate_candidate_quality_svg(qual_svg, cands, False, ("RUNNING", "#10b981"), {})
    assert "<svg" in score_svg.read_text(encoding="utf-8")
    assert "<svg" in qual_svg.read_text(encoding="utf-8")


def test_kg_dashboard_throughput_entities_per_chunk_and_apng_projection(tmp_path):
    from PIL import Image

    from scripts.watchdog import generate_kg_dashboard_svg

    kg_svg = tmp_path / "kg_dashboard.svg"
    sandbox_fig1 = {
        "counts": {"openie_chunks": 10},
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
    generate_kg_dashboard_svg(kg_svg, sandbox_fig1, {})
    svg_text = kg_svg.read_text(encoding="utf-8")
    assert "Throughput (s/chunk)" in svg_text
    assert "Extraction (entities/chunk)" in svg_text
    assert "Entities / chunk" in svg_text

    apng_dest = tmp_path / "knowledge_graph_2d_projection.png"
    frame_dir = tmp_path / "frames"
    sandbox_step1 = {
        "frame_dir": str(frame_dir),
        "counts": {"openie_chunks": 1},
        "kg_graph": {
            "edges": [("a", "b", "skos:related")],
            "entities": {"a": ("Alpha", "Def A", 1), "b": ("Beta", "Def B", 1)},
        },
    }
    sandbox_step2 = {
        "frame_dir": str(frame_dir),
        "counts": {"openie_chunks": 2},
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
    with Image.open(apng_dest) as img:
        assert getattr(img, "is_animated", False) is True
        assert getattr(img, "n_frames", 1) == 2


def test_projection_and_report_with_concept_schemes(tmp_path):
    output_svg = tmp_path / "graph_cs.svg"
    report_md = tmp_path / "report_cs.md"
    sandbox = {
        "counts": {"openie_chunks": 5, "entities": 3, "facts": 1},
        "kg_metrics": {
            "concept_schemes": 1,
            "synonym_merges": 0,
            "broader_pairs": 0,
            "related_pairs": 0,
            "different_from_pairs": 0,
        },
        "kg_graph": {
            "edges": [
                ("t1", "scheme-0", "skos:topConceptOf"),
                ("t1", "m2", "fact"),
            ],
            "entities": {
                "t1": ("Top Concept 1", "Definition of top concept", 1),
                "m2": ("Member 2", "Definition of member", 1),
                "scheme-0": ("[Scheme] Machine Learning", "Cluster of ML concepts", 6),
            },
            "concept_schemes": [
                {
                    "scheme_id": "scheme-0",
                    "pref_label": "Machine Learning",
                    "definition": "Cluster of ML concepts",
                    "history_note": "Modularity induced",
                    "confidence": 0.95,
                    "members": ["t1", "m2"],
                    "top_concepts": ["t1"],
                }
            ],
            "diff_from_rows": [],
            "related_rows": [],
            "broader_rows": [],
        },
    }

    result = generate_graph_projection_svg(output_svg, sandbox, ("TEST", "#38bdf8"), layout_iterations=50)
    assert result["nodes_rendered"] == 3
    assert result["scheme_nodes"] == 1
    assert result["top_concept_edges"] == 1
    svg_text = output_svg.read_text(encoding="utf-8")
    assert "skos:topConceptOf" in svg_text
    assert "skos:ConceptScheme" in svg_text
    assert "[Scheme] Machine Learning" in svg_text

    generate_report_markdown(report_md, sandbox, [], 0, False, ("TEST", "#38bdf8"))
    report_text = report_md.read_text(encoding="utf-8")
    assert "### Emergent Concept Schemes (`skos:ConceptScheme`)" in report_text
    assert "◆ **`Machine Learning`** (roots: `Top Concept 1`)" in report_text
    assert "`1` `skos:ConceptScheme`" in report_text
