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

    writer = threading.Thread(target=_create_delayed)
    writer.start()
    ro_conn = open_readonly_yaw_db(missing_db, timeout=1.0, backoff=(0.04, 0.08, 0.12))
    try:
        assert get_graph_counts(ro_conn) == (0, 0, 0, 0, 0)
    finally:
        ro_conn.close()
        writer.join(timeout=2.0)

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
