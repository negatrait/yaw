"""Semantic judgment notes remain attached to their particular RDF assertion."""

from contextlib import closing

from yaw.daemon.graph.storage import open_yaw_db, serialize_graph_turtle, upsert_entities, upsert_semantic_edges


def test_semantic_reasoning_exports_as_assertion_note_without_redefining_resources(tmp_path):
    reason = 'A café router is an architectural part of the "Campus network".\nIt is not a synonym.'
    with closing(open_yaw_db(tmp_path / "graph.db")) as conn:
        upsert_entities(
            conn,
            [
                ("entity-router", "Café router", "A device routing traffic."),
                ("entity-network", "Campus network", "A network serving the campus."),
            ],
        )
        upsert_semantic_edges(conn, [("entity-router", "entity-network", "skos:broader", 0.123456789, reason)])
        stored = conn.execute("SELECT reasoning FROM edges WHERE kind='skos:broader'").fetchone()[0]
        assert stored == reason
        turtle = serialize_graph_turtle(conn, [])
        assert turtle == serialize_graph_turtle(conn, [])
        assert "rdf:subject :entity-router" in turtle
        assert "rdf:predicate skos:broader" in turtle
        assert "rdf:object :entity-network" in turtle
        assert 'skos:note "A café router' in turtle
        assert '\\"Campus network\\"' in turtle
        assert "\\nIt is not a synonym." in turtle
        assert 'prov:value "0.123456789"^^xsd:double' in turtle
        assert "skos:broader :entity-network" in turtle
        assert 'skos:definition "A device routing traffic."' in turtle
        assert 'skos:definition "A network serving the campus."' in turtle
        assert [row[0] for row in conn.execute("SELECT definition FROM entities ORDER BY entity_id")] == [
            "A network serving the campus.",
            "A device routing traffic.",
        ]


def test_complete_selected_evidence_has_no_implicit_character_ceiling(tmp_path):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.storage import upsert_chunk_pointers, upsert_entity_labels
    from yaw.daemon.graph.storage.hydration import format_subgraph_for_qa, hydrate_budgeted_passages

    source = "Source evidence. " * 5000 + "SOURCE_TAIL"
    reason = "Grounded relational explanation. " * 3000 + "NOTE_TAIL"
    (tmp_path / "evidence.txt").write_text(source, encoding="utf-8")
    pointer = ChunkPointer("chunk-evidence", "evidence.txt", 1, 1, "Evidence")
    with closing(open_yaw_db(tmp_path / "graph.db")) as conn:
        upsert_chunk_pointers(conn, [(pointer, source)])
        upsert_entities(conn, [("entity-a", "Router", "A routing device."), ("entity-b", "Network", "A network.")])
        upsert_entity_labels(
            conn,
            [
                ("router", "entity-a", "skos:prefLabel", "Router", pointer.chunk_id),
                ("network", "entity-b", "skos:prefLabel", "Network", pointer.chunk_id),
            ],
        )
        upsert_semantic_edges(conn, [("entity-a", "entity-b", "skos:broader", 0.9, reason)])
        context = format_subgraph_for_qa(conn, [pointer.chunk_id])
        assert len(context) > 65536 and reason in context and "NOTE_TAIL" in context
        passages = hydrate_budgeted_passages([(pointer, 1.0)], tmp_path)
        assert passages[0].text == source
        assert "SOURCE_TAIL" in passages[0].text
        turtle = serialize_graph_turtle(conn, [])
        assert f'skos:note "{reason}"' in turtle
