"""Real, unmocked tests for yaw.db SQLite engine, sqlite-vec KNN, and contentless FTS5."""

from pathlib import Path

import numpy as np
import pytest

from yaw.core import ChunkPointer, OpenIERecord
from yaw.daemon.graph.storage import (
    delete_chunks,
    deserialize_f32_vector,
    find_entity_disambiguation_candidates,
    find_entity_synonym_candidates,
    get_chunk_pointer,
    get_chunk_pointers,
    get_entity_vectors,
    get_openie_checkpoints,
    initialize_schema,
    inspect_focus_node,
    lint_graph_shacl,
    load_graph_topology,
    merge_entities,
    open_managed_yaw_db,
    open_yaw_db,
    resolve_entity_labels,
    save_openie_checkpoints,
    search_chunk_vectors_knn,
    search_chunks_fts,
    search_entity_vectors_knn,
    search_fact_vectors_knn,
    search_facts_fts,
    serialize_f32_vector,
    serialize_graph_turtle,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
    upsert_fact_vectors,
    upsert_facts,
    upsert_semantic_edges,
    upsert_synonymy_edges,
)


def test_open_yaw_db_wal_and_schema_initialization(tmp_path: Path) -> None:
    db_file = tmp_path / "yaw.db"
    conn = open_yaw_db(db_file)
    try:
        # Check WAL mode
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        # Check foreign keys
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1

        # Check all 11 unified schema tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') ORDER BY name;")
        table_names = {row["name"] for row in cursor.fetchall()}
        expected_tables = {
            "meta",
            "chunks",
            "chunk_fts",
            "openie_chunks",
            "entities",
            "facts",
            "fact_fts",
            "edges",
            "edge_sources",
            "chunk_vec",
            "entity_vec",
            "fact_vec",
        }
        assert expected_tables.issubset(table_names)
    finally:
        conn.close()


def test_vector_serialization_and_bounded_knn(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_vec.db")
    try:
        # 1. Serialization roundtrip
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        blob1 = serialize_f32_vector(vec1)
        res1 = deserialize_f32_vector(blob1)
        np.testing.assert_array_equal(vec1, res1)

        # Invalid vector checks
        with pytest.raises(ValueError, match="Expected 1D"):
            serialize_f32_vector(np.zeros((2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="NaN or infinite"):
            serialize_f32_vector(np.array([1.0, float("nan")], dtype=np.float32))

        # 2. Insert into chunk_vec, entity_vec, fact_vec
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        vec3 = np.array([0.7071, 0.7071, 0.0], dtype=np.float32)

        upsert_chunk_vectors(conn, [("c1", vec1), ("c2", vec2), ("c3", vec3)])
        upsert_entity_vectors(conn, [("e1", vec1), ("e2", vec2)])
        upsert_fact_vectors(conn, [("f1", vec1), ("f2", vec3)])

        # 3. KNN search for vec1
        chunk_hits = search_chunk_vectors_knn(conn, vec1, top_k=2)
        assert len(chunk_hits) == 2
        assert chunk_hits[0][0] == "c1"
        assert chunk_hits[0][1] == pytest.approx(1.0, abs=1e-4)
        assert chunk_hits[1][0] == "c3"
        assert chunk_hits[1][1] == pytest.approx(0.7071, abs=1e-3)

        # 4. Entity vectors batch retrieval
        ent_vecs = get_entity_vectors(conn, ["e1", "e2"])
        assert "e1" in ent_vecs and "e2" in ent_vecs
        np.testing.assert_allclose(ent_vecs["e1"], vec1)

        # 5. Entity & Fact KNN
        ent_hits = search_entity_vectors_knn(conn, vec2, top_k=1)
        assert ent_hits[0][0] == "e2"
        assert ent_hits[0][1] == pytest.approx(1.0, abs=1e-4)

        fact_hits = search_fact_vectors_knn(conn, vec1, top_k=1)
        assert fact_hits[0][0] == "f1"
        assert fact_hits[0][1] == pytest.approx(1.0, abs=1e-4)
    finally:
        conn.close()


def test_contentless_fts5_indexing_and_l1_normalized_lexical_search(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_fts.db")
    try:
        p1 = ChunkPointer(
            chunk_id="chunk-1",
            rel_posix_path="docs/doc1.txt",
            start_line=1,
            end_line=5,
            title="Alan Turing Biography",
        )
        p2 = ChunkPointer(
            chunk_id="chunk-2",
            rel_posix_path="docs/doc2.txt",
            start_line=1,
            end_line=10,
            title="Radio City Bangalore",
        )
        p3 = ChunkPointer(
            chunk_id="chunk-3",
            rel_posix_path="docs/doc3.txt",
            start_line=1,
            end_line=3,
            title="Bletchley Park Cryptanalysis",
        )

        upsert_chunk_pointers(
            conn,
            [
                (
                    p1,
                    "Alan Turing was an English mathematician and computer scientist at Bletchley Park.",
                ),
                (p2, "Radio City 91.1 is an Indian radio station broadcasting in Bangalore."),
                (p3, "Bletchley Park was the principal centre of Allied code-breaking in WWII."),
            ],
        )

        # Verify chunks table stores metadata
        loaded_p1 = get_chunk_pointer(conn, "chunk-1")
        assert loaded_p1 is not None
        assert loaded_p1.title == "Alan Turing Biography"

        batch_ptrs = get_chunk_pointers(conn, ["chunk-1", "chunk-2"])
        assert len(batch_ptrs) == 2

        # Verify chunk_fts is contentless: SELECT body returns NULL
        row = conn.execute("SELECT body FROM chunk_fts WHERE chunk_id = 'chunk-1';").fetchone()
        assert row[0] is None

        # Lexical search for "Bletchley"
        hits = search_chunks_fts(conn, "Bletchley", top_k=5)
        assert len(hits) == 2
        hit_ids = [h[0] for h in hits]
        assert "chunk-1" in hit_ids and "chunk-3" in hit_ids

        # Verify scores are positive and L1-normalized (sum to 1.0)
        total_score = sum(h[1] for h in hits)
        assert total_score == pytest.approx(1.0, abs=1e-5)
        for _, score in hits:
            assert score > 0.0

        # Test updating chunk with contentless re-indexing
        p1_updated = ChunkPointer(
            chunk_id="chunk-1",
            rel_posix_path="docs/doc1.txt",
            start_line=1,
            end_line=5,
            title="Alan Turing Updated",
        )
        upsert_chunk_pointers(
            conn,
            [(p1_updated, "Alan Turing worked in London and Cambridge.")],
        )
        # "Bletchley" should now only match chunk-3
        hits_after = search_chunks_fts(conn, "Bletchley", top_k=5)
        assert len(hits_after) == 1
        assert hits_after[0][0] == "chunk-3"
    finally:
        conn.close()


def test_openie_checkpoints_cache(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_openie.db")
    try:
        rec1 = OpenIERecord(
            chunk_id="c1",
            entities=("Alan Turing", "Bletchley Park"),
            triples=(("Alan Turing", "worked at", "Bletchley Park"),),
        )
        rec2 = OpenIERecord(
            chunk_id="c2",
            entities=("Radio City", "Bangalore"),
            triples=(("Radio City", "located in", "Bangalore"),),
        )

        save_openie_checkpoints(conn, [rec1, rec2])

        loaded = get_openie_checkpoints(conn, ["c1", "c2", "missing"])
        assert len(loaded) == 2
        assert loaded["c1"] == rec1
        assert loaded["c2"] == rec2
    finally:
        conn.close()


def test_graph_upserts_and_atomic_cascading_deletion(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_cascade.db")
    try:
        # Setup 2 chunks
        p1 = ChunkPointer("c1", "doc1.txt", 1, 5, "Doc 1")
        p2 = ChunkPointer("c2", "doc2.txt", 1, 5, "Doc 2")
        upsert_chunk_pointers(conn, [(p1, "Text 1"), (p2, "Text 2")])
        upsert_chunk_vectors(conn, [("c1", [1.0, 0.0]), ("c2", [0.0, 1.0])])

        # Entities: E1 (exclusive to c1), E2 (shared c1 and c2), E3 (exclusive to c2)
        upsert_entities(conn, [("e1", "Alan Turing"), ("e2", "London"), ("e3", "UK")])
        upsert_entity_vectors(conn, [("e1", [1.0, 0.0]), ("e2", [0.0, 1.0]), ("e3", [1.0, 1.0])])

        # Facts: F1 (exclusive to c1), F2 (exclusive to c2)
        upsert_facts(
            conn,
            [
                ("f1", "e1", "born in", "e2", '["Alan Turing", "born in", "London"]'),
                ("f2", "e2", "capital of", "e3", '["London", "capital of", "UK"]'),
            ],
        )
        upsert_fact_vectors(conn, [("f1", [1.0, 0.0]), ("f2", [0.0, 1.0])])

        # Edges and provenance
        # Passage edges: c1 <-> e1, c1 <-> e2; c2 <-> e2, c2 <-> e3
        upsert_edges_with_sources(
            conn,
            [
                ("c1", "e1", "passage", 1.0, "c1"),
                ("c1", "e2", "passage", 1.0, "c1"),
                ("c2", "e2", "passage", 1.0, "c2"),
                ("c2", "e3", "passage", 1.0, "c2"),
                # Fact edges: e1 <-> e2 supported by c1, e2 <-> e3 supported by c2
                ("e1", "e2", "fact", 1.0, "c1"),
                ("e2", "e3", "fact", 1.0, "c2"),
            ],
        )

        # Synonymy edge between e1 and an alias e1_alias
        upsert_entities(conn, [("e1_alias", "A. Turing")])
        upsert_synonymy_edges(conn, [("e1", "e1_alias", 0.95)])

        # Assert initial state
        assert conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM facts;").fetchone()[0] == 2

        # Delete c1
        delete_chunks(conn, ["c1"])

        # 1. c1 metadata and vectors removed
        assert get_chunk_pointer(conn, "c1") is None
        assert get_chunk_pointer(conn, "c2") is not None
        assert conn.execute("SELECT COUNT(*) FROM chunk_vec WHERE chunk_id = 'c1';").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunk_vec WHERE chunk_id = 'c2';").fetchone()[0] == 1

        # 2. Fact F1 (exclusive to c1) is deleted, F2 (supported by c2) remains
        remaining_facts = conn.execute("SELECT fact_id FROM facts;").fetchall()
        assert [r[0] for r in remaining_facts] == ["f2"]
        assert conn.execute("SELECT COUNT(*) FROM fact_vec WHERE fact_id = 'f1';").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM fact_vec WHERE fact_id = 'f2';").fetchone()[0] == 1

        # 3. Entity E1 (exclusive to c1) is deleted, E2 and E3 remain
        remaining_entities = {r[0] for r in conn.execute("SELECT entity_id FROM entities;").fetchall()}
        assert "e1" not in remaining_entities
        assert "e2" in remaining_entities
        assert "e3" in remaining_entities
        assert conn.execute("SELECT COUNT(*) FROM entity_vec WHERE entity_id = 'e1';").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM entity_vec WHERE entity_id = 'e2';").fetchone()[0] == 1

        # 4. Synonymy edge incident to deleted e1 is pruned
        syn_edges = conn.execute("SELECT * FROM edges WHERE kind = 'synonymy';").fetchall()
        assert len(syn_edges) == 0
    finally:
        conn.close()


def test_find_entity_synonym_candidates_in_db(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_synonymy.db")
    try:
        upsert_entities(
            conn,
            [
                ("e1", "Alan Turing"),
                ("e2", "A. M. Turing"),
                ("e3", "Bletchley Park"),
            ],
        )
        upsert_entity_vectors(
            conn,
            [
                ("e1", [1.0, 0.0, 0.0]),
                ("e2", [0.95, 0.3122, 0.0]),  # cosine ~ 0.95 >= 0.8
                ("e3", [0.0, 1.0, 0.0]),  # cosine = 0.0 < 0.8
            ],
        )

        # Full scan candidate discovery
        candidates = find_entity_synonym_candidates(conn, threshold=0.8, top_k=5)
        assert len(candidates) == 1
        ent_a, ent_b, sim = candidates[0]
        assert {ent_a, ent_b} == {"e1", "e2"}
        assert sim == pytest.approx(0.95, abs=1e-3)

        # Incremental scan for a single entity
        inc_candidates = find_entity_synonym_candidates(conn, entity_ids=["e2"], threshold=0.8, top_k=5)
        assert len(inc_candidates) == 1
        assert {inc_candidates[0][0], inc_candidates[0][1]} == {"e1", "e2"}

        # Incremental scan for orthogonal entity returns empty
        assert find_entity_synonym_candidates(conn, entity_ids=["e3"], threshold=0.8) == []
    finally:
        conn.close()


def test_fts5_real_bm25_proportional_scores_and_long_query_cap(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_bm25.db")
    try:
        # 5 documents so term frequency n(q) <= 2 < N/2 (positive Robertson-Sparck Jones IDF in FTS5)
        p_dense = ChunkPointer("c_dense", "d1.txt", 1, 2, "Enigma Cryptanalysis")
        p_sparse = ChunkPointer("c_sparse", "d2.txt", 1, 2, "Notes")
        p_bg1 = ChunkPointer("c_bg1", "d3.txt", 1, 2, "Weather")
        p_bg2 = ChunkPointer("c_bg2", "d4.txt", 1, 2, "Cooking")
        p_bg3 = ChunkPointer("c_bg3", "d5.txt", 1, 2, "Gardening")

        upsert_chunk_pointers(
            conn,
            [
                (
                    p_dense,
                    "Enigma rotor cipher machine. Enigma rotors and enigma plugboard broke enigma codes.",
                ),
                (
                    p_sparse,
                    "The history of the modern world includes many machines and one enigma.",
                ),
                (p_bg1, "Sunny skies and warm temperatures across the coastal region today."),
                (p_bg2, "Fresh basil and olive oil make a classic pasta sauce."),
                (p_bg3, "Planting perennial flowers in early spring improves root growth."),
            ],
        )

        # c_dense matches both "enigma" (4x + title) and "rotor"; c_sparse matches only "enigma" (1x)
        hits = search_chunks_fts(conn, "enigma rotor", top_k=5)
        assert len(hits) == 2
        scores_by_id = dict(hits)
        assert scores_by_id["c_dense"] > scores_by_id["c_sparse"]
        # Genuine BM25 should not equal the naive 1/1, 1/2 (2/3 = 0.6667, 1/3 = 0.3333) ordinal harmonic series
        assert scores_by_id["c_dense"] > 0.70

        # Adversarial 200-word query with punctuation should be capped safely without FTS5 syntax error
        long_query = " ".join(["enigma", "cipher"] + [f"token_{i}" for i in range(180)] + ["?!@#$%^&*()"])
        long_hits = search_chunks_fts(conn, long_query, top_k=5)
        assert len(long_hits) >= 1
        assert sum(s for _, s in long_hits) == pytest.approx(1.0, abs=1e-5)
    finally:
        conn.close()


def test_high_load_batch_operations_exceeding_sqlite_variable_limits(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_high_load.db")
    try:
        n_items = 1200  # Exceeds standard 999 SQLite variable limit
        chunk_ids = [f"chunk-{i}" for i in range(n_items)]
        entity_ids = [f"ent-{i}" for i in range(n_items)]

        chunks = [
            (ChunkPointer(cid, "bulk.txt", i + 1, i + 1, f"Title {i}"), f"Body text {i}")
            for i, cid in enumerate(chunk_ids)
        ]
        upsert_chunk_pointers(conn, chunks)
        upsert_chunk_vectors(conn, [(cid, [1.0, float(i + 1)]) for i, cid in enumerate(chunk_ids)])

        upsert_entities(conn, [(eid, f"Entity {i}") for i, eid in enumerate(entity_ids)])
        upsert_entity_vectors(conn, [(eid, [1.0, float(i + 1)]) for i, eid in enumerate(entity_ids)])

        records = [
            OpenIERecord(cid, (f"Entity {i}",), ((f"Entity {i}", "rel", f"Entity {i}"),))
            for i, cid in enumerate(chunk_ids)
        ]
        save_openie_checkpoints(conn, records)

        # Batch lookups exceeding 1,000 IDs
        loaded_ptrs = get_chunk_pointers(conn, chunk_ids)
        assert len(loaded_ptrs) == n_items

        loaded_vecs = get_entity_vectors(conn, entity_ids)
        assert len(loaded_vecs) == n_items

        loaded_ckpts = get_openie_checkpoints(conn, chunk_ids)
        assert len(loaded_ckpts) == n_items

        # Shared fact edge supported by chunk-0 and chunk-1199
        upsert_facts(
            conn,
            [
                (
                    "f_shared",
                    "ent-0",
                    "linked_to",
                    "ent-1199",
                    '["Entity 0", "linked_to", "Entity 1199"]',
                )
            ],
        )
        upsert_edges_with_sources(
            conn,
            [
                ("ent-0", "ent-1199", "fact", 2.0, "chunk-0"),
                ("ent-0", "ent-1199", "fact", 2.0, "chunk-1199"),
            ],
        )

        # Delete first 1,199 chunks in one batch call, keeping only chunk-1199
        delete_chunks(conn, chunk_ids[:-1])

        assert conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 1
        # Shared fact edge weight must decrement from 2.0 to 1.0
        edge_row = conn.execute(
            "SELECT weight FROM edges WHERE src_id = 'ent-0' AND dst_id = 'ent-1199' AND kind = 'fact';"
        ).fetchone()
        assert edge_row is not None
        assert float(edge_row[0]) == pytest.approx(1.0)
    finally:
        conn.close()


def test_fact_fts_indexing_search_and_backfill(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_fact_fts.db")
    try:
        # 1. Upsert facts with triples
        facts = [
            (
                "fact-1",
                "ent-curie",
                "discovered",
                "ent-radium",
                '["Marie Curie", "discovered", "radium"]',
            ),
            (
                "fact-2",
                "ent-einstein",
                "developed",
                "ent-relativity",
                '["Albert Einstein", "developed", "general relativity"]',
            ),
            (
                "fact-3",
                "ent-turing",
                "broke",
                "ent-enigma",
                '["Alan Turing", "broke", "Enigma code at Bletchley"]',
            ),
        ]
        upsert_facts(conn, facts)

        # 2. Contentless verification: SELECT text returns NULL
        row = conn.execute("SELECT text FROM fact_fts WHERE fact_id = 'fact-1';").fetchone()
        assert row[0] is None

        # 3. Lexical search for "radium"
        hits = search_facts_fts(conn, "radium", top_k=5)
        assert len(hits) == 1
        assert hits[0][0] == "fact-1"
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

        # 4. Lexical search for "Bletchley"
        hits_turing = search_facts_fts(conn, "Bletchley", top_k=5)
        assert len(hits_turing) == 1
        assert hits_turing[0][0] == "fact-3"

        # 5. Multi-term query "Albert Einstein"
        hits_einstein = search_facts_fts(conn, "Albert Einstein", top_k=5)
        assert len(hits_einstein) >= 1
        assert hits_einstein[0][0] == "fact-2"

        # 6. Non-matching query returns empty list
        assert search_facts_fts(conn, "nonexistentquerytoken123", top_k=5) == []

        # 7. Cascade deletion via delete_chunks
        p1 = ChunkPointer("c1", "p.txt", 1, 1, "T1")
        p2 = ChunkPointer("c2", "p.txt", 2, 2, "T2")
        upsert_chunk_pointers(conn, [(p1, "text1"), (p2, "text2")])
        upsert_edges_with_sources(
            conn,
            [
                ("ent-curie", "ent-radium", "fact", 1.0, "c1"),
                ("ent-einstein", "ent-relativity", "fact", 1.0, "c2"),
            ],
        )
        delete_chunks(conn, ["c1"])
        # fact-1 should now be deleted from facts and fact_fts
        assert conn.execute("SELECT COUNT(*) FROM facts WHERE fact_id = 'fact-1';").fetchone()[0] == 0
        assert search_facts_fts(conn, "radium", top_k=5) == []
        # fact-2 should still be intact
        assert len(search_facts_fts(conn, "relativity", top_k=5)) == 1

        # 8. Test backfill: clear fact_fts and re-run initialize_schema
        conn.execute("INSERT INTO fact_fts(fact_fts) VALUES('delete-all');")
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM fact_fts;").fetchone()[0] == 0
        initialize_schema(conn)
        # Should be backfilled with remaining facts (fact-2 and fact-3)
        assert conn.execute("SELECT COUNT(*) FROM fact_fts;").fetchone()[0] == 2
        assert len(search_facts_fts(conn, "relativity", top_k=5)) == 1
    finally:
        conn.close()


def test_open_managed_yaw_db_lock_and_read_only_wal_snapshot(tmp_path: Path) -> None:
    db_file = tmp_path / "managed_yaw.db"
    managed = open_managed_yaw_db(db_file)
    try:
        with managed.transaction() as conn:
            upsert_entities(conn, [("ent-1", "Ada Lovelace", "First computer programmer")])

        with managed.readonly() as ro_conn:
            row = ro_conn.execute("SELECT name, definition FROM entities WHERE entity_id = 'ent-1';").fetchone()
            assert row is not None
            assert row["name"] == "Ada Lovelace"
            assert row["definition"] == "First computer programmer"
    finally:
        managed.close()


def test_skos_labels_s13_disjointness_and_homograph_resolution(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_skos.db")
    try:
        upsert_entities(
            conn,
            [
                ("ent-apple-inc", "Apple Inc.", "Technology company founded in Cupertino"),
                ("ent-apple-fruit", "Malus domestica", "Edible pome fruit grown on orchard trees"),
            ],
        )
        # Register "Apple" as skos:altLabel for both entities (homographs) and attempt S13 duplicate prefLabel as
        # altLabel
        upsert_entity_labels(
            conn,
            [
                ("apple inc.", "ent-apple-inc", "skos:altLabel", "Apple Inc.", "c1"),
                ("apple", "ent-apple-inc", "skos:altLabel", "Apple", "c1"),
                ("apple", "ent-apple-fruit", "skos:altLabel", "Apple", "c2"),
            ],
        )

        # Verify S13 disjointness: "Apple Inc." should only appear as skos:prefLabel, never as skos:altLabel
        s13_rows = conn.execute(
            "SELECT DISTINCT label_kind FROM entity_labels WHERE entity_id = 'ent-apple-inc' AND "
            "label_norm = 'apple inc.';"
        ).fetchall()
        assert [r["label_kind"] for r in s13_rows] == ["skos:prefLabel"]

        # Give each entity orthogonal vectors
        v_tech = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_fruit = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        upsert_entity_vectors(
            conn,
            [("ent-apple-inc", v_tech), ("ent-apple-fruit", v_fruit)],
        )

        # Resolve "Apple" with a tech query vector -> selects ent-apple-inc
        resolved_tech = resolve_entity_labels(
            conn, ["Apple"], query_vec=np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
        )
        assert resolved_tech["apple"] == ["ent-apple-inc"]

        # Resolve "Apple" with a fruit query vector -> selects ent-apple-fruit
        resolved_fruit = resolve_entity_labels(
            conn, ["Apple"], query_vec=np.array([0.05, 0.95, 0.0, 0.0], dtype=np.float32)
        )
        assert resolved_fruit["apple"] == ["ent-apple-fruit"]

        # 2-channel candidate generation finds lexical cousins even when cosine = 0.0
        upsert_entities(conn, [("ent-apple-corp", "Apple Corp.", "Record label founded by The Beatles")])
        upsert_entity_vectors(conn, [("ent-apple-corp", v_fruit)])
        candidates = find_entity_disambiguation_candidates(
            conn, ["ent-apple-inc"], dense_threshold=0.95, dense_top_k=5, lexical_threshold=0.6
        )
        pair_ids = {(a, b) for a, b, _, _ in candidates} | {(b, a) for a, b, _, _ in candidates}
        assert ("ent-apple-inc", "ent-apple-corp") in pair_ids
    finally:
        conn.close()


def test_merge_entities_and_semantic_edge_dominance(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_merge.db")
    try:
        p1 = ChunkPointer("c1", "doc.md", 1, 5, "Sec")
        upsert_chunk_pointers(conn, [(p1, "NYC is New York City.")])
        upsert_entities(
            conn,
            [
                ("ent-nyc-canon", "New York City", "Most populous city in the United States"),
                ("ent-nyc-alias", "NYC", "Abbreviation for New York City"),
                ("ent-usa", "United States", "Federal republic in North America"),
            ],
        )
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.98, 0.02, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        upsert_entity_vectors(conn, [("ent-nyc-canon", v1), ("ent-nyc-alias", v2), ("ent-usa", v3)])

        upsert_facts(
            conn,
            [
                (
                    "fact-alias-usa",
                    "ent-nyc-alias",
                    "located in",
                    "ent-usa",
                    '["NYC", "located in", "United States"]',
                    "NYC is located in the United States",
                )
            ],
        )
        upsert_edges_with_sources(
            conn,
            [
                ("ent-nyc-alias", "ent-usa", "fact", 1.0, "c1"),
                ("c1", "ent-nyc-alias", "passage", 1.0, "c1"),
            ],
        )

        # Merge src_id='ent-nyc-alias' into dst_id='ent-nyc-canon'
        merge_entities(conn, "ent-nyc-alias", "ent-nyc-canon")

        # Alias entity row is gone, its prefLabel "NYC" is now a skos:altLabel on ent-nyc-canon
        assert conn.execute("SELECT COUNT(*) FROM entities WHERE entity_id = 'ent-nyc-alias';").fetchone()[0] == 0
        alt_rows = conn.execute(
            "SELECT raw_label FROM entity_labels WHERE entity_id = 'ent-nyc-canon' AND label_kind = 'skos:altLabel';"
        ).fetchall()
        assert "NYC" in {r["raw_label"] for r in alt_rows}

        # Fact and edges are redirected to ent-nyc-canon
        fact_row = conn.execute("SELECT subj_id, obj_id FROM facts WHERE fact_id = 'fact-alias-usa';").fetchone()
        assert fact_row["subj_id"] == "ent-nyc-canon"
        assert fact_row["obj_id"] == "ent-usa"

        # Now test semantic edges and owl:differentFrom dominance
        upsert_semantic_edges(conn, [("ent-nyc-canon", "ent-usa", "skos:related", 0.85)])
        _, entity_ids, topo_edges = load_graph_topology(conn)
        assert ("ent-nyc-canon" in entity_ids) and ("ent-usa" in entity_ids)
        assert any(s == "ent-nyc-canon" and d == "ent-usa" for s, d, _ in topo_edges)

        # Asserting owl:differentFrom dominates and deletes skos:related
        upsert_semantic_edges(conn, [("ent-nyc-canon", "ent-usa", "owl:differentFrom", 0.90)])
        edge_kinds = {
            r["kind"]
            for r in conn.execute(
                "SELECT kind FROM edges WHERE src_id = 'ent-nyc-canon' AND dst_id = 'ent-usa';"
            ).fetchall()
        }
        assert "skos:related" not in edge_kinds
        assert "owl:differentFrom" in edge_kinds

        # Verify load_graph_topology excludes owl:differentFrom (if we remove the fact edge, no edge remains)
        conn.execute("DELETE FROM edges WHERE kind = 'fact';")
        conn.commit()
        _, _, topo_edges_after = load_graph_topology(conn)
        assert not any({s, d} == {"ent-nyc-canon", "ent-usa"} for s, d, _ in topo_edges_after)
    finally:
        conn.close()


def test_closed_world_shacl_lint_auto_heal_and_turtle_export(tmp_path: Path) -> None:
    conn = open_yaw_db(tmp_path / "test_shacl.db")
    try:
        # Insert an entity with empty definition, an entity with valid definition, and an asymmetric semantic edge
        conn.execute("INSERT INTO entities(entity_id, name, definition) VALUES ('ent-a', 'Entity A', '');")
        conn.execute(
            "INSERT INTO entities(entity_id, name, definition) VALUES ('ent-b', 'Entity B', 'Valid definition B');"
        )
        conn.execute(
            "INSERT INTO facts(fact_id, subj_id, pred, obj_id, raw_triple_json, definition) "
            "VALUES ('fact-ab', 'ent-a', 'relates to', 'ent-b', '[\"Entity A\", \"relates to\", \"Entity B\"]', '');"
        )
        conn.execute("INSERT INTO edges(src_id, dst_id, kind, weight) VALUES ('ent-a', 'ent-b', 'fact', 1.0);")
        conn.execute("INSERT INTO edges(src_id, dst_id, kind, weight) VALUES ('ent-a', 'ent-b', 'skos:related', 0.9);")
        conn.commit()

        violations_before = lint_graph_shacl(conn, auto_heal=False)
        shapes_before = {v.shape for v in violations_before}
        assert "ConceptShape/prefLabel" in shapes_before
        assert "ConceptShape/definition" in shapes_before
        assert "StatementShape/closedWorld" in shapes_before
        assert "EdgeShape/closedWorld" in shapes_before

        # Auto-heal repairs deterministic violations (prefLabels, empty definitions, asymmetric semantic edge)
        violations_after = lint_graph_shacl(conn, auto_heal=True)
        assert violations_after == []

        # Inspect focus node neighbourhood
        info = inspect_focus_node(conn, "ent-a")
        assert info["entity"] is not None
        assert info["entity"]["definition"] == "Entity A participates in relation: Entity A relates to Entity B."

        # Export W3C Turtle with SHACL report
        ttl = serialize_graph_turtle(conn, violations=violations_after)
        assert "@prefix skos: <http://www.w3.org/2004/02/skos/core#> ." in ttl
        assert "@prefix sh: <http://www.w3.org/ns/shacl#> ." in ttl
        assert "sh:conforms true" in ttl
        assert 'skos:prefLabel "Entity A"' in ttl
    finally:
        conn.close()


def test_delete_chunks_homograph_promotion_preserves_owl_different_from(tmp_path: Path) -> None:
    """Verify Hazard 1: promoting an altLabel to prefLabel during delete_chunks links surviving homographs via \
owl:differentFrom."""
    conn = open_yaw_db(tmp_path / "test_hazard1.db")
    try:
        upsert_chunk_pointers(
            conn,
            [
                (ChunkPointer("c1", "doc.md", 1, 5, "Doc 1"), "Apple Inc in Cupertino."),
                (ChunkPointer("c2", "doc.md", 6, 10, "Doc 2"), "Apple designs consumer hardware."),
                (ChunkPointer("c3", "doc.md", 11, 15, "Doc 3"), "Apple is an orchard fruit."),
            ],
        )
        upsert_entities(
            conn,
            [
                ("ent-tech", "Apple Inc", "Consumer electronics company in Cupertino"),
                ("ent-fruit", "Apple", "Edible pome fruit grown on orchard trees"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("apple inc", "ent-tech", "skos:prefLabel", "Apple Inc", "c1"),
                ("apple", "ent-tech", "skos:altLabel", "Apple", "c2"),
                ("apple", "ent-fruit", "skos:prefLabel", "Apple", "c3"),
            ],
        )
        upsert_edges_with_sources(
            conn,
            [
                ("c1", "ent-tech", "passage", 1.0, "c1"),
                ("ent-tech", "c1", "passage", 1.0, "c1"),
                ("c2", "ent-tech", "passage", 1.0, "c2"),
                ("ent-tech", "c2", "passage", 1.0, "c2"),
                ("c3", "ent-fruit", "passage", 1.0, "c3"),
                ("ent-fruit", "c3", "passage", 1.0, "c3"),
            ],
        )
        assert lint_graph_shacl(conn, auto_heal=False) == []

        # Deleting c1 removes "apple inc" prefLabel from ent-tech and promotes "apple" (from c2) to skos:prefLabel.
        # Step 7 in delete_chunks must link ent-tech and ent-fruit with symmetric owl:differentFrom (weight = 0.0).
        delete_chunks(conn, ["c1"])

        tech_name = conn.execute("SELECT name FROM entities WHERE entity_id = 'ent-tech'").fetchone()[0]
        assert tech_name.lower() == "apple"

        diff_rows = conn.execute(
            """
            SELECT src_id, dst_id, weight FROM edges
            WHERE kind = 'owl:differentFrom'
            ORDER BY src_id, dst_id
            """
        ).fetchall()
        assert len(diff_rows) == 2
        assert all(float(r[2]) == 0.0 for r in diff_rows)
        assert lint_graph_shacl(conn, auto_heal=False) == []
    finally:
        conn.close()


def test_multi_chunk_entity_labels_enforces_skos_s13_and_s14_disjointness(tmp_path: Path) -> None:
    "Verify Hazard 4: multi-chunk provenance in entity_labels never violates SKOS S13 or S14 across upsert and merge."
    conn = open_yaw_db(tmp_path / "test_hazard4.db")
    try:
        upsert_entities(
            conn,
            [
                ("ent-usa", "United States", "Federal republic in North America"),
                ("ent-usa-acro", "USA", "Acronym for the United States of America"),
            ],
        )
        # Attempt to insert conflicting label kinds across chunks c1 and c2
        upsert_entity_labels(
            conn,
            [
                ("united states", "ent-usa", "skos:prefLabel", "United States", "c1"),
                ("united states", "ent-usa", "skos:altLabel", "United States", "c2"),
                ("us", "ent-usa", "skos:hiddenLabel", "US", "c1"),
                ("us", "ent-usa", "skos:altLabel", "US", "c2"),
                ("usa", "ent-usa-acro", "skos:prefLabel", "USA", "c3"),
                ("united states", "ent-usa-acro", "skos:altLabel", "United States", "c4"),
            ],
        )
        # Merge ent-usa-acro into ent-usa with label_kind='skos:hiddenLabel'
        merge_entities(conn, "ent-usa-acro", "ent-usa", label_kind="skos:hiddenLabel")

        # Verify S13: every (entity_id, label_norm) has exactly 1 distinct label_kind across all chunks
        s13_conflicts = conn.execute(
            """
            SELECT label_norm, COUNT(DISTINCT label_kind)
            FROM entity_labels
            WHERE entity_id = 'ent-usa'
            GROUP BY label_norm
            HAVING COUNT(DISTINCT label_kind) > 1
            """
        ).fetchall()
        assert s13_conflicts == []

        # Verify S14: exactly 1 distinct label_norm ("united states") has skos:prefLabel
        pref_norms = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT label_norm FROM entity_labels WHERE entity_id = 'ent-usa' AND label_kind "
                "= 'skos:prefLabel'"
            ).fetchall()
        ]
        assert pref_norms == ["united states"]
        assert lint_graph_shacl(conn, auto_heal=False) == []
    finally:
        conn.close()


def test_step11_literal_facts_check_constraints_fact_sources_and_merge_collision(
    tmp_path: Path,
) -> None:
    """Verify H2: facts mutual-exclusion CHECK constraint, fact_sources provenance, delete_chunks orphan pruning, and merge_entities collision deduplication."""
    import sqlite3

    from yaw.daemon.graph.storage import get_facts_with_meta_by_ids, upsert_fact_sources

    conn = open_yaw_db(tmp_path / "test_step11_h2.db")
    try:
        upsert_chunk_pointers(
            conn,
            [
                (ChunkPointer("c1", "doc.md", 1, 5, "Doc 1"), "Alan Turing was born on 1912-06-23."),
                (ChunkPointer("c2", "doc.md", 6, 10, "Doc 2"), "Turing was born on 1912-06-23."),
            ],
        )
        upsert_entities(
            conn,
            [
                ("ent-turing", "Alan Turing", "English mathematician and computer scientist"),
                ("ent-turing-alias", "Turing", "Surname alias for Alan Turing"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("alan turing", "ent-turing", "skos:prefLabel", "Alan Turing", "c1"),
                ("turing", "ent-turing-alias", "skos:prefLabel", "Turing", "c2"),
            ],
        )

        # 1. Verify SQL CHECK constraints reject hybrid/invalid entity vs literal rows
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO facts (fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, obj_literal_lex) "
                "VALUES ('bad-1', 'ent-turing', 'born on', NULL, '[]', 'def', 'entity', NULL)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO facts (fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, obj_literal_lex) "
                "VALUES ('bad-2', 'ent-turing', 'born on', 'ent-turing', '[]', 'def', 'literal', '1912-06-23')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO facts (fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, obj_literal_lex) "
                "VALUES ('bad-3', 'ent-turing', 'born on', NULL, '[]', 'def', 'literal', '   ')"
            )

        # 2. Upsert valid literal-object facts + fact_sources provenance
        upsert_facts(
            conn,
            [
                (
                    "fact-lit-1",
                    "ent-turing",
                    "born on",
                    None,
                    '["Alan Turing", "born on", "1912-06-23"]',
                    "Alan Turing was born on 23 June 1912.",
                    "literal",
                    "1912-06-23",
                    "xsd:date",
                    "",
                ),
                (
                    "fact-lit-2",
                    "ent-turing-alias",
                    "born on",
                    None,
                    '["Turing", "born on", "1912-06-23"]',
                    "Turing was born on 23 June 1912.",
                    "literal",
                    "1912-06-23",
                    "xsd:date",
                    "",
                ),
            ],
        )
        upsert_fact_sources(conn, [("fact-lit-1", "c1"), ("fact-lit-2", "c2")])
        upsert_edges_with_sources(
            conn,
            [
                ("c1", "ent-turing", "passage", 1.0, "c1"),
                ("c2", "ent-turing-alias", "passage", 1.0, "c2"),
            ],
        )

        meta = get_facts_with_meta_by_ids(conn, ["fact-lit-1"])
        assert meta["fact-lit-1"]["obj_kind"] == "literal"
        assert meta["fact-lit-1"]["obj_id"] is None
        assert meta["fact-lit-1"]["obj_literal_lex"] == "1912-06-23"
        assert meta["fact-lit-1"]["obj_datatype"] == "xsd:date"

        # SHACL closed-world validation conforms for literal-object facts (obj_id IS NULL)
        assert lint_graph_shacl(conn, auto_heal=False) == []
        ttl = serialize_graph_turtle(conn)
        assert 'rdf:object "1912-06-23"^^<xsd:date>' in ttl

        # 3. Merge ent-turing-alias into ent-turing -> colliding literal fact is deduplicated and fact_sources merged
        merge_entities(conn, "ent-turing-alias", "ent-turing")
        surviving_facts = conn.execute("SELECT fact_id, subj_id, obj_id, obj_kind FROM facts").fetchall()
        assert len(surviving_facts) == 1
        surviving_fid = str(surviving_facts[0][0])
        src_chunks = {
            str(r[0])
            for r in conn.execute("SELECT chunk_id FROM fact_sources WHERE fact_id = ?", (surviving_fid,)).fetchall()
        }
        assert src_chunks == {"c1", "c2"}

        # 4. Deleting c1 keeps the fact alive (still supported by c2); deleting c2 prunes the orphaned literal fact
        delete_chunks(conn, ["c1"])
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
        delete_chunks(conn, ["c2"])
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM fact_sources").fetchone()[0] == 0
    finally:
        conn.close()


def test_step11_skos_broader_cycle_safe_cte_and_s27_disjointness(tmp_path: Path) -> None:
    """Verify H3: directed skos:broader edges, cycle-safe recursive CTE cycle detection, and SKOS S27 disjointness healing."""
    conn = open_yaw_db(tmp_path / "test_step11_h3.db")
    try:
        upsert_entities(
            conn,
            [
                ("ent-dog", "Dog", "Domesticated canine"),
                ("ent-mammal", "Mammal", "Warm-blooded vertebrate"),
                ("ent-animal", "Animal", "Multicellular eukaryotic organism"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("dog", "ent-dog", "skos:prefLabel", "Dog", "c1"),
                ("mammal", "ent-mammal", "skos:prefLabel", "Mammal", "c1"),
                ("animal", "ent-animal", "skos:prefLabel", "Animal", "c1"),
            ],
        )

        # 1. Directed skos:broader hierarchy: dog -> mammal -> animal
        upsert_semantic_edges(
            conn,
            [
                ("ent-dog", "ent-mammal", "skos:broader", 0.9),
                ("ent-mammal", "ent-animal", "skos:broader", 0.9),
            ],
        )
        broader_rows = conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader' ORDER BY src_id"
        ).fetchall()
        assert [(str(r[0]), str(r[1])) for r in broader_rows] == [
            ("ent-dog", "ent-mammal"),
            ("ent-mammal", "ent-animal"),
        ]
        assert lint_graph_shacl(conn, auto_heal=False) == []

        # 2. Inject a transitive S27 conflict (skos:related between dog and animal) and a 3-hop cycle (animal -> dog)
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-dog', 'ent-animal', 'skos:related', 0.7)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-animal', 'ent-dog', 'skos:related', 0.7)"
        )
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-animal', 'ent-dog', 'skos:broader', 0.8)"
        )
        conn.commit()

        violations = lint_graph_shacl(conn, auto_heal=False)
        shapes = {v.shape for v in violations}
        assert "ConceptShape/broaderCycle" in shapes
        assert "ConceptShape/S27-disjointBroaderRelated" in shapes

        # 3. Auto-heal breaks the hierarchy cycle and removes the conflicting skos:related edges
        remaining = lint_graph_shacl(conn, auto_heal=True)
        assert remaining == []
        assert conn.execute("SELECT COUNT(*) FROM edges WHERE kind = 'skos:related'").fetchone()[0] == 0
    finally:
        conn.close()

