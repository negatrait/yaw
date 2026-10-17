"""Real, unmocked tests for yaw.db SQLite engine, sqlite-vec KNN, and contentless FTS5."""

from pathlib import Path

import numpy as np
import pytest

from yaw.core import ChunkPointer, OpenIERecord
from yaw.daemon.graph.storage.db import (
    delete_chunks,
    deserialize_f32_vector,
    find_entity_synonym_candidates,
    get_chunk_pointer,
    get_chunk_pointers,
    get_entity_vectors,
    get_openie_checkpoints,
    initialize_schema,
    open_yaw_db,
    save_openie_checkpoints,
    search_chunk_vectors_knn,
    search_chunks_fts,
    search_entity_vectors_knn,
    search_fact_vectors_knn,
    search_facts_fts,
    serialize_f32_vector,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_vectors,
    upsert_fact_vectors,
    upsert_facts,
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
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') ORDER BY name;"
        )
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
                (p1, "Alan Turing was an English mathematician and computer scientist at Bletchley Park."),
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
        upsert_facts(conn, [
            ("f1", "e1", "born in", "e2", '["Alan Turing", "born in", "London"]'),
            ("f2", "e2", "capital of", "e3", '["London", "capital of", "UK"]'),
        ])
        upsert_fact_vectors(conn, [("f1", [1.0, 0.0]), ("f2", [0.0, 1.0])])

        # Edges and provenance
        # Passage edges: c1 <-> e1, c1 <-> e2; c2 <-> e2, c2 <-> e3
        upsert_edges_with_sources(conn, [
            ("c1", "e1", "passage", 1.0, "c1"),
            ("c1", "e2", "passage", 1.0, "c1"),
            ("c2", "e2", "passage", 1.0, "c2"),
            ("c2", "e3", "passage", 1.0, "c2"),
            # Fact edges: e1 <-> e2 supported by c1, e2 <-> e3 supported by c2
            ("e1", "e2", "fact", 1.0, "c1"),
            ("e2", "e3", "fact", 1.0, "c2"),
        ])

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
        inc_candidates = find_entity_synonym_candidates(
            conn, entity_ids=["e2"], threshold=0.8, top_k=5
        )
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
                (p_dense, "Enigma rotor cipher machine. Enigma rotors and enigma plugboard broke enigma codes."),
                (p_sparse, "The history of the modern world includes many machines and one enigma."),
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
        upsert_facts(conn, [("f_shared", "ent-0", "linked_to", "ent-1199", '["Entity 0", "linked_to", "Entity 1199"]')])
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
            ("fact-1", "ent-curie", "discovered", "ent-radium", '["Marie Curie", "discovered", "radium"]'),
            ("fact-2", "ent-einstein", "developed", "ent-relativity", '["Albert Einstein", "developed", "general relativity"]'),
            ("fact-3", "ent-turing", "broke", "ent-enigma", '["Alan Turing", "broke", "Enigma code at Bletchley"]'),
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


