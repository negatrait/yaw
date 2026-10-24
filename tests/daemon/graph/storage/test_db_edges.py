"""Real legacy SQLite migrations and transaction boundary contracts."""

import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from yaw.daemon.graph.storage import (
    apply_shacl_repairs,
    deserialize_f32_vector,
    find_entity_disambiguation_candidates,
    get_entities_by_label_norm,
    get_entity_metadata_batch,
    get_facts_with_meta_by_ids,
    initialize_schema,
    open_readonly_yaw_db,
    open_yaw_db,
    resolve_entity_labels,
    search_chunks_fts,
    search_entity_vectors_knn,
    search_facts_fts,
    set_meta,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
    upsert_facts,
)


def test_legacy_schema_migrates_rows_labels_edges_and_fact_search_without_loss(tmp_path):
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as legacy:
        legacy.executescript("""
            CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, rel_posix_path TEXT NOT NULL,
                start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, title TEXT NOT NULL, content_hash TEXT);
            INSERT INTO chunks VALUES ('chunk','source.md',1,2,'Source','legacy-hash');
            CREATE TABLE openie_chunks(chunk_id TEXT PRIMARY KEY, entities_json TEXT NOT NULL,
                triples_json TEXT NOT NULL);
            INSERT INTO openie_chunks VALUES ('chunk','["Ada"]','[]');
            CREATE TABLE entities(entity_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE);
            INSERT INTO entities VALUES ('ada','Ada');
            CREATE TABLE facts(fact_id TEXT PRIMARY KEY,subj_id TEXT NOT NULL,pred TEXT NOT NULL,
                obj_id TEXT NOT NULL,raw_triple_json TEXT NOT NULL);
            INSERT INTO facts VALUES ('fact','ada','wrote','notes','["Ada","wrote","Notes"]');
            INSERT INTO facts VALUES ('broken','ada','legacyPredicate','notes','not json');
            CREATE TABLE edges(src_id TEXT NOT NULL,dst_id TEXT NOT NULL,kind TEXT NOT NULL,
                weight REAL NOT NULL,PRIMARY KEY(src_id,dst_id,kind));
            INSERT INTO edges VALUES ('ada','notes','synonymy',0.8);
            INSERT INTO edges VALUES ('ada','orphan','retired-kind',0.2);
        """)
    with closing(open_yaw_db(path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        assert "content_hash" not in columns
        assert {"ast_island_start", "ast_island_end"} <= columns
        assert tuple(conn.execute("SELECT chunk_id,ast_island_start,ast_island_end FROM chunks").fetchone()) == (
            "chunk",
            0,
            0,
        )
        assert conn.execute("SELECT structured_triples_json FROM openie_chunks").fetchone()[0] == "[]"
        assert tuple(conn.execute("SELECT entity_id,name,definition FROM entities").fetchone()) == ("ada", "Ada", "")
        assert tuple(conn.execute("SELECT label_norm,label_kind,raw_label,chunk_id FROM entity_labels").fetchone()) == (
            "ada",
            "skos:prefLabel",
            "Ada",
            "__legacy__",
        )
        assert [tuple(row) for row in conn.execute("SELECT src_id,dst_id,kind,weight FROM edges")] == [
            ("ada", "notes", "skos:related", 0.8),
        ]
        assert search_facts_fts(conn, "Ada")[0][0] == "fact"
        assert search_facts_fts(conn, "legacyPredicate")[0][0] == "broken"
        # The old UNIQUE(name) constraint must be removed for distinct senses.
        upsert_entities(conn, [("ada-sense-two", "Ada", "Different entity")])
        assert conn.execute("SELECT count(*) FROM entities WHERE name='Ada'").fetchone()[0] == 2
        before = conn.execute("SELECT count(*) FROM fact_fts").fetchone()[0]
        initialize_schema(conn)
        assert conn.execute("SELECT count(*) FROM fact_fts").fetchone()[0] == before


def test_failed_owned_transaction_rolls_back_partial_writes_and_outer_owner_retains_control(tmp_path):
    with closing(open_yaw_db(tmp_path / "tx.db")) as conn:
        rows = json.loads('[["first","Valid"],["second",null]]')
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            upsert_entities(conn, rows)
        assert not conn.in_transaction
        assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 0
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            upsert_entities(conn, rows)
        assert conn.in_transaction
        assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 1
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 0


def test_fact_metadata_preserves_original_triples_and_optional_entity_definitions(tmp_path):
    with closing(open_yaw_db(tmp_path / "meta.db")) as conn:
        upsert_entities(conn, [("ada", "Ada", "Mathematician"), ("notes", "Notes", "Publication")])
        upsert_facts(conn, [("fact", "ada", "wrote", "notes", '["Ada","wrote","Notes"]', "Ada wrote Notes.")])
        assert get_facts_with_meta_by_ids(conn, []) == {}
        assert get_facts_with_meta_by_ids(conn, ["missing"]) == {}
        assert get_facts_with_meta_by_ids(conn, ["fact", "missing"])["fact"] == {
            "fact_id": "fact",
            "subj_id": "ada",
            "pred": "wrote",
            "obj_id": "notes",
            "triple": ("Ada", "wrote", "Notes"),
            "definition": "Ada wrote Notes.",
            "subj_def": "Mathematician",
            "obj_def": "Publication",
        }
        conn.execute("DELETE FROM entities WHERE entity_id='notes'")
        assert get_facts_with_meta_by_ids(conn, ["fact"])["fact"]["obj_def"] == ""


@pytest.mark.parametrize("blob", [b"", b"x", b"12345"])
def test_vector_deserialization_rejects_non_float32_byte_lengths(blob):
    with pytest.raises(ValueError, match="Invalid float32 byte buffer length"):
        deserialize_f32_vector(blob)


def test_fts_rejects_empty_queries_and_nonpositive_limits_without_sql_syntax_errors(tmp_path):
    with closing(open_yaw_db(tmp_path / "fts.db")) as conn:
        for search in (search_chunks_fts, search_facts_fts):
            assert search(conn, "!!!") == []
            assert search(conn, "a") == []
            assert search(conn, "term", top_k=0) == []


def test_label_priority_is_global_across_sources_even_before_entity_hydration(tmp_path):
    with closing(open_yaw_db(tmp_path / "labels.db")) as conn:
        upsert_entity_labels(
            conn,
            [
                ("canonical", "pending", "skos:prefLabel", "Canonical", "a"),
                ("alternative", "pending", "skos:prefLabel", "Alternative", "b"),
                ("canonical", "pending", "skos:hiddenLabel", "Canonical", "c"),
                ("hidden", "pending", "skos:hiddenLabel", "Hidden", "a"),
                ("hidden", "pending", "skos:altLabel", "Hidden", "b"),
            ],
        )
        labels = {tuple(row) for row in conn.execute("SELECT label_norm,label_kind,chunk_id FROM entity_labels")}
        assert ("canonical", "skos:prefLabel", "c") in labels
        assert ("alternative", "skos:altLabel", "b") in labels
        assert ("hidden", "skos:altLabel", "a") in labels
        assert ("hidden", "skos:altLabel", "b") in labels
        upsert_entities(conn, [("pending", "Alternative", "Restored entity")])
        upsert_entity_labels(
            conn,
            [
                ("alternative", "pending", "skos:prefLabel", "Alternative", "c"),
                ("secret", "pending", "skos:hiddenLabel", "Secret", "a"),
                ("   ", "pending", "skos:altLabel", "", "a"),
            ],
        )
        assert (
            conn.execute("SELECT DISTINCT label_kind FROM entity_labels WHERE label_norm='canonical'").fetchone()[0]
            == "skos:altLabel"
        )
        assert (
            conn.execute("SELECT label_kind FROM entity_labels WHERE label_norm='secret'").fetchone()[0]
            == "skos:hiddenLabel"
        )
        before = conn.execute("SELECT count(*) FROM entity_labels").fetchone()[0]
        with pytest.raises(ValueError, match="Invalid SKOS label_kind"):
            upsert_entity_labels(conn, [("invalid", "pending", "invalid", "Invalid", "a")])
        assert conn.execute("SELECT count(*) FROM entity_labels").fetchone()[0] == before
        assert get_entities_by_label_norm(conn, []) == {}
        assert get_entities_by_label_norm(conn, [" "]) == {}


def test_label_resolution_uses_available_vectors_and_deterministic_missing_vector_ties(tmp_path):
    with closing(open_yaw_db(tmp_path / "sense.db")) as conn:
        upsert_entities(conn, [("a", "Mercury", "Planet"), ("b", "Mercury", "Metal"), ("c", "Unique", "Unique")])
        upsert_entity_labels(
            conn,
            [
                ("mercury", "a", "skos:prefLabel", "Mercury", "one"),
                ("mercury", "b", "skos:prefLabel", "Mercury", "two"),
                ("unique", "c", "skos:prefLabel", "Unique", "three"),
            ],
        )
        assert resolve_entity_labels(conn, []) == {}
        assert resolve_entity_labels(conn, ["unique"], [1.0, 0.0]) == {"unique": ["c"]}
        assert resolve_entity_labels(conn, ["mercury"], [0.0, 0.0]) == {"mercury": ["a", "b"]}
        assert resolve_entity_labels(conn, ["mercury", "unique"], [1.0, 0.0]) == {"mercury": ["a"], "unique": ["c"]}
        upsert_entity_vectors(conn, [("a", [0.0, 0.0]), ("b", [1.0, 0.0])])
        assert resolve_entity_labels(conn, ["mercury"], [1.0, 0.0]) == {"mercury": ["b"]}


@pytest.mark.parametrize(
    ("name_a", "name_b", "threshold", "lexical"),
    [
        ("", "Beta", 0.8, 0.0),
        ("WHO", "World Health Organization", 0.8, 0.85),
        ("United States of America", "USA", 0.8, 0.85),
        ("alpha beta", "alphabeta", 0.8, 18 / 19),
        ("alpha beta", "alphabeta", 0.98, 1.0),
    ],
)
def test_disambiguation_keeps_dense_and_lexical_channels_independent(tmp_path, name_a, name_b, threshold, lexical):
    with closing(open_yaw_db(tmp_path / "candidate.db")) as conn:
        upsert_entities(conn, [("a", name_a), ("b", name_b)])
        upsert_entity_vectors(conn, [("a", np.array([1.0, 0.0])), ("b", np.array([1.0, 0.0]))])
        pairs = find_entity_disambiguation_candidates(conn, lexical_threshold=threshold)
        assert len(pairs) == 1
        assert pairs[0][:2] == ("a", "b")
        assert pairs[0][2] == pytest.approx(1.0)
        assert pairs[0][3] == pytest.approx(lexical)


def test_open_readonly_yaw_db_enforces_ro_uri_and_busy_timeout(tmp_path):
    db_file = tmp_path / "readonly.db"
    with closing(open_yaw_db(db_file)) as rw_conn:
        set_meta(rw_conn, "version", "1")
        upsert_entity_vectors(rw_conn, [("entity-1", [1.0, 0.0])])
    with closing(open_readonly_yaw_db(db_file, timeout=2.5)) as ro_conn:
        assert ro_conn.row_factory is sqlite3.Row
        assert ro_conn.execute("PRAGMA busy_timeout").fetchone()[0] == 2500
        assert ro_conn.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()["value"] == "1"
        knn_hits = search_entity_vectors_knn(ro_conn, [1.0, 0.0], top_k=1)
        assert knn_hits == [("entity-1", pytest.approx(1.0))]
        with pytest.raises(sqlite3.OperationalError):
            ro_conn.execute("INSERT INTO meta (key, value) VALUES ('x', 'y')")


def test_get_entity_metadata_batch_handles_empty_and_multi_batch_queries(tmp_path):
    with closing(open_yaw_db(tmp_path / "batch_meta.db")) as conn:
        assert get_entity_metadata_batch(conn, []) == {}
        rows = [(f"entity-{i:04d}", f"Name {i}", f"Def {i}" if i % 2 == 0 else "") for i in range(950)]
        upsert_entities(conn, rows)
        requested = [r[0] for r in rows] + ["entity-missing"]
        fetched = get_entity_metadata_batch(conn, requested)
        assert len(fetched) == 950
        assert fetched["entity-0000"] == ("Name 0", "Def 0")
        assert fetched["entity-0949"] == ("Name 949", "")
        assert "entity-missing" not in fetched


def test_apply_shacl_repairs_executes_closed_world_actions_and_custom_merge_callback(tmp_path):
    with closing(open_yaw_db(tmp_path / "repairs.db")) as conn:
        assert apply_shacl_repairs(conn, []) == 0
        upsert_entities(
            conn,
            [
                ("entity-1", "Alpha", ""),
                ("entity-2", "Alpha Alias", "Alias def"),
                ("entity-3", "Beta", "Beta def"),
                ("entity-orphan", "Orphan", ""),
            ],
        )
        upsert_facts(conn, [("fact-1", "entity-1", "relates", "entity-3", '["Alpha","relates","Beta"]', "")])

        def failing_merge(src: str, dst: str, lkind: str) -> bool:
            raise RuntimeError("Mid-batch repair failure must roll back owned transaction")

        with pytest.raises(RuntimeError, match="Mid-batch repair failure"):
            apply_shacl_repairs(
                conn,
                [
                    {"focus_node": "entity-1", "action": "set_definition", "value": "Should Roll Back"},
                    {"focus_node": "entity-2|entity-1", "action": "merge_altLabel", "target_node": "entity-1"},
                ],
                merge_entities_fn=failing_merge,
            )
        assert not conn.in_transaction
        assert get_entity_metadata_batch(conn, ["entity-1"])["entity-1"] == ("Alpha", "")

        merged_calls: list[tuple[str, str, str]] = []

        def custom_merge(src: str, dst: str, lkind: str) -> bool:
            merged_calls.append((src, dst, lkind))
            return True

        applied = apply_shacl_repairs(
            conn,
            [
                {"focus_node": "", "action": "set_definition", "value": "ignored"},
                {"focus_node": "entity-1", "action": "set_definition", "value": "Alpha definition"},
                {"focus_node": "fact-1", "action": "rewrite_definition", "value": "Fact definition"},
                {"focus_node": "entity-1", "action": "set_pref_label", "value": "Alpha Canonical"},
                {"focus_node": "entity-2|entity-1", "action": "merge_altLabel", "target_node": "entity-1"},
                {"focus_node": "entity-1|entity-3", "action": "assert_differentFrom"},
                {"focus_node": "entity-orphan", "action": "delete_orphan"},
            ],
            merge_entities_fn=custom_merge,
        )
        assert applied == 6
        assert merged_calls == [("entity-2", "entity-1", "skos:altLabel")]
        assert get_entity_metadata_batch(conn, ["entity-1", "entity-orphan"]) == {
            "entity-1": ("Alpha Canonical", "Alpha definition")
        }
        assert conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-1'").fetchone()[0] == "Fact definition"
        diff_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'owl:differentFrom'").fetchall()
        assert len(diff_edges) == 2
