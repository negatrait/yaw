import sqlite3
import struct

import pytest

from yaw.storage.db import clean_db, get_db


@pytest.fixture(autouse=True)
def clean_database():
    clean_db()
    yield


def test_space_integrity():
    conn = get_db()

    # 1. PRAGMA table_info(leaf_nodes) has no embedding column
    cols = {r[1] for r in conn.execute("PRAGMA table_info(leaf_nodes)").fetchall()}
    assert "embedding" not in cols

    # 2. aliases has no status column
    cols = {r[1] for r in conn.execute("PRAGMA table_info(aliases)").fetchall()}
    assert "status" not in cols

    # 3. close_matches table exists and enforces CHECK(src_id < dst_id)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(close_matches)").fetchall()}
    assert "src_id" in cols
    assert "dst_id" in cols

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO close_matches (src_id, dst_id, score) VALUES (?, ?, ?)",
            ("B", "A", 0.9),
        )

    # 4. Inserting a 300-float vector into entity_vec or fact_vec raises sqlite3.OperationalError
    wrong_vec = struct.pack("300f", *[0.1] * 300)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES (?, ?, ?)",
            ("ent", "ns", wrong_vec),
        )

    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO fact_vec (relation_id, embedding) VALUES (?, ?)",
            ("rel1", wrong_vec),
        )

    # 5. Inserting a 768-float vector works
    correct_vec = struct.pack("768f", *[0.1] * 768)
    conn.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES (?, ?, ?)",
        ("ent", "ns", correct_vec),
    )
    conn.execute(
        "INSERT INTO fact_vec (relation_id, embedding) VALUES (?, ?)",
        ("rel1", correct_vec),
    )


def test_space_s_and_r_prompt_formatters_and_entity_grounding():
    """Verify Space S/R prompt formatters and sentential definition grounding invariant."""
    from yaw.nlp.components import _entity_definition
    from yaw.nlp.embeddings import (
        format_space_r_doc,
        format_space_r_query,
        format_space_s,
    )

    # Space S formatter with and without definition
    assert format_space_s(
        "Albert Einstein", "Theoretical physicist who developed relativity."
    ) == (
        "task: sentence similarity | query: Albert Einstein: Theoretical physicist who developed relativity."
    )
    assert format_space_s("Albert Einstein", "") == (
        "task: sentence similarity | query: Albert Einstein"
    )
    assert format_space_s("Albert Einstein") == (
        "task: sentence similarity | query: Albert Einstein"
    )

    # Space R document and query formatters
    assert (
        format_space_r_doc("Einstein developed relativity.", "Physics Note")
        == "title: Physics Note | text: Einstein developed relativity."
    )
    assert format_space_r_doc("Einstein developed relativity.") == (
        "title: none | text: Einstein developed relativity."
    )
    assert format_space_r_query("Who developed general relativity?") == (
        "task: search result | query: Who developed general relativity?"
    )

    # Entity definition grounded in enclosing sentential context
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp("Albert Einstein was a theoretical physicist. He won the Nobel Prize.")
    span = doc[0:2]  # "Albert Einstein"
    definition = _entity_definition(span)
    assert definition == "Albert Einstein was a theoretical physicist."

    # Entity definition without sentential context defaults cleanly to ""
    doc_no_sent = spacy.blank("en")("Albert Einstein")
    span_no_sent = doc_no_sent[0:2]
    assert _entity_definition(span_no_sent) == ""
