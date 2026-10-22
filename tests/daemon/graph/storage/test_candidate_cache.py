"""Real SQLite cache budget, FIFO, replacement and rollback invariants."""

import sqlite3

import pytest

from yaw.daemon.graph.storage import CandidateCacheStore


def test_candidate_cache_shared_budget_replacement_and_rollback() -> None:
    with sqlite3.connect(":memory:") as conn:
        store = CandidateCacheStore(conn)
        store.put("openie", "a", b"a" * 12, max_bytes=24)
        store.put("shacl", "b", b"b" * 12, max_bytes=24)
        store.put("embedding", "c", b"c" * 12, max_bytes=24)
        assert store.get("openie", "a") is None
        assert store.get("shacl", "b") == b"b" * 12
        store.put("shacl", "b", b"b" * 6, max_bytes=24)
        assert conn.execute("SELECT total_bytes FROM candidate_cache_size").fetchone()[0] == 18
        conn.commit()
        with pytest.raises(RuntimeError), conn:
            store.put("synonym", "d", b"d" * 20, max_bytes=24)
            raise RuntimeError("Rollback must restore entries evicted by an aborted transaction")
        assert store.get("embedding", "c") == b"c" * 12
        assert store.get("shacl", "b") == b"b" * 6
        assert store.get("synonym", "d") is None
        assert conn.execute("SELECT total_bytes FROM candidate_cache_size").fetchone()[0] == 18
        store.put("shacl", "b", b"oversized" * 10, max_bytes=24)
        assert not store.contains("shacl", "b")
        assert conn.execute("SELECT total_bytes FROM candidate_cache_size").fetchone()[0] == 12
        store.delete("embedding", "c")
        assert conn.execute("SELECT total_bytes FROM candidate_cache_size").fetchone()[0] == 0


def test_candidate_cache_zero_budget_never_retains_entries() -> None:
    with sqlite3.connect(":memory:") as conn:
        store = CandidateCacheStore(conn)
        store.put("embedding", "key", b"", max_bytes=0)
        assert not store.contains("embedding", "key")
        with pytest.raises(ValueError):
            store.put("embedding", "key", b"x", max_bytes=-1)
