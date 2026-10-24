"""Cache corruption and codec contracts exercised against actual SQLite."""

import json
import zlib

import pytest

from yaw.daemon.graph.candidate_cache import CandidatePayloadCodec, CandidateStageCache
from yaw.daemon.graph.storage import CandidateCacheStore, open_managed_yaw_db


def test_payload_codec_preserves_legacy_snapshots_and_short_source_literals():
    codec = CandidatePayloadCodec(lambda: {"small": "tiny", "empty": ""})
    original = {"text": "a tiny literal", "passage": "tiny", "values": (1, None, True)}
    assert codec.decode(codec.encode(original)) == original
    assert codec.decode('{"version":0,"instruction":"preserved"}') == {
        "version": 0,
        "instruction": "preserved",
    }
    with pytest.raises(TypeError, match="Unsupported candidate-cache value"):
        codec.encode(object())
    with pytest.raises(ValueError, match="Unknown candidate-cache type"):
        codec.decode(json.dumps({"reflection_codec": 1, "value": {"type": "unknown", "value": []}}))


def test_missing_hydration_discards_trace_without_fabricating_passage(tmp_path, caplog):
    managed = open_managed_yaw_db(tmp_path / "cache.db")
    try:
        with managed.transaction() as conn:
            store = CandidateCacheStore(conn)
        cache = CandidateStageCache(store, managed, "trace", lambda: "version")
        cache[("example",)] = {"passage": "original source"}
        assert cache.get(("example",)) is None
        assert "requires the current hydrated passage" in caplog.text
        assert ("example",) not in cache
        assert cache.misses == 1
        with pytest.raises(KeyError):
            cache[("missing",)]
        assert cache.misses == 2
    finally:
        managed.close()


@pytest.mark.parametrize("payload", [b"not compressed", zlib.compress(b'{"type":"unknown","value":[]}')])
def test_corrupt_entry_is_deleted_inside_the_existing_transaction(tmp_path, payload):
    managed = open_managed_yaw_db(tmp_path / "cache.db")
    try:
        with managed.transaction() as conn:
            store = CandidateCacheStore(conn)
            cache = CandidateStageCache(store, managed, "stage", lambda: "version")
            cache[("key",)] = "value"
            conn.execute("UPDATE candidate_stage_cache SET payload = ?", (payload,))
            assert cache.get(("key",)) is None
            assert conn.execute("SELECT count(*) FROM candidate_stage_cache").fetchone()[0] == 0
            assert conn.in_transaction
        assert cache.hits == 0
        assert cache.misses == 1
    finally:
        managed.close()
