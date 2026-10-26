"""Indexing work stays bounded without losing candidates or their source evidence."""

import json
import sys
from contextlib import closing
from pathlib import Path

import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import (
    find_entity_disambiguation_candidates,
    iter_disambiguation_dossiers,
    open_yaw_db,
    upsert_chunk_pointers,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_semantic_edges,
)
from yaw.daemon.graph.storage.hydration import dehydrate_passage_ast, get_cached_code_source, hydrate_chunk_pointer


def test_hydration_reuses_ast_result_but_invalidates_source_and_preserves_call_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"title": "Original", "text": "Grounded source sentence. " * 500}), encoding="utf-8")
    pointer = ChunkPointer("c", "source.json", 1, 1, "Original")
    calls = 0

    def profile(frame, event, arg):
        nonlocal calls
        if event == "call" and frame.f_code is dehydrate_passage_ast.__code__:
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        first = hydrate_chunk_pointer(pointer, tmp_path, score=0.1, max_chars=900)
        second = hydrate_chunk_pointer(pointer, tmp_path, score=0.9, max_chars=900)
        assert first.text == second.text
        assert second.score == 0.9
        assert calls == 1
        short = hydrate_chunk_pointer(pointer, tmp_path, max_chars=400)
        assert len(short.text) <= 400
        assert short.text != first.text
        assert calls == 2
        source.write_text(
            json.dumps({"title": "Changed", "text": "Different source sentence. " * 500}), encoding="utf-8"
        )
        changed = hydrate_chunk_pointer(pointer, tmp_path, max_chars=900)
        assert "Different source" in changed.text
        assert "Grounded source" not in changed.text
        assert calls == 3
        source.unlink()
        assert hydrate_chunk_pointer(pointer, tmp_path, max_chars=900).stale_on_disk
    finally:
        sys.setprofile(previous)


def test_targeted_candidates_equal_full_lexical_search_without_duplicates(tmp_path: Path) -> None:
    with closing(open_yaw_db(tmp_path / "pairs.db")) as conn:
        upsert_entities(
            conn,
            [
                ("a", "WHO"),
                ("b", "World Health Organization"),
                ("c", "World Health Organisation"),
                ("d", "Apple"),
                ("e", "Apple Inc"),
                ("f", "alpha beta"),
                ("g", "alphabeta"),
                ("h", "Service v1"),
                ("i", "Service v2"),
                ("j", "42"),
            ],
        )
        upsert_semantic_edges(conn, [("a", "b", "owl:differentFrom", 1.0)])
        full = find_entity_disambiguation_candidates(conn, dense_top_k=0)
        assert full
        for targets in (["b"], ["a", "c", "g"], ["g", "g", "missing"], [], ["j"]):
            actual = find_entity_disambiguation_candidates(conn, targets, dense_top_k=0)
            assert actual == [p for p in full if p[0] in targets or p[1] in targets]
            assert len({(a, b) for a, b, _, _ in actual}) == len(actual)


def test_dossiers_share_profiles_split_on_budget_and_keep_every_pair_evidence(tmp_path: Path) -> None:
    with closing(open_yaw_db(tmp_path / "dossiers.db")) as conn:
        entities = [(f"e{i}", f"Concept {i}", f"Unique definition {i}") for i in range(4)]
        upsert_entities(conn, entities)
        shared = tmp_path / "shared.md"
        shared.write_text("Common evidence for all four entities.\n", encoding="utf-8")
        shared_pointer = ChunkPointer("shared", shared.name, 1, 1, "Common source")
        upsert_chunk_pointers(conn, [(shared_pointer, shared.read_text(encoding="utf-8"))])
        for eid, name, _ in entities:
            source = tmp_path / f"{eid}.md"
            source.write_text(f"Grounded evidence for {name}.\n", encoding="utf-8")
            pointer = ChunkPointer(f"c-{eid}", source.name, 1, 1, name)
            upsert_chunk_pointers(conn, [(pointer, source.read_text(encoding="utf-8"))])
            upsert_edges_with_sources(conn, [(pointer.chunk_id, eid, "passage", 1.0, pointer.chunk_id)])
            upsert_edges_with_sources(conn, [("shared", eid, "passage", 1.0, "shared")])
        candidates = [
            {
                "pair_id": f"e0|e{i}",
                "entity_a": "e0",
                "entity_b": f"e{i}",
                "name_a": "Concept 0",
                "name_b": f"Concept {i}",
                "def_a": "Unique definition 0",
                "def_b": f"Unique definition {i}",
                "dense_sim": 0.8,
                "lexical_sim": 0.9,
            }
            for i in range(1, 4)
        ]
        batches = list(iter_disambiguation_dossiers(conn, candidates, tmp_path))
        assert len(batches) == 1
        prompt = batches[0][1]
        assert prompt.count("Unique definition 0") == 1
        assert prompt.count("Grounded evidence for Concept 0.") == 1
        assert prompt.count("Common evidence for all four entities.") == 1
        for i in range(1, 4):
            assert f"# Candidate Pair `e0|e{i}`" in prompt
            assert f"Unique definition {i}" in prompt
            assert f"Grounded evidence for Concept {i}." in prompt

        single_prompts = list(iter_disambiguation_dossiers(conn, candidates, tmp_path, batch_size=1))
        limit = max(len(p) for _, p in single_prompts)
        split = list(iter_disambiguation_dossiers(conn, candidates, tmp_path, max_chars=limit))
        assert len(split) == 3
        assert [c for batch, _ in split for c in batch] == candidates
        for batch, text in split:
            assert len(text) <= limit
            assert "Unique definition 0" in text
            assert batch[0]["def_b"] in text
            assert text.count("Common evidence for all four entities.") == 1
            index = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
            assert index[0]["pair_id"] == batch[0]["pair_id"]
        with pytest.raises(ValueError, match="Complete disambiguation evidence"):
            list(iter_disambiguation_dossiers(conn, candidates, tmp_path, max_chars=100))


def test_dossier_identity_changes_when_source_evidence_changes(tmp_path: Path) -> None:
    with closing(open_yaw_db(tmp_path / "identity.db")) as conn:
        upsert_entities(conn, [("a", "Alpha", "Original definition"), ("b", "Beta", "Other definition")])
        source = tmp_path / "source.md"
        source.write_text("Original passage.\n", encoding="utf-8")
        pointer = ChunkPointer("c", source.name, 1, 1, "Source")
        upsert_chunk_pointers(conn, [(pointer, "Original passage.")])
        upsert_edges_with_sources(conn, [("c", "a", "passage", 1.0, "c")])
        candidates = [{"entity_a": "a", "entity_b": "b", "dense_sim": 0.8, "lexical_sim": 0.9}]
        original = list(iter_disambiguation_dossiers(conn, candidates, tmp_path))[0][1]
        source.write_text("Updated passage.\n", encoding="utf-8")
        edited = list(iter_disambiguation_dossiers(conn, candidates, tmp_path))[0][1]
        assert "Updated passage." in edited
        assert "Original passage." not in edited
        assert edited != original
        upsert_entities(conn, [("a", "Alpha", "Changed definition")])
        changed = list(iter_disambiguation_dossiers(conn, candidates, tmp_path))[0][1]
        assert edited != changed
        assert "Changed definition" in changed


def test_source_cache_evicts_old_ast_without_losing_hydration(tmp_path: Path) -> None:
    first_path = tmp_path / "0.md"
    first_path.write_text("First source\n", encoding="utf-8")
    first = get_cached_code_source(first_path)
    assert first is not None
    repeated = get_cached_code_source(first_path)
    assert repeated is not None
    assert repeated[0] is first[0]
    for i in range(1, 6):
        path = tmp_path / f"{i}.md"
        path.write_text(f"Source {i}\n", encoding="utf-8")
        assert get_cached_code_source(path) is not None
    reloaded = get_cached_code_source(first_path)
    assert reloaded is not None
    assert reloaded[0] is not first[0]
    assert reloaded[1] == first[1]
