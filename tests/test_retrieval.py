"""Unmocked integration tests for zero-LLM VectorSearch ranking, SQLite access, and disk hydration."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import sqlite_vec
from numpy.typing import NDArray

from yaw.core import ChunkPointer, RetrievalParams, YawConfig, format_query_for_embedding
from yaw.pipeline import close_pipeline, start_pipeline, update_pipeline_once
from yaw.retrieval import (
    VectorSearch,
    build_vector_search,
    extract_text_features,
    format_evidence_route,
    hydrate_passage,
    hydrate_ranked_passages,
    open_readonly_db,
    query_external_hints,
    search_vectors,
)


def _topic_embed(texts: list[str], *, dim: int = 16) -> NDArray[np.float32]:
    topics = (
        ("apollo", "lunar", "moon", "armstrong", "commander"),
        ("instrument", "xylophone", "zither", "music", "wapakoneta"),
        ("shuttle", "orbital", "booster", "cape"),
        ("mars", "rover", "perseverance", "crater"),
    )
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        low = text.lower()
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        base = (np.frombuffer(digest * 2, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.05
        for t_idx, kws in enumerate(topics):
            for kw in kws:
                if kw in low:
                    base[t_idx] += 2.0
        rows[row_idx] = base / max(float(np.linalg.norm(base)), 1e-12)
    return rows


def _make_cfg(root: Path, *, chunk_size: int = 400, chunk_overlap: int = 40, min_chunk_size: int = 20) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )


def test_empty_db_vector_search(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    session = start_pipeline(cfg, embed_fn=lambda ts: _topic_embed(ts, dim=16), vector_dim=16, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)

    conn = open_readonly_db(cfg.db_path)
    try:
        vs = build_vector_search(conn)
        assert isinstance(vs, VectorSearch)
        assert vs.size == 0
        q_vec = _topic_embed([format_query_for_embedding("anything", "distinct-titled")], dim=16)[0]
        trace = search_vectors(vs, q_vec, "anything", top_k=5)
        assert trace.ranked_chunks == []
        assert trace.num_particles == 0
        assert hydrate_ranked_passages(vs, trace.ranked_chunks, cfg.corpus_root, top_k=5) == ()
    finally:
        conn.close()


def test_multihop_search_maxp_hydration_and_staleness(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=140, chunk_overlap=20, min_chunk_size=20)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    (cfg.corpus_root / "alpha.md").write_text(
        "# Apollo 11 Mission\nNeil Armstrong served as the Apollo 11 Lunar Module commander.\n",
        encoding="utf-8",
    )
    bridge_file = cfg.corpus_root / "bridge.md"
    bridge_file.write_text(
        "# Wapakoneta Musician\nNeil Armstrong lived in Ohio.\n\n"
        "## Early Childhood\nNeil Armstrong played the XylophoneZither during childhood.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "noise.md").write_text(
        "# Mars Crater\nPerseverance rover explored Jezero crater on Mars.\n",
        encoding="utf-8",
    )
    long_line = "Alpha paragraph sentence one about orbital mechanics. " * 6
    (cfg.corpus_root / "multi.json").write_text(
        json.dumps({"title": "Orbital Long Doc", "text": long_line.strip()}),
        encoding="utf-8",
    )

    session = start_pipeline(cfg, embed_fn=lambda ts: _topic_embed(ts, dim=16), vector_dim=16, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)

    conn = open_readonly_db(cfg.db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM chunks")

        vs = build_vector_search(conn)
        assert vs.size >= 4

        query = "What unusual instrument did the Apollo 11 commander play?"
        q_vec = _topic_embed([format_query_for_embedding(query, "distinct-titled")], dim=16)[0]
        fts_scores, vec_scores = query_external_hints(conn, query, q_vec, fts_k=20, vec_k=20)
        assert fts_scores and vec_scores

        trace = search_vectors(
            vs,
            q_vec,
            query,
            top_k=max(2 * 4, 50),
            fts_scores=fts_scores,
            vec_scores=vec_scores,
            params=RetrievalParams(),
        )
        assert trace.ranked_chunks
        if trace.route:
            route_txt = format_evidence_route(trace.route, max_chars=240)
            assert route_txt and len(route_txt) <= 240

        hydrated = hydrate_ranked_passages(vs, trace.ranked_chunks, cfg.corpus_root, top_k=4)
        paths = [h.pointer.rel_posix_path for h in hydrated]
        assert "alpha.md" in paths
        assert "bridge.md" in paths
        assert all(not h.stale_on_disk and h.text for h in hydrated)

        # Verify MaxP deduplicates multiple sub-chunks of multi.json sharing (rel_path, 1, 1)
        json_ptrs = [p for p in vs.pointers if p.rel_posix_path == "multi.json"]
        assert len(json_ptrs) >= 2
        dup_ranked = [(int(p.chunk_id), 0.9 - 0.1 * idx) for idx, p in enumerate(json_ptrs)]
        deduped = hydrate_ranked_passages(vs, dup_ranked, cfg.corpus_root, top_k=5)
        assert len(deduped) == 1
        assert deduped[0].pointer.chunk_id == json_ptrs[0].chunk_id
        assert deduped[0].pointer.title == "Orbital Long Doc"

        # Verify lexical_grep empty-vector path (vec_k=0) and island hydration + cropping
        fts_only, vec_empty = query_external_hints(
            conn, "XylophoneZither", np.zeros((0,), np.float32), fts_k=5, vec_k=0
        )
        assert vec_empty == {}
        assert any(vs.pointers[vs.id_to_pos[cid]].rel_posix_path == "bridge.md" for cid in fts_only)

        bridge_ptr = next(
            p
            for p in vs.pointers
            if p.rel_posix_path == "bridge.md" and "XylophoneZither" in hydrate_passage(cfg.corpus_root, p).text
        )
        island_h = hydrate_passage(cfg.corpus_root, bridge_ptr, include_island=True, max_chars=60)
        assert not island_h.stale_on_disk
        assert "[... cropped ...]" in island_h.text

        # Post-index mutation and deletion must return stale_on_disk=True with empty text
        bridge_file.write_text("# Wapakoneta Musician\nContent altered after indexing.\n", encoding="utf-8")
        mutated = hydrate_passage(cfg.corpus_root, bridge_ptr)
        assert mutated.stale_on_disk is True
        assert mutated.text == ""

        noise_ptr = next(p for p in vs.pointers if p.rel_posix_path == "noise.md")
        (cfg.corpus_root / "noise.md").unlink()
        deleted = hydrate_passage(cfg.corpus_root, noise_ptr)
        assert deleted.stale_on_disk is True
        assert deleted.text == ""

        # Out-of-bounds span validation
        with pytest.raises(ValueError, match="Invalid hydration line range"):
            hydrate_passage(
                cfg.corpus_root, ChunkPointer(chunk_id=1, rel_posix_path="alpha.md", start_line=0, end_line=1)
            )
    finally:
        conn.close()


def test_retrieval_module_isolation_and_nlp() -> None:
    terms, (main_ents, attr_ents) = extract_text_features("Apollo 11", "Neil Armstrong commanded Eagle in July.")
    assert "apollo" in terms and "neil" in terms
    assert any("Neil Armstrong" in e for e in main_ents)
    assert isinstance(attr_ents, list)

    src = (Path(__file__).resolve().parent.parent / "yaw" / "retrieval.py").read_text(encoding="utf-8")
    low = src.lower()
    assert "manifold" not in low and "hmm" not in low
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("dspy", "cocoindex", "yaw.daemon", "yaw.pipeline"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(("dspy", "cocoindex", "yaw.daemon", "yaw.pipeline"))
    assert sqlite_vec.__name__ == "sqlite_vec"
