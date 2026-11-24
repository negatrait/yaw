"""Unmocked real-substrate tests for M_FINAL_1B 1-bit RaBitQ, cyclic diffusion, unicode61 FTS5, and hydration."""

from __future__ import annotations

import ast
import builtins
import contextlib
import hashlib
import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import yaw.retrieval as retrieval_mod
import yaw.retrieval.hydration as retrieval_hydration
import yaw.retrieval.lexical as retrieval_lexical
import yaw.retrieval.search as retrieval_search
from yaw.core import ChunkPointer, YawConfig, YawRuntimeError
from yaw.pipeline import close_pipeline, open_managed_db, reconcile_index_tables, start_pipeline, update_pipeline_once
from yaw.retrieval import (
    NO_MATCHES_TEXT,
    SearchResult,
    build_recall_payload,
    build_retrieval_index,
    compute_db_fingerprints,
    compute_index_artifacts,
    dedup_ranked_positions,
    dump_recall_json,
    empty_retrieval_index,
    expand_citation_neighbors,
    format_evidence_route,
    format_recall_text,
    hydrate_passage,
    hydrate_ranked_passages,
    lexical_search,
    normalize_fts_unicode61,
    open_readonly_db,
    sanitize_slip_label,
    search_index,
)
from yaw.retrieval.graph import lists_to_csr
from yaw.retrieval.types import CACHE_MAX_ENTRIES


def _make_cfg(root: Path, *, chunk_size: int = 300, chunk_overlap: int = 0, min_chunk_size: int = 20) -> YawConfig:
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


def _topic_embed(texts: list[str], *, dim: int = 64) -> NDArray[np.float32]:
    topics = (
        ("alpha", "lattice", "manifold", "protocol"),
        ("gamma", "theorem", "xylophone", "harmonic"),
        ("orbital", "booster", "valve", "emergency", "calibration"),
        ("nordkanal", "dienst", "muller", "mueller", "strasse", "café", "cafe"),
        ("geology", "basalt", "limestone", "survey"),
    )
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for idx_row, text in enumerate(texts):
        low = text.casefold()
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (dim + len(digest) - 1) // len(digest)
        vec = (np.frombuffer(digest * repeats, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.005
        for t_idx, kws in enumerate(topics):
            for kw in kws:
                if kw in low:
                    vec[t_idx] += 3.0
        rows[idx_row] = vec / max(float(np.linalg.norm(vec)), 1e-12)
    return rows


def _identical_embed(texts: list[str], *, dim: int = 64) -> NDArray[np.float32]:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    out[:, 0] = 1.0
    return out


def _index_corpus(
    cfg: YawConfig,
    *,
    embed_fn=_topic_embed,
) -> None:
    session = start_pipeline(cfg, embed_fn=embed_fn, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)


def test_retrieval_module_isolation_and_unicode61_normalization() -> None:
    assert retrieval_mod.__file__ is not None
    imported: set[str] = set()
    for py_file in sorted(Path(retrieval_mod.__file__).parent.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".")[0])
    assert {"sqlite_vec", "spacy", "optuna", "dspy", "fastmcp", "cocoindex", "sentence_transformers"}.isdisjoint(
        imported
    )

    assert normalize_fts_unicode61("Café Straße") == "café strasse"
    assert normalize_fts_unicode61("Müller Über") == "müller über"
    assert normalize_fts_unicode61("碎叶") == "碎叶"
    assert normalize_fts_unicode61("京都市区") == "京都 都市 市区"


def test_empty_db_and_not_ready_conditions(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.db"
    with pytest.raises(YawRuntimeError) as exc_missing:
        open_readonly_db(missing_db)
    assert exc_missing.value.code == "INDEX_NOT_READY"

    empty_db = tmp_path / "empty.db"
    with contextlib.closing(sqlite3.connect(empty_db)) as rw_conn:
        rw_conn.execute(
            "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, embedding_f16 BLOB, rel_path TEXT, "
            "source_hash TEXT, start_line INTEGER, end_line INTEGER, char_start INTEGER, "
            "char_end INTEGER, ast_island_start INTEGER, ast_island_end INTEGER, title TEXT)"
        )
        rw_conn.execute(
            'CREATE VIRTUAL TABLE chunk_fts USING fts5(title, body, tokenize="unicode61 remove_diacritics 1")'
        )
        rw_conn.commit()

    with contextlib.closing(open_readonly_db(empty_db)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.size == 0
        assert empty_retrieval_index().size == 0
        q_vec = _topic_embed(["alpha"])[0]
        res = search_index(idx, q_vec, query_text="alpha", conn=ro_conn, top_k=5)
        assert res.ranked == ()
        assert res.n_candidates == 0
        assert math.isinf(res.log_bf) and res.log_bf < 0
        assert lexical_search(idx, "alpha", ro_conn, top_k=5) == []
        with pytest.raises(YawRuntimeError) as exc_unknown:
            expand_citation_neighbors(idx, 123, corpus_root=tmp_path, conn=ro_conn)
        assert exc_unknown.value.code == "INVALID_INPUT"


def test_index_not_ready_for_unmaterialized_stale_or_mismatched_metadata(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "a.md").write_text("# Alpha Doc\nAlpha material science reference.\n", encoding="utf-8")
    (cfg.corpus_root / "b.md").write_text("# Beta Doc\nBeta structural engineering notes.\n", encoding="utf-8")
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx_ok = build_retrieval_index(
            ro_conn,
            expected_embedding_model=cfg.embedding_model,
            expected_task_mode=cfg.embedding_task_mode,
        )
        assert idx_ok.size >= 2

        with pytest.raises(YawRuntimeError) as exc_model:
            build_retrieval_index(ro_conn, expected_embedding_model="different-embed-model")
        assert exc_model.value.code == "INDEX_NOT_READY"

        with pytest.raises(YawRuntimeError) as exc_mode:
            build_retrieval_index(ro_conn, expected_task_mode="search")
        assert exc_mode.value.code == "INDEX_NOT_READY"

    with contextlib.closing(sqlite3.connect(cfg.db_path)) as rw_conn:
        rw_conn.execute("UPDATE rabitq_index SET vec_sha256 = 'stale_sha' WHERE id = 1")
        rw_conn.commit()
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        with pytest.raises(YawRuntimeError) as exc_stale:
            build_retrieval_index(ro_conn)
        assert exc_stale.value.code == "INDEX_NOT_READY"

    with contextlib.closing(sqlite3.connect(cfg.db_path)) as rw_conn:
        rw_conn.execute("DROP TABLE rabitq_index")
        rw_conn.commit()
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        with pytest.raises(YawRuntimeError) as exc_drop_r:
            build_retrieval_index(ro_conn)
        assert exc_drop_r.value.code == "INDEX_NOT_READY"


def test_missing_fts_vocab_corrupt_blobs_and_conn_none_raise_consistency_error(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "a.md").write_text("# Alpha\nFirst sample passage.\n", encoding="utf-8")
    (cfg.corpus_root / "b.md").write_text("# Beta\nSecond sample passage.\n", encoding="utf-8")
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        q_vec = _topic_embed(["sample"])[0]
        with pytest.raises(YawRuntimeError) as exc_s_none:
            search_index(idx, q_vec, query_text="sample", conn=None)
        assert exc_s_none.value.code == "CONSISTENCY_ERROR"
        with pytest.raises(YawRuntimeError) as exc_l_none:
            lexical_search(idx, "sample", None)
        assert exc_l_none.value.code == "CONSISTENCY_ERROR"
        with pytest.raises(YawRuntimeError) as exc_e_none:
            expand_citation_neighbors(idx, idx.chunk_ids[0], corpus_root=cfg.corpus_root, conn=None)
        assert exc_e_none.value.code == "CONSISTENCY_ERROR"

    # Corrupt rabitq_index codes_blob -> CONSISTENCY_ERROR
    with contextlib.closing(sqlite3.connect(cfg.db_path)) as rw_conn:
        rw_conn.execute("UPDATE rabitq_index SET codes_blob = X'0001' WHERE id = 1")
        rw_conn.commit()
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        with pytest.raises(YawRuntimeError) as exc_blob:
            build_retrieval_index(ro_conn)
        assert exc_blob.value.code == "CONSISTENCY_ERROR"

    # Dropped chunk_fts -> CONSISTENCY_ERROR
    no_fts = tmp_path / "no_fts"
    cfg_no_fts = _make_cfg(no_fts)
    assert cfg_no_fts.corpus_root is not None and cfg_no_fts.db_path is not None
    cfg_no_fts.corpus_root.mkdir(parents=True)
    (cfg_no_fts.corpus_root / "x.md").write_text("# Alpha\nSome body text here.\n", encoding="utf-8")
    (cfg_no_fts.corpus_root / "y.md").write_text("# Beta\nMore body text here.\n", encoding="utf-8")
    _index_corpus(cfg_no_fts)
    with contextlib.closing(sqlite3.connect(cfg_no_fts.db_path)) as rw_conn:
        rw_conn.execute("DROP TABLE chunk_fts")
        rw_conn.commit()
    with contextlib.closing(open_readonly_db(cfg_no_fts.db_path)) as ro_conn:
        idx_nf = build_retrieval_index(ro_conn)
        q_vec = _topic_embed(["body"])[0]
        with pytest.raises(YawRuntimeError) as exc_fts_s:
            search_index(idx_nf, q_vec, query_text="body", conn=ro_conn)
        assert exc_fts_s.value.code == "CONSISTENCY_ERROR"
        with pytest.raises(YawRuntimeError) as exc_fts_l:
            lexical_search(idx_nf, "body", ro_conn)
        assert exc_fts_l.value.code == "CONSISTENCY_ERROR"


def test_multihop_cyclic_diffusion_noisyor_dpp_maxp_and_staleness(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=260, min_chunk_size=20)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    (cfg.corpus_root / "doc_a.md").write_text(
        "# Alpha Lattice\nAlpha Lattice cites Beta Manifold and Delta Protocol for coordinate staging.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "doc_b.md").write_text(
        "# Beta Manifold\nBeta Manifold proves the bridge between Gamma Theorem and Alpha Lattice.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "doc_c.md").write_text(
        "# Gamma Theorem\nGamma Theorem extends Beta Manifold with harmonic XylophoneZither proof.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "doc_d.md").write_text(
        "# Delta Protocol\nDelta Protocol confirms Alpha Lattice intersection with Beta Manifold.\n",
        encoding="utf-8",
    )
    long_paras = [
        f"Booster section {k}: " + " ".join(f"Orbital booster spec calibration step {k}_{j}." for j in range(1, 16))
        for k in range(1, 6)
    ]
    (cfg.corpus_root / "long_multi.md").write_text(
        "# Booster Spec\nCross-check Booster Backup for orbital booster staging.\n\n" + "\n\n".join(long_paras) + "\n",
        encoding="utf-8",
    )
    backup_file = cfg.corpus_root / "backup.md"
    backup_file.write_text(
        "# Booster Backup\nSee Booster Spec. Orbital booster emergency valve procedure.\n",
        encoding="utf-8",
    )
    for i in range(6):
        (cfg.corpus_root / f"geology_{i}.md").write_text(
            f"# Geology Sample {i}\nBasalt limestone outcrop survey log entry {i}.\n",
            encoding="utf-8",
        )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.num_citation_edges > 0

        # Multi-hop cyclic citation diffusion: doc_c shares zero lexical tokens or topic axis with "alpha lattice"
        q_cyc = _topic_embed(["alpha lattice"])[0]
        res_cyc = search_index(idx, q_cyc, query_text="Alpha Lattice", conn=ro_conn, top_k=12)
        paths_cyc = [idx.pointers[idx.id_to_pos[cid]].rel_posix_path for cid, _ in res_cyc.ranked]
        assert "doc_a.md" in paths_cyc[:3]
        assert "doc_b.md" in paths_cyc[:4]
        assert "doc_c.md" in paths_cyc
        score_by_path = {idx.pointers[idx.id_to_pos[cid]].rel_posix_path: sc for cid, sc in res_cyc.ranked}
        assert score_by_path["doc_c.md"] > score_by_path["geology_0.md"]

        # Noisy-OR across >= 2 title references in query
        q_multi = _topic_embed(["alpha protocol"])[0]
        res_nor = search_index(
            idx,
            q_multi,
            query_text="How do Alpha Lattice and Delta Protocol connect via Beta Manifold?",
            conn=ro_conn,
            top_k=5,
        )
        assert isinstance(res_nor, SearchResult)
        nor_paths = [idx.pointers[idx.id_to_pos[cid]].rel_posix_path for cid, _ in res_nor.ranked]
        assert {"doc_a.md", "doc_d.md"}.issubset(set(nor_paths))

        # Rank-1 Cholesky DPP + MaxP deferral: multi-chunk doc must not starve backup.md
        q_boost = _topic_embed(["orbital booster"])[0]
        res_boost = search_index(idx, q_boost, query_text="orbital booster", conn=ro_conn, top_k=50)
        top2 = hydrate_ranked_passages(idx, res_boost.ranked, cfg.corpus_root, top_k=2)
        assert {hp.pointer.rel_posix_path for hp in top2} == {"backup.md", "long_multi.md"}
        assert all(not hp.stale_on_disk for hp in top2)

        # Disk mutation -> stale_on_disk=True and empty text on hydration
        backup_file.write_text("# Booster Backup\nMutated after indexing.\n", encoding="utf-8")
        stale = hydrate_ranked_passages(idx, res_boost.ranked, cfg.corpus_root, top_k=2)
        by_path = {hp.pointer.rel_posix_path: hp for hp in stale}
        assert by_path["backup.md"].stale_on_disk is True
        assert by_path["backup.md"].text == ""
        assert by_path["long_multi.md"].stale_on_disk is False


def test_unicode61_diacritics_koehn_knight_and_cjk_bigrams(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    (cfg.corpus_root / "umlaut.md").write_text(
        "# Müller Brücke\nDr. Müller lives near Hauptstraße and Café Übertragung.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "ascii_umlaut.md").write_text(
        "# Baecker Weg\nBaecker arbeitet in der Sonnenstrasse.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "decompound_left.md").write_text(
        "# Canal Log\nNordkanal water lock maintenance schedule.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "decompound_right.md").write_text(
        "# Service Log\nDienst shift roster for canal bridge.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "cjk.md").write_text(
        "# Silk Road Notes\nAncient travelers crossed 京都市区 before reaching 碎叶\n",
        encoding="utf-8",
    )
    for i in range(4):
        (cfg.corpus_root / f"filler_{i}.md").write_text(
            f"# Filler Note {i}\nStandard baseline engineering entry {i}.\n",
            encoding="utf-8",
        )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        by_cid = {ptr.chunk_id: ptr.rel_posix_path for ptr in idx.pointers}

        # Umlaut / ß / diacritic equivalence under unicode61 remove_diacritics 1
        hits_ae = [by_cid[cid] for cid, _ in lexical_search(idx, "Mueller Strasse Cafe Uebertragung", ro_conn, top_k=5)]
        assert "umlaut.md" in hits_ae

        hits_rev = [by_cid[cid] for cid, _ in lexical_search(idx, "Bäcker", ro_conn, top_k=5)]
        assert "ascii_umlaut.md" in hits_rev

        # Koehn-Knight compound splitting ("Nordkanaldienst" -> "Nordkanal" + "Dienst")
        hits_kk = [by_cid[cid] for cid, _ in lexical_search(idx, "Nordkanaldienst", ro_conn, top_k=5)]
        assert "decompound_left.md" in hits_kk or "decompound_right.md" in hits_kk

        # CJK bigrams + 2-char EOF match via cjk_ngram_2
        hits_cjk_eof = [by_cid[cid] for cid, _ in lexical_search(idx, "碎叶", ro_conn, top_k=5)]
        assert "cjk.md" in hits_cjk_eof
        hits_cjk_mid = [by_cid[cid] for cid, _ in lexical_search(idx, "都市", ro_conn, top_k=5)]
        assert "cjk.md" in hits_cjk_mid


def test_expand_citation_neighbors_filters_glob_masked_and_solo(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    (cfg.corpus_root / "sub").mkdir(parents=True)
    (cfg.corpus_root / "root_doc.md").write_text(
        "# Root Anchor\nRoot Anchor references Sub Alpha and Sub Beta.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "sub" / "sub_a.md").write_text(
        "# Sub Alpha\nSub Alpha expands Root Anchor measurements.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "sub" / "sub_b.md").write_text(
        "# Sub Beta\nSub Beta records distinct instrumentation from Root Anchor.\n",
        encoding="utf-8",
    )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        cid_by_path = {ptr.rel_posix_path: ptr.chunk_id for ptr in idx.pointers}
        root_cid = cid_by_path["root_doc.md"]

        all_nb = expand_citation_neighbors(idx, root_cid, corpus_root=cfg.corpus_root, conn=ro_conn, top_k=5)
        assert len(all_nb) == 2
        assert all(cid != root_cid for cid, _ in all_nb)

        glob_nb = expand_citation_neighbors(
            idx, root_cid, corpus_root=cfg.corpus_root, conn=ro_conn, top_k=5, path_glob="sub/*.md"
        )
        assert {cid_by_path["sub/sub_a.md"], cid_by_path["sub/sub_b.md"]} == {cid for cid, _ in glob_nb}

        masked_nb = expand_citation_neighbors(
            idx,
            root_cid,
            corpus_root=cfg.corpus_root,
            conn=ro_conn,
            top_k=5,
            masked_rel_paths=["sub/sub_a.md"],
        )
        assert [cid for cid, _ in masked_nb] == [cid_by_path["sub/sub_b.md"]]

        with pytest.raises(YawRuntimeError) as exc_bad_k:
            expand_citation_neighbors(idx, root_cid, corpus_root=cfg.corpus_root, conn=ro_conn, top_k=0)
        assert exc_bad_k.value.code == "INVALID_INPUT"

    # Solo N=1 corpus -> expand_citation_neighbors returns []
    solo_cfg = _make_cfg(tmp_path / "solo")
    assert solo_cfg.corpus_root is not None and solo_cfg.db_path is not None
    solo_cfg.corpus_root.mkdir(parents=True)
    (solo_cfg.corpus_root / "solo.md").write_text("# Solo\nSingle passage corpus only.\n", encoding="utf-8")
    _index_corpus(solo_cfg)
    with contextlib.closing(open_readonly_db(solo_cfg.db_path)) as s_conn:
        s_idx = build_retrieval_index(s_conn)
        assert expand_citation_neighbors(s_idx, s_idx.chunk_ids[0], corpus_root=solo_cfg.corpus_root, conn=s_conn) == []


def test_n1_and_identical_vectors_produce_zero_nan_and_finite_search_result(tmp_path: Path) -> None:
    # N=1 corpus
    n1_cfg = _make_cfg(tmp_path / "n1")
    assert n1_cfg.corpus_root is not None and n1_cfg.db_path is not None
    n1_cfg.corpus_root.mkdir(parents=True)
    (n1_cfg.corpus_root / "one.md").write_text(
        "# Only Note\nSingle document geometric bound check.\n",
        encoding="utf-8",
    )
    _index_corpus(n1_cfg, embed_fn=_topic_embed)

    with contextlib.closing(open_readonly_db(n1_cfg.db_path)) as conn1:
        idx1 = build_retrieval_index(conn1)
        assert idx1.size == 1 and idx1.d_eff >= 2.0
        assert np.all(np.isfinite(idx1.mean_vec)) and np.all(np.isfinite(idx1.ip_norm))
        q_vec = _topic_embed(["single document"])[0]
        res1 = search_index(idx1, q_vec, query_text="single document", conn=conn1, top_k=5)
        assert res1.n_candidates == 1
        assert len(res1.ranked) == 1
        assert math.isfinite(res1.log_bf)
        assert math.isfinite(res1.max_chunk_score)
        assert math.isfinite(res1.top1_cosine)
        assert math.isfinite(res1.d_eff) and res1.d_eff >= 2.0
        assert all(math.isfinite(sc) for _, sc in res1.ranked)

    # Identical embeddings across N=6 chunks
    id_cfg = _make_cfg(tmp_path / "identical")
    assert id_cfg.corpus_root is not None and id_cfg.db_path is not None
    id_cfg.corpus_root.mkdir(parents=True)
    for i in range(6):
        (id_cfg.corpus_root / f"doc_{i}.md").write_text(
            f"# Doc {i}\nIdentical embedding vector entry number {i}.\n",
            encoding="utf-8",
        )
    _index_corpus(id_cfg, embed_fn=_identical_embed)

    with contextlib.closing(open_readonly_db(id_cfg.db_path)) as conn_id:
        idx_id = build_retrieval_index(conn_id)
        assert idx_id.size == 6 and idx_id.d_eff >= 2.0
        assert np.all(np.isfinite(idx_id.mean_vec))
        assert np.all(np.isfinite(idx_id.ip_norm)) and np.all(idx_id.ip_norm > 0.0)
        q_id = _identical_embed(["entry"])[0]
        res_id = search_index(idx_id, q_id, query_text="entry number", conn=conn_id, top_k=4)
        assert 1 <= res_id.n_candidates <= 4
        assert math.isfinite(res_id.log_bf)
        assert math.isfinite(res_id.max_chunk_score)
        assert math.isfinite(res_id.top1_cosine)
        assert all(math.isfinite(sc) for _, sc in res_id.ranked)


def test_uncropped_hydration_corrupt_utf8_and_unbounded_routes(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=2000)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    long_line = "Detailed orbital specification line with unique marker TailTokenZ99. " * 20
    (cfg.corpus_root / "long.md").write_text(f"# Long Passage\n{long_line}\n", encoding="utf-8")
    (cfg.corpus_root / "other.md").write_text("# Other\nCompact passage body.\n", encoding="utf-8")
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        long_ptr = next(p for p in idx.pointers if p.rel_posix_path == "long.md")

        uncropped = hydrate_passage(long_ptr, cfg.corpus_root, score=0.9)
        assert not uncropped.stale_on_disk
        assert len(uncropped.text) > 1000
        assert "TailTokenZ99" in uncropped.text
        assert "[... cropped ...]" not in uncropped.text

        cropped = hydrate_passage(long_ptr, cfg.corpus_root, score=0.9, max_chars=220)
        assert len(cropped.text) <= 220

        route = format_evidence_route([uncropped])
        assert "TailTokenZ99" in route
        assert f"chunk_id={long_ptr.chunk_id}" in route

        # Corrupt UTF-8 bytes on disk must fail loudly with CONSISTENCY_ERROR rather than masking replacement chars
        bad_file = cfg.corpus_root / "bad_utf8.md"
        bad_file.write_bytes(b"\xff\xfe\xfd invalid utf8")
        bad_ptr = ChunkPointer(
            chunk_id=999,
            rel_path="bad_utf8.md",
            source_hash="0" * 64,
            start_line=1,
            end_line=1,
            char_start=0,
            char_end=4,
            ast_island_start=1,
            ast_island_end=1,
            title="",
        )
        with pytest.raises(YawRuntimeError) as exc_utf8:
            hydrate_passage(bad_ptr, cfg.corpus_root, score=0.5)
        assert exc_utf8.value.code == "CONSISTENCY_ERROR"


def test_single_pass_title_hydration_and_lazy_diffusion_operators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=220, min_chunk_size=20)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    multi_body = (
        "\n\n# Alpha \u2014 Bridge\n"
        "Alpha Bridge references Target Doc for telemetry calibration.\n\n"
        + "Detailed manifold lattice section one with orbital booster notes.\n\n"
        + "Detailed manifold lattice section two referencing Target Doc again.\n"
    )
    (cfg.corpus_root / "multi.md").write_text(multi_body, encoding="utf-8")
    (cfg.corpus_root / "target.md").write_text(
        "# Target Doc\nTarget Doc records unique XylophoneZither harmonic proof.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "plain.txt").write_text(
        "Plain unheaded note mentioning unrelated basalt sample.\n",
        encoding="utf-8",
    )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.doc_titles == {"multi.md": "Alpha - Bridge", "target.md": "Target Doc"}
        assert idx.diffusion_ops is not None
        assert idx.diffusion_ops.m_fwd_t is None
        assert idx.diffusion_ops.r_rev is None

        # target.md has 0 outgoing citations and 1 incoming citation -> has_rev must populate inv_m > 0
        target_pos = next(i for i, p in enumerate(idx.pointers) if p.rel_posix_path == "target.md")
        assert float(idx.diffusion_ops.inv_m[target_pos]) > 0.0

        # Query for Alpha Bridge diffuses across tangent-steered citation edge to Target Doc
        q_vec = _topic_embed(["alpha bridge"])[0]
        res = search_index(idx, q_vec, query_text="Alpha Bridge", conn=ro_conn, top_k=5)
        res_paths = [idx.pointers[idx.id_to_pos[cid]].rel_posix_path for cid, _ in res.ranked]
        assert "multi.md" in res_paths and "target.md" in res_paths

        multi_cid = next(p.chunk_id for p in idx.pointers if p.rel_posix_path == "multi.md")
        nbs = expand_citation_neighbors(idx, multi_cid, corpus_root=cfg.corpus_root, conn=ro_conn, top_k=5)
        assert any(idx.pointers[idx.id_to_pos[cid]].rel_posix_path == "target.md" for cid, _ in nbs)

        _, _, _, _, rows = compute_db_fingerprints(ro_conn)

    # Confirm compute_index_artifacts opens/reads each distinct corpus file exactly once (single pass)
    corpus_resolved = cfg.corpus_root.resolve()
    opens_by_name: dict[str, int] = {}
    opens_lock = threading.Lock()
    orig_open = Path.open

    def counted_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.resolve().parent == corpus_resolved:
            with opens_lock:
                opens_by_name[self.name] = opens_by_name.get(self.name, 0) + 1
        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    artifacts = compute_index_artifacts(rows, cfg.corpus_root)
    assert opens_by_name == {"multi.md": 1, "plain.txt": 1, "target.md": 1}
    assert artifacts.n_chunks == len(rows)

    # corpus_root=None opens 0 files and preserves SQLite-provided titles cleanly
    opens_by_name.clear()
    sql_titled_rows = [(*r[:-1], f"SQL Title {i}") for i, r in enumerate(rows)]
    sql_only_artifacts = compute_index_artifacts(sql_titled_rows, None)
    assert opens_by_name == {}
    assert len(sql_only_artifacts.doc_titles_blob) > 0
    assert len(sql_only_artifacts.graph_blob) > 0

    # Empty doc_titles_blob fails cleanly with CONSISTENCY_ERROR
    with contextlib.closing(sqlite3.connect(cfg.db_path)) as rw_conn:
        rw_conn.execute("UPDATE citation_graph SET doc_titles_blob = X'' WHERE id = 1")
        rw_conn.commit()
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        with pytest.raises(YawRuntimeError) as exc_dt:
            build_retrieval_index(ro_conn)
        assert exc_dt.value.code == "CONSISTENCY_ERROR"


def _compute_exact_post(
    idx: retrieval_mod.RetrievalIndex,
    q_vec: NDArray[np.float32],
    query_text: str,
    conn: sqlite3.Connection,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    eligible = retrieval_lexical.eligible_mask(idx)
    n_eff = int(np.sum(eligible))
    t_path = math.sqrt(max(1.0, idx.d_eff) / max(1.0, math.log(float(n_eff) + 1.0)))
    assert idx.diffusion_ops is not None
    post, _, _, _ = retrieval_search._score_query_evidence(
        idx, q_vec, query_text, conn, eligible, [], n_eff, t_path, idx.diffusion_ops
    )
    return post, eligible


def _verify_duplicate_span_coverage(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "dup_cov", chunk_size=140, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "doc_a.md").write_text(
        "# Alpha Hub\nOrbital booster primary valve calibration protocol.\n\n"
        + ("Geology basalt limestone secondary outcrop details. " * 3)
        + "\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "doc_b.md").write_text(
        "# Beta Node\nOrbital booster emergency stage check.\n",
        encoding="utf-8",
    )
    _index_corpus(cfg)

    with contextlib.closing(sqlite3.connect(cfg.db_path)) as rw_conn:
        row_top = rw_conn.execute(
            "SELECT chunk_id, embedding_f16, rel_path, source_hash, start_line, end_line, "
            "char_start, char_end, ast_island_start, ast_island_end, title "
            "FROM chunks WHERE rel_path = 'doc_a.md' ORDER BY start_line LIMIT 1"
        ).fetchone()
        assert row_top is not None
        dup_cid = int(rw_conn.execute("SELECT MAX(chunk_id) FROM chunks").fetchone()[0]) + 50
        rw_conn.execute(
            "INSERT INTO chunks (chunk_id, embedding_f16, rel_path, source_hash, start_line, end_line, "
            "char_start, char_end, ast_island_start, ast_island_end, title) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (dup_cid, *row_top[1:]),
        )
        rw_conn.execute(
            "INSERT INTO chunk_fts(rowid, title, body) VALUES (?, ?, ?)",
            (dup_cid, str(row_top[10]), "Orbital booster primary valve calibration protocol."),
        )
        rw_conn.commit()
    with contextlib.closing(open_managed_db(cfg.db_path)) as conn_mgr:
        reconcile_index_tables(
            conn_mgr,
            embedding_model=cfg.embedding_model,
            task_mode=cfg.embedding_task_mode,
            corpus_root=cfg.corpus_root,
        )

    q = "orbital booster primary valve"
    q_vec = _topic_embed([q])[0]
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.size >= 4
        res = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=3, pool_k=50)
        raw_top3_cids = [cid for cid, _ in res.ranked[:3]]
        dedup3 = dedup_ranked_positions(idx, res.ranked, limit=3)
        dedup_top3_cids = [cid for _, cid, _ in dedup3]
        assert len(dedup_top3_cids) == 3
        assert len(raw_top3_cids) == 3 and dup_cid in raw_top3_cids
        assert dedup_top3_cids != raw_top3_cids
        post, elig = _compute_exact_post(idx, q_vec, q, ro_conn)
        expected_cov = float(post[[p for p, _, _ in dedup3]].sum() / max(float(post[elig].sum()), 1e-12))
        naive_cov = float(post[[idx.id_to_pos[c] for c in raw_top3_cids]].sum() / max(float(post[elig].sum()), 1e-12))
        assert res.coverage == pytest.approx(expected_cov, abs=1e-12)
        assert res.coverage != pytest.approx(naive_cov, abs=1e-6)
        res_t1 = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=1, pool_k=50)
        assert dup_cid not in {c for c, _ in res_t1.next_candidates}
        assert len({idx.pointers[idx.id_to_pos[c]].span_label for c, _ in res_t1.next_candidates}) == len(
            res_t1.next_candidates
        )


def test_coverage_pool_k_parity_and_duplicate_span_top_k(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=300, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    for i in range(56):
        kw = "orbital booster valve protocol" if i < 18 else f"geology basalt survey note {i}"
        (cfg.corpus_root / f"doc_{i:02d}.md").write_text(
            f"# Doc {i:02d}\nEntry {i:02d}: {kw} calibration record {i}.\n",
            encoding="utf-8",
        )
    _index_corpus(cfg)

    q = "orbital booster valve"
    q_vec = _topic_embed([q])[0]
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.size == 56
        res_pool50 = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=50)
        res_u5 = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=5, pool_k=50)
        assert res_u5.ranked == res_pool50.ranked
        assert res_u5.n_candidates == res_pool50.n_candidates

        post, eligible = _compute_exact_post(idx, q_vec, q, ro_conn)
        denom = max(float(post[eligible].sum()), 1e-12)
        covs: list[float] = []
        for k in (2, 5, 10):
            rk = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=k, pool_k=50)
            top_pos = [p for p, _, _ in dedup_ranked_positions(idx, rk.ranked, limit=k)]
            expected_cov = float(post[top_pos].sum() / denom)
            assert 0.0 < rk.coverage <= 1.0
            assert rk.coverage == pytest.approx(expected_cov, abs=1e-12)
            assert len(rk.next_candidates) == 3
            covs.append(rk.coverage)
        assert covs[0] < covs[1] < covs[2]

    _verify_duplicate_span_coverage(tmp_path)


def test_dedup_next_labels_zero_tail_opens_and_cache_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=260, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    for i in range(12):
        if i == 3:
            body = 'Plain note "quoted"\twith   orbital booster notes\n'
        else:
            body = f'# Orbital "Title" {i:02d} {"X" * 70}\nOrbital booster calibration item {i:02d}.\n'
        (cfg.corpus_root / f"item_{i:02d}.md").write_text(body, encoding="utf-8")
    _index_corpus(cfg)

    q = "orbital booster calibration"
    q_vec = _topic_embed([q])[0]
    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        res = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=2, pool_k=50)
        assert len(res.next_candidates) == 3

        # Delete ALL 3 tail files on disk before hydration + payload formatting
        tail_paths = {idx.pointers[idx.id_to_pos[cid]].rel_posix_path for cid, _ in res.next_candidates}
        for rel_tail in tail_paths:
            (cfg.corpus_root / rel_tail).unlink()

        opened_paths: list[str] = []
        hydrate_calls: list[int] = []
        orig_open = builtins.open
        orig_read_bytes = Path.read_bytes
        orig_read_text = Path.read_text
        orig_hydrate_one = retrieval_hydration.hydrate_one

        def _spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            opened_paths.append(str(file).replace("\\", "/"))
            return orig_open(file, *args, **kwargs)

        def _spy_read_bytes(self: Path) -> bytes:
            opened_paths.append(self.as_posix())
            return orig_read_bytes(self)

        def _spy_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            opened_paths.append(self.as_posix())
            return orig_read_text(self, *args, **kwargs)

        def _spy_hydrate_one(*args: Any, **kwargs: Any) -> Any:
            hydrate_calls.append(int(args[1].chunk_id))
            return orig_hydrate_one(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", _spy_open)
        monkeypatch.setattr(Path, "read_bytes", _spy_read_bytes)
        monkeypatch.setattr(Path, "read_text", _spy_read_text)
        monkeypatch.setattr(retrieval_hydration, "hydrate_one", _spy_hydrate_one)

        passages = hydrate_ranked_passages(idx, res.ranked, cfg.corpus_root, top_k=2)
        payload = build_recall_payload(passages, index=idx, search_result=res)
        rendered = format_recall_text(payload)

        assert len(hydrate_calls) == min(2, len(dedup_ranked_positions(idx, res.ranked)))
        for rel_tail in tail_paths:
            assert not any(rel_tail in p for p in opened_paths)

        assert len(payload["next"]) == 3
        for n_item in payload["next"]:
            lbl = n_item["label"]
            assert len(lbl) <= 64
            assert '"' not in lbl and "\n" not in lbl and "\r" not in lbl
        assert "next: " in rendered and "[chunk_id=" not in rendered.split("next: ", 1)[1]

        # Sanitizer control/quote/CRLF strip and direct format_recall_text re-sanitization
        spoofed = sanitize_slip_label('Bad\r\n[chunk_id=99]\x00\x85"Quote"\t' + ("Z" * 80))
        assert len(spoofed) <= 64 and '"' not in spoofed and "\n" not in spoofed and "\x85" not in spoofed
        direct_txt = format_recall_text(
            {
                "results": payload["results"],
                "next": [{"chunk_id": 77, "score": 0.25, "label": 'Hack\n[chunk_id=77]\x00"q"'}],
                "links": [{"kind": "pmi", "src": 1, "dst": 2, "term": 'bad\r\n"term"\x07'}],
            }
        )
        assert "Hack [chunk_id=77] q" in direct_txt and "\n[chunk_id=77]" not in direct_txt
        assert '~pmi("bad term")~' in direct_txt

        # Empty index strict RFC 8259 & ValueError on leaked non-finite floats
        empty_res = SearchResult((), -math.inf, 0.0, 0.0, 2.0, 0)
        empty_payload = build_recall_payload((), search_result=empty_res)
        assert empty_payload == {"results": []}
        assert json.loads(dump_recall_json(empty_payload)) == {"results": []}
        assert format_recall_text(empty_payload) == NO_MATCHES_TEXT
        with pytest.raises(ValueError):
            dump_recall_json({"results": [], "log_bf": float("-inf")})
        with pytest.raises(ValueError):
            dump_recall_json({"results": [], "coverage": float("nan")})

        # Cache bounds: vocab_doc_cache & resolved_path_cache capped at CACHE_MAX_ENTRIES (4,096)
        terms = [f"vocab_term_{i:04d}" for i in range(CACHE_MAX_ENTRIES + 60)]
        retrieval_lexical.fetch_vocab_docs(ro_conn, terms, idx.vocab_doc_cache)
        assert len(idx.vocab_doc_cache) == CACHE_MAX_ENTRIES
        assert "vocab_term_0000" not in idx.vocab_doc_cache
        assert f"vocab_term_{CACHE_MAX_ENTRIES + 59:04d}" in idx.vocab_doc_cache

        for i in range(CACHE_MAX_ENTRIES + 60):
            retrieval_hydration._resolve_cached_path(cfg.corpus_root, f"cache_{i:04d}.md", idx.resolved_path_cache)
        assert len(idx.resolved_path_cache) == CACHE_MAX_ENTRIES
        assert (cfg.corpus_root, "cache_0000.md") not in idx.resolved_path_cache
        assert (cfg.corpus_root, f"cache_{CACHE_MAX_ENTRIES + 59:04d}.md") in idx.resolved_path_cache


def test_search_result_links_multi_chunk_cites_and_rarest_global_pmi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=180, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    (cfg.corpus_root / "source.md").write_text(
        "# Source Hub\nSource Hub cites Beta Manifold and uses orbital quantumflux_alpha with Gamma Node.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "beta.md").write_text(
        "# Beta Manifold\n"
        "Beta Manifold primary orbital booster section with internal shared_intra_phrase note.\n\n"
        + ("Filler secondary chunk mentioning shared_intra_phrase and orbital baseline details. " * 2)
        + "\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "gamma.md").write_text(
        "# Gamma Node\nGamma Node pairs orbital quantumflux_alpha alongside common_shared_pair calibration.\n",
        encoding="utf-8",
    )
    for k in range(4):
        (cfg.corpus_root / f"common_{k}.md").write_text(
            f"# Common {k}\nCommon {k} uses common_shared_pair and orbital valve baseline {k}.\n",
            encoding="utf-8",
        )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.citation_graph is not None
        beta_chunks = [p.chunk_id for p in idx.pointers if p.rel_posix_path == "beta.md"]
        assert len(beta_chunks) >= 2
        src_cid = next(p.chunk_id for p in idx.pointers if p.rel_posix_path == "source.md")

        io_calls: list[str] = []
        orig_open = builtins.open
        orig_rb = Path.read_bytes
        orig_rt = Path.read_text
        monkeypatch.setattr(builtins, "open", lambda f, *a, **kw: (io_calls.append(str(f)), orig_open(f, *a, **kw))[1])
        monkeypatch.setattr(Path, "read_bytes", lambda self: (io_calls.append(self.as_posix()), orig_rb(self))[1])
        monkeypatch.setattr(
            Path, "read_text", lambda self, *a, **kw: (io_calls.append(self.as_posix()), orig_rt(self, *a, **kw))[1]
        )

        q = "orbital booster Beta Manifold Source Hub Gamma Node"
        q_vec = _topic_embed([q])[0]
        res = search_index(idx, q_vec, query_text=q, conn=ro_conn, top_k=4, pool_k=50)
        assert io_calls == []

        deduped = dedup_ranked_positions(idx, res.ranked, limit=7)
        top_pos = [p for p, _, _ in deduped[:4]]
        next_pos = [p for p, _, _ in deduped[4:7]]
        cand_cids = [idx.chunk_ids[p] for p in [*top_pos, *next_pos]]
        best_beta_cid = next(cid for cid in cand_cids if cid in beta_chunks)
        assert ("cite", src_cid, best_beta_cid, "") in res.links
        for other_b in beta_chunks:
            if other_b != best_beta_cid:
                assert ("cite", src_cid, other_b, "") not in res.links

        # Verify <= 1 global rarest cross-doc PMI edge & latency budget on extracted link kernel
        pmi_links = [lnk for lnk in res.links if lnk[0] == "pmi"]
        assert len(pmi_links) <= 1
        assert all(idx.path_idx[idx.id_to_pos[lnk[1]]] != idx.path_idx[idx.id_to_pos[lnk[2]]] for lnk in res.links)

        # Confirm global rarest cross-doc PMI selection deterministically with two synthetic cross-doc terms
        cg = idx.citation_graph
        p0, p1, p2 = top_pos[0], top_pos[1], top_pos[2]
        assert len({int(idx.path_idx[p0]), int(idx.path_idx[p1]), int(idx.path_idx[p2])}) == 3
        cpmi_lists: list[list[int]] = [[] for _ in range(idx.size)]
        cpmi_lists[p0] = [0, 1]
        cpmi_lists[p1] = [0]
        cpmi_lists[p2] = [1]
        for extra_p in range(idx.size):
            if extra_p not in (p0, p1, p2):
                cpmi_lists[extra_p] = [0]
        pmi_indptr = np.array([0, idx.size - 2, idx.size], dtype=np.int32)
        pmi_indices = np.zeros(idx.size, dtype=np.int32)
        cpmi_indptr, cpmi_indices = lists_to_csr(cpmi_lists, idx.size)
        cg_pmi = retrieval_mod.CitationGraph(
            fi=cg.fi,
            fx=cg.fx,
            ri=cg.ri,
            rx=cg.rx,
            dc_indptr=cg.dc_indptr,
            dc_indices=cg.dc_indices,
            pmi_indptr=pmi_indptr,
            pmi_indices=pmi_indices,
            cpmi_indptr=cpmi_indptr,
            cpmi_indices=cpmi_indices,
            pmi_terms=("common_phrase", 'rare\n"phrase"\x07'),
            prop_nouns=cg.prop_nouns,
            doc_titles=cg.doc_titles,
        )
        idx_pmi = retrieval_mod.RetrievalIndex(
            chunk_ids=idx.chunk_ids,
            id_to_pos=idx.id_to_pos,
            pointers=idx.pointers,
            path_idx=idx.path_idx,
            distinct_paths=idx.distinct_paths,
            dim=idx.dim,
            mean_vec=idx.mean_vec,
            d_eff=idx.d_eff,
            codes=idx.codes,
            ip_norm=idx.ip_norm,
            rotator=idx.rotator,
            doc_titles=idx.doc_titles,
            title_by_first_word=idx.title_by_first_word,
            citation_graph=cg_pmi,
            diffusion_ops=idx.diffusion_ops,
        )
        links_pmi = retrieval_search._extract_candidate_links(idx_pmi, top_pos, next_pos)
        pmi_only = [lnk for lnk in links_pmi if lnk[0] == "pmi"]
        assert pmi_only == [("pmi", idx.chunk_ids[p0], idx.chunk_ids[p2], "rare phrase")]

        t0 = time.perf_counter()
        for _ in range(500):
            retrieval_search._extract_candidate_links(idx_pmi, top_pos, next_pos)
        per_call_ms = ((time.perf_counter() - t0) * 1000.0) / 500.0
        assert per_call_ms < 0.05

        direct_payload = build_recall_payload((), links=links_pmi) | {
            "results": [
                {
                    "rank": 1,
                    "chunk_id": idx.chunk_ids[p0],
                    "score": 1.0,
                    "text": "x",
                    "content_hash": "sha256:0",
                    "source": {"rel_posix_path": "a.md", "start_line": 1, "end_line": 1},
                }
            ]
        }
        rendered_links = format_recall_text(direct_payload)
        assert f'{idx.chunk_ids[p0]}~pmi("rare phrase")~{idx.chunk_ids[p2]}' in rendered_links
