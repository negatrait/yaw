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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cocoindex.connectorkits.fingerprint import fingerprint_str
from numpy.typing import NDArray

import yaw.retrieval as retrieval_mod
import yaw.retrieval.hydration as retrieval_hydration
import yaw.retrieval.lexical as retrieval_lexical
import yaw.retrieval.search as retrieval_search
from yaw.core import ChunkPointer, YawConfig, YawRuntimeError, split_lf_lines
from yaw.pipeline import (
    chunk_file_content,
    close_pipeline,
    open_managed_db,
    reconcile_index_tables,
    start_pipeline,
    update_pipeline_once,
)
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
from yaw.retrieval.graph import build_citation_graph, lists_to_csr
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
    imported_tops: set[str] = set()
    imported_full: set[str] = set()
    for py_file in sorted(Path(retrieval_mod.__file__).parent.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_tops.add(alias.name.split(".")[0])
                    imported_full.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_tops.add(node.module.split(".")[0])
                imported_full.add(node.module)
    assert {"sqlite_vec", "spacy", "optuna", "dspy", "fastmcp", "sentence_transformers"}.isdisjoint(imported_tops)
    assert all(
        mod == "cocoindex.connectorkits.fingerprint"
        for mod in imported_full
        if mod == "cocoindex" or mod.startswith("cocoindex.")
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

    # Missing or non-directory corpus_root fails fast with CONSISTENCY_ERROR across artifact & reconciliation builders
    missing_root = tmp_path / "missing_corpus_root"
    missing_any: Any = None
    for invalid_root in (missing_root, missing_any):
        with pytest.raises(YawRuntimeError) as exc_art:
            compute_index_artifacts(rows, invalid_root)
        assert exc_art.value.code == "CONSISTENCY_ERROR"
    with contextlib.closing(open_managed_db(cfg.db_path)) as bad_mgr:
        with pytest.raises(YawRuntimeError) as exc_rec:
            reconcile_index_tables(
                bad_mgr,
                embedding_model=cfg.embedding_model,
                task_mode=cfg.embedding_task_mode,
                corpus_root=missing_root,
            )
        assert exc_rec.value.code == "CONSISTENCY_ERROR"

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
        "# Alpha Hub\n\n"
        "## Section One\nOrbital booster primary valve calibration protocol.\n\n"
        "## Section Two\n" + ("Geology basalt limestone secondary outcrop details. " * 3) + "\n",
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
        dedup3 = dedup_ranked_positions(idx, res.ranked, limit=3, include_island=True)
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
            top_pos = [p for p, _, _ in dedup_ranked_positions(idx, rk.ranked, limit=k, include_island=True)]
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


def test_resolve_hydrated_line_span_contract() -> None:
    from yaw.retrieval.hydration import resolve_hydrated_line_span

    ptr = ChunkPointer(
        chunk_id=10,
        rel_path="doc.md",
        source_hash="0" * 64,
        start_line=5,
        end_line=7,
        ast_island_start=3,
        ast_island_end=12,
        title="Sec",
    )
    assert resolve_hydrated_line_span(ptr, include_island=False) == (5, 7)
    assert resolve_hydrated_line_span(ptr, include_island=False, total_lines=20) == (5, 7)
    assert resolve_hydrated_line_span(ptr, include_island=True) == (3, 12)
    assert resolve_hydrated_line_span(ptr, include_island=True, total_lines=20) == (3, 12)
    assert resolve_hydrated_line_span(ptr, include_island=True, total_lines=10) == (3, 10)

    # (1, 1) sentinel on multi-line file must NEVER widen (5, 7) to (1, 7)
    sentinel_ptr = ChunkPointer(
        chunk_id=11,
        rel_path="legacy.md",
        source_hash="0" * 64,
        start_line=5,
        end_line=7,
        ast_island_start=1,
        ast_island_end=1,
        title="",
    )
    assert resolve_hydrated_line_span(sentinel_ptr, include_island=True, total_lines=20) == (5, 7)

    # Cap below _MAX_HYDRATE_SPAN (10_000) while preserving [start_line, end_line]
    huge_ptr = ChunkPointer(
        chunk_id=12,
        rel_path="huge.md",
        source_hash="0" * 64,
        start_line=5,
        end_line=7,
        ast_island_start=1,
        ast_island_end=25_000,
        title="Huge",
    )
    es, ee = resolve_hydrated_line_span(huge_ptr, include_island=True, total_lines=30_000)
    assert es <= 5 <= 7 <= ee
    assert ee - es + 1 <= 10_000


def test_hydrate_passage_exact_and_ranked_maxp_island_expansion_and_drift(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=140, chunk_overlap=0, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    alpha_doc = cfg.corpus_root / "guide.md"
    alpha_doc.write_text(
        "# Doc Root\n\n"
        "## Alpha Section\n\n"
        "Alpha line 1 explains orbital booster valve telemetry clearly.\n\n"
        "Alpha line 2 expands secondary manifold pressure calibration data.\n\n"
        "Alpha line 3 closes alpha section diagnostics cleanly.\n\n"
        "## Beta Section\n\n"
        "Beta independent basalt survey summary.\n",
        encoding="utf-8",
    )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        alpha_ptrs = sorted(
            (p for p in idx.pointers if p.title == "Alpha Section"),
            key=lambda p: p.start_line,
        )
        beta_ptr = next(p for p in idx.pointers if p.title == "Beta Section")
        assert len(alpha_ptrs) >= 2
        p1, p2 = alpha_ptrs[0], alpha_ptrs[1]
        assert (p1.ast_island_start, p1.ast_island_end) == (p2.ast_island_start, p2.ast_island_end)
        es, ee = p1.ast_island_start, p1.ast_island_end
        assert es <= p1.start_line and ee > p1.end_line

        # hydrate_passage defaults to include_island=False -> exact [sl, el]
        exact = hydrate_passage(p2, cfg.corpus_root, score=0.8)
        assert not exact.stale_on_disk
        assert exact.resolved_start_line == p2.start_line
        assert exact.resolved_end_line == p2.end_line
        assert exact.hydrated_start_line == p2.start_line
        assert exact.hydrated_end_line == p2.end_line

        exp_from_p2 = hydrate_passage(p2, cfg.corpus_root, score=0.8, include_island=True)
        assert exp_from_p2.resolved_start_line == es and exp_from_p2.resolved_end_line == ee
        assert exp_from_p2.text.startswith("## Alpha Section\n")
        assert not exp_from_p2.text.startswith("Alpha Section\n## Alpha Section")

        # Non-hydrating dedup coalesces same island iff include_island=True
        ranked = [(p1.chunk_id, 0.9), (p2.chunk_id, 0.8), (beta_ptr.chunk_id, 0.7)]
        assert len(dedup_ranked_positions(idx, ranked, include_island=False)) == 3
        dedup_island = dedup_ranked_positions(idx, ranked, include_island=True)
        assert [cid for _, cid, _ in dedup_island] == [p1.chunk_id, beta_ptr.chunk_id]

        # hydrate_ranked_passages defaults to include_island=True -> MaxP island coalescing + (es-ee)
        passages = hydrate_ranked_passages(idx, ranked, cfg.corpus_root, top_k=5)
        assert len(passages) == 2
        hp_alpha, hp_beta = passages
        assert hp_alpha.pointer.chunk_id == p1.chunk_id
        assert hp_alpha.resolved_start_line == es
        assert hp_alpha.resolved_end_line == ee
        assert hp_alpha.text.startswith("## Alpha Section\n")
        assert not hp_alpha.text.startswith("Alpha Section\n## Alpha Section")
        assert "Alpha line 1" in hp_alpha.text and "Alpha line 3" in hp_alpha.text
        assert "Beta independent" not in hp_alpha.text

        payload = build_recall_payload(passages)
        src0 = payload["results"][0]["source"]
        assert src0["start_line"] == es and src0["end_line"] == ee and src0["title"] == "Alpha Section"
        txt = format_recall_text(payload)
        assert f"(guide.md:{es}-{ee})" in txt
        route = format_evidence_route(passages)
        assert f"guide.md:{es}-{ee}" in route
        assert f"guide.md:{p1.start_line}-{p1.end_line}" not in route

        # source_hash verification uses fingerprint_str (NOT sha256) on exact [sl, el] slice
        lines = alpha_doc.read_text(encoding="utf-8").splitlines()
        raw_p2 = "\n".join(lines[p2.start_line - 1 : p2.end_line])
        assert p2.source_hash == fingerprint_str(raw_p2).hex()
        sha_ptr = ChunkPointer(
            chunk_id=p2.chunk_id,
            rel_path=p2.rel_path,
            source_hash=hashlib.sha256(raw_p2.encode("utf-8")).hexdigest(),
            start_line=p2.start_line,
            end_line=p2.end_line,
            char_start=p2.char_start,
            char_end=p2.char_end,
            ast_island_start=p2.ast_island_start,
            ast_island_end=p2.ast_island_end,
            title=p2.title,
        )
        assert hydrate_passage(sha_ptr, cfg.corpus_root).stale_on_disk is True

        # Mutate a line inside p1's [sl, el] -> p1 flags stale while beta remains clean; strict=True raises
        mutated = list(lines)
        mutated[p1.start_line - 1] = "## Alpha Section MUTATED"
        alpha_doc.write_text("\n".join(mutated) + "\n", encoding="utf-8")
        stale_passages = hydrate_ranked_passages(idx, ranked, cfg.corpus_root, top_k=5)
        assert stale_passages[0].stale_on_disk is True and stale_passages[0].text == ""
        assert (
            stale_passages[1].stale_on_disk is False and hp_beta.pointer.chunk_id == stale_passages[1].pointer.chunk_id
        )
        with pytest.raises(YawRuntimeError) as exc_drift:
            retrieval_hydration.hydrate_one(cfg.corpus_root, p1, 0.9, None, True, False, strict=True)
        assert exc_drift.value.code == "CONSISTENCY_ERROR"


def test_fts5_title_only_match_for_continuation_chunks(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=130, chunk_overlap=0, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "title_match.md").write_text(
        "# Quantumflux TitleOnlyHeader\n\n"
        "Paragraph one describes standard turbine mechanics cleanly.\n\n"
        "Paragraph two describes UniqueBodySecond alloy tolerances.\n",
        encoding="utf-8",
    )
    for k in range(3):
        (cfg.corpus_root / f"decoy_{k}.md").write_text(
            f"# Decoy {k}\nUnrelated basalt calibration record {k}.\n",
            encoding="utf-8",
        )
    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        tm_ptrs = sorted((p for p in idx.pointers if p.rel_posix_path == "title_match.md"), key=lambda p: p.start_line)
        assert len(tm_ptrs) >= 2
        cont_cid = tm_ptrs[-1].chunk_id

        title_hits = {
            int(r[0])
            for r in ro_conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'title : titleonlyheader'")
        }
        body_hits = {
            int(r[0])
            for r in ro_conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'body : titleonlyheader'")
        }
        assert cont_cid in title_hits
        assert cont_cid not in body_hits

        lex_cids = [cid for cid, _ in lexical_search(idx, "TitleOnlyHeader", ro_conn, top_k=5)]
        assert cont_cid in lex_cids


def test_opening_header_hydration_doc_titles_and_lf_line_alignment(tmp_path: Path) -> None:
    doc_md = "# Main Doc\n\n## Section One\nFirst body.\n\n## Section Two\nSecond body.\n"
    (tmp_path / "main.md").write_text(doc_md, encoding="utf-8")
    md_pieces = chunk_file_content("main.md", doc_md, chunk_size=400, chunk_overlap=0, min_chunk_size=10)
    assert len(md_pieces) == 1 and md_pieces[0].title == "Main Doc"
    md_ptr = ChunkPointer(
        chunk_id=md_pieces[0].chunk_id,
        rel_path="main.md",
        source_hash=md_pieces[0].source_hash,
        start_line=md_pieces[0].start_line,
        end_line=md_pieces[0].end_line,
        char_start=md_pieces[0].char_start,
        char_end=md_pieces[0].char_end,
        ast_island_start=md_pieces[0].ast_island_start,
        ast_island_end=md_pieces[0].ast_island_end,
        title=md_pieces[0].title,
    )
    cg = build_citation_graph([md_ptr], ["main.md"], np.array([0], dtype=np.uint32), tmp_path)
    assert cg.doc_titles == ("Main Doc",)

    py_text = (
        "@dataclass(frozen=True, slots=True)\n"
        "class ChunkPointer:\n"
        "    chunk_id: int\n\n"
        "    @property\n"
        "    def span_label(self) -> str:\n"
        '        return f"{self.chunk_id}"\n'
    )
    (tmp_path / "ptr.py").write_text(py_text, encoding="utf-8")
    py_pieces = chunk_file_content("ptr.py", py_text, chunk_size=400, chunk_overlap=0, min_chunk_size=10)
    assert len(py_pieces) == 1 and py_pieces[0].title == "class ChunkPointer"
    py_ptr = ChunkPointer(
        chunk_id=py_pieces[0].chunk_id,
        rel_path="ptr.py",
        source_hash=py_pieces[0].source_hash,
        start_line=py_pieces[0].start_line,
        end_line=py_pieces[0].end_line,
        char_start=py_pieces[0].char_start,
        char_end=py_pieces[0].char_end,
        ast_island_start=py_pieces[0].ast_island_start,
        ast_island_end=py_pieces[0].ast_island_end,
        title=py_pieces[0].title,
    )
    hp_py = hydrate_passage(py_ptr, tmp_path)
    assert not hp_py.stale_on_disk
    assert hp_py.text == py_pieces[0].body
    assert not hp_py.text.startswith("class ChunkPointer\n@dataclass")

    ff_text = "Alpha\x0cFormFeed\u2028LineSep.\n\n## Heading Below\n\nBody.\n"
    (tmp_path / "ff.md").write_text(ff_text, encoding="utf-8")
    ff_pieces = chunk_file_content("ff.md", ff_text, chunk_size=40, chunk_overlap=0, min_chunk_size=10)
    assert len(ff_pieces) == 2
    assert ff_pieces[1].start_line == 3 and ff_pieces[1].end_line == 5 and ff_pieces[1].title == "Heading Below"
    lf_lines = split_lf_lines(ff_text)
    for piece in ff_pieces:
        assert piece.source_hash == fingerprint_str("\n".join(lf_lines[piece.start_line - 1 : piece.end_line])).hex()
        hp_ff = hydrate_passage(
            ChunkPointer(
                chunk_id=piece.chunk_id,
                rel_path="ff.md",
                source_hash=piece.source_hash,
                start_line=piece.start_line,
                end_line=piece.end_line,
                char_start=piece.char_start,
                char_end=piece.char_end,
                ast_island_start=piece.ast_island_start,
                ast_island_end=piece.ast_island_end,
                title=piece.title,
            ),
            tmp_path,
        )
        assert not hp_ff.stale_on_disk and hp_ff.text == piece.body


def test_citation_graph_h1_isolation_no_synthetic_title_dup_and_single_line_dedup(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=110, chunk_overlap=0, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    canonical_md = (
        "# Canonical Document Title\n\n"
        "Primary overview paragraph establishing canonical system baseline.\n\n"
        "## Appendix Zebra\n\n"
        "Appendix Zebra supplementary calibration table 1 details.\n\n"
        "## Notes Omega\n\n"
        "Notes Omega detailed operating notes record 1 summary.\n"
    )
    (cfg.corpus_root / "canonical.md").write_text(canonical_md, encoding="utf-8")
    (cfg.corpus_root / "alpha.md").write_text(
        "# Alpha Manifold\n\n"
        "## Methods\n\nAlpha methods orbital calibration telemetry.\n\n"
        "## Results\n\nAlpha results orbital convergence metric.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "beta.md").write_text(
        "# Beta System\n\nBeta system discusses results and omega behavior independently.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "worker.py").write_text(
        "def run(items):\n    for item in items:\n        yield item + 1\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "one.txt").write_text("Singleline token orbital calibration " * 12, encoding="utf-8")
    fenced_md = (
        "# Enclosing Section\n\n"
        "Introductory context paragraph.\n\n"
        "```python\n"
        "first_var = 'padding string inside fenced block to trigger split before comment'\n\n"
        "# fake comment\n"
        "def inner():\n"
        "    return 'payload'\n"
        "```\n"
    )
    (cfg.corpus_root / "fenced.md").write_text(fenced_md, encoding="utf-8")

    # Confirm hash ordering puts a later subsection before chunk 1 in canonical.md
    can_chunks = chunk_file_content("canonical.md", canonical_md, chunk_size=110, chunk_overlap=0, min_chunk_size=10)
    assert len(can_chunks) >= 3 and can_chunks[-1].chunk_id < can_chunks[0].chunk_id

    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.citation_graph is not None
        assert idx.doc_titles == {
            "alpha.md": "Alpha Manifold",
            "beta.md": "Beta System",
            "canonical.md": "Canonical Document Title",
            "fenced.md": "Enclosing Section",
        }
        assert "worker.py" not in idx.doc_titles
        for bad_first in ("methods", "results", "appendix", "notes", "def"):
            assert bad_first not in idx.title_by_first_word

        # Canonical document title resolves from earliest line coordinate even when hash-sorted chunks invert order
        can_di = idx.distinct_paths.index("canonical.md")
        assert idx.citation_graph.doc_titles[can_di] == "Canonical Document Title"
        with pytest.raises(YawRuntimeError) as exc_bad_cg:
            build_citation_graph(idx.pointers, idx.distinct_paths, idx.path_idx, tmp_path / "missing_cg_root")
        assert exc_bad_cg.value.code == "CONSISTENCY_ERROR"

        # Beta mentioning "results" and "omega" must create zero outgoing citation edges
        beta_positions = [i for i, p in enumerate(idx.pointers) if p.rel_posix_path == "beta.md"]
        assert beta_positions
        for bp in beta_positions:
            assert idx.citation_graph.fi[bp + 1] == idx.citation_graph.fi[bp]

        # hydrate_corpus_chunks returns strictly raw disk slices with zero synthetic ptr.title prepending
        raw_texts, disk_titles = retrieval_hydration.hydrate_corpus_chunks(
            idx.pointers, cfg.corpus_root, extract_titles_for=idx.distinct_paths
        )
        assert disk_titles == idx.doc_titles
        can_positions = sorted(
            (i for i, p in enumerate(idx.pointers) if p.rel_posix_path == "canonical.md"),
            key=lambda i: idx.pointers[i].start_line,
        )
        assert not raw_texts[can_positions[-1]].startswith("Notes Omega\n")
        assert raw_texts[can_positions[-1]].startswith("## Notes Omega")

        # Single-line multi-chunk dedup retains all sub-line spans when include_island=True
        one_ptrs = [p for p in idx.pointers if p.rel_posix_path == "one.txt"]
        assert len(one_ptrs) >= 3
        assert all(p.start_line == 1 and p.end_line == 1 for p in one_ptrs)
        one_ranked = [(p.chunk_id, 1.0 - 0.05 * k) for k, p in enumerate(one_ptrs)]
        assert len(dedup_ranked_positions(idx, one_ranked, include_island=True)) == len(one_ptrs)

        # Fenced # fake comment continuation chunk still prepends enclosing title on exact read,
        # while hydrate_corpus_chunks keeps strictly raw disk text without synthetic title duplication.
        fence_pos = next(
            i
            for i, p in enumerate(idx.pointers)
            if p.rel_posix_path == "fenced.md"
            and hydrate_passage(p, cfg.corpus_root).text.lstrip().startswith("Enclosing Section\n# fake comment")
        )
        fence_ptr = idx.pointers[fence_pos]
        hp_fence = hydrate_passage(fence_ptr, cfg.corpus_root, include_island=False)
        assert not hp_fence.stale_on_disk
        assert hp_fence.text.startswith("Enclosing Section\n# fake comment")
        assert raw_texts[fence_pos].startswith("# fake comment")


def test_frontmatter_citation_graph_and_symmetric_island_dedup(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, chunk_size=500, chunk_overlap=0, min_chunk_size=10)
    assert cfg.corpus_root is not None and cfg.db_path is not None
    cfg.corpus_root.mkdir(parents=True)

    (cfg.corpus_root / "alpha.md").write_text(
        "# Alpha Manifold\n\nAlpha Manifold baseline documentation.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "card.md").write_text(
        "---\nsources: [alpha.md]\ntags: [manifold]\n---\n\n"
        "# Card Heading\n\nCard summary referencing Alpha Manifold orbital calibration.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "setext.md").write_text(
        "Setext Root\n===========\n\nSetext root section body.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "unclosed_hr.md").write_text(
        "---\n\n# Heading After HR\n\nUnclosed leading horizontal rule body.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "two_hr.md").write_text(
        "---\nkey: val\n---\n\nPlain body without any heading.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "canonical.md").write_text(
        "# Canonical Document Title\n\nBody.\n\n## Notes Omega\n\nNotes.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "notes_h2.md").write_text(
        "## Sub Only\n\nSubsection without H1 heading.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "worker.py").write_text(
        "def run(items):\n    for item in items:\n        yield item + 1\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "one.txt").write_text("Singleline token orbital calibration " * 12, encoding="utf-8")

    _index_corpus(cfg)

    with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
        idx = build_retrieval_index(ro_conn)
        assert idx.citation_graph is not None
        assert idx.doc_titles == {
            "alpha.md": "Alpha Manifold",
            "canonical.md": "Canonical Document Title",
            "card.md": "Card Heading",
            "setext.md": "Setext Root",
            "unclosed_hr.md": "Heading After HR",
        }
        assert "two_hr.md" not in idx.doc_titles
        assert "notes_h2.md" not in idx.doc_titles
        assert "worker.py" not in idx.doc_titles

        card_pos = next(i for i, p in enumerate(idx.pointers) if p.rel_posix_path == "card.md")
        alpha_di = idx.distinct_paths.index("alpha.md")
        cg = idx.citation_graph
        cited_dis = set(cg.fx[int(cg.fi[card_pos]) : int(cg.fi[card_pos + 1])].tolist())
        assert alpha_di in cited_dis

        # Hydrating YAML frontmatter ATX & Setext cards (heading line >= 6) never duplicates title above frontmatter
        assert hydrate_passage(idx.pointers[card_pos], cfg.corpus_root, include_island=True).text.startswith("---\n")
        unclosed_pos = next(i for i, p in enumerate(idx.pointers) if p.rel_posix_path == "unclosed_hr.md")
        assert hydrate_passage(idx.pointers[unclosed_pos], cfg.corpus_root, include_island=True).text.startswith(
            "---\n"
        )

        disk_cg = build_citation_graph(idx.pointers, idx.distinct_paths, idx.path_idx, cfg.corpus_root)
        assert disk_cg.doc_titles == cg.doc_titles

    fm_cfg = _make_cfg(tmp_path / "fm_hydration", chunk_size=500, chunk_overlap=0, min_chunk_size=10)
    assert fm_cfg.corpus_root is not None and fm_cfg.db_path is not None
    fm_cfg.corpus_root.mkdir(parents=True)
    (fm_cfg.corpus_root / "multi_atx.md").write_text(
        "---\nsources:\n  - alpha.md\n  - beta.md\n  - gamma.md\n---\n\n"
        "# Multiline Card Heading\n\nBody summary referencing orbital calibration.\n",
        encoding="utf-8",
    )
    (fm_cfg.corpus_root / "multi_setext.md").write_text(
        "---\nsources:\n  - alpha.md\n  - beta.md\n  - gamma.md\n---\n\n"
        "Setext Card\n===========\n\nSetext summary referencing orbital calibration.\n",
        encoding="utf-8",
    )
    (fm_cfg.corpus_root / "fenced_sub.md").write_text(
        "## Fake H1\n\n```md\n# Fake H1\n```\n\nBody text inside section.\n",
        encoding="utf-8",
    )
    (fm_cfg.corpus_root / "deco_route.py").write_text(
        '@route("/v1(beta")\ndef ep():\n    return 1\n',
        encoding="utf-8",
    )
    _index_corpus(fm_cfg)
    with contextlib.closing(open_readonly_db(fm_cfg.db_path)) as fm_conn:
        fm_idx = build_retrieval_index(fm_conn)
        assert fm_idx.doc_titles == {
            "multi_atx.md": "Multiline Card Heading",
            "multi_setext.md": "Setext Card",
        }
        assert "fenced_sub.md" not in fm_idx.doc_titles
        assert "deco_route.py" not in fm_idx.doc_titles
        deco_ptr = next(p for p in fm_idx.pointers if p.rel_posix_path == "deco_route.py")
        assert deco_ptr.title == "def ep()"
        assert hydrate_passage(deco_ptr, fm_cfg.corpus_root, include_island=False).text.startswith(
            '@route("/v1(beta")\n'
        )
        assert hydrate_passage(deco_ptr, fm_cfg.corpus_root, include_island=True).text.startswith(
            '@route("/v1(beta")\n'
        )
        atx_ptr = next(p for p in fm_idx.pointers if p.rel_posix_path == "multi_atx.md")
        setext_ptr = next(p for p in fm_idx.pointers if p.rel_posix_path == "multi_setext.md")
        hp_atx = hydrate_passage(atx_ptr, fm_cfg.corpus_root, include_island=True)
        hp_setext = hydrate_passage(setext_ptr, fm_cfg.corpus_root, include_island=True)
        assert hp_atx.text.startswith("---\nsources:\n")
        assert not hp_atx.text.startswith("Multiline Card Heading\n")
        assert hp_setext.text.startswith("---\nsources:\n")
        assert not hp_setext.text.startswith("Setext Card\n")
        ranked_hps = hydrate_ranked_passages(
            fm_idx,
            [(atx_ptr.chunk_id, 0.9), (setext_ptr.chunk_id, 0.8)],
            fm_cfg.corpus_root,
            top_k=2,
            include_island=True,
        )
        assert len(ranked_hps) == 2
        assert all(hp.text.startswith("---\nsources:\n") for hp in ranked_hps)

    # Full-island chunk [3, 5] and sub-chunk [4, 5] share island (3, 5) and coalesce symmetrically
    island_doc = tmp_path / "island_doc"
    island_doc.mkdir()
    doc_txt = "# Root\n\n## Sec\nLine 4 detail.\nLine 5 detail.\n"
    (island_doc / "doc.md").write_text(doc_txt, encoding="utf-8")
    (island_doc / "one.txt").write_text("One line repeated token " * 20, encoding="utf-8")
    doc_lines = split_lf_lines(doc_txt)
    one_lines = split_lf_lines((island_doc / "one.txt").read_text(encoding="utf-8"))
    ptr1 = ChunkPointer(
        chunk_id=101,
        rel_path="doc.md",
        source_hash=fingerprint_str("\n".join(doc_lines[2:5])).hex(),
        start_line=3,
        end_line=5,
        char_start=8,
        char_end=len(doc_txt) - 1,
        ast_island_start=3,
        ast_island_end=5,
        title="Sec",
    )
    ptr2 = ChunkPointer(
        chunk_id=102,
        rel_path="doc.md",
        source_hash=fingerprint_str("\n".join(doc_lines[3:5])).hex(),
        start_line=4,
        end_line=5,
        char_start=15,
        char_end=len(doc_txt) - 1,
        ast_island_start=3,
        ast_island_end=5,
        title="Sec",
    )
    one_p1 = ChunkPointer(
        chunk_id=201,
        rel_path="one.txt",
        source_hash=fingerprint_str(one_lines[0]).hex(),
        start_line=1,
        end_line=1,
        char_start=0,
        char_end=119,
        ast_island_start=1,
        ast_island_end=1,
        title="",
    )
    one_p2 = ChunkPointer(
        chunk_id=202,
        rel_path="one.txt",
        source_hash=fingerprint_str(one_lines[0]).hex(),
        start_line=1,
        end_line=1,
        char_start=120,
        char_end=239,
        ast_island_start=1,
        ast_island_end=1,
        title="",
    )
    sym_idx = replace(
        empty_retrieval_index(),
        chunk_ids=(101, 102, 201, 202),
        id_to_pos={101: 0, 102: 1, 201: 2, 202: 3},
        pointers=(ptr1, ptr2, one_p1, one_p2),
        path_idx=np.array([0, 0, 1, 1], dtype=np.uint32),
        distinct_paths=("doc.md", "one.txt"),
    )
    sym_ranked = [(101, 0.9), (102, 0.8), (201, 0.7), (202, 0.6)]
    assert [cid for _, cid, _ in dedup_ranked_positions(sym_idx, sym_ranked, include_island=True)] == [101, 201, 202]
    hps = hydrate_ranked_passages(sym_idx, sym_ranked, island_doc, top_k=5)
    assert [hp.pointer.chunk_id for hp in hps] == [101, 201, 202]
    assert hps[0].resolved_start_line == 3 and hps[0].resolved_end_line == 5

    # H1 behind ## Preface still populates doc_titles and cross-doc citation edges cleanly
    pref_cfg = _make_cfg(tmp_path / "pref_h1", chunk_size=40, chunk_overlap=0, min_chunk_size=10)
    assert pref_cfg.corpus_root is not None and pref_cfg.db_path is not None
    pref_cfg.corpus_root.mkdir(parents=True)
    (pref_cfg.corpus_root / "pref.md").write_text(
        "## Preface\n\nIntro notes before main heading.\n\n# Main Document Title\n\nBody text.\n",
        encoding="utf-8",
    )
    (pref_cfg.corpus_root / "citing_pref.md").write_text(
        "# Referrer Note\n\nSummary citing Main Document Title calibration details.\n",
        encoding="utf-8",
    )
    _index_corpus(pref_cfg)
    with contextlib.closing(open_readonly_db(pref_cfg.db_path)) as pref_conn:
        pref_idx = build_retrieval_index(pref_conn)
        assert pref_idx.doc_titles == {
            "pref.md": "Main Document Title",
            "citing_pref.md": "Referrer Note",
        }
        assert pref_idx.citation_graph is not None
        pref_di = pref_idx.distinct_paths.index("pref.md")
        cite_positions = [i for i, p in enumerate(pref_idx.pointers) if p.rel_posix_path == "citing_pref.md"]
        cited = {
            int(dst)
            for pos in cite_positions
            for dst in pref_idx.citation_graph.fx[
                int(pref_idx.citation_graph.fi[pos]) : int(pref_idx.citation_graph.fi[pos + 1])
            ].tolist()
        }
        assert pref_di in cited

    # Tagged headers (# [RFC-12] Tagged Header) and YAML frontmatter comments resolve cleanly without corruption
    tag_cfg = _make_cfg(tmp_path / "tagged_yaml", chunk_size=45, chunk_overlap=0, min_chunk_size=10)
    assert tag_cfg.corpus_root is not None and tag_cfg.db_path is not None
    tag_cfg.corpus_root.mkdir(parents=True)
    (tag_cfg.corpus_root / "tagged.md").write_text(
        "# [RFC-12] Tagged Header\n\nBody details referencing calibration.\n",
        encoding="utf-8",
    )
    (tag_cfg.corpus_root / "card_yaml_comment.md").write_text(
        "---\n# yaml comment\nsources:\n  - a.md\n---\n\n# Real Heading\n\nBody text.\n",
        encoding="utf-8",
    )
    _index_corpus(tag_cfg)
    with contextlib.closing(open_readonly_db(tag_cfg.db_path)) as tag_conn:
        tag_idx = build_retrieval_index(tag_conn)
        assert tag_idx.doc_titles == {
            "tagged.md": "Tagged Header",
            "card_yaml_comment.md": "Real Heading",
        }
        tag_cg = build_citation_graph(tag_idx.pointers, tag_idx.distinct_paths, tag_idx.path_idx, tag_cfg.corpus_root)
        assert dict(zip(tag_idx.distinct_paths, tag_cg.doc_titles, strict=True)) == {
            "tagged.md": "Tagged Header",
            "card_yaml_comment.md": "Real Heading",
        }
        tag_ptr = next(p for p in tag_idx.pointers if p.rel_posix_path == "tagged.md" and p.start_line == 1)
        assert tag_ptr.title == "Tagged Header"
        hp_tag = hydrate_passage(tag_ptr, tag_cfg.corpus_root, include_island=True)
        assert hp_tag.text.startswith("# [RFC-12] Tagged Header\n")
        assert not hp_tag.text.startswith("Tagged Header\n")
