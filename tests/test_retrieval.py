"""Unmocked real-substrate tests for M_FINAL_1B 1-bit RaBitQ, cyclic diffusion, unicode61 FTS5, and hydration."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import yaw.retrieval as retrieval_mod
from yaw.core import ChunkPointer, YawConfig, YawRuntimeError
from yaw.pipeline import close_pipeline, start_pipeline, update_pipeline_once
from yaw.retrieval import (
    SearchResult,
    build_retrieval_index,
    compute_db_fingerprints,
    compute_index_artifacts,
    empty_retrieval_index,
    expand_citation_neighbors,
    format_evidence_route,
    hydrate_passage,
    hydrate_ranked_passages,
    lexical_search,
    normalize_fts_unicode61,
    open_readonly_db,
    search_index,
)


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
