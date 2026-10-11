"""Adversarial stress-test suite for Epic 2: Vault Purity, Canonical URI Regex, and Composite PK Multi-Note Persistence."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from yaw import config
from yaw.indexer.pipeline import _node_id, sync_vault
from yaw.retrieval.retriever import GraphRAGRetriever
from yaw.storage.db import ensure_schema_views, get_db

ENGINE_TERMS = ("fastmcp", "sqlite_vec", "pytest", "spacy_llm")
CANONICAL_URI_RE = re.compile(r"^[a-z_]+:[a-z0-9_-]+$")
CHUNK_URI_RE = re.compile(r"^.+#chunk:\d+$")


def test_adversarial_existing_vault_db_purity():
    """Verify that sample-llm-wiki/.cache/yaw.db exists and contains ZERO engine terms."""
    vault_path = Path("sample-llm-wiki").resolve()
    db_path = vault_path / ".cache" / "yaw.db"
    if not db_path.is_file():
        sync_vault(vault_dir=vault_path, force=False, verbose=False)
    assert db_path.is_file(), f"Database does not exist at {db_path}"

    with sqlite3.connect(db_path) as conn:
        ensure_schema_views(conn)

        # 1. Inspect leaf_nodes
        nodes = conn.execute(
            "SELECT node_id, name, namespace, definition, source_leaf_id FROM leaf_nodes"
        ).fetchall()
        assert len(nodes) > 0, "leaf_nodes is empty"

        for node_id, name, namespace, definition, source_leaf_id in nodes:
            combined = (
                f"{node_id} {name} {namespace} {definition} {source_leaf_id}".lower()
            )
            for term in ENGINE_TERMS:
                assert term not in combined, (
                    f"Engine term '{term}' found in leaf_nodes: {node_id} ({name})"
                )

            # Assert canonical regex
            assert CANONICAL_URI_RE.match(node_id), (
                f"node_id '{node_id}' does not match canonical URI regex"
            )
            assert "#" not in node_id, f"node_id contains path separator '#': {node_id}"

        # 2. Inspect unified_nodes
        u_nodes = conn.execute(
            "SELECT node_id, name, namespace, definition FROM unified_nodes"
        ).fetchall()
        assert len(u_nodes) >= len(nodes), (
            "unified_nodes must contain at least all leaf_nodes"
        )

        # Check leaf concepts inside unified_nodes match canonical regex
        for node_id, name, namespace, definition in u_nodes:
            if namespace != "chunk":
                assert CANONICAL_URI_RE.match(node_id), (
                    f"unified_nodes concept '{node_id}' does not match canonical URI regex"
                )
            else:
                assert CHUNK_URI_RE.match(node_id), (
                    f"unified_nodes chunk '{node_id}' does not match chunk URI regex"
                )

        # 3. Inspect leaf_relations and unified_edges
        u_edges = conn.execute(
            "SELECT src_id, dst_id, rel_type, weight FROM unified_edges"
        ).fetchall()
        assert len(u_edges) > 0, "unified_edges is empty"

        manifests_in = [e for e in u_edges if e[2] == "manifests_in"]
        mentions = [e for e in u_edges if e[2] == "mentions"]

        assert len(manifests_in) > 0, "No manifests_in edges in unified_edges"
        assert len(mentions) > 0, "No mentions edges in unified_edges"

        for src_id, dst_id, rel_type, weight in manifests_in:
            assert CANONICAL_URI_RE.match(src_id), (
                f"manifests_in src_id '{src_id}' is not canonical concept URI"
            )
            assert CHUNK_URI_RE.match(dst_id), (
                f"manifests_in dst_id '{dst_id}' is not chunk URI"
            )
            assert abs(weight - 1.0) < 1e-4, f"manifests_in weight is not 1.0: {weight}"


@pytest.mark.anyio
async def test_adversarial_sample_vault_retrieval():
    """Adversarially verify that retrieval works on the real sample-llm-wiki vault indexed via sync_vault."""
    vault_path = Path("sample-llm-wiki").resolve()
    db_path = vault_path / ".cache" / "yaw.db"
    if not db_path.is_file():
        sync_vault(vault_dir=vault_path, force=False, verbose=False)
    old_root = config.VAULT_ROOT
    try:
        config.set_vault(vault_path)
        retriever = GraphRAGRetriever()
        result = await retriever.retrieve("embedding gemma model", top_k=2)
        assert "_No ground-truth passages found._" not in result, (
            f"Retrieval failed on reference vault: {result}"
        )
    finally:
        config.set_vault(old_root)


def test_adversarial_canonical_node_id_generator_stress():
    """Adversarially stress-test _node_id with diverse, pathological, and edge-case inputs."""
    test_cases = [
        # Normal
        ("Quantum Mechanics", "concept", "concept:quantum-mechanics"),
        ("Transformer Architecture", "concepts", "concepts:transformer-architecture"),
        # Punctuation & symbols
        ("E = mc^2", "physics", "physics:e-mc-2"),
        ("C++ & Python", "languages", "languages:c-python"),
        ("U.S.A.", "entity", "entity:u-s-a"),
        ("Path/with#symbols@here", "symbols", "symbols:path-with-symbols-here"),
        ("foo---bar", "test", "test:foo-bar"),
        # Whitespace
        ("   leading trailing   ", "spaces", "spaces:leading-trailing"),
        ("\tmulti\nline\r\ntest", "whitespace", "whitespace:multi-line-test"),
        # Digits in concept name
        ("GPT-4", "models", "models:gpt-4"),
        ("Python 3.12", "version", "version:python-3-12"),
        ("123-abc-456", "mixed", "mixed:123-abc-456"),
        ("42", "numbers", "numbers:42"),
        # Non-ASCII / Accents
        ("Schrödinger", "physicists", "physicists:schrodinger"),
        ("naïve-bayes", "algorithms", "algorithms:naive-bayes"),
        ("café", "places", "places:cafe"),
        # Extreme / empty fallbacks
        ("", "concept", "concept:unknown"),
        ("   ", "concept", "concept:unknown"),
        ("---", "concept", "concept:unknown"),
        ("!@#$%^&*()", "concept", "concept:unknown"),
        ("量子力学", "concept", "concept:unknown"),
    ]

    for concept_name, ns, expected in test_cases:
        actual = _node_id(concept_name, namespace=ns)
        assert actual == expected, (
            f"Failed for ('{concept_name}', '{ns}'): expected '{expected}', got '{actual}'"
        )
        assert CANONICAL_URI_RE.match(actual), (
            f"Generated URI '{actual}' does not match canonical regex ^[a-z_]+:[a-z0-9_-]+$"
        )


def test_adversarial_namespace_variations():
    """Stress test namespace inputs to _node_id to ensure canonical regex compliance."""
    namespaces = [
        "concept",
        "CONCEPTS",
        "entity_type",
        "entity-name",
        "My_Special_Namespace",
        "",
        "   ",
        "---",
        "!!!",
    ]

    for ns in namespaces:
        uri = _node_id("My Concept", namespace=ns)
        # Check regex
        assert CANONICAL_URI_RE.match(uri), (
            f"URI '{uri}' from namespace '{ns}' failed canonical regex"
        )


def test_adversarial_composite_pk_multi_note_stress(tmp_path):
    """Stress-test composite PK with 20 distinct notes mentioning the same concept with identical and varying definitions."""
    db_path = tmp_path / "multi_note_stress.db"
    dummy_vec = np.ones(768, dtype=np.float32).tobytes()

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE leaf_nodes (
                node_id TEXT NOT NULL,
                source_leaf_id TEXT NOT NULL,
                name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                definition TEXT NOT NULL,
                embedding BLOB,
                PRIMARY KEY (node_id, source_leaf_id)
            )
        """)
        conn.execute("""
            CREATE TABLE note_chunks (
                entity_uri TEXT PRIMARY KEY,
                heading TEXT,
                rel_path TEXT
            )
        """)
        ensure_schema_views(conn)

        concept_id = "concepts:superposition"
        num_notes = 20

        # Insert 10 notes with identical definitions
        for i in range(1, 11):
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, ?, ?, ?, ?, ?)",
                (
                    concept_id,
                    f"raw/note_{i:02d}.md",
                    "Superposition",
                    "concepts",
                    "Quantum state principle.",
                    dummy_vec,
                ),
            )

        # Insert 10 notes with varying definitions
        for i in range(11, 21):
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, ?, ?, ?, ?, ?)",
                (
                    concept_id,
                    f"raw/note_{i:02d}.md",
                    "Superposition",
                    "concepts",
                    f"Varying definition #{i}.",
                    dummy_vec,
                ),
            )

        conn.commit()

        # Verify all 20 rows persist in leaf_nodes
        persisted = conn.execute(
            "SELECT source_leaf_id, definition FROM leaf_nodes WHERE node_id = ?",
            (concept_id,),
        ).fetchall()
        assert len(persisted) == num_notes, (
            f"Expected {num_notes} persisted rows, got {len(persisted)}"
        )

        # Verify duplicate (node_id, source_leaf_id) collision raises IntegrityError for every note
        for i in range(1, num_notes + 1):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO leaf_nodes VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        concept_id,
                        f"raw/note_{i:02d}.md",
                        "Superposition",
                        "concepts",
                        "Collision attempt",
                        dummy_vec,
                    ),
                )

        # Check unified_nodes view
        # The 10 identical definitions deduplicate to 1 row, plus 10 unique varying definitions = 11 rows
        u_rows = conn.execute(
            "SELECT node_id, name, namespace, definition FROM unified_nodes WHERE node_id = ?",
            (concept_id,),
        ).fetchall()
        assert len(u_rows) == 11, (
            f"Expected 11 deduplicated rows in unified_nodes, got {len(u_rows)}"
        )

        # If all 20 notes have identical definition, verify it deduplicates to exactly 1 row
        conn.execute("DELETE FROM leaf_nodes")
        for i in range(1, 21):
            conn.execute(
                "INSERT INTO leaf_nodes VALUES (?, ?, ?, ?, ?, ?)",
                (
                    concept_id,
                    f"raw/note_{i:02d}.md",
                    "Superposition",
                    "concepts",
                    "Universal definition.",
                    dummy_vec,
                ),
            )
        conn.commit()

        u_rows_identical = conn.execute(
            "SELECT node_id, name, namespace, definition FROM unified_nodes WHERE node_id = ?",
            (concept_id,),
        ).fetchall()
        assert len(u_rows_identical) == 1, (
            f"Expected exactly 1 row in unified_nodes for identical defs, got {len(u_rows_identical)}"
        )
        assert u_rows_identical[0][0] == concept_id


@pytest.mark.anyio
async def test_adversarial_synthetic_multi_note_vault_indexing_and_retrieval(tmp_path):
    """End-to-end stress test: create a vault with multiple notes referencing overlapping concepts, sync, and retrieve."""
    old_root = config.VAULT_ROOT
    try:
        config.set_vault(tmp_path)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        # Note A defines and mentions Quantum Mechanics and Uncertainty Principle
        (raw_dir / "heisenberg.md").write_text(
            "# Werner Heisenberg\n\n"
            "Werner Heisenberg formulated [[matrix-mechanics]] and collaborated with [[born]].\n\n"
            "## Matrix Mechanics\n\n"
            "Matrix mechanics was the first complete definition of quantum mechanics.\n",
            encoding="utf-8",
        )

        # Note B also defines Matrix Mechanics and mentions Born
        (raw_dir / "born.md").write_text(
            "# Max Born\n\n"
            "Max Born collaborated with [[heisenberg]] on [[matrix-mechanics]].\n\n"
            "## Quantum Probability\n\n"
            "Born introduced the statistical interpretation of wave functions.\n",
            encoding="utf-8",
        )

        # Note C references Matrix Mechanics and Born
        (raw_dir / "dirac.md").write_text(
            "# Paul Dirac\n\n"
            "Paul Dirac synthesized [[matrix-mechanics]] and wave mechanics, citing [[born]].\n",
            encoding="utf-8",
        )

        # Note D is matrix-mechanics.md
        (raw_dir / "matrix-mechanics.md").write_text(
            "# Matrix Mechanics\n\n"
            "Matrix mechanics is a formulation of quantum mechanics created by [[heisenberg]] and [[born]].\n",
            encoding="utf-8",
        )

        # Sync vault
        sync_vault(vault_dir=tmp_path, force=True, verbose=False)

        db_path = tmp_path / ".cache" / "yaw.db"
        assert db_path.is_file()

        with sqlite3.connect(db_path) as conn:
            ensure_schema_views(conn)

            # Check leaf_nodes
            leaf_rows = conn.execute(
                "SELECT node_id, source_leaf_id, name FROM leaf_nodes"
            ).fetchall()
            assert len(leaf_rows) > 0

            # All node_id must match canonical regex
            for nid, src_id, name in leaf_rows:
                assert CANONICAL_URI_RE.match(nid), (
                    f"node_id {nid} failed canonical regex"
                )

            # Check wikilinks
            wl_rows = conn.execute(
                "SELECT source_uri, target_uri FROM wikilinks"
            ).fetchall()
            assert len(wl_rows) >= 3

            # Check unified_edges has wikilinks and grounding edges
            u_edges = conn.execute(
                "SELECT src_id, dst_id, rel_type, weight FROM unified_edges"
            ).fetchall()
            assert len(u_edges) > 0

            # Verify manifests_in and mentions edges
            grounding = [e for e in u_edges if e[2] in ("manifests_in", "mentions")]
            assert len(grounding) > 0

            nc_rows = conn.execute("SELECT entity_uri FROM note_chunks").fetchall()
            assert len(nc_rows) > 0
            un_rows = conn.execute("SELECT * FROM unified_nodes").fetchall()
            assert len(un_rows) > 0

        # Now test retrieval with GraphRAGRetriever
        retriever = GraphRAGRetriever()
        result = await retriever.retrieve("matrix mechanics", top_k=2)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "_No ground-truth passages found._" not in result, (
            f"Retrieval returned empty results for indexed multi-note vault: {result}"
        )
        assert (
            "matrix" in result.lower()
            or "heisenberg" in result.lower()
            or "born" in result.lower()
        )

    finally:
        config.set_vault(old_root)


@pytest.mark.anyio
async def test_adversarial_corrupt_embeddings_resilience(tmp_path):
    """Verify that corrupt JSON strings, malformed bytes, and unaligned buffers in note_chunks/leaf_nodes
    do not crash candidate vector fetching, seed calculation, or retrieval."""
    from yaw.server.daemon import _get_candidate_vectors, compute_seed_candidates

    old_root = config.VAULT_ROOT
    try:
        config.set_vault(tmp_path)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "note.md").write_text(
            "# Resilience Note\n\nResilience against malformed embeddings.\n",
            encoding="utf-8",
        )

        sync_vault(vault_dir=tmp_path, force=True, verbose=False)
        db_path = tmp_path / ".cache" / "yaw.db"
        assert db_path.is_file()

        conn = get_db(load_vec=True)
        try:
            ensure_schema_views(conn)
            # Insert corrupted embeddings into note_chunks:
            # 1. Corrupted JSON string
            # 2. Odd-length byte buffer (13 bytes, not multiple of 4)
            conn.execute(
                """
                INSERT OR REPLACE INTO note_chunks 
                (chunk_id, entity_uri, rel_path, ordinal, heading, heading_slug, line_start, line_end, embedding)
                VALUES 
                ('cid:corrupt_json', 'raw/note.md#chunk:bad_json', 'raw/note.md', 10, 'Bad JSON', 'bad-json', 1, 2, '{not valid json'),
                ('cid:corrupt_bytes', 'raw/note.md#chunk:bad_bytes', 'raw/note.md', 11, 'Bad Bytes', 'bad-bytes', 3, 4, ?)
                """,
                (b"odd_bytes_123",),
            )

            # Insert corrupted embeddings into leaf_nodes:
            # 1. Corrupted JSON string
            # 2. Odd-length byte buffer (13 bytes, not multiple of 4)
            conn.execute(
                """
                INSERT OR REPLACE INTO leaf_nodes 
                (node_id, source_leaf_id, name, namespace, definition, embedding)
                VALUES 
                ('concepts:bad_json', 'raw/note.md', 'BadJsonConcept', 'concepts', 'Corrupt JSON def', '{not valid json'),
                ('concepts:bad_bytes', 'raw/note.md', 'BadByteConcept', 'concepts', 'Corrupt byte def', ?)
                """,
                (b"odd_bytes_456",),
            )
            conn.commit()

            # 1. Direct candidate vector extraction must not crash
            test_uris = [
                "raw/note.md#chunk:bad_json",
                "raw/note.md#chunk:bad_bytes",
                "concepts:bad_json",
                "concepts:bad_bytes",
            ]
            vecs = _get_candidate_vectors(test_uris)
            assert isinstance(vecs, dict)
            for uri in test_uris:
                assert uri not in vecs, f"Corrupted URI {uri} should have been skipped"

            # 2. compute_seed_candidates must execute without raising ValueError or crashing
            candidates = await compute_seed_candidates("resilience note", top_k=3)
            assert isinstance(candidates, list)

            # 3. GraphRAGRetriever must execute without crashing
            retriever = GraphRAGRetriever()
            ret_output = await retriever.retrieve("resilience note", top_k=2)
            assert isinstance(ret_output, str)
            assert len(ret_output) > 0
        finally:
            conn.close()
    finally:
        config.set_vault(old_root)


@pytest.mark.anyio
async def test_adversarial_null_distance_handling(tmp_path):
    """Verify that zero-norm vectors returning NULL distance in sqlite-vec do not cause
    TypeError or abort the candidate loops in retriever or daemon."""
    from yaw.server.daemon import _query_vectors, compute_seed_candidates

    old_root = config.VAULT_ROOT
    try:
        config.set_vault(tmp_path)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "zero.md").write_text(
            "# Zero Vector\n\nTesting zero vector cosine distance.\n",
            encoding="utf-8",
        )

        sync_vault(vault_dir=tmp_path, force=True, verbose=False)
        db_path = tmp_path / ".cache" / "yaw.db"
        assert db_path.is_file()

        zero_vec = np.zeros(768, dtype=np.float32)
        valid_vec = (np.ones(768, dtype=np.float32) / np.float32(np.sqrt(768))).astype(
            np.float32
        )

        conn = get_db(load_vec=True)
        try:
            ensure_schema_views(conn)
            # Insert zero-vector chunk & valid chunk
            conn.execute(
                """
                INSERT OR REPLACE INTO note_chunks 
                (chunk_id, entity_uri, rel_path, ordinal, heading, heading_slug, line_start, line_end, embedding)
                VALUES 
                ('cid:zero_chunk', 'raw/zero.md#chunk:zero', 'raw/zero.md', 20, 'Zero Chunk', 'zero-chunk', 1, 2, ?),
                ('cid:valid_chunk', 'raw/zero.md#chunk:valid', 'raw/zero.md', 21, 'Valid Chunk', 'valid-chunk', 3, 4, ?)
                """,
                (zero_vec.tobytes(), valid_vec.tobytes()),
            )

            # Insert zero-vector leaf node & valid leaf node
            conn.execute(
                """
                INSERT OR REPLACE INTO leaf_nodes 
                (node_id, source_leaf_id, name, namespace, definition, embedding)
                VALUES 
                ('concepts:zero_concept', 'raw/zero.md', 'ZeroConcept', 'concepts', 'Zero concept definition', ?),
                ('concepts:valid_concept', 'raw/zero.md', 'ValidConcept', 'concepts', 'Valid concept definition', ?)
                """,
                (zero_vec.tobytes(), valid_vec.tobytes()),
            )
            conn.commit()

            # Verify sqlite-vec returns NULL for zero-vector cosine distance
            sample_q = (
                np.ones(768, dtype=np.float32) / np.float32(np.sqrt(768))
            ).astype(np.float32)
            dist_rows = conn.execute(
                """
                SELECT entity_uri, vec_distance_cosine(embedding, ?) 
                FROM note_chunks WHERE entity_uri IN ('raw/zero.md#chunk:zero', 'raw/zero.md#chunk:valid')
                """,
                (sample_q.tobytes(),),
            ).fetchall()
            for uri, d in dist_rows:
                if uri == "raw/zero.md#chunk:zero":
                    assert d is None, (
                        f"Expected None cosine distance for zero vector, got {d}"
                    )
                elif uri == "raw/zero.md#chunk:valid":
                    assert d is not None, (
                        f"Expected numeric distance for valid vector, got {d}"
                    )

            # 1. _query_vectors in daemon must skip NULL distance rows without TypeError
            vec_res = _query_vectors(sample_q, top_n=20)
            res_uris = [r["entity_uri"] for r in vec_res]
            assert "raw/zero.md#chunk:zero" not in res_uris
            assert "concepts:zero_concept" not in res_uris
            assert "raw/zero.md#chunk:valid" in res_uris
            assert "concepts:valid_concept" in res_uris

            # 2. retriever.compute_seed_vector must skip NULL distance rows without TypeError
            retriever = GraphRAGRetriever()
            _p0, _q_ret_vec, cand_chunks = await retriever.compute_seed_vector(
                "zero vector", conn
            )
            cand_uris = [c["entity_uri"] for c in cand_chunks]
            assert "raw/zero.md#chunk:zero" not in cand_uris
            assert "raw/zero.md#chunk:valid" in cand_uris

            # 3. retriever.retrieve must complete end-to-end
            ret_output = await retriever.retrieve("zero vector", top_k=2)
            assert isinstance(ret_output, str)
            assert len(ret_output) > 0

            # 4. compute_seed_candidates in daemon must complete end-to-end
            candidates = await compute_seed_candidates("zero vector", top_k=3)
            assert isinstance(candidates, list)
        finally:
            conn.close()
    finally:
        config.set_vault(old_root)
