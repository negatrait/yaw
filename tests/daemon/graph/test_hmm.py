"""Unit, spectral invariant, CocoIndex batch-invariance, and shootout tests for pure-chunk hmm.py."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import redirect_stdout
from dataclasses import fields, replace
from graphlib import TopologicalSorter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import dspy
import numpy as np
import pytest

from daemon.graph.synonym_protocol import candidate_rows, complete_decisions
from daemon.graph.test_indexing_overlap import adapter
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer, compute_chunk_id, load_config
from yaw.daemon.cli import main
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.hmm import (
    EvidenceHop,
    ManifoldDerivedTelemetry,
    ManifoldHyperparams,
    SubStateManifold,
    SubStateResult,
    build_substate_manifold,
    derive_manifold_telemetry,
    format_evidence_route,
    run_substate_hmm,
)
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    YawProgram,
    create_dspy_lm,
    resolve_embedding_profile,
)
from yaw.daemon.graph.storage import load_manifold_chunks
from yaw.daemon.graph.storage.rows import validate_graph_state
from yaw.daemon.indexing import bind_native_indexing
from yaw.daemon.ingestion import prepare_dataset_bundle
from yaw.daemon.runtime import YawRuntime

_FORBIDDEN_SUB = "sub" + "span"


def _basis(dim: int, idx: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / max(norm, 1e-12)).astype(np.float32)


def _keyword_vec4(text: str) -> list[float]:
    lower = text.casefold()
    vec = [
        1.0 if ("helios" in lower or "propulsion" in lower) else 0.0,
        1.0 if ("bletchley" in lower or "turing" in lower) else 0.0,
        1.0 if ("lovelace" in lower or "analytical" in lower) else 0.0,
        0.1,
    ]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def test_empty_manifold_and_route_formatting_contracts() -> None:
    empty = build_substate_manifold([], {}, np.empty((0, 8), dtype=np.float32))
    assert isinstance(empty, SubStateManifold)
    assert empty.chunk_ids == ()
    assert empty.chunk_full_texts == ()
    res = run_substate_hmm(empty, "anything", np.ones(8, dtype=np.float32))
    assert isinstance(res, SubStateResult)
    assert res.ranked_chunks == []
    assert res.chunk_posteriors.size == 0
    assert res.hop_posteriors == ()
    assert res.hop_weights == ()
    assert res.route == ()
    assert res.num_particles == 0
    assert format_evidence_route(res.route) == ""

    hops = [
        EvidenceHop(0, "c0", "Alpha", "x" * 100, 0.9, bridge="Alpha", branch=0),
        EvidenceHop(1, "c1", "Beta", "y" * 100, 0.7, bridge="Beta", branch=1),
    ]
    assert format_evidence_route(hops, max_chars=0) == ""
    bounded = format_evidence_route(hops, max_chars=90)
    assert len(bounded) <= 90
    assert "Alpha" in bounded


def test_build_manifold_accepts_mapping_and_sequence_texts() -> None:
    ptrs = [
        ChunkPointer("c0", "a.md", 1, 2, title="Alpha", ast_island_start=1, ast_island_end=2),
        ChunkPointer("c1", "b.md", 1, 2, title="Beta", ast_island_start=1, ast_island_end=2),
    ]
    vecs = np.vstack([_basis(16, 0), _basis(16, 1)])
    text_seq = ["Alpha orbital relay notes.", "Beta propulsion test log."]
    text_map = {"c0": text_seq[0], "c1": text_seq[1]}

    m_seq = build_substate_manifold(ptrs, text_seq, vecs)
    m_map = build_substate_manifold(ptrs, text_map, vecs)
    assert m_seq.chunk_full_texts == tuple(text_seq)
    assert m_map.chunk_full_texts == tuple(text_seq)
    assert np.allclose(m_seq.u_chunk_iso, m_map.u_chunk_iso)
    assert m_seq.vocab_index == m_map.vocab_index

    with pytest.raises(ValueError):
        build_substate_manifold(ptrs, ["only one"], vecs)
    with pytest.raises(ValueError):
        build_substate_manifold(ptrs, text_seq, np.zeros((1, 16), dtype=np.float32))


def test_manifold_cocoindex_zero_llm_calls_and_config_fingerprint(tmp_path: Path, provider: Any) -> None:
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = _keyword_vec4

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc_a = corpus_dir / "helios.md"
    doc_b = corpus_dir / "lovelace.md"
    text_a = (
        "Project Helios coordinates orbital propulsion telemetry.\n\n"
        "The Project Helios bridge links propulsion relays through Bletchley Park archives."
    )
    text_b = (
        "Ada Lovelace authored the 1843 Analytical Engine notes.\n\n"
        "Her notes connect early symbolic computation with Bletchley Park cryptanalysis."
    )
    doc_a.write_text(text_a + "\n", encoding="utf-8")
    doc_b.write_text(text_b + "\n", encoding="utf-8")

    toml_path = tmp_path / "yaw.toml"
    toml_path.write_text(
        f"""
[paths]
corpus_root = "corpus"
db_path = ".yaw/yaw.db"
lock_path = ".yaw/yaw.db.lock"
state_dir = ".yaw/coco_lmdb"
log_path = ".yaw/yaw.log"
status_path = ".yaw/status.json"
compiled_program_path = ".yaw/compiled_program.json"

[inference]
base_url = "{url}"
enable_gbnf = false

[retrieval]
retriever_mode = "manifold"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg_manifold = load_config(toml_path)
    cfg_graph = replace(cfg_manifold, retriever_mode="graph")
    cfg_tuned = replace(cfg_manifold, manifold_overrides={"peclet": 0.55})
    assert cfg_manifold.retriever_mode == "manifold"
    assert cfg_manifold.config_fingerprint() != cfg_graph.config_fingerprint()
    assert cfg_manifold.config_fingerprint() == cfg_tuned.config_fingerprint()

    sb_manifold = cfg_manifold.for_benchmark("2wikimultihopqa", "deadbeef")
    sb_graph = cfg_graph.for_benchmark("2wikimultihopqa", "deadbeef")
    assert sb_manifold.db_path is not None and sb_graph.db_path is not None
    assert sb_manifold.db_path.parent.name.endswith("_manifold")
    assert not sb_graph.db_path.parent.name.endswith("_manifold")

    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=256)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=256)
    engine = bind_native_indexing(
        YawGraphEngine(
            tmp_path / "yaw.db",
            corpus_dir,
            program,
            adapter(url),
            retriever_mode="manifold",
            max_context_chars=2400,
        )
    )
    try:
        chunks = [
            (ChunkPointer(compute_chunk_id("helios.md", 1, 3, text_a), "helios.md", 1, 3, "helios", 1, 3), text_a),
            (
                ChunkPointer(compute_chunk_id("lovelace.md", 1, 3, text_b), "lovelace.md", 1, 3, "lovelace", 1, 3),
                text_b,
            ),
        ]
        published = engine.index_chunks(chunks)
        assert published == 2
        assert state.get("extraction_requests", []) == []
        assert engine.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
        assert engine.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
        assert {row[0] for row in engine.conn.execute("SELECT program_identity FROM chunk_inference")} == {
            "manifold-v3"
        }
        validate_graph_state(engine.conn)

        loaded_ptrs, loaded_vecs = load_manifold_chunks(engine.conn)
        assert len(loaded_ptrs) == 2
        assert loaded_vecs.shape == (2, 4)

        retrieved = engine.retrieve("Where do Project Helios and Ada Lovelace connect?", top_k=2)
        assert len(retrieved) >= 1
        qa_ctx = engine.format_qa_context(retrieved, include_subgraph=True)
        assert "Bletchley Park" in qa_ctx or any("Bletchley Park" in text for _, _, text in retrieved)
    finally:
        engine.close()


def test_cocoindex_batch_invariance_64_plus_64_equals_128(tmp_path: Path, provider: Any) -> None:
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = _keyword_vec4

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    chunks: list[tuple[ChunkPointer, str]] = []
    for idx in range(128):
        rel = f"doc_{idx:03d}.md"
        body = (
            f"Section {idx} explores Project Helios orbital relay {idx % 7}.\n"
            f"Ada Lovelace note reference {idx % 5} cites Bletchley Park archive {idx}."
        )
        (corpus_dir / rel).write_text(body + "\n", encoding="utf-8")
        ptr = ChunkPointer(compute_chunk_id(rel, 1, 2, body), rel, 1, 2, f"doc_{idx:03d}", 1, 2)
        chunks.append((ptr, body))

    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=256)
    prog = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=256)

    def _make_engine(arm: str, batch_size: int) -> YawGraphEngine:
        return bind_native_indexing(
            YawGraphEngine(tmp_path / arm / "yaw.db", corpus_dir, prog, adapter(url), retriever_mode="manifold"),
            batch_size=batch_size,
        )

    def _snapshot(conn: Any) -> dict[str, list[tuple[Any, ...]]]:
        return {
            "chunks": conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall(),
            "chunk_fts": conn.execute("SELECT chunk_id FROM chunk_fts ORDER BY chunk_id").fetchall(),
            "chunk_vec": conn.execute("SELECT chunk_id, embedding FROM chunk_vec ORDER BY chunk_id").fetchall(),
            "chunk_inference": conn.execute(
                "SELECT chunk_id, program_identity FROM chunk_inference ORDER BY chunk_id"
            ).fetchall(),
        }

    req_before = len(state.get("extraction_requests", []))
    snaps: dict[str, dict[str, list[tuple[Any, ...]]]] = {}
    for arm, batch_size in (("incremental", 64), ("reference", 128)):
        eng = _make_engine(arm, batch_size)
        try:
            if arm == "incremental":
                eng.index_chunks(chunks[:64])
                assert eng.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 64
                eng.close()
                eng = _make_engine(arm, batch_size)
            eng.index_chunks(chunks)
            validate_graph_state(eng.conn)
            snaps[arm] = _snapshot(eng.conn)
            assert {row[0] for row in eng.conn.execute("SELECT program_identity FROM chunk_inference")} == {
                "manifold-v3"
            }
        finally:
            eng.close()

    assert len(state.get("extraction_requests", [])) == req_before
    assert snaps["incremental"] == snaps["reference"]
    assert len(snaps["reference"]["chunk_vec"]) == 128


def test_load_manifold_chunks_sqlite_alignment(tmp_path: Path, provider: Any) -> None:
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = _keyword_vec4

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    text_a = "Project Helios coordinates orbital propulsion telemetry."
    text_b = "Ada Lovelace authored the 1843 Analytical Engine notes."
    (corpus_dir / "a.md").write_text(text_a + "\n", encoding="utf-8")
    (corpus_dir / "b.md").write_text(text_b + "\n", encoding="utf-8")

    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, max_tokens=256)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=256)
    engine = bind_native_indexing(
        YawGraphEngine(
            tmp_path / "yaw.db",
            corpus_dir,
            program,
            adapter(url),
            retriever_mode="manifold",
            max_context_chars=2400,
        )
    )
    try:
        chunks = [
            (ChunkPointer(compute_chunk_id("a.md", 1, 2, text_a), "a.md", 1, 2, "a", 1, 2), text_a),
            (ChunkPointer(compute_chunk_id("b.md", 1, 2, text_b), "b.md", 1, 2, "b", 1, 2), text_b),
        ]
        assert engine.index_chunks(chunks) == 2

        chunk_pointers, chunk_vectors = load_manifold_chunks(engine.conn)
        assert len(chunk_pointers) == 2
        assert chunk_vectors.shape == (2, 4)
        assert chunk_vectors.dtype == np.float32

        texts_by_id = {chunks[0][0].chunk_id: text_a, chunks[1][0].chunk_id: text_b}
        manifold = build_substate_manifold(chunk_pointers, texts_by_id, chunk_vectors)
        assert manifold.u_chunk_iso.shape == (2, 4)

        engine.conn.execute("DELETE FROM chunk_vec WHERE chunk_id = ?", (chunk_pointers[0].chunk_id,))
        engine.conn.commit()
        with pytest.raises(ValueError, match="chunk_vec"):
            load_manifold_chunks(engine.conn)
    finally:
        engine.close()


class _ShootoutTestLM(dspy.LM):
    def __init__(self) -> None:
        super().__init__(model="openai/shootout-test-lm")

    @property
    def supported_params(self) -> set[str]:
        return set()

    @property
    def supports_reasoning(self) -> bool:
        return False

    def __call__(self, prompt: Any = None, **kwargs: Any) -> Any:
        from dspy.lm15 import Message, Response, TextPart, Usage

        messages = [
            {"role": m.role, "content": "".join(p.text for p in m.parts if hasattr(p, "text"))} for m in prompt.messages
        ]
        if prompt.system:
            messages.insert(0, {"role": "system", "content": str(prompt.system)})
        req = str(messages)
        if "JudgeLabels" in req or "candidate_pairs" in req:
            packet_text = "\n".join(str(m.get("content", "")) for m in messages)
            decs = complete_decisions(candidate_rows(packet_text), [], stage_text=packet_text)
            ans = json.dumps({"decisions": decs})
        elif "ProposeSHACLRepairs" in req or "HealSHACLViolations" in req or "violations_json" in req:
            ans = '{"repairs": []}'
        elif "DefineMissingEntities" in req or "missing_entities" in req:
            ans = '{"entity_definitions": [["Alan Turing", "Mathematician"], ["Bletchley Park", "Center"]]}'
        elif "FilterTriples" in req or "fact_before_filter" in req:
            ans = '{"fact": [["Alan Turing", "worked at", "Bletchley Park"]]}'
        elif "ExtractTriples" in req or '"triples"' in req or "`triples`" in req:
            ans = (
                '{"triples": [{"sub": "Alan Turing", "pred": "worked at", "obj": "Bletchley Park", '
                '"def": "Alan Turing worked at Bletchley Park."}]}'
            )
        elif "ExtractEntities" in req or "named_entities" in req:
            ans = '{"named_entities": [["Alan Turing", "Mathematician"], ["Bletchley Park", "Codebreaking center"]]}'
        elif "ExtractQueryIntent" in req or "query_intent" in req:
            ans = '{"query_intent": "Alan Turing worked at Bletchley Park"}'
        else:
            ans = '{"plan": "Check retrieved evidence.", "answer": "Bletchley Park"}'
        return Response(
            id="shootout",
            model=self.model,
            message=Message(role="assistant", parts=(TextPart(text=ans),)),
            finish_reason="stop",
            usage=Usage(input_tokens=16, output_tokens=8),
        )


class _ShootoutTestEmbedder:
    index_identity = "shootout-test-embed-v1"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in texts:
            tl = t.casefold()
            v = np.zeros(self.dim, dtype=np.float32)
            if "turing" in tl or "bletchley" in tl:
                v[0] = 1.0
            else:
                digest = hashlib.md5(tl.encode("utf-8")).digest()
                for i in range(self.dim):
                    v[i] = float(digest[i % len(digest)]) / 255.0 + 0.01
            v /= float(np.linalg.norm(v))
            vecs.append(v.tolist())
        return vecs


@pytest.fixture
def shootout_models_url() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/v1/models", "/models"):
                self.send_error(404)
                return
            raw = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "fixture",
                            "loaded": True,
                            "context_length": 131072 if name == "gemma-4-e4b" else 8192,
                            "alias_for": f"fixture/{name}:Q8",
                        }
                        for name in ("gemma-4-e4b", "embeddinggemma-2")
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_benchmark_shootout_isolated_sandboxes_and_cli_round_trip(tmp_path: Path, shootout_models_url: str) -> None:
    ws = tmp_path / "ws"
    (ws / "corpus").mkdir(parents=True)
    toml_path = ws / "yaw.toml"
    toml_path.write_text(
        f"""
[paths]
corpus_root = "corpus"
db_path = ".yaw/yaw.db"
lock_path = ".yaw/yaw.db.lock"
state_dir = ".yaw/coco_lmdb"
log_path = ".yaw/yaw.log"
status_path = ".yaw/status.json"
compiled_program_path = ".yaw/compiled_program.json"

[inference]
base_url = "{shootout_models_url}"
orchestrator_model = "gemma-4-e4b"
embedding_model = "embeddinggemma-2"
enable_gbnf = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config(toml_path)
    lm = _ShootoutTestLM()
    prog = YawProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    profile = replace(resolve_embedding_profile("embeddinggemma-300m-GGUF", "distinct-titled"), dimensions=16)
    emb = DSPyEmbeddingAdapter(profile=profile, embedder=dspy.Embedder(_ShootoutTestEmbedder(dim=16), batch_size=64))

    samples = [
        {
            "id": "q-1",
            "question": "Where did Alan Turing work?",
            "answer": ["Bletchley Park"],
            "paragraphs": [
                {
                    "title": "Alan Turing",
                    "text": "Alan Turing worked at Bletchley Park during World War II. He led Hut 8 there.",
                    "is_supporting": True,
                }
            ],
        }
    ]
    corpus_rows = [
        {
            "title": "Alan Turing",
            "text": "Alan Turing worked at Bletchley Park during World War II. He led Hut 8 there.",
        }
    ]
    q_file = ws / "2wikimultihopqa.json"
    c_file = ws / "2wikimultihopqa_docs.json"
    q_file.write_text(json.dumps(samples), encoding="utf-8")
    c_file.write_text(json.dumps(corpus_rows), encoding="utf-8")
    bundle = prepare_dataset_bundle("2wikimultihopqa", c_file, q_file, ws / "outputs" / "datasets")

    with dspy.context(lm=lm):
        report = YawRuntime(cfg, program=prog, embedder=emb, auto_poll=False).benchmark(
            datasets=[str(bundle.root)],
            max_queries=1,
            mode="eval",
            retriever="shootout",
        )
    assert report["retriever"] == "shootout"
    assert report["integrity_passed"] is True
    assert report["status"] == "passed"
    assert "baselines" in report and "2wikimultihopqa" in report["baselines"]

    cand_g = report["candidates"]["current_yaw"]
    cand_m = report["candidates"]["substate_manifold_hmm"]
    assert cand_g["retriever"] == "graph"
    assert cand_m["retriever"] == "manifold"
    g_info = cand_g["datasets"]["2wikimultihopqa"]["graph"]
    m_info = cand_m["datasets"]["2wikimultihopqa"]["graph"]
    assert g_info["retriever_mode"] == "graph"
    assert m_info["retriever_mode"] == "manifold"
    assert g_info["num_fact_nodes"] > 0
    assert m_info["num_fact_nodes"] == 0 and m_info["num_chunk_vectors"] > 0
    for info in (g_info, m_info):
        assert isinstance(info["indexing_seconds"], float)
        assert isinstance(info["evaluation_seconds"], float)

    ds_cmp = report["comparison"]["datasets"]["2wikimultihopqa"]
    assert {"Recall@5", "ExactMatch", "F1"}.issubset(ds_cmp["deltas"])
    assert "speedup_x" in ds_cmp

    sandbox_dirs = [p.name for p in (ws / "outputs" / "benchmarks" / "indexes").iterdir() if p.is_dir()]
    assert any(not name.endswith("_manifold") for name in sandbox_dirs)
    assert any(name.endswith("_manifold") for name in sandbox_dirs)

    out_buf = io.StringIO()
    with dspy.context(lm=lm), redirect_stdout(out_buf):
        rc = main(
            [
                "dspy",
                "benchmark",
                "--config",
                str(toml_path),
                "--dataset",
                str(bundle.root),
                "--max-queries",
                "1",
                "--mode",
                "retrieval",
                "--retriever",
                "shootout",
            ],
            program=prog,
            embedder=emb,
        )
    assert rc == 0
    cli_payload = json.loads(out_buf.getvalue())
    assert cli_payload["retriever"] == "shootout"
    assert cli_payload["integrity_passed"] is True


def test_manifold_hyperparams_from_overrides_contract() -> None:
    base = ManifoldHyperparams()
    assert ManifoldHyperparams.from_overrides(None) == base
    assert ManifoldHyperparams.from_overrides({}) == base
    assert ManifoldHyperparams.from_overrides(()) == base

    from_dict = ManifoldHyperparams.from_overrides({"peclet": 1, "alpha_sprt": 0.04})
    assert from_dict.peclet == 1.0 and isinstance(from_dict.peclet, float)
    assert from_dict.alpha_sprt == pytest.approx(0.04)

    from_seq = ManifoldHyperparams.from_overrides(
        [("redundancy_curvature", 1.25), ("coalition_coupling", 0.30), ("external_trust", 0)]
    )
    assert from_seq.redundancy_curvature == pytest.approx(1.25)
    assert from_seq.coalition_coupling == pytest.approx(0.30)
    assert from_seq.external_trust == 0.0 and isinstance(from_seq.external_trust, float)

    with pytest.raises(ValueError, match="Unknown ManifoldHyperparams field"):
        ManifoldHyperparams.from_overrides({"nonexistent_field": 0.5})
    for bad_bool in ({"peclet": True}, {"external_trust": False}):
        with pytest.raises(ValueError, match="int or float"):
            ManifoldHyperparams.from_overrides(bad_bool)
    with pytest.raises(ValueError, match="int or float"):
        ManifoldHyperparams.from_overrides(cast(Any, {"peclet": "0.65"}))
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            ManifoldHyperparams.from_overrides({"peclet": non_finite})
    for out_of_domain in (
        {"peclet": 0.0},
        {"peclet": 1.2},
        {"alpha_sprt": 0.0},
        {"alpha_sprt": 0.5},
        {"redundancy_curvature": -0.1},
        {"redundancy_curvature": 2.1},
        {"coalition_coupling": -0.1},
        {"coalition_coupling": 1.1},
        {"external_trust": -0.1},
        {"external_trust": 1.1},
    ):
        with pytest.raises(ValueError):
            ManifoldHyperparams.from_overrides(out_of_domain)


def test_hmm_line_ceiling_ast_discipline_and_zero_banned_constructs() -> None:
    root_dir = Path(__file__).resolve().parents[3]
    hmm_path = root_dir / "yaw" / "daemon" / "graph" / "hmm.py"
    probe_path = root_dir / "scripts" / "probe_manifold.py"
    test_path = Path(__file__).resolve()

    hmm_source = hmm_path.read_text(encoding="utf-8")
    assert len(hmm_source.splitlines()) <= 500
    for forbidden in ("_STOPWORDS", "_tokenize", "_adjacent_bigrams", "np.add.at"):
        assert forbidden not in hmm_source

    for path in (hmm_path, probe_path, test_path):
        src = path.read_text(encoding="utf-8")
        assert _FORBIDDEN_SUB not in src.lower(), f"Found {_FORBIDDEN_SUB} in {path}"

    tree = ast.parse(hmm_source, filename=str(hmm_path))
    hp_field_names = {f.name for f in fields(ManifoldHyperparams)}
    assert hp_field_names == {
        "peclet",
        "alpha_sprt",
        "redundancy_curvature",
        "coalition_coupling",
        "external_trust",
    }
    accessed_attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert sorted(hp_field_names - accessed_attrs) == []
    assert "dense_mix" not in hmm_source

    tele = derive_manifold_telemetry(ManifoldHyperparams())
    assert isinstance(tele, ManifoldDerivedTelemetry)
    assert tele.default_top_k == 10
    assert tele.num_hops == 8
    assert tele.max_silent_hops == 2
    assert tele.route_max_chars == 600


def test_abtt_removes_rogue_anisotropy_and_csls_suppresses_hubs() -> None:
    rng = np.random.default_rng(42)
    n_chunks, dim = 40, 32
    ptrs = [ChunkPointer(f"c{i}", f"d{i}.md", 1, 2, title=f"Doc{i}") for i in range(n_chunks)]
    texts = [f"Unique passage {i} discussing astronomy topic{i}." for i in range(n_chunks)]

    rogue = _basis(dim, 0)
    raw = np.zeros((n_chunks, dim), dtype=np.float32)
    for i in range(n_chunks):
        signal = _unit(rng.standard_normal(dim).astype(np.float32) * 0.25 + 3.0 * rogue)
        raw[i] = signal
    raw[0] = _unit(3.0 * rogue + 1.5 * _basis(dim, 1))
    raw[1] = _unit(3.0 * rogue - 1.5 * _basis(dim, 1))

    manifold = build_substate_manifold(ptrs, texts, raw)
    assert manifold.abtt_removed_dims >= 1
    assert manifold.rmt_lambda_plus > 0.0
    assert abs(float(manifold.u_chunk_iso[0] @ rogue)) < abs(float(manifold.u_chunk_raw[0] @ rogue))
    assert float(manifold.csls_chunk.max()) >= 0.0
    assert np.all(manifold.csls_chunk >= 0.0)


def test_df_hub_pruning_and_semantic_island_isolation() -> None:
    n_chunks, dim = 24, 32
    ptrs = [ChunkPointer(f"c{i}", f"doc_{i}.md", 1, 2, title=f"Title{i}") for i in range(n_chunks)]
    texts = [f"common_hub_word general article {i} unique_tok_{i}." for i in range(n_chunks)]
    texts[0] = "common_hub_word IslandAlpha bridge_mercury connects crater."
    texts[1] = "common_hub_word IslandBeta bridge_mercury telemetry."
    texts[2] = "common_hub_word IslandGamma unrelated isolated_ocean reef."
    texts[3] = "common_hub_word IslandDelta bridge_mercury orbit."

    vecs = np.zeros((n_chunks, dim), dtype=np.float32)
    for i in range(n_chunks):
        vecs[i] = _basis(dim, (i % (dim - 4)) + 4)
    vecs[0] = _unit(_basis(dim, 0) + 0.4 * _basis(dim, 1))
    vecs[1] = _unit(0.7 * _basis(dim, 0) + 0.7 * _basis(dim, 1))
    vecs[2] = _basis(dim, 3)
    vecs[3] = _unit(0.4 * _basis(dim, 0) + 0.6 * _basis(dim, 1))

    manifold = build_substate_manifold(ptrs, texts, vecs)
    hub_idx = manifold.vocab_index["common_hub_word"]
    assert float(manifold.token_idf2[hub_idx]) == 0.0

    q_vec = _basis(dim, 0)
    res = run_substate_hmm(manifold, "IslandAlpha", q_vec, top_k=5)
    top_ids = [cid for cid, _ in res.ranked_chunks[:3]]
    assert "c0" in top_ids and "c1" in top_ids
    assert "c2" not in top_ids
    assert "c2" not in {hop.chunk_id for hop in res.route}


def test_external_hints_scaling_and_snr_split() -> None:
    ptrs = [ChunkPointer(f"c{i}", f"d{i}.md", 1, 2, title=f"Title{i}") for i in range(12)]
    texts = [f"Document {i} about topic {i}." for i in range(12)]
    vecs = np.vstack([_basis(24, i) for i in range(12)])
    manifold = build_substate_manifold(ptrs, texts, vecs)

    # Single positive external FTS hit rescues c5 without penalizing absent chunks.
    q_zero = _basis(24, 0)
    res_single = run_substate_hmm(
        manifold,
        "topic",
        q_zero,
        fts_chunk_scores={"c5": 0.2},
        params=ManifoldHyperparams(external_trust=0.90),
    )
    ranks_single = {cid: idx for idx, (cid, _) in enumerate(res_single.ranked_chunks)}
    assert ranks_single["c5"] <= 1

    # Spiky high-SNR FTS hint on c7 beats uniform noisy vec_chunk_scores across all 12 chunks.
    q_vec = _unit(_basis(24, 0) + _basis(24, 7))
    res = run_substate_hmm(
        manifold,
        "topic",
        q_vec,
        fts_chunk_scores={"c7": 12.0, "c2": 1.0},
        vec_chunk_scores={f"c{i}": 0.5 for i in range(12)},
        params=ManifoldHyperparams(external_trust=0.75),
    )
    assert res.ranked_chunks[0][0] == "c7"


def test_multihop_maxnode_bridge_ast_siblings_and_popqa_distractor_trap() -> None:
    dim = 64
    rng = np.random.default_rng(101)
    n_popqa = 300
    ptrs = [ChunkPointer(f"d{i}", f"popqa_{i}.md", 1, 3, title=f"Entity{i}") for i in range(n_popqa)]
    texts = [f"Entity{i} is an ordinary item catalog entry number {i}." for i in range(n_popqa)]
    texts[0] = "Paris is the capital of France in Western Europe."

    vecs = np.asarray(rng.standard_normal((n_popqa, dim)), dtype=np.float32)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    vecs[0] = _basis(dim, 0)
    for i in range(1, n_popqa):
        vecs[i, 0] = 0.0
        vecs[i] = _unit(vecs[i])

    trap_manifold = build_substate_manifold(ptrs, texts, vecs)
    popqa_res = run_substate_hmm(
        trap_manifold,
        "What is the capital of France?",
        _basis(dim, 0),
        fts_chunk_scores={"d0": 9.5},
        top_k=5,
    )
    assert popqa_res.ranked_chunks[0][0] == "d0"
    assert all(w == 0.0 for w in popqa_res.hop_weights[1:])

    # Multi-hop Ego-kNN + spaCy bridge + AST siblings
    hop_ptrs = [
        ChunkPointer("m0", "shared.py", 1, 10, title="Apollo Program", ast_island_start=1, ast_island_end=15),
        ChunkPointer("m1", "other.py", 1, 10, title="Saturn V", ast_island_start=20, ast_island_end=30),
        ChunkPointer("m2", "shared.py", 11, 20, title="Launch Control", ast_island_start=5, ast_island_end=18),
    ] + [ChunkPointer(f"bg{i}", f"bg_{i}.md", 1, 2, title=f"Bg{i}") for i in range(25)]
    hop_texts = [
        "Apollo Program relied on Wernher Von Braun for rocket architecture.",
        "Saturn V was engineered by Wernher Von Braun at Marshall Space Flight Center.",
        "Launch Control verified cryogenic propellant valves prior to ignition.",
    ] + [f"Background filler article {i} on municipal zoning ordinances." for i in range(25)]

    hop_vecs = np.asarray(rng.standard_normal((len(hop_ptrs), dim)), dtype=np.float32)
    for i in range(len(hop_vecs)):
        hop_vecs[i, :4] = 0.0
        hop_vecs[i] = _unit(hop_vecs[i])
    hop_vecs[0] = _basis(dim, 0)
    hop_vecs[1] = _unit(0.55 * _basis(dim, 0) + 0.80 * _basis(dim, 1))
    hop_vecs[2] = _unit(0.35 * _basis(dim, 0) + 0.35 * _basis(dim, 2))

    hop_manifold = build_substate_manifold(hop_ptrs, hop_texts, hop_vecs)
    assert 2 in hop_manifold.ast_siblings[0].tolist()
    mh_res = run_substate_hmm(
        hop_manifold,
        "Where did the designer of the Apollo Program build the rocket?",
        _unit(0.8 * _basis(dim, 0) + 0.6 * _basis(dim, 1)),
        top_k=5,
    )
    assert any(w > 0.0 for w in mh_res.hop_weights[1:])
    assert "m1" in {cid for cid, _ in mh_res.ranked_chunks[:5]}


def test_multi_entity_comparison_join_and_acyclic_routes() -> None:
    dim = 48
    rng = np.random.default_rng(77)
    ptrs = [
        ChunkPointer("a", "a.md", 1, 3, title="Alan Turing"),
        ChunkPointer("b", "b.md", 1, 3, title="Alonzo Church"),
        ChunkPointer("bridge", "bridge.md", 1, 3, title="Lambda Calculus"),
    ] + [ChunkPointer(f"z{i}", f"z_{i}.md", 1, 2, title=f"Noise{i}") for i in range(20)]
    texts = [
        "Alan Turing studied computability and Lambda Calculus foundations in 1936.",
        "Alonzo Church invented Lambda Calculus at Princeton University.",
        "Lambda Calculus connects Alan Turing machines and Alonzo Church decidability proofs.",
    ] + [f"Botanical taxonomy note {i}." for i in range(20)]

    vecs = np.asarray(rng.standard_normal((len(ptrs), dim)), dtype=np.float32)
    for i in range(len(vecs)):
        vecs[i, :4] = 0.0
        vecs[i] = _unit(vecs[i])
    vecs[0] = _basis(dim, 0)
    vecs[1] = _basis(dim, 1)
    vecs[2] = _unit(0.7 * _basis(dim, 0) + 0.7 * _basis(dim, 1) + 0.2 * _basis(dim, 2))

    manifold = build_substate_manifold(ptrs, texts, vecs)
    res = run_substate_hmm(
        manifold,
        "Compare Alan Turing and Alonzo Church through Lambda Calculus.",
        _unit(0.6 * _basis(dim, 0) + 0.6 * _basis(dim, 1)),
        top_k=5,
        params=ManifoldHyperparams(coalition_coupling=0.90),
    )
    assert res.num_particles >= 2
    assert len(res.routes) >= 2
    top_ids = [cid for cid, _ in res.ranked_chunks[:4]]
    assert "a" in top_ids and "b" in top_ids and "bridge" in top_ids

    # Route ordering must be strictly acyclic
    seen_cids = [hop.chunk_id for hop in res.route]
    assert len(seen_cids) == len(set(seen_cids))
    dag: dict[str, set[str]] = {}
    for branch_route in res.routes:
        prev: str | None = None
        for hop in branch_route:
            dag.setdefault(hop.chunk_id, set())
            if prev is not None and prev != hop.chunk_id:
                dag[hop.chunk_id].add(prev)
            prev = hop.chunk_id
    list(TopologicalSorter(dag).static_order())


def test_redundancy_curvature_suppresses_clones_and_all_five_params_sensitive() -> None:
    dim = 32
    rng = np.random.default_rng(19)
    ptrs = [
        ChunkPointer("orig", "orig.md", 1, 2, title="Hubble Telescope"),
        ChunkPointer("clone", "clone.md", 1, 2, title="Hubble Duplicate"),
        ChunkPointer("diverse", "div.md", 1, 2, title="James Webb"),
    ] + [ChunkPointer(f"bg{i}", f"bg_{i}.md", 1, 2, title=f"Bg{i}") for i in range(18)]
    texts = [
        "Hubble Telescope captured deep field galaxies using Spitzer calibration.",
        "Hubble Telescope captured deep field galaxies using Spitzer calibration duplicate.",
        "James Webb relies on Spitzer infrared calibration mirrors and cryogenic shields.",
    ] + [f"Unrelated pottery shard note {i}." for i in range(18)]

    vecs = np.asarray(rng.standard_normal((len(ptrs), dim)), dtype=np.float32)
    for i in range(len(vecs)):
        vecs[i, :4] = 0.0
        vecs[i] = _unit(vecs[i])
    vecs[0] = _basis(dim, 0)
    vecs[1] = _unit(0.99 * _basis(dim, 0) + 0.05 * _basis(dim, 1))
    vecs[2] = _unit(0.75 * _basis(dim, 2) + 0.35 * _basis(dim, 0))

    manifold = build_substate_manifold(ptrs, texts, vecs)
    q_vec = _unit(0.85 * _basis(dim, 0) + 0.30 * _basis(dim, 2))

    low_curv = run_substate_hmm(
        manifold,
        "Hubble Telescope Spitzer",
        q_vec,
        top_k=3,
        params=ManifoldHyperparams(redundancy_curvature=0.0),
    )
    high_curv = run_substate_hmm(
        manifold,
        "Hubble Telescope Spitzer",
        q_vec,
        top_k=3,
        params=ManifoldHyperparams(redundancy_curvature=1.5),
    )
    assert high_curv.ranked_chunks[1][0] == "diverse"
    assert high_curv.ranked_chunks != low_curv.ranked_chunks

    # Verify all 5 hyperparameters produce measurable output differences when varied.
    base_res = run_substate_hmm(
        manifold,
        "Hubble Telescope and James Webb Spitzer",
        q_vec,
        top_k=5,
        fts_chunk_scores={"diverse": 5.0},
        vec_chunk_scores={"orig": 0.9},
        params=ManifoldHyperparams(),
    )
    alt_configs = [
        ManifoldHyperparams(peclet=0.25),
        ManifoldHyperparams(alpha_sprt=0.005),
        ManifoldHyperparams(redundancy_curvature=1.50),
        ManifoldHyperparams(coalition_coupling=0.95),
        ManifoldHyperparams(external_trust=0.95),
    ]
    base_vec = base_res.chunk_posteriors
    for alt in alt_configs:
        if alt.alpha_sprt != 0.05:
            # Verify alpha_sprt toggles a borderline hop gate near tau_z.
            n_b = 40
            b_ptrs = [ChunkPointer(f"b{i}", f"b_{i}.md", 1, 2, title=f"B{i}") for i in range(n_b)]
            b_texts = [f"Neutral record {i}." for i in range(n_b)]
            b_texts[0] = "SeedEntity connects moderately to BorderTarget."
            b_texts[1] = "BorderTarget note."
            b_vecs = np.asarray(rng.standard_normal((n_b, dim)), dtype=np.float32)
            for i in range(n_b):
                b_vecs[i, :2] = 0.0
                b_vecs[i] = _unit(b_vecs[i])
            b_vecs[0] = _basis(dim, 0)
            b_vecs[1] = _unit(0.45 * _basis(dim, 0) + 0.89 * _basis(dim, 1))
            b_man = build_substate_manifold(b_ptrs, b_texts, b_vecs)
            tight = run_substate_hmm(b_man, "SeedEntity", _basis(dim, 0), params=ManifoldHyperparams(alpha_sprt=0.001))
            loose = run_substate_hmm(b_man, "SeedEntity", _basis(dim, 0), params=ManifoldHyperparams(alpha_sprt=0.45))
            assert (
                tight.hop_weights != loose.hop_weights
                or float(np.max(np.abs(tight.chunk_posteriors - loose.chunk_posteriors))) > 1e-4
            )
            continue
        alt_res = run_substate_hmm(
            manifold,
            "Hubble Telescope and James Webb Spitzer",
            q_vec,
            top_k=5,
            fts_chunk_scores={"diverse": 5.0},
            vec_chunk_scores={"orig": 0.9},
            params=alt,
        )
        max_diff = float(np.max(np.abs(alt_res.chunk_posteriors - base_vec)))
        assert (
            max_diff > 1e-4
            or alt_res.hop_weights != base_res.hop_weights
            or alt_res.ranked_chunks != base_res.ranked_chunks
        )


def test_multi_branch_frontier_exceeding_eight_retains_all_spacy_bridges() -> None:
    """Frontiers spanning >8 unique chunks across 3 branches never truncate spaCy bridge entities."""
    dim, n_c = 32, 60
    rng = np.random.default_rng(7)
    ptrs = [ChunkPointer(f"c{i}", f"c_{i}.md", 1, 2, title=f"Doc{i}") for i in range(n_c)]
    texts = [f"Unrelated filler note {i} on botany." for i in range(n_c)]
    # Realistic background dispersion across dims 0..14 so _robust_z MAD stays non-degenerate.
    vecs = 0.45 * rng.standard_normal((n_c, dim)).astype(np.float32)
    vecs[10:, :3] = np.clip(vecs[10:, :3], -1.0, 0.25)
    vecs[:, 16:] += 0.55 * rng.standard_normal((n_c, 16)).astype(np.float32)
    vecs[:, 15] = 0.0
    for i in range(n_c):
        vecs[i] = _unit(vecs[i])

    # 3 orthogonal query anchor branches: each has 2 Hop-1 frontier children (3 + 6 = 9 unique frontier nodes).
    texts[0] = "AlphaStation primary observatory report."
    texts[1] = "BetaHarbor coastal station log."
    texts[2] = "GammaRidge alpine telemetry post."
    texts[3] = "AlphaStation annex A with CobaltPeak reference."
    texts[4] = "AlphaStation annex B with QuartzValley reference."
    texts[5] = "BetaHarbor pier A with AmberCove reference."
    texts[6] = "BetaHarbor pier B with SlateBay reference."
    texts[7] = "GammaRidge camp A with IronCrest reference."
    # Chunk 8 (9th unique frontier chunk) carries a lexical spaCy PROPN bridge to isolated Target9th (c9).
    texts[8] = "GammaRidge camp B forwarded logs from ZetaBridge Observatory yesterday."
    texts[9] = "Confidential archive inside ZetaBridge Observatory revealed final proof."

    # Anchor c0..c2 aligned cleanly to query orthogonal directions [0..2]; children c3..c8 aligned to parent.
    for s_idx in range(3):
        vecs[s_idx] = _basis(dim, s_idx)
        vecs[3 + 2 * s_idx] = _unit(0.60 * _basis(dim, s_idx) + 0.80 * _basis(dim, 6 + 2 * s_idx))
        vecs[4 + 2 * s_idx] = _unit(0.55 * _basis(dim, s_idx) + 0.835 * _basis(dim, 7 + 2 * s_idx))
    # c9 has zero dense similarity to query or c0..c8 (only reachable via c8's Hop-2 spaCy lexical bridge).
    vecs[9] = _basis(dim, 15)

    manifold = build_substate_manifold(ptrs, texts, vecs)
    q_vec = _unit(_basis(dim, 0) + _basis(dim, 1) + _basis(dim, 2))
    res = run_substate_hmm(manifold, "Compare AlphaStation, BetaHarbor, and GammaRidge.", q_vec, top_k=12)

    ranked_ids = [cid for cid, _ in res.ranked_chunks]
    assert "c9" in ranked_ids
    assert any(h.chunk_id == "c9" and "ZetaBridge" in h.bridge for h in res.route)

    # Counterfactual check: dropping c8's precomputed spaCy entities excludes c9 from both ranking and route.
    trunc_ents = list(manifold.chunk_entities)
    trunc_ents[8] = ([], [])
    res_trunc = run_substate_hmm(
        replace(manifold, chunk_entities=tuple(trunc_ents)),
        "Compare AlphaStation, BetaHarbor, and GammaRidge.",
        q_vec,
        top_k=12,
    )
    assert "c9" not in [cid for cid, _ in res_trunc.ranked_chunks]
    assert not any(h.chunk_id == "c9" for h in res_trunc.route)
