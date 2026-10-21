"""Unmocked integration and invariant tests for YawGraphEngine with real SQLite and disk files."""

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import HippoRAGProgram


class DeterministicMockLM(dspy.LM):
    """Deterministic offline LM for testing HippoRAGProgram and YawGraphEngine."""

    def __init__(self) -> None:
        super().__init__(model="mock-lm")

    def __call__(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> list[str]:
        if getattr(dspy.settings, "usage_tracker", None) is not None:
            dspy.settings.usage_tracker.add_usage(
                self.model,
                {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            )
        req_str = str(prompt or messages or kwargs)
        if (
            "JudgeSynonyms" in req_str
            or "OrchestrateSynonymJudgment" in req_str
            or "candidate_pairs_json" in req_str
            or "candidate_pairs" in req_str
        ):
            return ['{"decisions": []}']
        if (
            "ProposeSHACLRepairs" in req_str
            or "HealSHACLViolations" in req_str
            or "violations_json" in req_str
        ):
            return ['{"repairs": []}']
        if (
            "DefineMissingEntities" in req_str
            or "OrchestrateMissingEntityDefinitions" in req_str
            or "missing_entities" in req_str
            or "entity_definitions" in req_str
        ):
            if "Lovelace" in req_str:
                return [
                    '{"entity_definitions": [["Ada Lovelace", "English mathematician"], ["Analytical Engine", "Mechanical general-purpose computer"]]}'
                ]
            return [
                '{"entity_definitions": [["Alan Turing", "English mathematician and computer scientist"], ["computer science", "Study of computation and information"]]}'
            ]
        # Filter triples stage
        if "FilterTriples" in req_str or "fact_before_filter" in req_str:
            if "Lovelace" in req_str:
                return ['{"fact": [["Ada Lovelace", "worked on", "Analytical Engine"]]}']
            return ['{"fact": [["Alan Turing", "father of", "computer science"]]}']
        # Triples stage
        if "ExtractTriples" in req_str or '"triples"' in req_str or "`triples`" in req_str:
            if "Lovelace" in req_str:
                return [
                    '{"triples": [{"sub": "Ada Lovelace", "pred": "worked on", "obj": "Analytical Engine", "def": "Ada Lovelace worked on the Analytical Engine."}]}'
                ]
            return [
                '{"triples": [{"sub": "Alan Turing", "pred": "father of", "obj": "computer science", "def": "Alan Turing is the father of computer science."}]}'
            ]
        # NER stage
        if "ExtractEntities" in req_str or "named_entities" in req_str:
            if "Lovelace" in req_str:
                return [
                    '{"named_entities": [["Ada Lovelace", "English mathematician"], ["Analytical Engine", "Mechanical general-purpose computer"]]}'
                ]
            return [
                '{"named_entities": [["Alan Turing", "English mathematician and computer scientist"], ["computer science", "Study of computation and information"]]}'
            ]
        # QA stage
        if (
            "AnswerQuestion" in req_str
            or "Retrieved evidence passages" in req_str
            or "context" in req_str
        ):
            if "Turing" in req_str:
                return [
                    '{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}'
                ]
            if "Lovelace" in req_str:
                return [
                    '{"reasoning": "Ada Lovelace worked on the Analytical Engine.", "answer": "Ada Lovelace"}'
                ]
            return [
                '{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}'
            ]
        return ['{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}']


class DeterministicMockEmbedder:
    """Deterministic offline embedder producing 64-dimensional float32 unit vectors and emitting telemetry."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.telemetry: list[dict[str, Any]] = []
        self._telemetry_hook: Any = None

    def bind_telemetry_hook(self, hook: Any) -> None:
        self._telemetry_hook = hook

    def reset_telemetry(self) -> None:
        self.telemetry.clear()

    def encode(
        self,
        texts: Sequence[str],
        instruction: str = "",
        *,
        stage: str | None = None,
    ) -> list[np.ndarray]:
        results: list[np.ndarray] = []
        for t in texts:
            combined = f"{instruction}:{t}"
            seed = int(hashlib.md5(combined.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            results.append(vec)
        total_chars = sum(len(t) for t in texts)
        tokens_est = max(1, (total_chars + 3) // 4) if texts else 0
        entry = {
            "stage": stage or "embed",
            "predictor": "embedder",
            "model": "mock-embedder",
            "mode": "embed",
            "num_inputs": len(texts),
            "tokens_used": tokens_est,
        }
        self.telemetry.append(entry)
        if self._telemetry_hook is not None:
            self._telemetry_hook(entry)
        return results


@pytest.fixture
def test_engine(tmp_path: Path) -> Iterator[YawGraphEngine]:
    db_path = tmp_path / "yaw.db"
    mock_lm = DeterministicMockLM()
    dspy.settings.configure(lm=mock_lm)

    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=db_path,
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    yield engine
    engine.close()


def test_engine_index_and_caching(test_engine: YawGraphEngine, tmp_path: Path) -> None:
    docs = [
        "Alan Turing was an English mathematician and computer scientist.\nHe broke German ciphers.",
        "Radio City operates FM radio stations across India.\nIt broadcasts in Mumbai.",
    ]

    # First index pass
    indexed_count = test_engine.index(docs)
    assert indexed_count == 2

    # Verify passages.txt written on disk
    passages_file = tmp_path / "corpus" / "passages.txt"
    assert passages_file.exists()
    content = passages_file.read_text(encoding="utf-8")
    assert "Alan Turing" in content
    assert "Radio City" in content

    # Verify SQLite tables populated
    chunks = test_engine.conn.execute(
        "SELECT chunk_id, start_line, end_line FROM chunks"
    ).fetchall()
    assert len(chunks) == 2

    # Check FTS5 populated
    fts_hits = test_engine.conn.execute(
        "SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH 'Turing'"
    ).fetchall()
    assert len(fts_hits) == 1

    # Check openie_chunks populated
    openie_rows = test_engine.conn.execute("SELECT chunk_id FROM openie_chunks").fetchall()
    assert len(openie_rows) == 2

    # Check entities and facts
    entities = test_engine.conn.execute("SELECT entity_id, name FROM entities").fetchall()
    assert len(entities) >= 2

    # Second index pass (idempotent re-index)
    reindexed = test_engine.index(docs)
    assert reindexed == 0


def test_engine_retrieve_and_rag_qa(test_engine: YawGraphEngine) -> None:
    docs = [
        "Alan Turing was an English mathematician and computer scientist.\nHe formulated the Turing machine.",
        "Ada Lovelace was a mathematician recognized for work on the Analytical Engine.",
    ]
    test_engine.index(docs)

    # Retrieval
    hits = test_engine.retrieve("Who formulated the Turing machine?", top_k=2)
    assert len(hits) == 2
    top_ptr, top_score, top_text = hits[0]
    assert isinstance(top_ptr, ChunkPointer)
    assert isinstance(top_score, float)
    assert "Alan Turing" in top_text

    # RAG QA
    qa_results = test_engine.rag_qa(
        ["Who formulated the Turing machine?"],
        gold_answers=["Alan Turing"],
    )
    assert len(qa_results) == 1
    res = qa_results[0]
    assert res["question"] == "Who formulated the Turing machine?"
    assert res["answer"] == "Alan Turing"
    assert len(res["retrieved_chunk_ids"]) == 2
    assert "Alan Turing" in res["context"]


def test_engine_end_to_end_trace_hook(test_engine: YawGraphEngine) -> None:
    docs = [
        "Alan Turing was an English mathematician and computer scientist.\nHe designed the ACE.",
    ]
    test_engine.index(docs)

    # Execute HippoRAGProgram.forward with only the question argument to exercise retrieval hook
    pred = test_engine.program(question="Who designed the ACE?")

    assert hasattr(pred, "answer")
    assert getattr(pred, "answer") == "Alan Turing"
    assert hasattr(pred, "context")
    assert "Alan Turing" in getattr(pred, "context")
    assert hasattr(pred, "retrieved_passages")
    assert len(getattr(pred, "retrieved_passages")) > 0


def test_engine_delete_chunks_cascades_and_invalidates_csr(test_engine: YawGraphEngine) -> None:
    docs = [
        "Chunk 1: Alan Turing.",
        "Chunk 2: Radio City.",
    ]
    test_engine.index(docs)

    # Ensure CSR graph is cached
    graph1 = test_engine._ensure_csr_graph()
    assert len(graph1.chunk_indices) == 2

    # Delete chunk 1
    c1_id = graph1.chunk_ids[0]
    test_engine.delete_chunks([c1_id])

    # Cached graph must be invalidated
    assert test_engine._csr_graph is None

    # Rebuilding CSR graph should now contain only 1 chunk
    graph2 = test_engine._ensure_csr_graph()
    assert len(graph2.chunk_indices) == 1
    assert c1_id not in graph2.chunk_ids


def test_engine_multi_batch_indexing_and_bounded_embedding_batches(tmp_path: Path) -> None:
    batch_sizes_seen: list[int] = []

    class TrackingEmbedder(DeterministicMockEmbedder):
        def encode(
            self,
            texts: Sequence[str],
            instruction: str = "",
            *,
            stage: str | None = None,
        ) -> list[np.ndarray]:
            batch_sizes_seen.append(len(texts))
            return super().encode(texts, instruction=instruction, stage=stage)

    mock_lm = DeterministicMockLM()
    dspy.settings.configure(lm=mock_lm)
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = TrackingEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_batches.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        embed_batch_size=2,
    )
    try:
        batch1 = [
            "Doc 1 line 1\nDoc 1 line 2",
            "Doc 2 line 1\nDoc 2 line 2\nDoc 2 line 3",
            "Doc 3 single line",
        ]
        assert engine.index(batch1) == 3

        batch2 = [
            "Doc 4 line 1\nDoc 4 line 2",
            "Doc 5 single line",
        ]
        assert engine.index(batch2) == 2

        # Every embedder call must respect embed_batch_size=2
        assert all(sz <= 2 for sz in batch_sizes_seen)

        # Verify line numbers in SQLite match exact lines in passages.txt (2 + 3 + 1 + 2 + 1 = 9 lines)
        max_end = engine.conn.execute("SELECT MAX(end_line) FROM chunks").fetchone()[0]
        assert max_end == 9

        # Verify DPR retrieval hydrates chunks from both batches accurately
        dpr_hits = engine.retrieve_dpr("Doc 4 line 1", top_k=5)
        assert len(dpr_hits) == 5
        hydrated_texts = {text for _, _, text in dpr_hits}
        assert "Doc 4 line 1\nDoc 4 line 2" in hydrated_texts
        assert "Doc 1 line 1\nDoc 1 line 2" in hydrated_texts
    finally:
        engine.close()


def test_engine_fact_score_weighting_and_thread_safe_retrieval(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    class ControlledFactEmbedder:
        """Assigns high similarity to Turing fact and low similarity to Lovelace fact."""

        def encode(self, texts: Sequence[str], instruction: str = "") -> list[np.ndarray]:
            vecs: list[np.ndarray] = []
            for t in texts:
                tl = t.lower()
                if "turing" in tl or "query_target" in tl:
                    vecs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                elif "lovelace" in tl:
                    vecs.append(np.array([0.2, 0.9798, 0.0, 0.0], dtype=np.float32))
                else:
                    vecs.append(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
            return vecs

    class KeepAllFactsLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            req_str = str(prompt or messages or kwargs)
            if "FilterTriples" in req_str or "fact_before_filter" in req_str:
                return [
                    '{"fact": [["alan turing", "father of", "computer science"], '
                    '["ada lovelace", "worked on", "analytical engine"]]}'
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=KeepAllFactsLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = ControlledFactEmbedder()

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_scores.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        passage_node_weight=0.0,  # Isolate entity seed PPR contribution
    )
    try:
        engine.index(
            [
                "Alan Turing founded computer science.",
                "Ada Lovelace worked on the Analytical Engine.",
            ]
        )

        hits, kept_facts = engine.retrieve_with_seeds("query_target", top_k=2)
        assert len(kept_facts) == 2
        assert len(hits) == 2
        # Because Turing fact has higher normalized similarity score than Lovelace fact,
        # the Turing passage must strictly outrank the Lovelace passage under PPR.
        assert "Alan Turing" in hits[0][2]
        assert hits[0][1] > hits[1][1]

        # Concurrent multi-threaded retrieval must succeed without sqlite3 ProgrammingError
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(engine.retrieve, "query_target", 2) for _ in range(8)]
            for fut in futures:
                thread_hits = fut.result()
                assert len(thread_hits) == 2
                assert "Alan Turing" in thread_hits[0][2]
    finally:
        engine.close()


def test_engine_evolve_prompts_miprov2_and_gepa_metric_contracts(
    test_engine: YawGraphEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_engine.index(["Alan Turing designed the ACE."])
    trainset = [
        dspy.Example(
            question="Who designed the ACE?",
            gold_answers=["Alan Turing"],
            gold_docs=["Alan Turing designed the ACE."],
        ).with_inputs("question")
    ]

    mipro_scores: list[Any] = []
    gepa_scores: list[Any] = []

    class FakeMIPROv2:
        def __init__(
            self,
            metric: Any,
            prompt_model: Any = None,
            task_model: Any = None,
            auto: str = "light",
            **kwargs: Any,
        ) -> None:
            self.metric = metric
            self.prompt_model = prompt_model
            self.task_model = task_model

        def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
            pred = student(question=trainset[0].question)
            res = self.metric(trainset[0], pred)
            mipro_scores.append(res)
            # MIPROv2 requires scalar float comparison
            assert isinstance(res, float)
            assert res > -1.0
            return student

    class FakeGEPA:
        def __init__(
            self,
            metric: Any,
            reflection_lm: Any = None,
            instruction_proposer: Any = None,
            auto: str = "light",
            track_stats: bool = False,
            track_best_outputs: bool = False,
            log_dir: str | None = None,
            **kwargs: Any,
        ) -> None:
            self.metric = metric
            self.reflection_lm = reflection_lm
            self.instruction_proposer = instruction_proposer
            self.auto = auto
            self.track_stats = track_stats
            self.track_best_outputs = track_best_outputs
            self.kwargs = kwargs

        def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
            assert self.instruction_proposer is not None
            assert callable(self.instruction_proposer)
            assert self.track_stats is True
            assert self.kwargs.get("component_selector") == "round_robin" or callable(
                self.kwargs.get("component_selector")
            )
            gepa_kw = self.kwargs.get("gepa_kwargs") or {}
            assert gepa_kw.get("frontier_type") == "hybrid"
            assert gepa_kw.get("use_cloudpickle") is True
            pred = student(question=trainset[0].question)
            res = self.metric(trainset[0], pred)
            gepa_scores.append(res)
            # GEPA requires Prediction(score=float, feedback=str)
            assert hasattr(res, "score") and isinstance(res.score, float)
            assert hasattr(res, "feedback") and isinstance(res.feedback, str)
            return student

    monkeypatch.setattr(dspy, "MIPROv2", FakeMIPROv2, raising=False)
    monkeypatch.setattr(dspy, "GEPA", FakeGEPA, raising=False)

    test_engine.evolve_prompts(trainset, mode="hybrid")
    assert len(mipro_scores) == 1
    assert len(gepa_scores) == 1


def test_engine_hybrid_fact_candidate_retrieval(tmp_path: Path) -> None:
    class HybridFactEmbedder:
        """Deterministic embedder: query is close to Turing in dense space."""

        def encode(self, texts: Sequence[str], instruction: str = "") -> list[np.ndarray]:
            vecs: list[np.ndarray] = []
            for t in texts:
                tl = t.lower()
                # Query has vector close to Turing, far from Lovelace
                if "query_hybrid" in tl or "turing" in tl:
                    vecs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                elif "lovelace" in tl:
                    vecs.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
                else:
                    vecs.append(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
            return vecs

    class InspectFilterLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            req_str = str(prompt or messages or kwargs)
            if "FilterTriples" in req_str or "fact_before_filter" in req_str:
                return [
                    '{"fact": [["ada lovelace", "worked on", "analytical engine"], '
                    '["alan turing", "father of", "computer science"]]}'
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=InspectFilterLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = HybridFactEmbedder()

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_hybrid.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        fact_candidate_pool_k=10,
        fact_fts_pool_k=10,
        fact_rrf_k=60,
        linking_top_k=2,
    )
    try:
        engine.index(
            [
                "Alan Turing was an English mathematician and father of computer science.",
                "Ada Lovelace worked on the Analytical Engine.",
            ]
        )

        # Query "query_hybrid Analytical"
        # - Vector similarity favors Turing (cosine = 1.0 vs Lovelace = 0.0)
        # - Lexical FTS5 BM25 matches "Analytical" in Lovelace triple
        # - RRF merges both: Turing gets dense rank 1, Lovelace gets FTS rank 1
        # - Both candidates enter pool and are forwarded to Recognition Memory
        hits, kept_facts = engine.retrieve_with_seeds("query_hybrid Analytical", top_k=2)

        # Verify kept facts are returned from both branches
        assert len(kept_facts) >= 1
        kept_subjs = {f[0].lower() for f in kept_facts}
        assert "alan turing" in kept_subjs or "ada lovelace" in kept_subjs

        # Verify hits are hydrated correctly
        assert len(hits) == 2
    finally:
        engine.close()


def test_engine_query_intent_escalation_once_without_infinite_recursion(tmp_path: Path) -> None:
    intent_calls: list[str] = []
    filter_calls: list[str] = []

    class IntentEscalationLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            req_str = str(prompt or messages or kwargs)
            if "ExtractQueryIntent" in req_str or "query_intent" in req_str:
                intent_calls.append(req_str)
                if "never_matches" in req_str:
                    return ['{"query_intent": "still_never_matches"}']
                return ['{"query_intent": "Ada Lovelace worked on Analytical Engine"}']
            if "FilterTriples" in req_str or "fact_before_filter" in req_str:
                filter_calls.append(req_str)
                # Only keep facts when the rewritten intent includes 'Ada Lovelace'
                if "Ada Lovelace worked on Analytical Engine" in req_str:
                    return ['{"fact": [["ada lovelace", "worked on", "analytical engine"]]}']
                return ['{"fact": []}']
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=IntentEscalationLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_intent.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        engine.index(
            [
                "Alan Turing was an English mathematician and father of computer science.",
                "Ada Lovelace worked on the Analytical Engine.",
            ]
        )

        # 1. Happy path: initial query keeps 0 facts -> escalates once -> succeeds on intent
        hits, kept_facts = engine.retrieve_with_seeds("Who built the early engine?", top_k=2)
        assert len(intent_calls) == 1
        assert len(filter_calls) == 2
        assert len(kept_facts) == 1
        assert kept_facts[0][0].lower() == "ada lovelace"
        assert len(hits) == 2

        # 2. Boundary envelope: when intent also keeps 0 facts, stop trying to reword (no infinite loop)
        intent_calls.clear()
        filter_calls.clear()
        hits_fallback, kept_empty = engine.retrieve_with_seeds("never_matches query", top_k=2)
        assert len(intent_calls) == 1
        assert len(filter_calls) == 2
        assert kept_empty == []
        assert len(hits_fallback) == 2
    finally:
        engine.close()


def test_engine_homograph_gate_edc_and_shacl_conformance(tmp_path: Path) -> None:
    """Verify Same-Label Homograph Gate splits conflicting senses via JudgeSynonyms and passes SHACL linting."""
    import json
    import re

    class HomographGateLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
                )
            req_str = str(prompt or messages or kwargs)
            if "JudgeSynonyms" in req_str or "candidate_pairs" in req_str:
                pair_ids = list(
                    dict.fromkeys(re.findall(r"entity-[0-9a-f]+\|entity-[0-9a-f]+", req_str))
                )
                decisions = [
                    {
                        "pair_id": pid,
                        "relation": "owl:differentFrom",
                        "canonical": pid.split("|", 1)[0],
                    }
                    for pid in pair_ids
                ]
                return [json.dumps({"decisions": decisions})]
            if "ExtractTriples" in req_str or "triples" in req_str:
                if "Cupertino" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    {
                                        "subj": "Apple",
                                        "subj_def": "Technology company headquartered in Cupertino",
                                        "pred": "headquartered in",
                                        "obj": "Cupertino",
                                        "obj_def": "City in Silicon Valley, California",
                                        "spo_def": "Apple is a technology company headquartered in Cupertino.",
                                    }
                                ]
                            }
                        )
                    ]
                if "orchard" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    {
                                        "subj": "Apple",
                                        "subj_def": "Edible pome fruit harvested from orchard trees",
                                        "pred": "harvested in",
                                        "obj": "Autumn",
                                        "obj_def": "Temperate harvest season",
                                        "spo_def": "Apple fruit is harvested in autumn from orchard trees.",
                                    }
                                ]
                            }
                        )
                    ]
            if "ExtractEntities" in req_str or "named_entities" in req_str:
                if "Cupertino" in req_str:
                    return [
                        json.dumps(
                            {
                                "named_entities": [
                                    ["Apple", "Technology company headquartered in Cupertino"],
                                    ["Cupertino", "City in Silicon Valley, California"],
                                ]
                            }
                        )
                    ]
                if "orchard" in req_str:
                    return [
                        json.dumps(
                            {
                                "named_entities": [
                                    ["Apple", "Edible pome fruit harvested from orchard trees"],
                                    ["Autumn", "Temperate harvest season"],
                                ]
                            }
                        )
                    ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=HomographGateLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_homograph.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        # Index first passage: creates primary "Apple" entity (tech company)
        n1 = engine.index(["Apple is a technology company headquartered in Cupertino."])
        assert n1 == 1

        # Index second passage: "Apple" with different inline definition triggers provisional sense split
        n2 = engine.index(
            ["The apple is an edible pome fruit harvested in orchard trees during Autumn."]
        )
        assert n2 == 1

        # Verify two distinct Apple entities exist, linked by owl:differentFrom and sharing prefLabel "apple"
        apple_entities = engine.conn.execute(
            "SELECT entity_id, name, definition FROM entities WHERE LOWER(name) LIKE 'apple%';"
        ).fetchall()
        assert len(apple_entities) == 2

        diff_edges = engine.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'owl:differentFrom';"
        ).fetchone()[0]
        assert diff_edges >= 2

        report = engine.lint_graph(as_turtle=True)
        assert report["conforms"] is True
        assert report["violations"] == []
        ttl = report["turtle"]
        assert "sh:conforms true" in ttl
        assert "owl:differentFrom" in ttl
    finally:
        engine.close()


def test_engine_same_label_provisional_split_merged_by_judge_synonyms_and_stage1_standalone_ner(
    tmp_path: Path,
) -> None:
    """Verify Hazard 2 (same-label cos < 0.85 merged back when JudgeSynonyms returns skos:altLabel) and Hazard 3 Option A (standalone Stage 1 NER entities preserved with definitions)."""
    import json
    import re

    class SynonymMergeAndStandaloneNerLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
                )
            req_str = str(prompt or messages or kwargs)
            if "JudgeSynonyms" in req_str or "candidate_pairs" in req_str:
                pair_ids = list(
                    dict.fromkeys(re.findall(r"entity-[0-9a-f]+\|entity-[0-9a-f]+", req_str))
                )
                decisions = []
                for pid in pair_ids:
                    ea, eb = pid.split("|", 1)
                    # Merge same-label provisional sense back into canonical entity via skos:altLabel
                    decisions.append(
                        {
                            "pair_id": pid,
                            "relation": "skos:altLabel",
                            "canonical": ea,
                        }
                    )
                return [json.dumps({"decisions": decisions})]
            if "ExtractTriples" in req_str or "triples" in req_str:
                if "iPhone" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    ["Apple", "designed", "iPhone"],
                                ]
                            }
                        )
                    ]
                if "MacBook" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    ["Apple", "manufactures", "MacBook"],
                                ]
                            }
                        )
                    ]
            if "ExtractEntities" in req_str or "named_entities" in req_str:
                if "iPhone" in req_str:
                    return [
                        json.dumps(
                            {
                                "named_entities": [
                                    ["Apple", "Consumer electronics maker founded by Steve Jobs"],
                                    ["iPhone", "Flagship smartphone line designed by Apple"],
                                    ["2007", "Year the original iPhone was unveiled"],
                                ]
                            }
                        )
                    ]
                if "MacBook" in req_str:
                    return [
                        json.dumps(
                            {
                                "named_entities": [
                                    [
                                        "Apple",
                                        "Multinational personal computer corporation in California",
                                    ],
                                    ["MacBook", "Line of Macintosh portable computers"],
                                    ["M3 Max", "System on a chip powering MacBook Pro laptops"],
                                ]
                            }
                        )
                    ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=SynonymMergeAndStandaloneNerLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_hazard2_3.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
        synonymy_threshold=0.99,  # Ensure only same-label candidate pair is surfaced
    )
    try:
        engine.index(["Apple designed the iPhone in 2007."])
        engine.index(["Apple manufactures the MacBook with the M3 Max chip."])

        # Hazard 2: Provisional sense split for "Apple" (since mock embedder cos < 0.85 for different defs)
        # had NO premature owl:differentFrom edge in Step 4 and was merged back into 1 Apple entity by JudgeSynonyms!
        apple_rows = engine.conn.execute(
            "SELECT entity_id, name, definition FROM entities WHERE name = 'apple'"
        ).fetchall()
        assert len(apple_rows) == 1
        diff_edges = engine.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'owl:differentFrom'"
        ).fetchone()[0]
        assert diff_edges == 0

        # Hazard 3 Option A: Standalone Stage 1 NER entities ("2007" and "m3 max") that did NOT appear in any triple
        # are preserved in entities with their Stage 1 definitions and linked via passage edges!
        standalone_rows = {
            str(r[0]): str(r[1])
            for r in engine.conn.execute(
                "SELECT name, definition FROM entities WHERE name IN ('2007', 'm3 max')"
            ).fetchall()
        }
        assert "2007" in standalone_rows
        assert "Year the original iPhone was unveiled" in standalone_rows["2007"]
        assert "m3 max" in standalone_rows
        assert "System on a chip powering MacBook Pro laptops" in standalone_rows["m3 max"]

        passage_edges_for_standalone = engine.conn.execute(
            """
            SELECT COUNT(*) FROM edges ed
            INNER JOIN entities e ON ed.src_id = e.entity_id
            WHERE ed.kind = 'passage' AND e.name IN ('2007', 'm3 max')
            """
        ).fetchone()[0]
        assert passage_edges_for_standalone == 2

        # Verify 0 SHACL violations (including ConceptShape/S13-disjointLabels across both chunks)
        assert engine._graph_rlm_stats["synonym_judge_tokens"] == 48
        assert engine._graph_rlm_stats["embedding_calls"] > 0
        assert engine._graph_rlm_stats["embedding_tokens"] > 0
        assert len(program.embedding_audit_trail) > 0
        report = engine.lint_graph()
        assert report["conforms"] is True
        assert report["violations"] == []
    finally:
        engine.close()


def test_engine_step7_heal_shacl_violations_rlm_repl_tools_and_token_stats(
    tmp_path: Path,
) -> None:
    """Verify Step 6 direct GBNF synonym_judge_tokens and Step 7 heal_shacl_violations via shacl_healer_rlm using all 5 bound REPL tools."""
    import json
    import re

    repl_tool_events: list[str] = []

    class RLMOrchestratorLM(dspy.LM):
        """RLM orchestrator LM (40 tokens per call, ChatAdapter, no GBNF)."""

        def __init__(self) -> None:
            super().__init__(model="qwen-rlm-orchestrator")
            self._yaw_recipe = "llamacpp"
            self._yaw_supports_gbnf = False

        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
                )
            req_str = str(prompt or messages or kwargs)
            if "qwen_propose_shacl_repairs" in req_str:
                repl_tool_events.append("shacl_healer_rlm_invoked")
                code = (
                    "import json\n"
                    "v_items = json.loads(violations)\n"
                    "focus_id = v_items[0]['focus_node']\n"
                    "node_ctx = inspect_focus_node(focus_id)\n"
                    "chunk_text = hydrate_chunk(node_ctx['chunk_ids'][0])\n"
                    "assert 'Alan Turing formulated the Turing machine' in chunk_text\n"
                    "repairs = qwen_propose_shacl_repairs(violations=violations, focus_context=chunk_text)\n"
                    "applied_first = apply_shacl_repairs([repairs[0]])\n"
                    "applied_dup = apply_shacl_repairs([repairs[0]])\n"
                    "remaining_mid = lint_shacl()\n"
                    "assert applied_first == 1 and applied_dup == 0 and len(remaining_mid) == 1\n"
                    "SUBMIT(repairs=repairs)"
                )
                return [
                    f"[[ ## reasoning ## ]]\nInspect focus node, hydrate chunk, propose/apply repairs, and lint.\n\n"
                    f"[[ ## code ## ]]\n```python\n{code}\n```\n\n[[ ## completed ## ]]"
                ]
            return ['{"repairs": []}']

    class QwenWorkerLM(dspy.LM):
        """GBNF-constrained GPU worker LM (30 tokens per call, JSONAdapter)."""

        def __init__(self) -> None:
            super().__init__(model="qwen-worker")
            self._yaw_recipe = "llamacpp"
            self._yaw_supports_gbnf = True

        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                )
            req_str = str(prompt or messages or kwargs)
            if (
                "[[ ## violations ## ]]" in req_str
                or "bounded closed-world SHACL repair" in req_str
            ):
                focus_matches = list(dict.fromkeys(re.findall(r"entity-[0-9a-f]+", req_str)))
                for fn in focus_matches:
                    repl_tool_events.append(f"qwen_propose_shacl_repairs:{fn}")
                return [
                    json.dumps(
                        {
                            "repairs": [
                                {
                                    "focus_node": fn,
                                    "action": "set_definition",
                                    "value": "Healed concept definition from Alan Turing formulated the Turing machine in 1936.",
                                }
                                for fn in focus_matches
                            ]
                        }
                    )
                ]
            if "[[ ## candidate_pairs ## ]]" in req_str or "candidate_pairs" in req_str:
                pair_ids = list(
                    dict.fromkeys(re.findall(r"entity-[0-9a-f]+\|entity-[0-9a-f]+", req_str))
                )
                decisions = [
                    {
                        "pair_id": pid,
                        "relation": "skos:related",
                        "canonical": pid.split("|", 1)[0],
                    }
                    for pid in pair_ids
                ]
                return [json.dumps({"decisions": decisions})]
            if (
                "[[ ## named_entities ## ]]" in req_str
                or "Resource Description Framework" in req_str
            ):
                if "Bletchley" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    {
                                        "subj": "Alan M. Turing",
                                        "subj_def": "British cryptanalyst at Bletchley Park",
                                        "pred": "worked at",
                                        "obj": "Bletchley Park",
                                        "obj_def": "British wartime codebreaking center",
                                        "spo_def": "Alan M. Turing worked on cryptanalysis at Bletchley Park.",
                                    }
                                ]
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "triples": [
                                {
                                    "subj": "Alan Turing",
                                    "subj_def": "English mathematician and logician",
                                    "pred": "formulated",
                                    "obj": "Turing machine",
                                    "obj_def": "Mathematical model of computation",
                                    "spo_def": "Alan Turing formulated the Turing machine in 1936.",
                                }
                            ]
                        }
                    )
                ]
            if "Bletchley" in req_str:
                return [
                    json.dumps(
                        {
                            "named_entities": [
                                ["Alan M. Turing", "British cryptanalyst at Bletchley Park"],
                                ["Bletchley Park", "British wartime codebreaking center"],
                            ]
                        }
                    )
                ]
            return [
                json.dumps(
                    {
                        "named_entities": [
                            ["Alan Turing", "English mathematician and logician"],
                            ["Turing machine", "Mathematical model of computation"],
                        ]
                    }
                )
            ]

    orch_lm = RLMOrchestratorLM()
    worker_lm = QwenWorkerLM()
    program = HippoRAGProgram(
        enable_gbnf=True,
        orchestrator_lm=orch_lm,
        rlm_sub_lm=worker_lm,
        rlm_max_iters=3,
    )
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_step7_rlm.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
        synonymy_threshold=-1.0,
    )
    try:
        # 1. Index first document so "alan turing" and "turing machine" are materialized and linked to chunk 1 on disk
        assert engine.index(["Alan Turing formulated the Turing machine in 1936."]) == 1
        assert engine._graph_rlm_stats["synonym_judge_tokens"] == 30

        # Verify _apply_shacl_repairs ignores nonexistent focus nodes without inserting orphan labels
        with engine.managed_conn.transaction() as conn:
            noop_cnt = engine._apply_shacl_repairs(
                conn,
                [
                    {
                        "focus_node": "entity-nonexistent",
                        "action": "set_definition",
                        "value": "Should not count",
                    },
                    {
                        "focus_node": "entity-nonexistent",
                        "action": "set_pref_label",
                        "value": "Ghost Label",
                    },
                ],
            )
            assert noop_cnt == 0
            orphan_lbls = conn.execute(
                "SELECT COUNT(*) FROM entity_labels WHERE entity_id = 'entity-nonexistent'"
            ).fetchone()[0]
            assert orphan_lbls == 0

            # Inject 2 ConceptShape/definition SHACL violations on "alan turing" and "turing machine"
            conn.execute(
                "UPDATE entities SET definition = '' WHERE name IN ('alan turing', 'turing machine')"
            )

        pre_lint = engine.lint_graph()
        assert pre_lint["conforms"] is False
        assert pre_lint["violation_count"] == 2
        assert all(v["shape"] == "ConceptShape/definition" for v in pre_lint["violations"])

        # 2. Index second document:
        #    - Step 6 runs direct GBNF synonym_judge on candidate pairs (30 tokens)
        #    - Step 7 runs shacl_healer_rlm using all 5 bound REPL tools (applying repair[0] in REPL + repair[1] post-RLM)
        program.telemetry.clear()
        program.synonym_audit_trail.clear()
        program.shacl_audit_trail.clear()

        assert engine.index(["Alan M. Turing worked on cryptanalysis at Bletchley Park."]) == 1

        # Verify all 5 REPL tools ran inside shacl_healer_rlm and healed both violations
        assert "shacl_healer_rlm_invoked" in repl_tool_events
        assert (
            sum(1 for ev in repl_tool_events if ev.startswith("qwen_propose_shacl_repairs:")) == 2
        )

        # Verify exact non-double-counted token accounting:
        # Step 6 (direct GBNF synonym_judge) = 30 tokens on doc 2 (total 30 + 30 = 60 across doc 1 + doc 2)
        assert len(program.synonym_audit_trail) == 1
        assert program.synonym_audit_trail[0]["mode"] == "direct"
        assert program.synonym_audit_trail[0]["tokens_used"] == 30
        assert engine._graph_rlm_stats["synonym_judge_tokens"] == 60

        # Step 7 (shacl_healer_rlm) = 40 orchestrator + 30 worker = 70 tokens
        assert len(program.shacl_audit_trail) == 2  # inner direct (30) + outer RLM (70)
        assert program.shacl_audit_trail[0]["mode"] == "direct"
        assert program.shacl_audit_trail[0]["tokens_used"] == 30
        assert program.shacl_audit_trail[-1]["mode"] == "rlm"
        assert program.shacl_audit_trail[-1]["tokens_used"] == 70
        assert engine._graph_rlm_stats["shacl_healer_tokens"] == 70

        # Verify Step 7 SHACL violation and deduplicated repair stats (1 in REPL + 1 post-RLM = 2, not 3)
        assert engine._graph_rlm_stats["shacl_violations_initial"] > 0
        assert engine._graph_rlm_stats["shacl_violations_initial"] == 2
        assert engine._graph_rlm_stats["shacl_violations_before"] == 2
        assert engine._graph_rlm_stats["shacl_violations_remaining"] == 0
        assert engine._graph_rlm_stats["shacl_violations_after"] == 0
        assert engine._graph_rlm_stats["shacl_repairs_applied"] == 2
        assert engine._graph_rlm_stats["shacl_conforms"] is True

        healed_defs = [
            str(r[0])
            for r in engine.conn.execute(
                "SELECT definition FROM entities WHERE name IN ('alan turing', 'turing machine')"
            ).fetchall()
        ]
        assert len(healed_defs) == 2
        assert all("formulated the Turing machine" in d for d in healed_defs)

        post_lint = engine.lint_graph()
        assert post_lint["conforms"] is True
        assert post_lint["violations"] == []
    finally:
        engine.close()


def test_engine_define_missing_entities_telemetry_and_no_heuristic_fallbacks(
    tmp_path: Path,
) -> None:
    """Verify index_chunks repairs missing entity definitions via DefineMissingEntities, tracks stats, and never uses heuristic fallbacks."""
    import json

    class MissingEntityRepairLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
                )
            req_str = str(prompt or messages or kwargs)
            if (
                "DefineMissingEntities" in req_str
                or "missing_entities" in req_str
                or "entity_definitions" in req_str
            ):
                return [
                    json.dumps(
                        {
                            "entity_definitions": [
                                [
                                    "Charles Babbage",
                                    "English polymath who originated the programmable computer",
                                ]
                            ]
                        }
                    )
                ]
            if "ExtractTriples" in req_str or "triples" in req_str:
                return [
                    json.dumps(
                        {
                            "triples": [
                                {
                                    "sub": "Ada Lovelace",
                                    "pred": "collaborated with",
                                    "obj": "Charles Babbage",
                                    "def": "Ada Lovelace collaborated with Charles Babbage.",
                                }
                            ]
                        }
                    )
                ]
            if "ExtractEntities" in req_str or "named_entities" in req_str:
                return [
                    json.dumps(
                        {
                            "named_entities": [
                                ["Ada Lovelace", "English mathematician and writer"],
                            ]
                        }
                    )
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=MissingEntityRepairLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_missing_defs.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        with dspy.context(track_usage=True):
            n = engine.index(["Ada Lovelace collaborated with Charles Babbage."])
        assert n == 1
        assert engine._graph_rlm_stats["missing_entity_defs_count"] == 1
        assert engine._graph_rlm_stats["missing_entity_defs_tokens"] == 45
        assert len(program.entity_def_audit_trail) == 1
        assert program.entity_def_audit_trail[0]["missing_count"] == 1
        assert program.entity_def_audit_trail[0]["tokens"] == 45

        rows = {
            str(r[0]): str(r[1])
            for r in engine.conn.execute("SELECT name, definition FROM entities").fetchall()
        }
        assert rows["ada lovelace"] == "English mathematician and writer"
        assert (
            rows["charles babbage"] == "English polymath who originated the programmable computer"
        )
        for defn in rows.values():
            assert "(in statement:" not in defn
            assert "(in passage:" not in defn
            assert "extracted from" not in defn

        report = engine.lint_graph()
        assert report["conforms"] is True
        assert report["violations"] == []
    finally:
        engine.close()


def test_engine_evolve_prompts_reindexes_on_extraction_mutation_and_captures_all_stage_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify mutating extraction/graph predictors in GEPA triggers candidate re-indexing, injects all stage traces into dspy.settings.trace, and emits per-predictor feedback."""
    import json

    ner_calls: list[str] = []

    class CandidateSensitiveExtractionLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                )
            req_str = str(prompt or messages or kwargs)
            if "ExtractEntities" in req_str or "named_entities" in req_str:
                ner_calls.append(req_str)
                if "MUTATED_NER_MARKER" in req_str:
                    return [
                        json.dumps(
                            {
                                "named_entities": [
                                    [
                                        "Alan Turing",
                                        "English mathematician and pioneer of theoretical computer science",
                                    ],
                                    ["Bletchley Park", "British wartime codebreaking center"],
                                ]
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "named_entities": [
                                ["Unknown Person", "Unspecified entity"],
                            ]
                        }
                    )
                ]
            if "ExtractTriples" in req_str or "triples" in req_str:
                if "Alan Turing" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    {
                                        "sub": "Alan Turing",
                                        "pred": "worked at",
                                        "obj": "Bletchley Park",
                                        "def": "Alan Turing worked at Bletchley Park.",
                                    }
                                ]
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "triples": [
                                {
                                    "sub": "Unknown Person",
                                    "pred": "visited",
                                    "obj": "Nowhere",
                                    "def": "Unknown Person visited Nowhere.",
                                }
                            ]
                        }
                    )
                ]
            if (
                "DefineMissingEntities" in req_str
                or "missing_entities" in req_str
                or "entity_definitions" in req_str
            ):
                return [
                    json.dumps(
                        {
                            "entity_definitions": [
                                ["Nowhere", "Unspecified place"],
                                ["Bletchley Park", "British wartime codebreaking center"],
                            ]
                        }
                    )
                ]
            if "JudgeSynonyms" in req_str or "candidate_pairs" in req_str:
                return [json.dumps({"decisions": []})]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    lm = CandidateSensitiveExtractionLM()
    dspy.settings.configure(lm=lm)
    program = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_evo_extract.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
        synonymy_threshold=-1.0,
    )
    try:
        doc_gold = "Alan Turing\nAlan Turing worked at Bletchley Park during World War II."
        doc_dist = "Distractor\nThe weather in London was rainy."
        engine.index([doc_gold, doc_dist])

        trainset = [
            dspy.Example(
                question="Where did Alan Turing work?",
                gold_answers=["Bletchley Park"],
                gold_docs=[doc_gold],
                item_docs=[doc_gold, doc_dist],
            ).with_inputs("question", "item_docs")
        ]

        # 1. Verify _extraction_fingerprint sensitivity
        fp_base = engine._extraction_fingerprint(program)
        qa_mutated = program.deepcopy()
        qa_mutated.qa.predict.signature = qa_mutated.qa.predict.signature.with_instructions(
            "QA-only mutation"
        )
        assert engine._extraction_fingerprint(qa_mutated) == fp_base

        ner_mutated = program.deepcopy()
        ner_mutated.ner.signature = ner_mutated.ner.signature.with_instructions(
            "MUTATED_NER_MARKER: Extract all entities with definitions."
        )
        assert engine._extraction_fingerprint(ner_mutated) != fp_base

        # 2. Verify GEPA compile loop triggers re-indexing on NER mutation, reuses cache on QA-only mutation,
        #    captures extraction + QA traces in dspy.settings.trace, and provides stage-specific feedback.
        observed_traces: list[Any] = []
        feedback_by_stage: dict[str, str] = {}

        class InspectingGEPA:
            def __init__(self, metric: Any, **kwargs: Any) -> None:
                self.metric = metric
                self.kwargs = kwargs

            def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
                ex = trainset[0]
                # Evaluate baseline candidate with trace capture enabled
                trace_base: list[Any] = []
                with dspy.context(trace=trace_base, track_usage=True):
                    _ = student(**ex.inputs())
                entities_before = {
                    str(r[0]) for r in engine.conn.execute("SELECT name FROM entities").fetchall()
                }
                assert "unknown person" in entities_before

                # Evaluate QA-only mutated candidate: must NOT re-run NER extraction
                ner_count_after_base = len(ner_calls)
                cand_qa = student.deepcopy()
                cand_qa.qa.predict.signature = cand_qa.qa.predict.signature.with_instructions(
                    "Answer concisely."
                )
                with dspy.context(trace=[], track_usage=True):
                    _ = cand_qa(**ex.inputs())
                assert len(ner_calls) == ner_count_after_base

                # Evaluate NER-mutated candidate: MUST wipe openie_chunks and re-run NER extraction!
                cand_ner = student.deepcopy()
                cand_ner.ner.signature = cand_ner.ner.signature.with_instructions(
                    "MUTATED_NER_MARKER: Extract all entities with definitions."
                )
                trace_mut: list[Any] = []
                with dspy.context(trace=trace_mut, track_usage=True):
                    pred_mut = cand_ner(**ex.inputs())
                assert len(ner_calls) > ner_count_after_base

                entities_after = {
                    str(r[0]) for r in engine.conn.execute("SELECT name FROM entities").fetchall()
                }
                assert "alan turing" in entities_after
                assert "bletchley park" in entities_after

                # Verify trace_mut contains extraction predictors (ner, triples) as well as QA predictors
                observed_traces.extend(trace_mut)
                named_preds = dict(cand_ner.named_predictors())
                for req_stage in ("ner", "triples", "qa.predict"):
                    target_sig = named_preds[req_stage].signature
                    matching = [t for t in trace_mut if t[0].signature.equals(target_sig)]
                    assert len(matching) >= 1, f"Missing trace for {req_stage}"
                    res = self.metric(
                        ex,
                        pred_mut,
                        trace_mut,
                        req_stage,
                        matching,
                    )
                    feedback_by_stage[req_stage] = res.feedback

                return cand_ner

        monkeypatch.setattr(dspy, "GEPA", InspectingGEPA, raising=False)
        compiled = engine.evolve_prompts(trainset, mode="gepa")
        assert "MUTATED_NER_MARKER" in compiled.ner.signature.instructions
        assert len(observed_traces) >= 3
        assert "[ner]" in feedback_by_stage["ner"]
        assert "Stage[ner]" in feedback_by_stage["ner"]
        assert "[triples]" in feedback_by_stage["triples"]
        assert "Stage[triples]" in feedback_by_stage["triples"]
        assert "[qa.predict]" in feedback_by_stage["qa.predict"]
    finally:
        engine.close()


def test_sprint3_engine_as_rm_and_as_retriever_and_dspy_retrieve_compatibility(
    tmp_path: Path,
) -> None:
    """Verify YawGraphEngine.as_rm, as_retriever, workspace DSPy cache isolation, and stock dspy.Retrieve(k=...) compatibility."""
    from yaw.daemon.graph.inference import YawPassage, YawRetriever, YawRM

    dspy.settings.configure(lm=DeterministicMockLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_rm_sprint3.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        assert (tmp_path / "dspy_cache").exists()
        assert getattr(dspy, "cache", None) is not None

        doc_turing = "Alan Turing designed the Automatic Computing Engine (ACE) at the National Physical Laboratory."
        doc_lovelace = (
            "Ada Lovelace wrote the first algorithm for Charles Babbage's Analytical Engine."
        )
        engine.index([doc_turing, doc_lovelace])

        rm = engine.as_rm(k=2, configure_dspy=True)
        assert isinstance(rm, YawRM)
        assert dspy.settings.rm is rm

        rm_hits = rm("Who designed the Automatic Computing Engine?", k=2)
        assert len(rm_hits) == 2
        assert all(isinstance(h, YawPassage) for h in rm_hits)
        assert any("Alan Turing" in h.long_text for h in rm_hits)
        assert all(h.pid.startswith("chunk-") for h in rm_hits)
        assert [h.rank for h in rm_hits] == [1, 2]

        stock_retrieve = dspy.Retrieve(k=2)
        stock_pred = stock_retrieve("Who designed the Automatic Computing Engine?")
        assert len(stock_pred.passages) == 2
        assert any("Alan Turing" in p for p in stock_pred.passages)

        retriever = engine.as_retriever(k=2, configure_dspy=True)
        assert isinstance(retriever, YawRetriever)
        assert isinstance(retriever, dspy.Retrieve)
        ret_pred = retriever("Who wrote the algorithm for the Analytical Engine?", k=1)
        assert len(ret_pred.passages) == 1
        assert len(ret_pred.scores) == 1
        assert ret_pred.context.strip() != ""
    finally:
        dspy.configure(rm=None)
        engine.close()


def test_sprint3_engine_search_graph_for_react_and_react_v2_fallback_integration(
    tmp_path: Path,
) -> None:
    """Verify _search_graph_for_react queries real SQLite yaw.db facts and passages and powers dspy.ReActV2 multi-hop fallback."""
    import json

    class ReActEngineMockLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 24, "completion_tokens": 12, "total_tokens": 36},
                )
            req_str = str(prompt or messages or kwargs)
            if "tool_name" in req_str or "tool_args" in req_str or "search_graph_facts" in req_str:
                if "Facts:" in req_str and "1969" in req_str:
                    return [
                        json.dumps(
                            {
                                "thought": "Found the year in the graph facts.",
                                "tool_name": "finish",
                                "tool_args": {
                                    "reasoning": "Alan Turing received the Presidential Medal of Freedom in 1969 according to the graph search.",
                                    "answer": "1969",
                                },
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "thought": "Search graph facts for Alan Turing.",
                            "tool_name": "search_graph_facts",
                            "tool_args": {"query": "Alan Turing"},
                        }
                    )
                ]
            if (
                "AnswerQuestion" in req_str
                or "Retrieved evidence passages" in req_str
                or ("context" in req_str and "question" in req_str and "triples" not in req_str)
            ):
                return [
                    json.dumps(
                        {
                            "reasoning": "Single-shot context lacks the medal year.",
                            "answer": "Insufficient context",
                        }
                    )
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    lm = ReActEngineMockLM()
    dspy.settings.configure(lm=lm)
    program = HippoRAGProgram(
        enable_gbnf=False,
        orchestrator_lm=lm,
        rlm_sub_lm=lm,
        enable_react_fallback=True,
    )
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_react_sprint3.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        engine.index(
            [
                "Alan Turing was an English mathematician and computer scientist who founded computer science.",
                "Alan Turing received the Presidential Medal of Freedom in 1969.",
            ]
        )
        graph_search_text = engine._search_graph_for_react("Alan Turing")
        assert "Alan Turing" in graph_search_text
        assert "computer science" in graph_search_text

        pred = engine.program(
            question="When did Alan Turing receive the Presidential Medal of Freedom?"
        )
        assert pred.answer == "1969"
    finally:
        engine.close()


def test_sprint3_engine_evaluate_and_pareto_frontier_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify YawGraphEngine.evaluate (dspy.Evaluate + answer_passage_match) and top-3 Pareto frontier export in optimize()."""
    import json
    from types import SimpleNamespace

    from yaw.daemon.graph.inference import ParetoProgramEnsemble

    lm = DeterministicMockLM()
    dspy.settings.configure(lm=lm)
    program = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_eval_sprint3.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        doc = "Alan Turing designed the ACE and founded computer science."
        engine.index([doc])

        devset = [
            dspy.Example(
                question="Who designed the ACE?",
                answer="Alan Turing",
                gold_answers=["Alan Turing"],
                gold_docs=[doc],
            ).with_inputs("question")
        ]

        eval_json_path = tmp_path / "direct_eval.json"
        eval_res = engine.evaluate(devset, save_as_json=eval_json_path)
        assert isinstance(eval_res, dspy.evaluate.EvaluationResult)
        assert eval_res.score == 100.0
        assert len(eval_res.results) == 1
        assert eval_json_path.exists()
        saved_eval_rows = json.loads(eval_json_path.read_text(encoding="utf-8"))
        assert isinstance(saved_eval_rows, list) and len(saved_eval_rows) == 1

        apm_res = engine.evaluate(devset, metric=dspy.evaluate.answer_passage_match)
        assert isinstance(apm_res, dspy.evaluate.EvaluationResult)
        assert apm_res.score == 100.0

        observed_feedbacks: list[str] = []

        class FrontierMockGEPA:
            def __init__(self, metric: Any, **kwargs: Any) -> None:
                self.metric = metric
                self.kwargs = kwargs

            def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
                with dspy.context(track_usage=True):
                    pred = student(question=trainset[0].question)
                score_obj = self.metric(trainset[0], pred)
                observed_feedbacks.append(str(score_obj.feedback))

                cand_0 = student.deepcopy()
                cand_0.qa.predict.signature = cand_0.qa.predict.signature.with_instructions(
                    "Pareto candidate 0 instruction."
                )
                cand_1 = student.deepcopy()
                cand_1.qa.predict.signature = cand_1.qa.predict.signature.with_instructions(
                    "Pareto candidate 1 instruction."
                )
                cand_2 = student.deepcopy()
                cand_2.qa.predict.signature = cand_2.qa.predict.signature.with_instructions(
                    "Pareto candidate 2 instruction."
                )
                cand_dup = student.deepcopy()
                cand_dup.qa.predict.signature = cand_dup.qa.predict.signature.with_instructions(
                    "Pareto candidate 0 instruction."
                )

                cand_0.detailed_results = SimpleNamespace(
                    candidates=[cand_0, cand_1, cand_2, cand_dup],
                    val_aggregate_scores=[0.92, 0.89, 0.87, 0.92],
                    per_val_instance_best_candidates={0: [0, 1], 1: [2]},
                    per_objective_best_candidates=None,
                    best_idx=0,
                )
                return cand_0

        monkeypatch.setattr(dspy, "GEPA", FrontierMockGEPA, raising=False)
        save_path = tmp_path / "compiled_sprint3.json"
        compiled = engine.optimize(
            devset,
            valset=devset,
            mode="gepa",
            save_path=save_path,
        )
        assert len(observed_feedbacks) == 1
        assert "AnswerPassageMatch=1" in observed_feedbacks[0]
        assert len(compiled.pareto_frontier_states) == 3
        assert isinstance(compiled.pareto_ensemble, ParetoProgramEnsemble)
        assert len(compiled.pareto_ensemble.programs) == 3
        assert save_path.exists()
        assert (tmp_path / "compiled_sprint3_eval.json").exists()

        reloaded = HippoRAGProgram(enable_gbnf=False, compiled_path=save_path)
        assert len(reloaded.pareto_frontier_states) == 3
        assert isinstance(reloaded.pareto_ensemble, ParetoProgramEnsemble)
    finally:
        engine.close()


def test_canonical_retrieval_hook_and_rag_qa_forward_unification(
    tmp_path: Path,
) -> None:
    """Verify _canonical_retrieval_hook unifies YawRM, YawRetriever, HippoRAGProgram.forward, and rag_qa with GEPA trace replay and ReActV2/Pareto support."""
    import json

    class UnifiedHookMockLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                )
            req_str = str(prompt or messages or kwargs)
            if "tool_name" in req_str or "tool_args" in req_str or "search_graph_facts" in req_str:
                if "Facts:" in req_str and "computer science" in req_str:
                    return [
                        json.dumps(
                            {
                                "thought": "Found the answer in graph facts.",
                                "tool_name": "finish",
                                "tool_args": {
                                    "reasoning": "Recovered via ReActV2 graph exploration.",
                                    "answer": "Alan Turing",
                                },
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "thought": "Search graph facts for Alan Turing.",
                            "tool_name": "search_graph_facts",
                            "tool_args": {"query": "Alan Turing"},
                        }
                    )
                ]
            if "ABSTAIN_DIRECT_QA" in req_str and (
                "AnswerQuestion" in req_str
                or "Retrieved evidence passages" in req_str
                or ("context" in req_str and "question" in req_str and "triples" not in req_str)
            ):
                return [
                    json.dumps(
                        {
                            "reasoning": "Abstaining on single-shot QA to trigger ReActV2 fallback.",
                            "answer": "Insufficient context",
                        }
                    )
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    lm = UnifiedHookMockLM()
    dspy.settings.configure(lm=lm)
    program = HippoRAGProgram(
        enable_gbnf=False,
        orchestrator_lm=lm,
        rlm_sub_lm=lm,
        enable_react_fallback=True,
    )
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_unified_hook.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        # 1. Oblivion verification: legacy dual hooks demolished, only _canonical_retrieval_hook bound
        assert [m for m in dir(engine) if m.endswith("_retrieval_hook")] == [
            "_canonical_retrieval_hook"
        ]
        assert engine.program._retrieval_hook == engine._canonical_retrieval_hook

        doc_turing = "Alan Turing\nAlan Turing was an English mathematician and computer scientist."
        doc_lovelace = "Ada Lovelace\nAda Lovelace worked on the Analytical Engine."
        engine.index([doc_turing, doc_lovelace])

        # 2. Canonical 4-tuple contract & parity across RM, Retriever, Program.forward, and rag_qa
        q = "Who was the English computer scientist?"
        ctx, texts, scores, pids = engine._canonical_retrieval_hook(q, k=2)
        assert isinstance(ctx, str) and "Alan Turing" in ctx
        assert len(texts) == 2 and all(not t.startswith("chunk-") for t in texts)
        assert len(scores) == 2 and all(isinstance(s, float) for s in scores)
        assert len(pids) == 2 and all(cid.startswith("chunk-") for cid in pids)

        rm = engine.as_rm(k=2, configure_dspy=True)
        assert rm.hook == engine._canonical_retrieval_hook

        # 3. Verify GEPA extraction trace replay works even when invoked via external YawRM / dspy.Retrieve
        ext_trace: list[Any] = []
        with dspy.context(trace=ext_trace, track_usage=True):
            rm_hits = rm(q, k=2, item_docs=[doc_turing, doc_lovelace])
        assert len(rm_hits) == 2
        assert [h.pid for h in rm_hits] == list(pids[:2])
        assert (
            len(ext_trace) >= 2
        )  # Captured NER and triples traces replayed into dspy.settings.trace!

        fwd_pred = engine.program(question=q)
        assert tuple(getattr(fwd_pred, "retrieved_chunk_ids", ()))[:2] == pids[:2]
        assert tuple(getattr(fwd_pred, "retrieved_passages", ()))[:2] == texts[:2]
        assert tuple(getattr(fwd_pred, "retrieved_scores", ()))[:2] == scores[:2]
        assert getattr(fwd_pred, "context", "") == ctx

        qa_out = engine.rag_qa([q])
        assert len(qa_out) == 1
        assert qa_out[0]["retrieved_chunk_ids"] == list(pids[:2])
        assert qa_out[0]["retrieved_scores"] == list(scores[:2])
        assert qa_out[0]["context"] == ctx
        assert qa_out[0]["answer"] == "Alan Turing"

        # 4. Verify rag_qa() executes through HippoRAGProgram.forward (triggering ReActV2 fallback on abstention)
        qa_fallback_out = engine.rag_qa(["ABSTAIN_DIRECT_QA: Who founded computer science?"])
        assert len(qa_fallback_out) == 1
        assert qa_fallback_out[0]["answer"] == "Alan Turing"
    finally:
        dspy.configure(rm=None)
        engine.close()


def test_hybrid_e2e_per_query_isolation_stage_caching_and_reflective_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify per-query sandbox isolation, 4-tier stage caching, component selector mutation resilience, and minibatch q_stats lookup."""
    import json
    from types import SimpleNamespace

    stage_call_counts = {"ner": 0, "synonym_judge": 0}

    class StageCountingMockLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                )
            req_str = str(prompt or messages or kwargs)
            if "ExtractEntities" in req_str or "named_entities" in req_str:
                stage_call_counts["ner"] += 1
            elif "JudgeSynonyms" in req_str or "candidate_pairs" in req_str:
                stage_call_counts["synonym_judge"] += 1
                return [json.dumps({"decisions": []})]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    lm = StageCountingMockLM()
    dspy.settings.configure(lm=lm)
    program = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_hybrid_e2e.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
        synonymy_threshold=-1.0,
    )
    try:
        doc_q1 = "Alan Turing\nAlan Turing designed the Automatic Computing Engine in 1945."
        doc_q2_a = "Ada Lovelace\nAda Lovelace wrote the first computer algorithm."
        doc_q2_b = "Charles Babbage\nCharles Babbage designed the Analytical Engine."

        ex1 = dspy.Example(
            question="Who designed the Automatic Computing Engine?",
            gold_answers=["Alan Turing"],
            gold_docs=[doc_q1],
            item_docs=[doc_q1],
        ).with_inputs("question", "item_docs")
        ex2 = dspy.Example(
            question="Who wrote the first computer algorithm?",
            gold_answers=["Ada Lovelace"],
            gold_docs=[doc_q2_a],
            item_docs=[doc_q2_a, doc_q2_b],
        ).with_inputs("question", "item_docs")

        # Verify _ActiveTraceComponentSelector continues matching after candidate instructions mutate
        selector = YawGraphEngine._ActiveTraceComponentSelector(program)
        mut_ner_sig = program.ner.signature.with_instructions("Mutated NER instruction v2")
        fake_pred_obj = SimpleNamespace(signature=mut_ner_sig)
        fake_state = SimpleNamespace(
            list_of_named_predictors=["ner", "qa.predict"],
            named_predictor_id_to_update_next_for_program_candidate=[0],
        )
        selected = selector(
            fake_state,
            [{"trace": [(fake_pred_obj, {}, {})]}],
            [0.8],
            0,
            {
                "ner": "Mutated NER instruction v2",
                "qa.predict": program.qa.predict.signature.instructions,
            },
        )
        assert selected == ["ner"]

        class VerifyingGEPA:
            def __init__(self, metric: Any, **kwargs: Any) -> None:
                self.metric = metric

            def compile(self, student: Any, trainset: list[Any], valset: Any = None) -> Any:
                # 1. Run ex1 then ex2 on student (pooled corpus graph indexed once, zero query-order drift)
                with dspy.context(trace=[], track_usage=True):
                    pred1_live = student(**ex1.inputs())
                c_cnt_q1 = engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                assert c_cnt_q1 == 3

                with dspy.context(trace=[], track_usage=True):
                    pred2_live = student(**ex2.inputs())
                c_cnt_q2 = engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                assert c_cnt_q2 == 3

                # Verify post-hoc reflective evaluation on ex1 vs ex2 uses each example's own q_stats
                score1_live = self.metric(ex1, pred1_live)
                score2_live = self.metric(ex2, pred2_live)
                assert score1_live.score > 0.0
                assert score2_live.score > 0.0

                # 2. Mutate QA only: OpenIE and SynonymJudge must hit stage caches (0 extra LLM calls)
                ner_before = stage_call_counts["ner"]
                syn_before = stage_call_counts["synonym_judge"]
                cand_qa = student.deepcopy()
                cand_qa.qa.predict.signature = cand_qa.qa.predict.signature.with_instructions(
                    "Mutated QA instruction"
                )
                with dspy.context(trace=[], track_usage=True):
                    pred1_cached = cand_qa(**ex1.inputs())
                assert stage_call_counts["ner"] == ner_before
                assert stage_call_counts["synonym_judge"] == syn_before
                score1_cached = self.metric(ex1, pred1_cached)
                # Verify token accounting and score parity between live indexing and cache hit
                assert abs(score1_live.score - score1_cached.score) < 1e-9

                # 3. Mutate SynonymJudge only: OpenIE hits cache (0 extra NER calls), SynonymJudge runs live!
                cand_syn = student.deepcopy()
                cand_syn.synonym_judge.signature = (
                    cand_syn.synonym_judge.signature.with_instructions(
                        "Mutated SynonymJudge instruction"
                    )
                )
                with dspy.context(trace=[], track_usage=True):
                    _ = cand_syn(**ex2.inputs())
                assert stage_call_counts["ner"] == ner_before
                assert stage_call_counts["synonym_judge"] > syn_before
                return cand_syn

        monkeypatch.setattr(dspy, "GEPA", VerifyingGEPA, raising=False)
        _ = engine.evolve_prompts([ex1, ex2], mode="gepa")
    finally:
        engine.close()


def test_evolve_prompts_real_unmocked_gepa_and_miprov2_with_context_limit_guard(
    tmp_path: Path,
) -> None:
    """Run real un-mocked dspy.MIPROv2 and dspy.GEPA end-to-end with a strict context-size guard and RLM proposer."""
    import json

    max_observed_prompt_chars = 0

    class ContextGuardedRealOptLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            nonlocal max_observed_prompt_chars
            if getattr(dspy.settings, "usage_tracker", None) is not None:
                dspy.settings.usage_tracker.add_usage(
                    self.model,
                    {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37},
                )
            req_str = str(prompt or messages or kwargs)
            max_observed_prompt_chars = max(max_observed_prompt_chars, len(req_str))
            if len(req_str) > 32000:
                raise RuntimeError(
                    f"Request ({len(req_str)} chars) exceeds available context size (32000 chars)"
                )
            if "variables_info" in req_str and "repl_history" in req_str:
                return [
                    "[[ ## reasoning ## ]]\n"
                    "Distill examples_with_feedback in the Python REPL and call synthesize_instruction.\n\n"
                    "[[ ## code ## ]]\n"
                    "```python\n"
                    "digest = [{'Inputs': {k: (v[:300] if isinstance(v, str) else v) for k, v in ex.get('Inputs', {}).items()}, 'Generated Outputs': ex.get('Generated Outputs', {}), 'Feedback': ex.get('Feedback', '')} for ex in examples_with_feedback[:3]]\n"
                    "res = synthesize_instruction(current_instruction=current_instruction, examples_with_feedback=digest)\n"
                    "SUBMIT(new_instruction=res)\n"
                    "```\n"
                    "[[ ## completed ## ]]"
                ]
            if "new_instruction" in req_str and (
                "examples_with_feedback" in req_str or "length_limit" in req_str
            ):
                return [
                    json.dumps(
                        {
                            "new_instruction": (
                                "Extract accurate entities, triples, and answers strictly grounded in the passage context. "
                                'Respond with JSON: {"named_entities": [["Entity", "Definition"]], "triples": [["s", "s_def", "p", "o", "o_def", "spo_def"]], "entity_definitions": [["E", "D"]], "decisions": [], "repairs": [], "fact": [["s", "p", "o"]], "query_intent": "...", "reasoning": "...", "answer": "..."}.'
                            )
                        }
                    )
                ]
            if "Automatic Computing Engine" in req_str and (
                "AnswerQuestion" in req_str or "Retrieved evidence passages" in req_str
            ):
                # Return a suboptimal answer on baseline so GEPA reflects and mutates
                if " strictly grounded in the passage context" not in req_str:
                    return [
                        json.dumps(
                            {
                                "reasoning": "Initial incomplete guess.",
                                "answer": "Babbage",
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "reasoning": "Alan Turing designed the Automatic Computing Engine.",
                            "answer": "Alan Turing",
                        }
                    )
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    lm = ContextGuardedRealOptLM()
    dspy.settings.configure(lm=lm)
    program = HippoRAGProgram(
        enable_gbnf=False,
        orchestrator_lm=lm,
        rlm_sub_lm=lm,
    )
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_real_unmocked_opt.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        doc_turing = "Alan Turing\nAlan Turing designed the Automatic Computing Engine in 1945."
        doc_lovelace = "Ada Lovelace\nAda Lovelace wrote the first computer algorithm."
        # Pad item_docs with large distractor strings so 10 raw examples would exceed 60,000 chars if un-distilled
        filler_docs = [
            f"Distractor {i}\n" + ("Historical archive notes on mechanical calculators. " * 25)
            for i in range(6)
        ]

        trainset = [
            dspy.Example(
                question="Who designed the Automatic Computing Engine?",
                gold_answers=["Alan Turing"],
                gold_docs=[doc_turing],
                item_docs=[doc_turing, *filler_docs[:3]],
            ).with_inputs("question", "item_docs"),
            dspy.Example(
                question="Who wrote the first computer algorithm?",
                gold_answers=["Ada Lovelace"],
                gold_docs=[doc_lovelace],
                item_docs=[doc_lovelace, *filler_docs[3:]],
            ).with_inputs("question", "item_docs"),
        ]
        valset = [trainset[0], trainset[1]]

        save_path = tmp_path / "real_compiled.json"
        compiled = engine.evolve_prompts(
            trainset,
            valset=valset,
            mode="hybrid",
            num_candidates=2,
            num_trials=5,
            save_path=save_path,
        )
        assert isinstance(compiled, HippoRAGProgram)
        assert max_observed_prompt_chars < 32000
        assert save_path.exists()
        assert getattr(compiled, "detailed_results", None) is not None
        assert len(compiled.shacl_healer_rlm.generate_action.demos) >= 1
        telemetry_path = tmp_path / "real_compiled_proposer_telemetry.json"
        assert telemetry_path.exists()
        telemetry_data = json.loads(telemetry_path.read_text(encoding="utf-8"))
        assert len(telemetry_data) >= 1
        assert all(entry.get("mode") == "rlm" for entry in telemetry_data)
    finally:
        engine.close()


def test_engine_homographs_paronyms_and_same_label_triple_permutations(
    tmp_path: Path,
) -> None:
    """Verify paronyms ('Cindy Real' vs 'Cinderella'), typos ('Cindarella'), and homograph permutations ('a ball' skos:related + owl:differentFrom 'a ball', and 'a ball, was handled, in a ball')."""
    import json
    import re

    from yaw.daemon.graph.storage.db import load_graph_topology

    class HomographAndParonymLM(DeterministicMockLM):
        def __call__(
            self,
            prompt: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            req_str = str(prompt or messages or kwargs)
            if "[[ ## candidate_pairs ## ]]" in req_str or "candidate_pairs" in req_str:
                decisions = []
                # Extract JSON array of candidate pairs from prompt
                for match in re.finditer(r"\{[^{}]*\"pair_id\"[^{}]*\}", req_str):
                    try:
                        p = json.loads(match.group(0))
                    except Exception:
                        continue
                    pid = str(p.get("pair_id", ""))
                    na = str(p.get("name_a", "")).lower()
                    nb = str(p.get("name_b", "")).lower()
                    names = {na, nb}
                    if names == {"cinderella", "cindarella"}:
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:hiddenLabel",
                                "canonical": "A" if na == "cinderella" else "B",
                            }
                        )
                    elif names == {"cinderella", "cindy real"} or names == {
                        "cindarella",
                        "cindy real",
                    }:
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "owl:differentFrom",
                                "canonical": "A",
                            }
                        )
                    elif na == "a ball" and nb == "a ball":
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:related",
                                "canonical": "A",
                            }
                        )
                return [json.dumps({"decisions": decisions})]

            if (
                "[[ ## named_entities ## ]]" in req_str
                or "Resource Description Framework" in req_str
            ):
                if "Cindy Real" in req_str:
                    return [
                        json.dumps(
                            {
                                "triples": [
                                    [
                                        "Cinderella",
                                        "folk tale heroine who lost a glass slipper",
                                        "attended",
                                        "a ball",
                                        "formal royal dance party",
                                        "Cinderella attended a formal royal dance ball.",
                                    ],
                                    [
                                        "Cindarella",
                                        "misspelled folk tale heroine who lost a glass slipper",
                                        "danced at",
                                        "a ball",
                                        "formal royal dance party",
                                        "Cindarella danced at a formal royal dance ball.",
                                    ],
                                    [
                                        "Cindy Real",
                                        "modern investigative journalist in Chicago",
                                        "wrote about",
                                        "a ball",
                                        "spherical leather sports toy",
                                        "Cindy Real wrote about a spherical sports ball.",
                                    ],
                                ]
                            }
                        )
                    ]
                return [
                    json.dumps(
                        {
                            "triples": [
                                [
                                    "a ball",
                                    "spherical golden toy object",
                                    "was handled in",
                                    "a ball",
                                    "formal royal dance party",
                                    "A spherical golden toy ball was handled in a formal royal dance ball.",
                                ],
                                [
                                    "a ball",
                                    "formal royal dance party",
                                    "featured",
                                    "a ball",
                                    "spherical golden toy object",
                                    "The formal royal dance ball featured a spherical golden toy ball.",
                                ],
                            ]
                        }
                    )
                ]

            if "Cindy Real" in req_str:
                return [
                    json.dumps(
                        {
                            "named_entities": [
                                ["Cinderella", "folk tale heroine who lost a glass slipper"],
                                [
                                    "Cindarella",
                                    "misspelled folk tale heroine who lost a glass slipper",
                                ],
                                ["Cindy Real", "modern investigative journalist in Chicago"],
                                ["a ball", "formal royal dance party"],
                                ["a ball", "spherical leather sports toy"],
                            ]
                        }
                    )
                ]
            return [
                json.dumps(
                    {
                        "named_entities": [
                            ["a ball", "spherical golden toy object"],
                            ["a ball", "formal royal dance party"],
                        ]
                    }
                )
            ]

    dspy.settings.configure(lm=HomographAndParonymLM())
    program = HippoRAGProgram(enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_homograph_permutations.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=5,
    )
    try:
        n = engine.index(
            [
                "Cinderella (sometimes misspelled Cindarella) attended a ball at the palace, while journalist Cindy Real wrote about a ball.",
                "During the ceremony, a ball was handled in a ball, and a ball featured a ball.",
            ]
        )
        assert n == 2

        # 1. Verify 'Cindarella' merged into 'cinderella' as skos:hiddenLabel
        cinderella_row = engine.conn.execute(
            "SELECT entity_id FROM entities WHERE name = 'cinderella'"
        ).fetchone()
        assert cinderella_row is not None
        assert (
            engine.conn.execute("SELECT entity_id FROM entities WHERE name = 'cindarella'").fetchone()
            is None
        )
        hidden_labels = [
            str(r[0])
            for r in engine.conn.execute(
                "SELECT label_norm FROM entity_labels WHERE entity_id = ? AND label_kind = 'skos:hiddenLabel'",
                (str(cinderella_row[0]),),
            ).fetchall()
        ]
        assert "cindarella" in hidden_labels

        # 2. Verify 'Cindy Real' remained distinct from 'Cinderella' with an owl:differentFrom edge
        cindy_real_row = engine.conn.execute(
            "SELECT entity_id FROM entities WHERE name = 'cindy real'"
        ).fetchone()
        assert cindy_real_row is not None
        diff_cindy = engine.conn.execute(
            """
            SELECT COUNT(*) FROM edges
            WHERE kind = 'owl:differentFrom'
              AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
            """,
            (
                str(cinderella_row[0]),
                str(cindy_real_row[0]),
                str(cindy_real_row[0]),
                str(cinderella_row[0]),
            ),
        ).fetchone()[0]
        assert int(diff_cindy) == 2

        # 3. Verify homograph 'a ball' entities exist as distinct sense nodes
        ball_rows = engine.conn.execute(
            "SELECT entity_id, definition FROM entities WHERE name = 'a ball'"
        ).fetchall()
        assert len(ball_rows) >= 2
        ball_ids = {str(r[0]) for r in ball_rows}

        # 4. Verify BOTH owl:differentFrom AND skos:related exist between 'a ball' homographs
        ball_diff_edges = engine.conn.execute(
            """
            SELECT src_id, dst_id, weight FROM edges
            WHERE kind = 'owl:differentFrom'
              AND src_id IN (SELECT entity_id FROM entities WHERE name = 'a ball')
              AND dst_id IN (SELECT entity_id FROM entities WHERE name = 'a ball')
            """
        ).fetchall()
        assert len(ball_diff_edges) >= 2
        assert all(float(r[2]) == 0.0 for r in ball_diff_edges)

        ball_rel_edges = engine.conn.execute(
            """
            SELECT src_id, dst_id, weight FROM edges
            WHERE kind = 'skos:related'
              AND src_id IN (SELECT entity_id FROM entities WHERE name = 'a ball')
              AND dst_id IN (SELECT entity_id FROM entities WHERE name = 'a ball')
            """
        ).fetchall()
        assert len(ball_rel_edges) >= 2
        assert all(float(r[2]) > 0.0 for r in ball_rel_edges)

        # 5. Verify both permutations of 'a ball -> pred -> a ball' exist as distinct non-self-loop facts
        ball_facts = engine.conn.execute(
            """
            SELECT fact_id, subj_id, pred, obj_id FROM facts
            WHERE subj_id IN (SELECT entity_id FROM entities WHERE name = 'a ball')
              AND obj_id IN (SELECT entity_id FROM entities WHERE name = 'a ball')
            """
        ).fetchall()
        assert len(ball_facts) == 2
        for fid, sid, pred, oid in ball_facts:
            assert str(sid) != str(oid), f"Homograph triple collapsed into self-loop on {fid} ({pred})"
            assert str(sid) in ball_ids and str(oid) in ball_ids

        # Verify the two permutations have reversed (subject_id, object_id) poles
        fact_poles = {(str(r[1]), str(r[3])) for r in ball_facts}
        assert len(fact_poles) == 2
        pole_a, pole_b = list(fact_poles)
        assert pole_a == (pole_b[1], pole_b[0])

        # 6. Verify CSR topology includes skos:related + fact edges and excludes owl:differentFrom (weight 0)
        _, topo_entity_ids, topo_edges = load_graph_topology(engine.conn)
        s0, o0 = pole_a
        assert s0 in topo_entity_ids and o0 in topo_entity_ids
        topo_edge_map = {(u, v): w for u, v, w in topo_edges}
        assert topo_edge_map.get((s0, o0), 0.0) > 0.0
        assert topo_edge_map.get((o0, s0), 0.0) > 0.0
        # Verify owl:differentFrom between Cinderella and Cindy Real is excluded from positive-weight CSR topology
        c_id, cr_id = str(cinderella_row[0]), str(cindy_real_row[0])
        assert (c_id, cr_id) not in topo_edge_map
        assert (cr_id, c_id) not in topo_edge_map

        # 7. Verify SHACL graph lint conforms with zero violations
        lint = engine.lint_graph()
        assert lint["conforms"] is True, f"Unexpected SHACL violations: {lint['violations']}"
    finally:
        engine.close()
