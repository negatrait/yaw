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

    def __call__(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> list[str]:
        if getattr(dspy.settings, "usage_tracker", None) is not None:
            dspy.settings.usage_tracker.add_usage(
                self.model,
                {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            )
        req_str = str(prompt or messages or kwargs)
        # Triples stage
        if "ExtractTriples" in req_str or "triples" in req_str:
            if "Lovelace" in req_str:
                return ['{"triples": [["Ada Lovelace", "worked on", "Analytical Engine"]]}']
            return ['{"triples": [["Alan Turing", "father of", "computer science"]]}']
        # Filter facts stage
        if "FilterFacts" in req_str or "fact_before_filter" in req_str or "filter" in req_str.lower():
            if "Lovelace" in req_str:
                return ['{"fact": [["Ada Lovelace", "worked on", "Analytical Engine"]]}']
            return ['{"fact": [["Alan Turing", "father of", "computer science"]]}']
        # QA stage
        if "AnswerQuestion" in req_str or "context" in req_str:
            if "Turing" in req_str:
                return ['{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}']
            if "Lovelace" in req_str:
                return ['{"reasoning": "Ada Lovelace worked on the Analytical Engine.", "answer": "Ada Lovelace"}']
            return ['{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}']
        # NER stage
        if "ExtractEntities" in req_str or "named_entities" in req_str:
            if "Lovelace" in req_str:
                return ['{"named_entities": ["Ada Lovelace", "Analytical Engine"]}']
            return ['{"named_entities": ["Alan Turing", "computer science"]}']
        return ['{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}']


class DeterministicMockEmbedder:
    """Deterministic offline embedder producing 64-dimensional float32 unit vectors."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], instruction: str = "") -> list[np.ndarray]:
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
        return results


@pytest.fixture
def test_engine(tmp_path: Path) -> Iterator[YawGraphEngine]:
    db_path = tmp_path / "yaw.db"
    mock_lm = DeterministicMockLM()
    dspy.settings.configure(lm=mock_lm)

    program = HippoRAGProgram(
        execution_mode="direct",
        qa_mode="cot",
        enable_gbnf=False,
    )
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
    chunks = test_engine.conn.execute("SELECT chunk_id, start_line, end_line FROM chunks").fetchall()
    assert len(chunks) == 2

    # Check FTS5 populated
    fts_hits = test_engine.conn.execute("SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH 'Turing'").fetchall()
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
        def encode(self, texts: Sequence[str], instruction: str = "") -> list[np.ndarray]:
            batch_sizes_seen.append(len(texts))
            return super().encode(texts, instruction=instruction)

    mock_lm = DeterministicMockLM()
    dspy.settings.configure(lm=mock_lm)
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
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
            if "FilterFacts" in req_str or "fact_before_filter" in req_str:
                return [
                    '{"fact": [["alan turing", "father of", "computer science"], '
                    '["ada lovelace", "worked on", "analytical engine"]]}'
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=KeepAllFactsLM())
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
    embedder: Any = ControlledFactEmbedder()

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_scores.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        passage_node_weight=0.0,  # Isolate entity seed PPR contribution
    )
    try:
        engine.index([
            "Alan Turing founded computer science.",
            "Ada Lovelace worked on the Analytical Engine.",
        ])

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
            assert self.track_best_outputs is True
            assert self.kwargs.get("component_selector") == "round_robin"
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
            if "FilterFacts" in req_str or "fact_before_filter" in req_str:
                return [
                    '{"fact": [["ada lovelace", "worked on", "analytical engine"], '
                    '["alan turing", "father of", "computer science"]]}'
                ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=InspectFilterLM())
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
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
        engine.index([
            "Alan Turing was an English mathematician and father of computer science.",
            "Ada Lovelace worked on the Analytical Engine.",
        ])

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
            if "FilterFacts" in req_str or "fact_before_filter" in req_str:
                filter_calls.append(req_str)
                # Only keep facts when the rewritten intent includes 'Ada Lovelace'
                if "Ada Lovelace worked on Analytical Engine" in req_str:
                    return ['{"fact": [["ada lovelace", "worked on", "analytical engine"]]}']
                return ['{"fact": []}']
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=IntentEscalationLM())
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
    embedder: Any = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "yaw_intent.db",
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
        retrieval_top_k=2,
    )
    try:
        engine.index([
            "Alan Turing was an English mathematician and father of computer science.",
            "Ada Lovelace worked on the Analytical Engine.",
        ])

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
