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
        # Triples stage
        if "ExtractTriples" in req_str or "triples" in req_str:
            if "Lovelace" in req_str:
                return [
                    '{"triples": [{"sub": "Ada Lovelace", "pred": "worked on", "obj": "Analytical Engine", "def": "Ada Lovelace worked on the Analytical Engine."}]}'
                ]
            return [
                '{"triples": [{"sub": "Alan Turing", "pred": "father of", "obj": "computer science", "def": "Alan Turing is the father of computer science."}]}'
            ]
        # Filter facts stage
        if "FilterFacts" in req_str or "fact_before_filter" in req_str or "filter" in req_str.lower():
            if "Lovelace" in req_str:
                return ['{"fact": [["Ada Lovelace", "worked on", "Analytical Engine"]]}']
            return ['{"fact": [["Alan Turing", "father of", "computer science"]]}']
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
        if "AnswerQuestion" in req_str or "Retrieved evidence passages" in req_str or "context" in req_str:
            if "Turing" in req_str:
                return ['{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}']
            if "Lovelace" in req_str:
                return ['{"reasoning": "Ada Lovelace worked on the Analytical Engine.", "answer": "Ada Lovelace"}']
            return ['{"reasoning": "Alan Turing founded computer science.", "answer": "Alan Turing"}']
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
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
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
        n2 = engine.index(["The apple is an edible pome fruit harvested in orchard trees during Autumn."])
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
                                    ["Apple", "Multinational personal computer corporation in California"],
                                    ["MacBook", "Line of Macintosh portable computers"],
                                    ["M3 Max", "System on a chip powering MacBook Pro laptops"],
                                ]
                            }
                        )
                    ]
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    dspy.settings.configure(lm=SynonymMergeAndStandaloneNerLM())
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
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
        report = engine.lint_graph()
        assert report["conforms"] is True
        assert report["violations"] == []
    finally:
        engine.close()


def test_engine_step7_heal_shacl_violations_rlm_repl_tools_and_token_stats(
    tmp_path: Path,
) -> None:
    """Verify Step 6 synonym_judge_tokens and Step 7 heal_shacl_violations in RLM mode using all 5 bound REPL tools without nested track_usage() double-counting."""
    import json
    import re

    repl_tool_events: list[str] = []

    class RLMOrchestratorLM(dspy.LM):
        """Heterogeneous NPU orchestrator LM (40 tokens per call, ChatAdapter, no GBNF)."""

        def __init__(self) -> None:
            super().__init__(model="mistral-orchestrator")
            self._yaw_recipe = "ryzenai-llm"
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
            if "qwen_judge_synonyms" in req_str:
                code = (
                    "decisions = qwen_judge_synonyms(candidate_pairs=candidate_pairs)\n"
                    "SUBMIT(decisions=decisions)"
                )
                return [
                    f"[[ ## reasoning ## ]]\nDelegate synonym judgment to Qwen worker.\n\n"
                    f"[[ ## code ## ]]\n```python\n{code}\n```\n\n[[ ## completed ## ]]"
                ]
            if "qwen_extract_triples" in req_str:
                code = (
                    "triples = qwen_extract_triples(passage=passage, named_entities=named_entities)\n"
                    "SUBMIT(triples=triples)"
                )
                return [
                    f"[[ ## reasoning ## ]]\nDelegate triple extraction to Qwen worker.\n\n"
                    f"[[ ## code ## ]]\n```python\n{code}\n```\n\n[[ ## completed ## ]]"
                ]
            code = (
                "entities = qwen_extract_entities(passage=passage)\n"
                "SUBMIT(named_entities=entities)"
            )
            return [
                f"[[ ## reasoning ## ]]\nDelegate entity extraction to Qwen worker.\n\n"
                f"[[ ## code ## ]]\n```python\n{code}\n```\n\n[[ ## completed ## ]]"
            ]

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
            if "[[ ## violations ## ]]" in req_str or "bounded closed-world SHACL repair" in req_str:
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
            if "[[ ## named_entities ## ]]" in req_str or "Resource Description Framework" in req_str:
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
        execution_mode="direct",
        qa_mode="rlm",
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

        # 2. Switch to RLM mode and index second document:
        #    - Step 6 runs synonym_judge_rlm on candidate pairs
        #    - Step 7 runs shacl_healer_rlm using all 5 bound REPL tools (applying repair[0] in REPL + repair[1] post-RLM)
        program.execution_mode = "rlm"
        program.telemetry.clear()
        program.synonym_audit_trail.clear()
        program.shacl_audit_trail.clear()

        assert (
            engine.index(["Alan M. Turing worked on cryptanalysis at Bletchley Park."])
            == 1
        )

        # Verify all 5 REPL tools ran inside shacl_healer_rlm and healed both violations
        assert "shacl_healer_rlm_invoked" in repl_tool_events
        assert sum(1 for ev in repl_tool_events if ev.startswith("qwen_propose_shacl_repairs:")) == 2

        # Verify exact non-double-counted token accounting (40 orchestrator + 30 worker = 70 tokens per RLM stage)
        assert len(program.synonym_audit_trail) == 2  # inner direct (30) + outer RLM (70)
        assert program.synonym_audit_trail[0]["mode"] == "direct"
        assert program.synonym_audit_trail[0]["tokens_used"] == 30
        assert program.synonym_audit_trail[-1]["mode"] == "rlm"
        assert program.synonym_audit_trail[-1]["tokens_used"] == 70
        assert engine._graph_rlm_stats["synonym_judge_tokens"] == 100  # 30 (doc 1 direct) + 70 (doc 2 RLM)

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
    program = HippoRAGProgram(execution_mode="direct", qa_mode="cot", enable_gbnf=False)
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
            for r in engine.conn.execute(
                "SELECT name, definition FROM entities"
            ).fetchall()
        }
        assert rows["ada lovelace"] == "English mathematician and writer"
        assert (
            rows["charles babbage"]
            == "English polymath who originated the programmable computer"
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
