import numpy as np
import pytest

from yaw.nlp import embeddings as embedding_gemma


def test_embeddinggemma_prompt_formatting():
    """Verify exact task prompt prefixes match EmbeddingGemma model specification."""
    doc_with_title = embedding_gemma.format_document("Doc body", title="Network Policy")
    assert doc_with_title == "title: Network Policy | text: Doc body"

    doc_no_title = embedding_gemma.format_document("Doc body")
    assert doc_no_title == "title: none | text: Doc body"

    query_prompt = embedding_gemma.format_prompt("what is ppr?", "Retrieval-query")
    assert query_prompt == "task: search result | query: what is ppr?"

    rerank_prompt = embedding_gemma.format_prompt("what is ppr?", "Reranking")
    assert rerank_prompt == "task: search result | query: what is ppr?"


def test_embeddinggemma_biencoder_reranking_seed_docs(monkeypatch):
    """Verify bi-encoder reranking with prompt Reranking deterministically ranks matching seed documents #1."""
    from conftest import _hash_embed

    async def fake_embed(texts, **kw):
        return [np.array(_hash_embed(t), dtype=np.float32) for t in texts]

    monkeypatch.setattr(embedding_gemma, "embed_texts", fake_embed)

    base_candidates = [
        {
            "id": "doc_neo4j",
            "title": "CocoIndex Neo4j Target",
            "content": "CocoIndex connector for Neo4j maps extracted entities and relationships into a property graph database using Cypher queries.",
            "channels": {},
            "graph_prob": 1.0,
            "vector": np.array(
                _hash_embed(
                    embedding_gemma.format_document(
                        "CocoIndex connector for Neo4j maps extracted entities and relationships into a property graph database using Cypher queries.",
                        title="CocoIndex Neo4j Target",
                    )
                ),
                dtype=np.float32,
            ),
        },
        {
            "id": "doc_ppr",
            "title": "Personalized PageRank Diffusion",
            "content": "Personalized PageRank computes graph diffusion over a SciPy CSR sparse matrix using power iteration until stationary distribution convergence.",
            "channels": {},
            "graph_prob": 1.0,
            "vector": np.array(
                _hash_embed(
                    embedding_gemma.format_document(
                        "Personalized PageRank computes graph diffusion over a SciPy CSR sparse matrix using power iteration until stationary distribution convergence.",
                        title="Personalized PageRank Diffusion",
                    )
                ),
                dtype=np.float32,
            ),
        },
        {
            "id": "doc_karpathy",
            "title": "Karpathy LLM-Wiki Pattern",
            "content": "Karpathy's LLM-wiki pattern separates raw immutable notes from compiled wiki pages, maintaining an evolving knowledge base with markdown interlinks.",
            "channels": {},
            "graph_prob": 1.0,
            "vector": np.array(
                _hash_embed(
                    embedding_gemma.format_document(
                        "Karpathy's LLM-wiki pattern separates raw immutable notes from compiled wiki pages, maintaining an evolving knowledge base with markdown interlinks.",
                        title="Karpathy LLM-Wiki Pattern",
                    )
                ),
                dtype=np.float32,
            ),
        },
    ]

    queries = [
        ("SciPy CSR sparse matrix power iteration", "doc_ppr"),
        ("Neo4j Cypher property graph connector", "doc_neo4j"),
        ("Karpathy compiled markdown vault wiki", "doc_karpathy"),
    ]

    for q, expected in queries:
        test_candidates = [
            {
                "id": c["id"],
                "title": c["title"],
                "content": c["content"],
                "channels": {},
                "graph_prob": 1.0,
                "vector": c["vector"],
            }
            for c in base_candidates
        ]
        reranked = pytest.importorskip("asyncio").run(
            embedding_gemma.rerank_candidates(q, test_candidates)
        )
        assert reranked[0]["id"] == expected, (
            f"Query '{q}' failed: expected {expected}, got {[r['id'] for r in reranked]}"
        )
