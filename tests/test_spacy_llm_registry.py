"""Comprehensive tests for spaCy-LLM canonical model registry, REST client, and Open IE alignment."""

from __future__ import annotations

import ast
import io
import json
import urllib.error
from collections.abc import Iterable
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import spacy
from spacy.tokens import Span
from spacy_llm.registry import registry
from spacy_llm.ty import ModelWithContextLength

from yaw import config
from yaw.nlp import (
    OPEN_IE_REL_TEMPLATE,
    build_leaf_pipeline,
    doc_to_taxonomy,
    embeddings,
    make_ollama_model,
)


def test_model_factory_registration_and_signature():
    """Invariant 1: Verify model factory is registered in registry.llm_models and satisfies ModelWithContextLength."""
    assert "yaw.Ollama.v1" in registry.llm_models
    assert "yaw.TestModelWrapper.v1" in registry.llm_models

    model = make_ollama_model(
        name="test-model",
        timeout_s=10.0,
        temperature=0.0,
        context_length=4096,
        max_tries=3,
        interval=0.1,
    )

    assert hasattr(model, "context_length")
    assert model.context_length == 4096
    assert isinstance(model, ModelWithContextLength)
    assert callable(model)


def test_declarative_pipeline_assembly_without_monkeypatch():
    """Invariant 2: Blank spaCy pipeline successfully adds llm pipe via pure declarative config."""
    nlp = spacy.blank("en")
    pipe: Any = nlp.add_pipe(
        "llm",
        name="llm_ner",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v3",
                "labels": ["concepts", "claims", "invariants"],
            },
            "model": {"@llm_models": "yaw.Ollama.v1"},
        },
    )

    assert pipe.name == "llm_ner"
    assert "llm_ner" in nlp.pipe_names
    assert set(pipe.labels) == {"concepts", "claims", "invariants"}


def test_loopback_http_chat_schema_conformance(mock_ollama_requests, tmp_path: Path):
    """Invariant 3: Outgoing HTTP payloads match Ollama /api/chat schema."""
    nlp = build_leaf_pipeline(cache_dir=tmp_path / "cache")
    doc = nlp("Albert Einstein developed general relativity at Princeton University.")

    assert doc is not None
    assert len(mock_ollama_requests) >= 1

    first_req = mock_ollama_requests[0]
    assert "model" in first_req
    assert first_req["model"] == config.CHAT_MODEL
    assert "messages" in first_req
    assert isinstance(first_req["messages"], list)
    assert len(first_req["messages"]) > 0

    msg = first_req["messages"][0]
    assert msg.get("role") == "user"
    assert isinstance(msg.get("content"), str)
    assert len(msg.get("content")) > 0

    assert first_req.get("stream") is False
    assert "options" in first_req
    assert "temperature" in first_req["options"]
    assert "num_ctx" in first_req["options"]


def test_open_ie_relation_template_parses_and_populates_doc_rel():
    """Invariant 4: OPEN_IE_REL_TEMPLATE parses without Jinja errors and extracts relations into doc._.rel."""

    class _MockRelCallable:
        @property
        def context_length(self) -> int:
            return 8192

        def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
            responses: list[list[str]] = []
            for doc_prompts in prompts:
                responses.append(['{"dep": 0, "dest": 1, "relation": "developed"}'])
            return responses

    from yaw.nlp.extractor import _model_to_config

    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        name="llm_rel",
        config={
            "validate_types": False,
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                "labels": [],
                "template": OPEN_IE_REL_TEMPLATE,
            },
            "model": _model_to_config(_MockRelCallable()),
        },
    )

    doc = nlp.make_doc("Albert Einstein developed general relativity.")
    doc.ents = [
        Span(doc, 0, 2, label="PERSON"),
        Span(doc, 3, 5, label="THEORY"),
    ]

    processed = nlp(doc)
    assert hasattr(processed._, "rel")
    assert len(processed._.rel) == 1

    rel_item = processed._.rel[0]
    assert rel_item.dep == 0
    assert rel_item.dest == 1
    assert rel_item.relation == "developed"

    tax = doc_to_taxonomy(processed, nlp=nlp)
    assert len(tax.triples) == 1
    triple = tax.triples[0]
    assert triple.subject.name == "Albert Einstein"
    assert triple.predicate.name == "develop"
    assert triple.object.name == "general relativity"


def test_ollama_model_retry_exponential_backoff_and_fast_fail():
    """Invariant 5: Exponential backoff retries transient 5xx/network errors and fast-fails on 4xx client errors."""
    # Test 1: Transient 503 error that recovers on second try
    call_count = 0

    def mock_urlopen_recovering(req, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            fp = io.BytesIO(b'{"error": "service temporarily unavailable"}')
            raise urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", Message(), fp
            )
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"message": {"content": "recovered_content"}}'
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    model = make_ollama_model(
        endpoint="http://127.0.0.1:11434/api/chat",
        max_tries=3,
        interval=0.01,
        backoff_factor=1.5,
    )

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_recovering):
        res = model([["test prompt"]])
        assert call_count == 2
        assert res == [["recovered_content"]]

    # Test 2: Persistent error exceeding max_tries raises ConnectionError
    persistent_count = 0

    def mock_urlopen_persistent(req, timeout):
        nonlocal persistent_count
        persistent_count += 1
        fp = io.BytesIO(b'{"error": "bad gateway"}')
        raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", Message(), fp)

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_persistent):
        with pytest.raises(ConnectionError, match="failed after 3 attempts"):
            model([["test prompt"]])
        assert persistent_count == 3

    # Test 3: Client HTTP 400 error fails fast on attempt 1 without retrying
    client_err_count = 0

    def mock_urlopen_client_error(req, timeout):
        nonlocal client_err_count
        client_err_count += 1
        fp = io.BytesIO(b'{"error": "invalid model parameter"}')
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", Message(), fp)

    with patch("urllib.request.urlopen", side_effect=mock_urlopen_client_error):
        with pytest.raises(RuntimeError, match="Ollama client HTTP 400 error"):
            model([["test prompt"]])
        assert client_err_count == 1


def test_zero_private_attribute_access_in_nlp_package():
    """Invariant 6: Strict audit for zero occurrences of ._model or ._task across src/yaw/nlp/."""
    nlp_dir = Path(__file__).resolve().parent.parent / "src" / "yaw" / "nlp"
    assert nlp_dir.is_dir()

    violating_lines: list[str] = []
    for py_file in nlp_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("_model", "_task"):
                violating_lines.append(
                    f"{py_file.name}:{node.lineno} uses private attribute .{node.attr}"
                )

    assert not violating_lines, (
        f"Private attribute violations detected: {violating_lines}"
    )


def test_embedding_gemma_prefix_ssot_and_purge_rerank_query_vector():
    """Invariant 7: Reranking query vector is purged and embed_query uses canonical Retrieval-query prefix."""
    import yaw.nlp

    assert hasattr(yaw.nlp, "embed_query")
    assert not hasattr(yaw.nlp, "rerank_query_vector")
    assert not hasattr(embeddings, "rerank_query_vector")

    assert (
        embeddings.PROMPT_PREFIXES["Retrieval-query"] == "task: search result | query: "
    )
    formatted = embeddings.format_prompt("quantum teleportation", "Retrieval-query")
    assert formatted == "task: search result | query: quantum teleportation"


def test_test_model_wrapper_clean_injection_in_build_leaf_pipeline(tmp_path: Path):
    """Invariant 8: Test callables inject into build_leaf_pipeline via yaw.TestModelWrapper.v1 without private mutations."""
    executed_prompts: list[str] = []

    class _MockInjectedModel:
        @property
        def context_length(self) -> int:
            return 8192

        def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
            res: list[list[str]] = []
            for batch in prompts:
                doc_res: list[str] = []
                for p in batch:
                    executed_prompts.append(str(p))
                    doc_res.append(
                        json.dumps(
                            [
                                {
                                    "subject": "Niels Bohr",
                                    "predicate": "collaborated_with",
                                    "object": "Werner Heisenberg",
                                }
                            ]
                        )
                    )
                res.append(doc_res)
            return res

    mock_model = _MockInjectedModel()
    nlp = build_leaf_pipeline(model=mock_model, cache_dir=tmp_path / "cache")

    raw_pipe: Any = nlp.get_pipe("llm_raw")
    assert raw_pipe is not None
    assert "llm_raw" in nlp.pipe_names

    doc = nlp("Niels Bohr and Werner Heisenberg met in Copenhagen.")
    assert len(executed_prompts) > 0
    assert doc._.leaf_taxonomy is not None
    assert len(doc._.leaf_taxonomy.triples) == 1
    assert doc._.leaf_taxonomy.triples[0].subject.name == "Niels Bohr"
    assert doc._.leaf_taxonomy.triples[0].predicate.name == "collaborate with"
    assert doc._.leaf_taxonomy.triples[0].object.name == "Werner Heisenberg"
