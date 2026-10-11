import hashlib
import itertools
import json
import os
import re
import sqlite3
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

for _k in ("no_proxy", "NO_PROXY"):
    os.environ[_k] = (
        f"{os.environ[_k]},127.0.0.1,localhost"
        if _k in os.environ
        else "127.0.0.1,localhost"
    )


def _hash_embed(text: str, dim: int = 768) -> list[float]:
    words = re.findall(r"\w+", text.lower())
    content_words = [
        w
        for w in words
        if w not in ("task", "search", "result", "query", "title", "text", "none")
    ]

    lex = np.zeros(dim, dtype=np.float32)
    for w in content_words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
        lex[h] += 1.0
    for w1, w2 in itertools.pairwise(content_words):
        h = int(hashlib.md5(f"{w1}_{w2}".encode()).hexdigest(), 16) % dim
        lex[h] += 2.0

    # Project lex to zero-mean (orthogonal to constant base)
    lex = lex - np.mean(lex)
    norm = float(np.linalg.norm(lex))
    if norm > 1e-12:
        lex /= norm

    alpha = 0.86
    base = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32)
    vec = alpha * base + np.sqrt(1.0 - alpha**2) * lex
    vec /= np.linalg.norm(vec)
    return vec.tolist()


_MOCK_ENGINE_TERMS = frozenset({"fastmcp", "sqlite_vec", "pytest", "spacy_llm"})


def _mock_canon_adjudication(prompt: str) -> str | None:
    if not (
        "adjudicating entity canonicalization" in prompt
        or "Candidate Canonical Entity:" in prompt
        or "Surface Mention:" in prompt
    ):
        return None

    if BaseTestHTTPHandler.canon_verdict_override is not None:
        verdict = BaseTestHTTPHandler.canon_verdict_override
        if verdict == "distinct" or verdict is None or verdict is False:
            return json.dumps(
                {
                    "match": None,
                    "confidence": 0.0,
                    "reasoning": "Explicit distinct verdict override.",
                }
            )
        return json.dumps(
            {
                "match": str(verdict),
                "confidence": 0.95,
                "reasoning": f"Explicit match override for {verdict}.",
            }
        )

    cand_id_m = re.search(r"-\s*ID:\s*([^\s\n\r]+)", prompt)
    cand_id = cand_id_m.group(1).strip() if cand_id_m else "concepts:entropy"
    surf_m = re.search(r"Surface Mention:\s*([^\n\r]+)", prompt)
    surf_name = surf_m.group(1).strip() if surf_m else ""

    if any(
        term in surf_name.lower()
        for term in (
            "distinct",
            "different",
            "unrelated",
            "mismatch",
            "nonmatch",
            "novel",
        )
    ):
        return json.dumps(
            {
                "match": None,
                "confidence": 0.0,
                "reasoning": f"Surface mention '{surf_name}' is distinct from candidate '{cand_id}'.",
            }
        )
    return json.dumps(
        {
            "match": cand_id,
            "confidence": 0.95,
            "reasoning": f"Surface mention '{surf_name}' matches candidate '{cand_id}'.",
        }
    )


def _mock_hypernym_inference(prompt: str) -> str | None:
    if not (
        "hierarchical hypernym induction" in prompt
        or "intermediate hypernym superclass concept" in prompt
        or "Cluster Member Concepts:" in prompt
    ):
        return None

    if BaseTestHTTPHandler.hypernym_override is not None:
        return json.dumps(BaseTestHTTPHandler.hypernym_override)
    parent_m = re.search(r"Parent Concept:\s*[\r\n]+\s*-\s*Name:\s*([^\n\r]+)", prompt)
    parent_name = parent_m.group(1).strip() if parent_m else "Information Theory"
    hypernym_name = f"{parent_name} Foundation" if parent_name else "Core Concept"
    definition = f"Superclass concept categorizing member concepts under {parent_name}."
    return json.dumps(
        {
            "hypernym_name": hypernym_name,
            "definition": definition,
        }
    )


def _extract_text_entity_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for wl in re.findall(r"\[\[(.*?)\]\]", text):
        cand = wl.split("|")[-1].strip()
        if (
            cand in text
            and cand not in candidates
            and len(cand) >= 2
            and cand.lower() not in _MOCK_ENGINE_TERMS
        ):
            candidates.append(cand)
    for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b", text):
        cand = m.group().strip()
        if (
            cand in text
            and cand not in candidates
            and len(cand) >= 2
            and cand.lower() not in _MOCK_ENGINE_TERMS
        ):
            candidates.append(cand)
    return candidates


def _classify_spancat_label(cand: str) -> str:
    concept_terms = (
        "mechanic",
        "principle",
        "theory",
        "law",
        "system",
        "probability",
        "model",
        "algorithm",
        "definition",
        "formulation",
        "concept",
    )
    c_lower = cand.lower()
    if any(term in c_lower for term in concept_terms):
        return "concepts"
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", cand) or (
        cand[0].isupper()
        and not any(term in c_lower for term in ("note", "raw", "wiki"))
    ):
        return "entities"
    return "concepts"


def _mock_open_ie_triples(prompt: str) -> str | None:
    if not (
        "extract all factual, directed relational triples" in prompt
        or "Each triple consists of:" in prompt
        or '{"subject": "<subject>"' in prompt
    ):
        return None

    if BaseTestHTTPHandler.raw_triples_override is not None:
        triples = BaseTestHTTPHandler.raw_triples_override
    else:
        text_matches = re.findall(r"Text:\s*'''(.*?)'''", prompt, flags=re.DOTALL)
        text = text_matches[-1].strip() if text_matches else prompt
        candidates = _extract_text_entity_candidates(text)

        triples = []
        if len(candidates) >= 2:
            for i in range(min(len(candidates) - 1, 3)):
                triples.append(
                    {
                        "subject": candidates[i],
                        "predicate": "relates to",
                        "object": candidates[i + 1],
                        "subject_type": "concepts",
                        "object_type": "concepts",
                    }
                )
        else:
            words = [
                w
                for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)
                if w.lower() not in _MOCK_ENGINE_TERMS
            ]
            if len(words) >= 2:
                triples.append(
                    {
                        "subject": words[0],
                        "predicate": "relates to",
                        "object": words[1],
                        "subject_type": "concepts",
                        "object_type": "concepts",
                    }
                )
            else:
                triples.append(
                    {
                        "subject": "entropy",
                        "predicate": "measures",
                        "object": "uncertainty",
                        "subject_type": "concepts",
                        "object_type": "concepts",
                    }
                )
    return f"```json\n{json.dumps(triples, indent=2)}\n```"


def _mock_spancat(prompt: str) -> str | None:
    if not (
        "expert Entity Recognition system" in prompt
        or "identify a list of entities" in prompt
    ):
        return None

    text = (
        prompt.split("Paragraph:")[-1].split("Answer:")[0].strip()
        if "Paragraph:" in prompt
        else prompt
    )
    candidates = _extract_text_entity_candidates(text)
    if candidates:
        lines = [
            f"{i + 1}. {c} | True | {_classify_spancat_label(c)} | mock {_classify_spancat_label(c)}"
            for i, c in enumerate(candidates[:10])
        ]
        return "\n".join(lines)
    return "{}"


def _mock_relations(prompt: str) -> str | None:
    if not (
        "extract all meaningful, directed semantic relations between entities" in prompt
        or "pre-extracted entities" in prompt
    ):
        return None

    ent_ids = re.findall(r"\[ENT(\d+):", prompt)
    if len(ent_ids) >= 2:
        rel_lines = [
            f'{{"dep": {ent_ids[i]}, "dest": {ent_ids[i + 1]}, "relation": "relates_to"}}'
            for i in range(min(len(ent_ids) - 1, 5))
        ]
        return "\n".join(rel_lines)
    return "{}"


def _mock_textcat(prompt: str) -> str | None:
    if "Classify the following text into a single concise category label" in prompt:
        return "concepts"
    return None


def _mock_chat_response(data: dict) -> str:
    messages = data.get("messages", [])
    prompt = messages[-1].get("content", "") if messages else ""
    if not prompt and "prompt" in data:
        prompt = str(data["prompt"])

    for handler in (
        _mock_canon_adjudication,
        _mock_hypernym_inference,
        _mock_open_ie_triples,
        _mock_spancat,
        _mock_relations,
        _mock_textcat,
    ):
        res = handler(prompt)
        if res is not None:
            return res

    return "{}"


class BaseTestHTTPHandler(BaseHTTPRequestHandler):
    received_requests: ClassVar[list[dict]] = []
    embed_status_code: ClassVar[int] = 200
    canon_verdict_override: ClassVar[str | None] = None
    raw_triples_override: ClassVar[list[dict] | None] = None
    hypernym_override: ClassVar[dict[str, str] | None] = None

    @classmethod
    def reset_requests(cls) -> None:
        cls.received_requests.clear()
        cls.canon_verdict_override = None
        cls.raw_triples_override = None
        cls.hypernym_override = None

    def do_GET(self):
        if self.path in ("/api/version", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"version": "0.1.32"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.embed_status_code != 200:
            self.send_response(self.embed_status_code)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Internal Server Error")
            return

        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        if self.path in ("/api/embed", "/api/embeddings"):
            BaseTestHTTPHandler.received_requests.append(data)
            inputs = data.get("input", data.get("inputs", []))
            if isinstance(inputs, str):
                inputs = [inputs]
            embeddings = [_hash_embed(t) for t in inputs] if inputs else [[0.05] * 768]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "embeddings": embeddings,
                        "embedding": embeddings[0] if embeddings else [0.05] * 768,
                    }
                ).encode("utf-8")
            )
        elif self.path == "/api/chat":
            BaseTestHTTPHandler.received_requests.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            content = _mock_chat_response(data)
            self.wfile.write(
                json.dumps({"message": {"content": content}}).encode("utf-8")
            )
        elif self.path == "/api/generate":
            BaseTestHTTPHandler.received_requests.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            content = _mock_chat_response(data)
            self.wfile.write(
                json.dumps({"response": content, "done": True}).encode("utf-8")
            )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


_MockOllamaHandler = BaseTestHTTPHandler


@pytest.fixture(scope="session", autouse=True)
def _ensure_ollama_mock():
    server = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    port = server.server_port
    os.environ["YAW_HOST"] = f"http://127.0.0.1:{port}"
    from yaw import config

    config.reload()
    config.OLLAMA_HOST = f"http://127.0.0.1:{port}"
    config.HOST = f"http://127.0.0.1:{port}"

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield
    server.shutdown()


@pytest.fixture
def mock_ollama_requests():
    """Provides inspectable record of outbound Ollama requests, automatically reset per test."""
    _MockOllamaHandler.reset_requests()
    yield _MockOllamaHandler.received_requests
    _MockOllamaHandler.reset_requests()


@pytest.fixture(autouse=True)
def restore_vault():
    """Autouse fixture resetting VAULT_ROOT after each test across the entire suite."""
    from yaw import config

    old_root = config.VAULT_ROOT
    try:
        yield
    finally:
        config.set_vault(old_root)


def get_cli_env() -> dict[str, str]:
    """Build isolated environment for yaw CLI subprocess execution."""
    env = os.environ.copy()
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir
    env["FASTMCP_SHOW_SERVER_BANNER"] = "false"
    return env


@pytest.fixture
def test_db(tmp_path: Path) -> Generator[sqlite3.Connection]:
    """Provide fully-migrated, isolated SQLite database with sqlite-vec and unified views."""
    from yaw.storage.db import (
        apply_pragmas,
        ensure_schema_tables,
        ensure_schema_views,
        load_vec_extension,
    )

    db_path = tmp_path / "yaw.db"
    conn = sqlite3.connect(db_path)
    apply_pragmas(conn)
    load_vec_extension(conn)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_nodes (
            node_id TEXT NOT NULL,
            source_leaf_id TEXT NOT NULL,
            name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            definition TEXT,
            embedding BLOB,
            PRIMARY KEY (node_id, source_leaf_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_relations (
            relation_id TEXT PRIMARY KEY,
            source_leaf_id TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            predicate_id TEXT,
            object_id TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wikilinks (
            source_uri TEXT NOT NULL,
            target_uri TEXT NOT NULL,
            link_label TEXT,
            PRIMARY KEY (source_uri, target_uri)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_chunks (
            chunk_id TEXT NOT NULL,
            entity_uri TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            ordinal INT NOT NULL,
            heading TEXT,
            heading_slug TEXT,
            line_start INT,
            line_end INT,
            embedding BLOB
        )
        """
    )

    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    conn.commit()

    try:
        yield conn
    finally:
        conn.close()
