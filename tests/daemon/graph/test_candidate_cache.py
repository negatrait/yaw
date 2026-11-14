"""Single-owner CocoIndex LMDB memoization and bounded in-memory query embedding LRU cache."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from embedding_fixture import IdentifiedEmbeddingCallable

from daemon.graph.synonym_protocol import candidate_rows, complete_decisions
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    FACT_INSTRUCTION,
    PASSAGE_INSTRUCTION,
    DSPyEmbeddingAdapter,
    YawProgram,
    create_dspy_embedder,
    resolve_embedding_profile,
)
from yaw.daemon.indexing import bind_native_indexing


def test_query_embedding_lru_cache_and_absence_of_retired_cache_tables(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.array(
            [[(b + 1) / 256 for b in hashlib.sha256(t.encode("utf-8")).digest()[:8]] for t in texts],
            dtype=np.float32,
        )

    program = YawProgram(enable_gbnf=False)
    embedder = DSPyEmbeddingAdapter(
        replace(
            resolve_embedding_profile(),
            dimensions=8,
            fact_query_prefix="task: fact search | query: ",
        ),
        create_dspy_embedder(IdentifiedEmbeddingCallable(embed, "candidate-cache-sha256-v1"), caching=False),
    )
    engine = bind_native_indexing(
        YawGraphEngine(
            db_path=tmp_path / "yaw.db",
            workspace_root=tmp_path,
            program=program,
            embedder=embedder,
        )
    )
    try:
        tables = {row[0] for row in engine.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {t for t in tables if "cache" in t} == set()

        v1 = engine._encode_texts(["Where did Turing work?"], instruction=PASSAGE_INSTRUCTION)[0]
        v2 = engine._encode_texts(["Where did Turing work?"], instruction=PASSAGE_INSTRUCTION)[0]
        np.testing.assert_array_equal(v1, v2)
        assert len(calls) == 2

        fact_vec = engine._encode_texts(["Where did Turing work?"], instruction=FACT_INSTRUCTION)[0]
        assert len(calls) == 3
        assert not np.array_equal(v1, fact_vec)

        program.embedding_query_prefix = "task: custom query | query: "
        engine._apply_program_embedding_prefixes(program)
        v3 = engine._encode_texts(["Where did Turing work?"], instruction=PASSAGE_INSTRUCTION)[0]
        assert len(calls) == 4
        assert not np.array_equal(v1, v3)
    finally:
        engine.close()


def test_candidate_stages_make_zero_requests_after_process_restart(tmp_path: Path) -> None:
    requests: list[str] = []
    # Keep both subprocesses on exactly the same implementation even when a
    # developer edits the checkout while this relatively expensive test runs.
    source_root = tmp_path / "implementation"
    shutil.copytree(
        Path(__file__).resolve().parents[3] / "yaw",
        source_root / "yaw",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            assert isinstance(self.server, ThreadingHTTPServer)
            if self.path.endswith("/health"):
                self._send_json(
                    {
                        "all_models_loaded": [
                            {
                                "id": model_id,
                                "backend_url": f"http://127.0.0.1:{self.server.server_port}",
                                "launch_command": ["server", "-m", "C:/models/cache-test.gguf"],
                            }
                            for model_id in ("cache-test", "cache-test-v2")
                        ]
                    }
                )
            elif self.path == "/props":
                self._send_json(
                    {
                        "model_path": "C:/models/cache-test.gguf",
                        "chat_template": "test-template",
                        "default_generation_settings": {"n_ctx": 131072},
                    }
                )
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                requests.append("embedding")
                texts = data["input"] if isinstance(data["input"], list) else [data["input"]]
                payload = {
                    "object": "list",
                    "model": data["model"],
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [(b + 1) / 256 for b in hashlib.sha256(t.encode()).digest() * 24],
                        }
                        for i, t in enumerate(texts)
                    ],
                    "usage": {"prompt_tokens": 20, "total_tokens": 20},
                }
            else:
                messages = data["messages"]
                prompt = str(messages)
                user = str(messages[-1]["content"])
                if "variables_info" in prompt and "repl_history" in prompt:
                    requests.append("rlm")
                    code = "repairs = propose_repairs()\nSUBMIT(repairs=repairs)"
                    content = (
                        "[[ ## reasoning ## ]]\nDelegate repairs to the worker.\n"
                        f"[[ ## code ## ]]\n```python\n{code}\n```\n[[ ## completed ## ]]"
                    )
                elif "[[ ## violations ## ]]" in user:
                    requests.append("shacl")
                    ids = sorted(set(re.findall(r"entity-[0-9a-f]+", user)))
                    content = json.dumps(
                        {
                            "repairs": [
                                {
                                    "focus_node": eid,
                                    "action": "set_definition",
                                    "value": "A grounded definition restored by the SHACL worker.",
                                }
                                for eid in ids
                            ]
                        }
                    )
                elif "[[ ## candidate_pairs ## ]]" in user:
                    requests.append("synonym")
                    content = json.dumps({"decisions": complete_decisions(candidate_rows(user), [], stage_text=prompt)})
                elif "[[ ## missing_entities ## ]]" in user:
                    requests.append("definition")
                    content = json.dumps({"entity_definitions": []})
                elif "[[ ## named_entities ## ]]" in user:
                    requests.append("triples")
                    if "Bletchley Park" in user:
                        triple = {
                            "sub": "Alan M. Turing",
                            "pred": "worked at",
                            "obj": "Bletchley Park",
                            "def": "A wartime cryptanalyst.",
                        }
                    else:
                        triple = {
                            "sub": "Alan Turing",
                            "pred": "formulated",
                            "obj": "Turing machine",
                            "def": "A model of computation.",
                        }
                    content = json.dumps({"triples": [triple]})
                else:
                    requests.append("ner")
                    entities = (
                        [
                            ["Alan M. Turing", "British cryptanalyst"],
                            ["Bletchley Park", "Wartime codebreaking center"],
                        ]
                        if "Bletchley Park" in user
                        else [
                            ["Alan Turing", "English mathematician"],
                            ["Turing machine", "A model of computation"],
                        ]
                    )
                    content = json.dumps({"named_entities": entities})
                payload = {
                    "id": "cache-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": data["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                }
            self._send_json(payload)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    def run_worker(url: str, mutation: str = "none") -> dict[str, Any]:
        env = dict(os.environ, PYTHONPATH=str(source_root))
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("candidate_cache_worker.py")),
                str(tmp_path),
                url,
                "--mutation",
                mutation,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert Path(report["implementation_root"]).is_relative_to(source_root)
        return report

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1"
            first = run_worker(url)
            assert {"embedding", "ner", "triples", "synonym"} <= set(requests)
            before = len(requests)
            second = run_worker(url)
            if len(requests) != before:
                raise AssertionError(
                    json.dumps(
                        {
                            "requests": requests[before:],
                            "first": first,
                            "second": second,
                        },
                        indent=2,
                    )
                )
            assert second["tokens"] == first["tokens"]
            assert second["openie_tokens"] == first["openie_tokens"]
            assert second["trace_names"] == first["trace_names"]
            print(
                json.dumps(
                    {
                        "initial_requests": before,
                        "replay_requests": len(requests) - before,
                        "preserved_tokens": second["tokens"],
                        "preserved_openie_tokens": second["openie_tokens"],
                    }
                )
            )
            run_worker(url, "query")
            assert requests[before:] == ["embedding", "embedding", "embedding"]
            before = len(requests)
            run_worker(url, "prompt")
            assert "ner" in requests[before:]
            before = len(requests)
            run_worker(url, "model")
            assert "ner" in requests[before:]
            before = len(requests)
            run_worker(url, "sampling")
            assert "ner" in requests[before:]
            before = len(requests)
            run_worker(url, "source")
            assert "ner" in requests[before:]
        finally:
            server.shutdown()
            thread.join(timeout=5)
