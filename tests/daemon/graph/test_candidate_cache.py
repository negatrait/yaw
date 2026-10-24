"""Typed disk-cache replay and actual inference requests across separate processes."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import dspy
import numpy as np
from dspy.primitives.repl_types import REPLHistory

from yaw.core import OpenIERecord, StructuredTriple
from yaw.daemon.graph.candidate_cache import CandidateStageCache
from yaw.daemon.graph.storage import CandidateCacheStore, open_managed_yaw_db


def test_stage_cache_roundtrips_typed_traces_without_storing_passage(tmp_path: Path) -> None:
    managed = open_managed_yaw_db(tmp_path / "yaw.db")
    try:
        # The test reads the actual serialized cache row, not a mocked serializer.
        with managed.transaction() as conn:
            store = CandidateCacheStore(conn)
        cache = CandidateStageCache(store, managed, "openie", lambda: "version-one")
        text = "Lähdeteksti: a source passage that must be hydrated.\nKeep the second line."
        record = OpenIERecord(
            "chunk",
            ("Ada",),
            (("Ada", "wrote", "Notes"),),
            (StructuredTriple("Ada", "Author", "wrote", "Notes", "Publication", "Ada wrote Notes"),),
            (("Ada", "Author"),),
        )
        value = (
            record,
            [("ner", {"passage": text}, dspy.Prediction(named_entities=["Ada"]))],
            {"ada"},
            1,
            7,
            30,
        )
        cache[("prompt", "source")] = value
        row = conn.execute("SELECT payload FROM candidate_stage_cache").fetchone()
        assert text not in zlib.decompress(row[0]).decode()
        loaded = cache.get(("prompt", "source"), passage=text)
        assert loaded[0] == record
        assert loaded[1][0][1]["passage"] == text
        assert loaded[1][0][2].named_entities == ["Ada"]
        assert loaded[2:] == value[2:]
        history = REPLHistory(max_output_chars=500).append(
            code="print(source)",
            output="Source text: " + text,
        )
        sources = {hashlib.sha256(text.encode()).hexdigest(): text}
        replay = CandidateStageCache(
            store,
            managed,
            "shacl",
            lambda: "version-one",
            sources=lambda: sources,
        )
        replay[("trace",)] = [
            (
                "shacl_healer_rlm.generate_action",
                {
                    "repl_history": history,
                    "focus_context": json.dumps({"text": text}),
                },
                dspy.Prediction(code="SUBMIT(repairs=[])"),
            )
        ]
        row = conn.execute("SELECT payload FROM candidate_stage_cache WHERE stage = 'shacl'").fetchone()
        assert text not in zlib.decompress(row[0]).decode()
        restored = replay[("trace",)]
        assert restored[0][1]["repl_history"].format() == history.format()
        assert restored[0][1]["focus_context"] == json.dumps({"text": text})
        vector_cache = CandidateStageCache(store, managed, "embedding", lambda: "version-one")
        vector_cache[("query",)] = np.array([0.1, 0.2], dtype=np.float32)
        np.testing.assert_array_equal(vector_cache[("query",)], [np.float32(0.1), np.float32(0.2)])
        changed = CandidateStageCache(store, managed, "openie", lambda: "version-two")
        assert changed.get(("prompt", "source"), passage=text) is None
        with managed.transaction() as conn:
            conn.execute("UPDATE candidate_stage_cache SET payload = X'00' WHERE stage = 'openie'")
        assert cache.get(("prompt", "source"), passage=text) is None
        assert ("prompt", "source") not in cache
    finally:
        managed.close()


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
                    code = (
                        "repairs = qwen_propose_shacl_repairs(violations=violations, "
                        "focus_context='')\nSUBMIT(repairs=repairs)"
                    )
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
                    content = json.dumps({"decisions": []})
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
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

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
            assert {"embedding", "ner", "triples", "synonym", "shacl", "rlm"} <= set(requests)
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
            assert all(hits > 0 for hits in second["hits"].values())
            print(
                json.dumps(
                    {
                        "initial_requests": before,
                        "replay_requests": len(requests) - before,
                        "replay_cache_hits": second["hits"],
                        "preserved_tokens": second["tokens"],
                        "preserved_openie_tokens": second["openie_tokens"],
                    }
                )
            )
            query_only = run_worker(url, "query")
            assert requests[before:] == ["embedding", "embedding"]
            assert all(hits > 0 for hits in query_only["hits"].values())
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
