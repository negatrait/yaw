"""Checks for the active optimizer's scoring, bare-ReAct proposer, lazy optuna, and workspace boundaries."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import pytest
from dspy.utils.callback import BaseCallback
from filelock import FileLock
from numpy.typing import NDArray

import yaw.optimize
from yaw.core import YawConfig, YawRuntimeError
from yaw.datasets import prepare_dataset_bundle, score_retrieval_recalls
from yaw.optimize import make_react_instruction_proposer, optimize_dspy, optimize_optuna, qa_gepa_metric
from yaw.qa import ABSTAIN_REPLY, QASignature, create_lm


def _tool_reply(thought: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "reasoning_content": thought,
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
            }
            for i, call in enumerate(calls)
        ],
    }


def _start_loopback_chat_server(responder: Any) -> tuple[ThreadingHTTPServer, list[dict[str, Any]]]:
    requests_seen: list[dict[str, Any]] = []
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with lock:
                turn_idx = len(requests_seen)
                requests_seen.append(payload)
            status_code, content = responder(turn_idx, payload)
            body = {
                "id": f"chatcmpl-{turn_idx}",
                "object": "chat.completion",
                "created": 1,
                "model": payload.get("model", "gemma-4-e4b"),
                "choices": [
                    {
                        "index": 0,
                        "message": content if isinstance(content, dict) else {"role": "assistant", "content": content},
                        "finish_reason": "tool_calls" if isinstance(content, dict) else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, requests_seen


def test_gepa_scores_inline_citations_against_react_observations():
    text = "Neil Armstrong commanded Apollo 11."
    gold = dspy.Example(gold_answers=["Neil Armstrong"], gold_docs=[text])
    history = {
        "messages": [
            {
                "tool_calls": {
                    "tool_calls": [{"id": "search", "name": "search_passages"}],
                    "tool_call_results": [
                        {"tool_call_id": "search", "value": f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{text}"}
                    ],
                }
            }
        ]
    }
    grounded = qa_gepa_metric(gold, dspy.Prediction(answer="Neil Armstrong [chunk_id=42]", history=history))
    invented = qa_gepa_metric(gold, dspy.Prediction(answer="Neil Armstrong [chunk_id=99]", history=history))
    assert grounded.score == pytest.approx(1.0)
    assert invented.score == pytest.approx(0.85)
    assert "citation_grounding=0.00" in invented.feedback


def test_gepa_abstention_error_tool_calls_and_deduped_gold_docs() -> None:
    assert ABSTAIN_REPLY.lower() in QASignature.instructions.lower()

    assert not hasattr(yaw.optimize, "_ABSTAIN_PHRASES")
    for reply in (ABSTAIN_REPLY, ABSTAIN_REPLY.swapcase()):
        res_abs = qa_gepa_metric(
            dspy.Example(gold_answers=["Neil Armstrong"]),
            dspy.Prediction(answer=reply),
        )
        assert res_abs.score == pytest.approx(0.15)
        assert "citation_grounding=1.00" in res_abs.feedback
        assert "Cite passages" not in res_abs.feedback

    res_non_canon = qa_gepa_metric(
        dspy.Example(gold_answers=["Neil Armstrong"]),
        dspy.Prediction(answer="insufficient evidence"),
    )
    assert res_non_canon.score == pytest.approx(0.0)
    assert "citation_grounding=0.00" in res_non_canon.feedback
    assert "Cite passages" in res_non_canon.feedback

    doc1 = "Neil Armstrong commanded Apollo 11."
    doc2 = "Buzz Aldrin piloted the lunar module."
    history = {
        "messages": [
            {
                "tool_calls": {
                    "tool_calls": [
                        {"id": "err1", "name": "read_chunk"},
                        {"id": "ok1", "name": "search_passages"},
                    ],
                    "tool_call_results": [
                        {
                            "tool_call_id": "err1",
                            "name": "read_chunk",
                            "value": "[chunk_id=999] (bad.md:1-1) score=1.000:\nError: bad id",
                            "is_error": True,
                        },
                        {
                            "tool_call_id": "ok1",
                            "name": "search_passages",
                            "value": f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{doc1}",
                            "is_error": False,
                        },
                    ],
                }
            },
            {
                "tool_calls": {
                    "tool_calls": [{"id": "sub1", "name": "submit"}],
                    "tool_call_results": [
                        {
                            "tool_call_id": "sub1",
                            "name": "submit",
                            "value": {"answer": "Neil Armstrong [chunk_id=42]"},
                            "is_error": False,
                        }
                    ],
                }
            },
        ]
    }
    gold_dup = dspy.Example(gold_answers=["Neil Armstrong"], gold_docs=[doc1, doc1, doc2])
    obs_blocks = [f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{doc1}"]
    expected_recall = score_retrieval_recalls([[doc1, doc1, doc2]], [obs_blocks], k_list=(1,))[0]["Recall@1"]
    assert expected_recall == pytest.approx(0.5)

    res_err_count = qa_gepa_metric(
        gold_dup,
        dspy.Prediction(answer="Neil Armstrong [chunk_id=42]", history=history),
    )
    assert "tool_calls=2." in res_err_count.feedback
    assert "citation_grounding=1.00" in res_err_count.feedback
    assert res_err_count.score == pytest.approx(0.55 + 0.15 + 0.15 + 0.15 * expected_recall)

    # Citing an errored observation must NOT count as grounded
    res_cite_err = qa_gepa_metric(
        gold_dup,
        dspy.Prediction(answer="Neil Armstrong [chunk_id=999]", history=history),
    )
    assert "tool_calls=2." in res_cite_err.feedback
    assert "citation_grounding=0.00" in res_cite_err.feedback

    # Dead trajectory / citations / pred_trace fields must be ignored when history is absent
    res_no_hist = qa_gepa_metric(
        dspy.Example(gold_answers=["Neil Armstrong"]),
        dspy.Prediction(
            answer="Neil Armstrong [chunk_id=42]",
            citations=["apollo.md:1-2"],
            trajectory={
                "tool_name_0": "search_passages",
                "observation_0": f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{doc1}",
            },
        ),
        pred_trace=[
            (
                None,
                {"trajectory": f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{doc1}"},
                dspy.Prediction(trajectory={"observation_0": f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{doc1}"}),
            )
        ],
    )
    assert "tool_calls=0." in res_no_hist.feedback
    assert "citation_grounding=0.00" in res_no_hist.feedback


def test_optimizers_reject_held_out_overlap_before_indexing(tmp_path):
    cfg = YawConfig(workspace_root=tmp_path)
    for optimize in (optimize_dspy, optimize_optuna):
        with pytest.raises(YawRuntimeError, match="held-out"):
            optimize(cfg, datasets="2wikimultihopqa", query_offset=0)
        with pytest.raises(
            YawRuntimeError, match=r"Prepared dataset bundle not found.*scripts/download-datasets\.py --dataset"
        ) as exc_missing:
            optimize(cfg, datasets="hotpotqa", query_offset=10)
        assert exc_missing.value.code == "INVALID_INPUT"

    guard_cfg = YawConfig(workspace_root=tmp_path / "int_guard")
    for bad_val in (0, -5, True, False):
        with pytest.raises(YawRuntimeError, match="trials") as exc_t:
            optimize_optuna(guard_cfg, datasets="sample", trials=bad_val)
        assert exc_t.value.code == "INVALID_INPUT"
    for bad_q in (0, -1, True, False):
        with pytest.raises(YawRuntimeError, match="max_queries") as exc_oq:
            optimize_optuna(guard_cfg, datasets="sample", max_queries=bad_q)
        assert exc_oq.value.code == "INVALID_INPUT"
        with pytest.raises(YawRuntimeError, match="max_queries") as exc_dq:
            optimize_dspy(guard_cfg, datasets="sample", max_queries=bad_q)
        assert exc_dq.value.code == "INVALID_INPUT"
    for bad_off in (-1, True, False):
        with pytest.raises(YawRuntimeError, match="query_offset") as exc_oo:
            optimize_optuna(guard_cfg, datasets="sample", query_offset=bad_off)
        assert exc_oo.value.code == "INVALID_INPUT"
        with pytest.raises(YawRuntimeError, match="query_offset") as exc_do:
            optimize_dspy(guard_cfg, datasets="sample", query_offset=bad_off)
        assert exc_do.value.code == "INVALID_INPUT"
    for bad_th in (0, -1, True, False):
        with pytest.raises(YawRuntimeError, match="threads") as exc_th:
            optimize_dspy(guard_cfg, datasets="sample", threads=bad_th)
        assert exc_th.value.code == "INVALID_INPUT"
    for bad_mc in (0, -1, True, False):
        with pytest.raises(YawRuntimeError, match="max_metric_calls") as exc_mc:
            optimize_dspy(guard_cfg, datasets="sample", max_metric_calls=bad_mc)
        assert exc_mc.value.code == "INVALID_INPUT"

    assert not (tmp_path / "int_guard" / ".yaw").exists()
    assert not list(tmp_path.rglob("*.db"))


def test_gepa_rejects_competing_writer(tmp_path):
    cfg = YawConfig(workspace_root=tmp_path)
    lock_path = cfg.for_optimization().lock_path
    assert lock_path is not None
    lock_path.parent.mkdir(parents=True)
    with FileLock(str(lock_path), timeout=0), pytest.raises(YawRuntimeError, match="BUSY"):
        optimize_dspy(cfg, datasets="2wikimultihopqa", query_offset=10)


def test_optimize_lazy_optuna_import_and_shared_em_f1_metric(tmp_path: Path) -> None:
    opt_path = Path(yaw.optimize.__file__)
    tree = ast.parse(opt_path.read_text(encoding="utf-8"), filename=str(opt_path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name != "optuna" and not alias.name.startswith("optuna.") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "") != "optuna" and not (node.module or "").startswith("optuna.")

    for retired in (
        "normalize_qa_answer",
        "compute_em",
        "compute_f1",
        "_NORM_ARTICLES_RE",
        "compute_recall_at_k",
        "_SPAN_HEADER_RE",
        "evaluate_optimization",
    ):
        assert not hasattr(yaw.optimize, retired)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from pathlib import Path\n"
                "import yaw.optimize\n"
                "assert 'optuna' not in sys.modules\n"
                "from yaw.core import YawConfig, YawRuntimeError\n"
                "cfg = YawConfig(workspace_root=Path(sys.argv[1]))\n"
                "try:\n"
                "    yaw.optimize.optimize_optuna(cfg, datasets='2wikimultihopqa', query_offset=0)\n"
                "except YawRuntimeError:\n"
                "    pass\n"
                "assert 'optuna' in sys.modules\n"
            ),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    # Uses yaw.datasets.compute_qa_em_f1 (NFD normalization) and skips blank gold answers
    nfd_res = qa_gepa_metric(dspy.Example(gold_answers=["Café", ""]), dspy.Prediction(answer="Cafe\u0301"))
    assert nfd_res.score == pytest.approx(0.85)
    blank_res = qa_gepa_metric(dspy.Example(gold_answers=["", "   ", "Paris"]), dspy.Prediction(answer=""))
    assert blank_res.score == pytest.approx(0.0)
    assert "EM=0.0" in blank_res.feedback


def test_react_instruction_proposer_untruncated_tools_stripped_and_native_errors(tmp_path: Path) -> None:
    dummy_lm = create_lm(YawConfig(workspace_root=tmp_path, base_url="http://127.0.0.1:1/v1"))
    for bad_chars in (0, -5, True, False):
        with pytest.raises(ValueError, match="max_chars"):
            make_react_instruction_proposer(dummy_lm, max_chars=bad_chars)
    for bad_iters in (0, -1, True, False):
        with pytest.raises(ValueError, match="max_iters"):
            make_react_instruction_proposer(dummy_lm, max_iters=bad_iters)

    long_context = "C" * 5600 + "_TAIL_5600"
    rows = [
        {
            "Inputs": {
                "question": "Who commanded Apollo 11?",
                "tools": "LEAKED_STUDENT_TOOLS_SCHEMA_XYZ",
                "Context": long_context,
            },
            "Generated Outputs": {"answer": "Wrong"},
            "Feedback": "F1=0.000, EM=0.0, citation_grounding=0.00.",
            "score": 0.0,
        },
        {
            "Inputs": {
                "question": "What instrument did Armstrong play?",
                "tools": "LEAKED_STUDENT_TOOLS_SCHEMA_ABC",
                "Context": "Short context",
            },
            "Generated Outputs": {"answer": "XylophoneZither [chunk_id=1]"},
            "Feedback": "F1=1.000, EM=1.0.",
            "score": 1.0,
        },
    ]

    def responder(turn_idx: int, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        assert payload.get("tool_choice") == "required"
        if turn_idx == 0:
            return 200, _tool_reply("List available examples", [{"name": "list_examples", "args": {}}])
        if turn_idx == 1:
            return 200, _tool_reply(
                "Probe boundary indices and missing field",
                [
                    {"name": "inspect_example", "args": {"index": -1}},
                    {"name": "inspect_example", "args": {"index": 999}},
                    {"name": "inspect_example", "args": {"index": True}},
                    {"name": "inspect_example", "args": {"index": 0, "field": "missing_field"}},
                ],
            )
        if turn_idx == 2:
            return 200, _tool_reply(
                "Inspect full example 0 and its Inputs",
                [
                    {"name": "inspect_example", "args": {"index": 0}},
                    {"name": "inspect_example", "args": {"index": 0, "field": "Inputs"}},
                ],
            )
        return 200, _tool_reply(
            "Submit improved instruction",
            [{"name": "submit", "args": {"improved_instruction": "Ground answers strictly in cited passages."}}],
        )

    captured_preds: list[dspy.Prediction] = []

    class _CaptureReAct(BaseCallback):
        def on_module_end(self, call_id: str, outputs: Any | None, exception: BaseException | None = None) -> None:
            _ = (call_id, exception)
            if isinstance(outputs, dspy.Prediction) and hasattr(outputs, "improved_instruction"):
                captured_preds.append(outputs)

    srv, requests_seen = _start_loopback_chat_server(responder)
    try:
        lm = create_lm(YawConfig(workspace_root=tmp_path, base_url=f"http://127.0.0.1:{srv.server_port}/v1"))
        propose = make_react_instruction_proposer(lm, max_chars=200, max_iters=6)
        with dspy.context(callbacks=[_CaptureReAct()]):
            updated = propose({"react": "Initial instruction."}, {"react": rows}, ["react"])
    finally:
        srv.shutdown()

    assert updated == {"react": "Ground answers strictly in cited passages."}
    assert len(requests_seen) == 4

    # Caller input rows must not be mutated in place
    assert rows[0]["Inputs"]["tools"] == "LEAKED_STUDENT_TOOLS_SCHEMA_XYZ"

    # Initial prompt conveys metadata cleanly without stuffing raw traces or leaked tools schemas
    turn0_blob = json.dumps(requests_seen[0]["messages"])
    assert "Initial instruction." in turn0_blob
    assert "Must be at most 200 characters." in turn0_blob
    assert "_TAIL_5600" not in turn0_blob
    for req in requests_seen:
        blob = json.dumps(req["messages"])
        assert "LEAKED_STUDENT_TOOLS_SCHEMA_XYZ" not in blob
        assert "LEAKED_STUDENT_TOOLS_SCHEMA_ABC" not in blob

    # Turn 1 carries list_examples JSON without "tools" in input_keys
    turn1_tools = [m["content"] for m in requests_seen[1]["messages"] if m.get("role") == "tool"]
    assert len(turn1_tools) == 1
    listed = json.loads(turn1_tools[0])
    assert [entry["index"] for entry in listed] == [0, 1]
    assert listed[0]["score"] == 0.0
    assert "tools" not in listed[0]["input_keys"]
    assert "tools" not in listed[1]["input_keys"]

    # Turn 2 carries 4 native error observations and zero clamped index-0 data
    turn2_tools = [m["content"] for m in requests_seen[2]["messages"] if m.get("role") == "tool"]
    err_obs = turn2_tools[-4:]
    assert all("_TAIL_5600" not in msg for msg in err_obs)
    assert any("-1" in msg and "out of range" in msg for msg in err_obs)
    assert any("999" in msg and "out of range" in msg for msg in err_obs)
    assert any("integer" in msg for msg in err_obs)
    assert any("missing_field" in msg for msg in err_obs)

    # Turn 3 carries full >5,600-char Context intact and zero "tools" inside Inputs
    turn3_tools = [m["content"] for m in requests_seen[3]["messages"] if m.get("role") == "tool"]
    obs_full, obs_inputs = json.loads(turn3_tools[-2]), json.loads(turn3_tools[-1])
    assert obs_full["Inputs"]["Context"] == long_context
    assert "tools" not in obs_full["Inputs"]
    assert obs_inputs["Context"] == long_context
    assert "tools" not in obs_inputs

    # Verify history recorded native is_error=True on all 4 bad tool calls
    assert len(captured_preds) == 1
    tool_results = [
        res
        for msg in captured_preds[0].history.messages
        if isinstance(msg, dict) and (tc := msg.get("tool_calls")) is not None
        for res in (getattr(getattr(tc, "tool_call_results", None), "tool_call_results", None) or ())
    ]
    err_results = [r for r in tool_results if getattr(r, "is_error", False)]
    assert len(err_results) == 4


def test_react_instruction_proposer_rejects_empty_overlong_and_handles_forced_submit(tmp_path: Path) -> None:
    rows = [{"Inputs": {"question": "Q?"}, "Generated Outputs": {"answer": "A"}, "Feedback": "Bad"}]

    # 1. Empty submitted instruction raises ValueError (no fallback to current)
    srv_empty, _ = _start_loopback_chat_server(
        lambda _t, _p: (200, _tool_reply("Empty", [{"name": "submit", "args": {"improved_instruction": "   "}}]))
    )
    try:
        lm_empty = create_lm(
            YawConfig(workspace_root=tmp_path, base_url=f"http://127.0.0.1:{srv_empty.server_port}/v1")
        )
        with pytest.raises(ValueError, match="empty instruction"):
            make_react_instruction_proposer(lm_empty)({"react": "Keep me?"}, {"react": rows}, ["react"])
    finally:
        srv_empty.shutdown()

    # 2. Overlong submitted instruction raises ValueError (no silent truncation)
    srv_long, _ = _start_loopback_chat_server(
        lambda _t, _p: (200, _tool_reply("Long", [{"name": "submit", "args": {"improved_instruction": "X" * 25}}]))
    )
    try:
        lm_long = create_lm(YawConfig(workspace_root=tmp_path, base_url=f"http://127.0.0.1:{srv_long.server_port}/v1"))
        with pytest.raises(ValueError, match="exceeds max_chars=20"):
            make_react_instruction_proposer(lm_long, max_chars=20)({"react": "Current"}, {"react": rows}, ["react"])
    finally:
        srv_long.shutdown()

    # 3. max_iters=1 forces submit cleanly on turn 1
    def forced_responder(turn_idx: int, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if turn_idx == 0:
            assert payload.get("tool_choice") == "required"
            return 200, _tool_reply("List first", [{"name": "list_examples", "args": {}}])
        assert payload.get("tool_choice") == {"type": "function", "function": {"name": "submit"}}
        return 200, _tool_reply("Forced submit", [{"name": "submit", "args": {"improved_instruction": "Concise."}}])

    srv_forced, forced_reqs = _start_loopback_chat_server(forced_responder)
    try:
        lm_forced = create_lm(
            YawConfig(workspace_root=tmp_path, base_url=f"http://127.0.0.1:{srv_forced.server_port}/v1")
        )
        res = make_react_instruction_proposer(lm_forced, max_iters=1)({"react": "Old"}, {"react": rows}, ["react"])
    finally:
        srv_forced.shutdown()

    assert len(forced_reqs) == 2
    assert res == {"react": "Concise."}

    # 4. Missing/empty reflective components or missing candidate keys skip HTTP calls cleanly
    srv_skip, skip_reqs = _start_loopback_chat_server(forced_responder)
    try:
        lm_skip = create_lm(YawConfig(workspace_root=tmp_path, base_url=f"http://127.0.0.1:{srv_skip.server_port}/v1"))
        propose = make_react_instruction_proposer(lm_skip)
        assert propose({"react": "Current"}, {}, ["react"]) == {}
        assert propose({"react": "Current"}, {"react": []}, ["react"]) == {}
        assert propose({}, {"react": rows}, ["react"]) == {}
    finally:
        srv_skip.shutdown()
    assert skip_reqs == []


def test_optimize_optuna_uses_maxp_pool_and_persists_best_params(tmp_path: Path) -> None:
    def _eval_int_ast(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return _eval_int_ast(node.left) * _eval_int_ast(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max":
            return max(_eval_int_ast(arg) for arg in node.args)
        raise AssertionError(f"Unexpected top_k AST expression: {ast.dump(node)}")

    opt_path = Path(yaw.optimize.__file__)
    tree = ast.parse(opt_path.read_text(encoding="utf-8"), filename=str(opt_path))
    opt_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "optimize_optuna")
    search_calls = [
        n
        for n in ast.walk(opt_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "search_vectors"
    ]
    assert len(search_calls) >= 1
    for call in search_calls:
        kw = next((k for k in call.keywords if k.arg == "top_k"), None)
        assert kw is not None
        assert _eval_int_ast(kw.value) >= 50

    corpus = [
        {
            "idx": "d1",
            "title": "Apollo Mission",
            "text": "Neil Armstrong commanded Apollo 11 to the Moon in July 1969.",
        },
        {
            "idx": "d2",
            "title": "Voyager Probe",
            "text": "Voyager 1 entered interstellar space in August 2012.",
        },
    ]
    questions = [
        {
            "question": "Who commanded Apollo 11 to the Moon?",
            "answer": "Neil Armstrong",
            "gold_docs": ["Apollo Mission\nNeil Armstrong commanded Apollo 11 to the Moon in July 1969."],
        },
        {
            "question": "When did Voyager 1 enter interstellar space?",
            "answer": "August 2012",
            "gold_docs": ["Voyager Probe\nVoyager 1 entered interstellar space in August 2012."],
        },
    ]
    corpus_path = tmp_path / "corpus.json"
    questions_path = tmp_path / "questions.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    bundle = prepare_dataset_bundle("sample", corpus_path, questions_path, tmp_path / "bundles")

    def _embed(texts: list[str]) -> NDArray[np.float32]:
        topics = (
            ("apollo", "armstrong", "moon", "commanded"),
            ("voyager", "interstellar", "space", "2012"),
        )
        rows = np.zeros((len(texts), 16), dtype=np.float32)
        for idx, text in enumerate(texts):
            low = text.lower()
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = (np.frombuffer(digest * 2, dtype=np.uint8)[:16].astype(np.float32) - 127.5) * 0.05
            for t_idx, kws in enumerate(topics):
                for kw in kws:
                    if kw in low:
                        vec[t_idx] += 2.0
            rows[idx] = vec / max(float(np.linalg.norm(vec)), 1e-12)
        return rows

    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = YawConfig(workspace_root=ws, chunk_size=240, chunk_overlap=0, min_chunk_size=20)
    payload = optimize_optuna(
        cfg,
        datasets=str(bundle.root),
        trials=1,
        max_queries=2,
        query_offset=0,
        fresh=True,
        embed_fn=_embed,
    )

    assert payload["best_score"] > 0.0
    assert payload["total_trials"] == 1
    assert {"hybrid_alpha", "ppr_alpha", "seed_top_k"} <= set(payload["best_params"])

    opt_db = cfg.for_optimization(".yaw/optuna_index.db").db_path
    assert opt_db is not None
    opt_dir = opt_db.parent
    params_path = opt_dir / "optuna_best_params.json"
    status_path = opt_dir / "optuna_status.json"
    study_db = opt_dir / "optuna_study.db"
    assert not (opt_dir / "optuna_corpus").exists()
    assert params_path.is_file()
    assert status_path.is_file()
    assert study_db.is_file()
    md_before = {p.name: p.read_bytes() for p in bundle.documents_root.glob("*.md")}
    assert set(md_before) == {"d1.md", "d2.md"}

    saved_params = json.loads(params_path.read_text(encoding="utf-8"))
    saved_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved_params["best_score"] == pytest.approx(payload["best_score"])
    assert saved_params["best_params"] == payload["best_params"]
    assert saved_status["state"] == "completed"
    assert saved_status["total_trials"] == 1

    with contextlib.closing(sqlite3.connect(f"file:{study_db.as_posix()}?mode=ro", uri=True)) as s_conn:
        study_names = [r[0] for r in s_conn.execute("SELECT study_name FROM studies").fetchall()]
        trial_count = s_conn.execute("SELECT count(*) FROM trials").fetchone()[0]
    assert "yaw_retrieval_ppr" in study_names
    assert trial_count == 1

    # Consecutive fresh=True run succeeds without staging copies or mutating bundle .md files
    payload_2 = optimize_optuna(
        cfg,
        datasets=str(bundle.root),
        trials=1,
        max_queries=2,
        query_offset=0,
        fresh=True,
        embed_fn=_embed,
    )
    assert payload_2["best_score"] == pytest.approx(payload["best_score"])
    assert payload_2["total_trials"] == 1
    assert not (opt_dir / "optuna_corpus").exists()
    assert {p.name: p.read_bytes() for p in bundle.documents_root.glob("*.md")} == md_before


def test_optimize_optuna_and_dspy_locks_and_redacted_failure_status(tmp_path: Path) -> None:
    corpus = [
        {"idx": "d1", "title": "Apollo", "text": "Neil Armstrong commanded Apollo 11."},
        {"idx": "d2", "title": "Voyager", "text": "Voyager 1 entered interstellar space."},
    ]
    questions = [
        {
            "question": "Who commanded Apollo 11?",
            "answer": "Neil Armstrong",
            "gold_docs": ["Apollo\nNeil Armstrong commanded Apollo 11."],
        },
        {
            "question": "What entered interstellar space?",
            "answer": "Voyager 1",
            "gold_docs": ["Voyager\nVoyager 1 entered interstellar space."],
        },
    ]
    cp, qp = tmp_path / "c.json", tmp_path / "q.json"
    cp.write_text(json.dumps(corpus), encoding="utf-8")
    qp.write_text(json.dumps(questions), encoding="utf-8")
    bundle = prepare_dataset_bundle("sample", cp, qp, tmp_path / "bundles")

    ws = tmp_path / "ws"
    ws.mkdir()
    secret = "secret-token-123"
    cfg = YawConfig(workspace_root=ws, api_key=secret, chunk_size=240, chunk_overlap=0, min_chunk_size=20)

    # 1. Optuna lock contention raises BUSY without writing failed status
    optuna_cfg = cfg.for_optimization(".yaw/optuna_index.db")
    assert optuna_cfg.lock_path is not None and optuna_cfg.db_path is not None
    optuna_cfg.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(optuna_cfg.lock_path), timeout=0), pytest.raises(YawRuntimeError, match="BUSY"):
        optimize_optuna(
            cfg,
            datasets=str(bundle.root),
            trials=1,
            max_queries=2,
            query_offset=0,
            embed_fn=lambda ts: np.ones((len(ts), 8), dtype=np.float32),
        )
    optuna_status = optuna_cfg.db_path.parent / "optuna_status.json"
    assert not optuna_status.exists()

    # 2. Pre-running held-out validation must NOT write optuna_status.json or dspy_status.json
    with pytest.raises(YawRuntimeError, match="held-out"):
        optimize_optuna(cfg, datasets="2wikimultihopqa", query_offset=0)
    assert not optuna_status.exists()
    dspy_cfg = cfg.for_optimization()
    assert dspy_cfg.db_path is not None
    dspy_status = dspy_cfg.db_path.parent / "dspy_status.json"
    with pytest.raises(YawRuntimeError, match="held-out"):
        optimize_dspy(cfg, datasets="2wikimultihopqa", query_offset=0)
    assert not dspy_status.exists()

    # 3. Post-running failure in optimize_optuna writes failed + redacted error (secret stripped)
    def _fail_on_query_embed(texts: list[str]) -> NDArray[np.float32]:
        if any(t.startswith("task:") for t in texts):
            raise RuntimeError(f"boom {secret}")
        return np.ones((len(texts), 8), dtype=np.float32)

    with pytest.raises(RuntimeError, match=secret):
        optimize_optuna(
            cfg,
            datasets=str(bundle.root),
            trials=1,
            max_queries=2,
            query_offset=0,
            fresh=True,
            embed_fn=_fail_on_query_embed,
        )
    st_opt = json.loads(optuna_status.read_text(encoding="utf-8"))
    assert st_opt["state"] == "failed"
    assert st_opt["pid"] is None
    assert secret not in st_opt["error"]
    assert "[REDACTED]" in st_opt["error"]

    # 4. Post-running failure in optimize_dspy writes failed + redacted error (secret stripped)
    def _fail_on_doc_embed(texts: list[str]) -> NDArray[np.float32]:
        if texts == ["dimension probe"]:
            return np.ones((1, 8), dtype=np.float32)
        raise RuntimeError(f"boom {secret}")

    with pytest.raises(Exception, match=secret):
        optimize_dspy(
            cfg,
            datasets=str(bundle.root),
            max_metric_calls=2,
            max_queries=2,
            query_offset=0,
            fresh=True,
            embed_fn=_fail_on_doc_embed,
        )
    st_dspy = json.loads(dspy_status.read_text(encoding="utf-8"))
    assert st_dspy["state"] == "failed"
    assert st_dspy["pid"] is None
    assert secret not in st_dspy["error"]
    assert "[REDACTED]" in st_dspy["error"]

    # 5. _compile_dspy reads optuna_best_params.json via resolve_retrieval_params
    opt_params_file = optuna_cfg.db_path.parent / "optuna_best_params.json"
    opt_params_file.write_text("{corrupt json", encoding="utf-8")
    dummy_lm = create_lm(replace(cfg, base_url="http://127.0.0.1:1/v1"))
    with pytest.raises(YawRuntimeError) as exc_dspy_cfg:
        optimize_dspy(
            cfg,
            datasets=str(bundle.root),
            max_metric_calls=2,
            max_queries=2,
            query_offset=0,
            fresh=True,
            embed_fn=lambda ts: np.ones((len(ts), 8), dtype=np.float32),
            student_lm=dummy_lm,
            reflection_lm=dummy_lm,
        )
    assert exc_dspy_cfg.value.code == "CONSISTENCY_ERROR"


def test_optimize_dspy_multithreaded_uses_per_call_readonly_sqlite(tmp_path: Path) -> None:
    corpus = [
        {"idx": "d1", "title": "Apollo 11", "text": "Neil Armstrong commanded Apollo 11 to the Moon."},
        {"idx": "d2", "title": "Voyager 1", "text": "Voyager 1 entered interstellar space in August 2012."},
        {"idx": "d3", "title": "Hubble", "text": "Hubble Space Telescope launched aboard Discovery in 1990."},
        {"idx": "d4", "title": "Cassini", "text": "Cassini orbited Saturn for thirteen years until 2017."},
    ]
    questions = [
        {
            "question": "Who commanded Apollo 11?",
            "answer": "Neil Armstrong",
            "gold_docs": ["Apollo 11\nNeil Armstrong commanded Apollo 11 to the Moon."],
        },
        {
            "question": "When did Voyager 1 enter interstellar space?",
            "answer": "August 2012",
            "gold_docs": ["Voyager 1\nVoyager 1 entered interstellar space in August 2012."],
        },
        {
            "question": "When did Hubble launch?",
            "answer": "1990",
            "gold_docs": ["Hubble\nHubble Space Telescope launched aboard Discovery in 1990."],
        },
        {
            "question": "Until what year did Cassini orbit Saturn?",
            "answer": "2017",
            "gold_docs": ["Cassini\nCassini orbited Saturn for thirteen years until 2017."],
        },
    ]
    cp, qp = tmp_path / "c.json", tmp_path / "q.json"
    cp.write_text(json.dumps(corpus), encoding="utf-8")
    qp.write_text(json.dumps(questions), encoding="utf-8")
    bundle = prepare_dataset_bundle("sample", cp, qp, tmp_path / "bundles")

    def _embed(texts: list[str]) -> NDArray[np.float32]:
        topics = (("apollo", "armstrong"), ("voyager", "interstellar"), ("hubble", "discovery"), ("cassini", "saturn"))
        rows = np.zeros((len(texts), 16), dtype=np.float32)
        for idx, text in enumerate(texts):
            low = text.lower()
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = (np.frombuffer(digest * 2, dtype=np.uint8)[:16].astype(np.float32) - 127.5) * 0.05
            for t_idx, kws in enumerate(topics):
                if any(kw in low for kw in kws):
                    vec[t_idx] += 2.0
            rows[idx] = vec / max(float(np.linalg.norm(vec)), 1e-12)
        return rows

    stat_lock = threading.Lock()
    tools_used: set[str] = set()
    reflection_temps: list[float | None] = []

    def _responder(_turn_idx: int, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        msgs = payload.get("messages", [])
        tool_names = {t["function"]["name"] for t in payload.get("tools", [])}
        has_tool_obs = any(m.get("role") == "tool" for m in msgs)
        blob = json.dumps(msgs)
        if "list_examples" in tool_names:
            with stat_lock:
                reflection_temps.append(payload.get("temperature"))
            if not has_tool_obs:
                return 200, _tool_reply("List examples", [{"name": "list_examples", "args": {}}])
            return 200, _tool_reply(
                "Propose instruction",
                [{"name": "submit", "args": {"improved_instruction": "Search and cite [chunk_id=<int>]."}}],
            )
        if not has_tool_obs:
            kw = next((k for k in ("Apollo", "Voyager", "Hubble", "Cassini") if k in blob), "Apollo")
            with stat_lock:
                tools_used.update({"search_passages", "lexical_grep"})
            return 200, _tool_reply(
                "Query vector and lexical tools",
                [
                    {"name": "search_passages", "args": {"query": kw, "top_k": 2}},
                    {"name": "lexical_grep", "args": {"query": kw, "top_k": 2}},
                ],
            )
        cids = re.findall(r"\[chunk_id=(\d+)\]", blob)
        ans = (
            "Neil Armstrong"
            if "Apollo" in blob
            else "August 2012"
            if "Voyager" in blob
            else "1990"
            if "Hubble" in blob
            else "2017"
        )
        return 200, _tool_reply(
            "Submit cited answer", [{"name": "submit", "args": {"answer": f"{ans} [chunk_id={cids[0]}]"}}]
        )

    srv, _ = _start_loopback_chat_server(_responder)
    try:
        ws = tmp_path / "ws"
        ws.mkdir()
        lm_cfg = YawConfig(
            workspace_root=ws,
            base_url=f"http://127.0.0.1:{srv.server_port}/v1",
            chunk_size=240,
            chunk_overlap=0,
            min_chunk_size=20,
        )
        audit = optimize_dspy(
            lm_cfg,
            datasets=str(bundle.root),
            threads=4,
            max_metric_calls=4,
            max_queries=4,
            query_offset=0,
            fresh=True,
            embed_fn=_embed,
            student_lm=create_lm(lm_cfg, temperature=1.0),
        )
    finally:
        srv.shutdown()

    assert audit["train_size"] == 3 and audit["val_size"] == 1
    assert {"search_passages", "lexical_grep"} <= tools_used
    assert reflection_temps and all(t == 1.0 for t in reflection_temps)
    opt_db = lm_cfg.for_optimization().db_path
    assert opt_db is not None
    opt_dir = opt_db.parent
    assert not (opt_dir / "dspy_corpus").exists()
    assert (opt_dir / "qa_program.json").is_file()
    assert (opt_dir / "dspy_gepa_audit.json").is_file()
    status_data = json.loads((opt_dir / "dspy_status.json").read_text(encoding="utf-8"))
    assert status_data["state"] == "completed"
