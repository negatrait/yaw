"""Unmocked integration tests for yaw.optimize (GEPA bare-ReAct instruction proposer and Optuna TPE)."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import dspy
import filelock
import numpy as np
import optuna
import pytest
from dspy.utils.exceptions import LMServerError

from yaw.core import RetrievalParams, YawConfig, YawRuntimeError, load_config
from yaw.datasets import prepare_dataset_bundle
from yaw.qa import build_qa_agent, make_qa_tools
from yaw.retrieval import build_vector_search, open_readonly_db

_CID_IN_TEXT_RE = re.compile(r"\[chunk_id=(\d+)\]")
_FORBIDDEN_WORDS = ("mani" + "fold", "h" + "mm", "hippo" + "rag")
_FORBIDDEN_COMMENTS = ("# " + "noqa", "# " + "type: ignore")


def _topic_embed_vec(text: str, *, dim: int = 16) -> np.ndarray:
    topics = (
        ("apollo", "lunar", "moon", "armstrong", "commander"),
        ("instrument", "xylophone", "zither", "music", "wapakoneta", "childhood"),
        ("orbital", "booster", "valve", "specification", "stage"),
        ("glacier", "fjord", "bergen", "harbor", "nordic"),
        ("telescope", "prism", "optics", "mirror", "lens"),
        ("copper", "smelter", "furnace", "alloy", "foundry"),
    )
    low = text.lower()
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    base = (np.frombuffer(digest * 2, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.05
    for idx, keywords in enumerate(topics):
        for kw in keywords:
            if kw in low:
                base[idx] += 2.0
    return base / max(float(np.linalg.norm(base)), 1e-12)


def _start_openai_stub(
    chat_responder: Any,
    *,
    delay_sec: float = 0.0,
    embed_fn: Any = None,
) -> tuple[ThreadingHTTPServer, list[dict[str, Any]], dict[str, int]]:
    requests_seen: list[dict[str, Any]] = []
    concurrency = {"in_flight": 0, "max_in_flight": 0}
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path.endswith("/embeddings"):
                raw_inputs = payload.get("input", [])
                texts = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
                vec_fn = embed_fn if embed_fn is not None else _topic_embed_vec
                data = [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": vec_fn(str(t)).tolist(),
                    }
                    for i, t in enumerate(texts)
                ]
                out: dict[str, Any] = {
                    "object": "list",
                    "data": data,
                    "model": payload.get("model", "embeddinggemma-2"),
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
                status_code = 200
            else:
                with lock:
                    concurrency["in_flight"] += 1
                    concurrency["max_in_flight"] = max(concurrency["max_in_flight"], concurrency["in_flight"])
                    turn_idx = len(requests_seen)
                    requests_seen.append(payload)
                try:
                    if delay_sec > 0:
                        import time

                        time.sleep(delay_sec)
                    status_code, content = chat_responder(turn_idx, payload)
                finally:
                    with lock:
                        concurrency["in_flight"] -= 1
                if status_code != 200:
                    err_raw = json.dumps({"error": {"message": str(content)}}).encode("utf-8")
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_raw)))
                    self.end_headers()
                    self.wfile.write(err_raw)
                    return
                out = {
                    "id": f"chatcmpl-{turn_idx}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "gemma-4-e4b"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }

            raw = json.dumps(out).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests_seen, concurrency


def _make_loopback_lm(server: ThreadingHTTPServer) -> dspy.LM:
    return dspy.LM(
        model="openai/gemma-4-e4b",
        model_type="chat",
        api_base=f"http://127.0.0.1:{server.server_port}/v1",
        api_key="test",
        temperature=1.0,
        cache=False,
        num_retries=0,
    )


def _xml_tool_reply(thought: str, calls: list[dict[str, Any]]) -> str:
    return f"<next_thought>{thought}</next_thought>\n<tool_calls>{json.dumps({'tool_calls': calls})}</tool_calls>"


def _make_cfg(root: Path, *, port: int = 8888) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        compiled_program_path=root / ".yaw" / "compiled_program.json",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="test",
        chunk_size=1500,
        chunk_overlap=0,
        min_chunk_size=20,
        react_max_iters=4,
    )


def _prepare_sample_bundle(root: Path, *, dataset_name: str = "sample", out_dir: Path | None = None) -> Path:
    corpus_rows = [
        {
            "idx": "doc_apollo",
            "title": "Apollo Mission",
            "text": "Neil Armstrong served as commander of Apollo 11 on the lunar surface.",
        },
        {
            "idx": "doc_instrument",
            "title": "Wapakoneta Music",
            "text": "Neil Armstrong played the XylophoneZither instrument during childhood in Ohio.",
        },
        {
            "idx": "doc_booster",
            "title": "Booster Spec",
            "text": "Orbital booster stage separation relies on the cryo valve specification.",
        },
        {
            "idx": "doc_fjord",
            "title": "Bergen Harbor",
            "text": "Nordic glacier water feeds the deep Bergen fjord harbor year round.",
        },
        {
            "idx": "doc_optics",
            "title": "Prism Optics",
            "text": "The observatory telescope mirror aligns with a fused silica prism lens.",
        },
        {
            "idx": "doc_foundry",
            "title": "Copper Alloy",
            "text": "Copper smelter furnace temperatures control bronze alloy density in the foundry.",
        },
        {
            "idx": "doc_unused_distractor",
            "title": "Unused Archive Note",
            "text": "Unreferenced distractor passage EXCLUDED_SLICE_MARKER never selected by slice.",
        },
    ]
    question_rows = [
        {
            "question": "Who commanded Apollo 11 on the lunar surface?",
            "answer": "Neil Armstrong",
            "gold_docs": ["Apollo Mission\nNeil Armstrong served as commander of Apollo 11 on the lunar surface."],
        },
        {
            "question": "What childhood instrument did Neil Armstrong play in Ohio?",
            "answer": "XylophoneZither",
            "gold_docs": [
                "Wapakoneta Music\nNeil Armstrong played the XylophoneZither instrument during childhood in Ohio."
            ],
        },
        {
            "question": "Which valve governs orbital booster stage separation?",
            "answer": "cryo valve",
            "gold_docs": ["Booster Spec\nOrbital booster stage separation relies on the cryo valve specification."],
        },
        {
            "question": "What feeds the deep Bergen fjord harbor?",
            "answer": "Nordic glacier water",
            "gold_docs": ["Bergen Harbor\nNordic glacier water feeds the deep Bergen fjord harbor year round."],
        },
        {
            "question": "What aligns with the observatory telescope mirror?",
            "answer": "fused silica prism lens",
            "gold_docs": ["Prism Optics\nThe observatory telescope mirror aligns with a fused silica prism lens."],
        },
        {
            "question": "What controls bronze alloy density in the foundry?",
            "answer": "Copper smelter furnace temperatures",
            "gold_docs": [
                "Copper Alloy\nCopper smelter furnace temperatures control bronze alloy density in the foundry."
            ],
        },
    ]
    src_dir = root / "_raw_src"
    src_dir.mkdir(parents=True, exist_ok=True)
    c_path = src_dir / "corpus.json"
    q_path = src_dir / "questions.json"
    c_path.write_text(json.dumps(corpus_rows), encoding="utf-8")
    q_path.write_text(json.dumps(question_rows), encoding="utf-8")
    dest_root = out_dir if out_dir is not None else (root / "outputs" / "datasets")
    bundle = prepare_dataset_bundle(dataset_name, c_path, q_path, dest_root)
    return bundle.root


def test_optimize_module_dag_lazy_optuna_zero_wrappers_and_clean_vocabulary() -> None:
    from yaw.optimize import ProposeInstruction

    opt_path = Path(__file__).resolve().parent.parent / "yaw" / "optimize.py"
    test_path = Path(__file__).resolve()
    for path in (opt_path, test_path):
        src = path.read_text(encoding="utf-8")
        low = src.lower()
        for word in _FORBIDDEN_WORDS:
            assert word not in low, f"Forbidden word {word!r} in {path}"
        for comment_marker in _FORBIDDEN_COMMENTS:
            assert comment_marker not in low, f"Forbidden marker {comment_marker!r} in {path}"

    opt_src = opt_path.read_text(encoding="utf-8")
    tree = ast.parse(opt_src, filename=str(opt_path))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert class_names == {"ProposeInstruction", "_DatasetSpecRef", "_OptunaSliceData"}
    assert issubclass(ProposeInstruction, dspy.Signature)
    assert not issubclass(ProposeInstruction, dspy.Module)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name != "ProposeInstruction":
            for base in node.bases:
                base_txt = ast.unparse(base)
                assert not any(w in base_txt for w in ("Module", "BaseLM", "Adapter"))

    allowed_roots = {
        "dspy",
        "filelock",
        "numpy",
        "optuna",
        "yaw",
    }
    allowed_yaw = {
        "yaw.core",
        "yaw.datasets",
        "yaw.pipeline",
        "yaw.retrieval",
        "yaw.qa",
        "yaw.runtime",
    }

    # Check top-level statements: zero top-level optuna imports
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                assert not alias.name.startswith("optuna")
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            assert not mod.startswith("optuna")

    optuna_func_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "optimize_optuna"
    ]
    assert len(optuna_func_defs) == 1
    optuna_imports_inside = [
        node
        for node in ast.walk(optuna_func_defs[0])
        if (isinstance(node, ast.Import) and any(a.name.startswith("optuna") for a in node.names))
        or (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("optuna"))
    ]
    assert len(optuna_imports_inside) >= 2

    import sys

    stdlib_names = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                assert root_pkg in stdlib_names or root_pkg in allowed_roots
                assert not alias.name.startswith(
                    ("yaw.daemon", "cocoindex", "fastmcp", "starlette", "uvicorn", "spacy")
                )
                if root_pkg == "yaw":
                    assert alias.name in allowed_yaw
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root_pkg = mod.split(".")[0]
            assert root_pkg in stdlib_names or root_pkg in allowed_roots
            assert not mod.startswith(("yaw.daemon", "cocoindex", "fastmcp", "starlette", "uvicorn", "spacy"))
            if root_pkg == "yaw":
                assert any(mod == allowed or mod.startswith(f"{allowed}.") for allowed in allowed_yaw)
                for alias in node.names:
                    assert not alias.name.startswith("_")


def test_qa_gepa_metric_em_f1_citations_and_missed_gold_hints() -> None:
    from yaw.optimize import qa_gepa_metric

    gold = dspy.Example(
        question="What instrument did Armstrong play?",
        gold_answers=["XylophoneZither"],
        gold_docs=["Wapakoneta Music\nNeil Armstrong played the XylophoneZither instrument."],
    ).with_inputs("question")

    # Build a realistic dspy.History with tool_calls (including submit) and an error call
    hit_calls = dspy.ToolCalls(
        tool_calls=[
            {"name": "search_passages", "args": {"query": "Armstrong"}},
            {"name": "read_chunk", "args": {"chunk_id": 999}},
            {"name": "submit", "args": {"answer": "XylophoneZither [chunk_id=42]"}},
        ],
        tool_call_results=dspy.ToolCallResults(
            tool_call_results=[
                {
                    "name": "search_passages",
                    "is_error": False,
                    "value": (
                        "[chunk_id=42] (doc.json:1-1) score=0.950:\n"
                        "Wapakoneta Music\nNeil Armstrong played the XylophoneZither instrument."
                    ),
                },
                {"name": "read_chunk", "is_error": True, "value": "[chunk_id=999] error should not count"},
                {"name": "submit", "is_error": False, "value": {"answer": "XylophoneZither [chunk_id=42]"}},
            ]
        ),
    )
    history = dspy.History(messages=[{"tool_calls": hit_calls}])
    pred_ok = dspy.Prediction(
        answer="XylophoneZither",
        termination_reason="submit",
        history=history,
    )
    res_ok = qa_gepa_metric(gold, pred_ok)
    assert isinstance(res_ok, dspy.Prediction)
    assert res_ok.score == pytest.approx(1.0)
    assert "EM=1.00" in res_ok.feedback
    assert "F1=1.00" in res_ok.feedback
    assert "submit" in res_ok.feedback
    assert "inline_citations=0" in res_ok.feedback
    assert "tool_seen_chunks=1" in res_ok.feedback

    pred_cited = dspy.Prediction(
        answer="XylophoneZither [chunk_id=42]",
        termination_reason="submit",
        history=history,
    )
    res_cited = qa_gepa_metric(gold, pred_cited)
    assert "inline_citations=1" in res_cited.feedback
    assert "tool_seen_chunks=1" in res_cited.feedback

    # Partial/missed prediction: submit arg [chunk_id=888] must NOT be counted as tool_seen!
    miss_calls = dspy.ToolCalls(
        tool_calls=[{"name": "submit", "args": {"answer": "Xylophone [chunk_id=888]"}}],
        tool_call_results=dspy.ToolCallResults(
            tool_call_results=[{"name": "submit", "is_error": False, "value": {"answer": "Xylophone [chunk_id=888]"}}]
        ),
    )
    pred_miss = dspy.Prediction(
        answer="Xylophone [chunk_id=888]",
        termination_reason="forced_submit",
        history=dspy.History(messages=[{"tool_calls": miss_calls}]),
    )
    res_miss = qa_gepa_metric(gold, pred_miss, pred_name="react")
    assert 0.0 <= res_miss.score < 1.0
    assert "EM=0.00" in res_miss.feedback
    assert "forced_submit" in res_miss.feedback
    assert "inline_citations=1" in res_miss.feedback
    assert "tool_seen_chunks=0" in res_miss.feedback
    assert "Wapakoneta Music" in res_miss.feedback

    # None pred safety
    res_none = qa_gepa_metric(gold, None)
    assert res_none.score == 0.0
    assert "EM=0.00" in res_none.feedback


def test_react_instruction_proposer_strips_tools_uncapped_and_validates() -> None:
    from yaw.optimize import make_react_instruction_proposer

    def unused_chat(_i: int, _p: dict[str, Any]) -> tuple[int, str]:
        return 200, _xml_tool_reply("done", [{"name": "submit", "args": {"new_instruction": "Concise"}}])

    srv_err, _, _ = _start_openai_stub(unused_chat)
    try:
        bad_lm = _make_loopback_lm(srv_err)
        with pytest.raises(YawRuntimeError) as exc_mc:
            make_react_instruction_proposer(lm=bad_lm, max_chars=0)
        assert exc_mc.value.code == "INVALID_INPUT"
        with pytest.raises(YawRuntimeError) as exc_mi:
            make_react_instruction_proposer(lm=bad_lm, max_iters=0)
        assert exc_mi.value.code == "INVALID_INPUT"
        with pytest.raises(YawRuntimeError) as exc_bool:
            make_react_instruction_proposer(lm=bad_lm, max_chars=True)
        assert exc_bool.value.code == "INVALID_INPUT"
    finally:
        srv_err.shutdown()

    huge_context = "CTX_" + ("A" * 4000)
    huge_feedback = "FB_" + ("B" * 4000)
    reflective_ds = {
        "react": [
            {
                "Inputs": {
                    "question": "Who commanded Apollo 11?",
                    "Context": huge_context,
                    "tools": "SECRET_TOOL_SCHEMA_SHOULD_BE_STRIPPED",
                    "turn_budget": "1 of 6",
                },
                "Generated Outputs": {"next_thought": "Searching...", "tool_calls": "[]"},
                "Feedback": huge_feedback,
            }
        ]
    }

    observed_tool_outputs: list[str] = []

    def proposer_responder(turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
        msgs = payload.get("messages", [])
        blob = json.dumps(msgs)
        assert "SECRET_TOOL_SCHEMA_SHOULD_BE_STRIPPED" not in blob
        tool_msgs = [str(m.get("content", "")) for m in msgs if m.get("role") == "tool"]
        if turn_idx == 0:
            return 200, _xml_tool_reply(
                "List reflective examples and probe invalid indices/fields",
                [
                    {"name": "list_examples", "args": {}},
                    {"name": "inspect_example", "args": {"idx": True}},
                    {"name": "inspect_example", "args": {"idx": 99}},
                    {"name": "inspect_example", "args": {"idx": -1}},
                    {"name": "inspect_example", "args": {"idx": 0, "field": "nonexistent_field"}},
                ],
            )
        if turn_idx == 1:
            observed_tool_outputs.extend(tool_msgs)
            return 200, _xml_tool_reply(
                "Inspect Context, Feedback, Inputs, and question fields",
                [
                    {"name": "inspect_example", "args": {"idx": 0, "field": ""}},
                    {"name": "inspect_example", "args": {"idx": 0, "field": "Context"}},
                    {"name": "inspect_example", "args": {"idx": 0, "field": "Feedback"}},
                    {"name": "inspect_example", "args": {"idx": 0, "field": "Inputs"}},
                    {"name": "inspect_example", "args": {"idx": 0, "field": "question"}},
                ],
            )
        observed_tool_outputs.extend(tool_msgs[5:])
        return 200, _xml_tool_reply(
            "Submit improved concise instruction",
            [{"name": "submit", "args": {"new_instruction": "X" * 35}}],
        )

    srv_ok, reqs_ok, _ = _start_openai_stub(proposer_responder)
    try:
        lm_ok = _make_loopback_lm(srv_ok)
        proposer = make_react_instruction_proposer(lm=lm_ok, max_chars=40, max_iters=6, lm_sem=threading.Semaphore(1))
        out = proposer({"react": "Initial instruction", "other": "Keep me"}, reflective_ds, ["react", "other"])
        assert len(reqs_ok) == 3
        assert out["react"] == "X" * 35
        assert out["other"] == "Keep me"
        assert "'items':" not in str(reqs_ok[0]["messages"][-1]["content"])
        assert "[0] question='Who commanded Apollo 11?'" in observed_tool_outputs[0]
        assert "True is not of type 'integer'" in observed_tool_outputs[1] or "TypeError" in observed_tool_outputs[1]
        assert "[error]" in observed_tool_outputs[2] and "IndexError" in observed_tool_outputs[2]
        assert "[error]" in observed_tool_outputs[3] and "IndexError" in observed_tool_outputs[3]
        assert "[error]" in observed_tool_outputs[4] and "KeyError" in observed_tool_outputs[4]
        for tool_out in observed_tool_outputs:
            assert "SECRET_TOOL_SCHEMA_SHOULD_BE_STRIPPED" not in tool_out
        assert any(huge_context in t for t in observed_tool_outputs[5:])
        assert any(huge_feedback in t for t in observed_tool_outputs[5:])
        assert any("Who commanded Apollo 11?" in t for t in observed_tool_outputs[5:])
    finally:
        srv_ok.shutdown()

    # Default max_chars=None accepts > 2000-char instruction unclipped
    long_instr = "L" * 2500
    srv_long, _, _ = _start_openai_stub(
        lambda _i, _p: (200, _xml_tool_reply("Long", [{"name": "submit", "args": {"new_instruction": long_instr}}]))
    )
    try:
        p_long = make_react_instruction_proposer(lm=_make_loopback_lm(srv_long), max_iters=2)
        out_long = p_long({"react": "Old"}, reflective_ds, ["react"])
        assert out_long["react"] == long_instr
    finally:
        srv_long.shutdown()

    # Empty new_instruction -> native ValueError
    srv_empty, _, _ = _start_openai_stub(
        lambda _i, _p: (200, _xml_tool_reply("Empty", [{"name": "submit", "args": {"new_instruction": "   "}}]))
    )
    try:
        p_empty = make_react_instruction_proposer(lm=_make_loopback_lm(srv_empty), max_iters=2)
        with pytest.raises(ValueError, match="empty"):
            p_empty({"react": "Old"}, reflective_ds, ["react"])
    finally:
        srv_empty.shutdown()

    # Over max_chars proposal -> native ValueError (no silent truncation)
    srv_over, _, _ = _start_openai_stub(
        lambda _i, _p: (200, _xml_tool_reply("Too long", [{"name": "submit", "args": {"new_instruction": "Y" * 41}}]))
    )
    try:
        p_over = make_react_instruction_proposer(lm=_make_loopback_lm(srv_over), max_chars=40, max_iters=2)
        with pytest.raises(ValueError, match="exceeds max_chars=40"):
            p_over({"react": "Old"}, reflective_ds, ["react"])
    finally:
        srv_over.shutdown()

    # HTTP 500 -> LMServerError propagates unmasked
    srv_500, _, _ = _start_openai_stub(lambda _i, _p: (500, "reflection backend failure"))
    try:
        p_500 = make_react_instruction_proposer(lm=_make_loopback_lm(srv_500), max_iters=2)
        with pytest.raises(LMServerError):
            p_500({"react": "Old"}, reflective_ds, ["react"])
    finally:
        srv_500.shutdown()


def test_optimize_boundary_guards_locks_and_fresh_preserves_coco_lmdb(tmp_path: Path) -> None:
    from dataclasses import replace

    from yaw.optimize import optimize_dspy, optimize_optuna

    cfg = _make_cfg(tmp_path)

    # Held-out guard for optimize_dspy (explicit and cfg.optimize_query_offset)
    for ds_name in ("2wikimultihopqa", "hotpotqa", "musique", "nq", "popqa"):
        with pytest.raises(YawRuntimeError) as exc_ho:
            optimize_dspy(cfg, datasets=ds_name, query_offset=0, max_queries=20, train_queries=15)
        assert exc_ho.value.code == "INVALID_INPUT"

    with pytest.raises(YawRuntimeError) as exc_cfg_ho:
        optimize_dspy(
            replace(cfg, optimize_query_offset=9), datasets="2wikimultihopqa", max_queries=15, train_queries=10
        )
    assert exc_cfg_ho.value.code == "INVALID_INPUT"
    assert "query_offset < 10" in str(exc_cfg_ho.value)

    with pytest.raises(YawRuntimeError) as exc_max50:
        optimize_dspy(cfg, datasets="2wikimultihopqa", query_offset=10, max_queries=51, train_queries=40)
    assert exc_max50.value.code == "INVALID_INPUT"

    # Mutually exclusive auto vs max_metric_calls
    with pytest.raises(YawRuntimeError) as exc_mut:
        optimize_dspy(cfg, datasets="sample", auto="light", max_metric_calls=10)
    assert exc_mut.value.code == "INVALID_INPUT"

    # Invalid split bounds
    with pytest.raises(YawRuntimeError) as exc_split:
        optimize_dspy(cfg, datasets="sample", max_queries=5, train_queries=5)
    assert exc_split.value.code == "INVALID_INPUT"

    # Corrupt dataset bundle fails fast with INVALID_INPUT
    corrupt_bundle = tmp_path / "corrupt_bundle"
    corrupt_bundle.mkdir(parents=True)
    (corrupt_bundle / "manifest.json").write_text("{corrupt json", encoding="utf-8")
    with pytest.raises(YawRuntimeError) as exc_corrupt:
        optimize_dspy(cfg, datasets=[str(corrupt_bundle)])
    assert exc_corrupt.value.code == "INVALID_INPUT"

    # Held-out guard for optimize_optuna
    with pytest.raises(YawRuntimeError) as exc_opt_ho:
        optimize_optuna(cfg, datasets="2wikimultihopqa", query_offset=9, max_queries=15, train_queries=10)
    assert exc_opt_ho.value.code == "INVALID_INPUT"

    # Lock safety: fresh=True must fail with BUSY and NOT wipe existing files or coco_lmdb when lock is held
    opt_cfg = cfg.for_optimization()
    assert opt_cfg.lock_path is not None and opt_cfg.db_path is not None and opt_cfg.state_dir is not None
    opt_cfg.lock_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_db = opt_cfg.db_path
    sentinel_db.write_text("keep-when-busy", encoding="utf-8")
    coco_marker = opt_cfg.state_dir / "keep_coco.marker"
    coco_marker.parent.mkdir(parents=True, exist_ok=True)
    coco_marker.write_text("coco-lmdb-state", encoding="utf-8")

    held_lock = filelock.FileLock(str(opt_cfg.lock_path), timeout=0)
    held_lock.acquire()
    try:
        with pytest.raises(YawRuntimeError) as exc_busy:
            optimize_dspy(cfg, datasets="sample", max_queries=4, train_queries=2, max_metric_calls=6, fresh=True)
        assert exc_busy.value.code == "BUSY"
        assert sentinel_db.read_text(encoding="utf-8") == "keep-when-busy"
        assert coco_marker.read_text(encoding="utf-8") == "coco-lmdb-state"
    finally:
        held_lock.release()

    # Optuna lock safety
    optuna_db = tmp_path / ".yaw" / "optimize_optuna.db"
    optuna_lock = filelock.FileLock(f"{optuna_db}.lock", timeout=0)
    optuna_lock.acquire()
    try:
        with pytest.raises(YawRuntimeError) as exc_ob:
            optimize_optuna(cfg, datasets="sample", query_offset=0, max_queries=4, train_queries=2, trials=2)
        assert exc_ob.value.code == "BUSY"
    finally:
        optuna_lock.release()


def test_optimize_dspy_end_to_end_gepa_compile_save_reload_and_fresh_cleanup(tmp_path: Path) -> None:
    from yaw.optimize import optimize_dspy

    slice_bundle = _prepare_sample_bundle(tmp_path, dataset_name="custom_slice")
    _prepare_sample_bundle(tmp_path, dataset_name="sample")
    proposer_calls = {"count": 0}

    def chat_responder(_turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
        msgs = payload.get("messages", [])
        sys_text = str(msgs[0].get("content", "")) if msgs else ""
        full_blob = json.dumps(msgs)
        last_content = str(msgs[-1].get("content", "")) if msgs else ""

        if "component_name" in full_blob or "list_examples" in sys_text:
            tool_msgs = [str(m.get("content", "")) for m in msgs if m.get("role") == "tool"]
            for tm in tool_msgs:
                assert '"tools"' not in tm
            if not tool_msgs:
                proposer_calls["count"] += 1
                return 200, _xml_tool_reply(
                    "List reflective examples first",
                    [{"name": "list_examples", "args": {}}],
                )
            if len(tool_msgs) == 1:
                return 200, _xml_tool_reply(
                    "Inspect failing example zero",
                    [{"name": "inspect_example", "args": {"idx": 0, "field": ""}}],
                )
            return 200, _xml_tool_reply(
                "Propose improved instruction",
                [
                    {
                        "name": "submit",
                        "args": {
                            "new_instruction": (
                                "OPTIMIZED_QA_PROMPT: Search passages and submit exact entity names "
                                "with [chunk_id=<int>]."
                            )
                        },
                    }
                ],
            )

        # QA Agent turns
        is_optimized = "OPTIMIZED_QA_PROMPT" in sys_text
        if "[[ ## question ## ]]" in last_content and "search_passages" not in last_content:
            q_match = re.search(r"Who commanded|What childhood instrument|Which valve|What feeds", last_content)
            q_term = q_match.group(0) if q_match else "Apollo 11"
            return 200, _xml_tool_reply(
                "Search corpus passages",
                [{"name": "search_passages", "args": {"query": q_term, "top_k": 3}}],
            )
        cids = [int(m.group(1)) for m in _CID_IN_TEXT_RE.finditer(full_blob)]
        cid = cids[0] if cids else 1
        if "Apollo" in full_blob:
            ans = f"Neil Armstrong [chunk_id={cid}]" if is_optimized else f"Armstrong [chunk_id={cid}]"
        elif "XylophoneZither" in full_blob:
            ans = f"XylophoneZither [chunk_id={cid}]" if is_optimized else f"Zither [chunk_id={cid}]"
        elif "booster" in full_blob.lower():
            ans = "cryo valve" if is_optimized else "unknown valve"
        else:
            ans = "Nordic glacier water" if is_optimized else "unknown water"
        return 200, _xml_tool_reply("Submit grounded answer", [{"name": "submit", "args": {"answer": ans}}])

    srv, _, _ = _start_openai_stub(chat_responder)
    try:
        cfg = _make_cfg(tmp_path, port=srv.server_port)
        res = optimize_dspy(
            cfg,
            datasets=[str(slice_bundle)],
            max_queries=4,
            train_queries=2,
            query_offset=0,
            max_metric_calls=12,
            fresh=True,
        )
        assert proposer_calls["count"] >= 1
        prog_path = Path(res["program_path"])
        eval_path = Path(res["eval_path"])
        audit_path = Path(res["audit_path"])
        assert prog_path.is_file()
        assert eval_path.is_file()
        assert audit_path.is_file()
        assert 0.0 <= res["best_score"] <= 1.0
        assert res["trials"] >= 1

        # Verify slice isolation: unused distractor was NOT staged or indexed
        stage_dir = tmp_path / ".yaw" / "optimize_yaw_corpus"
        assert stage_dir.is_dir()
        staged_text = "\n".join(p.read_text(encoding="utf-8") for p in stage_dir.glob("*.json"))
        assert "EXCLUDED_SLICE_MARKER" not in staged_text

        # Verify optimize_status.json stopped and coco_lmdb exists
        status_obj = json.loads((tmp_path / ".yaw" / "optimize_status.json").read_text(encoding="utf-8"))
        assert status_obj["lifecycle"] == "stopped"
        assert (tmp_path / ".yaw" / "coco_lmdb").is_dir()

        # Verify _eval.json and _audit.json contents
        eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
        assert eval_obj["val_queries"] == 2
        assert eval_obj["best_score"] == pytest.approx(res["best_score"])
        assert len(eval_obj["val_aggregate_scores"]) >= 1
        assert isinstance(eval_obj["best_val_subscores"], dict)
        audit_obj = json.loads(audit_path.read_text(encoding="utf-8"))
        assert len(audit_obj["candidate_records"]) >= 1

        # Verify compiled optimize_program.json reloads cleanly onto a fresh bare dspy.ReAct
        with contextlib.closing(open_readonly_db(tmp_path / ".yaw" / "optimize_yaw.db")) as conn:
            vs = build_vector_search(conn)
            fresh_agent = build_qa_agent(
                make_qa_tools(vs, stage_dir, lambda ts: _topic_embed_vec(ts[0])[None, :], conn)
            )
            fresh_agent.load(str(prog_path))
            reloaded_instr = next(iter(fresh_agent.named_predictors()))[1].signature.instructions
            assert "OPTIMIZED_QA_PROMPT" in reloaded_instr

        # Verify fresh=True preserves coco_lmdb sentinel while wiping stale artifacts
        # and omitting query_offset honors cfg.optimize_query_offset=10 end-to-end.
        from dataclasses import replace

        coco_keep = tmp_path / ".yaw" / "coco_lmdb" / "sentinel_keep.txt"
        coco_keep.write_text("keep-me", encoding="utf-8")
        stale_telem = prog_path.with_name(f"{prog_path.stem}_telemetry") / "stale.txt"
        stale_telem.parent.mkdir(parents=True, exist_ok=True)
        stale_telem.write_text("wipe-me", encoding="utf-8")

        offset10_root = tmp_path / "offset10_bundle"
        off_raw = offset10_root / "raw"
        off_raw.mkdir(parents=True, exist_ok=True)
        sample_corpus = json.loads((tmp_path / "_raw_src" / "corpus.json").read_text(encoding="utf-8"))
        sample_questions = json.loads((tmp_path / "_raw_src" / "questions.json").read_text(encoding="utf-8"))
        (off_raw / "corpus.json").write_text(json.dumps(sample_corpus), encoding="utf-8")
        (off_raw / "questions.json").write_text(json.dumps((sample_questions * 3)[:14]), encoding="utf-8")
        off_bundle = prepare_dataset_bundle(
            "offset10_slice", off_raw / "corpus.json", off_raw / "questions.json", offset10_root / "bundle"
        )

        res2 = optimize_dspy(
            replace(cfg, optimize_query_offset=10),
            datasets=[str(off_bundle.root)],
            max_queries=4,
            train_queries=2,
            max_metric_calls=4,
            fresh=True,
        )
        assert coco_keep.read_text(encoding="utf-8") == "keep-me"
        assert not stale_telem.exists()
        assert Path(res2["program_path"]).is_file()
        assert json.loads(Path(res2["eval_path"]).read_text(encoding="utf-8"))["query_offset"] == 10
    finally:
        srv.shutdown()


def test_optimize_optuna_end_to_end_five_objectives_and_toml_write(tmp_path: Path) -> None:
    from yaw.optimize import optimize_optuna

    bundle_dir = _prepare_sample_bundle(tmp_path, dataset_name="sample")
    toml_file = tmp_path / "yaw.toml"
    toml_file.write_text(
        '[paths]\ncorpus_root = "corpus"\n\n[inference]\nqa_model = "gemma-4-e4b"\n',
        encoding="utf-8",
    )

    srv, _, _ = _start_openai_stub(lambda _i, _p: (200, "unused"))
    try:
        cfg = _make_cfg(tmp_path, port=srv.server_port)
        out_json = tmp_path / ".yaw" / "optuna_out.json"
        res = optimize_optuna(
            cfg,
            datasets=[str(bundle_dir)],
            query_offset=0,
            max_queries=6,
            train_queries=4,
            index_queries=6,
            trials=4,
            study_db=".yaw/optimize_optuna.db",
            fresh=True,
            write_config=True,
            output_path=out_json,
            toml_path=toml_file,
        )
        assert res["trials"] == 4
        assert set(res["best_params"]) == {
            "peclet",
            "alpha_sprt",
            "redundancy_curvature",
            "coalition_coupling",
            "external_trust",
        }
        assert set(res["best_values"]) == {"Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"}
        assert all(0.0 <= v <= 1.0 for v in res["best_values"].values())
        assert "importances" in res
        assert res["importances"] is None or isinstance(res["importances"], dict)

        # Verify SQLite study DB contains real completed trials and native Optuna constraints only
        study_path = Path(res["study_db"])
        assert study_path.is_file()
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_path.as_posix()}")
        try:
            loaded = optuna.load_study(study_name="yaw_retrieval", storage=storage)
            assert len(loaded.trials) == 4
            assert all(
                len(t.constraints) == 3 and set(t.constraints) == {"cycle_count", "seed_deficit", "val_regression"}
                for t in loaded.trials
            )
            assert all("constraints" not in t.user_attrs for t in loaded.trials)
        finally:
            storage.engine.dispose()

        # Verify [retrieval] written cleanly to yaw.toml and zero forbidden section
        toml_text = toml_file.read_text(encoding="utf-8")
        assert "[retrieval]" in toml_text
        assert f"[{''.join(('mani', 'fold'))}]" not in toml_text.lower()
        reloaded_cfg = load_config(toml_file, workspace_override=tmp_path)
        assert reloaded_cfg.retrieval.peclet == pytest.approx(res["best_params"]["peclet"])
        assert out_json.is_file()
        assert json.loads(out_json.read_text(encoding="utf-8"))["trials"] == 4

        # Verify fresh=True wipes prior Optuna DB cleanly and starts fresh trial count
        res_fresh = optimize_optuna(
            cfg,
            datasets="sample",
            query_offset=0,
            max_queries=6,
            train_queries=4,
            index_queries=6,
            trials=2,
            study_db=".yaw/optimize_optuna.db",
            fresh=True,
        )
        assert res_fresh["trials"] == 2
        assert "importances" in res_fresh and res_fresh["importances"] is None
        with contextlib.closing(sqlite3.connect(str(study_path))) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
            assert cnt == 2
    finally:
        srv.shutdown()


def test_optimize_optuna_zero_feasible_trials_fails_fast(tmp_path: Path) -> None:
    from dataclasses import replace

    from yaw.optimize import optimize_optuna

    infeasible_root = tmp_path / "zero_feasible_ds"
    raw_dir = infeasible_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows = [
        {"idx": "doc_train", "title": "Apollo Mission", "text": "Neil Armstrong lunar landing record."},
        {"idx": "doc_gold_val", "title": "Needle", "text": "Entry 749201 archive."},
        *[{"idx": f"doc_d_{i}", "title": f"D{i}", "text": f"glacier tok{i} note."} for i in range(1, 7)],
        *[{"idx": f"doc_bg_{i}", "title": f"BG{i}", "text": f"neutral bg{i} record."} for i in range(7, 13)],
    ]
    questions_rows = [
        {
            "question": "glacier 749201",
            "answer": "Neil Armstrong",
            "contexts": [
                {"title": r["title"], "text": r["text"], "is_supporting": r["idx"] == "doc_train"} for r in corpus_rows
            ],
        },
        {
            "question": "glacier 749201",
            "answer": "749201",
            "contexts": [
                {"title": r["title"], "text": r["text"], "is_supporting": r["idx"] == "doc_gold_val"}
                for r in corpus_rows
            ],
        },
    ]
    c_path = raw_dir / "corpus.json"
    q_path = raw_dir / "questions.json"
    c_path.write_text(json.dumps(corpus_rows), encoding="utf-8")
    q_path.write_text(json.dumps(questions_rows), encoding="utf-8")
    bundle = prepare_dataset_bundle(
        "zero_feasible_sample",
        c_path,
        q_path,
        infeasible_root / "bundle",
        id_key="idx",
        title_key="title",
        body_key="text",
    )

    def _custom_embed(text: str) -> np.ndarray:
        vec = np.zeros(16, dtype=np.float32)
        if text.startswith("task:"):
            vec[0] = 1.0
        elif "749201" in text:
            vec[0] = 0.72
            vec[15] = 1.0
        else:
            matched = False
            for i in range(1, 7):
                if f"tok{i}" in text:
                    vec[0] = 0.75 + 0.02 * i
                    vec[i] = 1.0
                    matched = True
                    break
            if not matched:
                for i in range(7, 13):
                    if f"bg{i}" in text:
                        vec[0] = 0.1 * (i - 6)
                        vec[i] = 1.0
                        matched = True
                        break
            if not matched:
                vec[14] = 1.0
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    srv, _, _ = _start_openai_stub(lambda _i, _p: (200, "unused"), embed_fn=_custom_embed)
    try:
        cfg = replace(
            _make_cfg(tmp_path, port=srv.server_port),
            retrieval=RetrievalParams(peclet=1.0, external_trust=1.0),
        )
        with pytest.raises(YawRuntimeError) as exc_zf:
            optimize_optuna(
                cfg,
                datasets=[str(bundle.root)],
                query_offset=0,
                max_queries=2,
                train_queries=1,
                index_queries=2,
                trials=3,
                study_db=".yaw/zero_feasible.db",
                fresh=True,
            )
        assert exc_zf.value.code == "MODEL_ERROR"
        assert "zero feasible trials" in str(exc_zf.value).lower()
    finally:
        srv.shutdown()
