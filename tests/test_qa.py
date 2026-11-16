"""Unmocked integration tests for bare dspy.ReAct QA tools, grounding, and error boundaries."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import pytest
from dspy.utils.exceptions import LMServerError
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError
from yaw.pipeline import close_pipeline, start_pipeline, update_pipeline_once
from yaw.qa import AnswerQuestion, EmbedFn, ask_question, build_qa_agent, make_qa_tools
from yaw.retrieval import build_vector_search, open_readonly_db

_CID_IN_PROMPT_RE = re.compile(r"\[chunk_id=(\d+)\]")


def _topic_embed(texts: list[str], *, dim: int = 16) -> NDArray[np.float32]:
    topics = (
        ("apollo", "lunar", "moon", "armstrong", "commander"),
        ("instrument", "xylophone", "zither", "music", "wapakoneta", "childhood"),
        ("orbital", "booster", "valve", "specification", "stage"),
        ("longsection", "chronicle", "paragraph", "archive"),
    )
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        low = text.lower()
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        base = (np.frombuffer(digest * 2, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.05
        for t_idx, kws in enumerate(topics):
            for kw in kws:
                if kw in low:
                    base[t_idx] += 2.0
        rows[row_idx] = base / max(float(np.linalg.norm(base)), 1e-12)
    return rows


def _make_cfg(root: Path, *, chunk_size: int = 1500, chunk_overlap: int = 0, min_chunk_size: int = 20) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )


def _index_qa_corpus(root: Path, *, single_doc: bool = False) -> YawConfig:
    cfg = _make_cfg(root)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True, exist_ok=True)
    if single_doc:
        (cfg.corpus_root / "solo.md").write_text(
            "# Solo Note\nNeil Armstrong was the Apollo 11 lunar module commander.\n",
            encoding="utf-8",
        )
    else:
        (cfg.corpus_root / "alpha.md").write_text(
            "# Apollo 11 Mission\nNeil Armstrong served as the Apollo 11 Lunar Module commander on the Moon.\n",
            encoding="utf-8",
        )
        para_a = " ".join(
            f"Sentence {i}: Neil Armstrong grew up in Ohio (WapakonetaArchiveAlpha)." for i in range(1, 21)
        )
        para_b = "Neil Armstrong played the XylophoneZither during childhood in Ohio. " + " ".join(
            f"Music note {i} from childhood collection." for i in range(1, 28)
        )
        (cfg.corpus_root / "bridge.md").write_text(
            f"# Wapakoneta Musician\n{para_a}\n\n{para_b}\n",
            encoding="utf-8",
        )
        multi_paras = [
            f"Section {k}: " + " ".join(f"Orbital booster calibration item {k}_{j}." for j in range(1, 30))
            for k in range(1, 8)
        ]
        (cfg.corpus_root / "multi.json").write_text(
            json.dumps({"title": "Booster Manual", "text": "\n\n".join(multi_paras)}),
            encoding="utf-8",
        )
        (cfg.corpus_root / "booster_backup.md").write_text(
            "# Booster Backup\nOrbital booster backup valve specification for emergency stage separation.\n",
            encoding="utf-8",
        )

    session = start_pipeline(cfg, embed_fn=lambda ts: _topic_embed(ts, dim=16), vector_dim=16, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)
    return cfg


def _start_loopback_chat_server(
    responder: Any,
    *,
    delay_sec: float = 0.0,
) -> tuple[ThreadingHTTPServer, list[dict[str, Any]], dict[str, int]]:
    requests_seen: list[dict[str, Any]] = []
    concurrency = {"in_flight": 0, "max_in_flight": 0}
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with lock:
                concurrency["in_flight"] += 1
                concurrency["max_in_flight"] = max(concurrency["max_in_flight"], concurrency["in_flight"])
                turn_idx = len(requests_seen)
                requests_seen.append(payload)
            try:
                if delay_sec > 0:
                    time.sleep(delay_sec)
                status_code, content = responder(turn_idx, payload)
                if status_code != 200:
                    raw = json.dumps({"error": {"message": str(content)}}).encode("utf-8")
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                body = {
                    "id": f"chatcmpl-{turn_idx}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "gemma-4-e4b"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
                raw = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            finally:
                with lock:
                    concurrency["in_flight"] -= 1

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


def test_qa_module_isolation_and_zero_wrappers() -> None:
    qa_path = Path(__file__).resolve().parent.parent / "yaw" / "qa.py"
    src = qa_path.read_text(encoding="utf-8")
    low = src.lower()
    assert "manifold" not in low
    assert "hmm" not in low
    assert "hipporag" not in low

    tree = ast.parse(src, filename=str(qa_path))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert class_names <= {"EmbedFn", "AnswerQuestion"}
    assert issubclass(AnswerQuestion, dspy.Signature)
    assert not issubclass(AnswerQuestion, dspy.Module)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("cocoindex", "yaw.daemon", "yaw.pipeline", "sqlite_vec", "spacy"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(("cocoindex", "yaw.daemon", "yaw.pipeline", "sqlite_vec", "spacy"))

    def _dummy_tool(query: str) -> str:
        return query

    agent = build_qa_agent([_dummy_tool])
    assert type(agent) is dspy.ReAct
    assert agent.max_iters == 6
    assert set(agent.tools) == {"_dummy_tool", "submit"}
    assert EmbedFn is not None


def test_qa_tools_contracts_caps_and_maxp_pooling(tmp_path: Path) -> None:
    cfg = _index_qa_corpus(tmp_path / "main")
    assert cfg.corpus_root is not None and cfg.db_path is not None
    embed_calls: list[list[str]] = []

    def tracked_embed(texts: list[str]) -> NDArray[np.float32]:
        embed_calls.append(list(texts))
        return _topic_embed(texts, dim=16)

    with open_readonly_db(cfg.db_path) as conn:
        vs = build_vector_search(conn)
        assert vs.size >= 10
        with pytest.raises(ValueError, match="Unsupported embedding task_mode"):
            make_qa_tools(vs, cfg.corpus_root, tracked_embed, conn, task_mode="bogus")

        search_passages, lexical_grep, expand_neighbors, read_chunk = make_qa_tools(
            vs, cfg.corpus_root, tracked_embed, conn
        )

        # Blank queries do not invoke embedder and return standard no-match message
        assert search_passages("   ") == "No matching passages."
        assert len(embed_calls) == 0

        # MaxP pooling must pass pool_k >= 50 so multi.json (7 chunks) does not starve booster_backup.md
        booster_out = search_passages("orbital booster", top_k=2)
        assert len(embed_calls) == 1
        assert "multi.json" in booster_out
        assert "booster_backup.md" in booster_out
        assert len(_CID_IN_PROMPT_RE.findall(booster_out)) == 2

        # Clamp top_k <= 1 and check 1,200 char cropping on WapakonetaArchiveAlpha long chunk
        alpha_search = search_passages("WapakonetaArchiveAlpha", top_k=0)
        assert len(_CID_IN_PROMPT_RE.findall(alpha_search)) == 1
        assert "[... cropped ...]" in alpha_search
        assert len(alpha_search) <= 1350

        # lexical_grep must invoke ZERO embed calls and clamp top_k to 1..10
        calls_before_grep = len(embed_calls)
        assert lexical_grep("   ") == "No matching passages."
        grep_hit = lexical_grep("XylophoneZither", top_k=5)
        assert len(embed_calls) == calls_before_grep
        assert "bridge.md" in grep_hit and "XylophoneZither" in grep_hit

        grep_long = lexical_grep("WapakonetaArchiveAlpha", top_k=99)
        assert len(embed_calls) == calls_before_grep
        assert "[... cropped ...]" in grep_long
        assert len(_CID_IN_PROMPT_RE.findall(grep_long)) <= 10

        # Identify chunk 1 and chunk 2 of bridge.md to verify expand_neighbors and read_chunk
        bridge_cids = [int(p.chunk_id) for p in vs.pointers if p.rel_posix_path == "bridge.md"]
        assert len(bridge_cids) >= 2
        long_cid = int(_CID_IN_PROMPT_RE.findall(grep_long)[0])

        assert expand_neighbors(999999) == "[INVALID chunk_id=999999]"
        assert expand_neighbors(True) == "[INVALID chunk_id=True]"  # type: ignore[arg-type]
        nbr_out = expand_neighbors(long_cid, top_k=3)
        nbr_cids = [int(x) for x in _CID_IN_PROMPT_RE.findall(nbr_out)]
        assert long_cid not in nbr_cids
        assert any(cid in nbr_cids for cid in bridge_cids if cid != long_cid)

        # read_chunk respects score=1.000 header, invalid ID, and AST island expansion capped at 2,400 chars
        assert read_chunk(-1) == "[INVALID chunk_id=-1]"
        chunk_plain = read_chunk(long_cid, include_siblings=False)
        assert f"[chunk_id={long_cid}]" in chunk_plain and "score=1.000:" in chunk_plain
        chunk_island = read_chunk(long_cid, include_siblings=True)
        assert "[... cropped ...]" in chunk_island
        assert 2200 <= len(chunk_island) <= 2550

    # Solo 1-chunk VectorSearch returns No matching passages on expand_neighbors
    solo_cfg = _index_qa_corpus(tmp_path / "solo", single_doc=True)
    assert solo_cfg.corpus_root is not None and solo_cfg.db_path is not None
    with open_readonly_db(solo_cfg.db_path) as s_conn:
        s_vs = build_vector_search(s_conn)
        assert s_vs.size == 1
        solo_tools = make_qa_tools(s_vs, solo_cfg.corpus_root, tracked_embed, s_conn)
        assert solo_tools[2](int(s_vs.chunk_ids[0])) == "No matching passages."


def test_ask_question_multihop_submit_and_zero_db_writes(tmp_path: Path) -> None:
    cfg = _index_qa_corpus(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None

    with open_readonly_db(cfg.db_path) as conn:
        vs = build_vector_search(conn)
        ver_before = conn.execute("PRAGMA data_version").fetchone()[0]
        tools = make_qa_tools(vs, cfg.corpus_root, lambda ts: _topic_embed(ts, dim=16), conn)
        agent = build_qa_agent(tools, max_iters=6)

        observed_cids: list[int] = []

        def responder(turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
            prompt_blob = json.dumps(payload.get("messages", []))
            for raw_cid in _CID_IN_PROMPT_RE.findall(prompt_blob):
                cid = int(raw_cid)
                if cid not in observed_cids:
                    observed_cids.append(cid)
            if turn_idx == 0:
                return 200, _xml_tool_reply(
                    "Search Apollo mission and grep unique instrument",
                    [
                        {"name": "search_passages", "args": {"query": "Apollo 11 commander", "top_k": 3}},
                        {"name": "lexical_grep", "args": {"query": "XylophoneZither", "top_k": 2}},
                    ],
                )
            if turn_idx == 1:
                seed_cid = observed_cids[0]
                return 200, _xml_tool_reply(
                    "Expand neighbors and read sibling context",
                    [
                        {"name": "expand_neighbors", "args": {"chunk_id": seed_cid, "top_k": 2}},
                        {"name": "read_chunk", "args": {"chunk_id": seed_cid, "include_siblings": True}},
                    ],
                )
            c1, c2 = observed_cids[0], observed_cids[1]
            return 200, _xml_tool_reply(
                "Submit multi-hop grounded answer",
                [
                    {
                        "name": "submit",
                        "args": {"answer": f"Neil Armstrong played XylophoneZither [chunk_id={c1}] [chunk_id={c2}]."},
                    }
                ],
            )

        server, requests_seen, _ = _start_loopback_chat_server(responder)
        try:
            lm = _make_loopback_lm(server)
            res = ask_question(
                agent,
                "What unusual instrument did the Apollo 11 commander play?",
                corpus_root=cfg.corpus_root,
                vs=vs,
                lm=lm,
                lm_sem=threading.Semaphore(1),
            )
        finally:
            server.shutdown()

        ver_after = conn.execute("PRAGMA data_version").fetchone()[0]
        assert ver_before == ver_after

    assert len(requests_seen) == 3
    assert all(req.get("temperature") == 1.0 for req in requests_seen)
    assert res.termination_reason == "submit"
    assert res.turns == 3
    assert len(res.citations) == 2
    assert {int(c.pointer.chunk_id) for c in res.citations} == set(observed_cids[:2])
    assert all(not c.stale_on_disk and c.text for c in res.citations)
    assert res.to_dict()["termination_reason"] == "submit"


def test_forced_submit_unannotated_citations_and_semaphore_serialization(tmp_path: Path) -> None:
    cfg = _index_qa_corpus(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None

    with open_readonly_db(cfg.db_path) as conn:
        vs = build_vector_search(conn)
        tools = make_qa_tools(vs, cfg.corpus_root, lambda ts: _topic_embed(ts, dim=16), conn)
        agent_1 = build_qa_agent(tools, max_iters=1)

        def forced_responder(turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
            _ = payload
            if turn_idx == 0:
                return 200, _xml_tool_reply(
                    "Retrieve passages first",
                    [{"name": "lexical_grep", "args": {"query": "XylophoneZither", "top_k": 4}}],
                )
            return 200, _xml_tool_reply(
                "Submit concise unannotated answer",
                [{"name": "submit", "args": {"answer": "XylophoneZither"}}],
            )

        srv1, _, _ = _start_loopback_chat_server(forced_responder)
        try:
            res = ask_question(
                agent_1,
                "What instrument did he play?",
                corpus_root=cfg.corpus_root,
                vs=vs,
                lm=_make_loopback_lm(srv1),
                lm_sem=threading.Semaphore(1),
                max_citations=2,
            )
        finally:
            srv1.shutdown()

        assert res.termination_reason == "forced_submit"
        assert res.turns == 2
        assert 1 <= len(res.citations) <= 2
        assert any("XylophoneZither" in c.text for c in res.citations)

        # Verify Semaphore(1) serializes concurrent ask_question calls strictly to max_in_flight == 1
        agent_conc = build_qa_agent(tools, max_iters=1)

        def quick_submit(turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
            _ = (turn_idx, payload)
            return 200, _xml_tool_reply(
                "Direct insufficient answer",
                [{"name": "submit", "args": {"answer": "Insufficient references in corpus"}}],
            )

        srv2, _, concurrency = _start_loopback_chat_server(quick_submit, delay_sec=0.04)
        try:
            lm2 = _make_loopback_lm(srv2)
            shared_sem = threading.Semaphore(1)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [
                    pool.submit(
                        ask_question,
                        agent_conc,
                        f"Question {i}?",
                        corpus_root=cfg.corpus_root,
                        vs=vs,
                        lm=lm2,
                        lm_sem=shared_sem,
                    )
                    for i in range(4)
                ]
                results = [f.result() for f in futs]
        finally:
            srv2.shutdown()

    assert len(results) == 4
    assert all(r.citations == () for r in results)
    assert concurrency["max_in_flight"] == 1


def test_ask_question_grounding_consistency_and_error_boundaries(tmp_path: Path) -> None:
    cfg = _index_qa_corpus(tmp_path)
    assert cfg.corpus_root is not None and cfg.db_path is not None

    with open_readonly_db(cfg.db_path) as conn:
        vs = build_vector_search(conn)
        tools = make_qa_tools(vs, cfg.corpus_root, lambda ts: _topic_embed(ts, dim=16), conn)
        agent = build_qa_agent(tools, max_iters=2)

        def unused_responder(turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
            _ = (turn_idx, payload)
            return 200, _xml_tool_reply("Unused", [{"name": "submit", "args": {"answer": "unused"}}])

        srv_unused, reqs_unused, _ = _start_loopback_chat_server(unused_responder)
        try:
            lm_unused = _make_loopback_lm(srv_unused)
            sem = threading.Semaphore(1)

            with pytest.raises(YawRuntimeError) as exc_blank:
                ask_question(agent, "   ", corpus_root=cfg.corpus_root, vs=vs, lm=lm_unused, lm_sem=sem)
            assert exc_blank.value.code == "INVALID_INPUT"

            with pytest.raises(YawRuntimeError) as exc_mc:
                ask_question(
                    agent,
                    "Valid question?",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=lm_unused,
                    lm_sem=sem,
                    max_citations=0,
                )
            assert exc_mc.value.code == "INVALID_INPUT"

            empty_vs = type(vs)(
                (),
                {},
                (),
                (),
                (),
                np.zeros((0, 0), np.float32),
                np.zeros((0, 0), np.float32),
                np.zeros((0, 0), np.float32),
                0,
                1.0,
                np.zeros((0,), np.float32),
                {},
                (),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.float32),
                (),
                (),
                (),
                vs.params,
                0,
            )
            with pytest.raises(YawRuntimeError) as exc_empty:
                ask_question(agent, "Valid?", corpus_root=cfg.corpus_root, vs=empty_vs, lm=lm_unused, lm_sem=sem)
            assert exc_empty.value.code == "EMPTY_INDEX"
            assert reqs_unused == []
        finally:
            srv_unused.shutdown()

        # Non-existent chunk_id citation -> MODEL_ERROR
        srv_bad_id, _, _ = _start_loopback_chat_server(
            lambda _i, _p: (
                200,
                _xml_tool_reply("Invent id", [{"name": "submit", "args": {"answer": "Claim [chunk_id=777777]"}}]),
            )
        )
        try:
            with pytest.raises(YawRuntimeError) as exc_bad_id:
                ask_question(
                    agent,
                    "Who commanded?",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=_make_loopback_lm(srv_bad_id),
                    lm_sem=threading.Semaphore(1),
                )
            assert exc_bad_id.value.code == "MODEL_ERROR"
        finally:
            srv_bad_id.shutdown()

        # Real chunk_id present in vs.id_to_pos & mentioned in next_thought, but NEVER returned by a tool -> MODEL_ERROR
        real_unseen_cid = int(vs.chunk_ids[0])

        def hallucinated_real_id(turn_idx: int, _p: dict[str, Any]) -> tuple[int, str]:
            if turn_idx == 0:
                return 200, _xml_tool_reply(
                    f"Thinking about [chunk_id={real_unseen_cid}] without reading it",
                    [{"name": "read_chunk", "args": {"chunk_id": 999999}}],
                )
            return 200, _xml_tool_reply(
                f"Still referencing unseen [chunk_id={real_unseen_cid}]",
                [{"name": "submit", "args": {"answer": f"Grounded? [chunk_id={real_unseen_cid}]"}}],
            )

        srv_unseen, _, _ = _start_loopback_chat_server(hallucinated_real_id)
        try:
            with pytest.raises(YawRuntimeError) as exc_unseen:
                ask_question(
                    agent,
                    "Test ungrounded real id",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=_make_loopback_lm(srv_unseen),
                    lm_sem=threading.Semaphore(1),
                )
            assert exc_unseen.value.code == "MODEL_ERROR"
        finally:
            srv_unseen.shutdown()

        # Whitespace-only submitted answer -> MODEL_ERROR
        srv_ws, _, _ = _start_loopback_chat_server(
            lambda _i, _p: (200, _xml_tool_reply("Empty", [{"name": "submit", "args": {"answer": "   "}}]))
        )
        try:
            with pytest.raises(YawRuntimeError) as exc_ws:
                ask_question(
                    agent,
                    "Empty answer test?",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=_make_loopback_lm(srv_ws),
                    lm_sem=threading.Semaphore(1),
                )
            assert exc_ws.value.code == "MODEL_ERROR"
        finally:
            srv_ws.shutdown()

        # Unparseable LM output on all turns -> MODEL_ERROR
        srv_garbage, _, _ = _start_loopback_chat_server(lambda _i, _p: (200, "not valid xml or json at all"))
        try:
            with pytest.raises(YawRuntimeError) as exc_parse:
                ask_question(
                    agent,
                    "Malformed output test?",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=_make_loopback_lm(srv_garbage),
                    lm_sem=threading.Semaphore(1),
                )
            assert exc_parse.value.code == "MODEL_ERROR"
        finally:
            srv_garbage.shutdown()

        # Stale/mutated cited file -> CONSISTENCY_ERROR
        alpha_cid = next(int(p.chunk_id) for p in vs.pointers if p.rel_posix_path == "alpha.md")

        def stale_responder(turn_idx: int, _p: dict[str, Any]) -> tuple[int, str]:
            if turn_idx == 0:
                return 200, _xml_tool_reply("Read alpha", [{"name": "read_chunk", "args": {"chunk_id": alpha_cid}}])
            (cfg.corpus_root / "alpha.md").write_text("# Mutated\nChanged after read!\n", encoding="utf-8")
            return 200, _xml_tool_reply(
                "Cite alpha", [{"name": "submit", "args": {"answer": f"Armstrong [chunk_id={alpha_cid}]"}}]
            )

        srv_stale, _, _ = _start_loopback_chat_server(stale_responder)
        try:
            with pytest.raises(YawRuntimeError) as exc_stale:
                ask_question(
                    agent,
                    "Who commanded Apollo 11?",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=_make_loopback_lm(srv_stale),
                    lm_sem=threading.Semaphore(1),
                )
            assert exc_stale.value.code == "CONSISTENCY_ERROR"
        finally:
            srv_stale.shutdown()

        # HTTP 500 server error must propagate unmasked (LMServerError), NOT be wrapped or swallowed
        srv_500, _, _ = _start_loopback_chat_server(lambda _i, _p: (500, "backend exploded"))
        try:
            with pytest.raises(LMServerError):
                ask_question(
                    agent,
                    "Server down test?",
                    corpus_root=cfg.corpus_root,
                    vs=vs,
                    lm=_make_loopback_lm(srv_500),
                    lm_sem=threading.Semaphore(1),
                )
        finally:
            srv_500.shutdown()
