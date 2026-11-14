"""The grammar/logprob completion engine against a real loopback HTTP server (no mocks)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import dspy
import pytest
from dspy._vendor.lm15.errors import AuthError, ContextLengthError
from dspy.lm15 import Config, Message, Request

from yaw.daemon.graph.inference.chat import create_dspy_lm
from yaw.daemon.graph.inference.completions import GemmaCompletionsEngine, render_gemma4_prompt

GRAMMAR = 'root ::= "{\\"answer\\": \\"" [a-z]+ "\\"}"'
TEXT = '{"answer": "ok"}'


def _token(text: str, logprob: float) -> dict[str, Any]:
    return {
        "id": 1,
        "token": text,
        "bytes": list(text.encode()),
        "logprob": logprob,
        "top_logprobs": [{"id": 1, "token": text, "bytes": list(text.encode()), "logprob": logprob}],
    }


@pytest.fixture
def completions_server():
    state: dict[str, Any] = {"requests": [], "status": 200, "error_body": "", "finish_reason": "stop"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, self.headers["Authorization"], body))
            if state["status"] != 200:
                self.send_response(state["status"])
                self.end_headers()
                self.wfile.write(state["error_body"].encode())
                return
            text = state.get("text", TEXT)
            payload = {
                "id": "cmpl-1",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": state["finish_reason"],
                        "logprobs": (
                            {"content": [_token(text[:8], -0.5), _token(text[8:], -0.25)]}
                            if state["finish_reason"] == "stop"
                            else None
                        ),
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
                "timings": {"predicted_n": 3, "predicted_ms": 20.0, "predicted_per_second": 150.0},
            }
            if self.path == "/v1/chat/completions":
                payload["object"] = "chat.completion"
                payload["created"] = 0
                payload["choices"][0]["message"] = {"role": "assistant", "content": text}
                del payload["choices"][0]["text"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()


class Answer(dspy.Signature):
    """Answer tersely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def test_render_matches_the_gemma4_template_without_bos_or_thinking():
    prompt = render_gemma4_prompt("  Be terse. ", [("user", "Q1 "), ("assistant", "A1"), ("user", "Q2")])
    assert prompt == (
        "<|turn>system\nBe terse.<turn|>\n<|turn>user\nQ1<turn|>\n<|turn>model\nA1<turn|>\n"
        "<|turn>user\nQ2<turn|>\n<|turn>model\n"
    )
    assert render_gemma4_prompt(None, [("user", "hi")]) == "<|turn>user\nhi<turn|>\n<|turn>model\n"
    assert "<bos>" not in prompt
    assert "<|think|>" not in prompt


def test_render_rejects_unknown_roles():
    with pytest.raises(ValueError, match="Unsupported chat role"):
        render_gemma4_prompt(None, [("tool", "x")])


def test_worker_calls_use_text_completions_with_grammar_and_logprobs(completions_server):
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        pred = dspy.Predict(Answer)(question="why", config={"logprobs": True, "grammar": GRAMMAR})
    ((path, auth, body),) = completions_server["requests"]
    assert path == "/v1/completions"
    assert auth == "Bearer k"
    assert body["grammar"] == GRAMMAR
    assert body["logprobs"] == 1
    assert body["model"] == "gemma-4-e4b"
    assert body["prompt"].startswith("<|turn>system\n") and body["prompt"].endswith("<|turn>model\n")
    assert "messages" not in body
    assert pred.answer == "ok"
    content = pred.logprobs
    assert [item.logprob for item in content] == [-0.5, -0.25]
    assert bytes(content[0].bytes) + bytes(content[1].bytes) == TEXT.encode()
    assert lm.history[-1]["usage"]["prompt_tokens"] == 11


def test_no_logprobs_are_requested_unless_asked(completions_server):
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        dspy.Predict(Answer)(question="why", config={"grammar": GRAMMAR})
    assert "logprobs" not in completions_server["requests"][0][2]


def test_orchestrator_roles_keep_the_standard_chat_path():
    worker = create_dspy_lm("gemma-4-e4b", role="extraction", api_key="k")
    for role in ("rlm", "reflection"):
        lm = create_dspy_lm("gemma-4-e4b", role=role, api_key="k")
        assert not isinstance(lm.engine, GemmaCompletionsEngine)
        assert lm._yaw_supports_gbnf is False
    assert isinstance(worker.engine, GemmaCompletionsEngine)
    assert worker._yaw_supports_gbnf is True


def test_typed_request_carries_the_grammar_in_config_extensions(completions_server):
    engine = GemmaCompletionsEngine(completions_server["base_url"], "k")
    request = Request(
        model="openai/gemma-4-e4b",
        system="Be terse.",
        messages=(Message.user("hi"),),
        config=Config(max_tokens=20, logprobs=1, extensions={"grammar": GRAMMAR}),
    )
    response = engine.complete(request)
    body = completions_server["requests"][0][2]
    assert body["grammar"] == GRAMMAR and body["max_tokens"] == 20 and body["logprobs"] == 1
    assert body["prompt"] == "<|turn>system\nBe terse.<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
    assert response.text == TEXT
    assert response.usage.input_tokens == 11
    assert response.provider_data == {
        "timings": {"predicted_n": 3, "predicted_ms": 20.0, "predicted_per_second": 150.0}
    }


def test_auth_failure_is_the_canonical_lm15_error(completions_server):
    completions_server["status"] = 401
    completions_server["error_body"] = '{"error": {"message": "bad key"}}'
    engine = GemmaCompletionsEngine(completions_server["base_url"], "wrong")
    with pytest.raises(AuthError):
        engine.complete(Request(model="gemma-4-e4b", messages=(Message.user("hi"),), config=Config()))


def test_context_overflow_is_the_canonical_lm15_error(completions_server):
    completions_server["status"] = 400
    completions_server["error_body"] = '{"error": {"type": "exceed_context_size_error", "message": "too long"}}'
    engine = GemmaCompletionsEngine(completions_server["base_url"], "k")
    with pytest.raises(ContextLengthError):
        engine.complete(Request(model="gemma-4-e4b", messages=(Message.user("hi"),), config=Config()))


def test_non_gemma_worker_keeps_its_native_chat_template():
    lm = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", api_key="k")
    assert not isinstance(lm.engine, GemmaCompletionsEngine)
    assert lm._yaw_supports_gbnf is True


def test_program_checkpoint_preserves_deployment_lm_without_saving_credentials(tmp_path):
    from yaw.daemon.graph.inference.program import YawProgram

    lm = create_dspy_lm("gemma-4-e4b", api_key="checkpoint-secret", base_url="http://127.0.0.1:12345/v1")
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    path = tmp_path / "program.json"
    program.save(str(path))
    text = path.read_text()
    assert "checkpoint-secret" not in text
    assert "GemmaCompletionsEngine" in text
    deployment = create_dspy_lm("gemma-4-e4b", api_key="new-secret", base_url="http://127.0.0.1:12346/v1")
    restored = YawProgram(orchestrator_lm=deployment, rlm_sub_lm=deployment)
    restored.load(path)
    assert restored.ner.lm is deployment
    assert restored.triples.lm is deployment
    assert isinstance(restored.ner.lm.engine, GemmaCompletionsEngine)
    assert restored.ner.lm.engine.api_key == "new-secret"


def test_engine_state_restores_credentials_from_environment(monkeypatch):
    engine = GemmaCompletionsEngine("http://127.0.0.1:12345/v1", "old-secret", timeout=17)
    state = engine.dump_state()
    assert state == {"base_url": "http://127.0.0.1:12345/v1", "timeout": 17}
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", "new-secret")
    restored = GemmaCompletionsEngine.load_state(state)
    assert restored.api_key == "new-secret" and restored.timeout == 17


def test_cached_worker_calls_are_isolated_by_completion_endpoint(completions_server, tmp_path):
    from yaw.daemon.graph.inference.chat import configure_yaw_dspy_cache

    configure_yaw_dspy_cache(tmp_path / "cache")
    try:
        first = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k")
        second = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"] + "/other/v1", api_key="k")
        predictor = dspy.Predict(Answer)
        with dspy.context(adapter=dspy.JSONAdapter()):
            for lm in (first, first, second):
                with dspy.context(lm=lm):
                    assert predictor(question="why").answer == "ok"
        assert len(completions_server["requests"]) == 2
        assert completions_server["requests"][0][0] != completions_server["requests"][1][0]
    finally:
        configure_yaw_dspy_cache(enable_disk_cache=False, enable_memory_cache=False)


def test_async_adapter_uses_the_same_grammar_completion_transport(completions_server):
    import asyncio

    from yaw.daemon.graph.inference.adapter import YawGBNFAdapter

    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)

    async def call():
        with dspy.context(lm=lm, adapter=YawGBNFAdapter(enable_gbnf=True)):
            return await dspy.Predict(Answer).acall(question="why", config={"logprobs": True})

    pred = asyncio.run(call())
    ((path, auth, body),) = completions_server["requests"]
    assert path == "/v1/completions" and auth == "Bearer k"
    assert "grammar" in body and body["logprobs"] == 1
    assert pred.answer == "ok"
    assert pred.logprobs[0].logprob == -0.5


def test_length_finished_generation_is_rejected_even_when_json_is_complete(completions_server):
    from dspy.utils.exceptions import LMUnexpectedError

    completions_server["finish_reason"] = "length"
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    with (
        dspy.context(lm=lm, adapter=dspy.JSONAdapter()),
        pytest.raises(LMUnexpectedError, match="Incomplete generation"),
    ):
        dspy.Predict(Answer)(question="why", config={"grammar": GRAMMAR})
    assert len(completions_server["requests"]) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_chat_rejects_length_finished_json_and_does_not_cache_it(completions_server, tmp_path, asynchronous):
    import asyncio

    from yaw.daemon.graph.inference.chat import configure_yaw_dspy_cache

    configure_yaw_dspy_cache(tmp_path / "native-cache")
    lm = create_dspy_lm("native-model", base_url=completions_server["base_url"], api_key="k")
    predictor = dspy.Predict(Answer)

    def call():
        with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
            return asyncio.run(predictor.acall(question="why")) if asynchronous else predictor(question="why")

    try:
        completions_server["finish_reason"] = "length"
        with pytest.raises(ValueError, match="Incomplete generation"):
            call()
        completions_server["finish_reason"] = "stop"
        assert call().answer == "ok"
        assert call().answer == "ok"
        assert len(completions_server["requests"]) == 2
    finally:
        configure_yaw_dspy_cache(enable_disk_cache=False, enable_memory_cache=False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_direct_chat_rejects_length_before_text_conversion(completions_server, tmp_path, asynchronous, cache):
    import asyncio

    from yaw.daemon.graph.inference.chat import configure_yaw_dspy_cache

    configure_yaw_dspy_cache(tmp_path / "direct-cache")
    lm = create_dspy_lm("native-model", base_url=completions_server["base_url"], api_key="k", cache=cache)

    def call():
        return asyncio.run(lm.acall("why")) if asynchronous else lm("why")

    try:
        completions_server["finish_reason"] = "length"
        with pytest.raises(ValueError, match="Incomplete generation"):
            call()
        completions_server["finish_reason"] = "stop"
        assert call() == ['{"answer": "ok"}']
        assert len(completions_server["requests"]) == 2
    finally:
        configure_yaw_dspy_cache(enable_disk_cache=False, enable_memory_cache=False)


def test_complete_reasoning_longer_than_the_old_360_character_limit_survives(completions_server):
    from yaw.daemon.graph.inference.adapter import YawGBNFAdapter
    from yaw.daemon.graph.inference.prompts import JudgeRelations

    reasoning = "The source describes a resource constraint and the mitigation applied to it. " * 8
    decision = {"pair_id": "e0|e1", "reasoning": reasoning, "relation": "skos:related"}
    completions_server["text"] = json.dumps({"decisions": [decision]})
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    with dspy.context(lm=lm, adapter=YawGBNFAdapter(enable_gbnf=True)):
        result = dspy.Predict(JudgeRelations)(candidate_pairs="Full evidence")
    assert result.decisions[0]["reasoning"] == reasoning
    assert len(result.decisions[0]["reasoning"]) > 360
    grammar = completions_server["requests"][0][2]["grammar"]
    assert 'char+ "\\""' in grammar
    assert "{1,360}" not in grammar


def test_qa_grammar_matches_native_wire_plan_and_restores_reasoning(completions_server):
    from yaw.daemon.graph.inference.program import YawProgram

    completions_server["text"] = json.dumps({"plan": "The source names Ada.", "answer": "Ada"})
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=True)
    prediction = program.answer_question(context="Ada wrote Notes.", question="Who wrote Notes?")
    assert prediction.answer == "Ada"
    assert str(prediction.reasoning) == "The source names Ada."
    ((_, _, body),) = completions_server["requests"]
    assert "`plan`" in body["prompt"]
    assert '\\"plan\\"' in body["grammar"].splitlines()[0]
    assert '\\"reasoning\\"' not in body["grammar"].splitlines()[0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("partial", ["Partial café reasoning " * 600 + "PARTIAL_TAIL", chr(0xD800)])
def test_length_failure_saves_complete_header_free_diagnostic(completions_server, tmp_path, asynchronous, partial):
    import asyncio

    from dspy.utils.exceptions import LMUnexpectedError

    completions_server["text"] = partial
    completions_server["finish_reason"] = "length"
    path = tmp_path / "failures"
    lm = create_dspy_lm(
        "gemma-4-e4b",
        base_url=completions_server["base_url"],
        api_key="transport-secret",
        max_tokens=3,
        cache=False,
        diagnostic_dir=path,
    )
    predictor = dspy.Predict(Answer)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()), pytest.raises(LMUnexpectedError) as caught:
        if asynchronous:
            asyncio.run(predictor.acall(question="Complete this evidence."))
        else:
            predictor(question="Complete this evidence.")
    assert "requested max_tokens=3" in str(caught.value)
    assert "'prompt_tokens': 11" in str(caught.value)
    assert "'completion_tokens': 3" in str(caught.value)
    assert len(completions_server["requests"]) == 1
    [artifact] = path.glob("length-*.json")
    raw = artifact.read_text(encoding="utf-8")
    diagnostic = json.loads(raw)
    assert diagnostic["request"] == completions_server["requests"][0][2]
    assert diagnostic["response"]["choices"][0]["text"] == partial
    assert diagnostic["response"]["usage"]["total_tokens"] == 14
    assert "transport-secret" not in raw and "Authorization" not in diagnostic["request"]
    assert isinstance(caught.value.__cause__, ValueError)
    assert any(str(artifact) in note for note in caught.value.__cause__.__notes__)
    completions_server["finish_reason"] = "stop"
    completions_server["text"] = TEXT
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        assert predictor(question="Complete this evidence.").answer == "ok"
    assert len(list(path.glob("length-*.json"))) == 1


def test_diagnostic_io_failure_preserves_original_incomplete_generation(completions_server, tmp_path):
    from dspy.utils.exceptions import LMUnexpectedError

    blocked = tmp_path / "blocked"
    blocked.write_text("existing file", encoding="utf-8")
    completions_server["finish_reason"] = "length"
    lm = create_dspy_lm(
        "gemma-4-e4b",
        base_url=completions_server["base_url"],
        api_key="k",
        cache=False,
        diagnostic_dir=blocked,
    )
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()), pytest.raises(LMUnexpectedError) as caught:
        dspy.Predict(Answer)(question="why")
    assert isinstance(caught.value.__cause__, ValueError)
    assert "finish_reason='length'" in str(caught.value)
    assert any("Could not save failure diagnostic" in note for note in caught.value.__cause__.__notes__)
    assert len(completions_server["requests"]) == 1
    assert blocked.read_text(encoding="utf-8") == "existing file"


def test_configured_ner_failure_identifies_stage_and_saves_payload(completions_server, tmp_path):
    from dspy.utils.exceptions import LMUnexpectedError

    from yaw.core.config import YawConfig
    from yaw.daemon.graph.inference.program import YawProgram

    completions_server["finish_reason"] = "length"
    config = YawConfig(
        workspace_root=tmp_path, base_url=completions_server["base_url"], api_key="k", max_output_tokens=3
    )
    program = YawProgram.from_config(config)
    with pytest.raises(LMUnexpectedError) as caught:
        program.extract_openie("chunk-ner-failure", "Café is a business.", lm=program.rlm_sub_lm)
    assert "YAW stage: ner" in caught.value.__notes__
    assert len(list((tmp_path / "outputs" / "inference_failures").glob("length-*.json"))) == 1
    assert len(completions_server["requests"]) == 1


def test_native_length_failure_without_requested_limit(completions_server, tmp_path):
    completions_server["finish_reason"] = "length"
    engine = GemmaCompletionsEngine(completions_server["base_url"], "k", diagnostic_dir=tmp_path)
    with pytest.raises(ValueError, match="requested max_tokens=None"):
        engine.complete(Request(model="gemma-4-e4b", messages=(Message.user("hi"),), config=Config()))
    [artifact] = tmp_path.glob("length-*.json")
    assert json.loads(artifact.read_text(encoding="utf-8"))["request"] == completions_server["requests"][0][2]


def test_cocoindex_ner_failure_preserves_stage_and_chunk(completions_server, tmp_path):
    import cocoindex as coco

    from yaw.core.config import YawConfig
    from yaw.daemon.graph.inference.components import OPENIE_ENDPOINT, OpenIEEndpoint, OpenIEInput, extract_text
    from yaw.daemon.graph.inference.program import YawProgram

    completions_server["finish_reason"] = "length"
    program = YawProgram.from_config(
        YawConfig(workspace_root=tmp_path, base_url=completions_server["base_url"], api_key="k", max_output_tokens=3)
    )
    context = coco.ContextProvider()
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, program.rlm_sub_lm, "failure-test"))
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="ner_failure_context", environment=env),
        extract_text,
        OpenIEInput("chunk-ner-failure", "Café is a business.", "failure-test"),
    )
    try:
        with pytest.raises(Exception) as caught:
            app.update_blocking()
        import traceback

        failure = "".join(traceback.format_exception(caught.value))
        assert "YAW stage: ner" in failure
        assert "YAW extraction chunk: chunk-ner-failure" in failure
        assert "requested max_tokens=3" in failure
        assert "Full request and partial response saved to" in failure
        assert len(completions_server["requests"]) == 1
    finally:
        env.close()


@pytest.mark.parametrize("role", ["extraction", "rlm"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit", [None, 123])
def test_generation_limit_is_only_sent_when_explicit(completions_server, role, asynchronous, limit):
    import asyncio

    lm = create_dspy_lm(
        "gemma-4-e4b",
        role=role,
        base_url=completions_server["base_url"],
        api_key="k",
        max_tokens=limit,
        context_window_tokens=131072,
        cache=False,
    )
    predictor = dspy.Predict(Answer)
    with dspy.context(lm=lm.copy(), adapter=dspy.JSONAdapter()):
        result = asyncio.run(predictor.acall(question="why")) if asynchronous else predictor(question="why")
    assert result.answer == "ok"
    [(path, _, body)] = completions_server["requests"]
    assert path == ("/v1/completions" if role == "extraction" else "/v1/chat/completions")
    if limit is None:
        assert "max_tokens" not in body and "max_completion_tokens" not in body
    else:
        assert body.get("max_tokens", body.get("max_completion_tokens")) == limit
