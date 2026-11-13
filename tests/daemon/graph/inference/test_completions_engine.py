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
    state: dict[str, Any] = {"requests": [], "status": 200, "error_body": ""}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, self.headers["Authorization"], body))
            if state["status"] != 200:
                self.send_response(state["status"])
                self.end_headers()
                self.wfile.write(state["error_body"].encode())
                return
            payload = {
                "id": "cmpl-1",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": TEXT,
                        "finish_reason": "stop",
                        "logprobs": {"content": [_token(TEXT[:8], -0.5), _token(TEXT[8:], -0.25)]},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
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
        pred = dspy.Predict(Answer)(question="why", config={"logprobs": True, "extra_body": {"grammar": GRAMMAR}})
    ((path, auth, body),) = completions_server["requests"]
    assert path == "/v1/completions"
    assert auth == "Bearer k"
    assert body["grammar"] == GRAMMAR
    assert body["logprobs"] == 1
    assert body["model"] == "gemma-4-e4b"
    assert body["prompt"].startswith("<|turn>system\n") and body["prompt"].endswith("<|turn>model\n")
    assert "messages" not in body
    assert pred.answer == "ok"
    content = pred.logprobs["content"]
    assert [item["logprob"] for item in content] == [-0.5, -0.25]
    assert bytes(content[0]["bytes"]) + bytes(content[1]["bytes"]) == TEXT.encode()
    assert lm.history[-1]["usage"]["prompt_tokens"] == 11


def test_no_logprobs_are_requested_unless_asked(completions_server):
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        dspy.Predict(Answer)(question="why", config={"extra_body": {"grammar": GRAMMAR}})
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
