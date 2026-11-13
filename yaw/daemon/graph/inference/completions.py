"""Grammar-constrained worker completions for llama.cpp-class servers (spec: dspy-improvements.md §6, Route B).

Unsloth's chat route ignores ``grammar`` and rejects ``logprobs``; its ``/v1/completions`` route enforces
``grammar`` and returns logprobs. DSPy's own text-completion path applies no chat template, so this engine
renders the Gemma 4 turn format itself and plugs into ``dspy.LM(engine=...)``: DSPy keeps owning caching,
retries, history and usage; lm15 owns the error taxonomy.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request as HttpRequest
from urllib.request import urlopen

from dspy._vendor.lm15.errors import ContextLengthError, map_http_error
from dspy.clients.call_result import CallResult
from dspy.lm15 import Message, Request, Response, TextPart, TokenLogprob, TopLogprob, Usage, response_to_events

_TURN_ROLES = {"user": "user", "assistant": "model"}
_FINISH_REASONS = {"stop": "stop", "length": "length"}


def render_gemma4_prompt(system: str | None, turns: Sequence[tuple[str, str]]) -> str:
    """Render the Gemma 4 chat template for plain text turns with thinking off.

    The server prepends BOS itself: a bare ``hi`` costs 2 prompt tokens and a literal ``<bos>`` adds one more
    [measured: probe_bos_think_embed.py], so no BOS is emitted. ``<|think|>`` (thinking on) is deliberately
    absent: worker calls are grammar-constrained JSON.
    """
    parts = [f"<|turn>system\n{system.strip()}<turn|>\n"] if system is not None else []
    for role, text in turns:
        if role not in _TURN_ROLES:
            raise ValueError(f"Unsupported chat role for the Gemma 4 completion template: {role!r}")
        parts.append(f"<|turn>{_TURN_ROLES[role]}\n{text.strip()}<turn|>\n")
    parts.append("<|turn>model\n")
    return "".join(parts)


def _text(content: Any) -> str:
    if not isinstance(content, str):
        raise TypeError("The completion engine accepts plain text message content only")
    return content


def _legacy_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    system = None
    turns = []
    for index, message in enumerate(messages):
        if index == 0 and message["role"] in ("system", "developer"):
            system = _text(message["content"])
        else:
            turns.append((message["role"], _text(message["content"])))
    return render_gemma4_prompt(system, turns)


def _logprobs(choice: Mapping[str, Any]) -> tuple[TokenLogprob, ...] | None:
    content = (choice.get("logprobs") or {}).get("content")
    if content is None:
        return None
    return tuple(
        TokenLogprob(
            token=item["token"],
            logprob=float(item["logprob"]),
            bytes=tuple(item["bytes"]) if item.get("bytes") is not None else None,
            token_id=item.get("id"),
            top=tuple(
                TopLogprob(
                    token=top["token"],
                    logprob=float(top["logprob"]),
                    bytes=tuple(top["bytes"]) if top.get("bytes") is not None else None,
                    token_id=top.get("id"),
                )
                for top in item.get("top_logprobs", ())
            ),
        )
        for item in content
    )


class GemmaCompletionsEngine:
    """``dspy.LM`` engine posting rendered Gemma 4 prompts to ``/v1/completions`` with ``grammar`` and logprobs."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 240.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, body: dict[str, Any]) -> Response:
        http = HttpRequest(
            f"{self.base_url}/completions",
            data=json.dumps({key: value for key, value in body.items() if value is not None}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http, timeout=self.timeout) as reply:
                payload = json.load(reply)
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 400 and "exceed_context_size_error" in detail:
                raise ContextLengthError(detail, provider="openai-completions", status=exc.code) from exc
            raise map_http_error(exc.code, detail, provider="openai-completions") from exc
        choice = payload["choices"][0]
        usage = payload["usage"]
        return Response(
            id=payload.get("id"),
            model=payload["model"],
            message=Message.assistant([TextPart(choice["text"])]),
            finish_reason=_FINISH_REASONS[choice["finish_reason"]],
            usage=Usage(
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            ),
            logprobs=_logprobs(choice),
            provider_data={"timings": payload["timings"]} if "timings" in payload else None,
        )

    def _body(self, model: str, prompt: str, **options: Any) -> dict[str, Any]:
        logprobs = options.pop("logprobs")
        extra = options.pop("extra_body") or {}
        return {
            "model": model.removeprefix("openai/"),
            "prompt": prompt,
            "logprobs": None if not logprobs else 1,
            **options,
            **extra,
        }

    def complete_legacy(self, lm: Any, request: Mapping[str, Any], **context: Any) -> CallResult:
        """Adapter calls arrive as ordinary kwargs, so ``extra_body['grammar']`` and ``logprobs`` stay verbatim."""
        body = self._body(
            request["model"],
            _legacy_prompt(request["messages"]),
            logprobs=request.get("logprobs"),
            extra_body=request.get("extra_body"),
            max_tokens=request.get("max_tokens"),
            temperature=request.get("temperature"),
            top_p=request.get("top_p"),
            top_k=request.get("top_k"),
            stop=request.get("stop"),
            seed=request.get("seed"),
        )
        return CallResult.native(self._post(body), logprobs=bool(request.get("logprobs")))

    def complete(self, request: Request) -> Response:
        """Typed path: the grammar travels in ``Config.extensions['grammar']``."""
        config = request.config
        system = request.system if request.system is None or isinstance(request.system, str) else None
        if request.system is not None and system is None:
            raise TypeError("The completion engine accepts a plain text system prompt only")
        turns = [(message.role, message.text or "") for message in request.messages]
        body = self._body(
            request.model,
            render_gemma4_prompt(system, turns),
            logprobs=config.logprobs is not None,
            extra_body=dict(config.extensions or {}),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop=list(config.stop) or None,
            seed=config.seed,
        )
        return self._post(body)

    def stream(self, request: Request) -> Any:
        return response_to_events(self.complete(request))

    def close(self) -> None:
        return None
