"""Minimal Ollama HTTP client and canonical spaCy-LLM model registry connector."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

# spaCy-LLM registry protocol requires typing.Iterable for task validation
from typing import Any, Iterable  # noqa: UP035

from spacy_llm.registry import registry

from yaw.nlp.templates import PROMPT_PREFIXES
from yaw.nlp.utils import normalize_token

from .. import config


def call_ollama_json(
    endpoint: str,
    payload: dict[str, Any] | None = None,
    host: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Execute JSON request against Ollama endpoint."""
    clean_endpoint = "/" + normalize_token(endpoint).lstrip("/")
    url = f"{(host or config.OLLAMA_HOST).rstrip('/')}{clean_endpoint}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(
            req, timeout=timeout_s or config.CHAT_TIMEOUT_S
        ) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code} error: {err_body}") from e


def get_ollama_host() -> str:
    """Return configured Ollama host URL."""
    return str(config.OLLAMA_HOST)


def get_embedding_model() -> str:
    """Return configured embedding model name."""
    return str(config.EMBEDDING_MODEL)


def get_relevance_threshold() -> float:
    """Return configured relevance threshold."""
    return float(config.RELEVANCE_THRESHOLD)


def is_ollama_online(host: str | None = None, timeout_s: float = 0.5) -> bool:
    """Return True if Ollama service is responsive."""
    try:
        call_ollama_json("/api/version", host=host, timeout_s=timeout_s)
        return True
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        ConnectionError,
        OSError,
        json.JSONDecodeError,
        RuntimeError,
    ):
        return False


class _OllamaModel:
    """Type-consistent Ollama model connector adhering to ModelWithContextLength."""

    def __init__(
        self,
        name: str | None = None,
        endpoint: str | None = None,
        host: str | None = None,
        timeout_s: float | None = None,
        temperature: float = 0.0,
        context_length: int = 8192,
        max_tries: int = 4,
        interval: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.host = host
        self.timeout_s = timeout_s
        self.temperature = float(temperature)
        self._context_length = int(context_length)
        self.max_tries = max(1, int(max_tries))
        self.interval = float(interval)
        self.backoff_factor = float(backoff_factor)
        self.supported_tasks = list(PROMPT_PREFIXES)

    @property
    def context_length(self) -> int:
        return self._context_length

    def _resolve_url(self) -> str:
        target_host = self.host or config.OLLAMA_HOST
        if self.endpoint:
            if self.endpoint.startswith("http://") or self.endpoint.startswith(
                "https://"
            ):
                return self.endpoint
            return f"{target_host.rstrip('/')}/{self.endpoint.lstrip('/')}"
        return f"{target_host.rstrip('/')}/api/chat"

    def _resolve_timeout(self) -> float:
        return (
            self.timeout_s
            if self.timeout_s is not None
            else float(config.CHAT_TIMEOUT_S)
        )

    def _resolve_model(self) -> str:
        return self.name or config.CHAT_MODEL

    def _send_payload_with_retry(self, payload: dict[str, Any]) -> str:
        url = self._resolve_url()
        timeout = self._resolve_timeout()
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        last_error: Exception | None = None
        for attempt in range(self.max_tries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    return resp_data.get("message", {}).get("content", "")
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                # Fast fail on non-retryable 4xx client errors (429 is retryable)
                if 400 <= e.code < 500 and e.code != 429:
                    raise RuntimeError(
                        f"Ollama client HTTP {e.code} error: {err_body}"
                    ) from e
                last_error = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_error = e

            if attempt < self.max_tries - 1:
                delay = self.interval * (self.backoff_factor**attempt)
                time.sleep(delay)

        raise ConnectionError(
            f"Ollama request to {url} failed after {self.max_tries} attempts: {last_error}"
        ) from last_error

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        model_name = self._resolve_model()
        all_responses: list[list[str]] = []
        for doc_prompts in prompts:
            doc_res: list[str] = []
            for prompt_str in doc_prompts:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": str(prompt_str)}],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self._context_length,
                    },
                }
                content = self._send_payload_with_retry(payload)
                doc_res.append(content)
            all_responses.append(doc_res)
        return all_responses


@registry.llm_models("yaw.Ollama.v1")
def make_ollama_model(
    name: str | None = None,
    endpoint: str | None = None,
    host: str | None = None,
    timeout_s: float | None = None,
    temperature: float = 0.0,
    context_length: int = 8192,
    max_tries: int = 4,
    interval: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Canonical REST model connector for local Ollama instances adhering to spaCy-LLM registry protocol."""
    return _OllamaModel(
        name=name,
        endpoint=endpoint,
        host=host,
        timeout_s=timeout_s,
        temperature=temperature,
        context_length=context_length,
        max_tries=max_tries,
        interval=interval,
        backoff_factor=backoff_factor,
    )
