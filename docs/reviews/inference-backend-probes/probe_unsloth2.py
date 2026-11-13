import json
import os
import urllib.request

BASE = "http://127.0.0.1:8888"
KEY = os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"]


def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def show(title, res, limit=1400):
    status, data = res
    print(f"== {title}: HTTP {status}")
    print(json.dumps(data, indent=1)[:limit] if not isinstance(data, str) else data[:limit])


s, models = call("GET", "/v1/models")
print("model ids:", [(m["id"], m.get("context_length"), m.get("loaded")) for m in models["data"]])
msgs = [{"role": "user", "content": "Say hi in one word."}]
show(
    "chat (usage, logprobs)",
    call(
        "POST",
        "/v1/chat/completions",
        {"model": "gemma-4-e4b", "messages": msgs, "max_tokens": 8, "logprobs": True, "top_logprobs": 2},
    ),
    1800,
)
show(
    "chat + gbnf grammar",
    call(
        "POST",
        "/v1/chat/completions",
        {"model": "gemma-4-e4b", "messages": msgs, "max_tokens": 8, "grammar": 'root ::= "yes" | "no"'},
    ),
    900,
)
show(
    "chat + extra top_k",
    call("POST", "/v1/chat/completions", {"model": "gemma-4-e4b", "messages": msgs, "max_tokens": 8, "top_k": 64}),
    500,
)
show("count_tokens", call("POST", "/v1/chat/count_tokens", {"model": "gemma-4-e4b", "messages": msgs}))
show(
    "embeddings",
    call("POST", "/v1/embeddings", {"model": "embeddinggemma-2", "input": ["probe"], "encoding_format": "float"}),
    300,
)
