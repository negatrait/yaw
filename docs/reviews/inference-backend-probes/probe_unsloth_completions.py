import json
import os
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8888"
KEY = os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"]


def call(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


def show(title, res, limit=1600):
    status, data = res
    print(f"== {title}: HTTP {status}")
    print(json.dumps(data, indent=1)[:limit] if not isinstance(data, str) else data)


prompt = "<start_of_turn>user\nSay hi in one word.<end_of_turn>\n<start_of_turn>model\n"
show(
    "completions logprobs=3",
    call(
        "/v1/completions",
        {"model": "gemma-4-e4b", "prompt": prompt, "max_tokens": 3, "temperature": 0, "logprobs": 3},
    ),
)
show(
    "completions + grammar",
    call(
        "/v1/completions",
        {
            "model": "gemma-4-e4b",
            "prompt": prompt,
            "max_tokens": 6,
            "grammar": 'root ::= "yes" | "no"',
            "logprobs": 1,
        },
    ),
    700,
)
show(
    "completions + response_format",
    call(
        "/v1/completions",
        {
            "model": "gemma-4-e4b",
            "prompt": "Reply with a JSON cat. ",
            "max_tokens": 40,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "c",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"zzname": {"type": "string"}},
                        "required": ["zzname"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    ),
    700,
)
