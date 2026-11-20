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


msgs = [{"role": "user", "content": "Reply with any JSON object describing a cat."}]
schema = {
    "type": "object",
    "properties": {"zzname": {"type": "string"}, "zzlives": {"type": "integer"}},
    "required": ["zzname", "zzlives"],
    "additionalProperties": False,
}
variants = {
    "response_format json_schema": {
        "response_format": {"type": "json_schema", "json_schema": {"name": "cat", "schema": schema, "strict": True}}
    },
    "response_format json_object": {"response_format": {"type": "json_object"}},
    "json_schema field": {"json_schema": schema},
    "grammar field (long)": {"grammar": 'root ::= "{\\"zzname\\": \\"" [a-z]+ "\\", \\"zzlives\\": " [0-9]+ "}"'},
    "extra_body.grammar nested": {"extra_body": {"grammar": 'root ::= "yes"'}},
    "guided_json": {"guided_json": schema},
}
for name, extra in variants.items():
    status, data = call("/v1/chat/completions", {"model": "gemma-4-e4b", "messages": msgs, "max_tokens": 80, **extra})
    out = (data["choices"][0]["message"], data["choices"][0]["finish_reason"]) if isinstance(data, dict) else data
    print(f"== {name}: HTTP {status}: {out!r}")
