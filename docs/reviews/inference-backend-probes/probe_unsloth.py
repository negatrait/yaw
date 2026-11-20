import json
import os
import urllib.error
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
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]


def show(title, res):
    status, data = res
    print(f"== {title}: HTTP {status}")
    print(json.dumps(data, indent=1)[:1500] if not isinstance(data, str) else data)


show("GET /v1/models", call("GET", "/v1/models"))
show("GET /v1/status", call("GET", "/v1/status"))
