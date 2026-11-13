import json
import math
import os
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
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"HTTP": e.code, "body": e.read().decode()[:300]}


# BOS handling: token counts of a bare word with and without a literal <bos>.
for prompt in ("hi", "<bos>hi", "<bos><bos>hi"):
    out = call("/v1/completions", {"model": "gemma-4-e4b", "prompt": prompt, "max_tokens": 1})
    print(f"prompt {prompt!r}: prompt_tokens={out['usage']['prompt_tokens']}")

# Thinking contract: <|think|> as the first system-turn token.
q = "What is 17*23? Answer with just the number."
for name, prompt in {
    "no think": f"<|turn>user\n{q}<turn|>\n<|turn>model\n",
    "think": f"<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{q}<turn|>\n<|turn>model\n",
}.items():
    out = call("/v1/completions", {"model": "gemma-4-e4b", "prompt": prompt, "max_tokens": 200, "temperature": 0})
    print(f"{name}: tokens={out['usage']['completion_tokens']} text={out['choices'][0]['text'][:160]!r}")

# Embeddings.
out = call(
    "/v1/embeddings",
    {"model": "embeddinggemma-2", "input": ["probe", "another probe text"], "encoding_format": "float"},
)
vecs = [d["embedding"] for d in out["data"]]
print("embedding dims:", [len(v) for v in vecs], "norms:", [round(math.sqrt(sum(x * x for x in v)), 4) for v in vecs])
print("usage:", out["usage"], "model:", out["model"])
doc_prefix = "title: none | text: "
for label, text in {
    "query prefix": "task: question answering | query: hello",
    "doc prefix": doc_prefix + "hello",
}.items():
    o = call("/v1/embeddings", {"model": "embeddinggemma-2", "input": [text], "encoding_format": "float"})
    print(label, "tokens:", o["usage"]["prompt_tokens"])
long_text = "word " * 2500
o = call("/v1/embeddings", {"model": "embeddinggemma-2", "input": [long_text], "encoding_format": "float"})
print("2500-word input:", o.get("usage") or o)
