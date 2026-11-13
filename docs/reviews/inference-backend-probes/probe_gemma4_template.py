import json
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
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


question = "Say hi in one word."
variants = {
    "gemma4 markers, no bos": f"<|turn>user\n{question}<turn|>\n<|turn>model\n",
    "gemma4 markers, literal bos": f"<bos><|turn>user\n{question}<turn|>\n<|turn>model\n",
    "gemma4 + system": f"<|turn>system\nBe terse.<turn|>\n<|turn>user\n{question}<turn|>\n<|turn>model\n",
    "gemma3 markers (control)": f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n",
}
for name, prompt in variants.items():
    out = call(
        "/v1/completions",
        {"model": "gemma-4-e4b", "prompt": prompt, "max_tokens": 12, "temperature": 0, "logprobs": 1},
    )
    usage = out["usage"]
    print(f"{name}: prompt_tokens={usage['prompt_tokens']} text={out['choices'][0]['text']!r}")

chat_count = urllib.request.Request(
    BASE + "/v1/chat/count_tokens",
    data=json.dumps({"model": "gemma-4-e4b", "messages": [{"role": "user", "content": question}]}).encode(),
    method="POST",
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
)
print("chat count_tokens:", json.load(urllib.request.urlopen(chat_count, timeout=60)))
