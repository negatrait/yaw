import inspect
import os

import dspy
from dspy.lm15 import Config, Message, Request, Response

KEY = os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"]
print("LM signature:", inspect.signature(dspy.LM.__init__))

# 1. text-completion model_type through dspy.LM
for kw in ({"model_type": "text"},):
    try:
        lm = dspy.LM(
            "openai/gemma-4-e4b", api_base="http://127.0.0.1:8888/v1", api_key=KEY, cache=False, model_type="text"
        )
        out = lm(
            "<start_of_turn>user\nSay hi in one word.<end_of_turn>\n<start_of_turn>model\n", max_tokens=3, logprobs=3
        )
        print(kw, "->", out)
        h = lm.history[-1]
        print("history keys:", list(h.keys()))
        print("history response type:", type(h.get("response")).__name__)
        resp = h.get("response")
        print("response repr head:", repr(resp)[:600])
    except Exception as exc:  # noqa: BLE001
        print(kw, "ERROR", type(exc).__name__, str(exc)[:400])

# 2. lm15 chat path with Config.logprobs (expected 400 on Unsloth chat)
lm = dspy.LM("openai/gemma-4-e4b", api_base="http://127.0.0.1:8888/v1", api_key=KEY, cache=False)
try:
    resp = lm(Request(model=lm.model, messages=(Message.user("hi"),), config=Config(max_tokens=200, logprobs=True)))
    assert isinstance(resp, Response)
    print("lm15 chat logprobs ->", type(resp.logprobs), (resp.logprobs or [])[:1])
except Exception as exc:  # noqa: BLE001
    print("lm15 chat logprobs ERROR", type(exc).__name__, str(exc)[:300])
