import dataclasses
import os

import dspy
from dspy.lm15 import Config, Message, Request, Response

KEY = os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"]
lm = dspy.LM("openai/gemma-4-e4b", api_base="http://127.0.0.1:8888/v1", api_key=KEY, cache=False, max_tokens=1500)
print("lm class:", type(lm).__mro__[:3])
print("engine:", type(getattr(lm, "engine", None)))
print("Config fields:", [f.name for f in dataclasses.fields(Config)])
print("Request fields:", [f.name for f in dataclasses.fields(Request)])
try:
    resp = lm(Request(model=lm.model, messages=(Message.user("Say hi in one word."),), config=Config(max_tokens=1500)))
    assert isinstance(resp, Response)
    print("response type:", type(resp).__name__)
    print("text:", repr(resp.text), "finish:", resp.finish_reason)
    print("usage:", resp.usage)
    print("response fields:", [f.name for f in dataclasses.fields(type(resp))])
except Exception as exc:  # noqa: BLE001
    print("lm15 call error:", type(exc).__name__, str(exc)[:500])

# logprobs through lm15 config?
for name in ("logprobs", "top_logprobs", "extensions", "provider", "response_format", "top_k"):
    print(f"Config.{name}:", name in {f.name for f in dataclasses.fields(Config)})
print("Request.extensions?", "extensions" in {f.name for f in dataclasses.fields(Request)})


# structured output via dspy adapter path
class Cat(dspy.Signature):
    """Describe a cat."""

    prompt: str = dspy.InputField()
    zzname: str = dspy.OutputField()
    zzlives: int = dspy.OutputField()


with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
    pred = dspy.Predict(Cat)(prompt="a cat")
    print("JSONAdapter pred:", pred)
    h = lm.history[-1]
    print("history keys:", list(h.keys()) if isinstance(h, dict) else type(h))
    print("history usage:", h.get("usage") if isinstance(h, dict) else None)
    print("request kw:", {k: v for k, v in (h.get("kwargs") or {}).items()} if isinstance(h, dict) else None)
