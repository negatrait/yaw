import os

import dspy

from yaw.daemon.graph.inference.completions import GemmaCompletionsEngine

KEY = os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"]
engine = GemmaCompletionsEngine("http://127.0.0.1:8888/v1", KEY)
lm = dspy.LM("openai/gemma-4-e4b", engine=engine, cache=False, temperature=1.0, max_tokens=120)


class Cat(dspy.Signature):
    """Describe a cat."""

    prompt: str = dspy.InputField()
    zzname: str = dspy.OutputField()
    zzlives: int = dspy.OutputField()


GRAMMAR = 'root ::= "{\\"zzname\\": " name ", \\"zzlives\\": " [0-9]+ "}"\nname ::= "\\"" [A-Za-z]{1,12} "\\""'
with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
    pred = dspy.Predict(Cat)(prompt="a cat", config={"logprobs": True, "extra_body": {"grammar": GRAMMAR}})
print("prediction:", pred)
logprobs = pred.get("logprobs")
print("logprobs type:", type(logprobs).__name__, "tokens:", len(logprobs["content"]) if logprobs else None)
if logprobs:
    first = logprobs["content"][0]
    print("first token:", {k: first[k] for k in ("token", "logprob", "bytes")})
entry = lm.history[-1]
print("usage:", entry["usage"])
print("prompt head:", repr(entry["messages"][-1]["content"][:80]))
