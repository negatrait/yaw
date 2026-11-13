# Review Record: Provider-Agnostic Inference Backend (lm15)

Governing spec: [dspy-improvements.md §6](../dspy-improvements.md#6-inference-backend-contract-dspy-lm15-zero-provider-plumbing). Branch `dev`, starting commit `7fbec2b`.

## Scope and definition of done

Replace Lemonade-specific inference plumbing with DSPy `lm15` (`dspy.LM("openai/<alias>")` + `dspy.Embedder`) so Unsloth, vLLM and the OpenAI API are swapped by configuration only (spec goals G1–G4).

## Checkpoints

- [ ] B0 Open items resolved and recorded as Decisions
- [ ] B1 Client path
- [ ] B2 Retire `ServingTokenCounter` and registry evidence
- [ ] B3 Structured output and logprobs
- [ ] B4 Watchdog telemetry, docs and test sweep
- [ ] B5 Live acceptance on Unsloth

## Unit 0: spec and capability probes (2026-11-13)

- **What changed**: added spec §6 to `docs/dspy-improvements.md`; stored the probe scripts in [inference-backend-probes/](inference-backend-probes/). No code changed.
- **Gates run**: probes against Unsloth Studio on `127.0.0.1:8888` with a bearer key from `UNSLOTH_STUDIO_AUTH_TOKEN` (never stored in files). Results are tagged in spec §6.2.
- **Discoveries**: Unsloth rejects `logprobs` (HTTP 400), accepts but ignores `grammar`, enforces `response_format` json_schema; the served Gemma model emits `reasoning_content` that consumes `max_tokens`; `dspy.LM("openai/<alias>")` already runs through lm15 with usable `Usage`.
- **Does not establish**: behavior on vLLM or OpenAI, ContextLengthError normalization, embedding dimension, `list_models`, any quality claim.
- **Process notes**: the `research` and `critic` subagents hit `RESOURCE_EXHAUSTED` (429) during this unit, so the link walk and spec review were done by the parent agent. The spec has had no independent critic review.
- **Security note**: the user pasted an Unsloth API key into the chat; rotate it in Unsloth Settings → API. It was passed to probes through an environment variable in task command lines, which are retained in the conversation's task logs.
- **Decisions**: [§6 Decisions](../dspy-improvements.md#decisions).
