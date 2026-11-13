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

## Unit 1: Route B worker engine and embedder check (2026-11-13)

- **What changed**: `inference/completions.py:GemmaCompletionsEngine`, `render_gemma4_prompt`; `inference/chat.py:create_dspy_lm` passes the engine to worker roles (`recipe == "llamacpp"` and role not `rlm`/`reflection`); `core/config.py` `api_key` now defaults to `UNSLOTH_STUDIO_AUTH_TOKEN` (empty when unset, so the first call fails with the provider's auth error); `yaw.toml` no longer carries `api_key`; tests in `tests/daemon/graph/inference/test_completions_engine.py` (8 loopback tests) and an explicit key in `test_inference_budgets.py`.
- **Gates run**: engine tests 8 passed; `ruff` clean on touched files; `tests/core`, `test_runtime.py`, `test_runtime_contracts.py`, `test_embedding.py`, `test_inference_budgets.py` pass except the items under Discoveries.
- **Live evidence** (Unsloth Studio, `gemma-4-e4b`, 2026-11-13): `YawProgram.extract_openie` on 3 dataset chunks returned 22/9/15 entities and 18/5/10 triples in 22.2 s total with `GemmaCompletionsEngine` on both worker LMs [measured: `docs/reviews/inference-backend-probes/probe_program_live.py`, uncommitted adapter/grammar WIP replaced by HEAD during the run]. Grammar enforcement and logprob shape verified in `probe_engine_live.py`.
- **Embedder**: `embeddinggemma-2` returns 768-dim unit-norm vectors; 3 of 3 queries ranked their matching document first through `DSPyEmbeddingAdapter` with the `distinct-titled` prefixes; a 3000-word document split into 6 pieces and encoded (3905 tokens) [measured: `probe_embedder_adapter.py`]. Profile max input stays 2048 tokens although the server reports 8192.
- **Discoveries**: (1) `<bos>` must not be emitted: the server adds BOS (`hi` = 2 tokens, `<bos>hi` = 3) [measured: `probe_bos_think_embed.py`]. (2) `extra_body` and `extensions` cannot ride DSPy's typed ingress for custom engines; the grammar travels through the legacy `complete_legacy` shortcut, which DSPy deprecates for removal in 3.5 (the typed `complete()` path exists and takes the grammar from `Config.extensions`). (3) llama.cpp reports raw model logprobs: grammar-forced tokens show very low values (`"` at -19.8) while value tokens look normal, matching what the confidence code already reads. (4) The `<|think|>` thinking token in the system turn did not change a trivial arithmetic answer (4 completion tokens either way); thinking-on behavior is unverified and unused by worker roles.
- **Pre-existing or unrelated failures**: `test_chat.py` (2) and `test_architecture.py::test_zero_cross_module_private_attributes` (`ppr.py:504` accesses `_hmm`) come from the user's uncommitted edits to `models.py`, `adapter.py`, `grammars.py`, `ppr.py`, `types.py`; `adapter.py` WIP also raises `KeyError: 'build_dspy_ner_grammar'` in `resolve_grammar`.
- **Does not establish**: the 150 vs 60 tok/s claim (not measured here), the 64-chunk indexing gate (G4), behavior with caching on, async calls, or the orchestrator roles on the chat route.
- **Decisions**: [§6 Decisions](../dspy-improvements.md#decisions).
