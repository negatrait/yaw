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

## Unit 2: bounded requests replace tokenizer discovery (2026-11-13)

User approved the bounded-batch approach during the dev wiring audit. Observable
behavior: synonym admission uses exactly the configured pair-count limit;
reflection requests have an explicit character cap; the provider owns context
overflow and post-call usage. No health, props, apply-template, or tokenize
requests are made. No approximate token count or retry without a required feature.

Definition of done: remove the exact-counter API and all callers/configuration,
verify loopback semantic judgment and reflection without discovery endpoints,
verify oversized provider errors propagate, and run the mandatory quality gates
and independent critic. This unit does not establish 64-chunk real-model
publication, full-corpus parity, or portable backend support. Results pending.

### Unit 2 implementation evidence

- Removed `ServingTokenCounter`, `SynonymRequestBudget`, the synonym fit callback,
  retired tokenizer URL/token-limit configuration, and counter injection APIs
  across the program, planner, optimizer, proposer, engine and tests. Reflection
  uses `reflection_input_chars=65536`; semantic packets honor the configured
  pair count and retain complete source evidence within their character memory
  bound. No input token estimate is synthesized. Judgment audit records retain
  native post-call `usage`; provider errors are not retried with reduced features.
- Updated loopback fixtures to expose chat completions only: discovery routes
  return 404. Covered reflection sync/async/copy, mandatory content, demos and
  template options, whole optional-evidence removal, configured pair packet
  partitioning, native usage, rollback/cache replay and provider overflow. DSPy
  3.4.1 exposes overflow as `ContextWindowExceededError` with lm15
  `ContextLengthError` as its original cause; the test verifies both and one
  request, rather than incorrectly requiring the unwrapped inner exception.
- `outputs/wiring-audit/b2-targeted.xml`: **64 passed**, 35 upstream warnings,
  60.30 s. Full-repository `ty check`, `ruff check`, `ruff format --check`
  (201 files), and `git diff --check` passed at this checkpoint. Removal search
  for retired counters/fit/injection/config names over `yaw`, `tests`, `yaw.toml`
  returned no matches. Parent wiring audit owns final combined suite and critic.
- Live test entrypoints now use configured aliases and authenticated `/models`;
  missing credentials fail explicitly. This environment lacks
  `UNSLOTH_STUDIO_AUTH_TOKEN`, so the live model and 64-chunk publication gate
  are not established. No daemon was restarted and no benchmark parity claimed.
- B2's broader registry-evidence removal and config-only backend portability
  remain outside this bounded-request unit; no B2/B5 checkbox is marked complete.

### Unit 2 critic follow-up: one complete request boundary

The independent critic reproduced a near-cap mismatch: optional-evidence fitting
counted formatted messages alone, while final dispatch also counted default LM
options and adapter-injected grammar. The final guard correctly aborted, but the
optional packet had not been reduced first. `check_request` now always merges
LM defaults, and fitting includes the predictor's per-call configuration and
resolved adapter grammar. The final dispatch remains the authoritative guard.

A real loopback regression captures the dispatched messages/options, pins a cap
50 characters below that complete request, and verifies whole optional-evidence
removal followed by exactly one successful, bounded request. Both standard and
GBNF cases pass. `outputs/wiring-audit/b2-targeted-final.xml` records **66 passed**
in 66.94 s after the correction. Type, Ruff, format (201 files), and diff checks
passed again. No live inference or full-suite claim follows from this targeted
rerun; the parent wiring review records the combined and live evidence separately.

## Unit 3: benchmark inference configuration wiring (2026-11-13)

The dev wiring audit found `benchmark.py` still called Lemonade registry/health
preflight and constructed clients with module constants/default credentials.
Observable replacement: the benchmark loads `yaw.toml` (or explicit `--config`),
resolves model overrides against that configuration, and passes its endpoint,
authentication, context/output budgets and embedding batch settings through all
inference roles, including reflection and embedding probes. The active path
makes no registry/health/tokenizer calls. Reports distinguish configured identity
from remotely verified checkpoint/device claims. Generated state stays under
ignored outputs and the persistent configured CocoIndex cache.

Definition of done: actual loopback embedding and full benchmark requests require
the configured bearer key and aliases; real SQLite publication/retrieval completes;
source hydration respects configured chunk settings; required checks and critic
run. Loopback evidence validates orchestration only, not model quality or parity.
No checkpoint is ticked and no commit is made in this unit.

## Unit 4: deployment-aware embedding memo identity (2026-11-13)

The critic identified a stale-vector path: CocoIndex embedding requests and
`chunk_inference.embedding_identity` used the declared profile alone, so an
unchanged source and alias could reuse vectors from a different server. The
replacement has one public `DSPyEmbeddingAdapter.index_identity`: document-side
profile contract, actual configured provider/endpoint and vector-affecting
splitting budget. Credentials are excluded. This identifies a configured
contract, not a verified remote checkpoint. Query-only prefixes remain excluded.

The exact identity must flow into every `EmbeddingRequest` memo key, endpoint
validation, chunk inference row, graph settings, resume hydration, staging and
benchmark report. Synthetic providers must supply their own explicit identity;
production may not silently substitute a profile-only identity.

Definition of done: one real CocoIndex runtime and unchanged source using the
same alias against two authenticated loopback servers must call the second
server and publish its different vectors; repeating the same deployment must
remain a memo hit. Splitting-budget changes must invalidate the identity. All
callers/telemetry and required checks plus independent critic must be covered.
No live process is restarted and no remote checkpoint or quality is inferred.

## E1: native DSPy embedding cutover (2026-11-13)

Observable contract: one native DSPy Embedder transport for synchronous and asynchronous embedding; retain task prefixes, title formatting, deployment-aware memo identity and bounded request batches. Whole inputs are sent without splitting or overflow retries. Invalid or non-finite vectors and provider errors fail immediately. Async indexing admits two concurrent embedding requests.

Acceptance evidence: authenticated HTTP loopback sync and async requests, two-way concurrency, task prefixes, endpoint-aware cache identity, failure and vector validation; focused tests and applicable quality checks. Loopback does not establish live model quality, retrieval parity or process state. No checkpoint or commit is authorized in this unit.

E1 work in progress: adapter now delegates sync and async calls to native DSPy, sends each input whole, validates count/dimension/finiteness/nonzero before normalization, and propagates provider failures. Callable models require explicit index_identity; unsplit profile version invalidates previous split vectors. Indexing endpoint admission changed to two. Focused embedding and HTTP loopback tests passed 14/14; profile tests 2/2; targeted type check passed. The loopback exercised sequential sync/async calls, not concurrent admission or real model inference. Query and indexing admission are not yet shared; full suite, required global gates and independent audit remain outstanding. No checkpoint is ticked.

E1 follow-up: one adapter-owned bounded semaphore admits at most two native physical embedding calls across sync and async callers; endpoint duplicate admission removed. Cancellation of a waiting async task returns its eventual permit. A threaded HTTP loopback observed four mixed sync/async requests with peak concurrency exactly two; another verifies native cache reuse and separate endpoint/formatted-input identities. The adapter sends whole inputs and no longer retains piece mapping or pooling scaffolding. Focused embedding/profile/cache tests passed 20/20; focused ruff and ty passed. This is loopback orchestration evidence only, not live inference, full-suite acceptance or retrieval quality. The prior statement that query and indexing admission were separate is superseded. No checkpoint ticked.

E1 parent follow-up: engine embedding calls now retain whole batches and delegate directly to the single native adapter, removing the uncached per-text closure and its unused namespace argument/shared instruction state. Removed retired encode/forward aliases and ignored call kwargs. DSPy Embedder exposes vectors without provider usage, so embedding telemetry has tokens_used=null and usage={}, with unknown aggregate embedding token cost rather than a character/4 estimate presented as measured usage. Native model/endpoint cache identity remains required. The optimizer's measured-token aggregation excludes unavailable embedding usage; complete embedding-cost comparison is not established.

Live native 3.5 check: `outputs/wiring-audit/prompt-integrity/native35/result.json` records successful Gemma constrained QA (answer Ada, complete reasoning), one synchronous and two concurrent asynchronous embeddinggemma-2 calls, all (1,768), finite and L2-normalized. Exact QA request/prompt/native response are stored beside it without authentication headers. QA grammar now matches DSPy's wire-level plan field, restored by DSPy to application reasoning; native transport regression passed with the binding suite (41 tests). These do not establish retrieval quality, running optimizer reload or full-suite acceptance.


E1 audit follow-up: an injected hosted DSPy embedder could previously identify itself by the declared profile even when its actual model differed, allowing stale vectors to be reused. Construction now rejects that mismatch before provider access; callable fixtures retain explicit stable identities. The authenticated embedding HTTP suite passed 6/6, including mismatch rejection. This checks configuration identity, not remote model weights or live retrieval quality.


E1 independent audit follow-up: normalization now accumulates in float64 and returns normalized float32 vectors, preventing large finite float32 values from overflowing to zero. Both synchronous and asynchronous requests cap physical batches at 64 even when configured or requested larger. Real callable large-vector and authenticated loopback 130-input tests passed; the focused embedding suites passed 20 tests, with scoped type and Ruff checks passing. Repository-wide diff checking remains affected by line-ending whitespace elsewhere. No live-model or retrieval-quality claim follows.

## C1: active serving capacity and deployment identity

At runtime composition, read authenticated OpenAI-compatible models.list once, require each configured alias to resolve to a loaded model with a valid active context_length, and bind model-reported alias/checkpoint plus quantization to embedding memo identity. Do not use maximum/native advertised capacity as active capacity. The endpoint active chat limit is authoritative; an explicit output-token budget must fit within it. Service construction remains offline; composition fails visibly if metadata cannot be resolved. Gates: authenticated real HTTP loopback for active limits, alias replacement, invalid/missing/unloaded models, explicit cap and propagation into configured program and embedding identity; static checks. Model metadata describes reported deployment, not cryptographic weight attestation.

C1 decision: runtime uses the minimum active context_length of the selected extraction and QA models; the offline config context is an initialization snapshot, not a cap, and is replaced when the endpoint is bound. Explicit max_output_tokens remains an independently validated budget. The embedding deployment identity includes the server-reported active checkpoint/quantization and active embedding context. No maximum/native advertised context substitutes for active capacity.

C1 implementation evidence: runtime queries the authenticated native models.list endpoint at composition, validates selected aliases are loaded and have active contexts and checkpoint identities, uses the smaller selected chat context, and binds reported embedding checkpoint/quant and active context to its profile identity. Native DSPy embedding requests include the profile identity as an SDK extra header, so its cache distinguishes changed alias targets without changing the model name. Model IDs containing slashes use the openai provider prefix. Seven HTTP loopback cases passed, including same alias/endpoint/input returning a new vector after checkpoint replacement and an identical-deployment cache hit; targeted type/lint checks passed. The server metadata is a reported identity, not weight attestation. Older test servers lacking the mandatory metadata endpoint require migration; default-derived output budget rescaling for smaller active contexts remains to be resolved. No full-suite/live-runtime acceptance is established.

C1 budget follow-up: raw configuration retains automatic output and RLM budgets as None through benchmark and optimization clones. Runtime binds the active context before resolving these budgets; explicit output values that exceed it fail. Loopback verification included an 8192-token active context yielding a 512-token automatic output budget and preserving automatic intent on clones. Combined budget, model-resolution, runtime and telemetry selection passed 27 tests, with targeted type and lint checks passing. The 27-test runtime suite separately had 25 passes and two failures: the benchmark optional-context assertion was fixed and retested successfully; the crash/resume test recovered and retrieved the relevant passage but its two-hit expectation remains an H1 question. These checks do not establish live service acceptance.
