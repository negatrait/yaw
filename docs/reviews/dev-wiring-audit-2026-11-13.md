# Dev wiring audit (2026-11-13)

Baseline: `dev` at `876e524`. The checkout already contained uncommitted HMM,
grammar, resource-type and documentation work. Preserve those experiments; this
audit is not approval of their retrieval behavior or benchmark acceptance.

## Scope and observable behavior

Trace configured inference from live ingestion, native indexing, optimization,
benchmark entrypoints and watchdog consumers. Direct extraction must use the
configured extraction worker. The Gemma completion route must retain grammar,
logprobs, native usage, endpoint-isolated caching, async calls and portable prompt
checkpoint saves without credentials. Approved bounded batches replace retired
Lemonade token discovery. Optional reflection evidence must fit the same complete
request checked before transmission; oversized mandatory content and provider
errors remain hard failures.

Required gates: locked environment sync, type checks, Ruff, formatting, full
pytest, diff check, independent critic, loopback transport/checkpoint/cache and
runtime regressions, and explicitly distinguished real-provider checks. Do not
tick broader B1-B5 or graph acceptance checkpoints from these integration tests.

## Changes verified during this audit

- Grammar routes resolve builders from their owning module rather than missing
  globals. Fixed public JSON encoder access and type narrowing; removed the
  superseded duplicate scalar classifier that shadowed the optimized classifier.
- Gemma prompts are used only for the Gemma profile. Qwen retains its own chat
  template. Added custom engine state serialization without API keys, async
  transport and an endpoint/template cache namespace. Actual program save/load
  retains deployment-owned LMs and credentials.
- Live runtime extraction prefers `rlm_sub_lm`, matching native candidate indexing.
  The live create/delete regression passes with an unavailable separate QA LM.
- The user approved: "Apply the spec’s bounded-batch approach". Details, governing
  decision and the independent boundary correction are in `inference-backend.md`
  Unit 2. Removed exact counter/config/injection/fit APIs end-to-end; reflection
  uses an explicit character bound, synonym packets use configured pair counts,
  and native usage remains the token source. No surrogate token estimate.
- Updated stale model fixtures, invalid-optimization setup expectations and packet
  telemetry checks. The latter compares audit packets against actual provider
  calls rather than an obsolete tokenizer-dependent call count.
- Benchmark configuration propagation is being verified separately by the
  implementation agent; see the backend review continuation before acceptance.

## Evidence and limits

- `outputs/wiring-audit/b2-targeted-final.xml`: 66 passing focused tests.
- Completion engine regressions: 13 passed, including actual checkpoint files,
  endpoint-isolated disk cache and sync/async loopback HTTP requests.
- Live file ingestion with distinct/unavailable QA: 1 passed.
- `outputs/wiring-audit/live-inference.xml`: 2 real-provider tests passed, direct
  extraction and instruction reflection. `live-reflection-final.xml`: reflection
  passed again after the request-boundary correction. The existing Windows User
  credential was inherited only into test subprocesses; never persisted in files.
- `outputs/wiring-audit/integrated-suite.xml`: initial integrated full run,
  1028 passed, 22 failed, 3 subtests passed. It did not inherit the existing User
  credential; its two live failures are superseded by the authenticated live runs.
  Stale fixture failures were corrected and checked separately afterward. This is
  not a clean final-source full-suite claim.
- `outputs/wiring-audit/retrieval-contract.xml`: 13 passed, 9 failed. The retained
  HMM experiment changes normalization, drops meaningful prior effects, ignores
  activation/blend controls, and overrides damping below 0.78. The user was asked
  whether to preserve the established PPR path or finish the HMM contract cutover;
  no algorithmic decision was inferred. The experiment remains present.
- Six full-suite graph acceptance failures concern late-evidence/arrival-order
  canonical parity. `constraint-propagation.md` already records the unresolved
  receipt/contribution boundary. These are not fixed by inference transport work.
- Windows owned-child cleanup failed in the broad run and passed its isolated
  rerun (12.72 s). This does not establish that its timing-sensitive failure is
  resolved; production cleanup code was not changed.
- Independent available-model critic found the runtime worker selection and
  reflection request-fitting defects. Both were corrected. The final B2 boundary
  audit reported no actionable findings and independently passed four loopback
  cases. Requested `flash` model is not available in this session; no flash audit
  is claimed.
- Real 64-chunk publication is running in
  `outputs/wiring-audit/live-publication/yaw.db`, with its source manifest beside
  it and a persistent probe memo cache in `outputs/wiring-audit/coco_state`.
  Extraction reached 64/64; final graph publication remains unverified here.

No user index was wiped, no existing service was restarted, no benchmark parity
or live optimizer acceptance is claimed, and no commit/checkpoint was made while
mandatory acceptance gates remain failing.

## Resumption on DSPy 3.5 (2026-11-13)

The user approved the bounded-batch approach and completing H1, then prioritized prompt integrity and a two-stage relation judge. The previous statement that the HMM decision was pending is superseded. The 64-chunk live publication probe was interrupted; extraction reached 64/64 but publication was never accepted, and no probe process remained at the earlier stop verification. It must not be described as still running.

The current priority is the native DSPy 3.5 cutover and required checks, followed by H1. `dspy[deno,openai]>=3.5.0` is the declared dependency after explicit user approval. Unsloth now supplies two embeddinggemma-2 slots and one gemma-4-e4b generation slot. One native DSPy embedding adapter owns sync/async transport, shared two-call admission, batching, model/endpoint identity and cache behavior; splitting, overflow retry and competing transport paths are retired. Complete edge explanations remain stored and exported on RDF statement annotations as skos:note.

Current prompt/unit evidence is in prompt-context-integrity.md and inference-backend.md. New live native QA and sync/async embedding artifacts are in outputs/wiring-audit/prompt-integrity/native35. The complete current suite is running; no clean gate, process reload, retrieval quality, H1 acceptance, corpus publication or commit is inferred from focused checks.
