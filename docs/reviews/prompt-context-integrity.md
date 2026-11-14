# P1: prompt context integrity and complete model output

2026-11-13, dev at 876e524 plus the current user optimizations. The user interrupted
the broader wiring audit to prioritize abrupt context cuts, repeated evidence and
formatting in inspectable Unsloth requests, and explicitly requested Terra.

## Observable contract and plan

- Preserve complete selected source spans, definitions and facts in the semantic
  evidence packet. Store repeated entities/sources once with lossless references.
  Whole-pair batching may reject an oversized packet explicitly; it must never
  silently slice mandatory text, substitute empty stale evidence or invent types.
- Remove the hard-coded 360-character output-reasoning ceiling. Reasoning should
  be concise and complete under the model output-token budget. A length-finished
  generation is a visible failure, never accepted/cached/memoized partial work.
- Remove generic pre-send passage/context string slicing. Providers enforce their
  actual token window; a character heuristic is not proof of available context.
- Keep native DSPy signatures/formatting and the measured Gemma turn template.
  Clarify hierarchy direction and accepted G/I/P relations without changing the
  governing ontology: causation, mitigation and protocol use alone are not
  taxonomic/instance/architectural-part evidence. Inspect the actual rendered
  prompts, including active demos, rather than infer formatting from signatures.
- Preserve current user changes to caching, hydration, batched role-fact lookup,
  math and publication. The previous 64-chunk probe was interrupted; it is not
  current-source publication evidence. Broader wiring/HMM work is paused.

## Root causes reproduced in source

Terra independently identified the 360 grammar bound, 900-character/14-line/2-source
input cuts, capped facts/labels, duplicated inline entity/source payloads, and the
fabricated skos:Concept type on all resources. The generic direct-call slicer is a
separate path and does not explain candidate_pairs truncation. The current backend
review's previous complete-evidence assertion is superseded by these findings.

## Definition of done

Capture exact formatted requests and grammar against a real HTTP loopback; prove
long Unicode source tails survive, shared evidence appears once, subsets retain
all referenced evidence, and mandatory content is never sliced. Verify output
reasoning beyond 360 characters and explicit generation-length failures. Inspect
and save sanitized real Unsloth request/response evidence, including token usage
and finish reason, and run a bounded hierarchy case set. Required type/lint/format,
relevant tests, diff checks and an independent Terra audit follow. No acceptance
claim for HMM, graph batch parity, full corpus, or a running optimizer follows.

### P1 implementation evidence

- `storage/synonyms.py`: v3 packets reference shared entity and full hydrated source records, retain 1-indexed provenance, omit unsupported resource type, fail on stale/missing sources, and retain all source chunks and inspected labels/facts. Whole-pair batching retains its explicit character-bound failure; the batched shared-role query is unchanged.
- `core/types.py`: subset retains only transitively referenced entities and sources; v3 cache identity invalidates v2 request identities. Packet consumers and fixtures were migrated.
- `uv run --locked pytest --no-testmon tests/daemon/graph/inference/test_synonym_packets.py::test_lossless_shared_sources_and_transitive_subset tests/daemon/graph/inference/test_synonym_packets.py::test_all_source_chunks_and_oversize_failure tests/daemon/graph/storage/test_storage.py tests/daemon/graph/storage/test_synonym_constraints.py::test_union_remaps_separation_and_omits_only_collapsed_pairs -q`: 20 passed. Covers Unicode, long multiline text, shared source deduplication, subset references, stale source and oversized evidence.
- Broader packet/storage run: 28 passed, 10 failed because installed DSPy rejects `messages=None` in `YawDSPyLM.__call__` before dispatch. It does not establish real inference, live model quality, full-suite acceptance, or benchmark parity.

### R1: DSPy 3.5 reflection request migration

Observable behavior: reflection fitting and dispatch check the same native Request, including LM defaults, predictor settings and grammar. Complete mandatory evidence must fit; optional examples are removed whole. Original provider errors and native usage propagate. Gates: focused reflection and SIMBA tests, type/lint checks and real loopback evidence; these do not establish live inference or benchmark acceptance.


## N1: Native DSPy token log probabilities

Observable behavior: confidence measurement consumes native `TokenLogprob` sequences with byte-accurate spans and finite log probabilities; absent or invalid measurements fail explicitly without fabricated votes. Focused confidence tests and native request regressions are the quality gates. This unit does not establish live-model quality.

## P2: One relational judgment

Observable contract: strict identity is judged once; each distinct pair then receives one typed relation decision with complete concise evidence and A/B orientation. A direct fact skips identity; W1/W2 fact evidence establishes relatedness when no hierarchy is selected. BroaderG, broaderI, and architectural broaderP remain valid; causal, mitigation, protocol use, or co-occurrence alone does not establish hierarchy. No extra model call or fabricated confidence is attributed to deterministic evidence. Migrate memo identity, audit, token telemetry and all callers atomically.

Definition of done: native DSPy loopback captures prove label plus one relation call and direct-fact call count; test all four enum values, grammar/parser parity, orientation, source closure, deterministic rules, and native token-confidence. Run focused tests and required gates; independent critic review precedes any checkpoint or commit. This does not establish live model quality or benchmark parity.

### N1 implementation evidence

- Confidence measurement now consumes native DSPy `TokenLogprob` sequences, reconstructs exact bytes and preserves unknown confidence for absent metadata. Decision values are checked against parsed output before confidence is attached. Relation decisions measure the selected string's value span only; label decisions retain Boolean scoring.
- `tests/daemon/graph/inference/test_logprob_votes.py`: 16 tests passed, including Unicode, split tokens, invalid finite values, mismatches and a relation whose value tokens score 0.4 while quote and reasoning tokens are excluded. Native loopback real-vote separation passed before the concurrent relation-stage migration.
- A coherent loopback usage fixture verifies prompt, completion and total counts. A subsequent combined run was interrupted at collection by the in-progress replacement of hierarchy grammar and requires rerun after that change completes. These tests do not establish real-model quality.

R1 interim evidence: native Request/Response wrapping and request construction migrated; real loopback and sandbox selection passed. Focused reflection suite initially 36 passed, 3 failed due to obsolete test options and legacy adapter grammar; after test correction the no-grammar budget case passes and GBNF remains pending adapter migration. SIMBA suite collection was blocked by concurrent grammar/program edits. No live model or full-suite result is established.


## R2: Export the stored judgment explanation

The user identified the JSON reasoning as documentation for the chosen edge.
Current relational explanations already flow from decisions through the typed
EdgeRow.reasoning column, but Turtle exports discard them. Export a deterministic
reified rdf:Statement for each semantic relation, with the stored explanation as
skos:note and its current confidence as prov:value. Keep entity/fact definitions
unchanged and keep bare relations available. No new schema, inference call,
writer or claim of a complete judgment-history store. Verify with real SQLite,
Unicode/escaped multiline notes and stable repeat export. This is current-assertion
annotation, not completion of the separate CP2 historical judgment receipts.

R1 follow-up: reflection and edge tests passed 39/39 in the combined run, including native chat and Gemma completions HTTP loopbacks, character bounds, default settings, grammar, whole-example removal and sandbox evidence selection. After updating SIMBA fixtures for DSPy 3.5 plan fields and the real embedding adapter, SIMBA tests passed 8/8. Targeted ty and ruff checks passed. This does not establish real model quality, full-suite acceptance or running process state.


### P2 implementation evidence (in progress)

- Relation decisions use one typed four-value field. Fact-associated pairs constrain the grammar before inference; invariant validation rejects an unsupported negative even without grammar. The graph receives canonical broader direction while the audit retains the actual measured enum and reasoning.
- Native HTTP loopback tests `test_native_two_stage_relations_and_orientations`, `test_native_direct_fact_skips_identity_but_not_relation`, and `test_relation_grammar_and_invariants_constrain_fact_pairs` pass. They establish two calls for distinct unlinked pairs, one relation call for a direct-fact pair, four relation values, orientation and fact constraint; they do not establish live model quality or benchmark acceptance.
- Migrated parser tests: 10 passed. Native signature/schema selection: 6 passed. Existing three-stage fake-provider fixtures still require migration and the full quality gates remain pending. The optimizer fingerprint is `skos-shacl-v8-lossless-native-two-stage` to invalidate incompatible cached extraction and semantic records.
- Follow-up verification: migrated two-stage program, parser, task-quality and packet suites passed together (57 tests); native signature/schema selection passed 6 tests; grammar suite passed 11 tests. Scoped type checking of relation implementation and callers passed, as did scoped Ruff. Existing prompt-suite fixtures remain stale and full-suite gates have not passed. The model decision is copied before canonical edge projection so audit reasoning and `skos:narrower` remain unchanged.

P1 critic follow-up: packets now retain hidden labels and full fact definitions with each triple, and hydrate each referenced pointer before source-location deduplication. The real SQLite long-definition/hidden-label regression and focused packet/storage suite passed (35 tests). Hydration checks existence and spans but does not verify content-addressed IDs; distinct chunk hashes at the same coordinates remain unresolved because valid split chunks and externally assigned IDs cannot be verified from the current pointer alone. No stale-hash acceptance claim is made.

P1 finding dispositions: missing indexed chunk pointers now fail; hidden labels and full fact definitions are included; every pointer is hydrated before source-location deduplication. File/span absence is detected, but content-hash staleness is not established: this limitation predates the packet change (`hydrate_chunk_pointer` on HEAD also did not verify hashes), and valid split or externally identified chunks can share coordinates. Entity fact entries now contain `triple` and `definition`; production consumers inspect only pair-level direct/shared facts and do not parse entity fact entries. Native signatures describe shared evidence rather than imposing a competing input schema.

R2/native-chat follow-up review: semantic statement exports preserve stored weight using round-trip float representation, attach escaped notes, and retain bare edges; native chat rejects length-finished responses and cache replay checks native and serialized responses. The real loopback/cache and export tests plus packet/storage tests passed together: `uv run --locked pytest --no-testmon tests/daemon/graph/storage/test_reasoning_export.py tests/daemon/graph/inference/test_completions_engine.py tests/daemon/graph/inference/test_synonym_packets.py tests/daemon/graph/storage/test_storage.py -q --tb=short` (57 passed). This is not live-model or full-suite evidence and does not close a checkpoint.

### Live Unsloth evidence and R2 verification

- Isolated `outputs/wiring-audit/prompt-integrity/probe_live.py` used the configured Gemma completion transport with cache disabled and captured exact sanitized request bodies, rendered prompts, native responses and parsed audit. The Windows User environment supplied authentication to the child process only; no credentials were saved in artifacts.
- Five synthetic grounded pairs completed in 6.190467 seconds with exactly two model calls. Raw relation choices were related, related, broader, narrower and differentFrom; measured confidence was present on all five. Causal quantization constraints and protocol use did not become hierarchy. Canonical graph projection inverted narrower while the audit retained the model choice. This is a bounded live check, not a reconstruction of lost user examples or a corpus benchmark.
- Evidence: `request-01.json`, `request-02.json`, both corresponding `.prompt.txt` files, `response-01.json`, `response-02.json`, `input-evidence.json` and `result.json` in that artifact directory. Native DSPy JSONAdapter schema framing remains visible; no custom adapter was added to conceal it.
- Transport/context/export regression run: 21 passed (`transport-context.xml`). Full Unicode passage/context tails reach the HTTP body intact; default 360-character reasoning and 512-character/40-item extraction limits are absent; old compiled reasoning bounds do not override deployment configuration. Compiled prompts require the new complete-evidence two-stage contract.
- R2 real SQLite regression passed: complete Unicode/escaped multiline edge explanations are stored and exported as `skos:note` on deterministic reified statements; bare relations and resource definitions remain. Follow-up now uses round-trip float representation and verifies stored confidence 0.123456789 rather than a six-decimal rounded value. This is current-assertion annotation, not the separate CP2 receipt history.
- Independent Terra audit found missing hidden labels/fact definitions, a direct string-call length-finish hole, and confidence rounding; fixes and follow-up checks are in progress. Source hydration verifies files/spans, not arbitrary stored hash identity: same-location source IDs can legitimately refer to partial-line or JSON subchunks. Do not claim content-hash staleness detection without a stronger resolution contract.
- No process restart, optimizer reload, corpus publication, HMM acceptance, full-suite acceptance or commit is established by these results.

### Native direct-call and RLM follow-up

- Direct synchronous and asynchronous text calls inspect all native responses before DSPy converts them to output lists. Four HTTP loopback cases passed for sync/async with caching enabled and disabled, including a length response followed by successful output. Configured YAW cache rejected the incomplete response.
- Removed the obsolete interpreter-tool override; DSPy 3.5 supplies positional binding and typed-wire validation. A real sandbox and HTTP loopback test passed with a positional tool argument.
- DSPy 3.5 hosted embedding factory requires the optional OpenAI SDK, which is absent. Factory endpoint spelling updated from api_base to base_url; endpoint identity test passed. HTTP embedding test remains blocked pending dependency authorization. Focused lint passed; ty reports DSPy's kwargs annotation as incompatible with its scalar client options. These results do not establish live embedding inference or full acceptance.



### DSPy integration fixture follow-up

- Runtime and benchmark fixture suite passed 71 tests; extraction confidence and interrupted-process focused rerun passed 7 tests. A broader publication/evidence run passed 62 and exposed two stale fixtures that the focused rerun subsequently corrected.
- Native disk-cache suite passed 9 tests, including restart, cost accounting, endpoint isolation, request invalidation, and grammar invalidation through the supported Gemma completion engine. Generic native chat does not support grammar extensions. Loopbacks do not establish live-model quality or corpus acceptance.

## E1 telemetry owner boundary (in progress)

Observable behavior: each successful endpoint execution records one complete audit event with a unique request ID, regardless of how many vectors it returns. Its `num_batches` counts logical SDK batches. Async endpoints stage execution events without mutating engine state; the indexing and live-runtime owners drain them after update progress, including failure paths. Persistent memo reuse executes no endpoint call and records zero new endpoint executions. Missing provider token usage remains unknown (`None`), never a text-length estimate. Failed calls retain their original exception.

Definition of done: a real native CocoIndex plus HTTP loopback test batches 64 inputs and establishes one SDK batch and one audit event; a persistent memo reuse establishes zero new SDK batches and zero new execution events. Verify both owner paths, complete telemetry and counters, failure propagation, and required quality gates before any checkpoint. This does not establish full-corpus acceptance or model quality.


Native sandbox healer follow-up: DSPy 3.5 wraps submitted sandbox values in a wire envelope. The obsolete RLM output-normalization override consumed that envelope before native decoding and silently returned no repairs. Removing the override restores native decoding and typed validation. Two real-sandbox healer tests passed, including two returned repairs, hydrated evidence, direct worker delegation, grammar, call counts and token accounting. Broader non-live prompt/context/runtime run passed 50 tests with one separate stale embedding fixture still pending at the time of that run. Two live tests were not run successfully because the authentication token was unavailable. These checks do not establish live-model quality.
- A real loopback single endpoint batch of 64 vectors produced exactly one HTTP call and one complete execution event; in the separate CocoIndex mapped run the scheduler chose batches of 1 and 63, with one event per endpoint execution, not per vector. Reopening the persistent memo caused zero new HTTP calls and zero new events in that cache-disabled loopback test. `test_embedding_telemetry.py` and the graph semantic-publication telemetry regression passed.
- The runtime owner test confirms logical SDK batch counts and unique audit IDs after sync, unknown token usage, and no increase on no-op sync. Scoped type and lint checks remain required after final integration; these tests do not establish full daemon, HMM or corpus acceptance.

- SDK batch counts are logical dispatches: the native embedder may serve its own cache and does not expose whether a dispatch used the network. Events mark `memo_replayed=False` for actual CocoIndex endpoint executions only; memo hits produce no new event. Network-call claims above apply only to the observed loopback with SDK caching disabled.

## U1: evaluation-scoped usage accounting

Use only the current DSPy Prediction native LM usage for evaluation cost; never infer cost from a fixed slice of shared diagnostic history. Expose any existing query-scoped OpenIE value separately as reported historical diagnostics, without treating it as current indexing cost or adding it to the prediction total. Report when current prediction LM usage is unavailable and exclude unavailable embedding usage rather than estimating it. Validate a prediction aggregating more than 20 native usage events, preceding unrelated events, and a prediction with no native usage. Native usage is counted once; full-suite and live optimization acceptance remain separate.

U1 evidence: metric now reads the current native Prediction LM usage and sums total tokens; it no longer samples shared telemetry. Result and evaluation record expose LM-usage availability and mark embedding usage unavailable. A native DSPy tracker regression with 27 preceding and 25 current events verifies exactly 125 current tokens and zero/unavailable for a prediction without usage; focused pytest, Ruff and ty pass. Query-specific OpenIE values are separate historical diagnostics; the subsequent fresh-indexing test below establishes that current indexing usage is unavailable. No full-suite or live-run acceptance follows.


## C1: serving capacity and explicit context packing

The user rejected the arbitrary 65,536-character fallback and the last-20 telemetry sample. The metric now uses native current-Prediction usage, with unavailable usage explicitly marked (U1). No historical diagnostic tail is a cost source.

An endpoint reports token capacity, not a valid character/token conversion. Omitted max_context_chars therefore means no automatic character ceiling: selected passages, induced subgraph facts, definitions and complete edge notes remain intact; explicit positive packing budgets still constrain whole-line subgraph selection/AST expansion and never cut hit text. The server rejects oversized complete requests. Runtime binding resolves each loaded model's positive active context_length from authenticated native model metadata; max/native/VRAM advertised ceilings are not active capacity. Extraction and QA share the lower actual active serving capacity. Embedding checkpoint/quant/context metadata belongs in deployment/cache identity. Pure configuration construction remains offline; it does not establish a running server's capacity.

Definition of done: real HTTP metadata and whole-evidence regressions, capacity/alias changes invalidate serving/cache state, no automatic character cap, complete notes remain attached, native current-evaluation accounting, all applicable static gates and tests with independent critic. Real endpoint snapshot in outputs/wiring-audit/prompt-integrity/native35/serving-models.json reports Gemma active 131072 and embeddings active 8192. This does not establish token counting, corpus acceptance or process reload.

Native 3.5 fixture follow-up: migrated batch-budget coverage to a native DSPy Embedder and whole inputs; shared semantic packet assertions now reflect deduplicated evidence; publication uses an identified native callable; relation confidence reflects its single actual vote; engine restart loopback uses the configured native LM factory. The combined focused run had 33 passing and one obsolete batch-budget expectation, corrected and separately passing; publication rerun passed 6 and engine-edge rerun passed 5. Scoped Ruff and ty passed. This does not establish full-suite, live-model, retrieval or corpus acceptance.


U1 fresh-indexing scope follow-up: native HTTP loopback invoked a real prediction whose retrieval hook performed fresh CocoIndex publication. Its three LM calls consumed 45 reported tokens, while the prediction tracker recorded zero; indexing runs outside that tracker. Graph diagnostics separately recorded 15 relation tokens but did not supply OpenIE usage. Consequently the evaluation `tokens` value is explicitly current-prediction LM usage only; it no longer adds stale or missing OpenIE estimates, and the evaluation record and feedback mark candidate-indexing LM usage unavailable. Embedding usage remains unavailable. This is a scoped measure, not total candidate evaluation cost; it does not establish live optimization cost. SIMBA, confidence, publication and optimizer-focused checks passed 38 tests; the subsequent optimizer scope and SIMBA rerun passed 13 tests, including stale OpenIE value exclusion and the real fresh-indexing loopback.


C1 fixture follow-up: the authenticated loopback model-list responses now expose both configured loaded aliases, active context and checkpoint/quant identity. The detached-daemon test uses this real HTTP endpoint and verifies the authenticated model-list request before readiness and clean stop. Benchmark, benchmark-edge, engine-edge, embedding telemetry and CLI suites passed 78 tests; scoped Ruff and type checks passed. Loopback readiness does not establish production endpoint capacity, live corpus or model quality.

U1 critic disposition: any existing query-scoped OpenIE value is retained as `reported_openie_tokens` in the metric and audit, without attributing it to the current network operation or adding it to prediction tokens. A regression verifies a reported 100 remains separate from the prediction total of 5; current indexing LM usage remains explicitly unavailable.


Reflection sampling follow-up: role temperatures now inherit native bound LMs when no override is provided; unbound development uses 1.0, and explicit proposer overrides reach the respective predictor. Real loopback requests verified Gemma selector/writer 1.0/1.0, Qwen 0.8/0.7, and explicit 0.6/0.5, including every selector request and writer request. Reflection and SIMBA suites passed 35 tests; scoped type, lint and format checks passed. This is loopback request evidence, not live model quality.

## T1: exact Andrea Corr omission traced to Unsloth preview

The user's literal `[... 5,813 characters omitted ...]` was reproduced from the active Unsloth API monitor. The installed frontend api-monitor-BIzKDevs.js defines a 12,000-character preview (6,000 head + 6,000 tail), with that exact marker. Its `Show all` button expands retained content up to one million characters, and its Copy prompt button copies the full retained text rather than the displayed preview.

Authenticated read-only monitor retrieval located request apireq_ecb6ac5eb02d42dba2ecf8ca4ed0f6bf. Its full retained prompt has 17,813 characters, no omission marker, and a complete parseable synonym-evidence-v3 JSON packet (3 pairs, 6 entities, 6 sources). The display omission is exactly 17,813 - 12,000 = 5,813. The original request completed with 4,777 prompt tokens, 239 completion tokens and a 131,072-token active context. Reply reasoning lengths are 200, 179 and 255 characters. This observation establishes a frontend preview cutoff for this specific request, not a YAW field truncation.

Evidence saved under outputs/wiring-audit/prompt-integrity/: andrea-corr-full.prompt.txt, andrea-corr-hidden-preview-section.txt, andrea-corr-truncation-verification.json and monitor-apireq_ecb6ac5eb02d42dba2ecf8ca4ed0f6bf.json. No authentication headers or credentials were saved. No installed Unsloth files, running service or user optimization process was changed.

Separate limits remain distinct: Unsloth's monitor may evict full retained prompts after its aggregate 64 MiB storage budget and retains only 12,000 response characters; these are observability limits, not evidence of shortened model requests. DSPy RLM REPL output uses a different optional head/tail marker with parentheses. Neither separate mechanism produced the exact square-bracket marker in this record. This finding does not establish all future prompts, outputs, RLM histories, HMM retrieval or corpus acceptance. Other work was paused to resolve this report.

### Automatic budget decision verified (2026-11-14)

The user explicitly chose endpoint-derived automatic defaults after concurrent configuration edits pinned output and RLM budgets. Restored only max_output_tokens=None and rlm_max_output_chars=None; explicit positive overrides remain supported and validated against active serving capacity. The independently edited prompts.py was not modified. Capacity/configuration/real model-metadata loopback suite: 46 passed (outputs/wiring-audit/hmm/automatic-budget-final.xml). Native live reflection passed once with the restored automatic budgets (reflection-auto-budget.xml); the earlier full-run invalid evidence path remains recorded as a model-output validation failure, without parser relaxation or custom retries.

The completed pre-correction full run recorded 1072 passed and 17 failed (outputs/wiring-audit/hmm/full-suite.xml). Its nine configuration/capacity cases are covered by passing follow-ups, including the migrated serving-binding validation. The native cache-restart fixture now uses YAW's explicit chat model factory instead of dspy.LM's default Responses route; its cross-process reuse and prompt/model/sampling/source invalidation assertions pass (outputs/wiring-audit/candidate-cache-debug.log). Obsolete tokenization/template fixture endpoints were removed. The cache-disabled repeated query assertion now reflects actual dispatch count. No production cache fallback was added.

Type checking and read-only Ruff checks passed; formatting outside the user-owned prompts.py passed (211 files already formatted), and outside-prompt whitespace checking passed. These targeted follow-ups do not constitute a fresh all-green full-suite run. Six separately verified canonical-ingestion parity failures remain unresolved; no full-system acceptance, checkpoint or commit is claimed.

## L1: benchmark NER length-failure diagnostics (2026-11-14)

The user supplied a full `uv run yaw dspy benchmark` traceback showing OpenIE's NER stage failing in the completion transport on finish_reason=length. The failing response is no longer in the current 50-entry Unsloth monitor history; available logs establish neither actual requested token budget nor repetition. Do not change budgets, slice evidence, retry semantic failures or accept partial extraction without the original request/response.

Observable behavior: keep the original native LM failure; include requested max_tokens and provider usage; attach stage and extraction chunk context. Configured YAW/benchmark LMs save complete header-free request and provider response on a length failure under ignored outputs/inference_failures, with a path in the original error notes. Diagnostic I/O failure must not replace the model failure. Successful requests create no failure artifact. Sync and async transport share this behavior. Verify over real HTTP and filesystem, including no extra request, complete JSON still rejected, secret header absent, lossless saved partial output and a blocked diagnostic path. No prompt-file edit, budget increase, custom retry, running-process reload or recovery claim follows.

### L1 verification result

Configured Gemma transports now save the exact header-free wire request and complete provider response on finish_reason=length under outputs/inference_failures/length-<uuid>.json. The native error includes requested max_tokens and provider usage; original error notes identify the artifact, NER stage and extraction chunk through the actual CocoIndex runner. Omitted native limits report None rather than masking the failure with KeyError. Diagnostic filesystem errors add a note while preserving the original generation error. JSON escaping preserves partial Unicode, including lone surrogates. No partial generation is accepted and no additional model request is issued.

Real HTTP/filesystem/CocoIndex regression suite: 59 passed in 34.75s (outputs/wiring-audit/prompt-integrity/output-length/integrated-tests.xml and .log). Whole-repository ty check passed; scoped Ruff check, formatting and git diff --check passed. Terra independent review found the omitted-limit issue; corrected it and added regression coverage. Terra re-review reported no further actionable findings and did not independently execute tests. No Flash audit, completed acceptance checkpoint or commit is claimed.

This establishes failure observability, not the cause or resolution of the user's historical length response: its exact request and partial reply have expired from monitor history. No budget increase, prompt edit, semantic retry, benchmark rerun or running-process reload was performed. The six previously recorded canonical-ingestion parity failures and full-suite acceptance remain unresolved.

## B1: endpoint-owned generation capacity (2026-11-14)

The user explicitly removed the automatic output-token cap. Omitted max_output_tokens must remain None through configuration, standalone/program construction, runtime serving binding, compiled-state reload and DSPy. Neither max_tokens nor max_completion_tokens may appear in outbound JSON unless explicitly configured. The endpoint owns remaining generation capacity; no client token estimate or replacement finite default is introduced. Explicit positive limits remain validated and transmitted. RLM tool-output character safeguards are a separate contract. Preserve concurrent prompts.py edits. Acceptance requires real sync/async HTTP requests for both Gemma completion and native chat routes, explicit-limit preservation, runtime binding and stale-state checks, type/style checks, and independent Terra audit. This supersedes the earlier automatic token-budget decision; no full-system or live-process acceptance is inferred.

### B1 implementation and focused verification

Removed automatic output-token scaling from YawConfig.resolve_inference_budgets, create_dspy_lm and standalone YawProgram construction. None now propagates through runtime endpoint binding and GraphPlanSettings; optional character-budget calculations reserve no fixed output allowance when None. Explicit positive generation limits are still checked against active context and transmitted. Both native DSPy chat serialization and the Gemma completion transport omit unset limits; sync/async real HTTP tests verify no max_tokens or max_completion_tokens is sent and explicit 123-token limits survive LM copies unchanged.

Terra identified a cold-load escape where a compiled bound predictor could reconstruct an obsolete LM when the current program was unbound. load_state now imports instructions/demos without serialized clients, retaining current deployment LMs or the active DSPy context. Regressions cover explicit and omitted limits on bound deployments, old bound to new unbound loading, and hot reload. No prompt source file was edited.

Final relevant non-live selection: 151 passed, 2 live deselected in 66.54s (outputs/wiring-audit/prompt-integrity/output-length/uncapped-final-focused.xml and .log). The separate authenticated resident-Gemma checks passed 2 tests in 39.80s (uncapped-live-tests.xml and .log). Earlier unauthenticated checks failed for missing process credentials; those failures are not model failures and were not suppressed. Whole-repository ty check and read-only Ruff check passed; Ruff formatting excluded the concurrently owned prompts.py, with 211 files already formatted; outside-prompt git diff --check passed. Terra re-review, including optional GraphPlanSettings, found no further actionable findings; the reviewer did not independently run tests.

Read-only inspection of installed Unsloth confirms its GGUF managed routes accept omitted limits and defer to endpoint/server settings; no claim is made that the server has no separate configured limit. No custom retries, automatic replacement limit, live-process reload, benchmark-quality acceptance, Flash audit or commit was performed. The broader suite result follows separately.

### B1 full-suite result

The broad run completed with 1098 passed, 8 failed, 2 warnings and 3 subtests passed in 634.48s (outputs/wiring-audit/prompt-integrity/output-length/uncapped-full-suite.xml and .log). Two failures were the missing-authentication live checks, subsequently passed in the authenticated live selection above. The remaining six are the already-recorded canonical ingestion parity failures: fixed-evidence same-process/restart, ball-sense distinctness, late merged-member relation, paired G5/G6 acceptance report, and alias-class arrival order labels1. No new budget-contract failure was reported. The final focused selection independently covers the cold-load correction made after the broad run started. This is not a fresh all-green full-suite result; no acceptance checkpoint or commit is claimed. Existing running processes retain their in-memory settings; new invocations use the changed policy.
