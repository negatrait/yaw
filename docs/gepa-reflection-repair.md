# GEPA evidence and native proposal repair

The reflection path is now **training traces → sandbox evidence selection → native
DSPy InstructionProposer → downstream task scoring**. The previous synthesis
tool, prefix clipping and competing outer instruction output are removed.

## Task reward and textbook `dspy.GEPA` Pareto merging (2026-10-24)

GEPA and its SIMBA task scorer use `compute_task_quality_score`
(`yaw/daemon/graph/inference/parsers.py`, `task-quality-arithmetic-mean-v2`):

1. **Raw Pareto Frontier Objectives (`objective_scores`)**: All labeled raw metrics
   (`recall_at_1`, `recall_at_2`, `recall_at_5`, `recall_at_10`, `recall_at_20`,
   `em`, `f1`) are emitted in `objective_scores` so GEPA's validation-set
   `objective_pareto_front` (`GEPAState._aggregate_objective_scores`) macro-averages
   each metric across the validation set and retains non-dominated specialist
   parents across all 7 axes.
2. **Unweighted Arithmetic Mean Per-Example Scalar `score` (`task-quality-arithmetic-mean-v2`)**:
   Because GEPA's `StrictImprovementAcceptance` gate (`sum(new_minibatch) > sum(old_minibatch)`)
   and SIMBA's `score_candidate` operate on **per-example** scalar scores before a
   candidate ever reaches full validation, any geometric mean collapses `score` to
   `0.0` (or severely drags down the composite) whenever binary `EM == 0.0` (~94%
   of free-form `enterprise_rag_bench` answers even when `F1 = 0.67`), `F1 == 0.0`,
   or `Recall@1 == 0.0`. Instead, following `dspy-evaluation-harness`,
   `compute_task_quality_score` computes the unweighted arithmetic mean across all
   labeled metrics $M \subseteq \{\text{Recall@1}, \text{Recall@2}, \text{Recall@5}, \text{Recall@10}, \text{Recall@20}, \text{EM}, \text{F1}\}$:
   $$\text{score} = \frac{1}{|M|}\sum_{m \in M} m$$
   - With all 7 labels present, `score = (R1 + R2 + R5 + R10 + R20 + EM + F1) / 7`,
     naturally weighting retrieval at $5/7 \approx 71.4\%$ and QA at $2/7 \approx 28.6\%$.
   - QA-only/abstention examples use `(EM + F1) / 2`; retrieval-only examples use
     `mean(Recall@k)`.
   - Every incremental improvement in `Recall@1..20`, `F1`, or `EM` contributes
     linearly and additively to minibatch acceptance without any single `0.0`
     component erasing gains in the remaining components.
3. **Textbook `dspy.GEPA` + External `dspy.Ensemble(reduce_fn=dspy.majority)`**: Following
   textbook DSPy ([`docs/dspy-improvements.md`](dspy-improvements.md#2-canonical-dspygepa--dspyensemble-architecture-active-epic-1-rule), `dspy-gepa-optimizer`, `dspy-fundamentals`, `dspy-advanced-workflow`),
   `dspy.GEPA(..., candidate_selection_strategy="pareto", frontier_type="hybrid", use_merge=True, track_stats=True)`
   uses its Pareto frontier and system-aware `MergeProposer` at compile time to return a
   single `best_candidate` `HippoRAGProgram` saved via `compiled.save(path, save_program=False)`
   and exposes `compiled.detailed_results` (`DspyGEPAResult`). `HippoRAGProgram` never
   attaches internal ensemble attributes (`pareto_ensemble`, `qa_ensemble`, `_in_ensemble`)
   or `yaw_surface` ensemble states. Instead, `select_pareto_frontier_programs` extracts up
   to 3 strictly non-dominated whole `HippoRAGProgram` instances (`best_candidate` at index
   `0`, zero dominated backfill, deduplicated by query/QA predictor state) and wraps them
   from the outside via `ensemble = dspy.Ensemble(reduce_fn=dspy.majority).compile(programs[:3])`,
   with `HippoRAGProgram.forward` placing `answer` as the final `dspy.Prediction` field so
   `dspy.majority` returns the complete winning `dspy.Prediction`.

Tokens and graph repairs remain diagnostics. They neither penalize the scalar
reward nor create extra Pareto objectives. Missing token usage cannot invalidate
a quality evaluation. DSPy's normalized `answer_exact_match` supplies EM and
`dspy.evaluate.metrics.F1` supplies continuous best-reference token F1;
`answer_exact_match(frac < 1)` would return a thresholded boolean. LM-judged
`SemanticF1` remains an optional evaluation diagnostic via `include_semantic_metrics`.

The run and SIMBA cache identities identify `task-quality-arithmetic-mean-v2`.
When resuming an existing sandbox DB (`uv run yaw dspy optimize --resume`) across a
metric version upgrade, all expensive OpenIE extractions (`openie_chunks`), graph
tables (`chunks`, `entities`, `facts`, `edges`), and `candidate_stage_cache` entries
are preserved while stale pre-upgrade scalar subscores (`gepa_state.bin` and
`simba_task_score`) are invalidated so Candidate #0 and new candidates are scored
on the same `task-quality-arithmetic-mean-v2` scale.

## Behavior and configuration

`[inference].reflection_input_tokens` defaults to one quarter of the configured
serving window: **32768** at the **131072**-token Gemma default. An explicit
value pins the input cap. Automatic output allowance is one sixteenth of context
(8192 at 128K); RLM output characters and default proposal/diagnosis limits also
scale from their 64K deployment budgets. RLM turn/call limits remain independent.
Each final reflection
request is measured with the resident model's llama.cpp chat template/tokenizer,
including schema, demonstrations, instructions and evidence. Output tokens and a
512-token context margin are reserved independently. The external service must
already serve the selected model. The tokenizer backend URL is discovered from
Lemonade health; `reflection_tokenizer_url` supplies an explicit backend URL when
needed. Both `/v1` backend URLs and llama.cpp root URLs are supported. There is no
generic-tokenizer or character-estimate fallback. Different selector/writer models
require separate resident backends whose tokenizers can both be verified.

RLM selects stable example IDs, field paths and optional character spans. Host
code retrieves original values and verifies spans. Expected training answers
remain available as evidence. Invalid selections or exhausted RLM budgets fall
back to deterministic failure/success selection. The native writer receives the
component identity, immutable output contract and compact evidence. Optional
evidence is removed to fit; mandatory instructions/contracts are never sliced.
Native DSPy validates the instruction length and may make one compression call.

The native writer's predictors remain visible to SIMBA under the established
`qwen_proposer` / `qwen_compressor` names. Their native signatures replace the
previous implementations. SIMBA uses actual paired downstream task improvement,
retains bounded training examples and validates learned prompt capacity before
adoption. Held-out answers never enter reflection or SIMBA calibration.

## Persistence and compatibility

Evidence selections and proposals are independently cached in the existing
SQLite infrastructure. Each request checkpoints collecting, learning, learned and
complete phases. A bounded active-request snapshot is pinned in SQLite metadata
and committed with its learned worker state, independently of FIFO cache eviction.
On restart, incomplete learning starts from its recorded input state; completed
learning/proposals are reused. An interrupted in-flight generation can run again.

Worker state version is **2**. Old worker snapshots and mismatched model/data/code
identities fail compatibility checks; checkpoints are preserved. Corpus strings
and selected excerpts in snapshots use source identities/spans, hydrated from the
original corpus instead of duplicating corpus text in the database. Diagnostics
and learning histories remain bounded; reflection LM histories are disabled.
Reflection and SIMBA advice use isolated DSPy traces; only the active SIMBA
trajectory collection retains the training traces it needs.

MIPROv2 search/warmup policy is unchanged. Its existing shared instruction-worker
entrypoint uses the replacement evidence/native-writer implementation.

## Restart and verification checklist

- [x] Replace competing writers and preserve original training evidence.
- [x] Enforce serving-token budgets across the writer, RLM and SIMBA.
- [x] Pin active stages, persist learned state and hydrate source references.
- [x] Finish focused persistence, budget, sandbox and architecture verification.
- [x] Record the bounded serial EnterpriseRAG pilot and its limitations in evaluation.md.
- [ ] Complete all twelve held-out pairs: deadline reached after six; no quality gain established.

The pilot targets NER, synonym judging and QA, with two training cases and twelve
disjoint held-out cases. It compares the starting program with the proposed
program, measures paired Recall@1/@2/@5, EM/F1, task score, latency, logical tokens,
LM calls, cache hits and process memory. Missing component traces or incomplete
runs must be reported; they do not establish a quality gain. No overnight optimizer
is launched automatically.


Actual verification and partial pilot scores are recorded in
[evaluation.md, section 34](evaluation.md#34-evidence-driven-native-gepa-reflection-replacement-2026-10-23).
The pilot's Python peak working set was about 424 MiB, excluding model servers
and the sandbox. This is a measured bounded run, not a guarantee against resource
exhaustion. Start a new compatible optimization run for the replacement worker;
legacy version-1 worker state cannot be resumed by version 2.


At final verification, concurrent dependency edits left `pyproject.toml` and
`uv.lock` out of sync. Reconcile that dependency work before a locked launch;
reflection tests used the previously installed dependency set (CocoIndex 1.0.25).
