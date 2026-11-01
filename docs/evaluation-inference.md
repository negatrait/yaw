# Evaluation Record: DSPy Inference & Optimization

Domain evaluation ledger for DSPy signatures, GBNF worker vs unconstrained RLM orchestrator execution, GEPA / MIPROv2 / SIMBA optimization, candidate-stage & response caches, Gemma 4 128K capacity scaling, and external Pareto `dspy.Ensemble`. See the [Evaluation Index](evaluation.md) for all other domain ledgers.

## 20. Candidate-Indexed Optimizer Telemetry & Watchdog Staleness Fixes (2026-10-21)

- **Scope:**
  - **Engine (`yaw/daemon/graph/engine.py`):** Every scored metric call in `evolve_prompts` is now attributed to an optimizer candidate via `_candidate_fingerprint(program)` (SHA-256 over all predictor surfaces + embedding prefixes), captured from the program passed to `_canonical_retrieval_hook`. Candidates are numbered in first-evaluation order (MIPROv2 trials and GEPA proposals alike; re-evaluated parents map back to their original number). The audit JSON gains `candidate_records` (per-candidate n, mean/min/max score, R@1..20, EM, F1, s/eval, tokens, plus GEPA `gepa_idx`/`val_score`/`is_best` from `on_valset_evaluated`). Audit writes are atomic (`_write_text_atomic`).
  - **Fixed real crash:** `engine.py` used `time.time()` without `import time` (also missing at `HEAD`); `yaw optimize` failed with `NameError: name 'time' is not defined` right after validation indexing. `last_eval_ts` was also missing from the metric closure's `nonlocal`.
  - **Watchdog (`scripts/watchdog.py`):** Figs 3–5 are indexed by candidate # (not wall-clock). Removed stale/synthetic sources: fallback to other runs' audit files, scraping old task logs, and R@1/R@2/R@10/R@20/EM values fabricated from R@5. Reads only `compiled_program_optimize_<datestamp>_graph_rlm_audit.json` for the active sandbox, cached by (mtime, size) with last-good reuse on torn reads. Removed `immutable=1` SQLite reads on the live DB, scoped the DB snapshot cache per database, derived "running" from the status-file PID instead of "records found", wrote placeholders instead of skipping when data is empty, clamped Fig 1's time axis until throughput is measurable, rendered figures atomically and isolated them so one failure doesn't stop the rest.

- **Verification:**
  - Synthetic audit render check (1 candidate, 25 candidates, legacy untagged, empty, torn JSON) $\to$ all figures rendered, no temp files left, torn read reused last snapshot.
  - `uv run --no-sync pytest tests/core/test_architecture.py tests/daemon/graph/test_engine.py -k "architecture or homographs or evolve_prompts_reindexes"` $\to$ **11 passed**.
  - `test_evolve_prompts_real_unmocked_gepa_and_miprov2_with_context_limit_guard` $\to$ **fails** on `len(compiled.shacl_healer_rlm.generate_action.demos) >= 1` (demos are cleared when `max_*_demos == 0`). This assertion was unreachable at `HEAD` because of the `NameError`, and the code involved was not changed here.
  - `uv run --no-sync ruff check scripts/watchdog.py` $\to$ **All checks passed!**; `git diff --check` $\to$ **Clean**.
  - Live: `yaw optimize --db latest` resumed on `yaw_optimize_20261021_170138.db`; watchdog cycle reports `OPTIMIZER RUNNING • indexing: …` with no stale eval counts.

## 21. GEPA Metric Fix: Quality-Led Score, 10-Example Minibatch, Clean Objective Score Card (2026-10-22)

- **Observed failure (run `yaw_optimize_20261021_174008`, 150 train / 100 val, Qwen3-4B):** after 10 candidates (401 evals) the seed stayed "best" (val score `0.0573`) while candidate #5 had higher full-valset R@5 (`0.90` vs `0.87`) and F1 (`0.359` vs `0.350`). Six of ten candidates were judged on 3-example minibatches only.
- **Root causes (`compute_task_quality_score`, `yaw/daemon/graph/inference/parsers.py`):**
  - `target_tokens=100` against ~20k tokens/query made the token cost term ~228; a 50% token cut moved the score ~20% while a +10pt F1 gain moved it ~3%. The score tracked cost, not quality.
  - `objective_scores` (GEPA per-objective frontier -> `ParetoProgramEnsemble` / `dspy.Ensemble`) included `gm_bad` (lower-is-better), so the frontier kept the most expensive candidate, plus redundant composites (`gm_good`, `pareto_raw`, `quality_numerator`).
  - `reflection_minibatch_size=3` (`yaw/daemon/graph/engine.py`) let a single answer accept/reject a proposal.
- **Changes:**
  - `target_tokens` default `30_000` (about USD 0.10/query enterprise ceiling); token cost enters the denominator at half weight (`sqrt`).
  - Recall@5 and F1 carry triple weight in the quality geometric mean.
  - `objective_scores` now holds only higher-is-better metrics: Recall@1/2/5/10/20, EM, F1, `shacl_conforms`, `efficiency`.
  - `reflection_minibatch_size=10`.
- **Sensitivity at candidate #1's real numbers (R@5 0.87, F1 0.35, 22.8k tokens):** +10pt F1 -> x1.065; -50% tokens -> x1.031; +5pt R@5 -> x1.014 (previously -50% tokens dominated).
- **Verification:**
  - `uv run --no-sync pytest tests/daemon/graph/inference/test_prompts.py tests/daemon/test_runtime.py -k "pareto or sizecoding or gepa or metric or composite or Pareto or repair"` -> **5 passed** (new assertions: a +10pt F1 gain outranks halving tokens; no lower-is-better objective keys).
  - `git diff --check` -> **Clean**; no new ruff findings on changed lines (repo baseline already has 712).
- **Restart:** stopped the old run; archived its GEPA state and audit as `*_oldmetric` (old scores are not comparable); restarted `yaw optimize --db .yaw/yaw_optimize_20261021_174008.db`, reusing the indexed sandbox.

## 22. E1-S6 Explicit Optimization Resume and Checkpoint Provenance (2026-10-22)

- **Checkout verified:** `C:\Users\Harri.Ahokas\Documents\yaw`, origin
  `https://github.com/negatrait/yaw.git`. Existing working-tree edits and live
  optimization databases were preserved; no user run was stopped or launched.
- **Failures identified:** default CLI invocations create new sandboxes;
  `--db latest` selected by database size; `--db-path` and `--resume` were not
  parser options. YAW's preflight treated GEPA's serialized dictionary as an
  object and could delete a valid `gepa_state.bin`. Restart hydration assigned
  the current prompt fingerprint to old graph rows without durable provenance.
  Incremental ingestion could return an empty optimizer passage pool on restart.
- **Changes:** `--resume`, `--db-path latest`, and retained `--db` alias select
  existing runs by their filename creation timestamp. Missing runs fail.
  SQLite run metadata restores the original CLI dataset/slice/budget and custom
  output path. Writer-locked validation fingerprints ordered datasets/corpus,
  inference/indexing settings, graph/ingestion code, and relevant package versions.
  Incompatible and unverifiable legacy GEPA checkpoints are preserved and rejected.
  GEPA state is loaded through its own loader. A saved seed program supports
  restarting GEPA after completed MIPRO initialization. Fresh temporary CocoIndex
  scan state reconstructs the complete pointer pool without discarding saved graph
  results. Transactional per-chunk OpenIE/index fingerprints in `meta` distinguish
  matching cached results from mutated candidate prompts; legacy unverified rows
  are recomputed. DSPy response caching during evolution is unchanged (disabled).
- **Environment / installation:** first sandboxed `uv run --locked` failed while
  initializing the external uv cache; installed package-source reads were also
  denied by sandbox access. Approved access to the existing environment resolved
  verification. Python **3.14.7**, DSPy **3.4.1**, GEPA **0.1.4**.
  `uv sync --locked` resolved and checked **97 packages** without dependency or
  lockfile changes.
- **Focused verification:**
  - `uv run --locked pytest tests/core/test_config.py tests/core/test_architecture.py
    tests/daemon/test_runtime.py tests/daemon/graph/test_engine.py -q
    -k 'not real_unmocked_gepa_and_miprov2_with_context_limit_guard' --tb=short`
    -> **55 passed, 1 deselected** before adding the final missing-run CLI test
    (`outputs/resume-focused-tests.txt`).
  - Final `uv run --locked pytest tests/core/test_config.py tests/daemon/test_runtime.py
    tests/daemon/graph/test_engine.py -q
    -k 'resume or optimization_latest or optimization_custom' --tb=short`
    -> **9 passed, 39 deselected** (`outputs/resume-regression-tests.txt`). Includes
    bare resume and `--db-path latest`, recovery without a final compiled JSON,
    real GEPA state containing a superior saved candidate, unchanged stored
    frontier/budget, complete nonempty passage pools, persisted OpenIE reuse after
    object recreation, prompt-mutation invalidation, changed-data refusal with
    byte-identical checkpoint/corpus preservation, and legacy/missing-run errors.
  - The first new legacy-state test had a misplaced assertion from its neighboring
    test (`NameError`); corrected placement, and the final regression run passes.
  - Full focused suite initially: **48 passed, 1 failed**. The existing hybrid
    context-limit test asserts that SHACL RLM demos are nonempty although configured
    defaults clear demos. Running that test with the committed `HEAD` engine loaded
    in memory reproduces the same **`assert 0 >= 1`** without changing checkout
    sources (`outputs/resume-hybrid-baseline.txt`). It remains an existing failure.
  - Lint across the seven touched Python files reports **173 findings**. Comparison
    against committed source found only three extra line-length findings, all in
    the pre-existing working-tree candidate-telemetry edits; this resume patch adds
    no lint findings. Import-order check passes. `git diff --check` passes.
- **Scope of evidence:** real SQLite, CocoIndex, DSPy and GEPA checkpoint loading
  with deterministic offline model/embedding providers validate orchestration and
  storage reuse. No live inference, process-kill/power-loss experiment, retrieval
  quality benchmark, full-corpus Tier 2 parity, or HTTP transport acceptance is
  claimed. In-flight work can repeat; GEPA 0.1.4 reevaluates the seed validation
  batch before loading saved state. GEPA budgets are total metric-call caps,
  including calls already recorded; an exhausted run needs a larger budget.

## 23. E1-S6 Priority 1: Persistent Candidate-Stage Caches (2026-10-22)

- **Authorized scope:** prioritize candidate-stage persistence for low-memory
  machines. Native DSPy request/response caching (priority 2) remains disabled
  during optimization; no claim of caching every QA/filter/reflection request.
  Checkout and remote were reverified; pre-existing edits, experiments, and the
  user's concurrent `inference/prompts.py` improvements were preserved.
- **Implementation:** replace the four result dictionaries (embedding, OpenIE,
  synonym, SHACL) with disk-only `CandidateStageCache` views over one
  `candidate_stage_cache` table in `yaw.db`. The shared compressed-payload limit
  is **256 MiB**, with transactional FIFO eviction and trigger-maintained byte
  accounting. No extra dependencies, pickle, resident historical result maps,
  or source-text index files. Entries hold typed JSON with float32 vector bytes,
  OpenIE/StructuredTriple records, DSPy predictions, and typed REPLHistory.
  Trace source text is replaced with content references and hydrated from the
  current source pool, including text embedded in JSON/REPL history.
- **Validity and accounting:** keys include source/input content, predictor
  instructions/demos/schema/config, effective and bound LM settings, embedding
  profile/endpoint/settings, implementation hash, and DSPy/NumPy versions.
  SHACL also fingerprints the graph and source content visible to its tools.
  Original OpenIE/synonym/SHACL token statistics and traces are replayed, keeping
  cached stages from appearing artificially cheaper in GEPA. Failed SHACL calls
  are not memoized. Corrupt, incompatible, missing-source, and evicted entries
  miss and recompute. Uncommitted/in-flight work can repeat after a crash.
- **Verification:**
  - `uv sync --locked`: **97 packages resolved/checked**; dependencies and lockfile
    unchanged.
  - Initial architecture/stage checks: **12 passed, 19 deselected**
    (`outputs/candidate-cache-initial-tests.txt`).
  - Real SQLite tests verify shared-budget eviction, replacement accounting,
    transaction rollback restoring evicted entries, zero/negative limits, typed
    record/prediction/vector/RLM-history round trips, Unicode source hydration,
    invalidation, and corrupt-entry removal. Existing SHACL/RLM integration also
    exercises serialization of real DSPy trace objects and token statistics.
  - Broad focused run across architecture, storage, graph, runtime and new cache
    tests: **76 passed, 1 failed, 1 deselected**. The process test failed because
    the implementation changed between workers (recorded code fingerprints
    `d04cb5b338f99811015cff139803953fd3656ebe3c0ad113f1f71081a46bb7e7`
    and `06a9c732f301857675b876de8f5c6216fdd24997924ec6fe34170c34030d9500`).
    The user confirmed concurrent prompt improvements. This correctly invalidated
    cached results; it was not a valid unchanged-code replay measurement.
    Evidence: `outputs/candidate-cache-regression-tests.txt`.
  - The final process test copies one immutable implementation snapshot and
    asserts both workers imported it. `uv run --locked pytest
    tests/daemon/graph/test_candidate_cache.py -q -s --tb=short` -> **2 passed**
    in **130.19 s** (`outputs/candidate-cache-process-final.txt`). Uses a real
    loopback OpenAI-compatible HTTP server, native DSPy LM/embedding clients with
    native response caching disabled, actual SHACL RLM execution, and independent
    Python processes. Each worker deletes the materialized graph and OpenIE rows
    before rebuilding; only persisted candidate-stage results can satisfy replay.
  - **Measured:** first worker **16 HTTP inference requests**; second worker
    **0 additional requests**. Second-process cache hits: **embedding 8,
    OpenIE 2, synonym 2, SHACL 1**. Preserved logical tokens: OpenIE **60 + 60**,
    synonym **120**, SHACL **60**, missing definitions **0**. Trace predictor names
    match. Prompt, model, sampling-temperature, and source mutations each cause
    fresh inference requests.
  - New cache modules and test helpers: **ruff clean**. Comparing touched existing
    runtime modules against HEAD finds only the three already-existing telemetry
    line-length additions recorded in section 22. `git diff --check`: **clean**.
- **Development failures fixed:** nested ManagedConnection acquisition blocked
  the first test; cache writes now join an existing transaction. The initial
  loopback fixture returned the wrong embedding dimension and used an incorrect
  HTTP-server cleanup interface; corrected. Real RLM traces exposed REPLHistory
  as an unsupported type; added explicit lossless serialization rather than
  stringification or pickle. Cross-process verification now isolates source
  snapshots so concurrent edits cannot masquerade as cache-key drift.
- **Limits:** loopback responses are scripted and establish orchestration,
  persistence, trace/cost parity and actual HTTP request counts, not model quality
  or measured RAM/VRAM savings. No live inference-server benchmark, power-loss
  experiment, Tier 2 parity gate, or HTTP MCP transport acceptance was run. The
  pre-existing hybrid-demo assertion from section 22 remains deselected.

## 24. E1-S6 Priority 2: Native DSPy Response Disk Cache (2026-10-22)

- **Scope:** surgical feature in two runtime modules: `inference/chat.py` and
  `graph/engine.py`. One new focused test module, plus README/epic/evaluation
  updates. No GEPA, signature, stage-cache, dependency, or lockfile rewrites;
  the user's prompt improvements remain intact.
- **Implementation:** continue using `dspy.configure_cache` and DSPy's native
  request hashing, restricted disk serialization, eviction, cache controls, and
  response loading. A small adapter on the public `get`/`put` interface restores
  persisted logical usage into the active DSPy tracker on LM hits. DSPy's own
  LM-history usage remains zero for hits, keeping physical usage separate from
  optimization cost. Native records already include original usage; legacy
  LiteLLM SDK records get a small usage companion in the same DSPy cache.
  Old legacy entries without recoverable cost miss rather than scoring as free.
  Non-LM cache traffic passes through. Endpoint fields remain in cache keys;
  DSPy's default ignore list would otherwise conflate local servers sharing a
  model name. Explicit `cache=False` is respected.
- **Engine configuration:** native disk response caching is enabled and its
  memory tier disabled during evolution. Logical accounting is also enabled at
  engine initialization and after compilation so warm preparatory indexing and
  final evaluation have the same cost semantics. No new CLI flags are needed.
- **Installed-source inspection:** DSPy 3.4.1 persists native `CallResult.usage`,
  clears it in `CallResult.load`, and skips `usage_tracker.add_usage` for hits.
  The adapter supplies that missing accounting step without modifying DSPy or
  GEPA internals or replacing their provider execution path.
- **Verification:**
  - `uv run --locked pytest tests/daemon/graph/inference/test_response_cache.py
    -q --tb=short` -> **4 passed in 212.11 s**
    (`outputs/response-cache-tests.txt`). Real loopback HTTP and independent
    Python processes; memory response caching disabled. Identical requests:
    **1 initial HTTP call**, **0 extra calls after restart**, original logical
    usage **30 tokens per call**, native history usage **0 on hits**. Changed
    prompt/model/temperature/grammar/endpoint cause new calls. `cache=False`
    causes both repeated calls to execute. Legacy SDK disk hits retain costs.
  - Real `dspy.GEPA` is compiled independently in each process with distinct
    checkpoint directories and a one-call metric budget. The second compilation
    makes no additional HTTP call while retaining **30 logical tokens** and the
    identical validation score **1 / 1.3**. This verifies response caching rather
    than reuse of GEPA's saved evaluation scores.
  - Final checks after the small nullable-usage counter guard and passthrough
    test: architecture, real GEPA score parity, non-LM passthrough/unreported usage,
    and existing cache configuration -> **12 passed, 35 deselected in 52.04 s**
    (`outputs/response-cache-final-checks.txt`).
  - Broader targeted architecture/cache/GEPA/metric/resume run -> **14 passed,
    1 failed, 67 deselected** (`outputs/response-cache-regression-tests.txt`).
    The failure is the already-reproduced hybrid-demo assertion from section 22
    (`assert len(compiled.shacl_healer_rlm.generate_action.demos) >= 1`, actual 0).
    It is unrelated to request-cache accounting and is left unchanged.
  - `uv sync --locked`: **97 packages checked**, no dependency/lockfile changes.
    New test-module lint and `git diff --check` pass. Existing runtime-file lint
    diagnostics are not suppressed or broadened into this feature.
- **Limits:** scripted loopback responses verify actual HTTP request counts,
  persistence, explicit bypass/invalidation, and native GEPA cost/score parity;
  no live GPU/VRAM measurement, retrieval-quality benchmark, Tier 2 gate, or
  power-loss test is claimed. Requests with changed parameters, evicted entries,
  or missing legacy usage can recompute. GEPA still performs seed evaluation on
  restart, but identical LM calls within it can now be disk hits.

## 26. Optimization RAM retention and contained audit fixes (2026-10-22)

- **Priority and scope:** implement the audit's RAM-retention fixes first, plus
  query-prefix invalidation, effective embedding batch controls, and parser-helper
  tracking. Source-map reconstruction performance remains deferred. The existing
  user edits to `yaw.toml` and `yaw/core/config.py` were preserved; only the new
  embedding character-budget setting was added alongside them. No dependency,
  lockfile, prompt, benchmark-corpus, or production database changes.
- **RAM behavior:**
  - `evolve_prompts` uses DSPy's supported `disable_history=True` /
    `max_history_size=0` context throughout validation, compilation, reflection,
    and final evaluation. It suppresses full LM/module/global inspection history,
    while DSPy traces, disk response caching and usage trackers remain active.
    Context exit restores the caller's settings, including on failure.
  - Recent diagnostics retain 250 telemetry/embedding/definition records and 100
    synonym/SHACL/SIMBA records. Trimming occurs after accounting slices have
    been consumed, per chunk and at QA/reflection boundaries; it also runs on
    failed QA/reflection operations. Candidate deep copies omit these diagnostic
    lists and module inspection history while retaining predictor instructions
    and demonstrations. Per-predictor inspection history is capped at 100.
  - Reflection training retains the same last `max(simba_bsize, 8)` examples that
    SIMBA already consumes. Its seed/step counter is independent of trimmed
    diagnostics. Evaluation detail retains the last 500 records, matching the
    existing exported window; per-candidate aggregates remain available.
- **Other contained fixes:**
  - Shared stage namespaces exclude mutable query prefixes. Document vectors
    and graph materialization no longer invalidate for query-only evolution;
    query-vector keys use the effective adapter prefix, including when evaluating
    a candidate distinct from `engine.program`. Document/model/endpoint and
    other existing dependencies remain in the namespace.
  - `embed_batch_size` now reaches the adapter's request partitioner rather than
    silently falling back to 16. `[graph].embed_batch_chars` exposes the aggregate
    formatted-character budget, retaining the conservative 3,500 default and
    existing overflow splitting. Outer groups of eight indexing passages remain;
    no larger server request budget or concurrency increase was assumed safe.
  - CocoIndex's memoized per-file component tracks its parser module's source
    digest through `deps`, with a one-time version bump for existing memos. Plain
    helper/projection-constant edits therefore invalidate persistent parsing
    state without moving DSPy/graph mutations into CocoIndex memoized calls.
- **Focused verification:**
  - `uv run --locked pytest
    tests/daemon/graph/inference/test_diagnostics.py
    tests/daemon/graph/inference/test_batch_budget.py
    tests/daemon/ingestion/test_corpus.py
    tests/daemon/graph/inference/test_embedding.py
    tests/core/test_architecture.py -q --tb=short`
    -> **29 passed in 26.14 s**. Includes predictor-copy preservation, operation
    slice accounting and failure trimming, effective item/character budgets,
    and parser-helper invalidation across separate processes using real LMDB.
  - Cache integration/storage suites (`test_candidate_cache.py`,
    `test_response_cache.py`, storage `test_candidate_cache.py`) -> **10 passed
    in 376.69 s**. The 600-call real-DSPy QA test used real disk caching and
    loopback HTTP: **1 provider request**, **18,000 logical tokens**, no LM/module/
    global inspection-history growth, 250 retained telemetry entries, and
    **-9,056 bytes** change in retained tracemalloc allocation between calls
    300 and 600 (after GC). GEPA-required prediction traces remained present.
  - Strengthened separate-process stage test rerun -> **2 passed in 155.89 s**.
    Initial materialization made 16 HTTP requests; unchanged replay made zero,
    with embedding/OpenIE/synonym/SHACL hits and unchanged logical cost. Changing
    the query prefix produced different query vectors and exactly two embedding
    requests (original and changed query), with no additional graph-stage HTTP
    requests; repeating the changed query hit the stage cache.
  - Runtime config/parser/real-checkpoint-resume selection -> **3 passed,
    15 deselected in 23.86 s**.
- **Broader checks and failures:**
  - Diagnostic/batch tests plus the full prompt suite -> **34 passed, 2 failed
    in 51.16 s**. Failures: a case-sensitive `NEVER include specific` prompt-text
    assertion and an invariant assertion expecting duplicate NER names to lower
    the score. Both reproduce against isolated HEAD runtime sources: **2 failed,
    31 deselected in 14.56 s** (`outputs/stability-prompt-baseline.txt`). They do
    not involve these changes; prompt and invariant implementation files were
    left untouched. The existing live-server smoke tests in that suite passed;
    they are not throughput or benchmark-parity measurements.
  - Optimizer/telemetry selection -> **5 passed, 1 failed, 18 deselected in
    65.46 s**. The failure is the same previously reproduced zero-demo/hybrid
    assertion documented in sections 22 and 24; real GEPA/MIPRO compilation ran.
  - New modules/test helpers are ruff-clean. Whole touched-file lint still finds
    existing diagnostics; comparing current and HEAD source diagnostics reports
    **zero new diagnostics** (`outputs/check_stability_lint.py`).
    `git diff --check` passes. Initial candidate-copy test attempts incorrectly
    targeted the ChainOfThought wrapper rather than its `.predict` parameter;
    the corrected test verifies the actual serialized/optimized predictor.
- **Limits:** this bounds accumulated diagnostic history, not the entire process.
  Active corpus/graph data, current operation traces, and GEPA's required
  candidate/frontier state still consume RAM. The retention experiment measures
  Python allocations with a scripted loopback provider, not full-corpus RSS or
  peak VRAM. No unattended multi-hour run, increased-budget GPU throughput,
  Tier 1/Tier 2 quality gate, or power-loss durability result is claimed.

## 27. MIPROv2 zero-shot warm-up audit (2026-10-22)

- **Scope:** audit requested for unwanted few-shot work before GEPA. Verified
  the Documents checkout/remote and inspected installed DSPy 3.4.1 and native
  TPE behavior. No runtime/config/prompt/dependency changes for this audit.
  Full findings: [DSPy practical improvements](dspy-improvements.md).
- **Confirmed mismatches:**
  - Native MIPRO bootstraps proposal context even with both demo limits zero
    (installed constants: 3 bootstrapped, 0 labeled). Zero limits remove demo-set
    search, not the bootstrap stage itself.
  - YAW seeds built-in demos and clears them only after MIPRO evaluation. Native
    instruction-only selection does not clear inherited demos, so the scored
    program can differ from the demo-free GEPA seed.
  - Large-validation hybrid defaults force 3 candidates / 4 new trials rather
    than native auto sizing. TPE defaults to 10 startup observations; baseline
    plus 4 trials never reaches adaptive sampling. Setting only `--num-trials`
    also defaults the instruction candidate count to 1, eliminating alternatives.
  - MIPRO's default `track_stats=True` attaches program archives to its winner;
    YAW deepcopy carries those archives into later candidates. This is separate
    from the diagnostic-retention fix in section 26.
  - Reflection SIMBA has its own two-demo default, independent of task-level
    MIPRO limits. Current `yaw.toml` selects GEPA-only, which skips MIPRO entirely
    and also skips the MIPRO-specific postcompile demo clearing.
- **Probe:** `uv run --locked python outputs/audit_mipro.py` -> **passed**.
  Uses actual installed native bootstrap/search stages, real DSPy evaluation,
  Optuna and loopback HTTP with a small scripted task, without GPU calls.
  `outputs/mipro-audit.json` records:
  - Zero limits still made **4 bootstrap requests**, producing demo sets of
    sizes **[0, 1, 3]**.
  - Instruction-only search retained **1 inherited demo** in all **5 evaluations**
    (baseline plus 4 new trials), and in the selected program.
  - MIPRO attached **5 archived programs**; a YAW deepcopy retained independent
    copies of the archive and its program objects.
  - YAW's default seed has **15 built-in demos across 10 predictors**. Native
    light/zero-shot sizing for those predictors would request **51 new trials**,
    unlike the hybrid shortcut's four.
- **Limits:** the probe supplies instruction candidates directly to the native
  search and does not run the complete YAW CLI/RLM warm-up, measure full-run RSS,
  or establish quality lift. Proposed fixes are not implemented in this audit.
  `git diff --check` passes after recording it.

## 28. SIMBA reflection-worker wiring verification (2026-10-22)

- **Scope:** verify trigger, native compilation, metric, resampling and adoption
  of the learned reflection worker. No SIMBA runtime behavior was changed.
  Details: [DSPy practical improvements](dspy-improvements.md).
- `uv run --locked python outputs/audit_simba.py` -> **passed**. Uses real DSPy
  3.4.1 SIMBA with its default two-example/two-sample/one-step/two-demo bounds and
  a scripted loopback provider. **14 training HTTP requests + 1 next-call check**.
  Both sampled buckets had score variation; native `append_a_rule` produced the
  selected rule. Winner score **1.0** versus baseline **0.45** on the proxy metric;
  the next real worker request contained the adopted rule and returned its
  intended output. All four rollout requests used **temperature 1.0**.
  This deterministic run selected rules and added zero demos. Results:
  `outputs/simba-audit.json`.
- Existing integration contract (`test_prompts.py -k
  dspy_module_and_always_on_simba`) -> **1 passed, 32 deselected in 22.77 s**.
  The existing test confirms compilation telemetry; the new probe additionally
  verifies that the winning predictor is used by subsequent calls.
- **Limits found:** a sensible extraction instruction and an instruction to
  ignore every passage and emit an empty `named_entities` list both receive
  **1.0** from the meta-metric. It measures formatting/generalization proxies,
  not retrieval or answer lift. A fresh proposer loses learned worker state;
  YAW has no restore path for the SIMBA worker/training window/step counter.
  Compilation is attempted on each eligible proposal after the threshold, and
  caught failures are only debug-logged; the outer callback ignores False.
- Native SIMBA's rule-advice call emitted input-type warnings for stringified
  rewards/module names but completed; no upstream type behavior was patched.
  This verifies mechanics with scripted responses, not held-out quality, full
  interrupted-run restoration, GPU throughput, or multi-hour RAM behavior.

## 29. Gemma 4 runtime model migration (2026-10-22)

- **User-authorized change after the SIMBA check:** switch Qwen chat/extraction/
  QA/reflection roles to Gemma 4. Read-only registry/health inspection identified
  the already-downloaded and loaded **gemma-4-E4B-it-qat-q4_0-gguf-Q4_0**,
  `unsloth/gemma-4-E4B-it-qat-q4_0-gguf-Q4_0:Q4_K_M`, on the existing external GPU llama.cpp
  service with **32,768 context**, `--parallel 1 --cache-ram 0`. No model download,
  server restart, model-management API call, or embedding-selection change.
- Added the exact registry profile, updated native LM and `YawConfig` defaults
  plus `yaw.toml`, and reconciled README/FOUNDATIONAL/Epic 1 runtime selection.
  Historical Qwen benchmark targets and explicit Qwen profile tests remain.
  Role-aware GBNF flags remain enabled for direct/worker calls and disabled for
  RLM code generation. Existing temperatures are retained; the Gemma profile
  has no Qwen-specific frequency penalty.
- **Live verification:** `uv run --locked python outputs/verify_gemma4.py`
  -> **passed in 25.23 s**. Native OpenIE extracted Ada Lovelace, Charles Babbage
  and Analytical Engine with a grounded triple. Direct constrained QA answered
  **Ada Lovelace**. A bounded RLM reflection run invoked its GBNF worker and
  returned an instruction preserving `named_entities`. Reflection telemetry
  reports **3,101 tokens** under `openai/gemma-4-E4B-it-qat-q4_0-gguf-Q4_0`, including the
  worker call. Evidence: `outputs/gemma4-migration-smoke.json`.
- Profile/chat/core-config tests -> **19 passed, 3 subtests passed in 11.80 s**.
  Runtime config/resume-parser selection -> **2 passed, 25 deselected in 14.28 s**.
  The broad selection also deselected architecture tests; those were run
  separately: **9 passed in 14.52 s**. `git diff --check` passes.
  New-source lint comparison against HEAD reports **zero new
  diagnostics**; preexisting touched-file lint diagnostics remain.
- Initial disposable smoke-script attempts used the wrong `load_config` argument
  shape and failed before inference; corrected to its actual path plus
  `workspace_override` interface before the successful live run.
- **Limits:** live smoke validates the configured backend and inference paths,
  not GEPA convergence, benchmark parity, sustained memory use, or performance
  relative to Qwen. Model changes intentionally use distinct cache/run identities;
  this does not migrate Qwen optimization checkpoints into Gemma runs.

## 30. SIMBA task reward and durable worker recovery (2026-10-23)

- **Implemented:** replaced instruction-form scoring with a paired task-score
  improvement reward. Each proposal is applied to its complete GEPA candidate
  and evaluated against that candidate on the same first two training queries
  (one for a single-query trainset). This uses the existing GEPA composite
  retrieval/EM/F1/token/graph metric without adding validation-history records.
  Meta-example candidate snapshots are labels, excluded from worker inputs.
- **Persistence:** native DSPy predictor state for proposer and compressor,
  demos, bounded training examples, and step/seed counter are atomically stored
  in `yaw.db` metadata before compilation and immediately after each completed
  step. The GEPA seed-resume path restores the worker with current LM bindings.
  Incompatible model/task/data/retrieval/code identities reject restoration;
  task-evaluation failures retain the previous worker and warn; storage errors
  propagate. Task scores use the bounded SQLite candidate cache.
- **Final focused verification:**
  `uv run --locked pytest tests/daemon/graph/test_simba_learning.py -q`
  -> **7 passed in 40.23 s**, with two native DSPy warnings that rollout IDs do
  not change sampling at temperature zero. Real native SIMBA, real loopback
  HTTP, and real SQLite verify helpful versus harmful task rewards, correct
  task-model routing, two-query bounds, worker learning, fresh-process state
  restoration and continued proposal generation, zero extra task requests on
  cached rescoring after restart, incompatible-label rejection, task failure
  non-adoption, and preservation of the previous snapshot on a read-only write
  failure. The HTTP provider is scripted: this establishes orchestration and
  metric discrimination, not genuine model quality or benchmark parity.
- The updated standalone-proposer contract plus the initial six SIMBA tests
  passed together: **7 passed, 32 deselected in 44.56 s**. Architecture and
  bounded-diagnostic checks passed with the strengthened restart test:
  **17 passed in 42.56 s** (six SIMBA tests plus eleven architecture/diagnostic
  tests; the additional SQLite-write-failure case was then verified above).
- **Broader integration limitation:** the existing native MIPROv2/GEPA hybrid
  test completed optimization and saved its program, then failed its existing
  `len(compiled.shacl_healer_rlm.generate_action.demos) >= 1` assertion under
  zero-demo settings, as previously recorded in sections 22/24. Combined run:
  **11 passed, 1 failed, 21 deselected in 64.44 s**. Scripted provider parse
  errors also occurred during that test's MIPRO search. This is not reported
  as a passing end-to-end optimization acceptance run.
- New test-file Ruff check and `git diff --check` pass. Comparison of runtime
  source diagnostics against HEAD reports **zero new lint diagnostics**;
  the broad touched-file Ruff check still finds existing lint debt.
- **Limits:** no live-server training, full-corpus reindex, convergence study,
  or long-running memory soak was performed. Inner task evaluation costs are
  additional to GEPA's metric-call budget on cache misses. The small fixed
  calibration slice can overfit; outer GEPA validation still decides candidate
  acceptance. Recovery resumes completed SIMBA steps and their training window,
  not an in-flight native SIMBA compile's temporary candidates/trial logs.
  Legacy checkpoints without SIMBA state start a fresh worker.

## 31. Hybrid zero-demo assertion and saved-program regression (2026-10-23)

- Corrected the hybrid integration test's contradictory requirement for at
  least one SHACL demo. It now explicitly requests zero bootstrapped/labeled
  task demos and verifies every task predictor is demo-free both immediately
  after optimization and after reloading the saved program.
- The stronger check exposed a real loader defect: `HippoRAGProgram.load_state`
  reinserted `QA_DEMOS` whenever the restored QA demo list was empty. Removed
  that reseeding. Explicitly saved empty/default/custom demo lists now survive
  JSON round trips. Missing legacy predictor state still inherits the chosen
  constructor defaults through the existing missing-state handling.
- First corrected-test run failed on restored `qa.predict` retaining built-in
  demos (**1 failed in 61.00 s**). After fixing the loader,
  `uv run --locked pytest tests/daemon/graph/test_engine.py::test_evolve_prompts_real_unmocked_gepa_and_miprov2_with_context_limit_guard -q --tb=short`
  -> **1 passed in 69.71 s**. This supersedes the failed assertion reported in
  section 30; it also corrects the initial assessment that only the test needed
  changing.
- `uv run --locked pytest tests/daemon/graph/inference/test_program_state.py -q --tb=short`
  -> **5 passed in 13.94 s**. Cases cover saved empty/default/custom QA demos
  and missing legacy QA state with constructor demos enabled and disabled.
  The new file passes Ruff; baseline comparisons for changed runtime code and
  the existing integration-test file report zero new diagnostics.
  `git diff --check` passes.
- Existing retriever/program-state and Pareto-ensemble contracts plus the
  standalone proposer and SIMBA recovery tests:
  `uv run --locked pytest tests/daemon/graph/inference/test_prompts.py tests/daemon/graph/test_simba_learning.py -k 'yaw_rm_and_yaw_retriever_contract or pareto_program_ensemble or simba or task_reward' -q --tb=short`
  -> **10 passed, 30 deselected in 58.30 s**, with the same two native DSPy
  temperature-zero rollout warnings documented in section 30.
- These checks use real native DSPy optimizers and JSON persistence with a
  scripted inference provider. They verify orchestration/state behavior, not
  live-model quality, benchmark parity, or corpus-level gains.

## 32. Active GEPA model/context and watchdog audit (2026-10-23)

- Findings and implementation details:
  [DSPy practical improvements](dspy-improvements.md). `yaw.toml` overrode the QAT
  core default with the standard embedding model, and all 250 sampled recent
  embedding audit entries named that model. TOML and adapter defaults now
  share the exact QAT alias; explicit standard selection remains supported.
  The active optimizer was not switched between embedding identities.
- Read-only Lemonade health confirmed Gemma's launched serving context is
  **65,536**, distinct from its advertised architectural maximum of 131,072.
  Added explicit context/output/RLM budgets, scaled character safeguards,
  runtime forwarding of graph context capacity, current deployment overrides
  for saved ensembles, and compatible cache identities. Character safeguards
  are heuristics, not token-exact prompt fitting. Existing process settings are
  not hot-reloaded; changed inference/index identities require a fresh run.
- Dashboard: both relation types seed one hop; only the largest induced
  component is rendered. The two assertion lists follow Fig. 2, including ten
  recent active related pairs ordered by insertion rowid. Relaxation increases
  from 40/80 rounds to **1,000**, configurable with `--layout-iterations`.
  Labels cover ten recent disambiguation pairs in view. Actual full-validation
  scores replace misleading comparisons of mixed candidate F1 averages.
- **Focused checks:**
  `uv run --locked pytest tests/scripts/test_watchdog.py tests/daemon/graph/inference/test_inference_budgets.py tests/core/test_config.py tests/daemon/graph/inference/test_chat.py tests/daemon/graph/inference/test_embedding.py tests/daemon/graph/inference/test_program_state.py tests/core/test_architecture.py -q --tb=short`
  -> **37 passed in 27.86 s**. Real loopback HTTP verifies exact standard/QAT
  embedding requests and a long passage tail retained at 64K but truncated at
  32K. SQLite tests verify newest ten active canonical assertions, and rendered
  file tests verify component selection/report order. Other cases cover input
  budgeting, saved deployment overrides, validation, persistence and boundaries.
- Final context/ensemble/dashboard selection:
  `uv run --locked pytest tests/daemon/graph/inference/test_inference_budgets.py tests/scripts/test_watchdog.py tests/daemon/graph/inference/test_prompts.py -k 'inference_budgets or watchdog or pareto_program_ensemble' -q --tb=short`
  -> **9 passed, 32 deselected in 19.47 s**. One upstream Pydantic warning
  concerns a TypedDict ReadOnly qualifier, not a failed assertion.
- Dashboard/runtime selection -> **7 passed, 15 deselected in 28.19 s**.
  Native SIMBA recovery and bounded diagnostics -> **9 passed in 43.42 s**,
  with the previously documented two DSPy temperature-zero rollout warnings.
  Initial new embedding test used the nonexistent `embed_texts` interface
  (**1 failed, 22 passed**); corrected to the actual `batch_encode` API.
  Initial runtime selection expected the old repository embedding default
  (**1 failed, 2 passed**); updated that expectation to the selected QAT alias.
- New tests and watchdog pass Ruff. Baseline comparison reports **zero new
  runtime lint diagnostics**; `git diff --check` passes. Existing lint debt in
  touched runtime files remains.
- A read-only workspace preview generated all five figures/report in **60.47 s**
  with 1,000 layout rounds and 2,228 selected nodes. Reloaded only the verified
  watchdog process tree using the existing 200-second watch interval. The
  optimizer and all three model-server identities were verified unchanged
  across that reload. Inspected the published PNG (2,238 selected entities)
  and report: caption is readable, both lists follow Fig. 2, and validation
  comparison uses the seed score **0.4759** versus candidate 4 **0.4525**.
- **Later observed shutdown:** at 06:45:36 UTC the watchdog reported the
  optimizer stopped with 309 recorded evaluations. PID 82496 was absent;
  its status file retained `stopping`, `last_error=null`. Tests inspected for
  shutdown interaction use isolated temporary configs/databases and local
  runtime cleanup. The available evidence does not identify why the original
  process exited. It was not restarted, and its checkpoint was preserved.
- No new live-model generation, model loading/unloading, corpus rebuild or
  GEPA restart was performed. These results validate routing, budgets,
  persistence and dashboard behavior, not 64K live-model capacity under load
  or new corpus-level benchmark gains.

## 33. GEPA reflection signal inspection (2026-10-23)

- Read-only follow-up on the user's observation that MIPRO produces useful
  early candidates faster than recent GEPA runs. No optimizer/prompt changes
  or live inference were performed for this inspection.
- Both optimizers use YAW's `RLMInstructionProposer`: MIPRO's instruction
  proposer delegates to the same per-component reflection method. The inputs
  differ: MIPRO builds candidate banks from successful demos or dataset labels;
  GEPA supplies failure traces and diagnostic feedback. This is not a
  comparison of wholly separate generation models/templates.
- GEPA feedback preparation replaces expected answers and gold answers with
  `<TargetAnswer>` before reflection. A local no-inference probe confirmed
  `Predicted=North != Expected=['South']` becomes
  `Predicted=North != Expected=['<TargetAnswer>']`. This removes a concrete
  training-label contrast as well as discouraging instance copying. Whether
  that transformation reduces real task quality needs controlled comparison.
- The reflection worker accepts but ignores `component_name`; the signatures
  expose only current instruction and examples. Per-component task contracts
  are therefore implicit in the current instruction/trace. After any synthesis
  tool call, the outer proposer selects the last worker draft rather than the
  RLM's returned text. That worker-authoritative design can be intentional;
  its effect depends on whether the tool receives useful distilled diagnoses.
  No saved trace evidence establishes it as the cause of the stalled scores.
- A representative synthetic metric feedback plus stage diagnostic measured
  **556 characters**, with the stage hint at character **464**; its stage hint
  survived the old 600-character worker cap. This does not support claiming
  all stage diagnostics were truncated. Longer traces remain bounded by the
  configured character policy.
- Saved native `candidates.json` contained two accepted programs; the second
  changed `synonym_judge` relative to the seed. The six engine candidate records
  include scored/rejected proposals, not six accepted native candidates.
  Full-validation scores remain the previously recorded 0.4759 versus 0.4525.
  Exact rejected reflection inputs/tool trajectories are not recoverable from
  these summary JSON artifacts, so model-prompt quality is not established.
- A useful next comparison is a bounded, paired replay using identical training
  examples, task-demo policy, task metric and generation budget: native DSPy
  GEPA proposer versus the custom direct worker versus RLM distillation. Preserve
  expected training outputs for diagnosis while checking generated instructions
  for instance copying. Compare actual downstream scores and inference cost,
  not instruction wording alone. MIPRO's observed early gains remain a reason
  to retain a reliable warmup, subject to the handoff/stability fixes already
  documented in the MIPRO audit.
- `git diff --check` passes. This is diagnostic evidence and an experiment
  design, not a new convergence or benchmark result.


## 34. Evidence-driven native GEPA reflection replacement (2026-10-23)

Implemented the accepted replacement: original training traces stay in the RLM
sandbox; the selector returns validated example/field/span references and a short
diagnosis; DSPy's native `InstructionProposer` is the sole instruction writer.
Invalid selections and provider failures use deterministic evidence selection.
Task signatures retain their output schemas. The native compressor uses its real
`shortened_instruction` output field. Existing user task prompts/demos remain.

- `reflection_input_tokens=16384` by default. Final rendered messages, including
  learned demos/schema, use the resident llama.cpp model's chat template/tokenizer.
  Requests reserve their actual output allowance plus 512 context tokens. Optional
  evidence is removed whole; mandatory instructions/contracts are never sliced.
  Each reflection role verifies its own resident model/tokenizer. Serving weights,
  templates and generation defaults participate in reflection response/stage keys.
- Evidence, proposals and request phases use existing SQLite storage. The active
  request is pinned separately from FIFO cache eviction and committed atomically
  with learned state. Corpus text/excerpts hydrate by source identity/span.
  Completed stages replay without generation; interrupted in-flight work may run
  again. SIMBA uses task-improvement rewards and version-2 learned state, with
  request-local restoration preventing repeated application of a completed update.
  Legacy worker snapshots are incompatible and rejected; they are not migrated.
- LM histories are disabled, reflection traces are isolated, learning examples and
  diagnostic histories are bounded, and sandbox lifetimes end with each attempt.
  SIMBA native resampling temperatures now reach the worker; already compacted
  evidence is not silently clipped by SIMBA's demonstration-field default.

### Verification

- Focused suite: **41 passed**, 16 warnings, 245.23 s, covering reflection,
  SIMBA, candidate caches, architecture, inference budgets and program state.
  Actual native DSPy classes, Deno/Pyodide sandbox, SQLite, loopback HTTP and fresh
  processes are used. Loopback tokenizer counts validate the protocol/guard,
  not the accuracy or task quality of a real model.
- Fresh-process tests verify zero additional generation for completed proposals,
  including interruption immediately after a proposal becomes durable. Real
  cache eviction preserves the pinned active request and learned worker state;
  failed SQLite writes preserve the preceding snapshot. Changed training labels
  invalidate restoration. Native SIMBA restores learned predictions/task rewards.
- Four affected engine optimizer tests passed earlier (78.14 s), including native
  MIPROv2/GEPA orchestration. This does not change MIPRO warmup/search policy.
- After the final tokenizer-role and retention changes, reflection/SIMBA checks
  passed **23 tests** (103.09 s), and all **4 affected engine optimizer tests**
  passed again (73.27 s). Repeated native proposals release original input
  containers, leave outer traces empty and retain no LM/predictor histories.
  Missing selector-model tokenizer access fails before generation.
- The final reflection suite passed **17 tests** (74.32 s), including real SQLite
  replay invalidation: changing learned writer instructions runs the new writer
  exactly once while reusing evidence; subsequent identical requests reuse it.
  Default LM chat-template options also count toward the complete request budget.
- Final baseline-relative lint comparison found **no new diagnostics**;
  `git diff --check` passed. Existing unrelated lint debt remains. The last lint
  comparison used the already tested installed environment because concurrent
  dependency edits left the manifest and lock out of sync: `uv run --locked`
  reported that `uv.lock` needs updating. The manifest now requests
  `cocoindex>=1.0.26`, while the tested environment used 1.0.25. Those dependency
  edits and the lock were preserved. Locked launch/new-dependency compatibility
  remains to be reconciled by the dependency work; no packages were changed by
  this reflection replacement.
- A broader prompts/budget/state run exposed the existing unrelated
  `test_sprint2_validate_stage_invariants_and_refine_retry` assertion: differently
  defined `New York` / `new york` entries score 1.0 while that assertion expects
  less than 1.0. That run had 53 passes, one failure and two deselections. It is
  outside the reflection replacement; the full test suite is not claimed green.
- One verification invocation used an incorrect architecture-test path and
  collected no tests; the corrected invocation is the 41-pass run above. An
  intermediate fresh-process run overlapped source edits and failed compatibility;
  with source frozen, the isolated recovery test passed (34.66 s). Final recovery
  checks also run with source frozen.

### Bounded live pilot

A serial EnterpriseRAG pilot used gemma-4-E4B-it-qat-q4_0-gguf-Q4_0, its actual serving tokenizer
and a verified 65,536-token context on native Windows / Python 3.14.7, with DSPy
3.4.1. Two training queries (dataset indices 0–1) supplied reflection and SIMBA
calibration; twelve disjoint held-out queries (indices 2–13) were reserved for
comparison. No held-out answers entered learning. All three components (NER,
synonym judging and QA) produced proposals; two SIMBA updates completed. Synonym
selection used the deterministic fallback after an invalid RLM selection.

The first attempt was stopped after discovering an ambiguous generalization rule:
the model interpreted “do not copy instance facts into new instructions” as a ban
on names in extraction outputs. The native writer policy now explicitly requires
exact source names/identifiers in task outputs while keeping instruction text
general. The first attempt's report is preserved separately.

The corrected attempt stopped at its **60-minute deadline** (3,601.16 s), with
**6 of 12 paired comparisons completed**. The unpaired seventh baseline is excluded.
It recorded 524 logical LM invocations and 93 DSPy cache hits; these are not exact
physical inference-server request counts. Peak observed Python working set was
**423.64 MiB**; model-server, Deno and total system memory/VRAM were not measured.

| Completed-pair mean | Starting program | Proposed program |
| --- | ---: | ---: |
| Primary-reference F1 used during pilot | 0.3304 | 0.3349 |
| Full-reference F1, rescored identically after pilot | 0.4211 | 0.4305 |
| EM, either answer policy | 0 | 0 |
| Recall@1 / @2 / @5, 5 applicable queries | 0.6622 / 0.7244 / 0.9111 | 0.6622 / 0.7244 / 0.9111 |
| Logical task tokens per query | 21,975.5 | 22,806.2 |
| Task score under original pilot metric | 0.4460 | 0.4475 |
| Seconds per query | 211.08 | 182.23 |

Full-reference rescoring adds the dataset's `answer_aliases` and `answer_facts`,
as production answer extraction does. The original pilot/calibration used only
the primary answer; rescoring does not retroactively change learning. Recall
excludes the query without supporting contexts. Candidate runs followed baselines,
so cache/order effects confound latency. Identical starting task demos were retained
in both arms; this is not a complete GEPA zero-demo corpus run.

Per-query full-reference F1 (dataset index): 2, 0.4348→0.4348; 3, 0→0;
4, **0.8298→0.8172 regression**; 5, 0.5000→0.5000;
6, **0.2621→0.3312 improvement**; 7, 0.5000→0.5000.
Raw answers, component instructions and call diagnostics remain in
`outputs/gepa_reflection_pilot_v2/report.json`; paired rescoring is in
`outputs/gepa_reflection_pilot_v2/summary.json`, and the process observations are
in `outputs/gepa_reflection_pilot_v2/peak-memory.json`.

The live pilot exercised an in-progress replacement before the final SIMBA
resampling, trace isolation and deployment-key fixes. It therefore establishes
neither quality gain for the final source nor unattended stability. The small
partial score increase is inconclusive; both improvement and regression are
reported. The twelve-query target was not achieved within the authorized deadline.
No further live pilot, model loading/unloading or unattended optimizer was launched.

## 44. GEPA quality-only geometric reward (2026-10-24)

Accepted reward change: the scalar objective is the unweighted geometric mean
of labeled R@1, R@2, R@5, R@10, R@20, EM and token F1. Token, repair, latency
and SHACL penalties are removed, together with efficiency/integrity Pareto
objectives. Tokens and integrity remain diagnostics. There is no smoothing or
extra weighting; a zero component yields zero. QA-only examples no longer
substitute EM for unmeasured recall. Unlabeled examples fail explicitly.
Missing token telemetry no longer prevents quality evaluation.

QA now uses native DSPy normalized exact match and continuous best-reference
token F1. The previous substring-as-exact-match rule and F1 promotion are
removed. Installed `answer_exact_match(frac < 1)` was inspected: it thresholds
F1 to a boolean, so continuous F1 comes from `dspy.evaluate.metrics.F1`.
`SemanticF1` is an LM judge and remains an optional evaluation diagnostic.
Reflection's deterministic fallback recognizes the new feedback score format.
Run and SIMBA task-cache identities record `task-quality-geometric-mean-v1`;
old scored optimization state must not resume into this objective. Existing
processes keep their loaded implementation until restarted with fresh state.

Verification used actual DSPy metric/GEPA classes, real SQLite checkpoints and
loopback task providers. **126 passed in 122.54 seconds** in the focused metric,
grammar, parser and prompt packet. The broader reflection/SIMBA/engine/benchmark
packet initially had **104 passed and 1 failed in 160.34 seconds**: an old
cache-isolation test expected positive smoothed reward for a missed answer.
It now asserts zero composite, zero F1 and preserved R@20=1 diagnostics.
The corrected case plus real GEPA resume, legacy checkpoint rejection and
architecture checks passed **12 cases in 43.13 seconds**. SIMBA's partial-answer
baseline assertion was also updated for zero EM, retaining a real correct-to-wrong
regression check and zero repeated task calls for cached evaluations; the full
SIMBA packet passed in the broader run. Real GEPA runtime/persistence cases for
combined, QA-only and retrieval-only labels passed **3 cases in 28.74 seconds**,
verifying the serialized scores against the formula. Ruff, ty and
`git diff --check` pass. Existing DSPy message-API deprecation, temperature/rollout
and Pydantic metadata warnings remain visible.

These tests establish metric and orchestration behavior. They do not establish
better held-out model quality. No unattended optimization run was launched.
This packet does not update or complete the separate all-production coverage goal.

## 45. DSPy Signature, History and predictor construction audit (2026-10-24)

Read-only source audit and runtime probes against installed DSPy **3.4.1**.
Signatures are wrapped in native Predict/ChainOfThought modules; demonstrations
are native Example objects attached through predictor.demos. QA's generated
reasoning/answer signature and RLM's generated REPLHistory inputs are expected
DSPy behavior. DiagnosticHistory and LM.history are observational records, not
conversation inputs. Current QA signatures and forward APIs do not accept a
conversation history. A native History field rendered through YawGBNFAdapter
correctly produced system/user/assistant/user messages, including the prior answer.
Settings are read at call time; constructing predictors before configure/context
does not by itself invalidate them.

Confirmed defects in disposable probes, with no resident inference calls:

- A real loopback HTTP request through ner.lm.copy(temperature=0.9,
  max_tokens=123) sent temperature **0.1** and max_completion_tokens **77**,
  the original target settings. _PredictorBoundLM.copy stores overrides on the
  proxy, but its __call__ forwards neither those kwargs nor a copied target.
- Native program.ner.acall raised **NotImplementedError: Subclasses must
  implement this method** before sending an HTTP request. The proxy inherits
  BaseLM.acall without implementing its async execution contract. The separate
  program.as_async helper uses asyncify over synchronous execution.
- YawJSONAdapter.parse accepted {"reasoning": "Only reasoning"} for the
  ChainOfThought QA signature, returning no answer. Native JSONAdapter rejected
  the same payload with AdapterParseError. The repair fallback returns the
  first matching output field without validating all required outputs.

Additional integration risks: the LM proxy prefers ambient settings.lm over
its bound target, unlike native Predict's explicit-binding precedence. A context
probe resolved a bound predictor to the ambient model. Dedicated proxy identities
also make HippoRAGProgram.get_lm report multiple LMs even when all roles use one
underlying LM. These behaviors need explicit design decisions when replacing the
proxy. DSPy 3.4 emits legacy LM/message deprecations; its documented 3.5 cutoff
requires migrating custom integration to the native engine/request contract.
OpenIE output annotations remain list[Any], leaving structural guarantees to
GBNF and host parsing rather than the signature type itself.

A fresh default program contains 1 NER, 1 triples, 10 triple-filter and 2 QA
demonstrations. GEPA-only optimization retains seed demos; the zero-demo clearing
currently occurs inside the MIPRO branch. This is distinct from conversation
history and should be made explicit in any intended zero-shot policy.

Verification: **4 passed in 22.29 seconds** for the existing signature/demo,
canonical predictor, adapter and asyncify/embedding checks. Additional construction,
message rendering, parser and isolated HTTP probes produced the results above;
git diff --check passed. Those passing tests do not cover the confirmed defects.
No production source was changed and no optimization run was launched.

## 46. Native DSPy predictor binding and parsing repair (2026-10-24)

The custom per-predictor BaseLM proxy is removed. Task predictors now bind
native LMs by role; an unbound predictor follows DSPy's context. Explicit
per-call LM overrides are forwarded to Predict, including query-intent calls.
Native LM copies preserve their generation settings through candidate copies.
RLM explicit overrides operate on a copied module instead of mutating shared
predictors. The native instruction writer and compressor remain unbound so
their selected worker/SIMBA rollout context and verified token guard apply.
Predictor enumeration is inherited directly from DSPy.

The additional YAW JSON recovery adapter and legacy ReAct field-renaming helper
are removed. YawGBNFAdapter supplies grammar only and inherits native formatting
and parsing. DSPy's JSONAdapter already repairs JSON syntax and validates
declared output fields; YAW no longer guesses field names, accepts partial QA
objects or fabricates empty success results. This was legacy response-format
compatibility, not a GBNF parser. No reproduced server-side GBNF defect justified
that extra layer. Invalid signatures now expose invalid scripted fixtures too:
some fixtures answered query-intent requests with QA fields or confused an
extraction input field with the required output. Those fixtures now produce the
actual native schemas, including ReAct tool calls.

Fresh task and reflection programs have zero demonstrations by default. The
existing user-edited examples are retained unchanged for explicit opt-in, with
no claim of measured generalization. Native compiled demos remain intact on
load, including deliberately empty lists. Explicit deployment LMs are retained
before rebuilding Pareto members; missing legacy predictors get configured
defaults without importing serialized custom clients. Native LM identities
remain in candidate-cache keys while credentials are excluded. No implicit
conversational memory was added; History remains an explicit signature input.

Regression tests use native DSPy classes, real HTTP requests and JSON state files
for sync/async execution, copied parameters, role/context/explicit-call
precedence, candidate cloning, state restoration, zero-shot defaults and history
formatting. Malformed output cases exercise DSPy's own validator. Initial
targeted checks passed **16 cases in 22.51 seconds**; the complete prompt suite,
including resident-model GBNF and RLM smoke checks, passed **33 cases in 111.05
seconds**. A broader run passed **145 cases and failed 13 in 576.09 seconds**;
the failures identified legacy engine fixture schemas and an obsolete proxy
identity assertion in the cache-retention test, plus fixtures attempting to
reconstruct their custom test clients instead of supplying deployment LMs.

The corrected serial engine/native-binding/cache packet passed **43 cases and
failed 1 in 314.22 seconds**; the last failure was the custom test-LM restoration
case, corrected by explicitly supplying its deployment client. The final packet
passed **36 cases in 66.47 seconds**: all new native binding/parser/history
regressions, that corrected frontier export/load case, real GEPA runtime
optimization/persistence for QA-only, retrieval-only and combined labels, native
SIMBA learning/process restart and architecture checks. All failures encountered
in the affected packets were corrected and reverified. This is targeted
verification, not a fresh complete repository suite or a model-quality benchmark.

The 600-query disk-cache retention test passed with **one physical provider
request**, **18,000 logically accounted tokens**, no retained LM/global histories
and less than **1,000,000 bytes** of retained allocation growth after warmup.
Ruff, ty and git diff --check pass. Existing DSPy 3.4 message-API and
temperature/rollout warnings remain visible. No user processes were stopped and
no unattended optimization run was launched. The separate all-production
coverage goal is unchanged.

## 47. Gemma 128K capacity and automatic budgets (2026-10-24)

The configured and Python/model-profile Gemma default is **131,072 tokens**.
The user reports approximately **65 TPS without flash attention**; this packet
records that deployment evidence and does not remeasure throughput or modify
the external inference service. Embedding model limits remain independent.

Unpinned completion, reflection-input and RLM-output budgets scale from the
64K deployment baseline. At 128K they resolve to **8,192 output tokens**,
**32,768 reflection input tokens** and **20,000 RLM output characters**. The
repository TOML no longer pins those budgets at their old 64K values. Explicit
overrides remain authoritative; changing the context in TOML recomputes omitted
budgets on load. Standalone programs infer capacity from supplied LM metadata
or use the same 128K default. Existing smaller model profiles retain their
previous completion default unless an explicit deployment context is supplied.

Extraction, QA and graph hydration use available-input-scaled character limits;
the default graph context budget is **102,857 characters** at 128K/8K output.
SHACL focus context and metric hydration no longer have separate fixed small
caps. Synonym/SHACL prompt-size batch caps scale from 16 items at 64K to 32 at
128K; structured lists are fitted as whole JSON input items, never split mid-item.
Default native proposal and diagnosis length limits also scale. Explicit
unbounded proposal length is passed through to its grammar without a hidden
1,500-character fallback. These are input-budget changes, not a revival of the
removed custom response recovery parser.

The verified serving tokenizer still measures complete reflection requests and
reserves output plus a fixed **512-token safety margin**. RLM iteration/call
counts, retry limits, retrieval parameters and diagnostic retention stay
independent. Prompt-state loading, including hot reload and Pareto restoration,
cannot replace the deployment's serving context/output caps with stale values.
Changed settings/code retain the existing optimization-resume compatibility
checks; a running process retains its startup settings.

Verification: **73 passed in 93.18 seconds** across initial budget, native
binding, tokenizer/reflection and configuration tests. Extended real-HTTP and
SQLite graph/tokenizer/architecture checks passed **93 cases in 182.44 seconds**.
Native GEPA/SIMBA learning and persistence plus capacity checks passed **27 cases
in 68.12 seconds**. The final hot-reload/legacy-Pareto/real-GEPA and architecture
packet passed **28 cases in 41.09 seconds**. Tests show source text beyond the
old 64K safeguards reaching 128K requests, automatic and pinned settings,
context-sized structured batches, exact loopback tokenizer rejection before
generation, and preserved configured capacity on prompt reload. Ruff, ty and
git diff --check pass. Existing DSPy/Pydantic warnings remain visible. No
unattended optimization or resident-model performance benchmark was launched.

### Watchdog individual evaluation scores (2026-10-24)

Fig. 3 now plots individual evaluation scores in evaluation order, with the
current all-evaluation mean for each candidate overlaid. The latest evaluation
number, candidate, score and mean appear above the panel. Existing candidate
mean/min/max, validation/best-program and mean-latency panels remain available.
Individual points use the optimizer's existing bounded 500-record audit tail;
candidate aggregates still cover all recorded evaluations. Audits without
individual scores show an explicit unavailable message. The optimizer already
writes the audit after each completed evaluation, so a dashboard refresh can
show progress within a candidate without optimizer changes. An existing
watchdog process must restart to load the new renderer.

Verification: all **15 focused watchdog tests passed in 10.57 seconds**, including
real audit-file updates and SVG rendering as another evaluation completes on
the same candidate. Ruff checks/formatting, focused ty checks and
`git diff --check` passed. A three-candidate PNG preview was visually inspected.
Initial sandbox checks could not access uv's cache/Python environment; the
authorized rerun succeeded. No inference or optimization run was needed.

### Native GEPA progress in watchdog Fig. 5 (2026-10-24)

Replaced the dashboard-derived quality/latency frontier and its misleading
archive-size trend. Fig. 5 now uses a versioned `gepa_progress` snapshot in the
existing optimization audit. GEPA callbacks publish validation scores, scored
example counts and validation-set sizes, exact native frontier membership,
bounded addition/displacement history, iteration and execution phase. Audit
publication happens on callbacks as well as individual metric evaluations, so
reflection and validation completion are visible without waiting for another
candidate evaluation. GEPA's native CompositeCallback wraps optional observers.

Full-validation scores have a seed baseline and best-so-far line. Partial
validation results are marked separately and excluded from that line. Scores
have no smoothing floor. Native candidate indices start at zero and differ
from the dashboard's first-seen fingerprint numbering. Frontier membership
comes from `on_pareto_front_updated` and `get_pareto_front_mapping()`, including
the configured hybrid frontier. A restored native checkpoint supplies all
known candidate score/count summaries and current membership without requiring
new inference. The first frontier observation is a baseline, not an addition;
the latest 250 observations in the current process are retained. Programs,
outputs and traces are never held by this observer. Completed validation
coverage includes cached results; in-flight counters explicitly report metric
calls observed during the phase, which cache reuse may bypass.

Older audits explicitly report native telemetry unavailable. Restarting the
watchdog loads the renderer; restarting/resuming the optimizer with this code
enables native publication. No running optimizer was stopped or relaunched.

Verification: the final focused packet passed **30 tests in 37.57 seconds**,
covering a real deterministic GEPA optimization loop, native hybrid frontier
displacement, real checkpoint save/load, YAW CLI resume with unchanged native
evaluation counts and exact published frontier, real audit/SVG updates,
partial-validation rendering, legacy placeholders, architectural containment
and hybrid metric integration. Earlier test-adapter/interface failures were
corrected before the final run. Ruff and focused ty checks passed; a PNG preview
was visually inspected and `git diff --check` passed. The native GEPA test is an
offline arithmetic task: it verifies monitoring/orchestration, not model quality.

### SSOT/DRY canonicalization and legacy purge (2026-10-24)

Completed full-codebase SSOT/DRY canonicalization and legacy symbol retirement across `yaw/`, `benchmark.py`, `scripts/`, `tests/`, and `docs/`:

- **Retired legacy symbols, dead wrappers, and duplicate helpers**: Removed legacy GBNF synonym-judgment grammar helpers, dead assertion/refine wrappers, unused entity-node/fact-flattening helpers, batch-overflow repackers, mutable delta-compiler setter (replaced with read-only `CorpusIngestor.on_delta` property), candidate-list partitioning and the residual `16`-candidate cap in `YawGraphEngine` synonym discovery/adjudication (so all unadjudicated candidate pairs with `dense_sim >= synonymy_threshold` emit `skos:related` fallback edges), dead model-recycling helper, unused `--index-batch-size` / `--qa-batch-size` flags in `benchmark.py`, 11 private staticmethod bindings on `YawGraphEngine`, duplicate `corpus/{cid}.md` materialization (consolidated into `YawGraphEngine._materialize_doc_chunk`), dead `HippoRAGProgram` repair aliases, duplicate `DEFAULT_CHAT` / `DEFAULT_BASE_URL` constants in `models.py` and `embedding.py`, and hardcoded user paths (`YAW_ARTIFACT_DIR` env var or workspace default used instead).
- **Task quality & benchmark integrity**: Consolidated `compute_task_quality_score` in `parsers.py`. Added `check_benchmark_integrity(records, *, k_values=...)` in `parsers.py` and wired it into `benchmark.py` and `YawRuntime.benchmark` (`report["integrity"] = {"passed": ..., "errors": ...}`).
- **Entity definition canonicalization**: Unified `extract_entity_definitions(raw_input, *, normalize_keys=False)` in `parsers.py` as the single parser for list/dict/tuple inputs and delegated `HippoRAGProgram._merge_openie_definitions` to `extract_entity_definitions(..., normalize_keys=True)`.
- **SHACL action SSOT**: Synchronized `ProposeSHACLRepairs` docstring with `VALID_SHACL_ACTIONS` and `HealSHACLViolations` (all 11 canonical actions).
- **Runtime & CLI canonicalization**: Prevented duplicate `_apply_delta` execution in `YawRuntime._execute_sync_worker` when `ingestor.on_delta` is bound, preserved pre-startup `stop()` signals across `start()`, drained queued `_WorkItem` futures on worker exit, renamed `handle_dspy_benchmark`, inlined single-use subparser parents, and enforced canonical CLI grammar (`yaw {daemon|graph|dspy|config} <action>`, `--db-path`, `--gate`) with `allow_abbrev=False` across all subparsers.

Verification: All 17 Oblivion Demolition Table `Test-Path` and repository-wide `git grep -w` checks returned **0 matches**. Offline real-substrate pytest suite (`uv run --locked pytest -o addopts="" -k "not live_silicon" tests/core/ tests/daemon/ tests/scripts/ tests/test_benchmark.py tests/test_benchmark_edges.py tests/test_download_datasets.py`) passed **610 cases with 0 failures** (`88%` total line coverage across `yaw/`). Repository-wide `ruff check`, `ty check`, and `git diff --check` passed with zero diagnostics. Live `:13305` model-server tests were intentionally not executed while an external optimization process held the local endpoint.

## 48. Step 10 & Step 11: External Pareto `dspy.Ensemble`, Dual-Perspective Reporting, and SKOS/RDF/OWL/PPR Hardening (2026-10-25)

### Step 10: Canonical External `dspy.Ensemble` & Dual-Perspective (`recall@N` + `QA (F1, EM)`) Reporting
- **Prediction final-field invariant & `retrieve_only` mode (`yaw/daemon/graph/inference/program.py`)**: Enforced `answer` as the final keyword argument on `HippoRAGProgram.forward` predictions (`signature=None`) so `dspy.majority` resolves `field = "answer"` by default, and added `retrieve_only: bool = False` to execute `self.retriever(question, k=k)` while skipping `self.answer_question` (`answer=""`).
- **Dual-perspective Pareto frontier selection & reduction (`yaw/daemon/graph/inference/optimizer.py`)**:
  - `select_pareto_frontier_programs` filters strictly non-dominated candidates from `compiled.detailed_results`, deduplicates by query-time predictors (`query_intent`, `triple_filter`, `qa.predict`), and assigns Slot 0 (`best_candidate`), Slot 1 (`recall@N` Pareto leader), and Slot 2 (`QA (F1, EM)` Pareto leader), falling back to `<stem>_telemetry/pareto_member_*.json` when loaded from disk.
  - `reduce_pareto_predictions` delegates directly to `dspy.majority(list(predictions))` when any prediction has a non-empty `answer`, and fuses retrieval rankings via `rrf_fuse_passage_lists` (`1 / (60 + r)` + min-max normalized PPR scores) when all predictions have `answer == ""` (`retrieve_only=True`).
  - `compile_pareto_ensemble` unconditionally returns `dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs)`.
  - `promote_compiled_program` and `sync_pareto_telemetry_siblings` atomically synchronize `<stem>_telemetry/pareto_member_*.json` and `pareto_frontier_manifest.json` across `evolve_prompts` and `yaw dspy load-program`.
  - `evaluate_program` evaluates `compile_pareto_ensemble(program)` and attaches `recall_at_k` (`1, 2, 5, 10, 20`) alongside `em`, `f1`, and `score`.
- **Read-only graph index & retrieval memoization (`yaw/daemon/graph/engine.py`, `yaw/daemon/runtime.py`)**: `YawGraphEngine._canonical_retrieval_hook` memoizes post-optimization query retrievals by `(question, k, query_retrieval_fingerprint(active_program))` while replaying extraction traces when `_in_optimization` or `item_docs` is active. `YawRuntime._run_retrieve`, `YawRuntime._run_qa`, and `YawRuntime.benchmark` route unconditionally through `engine.ensemble`.

### Step 11: SKOS / RDF / OWL / SHACL & Relation-Aware PPR Hardening (`H1`–`H4`)
- **`H1` Baseline Diagnostics Measured (Pre-Purge)**:
  | Database | Entities | Facts | Edges | Literal-Like Entity Rate | Singleton Entity Rate | Duplicate Directed Fact-Pair Rate | Semantic Edge Share | SHACL Violations (Pre-Heal) |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `.yaw/yaw.db` | 8,659 | 11,086 | 56,116 | 3.48% | 62.78% | 11.96% | 14.74% | 19,745 (legacy pre-`fact_sources` rows) |
  | `.yaw/yaw_optimize_20261024_053147.db` | 4,838 | 5,112 | 25,444 | 0.10% | 89.25% | 5.34% | 14.31% | 5 |
  - All `.yaw/*optimize*` remnants were subsequently purged, leaving only `coco_state`, `corpus`, `dspy_cache`, and `yaw.db`.
- **`H2` Literal-Aware Facts & `fact_sources` Provenance (`schema.py`, `graph_ops.py`, `shacl.py`, `parsers.py`, `engine.py`)**:
  - Extended `facts` with mutual-exclusion `CHECK` constraint (`obj_kind = 'entity'` vs `obj_kind = 'literal'`) and added `fact_sources(fact_id, chunk_id)` provenance table with automatic migration/backfill from `edges`.
  - Added deterministic literal object classifier (`xsd:dateTime`, `xsd:date`, `xsd:gYear`, `xsd:integer`, `xsd:decimal`, `xsd:boolean`, `xsd:anyURI`, `xsd:string`); literal objects skip `entities`, `entity_labels`, `entity_vec`, and synonym mining, and seed PPR via `subj_id` plus `fact_sources` chunks.
  - Updated `delete_chunks` to prune orphaned facts via `fact_sources` and `merge_entities` to deduplicate colliding entity/literal facts while merging their `fact_sources`.
  - Updated `lint_graph_shacl` (`FactTripleShape` + `FactProvenanceShape`) and `serialize_graph_turtle` to validate and emit typed/language-tagged RDF literals.
- **`H3` Directed `skos:broader` Hierarchy & Cycle-Safe SHACL Validation (`shacl.py`, `graph_ops.py`, `engine.py`, `signatures.py`, `invariants.py`)**:
  - Added directed `skos:broader` (`src_id = narrower`, `dst_id = broader`) to `edges`, `VALID_SYNONYM_RELATIONS` (5 relations), and `VALID_SHACL_ACTIONS` (14 actions including `assert_broader`, `remove_hierarchy_cycle`, `remove_conflicting_related`).
  - Added cycle-safe recursive CTE cycle detection (`ConceptHierarchyCycleShape`, SKOS S19/S23) and disjointness validation against `skos:related` (`ConceptDisjointRelationShape`, SKOS S27) and `owl:differentFrom` (`ConceptDisjointDifferentFromBroaderShape`) in `lint_graph_shacl`.
- **`H4` Relation-Aware Directed PPR (`yaw/core/types.py`, `yaw/daemon/graph/ppr.py`, `engine.py`)**:
  - Added `RelationTraversalPolicy` (`passage`, `fact_forward`, `fact_reverse`, `skos_related`, `skos_broader_up`, `skos_broader_down`, `parallel_aggregation`, `parallel_cap`) and updated `build_transition_matrix` with asymmetric `skos:broader` transitions (`up = 1.0`, `down = 0.5` default) while preserving exact `0.0` matrix diff on symmetric graphs when `policy=None` or default `RelationTraversalPolicy()`.

### Verification
- Full offline pytest suite (`uv run --locked pytest -m "not live_llm" -q`): **652 passed, 0 failed** (648 passed in initial batch + all 4 edge/contract regressions resolved and verified in `task-687` and `task-700`).
- Focused Step 10 & Step 11 suite (`test_step10_*` and `test_step11_*` across `test_prompts.py`, `test_db.py`, `test_ppr.py`, `test_engine.py`): **6 passed in 19.71s**.
- Static analysis & formatting (`uv run --locked ruff check`, `uv run --locked ty check`, `git diff --check`): **All checks passed (0 diagnostics)**.
