# Evaluation Record: Ingestion, CocoIndex & Daemon Runtime

Domain evaluation ledger for CocoIndex file/inference/graph targets, dataset bundle preparation, bounded overlap pilots, and Epic 2 runtime specification reconciliation. See the [Evaluation Index](evaluation.md) for all other domain ledgers.

## Revalidation after intervening development (2026-11-04, `06b35d2`)

The checkout is clean `dev` before this follow-up and now requires CocoIndex
`>=1.1.0`. Earlier lifecycle edits and the synonym fixture SQL correction are
already committed. Removed the remaining suppression of environment-close errors
in `_build_coco_app`: a failed teardown must abort app replacement.

Executed `uv run --locked pytest tests/daemon/graph/inference/test_synonym_packets.py
tests/daemon/graph/test_engine.py --lf --tb=short -q`: 27 passed, eight failed in
66.45 s. Failures include concept-scheme calls incompatible with test provider
contracts, additional definition telemetry, and two candidate rebuilds returning
scheme IDs from entity similarity search that have no entity metadata. These
tests exercise the old direct engine indexing path; they do not establish native
candidate indexing acceptance. They remain unresolved in this follow-up.

Focused runtime tests (`-k 'live or stop_keeps' --no-testmon`): four passed,
23 deselected in 13.82 s. Repository `ty check`, Ruff check with fixes, Ruff format
(158 files unchanged), and `git diff --check` pass. No inference pilot or GEPA
run was launched. The broader test gate remains open; no completion commit.

## Candidate cutover audit correction (2026-11-04)

The October 31 full-cutover claim below is not supported by current source.
Snapshot acquisition uses the native app, but `_canonical_retrieval_hook` still
calls `_ensure_candidate_chunks_indexed` during optimization. That method and
`_prepare_optimization_pools` retain manual eight-chunk loops; `index_chunks`
executes chunk embeddings before OpenIE and owns direct graph persistence.
Native daemon publication therefore coexists with the old candidate path.
The consumer-update acceptance gate is reopened. Replacing candidate preparation
and its writes through native components/targets remains required; removal of
`EmbeddingLane` alone does not establish that requirement.

## Runtime native live activation (2026-11-04)

YawRuntime now hosts a persistent native live update when `auto_poll=True` and
waits for native readiness. The existing bounded command worker transitions the
app to catch-up before queued operations and restarts live mode after success.
No filesystem polling loop or custom inference executor was introduced. Native
publication invalidates cached retrieval state and refreshes runtime statistics.
Reload callbacks enqueue their command without waiting inside a live component;
completion writes the external result file, avoiding a callback/drain deadlock.
Live failures signal shutdown and propagate through cleanup; resource close
exceptions are no longer suppressed. SQLite closes only after active work exits.

Full runtime module: 26 passed in 41.56 s. After the final cleanup edits, focused
live/reload, parser-failure and worker-drain checks: 3 passed, 24 deselected in
18.70 s. All used `--no-testmon`, real files, CocoIndex and SQLite with test
inference providers; these are orchestration results, not model-quality evidence.
Focused Ruff, format and ty passed. Rapid edit races, provider timeout shutdown,
full repository gates and command catch-up overhead remain to be assessed.

## Runtime lifecycle acceptance reopened (2026-11-04)

Audit at `b658137` confirmed the native pipeline cutover but contradicted the
claim that YawRuntime activates daemon live ingestion: its only update call is
catch-up `app.update()` in `_execute_sync_worker`. Native component live tests
do not cover this runtime activation gap. The progress checklist now records it.

Shutdown also joined the command worker for only five seconds before closing
CocoIndex and SQLite. Removed that timeout: resources now remain open until the
active operation returns under its provider timeout/retry policy. A real runtime
and SQLite regression holds work beyond 5.5 seconds, verifies shutdown remains
pending, then releases the work and confirms its SQL query succeeds before final
closure. Shutdown and runtime lock tests: 2 passed, 23 deselected (15.51 s).
Native source/publication tests: 10 passed (1.98 s). Both used `--no-testmon`.
Focused Ruff, format, ty and diff checks passed. No real-model pilot was launched;
full quality gates and daemon live startup/readiness/draining remain outstanding.

## Native CocoIndex declarative cutover, dataset-agnostic catalog & bounded live pilot (2026-10-31)

Completed the full cutover of `YawRuntime` (`start`, `sync_now`, `optimize`, `benchmark`) and `snapshot_dataset_bundle` onto the single declarative CocoIndex 1.0.26 pipeline (`indexing_app_main`), demolished all competing ingestion/caching/polling paths (`CorpusIngestor`, `sync_corpus_once`, `EmbeddingLane`, `indexing_overlap`, `YawRuntime._poll_loop`, `openie_cache`, `ReActV2` fallback, `"degraded"` lifecycle mode, and silent parser/proposer/RLM error swallowers), and extracted all dataset catalog metadata and baseline datapoints into `data/benchmarks/baselines.json` so `yaw/daemon/ingestion/datasets.py` contains zero hardcoded dataset branches.

### Bounded live native CocoIndex pilot (`outputs/native_cocoindex_pilot_16chunks/pilot_report.json`)

Executed against the resident Lemonade gateway (`google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0` + `EmbeddingGemma` Q8 on CUDA) over the 4 frozen Enterprise RAG JSON documents (`16,637` bytes, yielding `21` chunks under the current `YawConfig` chunker) in disposable state under `outputs/native_cocoindex_pilot_16chunks/`:

- **Initial live pilot run (`task-1998`)** completed all 21 chunk extractions and exposed a real prebatching gap in `prepare_sense_resolution_state` (`yaw/daemon/graph/planner.py`): when an entity (`"Hedge-duplicate-rate"`) had its definitions merged via `—` across two single-sense chunks of the same document and then appeared in a subsequent chunk requiring cosine sense comparison, the merged definition string was missing from `prebatch_texts`. Fixed in `prepare_sense_resolution_state` and locked down with `test_same_document_merged_definition_is_prebatched_for_subsequent_sense_resolution` (`tests/daemon/graph/test_normalization.py`).
- **Completed graph publication & fresh-process replay (`task-2028` / `task-2078`)**:
  - Graph planning + atomic `GraphTargetHandler` publication (`cold_elapsed_s`): **360.44 s** (`23` physical embedding requests, `53` physical chat requests, `172,603` provider tokens).
  - Reopened fresh `YawRuntime.sync_now()` replay on the same SQLite + CocoIndex state (`replay_elapsed_s`): **0.83 s** with **0 physical provider requests** (`0` embedding calls, `0` chat calls).
  - Committed SQLite rows in `yaw.db`: `source_chunks = 21`, `chunk_inference = 21`, `chunks = 21`, `entities = 146`, `facts = 149`, `edges = 566`, `edge_sources = 574`, `entity_labels = 213`.
  - Closed-world SQL SHACL validation (`lint_graph_shacl`): **`shacl_conforms = true` (`0` violations)**.
  - Peak process working set: **333.6 MiB**; max sampled GPU memory: **6,881 MiB**.
  - Measured on-disk sizes: `corpus_bytes = 16,637` B, `yaw_db_bytes = 2,576,384` B (2.46 MiB), `cocoindex_state_bytes = 6,246,336` B (5.96 MiB), `total_index_bytes = 8,822,720` B (8.41 MiB).

## Extracted graph record normalization (2026-10-30)

Moved deterministic OpenIE normalization from `index_chunks` into the graph-owned
`normalize_openie_record` function, deleting the embedded implementation. The
function has no database or model access. The engine consumes its typed result;
missing-definition inference and subsequent graph planning remain in the engine.
Tests verify literal lexical/datatype preservation, distinct homograph definitions,
source surface forms, deterministic output and unchanged input records.
The combined normalization, graph-build, edge and architecture run had 25 passes
and 2 failures. The exact architecture module inventory was updated for the new
graph-owned module; rerunning architecture plus normalization passed 11 tests
in 3.36 s. The remaining edge-test failure is HTTP 501 from its loopback server's
unimplemented tokenizer health endpoint; it remains unresolved. Focused Ruff,
format, ty and `git diff --check` passed. Full cutover and quality gates are not
complete.

## Pointer-only native source targets (2026-10-30)

Removed full chunk text from the native `source_chunks` SQLite target. Its typed
record now contains identity, source hash, physical line/AST coordinates and title.
Inference receives text separately alongside that metadata. Both source app and
combined indexing app declare the new target schema; tests assert the text column
is absent. No migration or legacy schema path was added.
Real-file live updates, failed replacement preservation, restart, title-aware
inference and fresh-process inference recovery pass: 6 passed, 18 deselected in
41.85 s (`--no-testmon`), with one upstream Pydantic warning. Focused Ruff, format,
ty and diff checks pass. Parsing memo entries still retain text; this removes one
database copy, not all text retention or the remaining runtime scaffolding.

## Retired optimizer scheduling callbacks (2026-10-30)

Removed three unused GraphOptimizationHooks fields and their constructor wiring:
`ensure_candidate_indexed`, `reset_sandbox_graph`, and `index_chunks`. Reference
search showed these were declaration-only interfaces, not the actual candidate
preparation path. Their removal does not remove query-time indexing, which still
requires explicit candidate preparation during the runtime cutover. The affected
optimizer progress tests passed (3 tests, 1.70 s, `--no-testmon`). Focused Ruff,
format, ty and diff checks passed; the retired callback reference search found
no matches in `yaw` or `tests`.

## Fresh-process native inference recovery (2026-10-30)

A subprocess runs the real native file/inference app with SQLite targets and a
loopback provider, then exits abruptly with `os._exit(23)` after inference target
publication and before any graph publication. A second process opens the same
tracking and target databases. Both observe one artifact and its 16-byte vector;
the provider count remains exactly one embedding request plus two OpenIE calls.
The test passes (1 passed, 20 deselected, 38.79 s, `--no-testmon`); focused Ruff,
format, ty and diff checks pass. This verifies completed inference reuse across
processes, not interruption during target application, production daemon recovery,
or inference quality. DSPy response caching is disabled for this test.

## Atomic target validation (2026-10-30)

The native graph sink now runs the existing SQL SHACL linter with auto-healing
disabled before committing. Invalid prepared rows raise and roll back the
whole publication. A real SQLite test verifies a missing-definition replacement
preserves the prior entity definition, chunk vector and FTS entry and permits a
subsequent valid publication. The new guard also exposed asymmetric skos:related
fixtures; those fixtures now declare both required directions. Focused publication
tests: 5 passed in 0.85 s with `--no-testmon`; focused Ruff and ty passed.
This does not move graph planning out of the old engine transaction or complete
the runtime cutover.

## Native graph assertion withdrawal (2026-10-30)

The draft atomic graph target now tracks graph-wide edge identities separately
from chunk fingerprints. Withdrawal removes the empty-source provenance and
deletes the edge only if no other provenance remains. Previously, unchanged
chunks could leave a withdrawn semantic assertion in the published graph.
Real CocoIndex/SQLite tests reopen tracking state and replay both publications,
checking withdrawal with and without surviving chunk provenance. The graph
publication module passes 4 tests (`--no-testmon`, 1.55 s). Focused Ruff, format,
ty and `git diff --check` passed. Runtime integration and full acceptance remain
unfinished; these tests exercise the draft target directly.

## Native chunk embedding formatting (2026-10-30)

The native chunk component now passes source pointers and text through the
existing model-aware chunk formatter before preparing embedding requests.
Loopback HTTP tests cover plain and title-aware profiles, binary vector storage,
unchanged reuse, and OpenIE-only invalidation without another embedding call.
Focused pytest with `--no-testmon`: 2 passed, 18 deselected (12.44 s), with one
upstream Pydantic warning. Focused Ruff check/format, ty and `git diff --check`
passed. This proves orchestration and request formatting, not model quality or
runtime cutover; the native app is still not the production indexing path.

## Native indexing replacement: partial orchestration evidence (2026-10-28)

2026-10-29 LM retry policy: removed the `num_retries` overrides from
`create_dspy_lm` (was 1) and `create_dspy_rlm_pair` (was 0), so DSPy's native
default of 3 retries with `min(2^attempt, 60)` s backoff applies. Callers may
still pass `num_retries` explicitly through `dspy.LM` kwargs. The 240 s
per-attempt timeout is unchanged. No per-request latency telemetry exists. The
worst case was instead derived from live Lemonade `/api/v1/stats` (Gemma 4 E4B:
59.5 tok/s decode, 0.59 s TTFT for 1,103 prompt tokens) and the configured output
budget of 8,192 tokens (4,096 scaled to the 131,072-token context). That is about
140 s of decode plus prefill, so a 60–120 s timeout would cut off legitimate
maximum-length generations. The 11 test files using the LM factories had
**4 failed, 154 passed**. Three failures are the known pre-existing ones. The
fourth (`test_fresh_process_cache_survives_rollback_without_repeated_generation`)
passed alone (**1 passed in 39.07s**) and in the earlier full run; it pins
`num_retries=0` itself. Ruff, format and ty passed.

2026-10-29 runtime test harness: the full `pytest --no-testmon` run had
**16 failed, 800 passed in 788.60s**. Eight `tests/daemon/test_runtime.py`
failures predated the uncommitted work. Commit 52aa52e added two requirements
for synonym judgment that the runtime test fixture did not meet: a verified
serving tokenizer and one decision per candidate pair. The fixture now builds a
`ServingTokenCounter` against the existing loopback `provider` and answers with
the shared `synonym_protocol` helpers. No production code changed. The module
passed explicitly: **24 passed in 55.26s**. Ruff, format, ty, the architecture
tests (9 passed) and `git diff --check` passed. This validates orchestration with a
deterministic LM only; it says nothing about real-provider optimize behaviour.

2026-10-29 lifecycle follow-up: the public live-to-catch-up transition resolves
the earlier source-shutdown uncertainty. Awaiting catch-up on the same app and
then the original live handle completes both operations. The test writes another
file afterward, verifies it remains absent from SQLite, and confirms the next
catch-up ingests it. All three native source-target tests passed explicitly:
**3 passed in 2.14s**. Ruff and diff checks passed. This verifies filesystem
lifecycle only; runtime/provider/graph shutdown acceptance remains outstanding.

2026-10-29 live filesystem probe: a real Windows live-source test observed
startup, modification, rename and deletion through `UpdateHandle.watch()` and
verified the native SQLite rows at each step. Waiting for zero in-progress
components before closing the environment removed post-close tracking errors.
The focused event test passed (1 passed, 2 deselected, 2.33s).
Controlled shutdown remains unproven: a separate probe wrapped `app.update(live=True)`
in the public five-second timeout, but awaiting its result still exceeded ten
seconds. The inspected public UpdateHandle has no cancel operation; closing an
environment is documented as closing its database, not stopping live sources.
Do not treat this source-event test as completion of the daemon lifecycle gate.

2026-10-29 strict failure propagation: removed SHACL inference's broad exception
handler that logged an error and continued indexing with no proposals. Provider
and parsing failures now escape the graph transaction. This exposed a malformed
RLM response in an existing deterministic fixture; the fixture now returns the
required reasoning/code fields. Both SHACL build suites then passed explicitly:
**10 passed in 26.15s**. Focused Ruff, ty and diff checks passed.

2026-10-29 SHACL boundary cleanup: removed the RLM's direct repair-write tool,
the writable callback binding, and REPL-applied repair bookkeeping. The model
returns repair proposals; the host applies them and the final SQL SHACL gate
still checks the resulting graph. Both SHACL build suites plus program-state
tests ran without testmon filtering: **15 passed in 30.07s**. Focused Ruff, ty
and diff checks passed. This removes one write escape hatch but does not yet
move the remaining host graph-planning inference outside its transaction.

2026-10-29 follow-up: graph base-row persistence is extracted into
`PreparedGraphRows` / `apply_prepared_graph_rows`, used by the current engine.
The helper rejects calls without an active transaction and contains no inference.
Real SQLite verifies rollback across relational/vector/FTS rows and idempotent
replay. The publication test plus both SHACL build suites passed explicitly with
`--no-testmon`: **11 passed in 23.89s** (the earlier testmon run deselected 10).
Focused Ruff and production ty passed; the latest `git diff --check` is clean.
This is not yet the final graph sink: synonym/SHACL inference remains in the
existing engine transaction and must still move to graph preparation.

The new daemon-level app composes native file components, concurrent OpenIE and
embedding functions, and native SQLite source/inference targets. The existing
runtime has not yet switched to this app; graph planning/publication and legacy
scheduler removal remain incomplete.

Real-file/SQLite/loopback-HTTP checks executed individually:

- Source replacement/deletion and environment restart: passed.
- Malformed replacement preserves prior source rows; corrected replacement retries: passed.
- Native embedding batching and persistent environment reopen without new HTTP calls: passed.
- OpenIE/embedding concurrent admission, named trace restoration and unchanged-update reuse: passed.
- Composed file app produces one source row and one inference row using three HTTP
  requests, makes no further requests on unchanged update, and removes both rows
  on confirmed file deletion through native target reconciliation: passed.

The last check ran with `-k native_file_pipeline`: 1 passed, 18 deselected.
These checks establish orchestration only, not real-model quality or throughput.
No fresh-process crash, full runtime lifecycle or live-mode acceptance is claimed.
Focused production ty and Ruff checks passed. The latest repository diff check
reports pre-existing/user-edited trailing whitespace at AGENTS.md:31, left intact.

## Bounded indexing overlap (2026-10-28)

Implemented on `dev` at `https://github.com/negatrait/yaw.git`, preserving the
existing compact-synonym work and user configuration changes. The new scheduling
switch defaults to false. It is excluded from both index fingerprints and the
runtime optimization resume identity. One embedding worker performs only provider
work; the engine owner resolves caches, persists validated results and publishes
telemetry between OpenIE accounting slices. Graph writes, sense resolution,
synonyms and SHACL remain ordered. No new dependencies, index reset or commit.

The first live overlap attempt exposed an OpenAI SDK cold-import race:
`deadlock detected by _ModuleLock('openai.resources.chat')`. Resolve SDK resource
imports on the owner before starting the embedding worker. A fresh-process
loopback regression now verifies concurrent first chat/embedding calls. This fix
has not received a second live-model comparison: the original 20-minute budget
elapsed during diagnosis and verification. The continuation checked that same
deadline and sent no model requests. Overlap remains opt-in; no speedup is claimed.

### Authorized pilot evidence

Sixteen Enterprise chunks were frozen with physical source files under ignored
`.yaw/overlap_pilot/`. Both resident models were ready; no optimizer was active.
The models were Gemma 4 E4B (131,072 context) and the configured Q8 EmbeddingGemma,
both on CUDA. Separate disposable arm databases and DSPy cache directories were
used. The gateway proxy recorded physical provider requests and enforced the
shared deadline. No GEPA or full benchmark was launched.

| Observation | Serial, first eight chunks | Original overlap attempt |
|---|---:|---:|
| Completed batch | yes | no: SDK import race |
| Elapsed indexing | 297.88 s | 42.70 s before failure |
| OpenIE phase | 258.65 s | incomplete |
| Graph preparation after OpenIE | 5.09 s | not reached |
| Transaction, including synonym judgment | 31.87 s | not reached |
| Synonym generation | 31.51 s, 18 decisions | not reached |
| Physical embedding/chat requests | 11 / 18 | 0 / 2 |
| Peak process working set | 291.8 MiB | 283.5 MiB |
| Maximum sampled device memory | 7,091 MiB | 7,091 MiB |
| Committed chunks/entities/facts | 8 / 61 / 73 | 0 / 0 / 0 |

Serial chat requests consumed 30,853 provider-reported tokens. Its initial eight
chunk embeddings comprised two physical requests totaling 0.344 s at the proxy;
formatting, initialization and owner-side costs are outside that HTTP duration.
OpenIE dominated this sample. The failed arm is not a performance comparison or
a quality regression measurement. Remaining paired batches were not completed.
CPU regression tests also ran during the pilot, so even the serial timing is a
diagnostic observation rather than an isolated hardware benchmark.

### Verification

- Final affected graph, synonym, embedding, storage, configuration and architecture
  selection: **122 passed** in 238.64 s, `--testmon-noselect` (no testmon deselection).
- Added real-SQLite optimization resume identity check: **1 passed**, 32 explicitly
  deselected by `-k`, not by testmon. TOML scalar/array round trips, including the
  new switch: **8 passed** with `--testmon-noselect` in 1.38 s.
- Cold-process concurrent calls, serial/overlap graph and provenance equivalence,
  one request per lane, stage token attribution, invalid vectors, malformed chat,
  real HTTP timeout, failure draining, bounded retained results/threads, and
  fresh-process reuse of persisted embeddings are exercised with native DSPy,
  loopback HTTP and real SQLite. These establish orchestration, not model quality.
- Earlier compact/engine selection: **45 passed**. Mandatory pair-specific GBNF
  reached the native loopback provider and all 18 live serial-pilot synonym
  decisions passed coverage validation. Three obsolete fixtures were corrected;
  missing canonical IDs now assert rejection and no graph writes.
- Repository Ruff lint, formatting and ty checks pass; `git diff --check` passes.
  The formatter also corrected existing formatting in watchdog and two graph tests.
- One upstream Pydantic `ReadOnly` warning remains; it is not suppressed. The full
  repository test suite and full-corpus quality/performance gates were not run.

Artifacts: `.yaw/overlap_pilot/run.py`, `chunks.json`, `health.json`, `report.json`,
`continuation.json`, per-arm logs and `batch-0.json` reports. Production sources,
indexes and experiments were not overwritten.

## Compact synonym coverage failure (2026-10-27)

The user-reported optimization failure omitted one requested decision (`e0|e4`).
Inspection found that compact packets required exact coverage in host validation,
while generation still used a generic grammar permitting an empty/short list.
The repair generates a mandatory ordered decision rule for every supplied pair,
with fixed pair IDs, the existing five relation choices and pair-local canonical
IDs. Native DSPy dispatch passes this grammar in the request; full coverage is
still validated before graph writes. Changed grammar request keys prevent reuse
of the old incomplete raw response. Completed graph/OpenIE caches are preserved.

The runtime progress message now says it is preparing the optimizer and that
validation indexing precedes trials, instead of claiming evolution has started.
The 150/100 query split can use a shared corpus; validation passage counts describe
that corpus, not the number of validation questions.

Added a loopback regression for packet-specific grammar dispatch and unique IDs.
Execution was unverified at the time of this entry: automatic approval review rejected the preceding
pytest rerun because workspace credits were exhausted, explicitly stating that
the review could not complete (not that the command was unsafe). No alternate
execution route was used to bypass that block. The earlier compact-focused run
passed 24 tests; that result predates this coverage repair. Broader engine fixture
adaptations, final lint/type checks and the paired live pilot were outstanding.
The 2026-10-28 record above supersedes the regression-verification limitation;
a completed paired performance pilot remains unavailable.

## Epic 2 specification reconciliation (2026-10-27)

Documentation-only update on `dev`, origin `https://github.com/negatrait/yaw.git`.
Reconciled the Epic 2 parent and both sprint specs against the current runtime
and prepared-bundle ingestion contracts. Added a planned provenance-schema note
to FOUNDATIONAL; preserved other in-progress changes. No runtime implementation,
dependency installation, GEPA run or benchmark was performed for this update.

The specs now keep `D1-A6` open pending user-run Tier 2 holdout evidence. They
require indexed source fingerprints distinct from chunk IDs and returned-text
hashes, typed citation propagation, immediate pending-queue cancellation,
ordered progress, atomic sync coalescing and public lifecycle integration.
They distinguish sync receipts from awaited calls and explicit MCP cancellation
from transport disconnection. Deferred Epic 3 CLI/Optuna ideas do not override
FOUNDATIONAL or become Epic 2 prerequisites. Optuna's current manifest/lock
presence remains a separately identified policy discrepancy.

The preceding audit's real prepared-file/CocoIndex/hydration probe found that
recomputing chunk IDs from hydrated results mismatched both unchanged chunks
(2/2); Python's queue also lacked the prescribed `_queue` attribute. These
observations motivated replacing those prescriptions; the probe was not rerun
during this documentation update. No FastMCP HTTP acceptance or model-quality
gain is claimed.

Verification for this update: all 19 local file links across the three specs
resolve; `git diff --check` passes. Runtime tests, Ruff/ty and preflight were not
run because this packet changes documentation only. Future acceptance tests
remain unchecked and specify real SQLite/files, loopback TCP, event-controlled
concurrency, actual 32-slot saturation, source isolation and restart behavior.

## Dataset bundles and persistent ingestion (2026-10-27)

Implemented on `dev`, origin `https://github.com/negatrait/yaw.git`, using the
CocoIndex and engineering-real-tests workspace skills. This supersedes the
proposal-only ingestion audit below. Raw source files and existing experiments
were preserved. No optimizer, watchdog or unattended model run was launched.

### Delivered contracts and removals

Preparation now exports the complete configured corpus into
`outputs/datasets/<dataset>/documents/*.json`. Declared ID/title/body keys retain
their original values. QA bytes/order are preserved in `evaluation/questions.json`.
Opaque paths, original identities and hashes live in the manifest. Atomic file
writes and a manifest published last operate under a bundle lock; readers validate
the inventory and hashes under that same lock. Incomplete/stale bundles fail.
Unknown document references and conflicting duplicate IDs fail explicitly.

Enterprise quota selection remains seed 42, 150 training and 100 validation
examples, owned by the evaluation loader. Valid preassigned memberships are
preserved; duplicate question identities are rejected. Smaller optimization slices
select separately within those partitions. Other benchmark subset/distractor and
gold-answer policies remain in evaluation adapters. The reward was not changed.

| Removed responsibility | Replacement and verification |
| --- | --- |
| Downloader title/category/tool normalization, answer-alias synthesis, relevance-driven document selection, embedded Enterprise Parquet sampling and migration branches | Configured JSON inputs, direct declared-field export; all ten registry entries tested with real files, exact QA bytes and unchanged-file modification times |
| Monolithic filename guessing/reformatting, source-file fallback resolution, mixed document/label helpers and facade exports | Typed `DatasetBundle`, manifest lookup and original document-ID selection in CLI/runtime/benchmark |
| CocoIndex state resets and candidate pools reconstructed from changed targets | Persistent per-file memo components return complete typed snapshots independently of callbacks |
| Benchmark key allow/deny constants and injected projection defaults | Physical document/evaluation separation; production whitelist/blacklist settings remain |
| Hotpot tuples, LV-Eval markers and NarrativeQA parsing inside ingestion/hydration | Generic strict JSON/JSONL projection; dataset interpretation stays in evaluation adapters |
| Repeated whole-corpus/nested-span scans and duplicated chunk builders | Ordered native property spans with indexed record ranges, shared source ASTs and one chunk construction path |
| JSON brace repair, raw substitutes after parsing failure and broad splitter/scope exception recovery | Explicit projected-property grammar, strict UTF-8, propagated errors and source/key diagnostics for unsafe spans |
| Retired tests and documentation commands | Bundle fidelity, isolation, native restart reuse, settings/helper invalidation and strict-error tests; updated preparation/fresh-index commands |

Repository searches found no retired API references in `yaw/`, `scripts/`,
`benchmark.py` or tests. Historical audit and explicit removal-inventory mentions
remain in documentation. Genuine aliases, production JSON/JSONL filtering,
Markdown/code scopes and graph inference caches/checkpoints remain.

Native CocoIndex concurrency is bounded to four file components. Directory
enumeration completes before mounted processing, so partial scans cannot establish
deletion. The memo dependency includes parser code, shared core contracts and
the installed CocoIndex version. Scope rendering uses the public `render_match`
API with source-grounded anchors; touched modules have no private API access.
`poll()` remains incremental; `snapshot()` requires a successful scan and returns
the complete cached current tuple. Compact multi-record arrays and partially
excluded nested objects that cannot retain source identity in a safe physical
line span fail. Decoded string lines never become file coordinates.

### Configured Enterprise measurement

Final serial measurement used a fresh generated copy under
`.yaw/dataset-ingestion-audit/final/`, with the configured `1200/120/100`
chunk-size/overlap/minimum settings. It exported all **1,308** configured documents
and produced **9,062** chunks. Inputs outside the audit directory were untouched.

| Operation | Seconds | Documents reparsed | Complete chunks |
| --- | ---: | ---: | ---: |
| Initial preparation | 45.848 | n/a | n/a |
| Cold scan | 14.628 | 1,308 | 9,062 |
| Warm scan | 2.078 | 0 | 9,062 |
| Warm snapshot getter | 0.00783 | 0 | 9,062 |
| Reopened environment scan | 2.147 | 0 | 9,062 |
| One-document edit | 2.006 | 1 | 9,062 |

Warm scans emitted zero additions/deletions. The edit emitted one added and one
deleted chunk. Sampled peak process RSS was **144.96 MiB** (20 ms sampling).
Low-overhead Python monitoring counted parser entries. The getter releases the
previous retained tuple in this measurement, so its timing includes that release.
Fresh-process reuse is established separately by regression tests. Earlier
measurements varied with filesystem/cache activity; these timings establish
scan behavior, not an overnight stability guarantee or LLM throughput improvement.

### Verification and practical limits

The complete affected selection, explicitly disabling testmon deselection once,
passed **274 tests in 351.91 seconds**. It included ingestion, hydration, graph
indexing/storage/cache, architecture/configuration, download transport/CLI,
benchmark, runtime and CLI tests. One external Pydantic `ReadOnly` schema warning
remains. A preceding testmon selection passed 116 (154 deselected). After final
strict hydration and indexed-span changes, another testmon selection passed
52 (22 deselected), including fresh-process insertion, chunk/projection settings
invalidation and shared helper edits. The final Enterprise subset consumer
selection passed 18 (104 deselected) in 82.79 seconds. After replacing private
scope API access with the public renderer, the final scope/hydration/architecture
selection passed 57 (26 deselected) in 6.70 seconds. These overlapping counts
are not additive.

Real SQLite tests verify callback retry and that a failed scan cannot delete
unvisited sources. A separate-process loopback inference test forces graph
reconstruction and verifies **zero additional provider calls** for completed
unchanged indexing, plus invalidation after source/prompt/model/sampling changes.
This is orchestration evidence; no resident-model quality evaluation or corpus
parity benchmark was performed. Existing resume/GEPA orchestration tests passed.

Intermediate failures came from retired layout assertions, consumer fixtures and
document-pool padding before identity resolution; those were replaced/fixed. One
mistyped test path collected no tests before the corrected command ran. Repository
Ruff and ty checks passed. Final repository Ruff, ty, scoped formatting checks
and `git diff --check` passed. No dependency or lockfile update was required.

Prepare and start a **fresh** run (commands documented, not launched):

```powershell
uv run --locked python scripts/download-datasets.py --dataset enterprise_rag_bench --source-dir outputs/datasets
uv run --locked python scripts/download-datasets.py --dataset enterprise_rag_bench --check
uv run --locked yaw dspy optimize
```

The source directory above names the existing configured JSON inputs. For other
locations use explicit `--corpus` and `--questions`; for all registered inputs use
`--dataset all --source-dir <inputs>`. Do not resume a pre-refactor index. No
migration, compatibility alias or legacy-layout fallback was added.

## Indexing performance repair and ingestion proposal audit (2026-10-26–27)

Checkout verified on `dev`, origin `https://github.com/negatrait/yaw.git`. The graph performance
changes are now in `25f886f`. No optimization run or watchdog was launched during this work.

### Measured graph preparation changes

Read-only snapshot: `.yaw/yaw_optimize_20261026_144726.db`, 385 entities. Experimental removal
of adjudicated semantic edges occurred only in an in-memory SQLite copy. Comparisons used the
pre-change source and replacement in the same probe process; these are microbenchmarks, not
full indexing throughput or task-quality results.

| Probe | Before | After | Correctness check |
| --- | ---: | ---: | --- |
| 80 hydration calls, cold start | 0.5481 s | 0.2199 s | Exact rendered text equality |
| 80 repeated hydration calls | 0.4211 s | 0.0560 s | Same pointers and 900-character budget |
| Candidate search, last 100 entity IDs | 2.2893 s | 1.0735 s | Identical empty result on already adjudicated snapshot |
| Candidate search, unadjudicated copy | 2.3617 s | 1.1413 s | Identical 25 pairs and similarity scores |
| Dossiers for those 25 pairs | 75,327 characters, 2 requests | 64,644 characters, 2 requests | Old batching omitted 8 pair dossiers; replacement includes all 25 |

Changes: incident-pair enumeration instead of visiting all graph pairs; bounded lexical and
hydrated-text caches; reuse of a parsed source across splitter attempts; shared entity profiles
and source excerpts in synonym requests; request cache keys include the actual dossier;
candidate-search, synonym-judgment and SHACL progress messages are separate.

The hydration caches are bounded, in-process accelerators; they are not durable checkpoints.
Synonym stage-cache writes still participate in the graph transaction. They are not independently
committed after each request. Existing per-chunk OpenIE checkpoints remain separate and survive
graph rollback. Code changes invalidate the broad candidate-cache namespace; this does not erase
the OpenIE checkpoint tables or already committed graph. Reusing those requires compatible
model, prompt and run settings.

Verification before the final small edits: the focused graph/storage/cache suite passed
97 tests (4 testmon-deselected); the subsequent storage/SHACL/engine/architecture selection
passed 45 (34 deselected). These selections overlap and must not be added as a unique test count.
Ruff and ty passed at that checkpoint. The final additions were two progress notifications,
source-file-edit coverage and a source-cache eviction test. Formatting passed afterward, but
the final pytest/Ruff/ty rerun was not executed: automatic approval review reported exhausted
workspace credits. Final test status for those last additions remains pending.

The suggested JSON object-pattern substitution was also measured: `r"{\(PAIRS*\)}"` matched
1,383 object spans in approximately 13 seconds; `"{ ... }"` matched only one in approximately
0.7 seconds. The latter is not an equivalent optimization and was not adopted. Lexical thresholds,
dense top-k, the full shared retrieval corpus and the strict graph validation gate were preserved.

### Attached ingestion replacement: source audit

The 2026-10-27 attachment is useful design input, not an executable replacement for current YAW.
No ingestion source changes were made during this proposal review. Verified issues:

- `sync_source_files_for_docs` removes `state/coco_lmdb` before each sync, then reconstructs the
  complete optimization pool from `delta.added_chunks`. Record-level memoization alone cannot
  improve restart reuse while this reset remains. Removing the reset alone is also incorrect:
  unchanged chunks would disappear from the returned candidate pool. A complete snapshot read
  path must accompany persistent incremental state.
- The proposal retains the same variadic JSON object pattern and adds `json.loads` on matched
  objects. It does not establish a solution to the measured matching cost.
- `_split_and_build_chunks` derives physical JSON file lines from the decoded passage's line
  numbers. Escaped newlines in a JSON string are not source-file newlines. Long-record chunks
  must retain valid original record spans and AST bounds.
- The early tuple-pattern branch processes every matching tuple before applying its ancestor
  key path. A tuple beneath a blacklisted field can enter the corpus. Whole-object source spans
  can also reintroduce excluded fields during hydration. Projection and hydration must be tested
  together, including compact JSON and escaped Unicode/newlines.
- The proposal imports nonexistent YAW `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` exports,
  calls the three-argument `inclusive_end_line` with two arguments, changes exported function
  signatures and tuple layouts, and leaves `ChunkPointer.ast_island_*` at defaults. Its match
  access differs from the current working `m.chunks[0]` and capture-list API. Runtime compatibility
  was not established; reading the installed dependency source was sandbox-denied.
- Generic corporate records without a recognized body field become an empty result; programming
  language scope extraction is removed. The Markdown fence toggle also conflates backtick and
  tilde fences. These are lost behaviors, not proven performance improvements.
- The current implementation does contain avoidable work: per-line `render_ranges` calls,
  repeated nested-span scans and duplicated chunk assembly. It also masks JSON decoding errors,
  suppresses `CodeSource` construction errors, and substitutes whole passages when splitting
  unexpectedly produces no chunks. Those failure paths merit explicit, contextual errors.

Recommended implementation order: preserve a complete optimization corpus snapshot while keeping
CocoIndex state; define a typed projected-record contract with original source coordinates; replace
nested JSON searches with one ordered span traversal; memoize record transformation with tracked
helper dependencies; remove silent parser recovery and propagate component failures through poll;
then consolidate chunk construction and scope extraction. Acceptance requires unchanged-corpus
restart reuse, one-record edits, real deletion, malformed-input failure/retry, exact hydration and
blacklist exclusion, plus cold/warm timings on the full EnterpriseRAG corpus. Current snapshot
reconstruction, projection semantics and source-coordinate guarantees must not be weakened to
obtain a faster timing.

Date: 2026-10-13. This file owns measured results and open failures. The [daemon](epic-1-hipporag-daemon.md) and [FastMCP](epic-2-fastmcp-http.md) epic checklists remain uncompleted until implementation evidence is added.

## 25. CocoIndex optimization performance audit (2026-10-22)

- **Scope:** focused audit requested before a long DSPy optimization run, using
  `.agents/skills/cocoindex/SKILL.md`. Verified the designated Documents checkout
  and `https://github.com/negatrait/yaw.git` remote. Preserved existing config edits.
  No runtime, dependency, lockfile, source corpus, or production DB changes.
- **Report:** [Native CocoIndex indexing](native-cocoindex-indexing.md) records
  dependency tracking, target-state boundaries, batching, invalidation, and RAM
  limitations. Installed packages: CocoIndex **1.0.25**, DSPy **3.4.1**.
- **Real-substrate checks:**
  - `uv run --locked pytest tests/daemon/ingestion/test_corpus.py
    tests/daemon/graph/test_candidate_cache.py
    tests/daemon/graph/storage/test_candidate_cache.py
    tests/daemon/graph/inference/test_response_cache.py -q --tb=short`
    -> **21 passed in 381.28 s**. Includes add/no-op/edit/rename/delete ingestion,
    reopened persistent ingestion state, failed target compiler retry, typed
    SQLite stage payloads, FIFO eviction, independent-process zero-request stage
    replay, and native DSPy response-cache/GEPA accounting checks.
  - `uv run --locked python outputs/cocoindex_optimize_audit.py` -> **passed**;
    results saved in `outputs/cocoindex-optimize-audit.json`. Uses disposable
    source snapshots and real temporary LMDB/SQLite, without inference calls.
    A parsing-helper-only edit yielded **zero changes** with existing state,
    whereas fresh state emitted the changed title: missing helper-code tracking.
  - Configured outer batches of 64 were prepared as **4 / 11 / 32 batches** for
    100 / 500 / 1,200-character inputs respectively. Effective inner limits are
    16 items and 3,500 formatted characters, independent of the configured 64.
  - Query-prefix-only mutation preserved the document prefix but changed the
    shared stage namespace: avoidable invalidation of unrelated cached stages.
  - 100 warm SQLite reads with a 5,000-passage, 5,068,890-byte source pool took
    **3.142 s**, versus **0.061 s** without rebuilding the source map. Both ran
    with tracemalloc; the first showed **639,729 bytes** peak traced allocation.
    These are synthetic CPU measurements, not whole-run speedups or peak RSS.
- **Checks:** `git diff --check` passes after the audit documentation changes.
- **Limits:** loopback providers validate orchestration and request counts only.
  No live inference-server throughput, GPU/VRAM, full-optimization peak RSS,
  retrieval-quality benchmark, Tier 1/Tier 2 acceptance, or power-loss measurement
  was performed. The audit recommends contained fixes; it does not claim they
  have been implemented. Fresh optimization scans intentionally reconstruct the
  complete candidate pool, so the helper-tracking defect primarily affects
  persistent daemon ingestion rather than that fresh scan.

## 2026-11-07: Cross-Batch SHACL Repair Persistence, Focus-Isolated Healer Memoization, and Single-Pass Incremental Publication

- **Governing Tier 2 Specs & Tier 3 Review Records**:
  - Tier 2 specs: [native-cocoindex-graph.md](native-cocoindex-graph.md) ([§5.2](native-cocoindex-graph.md#52-batch-independent-identity-and-candidate-discovery), [§5.3](native-cocoindex-graph.md#53-resource-vocabulary-and-statement-boundary), [§9.2](native-cocoindex-graph.md#92-performance-review-measured-implementation-units), [§Decisions](native-cocoindex-graph.md#decisions)), [native-cocoindex-indexing.md](native-cocoindex-indexing.md) ([§3](native-cocoindex-indexing.md#3-revision-identity-memo-and-target-isolation), [§Decisions](native-cocoindex-indexing.md#decisions)), and [w3c-improvements.md](w3c-improvements.md) ([§3.2](w3c-improvements.md#32-end-of-round-active-edc-canonicalizer-yawdaemongraphstoragegraph_opspy), [§Decisions](w3c-improvements.md#decisions)).
  - Tier 3 review records: [reviews/shacl-healer-publication.md](reviews/shacl-healer-publication.md) and [reviews/incremental-publication-work.md](reviews/incremental-publication-work.md).
- **Measured Paired `64 + 64` vs. `128` Batch Equivalence Benchmark** (`outputs/benchmarks/batch_equivalence_c4.xml`, `tests/daemon/graph/test_batch_equivalence.py`, `2 passed in 26.30s` `[measured: outputs/benchmarks/batch_equivalence_c4.xml, c6fa080, loopback deterministic provider]`):
  - `same-process`: `incremental (64 + 64) = 6.7189 s`, `reference (128) = 6.1588 s`, $R_{\text{time}} = 1.0909 \in [0.80, 1.25]$ `[measured: outputs/benchmarks/batch_equivalence_c4.xml]` (`[target]`: $0.80 \le R_{\text{time}} \le 1.25$).
  - `restart-between-batches`: `incremental (64 + 64) = 5.6023 s`, `reference (128) = 4.9711 s`, $R_{\text{time}} = 1.1270 \in [0.80, 1.25]$ `[measured: outputs/benchmarks/batch_equivalence_c4.xml]` (`[target]`: $0.80 \le R_{\text{time}} \le 1.25$).
  - Canonical set equality across all 15 graph tables: `100%` identical (`entities: 8, labels: 256, mentions: 256, assertions: 256, facts: 133, fact_sources: 256, edges: 525, edge_sources: 640, chunks: 128, chunk_fts: 128, fact_fts: 133, extractions: 128, chunk_vectors: 128, entity_vectors: 8, fact_vectors: 133` `[measured: tests/daemon/graph/test_batch_equivalence.py]`).
- **Read-Only Snapshot Verification on `.yaw/yaw_optimize_20261106_221811.db`**:
  - Cloning `.yaw/yaw_optimize_20261106_221811.db` (`384` published chunks across `6` batches of `64` chunks) to `:memory:` and running `canonicalize_fact_definitions(mem, prior_definitions=prior_defs)` preserves `42` of `42` (`100%`) prior-batch SHACL-healed `facts.definition` rows (`healed_before=42 preserved=42` `[measured: .yaw/yaw_optimize_20261106_221811.db, c6fa080]`), compared to `0` of `42` (`100%` relapse) under the retired unconditional `ORDER BY assertion_id LIMIT 1` overwrite.
- **Assumption Probes & Verification Suite Results**:
  - Gate G1 (`test_evidence_revisions.py -k shacl_memo`): `1 passed in 8.17s` (`0` healer re-invocations for previously healed focus nodes when earlier-sorting `fact-<md5>` violations are added `[measured: tests/daemon/graph/test_evidence_revisions.py]`).
  - Gate G2 (`test_evidence_revisions.py -k fact_definition`): `1 passed in 4.89s` (cross-batch healed fact definition persists across unrelated and re-asserting batches, survives endpoint entity merges via `merged.fact_bindings`, and invalidates on endpoint definition collision `[measured: tests/daemon/graph/test_evidence_revisions.py]`).
  - Gate G3 (`test_grammars.py test_parsers.py -k shacl`): `3 passed in 0.05s` (lexical `label` rule on `set_pref_label`, empty abstention `{"repairs": []}` permitted, scalar/empty/partial repairs rejected `[measured: tests/daemon/graph/inference/test_grammars.py, tests/daemon/graph/inference/test_parsers.py]`).
  - Full repository test suite (`pytest -o addopts=""`): `938 passed, 3 subtests passed` (`0` failures `[measured: pytest -o addopts=""]`).
  - Static quality gates: `ty check`, `ruff check`, `ruff format --check`, and `git diff --check` all passed with `0` errors `[measured: c6fa080]`.
- **What This Measurement Establishes and Does Not Establish**:
  - Loopback/deterministic fixtures establish `64 + 64` vs. `128` canonical set equality, focus-isolated memo reuse, single-pass `reembed_facts` embedding, delta-scoped `fact_fts`/`entity_labels` updates, and `42/42` healed fact preservation on the captured `.yaw/yaw_optimize_20261106_221811.db` snapshot; they do not by themselves establish full-corpus GEPA QA/Recall benchmark completion or Tier 2 acceptance.
