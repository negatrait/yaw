# Native CocoIndex indexing and live ingestion

Status: revised target contract, 2026-11-05. Earlier acceptance on 2026-10-28 was
reopened for lifecycle work on 2026-11-04; this rewrite does not close those gates.

## 1. Outcome and division of responsibility

[Native graph publication](native-cocoindex-graph.md) defines the two controlling
goals: comparable cumulative compute and comparable final graph for 64+64 versus
128 chunks, with exact controlled-evidence batch independence. It owns numeric
tolerances, graph identity, publication, scheme removal and paired evidence.
This document owns sources, scheduling, memoization and lifecycle. Do not maintain
a second set of thresholds or a separate graph implementation here.

`skos:ConceptScheme` is removed from the current target: no clustering, scheme
LLM/embedding calls, tables, memberships, PPR seeds/traversal, QA or telemetry.
[W3C improvements](w3c-improvements.md) retains supported concept relations and
identity guardrails. Future schemes require demonstrated Recall@N, EM or F1
improvement and explicit resource justification, not a dormant production path.

Production publishes at most 64 changed source chunks per update cycle. The 128
reference exists only in the paired harness, using identical transforms and
bounded provider admission with a different publication boundary. It is not a
production dataset preset or an alternate planner. All work needed to settle
both arms counts toward the graph plan's timing comparison.

## 2. One native pipeline for live and frozen sources

Use one `coco.App` per environment with stable component paths. Live filesystem
input uses `localfs.walk_dir(..., live=True, rescan_interval=None)`; frozen input
provides immutable chunks and source identities. Both mount the same extraction,
embedding, semantic-resolution and final-target components. Neither benchmark nor
optimizer candidate preparation owns another sequential inference/write engine.
Unbound retrieval engines fail when indexing is requested; retrieval never starts
hidden candidate indexing.

CocoIndex owns change detection, dependency scheduling, memo and target tracking.
No custom executor, polling loop, serialized pending-delta journal, synthetic
acknowledgement scheduler or dataset-specific batching. Admit bounded source
work before mounting inference; an endpoint semaphore alone does not prevent
mounting the entire corpus and retaining all text/results in memory.

`source_chunks` and `chunk_inference`, where retained, are final provenance and
artifact targets of this computation. They are not an intermediate database
scanned by a second planner to discover changes. Native component outputs carry
keyed changes to graph preparation. Pointer-only metadata uses 1-indexed source
bounds; keep only bounded hydrated text. The final chunk vector has one float32
SQLite copy, rather than additional JSON arrays or duplicate inference columns.

For frozen input, use supplied text exactly. Missing input raises; never resolve
it from a mutable file by using `supplied_text or hydrate_from_disk(...)`.
Identical text at two source locations retains distinct occurrence provenance.
No train/validation labels, benchmark presets or gold-data rules enter production
indexing or semantic keys.

## 3. Revision identity, memo and target isolation

Stable occurrence identity is independent of the 64-chunk batch. Content/pointer,
parser/chunker and semantic profiles identify desired revisions. Replacement
under the same chunk ID is a change; ID set difference is insufficient. Source
rename behavior follows the connector key: preserve identity where guaranteed,
otherwise retire/insert explicitly without losing shared graph support.

Keep expensive pure inference memoized through native components with exact
source/text and program/model/profile keys. Separate extraction and embedding
identities so changing one does not unnecessarily invalidate the other. Memo hits
retain named DSPy traces but consume no new provider tokens.

Point `coco.Settings(db_path=...)` at persistent state independent of transient
experiment DB basenames. One database's target ownership/tracking must never be
reused as another database's applied state. Compatible pure inference memo may
be shared only with complete dependency keys and explicit environment support;
verify that capability before selecting a cache layout. Sharing an environment
path is not by itself proof that cross-database tracking is safe. Do not add a
parallel hand-rolled cache if the SDK cannot separate those concerns.

## 4. Provider and publication boundaries

Native DSPy remains the interface. Default LLM admission is `Semaphore(1)` for the
local single-slot service; embeddings batch at most 64 inputs. Preserve configured
request-size/context limits; batch size is not a license to overflow a request.
Sampling temperature is >=0.3, default 0.35, reflection up to 0.8. Transient model
retries belong only to `dspy.LM(num_retries=...)`; exhausted/service failures abort
with the original exception. Do not reduce retries or skip work for throughput.

Final graph tables, vectors, FTS and applied source checkpoints are declarative
CocoIndex targets with the atomic grouping capability required by the graph plan.
No provider call runs in a target transaction. No read lock spans inference.
Partial source discovery/parser failure cannot retire unseen valid sources.
Native memo/tracking durability and `yaw.db` commit are separate; target actions
must replay idempotently after a crash.

No production in-memory SQLite graph staging, graph cloning, full survivor target
envelope or imperative graph sink remains after cutover. Before replacing the
existing atomic sink, prove the locked connector's public grouping and validation
capabilities. An independent transaction for each table does not meet this gate.

## 5. Partial publication, readiness and lifecycle

A valid first64 publication is queryable as a partial graph. It is not a declaration
that a 128-chunk document is fully indexed or semantically isolated. Readiness
requires completed source enumeration, every desired occurrence revision applied,
all affected semantic work committed and zero failed native components. Later
chunks resolve against earlier senses/facts and can reopen earlier decisions.

Track progress against source revision and semantic watermark using native source/
tracking state plus committed graph metadata, not a second work queue. On edit,
the previous document revision no longer establishes readiness. Graph snapshots
used for benchmark/candidate evaluation bind a fixed completed watermark; readers
must not silently evaluate whichever partial graph happens to be visible.

Handle live and frozen work through one ordered runtime lifecycle. Stop source
admission, drain provider/semantic publication work, finish live handles, then
close environment/database. Timeout alone is not proof the worker stopped. Do
not close the DB while a worker still has access. Reload signals follow native
filesystem events, not timer polling or private cancellation APIs.

Historical 1.0.26 tests observed that root update completion could coexist with
failed mounted components and that catch-up could terminate a live source.
These are earlier-version observations, not current locked-version acceptance.
Revalidate public handle result/error statistics, `watch`/`result` consumption and
shutdown on the lockfile version. Never consume one final handle twice or assume
readiness from a returned root coroutine alone. Verify managed-connection reader
admission; WAL and cached stats do not prove concurrent live reads.

## 6. Integration path and restart record

Follow G0-G6 in the graph plan. Native lifecycle integration adds these checks:

- [ ] Live filesystem and frozen/candidate indexing use identical component and
  semantic keys, with the same final target owner rules.
- [ ] Runtime, `snapshot_dataset_bundle`, benchmark and optimizer candidate calls
  await a completed semantic watermark and preserve named traces/task scores.
- [ ] Retire engine-owned `index_chunks` inference/writes, query-triggered
  preparation and manual eight-chunk loops with all callers, once the native
  replacement is exercised. Do not delete loops while their callee still owns
  essential inference behavior.
- [ ] Prove all cases in the graph paired harness: first64 progress, second64
  readiness, final-chunk bridge, changed old evidence and restart between batches.
- [ ] Exercise actual Windows create/edit/rapid-edit/rename/delete, malformed
  replacement, partial enumeration, reload, stale publication and drain/shutdown.
- [ ] Verify fresh-process crash/replay, loopback provider accounting and actual
  resident-model behavior separately; fake providers prove orchestration only.
- [ ] Pass mandatory project checks, paired time/count/semantic gates, <=1-GiB
  worker memory and Epic 1 Tier 1/Tier 2 gates before claiming acceptance.

Keep source files and experiment DBs. Generated indexes/evidence belong in ignored
`outputs/` or temporary directories. Inspect host-visible PIDs before an authorized
live restart; disk edits do not reload optimize/watchdog/daemon. Verify new
schema/profile and artifact timestamps/data after restart. No process action is
part of this documentation revision.

## 7. Existing evidence and adjacent work

Earlier work records native source/artifact replacement/deletion tests, loopback
memo/batching and trace checks, atomic full-graph target tests, and Windows
lifecycle cases. They demonstrate specific orchestration behavior only; the old
in-memory/full-graph path and its checkmarks cannot pass this revision's gates.
Record new command/version/evidence artifacts before checking off cutover items.

Dataset snapshot cleanup remains a separate consumer task: keep production
preparation dataset-agnostic; resolve baselines/source-field metadata in the
configured dataset registry; do not invent gold passages from keyword overlap;
QA-only datasets omit unsupported Recall@N rather than reporting zero. Preserve
actual baselines and optimizer checkpoints; never rewrite them to pass the new
paired graph comparison. The graph plan's project-gate section governs commands,
resolved baseline artifacts and reported Recall@N/EM/F1 evidence.
