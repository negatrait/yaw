# Native CocoIndex indexing and live ingestion

Status: revised target contract, 2026-11-07. Earlier acceptance on 2026-10-28 was
reopened for lifecycle work on 2026-11-04; this rewrite does not close those gates. Pinned code: `c6fa080`.

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [daemon.md](../.agents/rules/daemon.md): `yaw/daemon/*.py`, `yaw/daemon/ingestion/*.py`; [graph-storage.md](../.agents/rules/graph-storage.md): `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py`; [inference.md](../.agents/rules/inference.md): `yaw/daemon/graph/inference/*.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/corpus.py`, `yaw/daemon/ingestion/chunker.py`, `yaw/daemon/graph/storage/hydration.py` |
| Peers | [native-cocoindex-graph.md](native-cocoindex-graph.md): owns graph identity, `publish_graph` publication semantics, SHACL healing, and the paired `64 + 64` vs. `128` acceptance gate; [w3c-improvements.md](w3c-improvements.md): constrains SKOS/OWL/SHACL closed-world shapes and Active EDC label canonicalization (`§3.2`); [epistemic-confidence-spec.md](epistemic-confidence-spec.md): constrains Bayesian confidence pooling; [dspy-improvements.md](dspy-improvements.md): constrains GBNF adapter and RLM healer architecture |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md): anchors HippoRAG 2 indexing and retrieval parity; [model-evaluation.md](model-evaluation.md): anchors local `llama.cpp` serving constraints (`parallel = 1` `[target]`) |
| Supersedes / superseded by | Supersedes earlier `skos:ConceptScheme` indexing hooks and unscoped per-batch re-embedding / label re-canonicalization during `64`-chunk micro-batch updates |
| Records | [incremental-publication-work.md](reviews/incremental-publication-work.md), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [resource-vocabulary-boundary.md](reviews/resource-vocabulary-boundary.md), [evaluation-ingestion.md](evaluation-ingestion.md), [evaluation-graph-storage.md](evaluation-graph-storage.md) |

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

Frozen admission must retain unchanged occurrence/revision declarations while
admitting changed chunks. Never replay a smaller prefix of an already completed
snapshot and thereby withdraw valid chunks. Reopening the native tracking state
must preserve this rule. Revision matching includes source pointer/AST bounds,
source hash and extraction/embedding identities; matching IDs alone is insufficient.
`index_chunks()` and the raw-document `index()` adapter receive complete desired
snapshots, including empty input for complete withdrawal. Callers retain surviving
inputs explicitly; the engine never silently unions new inputs with a previous
snapshot. Removal uses the same native declarations, without an imperative delete
API. Resume and progress readers use `chunk_inference` exclusively; no legacy
OpenIE cache or legacy fingerprint metadata can authorize reuse.
Implementation slices and remaining cutover work are tracked in
[atomic incremental publication units](reviews/incremental-publication-work.md).

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
The complete frozen source declaration owns liveness during publication; retired
child rows can remain visible until CocoIndex finishes parent cleanup and must
not keep withdrawn occurrences alive for another update. Require an inference
artifact for every declared source and fail on incomplete coverage. Native
filesystem catch-up must likewise publish the surviving source set in that update.
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
For each `64`-chunk micro-batch publication (`publish_graph` `@coco.fn(memo=True, version=8)`),
per-batch work is delta-scoped to avoid reprocessing unmodified prior-batch state:
- `heal_shacl_violations` (`@coco.fn(memo=True, version=4)`) receives focus-isolated
  `SHACLHealInput` payloads grouped by `focus_node` (`focus_node ASC`) with
  deterministically ordered SQLite dossiers (`inspect_entity_or_fact`,
  `inspect_focus_node`), so newly introduced violations in Batch $k$ do not shift
  slice boundaries or invalidate memo entries for earlier focus nodes.
- `canonicalize_fact_definitions(conn, prior_definitions=prior_fact_definitions)`
  preserves valid prior-batch `set_definition` repairs and `apply_shacl_repairs`
  records `set_pref_label` with `chunk_id = "__healed__"` so healed facts and
  entity labels do not relapse into SHACL violations on subsequent `64`-chunk
  batches ([native-cocoindex-graph.md §5.2, §5.3](native-cocoindex-graph.md#52-batch-independent-identity-and-candidate-discovery)).
- `plan_graph_publication` initializes `fact_nodes` with cased endpoint names,
  passes `fact_vectors=()` on initial staging, runs delta-only
  `canonicalize_fact_text`, scopes `canonicalize_entity_labels_edc` to
  `entity_ids=sorted(touched_entity_ids)`, and embeds/upserts `fact_vec` in a
  single post-canonicalization pass at step 5 for `reembed_facts` only.

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

Coherent graph rows and their applied checkpoint belong to one producing
component and one shared transactional `TargetActionSink`. The schema-driven
connector consumes typed row declarations and validates the SQLite post-state
before commit. Independent sinks are not a transaction group. The chosen public
boundary and remaining capability gates are specified in
[graph publication §6.1](native-cocoindex-graph.md#61-atomic-declarative-publication-capability-gate).

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

The remaining caller-to-callee cleanup inventory and its gates are in the graph
plan [§9.1](native-cocoindex-graph.md#91-remaining-cleanup-after-the-614-line-retirement).
Complete C1–C4 without removing active candidate checkpoints or source evidence;
the per-chunk `chunk_vec` ownership/replay removal remains G3/G4/U5 work.
The implementation performance inventory and corrective gates are in graph plan
[§9.2](native-cocoindex-graph.md#92-performance-review-measured-implementation-units):
measure candidate scans, discarded survivor hydration, clone work and per-query
topology construction. Dense all-pairs buffers and a faster `:memory:` clone do
not satisfy the bounded owner-based architecture or its <=1-GiB worker budget.

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

## Decisions

- 2026-11-06: "Step 4 should be included in this unit, but obviously, you can implement one unit at a time according to whichever spec you write into @[docs/native-cocoindex-graph.md] or @[docs/native-cocoindex-indexing.md] , also sounds like you'll need to update our rules and tests to prevent these kinds of failures surfacing again." (User). Specs are promoted into `docs/native-cocoindex-graph.md` and `docs/native-cocoindex-indexing.md` and executed one checkpoint unit at a time with regression tests.
- 2026-11-07: "Audit and review for root cause for batch indexing regressing back to exponential time behavior. Incremental indexing and graph publication was designed and tested to be near-constant time for 64 + 64 chunk batches vs 128 chunk batchless indexing." / "It’s a SHACL Violation CONSISTENCY_ERROR" / "you will need to /spec first." (User). Enforces delta-scoped `64`-chunk batch publication (`publish_graph` `version=8`, focus-isolated `heal_shacl_violations` `version=4`, repair-preserving `canonicalize_fact_definitions`, `"__healed__"` label provenance, `touched_entity_ids`-scoped Active EDC, delta-only `canonicalize_fact_text`, and single-pass post-canonicalization `reembed_facts` embedding).
