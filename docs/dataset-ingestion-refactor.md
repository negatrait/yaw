# Dataset and ingestion replacement

Accepted 2026-10-27. Work on `dev`; breaking layout/API change, fresh indexes,
no migration or compatibility wrappers. Preserve source inputs and experiments.

## Restart checklist

- [x] Verify checkout/remote and reconcile governing documents.
- [x] Add typed bundles, faithful per-document preparation and manifest validation.
- [x] Move Enterprise quota selection to the evaluation loader.
- [x] Replace parser heuristics with strict generic projection and shared construction.
- [x] Return native CocoIndex memo results in complete persistent snapshots.
- [x] Refactor CLI/runtime/benchmark consumers and remove retired APIs/settings.
- [x] Replace obsolete tests; verify files, SQLite, restart and loopback invariants.
- [x] Measure configured Enterprise corpus and document actual checks/failures.

## Contract

`outputs/datasets/<dataset>/{documents/*.json,evaluation/questions.json,manifest.json}`.
Only documents are ingested. Copy original source fields and QA values unchanged.
Export the complete configured corpus, including distractors. Stable opaque paths,
content hashes and original IDs live in the manifest. Serialize preparation and
reads with a bundle lock; stage writes, preserve unchanged files, publish manifest
last and reject incomplete exports. Enterprise retains seed 42 and existing
150-train/100-validation quotas in the loader, with disjoint partitions. Smaller
optimization slices sample within those partitions; an offset applies within each.

`CorpusIngestor.poll()` returns deltas; `snapshot()` returns all successful current
chunk results, including native memo hits. Use one source AST and at most four
inflight file components. JSON projection and hydration must respect the same
source bounds; unsafe spans fail. Parsing/splitting failures must reach the host
and remain retryable. Partial scans cannot establish source deletion.

## Removal gate

Remove downloader normalization, alias synthesis, QA-based document sampling and
migration branches; monolithic discovery/reformatting; `resolve_dataset_source_files`,
`sync_source_files_for_docs`, `extract_sample_docs_and_gold`, `_load_corpus_by_id`,
`_ensure_multiline_json_file`, benchmark projection constants and injected defaults;
state deletion, passage-string filtering and root-widening fallback; runtime
Hotpot/LV-Eval/NarrativeQA branches; JSON repair and silent parser fallbacks.
Move quota policy, do not delete it. Keep production key projection, dataset
evaluation adapters, aliases, scope extraction and graph checkpoints/caches.

## Acceptance and limitations

Use CocoIndex and engineering-real-tests skills. Test all registered schemas,
byte/value fidelity, full warm snapshots across processes, single edits and
deletions, source/helper/settings invalidation, strict errors and retry, real
hydration isolation, actual SQLite and loopback call counts. Measure preparation,
cold/warm scans, snapshots, edits and memory on the configured Enterprise corpus.
Run focused tests, Ruff, ty and diff checks; record testmon selection and blockers.
No unattended optimization. Do not call loopback orchestration a quality gain.

## Native API findings

CocoIndex 1.0.26 returns the root function result from `App.update_blocking()`;
`use_mount()` replays a memoized per-file result. `mount_each()` is not a snapshot
enumerator. The implementation uses stable `use_mount` paths through `coco.map`
with `max_inflight_components=4` and returns one complete tuple.

Native structural `CodePattern` matching does not support Markdown. Markdown
heading labels therefore use one deterministic fence-aware line pass, while
chunk boundaries use CocoIndex's native splitter. Public `render_match` exposes
frames around a ranges' common envelope; batching unrelated scopes drops their
headers. Code scope inspection instead selects source-grounded indentation/brace
anchor matches and shares the native AST across those queries, without accessing
private CocoIndex APIs. Both Python and unindented C scope
preservation have real-source regression tests. These are explicit algorithms,
not exception-triggered fallback paths.

Compact multi-record JSON arrays and partially excluded nested objects that
cannot retain their enclosing identity in a single safe pointer are rejected.
Prepared document objects and JSONL records retain actual source coordinates
even when their string values contain escaped newlines.

## Delivered verification

See [evaluation](evaluation.md#dataset-bundles-and-persistent-ingestion-2026-10-27)
for the removal inventory, actual test selections and configured-corpus measurements.
The full affected selection passed 274 tests. Warm Enterprise snapshots contain
all 9,062 chunks from 1,308 documents with zero reparsing; a single-document edit
reparses one file. These are scanning/orchestration results, not task-quality or
end-to-end model-throughput measurements.
