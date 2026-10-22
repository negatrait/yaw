# CocoIndex audit of DSPy optimization

Date: 2026-10-22. Scope: the `uv run --locked yaw dspy optimize` path,
its CocoIndex ingestion boundary, persistent inference caches, batching, and
memory growth. This is an audit; no runtime fixes or dependency changes were made.
Verified checkout: `C:\Users\Harri.Ahokas\Documents\yaw`, remote
`https://github.com/negatrait/yaw.git`. Installed versions: CocoIndex 1.0.25,
DSPy 3.4.1. Existing edits to `yaw.toml` and `yaw/core/config.py` were preserved.

## Outcome

Follow-up implementation on 2026-10-22: diagnostic retention is now bounded at
accounting boundaries, optimization suppresses DSPy inspection histories, and
candidate copies discard diagnostic history. Query-only prefix invalidation,
configurable embedding request limits, and parser-helper invalidation are also
addressed. The source-map reconstruction cost remains deferred. See evaluation
section 26 for measurements, checks, and remaining limits. The findings below
preserve the original audit snapshot rather than claiming all process RAM is
constant or that larger embedding requests have been hardware-benchmarked.

The pipeline has the right basic separation: CocoIndex produces source-pointer
changes, the runtime wires them to a single graph writer, and DSPy plus SQLite
stage caches reuse inference. Decorating every DSPy method with `coco.fn` would
not improve this path: those methods execute outside CocoIndex processing
components. Moving graph mutations into memoized calls would require an explicit
target-state design; cache hits do not replay arbitrary database side effects.

The following contained changes deserve priority before a long optimization run.
They are recommendations, not implemented changes or measured GPU speedups.

## Findings in recommended repair order

1. **Query-only prefix evolution invalidates unrelated cached stages (P2).**
   `engine.py:_candidate_cache_namespace` includes the entire mutable embedding
   profile in a namespace shared by embeddings, OpenIE, synonym decisions and
   SHACL repairs. A query-prefix-only change changes that namespace even when
   document encoding is identical. `_embed_fingerprint` likewise combines all
   prefixes, and `_extraction_fingerprint` includes query prefixes; the stale-target
   path can rebuild graph materialization. The isolated probe confirmed unchanged
   document prefix but changed shared namespace. OpenIE checkpoints and DSPy's
   response cache can still prevent some model calls, so this is unnecessary stage
   cache invalidation/rebuild work, not proof that every LLM request repeats.
   **Repair:** separate document-vector, query-vector, extraction and graph-stage
   dependency identities. Preserve source/model/endpoint/grammar dependencies;
   verify that query-only mutations reuse document vectors and OpenIE.

2. **Configured embedding batches are capped again inside the adapter (P2).**
   The engine uses `embed_batch_size=64`, but calls `encode` without passing its
   batch size. `_prepare_pieces_and_batches` defaults to 16 items and a hardcoded
   3,500 formatted-character budget. With 64 synthetic inputs, actual prepared
   batches were: 100 characters -> 4 batches, maximum 16 items; 500 characters ->
   11 batches, maximum 6; 1,200 characters -> 32 batches, maximum 2.
   Validation and candidate indexing also use outer groups of 8 passages, limiting
   chunk-vector batching further. Entity/fact counts can exceed that group size.
   **Repair:** make effective adapter batching honor configuration, with a separate
   explicit provider/context budget and existing overflow splitting retained.
   Benchmark larger budgets on the actual server before choosing defaults. Merely
   raising `embed_batch_size` or adding CocoIndex batching will not remove the
   character limit; larger batches are not automatically faster on this hardware.

3. **Cache replay repeatedly hashes the whole passage pool (P2).**
   Every successful OpenIE/synonym/SHACL stage-cache read evaluates `self.sources()`
   before decoding, even if the payload contains no source references.
   `engine.py:_candidate_cache_sources` reconstructs the digest-to-text map from
   the complete pool each time. Writes also rebuild the map and compare string
   fields against source spellings. A real-SQLite probe with 5,000 passages
   (5,068,890 source bytes) took 3.142 s for 100 warm reads, versus 0.061 s without
   source-map reconstruction. Both timings enabled tracemalloc; the first had
   639,729 bytes of peak traced allocation. This is a synthetic CPU microbenchmark,
   not a full-run speedup or process-memory measurement.
   **Repair:** reuse a source-reference resolver for a stable pool generation, or
   resolve only referenced digests. Invalidate it when pool contents change; do not
   introduce a resident dictionary of historical candidate results.

4. **Persistent ingestion does not track edits to plain parsing helpers (P2).**
   `process_corpus_file` is memoized, but `chunk_source_text` and its helper chain
   are ordinary Python functions. An isolated checkout was ingested, then only
   the helper's title calculation was edited. A fresh process using existing LMDB
   emitted zero changes; a fresh-state scan emitted `AUDIT_CHANGED_HELPER`.
   This can retain stale parsing/projection behavior after code changes. The
   optimizer's deliberately fresh scan avoids this particular failure, so this
   finding primarily concerns persistent daemon ingestion.
   **Repair:** track relevant pure parsing helpers and external behavior constants,
   with a one-time memo-key/version change to invalidate preexisting entries.
   Decorating new helpers alone does not retrofit dependencies into old memos.

5. **Disk-only stage caches do not establish bounded process RAM (P2).**
   The stage payload cache has a shared 256 MiB FIFO disk budget. However, active
   passage pools, captured traces, program telemetry/audit lists and predictor
   histories still live in RAM. `record_embedding_telemetry` and `_record_stage_telemetry`
   append records; `_persist_dspy_trails_to_db` slices what it saves but does not
   trim those in-memory lists. Candidate copies can carry program history too.
   **Repair:** measure process RSS over repeated candidates, then bound diagnostic
   history separately from traces required by GEPA and logical token accounting.
   Do not truncate lists blindly: current code uses lengths/slices to identify
   newly produced records. FIFO eviction also means old stage results may need
   recomputation; the cache is bounded reuse, not permanent retention.

## What is already configured sensibly

- `process_corpus_file` has `memo=True`, explicit parameters and `-> None`; its
  declared target states are the durable output. `corpus_app_main` uses
  `mount_each` over file keys; CocoIndex state is under `state_dir/coco_lmdb`.
- The target handler uses statediff and a compiler callback. Failure is raised
  before successful target-state acknowledgement; existing tests cover retry.
- CLI optimization starts with `auto_poll=False` and `sync_on_start=False`, avoiding
  an accidental background scan of the production corpus. Its supplied corpus
  slice uses temporary scan state to reconstruct all candidate pointers. Reusing
  a no-change incremental delta here would lose unchanged passages.
- OpenIE is checkpointed per chunk. Stage caches use typed disk payloads and
  restore traces and logical cost. Native DSPy response caching is enabled on
  disk during evolution with its in-memory cache disabled.
- Optimizer evaluation uses one thread and the engine serializes graph work.
  This fits the configured local server's single-slot setup. More concurrent
  extraction calls need hardware measurements and trace/transaction isolation;
  `coco.GPU` would not schedule an external inference server's GPU.

## Evidence and limits

Disposable probe: `outputs/cocoindex_optimize_audit.py`; captured measurements:
`outputs/cocoindex-optimize-audit.json`. Source mutation was confined to temporary
snapshots, and probe databases/LMDB state were temporary. No production corpus or
optimization database was modified by these probes; no inference-server calls
were made by them.

Focused regression command:

```powershell
uv run --locked pytest tests/daemon/ingestion/test_corpus.py tests/daemon/graph/test_candidate_cache.py tests/daemon/graph/storage/test_candidate_cache.py tests/daemon/graph/inference/test_response_cache.py -q --tb=short
```

Results are recorded in `docs/evaluation.md`. The HTTP tests use scripted loopback
responses and actual separate processes/storage; they verify orchestration and
request reuse, not model quality. No live GPU throughput, peak VRAM, full-run RSS,
Tier 1/Tier 2 retrieval parity, or power-loss durability is claimed by this audit.

Primary reference: [CocoIndex function tracking, memoization and batching](https://cocoindex.io/docs/programming_guide/function/).
The skill's batching shorthand must be read with the API reference: the async
batching decorator is `coco.fn.as_async(batching=True)`, not a blanket decorator
for synchronous DSPy methods.
