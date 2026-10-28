# Bounded indexing inference overlap

Accepted 2026-10-28. One embedding lane overlaps chunk embeddings with sequential
OpenIE in the current batch. SQLite writes, cache publication and telemetry remain
on the engine owner. No speculative future batches or concurrent chat requests.

## Restart checklist

- [x] Verify compact-synonym mandatory grammar and affected graph regressions.
- [x] Separate embedding preparation, provider execution and owner publication.
- [x] Add opt-in scheduling without semantic fingerprint changes.
- [x] Verify real HTTP overlap, SQLite recovery, failures and bounded retention.
- [x] Run Ruff, ty, affected tests and architecture checks.
- [x] Report the authorized pilot attempt, its failure and elapsed deadline.
- [ ] Complete a repaired paired live comparison before considering default enablement.

Public indexing interfaces and the eight-chunk optimization commit cadence remain
unchanged. Entity/fact vectors follow batch-wide sense resolution. Synonyms and
SHACL remain ordered. Preserve existing cache keys, pending-delta replay and source
coordinates. Keep embedding limits at 64 inputs and 3,500 formatted characters.

On failure, stop scheduling, drain in-flight requests within provider timeouts,
publish independently valid completed cache results and propagate the error.
Cancelling a future does not cancel an in-flight HTTP request.

`runtime.indexing_overlap` defaults to false pending measured benefit. The live
pilot uses disposable identical starting databases and separate response caches,
two eight-chunk batches per arm with reversed execution order for the second pair.
Do not run alongside optimization or launch GEPA. Report stage/total timings,
request counts, graph differences, peak RAM and available GPU memory observations
in evaluation.md. An incomplete pilot does not establish a speedup.

## Use

Enable explicitly with `uv run --locked yaw config set runtime.indexing_overlap true`.
Set it to `false` for serial execution. Restart the process for changed settings
to take effect. Toggling this setting alone preserves optimization resume identity.
No worker writes to SQLite or publishes program telemetry. At failure/shutdown,
the owner drains the active request and does not schedule additional micro-batches.

The SDK's resources are initialized before launching the worker to avoid concurrent
lazy imports on a cold process. See evaluation.md for the discovered live failure,
the fresh-process regression and why the default remains false.
