# Early Epic 1 audit JSON pilot (`E1-P6`)

Status: planned 2026-11-09; implementation and measurements pending.
Owners: [Epic 1 tracking](../epic-1-hipporag-daemon.md#3-checklist-summary) and
[Epic 3 bounded exception](../epic-3-performance.md#early-epic-1-exception-audit-json-pilot-e1-p6).
Performance parent: Epic 3 probe `E3-P6`; results go to
[evaluation](../evaluation.md). This record is the restart checklist.

## Observable behavior and scope

The optimizer publishes the same audit fields and values at the same callback,
evaluation and finalization points. Watchdog consumes those snapshots with the
same candidate, frontier and evaluation displays. Publication remains atomic
through the existing text writer. JSON indentation remains two spaces; exact
whitespace and Unicode escaping are not identity contracts for this sidecar.

Bounded production changes:

- `yaw/daemon/graph/inference/optimizer.py`: incremental
  `_publish_optimization_audit` and final audit encoding only.
- `scripts/watchdog.py`: `_read_audit` decoding only; retain its `(mtime, size)`
  cache and measure unchanged-cache hits separately from changed snapshots.
- `pyproject.toml` and `uv.lock`: promote the already locked msgspec 0.22.0 to a
  direct dependency without unrelated dependency upgrades.

Use reusable msgspec JSON encoder/decoder instances and existing dictionaries.
No Struct/domain-type migration, generic serde framework, new writer, fallback
codec, or production feature flag. Keep both audit publication sites on the
same encoding contract with the smallest local implementation. Include
formatting and bytes-to-text conversion in measured encoding costs.

Deferred: `chunk_inference`, graph publication fingerprints, persisted DSPy
trails, optimizer resume/checkpoint encoding, DSPy `.save()`/`.load()`, runtime
status/tool responses and all other watchdog JSON files. No live optimizer or
watchdog restart and no writes to existing `.yaw/` artifacts during this pilot.

## Verified starting point and dependencies

- Research baseline: `dev`, commit `bf383c4`. Re-pin the implementation baseline
  immediately before work because unrelated CocoIndex changes are in progress.
- Python 3.14.9 and msgspec 0.22.0 were verified by read-only imports during the
  preceding research. Reconfirm before benchmarking; lock presence alone does
  not establish installation.
- The incremental writer runs on GEPA progress publication and after recorded
  evaluations; a separate final writer emits the same audit envelope.
- Watchdog already caches parsed audits; decoder gains apply to changed files.
- Existing incremental publication suppresses OSError; watchdog retains cached
  snapshots for some read/parse failures. Do not add suppression or copy these
  behaviors into new code. Tests must establish that serialization contract
  errors escape. Any broader failure-policy correction requires explicit scope
  reconciliation before implementation.
- Preserve all unrelated working-tree edits. This pilot does not depend on the
  CocoIndex N2-N5 cutover and must not modify those files or their checkpoints.

## Compatibility contract

Compare old and new decoded trees recursively, including numeric types and
values, list order, Unicode, escaping, nulls and empty containers. Test the
existing audit envelope: `graph_rlm_stats`, `candidate_records`, `eval_records`
and `gepa_progress`. Keep the 500-evaluation bound and all payload construction
unchanged. Existing stdlib-written finite audit files remain readable.

Before switching encoders, inventory actual producer value types. The intended
wire domain is dictionaries with string keys, lists, strings, booleans, nulls,
integers and finite floats. Unsupported objects and nonfinite floats must raise
before replacing the destination; no `default=str`, dropped fields or conversion
of NaN/infinity to null. msgspec's nonfinite encoding needs an explicit guard or
producer validation; select the smallest verified implementation and include
its full cost in the benchmark. If existing producers legitimately emit values
outside this domain, stop and reconcile that contract before substitution.

Watchdog must handle malformed bytes through its existing parse-error policy,
using the appropriate msgspec exception rather than a fallback decoder. Verify
that a failed publication leaves the previously valid file intact.

## Ordered implementation and evidence checklist

- [x] P0: record scope, owner exception and acceptance gates before implementation.
- [ ] P1: re-pin HEAD, record working-tree state, verify imports, freeze payloads
  and build the stdlib baseline before changing the codec.
- [ ] P2: add meaningful real-file compatibility tests and implement the bounded
  producer/consumer replacement; promote the direct dependency.
- [ ] P3: run paired serde, atomic-file and watchdog measurements; retain or
  reject against the predefined gates below.
- [ ] P4: run quality gates, record results and limitations here and in the
  evaluation ledger, then commit only this unit with a conventional message.

P1 artifacts live under ignored `outputs/benchmarks/msgspec_audit_pilot/`:
`manifest.json` records commit, dirty paths/diff identity, Python/msgspec versions,
host, payload hashes, benchmark commands and settings; `payloads/` contains frozen
inputs; `results.json` and `summary.md` contain raw paired timings and conclusions.
The benchmark implementation is a bounded reproducible script under `scripts/`;
it must not become an alternate production serialization path.

## Predefined experiment and retention gates

1. Freeze a copy of the available real audit and deterministic larger fixtures
   exercising 50 and 500 evaluation records, candidate summaries, frontier
   history and graph statistics. Keep the real and constructed fixture results
   separate; record actual serialized sizes. Boundary fixtures include Finnish
   text, non-BMP Unicode, quotes/newlines, empty values, malformed JSON,
   unsupported objects and nonfinite floats.
2. Use identical immutable payloads for both implementations. Measure encode
   (including validation, two-space formatting and UTF-8 text conversion), decode
   of the same frozen bytes, and the combined serde cost. Warm each implementation,
   alternate order over at least 30 paired samples, and batch repetitions enough
   to exceed timer noise. Fix repetition counts after baseline calibration,
   before inspecting candidate results. Report medians, p95, paired ratios and
   absolute milliseconds. Fixture construction and imports are outside timing.
3. Retain only if the median combined serde ratio is strictly below 0.5 on each
   representative fixture, with neither encode nor decode median regressing.
   This is a local application of `E3-P6`, not completion of that whole probe.
4. Separately measure full atomic write plus watchdog changed-file read on real
   temporary files, including stat, I/O, formatting and conversion. Require its
   median time ratio below 1.0 on each representative fixture. Measure warm
   unchanged-file cache hits too; require no statistically supported regression.
   Report same-host warm-file results as such; do not claim cold-cache results
   without controlling the OS cache.
5. Measure peak process RSS in fresh isolated worker processes for both paths,
   including their codec imports and equivalent workload. Report absolute RSS,
   baseline RSS, peak allocations and payload sizes; require median peak RSS
   ratio at most 1.0 across repeated paired workers. An inconclusive/noisy gate
   remains pending; do not round it into a pass or loosen the threshold.
6. Real temporary-file tests must cover incremental and final publication,
   watchdog display inputs, cache reuse and refresh after atomic replacement,
   old finite JSON compatibility, invalid-input handling, and no replacement on
   serialization failure. Freeze timestamps or exclude only explicitly named
   volatile timestamp fields when comparing publication snapshots.
7. Run `uv sync --locked`, `uv run --locked ty check`,
   `uv run --locked ruff check --fix`, `uv run --locked ruff format`,
   `uv run --locked pytest --no-testmon`, and `git diff --check`. Inspect automatic
   fixes and preserve unrelated edits. Use existing optimization progress and
   watchdog suites plus focused tests; no LLM calls are needed for this boundary.

If any retention gate fails, remove the trial production changes and unused
direct dependency; preserve benchmark evidence and explain rejection. Do not
retain a dormant flag or alternate codec. Report whether evidence supports only
local CPU savings or an actual file-pipeline improvement; it cannot establish
overall optimization speed, retrieval quality, daemon acceptance, live process
cutover, or Epic 1/3 completion.

## Planning result

P0 records an authorized plan only. No production code, dependencies, saved
audit artifacts or running processes were changed. No performance or memory
result has been measured. Implementation gates P1-P4 remain unchecked.

Planning verification: `uv run --locked --no-sync pytest
tests/core/test_architecture.py --no-testmon -q` passed (`16 passed in 7.01s`);
`git diff --check` passed. The architecture result covers the current working
tree, including pre-existing unrelated edits; it does not verify those edits as
part of this pilot. Full implementation quality gates have not been run here.
