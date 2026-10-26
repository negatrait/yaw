# Complete production coverage

Target: 100% executable-line coverage with meaningful behavioral assertions, no
excluded production lines, no skipped cases and no testmon deselection in the
final gate. Preserve the existing clean Ruff/ty gates. The current report covers
yaw/ at 81.75% (8,168 / 9,992 lines). The final scope includes production
Python in yaw/, benchmark.py and executable scripts. Current incremental figures
initially covered yaw/ only. The combined development report now measures all
36 production files: 10,589 / 12,211 lines = 86.7169%, with 1,622 missing and zero
excluded. No complete-suite 100% acceptance result exists yet.

This is verification work within the first unfinished Epic 1 packet, E1-S6.
It does not establish Tier 2 corpus parity or a model-quality improvement.

Work order and restart checklist:

- [x] Core configuration, logging, data contracts and entry points (incremental verification).
- [ ] Parsing, embeddings, reflection validation and inference failures.
- [ ] Real SQLite migrations, hydration and graph edge cases.
- [ ] Runtime lifecycle, CLI process management and optimizer recovery.
- [ ] Executable-script scope and actual cross-platform paths.
- [ ] Full clean static/test gate and exact coverage audit.

Tests use real files, SQLite, sparse matrices, native DSPy and loopback HTTP.
Coverage from actual subprocess execution must be measured correctly; do not
count copied or modified fixture implementations as production execution.
Platform-specific behavior needs a real applicable platform or a justified
runtime contract refactor, not a fake operating-system label.

Baseline artifacts: outputs/preflight/20261023T164646Z-31068/coverage.json.
Prior quality changes are already present in the worktree and must be preserved.

Progress on 2026-10-23: real subprocess interruption/hard-kill/report tests,
parser provenance/null/homograph cases, real sparse matrix boundaries, SQLite
codec corruption, loopback registry identity, async embeddings and disk/AST
hydration are implemented. Core, parsers, graph candidate cache, PPR and model
registry measured 100% in accumulated development data. This is not final
acceptance; all source must be remeasured in a fresh complete suite.

Development data: outputs/coverage-work/.coverage and all-production-coverage.json.
The local
source-hash helper purges changed production files before append runs to avoid
attributing old line numbers to edited implementations. Final acceptance must
not rely on accumulated data, test deselection, exclusions or rounding to 100%.
Use all source flags for development append runs:
--cov=yaw --cov=benchmark --cov=scripts. The former yaw-only coverage.json and
scripts-coverage.json are retained as scope-specific historical measurements.

Next packet: remaining engine indexing normalization, optimizer/program state,
remaining SQLite integrity repairs, and runtime/CLI.
Dashboard and parity report rendering now have real SVG/audit tests; subprocess
downloader tests verify cached operation without external network. See evaluation
section 41 for the completed 60-case regression packet and actual reporting fixes.
Section 42 records native HTTP/SQLite checkpoint replay after stage-cache
eviction and complete reflection-module line coverage in development data.
Section 43 records complete benchmark orchestration against loopback HTTP,
durable success/failure reports, input and embedding verification cases, and
the numeric answer-alias type-contract repair. Remaining large groups are
program.py (304 lines), engine.py (298), watchdog.py (311), CLI/runtime, storage
integrity handling and genuine platform/error paths.
