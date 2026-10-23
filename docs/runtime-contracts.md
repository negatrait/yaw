# Runtime contract repair

The first packet in the preflight audit repairs executable contracts rather than
suppressing type diagnostics. It includes the pending native reflection
implementation required by the repaired checkpoint boundaries.

- QA signature, demos and LM accessors delegate to DSPy's actual predictor.
  Optimization, deepcopy and JSON restore therefore see the same state.
- LM factories return the capability-declaring native subclass. Copies preserve
  recipe, context capacity and grammar capability metadata.
- RLM tool wrappers retain inspectable signatures and coroutine identity.
- SIMBA compilation narrows the returned module before using worker predictors.
- Restored SIMBA predictors and active reflection records are validated before
  adoption. Invalid versions, counters, schemas, phases and proposals fail
  explicitly; an invalid write does not mutate SQLite. Evidence references are
  checked against their original objects and paths.
- Reflection configuration and verified tokenizer access are checked before
  indexing can delete or rebuild rows.
- The assertion audit query returns explicit six-column tuples.

Regression coverage uses native DSPy classes, JSON files, real SQLite, loopback
HTTP and process-boundary recovery. Loopback responses establish orchestration
and recovery behavior, not model quality. This repair does not establish an
unattended stability guarantee or close the other preflight audit packets.

Verification: the targeted runtime files pass ty; the native DSPy/SQLite/recovery
and architecture suite passed 61 tests in both the workspace and an isolated
export of the staged commit. Four affected optimizer/hybrid checks also passed.
The final tool metadata adjustment passed both sync and async regressions.
New regression/reflection files pass Ruff; existing unrelated lint debt remains.
See [evaluation record](evaluation.md#36-runtime-contract-reliability-repair-2026-10-23).
