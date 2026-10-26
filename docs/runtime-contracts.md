# Runtime contract repair

## Native DSPy binding and parsing (2026-10-24)

The per-predictor BaseLM proxy is removed. Task predictors bind the actual
extraction or QA LM; unbound predictors follow DSPy's context. Native per-call
overrides and LM copies determine the actual request settings. Candidate copies
retain their predictor LMs and settings. Native async Predict calls use the LM's
async execution path. Reflection writer/compressor predictors are deliberately
unbound and run under their worker/rollout context, including the verified budget
guard. The evidence-selector RLM binds its configured orchestrator.

Fresh task and reflection programs default to zero demos. Existing example sets
remain available with `include_demos=True`; they demonstrate schemas but have no
measured generalization guarantee. Loading a compiled program preserves its
saved demos, including empty lists. Explicit deployment clients are retained
while loading instructions and demos, before reconstructing Pareto members;
compiled prompt files do not replace supplied endpoints or credentials.
Serving context and completion capacities also remain deployment-owned on all
prompt-state loads, including hot reload and old Pareto members. Omitted budget
settings derive from context capacity; the 128K defaults are described in README.

The custom JSON recovery adapter and ReAct field-renaming shim are removed.
`YawGBNFAdapter` subclasses native `dspy.JSONAdapter` and only injects grammar.
DSPy's parser handles syntax and enforces complete typed output fields; missing
answers, wrong field names and invalid output types fail with its normal error.
No extra parser exists to convert malformed predictor output into a successful
empty result. `dspy.History` remains an explicit conversation input; the current
task API is stateless and keeps diagnostic history separate.

## Earlier contract repair

The first packet in the preflight audit repairs executable contracts rather than
suppressing type diagnostics. It includes the pending native reflection
implementation required by the repaired checkpoint boundaries.

- QA uses an unmodified native `dspy.ChainOfThought`. Signature, demos and LM
  state belong to `qa.predict`; callers use that predictor directly. The temporary
  delegating accessor subclass has been removed.
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
