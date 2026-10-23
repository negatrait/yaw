# MIPROv2 zero-shot warm-up audit

Date: 2026-10-22. Installed DSPy: 3.4.1. Scope: audit of YAW's MIPROv2/hybrid
warm-up and handoff to GEPA. No runtime/configuration fixes were made by this
audit. Existing worktree changes were preserved.

## Conclusion

The current path does not implement a strict, low-overhead zero-shot Bayesian
warm-up. Zero demo limits are passed correctly, but they do not prevent native
DSPy from bootstrapping proposal examples. YAW also keeps its built-in task
demos during the search, clears them after selection, and gives the default
large-validation hybrid warm-up too few trials to leave TPE's random startup.
MIPRO's retained search archive is another memory cost at the GEPA handoff.

Current `yaw.toml` selects `mode = "gepa"`, so an ordinary invocation with that
configuration does not run MIPRO at all. The findings concern `--mode hybrid`,
`--mode miprov2`, configurations selecting those modes, or restored run options.
GEPA-only still starts with built-in demos: the MIPRO zero-demo cleanup is not a
global no-demo policy. Reflection also has its own always-on SIMBA optimizer,
whose demo policy is independent of the two MIPRO settings.

## Findings

### P1: Search and GEPA seed are scored under different demo regimes

`HippoRAGProgram(include_demos=True)` is the default, also used by `from_config`.
The audited seed has 15 built-in demos across its 10 predictors: NER 1, triples
1, triple filter 10, QA 2, SHACL action generation 1; the others have zero.

`engine.py:evolve_prompts` records `baseline_demos`, compiles MIPRO, and only
then clears demonstrations when both limits are zero. Native MIPRO sets its
generated `demo_candidates` to `None` before instruction search, but this does
not clear existing predictor demos. Its selection method simply refrains from
assigning demo sets when `demo_candidates` is absent. Thus inherited task demos
remain during baseline/trial evaluation, and YAW subsequently removes them from
the selected program before GEPA. Instruction selection can favor a different
prompt regime from the program that GEPA receives.

**Repair:** resolve the task-demo policy before initial indexing, checkpoint
fingerprinting and MIPRO evaluation; clear inherited demos before scoring when
strict zero-shot is selected. Re-evaluate the same instruction-only winner at
handoff. Treat reflection/code-tool demonstrations as a separate explicit policy.

### P1: Four warm-up trials never reach adaptive TPE sampling

For hybrid mode with a validation set of at least 25 examples, absent explicit
trial/candidate arguments, YAW overrides `auto` with `auto=None`, 3 instruction
candidates per predictor and 4 new trials. With the default 100 validation
examples, that path is taken even if `auto=medium` or `heavy` is selected.

Native MIPRO uses `TPESampler(seed=seed, multivariate=True)` without setting
`n_startup_trials`; installed Optuna defaults to 10. MIPRO inserts one evaluated
baseline trial, then runs the requested new trials. Four new trials therefore
remain entirely within random startup. The search has an instruction-only
categorical space, but it is not yet adaptive Bayesian exploration.

The separate explicit-budget branch has further traps:

- Supplying only `--num-trials` sets `num_candidates` to 1. YAW's proposer then
  returns only each predictor's existing instruction: there is no instruction
  alternative to explore.
- Supplying only `--num-candidates` sets the trial count to 1.
- The same `--num-trials` value becomes MIPRO's new-trial count and GEPA's metric
  call cap, which are different units. Native MIPRO's baseline and bootstrap/
  proposal calls are additional work, not included in that new-trial count.

**Repair:** use coherent, separate warm-up and GEPA budgets; never silently turn
the unspecified half of a search budget into 1. With native sampler defaults,
allow enough completed trials for adaptive sampling, or explicitly describe a
smaller warm-up as random candidate screening. A strict small Bayesian budget
would need an intentional sampler/startup policy; MIPRO does not expose a public
sampler argument in its installed constructor.

### P1: MIPRO search archives propagate into GEPA candidate copies

MIPRO defaults `track_stats=True`; YAW does not override it. The native optimizer
attaches `trial_logs`, `candidate_programs` and `mb_candidate_programs` containing
program objects to the winner. YAW hands this object directly to GEPA.
`HippoRAGProgram.__deepcopy__` preserves LM clients/hooks but copies other fields,
including these archives. The diagnostic-retention fix does not remove optimizer
search archives; they are distinct from diagnostic lists.

The loopback probe produced five archived programs from baseline plus four
trials. Copying their archive through a YAW program created independent copies
of both the archive and the contained programs. This establishes retained-object
multiplication, not an estimate of full-run RSS or a proven sole cause of crashes.

**Repair:** pass `track_stats=False` for the warm-up if those archives are not
required, or persist compact summaries separately and remove program archives
before GEPA. Preserve the winning predictor state and score; do not carry a
full warm-up search history into each evolutionary candidate.

### P2: Zero limits still pay native bootstrap costs

DSPy 3.4.1's `_bootstrap_fewshot_examples` detects zero-shot and substitutes
3 bootstrapped examples and 0 labeled examples for instruction-proposal context.
`compile` calls this stage unconditionally, before instruction proposal, and
discards its demo candidate space only afterward. Merely setting
`fewshot_aware_proposer=False` does not skip the bootstrap call.

This is native DSPy behavior, not failure to pass YAW's zero settings. The probe
used zero limits and three candidate sets; it still made four provider requests
and produced sets of sizes `[0, 1, 3]`.

YAW replaces `_propose_instructions` with its RLM proposer. That proposer can
already fall back to training-record context when `demo_candidates=None`, so
requiring native bootstrap examples is not intrinsic to this custom path.

**Repair:** if the intended policy forbids bootstrap-generation overhead, use a
small, version-checked MIPRO adapter/subclass to skip that stage in zero-shot
mode while retaining native instruction-combination search. Do not globally
change DSPy constants or promise that the two demo limits alone skip this work.

### P2: Additional reflection optimization has a separate demo budget

`RLMInstructionProposer` defaults to `enable_simba=True`, `simba_bsize=2`,
`simba_max_demos=2`, and seeded reflection/action examples. GEPA reflection can
therefore launch SIMBA training even when task-level MIPRO demo limits are zero,
including in GEPA-only mode. These examples train the instruction proposer, not
the deployed task program, but their calls/prompt cost are real. The MIPRO custom
proposal method calls `propose_for_component` directly rather than the outer
GEPA ProposalFn path that accumulates SIMBA training examples.

**Repair:** label optimizer stages clearly in progress output and define a
separate reflection-training policy/budget. Changing the current always-on SIMBA
rule is an architecture decision, not something implied by task-demo limits.

## Verification

`uv run --locked python outputs/audit_mipro.py` passed. It exercises installed
native bootstrap and instruction-combination stages with real DSPy evaluation,
Optuna, and scripted loopback HTTP. Measurements are in `outputs/mipro-audit.json`:

- Zero limits: 4 bootstrap provider requests; generated set sizes `[0, 1, 3]`.
- Explicit instruction-only search: inherited demo count 1 during each of its
  five evaluations, also 1 in the selected winner.
- Default TPE startup: 10 completed/pruned trials; hybrid warm-up: 4 new trials
  plus its baseline.
- Five archived program records; YAW deepcopy retained independent archive and
  program objects.
- Native `auto=light` zero-shot sizing for YAW's ten predictors would request
  51 new trials, illustrating how different the four-trial hybrid shortcut is.

The probe supplies instruction candidates to the native search directly; it does
not run YAW's full RLM proposer, a full CLI optimization, a GPU benchmark, a
multi-hour RSS experiment, or a held-out quality comparison. It uses private
installed methods for diagnosis, not as a proposed public integration contract.

Primary sources: [DSPy MIPROv2](https://dspy.ai/current/api/optimizers/MIPROv2/),
[DSPy implementation](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py).
Installed source, rather than potentially newer online documentation, controlled
the version-specific findings.
