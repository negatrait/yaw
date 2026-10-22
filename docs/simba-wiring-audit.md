# SIMBA wiring verification

2026-10-22, DSPy 3.4.1. Scope: verify the existing reflection-worker optimizer,
not change its architecture or measure downstream benchmark lift.

## Verdict

SIMBA is mechanically connected correctly. GEPA's proposal callback accumulates
reflection examples; at two examples it runs native SIMBA over the instruction
proposer/compressor worker. Winner signatures and demos are copied back, and
the subsequent worker call uses them. It optimizes the reflection worker, not
YAW's retrieval/QA program directly.

`outputs/audit_simba.py` exercised native SIMBA with real DSPy and loopback HTTP.
Two samples per example produced quality variation; native rule generation
created a distinguishable winning instruction. The wrapper adopted it and the
next worker request included that rule and returned its intended output.
Results are saved in `outputs/simba-audit.json`:

- Native compilation completed; 14 training HTTP requests plus 1 verification
  request. The adopted instruction changed; this deterministic run selected rules
  and added no demonstrations.
- All four trajectory requests used temperature 1.0, confirming that SIMBA's
  resampled LM settings reach the actual worker call.
- The worker and outer proposer reference the same adopted predictor, and the
  next request used the winning rule.
- Existing integration contract: 1 passed, 32 deselected in 22.77 seconds.

## Limits and findings

1. **The reward is a format/generalization proxy, not measured task improvement.**
   `_score_proposer_prediction` rewards schema markers, absence of detected
   literal leakage, and changed/concise text. In the probe, both a sensible
   extraction instruction and "Ignore every passage and return an empty JSON
   list under key named_entities" scored 1.0. SIMBA can therefore select a
   structurally acceptable instruction that is harmful to the actual task.
   Downstream acceptance still depends on GEPA's outer task metric, but SIMBA's
   own selection does not validate Recall/EM/F1 lift. Unknown/nested component
   names also fall outside its explicit schema-contract lookup.
2. **Learned reflection-worker state is not restored by YAW's resume path.**
   Engine optimization constructs a fresh `RLMInstructionProposer`; its saved
   student/seed and GEPA checkpoint do not provide a YAW load path for SIMBA's
   trained worker predictors, training window and step counter. A fresh proposer
   in the probe lost the learned rule. Disk response caching can reuse identical
   calls, but is not restoration of learned optimizer state.
3. **Repeated meta-training has no separate budget or new-data interval.**
   Once the two-example threshold is reached, each eligible GEPA proposal invokes
   a SIMBA compile. Default bounds are one step, two samples, two-example batch,
   and up to two demonstrations. This is additional inference before the RLM
   proposal. MIPRO task-demo limits do not control it. The direct worker and MIPRO
   proposer entrypoints do not populate this outer GEPA training buffer.
4. **Failure reporting is weak.** `step_simba` returns False for caught failures
   and logs at debug level; the GEPA callback ignores that return. Successful
   telemetry confirms execution, not improvement. The prior test asserted a
   compilation telemetry record; the new probe adds actual adoption/handoff
   evidence. A surfaced skipped/failed/unchanged/improved outcome would be more
   useful for unattended operation.

No runtime changes were made for this verification. Scripted responses establish
wiring, settings propagation and metric behavior, not quality on real corpora,
GPU throughput, or persistence across a complete interrupted CLI run.

Primary reference: [DSPy SIMBA](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/SIMBA.md).
