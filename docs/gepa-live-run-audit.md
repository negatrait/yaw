# Active GEPA configuration and dashboard audit — 2026-10-23

Scope: inspect the running `20261022_195047` optimization and improve future
configuration plus its dashboard. GEPA and model servers were not restarted.
The watchdog alone was reloaded after verifying its process tree and protecting
the optimizer/server process identities.

## Embedding identity

`yaw.toml` explicitly selected `embeddinggemma-300m-GGUF`, overriding the QAT
default already present in `YawConfig`. The active run's last 250 embedding
audit entries all named the standard model. Lemonade health showed both
embedding models resident, so the unwanted model was actually being requested.
The adapter also had an independent standard-model default.

The TOML selection and adapter default now use the shared core default
`embeddinggemma-300m-qat-q8_0-GGUF-Q8_0`, preserving the existing explicit QAT
profile/checkpoint, dimensions, pooling, task prefixes and index identity.
Explicit standard-model selection remains supported; misspelled/unregistered
aliases fail rather than silently falling back. The model spelling beginning
`embeddiggemma` is not a registered alias.

This does not switch the running optimizer's embedder or unload its model.
Quantization/model changes require a distinct compatible index/run. Existing
running-process settings and checkpoints remain intact.

## Serving context and execution budgets

Lemonade's actual Gemma launch command and recipe options declared 65,536
tokens. Its advertised architectural maximum was 131,072; that is not the
configured serving window. The old profile's 32,768 field did not dynamically
control extraction. Actual constraints were independent constants: 24,000
passage characters, 40,000 QA-context characters, 12,000-character helper
inputs, 600-character reflection fields, and RLM execution caps.

`[inference]` now exposes `context_window_tokens`, `max_output_tokens`,
`rlm_max_iters`, `rlm_max_llm_calls`, and `rlm_max_output_chars`. Task and
reflection LMs receive the configured completion limit; context capacity is
local metadata, not an unsupported provider request parameter. Task/ensemble
deployment settings override older compiled capacity metadata. RLM execution
budgets are explicitly forwarded to SHACL and GEPA reflection; increasing
context does not automatically increase turns or tool calls.

Character safeguards scale by `(context_window_tokens - max_output_tokens) /
(32768 - 4096)`, preserving previous defaults at 32K/4K. At configured 64K/4K,
the passage safeguard and default graph context budget become 51,428
characters, QA's safeguard 85,714, and reflection fields 1,285. An explicit
`[graph].max_context_chars` still overrides the derived graph budget. Runtime
now forwards that graph budget (previously the config field was not forwarded).
The 1,500-character evolved-instruction ceiling and explicit corpus chunking
settings remain independent structural/cost controls.

These are character heuristics, not full-prompt tokenization or proof that any
arbitrary prompt fits. Embedding capacity remains profile-specific. Capacity
and execution settings enter configuration/stage/candidate identities; changed
settings must not reuse incompatible task results or validation scores. A fresh
compatible run is required for these new settings, not a hot change mid-run.

## Progress interpretation

The inspected snapshot contained six candidates and 306 evaluations. Two had
full-validation composite scores: seed candidate 1 at **0.4759**, candidate 4
at **0.4525**. Minibatch variation is promising, but this snapshot does not
establish a full-validation gain over the seed.

Candidate F1/recall aggregates combine evaluations from different phases; merely
having a `val_score` does not make those aggregate F1/recall values pure
full-validation results. The report and watchdog status now compare the actual
full-validation composite score and label the mixed candidate averages.

## Graph and assertion trail

Both `skos:related` and `owl:differentFrom` endpoints remain seeds. Expand
exactly one hop, take the induced graph, and render only its largest connected
component. Equal-size components have deterministic tie-breaking. Fact-only
islands and smaller seeded components no longer clutter the view.

The relaxation budget is now **1,000 rounds**, compared with 40 for large graphs
and 80 for small graphs previously. `--layout-iterations` permits tuning it.
A preview of all five figures and the report took **60.47 seconds** at 2,228
selected nodes, including database/telemetry reads and other renderers.
The live reloaded dashboard subsequently displayed 2,238 selected entities.
All selected nodes/edges remain; labels are limited to the ten newest
disambiguation pairs within the view to reduce overlap.

The existing differentFrom list and ten newest active `skos:related` canonical
pairs now appear immediately after Fig. 2, with both names and definitions.
Lists cover the sandbox, while the figure is filtered. Recency uses SQLite
insertion `rowid`, excluding symmetric duplicates and deleted pairs; the schema
does not retain reassertion timestamps. It is an active-assertion trail, not an
immutable event log.

The running watchdog was reloaded with `--watch 200 --layout-iterations 1000`.
Its first completed report, graph rendering, assertion ordering and validation
summary were inspected. The optimizer and all three model-server process
identities were unchanged across that reload.

## Verification

See [evaluation.md](evaluation.md) for executed commands and outcomes. Tests use
real loopback HTTP, SQLite, DSPy serialization and rendered files. No fresh
live-model inference, model management, corpus rebuild or GEPA restart was
performed. The process retained its original inference configuration while it
was running. A later watchdog refresh at 06:45:36 UTC reported it stopped at
309 recorded evaluations. PID 82496 was then absent; its status file retained
`lifecycle=stopping` and `last_error=null`. The available evidence does not
identify the shutdown cause. The audit did not restart it or alter its saved
checkpoint. The optimizer PID had been explicitly verified unchanged directly
after reloading only the watchdog.
