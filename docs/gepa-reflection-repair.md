# GEPA evidence and native proposal repair

The reflection path is now **training traces → sandbox evidence selection → native
DSPy InstructionProposer → downstream task scoring**. The previous synthesis
tool, prefix clipping and competing outer instruction output are removed.

## Behavior and configuration

`[inference].reflection_input_tokens` defaults to **16384**. Each final reflection
request is measured with the resident model's llama.cpp chat template/tokenizer,
including schema, demonstrations, instructions and evidence. Output tokens and a
512-token context margin are reserved independently. The external service must
already serve the selected model. The tokenizer backend URL is discovered from
Lemonade health; `reflection_tokenizer_url` supplies an explicit backend URL when
needed. Both `/v1` backend URLs and llama.cpp root URLs are supported. There is no
generic-tokenizer or character-estimate fallback. Different selector/writer models
require separate resident backends whose tokenizers can both be verified.

RLM selects stable example IDs, field paths and optional character spans. Host
code retrieves original values and verifies spans. Expected training answers
remain available as evidence. Invalid selections or exhausted RLM budgets fall
back to deterministic failure/success selection. The native writer receives the
component identity, immutable output contract and compact evidence. Optional
evidence is removed to fit; mandatory instructions/contracts are never sliced.
Native DSPy validates the instruction length and may make one compression call.

The native writer's predictors remain visible to SIMBA under the established
`qwen_proposer` / `qwen_compressor` names. Their native signatures replace the
previous implementations. SIMBA uses actual paired downstream task improvement,
retains bounded training examples and validates learned prompt capacity before
adoption. Held-out answers never enter reflection or SIMBA calibration.

## Persistence and compatibility

Evidence selections and proposals are independently cached in the existing
SQLite infrastructure. Each request checkpoints collecting, learning, learned and
complete phases. A bounded active-request snapshot is pinned in SQLite metadata
and committed with its learned worker state, independently of FIFO cache eviction.
On restart, incomplete learning starts from its recorded input state; completed
learning/proposals are reused. An interrupted in-flight generation can run again.

Worker state version is **2**. Old worker snapshots and mismatched model/data/code
identities fail compatibility checks; checkpoints are preserved. Corpus strings
and selected excerpts in snapshots use source identities/spans, hydrated from the
original corpus instead of duplicating corpus text in the database. Diagnostics
and learning histories remain bounded; reflection LM histories are disabled.
Reflection and SIMBA advice use isolated DSPy traces; only the active SIMBA
trajectory collection retains the training traces it needs.

MIPROv2 search/warmup policy is unchanged. Its existing shared instruction-worker
entrypoint uses the replacement evidence/native-writer implementation.

## Restart and verification checklist

- [x] Replace competing writers and preserve original training evidence.
- [x] Enforce serving-token budgets across the writer, RLM and SIMBA.
- [x] Pin active stages, persist learned state and hydrate source references.
- [x] Finish focused persistence, budget, sandbox and architecture verification.
- [x] Record the bounded serial EnterpriseRAG pilot and its limitations in evaluation.md.
- [ ] Complete all twelve held-out pairs: deadline reached after six; no quality gain established.

The pilot targets NER, synonym judging and QA, with two training cases and twelve
disjoint held-out cases. It compares the starting program with the proposed
program, measures paired Recall@1/@2/@5, EM/F1, task score, latency, logical tokens,
LM calls, cache hits and process memory. Missing component traces or incomplete
runs must be reported; they do not establish a quality gain. No overnight optimizer
is launched automatically.


Actual verification and partial pilot scores are recorded in
[evaluation.md, section 34](evaluation.md#34-evidence-driven-native-gepa-reflection-replacement-2026-10-23).
The pilot's Python peak working set was about 424 MiB, excluding model servers
and the sandbox. This is a measured bounded run, not a guarantee against resource
exhaustion. Start a new compatible optimization run for the replacement worker;
legacy version-1 worker state cannot be resumed by version 2.


At final verification, concurrent dependency edits left `pyproject.toml` and
`uv.lock` out of sync. Reconcile that dependency work before a locked launch;
reflection tests used the previously installed dependency set (CocoIndex 1.0.25).
