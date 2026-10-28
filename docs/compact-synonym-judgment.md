# Compact synonym judgment implementation packet

Accepted 2026-10-27 on `dev`. Replace verbose model dossiers with compact shared
evidence. Keep Markdown inspection for people, native DSPy signatures/instructions,
candidate generation, existing limits and eight-chunk graph commits.

## Restart checklist and delivery

- [x] Capture eight immutable baseline candidate/evidence batches before replacement.
- [x] Add typed packets, physical-source deduplication and reversible local IDs.
- [x] Integrate bounded complete requests, validation, cache identities and telemetry.
- [x] Verify real files/SQLite, DSPy, loopback inference and fresh-process recovery.
- [ ] Run a paired pilot after the active optimizer stops: 16 generation calls,
  20 minutes maximum, no retries; report time, tokens, memory and disagreements.
- [x] Run affected tests, architecture checks, Ruff, ty and diff checks; record
  actual evidence and remaining failures in evaluation.md.

Packets preserve current selected names, definitions, labels, directed relations,
literal types, similarity scores, shared-document/island information, facts and
source excerpts. Share a source only when physical coordinates, AST context and
rendered contents agree; retain originating chunk references. Split whole pairs
without truncation. Reject invalid decision coverage and IDs. Cache identities
include format, original identities, evidence and model/predictor settings.

Dependencies: existing SQLite candidate caches and verified serving tokenizer.
The llama.cpp grammar binds every pair ID to one mandatory decision in input
order and constrains canonical IDs to that pair's two entities. Generic valid
JSON is insufficient to establish complete pair coverage. Host validation remains
mandatory for truncated responses or providers that do not enforce GBNF.
No new packages or index reset. Preserve running experiments. Target 40% fewer
prompt tokens; measure stage time and report decision differences. Agreement is
not proof of correctness. Do not launch GEPA or full benchmarks.
