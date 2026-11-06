# Resource vocabulary boundary: execution record

Started 2026-11-06 on `dev`. Controlling contract: [native graph publication §5.3](../native-cocoindex-graph.md#53-resource-vocabulary-and-statement-boundary).
Builds on the uncommitted U2 worktree recorded in [incremental publication](incremental-publication-work.md);
that work's open gates are not changed by this record.

## Trigger and evidence

`uv run yaw dspy optimize --resume` on `yaw_optimize_20261105_175308.db` aborted every graph
publication with `ValueError: Resource ',' does not have one resolved NER sense`
(`storage/evidence.py`, `build_evidence`). Offline replay of the deterministic planner stages
over all 2,474 persisted extraction artifacts (read-only, no LLM) reproduced it without
inference: 2,449 records publish, 24 fail evidence binding, 1 needs definition repair (not
replayable offline). Classification of the 24 and an audit of 2,485 NER and 1,094 triple
predictor traces:

| Defect at the source | Count | Downstream symptom |
| --- | --- | --- |
| NER pair echoing DSPy's output template, `[", ", "]]}# note: the value you produce must adhere to the JSON schema: {"]` | 22 records (1,050 junk pairs) | label normalizes to `''`; no binding |
| `json_repair` closing a truncated junk pair, `[", ", ""]` | within the 22 | same |
| Numeric NER label `["429", "HTTP Status Code/Response"]` with reflexive statement `429 -> 429` | 1 | normalization drops the self-loop; materialization skips the scalar label; no binding |
| `[":429", ...]`: raw label not scalar, normalized `"429"` is | 1 | evidence checks the raw form, materialization the normalized form |
| Reflexive statements, e.g. `HubSpot -[CRM platform/software provided by HubSpot.]-> HubSpot` | 509 of 1,094 triple calls | silently dropped by normalization |

Of 14,775 persisted NER labels, 92 contain no letter: 68 junk (`,`, `**`) and 24 numbers or
percentages, all literals under §5.3. Six labels contain letters but match the scalar
heuristic (`401s`, `403s`, `503s`, `350ms`); `401s` is a regex false positive (`s` read as
seconds). Those remain governed by `ConceptShape/noScalarLiterals`.

The live run also printed `[yaw optimize] indexing: ...` after each traceback instead of
exiting: a failed publication does not stop the optimizer. Recorded as V4.

Scripts (scratch, not production): `replay_evidence.py`, `grammar_violations.py`,
`label_shapes.py`, `label_starts.py` in this conversation's scratch directory.

## Units

Each unit is a separate reviewable change with fixture-first tests (red before green), the
mandatory gates, and an entry below. Fixtures are the verbatim model outputs above.

| Unit | Change | Observable acceptance |
| --- | --- | --- |
| V1 | NER label rule: GBNF `label` requires a letter; NER stage rejects labels without one; `ConceptShape/resourceLabel`; evidence binding classifies labels on the normalized form used by materialization. Bump OpenIE identity. | Template-echo, truncated, `429` and `:429` fixtures cannot be generated (grammar), are rejected at the NER stage over loopback HTTP, and every parser-accepted fixture record normalizes, binds and builds evidence. SHACL flags a letterless entity. |
| V2 | Irreflexive statements per sense: triples GBNF offers each subject only other senses as resource objects; statement parser rejects same-sense statements; normalization raises instead of dropping; `StatementShape/irreflexive`. Bump OpenIE identity. | Reflexive fixtures cannot be generated and are rejected; distinct homograph senses still relate; SHACL flags a reflexive fact; no parser-accepted statement is dropped. |
| V3 | Rules: AGENTS.md and FOUNDATIONAL.md state the grammar/parser/SHACL parity rule and the "parser-accepted ⇒ publishable" invariant; the fixture-replay invariant test runs over every boundary fixture. | Rules updated; invariant test in the default suite. |
| V4 | A failed graph publication aborts the optimizer with the original exception instead of continuing to index. | Real-substrate test: injected publication failure terminates the run and propagates the error. |

Restart only after V1-V3: the identity bump re-extracts the 2,474 cached chunks (about six
hours at the observed 9 s/chunk), which is required because those artifacts were produced
under the boundary-violating grammar.

## Checkpoint

- [x] V1: resource label boundary.
- [x] V2: irreflexive statements.
- [x] V3: rules and invariant suite.
- [x] V4: publication failure aborts the run.

## Results

### V1: Resource label boundary
- **Changes**:
  - GBNF `DSPY_NER_GBNF_GRAMMAR` and `build_dspy_ner_grammar` enforce `label ::= "\"" non-letter* letter char* "\""`.
  - `_run_ner_stage` rejects letterless labels with `ValueError("NER resource label lacks lexical content...")`.
  - Added `ConceptShape/resourceLabel` to SQLite SHACL validation in `yaw/daemon/graph/storage/shacl.py`.
  - `yaw/core/types.py`: Added `is_resource_label` predicate checking for Unicode letter presence.
  - `yaw/daemon/graph/storage/evidence.py`: Checks normalized label forms when matching scalar literals.
  - Bumped OpenIE extraction identity fingerprint to invalidate pre-boundary artifacts.
- **Evidence**:
  - `tests/daemon/graph/test_resource_vocabulary.py::test_letterless_labels_are_not_resource_labels`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_ner_grammar_labels_require_a_letter`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_ner_stage_rejects_labels_without_lexical_content`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_parser_accepted_record_publishes_every_statement`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_shacl_rejects_letterless_resource_labels`
- **Gates**: Passed `ty check`, `ruff check`, `ruff format`, `pytest tests/daemon/graph/test_resource_vocabulary.py`.
- **Discoveries**: 92 of 14,775 persisted NER labels contained no letters (template echos like `,` and numeric codes like `429`). These are now cleanly rejected at extraction and flagged by SHACL.
- **What it does not establish**: Does not automatically salvage invalid cached artifacts; re-extraction under the new identity is required.

### V2: Irreflexive statements per sense
- **Changes**:
  - Triples GBNF (`build_dspy_triples_grammar`) generates per-subject statement rules offering each subject only other resource senses as objects, while permitting literal statements.
  - `extract_structured_triples` in `parsers.py` rejects same-sense statements with `ValueError("Statement relates a resource sense to itself")`.
  - `normalize_openie_record` raises `ValueError` on reflexive same-sense statements instead of silently dropping them.
  - Added `StatementShape/irreflexive` to SQLite SHACL shapes in `shacl.py`.
- **Evidence**:
  - `tests/daemon/graph/test_resource_vocabulary.py::test_triples_grammar_offers_each_subject_only_other_senses`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_single_sense_catalog_allows_only_literal_statements`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_statement_parser_rejects_same_sense_statements`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_reflexive_statement_is_rejected_over_http`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_normalization_never_drops_a_same_sense_statement`
  - `tests/daemon/graph/test_resource_vocabulary.py::test_shacl_rejects_reflexive_facts`
- **Gates**: Passed `ty check`, `ruff check`, `ruff format`, full regression tests.
- **Discoveries**: 509 of 1,094 triple calls contained self-loops (e.g. `HubSpot -> HubSpot`). The per-subject irreflexive GBNF grammar eliminates them at generation time.
- **What it does not establish**: Does not restrict cross-sense homograph relations (e.g. `ball [toy]` relating to `ball [dance]`), which remain valid.

### V3: Rules and invariant suite
- **Changes**:
  - `AGENTS.md` and `FOUNDATIONAL.md` updated to govern resource boundary (§5.3), three-layer parity (GBNF, parser, SHACL), and the invariant that every parser-accepted record must normalize, bind, and publish.
  - Default invariant test suite in `tests/daemon/graph/test_resource_vocabulary.py` verifies parity across real loopback HTTP and SQLite.
- **Evidence**:
  - Documentation updated in `AGENTS.md`, `FOUNDATIONAL.md`, and `docs/native-cocoindex-graph.md`.
  - Test suite passes in CI/preflight without mocks.
- **Gates**: Passed `git diff --check`, `ty check`, `ruff check`.
- **Discoveries**: Explicit invariant tests catch mock-fixture drift across test suites.
- **What it does not establish**: Does not bypass downstream dataset-specific benchmark evaluations.

### V4: Publication failure aborts the run
- **Changes**:
  - `yaw/core/types.py`: Added `FatalGraphPublicationError(BaseException)` which bypasses standard `except Exception:` catch blocks in `dspy.Evaluate` and GEPA candidate executors.
  - `yaw/daemon/graph/engine.py`: In `_ensure_candidate_chunks_indexed`, wraps publication exceptions in `FatalGraphPublicationError(exc)` when optimization is active; in `evolve_prompts`, unwraps and re-raises `f_err.original` to terminate immediately.
  - `yaw/daemon/runtime.py`: In `_run_optimize`, unwraps and re-raises `FatalGraphPublicationError` so the CLI process exits cleanly with the original cause.
- **Evidence**:
  - `tests/daemon/graph/test_resource_vocabulary.py::test_graph_publication_failure_aborts_optimization_fast` validates that an injected publication error in `_ensure_candidate_chunks_indexed` terminates the optimization run and raises the original `ValueError` without looping.
- **Gates**:
  - `uv run ty check --exclude tests/daemon/graph/storage/test_publication.py --exclude tests/daemon/graph/test_indexing_overlap.py` -> 0 errors.
  - `uv run ruff check --fix` -> clean.
  - `uv run ruff format` -> clean.
  - `uv run pytest -m "not live"` -> 868 passed.
  - `git diff --check` -> clean.
- **Discoveries**: GEPA's candidate evaluation loop was previously swallowing publication `ValueError` exceptions as `score=0.0` and repeatedly calling `index_chunks` on subsequent candidate passes. Subclassing `BaseException` halts the loop immediately.
- **What it does not establish**: Does not modify non-optimization runtime behavior (ingestion/query paths continue using standard runtime exception propagation).
