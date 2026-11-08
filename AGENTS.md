# Instructions for Coding Agents

> [!NOTE] YAW is meant to aid LLM-based agents.
> YAW offers recall and reasoning abilities over a large, ambiguous corpus to make the existence of an agent easier. YAW is not a memory, it's the librarian. Do YOU find the project interesting and useful? If yes, great; if not, what needs to be done to make this the most performant large-corpus GraphRAG ever conceived?

## Prime Directive

**A minimal line count codebase that fails fast is infinitely better than a fallback-heavy codebase that silently degrades performance to mask failures!**

- **Hidden Assumptions**: Surface uncertainty, ambiguity, inconsistencies, and trade-offs. Ask for clarification when ambiguity affects results; push back on incorrect claims with evidence.
- **Over-Engineering**: Write the smallest clear, idiomatic implementation satisfying the contract. Remove dead code, fallbacks, compatibility layers, and competing code paths. Prefer replacements over layering and robust dependencies over hand-rolled logic.
- **Masked Failures**: Retry transient model errors only via `dspy.LM(num_retries=...)` (never lower retry counts or write custom retry loops). When retries exhaust or a service is unavailable, abort and raise the original exception—no broad `except`, log-and-continue, defaults, or degraded modes.
- **Unintended Side Effects**: Change only what the task requires. Never alter code or comments you do not fully understand. Make clean breaking changes and update every caller and telemetry consumer atomically.
- **Missing Goal**: Define observable behavior, definition of done, and quality gates before implementation.
- **Be careful**: You have ample time, precision beats speed.

## Required Context & Architecture

Read [FOUNDATIONAL.md](FOUNDATIONAL.md) first. Verify checkout `C:\Users\Harri.Ahokas\Documents\yaw` on branch `dev`.

- **Stack**: Single `sqlite-vec` DB (`yaw.db`) via declarative CocoIndex `>=1.1.0` targets (`TableSchema.from_class`), contentless FTS5, 1-indexed hydration, closed-world W3C concept/entity semantics without ConceptSchemes, sparse CSR + Forward-Push PPR, native DSPy (`dspy.LM`, `dspy.Embedder`, `dspy.Signatures`, `dspy.RLM`, `dspy.GEPA`), FastMCP over HTTP. Do not restore `rdflib`, OWL 2 RL, `pySHACL`, or spaCy without benchmark proof.
- **Resource Boundary & Invariants (§5.3)**:
  - NER alone captures `skos:Concept` and `owl:NamedIndividual`; resource labels require lexical content (letters). Pure numbers, codes, dates, and percentages remain literals (`rdfs:Literal`).
  - Triples may point to literals as objects (`owl:DatatypeProperty`), but `rdf:subject` can never be a literal. Statements are strictly irreflexive per resource sense.
  - Three-layer parity: GBNF grammar, extraction parsers, and SQLite SHACL shapes enforce identical resource and statement contracts. Invariant: every parser-accepted record must normalize, bind, and publish without dropped statements.
  - Fail-fast publication: A failed publication immediately aborts optimization with the original exception (`FatalGraphPublicationError` subclasses `BaseException` to escape evaluation loop swallowers). Never repeat failed indexing or mask schema violations.
- **Containment (`test_architecture.py`)**: Keep code in `yaw/core/` and `yaw/daemon/{ingestion,graph/{storage,inference}}/`. Zero `hipporag` imports anywhere. Package `P` imports only `P.*` and `yaw.core.*`. Never import another module's `_internal` or `__protected` names.
- **Runtime & Ingestion**:
  - One CocoIndex runtime (`yaw/daemon/runtime.py`), one declarative Cocoindex writer for `yaw.db`, zero imperative bypass writing is allowed. Event-driven live ingestion (`localfs.walk_dir(..., live=True)` in `yaw/daemon/ingestion/corpus.py`), and a thin adapter in `yaw/daemon/mcp_server.py`.
  - Stream ingestion and candidate indexing in dataset-agnostic micro-batches of `64` chunks per `app.update()` cycle; never encode train/val splits or benchmark presets in production code.
  - Default LLM admission strictly to `parallel = 1` (`Semaphore(1)`) for local `llama.cpp` (`http://localhost:13305/api/v1`); batch embeddings (`max_batch_size = 64`). Keep sampling temp = 1 in development to prevent small-model looping.
  - Point CocoIndex `coco.Settings(db_path=...)` at a persistent, content-addressed cache across runs—never tie memo state to transient timestamped DB filenames.
  - Remove ConceptSchemes, clustering, scheme inference/embeddings/storage and all scheme consumers under [native graph publication](docs/native-cocoindex-graph.md). For fixed evidence, 64+64 and 128 chunks must yield identical canonical entity/fact/edge state with comparable cumulative compute. Permit legitimate semantic islands; never manufacture isolation from batch boundaries or fabricate links to force connectivity. Future schemes require measured Recall@N, EM or F1 improvement and acceptable resource/quality costs.

## Execution Order, Benchmarks & Safety

- Complete Epic 1 (Tier 1 dev parity, Tier 2 full-corpus acceptance gate) before Epic 2. Distinguish package resolution, installation, real inference, retrieval correctness, and daemon/transport acceptance (fake providers validate orchestration only).
- **Benchmarks**: Use `benchmark.py` and `scripts/verify_paper_parity_gate.py` against `MONKEYPATCHED_HIPPORAG_BASELINES` (`outputs/benchmarks/consolidated_historical_baselines.json`) and `PAPER_BASELINES`.
- **Safety**: Preserve source files and user experiments. Write generated indexes only to ignored `outputs/` or temp dirs. Rotate logs via `yaw/core/logger.py` (500-line FIFO). Remove superseded code and legacy fallbacks on sight.
- **Environment**: Python `>=3.14` (`.python-version`: `3.14.9`). Use `uv sync --locked` and `uv run --locked`. Add dependencies only when assigned (Epic 1: `dspy`, `sqlite-vec`, `scipy`, `cocoindex`, `filelock`; Epic 2: `fastmcp`).

## Live Process Verification & Completion

- Never mark a cutover complete until all callers and telemetry consumers (`scripts/watchdog.py`) are wired to the active schema and verified end-to-end.
- Editing code on disk does **not** reload a long-running process (`optimize`, `watchdog`, daemon) in memory, and the Windows command sandbox hides host processes—check host PIDs (`BypassSandbox: true`) and verify actual artifact timestamps and non-zero data before claiming live state.
- Keep current README commands separate from epic targets; update scope docs when decisions change; report only checks actually executed; commit with conventional commits after passing all gates.
- **Review records**: Before planning, read `docs/reviews/` for the subject and resume from its latest verified checkpoint. Every implementation unit appends its result (changes, evidence paths, gate results, discoveries, what it does not establish) to its `docs/reviews/<topic>.md` record in the same commit. Promote contract decisions into the governing spec; tick checkpoints only with evidence.

## Mandatory Quality Gates & Commands

```bash
uv run ty check
uv run ruff check --fix
uv run ruff format
uv run pytest
git diff --check
```

- `uv run yaw dspy optimize`: launch or resume GEPA optimization in `.yaw/optimize_yaw.db` (pass `--fresh` to wipe prior `optimize_*` state and start from scratch).
- `$env:YAW_ARTIFACT_DIR = "<directory>"; uv run watchdog --watch 300`: launch the DSPy optimization watchdog outputting to the dashboard directory.
