# Instructions for Coding Agents

> [!NOTE] YAW is meant to aid LLM based agents.
> YAW offers recall and reasoning abilities over a large, ambiguous corpus to make the existence of an agent easier. YAW is not a memory, it's the librarian. As such, the key question is: Do YOU find the project interesting and useful? If yes, great, if not, then what needs to be done to make this the most performant large corpus GraphRAG ever concieved?

## Prime Directive

Avoid these failure modes at all costs - deviation from the prime directives only moves the completion goal further away.

### Hidden Assumptions

- Surface uncertainty, ambiguity, inconsistencies, and trade-offs.
- Ask for clarification when ambiguity could affect the result.
- Push back on incorrect claims when supported by evidence.

### Over-Engineering

- Write the smallest clear, idiomatic implementation that satisfies the contract.
- Avoid unnecessary abstractions, fallbacks, branching, configurability, compatibility layers, and dead code.
- Prefer a single source of truth and explicit failures.
- Every extra layer of code just makes it harder for you to read and understand.

### Unintended Side Effects

- Change only what the task requires.
- Never alter code or comments you do not understand.
- Avoid unrelated refactoring and clean up artifacts introduced by your work.

### Missing Goal

- Define the observable behavior, definition of done, and quality gates before implementation. 
- Implement only what is required to satisfy that contract.

## Required Context

Read `FOUNDATIONAL.md` first.

YAW is a native HippoRAG 2-style graph and storage engine using:

- One SQLite database: `yaw.db`
- `sqlite-vec`
- Contentless FTS5
- Sparse CSR and Forward-Push PPR
- 1-indexed source-line hydration
- CocoIndex `>=1.0.0` ingestion
- Native DSPy inference
- FastMCP over HTTP

Do not restore `rdflib`, OWL 2 RL, `pySHACL`, or spaCy unless benchmarks prove a measurable improvement.

Before writing, verify the user-designated checkout and its Git remote:

- Repository: `C:\Users\Harri.Ahokas\Documents\yaw`
- Branch: `dev`

## Execution Order and Acceptance Gates

Complete Epic 1 before Epic 2.

Select the first unfinished packet and its observable exit condition.

Epic 1 must:

- Reach Tier 1 benchmark parity with upstream HippoRAG during development.
- Reach or exceed Tier 2 full-corpus benchmark parity as its acceptance gate.

Record actual verification results and failures. Distinguish between:

- Package resolution
- Installation
- Real inference
- Retrieval correctness
- Daemon and transport acceptance

A fake provider validates orchestration only.

## Architecture

Keep implementation inside this domain containment tree:

```text
yaw/
├── core/
└── daemon/
    ├── ingestion/
    └── graph/
        ├── storage/
        └── inference/
```

Enforce these rules:

- Zero `hipporag` imports in `yaw/`, `benchmark.py`, `scripts/`, and `tests/`.
- A package `P` may import only `P.*` and `yaw.core.*`.
- Enforce domain boundaries in `test_architecture.py`.
- Never import another module's `_internal` or `__protected` names.
- Do not migrate files or refactor across domains.

Use native DSPy components:

- `dspy.LM`
- `dspy.Embedder`
- `dspy.Signatures`
- `dspy.RLM`
- `dspy.GEPA`

## Runtime and Ingestion

Implement:

- One Cocoindex runtime in `yaw/daemon/runtime.py`
- One writer for `yaw.db`
- CocoIndex coalesced live ingestion in `yaw/daemon/ingestion/corpus.py`
- A thin FastMCP transport adapter in `yaw/daemon/mcp_server.py`

Treat the inference server at `http://localhost:13305/api/v1`, or the configured URL, as an external persistent service.

## Benchmarks

Use:

- `benchmark.py`
- `scripts/verify_paper_parity_gate.py`

Compare native YAW against these consolidated targets:

1. `MONKEYPATCHED_HIPPORAG_BASELINES`
   Historical upstream HippoRAG 2 results using Qwen3-4B and EmbeddingGemma, stored in `outputs/benchmarks/consolidated_historical_baselines.json`.
2. `PAPER_BASELINES`
   Published HippoRAG 2 results using Llama-3.3-70B and NV-Embed-v2.

## Repository Safety

- Preserve source files and user experiments.
- Store generated indexes only in ignored `outputs/` paths or disposable temporary directories.
- Never interpret a partial directory scan as evidence of source deletion.
- Rotate logs through `yaw/core/logger.py` using a 500-line FIFO.
- Make clean breaking changes and update every caller and consumer.
- Never add compatibility shims or fallback layers that mask failures or reduce performance.

## Environment and Dependencies

Target:

- `requires-python = ">=3.14"`
- `.python-version`: `3.14.9`

Use locked environments:

```bash
uv sync --locked
uv run --locked
```

Add dependencies only when assigned:

- Epic 1: `dspy`, `sqlite-vec`, `scipy`, `cocoindex`, `filelock`
- Epic 2: `fastmcp`

Update the lockfile deliberately.

## Documentation and Completion

- Keep currently available README commands separate from epic target commands.
- Update scope documents when accepted decisions change.
- Run focused verification and `git diff --check` before reporting completion.
- Report only checks actually executed and their real outcomes.

## Mandatory Quality Gates

All checks must pass without suppressions, cushioning, or hidden fallbacks:

```bash
uv run ty
uv run ruff check --fix
uv run ruff format
uv run pytest
git diff --check
```
