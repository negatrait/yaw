# Instructions for coding agents

Read [FOUNDATIONAL.md](FOUNDATIONAL.md) first. YAW is a native HippoRAG 2 -style graph & storage engine backed by a single SQLite database (`yaw.db`), CocoIndex (`cocoindex>=1.0.0`) ingestion, and FastMCP over HTTP. Heavy ontology stacks (`rdflib`, OWL 2 RL, `pySHACL`, `spaCy`) remain retired unless proven by benchmark lift.

- Verify the user-designated checkout and its Git remote before writing. Current repository: `C:\Users\Harri.Ahokas\Documents\yaw`, branch `dev`.
- Execute [Epic 1](docs/epic-1-hipporag-daemon.md) before [Epic 2](docs/epic-2-fastmcp-http.md). Select the first unfinished packet and its observable exit condition. Epic 1 must achieve **Tier 1 benchmark parity** against upstream `hipporag` during development, and **Tier 2 full-corpus benchmark parity (or beat!)** as the Epic 1 acceptance gate.
- Build the native HippoRAG 2 engine in the locked domain containment tree (`yaw/{core,daemon/{ingestion,graph/{storage,inference}}}/`) on top of a single `yaw.db` (`sqlite-vec` + contentless `FTS5` + sparse CSR/Forward-Push PPR + 1-indexed line hydration) and native DSPy inference (`dspy.LM`, `dspy.Embedder`, `dspy.Signature`s, `dspy.RLM`, `dspy.GEPA`) with **zero `hipporag` imports anywhere in the repository** (`yaw/`, `benchmark.py`, `scripts/`, `tests/`). Enforce zero horizontal/cross-domain imports (`test_architecture.py`: a package `P` only imports `P.*` and `yaw.core.*`) so files are never migrated or refactored again. Benchmark evaluation in `benchmark.py` and `scripts/verify_paper_parity_gate.py` compares native `yaw` against two consolidated targets: **Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`)** (historical upstream HippoRAG 2 runs on `Qwen3-4B` + `EmbeddingGemma`, archived in `outputs/benchmarks/consolidated_historical_baselines.json`) and **Target 2 (`PAPER_BASELINES`)** (published HippoRAG 2 paper results on `Llama-3.3-70B` + `NV-Embed-v2`).
- Implement one runtime (`yaw/daemon/runtime.py`) and one `yaw.db` writer fed by CocoIndex (`cocoindex>=1.0.0`) coalesced `SyncDelta`s (`yaw/daemon/ingestion/corpus.py`). FastMCP (`yaw/daemon/mcp_server.py`) is a thin transport adapter to that runtime. Treat the inference server (`http://localhost:13305/api/v1` or configured URL) as an external persistent service.
- Preserve source files and user experiments. Put generated indexes under ignored outputs or disposable temporary directories, and rotate logs with `yaw/core/logger.py` (500-line FIFO). Never treat a partial directory scan as evidence of source deletion.
- Target modern Python (`requires-python = ">=3.14"`, `.python-version` `3.14.9`). Use `uv sync --locked` and `uv run --locked`. Add only the dependencies assigned to the current epic (`dspy`, `sqlite-vec`, `scipy`, `cocoindex`, `filelock` in Epic 1; `fastmcp` in Epic 2), updating the lock deliberately.
- Record actual checks and failures in [evaluation](docs/evaluation.md). Distinguish package resolution, installation, real inference, retrieval correctness and daemon/transport acceptance. A fake provider validates orchestration only.
- Keep README's available commands separate from epic target commands. Update scope documents when accepted decisions change. Run focused verification and `git diff --check` before reporting completion.
- Never add backward compatibility shims, convoluted fallback layers or similar cop outs that degrade performance while masking failures. Make clean cut breaking changes and refactor all call sites and consumers to keep the codebase lean.

## Guardrails

- Never import `_internal` or `__protected` functions, variables or constants in other modules.
- Allways generate brief, performant, idiomatic textbook code only.

## Tool calls

- `uv run ty` - typecheck, must pass clean.
- `uv run ruff` - check and format, must pass clean.
- `uv run pytest` - unit and integration tests, must pass clean with zero cushioning.