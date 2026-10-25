# Epic 2: FastMCP Streamable HTTP Adapter (`yaw/daemon/mcp_server.py`)

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Prerequisite: [Epic 1](epic-1-hipporag-daemon.md) (`D1-A0..D1-A6` passed) | Results: [evaluation](evaluation.md)

## Containment & Transport Contract

- **Containment Law:** `yaw/daemon/mcp_server.py` resides in `yaw/daemon/` and imports **only** `yaw.daemon.runtime`, `yaw.core.*`, and `fastmcp`. It never imports `ingestion/`, `graph/`, `storage/`, or `inference/` directly (`test_architecture.py` enforced).
- Mount `FastMCP` onto the E1 `Runtime` lifespan at `http://127.0.0.1:13306/mcp` (`mcp.run(transport="http", host="127.0.0.1", port=13306, path="/mcp")`).
- One process, one `yaw.db` owner, one worker queue. No separate `serve-mcp` process, no stdio/SSE entry points, no wildcard CORS.
- Health routes: `/healthz` = 200 liveness; `/readyz` = 200 when committed index is ready/empty, 503 when starting/stopping/failed.

## Tools (5 Fixed Endpoints Delegating to `Runtime`)

All query inputs trimmed, non-empty, $\le 1024$ UTF-8 bytes.

| Tool | Args | Action & Return (`yaw/daemon/runtime.py` queue) |
| --- | --- | --- |
| `retrieve` | `query: str`, `top_k: int = 5` (1–20) | `readOnlyHint=true`. Queue `runtime.retrieve(query, top_k)`. Return `{rank, text, score, chunk_id, content_hash, stale_on_disk, source: {rel_posix_path, start_line, end_line, title}}` with MaxP canonical passage span aggregation ($S(\text{passage}) = \max_{c \in \text{passage}} S(c)$) and 1-indexed line hydration. |
| `rag_qa` | `question: str` | `readOnlyHint=true`. Queue `runtime.rag_qa(question)`. Return `{answer, evidence: passages[:5], last_sync_at}`. |
| `index_stats` | none | `readOnlyHint=true`. Read cached `runtime.stats()` (`chunks`, `entities`, `facts`, `edges`, `entity_labels`, `shacl_conforms`, config fingerprint) without blocking on worker thread. |
| `daemon_status` | none | `readOnlyHint=true`. Read cached `runtime.status()` (lifecycle, `provider_ready`, CocoIndex sync state, queue depth, active op, redacted last error). |
| `sync_sources` | none | `destructiveHint=true`. Call `runtime.request_sync()`; return `{scan_id, status: "accepted" \| "already_pending"}`. Triggers CocoIndex incremental sync across 30+ supported Tree-sitter file formats with $\pi_{\mathcal{W}, \mathcal{B}}$ key projection. |

## Execution & Error Rules

- Never block the `asyncio` loop; await `Runtime` queue futures.
- Emit real `Context.report_progress` stages (`queued`, `started`, `ready`).
- Cancel before execution removes the queued item; once running on the worker thread, finish the operation and discard the disconnected client's result.
- Return `isError=true` with fixed codes: `INVALID_INPUT`, `NOT_READY`, `EMPTY_INDEX` (QA only), `BUSY`, `MODEL_ERROR`, `CONSISTENCY_ERROR`, `SHUTTING_DOWN`.

## Sprints & Acceptance Checks

- **Copy/Adapt From:** `git show d1bc65e:src/yaw/server/daemon.py:90-160` (`FastMCP("yaw-mcp")` initialization, `@mcp.tool()` registration, `mcp.run(transport="http", host=..., port=...)`).
- **Drop (Anti-Spaghetti):** Drop `d1bc65e:daemon.py`'s ontology tools (`compile_wiki`, `lint_wiki`, `load_ontology`, `set_extraction_mode`, `get_ontology_schema`), `stdio` transport branches, and direct `yaw.retrieval`/`yaw.storage` imports.

| Sprint | Target Node & Files | Exit Condition |
| --- | --- | --- |
| **Sprint 1 (`E2-S1`)** | `pyproject.toml`, `uv.lock`, `yaw/daemon/mcp_server.py`, `yaw/daemon/cli.py` | Locked `fastmcp` installed; `/mcp`, `/healthz`, `/readyz` bind to `Runtime` lifespan; `test_architecture.py` passes (`0` cross-domain imports) |
| **Sprint 2 (`E2-S2`)** | `yaw/daemon/mcp_server.py`, `tests/daemon/test_mcp_http.py`, `README.md`, `docs/evaluation.md` | All 5 tools, progress notifications, cancellation, `BUSY` queue overflow, and Host/Origin checks pass over real loopback TCP |

- [ ] **E2-A1 (`E2-S1`):** FastMCP server mounts onto `Runtime` lifespan with zero direct `graph`/`storage`/`inference` imports; `/healthz` and `/readyz` reflect live `Runtime` state.
- [ ] **E2-A2 (`E2-S2`):** FastMCP client connects over TCP to `127.0.0.1:13306/mcp`, lists 5 tools, receives progress events, and retrieves 1-indexed line-hydrated passages + QA answers.
- [ ] **E2-A3 (`E2-S2`):** Two concurrent TCP clients share one `yaw.db` owner; `daemon_status` and `/healthz` respond in `<2s` during slow inference; 32-slot overflow returns `BUSY`; live CocoIndex edits update citations atomically.

---

## Next Epic: Epic 3 Proposal ("The Great Cleanup & Configurable Substrate")

### 1. Motivation & Goals
As YAW solidifies its native HippoRAG 2 engine, knowledge graph traversal, and FastMCP transport across Epics 1 and 2, historical scaffolding, compatibility shims, and hardcoded prompt/parameter surfaces must be systematically eliminated. Epic 3 will align the user-facing CLI with strict domain containment, introduce standalone Optuna hyperparameter optimization, and externalize all hyperparameters and seed prompts into declarative configuration.

### 2. Core Pillars of Epic 3

#### A. CLI Surface Reconsideration & Anti-Sprawl Cleanup
- **Purge the `yaw dspy` Historical Misnomer:** The `dspy` command group was an early development shim. Benchmarking, optimization, and program promotion belong to native engine operations.
- **Strict Domain-Aligned CLI Hierarchy:**
  ```text
  yaw daemon   {start|stop|status|run}
  yaw graph    {build|rebuild|lint|tune}
  yaw optimize {gepa|optuna|hybrid}
  yaw evaluate {benchmark|gate}
  yaw config   {get|set|show}
  ```
- **Deprecation of Legacy Shims:** Remove unused flags, dead parameter aliases, and obsolete compatibility wrappers across `cli.py` and `runtime.py`.

#### B. Standalone Optuna Hyperparameter Optimization (`yaw graph tune` / `yaw optimize --mode optuna`)
- **Fast, Non-LLM Graph Traversal Tuning:** Decouple continuous/discrete numerical hyperparameter tuning from slow LLM prompt reflection.
- **Trial Velocity:** Optuna evaluates candidate traversal settings on pre-indexed graphs via fast SciPy CSR PPR runs (executing 100 validation queries in seconds).
- **Comprehensive Traversal Parameter Search Space:**
  - PPR Dynamics: `damping` ($\alpha$), `beta` ($\beta$), `activation` (`scaled_tanh`, `linear`, `sigmoid`), `activation_param`, `dual_weight`.
  - Seeding & Disambiguation: `passage_seed_pool_k`, `fts_seed_pool_k`, `same_sense_cosine_gate`, `synonymy_threshold`.
  - SKOS Traversal Policies: `skos_broader_up`, `skos_broader_down`, `skos_related`, `fact_forward`, `fact_reverse`, `parallel_aggregation` (`max`, `sum`, `capped_sum`), `parallel_cap`.
- **Output:** Writes the best-performing `RelationTraversalPolicy` and retrieval parameters directly to `yaw.toml` or candidate presets.

#### C. Fully Configurable Hyperparameter Architecture (`yaw.toml`)
- Restructure `yaw.toml` with explicit domain sections so zero engine knobs are hardcoded in source:
  ```toml
  [graph.traversal]
  damping = 0.5
  beta = 0.5
  activation = "scaled_tanh"
  dual_weight = 0.5

  [graph.policy]
  skos_broader_up = 1.0
  skos_broader_down = 0.5
  skos_related = 1.0
  fact_forward = 1.0
  fact_reverse = 1.0
  parallel_aggregation = "max"
  parallel_cap = 2.0

  [graph.seeding]
  passage_seed_pool_k = 10
  fts_seed_pool_k = 10
  same_sense_cosine_gate = 0.85
  ```

#### D. Externalized Seed Prompts & Demonstrations
- **Eliminate Hardcoded Strings in Source:** Move hardcoded prompt docstrings, signature instructions, and seed few-shot demonstrations out of `yaw/daemon/graph/inference/prompts.py` and `program.py`.
- **Declarative Prompt Files (`prompts/*.toml` or `[inference.prompts]` in `yaw.toml`):**
  - Allow manual inspection, version-controlled prompt tuning, and custom few-shot demo injection without python code modifications.
  - DSPy signatures load instructions and demonstrations dynamically from config at startup while maintaining strict GBNF grammar validation.
