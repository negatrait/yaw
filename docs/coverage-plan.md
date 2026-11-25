# Complete Production Coverage & Real-Substrate Test Plan (`tests/**/*.py`)

Status: active (modular `L0`–`L6` `yaw/` + `scripts/*.py` under `M_FINAL_1B`, `5c4b115`; Sprint `3.1`–`3.3` `U1`–`U4.5` verified).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md) |
| Rules | [tests.md](../.agents/rules/tests.md): `tests/**/*.py`, `scripts/**/*.py`; [canonical-files.md](../.agents/rules/canonical-files.md): `tests/core/test_architecture.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `tests/**/*.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/__init__.py`, `yaw/__main__.py` & `yaw/core/*` (`tests/core/*.py`); [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `yaw/pipeline/**/*.py` & `yaw/datasets.py` (with bundled `yaw/baselines.json`; `tests/test_pipeline.py`, `tests/test_datasets.py`); [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `yaw/retrieval/**/*.py` (`tests/test_retrieval.py`); [dspy-improvements.md](dspy-improvements.md): owns `yaw/qa.py` & `yaw/optimize.py` (`tests/test_qa.py`, `tests/test_optimize.py`); [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md): owns `yaw/mcp_server.py`, `yaw/runtime.py` & `yaw/cli.py` (`tests/test_mcp_server.py`, `tests/test_runtime.py`, `tests/test_cli.py`); [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): owns `scripts/{benchmark,download_datasets,verify_paper_parity_gate,evaluation}.py` (`tests/test_benchmark*.py`, `tests/test_download_datasets.py`, `tests/scripts/test_{download_edges,evaluation,parity_rendering}.py`); [epic-3-performance.md](epic-3-performance.md): owns `scripts/watchdog.py` (`tests/scripts/test_watchdog*.py`) |
| Supersedes / superseded by | Replaces legacy pre-cutover multi-package coverage checklists retired under [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md) and `EVAL-5` ([mdl-dag-sprint.md](reviews/mdl-dag-sprint.md), `5c4b115`) |
| Records | [evaluation-testing.md](evaluation-testing.md), [evaluation.md](evaluation.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md) |

---

## 1. Scope & Target

- **Acceptance Target:** `100%` executable-line coverage with meaningful behavioral assertions across all active production Python files (`yaw/**/*.py` and `scripts/*.py`, currently `27` `yaw/**/*.py` files across `L0`–`L6` + `5` `scripts/*.py` in Sprint `3.3` `U4`), zero `# pragma: no cover` exclusions, zero `# noqa`/`C901` suppressions, zero skipped tests, zero Ruff/`ty` diagnostics, and `--testmon` disabled (`-o addopts=""`).
- **Historical vs. Active Baseline:** The pre-cutover `2026-10-23` coverage sweep (`36` legacy files at `86.7169%`, recorded in [evaluation-testing.md](evaluation-testing.md)) predates the clean-slate cutover (`CS-6`, `3cd3952`) and `M_FINAL_1B` (`EVAL-5`, `5c4b115`). On `dev`, the full uncontended offline suite (`tests/**/*.py`) runs cleanly via `uv run --offline --locked pytest -o addopts=""`; a fresh `100%` line-coverage measurement via `uv run --offline --locked pytest -o addopts="" --cov=yaw --cov=scripts --cov-report=term-missing` remains open until executed and recorded in [evaluation-testing.md](evaluation-testing.md).

---

## 2. Production-to-Test Matrix (`yaw/**/*.py` + `scripts/*.py` -> `23` `test_*.py` + `3` Support Modules)

| Owned Production Domain | Production Files | Verification Suites (`tests/**/*.py`) | Real-Substrate Invariants Covered |
| :--- | :--- | :--- | :--- |
| Core contracts, config, logging & architecture (`L0`/`L6`) | `yaw/__init__.py`, `yaw/__main__.py`, `yaw/core/{__init__,config,logger,types}.py` | `tests/core/{test_architecture,test_config,test_config_edges,test_contract_edges,test_logger,test_types}.py` | `31`-field TOML bijection, `for_optimization()`, atomic commits, `FIFOLineRotatingHandler`, `63`-bit chunk IDs, `HydratedPassage` resolved bounds (`hydrated_start_line`/`hydrated_end_line`, `resolved_start_line`/`resolved_end_line`), path confinement, monotonic `L0`–`L6` DAG, `scripts/*.py` `Rule 12` (`"openai"` in `"pipeline"`, exact `cocoindex.connectorkits.fingerprint` in `"retrieval"`), static `importlib` & `# noqa` ban, disjoint Tier-2 ownership |
| CocoIndex pipeline & dataset bundles (`L1`/`L2`) | `yaw/pipeline/{__init__,embedder,flow,session,targets}.py`, `yaw/datasets.py` | `tests/test_pipeline.py`, `tests/test_datasets.py` | Live/catch-up `open_pipeline_lifespan` & `SinkMutationTracker`, `.yaw/coco_lmdb` cross-DB memo reuse (`version=5`/`version=7`), public `render_match` AST headings & island bounds (`"title: <ast-title> | text: ..."`), `fingerprint_str` `source_hash` & `resolved_app`, pooled keep-alive `DspyBatchEmbedder` (`openai.AsyncOpenAI`, `"openai/"` stripping, `batch_size` slicing, `index` sorting, `azure` fallback, `aclose()`), `<f2>` `chunks` + `unicode61` `(title, body)` `chunk_fts` + `chunk_vocab` + `rabitq_index` + `citation_graph`, v2 `.md` manifest SHA-256 & EM/F1/Recall |
| `M_FINAL_1B` retrieval engine (`L1`) | `yaw/retrieval/{__init__,db,graph,hydration,index,lexical,rabitq,search,slip,types}.py` | `tests/test_retrieval.py` | `?mode=ro` WAL snapshots, `1`-bit SoA RaBitQ packing & `64 + 64 == 128` invariance, tangent-steered cyclic diffusion, Beta-tail LUT, conditional Noisy-OR, 1-bit DPP, `coverage`/`pool_k` parity, non-hydrating `dedup_ranked_positions(..., include_island=...)` (`0` tail opens), `SearchResult.next_candidates` + multi-chunk `cite` / rarest `pmi` `< 0.05 ms` `links`, `yaw/retrieval/slip.py` (`build_recall_payload`, `format_recall_text` `(path:es-ee)`, `dump_recall_json` `allow_nan=False`, `sanitize_slip_label`), `4,096`-cap `vocab_doc_cache` & `resolved_path_cache`, multi-chunk `(title, body)` & umlaut/compound/CJK FTS5, `INDEX_NOT_READY` vs `CONSISTENCY_ERROR`, `(es-ee)` MaxP island hydration + `(1, 1)` sentinel guard, and `fingerprint_str` disk hydration (`content_hash` & `compute_db_fingerprints` retain SHA-256) |
| Bare `dspy.ReAct` QA & `dspy.GEPA` (`L2`/`L4`) | `yaw/qa.py`, `yaw/optimize.py` | `tests/test_qa.py`, `tests/test_optimize.py` | Loopback `OpenAI`-compatible chat/tool server (`temperature=1.0`), 4 read-only corpus tools (`pool_k = max(k * 4, 50)`, shared `format_recall_text(build_recall_payload(...))`), `[chunk_id=<int>]` grounding & bare-ID `next`/`links` `MODEL_ERROR` rejection, `[STALE ON DISK]` `stale` / `stale_on_disk` `CONSISTENCY_ERROR`, `[0, 10)` slice guard, `FileLock` contention, `.yaw/qa_program.json` persistence |
| Runtime, FastMCP HTTP & CLI (`L3`/`L4`/`L5`) | `yaw/runtime.py`, `yaw/mcp_server.py`, `yaw/cli.py` | `tests/test_runtime.py`, `tests/test_mcp_server.py`, `tests/test_cli.py` | Unified `L3` `runtime_lifespan` (`query_sem` + headroom-clamped `embed_sem`, `DspyBatchEmbedder.acall` & `aclose()`, reactive sink hot-swap outside `index_lock`), `arecall_payload` / `recall_once_payload`, `acquire_daemon_control` (`0o600` `.yaw/control.token`), loopback `Host`/`Origin` `403`, `/healthz` & `/readyz?wait=` & `/status` & `/_internal/stop`, 3 MCP tools (`recall` via `ToolResult` with `build_recall_payload` sparse schema & `<= 0.85x` byte budget), `resolve_program_path`, `YAW_LISTENING`, POSIX/Win32 wait handles & `cli` subcommands (`yaw recall` sparse `next`/`links`/`stale` drift) |
| Benchmarks & dataset downloaders | `scripts/{benchmark,download_datasets,evaluation,verify_paper_parity_gate}.py` | `tests/{test_benchmark,test_benchmark_edges,test_download_datasets}.py`, `tests/scripts/test_{download_edges,evaluation,parity_rendering}.py` | Slice retrieval/QA orchestration, `baselines.json` comparisons, integrity checks, offline downloader cache validation, Markdown/SVG parity report rendering |
| Telemetry watchdog | `scripts/watchdog.py` | `tests/scripts/test_{watchdog,watchdog_edges,watchdog_llm_metrics}.py` | Read-only SQLite + `.yaw/dspy_status.json` / `.yaw/dspy_gepa_audit.json` polling and SVG dashboard rendering |

---

## 3. Real-Substrate Verification Rules & Gate Commands

1. **Real Substrates Only (`engineering-real-tests`):** Every test exercises real WAL SQLite (`yaw.db` in `tmp_path`), real `.yaw/coco_lmdb` environments, real UTF-8 filesystem spans, real NumPy/SciPy CSR ops, real loopback HTTP/FastMCP servers, and real child-process CLI/signal boundaries (`0` mocked databases, filesystems, or SQL queries).
2. **No Coverage Exclusion or Testmon Shortcuts:** Do not count mocked/copied fixtures toward production coverage, do not add `# pragma: no cover` exclusions, and do not use `--testmon` on schema, config, SQLite, or final acceptance gates.
3. **Offline Verification & Coverage Commands:**

```powershell
uv run --offline --locked ruff format --check
uv run --offline --locked ruff check
uv run --offline --locked ty check
uv run --offline --locked pytest -o addopts=""
uv run --offline --locked pytest -o addopts="" --cov=yaw --cov=scripts --cov-report=term-missing
git diff --check
```

---

## 4. Open Coverage Checklist

- [x] **Offline Suite Green (`EVAL-5`, `5c4b115`):** All `23` `test_*.py` real-substrate test suites (`26` `.py` files under `tests/` including `3` support modules) pass uncontended offline with `-o addopts=""` (`0` skipped).
- [ ] **Fresh Production `100%` Coverage Audit (`pytest --cov`):** Execute a clean unaccumulated `uv run --offline --locked pytest -o addopts="" --cov=yaw --cov=scripts --cov-report=term-missing` run across all active production files (`yaw/` and `scripts/`), close any remaining uncovered error/platform branches with real-substrate tests, and record the exact line report in [evaluation-testing.md](evaluation-testing.md).
