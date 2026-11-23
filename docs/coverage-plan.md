# Complete Production Coverage & Real-Substrate Test Plan (`tests/**/*.py`)

Status: active (modular `L0`–`L6` `yaw/` + `scripts/*.py` under `M_FINAL_1B`, `5c4b115`; Sprint `3.1` governance).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md) |
| Rules | [tests.md](../.agents/rules/tests.md): `tests/**/*.py`, `scripts/**/*.py`; [canonical-files.md](../.agents/rules/canonical-files.md): `tests/core/test_architecture.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `tests/**/*.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/__init__.py`, `yaw/__main__.py` & `yaw/core/*` (`tests/core/*.py`); [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `yaw/pipeline.py` & `yaw/datasets.py` (with bundled `yaw/baselines.json`; `tests/test_pipeline.py`, `tests/test_datasets.py`); [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `yaw/retrieval.py` (`tests/test_retrieval.py`); [dspy-improvements.md](dspy-improvements.md): owns `yaw/qa.py` & `yaw/optimize.py` (`tests/test_qa.py`, `tests/test_optimize.py`); [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md): owns `yaw/mcp_server.py`, `yaw/runtime.py` & `yaw/cli.py` (`tests/test_mcp_server.py`, `tests/test_runtime.py`, `tests/test_cli.py`); [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): owns `scripts/{benchmark,download_datasets,verify_paper_parity_gate,evaluation}.py` (`tests/test_benchmark*.py`, `tests/test_download_datasets.py`, `tests/scripts/test_{download_edges,evaluation,parity_rendering}.py`); [epic-3-performance.md](epic-3-performance.md): owns `scripts/watchdog.py` (`tests/scripts/test_watchdog*.py`) |
| Supersedes / superseded by | Replaces legacy pre-cutover multi-package coverage checklists retired under [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md) and `EVAL-5` ([mdl-dag-sprint.md](reviews/mdl-dag-sprint.md), `5c4b115`) |
| Records | [evaluation-testing.md](evaluation-testing.md), [evaluation.md](evaluation.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md) |

---

## 1. Scope & Target

- **Acceptance Target:** `100%` executable-line coverage with meaningful behavioral assertions across all active production Python files (`yaw/**/*.py` and `scripts/*.py`, currently `14` `yaw/` modules + `5` `scripts/*.py` in Sprint `3.1`), zero `# pragma: no cover` exclusions, zero `# noqa`/`C901` suppressions, zero skipped tests, zero Ruff/`ty` diagnostics, and `--testmon` disabled (`-o addopts=""`).
- **Historical vs. Active Baseline:** The pre-cutover `2026-10-23` coverage sweep (`36` legacy files at `86.7169%`, recorded in [evaluation-testing.md](evaluation-testing.md)) predates the clean-slate cutover (`CS-6`, `3cd3952`) and `M_FINAL_1B` (`EVAL-5`, `5c4b115`). On `dev`, the full uncontended offline suite (`tests/**/*.py`) runs cleanly via `uv run --offline --locked pytest -o addopts=""`; a fresh `100%` line-coverage measurement via `uv run --offline --locked pytest -o addopts="" --cov=yaw --cov=scripts --cov-report=term-missing` remains open until executed and recorded in [evaluation-testing.md](evaluation-testing.md).

---

## 2. Production-to-Test Matrix (`yaw/**/*.py` + `scripts/*.py` -> `23` `test_*.py` + `3` Support Modules)

| Owned Production Domain | Production Files | Verification Suites (`tests/**/*.py`) | Real-Substrate Invariants Covered |
| :--- | :--- | :--- | :--- |
| Core contracts, config, logging & architecture (`L0`/`L6`) | `yaw/__init__.py`, `yaw/__main__.py`, `yaw/core/{__init__,config,logger,types}.py` | `tests/core/{test_architecture,test_config,test_config_edges,test_contract_edges,test_logger,test_types}.py` | `31`-field TOML bijection, `for_optimization()`, atomic commits, `FIFOLineRotatingHandler`, `63`-bit chunk IDs, path confinement, monotonic `L0`–`L6` DAG, `scripts/*.py` `Rule 12`, static `importlib` & `# noqa` ban, disjoint Tier-2 ownership |
| CocoIndex pipeline & dataset bundles (`L1`/`L2`) | `yaw/pipeline.py`, `yaw/datasets.py` | `tests/test_pipeline.py`, `tests/test_datasets.py` | Live/catch-up `_indexing_main`, `.yaw/coco_lmdb` cross-DB memo reuse, `<f2>` `chunks` + `unicode61` `chunk_fts` + `chunk_vocab` + `rabitq_index` + `citation_graph`, v2 `.md` manifest SHA-256 & EM/F1/Recall |
| `M_FINAL_1B` retrieval engine (`L1`) | `yaw/retrieval.py` | `tests/test_retrieval.py` | `?mode=ro` WAL snapshots, `1`-bit SoA RaBitQ packing & `64 + 64 == 128` invariance, tangent-steered cyclic diffusion, Beta-tail LUT, conditional Noisy-OR, 1-bit DPP, umlaut/compound/CJK FTS5, `INDEX_NOT_READY` vs `CONSISTENCY_ERROR`, SHA-256 disk hydration |
| Bare `dspy.ReAct` QA & `dspy.GEPA` (`L2`/`L4`) | `yaw/qa.py`, `yaw/optimize.py` | `tests/test_qa.py`, `tests/test_optimize.py` | Loopback `OpenAI`-compatible chat/tool server (`temperature=1.0`), 4 read-only corpus tools, `[chunk_id=<int>]` grounding, `stale_on_disk` `CONSISTENCY_ERROR`, `[0, 10)` slice guard, `FileLock` contention, `.yaw/qa_program.json` persistence |
| Runtime, FastMCP HTTP & CLI (`L3`/`L4`/`L5`) | `yaw/runtime.py`, `yaw/mcp_server.py`, `yaw/cli.py` | `tests/test_runtime.py`, `tests/test_mcp_server.py`, `tests/test_cli.py` | `PRAGMA data_version` hot-swap outside `index_lock`, `BoundedSemaphore(32)` `BUSY`, loopback `Host`/`Origin` `403`, `/healthz` & `/readyz`, 3 MCP tools, `resolve_program_path`, Win32/POSIX signals & `cli` subcommands |
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
