# Evaluation Record Index

Measured verification outcomes, live-silicon benchmarks, and open failures for YAW are partitioned into four domain-specific evaluation ledgers aligned with `.agents/rules/*.md`:

| Domain Ledger | Scope & Owning Rules | Key Recorded Milestones |
| :--- | :--- | :--- |
| [evaluation-ingestion.md](evaluation-ingestion.md) | `yaw/daemon/*.py`, `yaw/daemon/ingestion/*.py` (`daemon.md`) | Native CocoIndex SQLite targets (`source_chunks`, `chunk_inference`, `GRAPH_TARGET`), fresh-process inference recovery, dataset bundles & persistent snapshots, bounded overlap pilot, compact synonym packet dispatch, and Epic 2 spec reconciliation. |
| [evaluation-graph-storage.md](evaluation-graph-storage.md) | `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py` (`graph-storage.md`) | Installed baseline, smoke diagnostics, embedding repeatability, Milvus rejection, split extraction/QA hardware profiles, Tier 0 & Tier 1 HippoRAG 2 benchmarks, Sprints 1–6 (`yaw.db`, `sqlite-vec`, contentless FTS5, sparse CSR PPR, Tier 1 native parity gate), unified retrieval hooks, W3C SKOS/RDF/OWL/SHACL build-time hardening, and `E1-M1` Pass-3 & Pass-4 pure-chunk `"manifold-v3"` shootout (`Retain & Replace Default`; `ppr.py` default retired per [Epic 1 Decisions 18–19](epic-1-hipporag-daemon.md#decisions)). |
| [evaluation-inference.md](evaluation-inference.md) | `yaw/daemon/graph/inference/*.py` (`inference.md`) | Candidate-indexed telemetry, `task-quality-arithmetic-mean-v2` GEPA reward, explicit optimization resume & provenance, SQLite candidate-stage caches, native DSPy disk response cache, RAM retention bounds, MIPROv2 zero-shot policy, SIMBA counterfactual task reward & recovery, Gemma 4 32K/128K capacity scaling, native `InstructionProposer` reflection replacement, `YawGBNFAdapter` binding, and external Pareto `dspy.Ensemble`. |
| [evaluation-testing.md](evaluation-testing.md) | `tests/**/*.py`, `benchmark.py`, `scripts/**/*.py` (`tests.md`) | Python 3.14.9 upgrade, `uv run preflight` delivery, complete Ruff/ty/pytest 261-test repair, and 100% production coverage packets (`SQLite/downloader/report`, `engine recovery/reflection`, `benchmark orchestration`). |

The [Epic 1 HippoRAG Daemon](epic-1-hipporag-daemon.md), [Native CocoIndex Indexing](native-cocoindex-indexing.md), and [Epic 2 FastMCP HTTP](epic-2-fastmcp-http.md) checklists remain governed by the measured evidence recorded in these four ledgers.

---

## 2026-11-15: Pass-4 E1-M1 Pure-Chunk "manifold-v3" Shootout & Gate Outcome

Summary of the Pass-4 `E1-M1` pure-chunk `"manifold-v3"` live 4-dataset held-out shootout on `[0, 10)` (`40q`, `integrity_passed = True`, `10/10` seeds per dataset; see [evaluation-graph-storage.md](evaluation-graph-storage.md#2026-11-15-pass-4-e1-m1-pure-chunk-manifold-v3-probes-64-trial-5-horizon-optuna-study--live-4-dataset-shootout)):

| Arm & Candidate | Macro `R@1 / R@2 / R@5 / R@10 / R@20` | QA `EM` | QA `F1` | Total Eval Time | Gate vs Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Current YAW` (`ppr.py`, baseline) | `0.3396 / 0.5521 / 0.7021 / 0.8292 / 0.9250` | `0.0750` | `0.1908` | `34.3296 s` | `0/4` (`FAIL`) |
| `SubStateManifoldHMM` (Pass-4 untuned, `shootout_pass4_v3_untuned_4ds_10q.json`) | `0.4333 / 0.6771 / 0.8021 / 0.8938 / 0.9854` | `0.1000` | `0.2185` | `93.2237 s` | `0/4` (`FAIL`) |
| `SubStateManifoldHMM` (Pass-4 Post-Optuna active, `shootout_pass4_v3_4ds_10q.json`) | **`0.4333 / 0.6583 / 0.8146 / 0.8938 / 0.9854`** | **`0.1250`** | **`0.2482`** | `45.4962 s` | `0/4` (`FAIL`) |
| **Deltas (`Optuna Manifold - Current YAW`)** | **`+0.0937 / +0.1062 / +0.1125 / +0.0646 / +0.0604`** | **`+0.0500`** | **`+0.0574`** | `+11.1666 s` | Clears `PAPER_BASELINES` `R@5` (`0.8146 >= 0.7828`) |
| `monkeypatched_hipporag` (target baseline) | — / — / `0.8727` / — / — | `0.5439` | `0.6639` | — | — |
| `PAPER_BASELINES` (`70b` reference) | — / — / `0.7828` / — / — | — | — | — | — |

- **Outcome Decision (`Clean-Slate Replacement`, [Epic 1 Decision 20](epic-1-hipporag-daemon.md#decisions)):** Historical Pass-4 `"manifold-v3"` numbers above remain recorded verbatim on baseline `63c9c00, dirty`. Under Decision 20 (`2026-11-16`), `ppr.py` and the legacy `yaw/daemon/graph/{storage,inference}` hierarchy are completely retired for a zero-backward-compatibility clean-slate rewrite: `hmm.py` is renamed and refactored into `yaw/retrieval.py` (`RetrievalIndex`, `rank_chunks`, eliminating `"manifold"`/`"hmm"` naming everywhere), declarative `cocoindex.connectors.sqlite` manages `chunks` `vec0` + `fts5` directly, QA uses `dspy.ReAct` with 4 bounded tools at Gemma 4 `temperature = 1.0`, `FastMCP` HTTP exposes `recall`/`qa`/`daemon_status`, and CLI unifies under `yaw daemon`, `yaw recall`, `yaw qa`, and `yaw optimize {dspy|optuna}`.
- **Bare-Bones Replacement Sprint ([bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md)):** Replaces legacy OpenIE/W3C/OWL/SKOS/`shacl.py`/`ppr.py` internals with a `14`-file zero-wrapper flat clean-slate package (`8` clean domain modules + `4` `yaw/core/` modules + `yaw/__init__.py` + `yaw/__main__.py`, `<= 3,000` LOC target): constant live CocoIndex pure-chunk indexing (`yaw.db`), verbatim zero-LLM `rank_chunks` `recall`, bare `dspy.ReAct` iterative tool-driven `qa`, and `FastMCP` + CLI transport.
- **Does not establish:** Post-replacement `dspy.ReAct` QA parity under `D1-A6` or full-corpus academic scaling (`Epic 3`).
