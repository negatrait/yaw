# Evaluation Record Index

Measured verification outcomes, live-silicon benchmarks, and open failures for YAW are recorded below. The top ledger (`CS-5` onward) tracks the clean-slate `14`-file replacement architecture, while the four domain-specific ledgers below preserve historical pre-cutover evidence from the retired `yaw/daemon/` hierarchy:

| Historical Pre-Cutover Ledger | Retired Scope & Current Owning Rules | Key Recorded Milestones |
| :--- | :--- | :--- |
| [evaluation-ingestion.md](evaluation-ingestion.md) | Historical `yaw/daemon/*.py`, `yaw/daemon/ingestion/*.py` (`daemon.md`) | Native CocoIndex SQLite targets (`source_chunks`, `chunk_inference`, `GRAPH_TARGET`), fresh-process inference recovery, dataset bundles & persistent snapshots, bounded overlap pilot, compact synonym packet dispatch, and Epic 2 spec reconciliation. |
| [evaluation-graph-storage.md](evaluation-graph-storage.md) | Historical `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py` (`graph-storage.md`) | Installed baseline, smoke diagnostics, embedding repeatability, Milvus rejection, split extraction/QA hardware profiles, Tier 0 & Tier 1 HippoRAG 2 benchmarks, Sprints 1–6 (`yaw.db`, `sqlite-vec`, contentless FTS5, sparse CSR PPR, Tier 1 native parity gate), unified retrieval hooks, W3C SKOS/RDF/OWL/SHACL build-time hardening, and `E1-M1` Pass-3 & Pass-4 pure-chunk `"manifold-v3"` shootout (`Retain & Replace Default`; `ppr.py` default retired per [Epic 1 Decisions 18–19](epic-1-hipporag-daemon.md#decisions)). |
| [evaluation-inference.md](evaluation-inference.md) | Historical `yaw/daemon/graph/inference/*.py` (`inference.md`) | Candidate-indexed telemetry, `task-quality-arithmetic-mean-v2` GEPA reward, explicit optimization resume & provenance, SQLite candidate-stage caches, native DSPy disk response cache, RAM retention bounds, MIPROv2 zero-shot policy, SIMBA counterfactual task reward & recovery, Gemma 4 32K/128K capacity scaling, native `InstructionProposer` reflection replacement, `YawGBNFAdapter` binding, and external Pareto `dspy.Ensemble`. |
| [evaluation-testing.md](evaluation-testing.md) | `tests/**/*.py`, `benchmark.py`, `scripts/**/*.py` (`tests.md`) | Python 3.14.9 upgrade, `uv run preflight` delivery, complete Ruff/ty/pytest 261-test repair, and 100% production coverage packets (`SQLite/downloader/report`, `engine recovery/reflection`, `benchmark orchestration`). |

The [Epic 1 HippoRAG Daemon](epic-1-hipporag-daemon.md), [Native CocoIndex Indexing](native-cocoindex-indexing.md), and [Epic 2 FastMCP HTTP](epic-2-fastmcp-http.md) documents preserve prior milestone context; [FOUNDATIONAL.md](../FOUNDATIONAL.md) §§1–5 and [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md) govern the active clean-slate implementation.

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
- **Bare-Bones Replacement Sprint ([bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md)):** Replaces legacy OpenIE/W3C/OWL/SKOS/`shacl.py`/`ppr.py` internals with a `14`-file zero-wrapper flat clean-slate package (`8` clean domain modules + `4` `yaw/core/` modules + `yaw/__init__.py` + `yaw/__main__.py`, revised in `CS-4a` from initial `<= 3,000` to a realistic `<= 5,500` LOC ceiling without code-golfing; `5,499` LOC measured across `14` files post-`CS-5`): constant live CocoIndex pure-chunk indexing (`yaw.db`), verbatim zero-LLM hybrid `recall`, bare `dspy.ReAct` iterative tool-driven `qa`, and `FastMCP` + CLI transport.
- **Does not establish (for Pass-4 above):** Post-replacement `dspy.ReAct` QA parity or full-corpus academic scaling ([Epic 3](epic-3-performance.md)); see the `2026-11-20` `CS-5` live-silicon record below.

---

## 2026-11-20: CS-5 Live Silicon Evaluation & Paired CS-6 Control Baseline

Live-silicon verification of the `CS-4`/`CS-5` clean-slate `14`-file architecture across the held-out `[0, 10)` 4-dataset parity gate (`n=40`, `index_queries=14`, `--fresh`) and a live `2`-query `dspy.GEPA` reflection/compilation smoke:

- **Live Endpoint & Configuration (`outputs/benchmarks/cs5_4ds_10q_gate.json`, `2026-11-20T17:35:14Z`):** `embeddinggemma-2` + `gemma-4-e4b` at `http://127.0.0.1:8888/v1` (zero-shot uncompiled bare `dspy.ReAct`, `temperature=1.0`, `tool_choice="required"`, `max_iters=6`, untuned `yaw/retrieval.py` default `seed_top_k=20`, `embedding_task_mode="distinct-titled"` with `title: none`, native `.md` v2 bundles). Integrity passed `4/4` (`0` errors, `0.0%` DPR fallback, `10/10` graph seeds per dataset).

| Dataset | Docs / Chunks | YAW `R@1 / R@2 / R@5` | DPR `R@5` | Hippo `R@5` | Paper 70B `R@5` | YAW `EM` (Hippo / Paper) | YAW `F1` (Hippo / Paper) | Idx (s) | Eval (s) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2wikimultihopqa` | `124` / `142` | `0.4250 / 0.7500 / 0.8500` | `0.9000` | `0.9091` | `0.9040` | **`0.8000`** (`0.6364` / `0.6500`) | **`0.9417`** (`0.7227` / `0.7100`) | `3.33` | `50.81` |
| `musique` | `277` / `297` | `0.2583 / 0.3917 / 0.6667` | `0.5917` | `0.6818` | `0.7470` | `0.4000` (`0.4091` / `0.3720`) | `0.4000` (`0.5485` / `0.4860`) | `5.13` | `135.56` |
| `hotpotqa` | `134` / `147` | `0.4000 / 0.6500 / 0.8500` | `0.8500` | `0.9000` | `0.9630` | **`0.6000`** (`0.4500` / `0.6270`) | `0.6267` (`0.6426` / `0.7550`) | `2.87` | `69.52` |
| `popqa` | `28` / `42` | `0.5000 / 0.5500 / 0.6000` | `0.6000` | `1.0000` | `0.5170` | `0.1000` (`0.6800` / `0.4900`) | `0.4319` (`0.7419` / `0.5620`) | `1.06` | `37.02` |
| **Macro Mean (`n=40`)** | `563` / `628` | **`0.3958 / 0.5854 / 0.7417`** | `0.7354` | `0.8727` | `0.7828` | **`0.4750`** (`0.5439` / —) | **`0.6001`** (`0.6639` / —) | **`12.39`** (`2,605.34` Hippo) | **`292.91`** (`867.75` Hippo) |

- **3-Run `QASignature` & Multi-ID Citation Fix Progression (uncompiled `gemma-4-e4b`, `[0, 10)` `n=40`, `4/4` integrity `passed=True`):**
  - Run 1 (original `QASignature` + single-ID `_CHUNK_ID_RE`, `run-*-{pa92v1vo,ec8tas9t,wty477lz,tk1iaebl}`): `R@5 = 0.7542`, `EM = 0.0000`, `F1 = 0.1516`. Root cause: `gemma-4-e4b` emitted prose explanations plus comma-joined citations (`[chunk_id=..., chunk_id=...]`) not matched by single-ID regex.
  - Run 2 (concise `QASignature` rev 1 + `_CITATION_BLOCK_RE`, `run-*-{93dd8ldn,qiykgkwq,2wkknlhc,_fu7190q}`): `R@5 = 0.7542`, `EM = 0.2750`, `F1 = 0.4526`.
  - Run 3 (minimal canonical `QASignature` + `_CITATION_BLOCK_RE`, active in [yaw/qa.py](../yaw/qa.py), `cs5_4ds_10q_gate.{json,md}`): `R@5 = 0.7417`, **`EM = 0.4750`**, **`F1 = 0.6001`** (`+0.3500 EM` and `+0.3519 F1` over Pass-4 Optuna `0.1250 / 0.2482`; indexing `12.39 s` vs `2,605.34 s` = ~210× faster).
- **Live `2`-Query `dspy.GEPA` Smoke (`2wikimultihopqa`, `uv run --offline --locked yaw optimize dspy --dataset 2wikimultihopqa --query-offset 10 --max-queries 2 --max-metric-calls 4 --fresh --json`):** exited `0`; `.yaw/dspy_status.json` recorded `state = "completed"`; `.yaw/dspy_gepa_audit.json` recorded `train_size = 1`, `val_size = 1`, `num_candidates = 2`, `best_val_score = 0.775`, `val_aggregate_scores = [0.775, 0.775]`; `.yaw/qa_program.json` persisted cleanly with top-level keys `["react", "metadata"]`.
- **Architecture LOC Audit (Post-`CS-5` Edit):** `yaw/` `14` files / **`5,499` LOC** (`<= 5,500`), `yaw/core/*` `4` files / **`938` LOC** (`<= 960`), `scripts/watchdog.py` **`347` LOC** (`<= 350`).
- **Gate Disposition & Explicit Non-Claims:**
  - `G1` (clean 14-file `<= 5,500` LOC package, zero wrappers), `G2` (declarative CocoIndex SQLite indexing with `0` LLM calls), `G4` (**`EM = 0.4750 > 0.1250`, `F1 = 0.6001 > 0.2482`** at `temperature=1.0`, `0` SQLite writes during QA, strict tool-observation citation grounding), and `G5` (`FastMCP` HTTP + CLI) pass.
  - `G3` untuned `yaw/retrieval.py` `R@5 = 0.7417` sits below Pass-4 Optuna `0.8146` (while beating Pass-4 on `musique`: `0.6667` vs `0.6333`), and is **explicitly accepted without throwaway `Optuna` tuning sweeps** as the paired `CS-6` control baseline under user-approved Option B and [mdl-dag-sprint.md](reviews/mdl-dag-sprint.md) (`§0.1`: `MDL-1` removes `optuna`, `spacy`, `optimize_optuna`, `SearchParams`, and `resolve_retrieval_params`).
  - **Does not establish:** Upstream HippoRAG 2 parity (`overall_gate_status = "FAIL"` in `cs5_4ds_10q_gate.json`), multi-dataset compiled GEPA generalization (`n=2` smoke only), or post-`CS-6` `mdl-dag-sprint` retrieval improvements.
