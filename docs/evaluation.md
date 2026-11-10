# Evaluation Record Index

Measured verification outcomes, live-silicon benchmarks, and open failures for YAW are partitioned into four domain-specific evaluation ledgers aligned with `.agents/rules/*.md`:

| Domain Ledger | Scope & Owning Rules | Key Recorded Milestones |
| :--- | :--- | :--- |
| [evaluation-ingestion.md](evaluation-ingestion.md) | `yaw/daemon/*.py`, `yaw/daemon/ingestion/*.py` (`daemon.md`) | Native CocoIndex SQLite targets (`source_chunks`, `chunk_inference`, `GRAPH_TARGET`), fresh-process inference recovery, dataset bundles & persistent snapshots, bounded overlap pilot, compact synonym packet dispatch, and Epic 2 spec reconciliation. |
| [evaluation-graph-storage.md](evaluation-graph-storage.md) | `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py` (`graph-storage.md`) | Installed baseline, smoke diagnostics, embedding repeatability, Milvus rejection, split extraction/QA hardware profiles, Tier 0 & Tier 1 HippoRAG 2 benchmarks, Sprints 1–6 (`yaw.db`, `sqlite-vec`, contentless FTS5, sparse CSR PPR, Tier 1 native parity gate), unified retrieval hooks, and W3C SKOS/RDF/OWL/SHACL build-time hardening. |
| [evaluation-inference.md](evaluation-inference.md) | `yaw/daemon/graph/inference/*.py` (`inference.md`) | Candidate-indexed telemetry, `task-quality-arithmetic-mean-v2` GEPA reward, explicit optimization resume & provenance, SQLite candidate-stage caches, native DSPy disk response cache, RAM retention bounds, MIPROv2 zero-shot policy, SIMBA counterfactual task reward & recovery, Gemma 4 32K/128K capacity scaling, native `InstructionProposer` reflection replacement, `YawGBNFAdapter` binding, and external Pareto `dspy.Ensemble`. |
| [evaluation-testing.md](evaluation-testing.md) | `tests/**/*.py`, `benchmark.py`, `scripts/**/*.py` (`tests.md`) | Python 3.14.9 upgrade, `uv run preflight` delivery, complete Ruff/ty/pytest 261-test repair, and 100% production coverage packets (`SQLite/downloader/report`, `engine recovery/reflection`, `benchmark orchestration`). |

The [Epic 1 HippoRAG Daemon](epic-1-hipporag-daemon.md), [Native CocoIndex Indexing](native-cocoindex-indexing.md), and [Epic 2 FastMCP HTTP](epic-2-fastmcp-http.md) checklists remain governed by the measured evidence recorded in these four ledgers.
