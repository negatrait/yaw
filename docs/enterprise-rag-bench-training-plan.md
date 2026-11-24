# EnterpriseRAG-Bench Dataset, Benchmark & Evaluation Plan

Status: active (clean-slate modular `M_FINAL_1B`, `5c4b115`; Sprint `3.1` `L0`–`L6` DAG).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [tests.md](../.agents/rules/tests.md): `scripts/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `scripts/benchmark.py`, `scripts/download_datasets.py`, `scripts/verify_paper_parity_gate.py`, `scripts/evaluation.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/core/` (`YawConfig`, `configure_logging`, `format_doc_for_embedding`, `format_query_for_embedding`, `write_text_atomic`); [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `yaw/datasets.py` & `yaw/pipeline/**/*.py` (bundle preparation/validation, slice scoping, `start_pipeline`, `update_pipeline_once`, `score_retrieval_recalls`, `score_qa_em_f1`, bundled `yaw/baselines.json`); [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `yaw/retrieval/**/*.py` (`build_retrieval_index`, `search_index`, `hydrate_ranked_passages`); [dspy-improvements.md](dspy-improvements.md): owns `yaw/qa.py` & `yaw/optimize.py` (`make_react_instruction_proposer`, `qa_gepa_metric`, `optimize_dspy`, `.yaw/qa_program.json`); [epic-3-performance.md](epic-3-performance.md): owns `scripts/watchdog.py` & tracks shipped `M_FINAL_1B` parity benchmarks; [coverage-plan.md](coverage-plan.md): owns `tests/**/*.py` (`tests/test_benchmark*.py`, `tests/test_download_datasets.py`, `tests/scripts/test_{download_edges,evaluation,parity_rendering}.py`) |
| References | [yaw/baselines.json](../yaw/baselines.json), [outputs/benchmarks/consolidated_historical_baselines.json](../outputs/benchmarks/consolidated_historical_baselines.json), [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md), [archive/pre-epic2-memo.md](archive/pre-epic2-memo.md) (historical context) |
| Records | [evaluation.md](evaluation.md), [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md), [mdl-dag-sprint.md](reviews/mdl-dag-sprint.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md) |

---

## 1. Consolidated Benchmark Reference Targets

Upstream `hipporag` is permanently retired (`0` imports repository-wide). YAW benchmarks compare against **two explicit frozen targets** stored in [yaw/baselines.json](../yaw/baselines.json) and [outputs/benchmarks/consolidated_historical_baselines.json](../outputs/benchmarks/consolidated_historical_baselines.json):

1. **Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`):** Historical local runs of upstream HippoRAG 2 on the legacy `Qwen3-4B-Instruct-2507-GGUF` + `embeddinggemma-300m-GGUF` setup (never re-executed).
2. **Target 2 (`PAPER_BASELINES`):** Published HippoRAG 2 paper results (`Llama-3.3-70B-Instruct` + `NV-Embed-v2`, `arXiv:2502.14802v1`).

| Dataset | Eval Queries | Candidate Docs | Target 1: Local Upstream HippoRAG 2 (`Qwen3-4B` + `EmbeddingGemma`) | Target 2: Published Paper SOTA (`Llama-3.3-70B` + `NV-Embed-v2`) |
| :--- | :---: | :---: | :--- | :--- |
| **`popqa`** | 25 | 70 | **R@1**: `0.500` \| **R@2**: `0.580` \| **R@5**: `0.720`<br/>**EM**: `0.360` \| **F1**: `0.5956` | **R@2**: `0.439` \| **R@5**: `0.517`<br/>**EM**: `0.490` \| **F1**: `0.562` |
| **`2wikimultihopqa`** | 25 | 307 | **R@1**: `0.450` \| **R@2**: `0.780` \| **R@5**: `0.960`<br/>**EM**: `0.360` \| **F1**: `0.4923` | **R@2**: `0.762` \| **R@5**: `0.904`<br/>**EM**: `0.650` \| **F1**: `0.710` |
| **`hotpotqa`** | 25 | 344 | **R@1**: `0.400` \| **R@2**: `0.740` \| **R@5**: `0.960`<br/>**EM**: `0.200` \| **F1**: `0.3964` | **R@2**: `0.835` \| **R@5**: `0.963`<br/>**EM**: `0.627` \| **F1**: `0.755` |
| **`musique`** | 10 | 197 | **R@1**: `0.183` \| **R@2**: `0.317` \| **R@5**: `0.542`<br/>**EM**: `0.200` \| **F1**: `0.3000` | **R@2**: `0.456` \| **R@5**: `0.780`<br/>**EM**: `0.485` \| **F1**: `0.633` |
| **`lveval`** | 25 | 70 | **R@1**: `0.440` \| **R@2**: `0.640` \| **R@5**: `0.740`<br/>**EM**: `0.240` \| **F1**: `0.3401` | **EM**: `0.080` \| **F1**: `0.129`<br/>*(Long-Context 256k)* |
| **`narrativeqa`** | 25 | 70 | **R@1**: `0.100` \| **R@2**: `0.160` \| **R@5**: `0.280`<br/>**EM**: `0.080` \| **F1**: `0.1384` | **EM**: `0.180` \| **F1**: `0.259` |
| **`nq`** | 1 | 20 | **R@1**: `0.000` \| **R@2**: `0.000` \| **R@5**: `1.000`<br/>**EM**: `0.000` \| **F1**: `0.0000` | **R@2**: `0.456` \| **R@5**: `0.780`<br/>**EM**: `0.485` \| **F1**: `0.633` |

---

## 2. Owned Benchmark/Dataset Tooling & EnterpriseRAG-Bench Quotas

1. **Prepared v2 `.md` Bundles (`scripts/download_datasets.py`, consuming `yaw/datasets.py` owned by [native-cocoindex-indexing.md](native-cocoindex-indexing.md)):**
   - `scripts/download_datasets.py` downloads datasets (or checks local bundles with `--check`) and invokes `prepare_dataset_bundle` to export each corpus row as a UTF-8 Markdown file under `documents/<document_filename(doc_id)>.md` (`# {title}` H1 header via `render_document_markdown`), accompanied by `evaluation/questions.json` and SHA-256 `manifest.json`.
   - `resolve_slice_corpus_scope` passes prepared `.md` files directly to `yaw/pipeline/` (`indexing_main`) with zero JSON-to-text re-staging and keeps `evaluation/questions.json` strictly outside `documents_root`.
2. **EnterpriseRAG-Bench Category Quotas (`yaw/baselines.json`):**
   - `datasets["enterprise_rag_bench"]` declares `train_partition_size = 135` (`135` train / `27` validation = `162` queries) across 10 categories in `category_quotas`:
     `Basic: [33, 5]`, `Semantic: [25, 4]`, `Intra-Document Reasoning: [15, 3]`, `Project Related: [15, 3]`, `Constrained: [11, 2]`, `Conflicting Info: [8, 2]`, `Completeness: [8, 2]`, `Miscellaneous: [8, 2]`, `Info Not Found: [8, 2]`, `High Level: [4, 2]`.
3. **Evaluation & Gate Harnesses (`scripts/benchmark.py`, `scripts/evaluation.py`, `scripts/verify_paper_parity_gate.py`):**
   - `scripts/benchmark.py` executes slice retrieval/QA benchmarks over `yaw/pipeline/`, `yaw/retrieval/` (`RetrievalIndex`), and `yaw/qa.py` (`build_qa_agent`, `ask_question`).
   - `scripts/evaluation.py` (`RetrievalRecall`, `QAExactMatch`, `QAF1Score`, `check_benchmark_integrity`, `evaluate_benchmark_gate`) and `scripts/verify_paper_parity_gate.py` evaluate YAW against Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`), Target 2 (`PAPER_BASELINES`), and Standard DPR (`--compiled-program .yaw/qa_program.json` or `.yaw/compiled_program.json`).
4. **GEPA Optimization Sandbox (Peer `yaw/optimize.py`, [dspy-improvements.md](dspy-improvements.md)):**
   - Default optimization config (`yaw.toml` `[optimization]`) targets `datasets = ["enterprise_rag_bench"]`, `max_queries = 162`, `query_offset = 0`, `auto = "light"` inside `.yaw/optimize_*` while sharing `.yaw/coco_lmdb` (`--fresh` wipes `.yaw/optimize_*`), saving `.yaw/qa_program.json`.
   - Optimizing on parity or Tier-2 benchmark datasets (`2wikimultihopqa`, `hotpotqa`, `musique`, `popqa`, `nq`, `lveval`, `narrativeqa`) enforces `_guard_optimization_slice`: `query_offset >= 10` is mandatory so `[0, 10)` remains strictly untouched.

---

## 3. Domain Containment & Non-Goals

- `yaw/datasets.py` (`L1`, owned by [native-cocoindex-indexing.md](native-cocoindex-indexing.md), bundling `yaw/baselines.json`) and `yaw/optimize.py` (`L4`, owned by [dspy-improvements.md](dspy-improvements.md)) obey `tests/core/test_architecture.py` containment; this spec owns `scripts/{benchmark,download_datasets,evaluation,verify_paper_parity_gate}.py`.
- Zero `hipporag` imports anywhere in `yaw/`, `scripts/`, or `tests/`.
- Zero custom optimization scripts or legacy `yaw/daemon/` / `YawProgram` / `sqlite-vec` dependencies.

---

## 4. Verification & Quality Gates

1. **Bundle Preparation & Integrity Check:**
   ```powershell
   uv run --offline --locked python -m scripts.download_datasets --dataset enterprise_rag_bench --source-dir outputs/datasets/sources
   uv run --offline --locked python -m scripts.download_datasets --dataset enterprise_rag_bench --check
   ```
2. **GEPA Compilation & Gate Verification:**
   ```powershell
   uv run --offline --locked yaw optimize dspy --dataset enterprise_rag_bench --fresh
   uv run --offline --locked python -m scripts.verify_paper_parity_gate --compiled-program .yaw/qa_program.json --mode eval
   ```
3. **Mandatory Quality Gates:**
   ```powershell
   uv run --offline --locked ty check
   uv run --offline --locked ruff check
   uv run --offline --locked ruff format --check
   uv run --offline --locked pytest -o addopts="" tests/test_datasets.py tests/test_optimize.py
   git diff --check
   ```
