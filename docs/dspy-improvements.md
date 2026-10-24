# DSPy Practical Improvements & Roadmap

Practical design patterns, lessons learned, and future architecture investigations for YAW.

---

## 1. Zero-Shot Optimization on Local Silicon (Active Epic 1 Rule)

- **The Issue**: Bootstrapping few-shot demonstrations on local 4B models (`Qwen3-4B-Instruct`) causes prompt bloat ($3,500+$ tokens), generation timeouts and repetition loops. Historical token penalties were removed on 2026-10-24: GEPA now rewards the unweighted arithmetic mean of labeled retrieval recall (`Recall@1, 2, 5, 10, 20`) and QA `EM`/`F1` (`task-quality-arithmetic-mean-v2`). See [the reward contract](gepa-reflection-repair.md#scope).
- **The Rule**: Always set `max_bootstrapped_demos = 0` and `max_labeled_demos = 0` in `yaw.toml`. Optimize pure instructions zero-shot.
- **Fresh program policy (2026-10-24)**: Task and reflection constructors default to `include_demos=False`, so a fresh GEPA-only run also starts without fixed examples. Existing example sets require explicit opt-in; loaded compiled demonstrations are preserved. Their generalization has not been established by a paired benchmark.

---

## 2. Canonical `dspy.GEPA` + `dspy.Ensemble` Architecture (Active Epic 1 Rule)

### 2.1. Distinct Roles of `dspy.GEPA` vs. `dspy.Ensemble` in DSPy 3.2.x

Per the official DSPy 3.2.x documentation (`https://dspy.ai/current/api/optimizers/Ensemble`, `https://dspy.ai/current/cheatsheet`, `https://dspy.ai/current/diving-deeper/choosing-an-optimizer`) and source (`dspy/teleprompt/gepa/gepa.py`, `dspy/teleprompt/ensemble.py`, `dspy/predict/aggregation.py`):

1. **Compile-Time Pareto Evolution & System-Aware Merging (`dspy.GEPA`)**:
   - `dspy.GEPA(metric=..., candidate_selection_strategy="pareto", use_merge=True, track_stats=True, gepa_kwargs={"frontier_type": "hybrid"})` maintains a multi-objective and per-instance Pareto frontier across validation evaluations and uses `MergeProposer` (`use_merge=True`) during compilation to splice complementary predictor instructions from Pareto-frontier parents into a single merged candidate.
   - `gepa.compile()` returns one single `dspy.Module` (`new_prog = adapter.build_program(gepa_result.best_candidate)`), saved directly via `compiled.save("compiled_program.json", save_program=False)` (`FOUNDATIONAL.md` Rule 6), and attaches `new_prog.detailed_results` (`DspyGEPAResult`) containing:
     - `candidates: list[dspy.Module]` — all candidate `HippoRAGProgram` instances built via `adapter.build_program(c)`
     - `val_aggregate_scores: list[float]` — per-candidate scalar validation score (`task-quality-arithmetic-mean-v2`)
     - `val_aggregate_subscores: list[dict[str, float]]` — per-candidate objective scores (`recall_at_1`, `recall_at_2`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `em`, `f1`)
     - `per_objective_best_candidates: dict[str, set[int]]` — candidate indices achieving the best validation score per objective
     - `per_val_instance_best_candidates: dict[Any, set[int]]` — candidate indices achieving the best validation score per validation example
     - `best_idx: int` and `best_candidate: dspy.Module`.

2. **External Inference-Time Reduction Over Whole Compiled Programs (`dspy.Ensemble`) Across Both `recall@N` and `QA (F1, EM)`**:
   - `dspy.Ensemble(reduce_fn=..., size=None, deterministic=False)` ([`dspy/teleprompt/ensemble.py:10-40`](file:///c:/Users/Harri.Ahokas/Documents/yaw/.venv/Lib/site-packages/dspy/teleprompt/ensemble.py#L10-L40)) is a 30-line external `Teleprompter`:
     ```python
     ensemble = dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs[:3])
     ```
   - It wraps a list of whole compiled `dspy.Module` (`HippoRAGProgram`) instances **from the outside** into an `EnsembledProgram(dspy.Module)` whose `forward(*args, **kwargs)` executes `outputs = [prog(*args, **kwargs) for prog in programs]` and passes `outputs` to `reduce_fn(outputs)`.
   - **Why plain `reduce_fn=dspy.majority` alone is insufficient for a dual `recall@N` + `QA (F1, EM)` system**:
     - Plain `dspy.majority` ([`dspy/predict/aggregation.py:9-54`](file:///c:/Users/Harri.Ahokas/Documents/yaw/.venv/Lib/site-packages/dspy/predict/aggregation.py#L9-L54)) only inspects `completion["answer"]` (normalized Exact Match) and returns `Prediction.from_completions([completion])` from whichever single member won the `answer` vote — **discarding the retrieved rankings (`retrieved_chunk_ids`, `retrieved_passages`, `retrieved_scores`) of the other Pareto frontier programs** (giving zero ensemble lift to `recall@N`) and **ignoring token-level `F1` overlap** when all 3 normalized answers are distinct (`1-1-1` EM tie).
     - `reduce_pareto_predictions(outputs)` takes into account **both** perspectives:
       1. **`recall@N` Perspective (Reciprocal Rank + Normalized PPR Score Fusion)**: Fuses `retrieved_chunk_ids`, `retrieved_passages`, and `retrieved_scores` across all member `dspy.Prediction`s using Reciprocal Rank Fusion ($\sum_{m} \frac{1}{60 + r_m(c)} + \hat{s}_m(c)$), producing the ensemble's fused top-$K$ `retrieved_chunk_ids`, `retrieved_passages`, `retrieved_scores`, and `context`.
       2. **`QA (F1, EM)` Perspective (`dspy.majority` EM Vote + Token-F1 Medoid Tie-Break)**: Uses `dspy.majority(outputs, field="answer")` when a normalized exact-match majority exists ($\max \text{count} \ge 2$, optimizing **EM**); when all normalized answers are distinct ($\max \text{count} == 1$), selects the member whose `answer` maximizes mean pairwise token-level `compute_qa_f1` across the ensemble (optimizing **F1**, tie-broken by earliest member index).

### 2.2. Four Non-Negotiable Engineering Rules for `dspy.Ensemble` + `dspy.GEPA` in YAW

1. **`HippoRAGProgram` Is a Pure Single `dspy.Module` (External `dspy.Ensemble` Only)**:
   - A `dspy.Module` (`HippoRAGProgram`) **never** attaches an ensemble of clones onto itself (`self.pareto_ensemble`, `self.qa_ensemble`, `self._in_ensemble`), **never** stuffs duplicate predictor states into `compiled_program.json` (`yaw_surface["pareto_frontier_states"]` or `yaw_surface["qa_ensemble_states"]`), and **never** implements bespoke per-stage reducers (`ParetoProgramEnsemble`).
   - Instead, `dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs[:3])` wraps whole `HippoRAGProgram` instances from the outside. When saving `<stem>.json` (`compiled_program.json`), any non-dominated Pareto sibling programs (`programs[1:]`) are saved as standalone `HippoRAGProgram.save("<stem>_telemetry/pareto_member_{i}.json", save_program=False)` artifacts so `compile_pareto_ensemble` can reconstruct the external `dspy.Ensemble` both in-memory after `gepa.compile()` and when loading `<stem>.json` in `YawRuntime` / `yaw dspy benchmark`.

2. **`dspy.Prediction` Final-Field Invariant in `HippoRAGProgram.forward` for `dspy.majority`**:
   - In [`dspy/predict/aggregation.py:30-35`](file:///c:/Users/Harri.Ahokas/Documents/yaw/.venv/Lib/site-packages/dspy/predict/aggregation.py#L30-L35), when `dspy.majority(outputs)` is called with `field=None`:
     ```python
     if not field:
         if signature:
             field = list(signature.output_fields.keys())[-1]
         else:
             field = list(completions[0].keys())[-1]
     ```
   - On `HippoRAGProgram.forward(context=..., question=...)`, `AnswerQuestion` has `signature.output_fields` ending in `"answer"`.
   - On `HippoRAGProgram.forward(question=...)` ([`program.py:1347-1354`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L1347-L1354)), the returned `dspy.Prediction` is constructed manually (`signature=None`), so `answer=getattr(pred, "answer", "")` **must** be the **final** keyword argument:
     ```python
     res = dspy.Prediction(
         context=hydrated_ctx,
         reasoning=getattr(pred, "reasoning", ""),
         retrieved_passages=passages,
         retrieved_scores=scores,
         retrieved_chunk_ids=pids,
         answer=getattr(pred, "answer", ""),
     )
     ```
   - This guarantees both bare `dspy.Ensemble(reduce_fn=dspy.majority)` and `dspy.Ensemble(reduce_fn=reduce_pareto_predictions)` work out-of-the-box on `HippoRAGProgram`.

3. **Dual-Perspective (`recall@N` + `QA (F1, EM)`) Strictly Non-Dominated Pareto Frontier Selection (`select_pareto_frontier_programs`)**:
   - Because `compute_task_quality_score` averages 5 retrieval metrics (`recall_at_1`, `recall_at_2`, `recall_at_5`, `recall_at_10`, `recall_at_20` — $5/7$ of the scalar score) and 2 QA metrics (`em`, `f1` — $2/7$ of the scalar score), ranking a $>3$-candidate Pareto frontier solely by `val_aggregate_scores[i]` would over-select retrieval-heavy candidates and drop the `QA (F1, EM)` Pareto specialist.
   - `select_pareto_frontier_programs(compiled, max_programs=3) -> list[HippoRAGProgram]` inspects `compiled.detailed_results` (`DspyGEPAResult`) and enforces:
     1. **Strict multi-objective non-dominance**: A candidate $i$ (`val_aggregate_scores[i] > 0.0`) is eligible if and only if no candidate $j \ne i$ Pareto-dominates $i$ across `val_aggregate_subscores` ($\forall m: \text{sub}_j[m] \ge \text{sub}_i[m] - 10^{-9}$ and $\exists m: \text{sub}_j[m] > \text{sub}_i[m] + 10^{-9}$).
     2. **Never backfill dominated losers**: If `best_candidate` (`best_idx`) dominates all other candidates, `select_pareto_frontier_programs` returns `[compiled]` (length `1`).
     3. **Query-time predictor deduplication**: Deduplicate surviving non-dominated programs by their query-time predictor states (`query_intent`, `triple_filter`, `qa.predict`) so identical query/QA prompts never cast duplicate votes.
     4. **Explicit `recall@N` + `QA (F1, EM)` Perspective Coverage across the 3 Slots**:
        - **Slot 0 (`best_candidate`)**: `compiled` (`best_idx`, overall scalar leader).
        - **Slot 1 (`recall@N` Pareto leader)**: Remaining non-dominated candidate maximizing the retrieval perspective mean $\bar{R}_i = \operatorname{mean}(\text{recall\_at\_1}, \text{recall\_at\_2}, \text{recall\_at\_5}, \text{recall\_at\_10}, \text{recall\_at\_20})$ (tie-broken by `val_aggregate_scores[i]`).
        - **Slot 2 (`QA (F1, EM)` Pareto leader)**: Remaining non-dominated candidate maximizing the QA perspective mean $\bar{Q}_i = \operatorname{mean}(\text{f1}, \text{em})$ (tie-broken by `val_aggregate_scores[i]`).
        - Any unfilled slot up to `max_programs` is filled from remaining strictly non-dominated candidates by balanced perspective score $0.5\bar{R}_i + 0.5\bar{Q}_i$.
   - Pass the resulting `programs[:3]` to `dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs[:3])`.

4. **Read-Only Shared Graph Index & Unified Consumer API (`YawRuntime.retrieve`, `YawRuntime.rag_qa`, `YawRuntime.benchmark`, FastMCP)**:
   - During `gepa.compile()`, `GraphEngine._canonical_retrieval_hook` runs `_ensure_candidate_chunks_indexed(active_program)` while `_in_optimization` is `True` so each candidate's OpenIE/synonym/SHACL prompts are evaluated.
   - Once `hooks.rehydrate_sandbox_after_optimization(compiled)` finalizes `yaw.db` to `best_candidate`'s graph index (`_in_optimization = False`), `YawGraphEngine` holds a single external `ensemble = compile_pareto_ensemble(compiled)` (`dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs[:3])`).
   - **All runtime consumers (including FastMCP's `retrieve` and `rag_qa` tools in [`docs/epic-2-fastmcp-http.md`](epic-2-fastmcp-http.md)) route through this single `dspy.Ensemble`**, which dynamically judges and fuses the 3 Pareto specialists per query:
     1. **Recall API (`runtime.retrieve` / FastMCP `retrieve` / `benchmark --mode retrieval`)**: Invokes `ensemble(question=query, retrieve_only=True, k=top_k)`. `HippoRAGProgram.forward(..., retrieve_only=True)` runs only `prog.retriever(question, k=k)` (`query_intent` $\to$ `triple_filter` $\to$ PPR) and **skips `prog.answer_question` (`qa`) completely**, while `reduce_pareto_predictions` dynamically fuses the member rankings via Reciprocal Rank + normalized PPR score fusion.
     2. **QA API (`runtime.rag_qa` / FastMCP `rag_qa` / `evaluate_program` / `benchmark --mode eval`)**: Invokes `ensemble(question=question)`, executing retrieval + QA across the Pareto members and dynamically returning **both** the RRF+PPR-fused `evidence` passages (`recall@N`) and the `dspy.majority` (EM) / token-F1 medoid (F1) winning `answer`.
     3. **Per-Query Retrieval Surface Memoization**: Within a query, `_canonical_retrieval_hook` memoizes retrieval results by `(question, k, query_retrieval_fingerprint(prog))` (`query_intent` + `triple_filter` surface) so if two Pareto members share identical retrieval prompts and differ only in `qa.predict`, retrieval runs only once.

---

## 3. Query-Time Embedding Prefix Tuning

- **The Idea**: `EmbeddingGemma` uses asymmetric task prefixes (e.g. `task: search query | query: `).
- **The Leverage**: Mutating query prefixes only affects search-time encoding ($\sim 0.05\text{s}$ per query). It avoids re-embedding the corpus while allowing `dspy.GEPA` to align dense retrieval with the reasoning prompts.
- **Implementation**: Expose `query_embedding_prefix` as a `dspy.Predict` signature in `HippoRAGProgram.named_predictors()`, syncing instructions into `DSPyEmbeddingAdapter.update_prefixes()`.

---

## 4. Dynamic Hyperparameter Tuning: Simple `dspy.Predict` + GBNF

If the objective is simply tuning runtime hyperparameters (`top_k`, `ppr_alpha`, `temperature`, and similar):
- Do **not** use complex sandboxed code execution for simple configuration.
- A lightweight `dspy.Predict` with a strict GBNF JSON grammar (`{"top_k": int, "ppr_alpha": float}`) with is faster, safer, and avoids interpreter overhead.

### 4.1. Prompt example for hyperparameter tuning

```md
## Mission

Your task is to find the best hyperparameter config for fulfilling the *user request* in our GraphRAG retrieval pipeline. Follow the workflow, output only a valid JSON configuration object.

## Workflow

1. Analyse the *user request*. Consider the following aspects:
  - **Phrasing of the request**: Is the user intent ambiguous? Ambiguous requests hint at higher temperatures and multi hop retrieval.
  - **Explicit keywords or domain specific jargon**: Do the keywords clearly point to a specific entity or concept? Explicit keywords hint at lower temperatures and single passage retrieval.
2. Choose the optimal hyperparameter for each:
  - `top_k: integer`:
    - For a query that hints at need for multi-hop reasoning, set high, for a query that hints at single passage retrieval, set lower.
    - Min: 3, Max: 25, Default: 10.
  - `ppr_alpha: float`:
    - For a query that hints at need for multi-hop reasoning, set high, for a query that hints at single passage retrieval, set lower.
    - Min: 0.01, Max: 1.0, Default: 0.5.
  - `temperature: float`:
    - For a query that hints at direct hit in a chunk, set lower. Set higher to avoid loops/repetition in generation.
    - Min: 0.2, Max: 1.0, Default: 0.3.
3. Output the hyperparameter configuration as a JSON object.
  - Use the output template from the following section, filling in the datatypes as appropriate.

## Output template

{ "top_k": integer, "ppr_alpha": float, "temperature": float }

## User Request

{{user_request}}
```

---

## 5. Post-Epic-2 Roadmap: Tool-Driven Orchestration via `dspy.Flex`

The real leverage of `dspy.Flex` is **arbitrary code and tool execution**, not parameter tweaking:

- **Concept**: Expose YAW's core retrieval and graph primitives as native tools to `dspy.Flex(Signature, tools=[...])`, for example, not limited to:
  - `dense_search(query, top_k)`
  - `ppr_walk(seed_entities, alpha, top_k)`
  - `query_fts(text, top_k)`
  - `inspect_node(entity_id)`
  - `hydrate_passages(chunk_ids)`
- **Mechanism**:
  - `dspy.Flex` defaults to an internal `dspy.RLM` over these tools.
  - `dspy.GEPA` rewrites the Python orchestrator class (`module_src`), allowing the model to write arbitrary multi-step retrieval loops, adaptive backtrack heuristics, and custom graph traversal logic.
  - All generated code executes isolated inside `BridgeRuntime` (fresh sandbox per query).
- **Status**: Deferred for investigation **after Epic 2** (FastMCP HTTP transport).
- **Open Questions**:
  - Which parts of the current `HippoRAGProgram` could be replaced by `dspy.Flex` + Python tools?
  - What is the highest leverage MVP, the "ask question" part of the retrieval pipeline (F1, EM scores) or the recall pipeline (Recall@N scores)?
  - How does this tie in into the existing `dspy.RLM` modules, what is the scope of refactoring and removal?
