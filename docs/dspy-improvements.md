# DSPy Practical Improvements & Roadmap

Practical design patterns, lessons learned, and future architecture investigations for YAW.

---

## 1. Zero-Shot Optimization, Reward Contract & Runtime Binding (Active Epic 1 Rules)

- **The Issue**: Bootstrapping few-shot demonstrations on local 4B models (`Qwen3-4B-Instruct`, `Gemma-4-E4B`) causes prompt bloat ($3,500+$ tokens), generation timeouts and repetition loops. Historical token penalties were removed on 2026-10-24: GEPA now rewards the unweighted arithmetic mean of labeled retrieval recall (`Recall@1, 2, 5, 10, 20`) and QA `EM`/`F1` (`task-quality-arithmetic-mean-v2`):
  $$\text{score} = \frac{1}{|M|}\sum_{m \in M} m \quad \text{for } M \subseteq \{\text{Recall@1}, \text{Recall@2}, \text{Recall@5}, \text{Recall@10}, \text{Recall@20}, \text{EM}, \text{F1}\}$$
  while preserving all 7 raw metrics in `objective_scores` for the GEPA Pareto frontier.
- **SIMBA Counterfactual Delta Reward & Checkpoint State**: Native DSPy SIMBA instruction evaluation scores candidates on the first two training queries using the bounded delta reward:
  $$\text{reward} = 0.5 + 0.5 \cdot \frac{\Delta}{1 + |\Delta|}$$
  and persists worker state (version 2) and counterfactual task-score cache entries atomically in `yaw.db` metadata.
- **Reflection Serving Tokenizer Budget**: Reflection requests (`training traces -> sandbox evidence selection -> native DSPy InstructionProposer -> downstream task scoring`) measure the complete rendered prompt against `inference.reflection_input_tokens` using the resident llama.cpp chat template and `/tokenize` endpoint (`ServingTokenCounter`).
- **Native `YawGBNFAdapter` & Deployment-Owned LM Binding**: Task predictors bind their actual `dspy.LM` directly (no `BaseLM` proxy). `YawGBNFAdapter` subclasses `dspy.JSONAdapter` without custom fallback parsing shims and fails hard on missing/invalid fields. Loading a compiled program JSON preserves deployment-owned endpoints, credentials, and context capacities rather than letting serialized prompt metadata overwrite them.
- **The Rule**: Always set `max_bootstrapped_demos = 0` and `max_labeled_demos = 0` in `yaw.toml`. Optimize pure instructions zero-shot.
- **Fresh program policy (2026-10-24)**: Task and reflection constructors default to `include_demos=False`, so a fresh GEPA-only run also starts without fixed examples. Existing example sets require explicit opt-in; loaded compiled demonstrations are preserved. Their generalization has not been established by a paired benchmark.

---

## 2. Canonical `dspy.GEPA` + `dspy.Ensemble` Architecture (Active Epic 1 Rule)

### 2.1. Distinct Roles of `dspy.GEPA` vs. `dspy.Ensemble` in DSPy 3.2.x

Per the official DSPy 3.2.x documentation (`https://dspy.ai/current/api/optimizers/Ensemble`, `https://dspy.ai/current/cheatsheet`, `https://dspy.ai/current/diving-deeper/choosing-an-optimizer`) and source (`dspy/teleprompt/gepa/gepa.py`, `dspy/teleprompt/ensemble.py`, `dspy/predict/aggregation.py`):

1. **Compile-Time Pareto Evolution & System-Aware Merging (`dspy.GEPA`)**:
   - `dspy.GEPA(metric=..., candidate_selection_strategy="pareto", use_merge=True, track_stats=True, gepa_kwargs={"frontier_type": "hybrid"})` maintains a multi-objective and per-instance Pareto frontier across validation evaluations and uses `MergeProposer` (`use_merge=True`) during compilation to splice complementary predictor instructions from Pareto-frontier parents into a single merged candidate.
   - `gepa.compile()` returns one single `dspy.Module` (`new_prog = adapter.build_program(gepa_result.best_candidate)`), saved directly via `compiled.save("compiled_program.json", save_program=False)` (`FOUNDATIONAL.md` Rule 6), and attaches `new_prog.detailed_results` (`DspyGEPAResult`) containing:
     - `candidates: list[dspy.Module]` — all candidate `YawProgram` instances built via `adapter.build_program(c)`
     - `val_aggregate_scores: list[float]` — per-candidate scalar validation score (`task-quality-arithmetic-mean-v2`)
     - `val_aggregate_subscores: list[dict[str, float]]` — per-candidate objective scores (`recall_at_1`, `recall_at_2`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `em`, `f1`)
     - `per_objective_best_candidates: dict[str, set[int]]` — candidate indices achieving the best validation score per objective
     - `per_val_instance_best_candidates: dict[Any, set[int]]` — candidate indices achieving the best validation score per validation example
     - `best_idx: int` and `best_candidate: dspy.Module`.

2. **External Inference-Time Reduction Over Whole Compiled Programs (`dspy.Ensemble`) Across `QA` (`dspy.majority`) and Retrieval-Only (`RRF`) Modes**:
   - `dspy.Ensemble(reduce_fn=..., size=None, deterministic=False)` ([`dspy/teleprompt/ensemble.py:10-40`](file:///c:/Users/Harri.Ahokas/Documents/yaw/.venv/Lib/site-packages/dspy/teleprompt/ensemble.py#L10-L40)) is a 30-line external `Teleprompter`:
     ```python
     ensemble = dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs[:3])
     ```
   - It wraps a list of whole compiled `dspy.Module` (`YawProgram`) instances **from the outside** into an `EnsembledProgram(dspy.Module)` whose `forward(*args, **kwargs)` executes `outputs = [prog(*args, **kwargs) for prog in programs]` and passes `outputs` to `reduce_fn(outputs)`.
   - **Why `reduce_pareto_predictions` dispatches between `dspy.majority` and RRF fusion**:
     - When `outputs` contain non-empty `answer` values (`any(str(getattr(p, "answer", "") or "").strip() for p in predictions)`), `reduce_pareto_predictions(outputs)` delegates directly to DSPy's built-in `dspy.majority(list(predictions))` ([`dspy/predict/aggregation.py:9-54`](file:///c:/Users/Harri.Ahokas/Documents/yaw/.venv/Lib/site-packages/dspy/predict/aggregation.py#L9-L54)).
     - When all predictions have `answer == ""` (`retrieve_only=True`), plain `dspy.majority` would normalize every empty answer to `None` and fail to fuse the retrieved passage lists. Instead, `reduce_pareto_predictions(outputs)` fuses `retrieved_chunk_ids`, `retrieved_passages`, and `retrieved_scores` across all member `dspy.Prediction`s via Reciprocal Rank Fusion (`rrf_fuse_passage_lists`, $\sum_{m} \frac{1}{60 + r_m(c)} + \hat{s}_m(c)$), returning the fused `dspy.Prediction(context=fused_context, reasoning="", retrieved_passages=fused_passages, retrieved_scores=fused_score_tuple, retrieved_chunk_ids=fused_pids, answer="")`.

### 2.2. Four Non-Negotiable Engineering Rules for `dspy.Ensemble` + `dspy.GEPA` in YAW

1. **`YawProgram` Is a Pure Single `dspy.Module` (External `dspy.Ensemble` Only)**:
   - A `dspy.Module` (`YawProgram`) **never** attaches an ensemble of clones onto itself (`self.pareto_ensemble`, `self.qa_ensemble`, `self._in_ensemble`), **never** stuffs duplicate predictor states into `compiled_program.json` (`yaw_surface["pareto_frontier_states"]` or `yaw_surface["qa_ensemble_states"]`), and **never** implements bespoke per-stage reducers (`ParetoProgramEnsemble`).
   - Instead, `dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs[:3])` wraps whole `YawProgram` instances from the outside. When saving `<stem>.json` (`compiled_program.json`), any non-dominated Pareto sibling programs (`programs[1:]`) are saved as standalone `YawProgram.save("<stem>_telemetry/pareto_member_{i}.json", save_program=False)` artifacts so `compile_pareto_ensemble` can reconstruct the external `dspy.Ensemble` both in-memory after `gepa.compile()` and when loading `<stem>.json` in `YawRuntime` / `yaw dspy benchmark`.

2. **`dspy.Prediction` Final-Field Invariant in `YawProgram.forward` for `dspy.majority`**:
   - In [`dspy/predict/aggregation.py:30-35`](file:///c:/Users/Harri.Ahokas/Documents/yaw/.venv/Lib/site-packages/dspy/predict/aggregation.py#L30-L35), when `dspy.majority(outputs)` is called with `field=None`:
     ```python
     if not field:
         if signature:
             field = list(signature.output_fields.keys())[-1]
         else:
             field = list(completions[0].keys())[-1]
     ```
   - On `YawProgram.forward(context=..., question=...)`, `AnswerQuestion` has `signature.output_fields` ending in `"answer"`.
   - On `YawProgram.forward(question=...)` ([`program.py:1347-1354`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L1347-L1354)), the returned `dspy.Prediction` is constructed manually (`signature=None`), so `answer=getattr(pred, "answer", "")` **must** be the **final** keyword argument:
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
   - This guarantees both bare `dspy.Ensemble(reduce_fn=dspy.majority)` and `dspy.Ensemble(reduce_fn=reduce_pareto_predictions)` work out-of-the-box on `YawProgram`.

3. **Dual-Perspective (`recall@N` + `QA (F1, EM)`) Strictly Non-Dominated Pareto Frontier Selection (`select_pareto_frontier_programs`)**:
   - Because `compute_task_quality_score` averages 5 retrieval metrics (`recall_at_1`, `recall_at_2`, `recall_at_5`, `recall_at_10`, `recall_at_20` — $5/7$ of the scalar score) and 2 QA metrics (`em`, `f1` — $2/7$ of the scalar score), ranking a $>3$-candidate Pareto frontier solely by `val_aggregate_scores[i]` would over-select retrieval-heavy candidates and drop the `QA (F1, EM)` Pareto specialist.
   - `select_pareto_frontier_programs(compiled, max_programs=3) -> list[YawProgram]` inspects `compiled.detailed_results` (`DspyGEPAResult`) and enforces:
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
     1. **Recall API (`runtime.retrieve` / FastMCP `retrieve` / `benchmark --mode retrieval`)**: Invokes `ensemble(question=query, retrieve_only=True, k=top_k)`. `YawProgram.forward(..., retrieve_only=True)` runs only `prog.retriever(question, k=k)` (`query_intent` $\to$ `triple_filter` $\to$ PPR) and **skips `prog.answer_question` (`qa`) completely**, while `reduce_pareto_predictions` fuses the member rankings via Reciprocal Rank + normalized PPR score fusion (`rrf_fuse_passage_lists`).
     2. **QA API (`runtime.rag_qa` / FastMCP `rag_qa` / `evaluate_program` / `benchmark --mode eval`)**: Invokes `ensemble(question=question)`, executing retrieval + QA across the Pareto members and delegating to `dspy.majority(list(predictions))` to return the winning `dspy.Prediction`.
     3. **Per-Query Retrieval Surface Memoization**: Within a query, `_canonical_retrieval_hook` memoizes retrieval results by `(question, k, query_retrieval_fingerprint(prog))` (`query_intent` + `triple_filter` surface) so if two Pareto members share identical retrieval prompts and differ only in `qa.predict`, retrieval runs only once.

---

## 3. Query-Time Embedding Prefix Tuning

- **The Idea**: `EmbeddingGemma` uses asymmetric task prefixes (e.g. `task: search query | query: `).
- **The Leverage**: Mutating query prefixes only affects search-time encoding ($\sim 0.05\text{s}$ per query). It avoids re-embedding the corpus while allowing `dspy.GEPA` to align dense retrieval with the reasoning prompts.
- **Implementation**: Expose `query_embedding_prefix` as a `dspy.Predict` signature in `YawProgram.named_predictors()`, syncing instructions into `DSPyEmbeddingAdapter.update_prefixes()`.

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
- **Status**: Deferred for investigation **after Epic 2** (FastMCP HTTP transport) under [Epic 3](epic-3-performance.md).
- **Open Questions**:
  - Which parts of the current `YawProgram` could be replaced by `dspy.Flex` + Python tools?
  - What is the highest leverage MVP, the "ask question" part of the retrieval pipeline (F1, EM scores) or the recall pipeline (Recall@N scores)?
  - How does this tie in into the existing `dspy.RLM` modules, what is the scope of refactoring and removal?

---

## 6. Inference Backend Contract: DSPy `lm15`, Zero Provider Plumbing

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md) (Prime Directive: no fallbacks, no degraded modes) |
| Rules | [inference.md](../.agents/rules/inference.md): `yaw/daemon/graph/inference/*.py` |
| Owns code | `inference/chat.py:create_dspy_lm`, `inference/models.py`, `inference/embedding.py:create_dspy_embedder`, `inference/reflection.py`, `inference/confidence.py`, `inference/adapter.py`, `benchmark.py` preflight, `scripts/watchdog.py:get_lemonade_server_metrics` |
| Peers | [model-evaluation.md](model-evaluation.md): hardware profiles (Lemonade-specific today, superseded for serving parameters by this section); [native-cocoindex-indexing.md](native-cocoindex-indexing.md): `ChunkInference` memo identity consumes program identity; [w3c-improvements.md](w3c-improvements.md): three-layer parity depends on §6.4 decision |
| References | [DSPy lm15 migration guide](https://dspy.ai/current/community/normalized-lm-api-migration/), [Unsloth API](https://unsloth.ai/docs/basics/api) |
| Records | [docs/reviews/inference-backend.md](reviews/inference-backend.md) (created before the first code edit) |

Pinned state: HEAD `7fbec2b` on `dev`; `dspy 3.4.1`, `litellm 1.106.0` [measured: `uv run --locked python`, 7fbec2b]. The working tree holds uncommitted user edits to `adapter.py`, `grammars.py`, `models.py` that this spec does not depend on.

### 6.1 Scope and non-goals

- **Goal**: swap the inference backend (local Unsloth Studio, vLLM, paid OpenAI API) by configuration only, using DSPy's normalized `dspy.LM` / `dspy.lm15` contract. YAW writes no provider adapters, no HTTP clients, no tokenizer or registry plumbing of its own.
- **Non-goals**: benchmarking quality across backends; supporting backends that lack an OpenAI-compatible `/v1/chat/completions`; changing signatures, metrics or GEPA behavior.

### 6.2 Problem, with evidence

- YAW talks to Lemonade-only endpoints: `/health` `all_models_loaded` + `launch_command` (`models.py:embedding_evidence`, `chat_evidence`, `check_registry`), llama-server `/props`, `/apply-template`, `/tokenize` (`reflection.py:ServingTokenCounter`), `/metrics` (`scripts/watchdog.py:get_lemonade_server_metrics`). Unsloth Studio answers `/health` with 404 and has no `/tokenize` [measured: unauthenticated probe of 127.0.0.1:8888, 7fbec2b].
- **Unsloth `/v1/chat/completions` rejects `logprobs`** (the `/v1/completions` route accepts them, see below) with HTTP 400 `unsupported_parameter` [measured: `docs/reviews/inference-backend-probes/probe_unsloth2.py`, Unsloth Studio on 127.0.0.1:8888, model `gemma-4-e4b`, 2026-11-13]. YAW requests logprobs for extraction confidence (`program.py` `collect_logprobs=True`, `confidence.py:extraction_logprob_votes`, `judge_logprob_votes`).
- **On `/v1/chat/completions` a custom top-level `grammar` field is accepted but not enforced**: output was free text while `grammar = root ::= "yes" | "no"` was sent [measured: `docs/reviews/inference-backend-probes/probe_unsloth2.py`, same conditions]. The same holds for `extra_body.grammar` and `guided_json` [measured: `docs/reviews/inference-backend-probes/probe_unsloth3.py`].
- **`response_format = {type: json_schema, strict: true}` is honored**: the reply used exactly the schema's keys (`zzname`, `zzlives`) [measured: `docs/reviews/inference-backend-probes/probe_unsloth3.py`]. This is the OpenAI-standard structured-output parameter that `dspy.JSONAdapter` emits natively.
- **`POST /v1/completions` on the same Unsloth server supports both missing features** (user-supplied route, probed 2026-11-13): `logprobs = 3` returns `choices[0].logprobs.content[i].top_logprobs[{id, token, logprob}]`; a top-level `grammar` field **is enforced** (`root ::= "yes" | "no"` produced `"yes"`; the reported `top_logprobs` still show the unconstrained distribution, e.g. `"Hello"` at -0.68 while the forced token `"y"` shows -12.56); `response_format` is **not** enforced on this route (free text returned) [measured: `docs/reviews/inference-backend-probes/probe_unsloth_completions.py`, Unsloth Studio, `system_fingerprint = b12284-77bf94e7c`]. The response carries llama.cpp `timings`, so this route is the llama-server passthrough.
- **`dspy.LM("openai/gemma-4-e4b", model_type="text", api_base=..., api_key=...)` reaches that route and surfaces the logprobs** (`TextCompletionResponse`, `Logprobs.content[i]['top_logprobs']`) [measured: `docs/reviews/inference-backend-probes/probe_lm15_completions.py`]. This text path is DSPy's legacy completion client, not an `lm15.Request`; `lm15.Config(logprobs=True)` raises `TypeError: logprobs must be an int` client-side [measured: same].
- **The served Gemma model thinks by default**: `reasoning_content` is filled and consumes `max_tokens`; with 80 tokens several replies had empty `content` and `finish_reason = length` [measured: `docs/reviews/inference-backend-probes/probe_unsloth3.py`].
- **DSPy 3.4.1 already routes `dspy.LM("openai/<alias>", api_base=..., api_key=...)` through lm15** against Unsloth: an explicit `Request(Config(max_tokens=...))` returned a `Response` with `Usage(input_tokens=22, output_tokens=3, ...)`, and `dspy.Predict` under `JSONAdapter` returned a typed `Prediction`; `lm.history[-1]['usage']` carries `prompt_tokens` [measured: `docs/reviews/inference-backend-probes/probe_lm15_unsloth.py`]. `lm15.Config` exposes `max_tokens, temperature, top_p, top_k, stop, response_format, seed, logprobs, probabilities, extensions` [measured: same].
- **Unsloth publishes model aliases and context lengths** in `GET /v1/models` (`gemma-4-e4b`: `context_length = 131072`; `embeddinggemma-2`: `context_length = 8192`, `loaded = true`) [measured: `docs/reviews/inference-backend-probes/probe_unsloth.py`]. This is Unsloth-specific metadata, not a portable contract.
- **Unsloth `count_tokens` equals the response's `prompt_tokens`** for one short chat (22 and 22) [observed: single probe; Unsloth-specific endpoint].
- Unknown: embedding vector dimension via `/v1/embeddings` at the aliased model, `lm15.OpenAIChatLM.list_models` behavior against a compatible endpoint, whether `lm15.Config.logprobs` yields `Response.logprobs` on OpenAI/vLLM, and ContextLengthError normalization on Unsloth/vLLM.

### 6.3 Goals as falsifiable acceptance

**G1. Config-only backend swap.**
- **Measure**: files changed to move the same program from Unsloth to another OpenAI-compatible endpoint.
- **Reference**: `yaw.toml` plus one environment variable only.
- **Passes when**: `grep` finds zero provider/endpoint names (`lemonade`, `13305`, `unsloth`, `llama-server`, `/tokenize`, `/props`, `/health`, `all_models_loaded`) in `yaw/` and `benchmark.py` outside tests and docs [target].
- **Fails if / does not count**: an `if provider == ...` branch, a dormant Lemonade path, or an adapter class reimplementing lm15.

**G2. Structured output uses one declared parameter (Route A: `response_format`; Route B: `grammar` on text completions).**
- **Measure**: request bodies produced by every direct-LM predictor against a loopback capture server.
- **Reference**: `response_format` of type `json_schema` produced by DSPy's `JSONAdapter`.
- **Passes when**: under Route A no request carries `grammar`, `guided_json` or `extra_body` keys; under Route B every request is a text completion carrying `grammar` and no chat route is used for extraction [target].
- **Fails if**: the custom GBNF path survives as a second path after the gate.

**G3. Fail fast, no silent degrade.**
- **Measure**: behavior when the backend lacks a required capability (logprobs) or the prompt overflows.
- **Passes when**: the original provider exception (`lm15` error or HTTP 400) propagates and aborts the run with no retry-without-feature and no default confidence value [target].
- **Fails if**: a `try/except` substitutes a constant confidence, or removes `logprobs` from the request and continues.

**G4. Live acceptance on real silicon.**
- **Measure**: one end-to-end `uv run yaw dspy optimize --fresh` indexing batch of 64 chunks against Unsloth.
- **Passes when**: the batch publishes with zero raised errors and the config fingerprint matches the new defaults [target].
- **Does not count**: fake-provider tests (orchestration only, per `AGENTS.md`).

Thresholds are not loosened after seeing a run.

### 6.4 Semantics and decisions to settle before code

- **Single client path**: every LM call is `dspy.LM(f"openai/{model}", api_base=base_url, api_key=os.environ[api_key_env])`; direct calls use `lm15.Request`/`Config`/`Response`. The API key is read from the environment variable named by `api_key_env` (default `UNSLOTH_STUDIO_AUTH_TOKEN`) and a missing variable raises at startup. It never appears in `yaw.toml`, logs, tests or docs.
- **Context window**: single owner `inference.context_window_tokens` in config (already present). Overflow surfaces as the provider's error through lm15 (`ContextLengthError`) and is never caught.
- **Token counting**: the pre-call exact count (`ServingTokenCounter`) is retired. Real counts come from `Response.usage` after each call. Pre-call batch sizing (`semantic_adjudication_fits`, `SynonymRequestBudget`) and the reflection guard (`ReflectionLM`, `reflection_input_tokens`) need a replacement; see Open items.
- **Embeddings**: `dspy.Embedder` (LiteLLM-backed in 3.4.1; lm15 has no embeddings API) with `openai/<alias>`; unchanged call shape.
- **Open: structured-output and logprob route.** Constraint: three-layer parity (grammar, parser, SHACL) must keep every parser-accepted record publishable; confidence needs logprobs; no silent default; one path. Facts: on Unsloth chat completions `grammar` is ignored, `logprobs` is rejected and `response_format` json_schema is enforced; on Unsloth `/v1/completions` `grammar` is enforced, `logprobs` works and `response_format` is ignored (§6.2). Portability of each parameter [assumption, probe on vLLM and OpenAI]: `logprobs` on completions is part of the OpenAI completions API and vLLM; `grammar` is a llama.cpp field (vLLM names it differently, OpenAI has none); `response_format` json_schema is OpenAI/vLLM standard.
  - **Route A (chat + JSON Schema)**: retire GBNF; JSON Schema (`maxItems`, label `enum`, `maxLength`) carries what it can, parsers and SHACL carry cross-field rules (irreflexivity). Logprob confidence is unavailable on Unsloth chat: fail fast or retire it.
  - **Route B (text completions, `model_type="text"`)**: keeps GBNF and logprobs on Unsloth/llama.cpp, so parity and confidence survive; costs: YAW renders the chat template itself (unprobed: how DSPy's text path turns adapter messages into a prompt for Gemma), `grammar` is not portable to OpenAI, so G1 holds only for llama.cpp-class backends.
  - **Route C (hybrid by capability)**: violates G1/G2 (two paths); not proposed.
  Resolved by: user decision plus two probes (DSPy text-path prompt rendering for Gemma; `logprobs` and grammar-equivalent on vLLM). No recommendation yet: A is portable but drops logprob confidence on Unsloth, B preserves today's behavior but ties the contract to llama.cpp.
- **Open: reasoning mode.** Constraint: extraction roles must not spend `max_tokens` on thinking. Probe: `lm15.Config.reasoning` against Unsloth for `gemma-4-e4b`, pass criterion `reasoning_tokens == 0` and non-empty `content` for 20 consecutive extraction calls [target].
- **Open: pre-call sizing.** Constraint: no custom tokenizer plumbing. Candidates: a fixed pair-count batch (existing `semantic_adjudication_batch_size`, no fit check) with overflow failing fast; reflection evidence bounded by a configured character cap. Resolved by: probe over the optimize workload comparing `Usage.input_tokens` maxima to `context_window_tokens`.
- **Open: model identity and preflight.** Lemonade checkpoint/launch-flag verification has no portable equivalent. Candidate: `GET /v1/models` membership (an OpenAI-standard route, `lm15.OpenAIChatLM.list_models`) and an embedding-dimension probe. Resolved by: probe `list_models` against Unsloth.

### 6.5 Capability gates

| Reliance | Version | Probe | Pass criterion | Stop rule |
| --- | --- | --- | --- | --- |
| `dspy.LM` + lm15 against OpenAI-compatible `api_base` | dspy 3.4.1 | `docs/reviews/inference-backend-probes/probe_lm15_unsloth.py` | `Response.usage.input_tokens > 0`, JSONAdapter `Prediction` typed | already passing at 7fbec2b |
| `response_format` json_schema enforced | Unsloth Studio (version unrecorded) | `docs/reviews/inference-backend-probes/probe_unsloth3.py` | reply keys equal schema keys | passing; re-run on vLLM and OpenAI before claiming portability |
| Logprobs through `dspy.LM(model_type="text")` | dspy 3.4.1 | `probe_lm15_completions.py` | `Logprobs.content[0]['top_logprobs']` non-empty | passing on Unsloth completions; unverified on vLLM and OpenAI |
| `lm15.Config.logprobs` on chat | dspy 3.4.1 | `probe_lm15_completions.py` | `Response.logprobs` non-empty | fails client-side (`TypeError: logprobs must be an int`) and Unsloth chat returns HTTP 400: follow the §6.4 decision, do not catch |
| `ContextLengthError` normalization | dspy 3.4.1 | send an over-window prompt per backend | `lm15.ContextLengthError` raised | amend spec with the observed exception type |
| `OpenAIChatLM.list_models` | dspy 3.4.1 | call against Unsloth | returns aliases | amend spec; no hand-rolled `GET /v1/models` |

### 6.6 Removal inventory

| Retired item | Location | Replacement | Exit check |
| --- | --- | --- | --- |
| `ServingTokenCounter`, `reflection_tokenizer_url` | `reflection.py`, `config.py`, `yaw.toml`, `program.py`, `optimizer.py` | usage-based accounting (per §6.4 Open items) | `git grep -n "ServingTokenCounter\|reflection_tokenizer_url"` returns nothing |
| Lemonade registry/evidence | `models.py:check_registry,preflight,embedding_evidence,chat_evidence,select_model,get_json,canonical` | `list_models`-based identity (Open) | `git grep -n "all_models_loaded\|launch_command"` returns nothing in `yaw/` |
| Recipe/`flm`/`ollama` branches | `chat.py:create_dspy_lm`, `embedding.py:create_dspy_embedder`, `ChatProfile.recipe` | single `openai/` path | `git grep -n "ollama\|flm\|recipe"` returns nothing in `yaw/` |
| Watchdog Lemonade metrics | `scripts/watchdog.py:get_lemonade_server_metrics`, `lemonade_metrics` keys, process-name match | tokens/s from YAW usage telemetry | `git grep -n "lemonade\|13305" scripts yaw` returns nothing |
| Custom GBNF transport (only if §6.4 Route A) | `adapter.py`, `grammars.py`, `lm_supports_gbnf`, `enable_gbnf`, `gbnf_*` config | JSON Schema via `JSONAdapter` | `git grep -n "gbnf\|GBNF"` returns nothing outside docs and negative tests |
| Docs and tests naming Lemonade | `README.md`, `model-evaluation.md`, `docs/**`, `tests/**` | provider-neutral wording | the same greps over `docs` and `tests` |

### 6.7 Checkpoints

| ID | Slice | Pipeline | Exit gate |
| --- | --- | --- | --- |
| B0 | Review record created; resolve the four Open items with the user and the capability probes | `research` → parent | Decisions recorded verbatim in §6.8 |
| B1 | Client path: `create_dspy_lm`/`create_dspy_embedder` reduced to `dspy.LM("openai/...")` + `dspy.Embedder`; `api_key_env` | `simplifier` → `critic (flash)` | G1 greps for chat/embedding code; unit tests on a loopback server |
| B2 | Retire `ServingTokenCounter` and registry evidence; usage-based accounting | `simplifier` → `critic (flash)` | removal-inventory greps; full suite |
| B3 | Structured output and logprobs per §6.4 decisions | `ontologist` → `simplifier` → `critic (flash)` | G2/G3 loopback capture fixtures; parity tests green |
| B4 | Watchdog telemetry from usage; docs and test sweep | `simplifier` → `critic (flash)` | G1 grep over repo |
| B5 | Live acceptance (G4) on Unsloth | parent | `--fresh` batch of 64 chunks published; evidence in the review record |

### 6.8 Verification

| Fixture | Required result |
| --- | --- |
| Loopback OpenAI-compatible server capturing request bodies | Route A: every predictor sends `response_format` json_schema and no `grammar`/`guided_json`/`extra_body` keys; Route B: every extraction predictor posts to `/v1/completions` with `grammar` |
| Missing `UNSLOTH_STUDIO_AUTH_TOKEN` | startup raises, nothing is sent |
| Loopback server returning HTTP 400 for `logprobs` | the run aborts with the original error; no retry without the field |
| Loopback server returning a context-overflow body | `lm15.ContextLengthError` propagates uncaught |
| Real Unsloth, 64-chunk indexing batch | zero errors; `lm.history` usage totals recorded in the review record |
| `git grep` removal patterns | zero matches outside docs, tests asserting absence |

Quality gates: `uv run ty check`, `uv run ruff check --fix`, `uv run ruff format`, `uv run pytest`, `git diff --check`, plus the mandatory `critic` audit per unit (`AGENTS.md`).

### Decisions

- 2026-11-12, user: "use lm15 (https://dspy.ai/current/community/normalized-lm-api-migration/) and zero custom plumbing! Spec it."
- 2026-11-12, user: "Would maybe be nice to have a solution that allows changing the inference [...] what if, at some stage, I want to use OpenAI paid API to run yaw? And production usually likes vLLM?"
- 2026-11-12, user: inference backend is Unsloth at `http://127.0.0.1:8888/v1`, aliases `embeddinggemma-2` and `gemma-4-e4b` (implemented in commit `7fbec2b`).
- 2026-11-13, user (pointing at the completions endpoint): "You get logprobs by calling the OpenAI-compatible **completions** endpoint with the `logprobs` parameter, then reading `response.choices[0].logprobs.content[0][\"top_logprobs\"]`."
- Pending: the four Open items in §6.4.
