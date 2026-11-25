# Epic 4: Post-`U6` Retrieval Refinements (`yaw/retrieval/` & Sub-Span Indexing)

Status: planned (`0` active code ownership; activated strictly on post-`U6` `5,755`-query compiled `bench` residual failures).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md) |
| Rules | [retrieval.md](../.agents/rules/retrieval.md): `yaw/retrieval/**/*.py`; [pipeline.md](../.agents/rules/pipeline.md): `yaw/pipeline/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | None (planned post-U6 retrieval refinements; zero active code ownership) |
| Peers | [epic-3-performance.md](epic-3-performance.md): prerequisite U5-U6 librarian wiki & compiled 5,755-query bench baseline; [native-cocoindex-graph.md](native-cocoindex-graph.md): target `yaw/retrieval/**/*.py` kernel for Mechanisms 2 and 3 |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md), [model-evaluation.md](model-evaluation.md) |
| Supersedes / superseded by | None (activated strictly on U6 compiled bench residual failures) |
| Records | [evaluation.md](evaluation.md), [evaluation-graph-storage.md](evaluation-graph-storage.md), [mdl-dag-experiment-report.md](reviews/mdl-dag-experiment-report.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md) |

---

## 1. Entry Gate (Post-`U6` Empirical Trigger)

- Epic 4 runs **only after `U6`** removes the uncompiled bare-LM confound via `dspy.GEPA` prompt compilation over `enterprise_rag_bench_dev` and evaluates compiled programs across the full `5,755`-query `bench` (`enterprise_rag_bench_bench` + all `7` `HippoRAG_2` datasets).
- Any candidate refinement promotes **strictly** via paired `Holm`-adjusted `95%` bootstrap intervals over frozen `M_FINAL_1B` and budget-matched random-after-seeds controls.
- Held-out `[0, 10)` slices stay strictly untouched (`0` overlap during discovery or tuning), and `0` tunable retrieval hyperparameters are permitted.

## 2. Closed / Rejected — Mechanism 1 (Query-Bit Residual Set Cover & `M1C_QWEIGHTED_DPP`)

- **Permanently rejected** on mathematical and empirical grounds:
  - The `3`-round Walsh-Hadamard rotation in `yaw/retrieval/rabitq.py` delocalizes semantic facets uniformly across all `768` rotated coordinates, yielding isotropic `|q_rot_j|` weights rather than facet-aligned axes.
  - At cosine `c = 0.60`, bit agreement is `~70.5%` across candidates, leaving query-weighted residual bit cover without separable facet signal.
  - `EVAL-3c` already falsified query-weighted bit residuals (`E1B_RESIDUAL_COV`: `-6.01 pp`; `E2A`/`E2B`: `-4.67 / -6.00 pp`), recorded in [mdl-dag-experiment-report.md](reviews/mdl-dag-experiment-report.md).
  - Neither `M1C_QWEIGHTED_DPP` nor bit-residual set cover enters implementation or re-ablation.

## 3. Mechanism 2 — Graph & Diffusion Precision

- **Empirical trigger:** Activated **iff** `MuSiQue 3/4-hop AH@5` or multi-hop `ERB` lags after `U6` compiled QA evaluation.
- **Candidate refinements (`yaw/retrieval/graph.py`, `yaw/retrieval/search.py`):**
  - Sentence-scoped `prop_nouns` extraction preventing cross-sentence false co-occurrence bridges inside multi-sentence chunks.
  - Asymmetric bridge weighting reflecting directional entity-to-target evidence flow across multi-hop chains.
  - Tangent-steered PMI refinements preserving `M_FINAL_1B`'s parameter-free convergence and exact adjoint invariants.

## 4. Mechanism 3 — Sub-Span Representation & Vectorized Per-Doc MaxP

- **Empirical trigger:** Activated **iff** multi-gold dense retrieval (`nq`, full-corpus `PopQA`) or high-`N` MaxP latency lags after `U6`.
- **Candidate refinements (`yaw/pipeline/flow.py`, `yaw/retrieval/search.py`):**
  - Title-propagated 1-bit sub-span codes paired with parent `MaxSim` aggregation when long multi-fact articles dilute single-chunk vectors.
  - `±1` adjacent-chunk fallback on headingless `.txt` prose where CocoIndex AST islands default to `(sl, el)` (`yaw/pipeline/flow.py`).
  - Vectorized per-doc MaxP replacing Python loop overhead in `_select_ranked_positions` (`yaw/retrieval/search.py`) while preserving exact tie-breaking.

## 5. Decisions

- `2026-11-25`: User decided to skip Mechanism 1 permanently, proceed directly from `U5` (`wiki_root`, `4+4` MCP, `C901 <= 15`) -> `U6` (`GEPA` compilation + `5,755`-query `bench` evaluation), and gate Mechanisms 2 and 3 strictly on post-`U6` empirical residual failures.
