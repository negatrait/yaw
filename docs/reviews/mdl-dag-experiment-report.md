# Benchmark-Gated Evaluation of Parameter-Free 1-Bit Tangent-Steered Citation Diffusion (`M_FINAL_1B`) and Full Ablation History (`EVAL-0`–`EVAL-3f`)

**Status:** `EVAL-3f` Confirmatory Holdout Complete — **`M_FINAL_1B` User-Approved for `EVAL-5` Production Cutover** (`EVAL-4` Manuscript Update Pending Whole-Repo Gates & Flash Critic; `0` edits to `yaw/**`, `pyproject.toml`, `yaw.toml`, [FOUNDATIONAL.md](../../FOUNDATIONAL.md), or [AGENTS.md](../../AGENTS.md) on `dev`)

| Relation | Reference |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../../FOUNDATIONAL.md), [AGENTS.md](../../AGENTS.md), [epic-3-performance.md](../epic-3-performance.md#decisions) |
| Rules | [docs.md](../../.agents/rules/docs.md), [graph-storage.md](../../.agents/rules/graph-storage.md), [tests.md](../../.agents/rules/tests.md), [canonical-files.md](../../.agents/rules/canonical-files.md) |
| Specification & Ledger | [mdl-dag-sprint.md](mdl-dag-sprint.md) (`Rev 4`) |
| Evaluation Ledgers | [evaluation.md](../evaluation.md), [evaluation-graph-storage.md](../evaluation-graph-storage.md) |
| Peer Specs | [bare-bones-replacement-sprint.md](bare-bones-replacement-sprint.md), [epic-3-fts-vec-graph-experiment.md](epic-3-fts-vec-graph-experiment.md), [substate-manifold-shootout.md](substate-manifold-shootout.md) |
| Code Provenance | Pinned `dev` baseline: `3cd3952` (`66e9a9e` `EVAL-0`, `9078d18` initial `M_PURE_0P` `EVAL-4`); isolated lab: branch `exp/mdl-dag-ablation` @ `5b58ada` (`032fe2d` `EVAL-3`, `325eea3` `EVAL-3b`, `d29531a` `EVAL-3c`–`3e`, `5b58ada` `EVAL-3f`) |

---

## 1. Abstract & Executive Gate Verdict

We evaluate a zero-LLM, zero-tunable-hyperparameter graph-diffusion retrieval architecture across six iterative and confirmatory stages (`EVAL-0`..`EVAL-3f`) over `37,570` documents (`52,959` chunks across five benchmarks) against YAW's unmodified clean-slate baseline (`B3_CS6`, [retrieval.py](../../yaw/retrieval.py) at `3cd3952`) under preregistered gates `P0`–`P5`, targeting atomic elimination of `sqlite-vec`/`vec0`, `optuna`, and `spacy`/`en-core-web-sm`.

1. **Pre-`3b` Trajectory (`M_FULL` $\to$ `M_WIN` $\to$ `M_WIN_0P` $\to$ `M_PURE_0P`):** An intermediate finite-horizon hybrid (`M_WIN_0P`) passed Tier B discovery yet **failed sealed Tier C** on reject calibration (`P4`: `0.6432 < 0.70 < top1 0.6804`) and component necessity (`P5`: `7/21 PASS`). Pivoting to parameter-free converged geometric-restart ($\rho_{\text{stop}} = 0.5$) citation + 1-bit PMI diffusion over 4-bit RaBitQ (**`M_PURE_0P`**) passed `P0`–`P4` on untouched disjoint `Tier D` (`N = 492`, `Pool R@5 = 0.8010`, `MH AH@5 = 0.7033`, `Rej = 0.7976`, `n = 100` `P3` QA `EM = 0.3800 vs 0.3300`), yet failed `P5` (`5/8 PASS`) because incumbent `unicode61` FTS5 tokenization (`AP3_UNICODE61_LEX`) outperformed `trigram` on Tier D (`0.8098 vs 0.8010`, `p95 = 64.5 vs 86.2 ms`) and Tier C (`0.8039 vs 0.7913`) while `trigram` inflated `chunk_fts_data` `5.61x` (`256.59` vs `45.76 MiB`).
2. **`EVAL-3b` Fair Tokenizer Shootout & 1-Bit SoA Pivot (`exp/mdl-dag-ablation` @ `325eea3`):** Harmonizing Unicode word/sentence regexes, unified decompounding (`_run_decomp_for_word`), and non-overlapping CJK bigram PMI across both tokenizers, and building bloat-free `candidate_unicode61.db` (`246.43 MiB` total 5-corpus DB footprint; `chunk_fts_data = 43.56 MiB`, **`5.89x` smaller** than `candidate_trigram.db` `256.59 MiB`), yielded 1-bit SoA RaBitQ **`M_PURE_U61_1B`**: `Tier D Pool R@5 = 0.8128`, `4-DS H4 = 0.7890`, `MH AH@5 = 0.7133`, `Rej = 0.7855`, `RAM = 22.55 MiB`, `p50/p95 = 23.1 / 44.3 ms`. However, `P5` (`6/8 PASS`) and paired `n = 100` `dspy.ReAct` QA (`P3 = FAIL`, `EM 0.3900 vs 0.4000`) motivated dual-operator discovery.
3. **`EVAL-3c`–`3e` Dualities, 16-Arm Factorial Lattice & Optimization (`exp/mdl-dag-ablation` @ `d29531a`):** Testing 13 parameter-free 1-bit dualities on `tier_b_dev` (`N = 243`) and running the full $2^4 = 16$-arm factorial lattice over `{F1, E1A, E3A, F4B}` identified `#1 C2_F1_E1A` (`Pool R@5 = 0.8155 [+0.0003, +0.0329]`, `4-DS H4 = 0.8000`, `MH AH@5 = 0.7200`, `ERB = 0.8878`, `PopQA = 0.6100`). Replacing query-time `betainc`/CSR rebuilds with static CSR buffers + a `16,384`-bin piecewise-linear `z = sqrt(-log1p(-c^2))` Beta-tail LUT sped up `C2_F1_E1A` `1.45x / 1.63x` (`37.51 / 83.47 -> 25.91 / 51.12 ms`) with `0` `Recall@{1,2,5,10}` / `AH@5` diffs, freezing **`M_FINAL_1B`** (`1-bit RaBitQ + Tangent-Steered Cyclic Diffusion + 1-bit DPP`).
4. **Untouched Disjoint `Tier E` Holdout (`EVAL-3f`, `exp/mdl-dag-ablation` @ `5b58ada`, `4` Public Datasets / `36,262` Docs / `40,157` Chunks, `N = 400`, `200` Reject Labels, `n = 200` Paired `P3` QA, Plus 5-Corpus `52,959`-Chunk IEEE `float16 -> float32` Parity):**
   - **`M_FINAL_1B` clears `P0`, `P1`, `P2`, `P3`, and `P4`:** `4-DS H4 = Pool R@5 = 0.8006` vs `B3_CS6 0.6208` (`+0.1798 [+0.1552, +0.2048]`, $p_{\text{Holm}} = 0.0004$), `MH AH@5 = 0.7100` vs `0.4067` (`+0.3033 [+0.2500, +0.3600]`, $p_{\text{Holm}} = 0.0004$), `MuSiQue = 0.7175 / 0.4100`, `Reject AUROC = 0.7068 >= 0.70 > top1_cosine 0.6345` (`0` masked leaks), `maxRAM / maxTM = 24.73 / 78.73 MiB` (`3.78x` below `B3_CS6 93.55 MiB`), `p50/p95 = 22.1 / 38.8 ms`, `0` concurrency errors, and **paired `n = 200` `gemma-4-e4b` `dspy.ReAct` QA (`0` errors across `400` runs): `EM = 0.4500 vs 0.4100` (`+4.00 pp`), `F1 = 0.6219 vs 0.5569` (`+6.50 pp`, $p_{\text{perm}} = 0.0022$, $d_z = +0.2117$)**.
   - **`P5` (`5/10 PASS`) & `float16` Parity:** Core diffusion (`AF3 +0.1163/+0.2300`), title citations (`AF4 +0.0981/+0.1667`), 1-bit PMI (`AF5 +0.0154/+0.0400`), conditional Noisy-OR (`AF6 +0.0163/+0.0367`), and MaxP deferral (`AF7 +0.0054/+0.0133`) clear `P5`, while `AF1_NO_RABITQ_EXACT` (`float32` outperformed 1-bit by `+0.0225 AH@5` at `2.39x` RAM), `AF2`, `AF8`, `AF9`, and `AF10` miss standalone `N = 400` LOO thresholds (`Tier E` evaluates the `4` public datasets [`36,262` docs / `40,157` chunks], excluding `ERB` whose `150` queries were exhausted in `Tiers A–D`). Across all `5` full corpora (`37,570` docs / `52,959` chunks), IEEE `<f2>` (`float16`) storage causes only `0.01057%` (`4,298 / 40,672,512`) 1-bit RaBitQ flips and **`0.0000` `Tier E` `4-DS H4` / `MH AH@5` diff**. Post-freeze Tier D & C runs confirm `ERB = 0.9152` and `Tier D Pool R@5 = 0.8149` (`4-DS H4 = 0.7919`). **The user approved `M_FINAL_1B` for the `EVAL-5` production cutover (§9.2).**

### Executive Gate Matrix (`P0`–`P5`)

| Gate | Requirement Summary | `M_WIN_0P` *(Tier C, Superseded)* | `M_PURE_0P` *(Tier D, Superseded)* | `M_PURE_U61_1B` *(Tier D, `EVAL-3b`)* | **`M_FINAL_1B`** *(`Untouched Tier E`, Frozen)* | **`M_FINAL_1B`** *(`Post-Freeze Tier D / Tier C`)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`P0`** | `0` Class-1 tunable retrieval hyperparameters | **PASS** *(0 scalars; kept `L<=3`)* | **PASS** *(0 scalars, 0 depth/budget)* | **PASS** *(0 scalars, 1-bit `u61`)* | **PASS** *(0 scalars, 0 caps)* | **PASS** *(0 scalars, 0 caps)* |
| **`P1`** | Beat `B5S` & `B5A` on `R@5` & `AH@5` ($p_{\text{Holm}} < 0.01$, CI $> 0$) | **PASS** (`p_holm = 0.0008`) | **PASS** (`p_holm = 0.0008`) | **PASS** (`p_holm = 0.0008`) | **PASS** (`p_holm = 0.0008`) | *comp* (`0` leaks) |
| **`P2`** | `MH AH@5` $\Delta \ge +0.030$, `macro R@5` $\Delta \ge +0.020$, non-inf | **PASS** (`+0.2800` / `+0.1576`) | **PASS** (`+0.2800` / `+0.1382`) | **PASS** (`+0.2900` / `+0.1501`) | **PASS** (`+0.3033` / `+0.1798`, all DS $> 0$) | *comp* (`D: 0.8149`, `C: 0.7980`) |
| **`P3`** | Paired `dspy.ReAct` QA non-regression vs `B3_CS6` | `NOT_RUN` | **PASS** (`n=100`: `+0.0500` EM) | **FAIL** (`n=100`: `-0.0100` EM) | **PASS** (`n=200`: `EM +0.0400`, **`F1 +0.0650`**, `p=0.0022`) | *covered in Tier D/E QA* |
| **`P4`** | RAM $\le$ `B3_CS6`, `p95 <= cap`, `0` conc errs, `Rej >= 0.70` & $>$ `top1` | **FAIL** (`0.6432 < 0.6804`) | **PASS** (`25.45 MiB`, `86.2 ms`, `0.7976`) | **PASS** (`22.55 MiB`, `44.3 ms`, `0.7855`) | **PASS** (`24.73 MiB`, `38.8 ms`, `0.7068 > 0.6345`) | **PASS** (`25.34 MiB`, `50.0 / 51.6 ms`, `0.7855 / 0.7703`) |
| **`P5`** | Every component justified by LOO (Occam's razor) | **FAIL** (`7/21 PASS`) | **FAIL** (`5/8 PASS`; `AP3` $>$ `tri`) | **FAIL** (`6/8 PASS`; `A1B2`/`A1B8`) | **FAIL** (`5/10 PASS`: `AF3`–`AF7` PASS; `AF1` `sys_win=False`) | *comp only* |
| **Overall** | Strict conjunction / Cutover disposition | **REJECT / SUPERSEDED** | **SUPERSEDED BY `3b`–`3f`** | **INTERMEDIATE STEP** | **`P0`–`P4` PASS (`P5` `5/10`) -> USER-APPROVED FOR `EVAL-5`** | **CONFIRMED PARITY** |

---

## 2. Mathematical Formulation & Architectural Evolution

All 10 pre-implementation engineering and statistical defects discovered at `3cd3952` during `EVAL-0` (including broken `corpus_root` inference, `sqlite-vec` `vec0` `IN (...)` full-table scans, `tokenize='trigram'` `.casefold()`/EOF CJK handling, `2-/3`-cycle Gibbs walks, double-JSON `extract_text_features` shims, cross-object `_` access, and `[10, 60)` split leakage) were resolved in the isolated lab (`exp/mdl-dag-ablation`) prior to live evaluation; see [mdl-dag-sprint.md §1.3](mdl-dag-sprint.md#13-ten-empirical-pre-implementation-defects-caught--fixed-at-3cd3952-eval-0).

### 2.1 Why Earlier Candidates Were Superseded

1. **`M_FULL` -> `M_WIN` (`H5A_NORM_4BIT`, Tier B only):** Early `M_FULL` collapsed ranking (`Rank_Efficiency = 0.4688`) because unnormalized `0.05`-nat edge costs let length-3 paths overpower Hop 0 by $\sim 17{,}000\times$ and `1`-bit `RaBitQ` required `3,476.6` exact `vec0` fetches/query (`p95 = 1,225.9 ms`). Switching to persisted 4-bit AoS Extended Rotated `RaBitQ` (`51.7` fetches, `p95 = 132.1 ms`), exact Kraft stopped-path priors ($\sum_P \pi(P) \le 1$), orthogonal step potential $q_\perp$, `cone_bridge_mdl` decode, and dimension-normalized `dim_norm_log_bf` fixed Tier B (`M_WIN` `9/13` LOO), yet still carried Class-1 heuristic budgets/cutoffs (`horizon=2`, `wave_cap=6`, `clone_jaccard_threshold=0.75`, `compact_search_budget` caps).
2. **`M_WIN_0P` (`EVAL-2d` -> Failed Sealed Tier C `P4` & `P5`):** Eliminating Class-1 scalars via clean title-ref + PMI + proper-noun CSR (`entity_graph_blob`), self-entity-free Kraft predicate routing, tempered `L <= 3` simple-path Gibbs, and 3-hop non-backtracking Entity-HMM BMA raised Tier B to `Pool R@5 = 0.8381`, but **failed sealed Tier C** (`P4` `Reject AUROC = 0.6432 < top1 0.6804`, `P5` `7/21 PASS`) while retaining hardcoded structural depth/engine heuristics (`horizon=3`, `L <= 3` simple paths, 3-step Entity-HMM unrolling, two-engine BMA).

### 2.2 Parameter-Free Converged Citation Diffusion (`M_PURE_0P`)

Under the user's strict mandate removing all arbitrary depth, budget, and weight hyperparameters, **`M_PURE_0P`** (`scripts/mdl_dag_lab/engine.py` @ `032fe2d`) operates directly over the **complete eligible full-corpus 4-bit AoS Extended Rotated RaBitQ cosine field** ($\hat{c}(q, v_i)$, `bnb_exact_fetches = 0`, zero Hop-0 truncation or candidate-subgraph caps). Let $N_{\text{eff}}$ denote the count of eligible chunks (`eligible` boolean mask; $N_{\text{eff}} = N$ when unfiltered) and let $D_{\text{eff}} = \exp\!\big(-\sum_j p_j \ln p_j\big)$ denote the Shannon spectral entropy dimension over normalized centered covariance eigenvalues (2,048-sample reservoir).

1. **Factored $O(\text{nnz})$ Citation + Non-Self-Loop PMI Operator (`CitationDiffusionOps`):**
   Built directly over eligible chunks $E \subseteq \{0,\dots,N-1\}$ and eligible documents $D$ (rebuilt strictly over `eligible` whenever any mask or `path_glob` is active so masked chunks receive `0` mass in denominators and `inv_m`):
   - **Document ownership (`D_own`):** $D_{\text{own}}[u, d] = 1 / |\{w \in E : \text{doc}(w) = d\}|$ for chunk $u \in E$ owned by $d$; $D_{\text{id}}^T[d, v] = \mathbf{1}[v \in E \land \text{doc}(v) = d]$.
   - **Directed forward title citation (`M_fwd`):** Chunk $u \in E$ mentioning title of eligible document $d$ with out-degree $d_{\text{out}}(u) > 0$ contributes $M_{\text{fwd}}[u, d] = 1 / d_{\text{out}}(u)$, yielding column-to-chunk step $D_{\text{own}} (M_{\text{fwd}}^T x)$.
   - **Exact Bayesian reverse citation (`R_rev`):** Defining document in-weight $w_{\text{in}}(d) = \sum_{u \in E} M_{\text{fwd}}[u, d]$, $R_{\text{rev}}[u, d] = M_{\text{fwd}}[u, d] / w_{\text{in}}(d)$ whenever $w_{\text{in}}(d) > 0$, yielding step $R_{\text{rev}} (D_{\text{id}}^T x)$.
   - **Self-loop-cancelled 1-bit PMI (`P2C`, `C2P`):** Over 2-word collocations satisfying Church–Hanks $\ge 1$-bit PMI ($N \cdot \text{df}(ab) \ge 2\,\text{df}(a)\,\text{df}(b)$, $\text{df} \in [2, d_{\text{bridge}}]$ across $\ge 2$ docs), define chunk-term weight $B_{\text{pmi}}[u, p] = \ln((N_{\text{eff}} + 1) / (\text{df}_p + 0.5))$, row-normalized $C_{2P} = \text{diag}(r_{\text{sum}}^{-1}) B_{\text{pmi}}$, and column-normalized $P_{2C} = B_{\text{pmi}} \text{diag}(c_{\text{sum}}^{-1})$. The exact diagonal self-loop weight is $s_u = \sum_p C_{2P}[u,p] P_{2C}[u,p]$. Nodes with $1 - s_u > 10^{-6}$ rescale $x_p = x_m / (1 - s_u)$ and apply $\max\big(0,\, P_{2C}(C_{2P}^T x_p) - s \odot x_p\big)$.
   - **Uniform channel average (`inv_m`):** Letting $m_{\text{cnt}}(u) = \mathbf{1}[\text{fwd}(u)] + \mathbf{1}[\text{rev}(u)] + \mathbf{1}[\text{pmi}(u)]$, every active node averages channels uniformly via $\text{inv\_m}(u) = 1 / m_{\text{cnt}}(u)$ (`0` arbitrary channel weights):
     $$y = (D_{\text{own}} M_{\text{fwd}}^T)(x_m) + (R_{\text{rev}} D_{\text{id}}^T)(x_m) + \max\!\big(0,\, (P_{2C} C_{2P}^T)(x_p) - s \odot x_p\big), \qquad x_m = (x \odot e) \odot \text{inv\_m}.$$

2. **Hop-0 Query Potentials, Bayes Factor & CLT Temperature:**
   - Exact scale laws: $d_{\text{bridge}} = \max(4, \lceil \sqrt{N_{\text{eff}}} \rceil)$, $d_{\text{query}} = \max(8, \lfloor N_{\text{eff}} / 2 \rfloor)$, and CLT temperature $T_{\text{path}} = \sqrt{\max(1, D_{\text{eff}}) / \max(1, \ln(N_{\text{eff}} + 1))}$.
   - Dense surprise uses regularized incomplete Beta null nats $s_{\text{sem}}(u) = I_{\text{null}}(\max(0, \hat{c}(q, v_u));\, D_{\text{eff}})$, summed with unweighted Jeffreys IDF over FTS5 anchors and query title-reference hits $s_{\text{lex}}(u) = \sum_a \ln((N_{\text{eff}}+1)/(\text{df}_a+0.5)) + \sum_t \ln((N_{\text{eff}}+1)/(\text{df}_t+0.5))$.
   - Seed distribution $a_0(u) \propto \exp\big((s_{\text{sem}}(u) + s_{\text{lex}}(u)) / T_{\text{path}}\big)$, semantic-only transition likelihood $l_{\text{vec}}(u) \propto \exp(s_{\text{sem}}(u) / T_{\text{path}})$, and exact Hop-0 log Bayes factor:
     $$\log \text{BF} = \ln\!\left(\frac{1}{N_{\text{eff}}} \sum_{u \in E} \exp\!\left(\frac{s_{\text{sem}}(u) + s_{\text{lex}}(u)}{T_{\text{path}}}\right)\right), \qquad \text{reject\_score} = -\log \text{BF}.$$

3. **Converged Geometric-Restart ($\rho_{\text{stop}} = 0.5$) Diffusion, Noisy-OR & MaxP Deferral:**
   Starting from $p^{(0)} = 0.5\, a_0$ with $a_{\text{cur}} = a_0$ and weight $w_h = 0.5^{h+1}$, iterate without any horizon cap until $w_h < 1 / \max(2, N_{\text{eff}})$ (weight tolerance) or $\sum (\text{step}(a_{\text{cur}}) \odot l_{\text{vec}}) \le 10^{-12}$:
   $$a_{\text{next}} = \frac{\text{step}(a_{\text{cur}}) \odot l_{\text{vec}}}{\|\text{step}(a_{\text{cur}}) \odot l_{\text{vec}}\|_1}, \qquad p \leftarrow p + w_h \, a_{\text{next}}.$$
   Upon fixed-point convergence $\|a_{\text{next}} - a_{\text{cur}}\|_\infty \le 10^{-9}$, add the exact closed-form geometric series tail $\sum_{k \ge 1} 0.5^k w_h a_{\text{next}} = w_h a_{\text{next}}$ and halt, renormalizing $p$. When $|T_q| \ge 2$ query title entities exist, each entity group $g$ diffuses from its semantic-only seed weighted by $w_g = \sigma(\max_{u \in g} s_{\text{sem}}(u) - \ln(N_{\text{eff}}+1))$ via Bayesian Noisy-OR $p_{\text{post}} = 1 - (1 - p)\prod_g (1 - w_g p_g)$. Finally, candidates sorted by `(-post[u], -cos[u], chunk_id)` emit one top chunk per distinct relative file path first (MaxP deferral) before remaining intra-document chunks.

### 2.3 Bloat-Free 1-Bit `unicode61` & Frozen `M_FINAL_1B` Extensions (`EVAL-3b`–`EVAL-3e`)

Following `EVAL-3`'s finding that `unicode61` outperformed `trigram` while cutting `chunk_fts_data`, `EVAL-3b`–`EVAL-3e` refined `M_PURE_0P` into **`M_FINAL_1B`** (`exp/mdl-dag-ablation` @ `d29531a`) with zero Class-1 tunable hyperparameters:

1. **1-Bit SoA Extended Rotated RaBitQ (`rabitq_bits = 1`, `M_PURE_U61_1B`):**
   Each unit vector $y_i = R\hat{v}_i$ is quantized purely to packed 1-bit sign codes $\bar{x}_i \in \{\pm 1/\sqrt{D}\}^D$ plus one `float32` scaling norm $s_i = \langle \bar{x}_i, y_i \rangle$ (`ip_norm`), reducing Stage-1 vector memory below 4-bit AoS while estimating unbiased query cosine via $\hat{c}(q, v_i) = \langle \bar{x}_i, R\hat{q} \rangle / \max(s_i, 10^{-6})$. Document sign centroids (`doc_codes`, `0.0` ties broken by the first eligible chunk) permit exact bitwise Hamming edge angles $\theta_{ud} = \pi h_{ud} / D$ and spherical edge cosines $c_{ud} = \cos\theta_{ud}$ (`_compute_edge_ham_cos`).
2. **Fair Script-Aware `unicode61` FTS5 + Shared Morphology (`candidate_unicode61.db`):**
   Harmonized Unicode word/sentence regexes, consolidated Koehn–Knight compound splitting (`_run_decomp_for_word`), and non-overlapping CJK bigram PMI equally across tokenizers; `candidate_unicode61.db` uses contentless `fts5(..., tokenize='unicode61 remove_diacritics 1')` normalized with `.casefold()` + overlapping CJK bigrams (`_normalize_fts_unicode61`), persistent `chunk_vocab` (`fts5vocab`), and native `stem*` prefix seeks.
3. **Query-Tangent Steered Cyclic Diffusion (`F1_TANGENT_STEER` + `16,384`-bin Beta-LUT):**
   For every forward chunk $\to$ document title-reference edge $(u \to d)$, clipped edge cosine $c_{ud} = \operatorname{clip}(\cos\theta_{ud}, \pm 0.999)$ and query alignments $c_{qd} = \operatorname{clip}((D_{\text{own}}^\top c_q)_d, \pm 0.9999)$ and $c_{qu} = \operatorname{clip}(c_{q,u}, \pm 0.9999)$ determine directional partial correlations orthogonal to the starting point (`_apply_tangent_steer_ops`):
   $$\rho_{\text{fwd}}(u \to d \mid q) = \operatorname{clip}\!\left(\frac{c_{qd} - c_{qu} c_{ud}}{\sqrt{(1 - c_{qu}^2)(1 - c_{ud}^2)}},\, \pm 0.999\right), \qquad \rho_{\text{rev}}(d \to u \mid q) = \operatorname{clip}\!\left(\frac{c_{qu} - c_{qd} c_{ud}}{\sqrt{(1 - c_{qd}^2)(1 - c_{ud}^2)}},\, \pm 0.999\right).$$
   Signed surprisal nats $\operatorname{sgn}(\rho)\, I_{\text{null}}(|\rho|;\, D_{\text{eff}} - 1) / T_{\text{path}}$ are evaluated via a `16,384`-bin piecewise-linear lookup table parameterized by $z = \sqrt{-\ln(1 - \rho^2)}$ (`max abs err < 1e-6`, replacing per-query `scipy.special.betainc`) and row/column softmax-normalized over pre-built static CSR index buffers (`edge_fwd_indptr`, `edge_t_order`, `edge_t_indptr`, `edge_t_indices`, `edge_inv_s_ud`) to rewire $M_{\text{fwd}}^\top$ and $R_{\text{rev}}$ per query with zero COO->CSR allocations.
4. **Exact Rank-1 Cholesky Greedy L-Ensemble 1-Bit DPP (`E1A_DPP_1BIT`):**
   Over the top-$m_{\text{pool}} = \min(n_{\text{distinct}}, m_{\text{law}})$ distinct-document MaxP-deferred candidates with $m_{\text{law}} = \lceil 2\ln(N_{\text{eff}} + 1) \rceil$ (selected from candidate buffer $k_{\text{pool}} = \min(N_{\text{elig}}, \max(m_{\text{target}} \cdot 8, 256))$ guarded by `p_cut` dominance), `dpp_1bit` locks $d_1 = \arg\max p$, forms the 1-bit Gram matrix $G = \bar{X}_{\text{pool}}\bar{X}_{\text{pool}}^\top$, initializes marginal gains $d_i = 1$, and iteratively picks $\arg\max_i \ln(1 + N_{\text{eff}}\,p_i\,d_i)$ followed by exact rank-1 Cholesky diagonal downdate $d_j \leftarrow \max(0, d_j - c_j^2)$.

### 2.4 Gate `P0` Zero-Hyperparameter Audit

Both `M_PURE_0P` and frozen **`M_FINAL_1B`** eliminate every Class-1 tunable scalar from the system (`0` scoring weights, `0` blend weights, `0` column weights, `0` clone Jaccard thresholds, `0` Wave/Hop/subgraph caps, `0` Optuna parameters). Only Class-2 canonical identities ($\rho_{\text{stop}} = 0.5$, $\ge 1$-bit PMI $2\,\text{df}_1\text{df}_2$, Jeffreys $+0.5$, scale laws $d_{\text{bridge}}, d_{\text{query}}, m_{\text{law}} = \lceil 2\ln(N_{\text{eff}}+1) \rceil$), FTS5 structural minima, and numerical/hardware constants (`rabitq_bits=1`, `_RABITQ_SEED=0x594157`, `16,384` LUT bins, $\ell_\infty = 10^{-9}$) remain.

---

## 3. Experimental Setup, Paired Indexes & Split Provenance

### 3.1 Environment, Software Stack & Live Models

- **Host & OS:** `Windows-11-10.0.26200-SP0`, 24 logical CPUs, Python `3.14.9` (`uv --offline --locked`).
- **Locked Dependencies:** `cocoindex==1.2.0`, SQLite `3.53.1`, `sqlite-vec==0.1.9`, `scipy==1.18.1`, `numpy==2.5.4`, `dspy==3.5.0` (control baseline `B3_CS6` calls unmodified `3cd3952` [retrieval.py](../../yaw/retrieval.py) + `optuna==5.0.0` / `spacy==3.8.18`).
- **Local Models (`http://127.0.0.1:8888/v1`):** `embeddinggemma-2` (300M parameters, `D = 768`) for all paired corpus/query embeddings; `gemma-4-e4b` (`temperature = 1.0`, zero-shot uncompiled `dspy.ReAct` with 4 read-only tools) for `P3` confirmatory QA.

### 3.2 Full-Corpus Paired Indexes (`37,570` `.md` Docs / `52,959` Chunks)

Paired CocoIndex `1.2.0` ingestion shared `@coco.fn(memo=True)` `_embed_chunk` across all DB targets (`EVAL-0` initial paired build generated `incumbent_unicode61.db` + `candidate_trigram.db`; `EVAL-3b` added the decoupled Stage-A/Stage-B writer producing bloat-free `candidate_unicode61.db` alongside refreshed `candidate_trigram.db` carrying `yaw_stage1`) so all schemas hold bit-identical chunk spans:

| Corpus | Unique `.md` Docs | Chunks | `incumbent_unicode61.db` (`MiB` / `MB`) | `candidate_trigram.db` (`EVAL-0` -> `EVAL-3b` `MiB`) | **`candidate_unicode61.db`** (`MiB` / `MB`) |
| :--- | ---: | ---: | :---: | :---: | :---: |
| `2wikimultihopqa` | `6,119` | `6,980` | `32.98` / `34.59` | `55.97` -> `56.65` | **`31.01` / `32.52`** |
| `musique` | `11,656` | `12,512` | `62.95` / `66.00` | `108.78` -> `110.08` | **`58.12` / `60.95`** |
| `hotpotqa` | `9,811` | `10,785` | `55.15` / `57.83` | `97.44` -> `98.59` | **`50.02` / `52.45`** |
| `popqa` | `8,676` | `9,880` | `49.65` / `52.06` | `83.18` -> `84.47` | **`44.80` / `46.98`** |
| `enterprise_rag_bench` | `1,308` | `12,802` | `73.73` / `77.32` | `142.05` -> `142.82` | **`62.48` / `65.51`** |
| **Total File Size (`os.stat`)** | **`37,570`** | **`52,959`** | **`274.46` / `287.80`** | **`487.42` -> `492.61` (`516.54 MB`)** | **`246.43` / `258.40`** |

#### Exact SQLite `dbstat` Component Allocation Across All 5 Full-Corpus DBs

| SQLite Component (`dbstat` Sum) | `incumbent_unicode61.db` (`MiB`) | `candidate_trigram.db` (`EVAL-3b` `MiB`) | **`candidate_unicode61.db` (`MiB`)** | Schema / Tokenizer Attribution |
| :--- | :---: | :---: | :---: | :--- |
| `chunks_vector_chunks00` | `162.23` | `162.23` | `162.23` | Bit-identical shared `768d` embeddings (`vec0`) |
| `chunks_auxiliary` | `62.52` | `6.65` | **`5.93`** | Incumbent carries `spaCy` `tokens_json`/`entities_json` (`-56.59 MiB` in `candidate_unicode61.db`) |
| `chunk_fts_data` (`chunk_fts_*` total) | `45.76` (`47.12`) | `256.59` (`258.18`) | **`43.56` (`44.73`)** | Pure FTS5 payload: **`5.89x`** smaller for `candidate_unicode61.db` vs `trigram` (`5.77x` total `chunk_fts_*`) |
| `yaw_stage1` | `0.00` | `31.94` | `31.94` | Persisted RaBitQ + compressed `entity_graph_blob` (`31.92 MiB` pre-`3b`; absent in incumbent DB) |
| `chunk_vocab` + remaining metadata | `1.63` | `1.71` | **`1.60`** | Virtual `fts5vocab` reads `chunk_fts` dynamically (`0.00 MiB`); `rowids`/`chunks`/`info` |
| **Total `dbstat` Allocated Pages (`File Size`)** | **`273.57` (`274.46`)** | **`460.71` (`492.61`)** | **`246.43` (`246.43`)** | `candidate_unicode61.db` beats even `incumbent_unicode61.db` by **`-28.03 MiB`** (`-10.2%`) and `candidate_trigram.db` by **`2.00x`** |

### 3.3 Split Manifests, Reject Probes & Statistical Protocol

1. **Disjoint Immutable Split Manifests (`0` question-ID overlap enforced automatically by `load_and_verify_splits`):**
   - `outputs/mdl-dag-eval/manifests/splits_v1.json`:
     - **Tier A (`tier_a_anchor`, `tier_a_sha256: e8c848e2ba0d4158c416ec0fb013fa01b9bee320383564bc79b1e594af2c361d`):** Offsets `[0, 10)` across 4 public datasets (`total=40`, `scorable=40`), anchoring `CS-5` continuity.
     - **Tier B (`tier_b_dev`, `tier_b_sha256: 14adc8c1293fc71805a38b815a203cad302270120b4ff6fa16f0b439b0e23814`):** Offsets `[30, 80)` across 4 public datasets (`200`; `musique` `2h: 21, 3h: 17, 4h: 12`) + `erb train[0:50]` (`43` scorable + `1` `"High Level"` empty-gold + `6` `"Info Not Found"`) -> `250` total (`243` retrieval-scorable) + `50` `musique[30:80)` paired hop-1 bridge-masked probes (`150` reject labels total). Used for Waves 1–3 (`EVAL-3c`–`EVAL-3e`).
     - **Tier C (`tier_c_sealed_holdout`, `tier_c_sha256: 4289edbc286d190e1125852b4888cf86fe2da4389684441d4ee714b2f80e0fb5`):** Offsets `[100, 200)` (`400`; `musique` `2h: 55, 3h: 28, 4h: 17`) + `erb val[150:250]` (`90` scorable + `4` `"High Level"` empty-gold + `6` `"Info Not Found"`) -> `500` total (`490` retrieval-scorable) + `50` `musique[100:150)` bridge-masked probes (`200` reject labels total). Consumed cleanly once for `M_WIN_0P`; subsequently used as a post-freeze comparison benchmark for `M_PURE_0P`, `M_PURE_U61_1B`, and `M_FINAL_1B` (NOT an untouched holdout after `M_WIN_0P`).
   - `outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json`:
     - **Tier D (`tier_d_confirmatory`, `tier_d_sha256: 0ae4fa1a343a9c6c9964b51f4808b30ca462cadd79b6280d73fd6b94fab3e4ff`):** Disjoint offsets `[200, 300)` (`400`; `musique` `2h: 58, 3h: 29, 4h: 13`) + `erb[50:150)` (`92` scorable + `5` `"High Level"` empty-gold + `3` `"Info Not Found"`) -> `500` total (`492` retrieval-scorable) + `50/50` unique `musique[200:250)` hop-1 bridge-masked probes (`200` reject labels total) + `n = 100` (`[200, 225)` × 4 public datasets) paired `dspy.ReAct` QA. Executed cleanly once after `M_PURE_0P` freeze; subsequently reused as a non-confirmatory comparison benchmark for `M_PURE_U61_1B` (`EVAL-3b`) and `M_FINAL_1B` (`EVAL-3f`).
   - `outputs/mdl-dag-eval/manifests/splits_tier_e_confirmatory.json` (`schema_version: 3`, `pinned_commit: 3cd3952`):
     - **Tier E (`tier_e_confirmatory`, `tier_sha256: fe1449a26a5d99ef289c99bca15167af4e34743c10c61072796dd4672fdefeee`):** Previously untouched disjoint offsets `[300, 400)` across `{2wikimultihopqa, musique, hotpotqa, popqa}` (`100` each across `36,262` full-corpus docs; `musique` `2-hop: 48, 3-hop: 36, 4-hop: 16`; `enterprise_rag_bench` omitted as its `150` queries were exhausted across Tiers A–D) -> `400` scorable queries + `200` paired `musique[300:400)` hop-1 bridge-masked reject probes (`100` intact vs `100` masked) + **`n = 200` (`[300, 350)` × 4)** paired `dspy.ReAct` QA. Executed **once** after freezing `M_FINAL_1B` (`EVAL-3f`).
2. **Reject Calibration (`P4`):** ERB labels unanswerability strictly on `question_type == "Info Not Found"` (`3` Tier D / `6` Tier C / `6` Tier B vs remaining ERB queries). Each `MuSiQue` probe masks **all** chunks sharing the hop-1 `masked_doc_path` across dense vectors, postings, DF tables, FTS5 `MATCH`, and `CitationDiffusionOps` (`masked_leaks = 0` verified across all runs). On `Tier E` (`200` probes), reject evaluation tests the `100` intact vs `100` hop-1 bridge-masked `MuSiQue` pairs cleanly without ERB contamination.
3. **Statistical Inference:** `10,000`-resample dataset-stratified bootstrap 95% CIs (`seed = 0x594157`), two-sided paired sample permutation tests ($n_{\text{perm}} = 10{,}000$), two-sided Wilcoxon signed-rank tests, Cohen's $d_z$, and exact Holm–Bonferroni FWER step-down. Null controls `B5S_RAND_SEEDS` / `B5A_RAND_WALK` average `20` seeded random draws per query into `1` paired observation/query (`n = 243 / 490 / 492 / 400`).

---

## 4. Main Empirical Results Across Splits

### 4.1 Tier A Continuity Anchor (`[0, 10)`, `n = 40`)

`outputs/mdl-dag-eval/reports/tier_a_anchor.json` & `tier_a_anchor_mfull_opt3.json` confirmed full `CS-5` continuity on `B3_CS6` (`4-DS R@1/2/5/10/20 = 0.3958 / 0.5854 / 0.7542 / 0.8521 / 0.9417`, `4-DS AH@5 = 0.5250`, `3-DS MH AH@5 = 0.6333`, `MRR = 0.9240`, `NDCG@5 = 0.7446`; per-dataset `R5/AH5`: `2Wiki 0.9000/0.7000`, `MuSiQue 0.6667/0.5000`, `HotpotQA 0.8500/0.7000`, `PopQA 0.6000/0.2000`) ahead of `B0_DPR` (`R@5 = 0.7312`, `MH AH@5 = 0.5333`) and behind tuned early `M_FULL` (`R@5 = 0.7792 -> 0.8292`, `MH AH@5 = 0.6333 -> 0.7667`).

### 4.2 Tier B Iterative Discovery Trajectory (`N = 243` Scorable + `150` Reject Labels)

Selected milestones from `eval2a_controls_mfull.json`, `eval2b_frontier.json`, `eval2c_mwin_loo.json`, `eval2d_mwin0p_loo.json`, and `eval2e_mpure0p_loo.json`:

| Arm (Tier B, `N = 243`) | Pool `R@5` | `4-DS H4` | `MH AH@5` | `2Wiki` | `MuSiQue` | `HotpotQA` | `PopQA` | `ERB` | `Rej` (`top1`) | `RAM` (`MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_DPR` | `0.6565` | `0.6221` | `0.4200` | `0.6350/0.3400` | `0.5133/0.1800` | `0.8700/0.7400` | `0.4700/0.0000` | `0.8164/0.7209` | `0.5000` (`0.6914`) | `7.65` | `127.8 / 258.5` |
| `B1_FTS_UNI` | `0.5813` | `0.5212` | `0.2267` | `0.6450/0.2800` | `0.3900/0.0200` | `0.6700/0.3800` | `0.3800/0.0000` | `0.8603/0.7907` | `0.5000` | `110.04` | `26.5 / 47.8` |
| `B2_FTS_TRI` | `0.5606` | `0.5075` | `0.2467` | `0.6000/0.2800` | `0.2800/0.0200` | `0.7100/0.4400` | `0.4400/0.0000` | `0.8074/0.7209` | `0.5000` | `7.65` | `77.5 / 260.2` |
| `B3_CS6` (incumbent) | `0.6377` | `0.5929` | `0.3400` | `0.6500/0.3200` | `0.4517/0.0400` | `0.8000/0.6600` | `0.4700/0.0000` | `0.8461/0.7442` | `0.5000` | `110.06` | `84.1 / 101.1` |
| `B4_HOP0_MDL` | `0.6772` | `0.6458` | `0.4067` | `0.6750/0.3400` | `0.5483/0.1400` | `0.8700/0.7400` | `0.4900/0.0000` | `0.8229/0.7442` | `0.6742` | `7.65` | `301.6 / 520.1` |
| `B5S_RAND_SEEDS` | `0.3920` | `0.3356` | `0.0153` | `0.3200/0.0150` | `0.2981/0.0050` | `0.4255/0.0260` | `0.2990/0.0060` | `0.6539/0.5512` | `0.5000` | `7.65` | `806.8 / 1438.6` |
| `B5A_RAND_WALK` | `0.3906` | `0.3338` | `0.0137` | `0.3193/0.0140` | `0.2978/0.0070` | `0.4215/0.0200` | `0.2965/0.0020` | `0.6549/0.5570` | `0.5000` | `7.65` | `804.4 / 1446.7` |
| `M_FULL` | `0.6381` | `0.5942` | `0.3867` | `0.5300/0.2200` | `0.5067/0.1800` | `0.8700/0.7600` | `0.4700/0.0000` | `0.8422/0.7442` | `0.6227` | `7.66` | `730.4 / 1225.9` |
| `M_WIN` (`H5A`) | `0.6855` | `0.6554` | `0.4467` | `0.6850/0.3600` | `0.5467/0.1800` | `0.9000/0.8000` | `0.4900/0.0000` | `0.8255/0.7442` | `0.7255` (`0.6914`) | `11.22` | `68.2 / 132.1` |
| `M_WIN_0P` *(superseded)* | `0.8381` | `0.8317` | `0.7267` | `0.9600/0.8800` | `0.7067/0.3800` | `0.9600/0.9200` | `0.7000/0.4000` | `0.8679/0.7907` | `0.7154` (`0.6914`) | `26.87` | `91.5 / 140.1` |
| **`M_PURE_0P`** | **`0.7998`** | **`0.7800`** | **`0.7000`** | **`0.9200/0.8200`** | **`0.6700/0.3600`** | **`0.9600/0.9200`** | **`0.5700/0.1400`** | **`0.8917/0.7907`** | **`0.7509`** (`0.6869`) | **`26.02`** | **`37.6 / 89.1`** |
| `AP3_UNICODE61_LEX` *(LOO)* | `0.8025` | `0.7842` | `0.7000` | `0.9300/0.8400` | `0.6667/0.3400` | `0.9600/0.9200` | `0.5800/0.1600` | `0.8878/0.7907` | `0.7435` (`0.6869`) | `26.05` | `32.5 / 65.1` |

#### Tier B `H6`–`H9` De-Parameterization & Frontier Stepping Stones (`eval2d_h6_deparam.json`, `eval2d_h78_frontier.json`, `eval2d_h9_frontier.json`)

| Frontier Arm (Tier B, `N = 243`) | Pool `R@5` | `4-DS H4` | `MH AH@5` | `2Wiki` | `MuSiQue` | `HotpotQA` | `PopQA` | `ERB` | `Rej AUROC` | `RAM` (`MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H6A_PURE_IDF` (`0.25` tiebreak removed) | `0.6855` | `0.6554` | `0.4467` | `0.6850/0.3600` | `0.5467/0.1800` | `0.9000/0.8000` | `0.4900/0.0000` | `0.8255/0.7442` | `0.7266` | `11.22` | `67.3 / 117.7` |
| `H6C_NO_CLONE` (clone threshold removed) | `0.6855` | `0.6554` | `0.4467` | `0.6850/0.3600` | `0.5467/0.1800` | `0.9000/0.8000` | `0.4900/0.0000` | `0.8255/0.7442` | `0.7257` | `11.22` | `71.1 / 129.0` |
| `H6D_NO_QUOTAS` (fixed quotas removed) | `0.6855` | `0.6554` | `0.4467` | `0.6850/0.3600` | `0.5467/0.1800` | `0.9000/0.8000` | `0.4900/0.0000` | `0.8255/0.7442` | `0.7255` | `11.22` | `114.8 / 176.9` |
| `H6_ALL` (`H6a`–`H6d` combined) | `0.6862` | `0.6554` | `0.4467` | `0.6850/0.3600` | `0.5467/0.1800` | `0.9000/0.8000` | `0.4900/0.0000` | `0.8293/0.7442` | `0.7272` | `11.22` | `122.0 / 179.0` |
| `H8A_TITLE_REF` (directed title citations) | `0.7064` | `0.6808` | `0.4867` | `0.7150/0.4200` | `0.5983/0.2200` | `0.9100/0.8200` | `0.5000/0.0000` | `0.8255/0.7442` | `0.7354` | `15.28` | `133.6 / 180.8` |
| `H8B_KRAFT_GS` (Kraft prior + GS) | `0.6828` | `0.6512` | `0.4333` | `0.6950/0.3800` | `0.5400/0.1600` | `0.8800/0.7600` | `0.4900/0.0000` | `0.8293/0.7442` | `0.7204` | `11.22` | `115.5 / 167.4` |
| `H7_SCALE_ONLY` (scale-law $b(N)$) | `0.7033` | `0.6771` | `0.4733` | `0.7150/0.4200` | `0.5933/0.2000` | `0.9000/0.8000` | `0.5000/0.0000` | `0.8255/0.7442` | `0.7405` | `14.64` | `126.4 / 183.7` |
| `H78_ZERO_PARAM` (`H6`–`H8` unified) | `0.7040` | `0.6779` | `0.4800` | `0.7250/0.4400` | `0.5867/0.2000` | `0.9000/0.8000` | `0.5000/0.0000` | `0.8255/0.7442` | `0.7409` | `14.64` | `146.1 / 220.3` |
| `H9A_TEMPERED_RQ4` (tempered + sub-RQ4) | `0.7862` | `0.7796` | `0.6000` | `0.8850/0.6800` | `0.6033/0.2400` | `0.9400/0.8800` | `0.6900/0.3800` | `0.8169/0.7209` | `0.7321` | `15.28` | `81.8 / 136.3` |
| `H9B_PATH_FRONTIER` | `0.7885` | `0.7829` | `0.6000` | `0.8750/0.6800` | `0.6267/0.2600` | `0.9300/0.8600` | `0.7000/0.4000` | `0.8143/0.7209` | `0.7361` | `14.64` | `79.9 / 130.6` |
| `H9C_EXACT_KRAFT` | `0.7695` | `0.7600` | `0.5933` | `0.8850/0.6800` | `0.6050/0.2800` | `0.9100/0.8200` | `0.6400/0.2800` | `0.8136/0.7209` | `0.7515` | `14.64` | `80.4 / 131.9` |
| `H9D_NO_DYN_H2` (`h <= 3` static) | `0.7960` | `0.7929` | `0.6267` | `0.8950/0.7200` | `0.6567/0.3200` | `0.9200/0.8400` | `0.7000/0.4000` | `0.8105/0.7209` | `0.7458` | `14.64` | `64.3 / 112.5` |

### 4.3 Untouched Disjoint Confirmatory Holdout — Tier D (`N = 492`, `200` Reject Labels, `n = 100` QA)

#### Retrieval & Reject Calibration (`eval3_tier_d_retrieval.json`)

`M_PURE_0P` was executed **once** on untouched Tier D (`outputs/mdl-dag-eval/reports/eval3_tier_d_retrieval.json`). Contextual external published targets (`70B` OpenIE + `7B NV-Embed-v2` / `671B DeepSeek-V3`) are listed below for reference only and are **not** run on identical embeddings or subsets.

| System (`Tier D`, `N = 492`) | Pool `R@1 / R@2 / R@5 / R@10 / R@20` | `4-DS H4 R@5` | `MH AH@5` | `Pool AH@5` | `MRR` | `NDCG@5` | `2Wiki R@5 / AH@5` | `MuSiQue R@5 / AH@5` | `HotpotQA R@5 / AH@5` | `PopQA R@5 / AH@5` | `ERB R@5 / AH@5` | `Rej` (`top1`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_DPR` | `0.3960 / 0.5661 / 0.6815 / 0.7381 / 0.7746` | `0.6498` | `0.4833` | `0.4329` | `0.8325` | `0.6734` | `0.7000/0.4400` | `0.4942/0.2100` | `0.9000/0.8000` | `0.5050/0.0100` | `0.8194/0.7283` | `0.5000` (`0.7402`) |
| `B1_FTS_UNI` | `0.3712 / 0.4855 / 0.6107 / 0.6972 / 0.7457` | `0.5585` | `0.3367` | `0.3496` | `0.7775` | `0.6053` | `0.6825/0.3900` | `0.4117/0.0700` | `0.7600/0.5500` | `0.3800/0.0000` | `0.8373/0.7717` | `0.5000` (`0.5000`) |
| `B2_FTS_TRI` | `0.3517 / 0.4793 / 0.6061 / 0.6567 / 0.7042` | `0.5604` | `0.3367` | `0.3435` | `0.7517` | `0.5893` | `0.6575/0.3900` | `0.4292/0.1000` | `0.7400/0.5200` | `0.4150/0.0000` | `0.8045/0.7391` | `0.5000` (`0.5000`) |
| `B3_CS6` (incumbent) | `0.3622 / 0.5267 / 0.6627 / 0.7444 / 0.7908` | `0.6248` | `0.4233` | `0.4004` | `0.7891` | `0.6378` | `0.7650/0.5200` | `0.4542/0.1000` | `0.8150/0.6500` | `0.4650/0.0000` | `0.8277/0.7609` | `0.5000` (`0.5000`) |
| `B5S_RAND_SEEDS` | `0.3960 / 0.4065 / 0.4114 / 0.4329 / 0.4535` | `0.3556` | `0.0172` | `0.1194` | `0.7519` | `0.4836` | `0.3936/0.0185` | `0.2778/0.0055` | `0.4703/0.0275` | `0.2807/0.0020` | `0.6540/0.5804` | `0.5000` (`0.5000`) |
| `B5A_RAND_WALK` | `0.3960 / 0.4076 / 0.4150 / 0.4322 / 0.4640` | `0.3611` | `0.0222` | `0.1218` | `0.7534` | `0.4864` | `0.4084/0.0375` | `0.2780/0.0050` | `0.4688/0.0240` | `0.2892/0.0035` | `0.6492/0.5755` | `0.5000` (`0.5000`) |
| **`M_PURE_0P`** *(Primary)* | **`0.4506 / 0.6795 / 0.8010 / 0.9025 / 0.9361`** | **`0.7746`** | **`0.7033`** | **`0.6199`** | **`0.9068`** | **`0.7936`** | **`0.9475/0.8500`** | **`0.6158/0.3400`** | **`0.9600/0.9200`** | **`0.5750/0.1500`** | **`0.9158/0.8587`** | **`0.7976`** (`0.7387`) |
| `AP3_UNICODE61_LEX` *(LOO)* | `0.4533 / 0.6804 / 0.8098 / 0.9008 / 0.9371` | `0.7858` | `0.7133` | `0.6260` | `0.9096` | `0.8006` | `0.9600/0.8700` | `0.6333/0.3300` | `0.9700/0.9400` | `0.5800/0.1600` | `0.9140/0.8478` | `0.7887` (`0.7387`) |
| *HippoRAG 2 (Table 3, 70B+7B ref)* | — | *0.7830* | — | — | — | — | *0.904 / —* | *0.747 / —* | *0.963 / —* | *0.517 / —* | — | — |
| *Youtu-GraphRAG (`arXiv:2508.19855`, 671B DeepSeek-V3 @10 / @20 ref)* | — | — | — | — | — | — | *0.726 @10 / 0.776 @20 (Acc 0.823 / 0.855)* | *0.469 @10 / 0.475 @20 (Acc 0.521 / 0.536)* | *0.789 @10 / 0.812 @20 (Acc 0.834 / 0.865)* | — | — | — |

Additional `M_PURE_0P` Tier D metrics: `Pool All-Hops@{2, 5, 10} = 0.4431 / 0.6199 / 0.8069`, Reject subset AUROC `ERB (n=100, 3 INF) = 0.7766` (vs `top1 = 0.7010`) and `MuSiQue (n=100) = 0.6872` (vs `top1 = 0.6166`), `bnb_exact_fetches = 0.0`, `masked_leaks = 0`. Note that `AP3_UNICODE61_LEX` achieved higher Tier D retrieval (`Pool R@5 = 0.8098`, `4-DS H4 = 0.7858` exceeding `0.7830`, `MH AH@5 = 0.7133`) at lower `ERB` Reject AUROC (`0.7045` vs `0.7766`).

#### Tier D `MuSiQue` Hop Stratification (`n = 100`)

| System | `2-hop` (`n=58`) `R@5 / AH@5` | `3-hop` (`n=29`) `R@5 / AH@5` | `4-hop` (`n=13`) `R@5 / AH@5` | `MuSiQue` Total `R@5 / AH@5` |
| :--- | :---: | :---: | :---: | :---: |
| `B3_CS6` (incumbent) | `0.4828 / 0.1552` | `0.4368 / 0.0345` | `0.3654 / 0.0000` | `0.4542 / 0.1000` |
| **`M_PURE_0P`** | **`0.7155 / 0.5345`** | **`0.5287 / 0.1034`** | `0.3654 / 0.0000` | **`0.6158 / 0.3400`** |
| `AP3_UNICODE61_LEX` *(LOO)* | `0.7155 / 0.5172` | `0.5287 / 0.0690` | **`0.5000 / 0.0769`** | **`0.6333 / 0.3300`** |

#### Tier D Paired Statistical Inference (`P1` & `P2` Contrasts, `10k` Stratified Bootstrap)

| Gate / Contrast (`M_PURE_0P` minus Baseline) | `n` | Mean A | Mean B | $\Delta$ Mean | `[95% CI]` | $p_{\text{perm}}$ | $p_{\text{Wilcoxon}}$ | $p_{\text{Holm}}$ | Cohen's $d_z$ |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `P1`: vs `B5S_RAND_SEEDS` `Recall@5` | `492` | `0.8010` | `0.4114` | `+0.3896` | `[+0.3665, +0.4129]` | `0.0002` | `1.77e-63` | `0.0008` | `+1.3530` |
| `P1`: vs `B5S_RAND_SEEDS` `All-Hops@5` | `492` | `0.6199` | `0.1194` | `+0.5005` | `[+0.4670, +0.5346]` | `0.0002` | `2.81e-47` | `0.0008` | `+1.0210` |
| `P1`: vs `B5A_RAND_WALK` `Recall@5` | `492` | `0.8010` | `0.4150` | `+0.3860` | `[+0.3623, +0.4093]` | `0.0002` | `6.89e-63` | `0.0008` | `+1.3342` |
| `P1`: vs `B5A_RAND_WALK` `All-Hops@5` | `492` | `0.6199` | `0.1218` | `+0.4981` | `[+0.4647, +0.5324]` | `0.0002` | `8.98e-48` | `0.0008` | `+1.0177` |
| `P2` Primary: vs `B3_CS6` `MH All-Hops@5` | `300` | `0.7033` | `0.4233` | `+0.2800` | `[+0.2267, +0.3333]` | `0.0002` | `1.99e-18` | `0.0004` | `+0.5851` |
| `P2` Primary: vs `B3_CS6` `macro Recall@5` | `492` | `0.8010` | `0.6627` | `+0.1382` | `[+0.1156, +0.1615]` | `0.0002` | `1.05e-24` | `0.0004` | `+0.5286` |
| `P2` Sub-DS: `2wikimultihopqa Recall@5` | `100` | `0.9475` | `0.7650` | `+0.1825` | `[+0.1350, +0.2300]` | `0.0002` | `3.76e-09` | — | `+0.7585` |
| `P2` Sub-DS: `enterprise_rag_bench Recall@5` | `92` | `0.9158` | `0.8277` | `+0.0881` | `[+0.0406, +0.1415]` | `0.0008` | `1.67e-03` | — | `+0.3512` |
| `P2` Sub-DS: `hotpotqa Recall@5` | `100` | `0.9600` | `0.8150` | `+0.1450` | `[+0.0950, +0.2000]` | `0.0002` | `1.86e-06` | — | `+0.5396` |
| `P2` Sub-DS: `musique Recall@5` | `100` | `0.6158` | `0.4542` | `+0.1617` | `[+0.1000, +0.2242]` | `0.0002` | `1.69e-06` | — | `+0.5056` |
| `P2` Sub-DS: `popqa Recall@5` | `100` | `0.5750` | `0.4650` | `+0.1100` | `[+0.0700, +0.1500]` | `0.0002` | `2.73e-06` | — | `+0.5284` |
| `P2` Horizon: `Recall@1` | `492` | `0.4506` | `0.3622` | `+0.0884` | `[+0.0658, +0.1111]` | `0.0002` | `4.85e-12` | — | `+0.3364` |
| `P2` Horizon: `Recall@2` | `492` | `0.6795` | `0.5267` | `+0.1528` | `[+0.1275, +0.1780]` | `0.0002` | `2.10e-24` | — | `+0.5303` |
| `P2` Horizon: `Recall@10` | `492` | `0.9025` | `0.7444` | `+0.1582` | `[+0.1379, +0.1785]` | `0.0002` | `1.31e-30` | — | `+0.5990` |
| `P2` Horizon: `Recall@20` | `492` | `0.9361` | `0.7908` | `+0.1453` | `[+0.1275, +0.1633]` | `0.0002` | `8.04e-30` | — | `+0.5818` |

Every single dataset and every single recall horizon $R@1\dots20$ exhibits strictly positive 95% CI lower bounds against `B3_CS6`.

#### Paired `P3` Tier-1 `dspy.ReAct` QA (`eval3_tier_d_qa.json`, `n = 100`, `0` Model Errors)

Evaluated using unmodified [qa.py](../../yaw/qa.py) `dspy.ReAct` on `gemma-4-e4b` (`temperature = 1.0`, uncompiled zero-shot, offsets `[200, 225)` across the 4 public QA sets; `200` total paired episodes completed fail-fast with `0` `MODEL_ERROR`s):

| Dataset / Metric (`n = 25` per DS) | `B3_CS6` `EM / F1` | `M_PURE_0P` `EM / F1` | Paired $\Delta$ (`EM` / `F1`) |
| :--- | :---: | :---: | :---: |
| `2wikimultihopqa` (`n=25`) | `0.4800 / 0.6843` | `0.5600 / 0.7133` | `+0.0800 / +0.0290` |
| `musique` (`n=25`) | `0.2400 / 0.4194` | `0.3200 / 0.4267` | `+0.0800 / +0.0073` |
| `hotpotqa` (`n=25`) | `0.5200 / 0.6269` | `0.6000 / 0.7141` | `+0.0800 / +0.0872` |
| `popqa` (`n=25`) | `0.0800 / 0.3635` | `0.0400 / 0.3917` | `-0.0400 / +0.0282` |
| **Pooled (`n = 100`, `P3 = PASS`)** | **`0.3300 / 0.5235`** | **`0.3800 / 0.5615`** | **`EM +0.0500 [-0.0200, +0.1200]`** ($p_{\text{perm}}=0.2614$, $p_{\text{W}}=0.1655$, $d_z=+0.1393$)<br>**`F1 +0.0379 [-0.0269, +0.1060]`** ($p_{\text{perm}}=0.2694$, $p_{\text{W}}=0.3497$, $d_z=+0.1106$) |

### 4.4 Superseded Sealed Tier C (`M_WIN_0P`) & Post-Freeze Tier C Comparison (`M_PURE_0P`)

Reports `eval3_tier_c_retrieval.json` (verbatim sealed record for `M_WIN_0P`) and `eval3_tier_c_mpure0p_comparison.json` (`N = 490` scorable + `200` reject labels; `M_PURE_0P` run as post-freeze simplification comparison against `M_WIN_0P`):

| System (`Tier C`, `N = 490`) | Pool `R@5` | `4-DS H4` | `MH AH@5` | `2Wiki` | `MuSiQue` (`2h/3h/4h` `AH@5`) | `HotpotQA` | `PopQA` | `ERB` | `Rej` (`top1`) | `RAM` (`MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_DPR` | `0.6749` | `0.6385` | `0.4500` | `0.6550/0.3700` | `0.5492/0.2600` | `0.8550/0.7200` | `0.4950/0.0000` | `0.8364/0.7333` | `0.5000` (`0.6804`) | `7.65` | `132.3 / 279.9` |
| `B1_FTS_UNI` | `0.6158` | `0.5502` | `0.3067` | `0.6450/0.3400` | `0.4508/0.1100` | `0.7100/0.4700` | `0.3950/0.0000` | `0.9072/0.8000` | `0.5000` | `110.04` | `26.4 / 51.2` |
| `B2_FTS_TRI` | `0.5790` | `0.5294` | `0.2533` | `0.6025/0.2900` | `0.4100/0.0700` | `0.6700/0.4000` | `0.4350/0.0000` | `0.7994/0.7111` | `0.5000` | `7.65` | `63.9 / 178.8` |
| `B3_CS6` | `0.6547` | `0.6098` | `0.4100` | `0.6850/0.4100` | `0.4742/0.1700` (`0.2545/0.1071/0.0000`) | `0.8000/0.6500` | `0.4800/0.0000` | `0.8542/0.7556` | `0.5000` | `110.04` | `81.0 / 100.0` |
| `B4_HOP0_MDL` | `0.7002` | `0.6642` | `0.4700` | `0.7125/0.4200` | `0.5692/0.2200` | `0.8850/0.7700` | `0.4900/0.0000` | `0.8605/0.7556` | `0.5944` | `7.65` | `303.1 / 489.4` |
| `B5S_RAND_SEEDS` | `0.3950` | `0.3319` | `0.0153` | `0.3823/0.0165` | `0.2624/0.0065` | `0.4350/0.0230` | `0.2480/0.0055` | `0.6753/0.5906` | `0.5000` | `7.65` | `723.4 / 1091.3` |
| `B5A_RAND_WALK` | `0.3976` | `0.3337` | `0.0183` | `0.3881/0.0310` | `0.2619/0.0085` | `0.4305/0.0155` | `0.2542/0.0070` | `0.6816/0.5950` | `0.5000` | `7.65` | `733.6 / 1063.7` |
| `M_WIN_0P` *(superseded)* | `0.8123` | `0.7948` | `0.6900` | `0.9225/0.8000` | `0.6617/0.3800` (`0.6364/0.1071/0.0000`) | `0.9450/0.8900` | `0.6500/0.3100` | `0.8901/0.8000` | **`0.6432`** (`0.6804`) **FAIL** | `26.55` | `90.3 / 135.9` |
| **`M_PURE_0P`** *(comp)* | **`0.7913`** | **`0.7688`** | **`0.6800`** | **`0.9250/0.8100`** | **`0.6450/0.3900`** (`0.6364/0.1429/0.0000`) | **`0.9200/0.8400`** | **`0.5850/0.1800`** | **`0.8916/0.8333`** | **`0.7731`** (`0.6807`) **PASS** | **`26.02`** | **`34.9 / 89.0`** |
| `AP3_UNICODE61_LEX` *(LOO)* | `0.8039` | `0.7765` | `0.6933` | `0.9325/0.8300` | `0.6383/0.3700` | `0.9400/0.8800` | `0.5950/0.2000` | `0.9257/0.8778` | `0.7641` (`0.6807`) | `26.05` | `32.7 / 63.7` |

Additional Tier C `M_PURE_0P` telemetry: `R@{1,2,10,20} = 0.4218 / 0.6425 / 0.8949 / 0.9319`, `AH@{2,5,10} = 0.3939 / 0.6061 / 0.8000`, `MRR = 0.8724`, `NDCG@5 = 0.7680`, `P1 = PASS` (`p_holm = 0.0008`), `P2 = PASS` (`MH AH@5 +0.2700 [+0.2167, +0.3233]`, `macro R@5 +0.1366 [+0.1144, +0.1596]`, `p_holm = 0.0004`), `Reject ERB = 0.8138` / `MuSiQue = 0.6792` vs `M_WIN_0P ERB = 0.8440` / `MuSiQue = 0.6536`. On Tier C, `M_PURE_0P` repaired `M_WIN_0P`'s `P4` failure (`+0.1299` `Reject AUROC`), matched `MH AH@5` within `1.0 pp` (`0.6800 vs 0.6900`), and cut median latency **`2.59x`** (`34.9 ms vs 90.3 ms`).

### 4.5 `EVAL-3b`: Fair `unicode61` vs `trigram` Shootout & 1-Bit SoA `M_PURE_U61_1B` (`325eea3`)

To resolve `EVAL-3` Path A rigorously, `EVAL-3b` corrected two pre-shootout tokenization asymmetries (unified Unicode word/sentence regexes + consolidated `_run_decomp_for_word` prefix-stemming/Koehn–Knight compound splitting + equal CJK non-overlapping bigram PMI across both tokenizers), added Stage-A/Stage-B decoupled CocoIndex `1.2.0` indexing producing bloat-free **`candidate_unicode61.db`** (`246.43 MiB`, `5.89x` smaller `chunk_fts_data` than `trigram` — see §3.2), and ran paired Tiers D & C comparisons (`eval3b_tier_d_shootout.json`, `eval3b_tier_c_shootout.json`, `eval3b_tier_d_1bit.json`, `eval3b_tier_c_1bit.json`, `eval3b_tier_d_1bit_qa.json`):

| Tier / Arm | Pool `R@5` | 4-DS `H4` | `MH AH@5` | `2Wiki` `R5/AH5` | `MuSiQue` `R5/AH5` (`2h / 3h / 4h`) | `HotpotQA` `R5/AH5` | `PopQA` `R5/AH5` | `ERB` `R5/AH5` | `R@10 / R@20` | `Rej AUROC` (`top1`) | `TM` (`MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Tier D` `B3_CS6` | `0.6627` | `0.6248` | `0.4233` | `0.7650/0.5200` | `0.4542/0.1000` (`0.4828/0.1552` / `0.4368/0.0345` / `0.3654/0.0000`) | `0.8150/0.6500` | `0.4650/0.0000` | `0.8277/0.7609` | `0.7444/0.7908` | `0.5000` (`0.5000`) | `211.08` *(RAM `110.05`)* | `79.7 / 100.6` |
| `Tier D` `M_PURE_0P` | `0.8010` | `0.7746` | `0.7033` | `0.9475/0.8500` | `0.6158/0.3400` (`0.7155/0.5345` / `0.5287/0.1034` / `0.3654/0.0000`) | `0.9600/0.9200` | `0.5750/0.1500` | `0.9158/0.8587` | `0.9025/0.9361` | `0.7976` (`0.7387`) | `75.92` *(RAM `25.46`)* | `35.8 / 89.3` |
| `Tier D` `M_PURE_U61` | `0.8118` | `0.7883` | `0.7167` | `0.9600/0.8700` | `0.6433/0.3400` (`0.7155/0.5172` / `0.5632/0.1034` / `0.5000/0.0769`) | `0.9700/0.9400` | `0.5800/0.1600` | `0.9140/0.8478` | `0.9031/0.9382` | `0.7941` (`0.7387`) | `75.87` *(RAM `25.41`)* | `25.3 / 48.8` |
| **`Tier D` `M_PURE_U61_1B`** | **`0.8128`** | **`0.7890`** | **`0.7133`** | **`0.9650/0.8800`** | **`0.6408/0.3200` (`0.7155/0.5172` / `0.5632/0.0345` / `0.4808/0.0769`)** | **`0.9700/0.9400`** | **`0.5800/0.1600`** | **`0.9165/0.8478`** | **`0.9095/0.9383`** | **`0.7855` (`0.7291`)** | **`73.27` *(RAM `22.55`)*** | **`23.1 / 44.3`** |
| `Tier C` `B3_CS6` | `0.6547` | `0.6098` | `0.4100` | `0.6850/0.4100` | `0.4742/0.1700` (`0.5636/0.2545` / `0.4167/0.1071` / `0.2794/0.0000`) | `0.8000/0.6500` | `0.4800/0.0000` | `0.8542/0.7556` | `0.7404/0.7887` | `0.5000` (`0.5000`) | `211.08` *(RAM `110.05`)* | `81.2 / 108.4` |
| `Tier C` `M_PURE_0P` | `0.7913` | `0.7688` | `0.6800` | `0.9250/0.8100` | `0.6450/0.3900` (`0.7909/0.6364` / `0.5357/0.1429` / `0.3529/0.0000`) | `0.9200/0.8400` | `0.5850/0.1800` | `0.8916/0.8333` | `0.8949/0.9319` | `0.7731` (`0.6807`) | `76.50` *(RAM `26.04`)* | `37.3 / 97.7` |
| `Tier C` `M_PURE_U61` | `0.8056` | `0.7785` | `0.6967` | `0.9325/0.8300` | `0.6467/0.3800` (`0.7818/0.6182` / `0.5595/0.1429` / `0.3529/0.0000`) | `0.9400/0.8800` | `0.5950/0.2000` | `0.9257/0.8778` | `0.9037/0.9364` | `0.7767` (`0.6807`) | `75.87` *(RAM `25.40`)* | `25.8 / 49.4` |
| **`Tier C` `M_PURE_U61_1B`** | **`0.8028`** | **`0.7775`** | **`0.6867`** | **`0.9350/0.8300`** | **`0.6300/0.3500` (`0.7727/0.6182` / `0.5357/0.0357` / `0.3235/0.0000`)** | **`0.9400/0.8800`** | **`0.6050/0.2200`** | **`0.9152/0.8556`** | **`0.8989/0.9385`** | **`0.7703` (`0.6783`)** | **`73.27` *(RAM `22.55`)*** | **`27.6 / 50.5`** |

- **Key `EVAL-3b` Outcome:** Under 100% fair tokenization, `M_PURE_U61_1B` (`1-bit` SoA RaBitQ) matched `M_PURE_U61` (`Pool R@5 = 0.8128 vs 0.8118` on Tier D; `R@{1,2} = 0.4507 / 0.6858`, `AH@{2,5,10} = 0.4451 / 0.6260 / 0.8089`, `MRR = 0.9083`, `NDCG@5 = 0.8034`) while reducing `VectorSearch` RAM **`4.88x`** (`22.55` vs `110.05 MiB`) and median latency to `23.1 ms` (`p95 = 44.3 ms`, `p99 = 53.6 ms`, `0` masked leaks, `0` concurrency errors). It cleared `P1` (`p_holm = 0.0008`), `P2` (`MH AH@5 +0.2900 [+0.2367, +0.3467]`, `macro R@5 +0.1501 [+0.1276, +0.1727]`, `p_holm = 0.0004` on Tier D), and `P4` (`0.7855` Tier D / `0.7703` Tier C), while failing `P5` (`6/8 PASS`, §5.3) and **`P3` (`eval3b_tier_d_1bit_qa.json`, `n = 100`, `0` model errors): `M_PURE_U61_1B` (`EM = 0.3900`, `F1 = 0.5786`; MH `0.5067/0.6523`, `2Wiki 0.5200/0.7243`, `MuSiQue 0.3200/0.4650`, `HotpotQA 0.6800/0.7674`, `PopQA 0.0400/0.3578`) vs `B3_CS6` (`EM = 0.4000`, `F1 = 0.5902`; MH `0.4667/0.6219`, `2Wiki 0.4800/0.6960`, `MuSiQue 0.3200/0.4823`, `HotpotQA 0.6000/0.6874`, `PopQA 0.2000/0.4950`), `dEM = -0.0100 [-0.0800, +0.0600]` (`p_perm = 1.0`), `dF1 = -0.0115 [-0.0785, +0.0575]` (`p_perm = 0.7447`)** — motivating the algebraic duality discovery waves (`EVAL-3c`–`EVAL-3e`).

### 4.6 `EVAL-3c / EVAL-3d / EVAL-3e`: Dualities, $2^4 = 16$-Arm Factorial Lattice & Frozen `M_FINAL_1B` (`d29531a`)

All Stage 1–3 discovery experiments executed strictly on `tier_b_dev` (`N = 243` scorable, `N = 250` latency, `150` reject probes, `top1_cosine AUROC = 0.6837`, `0` masked leaks):

#### Stage 1 & Stage 2: Parameter-Free Dualities (Waves 1 & 2, `eval3c_tier_b_wave1.json` & `eval3d_tier_b_wave2.json`)

| Wave / Arm (`tier_b_dev`, `N = 243`) | Pool `R@5` | $\Delta$ `R@5` `[95% CI]` (`W/L`, `p_perm`) | 4-DS `H4` | `MH AH@5` | `ERB R5` | `Rej AUROC` | `TM` (`MiB`) | `p50 / p95` (`ms`) | Disposition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Base` `M_PURE_U61_1B` *(W1 & W2 bit-exact)* | `0.7994` | *control* | `0.7854` | `0.7067` | `0.8645` | `0.7382` | `73.27` | `22.08 / 45.71` | Baseline |
| **W1: `E1A_DPP_1BIT`** *(1-bit Rank-1 Cholesky DPP)* | **`0.8083`** | **`+0.0089 [+0.0021, +0.0171]`** (`5W/0L`, `0.0602`) | **`0.7963`** | **`0.7200`** | `0.8645` | `0.7382` | `72.37` | `26.47 / 48.87` | **Advance** (`PopQA 0.6200`) |
| W1: `E1B_RESIDUAL_COV` *(residualized coverage)* | `0.7393` | `-0.0601 [-0.0931, -0.0273]` (`26W/53L`, `0.0008`) | `0.7583` | `0.5400` | `0.6510` | `0.7382` | `72.37` | `51.15 / 79.23` | **Falsified** (ERB/MH collapse) |
| W1: `E2A_DYN_RESID_UNIT` *(unit residual query)* | `0.8007` | `+0.0013 [-0.0192, +0.0215]` (`16W/16L`, `0.9257`) | `0.7900` | `0.6600` | `0.8503` | `0.7382` | `72.37` | `128.96 / 333.33` | **Falsified** (`-4.67 pp` MH, `7.3x p95`) |
| W1: `E2B_DYN_RESID_PROJ` *(centroid projection)* | `0.7983` | `-0.0011 [-0.0217, +0.0194]` (`16W/17L`, `0.9173`) | `0.7863` | `0.6467` | `0.8542` | `0.7382` | `72.70` | `126.95 / 328.86` | **Falsified** (centroid smear, `7.2x p95`) |
| W1: `E3A_NB_HASHIMOTO` *(non-backtracking CSR)* | `0.8011` | `+0.0017 [-0.0106, +0.0154]` (`5W/5L`, `0.8591`) | `0.7825` | `0.6933` | **`0.8878`** | `0.7382` | `74.64` | `24.63 / 51.42` | **Advance** (ERB `+2.33 pp`) |
| W1: `E3B_SEM_BONF` *(dense semantic edges)* | `0.7749` | `-0.0245 [-0.0381, -0.0123]` (`1W/16L`, `0.0006`) | `0.7571` | `0.6600` | `0.8578` | `0.7382` | `207.84` | `28.03 / 54.02` | **Falsified** (`-2.45 pp`, `2.8x TM`) |
| W1: `E3C_SEM_MKNN` *(mutual-kNN semantic)* | `0.7741` | `-0.0253 [-0.0434, -0.0077]` (`4W/22L`, `0.0056`) | `0.7517` | `0.6400` | `0.8784` | `0.7382` | `219.73` | `25.59 / 50.32` | **Falsified** (`-2.53 pp`, `3.0x TM`) |
| **W2: `F1_TANGENT_STEER`** *(tangent residual Beta)* | **`0.8128`** | **`+0.0134 [-0.0003, +0.0281]`** (`9W/3L`, `0.0868`) | **`0.7967`** | **`0.7267`** | **`0.8878`** | `0.7382` | `79.00` | `33.00 / 78.34` | **Advance** (`Hotpot 0.9800/0.9600`, `MUS 0.6717`) |
| W2: `F2_WORMHOLE_COND` *(rare-token wormholes)* | `0.8015` | `+0.0021 [+0.0000, +0.0062]` (`1W/0L`, `0.9815`) | `0.7879` | `0.7067` | `0.8645` | `0.7382` | `78.73` | `22.25 / 47.04` | Marginal (`PopQA +1.0 pp` only) |
| W2: `F3A_LOCAL_ENT_RERANK` *(local-density rerank)* | `0.8001` | `+0.0007 [-0.0089, +0.0103]` (`3W/3L`, `0.9847`) | `0.7862` | `0.7067` | `0.8645` | `0.7382` | `72.37` | `23.44 / 44.79` | Neutral |
| W2: `F3B_LOCAL_ENT_REJECT` *(local-KL reject)* | `0.7994` | `+0.0000` (`0W/0L`, `1.0000`, bit-exact) | `0.7854` | `0.7067` | `0.8645` | **`0.5161`** | `72.37` | `23.11 / 44.58` | **Falsified** (inverted: `11.85` vs `9.16` nats) |
| W2: `F4A_BRIDGE_PURE` *(pure backward bridge)* | `0.7792` | `-0.0202 [-0.0401, -0.0007]` (`7W/17L`, `0.0486`) | `0.7650` | `0.7000` | `0.8452` | `0.7382` | `75.32` | `37.97 / 63.78` | **Falsified** (`PopQA 0.5300`, `p = 0.0486`) |
| W2: `F4B_BRIDGE_NOISYOR` *(forward Noisy-OR bridge)* | `0.8011` | `+0.0017 [-0.0058, +0.0096]` (`3W/2L`, `0.8093`) | `0.7875` | `0.7067` | `0.8645` | `0.7382` | `74.99` | `38.06 / 64.99` | **Advance** (to lattice) |

- **Why `F3B_LOCAL_ENT_REJECT` Collapsed:** Unanswerable probes (e.g., bridge-ablated `MuSiQue` still matching the remaining surface entities tightly in one coherent distractor neighborhood) yielded *higher* mean local submatrix coherence KL (`11.85` vs `9.16` nats), inverting calibration (`0.7382 -> 0.5161`). Global corpus Bayes-factor evidence `raw_log_bf` remained strictly superior.

#### Stage 3: Complete $2^4 = 16$-Arm Factorial Lattice over `{F1, E1A, E3A, F4B}` (`eval3e_tier_b_lattice.json`)

Fixed pipeline execution order inside `_solve_pure_diffusion`: **`F1` (`edge_weight_mode`) $\to$ `E3A` (`non_backtracking`, wired to also step in `F4B`'s backward walk) $\to$ `F4B` (`bridge_mode`) $\to$ `E1A` (`rerank_mode`)**. All 16 configurations produced `0` masked leaks and invariant `Rej AUROC = 0.7382`:

| Factorial Arm (`F1, E1A, E3A, F4B`) | Pool `R@5` | $\Delta$ `R@5` `[95% CI]` (`W/L`, `p_perm`) | 4-DS `H4` | `MH AH@5` | `2Wiki` `R5/AH5` | `MuSiQue` `R5/AH5` | `HotpotQA` `R5/AH5` | `PopQA` `R5/AH5` | `ERB` `R5/AH5` | `TM` (`MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M_PURE_U61_1B` (`0,0,0,0`) | `0.7994` | *control* | `0.7854` | `0.7067` | `0.9450/0.8600` | `0.6367/0.3400` | `0.9600/0.9200` | `0.6000/0.2000` | `0.8645/0.7674` | `73.27` | `22.08 / 45.71` |
| `F1_TANGENT_STEER` (`1,0,0,0`) | `0.8128` | `+0.0134 [-0.0003, +0.0281]` (`9W/3L`, `0.0868`) | `0.7967` | **`0.7267`** | `0.9450/0.8600` | `0.6717/0.3600` | **`0.9800/0.9600`** | `0.5900/0.1800` | `0.8878/0.7907` | `79.00` | `33.00 / 78.34` |
| `E1A_DPP_1BIT` (`0,1,0,0`) | `0.8083` | `+0.0089 [+0.0021, +0.0171]` (`5W/0L`, `0.0602`) | `0.7963` | `0.7200` | `0.9450/0.8600` | `0.6600/0.3800` | `0.9600/0.9200` | `0.6200/0.2400` | `0.8645/0.7674` | `72.37` | `26.47 / 48.87` |
| `E3A_NB_HASHIMOTO` (`0,0,1,0`) | `0.8011` | `+0.0017 [-0.0106, +0.0154]` (`5W/5L`, `0.8591`) | `0.7825` | `0.6933` | `0.9250/0.8200` | `0.6350/0.3400` | `0.9600/0.9200` | `0.6100/0.2200` | `0.8878/0.7907` | `74.64` | `24.63 / 51.42` |
| `F4B_BRIDGE_NOISYOR` (`0,0,0,1`) | `0.8011` | `+0.0017 [-0.0058, +0.0096]` (`3W/2L`, `0.8093`) | `0.7875` | `0.7067` | `0.9450/0.8600` | `0.6450/0.3400` | `0.9600/0.9200` | `0.6000/0.2000` | `0.8645/0.7674` | `74.99` | `38.06 / 64.99` |
| **`C2_F1_E1A` (`1,1,0,0`, #1)** | **`0.8155`** | **`+0.0161 [+0.0003, +0.0329]`** (`11W/4L`, `0.0608`) | **`0.8000`** | `0.7200` | `0.9450/0.8600` | `0.6650/0.3400` | **`0.9800/0.9600`** | `0.6100/0.2200` | `0.8878/0.7907` | `79.00` | `37.51 / 83.47` |
| `C2_F1_E3A` (`1,0,1,0`) | `0.7980` | `-0.0014 [-0.0168, +0.0147]` (`7W/9L`, `0.9111`) | `0.7788` | `0.7000` | `0.9300/0.8200` | `0.6550/0.3600` | `0.9600/0.9200` | `0.5700/0.1400` | `0.8878/0.7907` | `78.73` | `36.81 / 79.61` |
| `C2_F1_F4B` (`1,0,0,1`) | `0.8059` | `+0.0065 [-0.0075, +0.0216]` (`8W/6L`, `0.4220`) | `0.7883` | `0.7133` | `0.9450/0.8600` | `0.6483/0.3400` | `0.9700/0.9400` | `0.5900/0.1800` | `0.8878/0.7907` | `78.73` | `49.70 / 98.87` |
| `C2_E1A_E3A` (`0,1,1,0`) | `0.8073` | `+0.0079 [-0.0055, +0.0223]` (`8W/4L`, `0.3148`) | `0.7892` | `0.7000` | `0.9250/0.8200` | `0.6517/0.3600` | `0.9600/0.9200` | `0.6200/0.2400` | `0.8917/0.7907` | `74.64` | `26.92 / 51.56` |
| `C2_E1A_F4B` (`0,1,0,1`) | `0.8094` | `+0.0099 [+0.0007, +0.0216]` (`6W/1L`, `0.0790`) | `0.7967` | **`0.7267`** | `0.9450/0.8600` | `0.6617/0.3800` | `0.9700/0.9400` | `0.6100/0.2200` | `0.8684/0.7674` | `75.32` | `40.74 / 69.63` |
| `C2_E3A_F4B` (`0,0,1,1`) | `0.8011` | `+0.0017 [-0.0123, +0.0168]` (`7W/7L`, `0.8727`) | `0.7825` | `0.6867` | `0.9250/0.8200` | `0.6450/0.3400` | `0.9500/0.9000` | `0.6100/0.2200` | `0.8878/0.7907` | `77.26` | `40.40 / 68.71` |
| `C3_F1_E1A_E3A` (`1,1,1,0`) | `0.8028` | `+0.0034 [-0.0130, +0.0213]` (`10W/9L`, `0.7289`) | `0.7837` | `0.7000` | `0.9300/0.8200` | `0.6550/0.3600` | `0.9600/0.9200` | `0.5900/0.1800` | **`0.8917/0.7907`** | `78.73` | `44.67 / 88.40` |
| `C3_F1_E1A_F4B` (`1,1,0,1`) | `0.8107` | `+0.0113 [-0.0058, +0.0298]` (`11W/7L`, `0.2350`) | `0.7933` | `0.7200` | `0.9450/0.8600` | `0.6583/0.3600` | `0.9700/0.9400` | `0.6000/0.2000` | `0.8917/0.7907` | `78.73` | `60.24 / 127.89` |
| `C3_F1_E3A_F4B` (`1,0,1,1`) | `0.8001` | `+0.0007 [-0.0144, +0.0165]` (`8W/9L`, `0.9725`) | `0.7812` | `0.7067` | `0.9300/0.8200` | `0.6550/0.3600` | `0.9700/0.9400` | `0.5700/0.1400` | `0.8878/0.7907` | `79.33` | `60.33 / 119.47` |
| `C3_E1A_E3A_F4B` (`0,1,1,1`) | `0.8118` | `+0.0123 [-0.0024, +0.0291]` (`10W/4L`, `0.1424`) | `0.7946` | `0.7133` | `0.9250/0.8200` | **`0.6733/0.4000`** | `0.9600/0.9200` | **`0.6200/0.2400`** | **`0.8917/0.7907`** | `77.26` | `44.09 / 70.41` |
| `C4_F1_E1A_E3A_F4B` (`1,1,1,1`) | `0.8035` | `+0.0041 [-0.0137, +0.0230]` (`10W/10L`, `0.6881`) | `0.7846` | `0.7067` | `0.9300/0.8200` | `0.6583/0.3800` | `0.9600/0.9200` | `0.5900/0.1800` | **`0.8917/0.7907`** | `79.26` | `53.61 / 103.30` |

- **Factorial Attribution & Functional Interference Discovery:** Across all 8 combinatorial backgrounds on `Pool R@5`: main effects / Shapley values are **`E1A = +0.622 / +0.620 pp`**, **`F1 = +0.124 / +0.169 pp`**, `F4B = -0.021 / +0.026 pp`, and `E3A = -0.467 / -0.403 pp`. Pairwise interaction contrasts prove strong destructive anti-interference **`I(F1, E3A) = -1.089 pp`** (`C3_F1_E1A_E3A - C2_F1_E1A = -0.0127 [-0.0247, -0.0021]`, `p_perm = 0.0408`) alongside `I(F1, E1A) = -0.454 pp`, `I(F1, F4B) = -0.403 pp`, `I(E3A, F4B) = +0.403 pp`, `I(E1A, F4B) = +0.111 pp`, and `I(E1A, E3A) = +0.009 pp`. Mechanistically, **`F1`'s query-tangent steering amplifies cyclic $A \leftrightarrow B$ mutual-citation pairs between two on-topic gold hops (`1 - (1-\rho)^2` closed-loop resonance)**, whereas `E3A`'s non-backtracking Hashimoto matrix explicitly forbids those 2-cycles (`A -> B -> A` masked), stripping gold mass while `F4B` adds `+44.4 ms` `p95` overhead (`C3_F1_E1A_F4B`).
- **Frozen `M_FINAL_1B` Systems Engineering (`eval3e_tier_b_mfinal_opt.json`):** Replacing query-time `scipy.special.betainc` + COO->CSR rebuild + `.T.tocsr()` inside `F1` with a static `16,384`-bin piecewise-linear `z = sqrt(-log1p(-c^2))` Beta-tail LUT (`max abs err < 1e-6`) and persisted static forward/transpose CSR arrays cut uncontended `tier_b_dev` latency by **`1.45x` median (`37.51 -> 25.91 ms`) and `1.63x` `p95` (`83.47 -> 51.12 ms`)** (`2Wiki` `16.60/24.48`, `MuSiQue` `32.01/50.44`, `HotpotQA` `26.83/38.86`, `PopQA` `20.66/25.83`, `ERB` `43.52/58.74 ms`, `RAM / maxTM = 25.35 / 79.00 MiB`) with **`0` diffs vs pre-optimization `C2_F1_E1A`** on `Recall@1/2/5/10` (`0.4169 / 0.6479 / 0.8155 / 0.9184`), `All-Hops@2/5` (`0.4074 / 0.6296`, `MH AH@5 = 0.7200`), `MRR` (`0.8959`), `NDCG@5` (`0.7925`), and `Rej AUROC` (`0.7382`).

### 4.7 `EVAL-3f`: Untouched `Tier E` Confirmatory Proof, `n = 200` `P3` QA, Post-Freeze Tiers C/D & `float16` Parity (`5b58ada`)

#### 1. Untouched `Tier E` Retrieval (`eval3f_tier_e_retrieval.json`, `[300, 400)`, `N = 400` + `200` Reject Probes, `0` Leaks)

| Tier E Arm (`N = 400`) | Pool `R@5` | 4-DS `H4` | `MH AH@5` | `2Wiki` `R5/AH5` | `MuSiQue` `R5/AH5` (`2h / 3h / 4h`) | `HotpotQA` `R5/AH5` | `PopQA` `R5/AH5` | `R@10 / R@20` | `MRR / NDCG@5` | `Rej AUROC` (`top1`) | `TM` (`RAM MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B3_CS6` (incumbent) | `0.6208` | `0.6208` | `0.4067` | `0.6800/0.3600` | `0.5333/0.2000` (`0.6875/0.4167` / `0.4259/0.0000` / `0.3125/0.0000`) | `0.8100/0.6600` | `0.4600/0.0000` | `0.6992/0.7550` | `0.7492 / 0.5833` | `0.5000` (`0.5000`) | `190.94` (`93.55`) | `78.9 / 100.9` |
| `B5S_RAND_SEEDS` | `0.3493` | `0.3493` | `0.0172` | `0.3845/0.0185` | `0.3048/0.0065` (`0.4141/0.0135` / `0.2134/0.0000` / `0.1828/0.0000`) | `0.4502/0.0265` | `0.2575/0.0040` | `0.3664/0.3804` | `0.7323 / 0.4309` | `0.5000` (`0.5000`) | `55.08` (`6.88`) | `771.3 / 1125.2` |
| `B5A_RAND_WALK` | `0.3552` | `0.3552` | `0.0232` | `0.3914/0.0320` | `0.3046/0.0050` (`0.4130/0.0104` / `0.2125/0.0000` / `0.1867/0.0000`) | `0.4527/0.0325` | `0.2720/0.0040` | `0.3689/0.3947` | `0.7360 / 0.4355` | `0.5000` (`0.5000`) | `55.08` (`6.88`) | `689.8 / 951.8` |
| `M_PURE_U61_1B` | `0.8023` | `0.8023` | `0.7067` | `0.9550/0.8700` | `0.6892/0.3600` (`0.8229/0.6667` / `0.6296/0.1111` / `0.4219/0.0000`) | `0.9450/0.8900` | `0.6200/0.2400` | `0.9006/0.9329` | `0.9089 / 0.7819` | `0.7068` (`0.6345`) | `72.41` (`21.95`) | `21.5 / 37.0` |
| **`M_FINAL_1B`** | **`0.8006`** | **`0.8006`** | **`0.7100`** | **`0.9450/0.8500`** | **`0.7175/0.4100` (`0.8438/0.7083` / `0.6667/0.1944` / `0.4531/0.0000`)** | **`0.9350/0.8700`** | **`0.6050/0.2100`** | **`0.8992/0.9258`** | **`0.9059 / 0.7810`** | **`0.7068` (`0.6345`)** | **`78.73` (`24.73`)** | **`22.1 / 38.8`** |

- **`Tier E` Confirmatory Gate Contrasts (`M_FINAL_1B`):**
  - **`P0 = PASS`:** `0` Class-1 tunable retrieval hyperparameters (`zero_magic_quotas=True`, `pure_diffusion=True`, `lexical_mode="pure_idf"`, `clone_dedup_mode="none"`, `rerank_mode="dpp_1bit"`, `edge_weight_mode="tangent_steer"`).
  - **`P1 = PASS` (`p_holm = 0.0008`):** `vs B5S_RAND_SEEDS`: `R@5 +0.4514 [+0.4269, +0.4759]`, `AH@5 +0.5711 [+0.5336, +0.6103]`; `vs B5A_RAND_WALK`: `R@5 +0.4454 [+0.4211, +0.4702]`, `AH@5 +0.5666 [+0.5291, +0.6055]`.
  - **`P2 = PASS` (`p_holm = 0.0004` vs `B3_CS6`):** Primary `MH AH@5 = +0.3033 [+0.2500, +0.3600]` (`0.7100 vs 0.4067`, $+74.6\%$ relative) and `macro R@5 = +0.1798 [+0.1552, +0.2048]` (`0.8006 vs 0.6208`, $d_z = +0.6811$). On `MuSiQue`, `M_FINAL_1B` lifted overall `AH@5` by `+21.0 pp` (`0.4100 vs 0.2000`, `2-hop 0.7083 vs 0.4167`, `3-hop 0.1944 vs 0.0000`) and `+5.0 pp` over `M_PURE_U61_1B` (`0.4100 vs 0.3600`). Every dataset (`2Wiki +0.2650 [+0.2175, +0.3125]`, `MuSiQue +0.1842 [+0.1317, +0.2367]`, `HotpotQA +0.1250 [+0.0700, +0.1800]`, `PopQA +0.1450 [+0.1000, +0.1900]`) and cutoff `Recall@{1,2,5,10,20}` (`+0.1079`, `+0.1890`, `+0.1798`, `+0.2000`, `+0.1708`) had strictly positive `ci_low > 0`.
  - **`P4 = PASS`:** `maxRAM = 24.73 MiB` (`3.78x` smaller than `B3_CS6` `93.55 MiB`), pooled `p50/p95 = 22.1 / 38.8 ms <= 151.35 ms` (`lat_cap_ms`), `conc_ok = True` (`0` concurrency errors across 16 worker threads), `Reject AUROC = 0.7068 >= 0.70 > top1_cosine 0.6345` (`0` masked leaks). (`P5 = FAIL`, `5/10 PASS` — §5.4.)

#### 2. Untouched `Tier E` Paired `P3` `dspy.ReAct` QA (`eval3f_tier_e_qa_n200.json`, `n = 200`, `50/DS`, `gemma-4-e4b`)

| Dataset (`n` / arm) | `B3_CS6` `R5 / AH5` | `B3_CS6` `EM / F1` | **`M_FINAL_1B` `R5 / AH5`** | **`M_FINAL_1B` `EM / F1`** | $\Delta$ `EM` / $\Delta$ `F1` (`M_FINAL_1B - B3_CS6`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `2wikimultihopqa` (`50`) | `0.7050 / 0.4000` | `0.5600 / 0.6129` | `0.9350 / 0.8400` | **`0.6400 / 0.7205`** | `+0.0800 / +0.1076` |
| `musique` (`50`) | `0.5383 / 0.2200` | `0.3400 / 0.4036` | `0.7283 / 0.4400` | **`0.3600 / 0.4893`** | `+0.0200 / +0.0857` |
| `hotpotqa` (`50`) | `0.8300 / 0.6600` | `0.5400 / 0.7095` | `0.9300 / 0.8600` | **`0.6200 / 0.7837`** | `+0.0800 / +0.0742` |
| `popqa` (`50`) | `0.4600 / 0.0000` | `0.2000 / 0.5015` | `0.6000 / 0.2000` | `0.1800 / 0.4941` | `-0.0200 / -0.0074` |
| `3-DS Multi-Hop` (`150`) | — | `0.4800 / 0.5753` | — | **`0.5400 / 0.6645`** | `+0.0600 / +0.0892` |
| **Total (`200`, `0` model errors)** | `0.6333 / 0.3200` | `0.4100 / 0.5569` | **`0.7983 / 0.5850`** | **`0.4500 / 0.6219`** | **`P3 = PASS` (`F1 +0.0650 [+0.0240, +0.1088]`, `p_perm = 0.0022`)** |

Exact `P3` statistical contrasts (`n = 200`, `400` paired ReAct episodes, `temperature = 1.0`, `0` model errors): **`ExactMatch = +0.0400 [-0.0100, +0.0900]`** (`p_perm = 0.1682`, `p_wilcoxon = 0.1167`, $d_z = +0.1113$); **`F1 = +0.0650 [+0.0240, +0.1088]`** (`p_perm = 0.0022`, `p_wilcoxon = 0.0072`, $d_z = +0.2117$).

#### 3. Post-Freeze Tiers C & D Comparison (`eval3f_tier_c_mfinal.json` & `eval3f_tier_d_mfinal.json`, Non-Confirmatory)

| Tier / Arm | Pool `R@5` | 4-DS `H4` | `MH AH@5` | `2Wiki` `R5/AH5` | `MuSiQue` `R5/AH5` (`2h / 3h / 4h`) | `HotpotQA` `R5/AH5` | `PopQA` `R5/AH5` | `ERB` `R5/AH5` | `R@10 / R@20` | `Rej AUROC` (`top1`) | `TM` (`RAM MiB`) | `p50 / p95` (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Tier C` `M_FINAL_1B`** | `0.7980` | `0.7717` | `0.6733` | `0.9275/0.8000` | `0.6292/0.3400` (`0.7455/0.5636` / `0.5952/0.1071` / `0.3088/0.0000`) | `0.9400/0.8800` | `0.5900/0.1900` | `0.9152/0.8556` | `0.8920/0.9261` | `0.7703` (`0.6783`) | `79.00` (`25.34`) | `26.7 / 51.6` |
| **`Tier D` `M_FINAL_1B`** | **`0.8149`** | **`0.7919`** | **`0.7167`** | `0.9625/0.8700` | **`0.6550/0.3500` (`0.7241/0.5000` / `0.5862/0.1379` / `0.5000/0.1538`)** | `0.9650/0.9300` | `0.5850/0.1700` | **`0.9152/0.8478`** | `0.8966/0.9294` | `0.7855` (`0.7291`) | `79.00` (`25.34`) | `25.7 / 50.0` |

Additional `M_FINAL_1B` telemetry: Tier C (`R@1/2 = 0.4189 / 0.6442`, `AH@5 = 0.6082`, `MRR = 0.8666`, `NDCG@5 = 0.7701`, `0` leaks); Tier D (`R@1/2 = 0.4476 / 0.6867`, `AH@5 = 0.6301`, `MRR = 0.9045`, `NDCG@5 = 0.8029`, `0` leaks).

#### 4. Full 5-Corpus IEEE `float16 -> float32` Storage & `Tier E` Retrieval Parity (`eval3f_f16_parity.json`)

| Corpus (`dim = 768`) | Chunks | `cos_dist` Max (`Mean`) | `l2_err` Max (`p99`) | 1-Bit RaBitQ Flips (`Rate`) | `ip_norm` Max Rel Diff | `d_eff` Abs Diff (`f32 -> f16`) |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| `2wikimultihopqa` | `6,980` | `1.439e-07` (`7.531e-09`) | `2.621e-04` (`2.415e-04`) | `552 / 5,360,640` (`1.030e-04`) | `3.607e-05` | `2.630e-04` (`232.6885 -> 232.6887`) |
| `musique` | `12,512` | `1.107e-07` (`7.348e-09`) | `2.703e-04` (`2.407e-04`) | `997 / 9,609,216` (`1.038e-04`) | `3.490e-05` | `2.097e-04` (`282.4618 -> 282.4620`) |
| `hotpotqa` | `10,785` | `1.340e-07` (`7.167e-09`) | `2.694e-04` (`2.417e-04`) | `834 / 8,282,880` (`1.007e-04`) | `3.905e-05` | `2.467e-04` (`281.4019 -> 281.4021`) |
| `popqa` | `9,880` | `1.245e-07` (`7.338e-09`) | `2.776e-04` (`2.414e-04`) | `772 / 7,587,840` (`1.017e-04`) | `3.375e-05` | `2.239e-04` (`258.9727 -> 258.9725`) |
| `enterprise_rag_bench` | `12,802` | `1.195e-07` (`7.858e-09`) | `2.717e-04` (`2.502e-04`) | `1,143 / 9,831,936` (`1.163e-04`) | `5.234e-05` | `2.635e-05` (`198.0119 -> 198.0118`) |
| **Pooled `5` Corpora** | **`52,959`** | **`1.439e-07` (`7.457e-09`)** | **`2.776e-04` (`2.444e-04`)** | **`4,298 / 40,672,512` (`0.01057%`)** | **`5.234e-05`** | **`Tier E` Parity: `4-DS H4 diff = 0.0000`, `MH AH@5 diff = 0.0000`** |

On `Tier E` (`N = 400` + `200` reject probes), replacing `float32` SQLite blobs with compressed IEEE `float16` (`<f2>`) vectors produced **`0` diffs on `Recall@1`, `Recall@5`, `Recall@10`, and `All-Hops@5`** across all `400` queries (`1` `Recall@2` diff on `PopQA`, `1` `Recall@20` diff on `MuSiQue`, `7` `top5_ids` swaps `[3, 2, 0, 2]`, `max |delta log_bf| = 0.1422`), `reject_auroc = 0.7068 (f32) vs 0.7065 (f16)` (`diff = -0.0003`), and `0` masked leaks — justifying `float16` persistence in plain SQLite (`chunks`) while halving vector disk size (`~81.1 MiB` vs `162.23 MiB` `vec0` chunks across `52,959` chunks).

---

## 5. Complete Component-Wise LOO Ablation Matrices (`P5` Occam's Razor Proof)

All contrasts compute `(Candidate - Ablation)` via `10,000`-resample stratified bootstrap (`+` means the component helps; `-` means the ablation outperformed the candidate).

### 5.1 `M_PURE_0P` Complete 8-Arm Single-Toggle LOO Matrix (`AP1`–`AP8`)

#### Untouched Tier D (`N = 492`, `eval3_tier_d_retrieval.json`, `P5 = 5/8 PASS`)

| Ablation (`M_PURE_0P` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ Pool `AH@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `Rej AUROC` | `RAM` (`MiB`, `Abl/Cand`) | `p95` (`ms`, `Abl/Cand`) | Harness `P5` |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `AP1_NO_RABITQ_EXACT` | `-0.0005` `[-0.0020, +0.0007]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0017` | `57.27` (`2.25x`) | `75.2` (`0.87x`) | **PASS** |
| `AP2_1BIT_RABITQ` | `-0.0039` `[-0.0120, +0.0042]` | `+0.0041` `[-0.0102, +0.0183]` | `+0.0067` `[-0.0133, +0.0267]` | `+0.0114` | `21.90` (`0.86x`) | `81.8` (`0.95x`) | **FAIL** |
| `AP3_UNICODE61_LEX` (`dep_deleted`) | **`-0.0088`** `[-0.0160, -0.0019]` | `-0.0061` `[-0.0203, +0.0061]` | `-0.0100` `[-0.0300, +0.0100]` | `+0.0089` | `26.05` (`1.02x`) | `64.5` (`0.75x`) | **FAIL** |
| `AP4_NO_DIFFUSION` | **`+0.0716`** `[+0.0534, +0.0898]` | **`+0.1484`** `[+0.1138, +0.1829]` | **`+0.1967`** `[+0.1467, +0.2500]` | `+0.0000` | `25.38` (`1.00x`) | `79.9` (`0.93x`) | **PASS** |
| `AP5_NO_TITLE_CIT` | **`+0.0632`** `[+0.0488, +0.0783]` | **`+0.1138`** `[+0.0874, +0.1423]` | **`+0.1367`** `[+0.0967, +0.1800]` | `+0.0045` | `19.81` (`0.78x`) | `81.0` (`0.94x`) | **PASS** |
| `AP6_NO_PMI_CHAN` | `+0.0123` `[-0.0008, +0.0259]` | `+0.0081` `[-0.0163, +0.0346]` | `+0.0133` `[-0.0200, +0.0500]` | `-0.0006` | `24.05` (`0.94x`) | `76.1` (`0.88x`) | **FAIL** |
| `AP7_NO_NOISYOR` | **`+0.0152`** `[+0.0086, +0.0224]` | **`+0.0325`** `[+0.0183, +0.0488]` | **`+0.0533`** `[+0.0300, +0.0800]` | `+0.0000` | `25.38` (`1.00x`) | `86.1` (`1.00x`) | **PASS** |
| `AP8_NO_MAXP` | `+0.0080` `[+0.0037, +0.0131]` | **`+0.0142`** `[+0.0041, +0.0264]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` | `25.38` (`1.00x`) | `84.8` (`0.98x`) | **PASS** |

#### Tier C Comparison (`N = 490`, `eval3_tier_c_mpure0p_comparison.json`, `P5 = 5/8 PASS`)

| Ablation (`M_PURE_0P` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ Pool `AH@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `Rej AUROC` | `RAM` (`MiB`, `Abl/Cand`) | `p95` (`ms`, `Abl/Cand`) | Harness `P5` |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `AP1_NO_RABITQ_EXACT` | `-0.0011` `[-0.0029, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0033` | `57.27` (`2.20x`) | `81.9` (`0.92x`) | **PASS** |
| `AP2_1BIT_RABITQ` | `-0.0030` `[-0.0109, +0.0049]` | `-0.0020` `[-0.0184, +0.0143]` | `+0.0000` `[-0.0233, +0.0233]` | `+0.0062` | `21.97` (`0.84x`) | `85.0` (`0.95x`) | **FAIL** |
| `AP3_UNICODE61_LEX` (`dep_deleted`) | **`-0.0126`** `[-0.0224, -0.0036]` | **`-0.0204`** `[-0.0367, -0.0041]` | `-0.0133` `[-0.0367, +0.0100]` | `+0.0089` | `26.05` (`1.00x`) | `63.7` (`0.71x`) | **FAIL** |
| `AP4_NO_DIFFUSION` | **`+0.0788`** `[+0.0588, +0.0982]` | **`+0.1571`** `[+0.1204, +0.1939]` | **`+0.2033`** `[+0.1500, +0.2567]` | `+0.0000` | `25.38` (`0.98x`) | `87.2` (`0.98x`) | **PASS** |
| `AP5_NO_TITLE_CIT` | **`+0.0712`** `[+0.0555, +0.0876]` | **`+0.1388`** `[+0.1082, +0.1714]` | **`+0.1600`** `[+0.1167, +0.2033]` | `+0.0135` | `19.81` (`0.76x`) | `91.1` (`1.02x`) | **PASS** |
| `AP6_NO_PMI_CHAN` | `+0.0088` `[-0.0034, +0.0217]` | `+0.0143` `[-0.0102, +0.0388]` | `+0.0000` `[-0.0333, +0.0333]` | `+0.0058` | `24.05` (`0.92x`) | `80.8` (`0.91x`) | **FAIL** |
| `AP7_NO_NOISYOR` | **`+0.0143`** `[+0.0087, +0.0209]` | **`+0.0306`** `[+0.0163, +0.0449]` | **`+0.0500`** `[+0.0267, +0.0733]` | `+0.0000` | `25.38` (`0.98x`) | `94.2` (`1.06x`) | **PASS** |
| `AP8_NO_MAXP` | `+0.0086` `[+0.0039, +0.0140]` | **`+0.0184`** `[+0.0082, +0.0306]` | `+0.0067` `[+0.0000, +0.0167]` | `+0.0000` | `25.45` (`0.98x`) | `100.8` (`1.13x`) | **PASS** |

#### Tier B (`N = 243`, `eval2e_mpure0p_loo.json`, `P5 = 5/8 PASS`)

On Tier B (`N=243`), `AP1` (`2.20x` RAM), `AP3` (`-0.0027` non-inferior on Tier B only, `dep_deleted=True`), `AP4` (`+0.1028 / +0.2600`), `AP5` (`+0.0652 / +0.1533`), and `AP7` (`+0.0168 / +0.0600`) passed, whereas `AP2` (`+0.0047`), `AP6` (`+0.0191 [-0.0034, +0.0410]`), and `AP8` (`+0.0082 [+0.0000, +0.0206]`) missed strict `N=243` bounds; see [mdl-dag-sprint.md §EVAL-2e](mdl-dag-sprint.md#eval-2e--2026-11-22).

**Key Diagnostic Finding Across All Three Splits (`N = 243`, `490`, `492`):**
1. The core parameter-free engine (`AP1` 4-bit RaBitQ + `AP4` converged diffusion + `AP5` directed/Bayesian reverse title citations + `AP7` multi-entity conditional Noisy-OR + `AP8` MaxP deferral) passes `P5` independently on both large splits (`N = 492` and `N = 490`) with large effect sizes and tight CIs.
2. Under `M_PURE_0P`'s continuous global scoring (unlike finite-subgraph `M_WIN`), exact unweighted trigram `lex_n` (`AP3` disabled -> enabled) **degraded** English accuracy across both Tier D (`-0.0088 [-0.0160, -0.0019]`, `unicode61` wins `17` queries vs trigram `5`, `470` ties) and Tier C (`-0.0126 [-0.0224, -0.0036]`, `unicode61` wins `18` vs `4`, including `ERB 0.9257 vs 0.8916`), while adding `+34%` Tier D (`64.5 ms -> 86.2 ms`) to `+40%` Tier C (`63.7 ms -> 89.0 ms`) `p95` latency and inflating the pure FTS5 index payload (`chunk_fts_data`) **`5.61x`** (`256.59 MiB` vs `45.76 MiB` in `dbstat`; Section 3.2 separates this from the `1.78x` schema-confounded `274.46` vs `487.42 MiB` whole-file delta). The root cause is that trigram prefixes/stems/compounds introduce spurious substring false positives across 52k chunks when no discrete subgraph cap filters them, whereas whole-token `unicode61` maintains sharp entity/technical term precision.
3. **`AP6` (1-bit PMI co-occurrence):** Adds `+0.0123` `Pool R@5` on Tier D and `+0.0191` on Tier B (improving `2Wiki`/`HotpotQA` topical bridging), yet misses CI $> 0$ (`-0.0008` Tier D, `-0.0034` Tier C/B) at `+1.40 MiB` RAM.
4. **`AP2` (`4-bit` vs `1-bit` RaBitQ):** Under full-matrix continuous diffusion (`bnb_exact_fetches = 0`), `4-bit` RaBitQ delivers nearly identical retrieval quality to `exact float32` (`dR5 = -0.0005` Tier D, `-0.0011` Tier C) at `2.25x` lower RAM (`25.45 vs 57.27 MiB`); `1-bit` RaBitQ (`21.90 MiB`) was non-inferior in `R@5` (`-0.0039`), losing `-0.0114` Tier D `Reject AUROC` (`0.7862 vs 0.7976`).

### 5.2 Complete Historical `M_FULL` (`A1`–`A9`), `M_WIN` (`AW1`–`AW13`) & `M_WIN_0P` (`AZ1`–`AZ21`) LOO Tables

#### Legacy Tier B `M_FULL` (`A1`–`A9`, `N = 243`, `eval2c_mfull_loo.json`, `P5 = 2/9 PASS`)

| Ablation (`M_FULL` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `Rej` | `RAM Ratio` | `p95 Ratio` | `P5` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `A1_NO_RABITQ_EXACT` | `-0.0010` `[-0.0072, +0.0051]` | `-0.0067` `[-0.0200, +0.0000]` | `+0.0000` | `5.73x` | `0.68x` | **PASS** |
| `A2_NO_PARTIAL_CORR` | `-0.0186` `[-0.0406, +0.0023]` | `-0.0400` `[-0.0933, +0.0133]` | `-0.0118` | `1.00x` | `0.73x` | **FAIL** |
| `A3_NO_GIBBS_HEURISTIC` | `-0.0391` `[-0.0650, -0.0120]` | `-0.0200` `[-0.0867, +0.0533]` | `-0.0527` | `1.00x` | `1.01x` | **FAIL** |
| `A4_WALK_GIBBS` | `+0.0147` `[-0.0038, +0.0346]` | `+0.0267` `[-0.0200, +0.0733]` | `-0.0099` | `1.00x` | `1.01x` | **FAIL** |
| `A5_REV1_GIBBS` | `+0.0568` `[+0.0284, +0.0871]` | `+0.0267` `[-0.0267, +0.0800]` | `-0.0703` | `1.00x` | `0.92x` | **PASS** |
| `A6_RENYI2_DEFF` | `-0.0056` `[-0.0208, +0.0086]` | `-0.0067` `[-0.0333, +0.0200]` | `+0.0049` | `1.00x` | `1.00x` | **FAIL** |
| `A7_NO_PMI_PHRASES` | `+0.0091` `[-0.0046, +0.0238]` | `+0.0133` `[-0.0200, +0.0467]` | `+0.0106` | `1.00x` | `0.90x` | **FAIL** |
| `A8_NO_DECOMPOUND` | `-0.0007` `[-0.0075, +0.0055]` | `+0.0000` `[-0.0200, +0.0200]` | `-0.0004` | `1.00x` | `1.01x` | **FAIL** |
| `A9_UNICODE61_LEX` (`dep_deleted`) | `-0.0018` `[-0.0314, +0.0282]` | `-0.0267` `[-0.0800, +0.0333]` | `-0.0245` | `1.00x` | `0.88x` | **FAIL** |

#### Legacy Tier B `M_WIN` (`AW1`–`AW13`, `N = 243`, `eval2c_mwin_loo.json`, `P5 = 9/13 PASS`)

| Ablation (`M_WIN` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `Rej` | `RAM Ratio` | `p95 Ratio` | `P5` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `AW1_NO_RABITQ_EXACT` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` | `3.91x` | `1.56x` | **PASS** |
| `AW2_NO_PARTIAL_CORR` | `+0.0096` `[-0.0007, +0.0206]` | `+0.0400` `[+0.0133, +0.0733]` | `+0.0027` | `1.00x` | `0.71x` | **PASS** |
| `AW3_NO_GIBBS_HEURISTIC` | `+0.0089` `[-0.0021, +0.0206]` | `+0.0400` `[+0.0133, +0.0733]` | `-0.0128` | `1.00x` | `0.94x` | **PASS** |
| `AW4_WALK_GIBBS` | `-0.0000` `[-0.0096, +0.0096]` | `+0.0000` `[-0.0333, +0.0333]` | `-0.0009` | `1.00x` | `1.02x` | **FAIL** |
| `AW5_UNNORM_PRIOR` | `-0.0105` `[-0.0226, +0.0009]` | `-0.0200` `[-0.0533, +0.0133]` | `+0.0361` | `1.00x` | `0.95x` | **FAIL** |
| `AW6_RENYI2_DEFF` | `+0.0160` `[+0.0027, +0.0309]` | `+0.0600` `[+0.0267, +0.1000]` | `-0.0114` | `1.00x` | `0.93x` | **PASS** |
| `AW7_NO_PMI_PHRASES` | `+0.0011` `[-0.0110, +0.0130]` | `+0.0000` `[-0.0400, +0.0400]` | `+0.0142` | `1.00x` | `0.92x` | **FAIL** |
| `AW8_NO_DECOMPOUND` | `+0.0027` `[-0.0027, +0.0082]` | `+0.0000` `[-0.0200, +0.0200]` | `+0.0201` | `1.00x` | `0.93x` | **FAIL** |
| `AW9_UNICODE61_LEX` (`dep_deleted`) | `+0.0120` `[-0.0082, +0.0329]` | `+0.0467` `[+0.0000, +0.1000]` | `+0.0175` | `1.00x` | `0.68x` | **PASS** |
| `AW10_LINEAR_DECODE` | `+0.0694` `[+0.0409, +0.0985]` | `+0.1267` `[+0.0533, +0.2000]` | `+0.0000` | `1.00x` | `1.06x` | **PASS** |
| `AW11_RAW_REJECT_SCORE` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0629` | `1.00x` | `0.99x` | **PASS** |
| `AW12_1BIT_RABITQ` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `-0.0006` | `0.68x` | `2.13x` | **PASS** |
| `AW13_FULL_BUDGET` | `-0.0003` `[-0.0082, +0.0075]` | `+0.0067` `[-0.0133, +0.0267]` | `-0.0123` | `1.00x` | `2.72x` | **PASS** |

#### Superseded Tier B `M_WIN_0P` (`AZ1`–`AZ21`, `N = 243`, `eval2d_mwin0p_loo.json`, `P5 = 9/21 PASS`)

| Ablation (`M_WIN_0P` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `Rej` | `RAM Ratio` | `p95 Ratio` | `P5` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `AZ1_NO_RABITQ_EXACT` | `-0.0041` `[-0.0146, +0.0035]` | `+0.0000` `[+0.0000, +0.0000]` | `-0.0034` | `2.16x` | `1.55x` | **PASS** |
| `AZ2_NO_PARTIAL_CORR` | `+0.0021` `[-0.0103, +0.0134]` | `+0.0067` `[+0.0000, +0.0200]` | `-0.0082` | `0.98x` | `0.79x` | **FAIL** |
| `AZ3_NO_GIBBS_HEURISTIC` | `+0.0597` `[+0.0370, +0.0819]` | `+0.0867` `[+0.0400, +0.1400]` | `-0.0175` | `0.99x` | `1.02x` | **PASS** |
| `AZ4_WALK_GIBBS` | `+0.0016` `[-0.0050, +0.0085]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0036` | `0.98x` | `1.17x` | **FAIL** |
| `AZ5_UNNORM_PRIOR` | `+0.0235` `[+0.0018, +0.0455]` | `+0.0333` `[-0.0133, +0.0800]` | `+0.0070` | `0.98x` | `0.96x` | **PASS** |
| `AZ6_RENYI2_DEFF` | `+0.0073` `[-0.0088, +0.0223]` | `+0.0133` `[-0.0133, +0.0400]` | `-0.0220` | `0.98x` | `0.99x` | **FAIL** |
| `AZ7_NO_PMI_PHRASES` | `-0.0010` `[-0.0168, +0.0144]` | `+0.0067` `[-0.0333, +0.0467]` | `+0.0188` | `0.93x` | `0.98x` | **FAIL** |
| `AZ8_NO_DECOMPOUND` | `+0.0046` `[-0.0009, +0.0111]` | `+0.0133` `[+0.0000, +0.0333]` | `+0.0095` | `0.98x` | `0.97x` | **FAIL** |
| `AZ9_UNICODE61_LEX` (`dep_deleted`) | `-0.0049` `[-0.0181, +0.0067]` | `-0.0133` `[-0.0400, +0.0133]` | `-0.0131` | `1.00x` | `0.95x` | **FAIL** |
| `AZ10_CONE_DECODE` | `+0.1406` `[+0.1140, +0.1683]` | `+0.2667` `[+0.2000, +0.3400]` | `+0.0000` | `0.98x` | `1.62x` | **PASS** |
| `AZ11_RAW_REJECT_SCORE` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0807` | `0.98x` | `1.03x` | **PASS** |
| `AZ12_1BIT_RABITQ` | `+0.0035` `[-0.0063, +0.0139]` | `+0.0000` `[-0.0267, +0.0267]` | `-0.0019` | `0.85x` | `4.12x` | **PASS** |
| `AZ13_NO_TITLE_REF` | `+0.1043` `[+0.0796, +0.1293]` | `+0.1733` `[+0.1133, +0.2400]` | `+0.0019` | `0.78x` | `1.60x` | **PASS** |
| `AZ14_DYN_HORIZON` | `-0.0011` `[-0.0142, +0.0109]` | `+0.0067` `[-0.0133, +0.0333]` | `-0.0180` | `0.98x` | `0.90x` | **FAIL** |
| `AZ15_FIXED_BUDGET` | `-0.0008` `[-0.0114, +0.0074]` | `+0.0067` `[+0.0000, +0.0200]` | `-0.0036` | `0.98x` | `1.64x` | **FAIL** |
| `AZ16_LINEAR_UNTEMPERED` | `+0.0750` `[+0.0477, +0.1020]` | `+0.1067` `[+0.0333, +0.1733]` | `+0.0000` | `0.98x` | `0.99x` | **PASS** |
| `AZ17_EXACT_SUBGRAPH_VECS` | `+0.0000` `[-0.0056, +0.0046]` | `+0.0000` `[+0.0000, +0.0000]` | `-0.0038` | `0.98x` | `1.22x` | **FAIL** |
| `AZ18_NO_ENTITY_CSR` | `-0.0017` `[-0.0179, +0.0134]` | `-0.0133` `[-0.0533, +0.0267]` | `-0.0028` | `0.98x` | `1.05x` | **FAIL** |
| `AZ19_NO_CAP_UNI` | `+0.0016` `[-0.0035, +0.0078]` | `+0.0000` `[-0.0200, +0.0200]` | `-0.0011` | `0.93x` | `0.98x` | **FAIL** |
| `AZ20_NO_ENTITY_HMM` | `+0.0123` `[+0.0003, +0.0245]` | `+0.0333` `[+0.0000, +0.0667]` | `+0.0000` | `0.84x` | `0.99x` | **PASS** |
| `AZ21_NO_PRED_ROUTING` | `-0.0020` `[-0.0088, +0.0040]` | `-0.0067` `[-0.0200, +0.0000]` | `-0.0059` | `0.98x` | `0.79x` | **FAIL** |

#### Tier C `M_WIN_0P` `AZ1`–`AZ21` Paired Results (`N = 490`, `eval3_tier_c_retrieval.json`, `P5 = 7/21 PASS`)

| Ablation (`M_WIN_0P` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ `MH AH@5` | $\Delta$ `Rej` | `RAM Ratio` | `p95 Ratio` | `P5` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `AZ1_NO_RABITQ_EXACT` | `-0.0051` `[-0.0097, -0.0015]` | `-0.0067` | `+0.0027` | `2.19x` | `1.59x` | **FAIL** |
| `AZ2_NO_PARTIAL_CORR` | `-0.0037` `[-0.0088, +0.0010]` | `+0.0000` | `-0.0019` | `0.99x` | `0.83x` | **FAIL** |
| `AZ3_NO_GIBBS_HEURISTIC` | `+0.0524` `[+0.0375, +0.0677]` | `+0.0667` | `-0.0592` | `0.99x` | `1.02x` | **PASS** |
| `AZ4_WALK_GIBBS` | `+0.0007` `[-0.0046, +0.0065]` | `-0.0033` | `-0.0140` | `0.99x` | `1.15x` | **FAIL** |
| `AZ5_UNNORM_PRIOR` | `+0.0283` `[+0.0150, +0.0423]` | `+0.0467` | `+0.0024` | `0.99x` | `1.03x` | **PASS** |
| `AZ6_RENYI2_DEFF` | `-0.0044` `[-0.0142, +0.0056]` | `+0.0067` | `-0.0079` | `1.01x` | `1.01x` | **FAIL** |
| `AZ7_NO_PMI_PHRASES` | `-0.0050` `[-0.0157, +0.0058]` | `+0.0000` | `+0.0048` | `0.94x` | `1.04x` | **FAIL** |
| `AZ8_NO_DECOMPOUND` | `+0.0001` `[-0.0029, +0.0034]` | `+0.0033` | `-0.0001` | `0.99x` | `3.20x` | **PASS** |
| `AZ9_UNICODE61_LEX` (`dep_deleted`) | `-0.0116` `[-0.0230, -0.0009]` | `+0.0000` | `+0.0001` | `1.01x` | `2.83x` | **FAIL** |
| `AZ10_CONE_DECODE` | `+0.0910` `[+0.0718, +0.1095]` | `+0.1767` | `+0.0000` | `0.99x` | `3.19x` | **PASS** |
| `AZ11_RAW_REJECT_SCORE` | `+0.0000` `[+0.0000, +0.0000]` | `+0.0000` | `+0.1062` | `0.99x` | `1.01x` | **PASS** |
| `AZ12_1BIT_RABITQ` | `-0.0058` `[-0.0133, +0.0015]` | `+0.0033` | `-0.0012` | `0.86x` | `2.21x` | **FAIL** |
| `AZ13_NO_TITLE_REF` | `+0.0799` `[+0.0633, +0.0964]` | `+0.1433` | `-0.0005` | `0.79x` | `1.89x` | **PASS** |
| `AZ14_DYN_HORIZON` | `-0.0080` `[-0.0146, -0.0017]` | `-0.0133` | `-0.0180` | `0.99x` | `1.62x` | **FAIL** |
| `AZ15_FIXED_BUDGET` | `+0.0000` `[-0.0060, +0.0061]` | `+0.0000` | `-0.0015` | `0.99x` | `1.87x` | **FAIL** |
| `AZ16_LINEAR_UNTEMPERED` | `+0.0588` `[+0.0408, +0.0773]` | `+0.0933` | `+0.0000` | `0.99x` | `1.15x` | **PASS** |
| `AZ17_EXACT_SUBGRAPH_VECS` | `-0.0020` `[-0.0061, +0.0020]` | `+0.0000` | `+0.0027` | `0.99x` | `1.22x` | **FAIL** |
| `AZ18_NO_ENTITY_CSR` | `-0.0117` `[-0.0230, -0.0003]` | `+0.0100` | `-0.0236` | `0.99x` | `1.08x` | **FAIL** |
| `AZ19_NO_CAP_UNI` | `+0.0027` `[-0.0024, +0.0082]` | `+0.0067` | `+0.0020` | `0.95x` | `0.94x` | **FAIL** |
| `AZ20_NO_ENTITY_HMM` | `+0.0047` `[-0.0065, +0.0164]` | `+0.0100` | `-0.0001` | `0.86x` | `1.00x` | **FAIL** |
| `AZ21_NO_PRED_ROUTING` | `+0.0014` `[-0.0037, +0.0065]` | `+0.0033` | `-0.0041` | `0.99x` | `0.84x` | **FAIL** |

### 5.3 `EVAL-3b` Single-Toggle LOO Matrices (`AU1`–`AU9` & `A1B1`–`A1B8`, `P5 = FAIL`)

Contrasts report `(Candidate - Ablation)` across `eval3b_tier_d_shootout.json`, `eval3b_tier_d_1bit.json`, and `eval3b_tier_c_1bit.json`:

| Tier D `M_PURE_U61` (`AU1`–`AU9`, `N = 492`, `7/9 PASS`) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `ERB R@5` | $\Delta$ `Rej AUROC` | `RAM` (`Abl/Cand`) | `p95` (`Abl/Cand`) | Harness `P5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AU1_NO_RABITQ_EXACT` | `-0.0009` `[-0.0042, +0.0021]` | `-0.0033` `[-0.0100, +0.0000]` | `-0.0012` | `+0.0008` | `2.25x` | `0.79x` | **PASS** |
| `AU2_1BIT_RABITQ` | `-0.0010` `[-0.0081, +0.0058]` | `+0.0033` `[-0.0167, +0.0233]` | `-0.0026` | `+0.0086` | `0.86x` | `0.94x` | **FAIL** |
| `AU3_TRIGRAM_LEX` | `+0.0108` `[+0.0039, +0.0182]` | `+0.0133` `[-0.0033, +0.0333]` | `-0.0018` | `-0.0035` | `1.01x` | `1.76x` | **PASS** |
| `AU4_NO_DIFFUSION` | `+0.0847` `[+0.0670, +0.1025]` | `+0.2233` `[+0.1733, +0.2733]` | `-0.0171` | `+0.0000` | `1.00x` | `0.83x` | **PASS** |
| `AU5_NO_TITLE_CIT` | `+0.0723` `[+0.0578, +0.0876]` | `+0.1533` `[+0.1133, +0.1933]` | `+0.0024` | `+0.0042` | `0.78x` | `0.89x` | **PASS** |
| `AU6_NO_PMI_CHAN` | `+0.0189` `[+0.0067, +0.0315]` | `+0.0300` `[+0.0000, +0.0633]` | `+0.0430` | `-0.0055` | `0.95x` | `0.90x` | **PASS** |
| `AU7_NO_NOISYOR` | `+0.0183` `[+0.0112, +0.0259]` | `+0.0533` `[+0.0300, +0.0800]` | `+0.0000` | `+0.0000` | `1.01x` | `1.02x` | **PASS** |
| `AU8_NO_MAXP` | `+0.0104` `[+0.0054, +0.0163]` | `+0.0033` `[+0.0000, +0.0100]` | `+0.0319` | `+0.0000` | `1.00x` | `1.03x` | **PASS** |
| `AU9_NO_DECOMPOUND` | `+0.0020` `[+0.0000, +0.0047]` | `+0.0033` `[+0.0000, +0.0100]` | `+0.0000` | `+0.0036` | `1.00x` | `1.20x` | **FAIL** |

On Tier C (`eval3b_tier_c_shootout.json`), `M_PURE_U61` similarly passed `7/9` (`AU3_TRIGRAM_LEX` `+0.0143` `R@5` / `1.96x` `p95`; `AU2` and `AU9` failed). Moving `M_PURE_U61` to 1-bit SoA RaBitQ (`M_PURE_U61_1B`, `A1B1`–`A1B8`, `6/8 PASS` on both Tier D and Tier C):

| `M_PURE_U61_1B` (`A1B1`–`A1B8`) | Tier D $\Delta$ `R@5` `[95% CI]` | Tier D $\Delta$ `MH AH@5` | Tier D `RAM/p95` (`P5`) | Tier C $\Delta$ `R@5` `[95% CI]` | Tier C $\Delta$ `MH AH@5` | Tier C `RAM/p95` (`P5`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `A1B1_NO_RABITQ_EXACT` | `+0.0001` `[-0.0069, +0.0072]` | `-0.0067` | `2.54x / 0.84x` (**PASS**) | `-0.0030` `[-0.0116, +0.0055]` | `-0.0100` | `2.54x / 0.74x` (**FAIL**) |
| `A1B2_TRIGRAM_LEX` | `+0.0080` `[+0.0007, +0.0154]` | `+0.0167` | `0.97x / 1.87x` (**FAIL**) | `+0.0085` `[+0.0007, +0.0171]` | `+0.0067` | `1.00x / 2.03x` (**PASS**) |
| `A1B3_NO_DIFFUSION` | `+0.0839` `[+0.0658, +0.1021]` | `+0.2200` | `0.97x / 0.81x` (**PASS**) | `+0.0905` `[+0.0716, +0.1094]` | `+0.2233` | `0.97x / 0.85x` (**PASS**) |
| `A1B4_NO_TITLE_CIT` | `+0.0725` `[+0.0576, +0.0879]` | `+0.1400` | `0.72x / 0.90x` (**PASS**) | `+0.0840` `[+0.0679, +0.1009]` | `+0.1833` | `0.72x / 0.90x` (**PASS**) |
| `A1B5_NO_PMI_CHAN` | `+0.0249` `[+0.0121, +0.0379]` | `+0.0500` | `0.91x / 0.87x` (**PASS**) | `+0.0205` `[+0.0065, +0.0351]` | `+0.0233` | `0.91x / 0.89x` (**PASS**) |
| `A1B6_NO_NOISYOR` | `+0.0185` `[+0.0113, +0.0261]` | `+0.0533` | `0.97x / 1.00x` (**PASS**) | `+0.0194` `[+0.0117, +0.0276]` | `+0.0667` | `0.97x / 0.86x` (**PASS**) |
| `A1B7_NO_MAXP` | `+0.0113` `[+0.0059, +0.0173]` | `+0.0033` | `0.97x / 1.00x` (**PASS**) | `+0.0169` `[+0.0096, +0.0254]` | `+0.0100` | `0.97x / 0.89x` (**PASS**) |
| `A1B8_NO_DECOMPOUND` | `+0.0030` `[-0.0010, +0.0075]` | `+0.0033` | `1.00x / 0.96x` (**FAIL**) | `+0.0031` `[+0.0000, +0.0071]` | `+0.0100` | `1.00x / 0.85x` (**FAIL**) |

### 5.4 `EVAL-3f` Complete Untouched `Tier E` `AF1`–`AF10` Matrix (`N = 400`, `M_FINAL_1B`, `P5 = 5/10 PASS`)

| Ablation (`M_FINAL_1B` minus 1 toggle) | $\Delta$ Pool `R@5` `[95% CI]` | $\Delta$ Pool `AH@5` `[95% CI]` | $\Delta$ `MH AH@5` `[95% CI]` | $\Delta$ `Rej AUROC` (`CI_low`) | `RAM` (`MiB`, `Abl/Cand`) | `p95` (`ms`, `Abl/Cand`) | Harness `P5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AF1_NO_RABITQ_EXACT` | `-0.0108` `[-0.0221, +0.0000]` | `-0.0225` `[-0.0450, -0.0025]` | `-0.0267` `[-0.0500, -0.0033]` | `+0.0008` `(-0.0143)` | `59.03` (`2.39x`) | `33.0` (`0.85x`) | **FAIL** |
| `AF2_TRIGRAM_LEX` | `+0.0073` `[-0.0013, +0.0167]` | `+0.0050` `[-0.0100, +0.0200]` | `+0.0000` `[-0.0200, +0.0200]` | `+0.0105` `(-0.0065)` | `24.73` (`1.00x`) | `57.9` (`1.49x`) | **FAIL** |
| `AF3_NO_DIFFUSION` | `+0.1163` `[+0.0938, +0.1388]` | `+0.2250` `[+0.1800, +0.2700]` | `+0.2300` `[+0.1767, +0.2833]` | `+0.0000` `(+0.0000)` | `24.73` (`1.00x`) | `25.0` (`0.64x`) | **PASS** |
| `AF4_NO_TITLE_CIT` | `+0.0981` `[+0.0779, +0.1185]` | `+0.1725` `[+0.1350, +0.2125]` | `+0.1667` `[+0.1233, +0.2133]` | `+0.0296` `(+0.0133)` | `16.34` (`0.66x`) | `27.5` (`0.71x`) | **PASS** |
| `AF5_NO_PMI_CHAN` | `+0.0154` `[+0.0010, +0.0302]` | `+0.0200` `[-0.0100, +0.0500]` | `+0.0400` `[+0.0067, +0.0733]` | `+0.0121` `(-0.0133)` | `23.39` (`0.95x`) | `33.8` (`0.87x`) | **PASS** |
| `AF6_NO_NOISYOR` | `+0.0163` `[+0.0081, +0.0256]` | `+0.0275` `[+0.0100, +0.0475]` | `+0.0367` `[+0.0133, +0.0633]` | `+0.0000` `(+0.0000)` | `24.79` (`1.00x`) | `36.8` (`0.95x`) | **PASS** |
| `AF7_NO_MAXP` | `+0.0054` `[+0.0013, +0.0108]` | `+0.0100` `[+0.0025, +0.0200]` | `+0.0133` `[+0.0033, +0.0267]` | `+0.0000` `(+0.0000)` | `24.82` (`1.00x`) | `39.0` (`1.01x`) | **PASS** |
| `AF8_NO_DECOMPOUND` | `+0.0010` `[-0.0019, +0.0044]` | `+0.0000` `[-0.0075, +0.0075]` | `+0.0000` `[-0.0100, +0.0100]` | `+0.0082` `(-0.0013)` | `24.73` (`1.00x`) | `38.3` (`0.99x`) | **FAIL** |
| `AF9_NO_TANGENT_STEER` | `-0.0046` `[-0.0125, +0.0027]` | `-0.0100` `[-0.0250, +0.0050]` | `+0.0000` `[-0.0167, +0.0167]` | `+0.0000` `(+0.0000)` | `21.91` (`0.89x`) | `36.0` (`0.93x`) | **FAIL** |
| `AF10_NO_DPP_1BIT` | `-0.0004` `[-0.0071, +0.0063]` | `-0.0025` `[-0.0150, +0.0100]` | `-0.0033` `[-0.0200, +0.0133]` | `+0.0000` `(+0.0000)` | `24.73` (`1.00x`) | `40.3` (`1.04x`) | **FAIL** |

The core diffusion, title citation, 1-bit PMI, conditional Noisy-OR, and MaxP operators (`AF3`–`AF7`) all pass strictly (`ci_low > 0`), proving zero redundant core machinery; `AF1` (`float32` exact) confirms the small accuracy–memory frontier (`+2.25 pp AH@5` at `2.39x` RAM / `4.36x` peak `tracemalloc`: `59.03 / 343.31 MiB` vs `24.73 / 78.73 MiB`), while `AF2` confirms `1.49x` faster `p95` under `unicode61` and `AF9`/`AF10` preserve `F1 + E1A`'s `+0.3033 MH AH@5` and `+21.0 pp MuSiQue AH@5` generalization on unseen data.

---

## 6. Systems Telemetry, `BnB` Fetch Ratios & Indexing Speed

All memory measurements report exact `bytes / 1048576.0` (`MiB`) per [mdl-dag-sprint.md](mdl-dag-sprint.md); per-dataset `M_PURE_0P` Tier D `VectorSearch` RAM is `14.76 / 25.45 / 22.13 / 21.39 / 18.24 MiB` (`2Wiki / MuSiQue / HotpotQA / PopQA / ERB`). Normalized `BnB Fetch Ratio` equals `mean(bnb_exact_fetches / bnb_total_chunks)` over all `500` Tier D queries (`B0_DPR` over `492` scorable queries is `2,428.9` fetches / `23.59%`). Historical Tier B/C BnB fetch ratios moved from `33.34%` (`3,476.6`, `M_FULL`) -> `0.52%` (`51.7`, `M_WIN`) -> `0.38%` (`38.4–38.9`, `M_WIN_0P`) -> **`0.00%` (`0.0`, `M_PURE_0P`)**.

| Tier D System Arm | `p50` (`ms`) | `p95` (`ms`) | `p99` (`ms`) | Max `VectorSearch` RAM (`MiB`) | Max Peak `tracemalloc` (`MiB`) | Max RSS Delta (`MiB`) | Mean `BnB` Fetches (`Ratio`, `N=500`) | 16-Thread Conc. Errs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `B0_DPR` | `125.7` | `248.1` | `320.9` | `7.65` | `55.08` | `54.68` | `2,415.8` (`23.42%`) | — |
| `B1_FTS_UNI` | `25.2` | `52.1` | `59.6` | `110.04` | `210.78` | `139.26` | `0.0` (`0.00%`) | — |
| `B2_FTS_TRI` | `60.1` | `158.3` | `212.3` | `7.65` | `54.82` | `71.92` | `0.0` (`0.00%`) | — |
| `B3_CS6` (incumbent) | `79.1` | `96.2` | `105.3` | `110.04` | `210.78` | `222.80` | `0.0` (`0.00%`) | — |
| `B5S_RAND_SEEDS` | `774.5` | `1,118.6` | `1,347.7` | `7.65` | `54.82` | `88.77` | `0.0` (`0.00%`) | — |
| `B5A_RAND_WALK` | `705.2` | `1,028.8` | `1,171.6` | `7.65` | `55.08` | `89.15` | `0.0` (`0.00%`) | — |
| **`M_PURE_0P`** | **`35.1`** | **`86.2`** | `107.0` | **`25.45`** | `75.91` | `110.79` | **`0.0` (`0.00%`)** | **`0`** |
| `AP1_NO_RABITQ_EXACT` | `27.6` | `75.2` | `111.3` | `57.27` | `202.95` | `141.02` | `0.0` (`0.00%`) | — |
| `AP2_1BIT_RABITQ` | `32.5` | `81.8` | `105.8` | `21.90` | `72.36` | `105.40` | `0.0` (`0.00%`) | — |
| `AP3_UNICODE61_LEX` | `31.6` | **`64.5`** | **`79.2`** | `26.05` | `252.19` | `188.97` | `0.0` (`0.00%`) | — |
| `AP4_NO_DIFFUSION` | `27.9` | `79.9` | `108.4` | `25.38` | `75.85` | `139.41` | `0.0` (`0.00%`) | — |
| `AP5_NO_TITLE_CIT` | `31.3` | `81.0` | `109.6` | `19.81` | `71.02` | `104.62` | `0.0` (`0.00%`) | — |
| `AP6_NO_PMI_CHAN` | `30.6` | `76.1` | `108.2` | `24.05` | `74.51` | `111.55` | `0.0` (`0.00%`) | — |
| `AP7_NO_NOISYOR` | `35.5` | `86.1` | `120.1` | `25.38` | `75.85` | `137.00` | `0.0` (`0.00%`) | — |
| `AP8_NO_MAXP` | `35.6` | `84.8` | `110.3` | `25.38` | `75.85` | `136.80` | `0.0` (`0.00%`) | — |

- **Paired Full-Corpus Indexing Speed Disclosure:** Paired CocoIndex `1.2.0` ingestion wall-clock times across the `37,570`-doc / `52,959`-chunk build were **not persisted inside `outputs/mdl-dag-eval/reports/*.json`** (the harness builds/opens cached DBs prior to measurement without emitting ingestion-time fields). The only persisted indexing-speed benchmark on disk is the `[0, 10)` `CS-5` clean-slate reference run (`12.39 s` across `563` docs / `628` chunks vs `HippoRAG 2,605.34 s`; [evaluation.md](../evaluation.md#2026-11-20-cs-5-live-silicon-evaluation--paired-cs-6-control-baseline)).
- **Cold-Open Memory Note (Resolved in `EVAL-3b` / `EVAL-3f`):** Earlier `AP3_UNICODE61_LEX` (`M_PURE_0P` LOO) measured `maxTM = 252.19 MiB` and `maxRSSd = 188.97 MiB` only because it opened legacy `incumbent_unicode61.db` (`spaCy` `tokens_json`/`entities_json` + missing persisted `yaw_stage1`, forcing full in-memory RaBitQ/CSR rebuild). On the clean `EVAL-3b` `candidate_unicode61.db` schema (`246.43 MiB` total disk footprint across 5 corpora), cold-open peak `tracemalloc` drops **`3.20x`** (`252.19 -> 78.73 MiB` for `Tier E` `M_FINAL_1B`; `72.41 MiB` for `M_PURE_U61_1B`; `79.00 MiB` on `Tiers B/C/D`), with `Max VectorSearch RAM = 24.73 MiB` on `Tier E` (`25.34–25.35 MiB` on `Tiers B/C/D` including `ERB`):

| `Tier E` System Arm (`N = 400`, `eval3f_tier_e_retrieval.json`) | `p50` (`ms`) | `p95` (`ms`) | `p99` (`ms`) | Max `VectorSearch` RAM (`MiB`) | Max Peak `tracemalloc` (`MiB`) | Max RSS Delta (`MiB`) | Mean `BnB` Ratio | 16-Thread Conc. Errs |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `B3_CS6` (incumbent) | `78.9` | `100.9` | `113.3` | `93.55` | `190.94` | `79.71` | `0.00%` | — |
| `B5S_RAND_SEEDS` | `771.3` | `1,125.2` | `1,471.8` | `6.88` | `55.08` | `84.38` | `0.00%` | — |
| `B5A_RAND_WALK` | `689.8` | `951.8` | `1,095.4` | `6.88` | `55.08` | `84.48` | `0.00%` | — |
| `M_PURE_U61_1B` | `21.5` | `37.0` | `46.2` | `21.95` | `72.41` | `103.84` | `0.00%` | — |
| **`M_FINAL_1B`** *(per-DS RAM `13.74/24.73/20.67/20.26`)* | **`22.1`** | **`38.8`** | **`50.6`** | **`24.73`** | **`78.73`** | `105.07` | **`0.00%`** | **`0`** |
| `AF1_NO_RABITQ_EXACT` | `18.3` | `33.0` | `45.3` | `59.03` | `343.31` | `140.69` | `0.00%` | — |
| `AF2_TRIGRAM_LEX` | `28.1` | `57.9` | `67.8` | `24.73` | `78.73` | `106.93` | `0.00%` | — |
| `AF3_NO_DIFFUSION` | `13.2` | `25.0` | `30.8` | `24.73` | `78.73` | `85.98` | `0.00%` | — |
| `AF4_NO_TITLE_CIT` | `15.7` | `27.5` | `30.5` | `16.34` | `67.55` | `96.43` | `0.00%` | — |
| `AF5_NO_PMI_CHAN` | `18.4` | `33.8` | `40.6` | `23.39` | `78.73` | `106.07` | `0.00%` | — |
| `AF6_NO_NOISYOR` | `22.2` | `36.8` | `41.8` | `24.79` | `78.73` | `107.08` | `0.00%` | — |
| `AF7_NO_MAXP` | `23.3` | `39.0` | `50.5` | `24.82` | `78.73` | `107.77` | `0.00%` | — |
| `AF8_NO_DECOMPOUND` | `22.8` | `38.3` | `51.0` | `24.73` | `78.73` | `106.72` | `0.00%` | — |
| `AF9_NO_TANGENT_STEER` | `20.3` | `36.0` | `44.4` | `21.91` | `72.37` | `104.67` | `0.00%` | — |
| `AF10_NO_DPP_1BIT` | `22.4` | `40.3` | `54.1` | `24.73` | `78.73` | `106.81` | `0.00%` | — |

---

## 7. Stratified Error Taxonomy & Post-Pivot Diagnostics

### 7.1 Complete 30-Case Tier B Failure Taxonomy (`tier_b_error_taxonomy.json`)

During `EVAL-2a`, inspecting all scorable Tier B misses under initial `M_FULL` identified `101` `Rank_Top5_Miss` vs `52` `Subgraph_Bridge_Miss` across `243` scorable queries and logged **all `30` stratified `Rank_Top5_Miss` cases** (`6` per dataset):

| # | Dataset | Question ID (`Stratum`) | `M_FULL` (`R5/AH5`) | `B3` / `B4` `R5` | `BnB` | `ms` | Question Summary |
| ---: | :--- | :--- | :---: | :---: | ---: | ---: | :--- |
| `1` | `2wikimultihopqa` | `0ce9a92008ed11ebbda7ac1f6bf848b6` | `0.50 / 0.00` | `1.00 / 1.00` | `2,470` | `540.8` | Movies `Inside The Room` & `Crude Set Drama` country of origin |
| `2` | `2wikimultihopqa` | `0dfe41f60bdc11eba7f7acde48001122` | `0.00 / 0.00` | `0.50 / 0.50` | `1,902` | `488.9` | Performer birth place of `Come Dance With Me (Song)` |
| `3` | `2wikimultihopqa` | `38da155f087511ebbd67ac1f6bf848b6` | `0.50 / 0.00` | `1.00 / 1.00` | `1,973` | `411.6` | Longevity comparison (`Sir Thomas Wheate` vs `Albert Édouard Le Brethon`) |
| `4` | `2wikimultihopqa` | `434dea7708d211ebbd95ac1f6bf848b6` | `0.50 / 0.00` | `1.00 / 1.00` | `2,634` | `490.7` | Longevity comparison (`Ludwig Elsbett` vs `Pamela Ann Rymer`) |
| `5` | `2wikimultihopqa` | `4a11106f08ea11ebbda7ac1f6bf848b6` | `0.50 / 0.00` | `1.00 / 1.00` | `2,950` | `537.6` | Birth-order comparison (`Marjatta Raita` vs `Bruce Coulter`) |
| `6` | `2wikimultihopqa` | `6b8402c40bdd11eba7f7acde48001122` | `0.00 / 0.00` | `0.00 / 0.50` | `2,333` | `519.1` | Child of director of film `An Event` |
| `7` | `musique` | `3hop1__635099_131926_87157` (`3-hop`) | `0.00 / 0.00` | `0.33 / 0.67` | `3,288` | `978.5` | Direction of flow of body of water by city where `Write This Down` formed |
| `8` | `musique` | `2hop__179322_83289` (`2-hop`) | `0.50 / 0.00` | `0.50 / 1.00` | `3,028` | `809.1` | Actor in `One Last Time` by performer of `Baby I` |
| `9` | `musique` | `3hop1__105837_160713_77246` (`3-hop`) | `0.33 / 0.00` | `0.33 / 0.67` | `2,433` | `838.1` | Word meaning of majority religion of area containing sub-region |
| `10` | `musique` | `3hop1__106845_5188_86687` (`3-hop`) | `0.33 / 0.00` | `0.67 / 0.67` | `2,697` | `867.8` | When country of top-ranking Warsaw Pact operatives was unified/founded |
| `11` | `musique` | `3hop2__127483_91678_10557` (`3-hop`) | `0.33 / 0.00` | `0.67 / 0.67` | `2,896` | `843.5` | Language `Auctor` originates from during era of crowned ruler |
| `12` | `musique` | `3hop2__87184_90327_76291` (`3-hop`) | `0.33 / 0.00` | `0.00 / 0.67` | `1,992` | `938.3` | When majority party in House of Representatives gained control of chamber |
| `13` | `hotpotqa` | `5a797bd95542994bb9457016` | `0.50 / 0.00` | `0.50 / 1.00` | `4,203` | `1,072.4` | Age of `Georgia Middleman` singing in 7th-most populated US city |
| `14` | `hotpotqa` | `5a7b3fc155429931da12ca3a` | `0.50 / 0.00` | `0.50 / 1.00` | `3,268` | `830.1` | Individual guiding criterion for `Wexner Graduate Fellowships` |
| `15` | `hotpotqa` | `5a83411655429966c78a6b5d` | `0.50 / 0.00` | `1.00 / 1.00` | `3,306` | `889.9` | Creator of NBC sitcom featuring `Johnny Pemberton` |
| `16` | `hotpotqa` | `5a84973e5542992a431d1a67` | `0.50 / 0.00` | `1.00 / 1.00` | `3,278` | `869.0` | Residence housing paintings by `Stockely Webster` |
| `17` | `hotpotqa` | `5ac3bd0c554299204fd21e21` | `0.50 / 0.00` | `1.00 / 1.00` | `5,389` | `1,002.9` | Shared profession of `Sam Mendes` and `Ruby Yang` |
| `18` | `hotpotqa` | `5ae0818e55429924de1b70e9` | `0.50 / 0.00` | `1.00 / 1.00` | `3,816` | `831.6` | Licensing body/venue associated with `Gleanntáin Ghlas' Ghaoth Dobhair` |
| `19` | `popqa` | `1872067` | `0.00 / 0.00` | `0.50 / 0.50` | `5,446` | `761.3` | Occupation of `Thomas Hinds` |
| `20` | `popqa` | `371747` | `0.00 / 0.00` | `0.50 / 0.50` | `4,639` | `711.7` | Occupation of `Gladwyn Jebb` |
| `21` | `popqa` | `1044970` | `0.50 / 0.00` | `0.50 / 0.50` | `4,485` | `697.3` | Occupation of `Mohd Sharkar Shamsudin` |
| `22` | `popqa` | `177542` | `0.50 / 0.00` | `0.50 / 0.50` | `3,806` | `731.1` | Occupation of `Claire Berlinski` |
| `23` | `popqa` | `1917994` | `0.50 / 0.00` | `0.50 / 0.50` | `4,536` | `705.9` | Occupation of `Sarah Dawn Finer` |
| `24` | `popqa` | `2311611` | `0.50 / 0.00` | `0.50 / 0.50` | `4,681` | `925.8` | Occupation of `Esquerita` |
| `25` | `enterprise_rag_bench` | `qst_0043` | `0.00 / 0.00` | `1.00 / 0.00` | `2,846` | `1,217.7` | Root cause & mitigation for `us-west-2` `p99` latency jump |
| `26` | `enterprise_rag_bench` | `qst_0187` | `0.00 / 0.00` | `1.00 / 0.00` | `3,783` | `1,622.8` | Automated platform actions on sustained harmful-output warnings |
| `27` | `enterprise_rag_bench` | `qst_0234` | `0.00 / 0.00` | `0.00 / 0.00` | `3,235` | `1,581.7` | Inference cost optimizer rollout checks |
| `28` | `enterprise_rag_bench` | `qst_0365` | `0.44 / 0.00` | `0.44 / 0.44` | `2,531` | `1,337.9` | Fast tier canary `p99` regression SLO targets & abort criteria |
| `29` | `enterprise_rag_bench` | `qst_0373` | `0.67 / 0.00` | `0.67 / 0.67` | `3,146` | `1,225.1` | Workflow for new Sev0/Sev1 incident |
| `30` | `enterprise_rag_bench` | `qst_0374` | `0.33 / 0.00` | `0.33 / 0.33` | `3,361` | `1,339.2` | Runtime 1.19 Dedicated canary TP watchdog mode & timeout |

Across these `30` cases, initial `M_FULL` failures fell into three pre-pivot pathologies: (a) length-3 random-walk bias and `node_post` dilution on `2Wiki`/`HotpotQA`/`MuSiQue` (`16` cases where `B3_CS6` scored higher than `M_FULL`), (b) heavy 1-bit `BnB` fetch storms (`3,806–5,446` vectors on `PopQA`) paired with occupation-hub drowning, and (c) `trigram` sub-identifier fragmentation on `enterprise_rag_bench` (`qst_0043`, `qst_0187`).

### 7.2 Post-Pivot `M_PURE_0P` Error Breakdown on Untouched Tier D (`N = 492`)

On untouched Tier D (`eval3_tier_d_retrieval.json`), `M_PURE_0P` eliminates almost all zero-recall failures on multi-hop/single-hop benchmarks (`R@5 == 0`: `2Wiki = 0/100`, `HotpotQA = 0/100`, `PopQA = 0/100`, `ERB = 3/92`, `MuSiQue = 9/100`). Remaining `All-Hops@5 < 1` errors concentrate in three clear regimes:

1. **Deep 3-/4-Hop Compositional Chains Without Surface Title Overlap (`MuSiQue`: `66/100` `AH@5 < 1`, `4-hop AH@5 = 0.0000`, `3-hop AH@5 = 0.1034`):** Without LLM-extracted `(s, p, o)` triples or latent Hop-1 entity reformulation at query time, pure geometric restart ($\rho_{\text{stop}} = 0.5$, weight $0.5^4 = 0.0625$ at hop 4) decays exponentially when intermediate hop entities are paraphrased rather than linked by exact Wikipedia titles or 1-bit PMI collocations. Even so, `M_PURE_0P` still delivers `+24.0 pp` `MuSiQue AH@5` over `B3_CS6` (`0.3400 vs 0.1000`).
2. **Same-Topic PopQA Occupation Distractors (`PopQA`: `85/100` `AH@5 < 1`, `R@5 = 0.5750`, `AH@5 = 0.1500`):** Biographical queries in `PopQA` require returning both entity biography chunks; diffusion pulls structurally related occupation/nationality neighbors slightly ahead of a second biographical passage (`0` complete misses `R@5 == 0`, while `M_PURE_0P` beats `B3_CS6` by `+0.1100 R@5` and `+0.1500 AH@5`).
3. **Trigram Substring Noise vs `unicode61` Word-Token Precision (`AP3_UNICODE61_LEX`):** On Tier D (`N = 492`), keeping `unicode61` instead of `trigram` improved `Recall@5` on `17` queries vs `5` (`+0.0088`), notably lifting `MuSiQue 4-hop AH@5` (`0.0000 -> 0.0769`, `R@5 0.3654 -> 0.5000`), `2Wiki` (`0.9475 -> 0.9600`), and `HotpotQA` (`0.9600 -> 0.9700`), and on Tier C (`N = 490`) beating `trigram` `18` to `4` (including `ERB 0.9257 vs 0.8916`).

### 7.3 `M_FINAL_1B` Error Breakdown on Untouched `Tier E` (`N = 400`) & Post-Freeze Tiers C/D

On untouched `Tier E` (`eval3f_tier_e_retrieval.json`, `N = 400`), `M_FINAL_1B` slashes complete zero-recall misses (`Recall@5 == 0`) by **`10.5x`** compared with `B3_CS6` (`2 / 400` total [`0` `2Wiki`, `2` `MuSiQue`, `0` `HotpotQA`, `0` `PopQA`] vs `21 / 400` [`0 / 9 / 4 / 8`]), improving on even `M_PURE_U61_1B` (`3 / 400`):

1. **Multi-Hop `MuSiQue` Hop Strata (`48 / 36 / 16`):** `M_FINAL_1B` reaches `2-hop = 0.8438 / 0.7083` (`R@5 / AH@5`, vs `B3_CS6 0.6875 / 0.4167`), `3-hop = 0.6667 / 0.1944` (`B3_CS6 0.4259 / 0.0000`, `M_PURE_U61_1B 0.6296 / 0.1111`), and `4-hop = 0.4531 / 0.0000` (`B3_CS6 0.3125 / 0.0000`; `0.5000 / 0.1538` on post-freeze `Tier D` `eval3f_tier_d_mfinal.json` where `R@5 == 0` is `7 / 492`, and `12 / 490` on `Tier C`, both retaining `ERB R@5 = 0.9152`). Remaining `4-hop` `AH@5` deficits stem from unbridged 4th-hop paraphrases below the top-5 slot ceiling (`Tier E` `All-Hops@10 = 0.7775` vs `AH@5 = 0.5850`) where `EVAL-6` query-time scout hops act.
2. **`PopQA` Entity-Biography vs Tier-1 QA (`Tier E` `n = 200`):** Standalone `PopQA` `R@5 = 0.6050` (`AH@5 = 0.2100`, `0` complete misses) puts `M_FINAL_1B` ahead of `B3_CS6` (`0.4600 / 0.0000`, `8` misses), while on multi-hop QA (`n = 150` across `2Wiki/MuSiQue/HotpotQA`) `M_FINAL_1B` lifts `dspy.ReAct` by `+6.00 pp EM` (`0.5400 vs 0.4800`) and `+8.92 pp F1` (`0.6645 vs 0.5753`) alongside `+6.50 pp` overall `F1` (`p_perm = 0.0022`).

---

## 8. Threats to Validity, Negative Results & Explicit Non-Claims

1. **100% English Real-Corpus Scope:** All five full-corpus datasets (`37,570` docs) are English. Non-English capabilities (`DE` decompounding / `ß <-> ss` casefolding, `ZH` overlapping CJK bigrams, `FI` inflections) are verified solely via real-SQLite/FTS5 lab unit tests in ``tests/test_mdl_dag_lab.py`` (`exp/mdl-dag-ablation@5b58ada`), NOT live non-English QA corpora. Choosing `unicode61` + `.casefold()` + overlapping CJK bigrams supports CJK 2-char phrases cleanly without `trigram`'s `5.89x` index bloat, while sacrificing arbitrary `<3`-char non-CJK infix substring lookups that lack word boundaries.
2. **Tier C / Tier D / Tier E Confirmatory Boundary:** Only `M_WIN_0P` consumed `Tier C` (`4289edbc...`) as an untouched holdout (and failed `P4`/`P5`), only `M_PURE_0P` consumed `Tier D` (`0ae4fa1a...`) as an untouched holdout (`P5 = FAIL`), and **`M_FINAL_1B`** consumed newly minted disjoint **`Tier E`** (`fe1449a2...`, `[300, 400)`, `N = 400` + `200` reject probes + `n = 200` paired QA) **once** after freeze. All `EVAL-3b` (`M_PURE_U61_1B`) and `EVAL-3f` (`M_FINAL_1B`) evaluations on `Tier C` (`eval3b_tier_c_*.json`, `eval3f_tier_c_mfinal.json`) and `Tier D` (`eval3b_tier_d_*.json`, `eval3f_tier_d_mfinal.json`) are non-confirmatory comparison benchmarks; `enterprise_rag_bench` is absent from `Tier E` because its `150` queries were exhausted across `Tiers A–D` (`M_FINAL_1B` achieves `ERB = 0.9152 / 0.8556` on `Tier C` and `0.9152 / 0.8478` on `Tier D`).
3. **`P3` Statistical Power (`Tier D n = 100` vs `Tier E n = 200`):** On Tier D `n = 100` (`25` per public dataset), `M_PURE_0P` outperformed `B3_CS6` by `+5.00 pp EM` and `+3.79 pp F1` (`P3 = PASS`) whereas `M_PURE_U61_1B` scored `-0.0100 EM` / `-0.0115 F1` (`P3 = FAIL`) partly due to `+7.00 pp EM` sampling variance on `B3_CS6` under `gemma-4-e4b` (`temperature = 1.0`). Expanding to `Tier E n = 200` (`50/dataset`, `400` episodes, `0` errors) yielded unambiguous **`P3 = PASS`** (`EM +0.0400`, `F1 +0.0650 [+0.0240, +0.1088]`, `p_perm = 0.0022`). Uncompiled zero-shot `dspy.ReAct` leaves PopQA headroom (`EM = 0.1800`) pending `EVAL-6` Graph Scout and `dspy.GEPA` compilation.
4. **Falsified / Negative Results Recorded Verbatim:** Early `M_FULL` unnormalized linear path-marginal Gibbs (`eval2c_mfull_loo.json`: `P5 = 2/9`, `Rej = 0.6227`); Marchenko–Pastur bulk whitening (`H1_MP_WHITEN`: `0.6492`, `p95 = 2,017.8 ms`); scalar multi-branch NNLS decode (`H2_MULTI_BRANCH`: `PopQA 0.4300`, `p95 = 1,631.5 ms`); empirical quantile tails (`H4_EMPIRICAL_TAIL`: `0.5868`); `M_WIN_0P` finite-horizon Entity-HMM + Gibbs BMA (`eval3_tier_c_retrieval.json`: `Rej = 0.6432 < 0.70`, `P5 = 7/21`); `trigram` beating `unicode61` (`AU3`/`A1B2` falsified across Tiers D/C/E); `M_PURE_U61_1B` on `n = 100` `P3` (`eval3b_tier_d_1bit_qa.json`: `-1.00 pp EM`); `E1B_RESIDUAL_COV` (`0.7393`, `MHAH5 = 0.5400`); `E2A_DYN_RESID_UNIT` / `E2B_DYN_RESID_PROJ` (`~7x p95`, `-4.7 / -6.0 pp MH AH@5`); `E3B_SEM_BONF` / `E3C_SEM_MKNN` (`-2.45 / -2.53 pp`, `2.8x–3.0x maxTM`); `E3A_NB_HASHIMOTO` paired with `F1_TANGENT_STEER` (`I(F1, E3A) = -1.089 pp` gold 2-cycle suppression); `F4A_BRIDGE_PURE` (`0.7792`, `p = 0.0486`); `F3B_LOCAL_ENT_REJECT` (inverted calibration `Rej 0.5161 < 0.7382`); and strict univariate `P5` on `Tier E` (`5/10 PASS`: `AF1` `float32` gains `+0.0225 AH@5` at `2.39x` RAM / `4.36x` `maxTM`).

---

## 9. Reproducibility & Explicit EVAL-5 Decision Options

### 9.1 Reproducibility Commands (`exp/mdl-dag-ablation` @ `5b58ada`)

```powershell
# 1. Verify full isolated lab test suite + architecture containment gates (434 passed on exp/mdl-dag-ablation@5b58ada)
git checkout exp/mdl-dag-ablation
uv run --offline --locked ty check
uv run --offline --locked ruff check
uv run --offline --locked pytest -o addopts=""

# 2. EVAL-3b shootout & 1-bit Tier D / C benchmarks + n=100 QA (isolated --cache-dir outputs/mdl-dag-eval/cache_eval3b)
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier d --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json --confirm-sealed-tier-d --cache-dir outputs/mdl-dag-eval/cache_eval3b --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_0P M_PURE_U61 AP1_NO_RABITQ_EXACT AP2_1BIT_RABITQ AP3_UNICODE61_LEX AP4_NO_DIFFUSION AP5_NO_TITLE_CIT AP6_NO_PMI_CHAN AP7_NO_NOISYOR AP8_NO_MAXP AU1_NO_RABITQ_EXACT AU2_1BIT_RABITQ AU3_TRIGRAM_LEX AU4_NO_DIFFUSION AU5_NO_TITLE_CIT AU6_NO_PMI_CHAN AU7_NO_NOISYOR AU8_NO_MAXP AU9_NO_DECOMPOUND --output outputs/mdl-dag-eval/reports/eval3b_tier_d_shootout.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier c --confirm-sealed-tier-c --cache-dir outputs/mdl-dag-eval/cache_eval3b --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_0P M_PURE_U61 AP1_NO_RABITQ_EXACT AP2_1BIT_RABITQ AP3_UNICODE61_LEX AP4_NO_DIFFUSION AP5_NO_TITLE_CIT AP6_NO_PMI_CHAN AP7_NO_NOISYOR AP8_NO_MAXP AU1_NO_RABITQ_EXACT AU2_1BIT_RABITQ AU3_TRIGRAM_LEX AU4_NO_DIFFUSION AU5_NO_TITLE_CIT AU6_NO_PMI_CHAN AU7_NO_NOISYOR AU8_NO_MAXP AU9_NO_DECOMPOUND --output outputs/mdl-dag-eval/reports/eval3b_tier_c_shootout.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier d --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json --confirm-sealed-tier-d --cache-dir outputs/mdl-dag-eval/cache_eval3b --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_U61_1B A1B1_NO_RABITQ_EXACT A1B2_TRIGRAM_LEX A1B3_NO_DIFFUSION A1B4_NO_TITLE_CIT A1B5_NO_PMI_CHAN A1B6_NO_NOISYOR A1B7_NO_MAXP A1B8_NO_DECOMPOUND --output outputs/mdl-dag-eval/reports/eval3b_tier_d_1bit.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier c --confirm-sealed-tier-c --cache-dir outputs/mdl-dag-eval/cache_eval3b --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_U61_1B A1B1_NO_RABITQ_EXACT A1B2_TRIGRAM_LEX A1B3_NO_DIFFUSION A1B4_NO_TITLE_CIT A1B5_NO_PMI_CHAN A1B6_NO_NOISYOR A1B7_NO_MAXP A1B8_NO_DECOMPOUND --output outputs/mdl-dag-eval/reports/eval3b_tier_c_1bit.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier d --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json --confirm-sealed-tier-d --cache-dir outputs/mdl-dag-eval/cache_eval3b --include-qa --qa-per-dataset 25 --arms B3_CS6 M_PURE_U61_1B --output outputs/mdl-dag-eval/reports/eval3b_tier_d_1bit_qa.json

# 3. EVAL-3c / 3d / 3e Tier B waves, 16-arm factorial lattice & static-CSR/Beta-LUT M_FINAL_1B check (isolated --cache-dir outputs/mdl-dag-eval/cache_eval3c)
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier b --cache-dir outputs/mdl-dag-eval/cache_eval3c --arms M_PURE_U61_1B E1A_DPP_1BIT E1B_RESIDUAL_COV E2A_DYN_RESID_UNIT E2B_DYN_RESID_PROJ E3A_NB_HASHIMOTO E3B_SEM_BONF E3C_SEM_MKNN --output outputs/mdl-dag-eval/reports/eval3c_tier_b_wave1.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier b --cache-dir outputs/mdl-dag-eval/cache_eval3c --arms M_PURE_U61_1B F1_TANGENT_STEER F2_WORMHOLE_COND F3A_LOCAL_ENT_RERANK F3B_LOCAL_ENT_REJECT F4A_BRIDGE_PURE F4B_BRIDGE_NOISYOR --output outputs/mdl-dag-eval/reports/eval3d_tier_b_wave2.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier b --cache-dir outputs/mdl-dag-eval/cache_eval3c --arms M_PURE_U61_1B F1_TANGENT_STEER E1A_DPP_1BIT E3A_NB_HASHIMOTO F4B_BRIDGE_NOISYOR C2_F1_E1A C2_F1_E3A C2_F1_F4B C2_E1A_E3A C2_E1A_F4B C2_E3A_F4B C3_F1_E1A_E3A C3_F1_E1A_F4B C3_F1_E3A_F4B C3_E1A_E3A_F4B C4_F1_E1A_E3A_F4B --output outputs/mdl-dag-eval/reports/eval3e_tier_b_lattice.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier b --cache-dir outputs/mdl-dag-eval/cache_eval3c --arms M_FINAL_1B --output outputs/mdl-dag-eval/reports/eval3e_tier_b_mfinal_opt.json

# 4. EVAL-3f untouched Tier E confirmatory retrieval + n=200 paired P3 QA + post-freeze Tiers C/D (isolated --cache-dir outputs/mdl-dag-eval/cache_eval3f)
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier e --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_e_confirmatory.json --confirm-sealed-tier-e --cache-dir outputs/mdl-dag-eval/cache_eval3f --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_U61_1B M_FINAL_1B AF1_NO_RABITQ_EXACT AF2_TRIGRAM_LEX AF3_NO_DIFFUSION AF4_NO_TITLE_CIT AF5_NO_PMI_CHAN AF6_NO_NOISYOR AF7_NO_MAXP AF8_NO_DECOMPOUND AF9_NO_TANGENT_STEER AF10_NO_DPP_1BIT --output outputs/mdl-dag-eval/reports/eval3f_tier_e_retrieval.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier e --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_e_confirmatory.json --confirm-sealed-tier-e --cache-dir outputs/mdl-dag-eval/cache_eval3f --include-qa --qa-per-dataset 50 --arms B3_CS6 M_FINAL_1B --output outputs/mdl-dag-eval/reports/eval3f_tier_e_qa_n200.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier c --confirm-sealed-tier-c --cache-dir outputs/mdl-dag-eval/cache_eval3f --arms M_FINAL_1B --output outputs/mdl-dag-eval/reports/eval3f_tier_c_mfinal.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier d --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json --confirm-sealed-tier-d --cache-dir outputs/mdl-dag-eval/cache_eval3f --arms M_FINAL_1B --output outputs/mdl-dag-eval/reports/eval3f_tier_d_mfinal.json

# 5. Offline IEEE <f2> roundtrip parity provenance: outputs/mdl-dag-eval/reports/eval3f_f16_parity.json
#    Produced by an offline <f2> (float16 -> float32) roundtrip parity script (not an eval_harness CLI flag) over the 5 full candidate_unicode61.db databases (37,570 docs / 52,959 chunks) + Tier E (4 public datasets, N = 400 scorable + 200 reject probes).
```

### 9.2 Approved `EVAL-5` Production Cutover Specification & Post-Merge `EVAL-6` Scope

With **`M_FINAL_1B`** clearing every untouched confirmatory retrieval and QA gate on `Tier E` (`P0–P4 = PASS`: `H4 = 0.8006`, `MH AH@5 = 0.7100 [+0.3033]`, `QA EM = 0.4500`, `QA F1 = 0.6219 [+0.0650, p = 0.0022]`, `RAM = 24.73 MiB` [`3.78x` reduction], `p50/p95 = 22.1 / 38.8 ms`, `Rej AUROC = 0.7068`) plus `float16` parity (`4-DS H4 diff = 0.0000`, `MH AH@5 diff = 0.0000`), **`EVAL-5`** is explicitly approved by the user on `2026-11-22` ([epic-3-performance.md#decisions](../epic-3-performance.md#decisions), waiving univariate `P5 = 5/10`) to execute the following clean-slate cutover onto `dev` (implementing **only** what was verified, with zero `M_PURE_0P` Path-B compromises and zero legacy shims):

1. **Frozen Production Algorithm (`M_FINAL_1B`):**
   - `1-bit Extended Rotated SoA RaBitQ` + `Tangent-Steered Cyclic Citation Diffusion` (forward/reverse title-ref + self-loop-cancelled 1-bit PMI channels, conditional Noisy-OR edge weighting, static forward/transpose CSR + `16,384`-bin `z = sqrt(-log1p(-c^2))` Beta-tail LUT [`max abs err < 1e-6`], zero Class-1 tunable retrieval hyperparameters) + `Rank-1 Cholesky Greedy L-Ensemble 1-bit DPP` + `MaxP Deferral`.
2. **Plain SQLite Persistent Schema (`yaw.db`, Zero `sqlite-vec` / `vec0`):**
   - **`chunks`:** Plain SQLite ROWID table persisting `<f2>` (IEEE `float16`) vector BLOBs + immutable pointer metadata (`rel_path`, `source_hash`, `start_line`, `end_line`, `char_start`, `char_end`, `ast_island_start`, `ast_island_end`, `title`) — **zero raw chunk text**, **zero `tokens_json` / `entities_json`**, **zero `vec0` virtual tables**.
   - **`chunk_fts` + `chunk_vocab`:** Contentless `fts5(title, body, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 1')` fed via `.casefold()` + overlapping CJK bigrams, paired with `chunk_vocab` (`fts5vocab(chunk_fts, 'row')`) for query-time prefix/Koehn–Knight decompounding (`_run_decomp_for_word`).
   - **`rabitq_index`:** Single-row persisted 1-bit SoA RaBitQ `codes_blob`, `ip_norm_blob`, centroid `mean_blob`, `d_eff_shannon`, and chunk/embedding identity fingerprints (`n_chunks`, `dim`, `embedding_model`, `task_mode`, `vec_sha256`, `source_sha256`).
   - **`citation_graph`:** Persisted `doc_titles` + compressed bipartite CSR adjacency (`fwd`/`rev` title-ref CSR, doc-owned chunk CSR `dc`, chunk<->PMI-phrase bipartite CSR + `pmi_terms`, and `prop_nouns`) used to build static forward/transpose diffusion operators.
3. **Self-Documenting Naming Migration (`yaw/**`):**
   - Replace experimental lab names and legacy retrieval types cleanly in place with idiomatic production symbols: **`RetrievalIndex`**, **`CitationGraph`**, **`DiffusionOperators`**, **`SearchResult`**, **`search_index`**, and **`expand_citation_neighbors`**.
4. **Atomic Dependency & Dead-Code Purge (`EVAL-5` Same-Commit Deletion):**
   - Remove **`sqlite-vec` / `vec0`**, **`optuna`**, and **`spacy` / `en-core-web-sm`** cleanly from `pyproject.toml`, `uv.lock`, `yaw/**`, `FOUNDATIONAL.md`, `.agents/rules/*.md`, and `tests/**`.
   - Delete superseded tuning infrastructure atomically: **`SearchParams`**, **`resolve_retrieval_params`**, **`optimize_optuna`**, `yaw optimize optuna` CLI subcommand, and all Optuna watchdog/DB telemetry hooks across `yaw/**`, `scripts/watchdog.py`, and `tests/**`.
5. **Post-Merge `EVAL-6` Scope (Strictly Deferred Until After `EVAL-5` Lands):**
   - **`SearchResult.completeness`:** Native MDL / Bayes-factor completeness indicator exposed on `SearchResult` once the `EVAL-5` schema is live.
   - **Lazy `.yaw/wiki/` Graph Scout:** Bounded zero-LLM-bloat graph scout layer built cleanly atop `RetrievalIndex` and `CitationGraph` after `EVAL-5` gates are green on `dev`.
