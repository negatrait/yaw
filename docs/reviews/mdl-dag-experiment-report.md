# Benchmark-Gated Evaluation of Parameter-Free Converged Citation Diffusion (`M_PURE_0P`) and Ablation History (`EVAL-0`–`EVAL-3`)

**Status:** Confirmatory Experiments Complete — **Conditional Promotion Decision Required Before `EVAL-5`** (`0` edits to `yaw/**`, `pyproject.toml`, `yaw.toml`, [FOUNDATIONAL.md](../../FOUNDATIONAL.md), or [AGENTS.md](../../AGENTS.md) on `dev`)

| Relation | Reference |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../../FOUNDATIONAL.md), [AGENTS.md](../../AGENTS.md), [epic-3-performance.md](../epic-3-performance.md#decisions) |
| Rules | [docs.md](../../.agents/rules/docs.md), [graph-storage.md](../../.agents/rules/graph-storage.md), [tests.md](../../.agents/rules/tests.md), [canonical-files.md](../../.agents/rules/canonical-files.md) |
| Specification & Ledger | [mdl-dag-sprint.md](mdl-dag-sprint.md) (`Rev 4`) |
| Evaluation Ledgers | [evaluation.md](../evaluation.md), [evaluation-graph-storage.md](../evaluation-graph-storage.md) |
| Peer Specs | [bare-bones-replacement-sprint.md](bare-bones-replacement-sprint.md), [epic-3-fts-vec-graph-experiment.md](epic-3-fts-vec-graph-experiment.md), [substate-manifold-shootout.md](substate-manifold-shootout.md) |
| Code Provenance | Pinned `dev` baseline: `3cd3952` (`66e9a9e` `EVAL-0`); isolated lab: branch `exp/mdl-dag-ablation` @ commit `032fe2d` |

---

## 1. Abstract & Executive Gate Verdict

We evaluate whether a zero-LLM, zero-tunable-hyperparameter graph-diffusion retriever (**`M_PURE_0P`**) built over committed SQLite chunks (`37,570` documents / `52,959` chunks across 5 benchmarks) outperforms YAW's unmodified clean-slate baseline (`B3_CS6`, [retrieval.py](../../yaw/retrieval.py) at `3cd3952`) under preregistered confirmatory gates `P0`–`P5` while removing all `optuna` tuning and `spaCy` runtime dependencies.

An intermediate de-parameterized hybrid (`M_WIN_0P`, fusing finite-horizon `L <= 3` simple-path Gibbs with 3-hop Entity-HMM BMA) passed Tier B discovery and Tier C retrieval (`Pool R@5 = 0.8123` vs `0.6547`), yet **failed sealed Tier C** on reject calibration (`P4`: `Reject AUROC = 0.6432 < 0.70 < top1_cosine 0.6804`) and component necessity (`P5`: `7/21 PASS`). Stripping every remaining depth, budget, and scalar-weight heuristic yielded **`M_PURE_0P`**: a single factored $O(\text{nnz})$ operator combining directed forward title ownership, exact Bayesian reverse citations, and self-loop-cancelled 1-bit PMI co-occurrence under converged geometric-restart ($\rho_{\text{stop}} = 0.5$) diffusion on a full-corpus 4-bit RaBitQ field.

On the **untouched disjoint confirmatory split (`Tier D`, `N = 492` retrieval-scorable queries, `200` reject labels, `n = 100` paired `gemma-4-e4b` `dspy.ReAct` QA queries)**:

- **`M_PURE_0P` clears `P0`, `P1`, `P2`, `P3`, and `P4`:** `Pool Recall@5 = 0.8010` vs `B3_CS6 0.6627` (`+0.1382`, $p_{\text{Holm}} = 0.0004$), `Multi-Hop All-Hops@5 = 0.7033` vs `0.4233` (`+0.2800`, $p_{\text{Holm}} = 0.0004$), `Reject AUROC = 0.7976` vs `top1_cosine 0.7387` (`0` masked leaks), `VectorSearch RAM = 25.45 MiB` (`4.32x` below `B3_CS6 110.04 MiB`), `p50/95 = 35.1 / 86.2 ms`, `0` 16-thread concurrency errors, and paired `dspy.ReAct` QA `EM = 0.3800` vs `0.3300` (`+0.0500`) / `F1 = 0.5615` vs `0.5235` (`+0.0379`).
- **Overall Gate Verdict (`P5 = FAIL`, `5/8 PASS`):** While 5 core mechanisms pass LOO (`AP1` 4-bit RaBitQ `2.25x` RAM, `AP4` diffusion `+0.0716 R@5 / +0.1967 MH AH@5`, `AP5` title citations `+0.0632 / +0.1367`, `AP7` conditional Noisy-OR `+0.0152 / +0.0533`, `AP8` MaxP deferral `+0.0080 R@5`), **`AP3_UNICODE61_LEX`** (incumbent `unicode61` FTS5 tokenizer) outperformed `trigram` tokenization on both Tier D (`0.8098 vs 0.8010 R@5`, `-0.0088 [-0.0160, -0.0019]`, `p95 = 64.5 ms vs 86.2 ms`) and Tier C comparison (`0.8039 vs 0.7913 R@5`, `p95 = 63.7 ms vs 89.0 ms`). Additionally `AP6_NO_PMI_CHAN` and `AP2_1BIT_RABITQ` missed strict Tier D 95% CIs. Conditional user sign-off is required before `EVAL-5` (§9).

### Executive Gate Matrix (`P0`–`P5`)

| Gate | Requirement Summary | `M_WIN_0P` *(Tier C, Superseded)* | `M_PURE_0P` *(Tier D, Untouched)* | `M_PURE_0P` *(Tier C Comparison)* |
| :--- | :--- | :---: | :---: | :---: |
| **`P0`** | `0` Class-1 tunable retrieval hyperparameters | **PASS** *(0 scalars; kept `horizon=3`, `L<=3`)* | **PASS** *(0 scalars, 0 depth/budget caps)* | **PASS** *(0 scalars, 0 depth/budget caps)* |
| **`P1`** | Beat `B5S` & `B5A` on `R@5` & `AH@5` ($p_{\text{Holm}} < 0.01$, CI $> 0$) | **PASS** (`p_holm = 0.0008`) | **PASS** (`p_holm = 0.0008`) | **PASS** (`p_holm = 0.0008`) |
| **`P2`** | `MH AH@5` $\Delta \ge +0.030$, `macro R@5` $\Delta \ge +0.020$, non-inf | **PASS** (`+0.2800` / `+0.1576`) | **PASS** (`+0.2800` / `+0.1382`, all DS $>0$) | **PASS** (`+0.2700` / `+0.1366`) |
| **`P3`** | Paired `n=100` `dspy.ReAct` QA non-regression vs `B3_CS6` | `NOT_RUN` | **PASS** (`EM +0.0500`, `F1 +0.0379`, `0` errs) | `NOT_RUN` |
| **`P4`** | RAM $\le$ `B3_CS6`, `p95 <= lat_cap`, `0` conc errs, `Rej >= 0.70` & $>$ `top1` | **FAIL** (`Rej 0.6432 < 0.6804`) | **PASS** (`25.45 MiB`, `86.2 ms`, `0.7976 > 0.7387`) | **PASS** (`26.02 MiB`, `89.0 ms`, `0.7731 > 0.6807`) |
| **`P5`** | Every component justified by LOO (Occam's razor) | **FAIL** (`7/21 PASS`) | **FAIL** (`5/8 PASS`; `AP3` > `trigram`) | **FAIL** (`5/8 PASS`; `AP3` > `trigram`) |
| **Overall** | Strict conjunction (`P0..P5`) | **REJECT / SUPERSEDED** | **PARTIAL (`P0–P4 PASS`, `P5 FAIL`)** | **PARTIAL (`P0–P4 PASS`, `P5 FAIL`)** |

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

### 2.3 Gate `P0` Zero-Hyperparameter Audit

`M_PURE_0P` eliminates every Class-1 tunable scalar from the system (`0` scoring weights, `0` blend weights, `0` column weights, `0` clone Jaccard thresholds, `0` Wave/Hop/subgraph caps, `0` Optuna parameters). Only Class-2 canonical identities ($\rho_{\text{stop}} = 0.5$, $\ge 1$-bit PMI $2\,\text{df}_1\text{df}_2$, Jeffreys $+0.5$), FTS5 trigram structural minima, and numerical/hardware constants (`rabitq_bits=4`, `_RABITQ_SEED=0x594157`, $\ell_\infty = 10^{-9}$) remain.

---

## 3. Experimental Setup, Paired Indexes & Split Provenance

### 3.1 Environment, Software Stack & Live Models

- **Host & OS:** `Windows-11-10.0.26200-SP0`, 24 logical CPUs, Python `3.14.9` (`uv --offline --locked`).
- **Locked Dependencies:** `cocoindex==1.2.0`, SQLite `3.53.1`, `sqlite-vec==0.1.9`, `scipy==1.18.1`, `numpy==2.5.4`, `dspy==3.5.0` (control baseline `B3_CS6` calls unmodified `3cd3952` [retrieval.py](../../yaw/retrieval.py) + `optuna==5.0.0` / `spacy==3.8.18`).
- **Local Models (`http://127.0.0.1:8888/v1`):** `embeddinggemma-2` (300M parameters, `D = 768`) for all paired corpus/query embeddings; `gemma-4-e4b` (`temperature = 1.0`, zero-shot uncompiled `dspy.ReAct` with 4 read-only tools) for `P3` confirmatory QA.

### 3.2 Full-Corpus Paired Indexes (`37,570` `.md` Docs / `52,959` Chunks)

Paired single-App CocoIndex `1.2.0` ingestion shared `@coco.fn(memo=True)` `_embed_chunk` across both targets so `incumbent_unicode61.db` (`tokens_json`/`entities_json` via `spaCy` for `B3_CS6`) and `candidate_trigram.db` (zero `spaCy`, `yaw_stage1` 4-bit AoS RaBitQ + compressed `entity_graph_blob`, `0` sidecar files) hold bit-identical chunk spans:

| Corpus | Unique `.md` Docs | Chunks (`vec0`) | `incumbent_unicode61.db` (`MiB` / `MB`) | `candidate_trigram.db` (`MiB` / `MB`) |
| :--- | ---: | ---: | :---: | :---: |
| `2wikimultihopqa` | `6,119` | `6,980` | `32.98` / `34.59` | `55.97` / `58.69` |
| `musique` | `11,656` | `12,512` | `62.95` / `66.00` | `108.78` / `114.06` |
| `hotpotqa` | `9,811` | `10,785` | `55.15` / `57.83` | `97.44` / `102.17` |
| `popqa` | `8,676` | `9,880` | `49.65` / `52.06` | `83.18` / `87.22` |
| `enterprise_rag_bench` | `1,308` | `12,802` | `73.73` / `77.32` | `142.05` / `148.95` |
| **Total File Size (`os.stat`)** | **`37,570`** | **`52,959`** | **`274.46` / `287.80`** | **`487.42` / `511.09`** |

#### Exact SQLite `dbstat` Component Allocation Across All 5 Full-Corpus Paired DBs

| SQLite Component (`dbstat` Sum) | `incumbent_unicode61.db` (`MiB`) | `candidate_trigram.db` (`MiB`) | Schema / Tokenizer Attribution |
| :--- | :---: | :---: | :--- |
| `chunks_vector_chunks00` | `162.23` | `162.23` | Bit-identical shared `768d` embeddings (`vec0`) |
| `chunks_auxiliary` | `62.52` | `6.65` | Incumbent carries `spaCy` `tokens_json`/`entities_json` (`-55.87 MiB` in candidate) |
| `chunk_fts_data` (`chunk_fts_*` total) | `45.76` (`47.12`) | `256.59` (`258.18`) | Pure FTS5 payload: **`5.61x`** larger for `trigram` vs `unicode61` |
| `yaw_stage1` | `0.00` | `31.92` | Persisted 4-bit AoS RaBitQ + compressed `entity_graph_blob` (absent in incumbent DB) |
| `chunk_vocab` + remaining metadata | `1.63` | `1.70` | Virtual `fts5vocab` reads `chunk_fts` dynamically (`0.00 MiB`); `rowids`/`chunks`/`info` |
| **Total `dbstat` Allocated Pages (`File Size`)** | **`273.57` (`274.46`)** | **`460.69` (`487.42`)** | Unconfounded `spaCy`-free `unicode61 + yaw_stage1` estimated `~250.5 MiB` `dbstat` *(unbuilt/untested)* |

### 3.3 Split Manifests, Reject Probes & Statistical Protocol

1. **Disjoint Immutable Split Manifests (`0` question-ID overlap enforced automatically by `load_and_verify_splits`):**
   - `outputs/mdl-dag-eval/manifests/splits_v1.json`:
     - **Tier A (`tier_a_anchor`, `tier_a_sha256: e8c848e2ba0d4158c416ec0fb013fa01b9bee320383564bc79b1e594af2c361d`):** Offsets `[0, 10)` across 4 public datasets (`total=40`, `scorable=40`), anchoring `CS-5` continuity.
     - **Tier B (`tier_b_dev`, `tier_b_sha256: 14adc8c1293fc71805a38b815a203cad302270120b4ff6fa16f0b439b0e23814`):** Offsets `[30, 80)` across 4 public datasets (`200`; `musique` `2h: 21, 3h: 17, 4h: 12`) + `erb train[0:50]` (`43` scorable + `1` `"High Level"` empty-gold + `6` `"Info Not Found"`) -> `250` total (`243` retrieval-scorable) + `50` `musique[30:80)` paired hop-1 bridge-masked probes (`150` reject labels total).
     - **Tier C (`tier_c_sealed_holdout`, `tier_c_sha256: 4289edbc286d190e1125852b4888cf86fe2da4389684441d4ee714b2f80e0fb5`):** Offsets `[100, 200)` (`400`; `musique` `2h: 55, 3h: 28, 4h: 17`) + `erb val[150:250]` (`90` scorable + `4` `"High Level"` empty-gold + `6` `"Info Not Found"`) -> `500` total (`490` retrieval-scorable) + `50` `musique[100:150)` bridge-masked probes (`200` reject labels total). Consumed cleanly once for `M_WIN_0P`; subsequently used as a post-freeze simplification comparison benchmark for `M_PURE_0P` (NOT an untouched holdout for `M_PURE_0P`).
   - `outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json`:
     - **Tier D (`tier_d_confirmatory`, `tier_d_sha256: 0ae4fa1a343a9c6c9964b51f4808b30ca462cadd79b6280d73fd6b94fab3e4ff`):** Untouched disjoint offsets `[200, 300)` (`400`; `musique` `2h: 58, 3h: 29, 4h: 13`) + `erb[50:150)` (`92` scorable + `5` `"High Level"` empty-gold + `3` `"Info Not Found"`) -> `500` total (`492` retrieval-scorable) + `50/50` unique `musique[200:250)` hop-1 bridge-masked probes (`200` reject labels total) + `n = 100` (`[200, 225)` × 4 public datasets) paired `dspy.ReAct` QA. Executed **once** after `M_PURE_0P` freeze.
2. **Reject Calibration (`P4`):** ERB labels unanswerability strictly on `question_type == "Info Not Found"` (`3` Tier D / `6` Tier C / `6` Tier B vs remaining ERB queries). Each `MuSiQue` probe masks **all** chunks sharing the hop-1 `masked_doc_path` across dense vectors, postings, DF tables, FTS5 `MATCH`, and `CitationDiffusionOps` (`masked_leaks = 0` verified across all runs).
3. **Statistical Inference:** `10,000`-resample dataset-stratified bootstrap 95% CIs (`seed = 0x594157`), two-sided paired sample permutation tests ($n_{\text{perm}} = 10{,}000$), two-sided Wilcoxon signed-rank tests, Cohen's $d_z$, and exact Holm–Bonferroni FWER step-down. Null controls `B5S_RAND_SEEDS` / `B5A_RAND_WALK` average `20` seeded random draws per query into `1` paired observation/query (`n = 243 / 490 / 492`).

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
- **Cold-Open Memory Note:** `AP3_UNICODE61_LEX` measured `maxTM = 252.19 MiB` and `maxRSSd = 188.97 MiB` in the isolated harness when opening `incumbent_unicode61.db` (which retains `spaCy` `tokens_json`/`entities_json` columns and omits persisted `yaw_stage1`). Cold-open `tracemalloc` and RAM on a `spaCy`-free `unicode61 + yaw_stage1` schema remain unmeasured.

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

---

## 8. Threats to Validity, Negative Results & Explicit Non-Claims

1. **100% English Real-Corpus Scope:** All five full-corpus datasets (`37,570` docs) are English. Non-English capabilities (`DE` decompounding / `ß <-> ss` casefolding, `ZH` 2-char EOF CJK seeks, `FI` inflections) are verified solely via adversarial real-SQLite/FTS5 lab unit tests in `tests/test_mdl_dag_lab.py` (`exp/mdl-dag-ablation@032fe2d`), NOT live non-English QA corpora. Switching to `unicode61` sacrifices trigram CJK substring matching unless supplemented explicitly.
2. **Tier C vs Tier D Confirmatory Boundary:** Only `M_WIN_0P` consumed Tier C (`4289edbc...`) as an untouched holdout (and failed `P4`/`P5`). `M_PURE_0P` was tested **once** on the newly generated disjoint `Tier D` holdout (`0ae4fa1a...`); Tier C results for `M_PURE_0P` (`eval3_tier_c_mpure0p_comparison.json`) serve purely as a post-freeze simplification comparison against `M_WIN_0P`. Similarly, `AP3_UNICODE61_LEX` appeared as an LOO arm inside `Tier D` and `Tier C` runs rather than a separately preregistered primary candidate.
3. **`P3` Statistical Power (`n = 100`):** On `100` paired `dspy.ReAct` QA queries (`25` per public dataset), `M_PURE_0P` outperformed `B3_CS6` by `+5.00 pp EM` and `+3.79 pp F1` and cleanly cleared the preregistered non-regression gate (`ci_high >= -0.015`, `mean_diff >= 0.0`), while individual per-query variance keeps two-sided $p_{\text{perm}} \approx 0.26$ (`0` model errors across all `200` runs). Uncompiled zero-shot `dspy.ReAct` QA still trails full `dspy.GEPA`-compiled upstream `70B` systems on `PopQA` (`EM = 0.0400 / 0.0800`).
4. **Falsified / Negative Results Recorded Verbatim:** Early `M_FULL` unnormalized linear path-marginal Gibbs (`eval2c_mfull_loo.json`: `P5 = 2/9`, `Rej = 0.6227`), Marchenko–Pastur bulk whitening (`H1_MP_WHITEN`: `eval2b_frontier.json`, `Pool R@5 = 0.6492`, `p95 = 2,017.8 ms`), scalar multi-branch NNLS decode (`H2_MULTI_BRANCH`: `PopQA 0.4300`, `p95 = 1,631.5 ms`), empirical quantile tails (`H4_EMPIRICAL_TAIL`: `Pool R@5 = 0.5868`), `M_WIN_0P`'s finite-horizon Entity-HMM + Gibbs BMA (`eval3_tier_c_retrieval.json`: Tier C `P4` `Reject AUROC = 0.6432 < 0.70`, `P5 = 7/21`), and `trigram` outperforming `unicode61` (`AP3_UNICODE61_LEX` falsified that hypothesis on both Tier D and Tier C).

---

## 9. Reproducibility & Explicit `EVAL-5` Decision Options

### 9.1 Reproducibility Commands (`exp/mdl-dag-ablation` @ `032fe2d`)

```powershell
# 1. Verify full test suite + architecture gates on isolated lab branch
git checkout exp/mdl-dag-ablation
uv run --offline --locked pytest -o addopts=""

# 2. Verify Tier B discovery + Tier D untouched confirmatory + P3 QA + Tier C comparison
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier b --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_0P AP1_NO_RABITQ_EXACT AP2_1BIT_RABITQ AP3_UNICODE61_LEX AP4_NO_DIFFUSION AP5_NO_TITLE_CIT AP6_NO_PMI_CHAN AP7_NO_NOISYOR AP8_NO_MAXP --output outputs/mdl-dag-eval/reports/eval2e_mpure0p_loo.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier d --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json --confirm-sealed-tier-d --arms B0_DPR B1_FTS_UNI B2_FTS_TRI B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_0P AP1_NO_RABITQ_EXACT AP2_1BIT_RABITQ AP3_UNICODE61_LEX AP4_NO_DIFFUSION AP5_NO_TITLE_CIT AP6_NO_PMI_CHAN AP7_NO_NOISYOR AP8_NO_MAXP --output outputs/mdl-dag-eval/reports/eval3_tier_d_retrieval.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier d --manifest-path outputs/mdl-dag-eval/manifests/splits_tier_d_confirmatory.json --confirm-sealed-tier-d --include-qa --arms B3_CS6 M_PURE_0P --output outputs/mdl-dag-eval/reports/eval3_tier_d_qa.json
uv run --offline --locked python -m scripts.mdl_dag_lab.eval_harness --tier c --confirm-sealed-tier-c --arms B3_CS6 B5S_RAND_SEEDS B5A_RAND_WALK M_PURE_0P AP1_NO_RABITQ_EXACT AP2_1BIT_RABITQ AP3_UNICODE61_LEX AP4_NO_DIFFUSION AP5_NO_TITLE_CIT AP6_NO_PMI_CHAN AP7_NO_NOISYOR AP8_NO_MAXP --output outputs/mdl-dag-eval/reports/eval3_tier_c_mpure0p_comparison.json
```

### 9.2 Decision Matrix for Conditional `EVAL-5` Promotion (Blocked Under Strict `P5 = FAIL`)

Under [mdl-dag-sprint.md §4](mdl-dag-sprint.md#4-subagent-execution-checkpoints--implementation-ledger), **`M_PURE_0P` clears `P0`–`P4` while failing strict `P5` (`5/8 PASS`)** because incumbent `AP3_UNICODE61_LEX` outperformed `trigram` on English and `AP6`/`AP2` missed strict 95% CIs. Consequently, **`EVAL-5` is blocked by default and MUST NOT begin until the user explicitly chooses one of three disposition paths:**

| Disposition | What Happens Next Before/During `EVAL-5` | Trade-Off & Evidence Status | Dependencies Deleted |
| :--- | :--- | :--- | :--- |
| **Path A (Option A)** *(Recommended — Requires Intermediate `EVAL-3b` First)* | **Run intermediate `EVAL-3b`** in `exp/mdl-dag-ablation` for `spaCy`-free `unicode61 + yaw_stage1` (plus user choice on keeping vs pruning `AP6` PMI and `4-bit` vs `1-bit` RaBitQ), measure cold-open `tracemalloc`/RAM + fresh disjoint holdout + `P3` `n=100` `dspy.ReAct` QA, and only then execute `EVAL-5`. | `AP3_UNICODE61_LEX` achieved top Tier D (`0.8098 R@5`, `0.7858 H4`, `0.7133 MH AH@5`, `64.5 ms p95`, `Rej 0.7887`) & Tier C (`0.8039 R@5`, `63.7 ms p95`) results, using `5.61x` smaller `chunk_fts_data` (`45.76 vs 256.59 MiB`). **Currently unmeasured:** `~250.5 MiB` estimated `dbstat` for `spaCy`-free `unicode61 + yaw_stage1` (`AP3` opened `incumbent_unicode61.db` at `maxTM = 252.19 MiB` in memory), its cold-open RAM, and `AP3` `P3` QA (`eval3_tier_d_qa.json` ran only `B3_CS6` and `M_PURE_0P`). Sacrifices native `<3`-char CJK `trigram` matching. | **`optuna`** + **`spacy`** / `en-core-web-sm` *(after `EVAL-3b`)* |
| **Path B (Option B)** *(Requires Explicit User `P5` Override)* | **With explicit user waiver of `P5 = 5/8 FAIL`,** promote **exact Tier-D `M_PURE_0P`** (`trigram` + 4-bit RaBitQ + 1-bit PMI + converged diffusion + conditional Noisy-OR + MaxP) into `dev` with zero post-holdout modifications. | Preserves strict post-holdout freeze (`Tier D P0–P4 PASS`: `0.8010 R@5 / 86.2 ms`, `EM +0.0500`, CJK/DE trigram support intact); accepts `P5 = 5/8` (`unicode61` faster/higher on English) and larger `487.42 MiB` DB footprint (`256.59 MiB` `chunk_fts_data`). | **`optuna`** + **`spacy`** / `en-core-web-sm` |
| **Path C (Option C)** *(Clean Rejection)* | **Reject promotion (`0` edits to `yaw/**`)** | Retains unmodified `3cd3952` `yaw/retrieval.py` (`B3_CS6`: `0.6627 R@5`, `0.4233 MH AH@5`, `110.04 MiB`) + `optuna`/`spacy`; archives `exp/mdl-dag-ablation` cleanly on its branch. | None |
