# Epic 3: Query-Time Evidence Inference over an FTS + Vector Index

**Hypothesis and experiment proposal — 2026-11-16.** Status: proposed research; no experiment in this paper has been executed. Research checkpoint: `E3-FVG-P0`. Source audit: `dev`, HEAD `5138c2ab4435ff0163b718ced45741425cf40359`, with concurrent untracked user work. Historical measurements below were read from existing artifacts, not rerun. Numerical budgets and acceptance margins are proposed preregistration values, not approved production defaults.

| Relation | Links and responsibility |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../../FOUNDATIONAL.md), [AGENTS.md](../../AGENTS.md); implementation remains subject to their acceptance and failure policies |
| Parent | [Epic 3](../epic-3-performance.md); this paper proposes its retrieval research program and does not replace the governing epic |
| Prerequisite | Accepted Epic 1/2 gates and a frozen [bare-bones baseline](bare-bones-replacement-sprint.md); authoring this proposal does not authorize running expensive studies or changing the current cutover |
| Rules | [Documentation](../../.agents/rules/docs.md), [graph storage/retrieval](../../.agents/rules/graph-storage.md), [inference](../../.agents/rules/inference.md), [tests](../../.agents/rules/tests.md) |
| Owns code | None. Experimental code, dependencies, model training, and production promotion require a subsequent scoped implementation contract |
| Evidence | [Weekend shootout](substate-manifold-shootout.md), [historical research](hmm-graph-substate-manifold-research.md), [H1 review](hmm-retrieval.md), [evaluation ledger](../evaluation-graph-storage.md) |
| References | [Local paper collection](../papers/epic-3-fts-vec-graph-experiment/README.md), [version/hash manifest](../papers/epic-3-fts-vec-graph-experiment/manifest.json) |
| Supersedes | No accepted specification. Corrects the interpretation that the weekend either established Bayesian path inference or ruled out useful subspace representations |

## Abstract

YAW seeks an algorithm that queries an FTS + vector index at retrieval time and exceeds contemporary multi-hop GraphRAG Recall@N, exact match (EM), and answer F1 without a persistent semantic graph. The motivation is to retain source evidence, avoid generative graph-construction cost and omissions, and recover the particular connections a question needs. The weekend trials support inexpensive expansion but leave candidate discovery, path inference, and complete-evidence selection confounded.

This proposal separates those problems. Two competing inference tracks use the same transient candidate graph: finite-horizon Gibbs path inference and query-conditioned scalar/vector/sheaf diffusion. The first supplies forward/backward path occupancies; the second tests whether transforming semantic components across heterogeneous connections improves evidence recovery. Low-rank tensor interactions are a separate scorer axis, alongside residual-subspace selection, higher-order evidence joins, and a query-conditioned graph neural policy. Ordinary hybrid retrieval, equally funded iterative search, and evidence-set beam search are strong controls.

The central counter-hypothesis is that the benefit comes from additional searches or better representations, not graph inference. A second is that ordinary source text contains relational distinctions which current embeddings and noun bridges cannot recover. Oracle diagnostics, fixed-candidate comparisons, controlled acquisition, and full-corpus evaluation are designed to distinguish these explanations. No theorem below establishes semantic correctness, efficient unrestricted evidence-set optimization, or SOTA performance. The result may properly be rejection of the exotic mechanisms.

## 1. Research question, scope, and definition of success

The user's research question is:

> What is the algorithm for querying a fts+vec index at retrieval, where we beat the SOTA multi-hop corpora GraphRAG Recall@N, EM and F1 metrics without a persistent graph?

The intended output is a ranked source-evidence set and an auditable account of how its members were discovered and connected. A short, plausible-looking trail is insufficient if it drops a required supporting passage. A correct answer alone is insufficient if it comes from the reader's parametric knowledge.

**Core storage contract.** Persist source pointers, hashes, ordinary FTS postings, chunk vectors, and the existing indexing runtime's required memoization. Do not persist corpus-level semantic edges, extracted triples, communities, semantic summaries, learned per-corpus node states, or a reusable descriptor-to-passage adjacency index. Ordinary FTS postings and any physical navigation structure internal to a vector index are permitted search infrastructure. A semantic kNN graph deliberately exposed for retrieval reasoning is an extra experimental storage arm, not part of the core claim.

Per-query candidate adjacency, coverage state, and features live in memory and expire after the request. Audit logs may persist for evaluation, but must never seed future queries. Global trained model weights are allowed only in explicitly supervised arms, with training cost and data disclosed. Query result caches are disabled for primary timing runs; frozen model-response caches may be used only for reproducible offline replay, never presented as live latency.

**Three distinct claims:**

1. **Mechanism gain:** an inference or selection mechanism beats equally funded simple retrieval on the same candidates or acquisition budget.
2. **System gain:** a graphless system beats reproduced graph competitors with a matched reader and stated resource constraints.
3. **SOTA gain:** the system exceeds the best verified comparable results under the exact benchmark task, corpus, split, metric, and supervision regime. A local baseline or a literature macro averaged over different corpora cannot establish this.

Strict Pareto domination requires no material worsening in declared cost dimensions and a meaningful quality improvement. A higher-quality point with higher latency may extend a frontier; it is not domination. A small production codebase remains a constraint, not a substitute for quality.

## 2. What the weekend established

### 2.1 Results read directly from saved artifacts

The [Pass-4 paired shootout](../../outputs/benchmarks/shootout_pass4_v3_4ds_10q.json) contains 40 questions, ten each from 2WikiMultiHopQA, MuSiQue, HotpotQA, and PopQA. Its overall acceptance gate is false for both candidates.

| Saved candidate | R@1 | R@2 | R@5 | R@10 | R@20 | EM | F1 | Recorded total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy graph | .3396 | .5521 | .7021 | .8292 | .9250 | .0750 | .1908 | 34.3296 |
| Pure-chunk, active parameters | .4333 | .6583 | .8146 | .8938 | .9854 | .1250 | .2482 | 45.4962 |

These are bounded paired improvements, not full-corpus parity or a universal speedup. The original serial graph-indexing savings and this warm comparison measure different workloads.

The [Pass-4 probe](../../outputs/benchmarks/probe_manifold_pass4_v3.json), offset 10, 80 questions, reports default R@5 .833333 versus hop-0 .755208 and raw vector .786458. At R@20, default .941667 trails raw vector .944792. The [pooled-distractor probe](../../outputs/benchmarks/pooled_distractors_v3_probe.json), 1,348 chunks and 120 questions, reports R@5 .795833 versus hop-0 .716667. Both support expansion; neither isolates the mathematical cause.

| Pass-3 ablation | R@5 | R@20 | Interpretation at these horizons |
| --- | ---: | ---: | --- |
| Full | .785417 | .972917 | Historical reference |
| Hop-0 only | .722917 | .966667 | Expansion helped |
| Scalar rank-1 | .766667 | .981250 | Higher rank helped R@5, hurt R@20 relative to rank-1 |
| No backward sweep | .785417 | .972917 | No demonstrated gain at these horizons |
| No CPTP rescue | .785417 | .960417 | No R@5 gain; R@20 changed |
| No coalition coupling | .785417 | .972917 | No demonstrated gain at these horizons |

Source: [ablation artifact](../../outputs/benchmarks/manifold_ablations_4ds_10q.json). No confidence interval or causal generalization is inferred from these small aggregates. Optuna importance measures and parameter sensitivity on synthetic tests do not establish independent real-corpus usefulness. Later pure-chunk changes were bundled, so they do not prove that every multi-vector/subspace representation is unnecessary.

### 2.2 Source-level gaps in what was tested

At the audited snapshot, [hmm.py](../../yaw/daemon/graph/hmm.py) implements pure-chunk heuristics, not classical HMM filtering/smoothing or calibrated Bayesian inference:

- `derive_manifold_telemetry` reports eight hops, while `range(1, min(3, eff_hops))` permits only two expansion steps. An eight-hop policy remains untested by this code.
- A shared `q_act` is deflated using the first nonempty branch's first frontier node. Branch hypotheses do not maintain independent residual states.
- `unmet = q_tids - t_sets[p_ci]` compares against a parent chunk rather than all selected evidence.
- Branch agreement includes the same `s0_norm` in every branch before a geometric mean. Positive agreement can occur without independently discovered support.
- The first hop-0 seed is forcibly emitted first. The terminal softmax and the diversity-reranked output are different objects; neither is a calibrated truth posterior.
- `_build_routes` adds both actual parent and previous-output dependencies, returns snippets rather than exact bridge spans, and operates separately from final passage selection. Acyclicity alone does not validate these dependencies or ensure returned evidence contains the required connectors.
- ABTT uses an uncentered second moment, a floored spectral cutoff, and at most eight directions. Its Marchenko-Pastur interpretation requires a null-model check; it is not justified merely by computing eigenvalues. CSLS-style preprocessing performs tiled all-pairs work.

The [probe script](../../scripts/probe_manifold.py) constructs an in-memory representation of all loaded chunks. Expansion can score the whole slice, rather than being confined to a seed-local acquisition frontier. The [historical slice builder](../../yaw/daemon/ingestion/datasets.py) pools question-associated passages and explicitly includes gold documents. That is a legitimate conditional development task, but it is easier than blind acquisition from a fixed full corpus. Repeatedly inspected weekend offsets are development material for this program, not a fresh final holdout.

## 3. Literature: supporting evidence and limits

The sources support components, not the full proposed algorithm. Links below identify the primary publications; version-pinned local PDFs are indexed in Section 15. Published and preprint results are not pooled numerically.

| Evidence family | What is supported | What it does not establish for YAW |
| --- | --- | --- |
| [AIR, ACL 2020](https://aclanthology.org/2020.acl-main.414/) | Unsupervised evidence retrieval using uncovered query terms and controlled expansion | Open-corpus acquisition with unknown answers; its multiple-choice tasks and candidate construction differ |
| [GoldEn, EMNLP 2019](https://aclanthology.org/D19-1261/) | Context can expose a missing entity and generate a useful ordinary search query | That noun extraction alone selects the correct bridge or relation |
| [MDR, ICLR 2021](https://arxiv.org/html/2009.12756v2) | Evidence-conditioned dense search works without a prebuilt semantic graph | That an untrained linear mixture of chunk/query vectors reproduces a trained multi-hop encoder |
| [Baleen, NeurIPS 2021](https://arxiv.org/abs/2101.00436) | Condensing accumulated evidence can make later retrieval more focused | That condensation is free or lossless; YAW's zero-generation lane must use extractive state |
| [Beam Retrieval, 2024 version](https://arxiv.org/html/2308.08973v2) | Competing partial passage sets mitigate early commitment | Its strong candidate-set results do not by themselves establish corpus-wide discovery |
| [IRCoT, ACL 2023](https://aclanthology.org/2023.acl-long.557/) | Retrieval and intermediate reasoning can improve each other | Zero-generation retrieval, or gains independent of extra model/search budget |
| [Randomized shortest paths, 2009](https://pubmed.ncbi.nlm.nih.gov/19323635/) and [linearly solvable control, NIPS 2006](https://papers.nips.cc/paper_files/paper/2006/hash/d806ca13ca3449af72a1ea5aedbed26a-Abstract.html) | Relative-entropy regularization and exponential transforms yield tractable path/control formulations under their assumptions | Useful semantic rewards, undiscovered edges, or tractability of arbitrary set-dependent rewards |
| [Schrodinger survey](https://arxiv.org/abs/1308.0215), [graph transport](https://arxiv.org/abs/1308.0226) | Endpoint-conditioned path laws and connections to optimal transport | An answer endpoint when the question's missing referent has not been found |
| [Cost-augmented bridges, October 2026 preprint](https://arxiv.org/html/2610.02195v1) | State-additive costs can be folded into reference dynamics; reports numerical graph solvers | Retrieval/QA improvement or a solver for coverage-dependent evidence rewards |
| [Subspace operations, NAACL 2024](https://aclanthology.org/2024.naacl-long.192/) | Continuous set membership and operations can help text similarity and set retrieval | Entailment, relational direction, or multi-hop evidence completeness |
| [DPP selection, AISTATS 2020](https://proceedings.mlr.press/v108/han20b.html) and [submodular summarization, ACL 2011](https://aclanthology.org/P11-1052/) | Quality/diversity and diminishing-return objectives have efficient useful approximations | That geometric novelty equals answer information, or all complementary evidence is submodular |
| [ColBERTv2, NAACL 2022](https://aclanthology.org/2022.naacl-main.272/) | Token-level late interaction retains matching detail with compression | Free multi-vector indexing, or equivalence between its trained token features and chunk embeddings |
| [G-Retriever](https://arxiv.org/html/2402.07630v3) | Prize-versus-cost subgraph selection is a practical RAG formulation | Query-time edge discovery or global optimization of arbitrary evidence novelty |
| [NBFNet](https://arxiv.org/abs/2106.06935), [GNN-RAG](https://arxiv.org/abs/2405.20139), [GFM-RAG](https://arxiv.org/abs/2502.01113) | Learned path/message operators and query-conditioned graph retrieval are established directions | Recovering absent source relations from a generic similarity graph |
| [DGCNN](https://arxiv.org/abs/1801.07829), [non-backtracking GNNs](https://arxiv.org/abs/2310.07430) | Dynamic feature-space neighborhoods and reduced immediate message recycling are feasible | Textual relation fidelity; non-backtracking also does not eliminate longer cycles |
| [GraphFlow, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/89d0d5c2f720921df93bbb8fef514571-Abstract-Conference.html) | Retrieval as sequential reward-guided flow is already researched | A graphless solution: its STaRK experiments concern text-rich KGs |
| [EAGR, November 2026 preprint](https://arxiv.org/pdf/2611.01910) | An implicit graph can be traversed through evidence-conditioned descriptor lookup | The strict storage contract: it precomputes semantic descriptors and uses generative assessment |
| [QueryFlow, November 2026 preprint](https://arxiv.org/html/2611.04625v1) | Query updates coupled to relational propagation address complementary support | Graphless indexing: its corpus entity graph remains fixed |
| [Bayesian experimental design](https://proceedings.mlr.press/v108/foster20a.html), [adaptive submodularity](https://arxiv.org/abs/1003.3967) | Principled sequential information acquisition, with explicit model/property assumptions | That heuristic score entropy is answer entropy, or greedy retrieval inherits the guarantees |
| [Retrieval diffusion, CVPR 2017](https://arxiv.org/abs/1611.05113) | Sparse diffusion over regional image representations can improve retrieval | Its image-manifold assumptions or offline graph work transfer to graphless textual acquisition |
| [Neural Sheaf Diffusion, NeurIPS 2022](https://arxiv.org/abs/2202.04579) | Endpoint restriction maps change feature alignment and diffusion behavior on heterophilic graphs | Learned maps establish source relations or recover missing passages; experiments concern graph learning |
| [Joint sheaf diffusion, GRaM 2024](https://proceedings.mlr.press/v251/hernandez-caralt24a.html) | Map construction needs an inductive bias; investigates limits of existing sheaf benchmarks | A generic feature-conditioned map is sufficient for text retrieval or relational correctness |
| [Polynomial sheaf diffusion, 2026 version](https://arxiv.org/abs/2512.00242v3) | Reports spectral filtering with economical diagonal maps; challenges the need for rich per-edge maps | Multi-hop QA gains, or that polynomial filters must be low-pass diffusion |
| [Neural tensor networks, NIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html), [Tensorizing Neural Networks, NIPS 2015](https://arxiv.org/abs/1509.06569) | Multilinear relational interactions and tensor-train parameter compression are established tools | Compressing parameters creates useful semantic factors or guarantees faster end-to-end retrieval |

### 3.1 What is plausibly new

No novelty claim is made for path integrals, transient graphs, hybrid search, GNNs, residual geometry, or beam search independently. The research contribution, if demonstrated, would be a small source-grounded composition that recovers full evidence sets from one ordinary index at a better measured quality/cost frontier, with clean causal ablations. Search coverage is substantial but not a systematic novelty review; expand it before publication.

### 3.2 Original four-paper verdict under this framing

The [subspace-sum graph](https://arxiv.org/html/1702.08245v1) and [its automorphisms](https://arxiv.org/html/1704.03787v1) concern algebraic subspace adjacency, not semantic evidence acquisition. Their global spanning condition is unsuitable for low-rank chunk representations. [Hessian Null Space Continuation](https://arxiv.org/html/2609.38081v1) concerns model-weight geometry and remains methodological background. [Graded neural networks](https://arxiv.org/html/2407.19031v2) require a justified grading and do not establish a suitable one for YAW. These are archived as excluded alternatives, not ingredients of the primary experiment.

## 4. Hypotheses and counter-hypotheses

| ID | Testable hypothesis | Falsification or competing explanation |
| --- | --- | --- |
| H1 | Backward-conditioned path inference improves complete-support recovery on a fixed discovered graph over equal-budget forward scoring and beam search | No gain when candidates, features, terminal potential, and compute are matched; benefits come only from a better terminal scorer |
| H2 | Evidence-conditioned acquisition finds gold support outside hop-0 more efficiently than repeated hybrid search | Equal-budget diverse or random bridge expansion matches it; it only buys more searches |
| H3 | Query-conditioned residual-subspace selection improves context-budgeted support completeness | Ordinary MMR, facility-location coverage, or the same multi-vector features without subspace operations match it |
| H4 | Sparse higher-order factors recover complementary branch evidence missed by additive path scores | Pairwise factors suffice; gains require gold facets, or the combinatorial cost dominates |
| H5 | A small inductive GNN learns transferable acquisition/selection rules without a corpus graph | An MLP/set scorer with identical features, supervision, and training budget matches it; topology supplies no incremental value |
| H6 | Adaptive utility-based stopping reduces cost at noninferior complete-support and QA quality | It stops before delayed-payoff bridges or requires expensive sufficiency judgments |
| H7 | A retained graphless combination exceeds strong reproduced GraphRAG under the stated budgets | Gains disappear on fixed full corpora, fresh queries, matched readers, or amortized workload cost |
| H8 | Query-conditioned transport of separate evidence components improves full-support recovery over scalar and identity-map diffusion | Equal-feature identity/diagonal controls match it, gains require oracle maps, or useful transport increases cost beyond its quality benefit |
| H9 | A low-rank query-parent-candidate tensor scorer improves relational expansion beyond additive, bilinear, and capacity-matched MLP scorers | Extra parameters/supervision explain the gain; role-reversal and cross-dataset performance do not improve |

**Null position:** more complex math has no privileged status. A strong hybrid search with a small learned scorer may be the winner. Failure to outperform that control is useful evidence and closes the mechanism rather than prompting a larger stack.

## 5. Formal problem and information boundary

Let `D` be a fixed corpus, `I=(I_fts,I_vec)` its ordinary index, `q` a question, and `A` the set of permitted index actions. A retrieval policy observes only `q`, retrieved source spans, model outputs charged to its budget, and previous action results. It does not observe gold answers, gold evidence, gold decompositions, test labels, or hidden full-corpus feature arrays.

The state at round `t` is `s_t=(q,C_t,S_t,U_t,L_t,B_t)`: candidates, selected evidence, unresolved facets, provenance, and remaining budget. Facets are estimated lexical/dependency features in the zero-generation lane; they are not assumed to be a correct question decomposition. Unresolved dates, predicates, negation, quantities, and roles remain part of the question. Noun-only bags are an ablation, not the default.

The ideal policy maximizes final supported-answer utility minus acquisition, latency, and context costs. The actual experiment estimates that utility with explicit proxies and tests their predictive value. Shannon information requires a latent answer/referent variable and an observation model; applying softmax to similarities supplies neither. If a model of answer hypotheses is introduced, calibration is evaluated on development data before reporting expected information gain. Realized informative evidence can increase uncertainty by revealing a contradiction.

No deterministic transformation of an embedding recovers corpus-specific distinctions that the embedding discarded. A trained policy can contribute prior knowledge, and source hydration can contribute missing detail; both must be distinguished from information already present in vectors. Candidate access is therefore the primary ceiling. If no action discovers a required support passage, no downstream inference algorithm can select it.

## 6. Proposed algorithm: discovery, inference, selection

### 6.1 Discovery through the existing index

1. Run FTS on the intact original query and vector search on its embedding. Merge with a fixed rank-fusion control, deduplicate by stable source identity, and hydrate candidates strictly from verified source pointers.
2. Parse only visited source windows. Propose a bounded set of bridge spans using spaCy noun phrases/proper names, predicates, dates, and document links available from the source. Keep exact coordinates. Do not parse the entire corpus into an undisclosed in-memory semantic index.
3. Form actions from the original question, an estimated unresolved facet, and a source-grounded bridge. Separate FTS phrase queries from vector probes. Preserve a direct-original-query action to measure drift.
4. Retrieve new candidates from `I`. Store transient directed discovery edges with parent span, exact action text/vector construction, child ID, rank, and scores. Label an edge as a retrieval association, not an entailed fact.
5. Recompute the candidate-state policy after each bounded round. Reject stale source hashes and service failures explicitly. No retries outside native DSPy policy and no degraded retrieval path.

Two separate resource lanes prevent a hidden change to the bare-bones one-query-embedding contract:

- **Lane A:** one query embedding; subsequent vector probes are documented algebraic combinations of existing vectors. Lexical bridge searches still issue real FTS requests. This is the primary fixed-representation lane.
- **Lane B:** fresh embeddings of contextual follow-up queries and, optionally, visited sentence windows. Every input, request, and token is charged. It is a research arm requiring an explicit Epic 3 contract amendment before execution; it does not silently change the current baseline.

### 6.2 Finite-horizon path inference

Freeze one discovered graph for an inference pass. Its states are `(depth, chunk)`; the initial distribution `pi` comes from seeds. Each state has reference probabilities `rho_t(j|i)` for proposed transitions and `rho_t(STOP|i)`, summing to one. At the horizon `H`, STOP has reference probability one. A zero-edge state can stop; an unreachable endpoint is never repaired by fabricating an edge.

For fixed query-conditioned edge rewards `r_t(i,j)`, nonnegative costs `c_t(i,j)`, terminal reward `u_t(i)`, and temperature `tau>0`, define:

\[
K_t(i,j)=\rho_t(j\mid i)\exp[(r_t(i,j)-\lambda c_t(i,j))/\tau],\qquad
g_t(i)=\rho_t(\mathrm{STOP}\mid i)\exp[u_t(i)/\tau].
\]

The resulting normalized distribution over finite stopped paths is a Gibbs tilt of the reference law. Evaluate in log space with log-sum-exp:

\[
b_H(i)=g_H(i),\quad b_t(i)=g_t(i)+\sum_jK_t(i,j)b_{t+1}(j),\quad Z=\sum_i\pi(i)b_0(i).
\]

Forward weights satisfy `a_0=pi`, `a_(t+1)=a_t K_t`. Edge occupancy is:

\[
\xi_t(i,j)=a_t(i)K_t(i,j)b_{t+1}(j)/Z.
\]

Stop probability at `(t,i)` is `a_t(i)g_t(i)/Z`; these stop probabilities sum to one for a feasible finite model. Summed node occupancy is an expected visit count and may exceed one when a chunk reappears. It is not a factual correctness probability. Use node/edge occupancies for ranking, max-product decoding for an auditable best route, and an explicit set selector for the final context.

With fixed potentials, complexity is `O(H(|V|+|E|))` for the repeated-state lattice, excluding candidate discovery, feature extraction, parameter fitting, and output selection. Sparse matrices need no inverse or matrix exponential in this discrete finite-horizon experiment. If states store last-edge identity for non-backtracking updates, report the larger line-graph transition count; do not claim the original bound automatically.

**Reward specification for the first experiment.** Fit scorer weights on training labels only; use development data for hyperparameter selection and calibration, then freeze both. Standardized local features are original-query relevance, unresolved-facet match, exact/soft bridge consistency, source continuity, hub frequency, and token/search cost. A scalar linear scorer is the first learned arm. Use identical features and weights for forward, beam, and forward/backward controls. Source-role/polarity judgments absent from these features remain a limitation.

For each inner inference pass, all features condition on the same frozen outer evidence set `S_t`. The terminal potential `u_h(i; q, S_t)` is a **local completion proxy** based on that set and the current chunk, not the sufficiency of an arbitrary hypothetical path reaching it. Different paths reaching `(h,i)` can contain different required evidence. Full-set sufficiency and the final insufficient-evidence decision belong to the evidence-set selector. Test constant, heuristic, and trained local terminal potentials separately. Gold terminal information is an oracle diagnostic only. A later coverage-mask state `(h,i,mask)` may distinguish recovered facets, but its state count can grow by `2^f` for `f` binary facets; exact visited-chunk history is still more expensive. Such variants must report their expanded transition/state complexity and cannot inherit the baseline bound silently.

### 6.3 Why this is not yet a Schrodinger-bridge solution

The core method is a finite Gibbs path model with soft termination. A classical Schrodinger bridge adds prescribed endpoint marginals. That extra constraint is attractive only if YAW can estimate useful endpoint support without answer leakage. Compare soft terminals first; run hard-marginal or unbalanced-transport variants only after a terminal-quality diagnostic shows headroom. Never force mass toward an answer location supplied by gold data in a headline result.

The Feynman-Kac literature supports absorbing additive state costs into a reference process. It does not solve path-dependent novelty, answer sufficiency, unknown graph discovery, or arbitrary branch conjunctions. A continuous-time bridge solver is not necessary merely because it has an exact mathematical characterization.

### 6.4 Selection with residual geometry

Let `X_c` contain feature vectors for candidate `c` and let `U_S` be an orthonormal basis for selected evidence features. An experimental novelty score is:

\[
R_c=(I-U_SU_S^\top)X_c,\qquad
\Delta_{vol}(c\mid S)=\log\det(I+\sigma^{-2}R_c^\top R_c).
\]

Apply query-dependent feature weights before factorization, using estimated unresolved facets. Limit rank and compare against equal-rank random projections, raw multi-vector MaxSim, MMR, and facility-location selection. Report singular-value sensitivity and rank saturation. In high dimensions, unrelated chunks can be nearly orthogonal and score highly; irrelevant-noise controls must catch this.

This residual formula is a geometric heuristic. It is not automatically a conditional DPP marginal or Bayesian information gain. An exact regularized log-determinant marginal instead uses `A_S=I+sigma^-2 sum_(s in S) X_s X_s^T` and `logdet(I+sigma^-2 X_c^T A_S^-1 X_c)` through the determinant lemma. Compare these explicitly rather than conflating them. Neither objective establishes entailment.

Set selection preserves useful paths by charging the entire newly required connector bundle against the context budget. Compare against selecting the same candidates without connectivity constraints. Independent comparison branches may share only a virtual query root. Do not force a fabricated relationship between them.

### 6.5 Sparse higher-order joins and optional neural inference

When two branches must jointly satisfy different roles, test a factor `psi_q(c_a,c_b,role)` over a bounded shortlist. Exact source identity, temporal compatibility, distinct role coverage, and relation-direction features can be cheap; rich entailment may require a trained cross-encoder or generation. These belong to separate charged arms. A low-rank tensor factorization reduces arithmetic only under its stated representation and rank; it does not eliminate candidate-combination search.

A query-conditioned GNN may replace the local transition/selection scorer while leaving indexing unchanged. Its graph is built per request from the same candidates and edge features. Compare with an MLP and a permutation-invariant set scorer using identical features, training examples, and parameter budget. Add shuffled-edge and no-message-passing controls. A better GNN result is evidence for topology only if these controls are beaten. Attention weights are not a source-grounded explanation.

### 6.6 Search procedure and exactness boundary

```text
seeds = hybrid_search(original_query)
states = initial_evidence_hypotheses(seeds)
while acquisition_budget_remains:
    for state in surviving_states:
        propose bounded source-grounded actions from unresolved needs
    execute policy-selected actions against the ordinary index
    freeze the discovered graph; score with fixed potentials
    run the nominated path or diffusion scorer; form candidate route bundles
    select competing evidence sets within the context budget
    retain continuation alternatives for delayed-payoff bridges
    stop if the locked stopping rule accepts, or budget is exhausted
return ranked source evidence, actual discovery edges, and stop reason
```

The fixed-potential inner calculation is exact for its finite model. The outer discovery policy, estimated rewards, history-dependent coverage, and final evidence-set optimization are approximate. Greedy selection is not a globally shortest maximum-evidence DAG algorithm. The layered computation is acyclic in depth, but original chunk identities can repeat; deduplication and genuine output-dependency cycle checks remain required. Non-backtracking excludes immediate reversals only. Multiple branches need separate state or higher-order factors.

### 6.7 Competing inference track: scalar, vector, and sheaf diffusion

**Research question.** Does relevance propagate better when a connection transforms the evidence components being followed, rather than transmitting a single score or averaging all components identically? A company-to-founder-to-birthplace chain illustrates a useful topic change. Ordinary similarity smoothing can instead retrieve more company passages. Heterophily in graph-learning papers is an analogy, not proof that those models solve this retrieval failure.

Start with a fixed nonnegative row-stochastic `P_q` on the same discovered associations. A scalar restart control is:

\[
f_{k+1}=(1-\alpha)y+\alpha P_q^\top f_k,\qquad 0\leq\alpha<1.
\]

Compare unconditioned and query-gated weights with the same topology and seed `y`. Sparse heat diffusion `exp(-tL)y` is another scalar control on an explicitly symmetrized affinity graph. Restart/resolvent and heat kernels weight path lengths differently; neither is automatically the finite soft-terminal model in Section 6.2. Tune diffusion time/restart only within the shared development allowance. A scalar convergence guarantee does not imply answer sufficiency or supply a stopping confidence.

For vector transport, attach `x_v in R^d` to each candidate for each explicitly separate branch/facet channel. Compare ordinary identity-map feature diffusion with a cellular-sheaf construction. For each geometric association `e={u,v}`, endpoint restrictions `F_(u,e)` and `F_(v,e)` map node states into a shared edge feature space. Define:

\[
\mathcal E_{\mathcal F}(x)=\frac12\sum_{e=\{u,v\}}w_e\|F_{u,e}x_u-F_{v,e}x_v\|^2,
\qquad L_{\mathcal F}=\delta_{\mathcal F}^{\top}W\delta_{\mathcal F}.
\]

For frozen maps and `w_e>=0`, this yields a symmetric PSD operator. A seeded experiment minimizes `E_F(x)+(gamma/2)||x-y||^2`, with `gamma>0`, using:

\[
x_{k+1}=x_k-\eta\big[L_{\mathcal F}x_k+\gamma(x_k-y)\big].
\]

Use a verified spectral upper bound and `0<eta<2/(lambda_max(L_F)+gamma)` for the fixed linear problem. An implicit linear solve is an equal-cost alternative; report its tolerance and iterations. Treat zero-degree states explicitly. For normalized operators, define and verify zero-block handling before using the claimed spectral range. Arbitrary gates, nonlinearities, or maps recomputed from evolving states invalidate the fixed-operator energy/convergence argument and form separate arms. Fail visibly on nonfinite values; do not hide unstable transport with clipping.

**Direction and meaning.** The geometric sheaf operator is formed from a declared symmetrized projection of discovered associations. Its bidirectional numerical coupling is not evidence of inverse factual relations. Keep original directed discovery provenance separately and allow only actual source-supported dependencies in the returned trail. Signed feature states and energy reductions are not nonnegative probability mass or truth confidence. The symmetric experiment does not inherit directional causal reasoning from its restriction maps.

**Construct maps from available information.** Freeze the outer evidence set during each pass. Estimate maps from query, parent/child features, bridge-span features, and estimated unresolved facets. Use training labels only, or an explicitly documented geometric rule; never use test support roles. Hold source features fixed across map arms. Identity, diagonal, and small orthogonal maps precede dense maps. A generic embedding projection does not acquire relation semantics by being labeled transport. If an oracle role-aligned map helps and learned maps do not, record a map-learning bottleneck. Novel referents or distinctions absent from all available features remain unresolved.

Use a fixed node readout for all diffusion arms, depending on the original query, seed features, and transported state. Compare against readout-only and the same readout applied to undiffused features. Paths are not identified by a final diffusion vector: test source-grounded route-bundle extraction separately, and charge the same extraction procedure to scalar/vector controls. No claim of an explanatory chain follows from a high diffused node score.

Map prediction, Laplacian assembly, normalization, diffusion, and readout all count toward latency and memory. Matrix-free application with dense `d x d` restrictions costs approximately `O(K|E|d^2 f)` for `K` updates and `f` feature channels, excluding map prediction and readout; diagonal maps reduce the transport term to `O(K|E|d f)`. Materializing dense blocks or factoring/normalizing them adds work. Report measured cost rather than assigning scalar CSR complexity to a tensor operator.

Polynomial spectral filters are a contingent variant if depth/oversmoothing diagnostics justify them. A learned filter can amplify high-frequency components, so it must be labeled spectral message passing rather than assumed dissipative heat diffusion. The polynomial-sheaf literature makes diagonal-map controls particularly important. Test depth curves, norm/energy behavior, relevance drift, role-changing bridges, duplicate floods, and source-supported branch coverage. These are query-time operators; no corpus-wide sheaf, adjacency, or contextualized node state is persisted.

### 6.8 Low-rank tensors as a scorer axis, not an indexing mandate

Separate three uses of tensor mathematics: (a) storing a multi-channel node state; (b) multilinear interaction between query, parent, and candidate; (c) factorizing large operators/weights. None guarantees the other two. Tensor-train compression is parameter organization, not a substitute for learning useful relations.

A CP-factorized transition score is:

\[
s(q,p,c)=\sum_{\ell=1}^{r}(a_\ell^\top q)(b_\ell^\top z_p)(d_\ell^\top z_c).
\]

Here `z_p,z_c` are the same fixed features available to competing scorers; the query may include a declared frozen residual-facet encoding. Ordered parent/candidate factors permit asymmetry but do not enforce correct relation direction. For input width `D`, the interaction has `3Dr` parameters before biases/readout, versus `D^3` for an unrestricted third-order coefficient tensor. Precomputing projected candidate features within the query can reduce per-edge contraction cost to `O(r)`; projection work and per-query memory still count. An equivalent multiplicative feature model may implement the same function; the tensor name is not itself a contribution.

Compare this scorer with additive, bilinear, and parameter-/training-budget-matched MLP scorers. Test it first on frozen candidate associations, then as a gate for scalar diffusion or the Gibbs reference/reward model, and only then as a map predictor. Do not change topology, transport, readout, and scoring simultaneously. If the tensor only beats an underparameterized linear model, H9 is not established. Require relation reversal, same-name, date/quantity, and unseen-entity tests as well as full-support recovery.

Tensor trains or Tucker/CP factorizations for larger operators become relevant only after profiling shows an actual operator-size bottleneck and a smaller dense/diagonal model cannot meet the quality gate. Vary rank, report approximation error and realized throughput, and compare identical effective feature access. Do not recreate the weekend's full density-matrix/Kraus state by default. If covariance states are later justified, compare diagonal and full off-diagonal states separately and charge PSD/normalization work; neither positivity nor trace preservation proves semantic accuracy.

## 7. Adversarial challenges to the proposal

| Challenge | Required control or diagnostic | Decision if the concern holds |
| --- | --- | --- |
| More queries explain the gain | Equal search count, embedding-input count, candidates inspected, and context budget; include repeated original-query and diverse-expansion baselines | Attribute gain to acquisition budget, reject an inference claim |
| Missing bridge is outside the graph | Candidate recall, complete-candidate recall, gold-edge and gold-node oracles reported separately | Improve acquisition or representations before tensor/GNN work |
| Terminal reward already solves the task | Compare terminal-only ranking, constant terminal, shuffled terminal, learned terminal, and gold-terminal oracle | Reject path inference if terminal-only matches it |
| Scores recycle the same evidence | Duplicate and paraphrase injection, common-source grouping, branch-shared-baseline ablation | Treat copied evidence as one observation; do not multiply confidence votes |
| Many weak paths overwhelm one good path | Degree/multiplicity sweeps; normalized reference law; compare MAP route and occupancy selection | Revise path prior or retain best-path method |
| Similarity misses relation direction | Acquired-by/acquired, parent/child, before/after, negation and same-name distractors | Introduce targeted source features or a charged relation scorer; no unsupported semantic guarantee |
| Diversity destroys useful near-duplicates | Minimal pairs differing in date, role, quantity, or exception | Reject/restrict volume reward |
| Adaptive stopping misses delayed payoff | 3-8 hop tasks with neutral intermediate bridges; fixed-budget control | Keep continuation beam or fixed budget |
| Training memorizes corpus nodes | Disjoint entities/documents where possible, cross-dataset transfer, anonymized names | No generalization claim for transductive wins |
| The reader supplies the answer | No-context and gold-evidence reader controls, supporting-set metrics, anonymized tasks | Separate reader ability from retrieval contribution |
| Diffusion reinforces topical hubs instead of relational bridges | Identity/query-gated scalar controls, topic-change tasks, common-name/duplicate injection, depth and energy curves | Reject H8 if bridge recovery does not improve or the benefit is only more smoothing |
| Learned maps invent alignment or hide reverse-relation leakage | Identity, diagonal, shuffled/constant maps, oracle-map diagnostic, directed provenance audit | Attribute oracle-only gains to missing map supervision; reject unsupported explanation claims |
| Tensors merely add capacity or compression overhead | Additive/bilinear/MLP controls with the same features, parameter/training budget, rank sweeps and real timing | Reject H9 if the equal-capacity control matches it or inference cost erases benefit |
| Zero persistent graph merely shifts cost online | Cold/warm updates, repetitions, workload amortization | Report the actual break-even point; admit a cached graph may win |

## 8. Baselines and controlled contrasts

| ID | System | Role |
| --- | --- | --- |
| B0 | FTS only; vector only; fixed hybrid rank fusion | Reproducible inexpensive anchors |
| B1 | Hybrid + MMR/facility-location; large one-shot candidate pool | Strong selection control |
| B2 | Frozen accepted bare-bones retriever; archive weekend implementation separately | Production and historical references |
| B3 | AIR-style residual hybrid search with the same action budget | Simple iterative acquisition control |
| B4 | Evidence-set beam search using the same features/terminal scorer | Non-greedy path/set control |
| B5-S | Uniform random candidate selection after identical hop-0 seeds | Mandatory selection floor for every evaluated retrieval mechanism |
| B5-A | Uniform random legal bridge actions after identical hop-0 seeds | Mandatory acquisition floor; separate from diverse expansion |
| B6 | Trained cross-encoder reranker or MDR-style retriever where reproducible | Strong learned non-graph control; training and representation changes disclosed |
| B7 | IRCoT / bare ReAct, with generation tokens and searches charged | Strong agentic control, outside zero-generation recall lane |
| G1 | Reproduced HippoRAG 2 | Persistent graph reference |
| G2 | Reproduced Youtu-GraphRAG; additional current reproducible graph competitor selected before final evaluation | Contemporary graph reference |
| X1 | Fixed-candidate Gibbs path inference | H1 |
| X2 | Same inference driving bounded new acquisition | H2 |
| X3 | X1 or B4 + residual-volume/regularized logdet selector | H3 |
| X4 | B4 or X1 + bounded pair/triple role factors | H4 |
| X5 | Same graph/actions + inductive GNN policy | H5 |
| D0 | Scalar restart and heat diffusion; unconditioned vs query-gated affinities | Strong inexpensive diffusion baseline |
| D1 | Same node features/readout with no propagation or identity-map vector diffusion | Isolate propagation from representation/readout |
| D2 | Query/bridge-conditioned diagonal sheaf restrictions | H8 primary transport arm |
| D3 | Small orthogonal then dense restrictions; constant/shuffled/oracle maps as diagnostics | Isolate useful geometry from capacity and label leakage |
| T1 | CP-factorized query-parent-candidate scorer vs additive/bilinear/MLP | H9, crossed with one frozen inference operator at a time |
| D4 | Bounded polynomial spectral filter on the same sheaf | Contingent depth/filter alternative, not automatically dissipative diffusion |

Do not implement the entire Cartesian product. Use development diagnostics to nominate at most two candidate systems for final confirmation. All baselines receive the same declared tuning budget. A failed or unavailable competitor is reported explicitly, not silently replaced by a historical table. Compare graph systems from pinned isolated external environments rather than importing upstream runtimes into YAW production code. Their output can be evaluated by a common harness.

**Two orthogonal comparisons:** (a) freeze the candidate graph and all feature access to measure inference/selection; (b) freeze action interfaces and budgets while allowing candidate discovery to differ. Then report end-to-end performance. This prevents an apparent inference improvement caused by quietly scanning the whole corpus.

### 8.1 Mandatory random-after-seeds controls

Every experimental retrieval arm, including path, subspace, tensor, diffusion, learned policy and stopping variants, must face **B5-S**. Freeze the query, hop-0 seed IDs/order, seed-retention rule, corpus, eligible deduplicated candidate pool and reader. Replace the experimental ranking of additional chunks with a uniform random permutation of that same pool, without replacement. For retrieval Recall@N, compare count-matched ranked prefixes before reader token packing. For token-budgeted support and QA, apply the same deterministic packing rule and token cap to each ranking, reporting actual included chunk counts; equal token caps need not yield equal counts. Never redraw or reject random permutations to favor short chunks. For adaptive stopping, compare the quality-versus-budget curve at fixed budgets as well as the method's chosen stop; do not make the random control stop early by inheriting a weak model decision. Random selection must not inspect gold labels or use relevance scores. Diverse/MMR expansion is a separate stronger control, not a substitute for randomness.

Preregister retained seed positions and leave at least one selectable position in the primary top-5 prefix, identically for the experimental arm and B5-S. All original seeds remain available as retrieval inputs. If an arm's output contract fixes all five positions to seeds, R@5 is identical by construction and cannot establish the required superiority; resolve that design on development data before nomination, without changing the primary endpoint after unsealing. At other N where no output position is selectable, report the identity explicitly rather than interpret it as evidence for or against a selection mechanism.

Acquisition arms must also face **B5-A**: from the same hop-0 seeds, sample uniformly among legal untried source-grounded frontier actions, then execute the same vec+FTS action interface. Match search/embedding allowances, per-action candidate limits, hydration and final context budgets; report actual consumption and frontier exhaustion rather than fabricate actions or edges to fill a budget. Its final selector is the same as the acquisition arm's, isolating action choice. An additional all-random pipeline may combine B5-A and B5-S, but cannot replace either controlled contrast. Conditioning B5-S on a method-discovered pool measures selection only; it cannot establish acquisition superiority.

Preregister 20 random draws per question using stable query-ID-derived seeds and one published seed list; do not choose a favorable draw. Report the random mean, spread and method-minus-random paired differences, including support recovered beyond hop-0. Keep random draws and repeated reader runs nested within each question in statistical resampling; they do not multiply the question count. Random-baseline execution and reader calls belong in the existing campaign budget, so reduce nominated arms if necessary. Report its genuine runtime, which may be much cheaper; budget matching does not authorize padding its compute cost.

Exploratory arms may fail this floor and remain published negative results. No arm may be retained or promoted as an improvement unless it passes the random-baseline gate in Section 11.2. This is a statistical claim on the declared evaluation population, not a promise to win on every query or every random draw.

## 9. Corpora, splits, and source integrity

Primary multi-hop tasks: MuSiQue, HotpotQA, 2WikiMultiHopQA. PopQA is a single-hop control included in a separately reported four-dataset macro. Evaluate the remaining Epic 1 datasets as transfer/acceptance tasks. Add anonymized and >4-hop challenge sets only with verified corpus, licenses, provenance, and available labels. The [MuSiQue paper](https://aclanthology.org/2022.tacl-1.31/) motivates testing genuine composition and shortcuts; [Youtu-GraphRAG](https://arxiv.org/html/2508.19855v3) motivates anonymized evaluation. Gold decompositions are diagnostic supervision, never implicit inference inputs.

**Corpus construction:** freeze full corpus manifests independently of selected evaluation questions; include the benchmark's official distractor corpus when that is the defined task. Keep provided-20-passage, sliced-corpus, and full-corpus tracks distinct. Never assemble a purported full corpus by adding only chosen questions' gold documents. Deduplicate exact source identities without collapsing distinct versions or role-changing near-duplicates. Source hydration, one-indexed spans, and SHA-256 verification remain unchanged.

**Proposed sample plan:** per primary/control dataset, 200 training questions, 100 development questions, and at least 500 fresh final questions, subject to official split availability. Use grouped sampling by shared source/question family where feasible. Training algorithms may use only training labels. Choose final sample size using development-estimated paired variance and a documented 80% power calculation for the locked minimum effect; 500 is a floor, not a power guarantee. If labels cannot support a disjoint sufficiently powered final set, report an exploratory result and do not assert confirmatory superiority. Exclude all inspected weekend questions and semantic near-duplicates from final confirmation.

Hash question IDs, corpus IDs, splits, model files, prompts, code, parameters, index files, and metric implementations before final runs. Do not inspect final metrics while selecting arms. A revised system after unsealing needs another final holdout. Run cross-dataset transfer of the chosen policy without retuning to expose corpus-specific learning.

## 10. Budgets and operational constraints

**Independent backend axis:** [Epic 3 E3-P5](../epic-3-performance.md#321-e3-p5-backend-experiment-and-promotion-contract) owns SQLite vs. SQLite + ephemeral FAISS vs. embedded LanceDB. Keep this storage/acquisition substrate comparison separate from the inference hypotheses: exact/ANN and lexical/fusion controls, fixed action/candidate replay, then repeated-hop end-to-end runs. Changing the backend does not inherit the core same-index cost claim in Section 11.2; charge index construction, text duplication where applicable, update/rebuild, maintenance, freshness and all memory. Retain the random-after-seeds controls on every backend, preregister backend contrasts within the shared statistical/campaign budget, and freeze the chosen substrate before final inference confirmation. No backend swap without E3-P5's measured Pareto gate and a subsequent atomic specification/production cutover.

These are proposed caps to be ratified in an execution manifest before implementation. They are not measurements or current runtime defaults.

| Budget | Small | Primary | Extended diagnostic |
| --- | ---: | ---: | ---: |
| Hybrid search actions per question, including seed action | 4 | 12 | 32 |
| Maximum expansion depth | 2 | 4 | 8 |
| Unique hydrated candidate chunks | 64 | 256 | 512 |
| Transient outgoing associations retained per node | 8 | 16 | 32 |
| Evidence-set beam | 2 | 4 | 8 |
| Final context tokens, including titles/provenance | 2,048 | 4,096 | 8,192 |
| Lane A query embedding inputs | 1 | 1 | 1 |
| Lane B contextual query embedding inputs | 4 | 12 | 32 |
| Diffusion/message updates per frozen graph | 2 | 4 | 8 |
| Sheaf stalk dimension `d` | 2 | 4 | 8 |
| Distinct branch/facet channels `f` | 1 | 2 | 4 |
| CP tensor rank `r` | 4 | 8 | 16 |

A hybrid action contains at most one FTS request and one vector probe. Log both individually. Count embedding inputs, HTTP requests, tokens, vector dimensions, and actual vectors/documents scored; batching does not make input work disappear. Lane B sentence-feature inputs have a separate cap of 64 strings per question and 64 per batch, with total tokens charged. If required processing exceeds a cap, terminate that arm with its explicit budget state rather than silently expanding it.

At most 32 development configurations per method family, at most two nominated confirmatory systems, and at most three stochastic seeds for training/QA are proposed. Before running a campaign, estimate wall time and generation/embedding tokens from the development pilot, set an explicit campaign ceiling, and record it. This paper alone supplies no permission to launch an unbounded optimization or multi-day benchmark.

The tensor/diffusion additions do not expand that campaign allowance. D0-D3 and T1 draw from the same family budget; D4 is conditional. Hold node feature width/channel access constant for comparisons that claim transport-only gains, and use parameter-matched controls for learned maps. Diffusion updates are not new retrieval hops: only an actual index action can discover a new corpus candidate. Compare inference wall-time curves as well as equal update counts, since tensor updates are more expensive than scalar steps.

Use the same consumer hardware, model precision, native DSPy adapter limits, reader prompt, reader context limit, and requested output length. Keep local generation serialized and embedding batches <=64. Freeze the accepted baseline's sampling settings; do not change temperature while attributing gains to retrieval. A second reader is a robustness check, not a replacement for a weak matched-reader result.

Feature formation is charged even when performed after retrieval. Offline per-chunk sentence embeddings, global kNN adjacency, corpus-wide noun maps, or contextualized GNN embeddings each define a separate indexing arm and invalidate the identical-index comparison. Uncertainty about cheap parsing cost is measured, not assumed away.

## 11. Metrics, statistical analysis, and Pareto accounting

### 11.1 Quality and mechanism metrics

- Recall@{1,2,5,10,20} using the benchmark's original passage/document definition; publish chunk-to-document mapping and MaxP/span-dedup policy.
- **Full-support recall@N:** fraction of questions whose entire annotated supporting set is recovered. Report the limitation that benchmark labels may omit valid alternative evidence.
- Token-budgeted full-support recall, support precision/F1, and coverage versus consumed context.
- Gold support in all acquired candidates; complete support in candidates; selected-support recall conditional on candidate availability. These separate discovery and selection failures.
- New support recovered beyond the identical hop-0 baseline per search, embedding token, and hydrated token. Gold labels score actions offline only.
- Answer EM/F1 under an identical reader; official normalization and answer variants are frozen. No-context and gold-context controls distinguish memory, retrieval, and reading ceilings.
- Provenance validity, returned-connector completeness, source-span validity, invalid-dependency rate, and DAG validity on original identities. Automated span checks do not replace entailment assessment.
- Stop-too-early rate on answerable questions, budget exhaustion, insufficient-evidence decisions, and contradiction handling. Evaluate support removal and distractor perturbations separately from natural tasks.
- For diffusion/tensors: bridge-changing versus same-topic support recall, component collapse/oversmoothing across depth, fixed-operator energy and norm traces, map type/rank, sparse operator footprint, map-prediction cost, and held-out relation-direction accuracy. These diagnostics do not replace retrieval/QA endpoints.

For relation-dependent explanation quality, use a fixed stratified 100-question audit sample, two blinded annotators with disagreements adjudicated, and a published rubric: source identity, bridge occurrence, actual action provenance, relation direction, and adequacy of the returned evidence bundle. If annotation cannot be supplied, report mechanical provenance validity only. Synthetic tasks test invariants and cannot establish benchmark quality.

### 11.2 Confirmatory thresholds

Lock these proposed margins before final evaluation; revisions require a dated new protocol while final labels remain sealed.

**Mandatory random-baseline floor:** every arm proposed for retention must beat B5-S on multi-hop macro full-support recall@5 and R@5, with positive multiplicity-adjusted simultaneous confidence lower bounds for paired differences. Acquisition arms must also pass against B5-A. For end-to-end QA retention, require positive bounds for EM and F1 against the corresponding random controls under the matched reader. A retrieval-only result cannot claim QA improvement without this evaluation. Apply the existing per-dataset and other-N noninferiority guardrails; preregister random contrasts within the same Section 11 family. Failure or inconclusive superiority blocks retention regardless of beating hop-0 or a weak deterministic comparator. Passing this floor does not replace the stronger mechanism/GraphRAG thresholds below.

1. **Mechanism promotion:** against the strongest development-selected simple control, at least +.03 absolute multi-hop macro full-support recall@5 and +.02 R@5, with simultaneous confidence lower bounds above zero. Require no material loss in QA. Apply on full fixed corpora, not weekend slices.
2. **Graph-system superiority:** positive simultaneous confidence lower bounds for multi-hop macro R@5, EM, and F1 against each nominated reproduced graph baseline; point gains at least +.02, +.01, and +.01 respectively. Full-support recall must improve. At R@1/2/10/20 and on each dataset, require confidence-supported noninferiority within .01 absolute; this is a broad-quality guardrail, not a claim of strict improvement at every N.
3. **SOTA claim:** separately enumerate the current comparable systems and official protocols before sealing. Exceed their verified comparable primary metrics, or explicitly limit the result to reproduced competitors. Strict improvement at every reported N requires positive bounds at every N; do not turn the guardrail in item 2 into that stronger claim.
4. **Cost claim:** primary graphless indexing has zero generative calls and the same persistent index. For strict matched-quality Pareto domination, upper confidence bounds on p95 retrieval latency and peak working memory ratios are <=1.05, with at least one declared cost dimension improved by >=20%. If quality rises at higher query cost, report a constrained frontier improvement instead of domination.

Compute per-question paired differences; use dataset-stratified, question-family-clustered bootstrap intervals (10,000 resamples, fixed analysis seed). Resample questions/families, not individual gold passages. Repeated reader seeds remain nested within the question; they are not extra independent questions. Apply Holm correction at familywise alpha .05 across preregistered promotion contrasts/endpoints; provide adjusted tests and compatible simultaneous bounds. Report both the three-task multi-hop macro and four-task control-inclusive macro, each dataset, all seed variability, and unrounded per-question results. Null results remain published.

If runtime failures occur, stop the campaign and preserve the original exception. Never score a partial run as a completed paired comparison or silently drop failed questions. After repair, rerun the whole affected locked paired block; charge failed work in operational accounting. Source absence, malformed labels, and unsupported task variants are adjudicated before unsealing.

### 11.3 True workload cost

For method `m`, report:

\[
C_m(Q,U)=C_{index,m}+C_{train,m}+Q C_{query,m}+U C_{update,m}.
\]

Keep money, CPU/GPU time, energy where measurable, and bytes as separate dimensions rather than inventing a single conversion. Plot query count `Q` and update count `U`; calculate break-even only where denominators and assumptions permit it. Include model load, cold caches, warm caches, index creation, source edits/deletes, and repeated query families. A persistent graph may win on a static corpus with many repeated queries. The proposal must allow that outcome.

## 12. Staged experiments and stop conditions

| Stage | Work and required artifacts | Gate / stop condition |
| --- | --- | --- |
| E0: freeze | Pin accepted baseline, source hashes, official task definitions, split manifest, model/adapters, metric checks, paper/competitor versions, B5-S/B5-A eligibility and budget rules, and 20 random seeds | No quality campaign until baseline and independent evaluation contract are reproducible |
| E1: locate ceiling | Full-corpus seed/candidate traces; gold-node, gold-edge, gold-terminal, and gold-context oracles, each clearly labeled | If complete candidate support is rare, prioritize discovery; do not optimize a nonexistent path |
| E2: isolate inference | On one frozen candidate graph per question compare terminal-only, forward, MAP, beam, Gibbs smoothing, constant terminal, and shuffled edges | Reject H1 if equal-feature simple controls match it; brute-force tiny-graph checks must match DP arithmetic |
| E2-D: diffusion and tensor controls | Same candidates/features/readout: D0 scalar, D1 identity, D2 diagonal transport, D3 orthogonal/dense only if needed; isolate T1 scoring and test shuffled/oracle maps | Reject H8/H9 if equal-capacity simple controls match them; fixed-map PSD/stability checks must pass; no budget expansion |
| E3: acquire | Compare original-query repetition, diverse/random bridge, AIR-style residual, and nominated path/diffusion-guided actions at equal budgets | Promote only if new complete support per unit cost improves; no full-corpus side scans |
| E4: select | Same acquired candidates/features; compare MMR, coverage, raw multi-vector matching, residual volume, exact regularized logdet, and connector bundles | Reject H3 if gains disappear with equal representation or relation-sensitive near-duplicates |
| E5: join or learn | Only if error analysis shows unresolved complementary/relational failures: bounded factors, MLP/set scorer, GNN, shuffled topology, cross-dataset transfer | No topology claim without beating equal-feature non-graph models; reject excessive state/rank growth |
| E6: confirm | At most two nominated systems; sealed full-corpus holdout, reproduced graph competitors, matched-reader and official-protocol tracks, statistics and cost curves | Apply Section 11 thresholds; publish nulls and scope limits |
| E7: promote | Reconcile governing docs, remove losing experimental paths, wire runtime/transport/watchdog, run real integration and acceptance gates | No cutover completion or checkpoint tick based solely on this research paper |

Stages E2-E5, including E2-D, use development data for selection; learned weights use training labels as specified in Section 9. H8/H9 use the same Section 11 mechanism-promotion margins and multiple-comparison family as other nominated mechanisms. Register the expanded contrast family before unsealing; do not add tensor/sheaf arms after seeing final results. There is no obligation to execute all stages or combine all mechanisms. The final system should retain the smallest mechanism set supported by evidence.

At every E2-E5 development stage apply B5-S, and B5-A wherever acquisition differs. Eliminate random-equivalent arms before expensive confirmation. E6 repeats the required random contrasts on the sealed holdout; development wins alone cannot satisfy the retention gate.

## 13. Artifact and implementation contract for later execution

Write generated indexes, traces, and results only under ignored `outputs/epic3-fts-vec/<run_id>/`. A run manifest must include source commit and dirty diff hash; model and corpus hashes; index/memo identity; split and metric versions; hardware, OS, numerical library threads; configuration, seed, clock basis, cache state, and budgets. Preserve the persistent content-addressed CocoIndex memo independently of transient benchmark DB names.

Required outputs: `manifest.json`, `candidate_traces.jsonl`, `action_costs.jsonl`, `evidence_edges.jsonl`, `per_query_metrics.jsonl`, `summary.json`, `paired_statistics.json`, `failures.jsonl`, and `resource_profile.json`. Include model-input/output hashes where source confidentiality prevents raw logs. Evidence edges store parent/child IDs, exact bridge span coordinates and hash, action provenance, association type, and whether a separate semantic relation check was performed. Unchecked associations are labeled accordingly.

For random controls, record seed IDs/order, eligible-pool/action hashes, sampled chunk/action IDs and RNG seed per draw, actual budgets, per-draw metrics, draw aggregates and paired random contrasts. These traces must distinguish random selection on a frozen pool from random acquisition on an evolving frontier.

Before implementation, write real arithmetic/structural checks: exhaustive finite-path enumeration versus DP on tiny graphs; mass conservation of stop probabilities; Viterbi route consistency; immediate backtracking versus longer-cycle counterexamples; duplicate-evidence invariance; reference multiplicity sensitivity; determinant-lemma equality; budget accounting; and source-hash failure propagation. Use real SQLite/vec0/FTS and actual source files for index-action integration. Remote fake endpoints establish orchestration only. Final quality uses real embeddings and readers.

For E2-D also require the sheaf energy/`delta^T W delta` identity, PSD symmetry for nonnegative weights, identity-map reduction to ordinary vector Laplacian diffusion, fixed-step energy decrease for the seeded objective under the stated bound, and a counterexample showing that recomputed maps do not inherit that fixed-operator guarantee. Verify CP contraction against the corresponding explicit small tensor; distinguish input-projection time from contraction time. Confirm numerical bidirectionality never creates a factual inverse edge in the output. Unit arithmetic cannot establish the semantic utility of learned maps.

No dependency or persistence schema change is authorized by this document. Production promotion must satisfy the then-current architecture contract. Research variants belong in an isolated experiment harness, not dormant production feature flags. At promotion remove superseded algorithms and update all callers and telemetry atomically.

## 14. Recommendation and expected outcomes

The first ambitious comparison should put **soft-terminal Gibbs path inference and query-conditioned diffusion against a strong evidence-set beam on identical candidates and features**. Within diffusion, scalar and identity-map controls precede diagonal sheaf transport; low-rank tensor scoring is an independent axis, not a mandatory addition. This directly tests delayed bridge credit and whether semantic components need different transport across topic-changing evidence connections. Hard endpoint-constrained transport remains a second-tier diagnostic because the endpoints are unknown. Residual geometry earns a separate experiment. Higher-order joins and general learned GNNs remain contingent on observed errors.

The most attractive outcome is that global conditioning improves complete-support recovery while ordinary indexing remains unchanged. A second useful outcome is that learned acquisition or multi-vector matching supplies the gain and the path machinery can be discarded. A third is that source relations require more expensive semantic discrimination; that establishes a cost boundary rather than a reason to hide extra LLM work. A fourth is that persistent graphs dominate at high query reuse. Each result resolves a real engineering decision.

The proposed frontier is therefore measurable: how much query-specific relational structure can be reconstructed from source evidence and one ordinary index, with how little total work, before a compiled graph or a stronger reader becomes the better investment?

## 15. Paper archive and reading order

The [collection README](../papers/epic-3-fts-vec-graph-experiment/README.md) links every downloaded PDF. The [manifest](../papers/epic-3-fts-vec-graph-experiment/manifest.json) records exact versions, URLs, hashes, sizes, page counts, and verification times; [sources.json](../papers/epic-3-fts-vec-graph-experiment/sources.json) records the intended collection. An archived arXiv preprint is not automatically identical to a later conference version cited in Section 3.

| Read order | Local arXiv papers | Purpose |
| --- | --- | --- |
| 1 | [AIR](../papers/epic-3-fts-vec-graph-experiment/2005.01218v1-air.pdf), [GoldEn](../papers/epic-3-fts-vec-graph-experiment/1910.07000v1-golden-retriever.pdf), [MDR](../papers/epic-3-fts-vec-graph-experiment/2009.12756v2-multi-hop-dense-retrieval.pdf), [Beam Retrieval](../papers/epic-3-fts-vec-graph-experiment/2308.08973v2-beam-retrieval.pdf) | Strong simple and learned alternatives before inventing a new algorithm |
| 2 | [Schrodinger survey](../papers/epic-3-fts-vec-graph-experiment/1308.0215v1-schrodinger-survey.pdf), [cost-augmented bridges](../papers/epic-3-fts-vec-graph-experiment/2610.02195v1-cost-augmented-schrodinger.pdf), [NBFNet](../papers/epic-3-fts-vec-graph-experiment/2106.06935v1-neural-bellman-ford.pdf) | Mathematical scope and exactness limits |
| 3 | [Subspace operations](../papers/epic-3-fts-vec-graph-experiment/2210.13034v4-subspace-set-operations.pdf), [ColBERTv2](../papers/epic-3-fts-vec-graph-experiment/2112.01488v3-colbertv2.pdf), [adaptive submodularity](../papers/epic-3-fts-vec-graph-experiment/1003.3967v5-adaptive-submodularity.pdf) | Representation, selection, and limits of greedy guarantees |
| 4 | [EAGR](../papers/epic-3-fts-vec-graph-experiment/2611.01910v1-eagr.pdf), [QueryFlow](../papers/epic-3-fts-vec-graph-experiment/2611.04625v1-queryflow.pdf), [GraphFlow](../papers/epic-3-fts-vec-graph-experiment/2510.16582v1-graphflow.pdf) | Contemporary close approaches and novelty challenge |
| 5 | [HippoRAG 2](../papers/epic-3-fts-vec-graph-experiment/2502.14802v1-hipporag2.pdf), [Youtu-GraphRAG](../papers/epic-3-fts-vec-graph-experiment/2508.19855v3-youtu-graphrag.pdf), [MuSiQue](../papers/epic-3-fts-vec-graph-experiment/2108.00573v3-musique.pdf) | Competitors, task definitions, and shortcut-resistant evaluation |
| 6 | [Retrieval diffusion](../papers/epic-3-fts-vec-graph-experiment/1611.05113v3-efficient-retrieval-diffusion.pdf), [Neural Sheaf Diffusion](../papers/epic-3-fts-vec-graph-experiment/2202.04579v4-neural-sheaf-diffusion.pdf), [joint sheaf diffusion](../papers/epic-3-fts-vec-graph-experiment/2407.20597v1-joint-sheaf-diffusion.pdf), [polynomial sheaf diffusion](../papers/epic-3-fts-vec-graph-experiment/2512.00242v3-polynomial-sheaf-diffusion.pdf), [Tensorizing Neural Networks](../papers/epic-3-fts-vec-graph-experiment/1509.06569v2-tensorizing-neural-networks.pdf) | Transport, stability, map-learning limitations, low-cost controls, and tensor compression |

Additional archive entries cover Baleen, IRCoT, G-Retriever, GNN-RAG, GFM-RAG, DGCNN, non-backtracking GNNs, graph optimal transport, Bayesian experimental design, HippoRAG, and the original four rejected/background papers. Sources without an established arXiv copy remain primary publication links in the collection README; no arXiv identifier is guessed.

All 35 archived papers also have [readable Markdown copies](../papers/epic-3-fts-vec-graph-experiment/README.md), produced with xberg-cli 1.3.7 without token reduction or OCR. Version-matched arXiv HTML supplied 34 reading copies; EAGR used its original LaTeX source because its HTML endpoint returned 404 and native PDF extraction interleaved columns. Each copy identifies its source and conversion limits. The [Markdown manifest](../papers/epic-3-fts-vec-graph-experiment/markdown-manifest.json) records provenance and output hashes. Consult original PDFs for authoritative figures, tables, references and mathematical layout.

## 16. Review record and restart checklist

**2026-11-16, backend proposal reconciliation (`E3-P5-DOC1`).** Updated the governing [Epic 3 E3-P5 protocol](../epic-3-performance.md#321-e3-p5-backend-experiment-and-promotion-contract) with SQLite / SQLite + FAISS / LanceDB arms, exact versus ANN controls, native DSPy lifecycle costs, full-corpus confirmation and a measured Pareto gate. Replaced E3-P5's entity/fact/CSR cache dependencies with the accepted bare-bones chunk-index and atomic snapshot lifecycle; called out connection-local SQLite change detection. Updated the probe, checkpoint, fixture matrix and dated decision in the parent spec, and linked its storage axis here. This unit changes documentation only; no backend was installed, swapped or benchmarked, and no retrieval/runtime cutover is established. Validation and independent audit results follow after execution.

**E3-P5-DOC1 validation:** both targeted documentation architecture checks passed (`2 passed, 14 deselected`); all 143 relative file links across the two edited documents resolved and the new E3-P5 section anchor exists. `git diff --check` passed. Independent session-model critic audit reported no substantive findings. The mandated Flash model is unavailable through the current agent interface; no implementation checkpoint was ticked or commit made. No production checks or backend performance experiments were run; the unrelated in-progress bare-bones implementation was preserved.

**2026-11-16, random-after-seeds amendment (`E3-FVG-P3`).** User required experiments to outperform random candidate selection after seeds. Split the earlier combined random/diverse B5 entry into mandatory B5-S selection and B5-A acquisition controls; fixed seed/pool/budget contracts, 20 preregistered draws, nested statistical treatment, explicit quality retention gates, stage checks and replay artifacts. The governing [Epic 3 decision](../epic-3-performance.md#decisions) records the instruction. This is an evaluation-design amendment only; no new benchmark result or guarantee of per-query superiority is established.

**P3 validation:** independent session-model review identified incompatible exact-count/token-cap wording and a degenerate top-5 comparison if all five seed positions were fixed. Clarified count-matched retrieval prefixes versus deterministic token-capped reader packing, prohibited favorable redraws, and required a preregistered selectable top-5 suffix. Independent re-audit reported no further findings. Both targeted documentation architecture checks passed (`2 passed, 14 deselected`), and `git diff --check` passed. No production benchmark, full code suite, commit or implementation checkpoint tick was performed.

**2026-11-16, tensor/diffusion amendment (`E3-FVG-P1`, proposed).** User requested updating the linked experiment collection after discussing tensor math and diffusion. Added H8/H9, scalar/vector/sheaf diffusion and low-rank tensor scorer contracts, fixed-operator stability and directionality limits, capacity/map controls, proposed rank/update budgets, E2-D, and five supporting arXiv papers. This amends the research proposal only; it does not restore the weekend's full tensor/Kraus architecture or establish a performance gain. The earlier P0 archive counts and verification below remain a historical record; current collection totals and amendment checks are recorded separately after validation.

**2026-11-16, readable-paper authoring unit (`E3-FVG-P2`).** Applied the user-invoked xberg format-selection workflow to all 35 papers. Selected Markdown content in JSON extraction envelopes for metadata retention. Native PDF samples scrambled two-column reading order, so 34 conversions use version-matched HTML with embedded TeX preserved, and EAGR uses expanded source LaTeX with discarded source macros excluded. Conversion headers and the Markdown manifest disclose limitations; originals are preserved. The collection now contains 35 PDFs, 846 pages and 73,472,212 PDF bytes, plus 35 linked Markdown reading copies. Exact PDF and Markdown hashes, version coverage, PDF headers and Poppler page counts were checked for every entry; proposal and collection README links resolve. Both targeted documentation architecture checks passed (`2 passed, 14 deselected`). Numerical checks passed for the original 30 stopped paths and 20 determinant-lemma cases, plus 20 seeded sheaf-energy/PSD/gradient and CP-factorization cases and identity-map reduction. These checks establish artifact integrity and arithmetic examples, not exhaustive transcription fidelity, retrieval gains or a completed implementation checkpoint. No production code or dependencies were changed, and no full production suite or retrieval benchmark was run.

**P1/P2 independent review:** no further substantive P1 findings. The critic independently matched all 35 PDF hashes and all 35 Markdown hashes, and identified double-escaped TeX commands in EAGR's inline math. Removed one extractor-added escaping layer inside math delimiters, regenerated its manifest record, and repeated the collection checks successfully. The conversion header records this correction. Independent re-audit reported no further findings and confirmed the corrected inline TeX, sampled display equations and updated output hash. `git diff --check` passed. As for P0, this is a session-model review; the prescribed Flash model is unavailable, and no commit or implementation checkpoint completion is claimed.

**2026-11-16, proposal authoring unit.** Read the governing foundation, current Epic 3 and bare-bones contracts, weekend reviews, retrieval source, probe/slice code, and saved benchmark/ablation artifacts. Searched supporting and competing primary literature. Added this proposal and a version-pinned local paper archive. This establishes a research design and auditable bibliography; it does not establish a new algorithm's quality, an accepted execution budget, a completed cutover, or a live process state. Validation and independent review results are appended below when performed.

**Independent review:** the session's available critic reviewed `E3-FVG-P0` and found terminal/path-sufficiency ambiguity and a training/development split inconsistency. Both were corrected in Section 6.2. The requested `flash` model is not exposed by the available agent tool; this independent session-model audit is not represented as the mandated Flash commit audit. No implementation checkpoint is ticked and no commit is made by this authoring unit.

**Validation performed:** the independent re-audit reported no further substantive findings. Both targeted documentation architecture checks passed (`2 passed, 14 deselected`); all relative links in this proposal and the archive README resolved; all 30 requested arXiv versions matched local manifest hashes and passed PDF-header/Poppler parsing checks. The archive contains 52,146,737 bytes and 723 pages. An ignored numerical verification script, `outputs/verify_epic3_math.py`, matched the forward/backward recurrence and stop/edge marginals against all 30 finite stopped paths in its two-state, three-transition example, and verified the determinant-lemma identity on 20 seeded numerical cases. `git diff --check` passed. These are documentation, archive-integrity, and arithmetic checks; no new retrieval/QA benchmark, full production test suite, or live-process acceptance was run.

Source fingerprints at authoring (SHA-256):

| Source | SHA-256 |
| --- | --- |
| `yaw/daemon/graph/hmm.py` | `a6dd9a6d93b9abf463057e79be09bd2f96affd80ab219166b7152ccc0302ed7b` |
| `scripts/probe_manifold.py` | `399f68f2732390bd50b62afe175788333f357580f3248fc113c903a190ee4c7b` |
| `yaw/daemon/ingestion/datasets.py` | `6e772170c707778a33e13bb308f3c5d86e5f408e67f55e10e51a18a316526ec6` |
| `manifold_ablations_4ds_10q.json` | `8457e4e31dd011c2064afd30668d51eb3b8b8fdd8346f7907fbf53c55708171f` |
| `probe_manifold_pass4_v3.json` | `4634efb67a1bdc5d2dc960915cd9e4f0f36b7c29b3d0b5845c9275e4d59f8f3e` |
| `shootout_pass4_v3_4ds_10q.json` | `623dfce9a2b63fc0e482774b4877ae1fbca3aed8745d85494beb300832b60780` |
| `pooled_distractors_v3_probe.json` | `9af342f4287af88790cb26345e6dc3f4dbc2b81e89813bcbce774cd93cb68ecb` |

Restart in this order:

- Verify accepted cutover state and baseline commit; the source audit above is historical once code changes.
- Resolve the proposed execution budgets, split availability, statistical power, and competitor versions in E0.
- Freeze full corpora independently of selected questions and keep weekend questions out of confirmation.
- Run oracle ceiling diagnostics before selecting an expensive mechanism.
- Preregister at most two final systems, thresholds, analysis family, and total campaign ceiling before unsealing.
- Promote only after independent critic review and the real runtime, telemetry, transport, and full-corpus gates.
