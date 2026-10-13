# Query-time Personalized PageRank

**Status:** Target retrieval specification; runtime conformance pending [Epic 7.8](epic7.8-open-ie-ontology-alignment.md).\
**Contract:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-012 as amended](adr/ADR-012.md), [ADR-014](adr/ADR-014.md).\
**Executable owner:** `src/yaw/retrieval/retriever.py`; one structured `GraphRAGRetriever.retrieve` result serves MCP, CLI and evaluation. Numerical settings live only in `src/yaw/config.py` / `config.toml`.

## Graph and conductance

Build a consistent versioned snapshot from active node/edge projections, including source occurrences, admitted domain resources and provenance-labelled associations. Do not use database-file modification time alone as a complete semantic cache key, particularly with WAL; track committed data/resource/projection generations.

Domain-specific property traits and membership come from semantic resources. Generic traversal policy can assign different upward/downward, sequential, wikilink and candidate-link weights. Configuration owns numerical weights. Never infer a property's semantics solely by stripping its prefix and comparing its local name.

Reverse navigation edges and candidate associations are retrieval structures; exporting them as logical assertions requires independent semantic justification. Human wikilinks and source-grounded semantic edges have equal baseline importance before explicit traversal policy.

The selected upward policy is `w_up(u,v) = max(w_floor, 1 / (1 + log2(1 + depth(v))))`, initially `w_floor=0.2`. Depth is the shortest root-to-node distance in the current admitted organizational/taxonomic traversal hierarchy, roots at zero. Compute depth on the directed acyclic condensation of hierarchy components; each component without incoming parent edges is a root, and members of a cycle share its component depth with an explicit cycle diagnostic. An isolated node has depth zero. Non-hierarchy edges do not use this policy. Downward hierarchy conductance is 1 before the common resource/candidate weights. This is a selected ranking heuristic, not an optimality or truth claim.

One `compute_upward_conductance` implementation uses that policy. Do not also store flat/linear upward multipliers on leaf edges or apply a second degree penalty. Offline evaluation can measure alternatives; the production graph has no algorithm-selection or recovery branch.

For nonnegative adjacency A, let D(u) = sum_v A(u,v) and P(u,v) = A(u,v)/D(u) for non-dangling rows. Dangling rows remain zero and their mass is redistributed to personalization during iteration.

## Seed channels

Use three channel families:

1. Alias candidates from the linguistic/KB index, combined with query-to-fact relevance.
2. Query-to-chunk relevance in space R.
3. Lexical FTS relevance.

For query mentions M(q), define alias mass:

$$s_{alias}(e) = \sum_{a \in M(q)} w(e \mid a)$$

Here w is normalized nonnegative candidate evidence within a mention's candidate set. It is not a calibrated probability of identity. Label equality alone must not collapse the candidate set.

Let F_q be the selected active, qualified facts retrieved in space R. For a node e, let r(q,e) be the mean nonnegative cosine relevance of facts in F_q that refer to e, or zero if none. Facts are rendered with their qualifications and semantic roles; embeddings must not erase the difference between positive and negated statements.

Combine L1-normalized nonempty alias and fact channels:

$$s_{concepts} = \lambda \widehat{s}_{alias} + (1-\lambda)\widehat{r}$$

If either channel has legitimately empty data, renormalize the remaining nonempty weights. A failed channel is an error, not empty data. At the outer level:

$$p_0 = \beta_1 \widehat{s}_{concepts} + \beta_2 \widehat{s}_{chunks,vec} + \beta_3 \widehat{s}_{chunks,fts}$$

Again, normalize active weights and the resulting mass. Clip negative cosine contributions to zero for probability-mass construction. The single FTS formatter in the retriever orders eligible unique occurrences by SQLite BM25 ascending, then occurrence ID; assign **`1 / rank` with one-based rank**, then L1-normalize. Three hits therefore contribute weights `6/11, 3/11, 2/11` before outer mixing. SQLite assigns better matches lower BM25 values; do not insert raw signed scores into p_0. [SQLite FTS5 BM25](https://sqlite.org/fts5.html#the_bm25_function).

Reject nonfinite/negative channel weights, all-zero configured outer weights, and nonpositive convergence/iteration limits at configuration validation. After omitting legitimately empty channels, if no positive configured weight remains, return complete/no evidence; do not turn on a disabled channel or invent uniform seeds. A disabled channel has weight zero and is not an outage recovery branch.

Space-S identity candidate vectors never serve as space-R query vectors. Retain additive alias seeding so a query-mentioned node can contribute even if absent from the selected fact set. The configured embedding model/index is required; unavailability or incompatible dimensions produces a typed query error. If every successfully computed channel is empty, return an explicit no-evidence result instead of inventing seed mass over unrelated nodes. There is no lexical-only outage mode.

## Power iteration

For continuation probability 0 < alpha < 1:

$$p^{(t+1)} = \alpha(P^T p^{(t)} + m_{dangling}p_0) + (1-\alpha)p_0$$

where m_dangling is the current probability mass in dangling rows. Start from p_0, preserve nonnegative finite mass, and stop at the configured L1 residual tolerance. Reaching the iteration limit without convergence is an explicit query failure, not a reason to select another solver/scorer. Do not advertise an unverified fixed iteration count or universal latency.

The result ranks graph relevance. It can help resolve an ambiguous query in context but does not prove identity, factual truth or logical entailment and must not write permanent merges.

## Chunk ranking and evidence

Build the candidate pool as the union of eligible vector hits, FTS hits and PPR-supported chunks, deduplicated by source occurrence ID. Do not discard dense/lexical candidates merely because they were absent from the top graph list. Apply this single versioned score to every candidate:

$$Score(c) = (p^*(c) + \sum_{e \in E_c} w_c(e)p^*(e))\,rel(q,c)$$

E_c contains distinct nodes supported by occurrences in c. To avoid an undefined alias a or accidental duplicate counting, w_c(e) is the maximum normalized candidate weight for e over the chunk's mention occurrences; the aggregation is part of the scorer version. rel(q,c) is max(0, cosine(q,c)) for the required compatible retrieval vectors. Missing required embeddings fail; there is no alternate formula. Sort descending by score, then stable occurrence ID for ties. Treat this score as a ranking heuristic and evaluate it, not as a Bayesian posterior.

Select evidence-bearing chunks, construct typed/provenanced connecting paths, then hydrate exact occurrences only after source-revision verification. For selected claims, include their current material conflict/supersession counterparts as explicit related evidence. Keep the ranked passage list separate from those relationship expansions so Recall@k retains a clear meaning. If the context budget prevents including both sides, return the omitted IDs and a truncation/coverage indicator; never present the selected side as uncontested.

Return structured passage/source/occurrence IDs, ranks/scores, evidence spans, qualified claims, typed paths, stored synthesis, source/compiled generations and completeness. Render Markdown only after this result is built. [The compiler guide](knowledge-compilation.md#6-publication-freshness-and-retraction) owns pending/failed scope handling. Stale source evidence cannot be silently replaced; an abstract path alone is not a grounded answer.

## Validation and performance

Test mass conservation, dangling rows, disconnected graphs, legitimately empty channels, explicit embedding/convergence failure, representation separation, unfamiliar property behavior, union candidate retention, counterevidence and source revision changes. Use actual sparse matrices. Fault injection must show that no alternate scorer or solver was invoked.

Report graph size, nonzero edges, iterations, memory, hardware and latency distributions. Per-iteration sparse traversal costs depend on edges and vertices; serializing k findings/results depends on k. Numerical parameters are read from configuration and calibrated only on the designated split. Report query-to-fact/candidate-link/alias-channel ablations on held-out data.

[HippoRAG 2](https://arxiv.org/html/2502.14802v2) motivates evaluating fact-based seeding; its measured gain is not a YAW performance guarantee. [Research extensions](scipy-graphrag-extensions.md) and [the PPR survey](efficient-ppr-algorithms.md) are references, not competing runtime contracts.
