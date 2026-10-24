# SKOS / RDF / OWL / SHACL hardening

Status: planned refactoring epic, 2026-10-24. No runtime changes or quality gains
are claimed by this outline. Owner contract: [FOUNDATIONAL](../FOUNDATIONAL.md).
Execution remains within Epic 1 and its Tier 1 development / Tier 2 acceptance
gates; this plan does not bypass unfinished Epic 1 packets.

## Goal and boundaries

Preserve relation identity and direction from extraction through SQLite to graph
propagation. Separate entity references from literal values. Make validation,
identity resolution and retrieval weighting explicit, independently testable
policies. Pursue all three workstreams: typed literals, SKOS hierarchy and
relation-aware propagation, with separate measurements so their effects remain
attributable.

Retain one yaw.db, the single writer, native DSPy, CocoIndex and sparse CSR.
Do not introduce rdflib, OWL reasoners, pySHACL, spaCy or a parallel graph store.
Preserve user prompts and experiments. Do not start an unattended optimization.
Before implementation, reconcile the accepted schema and traversal contracts in
FOUNDATIONAL and Epic 1; this outline describes proposed changes, not current
conformance.

## Verified starting point

- `facts` ([`yaw/daemon/graph/storage/db.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py#L77-L85)) stores the extracted predicate (`fact_id TEXT PRIMARY KEY, subj_id TEXT NOT NULL, pred TEXT NOT NULL, obj_id TEXT NOT NULL`), requiring an entity ID at both ends. [`StructuredTriple`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/core/types.py#L52-L79) has no explicit literal type. Endpoint validation ([`shacl.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/shacl.py#L230-L261), [`shacl.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/shacl.py#L530-L548)), orphan garbage collection ([`delete_chunks` in `graph_ops.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1323-L1339)), and several fact readers assume that both ends join to `entities`.
- `edges.kind` ([`db.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py#L99-L105)) distinguishes `passage`, `fact`, `skos:related` and `owl:differentFrom`. Related edges can already have different supplied weights from passage edges; `owl:differentFrom` is stored with zero traversal weight.
- Parallel predicates and directionality currently collapse in **four layers** upstream of and inside [`build_transition_matrix`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L73-L185):
  1. [`GraphEngine.index_chunks`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L1745-L1746) immediately appends both `(s_id, o_id, "fact", 1.0, cid)` and `(o_id, s_id, "fact", 1.0, cid)`, erasing asserted subject-to-object direction at write time.
  2. `edges` (`PRIMARY KEY (src_id, dst_id, kind)`) and `edge_sources` (`PRIMARY KEY (src_id, dst_id, kind, chunk_id)`) in [`db.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py#L99-L117) and [`upsert_edges_with_sources`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L920-L955) collapse distinct predicates `(u, p1, v)` and `(u, p2, v)` into a single `'fact'` kind (and into a single `edge_sources` row when extracted from the same `chunk_id`).
  3. [`load_graph_topology`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1105-L1125) executes `SELECT src_id, dst_id, MAX(weight) FROM edges WHERE src_id != dst_id AND weight > 0.0 AND kind != 'owl:differentFrom' GROUP BY src_id, dst_id`, stripping `kind` and collapsing `skos:related` and `fact` before [`ppr.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py) is called.
  4. [`build_transition_matrix`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L106-L142) ignores the relation tag, symmetrizes positive edges (`[u_arr, v_arr]` + `[v_arr, u_arr]`), and keeps only the maximum weight per ordered node pair (`np.lexsort((-data, keys))`). There is no `skos:broader` support today.
- Symmetric raw edge weights do not guarantee equal effective propagation in both directions: degree normalization depends on `beta`. The default `beta=0.5` uses a symmetric operator; other `beta` values can produce unequal effective entries. Default `scaled_tanh` propagation is not a conventional stochastic random walk, so its coefficients must not be reported as literal traversal probabilities.
- The operator is applied as `transition_t.dot(p)` in [`run_ppr`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L228): a directed contribution from source `u` to destination `v` belongs at matrix entry `adj[v, u]` (`row = v`, `col = u`). Today [`ppr.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L146-L175) computes `degrees = np.asarray(adj.sum(axis=1)).ravel()` (row sums) because `adj` is symmetric; for a directed operator, **out-degrees** of source `u` (`adj.indices`) are column sums `adj.sum(axis=0)` (`dangling_mask = out_degrees == 0.0`) and **in-degrees** of destination `v` (`row_indices`) are row sums `adj.sum(axis=1)`.

## Semantic and traversal contract

RDF predicates express relations; their meaning does not prescribe numerical
ranking weights. YAW must define and evaluate that retrieval policy explicitly.
Keep semantic assertions, structural incidence edges and the derived retrieval
operator distinct, with a documented mapping between them.

`rdf:predicate` is a statement-reification property identifying a statement's
predicate. It is not the generic name of every subject-to-object relation.
Extracted business predicates remain in facts; the internal `fact` edge kind
must not silently stand in for all their semantics.

| Relation / structure | Planned traversal policy |
| --- | --- |
| Passage/entity incidence | Explicit structural policy, initially symmetric |
| Extracted entity-object fact | Preserve predicate identity; initially use a shared fallback policy, with explicit inverse traversal settings |
| skos:related | Symmetric relation; configurable retrieval strength |
| skos:broader | Store child-to-parent assertion; configure upward and downward traversal strengths separately |
| skos:narrower | Normalize to the inverse broader assertion; do not double-count |
| owl:differentFrom | Identity constraint, excluded from positive traversal; no negative transition probabilities |
| Literal object | Preserve typed value and fact/source retrieval; do not mint an entity or automatic shared-value bridge |
| Labels, definitions and reification metadata | Available for matching/audit, not automatically traversable |

Define a small, versioned policy keyed by relation or explicitly configured
relation family, direction and structural role. Unknown OpenIE predicates use a
documented fallback; do not require an unbounded hand-maintained predicate list
or infer equivalence from similar spelling. Preserve asserted direction even
when a retrieval policy permits reverse traversal. Confidence and retrieval
strength have separate meanings and must not share one undocumented score.

## Ordered implementation packets

### H1 — Baseline and contracts

- Audit a bounded sample of value-like entities, hub degrees, predicate coverage,
  inference/embedding costs and numerical/date retrieval failures.
- Specify typed objects, directed relations, policy defaults, normalization and
  evidence ownership. Capture current ranking and resource measurements.
- Separate canonicalization/identity from hierarchy: a parent is not an alias.

Exit: reviewed contracts and reproducible baseline; no claimed savings from
unmeasured prevalence. This is the first unfinished packet.

### H2 — Typed literal extraction and durable storage

- Add an explicit entity-or-literal object representation in core types, DSPy
  signatures, GBNF, parsers and stage caches. Preserve original lexical form,
  datatype and optional language; retain units/context without inventing values.
  Do not use a definition sentinel. Do not classify solely by numeric regex.
- Apply the distinction to Stage 1 as well as triples so literal values do not
  continue through missing-definition, embedding or synonym stages as entities.
- Migrate facts with mutually exclusive entity/value fields and SQL invariants.
  Update identity keys, readers, FTS/vector text, serialization, merging,
  deletion, source bookkeeping and endpoint validation together.
- Keep fact-to-source/subject seeding usable when no object entity exists.
  Test dates, versions, ports, amounts and numeric-looking product names.
- Version extraction/cache identities. Re-extract ambiguous historical objects
  into a new compatible index; do not guess their types during migration.

Exit: real SQLite round trips and retrieval work for both object kinds; restart
reuse is verified; paired literal-only results are recorded.

### H3 — Directed hierarchy and deterministic validation

- Extend the existing pair judgment with explicit child and parent identities,
  keeping identity decisions separate. Retain existing candidate generation
  initially and measure its coverage and hierarchy classification precision.
- Add directed broader storage and provenance, migrations, serialization,
  merge/deletion handling, cache invalidation and bounded audit presentation.
- Check S27: related must not overlap broaderTransitive. Treat acyclicity as a
  YAW application rule, not a requirement imposed by SKOS.
- Use cycle-safe reachability without an arbitrary depth-eight correctness cap.
  Avoid eagerly materializing potentially quadratic transitive closure or adding
  every inferred ancestor to CSR. Bound memory and report incomplete checks
  explicitly rather than declaring them clean.
- Preserve conflicting evidence and resolve the interpretation; do not blindly
  delete related assertions because a new parent assertion appeared.
- Validate cardinality only for explicitly scoped property contracts. OWL RL
  prp-fp infers equality; it is not a generic conflicting-literal validator.

Exit: direction/provenance survives persistence and restart; cycles, deep paths,
S27 conflicts and source deletion have deterministic tests. Classification and
storage can be evaluated with hierarchy traversal disabled.

### H4 — Relation-aware sparse propagation

- Build sparse contributions using the versioned relation policy and [dst, src]
  orientation. Preserve present beta/activation choices as explicit settings;
  specify directed in/out-degree normalization rather than inheriting formulas
  that relied on an undirected adjacency matrix.
- Expose separate hierarchy upward/downward strengths. Calibrate them; 0.35 is
  a proposed trial value, not a standards-derived default.
- Deduplicate the same assertion, its inverse representation and repeated source
  evidence before aggregation. Combine distinct relation contributions with an
  explicit bounded policy; measure max versus sum/capped sum independently.
- Include policy identity in graph/operator caches, optimization/resume metadata
  and reports. Weight-only changes rebuild derived operators, not extraction;
  extraction contract changes invalidate the corresponding extraction stages.
- Verify linear-mode probability/mass invariants where normalization promises
  them; verify numerical stability and finite outputs for nonlinear modes.

Exit: tiny real sparse graphs prove direction, relation weights, parallel-edge
handling, zero-weight exclusions and normalization. Resource measurements show
bounded CSR construction and no full-corpus dense duplication.

### H5 — Paired evaluation and delivery

- Compare baseline, literals only, hierarchy stored without traversal, directional
  hierarchy traversal, and alternative aggregation policies as separate arms.
  Freeze other prompts, model identities, query splits and optimizer state.
- Tune on calibration queries; keep held-out answers out of extraction, learning
  and policy selection. Only then evaluate the combined configuration.
- Report Recall@1/2/5/10/20, QA EM/F1, per-query regressions, actual calls/tokens,
  indexing/retrieval latency, row/vector counts and peak RAM. Include numerical
  joins and broad-hub queries, plus historical and paper benchmark comparisons.
- Run fresh-process interruption/recovery checks with real SQLite and loopback
  inference; loopback validates orchestration, not model quality. Verify no
  repeated completed generation or duplicate learning/source updates.
- Record actual checks, failures and quality/resource differences in evaluation.md.
  Apply Tier 1/Tier 2 gates and report differences before choosing final weights;
  no invented degradation cutoff or guaranteed-gain claim.

Exit: reproducible evidence for retained ranking changes, documented limitations,
focused tests, preflight results and `git diff --check`; governing docs match the
implemented contract. Unproven ranking variants remain experimental.

## Codebase-Specific Engineering Specification & Watchouts for H1–H4 Execution

Following the `storage/` module decomposition ([`db.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py), [`graph_ops.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py), [`shacl.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/shacl.py), [`vectors.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/vectors.py)), **H1–H4** execution must enforce the following four concrete contracts across [`yaw/core/types.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/core/types.py), `yaw/daemon/graph/storage/`, [`yaw/daemon/graph/engine.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py), and [`yaw/daemon/graph/ppr.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py):

### 1. H2 Watchout: `facts` Uniqueness, Nullable `obj_id`, Explicit `fact_sources` Provenance & SHACL Gating
- **Current Schema & Upsert Contract:**
  - In [`yaw/daemon/graph/storage/db.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py#L77-L85), `facts` is currently keyed by a deterministic MD5 primary key `fact_id TEXT PRIMARY KEY` (`subj_id TEXT NOT NULL, pred TEXT NOT NULL, obj_id TEXT NOT NULL`), and [`upsert_facts`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L872-L901) deduplicates via `ON CONFLICT(fact_id) DO UPDATE`.
- **SQLite `NULL` Uniqueness & Mutual Exclusion Invariant:**
  - In standard SQLite `UNIQUE` constraints, `NULL != NULL`. When `obj_id` becomes nullable for literal-object facts (`obj_kind IN ('entity', 'literal')`), `facts` must enforce:
    ```sql
    CHECK (
        (obj_kind = 'entity' AND obj_id IS NOT NULL AND obj_literal_lex IS NULL)
        OR (obj_kind = 'literal' AND obj_id IS NULL AND obj_literal_lex IS NOT NULL)
    )
    ```
  - Compute `fact_id` deterministically from `(subj_id, pred, obj_kind, obj_id_or_literal_lex, obj_datatype, obj_lang)`, and if adding secondary unique indexes over `(subj_id, pred, ...)`, use either two partial unique indexes (`WHERE obj_id IS NOT NULL` vs. `WHERE obj_id IS NULL`) or a non-null `object_identity_key TEXT NOT NULL` column so re-indexing never inserts duplicate literal facts.
- **Orphan Fact Garbage Collection (`fact_sources`) & Entity Merge Re-Keying:**
  - Today, [`delete_chunks` (`graph_ops.py:1323-1339`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1323-L1339) garbage-collects orphaned `facts` by joining `facts` against `edges WHERE kind = 'fact' AND src_id = f.subj_id AND dst_id = f.obj_id`. Literal-object facts (`obj_id IS NULL`) emit no entity-to-entity `'fact'` edge in `edges` and would become immortal orphans on chunk deletion unless `facts` tracks explicit chunk provenance.
  - **H2 Requirement:** Add `fact_sources (fact_id TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (fact_id, chunk_id))` (indexed on `chunk_id`) so [`delete_chunks`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1323-L1339) prunes any `fact_id` with zero remaining rows in `fact_sources`, and update [`merge_entities` (`graph_ops.py:786-800`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L786-L800) so re-pointing `subj_id` or `obj_id` reconciles `fact_sources` and deduplicates colliding `(subj_id, pred, object_identity)` rows.
- **SHACL Closed-World Endpoint Validation (`shacl.py`):**
  - Update `StatementShape/closedWorld` in [`_scan_shacl_violations` (`shacl.py:230-261`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/shacl.py#L230-L261) and `Heal 5` in [`lint_graph_shacl` (`shacl.py:530-548`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/shacl.py#L530-L548) so `eo.entity_id IS NULL` is only a violation when `f.obj_kind = 'entity'` (`f.obj_id IS NOT NULL`), while literal-object facts validate `TRIM(COALESCE(f.obj_literal_lex, '')) != ''`.

### 2. H3 Watchout: Cycle-Safe Recursive CTE Termination Without `depth` in the `UNION` Key (`shacl.py`)
- **The SQLite Recursive `UNION` Trap:**
  - In SQLite, `WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (... UNION SELECT ...)` deduplicates rows across iterations and is guaranteed to terminate on any finite graph—even one with cycles—**if and only if** the projected columns come strictly from a finite domain (`(narrower_id, broader_id)` over `edges WHERE kind = 'skos:broader'`).
  - Projecting `depth + 1` inside the `UNION` (`SELECT bt.narrower_id, e.dst_id, bt.depth + 1`) breaks set-deduplication on cycles because `(u, v, 1)` and `(u, v, 4)` are distinct tuples.
- **H3 Requirement in [`yaw/daemon/graph/storage/shacl.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/shacl.py):**
  - Project **only** `(narrower_id, broader_id)` inside the recursive `UNION` CTE (with a row budget `LIMIT` guard to bound worst-case dense-subgraph work and report incomplete checks explicitly if the cap is hit):
    ```sql
    WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
        SELECT src_id, dst_id
        FROM edges
        WHERE kind = 'skos:broader'
        UNION
        SELECT bt.narrower_id, e.dst_id
        FROM broader_tc bt
        INNER JOIN edges e
          ON e.src_id = bt.broader_id
         AND e.kind = 'skos:broader'
        LIMIT :max_tc_pairs
    )
    ```
  - A hierarchy cycle is then any row in `broader_tc` where `narrower_id = broader_id`, and a **SKOS S27** conflict is any `(narrower_id, broader_id)` pair in `broader_tc` that overlaps `edges WHERE kind = 'skos:related'` (in either direction), with no arbitrary depth-8 cutoff.

### 3. H4 Watchout: Four-Layer Parallel-Fact & Direction Collapse Upstream of `ppr.py`
- **All Four Collapse Points Must Be Reconciled Together:**
  1. **Write-Time Symmetrization in [`GraphEngine.index_chunks` (`engine.py:1745-1746`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L1745-L1746):** Stop writing synthetic reverse `(o_id, s_id, "fact", 1.0, cid)` rows as if they were asserted forward facts; preserve asserted `subj_id -> obj_id` direction in storage and let the versioned traversal policy emit forward (`fact:fwd`) and inverse (`fact:rev`) contributions at operator construction time.
  2. **Same-Chunk Predicate Collision in `edges` / `edge_sources` ([`db.py:99-117`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py#L99-L117), [`graph_ops.py:920-955`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L920-L955)):** Because `edge_sources` keys on `(src_id, dst_id, kind, chunk_id)` with `kind = 'fact'`, two distinct predicates `(u, p1, v)` and `(u, p2, v)` in the same `chunk_id` currently collapse into one row. Derive fact-level relation contributions from `facts` + `fact_sources` (where `obj_id IS NOT NULL`) so distinct predicates `(u, p1, v)` and `(u, p2, v)` survive into [`load_graph_topology`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1105-L1125).
  3. **SQL `GROUP BY src_id, dst_id` Stripping in [`load_graph_topology` (`graph_ops.py:1115-1123`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1115-L1123):** Replace the relation-blind `SELECT src_id, dst_id, MAX(weight) FROM edges ... GROUP BY src_id, dst_id` query with relation-tagged directed edges `(src_id, dst_id, relation_kind, evidence_count_or_weight)` covering `passage`, `skos:related`, `skos:broader`, and distinct `(subj_id, obj_id, pred)` entity-object facts.
  4. **Undirected Symmetrization & Row-Sum Degree Normalization in [`build_transition_matrix` (`ppr.py:73-185`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L73-L185):**
     - Map each asserted relation `(u, v, kind, weight)` through the versioned `RelationTraversalPolicy` into directed `(src, dst, effective_weight)` transitions placed at CSR entry `adj[dst, src]` (`row = dst`, `col = src`) to match `graph.transition_t.dot(p)` in [`run_ppr` (`ppr.py:228`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L228).
     - Support configurable parallel-edge aggregation (`"max"`, `"sum"`, `"capped_sum"`) per ordered `(dst, src)` pair after deduplicating identical assertions.
     - Compute directed **out-degrees** from column sums `out_deg = np.asarray(adj.sum(axis=0)).ravel()` (`dangling_mask = out_deg == 0.0`) and **in-degrees** from row sums `in_deg = np.asarray(adj.sum(axis=1)).ravel()`, scaling `adj[dst, src]` by `in_deg[row_indices] ** (-beta) * out_deg[adj.indices] ** (-(1.0 - beta))`.

### 4. H1 Baseline Audit Query Checklist Against Current `yaw.db` Schema
Before mutating extraction or storage schemas in **H2–H4**, execute and record these four read-only baseline diagnostics against `yaw.db` using the exact table/column names in [`yaw/daemon/graph/storage/db.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py):
1. **Value-Like Entity Prevalence & Hub Degree:**
   - Fraction of rows in `entities` (`name`) and `entity_labels` (`WHERE label_kind = 'skos:prefLabel'`) whose surface form is purely numeric, date, version, currency, port, or boolean; their row count in `entity_vec`; and their max/p95 degree in `edges WHERE kind IN ('passage', 'fact')`.
2. **Parallel-Predicate Collapse Rate:**
   - Count of `(subj_id, obj_id)` entity pairs in `facts` with $>1$ distinct `pred` currently collapsed into a single `(src_id, dst_id, 'fact')` row in `edges`:
     ```sql
     SELECT subj_id, obj_id, COUNT(DISTINCT pred) AS pred_cnt
     FROM facts
     GROUP BY subj_id, obj_id
     HAVING pred_cnt > 1;
     ```
3. **Stage 3 Candidate Pair Breakdown:**
   - On a fixed sample of candidate pairs from [`find_entity_synonym_candidates`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L117-L171) and [`find_entity_disambiguation_candidates`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L174-L278), record the proportion of true aliases (`skos:altLabel` / `skos:hiddenLabel`), homographs (`owl:differentFrom`), taxonomic pairs (`skos:broader` / `skos:narrower`), and associative pairs (`skos:related`).
4. **Directed Operator Zero-Diff Regression Invariant:**
   - Verify on a real `CSRGraph` that [`build_transition_matrix`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L73-L185) with the directed `[dst, src]` operator and `in_deg` / `out_deg` normalization reproduces the exact bit-for-bit `CSRGraph.transition_t` and [`run_ppr`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L187-L250) scores when fed symmetric forward/reverse weights and `"max"` aggregation across `beta in (0.0, 0.5, 1.0)`.

## Restart checklist and limitations

- [ ] H1 baseline and contracts
- [ ] H2 literal representation, migration, cache identity and retrieval
- [ ] H3 hierarchy/provenance and validation
- [ ] H4 relation policy, directed operator and aggregation ablations
- [ ] H5 paired results and Tier 1/Tier 2 gates

At each packet, record the commit, completed exit checks, schema/cache/policy
versions, evaluation artifact locations and next action. Keep original databases
and source files recoverable. Refuse incompatible resume state with an actionable
message. Preserve already compatible completed stages across interruption.
Existing processes retain their original contracts; no silent live migration.

These changes do not guarantee fewer tokens, improved recall or lower latency.
More relation choices can make classification harder; literal removal can lose
useful joins; hierarchy can spread mass into generic hubs. Bound recursive work,
diagnostics and caches for unattended runs, and measure the actual tradeoffs.

## References and planning verification

- [RDF triples and literals](https://www.w3.org/TR/rdf11-concepts/)
- [SKOS semantic relations, S27 and cycles](https://www.w3.org/TR/skos-reference/)
- [OWL 2 RL property rules](https://www.w3.org/TR/owl2-profiles/)
- [SHACL constraints](https://www.w3.org/TR/shacl/)

Planning audit: checked the designated checkout and Git remote; inspected core
triple types, SQLite schema/validation, engine edge weights and CSR construction
and application. No model run or runtime test was performed for this outline.
