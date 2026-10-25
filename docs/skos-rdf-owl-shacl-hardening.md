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

### 1. The Core W3C Partition: Concepts (`skos:Concept` / `owl:Thing`) vs. Data Values (`rdfs:Literal` / `rdfs:Datatype`) Under `rdfs:Resource`

In W3C RDFS, OWL 2 DL, and SKOS:
- **Both `skos:Concept` (`owl:NamedIndividual`) and `rdfs:Literal` are subclasses of `rdfs:Resource`** (`rdfs:Literal rdfs:subClassOf rdfs:Resource`). Because W3C SKOS Reference (Section 7) defines documentation properties (`skos:definition`) with an unrestricted domain (`rdfs:Resource`), **even `rdfs:Literal` values may carry a contextual `skos:definition`** explaining their role in the passage.
- Within `rdfs:Resource`, however, **`skos:Concept` (`owl:Thing`) and `rdfs:Literal` are disjoint**, and the predicate (`pred`) linking a subject to an object is strictly partitioned into two disjoint OWL 2 property classes:

1. **`owl:ObjectProperty` $\implies$ Concept / Individual Target (`skos:Concept`, `owl:NamedIndividual`, `owl:Class`):**
   - Domain (`sub`): `owl:Class | owl:NamedIndividual` (`skos:Concept`).
   - Range (`obj`): `owl:Class | owl:NamedIndividual` (`skos:Concept`).
   - Example in `ExtractTriples`:
     ```json
     {
       "sub": ["The Program", "definition of the program"],
       "pred": ["programmed by", "owl:ObjectProperty"],
       "obj": ["Bob", "who is Bob the owl:NamedIndividual"],
       "def": "the reification definition of this fact triplet"
     }
     ```
   - Participates in `skos:broader` / `skos:narrower` hierarchies, associative `skos:related` networks, and Stage 3 `JudgeSynonyms` disambiguation, with full PPR traversal weight (`1.0`).
2. **`owl:DatatypeProperty` $\implies$ Data Value Target (`rdfs:Literal`):**
   - Domain (`sub`): `owl:Class | owl:NamedIndividual` (`skos:Concept`). A literal can **never** be the subject of an RDF statement.
   - Range (`obj`): `rdfs:Literal` (in RDF 1.1, untyped literals default automatically to `xsd:string`, so `"rdfs:Literal"` serves as the universal MVP datatype without requiring brittle per-XSD regex guessing).
   - Example in `ExtractTriples`:
     ```json
     {
       "sub": ["The Program", "definition of the program"],
       "pred": ["was released", "owl:DatatypeProperty"],
       "obj": ["19.5.1995", "definition is allowed since rdfs:Literal is in the scope of rdfs:Resource"],
       "def": "some definition"
     }
     ```
   - **Graph Traversal & Disambiguation Contract for `rdfs:Literal`:**
     - **Low-Priority PPR Edges (`datatype_property = 0.5`):** Rather than severing literals from the graph (which breaks multi-hop recall and prevents PPR from walking through a specific date, port, or version to its source chunks), `owl:DatatypeProperty` edges remain in the graph with a **moderated traversal weight (`0.5` default, since YAW's `beta=0.5` degree-normalized CSR operator and `1/df` seed damping already penalize high-degree hubs)**, analogous to how `owl:differentFrom` uses weight `0.0`.
     - **Excluded from `skos:Concept` Taxonomy:** Because `rdfs:Literal` is disjoint from `skos:Concept`, pure `rdfs:Literal` targets (nodes only reached via `owl:DatatypeProperty`) are excluded from Stage 3 `JudgeSynonyms` disambiguation and forbidden from participating in `skos:broader` or `skos:related` edges.

#### Concrete Modeling Classes in YAW

- **Dates and Timestamps (`"2026-10-25"`, `"19.5.1995"`, `"2024"`, `"2026-10-25T05:53:00Z"`):**
  - **Class/Datatype:** `rdfs:Literal` (with optional `xsd:date`, `xsd:dateTime`, `xsd:gYear` refinement).
  - **Predicate:** `["was released", "owl:DatatypeProperty"]`.
  - *Exception:* A named historical era, sprint, or freeze window (e.g., `:Q4_Freeze_Window`) linked by an `owl:ObjectProperty` is a `skos:Concept`; its calendar start/end dates are `rdfs:Literal` values attached via `owl:DatatypeProperty`.
- **Port Numbers, Version Numbers, and Codes (`"8080"`, `"443"`, `"v2.1"`):**
  - *Pattern A (Configuration Attribute on a Service):* `owl:DatatypeProperty` pointing to an `rdfs:Literal` (`"8080"`).
  - *Pattern B (Identifier Attached to a Concept, W3C SKOS Rule S15):* For standard protocols, error codes, or classifications (`HTTPS` = `443`, `Not Found` = `404`), W3C SKOS defines **`skos:notation`** (`skos:notation a owl:DatatypeProperty`).
- **Light-Verbs & Light-Verb Idioms (`"made a request to"`, `"took a dependency on"`, `"had a ball"`):**
  - A light-verb phrase is **never** an `rdfs:Literal`, because it expresses a relation or an event/state concept rather than a scalar measurement.
  - *Case A (Transitive Light-Verb Phrase):* Relational predicate. If target is an entity (`Service B`), `"pred"` is `["took a dependency on", "owl:ObjectProperty"]`. If target is a scalar (`"30s"`), `"pred"` is `["has a timeout of", "owl:DatatypeProperty"]`.
  - *Case B (Intransitive Light-Verb Idiom — e.g. "Cinderella had a ball", "Node 3 went into split-brain"):*
    - The state/eventuality itself is modeled as a **`skos:Concept`** (`owl:NamedIndividual`), possessing `skos:definition` and participating in `skos:broader` taxonomies (e.g. `having a ball` $\to$ `enjoyment`).
    - The subject links to this state concept via `["experienced", "owl:ObjectProperty"]`.

#### Lexical Normalization (`text_processing`) vs. Surface Label Preservation (`skos:prefLabel` / `raw_label`)

- **Token-Internal Structural Punctuation Preserved in `text_processing` ([`yaw/core/types.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/core/types.py)):**
  - `text_processing` normalizes strings for `label_norm` / hash key matching by casefolding (`casefold()`), collapsing whitespace, and stripping outer/prose punctuation (quotes, parentheses, brackets, commas, semicolons, exclamation/question marks, trailing periods), **while preserving token-internal structural punctuation** (`.`, `-`, `+`, `#`, `/`, `%`, `:`) attached to alphanumeric characters:
    - Technical symbols and suffixes after alphanumerics (`"C++"` $\to$ `"c++"`, `"C#"` $\to$ `"c#"`, `"99.9%"` $\to$ `"99.9%"`) are preserved so `"C++"`, `"C#"`, and `"C"` never collide into `"c"`.
    - Internal dots, slashes, colons, and numeric/identifier hyphens (`"Python 3.10"` $\to$ `"python 3.10"`, `"19.5.1995"` $\to$ `"19.5.1995"`, `"2026-10-25"` $\to$ `"2026-10-25"`, `"8080/tcp"` $\to$ `"8080/tcp"`, `"UTF-8"` $\to$ `"utf-8"`, `"Node.js"` $\to$ `"node.js"`) remain intact as single tokens.
- **Strict Separation Between `skos:prefLabel` Surface Form and `label_norm` Key Matching:**
  - `text_processing` is used **strictly** for `entity_labels.label_norm`, deterministic MD5 hash IDs (`compute_mdhash_id(label_norm, prefix="entity-")`), and normalized lookup keys.
  - The canonical surface form (`skos:prefLabel`, e.g., `"Python 3.10"`, `"C++"`, `"Alan Turing"`) is preserved in `entity_labels.raw_label` (and exported by `serialize_graph_turtle`), rather than overwriting surface labels with stripped `label_norm` strings.

#### Summary Mapping Table for YAW

| Extracted Item | Domain in W3C | Idiomatic W3C Class | Predicate Tuple (`pred`) | Graph & PPR Behavior in YAW (`yaw.db`) |
| :--- | :--- | :--- | :--- | :--- |
| **Concept / Named Entity** (`"The Program"`, `"Bob"`) | `rdfs:Resource` $\supset$ `owl:Thing` | `skos:Concept`, `owl:NamedIndividual` | `["...", "owl:ObjectProperty"]` | Full PPR weight (`1.0`); included in `JudgeSynonyms`, `skos:broader`, `skos:related` |
| **Date / Port / Metric / Version** (`"19.5.1995"`, `"8080"`, `"99.9%"`) | `rdfs:Resource` $\supset$ `rdfs:Literal` | `rdfs:Literal` (defaults to `xsd:string` in RDF 1.1) | `["...", "owl:DatatypeProperty"]` | Damped PPR weight (`0.5`) + `1/df` to reach chunks; excluded from `JudgeSynonyms` & `skos:broader` |
| **Transitive Light-Verb** (`"made a call to"`, `"has timeout"`) | `rdf:Property` | `owl:ObjectProperty` or `owl:DatatypeProperty` | *Is* the typed `pred` in `facts` | Determines whether `obj` is `skos:Concept` (`1.0`) or `rdfs:Literal` (`0.5`) |
| **Intransitive Light-Verb Idiom** (`"had a ball"`, `"took a dive"`) | `rdfs:Resource` $\supset$ `owl:Thing` | `skos:Concept` + `owl:NamedIndividual` | `["experienced", "owl:ObjectProperty"]` | `entities` (with `skos:definition` & `skos:broader`) + full `owl:ObjectProperty` weight |

---

### 2. Traversal Policy Contract

RDF predicates express relations; their meaning does not prescribe numerical ranking weights. YAW must define and evaluate that retrieval policy explicitly. Keep semantic assertions, structural incidence edges and the derived retrieval operator distinct, with a documented mapping between them.

`rdf:predicate` is a statement-reification property identifying a statement's
predicate. It is not the generic name of every subject-to-object relation.
Extracted business predicates remain in facts; the internal `fact` edge kind
must not silently stand in for all their semantics.

| Relation / structure | Planned traversal policy |
| --- | --- |
| Passage/entity incidence (`passage`) | Explicit structural policy connecting chunks and `rdfs:Resource` nodes (`skos:Concept` and `rdfs:Literal`) |
| `owl:ObjectProperty` fact (`obj_kind = 'entity'`) | Directed forward assertion (`1.0` default); configurable inverse traversal strength |
| `owl:DatatypeProperty` fact (`obj_kind = 'literal'`) | Moderated traversal weight (`0.5` default) + `1/df` damping + `beta=0.5` degree normalization so literals aid chunk recall without becoming super-hubs; excluded from `skos:broader` / `skos:related` |
| `skos:related` | Symmetric associative relation between `skos:Concept` nodes; configurable retrieval strength |
| `skos:broader` | Directed child-to-parent assertion between `skos:Concept` nodes; configure upward and downward traversal strengths separately |
| `skos:narrower` | Normalize to the inverse broader assertion; do not double-count |
| `owl:differentFrom` | Repulsive identity constraint (`0.0` traversal weight), excluded from positive PPR walks |
| Labels, definitions and reification metadata | Available on all `rdfs:Resource` nodes (`skos:Concept` and `rdfs:Literal`) for embedding/FTS matching and audit |

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

### H2 — Typed predicate disambiguation (`owl:ObjectProperty` vs. `owl:DatatypeProperty`) and low-priority literal traversal

- Disambiguate `skos:Concept` (`owl:NamedIndividual`) vs. `rdfs:Literal` objects at the extraction boundary by typing the predicate `"pred"` in `ExtractTriples` as `["<predicate phrase>", "owl:ObjectProperty" | "owl:DatatypeProperty"]` in core types, DSPy signatures, GBNF, parsers and stage caches.
- Because `rdfs:Literal rdfs:subClassOf rdfs:Resource` and the domain of `skos:definition` in W3C SKOS is `rdfs:Resource`, preserve contextual definitions (`obj_def`) on both `owl:ObjectProperty` and `owl:DatatypeProperty` objects. Do not classify solely by brittle numeric regexes.
- Keep `rdfs:Literal` nodes connected in the graph to their source chunks (`passage` edges) and subjects with a **moderated traversal weight (`datatype_property = 0.5` default)** + `1/df` damping so specific dates, versions, and port numbers contribute to PPR chunk recall without acting as un-damped super-hubs.
- Exclude pure `rdfs:Literal` targets from Stage 3 `JudgeSynonyms` disambiguation so they never consume LLM disambiguation calls or enter `skos:broader` / `skos:related` taxonomies.
- Version extraction/cache identities (`openie_fingerprint`, `extraction_fingerprint`, `shacl_fingerprint`, `candidate_fingerprint`).

Exit: real SQLite round trips and retrieval work for both `owl:ObjectProperty` and `owl:DatatypeProperty` triples; restart reuse is verified; paired results are recorded.

### H3 — Directed hierarchy and deterministic validation

- Extend the existing pair judgment with explicit child and parent identities,
  keeping identity decisions separate. Retain existing candidate generation
  initially and measure its coverage and hierarchy classification precision.
- **Disambiguation Prompt & Direction Hardening (`JudgeSynonyms`):**
  Clarify that for `skos:broader`, `canonical` strictly designates the `parent_broader_concept`
  (root concept), not the surviving preferred label or the more specific instance.
  Supply contrastive few-shot examples separating true taxonomic subsumption
  (`cryptanalysis (narrower) -> mathematics (broader)`) from coordinate siblings
  (`http 2 <-> http 3 (skos:related)`) and acronyms (`slos -> service level objectives (skos:altLabel)`).
- **Deterministic Lexical Subsumption Guard & Auto-Healing (`ConceptShape/lexicalSubsumption`):**
  Enforce that if concept $A$'s lexical tokens are a strict subset of concept $B$'s tokens
  ($A \subsetneq B$, e.g. `"inference"` vs `"redwood hosted inference"`), $B$ (the specialized
  compound) must be narrower than $A$ (the root). Any asserted edge $A \to B$ is a structural
  inversion violation; SHACL validation must flag it, and auto-healing must invert the edge to $B \to A$.
- **Coordinate Sibling & Ordinal Guard (`ConceptShape/siblingNotBroader`):**
  Concepts differing solely by trailing version digits or ordinal tiers (`http 2` vs `http 3`,
  `sev 1` vs `sev 2` vs `sev 3`) are coordinate peers/siblings. They belong in `skos:related`,
  and are forbidden from forming `skos:broader` edges.
- **Acronym & Coreferent Disjointness Guard:**
  Acronyms, initialisms, and lexical normalization variants (`slos` vs `service level objectives`,
  `sev1` vs `sev 1`) must merge as `skos:altLabel` on a single concept, physically prohibited
  from existing as separate entities linked by `skos:broader`.
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
S27 conflicts, lexical subsumption inversions, sibling misclassifications, and source
deletion have deterministic tests. Classification and storage can be evaluated with
hierarchy traversal disabled.

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

### 1. H2 Watchout: Typed Predicate Disambiguation (`owl:ObjectProperty` vs. `owl:DatatypeProperty`), Low-Weight Literal Edges & `fact_sources` Provenance
- **Typed Predicate Extraction Contract (`ExtractTriples`):**
  - Instead of relying on brittle numeric regexes or wiping `obj_def`, `ExtractTriples` emits `"pred"` as `["<predicate phrase>", "owl:ObjectProperty" | "owl:DatatypeProperty"]`.
  - When `"pred"` carries `"owl:ObjectProperty"`, the object is a `skos:Concept` (`obj_kind = 'entity'`) and emits a full-weight (`1.0`) directed forward fact edge.
  - When `"pred"` carries `"owl:DatatypeProperty"`, the object is an `rdfs:Literal` (`obj_kind = 'literal'`, `obj_datatype = 'rdfs:Literal'`). Because `rdfs:Literal rdfs:subClassOf rdfs:Resource` and the domain of `skos:definition` is `rdfs:Resource`, its contextual `obj_def` is preserved.
- **Low-Priority Literal Traversal & Taxonomy Exclusion:**
  - `rdfs:Literal` targets stay connected to their source chunks (`passage` edges) and subjects with a **moderated traversal weight (`datatype_property = 0.5` default, since `beta=0.5` degree normalization and `1/df` damping already penalize high-degree hubs)** so PPR can traverse specific dates, versions, and ports to reach source chunks without common literals acting as un-damped super-hubs.
  - Pure `rdfs:Literal` targets (nodes only reached via `owl:DatatypeProperty`) are excluded from Stage 3 `JudgeSynonyms` disambiguation and forbidden from participating in `skos:broader` or `skos:related` edges.
- **Explicit Fact Provenance (`fact_sources`) & SHACL Gating (`shacl.py`):**
  - `fact_sources (fact_id TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (fact_id, chunk_id))` tracks explicit chunk provenance for every fact so [`delete_chunks`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1323-L1339) prunes any `fact_id` with zero remaining rows in `fact_sources`, and `ProvenanceShape/factSources` verifies that every fact references at least one valid chunk in `chunks` (never inserting fake `'__healed__'` chunk IDs).

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

### 3. H3 Watchout: Semantic Inversion, Coordinate Sibling Leaks & SHACL Lexical Subsumption Validation
- **Empirical Live Audit Findings (2026-10-25):**
  - An inspection of 66 active `skos:broader` edges in `yaw_optimize_*.db` revealed that while graph-theoretic cycle prevention (`broaderCycle = 0`) and SKOS S27 disjointness (`S27-disjointBroaderRelated = 0`) passed with zero violations, the semantic taxonomy exhibited three severe linguistic breakdown modes:
    1. **Direct Inversions (24% of All Edges):** In 16 of 66 cases, the root concept was asserted to point to the specialized compound concept as its "broader parent" (e.g. `engineering -> security engineering`, `inference -> redwood hosted inference`, `rollout -> full rollout`, `priority -> priority p0`).
    2. **Coordinate Sibling Confusion:** Version identifiers and ordinal levels (`http 2 -> http 3`, `sev 3 -> sev 2 -> sev 1`) were misclassified as `skos:broader` rather than symmetric `skos:related`.
    3. **Acronym / Synonym Leaks:** Lexical aliases (`slos -> service level objectives`, `sev1 -> sev 1`) were misclassified as `skos:broader` instead of being merged via `skos:altLabel`.
- **Root Cause in `JudgeSynonyms` & Parsing:**
  - In `JudgeSynonyms`, the LLM prompt used `canonical` as an overloaded key: for `skos:altLabel`, it meant the surviving prefLabel; for `skos:broader`, the model often picked the "more concrete / specialized instance", causing `engine.py:2118-2120` to assign `child_id = tgt, parent_id = src` backwards.
- **Architectural & SHACL Mitigation:**
  1. **Deterministic Lexical Subsumption Invariant (`ConceptShape/lexicalSubsumption`):**
     - For any candidate pair $(A, B)$ where token set $T(A) \subsetneq T(B)$ (e.g. $A = \text{"inference"}$, $B = \text{"redwood hosted inference"}$), $B$ is linguistically a specialized sub-concept of root $A$.
     - At ingestion time in `engine.py`, enforce that if `skos:broader` is asserted, $B$ must be the narrower concept pointing to $A$ as broader ($B \to A$). If the LLM asserted $A \to B$, automatically invert the edge before write.
     - Add `ConceptShape/lexicalSubsumption` to `_scan_shacl_violations` and auto-heal in `lint_graph_shacl` by reversing any inverted edges.
  2. **Ordinal & Version Sibling Guard (`ConceptShape/siblingNotBroader`):**
     - Entities differing solely by trailing version tokens (`http 2` vs `http 3`, `v1` vs `v2`, `p0` vs `p1`) must be classified as `skos:related`. If asserted as `skos:broader`, SHACL flags the violation and demotes the edge to `skos:related`.
  3. **Disjointness with Lexical Aliases:**
     - Acronyms (e.g. all-caps letter sequences matching the initials of the compound target) must be routed to `skos:altLabel` entity merging, forbidden from forming `skos:broader` edges.
  4. **Disambiguation Prompt Hardening (`JudgeSynonyms`):**
     - Make `canonical` explicitly mean `parent_broader_concept` in the prompt signature, providing contrastive examples that explicitly demonstrate taxonomy direction (`cryptanalysis (narrower) -> mathematics (broader)`).

### 4. H4 Watchout: Four-Layer Parallel-Fact & Direction Collapse Upstream of `ppr.py`
- **All Four Collapse Points Must Be Reconciled Together:**
  1. **Write-Time Symmetrization in [`GraphEngine.index_chunks` (`engine.py:1745-1746`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L1745-L1746):** Stop writing synthetic reverse `(o_id, s_id, "fact", 1.0, cid)` rows as if they were asserted forward facts; preserve asserted `subj_id -> obj_id` direction in storage and let the versioned traversal policy emit forward (`fact:fwd`) and inverse (`fact:rev`) contributions at operator construction time.
  2. **Same-Chunk Predicate Collision in `edges` / `edge_sources` ([`db.py:99-117`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/db.py#L99-L117), [`graph_ops.py:920-955`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L920-L955)):** Because `edge_sources` keys on `(src_id, dst_id, kind, chunk_id)` with `kind = 'fact'`, two distinct predicates `(u, p1, v)` and `(u, p2, v)` in the same `chunk_id` currently collapse into one row. Derive fact-level relation contributions from `facts` + `fact_sources` (where `obj_id IS NOT NULL`) so distinct predicates `(u, p1, v)` and `(u, p2, v)` survive into [`load_graph_topology`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1105-L1125).
  3. **SQL `GROUP BY src_id, dst_id` Stripping in [`load_graph_topology` (`graph_ops.py:1115-1123`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L1115-L1123):** Replace the relation-blind `SELECT src_id, dst_id, MAX(weight) FROM edges ... GROUP BY src_id, dst_id` query with relation-tagged directed edges `(src_id, dst_id, relation_kind, evidence_count_or_weight)` covering `passage`, `skos:related`, `skos:broader`, and distinct `(subj_id, obj_id, pred)` entity-object facts.
  4. **Undirected Symmetrization & Row-Sum Degree Normalization in [`build_transition_matrix` (`ppr.py:73-185`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L73-L185):**
     - Map each asserted relation `(u, v, kind, weight)` through the versioned `RelationTraversalPolicy` into directed `(src, dst, effective_weight)` transitions placed at CSR entry `adj[dst, src]` (`row = dst`, `col = src`) to match `graph.transition_t.dot(p)` in [`run_ppr` (`ppr.py:228`)](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L228).
     - Support configurable parallel-edge aggregation (`"max"`, `"sum"`, `"capped_sum"`) per ordered `(dst, src)` pair after deduplicating identical assertions.
     - Compute directed **out-degrees** from column sums `out_deg = np.asarray(adj.sum(axis=0)).ravel()` (`dangling_mask = out_deg == 0.0`) and **in-degrees** from row sums `in_deg = np.asarray(adj.sum(axis=1)).ravel()`, scaling `adj[dst, src]` by `in_deg[row_indices] ** (-beta) * out_deg[adj.indices] ** (-(1.0 - beta))`.

### 5. H1 Baseline Audit Query Checklist Against Current `yaw.db` Schema
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
