# Test Suite Readiness: Epic 7 (Sprint 7A / Milestone M0)
**Pluggable RDF/OWL Ontology Substrates & Configurable Open-IE Extraction**
**Target Test File:** `tests/test_pluggable_ontology.py`  
**Governing Canon:** `docs/epic7-pluggable-ontology-integration-spec.md §5`, `TEST_INFRA.md`, `FOUNDATIONAL.md §5.4`, `docs/adr/ADR-008.md`

---

## 1. Executive Summary & Verification Matrix

The Milestone M0 unmocked TDD test suite for Epic 7 is authored, validated, and ready in `tests/test_pluggable_ontology.py`.
The suite operates against real SQLite databases on disk with WAL pragmas (via pytest's `tmp_path`), performs real `rdflib.Graph` parsing for sample ontologies, evaluates pySHACL validation over 1,000+ relational triples, and enforces all foundational invariants.

| # | Test Case | Target Subsystem / Quadrant | Governing Invariant | M0 Red Phase Status |
|---|-----------|-----------------------------|---------------------|---------------------|
| 1 | `test_quadrant_1_open_world_baseline` | Q1: `internal`, `open_ie=true` | Invariant 2 (Lossless Extraction) | **PASSED** (Baseline leaf relation projection works) |
| 2 | `test_quadrant_2_closed_predicates` | Q2: `internal`, `open_ie=false` | Invariant 2 (Non-Destructive View Scoping) | **FAILED (TDD RED)**: View filter not yet active (<5ms toggle) |
| 3 | `test_quadrant_3_open_world_external` | Q3: `external`, `open_ie=true` | Invariant 1 (Pluggable Ingestion Substrate) | **FAILED (TDD RED)**: `ingest_ontology` awaiting M1 |
| 4 | `test_quadrant_4_closed_domain` | Q4: `external`, `open_ie=false` | Invariant 2 (Closed Domain Scoping) | **FAILED (TDD RED)**: `ingest_ontology` & view filter awaiting M1 |
| 5 | `test_antonym_polarity_exclusion` | Predicate Normalizer | Invariant 3 (Polarity Exclusion Mask) | **FAILED (TDD RED)**: `normalize_predicate_antonym_guarded` awaiting M3 |
| 6 | `test_lean_pyshacl_latency` | pySHACL Engine | Invariant 4 (Relational Pre-Closure <200ms) | **FAILED (TDD RED)**: `validate_wiki_graph_fast` awaiting M4 |
| 7 | `test_active_subgraph_projection` | Retriever GraphRAG | Invariant 5 (Ghost Class Pruning) | **FAILED (TDD RED)**: Ingestion & retriever conductance awaiting M1/M4 |
| 8 | `test_invariant_7_non_punning_rejection` | Ingestion Metamodeling | Invariant 7 (Dual-Epistemic Separation) | **FAILED (TDD RED)**: Punning rejection validator awaiting M1 |
| 9 | `test_zero_cache_wipes` | Relational Storage | Invariant 2 (Zero Embedding Wipes) | **PASSED** (Mode toggling preserves embeddings & raw triples) |

---

## 2. Quality Gate Verification

- **Linter Check:**
  ```bash
  uv run ruff check tests/test_pluggable_ontology.py
  # Result: All checks passed! (0 errors, 0 warnings)
  ```
- **Type Check:**
  ```bash
  uv run ty check
  # Result: All checks passed! (0 diagnostics)
  ```
- **Test Collection:**
  ```bash
  uv run pytest tests/test_pluggable_ontology.py --collect-only --no-testmon
  # Result: 9 tests collected in 0.70s
  ```
- **Test Execution (TDD Red Phase):**
  ```bash
  uv run pytest tests/test_pluggable_ontology.py -v --no-testmon
  # Result: 2 passed, 7 failed (expected red phase awaiting Milestone M1-M4 implementations)
  ```

---

## 3. Grounded Invariant Mapping

1. **Invariant 1 (Pluggable Ingestion Substrate):**
   `test_quadrant_3_open_world_external` verifies end-to-end TTL parsing via `rdflib.Graph` and insertion into `dynamic_ontologies`, `dynamic_classes`, and `dynamic_properties`.
2. **Invariant 2 (Non-Destructive Scoping & Lossless Extraction):**
   `test_quadrant_1_open_world_baseline`, `test_quadrant_2_closed_predicates`, `test_quadrant_4_closed_domain`, and `test_zero_cache_wipes` verify that toggling between open-IE and closed-world filtering is instantaneous (<5ms), never alters `leaf_relations`, and preserves 100% of chunk embeddings.
3. **Invariant 3 (Polarity Exclusion Mask):**
   `test_antonym_polarity_exclusion` verifies that causal/dialectical antonym pairs (`prevents`/`causes`, `refutes`/`supports`, `contradicts`/`supports`) are blocked from semantic cosine vector collapse.
4. **Invariant 4 (Relational Pre-Closure & pySHACL Speed):**
   `test_lean_pyshacl_latency` populates 1,000+ relational triples on disk and verifies that relational recursive CTE closure + `inference=None` validation executes within the strict < 200 ms latency budget.
5. **Invariant 5 (Active Subgraph Projection):**
   `test_active_subgraph_projection` verifies that uninstantiated classes (`dico:GhostClass`) are strictly pruned from `GraphRAGRetriever._node_to_idx`, while active entities and their taxonomic ancestors are preserved.
6. **Invariant 7 (Dual-Epistemic Separation & Non-Punning Invariant):**
   `test_invariant_7_non_punning_rejection` asserts that an entity declaring both `owl:Class` and `skos:Concept` triggers an immediate `ValueError` and prevents database pollution.
