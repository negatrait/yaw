# E2E Test Infra: Pluggable RDF/OWL Ontology Substrates (Epic 7)

## Test Philosophy
- Opaque-box and disk-grounded unmocked integration tests.
- Real SQLite databases with WAL pragmas on temporary disk directories (`tmp_path`).
- Tests cover the 2x2 matrix (Quadrants 1–4) and all 7 foundational invariants ([`FOUNDATIONAL.md §2`](FOUNDATIONAL.md)).
- Zero mock cushions: tests validate real RDF parsing, real relational SQLite pre-closure, real pySHACL execution, real spaCy PhraseMatcher tokens, and real PageRank conductance.

## Feature Inventory & Test Mapping
| # | Test Case | Target Subsystem / Quadrant | Key Invariant / Assertion |
|---|-----------|-----------------------------|---------------------------|
| 1 | `test_quadrant_1_open_world_baseline` | Q1: internal, open_ie=true | Invariant 2: arbitrary predicates persist in `leaf_relations` and project to `unified_edges`. |
| 2 | `test_quadrant_2_closed_predicates` | Q2: internal, open_ie=false | Invariant 2: `unified_edges` filters open-IE predicates; `leaf_relations` retains all rows unmodified; toggle in < 5 ms. |
| 3 | `test_quadrant_3_open_world_external` | Q3: external, open_ie=true | Invariant 1: Ingest DiCo/BFO ontology; projects external dynamic taxonomy + open relations. |
| 4 | `test_quadrant_4_closed_domain` | Q4: external, open_ie=false | Invariant 2: Projects only active ontology properties and canonical predicates; strictly filters out-of-domain relations. |
| 5 | `test_antonym_polarity_exclusion` | Predicate Normalizer | Invariant 3: `POLARITY_EXCLUSION_MASK` prevents `prevents` -> `causes` and `refutes` -> `supports` vector collapse. |
| 6 | `test_lean_pyshacl_latency` | pySHACL Engine | Invariant 4: `validate_wiki_graph_fast` evaluates on 1,000+ triples with recursive CTE pre-closure in < 200 ms. |
| 7 | `test_active_subgraph_projection` | Retriever GraphRAG | Invariant 5: `_node_to_idx` includes instantiated classes and ancestors, strictly prunes uninstantiated `dico:GhostClass`. |
| 8 | `test_invariant_7_non_punning_rejection` | Ingestion Pipeline | Invariant 7: Ingesting ontology declaring a URI as both `owl:Class` and `skos:Concept` raises `ValueError`. |

## Test Architecture
- Test suite path: `tests/test_pluggable_ontology.py`
- Test runner invocation: `uv run pytest tests/test_pluggable_ontology.py -v`
- Pass/Fail semantics: All 8 tests must exit with code 0 without skipping or mock patching.
- Quality gates: `uv run ruff check .` (0 errors), `uv run ty check` (0 diagnostics).
