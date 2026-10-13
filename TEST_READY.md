# Test readiness and verification evidence

**Updated:** 2026-10-13\
**Active scope:** [Epic 7.8](docs/epic7.8-open-ie-ontology-alignment.md), E78-01–20 and the [all-gates boundary before 8.x](docs/epics-alignment-roadmap.md#execution-order-and-hard-entry-gate).\
**Evidence categories:** [TEST_INFRA](TEST_INFRA.md).

## Current status

| Gate | Status | Evidence / remaining work |
| --- | --- | --- |
| Epic 7.8 declaration | Recorded | Persisted work packages and restart checklist |
| Documentation reconciliation | Passed for this revision | Selected-path/authority review, agent execution map, current-repository link/anchor audit, 20 scenario IDs and table checks; details below |
| E78-01–10 semantics, grounding, cache and migration | Not run for the new contract | Implementation/test reconciliation required |
| E78-11 retrieval | Not run for the new contract | Real matrices and configured scorer/channel checks required |
| E78-12 FastMCP/wiki integration | Not run in this documentation task | Real client/lifecycle checks required |
| E78-13 Windows/Unix runtime | Not verified | This checkout's locked environment is not runtime-tested in this revision; the reviewed process-liveness path remains a required check |
| E78-14 actual-model pilot | Not run | Independent extraction/comparison/synthesis labels and declared sentinel outcomes required before closing 7.8 |
| E78-15–20 complete compiler, conflict/revision, zero fallbacks, rules, languages and synthesis | Not run | Shared application, failure injection, retraction and consumer agreement remain implementation gates |
| Retained 7.7 representation work | Not newly verified | Must pass current 7.8 contracts; no separate bypass |
| Epics 8.1 / 8.2 / 8.3 | Blocked by runtime gates | No 8.x implementation starts before every 7.8 gate passes; full held-out comparisons belong to 8.2 |

Do not convert the historical results below into current pass claims. For each future run record commit plus working-tree identity, environment, exact command, input manifest, result, skips and measured workload. A documentation-only check is not application runtime validation.

## Current checkout and resume point

The correct repository is `C:\Users\Harri.Ahokas\Documents\yaw`, remote `https://github.com/negatrait/yaw.git`, branch `dev`, starting HEAD `a4142060c4e3f77f51b6a6528a84307b21ae604c`. Before this correction its only working-tree change was the staged [alignment audit](docs/alignment-audit-12-10-2026.md). That file and its staged version are preserved byte-for-byte. Project metadata, `uv.lock` and source entry points are present.

The earlier documentation pass used another local copy by mistake. Its missing project files, missing Git object and spaCy availability observations do not describe this checkout. The [older review](docs/reviews/open-ie-ontology-review-2026-10-12.md) retains its own pinned historical baseline; the alignment audit matches this checkout's starting HEAD.

Read-only source inspection on 2026-10-13 reconfirmed the three-node claim model (`nlp/models.py`), first-candidate identity selection (`indexer/canonicalize.py`), Raw/REL coexistence (`nlp/extractor.py`), source-text response parsing (`indexer/pipeline.py`) and validation-focused compilation plus stub ingestion (`server/daemon.py`). The daemon also retains `os.kill(pid, 0)` in its liveness path; that probe was not executed. No prose edit closes these implementation gates.

**Agent resume:** Documentation gates 7.8.1–7.8.2 are complete. Begin **7.8.3 step 1** using [TEST_INFRA entry commands](TEST_INFRA.md#entry-commands) and the [delivery packet](docs/epic7.8-open-ie-ontology-alignment.md#delivery-packets). First prove the locked runtime and shared-state/publication contract; the complete semantic slice belongs to 7.8.5. No runtime packet has been completed by this documentation review.

## Actionability review — 2026-10-13

| Gap found | Correction and owner |
| --- | --- |
| First package required the complete pipeline before later packages supplied its dependencies | Ordered entry/work/exit packets in Epic 7.8; complete slice in 7.8.5 |
| Parser ownership and partial-envelope admission were underspecified | `nlp/parsing.py` owns one parser/grounding sequence; models stay in `nlp/models.py` |
| `force`, superseded publication and requested-scope status needed implementer choices | Compiler guide fixes non-destructive refresh, desired-generation fencing and scope/status rules |
| Verification commands could select only changed tests or exercise synthetic model replies | TEST_INFRA maps all E78 cases to test owners, disables testmon selection for gates and isolates the actual-model runner |
| Small organization inputs and lexical seed formatting had undefined cases | Hierarchy guide specifies small-input behavior; PPR guide specifies the single FTS rank formatter |
| Link verification depended on an untracked temporary helper | `tools/check_docs.py` provides a repeatable standard-library command in this repository |
| 8.x deliverables lacked concrete entry points/artifacts | Roadmap specifies workload/benchmark manifests, runner command contracts and installed-wheel verification |

**Verification:** `python tools/check_docs.py --json` passed: **45 documents, 557 local links/anchors, 20/20 mapped E78 cases and all eight epic packages**. The audit is included for link checking and remains unchanged. Eight isolated negative controls detected missing/nonportable links, invalid anchors/tables, missing/duplicate scenarios, a missing test mapping and a duplicate package; a fenced example did not cause a false failure. The control copy was restored and passed again.

The checker was executed with bundled Python 3.12.14 using only the standard library. Ruff 0.16.10 check/format checks on `tools/check_docs.py` and `git diff --check` passed. These are documentation/tool checks, not a Python 3.14 application result. File hashes and Git index entries verify that this review preserved runtime files, the staged audit and concurrent repository housekeeping edits. This round changes 13 Markdown files and adds the documentation checker; no application/runtime or actual-model gate was run. Future implementation records use the [run-record fields](TEST_INFRA.md#run-record).

## Previous documentation transfer verification

On 2026-10-13, documentation checks passed against the corrected repository: **44 reconciled documents, 432 repository-relative links/anchors and all 20 E78 scenario IDs**. No broken targets, invalid anchors, active table errors, flagged retired-path contradictions or obsolete machine-specific hyperlinks were found. The owner's staged audit is preserved as evidence and is excluded from rewriting and link normalization.

Commands, run from the repository above: bundled Python 3.12.14 executing `C:\Users\Harri.Ahokas\AppData\Local\Temp\yaw_check_correct_repo_docs_20261013.py --actual` (exit 0); `git -c safe.directory=C:/Users/Harri.Ahokas/Documents/yaw diff --check` (exit 0). The Python helper checks documentation only; its interpreter version is not a YAW compatibility result. File hashes and Git index entries were compared before and after application to confirm that runtime/source/configuration files and the staged audit were unchanged. The correction changes only Markdown documentation and agent guidance.

No YAW application test, actual-model evaluation or benchmark ran in this documentation correction. All corresponding gates remain open. The checked work packages are declaration and documentation only, 7.8.0–7.8.2.

## Historical Epic 7 readiness report

This is the original milestone report, preserved as evidence of what was claimed then. Its predicate-mask tests, timings and implementation status are not the Epic 7.8 acceptance contract.

<details>
<summary>Historical Epic 7 milestone report</summary>

# Test Suite Readiness: Epic 7 (Sprint 7A / Milestone M0)
**Pluggable RDF/OWL Ontology Substrates & Configurable Open-IE Extraction**
**Target Test File:** `tests/test_pluggable_ontology.py`\
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

</details>
