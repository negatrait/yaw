# YAW Architectural Audit Report & Living Ledger

> [!NOTE]
> **HISTORICAL ARCHIVE:** This report represents an earlier sprint audit (prior to the 5-Epic architectural roadmap reconciliation). Active engineering tasks, architectural technical debt, and quality gates are tracked live in [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md).

## Repository Overview
- **Path**: `/home/harri/yaw`
- **Total Source Files**: 46
- **Test Files**: 5
- **Sentrux Quality Signal**: 5542
- **Architecture Status**: Clean layering, 0 cycles, downward acyclic flow (DSM level breaks: 3, above_diagonal: 0).

---

## Active Findings
*(Zero active findings. Codebase is fully optimal and idiomatic.)*

---

## Closed Issues

### [CLOSED] ARCH-SIMPLIFY-002: Surgical Tightening & State Simplification in Extractor
- **Target File**: [`src/yaw/nlp/extractor.py`](file:///home/harri/yaw/src/yaw/nlp/extractor.py)
- **Resolution**:
  1. Replaced manual `_cached_pipeline` module global state with thread-safe `@functools.cache` decorator on `get_leaf_pipeline()`.
  2. Precomputed normalized relation strings outside of predicate matching loop in `normalize_predicate()`.
  3. Replaced class `__init__` boilerplate with class attribute in `_OllamaModel`.
  4. Streamlined fallback expressions in `map_namespace()`.
- **Physical Verification**: 21/21 tests passing across test suite (`/home/harri/.local/bin/uv run pytest`).

### [CLOSED] ARCH-WIRING-001: Stage 1 Leaf-Taxonomy Extraction Pipeline Breakdown & Shadowed Classification
- **Target File**: [`src/yaw/nlp/extractor.py`](file:///home/harri/yaw/src/yaw/nlp/extractor.py#L58-L340)
- **Resolution**:
  1. Fixed pipeline ordering in `build_leaf_pipeline()`: `nlp.add_pipe("fallback_llm_ner", after="llm_ner", ...)` is called before `nlp.disable_pipe("llm_ner")`, preventing spaCy `ValueError: [E002]`.
  2. Fixed string target extraction in `extract_leaf_taxonomy()` to evaluate `return doc._.leaf_taxonomy or doc_to_taxonomy(doc, nlp=nlp)`.
  3. Reordered `map_namespace()` logic to prioritize `doc_cat` (from `doc.cats`) over generic `"entities"` fallback.
  4. Streamlined namespace resolution eliminating unreachable branches.
- **Physical Verification**: 21/21 tests passing across test suite (`tests/test_yaw.py`).
