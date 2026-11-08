# Technical Specification: W3C Semantic Cascade & Active EDC Canonicalization (`docs/w3c-improvements.md`)

Current scope, 2026-11-08: sections 12 and 13 govern strict `JudgeLabels` identity
equivalence (`W1`), the Canonical Gemma 4 3-stage cascade without Python token
overrides or cross-label auto-merge gates (`W2`), universal statement irreflexivity
across entity and literal objects (`W3`), Tier 3 deterministic statement definition
synthesis (`W4`), single-surface `skos:prefLabel` (`clean_surface_label`),
non-transitive `1`-hop `skos:broader` S27 disjointness (`W5`), and scheme-free
batch-independent identity/relations. The graph plan owns the paired `64+64`/`128`
time and graph gates. Sections 1–11 retain earlier slice history; graph-wide staging,
multi-hop `broader_tc` closure, or Python token-override instructions there are
superseded by sections 12–13 and the native graph contract. Pinned code: `c98aebe`.

- Controls: [types.py](../yaw/core/types.py), [__init__.py](../yaw/core/__init__.py), [config.py](../yaw/core/config.py), [planner.py](../yaw/daemon/graph/planner.py), [normalization.py](../yaw/daemon/graph/normalization.py), [engine.py](../yaw/daemon/graph/engine.py), [prompts.py](../yaw/daemon/graph/inference/prompts.py), [grammars.py](../yaw/daemon/graph/inference/grammars.py), [adapter.py](../yaw/daemon/graph/inference/adapter.py), [parsers.py](../yaw/daemon/graph/inference/parsers.py), [program.py](../yaw/daemon/graph/inference/program.py), [invariants.py](../yaw/daemon/graph/inference/invariants.py), [components.py](../yaw/daemon/graph/inference/components.py), [evidence.py](../yaw/daemon/graph/storage/evidence.py), [graph_ops.py](../yaw/daemon/graph/storage/graph_ops.py), [shacl.py](../yaw/daemon/graph/storage/shacl.py), [rows.py](../yaw/daemon/graph/storage/rows.py), [synonyms.py](../yaw/daemon/graph/storage/synonyms.py), [__init__.py](../yaw/daemon/graph/storage/__init__.py), [indexing.py](../yaw/daemon/indexing.py), [runtime.py](../yaw/daemon/runtime.py) ([test_architecture.py](../tests/core/test_architecture.py))
- Depends on: [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [graph-storage.md](../.agents/rules/graph-storage.md), [inference.md](../.agents/rules/inference.md), [native-cocoindex-graph.md](native-cocoindex-graph.md), [epistemic-confidence-spec.md](epistemic-confidence-spec.md)
- Updated with: [w3c-improvements.md](w3c-improvements.md), [native-cocoindex-graph.md](native-cocoindex-graph.md), [epistemic-confidence-spec.md](epistemic-confidence-spec.md), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [incremental-publication-work.md](reviews/incremental-publication-work.md)

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [graph-storage.md](../.agents/rules/graph-storage.md): `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py`; [inference.md](../.agents/rules/inference.md): `yaw/daemon/graph/inference/*.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/daemon/graph/storage/synonyms.py`, `yaw/daemon/graph/storage/graph_ops.py:canonicalize_entity_labels_edc`, `yaw/daemon/graph/inference/prompts.py:JudgeLabels`, `yaw/daemon/graph/inference/prompts.py:JudgeHierarchy`, `yaw/daemon/graph/inference/prompts.py:JudgeRelated` |
| Peers | [native-cocoindex-graph.md](native-cocoindex-graph.md): owns native CocoIndex graph publication, closed-world resource/statement boundary (`§4`), and the paired `64 + 64` vs. `128` acceptance gate; [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `64`-chunk micro-batch source scheduling and runtime lifecycle; [epistemic-confidence-spec.md](epistemic-confidence-spec.md): owns `pool_confidences`, `bayesian_noisy_or`, and `1`-hop S27 confidence resolution; [dspy-improvements.md](dspy-improvements.md): constrains GBNF adapter and RLM healer architecture; [Epic 3](epic-3-performance.md): owns post-Epic 2 `owl:FunctionalProperty` / `owl:InverseFunctionalProperty` candidate mining, non-transitive `skos:related` walk barriers, and `prov:wasDerivedFrom` co-provenance reinforcement |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md): anchors HippoRAG 2 retrieval parity; [model-evaluation.md](model-evaluation.md): anchors local `llama.cpp` serving constraints (`parallel = 1` `[target]`) |
| Supersedes / superseded by | Supersedes `synonym-evidence-v1`, single-shot `JudgeSynonyms`, `skos:ConceptScheme` membership, `equivalent: false, alt_label/hidden_label: true` merge branches, `planner.py` Python token overrides (`is_modifier_head_specialization`, `_has_disjoint_non_acronym_tokens`, `_is_acronym`) and cross-label auto-merge gates (`auto_merge_cosine_gate`, `auto_merge_difflib_gate`), and multi-hop `broader_tc` / `query_broader_tc` / `ConceptShape/broaderTransitiveCapExceeded`; graph-wide staging instructions in §§1–11 are superseded by [native-cocoindex-graph.md](native-cocoindex-graph.md) and §§12–13 |
| Records | [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [incremental-publication-work.md](reviews/incremental-publication-work.md), [resource-vocabulary-boundary.md](reviews/resource-vocabulary-boundary.md), [concept-scheme-removal.md](reviews/concept-scheme-removal.md), [evaluation-graph-storage.md](evaluation-graph-storage.md), [evaluation-inference.md](evaluation-inference.md) |

## 1. Context & Root-Cause Problem Statement

In [`benchmark_mobile_report.md`](file:///C:/Users/Harri.Ahokas/.gemini/antigravity/brain/705b1f4b-1f82-432f-af64-b480ab176de5/benchmark_mobile_report.md), the 4B model (`gemma-4-E4B-it-qat-q4_0`) systematically mixes semantic relations during Stage 3 synonym/ontology adjudication—classifying taxonomic subtypes as `skos:related` (e.g., `pilot tenants` $\leftrightarrow$ `tenants`, `telemetry points` $\leftrightarrow$ `telemetry`) and classifying both subtypes and domain-coupled concepts as `owl:differentFrom` (e.g., `seat` $\neq$ `consumption`, `Consumption price` $\neq$ `consumption`, `Finance` $\neq$ `Finance-Legal`). Because [`build_transition_matrix`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/ppr.py#L168-L183) treats `owl:differentFrom` as an active negative inhibitor ($\alpha_{\text{diff}} = 0.5$) that subtracts Personalized PageRank mass, false `owl:differentFrom` edges actively suppress relevant neighbors during multi-hop retrieval.

Four concrete architectural defects cause this failure mode:
1. **Pointer-Table Cognitive Overload (`synonym-evidence-v1`)**: [`_Packet`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/synonyms.py#L19-L197) splits pair context across four separate lookup tables (`pairs`, `entities`, `facts`, `sources`) connected by synthetic pointer aliases (`e0`, `f0`, `s0`), forcing the 4B model to dereference multi-hop JSON pointers across batches of up to 64 pairs ([`planner.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/planner.py#L1086-L1094)).
2. **Zero-Scratchpad 5-Way Single-Shot Ask**: [`JudgeSynonyms`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/prompts.py#L266-L313) and [`build_dspy_synonym_judge_grammar`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/grammars.py#L195-L237) force token `#1` of each decision to select one of five relations (`skos:altLabel`, `skos:hiddenLabel`, `skos:broader`, `skos:related`, `owl:differentFrom`) with zero thinking/reasoning tokens and an overloaded `"canonical"` string slot.
3. **Ontological Conflation of `owl:differentFrom` and `skos:semanticRelation`**: OWL does not infer `owl:differentFrom` merely from different IRIs. Entity distinctness and a supported SKOS relationship are separate assertions; a pair can be distinct and still related. Presenting `owl:differentFrom` alongside `skos:broader` and `skos:related` in a single-shot prompt causes the model to emit `owl:differentFrom` whenever $A \neq B$.
4. **Missing End-of-Round Active EDC Canonicalization**: [`prepare_synonym_candidates`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/planner.py#L618-L626) and [`apply_synonym_adjudication`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/planner.py#L740-L755) select the surviving `skos:prefLabel` during pairwise merges using either LLM-emitted `"canonical"` strings or naive string-length tuples `(len(def_a), -len(name_a), ea)`, allowing 1-off extraction typos to usurp `skos:prefLabel`.

---

## 2. Architecture, Dataflow & Domain Containment

```mermaid
flowchart TD
    P["Candidate Pair (A, B)\nFlat Denormalized Evidence (v2)"] --> W0{"W3C Pre-Gates (SQLite/Python)\n• Same-norm sense consolidation [W2]?\n• Direct Fact (A --P--> B) [Rule W1]?"}
    W0 -- "Same-norm d_sim >= threshold" --> MERGE["Union Concept Cluster (A, B)"]
    W0 -- "Direct Fact [W1: Irreflexive]" --> S2
    W0 -- "Unresolved" --> S1["Stage 1: label_judge (JudgeLabels + GBNF Reasoning, max_reasoning_chars=360)\n{pair_id, reasoning, equivalent: true, definition, hidden_label: bool} | {pair_id, reasoning, equivalent: false}"]
    S1 -- "equivalent=true, hidden_label=false [W1/S13]" --> MERGE
    S1 -- "equivalent=true, hidden_label=true [W1]" --> MERGE_H["Union Cluster + Flag hiddenLabel Variant"]
    S1 -- "equivalent=false [Distinct Concepts]" --> S2["Stage 2: hierarchy_judge (JudgeHierarchy + GBNF Reasoning)\n{pair_id, reasoning, broader: bool, narrower: bool}"]
    S2 -- "(true, false)" --> E_AB["Emit A skos:broader B (1-hop non-transitive)"]
    S2 -- "(false, true) [SKOS S25 Inverse]" --> E_BA["Emit B skos:broader A (1-hop non-transitive)"]
    S2 -- "(true, true) [Asymmetry -> S27]" --> E_REL["Emit skos:related (A <-> B)"]
    S2 -- "(false, false)" --> W2{"W3C Statement Role Gate [W1/W2]\nDirect Fact (A --P--> B) OR\nShared Role (A --P--> X <--P-- B)?"}
    W2 -- "Yes (0 Stage-3 LLM calls)" --> E_REL
    W2 -- "No" --> S3["Stage 3: related_judge (JudgeRelated + GBNF Reasoning)\n{pair_id, reasoning, related: bool}"]
    S3 -- "true" --> E_REL
    S3 -- "false" --> E_DIFF["Emit owl:differentFrom (Homograph Gate only)"]
    MERGE --> EDC["End-of-Round Active EDC Pass (graph_ops.py)\nclean_surface_label + Score labels by (verbatim_in_chunk, in_ast_title, chunk_count)\nEnforce 1 raw_label skos:prefLabel + skos:altLabel / skos:hiddenLabel [S13/S14]"]
    MERGE_H --> EDC
```

### Domain Containment Check
All modifications strictly obey the domain containment tree enforced by [`tests/core/test_architecture.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/core/test_architecture.py):
- `yaw.core.*` ([`yaw/core/types.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/core/types.py)) imports only Python standard library modules.
- `yaw.daemon.graph.storage.*` ([`synonyms.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/synonyms.py), [`graph_ops.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py)) imports only `yaw.daemon.graph.storage.*` and `yaw.core.*`.
- `yaw.daemon.graph.inference.*` ([`prompts.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/prompts.py), [`grammars.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/grammars.py), [`parsers.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/parsers.py), [`adapter.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/adapter.py), [`invariants.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/invariants.py), [`diagnostics.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/diagnostics.py), [`program.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py), [`optimizer.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py)) imports only `yaw.daemon.graph.inference.*` and `yaw.core.*`.
- `yaw.daemon.graph.*` ([`planner.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/planner.py), [`engine.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py)) orchestrates `storage` and `inference` within `yaw.daemon.graph`.

---

## 3. Formal Contracts & Typed Interfaces

### 3.1 Denormalized Evidence Packet (`yaw/daemon/graph/storage/synonyms.py` & `yaw/core/types.py`)

```python
# JSON schema emitted by iter_synonym_evidence ("format": "synonym-evidence-v2"):
# {
#   "format": "synonym-evidence-v2",
#   "pairs": [
#     {
#       "pair_id": "e0|e1",
#       "dense_sim": 0.88,
#       "lexical_sim": 0.72,
#       "shared_documents": ["docs/pricing.md"],
#       "shared_islands": [["docs/pricing.md", 10, 45]],
#       "direct_facts": [["pilot tenants", "participate in", "initial testing phase"]],
#       "shared_role_facts": [["pilot tenants", "billed via", "invoices"], ["tenants", "billed via", "invoices"]],
#       "entity_a": {
#         "name": "tenants",
#         "definition": "Customer organizations provisioned on the platform.",
#         "type": "skos:Concept",
#         "labels": {"skos:prefLabel": ["tenants"], "skos:altLabel": []},
#         "facts": [["tenants", "own", "subscriptions"]],
#         "sources": [{"path": "docs/pricing.md", "title": "Tenants", "text": "..."}]
#       },
#       "entity_b": {
#         "name": "pilot tenants",
#         "definition": "A group of low-volume tenants used for the initial testing phase.",
#         "type": "skos:Concept",
#         "labels": {"skos:prefLabel": ["pilot tenants"], "skos:altLabel": []},
#         "facts": [["pilot tenants", "participate in", "initial testing phase"]],
#         "sources": [{"path": "docs/pricing.md", "title": "Pilot Phase", "text": "..."}]
#       }
#     }
#   ]
# }
```

### 3.2 End-of-Round Active EDC Canonicalizer (`yaw/daemon/graph/storage/graph_ops.py`)

```python
def canonicalize_entity_labels_edc(
    conn: sqlite3.Connection | YawConnection,
    corpus_root: Path | str | None = None,
    *,
    entity_ids: Sequence[str] | None = None,
    hidden_label_norms_by_entity: Mapping[str, set[str]] | None = None,
) -> int:
    """Run deterministic end-of-round Active EDC label canonicalization over entity_labels and entities.

    For each candidate entity:
    1. Cleans surface forms via `clean_surface_label(raw)` (stripping leading/trailing `:;,`). Demotes any
       label norm in `hidden_label_norms_by_entity`, any label norm whose `best_raw` fails
       `is_resource_label(best_raw)` or matches `is_scalar_label(best_raw)`, or any label norm absent
       verbatim from its hydrated source chunk on disk (when `corpus_root` is available and chunk is
       non-stale) to `skos:hiddenLabel` (`is_hidden = True`, `verbatim_in_source = 0`). SHACL-healed
       labels (`set_pref_label`) are stored with `chunk_id = "__healed__"` (`cid.startswith("__")` ->
       `verbatim_in_source = 1`, `is_hidden = False`).
    2. Scores all eligible (`skos:prefLabel` / `skos:altLabel`) label norms by provenance tuple:
       `(verbatim_in_source, in_ast_title, chunk_count, is_cased_or_multiword, -len(raw_label), label_norm)`.
    3. Promotes the single winning norm to `skos:prefLabel`, aligns `raw_label = winning_raw` across all
       `skos:prefLabel` rows of the entity (and `best_raw_by_norm[norm]` for `skos:altLabel` / `skos:hiddenLabel`)
       so every entity has 1 distinct `skos:prefLabel` surface string, updates `entities.name` to `winning_raw`,
       and demotes all other non-hidden norms to `skos:altLabel` (enforcing SKOS S13 & S14).
    Returns the number of entities whose canonical `skos:prefLabel` or `entities.name` was updated.
    """
```

In `apply_synonym_adjudication`, `canonicalize_entity_labels_edc` is scoped to `entity_ids=sorted(touched_entity_ids)` (`entity_ids=()` is an immediate no-op touching `0` rows). When `apply_shacl_repairs` executes `set_pref_label`, it validates `is_resource_label(val) and not is_scalar_label(val)`, demotes existing scalar/non-resource `entity_labels` rows on the focus entity to `skos:hiddenLabel`, and upserts the repaired label with `chunk_id = "__healed__"` unconditionally ([native-cocoindex-graph.md §4](native-cocoindex-graph.md#4-semantics-and-invariants)), preserving the 6-tuple `norm_scores: dict[str, tuple[int, int, int, int, int, str]]` across subsequent batches.

### 3.3 3-Stage Signatures & Grammars (`yaw/daemon/graph/inference/prompts.py` & `grammars.py`)

```python
SYNONYM_JUDGE_STAGE_NAMES: tuple[str, str, str] = (
    "label_judge",
    "hierarchy_judge",
    "related_judge",
)


class JudgeLabels(dspy.Signature):
    """Evaluate candidate entity pairs in `candidate_pairs` for strict W3C SKOS identity equivalence (`skos:altLabel` / `skos:hiddenLabel`).

    CORE PRINCIPLE:
    - Merging two nodes (`equivalent: true`) asserts they are the EXACT SAME underlying `skos:Concept` or `owl:NamedIndividual` in this domain.
    - If two nodes are distinct concepts or distinct individuals (`equivalent: false`), emit `"equivalent": false` so the pair advances to Stage 2 (`JudgeHierarchy`).

    MANDATORY FALSIFICATION RULES (state in 1 concise sentence in `reasoning` before `equivalent`):
    1. Named Individuals & Identifiers: NEVER merge (`equivalent: false`) two people, teams, regions, tiers, stages, versions, or percentiles that have different given names, surnames, numbers, or codes (e.g., 'Noah Patel' vs 'Ravi Patel', 'Priya Rao' vs 'Priya Shah', 'Tier 1' vs 'Tier 4', 'Fast tier' vs 'Standard tier', 'p95' vs 'p99').
    2. Modifier-Head & Category Distinctions: NEVER merge (`equivalent: false`) a general system/tier/resource with a specific component, metric, regression, or policy that modifies it (e.g., 'Fast tier' vs 'fast tier p99 regression', 'Tenant' vs 'Tenant-id', 'Latency' vs 'p99 TTFT regression', 'Redwood' vs 'Redwood Router').
    3. True Equivalence (`equivalent: true`): Emit `equivalent: true` ONLY when both `a` and `b` denote the exact same concept or individual, provide a unified `"definition"`, and set `"hidden_label": false` for a valid synonym/acronym (`skos:altLabel`) or `"hidden_label": true` for a misspelling/typo (`skos:hiddenLabel`)."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON object containing `pairs`: list of candidate entity pairs with definitions, direct_facts, shared_role_facts, and chunk context"
    )
    decisions: list[dict[str, Any]] = dspy.OutputField(
        desc='List of decision objects: `{"pair_id", "reasoning", "equivalent": true, "definition", "hidden_label": bool}` when equivalent, or `{"pair_id", "reasoning", "equivalent": false}` when distinct'
    )


class JudgeHierarchy(dspy.Signature):
    """Evaluate Stage 2 W3C SKOS taxonomic/partitive hierarchy (`skos:broader` and `skos:narrower`) for distinct concept pairs in `candidate_pairs`."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON object (`synonym-evidence-v2`) containing distinct candidate entity pairs with inline definitions, facts, and source excerpts."
    )
    decisions: list[dict[str, Any]] = dspy.OutputField(
        desc=(
            'List of {"pair_id": "e0|e1", "reasoning": "...", "broader": bool, "narrower": bool} objects '
            "where broader=True iff entity_a is a narrower subtype/part/instance of broader concept entity_b (`A skos:broader B`), "
            "and narrower=True iff entity_b is a narrower subtype/part/instance of broader concept entity_a (`A skos:narrower B`)."
        )
    )


class JudgeRelated(dspy.Signature):
    """Evaluate Stage 3 W3C SKOS associative relation (`skos:related` vs `owl:differentFrom`) for distinct, non-hierarchical pairs in `candidate_pairs`."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON object (`synonym-evidence-v2`) containing distinct, non-hierarchical entity pairs with inline definitions, facts, and source excerpts."
    )
    decisions: list[dict[str, Any]] = dspy.OutputField(
        desc='List of {"pair_id": "e0|e1", "reasoning": "...", "related": bool} objects.'
    )


def build_dspy_label_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = 360,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeLabels (`equiv-branch`: `"equivalent": true, "definition": ..., "hidden_label": bool` | `"equivalent": false`)."""


def build_dspy_hierarchy_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = 360,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeHierarchy (`{"decisions": [{"pair_id": ..., "reasoning": ..., "broader": bool, "narrower": bool}]}`)."""


def build_dspy_related_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = 360,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeRelated (`{"decisions": [{"pair_id": ..., "reasoning": ..., "related": bool}]}`)."""
```

### 3.4 Complete Replacement of Legacy `synonym_*` Predictors, Fingerprints, Caches, Audit Trails & Token Counters
1. **First-Class Predictors on `YawProgram` (`yaw/daemon/graph/inference/program.py`)**:
   - **Removes** `self.synonym_judge = dspy.Predict(JudgeSynonyms)`.
   - **Replaces** it with three first-class predictors on `YawProgram`:
     - `self.label_judge = dspy.Predict(JudgeLabels)`
     - `self.hierarchy_judge = dspy.Predict(JudgeHierarchy)`
     - `self.related_judge = dspy.Predict(JudgeRelated)`
   - Binds all three to `worker` LM in `_bind_predictor_lms`. Because they are `dspy.Predict` attributes on `YawProgram`, `program.named_predictors()` automatically exposes `"label_judge"`, `"hierarchy_judge"`, and `"related_judge"` to `dspy.GEPA`, `dspy.MIPROv2`, `ActiveTraceComponentSelector`, and `RLMInstructionProposer`.
2. **Replacement of `synonym_audit_trail` with Per-Stage Audit Trails (`program.py` & `diagnostics.py`)**:
   - **Removes** `self.synonym_audit_trail`.
   - **Replaces** it with three stage-specific `DiagnosticHistory` trails on `YawProgram`:
     - `self.label_judge_audit_trail = DiagnosticHistory()`
     - `self.hierarchy_judge_audit_trail = DiagnosticHistory()`
     - `self.related_judge_audit_trail = DiagnosticHistory()`
   - Updates `dump_audit_trails` ([`program.py:260-280`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L260-L280)) and `trim_diagnostics` ([`diagnostics.py:21-34`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/diagnostics.py#L21-L34)) to serialize and bound `label_judge_audit_trail`, `hierarchy_judge_audit_trail`, and `related_judge_audit_trail`.
3. **Replacement of `synonym_fingerprint` & `_synonym_fp_cache` with Per-Stage Fingerprints & Caches (`optimizer.py` & `engine.py`)**:
   - **Why Replace Instead of Update**: During GEPA optimization, [`ActiveTraceComponentSelector`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L762-L813) mutates **one** predictor per iteration (`label_judge`, OR `hierarchy_judge`, OR `related_judge`). If all three judges shared a single `synonym_fingerprint`, mutating `related_judge` (Stage 3) would invalidate cached decisions and traces for Stage 1 (`label_judge`) and Stage 2 (`hierarchy_judge`).
   - **Removes** `synonym_fingerprint(program)` and `self._synonym_fp_cache`.
   - **Replaces** them with:
     - `judge_stage_fingerprint(program: YawProgram, stage_name: Literal["label_judge", "hierarchy_judge", "related_judge"]) -> str` (plus convenience functions `label_judge_fingerprint`, `hierarchy_judge_fingerprint`, `related_judge_fingerprint`) in [`optimizer.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L300-L330).
     - `@coco.fn(memo=True, version=2)` on `judge_synonyms` in `<state_dir>/coco_lmdb` (`CandidateStageCache` retired under [native-cocoindex-graph.md §7](native-cocoindex-graph.md#7-removal-inventory)).
     - `extraction_fingerprint(program)` in [`optimizer.py:333-368`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L333-L368) replaces `"synonym_judge"` in `graph_stage_keys` with `"label_judge"`, `"hierarchy_judge"`, and `"related_judge"`.
4. **Replacement of `synonym_judge_tokens` with Per-Stage Token Counters (`engine.py`, `optimizer.py`, `parsers.py`)**:
   - **Removes** `synonym_judge_tokens` from [`HippoRAGEngine._empty_graph_rlm_stats`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L510), [`_hydrate_chunk_traces_and_stats_from_db`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L486), [`optimizer.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L1338), and [`compute_task_quality_score`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/parsers.py#L1058).
   - **Replaces** it with `label_judge_tokens`, `hierarchy_judge_tokens`, and `related_judge_tokens`.
5. **`STAGE_SCHEMA_CONTRACTS`, `validate_stage_invariants` & `YawGBNFAdapter` (`invariants.py`, `adapter.py`)**:
   - Removes `"synonym_judge"` from `STAGE_SCHEMA_CONTRACTS` and replaces it with contracts for `"label_judge"`, `"hierarchy_judge"`, and `"related_judge"` so `InstructionSynthesizerWorker` (`proposer.py:98`) injects the exact schema contract for each judge during GEPA/SIMBA instruction evolution.
   - Updates `validate_stage_invariants` to validate `"label_judge"` (`equivalent: true` requires non-empty `definition` and boolean `hidden_label`; `equivalent: false` rejects `alt_label=True` or `hidden_label=True`), `"hierarchy_judge"`, and `"related_judge"` predictions.
   - Updates `YawGBNFAdapter` to remove `JudgeSynonyms` and wire `JudgeLabels` $\to$ `build_dspy_label_judge_grammar`, `JudgeHierarchy` $\to$ `build_dspy_hierarchy_judge_grammar`, and `JudgeRelated` $\to$ `build_dspy_related_judge_grammar` with `max_reasoning_chars=self.max_reasoning_chars` (`360` `[target]`).
6. **Trace Capture for GEPA (`yaw/daemon/graph/engine.py`)**:
   - Updates `extract_named_trace_entries` at [`engine.py:1594-1598`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L1594-L1598) to capture per-stage traces for `{"label_judge"}`, `{"hierarchy_judge"}`, and `{"related_judge"}` so GEPA's `ActiveTraceComponentSelector` receives live execution trajectories for each individual judge stage.

---

## 4. Invariants & Explicit Non-Goals

### Non-Negotiable Invariants
1. **Short-Circuit Funnel Invariant**: Every pair resolved by a deterministic W3C pre-gate or an earlier LLM stage is immediately filtered out and **never** sent to subsequent LLM stages.
2. **Stage Decision Determinism (`W1`)**:
   - Stage 1 (`JudgeLabels` `equiv-branch` — Invariant `W1`, $\text{merge}(A, B) \iff \text{equivalent}(A, B) = \text{true}$):
     - `equivalent = True, hidden_label = False` (with non-empty unified `definition`) $\to$ `alt_label = True, hidden_label = False, relation = "skos:altLabel"`
     - `equivalent = True, hidden_label = True` (with non-empty unified `definition`) $\to$ `alt_label = False, hidden_label = True, relation = "skos:hiddenLabel"`
     - `equivalent = False` (JSON object closes immediately after `"equivalent": false`) $\to$ `alt_label = False, hidden_label = False, relation = None` (distinct concepts; advance to Stage 2).
   - Stage 2 `(broader, narrower)`:
     - `(True, False)` $\to$ `relation="skos:broader"`, `source_id=entity_a`, `target_id=entity_b` (`A skos:broader B`, `1`-hop non-transitive)
     - `(False, True)` $\to$ by SKOS S25 (`owl:inverseOf`), `relation="skos:broader"`, `source_id=entity_b`, `target_id=entity_a` (`B skos:broader A`, `1`-hop non-transitive)
     - `(True, True)` (SKOS hierarchy asymmetry conflict on distinct concepts) $\to$ by SKOS S27, `relation="skos:related"`
     - `(False, False)` $\to$ non-hierarchical; if `direct_facts` (Rule W1) or `shared_role_facts` (Rule W2) is non-empty, resolve deterministically to `relation="skos:related"`, else advance to Stage 3.
   - Stage 3 `(related)`:
     - `True` $\to$ `relation="skos:related"`
     - `False` $\to$ `relation="owl:differentFrom"` (subject to Guardrail 1 homograph check in `apply_synonym_adjudication`).
3. **Active EDC SKOS S13/S14 & Single-Surface `skos:prefLabel` Invariant**: At the end of every adjudication round, every entity in `stage_conn` has `1` `[target]` `skos:prefLabel` norm and `1` `[target]` distinct `raw_label` string with `label_kind = 'skos:prefLabel'` (cleaned via `clean_surface_label` with zero leading/trailing `:;,`), zero overlap between `skos:prefLabel`, `skos:altLabel`, and `skos:hiddenLabel`, and `entities.name` matches `winning_raw`.
4. **Full GEPA Visibility & Zero Old Judges/Trails**: `self.synonym_judge`, `JudgeSynonyms`, `synonym_fingerprint`, `_synonym_fp_cache`, `synonym_audit_trail`, and `synonym_judge_tokens` are completely removed and replaced with stage-explicit equivalents for `label_judge`, `hierarchy_judge`, and `related_judge`.

### Explicit Non-Goals (Anti-Invention Contract)
- **Zero SQLite Schema Migrations**: Do not add new tables or columns to `yaw.db`.
- **Zero Disjoint-Document `owl:differentFrom` Heuristics**: Never infer `owl:differentFrom` merely because two concepts reside in different documents; cross-document pairs that pass Stage 1 and Stage 2 without shared facts must be adjudicated by Stage 3 (`JudgeRelated`).
- **Zero Legacy Compatibility Shims**: Do not keep `synonym_judge`, `JudgeSynonyms`, `build_dspy_synonym_judge_grammar`, `synonym_fingerprint`, `synonym_audit_trail`, or `synonym_judge_tokens` as fallback aliases.

---

## 5. Superslim Implementation Slices & Oblivion Protocol

Each slice below is scoped to 1–3 production files so a `flash` model can implement it atomically, followed by a mandatory **Critic Round**, **Simplification Round**, and **Deprecated Code Removal Gate**.

---

### Slice 1: Denormalized Pair Evidence (`synonym-evidence-v2`) & W3C Structural Queries {#slice-01}

- **Files**:
  - `[MODIFY]` [`yaw/daemon/graph/storage/synonyms.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/synonyms.py#L19-L254)
  - `[MODIFY]` [`yaw/core/types.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/core/types.py#L180-L245)
  - `[MODIFY]` [`tests/daemon/graph/storage/test_storage.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/storage/test_storage.py)
- **Scope**:
  1. Refactor [`_Packet`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/synonyms.py#L19-L197) in `yaw/daemon/graph/storage/synonyms.py` to emit `"format": "synonym-evidence-v2"` containing only `"pairs": [...]` with inline `entity_a`, `entity_b`, `direct_facts` (Rule W1), and `shared_role_facts` (Rule W2: shared `(pred, obj_id)` or `(subj_id, pred)` in `facts`). *(Note: `broader_tc` / `query_broader_tc` was retired on `2026-11-08` under Invariant `W5` in §13 because W3C SKOS `skos:broader` is strictly `1`-hop non-transitive.)*
  2. Update [`SynonymEvidenceBatch.resolve_decisions`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/core/types.py#L204-L242) to resolve alias IDs (`e0`, `e1`) into real `entity_id` values directly from `source_id`/`target_id` or `pair_id` without requiring a `"canonical"` field, and add `SynonymEvidenceBatch.subset(pair_ids)` so downstream cascade stages receive filtered `SynonymEvidenceBatch` packets containing only unresolved pairs.
  3. Default `iter_synonym_evidence` micro-batch size to `batch_size: int = 6` so each stage prompt stays compact.
- **Verifiable Goal**:
  - `uv run --no-sync pytest tests/daemon/graph/storage/test_storage.py -k "synonym" --no-testmon -q` passes 100%, verifying that `"synonym-evidence-v2"` contains zero top-level `"entities"`, `"facts"`, or `"sources"` pointer tables and populates `direct_facts` and `shared_role_facts` (with `"broader_tc"` absent).
- **Critic Round**:
  - Delegate to `critic`: verify zero N+1 SQLite queries inside `_Packet` when computing `shared_role_facts`, and verify context-budget halving in `iter_synonym_evidence` preserves all pair fields.
- **Simplification Round**:
  - Delegate to `simplifier`: remove `_Packet.fact()`, `_Packet.source()`, `_Packet.facts`, `_Packet.sources`, and `_Packet.spans` indirection methods.
- **Deprecated Code Removal**:
  - `git grep -w "synonym-evidence-v1"` returns `0` matches across `yaw/` and `tests/`.

---

### Slice 2: End-of-Round Active EDC Label Canonicalizer {#slice-02}

- **Files**:
  - `[MODIFY]` [`yaw/daemon/graph/storage/graph_ops.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py#L630-L752)
  - `[MODIFY]` [`yaw/daemon/graph/storage/__init__.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/__init__.py)
  - `[MODIFY]` [`tests/daemon/graph/storage/test_storage.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/storage/test_storage.py)
- **Scope**:
  1. Implement `canonicalize_entity_labels_edc` in [`yaw/daemon/graph/storage/graph_ops.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/graph_ops.py) and export it in [`yaw/daemon/graph/storage/__init__.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/__init__.py).
  2. Compute provenance scores per `(entity_id, label_norm)` using:
      - `verbatim_in_source`: `1` if any `raw_label` for `(entity_id, label_norm)` appears case-insensitively in its hydrated chunk text on disk (or when `corpus_root is None` / chunk is synthetic `__*__`), `0` if the chunk exists on disk and `raw_label` is absent from the source span (demoting such extraction typos to `skos:hiddenLabel`).
      - `in_ast_title`: `1` if `raw_label` appears case-insensitively in `chunks.title` for any provenance chunk.
      - `support_count`: `COUNT(DISTINCT chunk_id)` in `entity_labels`. Canonical fact text is an output of label selection, so matching it would create a circular score that depends on earlier publication boundaries.
  3. Enforce SKOS S13 & S14: any label norm in `hidden_label_norms_by_entity` or failing verbatim grounding is set to `skos:hiddenLabel`; the highest-ranked eligible norm becomes `skos:prefLabel` (aligning `raw_label = winning_raw` and updating `entities.name`), and all remaining eligible norms become `skos:altLabel`.
- **Verifiable Goal**:
  - `uv run --no-sync pytest tests/daemon/graph/storage/test_storage.py -k "edc or merge_entities" --no-testmon -q` passes, proving that a 1-chunk typo (`"Neccessary"`) merged into or onto a multi-chunk/heading-backed entity (`"Necessary"`) is demoted to `skos:hiddenLabel`, `entities.name` becomes `"Necessary"`, and `lint_graph_shacl(conn)` reports `0` label violations.
- **Critic Round**:
  - Delegate to `critic`: verify `canonicalize_entity_labels_edc` runs inside `write_tx(conn)`, handles entities with only 1 label safely (never leaving an entity with 0 `skos:prefLabel` rows), and introduces zero SHACL S13 regressions.
- **Simplification Round**:
  - Delegate to `simplifier`: streamline SQL aggregation of `chunk_count` and `chunks.title` into a single batched query.
- **Deprecated Code Removal**:
  - Remove redundant post-merge label fixup branches superseded by `canonicalize_entity_labels_edc`.

---

### Slice 3: 3-Stage Paired-Boolean Signatures, GBNF Grammars, Adapter & Invariants {#slice-03}

- **Files**:
  - `[MODIFY]` [`yaw/daemon/graph/inference/prompts.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/prompts.py#L266-L313)
  - `[MODIFY]` [`yaw/daemon/graph/inference/grammars.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/grammars.py#L195-L237)
  - `[MODIFY]` [`yaw/daemon/graph/inference/parsers.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/parsers.py#L315-L380)
  - `[MODIFY]` [`yaw/daemon/graph/inference/adapter.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/adapter.py#L48-L100)
  - `[MODIFY]` [`yaw/daemon/graph/inference/invariants.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/invariants.py#L45-L82)
  - `[MODIFY]` [`yaw/daemon/graph/inference/__init__.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/__init__.py)
  - `[MODIFY]` [`tests/daemon/graph/inference/test_grammars.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/inference/test_grammars.py)
  - `[MODIFY]` [`tests/daemon/graph/inference/test_parsers.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/inference/test_parsers.py)
- **Scope**:
  1. Remove `JudgeSynonyms` from [`prompts.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/prompts.py#L266-L313) and replace it with `JudgeLabels`, `JudgeHierarchy`, and `JudgeRelated`.
  2. Remove `DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR` and `build_dspy_synonym_judge_grammar` from [`grammars.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/grammars.py#L195-L237) and replace with `build_dspy_label_judge_grammar`, `build_dspy_hierarchy_judge_grammar`, and `build_dspy_related_judge_grammar` backed by a shared helper `_build_pair_boolean_grammar(..., max_reasoning_chars=360)`.
  3. Replace `parse_synonym_decisions` in [`parsers.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/parsers.py) with `parse_label_decisions`, `parse_hierarchy_decisions`, and `parse_related_decisions` implementing the exact decision rules from Section 4, and replace `synonym_judge_tokens` in [`compute_task_quality_score`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/parsers.py#L1058) with `label_judge_tokens`, `hierarchy_judge_tokens`, and `related_judge_tokens`.
  4. Update [`YawGBNFAdapter.resolve_grammar`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/adapter.py#L48-L100) and [`STAGE_SCHEMA_CONTRACTS` / `validate_stage_invariants`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/invariants.py#L45-L82) to replace `JudgeSynonyms` / `"synonym_judge"` with `JudgeLabels` (`"label_judge"`), `JudgeHierarchy` (`"hierarchy_judge"`), and `JudgeRelated` (`"related_judge"`).
- **Verifiable Goal**:
  - `uv run --no-sync pytest tests/daemon/graph/inference/test_grammars.py tests/daemon/graph/inference/test_parsers.py --no-testmon -q` passes 100%.
- **Critic Round**:
  - Delegate to `critic`: verify that the GBNF grammars strictly constrain `"reasoning"` before the boolean fields, reject empty reasoning strings (`char{1,360}`), and wire all three signatures cleanly in `YawGBNFAdapter` and `invariants.py`.
- **Simplification Round**:
  - Delegate to `simplifier`: ensure all three GBNF functions share `_build_pair_boolean_grammar` (~25 lines total) and all three parser functions share common `pair_id` + `reasoning` validation.
- **Deprecated Code Removal**:
  - Remove `JudgeSynonyms`, `DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR`, `build_dspy_synonym_judge_grammar`, `parse_synonym_decisions`, and `synonym_judge_tokens` from `prompts.py`, `grammars.py`, `parsers.py`, `adapter.py`, `invariants.py`, and `__init__.py`.

---

### Slice 4: Short-Circuiting Cascade, Per-Stage Trails/Caches & GEPA Wiring in `program.py`, `diagnostics.py`, `optimizer.py`, `planner.py` & `engine.py` {#slice-04}

- **Files**:
  - `[MODIFY]` [`yaw/daemon/graph/inference/diagnostics.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/diagnostics.py#L21-L34)
  - `[MODIFY]` [`yaw/daemon/graph/inference/program.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L200-L285)
  - `[MODIFY]` [`yaw/daemon/graph/inference/optimizer.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L300-L368)
  - `[MODIFY]` [`yaw/daemon/graph/planner.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/planner.py#L674-L820)
  - `[MODIFY]` [`yaw/daemon/graph/engine.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L246-L260)
  - `[MODIFY]` [`tests/daemon/graph/inference/test_program.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/inference/test_program.py)
  - `[MODIFY]` [`tests/daemon/graph/test_planner.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/test_planner.py)
  - `[MODIFY]` [`tests/daemon/graph/inference/test_prompts.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/tests/daemon/graph/inference/test_prompts.py)
- **Scope**:
  1. **Replace predictor & audit trail in `program.py` and `diagnostics.py`**:
     - Remove `self.synonym_judge` and `self.synonym_audit_trail` from [`YawProgram.__init__`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L204-L216), [`dump_audit_trails`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L260-L280), [`_bind_predictor_lms`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L282-L300), and [`trim_diagnostics`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/diagnostics.py#L21-L34).
     - Register `self.label_judge = dspy.Predict(JudgeLabels)`, `self.hierarchy_judge = dspy.Predict(JudgeHierarchy)`, `self.related_judge = dspy.Predict(JudgeRelated)` and `self.label_judge_audit_trail`, `self.hierarchy_judge_audit_trail`, `self.related_judge_audit_trail`.
  2. **Replace `synonym_fingerprint` & `_synonym_fp_cache` with Per-Stage GEPA Fingerprints & Token Counters in `optimizer.py` and `engine.py`**:
     - Replace `synonym_fingerprint` in [`optimizer.py:300-330`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L300-L330) with `judge_stage_fingerprint(program, stage_name)` (`label_judge_fingerprint`, `hierarchy_judge_fingerprint`, `related_judge_fingerprint`) and update `extraction_fingerprint` (`graph_stage_keys`) to hash `"label_judge"`, `"hierarchy_judge"`, and `"related_judge"`.
     - Replace `synonym_judge_tokens` with `label_judge_tokens`, `hierarchy_judge_tokens`, and `related_judge_tokens` in `_empty_graph_rlm_stats`, `_hydrate_chunk_traces_and_stats_from_db`, and `optimizer.py`.
  3. **Implement the short-circuiting cascade with trace capture**:
     - Route pairs with non-empty `direct_facts` (Rule W1) directly to Stage 2 (`hierarchy_judge`), skipping Stage 1 (`label_judge`). *(Note: Rule W3 `broader_tc` pre-gate is retired under Invariant `W5` in §13.)*
     - Run Stage 1 (`label_judge`) on remaining pairs; record in `label_judge_audit_trail` and `label_judge_tokens`; short-circuit `equivalent=True` matches (`skos:altLabel`, `skos:hiddenLabel`) and pass `equivalent=False` pairs to Stage 2.
     - Run Stage 2 (`hierarchy_judge`) on unresolved pairs; record in `hierarchy_judge_audit_trail` and `hierarchy_judge_tokens`; short-circuit `skos:broader` (including SKOS S25 inverse when `narrower=True`) and `(True, True)` $\to$ `skos:related`.
     - Route Stage 2 `(False, False)` pairs with non-empty `direct_facts` (Rule W1) or `shared_role_facts` (Rule W2) deterministically to `skos:related` (0 Stage-3 LLM calls).
     - Run Stage 3 (`related_judge`) on the residual unresolved pairs; record in `related_judge_audit_trail` and `related_judge_tokens` to classify `skos:related` vs `owl:differentFrom`.
  4. **Update `apply_synonym_adjudication` in `planner.py`**:
     - Apply directed `skos:broader` (`source_id` $\to$ `target_id`), track `hidden_label` norms on merged entities, and call `canonicalize_entity_labels_edc(stage_conn, corpus_root, entity_ids=sorted(touched_entity_ids))` at the end of the round.
- **Verifiable Goal**:
  - `uv run --no-sync pytest tests/daemon/graph/inference/test_program.py tests/daemon/graph/test_planner.py --no-testmon -q` passes 100%, including asserting that:
    1. `dict(program.named_predictors())` exposes `"label_judge"`, `"hierarchy_judge"`, and `"related_judge"` (and not `"synonym_judge"`).
    2. Mutating only `program.related_judge.signature.instructions` hits 100% cache on `_label_judge_fp_cache` and `_hierarchy_judge_fp_cache` (0 Stage-1 or Stage-2 LLM calls re-executed).
  - Live Gemma-4-E4B verification (`uv run --no-sync pytest tests/daemon/graph/inference/test_prompts.py -m live --no-testmon -q`) passes 100%, verifying `pilot tenants` $\to$ `skos:broader` `tenants`, `seat` $\to$ `skos:broader` `consumption`, and `Consumption price` $\leftrightarrow$ `skos:related` `consumption`.
- **Critic Round**:
  - Delegate to `critic`: audit the end-to-end cascade in `program.py`, `diagnostics.py`, `optimizer.py`, `engine.py`, and `planner.py` for short-circuit completeness, per-stage cache isolation, GEPA predictor registration, per-stage audit trails/tokens, and domain boundary compliance.
- **Simplification Round**:
  - Delegate to `simplifier`: remove all legacy `"canonical"` string matching branches and unify per-stage execution/caching into a single clean helper.
- **Deprecated Code Removal**:
  - Verify `git grep -w` returns `0` matches across the entire repository (`yaw/`, `tests/`, `scripts/`) for all retired symbols in the Oblivion Demolition Table below.

---

### Oblivion Demolition Table

| Retired Symbol / Format | Previous Location | Replacement | Verification Gate |
| :--- | :--- | :--- | :--- |
| `"synonym-evidence-v1"` | [`yaw/daemon/graph/storage/synonyms.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/synonyms.py#L187) | `"synonym-evidence-v2"` (inline denormalized pair evidence) | `git grep -w "synonym-evidence-v1"` $\to$ `0` matches |
| `_Packet.fact` / `_Packet.source` | [`yaw/daemon/graph/storage/synonyms.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/storage/synonyms.py#L47-L97) | Inline pair facts and sources in `_Packet.add` | `git grep -E "def (fact|source)\(" yaw/daemon/graph/storage/synonyms.py` $\to$ `0` matches |
| `JudgeSynonyms` | [`yaw/daemon/graph/inference/prompts.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/prompts.py#L266) | `JudgeLabels`, `JudgeHierarchy`, `JudgeRelated` | `git grep -w "JudgeSynonyms"` $\to$ `0` matches |
| `self.synonym_judge` | [`yaw/daemon/graph/inference/program.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L211) | `self.label_judge`, `self.hierarchy_judge`, `self.related_judge` | `git grep -w "synonym_judge" yaw/` $\to$ `0` matches |
| `synonym_audit_trail` | [`yaw/daemon/graph/inference/program.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/program.py#L204) | `label_judge_audit_trail`, `hierarchy_judge_audit_trail`, `related_judge_audit_trail` | `git grep -w "synonym_audit_trail"` $\to$ `0` matches |
| `synonym_fingerprint` | [`yaw/daemon/graph/inference/optimizer.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/optimizer.py#L300) | `judge_stage_fingerprint`, `label_judge_fingerprint`, `hierarchy_judge_fingerprint`, `related_judge_fingerprint` | `git grep -w "synonym_fingerprint"` $\to$ `0` matches |
| `_synonym_fp_cache` | [`yaw/daemon/graph/engine.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L246) | `_label_judge_fp_cache`, `_hierarchy_judge_fp_cache`, `_related_judge_fp_cache` | `git grep -w "_synonym_fp_cache"` $\to$ `0` matches |
| `synonym_judge_tokens` | [`yaw/daemon/graph/engine.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/engine.py#L510) | `label_judge_tokens`, `hierarchy_judge_tokens`, `related_judge_tokens` | `git grep -w "synonym_judge_tokens"` $\to$ `0` matches |
| `DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR` | [`yaw/daemon/graph/inference/grammars.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/grammars.py#L51) | `DSPY_LABEL_JUDGE_GBNF_GRAMMAR`, `DSPY_HIERARCHY_JUDGE_GBNF_GRAMMAR`, `DSPY_RELATED_JUDGE_GBNF_GRAMMAR` | `git grep -w "DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR"` $\to$ `0` matches |
| `build_dspy_synonym_judge_grammar` | [`yaw/daemon/graph/inference/grammars.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/grammars.py#L195) | `build_dspy_label_judge_grammar`, `build_dspy_hierarchy_judge_grammar`, `build_dspy_related_judge_grammar` | `git grep -w "build_dspy_synonym_judge_grammar"` $\to$ `0` matches |
| `parse_synonym_decisions` | [`yaw/daemon/graph/inference/parsers.py`](file:///c:/Users/Harri.Ahokas/Documents/yaw/yaw/daemon/graph/inference/parsers.py#L319) | `parse_label_decisions`, `parse_hierarchy_decisions`, `parse_related_decisions` | `git grep -w "parse_synonym_decisions"` $\to$ `0` matches |

---

## 6. Real-Substrate Verification Strategy

### Per-Slice TDD & Subagent Workflow
For each slice (`Slice 1` $\to$ `Slice 2` $\to$ `Slice 3` $\to$ `Slice 4`):
1. **Implementer (`flash` subagent)**: Writes real-substrate tests (`tmp_path`, real SQLite `yaw.db`) and minimal implementation for that slice's files.
2. **Critic (`critic` subagent)**: Audits the slice against W3C SKOS/OWL invariants, GEPA predictor registration, per-stage cache/trail isolation, domain boundaries, and zero fallbacks.
3. **Simplifier (`simplifier` subagent)**: Resolves all Critic findings, removes dead code, and verifies the slice's Oblivion Gate (`git grep -w` == 0).

### Mandatory Quality Gates (End of Each Slice & Final Completion)
```powershell
uv run --no-sync ty check
uv run --no-sync ruff check --fix
uv run --no-sync ruff format
uv run --no-sync pytest -m "not live" --no-testmon -q
uv run --no-sync pytest tests/daemon/graph/inference/test_prompts.py -m live --no-testmon -q
git diff --check
```

---

# Epistemic Confidence Calculus & Bayesian Markov Graph Engine (Slice 5)

## 7. Context & Root-Cause Problem Statement

While Slices 1–4 resolve Stage 3 synonym/ontology decision cascades and Active EDC, edge traversal and fact representation across `yaw.db` still suffer from heuristic and fragile weighting:
1. **Unbounded Linear Fact Weighting & Modulo Fragility**:
   - `yaw/daemon/graph/storage/graph_ops.py` assigns hardcoded `1.0` (object property) or `0.5` (datatype property) multiplied linearly by `COUNT(edge_sources)`.
   - `yaw/daemon/graph/ppr.py` guesses datatype properties via `abs((base_w % 1.0) - 0.5) < 1e-6`, silently misclassifying every datatype property mentioned an even number of times ($2 \times 0.5 = 1.0$).
2. **Clamped & Token-Blind Taxonomic Weights**:
   - `yaw/daemon/graph/planner.py` sets `skos:broader` and `skos:related` weights to `max(0.5, sim)`, ignoring inference-time token probability confidences and flattening low-similarity distinctions.
3. **Blind SHACL S27 & Cycle Resolution (`CONSISTENCY_ERROR`)**:
   - Because `skos:broader` and `skos:related` on a pair $(A, B)$ receive the exact same `max(0.5, sim)` weight, SHACL S27 (`ConceptShape/S27-disjointBroaderRelated`) and cycle (`ConceptShape/noBroaderCycles`) healing lacks an empirical tie-breaker.
   - Combined with unexposed `remove_scalar_entity` repairs for `ConceptShape/noScalarLiterals`, `yaw/daemon/graph/engine.py` raises `YawRuntimeError: [CONSISTENCY_ERROR]`, zeroing out affected GEPA evaluations.
4. **Hub Siphoning in PPR**:
   - `ppr.py` normalizes only by raw node degree without destination node epistemic priors.

---

## 8. Unified Probability Mathematics (Strictly Two Canonical Operators)

To make probability calculations **100% scale-invariant in $(0, 1)$**, every operation is governed by **two canonical operators** with **zero magic threshold constants**:

```mermaid
flowchart LR
    subgraph Op1["Operator 1: pool_confidences (Logarithmic Opinion Pool)"]
        P1["Extraction Atomic Confidence\ngeomean(p_llm, s_vec)"]
        P2["Markov Step Transition\npi(k) * geomean(w_edge, p_node)"]
        P3["Query Seed Prior p_0\ngeomean(s_vec(q, x), p_item)"]
        P4["Direct 1-Hop Edge Atomic Confidence\ngeomean(p_llm, s_vec)"]
    end

    subgraph Op2["Operator 2: bayesian_noisy_or (Independent Evidence)"]
        N1["Multi-Chunk Entity/Fact Prior\n1 - prod(1 - c_i)"]
        N2["Multi-Fact / Parallel Edge Pooling\n1 - prod(1 - w_m)"]
    end

    subgraph S27["Empirical Bayes 1-Hop S27 & Cagle Resolution (Zero Magic Numbers)"]
        B1["Compare 1-Hop w_broader vs w_rel\nBayes Factor BF = 1 Boundary"]
        B2["Cagle Gate from yaw.db Stats\nw >= AVG(weight) in stage_conn"]
    end

    Op1 --> Op2 --> S27
```

### 8.1 Operator 1: `pool_confidences` — Logarithmic Opinion Pool (Geometric Mean)
Whenever we combine $K$ confidence measurements $x_1, \dots, x_K \in (0, 1]$ of a **single assertion, state transition, query seed, or adjudicated edge**, we use the **Logarithmic Opinion Pool** (geometric mean over the non-`None` known values $\mathcal{S} = \{x_k \mid x_k \neq \text{None}\}$):
$$\text{pool\_confidences}(x_1, \dots, x_K) = \begin{cases} \exp\left(\frac{1}{|\mathcal{S}|} \sum_{x \in \mathcal{S}} \log x\right) = \left(\prod_{x \in \mathcal{S}} x\right)^{1 / |\mathcal{S}|} & \text{if } |\mathcal{S}| \ge 1 \\ \text{None} & \text{if } |\mathcal{S}| = 0 \end{cases}$$

1. **Extraction-Time Mention Confidence ($c_i$)**:
   $$c_i = \text{pool\_confidences}\big(p_{\text{llm}, i}, \, s_{\text{vec}, i}\big)$$
   When `logprobs` are absent (`p_llm = None`), $c_i = s_{\text{vec}, i}$ remains on the exact same $(0, 1]$ scale as when both signals are present ($\sqrt{0.8 \times 0.8} = 0.80$).
2. **Markov Transition Step ($u \xrightarrow{k} v$ in `ppr.py`)**:
   $$A_{v, u}^{(k)} = \pi(k) \cdot \text{pool\_confidences}\big(w_{\text{edge}}(u, v, k), \, p_{\text{node}}(v)\big)$$
   Keeps every Markov step calibrated on $(0, 1]$ without square-penalizing nodes with known priors.
3. **Query Restart Seeding ($p_0$ in `retrieve_with_seeds`)**:
   $$p_0(x) = \text{pool\_confidences}\big(s_{\text{vec}}(q, x), \, p_{\text{item}}(x)\big)$$
4. **Adjudicated `1`-Hop Taxonomic & Associative Edge Atomic Confidence (`planner.py` & `shacl.py`)**:
   $$c_{\text{edge}}(A, C) = \text{pool\_confidences}\big(p_{\text{llm}}, \, s_{\text{vec}}(A, C)\big)$$
   *(Note: Multi-hop `broader_tc` chain confidence $w_{\text{chain}}(A \to^+ C)$ was retired on `2026-11-08` under Invariant `W5` in §13 because W3C SKOS `skos:broader` is strictly `1`-hop non-transitive.)*

### 8.2 Operator 2: `bayesian_noisy_or` — Additive Shannon Evidence
Whenever $m$ **independent parallel observations** $\{c_1, \dots, c_m\}$ corroborate the same entity, fact statement, or directed node pair $(u, v)$:
$$P(\text{true} \mid c_1, \dots, c_m) = 1 - \prod_{i=1}^{m} (1 - c_i) \in (0, 1)$$
Computes `entities.confidence`, `facts.confidence`, `edges.weight`, and parallel edge aggregation in `build_transition_matrix`.

---

## 9. Empirical Bayes `1`-Hop SHACL S27 Resolution & Cagle Reification (`W5`)

When `ConceptShape/S27-disjointBroaderRelated` detects a direct `1`-hop collision where concept pair $(A, C)$ has both a direct `skos:broader(A, C)` edge with confidence $w_{\text{broader}}(A, C)$ and a direct `skos:related(A, C)` / `skos:related(C, A)` edge with confidence $w_{\text{rel}}(A, C)$:

1. **Step 1: Check for Existing Reified Fact in `facts` (Deterministic Cagle Short-Circuit)**:
   - Query `facts` for any existing `rdf:Statement` between $A$ and $C$.
   - If a `fact` row already links $A$ and $C$, delete `skos:related(A, C)` and `skos:related(C, A)` (`remove_conflicting_related`) immediately with zero topological loss.
2. **Step 2: Empirical Database Means Gate for Cagle Reification (`reify_related_to_fact`)**:
   - Query live empirical means from `stage_conn`: $\bar{w}_{\text{broader}} = \text{AVG}(\text{weight})$, $\bar{w}_{\text{rel}} = \text{AVG}(\text{weight})$, $\bar{w}_{\text{fact}} = \text{AVG}(\text{confidence})$.
   - Cagle Reification triggers if and only if:
     - $w_{\text{broader}}(A, C) \ge \bar{w}_{\text{broader}}$ and $w_{\text{rel}}(A, C) \ge \bar{w}_{\text{rel}}$ (both have positive empirical Z-score in `stage_conn`).
     - $A$ and $C$ co-occur in at least one shared chunk $c^* \in \mathcal{C}(A) \cap \mathcal{C}(C)$.
     - The reified triple $f_{\text{new}} = (A, p, C)$ achieves atomic confidence $c(f_{\text{new}}, c^*) \ge \bar{w}_{\text{fact}}$.
   - When all three hold, insert `rdf:Statement` $(A, p, C)$ with `prov:value` into `facts` and delete `skos:related(A, C)` and `skos:related(C, A)`.
3. **Step 3: Direct Bayes Factor Comparison ($w_{\text{broader}}(A, C)$ vs. $w_{\text{rel}}(A, C)$)**:
   - **Case A: $w_{\text{broader}}(A, C) \ge w_{\text{rel}}(A, C)$**: Delete `skos:related(A, C)` and `skos:related(C, A)` (`remove_conflicting_related`).
   - **Case B: $w_{\text{rel}}(A, C) > w_{\text{broader}}(A, C)$**: Keep `skos:related(A, C)` (and `skos:related(C, A)`) intact at $w_{\text{rel}}(A, C)$, and delete the direct `1`-hop `(A, C, 'skos:broader')` edge from `edges` and `edge_sources` (retiring the former multi-hop `path_finder` CTE and `len(path_edges) >= 2` edge conversion).

---

## 10. Minimal Standard W3C Prefix Header

Zero custom `yaw:` or `dct:` namespaces:
```python
W3C_PREFIX_HEADER: str = (
    "@prefix : <urn:graph:> .\n"
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
    "@prefix prov: <http://www.w3.org/ns/prov#> .\n"
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
)
```

---

## 11. Oblivion Demolition Table (Slice 5)

| Retired Symbol / Parameter | Previous Location | Replacement | Verification Gate |
| :--- | :--- | :--- | :--- |
| `datatype_property_weight` | `planner.py`, `engine.py` | Deleted; literals live exclusively in `facts` with `prov:value` | `git grep -w "datatype_property_weight"` $\to$ `0` matches |
| `datatype_property`, `parallel_aggregation`, `parallel_cap` | `types.py`, `config.py` | Deleted; parallel edges aggregate via `bayesian_noisy_or` | `git grep -E "(parallel_aggregation|parallel_cap)" yaw/` $\to$ `0` matches |
| `.5` modulo hack on edge weights | `ppr.py:149-150` | Deleted; single 4-tuple typed transition matrix | `git grep "base_w % 1.0" yaw/` $\to$ `0` matches |
| `base_w * COUNT(*)` linear weight inflation | `graph_ops.py:1459-1479, 1929-1955` | `bayesian_noisy_or` complement aggregation | `git grep "COUNT(\*)" yaw/daemon/graph/storage/graph_ops.py` $\to$ `0` edge weight matches |
| `get_graph_topology(typed=False)` | `graph_ops.py:1703-1731` | Single typed `(chunks, entities, typed_edges, entity_priors)` | `git grep "get_graph_topology(typed=" yaw/` $\to$ `0` matches |
| `is_literal_node` (`a rdfs:Literal`) on `entities` | `shacl.py:250-255, 287-291` | Deleted; `entities` are strictly `skos:Concept` | `git grep "is_literal_node" yaw/` $\to$ `0` matches |

---

## 12. Strict `JudgeLabels` Identity Equivalence & Canonical Gemma 4 3-Stage Cascade Without Python Token Overrides or Auto-Merge Gates

Empirical audit of `.yaw/yaw_optimize_20261107_110501.db` (`703` published chunks, `767` extracted chunks, `2,293` entities, `2,338` facts `[observed: .yaw/yaw_optimize_20261107_110501.db, 50be17d]`) and `.yaw/dspy_cache` (`6,162` `JudgeLabels` pair decisions, `83` SHACL repair calls `[observed: .yaw/dspy_cache, 50be17d]`) established two ontological defects in Stage 1 adjudication and `planner.py`:
- **Contradictory `JudgeLabels` merge branches & `240`-char reasoning truncation**: Across `6,162` Stage 1 decisions (`2,892` `equivalent=true`, `679` `equivalent=false, alt_label=true`, `608` `equivalent=false, hidden_label=true`, `1,983` `all false`), `4,179 / 6,162` (`67.8%`) pairs were merged because `parse_label_decisions`, `YawProgram.judge_synonyms`, and `apply_synonym_adjudication` executed `merge_entities` whenever `equivalent or alt_label or hidden_label` was `True`. In `1,114 / 1,287` (`86.6%`) of the `equivalent=false, alt_label/hidden_label=true` decisions, the model's `reasoning` explicitly concluded that the `2` entities were **distinct**, yet emitted `alt_label: true` or `hidden_label: true` after hitting the `240`-character `reasoning-str` cap (`68.8%` of `alt_label=true` decisions hit `>= 238` chars mid-sentence). This collapsed `323 / 2,293` (`14.1%`) entities into mega-hubs (`Noah Patel` merged `7` distinct people; `Priya Rao` merged `7` distinct people; `standard tier` merged `Cost-efficient tier`, `Fast tier`, `fast tier p99 regression`, corrupting `["scheduler", "is suspected of causing", "fast tier p99 regression"]` into `["Scheduler", "is suspected of causing", "standard tier"]`; `Latency` merged `18` distinct metric/failure labels).
- **Post-hoc Python token overrides & cross-label auto-merge gates in `planner.py`**: Despite [graph-storage.md](../.agents/rules/graph-storage.md) and [inference.md](../.agents/rules/inference.md) forbidding Python regex/token overrides, `is_modifier_head_specialization` (`longer_toks[:len(shorter_toks)] == shorter_toks`) and `_has_disjoint_non_acronym_tokens` in `planner.py` intercepted Stage 1 decisions and rewrote English `[Modifier, Head]` compounds into inverted `skos:broader` assertions (`Tenant-id skos:broader Tenant`, `Retention policy skos:broader Retention`, `Telemetry-team skos:broader Telemetry`, `CustomerName skos:broader Customer`), while `auto_merge_cosine_gate` / `auto_merge_difflib_gate` merged distinct normalized labels before `JudgeLabels` was ever called.

### Invariant W1 (Strict Equivalence Merge Gate)
- Under W3C SKOS, `skos:altLabel` and `skos:hiddenLabel` are lexical properties of `1` `[target]` unified concept (`equivalent: true`), never relations between `2` `[target]` non-equivalent concepts (`equivalent: false`). For any candidate pair $(A, B)$ evaluated by `JudgeLabels`:
  $$\text{merge}(A, B) \iff \text{equivalent}(A, B) = \text{true}$$
- **Two-branch `equiv-branch` GBNF grammar (`build_dspy_label_judge_grammar`)**:
  - When `equivalent = true`: emits `"equivalent": true, "definition": <reasoning-str>, "hidden_label": <boolean>`, where `definition` is a non-empty unified `skos:definition` string (`1..360` `[target]` chars) and `hidden_label` selects `relation = "skos:altLabel"` (`false`) vs. `relation = "skos:hiddenLabel"` (`true`).
  - When `equivalent = false`: the JSON object closes immediately after `"equivalent": false`; `parse_label_decisions` sets `alt_label = False, hidden_label = False, relation = None`, `validate_stage_invariants("label_judge")` rejects any `equivalent: false` decision carrying `alt_label=True` or `hidden_label=True`, and $(A, B)$ **must** advance to Stage 2 (`JudgeHierarchy`).
- **Reasoning budget & memoization**: `YawConfig.gbnf_max_reasoning_chars` defaults to `360` `[target]` and is forwarded through `YawGBNFAdapter.resolve_grammar` and `YawProgram._run_judge_stage` to `build_dspy_{label,hierarchy,related}_judge_grammar`. `judge_synonyms` in `yaw/daemon/graph/inference/components.py` is bumped to `version=2` `[target]`.

### Invariant W2 (Same-Norm Pre-Consolidation Only & Canonical Gemma 4 Ontology SSOT)
- Pre-LLM sense consolidation in `prepare_synonym_candidates` merges $(A, B)$ if and only if:
  $$\text{text\_processing}(\text{name}_A) = \text{text\_processing}(\text{name}_B) \;\land\; d_{\text{sim}}(A, B) \ge \tau_{\text{syn}} \;\land\; \neg\text{is\_active\_fact\_pole}(A, B)$$
- Cross-label candidate pairs ($\text{text\_processing}(\text{name}_A) \neq \text{text\_processing}(\text{name}_B)$) are **never** merged or re-typed by Python heuristics in `planner.py`. `_is_acronym`, `is_modifier_head_specialization`, `_has_disjoint_non_acronym_tokens`, `auto_merge_cosine_gate`, `auto_merge_difflib_gate`, `_AUTO_MERGE_COSINE_GATE`, and `_AUTO_MERGE_DIFFLIB_GATE` are retired across `yaw/` (`0` `[target]` matches, superseding the former Guardrail 2).
- **Active W3C Semantic Guardrails**:
  1. **Homograph-Only `owl:differentFrom` (`apply_synonym_adjudication`)**: Emit `owl:differentFrom` when Stage 3 (`related_judge`) returns `related: false` **only** if `text_processing(name_a) == text_processing(name_b)` or `lexical_sim >= avg_alt_label_sim` on `stage_conn`; otherwise emit no edge.
  2. **Evidence-Based Identity Across Batches**: Canonical representatives (`MIN(sense_id)`) and surviving support are independent of batch/packet arrival order.
  3. **Falsification in `JudgeLabels` & `JudgeHierarchy` (`prompts.py`)**: `JudgeLabels` enforces `1`-sentence falsification against merging distinct `owl:NamedIndividual`s (given names, surnames, numbered tiers, versions, percentiles) or modifier-head specializations; `JudgeHierarchy` requires strict `skos:broaderG` (*"Every A is a kind of B"*), `skos:broaderI` (individual under role/class), or `skos:broaderP` (component/part of whole system) and rejects cross-category hierarchy assertions.

---

## 13. Universal Irreflexivity, Tier 3 Deterministic Statement Definition Synthesis, Single-Surface `skos:prefLabel`, Non-Transitive `1`-Hop `skos:broader` S27 Disjointness, and Scheme-Free Batch-Independent Semantics

### 13.1 Core Statement, Label & Hierarchy Invariants (`W3`–`W5`)

1. **Invariant W3 (Universal Statement Irreflexivity Across Entity & Literal Objects)**:
   - In `.yaw/yaw_optimize_20261107_110501.db`, `15` facts (from `chunk-868065a438a55fa5ea8a62d2d06e4231`) were pseudo-reflexive literal triples (`obj_kind = 'literal'`) where `subj == obj` and `pred == subj_def` (e.g., `["burst logic", "logic controlling request bursts", "burst logic"]`) because `literal-slot` allows arbitrary strings and irreflexivity was checked only for `obj_kind == 'entity'`.
   - `is_irreflexive_statement(subj, pred, obj, subj_def="", obj_def="", obj_kind="entity") -> bool` in `yaw/core/types.py` (exported in `yaw/core/__init__.py`) enforces across `build_evidence`, `materialize_entities_and_facts`, `rekey_facts`, `validate_stage_invariants("triples")`, `StatementShape/irreflexive`, and `auto_heal_deterministic_shacl` that no published fact $f \in \text{facts}$ may satisfy:
     $$\text{text\_processing}(\text{subj\_name}(f)) = \text{text\_processing}(\text{obj\_lex}(f)) \;\lor\; \left(\text{obj\_kind}(f) = \text{'literal'} \land \text{text\_processing}(\text{pred}(f)) = \text{text\_processing}(\text{subj\_def}(f))\right)$$
   - Every raw `FactAssertion` for an irreflexive entity or literal triple is preserved in `fact_assertions` with `fact_id = NULL`, while `normalize_openie_record` (`normalization.py:136-137`) continues to raise `ValueError` only when `not is_literal_obj` so `extract_chunk` remains at `version=2` `[target]` and all `767` cached OpenIE chunks remain valid (`publish_graph` bumped to `version=2` `[target]`).
2. **Invariant W4 (`3`-Tier `[target]` Deterministic Statement Definition Resolution)**:
   - Across `83` cached `ProposeSHACLRepairs` LLM calls in `.yaw/dspy_cache`, `82 / 83` (`98.8%`) were `set_definition` actions on `fact-*` focus nodes where `spo_def` duplicated `subj_def` (`60.0%`), `obj_def` (`36.3%`), or was empty (`3.8%`), caused by `canonicalize_fact_definitions` falling back to `min(assertions, ...)`.
   - For every $f \in \text{facts}$ with subject definition $d_s$, object definition $d_o$ (when `obj_kind == 'entity'`), and predicate $\text{pred}$, let $\text{valid}(d) \iff d.\text{strip}() \neq \text{''} \land d.\text{strip}().\text{lower}() \neq d_s.\text{strip}().\text{lower}() \land (\text{obj\_kind} \neq \text{'entity'} \lor d.\text{strip}().\text{lower}() \neq d_o.\text{strip}().\text{lower}())$. `canonicalize_fact_definitions(conn, prior_definitions)` resolves `f.definition` by strict `3`-tier `[target]` precedence (with `0` `[target]` LLM healer calls while keeping `fact_assertions.raw_json` immutable and reserving `heal_shacl_violations` for entity-level semantic violations):
     1. **Tier 1 (Valid Source Assertion)**: First `spo_def` in `fact_assertions` (`ORDER BY assertion_id ASC`) satisfying $\text{valid}(\text{spo\_def})$.
     2. **Tier 2 (Valid Prior / Current Repair)**: `prior_definitions.get(fid)` or `f.definition` satisfying $\text{valid}(d)$.
     3. **Tier 3 (Deterministic Canonical Synthesis)**: `synth_def = f"{es.name} {f.pred} {eo.name if f.obj_kind == 'entity' else f.obj_literal_lex}.".strip()` when $\text{valid}(\text{synth\_def})$ holds (and `f"{synth_def[:-1]} (statement)."` if `synth_def` itself collides with $d_s$ or $d_o$).
3. **Single-Surface `skos:prefLabel` & Punctuation-Free Label Hygiene**:
   - `clean_surface_label(raw: str) -> str` (`raw.strip().strip(":;,").strip()`) in `yaw/core/types.py` (exported in `yaw/core/__init__.py`) strips stray leading/trailing clause delimiters (`:;,`), and `preferred_surface` (`normalization.py`) orders candidates by `(text_processing(value) == label_norm, value != value.lower(), -len(value), value)`.
   - `canonicalize_entity_labels_edc`, `upsert_entity_labels`, and `upsert_entities` apply `clean_surface_label` and update `SET label_kind = ?, raw_label = ?` using `best_raw_by_norm[norm]` (`winning_raw` for `winning_norm`) so every entity satisfies `SELECT COUNT(DISTINCT raw_label) FROM entity_labels WHERE entity_id = ? AND label_kind = 'skos:prefLabel'` `== 1` `[target]`.
4. **Invariant W5 (Strictly Non-Transitive `1`-Hop `skos:broader` & `1`-Hop S27 Disjointness)**:
   - Under W3C SKOS (`03-SEMANTIC-RELATIONS.md:45-59`), `skos:broader` is **non-transitive**: `A skos:broader B` and `B skos:broader C` does **not** entail `A skos:broader C`. Because nothing in YAW adjudicates a separate transitive property (`skos:broaderTransitive`), `edges` (`kind = 'skos:broader'`) stores **only** asserted `1`-hop direct hierarchical links. `query_broader_tc`, `broader_tc_pairs`, `"broader_tc"`, the Rule W3 pre-gate in `YawProgram.judge_synonyms`, and `ConceptShape/broaderTransitiveCapExceeded` are retired (`0` `[target]` matches).
   - Directed `skos:broader` graphs remain cycle-free (`ConceptShape/broaderCycle`: no $A \xrightarrow{\text{skos:broader}^+} A$, healed via a bounded fixed-point loop `for _ in range(16)` in `auto_heal_deterministic_shacl`), and `ConceptShape/S27-disjointBroaderRelated` enforces **direct `1`-hop disjointness** between `skos:broader` and `skos:related`:
     $$\forall (A, B) \in \text{entities}^2:\quad \neg\big((A, B, \text{'skos:broader'}) \in \text{edges} \;\land\; ((A, B, \text{'skos:related'}) \in \text{edges} \lor (B, A, \text{'skos:related'}) \in \text{edges})\big)$$
   - `plan_graph_publication` re-runs `auto_heal_deterministic_shacl(staging_conn)` immediately after `apply_shacl_repairs(staging_conn, repairs, merge_entities_fn=merge_repaired_entities)` so any post-merge cycle, `1`-hop S27, or irreflexive collision is deterministically healed before `validate_graph_state`.

### 13.2 Capability Gates & Removal Inventory (`W1`–`W5`)

| Behavior relied on | Locked version | Probe | Pass criterion | Stop rule |
| --- | --- | --- | --- | --- |
| `CG-W3C-1`: `build_dspy_label_judge_grammar` (`max_reasoning_chars=360` `[target]`) and `parse_label_decisions` enforce `merge(A, B) <=> equivalent=true` without exceeding `SynonymRequestBudget` | `dspy == 3.1.2` `[measured: uv.lock, c98aebe]` | `tests/daemon/graph/inference/test_parsers.py::test_label_judge_grammar_and_parser_forbid_non_equivalent_merges` | Grammar requires `"equivalent": false` to close immediately without `alt_label`/`hidden_label` and `"equivalent": true` to include `"definition"` and `"hidden_label"`; parser maps `equivalent=False` to `relation=None`; `SynonymRequestBudget` reserves `360`-char `[target]` reasoning per pair | Stop and amend this spec with the evidence |
| `CG-W3C-2`: Tier `3` `[target]` deterministic synthesis in `canonicalize_fact_definitions` + irreflexive literal filtering + `1`-hop non-transitive `ConceptShape/S27-disjointBroaderRelated` heals `.yaw/yaw_optimize_20261107_110501.db` (`703` `[observed: .yaw/yaw_optimize_20261107_110501.db, 50be17d]` chunks) with `0` `[target]` SHACL violations and `0` `[target]` LLM healer calls | `sqlite-vec == 0.1.9`, `cocoindex == 1.1.0` `[measured: uv.lock, c98aebe]` | `tests/daemon/graph/test_evidence_revisions.py::test_deterministic_fact_definition_synthesis_and_irreflexive_literal_pruning` | `lint_graph_shacl(mem_conn) == []`, `validate_graph_state(mem_conn)` passes, `15` `[target]` pseudo-reflexive literal facts are pruned from `facts` with `fact_assertions.fact_id IS NULL` preserved, `2`-hop `skos:broader` + `skos:related` chains produce `0` `[target]` S27 violations, and `0` `[target]` `heal_shacl_violations` calls are triggered on endpoint-duplicating `spo_def` | Stop and amend this spec with the evidence |

| Retired item | Location (`c98aebe`) | Replacement or "none" | Exit check |
| --- | --- | --- | --- |
| Python token/acronym overrides in planner: `_is_acronym`, `is_modifier_head_specialization`, `_has_disjoint_non_acronym_tokens` | `yaw/daemon/graph/planner.py:530-585, 643-647, 807-814` | `JudgeLabels` -> `JudgeHierarchy` -> `JudgeRelated` Gemma 4 cascade + topological `_is_active_fact_pole` guard | `git grep -n -e "_is_acronym" -e "is_modifier_head_specialization" -e "_has_disjoint_non_acronym_tokens" yaw/ tests/` has `0` `[target]` matches |
| Cross-label Python auto-merge gates: `auto_merge_cosine_gate`, `auto_merge_difflib_gate`, `_AUTO_MERGE_COSINE_GATE`, `_AUTO_MERGE_DIFFLIB_GATE` | `yaw/core/config.py:183-184, 249-252, 440-441, 501-502`, `yaw/daemon/graph/engine.py`, `yaw/daemon/graph/planner.py:98-99, 593-594, 655, 1182-1183`, `yaw/daemon/indexing.py`, `yaw/daemon/runtime.py` | Same-norm sense consolidation (`text_processing(name_a) == text_processing(name_b) and d_sim >= synonymy_threshold`) | `git grep -n -e "auto_merge_cosine_gate" -e "auto_merge_difflib_gate" -e "_AUTO_MERGE_COSINE_GATE" -e "_AUTO_MERGE_DIFFLIB_GATE" yaw/ tests/` has `0` `[target]` matches |
| Non-equivalent merge branches (`equivalent: false, alt_label: true` / `equivalent: false, hidden_label: true`) | `yaw/daemon/graph/inference/{prompts,grammars,parsers,invariants,program}.py` | `2`-branch `[target]` `equiv-branch` grammar (`true` -> `"definition"` + `"hidden_label"`; `false` -> closes immediately) | `CG-W3C-1` unit & grammar tests pass |
| Multi-hop transitive `broader_tc` closure over non-transitive `skos:broader`: `query_broader_tc`, `broader_tc_pairs`, `"broader_tc"`, Rule W3 pre-gate, and `ConceptShape/broaderTransitiveCapExceeded` | `yaw/daemon/graph/storage/{synonyms,__init__,shacl}.py`, `yaw/daemon/graph/inference/program.py:1084-1115` | Direct `1`-hop `skos:broader` + `skos:related` S27 disjointness and fixed-point `ConceptShape/broaderCycle` healing | `git grep -n -e "query_broader_tc" -e "broader_tc" -e "broaderTransitiveCapExceeded" yaw/ tests/` has `0` `[target]` matches |
| Invalid-assertion fallback (`min(assertions, ...)`) in `canonicalize_fact_definitions` | `yaw/daemon/graph/storage/evidence.py:254-255` | Tier `3` `[target]` deterministic synthesis `f"{subj_name} {pred} {obj_surface}."` | `CG-W3C-2` unit & replay tests pass |

### 13.3 Scheme-Free, Batch-Independent Semantics

The former emergent ConceptScheme phase is removed, not postponed behind a runtime
switch. [Native graph publication](native-cocoindex-graph.md) is the controlling
implementation and acceptance path. Remove clustering, scheme definitions and
embeddings, scheme/member tables, top-concept membership, scheme PPR/QA integration
and all schema/telemetry consumers together. Retain ordinary entity definitions,
source-supported facts, labels, identity decisions and supported semantic edges.

The semantic acceptance invariant is: for fixed evidence and adjudications,
`64+64` chunks and `128` chunks together yield identical canonical bindings, fact
keys, typed edges, source supports and entity/fact/edge counts. Floating aggregate
confidence may differ by at most `1e-6`. The graph plan owns live-model tolerances
and cumulative timing measurement; this document does not define another gate.

Necessary semantic changes:

1. Batch/document boundaries do not scope identities or candidates. Normalize
   occurrences consistently, reuse persisted accepted senses and include both
   new-to-existing and new-to-new candidates. Late evidence invalidates affected
   earlier negative decisions and merges.
2. Candidate coverage and evidence dossiers must be independent of arrival.
   A new endpoint can alter an old top-$k$ neighborhood; handle its reverse
   dependencies instead of assuming only new endpoints ever need a query.
3. Pair/evidence/policy keys own decisions, including no-relation outcomes.
   Merely finding a related/broader/different edge is not proof that a decision
   remains applicable after evidence changes.
4. Choose canonical identity deterministically from accepted equivalence, never
   from arrival order or string similarity alone. Preserve raw arguments/supports
   so a merge can be revised without losing source assertions. Label/cosine
   matches generate candidates; they do not prove equivalence or distinctness.
5. Apply relation conflicts and deterministic repairs in canonical order and
   verify convergence on the same fixed evidence. Do not make packet composition
   or first-seen judgment an irreversible semantic input.
6. Retain provenance, literal typing, label consistency, supported hierarchy and
   relation constraints. Validate their affected closure transactionally. Remove
   scheme-membership constraints rather than inventing empty schemes to pass them.

Semantic islands are allowed. W3C SKOS does not require global connectivity; SHACL
validates declared constraints, not discovery completeness. A partial `64`/`128`-chunk
document is reported as incomplete, and only becomes semantically ready when all
its desired chunks and affected decisions are committed. No fake connection or
identity merge may be introduced simply to join components. Shared document
provenance is not a semantic PPR edge.

## 14. Evidence gate for any future scheme proposal

Reintroduction requires a separate design/benchmark against the completed
scheme-free baseline, with identical source/model/slice and predeclared Recall@N,
Exact Match or F1 objective. Report all supported quality metrics, paired
uncertainty, ingestion/publication cost, QA tokens and peak worker memory. A
repeatable held-out gain must justify resource/maintenance cost without
unacceptable regressions elsewhere. A richer-looking graph or extra node count
is not evidence. No clustering dependency or dormant scheme code is introduced
in the current refactor.

## Decisions

- 2026-11-06: "Should stay literals, NER is only allowed to capture skos:Concept and owl:NamedIndividual, triple extraction can point the rdf:Statement to rdf:object = rdfs:Literal but never rdf:subject = rdfs:Literal! This is why it's two stage. NER creates the allowed rdf:subject vocabulary, with predicates able to point towards literals but never the other way around - that's the W3C ontology rules and should be in our allowed SHACL shapes." (User). Enforced across GBNF grammars, extraction/repair parsers, Active EDC label canonicalization (§3.2), and closed-world SQLite SHACL shapes.
- 2026-11-07: "Audit and review for root cause for batch indexing regressing back to exponential time behavior. Incremental indexing and graph publication was designed and tested to be near-constant time for 64 + 64 chunk batches vs 128 chunk batchless indexing." / "It’s a SHACL Violation CONSISTENCY_ERROR" / "you will need to /spec first." (User). Preserves SHACL-healed entity labels across incremental batches via `chunk_id = "__healed__"` (`cid.startswith("__")` -> `verbatim_in_source = 1`, `is_hidden = False`), marks scalar/non-resource labels `is_hidden = True` in `canonicalize_entity_labels_edc`, and scopes Active EDC to `touched_entity_ids` (§3.2).
- 2026-11-08: "Option 1 (`skos:broader` is strictly 1-hop non-transitive; retire `query_broader_tc` and scope S27 to direct 1-hop collisions)" / W3C ontology simplification & deterministic statement canonicalization ([native-cocoindex-graph.md §4](native-cocoindex-graph.md#4-semantics-and-invariants), [epistemic-confidence-spec.md §§1.3, 2.4](epistemic-confidence-spec.md#13-empirical-bayes-1-hop-shacl-s27-resolution-zero-magic-thresholds-invariant-w5), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [incremental-publication-work.md](reviews/incremental-publication-work.md)): (1) enforces Invariant `W1` in `JudgeLabels` (`merge(A, B) <=> equivalent(A, B) = true`, `2`-branch `equiv-branch` GBNF grammar, `gbnf_max_reasoning_chars = 360`, `judge_synonyms` `version=2`), superseding the `equivalent: false, alt_label/hidden_label: true` merge branches; (2) enforces Invariant `W2` in `planner.py` (same-norm pre-consolidation only, retiring `_is_acronym`, `is_modifier_head_specialization`, `_has_disjoint_non_acronym_tokens`, `auto_merge_cosine_gate`, and `auto_merge_difflib_gate`), superseding former Guardrail 2; (3) enforces Invariant `W3` (`is_irreflexive_statement` across entity and literal objects, preserving raw `fact_assertions` with `fact_id = NULL` while keeping `extract_chunk` at `version=2` and bumping `publish_graph` to `version=2`); (4) enforces Invariant `W4` (Tier 3 deterministic `f"{subj_name} {pred} {obj_surface}."` synthesis in `canonicalize_fact_definitions`), superseding the `2026-11-07` invalid-assertion fallback (`min(assertions, ...)`); (5) enforces single-surface `skos:prefLabel` and `clean_surface_label`; and (6) enforces Invariant `W5` (strictly non-transitive `1`-hop `skos:broader`, retiring `query_broader_tc`, `broader_tc`, Rule W3 pre-gate, and `ConceptShape/broaderTransitiveCapExceeded`, scoping `ConceptShape/S27-disjointBroaderRelated` to direct `1`-hop collisions, healing `ConceptShape/broaderCycle` via fixed-point iteration, and re-running `auto_heal_deterministic_shacl` after `apply_shacl_repairs` in `plan_graph_publication`).
