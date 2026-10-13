# FOUNDATIONAL.md: Product Requirements & Architectural Specification for YAW

**Status:** Authoritative target architecture, reconciled 2026-10-12\
**Implementation state:** [Epic 7.8](docs/epic7.8-open-ie-ontology-alignment.md) is in progress. Documentation alignment does not certify runtime conformance.\
**Decision rationale:** [ADR-013](docs/adr/ADR-013.md), amended by [ADR-014](docs/adr/ADR-014.md)\
**Baseline evidence:** [Open IE and ontology review](docs/reviews/open-ie-ontology-review-2026-10-12.md)

## 0. Progressive Disclosure Architecture & Reading Guide

This document owns product invariants and cross-component contracts. ADRs record reasons and explicit amendments; guides own selected component algorithms; Epic 7.8 owns its work/completion checklist; the roadmap owns cross-epic sequence; TEST_READY owns verification evidence. Ontology resources own domain semantics. A copied example or historical implementation cannot override those owners.

| Read next | Purpose |
| --- | --- |
| [ADR ledger](docs/ADR.md) | Current decisions and supersession map |
| [Roadmap](docs/epics-alignment-roadmap.md), [Epic 7.8](docs/epic7.8-open-ie-ontology-alignment.md) | Work order, restart checklist and evidence required for completion |
| [spaCy integration](docs/how-to-handle-spacy-llm.md) | Public APIs and extraction/grounding implementation |
| [Continuous compilation](docs/knowledge-compilation.md) | The single application, stage order, publication, retraction and explicit errors |
| [Semantic substrate](docs/rdf-owl-shacl.md), [ontology integration](docs/epic7-pluggable-ontology-integration-spec.md) | Resource loading, interpretation and graph projection |
| [PPR retrieval](docs/power-pagerank.md), [hierarchy](docs/ontological-roots.md) | Retrieval algorithm and induced organization |
| [Acceptance scenarios](docs/USER_STORIES.md), [test infrastructure](TEST_INFRA.md), [readiness](TEST_READY.md) | Observable behavior and verification status |
| [Research foundation](docs/open-ie-authoritative-source.md) | Primary literature and limits of its applicability |

Documents explicitly marked historical, superseded or upstream reference are not live implementation instructions. Status claims require a commit, environment, commands and results; historical test counts do not certify Epic 7.8.

## 1. Vision & Core Philosophy

YAW is a local-first, continuously maintained knowledge compiler and associative retrieval engine over a Markdown vault. spaCy and spacy-llm discover entities, concepts and qualified relationships from notes. RDF/OWL/SKOS/SHACL resources express their semantics and constraints. CocoIndex incrementally maintains one SQLite substrate; SciPy retrieves through its graph; FastMCP exposes ingestion, compilation, retrieval and linting. Wiki materialization is optional.

The intended workload is **100–500 documents**, primarily English with occasional other languages. Optimize for supported answers, preserved disagreement, correct revisions and maintainable accumulated knowledge. Measure source tokens, claim/graph size, model calls and latency; no processing-time SLA has been set. The engine must beat a simple retrieval baseline on the declared quality criteria before a quality advantage is claimed.

Knowledge must emerge without requiring the author to design an exhaustive ontology first. Imported ontologies enrich interpretation. Unknown entities and predicates remain representable and queryable in an open view. Learned definitions and hierarchy are identifiable derived knowledge with provenance; they are not silently promoted into logical axioms.

Single-stage ingestion means new evidence is queryable without waiting for a separate batch consolidation barrier. Parsing, alignment, validation and projection are logical responsibilities within that incremental flow, not separate external databases or mandatory batch stages.

There is **one production implementation and zero fallbacks**. The selected application and its states are specified in the [compiler guide](docs/knowledge-compilation.md). Missing dependencies and failed stages return explicit errors; they never select another extractor, model, parser, compiler or scorer. Legitimate empty inputs, pending compilation, compatible cache reuse and an explicitly scoped view are states/data of this same path. All Epic 7.8 runtime gates must pass before any Epic 8 implementation begins.

## 2. The Seven Invariants of YAW

### Invariant 1: Unidirectional Substrate Mutation

The engine never edits or deletes existing human-authoritative raw notes. Explicit `ingest_raw` requests may append new source notes. Generated wiki material belongs in `vault/wiki/`; engine state belongs in `.cache/yaw.db`.

For a given source revision, original extraction records are immutable with respect to alignment, compilation and view configuration. Source edits produce a new revision and invalidate affected derived work; this is not a requirement to retain obsolete active rows forever. Preserve an audit trail for interpretation changes. Induced hierarchy lives separately from source assertions.

### Invariant 2: Zero-Prose Disk Pointers

SQLite stores source/revision identifiers, occurrences, offsets and extracted annotations, not duplicate full chunk prose. FTS uses contentless indexing. Source files remain the prose authority. LLM-authored wiki blocks are separately stored derived content with evidence and dependency versions; this does not make them a second source authority.

Hydration verifies the indexed revision against current bytes. If it differs or is missing, return an explicit stale/unavailable-evidence condition and schedule refresh; never label a different slice as verified evidence. Character spans and line citations follow the source identity and coordinates contract; byte coordinates, when exposed, are derived against the same verified revision.

### Invariant 3: Single In-Process SQLite Database

`.cache/yaw.db` holds relational, vector and graph state. There are no separate vector/graph databases or external graph servers. RDF serialization, the supported inference profile, SHACL validation and SciPy traversal execute in-process.

Use WAL, foreign-key enforcement where applicable, a configured busy timeout and appropriate serialized writes/read connections. Schema/migration definitions have one executable owner under `src/yaw/storage/`; documentation specifies their meaning rather than maintaining competing DDL.

### Invariant 4: Machine Semantics Equal Human Wikilinks

Human wikilinks and source-grounded semantic edges are first-class retrieval inputs, with equal baseline importance before explicit traversal policy. Preserve ambiguous mention candidates instead of forcing one identity.

Graph relevance weights, lexical evidence and normalized alias weights are ranking signals, not calibrated probabilities of truth. Topological ranking must not commit permanent identity merges.

### Invariant 5: Dual Epistemic Retrieval Contract

Retrieval returns both semantic context (nodes, typed paths and relevant assertions) and revision-verified source evidence. Induced, inferred, attributed and uncertain material remains distinguishable.

Production queries perform no generative synthesis or identity adjudication. Return stored context, material competing claims and verified evidence, with generation, coverage and truncation status. An empty result is explicit; semantic context without supporting source evidence must not be represented as a grounded answer. A failed dependency is an error, not an empty result or a reason to switch retrieval algorithms.

### Invariant 6: Real-Invariant Testing

Test real temporary files, SQLite, vector bytes, RDF/SHACL execution and sparse matrices. Use a loopback HTTP model fixture to verify transport and orchestration, and a separately reported actual-model evaluation to verify extraction quality. Do not mock the subsystem whose invariant is being tested.

See [acceptance scenarios](docs/USER_STORIES.md) and [test infrastructure](TEST_INFRA.md).

### Invariant 7: Single Source of Truth and Code Quality

Domain classes, predicates, lexical mappings, inverses, semantic characteristics and validation constraints are declared in versioned RDF/OWL/SKOS/SHACL resources. This includes the built-in profile and learned semantic resources. Python may compile those resources into indexes and implement generic algorithms; it must not maintain an independent domain predicate inventory, synonym/polarity table or SQL whitelist.

Fixed W3C vocabulary constants, record schemas, language-processing algorithms and operational parameters are legitimate code. Resource-derived projections carry their input version and are rebuildable. Configuration chooses profiles and operational settings; an empty optional profile must not restore an undisclosed fixed vocabulary.

Use supported library interfaces, shared schema/configuration owners, explicit types and clean Ruff/type checks. Do not duplicate inventories or implementation code in guides.

## 3. The Model Stack Contract

Retain the local stack and this single implementation recipe. The initial model/configuration profile is measured in Epic 7.8; revisions replace the selected profile through an explicit decision and cache migration, never automatic model selection after failure.

| Job | Baseline | Contract |
| --- | --- | --- |
| Linguistic analysis, space L | `en_core_web_lg` | Source-preserving tokens, morphology, dependencies and NER candidates; English hints never gate discovery or establish non-English semantics |
| Open extraction | `Qwen3-4B-Instruct-2507-GGUF` through `yaw.Ollama.v1` | `spacy.Raw.v1` plus one Pydantic/grounding adapter; one request per changed bounded extraction unit |
| Claim comparison and authored wiki | The same configured Ollama model and Raw adapter | Separate planned, versioned comparison/synthesis payloads at compilation time; no identity adjudication or alternate model |
| Entity candidate similarity, space S | `embed-gemma-300m-FLM` | Symmetric sentence-similarity prompt; candidates only |
| Query/chunk/fact relevance, space R | `embed-gemma-300m-FLM` | Asymmetric query/document prompt pairing; deterministic fact rendering includes assertion qualifiers |
| Alias candidate lookup | `InMemoryLookupKB` | Index of candidates and normalized evidence weights, not an independent similarity store or automatically trained disambiguator |

Model identifiers, prompt formatters and dimensions have one configuration/implementation owner. Persist the resolved model identity/digest and task/prompt/schema versions in extraction-cache identity. Changing model weights behind a tag must invalidate incompatible results.

Keep spaces L/S/R separate ([ADR-012](docs/adr/ADR-012.md), as amended by ADR-013/014). Never compare vectors from different spaces or silently accept wrong dimensions. Predicates use linguistic/resource alignment. Contextual source definitions accompany extraction; organizational labels and cross-document synthesis are scheduled compilation work. Record costs for extraction, comparison and synthesis separately.

The chosen task count and model names are not correctness guarantees. Calibration uses independent labelled data and a separate held-out evaluation. The [spaCy guide](docs/how-to-handle-spacy-llm.md) owns the fixed component recipe and the generation-cache boundary. Do not add optional production NLP stacks, DSPy, heuristic recovery or a trained linker without replacing the current decision and proving its acceptance gates.

## 4. System Components & Architecture

`raw notes → CocoIndex source units → spaCy/Raw qualified claims → shared corpus component → SQLite publication → query/lint/export`

The shared corpus component performs alignment, claim comparison, RDF enrichment/validation, organizational grouping and supported wiki synthesis, in that order. Versioned resources and accepted bindings are explicit inputs. [Continuous compilation](docs/knowledge-compilation.md) owns its dependencies and publication algorithm. One application handles watcher, CLI and MCP requests; no post-sync or manual-only alternate compiler exists.

At 100–500 documents, the selected strategy caches unchanged extraction/generated results and recomputes deterministic corpus projections. Updates retract stale dependencies and preserve other support. Semantic resources can be loaded, revised or removed without overwriting original extraction. View selection changes only its projection; resource changes may rebuild derived work. Numerical limits are versioned configuration, not universal latency promises.

## 5. Component Deep Dive & Business Logic

### Component 1: Single-Stage Ingestion & Leaf Ontology

#### Source identity and coordinates

A source occurrence identifies a vault/source, revision digest and chunk occurrence; a content digest alone is only a cache/deduplication key. Equal text in different notes or positions remains distinct evidence.

The source revision is a digest of the original bytes. Decode UTF-8 without newline normalization. Mention and evidence spans are half-open Unicode-code-point offsets into that exact decoded chunk; chunk-to-file mapping is recorded. Line citations are one-based inclusive. If byte spans are returned, convert from the original decoded text against the same bytes. Do not equate character, token, byte and line coordinates.

Each span must reproduce its cited text. Token-aligned annotations are separate and may not replace exact evidence spans. Preserve nested/overlapping occurrences. Discontinuous evidence is an ordered list of verified spans, not a fabricated contiguous quote.

#### Extraction payload v1 — logical contract

The host supplies source/revision and configuration identifiers; the model does not invent them. The validated output has these responsibilities, independent of the eventual physical table layout:

| Record | Required information |
| --- | --- |
| Envelope | Schema version, source/chunk occurrence, extraction configuration identity, result status and diagnostics |
| Mention | Occurrence ID, exact surface span(s), surface text; optional evidenced type/kind candidates, never a forced identity |
| Assertion | Assertion ID; subject/object references; original emitted predicate phrase and predicate evidence span(s); source evidence; qualifiers |
| Resource argument | Reference to a mention occurrence; resolved identity is a separate interpretation |
| Literal argument | Source occurrence plus lexical value; datatype or language only when supported; preserve units and quantities |
| Qualifier | Explicit polarity/modality/attribution/time/condition information or `unknown`/absent-with-defined-meaning, with evidence references |
| Definition | Subject/object/predicate description with source evidence and `source_supported`, `generated` or `unknown` status; never a fabricated required string |
| Interpretation | Separate alignment ID, candidate/accepted/unresolved status, semantic kind, target identifier, any argument reversal, evidence and resource version |

A model-emitted predicate may be a paraphrase: retain it as emitted and retain the actual supporting source spans separately; do not call the paraphrase a verbatim quote. Source validation checks span agreement, not truth in the world.

Result states distinguish `complete` (including a valid empty payload), `partial` and `failed`; pending/running are host lifecycle states. Schema errors, unmatched spans and model/network failures do not become silent empty success. Keep diagnostics and valid independent records; invalid records are excluded from trusted projection and a partial scope is never marked complete. The compiler guide defines error propagation; source text is never parsed as a substitute model response.

Extract mentions even when they participate in no binary relation. Negation, modality, attribution and conditions must survive fact rendering and retrieval. Uncertain type/identity stays unresolved.

The v1 semantic envelope contains binary assertions with qualifiers, mentions and literals, including evidence across the bounded source context. Resolve a local reference only with evidenced antecedent/context and mark the interpretation. Arbitrary cross-document coreference and n-ary event decomposition are outside this version: preserve their evidence and report unsupported/partial interpretation instead of inventing unconditional binary facts. Nested mentions remain supported. Occasional non-English text uses the same extraction contract; language/coverage diagnostics survive retrieval.

#### Alignment and identity

Raw extraction is open regardless of loaded domain vocabulary. Unknown predicates receive stable, collision-safe open identifiers with source provenance. Equal surface strings are lexical candidates, not proof of one sense. Keep original phrase, direction and qualifiers even after successful alignment.

Ontology-derived lexical frames can associate a phrase with a property and argument orientation. An inverse axiom alone does not identify the meaning of arbitrary language. Lemmas, labels, cosine similarity, acronyms and head containment propose alignments; unresolved cases remain open.

Hard identity requires a source-supported explicit same-kind identity assertion, an authoritative resource identifier used in context, or an owner-authored scoped binding. A sole retrieved candidate, equal normalized key or generated comparison is insufficient. Keep candidate distributions and reversible accepted bindings. There is no generative identity adjudication. A rejected merge does not imply `owl:differentFrom`.

#### Cache and incremental invalidation

CocoIndex memoization is the sole persistent generation-result cache; disable spaCy BatchCache. Cache typed validated results, not mutable Docs. Identity includes exact ordered primary/context input, task/schema version, rendered prompt/examples, generation settings, model identity and linguistic/tokenization configuration. Validate input equality on reuse and namespace extraction/comparison/synthesis entries. No cache incompatibility triggers a second cache/parser path (review R6).

A view toggle preserves raw extraction and compatible embeddings. Changed ontology/lexical/shape resources invalidate their dependent projections. Changed extraction prompts/models require affected re-extraction. Changed fact rendering invalidates affected fact vectors. Changed source revision invalidates its occurrence-dependent records. No universal “zero cache wipe” promise overrides correctness.

### Component 2: Storage & Schema Contracts

The following names identify current implementation families; they are not a claim that the current DDL already implements payload v1.

| Storage family | Ownership |
| --- | --- |
| `note_meta`, `note_chunks`, `wikilinks` | Source identities/revisions, occurrences, exact grounding pointers and explicit links |
| `leaf_nodes`, `leaf_relations` | Original extracted mentions/assertions; Epic 7.8 extends the current three-string model |
| `aliases`, `close_matches`, `canon_verdicts` | Accepted bindings, uncertainty and decision audit, with kind/scope/evidence/version distinctions |
| `dynamic_classes`, `dynamic_properties` and resource registry | Rebuildable projections of loaded/learned semantic resources; preserve the information needed by supported consumers |
| `compiled_nodes`, `compiled_relations` and compiled content/dependency records | Provenanced organization/inference, claim comparisons, stored authored wiki blocks and their invalidation dependencies, separate from raw extraction |
| `entity_vec`, `fact_vec`, chunk vectors | Space-specific representations with matching identity/version/dimensions |
| `compiled_violations` | Validation results keyed by data/resource/shape/profile versions and scope |
| `unified_nodes`, `unified_edges` | Active derived projections; ontology membership and behavior come from resources |

Preserve all required property parents, language-tagged labels, semantic kind, inverse/characteristic information and lexical-frame annotations in the source resources or lossless consumer projections. A small registry snapshot is not a substitute for the authoritative graph.

The existing [RDF module](src/yaw/storage/rdf.py) owns one public identifier constructor, imported directly by all consumers. Retain `http://yaw.local/` with plural kind paths (`entities/`, `concepts/`, `classes/`, `predicates/`) and `chunks/{occurrence_id}`. Occurrences include vault/source, revision and position. Preserve external IRIs exactly; normalized labels and content hashes alone are not occurrence identity. Migration mappings update every reference transactionally and do not imply `owl:sameAs`.

Migrations must be transactional/idempotent and state which data can be rebuilt. Previously overwritten predicate surfaces need re-extraction from available sources; deleting an alias cannot recover lost source data. Preserve stable references or provide migration mappings.

### Component 3: GraphRAG PPR Retrieval Engine

The graph contains source occurrences, admitted semantic resources, uncertain candidate associations and provenance-labelled derived structure. Traversal edges are distinct from asserted RDF facts: a reverse navigation edge is not automatically an inverse logical assertion.

Property-specific direction/traits come from resources; generic traversal weights and calibration parameters are algorithm configuration. Loaded namespaces must not be stripped before deciding a property's semantics.

Retain query-to-fact relevance in space R, alias candidate seeding, chunk vector relevance and FTS in one fixed algorithm. The normalized channel mixture, conductance, dangling-node handling and convergence contract are owned by [the PPR guide](docs/power-pagerank.md). Legitimately empty channels are removed and active weights renormalized; all-empty input returns no evidence. An unavailable service/index or non-convergent computation fails explicitly; there is no offline scorer or alternate retrieval path.

All retrieved facts preserve qualification and evidence references. Source hydration verifies revision/coordinates. Derived paths explain association, not logical proof or truth merely because PPR assigns high weight.

### Component 4: FastMCP Protocol Interface

The production interface remains `ingest_raw`, `compile_wiki`, `query_wiki` and `lint_wiki`.

- `ingest_raw` appends explicitly supplied source material and schedules the single application; it reports a job/source identity and actual processing state, never a fabricated indexed count.
- `compile_wiki` requests refresh of the same application's manifest/components and reports state/generation; it never runs another compiler or rewrites source assertions.
- `query_wiki` returns structured ranked passages, typed context, competing claims, evidence and generation/coverage status. MCP, CLI and evaluation use the same result builder.
- `lint_wiki` reads scoped versioned findings only. Pending/stale/failed scopes are explicit; refresh goes through `compile_wiki`, with no live-validator option.

Query/lint source scope and aggregate status follow the [compiler outcome contract](docs/knowledge-compilation.md#7-zero-fallbacks-and-explicit-outcomes): default to all active sources, resolve explicit source selections before retrieval, and never hide failed required work by choosing different hits.

Loading/selecting an ontology is operational configuration, not permission to discard unknown source assertions. Contract fields, errors and transport behavior require actual FastMCP integration tests. Result-size and graph-size costs must be included in latency claims.

### Component 5: W3C Semantic Web Substrate

RDF represents identified resources, literals and statements. Use an explicit assertion/provenance model for qualified or attributed claims; a named graph alone does not define those semantics. Do not project a negated/modal/conditional claim into the default graph as an unconditional positive fact. [RDF Concepts](https://www.w3.org/TR/rdf11-concepts/).

OWL/RDFS provide formal domain schema and admitted logical axioms. SKOS provides concept organization. The nine-root vault organization is a declared default profile, not an exhaustive domain vocabulary or a rule that all extracted nodes are OWL classes. Maintain the project policy that a URI is not simultaneously an OWL class and a SKOS concept. OWL's own permitted punning is broader than this project policy. [OWL Primer](https://www.w3.org/TR/owl2-primer/).

Projection must distinguish mention, individual, class, concept, property and literal. Instance membership, class subsumption/equivalence, individual identity and concept mapping are different operations. Select SKOS mapping only for concept endpoints, not from their namespace alone. Keep direct thesaurus hierarchy distinct from its transitive closure. [SKOS Reference](https://www.w3.org/TR/skos-reference/).

**Entailment support is explicit.** The [semantic guide](docs/rdf-owl-shacl.md) selects one pySHACL/RDFS/OWL-RL/SHACL-AF sequence with tested subclass/subproperty propagation, types, domain/range, inverse, symmetric/transitive properties, admitted identity and disjointness. Unsupported constructs are diagnosed and excluded from active inference; there is no second Python/SQL closure. This is a tested project profile, not a claim of complete OWL 2 conformance. Non-termination/resource limits cannot publish a complete generation. [OWL 2 Profiles](https://www.w3.org/TR/owl2-profiles/).

SHACL validates a declared data graph against versioned shapes with an explicit entailment regime. Domain/range inference is not closed-world input validation. Compile generic constraint execution from resources; do not enumerate domain contradiction pairs in Python. Validation of opposing claims fixes attribution, time, conditions and assertion status before claiming incompatibility. Structural conformance is not proof of factual truth. [SHACL](https://www.w3.org/TR/shacl/).

## 6. Repository Architecture & Directory Boundaries

- `src/yaw/nlp/`: linguistic components, public task/model adapters, payload validation and embedding formatters.
- `src/yaw/indexer/`: incremental ingestion, occurrence grounding and revisable alignment.
- `src/yaw/ontology/`, `src/yaw/ontology.ttl`: semantic resources, loading, supported inference and validation.
- `src/yaw/storage/`: database schema/migrations and derived graph projection.
- `src/yaw/retrieval/`: graph construction, seed channels, PPR and verified evidence hydration.
- `src/yaw/server/`: FastMCP and portable lifecycle integration.
- `docs/` and repository instructions: engineering material, never ingested automatically into a user's raw vault.
- `vault/raw/`: human-authoritative sources; `vault/wiki/`: optional generated distribution; `.cache/`: engine-owned state.

Windows and Unix share semantic resources and schemas. Platform adapters handle paths, cache locations, locking, process status and launchers. Never use a Unix signal-based liveness assumption as an untested Windows process probe.

## 7. Testing Rigor, Code Cleanliness & Static Hygiene

[USER_STORIES](docs/USER_STORIES.md) owns observable acceptance scenarios; [TEST_INFRA](TEST_INFRA.md) owns test execution categories; [TEST_READY](TEST_READY.md) records actual evidence. The [Epic 7.8 checklist](docs/epic7.8-open-ie-ontology-alignment.md) owns completion status.

Documentation reconciliation requires valid links, explicit supersession, one contract owner and no live requirements for retired paths. Epic 7.8 completion additionally requires the full functioning pipeline, zero-fallback failure tests, actual-model extraction/comparison/synthesis evaluation and platform checks, including retained 7.7 work. Every gate must pass before Epics 8.x start. Keep documentation and runtime results separate; do not infer passing application tests from prose changes.
