# Continuous claim-to-knowledge compilation

**Status:** Selected implementation contract, 2026-10-12; implementation pending [Epic 7.8](epic7.8-open-ie-ontology-alignment.md).\
**Authority:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-014](adr/ADR-014.md). This guide owns compilation order, dependencies, publication and failure behavior. It does not duplicate the payload schema or retrieval equations.

## Workload and one application

The target is 100–500 Markdown documents in `raw/`, mainly English with occasional other languages. Record document lengths and total tokens as well as document counts. Processing-time limits are not yet product SLAs; measure them on the target machine. Sources may disagree, change their position, use aliases or leave facts unstated.

Use one CocoIndex application and one SQLite substrate:

```text
source revisions -> source-preserving extraction units -> qualified claims
                                                        |
resource versions + accepted bindings -----------------+-> active corpus manifest
                                                              |
                             alignment -> comparison -> inference/validation
                                                              |
                                    organization -> supported wiki synthesis
                                                              |
                                      one generation publisher in SQLite
                                                              |
                                 query / cached lint / deterministic Markdown
```

These are dependency stages within the same application. There is no independently scheduled post-sync compiler. CLI indexing, the watcher, `ingest_raw` and `compile_wiki` enter this same application.

`force=True` requests re-evaluation of the requested derived scope through the same application. Re-read/version inputs and rebuild deterministic projections; exact compatible generation results remain reusable. It never deletes `yaw.db`, clears the whole CocoIndex cache, changes models or bypasses admission. A model/prompt/resource change is represented by its changed input identity. Remove the current destructive `sync_vault(force=True)` and cache-mismatch rebuild behavior during migration; missing/mismatched state requires an explicit reconciliation outcome.

Use CocoIndex's public processing components, change-tracked inputs and target-state lifecycle. All persisted leaf, FTS, vector and compiled outputs have a declared owner; one SQLite target adapter owns publication transactions. Remove hidden database side writes inside memoized extraction and mutable live-vocabulary reads from its inputs. [Processing components](https://cocoindex.io/docs/programming_guide/processing_component/), [context](https://cocoindex.io/docs/programming_guide/context/), [target state](https://cocoindex.io/docs/programming_guide/target_state/).

## 1. Source extraction

The [spaCy guide](how-to-handle-spacy-llm.md) owns the single linguistic/Raw/Pydantic recipe. Each unit has an exact primary chunk and a versioned context manifest containing its heading path and adjacent source context. Every context passage has its own source coordinates. Context may resolve a local reference only when both the referring expression and antecedent have evidence; unresolved references remain explicit. Repeated text is never located by choosing its first substring occurrence.

The host bounds units against the configured model's context limit before generation. The same deterministic chunker splits an oversized unit; no second LLM sharder, truncated prompt or smaller-model retry is used. Assertions anchored in an adjacent context passage belong to that passage's primary unit, preventing duplicate assertion ownership. A genuinely cross-boundary assertion can cite multiple occurrences while retaining one primary owner.

Persist validated mentions, qualified assertions, contextual definitions and outcome diagnostics under FOUNDATION's contract. Full source prose stays in the authoritative files. Leaf publication includes FTS and space-R chunk/fact vectors, plus space-S entity-description candidates. Predicates are properties and do not enter the entity-identity vector store.

## 2. Corpus manifest and alignment

The compiler consumes a stable, sorted manifest of active occurrence/result IDs, extraction outcomes, accepted binding revisions, resources, shapes/rules and all relevant configuration/model/schema versions. Hydrated evidence is revision-checked. Filesystem order, wall-clock time and untracked database state cannot determine the result.

At this scale, recompute deterministic alignment/projection and validation from the manifest. Reuse generation outputs only when their complete inputs match. Cache keys for a comparison or topic include membership, so a newly added competing claim invalidates the old summary even when existing claims did not change.

Compile resource labels, lexical frames and semantic traits once per resource version. Generate candidates through this index, spaCy linguistic annotations and space-S entity similarity. Preserve original predicate phrases, direction and qualification.

Accepted identity requires an explicit, scoped binding: an authoritative resource identifier used in source context, an explicit source-supported identity assertion, or an owner-authored binding. Mere equal strings, first KB candidate, similarity, acronym expansion and model-produced confidence are insufficient. The engine preserves unresolved distributions. No LLM or PPR call accepts identities; compiler generation cannot alter the binding table. Removing a binding rebuilds its dependent interpretation.

## 3. Cross-document claim comparison

Build comparison buckets using a shared accepted argument identity or unresolved argument-candidate association, together with the same/aligned predicate or an explicit resource lexical-frame match. An explicit source link can connect those argument candidates but does not make every assertion in both documents comparable. Sharing only a generic predicate never generates a corpus-wide Cartesian product. Preserve alternative candidate bindings and the qualifier scope in each bucket; candidates are not accepted identities. Sort/deduplicate pairs by assertion ID and process every pair in the declared bucket scope. Record uncovered predicates/associations as comparison coverage, not proof that no conflict exists. A work/context limit produces explicit incomplete coverage rather than silently dropping pairs.

For each candidate pair, the same configured Raw/Ollama adapter runs the comparison payload defined in `src/yaw/nlp/models.py`. Its input contains both qualified assertions, their verified evidence and relevant declared property/authority semantics. It emits one of `supports`, `potential_conflict`, `explicit_supersession`, `different_scope` or `unknown`, together with assertion IDs and supporting spans. This is scheduled interpretation work, not a failed-extraction recovery call. It cannot introduce source facts, accept identities, author executable rules or choose a winning source.

Admission checks validate referenced records/spans and qualification compatibility. A generated conflict is labelled a model interpretation; only a tested resource constraint may report a formal logical violation. `explicit_supersession` requires evidence of replacement and its applicable authority/scope; recency alone cannot establish it. Without that evidence, preserve both claims and report uncertainty. Distinguish document time, asserted validity time, speaker and decision authority. Compare values only with compatible units, quantities, entity bindings and temporal/conditional scopes.

The April June-deadline / May approved July-deadline / supplier still expecting June case is a required acceptance fixture. It must remain possible to retrieve the decision history and supplier disagreement together. A missing statement under OWL's open-world semantics is not its negation. [OWL primer](https://www.w3.org/TR/owl2-primer/).

## 4. RDF enrichment and validation

The [semantic guide](rdf-owl-shacl.md) owns the one RDFLib/pySHACL sequence, supported constructs and inference scope. It consumes admitted logical projections separately from the graph of attributed source claims. All built-in/imported shapes and admitted rules come from versioned resources. LLM-generated regularities are proposals, never executable rules on ingestion.

Enrichment creates derived records; validation creates findings. Store them separately from original assertions. Each derived triple references its inference scope, complete input assertion set and resource/rule-set versions. This is conservative scope-level derivation provenance, not a fabricated minimal proof or attribution to one rule. A removed premise invalidates and rebuilds its scope; another remaining support can derive the conclusion again.

## 5. Organization and authored knowledge

Run the single organizational procedure in the [hierarchy guide](ontological-roots.md). It produces evidence-linked topic groups and generated descriptions; it cannot turn a similarity cluster into an OWL subclass axiom. Unmet organization targets remain visible.

The synthesis payload in `src/yaw/nlp/models.py` generates entity/topic page blocks from the group's current qualified claims, interpretations, findings and derived records. Use the same Raw/Ollama adapter. Each substantive block cites its supporting assertion or derivation IDs and identifies contested, inferred and unresolved material. Generated connections remain labelled interpretations. A source statement and a model explanation cannot share an undifferentiated evidence status.

Store blocks, evidence/dependency IDs, authoring configuration and input fingerprint in SQLite. Validate schema, referenced support and source freshness before publication; semantic faithfulness additionally requires actual-model evaluation. Failed generation does not produce a template posing as authored synthesis. Reuse exact compatible cached blocks; never re-ingest generated pages as sources.

Markdown is an optional sink of the published blocks, including wikilinks, `index.md` and `log.md`. Export is deterministic for a stored generation and performs no generation, inference, embedding or identity decisions. Removing a page removes only engine-owned output; raw notes are never rewritten.

## 6. Publication, freshness and retraction

Use monotonic source and compiled generation identifiers plus content fingerprints. The SQLite target adapter stages outputs and publishes a completed component scope in one transaction, including rows, dependencies, findings and its completion marker. Its startup reconciliation resumes or discards uncommitted staging; it cannot expose half-published scopes. Before committing, compare the job's expected manifest/generation with the scope's current desired generation inside the serialized publication transaction. A superseded job cannot move the current-generation pointer; discard its staged publication and let the same scheduler process the newer input. Test an edit between computation and commit, not only a crash during row insertion. CocoIndex's separate targets do not supply cross-target atomicity automatically. [Transaction limits](https://cocoindex.io/docs/faq/).

Fresh source evidence is queryable while compilation is pending through the same retrieval function. Its result explicitly says `pending` and identifies completed/current scopes; stale derived scopes are excluded by dependency validation. This is ordinary lifecycle state, not a substitute retriever. A complete result requires all relevant scope manifests to match. Do not label pending or partial extraction as complete knowledge.

On edits, deletion, resource changes or binding corrections, invalidate affected scopes before they can appear current. Retain compatible leaf extraction for unrelated documents. Rebuild deterministic corpus projections and regenerate only changed comparison/topic inputs. Dependencies include whole group membership where absence/new evidence affects the conclusion. Garbage collection removes obsolete engine-owned rows and output pages after publication, retaining support still provided by other sources.

## 7. Zero fallbacks and explicit outcomes

One path includes legitimate empty data and declared lifecycle states. It excludes switching implementations after a failure.

Every component scope follows `pending → running → complete | partial | failed`. A source/resource/input change marks the old scope `stale` and creates a new pending input identity; successful publication alone sets `complete`. `empty` is a successful result payload with no records, not an error state. A compatible cache result follows the same validation/publication path. Changed inputs produce an ordinary memoization miss and recomputation by the same component; corrupt cache data is an explicit error. Explicit retry moves a failed scope to pending without selecting another implementation.

| Condition | Required behavior |
| --- | --- |
| Valid response containing no claims, no candidate or no applicable rule | Record the appropriate empty result; continue the same algorithm |
| SHACL returns a valid non-conforming report | Complete validation and persist the violations as findings; disagreement is data, not an engine failure |
| Individually invalid records in an otherwise parseable envelope | Keep diagnostics and independently valid records with `partial` status; never publish the scope as complete |
| Invalid envelope, unsupported required input, model/network error, missing dependency or wrong vector space | Typed operation failure; no substitute model, parser, heuristic record or zero vector |
| Comparison/inference/validation/synthesis/publication execution error | Mark its scope failed and invalidate dependent current results; no previous generation substituted as current |
| Query requires an embedding service or failed compiled scope | Return the explicit dependency/scope error; no lexical-only or old-scorer recovery |
| Source revision cannot be verified | Return stale/unavailable evidence diagnostics; never cite a different slice |
| Closure/work bound reached | Explicit incomplete/failure result; no complete-generation marker |

No automatic semantic repair, source-text-as-response parsing, recursive extraction, alternative task factory, restricted extractor, alternate compiler, live-lint validator or degraded retrieval mode survives the migration. Explicit retry of a failed job invokes the same implementation and versioned inputs. Compatible cache hits are normal memoization, not recovery from an incompatible result.

`lint_wiki` only reads scoped, versioned compiled findings; refresh uses `compile_wiki`, which schedules the same application. No-findings means no findings in the stated completed scope, never universal consistency. CLI, MCP, evaluation and Markdown presentation consume the same structured records.

Resolve the requested source scope from the source catalog **before retrieval**. An omitted source scope means all active sources; an explicit source-ID selection means exactly that set. Reject unknown IDs. Top-k hits or a free-text query do not narrow the promised completeness scope after the fact. A derived record is eligible only when its whole dependency set is current and contained in the requested source scope. Failed work outside that explicit scope does not invalidate independent results.

Aggregate required in-scope states in this order: `failed` → `partial` → `pending` → `complete`; running/stale work awaiting refresh counts as pending. Return the highest-priority state, involved scope IDs, generations, coverage and diagnostics. A failed operation returns its typed error, not a success-shaped passage payload. Pending/partial results may include independently valid fresh evidence and completed eligible projections, labelled with that state. `complete` with no passages/findings is valid empty data only after all required in-scope work completed. A cache hit does not change these admission rules.

## Proof before Epic 8

Deliver the complete slice in **7.8.5**, after 7.8.3's shared-state proof and 7.8.4's extraction/storage contract. Use two related source notes plus an unrelated third control. Fix source text, explicit identity/scope evidence, one unknown property, its later resource mapping, one admitted AF rule and one violating shape before running. Include enough supported topic members to exercise a real merge, plus empty/single-member organization controls. Keep expected assertions, support IDs and findings independent of the engine output.

Execute these phases through production entry points and actual CocoIndex/SQLite/RDF/SHACL/matrix/FastMCP boundaries:

| Phase | Required observation |
| --- | --- |
| Initial ingest | Qualified claims, comparison, rule consequence, completed validation findings, supported group/page block and structured evidence appear without a manual compile barrier |
| Repeat / `force` | Compatible model outputs are reused; current fingerprints and source rows remain correct; no database/cache wipe |
| Edit related source | Its dependent claim/comparison/derivation/group/block changes; unrelated extraction is reused |
| Remove support and delete source | Obsolete rows, FTS/vectors and pages disappear; independently supported knowledge survives |
| Change/remove resource | Affected interpretations/findings are rebuilt; compatible open extraction survives |
| Interrupt / supersede publication | No half-generation is visible; an older in-flight job cannot publish over a newer manifest |
| Restart / reorder discovery | The same active manifest and semantic state result, using fixed fixture responses or compatible cached generation |
| Request pending, partial or failed scope | Query/lint expose the declared state; no failed scope is hidden by selecting different top-k hits |
| Render twice | Identical stored-generation output and zero model calls; no generated page enters the source corpus |

Source-specific loopback responses prove this integration sequence. The separate actual-model pilot proves its declared semantic sentinel outcomes, including E78-16/19/20; fixed fixtures alone cannot close that gate. Every [Epic 7.8 completion gate](epic7.8-open-ie-ontology-alignment.md#completion-criteria) must pass before Epics 8.x begin.
