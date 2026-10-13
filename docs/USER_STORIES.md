# YAW acceptance scenarios

**Status:** Reconciled target scenarios; execution results belong in [TEST_READY](../TEST_READY.md).\
**Contract:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-014](adr/ADR-014.md).\
**Delivery:** [Epic 7.8](epic7.8-open-ie-ontology-alignment.md). Scenario IDs below are acceptance obligations, not claims that tests already exist.

The [test-target map](../TEST_INFRA.md#test-target-map) assigns these obligations to existing/planned test modules. Each implemented test identifies its E78 case; every required clause and control needs evidence. One passing example or an existing filename does not close a whole scenario.

## Open extraction and declarative semantics

### E78-01 — Unfamiliar property without code changes

Given a note expressing a relation absent from the selected ontology, ingestion retains its original predicate, arguments, qualifiers and evidence and exposes it in the open view.

When a resource declaring the applicable property, lexical frame, inverse relationship and relevant SHACL shape is loaded, generic consumers align/project/validate/retrieve it without edits to Python predicate tables or SQL whitelists. Removing or changing the resource refreshes interpretation while preserving the original assertion.

### E78-02 — View scope and empty profiles

Given the same source revision and extraction configuration, switching among built-in/external and open/scoped views preserves raw extraction. A scoped view derives property membership from selected resources and accepted alignments. An empty optional inventory does not restore a hidden base vocabulary. Namespace-local-name collisions do not equate distinct properties.

Resource updates may require rebuilding dependent projections; tests do not assert a universal constant-time update.

### E78-03 — Predicate direction and qualification

Given “A consists of B” and “A is part of B”, declared lexical frames produce the correct orientation or an explicit unresolved alignment. Neither original assertion is rewritten.

Given positive, negated, modal, attributed and conditional versions of a claim, stored assertions, fact rendering and retrieved context retain those differences. The engine does not project a modal/negative claim as an unconditional positive fact. Subject/object/predicate definitions distinguish source-supported text, generated explanations and unknown values; lack of a definition cannot cause fabricated prose or loss of the claim.

## Grounding and interpretation

### E78-04 — Occurrences and verified evidence

Given repeated identical strings, overlapping/nested mentions, Unicode, CRLF/LF and identical sections in different files, each source occurrence remains distinct and its spans reproduce the cited source text.

When the source changes after indexing, hydration reports stale evidence and schedules refresh. It does not cite the changed slice as verified indexed content. Source revision metadata does not require duplicating whole chunk prose in SQLite.

### E78-05 — Identity and semantic kinds

Given homonyms, ambiguous acronyms and “cell” versus “cancer cell”, lexical similarity or a single retrieved candidate is insufficient for a hard merge. Record candidates and accepted identities separately with evidence and scope.

Given individual/class/concept/property/literal endpoints, RDF projection uses the appropriate relationship and preserves datatype/language where supported. Namespace alone does not select SKOS mapping. Removing an accepted binding refreshes dependent projections. A non-merge does not generate differentFrom.

### E78-06 — Complete extraction outcome

A standalone mention remains recorded even without a binary relation. Nested mentions are retained independently.

Valid no-extractions, schema failure, unmatched evidence, network failure and partial/unsupported interpretation have distinct outcomes. A valid outer envelope with two records, one with invalid grounding, preserves only the independently valid record and reports partial; zero survivors after rejection is still partial, not successful empty. A malformed outer envelope fails without trying another parser. Local references within the bounded source context require supported antecedents and coordinates. Ambiguous pronouns, arbitrary cross-document coreference and unsupported n-ary interpretation are not silently converted into confident binary claims.

## Incremental correctness

### E78-07 — Ordered, configuration-aware caches

Reversing “Alice supports Bob” to “Bob supports Alice” cannot return the old Doc/annotations. Changing rendered examples, task/schema, model digest or generation settings invalidates incompatible extraction entries. Restart preserves valid entries and rejects mismatches. Different task/component caches do not collide.

### E78-08 — Supported inference and scoped validation

Declared subclass/subproperty/type, inverse/symmetric/transitive, admitted identity and disjointness capabilities have real positive/negative examples through the selected pySHACL sequence. Cycles terminate; unsupported constructs are inactive with diagnostics, and incomplete closure cannot publish a complete scope.

Opposing attributed/time-scoped claims are evaluated under a declared comparison policy. Structural conformance is not labelled factual truth. Shape/resource/data changes mark cached findings stale and refresh the correct scope.

### E78-09 — Derived hierarchy

A generated group label has support, generation metadata and organizational status. It is not automatically a class-subsumption axiom. Original source records remain unchanged. Re-running unchanged inputs reuses compatible generation. Zero/one/already-within-target inputs retain their supported roots without a linkage/merge call; a larger supported fixture proves an accepted merge. A target of nine umbrella groups does not force unrelated meanings together; an unmet target and rejected candidates are reported. Changed members invalidate both grouping and dependent summaries.

### E78-10 — Migration

Legacy rows with overwritten source predicates are marked for re-extraction from available authoritative notes. Old heuristic aliases are reviewed/rebuilt under the new evidence policy. Migration preserves audit/reference mappings, is transactional/idempotent, and reports unavailable source evidence.

## Retrieval, interface and portability

### E78-11 — Retrieval invariants

Real sparse matrices preserve finite nonnegative normalized mass, including dangling nodes and disconnected components. Empty channels are omitted/renormalized; all-empty input returns no evidence. A query-mentioned node can contribute through aliases even when absent from top facts. Wrong vector spaces/dimensions fail explicitly.

Returned paths distinguish candidates, organization and entailment; source evidence is revision-verified. The union of eligible dense/lexical/graph passage candidates survives to the one scorer. Embedding/index failure, wrong dimensions and non-convergent PPR return explicit errors; no offline scorer or alternate solver runs. Verify the shared logarithmic conductance policy, root/cycle handling and stable tie order. FTS normalization has an independent three-hit expected vector (`6/11, 3/11, 2/11`), and an all-empty/zero-active-weight case does not enable a disabled channel.

### E78-12 — FastMCP and wiki lifecycle

A real client exercises ingest, compile, query and lint through the same application and structured results. Queries coexist with incremental indexing under a declared workload. Compilation/materialization never modifies existing raw notes or overwrites source assertions. Generated wiki pages render persisted source-supported blocks; rendering an unchanged stored generation makes zero model calls. Lint reads completed findings and reports pending/failed scopes; it cannot run a live alternate validator.

### E78-13 — Platform behavior

Windows and Unix use identical semantic resources/configuration. Tests exercise native path handling, locks, safe non-terminating process-status probes, startup and shutdown. Do not execute a known unsafe Windows liveness probe as a test shortcut.

### E78-14 — Independent evaluation

Before closing 7.8, run the actual configured model on independent extraction/comparison/synthesis labels covering E78-01–06 and E78-16/19/20. Report missed evidence and incorrect output, false merges, direction/qualification/grounding errors, unsupported generated statements, comparison errors and cost separately. Expected outputs must include positive cases, so returning only empty/unknown results cannot satisfy the pilot. Every labelled sentinel case has a declared expected outcome before the run; unmet required outcomes keep the gate open.

Epic 8.2 extends this into disjoint calibration/held-out retrieval comparisons under the [roadmap protocol](epics-alignment-roadmap.md#epic-82--held-out-quality-and-cost-evaluation). Record complete model/resource/configuration versions. Loopback fixture success is not a model-accuracy result.

## Complete compiler and zero-fallback gates

### E78-15 — Automatic cross-document lifecycle

Given two documents supporting a shared topic and an unrelated third document, ingestion through the one CocoIndex application automatically extracts, compares, enriches, validates, organizes and stores supported wiki blocks. No manual compile/lint sequence is needed. An unfamiliar property works through resource changes without Python domain edits.

Edit one document, remove its support, then delete it. Dependent comparisons, derivations, groups, summaries, findings, FTS/vector rows and exported pages update correctly; independently supported knowledge survives. Unrelated extraction is reused. Restart after interrupted publication and reorder input discovery: the active manifest and derived semantic state converge. A newer desired manifest arriving before commit prevents the old job from becoming current. `force=True` re-evaluates the same scope without deleting database/cache state or invalidating compatible generation arbitrarily. Tests compare stored/versioned outputs rather than assuming uncached LLM generations are byte-identical.

### E78-16 — Contradiction, revision and scope

Use three independent source notes: April predicts June completion; May explicitly records the authorized replacement date of July; a supplier still expects June. Preserve all claims, attribution, validity/document times and evidence. Record supported supersession for the decision and the supplier's differing position; do not turn newest-document order into a truth rule.

Add controls with different entities, years, units and conditions, and a later note that merely repeats an opinion. These must not be labelled the same formal contradiction or authorized supersession. Unresolved identity/scope remains unknown. A retrieval answer about the deadline includes material competing evidence or explicit omitted IDs/coverage, rather than presenting one side as settled.

### E78-17 — No alternate implementation after failure

Inject a missing model, failed embedding service, malformed response, invalid span, wrong vector dimension, corrupt cache, incomplete closure and publication failure at their real boundaries. Verify explicit failure/partial states as defined by the compiler contract, with no success-shaped empty result or stale completed generation. Changed cache inputs are a separate control: they recompute through the same component rather than reuse incompatible output.

Record invoked task/component IDs. Assert that no REL, LLM NER/EntityLinker, alternate parser/model, direct secondary client, post-sync compiler, live-lint validator, lexical-only retriever or alternate PPR solver executes. A compatible cache hit and valid empty response remain distinguishable controls. An explicit retry uses the same selected implementation.

### E78-18 — AF enrichment, provenance and retraction

Supply an admitted resource rule that derives an expected fact and a shape that checks its consequence. Also require an AF consequence to feed a later entailment round. Prove the single fixed-point sequence terminates and validation consumes the completed result.

No-rule and invalid/unsupported-rule controls cannot silently derive that fact. Removed premises invalidate the scope; another surviving support can derive it again. Derived triples cite the actual complete input scope and rule-set version, never a guessed individual rule or minimal proof. Limit exhaustion produces no complete-generation marker. A valid non-conforming SHACL report instead completes validation with findings; both conflicting source accounts remain queryable.

### E78-19 — English and occasional other languages

Run original-text, non-English and mixed-language cases through the same extraction path, including language-tagged aliases/literals and at least one cross-language reference with explicit identity evidence. Verify Unicode coordinates and qualifications. English linguistic annotations cannot reject unfamiliar-language claims or silently translate their evidence.

Independent labels must expose unsupported or mistaken language interpretation. Model availability failures do not select another language/model pipeline. Record per-language coverage and errors; passing a few examples is not universal multilingual support.

### E78-20 — Stored synthesis, publication and consumer agreement

Generated page blocks reference current source claims/derivations and distinguish inferred, contested and uncertain material. Invalid evidence IDs fail structural admission; independently labelled model checks detect unsupported paraphrases that structural validation alone cannot find. New conflicting group membership invalidates the old summary even when prior claims remain unchanged.

MCP, CLI, evaluation and Markdown consume the same structured records/generations. During pending compilation, fresh evidence and explicitly completed scopes remain identifiable; stale dependencies are excluded. A failed required scope returns its error, never a substitute current result. An omitted source scope includes the active corpus; a narrowed explicit source-ID scope is resolved before candidate retrieval and excludes unrelated failures. Top-k selection cannot hide a failure in the promised scope. Aggregate status follows failed, partial, pending, complete precedence; findings from a completed non-conforming SHACL report do not cause failed status. Rendering twice from the same generation is identical and makes zero model calls. Generated wiki pages never enter the raw source corpus.
