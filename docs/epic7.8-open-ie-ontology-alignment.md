# Epic 7.8: One functioning claim-to-knowledge pipeline

**Declared / revised:** 2026-10-12 / 2026-10-13\
**Audience:** Coding agents implementing and verifying the selected path\
**Status:** IN PROGRESS — selected documentation path; implementation and runtime gates remain open\
**Repository:** `C:\Users\Harri.Ahokas\Documents\yaw` (`origin`: `https://github.com/negatrait/yaw.git`)\
**Baseline:** `dev`, `a4142060c4e3f77f51b6a6528a84307b21ae604c`\
**Contract:** [FOUNDATIONAL](../FOUNDATIONAL.md); [ADR-014](adr/ADR-014.md), amending [ADR-013](adr/ADR-013.md)\
**Sequence:** [Roadmap and Epics 8.x](epics-alignment-roadmap.md)

## Objective

Deliver one continuously maintained knowledge compiler and retrieval engine for **100–500 raw documents**, primarily English with occasional other languages. Preserve source evidence, qualifications, ambiguity, disagreement and revisions; compile useful cross-document knowledge before retrieval. Keep spaCy/spacy-llm, CocoIndex, Ollama, one SQLite database, RDFLib/pySHACL, SciPy and FastMCP.

**Every implementation and verification gate in this epic must pass before any Epic 8 implementation starts.** Core compilation, hierarchy, synthesis, export and a real-model pilot cannot be deferred to 8.x. A document revision does not establish a functioning implementation.

Zero fallbacks means one selected implementation for every responsibility. Errors do not select another model, extraction task, parser, compiler, semantic engine or retriever. Valid empty data, an unresolved identity, a pending job and an explicitly scoped view have defined states in this same path; they are not permission for degraded success.

No processing-time SLA has been agreed. Record indexing and update costs against document tokens, claim/graph size and hardware. The selected implementation is a concrete starting design, not a claim that a task count, model name, hierarchy or weight formula is optimal. Future measured changes replace this design explicitly; runtime alternatives are not retained.

## Authority and evidence

FOUNDATIONAL owns product contracts and payload meaning. Focused guides own the selected algorithms. This epic owns work and completion, the roadmap owns sequencing, and TEST_READY owns actual verification evidence. Do not copy executable schemas or configuration values into several guides.

- `review:R1–R10`: [Open IE and ontology review](reviews/open-ie-ontology-review-2026-10-12.md), historical baseline `05a81cb3f645a708e0b3fa3627591d3884d76386`. Its old checkout/environment observations are not current runtime evidence.
- `audit:C-* / H-* / S-* / R-* / D-*`: [alignment audit](alignment-audit-12-10-2026.md), baseline `a414206`, matching this repository's starting HEAD. The owner's staged audit is preserved; its proposals are resolved below, and its runtime claims still require executable verification.
- Owner's attached File Comments and subsequent discussion: retain the existing stack, source-grounded authored wiki, automatic corpus compilation, declarative semantics and open/scoped views; prioritize low-volume RAG quality and require zero competing paths before 8.x.

[ADR-014](adr/ADR-014.md) settles alternatives left by the previous consolidation. Historical ADR bodies and archived specifications are not implementation choices. DSPy migration and generic agent-framework adoption are outside this plan.

## Coding-agent execution contract

**Read in order:** AGENTS → FOUNDATIONAL → this epic → the guide for the package being implemented. Read ADR-014 for a decision conflict. Historical specifications are not implementation input. Do not load every historical ADR or research clipping to discover an alternative.

**Resume protocol:**

1. Confirm repository root, remote, HEAD and both staged/unstaged changes. Read TEST_READY's next action; preserve existing work.
2. Select the first unchecked package and the first unfinished step in its [delivery packet](#delivery-packets). Read only its owner guide and the [test-target map](../TEST_INFRA.md#test-target-map).
3. State the acceptance IDs, production owners and observable result for this work item. Find the existing callers before adding or moving an interface. `NEW`/`PLANNED` paths are deliverables, not imports that already work.
4. Implement through those owners, then run the named focused tests with testmon selection disabled. Record collection counts and results; a missing test, skip or expected failure does not satisfy a required case.
5. Run the package exit checks. Update TEST_READY with exact commands, evidence paths, remaining failures and the next step. Check a package only when every exit condition passes.

**Decision boundary:** Choose local function decomposition, physical columns and initial positive operating limits within the selected contracts; record them in their executable owners and tests. Do not ask the owner to choose routine names or tuning values. Replacing the selected library/algorithm, relaxing evidence semantics or accepting a failed required gate needs an explicit contract amendment with cited evidence before dependent work resumes. A dependency failure blocks that item, not independent authorized work. Never mark it passed or add a fallback.

**Implementer map:** rows describe the target, not symbols already present. `NEW` identifies planned modules; extend existing owners in place everywhere else. Import contracts directly; do not add compatibility re-exports.

| Owner | Required interface / responsibility | Acceptance |
| --- | --- | --- |
| `src/yaw/nlp/models.py` | Sole extraction/comparison/synthesis Pydantic record and result definitions; FOUNDATIONAL owns their meaning | E78-03/04/06/16/19 |
| `src/yaw/nlp/parsing.py` | One `parse_generation` adapter and evidence validator; validate the outer envelope, then independent records and their references | E78-03/04/06/17/19 |
| `src/yaw/nlp/extractor.py`, `components.py` | One public Raw/Ollama pipeline factory with the three declared task payloads; source-preserving linguistic/SpanRuler annotations | E78-01/03/04/06/17/19 |
| `src/yaw/indexer/pipeline.py` | Register extraction and shared corpus component in one application; `sync_vault` and `vault_pipeline_main` invoke it | E78-07/12/15/17 |
| `src/yaw/indexer/compiler.py` **NEW** | `compile_corpus(manifest)` computes alignment/comparison/semantics/organization/synthesis outputs; no publication side writes | E78-08/09/15/16/18/20 |
| `src/yaw/indexer/canonicalize.py` | Candidate generation and evidence-scoped reversible bindings; no first-candidate or generative acceptance | E78-05/10 |
| `src/yaw/storage/schema.py`, `db.py` | Schema/migrations and one `publish_generation` target adapter for rows, dependencies, findings and completion state | E78-07/10/15/18/20 |
| `src/yaw/storage/rdf.py` | One public IRI constructor and semantic-kind-correct projection; every caller imports it | E78-01/04/05/10 |
| `src/yaw/ontology/loader.py`, `shacl.py` | One resource snapshot and one `compile_semantic_scope` implementing the selected enrichment/validation sequence | E78-01/02/08/18 |
| `src/yaw/retrieval/retriever.py` | One `GraphRAGRetriever.retrieve` returning typed `QueryResult`; one PPR scorer and evidence hydration | E78-11/16/17/20 |
| `src/yaw/server/daemon.py` | Thin MCP handlers invoking the application, retriever and cached findings; no parallel business logic | E78-12/13/17/20 |
| `src/yaw/materialize.py` **NEW** | `render_generation` renders persisted blocks; no semantic/model work | E78-12/15/20 |
| `src/yaw/config.py`, `config.toml` | One typed configuration owner for models, context/work limits, scoring and view settings | E78-02/07/11/17 |
| `eval/pilot.py` **NEW in 7.8** | Actual-model pilot using production entry points and independent labels; shared by the later 8.2 runner | E78-14/16/19 |

`CorpusManifest`/compiler output types belong to the new compiler module; `QueryResult` belongs to the retriever. Do not duplicate those types in MCP or evaluation. QueryResult must expose status, source/compiled generations, ranked original passages, qualified claims, typed paths, stored synthesis, related/omitted evidence IDs and coverage. SQL DDL stays exclusively in the storage schema owner. The compiler guide owns stage inputs/dependencies and failure states.

**Configuration cleanup:** retire `taxonomy.llm_ner_if_ner_below`, `taxonomy.base_predicates`, heuristic identity/LLM-adjudication switches and ontology similarity/confidence settings that imply hard merges or truth. Replace them with the selected task, view and compiler settings in the existing configuration owner. Required context/round/triple/time/member limits must be explicit, validated and recorded with the runnable profile in 7.8.3; missing/invalid required settings fail startup. Values are operational tuning, not alternate algorithms. No dormant legacy flags survive 7.8.6.

**Removal starting points:** inspect `_get_steered_pipeline`, `classify_freeform_llm`, `_add_rel_pipe`, `_add_optional_pipes`, source-text response parsing, `_lookup_kb`'s first-candidate acceptance, `precompute_rdfs_closure`, dynamic whitelist shape generation, duplicate closures in `ontology/owl.py`, private factory patches in tests, embedding-outage branches in the retriever, and `sync_vault(force=True)` / cache-mismatch database deletion. Replace needed behavior through the selected owners and delete unused implementations/call sites. A renamed fallback does not satisfy the removal gate.

**Verification discipline:** for each package, implement the observable E78 cases at the real subsystem boundary, including its failure cases; run focused checks, then required regression/lint/type gates. Record exact commands/results and update only the completed package. Check code/call sites/configuration for retired paths as well as testing the selected path. Documentation gates can close independently; runtime gates cannot.

## Selected path and ownership

`source revisions → linguistic annotations + Raw extraction → qualified claims → shared corpus compilation → SQLite publication → structured query / cached lint / Markdown export`

| Responsibility | Single selected owner / contract | Retire from production |
| --- | --- | --- |
| Source and derived orchestration | One CocoIndex application with a shared corpus component; [compiler guide](knowledge-compilation.md) | Post-sync fallback compiler, manual-only pipeline, hidden mutable inputs and unmanaged side writes |
| Extraction and generation | One `spacy.Raw.v1`/Ollama adapter, task-specific Pydantic contracts; [spaCy guide](how-to-handle-spacy-llm.md) | REL/five-task alternatives, direct parallel model clients, source-text parsing, recursive or repair extraction |
| Generation memoization | CocoIndex with complete ordered input/configuration identity | spaCy BatchCache and competing persistent LLM caches |
| Source meaning and identity | FOUNDATION's qualified payload and explicit scoped binding policy | Three-string-only facts, first-candidate merges, generative identity adjudication and lossy predicate normalization |
| Semantic compilation | One resource loader and RDFLib/pySHACL sequence; [semantic guide](rdf-owl-shacl.md) | Python domain tables, SQL semantic closure, duplicate inline shapes and untracked rule output |
| Organization and authored wiki | Compiler stages with evidence/dependency records; [hierarchy guide](ontological-roots.md) | Forced class axioms, raw/wiki feedback loops and export-time generation |
| Retrieval | One structured result builder and scorer; [PPR guide](power-pagerank.md) | Embedding-outage lexical recovery, discarded dense candidates, legacy scorer branches and presentation-text evaluation |
| Interfaces | Ingest/compile schedule the application; query consumes its projections; lint reads its findings | Stub success/counts and live-lint execution path |

The shared component must be proven with the pinned CocoIndex version. If it cannot satisfy the dependency/publication contract, implementation stops at that gate until the design is explicitly amended; no automatic post-sync alternative is authorized. The library's [cross-file example](https://cocoindex.io/docs/examples/meeting-notes-to-knowledge-graph/) supports investigating this integration but does not certify YAW's implementation.

## Owner decisions consolidated

### D-1 / D-4 — Extraction and automatic corpus compilation

Keep `spacy.Raw.v1` with one validated parser, not Raw's unparsed response as the storage contract. Use the selected spaCy linguistic and resource-candidate components to assist extraction without gating unfamiliar or non-English claims. The schema in `src/yaw/nlp/models.py` is the executable owner for mentions, subject/object arguments, predicates, supported definitions, qualifiers and evidence.

One extraction request processes each changed bounded unit. Comparison and synthesis are separately scheduled compilation responsibilities using the same adapter and model; they are not another extractor or failed-output recovery. No LLM accepts an identity binding. Refer to the [compiler guide](knowledge-compilation.md) for stage order, comparison scope, explicit supersession, group dependencies and publication.

Cache unchanged extraction and generated outputs; recompute deterministic corpus projections from the active manifest. Prove edits, deletion, ontology changes and restart before optimizing this for larger workloads. Fresh evidence has explicit pending/complete status; failed compilation never masquerades as current completed knowledge.

### D-1 continuation — Supported organization

The compiler produces organizational groups and evidence-linked labels/definitions using the single procedure in the [hierarchy guide](ontological-roots.md). The goal of fewer than ten umbrella groups is represented by a target of nine, not a requirement to merge unrelated meanings. Unmet targets are reported. Organizational membership, SKOS hierarchy and OWL class subsumption remain different relationships.

Ward is the selected initial grouping algorithm, not a proven retrieval improvement or balanced-tree guarantee. Its geometry, ordering, limits and rejection behavior have one owner in that guide. Epic 8 measures benefit through offline ablation; core capability must already function in 7.8.

### D-2 — Declared enrichment, validation and conflict interpretation

Use the [semantic guide's](rdf-owl-shacl.md) one pySHACL inference/rule/validation sequence. Built-in and imported RDF resources own the supported axioms, lexical frames, shapes and admitted SHACL-AF rules. No admitted rules means no SHACL-AF-derived facts, not permission to invent rules with the LLM.

Keep rule-derived facts, scoped validation findings and model-proposed claim comparisons distinct. A newer account is not automatically true; a SHACL pass does not establish factual truth. Rule-derived records carry conservative input-scope and rule-set provenance; no fabricated per-rule proof is allowed. Changes retract and rebuild affected derivations while preserving independent support.

### D-3 — Authored wiki in SQLite, deterministic Markdown

Maintain source-supported entity/topic blocks, cross-document explanations, contested claims and links to findings during compilation. Store their evidence IDs, authoring identity and dependency fingerprints in SQLite. This generated prose is derived content; source chunks remain authoritative on disk.

Export renders the stored generation into Markdown, wikilinks, `index.md` and `log.md` with zero model calls. It is an optional output of the same compiler. `query_wiki` returns structured context/evidence, including material competing claims; `lint_wiki` reads scoped findings. Refresh uses `compile_wiki`, never a separate live-lint path.

### D-5 — One extractor, explicit output views

`open_ie=true` selects an open projection; `open_ie=false` selects the chosen profile's admitted vocabulary. Both operate on the same original open extraction. The [four-view contract](epic7-pluggable-ontology-integration-spec.md#four-view-configurations) is authoritative. There is no restricted-extraction mode or hidden base vocabulary.

Use semantic-kind-correct mappings, retain external IRIs and preserve unresolved terms in the open projection. The RDF constructor in FOUNDATION owns application IRIs. Namespace/label equality is not identity, and SKOS mapping is not a universal relation for individuals or properties.

### D-6 — Quality evidence before 8, full comparison in 8.2

Epic 7.8 includes independent actual-model extraction/comparison/synthesis checks and the complete structured passage API. Epic 8.2 extends that same interface into corpus-specific held-out evaluation and the requested MoG/HippoRAG/official benchmark adapters. Its [roadmap specification](epics-alignment-roadmap.md#epic-82--held-out-quality-and-cost-evaluation) owns datasets, scoring and comparison protocol.

Measure support and counterevidence retrieval, false identity merges, qualification/direction loss, synthesis faithfulness, revision handling, multilingual coverage and deletion/retraction correctness. Recall/accuracy cannot be replaced by SHACL pass rate or successful parsing. Compare against a lexical-plus-vector baseline under equal corpus/context/reader conditions; no held-out question or answer may steer compilation.

## Work packages and restart checklist

Execute in order. Package 7.8.3 proves the dependency/publication integration; 7.8.4 supplies qualified extraction; 7.8.5 completes the source-to-query slice. This avoids requiring the whole compiler before its dependencies exist. These are delivery packages for one runtime, not separately deployed stages. Remove an obsolete path when its replacement is integrated; 7.8.6 verifies the removal, rather than licensing temporary fallback dispatch.

- [x] **7.8.0 Persist the epic and evidence baselines.** Retain review/audit distinctions and owner decisions.
- [x] **7.8.1 Reconcile authority and the one-path decision.** FOUNDATION, ADR-014/013, ledger and active summaries agree; no live alternatives. Documentation gate only. `review:R1–R8,R10`; `audit:D-1–D-5,C-11`.
- [x] **7.8.2 Reconcile guides, scenarios and Epics 8.x.** Compiler/NLP/semantic/hierarchy/PPR/view guides, AGENTS, README, roadmap and test documents have valid links and compatible contracts. Documentation gate only. `review:R1–R10`; `audit:D-6,§6–§8`.
- [ ] **7.8.3 Establish the runtime and shared-state contract.** Verify the existing locked environment, safe platform probes, shared CocoIndex dependency lifecycle, transactional publisher and selected pySHACL entry points. This integration proof is not the complete semantic slice; see the packet below. No fallback if integration fails. `review:R1,R4,R7,R10`; `audit:C-1,C-2,C-7–C-9,S-12,R-2`.
- [ ] **7.8.4 Complete extraction, storage and cache migration.** Typed qualified claims and definitions, exact contextual occurrences, valid empty/partial/error outcomes, managed FTS and typed vector lifecycle. Delete competing extractors/parsers/caches, private API mutation and source-text response parsing. `review:R2,R5,R6`; `audit:C-3–C-6,C-9,C-10,H-1–H-10,S-1,S-2,S-4,S-6–S-11`.
- [ ] **7.8.5 Complete corpus knowledge, retrieval and the vertical slice.** Reversible bindings, comparison/status semantics, one inference engine, retractable rule output, organization and authored synthesis; one PPR policy and structured result builder with counterevidence. Prove automatic source-to-query/lint/export operation through real subsystem boundaries. Finish retained 7.7 representation work, fact-seed IDs, distinct alias joins, wikilink evidence and retained dense candidates. `review:R3,R4,R8`; `audit:C-2,C-6,C-7,C-11,H-6,H-11,H-12,S-3,S-5,S-12,R-1–R-8`.
- [ ] **7.8.6 Migrate and remove every obsolete production path.** Re-extract irrecoverably normalized source data, review heuristic bindings, migrate all identifiers and rebuild projections. Delete old call sites/feature switches instead of keeping compatibility execution branches. Verify native Windows/Unix startup, locks, safe liveness and failure behavior. `review:R1,R3,R5,R6,R10`; `audit:C-1,C-2,S-10,§6.2`.
- [ ] **7.8.7 Prove completion and authorize the 8.x handoff.** Run every E78 scenario, failure injection, real integration, independently labelled actual-model pilot and required regression/Ruff/type checks. Record commands/results/configuration/skips in TEST_READY and the remaining product limitations. No unrun required gate is a pass. `review:R9,R10`; `audit:§7,§8,D-6`.

## Delivery packets

The [test-target map](../TEST_INFRA.md#test-target-map) owns filenames and E78-to-test mapping. Each packet uses that map; do not create a second test suite per interface.

### 7.8.3 — Runtime and shared-state contract

**Enter:** Documentation contracts are reconciled; project/lock files are present. Runtime conformance is unverified.

1. Follow [TEST_INFRA entry commands](../TEST_INFRA.md#entry-commands). Record Python, locked package versions, model digests and resource hashes. Repair the Windows liveness probe before executing lifecycle tests. An import or install failure is recorded as such; do not upgrade dependencies silently to bypass the lock.
2. Implement the host-owned manifest/generation records and `publish_generation` through the listed compiler/schema/storage owners. Register the shared dependency pattern with the pinned CocoIndex API. Use a test-only two-source fixture with explicit expected rows to prove shared-input collection, input ordering, change/deletion propagation and reuse after restart. Exercise the production publisher and real SQLite/CocoIndex. This fixture proves the orchestration API, not extraction or authored-wiki quality; no fixture compiler is registered in production.
3. Interrupt publication and supersede a pending manifest. Prove rollback/no partial visibility and rejection of the superseded publication. An app/cache mismatch must produce a reconciliation diagnostic, never an automatic database/cache wipe.
4. Exercise `compile_semantic_scope` with a tiny declared graph/rule/shape fixture through the pinned pySHACL sequence. Assert one expected new triple, a no-rule control, a valid non-conforming report and bounded termination.

**Exit:** Dependency and native probe results plus the shared-state and semantic-adapter tests pass. Record the runnable configuration's required context/round/triple/time/member limits. The semantic compiler, full E78 matrix and actual-model pilot remain open; do not claim the end-to-end slice here.

### 7.8.4 — Qualified extraction and storage

**Enter:** 7.8.3 passes.

1. Define the payload records in `nlp/models.py`; implement the [single parsing/grounding sequence](how-to-handle-spacy-llm.md#typed-claims-and-grounding) in `nlp/parsing.py`. Add positive, empty, partially invalid and malformed-envelope cases before replacing callers.
2. Route extraction through the one Raw/Ollama factory, exact source/context coordinates and CocoIndex memoization. Replace REL/optional LLM tasks, live-vocabulary inputs, source-text response parsing and BatchCache at their call sites; remove the unused branches in the same change.
3. Extend the storage schema/migration and managed target outputs for source revisions, mentions, qualified claims, diagnostics, FTS and space-specific vectors. The new extraction format is the only active writer/reader format; legacy rows are migration input.
4. Exercise source edits/deletion, reordered input, cache hits/misses, Unicode/CRLF, standalone/nested mentions and resource/view changes. Check stored and retrieved records rather than only the returned model JSON.

**Exit:** The assigned extraction/cache/storage cases pass, raw source bytes are unchanged, and no old extraction/parser/cache branch is callable. Expected failing compiler cases remain recorded for 7.8.5; no full-epic pass claim.

### 7.8.5 — Corpus compiler and complete slice

**Enter:** 7.8.4 passes.

1. Implement the compiler stages in the guide's order: explicit binding/alignment, comparison, scoped enrichment/validation, organization and stored synthesis. Each stage consumes the manifest and emits typed outputs/dependencies for the shared publisher.
2. Integrate the structured retriever, thin MCP/CLI handlers, cached lint and deterministic Markdown renderer. Route all callers to these owners; delete replaced scorer/closure/live-lint paths.
3. Run [the vertical-slice fixture](knowledge-compilation.md#proof-before-epic-8): two related sources plus an unrelated control, real subsystem boundaries and source-specific loopback model responses. Exercise the complete mutation/interruption sequence and compare source IDs, generations, support and outputs across consumers.
4. Deliver `eval/pilot.py` and its versioned pilot manifest under [TEST_INFRA](../TEST_INFRA.md#actual-model-pilot). Run the same path against the configured actual models, outside pytest's autouse loopback fixture. Record semantic failures separately from fixture results and correct them before closing the packet.

**Exit:** Automatic ingest → compiler → query/lint/export works without a manual compile barrier. Assigned E78 cases and the declared actual-model sentinel outcomes pass; support removal, partial/failure states and counterevidence agree across consumers. This is the first complete vertical-slice gate.

### 7.8.6 — Migration and removal

**Enter:** 7.8.5 passes.

1. Apply the sole migration path to a temporary copy of a legacy database and its authoritative source fixture. Verify transactional rollback, repeat application, identifier/reference mappings and explicit re-extraction for lost evidence.
2. Trace each removal starting point to its callers/configuration/tests. Delete dead implementations and flags; rewrite tests that require retired behavior to assert the current contract, preserving equivalent regression coverage. Do not delete a failing test merely to obtain a pass.
3. Prove supported native Windows/Unix lifecycle behavior and stale-generation/output cleanup using isolated vaults. Preserve user raw notes and independent support.

**Exit:** Migration and native platform cases pass, and the removal record lists old symbol/flag, replacement owner, remaining call-site search result and test evidence. No old-format runtime reader, automatic destructive rebuild or fallback remains.

### 7.8.7 — Evidence and handoff

**Enter:** 7.8.3–7.8.6 pass.

1. Run the [full verification commands](../TEST_INFRA.md#completion-commands); include every required case and native platform. Do not use testmon selection as the final regression gate.
2. Reconcile all 20 E78 obligations to collected test IDs and the frozen actual-model report. Required missing/skipped/xfail cases leave this packet open, even if pytest exits zero.
3. Fill the [run record](../TEST_INFRA.md#run-record) in TEST_READY, including worktree/configuration/model/resource identity, actual results and limitations. Check this packet only after the records agree; the roadmap may then advance to 8.1.

**Exit:** All 7.8 completion criteria below have evidence. No separate approval is needed for routine implementation choices already inside the contract; release publication remains governed by 8.3.

## Completion criteria

All items are mandatory. Their observable cases are [E78-01–20](USER_STORIES.md); exact evidence belongs in [TEST_READY](../TEST_READY.md).

1. An unfamiliar relation is extracted, grounded, aligned after resource loading, enriched, validated, summarized and retrieved without editing Python domain tables or SQL whitelists.
2. Independent mentions, definitions, literals, attribution, negation, modality, time, conditions, direction and exact source occurrences survive the full path. Unsupported interpretation is explicit.
3. The actual source/parser/cache pipeline distinguishes valid empty, partial and failed results. Ordered input, source context, model/schema/prompt/resource versions and group membership drive the appropriate invalidation.
4. One shared CocoIndex compiler automatically updates cross-document knowledge. Edits/deletions/resource changes, restart and reordered ingestion converge; unrelated compatible extraction is reused.
5. Identity remains evidence-scoped and reversible. No normalized-label, first-candidate, generative or PPR hard merge survives. Rejected merges do not create `owl:differentFrom`.
6. Claim comparison distinguishes support, potential conflict, explicit revision, different scope and unknown; recency alone never chooses the truth. Retrieval includes material counterevidence and reports coverage/truncation.
7. One versioned semantic engine performs the supported entailment/AF/SHACL sequence. Rule dependencies retract correctly; unsupported constructs and incomplete closure cannot produce complete-generation claims.
8. Organizational groups and authored wiki blocks are persisted with support/status/dependencies. The target root count cannot force false axioms. Deterministic export of a stored generation makes zero model calls.
9. The same structured retrieval function serves MCP, CLI and evaluation, preserves eligible dense/lexical candidates and compatible vector spaces, and uses the single configured conductance/scoring policy.
10. Required-component failures are visible errors. Failure-injection tests prove no second parser/model/extractor/compiler/validator/scorer is invoked and no stale generation is substituted as current.
11. All old production paths and test-only private patches are removed. One executable schema, configuration, semantic resource loader, identifier constructor and result builder serve their consumers directly.
12. Real subsystem tests and actual-model quality results are separate. The live-model pilot covers extraction, comparison and synthesis, including non-English examples, and reports both missed evidence and incorrect assertions.
13. Native platform and regression/lint/type checks pass in a complete locked environment. Migrations are transactional/idempotent, and current runtime/readiness claims cite their evidence.
14. Documentation has one authority per contract and no active conflicting requirement. Every checklist gate is complete before 8.1 begins; 8.2 and 8.3 cannot bypass this boundary.

A failed quality or capability gate requires correcting this implementation or explicitly replacing its decision and rerunning the gates. It never authorizes a runtime fallback. Full held-out comparative evaluation remains Epic 8.2; no benchmark advantage is claimed by closing this document revision.

## Verified state and next action

Static inspection in the correct repository on 2026-10-13 reconfirmed a three-node `ExtractedTriple`, first-candidate KB lookup, coexisting Raw/REL setup, source-text response parsing, a validation-focused `compile_wiki` returning zero clustering counts, and an `ingest_raw` stub. These are implementation gaps; see the [audit](alignment-audit-12-10-2026.md) and [readiness record](../TEST_READY.md).

This checkout contains `pyproject.toml`, `uv.lock` and the source entry points. Earlier missing-file and Git-object failures belonged to a different local copy and are not blockers recorded against this repository. The documentation transfer preserves the staged audit and changes no runtime code. Historical test totals do not establish current runtime health.

**Next implementation action:** 7.8.3 step 1. Follow its delivery packet and TEST_INFRA entry commands, then prove shared state/publication. The complete semantic slice is delivered in 7.8.5. All 8.x implementation remains blocked until 7.8.7.
