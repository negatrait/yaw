# Open IE and ontology architecture review

Date: 2026-10-12\
Reviewed branch: **dev**, commit **05a81cb3f645a708e0b3fa3627591d3884d76386**\
Status: **Historical review; not current implementation instructions or runtime verification**

This review was made against a different local copy at the pinned commit above. Its findings motivated the accepted [Epic 7.8](../epic7.8-open-ie-ontology-alignment.md), [ADR-013](../adr/ADR-013.md) and [ADR-014](../adr/ADR-014.md). Use [TEST_READY](../../TEST_READY.md) for the correct repository and current verification status. Older environment observations and proposed alternatives below are historical evidence.

## Verdict and intended architecture

**The suite is not yet a consistent, implementation-ready specification. The central problem is split semantic authority:** ontology assets define relations, but Python dictionaries, configuration defaults, embedded validation rules, and SQL whitelists independently define overlapping meanings and membership rules.

YAW's purpose is to turn evolving source notes into a continuously maintained, evidence-backed knowledge graph and retrieve through its semantic structure. CocoIndex manages incremental work; spaCy and spacy-llm interpret language; ontology resources supply domain meaning; SQLite holds source assertions and derived projections; SciPy performs graph retrieval; FastMCP exposes those capabilities. Optional wiki pages are a materialized view of that knowledge.

The intended ownership is:

| Layer | Responsibility |
| --- | --- |
| spaCy | Tokenization, linguistic analysis, mention spans, lexical features, candidate generation and contextual disambiguation where configured and evaluated. |
| spacy-llm | Model orchestration and structured extraction through public task/model/parser interfaces. The output remains an extraction requiring grounding and interpretation. |
| RDF | Identifiers, graph statements, literals and dataset/provenance representation. |
| RDFS / OWL | Declared classes, properties and logical relationships, interpreted through an explicitly supported reasoning profile. |
| SKOS | Vocabulary labels, concept schemes and thesaurus navigation, with concept mappings kept distinct from individual identity. |
| SHACL | Executable constraints against a specified data graph and inference regime. |
| GraphRAG | Use those structures and source-grounded assertions to retrieve supporting passages. |

These roles follow the [spaCy integration documentation](https://spacy.io/usage/large-language-models), [RDF Concepts](https://www.w3.org/TR/rdf11-concepts/), [OWL Primer](https://www.w3.org/TR/owl2-primer/), [SKOS Reference](https://www.w3.org/TR/skos-reference/) and [SHACL specification](https://www.w3.org/TR/shacl/). Their composition into the architecture above is this review's recommendation, consistent with the user's clarified intent.

**Open discovery must remain open.** An unfamiliar entity or relation can be extracted and retained before a suitable ontology alignment is known. Loading an ontology should enrich interpretation and validation; changing a retrieval scope should not silently destroy source assertions.

Generic Python implementations of W3C rules, language processing, storage and graph algorithms are appropriate. Domain relation inventories, lexical mappings, inverse orientations and domain constraints should be declared in the semantic resources and compiled into runtime indexes. Generated indexes are compatible with a single source of truth; separately maintained semantic tables are not.

## Findings

Priority P1 means a correctness or central architecture issue to resolve before accepting the specification; P2 means an important specification, evaluation or maintenance correction. These are review priorities, not measured production incident severities.

### R1 — P1: Active semantic rules compete with the ontology

This is confirmed in executable paths, not only old examples.

| Location at the reviewed commit | Confirmed behavior |
| --- | --- |
| [src/yaw/config.py:70](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/config.py#L70) and [src/yaw/config.py:206](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/config.py#L206) | Python declares fourteen base predicates. An explicitly empty list falls back to that tuple. This contradicts the freeform-empty-list comment in [config.toml:33](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/config.toml#L33). |
| [src/yaw/nlp/components.py:130](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/components.py#L130), [src/yaw/nlp/components.py:301](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/components.py#L301), [src/yaw/nlp/components.py:429](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/components.py#L429) | Separate handwritten predicate aliases, polarity exclusions and synonym mappings. The synonym table is checked before dynamic ontology properties at [src/yaw/nlp/components.py:488](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/components.py#L488). |
| [src/yaw/nlp/parsing.py:250](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/parsing.py#L250) | Raw extraction replaces the surface predicate with the older normalizer's result before returning the triple. |
| [src/yaw/indexer/canonicalize.py:754](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/indexer/canonicalize.py#L754) | Later canonicalization uses the base inventory and another, polarity-guarded normalizer; successful base matches can become hard aliases. |
| [src/yaw/storage/schema.py:407](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/storage/schema.py#L407) | Closed-view filtering accepts a literal SQL whitelist in addition to alias matches. It is not determined solely by the active ontology's properties. |
| [src/yaw/retrieval/retriever.py:334](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/retrieval/retriever.py#L334) and [src/yaw/retrieval/retriever.py:461](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/retrieval/retriever.py#L461) | Selected relation names receive special reverse edges and conductance treatment. Domain property behavior is partly encoded as string comparisons. |
| [src/yaw/ontology/shacl.py:391](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/ontology/shacl.py#L391) | Fixed SHACL text embedded in Python enumerates particular contradictory predicate pairs. This is SHACL, but duplicates pairs already declared through `owl:propertyDisjointWith` in [src/yaw/ontology.ttl:146](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/ontology.ttl#L146). |

The NLP-facing ontology projection is also too narrow to replace these tables: [src/yaw/ontology/loader.py:270](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/ontology/loader.py#L270) extracts a single `rdfs:label`, one property parent and two characteristics; [src/yaw/ontology/registry.py:29](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/ontology/registry.py#L29) exposes no inverse orientation or lexical-frame information. This finding concerns that projection, not a claim that every RDF graph in the application discards all other axioms.

The specifications themselves perpetuate this split: [FOUNDATIONAL.md:269](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/FOUNDATIONAL.md#L269) mandates a fourteen-predicate inventory, and [ADR-012:84](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-012.md#L84) prescribes dictionary and base-predicate fallback. FOUNDATION's inventory includes `dependsOn` and `precedes`, while the Python tuple instead includes `mentions` and `prevents`. The correction therefore requires a clear change to the ownership contract as well as implementation reconciliation.

There is a concrete direction error: [src/yaw/nlp/components.py:458](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/components.py#L458) maps “consists of”, “comprises” and “composed of” to `partOf`. Applied without exchanging arguments, “A consists of B” becomes “A partOf B”. A predicate-string-only return value cannot express the required inversion. The same table is prescribed in [docs/epic7-pluggable-ontology-integration-spec.md:820](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/epic7-pluggable-ontology-integration-spec.md#L820).

**Correction:** make ontology profiles and their lexical annotations authoritative, including the built-in profile. Preserve the original predicate and arguments; store alignment separately, including any argument transformation and supporting evidence. Generate membership sets, lexical indexes, inverse handling and applicable constraints from those resources. Treat vector similarity as a candidate signal, not proof of predicate equivalence. A finite antonym list cannot protect arbitrary negation or modality; distributional similarity and synonymy are distinct, as illustrated by [counter-fitting research](https://aclanthology.org/N16-1018/).

Adding entries to the Python synonym table would perpetuate the problem. Resolve the ownership contract first.

### R2 — P1: Competing extraction pipelines and an incomplete output contract

[docs/adr/ADR-001.md:22](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-001.md#L22) explicitly replaces SpanCat + REL with one `spacy.Raw.v1` pass. [README.md:57](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/README.md#L57) still describes NER.v3 + REL.v1, while [docs/how-to-handle-spacy-llm.md:120](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/how-to-handle-spacy-llm.md#L120) prescribes five canonical tasks. [docs/open-ie-authoritative-source.md:77](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/open-ie-authoritative-source.md#L77) and parts of the roadmap retain the superseded pipeline.

The newer contract also promises more than it encodes. ADR-001's payload and [src/yaw/nlp/templates.py:70](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/templates.py#L70) contain only subject, predicate and object strings. [FOUNDATIONAL.md:266](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/FOUNDATIONAL.md#L266) promises extraction and nine-root typing, while the guide types generated entity definitions through TextCat. A document category is not a complete per-mention type assignment. [src/yaw/nlp/models.py:114](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/models.py#L114) lacks occurrence offsets, qualified-assertion fields and a literal-versus-resource argument distinction.

The omissions matter for ordinary text such as “According to Lee, A may not cause B under condition C.” Flattening this into `A causes B` changes the claim. Relation-only output also leaves standalone mentions, repeated occurrences, nested spans, coreference and n-ary relations unspecified.

**Correction:** specify one versioned extraction payload and its mapping into storage. It should represent mentions independently of relations; preserve source occurrences, raw predicates and qualifiers; distinguish entity/class/concept references from typed literals; and define malformed, empty, unsupported and ambiguous results. Decide explicitly which coreference, n-ary and cross-chunk cases are supported.

A structured response schema is compatible with Open IE: fixed field names do not impose a closed domain relation inventory. Use Raw with a validated adapter or a registered custom task through [spaCy's public task protocol](https://spacy.io/api/large-language-models). Retain one generative extraction pass unless an additional pass has an explicit quality benefit and cost budget. [MinIE](https://aclanthology.org/D17-1278/) motivates polarity, modality, attribution and quantity annotations; [OpenFact](https://aclanthology.org/2023.tacl-1.40/) further motivates explicit factuality treatment.

### R3 — P1: The identity protocol can convert lexical similarity into false identity

[docs/adr/ADR-012.md:65](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-012.md#L65) treats normalized keys and ontology labels as deterministic identity. Equal spellings can denote different things: Mercury the planet and mercury the element can both fall within a broad entity partition. Labels should retrieve candidates unless a particular binding is known to be unambiguous.

The second tier permits hard merging when one of the top five candidates passes a cosine threshold plus acronym, head-containment or initial-expansion corroboration. Head containment can express specialization rather than identity: “cell” and “cancer cell” share a head. Uniqueness within five retrieved candidates does not establish uniqueness in the world. Yet the resulting merge projects as `owl:sameAs`. See [docs/adr/ADR-012.md:73](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-012.md#L73).

The audit log is described as labelled calibration data at line 82, but the system's own heuristic verdicts are not independent ground truth. There is also a lifecycle contradiction: [docs/adr/ADR-012.md:86](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-012.md#L86) says wrong merges require re-ingestion to undo; [docs/adr/ADR-005.md:89](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-005.md#L89) says alias changes are immediately reversible through views.

**Correction:** separate mention occurrence, ambiguous lexical alias and canonical identity. Keep uncertainty explicit and preserve reversibility. Set a higher evidence requirement for identity than for a retrieval link, evaluate on independently annotated ambiguous mentions, and define which projections must be rebuilt after an alias correction. This need not introduce generative LLM adjudication. spaCy's [KnowledgeBase](https://spacy.io/api/inmemorylookupkb) supplies candidate priors; its [EntityLinker](https://spacy.io/api/entitylinker) is a separate contextual component requiring appropriate initialization/training and evaluation.

### R4 — P1: OWL classes, SKOS concepts and instances are still conflated

[docs/adr/ADR-009.md:51](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-009.md#L51) establishes a project-specific non-punning policy. Older normative-looking tables still assign both `owl:Class` and `skos:Concept` to the same node: [docs/adr/ADR-005.md:64](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-005.md#L64), [docs/adr/ADR-008.md:30](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-008.md#L30) and examples in the RDF manual. Mark their affected sections explicitly superseded and update the live mapping contract. OWL itself permits certain punning; YAW's stricter policy is an architectural choice, not a universal W3C prohibition.

The newer ADR-012 still allows note mentions to hard-merge with external ontology classes and emits `skos:closeMatch` for cross-ontology soft matches ([docs/adr/ADR-012.md:111](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-012.md#L111)). [src/yaw/storage/rdf.py:220](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/storage/rdf.py#L220) chooses SKOS mapping from endpoint namespace, without checking the endpoint's semantic kind. SKOS mapping relations entail concept membership through their property hierarchy. Therefore an OWL class endpoint can acquire `skos:Concept` typing, conflicting with YAW's policy. [SKOS Reference, semantic and mapping relations](https://www.w3.org/TR/skos-reference/#mapping).

**Correction:** base projection on semantic kind and evidence. Instance membership uses `rdf:type`; class subsumption uses `rdfs:subClassOf`; class equivalence requires its own evidence; SKOS mappings connect concepts. An uncertain cross-kind retrieval association needs a suitably defined application relation, not automatic identity or SKOS mapping. Recognizing a mention of a class is also different from asserting that an individual is that class.

Two more concrete inconsistencies belong in this correction:

- ADR-009 removes synthetic `predicates:subClassOf` wrappers at line 56, but [src/yaw/ontology.ttl:114](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/ontology.ttl#L114) still declares one.
- [docs/adr/ADR-008.md:34](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-008.md#L34) maps a textual invariant to an executable SHACL shape and an algorithm/workflow to a PROV activity. These need explicit distinctions: a described rule is not automatically a valid constraint, and a plan is different from an execution. See [PROV-O](https://www.w3.org/TR/prov-o/).

### R5 — P1: Grounding promises mix characters, bytes, lines and content identity

[FOUNDATIONAL.md:104](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/FOUNDATIONAL.md#L104) specifies line-based hydration; its architecture diagram and retrieval section specify byte seeking ([FOUNDATIONAL.md:471](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/FOUNDATIONAL.md#L471)). [src/yaw/retrieval/retriever.py:43](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/retrieval/retriever.py#L43) actually hydrates by text lines.

[docs/adr/ADR-001.md:43](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-001.md#L43) claims token contraction/expansion and overlap filtering guarantee exact byte provenance. spaCy's `char_span` uses character offsets; token snapping can change boundaries. `filter_spans` removes overlapping spans rather than preserving every nested mention. See [Doc API](https://spacy.io/api/doc). [src/yaw/nlp/parsing.py:34](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/nlp/parsing.py#L34) additionally searches text for a matching string, including fallback matching, so repeated occurrences need an explicit selection contract.

Chunk identity is another ambiguity: [FOUNDATIONAL.md:258](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/FOUNDATIONAL.md#L258) and [src/yaw/indexer/pipeline.py:182](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/indexer/pipeline.py#L182) derive it from heading and content, without source path or occurrence. Identical sections in different files therefore produce the same content identifier. Content deduplication can be useful, but that identifier cannot also uniquely identify a source occurrence without a separate association.

**Correction:** define source identity and revision, occurrence identity, offset units and newline handling. Preserve exact matched spans independently of token-aligned spans; keep overlaps in `Doc.spans`. Separate content-cache identity from source occurrence identity. Verify the source revision at hydration, or retain a referenced snapshot; a filesystem watcher alone cannot guarantee that a file has not changed between indexing and retrieval.

Acceptance cases must include repeated identical strings, nested mentions, non-ASCII text, CRLF/LF, identical sections in different notes and edits between ingestion and querying.

### R6 — P1: The cache guarantee is unsafe for the inspected dependency

[docs/adr/ADR-001.md:29](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-001.md#L29) claims deterministic caching of identical contents through `spacy.BatchCache.v1`.

The local installed source identifies itself as **spacy-llm 0.7.4**. In `.venv/lib/python3.14/site-packages/spacy_llm/cache.py:159–164`, `_doc_id` sums token `ORTH` identifiers. This is order-insensitive: “Alice supports Bob” and “Bob supports Alice” have the same token multiset and therefore the same ID. Lines 228–269 retrieve a cached Doc by that ID without an exact ordered-text comparison. This is a source-level correctness finding, not a completed Windows execution test.

Cache initialization at lines 71–120 compares the static prompt template, not a full identity of rendered examples, model weights/digest and extraction configuration. A template change is detected, but that does not establish all the invalidation guarantees YAW needs.

Source fingerprint, SHA-256: `427D448F5101F92F25456D59D6BE925765209704AA443ABBB59AD7B3A9373109`. This is locally inspected dependency evidence, not a claim that the upstream release tag was independently downloaded and verified.

**Correction:** require exact ordered input plus task/schema/model/prompt configuration in cache identity, with component-specific namespaces. Verify a suitable upstream fix or use the public cache extension interface. Test reversed arguments, changed examples/model and a persisted-cache restart before making a provenance guarantee.

### R7 — P1: “Lossless mode switching” conflicts with extraction-time restriction

[docs/adr/ADR-009.md:44](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-009.md#L44) promises immutable, lossless raw extractions and a read-time quadrant toggle. Its closed external mode simultaneously scopes mentions through NER/trie during ingestion. The Epic 7 specification repeats both ideas.

If those candidates are only alignment hints, document that clearly. If they restrict what the model may extract, switching the view cannot recover missing assertions without re-extraction. This is an unresolved specification ambiguity; it is not proof that every current execution uses a restrictive prompt.

**Correction:** keep open raw extraction independent of ontology-view filters, and version downstream alignment separately. A genuinely domain-restricted extraction mode must declare its own cache identity and recomputation behavior. The predicate replacement in R1 already violates the stronger promise that raw relation surfaces remain untouched.

### R8 — P2: Reasoning, validation and induced hierarchy need explicit limits

[FOUNDATIONAL.md:528](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/FOUNDATIONAL.md#L528) promises OWL 2 RL/RDFS inference, while [docs/adr/ADR-009.md:82](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-009.md#L82) specifies subsumption/subproperty pre-closure followed by `inference=None`. Those closures alone do not implement the entire [OWL 2 RL rule set](https://www.w3.org/TR/owl2-profiles/#Reasoning_in_OWL_2_RL_and_RDF_Graphs_using_Rules). Document the supported rules, treatment of equality, inverses, domains/ranges, disjointness, datatypes and depth limits. Test that profile against representative entailments and explicitly reject or report unsupported axioms.

SHACL needs a defined validation graph and inference policy. Opposing source claims can legitimately coexist in different contexts. “Causes” and “prevents” are not universally incompatible without fixing subject, object, conditions, time and assertion status. A SHACL report checks modeled constraints; it does not establish that the source statement is true. Keep conflict discovery distinct from logical inconsistency under the selected model.

[docs/adr/ADR-002.md:21](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/adr/ADR-002.md#L21) promotes LLM-named clusters directly into `subClassOf` edges under the concepts namespace. Cluster similarity does not establish class subsumption. Preserve provenance for induced hierarchy and distinguish suggested organization from admitted axioms. Where the objects are SKOS concepts, use the declared thesaurus model. In particular, SKOS separates `broader` from `broaderTransitive`; audit the transitive custom wrappers at [src/yaw/ontology.ttl:132](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/ontology.ttl#L132) against that intended distinction.

Performance claims also need bounded workloads: returning k violations costs at least proportional to k; “O(1)” whole-result retrieval and universal “<200 ms” validation are not workload-independent guarantees. Ward clustering is not a guarantee of balanced branches or unchanged partitions as new points arrive. Its metric requirements and tie behavior should follow [SciPy's linkage documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html), followed by YAW measurements.

### R9 — P2: Scientific sources support the direction, not all claimed guarantees

The literature base is useful, but several claims overreach it:

| Source | What it supports for YAW | Necessary limit or correction |
| --- | --- | --- |
| [Banko et al., IJCAI 2007](https://www.eng.utah.edu/~cs6961/papers/banko-ijca07.pdf) | Relation-independent extraction rather than a predefined domain relation list. | The block quote in [docs/open-ie-authoritative-source.md:23](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/open-ie-authoritative-source.md#L23) combines paraphrase and attribution as if verbatim. Replace it with an accurate paraphrase or a verified short quotation. |
| [Niklaus et al., COLING 2018](https://aclanthology.org/C18-1326/) | A survey of Open IE's goals and methods. | Historical cautions about NER do not establish a ban on using NER as one candidate source. Avoid making named entities the only discoverable arguments. |
| [Liu et al., arXiv:2208.08690v7](https://arxiv.org/abs/2208.08690v7) | A survey covering Open IE through 2024. | The verified record is a preprint revised 23 October 2024; the “ACL/EMNLP 2024” attribution in [docs/open-ie-authoritative-source.md:10](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/docs/open-ie-authoritative-source.md#L10) is not substantiated by that record. |
| [Zhang & Soh, EDC, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.548/) | Separating extraction, schema definition and canonicalization. | These are logical responsibilities, compatible with incremental ingestion. The paper does not validate YAW's particular no-LLM identity heuristics. |
| [HippoRAG 2](https://arxiv.org/html/2502.14802v2) | Motivation and ablations for query-to-fact retrieval. | Its reported 12.5-point Recall@5 difference is a benchmark result under that experiment, not a predicted YAW improvement. Model, extraction and filtering choices differ. |
| [BenchIE](https://aclanthology.org/2022.acl-long.307/) and [BenchIE-FL](https://aclanthology.org/2024.findings-acl.496/) | Fact-based extraction evaluation, including alternative valid realizations. | QA retrieval scores alone cannot verify mention grounding, qualified assertions, predicate direction or identity safety. |

ADR-012's separation of linguistic, similarity and retrieval representations is a worthwhile improvement. Its warning against interpreting an unmerged pair as `owl:differentFrom` is also sound. Keep those distinctions while correcting the hard-merge protocol and projection typing.

**Correction:** add independently annotated extraction/alignment cases alongside retrieval evaluation, with calibration and held-out splits. Measure precision/recall, false identity merges, direction/polarity errors, unsupported assertions, grounding accuracy, latency and model calls. Do not treat a loopback model fixture as semantic validation of the actual model. Derive thresholds from a baseline and acceptable error costs rather than inventing universal pass percentages.

### R10 — P2: Document authority and platform instructions have drifted

The docs need one explicit precedence and supersession map. In particular:

- Reconcile FOUNDATION, ADR-001, the spaCy guide and README around one extraction contract.
- Reconcile ADR-005/008 with ADR-009's semantic-kind policy and ADR-012's current identity model.
- Update live BDD scenarios and the old sprint plan that still require a `contested`/third-tier resolver after ADR-012's replacement.
- Replace host-specific `file:///home/harri/yaw/...` documentation links with repository-relative links.
- Correct README's Python prerequisite against [pyproject.toml:5](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/pyproject.toml#L5) (`>=3.14`).
- Replace copied semantic inventories/code examples with links to the authoritative ontology resources and generated reference material.

The older documents already marked SUPERSEDED were not counted as competing current requirements merely because their historical designs differ.

## Recommended contract to reconcile the suite

This is a review proposal, not a silently adopted replacement ADR.

1. **Source evidence:** stable source and revision identifiers; distinct chunk/mention occurrences; explicit character, byte and line coordinates.
2. **Open extraction:** one versioned payload for independent mentions and qualified assertions, retaining original wording and uncertain results.
3. **Semantic ownership:** ontology profiles, lexical resources and SHACL shapes define domain vocabulary and semantics. Runtime indexes are compiled from those resources. An empty optional profile is not replaced by a hidden Python vocabulary.
4. **Alignment:** source extraction is immutable; mappings, argument orientation, canonical identity candidates and accepted identities are separate, versioned records.
5. **Projection:** RDF distinguishes resources and literals, and projection respects individual/class/concept/property kinds. Uncertain and attributed assertions retain their context.
6. **Reasoning and validation:** an explicit supported entailment profile; a declared graph scope and shape version; induced hypotheses identifiable separately from admitted axioms.
7. **Retrieval:** PPR uses the resulting graph and returns source evidence with revision checks. Numerical relevance and alias weights are not presented as calibrated probabilities of truth.
8. **Change handling:** content, extraction model/task, ontology, alignment and validation versions have separate invalidation rules. A view change need not repeat extraction; a changed extraction prompt may.

### Using spaCy's capabilities appropriately

“Full capabilities” should mean choosing the right supported components, not requiring every generative task on every chunk.

| Capability | Appropriate use |
| --- | --- |
| Tokenizer, tagger, parser, morphology and lemmatizer | Preserve the source Doc and derive linguistic features. Lemmatization alone does not establish logical equivalence or argument orientation. |
| `Doc.spans` and [SpanRuler](https://spacy.io/api/spanruler) | Preserve nested/overlapping mentions and add ontology-derived lexical candidates without restricting open discovery. |
| Raw or a public custom spacy-llm task | One structured open extraction with schema validation and explicit error handling. |
| KnowledgeBase and EntityLinker | Candidate priors and, if selected and trained/evaluated, contextual entity linking. Keep this distinct from the separate LLM EntityLinker task. |
| TextCat / SpanCat / NER | Optional annotation tools with an explicit input/output purpose. Document-level TextCat does not automatically type each extracted entity. |
| Summarization | Optional derived descriptions; preserve their provenance and account for extra generation. |
| [Coreference component](https://spacy.io/api/coref) | An explicit, evaluated dependency if cross-mention resolution is required; it is not automatically provided by loading `en_core_web_lg`. |

### Acceptance cases before calling the specification coherent

| Case | Required observable result |
| --- | --- |
| Load an unfamiliar ontology property with labels, inverse information and a SHACL constraint | Alignment, projection and validation use it without edits to Python relation dictionaries or SQL whitelists. |
| Extract an unknown relation with no matching domain ontology | Preserve the surface assertion and evidence; retain a representable open relation. |
| Configure an empty predicate profile | Honor the documented meaning; do not silently restore a handwritten inventory. |
| “A consists of B” versus “A is part of B” | Correct direction or an explicitly unresolved alignment; original assertion retained. |
| Positive, negated, modal and attributed versions of a sentence | Preserve those differences in stored assertions and retrieved fact representations. |
| Homonyms, ambiguous acronyms and specialization pairs | No hard identity merely because the normalized label or one nearest candidate matches. |
| Match a note mention to an OWL class and to a SKOS concept | Correct kind-aware links; no accidental concept typing of a class through SKOS mapping. |
| Repeated/nested mentions, Unicode, CRLF and duplicate note sections | Correct distinct occurrences and verified source slices. |
| Reverse sentence arguments; change model or examples; restart | No inappropriate extraction cache reuse. |
| Toggle view scope, then change an ontology | Raw assertions remain intact; affected projections refresh with documented versioning. |
| Evaluate disjointness under different attribution/time scopes | Report the intended scoped conflict rather than an unqualified global contradiction. |
| Run real-model extraction and held-out retrieval evaluation | Report measured accuracy and costs separately from fixture/plumbing tests. |

## Windows and Unix

Use **one semantic configuration and ontology suite on both platforms**. Keep OS differences in launchers, path/cache locations, file locking, process management and service integration; avoid separate Windows and Unix predicate/type definitions.

The dev daemon already contains a Windows locking branch. However, [src/yaw/server/daemon.py:633](https://github.com/negatrait/yaw/blob/05a81cb3f645a708e0b3fa3627591d3884d76386/src/yaw/server/daemon.py#L633) still probes liveness with `os.kill(pid, 0)`. On Windows, Python documents non-control-event values as invoking termination, so this is not a safe portable liveness check. See [Python 3.14 os.kill](https://docs.python.org/3.14/library/os.html#os.kill). Use a non-terminating Windows process-status mechanism and exercise platform-specific lifecycle tests before treating the Windows daemon as verified. This review did not execute that probe.

The Windows environment setup remains incomplete: the available Windows virtual environment did not contain spaCy when checked. Local dependency source was inspected in the existing Linux virtual environment; that is not a Windows pipeline run.

## Coverage and validation

The review covered the documentation suite at the commit above, starting with FOUNDATION and all twelve ADRs. Depth followed relevance to the requested extraction/ontology contracts:

| Coverage | Files |
| --- | --- |
| Controlling architecture and decisions | `FOUNDATIONAL.md`; `docs/ADR.md`; `docs/adr/ADR-001.md` through `ADR-012.md`. |
| Detailed extraction, ontology and roadmap consistency | `docs/how-to-handle-spacy-llm.md`; `docs/open-ie-authoritative-source.md`; `docs/rdf-owl-shacl.md`; `docs/epic7-pluggable-ontology-integration-spec.md`; `docs/epics-alignment-roadmap.md`; `docs/yaw-architecture-plan.md`. |
| Retrieval and hierarchy contracts relevant to semantic interpretation | `docs/power-pagerank.md`; `docs/ontological-roots.md`; relevant YAW sections of `docs/efficient-ppr-algorithms.md` and `docs/scipy-graphrag-extensions.md`. This was not a proof audit of every imported mathematical derivation. |
| User-facing, implementation and test-plan consistency | `README.md`; `docs/USER_STORIES.md`; `docs/alignment-sprint-for-e5s2.md`; root test-readiness/infrastructure and historical audit documents. |
| Reference material, checked against current primary sources where relevant | `docs/cocoindex.md`; `docs/spacy-llm-api.md`; `docs/spacy-llm-docs.md`; `docs/karpathy-llm-wiki.md`. Copied upstream material was treated as reference, not a newer YAW decision. |
| Historical/deprecated status checked | `docs/cocoindex-neo4j.md`; `docs/llm-wiki-spec-v2.md`; `docs/llm-wiki-spec-addendum.md`; `docs/llm-wiki-fts-vector-graph-playbook.md`; `docs/how-to-handle-spacy-gliner.md`. |

Targeted source inspection traced the predicate normalizers, canonicalizer, ontology loader/registry, SQL views, RDF projection, SHACL construction, retrieval, grounding and cache behavior. It establishes the concrete source findings above, not complete runtime coverage of the application.

No production code, accepted ADR or existing specification was changed for this review. The recommended order is **semantic ownership and extraction/grounding contract first, kind-aware identity/projection next, then reasoning/evaluation contracts and documentation reconciliation**. Fixing isolated predicate strings before that would leave competing sources of truth in place.

## Alignment work sequence

This section turns the findings into a proposed sequence of changes. It is a work plan attached to the review, not another source of normative requirements.

### 1. Settle the ownership contract in FOUNDATION and the ADRs together

The first coherent documentation change should establish:

- **Open discovery:** entities, relation phrases and qualified assertions emerge from notes. Imported ontologies are optional alignment resources; their inventories do not bound discovery.
- **Emergent knowledge:** unfamiliar relations receive stable, source-grounded identities and remain queryable. Automatically proposed definitions, alignments and hierarchy carry derivation/provenance and status; uncertain induction does not silently become an asserted logical axiom.
- **Declarative domain semantics:** classes, property meanings, lexical annotations, inverse orientation and validation constraints belong to versioned RDF/OWL/SKOS/SHACL resources. The existing nine-root organization can remain a declared default profile without becoming an exhaustive domain type system.
- **Generic execution:** Python performs linguistic analysis, extraction orchestration, parsing, index compilation, supported inference, validation and retrieval. Runtime lookup structures are generated from the active semantic resources. A fixed application relation inventory must not be hidden in a fallback.
- **Separate evidence and interpretation:** source assertions are retained; alignments and identities are revisable. A lexical match or vector score alone does not establish logical equivalence.
- **Preserved architecture:** keep human-authoritative raw notes, a single incremental CocoIndex pipeline, one SQLite database, in-process reasoning/retrieval, optional wiki materialization and no query-time generative synthesis.

Moving the fourteen names from Python into Turtle, while still forcing every extraction into them, would not satisfy this contract.

Record the rationale in one focused amendment ADR with an explicit section-level supersession map. Update the affected live text and ADR ledger in the same change; preserve historical rationale behind clear supersession notices. Do not leave readers to infer which contradictory paragraph is newer.

### 2. Specify the smallest complete extraction-to-retrieval contract

Define one source-grounded, versioned logical payload before prescribing a database migration:

| Record | Minimum responsibility |
| --- | --- |
| Source occurrence | Source/revision identity, distinct chunk/mention occurrence, defined coordinate units. |
| Mention | Exact occurrence, surface form and any evidenced type annotations; independent of whether a relation was found. |
| Assertion | Mention/literal arguments, original relation phrase, direction, applicable qualifiers and evidence references. |
| Interpretation | Candidate or accepted alignment, semantic kind, argument transformation, evidence and resource version. |
| Derived graph structure | Declared versus induced origin and the status under which it participates in inference, validation and retrieval. |

Use one generative extraction pass through supported spaCy/spacy-llm interfaces. State unsupported cases and parse failures explicitly. Do not require all five LLM tasks merely because they exist.

Keep raw prose in its authoritative files. Revision hashes and occurrence metadata make citation verification possible without adding duplicate chunk-text storage. When source content has changed, report stale evidence and refresh the affected index; do not present the new file slice as verified old evidence.

Specify the supported OWL/RDFS rules and SHACL graph scope, not a blanket conformance claim. Define cache identity from ordered input and actual extraction configuration; define ontology/alignment invalidation independently. Keep Windows and Unix semantic behavior identical.

### 3. Reconcile existing documents by responsibility

| Existing document group | Required change |
| --- | --- |
| FOUNDATION; ADR ledger; ADR-001/003/005/006/008/009/011/012 | Establish the controlling contract; amend extraction, semantic ownership, identity, projection, scope switching and reasoning/cache claims. |
| ADR-002; hierarchy and retrieval guides | Distinguish induced organization from logical axioms; obtain relation-specific behavior from the declared profile; preserve representation-space separation and source grounding. |
| spaCy integration guide | Describe the selected public APIs and the one current payload/pipeline; remove competing NER/SpanCat/REL recipes and copied predicate tables. |
| RDF/OWL/SHACL manual; Epic 7 specification; architecture plan | Reconcile semantic kinds, registry information, validation scope and derived projections; link authoritative resources instead of copying their inventories. |
| Roadmap; USER_STORIES; TEST_INFRA/TEST_READY | Replace obsolete resolver scenarios, map acceptance cases to findings, separate intended behavior from implementation and measured verification status. |
| README; research/reference and archived documents | Give a concise current overview, repair scientific attribution and portable links, and make historical status unmistakable. |

Do this as a coordinated documentation change, followed by a consistency pass across the entire suite. The present implementation must be described as a migration starting point wherever it differs from the target; updating prose is not evidence that behavior has changed.

### 4. Make the first implementation slice prove semantic ownership

Use a small source note and an unfamiliar ontology property as the first complete example:

1. Extract the original relation without that ontology loaded.
2. Load its property declaration, lexical information, inverse relationship and a relevant SHACL shape.
3. Refresh derived alignment, projection and validation without modifying Python or losing the raw assertion.
4. Retrieve the result with the correct direction and verified source occurrence.
5. Change or remove the alignment resource; rebuild the affected interpretation while retaining the source assertion.

Then cover ambiguity, qualifiers, semantic kinds, cache collisions and scope changes with the acceptance cases above. Only after that behavior is established should the corresponding handwritten tables and SQL branches be removed.

Plan migration explicitly: records whose relation surfaces were already overwritten cannot be made lossless by changing a view. Mark them for re-extraction from available authoritative notes. Review aliases produced by the old hard-merge rules and rebuild dependent projections; preserve audit history and avoid silently blessing existing merges.

### 5. Keep the suite aligned thereafter

Give each contract one normative owner and link from its guides. Add bounded checks for documentation links, supersession/status consistency and semantic examples; use behavior tests to enforce ontology-driven extension rather than brittle searches banning all predicate strings. Legitimate vocabulary constants and test examples will still contain them.

Close a finding only with its documentation correction and the appropriate implementation evidence. Run real-model extraction evaluation separately from fixture tests, and report Windows/Unix runtime verification separately from semantic correctness.
