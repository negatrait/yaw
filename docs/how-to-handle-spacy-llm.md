# spaCy and spacy-llm integration

**Status:** Selected recipe; implementation pending [Epic 7.8](epic7.8-open-ie-ontology-alignment.md).\
**Authority:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-014](adr/ADR-014.md).\
**Compiler integration:** [Continuous compilation](knowledge-compilation.md).

## One component recipe

Keep spaCy/spacy-llm. Use its public pipeline, task, model, parser and extension interfaces. The following order is the production recipe, not a menu of alternatives:

1. Preserve the exact primary source chunk in `Doc.text`; attach the host's source/context manifest through documented extensions. `markdown_structure` annotates regions without deleting or replacing source text.
2. Run the configured `en_core_web_lg` linguistic pipeline, including NER, morphology, dependencies and sentence annotations. Retain candidates rather than overwriting them with a fixed domain classification.
3. Run one `llm` component using **`spacy.Raw.v1`**, the configured **`yaw.Ollama.v1`** model adapter and the versioned extraction prompt. Source context and the fixed linguistic annotations are its inputs; active ontology labels/examples are not. Disable spaCy's persistent BatchCache; CocoIndex owns generation-result memoization.
4. Pass the raw response once to `src/yaw/nlp/parsing.py::parse_generation`. It imports the Pydantic record types from `src/yaw/nlp/models.py` and owns grounding/admission. Store typed results and diagnostics; Doc attributes are not a second persistence schema.
5. In the compiler's alignment stage, apply a resource-derived `SpanRuler` to the source annotations for lexical/mention candidates. It uses `Doc.spans`, the active resource version and the same generic component implementation. Its outputs are derived annotations, never new source assertions. They do not enter the extraction prompt/cache or gate discovery. A resource change reruns alignment while reusing unchanged extraction.

Raw supports configured prompting/parsing but does not itself provide YAW's qualified schema or verify evidence. All LLM operations use the same public wrapper/adapter factory. [Public spacy-llm interfaces](https://spacy.io/api/large-language-models).

Delete REL generation, LLM NER/TextCat/EntityLinker chains, direct parallel Ollama extraction calls, injected/private task mutation, alternative parser ladders and source-text-as-model-response fallback. There is no generic capability-discovery dispatcher. Missing components or models fail startup/processing explicitly.

## Linguistic assistance and languages

Linguistic annotations supply sentence/argument candidates, lemmas and dependency context. They are hints, not ground truth or a whitelist. Preserve standalone and overlapping mentions in `Doc.spans`; a non-overlapping `Doc.ents` projection must not delete them. [Linguistic features](https://spacy.io/usage/linguistic-features), [SpanRuler](https://spacy.io/api/spanruler).

English is the configured linguistic profile, not an assumption that every source is English. Preserve original text, language-tagged labels and literals for all sources. The same extraction task reports source language and uncertainty; English POS/NER/lemmas cannot reject non-English claims or establish their meaning. No automatic language-model substitution or translation is selected. The real-model pilot must include non-English and mixed-language documents; unsupported interpretation is explicit partial/error coverage, never silent English normalization.

The host's deterministic chunk/context builder owns context limits. The Raw component does not silently re-shard or truncate. Primary and adjacent context have distinct coordinates; follow the [compiler's source-unit rules](knowledge-compilation.md#1-source-extraction). Local reference resolution needs evidence for the referring expression and antecedent. Arbitrary cross-document coreference and n-ary decomposition are outside payload v1 and retain unsupported diagnostics.

## Typed claims and grounding

FOUNDATIONAL owns payload meaning; `src/yaw/nlp/models.py` owns executable extraction, comparison and synthesis models. `src/yaw/nlp/parsing.py` owns the one `parse_generation` adapter and exact evidence validation. The factory, pipeline and compiler import these owners directly. Derive prompt schemas from the models; do not copy them into templates or fixtures.

Extract subject/object mentions or literals, the original predicate phrase, contextual definitions when supported, qualification and exact evidence. The host supplies source/revision/occurrence/configuration identifiers. Distinguish source-supported definitions, generated explanations and unknown values. A paraphrased predicate is not a verbatim source quote.

Validate Unicode-code-point spans against unchanged source text before using them as evidence. Preserve repeated, overlapping, nested and discontinuous occurrences. Token expansion/contraction is separate annotation and does not replace exact evidence. Never select the first occurrence of an ambiguous substring or snap an invalid model span into apparently valid evidence.

Use one parsing sequence for the declared task:

1. Decode exactly one JSON object. The selected prompt requires bare JSON; reject prose/fences, malformed JSON or an invalid outer container. Do not try a second grammar or parse source text.
2. Validate the outer envelope's task/schema discriminator and collection structure first, then validate each record with its Pydantic type. Do not validate all records as one atomic list and thereby lose valid independent records when one item fails.
3. Validate source coordinates, host-provided reference IDs and inter-record references. Reject a record whose required dependency was rejected; retain independent valid records and diagnostics. The host assigns persisted IDs and computes status; model-declared success cannot override validation.
4. A structurally invalid envelope is `failed`. A parseable envelope with any rejected/unsupported record is `partial`, including when no valid records survive. A valid result with zero records is `complete` with an empty payload. Comparison/synthesis follow the same rule at their independent record boundary.

These are validation steps in one parser, not fallback parsers. There is no substring search, span snapping, JSON repair or additional generation to make an invalid response pass. Verify a two-record response with one invalid span: exactly the supported record survives, the scope is partial and it is not cached/published as complete.

## Planned compiler generation

The same Raw/Ollama factory handles two additional versioned input/output contracts:

| Responsibility | Inputs and output | Boundary |
| --- | --- | --- |
| Claim comparison | A candidate pair, its qualification and evidence; support/conflict/supersession/scope/unknown interpretation | Cannot create original assertions, accept identity or decide truth by recency |
| Synthesis | Current group members, qualified claims, comparisons, findings and derivations; supported page blocks or organizational labels | Cannot create executable axioms/rules or conceal contested evidence |

These calls run at the stages defined in the compiler guide for changed inputs. They are not optional extraction passes, failure recovery or query-time generation. Bound their inputs before execution; version/cache outputs and record calls separately. Exact schema validation is not a claim of semantic faithfulness; that is measured with independent labels.

## Alignment and identity

Compile lexical frames, qualified property IDs, labels and semantic traits from resources. Preserve emitted phrases and evidence while recording any accepted property mapping and argument orientation separately. Lemmas or vector similarity cannot establish property equivalence. A declared inverse alone cannot interpret an arbitrary phrase.

`InMemoryLookupKB` indexes identity candidates and normalized evidence weights. It is not a trained contextual resolver. Keep all candidates under FOUNDATION's explicit binding policy; never accept the first entry. No trained spaCy EntityLinker or LLM EntityLinker is selected in this release. [KnowledgeBase](https://spacy.io/api/inmemorylookupkb), [EntityLinker](https://spacy.io/api/entitylinker).

## One generation-cache boundary

CocoIndex memoizes typed results using the exact ordered primary/context manifest (including occurrence/revision coordinates), rendered prompt/examples, task/schema identity, model digest/settings and linguistic configuration. Extraction examples are versioned task fixtures, not a live vocabulary query. The comparison/synthesis key also includes sorted membership, evidence, resource and binding versions. Outputs with invalid/partial coverage cannot be reused as complete results.

Disable spaCy BatchCache and remove the competing cache directories/call sites during migration. A model tag resolving to changed weights invalidates its entries. Resource changes invalidate their affected compiler stages; unrelated extraction remains compatible unless its actual inputs changed. No mutable database vocabulary read may be hidden inside a memoized extraction function.

## Verification

Use [E78-01–20](USER_STORIES.md), real storage/RDF/matrix tests and loopback fixtures for transport. The actual-model pilot separately measures mention/claim coverage, qualification/direction/grounding mistakes, false identity merges, comparison quality, multilingual coverage, synthesis faithfulness and cost. Fault injection must demonstrate that the selected path returns an error without invoking a second implementation.
