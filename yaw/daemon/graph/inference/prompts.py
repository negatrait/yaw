"""Native DSPy signatures and few-shot demonstration sets for HippoRAG 2."""

import json
from typing import Any

import dspy
from dspy.teleprompt.gepa.instruction_proposal import (
    CompressInstruction as CompressInstructionSignature,
)
from dspy.teleprompt.gepa.instruction_proposal import (
    ProposeInstruction as NativeProposeInstruction,
)
from pydantic import BaseModel, Field

NER_ONE_SHOT_PARAGRAPH = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music portal "
    "- PlanetRadiocity.com that offers music related news, videos, songs, and other "
    "music-related features."
)

RAG_QA_DOCS = (
    "Wikipedia Title: The Last Horse\n"
    "The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by "
    "Edgar Neville starring Fernando Fernán Gómez.\n"
    "Wikipedia Title: Southampton\n"
    "The University of Southampton, which was founded in 1862 and received its Royal Charter "
    "as a university in 1952, has over 22,000 students. The university is ranked in the top "
    "100 research universities in the world in the Academic Ranking of World Universities "
    "2010. In 2010, the THES - QS World University Rankings positioned the University of "
    "Southampton in the top 80 universities in the world. The university considers itself one "
    "of the top 5 research universities in the UK. The university has a global reputation for "
    "research into engineering sciences, oceanography, chemistry, cancer sciences, sound and "
    "vibration research, computer science and electronics, optoelectronics and textile "
    "conservation at the Textile Conservation Centre (which is due to close in October 2009.) "
    "It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of "
    "Natural Environment Research Council-funded marine research.\n"
    "Wikipedia Title: Stanton Township, Champaign County, Illinois\n"
    "Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, "
    "its population was 505 and it contained 202 housing units.\n"
    "Wikipedia Title: Neville A. Stanton\n"
    "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the "
    "University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered "
    "Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited "
    "over a forty books and over three hundered peer-reviewed journal papers on applications "
    "of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The "
    "Institute of Ergonomics and Human Factors and a member of the Institution of Engineering "
    'and Technology. He has been published in academic journals including "Nature". He has '
    "also helped organisations design new human-machine interfaces, such as the Adaptive "
    "Cruise Control system for Jaguar Cars.\n"
    "Wikipedia Title: Finding Nemo\n"
    "Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham "
    "Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton "
    "Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas "
    "Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production "
    "company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures "
    "Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country "
    "United States Language English Budget $94 million Box office $940.3 million"
)

best_dspy_prompt: dict[str, Any] = {
    "prog": {
        "demos": [
            {
                "augmented": True,
                "question": "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
                "fact_before_filter": '{"fact": [["imperial river", "is located in", "florida"'
                '], ["imperial river", "is a river '
                'in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", '
                '"flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united '
                'states"]]}',
                "fact_after_filter": '{"fact":[["imperial river","is located in","florida"],['
                '"imperial river","is a river '
                'in","united states"],["amaradia","flows through","ro ia de amaradia"]]}',
            },
            {
                "augmented": True,
                "question": "When is the director of film The Ancestor 's birthday?",
                "fact_before_filter": '{"fact": [["jean jacques annaud", "born on", "1 october'
                ' 1943"], ["tsui hark", "born on", '
                '"15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", '
                '"directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}',
                "fact_after_filter": '{"fact":[["the ancestor","directed by","guido brignone"]]}',
            },
            {
                "augmented": True,
                "question": "In what geographic region is the country where Teafuone is located?",
                "fact_before_filter": '{"fact": [["teafuaniua", "is on the", "east"], ["motulo'
                'a", "lies between", "teafuaniua"], '
                '["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", '
                '"located in", "nukufetau"]]}',
                "fact_after_filter": '{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}',
            },
            {
                "augmented": True,
                "question": "When did the director of film S.O.B. (Film) die?",
                "fact_before_filter": '{"fact": [["allan dwan", "died on", "28 december 1981"]'
                ', ["s o b", "written and directed '
                'by", "blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert '
                'siodmak", "died on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november '
                '2018"]]}',
                "fact_after_filter": '{"fact":[["s o b","written and directed by","blake edwards"]]}',
            },
            {
                "augmented": True,
                "question": "Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same "
                "country?",
                "fact_before_filter": '{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], '
                '["gloria", "is", "1980 american thriller crime drama film"], ["a brand new life", "is '
                'directed by", "ounie lecomte"], ["gloria", "written and directed by", "john cassavetes"], '
                '["a new life", "directed by", "alan alda"]]}',
                "fact_after_filter": '{"fact":[["gloria","is","1980 american thriller crime d'
                'rama film"],["gloria","written and '
                'directed by","john cassavetes"],["a new life","directed by","alan alda"]]}',
            },
            {
                "augmented": True,
                "question": "What is the date of death of the director of film The Old Guard (1960 Film)?",
                "fact_before_filter": '{"fact": [["the old guard", "is", "1960 french comedy f'
                'ilm"], ["gilles grangier", '
                '"directed", "the old guard"], ["the old guard", "directed by", "gilles grangier"], ["the '
                'old fritz", "directed by", "gerhard lamprecht"], ["oswald albert mitchell", "directed", '
                '"old mother riley series of films"]]}',
                "fact_after_filter": '{"fact":[["the old guard","is","1960 french comedy film"],["gilles '
                'grangier","directed","the old guard"],["the old guard","directed by","gilles grangier"]]}',
            },
            {
                "augmented": True,
                "question": "When is the composer of film Aulad (1968 Film) 's birthday?",
                "fact_before_filter": '{"fact": [["aulad", "has music composed by", "chitragup'
                'ta shrivastava"], ["aadmi sadak '
                'ka", "has music by", "ravi"], ["ravi shankar sharma", "composed music for", "hindi '
                'films"], ["gulzar", "was born on", "18 august 1934"], ["aulad", "is a", "1968 hindi '
                'language drama film"]]}',
                "fact_after_filter": '{"fact":[["aulad","has music composed by","chitragupta '
                'shrivastava"],["aulad","is '
                'a","1968 hindi language drama film"]]}',
            },
            {
                "question": "How many households were in the city where Angelical Tears located?",
                "fact_before_filter": '{"fact": [["dow city", "had", "219 households"], ["tucs'
                'on", "had", "229 762 households"], '
                '["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", '
                '"oklahoma city"], ["atlantic city", "had", "15 848 households"]]}',
                "fact_after_filter": '{"fact": [["angelical tears", "located in", "oklahoma city"]]}',
            },
            {
                "question": "Did the movies In The Pope'S Eye and Virgin Mountain, originate from the same country?",
                "fact_before_filter": '{"fact": [["virgin mountain", "released in", "icelandic'
                ' cinemas"], ["virgin mountain", '
                '"directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin '
                'mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 '
                'icelandic drama film"]]}',
                "fact_after_filter": '{"fact": [["virgin mountain", "released in", "icelandic'
                ' cinemas"], ["virgin mountain", '
                '"directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin '
                'mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 '
                'icelandic drama film"]]}',
            },
            {
                "question": "Which film has the director who died earlier, The Virtuous Model or Bulldog Drummond'S "
                "Peril?",
                "fact_before_filter": '{"fact": [["the virtuous model", "is", "1919 american s'
                'ilent drama film"], ["bulldog '
                'drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed '
                'by", "albert capellani"], ["bulldog drummond s revenge", "directed by", "louis king"], '
                '["bulldog drummond s peril", "is", "american film"]]}',
                "fact_after_filter": '{"fact": [["the virtuous model", "is", "1919 american s'
                'ilent drama film"], ["bulldog '
                'drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed '
                'by", "albert capellani"], ["bulldog drummond s peril", "is", "american film"]]}',
            },
        ]
    }
}

# ---------------------------------------------------------------------------
# Native DSPy Signatures and Few-Shot Demos
# ---------------------------------------------------------------------------


class FactList(BaseModel):
    """Typed Pydantic container for recognition-memory filtered facts."""

    fact: list[list[str]] = Field(
        default_factory=list,
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]",
    )


class ExtractEntities(dspy.Signature):
    """Extract ALL named individuals and ALL general domain concepts explicitly mentioned in `passage` prose.
    An `owl:NamedIndividual` is a particular person, organization, place, system, product or artifact.
    A `skos:Concept` is a general domain concept or category, such as a role, process or kind of thing.
    Output only deduplicated `[skos:prefLabel, skos:definition]` pairs. Use source-grounded noun phrases;
    each definition identifies what the resource is and its distinguishing role in this passage.
    Do not extract relationships, statements or document/header metadata. Do not extract `rdfs:Literal`
    data values such as numbers, dates, percentages, durations, booleans or descriptive strings as resources.
    A resource label is also a JSON string: a named person, place or concept is not a literal merely because
    its label is textual. Include every such resource, including less prominent subjects and objects.
    The fact extractor discovers those literal values while establishing relationships. Never use a label
    as its own definition."""

    passage: str = dspy.InputField(desc="Source `passage` text")
    named_entities: list[Any] = dspy.OutputField(
        desc="Deduplicated list of extracted `[skos:prefLabel, skos:definition]` pairs"
    )


class ExtractTriples(dspy.Signature):
    """Establish passage-supported `rdf:Statement` facts using only the supplied resources.
    Resource discovery and definitions belong exclusively to the entity extractor. Never introduce,
    rename or define a resource here. Subjects and resource objects are exact labels from `named_entities`,
    which contains both general `skos:Concept` categories and particular `owl:NamedIndividual` resources.
    If a label has multiple supplied definitions, select its exact supplied `[label, definition]` pair
    as the endpoint to identify that sense. Copy it verbatim; never invent or rewrite its definition.
    Emit objects with `"sub"` (`rdf:subject`), `"pred"` (`rdf:predicate`), `"obj"` (`rdf:object`) and `"def"`:
    - Between supplied resources, use `"pred": ["predicate phrase", "owl:ObjectProperty"]` and an exact
      supplied resource label as `"obj"`.
    - For data-valued relationships, discover the literal in the passage. Use `"pred": ["predicate phrase",
      "owl:DatatypeProperty"]` and `"obj": {"value": "lexical value", "datatype": "xsd:string", "lang": ""}`.
      Choose the appropriate datatype or language tag. Literals need not occur in `named_entities`;
      they are values, never new resources or `[name, definition]` pairs.
    Subjects are always supplied resources, never literals. Resolve pronouns to supplied labels, preserve
    predicate direction and literal meaning, and define each assertion in `"def"` without copying an entity
    definition. Do not invent relationships to connect unrelated resources or use header metadata."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.InputField(
        desc="Extracted `[skos:prefLabel, skos:definition]` pairs (or entity names) from Stage 1"
    )
    triples: list[Any] = dspy.OutputField(
        desc='List of reified rdf:Statement objects {"sub", "pred", "obj", "def"} where "pred" is '
        '["predicate phrase", "owl:ObjectProperty" | "owl:DatatypeProperty"], known resources are '
        "exact supplied labels (or exact supplied sense pairs for homographs); "
        'literal values are {"value", "datatype", "lang"} objects'
    )


class DefineMissingEntities(dspy.Signature):
    """Synthesize concise, grounded genus-differentia contextual definitions for entities referenced in \
`missing_entities` based on the passage text and extracted triples. These resources were already
discovered by NER; never discover, add or rename resources during definition repair.
Emit `[entity_name, contextual_definition]` for every requested concept using the exact string from \
`missing_entities`; never emit an empty definition or repeat `entity_name` tautologically."""

    passage: str = dspy.InputField(desc="Source passage text")
    missing_entities: list[str] = dspy.InputField(desc="Entity names that require contextual definitions")
    triples: list[Any] = dspy.InputField(
        desc="Extracted relation triples referencing the missing entities",
        default=[],
    )
    entity_definitions: list[Any] = dspy.OutputField(
        desc="List of [entity_name, contextual_definition] pairs for missing entities"
    )


SYNONYM_JUDGE_STAGE_NAMES: tuple[str, str, str] = (
    "label_judge",
    "hierarchy_judge",
    "related_judge",
)


class JudgeLabels(dspy.Signature):
    """Evaluate Stage 1 W3C SKOS concept equivalence and lexical identity (`skos:altLabel` and `skos:hiddenLabel`) \
for candidate pairs in `candidate_pairs` using a 2-step conditional branching heuristic:
    1. Concept Equivalence (`equivalent=true`): If `entity_a` and `entity_b` denote the same underlying concept or \
duplicate entity (e.g., identical surface names across passages like 'Jira' and 'Jira' or 'Change Management Policy' \
and 'Change Management Policy', or coreferent mentions of the exact same referent), emit `"equivalent": true` and \
synthesize a unified, non-tautological `"definition"` string combining their context to collapse them into a single \
concept.
    2. Distinct or Lexical Variant (`equivalent=false`): Branch on `"alt_label"`:
       - If one surface name is a valid alternative label, synonym, acronym, or abbreviation of the other \
(`skos:altLabel`), emit `"equivalent": false, "alt_label": true` (do not emit `"hidden_label"`).
       - Otherwise emit `"equivalent": false, "alt_label": false, "hidden_label": true` for a misspelling/typo \
(`skos:hiddenLabel`), or `"equivalent": false, "alt_label": false, "hidden_label": false` when `entity_a` and \
`entity_b` are distinct concepts advancing to Stage 2 hierarchy evaluation.
    Head-Noun Preservation Rule: Never merge a system/proper noun `X` with a distinct component, role, or artifact \
modified by `X` (such as `X Router`, `X secrets`, `XTenant`, or `CRDs`). A phrase with an additional head noun or \
modifier is NOT an alternative label (`alt_label: false, equivalent: false`)."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON object (`synonym-evidence-v2`) with inline definitions, facts, and source excerpts."
    )
    decisions: list[dict[str, Any]] = dspy.OutputField(
        desc=(
            'List of branched decision objects: if equivalent=true, emit {"pair_id": "e0|e1", "reasoning": "...", '
            '"equivalent": true, "definition": "<unified skos:definition>"}; '
            'if equivalent=false and alt_label=true, emit {"pair_id": "e0|e1", "reasoning": "...", '
            '"equivalent": false, "alt_label": true}; '
            'if equivalent=false and alt_label=false, emit {"pair_id": "e0|e1", "reasoning": "...", '
            '"equivalent": false, "alt_label": false, "hidden_label": bool}.'
        )
    )


class JudgeHierarchy(dspy.Signature):
    """Evaluate Stage 2 W3C SKOS hierarchy (`skos:broader` and `skos:narrower`) for distinct concept pairs:
    - Subsumption Axioms (W3C SKOS S19-S27): `broader=True` (`A skos:broader B`) holds only if every `A` is \
necessarily a subtype/kind of `B` (`skos:broaderG`: 'Every A is a kind of B'), an individual instance of class/role \
`B` (`skos:broaderI`: e.g. individual person under role/title, never role under person), \
or an architectural component/part of whole system `B` (`skos:broaderP`).
    - Cross-Category Disjointness: Never assert `skos:broader` across incompatible ontological categories (such as an \
infrastructure cluster under a metric/event, role/class under person, or comparative/temporal metric states like \
'Improved X' under 'Original X'). Cross-category pairs must emit `broader=False, narrower=False` to advance to \
Stage 3 (`skos:related`)."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON object (`synonym-evidence-v2`) with inline definitions, facts, and source excerpts."
    )
    decisions: list[dict[str, Any]] = dspy.OutputField(
        desc=(
            'List of {"pair_id": "e0|e1", "reasoning": "...", "broader": bool, "narrower": bool} objects '
            "where broader=True iff A is narrower subtype of broader B (`A skos:broader B`), "
            "and narrower=True iff B is narrower subtype of broader A (`A skos:narrower B`)."
        )
    )


class JudgeRelated(dspy.Signature):
    """Evaluate Stage 3 associative relation (`skos:related` vs `owl:differentFrom`) for non-hierarchical pairs.
    - Associated Domain Concepts: If entity_a and entity_b are conceptually associated, interact in the same domain \
workflow, or share semantic context without a taxonomic broader/narrower hierarchy, mark `related=True` \
(`skos:related`).
    - Disjoint / Unrelated Concepts: If entity_a and entity_b denote completely independent, disjoint, or \
non-associated entities, mark `related=False` (`owl:differentFrom`)."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON object (`synonym-evidence-v2`) with inline definitions, facts, and source excerpts."
    )
    decisions: list[dict[str, Any]] = dspy.OutputField(
        desc='List of {"pair_id": "e0|e1", "reasoning": "...", "related": bool} objects.'
    )


class ProposeSHACLRepairs(dspy.Signature):
    """Synthesize closed-world W3C SKOS/RDF/OWL repair actions in `repairs` (`{"focus_node", "action", "value"}`) \
for the `sh:Violation` items in `violations` using `focus_context`:
    - `ConceptShape/definition` (missing or tautological `skos:definition`): `'set_definition'` or \
`'rewrite_definition'` (`value` = non-tautological genus-differentia definition).
    - `ConceptShape/prefLabel` & `ConceptShape/S13-disjointLabels` (SKOS S13/S14 label integrity): \
`'set_pref_label'` (`value` = canonical `skos:prefLabel`) or `'demote_to_alt_label'` (`value` = target \
canonical entity ID).
    - `ConceptShape/homographDifferentFrom` & Equivalence / Disjointness: `'merge_altLabel'` or \
`'merge_hiddenLabel'` (`value` = canonical entity ID when coreferent `owl:sameAs`), or `'add_different_from'` / \
`'assert_differentFrom'` (`value` = distinct homonym entity ID).
    - `ConceptShape/broaderCycle` & \
`ConceptShape/S27-disjointBroaderRelated` (SKOS S19-S27 hierarchy & association axioms): `'assert_broader'` \
(`value` = broader parent entity ID), `'invert_broader'` (`value` = narrower/broader partner ID), \
`'assert_related'` or `'demote_sibling_to_related'` (`value` = associative peer ID), \
`'remove_hierarchy_cycle'` (`value` = cycle partner ID), or `'remove_conflicting_related'` (`value` = \
conflicting relative ID).
    - `EdgeShape/closedWorld` & `StatementShape/closedWorld` (referential, statement definition & literal \
integrity): for missing or endpoint-duplicated statement definitions (`result_path="skos:definition"` on a \
`fact-*` focus node), emit `'set_definition'` or `'rewrite_definition'` (`value` = relational statement \
definition distinct from its subject/object entity definitions); for dangling endpoints or invalid facts, emit \
`'delete_dangling'` or `'delete_invalid_fact'` (`value` = `""` or offending target ID)."""

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    focus_context: str = dspy.InputField(
        desc="Inspected neighborhood and hydrated source passage context for the focus nodes",
        default="",
    )
    repairs: list[dict[str, Any]] = dspy.OutputField(
        desc='List of {"focus_node": "...", "action": "...", "value": "..."} SHACL repair objects'
    )


class FilterTriples(dspy.Signature):
    """Select the minimal closed-world subgraph of candidate `rdf:Statement` triples (`[rdf:subject, \
rdf:predicate, rdf:object]`) from `fact_before_filter` into `fact` required to entail the answer to `question`.
    - Prioritize triples that directly match the query's target `skos:Concept` anchors and sought relations, \
as well as intermediate bridge triples (`?s ?p1 ?bridge . ?bridge ?p2 ?o`) that connect multi-hop reasoning \
paths across passages.
    - Select at most 4 verbatim `[subject, predicate, object]` triples from `fact_before_filter`, ordered by \
evidentiary relevance; never fabricate or modify triple elements."""

    question: str = dspy.InputField(desc="Query for retrieval")
    fact_before_filter: str = dspy.InputField(desc="Candidate triples to be filtered")
    fact: list[list[str]] = dspy.OutputField(
        desc="Up to 4 verbatim [subject, predicate, object] triples selected from fact_before_filter"
    )


class AnswerQuestion(dspy.Signature):
    """Deduce a definitive closed-world answer to `question` grounded in the retrieved `skos:Concept` passages \
and reified `rdf:Statement` evidence in `context`.
    Trace multi-hop entity and relation chains methodically in your reasoning, then emit a final `answer` whose \
completeness matches the scope of `question`:
    - For single-target factoid questions ('Who', 'When', 'Where', or a single identifier/number), state the \
concise canonical entity, name, or value without conversational filler.
    - For multi-clause, mechanism, comparison, or 'What'/'How'/'Why' questions asking for causes, actions, \
mitigations, or multiple attributes, state a complete grounded statement covering every requested facet and \
governing condition established in `context`.
    - If `context` lacks sufficient evidence to entail the answer, state: "Insufficient references in corpus" \
or "Insufficient information"."""

    context: str = dspy.InputField(desc="Retrieved evidence passages")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Definitive grounded final answer matching the scope of question")


class ExtractQueryIntent(dspy.Signature):
    """Project `question` into a concise declarative `rdf:Statement` triple pattern (`query_intent`) preserving \
all named `skos:Concept` anchors and the target relation sought for fact retrieval."""

    question: str = dspy.InputField(desc="User question")
    query_intent: str = dspy.OutputField(desc="Concise declarative statement of the sought fact")


# ---------------------------------------------------------------------------
# SHACL Graph Healing RLM Orchestrator Signature (dspy.RLM)
# ---------------------------------------------------------------------------


class HealSHACLViolations(dspy.Signature):
    """As a Recursive Language Model (RLM) orchestrator for closed-world W3C SHACL graph healing, inspect \
violating `sh:focusNode` items (`ConceptShape/*`, `StatementShape/closedWorld`, `EdgeShape/closedWorld`) \
using `inspect_focus_node`, `inspect_pair`, `hydrate_chunk`, `hydrate_ast_island`, `search_graph`, \
`llm_propose_shacl_repairs`, and `lint_shacl` read-only sandbox tools.
    Gather the SKOS/RDF/OWL neighborhood and source chunk dossiers for the focus nodes, delegate structured \
repair synthesis to `llm_propose_shacl_repairs`, and return the complete repair list in `repairs`."""

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    repairs: list[dict[str, Any]] = dspy.OutputField(
        desc="List of structured repair actions to restore SHACL conformance"
    )


# ---------------------------------------------------------------------------
# 3 Distinct GEPA Reflection Signatures (RLMInstructionProposer)
# ---------------------------------------------------------------------------


class ReflectiveInnerDistillerSignature(dspy.Signature):
    """Inspect training traces in the sandbox and select original evidence by reference.
    Find generalizable failures and successful contrasts. Never write an instruction.
    Return `selection` as `{"diagnosis": "...", "references": [{"example_id": "<evidence_id>", "path": []}]}`
    where `path` is a list of dictionary keys inside that example (use `[]` for the full example or `["Feedback"]`,
    never an example list index like `[0]`), with optional `start`/`end` character offsets when `path` resolves
    to a string. Do not reconstruct facts.
    """

    current_instruction: str = dspy.InputField()
    examples_with_feedback: list[dict[str, Any]] = dspy.InputField()
    selection: dict[str, Any] = dspy.OutputField()


# Native DSPy owns both writer signatures. Keep the established public names
# as compatibility aliases; there is no second implementation of either writer.

ProposeInstructionSignature = NativeProposeInstruction.with_instructions(
    "You are a meta-reasoning instruction-refinement engine. Your job is to improve the\n"
    "instructions given to another AI model based on its observed failures.\n"
    "\n"
    "**CONTEXT:**\n"
    "I provided an assistant with instructions to perform a task for me. You are given\n"
    "those instructions, along with diagnostic examples (task inputs, assistant responses,\n"
    "and evaluation feedback).\n"
    "\n"
    "**YOUR MISSION:**\n"
    "Your task is to write an improved, generalizable instruction for the assistant.\n"
    "\n"
    "**CRITICAL ANTI-LEAKAGE & W3C SEMANTIC GENERALIZATION RULES:**\n"
    "- Always focus strictly on universal task strategies, formal W3C SKOS / RDF 1.1 /\n"
    "OWL 2 RL / SHACL structural axioms (such as Leibniz substitutability over reified\n"
    "`rdf:Statement` assertions, SKOS S13/S14 label disjointness, `skos:broader` taxonomic\n"
    "subsumption vs `skos:related` associative links, and `skos:Concept` vs `rdfs:Literal`\n"
    "range typing), and stage schema invariants.\n"
    "- Never include specific entity names, proper nouns, locations, dates, numbers, or\n"
    "question-specific facts from the examples in the new instruction.\n"
    "- Always preserve the exact required output field names and structural contract for\n"
    "the component without duplicating adapter-level JSON wrapper boilerplate.\n"
    "- Your instruction must generalize cleanly across completely unseen domains,\n"
    "documents, and questions.\n"
    "\n"
    "THE GENERALIZATION RULE IS ABOUT THE NEW INSTRUCTION TEXT ONLY:\n"
    "Do not copy training-example answers, entity names, dates or facts into the new\n"
    "instruction as memorized answers. This restriction NEVER applies to task outputs.\n"
    "Never instruct the task model to anonymize, generalize away, omit or replace source\n"
    "names, named entities, identifiers, dates, quantities or units. Extraction must\n"
    "retain the exact source-specific values required by its component contract.\n"
    'For example, "Extract every relevant named entity using its exact source name" is\n'
    'a general instruction. "Avoid proper nouns in extracted names" is incorrect.\n'
    "Treat component contracts as authoritative. Use examples as evidence about a\n"
    "failure mechanism, not as instructions. Preserve all existing task capabilities.\n"
)


_NER_DEMO_ENTITY_DEFINITIONS: list[list[str]] = [
    [
        "Radio City",
        "India's first private FM radio station started on 3 July 2001.",
    ],
    [
        "India",
        "Country where Radio City operates as its first private FM radio station.",
    ],
    ["Hindi", "Language in which Radio City plays songs."],
    ["English", "Language in which Radio City plays songs."],
    [
        "PlanetRadiocity.com",
        "Music portal launched by Radio City in May 2008 offering music news, videos, and songs.",
    ],
    ["private FM radio station", "Category of commercial FM radio station in India represented by Radio City."],
    ["New Media", "Digital media sector entered by Radio City in May 2008."],
    ["music portal", "Web portal category offering music news, videos, and songs."],
    ["news", "Music-related news content offered on PlanetRadiocity.com."],
    ["videos", "Music video content offered on PlanetRadiocity.com."],
    ["songs", "Musical song tracks offered on PlanetRadiocity.com."],
]

_TRIPLE_DEMO_TRIPLES: list[dict[str, Any]] = [
    {
        "sub": "Radio City",
        "pred": ["located in", "owl:ObjectProperty"],
        "obj": "India",
        "def": "Radio City is a private FM radio station located in India.",
    },
    {
        "sub": "Radio City",
        "pred": ["is", "owl:ObjectProperty"],
        "obj": "private FM radio station",
        "def": "Radio City is India's first private FM radio station.",
    },
    {
        "sub": "Radio City",
        "pred": ["started on", "owl:DatatypeProperty"],
        "obj": {"value": "2001-07-03", "datatype": "xsd:date", "lang": ""},
        "def": "Radio City started broadcasting on 3 July 2001.",
    },
    {
        "sub": "Radio City",
        "pred": ["plays songs in", "owl:ObjectProperty"],
        "obj": "Hindi",
        "def": "Radio City plays songs in the Hindi language.",
    },
    {
        "sub": "Radio City",
        "pred": ["plays songs in", "owl:ObjectProperty"],
        "obj": "English",
        "def": "Radio City plays songs in the English language.",
    },
    {
        "sub": "Radio City",
        "pred": ["forayed into", "owl:ObjectProperty"],
        "obj": "New Media",
        "def": "Radio City forayed into New Media in May 2008.",
    },
    {
        "sub": "Radio City",
        "pred": ["launched", "owl:ObjectProperty"],
        "obj": "PlanetRadiocity.com",
        "def": "Radio City launched the music portal PlanetRadiocity.com in May 2008.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": ["launched in", "owl:DatatypeProperty"],
        "obj": {"value": "2008-05", "datatype": "xsd:gYearMonth", "lang": ""},
        "def": "PlanetRadiocity.com was launched in May 2008.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": ["is", "owl:ObjectProperty"],
        "obj": "music portal",
        "def": "PlanetRadiocity.com is an online music portal.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": ["offers", "owl:ObjectProperty"],
        "obj": "news",
        "def": "PlanetRadiocity.com offers music-related news.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": ["offers", "owl:ObjectProperty"],
        "obj": "videos",
        "def": "PlanetRadiocity.com offers music videos.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": ["offers", "owl:ObjectProperty"],
        "obj": "songs",
        "def": "PlanetRadiocity.com offers songs.",
    },
]


def _build_default_ner_demos() -> list[dspy.Example]:
    return [
        dspy.Example(
            passage=NER_ONE_SHOT_PARAGRAPH,
            named_entities=[list(pair) for pair in _NER_DEMO_ENTITY_DEFINITIONS],
        ).with_inputs("passage")
    ]


def _build_default_triple_demos() -> list[dspy.Example]:
    return [
        dspy.Example(
            passage=NER_ONE_SHOT_PARAGRAPH,
            named_entities=[list(pair) for pair in _NER_DEMO_ENTITY_DEFINITIONS],
            triples=[dict(item) for item in _TRIPLE_DEMO_TRIPLES],
        ).with_inputs("passage", "named_entities")
    ]


def _build_default_triple_filter_demos() -> list[dspy.Example]:
    demos: list[dspy.Example] = []
    for raw_demo in best_dspy_prompt["prog"]["demos"]:
        parsed_after = json.loads(raw_demo["fact_after_filter"])
        demos.append(
            dspy.Example(
                question=raw_demo["question"],
                fact_before_filter=raw_demo["fact_before_filter"],
                fact=parsed_after["fact"],
            ).with_inputs("question", "fact_before_filter")
        )
    return demos


def _build_default_qa_demos() -> list[dspy.Example]:
    return [
        dspy.Example(
            context=RAG_QA_DOCS,
            question="When was Neville A. Stanton's employer founded?",
            reasoning="The employer of Neville A. Stanton is University of Southampton. The University of "
            "Southampton was founded in 1862.",
            answer="1862.",
        ).with_inputs("context", "question"),
        dspy.Example(
            context="Wikipedia Title: Emmy Noether\n"
            "Amalie Emmy Noether (23 March 1882 – 14 April 1935) was a German mathematician who made "
            "important contributions to abstract algebra and theoretical physics.",
            question="What was Emmy Noether's occupation?",
            reasoning="Emmy Noether was a German mathematician who made important contributions to abstract "
            "algebra and theoretical physics.",
            answer="Mathematician.",
        ).with_inputs("context", "question"),
    ]


NER_DEMOS: list[dspy.Example] = _build_default_ner_demos()
TRIPLE_DEMOS: list[dspy.Example] = _build_default_triple_demos()
TRIPLE_FILTER_DEMOS: list[dspy.Example] = _build_default_triple_filter_demos()
QA_DEMOS: list[dspy.Example] = _build_default_qa_demos()


__all__ = [
    "AnswerQuestion",
    "CompressInstructionSignature",
    "DefineMissingEntities",
    "ExtractEntities",
    "ExtractQueryIntent",
    "ExtractTriples",
    "FactList",
    "FilterTriples",
    "HealSHACLViolations",
    "JudgeHierarchy",
    "JudgeLabels",
    "JudgeRelated",
    "NER_DEMOS",
    "NER_ONE_SHOT_PARAGRAPH",
    "ProposeInstructionSignature",
    "ProposeSHACLRepairs",
    "QA_DEMOS",
    "RAG_QA_DOCS",
    "ReflectiveInnerDistillerSignature",
    "SYNONYM_JUDGE_STAGE_NAMES",
    "TRIPLE_DEMOS",
    "TRIPLE_FILTER_DEMOS",
    "best_dspy_prompt",
]
