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
    """You are a critical component in a high-stakes SKOS/RDF/OWL based ontology graph system.
    Your task is to extract and named entities as `skos:Concept`, and their concise contextual \
definitions as `skos:definition` from the given paragraph.
    Respond with a JSON object in the format: {"named_entities": [["entity1", "concise contextual \
definition of entity1"], ["entity2", "concise contextual definition of entity2"], ...]}"""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.OutputField(
        desc="Deduplicated list of extracted [entity_name, contextual_definition] pairs"
    )


class ExtractTriples(dspy.Signature):
    """Your are a critical component in a high-stakes SKOS/RDF/OWL based ontology graph system.
    Your task is to construct an RDF (Resource Description Framework) `rdf:Statement` graph from the \
given passages and named entity lists.
    Respond with a JSON object in the format:
    {"triples": [{"sub": "known_entity" or ["new_entity", "contextual definition"], "pred": "predicate", \
"obj": "known_entity" or ["new_entity", "contextual definition"], "def": "concise contextual definition \
of the statement fact"}, ...]}
    Pay attention to the following requirements:
    - Each triple should contain at least one, but preferably two, of the named entities in the list for \
each passage.
    - Clearly resolve pronouns to their specific names to maintain clarity.
    - If a "sub" or "obj" entity is already in named_entities with a definition, you must emit its exact \
string name (e.g., "Cinderella") without repeating its definition.
    - If a "sub" or "obj" entity is a new entity not present in named_entities, you must emit a 2-element \
list [entity_name, contextual_definition] (e.g., ["glass slipper", "Delicate footwear lost by Cinderella \
at the ball"]), and always provide a concise statement definition in "def"."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.InputField(
        desc="Extracted [entity_name, contextual_definition] pairs (or entity names) from Stage 1"
    )
    triples: list[Any] = dspy.OutputField(
        desc='List of relation triple objects {"sub", "pred", "obj", "def"} where known entities are '
        "strings and new entities are [name, definition] pairs"
    )


class DefineMissingEntities(dspy.Signature):
    """You are a critical component in a high-stakes SKOS/RDF/OWL based ontology graph system.
    Your task is to provide concise contextual definitions grounded in the passage and extracted triples \
for any triple entities that were emitted without definitions.
    Respond with a JSON object in the format:
    {"entity_definitions": [["entity_name", "concise contextual definition grounded in the passage and \
triples"], ...]}"""

    passage: str = dspy.InputField(desc="Source passage text")
    missing_entities: list[str] = dspy.InputField(
        desc="Entity names referenced in triples that lack contextual definitions"
    )
    triples: list[Any] = dspy.InputField(
        desc="Extracted relation triples referencing the missing entities",
        default=[],
    )
    entity_definitions: list[list[str]] = dspy.OutputField(
        desc="List of [entity_name, contextual_definition] pairs for the missing entities"
    )


class JudgeSynonyms(dspy.Signature):
    """You are a critical component in a high-stakes SKOS/RDF/OWL based ontology graph system. Your task is to \
disambiguate entities.
    Judge candidate pairs of entities that look similar semantically and in writing/spelling (including \
typos, acronyms, lookalikes like "Cindy Real" vs "Cinderella", or homonyms/homographs with the same name \
like "a ball" vs "a ball"):
    - 'skos:altLabel': True synonym or coreferent alias referring to the exact same entity/concept (merge \
into canonical entity).
    - 'skos:hiddenLabel': Typo, misspelling, transcription error, acronym, or surface spelling variant \
referring to the exact same entity/concept (merge into canonical entity).
    - 'owl:differentFrom': Confusable homonym, homograph (e.g., "a ball" [formal dance] vs "a ball" \
[spherical toy]), or false-friend lookalike (e.g., "Cindy Real" vs "Cinderella") that is easy to mix up \
due to similar/identical surface form or context, but actually refers to a genuinely different \
entity/concept (keep separate, add repulsive owl:differentFrom edge).
    - 'skos:related': Distinct, different `skos:concept` entities that are associatively related in the \
domain without being coreferent synonyms or confusable homonyms/lookalikes (keep separate, add symmetric \
skos:related edge).
    Respond with JSON: {"decisions": [{"pair_id": "id_a|id_b", "relation": \
"skos:altLabel|skos:hiddenLabel|skos:related|owl:differentFrom", "canonical": "id_a_or_id_b"}, ...]}"""

    candidate_pairs: str = dspy.InputField(
        desc="JSON list of candidate entity pairs with names, definitions, and similarity scores"
    )
    decisions: list[dict[str, str]] = dspy.OutputField(
        desc="List of SKOS/OWL disambiguation decisions per candidate pair"
    )


class ProposeSHACLRepairs(dspy.Signature):
    """You are a critical component in a high-stakes SKOS/RDF/OWL based ontology graph system. Your task is to \
heal SHACL violations.
    Propose closed-world SKOS/RDF/OWL repairs for SHACL validation report violations.
    Supported actions: 'set_definition', 'rewrite_definition', 'set_pref_label', 'demote_to_alt_label', \
'merge_altLabel', 'merge_hiddenLabel', 'add_different_from', 'assert_differentFrom', 'assert_related', \
'delete_dangling', 'delete_invalid_fact'.
    Respond with JSON: {"repairs": [{"focus_node": "...", "action": "...", "value": "..."}, ...]}"""

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    focus_context: str = dspy.InputField(
        desc="Inspected neighborhood and hydrated source passage context for the focus nodes",
        default="",
    )
    repairs: list[dict[str, Any]] = dspy.OutputField(
        desc="List of structured repair actions to restore SHACL conformance"
    )


class FilterTriples(dspy.Signature):
    """You are a critical component of a high-stakes question-answering system used by top researchers and \
decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, \
ensuring that the most crucial information is presented to these stakeholders.
    The facts are presented to you in the form of `rdf:Statement` triples, they assert a text corpus \
bound fact.
    The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of \
information. You must select up to 4 relevant facts from the provided candidate list that have a strong \
connection to the query, aiding in reasoning and providing an accurate answer. The output should be in \
JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, \
return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly \
impact the decisions made by these high-level stakeholders. You must only use facts from the candidate \
list and not generate new facts. The future of critical decision-making relies on your ability to \
accurately filter and present relevant information."""

    question: str = dspy.InputField(desc="Query for retrieval")
    fact_before_filter: str = dspy.InputField(desc="Candidate triples to be filtered")
    fact: list[list[str]] = dspy.OutputField(
        desc="Selected relevant [subject, predicate, object] triples from the candidate list"
    )


class AnswerQuestion(dspy.Signature):
    """As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding \
questions meticulously.
    Break down the reasoning process methodically, illustrating how you arrive at conclusions, and \
present a concise, definitive final answer devoid of additional elaborations.
    - For 'Who', 'What', 'When', or 'Where' questions, state only the exact bare entity, name, or number \
without conversational phrases or articles.
    - For 'How' or 'Why' questions, state both the key condition/means and the resulting outcome from the \
passages.
    - If the provided context lacks sufficient information to answer the question, state: "Insufficient \
references in corpus", "Insufficient information", or similar."""

    context: str = dspy.InputField(desc="Retrieved evidence passages")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Concise, definitive final answer (bare entity or value)")


class ExtractQueryIntent(dspy.Signature):
    """Rewrite a question as a concise declarative statement or target relation phrase for fact retrieval."""

    question: str = dspy.InputField(desc="User question")
    query_intent: str = dspy.OutputField(desc="Concise declarative statement of the sought fact")


# ---------------------------------------------------------------------------
# SHACL Graph Healing RLM Orchestrator Signature (dspy.RLM)
# ---------------------------------------------------------------------------


class HealSHACLViolations(dspy.Signature):
    """As a Recursive RLM orchestrator for closed-world SHACL graph repair,
    you orchestrate the healing of SHACL violations using inspect_node and hydrate_chunk sandbox tools."""

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
    Return a diagnosis and up to five references: example_id, path (list of keys/indices),
    optional start/end character offsets for a string excerpt. Do not reconstruct facts.
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
    "**CRITICAL ANTI-LEAKAGE & GENERALIZATION RULES:**\n"
    "- Allways focus strictly on universal task strategies, structural extraction/reasoning\n"
    "rules, and schema invariants.\n"
    "- Never include specific entity names, proper nouns, locations, dates, numbers, or\n"
    "question-specific facts from the examples in the new instruction.\n"
    "- Allways preserve the exact required JSON output schema and structural contract for\n"
    "the component.\n"
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
    ["3 July 2001", "Date when Radio City was started."],
    ["Hindi", "Language in which Radio City plays songs."],
    ["English", "Language in which Radio City plays songs."],
    ["May 2008", "Month and year when Radio City launched PlanetRadiocity.com."],
    [
        "PlanetRadiocity.com",
        "Music portal launched by Radio City in May 2008 offering music news, videos, and songs.",
    ],
]

_TRIPLE_DEMO_TRIPLES: list[dict[str, Any]] = [
    {
        "sub": "Radio City",
        "pred": "located in",
        "obj": "India",
        "def": "Radio City is a private FM radio station located in India.",
    },
    {
        "sub": "Radio City",
        "pred": "is",
        "obj": [
            "private FM radio station",
            "Category of commercial FM radio station in India represented by Radio City.",
        ],
        "def": "Radio City is India's first private FM radio station.",
    },
    {
        "sub": "Radio City",
        "pred": "started on",
        "obj": "3 July 2001",
        "def": "Radio City started broadcasting on 3 July 2001.",
    },
    {
        "sub": "Radio City",
        "pred": "plays songs in",
        "obj": "Hindi",
        "def": "Radio City plays songs in the Hindi language.",
    },
    {
        "sub": "Radio City",
        "pred": "plays songs in",
        "obj": "English",
        "def": "Radio City plays songs in the English language.",
    },
    {
        "sub": "Radio City",
        "pred": "forayed into",
        "obj": [
            "New Media",
            "Digital media sector entered by Radio City in May 2008.",
        ],
        "def": "Radio City forayed into New Media in May 2008.",
    },
    {
        "sub": "Radio City",
        "pred": "launched",
        "obj": "PlanetRadiocity.com",
        "def": "Radio City launched the music portal PlanetRadiocity.com in May 2008.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": "launched in",
        "obj": "May 2008",
        "def": "PlanetRadiocity.com was launched in May 2008.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": "is",
        "obj": [
            "music portal",
            "Web portal category offering music news, videos, and songs.",
        ],
        "def": "PlanetRadiocity.com is an online music portal.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": "offers",
        "obj": [
            "news",
            "Music-related news content offered on PlanetRadiocity.com.",
        ],
        "def": "PlanetRadiocity.com offers music-related news.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": "offers",
        "obj": [
            "videos",
            "Music video content offered on PlanetRadiocity.com.",
        ],
        "def": "PlanetRadiocity.com offers music videos.",
    },
    {
        "sub": "PlanetRadiocity.com",
        "pred": "offers",
        "obj": [
            "songs",
            "Musical song tracks offered on PlanetRadiocity.com.",
        ],
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
    "JudgeSynonyms",
    "NER_DEMOS",
    "NER_ONE_SHOT_PARAGRAPH",
    "ProposeInstructionSignature",
    "ProposeSHACLRepairs",
    "QA_DEMOS",
    "RAG_QA_DOCS",
    "ReflectiveInnerDistillerSignature",
    "TRIPLE_DEMOS",
    "TRIPLE_FILTER_DEMOS",
    "best_dspy_prompt",
]
