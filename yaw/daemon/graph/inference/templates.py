"""Upstream HippoRAG 2 prompt strings, few-shot frames, and chat message formatters."""

import json
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# 1. Named Entity Recognition (NER) Prompts
# ---------------------------------------------------------------------------

ner_system = (
    "Your task is to extract named entities from the given paragraph. \n"
    "Respond with a JSON list of entities.\n"
)

one_shot_ner_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_ner_output = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

NER_SYSTEM = ner_system
NER_ONE_SHOT_PARAGRAPH = one_shot_ner_paragraph
NER_ONE_SHOT_OUTPUT = one_shot_ner_output


def render_ner_messages(passage: str) -> list[dict[str, str]]:
    """Render 4-message chat history for NER extraction."""
    return [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": NER_ONE_SHOT_PARAGRAPH},
        {"role": "assistant", "content": NER_ONE_SHOT_OUTPUT},
        {"role": "user", "content": passage},
    ]


# ---------------------------------------------------------------------------
# 2. Relation Triple Extraction Prompts
# ---------------------------------------------------------------------------

ner_conditioned_re_system = (
    "Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. \n"
    "Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. \n\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n\n"
)

ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output
)

ner_conditioned_re_output = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

TRIPLE_SYSTEM = ner_conditioned_re_system
TRIPLE_FRAME = ner_conditioned_re_frame
TRIPLE_ONE_SHOT_INPUT = ner_conditioned_re_input
TRIPLE_ONE_SHOT_OUTPUT = ner_conditioned_re_output


def render_triple_messages(
    passage: str, named_entities: list[str] | tuple[str, ...] | Sequence[str]
) -> list[dict[str, str]]:
    """Render 4-message chat history for triple extraction."""
    named_entity_json = json.dumps({"named_entities": list(named_entities)})
    user_content = TRIPLE_FRAME.format(passage=passage, named_entity_json=named_entity_json)
    return [
        {"role": "system", "content": TRIPLE_SYSTEM},
        {"role": "user", "content": TRIPLE_ONE_SHOT_INPUT},
        {"role": "assistant", "content": TRIPLE_ONE_SHOT_OUTPUT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# 3. Multi-Hop RAG QA Prompts
# ---------------------------------------------------------------------------

one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $94 million Box office $940.3 million"""
)

rag_qa_system = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. "
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    "\nThought: "
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
    "\nAnswer: 1862."
)

RAG_QA_SYSTEM = rag_qa_system
RAG_QA_DOCS = one_shot_rag_qa_docs
RAG_QA_ONE_SHOT_INPUT = one_shot_rag_qa_input
RAG_QA_ONE_SHOT_OUTPUT = one_shot_rag_qa_output


def render_rag_qa_messages(prompt_user: str) -> list[dict[str, str]]:
    """Render 4-message chat history for RAG QA."""
    return [
        {"role": "system", "content": RAG_QA_SYSTEM},
        {"role": "user", "content": RAG_QA_ONE_SHOT_INPUT},
        {"role": "assistant", "content": RAG_QA_ONE_SHOT_OUTPUT},
        {"role": "user", "content": prompt_user},
    ]


# ---------------------------------------------------------------------------
# 4. Recognition Memory / Fact Rerank (DSPyFilter) Prompts
# ---------------------------------------------------------------------------

best_dspy_prompt: dict[str, Any] = {
    "prog": {
        "lm": None,
        "traces": [],
        "train": [],
        "demos": [
            {
                "augmented": True,
                "question": "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
                "fact_before_filter": '{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}',
                "fact_after_filter": '{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united states"],["amaradia","flows through","ro ia de amaradia"]]}',
            },
            {
                "augmented": True,
                "question": "When is the director of film The Ancestor 's birthday?",
                "fact_before_filter": '{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}',
                "fact_after_filter": '{"fact":[["the ancestor","directed by","guido brignone"]]}',
            },
            {
                "augmented": True,
                "question": "In what geographic region is the country where Teafuone is located?",
                "fact_before_filter": '{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], ["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located in", "nukufetau"]]}',
                "fact_after_filter": '{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}',
            },
            {
                "augmented": True,
                "question": "When did the director of film S.O.B. (Film) die?",
                "fact_before_filter": '{"fact": [["allan dwan", "died on", "28 december 1981"], ["s o b", "written and directed by", "blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert siodmak", "died on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november 2018"]]}',
                "fact_after_filter": '{"fact":[["s o b","written and directed by","blake edwards"]]}',
            },
            {
                "augmented": True,
                "question": "Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same country?",
                "fact_before_filter": '{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], ["gloria", "is", "1980 american thriller crime drama film"], ["a brand new life", "is directed by", "ounie lecomte"], ["gloria", "written and directed by", "john cassavetes"], ["a new life", "directed by", "alan alda"]]}',
                "fact_after_filter": '{"fact":[["gloria","is","1980 american thriller crime drama film"],["gloria","written and directed by","john cassavetes"],["a new life","directed by","alan alda"]]}',
            },
            {
                "augmented": True,
                "question": "What is the date of death of the director of film The Old Guard (1960 Film)?",
                "fact_before_filter": '{"fact": [["the old guard", "is", "1960 french comedy film"], ["gilles grangier", "directed", "the old guard"], ["the old guard", "directed by", "gilles grangier"], ["the old fritz", "directed by", "gerhard lamprecht"], ["oswald albert mitchell", "directed", "old mother riley series of films"]]}',
                "fact_after_filter": '{"fact":[["the old guard","is","1960 french comedy film"],["gilles grangier","directed","the old guard"],["the old guard","directed by","gilles grangier"]]}',
            },
            {
                "augmented": True,
                "question": "When is the composer of film Aulad (1968 Film) 's birthday?",
                "fact_before_filter": '{"fact": [["aulad", "has music composed by", "chitragupta shrivastava"], ["aadmi sadak ka", "has music by", "ravi"], ["ravi shankar sharma", "composed music for", "hindi films"], ["gulzar", "was born on", "18 august 1934"], ["aulad", "is a", "1968 hindi language drama film"]]}',
                "fact_after_filter": '{"fact":[["aulad","has music composed by","chitragupta shrivastava"],["aulad","is a","1968 hindi language drama film"]]}',
            },
            {
                "question": "How many households were in the city where Angelical Tears located?",
                "fact_before_filter": '{"fact": [["dow city", "had", "219 households"], ["tucson", "had", "229 762 households"], ["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", "oklahoma city"], ["atlantic city", "had", "15 848 households"]]}',
                "fact_after_filter": '{"fact": [["angelical tears", "located in", "oklahoma city"]]}',
            },
            {
                "question": "Did the movies In The Pope'S Eye and Virgin Mountain, originate from the same country?",
                "fact_before_filter": '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}',
                "fact_after_filter": '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", "directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 icelandic drama film"]]}',
            },
            {
                "question": "Which film has the director who died earlier, The Virtuous Model or Bulldog Drummond'S Peril?",
                "fact_before_filter": '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s revenge", "directed by", "louis king"], ["bulldog drummond s peril", "is", "american film"]]}',
                "fact_after_filter": '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond s peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert capellani"], ["bulldog drummond s peril", "is", "american film"]]}',
            },
        ],
        "signature": {
            "instructions": 'You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.',
            "fields": [
                {"prefix": "Question:", "description": "Query for retrieval"},
                {"prefix": "Fact Before Filter:", "description": "Candidate facts to be filtered"},
                {"prefix": "Fact After Filter:", "description": "Filtered facts in JSON format"},
            ],
        },
        "system": 'Your input fields are:\n1. `question` (str): Query for retrieval\n2. `fact_before_filter` (str): Candidate facts to be filtered\n\nYour output fields are:\n1. `fact_after_filter` (Fact): Filtered facts in JSON format\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n[[ ## fact_after_filter ## ]]\n{fact_after_filter}        # note: the value you produce must be pareseable according to the following JSON schema: {"type": "object", "properties": {"fact": {"type": "array", "description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]", "items": {"type": "array", "items": {"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.',
    }
}

FACT_FILTER_SYSTEM = best_dspy_prompt["prog"]["system"]
FACT_FILTER_INPUT_TEMPLATE = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."""
FACT_FILTER_OUTPUT_TEMPLATE = (
    """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
)


def _build_fact_filter_base_messages() -> list[dict[str, str]]:
    messages = [{"role": "system", "content": FACT_FILTER_SYSTEM}]
    for demo in best_dspy_prompt["prog"]["demos"]:
        messages.append(
            {
                "role": "user",
                "content": FACT_FILTER_INPUT_TEMPLATE.format(
                    question=demo["question"],
                    fact_before_filter=demo["fact_before_filter"],
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": FACT_FILTER_OUTPUT_TEMPLATE.format(
                    fact_after_filter=demo["fact_after_filter"]
                ),
            }
        )
    return messages


FACT_FILTER_BASE_MESSAGES = _build_fact_filter_base_messages()


def render_fact_filter_messages(
    question: str,
    candidate_facts: Sequence[tuple[str, str, str]] | Sequence[list[str]],
) -> list[dict[str, str]]:
    """Render full chat messages for DSPy recognition memory fact reranking."""
    fact_before_filter = json.dumps({"fact": [list(f) for f in candidate_facts]})
    user_content = FACT_FILTER_INPUT_TEMPLATE.format(
        question=question,
        fact_before_filter=fact_before_filter,
    )
    messages = [dict(m) for m in FACT_FILTER_BASE_MESSAGES]
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# 5. Fixed JSON Schemas for Local LLM Constrained Decoding
# ---------------------------------------------------------------------------

PROMPT_JSON_TEMPLATE: dict[str, Any] = {
    "ner": {
        "type": "object",
        "properties": {
            "named_entities": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            }
        },
        "required": ["named_entities"],
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0,
            }
        },
        "required": ["triples"],
    },
    "fact": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "description": "A list of facts, each fact is a list of 3 strings: [subject, predicate, object]",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "title": "Fact",
            }
        },
        "required": ["fact"],
        "title": "Fact",
    },
}

__all__ = [
    "ner_system",
    "one_shot_ner_paragraph",
    "one_shot_ner_output",
    "NER_SYSTEM",
    "NER_ONE_SHOT_PARAGRAPH",
    "NER_ONE_SHOT_OUTPUT",
    "render_ner_messages",
    "ner_conditioned_re_system",
    "ner_conditioned_re_frame",
    "ner_conditioned_re_input",
    "ner_conditioned_re_output",
    "TRIPLE_SYSTEM",
    "TRIPLE_FRAME",
    "TRIPLE_ONE_SHOT_INPUT",
    "TRIPLE_ONE_SHOT_OUTPUT",
    "render_triple_messages",
    "one_shot_rag_qa_docs",
    "rag_qa_system",
    "one_shot_rag_qa_input",
    "one_shot_rag_qa_output",
    "RAG_QA_SYSTEM",
    "RAG_QA_DOCS",
    "RAG_QA_ONE_SHOT_INPUT",
    "RAG_QA_ONE_SHOT_OUTPUT",
    "render_rag_qa_messages",
    "best_dspy_prompt",
    "FACT_FILTER_SYSTEM",
    "FACT_FILTER_INPUT_TEMPLATE",
    "FACT_FILTER_OUTPUT_TEMPLATE",
    "FACT_FILTER_BASE_MESSAGES",
    "render_fact_filter_messages",
    "PROMPT_JSON_TEMPLATE",
]
