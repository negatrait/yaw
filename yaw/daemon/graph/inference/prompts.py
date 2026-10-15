"""Owned DSPy/GEPA-optimizable prompts, dspy.Signatures, RLM/CoT modules, and stateless parsers for HippoRAG 2."""

import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, Union

import dspy
from pydantic import BaseModel, Field

from yaw.core.types import OpenIERecord

# ---------------------------------------------------------------------------
# 1. Named Entity Recognition (NER) Prompts
# ---------------------------------------------------------------------------

ner_system = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

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


def render_ner_messages(passage: str) -> List[Dict[str, str]]:
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

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

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
    passage: str, named_entities: Union[List[str], Tuple[str, ...], Sequence[str]]
) -> List[Dict[str, str]]:
    """Render 4-message chat history for triple extraction."""
    named_entity_json = json.dumps({"named_entities": list(named_entities)})
    user_content = TRIPLE_FRAME.format(
        passage=passage, named_entity_json=named_entity_json
    )
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


def render_rag_qa_messages(prompt_user: str) -> List[Dict[str, str]]:
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

best_dspy_prompt: Dict[str, Any] = {
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
            "instructions": "You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {\"fact\": [[\"s1\", \"p1\", \"o1\"], [\"s2\", \"p2\", \"o2\"]]}, and if no facts are relevant, return an empty list, {\"fact\": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information.",
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
FACT_FILTER_OUTPUT_TEMPLATE = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""


def _build_fact_filter_base_messages() -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": FACT_FILTER_SYSTEM}]
    for demo in best_dspy_prompt["prog"]["demos"]:
        messages.append({
            "role": "user",
            "content": FACT_FILTER_INPUT_TEMPLATE.format(
                question=demo["question"],
                fact_before_filter=demo["fact_before_filter"],
            ),
        })
        messages.append({
            "role": "assistant",
            "content": FACT_FILTER_OUTPUT_TEMPLATE.format(
                fact_after_filter=demo["fact_after_filter"]
            ),
        })
    return messages


FACT_FILTER_BASE_MESSAGES = _build_fact_filter_base_messages()


def render_fact_filter_messages(
    question: str,
    candidate_facts: Union[Sequence[Tuple[str, str, str]], Sequence[List[str]]],
) -> List[Dict[str, str]]:
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

PROMPT_JSON_TEMPLATE: Dict[str, Any] = {
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


# ---------------------------------------------------------------------------
# 6. Stateless Parsers & String Normalizers
# ---------------------------------------------------------------------------

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute the MD5 hash of the given content string and optionally prepend a prefix."""
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def text_processing(text: Union[str, List[Any]]) -> Union[str, List[Any]]:
    """Entity and alias normalization: lowercase, alphanumeric and whitespace only."""
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    normalized = "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in text.casefold()
    )
    return " ".join(normalized.split())


def filter_invalid_triples(triples: Sequence[Any]) -> List[List[str]]:
    """Filter out invalid and duplicate triples, preserving original order."""
    unique_triples: set[Tuple[str, str, str]] = set()
    valid_triples: List[List[str]] = []

    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        valid_triple = [str(item) for item in triple]
        key = (valid_triple[0], valid_triple[1], valid_triple[2])
        if key not in unique_triples:
            unique_triples.add(key)
            valid_triples.append(valid_triple)

    return valid_triples


def repair_truncated_json(response: str, *, is_triples: bool = False) -> str:
    """String-aware bracket depth scanner to repair JSON responses truncated by max tokens."""
    try:
        json.loads(response)
        return response
    except (ValueError, TypeError):
        pass

    depth, in_str, esc, last_end = 0, False, False, -1
    for i, ch in enumerate(response):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if not is_triples and depth == 2:
                    last_end = i + 1
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            if is_triples and depth == 3 and ch == "]":
                last_end = i + 1
            depth -= 1

    if last_end == -1:
        return '{"triples": []}' if is_triples else '{"named_entities": []}'
    return response[:last_end] + "]}"


def _extract_json_list_field(response: str, field_name: str) -> List[Any]:
    """Scan a response string for the first JSON object containing `field_name` as a list."""
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, list):
            raise ValueError(f"OpenIE response field {field_name!r} must be a list.")
        return value
    raise ValueError(f"OpenIE response does not contain a valid JSON object with {field_name!r}.")


def _parse_bare_json_string_list(response: str) -> Union[List[str], None]:
    """Return the response as a list of strings if it is exactly one JSON array of strings, else None."""
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return None
    return payload


def extract_named_entities(response: str) -> Tuple[str, ...]:
    """Extract and deduplicate unique named entities from LLM response."""
    # Attempt direct or bare JSON extraction
    try:
        raw_list = _extract_json_list_field(response, "named_entities")
    except ValueError:
        bare = _parse_bare_json_string_list(response)
        if bare is not None:
            raw_list = bare
        else:
            # Try repairing truncation
            repaired = repair_truncated_json(response, is_triples=False)
            try:
                raw_list = _extract_json_list_field(repaired, "named_entities")
            except ValueError:
                bare_repaired = _parse_bare_json_string_list(repaired)
                if bare_repaired is not None:
                    raw_list = bare_repaired
                else:
                    return ()

    unique: List[str] = []
    seen: set[str] = set()
    for item in raw_list:
        if isinstance(item, str):
            clean = item.strip()
            if clean and clean not in seen:
                seen.add(clean)
                unique.append(clean)
    return tuple(unique)


def extract_triples(response: str) -> Tuple[Tuple[str, str, str], ...]:
    """Extract, validate, and deduplicate relation triples from LLM response."""
    try:
        raw_list = _extract_json_list_field(response, "triples")
    except ValueError:
        repaired = repair_truncated_json(response, is_triples=True)
        try:
            raw_list = _extract_json_list_field(repaired, "triples")
        except ValueError:
            return ()

    filtered = filter_invalid_triples(raw_list)
    return tuple((t[0], t[1], t[2]) for t in filtered)


def parse_filter_facts(response: str) -> List[List[str]]:
    """Parse candidate facts from DSPy recognition memory response."""
    sections: List[Tuple[Union[str, None], List[str]]] = [(None, [])]
    field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")
    for line in response.splitlines():
        match = field_header_pattern.match(line.strip())
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)

    parsed: List[List[str]] = []
    found_section = False
    for k, lines in sections:
        if k == "fact_after_filter":
            found_section = True
            value = "\n".join(lines).strip()
            parsed_value = None
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed_value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    parsed_value = value

            if isinstance(parsed_value, dict) and "fact" in parsed_value:
                raw_facts = parsed_value["fact"]
                if isinstance(raw_facts, list):
                    for item in raw_facts:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            parsed.append([str(x) for x in item])
            elif isinstance(parsed_value, list):
                for item in parsed_value:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        parsed.append([str(x) for x in item])

    # Fallback if model answered with raw JSON without DSPy marker
    if not found_section:
        text = response.strip()
        try:
            val = json.loads(text)
            if isinstance(val, dict) and "fact" in val and isinstance(val["fact"], list):
                for item in val["fact"]:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        parsed.append([str(x) for x in item])
        except Exception:
            pass

    return parsed


def filter_reranked_facts(
    candidate_facts: Sequence[Union[Tuple[str, str, str], List[str]]],
    response: str,
    max_facts: Union[int, None] = None,
) -> List[int]:
    """Map generated facts back to original candidate fact indices via difflib."""
    generated_facts = parse_filter_facts(response)
    if not generated_facts:
        return []

    candidate_str_list = [
        str(list(f) if isinstance(f, tuple) else f) for f in candidate_facts
    ]
    candidate_list_of_lists = [
        list(f) if isinstance(f, tuple) else f for f in candidate_facts
    ]

    result_indices: List[int] = []
    for generated_fact in generated_facts:
        gen_str = str(list(generated_fact) if isinstance(generated_fact, tuple) else generated_fact)
        matches = difflib.get_close_matches(
            gen_str, candidate_str_list, n=1, cutoff=0.0
        )
        if matches:
            closest_matched_fact = matches[0]
            try:
                idx = candidate_list_of_lists.index(ast.literal_eval(closest_matched_fact))
                if idx not in result_indices:
                    result_indices.append(idx)
            except Exception:
                pass

    if max_facts is not None:
        return result_indices[:max_facts]
    return result_indices


# ---------------------------------------------------------------------------
# 7. Native DSPy Signatures, Few-Shot Demos, and HippoRAGProgram (Predict, CoT, RLM)
# ---------------------------------------------------------------------------


class FactList(BaseModel):
    """Typed Pydantic container for recognition-memory filtered facts."""

    fact: list[list[str]] = Field(
        default_factory=list,
        description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]",
    )


class ExtractEntities(dspy.Signature):
    """Your task is to extract named entities from the given paragraph.
    Respond with a JSON list of entities."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[str] = dspy.OutputField(
        desc="Deduplicated list of extracted named entities"
    )


class ExtractTriples(dspy.Signature):
    """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
    Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
    Pay attention to the following requirements:
    - Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
    - Clearly resolve pronouns to their specific names to maintain clarity."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[str] = dspy.InputField(
        desc="Named entity list extracted from the passage"
    )
    triples: list[list[str]] = dspy.OutputField(
        desc="List of [subject, predicate, object] relation triples"
    )


class FilterFacts(dspy.Signature):
    """You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information."""

    question: str = dspy.InputField(desc="Query for retrieval")
    fact_before_filter: str = dspy.InputField(desc="Candidate facts to be filtered")
    fact: list[list[str]] = dspy.OutputField(
        desc="Selected relevant [subject, predicate, object] facts from the candidate list"
    )


class AnswerQuestion(dspy.Signature):
    """As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Break down the reasoning process methodically, illustrating how you arrive at conclusions, and present a concise, definitive final answer devoid of additional elaborations. For 'Who', 'What', 'When', or 'Where' questions, state only the exact entity or fact. For 'How' or 'Why' questions, state both the key condition/means and the resulting outcome from the passages."""

    context: str = dspy.InputField(desc="Retrieved evidence passages")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Concise, definitive final answer")


def _build_default_ner_demos() -> list[dspy.Example]:
    entities = json.loads(NER_ONE_SHOT_OUTPUT)["named_entities"]
    return [
        dspy.Example(
            passage=NER_ONE_SHOT_PARAGRAPH,
            named_entities=entities,
        ).with_inputs("passage")
    ]


def _build_default_triple_demos() -> list[dspy.Example]:
    entities = json.loads(NER_ONE_SHOT_OUTPUT)["named_entities"]
    triples = json.loads(TRIPLE_ONE_SHOT_OUTPUT)["triples"]
    return [
        dspy.Example(
            passage=NER_ONE_SHOT_PARAGRAPH,
            named_entities=entities,
            triples=triples,
        ).with_inputs("passage", "named_entities")
    ]


def _build_default_fact_filter_demos() -> list[dspy.Example]:
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
            reasoning="The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862.",
            answer="1862.",
        ).with_inputs("context", "question")
    ]


NER_DEMOS: list[dspy.Example] = _build_default_ner_demos()
TRIPLE_DEMOS: list[dspy.Example] = _build_default_triple_demos()
FACT_FILTER_DEMOS: list[dspy.Example] = _build_default_fact_filter_demos()
QA_DEMOS: list[dspy.Example] = _build_default_qa_demos()


def _coerce_generated_facts(raw_output: Any) -> list[list[str]]:
    """Extract 3-element string lists from a FactList, dict, JSON string, or list."""
    if isinstance(raw_output, FactList):
        candidates = raw_output.fact
    elif isinstance(raw_output, dict) and "fact" in raw_output:
        candidates = raw_output["fact"]
    elif isinstance(raw_output, list):
        candidates = raw_output
    elif isinstance(raw_output, str):
        return parse_filter_facts(raw_output)
    else:
        return []

    valid: list[list[str]] = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                valid.append([str(x) for x in item])
    return valid


def match_facts_to_candidate_indices(
    candidate_facts: Sequence[Union[Tuple[str, str, str], List[str]]],
    generated_facts: Sequence[Sequence[str]],
    max_facts: Union[int, None] = 4,
) -> List[int]:
    """Map generated [s, p, o] facts back to original candidate_facts indices via difflib."""
    if not generated_facts or not candidate_facts:
        return []

    candidate_str_list = [
        str(list(f) if isinstance(f, tuple) else f) for f in candidate_facts
    ]
    candidate_list_of_lists = [
        list(f) if isinstance(f, tuple) else f for f in candidate_facts
    ]

    result_indices: List[int] = []
    for generated_fact in generated_facts:
        gen_str = str(list(generated_fact))
        matches = difflib.get_close_matches(
            gen_str, candidate_str_list, n=1, cutoff=0.0
        )
        if matches:
            closest_matched_fact = matches[0]
            try:
                idx = candidate_list_of_lists.index(ast.literal_eval(closest_matched_fact))
                if idx not in result_indices:
                    result_indices.append(idx)
            except (ValueError, SyntaxError):
                pass

    if max_facts is not None:
        return result_indices[:max_facts]
    return result_indices


# ---------------------------------------------------------------------------
# 8. Pure-JSON Context-Free Grammars (GBNF) for dspy.JSONAdapter on llama.cpp
# ---------------------------------------------------------------------------

DSPY_NER_GBNF_GRAMMAR = r"""root ::= "{\"named_entities\": " entity-list "}"
entity-list ::= "[]" | "[" string (", " string)* "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_TRIPLES_GBNF_GRAMMAR = r"""root ::= "{\"triples\": " triple-list "}"
triple-list ::= "[]" | "[" triple (", " triple)* "]"
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_RERANK_GBNF_GRAMMAR = r"""root ::= "{\"fact\": [" fact-list "]}"
fact-list ::= (triple (", " triple)*)?
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QA_COT_GBNF_GRAMMAR = r"""root ::= "{\"reasoning\": " string ", \"answer\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""


def _gbnf_escape_literal(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def build_dspy_ner_grammar(max_entities: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ExtractEntities, optionally capping entity count for sizecoding."""
    if max_entities is None or max_entities <= 0:
        return DSPY_NER_GBNF_GRAMMAR
    seq = "string"
    for _ in range(max_entities - 1):
        seq = f'string (", " {seq})?'
    return (
        'root ::= "{\\"named_entities\\": [" entity-list "]}"\n'
        f"entity-list ::= ({seq})?\n"
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_triples_grammar(max_triples: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ExtractTriples, optionally capping triple count for sizecoding."""
    if max_triples is None or max_triples <= 0:
        return DSPY_TRIPLES_GBNF_GRAMMAR
    seq = "triple"
    for _ in range(max_triples - 1):
        seq = f'triple (", " {seq})?'
    return (
        'root ::= "{\\"triples\\": [" triple-list "]}"\n'
        f"triple-list ::= ({seq})?\n"
        'triple ::= "[" string ", " string ", " string "]"\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_qa_cot_grammar(max_reasoning_chars: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for AnswerQuestion CoT, optionally capping reasoning length."""
    if max_reasoning_chars is None or max_reasoning_chars <= 0:
        return DSPY_QA_COT_GBNF_GRAMMAR
    return (
        'root ::= "{\\"reasoning\\": " reasoning-str ", \\"answer\\": " string "}"\n'
        f'reasoning-str ::= "\\"" char{{1,{int(max_reasoning_chars)}}} "\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_rerank_grammar(
    candidate_facts: Sequence[Union[Tuple[str, str, str], List[str]]],
    max_facts: int = 4,
) -> str:
    """Build dynamic candidate-literal pure-JSON GBNF grammar for DSPy FilterFacts (JSONAdapter)."""
    candidates: list[str] = []
    for item in candidate_facts:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not all(isinstance(x, str) for x in item)
        ):
            return DSPY_RERANK_GBNF_GRAMMAR
        serialized = json.dumps(list(item), ensure_ascii=False)
        if serialized not in candidates:
            candidates.append(serialized)

    if not candidates:
        return 'root ::= "{\\"fact\\": []}"\n'

    limit = min(max_facts, len(candidates))
    seq = "triple"
    for _ in range(limit - 1):
        seq = f'triple (", " {seq})?'
    triple_rule = " | ".join(_gbnf_escape_literal(c) for c in candidates)
    return (
        'root ::= "{\\"fact\\": [" fact-list "]}"\n'
        f"fact-list ::= ({seq})?\n"
        f"triple ::= {triple_rule}\n"
    )


def compute_sizecoding_pareto_score(
    recall_at_5: float,
    *,
    tokens: int,
    elapsed_seconds: float,
    min_baseline_recall: float = 0.0,
    target_token_seconds: float = 100.0,
    pred_name: str | None = None,
    missed_items: Sequence[str] = (),
) -> dspy.Prediction:
    """Compute a sizecoding Pareto metric (`recall@5 / (tokens * time)`) for `dspy.GEPA`.

    - Hard Tier 1 baseline gate: if `recall_at_5 < min_baseline_recall`, the candidate is
      clamped below `0.05` with zero `pareto_raw`/`efficiency` objective scores so no
      token-saving mutation can ever drop below the locked Tier 1 baseline.
    - Above the floor: computes `pareto_raw = recall_at_5 / (tokens * elapsed_seconds)`,
      maps it to a bounded `[0.10, 1.0]` score for `dspy.Evaluate`, attaches
      `objective_scores` (`recall_at_5`, `pareto_raw`, `efficiency`) for GEPA's multi-objective
      Pareto frontier (`gepa_kwargs={"frontier_type": "hybrid"}`), and emits targeted textual
      feedback so the reflection LM compresses instructions, triples, and reasoning tokens.
    """
    clamped_recall = max(0.0, min(1.0, float(recall_at_5)))
    safe_tokens = max(int(tokens), 1)
    safe_seconds = max(float(elapsed_seconds), 1e-3)
    token_seconds = safe_tokens * safe_seconds
    pareto_raw = clamped_recall / token_seconds
    target_budget = max(float(target_token_seconds), 1e-3)
    efficiency = target_budget / (target_budget + token_seconds)
    stage_prefix = f"[{pred_name}] " if pred_name else ""

    if clamped_recall < min_baseline_recall:
        floor_ratio = clamped_recall / max(float(min_baseline_recall), 1e-6)
        gated_score = 0.05 * max(0.0, min(0.99, floor_ratio))
        missed_str = f" Missed gold items: {list(missed_items)}." if missed_items else ""
        feedback = (
            f"{stage_prefix}BASELINE GATE VIOLATION: Recall@5={clamped_recall:.4f} dropped below "
            f"locked Tier 1 baseline ({min_baseline_recall:.4f}). Cost={safe_tokens} tokens * "
            f"{safe_seconds:.3f}s ({token_seconds:.2f} tok*s).{missed_str} "
            "Restore missing entity/triple/fact coverage first before compressing output tokens."
        )
        return dspy.Prediction(
            score=gated_score,
            feedback=feedback,
            pareto_raw=0.0,
            recall_at_5=clamped_recall,
            tokens=safe_tokens,
            elapsed_seconds=safe_seconds,
            objective_scores={
                "recall_at_5": clamped_recall,
                "pareto_raw": 0.0,
                "efficiency": 0.0,
            },
        )

    score = 0.10 + 0.90 * clamped_recall * efficiency
    missed_str = f" Missed gold items: {list(missed_items)}." if missed_items else ""
    feedback = (
        f"{stage_prefix}PASS Tier 1 floor ({clamped_recall:.4f} >= {min_baseline_recall:.4f}). "
        f"Sizecoding Pareto={pareto_raw:.6f} (Recall@5={clamped_recall:.4f} / "
        f"[{safe_tokens} tokens * {safe_seconds:.3f}s = {token_seconds:.2f} tok*s], "
        f"efficiency={efficiency:.4f}).{missed_str} "
        "To increase Pareto score, preserve 100% of gold retrieval bridges while pruning "
        "redundant triples, verbose entity aliases, and filler reasoning tokens."
    )
    return dspy.Prediction(
        score=score,
        feedback=feedback,
        pareto_raw=pareto_raw,
        recall_at_5=clamped_recall,
        tokens=safe_tokens,
        elapsed_seconds=safe_seconds,
        objective_scores={
            "recall_at_5": clamped_recall,
            "pareto_raw": pareto_raw,
            "efficiency": efficiency,
        },
    )


def lm_supports_gbnf(lm: dspy.LM | None = None) -> bool:
    """Return True if `lm` runs on a `llama.cpp` backend that supports GBNF grammars.

    Ryzen AI NPU/OGA (`ryzenai-llm`, e.g. `Mistral-7B-Instruct-v0.3-Hybrid`) and FLM
    do NOT support GBNF (`extra_body={"grammar": ...}`). Only `llamacpp` GPU models
    (e.g. `Qwen3-4B-Instruct-2507-GGUF`) support GBNF constrained decoding.
    """
    target_lm = lm if lm is not None else getattr(dspy.settings, "lm", None)
    if target_lm is None:
        return True
    explicit = getattr(target_lm, "_yaw_supports_gbnf", None)
    if isinstance(explicit, bool):
        return explicit
    recipe = getattr(target_lm, "_yaw_recipe", None)
    if recipe in ("ryzenai-llm", "flm"):
        return False
    model_name = str(getattr(target_lm, "model", "")).lower()
    if "mistral" in model_name or "ryzenai" in model_name or "flm" in model_name:
        return False
    return True


class HippoRAGProgram(dspy.Module):
    """Unified DSPy Module for HippoRAG 2 OpenIE, Recognition Memory, and Multi-Hop QA.

    Supports two execution styles across **all** stages (`ner`, `triples`, `fact_filter`, `qa`):
    1. **Heterogeneous Mistral Orchestrator + Qwen RLM Worker (`execution_mode="rlm"`):**
       - Outer `dspy.RLM` runs on `orchestrator_lm` (`Mistral-7B-Instruct-v0.3-Hybrid` on
         Ryzen AI NPU/OGA, `ChatAdapter`, **never** GBNF) inside `dspy.LocalInterpreter`.
       - Inner `llm_query` and built-in `qwen_*` sandbox tools (`qwen_extract_entities`,
         `qwen_extract_triples`, `qwen_filter_facts`, `qwen_answer_cot`) execute on `rlm_sub_lm`
         (`Qwen3-4B-Instruct-2507-GGUF` on `llama.cpp` Vulkan GPU) using `dspy.JSONAdapter()`
         and **pure-JSON GBNF grammars** (`extra_body={"grammar": ...}`).
    2. **Direct Single-Call (`execution_mode="direct"`):**
       - Runs `dspy.Predict` / `dspy.ChainOfThought` directly with `dspy.JSONAdapter()` and
         attaches GBNF grammars only when `lm_supports_gbnf(lm)` is `True`.
    """

    def __init__(
        self,
        *,
        execution_mode: Literal["direct", "rlm"] = "direct",
        qa_mode: Literal["cot", "rlm"] | None = None,
        include_demos: bool = True,
        enable_gbnf: bool = False,
        gbnf_max_entities: int | None = None,
        gbnf_max_triples: int | None = None,
        gbnf_max_facts: int = 4,
        gbnf_max_reasoning_chars: int | None = None,
        adapter: dspy.Adapter | None = None,
        orchestrator_lm: dspy.LM | None = None,
        rlm_max_iters: int = 10,
        rlm_max_llm_calls: int = 20,
        rlm_max_output_chars: int = 10000,
        rlm_sub_lm: dspy.LM | None = None,
        rlm_tools: list[Callable[..., Any]] | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.LocalInterpreter,
        compiled_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if execution_mode not in ("direct", "rlm"):
            raise ValueError(
                f"Unsupported execution_mode: {execution_mode!r}. Expected 'direct' or 'rlm'."
            )
        resolved_qa_mode: Literal["cot", "rlm"] = (
            qa_mode if qa_mode is not None else ("rlm" if execution_mode == "rlm" else "cot")
        )
        if resolved_qa_mode not in ("cot", "rlm"):
            raise ValueError(
                f"Unsupported qa_mode: {resolved_qa_mode!r}. Expected 'cot' or 'rlm'."
            )
        self.execution_mode: Literal["direct", "rlm"] = execution_mode
        self.qa_mode: Literal["cot", "rlm"] = resolved_qa_mode
        self.enable_gbnf: bool = enable_gbnf
        self.gbnf_max_entities: int | None = gbnf_max_entities
        self.gbnf_max_triples: int | None = gbnf_max_triples
        self.gbnf_max_facts: int = gbnf_max_facts
        self.gbnf_max_reasoning_chars: int | None = gbnf_max_reasoning_chars
        self.json_adapter: dspy.Adapter = adapter or dspy.JSONAdapter()
        self.orchestrator_lm: dspy.LM | None = orchestrator_lm
        self.rlm_sub_lm: dspy.LM | None = rlm_sub_lm

        self.ner = dspy.Predict(ExtractEntities)
        self.triples = dspy.Predict(ExtractTriples)
        self.fact_filter = dspy.Predict(FilterFacts)
        self.qa_cot = dspy.ChainOfThought(AnswerQuestion)

        qwen_worker_tools = self._build_qwen_worker_tools()
        combined_tools = list(qwen_worker_tools)
        if rlm_tools:
            existing_names = {getattr(t, "__name__", "") for t in combined_tools}
            for user_tool in rlm_tools:
                if getattr(user_tool, "__name__", "") not in existing_names:
                    combined_tools.append(user_tool)

        self.ner_rlm = dspy.RLM(
            ExtractEntities,
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=combined_tools,
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.triples_rlm = dspy.RLM(
            ExtractTriples,
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=combined_tools,
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.fact_filter_rlm = dspy.RLM(
            FilterFacts,
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=combined_tools,
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.qa_rlm = dspy.RLM(
            AnswerQuestion,
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=combined_tools,
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )

        if include_demos:
            self.ner.demos = list(NER_DEMOS)
            self.triples.demos = list(TRIPLE_DEMOS)
            self.fact_filter.demos = list(FACT_FILTER_DEMOS)
            self.qa_cot.demos = list(QA_DEMOS)

        if compiled_path is not None:
            path = Path(compiled_path)
            if path.exists():
                self.load(str(path))

    def _resolve_worker_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return self.rlm_sub_lm if self.rlm_sub_lm is not None else fallback_lm

    def _resolve_orchestrator_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return fallback_lm if fallback_lm is not None else self.orchestrator_lm

    def _build_qwen_worker_tools(self) -> list[Callable[..., Any]]:
        """Build GBNF-constrained Qwen (`llama.cpp` GPU) worker tools exposed inside Mistral's RLM REPL."""

        def qwen_extract_entities(passage: str) -> list[str]:
            """Extract deduplicated named entities from a passage using the GBNF-constrained Qwen GPU model."""
            worker_lm = self._resolve_worker_lm()
            return list(
                self.extract_entities(
                    passage,
                    mode="direct",
                    lm=worker_lm,
                    use_gbnf=self.enable_gbnf,
                )
            )

        def qwen_extract_triples(passage: str, named_entities: list[str]) -> list[list[str]]:
            """Extract [subject, predicate, object] triples using the GBNF-constrained Qwen GPU model."""
            worker_lm = self._resolve_worker_lm()
            triples = self.extract_triples(
                passage,
                named_entities,
                mode="direct",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )
            return [list(t) for t in triples]

        def qwen_filter_facts(
            question: str, candidate_facts: list[list[str]]
        ) -> list[list[str]]:
            """Filter candidate [s, p, o] facts using the dynamic candidate-literal GBNF Qwen GPU model."""
            worker_lm = self._resolve_worker_lm()
            indices = self.filter_facts(
                question,
                candidate_facts,
                mode="direct",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )
            return [list(candidate_facts[i]) for i in indices if 0 <= i < len(candidate_facts)]

        def qwen_answer_cot(context: str, question: str) -> dict[str, str]:
            """Run GBNF-constrained Chain-of-Thought QA on the Qwen GPU model and return {reasoning, answer}."""
            worker_lm = self._resolve_worker_lm()
            pred = self.answer_question(
                context,
                question,
                mode="cot",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )
            return {
                "reasoning": str(getattr(pred, "reasoning", "")),
                "answer": str(getattr(pred, "answer", "")),
            }

        return [
            qwen_extract_entities,
            qwen_extract_triples,
            qwen_filter_facts,
            qwen_answer_cot,
        ]

    def _dspy_ctx(self, lm: dspy.LM | None, *, use_json_adapter: bool = True):
        ctx_kwargs: dict[str, Any] = {}
        if lm is not None:
            ctx_kwargs["lm"] = lm
        if use_json_adapter:
            ctx_kwargs["adapter"] = self.json_adapter
        return dspy.context(**ctx_kwargs)

    def _should_use_gbnf(self, use_gbnf: bool | None, lm: dspy.LM | None) -> bool:
        requested = self.enable_gbnf if use_gbnf is None else use_gbnf
        return bool(requested and lm_supports_gbnf(lm))

    def extract_entities(
        self,
        passage: str,
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> Tuple[str, ...]:
        """Run DSPy NER extraction (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        active_mode = mode or self.execution_mode
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                pred = self.ner_rlm(passage=passage)
        else:
            call_kwargs: dict[str, Any] = {}
            if self._should_use_gbnf(use_gbnf, lm):
                grammar = build_dspy_ner_grammar(self.gbnf_max_entities)
                call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
            with self._dspy_ctx(lm, use_json_adapter=True):
                pred = self.ner(passage=passage, **call_kwargs)

        raw_entities = getattr(pred, "named_entities", [])
        if isinstance(raw_entities, str):
            return extract_named_entities(raw_entities)
        if not isinstance(raw_entities, (list, tuple)):
            return ()
        unique: list[str] = []
        seen: set[str] = set()
        for item in raw_entities:
            if isinstance(item, str):
                clean = item.strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    unique.append(clean)
        return tuple(unique)

    def extract_triples(
        self,
        passage: str,
        named_entities: Sequence[str],
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> Tuple[Tuple[str, str, str], ...]:
        """Run DSPy triple extraction (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        active_mode = mode or self.execution_mode
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                pred = self.triples_rlm(
                    passage=passage,
                    named_entities=list(named_entities),
                )
        else:
            call_kwargs: dict[str, Any] = {}
            if self._should_use_gbnf(use_gbnf, lm):
                grammar = build_dspy_triples_grammar(self.gbnf_max_triples)
                call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
            with self._dspy_ctx(lm, use_json_adapter=True):
                pred = self.triples(
                    passage=passage,
                    named_entities=list(named_entities),
                    **call_kwargs,
                )

        raw_triples = getattr(pred, "triples", [])
        if isinstance(raw_triples, str):
            return extract_triples(raw_triples)
        if not isinstance(raw_triples, (list, tuple)):
            return ()
        filtered = filter_invalid_triples(raw_triples)
        return tuple((t[0], t[1], t[2]) for t in filtered)

    def extract_openie(
        self,
        chunk_id: str,
        passage: str,
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> OpenIERecord:
        """Execute two-stage OpenIE (NER -> Triples) for a passage chunk."""
        entities = self.extract_entities(
            passage, mode=mode, lm=lm, use_gbnf=use_gbnf
        )
        triples = self.extract_triples(
            passage, entities, mode=mode, lm=lm, use_gbnf=use_gbnf
        )
        return OpenIERecord(
            chunk_id=chunk_id,
            entities=entities,
            triples=triples,
        )

    def filter_facts(
        self,
        question: str,
        candidate_facts: Sequence[Union[Tuple[str, str, str], List[str]]],
        *,
        mode: Literal["direct", "rlm"] | None = None,
        max_facts: int | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> List[int]:
        """Filter candidate facts (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        if not candidate_facts:
            return []
        effective_max_facts = self.gbnf_max_facts if max_facts is None else max_facts
        fact_before_filter = json.dumps({"fact": [list(f) for f in candidate_facts]})
        active_mode = mode or self.execution_mode
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                pred = self.fact_filter_rlm(
                    question=question,
                    fact_before_filter=fact_before_filter,
                )
        else:
            call_kwargs: dict[str, Any] = {}
            if self._should_use_gbnf(use_gbnf, lm):
                grammar = build_dspy_rerank_grammar(
                    candidate_facts, max_facts=effective_max_facts
                )
                call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
            with self._dspy_ctx(lm, use_json_adapter=True):
                pred = self.fact_filter(
                    question=question,
                    fact_before_filter=fact_before_filter,
                    **call_kwargs,
                )
        raw_after = getattr(pred, "fact", getattr(pred, "fact_after_filter", None))
        generated = _coerce_generated_facts(raw_after)
        return match_facts_to_candidate_indices(
            candidate_facts, generated, max_facts=effective_max_facts
        )

    def answer_question(
        self,
        context: str,
        question: str,
        *,
        mode: Literal["cot", "rlm"] | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Answer a question over retrieved context using either ChainOfThought or RLM."""
        active_mode = mode or self.qa_mode
        if active_mode not in ("cot", "rlm"):
            raise ValueError(f"Unsupported QA mode: {active_mode!r}")
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                return self.qa_rlm(context=context, question=question)

        call_kwargs: dict[str, Any] = {}
        if self._should_use_gbnf(use_gbnf, lm):
            grammar = build_dspy_qa_cot_grammar(self.gbnf_max_reasoning_chars)
            call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
        with self._dspy_ctx(lm, use_json_adapter=True):
            return self.qa_cot(context=context, question=question, **call_kwargs)

    def forward(
        self,
        context: str,
        question: str,
        *,
        mode: Literal["cot", "rlm"] | None = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Default DSPy forward pass executes multi-hop QA (CoT or RLM) for GEPA evaluation."""
        return self.answer_question(
            context=context, question=question, mode=mode, use_gbnf=use_gbnf
        )
