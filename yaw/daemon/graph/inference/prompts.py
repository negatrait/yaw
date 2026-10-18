"""Native DSPy signatures and few-shot demonstration sets for HippoRAG 2."""

import json
from typing import Any

import dspy
from pydantic import BaseModel, Field

from .templates import (
    NER_ONE_SHOT_PARAGRAPH,
    RAG_QA_DOCS,
    best_dspy_prompt,
)

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
    """Your task is to extract named entities and their concise contextual definitions from the given paragraph.
    Respond with a JSON object in the format: {"named_entities": [["entity1", "concise contextual definition of entity1"], ["entity2", "concise contextual definition of entity2"], ...]}"""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.OutputField(
        desc="Deduplicated list of extracted [entity_name, contextual_definition] pairs"
    )


class ExtractTriples(dspy.Signature):
    """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
    Respond with a JSON object in the format:
    {"triples": [{"sub": "known_entity" or ["new_entity", "contextual definition"], "pred": "predicate", "obj": "known_entity" or ["new_entity", "contextual definition"], "def": "concise contextual definition of the statement"}, ...]}
    Pay attention to the following requirements:
    - Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
    - Clearly resolve pronouns to their specific names to maintain clarity.
    - If a "sub" or "obj" entity is already in named_entities with a definition, emit its exact string name (e.g., "Cinderella") without repeating its definition.
    - If a "sub" or "obj" entity is a new entity not present in named_entities, emit a 2-element list [entity_name, contextual_definition] (e.g., ["glass slipper", "Delicate footwear lost by Cinderella at the ball"]), and always provide a concise statement definition in "def"."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.InputField(
        desc="Extracted [entity_name, contextual_definition] pairs (or entity names) from Stage 1"
    )
    triples: list[Any] = dspy.OutputField(
        desc='List of relation triple objects {"sub", "pred", "obj", "def"} where known entities are strings and new entities are [name, definition] pairs'
    )


class DefineMissingEntities(dspy.Signature):
    """Your task is to provide concise contextual definitions grounded in the passage and extracted triples for any triple entities that were emitted without definitions.
    Respond with a JSON object in the format:
    {"entity_definitions": [["entity_name", "concise contextual definition grounded in the passage and triples"], ...]}"""

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
    """Classify candidate entity pairs into SKOS/OWL semantic relations:
    - 'skos:altLabel': Exact coreferent synonym (merge into canonical entity).
    - 'skos:hiddenLabel': Acronym, initialism, or lexical/spelling variant (merge into canonical entity).
    - 'skos:related': Associatively related but distinct concept (keep separate, add symmetric edge).
    - 'owl:differentFrom': Homograph or disjoint concept (keep separate, add repulsive edge).
    Respond with JSON: {"decisions": [{"pair_id": "id_a|id_b", "relation": "skos:altLabel|skos:hiddenLabel|skos:related|owl:differentFrom", "canonical": "id_a_or_id_b"}, ...]}"""

    candidate_pairs: str = dspy.InputField(
        desc="JSON list of candidate entity pairs with names, definitions, and similarity scores"
    )
    decisions: list[dict[str, str]] = dspy.OutputField(
        desc="List of SKOS/OWL disambiguation decisions per candidate pair"
    )


class ProposeSHACLRepairs(dspy.Signature):
    """Propose closed-world SKOS/RDF/OWL repairs for SHACL validation report violations.
    Supported actions: 'set_definition', 'set_pref_label', 'demote_to_alt_label', 'add_different_from', 'delete_dangling'.
    Respond with JSON: {"repairs": [{"focus_node": "...", "action": "...", "value": "..."}, ...]}"""

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    focus_context: str = dspy.InputField(
        desc="Inspected neighborhood and hydrated source passage context for the focus nodes",
        default="",
    )
    repairs: list[dict[str, str]] = dspy.OutputField(
        desc="List of structured repair actions to restore SHACL conformance"
    )


class FilterFacts(dspy.Signature):
    """You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information. You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer. The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return an empty list, {"fact\": []}. The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-making relies on your ability to accurately filter and present relevant information."""

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


class ExtractQueryIntent(dspy.Signature):
    """Rewrite a question as a concise declarative statement or target relation phrase for fact retrieval."""

    question: str = dspy.InputField(desc="User question")
    query_intent: str = dspy.OutputField(
        desc="Concise declarative statement of the sought fact"
    )


# ---------------------------------------------------------------------------
# 7 Distinct RLM Orchestrator Signatures (dspy.RLM)
# ---------------------------------------------------------------------------


class OrchestrateEntityExtraction(dspy.Signature):
    """Recursive RLM orchestrator for named entity and contextual definition extraction over source passages using sandboxed Python REPL and sub-LM calls."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.OutputField(
        desc="Deduplicated list of extracted [entity_name, contextual_definition] pairs"
    )


class OrchestrateTripleExtraction(dspy.Signature):
    """Recursive RLM orchestrator for RDF triple extraction ({"sub", "pred", "obj", "def"}) over source passages and Stage 1 [entity_name, contextual_definition] lists."""

    passage: str = dspy.InputField(desc="Source passage text")
    named_entities: list[Any] = dspy.InputField(
        desc="Extracted [entity_name, contextual_definition] pairs (or entity names) from Stage 1"
    )
    triples: list[Any] = dspy.OutputField(
        desc='List of relation triple objects {"sub", "pred", "obj", "def"} where known entities are strings (e.g., "Cinderella") and new entities are [name, definition] pairs (e.g., ["glass slipper", "Delicate footwear lost by Cinderella at the ball"])'
    )


class OrchestrateMissingEntityDefinitions(dspy.Signature):
    """Recursive RLM orchestrator for synthesizing concise contextual definitions for Stage 2 triple entities that were emitted without definitions."""

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


class OrchestrateSynonymJudgment(dspy.Signature):
    """Recursive RLM orchestrator for SKOS/OWL entity disambiguation using inspect_node and hydrate_chunk sandbox tools."""

    candidate_pairs: str = dspy.InputField(
        desc="JSON list of candidate entity pairs with names, definitions, and similarity scores"
    )
    decisions: list[dict[str, str]] = dspy.OutputField(
        desc="List of SKOS/OWL disambiguation decisions per candidate pair"
    )


class HealSHACLViolations(dspy.Signature):
    """Recursive RLM orchestrator for closed-world SHACL graph repair using inspect_node and hydrate_chunk sandbox tools."""

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    repairs: list[dict[str, str]] = dspy.OutputField(
        desc="List of structured repair actions to restore SHACL conformance"
    )


class OrchestrateFactFiltering(dspy.Signature):
    """Recursive RLM orchestrator for multi-hop recognition memory fact filtering."""

    question: str = dspy.InputField(desc="Query for retrieval")
    fact_before_filter: str = dspy.InputField(desc="Candidate facts to be filtered")
    fact: list[list[str]] = dspy.OutputField(
        desc="Selected relevant [subject, predicate, object] facts from the candidate list"
    )


class OrchestrateQuestionAnswering(dspy.Signature):
    """Recursive RLM orchestrator for multi-hop reading comprehension question answering."""

    context: str = dspy.InputField(desc="Retrieved evidence passages")
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Concise, definitive final answer")


# ---------------------------------------------------------------------------
# 4 Distinct GEPA Reflection Signatures (RLMInstructionProposer)
# ---------------------------------------------------------------------------


class ReflectiveOuterOrchestratorSignature(dspy.Signature):
    """Outer RLM orchestrator signature (Qwen GPU) for reflective GEPA instruction synthesis.
    Analyze task inputs, assistant responses, and evaluation feedback by delegating trace distillation to sub-LM calls, then synthesize an improved instruction."""

    current_instruction: str = dspy.InputField(
        desc="The instructions currently provided to the assistant."
    )
    examples_with_feedback: list[dict[str, Any]] = dspy.InputField(
        desc="Task inputs, assistant responses, and diagnostic evaluation feedback."
    )
    new_instruction: str = dspy.OutputField(
        desc="The new instruction for the assistant."
    )


class ReflectiveInnerDistillerSignature(dspy.Signature):
    """Inner RLM distiller signature (Mistral Hybrid) for reflective GEPA trace analysis.
    Distill failure patterns, missed entities/triples/facts, and domain rules from examples_with_feedback into a concise candidate instruction."""

    current_instruction: str = dspy.InputField(
        desc="The instructions currently provided to the assistant."
    )
    examples_with_feedback: list[dict[str, Any]] = dspy.InputField(
        desc="Task inputs, assistant responses, and diagnostic evaluation feedback."
    )
    new_instruction: str = dspy.OutputField(
        desc="The distilled new instruction for the assistant."
    )


class ProposeInstructionSignature(dspy.Signature):
    """I provided an assistant with instructions to perform a task for me. You are given those instructions, along with examples of different task inputs provided to the assistant, the assistant's response for each of them, and some feedback on how the assistant's response could be better.

    Your task is to write a new instruction for the assistant.

    Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

    Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well."""

    current_instruction: str = dspy.InputField(
        desc="The instructions I provided to the assistant."
    )
    examples_with_feedback: list[dict[str, Any]] = dspy.InputField(
        desc="Task inputs provided to the assistant, the assistant's response for each, and feedback on how the response could be better."
    )
    new_instruction: str = dspy.OutputField(
        desc="The new instruction for the assistant."
    )


class CompressInstructionSignature(dspy.Signature):
    """Shorten the instruction to fit within the stated limit. Preserve every
    strategy, rule, and constraint; cut redundancy and verbosity. Do not add
    new content. Output only the shortened instruction."""

    instruction: str = dspy.InputField()
    length_limit: str = dspy.InputField()
    new_instruction: str = dspy.OutputField(
        desc="The shortened instruction for the assistant."
    )


# Backwards-compatible alias
ReflectiveInstructionTask = ReflectiveOuterOrchestratorSignature


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


__all__ = [
    "AnswerQuestion",
    "CompressInstructionSignature",
    "DefineMissingEntities",
    "ExtractEntities",
    "ExtractQueryIntent",
    "ExtractTriples",
    "FACT_FILTER_DEMOS",
    "FactList",
    "FilterFacts",
    "HealSHACLViolations",
    "JudgeSynonyms",
    "NER_DEMOS",
    "OrchestrateEntityExtraction",
    "OrchestrateFactFiltering",
    "OrchestrateMissingEntityDefinitions",
    "OrchestrateQuestionAnswering",
    "OrchestrateSynonymJudgment",
    "OrchestrateTripleExtraction",
    "ProposeInstructionSignature",
    "ProposeSHACLRepairs",
    "QA_DEMOS",
    "ReflectiveInnerDistillerSignature",
    "ReflectiveInstructionTask",
    "ReflectiveOuterOrchestratorSignature",
    "TRIPLE_DEMOS",
]
