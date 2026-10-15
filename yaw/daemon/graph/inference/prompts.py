"""Native DSPy signatures and few-shot demonstration sets for HippoRAG 2."""

import json
from typing import Any

import dspy
from pydantic import BaseModel, Field

from .parsers import (
    _coerce_generated_facts,
    _strip_mistral_bracket_prefix,
    compute_mdhash_id,
    extract_named_entities,
    extract_triples,
    filter_invalid_triples,
    filter_reranked_facts,
    match_facts_to_candidate_indices,
    parse_filter_facts,
    repair_truncated_json,
    text_processing,
)
from .templates import (
    FACT_FILTER_BASE_MESSAGES,
    FACT_FILTER_INPUT_TEMPLATE,
    FACT_FILTER_OUTPUT_TEMPLATE,
    FACT_FILTER_SYSTEM,
    NER_ONE_SHOT_OUTPUT,
    NER_ONE_SHOT_PARAGRAPH,
    NER_SYSTEM,
    PROMPT_JSON_TEMPLATE,
    RAG_QA_DOCS,
    RAG_QA_ONE_SHOT_INPUT,
    RAG_QA_ONE_SHOT_OUTPUT,
    RAG_QA_SYSTEM,
    TRIPLE_FRAME,
    TRIPLE_ONE_SHOT_INPUT,
    TRIPLE_ONE_SHOT_OUTPUT,
    TRIPLE_SYSTEM,
    best_dspy_prompt,
    ner_conditioned_re_frame,
    ner_conditioned_re_input,
    ner_conditioned_re_output,
    ner_conditioned_re_system,
    ner_system,
    one_shot_ner_output,
    one_shot_ner_paragraph,
    one_shot_rag_qa_docs,
    one_shot_rag_qa_input,
    one_shot_rag_qa_output,
    rag_qa_system,
    render_fact_filter_messages,
    render_ner_messages,
    render_rag_qa_messages,
    render_triple_messages,
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


# Re-exports from .grammars for backwards compatibility
from .grammars import (
    DSPY_NER_GBNF_GRAMMAR,
    DSPY_QA_COT_GBNF_GRAMMAR,
    DSPY_RERANK_GBNF_GRAMMAR,
    DSPY_TRIPLES_GBNF_GRAMMAR,
    _gbnf_escape_literal,
    build_dspy_ner_grammar,
    build_dspy_qa_cot_grammar,
    build_dspy_rerank_grammar,
    build_dspy_triples_grammar,
    compute_sizecoding_pareto_score,
    lm_supports_gbnf,
)


def __getattr__(name: str) -> Any:
    if name in (
        "HippoRAGProgram",
        "_PositionalSafeRLM",
        "_STAGE_RLM_DELEGATION_RULES",
        "_build_rlm_action_demos",
    ):
        from .program import (
            HippoRAGProgram,
            _PositionalSafeRLM,
            _STAGE_RLM_DELEGATION_RULES,
            _build_rlm_action_demos,
        )

        mapping = {
            "HippoRAGProgram": HippoRAGProgram,
            "_PositionalSafeRLM": _PositionalSafeRLM,
            "_STAGE_RLM_DELEGATION_RULES": _STAGE_RLM_DELEGATION_RULES,
            "_build_rlm_action_demos": _build_rlm_action_demos,
        }
        val = mapping[name]
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
