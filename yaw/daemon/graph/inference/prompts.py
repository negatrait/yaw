"""Native DSPy signatures for HippoRAG 2."""

from typing import Any, Literal, NotRequired, TypedDict

import dspy
from dspy.teleprompt.gepa.instruction_proposal import (
    CompressInstruction as CompressInstructionSignature,
)
from dspy.teleprompt.gepa.instruction_proposal import (
    ProposeInstruction as NativeProposeInstruction,
)
from pydantic import ConfigDict, with_config

# ---------------------------------------------------------------------------
# Native DSPy Signatures
# ---------------------------------------------------------------------------

EntitySensePair = tuple[str, str]
FactTriple = tuple[str, str, str]
ObjectPropertyPredicate = tuple[str, Literal["owl:ObjectProperty"]]
DatatypePropertyPredicate = tuple[str, Literal["owl:DatatypeProperty"]]


@with_config(ConfigDict(extra="forbid"))
class LiteralObject(TypedDict):
    value: str
    datatype: str
    lang: str


ObjectPropertyStatement = with_config(ConfigDict(extra="forbid"))(
    TypedDict(
        "ObjectPropertyStatement",
        {
            "sub": str | EntitySensePair,
            "pred": str | ObjectPropertyPredicate,
            "obj": str | EntitySensePair,
            "def": str,
        },
    )
)

DatatypePropertyStatement = with_config(ConfigDict(extra="forbid"))(
    TypedDict(
        "DatatypePropertyStatement",
        {
            "sub": str | EntitySensePair,
            "pred": str | DatatypePropertyPredicate,
            "obj": LiteralObject,
            "def": str,
        },
    )
)

ReifiedStatement = ObjectPropertyStatement | DatatypePropertyStatement


@with_config(ConfigDict(extra="forbid"))
class EquivalentLabelDecision(TypedDict):
    pair_id: str
    reasoning: str
    equivalent: Literal[True]
    definition: str
    hidden_label: bool


@with_config(ConfigDict(extra="forbid"))
class DistinctLabelDecision(TypedDict):
    pair_id: str
    reasoning: str
    equivalent: Literal[False]


LabelDecision = EquivalentLabelDecision | DistinctLabelDecision


@with_config(ConfigDict(extra="forbid"))
class RelationDecision(TypedDict):
    pair_id: str
    reasoning: str
    relation: Literal["skos:broader", "skos:narrower", "skos:related", "owl:differentFrom"]


SHACLRepairAction = Literal[
    "set_definition",
    "set_pref_label",
    "demote_to_alt_label",
    "merge_altLabel",
    "merge_hiddenLabel",
    "add_different_from",
    "assert_related",
    "assert_broader",
    "invert_broader",
    "demote_sibling_to_related",
    "remove_hierarchy_cycle",
    "remove_conflicting_related",
    "delete_dangling",
    "delete_invalid_fact",
]


@with_config(ConfigDict(extra="forbid"))
class SHACLRepair(TypedDict):
    focus_node: str
    action: SHACLRepairAction
    value: str


@with_config(ConfigDict(extra="forbid"))
class DistillerReference(TypedDict):
    example_id: str
    path: list[str]
    start: NotRequired[int]
    end: NotRequired[int]


@with_config(ConfigDict(extra="forbid"))
class DistillerSelection(TypedDict):
    diagnosis: str
    references: list[DistillerReference]


class ExtractEntities(dspy.Signature):
    # fmt: off
    (
        '\n    **Extract ALL named individuals and ALL general domain concepts** '
        'explicitly mentioned in `passage` prose.\n\n    - An '
        '`owl:NamedIndividual` is a particular person, organization, place, '
        'system, product or artifact.\n    - A `skos:Concept` is a general domain '
        'concept or category, such as a role, process or kind of thing.\n    - '
        'Provide a deduplicated source-grounded noun-phrase label and a concise '
        'one sentence contextual definition for each extracted resource.\n    - '
        'Never extract relationships, statements or document/header metadata.\n   '
        ' - Never extract `rdfs:Literal` data values such as numbers, dates, '
        'percentages, durations, booleans or descriptive strings as resources.\n  '
        '  - A resource label is also a JSON string: a named person, place or '
        'concept is not a literal merely because its label is textual.\n    - '
        'Include every such resource, including less prominent subjects and '
        'objects.\n    - The fact extractor discovers those literal values while '
        'establishing relationships.\n    - Never use a label as its own '
        'definition.\n    '
    )
    # fmt: on

    passage: str = dspy.InputField(desc="Source `passage` text")
    named_entities: list[EntitySensePair] = dspy.OutputField(
        desc="Deduplicated list of extracted (resource label, contextual definition) pairs"
    )


class ExtractTriples(dspy.Signature):
    """
    **Establish all passage-supported `rdf:Statement` facts** between supplied resources or literal values.
    - Each `pred` is a short relationship phrase.
    - Each separate `def` is a factual sentence stating how the subject relates to the object through that predicate.
    - Mention both endpoint names or the subject literal value, preserve source qualifiers.
    - Describe the relationship, endpoints have their definitions already.
    - Never copy or paraphrase a resource definition into `pred` or `def`.
    - NER alone discovers resources and their resource definitions. Do not introduce, rename or redefine resources.
    - Subjects and resource objects must use exact supplied labels.
    - For a homograph, select the exact supplied [label, definition] pair as that endpoint; copying this pair does not supply the statement's separate `def`.
    - Tag each relationship predicate with `owl:ObjectProperty` when connecting two supplied resources, `owl:DatatypeProperty` when connecting a subject resource to a passage literal value with its lexical string, datatype, and language tag.
    - Literal objects must not occur in named_entities. Subjects are resources, never literals.
    - Resolve pronouns to supplied labels; preserve predicate direction and literal meaning.
    - Never invent relationships to connect unrelated resources or use header metadata.
    """

    passage: str = dspy.InputField(desc="Source `passage` text to base `rdf:Statement` reifications.")
    named_entities: list[EntitySensePair] | list[str] = dspy.InputField(
        desc="Pre-extracted (`skos:Concept` | `owl:NamedIndividual`, `skos:definition`) pairs from Stage 1"
    )
    triples: list[ReifiedStatement] = dspy.OutputField(
        desc=(

                'Your reified `rdf:Statement` facts coupling `owl:ObjectProperty` predi'
                'cates with resource objects and `owl:DatatypeProperty` predicates with'
                ' literal objects'

        )
    )


class DefineMissingEntities(dspy.Signature):
    """
    **Synthesize concise, grounded genus-differentia contextual definitions** for entities referenced in `missing_entities` based on the passage text and extracted triples.
    - These resources were already discovered by NER; never discover, add or rename resources during definition repair.
    - Provide a contextual definition for every requested entity using the exact string from `missing_entities`; never emit an empty definition or repeat the entity name tautologically.
    """

    passage: str = dspy.InputField(desc="Source passage text")
    missing_entities: list[str] = dspy.InputField(desc="Entity names that require contextual definitions")
    triples: list[FactTriple | ReifiedStatement] = dspy.InputField(
        desc="Extracted relation triples referencing the missing entities"
    )
    entity_definitions: list[EntitySensePair] = dspy.OutputField(
        desc="Contextual definitions paired with each requested entity name"
    )


SEMANTIC_JUDGE_STAGE_NAMES: tuple[str, str] = ("label_judge", "relation_judge")


class JudgeLabels(dspy.Signature):
    """
    **Evaluate candidate entity pairs in `candidate_pairs`** for strict W3C SKOS identity equivalence (`skos:altLabel` / `skos:hiddenLabel`).
    
    CORE PRINCIPLE:
    - Merging two nodes (`equivalent: true`) asserts they are the EXACT SAME underlying `skos:Concept` or `owl:NamedIndividual` in this domain.
    - If two nodes are distinct concepts or distinct individuals (`equivalent: false`), emit `"equivalent": false` so the pair advances to the relation judge (`JudgeRelations`).
    
    MANDATORY FALSIFICATION RULES (state in 1 concise sentence in `reasoning` before `equivalent`):
    1. Named Individuals & Identifiers: NEVER merge (`equivalent: false`) two people, teams, regions, tiers, stages, versions, or percentiles that have different given names, surnames, numbers, or codes (e.g., 'Cinderella' vs 'Cindy Real', 'Michael Page' vs 'Michael Jackson', 'Tier 1' vs 'Tier 4', 'Free tier' vs 'Paid tier', 'p95' vs 'p99').
    2. Modifier-Head & Category Distinctions: NEVER merge (`equivalent: false`) a general system/tier/resource with a specific component, metric, regression, or policy that modifies it (e.g., 'Fast tier' vs 'fast tier p99 regression', 'Tenant' vs 'Tenant-id', 'Latency' vs 'p99 TTFT regression', 'Redwood' vs 'Redwood Router').
    3. True Equivalence (`equivalent: true`): Emit `equivalent: true` ONLY when both `a` and `b` denote the exact same concept or individual, provide a unified `"definition"`, and set `"hidden_label": false` for a valid synonym/acronym (`skos:altLabel`) or `"hidden_label": true` for a misspelling/typo (`skos:hiddenLabel`).
    """

    candidate_pairs: str = dspy.InputField(
        desc=(
            "JSON object containing `pairs`: list of candidate entity pairs with definitions, direct_facts, "
            "shared_role_facts, and chunk context"
        )
    )
    decisions: list[LabelDecision] = dspy.OutputField(
        desc=(
            "Equivalence decisions for each candidate pair, supplying a unified definition "
            "and hidden_label flag only when equivalent"
        )
    )
    # fmt: on


class JudgeRelations(dspy.Signature):
    """
    Decide the relation between distinct entities A and B from the evidence.
    Give one concise, complete evidence-based reason and exactly one relation.
    `skos:broader` means A is a kind, individual instance, or architectural part of B.
    `skos:narrower` means B is a kind, individual instance, or architectural part of A.
    Use `skos:related` for a non-hierarchical association, and `owl:differentFrom`
    for unrelated distinct entities. Causation, mitigation, protocol use, or mere
    co-occurrence alone does not establish hierarchy. Never place a role under a person.
    """

    candidate_pairs: str = dspy.InputField(desc="JSON evidence packet with pairs, entities and sources")
    decisions: list[RelationDecision] = dspy.OutputField(desc="One oriented relation and reasoning per pair")


class ProposeSHACLRepairs(dspy.Signature):
    """
    Repair only the reported graph defects using focus_context. Preserve source-supported
    entities and statements; do not invent facts.
    For a fact-* skos:definition defect, rewrite the rdf:Statement's definition as a factual
    sentence stating its subject-predicate-object relationship. Mention the supplied subject
    and object names and preserve the predicate's meaning. Describe the relationship, not an
    endpoint's meaning. Do not copy or paraphrase endpoint definitions. Return set_definition
    with that relational sentence as value.
    The supplied statement already gives its subject, predicate and object: repair its
    description, not its truth or endpoints. Use one concise sentence with endpoint names,
    never internal IDs or extra explanations of what the endpoints mean.
    For an entity definition defect, return set_definition with a non-tautological
    genus-differentia description. For noScalarLiterals, return set_pref_label with a lexical
    resource name grounded in its source/definition, or leave the defect unresolved.
    For label integrity, use set_pref_label or demote_to_alt_label. For homographs, use
    merge_altLabel/merge_hiddenLabel only when coreferent, otherwise add_different_from.
    For hierarchy/association defects, use assert_broader, invert_broader, assert_related,
    demote_sibling_to_related, remove_hierarchy_cycle or remove_conflicting_related as the
    evidence warrants. Only genuinely dangling endpoints or invalid facts permit
    delete_dangling/delete_invalid_fact; a defective definition never permits deletion.
    Return repairs for every violation, not just the first. Do not omit focus nodes.
    Each repair contains focus_node, action and value. If the batch cannot be repaired,
    return an empty repairs list rather than inventing repairs.
    """

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    focus_context: str = dspy.InputField(
        desc="Inspected neighborhood and hydrated source passage context for the focus nodes",
        default="",
    )
    repairs: list[SHACLRepair] = dspy.OutputField(desc="Closed-world SHACL repair actions for the reported focus nodes")


class FilterTriples(dspy.Signature):
    """
    Select the minimal closed-world subgraph of candidate `rdf:Statement` triples (`[rdf:subject, rdf:predicate, rdf:object]`) from `fact_before_filter` into `fact` required to entail the answer to `question`.
    - Prioritize triples that directly match the query's target `skos:Concept` anchors and sought relations, as well as intermediate bridge triples (`?s ?p1 ?bridge . ?bridge ?p2 ?o`) that connect multi-hop reasoning paths across passages.
    - Select at most 4 verbatim `[subject, predicate, object]` triples from `fact_before_filter`, ordered by evidentiary relevance; never fabricate or modify triple elements.
    """

    question: str = dspy.InputField(desc="Query for retrieval")
    fact_before_filter: str = dspy.InputField(desc="Candidate triples to be filtered")
    fact: list[FactTriple] = dspy.OutputField(
        desc="Up to 4 verbatim subject-predicate-object triples selected from fact_before_filter"
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
    """As a Recursive Language Model (RLM) orchestrator for closed-world W3C SHACL graph healing, delegate \
violating `sh:focusNode` items (`ConceptShape/*`, `StatementShape/closedWorld`, `EdgeShape/closedWorld`) \
to `propose_repairs()`, which takes zero arguments and automatically supplies the complete structured focus \
dossiers and source passages for the bound report, and immediately SUBMIT its result as `repairs` in a single turn.
    Tools (`inspect_focus_node`, `inspect_pair`, `hydrate_chunk`, `hydrate_ast_island`, `search_graph`, \
`propose_repairs`, and `lint_shacl`) read an immutable proposal snapshot. Repairs are applied and validated \
only after you return; repeated `lint_shacl` calls cannot reflect proposed repairs."""

    violations: str = dspy.InputField(desc="JSON list of SHACLViolation report items")
    repairs: list[SHACLRepair] = dspy.OutputField(desc="List of structured repair actions to restore SHACL conformance")


# ---------------------------------------------------------------------------
# 3 Distinct GEPA Reflection Signatures (RLMInstructionProposer)
# ---------------------------------------------------------------------------


class ReflectiveInnerDistillerSignature(dspy.Signature):
    """Inspect training traces in the sandbox and select original evidence by reference.
    Find generalizable failures and successful contrasts. Never write an instruction.
    Populate selection with a concise failure diagnosis and evidence references where each path
    lists dictionary keys inside that example (use [] for the full example or ["Feedback"],
    never an example list index like [0]), with optional start/end character offsets when path
    resolves to a string. Do not reconstruct facts.
    """

    current_instruction: str = dspy.InputField()
    examples_with_feedback: list[dict[str, Any]] = dspy.InputField()
    selection: DistillerSelection = dspy.OutputField()


# Native DSPy owns both writer signatures. Keep the established public names
# as compatibility aliases; there is no second implementation of either writer.

ProposeInstructionSignature = NativeProposeInstruction.with_instructions(
    "Refine `current_instruction` using the failure mechanisms and contrasts in `examples_with_feedback`.\n"
    "Focus on universal task strategies, formal W3C SKOS / RDF 1.1 / OWL 2 RL / SHACL structural axioms "
    "(Leibniz substitutability over reified `rdf:Statement` assertions, SKOS S13/S14 label disjointness, "
    "`skos:broader` taxonomic subsumption vs `skos:related` associative links, and `skos:Concept` vs "
    "`rdfs:Literal` range typing), and stage schema invariants while preserving all required output field "
    "names and existing capabilities without duplicating adapter JSON wrapper boilerplate.\n"
    "Never copy example entity names, proper nouns, locations, dates, numbers, or facts into the new "
    "instruction text. That anti-leakage rule applies only to the instruction text itself: never instruct "
    "the task model to anonymize, generalize away, omit, or replace exact source names, identifiers, "
    "dates, quantities, or units required by its component contract."
)


__all__ = [
    "AnswerQuestion",
    "CompressInstructionSignature",
    "DatatypePropertyPredicate",
    "DatatypePropertyStatement",
    "DefineMissingEntities",
    "DistillerReference",
    "DistillerSelection",
    "DistinctLabelDecision",
    "EntitySensePair",
    "EquivalentLabelDecision",
    "ExtractEntities",
    "ExtractQueryIntent",
    "ExtractTriples",
    "FactTriple",
    "FilterTriples",
    "HealSHACLViolations",
    "JudgeLabels",
    "JudgeRelations",
    "LabelDecision",
    "LiteralObject",
    "ObjectPropertyPredicate",
    "ObjectPropertyStatement",
    "ProposeInstructionSignature",
    "ProposeSHACLRepairs",
    "ReflectiveInnerDistillerSignature",
    "ReifiedStatement",
    "RelationDecision",
    "SHACLRepair",
    "SHACLRepairAction",
    "SEMANTIC_JUDGE_STAGE_NAMES",
]
