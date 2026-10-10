"""Normalizers, spaCy pipeline components, factories, and model wrappers."""

from __future__ import annotations

import functools
import logging
import re
import typing
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy_llm.registry import registry

from yaw import config
from yaw.nlp.models import (
    ExtractedNode,
    ExtractedTriple,
    RelDict,
    _extract_rel_fields,
    map_namespace,
    normalize_label,
)
from yaw.nlp.utils import is_valid_entity_text
from yaw.ontology.registry import get_ontology_registry

logger = logging.getLogger("yaw.leaf_extractor")


@functools.lru_cache(maxsize=4)
def get_base_spacy(model_name: str | None = None) -> Language:
    """Return lightweight base linguistic pipeline for tokenization and lemmatization."""
    target = model_name or config.get_spacy_model_name()
    try:
        return spacy.load(target, exclude=["ner"])
    except OSError:
        fallback = config.get_spacy_model_name()
        return spacy.load(fallback, exclude=["ner"])


@functools.lru_cache(maxsize=4)
def get_spacy_pipeline(model_name: str = "en_core_web_lg") -> Language:
    """Standard spaCy pipeline with all components set to default behavior."""
    return spacy.load(model_name)


CONCEPT_WORDS: frozenset[str] = frozenset(
    {
        "mechanics",
        "physics",
        "theory",
        "principle",
        "law",
        "equation",
        "theorem",
        "effect",
        "model",
        "hypothesis",
        "constant",
        "superconductivity",
        "relativity",
        "algorithm",
        "method",
        "system",
        "network",
        "entropy",
        "information",
    }
)


def _has_named_entity_spans(doc: Doc, cleaned: str) -> bool:
    """Check if doc contains named entity labels or proper noun tokens."""
    for e in doc.ents:
        if e.text.strip().lower() == cleaned.lower() and e.label_ in {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "NORP",
            "FAC",
        }:
            return True
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    return len(tokens) >= 2 and all(
        t.pos_ == "PROPN"
        or t.tag_ in {"NNP", "NNPS"}
        or (t.text[0].isupper() and not t.is_stop and t.pos_ != "NOUN")
        for t in tokens
    )


def _check_ner(cleaned: str, pipe: Language) -> bool:
    """Safely parse doc and evaluate entity spans."""
    try:
        return _has_named_entity_spans(pipe(cleaned), cleaned)
    except (KeyError, ValueError, IndexError, TypeError) as err:
        logger.debug("Named entity detection bypassed: %s", err)
        return False


def is_named_entity(text: str, nlp: Language | None = None) -> bool:
    """Determine whether an entity string represents a named entity (PERSON, ORG, GPE, proper noun)
    rather than an epistemological domain concept.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 2:
        return False

    words = [w.lower() for w in cleaned.split() if w.isalpha()]
    if any(w in CONCEPT_WORDS for w in words):
        return False

    pipe = (
        nlp if (nlp is not None and "ner" in nlp.pipe_names) else get_spacy_pipeline()
    )
    if _check_ner(cleaned, pipe):
        return True

    parts = cleaned.split()
    return len(parts) >= 2 and all(p[0].isupper() and p.isalpha() for p in parts)


PREDICATE_ALIASES: dict[str, str] = {
    # is_a
    "is_a": "is_a",
    "isa": "is_a",
    "is a": "is_a",
    "is an": "is_a",
    "are": "is_a",
    "type of": "is_a",
    "instance of": "is_a",
    # partOf
    "partof": "partOf",
    "part_of": "partOf",
    "part of": "partOf",
    "component of": "partOf",
    "subset of": "partOf",
    # subClassOf
    "subclassof": "subClassOf",
    "sub_class_of": "subClassOf",
    "subclass of": "subClassOf",
    "specialization of": "subClassOf",
    "subtype of": "subClassOf",
}


def _match_base_or_alias(rel: str, bp_tuple: tuple[str, ...]) -> str | None:
    """Direct case/underscore-insensitive match or canonical alias dictionary match."""
    if not bp_tuple or not rel:
        return None
    rel_clean = rel.lower().replace("_", "").replace(" ", "")
    for bp in bp_tuple:
        if (
            bp.lower() == rel.lower()
            or bp.lower().replace("_", "").replace(" ", "") == rel_clean
        ):
            return bp
    raw_lower = rel.lower().strip()
    underscored = re.sub(r"\s+", "_", raw_lower)
    spaced = raw_lower.replace("_", " ")
    squashed = raw_lower.replace("_", "").replace(" ", "")
    for cand in (raw_lower, underscored, spaced, squashed):
        if cand in PREDICATE_ALIASES and PREDICATE_ALIASES[cand] in bp_tuple:
            return PREDICATE_ALIASES[cand]
    return None


def _lemmatize_phrase(text: str, base_pipe: Language) -> str:
    """Run isolated token lemmatization on a phrase."""
    doc = base_pipe.make_doc(text)
    for pipe_name in ("tok2vec", "tagger", "attribute_ruler", "lemmatizer"):
        if pipe_name in base_pipe.pipe_names:
            doc = base_pipe.get_pipe(pipe_name)(doc)
    return " ".join(
        t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space
    ).strip()


def _get_vector_or_doc(text: str, nlp_or_vocab: Any) -> Any:
    """Retrieve Lexeme with vector or create Doc with average token vector."""
    vocab = getattr(nlp_or_vocab, "vocab", nlp_or_vocab)
    try:
        lex = vocab[text]
        if lex.has_vector and float(getattr(lex, "vector_norm", 0.0)) > 0.0:
            return lex
    except KeyError, ValueError, IndexError, TypeError:
        pass
    if hasattr(nlp_or_vocab, "make_doc"):
        doc = nlp_or_vocab.make_doc(text)
    else:
        words = text.split()
        doc = Doc(vocab, words=words) if words else None
    if (
        doc is not None
        and doc.has_vector
        and float(getattr(doc, "vector_norm", 0.0)) > 0.0
    ):
        return doc
    return None


def _find_best_vector_match(
    target_text: str,
    bp_tuple: tuple[str, ...],
    nlp_or_vocab: Any,
    threshold: float,
) -> str | None:
    """Find best matching base predicate via GloVe vector cosine similarity > threshold."""
    if not bp_tuple or not target_text:
        return None
    vocab = getattr(nlp_or_vocab, "vocab", nlp_or_vocab)
    vectors = getattr(vocab, "vectors", None)
    if vectors is None or getattr(vectors, "n_keys", 0) == 0:
        return None

    target_obj = _get_vector_or_doc(target_text, nlp_or_vocab)
    if target_obj is None:
        return None

    best_bp: str | None = None
    best_sim = 0.0
    for bp in bp_tuple:
        bp_clean = (
            re.sub(r"([a-z])([A-Z])", r"\1 \2", bp).replace("_", " ").lower().strip()
        )
        bp_obj = _get_vector_or_doc(bp_clean, nlp_or_vocab)
        if bp_obj is None:
            bp_obj = _get_vector_or_doc(bp, nlp_or_vocab)
        if bp_obj is not None:
            sim = float(target_obj.similarity(bp_obj))
            if sim > best_sim:
                best_sim = sim
                best_bp = bp

    if best_sim >= threshold:
        return best_bp
    return None


def normalize_predicate(
    predicate: str,
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
    threshold: float = config.CANON_SIM_THRESHOLD,
) -> str:
    """Directly normalize predicate: exact alias -> isolated lemma -> GloVe cosine > threshold.

    Zero recursion and zero LLM calls.
    """
    rel = predicate.strip()
    if not rel:
        return ""

    bp_tuple = (
        tuple(base_predicates)
        if base_predicates is not None
        else tuple(config.BASE_PREDICATES)
    )

    # 1. Direct or alias match
    direct = _match_base_or_alias(rel, bp_tuple)
    if direct is not None:
        return direct

    # 2. Isolated spaCy lemmatization
    decamel = re.sub(r"([a-z])([A-Z])", r"\1 \2", rel).replace("_", " ").lower().strip()
    base_pipe = (
        nlp
        if (nlp is not None and "lemmatizer" in nlp.pipe_names)
        else get_base_spacy()
    )
    lemma = _lemmatize_phrase(decamel, base_pipe)

    # Check lemmatized direct or alias match
    if lemma:
        lemma_match = _match_base_or_alias(lemma, bp_tuple)
        if lemma_match is not None:
            return lemma_match

    # 3. In-process vector similarity > threshold
    pipe_for_vec = nlp if nlp is not None else base_pipe
    for cand_text in (lemma, decamel):
        if not cand_text:
            continue
        vector_match = _find_best_vector_match(
            cand_text, bp_tuple, pipe_for_vec, threshold
        )
        if vector_match is not None:
            return vector_match

    return lemma or decamel


POLARITY_EXCLUSION_MASK: dict[str, frozenset[str]] = {
    "causes": frozenset(
        {
            "prevent",
            "prevents",
            "prevented",
            "preventing",
            "prevention",
            "inhibit",
            "inhibits",
            "inhibited",
            "inhibiting",
            "inhibition",
            "block",
            "blocks",
            "blocked",
            "blocking",
            "stop",
            "stops",
            "stopped",
            "stopping",
            "thwart",
            "thwarts",
            "thwarted",
            "preclude",
            "precludes",
        }
    ),
    "prevents": frozenset(
        {
            "cause",
            "causes",
            "caused",
            "causing",
            "induce",
            "induces",
            "induced",
            "inducing",
            "trigger",
            "triggers",
            "triggered",
            "triggering",
            "produce",
            "produces",
            "produced",
            "producing",
            "generate",
            "generates",
            "generated",
            "lead to",
            "results in",
        }
    ),
    "supports": frozenset(
        {
            "refute",
            "refutes",
            "refuted",
            "refuting",
            "contradict",
            "contradicts",
            "contradicted",
            "contradicting",
            "oppose",
            "opposes",
            "opposed",
            "opposing",
            "disprove",
            "disproves",
            "disproved",
            "disproving",
            "rebut",
            "rebuts",
            "undermine",
            "undermines",
        }
    ),
    "refutes": frozenset(
        {
            "support",
            "supports",
            "supported",
            "supporting",
            "confirm",
            "confirms",
            "confirmed",
            "confirming",
            "prove",
            "proves",
            "proved",
            "proving",
            "corroborate",
            "corroborates",
            "substantiate",
            "uphold",
        }
    ),
    "contradicts": frozenset(
        {
            "support",
            "supports",
            "supported",
            "supporting",
            "agree",
            "agrees",
            "agreed",
            "align",
            "aligns",
            "aligned",
            "match",
            "matches",
            "consistent with",
        }
    ),
    "precedes": frozenset(
        {
            "follow",
            "follows",
            "followed",
            "following",
            "succeed",
            "succeeds",
            "succeeded",
            "succeeding",
        }
    ),
}

SYNSET_DICTIONARY: dict[str, str] = {
    # Causal
    "induces": "causes",
    "induce": "causes",
    "leads to": "causes",
    "lead to": "causes",
    "triggers": "causes",
    "trigger": "causes",
    "results in": "causes",
    "sparks": "causes",
    "brings about": "causes",
    "produces": "causes",
    "generates": "causes",
    # Dialectical supports
    "corroborates": "supports",
    "substantiates": "supports",
    "upholds": "supports",
    "endorses": "supports",
    "backs": "supports",
    "confirms": "supports",
    "verifies": "supports",
    # Dialectical refutes / contradicts
    "disproves": "refutes",
    "debunks": "refutes",
    "rebuts": "refutes",
    "invalidates": "refutes",
    "opposes": "contradicts",
    "negates": "contradicts",
    # Mereology
    "consists of": "partOf",
    "comprises": "partOf",
    "composed of": "partOf",
    "constitutes": "partOf",
    "belongs to": "partOf",
    "element of": "partOf",
    # Subsumption
    "is a type of": "subClassOf",
    "is a kind of": "subClassOf",
    "categorized as": "subClassOf",
}


def is_antonym_conflict(candidate: str, target_predicate: str) -> bool:
    """Check if candidate text violates polarity exclusion mask for target predicate."""
    norm_target = target_predicate.split(":", 1)[-1]
    if norm_target in POLARITY_EXCLUSION_MASK:
        cand_clean = candidate.lower().strip()
        mask = POLARITY_EXCLUSION_MASK[norm_target]
        if cand_clean in mask:
            return True
        tokens = set(re.findall(r"\w+", cand_clean))
        if tokens & mask:
            return True
        for blocked in mask:
            if " " in blocked and blocked in cand_clean:
                return True
    return False


def _normalize_tier1_lookup(
    rel: str,
    raw_lower: str,
    decamel: str,
    bp_tuple: tuple[str, ...],
) -> str | None:
    """TIER 1: Synsets, dynamic properties, and direct alias matching."""
    if raw_lower in SYNSET_DICTIONARY:
        mapped = SYNSET_DICTIONARY[raw_lower]
        if mapped in bp_tuple:
            return mapped

    if decamel in SYNSET_DICTIONARY:
        mapped = SYNSET_DICTIONARY[decamel]
        if mapped in bp_tuple:
            return mapped

    for bp in bp_tuple:
        if bp.lower() in (raw_lower, decamel):
            return bp

    reg = get_ontology_registry()
    snapshot = reg.snapshot
    if rel in snapshot.properties:
        return snapshot.properties[rel].curie
    for curie, prop_meta in snapshot.properties.items():
        if curie.lower() == raw_lower or prop_meta.label.lower() == decamel:
            return prop_meta.curie
    return None


def _normalize_tier2_vector(
    decamel: str,
    bp_tuple: tuple[str, ...],
    nlp: Language | None,
    threshold: float,
) -> str | None:
    """TIER 2: Antonym-guarded GloVe vector similarity matching."""
    base_pipe = nlp or get_base_spacy()
    target_doc = base_pipe(decamel)
    if not (target_doc and target_doc.has_vector and target_doc.vector_norm > 0):
        return None

    best_sim = 0.0
    best_bp: str | None = None

    for bp in bp_tuple:
        if is_antonym_conflict(decamel, bp):
            continue

        bp_clean = (
            re.sub(r"([a-z])([A-Z])", r"\1 \2", bp).replace("_", " ").lower().strip()
        )
        bp_doc = base_pipe(bp_clean)
        if bp_doc and bp_doc.has_vector and bp_doc.vector_norm > 0:
            sim = float(target_doc.similarity(bp_doc))
            if sim > best_sim:
                best_sim = sim
                best_bp = bp

    if best_sim >= threshold and best_bp is not None:
        return best_bp
    return None


def normalize_predicate_antonym_guarded(
    predicate: str,
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
    threshold: float = config.CANON_SIM_THRESHOLD,
) -> str:
    """Tri-level predicate normalizer: (1) Synsets & Aliases -> (2) Antonym-Masked Cosine -> (3) Closure."""
    rel = predicate.strip()
    if not rel:
        return ""

    bp_tuple = tuple(base_predicates or config.BASE_PREDICATES)
    raw_lower = rel.lower()
    decamel = re.sub(r"([a-z])([A-Z])", r"\1 \2", rel).replace("_", " ").lower().strip()

    tier1 = _normalize_tier1_lookup(rel, raw_lower, decamel, bp_tuple)
    if tier1 is not None:
        return tier1

    tier2 = _normalize_tier2_vector(decamel, bp_tuple, nlp, threshold)
    if tier2 is not None:
        return tier2

    # --- TIER 3: Compile-Time Property Closure Fallback ---
    reg = get_ontology_registry()
    if ":" in rel:
        curie = reg.uri_to_curie(rel)
        if curie in reg.snapshot.properties:
            return curie

    return decamel.replace(" ", "_")


class DynamicConceptMentionScoper:
    """Bottom-up candidate scoping for closed-domain extraction (Quadrant 4).

    Pre-matches text chunks against registered ontology labels via spaCy C-speed
    PhraseMatcher (Trie) to constrain candidate alignments without bloating LLM prompt prefixes.
    """

    def __init__(self, nlp: Language) -> None:
        self.nlp = nlp
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._rebuild()

    def _rebuild(self) -> None:
        reg = get_ontology_registry()
        snapshot = reg.snapshot
        patterns: dict[str, list[Doc]] = {}
        for curie, meta in snapshot.classes.items():
            if curie.startswith("http"):
                continue
            doc = self.nlp.make_doc(meta.label)
            patterns.setdefault(curie, []).append(doc)
        for curie, docs in patterns.items():
            self.matcher.add(curie, docs)

    def scope_chunk_candidates(self, text: str) -> set[str]:
        """Find candidate class CURIEs present in chunk text (O(N) token scan)."""
        doc = self.nlp.make_doc(text)
        matches = self.matcher(doc)
        return {self.nlp.vocab.strings[match_id] for match_id, _, _ in matches}


def _entity_definition(ent: Span, clean_name: str | None = None) -> str:
    """Format definition using native spaCy entity label."""
    name = (
        clean_name
        if clean_name is not None
        else ent.text.strip(" \t\n\r\"'.,:;!?(){}[]<>")
    )
    return f"{name} ({ent.label_})"


def _filter_code_spans(spans: Iterable[Span], code_spans: list[Span]) -> list[Span]:
    """Filter out spans that overlap with markdown code regions or code tokens."""
    valid: list[Span] = []
    for span in spans:
        overlaps = any(
            (span.start_char < cs.end_char and span.end_char > cs.start_char)
            for cs in code_spans
        ) or any(getattr(t._, "is_code", False) for t in span)
        if not overlaps:
            valid.append(span)
    return valid


@Language.factory("markdown_structure")
def make_markdown_structure(nlp: Language, name: str):
    """Identifies markdown code blocks and inline code, recording code spans and token tags."""
    Token.set_extension("is_code", default=False, force=True)
    pattern = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~|`+[^`\n]+`+")

    def _tag_markdown_structure(doc: Doc) -> Doc:
        code_spans: list[Span] = []
        for match in pattern.finditer(doc.text):
            span = doc.char_span(
                match.start(), match.end(), label="code", alignment_mode="expand"
            )
            if span is not None:
                code_spans.append(span)
                for token in span:
                    token._.is_code = True
        doc.spans["code"] = code_spans
        return doc

    return _tag_markdown_structure


@Language.factory("span_reconciler")
def make_span_reconciler(nlp: Language, name: str):
    """Reconciles SpanCat / candidate spans into non-code, non-overlapping doc.ents."""

    def _reconcile(doc: Doc) -> Doc:
        code_spans = doc.spans.get("code", [])
        candidate_spans = list(doc.ents) if doc.ents else list(doc.spans.get("sc", []))
        clean_spans = _filter_code_spans(candidate_spans, code_spans)
        doc.ents = filter_spans(clean_spans)
        return doc

    return _reconcile


def _extract_entity_nodes(
    doc: Doc, node_map: dict[str, ExtractedNode]
) -> list[ExtractedNode | None]:
    """Extract valid, deduplicated entity nodes from doc spans."""
    ent_nodes: list[ExtractedNode | None] = []
    code_spans = doc.spans.get("code", [])
    candidate_ents = list(doc.ents) if doc.ents else list(doc.spans.get("sc", []))
    valid_ents = _filter_code_spans(candidate_ents, code_spans)

    for ent in valid_ents:
        clean_name = ent.text.strip(" \t\n\r\"'.,:;!?(){}[]<>")
        if not is_valid_entity_text(clean_name):
            ent_nodes.append(None)
            continue
        ns = map_namespace(ent)
        key = f"{ns}:{clean_name.casefold()}"
        if key not in node_map:
            node_map[key] = ExtractedNode(
                name=clean_name,
                namespace=ns,
                definition=_entity_definition(ent, clean_name=clean_name),
            )
        ent_nodes.append(node_map[key])
    return ent_nodes


def _resolve_base_predicates(
    base_predicates: tuple[str, ...] | list[str] | None,
    nlp: Language | None,
) -> list[str] | None:
    """Resolve active base predicate labels from arguments or pipeline configuration."""
    if base_predicates is not None:
        return list(base_predicates)
    if nlp is not None and "llm_rel" in nlp.pipe_names:
        rel_pipe = nlp.get_pipe("llm_rel")
        labels = getattr(rel_pipe, "labels", None)
        if isinstance(labels, (list, tuple, set)):
            return [str(x) for x in labels]
    return None


def _build_single_triple(
    r: Any,
    ent_nodes: list[ExtractedNode | None],
    node_map: dict[str, ExtractedNode],
    bp_list: list[str] | None,
    nlp: Language | None,
    seen_triples: set[tuple[str, str, str]],
) -> ExtractedTriple | None:
    dep, dest, relation = _extract_rel_fields(r)
    if dep is None or dest is None or not relation:
        return None
    if not (0 <= dep < len(ent_nodes) and 0 <= dest < len(ent_nodes)):
        return None
    sub, obj = ent_nodes[dep], ent_nodes[dest]
    if sub is None or obj is None:
        return None

    norm = normalize_predicate(relation, base_predicates=bp_list, nlp=nlp)
    norm_rel = norm or relation.strip() or "relates to"
    key = (sub.name.casefold(), norm_rel.casefold(), obj.name.casefold())
    if key in seen_triples:
        return None
    seen_triples.add(key)

    pred_key = f"predicates:{norm_rel.casefold()}"
    if pred_key not in node_map:
        node_map[pred_key] = ExtractedNode(
            name=norm_rel,
            namespace="predicates",
            definition=f"Relation '{norm_rel}' connecting {sub.name} and {obj.name}.",
        )
    return ExtractedTriple(subject=sub, predicate=node_map[pred_key], object=obj)


def _extract_triples(
    doc: Doc,
    ent_nodes: list[ExtractedNode | None],
    node_map: dict[str, ExtractedNode],
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
) -> list[ExtractedTriple]:
    """Extract deduplicated relational triples from doc._.rel."""
    triples: list[ExtractedTriple] = []
    seen_triples: set[tuple[str, str, str]] = set()
    bp_list = _resolve_base_predicates(base_predicates, nlp)

    raw_rel = getattr(doc._, "rel", None) or []
    rel_source = (
        RelDict(raw_rel)
        if isinstance(raw_rel, dict) and not isinstance(raw_rel, RelDict)
        else raw_rel
    )

    for r in rel_source:
        triple = _build_single_triple(
            r, ent_nodes, node_map, bp_list, nlp, seen_triples
        )
        if triple is not None:
            triples.append(triple)
    return triples


_INJECTED_TEST_MODELS: dict[str, Any] = {}


@dataclass(slots=True)
class _TestModelWrapper:
    """Type-consistent wrapper for test callables injected into spacy-llm."""

    target: Any
    context_length: int = 8192

    def __post_init__(self) -> None:
        self.context_length = int(
            getattr(self.target, "context_length", self.context_length)
        )

    def __call__(
        self, prompts: typing.Iterable[typing.Iterable[str]]
    ) -> typing.Iterable[typing.Iterable[str]]:
        return self.target(prompts)


@registry.llm_models("yaw.TestModelWrapper.v1")
def make_test_model_wrapper(
    model_id: str,
) -> Callable[
    [typing.Iterable[typing.Iterable[str]]], typing.Iterable[typing.Iterable[str]]
]:
    """Wraps an injected test callable into a type-consistent spaCy-LLM model executor."""
    target = _INJECTED_TEST_MODELS.get(model_id)
    if target is None:
        raise KeyError(f"No injected test model found for ID '{model_id}'")
    return _TestModelWrapper(target)


def _model_to_config(model: Any | None) -> dict[str, Any]:
    """Resolve model argument into a serializable spaCy-LLM model config."""
    if model is None:
        return {
            "@llm_models": "yaw.Ollama.v1",
        }
    if isinstance(model, dict):
        return model
    if isinstance(model, str):
        if model in registry.llm_models:
            return {"@llm_models": model}
        return {
            "@llm_models": "yaw.Ollama.v1",
            "name": model,
        }
    if callable(model):
        model_id = f"mock_{id(model)}_{uuid.uuid4().hex}"
        _INJECTED_TEST_MODELS[model_id] = model
        return {
            "@llm_models": "yaw.TestModelWrapper.v1",
            "model_id": model_id,
        }
    raise TypeError(
        f"Invalid model specification of type {type(model).__name__}: {model!r}"
    )


_orig_create_pipe = Language.create_pipe


def _patched_create_pipe(
    self: Language,
    factory_name: str,
    name: str | None = None,
    *,
    config: Any = None,
    raw_config: Any = None,
    validate: bool = True,
) -> Any:
    if factory_name == "llm" and isinstance(config, dict) and "model" in config:
        model_val = config.get("model")
        if callable(model_val) and not isinstance(model_val, dict):
            config = dict(config)
            config["model"] = _model_to_config(model_val)
            config.setdefault("validate_types", False)
    kwargs: dict[str, Any] = {
        "raw_config": raw_config,
        "validate": validate,
    }
    if config is not None:
        kwargs["config"] = config
    return _orig_create_pipe(self, factory_name, name=name, **kwargs)


_YAW_CREATE_PIPE_PATCHED: bool = False
if not _YAW_CREATE_PIPE_PATCHED:
    Language.create_pipe = _patched_create_pipe  # type: ignore[method-assign]
    _YAW_CREATE_PIPE_PATCHED = True


@registry.misc("yaw.LabelNormalizer.v1")
def make_label_normalizer() -> Callable[[str], str]:
    """Canonical label normalizer mapping singular/plural case-folded entity labels to taxonomy namespaces."""
    return normalize_label


@registry.misc("yaw.PredicateNormalizer.v1")
def make_predicate_normalizer(
    base_model: str = "en_core_web_lg",
    base_predicates: tuple[str, ...] | list[str] | None = None,
) -> Callable[[str], str]:
    """Canonical predicate normalizer wrapping normalize_predicate with spaCy decamelization and lemmatization."""
    pipe = get_spacy_pipeline(base_model) if base_model else None

    def _normalize(predicate: str) -> str:
        return normalize_predicate(predicate, base_predicates=base_predicates, nlp=pipe)

    return _normalize


__all__ = [
    "CONCEPT_WORDS",
    "POLARITY_EXCLUSION_MASK",
    "PREDICATE_ALIASES",
    "SYNSET_DICTIONARY",
    "DynamicConceptMentionScoper",
    "_TestModelWrapper",
    "_build_single_triple",
    "_check_ner",
    "_entity_definition",
    "_extract_entity_nodes",
    "_extract_triples",
    "_filter_code_spans",
    "_find_best_vector_match",
    "_get_vector_or_doc",
    "_has_named_entity_spans",
    "_lemmatize_phrase",
    "_match_base_or_alias",
    "_model_to_config",
    "_patched_create_pipe",
    "_resolve_base_predicates",
    "get_base_spacy",
    "get_spacy_pipeline",
    "is_antonym_conflict",
    "is_named_entity",
    "make_label_normalizer",
    "make_markdown_structure",
    "make_predicate_normalizer",
    "make_span_reconciler",
    "make_test_model_wrapper",
    "normalize_predicate",
    "normalize_predicate_antonym_guarded",
]
