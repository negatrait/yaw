"""Stage 1 leaf-taxonomy extraction using spaCy and spacy-llm."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import filter_spans

from yaw import config
from yaw.nlp.client import is_ollama_online
from yaw.nlp.components import (
    CONCEPT_WORDS,
    PREDICATE_ALIASES,
    _build_single_triple,
    _check_ner,
    _entity_definition,
    _extract_entity_nodes,
    _extract_triples,
    _filter_code_spans,
    _find_best_vector_match,
    _get_vector_or_doc,
    _has_named_entity_spans,
    _lemmatize_phrase,
    _match_base_or_alias,
    _model_to_config,
    _patched_create_pipe,
    _resolve_base_predicates,
    _TestModelWrapper,
    get_base_spacy,
    get_spacy_pipeline,
    is_named_entity,
    make_label_normalizer,
    make_markdown_structure,
    make_predicate_normalizer,
    make_span_reconciler,
    make_test_model_wrapper,
    normalize_predicate,
)
from yaw.nlp.examples import CanonicalExampleCollection  # noqa: F401
from yaw.nlp.models import (
    GENERIC_LABELS,
    LABEL_MAPPINGS,
    ExtractedNode,
    ExtractedTriple,
    LeafTaxonomy,
    RelDict,
    _extract_rel_fields,
    _get_rel,
    _set_rel,
    map_namespace,
    normalize_label,
)
from yaw.nlp.parsing import (
    _anchor_span,
    _anchor_triple_in_doc,
    _clean_candidate_dict,
    _get_first_str,
    _parse_json_array,
    _parse_json_object,
    _parse_json_stream,
    _parse_single_candidate,
    _resolve_triple_namespace,
    align_span,
    extract_json_payload,
    parse_extracted_triples,
    parse_raw_triples,
    parse_raw_triples_json,
    parse_raw_triples_to_taxonomy,
)
from yaw.nlp.templates import OPEN_IE_RAW_TRIPLES_TEMPLATE, OPEN_IE_REL_TEMPLATE

logger = logging.getLogger("yaw.leaf_extractor")


def _add_textcat_pipe(
    nlp: Language,
    model: Any | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Add spacy.TextCat.v3 to pipeline with label normalizer."""
    tc_labels = list(
        labels
        if labels is not None
        else [ns for ns in config.TAXONOMY_NAMESPACES if ns != "predicates"]
    )
    if not tc_labels:
        return
    tc_cfg: dict[str, Any] = {
        "validate_types": False,
        "task": {
            "@llm_tasks": "spacy.TextCat.v3",
            "labels": tc_labels,
            "normalizer": {"@misc": "yaw.LabelNormalizer.v1"},
        },
        "model": _model_to_config(model),
    }
    nlp.add_pipe("llm", name="llm_textcat", last=True, config=tc_cfg)


def _add_raw_pipe(
    nlp: Language,
    model: Any | None = None,
    template: str | None = None,
    field: str = "llm_reply",
    cache: dict[str, Any] | None = None,
) -> None:
    """Add spacy.Raw.v1 to pipeline with Open IE raw triples template and optional cache."""
    raw_cfg: dict[str, Any] = {
        "validate_types": False,
        "task": {
            "@llm_tasks": "spacy.Raw.v1",
            "template": template or OPEN_IE_RAW_TRIPLES_TEMPLATE,
            "field": field,
        },
        "model": _model_to_config(model),
    }
    if cache is not None:
        raw_cfg["cache"] = cache
    nlp.add_pipe("llm", name="llm_raw", last=True, config=raw_cfg)


def _add_entity_linker_pipe(
    nlp: Language,
    model: Any | None = None,
    candidate_selector: Any | None = None,
) -> None:
    """Add spacy.EntityLinker.v1 to pipeline and set candidate selector if provided."""
    el_cfg: dict[str, Any] = {
        "validate_types": False,
        "task": {
            "@llm_tasks": "spacy.EntityLinker.v1",
        },
        "model": _model_to_config(model),
    }
    el_pipe = nlp.add_pipe("llm", name="llm_entity_linker", last=True, config=el_cfg)
    if candidate_selector is not None:
        el_task = getattr(el_pipe, "task", None)
        if el_task is not None and hasattr(el_task, "set_candidate_selector"):
            el_task.set_candidate_selector(candidate_selector, nlp.vocab)


def _add_rel_pipe(
    nlp: Language,
    model: Any | None = None,
    labels: list[str] | None = None,
    template: str | None = None,
    fewshot_examples: Any | None = None,
    cache: dict[str, Any] | None = None,
) -> None:
    """Add spacy.REL.v1 to pipeline declaratively with zero monkeypatching."""
    rel_task_cfg: dict[str, Any] = {
        "@llm_tasks": "spacy.REL.v1",
        "labels": labels or list(config.BASE_PREDICATES),
        "template": template or OPEN_IE_REL_TEMPLATE,
    }
    if fewshot_examples is not None:
        rel_task_cfg["examples"] = fewshot_examples
    else:
        rel_task_cfg["examples"] = {"@misc": "yaw.CanonicalExamples.v1"}

    rel_cfg: dict[str, Any] = {
        "validate_types": False,
        "task": rel_task_cfg,
        "model": _model_to_config(model),
    }
    if cache is not None:
        rel_cfg["cache"] = cache
    nlp.add_pipe("llm", name="llm_rel", last=True, config=rel_cfg)


def _load_spacy_base(base_model_name: str | None) -> Language:
    model_name = base_model_name or config.get_spacy_model_name()
    if not is_ollama_online(timeout_s=0.01):
        logger.debug("Ollama service offline during pipeline assembly")
    try:
        return spacy.load(model_name, exclude=["ner"])
    except OSError:
        fallback = config.get_spacy_model_name()
        return spacy.load(fallback, exclude=["ner"])


def _add_optional_pipes(
    nlp: Language,
    model: Any | None,
    cache_cfg: dict[str, Any] | None,
    *,
    textcat_model: Any | None = None,
    textcat_labels: list[str] | None = None,
    enable_entity_linker: bool = False,
    entity_linker_model: Any | None = None,
    candidate_selector: Any | None = None,
    enable_rel: bool = False,
    rel_model: Any | None = None,
    rel_labels: list[str] | None = None,
    rel_examples: Any | None = None,
    fewshot_examples: Any | None = None,
) -> None:
    if textcat_labels is not None or textcat_model is not None:
        _add_textcat_pipe(nlp, textcat_model or model, textcat_labels)

    if enable_entity_linker or candidate_selector is not None:
        _add_entity_linker_pipe(nlp, entity_linker_model or model, candidate_selector)

    rel_ex = fewshot_examples if fewshot_examples is not None else rel_examples
    if (
        enable_rel
        or rel_model is not None
        or rel_labels is not None
        or rel_ex is not None
    ):
        ex_cfg = {"@misc": "yaw.CanonicalExamples.v1"} if rel_ex is True else rel_ex
        _add_rel_pipe(
            nlp,
            model=rel_model or model,
            labels=rel_labels,
            fewshot_examples=ex_cfg,
            cache=cache_cfg,
        )


def doc_to_taxonomy(
    doc: Doc,
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
) -> LeafTaxonomy:
    """Build LeafTaxonomy directly from spaCy doc annotations (ents, rel, cats) or raw LLM output."""
    node_map: dict[str, ExtractedNode] = {}
    ent_nodes = _extract_entity_nodes(doc, node_map)
    triples = _extract_triples(doc, ent_nodes, node_map, base_predicates, nlp)

    # Check for spacy.Raw.v1 output if triples are empty
    raw_reply = getattr(getattr(doc, "_", None), "llm_reply", None)
    if not raw_reply:
        raw_reply = getattr(getattr(doc, "_", None), "raw_triples", None)
    if not triples and raw_reply:
        raw_triples = parse_raw_triples(
            raw_reply, doc=doc, base_predicates=base_predicates, nlp=nlp
        )
        for t in raw_triples:
            for n in (t.subject, t.predicate, t.object):
                key = f"{n.namespace}:{n.name.casefold()}"
                if key not in node_map:
                    node_map[key] = n
            triples.append(t)

    cats = getattr(doc, "cats", None)
    cat = (
        max(cats, key=cats.get)
        if (cats and max(cats.values(), default=0.0) > 0.0)
        else None
    )
    tax = LeafTaxonomy(nodes=list(node_map.values()), triples=triples, category=cat)
    logger.debug(
        "Built LeafTaxonomy with %d nodes, %d triples", len(tax.nodes), len(tax.triples)
    )
    return tax


@Language.factory("leaf_taxonomy_compiler")
def make_leaf_taxonomy_compiler(nlp: Language, name: str):
    """Compiles extracted doc entities and relations into LeafTaxonomy schema."""

    def _compile(doc: Doc) -> Doc:
        code_spans = doc.spans.get("code", [])
        candidate_spans = list(doc.ents) if doc.ents else list(doc.spans.get("sc", []))
        clean_spans = _filter_code_spans(candidate_spans, code_spans)
        doc.ents = filter_spans(clean_spans)
        doc._.leaf_taxonomy = doc_to_taxonomy(doc, nlp=nlp)
        return doc

    return _compile


def build_leaf_pipeline(
    model: Any | None = None,
    cache_dir: Path | str | None = None,
    base_model_name: str | None = None,
    cache_batch_size: int = 1,
    *,
    textcat_model: Any | None = None,
    textcat_labels: list[str] | None = None,
    enable_entity_linker: bool = False,
    entity_linker_model: Any | None = None,
    candidate_selector: Any | None = None,
    enable_rel: bool = False,
    rel_model: Any | None = None,
    rel_labels: list[str] | None = None,
    rel_examples: Any | None = None,
    fewshot_examples: Any | None = None,
    kb: Any | None = None,
    **kwargs: Any,
) -> Language:
    """Builds a single unified spaCy Language pipeline with markdown_structure,
    spacy.Raw.v1 Open IE task, mounted spacy.BatchCache.v1, and leaf_taxonomy_compiler.
    """
    nlp = _load_spacy_base(base_model_name)
    nlp.add_pipe("markdown_structure", first=True)

    cache_path = Path(cache_dir or config.SPACY_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_cfg: dict[str, Any] = {
        "@llm_misc": "spacy.BatchCache.v1",
        "path": str(cache_path),
        "batch_size": cache_batch_size,
        "max_batches_in_mem": 4,
    }
    raw_cache_cfg = cache_cfg
    rel_cache_cfg = (
        dict(cache_cfg, path=str(cache_path / "rel")) if cache_dir is not None else None
    )

    raw_model = kwargs.get("raw_model") or model
    _add_raw_pipe(nlp, model=raw_model, cache=raw_cache_cfg)

    _add_optional_pipes(
        nlp,
        model=model,
        cache_cfg=rel_cache_cfg,
        textcat_model=textcat_model,
        textcat_labels=textcat_labels,
        enable_entity_linker=enable_entity_linker,
        entity_linker_model=entity_linker_model,
        candidate_selector=candidate_selector,
        enable_rel=enable_rel,
        rel_model=rel_model,
        rel_labels=rel_labels,
        rel_examples=rel_examples,
        fewshot_examples=fewshot_examples,
    )

    if kb is not None:
        nlp.add_pipe("kb_entity_disambiguator", config={"kb": kb})

    nlp.add_pipe("leaf_taxonomy_compiler", last=True)
    return nlp


@functools.cache
def get_leaf_pipeline() -> Language:
    """Returns a singleton instance of the leaf extraction pipeline."""
    return build_leaf_pipeline()


def extract_leaf_doc(text: str, nlp: Language | None = None) -> Doc:
    """Processes text through spaCy pipeline, returning annotated Doc."""
    pipeline = nlp or get_leaf_pipeline()
    return pipeline(text)


def extract_leaf_taxonomy(
    target: str | Doc, nlp: Language | None = None
) -> LeafTaxonomy:
    """Extracts a validated LeafTaxonomy schema from text or a processed Doc."""
    try:
        doc = target if isinstance(target, Doc) else extract_leaf_doc(target, nlp=nlp)
        return doc._.leaf_taxonomy or doc_to_taxonomy(doc, nlp=nlp)
    except (ConnectionError, TimeoutError, OSError) as net_err:
        logger.warning(
            "Extraction pipeline network offline (%s); falling back to base annotations",
            net_err,
        )
        base = nlp if nlp is not None else get_base_spacy()
        doc = base.make_doc(target if isinstance(target, str) else target.text)
        return doc_to_taxonomy(doc, nlp=base)


__all__ = [
    "CONCEPT_WORDS",
    "GENERIC_LABELS",
    "LABEL_MAPPINGS",
    "PREDICATE_ALIASES",
    "ExtractedNode",
    "ExtractedTriple",
    "LeafTaxonomy",
    "RelDict",
    "_TestModelWrapper",
    "_add_entity_linker_pipe",
    "_add_optional_pipes",
    "_add_raw_pipe",
    "_add_rel_pipe",
    "_add_textcat_pipe",
    "_anchor_span",
    "_anchor_triple_in_doc",
    "_build_single_triple",
    "_check_ner",
    "_clean_candidate_dict",
    "_entity_definition",
    "_extract_entity_nodes",
    "_extract_rel_fields",
    "_extract_triples",
    "_filter_code_spans",
    "_find_best_vector_match",
    "_get_first_str",
    "_get_rel",
    "_get_vector_or_doc",
    "_has_named_entity_spans",
    "_lemmatize_phrase",
    "_load_spacy_base",
    "_match_base_or_alias",
    "_model_to_config",
    "_parse_json_array",
    "_parse_json_object",
    "_parse_json_stream",
    "_parse_single_candidate",
    "_patched_create_pipe",
    "_resolve_base_predicates",
    "_resolve_triple_namespace",
    "_set_rel",
    "align_span",
    "build_leaf_pipeline",
    "doc_to_taxonomy",
    "extract_json_payload",
    "extract_leaf_doc",
    "extract_leaf_taxonomy",
    "get_base_spacy",
    "get_leaf_pipeline",
    "get_spacy_pipeline",
    "is_named_entity",
    "make_label_normalizer",
    "make_leaf_taxonomy_compiler",
    "make_markdown_structure",
    "make_predicate_normalizer",
    "make_span_reconciler",
    "make_test_model_wrapper",
    "map_namespace",
    "normalize_label",
    "normalize_predicate",
    "parse_extracted_triples",
    "parse_raw_triples",
    "parse_raw_triples_json",
    "parse_raw_triples_to_taxonomy",
]
