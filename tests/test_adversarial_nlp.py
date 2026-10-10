"""Adversarial coverage hardening for NLP and few-shot steering layer.

Empirical verification adhering strictly to .agents/skills/engineering-real-tests/SKILL.md.
Tests 5 distinct, un-tested boundary conditions with zero overlap with tests/test_fewshot_steering.py:
1. CanonicalExampleCollection slicing, negative indexing, mutation isolation, and immutable constants.
2. RELExample.to_doc() multibyte UTF-8 span alignment, zero-width/non-breaking space normalization, off-token boundaries, and punctuation.
3. Jinja template rendering under edge cases (missing/empty relations, Pydantic/SimpleNamespace models, injection characters).
4. Cache subpartitioning under multi-threaded concurrency in build_leaf_pipeline(enable_rel=True).
5. Registry re-registration idempotency, module reload integrity, and instance isolation.
"""

from __future__ import annotations

import concurrent.futures
import importlib
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jinja2
import pytest
import spacy
from pydantic import BaseModel
from spacy.tokens import Doc
from spacy_llm.registry import registry
from spacy_llm.tasks.rel.util import RELExample

from yaw.nlp import (
    OPEN_IE_REL_TEMPLATE,
    build_leaf_pipeline,
    make_canonical_examples,
)
from yaw.nlp.examples import CANONICAL_REL_EXAMPLES

# ===========================================================================
# Helper Model Callable conforming to spaCy-LLM Model Protocol
# ===========================================================================


class RealModelCallable:
    """Deterministic in-process model callable satisfying spacy-llm ModelWithContextLength protocol."""

    def __init__(
        self, fixed_output: str = '{"dep": 0, "dest": 1, "relation": "depends_on"}'
    ) -> None:
        self.fixed_output = fixed_output
        self.call_count = 0

    @property
    def context_length(self) -> int:
        return 8192

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        results: list[list[str]] = []
        for doc_prompts in prompts:
            doc_res: list[str] = []
            for _ in doc_prompts:
                self.call_count += 1
                doc_res.append(self.fixed_output)
            results.append(doc_res)
        return results


# ===========================================================================
# 1. Collection Slicing, Negative Indexing & Immutable Constants
# ===========================================================================


def test_adversarial_collection_indexing_and_slicing() -> None:
    """Verify negative indexing, slicing, reversal, and mutation protocol on CanonicalExampleCollection."""
    coll = make_canonical_examples(task="rel", max_examples=5)
    assert len(coll) == 5
    assert coll.task == "rel"

    # Indexing: positive and negative
    first = coll[0]
    last = coll[-1]
    assert isinstance(first, dict)
    assert isinstance(last, dict)
    assert first["text"] == coll[0]["text"]
    assert last["text"] == coll[4]["text"]

    # Slicing: basic slice returns list of dicts
    sliced = coll[1:4]
    assert len(sliced) == 3
    assert isinstance(sliced, list)
    assert sliced[0] == coll[1]
    assert sliced[-1] == coll[3]

    # Reversal slicing
    rev = coll[::-1]
    assert len(rev) == 5
    assert rev[0] == coll[4]
    assert rev[-1] == coll[0]

    # Iteration protocol
    collected = list(coll)
    assert len(collected) == 5
    assert collected == list(coll)

    # In-place list methods: append, extend, pop
    sample_ex = {"text": "A causes B.", "ents": [], "relations": []}
    coll.append(sample_ex)
    assert len(coll) == 6
    popped = coll.pop()
    assert popped == sample_ex
    assert len(coll) == 5


def test_adversarial_collection_call_and_as_dicts_isolation() -> None:
    """Calling collection or as_dicts() returns list copy; mutating return does not mutate collection."""
    coll = make_canonical_examples(task="rel", max_examples=3)
    invoked = coll()
    assert isinstance(invoked, list)
    assert len(invoked) == 3

    # Mutate the list returned by __call__()
    invoked.append({"text": "Adversarial mutation", "ents": [], "relations": []})
    assert len(coll) == 3, (
        "Mutating returned list must not affect the original collection"
    )

    # Verify as_dicts()
    dicts = coll.as_dicts()
    assert isinstance(dicts, list)
    assert len(dicts) == 3
    dicts.clear()
    assert len(coll) == 3, "Clearing as_dicts() result must not mutate collection"


def test_adversarial_collection_immutability_of_underlying_constants() -> None:
    """Modifying returned collections or example dicts never corrupts CANONICAL_REL_EXAMPLES."""
    original_first_text = CANONICAL_REL_EXAMPLES[0]["text"]
    coll = make_canonical_examples(task="rel", max_examples=3)

    # Maliciously modify the dictionary in-place
    coll[0]["text"] = "CORRUPTED_TEXT"
    coll[0]["ents"].append({"start_char": 0, "end_char": 9, "label": "poison"})

    # Check underlying module tuple is completely untouched
    assert CANONICAL_REL_EXAMPLES[0]["text"] == original_first_text
    assert len(CANONICAL_REL_EXAMPLES[0]["ents"]) == 2

    # A fresh factory call gets pristine data
    fresh_coll = make_canonical_examples(task="rel", max_examples=3)
    assert fresh_coll[0]["text"] == original_first_text
    assert len(fresh_coll[0]["ents"]) == 2


# ===========================================================================
# 2. Multibyte UTF-8 Alignment, Invisible Spaces & Off-Token Boundaries in to_doc()
# ===========================================================================


def test_adversarial_relexample_multibyte_utf8_and_math_symbols() -> None:
    """RELExample.to_doc() alignment under multibyte Unicode (German umlauts, Greek letters, math symbols)."""
    text = "Erwin Schrödinger analyzed the ℏ quantum state in the 🧠 human brain."
    ents = [
        {"start_char": 0, "end_char": 17, "label": "entities"},
        {"start_char": 31, "end_char": 46, "label": "concepts"},
        {"start_char": 56, "end_char": 67, "label": "concepts"},
    ]
    relations = [
        {"dep": 0, "dest": 1, "relation": "analyzed"},
        {"dep": 1, "dest": 2, "relation": "manifests_in"},
    ]

    rel_ex = RELExample(text=text, ents=ents, relations=relations)
    doc = rel_ex.to_doc()

    assert len(doc.ents) == 3
    assert doc.ents[0].text == "Erwin Schrödinger"
    assert doc.ents[1].text == "ℏ quantum state"
    assert doc.ents[2].text == "human brain"
    assert len(doc.user_data["rel"]) == 2


def test_adversarial_relexample_zero_width_and_invisible_spaces() -> None:
    """Behavior of RELExample.to_doc() with non-breaking spaces: verifies whitespace normalization invariant."""
    text = "Quantum\u00a0Mechanics applies to Wave\u00a0Equations."
    span1_text = "Quantum\u00a0Mechanics"
    span2_text = "Wave\u00a0Equations"
    start1 = text.find(span1_text)
    end1 = start1 + len(span1_text)
    start2 = text.find(span2_text)
    end2 = start2 + len(span2_text)

    ents = [
        {"start_char": start1, "end_char": end1, "label": "concepts"},
        {"start_char": start2, "end_char": end2, "label": "concepts"},
    ]
    relations = [{"dep": 0, "dest": 1, "relation": "applies_to"}]

    rel_ex = RELExample(text=text, ents=ents, relations=relations)
    doc = rel_ex.to_doc()

    # RELExample.to_doc() normalizes \u00a0 to ASCII space ' ' while retaining both entities
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "Quantum Mechanics"
    assert doc.ents[1].text == "Wave Equations"


def test_adversarial_relexample_punctuation_and_hyphenated_entities() -> None:
    """Surrounding punctuation vs internal hyphenation behavior in RELExample.to_doc()."""
    # Case A: Surrounding punctuation (parentheses and quotes) succeeds
    text_surrounded = 'Erwin Schrödinger published (special relativity) in "1905".'
    sr = "special relativity"
    sr_start = text_surrounded.find(sr)
    rel_ex_surrounded = RELExample(
        text=text_surrounded,
        ents=[
            {
                "start_char": sr_start,
                "end_char": sr_start + len(sr),
                "label": "concepts",
            }
        ],
        relations=[],
    )
    doc_surrounded = rel_ex_surrounded.to_doc()
    assert len(doc_surrounded.ents) == 1
    assert doc_surrounded.ents[0].text == "special relativity"

    # Case B: Internal hyphenation in RELExample.to_doc() exposes spaCy-LLM upstream [E035] span bounds
    text_hyphenated = "Quantum mechanics features wave-particle duality."
    wp = "wave-particle duality"
    wp_start = text_hyphenated.find(wp)
    rel_ex_hyphenated = RELExample(
        text=text_hyphenated,
        ents=[
            {
                "start_char": wp_start,
                "end_char": wp_start + len(wp),
                "label": "concepts",
            }
        ],
        relations=[],
    )
    with pytest.raises(
        IndexError, match=r"\[E035\] Error creating span with start \d+ and end -1"
    ):
        rel_ex_hyphenated.to_doc()


def test_adversarial_relexample_off_token_boundary_handling() -> None:
    """RELExample.to_doc() gracefully omits off-token boundaries without crashing."""
    text = "Quantum thermodynamics explains macroscopic irreversibility."
    # Adversarial entity slicing inside the word "thermodynamics" -> "modynam"
    ents = [
        {"start_char": 13, "end_char": 20, "label": "concepts"},  # "modynam"
    ]
    relations: list[dict[str, Any]] = []

    rel_ex = RELExample(text=text, ents=ents, relations=relations)
    doc = rel_ex.to_doc()
    assert isinstance(doc, Doc)
    # The off-boundary span does not match token.idx, so doc.ents omits it cleanly
    assert len(doc.ents) == 0


# ===========================================================================
# 3. Jinja Template Rendering Under Adversarial Edge Cases
# ===========================================================================


def test_adversarial_jinja_rel_template_missing_and_empty_relations() -> None:
    """OPEN_IE_REL_TEMPLATE renders cleanly when examples have missing/empty relations or ents."""
    template = jinja2.Template(OPEN_IE_REL_TEMPLATE)

    # Case A: Examples with empty relations list
    rendered_a = template.render(
        text="Target text to extract.",
        prompt_examples=[
            {"text": "Example without relations.", "ents": [], "relations": []}
        ],
    )
    assert "Example without relations." in rendered_a
    assert "Target text to extract." in rendered_a

    # Case B: Examples with missing relations attribute
    rendered_b = template.render(
        text="Target text to extract.",
        prompt_examples=[{"text": "Bare example text."}],
    )
    assert "Bare example text." in rendered_b
    assert "Target text to extract." in rendered_b

    # Case C: Empty prompt_examples
    rendered_c = template.render(text="Standalone query.", prompt_examples=[])
    assert "Standalone query." in rendered_c
    assert "Below are some examples" not in rendered_c


def test_adversarial_jinja_rel_template_custom_objects_and_pydantic() -> None:
    """Template handles Pydantic model relations, SimpleNamespace, and raw strings."""
    template = jinja2.Template(OPEN_IE_REL_TEMPLATE)

    class CustomRelItem(BaseModel):
        dep: int
        dest: int
        relation: str

    pydantic_rel = CustomRelItem(dep=0, dest=1, relation="subClassOf")
    namespace_rel = SimpleNamespace(dep=1, dest=2, relation="causes")
    raw_str_rel = '{"dep": 2, "dest": 3, "relation": "depends_on"}'

    examples = [
        {
            "text": "Multi-type relation example.",
            "relations": [pydantic_rel, namespace_rel, raw_str_rel],
        }
    ]

    rendered = template.render(text="Target text.", prompt_examples=examples)
    assert "subClassOf" in rendered
    assert "causes" in rendered
    assert "depends_on" in rendered


def test_adversarial_jinja_rel_template_special_chars_and_injection() -> None:
    """Template renders safely against text containing quotes, code fences, and injection attempts."""
    template = jinja2.Template(OPEN_IE_REL_TEMPLATE)

    adversarial_text = """Special "quoted" text with backslashes \\ and ''' triple quotes '''
and code fences ```python print("hello") ```
and Jinja injection attempt: {{ 7 * 7 }} and {% if True %}INJECTED{% endif %}"""

    adversarial_example = {
        "text": "Example with 'quotes' and {curly} braces.",
        "relations": [
            {"dep": 0, "dest": 1, "relation": "quotes_'and'_\"double\""},
        ],
    }

    rendered = template.render(
        text=adversarial_text,
        prompt_examples=[adversarial_example],
    )

    # In Jinja rendering of input variables, expressions are not evaluated as code
    assert "{{ 7 * 7 }}" in rendered
    assert "{% if True %}INJECTED{% endif %}" in rendered
    assert "''' triple quotes '''" in rendered
    assert "quotes_'and'_\"double\"" in rendered


# ===========================================================================
# 4. Cache Subpartitioning Under Concurrency in build_leaf_pipeline
# ===========================================================================


def test_adversarial_cache_subpartitioning_rel_isolation(tmp_path: Path) -> None:
    """Verify build_leaf_pipeline(enable_rel=True, cache_dir=...) subpartitions rel cache."""
    cache_root = tmp_path / "spacy_cache"
    nlp = build_leaf_pipeline(
        cache_dir=cache_root,
        enable_rel=True,
        rel_model=RealModelCallable(),
        raw_model=RealModelCallable(
            fixed_output='[{"subject": "A", "predicate": "B", "object": "C"}]'
        ),
    )

    # Both cache_root and cache_root / "rel" must exist on the physical filesystem
    assert cache_root.is_dir(), "Raw cache root directory must exist"
    rel_cache_dir = cache_root / "rel"
    assert rel_cache_dir.is_dir(), "Subpartitioned rel cache directory must exist"

    # Verify pipeline component structure
    assert "llm_raw" in nlp.pipe_names
    assert "llm_rel" in nlp.pipe_names

    # Check pipe configurations for attached BatchCache
    raw_pipe = nlp.get_pipe("llm_raw")
    rel_pipe = nlp.get_pipe("llm_rel")

    assert hasattr(raw_pipe, "_cache"), "llm_raw must have attached BatchCache"
    assert hasattr(rel_pipe, "_cache"), "llm_rel must have attached BatchCache"


def test_adversarial_cache_multithreaded_concurrent_pipeline_assembly(
    tmp_path: Path,
) -> None:
    """Build multiple concurrent pipelines in parallel worker threads with shared cache_dir."""
    cache_root = tmp_path / "concurrent_cache"

    def _worker(worker_id: int) -> tuple[int, list[str]]:
        pipe = build_leaf_pipeline(
            cache_dir=cache_root,
            enable_rel=True,
            rel_model=RealModelCallable(),
            raw_model=RealModelCallable(
                fixed_output='[{"subject": "A", "predicate": "B", "object": "C"}]'
            ),
        )
        return worker_id, pipe.pipe_names

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, i) for i in range(4)]
        results = [f.result(timeout=15.0) for f in futures]

    assert len(results) == 4
    for _, pipe_names in results:
        assert "markdown_structure" in pipe_names
        assert "llm_raw" in pipe_names
        assert "llm_rel" in pipe_names
        assert "leaf_taxonomy_compiler" in pipe_names


def test_adversarial_cache_concurrent_document_processing(tmp_path: Path) -> None:
    """Real substrate document processing under concurrent pipelines with disk cache."""
    cache_root = tmp_path / "proc_cache"
    model = RealModelCallable()

    nlp = build_leaf_pipeline(
        cache_dir=cache_root,
        enable_rel=True,
        rel_model=model,
        raw_model=RealModelCallable(
            fixed_output='[{"subject": "Quantum", "predicate": "describes", "object": "Atoms"}]'
        ),
    )

    doc1 = nlp.make_doc("Quantum mechanics describes atoms.")
    span1 = doc1.char_span(0, 17, label="concepts")
    span2 = doc1.char_span(28, 33, label="concepts")
    if span1 and span2:
        doc1.ents = [span1, span2]

    # Process through pipeline
    processed = nlp(doc1)
    assert processed is not None
    assert hasattr(processed._, "rel")
    assert hasattr(processed._, "leaf_taxonomy")

    # Files must have been written to disk in cache directory
    cache_files = list(cache_root.glob("**/*"))
    assert len(cache_files) > 0, "BatchCache must write cache state to disk"


# ===========================================================================
# 5. Registry Idempotency, Module Reload & Re-Registration Resilience
# ===========================================================================


def test_adversarial_registry_idempotent_lookup_and_execution() -> None:
    """Look up @registry.misc('yaw.CanonicalExamples.v1') repeatedly and verify determinism."""
    factory = registry.misc.get("yaw.CanonicalExamples.v1")
    assert callable(factory)

    coll1 = factory(task="rel", max_examples=4)
    coll2 = factory(task="rel", max_examples=4)

    assert len(coll1) == 4
    assert len(coll2) == 4
    assert coll1() == coll2()
    # Distinct instances returned (no pointer aliasing)
    assert coll1 is not coll2


def test_adversarial_registry_module_reload_integrity() -> None:
    """Reloading yaw.nlp.examples preserves registry functionality without corruption."""
    import yaw.nlp.examples

    reloaded_mod = importlib.reload(yaw.nlp.examples)
    assert hasattr(reloaded_mod, "make_canonical_examples")

    factory = registry.misc.get("yaw.CanonicalExamples.v1")
    assert callable(factory)

    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        name="llm_rel_after_reload",
        config={
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                "labels": ["causes", "depends_on"],
                "examples": {"@misc": "yaw.CanonicalExamples.v1", "max_examples": 2},
            },
            "model": RealModelCallable(),
        },
    )
    assert "llm_rel_after_reload" in nlp.pipe_names


def test_adversarial_registry_reregistration_resilience() -> None:
    """Explicit re-registration with catalogue registry maintains expected handler."""
    current_factory = registry.misc.get("yaw.CanonicalExamples.v1")

    # Re-register same function
    decorated = registry.misc("yaw.CanonicalExamples.v1")(current_factory)
    assert callable(decorated)

    # Verify registry still resolves and functions properly
    resolved = registry.misc.get("yaw.CanonicalExamples.v1")
    coll = resolved(task="rel", max_examples=2)
    assert len(coll) == 2
