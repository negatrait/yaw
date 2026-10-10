"""Canonical few-shot examples registry and dynamic vocabulary steering for spaCy-LLM."""

from __future__ import annotations

import copy
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spacy_llm.registry import registry

from yaw import config

NOISY_PREDICATES: frozenset[str] = frozenset(
    {
        "is_a",
        "isa",
        "child_of",
        "childof",
        "kind_of",
        "kindof",
        "type_of",
        "typeof",
        "instance_of",
        "instanceof",
        "wikilink",
        "relates_to",
        "related_to",
    }
)

CANONICAL_REL_EXAMPLES: tuple[dict[str, Any], ...] = (
    # 1. subClassOf
    {
        "text": "Quantum mechanics is a subClassOf theoretical physics.",
        "ents": [
            {"start_char": 0, "end_char": 17, "label": "concepts"},
            {"start_char": 34, "end_char": 53, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "subClassOf"}],
    },
    # 2. partOf
    {
        "text": "The hippocampus is partOf the limbic system in the brain.",
        "ents": [
            {"start_char": 4, "end_char": 15, "label": "concepts"},
            {"start_char": 30, "end_char": 43, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "partOf"}],
    },
    # 3. manifestsIn
    {
        "text": "General relativity manifests in gravitational lensing observations.",
        "ents": [
            {"start_char": 0, "end_char": 18, "label": "concepts"},
            {"start_char": 32, "end_char": 53, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "manifestsIn"}],
    },
    # 4. mentions
    {
        "text": "The research monograph mentions quantum entanglement.",
        "ents": [
            {"start_char": 4, "end_char": 22, "label": "sources"},
            {"start_char": 32, "end_char": 52, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "mentions"}],
    },
    # 5. derivedFrom
    {
        "text": "Special relativity is derived from the 1905 electrodynamics paper.",
        "ents": [
            {"start_char": 0, "end_char": 18, "label": "concepts"},
            {"start_char": 39, "end_char": 65, "label": "sources"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "derivedFrom"}],
    },
    # 6. instanceOf
    {
        "text": "Albert Einstein is an instance of theoretical physicist.",
        "ents": [
            {"start_char": 0, "end_char": 15, "label": "entities"},
            {"start_char": 34, "end_char": 55, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "instanceOf"}],
    },
    # 7. refutes
    {
        "text": "Empirical observation refutes the steady state cosmological model.",
        "ents": [
            {"start_char": 0, "end_char": 21, "label": "claims"},
            {"start_char": 34, "end_char": 65, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "refutes"}],
    },
    # 8. produces
    {
        "text": "Nuclear fusion produces stellar energy in main sequence stars.",
        "ents": [
            {"start_char": 0, "end_char": 14, "label": "processes"},
            {"start_char": 24, "end_char": 38, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "produces"}],
    },
    # 9. causes
    {
        "text": "Mass energy density causes spacetime curvature.",
        "ents": [
            {"start_char": 0, "end_char": 19, "label": "concepts"},
            {"start_char": 27, "end_char": 46, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "causes"}],
    },
    # 10. supersedes
    {
        "text": "General relativity supersedes Newtonian gravitation for strong fields.",
        "ents": [
            {"start_char": 0, "end_char": 18, "label": "concepts"},
            {"start_char": 30, "end_char": 51, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "supersedes"}],
    },
    # 11. prevents
    {
        "text": "Degeneracy pressure prevents white dwarf gravitational collapse.",
        "ents": [
            {"start_char": 0, "end_char": 19, "label": "concepts"},
            {"start_char": 29, "end_char": 63, "label": "processes"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "prevents"}],
    },
    # 12. supports
    {
        "text": "Mercury perihelion precession supports general relativity.",
        "ents": [
            {"start_char": 0, "end_char": 29, "label": "claims"},
            {"start_char": 39, "end_char": 57, "label": "concepts"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "supports"}],
    },
    # 13. contradicts
    {
        "text": "Classical determinism contradicts quantum uncertainty.",
        "ents": [
            {"start_char": 0, "end_char": 21, "label": "claims"},
            {"start_char": 34, "end_char": 53, "label": "invariants"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "contradicts"}],
    },
    # 14. cites
    {
        "text": "The research article cites earlier observational surveys.",
        "ents": [
            {"start_char": 4, "end_char": 20, "label": "sources"},
            {"start_char": 27, "end_char": 56, "label": "sources"},
        ],
        "relations": [{"dep": 0, "dest": 1, "relation": "cites"}],
    },
)

CANONICAL_SPANCAT_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "text": "Albert Einstein published the special relativity treatise in 1905.",
        "entities": {
            "entities": ["Albert Einstein"],
            "concepts": ["special relativity"],
            "sources": ["special relativity treatise"],
        },
        "spans": [
            {
                "text": "Albert Einstein",
                "is_entity": True,
                "label": "entities",
                "reason": "historical physicist and person",
            },
            {
                "text": "special relativity",
                "is_entity": True,
                "label": "concepts",
                "reason": "fundamental theoretical physics framework",
            },
            {
                "text": "special relativity treatise",
                "is_entity": True,
                "label": "sources",
                "reason": "scientific monograph published in 1905",
            },
        ],
    },
    {
        "text": "The Pauli exclusion principle mandates that no two identical fermions occupy the same quantum state.",
        "entities": {
            "invariants": ["Pauli exclusion principle"],
            "entities": ["fermions"],
            "concepts": ["quantum state"],
        },
        "spans": [
            {
                "text": "Pauli exclusion principle",
                "is_entity": True,
                "label": "invariants",
                "reason": "immutable physical law and quantum invariant",
            },
            {
                "text": "fermions",
                "is_entity": True,
                "label": "entities",
                "reason": "subatomic particle class subject to Fermi-Dirac statistics",
            },
            {
                "text": "quantum state",
                "is_entity": True,
                "label": "concepts",
                "reason": "mathematical entity describing a physical system",
            },
        ],
    },
    {
        "text": "The Copenhagen interpretation asserts that physical systems remain in indeterminate superpositions until measurement.",
        "entities": {
            "decisions": ["Copenhagen interpretation"],
            "claims": ["physical systems remain in indeterminate superpositions"],
        },
        "spans": [
            {
                "text": "Copenhagen interpretation",
                "is_entity": True,
                "label": "decisions",
                "reason": "methodological and philosophical interpretation choice",
            },
            {
                "text": "physical systems remain in indeterminate superpositions",
                "is_entity": True,
                "label": "claims",
                "reason": "epistemological thesis asserted by Copenhagen theorists",
            },
        ],
    },
    {
        "text": "Data preprocessing and tokenization precede model training during pipeline execution in laboratory environments.",
        "entities": {
            "processes": ["Data preprocessing", "model training"],
            "contexts": ["laboratory environments"],
        },
        "spans": [
            {
                "text": "Data preprocessing",
                "is_entity": True,
                "label": "processes",
                "reason": "sequence of data cleaning operations",
            },
            {
                "text": "model training",
                "is_entity": True,
                "label": "processes",
                "reason": "computational optimization process",
            },
            {
                "text": "laboratory environments",
                "is_entity": True,
                "label": "contexts",
                "reason": "operational environment and domain context",
            },
        ],
    },
    {
        "text": "The ontology defines subClassOf as the canonical subsumption relation between types.",
        "entities": {
            "predicates": ["subClassOf"],
            "concepts": ["subsumption relation"],
        },
        "spans": [
            {
                "text": "subClassOf",
                "is_entity": True,
                "label": "predicates",
                "reason": "canonical taxonomy relation predicate",
            },
            {
                "text": "subsumption relation",
                "is_entity": True,
                "label": "concepts",
                "reason": "abstract ontological relationship",
            },
        ],
    },
)


class CanonicalExampleCollection(list[dict[str, Any]]):
    """Polymorphic container behaving both as a Sequence[dict] and Callable[[], list[dict]].

    Satisfies spacy-llm tasks calling `examples() if callable(examples) else examples`.
    """

    def __init__(
        self,
        items: Iterable[dict[str, Any]] | None = None,
        task: str = "rel",
    ) -> None:
        super().__init__(copy.deepcopy(list(items)) if items else [])
        self.task = task

    def __call__(self) -> list[dict[str, Any]]:
        """Return a shallow copy of contained example dictionaries."""
        return list(self)

    def as_dicts(self) -> list[dict[str, Any]]:
        """Explicitly return a shallow copy of inner dictionary instances."""
        return list(self)

    def __repr__(self) -> str:
        return f"CanonicalExampleCollection(task={self.task!r}, count={len(self)})"


def query_live_predicates(db_path: str | Path, limit: int = 20) -> list[str]:
    """Query top distinct edge predicates from unified_edges in SQLite.

    Opens the database in read-only mode to adhere to Invariant 1.
    Gracefully handles uninitialized databases or missing views.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        return []

    uri = f"file:{path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT rel_type, COUNT(*) as cnt
                FROM unified_edges
                WHERE rel_type IS NOT NULL AND TRIM(rel_type) != ''
                GROUP BY rel_type
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (limit * 2,),
            )
            rows = cursor.fetchall()
    except sqlite3.OperationalError, sqlite3.DatabaseError:
        return []

    predicates: list[str] = []
    for row in rows:
        pred = str(row[0]).strip()
        if pred.lower() not in NOISY_PREDICATES and pred not in predicates:
            predicates.append(pred)
            if len(predicates) >= limit:
                break
    return predicates


def _clean_active_predicates(active_predicates: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for p in active_predicates:
        name = str(p).strip()
        if name and name not in seen and name.lower() not in NOISY_PREDICATES:
            seen.add(name)
            cleaned.append(name)
    return cleaned


def _example_matches_predicate(ex: dict[str, Any], predicate: str) -> bool:
    pred_lower = predicate.lower()
    for rel in ex.get("relations", []):
        name = str(rel.get("relation", "")).strip()
        if name == predicate or name.lower() == pred_lower:
            return True
    return False


def prioritize_rel_examples(
    base_examples: Iterable[dict[str, Any]],
    active_predicates: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Order base examples prioritizing those matching active_predicates."""
    examples_list = [copy.deepcopy(ex) for ex in base_examples]
    if not active_predicates:
        return examples_list

    ordered_preds = _clean_active_predicates(active_predicates)
    prioritized: list[dict[str, Any]] = []
    used_indices: set[int] = set()

    for p in ordered_preds:
        for idx, ex in enumerate(examples_list):
            if idx not in used_indices and _example_matches_predicate(ex, p):
                prioritized.append(ex)
                used_indices.add(idx)

    for idx, ex in enumerate(examples_list):
        if idx not in used_indices:
            prioritized.append(ex)

    return prioritized


def _resolve_live_predicates(
    include_live: bool,
    active_predicates: list[str] | tuple[str, ...] | None,
    db_path: str | Path | None,
) -> list[str]:
    if not include_live:
        return []
    if active_predicates is not None:
        return _clean_active_predicates(active_predicates)
    if db_path is not None:
        return query_live_predicates(db_path, limit=20)
    return []


@registry.misc("yaw.CanonicalExamples.v1")
def make_canonical_examples(
    task: str = "rel",
    *,
    task_type: str | None = None,
    max_examples: int = 5,
    limit: int | None = None,
    db_path: str | Path | None = None,
    include_live: bool = True,
    active_predicates: list[str] | tuple[str, ...] | None = None,
) -> CanonicalExampleCollection:
    """Factory returning few-shot examples for spacy.REL.v1 or spacy.SpanCat.v3.

    Args:
        task: Target task type ("rel", "spancat", "spancat_cot").
        task_type: Alias for task to support alternative configuration keys.
        max_examples: Maximum number of examples to include (capped by CANON_FEWSHOT_MAX).
        limit: Alias for max_examples.
        db_path: Optional path to SQLite vectors.db for live vocabulary enrichment.
        include_live: Whether to blend active database predicates with base canonical examples.
        active_predicates: Injected list of active predicates to prioritize.

    Returns:
        CanonicalExampleCollection instance containing formatted few-shot dictionaries.

    Raises:
        ValueError: If task is unsupported.
    """
    target_task = (task_type or task or "rel").strip().lower()

    raw_max = limit if limit is not None else max_examples
    canon_cap = getattr(config, "CANON_FEWSHOT_MAX", 5)
    effective_max = max(0, min(raw_max, canon_cap))

    if target_task in ("rel", "spacy.rel.v1", "relations"):
        live_preds = _resolve_live_predicates(include_live, active_predicates, db_path)
        ordered = prioritize_rel_examples(CANONICAL_REL_EXAMPLES, live_preds)
        return CanonicalExampleCollection(ordered[:effective_max], task="rel")

    if target_task in ("spancat", "spacy.spancat.v3", "spancat_cot", "ner"):
        selected = [
            copy.deepcopy(ex) for ex in CANONICAL_SPANCAT_EXAMPLES[:effective_max]
        ]
        return CanonicalExampleCollection(selected, task="spancat")

    raise ValueError(
        f"Unsupported task type: '{target_task}'. Supported: 'rel', 'spancat'."
    )
