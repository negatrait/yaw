"""Data models, Doc extensions, and label mappings for NLP extraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from spacy.tokens import Doc, Span, Token

from yaw import config
from yaw.utils import format_proposition


def _extract_rel_fields(item: Any) -> tuple[int | None, int | None, str | None]:
    if isinstance(item, dict):
        return item.get("dep"), item.get("dest"), item.get("relation")
    return (
        getattr(item, "dep", None),
        getattr(item, "dest", None),
        getattr(item, "relation", None),
    )


class RelDict(dict[Any, Any]):
    """Polymorphic container satisfying both dict and list protocols for doc._.rel."""

    def __init__(self, items: Any = None) -> None:
        super().__init__()
        self._items: list[Any] = []
        if items is not None:
            self.extend(items)

    def _add_single_rel(
        self, dep: int | None, dest: int | None, rel: str | None
    ) -> None:
        if dep is not None and dest is not None and rel is not None:
            self.setdefault((dep, dest), {})[str(rel)] = 1.0

    def append(self, item: Any) -> None:
        self._items.append(item)
        dep, dest, rel = _extract_rel_fields(item)
        self._add_single_rel(dep, dest, rel)

    def extend(self, items: Any) -> None:
        if isinstance(items, RelDict):
            self.update(items)
            self._items.extend(items._items)
        elif isinstance(items, dict):
            self.update(items)
            self._load_dict_entries(items)
        elif isinstance(items, (list, tuple)):
            for item in items:
                self.append(item)

    def _load_dict_entries(self, items: dict[Any, Any]) -> None:
        for k, v in items.items():
            if isinstance(k, (tuple, list)) and len(k) == 2:
                dep, dest = k[0], k[1]
                if isinstance(v, dict):
                    for r_name in v:
                        self._items.append(
                            SimpleNamespace(dep=dep, dest=dest, relation=str(r_name))
                        )
                elif isinstance(v, str):
                    self._items.append(SimpleNamespace(dep=dep, dest=dest, relation=v))

    def __iter__(self) -> Any:
        if self._items:
            return iter(self._items)
        return super().__iter__()

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int) and self._items:
            return self._items[key]
        return super().__getitem__(key)

    def __len__(self) -> int:
        if self._items:
            return len(self._items)
        return super().__len__()

    def __bool__(self) -> bool:
        return bool(self._items) or super().__len__() > 0


def _get_rel(doc: Doc) -> RelDict | None:
    val = doc.user_data.get("_yaw_rel")
    if val is None:
        return None
    return val if isinstance(val, RelDict) else RelDict(val)


def _set_rel(doc: Doc, value: Any) -> None:
    if value is None:
        doc.user_data["_yaw_rel"] = None
    else:
        doc.user_data["_yaw_rel"] = RelDict(value)


if not Doc.has_extension("leaf_taxonomy"):
    Doc.set_extension("leaf_taxonomy", default=None)
Doc.set_extension("rel", getter=_get_rel, setter=_set_rel, force=True)
if not Doc.has_extension("llm_reply"):
    Doc.set_extension("llm_reply", default=None)
if not Doc.has_extension("raw_triples"):
    Doc.set_extension("raw_triples", default=None)
if not Token.has_extension("is_code"):
    Token.set_extension("is_code", default=False)
if not Span.has_extension("category"):
    Span.set_extension("category", default=None)


class ExtractedNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    namespace: str
    definition: str = Field(default="", description="Sentential or lexical definition")
    canonical_id: str | None = None

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Namespace cannot be empty")
        return v.strip().lower()


class ExtractedTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: ExtractedNode
    predicate: ExtractedNode
    object: ExtractedNode

    def to_proposition(self) -> str:
        """Format triple into clean declarative proposition."""
        return format_proposition(
            self.subject.name, self.predicate.name, self.object.name
        )


class LeafTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[ExtractedNode]
    triples: list[ExtractedTriple]
    category: str | None = None


GENERIC_LABELS = frozenset({"", "entity", "entities", "misc", "generic"})

LABEL_MAPPINGS: dict[str, str] = {
    "concept": "concepts",
    "concepts": "concepts",
    "entity": "entities",
    "entities": "entities",
    "claim": "claims",
    "claims": "claims",
    "invariant": "invariants",
    "invariants": "invariants",
    "decision": "decisions",
    "decisions": "decisions",
    "process": "processes",
    "processes": "processes",
    "source": "sources",
    "sources": "sources",
    "context": "contexts",
    "contexts": "contexts",
}


def normalize_label(label: str) -> str:
    """Canonical label normalizer mapping singular/plural case-folded entity labels to taxonomy namespaces."""
    cleaned = label.strip().lower()
    return LABEL_MAPPINGS.get(cleaned, cleaned)


def map_namespace(ent: Span) -> str:
    """Map spaCy entity to configured taxonomy namespace."""
    lbl = normalize_label(ent.label_) if ent.label_ else ""
    ns = config.TAXONOMY_NAMESPACES
    if lbl in ns and lbl not in GENERIC_LABELS:
        return lbl
    ent_cat = getattr(getattr(ent, "_", None), "category", None)
    if ent_cat:
        norm_cat = normalize_label(ent_cat)
        if norm_cat in ns:
            return norm_cat
    if "entities" in ns:
        return "entities"
    return ns[0] if ns else (lbl if lbl else "entities")


__all__ = [
    "GENERIC_LABELS",
    "LABEL_MAPPINGS",
    "ExtractedNode",
    "ExtractedTriple",
    "LeafTaxonomy",
    "RelDict",
    "_extract_rel_fields",
    "_get_rel",
    "_set_rel",
    "map_namespace",
    "normalize_label",
]
