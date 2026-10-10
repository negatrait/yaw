"""Thread-safe in-memory registry of active ontology namespaces, prefixes, and axioms."""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import ClassVar, Self, cast

from rdflib import URIRef

from yaw.storage.rdf import ROOT_NAMESPACES, W3C_NAMESPACES, YAW_PREFIX


@dataclass(frozen=True)
class DynamicClassMeta:
    curie: str
    uri: URIRef
    ontology_id: str
    label: str
    parent_uri: URIRef | None
    depth: int = 0
    is_root: bool = False
    epistemic_type: str = "class"  # 'class' (OWL) or 'concept' (SKOS)


@dataclass(frozen=True)
class DynamicPropertyMeta:
    curie: str
    uri: URIRef
    ontology_id: str
    label: str
    parent_uri: URIRef | None
    is_transitive: bool = False
    is_symmetric: bool = False


@dataclass(frozen=True)
class RegistrySnapshot:
    """Immutable state snapshot for thread-safe lock-free reads."""

    prefix_to_uri: dict[str, str]
    uri_to_prefix: dict[str, str]
    concept_prefixes: frozenset[str]
    classes: dict[str, DynamicClassMeta]
    properties: dict[str, DynamicPropertyMeta]
    class_depths: dict[str, int]


class OntologyRegistry:
    """Singleton registry holding active ontology prefix maps and property characteristics.

    Guarantees thread-safe lock-free reads via atomic immutable RegistrySnapshot swapping.
    """

    _instance: ClassVar[OntologyRegistry | None] = None
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __new__(cls) -> Self:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_defaults()
            return cast(Self, cls._instance)

    def _init_defaults(self) -> None:
        self._write_lock = threading.RLock()
        prefix_to_uri: dict[str, str] = {
            "yaw": YAW_PREFIX,
            "rdf": str(W3C_NAMESPACES["rdf"]),
            "rdfs": str(W3C_NAMESPACES["rdfs"]),
            "owl": str(W3C_NAMESPACES["owl"]),
            "sh": str(W3C_NAMESPACES["sh"]),
            "skos": str(W3C_NAMESPACES["skos"]),
            "xsd": str(W3C_NAMESPACES["xsd"]),
            **{k: str(v) for k, v in ROOT_NAMESPACES.items()},
        }
        uri_to_prefix: dict[str, str] = {v: k for k, v in prefix_to_uri.items()}
        self._snapshot = RegistrySnapshot(
            prefix_to_uri=prefix_to_uri,
            uri_to_prefix=uri_to_prefix,
            concept_prefixes=frozenset(ROOT_NAMESPACES.keys()),
            classes={},
            properties={},
            class_depths={},
        )

    @property
    def snapshot(self) -> RegistrySnapshot:
        """Return current immutable snapshot."""
        return self._snapshot

    def is_concept_namespace(self, prefix: str) -> bool:
        """O(1) check whether prefix is a registered concept namespace."""
        return prefix.lower().strip(":") in self._snapshot.concept_prefixes

    def curie_to_uriref(self, curie: str | None) -> URIRef:
        """Resolve CURIE to full W3C URIRef using active registry prefix mappings."""
        if not curie or not curie.strip():
            return URIRef(f"{YAW_PREFIX}unknown")
        raw = curie.strip()
        if raw.startswith(("http://", "https://", "urn:")):
            return URIRef(raw)
        snap = self._snapshot
        if ":" in raw:
            prefix, suffix = raw.split(":", 1)
            prefix_clean = prefix.lower()
            if prefix_clean in snap.prefix_to_uri:
                base = snap.prefix_to_uri[prefix_clean]
                return URIRef(f"{base}{suffix}")
        # Default fallback to yaw concepts
        slug = re.sub(r"\s+", "-", raw.replace("yaw:", ""))
        return URIRef(f"{YAW_PREFIX}concepts/{slug}")

    def uri_to_curie(self, uri: str | URIRef) -> str:
        """Contract full URIRef to registered CURIE prefix:suffix format."""
        uri_str = str(uri)
        snap = self._snapshot
        for base_uri, prefix in snap.uri_to_prefix.items():
            if uri_str.startswith(base_uri):
                suffix = uri_str[len(base_uri) :]
                return f"{prefix}:{suffix}"
        return uri_str

    def get_active_prefixes(self) -> dict[str, str]:
        """Return shallow copy of current prefix-to-URI mapping."""
        return dict(self._snapshot.prefix_to_uri)

    def get_class_depth(self, curie_or_uri: str) -> int:
        """Return taxonomic depth of class relative to its root (0 for root)."""
        return self._snapshot.class_depths.get(curie_or_uri, 0)

    def get_class(self, curie_or_uri: str) -> DynamicClassMeta | None:
        """Return class metadata by curie or uri."""
        return self._snapshot.classes.get(curie_or_uri)

    def get_property(self, curie_or_uri: str) -> DynamicPropertyMeta | None:
        """Return property metadata by curie or uri."""
        return self._snapshot.properties.get(curie_or_uri)

    def sync_from_db(self, conn: sqlite3.Connection) -> None:
        """Re-hydrate registry state atomically via immutable RegistrySnapshot pointer swap."""
        with self._write_lock:
            new_prefix_to_uri = dict(self._snapshot.prefix_to_uri)
            try:
                for row in conn.execute(
                    "SELECT ontology_id, uri_prefix FROM dynamic_ontologies"
                ).fetchall():
                    ont_id, uri_prefix = str(row[0]), str(row[1])
                    new_prefix_to_uri[ont_id.lower()] = uri_prefix
            except sqlite3.OperationalError:
                pass
            new_uri_to_prefix = {v: k for k, v in new_prefix_to_uri.items()}

            new_concept_prefixes = set(self._snapshot.concept_prefixes)
            for ont_id in new_prefix_to_uri:
                new_concept_prefixes.add(ont_id)

            new_classes: dict[str, DynamicClassMeta] = {}
            new_depths: dict[str, int] = {}
            try:
                for row in conn.execute(
                    "SELECT curie, class_uri, ontology_id, label, parent_class_uri, depth, is_root, epistemic_type FROM dynamic_classes"
                ).fetchall():
                    (
                        curie,
                        uri_str,
                        ont_id,
                        label,
                        parent_uri,
                        depth,
                        is_root,
                        ep_type,
                    ) = (
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                        str(row[3]),
                        str(row[4]) if row[4] else None,
                        int(row[5]),
                        bool(row[6]),
                        str(row[7]),
                    )
                    meta = DynamicClassMeta(
                        curie=curie,
                        uri=URIRef(uri_str),
                        ontology_id=ont_id,
                        label=label,
                        parent_uri=URIRef(parent_uri) if parent_uri else None,
                        depth=depth,
                        is_root=is_root,
                        epistemic_type=ep_type,
                    )
                    new_classes[curie] = meta
                    new_classes[uri_str] = meta
                    new_depths[curie] = depth
                    new_depths[uri_str] = depth
            except sqlite3.OperationalError:
                pass

            new_properties: dict[str, DynamicPropertyMeta] = {}
            try:
                for row in conn.execute(
                    "SELECT curie, property_uri, ontology_id, label, parent_property_uri, is_transitive, is_symmetric FROM dynamic_properties"
                ).fetchall():
                    curie, uri_str, ont_id, label, parent_uri, is_trans, is_symm = (
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                        str(row[3]),
                        str(row[4]) if row[4] else None,
                        bool(row[5]),
                        bool(row[6]),
                    )
                    prop_meta = DynamicPropertyMeta(
                        curie=curie,
                        uri=URIRef(uri_str),
                        ontology_id=ont_id,
                        label=label,
                        parent_uri=URIRef(parent_uri) if parent_uri else None,
                        is_transitive=is_trans,
                        is_symmetric=is_symm,
                    )
                    new_properties[curie] = prop_meta
                    new_properties[uri_str] = prop_meta
            except sqlite3.OperationalError:
                pass

            self._snapshot = RegistrySnapshot(
                prefix_to_uri=new_prefix_to_uri,
                uri_to_prefix=new_uri_to_prefix,
                concept_prefixes=frozenset(new_concept_prefixes),
                classes=new_classes,
                properties=new_properties,
                class_depths=new_depths,
            )


def get_ontology_registry() -> OntologyRegistry:
    """Return the global OntologyRegistry singleton."""
    return OntologyRegistry()


__all__ = [
    "DynamicClassMeta",
    "DynamicPropertyMeta",
    "OntologyRegistry",
    "RegistrySnapshot",
    "get_ontology_registry",
]
