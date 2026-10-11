"""External ontology ingestion and alias materialization pipeline."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import rdflib
from rdflib import OWL, RDF, RDFS, SKOS, URIRef

from yaw.ontology.registry import get_ontology_registry
from yaw.storage.db import insert_alias

logger = logging.getLogger(__name__)


def _parse_source_graph(
    source: str | Path | rdflib.Graph,
) -> tuple[rdflib.Graph, str]:
    """Parse RDF graph from source and return graph with source identifier."""
    g = rdflib.Graph()
    if isinstance(source, rdflib.Graph):
        return source, "in-memory"

    source_str = str(source)
    if "\n" in source_str or "@prefix" in source_str or "PREFIX" in source_str:
        g.parse(data=source_str, format="turtle")
        return g, "string"

    source_path = Path(source)
    source_uri = str(source_path.resolve()) if source_path.exists() else str(source)
    fmt = "turtle" if str(source).endswith((".ttl", ".owl")) else None
    g.parse(source, format=fmt)
    return g, source_uri


def _detect_prefix_and_base(
    g: rdflib.Graph,
    ontology_id: str | None,
) -> tuple[str, str]:
    """Detect ontology prefix and base URI from OWL Ontology or SKOS ConceptScheme, falling back to namespace inspection."""
    detected_prefix = ontology_id
    base_uri: str | None = None

    for ont_subj in g.subjects(RDF.type, OWL.Ontology):
        base_uri = str(ont_subj).rstrip("#/") + "#"
        if not detected_prefix:
            detected_prefix = (
                str(ont_subj).rsplit("/", 1)[-1].rsplit("#", 1)[-1].lower()
            )
            break

    if not base_uri:
        for scheme_subj in g.subjects(RDF.type, SKOS.ConceptScheme):
            base_uri = str(scheme_subj).rstrip("#/") + "#"
            if not detected_prefix:
                detected_prefix = (
                    str(scheme_subj).rsplit("/", 1)[-1].rsplit("#", 1)[-1].lower()
                )
                break

    # If no OWL Ontology or SKOS ConceptScheme found, inspect user-declared namespaces
    if not base_uri:
        std_prefixes = {
            "xml",
            "rdf",
            "rdfs",
            "owl",
            "xsd",
            "sh",
            "skos",
            "dc",
            "dcterms",
            "foaf",
        }
        ns_map = {prefix: str(uri) for prefix, uri in g.namespaces() if prefix}
        if detected_prefix and detected_prefix in ns_map:
            base_uri = ns_map[detected_prefix]
        else:
            for p, u in g.namespaces():
                if p and p not in std_prefixes:
                    if not detected_prefix:
                        detected_prefix = p
                    base_uri = str(u)
                    break

    detected_prefix = detected_prefix or "ext"
    base_uri = base_uri or f"http://yaw.local/{detected_prefix}/"
    return detected_prefix, base_uri


def _validate_invariant_7(g: rdflib.Graph) -> None:
    """Enforce Invariant 7: strict separation of OWL Class and SKOS Concept semantics."""
    class_types = {OWL.Class, RDFS.Class}
    concept_types = {SKOS.Concept}

    class_nodes: set[URIRef] = set()
    concept_nodes: set[URIRef] = set()

    for s, _, o in g.triples((None, RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        if o in class_types:
            class_nodes.add(s)
        if o in concept_types:
            concept_nodes.add(s)

    # 1. Direct dual declaration (punning on same subject)
    direct_punning = class_nodes & concept_nodes
    if direct_punning:
        node = next(iter(direct_punning))
        raise ValueError(
            f"Cross-epistemic metamodeling violation: URI {node} is declared as both an OWL Class "
            f"and a SKOS Concept. Punning is strictly prohibited under Invariant 7."
        )

    # 2. owl:Class has rdfs:subClassOf a skos:Concept
    for c in class_nodes:
        for parent in g.objects(c, RDFS.subClassOf):
            if parent in concept_nodes:
                raise ValueError(
                    f"Cross-epistemic metamodeling violation: OWL Class {c} declares rdfs:subClassOf "
                    f"SKOS Concept {parent}. Punning is strictly prohibited under Invariant 7."
                )

    # 3. skos:Concept has skos:broader (or rdfs:subClassOf) an owl:Class
    for c in concept_nodes:
        for parent in g.objects(c, SKOS.broader):
            if parent in class_nodes:
                raise ValueError(
                    f"Cross-epistemic metamodeling violation: SKOS Concept {c} declares skos:broader "
                    f"OWL Class {parent}. Punning is strictly prohibited under Invariant 7."
                )
        for parent in g.objects(c, RDFS.subClassOf):
            if parent in class_nodes:
                raise ValueError(
                    f"Cross-epistemic metamodeling violation: SKOS Concept {c} declares rdfs:subClassOf "
                    f"OWL Class {parent}. Punning is strictly prohibited under Invariant 7."
                )

    # 4. Individual declared as both owl:Class and instance of a skos:Concept
    # (Or individual declared as skos:Concept and instance of an owl:Class)
    for s, _, o in g.triples((None, RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        if s in class_nodes and o in concept_nodes:
            raise ValueError(
                f"Cross-epistemic metamodeling violation: Individual {s} declared as OWL Class "
                f"and instance of SKOS Concept {o}. Punning is strictly prohibited under Invariant 7."
            )
        if s in concept_nodes and o in class_nodes:
            raise ValueError(
                f"Cross-epistemic metamodeling violation: Individual {s} declared as SKOS Concept "
                f"and instance of OWL Class {o}. Punning is strictly prohibited under Invariant 7."
            )


def _compute_taxonomic_depths(
    raw_classes: list[dict[str, Any]],
) -> None:
    """Compute topological depth and root status for raw classes in-place."""
    class_parent_map: dict[str, list[str]] = {
        rc["uri_str"]: rc["parents"] for rc in raw_classes
    }

    def compute_depth(uri_s: str, visited: set[str]) -> int:
        if uri_s in visited:
            return 0
        parents = class_parent_map.get(uri_s, [])
        if not parents:
            return 0
        new_visited = visited | {uri_s}
        parent_depths = [
            compute_depth(p, new_visited) for p in parents if p in class_parent_map
        ]
        return 1 + max(parent_depths, default=-1)

    for rc in raw_classes:
        d = compute_depth(rc["uri_str"], set())
        rc["depth"] = d
        rc["is_root"] = 1 if d == 0 else 0


def _parse_single_class(
    g: rdflib.Graph,
    c_uri: URIRef,
    detected_prefix: str,
    class_types: set[Any],
    concept_types: set[Any],
) -> dict[str, Any] | None:
    """Parse class or concept node and enforce Invariant 7 non-punning."""
    types = set(g.objects(c_uri, RDF.type))
    has_class = bool(types & class_types)
    has_concept = bool(types & concept_types)

    if not (has_class or has_concept):
        return None

    if has_class and has_concept:
        raise ValueError(
            f"Metamodeling violation: URI {c_uri} is declared as both an OWL Class "
            f"and a SKOS Concept. Punning is strictly prohibited under Invariant 7."
        )

    epistemic_type = "class" if has_class else "concept"
    uri_str = str(c_uri)
    suffix = uri_str.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    curie = f"{detected_prefix}:{suffix}"

    labels = [str(l) for l in g.objects(c_uri, RDFS.label)]
    labels.extend(str(l) for l in g.objects(c_uri, SKOS.prefLabel))
    pref_label = labels[0] if labels else suffix

    defs = [str(d) for d in g.objects(c_uri, RDFS.comment)]
    defs.extend(str(d) for d in g.objects(c_uri, SKOS.definition))
    definition = defs[0] if defs else ""

    if epistemic_type == "class":
        parents = [
            str(p)
            for p in g.objects(c_uri, RDFS.subClassOf)
            if isinstance(p, URIRef) and p != OWL.Thing
        ]
    else:
        parents = [
            str(p) for p in g.objects(c_uri, SKOS.broader) if isinstance(p, URIRef)
        ]

    alt_labels = [str(alt).lower().strip() for alt in g.objects(c_uri, SKOS.altLabel)]

    return {
        "uri": c_uri,
        "uri_str": uri_str,
        "curie": curie,
        "suffix": suffix,
        "label": pref_label,
        "definition": definition,
        "epistemic_type": epistemic_type,
        "parents": parents,
        "alt_labels": alt_labels,
    }


def _extract_classes(
    g: rdflib.Graph,
    detected_prefix: str,
) -> list[dict[str, Any]]:
    """Extract and validate Classes & Concepts, enforcing Invariant 7 non-punning."""
    class_types = {OWL.Class, RDFS.Class}
    concept_types = {SKOS.Concept}
    raw_classes: list[dict[str, Any]] = []

    for c_uri in set(g.subjects(RDF.type, None)):
        if not isinstance(c_uri, URIRef):
            continue
        parsed = _parse_single_class(
            g, c_uri, detected_prefix, class_types, concept_types
        )
        if parsed:
            raw_classes.append(parsed)

    _compute_taxonomic_depths(raw_classes)
    return raw_classes


def _extract_properties(
    g: rdflib.Graph,
    detected_prefix: str,
) -> list[dict[str, Any]]:
    """Extract OWL and RDF properties from ontology graph."""
    prop_types = {
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        RDF.Property,
        OWL.TransitiveProperty,
    }
    raw_props: list[dict[str, Any]] = []

    for p_uri in set(g.subjects(RDF.type, None)):
        if not isinstance(p_uri, URIRef):
            continue
        types = set(g.objects(p_uri, RDF.type))
        if not (types & prop_types):
            continue

        uri_str = str(p_uri)
        suffix = uri_str.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        curie = f"{detected_prefix}:{suffix}"

        labels = [str(l) for l in g.objects(p_uri, RDFS.label)]
        pref_label = labels[0] if labels else suffix

        defs = [str(d) for d in g.objects(p_uri, RDFS.comment)]
        definition = defs[0] if defs else ""

        parents = [
            str(p)
            for p in g.objects(p_uri, RDFS.subPropertyOf)
            if isinstance(p, URIRef)
        ]
        parent_uri = parents[0] if parents else None

        raw_props.append(
            {
                "uri": p_uri,
                "uri_str": uri_str,
                "curie": curie,
                "suffix": suffix,
                "label": pref_label,
                "definition": definition,
                "parent_uri": parent_uri,
                "is_transitive": 1 if OWL.TransitiveProperty in types else 0,
                "is_symmetric": 1 if OWL.SymmetricProperty in types else 0,
            }
        )

    return raw_props


def _insert_class_records(
    conn: sqlite3.Connection,
    raw_classes: list[dict[str, Any]],
    detected_prefix: str,
    has_class_def: bool,
) -> tuple[int, int]:
    """Insert class records into dynamic_classes and dynamic_class_parents."""
    classes_ingested = 0
    aliases_materialized = 0

    for rc in raw_classes:
        parent_uri = rc["parents"][0] if rc["parents"] else None
        if has_class_def:
            conn.execute(
                """INSERT OR REPLACE INTO dynamic_classes 
                   (curie, class_uri, ontology_id, label, definition, parent_class_uri, depth, is_root, epistemic_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rc["curie"],
                    rc["uri_str"],
                    detected_prefix,
                    rc["label"],
                    rc["definition"],
                    parent_uri,
                    rc["depth"],
                    rc["is_root"],
                    rc["epistemic_type"],
                ),
            )
        else:
            conn.execute(
                """INSERT OR REPLACE INTO dynamic_classes 
                   (curie, class_uri, ontology_id, label, parent_class_uri, depth, is_root, epistemic_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rc["curie"],
                    rc["uri_str"],
                    detected_prefix,
                    rc["label"],
                    parent_uri,
                    rc["depth"],
                    rc["is_root"],
                    rc["epistemic_type"],
                ),
            )
        classes_ingested += 1

        for sec_parent in rc["parents"][1:]:
            conn.execute(
                """INSERT OR IGNORE INTO dynamic_class_parents (curie, parent_class_uri)
                   VALUES (?, ?)""",
                (rc["curie"], sec_parent),
            )

        label_clean = rc["label"].strip()
        class_aliases = {
            rc["curie"],
            label_clean,
            label_clean.lower(),
            rc["suffix"],
            rc["suffix"].lower(),
        }
        for a_form in class_aliases:
            if a_form:
                insert_alias(
                    conn,
                    alias_id=a_form,
                    canonical_id=rc["curie"],
                    kind="entity",
                    method="ontology",
                    score=1.0,
                )
                aliases_materialized += 1

        for alt in rc["alt_labels"]:
            alt_clean = alt.strip()
            for alt_form in {alt_clean, alt_clean.lower()}:
                if alt_form:
                    insert_alias(
                        conn,
                        alias_id=alt_form,
                        canonical_id=rc["curie"],
                        kind="entity",
                        method="ontology",
                        score=0.95,
                    )
                    aliases_materialized += 1

    return classes_ingested, aliases_materialized


def _insert_prop_records(
    conn: sqlite3.Connection,
    raw_props: list[dict[str, Any]],
    detected_prefix: str,
    has_prop_def: bool,
) -> tuple[int, int]:
    """Insert property records into dynamic_properties and materialize aliases."""
    props_ingested = 0
    aliases_materialized = 0

    for rp in raw_props:
        if has_prop_def:
            conn.execute(
                """INSERT OR REPLACE INTO dynamic_properties 
                   (curie, property_uri, ontology_id, label, definition, parent_property_uri, is_transitive, is_symmetric)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rp["curie"],
                    rp["uri_str"],
                    detected_prefix,
                    rp["label"],
                    rp["definition"],
                    rp["parent_uri"],
                    rp["is_transitive"],
                    rp["is_symmetric"],
                ),
            )
        else:
            conn.execute(
                """INSERT OR REPLACE INTO dynamic_properties 
                   (curie, property_uri, ontology_id, label, parent_property_uri, is_transitive, is_symmetric)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rp["curie"],
                    rp["uri_str"],
                    detected_prefix,
                    rp["label"],
                    rp["parent_uri"],
                    rp["is_transitive"],
                    rp["is_symmetric"],
                ),
            )
        props_ingested += 1

        prop_clean = rp["label"].strip()
        prop_aliases = {
            rp["curie"],
            prop_clean,
            prop_clean.lower(),
            rp["suffix"],
            rp["suffix"].lower(),
        }
        for a_form in prop_aliases:
            if a_form:
                insert_alias(
                    conn,
                    alias_id=a_form,
                    canonical_id=rp["curie"],
                    kind="predicate",
                    method="ontology",
                    score=1.0,
                )
                aliases_materialized += 1

    return props_ingested, aliases_materialized


def _index_ontology_fts_and_vectors(
    conn: sqlite3.Connection,
    raw_classes: list[dict[str, Any]],
    raw_props: list[dict[str, Any]],
    detected_prefix: str,
) -> None:
    """Index dynamic ontology classes and properties into query_fts and compute vector embeddings."""
    from yaw.nlp import embed_texts_sync, format_prompt, is_ollama_online
    from yaw.storage.db import serialize_embedding

    master_tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }

    if "query_fts" not in master_tables:
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS query_fts USING fts5("
                "entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, "
                "content='', contentless_unindexed=1, contentless_delete=1"
                ")"
            )
            master_tables.add("query_fts")
        except sqlite3.OperationalError as op_err:
            logger.debug("Could not create query_fts virtual table: %s", op_err)

    if "query_fts" in master_tables:
        for rc in raw_classes:
            curie = str(rc["curie"])
            label = str(rc.get("label", "")).strip()
            definition = str(rc.get("definition", "")).strip()
            content = f"{label}: {definition}" if definition else label
            rel_path = f"dynamic_ontology:{detected_prefix}"
            conn.execute(
                "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                (curie, rel_path, label, content),
            )

        for rp in raw_props:
            curie = str(rp["curie"])
            label = str(rp.get("label", "")).strip()
            definition = str(rp.get("definition", "")).strip()
            content = f"{label}: {definition}" if definition else label
            rel_path = f"dynamic_ontology:{detected_prefix}"
            conn.execute(
                "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                (curie, rel_path, label, content),
            )

    if "leaf_nodes" in master_tables and raw_classes:
        prompts = [
            format_prompt(
                f"{str(rc.get('label', '')).strip()}: {str(rc.get('definition', '')).strip()}"
                if str(rc.get("definition", "")).strip()
                else str(rc.get("label", "")).strip(),
                task="Classification",
            )
            for rc in raw_classes
        ]

        embs: list[np.ndarray] = []
        if is_ollama_online(timeout_s=0.5):
            try:
                embs = embed_texts_sync(prompts)
            except (OSError, TimeoutError, ValueError, RuntimeError) as err:
                logger.warning(
                    "Failed to compute embeddings for ontology classes: %s", err
                )

        for i, rc in enumerate(raw_classes):
            curie = str(rc["curie"])
            label = str(rc.get("label", "")).strip()
            definition = str(rc.get("definition", "")).strip()
            source_leaf_id = f"dynamic_ontology:{detected_prefix}"
            emb_bytes = serialize_embedding(embs[i]) if i < len(embs) else None
            conn.execute(
                """
                INSERT OR REPLACE INTO leaf_nodes (
                    node_id, source_leaf_id, name, namespace, definition, embedding
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    curie,
                    source_leaf_id,
                    label,
                    detected_prefix,
                    definition,
                    emb_bytes,
                ),
            )


def _persist_ontology_catalog(
    conn: sqlite3.Connection,
    detected_prefix: str,
    base_uri: str,
    source_uri: str,
    is_default: bool,
    raw_classes: list[dict[str, Any]],
    raw_props: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Persist dynamic ontology catalog entries and materialize aliases in SQLite."""
    try:
        conn.execute(
            """INSERT OR REPLACE INTO dynamic_ontologies 
               (ontology_id, uri_prefix, source_uri, is_default, loaded_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                detected_prefix,
                base_uri,
                source_uri,
                1 if is_default else 0,
                time.time(),
            ),
        )

        class_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(dynamic_classes)").fetchall()
        }
        has_class_def = "definition" in class_cols

        prop_cols = {
            r[1]
            for r in conn.execute("PRAGMA table_info(dynamic_properties)").fetchall()
        }
        has_prop_def = "definition" in prop_cols

        c_ingested, c_aliases = _insert_class_records(
            conn, raw_classes, detected_prefix, has_class_def
        )
        p_ingested, p_aliases = _insert_prop_records(
            conn, raw_props, detected_prefix, has_prop_def
        )
        _index_ontology_fts_and_vectors(conn, raw_classes, raw_props, detected_prefix)
        conn.commit()
        return c_ingested, p_ingested, c_aliases + p_aliases
    except Exception:
        conn.rollback()
        raise


def sync_dynamic_ontology_to_kb(
    kb: Any,
    conn: sqlite3.Connection,
    ontology_id: str,
) -> None:
    """Synchronize dynamically ingested classes and aliases with candidate priors into InMemoryLookupKB."""
    from yaw.indexer.canonicalize import _register_alias_in_kb

    # 1. Register canonical classes
    class_rows = conn.execute(
        """
        SELECT curie, label, epistemic_type
        FROM dynamic_classes
        WHERE ontology_id = ?
        """,
        (ontology_id,),
    ).fetchall()

    for curie_val, label_val, _ep_type in class_rows:
        curie = str(curie_val)
        label = str(label_val).strip() if label_val else ""
        if not kb.contains_entity(curie):
            vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
            if label and label in kb.vocab and kb.vocab[label].has_vector:
                arr = np.asarray(kb.vocab[label].vector, dtype=np.float32)
                if len(arr) == kb.entity_vector_length:
                    norm = float(np.linalg.norm(arr))
                    vec = arr / norm if norm > 1e-12 else arr
            kb.add_entity(entity=curie, freq=1, entity_vector=vec)

        _register_alias_in_kb(kb, alias=curie, canonical_id=curie, score=1.0)

    # 2. Register dynamic properties
    prop_rows = conn.execute(
        """
        SELECT curie, label
        FROM dynamic_properties
        WHERE ontology_id = ?
        """,
        (ontology_id,),
    ).fetchall()

    for curie_val, _label_val in prop_rows:
        curie = str(curie_val)
        if not kb.contains_entity(curie):
            vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
            kb.add_entity(entity=curie, freq=1, entity_vector=vec)
        _register_alias_in_kb(kb, alias=curie, canonical_id=curie, score=1.0)

    # 3. Synchronize active aliases for this ontology with grouped prior normalization
    alias_rows = conn.execute(
        """
        SELECT a.alias_id, a.canonical_id, a.score
        FROM aliases a
        WHERE a.status = 'active'
          AND (
            a.canonical_id IN (SELECT curie FROM dynamic_classes WHERE ontology_id = ?)
            OR a.canonical_id IN (SELECT curie FROM dynamic_properties WHERE ontology_id = ?)
          )
        """,
        (ontology_id, ontology_id),
    ).fetchall()

    alias_groups: dict[str, dict[str, float]] = {}
    for a_id, c_id, score in alias_rows:
        if not a_id or not c_id:
            continue
        raw_alias = str(a_id).strip()
        c_str = str(c_id).strip()
        curr_score = float(score) if score is not None else 1.0
        for a_key in {raw_alias, raw_alias.lower()}:
            sub_map = alias_groups.setdefault(a_key, {})
            sub_map[c_str] = max(sub_map.get(c_str, 0.0), curr_score)

    for a_key, cand_map in alias_groups.items():
        if kb.contains_alias(a_key):
            existing_cands = kb.get_alias_candidates(a_key)
            existing_ent_ids = [c.entity_ for c in existing_cands]
            current_sum = sum(c.prior_prob for c in existing_cands)
            rem = max(0.0, 1.0 - current_sum)
            new_cands = [c for c in cand_map if c not in existing_ent_ids]
            if new_cands and rem > 0.01:
                raw_scores = [max(0.001, cand_map[c]) for c in new_cands]
                tot_score = sum(raw_scores)
                avail = rem
                for i, c_str in enumerate(new_cands):
                    if i == len(new_cands) - 1:
                        allot = min(rem, max(0.001, rem))
                    else:
                        allot = min(rem, (cand_map[c_str] / tot_score) * avail)
                        rem -= allot
                    if allot > 0.001:
                        kb.append_alias(
                            alias=a_key,
                            entity=c_str,
                            prior_prob=allot,
                        )
        else:
            cands = list(cand_map.keys())
            raw_probs = [max(0.001, cand_map[c]) for c in cands]
            prob_sum = sum(raw_probs)
            probs = (
                [p / prob_sum for p in raw_probs]
                if prob_sum > 0
                else [1.0 / len(cands)] * len(cands)
            )
            kb.add_alias(alias=a_key, entities=cands, probabilities=probs)


def ingest_ontology(
    source: str | Path | rdflib.Graph,
    conn: sqlite3.Connection,
    ontology_id: str | None = None,
    is_default: bool = False,
    kb: Any | None = None,
) -> dict[str, Any]:
    """Parse external RDF/OWL ontology into dynamic catalog tables and aliases.

    Args:
        source: File path, URL string, or pre-loaded rdflib.Graph.
        conn: Open SQLite connection with WAL pragmas applied.
        ontology_id: Prefix identifier (e.g. 'dico', 'prov'). Auto-detected if None.
        is_default: Whether this ontology serves as default fallback for Quad 3 & 4.
        kb: Optional InMemoryLookupKB to synchronize with dynamic classes and aliases.

    Returns:
        Dictionary summarizing ingested classes, properties, and materialized aliases.
    """
    g, source_uri = _parse_source_graph(source)
    _validate_invariant_7(g)
    detected_prefix, base_uri = _detect_prefix_and_base(g, ontology_id)
    raw_classes = _extract_classes(g, detected_prefix)
    raw_props = _extract_properties(g, detected_prefix)

    c_count, p_count, a_count = _persist_ontology_catalog(
        conn,
        detected_prefix,
        base_uri,
        source_uri,
        is_default,
        raw_classes,
        raw_props,
    )

    reg = get_ontology_registry()
    reg.sync_from_db(conn)

    from yaw.indexer.canonicalize import get_active_kb

    target_kb = kb if kb is not None else get_active_kb()
    if target_kb is not None:
        sync_dynamic_ontology_to_kb(target_kb, conn, detected_prefix)

    return {
        "status": "success",
        "ontology_id": detected_prefix,
        "classes_ingested": c_count,
        "properties_ingested": p_count,
        "aliases_materialized": a_count,
    }


__all__ = ["ingest_ontology", "sync_dynamic_ontology_to_kb"]
