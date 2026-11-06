"""Closed-world SQL SHACL linter, repair execution, focus-node inspection, and W3C Turtle serializer."""

import json
import math
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yaw.core import (
    SHACLViolation,
    compute_mdhash_id,
    detect_scalar_literal,
    is_resource_label,
    text_processing,
)

from .chunks import get_chunk_pointers
from .db import write_tx
from .graph_ops import (
    get_existing_ids,
    merge_entities,
    pool_confidences,
    upsert_entity_labels,
    upsert_semantic_edges,
)
from .hydration import (
    hydrate_ast_parent_context,
    hydrate_chunk_pointer,
    resolve_one_hop_upward_ast_bounds,
)

W3C_PREFIX_HEADER: str = (
    "@prefix : <urn:graph:> .\n"
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
    "@prefix prov: <http://www.w3.org/ns/prov#> .\n"
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
)


def inspect_entity_or_fact(
    conn: sqlite3.Connection,
    focus_node: str,
    corpus_root: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect a single entity or fact node with resolved neighbor names, reified facts, and AST chunk metadata."""
    root_path = Path(corpus_root) if corpus_root is not None else None
    ent_row = conn.execute(
        "SELECT entity_id, name, definition FROM entities WHERE entity_id = ? OR name = ? LIMIT 1",
        (focus_node, focus_node),
    ).fetchone()
    if ent_row is None and not focus_node.startswith("fact-"):
        norm_focus = str(text_processing(focus_node)) or focus_node.strip().lower()
        if norm_focus:
            ent_row = conn.execute(
                """
                SELECT e.entity_id, e.name, e.definition
                FROM entities e
                INNER JOIN entity_labels el ON el.entity_id = e.entity_id
                WHERE el.label_norm = ?
                ORDER BY CASE el.label_kind
                             WHEN 'skos:prefLabel' THEN 0
                             WHEN 'skos:altLabel' THEN 1
                             ELSE 2
                         END ASC,
                         e.entity_id ASC
                LIMIT 1
                """,
                (norm_focus,),
            ).fetchone()
    eid = str(ent_row[0]) if ent_row is not None else focus_node
    entity_info = (
        {"entity_id": str(ent_row[0]), "name": str(ent_row[1]), "definition": str(ent_row[2])}
        if ent_row is not None
        else None
    )

    labels = [
        {
            "label_norm": str(r[0]),
            "label_kind": str(r[1]),
            "raw_label": str(r[2]),
            "chunk_id": str(r[3]),
        }
        for r in conn.execute(
            "SELECT label_norm, label_kind, raw_label, chunk_id FROM entity_labels WHERE entity_id = ?",
            (eid,),
        )
    ]

    raw_fact_rows = conn.execute(
        """
        SELECT
            fact_id,
            subj_id,
            pred,
            obj_id,
            raw_triple_json,
            definition,
            COALESCE(obj_kind, 'entity'),
            obj_literal_lex,
            COALESCE(obj_datatype, ''),
            COALESCE(obj_lang, '')
        FROM facts
        WHERE fact_id = ? OR subj_id = ? OR obj_id = ?
        """,
        (focus_node, eid, eid),
    ).fetchall()

    fact_ids = [str(r[0]) for r in raw_fact_rows]
    fact_chunks_map: dict[str, list[str]] = {}
    if fact_ids:
        placeholders = ",".join("?" for _ in fact_ids)
        for fr in conn.execute(
            f"SELECT fact_id, chunk_id FROM fact_sources WHERE fact_id IN ({placeholders}) ORDER BY chunk_id ASC",
            fact_ids,
        ):
            fact_chunks_map.setdefault(str(fr[0]), []).append(str(fr[1]))

    facts = [
        {
            "fact_id": str(r[0]),
            "subj_id": str(r[1]),
            "pred": str(r[2]),
            "obj_id": str(r[3]) if r[3] is not None else "",
            "raw_triple": json.loads(r[4]),
            "definition": str(r[5]),
            "obj_kind": str(r[6]),
            "property_type": "owl:DatatypeProperty" if str(r[6]) == "literal" else "owl:ObjectProperty",
            "obj_literal_lex": str(r[7]) if r[7] is not None else "",
            "obj_datatype": str(r[8]),
            "obj_lang": str(r[9]),
            "chunk_ids": fact_chunks_map.get(str(r[0]), []),
        }
        for r in raw_fact_rows
    ]

    raw_edges = conn.execute(
        "SELECT src_id, dst_id, kind, weight FROM edges WHERE src_id IN (?, ?) OR dst_id IN (?, ?)",
        (focus_node, eid, focus_node, eid),
    ).fetchall()

    neighbor_ids = {
        str(other)
        for r in raw_edges
        for other in (r[0], r[1])
        if str(other) not in (focus_node, eid) and str(other).startswith("entity-")
    }
    neighbor_meta: dict[str, tuple[str, str]] = {}
    if neighbor_ids:
        n_list = sorted(neighbor_ids)
        placeholders = ",".join("?" for _ in n_list)
        for nr in conn.execute(
            f"SELECT entity_id, name, definition FROM entities WHERE entity_id IN ({placeholders})",
            n_list,
        ):
            neighbor_meta[str(nr[0])] = (str(nr[1]), str(nr[2]))

    edges = []
    for r in raw_edges:
        s_id, d_id, k_str, w_val = str(r[0]), str(r[1]), str(r[2]), float(r[3])
        other_id = d_id if s_id in (focus_node, eid) else s_id
        other_name, other_def = neighbor_meta.get(other_id, ("", ""))
        edges.append(
            {
                "src_id": s_id,
                "dst_id": d_id,
                "kind": k_str,
                "weight": w_val,
                "neighbor_id": other_id,
                "neighbor_name": other_name,
                "neighbor_definition": other_def,
            }
        )

    chunk_ids_set: set[str] = {
        item["chunk_id"] for item in labels if item["chunk_id"] and not item["chunk_id"].startswith("__")
    }
    for r in conn.execute(
        "SELECT DISTINCT chunk_id FROM edge_sources WHERE src_id IN (?, ?) OR dst_id IN (?, ?)",
        (focus_node, eid, focus_node, eid),
    ):
        chunk_ids_set.add(str(r[0]))
    for f_cids in fact_chunks_map.values():
        chunk_ids_set.update(f_cids)
    for e in edges:
        if e["kind"] == "passage" and e["dst_id"] == eid:
            chunk_ids_set.add(str(e["src_id"]))

    sorted_cids = sorted(c for c in chunk_ids_set if c and not c.startswith("__"))
    chunk_pointers = get_chunk_pointers(conn, sorted_cids) if sorted_cids else {}
    chunks_meta: list[dict[str, Any]] = []
    for cid in sorted_cids:
        ptr = chunk_pointers.get(cid)
        if ptr is not None:
            parent_s, parent_e, parent_title = (
                resolve_one_hop_upward_ast_bounds(root_path / ptr.rel_posix_path, ptr)
                if root_path is not None
                else (ptr.ast_island_start, ptr.ast_island_end, ptr.title)
            )
            chunks_meta.append(
                {
                    "chunk_id": cid,
                    "rel_posix_path": ptr.rel_posix_path,
                    "title": ptr.title,
                    "start_line": ptr.start_line,
                    "end_line": ptr.end_line,
                    "ast_island_start": ptr.ast_island_start,
                    "ast_island_end": ptr.ast_island_end,
                    "parent_ast_start": parent_s,
                    "parent_ast_end": parent_e,
                    "parent_ast_title": parent_title,
                }
            )

    return {
        "focus_node": focus_node,
        "entity": entity_info,
        "labels": labels,
        "facts": facts,
        "edges": edges,
        "chunk_ids": sorted_cids,
        "chunks_meta": chunks_meta,
    }


def format_entity_skos_markdown(
    conn: sqlite3.Connection,
    entity_id: str,
    corpus_root: Path | str | None = None,
    *,
    heading_prefix: str = "## Entity",
    ast_context_hops: int = 0,
    max_facts: int = 6,
    max_chunks: int = 2,
    max_chunk_chars: int = 900,
    info: dict[str, Any] | None = None,
    shared_source_excerpts: dict[str, str] | None = None,
) -> str:
    """Format an entity or literal node's SKOS/RDF/OWL profile, reified facts, and source chunks."""
    node_info = info if info is not None else inspect_entity_or_fact(conn, entity_id, corpus_root)
    ent = node_info.get("entity") or {"entity_id": entity_id, "name": entity_id, "definition": ""}
    eid = str(ent.get("entity_id", entity_id))
    ename = str(ent.get("name", eid))
    edef = str(ent.get("definition", "")).strip()
    labels: list[dict[str, str]] = list(node_info.get("labels") or [])
    facts: list[dict[str, Any]] = list(node_info.get("facts") or [])
    edges: list[dict[str, Any]] = list(node_info.get("edges") or [])
    chunks_meta: list[dict[str, Any]] = list(node_info.get("chunks_meta") or [])

    pref_labels = sorted(
        {f'"{it["raw_label"]}" (norm: "{it["label_norm"]}")' for it in labels if it["label_kind"] == "skos:prefLabel"}
    )
    alt_labels = sorted({f'"{it["raw_label"]}"' for it in labels if it["label_kind"] == "skos:altLabel"})
    hidden_labels = sorted({f'"{it["raw_label"]}"' for it in labels if it["label_kind"] == "skos:hiddenLabel"})

    broader_targets: list[str] = []
    narrower_sources: list[str] = []
    related_neighbors: list[str] = []
    diff_neighbors: list[str] = []
    seen_sem: set[tuple[str, str]] = set()
    for e in edges:
        k = str(e.get("kind", ""))
        n_id = str(e.get("neighbor_id", ""))
        n_name = str(e.get("neighbor_name", "")) or n_id
        n_def = str(e.get("neighbor_definition", "")).strip()
        if not n_id or (k, n_id) in seen_sem:
            continue
        if k == "skos:broader":
            if e["src_id"] == eid:
                seen_sem.add(("broader", n_id))
                broader_targets.append(f'"{n_name}" ({n_id})')
            elif e["dst_id"] == eid:
                seen_sem.add(("narrower", n_id))
                narrower_sources.append(f'"{n_name}" ({n_id})')
        elif k == "skos:related" and e["src_id"] == eid:
            seen_sem.add(("related", n_id))
            related_neighbors.append(f'"{n_name}" ({n_id})')
        elif k == "owl:differentFrom" and e["src_id"] == eid:
            seen_sem.add(("diff", n_id))
            short_def = f" — {n_def[:80]}" if n_def else ""
            diff_neighbors.append(f'"{n_name}{short_def}" ({n_id})')

    lines: list[str] = [f"{heading_prefix}: `{ename}` (`{eid}`)"]
    lines.append("### SKOS / RDF / OWL Profile")
    lines.append("```turtle")
    lines.append("a skos:Concept ;")
    if pref_labels:
        lines.append(f"skos:prefLabel {', '.join(pref_labels[:4])} ;")
    else:
        lines.append(f'skos:prefLabel "{ename}" ;')
    if alt_labels:
        lines.append(f"skos:altLabel {', '.join(alt_labels[:8])} ;")
    if hidden_labels:
        lines.append(f"skos:hiddenLabel {', '.join(hidden_labels[:6])} ;")
    if broader_targets:
        lines.append(f"skos:broader {', '.join(broader_targets[:5])} ;")
    if narrower_sources:
        lines.append(f"skos:narrower {', '.join(narrower_sources[:5])} ;")
    if related_neighbors:
        lines.append(f"skos:related {', '.join(related_neighbors[:6])} ;")
    if diff_neighbors:
        lines.append(f"owl:differentFrom {', '.join(diff_neighbors[:5])} ;")
    escaped_def = edef.replace('"', '\\"') if edef else "(empty)"
    lines.append(f'skos:definition "{escaped_def}" .')
    lines.append("```")

    lines.append("### Reified Facts (`rdf:Statement`)")
    if facts:
        for f in facts[:max_facts]:
            rt = f.get("raw_triple") or [f["subj_id"], f["pred"], f["obj_id"] or f["obj_literal_lex"]]
            s_str = str(rt[0]) if len(rt) > 0 else f["subj_id"]
            p_str = str(rt[1]) if len(rt) > 1 else f["pred"]
            o_str = str(rt[2]) if len(rt) > 2 else (f["obj_literal_lex"] or f["obj_id"])
            prop_type = f.get("property_type", "owl:ObjectProperty")
            if f.get("obj_kind") == "literal":
                dt = f.get("obj_datatype") or "rdfs:Literal"
                lg = f"@{f['obj_lang']}" if f.get("obj_lang") else ""
                o_fmt = f'"{o_str}"^^{dt}{lg}'
            else:
                o_fmt = f"`{o_str}`"
            f_def = str(f.get("definition", "")).strip()
            lines.append(f"- (`{s_str}`, `{p_str}` [{prop_type}], {o_fmt}) — *{f_def or 'No fact definition.'}*")
    else:
        lines.append("- *(No reified facts attached)*")

    lines.append(
        f"### Source Chunks ({ast_context_hops}-Hop AST Context)" if ast_context_hops > 0 else "### Source Chunks"
    )
    root_path = Path(corpus_root) if corpus_root is not None else None
    if chunks_meta:
        ptrs_map = get_chunk_pointers(conn, [cm["chunk_id"] for cm in chunks_meta[:max_chunks]])
        for cm in chunks_meta[:max_chunks]:
            cid = cm["chunk_id"]
            lines.append(
                f"- **Document `{cm['rel_posix_path']}` | AST Section `{cm['title'] or '(root)'}` "
                f"(Chunk Lines {cm['start_line']}–{cm['end_line']}, "
                f"AST Island {cm['ast_island_start']}–{cm['ast_island_end']})** [`{cid}`]:"
            )
            if shared_source_excerpts is not None and cid in shared_source_excerpts:
                lines.append(f"  See shared source excerpt `{cid}`.")
                continue
            ptr = ptrs_map.get(cid)
            hydrated_body = ""
            parent_header = ""
            if ptr is not None and root_path is not None:
                if ast_context_hops > 0:
                    parent_header = hydrate_ast_parent_context(
                        ptr,
                        root_path,
                        hops=ast_context_hops,
                        max_chars=max_chunk_chars // 2,
                    )
                hp = hydrate_chunk_pointer(ptr, root_path, max_chars=max_chunk_chars)
                if not hp.stale_on_disk and hp.text:
                    hydrated_body = hp.text.strip()
            excerpt_lines: list[str] = []
            if parent_header:
                indented_parent = "\n".join(f"  > [AST Parent Header] {ln}" for ln in parent_header.splitlines()[:6])
                excerpt_lines.append(indented_parent)
            if hydrated_body:
                indented_body = "\n".join(f"  > {ln}" for ln in hydrated_body.splitlines()[:14])
                excerpt_lines.append(indented_body)
            if shared_source_excerpts is None:
                lines.extend(excerpt_lines)
            else:
                shared_source_excerpts[cid] = (
                    f"## Source `{cid}`\n" + "\n".join(excerpt_lines)
                    if excerpt_lines
                    else f"## Source `{cid}`\n*(Source text unavailable)*"
                )
                lines.append(f"  See shared source excerpt `{cid}`.")
    elif node_info.get("chunk_ids"):
        lines.append(f"- Chunk IDs: {', '.join(node_info['chunk_ids'][:max_chunks])}")
    else:
        lines.append("- *(No source chunks recorded)*")

    return "\n".join(lines)


def format_pair_disambiguation_dossier(
    conn: sqlite3.Connection,
    entity_a: str,
    entity_b: str,
    corpus_root: Path | str | None = None,
    *,
    ast_context_hops: int = 0,
    dense_sim: float | None = None,
    lexical_sim: float | None = None,
    max_chunk_chars: int = 900,
    info_a: dict[str, Any] | None = None,
    info_b: dict[str, Any] | None = None,
    include_entity_profiles: bool = True,
) -> str:
    """Build a structured Markdown + SKOS/RDF/OWL comparison dossier for a candidate entity/literal pair."""
    ia = info_a if info_a is not None else inspect_entity_or_fact(conn, entity_a, corpus_root)
    ib = info_b if info_b is not None else inspect_entity_or_fact(conn, entity_b, corpus_root)

    docs_a = {cm["rel_posix_path"] for cm in ia.get("chunks_meta") or [] if cm.get("rel_posix_path")}
    docs_b = {cm["rel_posix_path"] for cm in ib.get("chunks_meta") or [] if cm.get("rel_posix_path")}
    shared_docs = sorted(docs_a & docs_b)

    islands_a = {
        (cm["rel_posix_path"], cm["parent_ast_start"], cm["parent_ast_end"])
        for cm in ia.get("chunks_meta") or []
        if cm.get("rel_posix_path")
    }
    islands_b = {
        (cm["rel_posix_path"], cm["parent_ast_start"], cm["parent_ast_end"])
        for cm in ib.get("chunks_meta") or []
        if cm.get("rel_posix_path")
    }
    shared_islands = sorted(islands_a & islands_b)

    direct_facts = [
        f
        for f in ia.get("facts") or []
        if (f["subj_id"] == entity_a and f["obj_id"] == entity_b)
        or (f["subj_id"] == entity_b and f["obj_id"] == entity_a)
    ]

    header_lines = [f"# Candidate Pair `{entity_a}|{entity_b}`"]
    sim_parts: list[str] = []
    if dense_sim is not None:
        sim_parts.append(f"dense=`{dense_sim:.3f}`")
    if lexical_sim is not None:
        sim_parts.append(f"lexical=`{lexical_sim:.3f}`")
    if sim_parts:
        header_lines.append(f"- **Similarity**: {', '.join(sim_parts)}")
    if shared_islands:
        isl_str = ", ".join(f"`{p}` (AST lines {s}–{e})" for p, s, e in shared_islands[:2])
        header_lines.append(f"- **Shared AST Island / Document**: `YES` ({isl_str})")
    elif shared_docs:
        header_lines.append(f"- **Shared Document**: `YES` ({', '.join(f'`{d}`' for d in shared_docs[:2])})")
    else:
        header_lines.append("- **Shared Document / AST Island**: `NO` (disjoint source documents)")
    if direct_facts:
        df_str = "; ".join(f"`{f.get('raw_triple')}`" for f in direct_facts[:2])
        header_lines.append(f"- **Direct Fact Link Between A and B**: `YES` ({df_str})")

    if not include_entity_profiles:
        return "\n".join(header_lines)

    sec_a = format_entity_skos_markdown(
        conn,
        entity_a,
        corpus_root,
        heading_prefix="## Entity A",
        ast_context_hops=ast_context_hops,
        max_chunk_chars=max_chunk_chars,
        info=ia,
    )
    sec_b = format_entity_skos_markdown(
        conn,
        entity_b,
        corpus_root,
        heading_prefix="## Entity B",
        ast_context_hops=ast_context_hops,
        max_chunk_chars=max_chunk_chars,
        info=ib,
    )
    return "\n".join(header_lines) + "\n\n" + sec_a + "\n\n" + sec_b


def inspect_focus_node(
    conn: sqlite3.Connection,
    focus_node: str,
    corpus_root: Path | str | None = None,
    *,
    ast_context_hops: int = 0,
) -> dict[str, Any]:
    """Read-only neighborhood inspection tool for RLM synonym judgment and SHACL healing.

    Supports single entity IDs (`entity-...`), fact IDs (`fact-...`), and pair/edge focus nodes
    (`entity-a|entity-b` or `entity-a->entity-b`).
    """
    clean_focus = str(focus_node or "").strip()
    sep = "|" if "|" in clean_focus else ("->" if "->" in clean_focus else None)
    if sep is not None:
        parts = [p.strip() for p in clean_focus.split(sep, 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            node_a, node_b = parts[0], parts[1]
            info_a = inspect_entity_or_fact(conn, node_a, corpus_root)
            info_b = inspect_entity_or_fact(conn, node_b, corpus_root)
            dossier = format_pair_disambiguation_dossier(
                conn,
                node_a,
                node_b,
                corpus_root,
                ast_context_hops=ast_context_hops,
                info_a=info_a,
                info_b=info_b,
            )
            docs_a = {cm["rel_posix_path"] for cm in info_a.get("chunks_meta") or [] if cm.get("rel_posix_path")}
            docs_b = {cm["rel_posix_path"] for cm in info_b.get("chunks_meta") or [] if cm.get("rel_posix_path")}
            return {
                "focus_node": clean_focus,
                "focus_kind": "pair",
                "entity": info_a.get("entity"),
                "partner_entity": info_b.get("entity"),
                "node_a": info_a,
                "node_b": info_b,
                "shared_documents": sorted(docs_a & docs_b),
                "labels": [*info_a.get("labels", []), *info_b.get("labels", [])],
                "facts": [*info_a.get("facts", []), *info_b.get("facts", [])],
                "edges": [*info_a.get("edges", []), *info_b.get("edges", [])],
                "chunk_ids": sorted(set(info_a.get("chunk_ids", [])) | set(info_b.get("chunk_ids", []))),
                "markdown_dossier": dossier,
            }

    single = inspect_entity_or_fact(conn, clean_focus, corpus_root)
    if single.get("entity") is not None:
        single["focus_kind"] = "entity"
        single["markdown_dossier"] = format_entity_skos_markdown(
            conn,
            str(single["entity"]["entity_id"]),
            corpus_root,
            heading_prefix="## Focus Entity",
            ast_context_hops=ast_context_hops,
            info=single,
        )
    else:
        single["focus_kind"] = "fact" if clean_focus.startswith("fact-") else "unknown"
    return single


def _scan_shacl_violations(conn: sqlite3.Connection) -> list[SHACLViolation]:
    violations: list[SHACLViolation] = []

    # 1. ConceptShape/prefLabel: Every entity must have exactly 1 distinct skos:prefLabel norm
    for row in conn.execute(
        """
        SELECT
            e.entity_id,
            COUNT(DISTINCT CASE WHEN el.label_kind = 'skos:prefLabel' THEN el.label_norm END) AS pref_cnt
        FROM entities e
        LEFT JOIN entity_labels el ON el.entity_id = e.entity_id
        GROUP BY e.entity_id
        HAVING pref_cnt != 1
        ORDER BY e.entity_id ASC
        """
    ):
        eid, cnt = str(row[0]), int(row[1])
        msg = (
            f"Entity {eid} has no skos:prefLabel in entity_labels."
            if cnt == 0
            else f"Entity {eid} has {cnt} distinct skos:prefLabel norms (expected 1)."
        )
        violations.append(
            SHACLViolation(
                shape="ConceptShape/prefLabel",
                focus_node=eid,
                result_path="skos:prefLabel",
                message=msg,
            )
        )

    # 2. ConceptShape/definition: Every entity must have a non-empty definition
    for row in conn.execute(
        """
        SELECT entity_id, name
        FROM entities
        WHERE TRIM(definition) = ''
        ORDER BY entity_id ASC
        """
    ):
        eid, name = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/definition",
                focus_node=eid,
                result_path="skos:definition",
                message=f"Entity {eid} ({name!r}) has an empty skos:definition.",
            )
        )

    # 3. ConceptShape/S13-disjointLabels:
    # (a) No (entity_id, label_norm) pair may have >1 distinct label_kind across chunks
    # (b) No entity's canonical name (LOWER(TRIM(e.name))) may appear as skos:altLabel or skos:hiddenLabel
    seen_s13: set[tuple[str, str]] = set()
    for row in conn.execute(
        """
        SELECT entity_id, label_norm, COUNT(DISTINCT label_kind) AS kind_cnt
        FROM entity_labels
        GROUP BY entity_id, label_norm
        HAVING kind_cnt > 1
        ORDER BY entity_id ASC, label_norm ASC
        """
    ):
        eid, lnorm = str(row[0]), str(row[1])
        seen_s13.add((eid, lnorm))
        violations.append(
            SHACLViolation(
                shape="ConceptShape/S13-disjointLabels",
                focus_node=eid,
                result_path="skos:prefLabel|skos:altLabel|skos:hiddenLabel",
                message=f"Entity {eid} label {lnorm!r} violates SKOS S13 disjointness across chunks.",
            )
        )
    for row in conn.execute(
        """
        SELECT DISTINCT el.entity_id, el.label_norm, e.name,
               EXISTS (
                   SELECT 1 FROM entity_labels el2
                   WHERE el2.entity_id = el.entity_id
                     AND el2.label_kind = 'skos:prefLabel'
                     AND el2.label_norm != el.label_norm
               ) AS has_other_pref
        FROM entity_labels el
        INNER JOIN entities e ON e.entity_id = el.entity_id
        WHERE el.label_kind IN ('skos:altLabel', 'skos:hiddenLabel')
        ORDER BY el.entity_id ASC, el.label_norm ASC
        """
    ):
        eid, lnorm, ename, has_other_pref = str(row[0]), str(row[1]), str(row[2]), bool(row[3])
        raw_norm = ename.strip().lower()
        canon_norm = str(text_processing(ename)) or raw_norm
        if (lnorm == canon_norm or (lnorm == raw_norm and not has_other_pref)) and (eid, lnorm) not in seen_s13:
            seen_s13.add((eid, lnorm))
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/S13-disjointLabels",
                    focus_node=eid,
                    result_path="skos:prefLabel|skos:altLabel|skos:hiddenLabel",
                    message=f"Entity {eid} canonical name {lnorm!r} appears as non-prefLabel in entity_labels.",
                )
            )

    # 4. ConceptShape/homographDifferentFrom: Distinct concepts sharing a skos:prefLabel norm must either have
    # distinct definitions with an adjudicated semantic relation, or be merged if their definitions are identical.
    for row in conn.execute(
        """
        SELECT a.entity_id, b.entity_id, a.label_norm,
               LOWER(TRIM(COALESCE(ea.definition, ''))) AS def_a,
               LOWER(TRIM(COALESCE(eb.definition, ''))) AS def_b,
               EXISTS (
                   SELECT 1 FROM edges e
                   WHERE e.src_id = a.entity_id
                     AND e.dst_id = b.entity_id
                     AND e.kind = 'owl:differentFrom'
               ) AS has_diff_edge
        FROM entity_labels a
        INNER JOIN entity_labels b
          ON a.label_norm = b.label_norm
         AND a.label_kind = 'skos:prefLabel'
         AND b.label_kind = 'skos:prefLabel'
         AND a.entity_id < b.entity_id
        INNER JOIN entities ea ON ea.entity_id = a.entity_id
        INNER JOIN entities eb ON eb.entity_id = b.entity_id
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e
            WHERE (
                (e.src_id = a.entity_id AND e.dst_id = b.entity_id)
                OR (e.src_id = b.entity_id AND e.dst_id = a.entity_id)
            )
              AND e.kind IN ('owl:differentFrom', 'skos:related', 'skos:broader')
        )
        OR (
            LOWER(TRIM(COALESCE(ea.definition, ''))) != ''
            AND LOWER(TRIM(COALESCE(ea.definition, ''))) = LOWER(TRIM(COALESCE(eb.definition, '')))
        )
        GROUP BY a.entity_id, b.entity_id, a.label_norm
        ORDER BY a.entity_id ASC, b.entity_id ASC
        """
    ):
        ea, eb, lnorm, def_a, def_b, has_diff = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            bool(row[5]),
        )
        if def_a and def_a == def_b and has_diff:
            msg = (
                f"Duplicate concepts {ea} and {eb} share prefLabel {lnorm!r} and identical definition "
                "but are separated by owl:differentFrom instead of being merged."
            )
        else:
            msg = f"Homograph concepts {ea} and {eb} share prefLabel {lnorm!r} without owl:differentFrom."
        violations.append(
            SHACLViolation(
                shape="ConceptShape/homographDifferentFrom",
                focus_node=f"{ea}|{eb}",
                result_path="owl:differentFrom",
                message=msg,
            )
        )

    # 5. StatementShape/closedWorld: Facts must have valid subj_id in entities, valid entity/literal object,
    # and non-empty definition that does not duplicate an endpoint entity definition
    for row in conn.execute(
        """
        SELECT
            f.fact_id,
            f.subj_id,
            f.obj_id,
            f.definition,
            es.entity_id IS NOT NULL AS has_subj,
            CASE
                WHEN COALESCE(f.obj_kind, 'entity') = 'literal'
                THEN ((f.obj_id IS NULL OR eo.entity_id IS NOT NULL) AND TRIM(COALESCE(f.obj_literal_lex, '')) != '')
                ELSE (eo.entity_id IS NOT NULL)
            END AS has_valid_obj,
            COALESCE(f.obj_kind, 'entity') AS obj_kind,
            (
                TRIM(f.definition) != ''
                AND (
                    LOWER(TRIM(f.definition)) = LOWER(TRIM(COALESCE(es.definition, '')))
                    OR (
                        COALESCE(f.obj_kind, 'entity') = 'entity'
                        AND LOWER(TRIM(f.definition)) = LOWER(TRIM(COALESCE(eo.definition, '')))
                    )
                )
            ) AS is_dup_def
        FROM facts f
        LEFT JOIN entities es ON es.entity_id = f.subj_id
        LEFT JOIN entities eo ON eo.entity_id = f.obj_id
        WHERE es.entity_id IS NULL
           OR (COALESCE(f.obj_kind, 'entity') = 'entity' AND eo.entity_id IS NULL)
           OR (
               COALESCE(f.obj_kind, 'entity') = 'literal'
               AND ((f.obj_id IS NOT NULL AND eo.entity_id IS NULL) OR TRIM(COALESCE(f.obj_literal_lex, '')) = '')
           )
           OR TRIM(f.definition) = ''
           OR LOWER(TRIM(f.definition)) = LOWER(TRIM(COALESCE(es.definition, '')))
           OR (
               COALESCE(f.obj_kind, 'entity') = 'entity'
               AND LOWER(TRIM(f.definition)) = LOWER(TRIM(COALESCE(eo.definition, '')))
           )
        ORDER BY f.fact_id ASC
        """
    ):
        fid, sid, oid, fdef, has_s, has_o, okind, is_dup_def = (
            str(row[0]),
            str(row[1]),
            str(row[2]) if row[2] is not None else "",
            str(row[3]),
            bool(row[4]),
            bool(row[5]),
            str(row[6]),
            bool(row[7]),
        )
        if not has_s or not has_o:
            msg = (
                f"Literal fact {fid} has missing subject ({sid}) or invalid literal payload."
                if okind == "literal"
                else f"Fact {fid} references missing entity endpoint ({sid} -> {oid})."
            )
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="rdf:subject|rdf:object",
                    message=msg,
                )
            )
        if not fdef.strip():
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="skos:definition",
                    message=f"Fact {fid} has an empty statement definition.",
                )
            )
        elif is_dup_def:
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="skos:definition",
                    message=f"Fact {fid} statement definition duplicates an endpoint entity definition.",
                )
            )

    # 5b. StatementShape/irreflexive: a statement never relates a resource sense to itself
    for (fid,) in conn.execute(
        "SELECT fact_id FROM facts WHERE COALESCE(obj_kind, 'entity') = 'entity' AND subj_id = obj_id "
        "ORDER BY fact_id ASC"
    ):
        violations.append(
            SHACLViolation(
                shape="StatementShape/irreflexive",
                focus_node=str(fid),
                result_path="rdf:subject|rdf:object",
                message=f"Fact {fid} relates a resource sense to itself.",
            )
        )

    # 6. EdgeShape/closedWorld: Endpoints exist, no skos:related + owl:differentFrom collision, symmetric semantic edges
    for row in conn.execute(
        """
        SELECT e.src_id, e.dst_id, e.kind
        FROM edges e
        WHERE (
            e.kind = 'passage'
            AND NOT (
                (
                    EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.src_id)
                    AND EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = e.dst_id)
                )
                OR (
                    EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = e.src_id)
                    AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.dst_id)
                )
            )
        )
        OR (
            e.kind IN ('fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
            AND (
                NOT EXISTS (SELECT 1 FROM entities ent1 WHERE ent1.entity_id = e.src_id)
                OR NOT EXISTS (SELECT 1 FROM entities ent2 WHERE ent2.entity_id = e.dst_id)
            )
        )
        ORDER BY e.src_id ASC, e.dst_id ASC, e.kind ASC
        """
    ):
        s, d, k = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}->{d}",
                result_path=k,
                message=f"Edge ({s}, {d}, {k}) has a dangling endpoint outside closed-world tables.",
            )
        )

    for row in conn.execute(
        """
        SELECT a.src_id, a.dst_id
        FROM edges a
        INNER JOIN edges b
          ON a.src_id = b.src_id
         AND a.dst_id = b.dst_id
         AND a.kind = 'skos:related'
         AND b.kind = 'owl:differentFrom'
        WHERE a.src_id < a.dst_id
        ORDER BY a.src_id ASC, a.dst_id ASC
        """
    ):
        s, d = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}|{d}",
                result_path="skos:related|owl:differentFrom",
                message=f"Node pair ({s}, {d}) has conflicting skos:related and owl:differentFrom edges.",
            )
        )

    for row in conn.execute(
        """
        SELECT a.src_id, a.dst_id, a.kind
        FROM edges a
        WHERE a.kind IN ('skos:related', 'owl:differentFrom')
          AND NOT EXISTS (
              SELECT 1 FROM edges b
              WHERE b.src_id = a.dst_id AND b.dst_id = a.src_id AND b.kind = a.kind
          )
        ORDER BY a.src_id ASC, a.dst_id ASC, a.kind ASC
        """
    ):
        s, d, k = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}->{d}",
                result_path=k,
                message=f"Asymmetric semantic edge ({s}, {d}, {k}) missing reverse direction.",
            )
        )

    # 7. ConceptShape/broaderCycle & ConceptShape/S27-disjointBroaderRelated via cycle-safe recursive CTE
    tc_count_row = conn.execute(
        """
        WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
            SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
            UNION
            SELECT tc.narrower_id, e.dst_id
            FROM broader_tc tc
            JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
            LIMIT 10000
        )
        SELECT COUNT(*) FROM broader_tc
        """
    ).fetchone()
    if tc_count_row is not None and int(tc_count_row[0]) >= 10000:
        violations.append(
            SHACLViolation(
                shape="ConceptShape/broaderTransitiveCapExceeded",
                focus_node="skos:broader",
                result_path="skos:broader",
                message=(
                    "Transitive skos:broader closure reached the 10000-pair safety cap; "
                    "hierarchy cycle and S27 disjointness checks may be incomplete."
                ),
            )
        )

    for row in conn.execute(
        """
        WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
            SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
            UNION
            SELECT tc.narrower_id, e.dst_id
            FROM broader_tc tc
            JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
            LIMIT 10000
        )
        SELECT DISTINCT narrower_id
        FROM broader_tc
        WHERE narrower_id = broader_id
        ORDER BY narrower_id ASC
        """
    ):
        nid = str(row[0])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/broaderCycle",
                focus_node=nid,
                result_path="skos:broader",
                message=f"Concept {nid} participates in a skos:broader hierarchy cycle.",
            )
        )

    for row in conn.execute(
        """
        WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
            SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
            UNION
            SELECT tc.narrower_id, e.dst_id
            FROM broader_tc tc
            JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
            LIMIT 10000
        )
        SELECT DISTINCT tc.narrower_id, tc.broader_id
        FROM broader_tc tc
        INNER JOIN edges rel
          ON rel.kind = 'skos:related'
         AND (
             (rel.src_id = tc.narrower_id AND rel.dst_id = tc.broader_id)
             OR (rel.src_id = tc.broader_id AND rel.dst_id = tc.narrower_id)
         )
        WHERE tc.narrower_id != tc.broader_id
        ORDER BY tc.narrower_id ASC, tc.broader_id ASC
        """
    ):
        s, d = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/S27-disjointBroaderRelated",
                focus_node=f"{s}|{d}",
                result_path="skos:broader|skos:related",
                message=(
                    f"Concept pair ({s}, {d}) violates SKOS S27 disjointness between "
                    "skos:broaderTransitive and skos:related."
                ),
            )
        )

    # 8. ConceptShape/noScalarLiterals: Scalar literals cannot be subjects, ObjectProperty objects, or SKOS nodes
    for row in conn.execute(
        """
        SELECT
            e.entity_id,
            e.name,
            EXISTS (SELECT 1 FROM facts f WHERE f.subj_id = e.entity_id) AS is_subj,
            EXISTS (
                SELECT 1 FROM facts f
                WHERE f.obj_id = e.entity_id AND COALESCE(f.obj_kind, 'entity') = 'entity'
            ) AS is_ent_obj,
            EXISTS (
                SELECT 1 FROM edges ed
                WHERE (ed.src_id = e.entity_id OR ed.dst_id = e.entity_id)
                  AND ed.kind IN ('skos:broader', 'skos:related')
            ) AS has_skos_edge,
            EXISTS (
                SELECT 1 FROM facts f
                WHERE f.obj_id = e.entity_id AND f.obj_kind = 'literal'
            ) AS has_lit_fact,
            EXISTS (
                SELECT 1 FROM edges ed
                WHERE (ed.src_id = e.entity_id OR ed.dst_id = e.entity_id)
                  AND ed.kind = 'passage'
            ) AS has_passage_edge
        FROM entities e
        ORDER BY e.entity_id ASC
        """
    ):
        eid, ename = str(row[0]), str(row[1])
        is_subj, is_ent_obj, has_skos_edge, has_lit_fact, has_passage_edge = (
            bool(row[2]),
            bool(row[3]),
            bool(row[4]),
            bool(row[5]),
            bool(row[6]),
        )
        is_pure_literal = has_lit_fact and not is_subj and not is_ent_obj
        is_scalar, _ = detect_scalar_literal(ename)
        if (is_pure_literal and has_skos_edge) or (
            is_scalar and (is_subj or is_ent_obj or has_skos_edge or (not has_lit_fact and not has_passage_edge))
        ):
            msg = (
                f"Pure literal entity {eid} ({ename!r}) cannot participate in skos:broader or skos:related edges."
                if (is_pure_literal and has_skos_edge and not is_scalar)
                else f"Entity {eid} ({ename!r}) is a scalar literal and cannot be a Concept node."
            )
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/noScalarLiterals",
                    focus_node=eid,
                    result_path="skos:prefLabel",
                    message=msg,
                )
            )

    # 8b. ConceptShape/resourceLabel: NER resources need lexical content; literals never become resources
    for eid, ename in conn.execute("SELECT entity_id, name FROM entities ORDER BY entity_id ASC"):
        if not is_resource_label(str(ename)):
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/resourceLabel",
                    focus_node=str(eid),
                    result_path="skos:prefLabel",
                    message=f"Entity {eid} label {ename!r} has no lexical content; literals are not resources.",
                )
            )

    # 9. ProvenanceShape/factSources: Every fact must have at least one valid chunk entry in fact_sources
    for row in conn.execute(
        """
        SELECT f.fact_id
        FROM facts f
        WHERE NOT EXISTS (
            SELECT 1 FROM fact_sources fs
            INNER JOIN chunks c ON c.chunk_id = fs.chunk_id
            WHERE fs.fact_id = f.fact_id
        )
        ORDER BY f.fact_id ASC
        """
    ):
        fid = str(row[0])
        violations.append(
            SHACLViolation(
                shape="ProvenanceShape/factSources",
                focus_node=fid,
                result_path="prov:wasDerivedFrom",
                message=f"Fact {fid} lacks provenance in fact_sources.",
            )
        )

    # 10. ProvenanceConfidenceShape: Epistemic confidence in entities, facts, and sources must be in (0.0, 1.0]
    for row in conn.execute("SELECT entity_id, confidence FROM entities WHERE confidence IS NOT NULL"):
        eid, conf = str(row[0]), float(row[1])
        if conf <= 0.0 or conf > 1.0 or math.isnan(conf):
            violations.append(
                SHACLViolation(
                    shape="ProvenanceConfidenceShape",
                    focus_node=eid,
                    result_path="prov:value",
                    message=f"Entity {eid} confidence {conf} out of valid bounds (0.0, 1.0].",
                )
            )
    for row in conn.execute("SELECT fact_id, confidence FROM facts WHERE confidence IS NOT NULL"):
        fid, conf = str(row[0]), float(row[1])
        if conf <= 0.0 or conf > 1.0 or math.isnan(conf):
            violations.append(
                SHACLViolation(
                    shape="ProvenanceConfidenceShape",
                    focus_node=fid,
                    result_path="prov:value",
                    message=f"Fact {fid} confidence {conf} out of valid bounds (0.0, 1.0].",
                )
            )
    for row in conn.execute(
        "SELECT fact_id, chunk_id, atomic_confidence FROM fact_sources WHERE atomic_confidence IS NOT NULL"
    ):
        fid, cid, conf = str(row[0]), str(row[1]), float(row[2])
        if conf <= 0.0 or conf > 1.0 or math.isnan(conf):
            violations.append(
                SHACLViolation(
                    shape="ProvenanceConfidenceShape",
                    focus_node=f"{fid}@{cid}",
                    result_path="prov:value",
                    message=f"Fact source {fid}@{cid} atomic_confidence {conf} out of valid bounds (0.0, 1.0].",
                )
            )

    return violations


def lint_graph_shacl(
    conn: sqlite3.Connection,
) -> list[SHACLViolation]:
    """Run read-only closed-world SQL SHACL validation shapes over yaw.db."""
    return _scan_shacl_violations(conn)


def apply_shacl_repairs(
    conn: sqlite3.Connection,
    repairs: Sequence[Mapping[str, Any]],
    *,
    merge_entities_fn: Callable[[str, str, str], bool] | None = None,
) -> int:
    """Execute validated closed-world SHACL repair actions inside an atomic SQLite write transaction."""
    if not repairs:
        return 0
    applied = 0
    with write_tx(conn):
        for rep in repairs:
            focus = str(rep.get("focus_node", "")).strip()
            action = str(rep.get("action", "")).strip()
            val = str(rep.get("value", "")).strip()
            target = str(rep.get("target_node", "")).strip() or val
            if not focus or not action:
                continue
            if action in ("set_definition", "rewrite_definition") and val:
                if focus.startswith("fact-"):
                    cur = conn.execute(
                        "UPDATE facts SET definition = ? WHERE fact_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                else:
                    cur = conn.execute(
                        "UPDATE entities SET definition = ? WHERE entity_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action == "set_pref_label" and val and not focus.startswith("fact-"):
                cur = conn.execute(
                    "UPDATE entities SET name = ? WHERE entity_id = ?",
                    (val, focus),
                )
                if cur.rowcount > 0:
                    norm_val = str(text_processing(val)) or val.strip().lower()
                    cid_row = conn.execute(
                        "SELECT chunk_id FROM entity_labels WHERE entity_id = ? AND chunk_id != '' LIMIT 1",
                        (focus,),
                    ).fetchone()
                    cid = str(cid_row[0]) if cid_row is not None and str(cid_row[0]) else "__healed__"
                    upsert_entity_labels(conn, [(norm_val, focus, "skos:prefLabel", val, cid)])
                    applied += 1
            elif action in ("demote_to_alt_label", "demote_label") and not focus.startswith("fact-"):
                ent_row = conn.execute("SELECT name FROM entities WHERE entity_id = ?", (focus,)).fetchone()
                ename = str(ent_row[0]).strip() if ent_row and ent_row[0] else ""
                tok_norm = str(text_processing(ename)) if ename else ""
                raw_norm = ename.lower() if ename else ""
                rows = conn.execute(
                    "SELECT DISTINCT label_norm FROM entity_labels WHERE entity_id = ? AND label_kind = "
                    "'skos:prefLabel' ORDER BY label_norm ASC",
                    (focus,),
                ).fetchall()
                if len(rows) > 1:
                    norms = [str(r[0]) for r in rows]
                    if tok_norm and tok_norm in norms:
                        winner = tok_norm
                    elif raw_norm and raw_norm in norms:
                        winner = raw_norm
                    else:
                        winner = norms[0]
                    for extra in norms:
                        if extra == winner:
                            continue
                        conn.execute(
                            "UPDATE entity_labels SET label_kind = 'skos:altLabel' WHERE entity_id = ? AND label_norm "
                            "= ? AND label_kind = 'skos:prefLabel'",
                            (focus, extra),
                        )
                    applied += 1
                else:
                    s13_rows = conn.execute(
                        """
                        SELECT label_norm
                        FROM entity_labels
                        WHERE entity_id = ?
                        GROUP BY label_norm
                        HAVING COUNT(DISTINCT label_kind) > 1
                        """,
                        (focus,),
                    ).fetchall()
                    for r in s13_rows:
                        lnorm = str(r[0])
                        target_kind = (
                            "skos:prefLabel"
                            if (lnorm in (tok_norm, raw_norm) or (len(rows) == 1 and str(rows[0][0]) == lnorm))
                            else "skos:altLabel"
                        )
                        conn.execute(
                            "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                            (target_kind, focus, lnorm),
                        )
                        applied += 1
                    if not s13_rows and not rows:
                        alt_row = conn.execute(
                            """
                            SELECT label_norm, raw_label
                            FROM entity_labels
                            WHERE entity_id = ?
                            ORDER BY CASE WHEN label_norm = ? THEN 0
                                          WHEN label_norm = ? THEN 1
                                          ELSE 2 END ASC,
                                     label_norm ASC
                            LIMIT 1
                            """,
                            (focus, tok_norm, raw_norm),
                        ).fetchone()
                        if alt_row is not None:
                            lnorm, rlabel = str(alt_row[0]), str(alt_row[1])
                            conn.execute(
                                "UPDATE entity_labels SET label_kind = 'skos:prefLabel' "
                                "WHERE entity_id = ? AND label_norm = ?",
                                (focus, lnorm),
                            )
                            if rlabel and (not ename or lnorm not in (tok_norm, raw_norm)):
                                conn.execute("UPDATE entities SET name = ? WHERE entity_id = ?", (rlabel, focus))
                            applied += 1
            elif action in ("merge_altLabel", "merge_hiddenLabel") and (target or "|" in focus or "->" in focus):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                    if target == ea:
                        src_node, dst_node = eb, ea
                    else:
                        src_node, dst_node = ea, eb
                elif "->" in focus:
                    ea, eb = (x.strip() for x in focus.split("->", 1))
                    if target == ea:
                        src_node, dst_node = eb, ea
                    else:
                        src_node, dst_node = ea, eb
                else:
                    src_node, dst_node = focus, target
                lkind = "skos:hiddenLabel" if action == "merge_hiddenLabel" else "skos:altLabel"
                if src_node and dst_node and src_node != dst_node:
                    if merge_entities_fn is not None:
                        if merge_entities_fn(src_node, dst_node, lkind):
                            applied += 1
                    else:
                        existing_pair = get_existing_ids(conn, "entity", [src_node, dst_node])
                        if src_node in existing_pair and dst_node in existing_pair:
                            merge_entities(conn, src_node, dst_node, label_kind=lkind)
                            applied += 1
            elif action in ("add_different_from", "assert_differentFrom"):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                elif "->" in focus:
                    ea, eb = (x.strip() for x in focus.split("->", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "owl:differentFrom", 0.0)])
                        applied += 1
            elif action == "assert_related":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                elif "->" in focus:
                    ea, eb = (x.strip() for x in focus.split("->", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "skos:related", 0.5)])
                        applied += 1
            elif action == "assert_broader":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                elif "->" in focus:
                    ea, eb = (x.strip() for x in focus.split("->", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "skos:broader", 0.5)])
                        applied += 1
            elif action == "remove_hierarchy_cycle":
                if "->" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("->", 1))
                    cur = conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                elif "|" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("|", 1))
                    cur = conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                else:
                    cur = conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ?",
                        (focus,),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action == "remove_conflicting_related":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                elif "->" in focus:
                    ea, eb = (x.strip() for x in focus.split("->", 1))
                else:
                    ea, eb = focus, target
                if ea and eb:
                    cur = conn.execute(
                        """
                        DELETE FROM edges
                        WHERE kind = 'skos:related'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (ea, eb, eb, ea),
                    )
                    conn.execute(
                        """
                        DELETE FROM edge_sources
                        WHERE kind = 'skos:related'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (ea, eb, eb, ea),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action in ("ground_fact_source", "set_fact_source") and val:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id) VALUES (?, ?)",
                    (focus, val),
                )
                if cur.rowcount > 0:
                    applied += 1
            elif action == "invert_broader":
                if "->" in focus:
                    s, d = (x.strip() for x in focus.split("->", 1))
                    w_row = conn.execute(
                        "SELECT weight FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                        (s, d),
                    ).fetchone()
                    w = float(w_row[0]) if w_row is not None and w_row[0] is not None else 0.8
                    conn.execute("DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?", (s, d))
                    conn.execute(
                        "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, 'skos:broader', ?)",
                        (d, s, w),
                    )
                    conn.execute(
                        """
                        UPDATE OR IGNORE edge_sources
                        SET src_id = ?, dst_id = ?
                        WHERE src_id = ? AND dst_id = ? AND kind = 'skos:broader'
                        """,
                        (d, s, s, d),
                    )
                    applied += 1
            elif action == "demote_sibling_to_related":
                if "->" in focus:
                    s, d = (x.strip() for x in focus.split("->", 1))
                    w_row = conn.execute(
                        """
                        SELECT weight FROM edges
                        WHERE kind = 'skos:broader'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (s, d, d, s),
                    ).fetchone()
                    w = float(w_row[0]) if w_row is not None and w_row[0] is not None else 0.8
                    conn.execute(
                        """
                        DELETE FROM edges
                        WHERE kind = 'skos:broader'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (s, d, d, s),
                    )
                    upsert_semantic_edges(conn, [(s, d, "skos:related", w)])
                    conn.execute(
                        """
                        UPDATE OR IGNORE edge_sources
                        SET kind = 'skos:related'
                        WHERE ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                          AND kind = 'skos:broader'
                        """,
                        (s, d, d, s),
                    )
                    applied += 1
            elif action == "remove_scalar_entity":
                ent_row = conn.execute(
                    "SELECT name FROM entities WHERE entity_id = ?",
                    (focus,),
                ).fetchone()
                if ent_row is not None:
                    ename = str(ent_row[0])
                    conn.execute(
                        """
                        UPDATE facts
                        SET obj_kind = 'literal',
                            obj_id = NULL,
                            obj_literal_lex = ?,
                            obj_datatype = ?,
                            obj_lang = ''
                        WHERE obj_id = ?
                        """,
                        (
                            ename,
                            detect_scalar_literal(ename)[1],
                            focus,
                        ),
                    )
                    affected_fids = [
                        str(r[0])
                        for r in conn.execute("SELECT fact_id FROM facts WHERE subj_id = ?", (focus,)).fetchall()
                    ]
                    for fid in affected_fids:
                        conn.execute("DELETE FROM fact_sources WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
                    conn.execute("DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (focus, focus))
                    conn.execute("DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?", (focus, focus))
                    conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entities WHERE entity_id = ?", (focus,))
                    applied += 1
            elif action in ("delete_dangling", "delete_orphan", "delete_invalid_fact"):
                if "->" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("->", 1))
                    conn.execute(
                        "DELETE FROM edges WHERE src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    applied += 1
                else:
                    affected_fids = [
                        str(r[0])
                        for r in conn.execute(
                            "SELECT fact_id FROM facts WHERE fact_id = ? OR subj_id = ? OR obj_id = ?",
                            (focus, focus, focus),
                        ).fetchall()
                    ]
                    for fid in affected_fids:
                        conn.execute("DELETE FROM fact_sources WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
                    conn.execute(
                        "DELETE FROM edges WHERE src_id = ? OR dst_id = ?",
                        (focus, focus),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?",
                        (focus, focus),
                    )
                    conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entities WHERE entity_id = ?", (focus,))
                    applied += 1
    return applied


def _escape_turtle_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
    return f'"{escaped}"'


def serialize_graph_turtle(
    conn: sqlite3.Connection,
    violations: Sequence[SHACLViolation] | None = None,
) -> str:
    """Serialize the yaw.db SKOS/RDF/OWL closed-world knowledge graph into W3C Turtle format."""
    if violations is None:
        violations = _scan_shacl_violations(conn)

    lines: list[str] = [
        "@prefix : <urn:graph:> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "",
    ]

    labels_by_eid: dict[str, list[tuple[str, str, str]]] = {}
    for row in conn.execute(
        """
        SELECT entity_id, label_kind, raw_label, label_norm, chunk_id
        FROM entity_labels
        ORDER BY entity_id ASC, label_kind ASC, label_norm ASC
        """
    ):
        eid = str(row[0])
        lkind = str(row[1])
        lbl = str(row[2]) if str(row[2]) else str(row[3])
        cid = str(row[4])
        labels_by_eid.setdefault(eid, []).append((lkind, lbl, cid))

    sem_edges_by_eid: dict[str, list[tuple[str, str]]] = {}
    for row in conn.execute(
        """
        SELECT src_id, kind, dst_id
        FROM edges
        WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')
        ORDER BY src_id ASC, kind ASC, dst_id ASC
        """
    ):
        sem_edges_by_eid.setdefault(str(row[0]), []).append((str(row[1]), str(row[2])))

    for row in conn.execute("SELECT entity_id, name, definition, confidence FROM entities ORDER BY entity_id ASC"):
        eid, ename, edef, econf = str(row[0]), str(row[1]), str(row[2]), row[3]
        pred_objs: list[str] = ["a skos:Concept"]
        seen_labels: set[tuple[str, str]] = set()
        prov_chunks: set[str] = set()
        for lkind, lbl, cid in labels_by_eid.get(eid, []):
            if (lkind, lbl) not in seen_labels:
                seen_labels.add((lkind, lbl))
                pred_objs.append(f"{lkind} {_escape_turtle_literal(lbl)}")
            if cid and not cid.startswith("__"):
                prov_chunks.add(cid)
        if not seen_labels:
            pred_objs.append(f"skos:prefLabel {_escape_turtle_literal(ename)}")
        if edef.strip():
            pred_objs.append(f"skos:definition {_escape_turtle_literal(edef.strip())}")
        if econf is not None:
            pred_objs.append(f'prov:value "{float(econf):.6f}"^^xsd:double')
        for kind, dst_id in sem_edges_by_eid.get(eid, []):
            pred_objs.append(f"{kind} :{dst_id}")
        for cid in sorted(prov_chunks):
            pred_objs.append(f"prov:wasDerivedFrom :{cid}")

        lines.append(f":{eid} " + " ;\n    ".join(pred_objs) + " .")
        lines.append("")

    for row in conn.execute(
        """
        SELECT
            fact_id,
            subj_id,
            pred,
            obj_id,
            definition,
            COALESCE(obj_kind, 'entity'),
            obj_literal_lex,
            COALESCE(obj_datatype, ''),
            COALESCE(obj_lang, ''),
            confidence
        FROM facts
        ORDER BY fact_id ASC
        """
    ):
        fid, sid, pred, oid, fdef, okind, olex, odt, olang, fconf = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]) if row[3] is not None else "",
            str(row[4]),
            str(row[5]),
            str(row[6]) if row[6] is not None else "",
            str(row[7]),
            str(row[8]),
            row[9],
        )
        if okind == "literal":
            lit_token = _escape_turtle_literal(olex)
            clean_odt = odt.strip()
            if clean_odt.startswith("<") and clean_odt.endswith(">"):
                clean_odt = clean_odt[1:-1].strip()
            if olang:
                lit_token = f"{lit_token}@{olang}"
            elif clean_odt and clean_odt not in ("", "rdfs:Literal", "xsd:string"):
                if clean_odt.startswith(
                    ("xsd:", "rdf:", "rdfs:", "owl:", "skos:", "prov:", "sh:")
                ) and not clean_odt.startswith(("http://", "https://", "urn:")):
                    lit_token = f"{lit_token}^^{clean_odt}"
                else:
                    lit_token = f"{lit_token}^^<{clean_odt}>"
            rdf_obj = lit_token
        else:
            rdf_obj = f":{oid}"
        stmt_parts = [
            "a rdf:Statement",
            f"rdf:subject :{sid}",
            f"rdf:predicate {_escape_turtle_literal(pred)}",
            f"rdf:object {rdf_obj}",
        ]
        if fdef.strip():
            stmt_parts.append(f"skos:definition {_escape_turtle_literal(fdef.strip())}")
        if fconf is not None:
            stmt_parts.append(f'prov:value "{float(fconf):.6f}"^^xsd:double')
        lines.append(f":{fid} " + " ;\n    ".join(stmt_parts) + " .")
        lines.append("")

    conforms_str = "true" if len(violations) == 0 else "false"
    report_parts = [
        "a sh:ValidationReport",
        f"sh:conforms {conforms_str}",
    ]
    for v in violations:
        report_parts.append(
            "sh:result [ "
            "a sh:ValidationResult ; "
            f"sh:sourceShape {_escape_turtle_literal(v.shape)} ; "
            f"sh:focusNode {_escape_turtle_literal(v.focus_node)} ; "
            f"sh:resultPath {_escape_turtle_literal(v.result_path)} ; "
            f"sh:resultMessage {_escape_turtle_literal(v.message)} "
            "]"
        )
    lines.append(":ValidationReport " + " ;\n    ".join(report_parts) + " .")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def format_subgraph_turtle(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str] = (),
    seed_facts: Sequence[tuple[str, str, str]] = (),
    *,
    max_chars: int = 10000,
) -> str:
    """Format the closed-world SKOS/OWL/RDF subgraph induced by chunk_ids and seed_facts in W3C Turtle."""
    active_entity_ids: set[str] = set()
    for item in seed_facts:
        if len(item) >= 3:
            s, _, o = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
            active_entity_ids.add(s)
            active_entity_ids.add(o)

    chunk_list = list(dict.fromkeys(chunk_ids))
    fact_rows: list[sqlite3.Row] = []
    if chunk_list:
        ph = ",".join("?" for _ in chunk_list)
        fact_rows = conn.execute(
            f"""
            SELECT f.fact_id, f.subj_id, f.pred, f.obj_kind, f.obj_id, f.obj_literal_lex,
                   COALESCE(f.definition, ''), COALESCE(f.obj_datatype, ''), COALESCE(f.obj_lang, ''), f.confidence
            FROM facts f
            JOIN fact_sources fs ON f.fact_id = fs.fact_id
            WHERE fs.chunk_id IN ({ph})
            ORDER BY f.fact_id ASC
            """,
            chunk_list,
        ).fetchall()
        for r in fact_rows:
            active_entity_ids.add(str(r[1]))
            if r[3] == "entity" and r[4]:
                active_entity_ids.add(str(r[4]))

        ent_rows = conn.execute(
            f"SELECT DISTINCT entity_id FROM entity_labels WHERE chunk_id IN ({ph})",
            chunk_list,
        ).fetchall()
        for (eid,) in ent_rows:
            active_entity_ids.add(str(eid))

    lines: list[str] = [W3C_PREFIX_HEADER.strip(), ""]

    if active_entity_ids:
        ph_ent = ",".join("?" for _ in active_entity_ids)
        entities_data = conn.execute(
            f"""
            SELECT entity_id, name, definition, confidence
            FROM entities
            WHERE entity_id IN ({ph_ent})
            ORDER BY entity_id ASC
            """,
            list(active_entity_ids),
        ).fetchall()

        labels_data = conn.execute(
            f"""
            SELECT entity_id, label_kind, raw_label, label_norm
            FROM entity_labels
            WHERE entity_id IN ({ph_ent})
            ORDER BY entity_id ASC, label_kind ASC
            """,
            list(active_entity_ids),
        ).fetchall()
        labels_by_eid: dict[str, list[tuple[str, str]]] = {}
        for r in labels_data:
            eid_l = str(r[0])
            lbl = str(r[2]) if str(r[2]) else str(r[3])
            labels_by_eid.setdefault(eid_l, []).append((str(r[1]), lbl))

        sem_edges_data = conn.execute(
            f"""
            SELECT src_id, kind, dst_id
            FROM edges
            WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')
              AND (src_id IN ({ph_ent}) OR dst_id IN ({ph_ent}))
            ORDER BY src_id ASC, kind ASC
            """,
            list(active_entity_ids) * 2,
        ).fetchall()
        edges_by_eid: dict[str, list[tuple[str, str]]] = {}
        for r in sem_edges_data:
            edges_by_eid.setdefault(str(r[0]), []).append((str(r[1]), str(r[2])))

        for row in entities_data:
            eid, ename, edef, econf = str(row[0]), str(row[1]), str(row[2]), row[3]
            pred_objs = ["a skos:Concept"]
            seen_lbls: set[tuple[str, str]] = set()
            for lkind, lbl in labels_by_eid.get(eid, []):
                if (lkind, lbl) not in seen_lbls:
                    seen_lbls.add((lkind, lbl))
                    pred_objs.append(f"{lkind} {_escape_turtle_literal(lbl)}")
            if not seen_lbls:
                pred_objs.append(f"skos:prefLabel {_escape_turtle_literal(ename)}")
            if edef.strip():
                pred_objs.append(f"skos:definition {_escape_turtle_literal(edef.strip())}")
            if econf is not None:
                pred_objs.append(f'prov:value "{float(econf):.6f}"^^xsd:double')
            for k, d_id in edges_by_eid.get(eid, []):
                pred_objs.append(f"{k} :{d_id}")
            lines.append(f":{eid} " + " ;\n    ".join(pred_objs) + " .")
            lines.append("")

    for r in fact_rows:
        fid, sid, pred, okind, oid, olex, fdef, odt, olang, fconf = (
            str(r[0]),
            str(r[1]),
            str(r[2]),
            str(r[3]),
            str(r[4] or ""),
            str(r[5] or ""),
            str(r[6]),
            str(r[7]),
            str(r[8]),
            r[9],
        )
        if okind == "literal":
            lit_token = _escape_turtle_literal(olex)
            clean_odt = odt.strip()
            if clean_odt.startswith("<") and clean_odt.endswith(">"):
                clean_odt = clean_odt[1:-1].strip()
            if olang:
                lit_token = f"{lit_token}@{olang}"
            elif clean_odt and clean_odt not in ("", "rdfs:Literal", "xsd:string"):
                if clean_odt.startswith(("xsd:", "rdf:", "rdfs:", "owl:", "skos:", "prov:", "sh:")):
                    lit_token = f"{lit_token}^^{clean_odt}"
                else:
                    lit_token = f"{lit_token}^^<{clean_odt}>"
            rdf_obj = lit_token
        else:
            rdf_obj = f":{oid}"
        stmt_parts = [
            "a rdf:Statement",
            f"rdf:subject :{sid}",
            f"rdf:predicate {_escape_turtle_literal(pred)}",
            f"rdf:object {rdf_obj}",
        ]
        if fdef.strip():
            stmt_parts.append(f"skos:definition {_escape_turtle_literal(fdef.strip())}")
        if fconf is not None:
            stmt_parts.append(f'prov:value "{float(fconf):.6f}"^^xsd:double')
        lines.append(f":{fid} " + " ;\n    ".join(stmt_parts) + " .")
        lines.append("")

    res = "\n".join(lines).strip() + "\n"
    if len(res) > max_chars:
        res = res[:max_chars]
    return res


def query_graph_provenance(conn: sqlite3.Connection) -> dict[str, Any]:
    """Query epistemic confidence statistics and provenance metadata across entities, facts, and edges."""
    ent_stats = conn.execute(
        """
        SELECT
            COUNT(*),
            COUNT(confidence),
            AVG(confidence),
            MIN(confidence),
            MAX(confidence)
        FROM entities
        """
    ).fetchone() or (0, 0, None, None, None)

    fact_stats = conn.execute(
        """
        SELECT
            COUNT(*),
            COUNT(confidence),
            AVG(confidence),
            MIN(confidence),
            MAX(confidence)
        FROM facts
        """
    ).fetchone() or (0, 0, None, None, None)

    edge_stats = conn.execute(
        """
        SELECT
            COUNT(*),
            AVG(weight),
            MIN(weight),
            MAX(weight)
        FROM edges
        """
    ).fetchone() or (0, None, None, None)

    src_stats = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM entity_labels),
            (SELECT COUNT(*) FROM fact_sources),
            (SELECT COUNT(*) FROM edge_sources)
        """
    ).fetchone() or (0, 0, 0)

    return {
        "entities": {
            "total": int(ent_stats[0]),
            "calibrated": int(ent_stats[1]),
            "avg_confidence": float(ent_stats[2]) if ent_stats[2] is not None else None,
            "min_confidence": float(ent_stats[3]) if ent_stats[3] is not None else None,
            "max_confidence": float(ent_stats[4]) if ent_stats[4] is not None else None,
        },
        "facts": {
            "total": int(fact_stats[0]),
            "calibrated": int(fact_stats[1]),
            "avg_confidence": float(fact_stats[2]) if fact_stats[2] is not None else None,
            "min_confidence": float(fact_stats[3]) if fact_stats[3] is not None else None,
            "max_confidence": float(fact_stats[4]) if fact_stats[4] is not None else None,
        },
        "edges": {
            "total": int(edge_stats[0]),
            "avg_weight": float(edge_stats[1]) if edge_stats[1] is not None else None,
            "min_weight": float(edge_stats[2]) if edge_stats[2] is not None else None,
            "max_weight": float(edge_stats[3]) if edge_stats[3] is not None else None,
        },
        "provenance_sources": {
            "entity_labels": int(src_stats[0]),
            "fact_sources": int(src_stats[1]),
            "edge_sources": int(src_stats[2]),
        },
    }


def auto_heal_deterministic_shacl(conn: sqlite3.Connection) -> int:
    """Deterministically auto-heal closed-world SHACL violations on yaw.db without LLM inference.

    Executes:
    1. Prune scalar entities via remove_scalar_entity repair action.
    2. Cut skos:broader cycles at the weakest empirical link.
    3. Empirical Bayes S27 & Cagle resolution:
       - Step 1: Existing fact check -> delete conflicting related edge.
       - Step 2: Live empirical database means gate -> Cagle reification to fact.
       - Step 3: Bayes Factor comparison:
           - w_chain >= w_rel: delete skos:related edge.
           - w_rel > w_chain: keep skos:related edge, delete weakest broader link,
             and if k >= 2 convert it to skos:related preserving calibrated weight.
    """
    repaired = 0
    with write_tx(conn):
        # 1. Prune scalar entities
        scalar_eids = [
            str(r[0])
            for r in conn.execute("SELECT entity_id, name FROM entities").fetchall()
            if detect_scalar_literal(str(r[1]))[0]
        ]
        for eid in scalar_eids:
            repaired += apply_shacl_repairs(conn, [{"focus_node": eid, "action": "remove_scalar_entity", "value": ""}])

        # 2. Break skos:broader cycles at minimum-weight edge
        cycle_nodes = [
            str(r[0])
            for r in conn.execute(
                """
                WITH RECURSIVE broader_tc(narrower_id, broader_id, path_str) AS (
                    SELECT src_id, dst_id, src_id || '->' || dst_id
                    FROM edges WHERE kind = 'skos:broader'
                    UNION
                    SELECT tc.narrower_id, e.dst_id, tc.path_str || '->' || e.dst_id
                    FROM broader_tc tc
                    JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
                    WHERE INSTR(tc.path_str, e.dst_id) = 0 OR e.dst_id = tc.narrower_id
                    LIMIT 10000
                )
                SELECT DISTINCT narrower_id
                FROM broader_tc
                WHERE narrower_id = broader_id
                """
            ).fetchall()
        ]
        for cn in cycle_nodes:
            cycle_edge = conn.execute(
                """
                WITH RECURSIVE cycle_path(src_id, dst_id, weight, depth) AS (
                    SELECT src_id, dst_id, weight, 1
                    FROM edges WHERE kind = 'skos:broader' AND src_id = ?
                    UNION ALL
                    SELECT e.src_id, e.dst_id, e.weight, cp.depth + 1
                    FROM cycle_path cp
                    JOIN edges e ON e.src_id = cp.dst_id AND e.kind = 'skos:broader'
                    WHERE cp.depth < 10
                )
                SELECT src_id, dst_id, weight
                FROM cycle_path
                WHERE dst_id = ?
                ORDER BY weight ASC
                LIMIT 1
                """,
                (cn, cn),
            ).fetchone()
            if cycle_edge is not None:
                c_src, c_dst = str(cycle_edge[0]), str(cycle_edge[1])
                conn.execute(
                    "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                    (c_src, c_dst),
                )
                conn.execute(
                    "DELETE FROM edge_sources WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                    (c_src, c_dst),
                )
                repaired += 1

        # 3. Empirical Bayes S27 & Cagle resolution
        avg_b_row = conn.execute("SELECT AVG(weight) FROM edges WHERE kind = 'skos:broader'").fetchone()
        avg_broader = float(avg_b_row[0]) if avg_b_row and avg_b_row[0] is not None else 0.5

        avg_r_row = conn.execute("SELECT AVG(weight) FROM edges WHERE kind = 'skos:related'").fetchone()
        avg_rel = float(avg_r_row[0]) if avg_r_row and avg_r_row[0] is not None else 0.5

        avg_f_row = conn.execute("SELECT AVG(confidence) FROM facts WHERE confidence IS NOT NULL").fetchone()
        avg_fact = float(avg_f_row[0]) if avg_f_row and avg_f_row[0] is not None else 0.5

        s27_pairs = conn.execute(
            """
            WITH RECURSIVE broader_tc(narrower_id, broader_id) AS (
                SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'
                UNION
                SELECT tc.narrower_id, e.dst_id
                FROM broader_tc tc
                JOIN edges e ON e.src_id = tc.broader_id AND e.kind = 'skos:broader'
                LIMIT 10000
            )
            SELECT DISTINCT tc.narrower_id, tc.broader_id
            FROM broader_tc tc
            INNER JOIN edges rel
              ON rel.kind = 'skos:related'
             AND (
                 (rel.src_id = tc.narrower_id AND rel.dst_id = tc.broader_id)
                 OR (rel.src_id = tc.broader_id AND rel.dst_id = tc.narrower_id)
             )
            WHERE tc.narrower_id != tc.broader_id
            ORDER BY tc.narrower_id ASC, tc.broader_id ASC
            """
        ).fetchall()

        for row in s27_pairs:
            a_id, c_id = str(row[0]), str(row[1])

            rel_row = conn.execute(
                """
                SELECT weight FROM edges
                WHERE kind = 'skos:related'
                  AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                LIMIT 1
                """,
                (a_id, c_id, c_id, a_id),
            ).fetchone()
            w_rel = float(rel_row[0]) if rel_row and rel_row[0] is not None else 0.8

            # Step 1: Existing fact check (Deterministic Cagle Short-Circuit)
            fact_exists = (
                conn.execute(
                    """
                SELECT 1 FROM facts
                WHERE (subj_id = ? AND obj_id = ?) OR (subj_id = ? AND obj_id = ?)
                LIMIT 1
                """,
                    (a_id, c_id, c_id, a_id),
                ).fetchone()
                is not None
            )

            if fact_exists:
                conn.execute(
                    """
                    DELETE FROM edges
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (a_id, c_id, c_id, a_id),
                )
                conn.execute(
                    """
                    DELETE FROM edge_sources
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (a_id, c_id, c_id, a_id),
                )
                repaired += 1
                continue

            # Find path in skos:broader from A to C
            path_row = conn.execute(
                """
                WITH RECURSIVE path_finder(curr_id, path_str, min_w, hop_count) AS (
                    SELECT dst_id, src_id || '->' || dst_id, weight, 1
                    FROM edges
                    WHERE kind = 'skos:broader' AND src_id = ?
                    UNION ALL
                    SELECT e.dst_id, pf.path_str || '->' || e.dst_id, MIN(pf.min_w, e.weight), pf.hop_count + 1
                    FROM path_finder pf
                    JOIN edges e ON e.src_id = pf.curr_id AND e.kind = 'skos:broader'
                    WHERE pf.hop_count < 10
                )
                SELECT path_str, hop_count
                FROM path_finder
                WHERE curr_id = ?
                ORDER BY hop_count ASC
                LIMIT 1
                """,
                (a_id, c_id),
            ).fetchone()

            if path_row is None:
                continue

            path_str = str(path_row[0])
            nodes_in_path = path_str.split("->")
            path_edges: list[tuple[str, str, float]] = []
            for i in range(len(nodes_in_path) - 1):
                u, v = nodes_in_path[i], nodes_in_path[i + 1]
                ew_row = conn.execute(
                    "SELECT weight FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                    (u, v),
                ).fetchone()
                ew = float(ew_row[0]) if ew_row and ew_row[0] is not None else 0.8
                path_edges.append((u, v, ew))

            w_chain = pool_confidences(*[ew for _, _, ew in path_edges]) or 0.8

            # Step 2: Empirical Database Means Gate for Cagle Reification
            above_avg = (w_chain >= avg_broader) and (w_rel >= avg_rel)
            shared_chunk: str | None = None
            if above_avg:
                chunk_row = conn.execute(
                    """
                    SELECT c1.chunk_id
                    FROM (
                        SELECT chunk_id FROM entity_labels WHERE entity_id = ?
                        UNION
                        SELECT fs.chunk_id FROM fact_sources fs
                        JOIN facts f ON f.fact_id = fs.fact_id WHERE f.subj_id = ? OR f.obj_id = ?
                    ) c1
                    INNER JOIN (
                        SELECT chunk_id FROM entity_labels WHERE entity_id = ?
                        UNION
                        SELECT fs.chunk_id FROM fact_sources fs
                        JOIN facts f ON f.fact_id = fs.fact_id WHERE f.subj_id = ? OR f.obj_id = ?
                    ) c2 ON c1.chunk_id = c2.chunk_id
                    LIMIT 1
                    """,
                    (a_id, a_id, a_id, c_id, c_id, c_id),
                ).fetchone()
                if chunk_row is not None:
                    shared_chunk = str(chunk_row[0])

            atomic_conf = pool_confidences(w_chain, w_rel) or 0.8
            if above_avg and shared_chunk is not None and atomic_conf >= avg_fact:
                new_fid = compute_mdhash_id(str((a_id, "relatedTo", c_id)), prefix="fact-")
                raw_json = json.dumps([a_id, "relatedTo", c_id])
                defn = f"Reified from high-confidence association between {a_id} and {c_id}."
                conn.execute(
                    """
                    INSERT OR REPLACE INTO facts
                    (fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, confidence)
                    VALUES (?, ?, 'relatedTo', ?, ?, ?, 'entity', ?)
                    """,
                    (new_fid, a_id, c_id, raw_json, defn, atomic_conf),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id, atomic_confidence)
                    VALUES (?, ?, ?)
                    """,
                    (new_fid, shared_chunk, atomic_conf),
                )
                conn.execute(
                    """
                    DELETE FROM edges
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (a_id, c_id, c_id, a_id),
                )
                conn.execute(
                    """
                    DELETE FROM edge_sources
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (a_id, c_id, c_id, a_id),
                )
                repaired += 1
                continue

            # Step 3: Direct Bayes Factor Comparison (w_chain vs w_rel)
            if w_chain >= w_rel:
                conn.execute(
                    """
                    DELETE FROM edges
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (a_id, c_id, c_id, a_id),
                )
                conn.execute(
                    """
                    DELETE FROM edge_sources
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (a_id, c_id, c_id, a_id),
                )
                repaired += 1
            else:
                weakest_u, weakest_v, min_w = min(path_edges, key=lambda e: e[2])
                conn.execute(
                    "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                    (weakest_u, weakest_v),
                )
                conn.execute(
                    "DELETE FROM edge_sources WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                    (weakest_u, weakest_v),
                )
                if len(path_edges) >= 2:
                    upsert_semantic_edges(conn, [(weakest_u, weakest_v, "skos:related", min_w)])
                repaired += 1

    return repaired
