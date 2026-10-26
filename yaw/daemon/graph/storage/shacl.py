"""Closed-world SQL SHACL linter, repair execution, focus-node inspection, and W3C Turtle serializer."""

import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from yaw.core import (
    SHACLViolation,
    check_lexical_subsumption_inversion,
    detect_scalar_literal,
    is_version_or_ordinal_sibling,
    text_processing,
)

from .db import YawConnection, batched, write_tx
from .graph_ops import (
    LABEL_KIND_PRIORITY,
    get_chunk_pointers,
    get_existing_ids,
    merge_entities,
    upsert_entity_labels,
    upsert_semantic_edges,
)
from .hydration import (
    hydrate_ast_parent_context,
    hydrate_chunk_pointer,
    resolve_one_hop_upward_ast_bounds,
)


def _inspect_single_entity_or_fact(
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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
    node_info = info if info is not None else _inspect_single_entity_or_fact(conn, entity_id, corpus_root)
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

    is_subj_or_ent_obj = any(
        f["subj_id"] == eid or (f["obj_id"] == eid and f.get("obj_kind", "entity") == "entity") for f in facts
    )
    lit_facts_as_obj = [f for f in facts if f["obj_id"] == eid and f.get("obj_kind") == "literal"]
    is_literal_node = bool(lit_facts_as_obj and not is_subj_or_ent_obj)
    lit_datatypes = sorted(
        {str(f.get("obj_datatype") or "rdfs:Literal") for f in lit_facts_as_obj if f.get("obj_datatype")}
    )
    lit_langs = sorted({str(f.get("obj_lang")) for f in lit_facts_as_obj if f.get("obj_lang")})

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
    if is_literal_node:
        dt_str = ", ".join(lit_datatypes) if lit_datatypes else "rdfs:Literal"
        lang_str = f" @{','.join(lit_langs)}" if lit_langs else ""
        lines.append(f"a rdfs:Literal ; # datatype: {dt_str}{lang_str}")
    else:
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
    conn: sqlite3.Connection | YawConnection,
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
    ia = info_a if info_a is not None else _inspect_single_entity_or_fact(conn, entity_a, corpus_root)
    ib = info_b if info_b is not None else _inspect_single_entity_or_fact(conn, entity_b, corpus_root)

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


def iter_disambiguation_dossiers(
    conn: sqlite3.Connection | YawConnection,
    candidates: Sequence[dict[str, Any]],
    corpus_root: Path,
    *,
    ast_context_hops: int = 0,
    batch_size: int = 16,
    max_chars: int = 40000,
) -> Iterator[tuple[list[dict[str, Any]], str]]:
    """Pack complete pair evidence with each entity profile rendered once per request.

    The caller must keep the graph snapshot stable while consuming this iterator.
    Each yielded batch owns only its bounded set of entity inspections. Every
    pair receives both profiles; a full batch flushes instead of dropping evidence.
    """
    batch: list[dict[str, Any]] = []
    headers: list[str] = []
    profiles: dict[str, str] = {}
    infos: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}

    def render(
        items: Sequence[dict[str, Any]],
        pair_headers: Sequence[str],
        entity_profiles: Mapping[str, str],
        source_excerpts: Mapping[str, str],
    ) -> str:
        index = [{k: v for k, v in c.items() if k not in ("def_a", "def_b")} for c in items]
        return (
            "# Candidate Pairs Index\n```json\n"
            + json.dumps(index, ensure_ascii=False)
            + "\n```\n\n"
            + "\n\n".join(pair_headers)
            + "\n\n# Shared Entity Profiles\n\n"
            + "\n\n".join(entity_profiles.values())
            + "\n\n# Shared Source Excerpts\n\n"
            + "\n\n".join(source_excerpts.values())
        )

    for candidate in candidates:
        ea, eb = str(candidate["entity_a"]), str(candidate["entity_b"])
        new_profiles = {}
        trial_sources = dict(sources)
        for eid in (ea, eb):
            if eid not in infos:
                infos[eid] = _inspect_single_entity_or_fact(conn, eid, corpus_root)
            if eid not in profiles:
                new_profiles[eid] = format_entity_skos_markdown(
                    conn,
                    eid,
                    corpus_root,
                    ast_context_hops=ast_context_hops,
                    info=infos[eid],
                    shared_source_excerpts=trial_sources,
                )
        header = format_pair_disambiguation_dossier(
            conn,
            ea,
            eb,
            corpus_root,
            ast_context_hops=ast_context_hops,
            dense_sim=float(candidate["dense_sim"]),
            lexical_sim=float(candidate["lexical_sim"]),
            info_a=infos[ea],
            info_b=infos[eb],
            include_entity_profiles=False,
        )
        trial_profiles = profiles | new_profiles
        trial = render([*batch, candidate], [*headers, header], trial_profiles, trial_sources)
        if batch and (len(batch) >= batch_size or len(trial) > max_chars):
            yield batch, render(batch, headers, profiles, sources)
            # Only carry the next pair across the flush; don't retain the graph.
            infos = {eid: infos[eid] for eid in (ea, eb)}
            trial_profiles = {eid: trial_profiles[eid] for eid in (ea, eb)}
            source_ids = dict.fromkeys(
                cm["chunk_id"] for eid in (ea, eb) for cm in (infos[eid].get("chunks_meta") or [])[:2]
            )
            trial_sources = {cid: trial_sources[cid] for cid in source_ids if cid in trial_sources}
            batch, headers, profiles = [], [], {}
            trial = render([candidate], [header], trial_profiles, trial_sources)
        if len(trial) > max_chars:
            raise ValueError(f"Complete disambiguation evidence for {ea}|{eb} exceeds {max_chars} characters")
        batch.append(candidate)
        headers.append(header)
        profiles = trial_profiles
        sources = trial_sources
    if batch:
        yield batch, render(batch, headers, profiles, sources)


def inspect_focus_node(
    conn: sqlite3.Connection | YawConnection,
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
            info_a = _inspect_single_entity_or_fact(conn, node_a, corpus_root)
            info_b = _inspect_single_entity_or_fact(conn, node_b, corpus_root)
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

    single = _inspect_single_entity_or_fact(conn, clean_focus, corpus_root)
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


def _scan_shacl_violations(conn: sqlite3.Connection | YawConnection) -> list[SHACLViolation]:
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

    # 4. ConceptShape/homographDifferentFrom: Distinct entities sharing a skos:prefLabel norm must either have
    # distinct definitions with owl:differentFrom, or be merged if their definitions are identical.
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
            WHERE e.src_id = a.entity_id
              AND e.dst_id = b.entity_id
              AND e.kind = 'owl:differentFrom'
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
    # and non-empty definition
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
            COALESCE(f.obj_kind, 'entity') AS obj_kind
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
        ORDER BY f.fact_id ASC
        """
    ):
        fid, sid, oid, fdef, has_s, has_o, okind = (
            str(row[0]),
            str(row[1]),
            str(row[2]) if row[2] is not None else "",
            str(row[3]),
            bool(row[4]),
            bool(row[5]),
            str(row[6]),
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
          AND NOT EXISTS (
              SELECT 1 FROM entities ea
              INNER JOIN entities eb ON ea.entity_id = a.src_id AND eb.entity_id = a.dst_id
              WHERE LOWER(TRIM(ea.name)) != '' AND LOWER(TRIM(ea.name)) = LOWER(TRIM(eb.name))
          )
          AND NOT EXISTS (
              SELECT 1 FROM entity_labels la
              INNER JOIN entity_labels lb
                ON la.label_norm = lb.label_norm
               AND la.label_kind = 'skos:prefLabel'
               AND lb.label_kind = 'skos:prefLabel'
              WHERE la.entity_id = a.src_id AND lb.entity_id = a.dst_id
          )
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

    # 9. ConceptShape/lexicalSubsumption & 10. ConceptShape/siblingNotBroader on skos:broader edges
    for row in conn.execute(
        """
        SELECT e.src_id, e.dst_id, es.name, ed.name
        FROM edges e
        INNER JOIN entities es ON es.entity_id = e.src_id
        INNER JOIN entities ed ON ed.entity_id = e.dst_id
        WHERE e.kind = 'skos:broader'
        ORDER BY e.src_id ASC, e.dst_id ASC
        """
    ):
        s_id, d_id, s_name, d_name = str(row[0]), str(row[1]), str(row[2]), str(row[3])
        if check_lexical_subsumption_inversion(s_name, d_name):
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/lexicalSubsumption",
                    focus_node=f"{s_id}->{d_id}",
                    result_path="skos:broader",
                    message=f"Broader edge ({s_id} -> {d_id}) violates lexical subsumption ({s_name!r} vs {d_name!r}).",
                )
            )
        if is_version_or_ordinal_sibling(s_name, d_name):
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/siblingNotBroader",
                    focus_node=f"{s_id}->{d_id}",
                    result_path="skos:broader",
                    message=(
                        f"Broader edge ({s_id} -> {d_id}) connects version/ordinal siblings ({s_name!r} vs {d_name!r})."
                    ),
                )
            )

    # 11. ProvenanceShape/factSources: Every fact must have at least one valid chunk entry in fact_sources
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

    return violations


def lint_graph_shacl(
    conn: sqlite3.Connection | YawConnection,
    *,
    auto_heal: bool = False,
    heal_definitions: bool = True,
    heal_homographs: bool = True,
) -> list[SHACLViolation]:
    """Run closed-world SQL SHACL validation shapes over yaw.db and optionally apply deterministic SQL repairs."""
    violations = _scan_shacl_violations(conn)
    if not violations or not auto_heal:
        return violations

    with write_tx(conn):
        # Heal 1: ConceptShape/prefLabel
        missing_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el
                WHERE el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            )
            """
        ).fetchall()
        for r in missing_pref:
            eid, ename = str(r[0]), str(r[1])
            tok_norm = str(text_processing(ename)) or ename.strip().lower()
            raw_norm = ename.strip().lower()
            alt_row = conn.execute(
                """
                SELECT label_norm, raw_label
                FROM entity_labels
                WHERE entity_id = ?
                ORDER BY CASE WHEN label_norm = ? THEN 0
                              WHEN label_norm = ? THEN 1
                              ELSE 2 END ASC,
                         CASE label_kind WHEN 'skos:altLabel' THEN 1 ELSE 2 END ASC,
                         label_norm ASC
                LIMIT 1
                """,
                (eid, tok_norm, raw_norm),
            ).fetchone()
            if alt_row is not None:
                lnorm, rlabel = str(alt_row[0]), str(alt_row[1])
                conn.execute(
                    "UPDATE entity_labels SET label_kind = 'skos:prefLabel' WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
                if rlabel and (not ename or lnorm not in (tok_norm, raw_norm)):
                    conn.execute("UPDATE entities SET name = ? WHERE entity_id = ?", (rlabel, eid))
            else:
                norm = str(text_processing(ename)) or ename.strip().lower() or eid.lower()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                    VALUES (?, ?, 'skos:prefLabel', ?, '__healed__')
                    """,
                    (norm, eid, ename or eid),
                )

        multi_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            INNER JOIN entity_labels el ON el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            GROUP BY e.entity_id
            HAVING COUNT(DISTINCT el.label_norm) > 1
            """
        ).fetchall()
        for r in multi_pref:
            eid, ename = str(r[0]), str(r[1])
            tok_norm = str(text_processing(ename))
            raw_norm = ename.strip().lower()
            norms = [
                str(nr[0])
                for nr in conn.execute(
                    """
                    SELECT DISTINCT label_norm FROM entity_labels
                    WHERE entity_id = ? AND label_kind = 'skos:prefLabel'
                    ORDER BY label_norm ASC
                    """,
                    (eid,),
                )
            ]
            winner = tok_norm if (tok_norm and tok_norm in norms) else (raw_norm if raw_norm in norms else norms[0])
            conn.execute(
                """
                UPDATE entity_labels
                SET label_kind = 'skos:altLabel'
                WHERE entity_id = ? AND label_kind = 'skos:prefLabel' AND label_norm != ?
                """,
                (eid, winner),
            )

        # Heal 2: ConceptShape/S13-disjointLabels
        s13_rows = conn.execute(
            """
            SELECT entity_id, label_norm
            FROM entity_labels
            GROUP BY entity_id, label_norm
            HAVING COUNT(DISTINCT label_kind) > 1
            """
        ).fetchall()
        for r in s13_rows:
            eid, lnorm = str(r[0]), str(r[1])
            kinds = [
                str(kr[0])
                for kr in conn.execute(
                    "SELECT DISTINCT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
            ]
            best = max(kinds, key=lambda k: LABEL_KIND_PRIORITY.get(k, 0))
            conn.execute(
                "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                (best, eid, lnorm),
            )

        s13_canon_rows = conn.execute(
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
        ).fetchall()
        for r in s13_canon_rows:
            eid, lnorm, ename, has_other_pref = str(r[0]), str(r[1]), str(r[2]), bool(r[3])
            raw_norm = ename.strip().lower()
            canon_norm = str(text_processing(ename)) or raw_norm
            if lnorm == canon_norm or (lnorm == raw_norm and not has_other_pref):
                conn.execute(
                    """
                    UPDATE entity_labels
                    SET label_kind = 'skos:altLabel'
                    WHERE entity_id = ? AND label_kind = 'skos:prefLabel' AND label_norm != ?
                    """,
                    (eid, lnorm),
                )
                conn.execute(
                    "UPDATE entity_labels SET label_kind = 'skos:prefLabel' WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )

        # Heal 3: ConceptShape/homographDifferentFrom
        # 3a. Identical-definition duplicates sharing skos:prefLabel are NEVER homographs;
        # deterministically collapse them via merge_entities even before RLM healing.
        identical_def_clones = conn.execute(
            """
            SELECT a.entity_id, b.entity_id
            FROM entity_labels a
            INNER JOIN entity_labels b
              ON a.label_norm = b.label_norm
             AND a.label_kind = 'skos:prefLabel'
             AND b.label_kind = 'skos:prefLabel'
             AND a.entity_id < b.entity_id
            INNER JOIN entities ea ON ea.entity_id = a.entity_id
            INNER JOIN entities eb ON eb.entity_id = b.entity_id
            WHERE LOWER(TRIM(COALESCE(ea.definition, ''))) != ''
              AND LOWER(TRIM(COALESCE(ea.definition, ''))) = LOWER(TRIM(COALESCE(eb.definition, '')))
            GROUP BY a.entity_id, b.entity_id
            ORDER BY a.entity_id ASC, b.entity_id ASC
            """
        ).fetchall()
        merged_clone_map: dict[str, str] = {}
        for r in identical_def_clones:
            dst_eid = str(r[0])
            src_eid = str(r[1])
            while dst_eid in merged_clone_map:
                dst_eid = merged_clone_map[dst_eid]
            while src_eid in merged_clone_map:
                src_eid = merged_clone_map[src_eid]
            if dst_eid != src_eid:
                merge_entities(conn, src_eid, dst_eid, label_kind="skos:altLabel")
                merged_clone_map[src_eid] = dst_eid

        # 3b. Only stamp fallback owl:differentFrom on distinct-definition homographs when
        # heal_homographs=True (post-RLM safety pass). In the pre-RLM pass (heal_homographs=False),
        # leave ambiguous homographs open so shacl_healer_rlm can inspect their AST dossiers.
        if heal_homographs:
            homographs = conn.execute(
                """
                SELECT a.entity_id, b.entity_id
                FROM entity_labels a
                INNER JOIN entity_labels b
                  ON a.label_norm = b.label_norm
                 AND a.label_kind = 'skos:prefLabel'
                 AND b.label_kind = 'skos:prefLabel'
                 AND a.entity_id < b.entity_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM edges e
                    WHERE e.src_id = a.entity_id
                      AND e.dst_id = b.entity_id
                      AND e.kind = 'owl:differentFrom'
                )
                GROUP BY a.entity_id, b.entity_id
                """
            ).fetchall()
            if homographs:
                upsert_semantic_edges(
                    conn,
                    [(str(r[0]), str(r[1]), "owl:differentFrom", 0.0) for r in homographs],
                )

        # Heal 4: ConceptShape/definition
        if heal_definitions:
            empty_ent_defs = conn.execute("SELECT entity_id, name FROM entities WHERE TRIM(definition) = ''").fetchall()
            for r in empty_ent_defs:
                eid, ename = str(r[0]), str(r[1])
                fact_row = conn.execute(
                    "SELECT raw_triple_json, definition FROM facts WHERE subj_id = ? OR obj_id = ? LIMIT 1",
                    (eid, eid),
                ).fetchone()
                if fact_row is not None and str(fact_row[1]).strip():
                    synth_def = f"{ename}: {str(fact_row[1]).strip()}"
                elif fact_row is not None:
                    try:
                        t = json.loads(fact_row[0])
                        synth_def = f"{ename} participates in relation: {t[0]} {t[1]} {t[2]}."
                    except Exception:
                        synth_def = f"Concept representing {ename}."
                else:
                    synth_def = f"Concept representing {ename}."
                conn.execute(
                    "UPDATE entities SET definition = ? WHERE entity_id = ?",
                    (synth_def, eid),
                )

        # Heal 5: StatementShape/closedWorld
        dangling_fids = [
            str(r[0])
            for r in conn.execute(
                """
                SELECT f.fact_id
                FROM facts f
                LEFT JOIN entities es ON es.entity_id = f.subj_id
                LEFT JOIN entities eo ON eo.entity_id = f.obj_id
                WHERE es.entity_id IS NULL
                   OR (COALESCE(f.obj_kind, 'entity') = 'entity' AND eo.entity_id IS NULL)
                   OR (
                       COALESCE(f.obj_kind, 'entity') = 'literal'
                       AND (
                           (f.obj_id IS NOT NULL AND eo.entity_id IS NULL)
                           OR TRIM(COALESCE(f.obj_literal_lex, '')) = ''
                       )
                   )
                """
            )
        ]
        if dangling_fids:
            for batch in batched(dangling_fids):
                placeholders = ",".join("?" for _ in batch)
                conn.execute(f"DELETE FROM fact_sources WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM fact_vec WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM fact_fts WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM facts WHERE fact_id IN ({placeholders})", tuple(batch))

        if heal_definitions:
            empty_fact_defs = conn.execute(
                "SELECT fact_id, pred, raw_triple_json FROM facts WHERE TRIM(definition) = ''"
            ).fetchall()
            for r in empty_fact_defs:
                fid, pred, raw_json = str(r[0]), str(r[1]), str(r[2])
                try:
                    t = json.loads(raw_json)
                    synth_fdef = f"{t[0]} {t[1]} {t[2]}."
                except Exception:
                    synth_fdef = f"Statement with predicate {pred}."
                conn.execute(
                    "UPDATE facts SET definition = ? WHERE fact_id = ?",
                    (synth_fdef, fid),
                )

        # Heal 11: ProvenanceShape/factSources (run before Heal 8 so scalar object IDs ground chunk provenance)
        conn.execute(
            """
            DELETE FROM fact_sources
            WHERE NOT EXISTS (
                SELECT 1 FROM chunks c WHERE c.chunk_id = fact_sources.chunk_id
            )
            """
        )
        missing_fact_sources = conn.execute(
            """
            SELECT f.fact_id, f.subj_id, f.obj_id
            FROM facts f
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_sources fs
                INNER JOIN chunks c ON c.chunk_id = fs.chunk_id
                WHERE fs.fact_id = f.fact_id
            )
            ORDER BY f.fact_id ASC
            """
        ).fetchall()
        orphan_fids: list[str] = []
        for r in missing_fact_sources:
            fid, sid, oid = str(r[0]), str(r[1]), str(r[2]) if r[2] is not None else ""
            cid_row = conn.execute(
                """
                SELECT el.chunk_id
                FROM entity_labels el
                INNER JOIN chunks c ON c.chunk_id = el.chunk_id
                WHERE el.entity_id IN (?, ?)
                ORDER BY (el.entity_id = ?) DESC, el.chunk_id ASC
                LIMIT 1
                """,
                (sid, oid, sid),
            ).fetchone()
            cid = str(cid_row[0]) if cid_row is not None and str(cid_row[0]) else ""
            if not cid:
                es_row = conn.execute(
                    """
                    SELECT es.chunk_id
                    FROM edge_sources es
                    INNER JOIN chunks c ON c.chunk_id = es.chunk_id
                    WHERE (es.src_id = ? AND es.dst_id = ?)
                       OR es.src_id IN (?, ?)
                       OR es.dst_id IN (?, ?)
                    ORDER BY (es.src_id = ? AND es.dst_id = ?) DESC, es.chunk_id ASC
                    LIMIT 1
                    """,
                    (sid, oid, sid, oid, sid, oid, sid, oid),
                ).fetchone()
                cid = str(es_row[0]) if es_row is not None and str(es_row[0]) else ""
            if not cid:
                pe_row = conn.execute(
                    """
                    SELECT CASE
                        WHEN EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.src_id) THEN e.src_id
                        ELSE e.dst_id
                    END AS valid_cid
                    FROM edges e
                    WHERE e.kind = 'passage'
                      AND (
                          (e.dst_id IN (?, ?) AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.src_id))
                          OR (e.src_id IN (?, ?) AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.dst_id))
                      )
                    ORDER BY valid_cid ASC
                    LIMIT 1
                    """,
                    (sid, oid, sid, oid),
                ).fetchone()
                cid = str(pe_row[0]) if pe_row is not None and str(pe_row[0]) else ""
            if cid:
                conn.execute(
                    "INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id) VALUES (?, ?)",
                    (fid, cid),
                )
            else:
                orphan_fids.append(fid)
        if orphan_fids:
            for batch in batched(orphan_fids):
                placeholders = ",".join("?" for _ in batch)
                conn.execute(f"DELETE FROM fact_sources WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM fact_vec WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM fact_fts WHERE fact_id IN ({placeholders})", tuple(batch))
                conn.execute(f"DELETE FROM facts WHERE fact_id IN ({placeholders})", tuple(batch))

        # Heal 8: ConceptShape/noScalarLiterals
        scalar_rows = conn.execute(
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
        ).fetchall()
        for r in scalar_rows:
            eid, ename = str(r[0]), str(r[1])
            is_subj, is_ent_obj, has_skos_edge, has_lit_fact, has_passage_edge = (
                bool(r[2]),
                bool(r[3]),
                bool(r[4]),
                bool(r[5]),
                bool(r[6]),
            )
            is_pure_literal = has_lit_fact and not is_subj and not is_ent_obj
            is_scalar, _ = detect_scalar_literal(ename)
            if is_pure_literal:
                if has_skos_edge:
                    conn.execute(
                        "DELETE FROM edges WHERE (src_id = ? OR dst_id = ?) "
                        "AND kind IN ('skos:broader', 'skos:related')",
                        (eid, eid),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE (src_id = ? OR dst_id = ?) "
                        "AND kind IN ('skos:broader', 'skos:related')",
                        (eid, eid),
                    )
            elif is_scalar and (is_subj or is_ent_obj or has_skos_edge or (not has_lit_fact and not has_passage_edge)):
                # Convert obj_kind='entity' facts where scalar entity was an object to obj_kind='literal'
                conn.execute(
                    """
                    UPDATE facts
                    SET obj_kind = 'literal',
                        obj_id = CASE WHEN ? AND subj_id != ? THEN ? ELSE NULL END,
                        obj_literal_lex = ?,
                        obj_datatype = 'rdfs:Literal',
                        obj_lang = ''
                    WHERE obj_id = ? AND COALESCE(obj_kind, 'entity') = 'entity'
                    """,
                    (1 if has_passage_edge else 0, eid, eid, ename, eid),
                )
                # Literals cannot be subjects: delete subject facts
                subj_fids = [str(fr[0]) for fr in conn.execute("SELECT fact_id FROM facts WHERE subj_id = ?", (eid,))]
                if subj_fids:
                    for b in batched(subj_fids):
                        ph = ",".join("?" for _ in b)
                        conn.execute(f"DELETE FROM fact_sources WHERE fact_id IN ({ph})", tuple(b))
                        conn.execute(f"DELETE FROM fact_vec WHERE fact_id IN ({ph})", tuple(b))
                        conn.execute(f"DELETE FROM fact_fts WHERE fact_id IN ({ph})", tuple(b))
                        conn.execute(f"DELETE FROM facts WHERE fact_id IN ({ph})", tuple(b))
                if has_skos_edge:
                    conn.execute(
                        "DELETE FROM edges WHERE (src_id = ? OR dst_id = ?) "
                        "AND kind IN ('skos:broader', 'skos:related')",
                        (eid, eid),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE (src_id = ? OR dst_id = ?) "
                        "AND kind IN ('skos:broader', 'skos:related')",
                        (eid, eid),
                    )
                conn.execute(
                    """
                    DELETE FROM edges
                    WHERE kind = 'fact'
                      AND (src_id = ? OR dst_id = ?)
                      AND NOT EXISTS (
                          SELECT 1 FROM facts f
                          WHERE f.subj_id = edges.src_id AND f.obj_id = edges.dst_id
                      )
                    """,
                    (eid, eid),
                )
                conn.execute(
                    """
                    DELETE FROM edge_sources
                    WHERE kind = 'fact'
                      AND (src_id = ? OR dst_id = ?)
                      AND NOT EXISTS (
                          SELECT 1 FROM facts f
                          WHERE f.subj_id = edge_sources.src_id AND f.obj_id = edge_sources.dst_id
                      )
                    """,
                    (eid, eid),
                )
                conn.execute(
                    """
                    UPDATE edges
                    SET weight = 0.5 * COALESCE(
                        (
                            SELECT CASE WHEN COUNT(*) > 0 THEN COUNT(*) ELSE 1 END
                            FROM edge_sources es
                            WHERE es.src_id = edges.src_id
                              AND es.dst_id = edges.dst_id
                              AND es.kind = 'fact'
                        ),
                        1
                    )
                    WHERE kind = 'fact'
                      AND dst_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM facts f
                          WHERE f.subj_id = edges.src_id
                            AND f.obj_id = edges.dst_id
                            AND COALESCE(f.obj_kind, 'entity') = 'entity'
                      )
                    """,
                    (eid,),
                )
                rem_lit_fact = (
                    conn.execute(
                        "SELECT 1 FROM facts WHERE obj_id = ? AND obj_kind = 'literal' LIMIT 1",
                        (eid,),
                    ).fetchone()
                    is not None
                )
                if not rem_lit_fact and not has_passage_edge:
                    # Delete any edges touching the scalar entity
                    conn.execute("DELETE FROM edges WHERE src_id = ? OR dst_id = ?", (eid, eid))
                    conn.execute("DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?", (eid, eid))
                    # Delete entity from entity tables
                    conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (eid,))
                    conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (eid,))
                    conn.execute("DELETE FROM entities WHERE entity_id = ?", (eid,))

        # Multi-pass edge healing convergence loop (Heal 6, Heal 9, Heal 10, Heal 7)
        for _pass in range(3):
            # Heal 6: EdgeShape/closedWorld
            conn.execute(
                """
                DELETE FROM edges
                WHERE (
                    kind = 'passage'
                    AND NOT (
                        (
                            EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = edges.src_id)
                            AND EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = edges.dst_id)
                        )
                        OR (
                            EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = edges.src_id)
                            AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = edges.dst_id)
                        )
                    )
                )
                OR (
                    kind IN ('fact', 'skos:related', 'skos:broader', 'owl:differentFrom')
                    AND (
                        NOT EXISTS (SELECT 1 FROM entities ent1 WHERE ent1.entity_id = edges.src_id)
                        OR NOT EXISTS (SELECT 1 FROM entities ent2 WHERE ent2.entity_id = edges.dst_id)
                    )
                )
                """
            )
            conn.execute(
                """
                DELETE FROM edges
                WHERE kind = 'skos:related'
                  AND EXISTS (
                      SELECT 1 FROM edges b
                      WHERE b.src_id = edges.src_id
                        AND b.dst_id = edges.dst_id
                        AND b.kind = 'owl:differentFrom'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM entities ea
                      INNER JOIN entities eb ON ea.entity_id = edges.src_id AND eb.entity_id = edges.dst_id
                      WHERE LOWER(TRIM(ea.name)) != '' AND LOWER(TRIM(ea.name)) = LOWER(TRIM(eb.name))
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM entity_labels la
                      INNER JOIN entity_labels lb
                        ON la.label_norm = lb.label_norm
                       AND la.label_kind = 'skos:prefLabel'
                       AND lb.label_kind = 'skos:prefLabel'
                      WHERE la.entity_id = edges.src_id AND lb.entity_id = edges.dst_id
                  )
                """
            )
            asym_rows = conn.execute(
                """
                SELECT a.src_id, a.dst_id, a.kind, a.weight
                FROM edges a
                WHERE a.kind IN ('skos:related', 'owl:differentFrom')
                  AND NOT EXISTS (
                      SELECT 1 FROM edges b
                      WHERE b.src_id = a.dst_id AND b.dst_id = a.src_id AND b.kind = a.kind
                  )
                """
            ).fetchall()
            for r in asym_rows:
                conn.execute(
                    "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, ?, ?)",
                    (str(r[1]), str(r[0]), str(r[2]), float(r[3])),
                )

            # Heal 9: ConceptShape/lexicalSubsumption
            subsumption_rows = conn.execute(
                """
                SELECT e.src_id, e.dst_id, es.name, ed.name, e.weight
                FROM edges e
                INNER JOIN entities es ON es.entity_id = e.src_id
                INNER JOIN entities ed ON ed.entity_id = e.dst_id
                WHERE e.kind = 'skos:broader'
                """
            ).fetchall()
            for r in subsumption_rows:
                s_id, d_id, s_name, d_name, weight = str(r[0]), str(r[1]), str(r[2]), str(r[3]), float(r[4])
                if check_lexical_subsumption_inversion(s_name, d_name):
                    conn.execute(
                        "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                        (s_id, d_id),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, 'skos:broader', ?)",
                        (d_id, s_id, weight),
                    )
                    conn.execute(
                        """
                        UPDATE OR IGNORE edge_sources
                        SET src_id = ?, dst_id = ?
                        WHERE src_id = ? AND dst_id = ? AND kind = 'skos:broader'
                        """,
                        (d_id, s_id, s_id, d_id),
                    )

            # Heal 10: ConceptShape/siblingNotBroader
            sibling_rows = conn.execute(
                """
                SELECT e.src_id, e.dst_id, es.name, ed.name
                FROM edges e
                INNER JOIN entities es ON es.entity_id = e.src_id
                INNER JOIN entities ed ON ed.entity_id = e.dst_id
                WHERE e.kind = 'skos:broader'
                """
            ).fetchall()
            for r in sibling_rows:
                s_id, d_id, s_name, d_name = str(r[0]), str(r[1]), str(r[2]), str(r[3])
                if is_version_or_ordinal_sibling(s_name, d_name):
                    conn.execute(
                        """
                        DELETE FROM edges
                        WHERE kind = 'skos:broader'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (s_id, d_id, d_id, s_id),
                    )
                    upsert_semantic_edges(conn, [(s_id, d_id, "skos:related", 0.5)])
                    conn.execute(
                        """
                        UPDATE OR IGNORE edge_sources
                        SET kind = 'skos:related'
                        WHERE ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                          AND kind = 'skos:broader'
                        """,
                        (s_id, d_id, d_id, s_id),
                    )

            # Heal 7: ConceptShape/broaderCycle & ConceptShape/S27-disjointBroaderRelated
            conn.execute("DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = dst_id")
            for _ in range(16):
                cycle_nodes = [
                    str(r[0])
                    for r in conn.execute(
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
                    ).fetchall()
                ]
                if not cycle_nodes:
                    break
                placeholders = ",".join("?" for _ in cycle_nodes)
                back_edge = conn.execute(
                    f"""
                    SELECT src_id, dst_id
                    FROM edges
                    WHERE kind = 'skos:broader'
                      AND src_id IN ({placeholders})
                      AND dst_id IN ({placeholders})
                    ORDER BY (src_id > dst_id) DESC, src_id DESC, dst_id ASC
                    LIMIT 1
                    """,
                    tuple(cycle_nodes) + tuple(cycle_nodes),
                ).fetchone()
                if back_edge is None:
                    break
                conn.execute(
                    "DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?",
                    (str(back_edge[0]), str(back_edge[1])),
                )

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
                """
            ).fetchall()
            for r in s27_pairs:
                nid, bid = str(r[0]), str(r[1])
                conn.execute(
                    """
                    DELETE FROM edges
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (nid, bid, bid, nid),
                )
                conn.execute(
                    """
                    DELETE FROM edge_sources
                    WHERE kind = 'skos:related'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (nid, bid, bid, nid),
                )

            rem_v = _scan_shacl_violations(conn)
            if not rem_v or (not heal_definitions and all(v.result_path == "skos:definition" for v in rem_v)):
                break

    return _scan_shacl_violations(conn)


def apply_shacl_repairs(
    conn: sqlite3.Connection | YawConnection,
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
                if focus.startswith("entity-"):
                    cur = conn.execute(
                        "UPDATE entities SET definition = ? WHERE entity_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                elif focus.startswith("fact-"):
                    cur = conn.execute(
                        "UPDATE facts SET definition = ? WHERE fact_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action == "set_pref_label" and val and focus.startswith("entity-"):
                cur = conn.execute(
                    "UPDATE entities SET name = ? WHERE entity_id = ?",
                    (val, focus),
                )
                if cur.rowcount > 0:
                    norm_val = str(text_processing(val)) or val.strip().lower()
                    upsert_entity_labels(conn, [(norm_val, focus, "skos:prefLabel", val, "__healed__")])
                    applied += 1
            elif action in ("demote_to_alt_label", "demote_label") and focus.startswith("entity-"):
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
            elif action in ("merge_altLabel", "merge_hiddenLabel") and (target or "|" in focus):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
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
            elif action == "invert_broader":
                if "->" in focus:
                    s, d = (x.strip() for x in focus.split("->", 1))
                    conn.execute("DELETE FROM edges WHERE kind = 'skos:broader' AND src_id = ? AND dst_id = ?", (s, d))
                    conn.execute(
                        "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, 'skos:broader', 0.5)",
                        (d, s),
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
                    conn.execute(
                        """
                        DELETE FROM edges
                        WHERE kind = 'skos:broader'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (s, d, d, s),
                    )
                    upsert_semantic_edges(conn, [(s, d, "skos:related", 0.5)])
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
                    """
                    SELECT
                        e.name,
                        EXISTS (SELECT 1 FROM facts f WHERE f.subj_id = e.entity_id) AS is_subj,
                        EXISTS (
                            SELECT 1 FROM facts f
                            WHERE f.obj_id = e.entity_id AND COALESCE(f.obj_kind, 'entity') = 'entity'
                        ) AS is_ent_obj,
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
                    WHERE e.entity_id = ?
                    """,
                    (focus,),
                ).fetchone()
                if ent_row is not None:
                    ename = str(ent_row[0])
                    is_subj, is_ent_obj, has_lit_fact, has_passage_edge = (
                        bool(ent_row[1]),
                        bool(ent_row[2]),
                        bool(ent_row[3]),
                        bool(ent_row[4]),
                    )
                    if has_lit_fact and not is_subj and not is_ent_obj:
                        conn.execute(
                            "DELETE FROM edges WHERE (src_id = ? OR dst_id = ?) "
                            "AND kind IN ('skos:broader', 'skos:related')",
                            (focus, focus),
                        )
                        conn.execute(
                            "DELETE FROM edge_sources WHERE (src_id = ? OR dst_id = ?) "
                            "AND kind IN ('skos:broader', 'skos:related')",
                            (focus, focus),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE facts
                            SET obj_kind = 'literal',
                                obj_id = CASE WHEN ? AND subj_id != ? THEN ? ELSE NULL END,
                                obj_literal_lex = ?,
                                obj_datatype = 'rdfs:Literal',
                                obj_lang = ''
                            WHERE obj_id = ? AND COALESCE(obj_kind, 'entity') = 'entity'
                            """,
                            (1 if has_passage_edge else 0, focus, focus, ename, focus),
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
                        conn.execute(
                            "DELETE FROM edges WHERE (src_id = ? OR dst_id = ?) "
                            "AND kind IN ('skos:broader', 'skos:related')",
                            (focus, focus),
                        )
                        conn.execute(
                            "DELETE FROM edge_sources WHERE (src_id = ? OR dst_id = ?) "
                            "AND kind IN ('skos:broader', 'skos:related')",
                            (focus, focus),
                        )
                        conn.execute(
                            """
                            DELETE FROM edges
                            WHERE kind = 'fact'
                              AND (src_id = ? OR dst_id = ?)
                              AND NOT EXISTS (
                                  SELECT 1 FROM facts f
                                  WHERE f.subj_id = edges.src_id AND f.obj_id = edges.dst_id
                              )
                            """,
                            (focus, focus),
                        )
                        conn.execute(
                            """
                            DELETE FROM edge_sources
                            WHERE kind = 'fact'
                              AND (src_id = ? OR dst_id = ?)
                              AND NOT EXISTS (
                                  SELECT 1 FROM facts f
                                  WHERE f.subj_id = edge_sources.src_id AND f.obj_id = edge_sources.dst_id
                              )
                            """,
                            (focus, focus),
                        )
                        conn.execute(
                            """
                            UPDATE edges
                            SET weight = 0.5 * COALESCE(
                                (
                                    SELECT CASE WHEN COUNT(*) > 0 THEN COUNT(*) ELSE 1 END
                                    FROM edge_sources es
                                    WHERE es.src_id = edges.src_id
                                      AND es.dst_id = edges.dst_id
                                      AND es.kind = 'fact'
                                ),
                                1
                            )
                            WHERE kind = 'fact'
                              AND dst_id = ?
                              AND NOT EXISTS (
                                  SELECT 1 FROM facts f
                                  WHERE f.subj_id = edges.src_id
                                    AND f.obj_id = edges.dst_id
                                    AND COALESCE(f.obj_kind, 'entity') = 'entity'
                              )
                            """,
                            (focus,),
                        )
                        rem_lit_fact = (
                            conn.execute(
                                "SELECT 1 FROM facts WHERE obj_id = ? AND obj_kind = 'literal' LIMIT 1",
                                (focus,),
                            ).fetchone()
                            is not None
                        )
                        if not rem_lit_fact and not has_passage_edge:
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
    conn: sqlite3.Connection | YawConnection,
    violations: Sequence[SHACLViolation] | None = None,
) -> str:
    """Serialize the yaw.db SKOS/RDF/OWL closed-world knowledge graph into W3C Turtle format."""
    if violations is None:
        violations = _scan_shacl_violations(conn)

    lines: list[str] = [
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix yaw: <https://yaw.local/vocab#> .",
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

    for row in conn.execute("SELECT entity_id, name, definition FROM entities ORDER BY entity_id ASC"):
        eid, ename, edef = str(row[0]), str(row[1]), str(row[2])
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
        for kind, dst_id in sem_edges_by_eid.get(eid, []):
            pred_objs.append(f"{kind} yaw:{dst_id}")
        for cid in sorted(prov_chunks):
            pred_objs.append(f"prov:wasDerivedFrom yaw:{cid}")

        lines.append(f"yaw:{eid} " + " ;\n    ".join(pred_objs) + " .")
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
            COALESCE(obj_lang, '')
        FROM facts
        ORDER BY fact_id ASC
        """
    ):
        fid, sid, pred, oid, fdef, okind, olex, odt, olang = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]) if row[3] is not None else "",
            str(row[4]),
            str(row[5]),
            str(row[6]) if row[6] is not None else "",
            str(row[7]),
            str(row[8]),
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
                    ("xsd:", "rdf:", "rdfs:", "owl:", "skos:", "prov:", "sh:", "yaw:")
                ) and not clean_odt.startswith(("http://", "https://", "urn:")):
                    lit_token = f"{lit_token}^^{clean_odt}"
                else:
                    lit_token = f"{lit_token}^^<{clean_odt}>"
            rdf_obj = lit_token
        else:
            rdf_obj = f"yaw:{oid}"
        stmt_parts = [
            "a rdf:Statement",
            f"rdf:subject yaw:{sid}",
            f"rdf:predicate {_escape_turtle_literal(pred)}",
            f"rdf:object {rdf_obj}",
        ]
        if fdef.strip():
            stmt_parts.append(f"skos:definition {_escape_turtle_literal(fdef.strip())}")
        lines.append(f"yaw:{fid} " + " ;\n    ".join(stmt_parts) + " .")
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
    lines.append("yaw:ValidationReport " + " ;\n    ".join(report_parts) + " .")
    lines.append("")

    return "\n".join(lines).strip() + "\n"
