"""Transaction-free CocoIndex graph planner and shared Stage 1-4 deterministic graph builders."""

import asyncio
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np

from yaw.core import (
    ChunkPointer,
    OpenIERecord,
    SHACLViolation,
    StructuredTriple,
    YawRuntimeError,
    compute_mdhash_id,
    detect_scalar_literal,
    text_processing,
)
from yaw.core.config import context_char_limit, scale_context_budget
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, EmbeddingRequest
from yaw.daemon.graph.inference.components import (
    EMBEDDING_ENDPOINT,
    OPENIE_ENDPOINT,
    MissingEntityInput,
    PredictorTrace,
    SHACLHealInput,
    SynonymJudgeInput,
    define_missing_entities,
    embed_text,
    heal_shacl_violations,
    judge_synonyms,
)
from yaw.daemon.graph.normalization import normalize_openie_record
from yaw.daemon.graph.storage import (
    PreparedGraphRows,
    apply_prepared_graph_rows,
    apply_shacl_repairs,
    canonicalize_entity_labels_edc,
    deserialize_f32_vector,
    deserialize_openie_record,
    filter_valid_concept_entity_ids,
    find_entity_disambiguation_candidates,
    get_entities_by_label_norm,
    get_entity_metadata_batch,
    inspect_focus_node,
    iter_synonym_evidence,
    lint_graph_shacl,
    merge_entities,
    open_yaw_db,
    resolve_chunk_text,
    upsert_semantic_edges,
)
from yaw.daemon.graph.storage.publication import GRAPH_DB


@dataclass(frozen=True, slots=True)
class GraphPlanSettings:
    corpus_root: Path
    program_identity: str
    embedding_identity: str
    synonym_identity: str = ""
    shacl_identity: str = ""
    auto_merge_cosine_gate: float = 0.95
    auto_merge_difflib_gate: float = 0.96
    synonymy_threshold: float = 0.8
    synonymy_top_k: int = 5
    datatype_property_weight: float = 0.5
    ast_context_hops: int = 1
    context_window_tokens: int = 32768
    max_output_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50


@dataclass(frozen=True, slots=True)
class GraphPlanResult:
    rows: PreparedGraphRows
    traces: tuple[PredictorTrace, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()
    label_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    hierarchy_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    related_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    shacl_audit_trail: tuple[dict[str, Any], ...] = ()
    rlm_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SenseResolutionState:
    senses_by_label: dict[str, list[tuple[str, str, str]]]


@dataclass(slots=True)
class MaterializedGraphBatch:
    entity_nodes: list[tuple[str, str, str]]
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]]
    fact_nodes: list[tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str]]
    fact_sources_batch: list[tuple[str, str]]
    edges_with_sources: list[tuple[str, str, str, float, str]]
    fact_linked_pairs: set[tuple[str, str]]
    homograph_splits: int


def load_active_chunk_inferences(
    conn: sqlite3.Connection,
    *,
    program_identity: str,
    embedding_identity: str,
) -> list[tuple[ChunkPointer, OpenIERecord, tuple[PredictorTrace, ...]]]:
    """Read active source_chunks joined with chunk_inference on a read-only snapshot."""
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunk_inference', 'source_chunks')"
        ).fetchall()
    }
    if tables != {"chunk_inference", "source_chunks"}:
        return []
    rows = conn.execute(
        """
        SELECT
            s.chunk_id,
            s.rel_posix_path,
            s.start_line,
            s.end_line,
            s.title,
            s.ast_island_start,
            s.ast_island_end,
            ci.extraction_json,
            ci.traces_json
        FROM source_chunks s
        JOIN chunk_inference ci
          ON ci.chunk_id = s.chunk_id
         AND ci.source_hash = s.source_hash
        WHERE ci.program_identity = ?
          AND ci.embedding_identity = ?
        ORDER BY s.chunk_id ASC
        """,
        (program_identity, embedding_identity),
    ).fetchall()
    items: list[tuple[ChunkPointer, OpenIERecord, tuple[PredictorTrace, ...]]] = []
    for r in rows:
        ptr = ChunkPointer(
            chunk_id=str(r[0]),
            rel_posix_path=str(r[1]),
            start_line=int(r[2]),
            end_line=int(r[3]),
            title=str(r[4] or ""),
            ast_island_start=int(r[5] or 0),
            ast_island_end=int(r[6] or 0),
        )
        rec = deserialize_openie_record(str(r[7]), chunk_id=str(r[0]))
        raw_traces = json.loads(str(r[8] or "[]"))
        traces = tuple(
            PredictorTrace(
                predictor=str(t.get("predictor", "")),
                inputs=dict(t.get("inputs") or {}),
                outputs=dict(t.get("outputs") or {}),
            )
            for t in raw_traces
            if isinstance(t, Mapping)
        )
        items.append((ptr, rec, traces))
    return items


def clone_graph_to_memory(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Clone an existing yaw.db graph and vector tables into an isolated in-memory SQLite database."""
    stage_conn = open_yaw_db(":memory:")
    with stage_conn:
        for table, cols in (
            ("chunks", "chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end"),
            ("entities", "entity_id, name, definition"),
            ("entity_labels", "label_norm, entity_id, label_kind, raw_label, chunk_id"),
            (
                "facts",
                "fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, "
                "obj_literal_lex, obj_datatype, obj_lang",
            ),
            ("fact_sources", "fact_id, chunk_id"),
            ("edges", "src_id, dst_id, kind, weight"),
            ("edge_sources", "src_id, dst_id, kind, chunk_id"),
            ("chunk_vec", "chunk_id, embedding"),
            ("entity_vec", "entity_id, embedding"),
            ("fact_vec", "fact_id, embedding"),
        ):
            rows = conn.execute(f"SELECT {cols} FROM {table}").fetchall()
            if rows:
                placeholders = ",".join("?" for _ in cols.split(","))
                stage_conn.executemany(
                    f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                    [tuple(r) for r in rows],
                )
    return stage_conn


def record_surface_form(
    surface_by_norm: dict[str, str],
    chunk_surface_by_norm: dict[tuple[str, str], str],
    norm_key: str,
    raw_val: str,
    cid_key: str = "",
) -> None:
    """Track preferred cased surface forms globally and per chunk."""
    clean_raw = raw_val.strip()
    if not norm_key or not clean_raw:
        return
    prev = surface_by_norm.get(norm_key)
    if prev is None or (prev == prev.lower() and clean_raw != clean_raw.lower()):
        surface_by_norm[norm_key] = clean_raw
    if cid_key:
        prev_c = chunk_surface_by_norm.get((cid_key, norm_key))
        if prev_c is None or (prev_c == prev_c.lower() and clean_raw != clean_raw.lower()):
            chunk_surface_by_norm[(cid_key, norm_key)] = clean_raw


def collect_missing_entity_norms(
    normalized_sts: Sequence[StructuredTriple],
    ner_entities: Sequence[tuple[str, str]],
    stage1_defs: Mapping[str, str],
    already_attempted: Iterable[str] = (),
) -> list[str]:
    """Identify normalized entity mentions lacking contextual definitions after Stage 1/2 OpenIE."""
    missing_norms: list[str] = []
    seen_missing: set[str] = set(already_attempted)
    for st in normalized_sts:
        if not st.subj_def and not stage1_defs.get(st.subj) and st.subj not in seen_missing:
            seen_missing.add(st.subj)
            missing_norms.append(st.subj)
        if st.obj_kind != "literal" and not st.obj_def and not stage1_defs.get(st.obj) and st.obj not in seen_missing:
            seen_missing.add(st.obj)
            missing_norms.append(st.obj)
    for e_norm, e_def in ner_entities:
        if not e_def and not stage1_defs.get(e_norm) and e_norm not in seen_missing:
            seen_missing.add(e_norm)
            missing_norms.append(e_norm)
    return missing_norms


def apply_stage1_definitions(
    cid: str,
    normalized_sts: list[StructuredTriple],
    ner_entities: list[tuple[str, str]],
    stage1_defs: dict[str, str],
    repaired_pairs: Sequence[tuple[str, str]],
    surface_by_norm: dict[str, str],
    chunk_surface_by_norm: dict[tuple[str, str], str],
) -> tuple[list[StructuredTriple], list[tuple[str, str]]]:
    """Merge repaired missing-entity definitions into Stage 1 definitions, triples, and NER entities."""
    for r_name, r_def in repaired_pairs:
        r_norm = str(text_processing(str(r_name)))
        record_surface_form(surface_by_norm, chunk_surface_by_norm, r_norm, str(r_name), cid)
        r_clean = str(r_def).strip()
        if r_norm and r_clean:
            stage1_defs[r_norm] = r_clean
    if stage1_defs:
        normalized_sts = [
            StructuredTriple(
                subj=st.subj,
                pred=st.pred,
                obj=st.obj,
                spo_def=st.spo_def,
                subj_def=st.subj_def or stage1_defs.get(st.subj, ""),
                obj_def=(
                    st.obj_def or stage1_defs.get(st.obj, "") or stage1_defs.get(str(text_processing(st.obj)), "")
                ),
                obj_kind=st.obj_kind,
                obj_datatype=st.obj_datatype,
                obj_lang=st.obj_lang,
            )
            for st in normalized_sts
        ]
        ner_entities = [(e_norm, e_def or stage1_defs.get(e_norm, "")) for e_norm, e_def in ner_entities]
    return normalized_sts, ner_entities


def prepare_sense_resolution_state(
    chunks: Sequence[tuple[ChunkPointer, str]],
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    all_norm_labels: Iterable[str],
    surface_by_norm: Mapping[str, str],
    *,
    conn: sqlite3.Connection | None = None,
) -> SenseResolutionState:
    """Prepare existing sense state for deterministic mention materialization."""
    del chunks, record_structured
    senses_by_label: dict[str, list[tuple[str, str, str]]] = {}

    if conn is not None:
        existing_by_label = get_entities_by_label_norm(conn, list(all_norm_labels))
        all_existing_eids = list({eid for eids in existing_by_label.values() for eid in eids})
        entity_info_by_id: dict[str, tuple[str, str]] = {}
        if all_existing_eids:
            entity_info_by_id = get_entity_metadata_batch(conn, all_existing_eids)
        for lbl, eids in existing_by_label.items():
            senses_by_label[lbl] = [
                (
                    eid,
                    entity_info_by_id.get(eid, (surface_by_norm.get(lbl, lbl), ""))[0],
                    entity_info_by_id.get(eid, (surface_by_norm.get(lbl, lbl), ""))[1],
                )
                for eid in eids
            ]

    return SenseResolutionState(senses_by_label=senses_by_label)


def materialize_entities_and_facts(
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    surface_by_norm: Mapping[str, str],
    chunk_surface_by_norm: Mapping[tuple[str, str], str],
    sense_state: SenseResolutionState,
    *,
    datatype_property_weight: float = 0.5,
) -> MaterializedGraphBatch:
    """Resolve entity mentions with provisional sense-splitting and materialize reified facts and edges."""
    senses_by_label = sense_state.senses_by_label

    entity_map: dict[str, tuple[str, str, str]] = {}
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]] = []
    homograph_splits = 0

    def _label_kind_for(entity_name: str, lbl_norm: str) -> str:
        return (
            "skos:prefLabel"
            if (
                lbl_norm == entity_name
                or lbl_norm.lower() == entity_name.lower()
                or lbl_norm == str(text_processing(entity_name))
            )
            else "skos:altLabel"
        )

    def _resolve_mention_entity_id(
        raw_mention: str,
        defn: str,
        chunk_id: str,
        *,
        exclude_eid: str | None = None,
        allow_sense_split: bool = True,
    ) -> str:
        nonlocal homograph_splits
        raw_clean = str(raw_mention).strip()
        label_norm = str(text_processing(raw_clean)) or raw_clean.lower()
        clean_def = defn.strip()
        canon_surface = (
            surface_by_norm.get(label_norm)
            or surface_by_norm.get(raw_clean)
            or (raw_clean if raw_clean != raw_clean.lower() else label_norm)
        )
        mention_raw = (
            chunk_surface_by_norm.get((chunk_id, label_norm))
            or chunk_surface_by_norm.get((chunk_id, raw_clean))
            or canon_surface
        )
        existing_senses = senses_by_label.get(label_norm, [])
        if not existing_senses:
            eid = compute_mdhash_id(label_norm, prefix="entity-")
            senses_by_label[label_norm] = [(eid, canon_surface, clean_def)]
            entity_map[eid] = (eid, canon_surface, clean_def)
            entity_labels_to_upsert.append((label_norm, eid, "skos:prefLabel", mention_raw, chunk_id))
            return eid

        def _best_surface(curr_name: str) -> str:
            if (
                curr_name == curr_name.lower()
                and canon_surface != canon_surface.lower()
                and (str(text_processing(curr_name)) or curr_name.lower()) == label_norm
            ):
                return canon_surface
            return curr_name

        if not clean_def or not allow_sense_split:
            for idx_s, (eid, name, prev_def) in enumerate(existing_senses):
                if exclude_eid is not None and eid == exclude_eid:
                    continue
                best_name = _best_surface(name)
                merged_def = prev_def or clean_def
                if best_name != name or merged_def != prev_def:
                    existing_senses[idx_s] = (eid, best_name, merged_def)
                    entity_map[eid] = (eid, best_name, merged_def)
                elif eid not in entity_map:
                    entity_map[eid] = (eid, best_name, merged_def)
                entity_labels_to_upsert.append(
                    (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                )
                return eid
            eid, name, prev_def = existing_senses[0]
            best_name = _best_surface(name)
            merged_def = prev_def or clean_def
            if best_name != name or merged_def != prev_def:
                existing_senses[0] = (eid, best_name, merged_def)
                entity_map[eid] = (eid, best_name, merged_def)
            elif eid not in entity_map:
                entity_map[eid] = (eid, best_name, merged_def)
            entity_labels_to_upsert.append(
                (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
            )
            return eid

        for idx_s, (eid, name, prev_def) in enumerate(existing_senses):
            if exclude_eid is not None and eid == exclude_eid:
                continue
            if not prev_def or prev_def.lower() == clean_def.lower():
                best_name = _best_surface(name)
                merged_def = clean_def if len(clean_def) > len(prev_def) else (prev_def or clean_def)
                existing_senses[idx_s] = (eid, best_name, merged_def)
                entity_map[eid] = (eid, best_name, merged_def)
                entity_labels_to_upsert.append(
                    (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                )
                return eid

        sense_id = compute_mdhash_id(f"{label_norm}::{clean_def.lower()}", prefix="entity-")
        existing_senses.append((sense_id, canon_surface, clean_def))
        entity_map[sense_id] = (sense_id, canon_surface, clean_def)
        entity_labels_to_upsert.append((label_norm, sense_id, "skos:prefLabel", mention_raw, chunk_id))
        homograph_splits += 1
        return sense_id

    chunk_to_entities: dict[str, set[str]] = {}
    fact_nodes_map: dict[
        str,
        tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str],
    ] = {}
    fact_sources_batch: list[tuple[str, str]] = []
    edges_with_sources: list[tuple[str, str, str, float, str]] = []
    fact_linked_pairs: set[tuple[str, str]] = set()

    for cid, st_list, ner_entities in record_structured:
        chunk_to_entities.setdefault(cid, set())
        resolved_in_chunk: set[tuple[str, str]] = set()
        for st in st_list:
            s_norm_key = str(text_processing(st.subj)) or st.subj.strip().lower()
            s_id = _resolve_mention_entity_id(st.subj, st.subj_def, cid)
            resolved_in_chunk.add((s_norm_key, st.subj_def.strip().lower()))
            chunk_to_entities[cid].add(s_id)

            if st.obj_kind == "literal":
                o_id: str | None = None
                o_norm_key = str(text_processing(st.obj)) or st.obj.strip().lower()
                is_self_literal = o_norm_key == s_norm_key
                eff_obj_dt = (
                    st.obj_datatype
                    if (st.obj_datatype and st.obj_datatype != "rdfs:Literal")
                    else detect_scalar_literal(st.obj)[1]
                )
                if not is_self_literal and not (eff_obj_dt.startswith("xsd:") and not st.obj_def.strip()):
                    has_existing_sense = bool(senses_by_label.get(o_norm_key))
                    eff_obj_def = (
                        st.obj_def.strip()
                        if (st.obj_def.strip() or has_existing_sense)
                        else (st.spo_def.strip() or f"Literal value {st.obj}.")
                    )
                    o_id = _resolve_mention_entity_id(st.obj, eff_obj_def, cid, allow_sense_split=False)
                    resolved_in_chunk.add((o_norm_key, st.obj_def.strip().lower()))
                    chunk_to_entities[cid].add(o_id)
                fact_oid = o_id if (o_id is not None and o_id != s_id) else None
                fid = compute_mdhash_id(
                    str((s_id, st.pred, f"literal:{st.obj}:{eff_obj_dt}:{st.obj_lang}")),
                    prefix="fact-",
                )
                raw_json = json.dumps([st.subj, st.pred, st.obj])
                prev_fact = fact_nodes_map.get(fid)
                if prev_fact is None or (not prev_fact[5] and st.spo_def):
                    fact_nodes_map[fid] = (
                        fid,
                        s_id,
                        st.pred,
                        fact_oid,
                        raw_json,
                        st.spo_def,
                        st.subj_def,
                        st.obj_def,
                        "literal",
                        st.obj,
                        eff_obj_dt,
                        st.obj_lang,
                    )
                fact_sources_batch.append((fid, cid))
                if fact_oid is not None:
                    fact_linked_pairs.add((min(s_id, fact_oid), max(s_id, fact_oid)))
                    edges_with_sources.append((s_id, fact_oid, "fact", datatype_property_weight, cid))
            else:
                o_norm_key = str(text_processing(st.obj)) or st.obj.strip().lower()
                distinct_homograph_pole = (
                    o_norm_key == s_norm_key
                    and bool(st.subj_def.strip())
                    and bool(st.obj_def.strip())
                    and st.subj_def.strip().lower() != st.obj_def.strip().lower()
                )
                o_id = _resolve_mention_entity_id(
                    st.obj,
                    st.obj_def,
                    cid,
                    exclude_eid=s_id if distinct_homograph_pole else None,
                )
                resolved_in_chunk.add((o_norm_key, st.obj_def.strip().lower()))
                chunk_to_entities[cid].add(o_id)

                base_s_id = compute_mdhash_id(s_norm_key, prefix="entity-")
                base_o_id = compute_mdhash_id(o_norm_key, prefix="entity-")
                if s_norm_key == o_norm_key or s_id != base_s_id or o_id != base_o_id:
                    fid = compute_mdhash_id(str((s_id, st.pred, o_id)), prefix="fact-")
                else:
                    fid = compute_mdhash_id(str((st.subj, st.pred, st.obj)), prefix="fact-")
                raw_json = json.dumps([st.subj, st.pred, st.obj])
                prev_fact = fact_nodes_map.get(fid)
                if prev_fact is None or (not prev_fact[5] and st.spo_def):
                    fact_nodes_map[fid] = (
                        fid,
                        s_id,
                        st.pred,
                        o_id,
                        raw_json,
                        st.spo_def,
                        st.subj_def,
                        st.obj_def,
                        "entity",
                        None,
                        "",
                        "",
                    )
                fact_sources_batch.append((fid, cid))
                if s_id != o_id:
                    fact_linked_pairs.add((min(s_id, o_id), max(s_id, o_id)))
                    edges_with_sources.append((s_id, o_id, "fact", 1.0, cid))

        for e_raw_lbl, e_def in ner_entities:
            e_norm = str(text_processing(e_raw_lbl)) or e_raw_lbl.strip().lower()
            e_key = (e_norm, e_def.strip().lower())
            if e_key in resolved_in_chunk:
                continue
            if not e_def.strip() and any(k[0] == e_norm for k in resolved_in_chunk):
                continue
            resolved_in_chunk.add(e_key)
            ner_eid = _resolve_mention_entity_id(e_raw_lbl, e_def.strip(), cid)
            chunk_to_entities[cid].add(ner_eid)

    for cid, eids in chunk_to_entities.items():
        for eid in eids:
            edges_with_sources.append((cid, eid, "passage", 1.0, cid))
            edges_with_sources.append((eid, cid, "passage", 1.0, cid))

    return MaterializedGraphBatch(
        entity_nodes=list(entity_map.values()),
        entity_labels_to_upsert=entity_labels_to_upsert,
        fact_nodes=list(fact_nodes_map.values()),
        fact_sources_batch=fact_sources_batch,
        edges_with_sources=edges_with_sources,
        fact_linked_pairs=fact_linked_pairs,
        homograph_splits=homograph_splits,
    )


def prepare_synonym_candidates(
    stage_conn: sqlite3.Connection,
    entity_nodes: Sequence[tuple[str, str, str]],
    *,
    synonymy_threshold: float,
    synonymy_top_k: int,
    auto_merge_cosine_gate: float = 0.95,
    auto_merge_difflib_gate: float = 0.96,
    fact_linked_pairs: set[tuple[str, str]] | None = None,
    rlm_stats: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]], int, list[tuple[str, str, str, str | None]]]:
    """Filter valid SKOS concepts, apply the near-1 (cosine, difflib) auto-merge gate, and rank remaining pairs."""
    valid_entity_ids = filter_valid_concept_entity_ids(stage_conn, entity_nodes)
    if not valid_entity_ids:
        return [], {}, 0, []
    if progress_callback is not None:
        progress_callback(f"indexing: disambiguation candidate search ({len(valid_entity_ids)} changed entities)")
    raw_candidates = find_entity_disambiguation_candidates(
        stage_conn,
        entity_ids=valid_entity_ids,
        dense_threshold=synonymy_threshold,
        dense_top_k=synonymy_top_k,
    )
    cand_eids = list({eid for ea, eb, _, _ in raw_candidates for eid in (ea, eb)})
    cand_meta = get_entity_metadata_batch(stage_conn, cand_eids)

    merged_into: dict[str, str] = {}
    auto_merges: list[tuple[str, str, str, str | None]] = []
    active_fact_pairs = fact_linked_pairs or set()

    def _resolve_active_eid(eid: str) -> str:
        curr = eid
        visited: set[str] = set()
        while curr in merged_into and curr not in visited:
            visited.add(curr)
            curr = merged_into[curr]
        return curr

    def _is_active_fact_pole(u: str, v: str) -> bool:
        target_pair = (min(u, v), max(u, v))
        for fa, fb in active_fact_pairs:
            ra, rb = _resolve_active_eid(fa), _resolve_active_eid(fb)
            if (min(ra, rb), max(ra, rb)) == target_pair:
                return True
        return False

    with stage_conn:
        for raw_a, raw_b, d_sim, l_sim in raw_candidates:
            if float(d_sim) < auto_merge_cosine_gate or float(l_sim) < auto_merge_difflib_gate:
                continue
            ea = _resolve_active_eid(raw_a)
            eb = _resolve_active_eid(raw_b)
            if ea == eb:
                continue
            name_a, def_a = cand_meta.get(ea, (ea, ""))
            name_b, def_b = cand_meta.get(eb, (eb, ""))
            if text_processing(name_a) != text_processing(name_b) and _is_active_fact_pole(ea, eb):
                continue
            key_a = (len(def_a), -len(name_a), ea)
            key_b = (len(def_b), -len(name_b), eb)
            dst, src = (ea, eb) if key_a >= key_b else (eb, ea)
            merge_entities(stage_conn, src, dst, label_kind="skos:altLabel")
            auto_merges.append((src, dst, "skos:altLabel", None))
            merged_into[src] = dst
            dst_name, dst_def = cand_meta.get(dst, (dst, ""))
            src_name, src_def = cand_meta.get(src, (src, ""))
            cand_meta[dst] = (dst_name or src_name, dst_def if len(dst_def) >= len(src_def) else src_def)
            if rlm_stats is not None:
                rlm_stats["merged_entities"] = int(rlm_stats.get("merged_entities", 0)) + 1

    dedup_remaining: dict[tuple[str, str], tuple[float, float]] = {}
    for raw_a, raw_b, d_sim, l_sim in raw_candidates:
        ea = _resolve_active_eid(raw_a)
        eb = _resolve_active_eid(raw_b)
        if ea == eb:
            continue
        pkey = (min(ea, eb), max(ea, eb))
        prev = dedup_remaining.get(pkey)
        if prev is None:
            dedup_remaining[pkey] = (float(d_sim), float(l_sim))
        else:
            dedup_remaining[pkey] = (max(prev[0], float(d_sim)), max(prev[1], float(l_sim)))

    candidates: list[dict[str, Any]] = []
    for (ea, eb), (d_sim, l_sim) in dedup_remaining.items():
        name_a, def_a = cand_meta.get(ea, (ea, ""))
        name_b, def_b = cand_meta.get(eb, (eb, ""))
        sim = max(float(d_sim), float(l_sim))
        candidates.append(
            {
                "pair_id": f"{ea}|{eb}",
                "entity_a": ea,
                "name_a": name_a,
                "def_a": def_a,
                "entity_b": eb,
                "name_b": name_b,
                "def_b": def_b,
                "dense_sim": float(d_sim),
                "lexical_sim": float(l_sim),
                "similarity": sim,
            }
        )
    candidates.sort(
        key=lambda c: (
            text_processing(str(c["name_a"])) == text_processing(str(c["name_b"])),
            float(c["lexical_sim"]) >= 0.70,
            0.5 * float(c["lexical_sim"]) + 0.5 * float(c["dense_sim"]),
            float(c["similarity"]),
        ),
        reverse=True,
    )
    return candidates, cand_meta, len(raw_candidates), auto_merges


def apply_synonym_adjudication(
    stage_conn: sqlite3.Connection,
    candidates: Sequence[dict[str, Any]],
    cand_meta: Mapping[str, tuple[str, str]],
    decisions: Sequence[Mapping[str, Any]],
    fact_linked_pairs: set[tuple[str, str]],
    *,
    synonymy_threshold: float,
    rlm_stats: dict[str, Any],
    corpus_root: Path | str | None = None,
) -> tuple[list[tuple[str, str, str, str | None]], list[tuple[str, str, str, float]]]:
    """Apply Stage 3 Gemma 4 synonym decisions on stage_conn with two-phase topological resolution."""
    _ = synonymy_threshold
    merged_into: dict[str, str] = {}
    hidden_label_norms: dict[str, set[str]] = defaultdict(set)

    def _resolve_active_eid(eid: str) -> str:
        curr = eid
        visited: set[str] = set()
        while curr in merged_into and curr not in visited:
            visited.add(curr)
            curr = merged_into[curr]
        return curr

    staged_merges: list[tuple[str, str, str, str | None]] = []
    raw_semantic_batch: list[tuple[str, str, str, float]] = []
    cand_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        ea, eb = str(c["entity_a"]), str(c["entity_b"])
        cand_by_pair[(min(ea, eb), max(ea, eb))] = c

    def _is_active_fact_pole(u: str, v: str) -> bool:
        target_pair = (min(u, v), max(u, v))
        for fa, fb in fact_linked_pairs:
            ra, rb = _resolve_active_eid(fa), _resolve_active_eid(fb)
            if (min(ra, rb), max(ra, rb)) == target_pair:
                return True
        return False

    with stage_conn:
        for dec in decisions:
            pid = str(dec.get("pair_id", "")).strip()
            if dec.get("source_id") and dec.get("target_id"):
                raw_src = str(dec["source_id"]).strip()
                raw_tgt = str(dec["target_id"]).strip()
            elif "|" in pid:
                ea, eb = pid.split("|", 1)
                raw_src, raw_tgt = ea.strip(), eb.strip()
            else:
                raw_src = str(dec.get("source_id", "")).strip()
                raw_tgt = str(dec.get("target_id", "")).strip()
            rel = str(dec.get("relation") or dec.get("decision") or "").strip()
            if not raw_src or not raw_tgt:
                continue
            pkey = (min(raw_src, raw_tgt), max(raw_src, raw_tgt))
            if pkey not in cand_by_pair:
                continue
            src = _resolve_active_eid(raw_src)
            tgt = _resolve_active_eid(raw_tgt)
            if src == tgt:
                continue
            cand_info = cand_by_pair[pkey]
            sim = float(cand_info.get("similarity", 0.5) or 0.5)
            name_src, _ = cand_meta.get(src, cand_meta.get(raw_src, (src, "")))
            name_tgt, _ = cand_meta.get(tgt, cand_meta.get(raw_tgt, (tgt, "")))

            if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match"):
                if text_processing(name_src) != text_processing(name_tgt) and _is_active_fact_pole(src, tgt):
                    raw_semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                    continue
                canon = str(dec.get("canonical", "") or "").strip()
                canon_id = _resolve_active_eid(canon) if canon else ""
                if canon and (
                    canon.upper() == "A"
                    or canon in (src, raw_src)
                    or canon_id == src
                    or (
                        text_processing(canon) == text_processing(name_src)
                        and text_processing(canon) != text_processing(name_tgt)
                    )
                ):
                    merge_src, merge_dst = tgt, src
                elif canon and (
                    canon.upper() == "B"
                    or canon in (tgt, raw_tgt)
                    or canon_id == tgt
                    or (
                        text_processing(canon) == text_processing(name_tgt)
                        and text_processing(canon) != text_processing(name_src)
                    )
                ):
                    merge_src, merge_dst = src, tgt
                else:
                    merge_src, merge_dst = src, tgt
                lkind = (
                    "skos:hiddenLabel"
                    if rel == "skos:hiddenLabel" or str(dec.get("label_type", "")) == "hiddenLabel"
                    else "skos:altLabel"
                )
                if lkind == "skos:hiddenLabel":
                    typo_name = name_src if merge_src == src else name_tgt
                    src_norm = str(text_processing(typo_name)) or typo_name.strip().lower()
                    if src_norm:
                        hidden_label_norms[merge_dst].add(src_norm)
                if merge_src in hidden_label_norms:
                    hidden_label_norms[merge_dst].update(hidden_label_norms.pop(merge_src))

                merged_def = str(dec.get("definition", "") or "").strip() or None
                merge_entities(
                    stage_conn,
                    merge_src,
                    merge_dst,
                    label_kind=lkind,
                    corpus_root=corpus_root,
                    merged_definition=merged_def,
                )
                staged_merges.append((merge_src, merge_dst, lkind, merged_def))
                merged_into[merge_src] = merge_dst
                rlm_stats["merged_entities"] = int(rlm_stats.get("merged_entities", 0)) + 1
            elif rel in ("owl:differentFrom", "different", "different_from"):
                raw_semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
            elif rel == "skos:broader":
                child_id, parent_id = src, tgt
                raw_semantic_batch.append((child_id, parent_id, "skos:broader", max(0.5, sim)))
            elif rel in ("skos:related", "related"):
                raw_semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))

        edge_priority = {"skos:related": 1, "skos:broader": 2, "owl:differentFrom": 3}
        pair_edges: dict[tuple[str, str], tuple[str, str, str, float]] = {}
        for s_raw, o_raw, r_kind, w in raw_semantic_batch:
            s_act = _resolve_active_eid(s_raw)
            o_act = _resolve_active_eid(o_raw)
            if s_act == o_act:
                continue
            upair = (min(s_act, o_act), max(s_act, o_act))
            prev = pair_edges.get(upair)
            if prev is None:
                pair_edges[upair] = (s_act, o_act, r_kind, w)
            else:
                _, _, prev_kind, prev_w = prev
                if edge_priority[r_kind] > edge_priority[prev_kind] or (
                    edge_priority[r_kind] == edge_priority[prev_kind] and w > prev_w
                ):
                    pair_edges[upair] = (s_act, o_act, r_kind, w)

        resolved_semantic_batch: list[tuple[str, str, str, float]] = []
        for s_act, o_act, r_kind, w in pair_edges.values():
            resolved_semantic_batch.append((s_act, o_act, r_kind, w))
            if r_kind == "owl:differentFrom":
                rlm_stats["different_from_edges"] = int(rlm_stats.get("different_from_edges", 0)) + 1
            elif r_kind == "skos:broader":
                rlm_stats["broader_edges"] = int(rlm_stats.get("broader_edges", 0)) + 1
            elif r_kind == "skos:related":
                rlm_stats["related_edges"] = int(rlm_stats.get("related_edges", 0)) + 1

        if resolved_semantic_batch:
            upsert_semantic_edges(stage_conn, resolved_semantic_batch)

        canonicalize_entity_labels_edc(
            stage_conn,
            corpus_root=corpus_root,
            hidden_label_norms_by_entity=hidden_label_norms,
        )

    return staged_merges, resolved_semantic_batch


def collect_shacl_violation_dossiers(
    stage_conn: sqlite3.Connection,
    corpus_root: Path,
    *,
    ast_context_hops: int,
    context_window_tokens: int,
    attempted_keys: set[tuple[str, str]] | None = None,
    batch_node_ids: set[str] | None = None,
) -> tuple[list[SHACLViolation], list[SHACLViolation], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Lint stage_conn for SHACL violations, filter by batch scope if incremental, and build focus dossiers."""
    all_violations = lint_graph_shacl(stage_conn)
    if attempted_keys is not None and batch_node_ids is not None:
        violations = [
            v
            for v in all_violations
            if (v.focus_node, v.shape) not in attempted_keys
            or v.focus_node in batch_node_ids
            or ("|" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("|")))
            or ("->" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("->")))
        ]
    else:
        violations = list(all_violations)
    focus_dossiers: dict[str, dict[str, Any]] = {}
    focus_contexts: list[dict[str, Any]] = []
    for v in violations[: scale_context_budget(16, context_window_tokens)]:
        if v.focus_node not in focus_dossiers:
            focus_dossiers[v.focus_node] = inspect_focus_node(
                stage_conn,
                v.focus_node,
                corpus_root=corpus_root,
                ast_context_hops=ast_context_hops,
            )
        focus_contexts.append(focus_dossiers[v.focus_node])
    return all_violations, violations, focus_dossiers, focus_contexts


@coco.fn
async def plan_graph_publication(
    settings: GraphPlanSettings,
    chunk_rows: Sequence[tuple[ChunkPointer, OpenIERecord]] | None = None,
) -> GraphPlanResult:
    """Plan a complete SHACL-conformant PreparedGraphRows without holding any yaw.db write transaction."""
    embeddings = coco.use_context(EMBEDDING_ENDPOINT)
    extraction = coco.use_context(OPENIE_ENDPOINT)
    if (
        settings.program_identity != extraction.program_identity
        or settings.embedding_identity != embeddings.adapter.profile.index_identity
    ):
        raise ValueError("Graph plan settings identity does not match the bound inference endpoints")

    all_traces: list[PredictorTrace] = []
    all_telemetry: list[dict[str, Any]] = []
    entity_def_audit: list[dict[str, Any]] = []
    label_judge_audit: list[dict[str, Any]] = []
    hierarchy_judge_audit: list[dict[str, Any]] = []
    related_judge_audit: list[dict[str, Any]] = []
    shacl_audit: list[dict[str, Any]] = []
    rlm_stats: dict[str, Any] = {
        "missing_entity_defs_count": 0,
        "missing_entity_defs_tokens": 0,
        "homograph_splits": 0,
        "candidate_pairs": 0,
        "label_judge_tokens": 0,
        "hierarchy_judge_tokens": 0,
        "related_judge_tokens": 0,
        "merged_entities": 0,
        "different_from_edges": 0,
        "broader_edges": 0,
        "related_edges": 0,
        "shacl_violations_before": 0,
        "shacl_violations_initial": 0,
        "shacl_violations_after": 0,
        "shacl_violations_remaining": 0,
        "shacl_healer_tokens": 0,
        "shacl_repairs_applied": 0,
        "shacl_conforms": True,
    }

    if chunk_rows is not None:
        active_items = [(ptr, rec, ()) for ptr, rec in sorted(chunk_rows, key=lambda item: item[0].chunk_id)]
    else:
        with coco.use_context(GRAPH_DB).readonly() as ro_conn:
            active_items = load_active_chunk_inferences(
                ro_conn,
                program_identity=settings.program_identity,
                embedding_identity=settings.embedding_identity,
            )

    if not active_items:
        empty_rows = PreparedGraphRows((), (), (), (), (), (), (), (), ())
        return GraphPlanResult(rows=empty_rows, rlm_stats=rlm_stats)

    chunks: list[tuple[ChunkPointer, str]] = []
    all_records: list[OpenIERecord] = []
    for ptr, rec, chunk_traces in active_items:
        chunk_text = resolve_chunk_text(
            ptr,
            settings.corpus_root,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        )
        chunks.append((ptr, chunk_text))
        all_records.append(rec)
        all_traces.extend(chunk_traces)

    fmt_chunk = getattr(
        embeddings.adapter,
        "format_chunk_for_embedding",
        lambda p, t: DSPyEmbeddingAdapter.format_chunk_for_embedding(embeddings.adapter, p, t),
    )
    fmt_entity = DSPyEmbeddingAdapter.format_entity_for_embedding
    fmt_fact = DSPyEmbeddingAdapter.format_fact_for_embedding

    def _prepare_embed_req(text: str, stage: str) -> EmbeddingRequest:
        if hasattr(embeddings.adapter, "prepare_request"):
            return embeddings.adapter.prepare_request([text], batch_size=64, stage=stage)
        return EmbeddingRequest(
            texts=(text,),
            profile=embeddings.adapter.profile,
            task_mode=getattr(embeddings.adapter, "task_mode", "distinct-titled"),
            batch_size=64,
            max_batch_chars=getattr(embeddings.adapter, "max_batch_chars", 3500),
            stage=stage,
        )

    async def _embed_strings(texts: Sequence[str], *, stage: str) -> list[np.ndarray]:
        if not texts:
            return []
        unique_texts = list(dict.fromkeys(texts))
        embedded_unique = await asyncio.gather(*(embed_text(_prepare_embed_req(t, stage)) for t in unique_texts))
        vec_by_text: dict[str, np.ndarray] = {}
        for t, emb in zip(unique_texts, embedded_unique, strict=True):
            vec_by_text[t] = deserialize_f32_vector(emb.vector_f32)
            all_telemetry.append(dict(emb.telemetry))
        return [vec_by_text[t] for t in texts]

    # 1. Chunk dense vectors (hits @coco.fn(memo=True) from process_chunk when already computed)
    chunk_embed_texts = [fmt_chunk(ptr, text) for ptr, text in chunks]
    chunk_vecs = await _embed_strings(chunk_embed_texts, stage="chunk_embed")
    chunk_vec_pairs: list[tuple[str, np.ndarray]] = [
        (ptr.chunk_id, vec) for (ptr, _), vec in zip(chunks, chunk_vecs, strict=True)
    ]

    # 2. Normalize OpenIE records & synthesize missing entity definitions via memoized define_missing_entities
    chunk_text_by_id = {ptr.chunk_id: text for ptr, text in chunks}
    record_structured: list[tuple[str, list[StructuredTriple], list[tuple[str, str]]]] = []
    all_norm_labels: set[str] = set()
    surface_by_norm: dict[str, str] = {}
    chunk_surface_by_norm: dict[tuple[str, str], str] = {}

    for rec in all_records:
        cid = rec.chunk_id
        normalized = normalize_openie_record(rec)
        normalized_sts = normalized.triples
        ner_entities = normalized.entities
        stage1_defs = normalized.definitions
        all_norm_labels.update(normalized.labels)
        for norm_key, surface in normalized.surfaces.items():
            record_surface_form(surface_by_norm, chunk_surface_by_norm, norm_key, surface, cid)

        missing_norms = collect_missing_entity_norms(normalized_sts, ner_entities, stage1_defs)
        repaired_pairs: Sequence[tuple[str, str]] = ()
        if missing_norms:
            defined = await define_missing_entities(
                MissingEntityInput(
                    chunk_id=cid,
                    passage=chunk_text_by_id.get(cid, ""),
                    missing_entities=tuple(missing_norms),
                    triples=tuple(
                        {
                            "sub": st.subj,
                            "pred": st.pred,
                            "obj": st.obj,
                            "def": st.spo_def,
                        }
                        for st in normalized_sts
                    ),
                    program_identity=settings.program_identity,
                )
            )
            all_traces.extend(defined.traces)
            all_telemetry.extend(defined.telemetry)
            entity_def_audit.extend(defined.audit_trail)
            for entry in defined.audit_trail:
                rlm_stats["missing_entity_defs_count"] += int(entry.get("missing_count", 0))
                rlm_stats["missing_entity_defs_tokens"] += int(entry.get("tokens", 0))
            repaired_pairs = defined.definitions

        normalized_sts, ner_entities = apply_stage1_definitions(
            cid,
            normalized_sts,
            ner_entities,
            stage1_defs,
            repaired_pairs,
            surface_by_norm,
            chunk_surface_by_norm,
        )
        record_structured.append((cid, normalized_sts, ner_entities))

    sense_state = prepare_sense_resolution_state(
        chunks,
        record_structured,
        all_norm_labels,
        surface_by_norm,
    )
    materialized = materialize_entities_and_facts(
        record_structured,
        surface_by_norm,
        chunk_surface_by_norm,
        sense_state,
        datatype_property_weight=settings.datatype_property_weight,
    )
    rlm_stats["homograph_splits"] += materialized.homograph_splits
    entity_nodes = materialized.entity_nodes
    fact_nodes = materialized.fact_nodes

    entity_vec_by_id: dict[str, np.ndarray] = {}
    if entity_nodes:
        e_texts = [fmt_entity(name, defn) for _, name, defn in entity_nodes]
        e_vecs = await _embed_strings(e_texts, stage="entity_embed")
        for (eid, _, _), vec in zip(entity_nodes, e_vecs, strict=True):
            entity_vec_by_id[eid] = vec

    fact_vec_by_id: dict[str, np.ndarray] = {}
    if fact_nodes:
        fact_texts: list[str] = []
        for f_row in fact_nodes:
            _, _s_id, pred, _o_id, raw_json, spo_d, subj_d, obj_d = f_row[:8]
            items = json.loads(raw_json)
            fact_texts.append(fmt_fact(items[0], pred, items[2], spo_d, subj_d, obj_d))
        f_vecs = await _embed_strings(fact_texts, stage="fact_embed")
        for f_row, vec in zip(fact_nodes, f_vecs, strict=True):
            fact_vec_by_id[f_row[0]] = vec

    initial_rows = PreparedGraphRows(
        chunks=tuple(chunks),
        entities=tuple(entity_nodes),
        entity_labels=tuple(materialized.entity_labels_to_upsert),
        facts=tuple(
            (fid, sid, pred, oid, raw, definition, kind, lexical, datatype, language)
            for fid, sid, pred, oid, raw, definition, _sd, _od, kind, lexical, datatype, language in fact_nodes
        ),
        fact_sources=tuple(materialized.fact_sources_batch),
        edges=tuple(materialized.edges_with_sources),
        chunk_vectors=tuple(chunk_vec_pairs),
        entity_vectors=tuple(entity_vec_by_id.items()),
        fact_vectors=tuple(fact_vec_by_id.items()),
    )

    # 3. Stage initial rows in an isolated in-memory SQLite database (zero yaw.db write lock)
    staging_conn = open_yaw_db(":memory:")
    try:
        with staging_conn:
            staging_conn.execute("BEGIN")
            apply_prepared_graph_rows(staging_conn, initial_rows)

        candidates, cand_meta, raw_cand_count, _auto_merges = prepare_synonym_candidates(
            staging_conn,
            entity_nodes,
            synonymy_threshold=settings.synonymy_threshold,
            synonymy_top_k=settings.synonymy_top_k,
            auto_merge_cosine_gate=settings.auto_merge_cosine_gate,
            auto_merge_difflib_gate=settings.auto_merge_difflib_gate,
            fact_linked_pairs=materialized.fact_linked_pairs,
            rlm_stats=rlm_stats,
        )
        rlm_stats["candidate_pairs"] += raw_cand_count
        if candidates or raw_cand_count:
            decisions: list[dict[str, str]] = []
            if candidates:
                batch_limit = max(1, scale_context_budget(8, settings.context_window_tokens))
                max_dossier_chars = min(
                    40000,
                    context_char_limit(12000, settings.context_window_tokens, settings.max_output_tokens),
                )
                evidence_batches = list(
                    iter_synonym_evidence(
                        staging_conn,
                        candidates,
                        settings.corpus_root,
                        ast_context_hops=settings.ast_context_hops,
                        batch_size=batch_limit,
                        max_chars=max_dossier_chars,
                        fits=extraction.program.synonym_batch_fits,
                    )
                )
                judged_batches = await asyncio.gather(
                    *(
                        judge_synonyms(
                            SynonymJudgeInput(
                                batch=ev,
                                program_identity=settings.program_identity,
                                synonym_identity=settings.synonym_identity,
                            )
                        )
                        for ev in evidence_batches
                    )
                )
                for judged in judged_batches:
                    decisions.extend(judged.decisions)
                    all_traces.extend(judged.traces)
                    all_telemetry.extend(judged.telemetry)
                    for entry in judged.audit_trail:
                        pred_name = entry.get("predictor") or entry.get("stage")
                        if pred_name == "hierarchy_judge":
                            hierarchy_judge_audit.append(entry)
                        elif pred_name == "related_judge":
                            related_judge_audit.append(entry)
                        else:
                            label_judge_audit.append(entry)
                    rlm_stats["label_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0))
                        for x in (judged.audit_trail or judged.telemetry)
                        if (x.get("predictor") or x.get("stage")) == "label_judge"
                    )
                    rlm_stats["hierarchy_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0))
                        for x in (judged.audit_trail or judged.telemetry)
                        if (x.get("predictor") or x.get("stage")) == "hierarchy_judge"
                    )
                    rlm_stats["related_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0))
                        for x in (judged.audit_trail or judged.telemetry)
                        if (x.get("predictor") or x.get("stage")) == "related_judge"
                    )

            apply_synonym_adjudication(
                staging_conn,
                candidates,
                cand_meta,
                decisions,
                materialized.fact_linked_pairs,
                synonymy_threshold=settings.synonymy_threshold,
                rlm_stats=rlm_stats,
                corpus_root=settings.corpus_root,
            )

        # 4. Closed-World SHACL Linting & RLM Healing on staging_conn
        _, violations, focus_dossiers, _ = collect_shacl_violation_dossiers(
            staging_conn,
            settings.corpus_root,
            ast_context_hops=settings.ast_context_hops,
            context_window_tokens=settings.context_window_tokens,
        )
        rlm_stats["shacl_violations_before"] = len(violations)
        rlm_stats["shacl_violations_initial"] = len(violations)
        if violations:
            healed = await heal_shacl_violations(
                SHACLHealInput(
                    violations=tuple(violations),
                    focus_dossiers=focus_dossiers,
                    focus_context=json.dumps(list(focus_dossiers.values()), ensure_ascii=False),
                    program_identity=settings.program_identity,
                    shacl_identity=settings.shacl_identity,
                )
            )
            all_traces.extend(healed.traces)
            all_telemetry.extend(healed.telemetry)
            shacl_audit.extend(healed.audit_trail)
            rlm_stats["shacl_healer_tokens"] += sum(
                int(x.get("tokens_used", 0)) for x in (healed.audit_trail or healed.telemetry)
            )
            if healed.repairs:
                applied = apply_shacl_repairs(staging_conn, healed.repairs)
                rlm_stats["shacl_repairs_applied"] += applied

        remaining_violations = lint_graph_shacl(staging_conn)
        rlm_stats["shacl_violations_after"] = len(remaining_violations)
        rlm_stats["shacl_violations_remaining"] = len(remaining_violations)
        rlm_stats["shacl_conforms"] = len(remaining_violations) == 0
        if remaining_violations:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"Graph build left {len(remaining_violations)} SHACL violations: {remaining_violations[:3]}",
            )

        # 5. Extract final post-merge, post-SHACL-repair PreparedGraphRows from staging_conn
        final_entities = tuple(
            (str(r[0]), str(r[1]), str(r[2]))
            for r in staging_conn.execute(
                "SELECT entity_id, name, definition FROM entities ORDER BY entity_id ASC"
            ).fetchall()
        )
        final_labels = tuple(
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]))
            for r in staging_conn.execute(
                "SELECT label_norm, entity_id, label_kind, raw_label, chunk_id "
                "FROM entity_labels ORDER BY entity_id ASC, label_norm ASC, chunk_id ASC"
            ).fetchall()
        )
        final_facts = tuple(
            (
                str(r[0]),
                str(r[1]),
                str(r[2]),
                str(r[3]) if r[3] is not None else None,
                str(r[4]),
                str(r[5]),
                str(r[6]),
                str(r[7]) if r[7] is not None else None,
                str(r[8]) if r[8] is not None else None,
                str(r[9]) if r[9] is not None else None,
            )
            for r in staging_conn.execute(
                "SELECT fact_id, subj_id, pred, obj_id, raw_triple_json, definition, "
                "obj_kind, obj_literal_lex, obj_datatype, obj_lang FROM facts ORDER BY fact_id ASC"
            ).fetchall()
        )
        final_fact_sources = tuple(
            (str(r[0]), str(r[1]))
            for r in staging_conn.execute(
                "SELECT fact_id, chunk_id FROM fact_sources ORDER BY fact_id ASC, chunk_id ASC"
            ).fetchall()
        )
        entity_fact_pairs = {
            (str(r[0]), str(r[1]))
            for r in staging_conn.execute(
                "SELECT DISTINCT subj_id, obj_id FROM facts "
                "WHERE COALESCE(obj_kind, 'entity') = 'entity' AND obj_id IS NOT NULL"
            ).fetchall()
        }
        final_edges = tuple(
            (
                str(r[0]),
                str(r[1]),
                str(r[2]),
                (
                    (1.0 if (str(r[0]), str(r[1])) in entity_fact_pairs else settings.datatype_property_weight)
                    if str(r[2]) == "fact"
                    else float(r[3])
                ),
                str(r[4] or ""),
            )
            for r in staging_conn.execute(
                """
                SELECT e.src_id, e.dst_id, e.kind, e.weight, COALESCE(es.chunk_id, '')
                FROM edges e
                LEFT JOIN edge_sources es
                  ON es.src_id = e.src_id AND es.dst_id = e.dst_id AND es.kind = e.kind
                ORDER BY e.src_id ASC, e.dst_id ASC, e.kind ASC, COALESCE(es.chunk_id, '') ASC
                """
            ).fetchall()
        )

        # Ensure any entity/fact whose text changed during merge or SHACL repair is embedded
        initial_entity_meta = {eid: (name, defn) for eid, name, defn in entity_nodes}
        reembed_entities = [
            (eid, name, defn)
            for eid, name, defn in final_entities
            if eid not in entity_vec_by_id or initial_entity_meta.get(eid) != (name, defn)
        ]
        if reembed_entities:
            re_vecs = await _embed_strings(
                [fmt_entity(name, defn) for _, name, defn in reembed_entities],
                stage="entity_embed",
            )
            for (eid, _, _), vec in zip(reembed_entities, re_vecs, strict=True):
                entity_vec_by_id[eid] = vec

        initial_fact_defs = {f_row[0]: f_row[5] for f_row in fact_nodes}
        reembed_facts = [
            f_row
            for f_row in final_facts
            if f_row[0] not in fact_vec_by_id or initial_fact_defs.get(f_row[0]) != f_row[5]
        ]
        if reembed_facts:
            rf_texts: list[str] = []
            for f_row in reembed_facts:
                items = json.loads(f_row[4])
                rf_texts.append(fmt_fact(items[0], f_row[2], items[2], f_row[5]))
            rf_vecs = await _embed_strings(rf_texts, stage="fact_embed")
            for f_row, vec in zip(reembed_facts, rf_vecs, strict=True):
                fact_vec_by_id[f_row[0]] = vec

        final_entity_vectors = tuple((eid, entity_vec_by_id[eid]) for eid, _, _ in final_entities)
        final_fact_vectors = tuple((f_row[0], fact_vec_by_id[f_row[0]]) for f_row in final_facts)

        final_rows = PreparedGraphRows(
            chunks=tuple(chunks),
            entities=final_entities,
            entity_labels=final_labels,
            facts=final_facts,
            fact_sources=final_fact_sources,
            edges=final_edges,
            chunk_vectors=tuple(chunk_vec_pairs),
            entity_vectors=final_entity_vectors,
            fact_vectors=final_fact_vectors,
        )
        return GraphPlanResult(
            rows=final_rows,
            traces=tuple(all_traces),
            telemetry=tuple(all_telemetry),
            entity_def_audit_trail=tuple(entity_def_audit),
            label_judge_audit_trail=tuple(label_judge_audit),
            hierarchy_judge_audit_trail=tuple(hierarchy_judge_audit),
            related_judge_audit_trail=tuple(related_judge_audit),
            shacl_audit_trail=tuple(shacl_audit),
            rlm_stats=rlm_stats,
        )
    finally:
        staging_conn.close()
