"""Transaction-free CocoIndex graph planner and shared Stage 1-4 deterministic graph builders."""

import asyncio
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectorkits.fingerprint import fingerprint_object

from yaw.core import (
    ChunkPointer,
    OpenIERecord,
    SHACLViolation,
    StructuredTriple,
    YawRuntimeError,
    clean_surface_label,
    detect_scalar_literal,
    is_irreflexive_statement,
    is_scalar_label,
    text_processing,
)
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, EmbeddingRequest
from yaw.daemon.graph.inference.components import (
    EMBEDDING_ENDPOINT,
    OPENIE_ENDPOINT,
    ChunkProcessOutput,
    MissingEntityInput,
    PredictorTrace,
    SemanticAdjudicationInput,
    SHACLHealInput,
    define_missing_entities,
    embed_text,
    heal_shacl_violations,
    judge_semantics,
)
from yaw.daemon.graph.normalization import normalize_openie_record, preferred_surface
from yaw.daemon.graph.storage import (
    apply_shacl_repairs,
    auto_heal_deterministic_shacl,
    canonicalize_entity_labels_edc,
    deserialize_f32_vector,
    deserialize_openie_record,
    find_entity_candidates,
    get_entity_metadata_batch,
    get_meta,
    inspect_focus_node,
    iter_synonym_evidence,
    lint_graph_shacl,
    load_affected_neighborhood,
    merge_entities,
    open_yaw_db,
    pool_confidences,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
    upsert_fact_sources,
    upsert_fact_vectors,
    upsert_facts,
    upsert_semantic_edges,
    write_tx,
)
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.evidence import (
    CandidateVector,
    EntityMention,
    EntityMerge,
    FactAssertion,
    bind_merged_evidence,
    build_evidence,
    canonicalize_fact_definitions,
    canonicalize_fact_text,
    evidence_schemas,
    load_sense_bindings,
    refresh_source_confidences,
    restore_asserted_fact_sources,
    sense_id,
    upsert_evidence,
)
from yaw.daemon.graph.storage.evidence import (
    fact_id as canonical_fact_id,
)
from yaw.daemon.graph.storage.managed_sqlite import create_table
from yaw.daemon.graph.storage.rows import (
    DesiredTable,
    empty_delta_graph_tables,
    empty_graph_tables,
    graph_tables,
    snapshot_table_rows,
)
from yaw.daemon.graph.storage.synonyms import SynonymConstraints


@dataclass(frozen=True, slots=True)
class GraphPlanSettings:
    corpus_root: Path
    program_identity: str
    embedding_identity: str
    synonym_identity: str = ""
    shacl_identity: str = ""
    candidate_dense_threshold: float = 0.8
    candidate_lexical_threshold: float = 0.625
    auto_merge_threshold: float = 0.98
    synonym_batch_size: int = 4
    ast_context_hops: int = 1
    context_window_tokens: int = 32768
    max_output_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50


type FactRow = tuple[str, str, str, str | None, str, str, str, str | None, str | None, str | None]
type VectorRow = tuple[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class PreparedGraphRows:
    chunks: tuple[tuple[ChunkPointer, str], ...]
    entities: tuple[tuple[str, str, str], ...] | tuple[tuple[str, str, str, float | None], ...]
    entity_labels: tuple[tuple[str, str, str, str, str], ...]
    facts: tuple[FactRow, ...]
    fact_sources: tuple[tuple[str, str], ...] | tuple[tuple[str, str, float | None, float | None, float | None], ...]
    edges: tuple[tuple[str, str, str, float, str], ...]
    chunk_vectors: tuple[VectorRow, ...]
    entity_vectors: tuple[VectorRow, ...]
    fact_vectors: tuple[VectorRow, ...]
    mentions: tuple[EntityMention, ...]
    assertions: tuple[FactAssertion, ...]
    candidate_vectors: tuple[CandidateVector, ...] = ()
    input_fingerprints: tuple[tuple[str, str], ...] = ()


def stage_graph_rows(conn: sqlite3.Connection, rows: PreparedGraphRows) -> None:
    """Apply prepared rows and FTS updates; never perform inference or commit."""
    if not conn.in_transaction:
        raise RuntimeError("Graph publication requires an active transaction")
    if conn.execute("PRAGMA database_list").fetchone()[2] != "":
        raise ValueError("Staging must never write a persistent database")
    upsert_chunk_vectors(conn, rows.chunk_vectors)
    upsert_chunk_pointers(conn, rows.chunks)
    upsert_entities(conn, rows.entities)
    upsert_entity_labels(conn, rows.entity_labels)
    upsert_facts(conn, rows.facts)
    upsert_fact_sources(conn, rows.fact_sources)
    upsert_entity_vectors(conn, rows.entity_vectors)
    upsert_fact_vectors(conn, rows.fact_vectors)
    upsert_edges_with_sources(conn, rows.edges)
    upsert_evidence(conn, rows.mentions, rows.assertions)
    conn.executemany(
        "INSERT OR REPLACE INTO entity_candidate_vectors VALUES (?,?,?,?,?)",
        ((r.sense_id, r.entity_id, r.name, r.definition, r.embedding) for r in rows.candidate_vectors),
    )


@dataclass(frozen=True, slots=True)
class GraphPlanResult:
    tables: tuple[DesiredTable, ...]
    chunk_ids: tuple[str, ...]
    traces: tuple[PredictorTrace, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()
    label_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    hierarchy_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    related_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    shacl_audit_trail: tuple[dict[str, Any], ...] = ()
    rlm_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphPlanSummary:
    """Publication diagnostics without retaining a second copy of graph vectors."""

    chunk_ids: tuple[str, ...]
    traces: tuple[PredictorTrace, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()
    label_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    hierarchy_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    related_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    shacl_audit_trail: tuple[dict[str, Any], ...] = ()
    rlm_stats: dict[str, Any] = field(default_factory=dict)


def summarize_graph_plan(plan: GraphPlanResult) -> GraphPlanSummary:
    return GraphPlanSummary(
        chunk_ids=plan.chunk_ids,
        traces=plan.traces,
        telemetry=plan.telemetry,
        entity_def_audit_trail=plan.entity_def_audit_trail,
        label_judge_audit_trail=plan.label_judge_audit_trail,
        hierarchy_judge_audit_trail=plan.hierarchy_judge_audit_trail,
        related_judge_audit_trail=plan.related_judge_audit_trail,
        shacl_audit_trail=plan.shacl_audit_trail,
        rlm_stats=dict(plan.rlm_stats),
    )


@dataclass(slots=True)
class MaterializedGraphBatch:
    entity_nodes: list[tuple[str, str, str]]
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]]
    fact_nodes: list[tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str]]
    fact_sources_batch: list[tuple[str, str]]
    edges_with_sources: list[tuple[str, str, str, float, str]]
    fact_linked_pairs: set[tuple[str, str]]
    homograph_splits: int
    bindings: dict[tuple[str, str, str], tuple[str, str]]


def record_surface_form(
    surface_by_norm: dict[str, str],
    chunk_surface_by_norm: dict[tuple[str, str], str],
    norm_key: str,
    raw_val: str,
    cid_key: str = "",
) -> None:
    """Track preferred cased surface forms globally and per chunk."""
    clean_raw = clean_surface_label(raw_val) or raw_val.strip()
    if not norm_key or not clean_raw:
        return
    surface_by_norm[norm_key] = preferred_surface(
        surface_by_norm.get(norm_key, clean_raw), clean_raw, label_norm=norm_key
    )
    if cid_key:
        key = (cid_key, norm_key)
        chunk_surface_by_norm[key] = preferred_surface(
            chunk_surface_by_norm.get(key, clean_raw), clean_raw, label_norm=norm_key
        )


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
                llm_prob=st.llm_prob,
            )
            for st in normalized_sts
        ]
        ner_entities = [(e_norm, e_def or stage1_defs.get(e_norm, "")) for e_norm, e_def in ner_entities]
    return normalized_sts, ner_entities


def materialize_entities_and_facts(
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    surface_by_norm: Mapping[str, str],
    chunk_surface_by_norm: Mapping[tuple[str, str], str],
    existing_bindings: Mapping[str, tuple[str, str, str]],
) -> MaterializedGraphBatch:
    """Resolve entity mentions with provisional sense-splitting and materialize reified facts and edges."""
    bindings: dict[tuple[str, str, str], tuple[str, str]] = {}
    definitions_by_label: dict[str, set[str]] = defaultdict(set)

    entity_map: dict[str, tuple[str, str, str]] = {}
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]] = []

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

    def _resolve_mention_entity_id(raw_mention: str, defn: str, chunk_id: str) -> str:
        label_norm = str(text_processing(raw_mention))
        clean_def = defn.strip()
        if not label_norm or not clean_def:
            raise ValueError("Entity materialization requires a NER label and resolved definition")
        candidate = sense_id(label_norm, clean_def)
        surface = surface_by_norm[label_norm]
        eid, name, definition = existing_bindings.get(candidate, (candidate, surface, clean_def))
        previous = entity_map.get(eid, (eid, name, definition))
        name = preferred_surface(name, previous[1], surface, label_norm=str(text_processing(name)))
        definition = max((definition, previous[2], clean_def), key=lambda value: (len(value), value))
        entity_map[eid] = (eid, name, definition)
        mention_raw = chunk_surface_by_norm[(chunk_id, label_norm)]
        entity_labels_to_upsert.append((label_norm, eid, _label_kind_for(name, label_norm), mention_raw, chunk_id))
        bindings[(chunk_id, label_norm, clean_def.casefold())] = (eid, clean_def)
        definitions_by_label[label_norm].add(clean_def.casefold())
        return eid

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

            is_literal_obj = st.obj_kind == "literal" or is_scalar_label(st.obj) or detect_scalar_literal(st.obj)[0]
            if is_literal_obj:
                if not is_irreflexive_statement(st.subj, st.pred, st.obj, st.subj_def, "", "literal"):
                    continue
                eff_obj_dt = (
                    st.obj_datatype
                    if (st.obj_datatype and st.obj_datatype != "rdfs:Literal")
                    else detect_scalar_literal(st.obj)[1]
                )
                fid = canonical_fact_id(s_id, st.pred, "literal", st.obj, eff_obj_dt, st.obj_lang)
                raw_json = json.dumps([entity_map[s_id][1], st.pred, st.obj])
                prev_fact = fact_nodes_map.get(fid)
                if prev_fact is None or (not prev_fact[5] and st.spo_def):
                    fact_nodes_map[fid] = (
                        fid,
                        s_id,
                        st.pred,
                        None,  # obj_id is strictly None for scalar literals
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
                # Scalar literals NEVER create entities, entity_labels, or edges
            else:
                o_norm_key = str(text_processing(st.obj)) or st.obj.strip().lower()
                o_id = _resolve_mention_entity_id(st.obj, st.obj_def, cid)
                resolved_in_chunk.add((o_norm_key, st.obj_def.strip().lower()))
                chunk_to_entities[cid].add(o_id)

                fid = canonical_fact_id(s_id, st.pred, "entity", o_id)
                raw_json = json.dumps([entity_map[s_id][1], st.pred, entity_map[o_id][1]])
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
                    edges_with_sources.append((s_id, o_id, "fact", 0.8, cid))

        for e_raw_lbl, e_def in ner_entities:
            if is_scalar_label(e_raw_lbl):
                continue
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
        homograph_splits=sum(max(0, len(values) - 1) for values in definitions_by_label.values()),
        bindings=bindings,
    )


def prepare_synonym_candidates(
    stage_conn: sqlite3.Connection,
    entity_nodes: Sequence[tuple[str, str, str]],
    *,
    candidate_dense_threshold: float,
    candidate_lexical_threshold: float,
    auto_merge_threshold: float = 0.98,
    fact_linked_pairs: set[tuple[str, str]] | None = None,
    rlm_stats: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]], int, list[EntityMerge]]:
    """Prepare sense-preserving candidates for the complete three-stage cascade."""
    valid_entity_ids = [eid for eid, name, _ in entity_nodes if bool(text_processing(name))]
    if not valid_entity_ids:
        return [], {}, 0, []
    if progress_callback is not None:
        progress_callback(f"indexing: disambiguation candidate search ({len(valid_entity_ids)} changed entities)")
    raw_candidates = find_entity_candidates(
        stage_conn,
        entity_ids=valid_entity_ids,
        dense_threshold=candidate_dense_threshold,
        lexical_threshold=candidate_lexical_threshold,
    )
    cand_eids = list({eid for ea, eb, _, _ in raw_candidates for eid in (ea, eb)})
    cand_meta = get_entity_metadata_batch(stage_conn, cand_eids)

    merges: list[EntityMerge] = []
    representatives: dict[str, str] = {}

    def resolve(eid: str) -> str:
        while eid in representatives:
            eid = representatives[eid]
        return eid

    separations = {
        tuple(sorted((str(a), str(b))))
        for a, b in stage_conn.execute(
            "SELECT src_id,dst_id FROM edges WHERE kind IN ('owl:differentFrom','skos:broader','skos:related','fact')"
        )
    }
    blocked_pairs = separations | set(fact_linked_pairs or ())
    # Only extreme candidate cosines pay for difflib; label AND definition must pass.
    for a, b, dense_sim, _ in raw_candidates:
        a, b = sorted((resolve(a), resolve(b)))
        if a == b or dense_sim < auto_merge_threshold:
            continue
        na, da = cand_meta[a]
        nb, db = cand_meta[b]
        if not da.strip() or not db.strip():
            continue
        if any({resolve(x), resolve(y)} == {a, b} for x, y in blocked_pairs):
            continue
        if any(
            SequenceMatcher(None, str(text_processing(x)), str(text_processing(y)), autojunk=False).ratio()
            < auto_merge_threshold
            for x, y in ((na, nb), (da, db))
        ):
            continue
        preferred = max(
            (a, b), key=lambda eid: (any(c.isupper() for c in cand_meta[eid][0]), len(cand_meta[eid][1]), eid)
        )
        merged = merge_entities(stage_conn, b if preferred == a else a, preferred, label_kind="skos:altLabel")
        representatives[merged.removed_id] = merged.canonical_id
        merges.append(merged)
        cand_meta.update(get_entity_metadata_batch(stage_conn, [merged.canonical_id]))
    if rlm_stats is not None:
        rlm_stats["merged_entities"] = int(rlm_stats.get("merged_entities", 0)) + len(merges)
    raw_count = len(raw_candidates)
    if merges:
        # Complete scores from all retained members, including raw pairs below admission.
        raw_candidates = find_entity_candidates(
            stage_conn,
            entity_ids=sorted({resolve(eid) for eid in valid_entity_ids}),
            dense_threshold=candidate_dense_threshold,
            lexical_threshold=candidate_lexical_threshold,
        )
    candidates: list[dict[str, Any]] = []
    for ea, eb, d_sim, l_sim in raw_candidates:
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
    return candidates, cand_meta, raw_count, merges


def apply_synonym_adjudication(
    stage_conn: sqlite3.Connection,
    candidates: Sequence[dict[str, Any]],
    cand_meta: Mapping[str, tuple[str, str]],
    decisions: Sequence[Mapping[str, Any]],
    fact_linked_pairs: set[tuple[str, str]],
    *,
    rlm_stats: dict[str, Any],
    corpus_root: Path | str | None = None,
    touched_entity_ids: Sequence[str] | None = None,
) -> tuple[list[EntityMerge], list[tuple[str, str, str, float, str]]]:
    """Apply Stage 3 Gemma 4 synonym decisions on stage_conn with two-phase topological resolution."""
    cand_meta = dict(cand_meta)
    constraints = SynonymConstraints(stage_conn, tuple(fact_linked_pairs))
    hidden_label_norms: dict[str, set[str]] = defaultdict(set)

    staged_merges: list[EntityMerge] = []
    raw_semantic_batch: list[tuple[str, str, str, float, str]] = []
    cand_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        ea, eb = str(c["entity_a"]), str(c["entity_b"])
        cand_by_pair[(min(ea, eb), max(ea, eb))] = c

    def _is_active_fact_pole(u: str, v: str) -> bool:
        target_pair = (min(u, v), max(u, v))
        for fa, fb in constraints.fact_pairs:
            ra, rb = constraints.resolve(fa), constraints.resolve(fb)
            if (min(ra, rb), max(ra, rb)) == target_pair:
                return True
        return False

    with write_tx(stage_conn):
        candidate_order = {pair: index for index, pair in enumerate(cand_by_pair)}
        for dec in sorted(
            decisions,
            key=lambda d: candidate_order.get(tuple(sorted(str(d.get("pair_id", "")).split("|"))), len(candidates)),
        ):
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
            src = constraints.resolve(raw_src)
            tgt = constraints.resolve(raw_tgt)
            if src == tgt:
                continue
            cand_info = cand_by_pair[pkey]
            sim = float(cand_info["similarity"])
            name_src, _ = cand_meta.get(src, cand_meta.get(raw_src, (src, "")))
            name_tgt, _ = cand_meta.get(tgt, cand_meta.get(raw_tgt, (tgt, "")))

            reas = str(dec.get("reasoning", "") or "").strip()
            dec_prob = dec.get("llm_prob")
            dec_llm_prob = float(dec_prob) if dec_prob is not None else None
            edge_weight = pool_confidences(dec_llm_prob, sim)
            assert edge_weight is not None
            if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match"):
                if _is_active_fact_pole(src, tgt):
                    raw_semantic_batch.append((src, tgt, "skos:related", edge_weight, reas))
                    continue
                if not constraints.merge(src, tgt):
                    continue
                merged_def = str(dec.get("definition", "") or "").strip() or None
                if (
                    merged_def
                    and name_src.strip().casefold() in merged_def.casefold()
                    and name_tgt.strip().casefold() not in merged_def.casefold()
                ):
                    merge_src, merge_dst = tgt, src
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
                merged = merge_entities(
                    stage_conn,
                    merge_src,
                    merge_dst,
                    label_kind=lkind,
                    corpus_root=corpus_root,
                    merged_definition=merged_def,
                )
                merge_src, merge_dst = merged.removed_id, merged.canonical_id
                # Label preference and the stable resource representative are independent.
                for owner in (src, tgt):
                    if owner != merge_dst and owner in hidden_label_norms:
                        hidden_label_norms[merge_dst].update(hidden_label_norms.pop(owner))
                cand_meta.update(get_entity_metadata_batch(stage_conn, [merge_dst]))
                staged_merges.append(merged)
                rlm_stats["merged_entities"] = int(rlm_stats.get("merged_entities", 0)) + 1
            else:
                if rel in ("owl:differentFrom", "different", "different_from"):
                    constraints.separate(src, tgt)
                    raw_semantic_batch.append((src, tgt, "owl:differentFrom", edge_weight, reas))
                elif rel == "skos:broader":
                    constraints.separate(src, tgt)
                    child_id, parent_id = src, tgt
                    raw_semantic_batch.append((child_id, parent_id, "skos:broader", edge_weight, reas))
                elif rel in ("skos:related", "related"):
                    constraints.separate(src, tgt)
                    raw_semantic_batch.append((src, tgt, "skos:related", edge_weight, reas))

        edge_priority = {"skos:related": 1, "skos:broader": 2, "owl:differentFrom": 3}
        pair_edges: dict[tuple[str, str], tuple[str, str, str, float, str]] = {}
        for s_raw, o_raw, r_kind, w, r_reas in raw_semantic_batch:
            s_act = constraints.resolve(s_raw)
            o_act = constraints.resolve(o_raw)
            if s_act == o_act:
                continue
            upair = (min(s_act, o_act), max(s_act, o_act))
            prev = pair_edges.get(upair)
            if prev is None:
                pair_edges[upair] = (s_act, o_act, r_kind, w, r_reas)
            else:
                _, _, prev_kind, prev_w, prev_reas = prev
                if edge_priority[r_kind] > edge_priority[prev_kind] or (
                    edge_priority[r_kind] == edge_priority[prev_kind] and w > prev_w
                ):
                    pair_edges[upair] = (s_act, o_act, r_kind, w, r_reas or prev_reas)

        resolved_semantic_batch: list[tuple[str, str, str, float, str]] = []
        for s_act, o_act, r_kind, w, r_reas in pair_edges.values():
            resolved_semantic_batch.append((s_act, o_act, r_kind, w, r_reas))
            if r_kind == "owl:differentFrom":
                rlm_stats["different_from_edges"] = int(rlm_stats.get("different_from_edges", 0)) + 1
            elif r_kind == "skos:broader":
                rlm_stats["broader_edges"] = int(rlm_stats.get("broader_edges", 0)) + 1
            elif r_kind == "skos:related":
                rlm_stats["related_edges"] = int(rlm_stats.get("related_edges", 0)) + 1

        if resolved_semantic_batch:
            upsert_semantic_edges(stage_conn, resolved_semantic_batch)

        if touched_entity_ids is not None:
            active_touched: list[str] | None = sorted(
                {constraints.resolve(eid) for eid in touched_entity_ids}
                | {constraints.resolve(m.canonical_id) for m in staged_merges}
                | {constraints.resolve(eid) for eid in hidden_label_norms}
            )
        else:
            active_touched = (
                sorted(
                    {constraints.resolve(eid) for c in candidates for eid in (str(c["entity_a"]), str(c["entity_b"]))}
                    | {constraints.resolve(m.canonical_id) for m in staged_merges}
                    | {constraints.resolve(eid) for eid in hidden_label_norms}
                )
                or None
            )
        canonicalize_entity_labels_edc(
            stage_conn,
            entity_ids=active_touched,
            corpus_root=corpus_root,
            hidden_label_norms_by_entity=hidden_label_norms,
        )

    return staged_merges, resolved_semantic_batch


def collect_shacl_violation_dossiers(
    stage_conn: sqlite3.Connection,
    corpus_root: Path,
    *,
    ast_context_hops: int,
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
    for v in violations:
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
    chunk_outputs: Sequence[ChunkProcessOutput] = (),
    *,
    db_identity: str = "",
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

    ordered_outputs = tuple(sorted(chunk_outputs, key=lambda item: item.chunk_id))
    with coco.use_context(GRAPH_DB).readonly() as ro_conn:
        resolved_db_identity = db_identity or get_meta(ro_conn, "db_identity") or ""
        has_chunks_table = (
            ro_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone() is not None
        )
        saved = get_meta(ro_conn, "graph_input_fingerprints") if has_chunks_table else None
        published_inputs = dict(json.loads(saved)) if saved is not None else {}

    if not ordered_outputs:
        prior_rows: dict[str, dict[tuple[Any, ...], Any]] | None = None
        if published_inputs and has_chunks_table:
            with coco.use_context(GRAPH_DB).readonly() as ro_conn:
                prior_rows = snapshot_table_rows(ro_conn)
        return GraphPlanResult(
            tables=empty_graph_tables(seeded_rows_by_table=prior_rows, db_identity=resolved_db_identity),
            chunk_ids=(),
            rlm_stats=rlm_stats,
        )

    # Reuse only verified unchanged contributions. Edits, removals and semantic
    # profile changes require recomputing their graph interpretation.
    input_fingerprints = {
        out.chunk_id: fingerprint_object((settings, out.pointer, out.text)).hex() for out in ordered_outputs
    }
    incremental = (
        bool(published_inputs)
        and has_chunks_table
        and all(input_fingerprints.get(cid) == digest for cid, digest in published_inputs.items())
    )
    complete_chunks = tuple((out.pointer, out.text) for out in ordered_outputs)
    outputs_to_process = (
        tuple(out for out in ordered_outputs if out.chunk_id not in published_inputs)
        if incremental
        else ordered_outputs
    )
    rlm_stats["reused_chunks"] = len(published_inputs) if incremental else 0
    rlm_stats["processed_chunks"] = len(outputs_to_process)

    if incremental and not outputs_to_process:
        return GraphPlanResult(
            tables=empty_delta_graph_tables(
                tuple(sorted(input_fingerprints.items())),
                db_identity=resolved_db_identity,
            ),
            chunk_ids=tuple(out.chunk_id for out in ordered_outputs),
            rlm_stats=rlm_stats,
        )

    chunks: list[tuple[ChunkPointer, str]] = []
    all_records: list[OpenIERecord] = []
    chunk_vec_pairs: list[tuple[str, np.ndarray]] = []
    for out in outputs_to_process:
        chunks.append((out.pointer, out.text))
        all_records.append(deserialize_openie_record(out.extraction_json, chunk_id=out.chunk_id))
        raw_traces = json.loads(out.traces_json or "[]")
        for t in raw_traces:
            if isinstance(t, Mapping):
                all_traces.append(
                    PredictorTrace(
                        predictor=str(t.get("predictor", "")),
                        inputs=dict(t.get("inputs") or {}),
                        outputs=dict(t.get("outputs") or {}),
                    )
                )
        chunk_vec_pairs.append((out.chunk_id, deserialize_f32_vector(out.vector_f32)))

    fmt_entity = DSPyEmbeddingAdapter.format_entity_for_embedding
    fmt_fact = DSPyEmbeddingAdapter.format_fact_for_embedding

    async def _embed_strings(texts: Sequence[str], *, stage: str) -> list[np.ndarray]:
        if not texts:
            return []
        unique_texts = list(dict.fromkeys(texts))
        requests = [
            embeddings.adapter.prepare_request([t], batch_size=64, stage=stage)
            if hasattr(embeddings.adapter, "prepare_request")
            else EmbeddingRequest(
                texts=(t,),
                profile=embeddings.adapter.profile,
                task_mode=getattr(embeddings.adapter, "task_mode", "distinct-titled"),
                batch_size=64,
                max_batch_chars=getattr(embeddings.adapter, "max_batch_chars", 3500),
                stage=stage,
            )
            for t in unique_texts
        ]
        embedded_unique = await asyncio.gather(*(embed_text(req) for req in requests))
        vec_by_text: dict[str, np.ndarray] = {}
        for t, emb in zip(unique_texts, embedded_unique, strict=True):
            vec_by_text[t] = deserialize_f32_vector(emb.vector_f32)
            all_telemetry.append(dict(emb.telemetry))
        return [vec_by_text[t] for t in texts]

    # 1. Normalize OpenIE records & synthesize missing entity definitions via memoized define_missing_entities
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

    with coco.use_context(GRAPH_DB).readonly() as ro_conn:
        if (
            incremental
            and ro_conn.execute("SELECT 1 FROM sqlite_master WHERE name='entity_candidate_vectors'").fetchone() is None
        ):
            raise ValueError("Candidate source-vector schema changed; rebuild this index from source")
        existing_bindings = load_sense_bindings(ro_conn, all_norm_labels) if incremental else {}
    materialized = materialize_entities_and_facts(
        record_structured,
        surface_by_norm,
        chunk_surface_by_norm,
        existing_bindings,
    )
    mentions: list[EntityMention] = []
    assertions: list[FactAssertion] = []
    for (pointer, text), record in zip(chunks, all_records, strict=True):
        raw_mentions, raw_assertions = build_evidence(
            pointer, text, settings.program_identity, record, materialized.bindings
        )
        mentions.extend(raw_mentions)
        assertions.extend(raw_assertions)
    rlm_stats["homograph_splits"] += materialized.homograph_splits
    entity_nodes = materialized.entity_nodes
    fact_nodes = materialized.fact_nodes

    entity_vec_by_id: dict[str, np.ndarray] = {}
    if entity_nodes:
        e_texts = [fmt_entity(str(text_processing(name)), defn.strip().casefold()) for _, name, defn in entity_nodes]
        e_vecs = await _embed_strings(e_texts, stage="entity_embed")
        for (eid, _, _), vec in zip(entity_nodes, e_vecs, strict=True):
            entity_vec_by_id[eid] = vec

    source_senses = {
        mention.sense_id: (
            mention.entity_id,
            str(text_processing(mention.name)),
            mention.resolved_definition.strip().casefold(),
        )
        for mention in mentions
    }
    source_embeddings = await _embed_strings(
        [fmt_entity(name, definition) for _, name, definition in source_senses.values()],
        stage="entity_embed",
    )
    candidate_vectors = tuple(
        CandidateVector(sid, eid, name, definition, vector.astype("<f4", copy=False).tobytes())
        for (sid, (eid, name, definition)), vector in zip(source_senses.items(), source_embeddings, strict=True)
    )

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
        fact_vectors=(),
        candidate_vectors=candidate_vectors,
        mentions=tuple(mentions),
        assertions=tuple(assertions),
    )

    schemas = await evidence_schemas()

    # 2. Seed bounded affected neighborhood in an isolated in-memory SQLite database (zero full-graph clone)
    staging_conn = open_yaw_db(":memory:")
    try:
        with write_tx(staging_conn):
            for schema in schemas:
                create_table(staging_conn, schema)
        seeded_rows_by_table: dict[str, dict[tuple[Any, ...], Any]] | None = None
        if incremental:
            seed_entity_ids = tuple(
                {
                    *(eid for eid, _, _ in existing_bindings.values()),
                    *(eid for eid, _, _ in entity_nodes),
                }
            )
            delta_entity_names = {eid: name for eid, name, _ in entity_nodes}
            with coco.use_context(GRAPH_DB).readonly() as ro_conn:
                load_affected_neighborhood(
                    ro_conn,
                    staging_conn,
                    seed_entity_ids=seed_entity_ids,
                    seed_label_norms=tuple(all_norm_labels),
                    delta_entity_names=delta_entity_names,
                    candidate_vectors=candidate_vectors,
                    candidate_dense_threshold=settings.candidate_dense_threshold,
                    candidate_lexical_threshold=settings.candidate_lexical_threshold,
                )
            seeded_rows_by_table = snapshot_table_rows(staging_conn)
        elif published_inputs and has_chunks_table:
            with coco.use_context(GRAPH_DB).readonly() as ro_conn:
                seeded_rows_by_table = snapshot_table_rows(ro_conn)
        prior_fact_definitions = {
            str(r[0]): str(r[1]) for r in staging_conn.execute("SELECT fact_id, definition FROM facts")
        }
        published_entity_vec_ids = {str(r[0]) for r in staging_conn.execute("SELECT entity_id FROM entity_vec")}
        initial_entity_meta = {
            str(r[0]): (str(r[1]), str(r[2]))
            for r in staging_conn.execute("SELECT entity_id, name, definition FROM entities")
        }
        initial_entity_meta.update({eid: (name, definition) for eid, name, definition in entity_nodes})
        published_fact_vec_ids = {str(r[0]) for r in staging_conn.execute("SELECT fact_id FROM fact_vec")}
        initial_fact_defs = {
            str(r[0]): (str(r[1]), str(r[2]))
            for r in staging_conn.execute("SELECT fact_id, raw_triple_json, definition FROM facts")
        }
        with staging_conn:
            staging_conn.execute("BEGIN")
            stage_graph_rows(staging_conn, initial_rows)

        candidates, cand_meta, raw_cand_count, merges = prepare_synonym_candidates(
            staging_conn,
            entity_nodes,
            candidate_dense_threshold=settings.candidate_dense_threshold,
            candidate_lexical_threshold=settings.candidate_lexical_threshold,
            auto_merge_threshold=settings.auto_merge_threshold,
            fact_linked_pairs=materialized.fact_linked_pairs,
            rlm_stats=rlm_stats,
            progress_callback=extraction.progress_callback,
        )
        rlm_stats["candidate_pairs"] += raw_cand_count
        touched_eids = [*(eid for eid, _, _ in entity_nodes), *(m.canonical_id for m in merges)]
        if candidates or raw_cand_count:
            decisions: list[dict[str, Any]] = []
            if candidates:
                constraints = SynonymConstraints(staging_conn, tuple(materialized.fact_linked_pairs))
                candidate_order = {str(c["pair_id"]): index for index, c in enumerate(candidates)}
                # Cognitive Load Safeguard: clamp batch_limit to max 4 pairs. Smaller models (4B-8B)
                # suffer attention dispersion and conservative rejection when evaluating too many pairs at once.
                # Actual prompt context fit is strictly governed by fits=extraction.program.synonym_batch_fits
                # via DSPy adapter token counting. max_dossier_chars serves purely as a generous in-memory backstop
                # (context_window_tokens * 3) to prevent unbounded memory growth before tokenization, ensuring
                # individual pairs with rich context/excerpts are never prematurely split or rejected by
                # character heuristics.
                batch_limit = max(1, min(settings.synonym_batch_size, 4))
                max_dossier_chars = max(40000, settings.context_window_tokens * 3)
                evidence_batches = iter_synonym_evidence(
                    staging_conn,
                    candidates,
                    settings.corpus_root,
                    ast_context_hops=settings.ast_context_hops,
                    batch_size=batch_limit,
                    max_chars=max_dossier_chars,
                    fits=extraction.program.synonym_batch_fits,
                )
                for batch_idx, ev in enumerate(evidence_batches, start=1):
                    original_count = len(ev.candidates)
                    ev = constraints.prune(ev)
                    rlm_stats["constraint_resolved_pairs"] = int(rlm_stats.get("constraint_resolved_pairs", 0)) + (
                        original_count - len(ev.candidates)
                    )
                    if not ev.candidates:
                        continue
                    if extraction.progress_callback is not None:
                        extraction.progress_callback(
                            f"indexing: synonym adjudication (batch {batch_idx}, {len(ev.candidates)} candidate pairs)"
                        )
                    judged = await judge_semantics(
                        SemanticAdjudicationInput(
                            batch=ev,
                            program_identity=settings.program_identity,
                            synonym_identity=settings.synonym_identity,
                        )
                    )
                    decisions.extend(judged.decisions)
                    for decision in sorted(judged.decisions, key=lambda d: candidate_order[d["pair_id"]]):
                        endpoints = decision["pair_id"].split("|", 1)
                        relation = decision.get("relation")
                        if relation in ("skos:altLabel", "skos:hiddenLabel", "exact_match"):
                            constraints.merge(*endpoints)
                        elif relation in ("skos:related", "skos:broader", "owl:differentFrom"):
                            constraints.separate(*endpoints)
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

            judged_merges, _ = apply_synonym_adjudication(
                staging_conn,
                candidates,
                cand_meta,
                decisions,
                materialized.fact_linked_pairs,
                rlm_stats=rlm_stats,
                corpus_root=settings.corpus_root,
                touched_entity_ids=touched_eids,
            )
            merges.extend(judged_merges)
        elif touched_eids:
            canonicalize_entity_labels_edc(
                staging_conn,
                entity_ids=sorted(set(touched_eids)),
                corpus_root=settings.corpus_root,
            )

        for merged in merges:
            bind_merged_evidence(staging_conn, merged)
            for old_fid, new_fid in merged.fact_bindings:
                if new_fid is not None and old_fid in prior_fact_definitions:
                    prev_def = prior_fact_definitions[old_fid]
                    existing_def = prior_fact_definitions.get(new_fid, "")
                    prior_fact_definitions[new_fid] = max((prev_def, existing_def), key=lambda v: (len(v), v))
        canonicalize_fact_definitions(staging_conn, prior_definitions=prior_fact_definitions)
        canonicalize_fact_text(staging_conn)

        # 3. Closed-World SHACL Deterministic Auto-Healing & RLM Healing on staging_conn
        auto_healed = restore_asserted_fact_sources(staging_conn)
        refresh_source_confidences(staging_conn)
        deterministic_healed = auto_heal_deterministic_shacl(staging_conn)
        auto_healed += deterministic_healed
        if deterministic_healed:
            refresh_source_confidences(staging_conn)
        if auto_healed > 0:
            canonicalize_fact_text(staging_conn)
        rlm_stats["shacl_deterministic_healed"] = auto_healed
        rlm_stats["shacl_repairs_applied"] += auto_healed

        _, violations, focus_dossiers, _ = collect_shacl_violation_dossiers(
            staging_conn,
            settings.corpus_root,
            ast_context_hops=settings.ast_context_hops,
        )
        rlm_stats["shacl_violations_before"] = len(violations)
        rlm_stats["shacl_violations_initial"] = len(violations)
        if violations:
            source_text = {ptr.chunk_id: text for ptr, text in complete_chunks}
            for dossier in focus_dossiers.values():
                dossier["source_passages"] = {cid: source_text[cid] for cid in dossier["chunk_ids"]}
            repairs = []
            violations_by_focus: dict[str, list[SHACLViolation]] = {}
            for v in violations:
                violations_by_focus.setdefault(v.focus_node, []).append(v)
            for focus_idx, focus_node in enumerate(sorted(violations_by_focus), start=1):
                if extraction.progress_callback is not None:
                    extraction.progress_callback(
                        f"indexing: SHACL healing ({focus_idx}/{len(violations_by_focus)} focus nodes)"
                    )
                focus_batch = tuple(violations_by_focus[focus_node])
                dossiers = {focus_node: focus_dossiers[focus_node]}
                healed = await heal_shacl_violations(
                    SHACLHealInput(
                        violations=focus_batch,
                        focus_dossiers=dossiers,
                        focus_context=json.dumps([focus_dossiers[focus_node]], ensure_ascii=False),
                        program_identity=settings.program_identity,
                        shacl_identity=settings.shacl_identity,
                    )
                )
                repairs.extend(healed.repairs)
                all_traces.extend(healed.traces)
                all_telemetry.extend(healed.telemetry)
                shacl_audit.extend(healed.audit_trail)
                rlm_tokens = sum(int(x.get("tokens_used", 0)) for x in healed.audit_trail if x.get("mode") == "rlm")
                direct_tokens = sum(
                    int(x.get("tokens_used", 0)) for x in healed.audit_trail if x.get("mode") == "direct"
                )
                rlm_stats["shacl_healer_tokens"] += rlm_tokens if rlm_tokens > 0 else direct_tokens
            if repairs:

                def merge_repaired_entities(src: str, dst: str, kind: str) -> bool:
                    merged = merge_entities(staging_conn, src, dst, label_kind=kind)
                    bind_merged_evidence(staging_conn, merged)
                    for old_fid, new_fid in merged.fact_bindings:
                        if new_fid is not None and old_fid in prior_fact_definitions:
                            prev_def = prior_fact_definitions[old_fid]
                            existing_def = prior_fact_definitions.get(new_fid, "")
                            prior_fact_definitions[new_fid] = max((prev_def, existing_def), key=lambda v: (len(v), v))
                    return True

                applied = apply_shacl_repairs(staging_conn, repairs, merge_entities_fn=merge_repaired_entities)
                for rep in repairs:
                    if str(rep.get("action", "")).strip() == "set_definition":
                        f_node = str(rep.get("focus_node", "")).strip()
                        f_val = str(rep.get("value", "")).strip()
                        if f_node.startswith("fact-") and f_val:
                            prior_fact_definitions[f_node] = f_val
                refresh_source_confidences(staging_conn)
                if auto_heal_deterministic_shacl(staging_conn):
                    refresh_source_confidences(staging_conn)
                canonicalize_fact_definitions(staging_conn, prior_definitions=prior_fact_definitions)
                canonicalize_fact_text(staging_conn)
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

        # 4. Extract final post-merge, post-SHACL-repair PreparedGraphRows from staging_conn
        final_entities = tuple(
            (str(r[0]), str(r[1]), str(r[2]))
            for r in staging_conn.execute(
                "SELECT entity_id, name, definition FROM entities ORDER BY entity_id ASC"
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
        # Ensure any entity/fact whose text changed during merge or SHACL repair is embedded
        reembed_entities = [
            (eid, name, defn)
            for eid, name, defn in final_entities
            if (eid not in published_entity_vec_ids and eid not in entity_vec_by_id)
            or initial_entity_meta.get(eid) != (name, defn)
        ]
        if reembed_entities:
            re_vecs = await _embed_strings(
                [fmt_entity(name, defn) for _, name, defn in reembed_entities],
                stage="entity_embed",
            )
            for (eid, _, _), vec in zip(reembed_entities, re_vecs, strict=True):
                entity_vec_by_id[eid] = vec
            upsert_entity_vectors(staging_conn, [(eid, entity_vec_by_id[eid]) for eid, _, _ in reembed_entities])

        reembed_facts = [
            f_row
            for f_row in final_facts
            if f_row[0] not in published_fact_vec_ids or initial_fact_defs.get(f_row[0]) != (f_row[4], f_row[5])
        ]
        if reembed_facts:
            rf_texts: list[str] = []
            for f_row in reembed_facts:
                items = json.loads(f_row[4])
                rf_texts.append(fmt_fact(items[0], f_row[2], items[2], f_row[5]))
            rf_vecs = await _embed_strings(rf_texts, stage="fact_embed")
            upsert_fact_vectors(
                staging_conn,
                [(f_row[0], vec) for f_row, vec in zip(reembed_facts, rf_vecs, strict=True)],
            )
        staging_conn.execute(
            "DELETE FROM entity_candidate_vectors WHERE sense_id NOT IN (SELECT sense_id FROM entity_mentions)"
        )
        return GraphPlanResult(
            tables=graph_tables(
                staging_conn,
                chunks if incremental else complete_chunks,
                tuple(sorted(input_fingerprints.items())),
                seeded_rows_by_table=seeded_rows_by_table,
                db_identity=resolved_db_identity,
            ),
            chunk_ids=tuple(out.chunk_id for out in ordered_outputs),
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
