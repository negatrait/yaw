"""Deterministic OpenIE normalization without inference or storage access."""

from dataclasses import dataclass

from yaw.core import (
    OpenIERecord,
    StructuredTriple,
    clean_schema_definition,
    clean_surface_label,
    detect_scalar_literal,
    is_scalar_label,
    text_processing,
)


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    triples: list[StructuredTriple]
    entities: list[tuple[str, str]]
    definitions: dict[str, str]
    labels: set[str]
    surfaces: dict[str, str]


def preferred_surface(*values: str, label_norm: str) -> str:
    """Choose a clean display spelling by a total order, independently of arrival."""
    cleaned = [clean_surface_label(v) or v.strip() for v in values if v.strip()]
    pool = cleaned or list(values)
    return max(
        pool,
        key=lambda value: (
            text_processing(value) == label_norm,
            value != value.lower(),
            -len(value),
            value,
        ),
    )


def normalize_openie_record(rec: OpenIERecord) -> NormalizedRecord:
    """Preserve literal values and distinct senses while normalizing graph labels."""
    all_norm_labels: set[str] = set()
    surfaces: dict[str, str] = {}

    def _record_surface(norm_key: str, raw_val: str) -> None:
        clean_raw = clean_surface_label(raw_val) or raw_val.strip()
        if not norm_key or not clean_raw:
            return
        surfaces[norm_key] = preferred_surface(surfaces.get(norm_key, clean_raw), clean_raw, label_norm=norm_key)

    raw_st_items = list(getattr(rec, "structured_triples", ()) or ())
    entity_role_raws: set[str] = set()
    literal_only_raws: set[str] = set()
    for st_item in raw_st_items:
        s_r = str(getattr(st_item, "subj", "")).strip()
        o_r = str(getattr(st_item, "obj", "")).strip()
        if s_r:
            entity_role_raws.add(s_r)
            entity_role_raws.add(str(text_processing(s_r)))
        if getattr(st_item, "obj_kind", "entity") == "literal":
            if o_r:
                literal_only_raws.add(o_r)
        elif o_r:
            entity_role_raws.add(o_r)
            entity_role_raws.add(str(text_processing(o_r)))
    literal_only_raws -= entity_role_raws

    stage1_defs: dict[str, str] = {}
    stage1_senses: dict[str, list[str]] = {}
    for raw_pair in getattr(rec, "entity_definitions", ()) or ():
        if isinstance(raw_pair, (list, tuple)) and len(raw_pair) >= 2:
            raw_e0 = str(raw_pair[0]).strip()
            is_lit_norm = raw_e0 in literal_only_raws or is_scalar_label(raw_e0)
            enorm = raw_e0 if is_lit_norm else str(text_processing(raw_e0))
            _record_surface(enorm, raw_e0)
            edef = clean_schema_definition(raw_pair[1])
            if enorm and edef:
                if enorm not in stage1_defs:
                    stage1_defs[enorm] = edef
                senses_lst = stage1_senses.setdefault(enorm, [])
                if all(d.lower() != edef.lower() for d in senses_lst):
                    senses_lst.append(edef)

    ner_entities: list[tuple[str, str]] = []
    seen_ner_norms: set[str] = set()
    seen_ner_sense_keys: set[tuple[str, str]] = set()
    for raw_ent in getattr(rec, "entities", ()) or ():
        if isinstance(raw_ent, (list, tuple)) and len(raw_ent) >= 1:
            e_str = str(raw_ent[0]).strip()
            e_inline_def = clean_schema_definition(raw_ent[1]) if len(raw_ent) >= 2 else ""
        else:
            e_str = str(raw_ent).strip()
            e_inline_def = ""
        is_lit_norm = e_str in literal_only_raws or is_scalar_label(e_str)
        e_norm = e_str if is_lit_norm else str(text_processing(e_str))
        if not e_norm:
            continue
        _record_surface(e_norm, e_str)
        e_def = e_inline_def or stage1_defs.get(e_norm, "")
        s_key = (e_norm, e_def.lower())
        if s_key in seen_ner_sense_keys or (not e_def and e_norm in seen_ner_norms):
            continue
        seen_ner_norms.add(e_norm)
        seen_ner_sense_keys.add(s_key)
        if e_inline_def:
            if e_norm not in stage1_defs:
                stage1_defs[e_norm] = e_inline_def
            senses_lst = stage1_senses.setdefault(e_norm, [])
            if all(d.lower() != e_inline_def.lower() for d in senses_lst):
                senses_lst.append(e_inline_def)
        all_norm_labels.add(e_norm)
        ner_entities.append((e_norm, e_def))

    for enorm, senses_lst in stage1_senses.items():
        for edef in senses_lst:
            s_key = (enorm, edef.lower())
            if s_key not in seen_ner_sense_keys:
                seen_ner_norms.add(enorm)
                seen_ner_sense_keys.add(s_key)
                all_norm_labels.add(enorm)
                ner_entities.append((enorm, edef))

    st_list: list[StructuredTriple] = list(raw_st_items)
    normalized_sts: list[StructuredTriple] = []
    for st in st_list:
        s_raw_str = str(st.subj).strip()
        o_raw_str = str(st.obj).strip()
        s_norm = str(text_processing(st.subj))
        p_norm = str(text_processing(st.pred))
        is_scalar, detected_dt = detect_scalar_literal(str(st.obj))
        is_literal_obj = getattr(st, "obj_kind", "entity") == "literal" or is_scalar or is_scalar_label(str(st.obj))
        o_val = o_raw_str if is_literal_obj else str(text_processing(st.obj))
        if not s_norm or not p_norm or not o_val:
            continue
        o_tok_norm = str(text_processing(o_val))
        raw_subj_d = clean_schema_definition(st.subj_def)
        subj_d = raw_subj_d or stage1_defs.get(s_norm, "")
        if subj_d:
            if s_norm not in stage1_defs:
                stage1_defs[s_norm] = subj_d
            senses_lst = stage1_senses.setdefault(s_norm, [])
            if all(d.lower() != subj_d.lower() for d in senses_lst):
                senses_lst.append(subj_d)

        raw_obj_d = clean_schema_definition(st.obj_def)
        obj_d = raw_obj_d or stage1_defs.get(o_val, "") or stage1_defs.get(o_tok_norm, "")
        if s_norm == o_tok_norm and not is_literal_obj and (not subj_d or not obj_d or subj_d.lower() == obj_d.lower()):
            raise ValueError(f"Statement relates a resource sense to itself: {s_raw_str!r} -> {o_raw_str!r}")

        _record_surface(s_norm, s_raw_str)
        _record_surface(o_val, o_raw_str)
        all_norm_labels.add(s_norm)
        all_norm_labels.add(o_val)
        if o_tok_norm:
            all_norm_labels.add(o_tok_norm)
            _record_surface(o_tok_norm, o_raw_str)
        spo_d = st.spo_def.strip() or f"{s_norm} {p_norm} {o_val}."
        if obj_d:
            if o_val not in stage1_defs:
                stage1_defs[o_val] = obj_d
            senses_lst = stage1_senses.setdefault(o_val, [])
            if all(d.lower() != obj_d.lower() for d in senses_lst):
                senses_lst.append(obj_d)

        raw_dt = str(getattr(st, "obj_datatype", "") or "").strip()
        resolved_dt = (raw_dt if raw_dt and raw_dt != "rdfs:Literal" else detected_dt) if is_literal_obj else raw_dt
        normalized_sts.append(
            StructuredTriple(
                subj=s_norm,
                pred=st.pred.strip(),
                obj=o_val,
                spo_def=spo_d,
                subj_def=subj_d,
                obj_def=obj_d,
                obj_kind="literal" if is_literal_obj else "entity",
                obj_datatype=resolved_dt,
                obj_lang=str(getattr(st, "obj_lang", "") or ""),
                llm_prob=st.llm_prob,
            )
        )

    return NormalizedRecord(normalized_sts, ner_entities, stage1_defs, all_norm_labels, surfaces)
