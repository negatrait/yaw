"""Deterministic OpenIE normalization without inference or storage access."""

from dataclasses import dataclass

from yaw.core import OpenIERecord, StructuredTriple, detect_scalar_literal
from yaw.daemon.graph.inference import text_processing


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    triples: list[StructuredTriple]
    entities: list[tuple[str, str]]
    definitions: dict[str, str]
    labels: set[str]
    surfaces: dict[str, str]


def normalize_openie_record(rec: OpenIERecord) -> NormalizedRecord:
    """Preserve literal values and distinct senses while normalizing graph labels."""
    all_norm_labels: set[str] = set()
    surfaces: dict[str, str] = {}

    def _record_surface(norm_key: str, raw_val: str) -> None:
        clean_raw = raw_val.strip()
        if not norm_key or not clean_raw:
            return
        previous = surfaces.get(norm_key)
        if previous is None or (previous == previous.lower() and clean_raw != clean_raw.lower()):
            surfaces[norm_key] = clean_raw

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
            is_lit_norm = raw_e0 in literal_only_raws or detect_scalar_literal(raw_e0)[0]
            enorm = raw_e0 if is_lit_norm else str(text_processing(raw_e0))
            _record_surface(enorm, raw_e0)
            edef = str(raw_pair[1]).strip()
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
            e_inline_def = str(raw_ent[1]).strip() if len(raw_ent) >= 2 and raw_ent[1] is not None else ""
        else:
            e_str = str(raw_ent).strip()
            e_inline_def = ""
        is_lit_norm = e_str in literal_only_raws or detect_scalar_literal(e_str)[0]
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
    if not st_list and getattr(rec, "triples", ()):
        for t in rec.triples:
            if len(t) >= 3:
                s_raw, p_raw, o_raw = (
                    str(t[0]).strip(),
                    str(t[1]).strip(),
                    str(t[2]).strip(),
                )
                spo_synth = f"{s_raw} {p_raw} {o_raw}."
                s_raw_norm = str(text_processing(s_raw))
                o_raw_norm = str(text_processing(o_raw))
                s_def_val = stage1_defs.get(s_raw_norm, "")
                o_def_val = stage1_defs.get(o_raw_norm, "")
                if s_raw_norm and s_raw_norm == o_raw_norm and len(stage1_senses.get(s_raw_norm, ())) >= 2:
                    s_def_val = stage1_senses[s_raw_norm][0]
                    o_def_val = stage1_senses[s_raw_norm][1]
                st_list.append(
                    StructuredTriple(
                        subj=s_raw,
                        subj_def=s_def_val,
                        pred=p_raw,
                        obj=o_raw,
                        obj_def=o_def_val,
                        spo_def=spo_synth,
                    )
                )
    normalized_sts: list[StructuredTriple] = []
    for st in st_list:
        s_raw_str = str(st.subj).strip()
        o_raw_str = str(st.obj).strip()
        s_norm = str(text_processing(st.subj))
        p_norm = str(text_processing(st.pred))
        is_literal_obj = getattr(st, "obj_kind", "entity") == "literal" or detect_scalar_literal(str(st.obj))[0]
        o_val = o_raw_str if is_literal_obj else str(text_processing(st.obj))
        if not s_norm or not p_norm or not o_val:
            continue
        if detect_scalar_literal(s_norm)[0]:
            if not is_literal_obj:
                s_norm, o_val = o_val, s_norm
                s_raw_str, o_raw_str = o_raw_str, s_raw_str
                is_literal_obj = True
            else:
                continue
        _record_surface(s_norm, s_raw_str)
        _record_surface(o_val, o_raw_str)
        all_norm_labels.add(s_norm)
        all_norm_labels.add(o_val)
        o_tok_norm = str(text_processing(o_val))
        if o_tok_norm:
            all_norm_labels.add(o_tok_norm)
            _record_surface(o_tok_norm, o_raw_str)
        spo_d = st.spo_def.strip() or f"{s_norm} {p_norm} {o_val}."
        raw_subj_d = st.subj_def.strip()
        subj_d = raw_subj_d or stage1_defs.get(s_norm, "")
        if subj_d:
            if s_norm not in stage1_defs:
                stage1_defs[s_norm] = subj_d
            senses_lst = stage1_senses.setdefault(s_norm, [])
            if all(d.lower() != subj_d.lower() for d in senses_lst):
                senses_lst.append(subj_d)

        raw_obj_d = st.obj_def.strip()
        obj_d = raw_obj_d or stage1_defs.get(o_val, "") or stage1_defs.get(o_tok_norm, "")
        if (
            not is_literal_obj
            and s_norm == o_val
            and subj_d
            and obj_d
            and subj_d.lower() == obj_d.lower()
            and len(stage1_senses.get(o_val, ())) >= 2
        ):
            obj_d = next(
                (d for d in stage1_senses[o_val] if d.lower() != subj_d.lower()),
                obj_d,
            )
        if obj_d:
            if o_val not in stage1_defs:
                stage1_defs[o_val] = obj_d
            senses_lst = stage1_senses.setdefault(o_val, [])
            if all(d.lower() != obj_d.lower() for d in senses_lst):
                senses_lst.append(obj_d)

        normalized_sts.append(
            StructuredTriple(
                subj=s_norm,
                pred=p_norm,
                obj=o_val,
                spo_def=spo_d,
                subj_def=subj_d,
                obj_def=obj_d,
                obj_kind="literal" if is_literal_obj else "entity",
                obj_datatype=str(getattr(st, "obj_datatype", "") or ""),
                obj_lang=str(getattr(st, "obj_lang", "") or ""),
            )
        )

    return NormalizedRecord(normalized_sts, ner_entities, stage1_defs, all_norm_labels, surfaces)
