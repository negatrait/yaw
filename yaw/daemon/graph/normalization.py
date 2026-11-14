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

_SEQ = (list, tuple)


def _attr(obj, name, default=""):
    return getattr(obj, name, None) or default


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    triples: list[StructuredTriple]
    entities: list[tuple[str, str]]
    definitions: dict[str, str]
    labels: set[str]
    surfaces: dict[str, str]


class _TPCache(dict):
    def __missing__(self, key):
        val = f"{text_processing(key)!s}"
        if key.__class__ is str:
            self[key] = val
        return val


def _rec_surf(surfaces, key, raw):
    if key and (prev := surfaces.get(key)) != raw and (c := clean_surface_label(raw) or raw) and prev != c:
        surfaces[key] = preferred_surface(prev, c, label_norm=key) if prev else c
    return key


def _norm_ent(raw, lit_raws, tp, surfaces):
    s = f"{raw!s}".strip()
    return _rec_surf(surfaces, s if s in lit_raws or is_scalar_label(s) else tp[s], s)


def _resolve_def(defs, raw, key, fb=""):
    if d := clean_schema_definition(raw) or defs.get(key) or defs.get(fb, ""):
        defs.setdefault(key, d)
    return d


def preferred_surface(*values: str, label_norm: str) -> str:
    """Choose a clean display spelling by a total order, independently of arrival."""
    pool = {}
    for v in values:
        if s := v.strip():
            pool[clean_surface_label(v) or s] = 0
    return max(
        pool or values,
        key=lambda v: (text_processing(v) == label_norm, v != v.lower(), -len(v), v),
    )


def normalize_openie_record(rec: OpenIERecord) -> NormalizedRecord:
    """Preserve literal values and distinct senses while normalizing graph labels."""
    surfaces: dict[str, str] = {}
    labels: set[str] = set()
    tp = _TPCache()

    st_rows = []
    ent_raws = set()
    lit_raws = set()
    for st in _attr(rec, "structured_triples", ()):
        s_raw = f"{st.subj!s}".strip()
        o_str = f"{st.obj!s}"
        o_raw = o_str.strip()
        kind_lit = _attr(st, "obj_kind") == "literal"
        if s_raw:
            ent_raws.add(s_raw)
        if o_raw:
            (lit_raws if kind_lit else ent_raws).add(o_raw)
        st_rows.append((st, s_raw, o_str, o_raw, kind_lit))
    if lit_raws:
        lit_raws.difference_update(ent_raws, map(tp.__getitem__, ent_raws))

    defs: dict[str, str] = {}
    senses: dict[str, list[tuple[str, str]]] = {}
    for pair in _attr(rec, "entity_definitions", ()):
        if (
            isinstance(pair, _SEQ)
            and len(pair) > 1
            and (enorm := _norm_ent(pair[0], lit_raws, tp, surfaces))
            and (edef := clean_schema_definition(pair[1]))
        ):
            defs.setdefault(enorm, edef)
            senses.setdefault(enorm, []).append((enorm, edef))

    ent_pairs = []
    for ent in _attr(rec, "entities", ()):
        is_seq = isinstance(ent, _SEQ) and ent
        if e_norm := _norm_ent(ent[0] if is_seq else ent, lit_raws, tp, surfaces):
            ent_pairs.append((e_norm, _resolve_def(defs, ent[1] if is_seq and len(ent) > 1 else "", e_norm)))
    for pairs in senses.values():
        ent_pairs.extend(pairs)

    seen_senses: dict[tuple[str, str], tuple[str, str]] = {}
    for k, d in ent_pairs:
        labels.add(k)
        seen_senses.setdefault((k, d.lower()), (k, d))

    triples: list[StructuredTriple] = []
    for st, s_raw, o_str, o_raw, kind_lit in st_rows:
        if not ((s_norm := tp[st.subj]) and (p_norm := tp[st.pred])):
            continue
        is_scalar, det_dt = detect_scalar_literal(o_str)
        is_lit = kind_lit or is_scalar or is_scalar_label(o_str)
        if not (o_val := o_raw if is_lit else tp[st.obj]):
            continue
        o_tok = tp[o_val]
        subj_d = _resolve_def(defs, st.subj_def, s_norm)
        obj_d = _resolve_def(defs, st.obj_def, o_val, o_tok)
        if s_norm == o_tok and not is_lit and (not (subj_d and obj_d) or subj_d.lower() == obj_d.lower()):
            raise ValueError(f"Statement relates a resource sense to itself: {s_raw!r} -> {o_raw!r}")

        labels.update((s_norm, o_val, o_tok or s_norm))
        _rec_surf(surfaces, s_norm, s_raw)
        _rec_surf(surfaces, o_val, o_raw)
        if o_tok != o_val:
            _rec_surf(surfaces, o_tok, o_raw)

        raw_dt = f"{_attr(st, 'obj_datatype')!s}".strip()
        triples.append(
            StructuredTriple(
                subj=s_norm,
                pred=st.pred.strip(),
                obj=o_val,
                spo_def=st.spo_def.strip() or f"{s_norm} {p_norm} {o_val}.",
                subj_def=subj_d,
                obj_def=obj_d,
                obj_kind="literal" if is_lit else "entity",
                obj_datatype=det_dt if is_lit and raw_dt in ("", "rdfs:Literal") else raw_dt,
                obj_lang=f"{_attr(st, 'obj_lang')!s}",
                llm_prob=st.llm_prob,
            )
        )

    return NormalizedRecord(triples, [*seen_senses.values()], defs, labels, surfaces)
