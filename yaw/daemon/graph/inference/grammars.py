"""Pure-JSON Context-Free Grammars (GBNF) for DSPy."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import dspy

# ---------------------------------------------------------------------------
# Pure-JSON Context-Free Grammars (GBNF) for dspy.JSONAdapter on llama.cpp
# ---------------------------------------------------------------------------

DSPY_NER_GBNF_GRAMMAR = r"""root ::= "{\"named_entities\": " entity-list "}"
entity-list ::= "[]" | "[" entity-pair (", " entity-pair)* "]"
entity-pair ::= "[" string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_TRIPLES_GBNF_GRAMMAR = r"""root ::= "{\"triples\": " triple-list "}"
triple-list ::= "[]" | "[" triple (", " triple)* "]"
triple ::= "{\"sub\": " entity-slot ", \"pred\": " pred-slot ", \"obj\": " obj-slot ", \"def\": " string "}"
pred-slot ::= string | "[" string ", " ("\"owl:ObjectProperty\"" | "\"owl:DatatypeProperty\"") "]"
entity-slot ::= new-entity-pair | string
obj-slot ::= entity-slot | string
new-entity-pair ::= "[" string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR = r"""root ::= "{\"entity_definitions\": " entity-list "}"
entity-list ::= "[]" | "[" entity-pair (", " entity-pair)* "]"
entity-pair ::= "[" string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_SHACL_REPAIR_GBNF_GRAMMAR = (
    'root ::= "{\\"repairs\\": " repair-list "}"\n'
    'repair-list ::= "[]" | "[" repair (", " repair)* "]"\n'
    'repair ::= "{\\"focus_node\\": " string ", \\"action\\": " action-kind ", \\"value\\": " string '
    '"}"\n'
    'action-kind ::= "\\"set_definition\\"" | "\\"rewrite_definition\\"" | "\\"set_pref_label\\"" | '
    '"\\"demote_to_alt_label\\"" | "\\"merge_altLabel\\"" | "\\"merge_hiddenLabel\\"" | '
    '"\\"add_different_from\\"" | "\\"assert_differentFrom\\"" | "\\"assert_related\\"" | '
    '"\\"assert_broader\\"" | "\\"invert_broader\\"" | "\\"demote_sibling_to_related\\"" | '
    '"\\"remove_hierarchy_cycle\\"" | "\\"remove_conflicting_related\\"" | '
    '"\\"delete_dangling\\"" | "\\"delete_invalid_fact\\""\n'
    'string ::= "\\"" char+ "\\""\n'
    'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
)

DSPY_RERANK_GBNF_GRAMMAR = r"""root ::= "{\"fact\": [" fact-list "]}"
fact-list ::= (triple (", " triple)*)?
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QA_GBNF_GRAMMAR = r"""root ::= "{\"reasoning\": " string ", \"answer\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR = r"""root ::= "{\"new_instruction\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QUERY_INTENT_GBNF_GRAMMAR = r"""root ::= "{\"query_intent\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_CONCEPT_SCHEME_DEFS_GBNF_GRAMMAR: str = (
    'root ::= "{\\"entity_definitions\\": [" scheme-list "]}"\n'
    'scheme-list ::= (scheme-item (", " scheme-item)*)?\n'
    'scheme-item ::= "{\\"scheme_id\\": " string ", \\"pref_label\\": " string '
    '", \\"definition\\": " string ", \\"history_note\\": " string "}"\n'
    'string ::= "\\"" char+ "\\""\n'
    'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
)


def gbnf_escape_literal(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def build_dspy_ner_grammar(max_entities: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ExtractEntities, optionally capping entity count for sizecoding."""
    if max_entities is None or max_entities <= 0:
        return DSPY_NER_GBNF_GRAMMAR
    seq = "entity-pair"
    for _ in range(max_entities - 1):
        seq = f'entity-pair (", " {seq})?'
    return (
        'root ::= "{\\"named_entities\\": [" entity-list "]}"\n'
        f"entity-list ::= ({seq})?\n"
        'entity-pair ::= "[" string ", " string "]"\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_triples_grammar(
    max_triples: int | None = None,
    known_entities: Sequence[Any] | None = None,
) -> str:
    """Build pure-JSON GBNF grammar for ExtractTriples, constraining known Stage 1 entities as bare strings and \
novel entities as [name, def] pairs."""
    deduped_known: list[str] = []
    if known_entities:
        for item in known_entities:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                name = item[0].strip()
            else:
                continue
            if name and name not in deduped_known:
                deduped_known.append(name)

    if (max_triples is None or max_triples <= 0) and not deduped_known:
        return DSPY_TRIPLES_GBNF_GRAMMAR

    if max_triples is None or max_triples <= 0:
        root_line = 'root ::= "{\\"triples\\": " triple-list "}"\n'
        list_line = 'triple-list ::= "[]" | "[" triple (", " triple)* "]"\n'
    else:
        seq = "triple"
        for _ in range(max_triples - 1):
            seq = f'triple (", " {seq})?'
        root_line = 'root ::= "{\\"triples\\": [" triple-list "]}"\n'
        list_line = f"triple-list ::= ({seq})?\n"

    triple_line = (
        'triple ::= "{\\"sub\\": " entity-slot ", \\"pred\\": " pred-slot '
        '", \\"obj\\": " obj-slot ", \\"def\\": " string "}"\n'
        'pred-slot ::= string | "[" string ", " ("\\"owl:ObjectProperty\\"" | "\\"owl:DatatypeProperty\\"") "]"\n'
    )
    if deduped_known:
        known_rule = " | ".join(gbnf_escape_literal(json.dumps(e, ensure_ascii=False)) for e in deduped_known)
        slot_lines = (
            "entity-slot ::= known-entity | new-entity-pair\n"
            "obj-slot ::= entity-slot | string\n"
            f"known-entity ::= {known_rule}\n"
            'new-entity-pair ::= "[" string ", " string "]"\n'
        )
    else:
        slot_lines = (
            "entity-slot ::= new-entity-pair | string\n"
            "obj-slot ::= entity-slot | string\n"
            'new-entity-pair ::= "[" string ", " string "]"\n'
        )

    return (
        root_line
        + list_line
        + triple_line
        + slot_lines
        + 'string ::= "\\"" char+ "\\""\n'
        + 'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_missing_entity_defs_grammar(
    missing_names: Sequence[str] | None = None,
) -> str:
    """Build pure-JSON GBNF grammar for DefineMissingEntities (`{"entity_definitions": [[name, def], ...]}`)."""
    deduped: list[str] = []
    if missing_names:
        for name in missing_names:
            if isinstance(name, str) and name.strip() and name.strip() not in deduped:
                deduped.append(name.strip())
    if not deduped:
        return DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR
    seq = "entity-pair"
    for _ in range(len(deduped) - 1):
        seq = f'entity-pair (", " {seq})?'
    name_rule = " | ".join(gbnf_escape_literal(json.dumps(name, ensure_ascii=False)) for name in deduped)
    return (
        'root ::= "{\\"entity_definitions\\": [" entity-list "]}"\n'
        f"entity-list ::= ({seq})?\n"
        'entity-pair ::= "[" entity-name ", " string "]"\n'
        f"entity-name ::= {name_rule}\n"
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_concept_scheme_defs_grammar(
    scheme_ids: Sequence[str] | Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Build pure-JSON GBNF grammar for DefineMissingEntities synthesizing concept schemes \
(`{"entity_definitions": [{"scheme_id": ..., "pref_label": ..., "definition": ..., "history_note": ...}, ...]}`)."""
    extracted_ids: list[str] = []
    if scheme_ids:
        for item in scheme_ids:
            if isinstance(item, str):
                sid = item.strip()
            elif isinstance(item, Mapping):
                sid = str(item.get("scheme_id") or item.get("id") or "").strip()
            else:
                sid = str(item).strip()
            if sid and sid not in extracted_ids:
                extracted_ids.append(sid)
    if not extracted_ids:
        return DSPY_CONCEPT_SCHEME_DEFS_GBNF_GRAMMAR

    rules: list[str] = []
    item_names: list[str] = []
    for i, sid in enumerate(extracted_ids):
        sid_literal = gbnf_escape_literal(json.dumps(sid, ensure_ascii=False))
        rules.append(
            f'scheme-item-{i} ::= "{{\\"scheme_id\\": " {sid_literal} '
            f'", \\"pref_label\\": " string '
            f'", \\"definition\\": " string '
            f'", \\"history_note\\": " string "}}"'
        )
        item_names.append(f"scheme-item-{i}")

    seq = ' ", " '.join(item_names)
    root_line = f'root ::= "{{\\"entity_definitions\\": [" {seq} "]}}"\n'
    return (
        root_line
        + "\n".join(rules)
        + '\nstring ::= "\\"" char+ "\\""\n'
        + 'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def _build_pair_boolean_grammar(
    bool_fields: Sequence[str],
    *,
    pair_ids: Sequence[str] | None = None,
    max_decisions: int | None = None,
    max_reasoning_chars: int | None = 240,
    custom_suffix: str | None = None,
    extra_rules: Sequence[str] = (),
) -> str:
    r_limit = f"{{1,{int(max_reasoning_chars)}}}" if max_reasoning_chars and max_reasoning_chars > 0 else "+"
    bool_suffix = (
        custom_suffix
        if custom_suffix is not None
        else " ".join(f"{gbnf_escape_literal(f', "{field}": ')} boolean" for field in bool_fields)
    )
    rules: list[str] = []
    if pair_ids is not None:
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Synonym grammar requires nonempty, unique pair IDs")
        for i, pair_id in enumerate(pair_ids):
            ids = pair_id.split("|")
            if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
                raise ValueError(f"Invalid synonym pair ID {pair_id!r}")
            prefix = gbnf_escape_literal('{"pair_id": ' + json.dumps(pair_id) + ', "reasoning": ')
            rules.append(f"decision-{i} ::= {prefix} reasoning-str {bool_suffix} {gbnf_escape_literal('}')}")
        seq = ' ", " '.join(f"decision-{i}" for i in range(len(pair_ids)))
        rules.insert(0, f"root ::= {gbnf_escape_literal('{"decisions": [')} {seq} {gbnf_escape_literal(']}')}")
    else:
        dec_pfx = gbnf_escape_literal('{"pair_id": ')
        dec_rsn = gbnf_escape_literal(', "reasoning": ')
        dec_cls = gbnf_escape_literal("}")
        dec_rule = f"decision ::= {dec_pfx} string {dec_rsn} reasoning-str {bool_suffix} {dec_cls}"
        if max_decisions is not None and max_decisions > 0:
            seq = "decision"
            for _ in range(max_decisions - 1):
                seq = f'decision (", " {seq})?'
            rules.extend(
                [
                    'root ::= "{\\"decisions\\": [" decision-list "]}"',
                    f"decision-list ::= ({seq})?",
                    dec_rule,
                ]
            )
        else:
            rules.extend(
                [
                    'root ::= "{\\"decisions\\": [" decision-list "]}"',
                    'decision-list ::= (decision (", " decision)*)?',
                    dec_rule,
                ]
            )
        rules.append('string ::= "\\"" char+ "\\""')
    rules.extend(extra_rules)
    rules.extend(
        [
            f'reasoning-str ::= "\\"" char{r_limit} "\\""',
            'boolean ::= "true" | "false"',
            'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})',
        ]
    )
    return "\n".join(rules) + "\n"


def build_dspy_label_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = 240,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeLabels with conditional equivalent/alt_label/hidden_label branching."""
    equiv_prefix = gbnf_escape_literal(', "equivalent": ')
    def_key = gbnf_escape_literal('true, "definition": ')
    alt_key = gbnf_escape_literal('false, "alt_label": ')
    hid_key = gbnf_escape_literal('false, "hidden_label": ')
    return _build_pair_boolean_grammar(
        (),
        pair_ids=pair_ids,
        max_decisions=max_decisions,
        max_reasoning_chars=max_reasoning_chars,
        custom_suffix=f"{equiv_prefix} equiv-branch",
        extra_rules=(
            f"equiv-branch ::= {def_key} reasoning-str | {alt_key} alt-branch",
            f'alt-branch ::= "true" | {hid_key} boolean',
        ),
    )


def build_dspy_hierarchy_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = 240,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeHierarchy."""
    return _build_pair_boolean_grammar(
        ("broader", "narrower"),
        pair_ids=pair_ids,
        max_decisions=max_decisions,
        max_reasoning_chars=max_reasoning_chars,
    )


def build_dspy_related_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = 240,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeRelated."""
    return _build_pair_boolean_grammar(
        ("related",),
        pair_ids=pair_ids,
        max_decisions=max_decisions,
        max_reasoning_chars=max_reasoning_chars,
    )


def build_dspy_shacl_repair_grammar(max_repairs: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ProposeSHACLRepairs (`{"repairs": [...]}`)."""
    if max_repairs is None or max_repairs <= 0:
        return DSPY_SHACL_REPAIR_GBNF_GRAMMAR
    seq = "repair"
    for _ in range(max_repairs - 1):
        seq = f'repair (", " {seq})?'
    return (
        'root ::= "{\\"repairs\\": [" repair-list "]}"\n'
        f"repair-list ::= ({seq})?\n"
        'repair ::= "{\\"focus_node\\": " string ", \\"action\\": " action-kind ", \\"value\\": " string "}"\n'
        'action-kind ::= "\\"set_definition\\"" | "\\"rewrite_definition\\"" | "\\"set_pref_label\\"" | '
        '"\\"demote_to_alt_label\\"" | "\\"merge_altLabel\\"" | "\\"merge_hiddenLabel\\"" | '
        '"\\"add_different_from\\"" | "\\"assert_differentFrom\\"" | "\\"assert_related\\"" | '
        '"\\"assert_broader\\"" | "\\"invert_broader\\"" | "\\"demote_sibling_to_related\\"" | '
        '"\\"remove_hierarchy_cycle\\"" | "\\"remove_conflicting_related\\"" | '
        '"\\"delete_dangling\\"" | "\\"delete_invalid_fact\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_qa_grammar(max_reasoning_chars: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for AnswerQuestion, optionally capping reasoning length."""
    if max_reasoning_chars is None or max_reasoning_chars <= 0:
        return DSPY_QA_GBNF_GRAMMAR
    return (
        'root ::= "{\\"reasoning\\": " reasoning-str ", \\"answer\\": " string "}"\n'
        f'reasoning-str ::= "\\"" char{{1,{int(max_reasoning_chars)}}} "\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_instruction_proposal_grammar(max_chars: int | None = 1500) -> str:
    """Build pure-JSON GBNF grammar for GEPA/MIPROv2 instruction synthesis on Qwen (`llama.cpp` GPU)."""
    if max_chars is None or max_chars <= 0:
        return DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR
    return (
        'root ::= "{\\"new_instruction\\": " instruction-str "}"\n'
        f'instruction-str ::= "\\"" char{{1,{int(max_chars)}}} "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_query_intent_grammar() -> str:
    """Build pure-JSON GBNF grammar for ExtractQueryIntent (`{"query_intent": "..."}`)."""
    return DSPY_QUERY_INTENT_GBNF_GRAMMAR


def build_dspy_rerank_grammar(
    candidate_facts: Sequence[tuple[str, str, str] | list[str]],
    max_facts: int = 4,
) -> str:
    """Build dynamic candidate-literal pure-JSON GBNF grammar for DSPy FilterTriples (JSONAdapter)."""
    candidates: list[str] = []
    for item in candidate_facts:
        if not isinstance(item, (list, tuple)) or len(item) != 3 or not all(isinstance(x, str) for x in item):
            return DSPY_RERANK_GBNF_GRAMMAR
        serialized = json.dumps(list(item), ensure_ascii=False)
        if serialized not in candidates:
            candidates.append(serialized)

    if not candidates:
        return 'root ::= "{\\"fact\\": []}"\n'

    limit = min(max_facts, len(candidates))
    seq = "triple"
    for _ in range(limit - 1):
        seq = f'triple (", " {seq})?'
    triple_rule = " | ".join(gbnf_escape_literal(c) for c in candidates)
    return f'root ::= "{{\\"fact\\": [" fact-list "]}}"\nfact-list ::= ({seq})?\ntriple ::= {triple_rule}\n'


def lm_supports_gbnf(lm: Any = None) -> bool:
    """Return True if `lm` runs on a `llama.cpp` backend that supports GBNF grammars."""
    target_lm = lm if lm is not None else getattr(dspy.settings, "lm", None)
    if target_lm is None:
        return True
    explicit = getattr(target_lm, "_yaw_supports_gbnf", None)
    if isinstance(explicit, bool):
        return explicit
    recipe = getattr(target_lm, "_yaw_recipe", None)
    if recipe == "flm":
        return False
    model_name = str(getattr(target_lm, "model", "")).lower()
    return "flm" not in model_name


__all__ = [
    "DSPY_NER_GBNF_GRAMMAR",
    "DSPY_TRIPLES_GBNF_GRAMMAR",
    "DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR",
    "DSPY_CONCEPT_SCHEME_DEFS_GBNF_GRAMMAR",
    "DSPY_SHACL_REPAIR_GBNF_GRAMMAR",
    "DSPY_RERANK_GBNF_GRAMMAR",
    "DSPY_QA_GBNF_GRAMMAR",
    "DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR",
    "DSPY_QUERY_INTENT_GBNF_GRAMMAR",
    "gbnf_escape_literal",
    "build_dspy_ner_grammar",
    "build_dspy_triples_grammar",
    "build_dspy_missing_entity_defs_grammar",
    "build_dspy_concept_scheme_defs_grammar",
    "build_dspy_hierarchy_judge_grammar",
    "build_dspy_label_judge_grammar",
    "build_dspy_related_judge_grammar",
    "build_dspy_shacl_repair_grammar",
    "build_dspy_qa_grammar",
    "build_dspy_instruction_proposal_grammar",
    "build_dspy_query_intent_grammar",
    "build_dspy_rerank_grammar",
    "lm_supports_gbnf",
]
