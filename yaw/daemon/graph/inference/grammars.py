"""Pure-JSON Context-Free Grammars (GBNF) for DSPy."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import dspy

from yaw.core import clean_schema_definition, text_processing

# ---------------------------------------------------------------------------
# Pure-JSON Context-Free Grammars (GBNF) for dspy.JSONAdapter on llama.cpp
# ---------------------------------------------------------------------------

# A resource label needs lexical content (yaw.core.is_resource_label): punctuation, numbers, dates and
# other data values are literals, which only the triple stage may introduce as objects.
_NER_PAIR_GBNF_RULES = r"""entity-pair ::= "[" label ", " string "]"
label ::= "\"" non-letter* letter char* "\""
letter ::= [a-zA-Z\x80-\U0010FFFF]
non-letter ::= [\x20\x21\x23-\x40\x5B\x5D-\x60\x7B-\x7E]
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_NER_GBNF_GRAMMAR = (
    'root ::= "{\\"named_entities\\": " entity-list "}"\n'
    'entity-list ::= "[]" | "[" entity-pair (", " entity-pair)* "]"\n' + _NER_PAIR_GBNF_RULES
)

_STATEMENT_GBNF_RULES = r"""literal-statement ::= literal-pred ", \"obj\": " literal-slot
resource-pred ::= string | "[" string ", \"owl:ObjectProperty\"]"
literal-pred ::= string | "[" string ", \"owl:DatatypeProperty\"]"
literal-slot ::= "{\"value\": " string ", \"datatype\": " string ", \"lang\": " language "}"
language ::= "\"" char* "\""
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

_OPEN_TRIPLE_GBNF_RULES = (
    'triple ::= "{\\"sub\\": " string ", \\"pred\\": " pred-obj ", \\"def\\": " string "}"\n'
    'pred-obj ::= resource-pred ", \\"obj\\": " string | literal-statement\n' + _STATEMENT_GBNF_RULES
)

DSPY_TRIPLES_GBNF_GRAMMAR = (
    'root ::= "{\\"triples\\": " triple-list "}"\n'
    'triple-list ::= "[]" | "[" triple (", " triple)* "]"\n' + _OPEN_TRIPLE_GBNF_RULES
)

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
    'action-kind ::= "\\"set_definition\\"" | "\\"set_pref_label\\"" | '
    '"\\"demote_to_alt_label\\"" | "\\"merge_altLabel\\"" | "\\"merge_hiddenLabel\\"" | '
    '"\\"add_different_from\\"" | "\\"assert_related\\"" | '
    '"\\"assert_broader\\"" | "\\"invert_broader\\"" | "\\"demote_sibling_to_related\\"" | '
    '"\\"remove_hierarchy_cycle\\"" | "\\"remove_conflicting_related\\"" | '
    '"\\"delete_dangling\\"" | "\\"delete_invalid_fact\\""\n'
    'label ::= "\\"" non-letter* letter char* "\\""\n'
    "letter ::= [a-zA-Z\\x80-\\U0010FFFF]\n"
    "non-letter ::= [\\x20\\x21\\x23-\\x40\\x5B\\x5D-\\x60\\x7B-\\x7E]\n"
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
    return f'root ::= "{{\\"named_entities\\": [" entity-list "]}}"\nentity-list ::= ({seq})?\n' + _NER_PAIR_GBNF_RULES


def build_dspy_triples_grammar(
    max_triples: int | None = None,
    known_entities: Sequence[Any] | None = None,
) -> str:
    """Pin resource endpoints to NER labels; only literal objects may introduce values."""
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

    if known_entities is not None and not deduped_known:
        return 'root ::= "{\\"triples\\": []}"\n'
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

    if not deduped_known:
        return root_line + list_line + _OPEN_TRIPLE_GBNF_RULES

    pairs = [list(item[:2]) for item in known_entities or () if isinstance(item, (list, tuple)) and len(item) >= 2]
    ambiguous = {name for name in deduped_known if len({pair[1] for pair in pairs if pair[0] == name}) > 1}
    endpoints: list[Any] = [name for name in deduped_known if name not in ambiguous]
    endpoints.extend(pair for pair in pairs if pair[0] in ambiguous and pair not in endpoints)
    definition_of = {pair[0]: pair[1] for pair in pairs if pair[0] not in ambiguous}

    def sense(endpoint: Any) -> tuple[str, str]:
        label, definition = (endpoint, definition_of.get(endpoint, "")) if isinstance(endpoint, str) else endpoint
        return text_processing(label), clean_schema_definition(definition).strip().casefold()

    # A statement never relates a resource sense to itself (native graph spec §5.3).
    senses = [sense(endpoint) for endpoint in endpoints]
    lines = [
        root_line,
        list_line,
        'triple ::= "{\\"sub\\": " statement ", \\"def\\": " string "}"\n',
        "statement ::= " + " | ".join(f"subject-{i}" for i in range(len(endpoints))) + "\n",
    ]
    for i, endpoint in enumerate(endpoints):
        lines.append(f"endpoint-{i} ::= {gbnf_escape_literal(json.dumps(endpoint, ensure_ascii=False))}\n")
        others = [f"endpoint-{j}" for j in range(len(endpoints)) if senses[j] != senses[i]]
        if others:
            lines.append(f"objects-{i} ::= {' | '.join(others)}\n")
            branch = f'(resource-pred ", \\"obj\\": " objects-{i} | literal-statement)'
        else:
            branch = "literal-statement"
        lines.append(f'subject-{i} ::= endpoint-{i} ", \\"pred\\": " {branch}\n')
    return "".join(lines) + _STATEMENT_GBNF_RULES


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


def build_dspy_shacl_repair_grammar(
    max_repairs: int | None = None, *, allowed_actions: Mapping[str, set[str]] | None = None
) -> str:
    """Build pure-JSON GBNF grammar for ProposeSHACLRepairs (`{"repairs": [...]}`)."""
    if allowed_actions:
        if max_repairs is not None:
            raise ValueError("Report-bound repair coverage cannot use a generic repair-count cap")
        branches = []
        for i, (focus, actions) in enumerate(sorted(allowed_actions.items())):
            choices = " | ".join(gbnf_escape_literal(json.dumps(action)) for action in sorted(actions))
            value_rule = "label" if actions == {"set_pref_label"} else "string"
            branches.append(
                f'repair-{i} ::= "{{\\"focus_node\\": " {gbnf_escape_literal(json.dumps(focus))} '
                f'", \\"action\\": " ({choices}) ", \\"value\\": " {value_rule} "}}"'
            )
        grammar = DSPY_SHACL_REPAIR_GBNF_GRAMMAR
        coverage = ' ", " '.join(f'repair-{i} (", " repair-{i})*' for i in range(len(branches)))
        return (
            "\n".join(
                "repair ::= " + " | ".join(f"repair-{i}" for i in range(len(branches)))
                if line.startswith("repair ::=")
                else f'repair-list ::= "[]" | "[" {coverage} "]"'
                if line.startswith("repair-list ::=")
                else line
                for line in grammar.splitlines()
            )
            + "\n"
            + "\n".join(branches)
        )
    if max_repairs is None or max_repairs <= 0:
        return DSPY_SHACL_REPAIR_GBNF_GRAMMAR
    seq = "repair"
    for _ in range(max_repairs - 1):
        seq = f'repair (", " {seq})?'
    return (
        'root ::= "{\\"repairs\\": [" repair-list "]}"\n'
        f"repair-list ::= ({seq})?\n"
        'repair ::= "{\\"focus_node\\": " string ", \\"action\\": " action-kind ", \\"value\\": " string "}"\n'
        'action-kind ::= "\\"set_definition\\"" | "\\"set_pref_label\\"" | '
        '"\\"demote_to_alt_label\\"" | "\\"merge_altLabel\\"" | "\\"merge_hiddenLabel\\"" | '
        '"\\"add_different_from\\"" | "\\"assert_related\\"" | '
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
    "DSPY_SHACL_REPAIR_GBNF_GRAMMAR",
    "DSPY_RERANK_GBNF_GRAMMAR",
    "DSPY_QA_GBNF_GRAMMAR",
    "DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR",
    "DSPY_QUERY_INTENT_GBNF_GRAMMAR",
    "gbnf_escape_literal",
    "build_dspy_ner_grammar",
    "build_dspy_triples_grammar",
    "build_dspy_missing_entity_defs_grammar",
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
