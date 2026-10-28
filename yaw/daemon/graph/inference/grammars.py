"""Pure-JSON Context-Free Grammars (GBNF) for DSPy."""

import json
from collections.abc import Sequence
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
triple ::= "{\"sub\": " entity-slot ", \"pred\": " pred-slot ", \"obj\": " entity-slot ", \"def\": " string "}"
pred-slot ::= string | "[" string ", " ("\"owl:ObjectProperty\"" | "\"owl:DatatypeProperty\"") "]"
entity-slot ::= new-entity-pair | string
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

DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR = (
    'root ::= "{\\"decisions\\": " decision-list "}"\n'
    'decision-list ::= "[]" | "[" decision (", " decision)* "]"\n'
    'decision ::= "{\\"pair_id\\": " string ", \\"relation\\": " relation-kind ", \\"canonical\\": " string "}"\n'
    'relation-kind ::= "\\"skos:altLabel\\"" | "\\"skos:hiddenLabel\\"" | "\\"skos:related\\"" | '
    '"\\"skos:broader\\"" | "\\"owl:differentFrom\\""\n'
    'string ::= "\\"" char+ "\\""\n'
    'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
)

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
        '", \\"obj\\": " entity-slot ", \\"def\\": " string "}"\n'
        'pred-slot ::= string | "[" string ", " ("\\"owl:ObjectProperty\\"" | "\\"owl:DatatypeProperty\\"") "]"\n'
    )
    if deduped_known:
        known_rule = " | ".join(gbnf_escape_literal(json.dumps(e, ensure_ascii=False)) for e in deduped_known)
        slot_lines = (
            "entity-slot ::= known-entity | new-entity-pair\n"
            f"known-entity ::= {known_rule}\n"
            'new-entity-pair ::= "[" string ", " string "]"\n'
        )
    else:
        slot_lines = 'entity-slot ::= new-entity-pair | string\nnew-entity-pair ::= "[" string ", " string "]"\n'

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


def build_dspy_synonym_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
) -> str:
    """Build pure-JSON GBNF grammar for JudgeSynonyms (`{"decisions": [...]}`)."""
    if pair_ids is not None:
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Synonym grammar requires nonempty, unique pair IDs")
        sequence = ' ", " '.join(f"decision-{i}" for i in range(len(pair_ids)))
        opening = gbnf_escape_literal('{"decisions": [')
        closing = gbnf_escape_literal("]}")
        rules = [f"root ::= {opening} {sequence} {closing}"]
        for i, pair_id in enumerate(pair_ids):
            ids = pair_id.split("|")
            if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
                raise ValueError(f"Invalid synonym pair ID {pair_id!r}")
            prefix = '{"pair_id": ' + json.dumps(pair_id) + ', "relation": '
            middle = ', "canonical": '
            choices = " | ".join(gbnf_escape_literal(json.dumps(eid)) for eid in ids)
            rules.append(
                f"decision-{i} ::= {gbnf_escape_literal(prefix)} relation-kind "
                f"{gbnf_escape_literal(middle)} ({choices}) {gbnf_escape_literal('}')}"
            )
        rules.extend(
            line for line in DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR.splitlines() if line.startswith("relation-kind ::=")
        )
        return "\n".join(rules) + "\n"
    if max_decisions is None or max_decisions <= 0:
        return DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR
    seq = "decision"
    for _ in range(max_decisions - 1):
        seq = f'decision (", " {seq})?'
    return (
        'root ::= "{\\"decisions\\": [" decision-list "]}"\n'
        f"decision-list ::= ({seq})?\n"
        'decision ::= "{\\"pair_id\\": " string ", \\"relation\\": " relation-kind ", \\"canonical\\": " string "}"\n'
        'relation-kind ::= "\\"skos:altLabel\\"" | "\\"skos:hiddenLabel\\"" | "\\"skos:related\\"" | '
        '"\\"skos:broader\\"" | "\\"owl:differentFrom\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
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
    "DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR",
    "DSPY_SHACL_REPAIR_GBNF_GRAMMAR",
    "DSPY_RERANK_GBNF_GRAMMAR",
    "DSPY_QA_GBNF_GRAMMAR",
    "DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR",
    "DSPY_QUERY_INTENT_GBNF_GRAMMAR",
    "gbnf_escape_literal",
    "build_dspy_ner_grammar",
    "build_dspy_triples_grammar",
    "build_dspy_missing_entity_defs_grammar",
    "build_dspy_synonym_judge_grammar",
    "build_dspy_shacl_repair_grammar",
    "build_dspy_qa_grammar",
    "build_dspy_instruction_proposal_grammar",
    "build_dspy_query_intent_grammar",
    "build_dspy_rerank_grammar",
    "lm_supports_gbnf",
]
