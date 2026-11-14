"""Pure-JSON Context-Free Grammars (GBNF) for DSPy."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import dspy

from yaw.core import clean_schema_definition, text_processing

_SEQ = (list, tuple)
_CACHE = lru_cache(64)
_DUMP_UTF8 = json.JSONEncoder(ensure_ascii=False).encode
_DUMP_ASCII = json.JSONEncoder().encode

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
    'triple ::= "{\\"sub\\": " resource-slot ", \\"pred\\": " pred-obj ", \\"def\\": " string "}"\n'
    'pred-obj ::= resource-pred ", \\"obj\\": " resource-slot | literal-statement\n'
    'resource-slot ::= string | "[" string ", " string "]"\n' + _STATEMENT_GBNF_RULES
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

_SHACL_ACTION_KIND = (
    'action-kind ::= "\\"set_definition\\"" | "\\"set_pref_label\\"" | '
    '"\\"demote_to_alt_label\\"" | "\\"merge_altLabel\\"" | "\\"merge_hiddenLabel\\"" | '
    '"\\"add_different_from\\"" | "\\"assert_related\\"" | '
    '"\\"assert_broader\\"" | "\\"invert_broader\\"" | "\\"demote_sibling_to_related\\"" | '
    '"\\"remove_hierarchy_cycle\\"" | "\\"remove_conflicting_related\\"" | '
    '"\\"delete_dangling\\"" | "\\"delete_invalid_fact\\""\n'
)

_SHACL_TAIL = (
    _SHACL_ACTION_KIND + 'label ::= "\\"" non-letter* letter char* "\\""\n'
    "letter ::= [a-zA-Z\\x80-\\U0010FFFF]\n"
    "non-letter ::= [\\x20\\x21\\x23-\\x40\\x5B\\x5D-\\x60\\x7B-\\x7E]\n"
    'string ::= "\\"" char+ "\\""\n'
    'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
)

DSPY_SHACL_REPAIR_GBNF_GRAMMAR = (
    'root ::= "{\\"repairs\\": " repair-list "}"\n'
    'repair-list ::= "[]" | "[" repair (", " repair)* "]"\n'
    'repair ::= "{\\"focus_node\\": " string ", \\"action\\": " action-kind ", \\"value\\": " string "}"\n'
    + _SHACL_TAIL
)

DSPY_RERANK_GBNF_GRAMMAR = r"""root ::= "{\"fact\": [" fact-list "]}"
fact-list ::= (triple (", " triple)*)?
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QA_GBNF_GRAMMAR = r"""root ::= "{\"plan\": " string ", \"answer\": " string "}"
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


@_CACHE
def _seq(sym, n):
    return f"""({(sym + ' (", " ') * (k := n - 1)}{sym}{")?" * k})?"""


def _jesc(s):
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _ju(obj):
    return _jesc(_DUMP_UTF8(obj))


def _ja(obj):
    return _jesc(_DUMP_ASCII(obj))


def gbnf_escape_literal(text: str) -> str:
    return _jesc(text).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


@_CACHE
def build_dspy_ner_grammar(max_entities: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ExtractEntities, optionally capping entity count for sizecoding."""
    if not max_entities or max_entities < 0:
        return DSPY_NER_GBNF_GRAMMAR
    return (
        'root ::= "{\\"named_entities\\": [" entity-list "]}"\n'
        f"entity-list ::= {_seq('entity-pair', max_entities)}\n{_NER_PAIR_GBNF_RULES}"
    )


def build_dspy_triples_grammar(max_triples: int | None = None, known_entities: Sequence[Any] | None = None) -> str:
    """Pin resource endpoints to NER labels; only literal objects may introduce values."""
    if known_entities is not None:
        deduped = {}
        clean_pairs = {}
        def_lookup = {}
        defs_seen = {}
        ambiguous = set()
        for item in known_entities:
            is_seq = isinstance(item, _SEQ)
            raw = item[0] if is_seq and item else item
            if isinstance(raw, str) and (name := raw.strip()):
                deduped[name] = None
                if is_seq and len(item) > 1 and (clean_v := clean_schema_definition(item[1]).strip()):
                    def_lookup.setdefault(name, clean_v)
                    if norm_k := f"{text_processing(name)!s}":
                        def_lookup.setdefault(norm_k, clean_v)
                    if defs_seen.setdefault(name, clean_v) != clean_v:
                        ambiguous.add(name)
                    clean_pairs[name, clean_v] = None
        if not deduped:
            return 'root ::= "{\\"triples\\": []}"\n'
    elif not max_triples or max_triples < 0:
        return DSPY_TRIPLES_GBNF_GRAMMAR

    header = (
        f'root ::= "{{\\"triples\\": [" triple-list "]}}"\ntriple-list ::= {_seq("triple", max_triples)}\n'
        if max_triples and max_triples > 0
        else 'root ::= "{\\"triples\\": " triple-list "}"\ntriple-list ::= "[]" | "[" triple (", " triple)* "]"\n'
    )
    if known_entities is None:
        return header + _OPEN_TRIPLE_GBNF_RULES

    records = []
    subs = []
    for ep in (*deduped, *clean_pairs):
        t = isinstance(ep, tuple)
        label = ep[0] if t else ep
        if (label in ambiguous) != t:
            continue
        raw_def = ep[1] if t else defs_seen.get(label, "")
        norm_label = f"{text_processing(label)!s}"
        clean_def = (
            clean_schema_definition(raw_def).strip()
            or def_lookup.get(label, "")
            or def_lookup.get(norm_label, "")
            or ""
        )
        i = len(records)
        records.append((ep, (norm_label, clean_def.strip().casefold()), f"endpoint-{i}"))
        subs.append(f"subject-{i}")

    # A statement never relates a resource sense to itself (native graph spec Ã‚Â§5.3).
    lines = [
        f'{header}triple ::= "{{\\"sub\\": " statement ", \\"def\\": " string "}}"\nstatement ::= {" | ".join(subs)}\n'
    ]
    for i, (ep, si, ename) in enumerate(records):
        others = []
        for _, s, en in records:
            if s != si:
                others.append(en)
        obj_line = f"objects-{i} ::= {' | '.join(others)}\n" if others else ""
        rhs = f'(resource-pred ", \\"obj\\": " objects-{i} | literal-statement)' if others else "literal-statement"
        lines.append(f'{ename} ::= {_ju(ep)}\n{obj_line}subject-{i} ::= {ename} ", \\"pred\\": " {rhs}\n')
    lines.append(_STATEMENT_GBNF_RULES)
    return "".join(lines)


def build_dspy_missing_entity_defs_grammar(missing_names: Sequence[str] | None = None) -> str:
    """Build pure-JSON GBNF grammar for DefineMissingEntities (`{"entity_definitions": [[name, def], ...]}`)."""
    deduped = {}
    if missing_names:
        for name in missing_names:
            if isinstance(name, str) and (s := name.strip()) and s not in deduped:
                deduped[s] = _ju(s)
    if not deduped:
        return DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR
    return (
        'root ::= "{\\"entity_definitions\\": [" entity-list "]}"\n'
        f"entity-list ::= {_seq('entity-pair', len(deduped))}\n"
        'entity-pair ::= "[" entity-name ", " string "]"\n'
        f"entity-name ::= {' | '.join(deduped.values())}\n"
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def _judge_grammar(suffix, pair_ids, max_decisions, max_reasoning_chars, extra=""):
    r_limit = f"{{1,{int(max_reasoning_chars)}}}" if max_reasoning_chars and max_reasoning_chars > 0 else "+"
    if pair_ids is not None:
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("Synonym grammar requires nonempty, unique pair IDs")
        decs = []
        dnames = []
        for i, pair_id in enumerate(pair_ids):
            ids = pair_id.split("|")
            if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
                raise ValueError(f"Invalid synonym pair ID {pair_id!r}")
            dnames.append(dn := f"decision-{i}")
            pfx = _jesc(f'{{"pair_id": {_DUMP_ASCII(pair_id)}, "reasoning": ')
            decs.append(f'{dn} ::= {pfx} reasoning-str {suffix} "}}"')
        head = f"""root ::= "{{\\"decisions\\": [" {' ", " '.join(dnames)} "]}}"\n""" + "\n".join(decs) + "\n"
    else:
        dlist = (
            _seq("decision", max_decisions) if max_decisions and max_decisions > 0 else '(decision (", " decision)*)?'
        )
        head = (
            'root ::= "{\\"decisions\\": [" decision-list "]}"\n'
            f"decision-list ::= {dlist}\n"
            f'decision ::= "{{\\"pair_id\\": " string ", \\"reasoning\\": " reasoning-str {suffix} "}}"\n'
            'string ::= "\\"" char+ "\\""\n'
        )
    return (
        f"{head}{extra}"
        f'reasoning-str ::= "\\"" char{r_limit} "\\""\n'
        'boolean ::= "true" | "false"\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_label_judge_grammar(
    max_decisions: int | None = None, *, pair_ids: Sequence[str] | None = None, max_reasoning_chars: int | None = None
) -> str:
    """Build pure-JSON GBNF grammar for JudgeLabels with conditional equivalent/hidden_label branching."""
    return _judge_grammar(
        '", \\"equivalent\\": " equiv-branch',
        pair_ids,
        max_decisions,
        max_reasoning_chars,
        'equiv-branch ::= "true, \\"definition\\": " reasoning-str ", \\"hidden_label\\": " boolean | "false"\n',
    )


def build_dspy_relation_judge_grammar(
    max_decisions: int | None = None,
    *,
    pair_ids: Sequence[str] | None = None,
    max_reasoning_chars: int | None = None,
    associated_pair_ids: Sequence[str] = (),
) -> str:
    """Constrain fact-associated pairs to non-negative relations before sampling."""
    choices = ("skos:broader", "skos:narrower", "skos:related", "owl:differentFrom")
    if pair_ids is None:
        return _judge_grammar(
            '", \\"relation\\": " relation',
            None,
            max_decisions,
            max_reasoning_chars,
            "relation ::= " + " | ".join(_ja(x) for x in choices) + "\n",
        )
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Synonym grammar requires nonempty, unique pair IDs")
    associated = set(associated_pair_ids)
    rules = []
    for i, pair_id in enumerate(pair_ids):
        ids = pair_id.split("|")
        if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
            raise ValueError(f"Invalid synonym pair ID {pair_id!r}")
        prefix = _jesc(f'{{"pair_id": {_DUMP_ASCII(pair_id)}, "reasoning": ')
        allowed = choices[:3] if pair_id in associated else choices
        rules.append(
            f'decision-{i} ::= {prefix} reasoning-str ", \\"relation\\": " ({" | ".join(_ja(x) for x in allowed)}) "}}"'
        )
    limit = f"{{1,{int(max_reasoning_chars)}}}" if max_reasoning_chars and max_reasoning_chars > 0 else "+"
    return (
        'root ::= "{\\"decisions\\": [" '
        + ' ", " '.join(f"decision-{i}" for i in range(len(pair_ids)))
        + ' "]}"\n'
        + "\n".join(rules)
        + "\n"
        + f'reasoning-str ::= "\\"" char{limit} "\\""\n'
        + 'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_shacl_repair_grammar(
    max_repairs: int | None = None, *, allowed_actions: Mapping[str, set[str]] | None = None
) -> str:
    """Build pure-JSON GBNF grammar for ProposeSHACLRepairs (`{"repairs": [...]}`)."""
    if allowed_actions:
        if max_repairs is not None:
            raise ValueError("Report-bound repair coverage cannot use a generic repair-count cap")
        branches = []
        cov = []
        rnames = []
        for i, (focus, actions) in enumerate(sorted(allowed_actions.items())):
            rnames.append(rn := f"repair-{i}")
            cov.append(f'{rn} (", " {rn})*')
            choices = " | ".join(map(_ja, sorted(actions)))
            vrule = "label" if actions == {"set_pref_label"} else "string"
            branches.append(
                f'{rn} ::= "{{\\"focus_node\\": " {_ja(focus)} '
                f'", \\"action\\": " ({choices}) ", \\"value\\": " {vrule} "}}"'
            )
        return (
            'root ::= "{\\"repairs\\": " repair-list "}"\n'
            f"""repair-list ::= "[]" | "[" {' ", " '.join(cov)} "]"\n"""
            f"repair ::= {' | '.join(rnames)}\n{_SHACL_TAIL}" + "\n".join(branches)
        )
    if not max_repairs or max_repairs < 0:
        return DSPY_SHACL_REPAIR_GBNF_GRAMMAR
    return (
        'root ::= "{\\"repairs\\": [" repair-list "]}"\n'
        f"repair-list ::= {_seq('repair', max_repairs)}\n"
        'repair ::= "{\\"focus_node\\": " string ", \\"action\\": " action-kind ", \\"value\\": " string "}"\n'
        f"{_SHACL_ACTION_KIND}"
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


@_CACHE
def build_dspy_qa_grammar(max_reasoning_chars: int | None = None) -> str:
    """Constrain DSPy 3.5's wire-level plan, which it restores as QA reasoning."""
    if not max_reasoning_chars or max_reasoning_chars < 0:
        return DSPY_QA_GBNF_GRAMMAR
    return (
        'root ::= "{\\"plan\\": " reasoning-str ", \\"answer\\": " string "}"\n'
        f'reasoning-str ::= "\\"" char{{1,{int(max_reasoning_chars)}}} "\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


@_CACHE
def build_dspy_instruction_proposal_grammar(max_chars: int | None = 1500) -> str:
    """Build pure-JSON GBNF grammar for GEPA/MIPROv2 instruction synthesis on local LLM (`llama.cpp` GPU)."""
    if not max_chars or max_chars < 0:
        return DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR
    return (
        'root ::= "{\\"new_instruction\\": " instruction-str "}"\n'
        f'instruction-str ::= "\\"" char{{1,{int(max_chars)}}} "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_query_intent_grammar() -> str:
    """Build pure-JSON GBNF grammar for ExtractQueryIntent (`{"query_intent": "..."}`)."""
    return DSPY_QUERY_INTENT_GBNF_GRAMMAR


def build_dspy_rerank_grammar(candidate_facts: Sequence[tuple[str, str, str] | list[str]], max_facts: int = 4) -> str:
    """Build dynamic candidate-literal pure-JSON GBNF grammar for DSPy FilterTriples (JSONAdapter)."""
    candidates = {}
    for item in candidate_facts:
        if not (
            isinstance(item, _SEQ)
            and len(item) == 3
            and isinstance(item[0], str)
            and isinstance(item[1], str)
            and isinstance(item[2], str)
        ):
            return DSPY_RERANK_GBNF_GRAMMAR
        candidates[_ju(item)] = None
    if not candidates:
        return 'root ::= "{\\"fact\\": []}"\n'
    return (
        'root ::= "{\\"fact\\": [" fact-list "]}"\n'
        f"fact-list ::= {_seq('triple', min(max_facts, len(candidates)))}\n"
        f"triple ::= {' | '.join(candidates)}\n"
    )


def lm_supports_gbnf(lm: Any = None) -> bool:
    """Return True if `lm` runs on a `llama.cpp` backend that supports GBNF grammars."""
    if (target := lm if lm is not None else getattr(dspy.settings, "lm", None)) is None:
        return True
    if isinstance(explicit := getattr(target, "_yaw_supports_gbnf", None), bool):
        return explicit
    return getattr(target, "_yaw_recipe", None) != "flm" and "flm" not in f"{getattr(target, 'model', '')!s}".lower()


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
    "build_dspy_label_judge_grammar",
    "build_dspy_relation_judge_grammar",
    "build_dspy_shacl_repair_grammar",
    "build_dspy_qa_grammar",
    "build_dspy_instruction_proposal_grammar",
    "build_dspy_query_intent_grammar",
    "build_dspy_rerank_grammar",
    "lm_supports_gbnf",
]
