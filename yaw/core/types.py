"""Frozen, zero-dependency data contracts and pure coordinate/ID helpers for YAW."""

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, overload

ErrorCode = Literal[
    "INVALID_INPUT",
    "NOT_READY",
    "EMPTY_INDEX",
    "BUSY",
    "MODEL_ERROR",
    "CONSISTENCY_ERROR",
    "SHUTTING_DOWN",
]


class YawRuntimeError(RuntimeError):
    """Structured runtime error carrying a canonical ErrorCode for CLI and MCP transport."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code: ErrorCode = code
        self.message: str = message


class FatalGraphPublicationError(BaseException):
    """Fatal graph publication failure that aborts optimizer execution without being swallowed."""

    def __init__(self, original: BaseException) -> None:
        self.original = original
        super().__init__(str(original))


def json_document_fields(data: object, default_title: str = "") -> tuple[str, str]:
    """One rendering contract for decoded document objects and projected values."""
    if isinstance(data, str):
        return default_title, data
    if not isinstance(data, dict):
        raise ValueError("JSON document must be an object or string")
    title = data.get("title", default_title)
    if "text" in data:
        body = data["text"]
    elif "content" in data:
        body = data["content"]
    else:
        body = json.dumps(data, ensure_ascii=False)
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("JSON document title/body must be strings")
    return title, body


def render_document_text(title: str, body: str) -> str:
    """Render source fields once, preserving an already present title line."""
    if title and body != title and not body.startswith((title + "\n", title + "\r\n")):
        return f"{title}\n{body}"
    return body


@dataclass(slots=True, frozen=True)
class ChunkPointer:
    """1-indexed line span pointer to an immutable markdown/text chunk on disk."""

    chunk_id: str
    rel_posix_path: str
    start_line: int
    end_line: int
    title: str = ""
    ast_island_start: int = 0
    ast_island_end: int = 0


@dataclass(slots=True, frozen=True)
class HydratedPassage:
    """A passage hydrated on demand from disk with optional stale detection."""

    pointer: ChunkPointer
    text: str
    score: float
    stale_on_disk: bool = False


@dataclass(slots=True, frozen=True)
class SynonymEvidenceBatch:
    """A bounded model request with reversible IDs and content-addressed evidence."""

    candidates: tuple[dict[str, Any], ...]
    payload: str
    identifiers: dict[str, str]
    construction_seconds: float = 0.0

    @property
    def cache_identity(self) -> str:
        value = ["synonym-evidence-v2", self.payload, self.identifiers, self.candidates]
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def resolve_decisions(self, decisions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Validate exact pair coverage and resolve request-local IDs before graph writes."""
        pairs = {p["pair_id"] for p in json.loads(self.payload)["pairs"]}
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for decision in decisions:
            pair = str(decision.get("pair_id", "") or "").strip()
            if not pair or pair not in pairs or pair in seen:
                raise ValueError(f"Unknown or duplicate synonym pair {pair!r}")
            if "|" not in pair:
                raise ValueError(f"Invalid synonym pair ID {pair!r}")
            a, b = pair.split("|", 1)
            real_a = self.identifiers.get(a, a)
            real_b = self.identifiers.get(b, b)
            resolved_entry: dict[str, Any] = {
                **decision,
                "pair_id": f"{real_a}|{real_b}",
            }
            if "source_id" in decision and decision["source_id"]:
                src = str(decision["source_id"])
                if src not in (a, b):
                    raise ValueError(f"Invalid source_id {src!r} for synonym pair {pair!r}")
                resolved_entry["source_id"] = self.identifiers.get(src, src)
            if "target_id" in decision and decision["target_id"]:
                tgt = str(decision["target_id"])
                if tgt not in (a, b):
                    raise ValueError(f"Invalid target_id {tgt!r} for synonym pair {pair!r}")
                resolved_entry["target_id"] = self.identifiers.get(tgt, tgt)
            if "canonical" in decision and decision["canonical"]:
                canonical = str(decision["canonical"])
                if canonical not in (a, b):
                    raise ValueError(f"Invalid canonical ID {canonical!r} for synonym pair {pair!r}")
                resolved_entry["canonical"] = self.identifiers.get(canonical, canonical)
            result.append(resolved_entry)
            seen.add(pair)
        if seen != pairs:
            raise ValueError(f"Missing synonym decisions for {sorted(pairs - seen)}")
        return result

    def subset(self, pair_ids: Collection[str]) -> SynonymEvidenceBatch:
        """Return a new SynonymEvidenceBatch containing only the requested pair IDs."""
        target_ids = set(pair_ids)
        parsed = json.loads(self.payload)
        all_pairs = parsed.get("pairs", [])
        filtered_pairs = []
        for p in all_pairs:
            alias_pair = p.get("pair_id", "")
            if "|" in alias_pair:
                a, b = alias_pair.split("|", 1)
                real_pair = f"{self.identifiers.get(a, a)}|{self.identifiers.get(b, b)}"
            else:
                real_pair = alias_pair
            if alias_pair in target_ids or real_pair in target_ids:
                filtered_pairs.append(p)

        filtered_pair_aliases = {p["pair_id"] for p in filtered_pairs}
        filtered_pair_reals = {
            f"{self.identifiers.get(p['pair_id'].split('|', 1)[0], p['pair_id'].split('|', 1)[0])}|"
            f"{self.identifiers.get(p['pair_id'].split('|', 1)[1], p['pair_id'].split('|', 1)[1])}"
            for p in filtered_pairs
            if "|" in p.get("pair_id", "")
        }

        filtered_candidates = [
            c
            for c in self.candidates
            if c.get("pair_id") in filtered_pair_aliases
            or c.get("pair_id") in filtered_pair_reals
            or f"{c.get('entity_a')}|{c.get('entity_b')}" in filtered_pair_reals
            or f"{c.get('entity_a')}|{c.get('entity_b')}" in filtered_pair_aliases
        ]

        needed_aliases = {
            part for p in filtered_pairs if "|" in p.get("pair_id", "") for part in p["pair_id"].split("|", 1)
        }
        filtered_identifiers = {k: v for k, v in self.identifiers.items() if k in needed_aliases}

        ratio = (len(filtered_pairs) / len(all_pairs)) if all_pairs else 1.0
        new_construction_seconds = self.construction_seconds * ratio

        new_payload = json.dumps(
            {
                "format": "synonym-evidence-v2",
                "pairs": filtered_pairs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return SynonymEvidenceBatch(
            candidates=tuple(filtered_candidates),
            payload=new_payload,
            identifiers=filtered_identifiers,
            construction_seconds=new_construction_seconds,
        )


@dataclass(slots=True, frozen=True)
class StructuredTriple:
    """An OpenIE relation triple enriched with inline subject, object, and statement definitions."""

    subj: str
    subj_def: str = ""
    pred: str = ""
    obj: str = ""
    obj_def: str = ""
    spo_def: str = ""
    obj_kind: Literal["entity", "literal"] = "entity"
    obj_datatype: str = ""
    obj_lang: str = ""

    @property
    def subject(self) -> str:
        return self.subj

    @property
    def predicate(self) -> str:
        return self.pred

    @property
    def object(self) -> str:
        return self.obj

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subj, self.pred, self.obj)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "subj": self.subj,
            "subj_def": self.subj_def,
            "pred": self.pred,
            "obj": self.obj,
            "obj_def": self.obj_def,
            "spo_def": self.spo_def,
        }
        if self.obj_kind != "entity" or self.obj_datatype or self.obj_lang:
            d["obj_kind"] = self.obj_kind
            d["obj_datatype"] = self.obj_datatype
            d["obj_lang"] = self.obj_lang
        return d


@dataclass(slots=True, frozen=True)
class RelationTraversalPolicy:
    """Relation-aware directed Markov transition multipliers for Personalized PageRank."""

    passage: float = 1.0
    fact_forward: float = 1.0
    fact_reverse: float = 1.0
    skos_related: float = 1.0
    skos_broader_up: float = 1.0
    skos_broader_down: float = 0.5
    owl_different_from: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class OpenIERecord:
    """NER entities and relation triples extracted from a single chunk."""

    chunk_id: str
    entities: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]
    structured_triples: tuple[StructuredTriple, ...] = ()
    entity_definitions: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True, frozen=True)
class SHACLViolation:
    """Closed-world SHACL validation report item on the SQLite knowledge graph."""

    shape: str
    focus_node: str
    result_path: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """Retrieved top-k hydrated passages and optional generated answer."""

    question: str
    passages: tuple[HydratedPassage, ...]
    answer: str | None = None


@dataclass(slots=True, frozen=True)
class IndexStats:
    """Cached non-blocking graph index statistics and configuration fingerprint."""

    chunks: int = 0
    entities: int = 0
    facts: int = 0
    edges: int = 0
    config_fingerprint: str = ""
    openie_checkpoints: int = 0
    entity_labels: int = 0
    shacl_conforms: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DaemonStatus:
    """Cached non-blocking daemon lifecycle, provider health, and worker queue snapshot."""

    lifecycle: str = "stopped"
    provider_ready: bool = False
    sync_state: str = "idle"
    queue_depth: int = 0
    active_op: str | None = None
    last_error: str | None = None
    last_sync_at: float | None = None
    pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute the MD5 hash of the given content string and optionally prepend a prefix."""
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


@overload
def text_processing(text: list[Any]) -> list[Any]: ...


@overload
def text_processing(text: Any) -> str: ...


def text_processing(text: Any) -> str | list[Any]:
    """Normalize entity/fact text to lowercase tokens while preserving significant technical punctuation."""
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    folded = text.casefold()
    n = len(folded)
    out: list[str] = []
    for i, ch in enumerate(folded):
        if (
            ch.isalnum()
            or ch.isspace()
            or (ch in "+#%" and i > 0 and (folded[i - 1].isalnum() or (out and out[-1] in "+#%")))
            or (ch in "./:" and 0 < i < n - 1 and folded[i - 1].isalnum() and folded[i + 1].isalnum())
            or (
                ch == "-"
                and 0 < i < n - 1
                and folded[i - 1].isalnum()
                and folded[i + 1].isalnum()
                and (folded[i - 1].isdigit() or folded[i + 1].isdigit())
            )
        ):
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def is_resource_label(label: str) -> bool:
    """A NER resource label has lexical content: an ASCII letter or any non-ASCII character.

    Mirrors the GBNF NER `label` rule (GBNF has no Unicode categories, so every non-ASCII
    character counts). Punctuation, numbers, dates and other data values are literals.
    """
    return any(ch.isalpha() or not ch.isascii() for ch in label)


def is_scalar_label(label: str) -> bool:
    """Whether a label's normalized form is an XSD scalar literal.

    The single downstream classification of NER labels: normalization, materialization and
    evidence binding must agree on which labels bind as resources (native graph spec §5.3).
    """
    return detect_scalar_literal(text_processing(label))[0]


def clean_surface_label(raw: str) -> str:
    """Strip whitespace and stray leading/trailing clause delimiters (:;,)."""
    return raw.strip().strip(":;,").strip()


def is_irreflexive_statement(
    subj: str,
    pred: str,
    obj: str,
    subj_def: str = "",
    obj_def: str = "",
    obj_kind: str = "entity",
) -> bool:
    """Return True iff the statement does not relate a resource sense to itself or repeat subj as a literal."""
    s_norm = str(text_processing(subj))
    o_norm = str(text_processing(obj))
    if not s_norm or not o_norm:
        return False
    if obj_kind == "literal":
        if s_norm == o_norm:
            return False
        p_norm = str(text_processing(pred))
        sd_norm = str(text_processing(subj_def))
        return not (p_norm and sd_norm and p_norm == sd_norm)
    return not (
        s_norm == o_norm
        and (not subj_def.strip() or not obj_def.strip() or subj_def.strip().casefold() == obj_def.strip().casefold())
    )


def compute_chunk_id(rel_posix_path: str, start_line: int, end_line: int, text: str) -> str:
    """Compute deterministic canonical chunk ID from relative path, line span, and slice text."""
    payload = f"{rel_posix_path}:{start_line}:{end_line}:{text}"
    return compute_mdhash_id(payload, prefix="chunk-")


def inclusive_end_line(start_line: int, end_line: int, end_column: int) -> int:
    """Convert half-open coordinate (end_line, end_column) to 1-indexed inclusive end line."""
    if end_column == 1 and end_line > start_line:
        return end_line - 1
    return end_line


def resolve_corpus_path(corpus_root: Path, rel_posix_path: str) -> Path:
    """Resolve a relative POSIX path against a corpus root, rejecting path traversals."""
    root_resolved = corpus_root.resolve()
    target_path = (root_resolved / rel_posix_path).resolve()
    if not target_path.is_relative_to(root_resolved):
        raise ValueError(f"Path traversal detected: {rel_posix_path!r} resolves outside corpus root {corpus_root}")
    return target_path


_XSD_BOOLEAN_RE = re.compile(r"^(?:true|false)$", re.IGNORECASE)
_XSD_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_XSD_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+$")
_XSD_DATETIME_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$",
    re.IGNORECASE,
)
_XSD_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
_XSD_TIME_RE = re.compile(
    r"^(?:(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?|(?:1[0-2]|0?[1-9]):[0-5]\d(?::[0-5]\d)?\s*(?:am|pm))$",
    re.IGNORECASE,
)
_XSD_DURATION_RE = re.compile(
    r"^-?P(?=\d|T\d)(?:(?:\d+Y)?(?:\d+M)?(?:\d+D)?)(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?$",
    re.IGNORECASE,
)
_CURRENCY_PREFIX_RE = re.compile(
    r"^[\$€£¥]\s*\d+(?:\.\d+)?(?:\s*[kKmMbBtT]|(?:\s*(?:thousand|million|billion|trillion)))?$",
    re.IGNORECASE,
)
_CURRENCY_SUFFIX_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|cents?|dollars?)$", re.IGNORECASE)
_DURATION_LATENCY_RE = re.compile(
    r"^(?:p\d+\s+)?\d+(?:\.\d+)?\s*(?:ns|µs|us|ms|s|sec|seconds?|min|minutes?|h|hours?|days?)$",
    re.IGNORECASE,
)
_PERCENT_STORAGE_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:%|MB|GB|KB|TB|PB|bytes?)$", re.IGNORECASE)
_QUANTIFIED_COUNT_RE = re.compile(r"^\d+\s+(?:events?|requests?|tokens?|queries|pods?|nodes?)$", re.IGNORECASE)


def detect_scalar_literal(text: str, definition: str = "") -> tuple[bool, str]:
    """Classify clear-cut scalar literals into W3C XSD datatypes; default rdfs:Literal fallback is xsd:string."""
    _ = definition
    if not isinstance(text, str):
        return (False, "xsd:string")
    val = text.strip()
    if not val:
        return (False, "xsd:string")
    if _XSD_BOOLEAN_RE.match(val):
        return (True, "xsd:boolean")
    if _XSD_INTEGER_RE.match(val):
        return (True, "xsd:integer")
    if _XSD_DECIMAL_RE.match(val):
        return (True, "xsd:decimal")
    if _XSD_DATETIME_RE.match(val):
        return (True, "xsd:dateTime")
    if _XSD_DATE_RE.match(val):
        return (True, "xsd:date")
    if _XSD_TIME_RE.match(val):
        return (True, "xsd:time")
    if _XSD_DURATION_RE.match(val) or _DURATION_LATENCY_RE.match(val):
        return (True, "xsd:duration")
    if (
        _CURRENCY_PREFIX_RE.match(val)
        or _CURRENCY_SUFFIX_RE.match(val)
        or _PERCENT_STORAGE_RE.match(val)
        or _QUANTIFIED_COUNT_RE.match(val)
    ):
        return (True, "xsd:string")
    return (False, "xsd:string")


def pool_confidences(*signals: float | None, eps: float = 1e-6) -> float | None:
    """Logarithmic Opinion Pool (Geometric Mean) over non-None signals in [eps, 1 - eps], or None if all None.

    Invariant: pool_confidences(p, p) == p, pool_confidences(p, None) == p,
    pool_confidences(None, None) is None.
    """
    valid = [float(s) for s in signals if s is not None]
    if not valid:
        return None
    clamped = [min(1.0 - eps, max(eps, s)) for s in valid]
    log_sum = sum(math.log(s) for s in clamped)
    return float(math.exp(log_sum / len(clamped)))


def bayesian_noisy_or(confidences: Sequence[float | None], *, eps: float = 1e-6) -> float | None:
    """Compute 1 - prod(1 - c_i) over non-None observations in [eps, 1 - eps], or None if all None.

    Each observation contributes additive Shannon evidence: -log(1 - c_i).
    """
    valid = [float(c) for c in confidences if c is not None]
    if not valid:
        return None
    clamped = [min(1.0 - eps, max(eps, c)) for c in valid]
    prod_complement = 1.0
    for c in clamped:
        prod_complement *= 1.0 - c
    res = 1.0 - prod_complement
    return float(min(1.0 - eps, max(eps, res)))
