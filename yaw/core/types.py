"""Frozen, zero-dependency data contracts and pure coordinate/ID helpers for YAW."""

import hashlib
import json
import re
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
        value = ["synonym-evidence-v1", self.payload, self.identifiers, self.candidates]
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def resolve_decisions(self, decisions: list[dict[str, str]]) -> list[dict[str, str]]:
        """Validate exact pair coverage and resolve request-local IDs before graph writes."""
        pairs = {p["pair_id"] for p in json.loads(self.payload)["pairs"]}
        seen: set[str] = set()
        result = []
        for decision in decisions:
            pair = decision.get("pair_id", "")
            if pair not in pairs or pair in seen:
                raise ValueError(f"Unknown or duplicate synonym pair {pair!r}")
            a, b = pair.split("|")
            canonical = decision.get("canonical", "")
            if canonical not in (a, b):
                raise ValueError(f"Invalid canonical ID {canonical!r} for synonym pair {pair!r}")
            result.append(
                {
                    "pair_id": f"{self.identifiers[a]}|{self.identifiers[b]}",
                    "relation": decision["relation"],
                    "canonical": self.identifiers[canonical],
                }
            )
            seen.add(pair)
        if seen != pairs:
            raise ValueError(f"Missing synonym decisions for {sorted(pairs - seen)}")
        return result


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
    """Versioned relation-aware directed traversal policy for Personalized PageRank."""

    passage: float = 1.0
    fact_forward: float = 1.0
    fact_reverse: float = 1.0
    datatype_property: float = 0.5
    skos_related: float = 1.0
    skos_broader_up: float = 1.0
    skos_broader_down: float = 0.5
    owl_different_from: float = 0.0
    parallel_aggregation: Literal["max", "sum", "capped_sum"] = "max"
    parallel_cap: float = 2.0

    @property
    def fact_fwd(self) -> float:
        return self.fact_forward

    @property
    def fact_rev(self) -> float:
        return self.fact_reverse

    @property
    def skos_broader(self) -> float:
        return self.skos_broader_up

    @property
    def skos_narrower(self) -> float:
        return self.skos_broader_down

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


def to_rel_posix_path(corpus_root: Path, file_path: Path | str) -> str:
    """Convert an absolute or relative file path to a relative POSIX path under corpus_root."""
    root_resolved = corpus_root.resolve()
    target_path = Path(file_path).resolve()
    if not target_path.is_relative_to(root_resolved):
        raise ValueError(f"File path {file_path!r} resolves outside corpus root {corpus_root}")
    return target_path.relative_to(root_resolved).as_posix()


_XSD_BOOLEAN_RE = re.compile(r"^(?:true|false)$", re.IGNORECASE)
_XSD_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_XSD_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+$")
_XSD_DATETIME_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$",
    re.IGNORECASE,
)
_XSD_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")


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
    return (False, "xsd:string")
