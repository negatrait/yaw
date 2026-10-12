"""Authoritative configuration loader using Python 3.14 tomllib."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from yaw.utils import normalize_token

for _k in ("no_proxy", "NO_PROXY"):
    if "127.0.0.1" not in os.environ.get(_k, ""):
        os.environ[_k] = (
            f"{os.environ[_k]},127.0.0.1,localhost"
            if _k in os.environ
            else "127.0.0.1,localhost"
        )


def _find_cfg_path() -> Path:
    env_cfg = os.environ.get("YAW_CONFIG")
    if env_cfg:
        return Path(env_cfg).resolve()
    for candidate in (
        Path.cwd() / "config.toml",
        Path(__file__).resolve().parent.parent.parent / "config.toml",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return (Path.cwd() / "config.toml").resolve()


CFG_PATH = _find_cfg_path()
raw: dict = {}

# Module-level attribute declarations for static analysis (ruff, ty)
VAULT_ROOT: Path = Path.cwd()
DB_PATH: Path = Path.cwd() / ".cache" / "yaw.db"
PID_FILE: Path = Path.cwd() / ".cache" / "yaw.pid"
HTTP_HOST: str = "127.0.0.1"
HTTP_PORT: int = 8765
MCP_URL: str = "http://127.0.0.1:8765/mcp"
BUSY_TIMEOUT_S: float = 60.0
CONNECT_TIMEOUT_S: float = 60.0
JOURNAL_MODE: str = "WAL"
SYNCHRONOUS: str = "NORMAL"
OLLAMA_HOST: str = "http://localhost:11434"
HOST: str = "http://localhost:11434"
HOST_PROVIDER: str = "ollama"
EMBEDDING_MODEL: str = "embed-gemma-300m-FLM"
CHAT_MODEL: str = "Qwen3-4B-Instruct-2507-GGUF"
SPACY_MODEL: str = "en_core_web_lg"
SPACY_CACHE_DIR: Path = Path(".cache/spacy_llm")
EXTRACTION_CONTEXT_TOKENS: int = 32768
EMBED_TIMEOUT_S: float = 300.0
CHAT_TIMEOUT_S: float = 300.0
EMBED_MAX_INFLIGHT: int = 4
TAXONOMY_NAMESPACES: tuple[str, ...] = (
    "entities",
    "concepts",
    "predicates",
    "invariants",
    "decisions",
    "claims",
    "processes",
    "sources",
    "contexts",
)
BASE_PREDICATES: tuple[str, ...] = (
    "partOf",
    "subClassOf",
    "manifestsIn",
    "mentions",
    "causes",
    "prevents",
    "supports",
    "refutes",
    "contradicts",
    "supersedes",
    "produces",
    "derivedFrom",
    "cites",
    "instanceOf",
)
INFER_ONTOLOGIES: tuple[str, ...] = ()
OPEN_IE: bool = True
TARGET_ONTOLOGY: str = "internal"
CANON_FEWSHOT_MAX: int = 5
USE_LLM_IF_NER_BELOW: int = 2
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 150
INDEXING_TIMEOUT_S: float = 60.0
INDEX_CONCURRENCY: int = 4
INCLUDED_PATTERNS: tuple[str, ...] = ("raw/**/*.md",)
EXCLUDED_PATTERNS: tuple[str, ...] = (".*/**", ".*")
FTS_TOP_N: int = 50
VEC_TOP_N: int = 50
RERANK_TOP_N: int = 10
RELEVANCE_THRESHOLD: float = 0.10
CANON_SIM_HIGH: float = 0.95
CANON_SIM_LOW: float = 0.80
CANON_SIM_THRESHOLD: float = 0.80
CANON_LEXICAL_MIN: float = 0.85
CANON_TOP_K: int = 5
CANON_LLM_CONFIDENCE: float = 0.80
LLM_DEFINE_TEMPERATURE: float = 0.1
DEFINITION_MAX_WORDS: int = 50
FACT_TOP_K: int = 5
FACT_SEED_LAMBDA: float = 0.5


def _get(sec: str, key: str, default, cast=None):
    v = os.environ.get(f"YAW_{key.upper()}", raw.get(sec, {}).get(key, default))
    if cast and v is not None:
        try:
            return cast(v)
        except ValueError, TypeError:
            return default
    return v


def _populate_paths(
    g: dict[str, Any],
    raw: dict[str, Any],
    vault_override: str | Path | None = None,
    port_override: int | None = None,
) -> None:
    """Populate filesystem paths, database location, server endpoints, and SQLite pragmas."""
    vr = (
        vault_override
        or os.environ.get("YAW_VAULT_ROOT")
        or raw.get("storage", {}).get("vault_root", "sample-llm-wiki")
    )
    cand = Path(vr)
    g["VAULT_ROOT"] = (
        cand.resolve() if cand.is_absolute() else (CFG_PATH.parent / cand).resolve()
    )

    dp = Path(
        os.environ.get(
            "YAW_DB_PATH", raw.get("storage", {}).get("db_path", ".cache/yaw.db")
        )
    )
    g["DB_PATH"] = (
        dp.resolve() if dp.is_absolute() else (g["VAULT_ROOT"] / dp).resolve()
    )
    g["PID_FILE"] = g["DB_PATH"].parent / "yaw.pid"

    g["HTTP_HOST"] = _get(
        "server", "host", raw.get("mcp", {}).get("host", "127.0.0.1"), str
    )
    g["HTTP_PORT"] = (
        port_override
        if port_override is not None
        else _get("server", "port", raw.get("mcp", {}).get("port", 8765), int)
    )
    g["MCP_URL"] = f"http://{g['HTTP_HOST']}:{g['HTTP_PORT']}/mcp"

    g["BUSY_TIMEOUT_S"] = _get("storage", "busy_timeout_s", 60.0, float)
    g["CONNECT_TIMEOUT_S"] = _get("storage", "connect_timeout_s", 60.0, float)
    g["JOURNAL_MODE"] = _get("storage", "journal_mode", "WAL", str)
    g["SYNCHRONOUS"] = _get("storage", "synchronous", "NORMAL", str)


def _populate_nlp(g: dict[str, Any], raw: dict[str, Any]) -> None:
    """Populate models, Ollama endpoints, spaCy configurations, and ontology settings."""
    g["OLLAMA_HOST"] = g["HOST"] = _get(
        "models", "host", "http://localhost:11434", str
    ).rstrip("/")
    g["HOST_PROVIDER"] = _get("models", "provider", "ollama", str)
    g["EMBEDDING_MODEL"] = _get(
        "models", "embedding_model", "embed-gemma-300m-FLM", str
    )
    g["CHAT_MODEL"] = _get("models", "chat_model", "Qwen3-4B-Instruct-2507-GGUF", str)
    g["SPACY_MODEL"] = _get(
        "nlp",
        "spacy_model",
        raw.get("models", {}).get("spacy_model", "en_core_web_lg"),
        str,
    )
    raw_spacy_cache = _get(
        "nlp",
        "cache_dir",
        raw.get("storage", {}).get("spacy_cache_dir", ".cache/spacy_llm"),
        str,
    )
    g["SPACY_CACHE_DIR"] = Path(raw_spacy_cache)
    g["EXTRACTION_CONTEXT_TOKENS"] = _get(
        "models", "extraction_context_tokens", 32768, int
    )
    g["EMBED_TIMEOUT_S"] = _get("models", "embed_timeout_s", 300.0, float)
    g["CHAT_TIMEOUT_S"] = _get("models", "chat_timeout_s", 300.0, float)
    g["EMBED_MAX_INFLIGHT"] = _get("models", "embed_max_inflight", 4, int)

    raw_ns = _get(
        "taxonomy",
        "namespaces",
        raw.get("ontology", {}).get("namespaces", TAXONOMY_NAMESPACES),
    )
    g["TAXONOMY_NAMESPACES"] = (
        tuple(normalize_token(str(x)) for x in raw_ns)
        if isinstance(raw_ns, (list, tuple))
        else ()
    )
    raw_bp = _get(
        "taxonomy",
        "base_predicates",
        raw.get("ontology", {}).get("base_predicates", BASE_PREDICATES),
    )
    g["BASE_PREDICATES"] = (
        tuple(str(x).strip() for x in raw_bp)
        if isinstance(raw_bp, (list, tuple)) and raw_bp
        else BASE_PREDICATES
    )
    raw_infer = _get(
        "ontology",
        "infer_ontologies",
        raw.get("ontology", {}).get("infer_ontologies", INFER_ONTOLOGIES),
    )
    g["INFER_ONTOLOGIES"] = (
        tuple(str(x).strip() for x in raw_infer)
        if isinstance(raw_infer, (list, tuple))
        else ()
    )
    g["OPEN_IE"] = _get(
        "ontology", "open_ie", raw.get("ontology", {}).get("open_ie", OPEN_IE), bool
    )
    g["TARGET_ONTOLOGY"] = str(
        _get(
            "ontology",
            "target_ontology",
            raw.get("ontology", {}).get("target_ontology", TARGET_ONTOLOGY),
        )
    ).lower()
    g["CANON_FEWSHOT_MAX"] = _get(
        "taxonomy",
        "canon_fewshot_max",
        raw.get("ontology", {}).get("canon_fewshot_max", 5),
        int,
    )
    g["USE_LLM_IF_NER_BELOW"] = _get("taxonomy", "use_llm_if_ner_below", 2, int)

    g["CANON_SIM_HIGH"] = _get("canonicalization", "sim_high", 0.95, float)
    g["CANON_SIM_LOW"] = _get("canonicalization", "sim_low", 0.80, float)
    g["CANON_SIM_THRESHOLD"] = _get(
        "canonicalization",
        "sim_threshold",
        raw.get("canonicalization", {}).get("sim_low", 0.80),
        float,
    )
    g["CANON_LEXICAL_MIN"] = _get("canonicalization", "lexical_min", 0.85, float)
    g["CANON_TOP_K"] = _get("canonicalization", "top_k", 5, int)
    g["CANON_LLM_CONFIDENCE"] = _get("canonicalization", "llm_confidence", 0.80, float)
    g["LLM_DEFINE_TEMPERATURE"] = _get(
        "canonicalization", "llm_temperature", 0.1, float
    )
    g["DEFINITION_MAX_WORDS"] = _get("taxonomy", "definition_max_words", 50, int)


def _populate(
    vault_override: str | Path | None = None, port_override: int | None = None
) -> None:
    global raw
    raw = tomllib.loads(CFG_PATH.read_text("utf-8")) if CFG_PATH.is_file() else {}
    g = globals()

    _populate_paths(g, raw, vault_override=vault_override, port_override=port_override)
    _populate_nlp(g, raw)

    g["CHUNK_SIZE"] = _get("indexing", "chunk_size", 1200, int)
    g["CHUNK_OVERLAP"] = _get("indexing", "chunk_overlap", 150, int)
    g["INDEXING_TIMEOUT_S"] = _get("indexing", "timeout_s", 60.0, float)
    g["INDEX_CONCURRENCY"] = _get("indexing", "concurrency", 4, int)
    raw_inc = _get("indexing", "included_patterns", ["raw/**/*.md"])
    g["INCLUDED_PATTERNS"] = (
        tuple(raw_inc) if isinstance(raw_inc, (list, tuple)) else (str(raw_inc),)
    )
    raw_exc = _get("indexing", "excluded_patterns", [".*/**", ".*"])
    g["EXCLUDED_PATTERNS"] = (
        tuple(raw_exc) if isinstance(raw_exc, (list, tuple)) else (str(raw_exc),)
    )

    g["FTS_TOP_N"] = _get("retrieval", "fts_top_n", 50, int)
    g["VEC_TOP_N"] = _get("retrieval", "vec_top_n", 50, int)
    g["RERANK_TOP_N"] = _get("retrieval", "rerank_top_n", 10, int)
    g["RELEVANCE_THRESHOLD"] = _get(
        "retrieval",
        "relevance_threshold",
        raw.get("reranker", {}).get("relevance_threshold", 0.10),
        float,
    )
    g["FACT_TOP_K"] = _get("retrieval", "fact_top_k", 5, int)
    g["FACT_SEED_LAMBDA"] = _get("retrieval", "fact_seed_lambda", 0.5, float)


def get_spacy_model_name(preferred: str | None = None) -> str:
    """Return the authoritative base spaCy model name per FOUNDATIONAL.md.

    FOUNDATIONAL.md mandates en_core_web_lg for static 300d GloVe vectors.
    Zero silent fallback to sm/md. If missing, raises RuntimeError.
    """
    target = preferred or globals().get("SPACY_MODEL", "en_core_web_lg")
    try:
        import spacy.util

        if spacy.util.is_package(target):
            return target
    except ImportError, AttributeError:
        pass

    raise RuntimeError(
        f"Authoritative spaCy model '{target}' is not installed. "
        "FOUNDATIONAL.md mandates en_core_web_lg with static GloVe vectors."
    )


def set_vault(p: str | Path) -> Path:
    r = Path(p).resolve()
    os.environ["YAW_VAULT_ROOT"] = str(r)
    _populate(vault_override=r)
    return r


def set_port(port: int) -> int:
    os.environ["YAW_PORT"] = str(port)
    _populate(port_override=port)
    return port


def reload() -> None:
    global CFG_PATH
    CFG_PATH = _find_cfg_path()
    _populate()


_populate()


__all__ = [
    "BASE_PREDICATES",
    "BUSY_TIMEOUT_S",
    "CANON_LEXICAL_MIN",
    "CANON_LLM_CONFIDENCE",
    "CANON_SIM_HIGH",
    "CANON_SIM_LOW",
    "CANON_SIM_THRESHOLD",
    "CANON_TOP_K",
    "CFG_PATH",
    "CHAT_MODEL",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "CONNECT_TIMEOUT_S",
    "DB_PATH",
    "DEFINITION_MAX_WORDS",
    "EMBEDDING_MODEL",
    "EXCLUDED_PATTERNS",
    "EXTRACTION_CONTEXT_TOKENS",
    "FTS_TOP_N",
    "HOST",
    "HOST_PROVIDER",
    "HTTP_HOST",
    "HTTP_PORT",
    "INCLUDED_PATTERNS",
    "INDEXING_TIMEOUT_S",
    "INDEX_CONCURRENCY",
    "INFER_ONTOLOGIES",
    "JOURNAL_MODE",
    "LLM_DEFINE_TEMPERATURE",
    "MCP_URL",
    "OLLAMA_HOST",
    "OPEN_IE",
    "PID_FILE",
    "RELEVANCE_THRESHOLD",
    "RERANK_TOP_N",
    "SPACY_CACHE_DIR",
    "SPACY_MODEL",
    "SYNCHRONOUS",
    "TARGET_ONTOLOGY",
    "TAXONOMY_NAMESPACES",
    "USE_LLM_IF_NER_BELOW",
    "VAULT_ROOT",
    "VEC_TOP_N",
    "get_spacy_model_name",
    "reload",
    "set_port",
    "set_vault",
]
