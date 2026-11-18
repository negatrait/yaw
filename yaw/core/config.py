"""Frozen TOML configuration contract and loader for YAW (stdlib-only)."""

import contextlib
import hashlib
import json
import math
import os
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin, get_type_hints

from .types import YawRuntimeError

DEFAULT_EMBEDDING_MODEL = "embeddinggemma-2"
DEFAULT_CONTEXT_WINDOW_TOKENS = 131072
DEFAULT_INCLUDED_PATTERNS: tuple[str, ...] = (
    "*.md",
    "*.mdx",
    "*.txt",
    "*.rst",
    "*.adoc",
    "*.org",
    "*.py",
    "*.pyi",
    "*.rs",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.mjs",
    "*.cjs",
    "*.go",
    "*.java",
    "*.kt",
    "*.kts",
    "*.c",
    "*.h",
    "*.cpp",
    "*.hpp",
    "*.cc",
    "*.cs",
    "*.rb",
    "*.php",
    "*.swift",
    "*.scala",
    "*.lua",
    "*.r",
    "*.R",
    "*.sh",
    "*.bash",
    "*.zsh",
    "*.sql",
    "*.graphql",
    "*.proto",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.json",
    "*.jsonl",
    "*.xml",
    "*.html",
    "*.css",
    "*.scss",
    "*.tex",
    "*.bib",
    "*.pas",
    "*.dpr",
    "*.f90",
    "*.sol",
    "*.dtd",
)
DEFAULT_EXCLUDED_PATTERNS: tuple[str, ...] = (".*", "*/.*", "__pycache__/*")


def write_via_temp_atomic(
    path: Path | str,
    write_fn: Callable[[Path], Any],
    *,
    retries: int = 6,
) -> None:
    """Write via a sibling temporary file and atomically replace target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    tmp = target.with_name(f".{target.stem}.{os.getpid()}.{time.time_ns()}{suffix}")
    max_attempts = max(1, int(retries))
    try:
        write_fn(tmp)
        for attempt in range(max_attempts):
            try:
                tmp.replace(target)
                return
            except PermissionError:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def write_text_atomic(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    retries: int = 6,
    newline: str | None = None,
) -> None:
    """Atomically write UTF-8 text via a sibling temporary file with Windows PermissionError backoff."""
    write_via_temp_atomic(
        path,
        lambda tmp: tmp.write_text(content, encoding=encoding, newline=newline),
        retries=retries,
    )


def parse_dataset_list(raw: str | Sequence[str] | None) -> list[str]:
    """Normalize a string or sequence of comma-separated dataset names."""
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [part.strip() for item in items for part in str(item).split(",") if part.strip()]


def resolve_coco_state_root(state_dir: Path | str) -> Path:
    """Return the single persistent CocoIndex LMDB directory (<state_dir>/coco_lmdb).

    Never append transient per-candidate SQLite filenames.
    """
    resolved = Path(state_dir).resolve()
    if resolved.name == "coco_lmdb":
        return resolved
    if (
        resolved.name.startswith("coco_state")
        or resolved.suffix in {".db", ".sqlite", ".sqlite3"}
        or resolved.is_file()
    ):
        return resolved.parent / "coco_lmdb"
    return resolved / "coco_lmdb"


def scale_context_budget(base_budget: int, context_tokens: int) -> int:
    """Scale a 64K deployment budget with the configured serving capacity."""
    if base_budget < 1 or context_tokens < 1:
        raise ValueError("Context capacity and base budget must be positive")
    return max(1, base_budget * context_tokens // 65536)


def context_char_limit(base_chars: int, context_tokens: int, output_tokens: int | None) -> int:
    """Scale conservative character safeguards by available input capacity.

    An omitted output limit reserves no fixed token allowance. This is a character
    heuristic, not a tokenizer or a guarantee that an entire prompt will fit.
    """
    if context_tokens < 2 or (output_tokens is not None and not 0 < output_tokens < context_tokens):
        raise ValueError("Require 0 < output_tokens < context_tokens")
    return max(1, base_chars * (context_tokens - (output_tokens or 0)) // (131072 - 8192))


_DEFAULT_RELATIVE_PATHS: dict[str, str] = {
    "corpus_root": "corpus",
    "db_path": ".yaw/yaw.db",
    "lock_path": ".yaw/yaw.db.lock",
    "state_dir": ".yaw/coco_lmdb",
    "log_path": ".yaw/yaw.log",
    "status_path": ".yaw/status.json",
    "compiled_program_path": ".yaw/compiled_program.json",
}


def _resolve_path(base_dir: Path, raw: str | Path, default_rel: str) -> Path:
    val = Path(raw) if raw else Path(default_rel)
    return val.resolve() if val.is_absolute() else (base_dir / val).resolve()


@dataclass(slots=True, frozen=True)
class RetrievalParams:
    """5 dimensionless parameters governing hybrid zero-LLM chunk retrieval."""

    peclet: float = 0.3375533823900754
    alpha_sprt: float = 0.07953934952508675
    redundancy_curvature: float = 0.834659004285133
    coalition_coupling: float = 0.3887505167779042
    external_trust: float = 0.26842185514853145

    def __post_init__(self) -> None:
        bounds = {"peclet": (0.0, 1.0, False, True), "alpha_sprt": (0.0, 0.5, False, False)}
        for f_name in self.__dataclass_fields__:
            raw = getattr(self, f_name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{f_name} must be an int or float")
            if not math.isfinite(val := float(raw)):
                raise ValueError(f"{f_name} must be finite")
            lo, hi, inc_lo, inc_hi = bounds.get(
                f_name,
                (0.0, 2.0 if f_name == "redundancy_curvature" else 1.0, True, True),
            )
            if not ((val >= lo if inc_lo else val > lo) and (val <= hi if inc_hi else val < hi)):
                raise ValueError(f"{f_name} out of domain [{lo}, {hi}]: {val}")
            object.__setattr__(self, f_name, val)

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None) -> Self:
        """Build validated RetrievalParams from a mapping or key-value sequence."""
        cleaned = dict(overrides.items() if isinstance(overrides, Mapping) else (overrides or ()))
        if bad := [k for k in cleaned if k not in cls.__dataclass_fields__]:
            raise ValueError(f"Unknown RetrievalParams field: {bad[0]!r}")
        return replace(cls(), **cleaned)


@dataclass(slots=True, frozen=True)
class YawConfig:
    """Immutable daemon, ingestion, inference, and graph configuration loaded from yaw.toml."""

    workspace_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    corpus_root: Path | None = None
    db_path: Path | None = None
    lock_path: Path | None = None
    state_dir: Path | None = None
    log_path: Path | None = None
    status_path: Path | None = None
    compiled_program_path: Path | None = None

    # Ingestion settings
    included_patterns: tuple[str, ...] = DEFAULT_INCLUDED_PATTERNS
    excluded_patterns: tuple[str, ...] = DEFAULT_EXCLUDED_PATTERNS
    json_keys_whitelist: tuple[str, ...] = ("*",)
    json_keys_blacklist: tuple[str, ...] = ()
    chunk_size: int = 1200
    chunk_overlap: int = 120
    min_chunk_size: int = 100
    poll_interval_seconds: float = 10.0

    # Inference settings
    base_url: str = "http://127.0.0.1:8888/v1"
    api_key: str = field(
        default_factory=lambda: os.environ.get("UNSLOTH_STUDIO_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
    )
    provider: str = "openai"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_task_mode: str = "distinct-titled"
    extraction_model: str = "gemma-4-e4b"
    qa_model: str = "gemma-4-e4b"
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    max_output_tokens: int | None = None
    react_max_iters: int = 6
    rlm_max_iters: int = 20
    rlm_max_llm_calls: int = 50
    rlm_max_output_chars: int | None = None
    reflection_input_chars: int = 65536
    enable_gbnf: bool = True
    gbnf_max_entities: int = 8
    gbnf_max_triples: int = 8
    gbnf_max_facts: int = 8
    gbnf_max_reasoning_chars: int | None = None

    # Graph and HMM settings
    linking_top_k: int = 8
    retrieval_top_k: int = 10
    damping: float = 0.5
    retriever_mode: Literal["graph", "manifold"] = "graph"

    candidate_dense_threshold: float = 0.85
    candidate_lexical_threshold: float = 0.675
    auto_merge_threshold: float = 0.98
    candidate_volume_ratio: float = 0.5
    semantic_adjudication_batch_size: int = 5
    embed_batch_size: int = 64
    embed_max_in_flight: int = 2
    max_context_chars: int | None = None
    max_ast_expansion_hops: int = 1
    ast_context_hops: int = 1

    # Runtime & HTTP transport settings
    queue_maxsize: int = 32
    http_host: str = "127.0.0.1"
    http_port: int = 8889
    http_path: str = "/mcp"

    # Prompt optimization settings ([optimization] in yaw.toml)
    optimize_datasets: tuple[str, ...] = ("2wikimultihopqa",)
    optimize_mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid"
    optimize_max_queries: int = 35
    optimize_train_queries: int = 30
    optimize_query_offset: int = 0
    optimize_auto: Literal["light", "medium", "heavy"] = "light"
    optimize_sleep_seconds: float = 0.0
    optimize_max_bootstrapped_demos: int = 1
    optimize_max_labeled_demos: int = 1
    retrieval: RetrievalParams = field(default_factory=RetrievalParams)
    manifold_overrides: tuple[tuple[str, int | float], ...] = ()

    def __post_init__(self) -> None:
        ws = Path(self.workspace_root).resolve()
        object.__setattr__(self, "workspace_root", ws)
        for attr, default_rel in _DEFAULT_RELATIVE_PATHS.items():
            object.__setattr__(
                self,
                attr,
                _resolve_path(ws, getattr(self, attr) or "", default_rel),
            )
        for int_field in _INT_FIELDS:
            raw_int = getattr(self, int_field)
            if raw_int is None:
                if int_field not in _OPTIONAL_INT_FIELDS and int_field not in (
                    "optimize_max_bootstrapped_demos",
                    "optimize_max_labeled_demos",
                ):
                    raise YawRuntimeError(
                        "INVALID_INPUT",
                        f"Configuration field {int_field!r} requires an integer, got None",
                    )
                continue
            if isinstance(raw_int, bool) or not isinstance(raw_int, int):
                raise YawRuntimeError(
                    "INVALID_INPUT",
                    f"Configuration field {int_field!r} requires an integer, got {raw_int!r}",
                )
        if not all(
            math.isfinite(value) for value in (self.candidate_dense_threshold, self.candidate_lexical_threshold)
        ):
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): Candidate admission thresholds must be finite")
        if not 0 < self.auto_merge_threshold <= 1:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): auto_merge_threshold must be in (0, 1]")
        if not 0 < self.candidate_volume_ratio < float("inf"):
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): candidate_volume_ratio must be finite and positive")
        if self.queue_maxsize < 1:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): queue_maxsize must be >= 1")
        if self.chunk_size < 1:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): chunk_size must be >= 1")
        for name in ("embed_batch_size", "embed_max_in_flight"):
            if getattr(self, name) < 1:
                raise YawRuntimeError("INVALID_INPUT", f"YawConfig(): {name} must be >= 1")
        if self.context_window_tokens < 2:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): context_window_tokens must be >= 2")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): max_output_tokens must be positive")
        for name in (
            "react_max_iters",
            "rlm_max_iters",
            "rlm_max_llm_calls",
            "reflection_input_chars",
        ):
            value = getattr(self, name)
            if value is None or value < 1:
                raise YawRuntimeError("INVALID_INPUT", f"YawConfig(): {name} must be positive")
        if self.rlm_max_output_chars is not None and self.rlm_max_output_chars < 1:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): rlm_max_output_chars must be positive")
        if self.max_context_chars is not None and self.max_context_chars < 1:
            raise YawRuntimeError("INVALID_INPUT", "YawConfig(): max_context_chars must be positive")
        if self.optimize_mode not in ("miprov2", "gepa", "hybrid"):
            raise YawRuntimeError("INVALID_INPUT", f"YawConfig(): Unsupported optimize_mode: {self.optimize_mode!r}")
        if self.retriever_mode not in ("graph", "manifold"):
            raise YawRuntimeError("INVALID_INPUT", f"YawConfig(): Unsupported retriever_mode: {self.retriever_mode!r}")
        if self.optimize_auto not in ("light", "medium", "heavy"):
            raise YawRuntimeError("INVALID_INPUT", f"YawConfig(): Unsupported optimize_auto: {self.optimize_auto!r}")
        raw_sleep = self.optimize_sleep_seconds
        if raw_sleep is None:
            sleep_seconds = 0.0
        elif isinstance(raw_sleep, bool) or not isinstance(raw_sleep, (int, float)):
            raise YawRuntimeError(
                "INVALID_INPUT",
                f"YawConfig(): optimize_sleep_seconds must be a non-negative finite number, got {raw_sleep!r}",
            )
        else:
            sleep_seconds = float(raw_sleep)
            if not math.isfinite(sleep_seconds) or sleep_seconds < 0.0:
                raise YawRuntimeError(
                    "INVALID_INPUT",
                    f"YawConfig(): optimize_sleep_seconds must be a non-negative finite number, got {raw_sleep!r}",
                )
        object.__setattr__(self, "optimize_sleep_seconds", sleep_seconds)
        for demo_field in ("optimize_max_bootstrapped_demos", "optimize_max_labeled_demos"):
            raw_demo = getattr(self, demo_field)
            if raw_demo is None:
                demo_val = 1
            elif isinstance(raw_demo, bool) or not isinstance(raw_demo, int) or raw_demo < 0:
                raise YawRuntimeError(
                    "INVALID_INPUT",
                    f"YawConfig(): {demo_field} must be a non-negative integer, got {raw_demo!r}",
                )
            else:
                demo_val = raw_demo
            object.__setattr__(self, demo_field, demo_val)
        raw_ds = self.optimize_datasets
        if isinstance(raw_ds, str):
            ds_tuple = tuple(p.strip() for p in raw_ds.split(",") if p.strip())
        elif isinstance(raw_ds, (list, tuple)):
            ds_tuple = tuple(str(p).strip() for p in raw_ds if str(p).strip())
        else:
            ds_tuple = ("2wikimultihopqa",)
        object.__setattr__(self, "optimize_datasets", ds_tuple or ("2wikimultihopqa",))

        for j_attr, j_default in (
            ("json_keys_whitelist", ("*",)),
            ("json_keys_blacklist", ()),
        ):
            raw_j = getattr(self, j_attr)
            if isinstance(raw_j, str):
                j_tuple = tuple(p.strip() for p in raw_j.split(",") if p.strip())
            elif isinstance(raw_j, (list, tuple)):
                j_tuple = tuple(str(p).strip() for p in raw_j if str(p).strip())
            else:
                j_tuple = j_default
            object.__setattr__(
                self,
                j_attr,
                j_tuple if j_attr == "json_keys_blacklist" else (j_tuple or j_default),
            )
        object.__setattr__(self, "manifold_overrides", _normalize_manifold_overrides(self.manifold_overrides))

    def resolve_inference_budgets(self, context_tokens: int) -> YawConfig:
        """Bind serving capacity and tool budgets; leave omitted generation limits to the endpoint."""
        if context_tokens < 2:
            raise YawRuntimeError("INVALID_INPUT", "resolve_inference_budgets(): context_window_tokens must be >= 2")
        rlm_chars = (
            self.rlm_max_output_chars
            if self.rlm_max_output_chars is not None
            else scale_context_budget(10000, context_tokens)
        )
        if self.max_output_tokens is not None and self.max_output_tokens >= context_tokens:
            raise YawRuntimeError(
                "INVALID_INPUT", "resolve_inference_budgets(): Require 0 < max_output_tokens < context_window_tokens"
            )
        return replace(self, context_window_tokens=context_tokens, rlm_max_output_chars=rlm_chars)

    def for_optimization(
        self,
        db_override: Path | str | None = None,
    ) -> YawConfig:
        """Derive an isolated configuration for prompt optimization."""
        base_db = Path(self.db_path) if self.db_path else Path(".yaw/yaw.db")
        base_dir = base_db.parent

        if db_override is not None:
            raw_path = str(db_override).strip()
            if not raw_path:
                raise YawRuntimeError(
                    "INVALID_INPUT", "for_optimization(): Optimization database path cannot be empty."
                )
            opt_db = _resolve_path(self.workspace_root, raw_path, raw_path)
            base_dir = opt_db.parent
        else:
            opt_db = base_dir / "optimize_yaw.db"

        suffix = opt_db.stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
        return replace(
            self,
            corpus_root=self.workspace_root,
            db_path=opt_db,
            lock_path=Path(str(opt_db) + ".lock"),
            status_path=base_dir / (f"optimize_status_{suffix}.json" if suffix else "optimize_status.json"),
            state_dir=base_dir / "coco_lmdb",
            compiled_program_path=base_dir / (f"optimize_program_{suffix}.json" if suffix else "optimize_program.json"),
            poll_interval_seconds=0.0,
        )

    def for_benchmark(
        self,
        dataset: str,
        slice_hash: str,
        *,
        index_root: Path | None = None,
        fresh: bool = False,
    ) -> YawConfig:
        """Derive an isolated sandbox configuration for per-dataset benchmark runs."""
        import re
        import shutil

        safe_ds = re.sub(r"[^a-zA-Z0-9_.-]+", "_", dataset.strip().lower()).strip("_") or "dataset"
        safe_hash = re.sub(r"[^a-zA-Z0-9_.-]+", "_", slice_hash.strip().lower()).strip("_") or "default"
        ws = Path(self.workspace_root).resolve() if self.workspace_root else Path.cwd().resolve()
        root_dir = Path(index_root).resolve() if index_root is not None else (ws / "outputs" / "benchmarks" / "indexes")
        suffix = "_manifold" if self.retriever_mode == "manifold" else ""
        sandbox_dir = root_dir / f"{safe_ds}_{safe_hash}{suffix}"

        if fresh and sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        return replace(
            self,
            workspace_root=sandbox_dir,
            corpus_root=sandbox_dir,
            db_path=sandbox_dir / "yaw.db",
            lock_path=sandbox_dir / "yaw.db.lock",
            status_path=sandbox_dir / "status.json",
            state_dir=sandbox_dir / "coco_state",
            log_path=sandbox_dir / "yaw.log",
            poll_interval_seconds=0.0,
        )

    def config_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of index-affecting settings."""
        payload = {
            "corpus_root": Path(self.corpus_root).as_posix() if self.corpus_root else "",
            "included_patterns": list(self.included_patterns),
            "excluded_patterns": list(self.excluded_patterns),
            "json_keys_whitelist": list(self.json_keys_whitelist),
            "json_keys_blacklist": list(self.json_keys_blacklist),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "embedding_model": self.embedding_model,
            "embedding_task_mode": self.embedding_task_mode,
            "extraction_model": self.extraction_model,
            "qa_model": self.qa_model,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "rlm_max_iters": self.rlm_max_iters,
            "rlm_max_llm_calls": self.rlm_max_llm_calls,
            "rlm_max_output_chars": self.rlm_max_output_chars,
            "reflection_input_chars": self.reflection_input_chars,
            "max_context_chars": self.max_context_chars,
            "candidate_dense_threshold": self.candidate_dense_threshold,
            "candidate_lexical_threshold": self.candidate_lexical_threshold,
            "auto_merge_threshold": self.auto_merge_threshold,
            "candidate_volume_ratio": self.candidate_volume_ratio,
            "semantic_adjudication_batch_size": self.semantic_adjudication_batch_size,
            "damping": self.damping,
            "retriever_mode": self.retriever_mode,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return "yaw-cfg-v1:" + hashlib.sha256(raw).hexdigest()[:16]


_OPTIMIZATION_KEY_ALIASES: dict[str, str] = {
    "datasets": "optimize_datasets",
    "mode": "optimize_mode",
    "max_queries": "optimize_max_queries",
    "train_queries": "optimize_train_queries",
    "query_offset": "optimize_query_offset",
    "auto": "optimize_auto",
    "sleep_seconds": "optimize_sleep_seconds",
    "sleep": "optimize_sleep_seconds",
    "max_bootstrapped_demos": "optimize_max_bootstrapped_demos",
    "max_labeled_demos": "optimize_max_labeled_demos",
}

_FIELD_TO_SECTION: dict[str, str] = {
    "corpus_root": "paths",
    "db_path": "paths",
    "lock_path": "paths",
    "state_dir": "paths",
    "log_path": "paths",
    "status_path": "paths",
    "compiled_program_path": "paths",
    "included_patterns": "ingestion",
    "excluded_patterns": "ingestion",
    "json_keys_whitelist": "ingestion",
    "json_keys_blacklist": "ingestion",
    "chunk_size": "ingestion",
    "chunk_overlap": "ingestion",
    "min_chunk_size": "ingestion",
    "poll_interval_seconds": "ingestion",
    "base_url": "inference",
    "api_key": "inference",
    "provider": "inference",
    "embedding_model": "inference",
    "embedding_task_mode": "inference",
    "extraction_model": "inference",
    "qa_model": "inference",
    "context_window_tokens": "inference",
    "max_output_tokens": "inference",
    "rlm_max_iters": "inference",
    "rlm_max_llm_calls": "inference",
    "rlm_max_output_chars": "inference",
    "react_max_iters": "inference",
    "reflection_input_chars": "inference",
    "enable_gbnf": "inference",
    "gbnf_max_entities": "inference",
    "gbnf_max_triples": "inference",
    "gbnf_max_facts": "inference",
    "gbnf_max_reasoning_chars": "inference",
    "linking_top_k": "graph",
    "retriever_mode": "retrieval",
    "retrieval_top_k": "graph",
    "damping": "graph",
    "candidate_dense_threshold": "graph",
    "candidate_lexical_threshold": "graph",
    "auto_merge_threshold": "graph",
    "candidate_volume_ratio": "graph",
    "semantic_adjudication_batch_size": "graph",
    "embed_batch_size": "ingestion",
    "embed_max_in_flight": "inference",
    "max_context_chars": "graph",
    "max_ast_expansion_hops": "graph",
    "ast_context_hops": "graph",
    "queue_maxsize": "runtime",
    "http_host": "runtime",
    "http_port": "runtime",
    "http_path": "runtime",
    "optimize_datasets": "optimization",
    "optimize_mode": "optimization",
    "optimize_max_queries": "optimization",
    "optimize_train_queries": "optimization",
    "optimize_query_offset": "optimization",
    "optimize_auto": "optimization",
    "optimize_sleep_seconds": "optimization",
    "optimize_max_bootstrapped_demos": "optimization",
    "optimize_max_labeled_demos": "optimization",
}


def _yaw_config_field_kinds() -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
]:
    hints = get_type_hints(YawConfig)
    bool_fields: set[str] = set()
    seq_fields: set[str] = set()
    str_path_fields: set[str] = set()
    opt_int_fields: set[str] = set()
    int_fields: set[str] = set()
    float_fields: set[str] = set()
    for f in fields(YawConfig):
        ann = hints.get(f.name, f.type)
        origin = get_origin(ann)
        args = get_args(ann)
        if ann is bool:
            bool_fields.add(f.name)
        elif origin is tuple and args == (str, ...):
            seq_fields.add(f.name)
        elif ann in (str, Path) or origin is Literal or (origin is Union and any(a in (str, Path) for a in args)):
            str_path_fields.add(f.name)
        elif ann is int or (origin is Union and int in args and float not in args):
            int_fields.add(f.name)
            if origin is Union and type(None) in args:
                opt_int_fields.add(f.name)
        elif ann is float or (origin is Union and float in args):
            float_fields.add(f.name)
    return (
        frozenset(bool_fields),
        frozenset(seq_fields),
        frozenset(str_path_fields),
        frozenset(opt_int_fields),
        frozenset(int_fields),
        frozenset(float_fields),
    )


_BOOL_FIELDS, _SEQUENCE_FIELDS, _STRING_PATH_FIELDS, _OPTIONAL_INT_FIELDS, _INT_FIELDS, _FLOAT_FIELDS = (
    _yaw_config_field_kinds()
)


def _strict_toml_value(val_str: str, cleaned: str) -> Any:
    try:
        parsed = tomllib.loads(f"v = {cleaned}")
    except tomllib.TOMLDecodeError as exc:
        raise YawRuntimeError("INVALID_INPUT", f"_strict_toml_value(): Invalid TOML scalar value: {val_str!r}") from exc
    if len(parsed) != 1:
        raise YawRuntimeError("INVALID_INPUT", f"_strict_toml_value(): Invalid TOML scalar value: {val_str!r}")
    return parsed["v"]


def _parse_config_scalar(val_str: str, field_name: str | None = None) -> Any:
    cleaned = val_str.strip()
    if field_name in _BOOL_FIELDS:
        low = cleaned.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        raise YawRuntimeError(
            "INVALID_INPUT", f"_parse_config_scalar(): Invalid boolean value for {field_name}: {val_str!r}"
        )

    if field_name in _SEQUENCE_FIELDS:
        if not cleaned:
            return ""
        if cleaned[0] in ("[", '"', "'"):
            return _strict_toml_value(val_str, cleaned)
        return cleaned

    if field_name in _STRING_PATH_FIELDS:
        if len(cleaned) >= 2 and (
            (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'"))
        ):
            return _strict_toml_value(val_str, cleaned)
        return cleaned

    if field_name in _INT_FIELDS or field_name in _FLOAT_FIELDS or field_name is None:
        if field_name in _OPTIONAL_INT_FIELDS and cleaned.lower() in ("none", "null"):
            return None
        try:
            parsed = tomllib.loads(f"v = {cleaned}")
        except tomllib.TOMLDecodeError as exc:
            if field_name is None:
                raise ValueError(f"Invalid numeric scalar value: {val_str!r}") from exc
            raise YawRuntimeError(
                "INVALID_INPUT", f"_parse_config_scalar(): Invalid TOML scalar value for {field_name}: {val_str!r}"
            ) from exc
        if len(parsed) != 1:
            if field_name is None:
                raise ValueError(f"Invalid numeric scalar value: {val_str!r}")
            raise YawRuntimeError(
                "INVALID_INPUT", f"_parse_config_scalar(): Invalid TOML scalar value for {field_name}: {val_str!r}"
            )
        val = parsed["v"]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            if field_name is None:
                raise ValueError(f"_parse_config_scalar(): Expected numeric value, got {type(val).__name__}")
            raise YawRuntimeError(
                "INVALID_INPUT", f"_parse_config_scalar(): Expected numeric value for {field_name}, got {val!r}"
            )
        if field_name in _INT_FIELDS and not isinstance(val, int):
            raise YawRuntimeError(
                "INVALID_INPUT",
                f"_parse_config_scalar(): Configuration field {field_name!r} requires an integer, got {val!r}",
            )
        return val

    raise YawRuntimeError(
        "INVALID_INPUT", f"_parse_config_scalar(): Unsupported scalar configuration field: {field_name!r}"
    )


def _format_toml_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple)):
        items = [_format_toml_scalar(x) for x in val]
        return f"[{', '.join(items)}]"
    return json.dumps(str(val))


def _normalize_manifold_overrides(
    raw: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
) -> tuple[tuple[str, int | float], ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            f"Manifold overrides must be a mapping or sequence of (key, value) pairs, got {type(raw).__name__}"
        )
    items = raw.items() if isinstance(raw, Mapping) else raw
    normalized: dict[str, int | float] = {}
    for pair in items:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Manifold override entry must be a (key, value) pair, got {pair!r}")
        key, val = pair
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"Manifold override key must be a valid identifier string, got {key!r}")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"Manifold override {key!r} must be int or float (not bool), got {type(val).__name__}")
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f"Manifold override {key!r} must be finite, got {val!r}")
        normalized[key] = val
    return tuple(normalized.items())


def _extract_toml_section_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[["):
        return None
    close_idx = stripped.find("]")
    if close_idx == -1:
        return None
    trailing = stripped[close_idx + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        return None
    section = stripped[1:close_idx].strip()
    return section or None


def load_config(
    path: Path | str | None = None,
    *,
    workspace_override: Path | str | None = None,
) -> YawConfig:
    """Load and validate a YawConfig from a TOML file (defaults to ./yaw.toml if present)."""
    base_dir = Path(workspace_override).resolve() if workspace_override else Path.cwd().resolve()
    toml_file: Path | None = None

    if path is not None:
        toml_file = Path(path).resolve()
        if not toml_file.exists():
            raise YawRuntimeError("INVALID_INPUT", f"load_config(): Configuration file not found: {toml_file}")
        if workspace_override is None:
            base_dir = toml_file.parent
    elif (base_dir / "yaw.toml").exists():
        toml_file = base_dir / "yaw.toml"

    data: dict[str, Any] = {}
    if toml_file is not None:
        try:
            data = tomllib.loads(toml_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise YawRuntimeError("INVALID_INPUT", f"load_config(): Invalid TOML in {toml_file}: {exc}") from exc

    valid_fields = {f.name for f in fields(YawConfig)}
    retrieval_fields = frozenset(RetrievalParams.__dataclass_fields__)
    retrieval_raw: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    for sec_name in ("paths", "ingestion", "inference", "graph", "retrieval", "runtime", "optimization"):
        section = data.get(sec_name)
        if sec_name == "retrieval" and section is not None and not isinstance(section, dict):
            raise ValueError(f"[retrieval] section in {toml_file} must be a TOML table, got {type(section).__name__}")
        if isinstance(section, dict):
            for k, v in section.items():
                if sec_name == "retrieval" and k in retrieval_fields:
                    retrieval_raw[k] = v
                    continue
                if sec_name == "graph" and k in {"synonymy_threshold", "synonymy_top_k"}:
                    raise YawRuntimeError("INVALID_INPUT", f"load_config(): Retired candidate setting: graph.{k}")
                if k == "embed_batch_chars":
                    raise YawRuntimeError(
                        "INVALID_INPUT",
                        "load_config(): embed_batch_chars is retired; use ingestion.embed_batch_size "
                        "and inference.embed_max_in_flight",
                    )
                if k in {"embed_batch_size", "embed_max_in_flight"} and sec_name != _FIELD_TO_SECTION[k]:
                    raise YawRuntimeError(
                        "INVALID_INPUT", f"load_config(): {k} belongs in [{_FIELD_TO_SECTION[k]}], not [{sec_name}]"
                    )
                mapped_k = _OPTIMIZATION_KEY_ALIASES.get(k, k) if sec_name == "optimization" else k
                if mapped_k in valid_fields:
                    if mapped_k == "optimize_datasets":
                        kwargs[mapped_k] = tuple(
                            parse_dataset_list(v if isinstance(v, (str, list, tuple)) else [str(v)])
                        )
                    else:
                        kwargs[mapped_k] = tuple(v) if isinstance(v, list) else v

    if retrieval_raw:
        kwargs["retrieval"] = RetrievalParams.from_overrides(retrieval_raw)

    if "manifold" in data:
        manifold_raw = data["manifold"]
        if not isinstance(manifold_raw, dict):
            raise ValueError(
                f"[manifold] section in {toml_file} must be a TOML table, got {type(manifold_raw).__name__}"
            )
        kwargs["manifold_overrides"] = _normalize_manifold_overrides(manifold_raw)

    ws_raw = kwargs.pop("workspace_root", None)
    kwargs["workspace_root"] = _resolve_path(base_dir, ws_raw, ".") if ws_raw and not workspace_override else base_dir
    return YawConfig(**kwargs)


def get_config_value(
    path: Path | str | None = None,
    key: str | None = None,
    *,
    workspace_override: Path | str | None = None,
) -> Any:
    """Read a specific key or all configuration values from yaw.toml with resolved YawConfig defaults."""
    cfg = load_config(path, workspace_override=workspace_override)
    if key is None:
        out: dict[str, Any] = {}
        for f in fields(cfg):
            val = getattr(cfg, f.name)
            if isinstance(val, RetrievalParams):
                val = asdict(val)
            elif isinstance(val, Path):
                val = str(val)
            elif isinstance(val, tuple):
                val = list(val)
            out[f.name] = val
        return out

    cleaned_key = key.strip()
    if cleaned_key == "retrieval":
        return asdict(cfg.retrieval)
    if cleaned_key.startswith("retrieval."):
        subkey = cleaned_key.split(".", 1)[1].strip()
        if subkey in RetrievalParams.__dataclass_fields__:
            return getattr(cfg.retrieval, subkey)
    if cleaned_key == "manifold":
        return dict(cfg.manifold_overrides)
    if cleaned_key.startswith("manifold."):
        subkey = cleaned_key.split(".", 1)[1].strip()
        overrides_map = dict(cfg.manifold_overrides)
        if subkey not in overrides_map:
            raise KeyError(f"Unset manifold configuration key: {cleaned_key!r}")
        return overrides_map[subkey]

    if hasattr(cfg, cleaned_key):
        val = getattr(cfg, cleaned_key)
        return str(val) if isinstance(val, Path) else list(val) if isinstance(val, tuple) else val

    mapped = _OPTIMIZATION_KEY_ALIASES.get(cleaned_key)
    if mapped and hasattr(cfg, mapped):
        val = getattr(cfg, mapped)
        return str(val) if isinstance(val, Path) else list(val) if isinstance(val, tuple) else val

    if "." in cleaned_key:
        _, subkey = cleaned_key.split(".", 1)
        sub_mapped = _OPTIMIZATION_KEY_ALIASES.get(subkey, subkey)
        if hasattr(cfg, sub_mapped):
            val = getattr(cfg, sub_mapped)
            return str(val) if isinstance(val, Path) else list(val) if isinstance(val, tuple) else val

    raise YawRuntimeError("INVALID_INPUT", f"get_config_value(): Unknown configuration key: {key!r}")


def set_config_value(
    key: str,
    raw_value: str,
    path: Path | str | None = None,
    *,
    workspace_override: Path | str | None = None,
) -> tuple[str, Any]:
    """Safely update a configuration key in yaw.toml, verifying validity before atomic commit."""
    base_dir = Path(workspace_override).resolve() if workspace_override else Path.cwd().resolve()
    toml_path = Path(path).resolve() if path else base_dir / "yaw.toml"
    if not toml_path.exists():
        raise YawRuntimeError("INVALID_INPUT", f"set_config_value(): Configuration file not found: {toml_path}")

    cleaned_key = key.strip()
    section_name: str | None = None
    prop_name: str = cleaned_key

    if "." in cleaned_key:
        section_name, prop_name = cleaned_key.split(".", 1)
        section_name = section_name.strip()
        prop_name = prop_name.strip()

    if section_name == "retrieval" and prop_name in RetrievalParams.__dataclass_fields__:
        raw_parsed = _parse_config_scalar(raw_value, None)
        validated = RetrievalParams.from_overrides({prop_name: raw_parsed})
        parsed_val = getattr(validated, prop_name)
        resolved_section = "retrieval"
        target_prop = prop_name
    elif section_name == "manifold":
        if not prop_name:
            raise ValueError(f"Invalid manifold configuration key: {key!r}")
        raw_parsed = _parse_config_scalar(raw_value, None)
        validated_key, parsed_val = _normalize_manifold_overrides([(prop_name, raw_parsed)])[0]
        resolved_section = "manifold"
        target_prop = validated_key
    else:
        field_name = (
            _OPTIMIZATION_KEY_ALIASES.get(prop_name, prop_name) if section_name in (None, "optimization") else prop_name
        )
        resolved_section = section_name or _FIELD_TO_SECTION.get(field_name)
        if not resolved_section:
            raise YawRuntimeError("INVALID_INPUT", f"set_config_value(): Cannot determine section for key: {key!r}")
        if field_name not in {field.name for field in fields(YawConfig)}:
            raise YawRuntimeError("INVALID_INPUT", f"set_config_value(): Unknown configuration key: {key!r}")
        parsed_val = _parse_config_scalar(raw_value, field_name)
        target_prop = prop_name.removeprefix("optimize_") if prop_name.startswith("optimize_") else prop_name

    formatted_val = None if parsed_val is None else _format_toml_scalar(parsed_val)

    lines = toml_path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    current_sec: str | None = None
    key_written = False

    for line in lines:
        sec_hdr = _extract_toml_section_name(line)
        if sec_hdr is not None:
            if formatted_val is not None and current_sec == resolved_section and not key_written:
                new_lines.append(f"{target_prop} = {formatted_val}")
                key_written = True
            current_sec = sec_hdr
            new_lines.append(line)
            continue

        if current_sec == resolved_section:
            parts = line.strip().split("=", 1)
            if len(parts) == 2 and parts[0].strip() in (target_prop, prop_name):
                if formatted_val is not None:
                    new_lines.append(f"{target_prop} = {formatted_val}")
                    key_written = True
                continue

        new_lines.append(line)

    if formatted_val is not None and not key_written:
        if current_sec == resolved_section:
            new_lines.append(f"{target_prop} = {formatted_val}")
        else:
            new_lines.append(f"\n[{resolved_section}]")
            new_lines.append(f"{target_prop} = {formatted_val}")

    updated_text = "\n".join(new_lines) + "\n"

    def _write_validated(tmp: Path) -> None:
        tmp.write_text(updated_text, encoding="utf-8", newline="\n")
        load_config(tmp, workspace_override=workspace_override)

    write_via_temp_atomic(toml_path, _write_validated)

    return f"{resolved_section}.{target_prop}", parsed_val


def write_manifold_section(
    overrides: Mapping[str, int | float] | Sequence[tuple[str, int | float]],
    path: Path | str | None = None,
    *,
    workspace_override: Path | str | None = None,
) -> Path:
    """Atomically replace or append the [manifold] section in yaw.toml after validating via load_config."""
    base_dir = Path(workspace_override).resolve() if workspace_override else Path.cwd().resolve()
    toml_path = Path(path).resolve() if path else base_dir / "yaw.toml"
    if not toml_path.exists():
        raise YawRuntimeError("INVALID_INPUT", f"write_manifold_section(): Configuration file not found: {toml_path}")

    normalized = _normalize_manifold_overrides(overrides)
    section_lines = ["[manifold]", *(f"{k} = {_format_toml_scalar(v)}" for k, v in normalized)]

    existing_lines = toml_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    in_manifold = False
    replaced = False

    for line in existing_lines:
        sec_hdr = _extract_toml_section_name(line)
        if sec_hdr is not None:
            if in_manifold:
                in_manifold = False
            if sec_hdr == "manifold":
                if not replaced:
                    out_lines.extend(section_lines)
                    out_lines.append("")
                    replaced = True
                in_manifold = True
                continue
        if not in_manifold:
            out_lines.append(line)

    if not replaced:
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()
        if out_lines:
            out_lines.append("")
        out_lines.extend(section_lines)

    updated_text = "\n".join(out_lines).rstrip("\n") + "\n"

    def _write_validated(tmp: Path) -> None:
        tmp.write_text(updated_text, encoding="utf-8", newline="\n")
        load_config(tmp, workspace_override=workspace_override)

    write_via_temp_atomic(toml_path, _write_validated)
    return toml_path


def write_retrieval_section(
    overrides: Mapping[str, int | float] | Sequence[tuple[str, int | float]],
    path: Path | str | None = None,
    *,
    workspace_override: Path | str | None = None,
) -> Path:
    """Atomically replace or append the [retrieval] section in yaw.toml after validating RetrievalParams."""
    base_dir = Path(workspace_override).resolve() if workspace_override else Path.cwd().resolve()
    toml_path = Path(path).resolve() if path else base_dir / "yaw.toml"
    if not toml_path.exists():
        raise YawRuntimeError("INVALID_INPUT", f"write_retrieval_section(): Configuration file not found: {toml_path}")

    validated = RetrievalParams.from_overrides(overrides)
    section_lines = [
        "[retrieval]",
        *(f"{k} = {_format_toml_scalar(getattr(validated, k))}" for k in validated.__dataclass_fields__),
    ]

    existing_lines = toml_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    in_section = False
    replaced = False

    for line in existing_lines:
        sec_hdr = _extract_toml_section_name(line)
        if sec_hdr is not None:
            in_section = False
            if sec_hdr == "retrieval":
                if not replaced:
                    out_lines.extend(section_lines)
                    out_lines.append("")
                    replaced = True
                in_section = True
                continue
        if not in_section:
            out_lines.append(line)

    if not replaced:
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()
        if out_lines:
            out_lines.append("")
        out_lines.extend(section_lines)

    updated_text = "\n".join(out_lines).rstrip("\n") + "\n"

    def _write_validated(tmp: Path) -> None:
        tmp.write_text(updated_text, encoding="utf-8", newline="\n")
        load_config(tmp, workspace_override=workspace_override)

    write_via_temp_atomic(toml_path, _write_validated)
    return toml_path


__all__ = [
    "DEFAULT_EXCLUDED_PATTERNS",
    "DEFAULT_INCLUDED_PATTERNS",
    "RetrievalParams",
    "YawConfig",
    "get_config_value",
    "load_config",
    "parse_dataset_list",
    "resolve_coco_state_root",
    "set_config_value",
    "write_manifold_section",
    "write_retrieval_section",
    "write_text_atomic",
    "write_via_temp_atomic",
]
