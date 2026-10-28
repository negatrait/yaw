"""Frozen TOML configuration contract and loader for YAW (stdlib-only)."""

import contextlib
import hashlib
import json
import os
import time
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from .types import YawRuntimeError

DEFAULT_EMBEDDING_MODEL = "embeddinggemma-300m-qat-q8_0-GGUF-Q8_0"
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


def write_text_atomic(path: Path | str, content: str, *, retries: int = 6, newline: str | None = None) -> None:
    """Atomically write UTF-8 text via a sibling temporary file with Windows PermissionError backoff."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}_{time.time_ns()}.tmp")
    max_attempts = max(1, int(retries))
    try:
        tmp.write_text(content, encoding="utf-8", newline=newline)
        for attempt in range(max_attempts):
            try:
                os.replace(tmp, target)
                return
            except PermissionError:
                if attempt == max_attempts - 1:
                    target.write_text(content, encoding="utf-8")
                    return
                time.sleep(0.01 * (2**attempt))
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def scale_context_budget(base_budget: int, context_tokens: int) -> int:
    """Scale a 64K deployment budget with the configured serving capacity."""
    if base_budget < 1 or context_tokens < 1:
        raise ValueError("Context capacity and base budget must be positive")
    return max(1, base_budget * context_tokens // 65536)


def context_char_limit(base_chars: int, context_tokens: int, output_tokens: int) -> int:
    """Scale conservative character safeguards by available input capacity.

    Preserve the established 32K-context/4K-output limits. This is a character
    heuristic, not a tokenizer or a guarantee that an entire prompt will fit.
    """
    if not 0 < output_tokens < context_tokens:
        raise ValueError("Require 0 < output_tokens < context_tokens")
    return max(1, base_chars * (context_tokens - output_tokens) // (32768 - 4096))


_DEFAULT_RELATIVE_PATHS: dict[str, str] = {
    "corpus_root": "corpus",
    "db_path": ".yaw/yaw.db",
    "lock_path": ".yaw/yaw.db.lock",
    "state_dir": ".yaw/coco_state",
    "log_path": ".yaw/yaw.log",
    "status_path": ".yaw/status.json",
    "compiled_program_path": ".yaw/compiled_program.json",
}


def _resolve_path(base_dir: Path, raw: str | Path, default_rel: str) -> Path:
    val = Path(raw) if raw else Path(default_rel)
    return val.resolve() if val.is_absolute() else (base_dir / val).resolve()


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
    base_url: str = "http://localhost:13305/api/v1"
    api_key: str = "lemonade"
    provider: str = "openai"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_task_mode: str = "distinct-titled"
    extraction_model: str = "gemma-4-E4B-it-qat-q4_0-gguf-Q4_0"
    qa_model: str = "gemma-4-E4B-it-qat-q4_0-gguf-Q4_0"
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    max_output_tokens: int | None = None
    rlm_max_iters: int = 10
    rlm_max_llm_calls: int = 20
    rlm_max_output_chars: int | None = None
    reflection_input_tokens: int | None = None
    reflection_tokenizer_url: str = ""
    enable_gbnf: bool = True
    gbnf_max_entities: int | None = None
    gbnf_max_triples: int | None = None
    gbnf_max_facts: int = 4
    gbnf_max_reasoning_chars: int | None = None

    # Graph & PPR settings
    linking_top_k: int = 8
    retrieval_top_k: int = 10
    damping: float = 0.5

    synonymy_threshold: float = 0.8
    synonymy_top_k: int = 100
    embed_batch_size: int = 64
    embed_batch_chars: int = 3500
    max_context_chars: int | None = None
    max_ast_expansion_hops: int = 1
    ast_context_hops: int = 0

    # Runtime & HTTP transport settings
    queue_maxsize: int = 32
    indexing_overlap: bool = False
    http_host: str = "127.0.0.1"
    http_port: int = 13306
    http_path: str = "/mcp"

    # Prompt optimization settings ([optimization] in yaw.toml)
    optimize_datasets: tuple[str, ...] = ("2wikimultihopqa",)
    optimize_mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid"
    optimize_max_queries: int = 20
    optimize_train_queries: int = 7
    optimize_query_offset: int = 0
    optimize_auto: Literal["light", "medium", "heavy"] = "light"
    optimize_sleep_seconds: float = 0.0
    optimize_max_bootstrapped_demos: int = 0
    optimize_max_labeled_demos: int = 0

    def __post_init__(self) -> None:
        ws = Path(self.workspace_root).resolve()
        object.__setattr__(self, "workspace_root", ws)
        for attr, default_rel in _DEFAULT_RELATIVE_PATHS.items():
            object.__setattr__(
                self,
                attr,
                _resolve_path(ws, getattr(self, attr) or "", default_rel),
            )
        if self.queue_maxsize < 1:
            raise YawRuntimeError("INVALID_INPUT", "queue_maxsize must be >= 1")
        if self.chunk_size < 1:
            raise YawRuntimeError("INVALID_INPUT", "chunk_size must be >= 1")
        if self.context_window_tokens < 2:
            raise YawRuntimeError("INVALID_INPUT", "context_window_tokens must be >= 2")
        for name, base_budget in (
            ("max_output_tokens", 4096),
            ("rlm_max_output_chars", 10000),
            ("reflection_input_tokens", 16384),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, scale_context_budget(base_budget, self.context_window_tokens))
        output_tokens = self.max_output_tokens
        assert output_tokens is not None
        if not 0 < output_tokens < self.context_window_tokens:
            raise YawRuntimeError("INVALID_INPUT", "Require 0 < max_output_tokens < context_window_tokens")
        for name in (
            "rlm_max_iters",
            "rlm_max_llm_calls",
            "rlm_max_output_chars",
            "reflection_input_tokens",
        ):
            value = getattr(self, name)
            if value is None or value < 1:
                raise YawRuntimeError("INVALID_INPUT", f"{name} must be positive")
        context_chars = self.max_context_chars
        if context_chars is None:
            context_chars = context_char_limit(24000, self.context_window_tokens, output_tokens)
            object.__setattr__(self, "max_context_chars", context_chars)
        if context_chars < 1:
            raise YawRuntimeError("INVALID_INPUT", "max_context_chars must be positive")
        if self.optimize_mode not in ("miprov2", "gepa", "hybrid"):
            raise YawRuntimeError("INVALID_INPUT", f"Unsupported optimize_mode: {self.optimize_mode!r}")
        if self.optimize_auto not in ("light", "medium", "heavy"):
            raise YawRuntimeError("INVALID_INPUT", f"Unsupported optimize_auto: {self.optimize_auto!r}")
        object.__setattr__(
            self,
            "optimize_sleep_seconds",
            max(0.0, float(self.optimize_sleep_seconds or 0.0)),
        )
        object.__setattr__(
            self,
            "optimize_max_bootstrapped_demos",
            max(0, int(self.optimize_max_bootstrapped_demos or 0)),
        )
        object.__setattr__(
            self,
            "optimize_max_labeled_demos",
            max(0, int(self.optimize_max_labeled_demos or 0)),
        )
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

    def for_optimization(
        self,
        datestamp: str | None = None,
        db_override: Path | str | None = None,
    ) -> YawConfig:
        """Derive an isolated configuration for prompt optimization using *_optimize_<datestamp>.* paths."""
        import re
        from datetime import UTC, datetime

        base_db = Path(self.db_path) if self.db_path else Path(".yaw/yaw.db")
        base_dir = base_db.parent

        if db_override is not None:
            raw_path = str(db_override).strip()
            if raw_path.lower() == "latest":
                candidates = sorted(
                    base_dir.glob("yaw_optimize_*.db"),
                    # Run creation time is encoded in the name. Writes to an older
                    # run must not make it newer than a subsequently created run.
                    key=lambda p: p.name,
                    reverse=True,
                )
                candidates = [
                    p for p in candidates if p.is_file() and re.fullmatch(r"yaw_optimize_\d{8}_\d{6}\.db", p.name)
                ]
                if not candidates:
                    raise YawRuntimeError("INVALID_INPUT", "No optimization run exists to resume.")
                opt_db = candidates[0]
            else:
                opt_db = _resolve_path(self.workspace_root, raw_path, raw_path)
                if not opt_db.is_file():
                    raise YawRuntimeError("INVALID_INPUT", f"Optimization database does not exist: {opt_db}")
            base_dir = opt_db.parent
            ds_match = re.search(r"yaw_optimize_(\d{8}_\d{6})", opt_db.name)
            ds = ds_match.group(1) if ds_match else opt_db.stem
        else:
            ds = datestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            opt_db = base_dir / f"yaw_optimize_{ds}.db"

        opt_lock = Path(str(opt_db) + ".lock")
        opt_status = base_dir / f"status_optimize_{ds}.json"
        opt_state = base_dir / f"coco_state_optimize_{ds}"
        opt_program = base_dir / f"compiled_program_optimize_{ds}.json"

        cfg_dict: dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in (
                "corpus_root",
                "db_path",
                "lock_path",
                "status_path",
                "state_dir",
                "compiled_program_path",
            )
        }
        cfg_dict["corpus_root"] = self.workspace_root
        cfg_dict["db_path"] = opt_db
        cfg_dict["lock_path"] = opt_lock
        cfg_dict["status_path"] = opt_status
        cfg_dict["state_dir"] = opt_state
        cfg_dict["compiled_program_path"] = opt_program
        cfg_dict["poll_interval_seconds"] = 0.0
        return YawConfig(**cfg_dict)

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
        sandbox_dir = root_dir / f"{safe_ds}_{safe_hash}"

        if fresh and sandbox_dir.exists():
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        bench_db = sandbox_dir / "yaw.db"
        bench_lock = sandbox_dir / "yaw.db.lock"
        bench_status = sandbox_dir / "status.json"
        bench_state = sandbox_dir / "coco_state"
        bench_log = sandbox_dir / "yaw.log"

        cfg_dict: dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name
            not in (
                "workspace_root",
                "corpus_root",
                "db_path",
                "lock_path",
                "status_path",
                "state_dir",
                "log_path",
            )
        }
        cfg_dict["workspace_root"] = sandbox_dir
        cfg_dict["corpus_root"] = sandbox_dir
        cfg_dict["db_path"] = bench_db
        cfg_dict["lock_path"] = bench_lock
        cfg_dict["status_path"] = bench_status
        cfg_dict["state_dir"] = bench_state
        cfg_dict["log_path"] = bench_log
        cfg_dict["poll_interval_seconds"] = 0.0
        return YawConfig(**cfg_dict)

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
            "reflection_input_tokens": self.reflection_input_tokens,
            "reflection_tokenizer_url": self.reflection_tokenizer_url,
            "max_context_chars": self.max_context_chars,
            "synonymy_threshold": self.synonymy_threshold,
            "damping": self.damping,
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
    "reflection_input_tokens": "inference",
    "reflection_tokenizer_url": "inference",
    "enable_gbnf": "inference",
    "gbnf_max_entities": "inference",
    "gbnf_max_triples": "inference",
    "gbnf_max_facts": "inference",
    "gbnf_max_reasoning_chars": "inference",
    "linking_top_k": "graph",
    "retrieval_top_k": "graph",
    "damping": "graph",
    "synonymy_threshold": "graph",
    "synonymy_top_k": "graph",
    "embed_batch_size": "graph",
    "embed_batch_chars": "graph",
    "max_context_chars": "graph",
    "max_ast_expansion_hops": "graph",
    "ast_context_hops": "graph",
    "queue_maxsize": "runtime",
    "indexing_overlap": "runtime",
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


def _parse_toml_scalar(val_str: str) -> Any:
    cleaned = val_str.strip()
    if cleaned.lower() == "true":
        return True
    if cleaned.lower() == "false":
        return False
    if cleaned.startswith("[") and cleaned.endswith("]"):
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(cleaned)
    with contextlib.suppress(ValueError):
        return int(cleaned)
    with contextlib.suppress(ValueError):
        return float(cleaned)
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        return cleaned[1:-1]
    return cleaned


def _format_toml_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple)):
        items = [_format_toml_scalar(x) for x in val]
        return f"[{', '.join(items)}]"
    return json.dumps(str(val))


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
            raise YawRuntimeError("INVALID_INPUT", f"Configuration file not found: {toml_file}")
        if workspace_override is None:
            base_dir = toml_file.parent
    elif (base_dir / "yaw.toml").exists():
        toml_file = base_dir / "yaw.toml"

    data: dict[str, Any] = {}
    if toml_file is not None:
        try:
            data = tomllib.loads(toml_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise YawRuntimeError("INVALID_INPUT", f"Invalid TOML in {toml_file}: {exc}") from exc

    valid_fields = {f.name for f in fields(YawConfig)}
    kwargs: dict[str, Any] = {}
    for sec_name in ("paths", "ingestion", "inference", "graph", "runtime", "optimization"):
        section = data.get(sec_name)
        if isinstance(section, dict):
            for k, v in section.items():
                mapped_k = _OPTIMIZATION_KEY_ALIASES.get(k, k) if sec_name == "optimization" else k
                if mapped_k in valid_fields:
                    if mapped_k == "optimize_datasets":
                        if isinstance(v, str):
                            kwargs[mapped_k] = tuple(p.strip() for p in v.split(",") if p.strip())
                        elif isinstance(v, (list, tuple)):
                            kwargs[mapped_k] = tuple(str(p).strip() for p in v if str(p).strip())
                        else:
                            kwargs[mapped_k] = (str(v),)
                    else:
                        kwargs[mapped_k] = tuple(v) if isinstance(v, list) else v

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
            if isinstance(val, Path):
                val = str(val)
            elif isinstance(val, tuple):
                val = list(val)
            out[f.name] = val
        return out

    cleaned_key = key.strip()
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

    raise YawRuntimeError("INVALID_INPUT", f"Unknown configuration key: {key!r}")


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
        raise YawRuntimeError("INVALID_INPUT", f"Configuration file not found: {toml_path}")

    cleaned_key = key.strip()
    section_name: str | None = None
    prop_name: str = cleaned_key

    if "." in cleaned_key:
        section_name, prop_name = cleaned_key.split(".", 1)
        section_name = section_name.strip()
        prop_name = prop_name.strip()

    field_name = (
        _OPTIMIZATION_KEY_ALIASES.get(prop_name, prop_name) if section_name in (None, "optimization") else prop_name
    )
    resolved_section = section_name or _FIELD_TO_SECTION.get(field_name)
    if not resolved_section:
        raise YawRuntimeError("INVALID_INPUT", f"Cannot determine section for key: {key!r}")
    if field_name not in {field.name for field in fields(YawConfig)}:
        raise YawRuntimeError("INVALID_INPUT", f"Unknown configuration key: {key!r}")

    parsed_val = _parse_toml_scalar(raw_value)
    formatted_val = _format_toml_scalar(parsed_val)

    lines = toml_path.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    current_sec: str | None = None
    key_written = False

    target_prop = prop_name
    if prop_name.startswith("optimize_"):
        target_prop = prop_name.removeprefix("optimize_")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_sec == resolved_section and not key_written:
                new_lines.append(f"{target_prop} = {formatted_val}")
                key_written = True
            current_sec = stripped[1:-1].strip()
            new_lines.append(line)
            continue

        if current_sec == resolved_section:
            parts = stripped.split("=", 1)
            if len(parts) == 2 and parts[0].strip() in (target_prop, prop_name):
                new_lines.append(f"{target_prop} = {formatted_val}")
                key_written = True
                continue

        new_lines.append(line)

    if not key_written:
        if current_sec == resolved_section:
            new_lines.append(f"{target_prop} = {formatted_val}")
        else:
            new_lines.append(f"\n[{resolved_section}]")
            new_lines.append(f"{target_prop} = {formatted_val}")

    updated_text = "\n".join(new_lines) + "\n"
    tmp_path = toml_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        tmp_path.write_text(updated_text, encoding="utf-8")
        # Validate that the modified file produces a valid YawConfig
        load_config(tmp_path, workspace_override=workspace_override)
        tmp_path.replace(toml_path)
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)

    return f"{resolved_section}.{target_prop}", parsed_val


__all__ = [
    "DEFAULT_EXCLUDED_PATTERNS",
    "DEFAULT_INCLUDED_PATTERNS",
    "YawConfig",
    "get_config_value",
    "load_config",
    "set_config_value",
    "write_text_atomic",
]
