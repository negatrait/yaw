"""Frozen TOML configuration contract and loader for YAW (stdlib-only)."""

import contextlib
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from .types import YawRuntimeError

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
    included_patterns: tuple[str, ...] = (
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
    excluded_patterns: tuple[str, ...] = (".*", "*/.*", "__pycache__/*")
    json_keys_whitelist: tuple[str, ...] = ("*",)
    json_keys_blacklist: tuple[str, ...] = ()
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    poll_interval_seconds: float = 5.0

    # Inference settings
    base_url: str = "http://localhost:13305/api/v1"
    api_key: str = "lemonade"
    provider: str = "openai"
    embedding_model: str = "embeddinggemma-300m-GGUF"
    embedding_task_mode: str = "distinct-titled"
    extraction_model: str = "Qwen3-4B-Instruct-2507-GGUF"
    qa_model: str = "Qwen3-4B-Instruct-2507-GGUF"
    enable_gbnf: bool = True
    gbnf_max_entities: int | None = None
    gbnf_max_triples: int | None = None
    gbnf_max_facts: int = 4
    gbnf_max_reasoning_chars: int | None = None

    # Graph & PPR settings
    linking_top_k: int = 8
    retrieval_top_k: int = 5
    damping: float = 0.5
    synonymy_threshold: float = 0.8
    synonymy_top_k: int = 100
    embed_batch_size: int = 64
    max_context_chars: int = 24000
    max_ast_expansion_hops: int = 1

    # Runtime & HTTP transport settings
    queue_maxsize: int = 32
    http_host: str = "127.0.0.1"
    http_port: int = 13306
    http_path: str = "/mcp"

    # Prompt optimization settings ([optimization] in yaw.toml)
    optimize_datasets: tuple[str, ...] = ("2wikimultihopqa",)
    optimize_mode: Literal["miprov2", "gepa", "hybrid"] = "gepa"
    optimize_max_queries: int = 20
    optimize_train_queries: int = 7
    optimize_query_offset: int = 0
    optimize_auto: Literal["light", "medium", "heavy"] = "light"

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
        if self.optimize_mode not in ("miprov2", "gepa", "hybrid"):
            raise YawRuntimeError(
                "INVALID_INPUT", f"Unsupported optimize_mode: {self.optimize_mode!r}"
            )
        if self.optimize_auto not in ("light", "medium", "heavy"):
            raise YawRuntimeError(
                "INVALID_INPUT", f"Unsupported optimize_auto: {self.optimize_auto!r}"
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

    def for_optimization(self, datestamp: str | None = None) -> "YawConfig":
        """Derive an isolated configuration for prompt optimization using *_optimize_<datestamp>.* paths."""
        from datetime import UTC, datetime

        ds = datestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        base_db = Path(self.db_path) if self.db_path else Path(".yaw/yaw.db")
        base_dir = base_db.parent

        opt_db = base_dir / f"yaw_optimize_{ds}.db"
        opt_lock = base_dir / f"yaw_optimize_{ds}.db.lock"
        opt_status = base_dir / f"status_optimize_{ds}.json"
        opt_state = base_dir / f"coco_state_optimize_{ds}"
        opt_program = base_dir / f"compiled_program_optimize_{ds}.json"

        cfg_dict: dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in (
                "db_path",
                "lock_path",
                "status_path",
                "state_dir",
                "compiled_program_path",
            )
        }
        cfg_dict["db_path"] = opt_db
        cfg_dict["lock_path"] = opt_lock
        cfg_dict["status_path"] = opt_status
        cfg_dict["state_dir"] = opt_state
        cfg_dict["compiled_program_path"] = opt_program
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
    "max_context_chars": "graph",
    "max_ast_expansion_hops": "graph",
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
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
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
                mapped_k = (
                    _OPTIMIZATION_KEY_ALIASES.get(k, k)
                    if sec_name == "optimization"
                    else k
                )
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
    kwargs["workspace_root"] = (
        _resolve_path(base_dir, ws_raw, ".") if ws_raw and not workspace_override else base_dir
    )
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
            return (
                str(val) if isinstance(val, Path) else list(val) if isinstance(val, tuple) else val
            )

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
        _OPTIMIZATION_KEY_ALIASES.get(prop_name, prop_name)
        if section_name in (None, "optimization")
        else prop_name
    )
    resolved_section = section_name or _FIELD_TO_SECTION.get(field_name)
    if not resolved_section:
        raise YawRuntimeError("INVALID_INPUT", f"Cannot determine section for key: {key!r}")

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
    "YawConfig",
    "get_config_value",
    "load_config",
    "set_config_value",
]
