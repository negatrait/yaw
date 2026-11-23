"""Frozen TOML configuration contract and loader for YAW (stdlib-only)."""

import contextlib
import json
import os
import re
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from .types import YawRuntimeError

DEFAULT_EMBEDDING_MODEL = "embeddinggemma-2"
_INCLUDED_EXTS = (
    "md mdx txt rst adoc org py pyi rs ts tsx js jsx mjs cjs go java kt kts c h cpp hpp cc cs "
    "rb php swift scala lua r R sh bash zsh sql graphql proto yaml yml toml json jsonl xml html "
    "css scss tex bib pas dpr f90 sol dtd"
)
DEFAULT_INCLUDED_PATTERNS: tuple[str, ...] = tuple(f"*.{ext}" for ext in _INCLUDED_EXTS.split())
DEFAULT_EXCLUDED_PATTERNS: tuple[str, ...] = (".*", "*/.*", "__pycache__/*")

_TASK_MODES: frozenset[str] = frozenset({"search", "titled", "qa", "distinct", "distinct-titled", "fact"})
_AUTO_MODES: frozenset[str] = frozenset({"light", "medium", "heavy"})
_DEFAULT_RELATIVE_PATHS: dict[str, str] = {
    "corpus_root": "corpus",
    "db_path": ".yaw/yaw.db",
    "lock_path": ".yaw/yaw.db.lock",
    "state_dir": ".yaw/coco_lmdb",
    "log_path": ".yaw/yaw.log",
    "status_path": ".yaw/status.json",
    "compiled_program_path": ".yaw/compiled_program.json",
}
_OPTIMIZATION_KEY_ALIASES: dict[str, str] = {
    "datasets": "optimize_datasets",
    "max_queries": "optimize_max_queries",
    "query_offset": "optimize_query_offset",
    "auto": "optimize_auto",
}
_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "paths": tuple(_DEFAULT_RELATIVE_PATHS),
    "ingestion": ("included_patterns", "excluded_patterns", "chunk_size")
    + ("chunk_overlap", "min_chunk_size", "embed_batch_size"),
    "inference": ("base_url", "api_key", "provider", "embedding_model", "embedding_task_mode")
    + ("embed_max_in_flight", "qa_model", "max_output_tokens", "react_max_iters"),
    "runtime": ("queue_maxsize", "http_host", "http_port", "http_path"),
    "optimization": ("optimize_datasets", "optimize_max_queries", "optimize_query_offset", "optimize_auto"),
}
_ALLOWED_SECTIONS: frozenset[str] = frozenset(_SECTION_FIELDS)
_FIELD_TO_SECTION: dict[str, str] = {name: sec for sec, names in _SECTION_FIELDS.items() for name in names}


def _invalid(message: str) -> YawRuntimeError:
    return YawRuntimeError("INVALID_INPUT", message)


def write_via_temp_atomic(path: Path | str, write_fn: Callable[[Path], Any], *, retries: int = 6) -> None:
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
    path: Path | str, content: str, *, encoding: str = "utf-8", retries: int = 6, newline: str | None = None
) -> None:
    """Atomically write UTF-8 text via a sibling temporary file with Windows PermissionError backoff."""
    write_via_temp_atomic(
        path, lambda tmp: tmp.write_text(content, encoding=encoding, newline=newline), retries=retries
    )


def parse_dataset_list(raw: str | Sequence[str] | None) -> list[str]:
    """Normalize a string or sequence of comma-separated dataset names."""
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [part.strip() for item in items for part in str(item).split(",") if part.strip()]


def resolve_coco_state_root(state_dir: Path | str) -> Path:
    """Return the persistent CocoIndex LMDB directory (<state_dir>/coco_lmdb)."""
    resolved = Path(state_dir).resolve()
    if resolved.name == "coco_lmdb":
        return resolved
    if resolved.suffix in {".db", ".sqlite", ".sqlite3"} or resolved.is_file():
        return resolved.parent / "coco_lmdb"
    return resolved / "coco_lmdb"


def _resolve_path(base_dir: Path, raw: object, default_rel: str, *, field_name: str = "path") -> Path:
    if raw is None or (isinstance(raw, str) and not (raw := raw.strip())):
        val = Path(default_rel)
    elif isinstance(raw, (str, Path)):
        val = Path(raw)
    else:
        raise _invalid(f"Configuration field {field_name!r} must be a path string or Path, got {raw!r}")
    return val.resolve() if val.is_absolute() else (base_dir / val).resolve()


def _normalize_str_tuple(raw: object, default: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    if isinstance(raw, (list, tuple)):
        cleaned: list[str] = []
        for p in raw:
            if not isinstance(p, str):
                raise _invalid(f"Configuration field {field_name!r} elements must be strings, got {p!r}")
            if s := p.strip():
                cleaned.append(s)
        return tuple(cleaned)
    raise _invalid(
        f"Configuration field {field_name!r} must be a list of strings or comma-separated string, got {raw!r}"
    )


@dataclass(slots=True, frozen=True)
class YawConfig:
    """Immutable daemon, ingestion, inference, runtime, and optimization configuration."""

    workspace_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    corpus_root: Path | None = None
    db_path: Path | None = None
    lock_path: Path | None = None
    state_dir: Path | None = None
    log_path: Path | None = None
    status_path: Path | None = None
    compiled_program_path: Path | None = None

    # Ingestion settings ([ingestion])
    included_patterns: tuple[str, ...] = DEFAULT_INCLUDED_PATTERNS
    excluded_patterns: tuple[str, ...] = DEFAULT_EXCLUDED_PATTERNS
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_size: int = 100
    embed_batch_size: int = 5

    # Inference settings ([inference])
    base_url: str = "http://127.0.0.1:8888/v1"
    api_key: str = field(
        default_factory=lambda: os.environ.get("UNSLOTH_STUDIO_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
    )
    provider: str = "openai"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_task_mode: str = "distinct-titled"
    embed_max_in_flight: int = 2
    qa_model: str = "gemma-4-e4b"
    max_output_tokens: int | None = None
    react_max_iters: int = 6

    # Runtime & HTTP transport settings ([runtime])
    queue_maxsize: int = 32
    http_host: str = "127.0.0.1"
    http_port: int = 13306
    http_path: str = "/mcp"

    # Prompt optimization settings ([optimization])
    optimize_datasets: tuple[str, ...] = ("enterprise_rag_bench",)
    optimize_max_queries: int = 162
    optimize_query_offset: int = 0
    optimize_auto: Literal["light", "medium", "heavy"] = "light"

    def __post_init__(self) -> None:
        raw_ws: Any = self.workspace_root
        if (
            isinstance(raw_ws, bool)
            or not isinstance(raw_ws, (str, Path))
            or (isinstance(raw_ws, str) and not raw_ws.strip())
        ):
            raise _invalid(
                f"Configuration field 'workspace_root' must be a non-empty path string or Path, got {raw_ws!r}"
            )
        ws = Path(raw_ws).resolve()
        object.__setattr__(self, "workspace_root", ws)
        for attr, default_rel in _DEFAULT_RELATIVE_PATHS.items():
            object.__setattr__(self, attr, _resolve_path(ws, getattr(self, attr), default_rel, field_name=attr))

        for int_field in _INT_FIELDS:
            raw_int = getattr(self, int_field)
            if raw_int is None:
                if int_field not in _OPTIONAL_INT_FIELDS:
                    raise _invalid(f"Configuration field {int_field!r} requires an integer, got None")
                continue
            if isinstance(raw_int, bool) or not isinstance(raw_int, int):
                raise _invalid(f"Configuration field {int_field!r} requires an integer, got {raw_int!r}")
            if int_field not in ("chunk_overlap", "optimize_query_offset", "max_output_tokens") and raw_int < 1:
                raise _invalid(f"YawConfig(): {int_field} must be >= 1")

        for str_field in _STR_FIELDS:
            if not isinstance(raw_str := getattr(self, str_field), str):
                raise _invalid(f"Configuration field {str_field!r} must be a string, got {raw_str!r}")

        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise _invalid("YawConfig(): chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise _invalid("YawConfig(): max_output_tokens must be positive")
        if not (1 <= self.http_port <= 65535):
            raise _invalid(f"YawConfig(): http_port must be in 1..65535, got {self.http_port!r}")
        if not self.http_path.startswith("/"):
            raise _invalid(f"YawConfig(): http_path must start with '/', got {self.http_path!r}")
        if self.optimize_query_offset < 0:
            raise _invalid("YawConfig(): optimize_query_offset must be >= 0")
        if self.optimize_auto not in _AUTO_MODES:
            raise _invalid(f"YawConfig(): Unsupported optimize_auto: {self.optimize_auto!r}")
        if self.embedding_task_mode not in _TASK_MODES:
            raise _invalid(f"YawConfig(): Unsupported embedding_task_mode: {self.embedding_task_mode!r}")

        for attr, default in (
            ("optimize_datasets", ("enterprise_rag_bench",)),
            ("included_patterns", DEFAULT_INCLUDED_PATTERNS),
            ("excluded_patterns", DEFAULT_EXCLUDED_PATTERNS),
        ):
            object.__setattr__(self, attr, _normalize_str_tuple(getattr(self, attr), default, field_name=attr))
        if not self.optimize_datasets:
            raise _invalid("YawConfig(): optimize_datasets must not be empty")

    def for_optimization(self, db_override: Path | str | None = None) -> YawConfig:
        """Derive an isolated configuration for DSPy prompt optimization."""
        base_dir = (Path(self.db_path) if self.db_path else Path(".yaw/yaw.db")).parent
        if db_override is None:
            opt_db, status_name, prog_name = base_dir / "optimize_yaw.db", "dspy_status.json", "qa_program.json"
        else:
            if not (raw_path := str(db_override).strip()):
                raise _invalid("for_optimization(): Optimization database path cannot be empty.")
            opt_db = _resolve_path(self.workspace_root, raw_path, raw_path, field_name="db_override")
            base_dir = opt_db.parent
            suffix = opt_db.stem.removeprefix("optimize_yaw").removeprefix("yaw_optimize").lstrip("_")
            status_name = f"optimize_status_{suffix}.json" if suffix else "dspy_status.json"
            prog_name = f"optimize_program_{suffix}.json" if suffix else "qa_program.json"
        return replace(
            self,
            corpus_root=self.workspace_root,
            db_path=opt_db,
            lock_path=Path(f"{opt_db}.lock"),
            status_path=base_dir / status_name,
            state_dir=base_dir / "coco_lmdb",
            compiled_program_path=base_dir / prog_name,
        )


_SEQUENCE_FIELDS: frozenset[str] = frozenset({"included_patterns", "excluded_patterns", "optimize_datasets"})
_OPTIONAL_INT_FIELDS: frozenset[str] = frozenset({"max_output_tokens"})
_INT_FIELDS: frozenset[str] = (
    _OPTIONAL_INT_FIELDS
    | {"chunk_size", "chunk_overlap", "min_chunk_size", "embed_batch_size", "embed_max_in_flight"}
    | {"react_max_iters", "queue_maxsize", "http_port", "optimize_max_queries", "optimize_query_offset"}
)
_STRING_PATH_FIELDS: frozenset[str] = (
    (frozenset(_FIELD_TO_SECTION) | {"workspace_root"}) - _SEQUENCE_FIELDS - _INT_FIELDS
)
_STR_FIELDS: frozenset[str] = _STRING_PATH_FIELDS - _DEFAULT_RELATIVE_PATHS.keys() - {"workspace_root"}


def _strict_toml_value(val_str: str, cleaned: str) -> Any:
    try:
        parsed = tomllib.loads(f"v = {cleaned}")
    except tomllib.TOMLDecodeError as exc:
        raise _invalid(f"_strict_toml_value(): Invalid TOML scalar value: {val_str!r}") from exc
    if len(parsed) != 1:
        raise _invalid(f"_strict_toml_value(): Invalid TOML scalar value: {val_str!r}")
    return parsed["v"]


def _parse_config_scalar(val_str: str, field_name: str) -> Any:
    cleaned = val_str.strip()
    if field_name in _SEQUENCE_FIELDS:
        if not cleaned or cleaned[0] not in ("[", '"', "'"):
            return cleaned
        val = _strict_toml_value(val_str, cleaned)
        if not isinstance(val, str) and not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
            raise _invalid(f"_parse_config_scalar(): {field_name!r} requires a string or list of strings, got {val!r}")
        return val
    if field_name in _STRING_PATH_FIELDS:
        if len(cleaned) >= 2 and (cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'")):
            return _strict_toml_value(val_str, cleaned)
        return cleaned
    if field_name in _INT_FIELDS:
        if field_name in _OPTIONAL_INT_FIELDS and cleaned.lower() in ("none", "null"):
            return None
        val = _strict_toml_value(val_str, cleaned)
        if isinstance(val, bool) or not isinstance(val, int):
            raise _invalid(f"_parse_config_scalar(): {field_name!r} requires an integer, got {val!r}")
        return val
    raise _invalid(f"_parse_config_scalar(): Unsupported scalar field: {field_name!r}")


def _format_toml_scalar(val: Any) -> str:
    if isinstance(val, str):
        return json.dumps(val)
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, tuple)):
        return f"[{', '.join(_format_toml_scalar(x) for x in val)}]"
    raise _invalid(f"_format_toml_scalar(): Unsupported TOML scalar type {type(val).__name__}: {val!r}")


def _extract_toml_section_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[[") or (close_idx := stripped.find("]")) == -1:
        return None
    trailing = stripped[close_idx + 1 :].strip()
    return stripped[1:close_idx].strip() or None if (not trailing or trailing.startswith("#")) else None


_TOML_SCAN_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'[^\']*\'|#.*$|(\[)|(\])')


def _toml_bracket_delta(line: str) -> int:
    return sum(1 if m.group(1) else -1 for m in _TOML_SCAN_RE.finditer(line) if m.group(1) or m.group(2))


def _resolve_toml_target(
    path: Path | str | None, workspace_override: Path | str | None, *, require_exists: bool, caller: str
) -> tuple[Path, Path | None]:
    has_ws = workspace_override is not None and bool(workspace_override)
    base_dir = Path(workspace_override).resolve() if has_ws else Path.cwd().resolve()
    if path is not None:
        toml_file = Path(path).resolve()
        if not toml_file.exists():
            raise _invalid(f"{caller}(): Configuration file not found: {toml_file}")
        return (base_dir if workspace_override is not None else toml_file.parent), toml_file
    candidate = base_dir / "yaw.toml"
    if candidate.exists():
        return base_dir, candidate
    if require_exists:
        raise _invalid(f"{caller}(): Configuration file not found: {candidate}")
    return base_dir, None


def load_config(path: Path | str | None = None, *, workspace_override: Path | str | None = None) -> YawConfig:
    """Load and validate a YawConfig from a TOML file (defaults to ./yaw.toml if present)."""
    base_dir, toml_file = _resolve_toml_target(path, workspace_override, require_exists=False, caller="load_config")
    data: dict[str, Any] = {}
    if toml_file is not None:
        try:
            data = tomllib.loads(toml_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise _invalid(f"load_config(): Invalid TOML in {toml_file}: {exc}") from exc

    kwargs: dict[str, Any] = {}
    for sec_name, section in data.items():
        if not isinstance(section, dict):
            raise _invalid(f"load_config(): [{sec_name}] in {toml_file} must be a table, got {type(section).__name__}")
        if "embed_batch_chars" in section:
            raise _invalid("load_config(): embed_batch_chars is retired; use embed_batch_size and embed_max_in_flight")
        if sec_name not in _ALLOWED_SECTIONS:
            if wrong := next((k for k in section if k in _FIELD_TO_SECTION), None):
                raise _invalid(f"load_config(): {wrong} belongs in [{_FIELD_TO_SECTION[wrong]}], not [{sec_name}]")
            raise _invalid(f"load_config(): Unknown or retired section: [{sec_name}]")

        for k, v in section.items():
            mapped_k = _OPTIMIZATION_KEY_ALIASES.get(k, k) if sec_name == "optimization" else k
            expected_sec = "paths" if mapped_k == "workspace_root" else _FIELD_TO_SECTION.get(mapped_k)
            if expected_sec is None:
                if k in _OPTIMIZATION_KEY_ALIASES:
                    raise _invalid(f"load_config(): {k} belongs in [optimization], not [{sec_name}]")
                raise _invalid(f"load_config(): Unknown or retired key: {sec_name}.{k}")
            if sec_name != expected_sec:
                raise _invalid(f"load_config(): {k} belongs in [{expected_sec}], not [{sec_name}]")
            if mapped_k == "optimize_datasets":
                if isinstance(v, str) or (isinstance(v, (list, tuple)) and all(isinstance(item, str) for item in v)):
                    kwargs[mapped_k] = tuple(parse_dataset_list(v))
                else:
                    raise _invalid(f"load_config(): 'optimize_datasets' must be a string or list of strings, got {v!r}")
            else:
                kwargs[mapped_k] = tuple(v) if isinstance(v, list) else v

    ws_raw = kwargs.pop("workspace_root", None)
    kwargs["workspace_root"] = (
        _resolve_path(base_dir, ws_raw, ".", field_name="workspace_root")
        if ws_raw is not None and (workspace_override is None or not bool(workspace_override))
        else base_dir
    )
    return YawConfig(**kwargs)


def _serialize_config_value(val: Any) -> Any:
    return str(val) if isinstance(val, Path) else list(val) if isinstance(val, tuple) else val


def get_config_value(
    path: Path | str | None = None, key: str | None = None, *, workspace_override: Path | str | None = None
) -> Any:
    """Read a specific key or all configuration values from yaw.toml with resolved YawConfig defaults."""
    cfg = load_config(path, workspace_override=workspace_override)
    if key is None:
        return {f.name: _serialize_config_value(getattr(cfg, f.name)) for f in fields(cfg)}

    cleaned_key = key.strip()
    if "." in cleaned_key:
        sec_part, subkey = (part.strip() for part in cleaned_key.split(".", 1))
        if sec_part in _ALLOWED_SECTIONS:
            sub_mapped = _OPTIMIZATION_KEY_ALIASES.get(subkey, subkey) if sec_part == "optimization" else subkey
            expected = "paths" if sub_mapped == "workspace_root" else _FIELD_TO_SECTION.get(sub_mapped)
            if expected == sec_part:
                return _serialize_config_value(getattr(cfg, sub_mapped))
    else:
        target_field = _OPTIMIZATION_KEY_ALIASES.get(cleaned_key, cleaned_key)
        if target_field == "workspace_root" or target_field in _FIELD_TO_SECTION:
            return _serialize_config_value(getattr(cfg, target_field))
    raise _invalid(f"get_config_value(): Unknown configuration key: {key!r}")


def set_config_value(
    key: str, raw_value: str, path: Path | str | None = None, *, workspace_override: Path | str | None = None
) -> tuple[str, Any]:
    """Safely update a configuration key in yaw.toml, verifying validity before atomic commit."""
    _, toml_path = _resolve_toml_target(path, workspace_override, require_exists=True, caller="set_config_value")
    assert toml_path is not None

    cleaned_key = key.strip()
    section_name: str | None = None
    prop_name = cleaned_key
    if "." in cleaned_key:
        section_name, prop_name = (part.strip() for part in cleaned_key.split(".", 1))
        if section_name not in _ALLOWED_SECTIONS:
            raise _invalid(f"set_config_value(): Unknown section: [{section_name}]")

    field_name = (
        _OPTIMIZATION_KEY_ALIASES.get(prop_name, prop_name) if section_name in (None, "optimization") else prop_name
    )
    if field_name not in _FIELD_TO_SECTION:
        raise _invalid(f"set_config_value(): Unknown configuration key: {key!r}")
    resolved_section = _FIELD_TO_SECTION[field_name]
    if section_name is not None and section_name != resolved_section:
        raise _invalid(f"set_config_value(): {prop_name} belongs in [{resolved_section}], not [{section_name}]")

    parsed_val = _parse_config_scalar(raw_value, field_name)
    target_prop = prop_name.removeprefix("optimize_") if prop_name.startswith("optimize_") else prop_name
    formatted_val = None if parsed_val is None else _format_toml_scalar(parsed_val)

    new_lines: list[str] = []
    current_sec: str | None = None
    key_written = False
    skip_depth = 0
    for line in toml_path.read_text(encoding="utf-8").splitlines():
        if skip_depth > 0:
            skip_depth = max(0, skip_depth + _toml_bracket_delta(line))
            continue
        if (sec_hdr := _extract_toml_section_name(line)) is not None:
            if formatted_val is not None and current_sec == resolved_section and not key_written:
                new_lines.append(f"{target_prop} = {formatted_val}")
                key_written = True
            current_sec = sec_hdr
            new_lines.append(line)
            continue
        if current_sec == resolved_section:
            parts = line.strip().split("=", 1)
            if len(parts) == 2 and parts[0].strip() in (target_prop, prop_name):
                skip_depth = max(0, _toml_bracket_delta(parts[1]))
                if formatted_val is not None:
                    new_lines.append(f"{target_prop} = {formatted_val}")
                    key_written = True
                continue
        new_lines.append(line)

    if formatted_val is not None and not key_written:
        if current_sec != resolved_section:
            new_lines.append(f"\n[{resolved_section}]")
        new_lines.append(f"{target_prop} = {formatted_val}")

    updated_text = "\n".join(new_lines) + "\n"

    def _write_validated(tmp: Path) -> None:
        tmp.write_text(updated_text, encoding="utf-8", newline="\n")
        load_config(tmp, workspace_override=workspace_override)

    write_via_temp_atomic(toml_path, _write_validated)
    return f"{resolved_section}.{target_prop}", parsed_val


__all__ = [
    "DEFAULT_EXCLUDED_PATTERNS",
    "DEFAULT_INCLUDED_PATTERNS",
    "YawConfig",
    "get_config_value",
    "load_config",
    "parse_dataset_list",
    "resolve_coco_state_root",
    "set_config_value",
    "write_text_atomic",
    "write_via_temp_atomic",
]
