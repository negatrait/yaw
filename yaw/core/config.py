"""Frozen TOML configuration contract and loader for YAW (stdlib-only)."""

import hashlib
import json
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
    included_patterns: tuple[str, ...] = ("*.md", "*.txt")
    excluded_patterns: tuple[str, ...] = (".*", "*/.*", "__pycache__/*")
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
    qa_model: str = "Mistral-7B-Instruct-v0.3-Hybrid"
    execution_mode: Literal["direct", "rlm"] = "direct"
    qa_mode: Literal["cot", "rlm"] = "cot"
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

    # Runtime & HTTP transport settings
    queue_maxsize: int = 32
    http_host: str = "127.0.0.1"
    http_port: int = 13306
    http_path: str = "/mcp"

    # Prompt optimization settings ([optimization] in yaw.toml)
    optimize_dataset: str | tuple[str, ...] | None = None
    optimize_datasets: tuple[str, ...] = ("2wikimultihopqa",)
    optimize_mode: Literal["miprov2", "gepa", "hybrid"] = "gepa"
    optimize_max_queries: int = 20
    optimize_train_queries: int = 7
    optimize_query_offset: int = 0
    optimize_max_metric_calls: int | None = 20

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
        if self.execution_mode not in ("direct", "rlm"):
            raise YawRuntimeError(
                "INVALID_INPUT", f"Unsupported execution_mode: {self.execution_mode!r}"
            )
        if self.qa_mode not in ("cot", "rlm"):
            raise YawRuntimeError("INVALID_INPUT", f"Unsupported qa_mode: {self.qa_mode!r}")
        if self.optimize_mode not in ("miprov2", "gepa", "hybrid"):
            raise YawRuntimeError(
                "INVALID_INPUT", f"Unsupported optimize_mode: {self.optimize_mode!r}"
            )
        raw_ds = (
            self.optimize_datasets
            if self.optimize_datasets != ("2wikimultihopqa",)
            else (self.optimize_dataset or self.optimize_datasets)
        )
        if isinstance(raw_ds, str):
            ds_tuple = tuple(p.strip() for p in raw_ds.split(",") if p.strip())
        elif isinstance(raw_ds, (list, tuple)):
            ds_tuple = tuple(str(p).strip() for p in raw_ds if str(p).strip())
        else:
            ds_tuple = ("2wikimultihopqa",)
        object.__setattr__(self, "optimize_datasets", ds_tuple or ("2wikimultihopqa",))
        object.__setattr__(
            self,
            "optimize_dataset",
            ds_tuple[0] if len(ds_tuple) == 1 else ds_tuple,
        )

    def config_fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of index-affecting settings."""
        payload = {
            "corpus_root": Path(self.corpus_root).as_posix() if self.corpus_root else "",
            "included_patterns": list(self.included_patterns),
            "excluded_patterns": list(self.excluded_patterns),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "embedding_model": self.embedding_model,
            "embedding_task_mode": self.embedding_task_mode,
            "extraction_model": self.extraction_model,
            "qa_model": self.qa_model,
            "execution_mode": self.execution_mode,
            "synonymy_threshold": self.synonymy_threshold,
            "damping": self.damping,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return "yaw-cfg-v1:" + hashlib.sha256(raw).hexdigest()[:16]


_OPTIMIZATION_KEY_ALIASES: dict[str, str] = {
    "dataset": "optimize_datasets",
    "datasets": "optimize_datasets",
    "mode": "optimize_mode",
    "evolve": "optimize_mode",
    "max_queries": "optimize_max_queries",
    "train_queries": "optimize_train_queries",
    "query_offset": "optimize_query_offset",
    "max_metric_calls": "optimize_max_metric_calls",
}


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


__all__ = [
    "YawConfig",
    "load_config",
]
