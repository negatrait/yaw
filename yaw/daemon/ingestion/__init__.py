"""Ingestion leaf package: Native CocoIndex v1 incremental line-pointer ingestion."""

import importlib
from typing import Any

# Single source of truth: only these symbols are exposed as the public package contract.
_PUBLIC_API: dict[str, str] = {
    "select_enterprise_questions": ".datasets",
    "ENTERPRISE_CATEGORY_QUOTAS": ".datasets",
    "DATASET_SOURCE_FIELDS": ".datasets",
    "resolve_dataset_bundle": ".datasets",
    "document_text": ".bundles",
    "prepare_dataset_bundle": ".bundles",
    "load_dataset_bundle": ".bundles",
    "validate_dataset_bundle": ".bundles",
    "DatasetDocument": ".bundles",
    "DatasetBundle": ".bundles",
    "SOURCE_DB": ".source_targets",
    "SOURCE_ROOT": ".source_targets",
    "SourceMetadata": ".source_targets",
    "SourceSettings": ".source_targets",
    "DATASETS": ".datasets",
    "DATASET_REGISTRY": ".datasets",
    "DEFAULT_PARITY_DATASETS": ".datasets",
    "HF_BASE_URL": ".datasets",
    "PINNED_COMMIT": ".datasets",
    "QA_ONLY_DATASETS": ".datasets",
    "TIER2_CORPORA": ".datasets",
    "UPSTREAM_DATASET_BASE_URL": ".datasets",
    "build_benchmark_slice": ".datasets",
    "chunk_source_text": ".corpus",
    "detect_corpus_language": ".corpus",
    "download_dataset_file": ".datasets",
    "extract_ast_headings": ".corpus",
    "extract_chunk_title": ".corpus",
    "extract_gold_answers": ".datasets",
    "extract_markdown_headings": ".corpus",
    "get_gold_answers": ".datasets",
    "get_gold_docs": ".datasets",
    "load_dataset_files": ".datasets",
    "load_dataset_splits": ".datasets",
    "project_json_records": ".corpus",
    "resolve_dataset_name": ".datasets",
    "resolve_heading_for_line": ".corpus",
    "should_keep_json_key_path": ".corpus",
    "build_source_path_matcher": ".source_targets",
    "chunk_file_memo": ".source_targets",
    "declare_source_file": ".source_targets",
    "normalize_chunking_settings": ".source_targets",
    "acquire_coco_environment": ".source_targets",
    "close_coco_environment": ".source_targets",
    "SharedCocoRuntime": ".source_targets",
    "SHARED_COCO_RUNTIMES": ".source_targets",
}

__all__ = sorted(_PUBLIC_API.keys())


def __getattr__(name: str) -> Any:
    if name in _PUBLIC_API:
        mod_name = _PUBLIC_API[name]
        mod = importlib.import_module(mod_name, __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
