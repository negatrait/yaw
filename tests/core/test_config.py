"""Unit tests for slim YawConfig contract, active local defaults, yaw.toml bijection, and strict loading."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

import yaw.core.config as config_mod
from yaw.core import (
    DEFAULT_EXCLUDED_PATTERNS,
    DEFAULT_INCLUDED_PATTERNS,
    YawConfig,
    YawRuntimeError,
    get_config_value,
    load_config,
    parse_dataset_list,
    resolve_coco_state_root,
    set_config_value,
    write_text_atomic,
    write_via_temp_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OPTIONAL_COMMENT_KV_RE = re.compile(r"^#\s*([a-z][a-z0-9_]*)\s*=")
_SECTION_HEADER_RE = re.compile(r"^\[([a-z][a-z0-9_]*)\]\s*$")


def test_yaw_config_defaults() -> None:
    cfg = YawConfig()
    assert len(fields(YawConfig)) == 32
    assert cfg.corpus_root is not None and cfg.corpus_root.name == "corpus"
    assert cfg.wiki_root is None
    assert cfg.state_dir is not None and cfg.state_dir.name == "coco_lmdb"
    assert cfg.included_patterns == DEFAULT_INCLUDED_PATTERNS
    assert cfg.excluded_patterns == DEFAULT_EXCLUDED_PATTERNS
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 150
    assert cfg.min_chunk_size == 100
    assert cfg.embed_batch_size == 5
    assert cfg.base_url == "http://127.0.0.1:8888/v1"
    assert cfg.provider == "openai"
    assert cfg.embedding_model == "embeddinggemma-2"
    assert cfg.embedding_task_mode == "distinct-titled"
    assert cfg.embed_max_in_flight == 2
    assert cfg.qa_model == "gemma-4-e4b"
    assert cfg.max_output_tokens is None
    assert cfg.react_max_iters == 6
    assert cfg.queue_maxsize == 32
    assert cfg.http_host == "127.0.0.1"
    assert cfg.http_port == 13306
    assert cfg.http_path == "/mcp"
    assert cfg.optimize_datasets == ("enterprise_rag_bench",)
    assert cfg.optimize_max_queries == 162
    assert cfg.optimize_query_offset == 0
    assert cfg.optimize_auto == "light"

    for retired in (
        "poll_interval_seconds",
        "json_keys_whitelist",
        "json_keys_blacklist",
        "extraction_model",
        "context_window_tokens",
        "rlm_max_iters",
        "rlm_max_llm_calls",
        "rlm_max_output_chars",
        "reflection_input_chars",
        "enable_gbnf",
        "gbnf_max_entities",
        "gbnf_max_triples",
        "gbnf_max_facts",
        "gbnf_max_reasoning_chars",
        "linking_top_k",
        "retrieval_top_k",
        "damping",
        "retriever_mode",
        "candidate_dense_threshold",
        "candidate_lexical_threshold",
        "auto_merge_threshold",
        "candidate_volume_ratio",
        "semantic_adjudication_batch_size",
        "max_context_chars",
        "max_ast_expansion_hops",
        "ast_context_hops",
        "optimize_mode",
        "optimize_train_queries",
        "optimize_sleep_seconds",
        "optimize_max_bootstrapped_demos",
        "optimize_max_labeled_demos",
        "retrieval",
        "manifold_overrides",
    ):
        assert not hasattr(cfg, retired), f"Retired attribute {retired} still on YawConfig"

    for retired_sym in (
        "RetrievalParams",
        "write_manifold_section",
        "write_retrieval_section",
        "scale_context_budget",
        "context_char_limit",
        "DEFAULT_CONTEXT_WINDOW_TOKENS",
    ):
        assert not hasattr(config_mod, retired_sym), f"Retired symbol {retired_sym} still in yaw.core.config"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimize_auto": "invalid"},
        {"embedding_task_mode": "invalid"},
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 150},
        {"min_chunk_size": 0},
        {"embed_batch_size": 0},
        {"embed_max_in_flight": 0},
        {"react_max_iters": 0},
        {"queue_maxsize": 0},
        {"max_output_tokens": 0},
        {"max_output_tokens": -5},
        {"max_output_tokens": True},
        {"max_output_tokens": 1.5},
        {"http_port": 0},
        {"http_port": 65536},
        {"http_path": "mcp"},
        {"http_path": ""},
        {"optimize_max_queries": 0},
        {"optimize_query_offset": -1},
        {"chunk_size": True},
        {"http_port": 80.5},
        {"react_max_iters": None},
    ],
)
def test_yaw_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(YawRuntimeError) as exc_info:
        YawConfig(**kwargs)
    assert exc_info.value.code == "INVALID_INPUT"


def test_yaw_config_for_optimization(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    opt = cfg.for_optimization()

    assert opt.corpus_root == cfg.workspace_root
    assert opt.db_path is not None and opt.db_path.name == "optimize_yaw.db"
    assert opt.lock_path is not None and opt.lock_path.name == "optimize_yaw.db.lock"
    assert opt.status_path is not None and opt.status_path.name == "dspy_status.json"
    assert opt.state_dir is not None and opt.state_dir.name == "coco_lmdb"
    assert opt.compiled_program_path is not None and opt.compiled_program_path.name == "qa_program.json"
    assert opt.embedding_model == cfg.embedding_model
    assert opt.qa_model == cfg.qa_model
    assert opt.optimize_auto == cfg.optimize_auto
    assert opt.react_max_iters == cfg.react_max_iters

    custom = cfg.for_optimization(tmp_path / "custom" / "optimize_yaw_custom.db")
    assert custom.db_path == (tmp_path / "custom" / "optimize_yaw_custom.db").resolve()
    assert custom.status_path == (tmp_path / "custom" / "optimize_status_custom.json").resolve()
    assert custom.compiled_program_path == (tmp_path / "custom" / "optimize_program_custom.json").resolve()
    assert resolve_coco_state_root(tmp_path / "coco_state") == (tmp_path / "coco_state" / "coco_lmdb").resolve()


def test_optimization_empty_db_override_raises(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    with pytest.raises(YawRuntimeError, match="cannot be empty"):
        cfg.for_optimization(db_override="   ")
    assert not (tmp_path / ".yaw").exists()


def test_optimization_custom_database_has_stable_paths(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    database = tmp_path / "custom" / "experiment.db"
    database.parent.mkdir()
    database.touch()
    first = cfg.for_optimization(db_override="custom/experiment.db")
    second = cfg.for_optimization(db_override=database)
    assert first == second
    assert first.lock_path == database.with_suffix(".db.lock")
    assert first.status_path == (database.parent / "optimize_status_experiment.json")
    assert first.state_dir == (database.parent / "coco_lmdb")
    assert first.compiled_program_path == (database.parent / "optimize_program_experiment.json")


def test_parse_dataset_list() -> None:
    assert parse_dataset_list(None) == []
    assert parse_dataset_list("") == []
    assert parse_dataset_list("popqa, 2wikimultihopqa , , hotpotqa") == [
        "popqa",
        "2wikimultihopqa",
        "hotpotqa",
    ]
    assert parse_dataset_list(["popqa,musique", " hotpotqa "]) == [
        "popqa",
        "musique",
        "hotpotqa",
    ]


def test_config_get_and_set_values(tmp_path: Path) -> None:
    toml_file = tmp_path / "yaw.toml"
    toml_file.write_text(
        """[optimization]
auto = "light"

[inference]
qa_model = "gemma-4-e4b"
""",
        encoding="utf-8",
    )

    assert get_config_value(path=toml_file, key="optimization.auto", workspace_override=tmp_path) == "light"
    assert get_config_value(path=toml_file, key="auto", workspace_override=tmp_path) == "light"

    key, val = set_config_value(
        key="optimization.auto",
        raw_value="heavy",
        path=toml_file,
        workspace_override=tmp_path,
    )
    assert key == "optimization.auto"
    assert val == "heavy"
    assert get_config_value(path=toml_file, key="optimization.auto", workspace_override=tmp_path) == "heavy"

    all_cfg = get_config_value(path=toml_file, workspace_override=tmp_path)
    assert isinstance(all_cfg, dict)
    assert all_cfg["optimize_auto"] == "heavy"


def test_yaw_config_patterns(tmp_path: Path) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    for ext in ("*.md", "*.txt", "*.json", "*.jsonl", "*.py", "*.rs", "*.ts", "*.html", "*.toml", "*.yaml"):
        assert ext in cfg.included_patterns
    assert not hasattr(cfg, "for_benchmark")
    assert not hasattr(cfg, "config_fingerprint")


def test_default_patterns_and_write_text_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = YawConfig(workspace_root=tmp_path)
    assert cfg.included_patterns == DEFAULT_INCLUDED_PATTERNS
    assert cfg.excluded_patterns == DEFAULT_EXCLUDED_PATTERNS
    assert "*.md" in DEFAULT_INCLUDED_PATTERNS
    assert "__pycache__/*" in DEFAULT_EXCLUDED_PATTERNS

    target = tmp_path / "nested" / "atomic.json"
    write_text_atomic(target, '{"status": "ok"}\n')
    assert target.read_text(encoding="utf-8") == '{"status": "ok"}\n'
    write_text_atomic(str(target), '{"status": "updated"}\n', retries=2)
    assert target.read_text(encoding="utf-8") == '{"status": "updated"}\n'
    write_via_temp_atomic(target, lambda tmp: tmp.write_bytes(b'{"status": "via_fn"}\n'))
    assert target.read_text(encoding="utf-8") == '{"status": "via_fn"}\n'
    assert not list(target.parent.glob(".*tmp*"))

    real_replace = os.replace
    attempts = 0

    def flaky_replace(src: Any, dst: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("Simulated transient Windows file lock")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    write_text_atomic(target, '{"status": "recovered"}\n', retries=2)
    assert attempts == 2
    assert target.read_text(encoding="utf-8") == '{"status": "recovered"}\n'

    def always_locked_replace(_src: Any, _dst: Any) -> None:
        raise PermissionError("Simulated persistent Windows file lock")

    monkeypatch.setattr(os, "replace", always_locked_replace)
    with pytest.raises(PermissionError, match="Simulated persistent Windows file lock"):
        write_text_atomic(target, '{"status": "fallback"}\n', retries=2)
    assert target.read_text(encoding="utf-8") == '{"status": "recovered"}\n'
    assert not list(target.parent.glob(".*tmp*"))


def test_project_yaw_toml_bijection_with_yaw_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked-in yaw.toml loads equal to YawConfig defaults and matches _FIELD_TO_SECTION 1-to-1."""
    monkeypatch.delenv("UNSLOTH_STUDIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    toml_path = PROJECT_ROOT / "yaw.toml"
    loaded = load_config(toml_path)
    expected = YawConfig(workspace_root=PROJECT_ROOT)
    assert replace(loaded, api_key="") == replace(expected, api_key="")
    assert loaded == expected

    raw_text = toml_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(raw_text)
    assert set(parsed.keys()) == {"paths", "ingestion", "inference", "runtime", "optimization"}

    toml_pairs: set[tuple[str, str]] = set()
    for sec_name, table in parsed.items():
        assert isinstance(table, dict)
        for key in table:
            field_name = config_mod._OPTIMIZATION_KEY_ALIASES.get(key, key) if sec_name == "optimization" else key
            toml_pairs.add((sec_name, field_name))

    commented_optional: set[tuple[str, str]] = set()
    cur_section: str | None = None
    for line in raw_text.splitlines():
        stripped = line.strip()
        if sec_match := _SECTION_HEADER_RE.match(stripped):
            sec_name_match = sec_match.group(1)
            assert isinstance(sec_name_match, str)
            cur_section = sec_name_match
            continue
        if cur_section is not None and (opt_match := _OPTIONAL_COMMENT_KV_RE.match(stripped)):
            commented_optional.add((cur_section, opt_match.group(1)))

    assert commented_optional == {
        ("paths", "wiki_root"),
        ("inference", "api_key"),
        ("inference", "max_output_tokens"),
    }
    assert toml_pairs.isdisjoint(commented_optional)

    expected_pairs = {(sec, field_name) for field_name, sec in config_mod._FIELD_TO_SECTION.items()}
    assert len(config_mod._FIELD_TO_SECTION) == 31
    assert (toml_pairs | commented_optional) == expected_pairs


@pytest.mark.parametrize(
    "bad_toml",
    [
        "[graph]\n",
        "[graph]\nlinking_top_k = 8\n",
        "[retrieval]\n",
        "[retrieval]\npeclet = 0.338\n",
        "[manifold]\npeclet = 0.55\n",
        "[unknown]\nfoo = 1\n",
        'manifold = "not_a_table"\n',
        "manifold = [1, 2, 3]\n",
        'inference = "not_a_table"\n',
        "[ingestion]\npoll_interval_seconds = 0.2\n",
        "[ingestion]\nembed_batch_chars = 2048\n",
        "[inference]\nrlm_max_iters = 20\n",
        "[inference]\nextraction_model = 'gemma-4-e4b'\n",
        "[optimization]\nmode = 'gepa'\n",
        "[optimization]\ntrain_queries = 10\n",
        "[paths]\ncorpus_dir = 'corpus'\n",
        "[runtime]\nchunk_size = 500\n",
        "[ingestion]\nhttp_port = 13306\n",
        "[paths]\nqa_model = 'gemma-4-e4b'\n",
        "[inference]\nworkspace_root = '.'\n",
        "[inference]\nauto = 'light'\n",
        "[inference]\noptimize_auto = 'light'\n",
    ],
)
def test_load_config_rejects_retired_sections_keys_and_wrong_sections(tmp_path: Path, bad_toml: str) -> None:
    toml_path = tmp_path / "yaw.toml"
    toml_path.write_text(bad_toml, encoding="utf-8")
    with pytest.raises(YawRuntimeError) as exc_info:
        load_config(toml_path)
    assert exc_info.value.code == "INVALID_INPUT"


def test_get_and_set_reject_retired_and_wrong_section_keys(tmp_path: Path) -> None:
    toml_path = tmp_path / "yaw.toml"
    original = "[runtime]\nhttp_port = 13306\n\n[ingestion]\nchunk_size = 1000\n"
    toml_path.write_text(original, encoding="utf-8")

    for bad_key in (
        "manifold",
        "manifold.peclet",
        "retrieval",
        "retrieval.peclet",
        "poll_interval_seconds",
        "mode",
        "runtime.chunk_size",
        "ingestion.http_port",
        "paths.qa_model",
    ):
        with pytest.raises(YawRuntimeError) as exc_get:
            get_config_value(toml_path, bad_key)
        assert exc_get.value.code == "INVALID_INPUT"

        with pytest.raises(YawRuntimeError) as exc_set:
            set_config_value(bad_key, "1", toml_path)
        assert exc_set.value.code == "INVALID_INPUT"

    assert toml_path.read_text(encoding="utf-8") == original


def test_wiki_root_validation_disjointness_toml_and_optimization_isolation(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = YawConfig(workspace_root=ws, wiki_root=Path("wiki"))
    assert cfg.wiki_root == (ws / "wiki").resolve()
    assert YawConfig(workspace_root=ws, wiki_root=cast(Any, "wiki")).wiki_root == (ws / "wiki").resolve()

    opt = cfg.for_optimization()
    assert opt.wiki_root is None

    ext_wiki = (tmp_path / "ext_wiki").resolve()
    opt_with_wiki = cfg.for_optimization(db_override=ws / "opt" / "yaw.db", wiki_root=ext_wiki)
    assert opt_with_wiki.wiki_root == ext_wiki

    with pytest.raises(YawRuntimeError) as exc_opt_overlap:
        cfg.for_optimization(wiki_root=ws / "wiki")
    assert exc_opt_overlap.value.code == "INVALID_INPUT"

    for bad_wiki in ("", "   ", True, 123):
        with pytest.raises(YawRuntimeError) as exc_bad:
            YawConfig(workspace_root=ws, wiki_root=cast(Any, bad_wiki))
        assert exc_bad.value.code == "INVALID_INPUT"

    for overlap_kwargs in (
        {"corpus_root": "corpus", "wiki_root": "corpus"},
        {"corpus_root": "corpus", "wiki_root": "corpus/cards"},
        {"corpus_root": "corpus/sub", "wiki_root": "corpus"},
        {"corpus_root": "corpus", "wiki_root": "."},
    ):
        with pytest.raises(YawRuntimeError) as exc_overlap:
            YawConfig(workspace_root=ws, **cast(dict[str, Any], overlap_kwargs))
        assert exc_overlap.value.code == "INVALID_INPUT"

    toml_path = ws / "yaw.toml"
    toml_path.write_text('[paths]\ncorpus_root = "corpus"\n', encoding="utf-8")
    assert get_config_value(toml_path, "paths.wiki_root") is None

    assert set_config_value("paths.wiki_root", "wiki", toml_path) == ("paths.wiki_root", "wiki")
    assert load_config(toml_path).wiki_root == (ws / "wiki").resolve()
    assert get_config_value(toml_path, "paths.wiki_root") == str((ws / "wiki").resolve())

    assert set_config_value("paths.wiki_root", "null", toml_path) == ("paths.wiki_root", None)
    assert load_config(toml_path).wiki_root is None

    with pytest.raises(YawRuntimeError) as exc_empty:
        set_config_value("paths.wiki_root", '""', toml_path)
    assert exc_empty.value.code == "INVALID_INPUT"
    assert load_config(toml_path).wiki_root is None
