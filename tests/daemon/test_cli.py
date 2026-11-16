"""End-to-end CLI config [manifold] and Optuna MOTPE tuning tests over real SQLite + loopback HTTP."""

import json
import re
from pathlib import Path

import dspy
import numpy as np
import optuna
import pytest
from test_benchmark import benchmark_inference_server as benchmark_inference_server

import benchmark
from yaw.core import load_config
from yaw.daemon.cli import main
from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter
from yaw.daemon.ingestion import prepare_dataset_bundle
from yaw.daemon.manifold_optuna import tune_manifold
from yaw.daemon.runtime import YawRuntime


def test_cli_config_manifold_get_set_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """yaw config get/set supports [manifold] scalars, rejects invalid values, and exits 1 on unset keys."""
    toml_path = tmp_path / "yaw.toml"
    toml_path.write_text(
        '[paths]\ncorpus_dir = "corpus"\n\n[retrieval]\nretriever_mode = "manifold"\n', encoding="utf-8"
    )

    assert main(["config", "set", "manifold.peclet", "0.55", "--config", str(toml_path)]) == 0
    set_out = json.loads(capsys.readouterr().out)
    assert set_out == {"updated": True, "key": "manifold.peclet", "value": 0.55}

    assert main(["config", "get", "manifold.peclet", "--config", str(toml_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"manifold.peclet": 0.55}

    assert main(["config", "get", "manifold", "--config", str(toml_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"manifold": {"peclet": 0.55}}

    assert main(["config", "get", "manifold.unset", "--config", str(toml_path)]) == 1
    capsys.readouterr()

    assert main(["config", "set", "manifold.peclet", "true", "--config", str(toml_path)]) == 1
    capsys.readouterr()


def test_cli_dspy_tune_manifold_persists_optuna_and_applies_toml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    benchmark_inference_server: tuple[str, list[object]],
) -> None:
    """End-to-end tune-manifold builds SQLite index, persists 5-horizon Optuna study, and saves yaw.toml."""
    base_url, _ = benchmark_inference_server

    docs = [
        {
            "idx": "turing",
            "title": "Alan Turing",
            "text": "Alan Turing worked at Bletchley Park cracking Enigma ciphers during the war.",
        },
        {
            "idx": "curie",
            "title": "Marie Curie",
            "text": "Marie Curie isolated radium and polonium inside her Paris laboratory.",
        },
        {
            "idx": "lovelace",
            "title": "Ada Lovelace",
            "text": "Ada Lovelace wrote notes describing an algorithm for Babbage Analytical Engine.",
        },
        {
            "idx": "hopper",
            "title": "Grace Hopper",
            "text": "Grace Hopper pioneered early machine-independent compiler design and COBOL tools.",
        },
        {
            "idx": "shannon",
            "title": "Claude Shannon",
            "text": "Claude Shannon authored foundational mathematical theory of communication and entropy.",
        },
        {
            "idx": "noether",
            "title": "Emmy Noether",
            "text": "Emmy Noether proved symmetry conservation laws in mathematical physics.",
        },
    ]
    sample_specs = [
        ("Where did Alan Turing work cracking Enigma?", "Bletchley Park", "Alan Turing", docs[0]["text"]),
        ("Which elements did Marie Curie isolate in Paris?", "radium and polonium", "Marie Curie", docs[1]["text"]),
        ("Who wrote notes for Analytical Engine?", "Ada Lovelace", "Ada Lovelace", docs[2]["text"]),
        ("Who pioneered compiler design and COBOL?", "Grace Hopper", "Grace Hopper", docs[3]["text"]),
        ("Who authored mathematical theory of communication?", "Claude Shannon", "Claude Shannon", docs[4]["text"]),
        ("Who proved symmetry conservation laws?", "Emmy Noether", "Emmy Noether", docs[5]["text"]),
    ]
    # Emit 14 distinct questions: [0, 10) simulates held-out eval, [10, 14) gives 2 train + 2 val tuning queries.
    questions = []
    for idx in range(14):
        q_tmpl, ans, title, text = sample_specs[idx % len(sample_specs)]
        questions.append(
            {
                "question": f"{q_tmpl} (case {idx})",
                "answer": ans,
                "context": [[title, [text]]],
            }
        )

    corpus_path = tmp_path / "documents.json"
    questions_path = tmp_path / "questions.json"
    corpus_path.write_text(json.dumps(docs), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    prepare_dataset_bundle("sample", corpus_path, questions_path, tmp_path / "outputs" / "datasets")

    config_path = tmp_path / "yaw.toml"
    config_path.write_text(
        f'[inference]\nbase_url = "{base_url}"\napi_key = "benchmark-key"\n'
        'embedding_model = "embeddinggemma-2"\nextraction_model = "gemma-4-e4b"\nqa_model = "gemma-4-e4b"\n'
        "context_window_tokens = 32768\nmax_output_tokens = 1024\nenable_gbnf = true\n"
        "[ingestion]\nchunk_size = 256\nchunk_overlap = 32\nmin_chunk_size = 24\n"
        '[retrieval]\nretriever_mode = "manifold"\n',
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    with pytest.raises(ValueError, match="overlaps held-out evaluation range"):
        tune_manifold(
            cfg,
            datasets=("sample",),
            query_offset=0,
            max_queries=4,
            train_queries=2,
            index_queries=4,
            trials=2,
        )

    assert (
        main(
            [
                "dspy",
                "tune-manifold",
                "--config",
                str(config_path),
                "--query-offset",
                "0",
                "--max-queries",
                "4",
                "--train-queries",
                "2",
                "--index-queries",
                "4",
                "--trials",
                "2",
                "--dataset",
                "sample",
            ]
        )
        == 1
    )
    capsys.readouterr()

    study_db = tmp_path / "optuna_test.db"
    extra_out = tmp_path / "custom_tune_out.json"
    exit_code = main(
        [
            "dspy",
            "tune-manifold",
            "--config",
            str(config_path),
            "--dataset",
            "sample",
            "--query-offset",
            "10",
            "--max-queries",
            "4",
            "--train-queries",
            "2",
            "--index-queries",
            "4",
            "--trials",
            "2",
            "--study-db",
            str(study_db),
            "--study-name",
            "test_motpe",
            "--seed",
            "42",
            "--apply",
            "--output",
            str(extra_out),
        ]
    )
    assert exit_code == 0
    captured_first = capsys.readouterr()
    payload = json.loads(captured_first.out)
    assert "[tune-manifold]" in captured_first.err

    assert payload["applied"] is True
    assert payload["applied_config_path"] == str(config_path.resolve())
    assert payload["selected"]["val"]["cycles"] == 0
    assert payload["selected"]["val"]["seeds"] == payload["selected"]["val"]["queries"]
    assert payload["study"]["trials_total"] == 2
    assert payload["study"]["completed_trials"] == 2
    assert len(payload["study"]["specs"]) == 5
    assert isinstance(payload["study"]["param_importances"], dict)
    assert "stage_a" not in payload
    assert "stage_b" not in payload
    assert "selected_source" in payload["selected"]
    for block in (payload["baseline"]["train"], payload["baseline"]["val"], payload["selected"]["val"]):
        for key in ("r1", "r2", "r5", "r10", "r20", "combined"):
            assert key in block

    best_json = tmp_path / ".yaw" / "optuna_manifold_best.json"
    assert best_json.exists()
    assert extra_out.exists()
    assert json.loads(best_json.read_text(encoding="utf-8")) == payload

    assert study_db.exists()
    storage_url = f"sqlite:///{study_db.as_posix()}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    try:
        loaded = optuna.load_study(study_name="test_motpe", storage=storage)
        assert len(loaded.directions) == 5
        assert len(loaded.trials) == 2
        trial_vals = loaded.trials[0].values
        assert trial_vals is not None
        assert len(trial_vals) == 5
    finally:
        storage.engine.dispose()

    reloaded_cfg = load_config(config_path)
    assert dict(reloaded_cfg.manifold_overrides) == payload["best_overrides"]
    assert "[manifold]" in config_path.read_text(encoding="utf-8")


def test_tune_manifold_baseline_fallback_warning_and_dir_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    benchmark_inference_server: tuple[str, list[object]],
) -> None:
    """tune_manifold accepts directory dataset paths and flags baseline_fallback loudly on stderr."""
    base_url, _ = benchmark_inference_server
    docs = [
        {"idx": f"d{i}", "title": f"Entity {i}", "text": f"Entity {i} discovered constant kappa_{i} in sector {i}."}
        for i in range(6)
    ]
    questions = [
        {
            "question": f"What did Entity {i % 6} discover? (case {i})",
            "answer": f"kappa_{i % 6}",
            "context": [[f"Entity {i % 6}", [docs[i % 6]["text"]]]],
        }
        for i in range(14)
    ]
    corpus_path = tmp_path / "docs.json"
    questions_path = tmp_path / "qs.json"
    corpus_path.write_text(json.dumps(docs), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    bundle_dir = prepare_dataset_bundle("sample", corpus_path, questions_path, tmp_path / "custom_bundles").root

    cfg_path = tmp_path / "yaw.toml"
    cfg_path.write_text(
        f'[paths]\ncorpus_dir = "corpus"\n'
        f'[inference]\nbase_url = "{base_url}"\napi_key = "benchmark-key"\n'
        'embedding_model = "embeddinggemma-2"\nextraction_model = "gemma-4-e4b"\nqa_model = "gemma-4-e4b"\n'
        "context_window_tokens = 32768\nmax_output_tokens = 1024\nenable_gbnf = true\n"
        "[ingestion]\nchunk_size = 256\nchunk_overlap = 32\nmin_chunk_size = 24\n"
        '[retrieval]\nretriever_mode = "manifold"\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    result = tune_manifold(
        cfg,
        datasets=str(bundle_dir),
        query_offset=10,
        max_queries=4,
        train_queries=2,
        index_queries=4,
        trials=0,
        study_db=tmp_path / "fallback.db",
        study_name="fallback_test",
    )
    err_text = capsys.readouterr().err
    assert result["datasets"] == ["sample"]
    assert result["best_source"] == "baseline_fallback"
    assert result["best_overrides"] == {}
    assert result["selected"]["selected_source"] == "baseline_fallback"
    assert "baseline_fallback" in err_text


def test_runtime_and_script_benchmark_five_point_recall_preserves_qa_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_inference_server: tuple[str, list[tuple[str, str | None, dict[str, object] | None]]],
) -> None:
    """YawRuntime.benchmark() and benchmark.main() report 5-pt Recall@{1,2,5,10,20} while keeping QA top-k=5."""
    base_url, requests = benchmark_inference_server
    docs = [
        {
            "idx": f"doc_{i}",
            "title": f"MarkerDoc{i:02d}",
            "text": f"MarkerDoc{i:02d} records Alan Turing working in Cambridge note {i}.",
        }
        for i in range(8)
    ]
    questions = [
        {
            "question": "Where did Alan Turing work?",
            "answer": "Cambridge",
            "context": [["MarkerDoc00", [docs[0]["text"]]], ["MarkerDoc01", [docs[1]["text"]]]],
        }
    ]
    corpus_path = tmp_path / "documents.json"
    questions_path = tmp_path / "questions.json"
    corpus_path.write_text(json.dumps(docs), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    prepare_dataset_bundle("sample", corpus_path, questions_path, tmp_path / "outputs" / "datasets")

    cfg_path = tmp_path / "yaw.toml"
    cfg_path.write_text(
        f'[paths]\ncorpus_dir = "corpus"\n'
        f'[inference]\nbase_url = "{base_url}"\napi_key = "benchmark-key"\n'
        'embedding_model = "embeddinggemma-2"\nextraction_model = "gemma-4-e4b"\nqa_model = "gemma-4-e4b"\n'
        "context_window_tokens = 32768\nmax_output_tokens = 1024\nenable_gbnf = true\n"
        "[ingestion]\nchunk_size = 256\nchunk_overlap = 32\nmin_chunk_size = 24\n"
        '[retrieval]\nretriever_mode = "manifold"\nretrieval_top_k = 5\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    rt = YawRuntime(cfg, auto_poll=False)
    requests.clear()
    shootout_report = rt.benchmark(
        datasets="sample",
        max_queries=1,
        index_queries=8,
        mode="eval",
        retriever="shootout",
        fresh=True,
    )

    for arm in ("current_yaw", "substate_manifold_hmm"):
        ret_metrics = shootout_report["candidates"][arm]["datasets"]["sample"]["evaluation_metrics"]["hipporag"][
            "retrieval"
        ]
        for k_key in ("Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"):
            assert k_key in ret_metrics
    macro = shootout_report["comparison"]["macro_average"]
    for k_key in ("Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"):
        assert k_key in macro["current_yaw"]
        assert k_key in macro["substate_manifold_hmm"]

    qa_prompts = [
        str(body["prompt"])
        for path, _, body in requests
        if path == "/v1/completions" and isinstance(body, dict) and "answer" in str(body.get("grammar", ""))
    ]
    assert qa_prompts
    for prompt in qa_prompts:
        passages_part = prompt.split("### Retrieved Passages")[-1] if "### Retrieved Passages" in prompt else prompt
        marker_hits = set(re.findall(r"MarkerDoc\d{2}", passages_part))
        assert len(marker_hits) <= 5

    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    requests.clear()
    script_report = benchmark.main(
        dataset="sample",
        config_path=cfg_path,
        mode="eval",
        max_queries=1,
        index_queries=8,
        fresh_index=True,
    )
    script_ret = script_report["evaluation_metrics"]["hipporag"]["retrieval"]
    for k_key in ("Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"):
        assert k_key in script_ret
    script_qa_prompts = [
        str(body["prompt"])
        for path, _, body in requests
        if path == "/v1/completions" and isinstance(body, dict) and "answer" in str(body.get("grammar", ""))
    ]
    assert script_qa_prompts
    for prompt in script_qa_prompts:
        passages_part = prompt.split("### Retrieved Passages")[-1] if "### Retrieved Passages" in prompt else prompt
        marker_hits = set(re.findall(r"MarkerDoc\d{2}", passages_part))
        assert len(marker_hits) <= 5


def test_tune_manifold_selects_feasible_pareto_and_ped_anova_importances(
    tmp_path: Path,
    benchmark_inference_server: tuple[str, list[object]],
) -> None:
    """tune_manifold selects a feasible Pareto winner over defaults and computes PedAnova importances."""
    base_url, _ = benchmark_inference_server

    class _ParetoEmbedModel:
        index_identity = "test://pareto-embedder-v1"

        def __call__(self, texts: list[str]) -> np.ndarray:
            out = []
            for text in texts:
                s = str(text)
                v = np.zeros(768, dtype=np.float32)
                if "GoldOne" in s or "primary optical" in s:
                    v[0], v[10] = 0.92, 0.39
                elif "Distractor" in s or "clone optical" in s:
                    v[0], v[1], v[10] = 0.82, 0.42, 0.39
                elif "GoldTwo" in s or "secondary acoustic" in s:
                    v[2], v[10] = 0.92, 0.39
                elif "Alpha optical inquiry" in s:
                    v[0], v[1], v[2], v[10] = 0.78, 0.25, 0.52, 0.25
                else:
                    idx = sum(ord(ch) for ch in s) % 8
                    v[3 + idx], v[10] = 0.30, 0.95
                v /= max(float(np.linalg.norm(v)), 1e-12)
                out.append(v)
            return np.asarray(out, dtype=np.float32)

    custom_emb = DSPyEmbeddingAdapter(embedder=dspy.Embedder(_ParetoEmbedModel(), caching=False))

    docs = [
        {"idx": "g1", "title": "GoldOne", "text": "Alpha primary optical findings from archive 0."},
        {"idx": "d1", "title": "Distractor", "text": "Alpha clone optical findings from archive 1."},
        {"idx": "g2", "title": "GoldTwo", "text": "secondary acoustic findings from archive 2."},
        *[{"idx": f"o{i}", "title": f"Other{i}", "text": f"unrelated background entry {i}."} for i in range(10)],
    ]
    questions = [
        {
            "question": f"Alpha optical inquiry case {idx}",
            "answer": "Alpha",
            "context": [["Distractor", [docs[1]["text"]]]],
        }
        for idx in range(18)
    ]

    corpus_path = tmp_path / "documents.json"
    questions_path = tmp_path / "questions.json"
    corpus_path.write_text(json.dumps(docs), encoding="utf-8")
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    prepare_dataset_bundle("sample", corpus_path, questions_path, tmp_path / "outputs" / "datasets")

    config_path = tmp_path / "yaw.toml"
    config_path.write_text(
        f'[paths]\ncorpus_dir = "corpus"\n'
        f'[inference]\nbase_url = "{base_url}"\napi_key = "benchmark-key"\n'
        'embedding_model = "embeddinggemma-2"\nextraction_model = "gemma-4-e4b"\nqa_model = "gemma-4-e4b"\n'
        "context_window_tokens = 32768\nmax_output_tokens = 1024\nenable_gbnf = true\n"
        "[ingestion]\nchunk_size = 256\nchunk_overlap = 32\nmin_chunk_size = 24\n"
        '[retrieval]\nretriever_mode = "manifold"\n',
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    study_db = tmp_path / "optuna_pareto.db"
    result = tune_manifold(
        cfg,
        datasets=("sample",),
        query_offset=10,
        max_queries=4,
        train_queries=2,
        index_queries=4,
        trials=10,
        study_db=study_db,
        study_name="pareto_study",
        seed=42,
        embedder=custom_emb,
    )
    assert result["study"]["completed_trials"] == 10
    assert result["best_source"] == "optuna_pareto"
    assert result["best_overrides"] != {}
    assert sum(result["study"]["param_importances"].values()) > 0.0
    assert "stage_a" not in result
    assert "stage_b" not in result
