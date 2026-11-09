"""Separate-process client for the loopback candidate-cache integration test."""

import argparse
import inspect
import json
from pathlib import Path

import dspy
import numpy as np

from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import PASSAGE_INSTRUCTION, DSPyEmbeddingAdapter, YawProgram
from yaw.daemon.indexing import bind_native_indexing
from yaw.daemon.ingestion import prepare_dataset_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("url")
    parser.add_argument("--mutation", default="none")
    args = parser.parse_args()
    lm = dspy.LM(
        "openai/cache-test-v2" if args.mutation == "model" else "openai/cache-test",
        api_base=args.url,
        api_key="test",
        cache=False,
        temperature=0.3 if args.mutation == "sampling" else 0.0,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)
    program = YawProgram(
        enable_gbnf=False,
        orchestrator_lm=lm,
        rlm_sub_lm=lm,
        rlm_max_iters=2,
    )
    if args.mutation == "prompt":
        program.ner.signature = program.ner.signature.with_instructions(
            program.ner.signature.instructions + " Preserve every named person."
        )
    embedder = DSPyEmbeddingAdapter(base_url=args.url, api_key="test", caching=False)
    engine = bind_native_indexing(
        YawGraphEngine(
            db_path=args.root / "yaw.db",
            workspace_root=args.root / "datasets",
            program=program,
            embedder=embedder,
            synonymy_threshold=-1.0,
        )
    )
    try:
        if args.mutation == "query":
            old_vector = engine._encode_texts(
                ["Where did Turing work?"],
                instruction=PASSAGE_INSTRUCTION,
            )[0]
            program.embedding_query_prefix += "A different query-only task. "
            program.embedding_fact_prefix += "A different fact-query task. "
            engine._apply_program_embedding_prefixes(program)
            new_vector = engine._encode_texts(
                ["Where did Turing work?"],
                instruction=PASSAGE_INSTRUCTION,
            )[0]
            assert not np.array_equal(old_vector, new_vector)
            np.testing.assert_array_equal(
                new_vector,
                engine._encode_texts(
                    ["Where did Turing work?"],
                    instruction=PASSAGE_INSTRUCTION,
                )[0],
            )
        doc = "Alan Turing formulated the Turing machine in 1936."
        if args.mutation == "source":
            doc += " This source was edited after the previous run."
        source, questions = args.root / "source.json", args.root / "questions.json"
        source.write_text(
            json.dumps(
                [
                    {"idx": "first", "title": "", "text": doc},
                    {"idx": "second", "title": "", "text": "Alan M. Turing worked on cryptanalysis at Bletchley Park."},
                ]
            ),
            encoding="utf-8",
        )
        questions.write_text("[]", encoding="utf-8")
        bundle = prepare_dataset_bundle("cache_documents", source, questions, args.root / "datasets")
        engine.workspace_root = bundle.root
        matched_files = tuple(doc.path for doc in bundle.documents if doc.document_id in ["first", "second"])
        engine.index_chunks(included_files=matched_files)
        assert engine.lint_graph()["conforms"]
        print(
            json.dumps(
                {
                    "tokens": {
                        key: engine._graph_rlm_stats[key]
                        for key in (
                            "label_judge_tokens",
                            "hierarchy_judge_tokens",
                            "related_judge_tokens",
                            "shacl_healer_tokens",
                            "missing_entity_defs_tokens",
                        )
                    },
                    "openie_tokens": [stats["openie_tokens"] for _, stats in sorted(engine._chunk_rlm_stats.items())],
                    "trace_names": sorted({name for name, _, _ in engine._captured_graph_traces}),
                    "namespace": engine._embedding_cache_namespace(),
                    "embedding_kwargs": embedder.default_kwargs,
                    "implementation_root": str(Path(inspect.getfile(YawGraphEngine)).parent),
                }
            )
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
