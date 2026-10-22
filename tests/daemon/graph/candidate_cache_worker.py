"""Separate-process client for the loopback candidate-cache integration test."""

import argparse
import json
import sys
from pathlib import Path

import dspy
import numpy as np

from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import PASSAGE_INSTRUCTION, DSPyEmbeddingAdapter, HippoRAGProgram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("url")
    parser.add_argument("--mutation", default="none")
    args = parser.parse_args()
    lm = dspy.LM(
        "openai/cache-test-v2" if args.mutation == "model" else "openai/cache-test",
        api_base=args.url, api_key="test", cache=False,
        temperature=0.3 if args.mutation == "sampling" else 0.0, max_tokens=1000,
    )
    dspy.configure(lm=lm)
    program = HippoRAGProgram(
        enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm, rlm_max_iters=2,
    )
    if args.mutation == "prompt":
        program.ner.signature = program.ner.signature.with_instructions(
            program.ner.signature.instructions + " Preserve every named person."
        )
    embedder = DSPyEmbeddingAdapter(base_url=args.url, api_key="test", caching=False)
    engine = YawGraphEngine(
        db_path=args.root / "yaw.db", workspace_root=args.root,
        program=program, embedder=embedder, synonymy_threshold=-1.0,
    )
    try:
        if args.mutation == "query":
            old_vector = engine._encode_texts(
                ["Where did Turing work?"], instruction=PASSAGE_INSTRUCTION,
            )[0]
            program.embedding_query_prefix += "A different query-only task. "
            program.embedding_fact_prefix += "A different fact-query task. "
            engine._apply_program_embedding_prefixes(program)
            new_vector = engine._encode_texts(
                ["Where did Turing work?"], instruction=PASSAGE_INSTRUCTION,
            )[0]
            assert not np.array_equal(old_vector, new_vector)
            np.testing.assert_array_equal(
                new_vector,
                engine._encode_texts(
                    ["Where did Turing work?"], instruction=PASSAGE_INSTRUCTION,
                )[0],
            )
        # Force materialization from scratch while retaining only the disk stage
        # cache. A second process cannot benefit from graph rows or Python RAM.
        engine._reset_sandbox_graph(clear_openie=True)
        doc = "Alan Turing formulated the Turing machine in 1936."
        if args.mutation == "source":
            doc += " This source was edited after the previous run."
        engine.index([doc])
        with engine.managed_conn.transaction() as conn:
            conn.execute(
                "UPDATE entities SET definition = '' "
                "WHERE name IN ('alan turing', 'turing machine')"
            )
        assert not engine.lint_graph()["conforms"]
        engine.index(["Alan M. Turing worked on cryptanalysis at Bletchley Park."])
        assert engine.lint_graph()["conforms"]
        print(json.dumps({
            "tokens": {key: engine._graph_rlm_stats[key] for key in (
                "synonym_judge_tokens", "shacl_healer_tokens", "missing_entity_defs_tokens",
            )},
            "openie_tokens": [
                stats["openie_tokens"] for _, stats in sorted(engine._chunk_rlm_stats.items())
            ],
            "hits": {cache.stage: cache.hits for cache in (
                engine._text_vec_cache, engine._openie_fp_cache,
                engine._synonym_fp_cache, engine._shacl_fp_cache,
            )},
            "cache_rows": engine.conn.execute(
                "SELECT COUNT(*) FROM candidate_stage_cache"
            ).fetchone()[0],
            "trace_names": sorted({name for name, _, _ in engine._captured_graph_traces}),
            "namespace": engine._candidate_cache_namespace(),
            "code_fp": engine._candidate_cache_code_fp,
            "embedding_kwargs": embedder.default_kwargs,
            "implementation_root": str(
                Path(sys.modules[YawGraphEngine.__module__].__file__).parent
            ),
        }))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
