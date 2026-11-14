import json
import os
import time
from pathlib import Path

from yaw.core.config import load_config
from yaw.daemon.graph.inference.program import YawProgram
from yaw.daemon.ingestion.corpus import chunk_source_text

assert os.environ.get("UNSLOTH_STUDIO_AUTH_TOKEN"), "export UNSLOTH_STUDIO_AUTH_TOKEN first"
cfg = load_config("yaw.toml")
print("base_url:", cfg.base_url, "| models:", cfg.extraction_model, cfg.qa_model, "| key set:", bool(cfg.api_key))
program = YawProgram.from_config(cfg)
assert program.orchestrator_lm is not None
print(
    "worker LM engine:",
    type(program.orchestrator_lm.engine).__name__,
    "| sub LM:",
    type(program.rlm_sub_lm.engine).__name__ if program.rlm_sub_lm else None,
)

documents = sorted(Path("outputs/datasets/enterprise_rag_bench/documents").glob("*.json"))[:3]
total_tokens = 0
total_seconds = 0.0
for index, path in enumerate(documents):
    content = path.read_text(encoding="utf-8")
    chunk, text = chunk_source_text(
        content,
        f"enterprise_rag_bench/documents/{path.name}",
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chunk_size=cfg.min_chunk_size,
    )[0]
    start = time.perf_counter()
    record = program.extract_openie(chunk.chunk_id, text)
    elapsed = time.perf_counter() - start
    stats = getattr(record, "rlm_stats", None) or {}
    print(f"chunk {index}: {elapsed:.1f}s entities={len(record.entities)} triples={len(record.triples)}")
    for triple in list(record.triples)[:3]:
        print("   ", triple)
    total_seconds += elapsed
print(json.dumps({"seconds": round(total_seconds, 1)}))
