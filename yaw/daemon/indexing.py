"""Native source and inference target composition at the daemon boundary."""

import asyncio
import json
from dataclasses import asdict, dataclass

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.daemon.graph.inference.components import (
    EMBEDDING_ENDPOINT,
    OPENIE_ENDPOINT,
    OpenIEInput,
    embed_text,
    extract_text,
)
from yaw.daemon.ingestion.source_targets import (
    SOURCE_DB,
    SOURCE_ROOT,
    SourceChunk,
    SourceSettings,
    declare_source_file,
)


@dataclass(frozen=True, slots=True)
class InferenceIdentity:
    program: str
    embedding: str


@dataclass(frozen=True, slots=True)
class ChunkInference:
    chunk_id: str
    source_hash: str
    program_identity: str
    embedding_identity: str
    vector_json: str
    extraction_json: str
    traces_json: str


@coco.fn
async def process_chunk(
    chunk: SourceChunk, target: sqlite.TableTarget[ChunkInference], identity: InferenceIdentity
) -> None:
    """Declare a complete inference artifact only after both operations succeed."""
    embeddings = coco.use_context(EMBEDDING_ENDPOINT)
    extraction = coco.use_context(OPENIE_ENDPOINT)
    if (
        identity.program != extraction.program_identity
        or identity.embedding != embeddings.adapter.profile.index_identity
    ):
        raise ValueError("Indexing identity does not match the bound inference resources")
    vector, record = await asyncio.gather(
        embed_text(embeddings.adapter.prepare_request([chunk.text], batch_size=64, stage="chunk_embed")),
        extract_text(OpenIEInput(chunk.chunk_id, chunk.text, extraction.program_identity)),
    )
    target.declare_row(
        row=ChunkInference(
            chunk.chunk_id,
            chunk.source_hash,
            extraction.program_identity,
            embeddings.adapter.profile.index_identity,
            json.dumps(vector.vector),
            json.dumps(asdict(record.record)),
            json.dumps([asdict(trace) for trace in record.traces]),
        )
    )


@coco.fn
async def process_file(
    file: FileLike,
    sources: sqlite.TableTarget[SourceChunk],
    artifacts: sqlite.TableTarget[ChunkInference],
    settings: SourceSettings,
    identity: InferenceIdentity,
) -> None:
    chunks = await declare_source_file(file, sources, settings)
    for chunk in chunks:
        await coco.mount(coco.component_subpath("chunk", chunk.chunk_id), process_chunk, chunk, artifacts, identity)


@coco.fn
async def indexing_app_main(settings: SourceSettings, identity: InferenceIdentity) -> None:
    sources = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceChunk].from_class(SourceChunk, primary_key=["chunk_id"]),
    )
    artifacts = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="chunk_inference",
        table_schema=await sqlite.TableSchema[ChunkInference].from_class(ChunkInference, primary_key=["chunk_id"]),
    )
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        live=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=list(settings.included_patterns),
            excluded_patterns=list(settings.excluded_patterns),
        ),
    )
    await coco.mount_each(process_file, files.items(), sources, artifacts, settings, identity)
