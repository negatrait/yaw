"""CocoIndex-owned inference operations; no graph or database side effects."""

import asyncio
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

import cocoindex as coco
import dspy

from yaw.core import OpenIERecord

from .embedding import DSPyEmbeddingAdapter, EmbeddingRequest
from .program import HippoRAGProgram


@dataclass(frozen=True, slots=True)
class EmbeddedText:
    vector: tuple[float, ...]
    request_id: str
    telemetry: dict[str, Any]


@dataclass
class EmbeddingEndpoint:
    adapter: DSPyEmbeddingAdapter
    admission: asyncio.Semaphore

    @coco.fn.as_async(batching=True, max_batch_size=64)
    async def embed(self, requests: list[EmbeddingRequest]) -> list[EmbeddedText]:
        """Batch compatible contracts without database or shared history writes."""
        groups: dict[EmbeddingRequest, list[int]] = {}
        for index, request in enumerate(requests):
            if len(request.texts) != 1:
                raise ValueError("Native embedding calls require exactly one text")
            groups.setdefault(replace(request, texts=()), []).append(index)
        results: dict[int, EmbeddedText] = {}
        for contract, indices in groups.items():
            request = replace(
                contract,
                texts=tuple(requests[index].texts[0] for index in indices),
                batch_size=min(contract.batch_size, 64),
            )
            async with self.admission:
                result = await self.adapter.aexecute_request(request)
            request_id = uuid4().hex
            for index, vector in zip(indices, result.vectors, strict=True):
                results[index] = EmbeddedText(tuple(float(value) for value in vector), request_id, result.telemetry)
        return [results[index] for index in range(len(requests))]


EMBEDDING_ENDPOINT = coco.ContextKey[EmbeddingEndpoint]("yaw/embedding_endpoint")


@coco.fn(memo=True)
async def embed_text(request: EmbeddingRequest) -> EmbeddedText:
    """Persist each semantic result independently of its provider batch."""
    return await coco.use_context(EMBEDDING_ENDPOINT).embed(request)


@dataclass(frozen=True, slots=True)
class OpenIEInput:
    chunk_id: str
    passage: str
    program_identity: str


@dataclass(frozen=True, slots=True)
class PredictorTrace:
    predictor: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractedText:
    record: OpenIERecord
    traces: tuple[PredictorTrace, ...]
    telemetry: tuple[dict[str, Any], ...]


@dataclass
class OpenIEEndpoint:
    program: HippoRAGProgram
    lm: dspy.LM
    program_identity: str
    admission: asyncio.Semaphore

    @coco.fn.as_async
    def extract(self, /, request: OpenIEInput) -> ExtractedText:
        """Run the existing synchronous semantic pipeline on an isolated program.

        CocoIndex owns the synchronous-function runner. The per-call DSPy context
        and module copy prevent predictor traces and histories crossing calls.
        """
        if request.program_identity != self.program_identity:
            raise ValueError("OpenIE program identity does not match the bound program")
        program = self.program.deepcopy()
        program.telemetry.clear()
        lm = self.lm.copy()
        traces: list[Any] = []
        with dspy.context(lm=lm, trace=traces):
            record = program.extract_openie(request.chunk_id, request.passage, lm=lm)
        names = {id(predictor): name for name, predictor in program.named_predictors()}
        return ExtractedText(
            record,
            tuple(
                PredictorTrace(names[id(predictor)], dict(inputs), dict(outputs))
                for predictor, inputs, outputs in traces
            ),
            tuple(dict(entry) for entry in program.telemetry),
        )


OPENIE_ENDPOINT = coco.ContextKey[OpenIEEndpoint]("yaw/openie_endpoint")


@coco.fn(memo=True)
async def extract_text(request: OpenIEInput) -> ExtractedText:
    endpoint = coco.use_context(OPENIE_ENDPOINT)
    async with endpoint.admission:
        return await endpoint.extract(request)
