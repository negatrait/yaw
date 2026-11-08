"""CocoIndex-owned inference operations; no graph or database side effects."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

import cocoindex as coco
import dspy
import numpy as np

from yaw.core import ChunkPointer, OpenIERecord, SHACLViolation, SynonymEvidenceBatch

from .embedding import DSPyEmbeddingAdapter, EmbeddingRequest, EmbeddingResult


@dataclass(frozen=True, slots=True)
class EmbeddedText:
    vector_f32: bytes
    request_id: str
    telemetry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChunkInference:
    chunk_id: str
    source_hash: str
    program_identity: str
    embedding_identity: str
    extraction_json: str
    traces_json: str


@dataclass(frozen=True, slots=True)
class ChunkProcessOutput:
    pointer: ChunkPointer
    text: str
    inference: ChunkInference
    embedding: bytes

    @property
    def chunk_id(self) -> str:
        return self.pointer.chunk_id

    @property
    def vector_f32(self) -> bytes:
        return self.embedding


@dataclass
class EmbeddingEndpoint:
    adapter: DSPyEmbeddingAdapter | Any
    admission: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))

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
                if hasattr(self.adapter, "aexecute_request"):
                    result = await self.adapter.aexecute_request(request)
                else:
                    encode_fn = self.adapter.encode
                    vecs = await asyncio.to_thread(
                        encode_fn, list(request.texts), instruction=request.instruction or ""
                    )
                    result = EmbeddingResult(
                        np.asarray(vecs, dtype=np.float32),
                        {"stage": request.stage or "embed", "tokens_used": 0},
                    )
            request_id = uuid4().hex
            for index, vector in zip(indices, result.vectors, strict=True):
                results[index] = EmbeddedText(vector.astype("<f4", copy=False).tobytes(), request_id, result.telemetry)
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
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class MissingEntityInput:
    chunk_id: str
    passage: str
    missing_entities: tuple[str, ...]
    triples: tuple[dict[str, str], ...]
    program_identity: str


@dataclass(frozen=True, slots=True)
class DefinedEntities:
    definitions: tuple[tuple[str, str], ...]
    traces: tuple[PredictorTrace, ...]
    telemetry: tuple[dict[str, Any], ...]
    audit_trail: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class SynonymJudgeInput:
    batch: SynonymEvidenceBatch
    program_identity: str
    synonym_identity: str = ""

    def __post_init__(self) -> None:
        if self.batch.construction_seconds != 0.0:
            object.__setattr__(self, "batch", replace(self.batch, construction_seconds=0.0))


@dataclass(frozen=True, slots=True)
class JudgedSynonyms:
    decisions: tuple[dict[str, str], ...]
    traces: tuple[PredictorTrace, ...]
    telemetry: tuple[dict[str, Any], ...]
    audit_trail: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class SHACLHealInput:
    violations: tuple[SHACLViolation, ...]
    focus_dossiers: dict[str, dict[str, Any]]
    focus_context: str
    program_identity: str
    shacl_identity: str = ""


@dataclass(frozen=True, slots=True)
class HealedSHACL:
    repairs: tuple[dict[str, str], ...]
    traces: tuple[PredictorTrace, ...]
    telemetry: tuple[dict[str, Any], ...]
    audit_trail: tuple[dict[str, Any], ...] = ()


@dataclass
class OpenIEEndpoint:
    program: Any
    lm: Any = None
    program_identity: str = ""
    admission: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))
    progress_callback: Callable[[str], None] | None = None

    def _resolve_lm(self) -> Any:
        active_lm = self.lm or getattr(dspy.settings, "lm", None)
        return active_lm.copy() if active_lm is not None and hasattr(active_lm, "copy") else active_lm

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
        program.entity_def_audit_trail.clear()
        self.program.telemetry.clear()
        self.program.entity_def_audit_trail.clear()
        lm = self._resolve_lm()
        traces: list[Any] = []
        with dspy.context(lm=lm, trace=traces):
            record = program.extract_openie(request.chunk_id, request.passage, lm=lm)
        names = {id(predictor): name for src in (self.program, program) for name, predictor in src.named_predictors()}
        telemetry_src = program.telemetry or self.program.telemetry
        audit_src = program.entity_def_audit_trail or self.program.entity_def_audit_trail
        return ExtractedText(
            record,
            tuple(
                PredictorTrace(names[id(predictor)], dict(inputs), dict(outputs))
                for predictor, inputs, outputs in traces
                if id(predictor) in names
            ),
            tuple(dict(entry) for entry in telemetry_src),
            tuple(dict(entry) for entry in audit_src),
        )

    @coco.fn.as_async
    def define_entities(self, /, request: MissingEntityInput) -> DefinedEntities:
        if request.program_identity != self.program_identity:
            raise ValueError("OpenIE program identity does not match the bound program")
        program = self.program.deepcopy()
        program.telemetry.clear()
        program.entity_def_audit_trail.clear()
        lm = self._resolve_lm()
        traces: list[Any] = []
        with dspy.context(lm=lm, trace=traces):
            defs = program.define_missing_entities(
                request.passage,
                request.missing_entities,
                request.triples,
                lm=lm,
            )
        names = {id(predictor): name for src in (self.program, program) for name, predictor in src.named_predictors()}
        return DefinedEntities(
            tuple((str(k), str(v)) for k, v in defs),
            tuple(
                PredictorTrace(names[id(predictor)], dict(inputs), dict(outputs))
                for predictor, inputs, outputs in traces
                if id(predictor) in names
            ),
            tuple(dict(entry) for entry in program.telemetry),
            tuple(dict(entry) for entry in program.entity_def_audit_trail),
        )

    @coco.fn.as_async
    def judge_synonyms(self, /, request: SynonymJudgeInput) -> JudgedSynonyms:
        if request.program_identity != self.program_identity:
            raise ValueError("OpenIE program identity does not match the bound program")
        program = self.program.deepcopy()
        program.telemetry.clear()
        program.label_judge_audit_trail.clear()
        program.hierarchy_judge_audit_trail.clear()
        program.related_judge_audit_trail.clear()
        lm = self._resolve_lm()
        traces: list[Any] = []
        with dspy.context(lm=lm, trace=traces):
            decisions = program.judge_synonyms(request.batch, lm=lm)
        names = {id(predictor): name for src in (self.program, program) for name, predictor in src.named_predictors()}
        audit_trail = [
            *program.label_judge_audit_trail,
            *program.hierarchy_judge_audit_trail,
            *program.related_judge_audit_trail,
        ]
        return JudgedSynonyms(
            tuple({str(k): str(v) for k, v in d.items()} for d in decisions),
            tuple(
                PredictorTrace(names[id(predictor)], dict(inputs), dict(outputs))
                for predictor, inputs, outputs in traces
                if id(predictor) in names
            ),
            tuple(dict(entry) for entry in program.telemetry),
            tuple(dict(entry) for entry in audit_trail),
        )

    @coco.fn.as_async
    def heal_shacl(self, /, request: SHACLHealInput) -> HealedSHACL:
        if request.program_identity != self.program_identity:
            raise ValueError("OpenIE program identity does not match the bound program")
        program = self.program.deepcopy()
        program.telemetry.clear()
        program.shacl_audit_trail.clear()
        program.set_graph_tools(
            node_inspector=lambda fn: dict(request.focus_dossiers.get(fn, {})),
            pair_inspector=lambda a, b: str(
                request.focus_dossiers.get(
                    f"{a}|{b}",
                    request.focus_dossiers.get(f"{b}|{a}", {}),
                ).get("markdown_dossier", "")
            ),
            shacl_linter=lambda: [v.to_dict() for v in request.violations],
            chunk_hydrator=program._chunk_hydrator,
            ast_island_hydrator=program._ast_island_hydrator,
        )
        lm = self._resolve_lm()
        traces: list[Any] = []
        with dspy.context(lm=lm, trace=traces):
            repairs = program.heal_shacl_violations(
                request.violations,
                focus_context=request.focus_context,
                mode="direct" if request.focus_context.strip() else "rlm",
            )
        names = {id(predictor): name for src in (self.program, program) for name, predictor in src.named_predictors()}
        return HealedSHACL(
            tuple({str(k): str(v) for k, v in r.items()} for r in repairs),
            tuple(
                PredictorTrace(names[id(predictor)], dict(inputs), dict(outputs))
                for predictor, inputs, outputs in traces
                if id(predictor) in names
            ),
            tuple(dict(entry) for entry in program.telemetry),
            tuple(dict(entry) for entry in program.shacl_audit_trail),
        )


OPENIE_ENDPOINT = coco.ContextKey[OpenIEEndpoint]("yaw/openie_endpoint")


@coco.fn(memo=True)
async def extract_text(request: OpenIEInput) -> ExtractedText:
    endpoint = coco.use_context(OPENIE_ENDPOINT)
    async with endpoint.admission:
        return await endpoint.extract(request)


@coco.fn(memo=True)
async def define_missing_entities(request: MissingEntityInput) -> DefinedEntities:
    endpoint = coco.use_context(OPENIE_ENDPOINT)
    async with endpoint.admission:
        return await endpoint.define_entities(request)


@coco.fn(memo=True)
async def judge_synonyms(request: SynonymJudgeInput) -> JudgedSynonyms:
    endpoint = coco.use_context(OPENIE_ENDPOINT)
    async with endpoint.admission:
        return await endpoint.judge_synonyms(request)


@coco.fn(memo=True, version=5)
async def heal_shacl_violations(request: SHACLHealInput) -> HealedSHACL:
    endpoint = coco.use_context(OPENIE_ENDPOINT)
    async with endpoint.admission:
        return await endpoint.heal_shacl(request)
