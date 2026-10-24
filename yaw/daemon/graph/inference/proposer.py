"""RLM evidence selection and SIMBA-optimized native DSPy instruction proposer."""

import copy
import json
import logging
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import dspy
from dspy.utils.exceptions import AdapterParseError, LMError

from yaw.core.config import DEFAULT_CONTEXT_WINDOW_TOKENS, scale_context_budget

from .adapter import YawGBNFAdapter, sum_usage_tokens
from .diagnostics import DiagnosticHistory, diagnostic_boundary, trim_diagnostics
from .invariants import STAGE_SCHEMA_CONTRACTS
from .prompts import (
    CompressInstructionSignature,
    ProposeInstructionSignature,
    ReflectiveInnerDistillerSignature,
)
from .reflection import (
    ReflectionBudgetError,
    ReflectionLM,
    checkpoint_object,
    deterministic_evidence,
    reflection_digest,
    reflection_examples,
    resolve_evidence,
    validate_request_checkpoint,
    validate_simba_state,
)
from .rlm import STAGE_RLM_DELEGATION_RULES, PositionalSafeRLM, build_rlm_action_demos


class InstructionSynthesizerWorker(dspy.Module):
    """Inner GBNF instruction-synthesis worker module of `RLMInstructionProposer`, optimized by `dspy.SIMBA`."""

    def __init__(
        self,
        *,
        qwen_proposer: Any,
        qwen_compressor: Any,
        worker_lm: Any = None,
        orchestrator_lm: Any = None,
        enable_gbnf: bool = True,
        max_instruction_chars: int | None = 1500,
        max_field_chars: int = 600,
        worker_temperature: float = 0.7,
        json_adapter: dspy.Adapter | None = None,
    ) -> None:
        super().__init__()
        self.qwen_proposer = qwen_proposer
        self.qwen_compressor = qwen_compressor
        self.worker_lm = worker_lm
        self.orchestrator_lm = orchestrator_lm
        self.enable_gbnf = bool(enable_gbnf)
        self.max_instruction_chars = max_instruction_chars
        self.max_field_chars = max_field_chars
        self.worker_temperature = float(worker_temperature)
        self.json_adapter: dspy.Adapter = json_adapter or dspy.JSONAdapter()
        self._rollout_lm: Any = None
        self.sub_telemetry: list[dict[str, Any]] = []

    def get_lm(self) -> Any:
        return self._rollout_lm or self.worker_lm or self.orchestrator_lm or getattr(self.qwen_proposer, "lm", None)

    def set_lm(self, lm: Any) -> None:
        self._rollout_lm = lm

    def __deepcopy__(self, memo: dict[int, Any]) -> InstructionSynthesizerWorker:
        cls = self.__class__
        new_mod = cls.__new__(cls)
        memo[id(self)] = new_mod
        for k, v in self.__dict__.items():
            if k in ("worker_lm", "orchestrator_lm", "json_adapter", "_rollout_lm"):
                setattr(new_mod, k, v)
            else:
                setattr(new_mod, k, copy.deepcopy(v, memo))
        return new_mod

    def forward(
        self,
        current_instruction: str = "",
        examples_with_feedback: Any = None,
        component_name: str = "",
    ) -> dspy.Prediction:
        from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

        native = InstructionProposer(
            base_instructions=ProposeInstructionSignature.__doc__,
            max_chars=self.max_instruction_chars,
            adapter=self.json_adapter,
        )
        native.propose = self.qwen_proposer
        native.compress = self.qwen_compressor if self.max_instruction_chars else None
        examples = self.fit_evidence(current_instruction, examples_with_feedback, component_name)
        stage = component_name.removesuffix(".predict")
        contract = STAGE_SCHEMA_CONTRACTS.get(stage, ("", (), "Preserve the existing output schema."))[2]
        inputs = {
            "examples_with_feedback": [
                {"component": component_name, "contract": contract},
                *examples,
            ]
        }
        lm = self._rollout_lm or self.worker_lm or self.orchestrator_lm or dspy.settings.lm
        with dspy.context(lm=lm, adapter=self.json_adapter):
            result = native(
                {component_name: current_instruction},
                {component_name: inputs["examples_with_feedback"]},
                [component_name],
            )[component_name]
        return dspy.Prediction(new_instruction=result)

    def fit_evidence(
        self, current_instruction: str, examples_with_feedback: Any, component_name: str
    ) -> list[dict[str, Any]]:
        examples = list(examples_with_feedback or [])
        stage = component_name.removesuffix(".predict")
        contract = STAGE_SCHEMA_CONTRACTS.get(stage, ("", (), "Preserve the existing output schema."))[2]
        mandatory = {"component": component_name, "contract": contract}
        lm = self._rollout_lm or self.worker_lm or self.orchestrator_lm or dspy.settings.lm
        inputs = {
            "current_instruction": current_instruction,
            "examples_with_feedback": [mandatory, *examples],
        }
        if self.max_instruction_chars is not None:
            inputs["length_limit"] = f"The new instruction must be at most {self.max_instruction_chars} characters."
        while isinstance(lm, ReflectionLM):
            try:
                lm.count(self.json_adapter.format(self.qwen_proposer.signature, self.qwen_proposer.demos, inputs))
                break
            except ReflectionBudgetError:
                if not examples:
                    raise
                examples.pop()
                inputs["examples_with_feedback"] = [mandatory, *examples]
        return examples


class RLMInstructionProposer:
    """RLM evidence selection followed by a SIMBA-optimized native DSPy proposer.

    Conforms to `gepa.core.adapter.ProposalFn` and stock DSPy `InstructionProposer`:
    `(candidate, reflective_dataset, components_to_update) -> dict[str, str]`, while its
    inner worker (`self.worker_module`: `InstructionSynthesizerWorker(dspy.Module)` wrapping
    `qwen_proposer` + `qwen_compressor`) is continuously refined by an always-on `dspy.SIMBA`
    meta-optimizer during prompt evolution.
    """

    def __init__(
        self,
        *,
        orchestrator_lm: Any = None,
        worker_lm: Any = None,
        enable_gbnf: bool = True,
        enable_simba: bool = True,
        simba_bsize: int = 2,
        simba_num_candidates: int = 2,
        simba_max_steps: int = 1,
        simba_max_demos: int = 2,
        max_instruction_chars: int | Literal["auto"] | None = "auto",
        max_field_chars: int | None = None,
        max_feedback_examples: int = 5,
        reflection_temperature: float = 0.8,
        worker_temperature: float = 0.7,
        rlm_max_iters: int = 10,
        rlm_max_llm_calls: int = 20,
        rlm_max_output_chars: int | None = None,
        include_demos: bool = False,
        adapter: dspy.Adapter | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.PythonInterpreter,
        task_scorer: Callable[[Mapping[str, str]], float] | None = None,
        save_simba_state: Callable[[dict[str, Any]], None] | None = None,
        reflection_input_tokens: int | None = None,
        token_counter: Any = None,
        checkpoint_get: Callable[[str], Any] | None = None,
        checkpoint_put: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.context_window_tokens = (
            getattr(orchestrator_lm, "_yaw_context_window_tokens", None)
            or getattr(worker_lm, "_yaw_context_window_tokens", None)
            or getattr(token_counter, "context_tokens", None)
            or DEFAULT_CONTEXT_WINDOW_TOKENS
        )
        if max_instruction_chars == "auto":
            max_instruction_chars = scale_context_budget(1500, self.context_window_tokens)
        if max_field_chars is None:
            max_field_chars = scale_context_budget(600, self.context_window_tokens)
        if rlm_max_output_chars is None:
            rlm_max_output_chars = scale_context_budget(10000, self.context_window_tokens)
        if reflection_input_tokens is None:
            reflection_input_tokens = scale_context_budget(16384, self.context_window_tokens)
        if max_instruction_chars is not None and max_instruction_chars <= 0:
            raise ValueError(f"max_instruction_chars must be > 0 or None, got {max_instruction_chars!r}.")

        if reflection_input_tokens < 1:
            raise ValueError("reflection_input_tokens must be positive")
        self.reflection_input_tokens = reflection_input_tokens
        self.token_counter = token_counter
        self.checkpoint_get = checkpoint_get
        self.checkpoint_put = checkpoint_put
        self.orchestrator_lm = orchestrator_lm
        self.worker_lm = worker_lm
        self.enable_gbnf = bool(enable_gbnf)
        self.enable_simba = bool(enable_simba)
        self.simba_bsize = max(1, int(simba_bsize))
        self.simba_num_candidates = max(1, int(simba_num_candidates))
        self.simba_max_steps = max(1, int(simba_max_steps))
        self.simba_max_demos = max(0, int(simba_max_demos))
        self.max_instruction_chars = int(max_instruction_chars) if max_instruction_chars is not None else None
        self.max_field_chars = max(120, int(max_field_chars))
        self.max_feedback_examples = min(5, max(1, int(max_feedback_examples)))
        self.reflection_temperature = float(reflection_temperature)
        self.worker_temperature = float(worker_temperature)
        self.json_adapter: dspy.Adapter = adapter or YawGBNFAdapter(
            enable_gbnf=self.enable_gbnf,
            max_instruction_chars=self.max_instruction_chars,
        )
        self.telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self.simba_telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self._simba_steps = 0
        self.task_scorer = task_scorer
        self.save_simba_state = save_simba_state
        self._simba_reward_errors = 0
        self._simba_examples: list[dspy.Example] = []
        self._sub_telemetry: list[dict[str, Any]] = []

        from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

        native = InstructionProposer(
            base_instructions=ProposeInstructionSignature.__doc__,
            max_chars=self.max_instruction_chars,
        )
        self.qwen_proposer = native.propose
        # Keep both native predictors visible to SIMBA and state serialization.
        self.qwen_compressor = native.compress or dspy.Predict(CompressInstructionSignature)
        self.worker_module = InstructionSynthesizerWorker(
            qwen_proposer=self.qwen_proposer,
            qwen_compressor=self.qwen_compressor,
            worker_lm=self.worker_lm,
            orchestrator_lm=self.orchestrator_lm,
            enable_gbnf=self.enable_gbnf,
            max_instruction_chars=self.max_instruction_chars,
            max_field_chars=self.max_field_chars,
            worker_temperature=self.worker_temperature,
            json_adapter=self.json_adapter,
        )

        self.inner_rlm = PositionalSafeRLM(
            ReflectiveInnerDistillerSignature,
            stage_rule=STAGE_RLM_DELEGATION_RULES["instruction_proposer_inner"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=[],
            sub_lm=worker_lm or orchestrator_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.inner_rlm.generate_action.config = {"temperature": self.reflection_temperature}
        self.inner_rlm.extract.config = {"temperature": self.reflection_temperature}
        self.rlm = self.inner_rlm

        self._bind_predictor_lms()

        if include_demos:
            rlm_demos = build_rlm_action_demos()
            self.inner_rlm.generate_action.demos = list(rlm_demos["instruction_proposer_inner"])

    def _bind_predictor_lms(self) -> None:
        """Use native contextual writer LMs and a role-bound evidence selector."""
        # The native proposer runs under the worker's selected rollout context.
        # Leaving its predictors unbound makes DSPy's normal context precedence
        # apply, including the verified ReflectionLM token-budget guard.
        self.qwen_proposer.set_lm(None)
        self.qwen_compressor.set_lm(None)
        self.worker_module.qwen_proposer = self.qwen_proposer
        self.worker_module.qwen_compressor = self.qwen_compressor
        self.worker_module.worker_lm = self.worker_lm
        self.worker_module.orchestrator_lm = self.orchestrator_lm

        self.inner_rlm.set_lm(self.orchestrator_lm or self.worker_lm)

    def get_lm(self) -> Any:
        return self.worker_module.get_lm()

    def set_lm(self, lm: Any) -> None:
        self.worker_module.set_lm(lm)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        callbacks = {
            "orchestrator_lm",
            "worker_lm",
            "json_adapter",
            "token_counter",
            "task_scorer",
            "save_simba_state",
            "checkpoint_get",
            "checkpoint_put",
        }
        for key, value in self.__dict__.items():
            setattr(new, key, value if key in callbacks else copy.deepcopy(value, memo))
        return new

    def preflight(self) -> None:
        """Require an exact tokenizer before any reflection or meta-learning call."""
        if self.token_counter is None:
            raise ReflectionBudgetError("Reflection requires a verified serving tokenizer")
        for name in ("worker_lm", "orchestrator_lm"):
            lm = getattr(self, name)
            if lm is not None and not isinstance(lm, ReflectionLM):
                setattr(
                    self,
                    name,
                    ReflectionLM(
                        lm,
                        self.token_counter,
                        self.reflection_input_tokens,
                        getattr(lm, "_yaw_context_window_tokens", None) or self.context_window_tokens,
                        int(lm.kwargs.get("max_tokens") or scale_context_budget(4096, self.context_window_tokens)),
                    ),
                )
        self.inner_rlm.sub_lm = self.worker_lm or self.orchestrator_lm
        self._bind_predictor_lms()

    def all_reflection_predictors(self) -> list[tuple[str, Any]]:
        """Return all 3 active reflection predictors (`inner_rlm`, `qwen_proposer`, `qwen_compressor`)."""
        return [
            ("inner_rlm", self.inner_rlm),
            ("qwen_proposer", self.qwen_proposer),
            ("qwen_compressor", self.qwen_compressor),
        ]

    def dump_simba_state(self) -> dict[str, Any]:
        """JSON-safe DSPy predictor state; exclude clients, traces and optimizer archives."""
        predictors = {}
        for name in ("qwen_proposer", "qwen_compressor"):
            state = getattr(self, name).dump_state()
            state.update(lm=None, traces=[], train=[])
            predictors[name] = state
        return {
            "version": 2,
            "steps": self._simba_steps,
            "examples": [ex.toDict() for ex in self._simba_examples[-max(self.simba_bsize, 8) :]],
            "predictors": predictors,
        }

    def load_simba_state(self, state: object) -> None:
        """Restore learned parameters into fresh predictors, then bind current clients."""
        state = validate_simba_state(state)
        restored = {}
        for name in ("qwen_proposer", "qwen_compressor"):
            predictor = dspy.Predict(getattr(self, name).signature)
            saved = dict(state["predictors"][name])
            saved.update(lm=None, traces=[], train=[])
            signature = predictor.signature
            if signature is None or len(saved["signature"]["fields"]) != len(signature.fields):
                raise ValueError(f"SIMBA predictor {name} has an incompatible field schema")
            try:
                predictor.load_state(saved)
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"Malformed SIMBA predictor {name}") from exc
            restored[name] = predictor
        examples = [
            dspy.Example(**ex).with_inputs("component_name", "current_instruction", "examples_with_feedback")
            for ex in state["examples"][-max(self.simba_bsize, 8) :]
        ]
        for name, predictor in restored.items():
            # Preserve the identities also held by the synthesis tool and worker.
            getattr(self, name).load_state(predictor.dump_state())
        self._simba_examples = examples
        self._simba_steps = int(state["steps"])
        self._bind_predictor_lms()

    def _score_task_improvement(self, example: Any, prediction: Any) -> float:
        """Reward a counterfactual task-score gain, never instruction formatting."""
        try:
            if self.task_scorer is None:
                raise RuntimeError("SIMBA requires a task scorer")
            candidate = dict(example.task_candidate)
            component = str(example.component_name)
            instruction = self._ensure_stage_schema_contract(
                component,
                str(example.current_instruction),
                str(getattr(prediction, "new_instruction", "") or "").strip(),
            )
            baseline = float(self.task_scorer(candidate))
            candidate[component] = instruction
            improvement = float(self.task_scorer(candidate)) - baseline
            if not math.isfinite(improvement):
                raise ValueError("Non-finite SIMBA task improvement")
            return 0.5 + 0.5 * improvement / (1.0 + abs(improvement))
        except Exception:
            # DSPy can swallow metric exceptions as zero scores. Refuse to adopt
            # a worker selected from incomplete/failed task measurements.
            self._simba_reward_errors += 1
            raise

    def step_simba(self) -> bool:
        """Run `dspy.SIMBA` on the inner worker (`self.worker_module`) over buffered reflection examples."""
        if not self.enable_simba or self.task_scorer is None or len(self._simba_examples) < self.simba_bsize:
            return False
        if not hasattr(dspy, "SIMBA"):
            return False

        self.preflight()
        eff_worker = self.worker_lm or self.orchestrator_lm or getattr(dspy.settings, "lm", None)
        if eff_worker is None:
            return False

        bsize = min(self.simba_bsize, len(self._simba_examples))
        trainset = list(self._simba_examples[-max(bsize, 8) :])
        prev_inst = str(getattr(self.qwen_proposer.signature, "instructions", "") or "")
        prev_demos = len(getattr(self.qwen_proposer, "demos", ()) or ())

        try:
            self._simba_reward_errors = 0
            simba = dspy.SIMBA(
                metric=self._score_task_improvement,
                bsize=bsize,
                num_candidates=self.simba_num_candidates,
                max_steps=self.simba_max_steps,
                max_demos=self.simba_max_demos,
                demo_input_field_maxlen=sys.maxsize,
                prompt_model=eff_worker,
                num_threads=1,
            )
            with dspy.context(lm=eff_worker, adapter=self.json_adapter, trace=[]):
                compiled_worker = simba.compile(
                    self.worker_module,
                    trainset=trainset,
                    seed=self._simba_steps,
                )
            if self._simba_reward_errors:
                raise RuntimeError("SIMBA task evaluations failed; retaining previous worker")
            if not isinstance(compiled_worker, InstructionSynthesizerWorker):
                raise TypeError("SIMBA returned an incompatible instruction worker")
            # Learned instructions/demos are also part of the next rendered prompt.
            # Reject an oversized worker before adopting or checkpointing its state.
            compiled_worker._rollout_lm = None
            for example in trainset:
                compiled_worker.fit_evidence(
                    example.current_instruction,
                    example.examples_with_feedback,
                    example.component_name,
                )
            new_inst = str(compiled_worker.qwen_proposer.signature.instructions).strip()
            self.qwen_proposer.signature = compiled_worker.qwen_proposer.signature
            self.qwen_proposer.demos = list(getattr(compiled_worker.qwen_proposer, "demos", ()) or ())
            self.qwen_compressor.signature = compiled_worker.qwen_compressor.signature
            self.qwen_compressor.demos = list(getattr(compiled_worker.qwen_compressor, "demos", ()) or ())
            self._bind_predictor_lms()

            entry = {
                "optimizer": "dspy.SIMBA",
                "target": "worker_module.qwen_proposer",
                "trainset_size": len(trainset),
                "bsize": bsize,
                "num_candidates": self.simba_num_candidates,
                "max_steps": self.simba_max_steps,
                "demos_before": prev_demos,
                "demos_after": len(self.qwen_proposer.demos),
                "instruction_changed": new_inst != prev_inst.strip(),
                "proposer_instruction": new_inst,
            }
            self.simba_telemetry.append(entry)
            self._simba_steps += 1
            trim_diagnostics(self)
        except (
            AdapterParseError,
            LMError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            StopIteration,
            KeyError,
            IndexError,
            AssertionError,
        ) as exc:
            logging.getLogger(__name__).warning(
                "SIMBA inner-worker step skipped (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False
        # Persistence failures must reach the caller, rather than look like a
        # successful durable step. SQLite keeps the previous complete snapshot.
        if self.save_simba_state is not None:
            self.save_simba_state(self.dump_simba_state())
        return True

    @staticmethod
    def _prepare_examples(
        examples: Any,
        *,
        max_examples: int = 5,
    ) -> list[dict[str, Any]]:
        """Prepare reflective examples, prioritizing top-5 failing examples and anonymizing gold literals."""
        if isinstance(examples, str):
            examples = json.loads(examples)
        if not examples:
            return []
        return reflection_examples(examples)[:max_examples]

    def _ensure_stage_schema_contract(
        self,
        component_name: str,
        current_instruction: str,
        draft: str,
    ) -> str:
        """Validate the instruction; the immutable DSPy Signature owns the output schema."""
        _ = (component_name, current_instruction)
        if not draft.strip():
            raise ValueError("Empty instruction")
        if self.max_instruction_chars is not None and len(draft) > self.max_instruction_chars:
            raise ValueError("Instruction exceeds the native writer length limit")
        return draft

    def cache_identity(self) -> dict[str, Any]:
        """All deployment/policy settings that affect reflection stage results."""
        models = []
        for lm in (self.orchestrator_lm, self.worker_lm):
            models.append(
                {
                    "model": getattr(lm, "model", None),
                    "kwargs": {
                        key: value
                        for key, value in getattr(lm, "kwargs", {}).items()
                        if key not in {"api_key", "api_token"}
                    },
                    "context": getattr(lm, "_yaw_context_window_tokens", None),
                    "tokenizer": getattr(getattr(lm, "counter", None), "identity", None),
                }
            )
        return {
            "version": 2,
            "models": models,
            "tokenizer": getattr(self.token_counter, "identity", None),
            "input_tokens": self.reflection_input_tokens,
            "settings": [
                self.max_instruction_chars,
                self.max_feedback_examples,
                self.worker_temperature,
                self.reflection_temperature,
                self.enable_gbnf,
            ],
            "selector": [
                self.inner_rlm.max_iters,
                self.inner_rlm.max_llm_calls,
                self.inner_rlm.max_output_chars,
            ],
            "adapter": type(self.json_adapter).__qualname__,
        }

    def select_evidence(
        self,
        component_name: str,
        current_instruction: str,
        component_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Checkpoint validated references, retaining exact training evidence."""
        self.preflight()
        originals = reflection_examples(component_examples)
        evidence_key = reflection_digest(
            {
                "version": 2,
                "deployment": self.cache_identity(),
                "component": component_name,
                "instruction": current_instruction,
                "examples": originals,
                "input_tokens": self.reflection_input_tokens,
                "writer_settings": [
                    self.max_instruction_chars,
                    self.max_feedback_examples,
                    self.worker_temperature,
                    self.reflection_temperature,
                    self.enable_gbnf,
                ],
                "selector": self.inner_rlm.generate_action.signature.instructions,
                "selector_limits": [
                    self.inner_rlm.max_iters,
                    self.inner_rlm.max_llm_calls,
                    self.inner_rlm.max_output_chars,
                ],
                "model": self.orchestrator_lm.model if self.orchestrator_lm else None,
                "tokenizer": getattr(self.token_counter, "identity", str(self.token_counter)),
            }
        )
        get = self.checkpoint_get or (lambda key: None)
        put = self.checkpoint_put or (lambda key, value: None)
        saved = get("evidence:" + evidence_key)
        if saved is None:
            fallback = None
            try:
                with dspy.context(lm=self.orchestrator_lm or self.worker_lm, trace=[]):
                    result = self.inner_rlm(current_instruction=current_instruction, examples_with_feedback=originals)
                selection = checkpoint_object(result.selection, "evidence selection")
                resolve_evidence(originals, selection, self.max_feedback_examples)
            except (
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                IndexError,
                AdapterParseError,
                LMError,
                OSError,
            ) as exc:
                fallback = type(exc).__name__
                selection = deterministic_evidence(originals, self.max_feedback_examples)
            saved = {"selection": selection, "fallback": fallback}
            put("evidence:" + evidence_key, saved)
        saved = checkpoint_object(saved, "evidence checkpoint")
        selection = checkpoint_object(saved.get("selection"), "evidence selection")
        fallback = saved.get("fallback")
        if fallback is not None and not isinstance(fallback, str):
            raise ValueError("Malformed evidence checkpoint fallback")
        packet = resolve_evidence(originals, selection, self.max_feedback_examples)
        diagnosis = selection.get("diagnosis", "")
        if not isinstance(diagnosis, str):
            raise ValueError("Malformed evidence diagnosis")
        packet.insert(0, {"diagnosis": diagnosis[: scale_context_budget(2000, self.context_window_tokens)]})
        return evidence_key, packet, fallback

    @diagnostic_boundary
    def propose_for_component(
        self,
        component_name: str,
        current_instruction: str,
        component_examples: Sequence[Mapping[str, Any]],
    ) -> str:
        evidence_key, packet, fallback = self.select_evidence(component_name, current_instruction, component_examples)
        get = self.checkpoint_get or (lambda key: None)
        put = self.checkpoint_put or (lambda key, value: None)
        key = "proposal:" + reflection_digest(
            {
                "evidence": evidence_key,
                "state": self.dump_simba_state(),
                "worker": self.worker_lm.model if self.worker_lm else None,
            }
        )
        completed = get(key)
        if completed is not None:
            completed = checkpoint_object(completed, "proposal checkpoint")
            instruction = completed.get("instruction")
            if not isinstance(instruction, str):
                raise ValueError("Malformed proposal checkpoint instruction")
            return self._ensure_stage_schema_contract(
                component_name.removesuffix(".predict"), current_instruction, instruction
            )
        from dspy.utils.usage_tracker import track_usage

        self.worker_module.sub_telemetry.clear()
        with dspy.context(trace=[]), track_usage() as tracker:
            prediction = self.worker_module(
                current_instruction=current_instruction,
                examples_with_feedback=packet,
                component_name=component_name,
            )
        usage = tracker.get_total_tokens()
        instruction = self._ensure_stage_schema_contract(
            component_name.removesuffix(".predict"),
            current_instruction,
            str(prediction.new_instruction),
        )
        put(key, {"instruction": instruction})
        self.telemetry.append(
            {
                "component_name": component_name,
                "mode": "evidence_native",
                "fallback": fallback,
                "evidence_key": evidence_key,
                "usage": usage,
                "tokens_used": sum_usage_tokens(usage),
                "new_instruction": instruction,
                "simba_steps": self._simba_steps,
            }
        )
        return instruction

    def forward(
        self,
        candidate: dict[str, str] | None = None,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        components_to_update: Sequence[str] | None = None,
        *,
        component_name: str = "",
        current_instruction: str = "",
        examples_with_feedback: Any = None,
    ) -> Any:
        """Dual entrypoint: GEPA `ProposalFn` `(candidate, reflective_dataset, components_to_update)` or \
`dspy.Module` `(component_name=..., current_instruction=..., examples_with_feedback=...)`."""
        if candidate is None or not isinstance(candidate, Mapping):
            self.preflight()
            prepared = self._prepare_examples(
                examples_with_feedback,
                max_examples=self.max_feedback_examples,
            )
            with dspy.context(trace=[]):
                pred = self.worker_module(
                    current_instruction=current_instruction,
                    examples_with_feedback=prepared,
                    component_name=component_name,
                )
            inst = self._ensure_stage_schema_contract(
                component_name,
                current_instruction,
                str(getattr(pred, "new_instruction", "") or "").strip(),
            )
            return dspy.Prediction(new_instruction=inst)

        if not components_to_update or reflective_dataset is None:
            return {}

        self.preflight()
        get = self.checkpoint_get or (lambda key: None)
        put = self.checkpoint_put or (lambda key, value: None)
        request = reflection_digest(
            {
                "version": 2,
                "deployment": self.cache_identity(),
                "candidate": candidate,
                "input_tokens": self.reflection_input_tokens,
                "tokenizer": getattr(self.token_counter, "identity", None),
                "writer_settings": [
                    self.max_instruction_chars,
                    self.max_feedback_examples,
                    self.worker_temperature,
                    self.reflection_temperature,
                    self.enable_gbnf,
                ],
                "data": reflection_examples([{"component": k, "examples": v} for k, v in reflective_dataset.items()]),
                "components": list(components_to_update),
            }
        )
        checkpoint_key = "request:" + request

        def snapshot(phase: str, **extra: Any) -> dict[str, Any]:
            state = self.dump_simba_state()
            return {
                "phase": phase,
                "state": state,
                "writer": reflection_digest(state["predictors"]),
                **extra,
            }

        record = get(checkpoint_key)
        if record is not None:
            record = validate_request_checkpoint(record)
        current_writer = reflection_digest(self.dump_simba_state()["predictors"])
        if record is not None and record["phase"] == "complete":
            saved_writer = record.get("writer", reflection_digest(record["state"]["predictors"]))
            if saved_writer != current_writer:
                record = None
        if record is not None:
            if record["phase"] != "complete" or record["state"]["steps"] > self._simba_steps:
                self.load_simba_state(record["state"])
            if record["phase"] == "complete":
                allowed = set(components_to_update) & set(candidate)
                if set(record["proposals"]) - allowed:
                    raise ValueError("Completed proposal has an unknown component")
                for name, instruction in record["proposals"].items():
                    self._ensure_stage_schema_contract(name, candidate[name], instruction)
                return record["proposals"]
        else:
            record = snapshot("collecting")
            put(checkpoint_key, record)
        if record["phase"] == "collecting":
            for name in components_to_update:
                if name not in candidate or not reflective_dataset.get(name):
                    continue
                _, packet, _ = self.select_evidence(name, candidate[name], reflective_dataset[name])
                prepared = self.worker_module.fit_evidence(candidate[name], packet, name)
                self._simba_examples.append(
                    dspy.Example(
                        task_candidate=dict(candidate),
                        component_name=name,
                        current_instruction=candidate[name],
                        examples_with_feedback=prepared,
                    ).with_inputs("component_name", "current_instruction", "examples_with_feedback")
                )
            del self._simba_examples[: -max(self.simba_bsize, 8)]
            # The request-local state is authoritative on restart, so an incomplete
            # SIMBA call is retried from its input snapshot, never applied twice.
            record = snapshot("learning")
            put(checkpoint_key, record)
        if record["phase"] == "learning":
            self.step_simba()
            record = snapshot("learned")
            put(checkpoint_key, record)
        updated = {
            name: self.propose_for_component(name, candidate[name], reflective_dataset[name])
            for name in components_to_update
            if name in candidate and reflective_dataset.get(name)
        }
        put(
            checkpoint_key,
            snapshot("complete", proposals=updated),
        )
        if self.save_simba_state is not None:
            self.save_simba_state(self.dump_simba_state())
        return updated

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


__all__ = [
    "InstructionSynthesizerWorker",
    "RLMInstructionProposer",
]
