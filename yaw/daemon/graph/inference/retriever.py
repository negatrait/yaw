"""First-class DSPy Retrieval Model (`YawRM`), `YawPassage`, and `YawRetriever` modules."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import dspy


class YawPassage(dspy.Prediction):
    """Passage object returned by `YawRM` exposing `.long_text`, `.pid`, `.score`, and `.rank` for `dspy.Retrieve`."""

    def __init__(
        self,
        long_text: str,
        *,
        pid: str = "",
        score: float = 0.0,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            long_text=str(long_text),
            pid=str(pid),
            score=float(score),
            rank=int(rank),
            **kwargs,
        )


class YawRM:
    """First-class DSPy Retrieval Model (`dspy.RM`) backed by YAW's Forward-Push PPR & recognition memory hook."""

    def __init__(
        self,
        hook: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None = None,
        *,
        program: Any = None,
        default_k: int = 10,
    ) -> None:
        self.hook = hook
        self.program = program
        self.default_k = int(default_k)

    def _invoke_hook_with_pids(
        self,
        query: str,
        *,
        k: int | None = None,
        item_docs: Sequence[str] | None = None,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
        if self.hook is None:
            raise ValueError("No retrieval hook is registered on YawRM.")
        try:
            sig = inspect.signature(self.hook)
            param_names = list(sig.parameters.keys())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
            accepts_prog = len(param_names) >= 2 or has_var_pos
            accepts_item_docs = "item_docs" in sig.parameters or len(param_names) >= 3 or has_var_kw
            accepts_k = "k" in sig.parameters or has_var_kw
        except TypeError, ValueError:
            accepts_prog = False
            accepts_item_docs = False
            accepts_k = False
        kw: dict[str, Any] = {}
        if accepts_item_docs and self.program is not None:
            kw["item_docs"] = item_docs
        if accepts_k and k is not None:
            kw["k"] = int(k)
        if accepts_prog and self.program is not None:
            raw = self.hook(query, self.program, **kw)
        else:
            raw = self.hook(query, **kw)
        if isinstance(raw, tuple) and len(raw) == 4:
            ctx, passages, scores, pids = raw
            return str(ctx), tuple(passages), tuple(scores), tuple(pids)
        ctx, passages, scores = raw
        pids = tuple(str(p) if str(p).startswith("chunk-") else f"chunk-{i}" for i, p in enumerate(passages))
        return str(ctx), tuple(passages), tuple(scores), pids

    def _invoke_hook(
        self,
        query: str,
        *,
        item_docs: Sequence[str] | None = None,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
        ctx, passages, scores, _ = self._invoke_hook_with_pids(query, item_docs=item_docs)
        return ctx, passages, scores

    def __call__(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
        **kwargs: Any,
    ) -> list[YawPassage]:
        effective_k = self.default_k if k is None else int(k)
        item_docs = kwargs.get("item_docs")
        queries = [query_or_queries] if isinstance(query_or_queries, str) else [str(q) for q in query_or_queries]
        results: list[YawPassage] = []
        for q in queries:
            _, passages, scores, pids = self._invoke_hook_with_pids(q, k=effective_k, item_docs=item_docs)
            for idx, text in enumerate(passages[:effective_k]):
                sc = float(scores[idx]) if idx < len(scores) else 0.0
                pid = str(pids[idx]) if idx < len(pids) else f"chunk-{idx}"
                results.append(
                    YawPassage(
                        long_text=str(text),
                        pid=pid,
                        score=sc,
                        rank=idx + 1,
                    )
                )
        return results


class YawRetriever(dspy.Retrieve):
    """First-class `dspy.Retrieve` module backed by `YawRM` and YAW's Forward-Push PPR + recognition memory."""

    def __init__(
        self,
        k: int = 10,
        *,
        rm: YawRM | None = None,
        hook: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None = None,
    ) -> None:
        super().__init__(k=k)
        self.rm: YawRM = rm if rm is not None else YawRM(hook=hook, default_k=k)
        if hook is not None and self.rm.hook is None:
            self.rm.hook = hook

    @property
    def hook(
        self,
    ) -> (
        Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None
    ):
        return self.rm.hook

    @hook.setter
    def hook(
        self,
        value: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None,
    ) -> None:
        self.rm.hook = value

    def dump_state(self, json_mode: bool = True, **_kwargs: Any) -> dict[str, Any]:
        _ = json_mode
        return {"k": self.k}

    def load_state(self, state: dict[str, Any], **_kwargs: Any) -> None:
        if isinstance(state, Mapping) and "k" in state:
            self.k = int(state["k"])

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        effective_k = self.k if k is None else int(k)
        if self.rm.hook is not None:
            hydrated_ctx, raw_passages, raw_scores, raw_pids = self.rm._invoke_hook_with_pids(
                query, k=effective_k, item_docs=kwargs.get("item_docs")
            )
            sliced_passages = list(raw_passages[:effective_k])
            sliced_scores = list(raw_scores[:effective_k])
            return dspy.Prediction(
                passages=sliced_passages,
                context=hydrated_ctx,
                scores=sliced_scores,
                pids=tuple(raw_pids[:effective_k]),
                all_passages=tuple(raw_passages),
                all_scores=tuple(raw_scores),
                all_pids=tuple(raw_pids),
            )
        active_rm = getattr(dspy.settings, "rm", None)
        if active_rm is not None:
            hits = active_rm(query, k=effective_k, **kwargs)
            passages = [
                str(
                    getattr(
                        psg,
                        "long_text",
                        psg.get("long_text", psg) if isinstance(psg, Mapping) else psg,
                    )
                )
                for psg in hits
            ]
            scores = [
                float(
                    getattr(
                        psg,
                        "score",
                        psg.get("score", 0.0) if isinstance(psg, Mapping) else 0.0,
                    )
                )
                for psg in hits
            ]
            pids = tuple(
                str(
                    getattr(
                        psg,
                        "pid",
                        psg.get("pid", f"chunk-{i}")
                        if isinstance(psg, Mapping)
                        else (psg if str(psg).startswith("chunk-") else f"chunk-{i}"),
                    )
                )
                for i, psg in enumerate(hits)
            )
            return dspy.Prediction(
                passages=passages,
                context="\n\n".join(passages),
                scores=scores,
                pids=pids,
                all_passages=tuple(passages),
                all_scores=tuple(scores),
                all_pids=pids,
            )
        raise ValueError("No retrieval hook or dspy.settings.rm is configured for YawRetriever.")


__all__ = [
    "YawPassage",
    "YawRM",
    "YawRetriever",
]
