"""Disk-only candidate stage caches with safe, typed JSON and source hydration."""

import base64
import hashlib
import json
import logging
import zlib
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import dspy
import numpy as np
from dspy.primitives.repl_types import REPLHistory

from yaw.core import OpenIERecord, StructuredTriple
from yaw.daemon.graph.storage import CandidateCacheStore, ManagedConnection

logger = logging.getLogger(__name__)


def _encode_string(value: str, sources: dict[str, str]) -> Any:
    if len(value) >= 16:
        for digest, source in sources.items():
            start = source.find(value)
            if start >= 0:
                return {
                    "type": "source_span",
                    "value": digest,
                    "start": start,
                    "end": start + len(value),
                }
    parts: list[Any] = [value]
    for digest, source in sources.items():
        for spelling, escaped, ascii_only in (
            (source, False, False),
            (json.dumps(source)[1:-1], True, True),
            (json.dumps(source, ensure_ascii=False)[1:-1], True, False),
        ):
            if not spelling or (len(spelling) < 16 and spelling != value):
                continue
            replaced: list[Any] = []
            for part in parts:
                if not isinstance(part, str) or spelling not in part:
                    replaced.append(part)
                    continue
                fragments = part.split(spelling)
                for i, fragment in enumerate(fragments):
                    if i:
                        replaced.append(
                            {
                                "type": "source",
                                "value": digest,
                                "escaped": escaped,
                                "ascii_only": ascii_only,
                            }
                        )
                    if fragment:
                        replaced.append(fragment)
            parts = replaced
    return value if parts == [value] else {"type": "source_parts", "value": parts}


def _encode(value: Any, sources: dict[str, str], *, source_passages: bool = False) -> Any:
    def encode(item: Any) -> Any:
        return _encode(item, sources, source_passages=source_passages)

    if isinstance(value, str):
        return _encode_string(value, sources)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": [
                [
                    key,
                    {"type": "passage"}
                    if key == "passage" and isinstance(item, str) and not source_passages
                    else encode(item),
                ]
                for key, item in value.items()
            ],
        }
    if isinstance(value, (OpenIERecord, StructuredTriple)):
        return {
            "type": type(value).__name__,
            "value": encode({f.name: getattr(value, f.name) for f in fields(value)}),
        }
    if isinstance(value, dspy.Prediction):
        return {"type": "prediction", "value": encode(value.toDict())}
    if isinstance(value, REPLHistory):
        return {
            "type": "repl_history",
            "value": encode(
                {
                    "entries": [entry.model_dump() for entry in value.entries],
                    "max_output_chars": value.max_output_chars,
                }
            ),
        }
    if isinstance(value, np.ndarray):
        vector = np.asarray(value, dtype=np.float32)
        return {
            "type": "array",
            "shape": list(vector.shape),
            "value": base64.b64encode(vector.tobytes()).decode("ascii"),
        }
    if isinstance(value, (tuple, list, set)):
        return {"type": type(value).__name__, "value": [encode(item) for item in value]}
    raise TypeError(f"Unsupported candidate-cache value: {type(value).__name__}")


def _decode(value: Any, passage: str | None, sources: dict[str, str]) -> Any:
    if not isinstance(value, dict):
        return value
    tag = value["type"]
    if tag == "source_span":
        return sources[value["value"]][value["start"] : value["end"]]
    if tag == "source":
        source = sources[value["value"]]
        return json.dumps(source, ensure_ascii=value["ascii_only"])[1:-1] if value["escaped"] else source
    if tag == "source_parts":
        return "".join(_decode(part, passage, sources) for part in value["value"])
    if tag == "passage":
        if passage is None:
            raise ValueError("OpenIE trace replay requires the current hydrated passage.")
        return passage
    if tag == "array":
        return np.frombuffer(base64.b64decode(value["value"]), dtype=np.float32).reshape(value["shape"]).copy()
    if tag == "dict":
        return {k: _decode(v, passage, sources) for k, v in value["value"]}
    if tag in {"list", "tuple", "set"}:
        decoded = [_decode(item, passage, sources) for item in value["value"]]
        return {"list": list, "tuple": tuple, "set": set}[tag](decoded)
    decoded = _decode(value["value"], passage, sources)
    if tag == "OpenIERecord":
        return OpenIERecord(**decoded)
    if tag == "StructuredTriple":
        return StructuredTriple(**decoded)
    if tag == "prediction":
        return dspy.Prediction(**decoded)
    if tag == "repl_history":
        return REPLHistory.model_validate(decoded)
    raise ValueError(f"Unknown candidate-cache type: {tag}")


class CandidatePayloadCodec:
    """JSON snapshots with corpus excerpts hydrated by source identity and span."""

    def __init__(self, sources: Callable[[], dict[str, str]]) -> None:
        self.sources = sources

    def encode(self, value: Any) -> str:
        return json.dumps(
            {"reflection_codec": 1, "value": _encode(value, self.sources(), source_passages=True)},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def decode(self, value: str) -> Any:
        parsed = json.loads(value)
        if parsed.get("reflection_codec") == 1:
            return _decode(parsed["value"], None, self.sources())
        return parsed  # Legacy snapshots still reach explicit state/context validation.


class CandidateStageCache:
    """One stage's view of a shared 256 MiB SQLite payload budget.

    No resident result dictionary or pickle. Cache keys hash inputs rather than
    storing source text; OpenIE traces hydrate passages supplied by the caller.
    Namespace changes invalidate results after model, settings or code changes.
    """

    def __init__(
        self,
        store: CandidateCacheStore,
        managed_conn: ManagedConnection,
        stage: str,
        namespace: Callable[[], str],
        *,
        max_bytes: int = 256 * 1024 * 1024,
        sources: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self.store = store
        self.managed_conn = managed_conn
        self.stage = stage
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.sources = sources or (lambda: {})
        self.hits = 0
        self.misses = 0

    def _key(self, key: tuple[str, ...]) -> str:
        raw = json.dumps([self.namespace(), key], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def __contains__(self, key: tuple[str, ...]) -> bool:
        return self.store.contains(self.stage, self._key(key))

    def get(self, key: tuple[str, ...], *, passage: str | None = None) -> Any:
        digest = self._key(key)
        payload = self.store.get(self.stage, digest)
        if payload is None:
            self.misses += 1
            return None
        try:
            result = _decode(json.loads(zlib.decompress(payload)), passage, self.sources())
        except (zlib.error, UnicodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Discarding invalid %s candidate-cache entry: %s", self.stage, exc)
            if self.store.conn.in_transaction:
                self.store.delete(self.stage, digest)
            else:
                with self.managed_conn.transaction():
                    self.store.delete(self.stage, digest)
            self.misses += 1
            return None
        self.hits += 1
        return result

    def __getitem__(self, key: tuple[str, ...]) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: tuple[str, ...], value: Any) -> None:
        payload = zlib.compress(
            json.dumps(
                _encode(value, self.sources()),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if self.store.conn.in_transaction:
            self.store.put(self.stage, self._key(key), payload, max_bytes=self.max_bytes)
        else:
            with self.managed_conn.transaction():
                self.store.put(self.stage, self._key(key), payload, max_bytes=self.max_bytes)
