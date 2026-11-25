"""Dataset registry, manifest bundles, benchmark slicing, and pure evaluation metrics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from filelock import FileLock

from yaw.core import render_document_text, write_text_atomic

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CATALOG: dict[str, Any] = json.loads(resources.files("yaw").joinpath("baselines.json").read_text(encoding="utf-8"))
_DATASET_SPECS: dict[str, dict[str, Any]] = dict(_CATALOG["datasets"])

BASELINES: dict[str, dict[str, Any]] = dict(_CATALOG["baselines"])
PINNED_COMMIT: str = str(_CATALOG["pinned_commit"])
HF_BASE_URL: str = str(_CATALOG["hf_base_url"])
UPSTREAM_DATASET_BASE_URL: str = str(_CATALOG["upstream_dataset_base_url"])
DATASETS: tuple[str, ...] = tuple(_DATASET_SPECS.keys())
DEFAULT_PARITY_DATASETS: tuple[str, ...] = tuple(k for k, v in _DATASET_SPECS.items() if v.get("default_parity"))
TIER2_CORPORA: frozenset[str] = frozenset(k for k, v in _DATASET_SPECS.items() if v.get("tier2"))
QA_ONLY_DATASETS: frozenset[str] = frozenset(k for k, v in _DATASET_SPECS.items() if v.get("qa_only"))
DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    k: (str(v["questions_file"]), str(v["corpus_file"])) for k, v in _DATASET_SPECS.items()
}
DATASET_SOURCE_FIELDS: dict[str, dict[str, Any]] = {k: dict(v["source_fields"]) for k, v in _DATASET_SPECS.items()}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_OMITTED_LINES_RE = re.compile(r"\n?\[\.\.\.\s*lines\s+\d+-\d+\s+omitted\s*\.\.\.\]\n?")


def _dataset_spec(name: str | Path | None) -> dict[str, Any]:
    if name is None or not bool(name) or not (raw := str(name).strip()):
        return {}
    lowered = raw.lower()
    if lowered in _DATASET_SPECS:
        return _DATASET_SPECS[lowered]
    stem = Path(raw).stem.strip().lower()
    if stem in _DATASET_SPECS:
        return _DATASET_SPECS[stem]
    file_name = Path(raw).name.strip().lower()
    for key, spec in _DATASET_SPECS.items():
        if file_name in (str(spec.get("questions_file", "")).lower(), str(spec.get("corpus_file", "")).lower()):
            return spec
        if stem.startswith(key):
            return spec
    return {}


_WIN_RESERVED_NAMES: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)
_SAFE_DOC_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def allow_upstream_github_fallback(name: str | Path | None) -> bool:
    """Return whether raw dataset downloads may fall back to upstream GitHub URLs."""
    return bool(_dataset_spec(name).get("allow_upstream_github_fallback", True))


def document_filename(doc_id: str) -> str:
    """Return a safe `.md` filename inside `documents/` for `doc_id`."""
    if (
        _SAFE_DOC_ID_RE.match(doc_id)
        and doc_id not in (".", "..")
        and len(doc_id) <= 128
        and doc_id.split(".", 1)[0].upper() not in _WIN_RESERVED_NAMES
    ):
        return f"{doc_id}.md"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("._") or "doc"
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized[:80]}_{digest}.md"


def render_document_markdown(title: str, body: str) -> str:
    """Render `(title, body)` as UTF-8 Markdown with an H1 heading when distinct."""
    clean_title = title.replace("\r\n", "\n").replace("\r", "\n").strip()
    norm_body = body.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in norm_body and norm_body.count("\\n") >= 2:
        norm_body = norm_body.replace("\\r\\n", "\n").replace("\\n", "\n")
    if clean_title and norm_body != clean_title and not norm_body.startswith((f"{clean_title}\n", "# ")):
        rendered = f"# {clean_title}\n\n{norm_body}"
    else:
        rendered = norm_body
    return f"{rendered}\n" if rendered and not rendered.endswith("\n") else rendered


def _strip_markdown_h1(md_text: str, title: str) -> str:
    """Strip at most one trailing newline and synthetic `# {title}` H1 prefix."""
    text = md_text[:-1] if md_text.endswith("\n") else md_text
    if (clean_title := title.strip()) and text.startswith(prefix := f"# {clean_title}\n\n"):
        return text[len(prefix) :]
    return text


@dataclass(frozen=True, slots=True)
class DatasetDocument:
    """One immutable dataset corpus document entry recorded in manifest.json."""

    document_id: str
    path: str
    content_hash: str
    title: str = ""


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Validated content-addressed dataset bundle anchored at `root`."""

    dataset: str
    root: Path
    documents: tuple[DatasetDocument, ...]
    title_key: str = "title"
    body_key: str = "text"
    id_key: str | None = "idx"

    @property
    def documents_root(self) -> Path:
        return self.root / "documents"

    @property
    def evaluation_path(self) -> Path:
        return self.root / "evaluation" / "questions.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _doc_row(self, doc: DatasetDocument) -> dict[str, Any]:
        md = (self.root / doc.path).read_text(encoding="utf-8")
        row = {self.title_key: doc.title, self.body_key: _strip_markdown_h1(md, doc.title), "idx": doc.document_id}
        if self.id_key is not None and bool(self.id_key) and self.id_key != "idx":
            row[self.id_key] = doc.document_id
        return row

    def read_documents(self) -> list[dict[str, Any]]:
        """Validate manifest and return all prepared corpus document dicts."""
        with FileLock(str(self.root / "bundle.lock")):
            validate_dataset_bundle(self)
            return [self._doc_row(doc) for doc in self.documents]

    def read_inputs(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Validate manifest and return `(evaluation_questions, corpus_documents)`."""
        with FileLock(str(self.root / "bundle.lock")):
            validate_dataset_bundle(self)
            questions = json.loads(self.evaluation_path.read_bytes())
            documents = [self._doc_row(doc) for doc in self.documents]
            return questions, documents

    def corpus_by_id(self) -> dict[str, str]:
        """Return `{document_id: markdown_text}` across the bundle."""
        with FileLock(str(self.root / "bundle.lock")):
            validate_dataset_bundle(self)
            return {
                doc.document_id: (self.root / doc.path).read_text(encoding="utf-8").rstrip("\n")
                for doc in self.documents
            }


def document_text(row: Mapping[str, Any], title_key: str = "title", body_key: str = "text") -> str:
    """Render `(title, body)` text from one corpus row using strict string validation."""
    title, body = row[title_key], row[body_key]
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError(f"Document fields {title_key!r}/{body_key!r} must be strings")
    return render_document_text(title, body)


def document_identity(
    row: Mapping[str, Any],
    *,
    id_key: str | None = "idx",
    title_key: str = "title",
    body_key: str = "text",
) -> str:
    """Return deterministic document identity from `id_key` or SHA-256[:16] of Markdown."""
    if id_key is not None and bool(id_key) and row.get(id_key) is not None:
        return str(row[id_key])
    if row.get("idx") is not None:
        return str(row["idx"])
    md = render_document_markdown(str(row.get(title_key, "")), str(row.get(body_key, "")))
    return hashlib.sha256(md.encode("utf-8")).hexdigest()[:16]


def _collect_bundle_documents(
    corpus_path: Path,
    corpus: list[Any],
    *,
    id_key: str | None,
    title_key: str,
    body_key: str,
) -> dict[str, tuple[DatasetDocument, str]]:
    prepared: dict[str, tuple[DatasetDocument, str]] = {}
    seen_lower_paths: dict[str, str] = {}
    for index, row in enumerate(corpus):
        if not isinstance(row, dict):
            raise ValueError(f"Corpus record {index} must be an object")
        if missing := ({title_key, body_key} - row.keys()):
            raise ValueError(f"{corpus_path}: record {index} missing document fields {sorted(missing)}")
        title_val, body_val = row[title_key], row[body_key]
        if not isinstance(title_val, str) or not isinstance(body_val, str):
            raise ValueError(f"Document fields {title_key!r}/{body_key!r} must be strings")
        rendered = render_document_markdown(title_val, body_val)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        raw_id = row.get(id_key) if id_key is not None and bool(id_key) else None
        doc_id = str(raw_id) if raw_id is not None else digest[:16]
        if not doc_id:
            raise ValueError(f"Empty document ID in corpus record {index}")
        rel_path = f"documents/{document_filename(doc_id)}"
        if (owner := seen_lower_paths.get(rel_path.lower())) is not None and owner != doc_id:
            raise ValueError(f"Case-insensitive filename collision between {owner!r} and {doc_id!r}: {rel_path}")
        seen_lower_paths[rel_path.lower()] = doc_id
        doc = DatasetDocument(doc_id, rel_path, digest, title_val)
        previous = prepared.get(doc_id)
        if previous is not None and previous[0].content_hash != digest:
            raise ValueError(f"Conflicting duplicate document ID {doc_id!r}")
        prepared[doc_id] = (doc, rendered)
    return prepared


def _sync_bundle_files(
    bundle: DatasetBundle,
    prepared: Mapping[str, tuple[DatasetDocument, str]],
    question_bytes: bytes,
    manifest: Mapping[str, Any],
) -> None:
    root = bundle.root
    with FileLock(str(root / "bundle.lock")):
        if bundle.documents_root.is_symlink() or bundle.evaluation_path.parent.is_symlink():
            raise ValueError("Bundle directories must not be symlinks")
        previous_paths: set[str] = set()
        if bundle.manifest_path.is_file():
            previous_paths = {
                str(doc["path"]) for doc in json.loads(bundle.manifest_path.read_text("utf-8"))["documents"]
            }
        desired_paths = {doc.path for doc in bundle.documents}
        actual_paths = {p.relative_to(root).as_posix() for p in bundle.documents_root.iterdir() if p.is_file()}
        if actual_paths - previous_paths - desired_paths:
            raise ValueError("Unmanaged files in prepared documents directory; choose a fresh destination")
        for doc, rendered in prepared.values():
            target = root / doc.path
            if not target.exists() or target.read_bytes() != rendered.encode("utf-8"):
                write_text_atomic(target, rendered, newline="")
        if not bundle.evaluation_path.exists() or bundle.evaluation_path.read_bytes() != question_bytes:
            write_text_atomic(bundle.evaluation_path, question_bytes.decode("utf-8"), newline="")
        live_paths = {(root / doc.path).resolve() for doc in bundle.documents}
        for stale in bundle.documents_root.iterdir():
            if stale.is_file() and stale.resolve() not in live_paths:
                stale.unlink()
        rendered_manifest = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        write_text_atomic(bundle.manifest_path, rendered_manifest, newline="")
        validate_dataset_bundle(bundle)


def prepare_dataset_bundle(
    dataset: str,
    corpus_path: Path,
    questions_path: Path,
    output_root: Path,
    *,
    id_key: str | None = "idx",
    title_key: str = "title",
    body_key: str = "text",
) -> DatasetBundle:
    """Write an atomic content-verified v2 Markdown dataset bundle under `output_root / dataset`."""
    if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
        raise ValueError("Dataset identity must be a single directory name")
    corpus_bytes, question_bytes = corpus_path.read_bytes(), questions_path.read_bytes()
    corpus, questions = json.loads(corpus_bytes), json.loads(question_bytes)
    if (
        not isinstance(corpus, list)
        or not isinstance(questions, list)
        or not all(isinstance(q, dict) for q in questions)
    ):
        raise ValueError("Corpus and evaluation inputs must be JSON arrays of objects")

    prepared = _collect_bundle_documents(corpus_path, corpus, id_key=id_key, title_key=title_key, body_key=body_key)
    root = output_root.resolve() / dataset
    root.mkdir(parents=True, exist_ok=True)
    (root / "documents").mkdir(exist_ok=True)
    bundle = DatasetBundle(dataset, root, tuple(doc for doc, _ in prepared.values()), title_key, body_key, id_key)
    manifest = {
        "version": 2,
        "dataset": dataset,
        "title_key": title_key,
        "body_key": body_key,
        "id_key": id_key,
        "source_corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "documents": [asdict(d) for d in bundle.documents],
        "evaluation_sha256": hashlib.sha256(question_bytes).hexdigest(),
    }
    _sync_bundle_files(bundle, prepared, question_bytes, manifest)
    return bundle


def load_dataset_bundle(root: Path | str) -> DatasetBundle:
    """Load and validate a prepared DatasetBundle from `root`."""
    resolved = Path(root).resolve()
    manifest_file = resolved / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Dataset manifest missing: {manifest_file}")
    with FileLock(str(resolved / "bundle.lock")):
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        if data.get("version") != 2:
            raise ValueError("Unsupported dataset manifest version; prepare a fresh bundle")
        bundle = DatasetBundle(
            data["dataset"],
            resolved,
            tuple(DatasetDocument(**doc) for doc in data["documents"]),
            data["title_key"],
            data["body_key"],
            data["id_key"],
        )
        validate_dataset_bundle(bundle)
        return bundle


def validate_dataset_bundle(bundle: DatasetBundle) -> None:
    """Verify manifest metadata, evaluation checksum, per-document hashes, and file inventory."""
    if not bundle.manifest_path.is_file():
        raise ValueError(f"Dataset manifest missing: {bundle.manifest_path}")
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise ValueError("Unsupported dataset manifest version")
    current = (manifest["dataset"], manifest["title_key"], manifest["body_key"], manifest["id_key"])
    docs = tuple(DatasetDocument(**doc) for doc in manifest["documents"])
    if current != (bundle.dataset, bundle.title_key, bundle.body_key, bundle.id_key) or docs != bundle.documents:
        raise ValueError("Dataset bundle changed; resolve the current manifest before reading")
    if hashlib.sha256(bundle.evaluation_path.read_bytes()).hexdigest() != manifest["evaluation_sha256"]:
        raise ValueError(f"Incomplete or corrupt evaluation export: {bundle.evaluation_path}")
    seen_ids: set[str] = set()
    docs_root = bundle.documents_root.resolve()
    for doc in bundle.documents:
        path = (bundle.root / doc.path).resolve()
        if path.parent != docs_root or path.suffix != ".md":
            raise ValueError(f"Invalid document path: {doc.path}")
        if doc.document_id in seen_ids:
            raise ValueError(f"Duplicate document ID: {doc.document_id}")
        seen_ids.add(doc.document_id)
        if hashlib.sha256(path.read_bytes()).hexdigest() != doc.content_hash:
            raise ValueError(f"Corrupt or incomplete document export: {path}")
    actual = {p.resolve() for p in bundle.documents_root.iterdir() if p.is_file()}
    if actual != {(bundle.root / doc.path).resolve() for doc in bundle.documents}:
        raise ValueError("Dataset document inventory does not match manifest")


def resolve_dataset_name(name: str) -> str:
    """Validate and normalize a registered dataset identifier."""
    cleaned = name.strip().lower()
    if cleaned not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Valid datasets: {', '.join(DATASETS)}")
    return cleaned


def resolve_dataset_bundle(dataset: Path | str, *, workspace_root: Path | None = None) -> DatasetBundle:
    """Resolve a bundle from directory path, manifest.json path, or canonical dataset name."""
    path = Path(dataset)
    if path.is_dir():
        return load_dataset_bundle(path)
    if path.name == "manifest.json" and path.is_file():
        return load_dataset_bundle(path.parent)
    if path.suffix or len(path.parts) > 1:
        raise ValueError(f"Expected a prepared bundle directory or manifest, got {dataset}")
    canonical = resolve_dataset_name(str(dataset))
    root = (workspace_root or _REPO_ROOT) / "outputs" / "datasets" / canonical
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Prepared dataset missing: {root}")
    return load_dataset_bundle(root)


def load_dataset_files(
    dataset_name: str,
    dataset_dir: Path | None = None,
    *,
    max_queries: int | None = None,
    include_corpus: bool = True,
    project_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load `(samples, corpus)` from prepared bundle directories or local raw benchmark files."""
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be positive")
    canonical = resolve_dataset_name(dataset_name)
    bundle: DatasetBundle | None = None
    if dataset_dir is not None and (dataset_dir / canonical / "manifest.json").is_file():
        bundle = load_dataset_bundle(dataset_dir / canonical)
    elif dataset_dir is None:
        bundle = resolve_dataset_bundle(canonical, workspace_root=project_root)
    if bundle is not None:
        samples, corpus_rows = bundle.read_inputs()
    else:
        assert dataset_dir is not None
        q_file, c_file = DATASET_REGISTRY[canonical]
        samples = json.loads((dataset_dir / q_file).read_text(encoding="utf-8"))
        corpus_rows = json.loads((dataset_dir / c_file).read_text(encoding="utf-8")) if include_corpus else []
    if max_queries is not None:
        samples = samples[:max_queries]
    return samples, (corpus_rows if include_corpus else [])


def extract_gold_answers(sample: Mapping[str, Any]) -> list[str]:
    """Extract deduplicated gold answers and aliases from a single QA sample."""
    gold_answer: Any = None
    if sample.get("answers") is not None:
        gold_answer = sample["answers"]
    elif any(k in sample for k in ("answer", "gold_ans", "gold_answer")):
        for key in ("answer", "gold_ans", "gold_answer"):
            if key in sample:
                gold_answer = sample[key]
                break
    elif sample.get("gold_answers") is not None:
        gold_answer = sample["gold_answers"]
    elif "reference" in sample:
        gold_answer = sample["reference"]
    elif "obj" in sample:
        gold_answer = [sample["obj"]]
        if sample.get("o_wiki_title"):
            gold_answer.append(sample["o_wiki_title"])
        for field in ("possible_answers", "o_aliases"):
            value = sample.get(field, [])
            if isinstance(value, str):
                if value.lstrip().startswith(("[", "{")):
                    parsed = json.loads(value)
                    gold_answer.extend(parsed if isinstance(parsed, list) else ([parsed] if parsed else []))
                else:
                    gold_answer.append(value)
            elif isinstance(value, list):
                gold_answer.extend(value)
            elif value:
                gold_answer.append(value)
    else:
        raise ValueError("Each query sample must contain an answer field")
    if gold_answer is None:
        raise ValueError("Answer fields must not be null")
    answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
    answers.extend(sample.get("answer_aliases", []))
    return list(dict.fromkeys(str(ans) for ans in answers))


def get_gold_answers(samples: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Extract gold answer lists across a sequence of dataset samples."""
    return [extract_gold_answers(sample) for sample in samples]


def _rendered_row_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    body = str(row.get("text") if "text" in row else row.get("content", "")).strip()
    return render_document_text(title, body)


def _corpus_lookup(corpus: Sequence[Mapping[str, Any]] | None) -> dict[str, str]:
    id_map: dict[str, str] = {}
    for row in corpus or ():
        cid = str(row.get("idx") or row.get("id") or row.get("doc_id") or "")
        if cid:
            id_map[cid] = _rendered_row_text(row)
    return id_map


def _extract_sample_passages(
    sample: Mapping[str, Any],
    *,
    separator: str = " ",
    supporting_only: bool = False,
    corpus_by_id: Mapping[str, str] | None = None,
) -> list[str]:
    if sample.get("gold_docs"):
        return [
            render_document_text(str(it.get("title", "")).strip(), str(it.get("text", "")).strip())
            if isinstance(it, Mapping)
            else str(it)
            for it in sample["gold_docs"]
        ]
    if supporting_only and "supporting_facts" in sample:
        titles = {item[0] for item in sample["supporting_facts"]}
        return [
            render_document_text(str(it[0]).strip(), separator.join(it[1]).strip())
            for it in sample["context"]
            if it[0] in titles
        ]
    if "contexts" in sample:
        items = (
            [it for it in sample["contexts"] if it.get("is_supporting", False)]
            if supporting_only
            else sample["contexts"]
        )
        return [render_document_text(str(it.get("title", "")).strip(), str(it.get("text", "")).strip()) for it in items]
    if "paragraphs" in sample:
        items = (
            [it for it in sample["paragraphs"] if it.get("is_supporting", True)]
            if supporting_only
            else sample["paragraphs"]
        )
        return [
            render_document_text(
                str(it.get("title", "")).strip(), str(it.get("text", it.get("paragraph_text", ""))).strip()
            )
            for it in items
        ]
    if isinstance(sample.get("context"), list):
        return [render_document_text(str(it[0]).strip(), separator.join(it[1]).strip()) for it in sample["context"]]
    if "expected_doc_ids" in sample:
        lookup = corpus_by_id or {}
        return [lookup[str(did)] for did in sample.get("expected_doc_ids", []) if supporting_only or str(did) in lookup]
    if supporting_only:
        raise ValueError("Sample needs supporting_facts, contexts, paragraphs, context, or expected_doc_ids")
    return []


def get_gold_docs(
    samples: Sequence[Mapping[str, Any]],
    dataset_name: str | Path | None = None,
    corpus: Sequence[Mapping[str, Any]] | None = None,
    *,
    corpus_by_id: Mapping[str, str] | None = None,
) -> list[list[str]] | None:
    """Extract deduplicated supporting gold passages, or `None` for QA-only datasets."""
    spec = _dataset_spec(dataset_name)
    if spec.get("qa_only"):
        return None
    separator = str(spec.get("sentence_separator", " "))
    id_map: dict[str, str] = (
        dict(corpus_by_id) if corpus_by_id is not None and bool(corpus_by_id) else _corpus_lookup(corpus)
    )
    return [
        list(dict.fromkeys(_extract_sample_passages(s, separator=separator, supporting_only=True, corpus_by_id=id_map)))
        for s in samples
    ]


def _collect_slice_texts(
    samples: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    id_map: Mapping[str, str],
    *,
    start: int,
    max_queries: int | None,
    full_corpus: bool,
    index_queries: int | None,
) -> tuple[list[str], int]:
    if full_corpus or max_queries is None or bool(spec.get("full_corpus_slice")):
        return [_rendered_row_text(doc) for doc in corpus], 0

    eff_iq = int(max_queries * 1.4) if index_queries is None else index_queries
    pooled_end = start + eff_iq
    separator = str(spec.get("sentence_separator", " "))
    docs = [
        p
        for s in samples[start:pooled_end]
        for p in _extract_sample_passages(s, separator=separator, corpus_by_id=id_map)
    ]
    min_per_q = int(spec.get("min_docs_per_query", 0))
    if min_per_q <= 0:
        return docs, 0

    minimum_docs = eff_iq * min_per_q
    docs = list(dict.fromkeys(docs))
    seen_docs = set(docs)
    for sample in samples[pooled_end:]:
        if len(docs) >= minimum_docs:
            break
        for cand in _extract_sample_passages(sample, separator=separator, corpus_by_id=id_map):
            if cand not in seen_docs:
                docs.append(cand)
                seen_docs.add(cand)
                if len(docs) >= minimum_docs:
                    break
    return docs, minimum_docs


def _select_slice_doc_ids(
    corpus: Sequence[Mapping[str, Any]],
    selected_text: set[str],
    minimum_docs: int,
    spec: Mapping[str, Any],
) -> list[str]:
    sf = spec.get("source_fields", {})
    id_k = sf.get("id_key", "idx")
    t_k, b_k = str(sf.get("title_key", "title")), str(sf.get("body_key", "text"))
    document_ids = [
        document_identity(row, id_key=id_k, title_key=t_k, body_key=b_k)
        for row in corpus
        if _rendered_row_text(row) in selected_text
    ]
    seen_ids = set(document_ids)
    for row in corpus:
        if len(document_ids) >= minimum_docs:
            break
        did = document_identity(row, id_key=id_k, title_key=t_k, body_key=b_k)
        if did not in seen_ids:
            document_ids.append(did)
            seen_ids.add(did)
    return document_ids


def build_benchmark_slice(
    dataset_name: str,
    samples: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    max_queries: int | None = None,
    query_offset: int = 0,
    full_corpus: bool = False,
    index_queries: int | None = None,
) -> tuple[list[str], list[str], list[list[str]] | None, list[list[str]]]:
    """Select slice document IDs, queries, gold passages, and gold answers."""
    if (max_queries is not None and max_queries < 1) or (index_queries is not None and index_queries < 1):
        raise ValueError("max_queries and index_queries must be positive integers when specified.")
    if query_offset < 0:
        raise ValueError("query_offset must be non-negative.")

    start = max(0, query_offset)
    selected = list(samples[start : None if max_queries is None else start + max_queries])
    queries = [str(sample["question"]) for sample in selected]
    spec = _dataset_spec(dataset_name)
    id_map = _corpus_lookup(corpus)
    gold_docs = get_gold_docs(selected, dataset_name, corpus=corpus, corpus_by_id=id_map)
    gold_answers = get_gold_answers(selected)

    docs, minimum_docs = _collect_slice_texts(
        samples,
        corpus,
        spec,
        id_map,
        start=start,
        max_queries=max_queries,
        full_corpus=full_corpus,
        index_queries=index_queries,
    )
    if gold_docs is not None:
        for glist in gold_docs:
            docs.extend(glist)
    document_ids = _select_slice_doc_ids(corpus, set(docs), minimum_docs, spec)
    return document_ids, queries, gold_docs, gold_answers


def resolve_slice_corpus_scope(
    bundle_slices: Sequence[tuple[DatasetBundle, Sequence[str]]],
) -> tuple[Path, tuple[str, ...]]:
    """Return `(corpus_root, included_patterns)` indexing prepared `.md` files directly."""
    if not bundle_slices:
        raise ValueError("At least one bundle slice is required.")
    selected: list[tuple[DatasetBundle, list[DatasetDocument]]] = []
    for bundle, doc_ids in bundle_slices:
        by_id = {doc.document_id: doc for doc in bundle.documents}
        selected.append((bundle, [by_id[str(did)] for did in dict.fromkeys(doc_ids) if str(did) in by_id]))
    if len(selected) == 1:
        bundle, matched = selected[0]
        if not matched:
            raise ValueError(f"Dataset slice {bundle.dataset!r} matched zero bundle documents.")
        patterns = ("*.md",) if len(matched) == len(bundle.documents) else tuple(Path(d.path).name for d in matched)
        return bundle.documents_root.resolve(), patterns
    common_root = Path(os.path.commonpath([str(b.documents_root.resolve()) for b, _ in selected])).resolve()
    patterns = tuple(
        dict.fromkeys((b.root / d.path).resolve().relative_to(common_root).as_posix() for b, m in selected for d in m)
    )
    if not patterns:
        raise ValueError("Selected bundle slices matched zero documents.")
    return common_root, patterns


def normalize_answer(text: str) -> str:
    """Apply canonical SQuAD NFD normalization (lowercase, drop punctuation and articles)."""
    cleaned = unicodedata.normalize("NFD", text).lower().translate(_PUNCT_TABLE)
    return " ".join(_ARTICLE_RE.sub(" ", cleaned).split())


def _single_f1(prediction: str, ground_truth: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(ground_truth).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0 or not pred_toks or not gold_toks:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2.0 * precision * recall) / (precision + recall)


def compute_qa_em_f1(pred_str: str, gold_answers: Sequence[str]) -> tuple[float, float]:
    """Compute best-reference normalized Exact Match and token F1 across `gold_answers`."""
    if not gold_answers:
        return 0.0, 0.0
    norm_pred = normalize_answer(pred_str)
    em = max((1.0 if norm_pred == normalize_answer(gold) else 0.0) for gold in gold_answers)
    f1 = max(_single_f1(pred_str, gold) for gold in gold_answers)
    return float(em), float(f1)


def _strip_doc_title_prefixes(clean_ret: str, clean_gold: str, segments: list[str]) -> tuple[str, str, list[str]]:
    segs = list(segments)
    for pfx in ("Wikipedia Title: ", "Title: ", "# "):
        if clean_ret.startswith(pfx):
            clean_ret = clean_ret[len(pfx) :].strip()
        if clean_gold.startswith(pfx):
            clean_gold = clean_gold[len(pfx) :].strip()
        if segs and segs[0].startswith(pfx):
            segs[0] = segs[0][len(pfx) :].strip()
    return clean_ret, clean_gold, segs


def _segments_match_gold(segments: list[str], ws_gold: str) -> bool:
    ws_segs = [" ".join(seg.split()) for seg in segments if " ".join(seg.split())]
    if ws_segs and sum(len(s) for s in ws_segs) >= 20 and all(s in ws_gold for s in ws_segs):
        return True
    if "\n" not in segments[0]:
        return False
    seg0_body = " ".join(segments[0].split("\n", 1)[1].split())
    body_segs = [seg0_body, *ws_segs[1:]] if seg0_body else ws_segs[1:]
    return bool(body_segs and sum(len(s) for s in body_segs) >= 20 and all(s in ws_gold for s in body_segs))


def _body_matches_gold(clean_ret: str, clean_gold: str, ws_ret: str, ws_gold: str) -> bool:
    if len(ws_ret) >= 20 and (ws_ret in ws_gold or ws_gold in ws_ret):
        return True
    if "\n" in clean_ret:
        body = " ".join(clean_ret.split("\n", 1)[1].split())
        if len(body) >= 20 and (body in ws_gold or ws_gold in body):
            return True
    if "\n" in clean_gold and len(gold_body := " ".join(clean_gold.split("\n", 1)[1].split())) >= 20:
        return gold_body in ws_ret
    return False


def doc_matches_gold(retrieved_doc: str, gold_doc: str) -> bool:
    """Match retrieved passages to gold passages across crops, title prefixes, and omitted spans."""
    if retrieved_doc == gold_doc:
        return True
    raw_ret = retrieved_doc.split("\n[... AST section cropped", 1)[0].strip()
    clean_gold = gold_doc.strip()
    if not raw_ret or not clean_gold:
        return False
    segments = [seg.strip() for seg in _OMITTED_LINES_RE.split(raw_ret) if seg.strip()]
    clean_ret = "\n".join(segments) if segments else ""
    if not clean_ret:
        return False
    if (
        clean_ret == clean_gold
        or (clean_ret.startswith("chunk-") and clean_ret == "chunk-" + hashlib.md5(clean_gold.encode()).hexdigest())
        or (clean_gold.startswith("chunk-") and clean_gold == "chunk-" + hashlib.md5(clean_ret.encode()).hexdigest())
    ):
        return True
    clean_ret, clean_gold, segments = _strip_doc_title_prefixes(clean_ret, clean_gold, segments)
    ws_ret, ws_gold = " ".join(clean_ret.split()), " ".join(clean_gold.split())
    if clean_ret == clean_gold or ws_ret == ws_gold:
        return True
    if len(segments) > 1 and _segments_match_gold(segments, ws_gold):
        return True
    return _body_matches_gold(clean_ret, clean_gold, ws_ret, ws_gold)


def score_retrieval_recalls(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_list: Sequence[int] = (1, 2, 5, 10, 20),
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute proportional per-query and pooled `Recall@k` scores."""
    if len(gold_docs) != len(retrieved_docs):
        raise ValueError("gold_docs and retrieved_docs must have the same length.")
    sorted_k = sorted(set(k_list))
    if not sorted_k or any(k < 1 for k in sorted_k):
        raise ValueError("k_list must contain at least one positive integer.")
    per_ex: list[dict[str, float]] = []
    pooled: dict[str, float] = {f"Recall@{k}": 0.0 for k in sorted_k}
    for golds, retrieved in zip(gold_docs, retrieved_docs, strict=True):
        unique_gold = list(dict.fromkeys(golds))
        ex_score: dict[str, float] = {}
        for k in sorted_k:
            if unique_gold:
                matched = sum(1 for gold in unique_gold if any(doc_matches_gold(ret, gold) for ret in retrieved[:k]))
                ex_score[f"Recall@{k}"] = matched / len(unique_gold)
            else:
                ex_score[f"Recall@{k}"] = 0.0
            pooled[f"Recall@{k}"] += ex_score[f"Recall@{k}"]
        per_ex.append(ex_score)
    if not gold_docs:
        return pooled, per_ex
    return {k: round(v / len(gold_docs), 4) for k, v in pooled.items()}, per_ex


def score_qa_em_f1(
    gold_answers: Sequence[Sequence[str]],
    predicted_answers: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Compute pooled and per-example best-reference `ExactMatch` and `F1`."""
    if len(gold_answers) != len(predicted_answers):
        raise ValueError("gold_answers and predicted_answers must have the same length.")
    per_ex: list[dict[str, float]] = []
    em_sum, f1_sum = 0.0, 0.0
    for golds, pred in zip(gold_answers, predicted_answers, strict=True):
        em, f1 = compute_qa_em_f1(pred, golds)
        per_ex.append({"ExactMatch": em, "F1": f1})
        em_sum += em
        f1_sum += f1
    n = len(gold_answers)
    return {"ExactMatch": em_sum / n if n else 0.0, "F1": f1_sum / n if n else 0.0}, per_ex
