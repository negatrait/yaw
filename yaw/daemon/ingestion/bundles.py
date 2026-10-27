"""Faithful document-file preparation and explicit dataset bundle manifests."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock

from yaw.core import ChunkPointer, render_document_text, write_text_atomic

from .corpus import CorpusIngestor


@dataclass(frozen=True, slots=True)
class DatasetDocument:
    document_id: str
    path: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class DatasetBundle:
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

    def read_documents(self) -> list[dict[str, Any]]:
        """Read only declared document files; identity is supplied by the manifest."""
        with FileLock(str(self.root / "bundle.lock")):
            validate_dataset_bundle(self)
            return [json.loads((self.root / doc.path).read_text(encoding="utf-8")) for doc in self.documents]

    def read_inputs(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Read one validated generation of evaluation and document inputs."""
        with FileLock(str(self.root / "bundle.lock")):
            validate_dataset_bundle(self)
            return json.loads(self.evaluation_path.read_bytes()), [
                json.loads((self.root / doc.path).read_bytes()) for doc in self.documents
            ]

    def corpus_by_id(self) -> dict[str, str]:
        rows = self.read_documents()
        return {
            doc.document_id: document_text(row, self.title_key, self.body_key)
            for doc, row in zip(self.documents, rows, strict=True)
        }


def document_text(row: Mapping[str, Any], title_key: str = "title", body_key: str = "text") -> str:
    title, body = row[title_key], row[body_key]
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError(f"Document fields {title_key!r}/{body_key!r} must be strings")
    return render_document_text(title, body)


def document_identity(row: Mapping[str, Any], *, id_key: str | None = "idx") -> str:
    if id_key and row.get(id_key) is not None:
        return str(row[id_key])
    rendered = json.dumps({"title": row["title"], "text": row["text"]}, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


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
    """Copy declared document keys and all QA values; never select documents by labels."""
    if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
        raise ValueError("Dataset identity must be a single directory name")
    corpus_bytes, question_bytes = corpus_path.read_bytes(), questions_path.read_bytes()
    corpus = json.loads(corpus_bytes)
    questions = json.loads(question_bytes)
    if not isinstance(corpus, list) or not isinstance(questions, list):
        raise ValueError("Corpus and evaluation inputs must be JSON arrays")
    if not all(isinstance(q, dict) for q in questions):
        raise ValueError("Evaluation rows must be JSON objects")
    prepared: dict[str, tuple[DatasetDocument, str]] = {}
    for index, row in enumerate(corpus):
        if not isinstance(row, dict):
            raise ValueError(f"Corpus record {index} must be an object")
        missing = {title_key, body_key} - row.keys()
        if missing:
            raise ValueError(f"{corpus_path}: record {index} missing document fields {sorted(missing)}")
        fields = {key: row[key] for key in (title_key, body_key)}
        if id_key and id_key in row:
            fields[id_key] = row[id_key]
        document_text(fields, title_key, body_key)
        rendered = json.dumps(fields, ensure_ascii=False, indent=2) + "\n"
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        raw_id = row.get(id_key) if id_key else None
        document_id = str(raw_id) if raw_id is not None else digest
        if not document_id:
            raise ValueError(f"Empty document ID in corpus record {index}")
        filename = hashlib.sha256(f"{dataset}\0{document_id}".encode()).hexdigest() + ".json"
        doc = DatasetDocument(document_id, f"documents/{filename}", digest)
        previous = prepared.get(document_id)
        if previous is not None and previous[0].content_hash != digest:
            raise ValueError(f"Conflicting duplicate document ID {document_id!r}")
        prepared[document_id] = (doc, rendered)
    root = output_root.resolve() / dataset
    root.mkdir(parents=True, exist_ok=True)
    (root / "documents").mkdir(exist_ok=True)
    bundle = DatasetBundle(dataset, root, tuple(doc for doc, _ in prepared.values()), title_key, body_key, id_key)
    manifest = {
        "version": 1,
        "dataset": dataset,
        "title_key": title_key,
        "body_key": body_key,
        "id_key": id_key,
        "source_corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "documents": [
            {"document_id": d.document_id, "path": d.path, "content_hash": d.content_hash} for d in bundle.documents
        ],
        "evaluation_sha256": hashlib.sha256(question_bytes).hexdigest(),
    }
    # Validate all inputs before publishing any file. The last manifest is the
    # commit marker; interrupted writes are caught by content validation.
    with FileLock(str(root / "bundle.lock")):
        if bundle.documents_root.is_symlink():
            raise ValueError("Bundle documents directory must not be a symlink")
        previous_paths = set()
        if bundle.manifest_path.is_file():
            previous_manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
            previous_paths = {str(doc["path"]) for doc in previous_manifest["documents"]}
        desired_paths = {doc.path for doc in bundle.documents}
        actual_paths = {p.relative_to(root).as_posix() for p in bundle.documents_root.glob("*.json")}
        if actual_paths - previous_paths - desired_paths:
            raise ValueError("Unmanaged files in prepared documents directory; choose a fresh destination")
        for doc, rendered in prepared.values():
            target = root / doc.path
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                write_text_atomic(target, rendered, newline="")
        evaluation = question_bytes.decode("utf-8")
        if not bundle.evaluation_path.exists() or bundle.evaluation_path.read_bytes() != evaluation.encode("utf-8"):
            write_text_atomic(bundle.evaluation_path, evaluation, newline="")
        live_paths = {root / d.path for d in bundle.documents}
        for stale in bundle.documents_root.glob("*.json"):
            if stale not in live_paths:
                stale.unlink()
        rendered_manifest = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if not bundle.manifest_path.exists() or bundle.manifest_path.read_text(encoding="utf-8") != rendered_manifest:
            write_text_atomic(bundle.manifest_path, rendered_manifest, newline="")
        validate_dataset_bundle(bundle)
    return bundle


def load_dataset_bundle(root: Path | str) -> DatasetBundle:
    root = Path(root).resolve()
    with FileLock(str(root / "bundle.lock")):
        data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("Unsupported dataset manifest version; prepare a fresh bundle")
        bundle = DatasetBundle(
            data["dataset"],
            root,
            tuple(DatasetDocument(**doc) for doc in data["documents"]),
            data["title_key"],
            data["body_key"],
            data["id_key"],
        )
        validate_dataset_bundle(bundle)
        return bundle


def validate_dataset_bundle(bundle: DatasetBundle) -> None:
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    current = (
        manifest["dataset"],
        manifest["title_key"],
        manifest["body_key"],
        manifest["id_key"],
        tuple(DatasetDocument(**doc) for doc in manifest["documents"]),
    )
    if current != (bundle.dataset, bundle.title_key, bundle.body_key, bundle.id_key, bundle.documents):
        raise ValueError("Dataset bundle changed; resolve the current manifest before reading")
    if hashlib.sha256(bundle.evaluation_path.read_bytes()).hexdigest() != manifest["evaluation_sha256"]:
        raise ValueError(f"Incomplete dataset export: {bundle.evaluation_path}")
    seen: set[str] = set()
    for doc in bundle.documents:
        path = (bundle.root / doc.path).resolve()
        if path.parent != bundle.documents_root.resolve() or path.suffix != ".json":
            raise ValueError(f"Invalid document path: {doc.path}")
        if doc.document_id in seen:
            raise ValueError(f"Duplicate document ID: {doc.document_id}")
        seen.add(doc.document_id)
        if hashlib.sha256(path.read_bytes()).hexdigest() != doc.content_hash:
            raise ValueError(f"Incomplete dataset export: {path}")
    actual = {p.resolve() for p in bundle.documents_root.glob("*.json")}
    if actual != {(bundle.root / d.path).resolve() for d in bundle.documents}:
        raise ValueError("Dataset document inventory does not match its manifest")


def snapshot_dataset_bundle(
    bundle: DatasetBundle,
    state_dir: Path,
    *,
    document_ids: Sequence[str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
) -> list[tuple[ChunkPointer, str]]:
    """Return a complete native-memo snapshot under a stable source namespace."""
    selected = set(document_ids) if document_ids is not None else None
    by_name = {Path(d.path).name: d.document_id for d in bundle.documents}
    if selected is not None and (unknown := selected - set(by_name.values())):
        raise ValueError(f"Document IDs missing from {bundle.dataset} manifest: {sorted(unknown)}")
    with FileLock(str(bundle.root / "bundle.lock")):
        validate_dataset_bundle(bundle)
        with CorpusIngestor(
            bundle.documents_root,
            state_dir,
            included_patterns=("*.json",),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        ) as ingestor:
            ingestor.poll()
            return [
                (
                    ChunkPointer(
                        ptr.chunk_id,
                        f"{bundle.dataset}/documents/{ptr.rel_posix_path}",
                        ptr.start_line,
                        ptr.end_line,
                        ptr.title,
                        ptr.ast_island_start,
                        ptr.ast_island_end,
                    ),
                    text,
                )
                for ptr, text in ingestor.snapshot()
                if selected is None or by_name[ptr.rel_posix_path] in selected
            ]
