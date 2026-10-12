"""Single-Pass Cocoindex LLM-Wiki Indexer & Disk-First Storage Engine."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import posixpath
import re
import shutil
import sqlite3
import sys
import urllib.error
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import spacy

logger = logging.getLogger(__name__)

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.file import PatternFilePathMatcher
from spacy.kb import InMemoryLookupKB

from yaw.indexer.canonicalize import (
    _node_id,
    create_in_memory_kb,
    normalize_token,
    resolve,
    slugify,
    sync_kb_from_db,
)
from yaw.nlp import (
    ExtractedNode,
    ExtractedTriple,
    align_span,
    build_leaf_pipeline,
    call_ollama_json,
    embed_texts,
    extract_leaf_taxonomy,
    format_document,
    format_space_r_doc,
    format_space_s,
    get_leaf_pipeline,
    is_ollama_online,
    parse_extracted_triples,
    parse_raw_triples_json,
    query_live_predicates,
)
from yaw.nlp.components import get_base_spacy
from yaw.storage import (
    apply_pragmas,
    clean_db,
    ensure_schema_views,
    get_db,
    insert_entity_vec,
    insert_fact_vec,
    serialize_embedding,
)

from .. import config

DB_KEY = coco.ContextKey[sqlite.ManagedConnection]("vault_db")
_active_managed_conn: sqlite.ManagedConnection | None = None
KB_KEY = coco.ContextKey[InMemoryLookupKB]("vault_kb")
_active_managed_kb: InMemoryLookupKB | None = None

_splitter = RecursiveSplitter()

VAULT_MATCHER = PatternFilePathMatcher(
    included_patterns=list(config.INCLUDED_PATTERNS)
    if config.INCLUDED_PATTERNS
    else None,
    excluded_patterns=list(config.EXCLUDED_PATTERNS)
    if config.EXCLUDED_PATTERNS
    else None,
)

_RE_TITLE_YAML = re.compile(r"^title:\s*(.+)$", re.MULTILINE)
_RE_TITLE_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_RE_HEADING = re.compile(r"^#+\s+(.+)$", re.MULTILINE)
_RE_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")


@dataclass
class NoteMetaRow:
    rel_path: str
    title: str
    category: str
    modified_at: float


@dataclass
class NoteChunkRow:
    chunk_id: str
    entity_uri: str
    rel_path: str
    ordinal: int
    heading: str
    heading_slug: str
    line_start: int
    line_end: int
    embedding: list[float]


@dataclass
class WikilinkRow:
    source_uri: str
    target_uri: str
    link_label: str


NoteEdgeRow = WikilinkRow


@dataclass
class LeafNodeRow:
    node_id: str
    source_leaf_id: str
    name: str
    namespace: str
    definition: str


@dataclass
class LeafRelationRow:
    relation_id: str
    source_leaf_id: str
    subject_id: str
    predicate: str
    predicate_id: str
    object_id: str
    weight: float = 1.0


class MarkdownChunk(NamedTuple):
    chunk_id: str
    heading: str
    heading_slug: str
    text: str
    ordinal: int
    line_start: int
    line_end: int


def build_entity_uri(rel: str, ordinal: int) -> str:
    """Canonical URI for chunk addressing."""
    return f"{rel}#chunk:{ordinal}"


def extract_title(rel: str, text: str) -> str:
    """Extract document title from frontmatter YAML, H1 markdown heading, or fallback to file stem."""
    m = _RE_TITLE_YAML.search(text) or _RE_TITLE_H1.search(text)
    return m.group(1).strip() if m else Path(rel).stem.replace("-", " ").title()


def chunk_markdown(content: str) -> list[MarkdownChunk]:
    """Split markdown text into chunks with exact byte offsets and line numbers."""
    if not content or not content.strip():
        return []
    doc_m = _RE_TITLE_YAML.search(content) or _RE_TITLE_H1.search(content)
    chunks = _splitter.split(
        content,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        language="markdown",
    )
    res: list[MarkdownChunk] = []
    cur_h = doc_m.group(1).strip() if doc_m else ""
    for ord_idx, c in enumerate(chunks or []):
        h = _RE_HEADING.search(c.text)
        if h:
            cur_h = h.group(1).strip()
        cid = hashlib.sha256(f"{cur_h}\n{c.text}".encode()).hexdigest()[:16]

        if hasattr(c.start, "line"):
            l_start = c.start.line
            l_end = c.end.line
        else:
            l_start = content[: c.start].count("\n") + 1
            l_end = l_start + c.text.count("\n")

        res.append(
            MarkdownChunk(
                chunk_id=cid,
                heading=cur_h,
                heading_slug=slugify(cur_h),
                text=c.text,
                ordinal=ord_idx,
                line_start=l_start,
                line_end=l_end,
            )
        )
    return res


@functools.lru_cache(maxsize=1)
def get_valid_targets(vault_root: Path | None = None) -> set[str]:
    """Retrieve all markdown files in vault matching pattern inclusions."""
    root = vault_root or config.VAULT_ROOT
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.md")
        if VAULT_MATCHER.is_file_included(p.relative_to(root))
    }


def clean_edge_target(
    source_rel: str, raw_target: str, valid_targets: set[str]
) -> str | None:
    """Normalize and validate wikilink targets, pruning self-loops and external links."""
    if not raw_target or not raw_target.strip():
        return None
    raw = raw_target.strip()
    if "://" in raw or raw.startswith("mailto:"):
        return None
    t = raw.split("?")[0].split("#")[0].split("§")[0].strip()
    ext = posixpath.splitext(t)[1].lower()
    if not t or (ext and ext != ".md"):
        return None
    if not t.endswith(".md"):
        t = f"{t}.md"
    norm = posixpath.normpath(posixpath.join(posixpath.dirname(source_rel), t))
    for cand in (t, f"wiki/{t}", f"raw/{t}", norm):
        if cand in valid_targets and cand != source_rel:
            return cand
    suffix = f"/{t}"
    for vt in valid_targets:
        if vt.endswith(suffix) and vt != source_rel:
            return vt
    return None


def extract_wikilinks(
    rel: str,
    chunks: list[MarkdownChunk],
    valid_targets: set[str],
) -> list[WikilinkRow]:
    """Extract chunk-level wikilink edges."""
    edges: dict[tuple[str, str], str] = {}
    for chunk in chunks:
        cur_uri = build_entity_uri(rel, chunk.ordinal)
        for m in _RE_WIKILINK.finditer(chunk.text):
            tgt = clean_edge_target(rel, m.group(1).strip(), valid_targets)
            if tgt:
                k = (cur_uri, tgt)
                lbl = (m.group(2) or "").strip()
                if k not in edges or (lbl and not edges[k]):
                    edges[k] = lbl
    return [
        WikilinkRow(source_uri=s, target_uri=t, link_label=lbl)
        for (s, t), lbl in edges.items()
    ]


extract_edges = extract_wikilinks


def _relation_id(source_leaf_id: str, sub: str, pred: str, obj: str) -> str:
    s = slugify(sub) or normalize_token(sub)
    p = slugify(pred) or normalize_token(pred)
    o = slugify(obj) or normalize_token(obj)
    sep = ":" if "#" in source_leaf_id else "#"
    return f"{source_leaf_id}{sep}{s}:{p}:{o}"


_CATEGORY_NET_EXCEPTIONS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
    json.JSONDecodeError,
    RuntimeError,
)

_EXTRACTION_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
    KeyError,
    IndexError,
    AssertionError,
)

_CANON_EXCEPTIONS = (
    sqlite3.Error,
    ValueError,
    RuntimeError,
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
    KeyError,
    IndexError,
)


def classify_freeform_llm(text: str) -> str:
    """Prompt freeform LLM classification when TAXONOMY_NAMESPACES is empty."""
    if not is_ollama_online():
        return ""
    prompt = (
        "Classify the following text into a single concise category label (one or two words lowercase, e.g. concepts, notes, reference):\n\n"
        f"{text[:1000]}\n\nCategory:"
    )
    try:
        data = call_ollama_json(
            "/api/chat",
            payload={
                "model": config.CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout_s=30.0,
        )
        raw_cat = str(data.get("message", {}).get("content", ""))
        cleaned = (
            re.sub(r"[^\w\s-]", "", raw_cat.strip().lower()).split("\n")[0].strip()
        )
        return cleaned or "notes"
    except _CATEGORY_NET_EXCEPTIONS as err:
        logger.warning("Category inference failed, defaulting to 'notes': %s", err)
        return "notes"


def _insert_query_fts(
    rel: str, chunks: list[MarkdownChunk], nodes: list[ExtractedNode] | None = None
) -> None:
    """Insert chunk and taxonomy texts into contentless FTS5 virtual table."""
    with sqlite3.connect(config.DB_PATH) as conn:
        apply_pragmas(conn)
        with conn:
            for c in chunks:
                uri = build_entity_uri(rel, c.ordinal)
                conn.execute(
                    "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                    (uri, rel, c.heading, c.text),
                )
            if nodes:
                for n in nodes:
                    conn.execute(
                        "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
                        (
                            _node_id(n.name, n.namespace),
                            rel,
                            n.name,
                            f"{n.name}: {n.definition}",
                        ),
                    )


@coco.lifespan
async def vault_lifespan(builder: coco.EnvironmentBuilder):
    """Lifecycle provider establishing SQLite connection and synchronized InMemoryLookupKB."""
    global _active_managed_conn, _active_managed_kb
    (config.VAULT_ROOT / ".cache" / "cocoindex").mkdir(parents=True, exist_ok=True)
    builder.settings.db_path = config.VAULT_ROOT / ".cache" / "cocoindex"
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.DB_PATH) as init_c:
        apply_pragmas(init_c)
        init_c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS query_fts USING fts5("
            "entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, "
            "content='', contentless_unindexed=1, contentless_delete=1"
            ")"
        )
    conn = sqlite.connect(config.DB_PATH, load_vec=True)
    _active_managed_conn = conn
    builder.provide(DB_KEY, conn)

    nlp = get_base_spacy()
    kb = create_in_memory_kb(nlp)
    with sqlite3.connect(config.DB_PATH) as raw_c:
        sync_kb_from_db(kb, raw_c)
    _active_managed_kb = kb
    builder.provide(KB_KEY, kb)

    try:
        yield
    finally:
        conn.close()
        _active_managed_conn = None
        _active_managed_kb = None


def _register_raw_nodes(
    nodes: Iterable[ExtractedNode],
    source_leaf_id: str,
    raw_nodes: dict[str, tuple[str, ExtractedNode]],
    node_provenance: dict[str, set[str]] | None = None,
) -> None:
    """Register extracted nodes in raw_nodes map under canonical node_id and track chunk provenance."""
    for node in nodes:
        nid = _node_id(node.name, node.namespace)
        if nid not in raw_nodes:
            raw_nodes[nid] = (source_leaf_id, node)
        if node_provenance is not None:
            node_provenance.setdefault(nid, set()).add(source_leaf_id)


@functools.lru_cache(maxsize=8)
def _get_steered_pipeline(
    active_predicates_tuple: tuple[str, ...],
    db_path_str: str | None = None,
) -> spacy.language.Language:
    """Build and cache a steered leaf pipeline parameterized by active predicates."""
    if active_predicates_tuple:
        fewshot_cfg: dict[str, Any] = {
            "@misc": "yaw.CanonicalExamples.v1",
            "active_predicates": list(active_predicates_tuple),
            "db_path": db_path_str,
        }
        return build_leaf_pipeline(fewshot_examples=fewshot_cfg, enable_rel=True)
    return get_leaf_pipeline()


def _extract_chunk_taxonomy(
    text: str,
    source_leaf_id: str,
    raw_nodes: dict[str, tuple[str, ExtractedNode]],
    all_triples: list[tuple[str, ExtractedTriple]],
    doc_categories: list[str],
    node_provenance: dict[str, set[str]] | None = None,
    nlp: spacy.language.Language | None = None,
    active_predicates: Sequence[str] | None = None,
) -> None:
    """Extract leaf taxonomy entities and triples for a single text chunk."""
    if not text or not text.strip():
        return
    try:
        taxonomy = extract_leaf_taxonomy(text, nlp=nlp)
        if taxonomy.category:
            doc_categories.append(taxonomy.category)

        _register_raw_nodes(taxonomy.nodes, source_leaf_id, raw_nodes, node_provenance)
        for triple in taxonomy.triples:
            all_triples.append((source_leaf_id, triple))
            _register_raw_nodes(
                (triple.subject, triple.predicate, triple.object),
                source_leaf_id,
                raw_nodes,
                node_provenance,
            )

        # ADR 001: Defensive single-pass triple fallback if taxonomy.triples is empty
        if not taxonomy.triples:
            raw_fallback = parse_extracted_triples(
                text,
                base_predicates=list(active_predicates)
                if active_predicates is not None
                else None,
            )
            for triple in raw_fallback:
                all_triples.append((source_leaf_id, triple))
                _register_raw_nodes(
                    (triple.subject, triple.predicate, triple.object),
                    source_leaf_id,
                    raw_nodes,
                    node_provenance,
                )
    except _EXTRACTION_EXCEPTIONS as nlp_err:
        logger.warning(
            "Leaf taxonomy extraction failed for %s: %s", source_leaf_id, nlp_err
        )


async def _resolve_and_store_nodes(
    raw_nodes: dict[str, tuple[str, ExtractedNode]],
    nodes_target: sqlite.TableTarget[LeafNodeRow],
    conn: sqlite3.Connection | None,
    kb: InMemoryLookupKB | None = None,
) -> None:
    """Step 4: Embed in Space S, resolve canonical entities, and persist LeafNodeRow."""
    if not raw_nodes:
        return
    formatted = [
        format_space_s(node.name, node.definition) for _, node in raw_nodes.values()
    ]
    node_vecs = await embed_texts(formatted)

    for (node_id, (source_leaf_id, node)), vec in zip(raw_nodes.items(), node_vecs):
        node_vec = vec.tolist()
        try:
            resolve(
                name=node.name,
                kind="entity",
                definition_ctx=node.definition,
                conn=conn,
                embedding=node_vec,
                source_leaf_id=source_leaf_id,
                namespace=node.namespace,
                kb=kb,
                canonical_id=node.canonical_id,
            )

            if conn is not None:
                vec_bytes = serialize_embedding(node_vec)
                if vec_bytes:
                    insert_entity_vec(conn, node_id, node.namespace, vec_bytes)
        except _CANON_EXCEPTIONS as res_err:
            logger.warning("Entity resolution failed for %s: %s", node.name, res_err)

        nodes_target.declare_row(
            row=LeafNodeRow(
                node_id=node_id,
                source_leaf_id=source_leaf_id,
                name=node.name,
                namespace=node.namespace,
                definition=node.definition,
            )
        )


def _resolve_and_store_relations(
    all_triples: list[tuple[str, ExtractedTriple]],
    relations_target: sqlite.TableTarget[LeafRelationRow],
    conn: sqlite3.Connection | None,
    seen_rel_ids: set[str],
    kb: InMemoryLookupKB | None = None,
    fact_embeddings: dict[str, list[float]] | None = None,
) -> None:
    """Step 5: Resolve predicate universals, persist LeafRelationRow, and index fact vectors in Space R."""
    for source_leaf_id, triple in all_triples:
        try:
            resolve(
                name=triple.predicate.name,
                kind="predicate",
                definition_ctx="",
                conn=conn,
                source_leaf_id=source_leaf_id,
                namespace="predicates",
                kb=kb,
                canonical_id=triple.predicate.canonical_id,
            )
        except _CANON_EXCEPTIONS as res_pred_err:
            logger.warning(
                "Predicate resolution failed for %s: %s",
                triple.predicate.name,
                res_pred_err,
            )

        sub_id = _node_id(triple.subject.name, triple.subject.namespace)
        pred_id = _node_id(triple.predicate.name, triple.predicate.namespace)
        obj_id = _node_id(triple.object.name, triple.object.namespace)
        rel_id = _relation_id(
            source_leaf_id,
            triple.subject.name,
            triple.predicate.name,
            triple.object.name,
        )
        if rel_id not in seen_rel_ids:
            seen_rel_ids.add(rel_id)
            relations_target.declare_row(
                row=LeafRelationRow(
                    relation_id=rel_id,
                    source_leaf_id=source_leaf_id,
                    subject_id=sub_id,
                    predicate=triple.predicate.name,
                    predicate_id=pred_id,
                    object_id=obj_id,
                    weight=1.0,
                )
            )
            if conn is not None and fact_embeddings:
                fact_vec = fact_embeddings.get(rel_id)
                if fact_vec:
                    emb_bytes = serialize_embedding(fact_vec)
                    if emb_bytes:
                        insert_fact_vec(conn, rel_id, emb_bytes)


def _store_chunk_concept_edges(
    raw_nodes: dict[str, tuple[str, ExtractedNode]],
    chunks: list[MarkdownChunk],
    rel: str,
    relations_target: sqlite.TableTarget[LeafRelationRow],
    seen_rel_ids: set[str],
    node_provenance: dict[str, set[str]] | None = None,
) -> None:
    """Declare bidirectional chunk-to-concept grounding edges."""
    if not (raw_nodes and chunks):
        return
    for nid, (first_source, node) in raw_nodes.items():
        node_name_lower = node.name.casefold()
        prov_sources: set[str] = set()
        if node_provenance and nid in node_provenance:
            prov_sources.update(node_provenance[nid])
        if first_source:
            prov_sources.add(first_source)

        target_chunks = [
            c
            for c in chunks
            if any(f"#chunk:{c.ordinal}" in s for s in prov_sources)
            or node_name_lower in c.text.casefold()
        ]
        for c in target_chunks:
            chunk_uri = build_entity_uri(rel, c.ordinal)
            down_id = f"{rel}#{nid}:manifests_in:{chunk_uri}"
            if down_id not in seen_rel_ids:
                seen_rel_ids.add(down_id)
                relations_target.declare_row(
                    row=LeafRelationRow(
                        relation_id=down_id,
                        source_leaf_id=rel,
                        subject_id=nid,
                        predicate="manifests_in",
                        predicate_id="predicates:manifests-in",
                        object_id=chunk_uri,
                        weight=1.0,
                    )
                )
            up_id = f"{rel}#{chunk_uri}:mentions:{nid}"
            if up_id not in seen_rel_ids:
                seen_rel_ids.add(up_id)
                relations_target.declare_row(
                    row=LeafRelationRow(
                        relation_id=up_id,
                        source_leaf_id=rel,
                        subject_id=chunk_uri,
                        predicate="mentions",
                        predicate_id="predicates:mentions",
                        object_id=nid,
                        weight=0.2,
                    )
                )


async def _declare_chunk_embeddings(
    chunks: list[MarkdownChunk],
    title: str,
    rel: str,
    chunks_target: sqlite.TableTarget[NoteChunkRow],
) -> None:
    """Format prompts, batch embed chunks, and declare NoteChunkRow instances."""
    if not chunks:
        return
    chunk_prompts = [format_document(c.text, title=title) for c in chunks]
    vectors = await embed_texts(chunk_prompts)
    for c, vec in zip(chunks, vectors):
        chunks_target.declare_row(
            row=NoteChunkRow(
                chunk_id=c.chunk_id,
                entity_uri=build_entity_uri(rel, c.ordinal),
                rel_path=rel,
                ordinal=c.ordinal,
                heading=c.heading,
                heading_slug=c.heading_slug,
                line_start=c.line_start,
                line_end=c.line_end,
                embedding=vec.tolist(),
            )
        )


async def _embed_relation_facts(
    all_triples: list[tuple[str, ExtractedTriple]],
) -> dict[str, list[float]]:
    """Render deterministic relation strings in Space R and embed them in batches."""
    if not all_triples:
        return {}
    unique_facts: dict[str, str] = {}
    for source_leaf_id, triple in all_triples:
        rel_id = _relation_id(
            source_leaf_id,
            triple.subject.name,
            triple.predicate.name,
            triple.object.name,
        )
        if rel_id not in unique_facts:
            fact_str = (
                f"{triple.subject.name} {triple.predicate.name} {triple.object.name}"
            )
            unique_facts[rel_id] = fact_str

    if not unique_facts:
        return {}

    fact_rel_ids = list(unique_facts.keys())
    fact_prompts = [
        format_space_r_doc(title="", text=unique_facts[rid]) for rid in fact_rel_ids
    ]
    try:
        fact_vecs = await embed_texts(fact_prompts)
        return {rid: vec.tolist() for rid, vec in zip(fact_rel_ids, fact_vecs)}
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as emb_err:
        logger.warning("Failed to embed relation facts: %s", emb_err)
        return {}


async def _persist_canonical_knowledge(
    raw_nodes: dict[str, tuple[str, ExtractedNode]],
    all_triples: list[tuple[str, ExtractedTriple]],
    chunks: list[MarkdownChunk],
    rel: str,
    nodes_target: sqlite.TableTarget[LeafNodeRow],
    relations_target: sqlite.TableTarget[LeafRelationRow],
    node_provenance: dict[str, set[str]] | None = None,
) -> None:
    """Manage SQLite connection lifecycle and persist canonical nodes and relation edges."""
    conn: sqlite3.Connection | None = None
    try:
        conn = get_db(load_vec=True)
    except (sqlite3.Error, OSError, RuntimeError) as db_err:
        logger.warning(
            "Could not acquire db connection for canonicalization: %s", db_err
        )
        conn = None

    kb = _active_managed_kb
    try:
        await _resolve_and_store_nodes(raw_nodes, nodes_target, conn, kb=kb)
        seen_rel_ids: set[str] = set()
        if all_triples:
            fact_embeddings: dict[str, list[float]] = {}
            if conn is not None:
                fact_embeddings = await _embed_relation_facts(all_triples)
            _resolve_and_store_relations(
                all_triples,
                relations_target,
                conn,
                seen_rel_ids,
                kb=kb,
                fact_embeddings=fact_embeddings,
            )
        _store_chunk_concept_edges(
            raw_nodes,
            chunks,
            rel,
            relations_target,
            seen_rel_ids,
            node_provenance=node_provenance,
        )
    finally:
        if conn is not None:
            try:
                conn.commit()
            except (sqlite3.Error, OSError) as commit_err:
                logger.debug("Failed to commit connection: %s", commit_err)
            try:
                conn.close()
            except (sqlite3.Error, OSError) as close_err:
                logger.debug("Failed to close connection: %s", close_err)


@coco.fn(memo=True)
async def process_vault_raw(
    file: localfs.File,
    meta_target: sqlite.TableTarget[NoteMetaRow],
    wikilinks_target: sqlite.TableTarget[WikilinkRow],
    chunks_target: sqlite.TableTarget[NoteChunkRow],
    nodes_target: sqlite.TableTarget[LeafNodeRow],
    relations_target: sqlite.TableTarget[LeafRelationRow],
) -> None:
    """Single-pass processor: chunks text, extracts spacy-llm leaf ontology, embeds, writes disk offsets."""
    timeout_ctx = (
        coco.timeout(timedelta(seconds=config.INDEXING_TIMEOUT_S))
        if config.INDEXING_TIMEOUT_S > 0
        else nullcontext()
    )
    with timeout_ctx:
        rel = file.file_path.path.relative_to(config.VAULT_ROOT).as_posix()
        txt = await file.read_text()
        title = extract_title(rel, txt)
        mtime = Path(file.file_path.path).stat().st_mtime

        chunks = chunk_markdown(txt)
        valid_targets = get_valid_targets(config.VAULT_ROOT)

        # 1. Structural Wikilink Edges
        for edge in extract_wikilinks(rel, chunks, valid_targets):
            wikilinks_target.declare_row(row=edge)

        # 2. Chunks & Offsets (Zero chunk content in SQLite!)
        await _declare_chunk_embeddings(chunks, title, rel, chunks_target)

        # 3. Leaf Taxonomy per note document (Stage 1 leaf taxonomy via spacy-llm)
        raw_nodes: dict[str, tuple[str, ExtractedNode]] = {}
        all_triples: list[tuple[str, ExtractedTriple]] = []
        doc_categories: list[str] = []
        node_provenance: dict[str, set[str]] = defaultdict(set)

        active_predicates = query_live_predicates(config.DB_PATH, limit=20)
        if active_predicates:
            steered_nlp = _get_steered_pipeline(
                tuple(active_predicates), str(config.DB_PATH)
            )
        else:
            steered_nlp = None

        if chunks:
            for c in chunks:
                chunk_leaf_id = build_entity_uri(rel, c.ordinal)
                _extract_chunk_taxonomy(
                    c.text,
                    chunk_leaf_id,
                    raw_nodes,
                    all_triples,
                    doc_categories,
                    node_provenance=node_provenance,
                    nlp=steered_nlp,
                    active_predicates=active_predicates,
                )
        else:
            _extract_chunk_taxonomy(
                txt,
                rel,
                raw_nodes,
                all_triples,
                doc_categories,
                node_provenance=node_provenance,
                nlp=steered_nlp,
                active_predicates=active_predicates,
            )

        _insert_query_fts(rel, chunks, [node for _, node in raw_nodes.values()])

        # 4. Categorization
        if doc_categories:
            category = max(set(doc_categories), key=doc_categories.count)
        elif not config.TAXONOMY_NAMESPACES:
            category = classify_freeform_llm(txt)
        else:
            category = rel.split("/")[0]
        meta_target.declare_row(
            row=NoteMetaRow(
                rel_path=rel, title=title, category=category, modified_at=mtime
            )
        )

        # 5. Canonicalization & Knowledge Graph Persistence
        await _persist_canonical_knowledge(
            raw_nodes=raw_nodes,
            all_triples=all_triples,
            chunks=chunks,
            rel=rel,
            nodes_target=nodes_target,
            relations_target=relations_target,
            node_provenance=node_provenance,
        )


async def vault_pipeline_main(vault_dir: Path) -> None:
    """Builds and mounts the single-pass Cocoindex vault pipeline."""
    conn = coco.use_context(DB_KEY)

    meta_s = await sqlite.TableSchema[Any].from_class(
        NoteMetaRow, primary_key=["rel_path"]
    )
    wikilinks_s = await sqlite.TableSchema[Any].from_class(
        WikilinkRow, primary_key=["source_uri", "target_uri"]
    )
    chunks_s = await sqlite.TableSchema[Any].from_class(
        NoteChunkRow, primary_key=["entity_uri"]
    )
    leaf_nodes_s = await sqlite.TableSchema[Any].from_class(
        LeafNodeRow, primary_key=["node_id", "source_leaf_id"]
    )
    leaf_relations_s = await sqlite.TableSchema[Any].from_class(
        LeafRelationRow, primary_key=["relation_id"]
    )

    meta_t = await sqlite.mount_table_target(DB_KEY, "note_meta", meta_s)
    wikilinks_t = await sqlite.mount_table_target(DB_KEY, "wikilinks", wikilinks_s)
    chunks_t = await sqlite.mount_table_target(DB_KEY, "note_chunks", chunks_s)
    leaf_nodes_t = await sqlite.mount_table_target(DB_KEY, "leaf_nodes", leaf_nodes_s)
    leaf_relations_t = await sqlite.mount_table_target(
        DB_KEY, "leaf_relations", leaf_relations_s
    )

    with conn.transaction() as c:
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_rel ON note_chunks(rel_path)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_wikilinks_tgt ON wikilinks(target_uri)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_leaf_nodes_source ON leaf_nodes(source_leaf_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_leaf_relations_source ON leaf_relations(source_leaf_id)"
        )
        c.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS query_fts USING fts5("
            "entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, "
            "content='', contentless_unindexed=1, contentless_delete=1"
            ")"
        )
        ensure_schema_views(c)

    files = localfs.walk_dir(
        vault_dir, recursive=True, path_matcher=VAULT_MATCHER, live=True
    )
    await coco.mount_each(
        process_vault_raw,
        files.items(),
        meta_t,
        wikilinks_t,
        chunks_t,
        leaf_nodes_t,
        leaf_relations_t,
    )


_app_counter = 0


def sync_vault(
    vault_dir: Path | None = None,
    force: bool = False,
    verbose: bool = False,
) -> int:
    """Authoritative entrypoint to sync vault index."""
    global _app_counter
    _app_counter += 1
    target_vault = Path(vault_dir).resolve() if vault_dir else config.VAULT_ROOT
    if vault_dir:
        config.set_vault(target_vault)

    cache_path = target_vault / ".cache" / "cocoindex"
    has_coco_cache = cache_path.is_dir() and (
        (cache_path / "data.mdb").is_file()
        or (cache_path / "mdb" / "data.mdb").is_file()
    )
    db_exists = config.DB_PATH.is_file()

    if _active_managed_conn is not None:
        try:
            _active_managed_conn._conn.close()
        except sqlite3.Error as close_err:
            logger.warning(
                "Error closing active managed connection during clean rebuild: %s",
                close_err,
            )

    # Invariant: yaw.db and cocoindex LMDB cache must be synchronized.
    # If one exists without the other, or if force is requested, re-initialize cleanly.
    if force or (db_exists != has_coco_cache):
        clean_db(config.DB_PATH)
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
        if verbose:
            sys.stderr.write("[clean-rebuild] Wiped yaw.db and cocoindex cache.\n")

    if _active_managed_conn is not None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(config.DB_PATH) as init_c:
            apply_pragmas(init_c)
            init_c.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS query_fts USING fts5("
                "entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, "
                "content='', contentless_unindexed=1, contentless_delete=1"
                ")"
            )
        new_mc = sqlite.connect(config.DB_PATH, load_vec=True)
        _active_managed_conn._conn = new_mc._conn
        _active_managed_conn._loaded_extensions = new_mc._loaded_extensions

    get_valid_targets.cache_clear()
    stage1 = coco.App(
        coco.AppConfig(
            name=f"LLMVaultIndexer_{_app_counter}", max_inflight_components=1
        ),
        vault_pipeline_main,
        target_vault,
    )
    stage1.update_blocking(report_to_stdout=verbose)
    return 0


__all__ = [
    "LeafNodeRow",
    "LeafRelationRow",
    "MarkdownChunk",
    "NoteChunkRow",
    "NoteEdgeRow",
    "NoteMetaRow",
    "align_span",
    "build_entity_uri",
    "chunk_markdown",
    "classify_freeform_llm",
    "clean_edge_target",
    "extract_title",
    "extract_wikilinks",
    "parse_extracted_triples",
    "parse_raw_triples_json",
    "process_vault_raw",
    "sync_vault",
]
