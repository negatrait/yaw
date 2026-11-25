"""Unmocked real-substrate tests for L3 yaw/wiki cards, dual-root CocoIndex indexing, linting, and scouting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

import dspy
import numpy as np
import pytest
import yaml
from cocoindex.connectorkits.fingerprint import fingerprint_str
from numpy.typing import NDArray

from yaw.core import (
    ChunkPointer,
    HydratedPassage,
    YawConfig,
    YawRuntimeError,
    format_query_for_embedding,
)
from yaw.pipeline import build_wiki_path_matcher, close_pipeline, start_pipeline, update_pipeline_once
from yaw.qa import create_lm
from yaw.retrieval import (
    SearchResult,
    build_retrieval_index,
    expand_citation_neighbors,
    hydrate_ranked_passages,
    open_readonly_db,
    search_index,
    verify_source_fingerprint,
)
from yaw.wiki import (
    WikiCardSource,
    is_fresh_passage,
    is_ignored_wiki_rel,
    lint_wiki,
    list_wiki_cards,
    parse_wiki_card,
    read_wiki_card,
    render_wiki_card,
    should_queue_scout,
    synthesize_scout_card,
    validate_wiki_slug,
    write_wiki_card,
)


def _topic_embed(texts: list[str], *, dim: int = 32) -> NDArray[np.float32]:
    topics = (
        ("cobalt", "lunar", "rover", "battery"),
        ("quartz", "arctic", "turbine", "telemetry"),
        ("obsidian", "synthesis", "bridge", "notes"),
    )
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        low = text.lower()
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (dim + len(digest) - 1) // len(digest)
        base = (np.frombuffer(digest * repeats, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.01
        for t_idx, kws in enumerate(topics):
            for kw in kws:
                if kw in low:
                    base[t_idx] += 3.0
        rows[row_idx] = base / max(float(np.linalg.norm(base)), 1e-12)
    return rows


def _make_cfg(
    root: Path,
    *,
    wiki_root: Path | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
    min_chunk_size: int = 20,
) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        wiki_root=wiki_root,
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )


def test_wiki_card_render_parse_fenced_immunity_and_slug_validation(tmp_path: Path) -> None:
    wiki_dir = tmp_path / "wiki"
    src_hash = fingerprint_str("Cobalt rovers use solid-state cells.").hex()
    src = WikiCardSource(rel_path="guide.md", start_line=2, end_line=2, source_hash=src_hash)

    card_path, wrote = write_wiki_card(
        wiki_dir,
        "topic/rover-notes.md",
        "Rover Synthesis",
        "```bash\n# Fake Fenced Heading\necho ok\n```\n\nCobalt rovers use solid-state cells.\n",
        sources=[src],
        kind="synthesis",
    )
    assert wrote is True
    assert card_path == (wiki_dir / "topic" / "rover-notes.md").resolve()

    raw_text = read_wiki_card(wiki_dir, "topic/rover-notes")
    assert "\ntitle:" not in raw_text
    assert raw_text.startswith("---\n")
    end_idx = raw_text.index("\n---\n", 4)
    meta = yaml.safe_load(raw_text[4:end_idx])
    assert "title" not in meta
    assert meta["kind"] == "synthesis"
    assert meta["stale"] is False
    assert meta["sources"][0]["source_hash"] == src_hash

    parsed = parse_wiki_card(card_path, wiki_dir)
    assert parsed.slug == "topic/rover-notes"
    assert parsed.rel_path == "topic/rover-notes.md"
    assert parsed.title == "Rover Synthesis"
    assert parsed.kind == "synthesis"
    assert parsed.stale is False
    assert parsed.sources == (src,)
    assert "# Fake Fenced Heading" in parsed.body

    # Idempotent write preserves updated_at and reports wrote=False.
    _, wrote_again = write_wiki_card(
        wiki_dir,
        "topic/rover-notes",
        "# Rover Synthesis",
        "# Rover Synthesis\n\n```bash\n# Fake Fenced Heading\necho ok\n```\n\nCobalt rovers use solid-state cells.",
        sources=[src],
        kind="synthesis",
    )
    assert wrote_again is False
    assert parse_wiki_card(card_path, wiki_dir).updated_at == parsed.updated_at

    # Fenced heading before real H1 is ignored by parser.
    fenced_first = wiki_dir / "fenced.md"
    fenced_first.write_text(
        "---\nkind: note\nstale: false\nsources: []\n---\n\n"
        "```md\n# Comment Inside Fence\n```\n\n# Real Heading\n\nBody.\n",
        encoding="utf-8",
    )
    parsed_fenced = parse_wiki_card(fenced_first, wiki_dir)
    assert parsed_fenced.title == "Real Heading"
    assert "Body." in parsed_fenced.body
    assert "# Comment Inside Fence" in parsed_fenced.body

    listed = list_wiki_cards(wiki_dir)
    assert [item["slug"] for item in listed] == ["fenced", "topic/rover-notes"]

    for bad_slug in (
        "",
        "   ",
        "../escape",
        "sub/../escape",
        "/leading-slash",
        "C:drive-letter",
        ".hidden",
        "dir/.hidden",
        "draft.tmp",
        "draft.tmp.md",
        "bad\nline",
    ):
        with pytest.raises(ValueError):
            validate_wiki_slug(bad_slug)

    rendered = render_wiki_card("Clean Title", "Body paragraph.", updated_at="2026-11-25T00:00:00Z")
    assert "# Clean Title\n\nBody paragraph.\n" in rendered
    with pytest.raises(YawRuntimeError) as exc_info:
        read_wiki_card(wiki_dir, "does-not-exist")
    assert exc_info.value.code == "INVALID_INPUT"


def test_dual_root_indexing_colliding_names_and_corpus_memo_isolation(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    wiki_dir = tmp_path / "wiki"
    corpus_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)

    (corpus_dir / "notes.md").write_text(
        "# Corpus Notes\nCobalt lunar rover battery telemetry remains nominal on Mare Tranquillitatis.\n",
        encoding="utf-8",
    )
    (corpus_dir / "arctic.md").write_text(
        "# Arctic Station\nQuartz turbine telemetry verifies subzero generator stability.\n",
        encoding="utf-8",
    )

    embed_calls: list[str] = []

    def tracked_embed(texts: list[str]) -> NDArray[np.float32]:
        embed_calls.extend(texts)
        return _topic_embed(texts)

    cfg = _make_cfg(tmp_path, wiki_root=wiki_dir)
    session = start_pipeline(cfg, embed_fn=tracked_embed, live=False)
    try:
        update_pipeline_once(session)
        initial_calls = len(embed_calls)
        assert initial_calls == 2
        embed_calls.clear()

        # Add colliding wiki_root/notes.md after initial corpus index.
        write_wiki_card(
            wiki_dir,
            "notes",
            "Wiki Notes",
            "Obsidian synthesis notes link Corpus Notes and Arctic Station telemetry bridges.",
        )
        update_pipeline_once(session)

        # Unchanged corpus_root files must NOT re-embed when a wiki_root card is added.
        assert len(embed_calls) == 1
        assert "Obsidian synthesis" in embed_calls[0]
        embed_calls.clear()

        # Editing the wiki card embeds ONLY the edited wiki card, zero corpus re-embeds.
        write_wiki_card(
            wiki_dir,
            "notes",
            "Wiki Notes",
            "Obsidian synthesis notes link Corpus Notes and Arctic Station with updated bridge evidence.",
        )
        update_pipeline_once(session)
        assert len(embed_calls) == 1
        assert "updated bridge evidence" in embed_calls[0]
    finally:
        close_pipeline(session)

    assert cfg.db_path is not None
    with open_readonly_db(cfg.db_path) as conn:
        rows = conn.execute("SELECT chunk_id, rel_path, title FROM chunks ORDER BY rel_path").fetchall()
        assert len(rows) == 3
        rel_paths = [str(r[1]) for r in rows]
        chunk_ids = [int(r[0]) for r in rows]
        assert rel_paths == ["arctic.md", "notes.md", "wiki:notes.md"]
        assert len(set(chunk_ids)) == 3

        idx = build_retrieval_index(
            conn,
            expected_embedding_model=cfg.embedding_model,
            expected_task_mode=cfg.embedding_task_mode,
        )
        assert idx.doc_titles.get("notes.md") == "Corpus Notes"
        assert idx.doc_titles.get("wiki:notes.md") == "Wiki Notes"

        q_emb = _topic_embed([format_query_for_embedding("obsidian synthesis notes bridge", cfg.embedding_task_mode)])[
            0
        ]
        res = search_index(idx, q_emb, query_text="obsidian synthesis notes bridge", conn=conn, top_k=5)
        passages = hydrate_ranked_passages(idx, res.ranked, corpus_dir, wiki_root=wiki_dir, top_k=5)
        by_rel = {hp.pointer.rel_posix_path: hp for hp in passages}
        assert "notes.md" in by_rel
        assert "wiki:notes.md" in by_rel
        assert by_rel["notes.md"].pointer.provenance == "corpus"
        assert by_rel["wiki:notes.md"].pointer.provenance == "wiki"
        assert "Cobalt lunar rover" in by_rel["notes.md"].text
        assert "Obsidian synthesis notes" in by_rel["wiki:notes.md"].text
        assert not by_rel["notes.md"].stale_on_disk
        assert not by_rel["wiki:notes.md"].stale_on_disk

        wiki_cid = by_rel["wiki:notes.md"].pointer.chunk_id
        neighbors = expand_citation_neighbors(
            idx,
            wiki_cid,
            corpus_root=corpus_dir,
            wiki_root=wiki_dir,
            conn=conn,
            top_k=5,
        )
        assert neighbors


def test_lint_wiki_detects_preserved_mtime_drift_post_reindex_drift_and_collisions(tmp_path: Path) -> None:
    # Unconfigured wiki_root -> ok=True, 0 cards.
    cfg_no_wiki = _make_cfg(tmp_path)
    empty_report = lint_wiki(cfg_no_wiki)
    assert empty_report.ok is True
    assert empty_report.cards_checked == 0

    corpus_dir = tmp_path / "corpus"
    wiki_dir = tmp_path / "wiki"
    corpus_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)

    # Empty wiki_root -> ok=True, 0 cards.
    cfg = _make_cfg(tmp_path, wiki_root=wiki_dir)
    assert lint_wiki(cfg).ok is True
    assert lint_wiki(cfg).cards_checked == 0

    guide_file = corpus_dir / "guide.md"
    line_1 = "# Reactor Guide"
    line_2 = "Core temperature limit is 420 Kelvin during steady cruise."
    guide_file.write_text(f"{line_1}\n{line_2}\n", encoding="utf-8")

    # Ignored temporary/hidden files under wiki_root.
    (wiki_dir / ".hidden.md").write_text("# Hidden\nIgnored.\n", encoding="utf-8")
    (wiki_dir / "scratch.tmp").write_text("Ignored.\n", encoding="utf-8")

    orig_hash = fingerprint_str(line_2).hex()
    ok_fp, actual_fp = verify_source_fingerprint(corpus_dir, "guide.md", 2, 2, orig_hash)
    assert ok_fp is True
    assert actual_fp == orig_hash

    card_path, _ = write_wiki_card(
        wiki_dir,
        "reactor-note",
        "Reactor Note",
        "Steady cruise temperature limit is 420 Kelvin.",
        sources=[WikiCardSource(rel_path="guide.md", start_line=2, end_line=2, source_hash=orig_hash)],
    )
    clean_report = lint_wiki(cfg)
    assert clean_report.ok is True
    assert clean_report.cards_checked == 1
    assert clean_report.stale_cards == ()
    assert clean_report.issues == ()

    # Pre-reindex same-size + preserved-mtime edit on disk.
    orig_stat = guide_file.stat()
    modified_line_2 = "Core temperature limit is 590 Kelvin during steady cruise."
    assert len(modified_line_2.encode("utf-8")) == len(line_2.encode("utf-8"))
    guide_file.write_text(f"{line_1}\n{modified_line_2}\n", encoding="utf-8")
    os.utime(guide_file, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))

    pre_reindex_report = lint_wiki(cfg)
    assert pre_reindex_report.ok is False
    assert pre_reindex_report.stale_cards == ("reactor-note.md",)
    assert any(issue.code == "STALE_SOURCE" and issue.severity == "error" for issue in pre_reindex_report.issues)

    # Post-reindex drift: chunks.source_hash updates to disk, but wiki card still cites orig_hash.
    session = start_pipeline(cfg, embed_fn=_topic_embed, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)

    post_reindex_report = lint_wiki(cfg)
    assert post_reindex_report.ok is False
    assert post_reindex_report.stale_cards == ("reactor-note.md",)
    assert any(issue.code == "STALE_SOURCE" for issue in post_reindex_report.issues)

    # Mark stale: true and verify repeated lint_wiki never clears stale or rewrites source_hash.
    write_wiki_card(
        wiki_dir,
        "reactor-note",
        "Reactor Note",
        "Steady cruise temperature limit is 420 Kelvin.",
        sources=[WikiCardSource(rel_path="guide.md", start_line=2, end_line=2, source_hash=orig_hash)],
        stale=True,
    )
    before_bytes = card_path.read_bytes()
    stale_report = lint_wiki(cfg)
    assert stale_report.ok is False
    assert any(issue.code == "CARD_MARKED_STALE" for issue in stale_report.issues)
    assert card_path.read_bytes() == before_bytes
    assert lint_wiki(cfg).to_dict()["ok"] is False

    # Colliding relative path emits REL_PATH_COLLISION warning while keeping ok=True for fresh cards.
    (corpus_dir / "notes.md").write_text("# Corpus Notes\nAlpha.\n", encoding="utf-8")
    notes_hash = fingerprint_str("Alpha.").hex()
    write_wiki_card(
        wiki_dir,
        "notes",
        "Wiki Notes",
        "Synthesized notes.",
        sources=[WikiCardSource(rel_path="notes.md", start_line=2, end_line=2, source_hash=notes_hash)],
    )
    # Remove the intentionally stale reactor-note to isolate collision warning ok=True check.
    card_path.unlink()
    collision_report = lint_wiki(cfg)
    assert collision_report.ok is True
    assert collision_report.cards_checked == 1
    assert any(
        issue.code == "REL_PATH_COLLISION" and issue.severity == "warning" and issue.card_rel_path == "notes.md"
        for issue in collision_report.issues
    )


def test_scout_gating_and_loopback_dspy_synthesis(tmp_path: Path) -> None:
    ptr = ChunkPointer(
        chunk_id=101,
        rel_path="alpha.md",
        source_hash=fingerprint_str("Cobalt rover battery.").hex(),
        start_line=1,
        end_line=2,
        title="Alpha",
    )
    hp = HydratedPassage(pointer=ptr, text="Cobalt rover battery.", score=1.2)

    assert should_queue_scout(None, [hp]) is False
    assert should_queue_scout(SearchResult(((101, 1.2),), -1.0, 1.2, 0.5, 8.0, 1), ()) is False
    assert should_queue_scout(SearchResult(((101, 1.2),), -math.inf, 1.2, 0.5, 8.0, 1), [hp]) is True
    assert should_queue_scout(SearchResult(((101, 1.2),), 0.0, 1.2, 0.5, 8.0, 1), [hp]) is True
    assert (
        should_queue_scout(
            SearchResult(((101, 1.2),), 2.5, 1.2, 0.7, 8.0, 2, coverage=0.3, next_candidates=((102, 0.9),)),
            [hp],
        )
        is True
    )
    assert (
        should_queue_scout(
            SearchResult(((101, 1.2),), 2.5, 1.2, 0.7, 8.0, 2, coverage=0.8, next_candidates=((102, 0.9),)),
            [hp],
        )
        is False
    )

    card_replies = iter(
        [
            {
                "slug": "cobalt-rover-note",
                "title": "Cobalt Rover Note",
                "kind": "note",
                "body": "Cobalt powers the rover.",
            },
            {
                "slug": "ignored-slug",
                "title": "Ignored",
                "kind": "note",
                "body": "ABSTAIN",
            },
        ]
    )

    def _format_reply(fields: dict[str, str], req_payload: dict[str, Any]) -> str:
        resp_fmt = req_payload.get("response_format")
        if isinstance(resp_fmt, dict) and resp_fmt.get("type") == "json_object":
            return json.dumps(fields)
        sys_prompt = "\n".join(str(m.get("content", "")) for m in req_payload.get("messages", []))
        if "<slug>" in sys_prompt:
            return (
                f"<slug>{fields['slug']}</slug>\n"
                f"<title>{fields['title']}</title>\n"
                f"<kind>{fields['kind']}</kind>\n"
                f"<body>{fields['body']}</body>\n"
            )
        return (
            f"[[ ## slug ## ]]\n{fields['slug']}\n"
            f"[[ ## title ## ]]\n{fields['title']}\n"
            f"[[ ## kind ## ]]\n{fields['kind']}\n"
            f"[[ ## body ## ]]\n{fields['body']}\n"
            "[[ ## completed ## ]]\n"
        )

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            req_payload = json.loads(self.rfile.read(length).decode("utf-8"))
            content = _format_reply(next(card_replies), req_payload)
            body = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": "mock-scout",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = YawConfig(
            workspace_root=tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            qa_model="openai/mock-scout",
        )
        lm: dspy.LM = create_lm(cfg)
        card_out = synthesize_scout_card("What powers the rover?", [hp], lm)
        assert card_out == ("cobalt-rover-note", "Cobalt Rover Note", "Cobalt powers the rover.", "note")
        abstained = synthesize_scout_card("Unknown topic?", [hp], lm)
        assert abstained is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_scout_rejects_all_stale_and_filters_mixed_passages(tmp_path: Path) -> None:
    stale_ptr = ChunkPointer(101, "stale.md", "h1", 1, 2, title="Stale")
    empty_ptr = ChunkPointer(102, "empty.md", "h2", 1, 2, title="Empty")
    fresh_ptr = ChunkPointer(103, "fresh.md", "h3", 1, 2, title="Fresh")
    stale_hp = HydratedPassage(pointer=stale_ptr, text="STALE_MARKER_TEXT", score=1.2, stale_on_disk=True)
    empty_hp = HydratedPassage(pointer=empty_ptr, text="   \n", score=1.1, stale_on_disk=False)
    fresh_hp = HydratedPassage(pointer=fresh_ptr, text="FRESH_ROVER_EVIDENCE", score=1.0, stale_on_disk=False)

    assert [is_fresh_passage(hp) for hp in (stale_hp, empty_hp, fresh_hp)] == [False, False, True]
    weak_sr = SearchResult(((101, 1.2),), -1.0, 1.2, 0.4, 8.0, 1, coverage=0.2, next_candidates=((102, 0.8),))
    assert should_queue_scout(weak_sr, [stale_hp, empty_hp]) is False
    assert should_queue_scout(weak_sr, [stale_hp, fresh_hp]) is True

    captured_bodies: list[str] = []

    class _MixedScoutHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            captured_bodies.append(raw)
            req_payload = json.loads(raw)
            fields = {
                "slug": "fresh-only-card",
                "title": "Fresh Only Card",
                "kind": "note",
                "body": "Fresh rover synthesis.",
            }
            if isinstance(req_payload.get("response_format"), dict):
                content = json.dumps(fields)
            elif "<slug>" in "\n".join(str(m.get("content", "")) for m in req_payload.get("messages", [])):
                content = (
                    f"<slug>{fields['slug']}</slug>\n"
                    f"<title>{fields['title']}</title>\n"
                    f"<kind>{fields['kind']}</kind>\n"
                    f"<body>{fields['body']}</body>\n"
                )
            else:
                content = (
                    f"[[ ## slug ## ]]\n{fields['slug']}\n"
                    f"[[ ## title ## ]]\n{fields['title']}\n"
                    f"[[ ## kind ## ]]\n{fields['kind']}\n"
                    f"[[ ## body ## ]]\n{fields['body']}\n"
                    "[[ ## completed ## ]]"
                )
            body = json.dumps(
                {
                    "id": "chatcmpl-mixed",
                    "object": "chat.completion",
                    "model": "mock-scout",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MixedScoutHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = YawConfig(
            workspace_root=tmp_path, base_url=f"http://127.0.0.1:{server.server_port}/v1", qa_model="openai/m"
        )
        lm: dspy.LM = create_lm(cfg)
        assert synthesize_scout_card("Rover?", [stale_hp, empty_hp], lm) is None
        assert captured_bodies == []
        res = synthesize_scout_card("Rover?", [stale_hp, empty_hp, fresh_hp], lm)
        assert res == ("fresh-only-card", "Fresh Only Card", "Fresh rover synthesis.", "note")
        assert len(captured_bodies) == 1
        assert "FRESH_ROVER_EVIDENCE" in captured_bodies[0]
        assert "STALE_MARKER_TEXT" not in captured_bodies[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_tmp_exclusions_across_matcher_cards_pipeline_and_malformed_yaml(tmp_path: Path) -> None:
    matcher = build_wiki_path_matcher()
    for excluded_rel in ("draft.tmp.md", "sub/draft.tmp.md", "dir.tmp/card.md", "a/dir.tmp/card.md", ".hidden/card.md"):
        assert is_ignored_wiki_rel(Path(excluded_rel).parts) is True
        assert matcher.is_file_included(Path(excluded_rel)) is False
    for kept_rel in ("card.md", "topic/rover-notes.md"):
        assert is_ignored_wiki_rel(Path(kept_rel).parts) is False
        assert matcher.is_file_included(Path(kept_rel)) is True

    corpus_dir = tmp_path / "corpus"
    wiki_dir = tmp_path / "wiki"
    corpus_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)
    (corpus_dir / "base.md").write_text("# Base\nCobalt lunar rover.\n", encoding="utf-8")
    (wiki_dir / "draft.tmp.md").write_text("# Draft\nShould be ignored.\n", encoding="utf-8")
    (wiki_dir / "sub").mkdir()
    (wiki_dir / "sub" / "draft.tmp.md").write_text("# Sub Draft\nShould be ignored.\n", encoding="utf-8")
    (wiki_dir / "a" / "dir.tmp").mkdir(parents=True)
    (wiki_dir / "a" / "dir.tmp" / "card.md").write_text("# Tmp Dir\nShould be ignored.\n", encoding="utf-8")
    write_wiki_card(wiki_dir, "valid", "Valid Card", "Cobalt lunar rover summary.")

    cfg = _make_cfg(tmp_path, wiki_root=wiki_dir)
    session = start_pipeline(cfg, embed_fn=_topic_embed, live=False)
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)

    assert cfg.db_path is not None
    with open_readonly_db(cfg.db_path) as conn:
        wiki_rels = [
            str(r[0])
            for r in conn.execute("SELECT rel_path FROM chunks WHERE rel_path LIKE 'wiki:%' ORDER BY rel_path")
        ]
    assert wiki_rels == ["wiki:valid.md"]
    assert [item["slug"] for item in list_wiki_cards(wiki_dir)] == ["valid"]
    assert lint_wiki(cfg).cards_checked == 1

    (wiki_dir / "broken.md").write_text("---\nkind: [unterminated\n---\n\n# Broken\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError) as exc_cons:
        list_wiki_cards(wiki_dir)
    assert exc_cons.value.code == "CONSISTENCY_ERROR"

    lint_report = lint_wiki(cfg)
    assert lint_report.ok is False
    assert any(issue.code == "INVALID_CARD" and issue.card_rel_path == "broken.md" for issue in lint_report.issues)


def test_multiline_setext_h1_parse_and_write_no_orphan_underline(tmp_path: Path) -> None:
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir(parents=True)

    two_line_card = wiki_dir / "two-line.md"
    two_line_card.write_text(
        "---\nkind: note\nstale: false\nsources: []\n---\n\nFirst Line\nSecond Line\n===\n\nBody text.\n",
        encoding="utf-8",
    )
    parsed = parse_wiki_card(two_line_card, wiki_dir)
    assert parsed.title == "First Line Second Line"
    assert parsed.body == "Body text."
    assert "===" not in parsed.body
    assert "Second Line" not in parsed.body

    three_line_card = wiki_dir / "three-line.md"
    three_line_card.write_text(
        "---\nkind: note\nstale: false\nsources: []\n---\n\nAlpha\nBeta\nGamma\n===\n\nTrailing body.\n",
        encoding="utf-8",
    )
    parsed_three = parse_wiki_card(three_line_card, wiki_dir)
    assert parsed_three.title == "Alpha Beta Gamma"
    assert parsed_three.body == "Trailing body."

    written_path, wrote = write_wiki_card(
        wiki_dir,
        "setext-strip",
        "First Line Second Line",
        "First Line\nSecond Line\n===\n\nBody text.",
    )
    assert wrote is True
    reparsed = parse_wiki_card(written_path, wiki_dir)
    assert reparsed.title == "First Line Second Line"
    assert reparsed.body == "Body text."
    assert "===" not in reparsed.body
    assert "Second Line" not in reparsed.body

    _, wrote_again = write_wiki_card(
        wiki_dir,
        "setext-strip",
        "First Line Second Line",
        "First Line\nSecond Line\n===\n\nBody text.",
    )
    assert wrote_again is False
