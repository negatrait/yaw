# Native CocoIndex graph & `M_FINAL_1B` retrieval engine (`yaw/retrieval/`)

Status: active (frozen `M_FINAL_1B` in modular `L1` `yaw/retrieval/**/*.py`, `C901 <= 15`, `U4` `SearchResult` sparse slip & bounded caches).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [retrieval.md](../.agents/rules/retrieval.md): `yaw/retrieval/**/*.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/retrieval/**/*.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/core/` (`ChunkPointer`, `HydratedPassage`, `resolve_corpus_path`, `YawRuntimeError`, `format_query_for_embedding`); [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `yaw/pipeline/**/*.py` & `yaw/datasets.py` (writes `chunks`, `chunk_fts`, `chunk_vocab` and invokes `compute_db_fingerprints`, `compute_index_artifacts`, `normalize_fts_unicode61` to materialize `rabitq_index` and `citation_graph`); [dspy-improvements.md](dspy-improvements.md): owns `yaw/qa.py` & `yaw/optimize.py` consuming `RetrievalIndex`, `search_index`, `lexical_search`, `expand_citation_neighbors`, `hydrate_ranked_passages`, `hydrate_passage`, `build_recall_payload`, and `format_recall_text`; [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md): owns `yaw/runtime.py`, `yaw/mcp_server.py` & `yaw/cli.py` (`get_retrieval_index`, `reload_retrieval_index` reactive swaps, `arecall_payload`/`arecall`/`recall`, `aask`/`qa`, re-exported `build_recall_payload`/`format_recall_text`/`dump_recall_json`); [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): owns `scripts/{benchmark,download_datasets,verify_paper_parity_gate,evaluation}.py` evaluating `build_retrieval_index` and `search_index` parity; [epic-3-performance.md](epic-3-performance.md): owns `scripts/watchdog.py` (`open_readonly_db`) & deferred `EVAL-6` (`SearchResult.completeness`, Graph Scout); [coverage-plan.md](coverage-plan.md): owns `tests/**/*.py` (`tests/test_retrieval.py`) |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md): anchors HippoRAG 2 retrieval parity; [model-evaluation.md](model-evaluation.md): anchors local serving constraints |
| Supersedes / superseded by | Supersedes legacy OpenIE/W3C/SHACL/PPR/HMM graph storage ([archive/w3c-improvements.md](archive/w3c-improvements.md), [archive/epistemic-confidence-spec.md](archive/epistemic-confidence-spec.md)), `sqlite-vec` `vec0` virtual tables, `CandidateStageCache`, and `3cd3952` `VectorSearch` with `M_FINAL_1B` ([mdl-dag-sprint.md](reviews/mdl-dag-sprint.md), `5c4b115`) |
| Records | [mdl-dag-sprint.md](reviews/mdl-dag-sprint.md), [mdl-dag-experiment-report.md](reviews/mdl-dag-experiment-report.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md), [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md), [incremental-publication-work.md](reviews/incremental-publication-work.md), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [evaluation-graph-storage.md](evaluation-graph-storage.md), [evaluation.md](evaluation.md) |

## 1. Scope & Pure Dependency Boundary

In scope: Read-only hybrid retrieval, 1-bit Extended SoA RaBitQ quantization, single-pass corpus hydration & `# Heading` title extraction, compressed `CitationGraph` & `DiffusionOperators`, tangent-steered cyclic citation + 1-bit PMI diffusion, conditional Noisy-OR evidence aggregation, Rank-1 Cholesky 1-bit DPP reranking, decoupled `pool_k` with non-hydrating 5-tuple MaxP `dedup_ranked_positions`, `SearchResult` (`coverage`, `next_candidates`, sparse cross-doc `links`), canonical sparse slip rendering/serialization (`yaw/retrieval/slip.py`: `build_recall_payload`, `format_recall_text`, `dump_recall_json`, `sanitize_slip_label`), `4,096`-entry bounded `vocab_doc_cache` & `resolved_path_cache`, multilingual lexical search (`unicode61`, umlaut folding, Koehn–Knight compound splitting, CJK bigrams/PMI), `?mode=ro` WAL SHA-256 disk hydration, and strict `INDEX_NOT_READY` vs `CONSISTENCY_ERROR` failure contracts in `yaw/retrieval/`.

Non-goals: Persistent writes to `yaw.db` or `.yaw/coco_lmdb` (owned by `yaw/pipeline/` in [native-cocoindex-indexing.md](native-cocoindex-indexing.md)); runtime locking/polling orchestration (`yaw/runtime.py` in [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md)); `U5` dual-root Librarian `wiki_root` and `Sprint 3.4+` performance items ([epic-3-performance.md](epic-3-performance.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md)).

- **Pure Kernel Containment (`Rule 12`):** `yaw/retrieval/` imports **only** Python `stdlib`, `numpy`, `scipy`, `sqlite3`, and `yaw.core.*` — zero `dspy`, `cocoindex`, HTTP clients, or retired stack dependencies (`sqlite-vec`, `spacy`, `optuna`, OpenIE/SHACL).

<a id="11-pipeline-order-extract-local-evidence-before-global-adjudication"></a>
<a id="evidence-backed-provenance-restoration"></a>
<a id="constraint-propagation-during-adjudication"></a>
<a id="2-problem-with-evidence"></a>
<a id="3-goals"></a>

## 2. Read-Only SQLite Artifacts & Index Materialization

- **`?mode=ro` WAL Snapshots (`open_readonly_db`, `_snapshot_tx`):** Opens SQLite in URI read-only mode (`?mode=ro`, `check_same_thread=False`, `isolation_level=None`) with `PRAGMA busy_timeout=30000`, `cache_size=-65536`, `temp_store=MEMORY`, `query_only=ON`, and zero extensions. `_snapshot_tx` acquires a per-connection `RLock` and wraps reads in a single `BEGIN` ... `ROLLBACK` snapshot transaction, validating `chunk_fts` and `chunk_vocab` presence (`CONSISTENCY_ERROR` when `require_fts=True` and absent).
- **Deterministic SHA-256 Fingerprints (`compute_db_fingerprints`):** Queries `chunks ORDER BY chunk_id ASC` (`chunk_id, embedding_f16, rel_path, source_hash, start_line, end_line, char_start, char_end, ast_island_start, ast_island_end, title`), verifying even/uniform IEEE `<f2>` `float16` LE byte lengths (`CONSISTENCY_ERROR` on null or mismatched rows) and hashing ordered `(chunk_id, embedding_f16)` into `vec_sha256` and chunk coordinates/titles into `source_sha256`.
- **Single-Pass Disk Hydration & `# Heading` Extraction (`compute_index_artifacts`, `hydrate_corpus_chunks`, `build_citation_graph`):**
  - Materializes `IndexArtifacts` for `yaw/pipeline/`: `reconcile_index_tables` iterates sequentially over distinct `rel_posix_path` (`0` `concurrent.futures` pools in `yaw/retrieval/`), normalizing CRLF/CR to `\n`, decoding UTF-8 strictly, verifying line bounds and SHA-256 `source_hash` (`strict=True` raises `CONSISTENCY_ERROR`), and extracting the first non-empty Markdown `# Heading` (`extract_markdown_h1`) whenever `ptr.title` is empty.
  - Persists two single-row (`id = 1`) plain SQLite tables:
    1. `rabitq_index`: `n_chunks`, `dim`, `embedding_model`, `task_mode`, `vec_sha256`, `source_sha256`, `n_words`, `codes_blob`, `ip_norm_blob`, `mean_blob`, `d_eff_shannon`.
    2. `citation_graph`: `n_chunks`, `vec_sha256`, `source_sha256`, `doc_titles_blob` (zlib-compressed JSON map), `graph_blob` (`YCG1` binary header + little-endian `<i4` CSR arrays + UTF-8 `[pmi_terms, prop_nouns, doc_titles]` metadata under zlib compression).

## 3. 1-Bit Extended SoA RaBitQ & Tangent-Steered Cyclic Citation Diffusion

1. **1-Bit Extended SoA RaBitQ (`_Rotator`, `_pack_rabitq_soa`, `_rabitq_dot_lut`):**
   - Decodes `<f2>` embeddings to `float32`, subtracts corpus `mean_vec` (`n >= 2`), L2-normalizes (`_transform_vectors`), and computes Shannon spectral entropy effective dimension $d_{\text{eff}} = \max(2.0, \exp(-\sum p_i \ln p_i))$ over singular values sampled up to `_SPECTRUM_SAMPLE_CAP = 2048`.
   - Applies a deterministic 3-round orthogonal rotator (`_RABITQ_SEED + dim`): random `{-1, +1}` sign flips, coordinate permutations, `64`-dim Hadamard block transforms (`_make_hadamard(64)`), plus a QR orthogonal tail block when `dim % 64 != 0`.
   - Packs signs into Structure-of-Arrays `uint8[n_words * 8, N]` (`bitorder="little"`, `n_words = (dim + 63) // 64`) and stores `ip_norm = max(<rotated, signed_vertex>, 0.5 / sqrt(dim))`. Query dot products evaluate via vectorized 256-entry byte LUTs (`_rabitq_dot_lut`) divided by `ip_norm`, supporting QR orthogonal subspace deflation when `exclude_chunk_ids` are supplied (`_prepare_query_vector`).
2. **Title-Reference & 1-Bit PMI Citation Graph (`_build_citation_graph`, `_build_diffusion_operators`):**
   - Extracts document title keys (`_extract_title_keys`, supporting parenthetical base acronyms when `prop_nouns` or multi-word) and matches title mentions in chunk prose (`df <= ceil(sqrt(n))`) to build forward (`fi`, `fx`) and reverse (`ri`, `rx`) chunk-to-document CSR adjacency alongside document ownership (`dc_indptr`, `dc_indices`).
   - Collects adjacent word and CJK bigram pairs satisfying Positive-PMI (`len(cl) * n >= 2 * w_df[a] * w_df[b]`, `2 <= len(cl) <= db`, cross-document `>= 2` paths) into symmetric chunk-term CSR matrices (`pmi_indptr`, `pmi_indices`, `cpmi_indptr`, `cpmi_indices`).
   - Packs per-document 1-bit centroids (`_pack_doc_1bit_codes`) and evaluates edge Hamming cosines $c_{ud} = \cos(\pi H(u, d) / \text{dim})$ via XOR + `np.bitwise_count`.
3. **Tangent-Steered Diffusion, Beta-Tail LUT & Conditional Noisy-OR (`_apply_tangent_steer_ops`, `_diffuse_citation`, `search_index`):**
   - Transforms dense query cosines via exact Student-t / regularized incomplete Beta null nats (`_parametric_null_nats_vec` over $d_{\text{eff}}$) and combines them with RSJ IDF lexical/title nats at temperature $t_{\text{path}} = \sqrt{\max(1, d_{\text{eff}}) / \max(1, \ln(n_{\text{eff}} + 1))}$ into stabilized initial prior $a_0$.
   - Computes tangent-steered partial correlations $\rho_{\text{fwd}} = (c_{qd} - c_{qu} c_{ud}) / \sqrt{(1 - c_{qu}^2)(1 - c_{ud}^2)}$ and symmetric $\rho_{\text{rev}}$, maps through a precomputed `16,384`-bin $z = \sqrt{-\ln(1 - \rho^2)}$ Beta-tail LUT (`_build_beta_nats_lut`, `_eval_signed_beta_nats_lut`), and normalizes per-source rows via `_grouped_softmax`.
   - Each diffusion step blends active forward citation (`d_own @ m_fwd_t`), reverse citation (`r_rev @ d_id_t`), and self-loop-cancelled 1-bit PMI (`max(0, p2c @ c2p_t - self_pmi) * pmi_scale`) channels equally via `inv_m`, running geometric-restart diffusion ($\rho_{\text{stop}} = 0.5$) until $w < 1 / \max(2, n_{\text{eff}})$ or $\|a_{\text{next}} - a_{\text{cur}}\|_\infty \le 10^{-9}$.
   - When `len(q_trefs) >= 2`, applies **conditional Noisy-OR** across per-title-reference diffused posteriors $p_g$ with gate $w_g = \sigma(\max \text{sem\_n} - \ln(n_{\text{eff}} + 1))$: `post = 1 - prod(1 - wg * pg)`.
4. **Rank-1 Cholesky 1-Bit DPP, `pool_k` MaxP `dedup_ranked_positions`, `coverage`, `next_candidates`, & Sparse `links` (`rerank_dpp_1bit`, `search_index`, `dedup_ranked_positions`, `hydrate_ranked_passages`):**
   - Orders distinct-document top candidates (`_order_candidates_distinct_first`), reranks `m_pool = min(n_distinct_found, ceil(2 * log(n_eff + 1)))` via Rank-1 Cholesky greedy L-ensemble 1-bit DPP over unpacked signed binary vectors (`w = n_eff * post`, tie-broken by `-chunk_id`) with `dpp_k = pool_k if pool_k is not None else top_k`, and enforces strictly decreasing rank scores. Retrieval callers (`yaw/runtime.py`, `yaw/qa.py`, `scripts/benchmark.py`) pass `top_k=user_top_k, pool_k=max(user_top_k * 4, 50)`, preserving exact `M_FINAL_1B` scores (`0` diffs) while keeping `top_k` decoupled.
   - `dedup_ranked_positions(idx, ranked_chunks, *, limit=None) -> tuple[tuple[int, int, float], ...]` applies non-hydrating 5-tuple MaxP span deduplication over `(rel_posix_path, start_line, end_line, char_start, char_end)` in RAM (`0` disk opens). Inside `search_index`, `deduped = dedup_ranked_positions(idx, ranked_out, limit=top_k + 3)` partitions `top_entries = deduped[:top_k]` (`top_k_pos`) and `next_entries = deduped[top_k : top_k + 3]` (`next_pos`):
     - `SearchResult.coverage: float = min(1.0, max(0.0, float(post[top_k_pos].sum() / max(float(post[eligible].sum()), 1e-12))))` (`0.0` if `top_k_pos` is empty).
     - `SearchResult.next_candidates: tuple[tuple[int, float], ...] = tuple((cid, sc) for _, cid, sc in next_entries)` (`<= 3` deduplicated tail candidates; `0` disk opens).
     - `SearchResult.links: tuple[tuple[Literal["cite", "pmi"], int, int, str], ...]` computes cross-doc evidence links strictly among `cand_pos = [*top_k_pos, *next_pos]` in RAM (`0` disk opens, `< 0.05 ms`): `_collect_cite_links` builds rank-ordered `doc_best_pos` (`setdefault(int(idx.path_idx[p]), p)`) to emit deduplicated cross-doc citation edges `("cite", src_cid, dst_cid, "")` from `cg.fi`/`cg.fx`, and `_find_rarest_pmi_link` selects at most `1` global rarest cross-doc shared PMI edge `("pmi", src_cid, dst_cid, clean_term)` from `cg.cpmi_*`/`cg.pmi_*` minimizing `(int(cg.pmi_indptr[eid + 1] - cg.pmi_indptr[eid]), clean_term, rank_u, rank_v)` (`sanitize_slip_label`).
   - `hydrate_ranked_passages` reuses `dedup_ranked_positions(idx, ranked_chunks, limit=top_k)` so only the returned `top_k` slice hydrates from disk (`0` tail opens; `max_chars=None` uncropped by default; `stale_on_disk=True` with `text=""` on missing, truncated, or SHA-256-mismatched files).

<a id="4-semantics-and-invariants"></a>
<a id="52-batch-independent-identity-and-candidate-discovery"></a>
<a id="53-resource-vocabulary-and-statement-boundary"></a>

## 4. Multilingual Lexical Search, Bounded Caches, Slip & Fail-Fast Contracts

- **Multilingual `unicode61` Search (`lexical_search`, `extract_multilingual_anchors`, `normalize_fts_unicode61`):**
  - Casefolds and expands CJK runs into overlapping bigrams at index and query time (`normalize_fts_unicode61`).
  - Queries `chunk_fts` and `chunk_vocab` with umlaut primary/reverse folding (`ä/ö/ü/ß` <-> `ae/oe/ue/ss`), `chunk_vocab`-backed left-prefix stemming (`len >= 6`, `df < 2`), Koehn–Knight compound splitting (`len >= 8` checking left/right `chunk_vocab` coverage), adjacent Positive-PMI word collocations, and CJK 1–4 char spans, scored via RSJ IDF.
- **Bounded `RetrievalIndex` Caches (`CACHE_MAX_ENTRIES = 4_096`):** `vocab_doc_cache` (`fetch_vocab_docs`) and `resolved_path_cache` (`_resolve_cached_path`) enforce FIFO eviction capped at `4,096` entries under `CACHE_LOCK`.
- **Canonical Sparse Slip (`yaw/retrieval/slip.py`):**
  - `sanitize_slip_label(raw)`: strips `"`, collapses C0/C1 control characters and whitespace (`[\x00-\x1f\x7f-\x9f\s]+ -> " "`), and truncates labels/PMI terms to `64` chars.
  - `build_recall_payload(...)`: returns sparse `{"results": [...]}` omitting empty/false/non-finite optional keys (`log_bf`, `coverage` omitted when `results == []`, `next: [{"chunk_id", "score", "label"}]` resolved strictly in RAM via `ptr.title or idx.doc_titles.get(ptr.rel_posix_path, "") or ptr.rel_posix_path`, `links: [{"kind": "cite", "src", "dst"} | {"kind": "pmi", "src", "dst", "term"}]`, `card_queued: True`), with item `"stale": True` (omitted if `False`), optional `source.title` (omitted if empty), and `"sha256:"` `content_hash`.
  - `format_recall_text(payload)` & `dump_recall_json(payload)`: `format_recall_text` renders `NO_MATCHES_TEXT` when empty, or optional `certainty: log_bf=... coverage=...`, `[chunk_id=<int>] (path:start-end) score=...:` blocks (`[STALE ON DISK] <rel_posix_path>` on drift), optional `next: <id> "<label>" (<score>)`, and `links: <src>-><dst>, <src>~pmi("<term>")~<dst>` with bare integer IDs; `dump_recall_json` serializes strictly per RFC 8259 via `json.dumps(..., allow_nan=False)`.
- **Explicit `INDEX_NOT_READY` vs `CONSISTENCY_ERROR` Semantics (`build_retrieval_index`, `search_index`, `expand_citation_neighbors`):**
  - `INDEX_NOT_READY`: missing DB file (`open_readonly_db`), missing `rabitq_index` or `citation_graph` table/row `id = 1`, `vec_sha256` / `source_sha256` / `n_chunks` / `dim` drift against `chunks`, or `embedding_model` / `task_mode` mismatch (allowing live `YawRuntime.get_retrieval_index` to keep serving its prior `RetrievalIndex` snapshot while indexing catches up, or failing immediately in one-shot `recall`/`qa`).
  - `CONSISTENCY_ERROR`: missing `chunk_fts` or `chunk_vocab`, `conn=None`, corrupt blob byte lengths, `n_words != (dim + 63) // 64`, non-finite or non-positive `ip_norm` / `d_eff_shannon`, invalid `YCG1` zlib/CSR/JSON payloads, or missing/unreadable UTF-8 files during `strict=True` index/neighbor hydration. Never silently degrades or falls back.

<a id="5-design-current-state"></a>

## 5. Design (current state)

```python
CACHE_MAX_ENTRIES = 4_096

@dataclass(slots=True)
class RetrievalIndex:
    """Read-only M_FINAL_1B retrieval index with 1-bit SoA RaBitQ codes, citation graph, and caches."""

    chunk_ids: tuple[int, ...]
    id_to_pos: dict[int, int]
    pointers: tuple[ChunkPointer, ...]
    path_idx: NDArray[np.uint32]
    distinct_paths: tuple[str, ...]
    dim: int
    mean_vec: NDArray[np.float32]
    d_eff: float
    codes: NDArray[np.uint8]
    ip_norm: NDArray[np.float32]
    rotator: Rotator | None
    doc_titles: dict[str, str]
    title_by_first_word: dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]]
    citation_graph: CitationGraph | None
    diffusion_ops: DiffusionOperators | None
    vocab_doc_cache: dict[str, int] = field(default_factory=dict)
    resolved_path_cache: dict[tuple[Path, str], Path] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ranked chunk hits, certainty metrics, next candidates, and evidence links."""

    ranked: tuple[tuple[int, float], ...]
    log_bf: float
    max_chunk_score: float
    top1_cosine: float
    d_eff: float
    n_candidates: int
    coverage: float = 0.0
    next_candidates: tuple[tuple[int, float], ...] = ()
    links: tuple[tuple[Literal["cite", "pmi"], int, int, str], ...] = ()
```

Primary public exports in `yaw/retrieval/__init__.py` (`_PUBLIC_API`): `empty_retrieval_index()`, `open_readonly_db(db_path)`, `normalize_fts_unicode61(text)`, `compute_db_fingerprints(conn)`, `compute_index_artifacts(rows, corpus_root=None)`, `build_retrieval_index(conn, *, expected_embedding_model=None, expected_task_mode=None)`, `lexical_search(...)`, `search_index(..., top_k=10, pool_k=None) -> SearchResult`, `expand_citation_neighbors(...)`, `dedup_ranked_positions(...)`, `hydrate_passage(...) -> HydratedPassage`, `hydrate_ranked_passages(...) -> tuple[HydratedPassage, ...]`, `format_evidence_route(passages)`, `NO_MATCHES_TEXT`, `sanitize_slip_label(raw)`, `build_recall_payload(...)`, `format_recall_text(payload)`, `dump_recall_json(payload)`.

## 6. Capability Gates (`M_FINAL_1B` Confirmatory Proof)

- **`P0`–`P4` Cleared on Untouched `Tier E` (`N = 400`, `n = 200` Live `gemma-4-e4b` QA at `temperature = 1.0`, [evaluation.md](evaluation.md)):**
  - `0` tunable retrieval hyperparameters (`P0`); beats `B5S_RAND_SEEDS` / `B5A_RAND_WALK` (`p_holm = 0.0008`, `P1`); `MH AH@5 +0.3033` and `macro R@5 +0.1798` (`p_holm = 0.0004`, `P2`); paired `dspy.ReAct` QA `EM = 0.4500 vs 0.4100` and `F1 = 0.6219 vs 0.5569` (`p_perm = 0.0022`, `P3`); `3.78x` RAM reduction (`24.73 vs 93.55 MiB`), `p50 / p95 = 22.1 / 38.8 ms`, `Reject AUROC = 0.7068 > top1 0.6345`, `0` concurrency or masked errors (`P4`).
- **IEEE `<f2>` `float16` Storage Parity (`52,959` chunks):** `0.01057%` 1-bit RaBitQ flips and `0.0000` `Tier E` `4-DS H4` / `MH AH@5` metric delta against `float32`.

<a id="7-removal-inventory"></a>
<a id="91-remaining-cleanup-after-the-614-line-retirement"></a>

## 7. Removal Summary

Retired legacy OpenIE/W3C/SHACL/PPR/HMM storage (`yaw/daemon/graph/**`), `sqlite-vec` / `vec0`, `CandidateStageCache`, `yaw_stage1`, `tokens_json`, `entities_json`, `SearchParams`, `resolve_retrieval_params`, and `VectorSearch` completely under [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md) and `EVAL-5` ([mdl-dag-sprint.md](reviews/mdl-dag-sprint.md)).

<a id="8-checkpoints"></a>
<a id="92-performance-review-measured-implementation-units"></a>

## 8. Checkpoints

- `EVAL-5` (`5c4b115`): `M_FINAL_1B` shipped on `dev` with plain SQLite 5-table schema (`chunks`, `chunk_fts`, `chunk_vocab`, `rabitq_index`, `citation_graph`).
- Sprint `3.2`–`3.3` (`U3`–`U4` shipped; `U5`–`U6` next): `U3a` modularized `yaw/retrieval/**/*.py` (`C901 <= 15`, `0.0` `M_FINAL_1B` & `64+64==128` parity); `U4` landed `pool_k` decoupling, `dedup_ranked_positions`, `SearchResult` (`coverage`, `next_candidates`, `< 0.05 ms` sparse cross-doc `links`), `yaw/retrieval/slip.py` (`build_recall_payload`, `format_recall_text`, `dump_recall_json`, `sanitize_slip_label`), and `4,096`-entry bounded `vocab_doc_cache` & `resolved_path_cache`; `U5` adds dual-root Librarian `wiki_root` ([epic-3-performance.md](epic-3-performance.md), [epic-3-librarian-wiki-sprint.md](reviews/epic-3-librarian-wiki-sprint.md)).

<a id="9-verification"></a>

## 9. Verification

Real-substrate tests live in `tests/test_retrieval.py` (unmocked SQLite WAL snapshots, `1`-bit SoA RaBitQ packing, Beta-tail LUT parity, cyclic citation diffusion, conditional Noisy-OR, 1-bit DPP, `coverage`/`pool_k` parity & duplicate-span `top_k`, non-hydrating `dedup_ranked_positions` with `0` tail opens, `SearchResult.links` multi-chunk `cite` + global rarest `pmi` `< 0.05 ms`, `allow_nan=False` strict RFC 8259 validation, `4,096`-cap `vocab_doc_cache` & `resolved_path_cache`, umlaut/compound/CJK FTS5, `N=1` & identical-vector edge cases, `INDEX_NOT_READY` vs `CONSISTENCY_ERROR`, and disk SHA-256 hydration).

```powershell
uv run --offline --locked ty check
uv run --offline --locked ruff check
uv run --offline --locked ruff format --check
uv run --offline --locked pytest -o addopts="" tests/test_retrieval.py
git diff --check
```

Measurements recorded in: [evaluation-graph-storage.md](evaluation-graph-storage.md) and [evaluation.md](evaluation.md).

## Decisions

- 2026-11-06: "Should stay literals, NER is only allowed to capture skos:Concept and owl:NamedIndividual, triple extraction can point the rdf:Statement to rdf:object = rdfs:Literal but never rdf:subject = rdfs:Literal! This is why it's two stage. NER creates the allowed rdf:subject vocabulary, with predicates able to point towards literals but never the other way around - that's the W3C ontology rules and should be in our allowed SHACL shapes." (User). Enforced across GBNF grammars, extraction/repair parsers, and closed-world SQLite SHACL shapes (§4).
- 2026-11-06: "Step 4 should be included in this unit, but obviously, you can implement one unit at a time according to whichever spec you write into @[docs/native-cocoindex-graph.md] or @[docs/native-cocoindex-indexing.md] , also sounds like you'll need to update our rules and tests to prevent these kinds of failures surfacing again." (User). Specs are promoted into `docs/native-cocoindex-graph.md` and `docs/native-cocoindex-indexing.md` and executed `1` `[target]` checkpoint unit at a time with regression tests.
- 2026-11-07: "Audit and review for root cause for batch indexing regressing back to exponential time behavior. Incremental indexing and graph publication was designed and tested to be near-constant time for 64 + 64 chunk batches vs 128 chunk batchless indexing." / "It’s a SHACL Violation CONSISTENCY_ERROR" / "you will need to /spec first." (User). Supersedes the single-batch unconditional `canonicalize_fact_definitions` overwrite from [shacl-healer-publication.md](reviews/shacl-healer-publication.md) (`c6fa080`) with validity-ordered (`3`-tier `[target]`), repair-preserving `canonicalize_fact_definitions`, `"__healed__"` label provenance, focus-isolated `SHACLHealInput` memoization (`heal_shacl_violations` `version=4` `[observed: commit c6fa080]`, `publish_graph` `version=8` `[observed: commit c6fa080]`), single-pass post-canonicalization fact embedding, and delta-scoped table updates (§4, §5).
- 2026-11-07: "SHACL healing is taking extremely long - what is the performance bottleneck here?" / "/spec please spec the remedy for your debug audit above." (User). Routes pre-hydrated focus-isolated `SHACLHealInput` payloads directly to `YawProgram.shacl_repair` (`mode="direct"`, `ProposeSHACLRepairs` with GBNF; `heal_shacl_violations` `version=5` `[observed: yaw/daemon/graph/inference/components.py:309, 50be17d]`, `publish_graph` `version=9` `[observed: yaw/daemon/indexing.py:403, 50be17d]`), aligns `STAGE_RLM_DELEGATION_RULES["shacl_repair"]`, `build_rlm_action_demos()["shacl_repair"]`, and `HealSHACLViolations` with single-step zero-argument `propose_repairs()` delegation and bound `self._focus_context` reuse, and wires `OpenIEEndpoint.progress_callback` from `YawGraphEngine.emit_indexing_progress` for intra-batch semantic adjudication and SHACL healing progress telemetry (§4, §5).
- 2026-11-07: "/spec remedy gaps 1-5 in @[docs/native-cocoindex-indexing.md] and @[docs/native-cocoindex-graph.md] - remember the separation of concerns, one is for indexing and the other is for graphing. Use subagents to protect your own context from bloat." (User). Remedies Gaps 1–5 in the graph specification: (1) resolves the per-owner declaration vs. single-component atomicity contradiction by keeping `publish_graph` (`DatabaseSink.__call__`) as the sole atomic producer for all `15` `[target]` graph tables (including `chunk_vec` populated from `ChunkProcessOutput.vector_f32` with `0` `[target]` planner `embed_text` calls) while `plan_graph_publication` retires `clone_graph_to_memory` and full-table `SELECT *` survivor scans (`CG-GRAPH-2`); (2) keys `managed_sqlite.py:mount_table` by `TableKey(db_identity, name)` so shared `<state_dir>/coco_lmdb` never collides across candidate DBs; (3) retires `load_active_chunk_inferences` in favor of direct `Sequence[ChunkProcessOutput]` input from `publish_graph`; (4) specifies `sqlite-vec == 0.1.9` `[measured: uv.lock:2221-2222, 50be17d]` `vec0` virtual tables (`USING vec0(<pk_col> TEXT PRIMARY KEY, embedding float[N] distance_metric=cosine)`), `DELETE`+`INSERT` virtual-table upserts, and `WHERE embedding MATCH ? AND k = ?` KNN queries in `vectors.py` and `graph_ops.py` (`CG-GRAPH-1`); and (5) supersedes the `C3` exemption to retire `CandidateStageCache`, `candidate_stage_cache`, and `_text_vec_cache` completely (`C5`).
- 2026-11-07: "Plan doesn’t take subagent orchestration into account, should be explicitly defined." (User). Defines the mandatory subagent orchestration pipeline (`research` -> `ontologist` -> `simplifier` -> `critic (Model="flash")`) across §8 Checkpoints so the parent agent orchestrates slices without context bloat.
- 2026-11-08: W3C ontology simplification & deterministic statement canonicalization ([w3c-improvements.md §§12–13](archive/w3c-improvements.md#12-strict-judgelabels-identity-equivalence--canonical-gemma-4-3-stage-cascade-without-python-token-overrides-or-auto-merge-gates), [epistemic-confidence-spec.md §§1.3, 2.4](archive/epistemic-confidence-spec.md#13-empirical-bayes-1-hop-shacl-s27-resolution-zero-magic-thresholds-invariant-w5), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [incremental-publication-work.md](reviews/incremental-publication-work.md)): (a) retires cross-label auto-merge gates (`auto_merge_cosine_gate`, `auto_merge_difflib_gate`) and Python token overrides (`_is_acronym`, `is_modifier_head_specialization`, `_has_disjoint_non_acronym_tokens`) in `planner.py`, keeping only same-norm sense consolidation (`text_processing(name_a) == text_processing(name_b) and d_sim >= synonymy_threshold and not _is_active_fact_pole(ea, eb)`, Invariant `W2`) and strict `JudgeLabels` equivalence (`merge(A, B) <=> equivalent(A, B) = true`, Invariant `W1`); (b) supersedes the `2026-11-07` Tier 3 invalid-assertion `spo_def` fallback in `canonicalize_fact_definitions` with Tier 3 deterministic synthesis (`f"{subj_name} {pred} {obj_surface}."`, Invariant `W4`), reserving `heal_shacl_violations` for entity-level semantic violations; (c) enforces universal statement irreflexivity (`is_irreflexive_statement`, Invariant `W3`) across entity and literal objects; (d) enforces single-surface `skos:prefLabel` and `clean_surface_label`; and (e) enforces non-transitive `1`-hop `skos:broader` S27 disjointness + post-repair `auto_heal_deterministic_shacl` in `plan_graph_publication` (Invariant `W5`, bumping `judge_semantics` and `publish_graph` to `version=2` while keeping `extract_chunk` at `version=2`).
- 2026-11-09: "Review our cocoindex implementation against cocoindex skill and docs, are we inventing our own orchestration, caching, fingerprinting etc when cocoindex already provides it? Make a comprehensive audit." / "Please /spec these changes." (User). Remediates Anti-Pattern 6 (`ManagedTableTarget.delete_row(key)` in `yaw/daemon/graph/storage/managed_sqlite.py` replacing `target._provider.target_state(key, DELETE_ROW)` in `yaw/daemon/graph/storage/rows.py:333` at `28a1113`, plus the AST cross-object private attribute gate in `tests/core/test_architecture.py`) and updates `plan_graph_publication` (`yaw/daemon/graph/planner.py:854` at `28a1113`) to compute `input_fingerprints = {out.chunk_id: fingerprint_object((settings, out.pointer, out.text)).hex() for out in ordered_outputs}` in support of Anti-Pattern 1 ([native-cocoindex-indexing.md §5](native-cocoindex-indexing.md#5-design-current-state)).
- 2026-11-09: "Why exactly do we have the distinction between \"yaw_indexing\" and \"yaw_frozen_indexing\" - what's the point with having both instead of only \"yaw_indexing\"?" / "But why on earth is it indexing in memory instead of from the disk? Sounds super complex just to point the indexing pipeline at two different folders, production-ingestion/** and optimization-ingestion/**, and what happens when we want to optimize from the actual production corpus with GEPA?" / "Basically, the way I'd like to see this work, is that when uv run yaw dspy optimize is called, an optimize cocoindex app is launched, pointed into whatever was set up as the corpus, using the exact same pipeline as the production ingestion pipeline. The only difference, imo, should be the GEPA optimization and the corpus directory, to avoid polluting the production database... that's what the sandbox is supposed to mean." / "The point was to simply remove the hand rolled parallel ingestion path, and point the pipeline to a separate Target.optimize = F(Source.optimize) instead of Target.production = F(Source.production). The F stays the same." / "I don't understand this either. Why is there a separate benchmarking ingestion path?" (User). Unifies production ($\text{Target}_{\text{production}} = F(\text{Source}_{\text{production}})$), GEPA optimization ($\text{Target}_{\text{optimize}} = F(\text{Source}_{\text{optimize}})$), and benchmarking ($\text{Target}_{\text{benchmark}} = F(\text{Source}_{\text{benchmark}})$) onto the single `indexing_app_main` (`"yaw_indexing"`) pipeline in [native-cocoindex-indexing.md](native-cocoindex-indexing.md), retiring the parallel in-memory frozen pipeline (`frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `partition_frozen_chunks`, `FrozenChunkStream`, `"yaw_frozen_indexing"`).

### P2 relation decision (2026-11-13)

The former hierarchy then related model cascade is replaced by one typed relation judgment after strict identity, with `skos:broader`, `skos:narrower`, `skos:related`, or `owl:differentFrom` relative to A and B. BroaderG, individual-to-class broaderI, and architectural-part broaderP remain eligible; causation, mitigation, protocol use and co-occurrence alone do not establish hierarchy. Direct facts bypass identity and direct/shared-role evidence establishes relatedness if no hierarchy is selected. Deterministic evidence adds no model vote. This is a target until the P2 review records implementation and gates.
