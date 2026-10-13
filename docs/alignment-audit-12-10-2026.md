# YAW Alignment Audit — 2026-10-12

**Scope:** Compare the intended operation (stated by the owner on 2026-10-12), the governing docs ([`FOUNDATIONAL.md`](../FOUNDATIONAL.md), [`ADR-001`…`ADR-012`](ADR.md), [`epics-alignment-roadmap.md`](epics-alignment-roadmap.md), [`how-to-handle-spacy-llm.md`](how-to-handle-spacy-llm.md)), and the source code at commit `a414206` (clean working tree).
**Method:** I read every module in `src/yaw/` line by line. I grepped for symbol usage. I read the installed library sources (`spacy_llm`, `pyshacl` 0.40.1) and the vendored CocoIndex skill references. I made a separate pass over the docs and ADRs. I did a static review of `tests/`. I did not modify any code.
**Line references** are to the file as checked out. Links are relative to `docs/`.

---

## 0. Executive Verdict

The docs and the code have drifted from **each other**. Both have also drifted from the **intended architecture**:

| # | Intended step (owner, 2026-10-12) | What the docs say | What the code does | Verdict |
|---|---|---|---|---|
| 1 | CocoIndex indexes `raw/` into **sqlite-vec** | CocoIndex flow; vectors in `vec0` (F §5 C2) | Chunk vectors are written as **JSON `TEXT`**, not sqlite-vec BLOB/`vec0`. Half the graph state is written **outside CocoIndex target states**. | 🔴 Drift |
| 2 | spaCy + spacy-llm do **Open IE**, building the ontology | Single-pass `spacy.Raw.v1` at ingestion (ADR-001). The AGENTS-mandated how-to still prescribes a 5-task stack. | Raw.v1 does the real extraction. A dead `REL.v1` pipe burns one LLM call per chunk. Predicates are **normalised before storage** (lossy). Typing and definitions come from **hardcoded heuristics**, not spaCy/LLM. | 🔴 Drift |
| 3 | **pySHACL enriches** the ontology | SHACL is **validation only**. No `sh:rule` anywhere in the docs. | Validation only. The report is cached and **never fed back** into the graph. | 🔴 Doc ≠ intent |
| 4 | Steps 2–3 **are** `compile_wiki`, projected into `wiki/` per LLM-Wiki | Open IE runs at **ingestion**. `compile_wiki` = HAC + hypernyms + SHACL + **deterministic, zero-LLM** dump. | `compile_wiki` runs **only SHACL**. It returns hardcoded `clusters_bisected: 0`. **Nothing is ever written to `wiki/`.** | 🔴 Doc ≠ intent, code ≠ doc |
| 5 | FastMCP retrieval with PPR diffusion down to source leaves | Matches intent (F §5 C3) | PPR exists, but **identity merges never reach the graph** (key mismatch). Predicates become **hub nodes**. Output is **markdown only**. | 🟠 Partially working |
| 6 | Beat **2WikiMultiHopQA** and **MuSiQue** | Epic 8: thin spec. `eval/` is referenced but does not exist. | **No harness, no dataset loader, no structured retrieval API.** | 🔴 Absent |

> [!IMPORTANT]
> The owner's mental model and `FOUNDATIONAL.md` disagree on **where Open IE lives** (compile vs ingestion), **what SHACL does** (enrich vs validate) and **what `wiki/` is** (LLM-Wiki vs deterministic dump). These are architectural decisions, not bugs. They must be settled (§9) before any remediation code is written. Otherwise the next epic will drift again.

---

## 1. Layer Responsibility Matrix (Intended vs Actual)

"Competing responsibilities" means more than one module owns the same concern. The table lists the owners found in code.

| Concern | Intended owner | Actual owners in code | Problem |
|---|---|---|---|
| Change detection / incremental state | CocoIndex | CocoIndex (5 tables) **plus** raw `sqlite3` writes for `query_fts`, `entity_vec`, `fact_vec`, `aliases`, `close_matches`, `canon_verdicts`, and `leaf_nodes` (canonical induction) | Deleting or editing a note never removes the side-written rows. See C-1. |
| Entity typing (9 roots) | spacy-llm (Open IE / TextCat) | Hardcoded `CONCEPT_WORDS` + a second full `en_core_web_lg` NER pipeline + LLM `subject_type`, with "concepts" as fallback | Three-way heuristic. See H-2. |
| Entity definitions | `spacy.Summarization.v1` (F §3 table) | Synthetic strings `"{name} ({ns})"` | Summarization is not implemented anywhere. See C-5. |
| Predicate normalisation | One space-L normaliser (ADR-006/012) | `normalize_predicate` (no polarity guard) at extraction **and** `normalize_predicate_antonym_guarded` at canonicalisation | Two competing normalisers in series. See C-4. |
| Relation extraction | `spacy.Raw.v1` (ADR-001) | `llm_raw` **and** `llm_rel` (REL.v1) when steering is active | REL cannot emit anything. See C-3. |
| Chunk↔concept grounding | Extraction provenance | Substring match `name in chunk.text` | Heuristic edges. See H-5. |
| SHACL shapes | `wiki_shapes.ttl` (F §5 C5) | `wiki_shapes.ttl` (slow path) **and** an inline TTL string in `shacl.py` (fast path, used by `compile_wiki`) | Two shape SSOTs. See S-3. |
| Namespace list | `config.TAXONOMY_NAMESPACES` (9) | `config` (9), `rdf.ROOT_NAMESPACES` (11, adds `rules`, `queries`), `ontology.ttl` ("8 Universal Node Root Classes") | Three lists. See S-2. |
| Base predicates | `config.BASE_PREDICATES` | `config` (14), F L269-273 (a different 14), hardcoded SQL whitelist in `unified_edges` (22) | Three lists. See S-1. |
| Retrieval hyperparameters | `config.py` | β inline in `retriever.py`, α and γ literals, `LIMIT 8` | Not configurable. Cannot be calibrated for Epic 8. See S-5. |
| Indexing trigger | `compile_wiki` / `ingest_raw` (MCP) | CLI `yaw index` only. `ingest_raw` is a stub. `compile_wiki` does not index. | The MCP surface cannot build the KB. See C-8. |

---

## 2. Critical Findings (🔴 correctness or benchmark blockers)

### C-1 · Graph state is written outside CocoIndex's target-state model
- **Code:** `_insert_query_fts` opens its own `sqlite3` connection ([pipeline.py:337-360](../src/yaw/indexer/pipeline.py#L337-L360)).
  - `insert_entity_vec` / `insert_fact_vec` / `resolve()` write through a raw `get_db()` connection ([pipeline.py:702-754](../src/yaw/indexer/pipeline.py#L702-L754), [pipeline.py:507-510](../src/yaw/indexer/pipeline.py#L507-L510), [pipeline.py:575-580](../src/yaw/indexer/pipeline.py#L575-L580)).
  - `_create_canonical_node` inserts **directly into `leaf_nodes`** with `source_leaf_id='system:canonical_induction'` ([canonicalize.py:353-385](../src/yaw/indexer/canonicalize.py#L353-L385)).
- **Canon:** CocoIndex auto-cleans only *declared* target states ("Deleting source file automatically removes output file", [patterns.md:57](../.agents/skills/cocoindex/references/patterns.md)). "Mixing Target State with Side Effects" is a listed anti-pattern ([patterns.md:455-465](../.agents/skills/cocoindex/references/patterns.md)).
- **Impact:**
  - Editing or deleting a note leaves orphan FTS rows, vectors, aliases and close_matches. Re-indexing an edited note **appends duplicate FTS rows**, because the contentless table has no key.
  - `leaf_nodes` is no longer "note extractions only" (F L299, Invariant 1).
  - Benchmark runs that re-ingest corpora will be contaminated.

### C-2 · Identity keys disagree between pipeline and canonicaliser, so merges never reach the graph
- **Code:**
  - The pipeline mints node ids as `{ns}:{slugify(name)}` via `_node_id` ([canonicalize.py:388-402](../src/yaw/indexer/canonicalize.py#L388-L402); used at [pipeline.py:405](../src/yaw/indexer/pipeline.py#L405) and [pipeline.py:553-555](../src/yaw/indexer/pipeline.py#L553-L555)). `slugify` is plain ASCII folding ([utils.py:15-20](../src/yaw/utils.py#L15-L20)).
  - `resolve()` computes `surface_id = {ns}:{_normalize_entity_key(name)}`, which strips determiners and lemmatises NOUNs ([canonicalize.py:456-521](../src/yaw/indexer/canonicalize.py#L456-L521), [canonicalize.py:576-577](../src/yaw/indexer/canonicalize.py#L576-L577)). It records aliases under `surface_id` and the raw name ([canonicalize.py:1208-1229](../src/yaw/indexer/canonicalize.py#L1208-L1229)).
  - `unified_edges` joins `aliases.alias_id = leaf_relations.subject_id` ([schema.py:396-401](../src/yaw/storage/schema.py#L396-L401)).
- **Impact:**
  - For any mention where slug ≠ normalised key, the alias join misses. Examples: "The Transformers" gives slug `the-transformers` but key `transformer`; plurals and possessives behave the same way.
  - Tier 1/Tier 2 merges therefore never collapse edges.
  - Induction also writes a **second** `leaf_nodes` row and a **second** `entity_vec` row under the other key. The next mention's Tier 2 KNN then sees duplicate candidates.
  - This directly defeats the bridging-entity consolidation that 2Wiki/MuSiQue depend on. HippoRAG attributes its largest 2Wiki gain to entity linking and synonymy edges ([HippoRAG](https://arxiv.org/abs/2405.14831)).

### C-3 · `spacy.REL.v1` is structurally dead, yet invoked on every chunk after the first note
- **Code:**
  - The base pipeline is loaded with `exclude=["ner"]` ([extractor.py:177-185](../src/yaw/nlp/extractor.py#L177-L185)).
  - No span producer runs before `llm_rel` ([extractor.py:302-341](../src/yaw/nlp/extractor.py#L302-L341)).
  - The steered pipeline (`enable_rel=True`) activates whenever `unified_edges` has any predicates ([pipeline.py:794-800](../src/yaw/indexer/pipeline.py#L794-L800), [pipeline.py:412-425](../src/yaw/indexer/pipeline.py#L412-L425)).
- **Library:** the spacy-llm REL parser keeps a relation only if `0 <= dep < len(doc.ents)` (`.venv/.../spacy_llm/tasks/rel/parser.py:27`). With `doc.ents == ()` every relation is dropped.
- **Impact:**
  - Each chunk costs an extra LLM call with guaranteed-empty output.
  - Extraction behaviour depends on whether the DB was empty when the note was indexed.
  - The custom `OPEN_IE_REL_TEMPLATE` never renders `labels` ([templates.py:7-52](../src/yaw/nlp/templates.py#L7-L52)). Even with ents present, REL's `labels=BASE_PREDICATES` ([extractor.py:159](../src/yaw/nlp/extractor.py#L159)) would be inert.

### C-4 · Open IE output is made lossy before persistence, by an unguarded normaliser
- **Code:**
  - `_parse_single_candidate` replaces the surface predicate with `normalize_predicate(...)` before the triple is built ([parsing.py:250-252](../src/yaw/nlp/parsing.py#L250-L252)). `_build_single_triple` does the same ([components.py:768-769](../src/yaw/nlp/components.py#L768-L769)).
  - `normalize_predicate` falls back to GloVe cosine ≥ `CANON_SIM_THRESHOLD` against base predicates **without** the polarity mask ([components.py:247-298](../src/yaw/nlp/components.py#L247-L298)).
  - The polarity-guarded version exists but runs only later, in canonicalisation ([components.py:553-583](../src/yaw/nlp/components.py#L553-L583), [canonicalize.py:782](../src/yaw/indexer/canonicalize.py#L782)). By then the stored `leaf_relations.predicate` is already rewritten.
- **Canon violated:**
  - Invariant 1, "lossless, never drops out-of-vocabulary relations" (F L101).
  - ADR-012 predicate ladder: "polarity-guarded static-vector fallback" (F L177).
  - The fact text is specified as `{subject} {surface_predicate} {object}` (F L264). The code embeds the *normalised* predicate ([pipeline.py:682-684](../src/yaw/indexer/pipeline.py#L682-L684)).
- **Impact:**
  - Antonym collapse is possible, e.g. "inhibits" → `causes`. Distributional vectors place antonyms close together ([Mrkšić et al., 2016](https://aclanthology.org/N16-1018/)), as F L162 itself cites.
  - This produces false `DialecticalIntegrityShape` violations and corrupts query-to-fact seeding.

### C-5 · Definitions are synthetic, so space-S embeddings are name-only and Tier 2 is weakened
- **Code:**
  - Raw-triple nodes get `definition=f"{name} ({ns})"` ([parsing.py:346-360](../src/yaw/nlp/parsing.py#L346-L360)).
  - `_entity_definition` (sentence context) only applies to `doc.ents`. `doc.ents` is empty at that point, because anchoring happens later inside `parse_raw_triples` ([extractor.py:233-250](../src/yaw/nlp/extractor.py#L233-L250)).
  - `spacy.Summarization.v1` is referenced in F L155 but appears **nowhere** in `src/`.
- **Impact:**
  - Space-S input becomes `"task: sentence similarity | query: X: X (concepts)"`. The docstring at [components.py:620-624](../src/yaw/nlp/components.py#L620-L624) explicitly calls this kind of tautological string undesirable.
  - Blocking recall for Tier 2 falls to surface-string similarity.

### C-6 · Predicates are treated as entities, so they get embedded, enter Tier 2, and become PPR hubs
- **Code:**
  - Triple predicates are registered as raw nodes ([pipeline.py:449-454](../src/yaw/indexer/pipeline.py#L449-L454)).
  - All raw nodes are resolved with `kind="entity"` and embedded in space S ([pipeline.py:487-510](../src/yaw/indexer/pipeline.py#L487-L510)).
  - `_store_chunk_concept_edges` emits `manifests_in`/`mentions` edges for them too ([pipeline.py:594-637](../src/yaw/indexer/pipeline.py#L594-L637)).
  - `_is_concept("predicates:…")` is true because `predicates` is in `TAXONOMY_NAMESPACES` ([retriever.py:95-98](../src/yaw/retrieval/retriever.py#L95-L98)).
- **Canon violated:** "Predicates never enter Tier 2 … Predicates trigger **zero** embedding calls" (F L169, L177, L289).
- **Impact:** `predicates:causes` links every chunk containing a causal statement with downward conductance 1.0 ([retriever.py:475-476](../src/yaw/retrieval/retriever.py#L475-L476)). This is topic leakage of exactly the kind conductance asymmetry was designed to prevent. On multi-hop QA it diffuses mass to unrelated passages.

### C-7 · `compile_wiki` does not compile; `wiki/` is never materialised
- **Code:** [daemon.py:236-282](../src/yaw/server/daemon.py#L236-L282) runs `validate_wiki_graph_fast`, caches violations, and returns literal `"clusters_bisected": 0, "rewired_edges": 0`.
  - A grep finds no writer for `compiled_nodes` or `compiled_relations`.
  - There is no Ward HAC.
  - `HYPERNYM_INFERENCE_TEMPLATE` is defined ([templates.py:101-151](../src/yaw/nlp/templates.py#L101-L151)) but never used.
  - No code writes `wiki/`.
- **Canon violated:** F L483-492, ADR-002, ADR-011 L95.
- **Roadmap claim:** Epics 1–7.6 are "COMPLETE" (roadmap L25-39), yet ADR-002 (Epic 2) has no implementation.

### C-8 · The MCP surface cannot ingest or index
- **Code:**
  - `ingest_raw` returns a hardcoded dict and does nothing ([daemon.py:416-428](../src/yaw/server/daemon.py#L416-L428)). Its signature also differs from F L481 (`file_path` vs `content/title/url`).
  - Indexing is reachable only via the CLI `yaw index` ([cli.py:139-149](../src/yaw/cli.py#L139-L149)).
  - `walk_dir(live=True)` is passed, but `update_blocking` runs one pass ([pipeline.py:901-903](../src/yaw/indexer/pipeline.py#L901-L903), [pipeline.py:977](../src/yaw/indexer/pipeline.py#L977)).

### C-9 · `memo=True` is unsound: extraction depends on mutable DB state
- **Code:** `process_vault_raw` is `@coco.fn(memo=True)` ([pipeline.py:757](../src/yaw/indexer/pipeline.py#L757)). Its output depends on:
  - `query_live_predicates(DB)`, i.e. few-shot steering ([pipeline.py:794](../src/yaw/indexer/pipeline.py#L794)), and
  - the live KB and aliases accumulated from previously processed files.
- **Canon:** inputs that affect computation must be declared, e.g. via `ContextKey(detect_change=True)` ([patterns.md:395](../.agents/skills/cocoindex/references/patterns.md)).
- **Impact:**
  - Results are **order-dependent and non-reproducible**.
  - `spacy.BatchCache.v1` keys on the prompt, which includes DB-dependent few-shots, so identical text re-calls the LLM. This contradicts ADR-001 L39 (flagged by the doc pass).
  - It also makes benchmark runs non-deterministic.

### C-10 · ADR-001 "defensive fallback" parses the **source note** as LLM JSON
- **Code:** if no triples come back, `parse_extracted_triples(text, …)` runs on the chunk text itself ([pipeline.py:456-471](../src/yaw/indexer/pipeline.py#L456-L471)). `parse_extracted_triples` is an alias of `parse_raw_triples`, a JSON reply parser ([parsing.py:389](../src/yaw/nlp/parsing.py#L389)).
- **Impact:** notes containing JSON code blocks with `subject`/`object` keys get them ingested as facts. Other notes produce nothing. Either way, this is not Open IE.

### C-11 · Ontology punning violates the Non-Punning Invariant
- **Code:** every root is declared `a owl:Class , skos:Concept` ([ontology.ttl:38-95](../src/yaw/ontology.ttl#L38-L95)).
- **Canon:** "Punning `concepts:Foo a owl:Class, skos:Concept` is strictly forbidden" (F §5 C5.3, ADR-009). ADR-008 L115-177 still *mandates* punning and is not marked as amended by ADR-009.

---

## 3. Hardcoded Extraction Inventory

These are places where hand-written rules replace spaCy, spacy-llm or ontology-driven behaviour. The owner names this as a core concern.

| ID | Location | What is hardcoded | Effect | Should come from |
|---|---|---|---|---|
| H-1 | [components.py:53-75](../src/yaw/nlp/components.py#L53-L75) | `CONCEPT_WORDS`: 19 physics/CS words ("superconductivity", "relativity", …) | Any mention containing one of them is forced to `concepts` | Ontology / spacy-llm typing |
| H-2 | [components.py:108-127](../src/yaw/nlp/components.py#L108-L127), [parsing.py:196-215](../src/yaw/nlp/parsing.py#L196-L215) | Namespace typing: CONCEPT_WORDS → out-of-context NER on the bare string (loads a 2nd full `en_core_web_lg`) → "all words capitalised" → LLM `*_type` → `concepts` | Namespace drives the Tier 2 partition and the node id, so one entity can split across `entities:`/`concepts:` | One typing decision, in the Open IE prompt or TextCat |
| H-3 | [components.py:130-151](../src/yaw/nlp/components.py#L130-L151) | `PREDICATE_ALIASES` mapping onto `is_a`, which is **not** in `BASE_PREDICATES` | Dead entries. "instance of" → `is_a` instead of `instanceOf` | Ontology `skos:altLabel` on properties |
| H-4 | [components.py:429-468](../src/yaw/nlp/components.py#L429-L468) | `SYNSET_DICTIONARY`: `produces`→`causes` (L440) even though `produces` is a base predicate; `consists of`/`comprises`→`partOf` (L458-459) **inverts direction** | Semantic corruption | Ontology property labels + `owl:inverseOf` |
| H-5 | [pipeline.py:602-607](../src/yaw/indexer/pipeline.py#L602-L607) | Chunk→concept grounding by `name.casefold() in chunk.text.casefold()` | Short names ("AI", "map") attach to unrelated chunks | Extraction provenance (the chunk the triple came from) |
| H-6 | [pipeline.py:618-636](../src/yaw/indexer/pipeline.py#L618-L636) | `mentions` weight `0.2` baked into storage | Conductance policy lives in two places (storage + retriever L477-478) | Retriever only |
| H-7 | [pipeline.py:309-334](../src/yaw/indexer/pipeline.py#L309-L334) | `classify_freeform_llm` calls `/api/chat` directly, bypassing spacy-llm and BatchCache | Violates the model-stack contract (F §3) | `spacy.TextCat` via the pipeline, or remove |
| H-8 | [pipeline.py:832-835](../src/yaw/indexer/pipeline.py#L832-L835) | Category fallback `rel.split("/")[0]` → always `"raw"` | Meaningless metadata | — |
| H-9 | [examples.py:33-160](../src/yaw/nlp/examples.py#L33-L160) | 14 physics-domain few-shots in REL format (`dep/dest`), fed to a dead REL pipe | Domain bias. No few-shots reach the Raw.v1 task that actually runs | Raw-format few-shots, domain-neutral |
| H-10 | [schema.py:407-417](../src/yaw/storage/schema.py#L407-L417) | 22-item predicate whitelist inside the `unified_edges` SQL | A third, divergent predicate list | `config.BASE_PREDICATES` + ontology properties |
| H-11 | [shacl.py:385-533](../src/yaw/ontology/shacl.py#L385-L533) | Entire fast-path SHACL shapes graph as an inline Python string | Shapes SSOT split | `wiki_shapes.ttl` |
| H-12 | [components.py:301-427](../src/yaw/nlp/components.py#L301-L427) | `POLARITY_EXCLUSION_MASK` with inflected word lists | Mandated by epic7 spec. Duplicates what `owl:propertyDisjointWith` already states in `ontology.ttl` L150-174 | Derive from ontology disjointness |

---

## 4. SSOT and Cleanliness Violations (🟠)

| ID | Finding | Evidence |
|---|---|---|
| S-1 | Three divergent base-predicate sets | F L269-273 has `dependsOn`, `precedes`; [config.py:70-85](../src/yaw/config.py#L70-L85) has `prevents`, `mentions` instead; [schema.py:410-416](../src/yaw/storage/schema.py#L410-L416) has 22 items. Doc pass: `prevents` is used by shapes and masks but is not among F's 14. |
| S-2 | Three namespace lists | [config.py:59-69](../src/yaw/config.py#L59-L69) (9), [rdf.py:38-50](../src/yaw/storage/rdf.py#L38-L50) (11), [ontology.ttl](../src/yaw/ontology.ttl) ("8 Universal Node Root Classes") |
| S-3 | Two SHACL shape graphs; compile uses the inline one | [shacl.py:132-156](../src/yaw/ontology/shacl.py#L132-L156) vs [shacl.py:385-533](../src/yaw/ontology/shacl.py#L385-L533); [daemon.py:259](../src/yaw/server/daemon.py#L259) |
| S-4 | Two predicate normalisers | See C-4 |
| S-5 | Retrieval hyperparameters are not in config | β `0.60/0.20/0.20` inline ([retriever.py:800](../src/yaw/retrieval/retriever.py#L800)), also `0.20*sim` at [retriever.py:592](../src/yaw/retrieval/retriever.py#L592); γ ([retriever.py:445](../src/yaw/retrieval/retriever.py#L445)); α/iterations ([retriever.py:1165](../src/yaw/retrieval/retriever.py#L1165)); context `LIMIT 8` ([retriever.py:987](../src/yaw/retrieval/retriever.py#L987)). F L454 requires calibration of exactly these. |
| S-6 | Global monkeypatch of `spacy.Language.create_pipe` at import time | [components.py:874-904](../src/yaw/nlp/components.py#L874-L904). Contradicts "zero monkeypatching" ([how-to-handle-spacy-llm.md](how-to-handle-spacy-llm.md), AGENTS.md). Test-model injection (`_TestModelWrapper`, `_INJECTED_TEST_MODELS`) lives in production code ([components.py:813-871](../src/yaw/nlp/components.py#L813-L871)). |
| S-7 | Private-attribute mutation of the CocoIndex connection | `_active_managed_conn._conn = …` ([pipeline.py:937-967](../src/yaw/indexer/pipeline.py#L937-L967)) |
| S-8 | Dead config keys | `OPEN_IE`, `INFER_ONTOLOGIES`, `USE_LLM_IF_NER_BELOW`, `CANON_LLM_CONFIDENCE`, `CANON_SIM_HIGH/LOW`, `CANON_LEXICAL_MIN`, `LLM_DEFINE_TEMPERATURE` are declared in [config.py](../src/yaw/config.py) and read nowhere in `src/`. Runtime toggles live in the `runtime_settings` table instead. |
| S-9 | Dead code | `DynamicConceptMentionScoper` ([components.py:586-614](../src/yaw/nlp/components.py#L586-L614)), `HYPERNYM_INFERENCE_TEMPLATE`, `rerank_candidates`, plus a block of "backwards-compatibility" re-exports in [daemon.py:70-79](../src/yaw/server/daemon.py#L70-L79) and [retriever.py:1212-1557](../src/yaw/retrieval/retriever.py#L1212-L1557) kept for tests |
| S-10 | Schema ≠ spec | `entity_vec(entity_id, partition_key)` with no `PARTITION KEY` ([schema.py:58-63](../src/yaw/storage/schema.py#L58-L63)) vs F L330-334 `node_id`, `namespace PARTITION KEY`. `close_matches` lacks `CHECK(score>0)` and uses `resolved_at/evidence` vs F L318-326. |
| S-11 | Brute-force vector scans | Tier 2 "KNN" is `ORDER BY vec_distance_cosine` over all of `entity_vec` ([canonicalize.py:1083-1093](../src/yaw/indexer/canonicalize.py#L1083-L1093)). Chunk similarity is the same over `note_chunks` JSON text ([retriever.py:567-577](../src/yaw/retrieval/retriever.py#L567-L577)). Only `fact_vec` uses `MATCH … AND k=?`. |
| S-12 | Over-broad alias join | `a_pred` join on `predicate_id OR predicate OR replace(lower(...))` ([schema.py:402-406](../src/yaw/storage/schema.py#L402-L406)) can multiply rows when several alias rows match, which inflates edge weight in PPR. |

---

## 5. Retrieval Layer Drift (🟠, PPR "all the way to the leaves")

| ID | Finding | Evidence | Benchmark impact |
|---|---|---|---|
| R-1 | Wikilink targets are **file paths** (`raw/b.md`), not chunk or entity nodes, so they become dead-end nodes | [pipeline.py:229-238](../src/yaw/indexer/pipeline.py#L229-L238); retriever admits every non-ontology endpoint ([retriever.py:216-227](../src/yaw/retrieval/retriever.py#L216-L227)) | Low on benchmarks; high on real vaults |
| R-2 | Fact seeds use raw `leaf_relations.subject_id/object_id`, but graph nodes are alias-rewritten canonical ids | [retriever.py:693-708](../src/yaw/retrieval/retriever.py#L693-L708) vs [schema.py:390-391](../src/yaw/storage/schema.py#L390-L391) | Merged entities lose their query-to-fact seed mass (compounds C-2) |
| R-3 | `P(e\|a)` ignores `close_matches`. The KB is synced only from `leaf_nodes`, `aliases` and predicates. | [canonicalize.py:169-287](../src/yaw/indexer/canonicalize.py#L169-L287) vs F L168, L279-280 | Polysemy handling (Invariant 4) is not implemented |
| R-4 | Winner-take-all at ingestion: `_lookup_kb` returns `candidates[0]`; Tier 1 matches across namespaces via `LIKE '%:{slug}'` | [canonicalize.py:609-620](../src/yaw/indexer/canonicalize.py#L609-L620), [canonicalize.py:679-688](../src/yaw/indexer/canonicalize.py#L679-L688) | Directly contradicts Invariant 4's anti-winner-take-all rule. "Mercury" the planet and the element merge. |
| R-5 | Evidence score uses `aliases.MAX(score)` and a default `1.0` in place of `P(e\|a)` | [retriever.py:898-934](../src/yaw/retrieval/retriever.py#L898-L934) vs F L64 | Score formula ≠ spec |
| R-6 | The dense-candidate chunk pool from seeding is discarded; candidates come from PPR mass only (`pool = max(5k, 20)`) | [retriever.py:1144](../src/yaw/retrieval/retriever.py#L1144), [retriever.py:1055-1062](../src/yaw/retrieval/retriever.py#L1055-L1062) | Strong dense hits outside the top PPR mass can never be returned |
| R-7 | Query mentions include **every non-stop token and every 1–3-gram** | [retriever.py:141-185](../src/yaw/retrieval/retriever.py#L141-L185) | Over-seeding of generic nodes; noisy `s_alias` |
| R-8 | The only retrieval output is a markdown string | [retriever.py:1139-1205](../src/yaw/retrieval/retriever.py#L1139-L1205) | Recall@k requires parsing markdown. A structured ranked-passage API is required. |

---

## 6. Docs vs Intent, and Doc-Internal Contradictions

### 6.1 Where the docs themselves diverge from the owner's intent
| Topic | Owner intent | Docs | Evidence |
|---|---|---|---|
| Open IE placement | Part of `compile_wiki` | Continuous, at ingestion | F L83-85, L266-273; ADR-011 L28-31; F §5 C5 ("pipeline.py does not invoke SHACL") |
| SHACL role | **Enrichment** | Closed-world **validation** only. No `sh:rule` / SHACL-AF anywhere. Inference is done by SQL CTE pre-closure. | ADR-009 L83-84, F L147. pySHACL 0.40.1 ships rule expansion: `pyshacl.shacl_rules()` (`entrypoints.py:362`), `TripleRule`/`SPARQLRule`, `iterate_rules`; standard: [SHACL-AF](https://www.w3.org/TR/shacl-af/) |
| `wiki/` | LLM-Wiki projection | Optional, deterministic, zero-LLM dump | F L89, L488; Karpathy: the LLM writes summaries, synthesis pages, `index.md`, `log.md`, filed-back answers ([karpathy-llm-wiki.md](karpathy-llm-wiki.md) L28-32, L48, L54-66) |
| Benchmarks as design driver | Steps 1–5 tuned to win 2Wiki/MuSiQue | Epic 8 is "specified", blocked on 7.7. No dataset versions, splits, corpus construction or reader model. | roadmap L42, L469-507 |

### 6.2 Doc-internal contradictions (from the doc pass; highest-impact only)
1. **Three extraction stacks.**
   - ADR-001 L22-24: Raw.v1 only.
   - [how-to-handle-spacy-llm.md](how-to-handle-spacy-llm.md) L17-22, L120-264: SpanCat + REL + TextCat + EntityLinker + Summarization. This is the guide AGENTS.md mandates.
   - F L594/L606 and README L57-59: `spacy.NER.v3 (primary)`.
   - The how-to also prescribes **LLM gray-zone adjudication** (L235, L277), which contradicts "zero LLM adjudication" (F L169, ADR-012).
2. **Upward conductance** has three formulas: flat 0.2 (ADR-002, F L236), linear depth (ADR-009 L69, F §5 C3), log (epic7 L30).
3. **Punning:** ADR-008 mandates it; ADR-009 and F forbid it. ADR-008 is not marked amended.
4. **IRI schemes:** `https://yaw.local/concept/` (ADR-005 L64) vs `http://yaw.local/concepts/` (ADR-008 L46). There are four different chunk-URI schemes.
5. **Status claims:** F L5 says "Epic 6.2 Complete (375 tests)". F L21 and the roadmap say 1–7.6 complete. [TEST_READY.md](../TEST_READY.md) shows the Epic 7 red phase (2 pass / 7 fail) but is cited by F as authoritative. Roadmap L344 references a nonexistent `pre_epic8_realignment_plan.md`.
6. **Stale links:** 343 `file:///home/harri/yaw/...` links (F: 77, roadmap: 71). Line-anchored links such as `db.py#L85-L168` point at code that has since moved to `schema.py`.
7. **Synset direction bug in the spec itself:** epic7 L802/L820-823 (`produces`→`causes`, `comprises`→`partOf`). H-4 implemented it faithfully.

---

## 7. Test Suite Reality

| Check | Result |
|---|---|
| Fake LLM Open IE | [conftest.py:184-237](../tests/conftest.py#L184-L237) returns adjacent capitalised candidates chained with the single predicate **"relates to"**, wrapped in a ```` ```json ```` fence |
| What this cannot catch | Predicate normalisation and polarity (C-4), real typing (H-2), the dead REL path (C-3), definition quality (C-5), multi-hop bridging. Extraction-level tests largely verify the fake's own heuristic. |
| Internal patching | `monkeypatch.setattr` / `mock` appear in 15 test files (31 occurrences). Classifying each against Invariant 6 is still an open item; the subagent run was interrupted. |
| End-to-end path | No test drives raw note → CocoIndex → spacy-llm → SHACL compile → `wiki/` → `query_wiki` with evidence, because `compile_wiki` and `wiki/` do not exist |
| Benchmark harness | **None.** "2wiki"/"musique" appear only in `docs/` (ADR-012, roadmap). `eval/` does not exist. |
| Suite health | A partial run (`-x`) stopped at the first failure: `tests/test_adversarial_epic2.py:34` → `no such table: leaf_nodes` against the shared `sample-llm-wiki/.cache/yaw.db`. That test reads shared mutable state instead of `tmp_path`, which violates the [engineering-real-tests](../.agents/skills/engineering-real-tests/SKILL.md) skill. The full-suite result was not obtained before interruption. |
| Milestone-named suites | `test_m1/m2/m3/m3_2_adversarial_challenge.py`, `test_adversarial_*`: names track project history rather than invariants; consolidation candidates |

---

## 8. Benchmark Readiness (2WikiMultiHopQA, MuSiQue)

**Reference protocol** (HippoRAG / HippoRAG 2, which YAW's ADR-012 already cites):
- 1,000 sampled dev questions per dataset over a shared passage corpus.
- Retrieval metrics: passage Recall@2 and Recall@5.
- QA metrics: EM/F1 from a reader LM.
- Indexing uses LLM Open IE over passages, then a KG with synonymy edges, then PPR ([HippoRAG](https://arxiv.org/abs/2405.14831), [HippoRAG 2](https://arxiv.org/abs/2502.14802); corpus/query JSON format in [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)).
- Dataset sources: [2WikiMultiHopQA](https://arxiv.org/abs/2011.01060), [MuSiQue](https://arxiv.org/abs/2108.00573).

| Requirement | Status | Blocking findings |
|---|---|---|
| Corpus loader (passages → `raw/` notes, title as frontmatter) | ❌ | — |
| Deterministic, order-independent indexing | ❌ | C-9, C-3 |
| Entity consolidation across passages (the core multi-hop signal) | ❌ | C-2, C-5, R-4, H-2 |
| Clean fact graph without predicate hubs | ❌ | C-6 |
| Query-to-fact seeding (HippoRAG 2 §6.1) | ⚠️ implemented, undermined | C-4 (normalised fact text), R-2 |
| Structured ranked passage output | ❌ | R-8 |
| Configurable hyperparameters for calibration | ❌ | S-5 |
| Recall@2/@5 metric (not in the official scripts; it is the HippoRAG convention) | ❌ | The roadmap's "zero handrolled metrics" rule conflicts with this; needs a decision |
| Reader for EM/F1 | ❌ unspecified | `query_wiki` does "zero synthesis" by design |
| Throughput at about 10k passages | ⚠️ | `max_inflight_components=1` ([pipeline.py:972](../src/yaw/indexer/pipeline.py#L972)), 2 LLM calls per chunk (C-3), brute-force scans (S-11) |

---

## 9. Decisions Required Before Remediation

| # | Decision | Options | Recommendation (with rationale) |
|---|---|---|---|
| D-1 | Where does Open IE run? | (a) In the CocoIndex flow at ingestion (current docs); (b) in `compile_wiki` (owner's statement) | **(a), with `compile_wiki` = consolidation + SHACL + projection.** CocoIndex's memoised per-file components are its documented LLM-extraction pattern ([patterns.md:223-302](../.agents/skills/cocoindex/references/patterns.md)). Moving extraction to compile would forfeit incremental re-extraction. Under (a), "compile" still owns cross-document steps (identity, enrichment, wiki). Then amend the owner-facing description, or else amend F. |
| D-2 | What does SHACL do? | (a) Validate only; (b) validate + enrich via SHACL-AF rules (`pyshacl.shacl_rules`) | **(b).** It matches intent, and pySHACL 0.40.1 already supports it. Enrichment output must land in an engine-owned table (e.g. `compiled_relations`, `source_rule='shacl:<shape>'`) so Invariant 1 holds. This replaces ad-hoc Python closures and the SYNSET/POLARITY tables with declarative rules. |
| D-3 | What is `wiki/`? | (a) Deterministic projection; (b) Karpathy LLM-Wiki (LLM-authored pages, `index.md`, `log.md`) | Ask the owner. **(a) first, (b) layered on top** keeps retrieval reproducible for benchmarks while satisfying LLM-Wiki for humans. Pages must never be re-indexed (F §5 C5). |
| D-4 | Which extraction stack is canonical? | ADR-001 (Raw.v1) vs how-to (5 tasks) vs F (NER.v3) | **ADR-001 Raw.v1, with typing and definitions in the same JSON schema.** Rewrite or retire the how-to. Delete the REL path. |
| D-5 | Are fixed vocabularies allowed at extraction? | 14 base predicates, CONCEPT_WORDS, synsets, masks | **Store surface predicates losslessly. Map to base predicates only at view/compile time, via ontology labels and SHACL rules.** This matches Invariant 1 and makes H-1…H-12 removable. |
| D-6 | Benchmark metric policy | Official EM/F1 only vs also Recall@k | **Both.** Recall@2/5 is the comparable number for retrieval engines (HippoRAG). Implement it once in `eval/` as the SSOT. |

---

## 10. Suggested Remediation Order (after D-1…D-6)

These map onto ADR-012 / Epic 7.7 where possible. Each step is test-first, with real SQLite and `tmp_path`.

1. **Single identity key (C-2, R-2).** `_node_id` must be `resolve()`'s normalised key, from one function. Invariant test: an alias written by `resolve` rewrites the corresponding `unified_edges` row.
2. **All state through CocoIndex targets (C-1, C-9).**
   - Declare FTS, vector, alias and close-match rows as target states, or move cross-document identity into a compile stage that rebuilds from leaf tables.
   - Remove DB-dependent steering from memoised components.
   - Use `Annotated[NDArray, VectorSchema]` for chunk vectors ([connectors.md:174](../.agents/skills/cocoindex/references/connectors.md)).
3. **Extraction contract (C-3, C-4, C-5, C-10, H-1…H-9).**
   - One Raw.v1 prompt returning `{subject, subject_type, predicate, object, object_type, subject_definition?}` with surface predicates.
   - Delete REL, the source-text "fallback", CONCEPT_WORDS and the unguarded normaliser.
   - Predicates are not nodes (C-6).
4. **Compile = consolidation + SHACL-AF enrichment + projection (C-7, D-2, D-3).** Implement HAC/hypernyms only if benchmark ablation shows a gain; otherwise de-scope ADR-002 explicitly.
5. **Retrieval fixes (R-3…R-8, S-5).**
   - Build `P(e|a)` from `close_matches`; drop the cross-namespace `LIKE`.
   - Move all hyperparameters into config.
   - Add a structured `retrieve_passages()` API used by both `query_wiki` and `eval/`.
6. **`eval/` harness (§8).** Loader, deterministic index, Recall@2/5, EM/F1, calibration/held-out split per roadmap L503-507, plus the ablations already specified (λ=1, closeMatch off, λ=0).
7. **Doc consolidation.**
   - Fix the 343 stale links.
   - Mark ADR-008 as amended by ADR-009/012.
   - Retire or rewrite the how-to.
   - Reset the status lines in F and the roadmap to verified facts.

---

## Appendix A — Files Read

`src/yaw/{config,cli,utils}.py`, `indexer/{pipeline,canonicalize}.py`, `nlp/{extractor,components,parsing,templates,examples,models,embeddings}.py`, `storage/{db,schema,rdf}.py`, `retrieval/retriever.py`, `server/daemon.py`, `ontology/shacl.py`, `ontology.ttl`, `tests/conftest.py`; `FOUNDATIONAL.md`, `docs/ADR.md`, ADR-001…012, roadmap, how-to-handle-spacy-llm, open-ie-authoritative-source, karpathy-llm-wiki, rdf-owl-shacl, power-pagerank, epic7 spec, USER_STORIES, TEST_READY, audit_report; `.agents/skills/cocoindex/references/{connectors,patterns,api_reference}.md`; `.venv/.../spacy_llm/tasks/rel/{parser,task}.py`; `.venv/.../pyshacl/{entrypoints,__init__}.py`.

**Not covered in depth:** `ontology/{loader,owl,registry,validation}.py` (surveyed by grep only), and per-occurrence classification of test patching (§7).
