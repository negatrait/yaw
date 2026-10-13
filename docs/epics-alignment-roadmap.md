# YAW engineering roadmap

**Audience:** Coding agents implementing and verifying YAW.\
**Updated:** 2026-10-12\
**Authority:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-014](adr/ADR-014.md); implementation status lives in [Epic 7.8](epic7.8-open-ie-ontology-alignment.md) and [TEST_READY](../TEST_READY.md).

## Execution order and hard entry gate

**7.8 complete → 8.1 complete → 8.2 complete → 8.3 release.** There is no parallel start of 8.x while a required 7.8 gate remains unverified. The complete extraction/compilation/retrieval path must already function before 8.1.

| Work | Current scope | Status |
| --- | --- | --- |
| Epics 1–7.6 | Historical development; their old counts/results are baseline-specific | Historical; no current conformance claim |
| Epic 7.7 | L/S/R stores and formatters, fact/alias seeding, candidate distributions and dimension errors | Retained work absorbed into 7.8 completion; not an independent bypass |
| **Epic 7.8** | One functioning source-to-knowledge pipeline, migration, deletion/restart, zero fallbacks, real-model pilot and native platform proof | **IN PROGRESS; runtime gates open** |
| Epic 8.1 | Operational hardening of that same functioning pipeline | **BLOCKED by all 7.8 gates** |
| Epic 8.2 | Held-out corpus and external benchmark quality/cost evaluation | **BLOCKED by 8.1** |
| Epic 8.3 | Packaging, CI and release of the verified engine | **BLOCKED by 8.2** |

Start with the [coding-agent execution contract](epic7.8-open-ie-ontology-alignment.md#coding-agent-execution-contract). Do not use the old roadmap's five-task extraction, B-tree hub splitting, post-sync compiler, live-lint, SSE endpoint assumptions, obsolete Python matrix or arbitrary latency guarantees. Those duplicated implementation plans are removed from this active roadmap; historical rationale remains in the amended [ADR records](ADR.md).

The 7.8 handoff must identify a commit, a complete locked runtime, native platforms, configuration/model/resource digests, all E78 results, real-model pilot outcomes, migration evidence and a search/integration audit proving obsolete production paths are removed. Documentation-only completion is insufficient.

## Retained Epic 7.7 work

Implement representation-space separation and query-to-fact/alias/chunk/FTS seeding under the current [PPR guide](power-pagerank.md). Candidate similarity never becomes automatic identity. Preserve property/resource IRIs, qualifiers and dimensional validation. Retire normalized-label hard merges, unqualified fact rendering, namespace-only SKOS mapping and self-generated calibration ground truth. Work is complete only with the 7.8 acceptance suite.

## Epic 8.1 — Operational hardening

**Entry:** Every 7.8 gate is checked with runtime evidence.\
**Owners:** `src/yaw/server/daemon.py`, `src/yaw/indexer/pipeline.py`, `src/yaw/indexer/compiler.py`, `src/yaw/storage/db.py`, platform launcher/lifecycle code.\
**Contract:** [Compiler guide](knowledge-compilation.md), [FOUNDATIONAL interfaces](../FOUNDATIONAL.md#component-4-fastmcp-protocol-interface).

### Work and acceptance

1. Exercise the existing same-path handlers for `ingest_raw`, `compile_wiki`, `query_wiki` and `lint_wiki` through real FastMCP clients. Use Streamable HTTP at the configured `/mcp` endpoint and stdio through the same handlers. Do not introduce legacy SSE `/sse` + `/messages/` implementations. [FastMCP transports](https://gofastmcp.com/deployment/running-server).
2. Harden startup/shutdown, cancellation, interrupted publication, watcher event coalescing, explicit failed-job retry, resource reload and reconciliation after restart. Retry runs the same implementation/input policy; no second daemon/compiler is a recovery mechanism.
3. Verify serialized SQLite publication and consistent read snapshots under simultaneous queries and source updates. Report pending/partial/failed scopes, queue state and source/compiled generations. Do not silently serve stale compiled knowledge as current.
4. Exercise the already functioning deterministic Markdown sink, output pruning and unchanged-generation reuse; raw notes and source evidence remain authoritative.
5. Run native Windows and Unix lifecycle/path/lock tests and workload tests at 100 and 500 documents. Record token/claim/graph counts, hardware, indexing/update/delete/restart time, query/lint latency distributions, memory and model calls. The document count alone is not a load specification.

**Deliver in order:** (1) lifecycle/cancellation/retry tests through the existing handlers; (2) concurrent read/publication and restart tests; (3) native workload runs. Extend the [E78 test owners](../TEST_INFRA.md#test-target-map). Add `eval/run_workload.py` and versioned `eval/manifests/workload-100.json` / `workload-500.json` as **PLANNED 8.1 deliverables**. Each manifest fixes sources/token sizes, update/delete sequence, concurrent query schedule, model/configuration identity and report path. The runner returns nonzero for correctness failures and records timings without inventing an SLA.

After implementing the runner, execute `uv run --locked python eval/run_workload.py --manifest eval/manifests/workload-100.json`, then the 500-document manifest. Save per-phase initial-ingest, unchanged-reuse, edit/delete and interruption/restart results. A workload run that skips a phase does not satisfy the exit gate.

**Verification owners:** real client/lifecycle tests, publisher interruption tests and workload fixtures under `tests/`. Existing test filenames may be extended; tests must cite the applicable E78 obligations. Record exact commands, failures/skips and workload manifests in TEST_READY.

**Exit:** No unresolved correctness/lifecycle failures in the declared workload. Operational measurements and known limits are published. No unmeasured fixed latency guarantee is required. Introducing the missing compiler, schema or authored wiki here is a failed 7.8 handoff, not new 8.1 scope.

## Epic 8.2 — Held-out quality and cost evaluation

**Entry:** 8.1 passes; freeze the tested production commit/configuration for evaluation.\
**Owners:** `eval/` for dataset adapters, runner, scoring adapters and reports; reuse the 7.8 pilot support and production structured retrieval API. `eval/run_benchmark.py` is the benchmark entry point.\
**Production boundary:** Evaluation readers, judges, ablations and the lexical-plus-vector baseline exist only in `eval/`; they cannot become runtime fallback paths.

**Deliver in order:** (1) corpus/dataset adapters with fixed passage IDs and disjoint question/label manifests; (2) the production retriever and one evaluation-only baseline adapter; (3) pinned official/upstream scoring and per-question reports; (4) calibration, configuration freeze and held-out runs. Add `eval/manifests/epic82.json` as a **PLANNED** versioned experiment manifest. It records all inputs required below and separate calibration/held-out IDs.

The runner's planned command contract is `uv run --locked python eval/run_benchmark.py --manifest eval/manifests/epic82.json --phase calibration`, followed by `--phase held-out` only after the configuration/scorer/model hashes are frozen. The held-out phase rejects changed frozen inputs. Save raw predictions, normalized scorer inputs, per-question scores, paired baseline results, failures and cost metadata; an aggregate score alone is not a deliverable. Missing scorer/reader/data or failed required observations return nonzero with a named incomplete comparison.

### A. Corpus-specific acceptance

Use independent labels for representative 100–500-document corpora, mostly English with a declared non-English/mixed-language subset. Include unfamiliar concepts, homonyms, qualification, contradictory sources, explicit supersession, changed/deleted evidence and questions whose support spans documents. Record exact cases and expected supporting/opposing source IDs before evaluation.

Report extraction coverage/precision, false merges, direction/qualification/grounding errors, comparison classification errors, unsupported synthesis statements, support/counterevidence retrieval and update/retraction correctness. Structural and lifecycle assertions remain deterministic tests; model quality has its own report. A SHACL pass rate or empty extraction cannot stand in for these measures.

Compare the production engine against one lexical-plus-vector baseline with the same corpus, encoder, questions, context budget and evaluation reader. Report paired per-question results and uncertainty, rather than only an average gain. The baseline is an experiment, never a production outage mode. A quality advantage is claimed only if these results establish it; otherwise document the result and amend the chosen design before making such a claim.

### B. Requested external comparison protocol

1. Pin dataset revisions, licenses, splits, passage IDs, corpus boundaries, sample manifests/seeds and file hashes for **2WikiMultiHopQA and MuSiQue**. Include **HotpotQA** when claiming the requested three-dataset comparison. Load corpus passages into isolated benchmark vaults; questions/answers/judge labels never enter compilation.
2. Use the production structured ranked-passage list directly. Keep related-evidence expansions and generated summaries separately identified so retrieval Recall@k has a stable passage meaning. Do not parse rendered Markdown or count synthetic pages as original benchmark gold passages.
3. Implement one adapter to the pinned [HippoRAG retrieval scorer](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/evaluation/retrieval_eval.py) for **Recall@2 and Recall@5**. Report corpus/passage mapping and missing-gold handling. Do not maintain another near-identical recall implementation.
4. Implement the requested [MoG](https://github.com/DEEP-PolyU/MoG#main-results-deepseek-v3) comparison using its pinned reader/judge, prompts, answer normalization, context budget and iterative/non-iterative protocol for **LLM-Acc / Match-Acc**. Inspect and reuse the upstream scorer; the names alone do not define the metrics. If an upstream component is unavailable, report that comparison blocked rather than substitute a homemade metric under the same name. Preserve the benchmark passage boundaries reflected in [MoG's configuration](https://github.com/DEEP-PolyU/MoG/blob/main/config/mog_config.yaml).
5. Report dataset-official EM/F1 and supporting-fact measures only through their pinned official scorer, under distinct names. They are additional metrics, not aliases for MoG accuracy or retrieval recall. A different reader/model or passage protocol must be labelled as a different experiment, not a controlled reproduction.

### C. Calibration and ablations

Partition calibration and held-out IDs once, using a committed seed. Where official labels require splitting a development set, commit that partition before tuning. Freeze configuration, prompts, metric code, context limits and model digests before the held-out run. No held-out result feeds back into tuning; a later model/configuration revision requires a newly declared evaluation protocol.

Run offline ablations for fact/alias/FTS channels, candidate links, organization, AF enrichment and stored synthesis. These experiments alter explicit runner inputs; there is one selected production profile. Record reader/judge calls separately from compilation/query calls, along with cost, tokens, memory and latency.

**Exit:** Reproducible per-question results, calibrated configuration, supported quality claims, unchanged deterministic correctness gates and explicit limits. All required external comparisons are completed or the epic stays open; an unavailable scorer is not silently replaced. A failing quality goal returns work to an explicit design amendment and implementation, not a production fallback.

## Epic 8.3 — Packaging, CI and release

**Entry:** 8.2 results and all earlier runtime gates are recorded.\
**Owners:** project/lock metadata, CLI entry point, CI configuration, README and TEST_READY.

1. Package the verified entry points and semantic resources. Resolve the current `package = false` source-checkout configuration deliberately. Build a wheel (`uv build --wheel`) and add **NEW** `tests/test_distribution.py` to install that wheel into a fresh environment, launch `yaw` from an empty working directory with repository `PYTHONPATH` absent, and verify bundled resources/migration/end-to-end behavior.
2. CI runs the declared Python/dependency matrix on native Windows and Unix. The initial Python target is 3.14; broader version support requires actual dependency/install/test evidence. Do not copy the former 3.10–3.13 matrix or an unverified “3.14+” promise.
3. Run [TEST_INFRA completion commands](../TEST_INFRA.md#completion-commands), integration/migration and distribution checks against the release candidate. Separate model-dependent evaluation from deterministic CI while retaining its pinned report and explicit status.
4. Verify the package contains one selected pipeline, the intended versioned resources and no dormant legacy feature flags. Publish setup steps, error states, measured workloads, quality limits and migration instructions.
5. Prepare release artifacts and notes. Tag/publish only under the repository's release authorization; writing this roadmap is not publication approval.

**Exit:** Reproducible install and verified release artifacts backed by current evidence. No historical COMPLETE label or old test total can substitute for any entry/exit gate.
