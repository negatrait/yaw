# Instructions for Coding Agents

> [!NOTE] YAW is the Librarian, not a Chatbot or Memory Bank.
> Read [FOUNDATIONAL.md](FOUNDATIONAL.md) first—it is the Single Source of Truth (SSOT) for system purpose, 14-file boundaries, storage/retrieval/runtime invariants, and failure policy. Never duplicate its architecture rules into other docs.

**Precedence during active cutovers:** Active `docs/reviews/<sprint>.md` > `FOUNDATIONAL.md` > `docs/epic-*.md`. Verify branch `dev` and `git status` before editing.

## Prime Directive (Anti-Spaghetti Law)

**A minimal line count codebase that fails fast is infinitely better than a fallback-heavy codebase that silently degrades performance to mask failures!**

- **Zero Corporate Boilerplate:** Do **not** create manager/factory/helper classes, `ABC`/`Protocol` hierarchies, single-caller wrappers, compatibility shims, or speculative config knobs. Write flat functions, `frozen=True, slots=True` dataclasses, `stdlib` (`zlib`, `hashlib`, `graphlib`, `sqlite3`), and vectorized `NumPy`/`SciPy` kernels.
- **Exact Math Over Heuristics:** Never replace specified linear algebra (`svd`, `qr`, `nnls`), `RaBitQ` concentration bounds, or Shannon bit costs ($\log_2 N$, $\log_2 \text{df}$) with arbitrary scalar weights or stopword lists.
- **Zero Masked Failures:** Obey `FOUNDATIONAL.md §5` strictly—no custom retry loops, no broad `except`, and no degraded Python fallbacks when SQLite or model endpoints fail.
- **Atomic Deletions & DRY Docs:** Delete superseded code, config keys, and dead types in the same commit as their replacement, updating all callers and `scripts/watchdog.py` atomically. Keep documentation DRY—each invariant lives in **one** file.
- **Push Back With Proof:** Surface hidden assumptions, degenerate math cases (`N < 8`, `df == 0`, graph cycles, `NaN` norms), and contradictions with evidence before coding. Precision beats speed.

## Workflow, Host Processes & Review Discipline

- **Environment & Hygiene:** Python `>=3.14`; run `uv sync --locked` and `uv run --locked`. Write generated artifacts only to ignored `.yaw/`, `outputs/`, or temp dirs; never mutate corpus files. Log via `yaw/core/logger.py` (500-line FIFO).
- **Live Process Truth:** Editing code on disk does **not** reload resident `yaw daemon`, `optimize`, or `watchdog` processes, and the Windows sandbox hides host PIDs. Inspect host processes (`BypassSandbox: true` when supported) and verify fresh artifact timestamps and non-zero rows before claiming live state.
- **Review Ledger (`docs/reviews/<topic>.md`):** Read the active review record first and resume from its latest verified checkpoint. Append changed files, gate outputs, empirical findings, and explicit non-claims to that record in the same commit. Promote permanent contract changes into `FOUNDATIONAL.md` rather than duplicating prose.
- **Mandatory Critic Audit:** Before committing or ticking any checkpoint, dispatch `critic (Model="flash")` with the unguided prompt:
  `"Review the implementation checkpoint <ID>, defined in <spec>. Analyse the implementation in files <files> and audit the git diff. Report any findings."`
  Never pass leading checklists. (If running outside a harness with `critic`, audit `git diff` line-by-line against `FOUNDATIONAL.md` and this file before committing.)

## Mandatory Quality Gates & CLI

Run **all** gates before every commit (`-o addopts=""` bypasses `--testmon` so SQLite/schema/config changes are never skipped):

```bash
uv run ty check
uv run ruff check --fix
uv run ruff format
uv run pytest -o addopts=""
git diff --check
```

- `uv run yaw daemon {start|stop|status|run}`: Streamable HTTP FastMCP + live ingestion daemon.
- `uv run yaw recall "<query>" --top-k 5 [--json]`: standalone read-only Tier-0 zero-LLM recall.
- `uv run yaw qa "<question>" [--json]`: Tier-1 `dspy.ReAct` benchmark QA.
- `uv run yaw optimize dspy [--fresh]`: launch or resume `dspy.GEPA` QA prompt compilation.
- `uv run yaw config {show|get|set}`: inspect or update `yaw.toml` atomically (secrets redacted).
- `$env:YAW_ARTIFACT_DIR = "<dir>"; uv run watchdog --watch 300`: optimization telemetry watchdog.
