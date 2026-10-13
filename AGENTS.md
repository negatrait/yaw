# Instructions for coding agents

Read [FOUNDATIONAL.md](FOUNDATIONAL.md) first. The current task is evaluating upstream HippoRAG 2, not rebuilding the retired YAW architecture.

- Verify the repository path and Git remote before editing. The user-designated checkout is authoritative.
- Follow the ordered evaluation steps in FOUNDATIONAL. State which step is being executed and its observable result.
- Keep HippoRAG as the only direct dependency. Use the pinned public API; do not add a framework, fork, wrapper hierarchy or semantic engine without a demonstrated need and a new scope decision.
- Use `uv sync --locked` and `uv run --locked`. Preserve the pinned source revision unless intentionally upgrading and validating it.
- Keep source notes and user data intact. Put generated indexes in ignored `outputs/`.
- Report installation checks separately from actual model evaluations. Preserve errors and retrieved evidence; do not invent results or silently switch models/backends.
- Update README and FOUNDATIONAL when an accepted decision changes the baseline. Historical Git versions of removed epics and ADRs are not active requirements.
- Run checks appropriate to the change and `git diff --check`. Do not create tests that merely restate configuration or documentation.
