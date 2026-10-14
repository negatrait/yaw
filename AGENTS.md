# Instructions for coding agents

Read [FOUNDATIONAL.md](FOUNDATIONAL.md) first. YAW is a small HippoRAG integration; the retired YAW architecture is not active scope.

- Verify the user-designated checkout and its Git remote before writing. Current repository: `C:\Users\Harri.Ahokas\Documents\yaw`.
- Execute [Epic 1](docs/epic-1-hipporag-daemon.md) before [Epic 2](docs/epic-2-fastmcp-http.md). Select the first unfinished packet and its observable exit condition. D1-01 has a passing adapter/smoke regression; its remaining work includes multilingual/chunk-boundary validation. Apply Epic 1's model-compatibility gate before admitting additional chat/embedding combinations or closing the development epics.
- Keep the pinned HippoRAG public API as the engine. Use the existing small injected embedding adapter; no global monkeypatch or copied upstream retrieval/service implementation.
- Implement one runtime and corpus writer. MCP is an adapter to it. Follow the source-ownership, serialization and dirty-state contracts; do not introduce alternate model/backend paths.
- Preserve source files and user experiments. Put generated indexes under ignored outputs or disposable temporary directories. Never treat a partial directory scan as evidence of source deletion.
- Use `uv sync --locked` and `uv run --locked`. Add only the dependencies assigned to the current epic, updating the lock deliberately. Milvus and the upstream MCP extra are not part of this plan.
- Record actual checks and failures in [evaluation](docs/evaluation.md). Distinguish package resolution, installation, real inference, retrieval correctness and daemon/transport acceptance. A fake provider validates orchestration only.
- Keep README's available commands separate from epic target commands. Update scope documents when accepted decisions change. Run focused verification and `git diff --check` before reporting completion.
