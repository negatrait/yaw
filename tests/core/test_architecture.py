"""AST-enforced architecture containment and zero-cross-domain import boundary tests."""

import ast
import re
import sys
import unittest
from pathlib import Path, PurePosixPath

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAW_DIR = PROJECT_ROOT / "yaw"
DOCS_DIR = PROJECT_ROOT / "docs"
RULES_DIR = PROJECT_ROOT / ".agents" / "rules"
AGENTS_DIR = PROJECT_ROOT / ".agents" / "agents"

_VALID_SUBAGENT_TOOLS = frozenset(
    {
        "send_message",
        "read_file",
        "search_web",
        "schedule",
        "run_command",
        "manage_task",
        "write_file",
        "edit_file",
    }
)
_CANONICAL_DOCS_INVENTORY = frozenset(
    {
        "arxiv-2405.14831v3-hipporag.md",
        "arxiv-2502.14802v1-hipporag2.md",
        "coverage-plan.md",
        "dspy-improvements.md",
        "enterprise-rag-bench-training-plan.md",
        "epic-1-hipporag-daemon.md",
        "epic-2-fastmcp-http.md",
        "epistemic-confidence-spec.md",
        "evaluation-graph-storage.md",
        "evaluation-inference.md",
        "evaluation-ingestion.md",
        "evaluation-testing.md",
        "evaluation.md",
        "model-evaluation.md",
        "native-cocoindex-indexing.md",
        "pre-epic2-memo.md",
        "w3c-improvements.md",
    }
)
_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_WIKI_LINK_RE = re.compile(r"\[\[[^\]]+\]\]")


def _parse_yaml_frontmatter(md_path: Path) -> tuple[dict[str, object], str]:
    """Parse YAML frontmatter delimited by --- at the top of a Markdown file."""
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AssertionError(f"{md_path.relative_to(PROJECT_ROOT)} is missing opening YAML frontmatter '---'")
    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        raise AssertionError(f"{md_path.relative_to(PROJECT_ROOT)} is missing closing YAML frontmatter '---'")
    raw_yaml = text[4:end_idx]
    body = text[end_idx + 5 :]
    loaded = yaml.safe_load(raw_yaml)
    if not isinstance(loaded, dict):
        raise AssertionError(f"{md_path.relative_to(PROJECT_ROOT)} frontmatter did not parse to a mapping")
    return loaded, body


def _strip_fenced_code_blocks(markdown_text: str) -> str:
    """Remove fenced code blocks and inline code spans before scanning prose links."""
    lines: list[str] = []
    in_fence = False
    for line in markdown_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(lines)


def parse_module_imports(file_path: Path) -> list[tuple[int, str]]:
    """Parse all import statements in a python file and return (lineno, imported_module)."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    imports: list[tuple[int, str]] = []

    # Calculate package name for relative imports
    rel_path = file_path.relative_to(PROJECT_ROOT)
    parts = list(rel_path.with_suffix("").parts)
    # If the file is __init__.py, its package is parts[:-1], otherwise parts[:-1]
    current_pkg_parts = parts[:-1] if file_path.name == "__init__.py" else parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import: resolve against current_pkg_parts
                # level=1 means from . import X (current package)
                # level=2 means from .. import X (parent package)
                drop = node.level - 1
                base_parts = [] if drop > len(current_pkg_parts) else current_pkg_parts[: len(current_pkg_parts) - drop]
                mod = ".".join(base_parts)
                full_mod = (f"{mod}.{node.module}" if mod else node.module) if node.module else mod
                imports.append((node.lineno, full_mod))
            elif node.module:
                imports.append((node.lineno, node.module))

    return imports


def get_canonical_package(module_path: Path) -> str:
    """Return the dotted canonical package name for a given file inside yaw/."""
    rel = module_path.relative_to(PROJECT_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        return ".".join(parts[:-1])
    return ".".join(parts[:-1])


def is_allowed_internal_import(importer_pkg: str, imported_mod: str) -> bool:
    """Validate containment law: package P imports only P.* (descendants) or yaw.core.*."""
    if not imported_mod.startswith("yaw.") and imported_mod != "yaw":
        # External library or stdlib
        return True

    # Rule: yaw.core.* is globally accessible to all yaw modules
    if imported_mod == "yaw.core" or imported_mod.startswith("yaw.core."):
        return True

    # Rule: yaw/core/* cannot import any other yaw package
    if importer_pkg == "yaw.core" or importer_pkg.startswith("yaw.core."):
        return False

    # Rule: Top-level __main__.py can import daemon.cli
    if importer_pkg == "yaw" and (imported_mod == "yaw.daemon.cli" or imported_mod.startswith("yaw.daemon.cli.")):
        return True

    # Rule: daemon.runtime is the daemon orchestrator and can import direct children (ingestion, graph)
    if importer_pkg == "yaw.daemon" and imported_mod.startswith("yaw.daemon."):
        # runtime.py or cli.py or mcp_server.py
        return True

    # Rule: graph.engine is the graph orchestrator and can import direct children (storage, inference, ppr)
    if importer_pkg == "yaw.daemon.graph" and imported_mod.startswith("yaw.daemon.graph."):
        return True

    # General Rule: package P imports only P.* (itself or its descendants)
    return bool(imported_mod == importer_pkg or imported_mod.startswith(f"{importer_pkg}."))


def is_leaf_package(package_dir: Path) -> bool:
    """Return True if package_dir contains no subdirectories with an __init__.py file."""
    return all(not (child.is_dir() and (child / "__init__.py").exists()) for child in package_dir.iterdir())


def build_internal_import_graph() -> dict[str, set[str]]:
    """Build the directed module-level import graph for all Python modules in yaw/."""
    module_map: dict[str, Path] = {}
    for p in YAW_DIR.rglob("*.py"):
        rel = p.relative_to(PROJECT_ROOT).with_suffix("")
        parts = list(rel.parts)
        mod_name = ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)
        module_map[mod_name] = p

    graph: dict[str, set[str]] = {m: set() for m in module_map}

    for mod, file_path in module_map.items():
        rel = file_path.relative_to(PROJECT_ROOT).with_suffix("")
        parts = list(rel.parts)
        pkg_parts = parts[:-1]
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("yaw.") or name == "yaw":
                        cand = name
                        while cand and cand not in module_map:
                            cand = cand.rsplit(".", 1)[0] if "." in cand else None
                        if cand and cand != mod:
                            graph[mod].add(cand)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    drop = node.level - 1
                    base = pkg_parts[: len(pkg_parts) - drop] if drop <= len(pkg_parts) else []
                    m_str = ".".join(base)
                    full = f"{m_str}.{node.module}" if node.module else m_str
                    cand = full
                    while cand and cand not in module_map:
                        cand = cand.rsplit(".", 1)[0] if "." in cand else None
                    if cand and cand != mod:
                        graph[mod].add(cand)
                    for alias in node.names:
                        alias_full = f"{full}.{alias.name}" if full else alias.name
                        if alias_full in module_map and alias_full != mod:
                            graph[mod].add(alias_full)
                elif node.module:
                    if node.module.startswith("yaw.") or node.module == "yaw":
                        cand = node.module
                        while cand and cand not in module_map:
                            cand = cand.rsplit(".", 1)[0] if "." in cand else None
                        if cand and cand != mod:
                            graph[mod].add(cand)
                        for alias in node.names:
                            alias_full = f"{node.module}.{alias.name}"
                            if alias_full in module_map and alias_full != mod:
                                graph[mod].add(alias_full)

    return graph


def find_import_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Detect cycles in a directed graph using DFS back-edge detection. Returns all detected cycle paths."""
    visited: dict[str, int] = {}  # 0: unvisited, 1: visiting (recursion stack), 2: visited
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        visited[node] = 1
        path.append(node)
        for neighbor in sorted(graph.get(node, set())):
            if visited.get(neighbor, 0) == 1:
                idx = path.index(neighbor)
                cycles.append(path[idx:] + [neighbor])
            elif visited.get(neighbor, 0) == 0:
                dfs(neighbor, path)
        path.pop()
        visited[node] = 2

    for node in sorted(graph):
        if visited.get(node, 0) == 0:
            dfs(node, [])

    return cycles


class ArchitectureTests(unittest.TestCase):
    def test_all_init_files_are_strictly_empty(self):
        """Rule 1: Intermediate package __init__.py files in yaw/ are strictly empty (0 bytes or comments only).

        Leaf packages (e.g. yaw/core, yaw/daemon/graph/inference) may define public API facades
        and re-exports, while intermediate parent packages (yaw/, yaw/daemon/, yaw/daemon/graph/)
        must remain strictly empty to prevent parent-level side-effects or horizontal coupling.
        """
        init_files = list(YAW_DIR.rglob("__init__.py"))
        self.assertGreater(len(init_files), 0, "No __init__.py files found in yaw/")

        intermediate_inits = [f for f in init_files if not is_leaf_package(f.parent)]
        self.assertGreater(len(intermediate_inits), 0, "No intermediate packages found in yaw/")

        for init_file in intermediate_inits:
            content = init_file.read_text(encoding="utf-8").strip()
            # Must not contain any non-comment executable code
            lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
            self.assertEqual(
                lines,
                [],
                f"Non-empty intermediate __init__.py with executable statements "
                f"found at: {init_file.relative_to(PROJECT_ROOT)}",
            )

    def test_leaf_package_facades_allow_contract_import_without_knowing_file(self):
        """Rule 1b: Leaf packages export their public API contract for caller convenience and leaf mutation."""
        from yaw.core import (
            ChunkPointer,
            DaemonStatus,
            IndexStats,
            YawConfig,
            YawRuntimeError,
            load_config,
        )

        self.assertTrue(issubclass(ChunkPointer, object))
        self.assertTrue(issubclass(YawConfig, object))
        self.assertTrue(issubclass(IndexStats, object))
        self.assertTrue(issubclass(DaemonStatus, object))
        self.assertTrue(issubclass(YawRuntimeError, RuntimeError))
        self.assertTrue(callable(load_config))

        import yaw.daemon.graph.inference as inference

        self.assertTrue(hasattr(inference, "create_dspy_lm"))
        self.assertTrue(hasattr(inference, "HippoRAGProgram"))
        self.assertTrue(hasattr(inference, "DSPyEmbeddingAdapter"))
        self.assertTrue(hasattr(inference, "compute_task_quality_score"))
        self.assertTrue(hasattr(inference, "check_benchmark_integrity"))
        self.assertTrue(hasattr(inference, "GepaProgress"))
        self.assertTrue(hasattr(inference, "GraphOptimizationHooks"))
        self.assertTrue(hasattr(inference, "ActiveTraceComponentSelector"))
        self.assertTrue(hasattr(inference, "configure_simba_task_learning"))
        self.assertTrue(hasattr(inference, "evolve_program_prompts"))
        self.assertTrue(hasattr(inference, "evaluate_program"))
        self.assertFalse(hasattr(inference, "RetrievalEmbedding"))
        self.assertFalse(hasattr(inference, "LocalChat"))
        self.assertFalse(hasattr(inference, "CacheOpenAI"))
        self.assertFalse(hasattr(inference, "chat_identity"))

        # Graph orchestration and deterministic transforms remain in their owning domain.
        graph_files = {f.name for f in (YAW_DIR / "daemon" / "graph").glob("*.py")}
        self.assertEqual(graph_files, {"__init__.py", "engine.py", "ppr.py", "normalization.py", "planner.py"})

        # Verify storage public facade
        import yaw.daemon.graph.storage as storage

        self.assertTrue(hasattr(storage, "open_yaw_db"))
        self.assertTrue(hasattr(storage, "search_chunks_fts"))
        self.assertTrue(hasattr(storage, "search_chunk_vectors_knn"))
        self.assertTrue(hasattr(storage, "hydrate_chunk_pointer"))
        self.assertTrue(hasattr(storage, "make_chunk_hydrator"))
        self.assertTrue(hasattr(storage, "get_graph_counts"))
        self.assertTrue(hasattr(storage, "get_meta"))
        self.assertTrue(hasattr(storage, "set_meta"))

        with self.assertRaises(AttributeError):
            _ = storage._CREATE_SCHEMA_SQL

        # Verify ingestion public facade
        import yaw.daemon.ingestion as ingestion

        self.assertTrue(hasattr(ingestion, "declare_source_file"))
        self.assertTrue(hasattr(ingestion, "chunk_source_text"))
        self.assertTrue(hasattr(ingestion, "SourceMetadata"))
        self.assertTrue(hasattr(ingestion, "extract_chunk_title"))
        self.assertFalse(hasattr(ingestion, "chunk_markdown_text"))
        self.assertFalse(hasattr(ingestion, "chunk_json_text"))
        self.assertFalse(hasattr(ingestion, "CorpusIngestor"))
        self.assertFalse(hasattr(ingestion, "sync_corpus_once"))

        with self.assertRaises(AttributeError):
            _ = ingestion._CHUNK_HANDLER
        with self.assertRaises(AttributeError):
            _ = ingestion._ChunkSpec

        # Verify facade is strictly constricted: internal helpers cannot leak through
        with self.assertRaises(AttributeError):
            _ = inference._nullable_text
        with self.assertRaises(AttributeError):
            _ = inference._build_default_ner_demos
        with self.assertRaises(AttributeError):
            _ = inference._extract_json_list_field
        with self.assertRaises(AttributeError):
            _ = inference._PositionalSafeRLM
        with self.assertRaises(AttributeError):
            _ = inference._LLMConfig

    def test_domain_containment_and_zero_cross_domain_imports(self):
        """Rule 2 & 3: Package P imports only P.* and yaw.core.* (zero horizontal cross-domain edges)."""
        py_files = list(YAW_DIR.rglob("*.py"))
        self.assertGreater(len(py_files), 0, "No .py files found in yaw/")

        violations = []
        for py_file in py_files:
            importer_pkg = get_canonical_package(py_file)
            imports = parse_module_imports(py_file)

            for lineno, imported_mod in imports:
                if (imported_mod.startswith("yaw.") or imported_mod == "yaw") and not is_allowed_internal_import(
                    importer_pkg, imported_mod
                ):
                    violations.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: "
                        f"package '{importer_pkg}' forbidden to import '{imported_mod}'"
                    )

        self.assertEqual(
            violations,
            [],
            "Architecture containment violations detected:\n" + "\n".join(violations),
        )

    def test_core_imports_stdlib_only(self):
        """Rule: yaw/core/* must strictly import standard library only (zero third-party, zero daemon)."""
        core_files = list((YAW_DIR / "core").glob("*.py"))
        self.assertEqual(
            {f.name for f in core_files},
            {"__init__.py", "config.py", "constants.py", "logger.py", "types.py"},
        )

        # Python supplies the authoritative set, including platform-specific modules.
        stdlib_whitelist = sys.stdlib_module_names

        violations = []
        for py_file in core_files:
            imports = parse_module_imports(py_file)
            for lineno, mod in imports:
                top_pkg = mod.split(".")[0]
                if top_pkg == "yaw":
                    # Only yaw.core is permitted inside yaw.core
                    if not (mod == "yaw.core" or mod.startswith("yaw.core.")):
                        violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: imports non-core '{mod}'")
                elif top_pkg not in stdlib_whitelist:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: imports non-stdlib '{mod}'")

        self.assertEqual(
            violations,
            [],
            "yaw/core non-stdlib imports detected:\n" + "\n".join(violations),
        )

    def test_synthetic_boundary_violation_detection(self):
        """Verify the test architecture checker properly rejects synthetic cross-domain and parent imports."""
        # storage importing inference (horizontal sibling edge) -> must be rejected
        self.assertFalse(is_allowed_internal_import("yaw.daemon.graph.storage", "yaw.daemon.graph.inference.models"))
        # inference importing storage (horizontal sibling edge) -> must be rejected
        self.assertFalse(is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.daemon.graph.storage.db"))
        # ingestion importing graph (horizontal sibling edge) -> must be rejected
        self.assertFalse(is_allowed_internal_import("yaw.daemon.ingestion", "yaw.daemon.graph.engine"))
        # inference importing daemon runtime (child importing parent) -> must be rejected
        self.assertFalse(is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.daemon.runtime"))
        # core importing daemon (core importing non-core) -> must be rejected
        self.assertFalse(is_allowed_internal_import("yaw.core", "yaw.daemon.graph.inference"))
        # inference importing core -> allowed
        self.assertTrue(is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.core.types"))
        # inference importing internal sibling -> allowed
        self.assertTrue(is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.daemon.graph.inference.models"))

    def test_zero_hipporag_imports_in_yaw(self):
        """Rule 4: Zero hipporag imports across the entire repository (yaw/, benchmark.py, scripts/, tests/)."""
        py_files = (
            list(YAW_DIR.rglob("*.py"))
            + [PROJECT_ROOT / "benchmark.py"]
            + list((PROJECT_ROOT / "scripts").rglob("*.py"))
            + list((PROJECT_ROOT / "tests").rglob("*.py"))
        )
        self.assertGreater(len(py_files), 0)

        violations = []
        for py_file in py_files:
            if not py_file.exists():
                continue
            imports = parse_module_imports(py_file)
            for lineno, mod in imports:
                top_pkg = mod.split(".")[0]
                if top_pkg == "hipporag":
                    violations.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: forbidden upstream import '{mod}'"
                    )

        self.assertEqual(
            violations,
            [],
            "Forbidden hipporag imports detected in repository:\n" + "\n".join(violations),
        )

    def test_zero_cross_module_private_imports_and_exports(self):
        """Rule 5: Zero protected (_) or private (__) symbols in public facades or cross-module imports."""
        # 1. Check leaf package facades
        init_files = list(YAW_DIR.rglob("__init__.py"))
        for init_file in init_files:
            source = init_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(init_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "_PUBLIC_API"
                            and isinstance(node.value, ast.Dict)
                        ):
                            for key in node.value.keys:
                                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                    self.assertFalse(
                                        key.value.startswith("_"),
                                        f"Public facade {init_file.relative_to(PROJECT_ROOT)} exports "
                                        f"protected/private symbol '{key.value}'",
                                    )

        # 2. Check all yaw/**/*.py files for cross-module private imports
        py_files = list(YAW_DIR.rglob("*.py"))
        violations: list[str] = []
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    is_internal = node.level > 0 or (
                        bool(node.module) and (node.module.startswith("yaw.") or node.module == "yaw")
                    )
                    if is_internal:
                        for alias in node.names:
                            if alias.name.startswith("_"):
                                violations.append(
                                    f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                                    f"cross-module import of protected/private symbol '{alias.name}' is forbidden"
                                )

        self.assertEqual(
            violations,
            [],
            "Protected/private cross-module imports detected:\n" + "\n".join(violations),
        )

    def test_zero_circular_dependencies_across_all_yaw_modules(self):
        """Rule 6: The global module import graph must form a strict DAG with zero circular dependencies."""
        graph = build_internal_import_graph()
        self.assertGreater(len(graph), 0, "No yaw modules detected for cycle analysis.")

        cycles = find_import_cycles(graph)
        cycle_strs = [" -> ".join(c) for c in cycles]
        self.assertEqual(
            cycle_strs,
            [],
            f"Detected {len(cycles)} circular import cycle(s) in yaw:\n" + "\n".join(cycle_strs),
        )

    def test_synthetic_cycle_detection(self):
        """Verify the cycle finder algorithm reliably identifies cycles when introduced."""
        synthetic_dag = {
            "a": {"b", "c"},
            "b": {"c"},
            "c": set(),
        }
        self.assertEqual(find_import_cycles(synthetic_dag), [])

        synthetic_cyclic = {
            "a": {"b"},
            "b": {"c"},
            "c": {"a"},
        }
        cycles = find_import_cycles(synthetic_cyclic)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0], ["a", "b", "c", "a"])

    def test_agent_rules_yaml_frontmatter_and_size_caps(self):
        """Rule 7: All .agents/rules/*.md files have valid YAML frontmatter, <4 KB size, and zero @[] includes."""
        rule_files = sorted(RULES_DIR.glob("*.md"))
        self.assertEqual(
            {f.name for f in rule_files},
            {
                "yaw.md",
                "daemon.md",
                "graph-storage.md",
                "inference.md",
                "tests.md",
                "canonical-files.md",
                "docs.md",
                "agent-rules.md",
            },
        )

        for rule_file in rule_files:
            rel = rule_file.relative_to(PROJECT_ROOT)
            size_bytes = len(rule_file.read_bytes())
            self.assertLess(size_bytes, 4096, f"Rule {rel} exceeds 4 KB cap ({size_bytes} bytes)")

            meta, body = _parse_yaml_frontmatter(rule_file)
            self.assertEqual(meta.get("trigger"), "glob", f"Rule {rel} must set trigger: glob")
            globs_val = meta.get("globs")
            self.assertIsInstance(globs_val, str, f"Rule {rel} globs must be a YAML string")
            assert isinstance(globs_val, str)
            patterns = [p.strip() for p in globs_val.split(",") if p.strip()]
            self.assertGreater(len(patterns), 0, f"Rule {rel} must declare at least one glob pattern")

            desc = meta.get("description")
            self.assertIsInstance(desc, str, f"Rule {rel} must declare a description string")
            assert isinstance(desc, str)
            word_count = len(desc.split())
            self.assertTrue(1 <= word_count <= 60, f"Rule {rel} description must be 1..60 words (got {word_count})")
            self.assertNotIn("@[", body, f"Rule {rel} must use pointer links [label](path), never @[label](path)")

    def test_custom_subagents_yaml_frontmatter_and_tool_validity(self):
        """Rule 8: Custom subagents in .agents/agents/*/agent.md mount visibly with valid tools and current stack."""
        expected_agents = {"critic", "simplifier", "ontologist"}
        agent_files = {p.parent.name: p for p in AGENTS_DIR.glob("*/agent.md")}
        self.assertEqual(set(agent_files.keys()), expected_agents)

        retired_markers = ("sample-llm-wiki", "pyshacl.validate", "lean-ctx", "audit_report.md")
        for agent_name, agent_path in sorted(agent_files.items()):
            rel = agent_path.relative_to(PROJECT_ROOT)
            meta, body = _parse_yaml_frontmatter(agent_path)
            self.assertEqual(meta.get("name"), agent_name, f"Subagent {rel} name mismatch")
            self.assertIs(meta.get("hidden"), False, f"Subagent {rel} must set hidden: false to mount in <subagents>")
            self.assertIs(
                meta.get("inheritCustomizations"),
                True,
                f"Subagent {rel} must set inheritCustomizations: true",
            )

            desc = meta.get("description")
            self.assertIsInstance(desc, str, f"Subagent {rel} must declare a description string")
            assert isinstance(desc, str)
            word_count = len(desc.split())
            self.assertTrue(1 <= word_count <= 60, f"Subagent {rel} description must be 1..60 words (got {word_count})")

            tools = meta.get("tools")
            self.assertIsInstance(tools, list, f"Subagent {rel} tools must be a YAML list")
            assert isinstance(tools, list)
            unknown_tools = set(tools) - _VALID_SUBAGENT_TOOLS
            self.assertEqual(unknown_tools, set(), f"Subagent {rel} declares invalid tools: {sorted(unknown_tools)}")
            if agent_name == "critic":
                self.assertTrue(
                    {"write_file", "edit_file"}.isdisjoint(tools),
                    "Critic subagent must remain read-only (no write_file or edit_file)",
                )

            for marker in retired_markers:
                self.assertNotIn(marker, body, f"Subagent {rel} contains retired reference '{marker}'")

    def test_every_repo_target_file_matches_at_least_one_rule_glob(self):
        """Rule 9: Every repo target file matches its owning .agents/rules glob without unintended overlap."""
        rule_globs_by_name: dict[str, list[str]] = {}
        for rule_file in sorted(RULES_DIR.glob("*.md")):
            meta, _ = _parse_yaml_frontmatter(rule_file)
            globs_val = meta.get("globs")
            assert isinstance(globs_val, str)
            rule_globs_by_name[rule_file.name] = [p.strip() for p in globs_val.split(",") if p.strip()]

        target_files = (
            list(YAW_DIR.rglob("*.py"))
            + list((PROJECT_ROOT / "tests").rglob("*.py"))
            + [PROJECT_ROOT / "benchmark.py"]
            + list((PROJECT_ROOT / "scripts").rglob("*.py"))
            + list(DOCS_DIR.rglob("*.md"))
            + list(RULES_DIR.glob("*.md"))
            + list(AGENTS_DIR.rglob("*.md"))
            + [PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "FOUNDATIONAL.md", PROJECT_ROOT / "README.md"]
        )
        self.assertGreater(len(target_files), 0)

        unmatched: list[str] = []
        overlapping: list[str] = []
        for file_path in target_files:
            rel_posix = PurePosixPath(file_path.relative_to(PROJECT_ROOT).as_posix())
            matched_rules = [
                rule_name
                for rule_name, patterns in rule_globs_by_name.items()
                if any(rel_posix.full_match(pattern) for pattern in patterns)
            ]
            if not matched_rules:
                unmatched.append(str(rel_posix))
            elif len(matched_rules) > 1 and str(rel_posix) != "tests/core/test_architecture.py":
                overlapping.append(f"{rel_posix}: matched multiple rules {matched_rules}")

        self.assertEqual(
            unmatched,
            [],
            "Repository files not covered by any .agents/rules/*.md glob:\n" + "\n".join(unmatched),
        )
        self.assertEqual(
            overlapping,
            [],
            "Repository files matched overlapping .agents/rules/*.md globs:\n" + "\n".join(overlapping),
        )

    def test_zero_broken_markdown_links_and_zero_wiki_links(self):
        """Rule 10: Zero [[wiki-links]] and zero broken relative Markdown links across rules, agents, and docs."""
        non_arxiv_docs = [p for p in sorted(DOCS_DIR.glob("*.md")) if not p.name.startswith("arxiv-")]
        checked_files = (
            [PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "FOUNDATIONAL.md", PROJECT_ROOT / "README.md"]
            + sorted(RULES_DIR.glob("*.md"))
            + sorted(AGENTS_DIR.glob("*/agent.md"))
            + non_arxiv_docs
        )

        wiki_violations: list[str] = []
        broken_links: list[str] = []
        for md_file in checked_files:
            rel_source = md_file.relative_to(PROJECT_ROOT)
            prose = _strip_fenced_code_blocks(md_file.read_text(encoding="utf-8"))
            for match in _WIKI_LINK_RE.finditer(prose):
                wiki_violations.append(f"{rel_source}: forbidden wiki-link {match.group(0)}")

            for match in _MD_LINK_RE.finditer(prose):
                raw_target = match.group(1).strip()
                if not raw_target or raw_target.startswith(
                    ("http://", "https://", "mailto:", "file://", "conversation://", "#")
                ):
                    continue
                path_part = raw_target.split("#", 1)[0].strip()
                if not path_part:
                    continue
                resolved = (md_file.parent / path_part).resolve()
                if not resolved.exists():
                    broken_links.append(f"{rel_source}: broken relative link '{raw_target}' -> {resolved}")

        self.assertEqual(wiki_violations, [], "Forbidden [[wiki-links]] found:\n" + "\n".join(wiki_violations))
        self.assertEqual(broken_links, [], "Broken relative Markdown links found:\n" + "\n".join(broken_links))

    def test_every_doc_is_reachable_from_rules_and_canonical_inventory(self):
        """Rule 11: docs/ matches the canonical 15-file inventory and every doc is reachable from .agents/rules/*.md."""
        actual_docs = {p.name for p in DOCS_DIR.glob("*.md")}
        self.assertEqual(
            actual_docs,
            _CANONICAL_DOCS_INVENTORY,
            f"docs/ inventory drift. Extra: {sorted(actual_docs - _CANONICAL_DOCS_INVENTORY)}, "
            f"Missing: {sorted(_CANONICAL_DOCS_INVENTORY - actual_docs)}",
        )

        domain_rules = ("daemon.md", "graph-storage.md", "inference.md", "tests.md")
        linked_docs: set[str] = set()
        for rule_name in domain_rules:
            rule_file = RULES_DIR / rule_name
            _, body = _parse_yaml_frontmatter(rule_file)
            for match in _MD_LINK_RE.finditer(body):
                raw_target = match.group(1).split("#", 1)[0].strip()
                if not raw_target or raw_target.startswith(("http://", "https://", "file://", "conversation://")):
                    continue
                resolved = (rule_file.parent / raw_target).resolve()
                if resolved.parent == DOCS_DIR.resolve() and resolved.suffix == ".md":
                    linked_docs.add(resolved.name)

        unreachable = sorted(_CANONICAL_DOCS_INVENTORY - linked_docs)
        self.assertEqual(
            unreachable,
            [],
            f"docs/*.md files not reachable from domain rules {domain_rules}: {unreachable}",
        )

    def test_canonical_module_purity_and_behavioral_contracts(self):
        """Rule 12: Canonical Category A modules preserve strict AST import purity and behavioral boundaries."""
        stdlib = sys.stdlib_module_names

        # 1. ppr.py: pure numpy/scipy/yaw.core sparse math operator (zero sqlite3, dspy, cocoindex, or daemon imports)
        ppr_path = YAW_DIR / "daemon" / "graph" / "ppr.py"
        for lineno, mod in parse_module_imports(ppr_path):
            top = mod.split(".")[0]
            self.assertIn(
                top,
                stdlib | {"numpy", "scipy", "yaw"},
                f"ppr.py:{lineno} imports forbidden package '{mod}'",
            )
            self.assertNotEqual(top, "sqlite3", f"ppr.py:{lineno} must not import sqlite3")
            if top == "yaw":
                self.assertTrue(
                    mod == "yaw.core" or mod.startswith("yaw.core."),
                    f"ppr.py:{lineno} must only import yaw.core, got '{mod}'",
                )

        # 2. normalization.py: stateless OpenIE normalization (zero sqlite3, dspy, cocoindex, network, or storage)
        norm_path = YAW_DIR / "daemon" / "graph" / "normalization.py"
        forbidden_norm = {"sqlite3", "dspy", "cocoindex", "httpx", "openai", "urllib", "http", "socket"}
        for lineno, mod in parse_module_imports(norm_path):
            top = mod.split(".")[0]
            self.assertNotIn(top, forbidden_norm, f"normalization.py:{lineno} imports forbidden module '{mod}'")
            self.assertFalse(
                mod.startswith("yaw.daemon.graph.storage"),
                f"normalization.py:{lineno} must not import storage '{mod}'",
            )

        # 3. Storage leaf modules: zero network/HTTP imports anywhere in storage/*.py,
        #    and zero dspy imports in canonical graph storage modules
        storage_dir = YAW_DIR / "daemon" / "graph" / "storage"
        forbidden_storage_net = {"openai", "httpx", "urllib", "http", "socket"}
        for storage_file in sorted(storage_dir.glob("*.py")):
            rel = storage_file.relative_to(PROJECT_ROOT)
            for lineno, mod in parse_module_imports(storage_file):
                top = mod.split(".")[0]
                self.assertNotIn(top, forbidden_storage_net, f"{rel}:{lineno} imports forbidden module '{mod}'")

        canonical_storage_files = ("db.py", "vectors.py", "graph_ops.py", "publication.py", "hydration.py", "shacl.py")
        for storage_name in canonical_storage_files:
            storage_file = storage_dir / storage_name
            for lineno, mod in parse_module_imports(storage_file):
                top = mod.split(".")[0]
                self.assertNotEqual(top, "dspy", f"{storage_name}:{lineno} must not import dspy ('{mod}')")

        # 3b. vectors.py: pure sqlite3 + numpy + .db vector/FTS primitives
        vectors_path = storage_dir / "vectors.py"
        for lineno, mod in parse_module_imports(vectors_path):
            top = mod.split(".")[0]
            self.assertIn(
                top,
                stdlib | {"numpy", "yaw"},
                f"vectors.py:{lineno} imports unexpected package '{mod}'",
            )
            if top == "yaw":
                self.assertEqual(mod, "yaw.daemon.graph.storage.db", f"vectors.py:{lineno} unexpected import '{mod}'")

        # 3c. shacl.py: pure in-process SQLite SHACL & Turtle serializer (stdlib + yaw.core + storage siblings only)
        shacl_path = storage_dir / "shacl.py"
        for lineno, mod in parse_module_imports(shacl_path):
            top = mod.split(".")[0]
            self.assertIn(top, stdlib | {"yaw"}, f"shacl.py:{lineno} imports unexpected package '{mod}'")

        # 3d. publication.py: apply_prepared_graph_rows must never call .commit()
        pub_path = storage_dir / "publication.py"
        pub_tree = ast.parse(pub_path.read_text(encoding="utf-8"), filename=str(pub_path))
        for node in ast.walk(pub_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(
                    node.func.attr,
                    "commit",
                    f"publication.py:{node.lineno} must not call .commit() directly",
                )

        # 4. adapter.py & retriever.py: thin DSPy bridges (zero sqlite3, cocoindex, or raw HTTP imports;
        #    YawGBNFAdapter never overrides JSONAdapter.parse or JSONAdapter.format)
        inference_dir = YAW_DIR / "daemon" / "graph" / "inference"
        forbidden_bridge = {"sqlite3", "cocoindex", "httpx", "openai", "urllib", "http", "socket"}
        for bridge_name in ("adapter.py", "retriever.py"):
            bridge_path = inference_dir / bridge_name
            for lineno, mod in parse_module_imports(bridge_path):
                top = mod.split(".")[0]
                self.assertNotIn(top, forbidden_bridge, f"{bridge_name}:{lineno} imports forbidden module '{mod}'")

        adapter_path = inference_dir / "adapter.py"
        adapter_tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
        adapter_classes = [n for n in adapter_tree.body if isinstance(n, ast.ClassDef) and n.name == "YawGBNFAdapter"]
        self.assertEqual(len(adapter_classes), 1, "YawGBNFAdapter class not found in adapter.py")
        adapter_methods = {
            n.name for n in adapter_classes[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(
            adapter_methods,
            {"__init__", "resolve_grammar", "_prepare_gbnf_kwargs", "__call__", "acall"},
            f"YawGBNFAdapter must not override parse/format or add recovery methods: {sorted(adapter_methods)}",
        )
