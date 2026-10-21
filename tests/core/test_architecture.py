"""AST-enforced architecture containment and zero-cross-domain import boundary tests."""

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAW_DIR = PROJECT_ROOT / "yaw"


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
                if drop > len(current_pkg_parts):
                    base_parts = []
                else:
                    base_parts = current_pkg_parts[: len(current_pkg_parts) - drop]
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
    if importer_pkg == "yaw" and (
        imported_mod == "yaw.daemon.cli" or imported_mod.startswith("yaw.daemon.cli.")
    ):
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
    for child in package_dir.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            return False
    return True


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
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            self.assertEqual(
                lines,
                [],
                f"Non-empty intermediate __init__.py with executable statements found at: {init_file.relative_to(PROJECT_ROOT)}",
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
        self.assertTrue(hasattr(inference, "compute_sizecoding_pareto_score"))
        self.assertFalse(hasattr(inference, "RetrievalEmbedding"))
        self.assertFalse(hasattr(inference, "LocalChat"))
        self.assertFalse(hasattr(inference, "CacheOpenAI"))
        self.assertFalse(hasattr(inference, "chat_identity"))

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

        self.assertTrue(hasattr(ingestion, "CorpusIngestor"))
        self.assertTrue(hasattr(ingestion, "chunk_markdown_text"))
        self.assertTrue(hasattr(ingestion, "sync_corpus_once"))
        self.assertTrue(hasattr(ingestion, "extract_chunk_title"))

        with self.assertRaises(AttributeError):
            _ = getattr(ingestion, "_CHUNK_HANDLER")
        with self.assertRaises(AttributeError):
            _ = getattr(ingestion, "_ChunkSpec")

        # Verify facade is strictly constricted: internal helpers cannot leak through
        with self.assertRaises(AttributeError):
            _ = getattr(inference, "_strip_mistral_bracket_prefix")
        with self.assertRaises(AttributeError):
            _ = getattr(inference, "_build_default_ner_demos")
        with self.assertRaises(AttributeError):
            _ = getattr(inference, "_extract_json_list_field")
        with self.assertRaises(AttributeError):
            _ = getattr(inference, "_PositionalSafeRLM")
        with self.assertRaises(AttributeError):
            _ = getattr(inference, "_LLMConfig")

    def test_domain_containment_and_zero_cross_domain_imports(self):
        """Rule 2 & 3: Package P imports only P.* and yaw.core.* (zero horizontal cross-domain edges)."""
        py_files = list(YAW_DIR.rglob("*.py"))
        self.assertGreater(len(py_files), 0, "No .py files found in yaw/")

        violations = []
        for py_file in py_files:
            importer_pkg = get_canonical_package(py_file)
            imports = parse_module_imports(py_file)

            for lineno, imported_mod in imports:
                if (
                    imported_mod.startswith("yaw.") or imported_mod == "yaw"
                ) and not is_allowed_internal_import(importer_pkg, imported_mod):
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
        self.assertGreater(len(core_files), 0)

        # Common known stdlib top-level modules
        stdlib_whitelist = {
            "ast",
            "collections",
            "contextlib",
            "dataclasses",
            "datetime",
            "functools",
            "hashlib",
            "itertools",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "queue",
            "re",
            "shutil",
            "stat",
            "string",
            "sys",
            "tempfile",
            "threading",
            "time",
            "tomllib",
            "types",
            "typing",
            "unittest",
            "urllib",
            "uuid",
            "weakref",
            "enum",
            "abc",
        }

        violations = []
        for py_file in core_files:
            imports = parse_module_imports(py_file)
            for lineno, mod in imports:
                top_pkg = mod.split(".")[0]
                if top_pkg == "yaw":
                    # Only yaw.core is permitted inside yaw.core
                    if not (mod == "yaw.core" or mod.startswith("yaw.core.")):
                        violations.append(
                            f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: imports non-core '{mod}'"
                        )
                elif top_pkg not in stdlib_whitelist:
                    violations.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: imports non-stdlib '{mod}'"
                    )

        self.assertEqual(
            violations,
            [],
            "yaw/core non-stdlib imports detected:\n" + "\n".join(violations),
        )

    def test_synthetic_boundary_violation_detection(self):
        """Verify the test architecture checker properly rejects synthetic cross-domain and parent imports."""
        # storage importing inference (horizontal sibling edge) -> must be rejected
        self.assertFalse(
            is_allowed_internal_import(
                "yaw.daemon.graph.storage", "yaw.daemon.graph.inference.models"
            )
        )
        # inference importing storage (horizontal sibling edge) -> must be rejected
        self.assertFalse(
            is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.daemon.graph.storage.db")
        )
        # ingestion importing graph (horizontal sibling edge) -> must be rejected
        self.assertFalse(
            is_allowed_internal_import("yaw.daemon.ingestion", "yaw.daemon.graph.engine")
        )
        # inference importing daemon runtime (child importing parent) -> must be rejected
        self.assertFalse(
            is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.daemon.runtime")
        )
        # core importing daemon (core importing non-core) -> must be rejected
        self.assertFalse(is_allowed_internal_import("yaw.core", "yaw.daemon.graph.inference"))
        # inference importing core -> allowed
        self.assertTrue(is_allowed_internal_import("yaw.daemon.graph.inference", "yaw.core.types"))
        # inference importing internal sibling -> allowed
        self.assertTrue(
            is_allowed_internal_import(
                "yaw.daemon.graph.inference", "yaw.daemon.graph.inference.models"
            )
        )

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
                                        f"Public facade {init_file.relative_to(PROJECT_ROOT)} exports protected/private symbol '{key.value}'",
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
                        bool(node.module)
                        and (node.module.startswith("yaw.") or node.module == "yaw")
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
