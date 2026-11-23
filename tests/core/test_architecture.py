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
SKILLS_DIR = PROJECT_ROOT / ".agents" / "skills"

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
        "epic-3-performance.md",
        "evaluation-graph-storage.md",
        "evaluation-inference.md",
        "evaluation-ingestion.md",
        "evaluation-testing.md",
        "evaluation.md",
        "model-evaluation.md",
        "native-cocoindex-graph.md",
        "native-cocoindex-indexing.md",
    }
)
_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_WIKI_LINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_HTML_ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", re.IGNORECASE)
_LINE_FRAG_RE = re.compile(r"^L(\d+)(?:-L(\d+))?$")


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
    loaded: object = yaml.safe_load(raw_yaml)
    if not isinstance(loaded, dict):
        raise AssertionError(f"{md_path.relative_to(PROJECT_ROOT)} frontmatter did not parse to a mapping")
    meta: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise AssertionError(f"{md_path.relative_to(PROJECT_ROOT)} frontmatter key {key!r} is not a string")
        meta[key] = value
    return meta, body


def _strip_fenced_lines(markdown_text: str) -> list[str]:
    """Return lines outside fenced code blocks."""
    lines: list[str] = []
    in_fence = False
    for line in markdown_text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return lines


def _strip_fenced_code_blocks(markdown_text: str) -> str:
    """Remove fenced code blocks and mask inline code spans before scanning prose links."""
    return "\n".join(re.sub(r"`[^`]*`", "CODE", line) for line in _strip_fenced_lines(markdown_text))


def _collect_anchors_and_line_count(target: Path) -> tuple[set[str], int]:
    """Collect GitHub-style heading slugs, HTML anchor IDs, and line count for a local target file."""
    raw_text = target.read_text(encoding="utf-8")
    total_lines = len(raw_text.splitlines())
    if target.suffix.lower() != ".md":
        return set(), total_lines

    anchors: set[str] = set()
    heading_counts: dict[str, int] = {}
    for line in _strip_fenced_lines(raw_text):
        for anchor_match in _HTML_ANCHOR_RE.finditer(line):
            anchors.add(anchor_match.group(1).lower())
        heading_match = _HEADING_RE.match(line)
        if not heading_match:
            continue
        heading = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", heading_match.group(1))
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading).replace("`", "")
        base = "".join(ch for ch in heading.lower() if ch.isalnum() or ch in (" ", "-", "_")).replace(" ", "-")
        idx = heading_counts.get(base, 0)
        heading_counts[base] = idx + 1
        anchors.add(base if idx == 0 else f"{base}-{idx}")
    return anchors, total_lines


def _parse_opening_relation_table(doc_file: Path) -> dict[str, str]:
    """Parse the opening '| Relation | Links |' navigation table before any '##' section."""
    lines = doc_file.read_text(encoding="utf-8").splitlines()
    table_start: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if re.match(r"^\|\s*Relation\s*\|\s*Links\s*\|$", stripped):
            table_start = idx
            break
    if table_start is None:
        return {}

    rows: dict[str, str] = {}
    for line in lines[table_start + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        if re.match(r"^\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|$", stripped):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) >= 2:
            rows[parts[0]] = " | ".join(parts[1:])
    return rows


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
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import: resolve against current_pkg_parts
                # level=1 means from . import X (current package)
                # level=2 means from .. import X (parent package)
                drop = node.level - 1
                base_parts = [] if drop > len(current_pkg_parts) else current_pkg_parts[: len(current_pkg_parts) - drop]
                mod = ".".join(base_parts)
                full_mod = (f"{mod}.{node.module}" if mod else node.module) if node.module is not None else mod
                imports.append((node.lineno, full_mod))
            elif node.module is not None:
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

    # General Rule: package P imports only P.* (itself or its descendants)
    return bool(imported_mod == importer_pkg or imported_mod.startswith(f"{importer_pkg}."))


def is_leaf_package(package_dir: Path) -> bool:
    """Return True if package_dir contains no subdirectories with an __init__.py file."""
    return all(not (child.is_dir() and (child / "__init__.py").exists()) for child in package_dir.iterdir())


def _resolve_graph_target(name: str, module_map: dict[str, Path]) -> str | None:
    cand: str | None = name
    while cand is not None and cand not in module_map:
        cand = cand.rsplit(".", 1)[0] if "." in cand else None
    return cand


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
                        cand = _resolve_graph_target(name, module_map)
                        if cand is not None and cand != mod:
                            graph[mod].add(cand)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    drop = node.level - 1
                    base = pkg_parts[: len(pkg_parts) - drop] if drop <= len(pkg_parts) else []
                    m_str = ".".join(base)
                    full = f"{m_str}.{node.module}" if node.module is not None else m_str
                    cand = _resolve_graph_target(full, module_map)
                    if cand is not None and cand != mod:
                        graph[mod].add(cand)
                    for alias in node.names:
                        alias_full = f"{full}.{alias.name}" if full else alias.name
                        if alias_full in module_map and alias_full != mod:
                            graph[mod].add(alias_full)
                elif node.module is not None and (node.module.startswith("yaw.") or node.module == "yaw"):
                    cand = _resolve_graph_target(node.module, module_map)
                    if cand is not None and cand != mod:
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
        """Rule 1b: yaw.core exports its slim public contract and yaw/ preserves the exact 14-file boundary."""
        import yaw.core as core
        from yaw.core import (
            ChunkPointer,
            DaemonStatus,
            HydratedPassage,
            QAResult,
            YawConfig,
            YawRuntimeError,
            compute_chunk_int_id,
            configure_logging,
            format_doc_for_embedding,
            format_query_for_embedding,
            get_config_value,
            get_logger,
            inclusive_end_line,
            load_config,
            parse_dataset_list,
            redact_secrets,
            render_document_text,
            resolve_coco_state_root,
            resolve_corpus_path,
            set_config_value,
            write_text_atomic,
            write_via_temp_atomic,
        )

        for cls in (ChunkPointer, DaemonStatus, HydratedPassage, QAResult, YawConfig):
            self.assertTrue(isinstance(cls, type))
        self.assertTrue(issubclass(YawRuntimeError, RuntimeError))
        for fn in (
            compute_chunk_int_id,
            configure_logging,
            format_doc_for_embedding,
            format_query_for_embedding,
            get_config_value,
            get_logger,
            inclusive_end_line,
            load_config,
            parse_dataset_list,
            redact_secrets,
            render_document_text,
            resolve_coco_state_root,
            resolve_corpus_path,
            set_config_value,
            write_text_atomic,
            write_via_temp_atomic,
        ):
            self.assertTrue(callable(fn))

        for retired in (
            "json_document_fields",
            "IndexStats",
            "FatalGraphPublicationError",
            "RetrievalParams",
            "write_manifold_section",
            "write_retrieval_section",
            "compute_chunk_id",
        ):
            self.assertFalse(hasattr(core, retired), f"Retired symbol {retired} leaked through yaw.core")

        self.assertTrue(all(not name.startswith("_") for name in core.__all__))
        with self.assertRaises(AttributeError):
            _ = core._parse_config_scalar

        self.assertFalse((YAW_DIR / "daemon").exists(), "Retired yaw/daemon/ directory must remain absent")
        self.assertFalse((YAW_DIR / "core" / "constants.py").exists(), "Retired yaw/core/constants.py must be deleted")

        actual_files = {p.relative_to(YAW_DIR).as_posix() for p in YAW_DIR.rglob("*.py")}
        expected_files = {
            "__init__.py",
            "__main__.py",
            "cli.py",
            "core/__init__.py",
            "core/config.py",
            "core/logger.py",
            "core/types.py",
            "datasets.py",
            "mcp_server.py",
            "optimize.py",
            "pipeline.py",
            "qa.py",
            "retrieval.py",
            "runtime.py",
        }
        self.assertEqual(actual_files, expected_files)

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
        """Rule: yaw/core/* must strictly import standard library only (zero third-party, zero runtime)."""
        core_files = list((YAW_DIR / "core").glob("*.py"))
        self.assertEqual(
            {f.name for f in core_files},
            {"__init__.py", "config.py", "logger.py", "types.py"},
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
        self.assertFalse(is_allowed_internal_import("yaw.core", "yaw.runtime"))
        self.assertTrue(is_allowed_internal_import("yaw.core", "yaw.core.types"))
        self.assertFalse(is_allowed_internal_import("yaw.pkg_a", "yaw.pkg_b"))
        self.assertFalse(is_allowed_internal_import("yaw.pkg_a.sub", "yaw.pkg_a"))
        self.assertTrue(is_allowed_internal_import("yaw.pkg_a", "yaw.pkg_a.sub"))
        self.assertTrue(is_allowed_internal_import("yaw.pkg_a", "yaw.core.types"))

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
                        violations.extend(
                            f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                            f"cross-module import of protected/private symbol '{alias.name}' is forbidden"
                            for alias in node.names
                            if alias.name.startswith("_")
                        )

        self.assertEqual(
            violations,
            [],
            "Protected/private cross-module imports detected:\n" + "\n".join(violations),
        )

    def test_zero_cross_module_private_attributes(self):
        """Rule 5b: Zero cross-object private/protected attribute access across all 14 files in yaw/.

        Forbids accessing _-prefixed attributes on objects other than self, cls, or super()
        (allowing stdlib os._exit), and forbids any access to ._provider across yaw/.
        """

        def is_self_cls_super(val_node: ast.AST) -> bool:
            if isinstance(val_node, ast.Name) and val_node.id in {"self", "cls"}:
                return True
            return bool(
                isinstance(val_node, ast.Call) and isinstance(val_node.func, ast.Name) and val_node.func.id == "super"
            )

        target_files = sorted(YAW_DIR.rglob("*.py"))
        self.assertEqual(len(target_files), 14, f"Expected exact 14 Python files in yaw/, got {len(target_files)}")

        violations: list[str] = []
        provider_violations: list[str] = []
        for py_file in target_files:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr.startswith("_")
                    and not node.attr.startswith("__")
                    and not is_self_cls_super(node.value)
                ):
                    if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr == "_exit":
                        continue
                    violations.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"cross-object access of private attribute '{node.attr}'"
                    )
                if isinstance(node, ast.Attribute) and node.attr == "_provider":
                    provider_violations.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                        f"access of ._provider attribute is forbidden"
                    )

        self.assertEqual(
            violations,
            [],
            "Cross-object private attribute accesses detected:\n" + "\n".join(violations),
        )
        self.assertEqual(
            provider_violations,
            [],
            "Forbidden ._provider attribute accesses detected in yaw/:\n" + "\n".join(provider_violations),
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
                "core.md",
                "retrieval.md",
                "pipeline.md",
                "runtime.md",
                "qa.md",
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
        """Rule 8: Subagents and skills mount cleanly with valid YAML frontmatter, tools, and zero retired markers."""
        expected_agents = {"critic", "implementer", "ontologist"}
        agent_files = {p.parent.name: p for p in AGENTS_DIR.glob("*/agent.md")}
        self.assertEqual(set(agent_files.keys()), expected_agents)

        retired_markers = (
            "sample-llm-wiki",
            "pyshacl.validate",
            "lean-ctx",
            "audit_report.md",
            "vec0",
            "DatabaseSink",
            "ppr.py",
            "hmm.py",
            "spacy",
            "sqlite-vec",
            "sqlite_vec",
            "optuna",
            "dspy.RLM",
            "dspy-rlm-module",
        )
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

            body_cf = body.casefold()
            for marker in retired_markers:
                self.assertNotIn(marker.casefold(), body_cf, f"Subagent {rel} contains retired reference '{marker}'")

        # Validate active .agents/skills/*/SKILL.md frontmatter and enforce retired-marker ban across skills and rules.
        self.assertFalse((SKILLS_DIR / "dspy-rlm-module").exists(), "Retired dspy-rlm-module skill must not exist")
        self.assertFalse((SKILLS_DIR / "spacy").exists(), "Retired spacy skill must not exist")

        skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
        self.assertGreater(len(skill_dirs), 0, "No skill directories found in .agents/skills/")
        for skill_dir in skill_dirs:
            skill_md = skill_dir / "SKILL.md"
            self.assertTrue(skill_md.is_file(), f"Skill directory {skill_dir.name} missing SKILL.md")
            rel = skill_md.relative_to(PROJECT_ROOT)
            meta, _ = _parse_yaml_frontmatter(skill_md)
            self.assertEqual(meta.get("name"), skill_dir.name, f"Skill {rel} name must match directory")
            desc = meta.get("description")
            self.assertIsInstance(desc, str, f"Skill {rel} must declare a description string")
            assert isinstance(desc, str)
            word_count = len(desc.split())
            self.assertTrue(1 <= word_count <= 60, f"Skill {rel} description must be 1..60 words (got {word_count})")

        extended_markers = retired_markers + ("vec_distance_cosine", "temperature=0.0")
        for skill_file in sorted(p for p in SKILLS_DIR.rglob("*") if p.is_file() and p.suffix in {".md", ".py"}):
            rel = skill_file.relative_to(PROJECT_ROOT)
            text_cf = skill_file.read_text(encoding="utf-8").casefold()
            for marker in extended_markers:
                self.assertNotIn(marker.casefold(), text_cf, f"Skill {rel} contains retired marker '{marker}'")

        rule13_doc_tokens = {"spacy", "sqlite_vec", "sqlite-vec", "optuna"}
        for rule_file in sorted(RULES_DIR.glob("*.md")):
            rel = rule_file.relative_to(PROJECT_ROOT)
            for lineno, line in enumerate(rule_file.read_text(encoding="utf-8").splitlines(), start=1):
                line_cf = line.casefold()
                is_rule13_desc = (
                    rule_file.name == "canonical-files.md"
                    and "rule 13 (`test_purged_dependencies_banned_across_yaw`)" in line_cf
                )
                for marker in extended_markers:
                    if is_rule13_desc and marker.casefold() in rule13_doc_tokens:
                        continue
                    self.assertNotIn(
                        marker.casefold(),
                        line_cf,
                        f"Rule {rel}:{lineno} contains retired marker '{marker}'",
                    )

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
            + list(SKILLS_DIR.rglob("*.md"))
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
        """Rule 10: Zero [[wiki-links]] or broken relative Markdown links across rules, agents, skills, and docs."""
        non_arxiv_docs = [
            p for p in sorted(DOCS_DIR.rglob("*.md")) if "papers" not in p.parts and not p.name.startswith("arxiv-")
        ]
        checked_files = (
            [PROJECT_ROOT / "AGENTS.md", PROJECT_ROOT / "FOUNDATIONAL.md", PROJECT_ROOT / "README.md"]
            + sorted(RULES_DIR.glob("*.md"))
            + sorted(AGENTS_DIR.glob("*/agent.md"))
            + sorted(SKILLS_DIR.rglob("*.md"))
            + non_arxiv_docs
        )

        wiki_violations: list[str] = []
        broken_links: list[str] = []
        target_cache: dict[Path, tuple[set[str], int]] = {}
        for md_file in checked_files:
            rel_source = md_file.relative_to(PROJECT_ROOT)
            prose = _strip_fenced_code_blocks(md_file.read_text(encoding="utf-8"))
            wiki_violations.extend(
                f"{rel_source}: forbidden wiki-link {match.group(0)}" for match in _WIKI_LINK_RE.finditer(prose)
            )

            for match in _MD_LINK_RE.finditer(prose):
                raw_target = match.group(1).strip()
                if not raw_target or raw_target.startswith(
                    ("http://", "https://", "mailto:", "file://", "conversation://")
                ):
                    continue
                path_part, _, raw_frag = raw_target.partition("#")
                path_part = path_part.strip()
                frag = raw_frag.strip()
                resolved = md_file.resolve() if not path_part else (md_file.parent / path_part).resolve()
                if not resolved.exists():
                    broken_links.append(f"{rel_source}: broken relative link '{raw_target}' -> {resolved}")
                    continue
                if not frag:
                    continue
                if not resolved.is_file():
                    broken_links.append(f"{rel_source}: fragment '#{frag}' on non-file target '{raw_target}'")
                    continue
                if resolved not in target_cache:
                    target_cache[resolved] = _collect_anchors_and_line_count(resolved)
                anchors, total_lines = target_cache[resolved]
                if line_match := _LINE_FRAG_RE.match(frag):
                    start = int(line_match.group(1))
                    end = int(line_match.group(2)) if line_match.group(2) else start
                    if not (1 <= start <= end <= total_lines):
                        broken_links.append(
                            f"{rel_source}: out-of-range line fragment '{raw_target}' (lines=1..{total_lines})"
                        )
                elif frag.lower() not in anchors:
                    broken_links.append(f"{rel_source}: broken heading/anchor fragment '{raw_target}'")

        self.assertEqual(wiki_violations, [], "Forbidden [[wiki-links]] found:\n" + "\n".join(wiki_violations))
        self.assertEqual(broken_links, [], "Broken relative Markdown links found:\n" + "\n".join(broken_links))

    def test_every_doc_is_reachable_from_rules_and_canonical_inventory(self):
        """Rule 11: docs/ matches the canonical 16-file inventory, every doc is reachable, and nav rows stay current."""
        actual_docs = {p.name for p in DOCS_DIR.glob("*.md")}
        self.assertEqual(
            actual_docs,
            _CANONICAL_DOCS_INVENTORY,
            f"docs/ inventory drift. Extra: {sorted(actual_docs - _CANONICAL_DOCS_INVENTORY)}, "
            f"Missing: {sorted(_CANONICAL_DOCS_INVENTORY - actual_docs)}",
        )

        domain_rules = ("retrieval.md", "pipeline.md", "runtime.md", "qa.md", "tests.md")
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

        stale_nav: list[str] = []
        for doc_name in sorted(_CANONICAL_DOCS_INVENTORY):
            doc_file = DOCS_DIR / doc_name
            for lineno, line in enumerate(doc_file.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.lstrip()
                if re.match(r"^\|\s*(?:Owns code|Rules)\s*\|", stripped) and re.search(
                    r"\byaw/daemon(?:/|\b)", stripped
                ):
                    stale_nav.append(f"docs/{doc_name}:{lineno}: retired yaw/daemon path in navigation row")
        self.assertEqual(
            stale_nav,
            [],
            "Retired yaw/daemon/ paths found in canonical docs navigation tables:\n" + "\n".join(stale_nav),
        )

        tier2_specs = (
            "epic-1-hipporag-daemon.md",
            "epic-2-fastmcp-http.md",
            "epic-3-performance.md",
            "native-cocoindex-indexing.md",
            "native-cocoindex-graph.md",
            "dspy-improvements.md",
            "enterprise-rag-bench-training-plan.md",
            "coverage-plan.md",
        )
        yaw_files = sorted(p.relative_to(PROJECT_ROOT).as_posix() for p in YAW_DIR.rglob("*.py"))
        self.assertEqual(len(yaw_files), 14, f"Expected exact 14 Python files in yaw/, got {len(yaw_files)}")
        owners_by_yaw_file: dict[str, list[str]] = {rel: [] for rel in yaw_files}
        peers_by_spec: dict[str, set[str]] = {}
        for spec_name in tier2_specs:
            rows = _parse_opening_relation_table(DOCS_DIR / spec_name)
            self.assertTrue(rows, f"docs/{spec_name} must open with a '| Relation | Links |' navigation table")
            for required_rel in ("Owns code", "Rules", "Peers"):
                self.assertIn(
                    required_rel,
                    rows,
                    f"docs/{spec_name} opening navigation table is missing required row {required_rel!r}",
                )
            owns_cell = rows["Owns code"]
            self.assertFalse(
                "{" in owns_cell or "}" in owns_cell,
                f"docs/{spec_name} Owns code row must list explicit paths or '*' globs without brace expansion",
            )
            for token in re.findall(r"\byaw/[A-Za-z0-9_./*-]+", owns_cell):
                matched = [rel for rel in yaw_files if PurePosixPath(rel).full_match(token)]
                self.assertTrue(matched, f"docs/{spec_name} Owns code token {token!r} matched 0 files under yaw/")
                for rel in matched:
                    owners_by_yaw_file[rel].append(spec_name)

            peer_targets: set[str] = set()
            for match in _MD_LINK_RE.finditer(rows["Peers"]):
                raw_target = match.group(1).split("#", 1)[0].strip()
                resolved = (DOCS_DIR / raw_target).resolve()
                self.assertEqual(
                    resolved.parent,
                    DOCS_DIR.resolve(),
                    f"docs/{spec_name} Peers link {raw_target!r} must target an active Tier-2 spec in docs/",
                )
                self.assertIn(
                    resolved.name,
                    tier2_specs,
                    f"docs/{spec_name} Peers link {raw_target!r} is not an active Tier-2 spec",
                )
                self.assertNotEqual(resolved.name, spec_name, f"docs/{spec_name} Peers row must not self-link")
                peer_targets.add(resolved.name)
            peers_by_spec[spec_name] = peer_targets

        unowned = [rel for rel, owners in owners_by_yaw_file.items() if not owners]
        multi_owned = [f"{rel}: {owners}" for rel, owners in owners_by_yaw_file.items() if len(owners) > 1]
        self.assertEqual(unowned, [], f"yaw/ modules with no Owns code spec owner: {unowned}")
        self.assertEqual(multi_owned, [], f"yaw/ modules with multiple Owns code spec owners: {multi_owned}")

        asymmetric_peers = [
            f"docs/{a} lists {b} in Peers, but docs/{b} omits {a}"
            for a in tier2_specs
            for b in sorted(peers_by_spec[a])
            if a not in peers_by_spec[b]
        ]
        self.assertEqual(asymmetric_peers, [], "Asymmetric Tier-2 Peers rows:\n" + "\n".join(asymmetric_peers))

    def test_canonical_module_purity_and_behavioral_contracts(self):
        """Rule 12: Clean-slate 14-file architecture obeys LOC cap, zero wrappers/legacy terms, and L0-L5 DAG."""
        stdlib = set(sys.stdlib_module_names)
        py_files = sorted(YAW_DIR.rglob("*.py"))
        self.assertEqual(len(py_files), 14, f"Expected exact 14 Python files in yaw/, got {len(py_files)}")
        self.assertFalse((YAW_DIR / "daemon").exists(), "Retired yaw/daemon/ directory must not exist")

        total_loc = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in py_files)
        self.assertLessEqual(
            total_loc,
            8000,
            f"yaw/ total LOC {total_loc} exceeds 8,000 ceiling (calibrated per docs/epic-3-performance.md#decisions)",
        )
        core_files = sorted((YAW_DIR / "core").glob("*.py"))
        core_loc = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in core_files)
        self.assertLessEqual(core_loc, 960, f"yaw/core/* LOC {core_loc} exceeds 960 ceiling")
        watchdog_loc = len((PROJECT_ROOT / "scripts" / "watchdog.py").read_text(encoding="utf-8").splitlines())
        self.assertLessEqual(watchdog_loc, 350, f"scripts/watchdog.py LOC {watchdog_loc} exceeds 350 ceiling")

        legacy_terms_re = re.compile(
            r"manifold|hmm|hipporag|DatabaseSink|ReflectionLM|YawQA|"
            r"DSPyEmbeddingAdapter|RLMInstructionProposer|YawProgram|YawGBNFAdapter",
            re.IGNORECASE,
        )
        forbidden_bases = {"Module", "BaseLM", "Adapter", "dspy.Module", "dspy.BaseLM", "dspy.Adapter"}

        for py_file in py_files:
            rel = py_file.relative_to(PROJECT_ROOT)
            text = py_file.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if m := legacy_terms_re.search(line):
                    self.fail(f"{rel}:{lineno} contains forbidden legacy symbol/term {m.group(0)!r}")
            tree = ast.parse(text, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        base_txt = ast.unparse(base)
                        self.assertFalse(
                            base_txt in forbidden_bases or base_txt.endswith((".Module", ".BaseLM", ".Adapter")),
                            f"{rel}:{node.lineno}: class {node.name} must not subclass {base_txt}",
                        )

        http_mods = {"httpx", "httpx2", "openai", "urllib", "http", "socket", "requests"}
        layer_rules: dict[str, tuple[set[str], set[str]]] = {
            # Maps module filename to (allowed_top_level_packages, allowed_yaw_internal_modules).
            "datasets.py": (stdlib | {"filelock", "numpy", "yaw"}, {"yaw.core"}),
            "retrieval.py": (
                stdlib | {"numpy", "scipy", "sqlite3", "yaw"},
                {"yaw.core"},
            ),
            "pipeline.py": (
                stdlib | {"cocoindex", "dspy", "numpy", "sqlite3", "yaw"},
                {"yaw.core", "yaw.retrieval"},
            ),
            "qa.py": (stdlib | {"dspy", "numpy", "sqlite3", "yaw"}, {"yaw.core", "yaw.retrieval"}),
            "runtime.py": (
                stdlib | {"dspy", "filelock", "numpy", "sqlite3", "yaw"},
                {"yaw.core", "yaw.pipeline", "yaw.qa", "yaw.retrieval"},
            ),
            "mcp_server.py": (
                stdlib | {"fastmcp", "pydantic", "starlette", "uvicorn", "yaw"},
                {"yaw.core", "yaw.runtime"},
            ),
            "optimize.py": (
                stdlib | {"dspy", "filelock", "numpy", "sqlite3", "yaw"},
                {"yaw.core", "yaw.datasets", "yaw.pipeline", "yaw.qa", "yaw.retrieval"},
            ),
            "cli.py": (
                stdlib | {"filelock", "yaw"},
                {"yaw.core", "yaw.runtime", "yaw.mcp_server", "yaw.optimize"},
            ),
            "__main__.py": (stdlib | {"yaw"}, {"yaw.cli"}),
        }
        forbidden_explicit: dict[str, set[str]] = {
            "datasets.py": {"dspy", "cocoindex", "sqlite3"} | http_mods,
            "retrieval.py": {"dspy", "cocoindex"} | http_mods,
            "qa.py": {"cocoindex"} | http_mods,
            "runtime.py": {"cocoindex", "fastmcp", "starlette", "uvicorn"},
            "cli.py": {"dspy", "sqlite3", "cocoindex", "fastmcp", "starlette", "uvicorn", "numpy"},
        }

        for mod_name, (allowed_top, allowed_yaw) in layer_rules.items():
            mod_path = YAW_DIR / mod_name
            for lineno, mod in parse_module_imports(mod_path):
                top = mod.split(".")[0]
                self.assertIn(top, allowed_top, f"{mod_name}:{lineno} imports forbidden package '{mod}'")
                self.assertNotIn(
                    top,
                    forbidden_explicit.get(mod_name, set()),
                    f"{mod_name}:{lineno} explicitly forbidden import '{mod}'",
                )
                if top == "yaw":
                    self.assertTrue(
                        any(mod == a or mod.startswith(f"{a}.") for a in allowed_yaw),
                        f"{mod_name}:{lineno} violates DAG by importing '{mod}' (allowed: {sorted(allowed_yaw)})",
                    )

        # Verify lazy top-level exclusions in cli.py
        cli_tree = ast.parse((YAW_DIR / "cli.py").read_text(encoding="utf-8"))
        for stmt in cli_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    self.assertFalse(
                        alias.name.startswith(("yaw.optimize", "yaw.datasets")),
                        f"cli.py:{stmt.lineno} must lazy-import {alias.name}",
                    )
            elif isinstance(stmt, ast.ImportFrom) and stmt.module is not None:
                self.assertFalse(
                    stmt.module.startswith(("yaw.optimize", "yaw.datasets")),
                    f"cli.py:{stmt.lineno} must lazy-import {stmt.module}",
                )

        self.assertLessEqual(len((YAW_DIR / "__main__.py").read_text(encoding="utf-8").splitlines()), 10)

    def test_purged_dependencies_banned_across_yaw(self):
        """Rule 13: Purged dependencies (sqlite_vec, spacy, en_core_web_sm, optuna) are absent across all yaw/**."""
        banned_pkgs = {"sqlite_vec", "spacy", "en_core_web_sm", "optuna"}
        banned_re = re.compile(r"(sqlite_vec|sqlite-vec|spacy|en_core_web_sm|en-core-web-sm|optuna)", re.IGNORECASE)
        source_files = sorted(
            p
            for p in YAW_DIR.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and "outputs" not in p.parts and p.suffix != ".pyc"
        )
        self.assertGreater(len(source_files), 0)

        violations: list[str] = []
        for file_path in source_files:
            rel = file_path.relative_to(PROJECT_ROOT)
            text = file_path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if m := banned_re.search(line):
                    violations.append(f"{rel}:{lineno}: banned dependency reference {m.group(0)!r}")
            if file_path.suffix == ".py":
                for lineno, mod in parse_module_imports(file_path):
                    top = mod.split(".")[0]
                    if top in banned_pkgs:
                        violations.append(f"{rel}:{lineno}: banned import '{mod}'")

        self.assertEqual(
            violations,
            [],
            "Purged dependency references found under yaw/:\n" + "\n".join(violations),
        )
