"""AST-enforced architecture containment and zero-cross-domain import boundary tests."""

import ast
import io
import re
import sys
import tempfile
import tokenize
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
_BACKTICKED_LINK_RE = re.compile(r"`\[[^\]]+\]\(([^)]+)\)`")
_WIKI_LINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_HTML_ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=[\"']([^\"']+)[\"']", re.IGNORECASE)
_LINE_FRAG_RE = re.compile(r"^L(\d+)(?:-L(\d+))?$")
_SUPPRESSION_COMMENT_RE = re.compile(r"#\s*(?:(?:ruff|flake8):\s*)?noqa\b|\bC901\b", re.IGNORECASE)

_ALLOWED_DOMAIN_IMPORTS: dict[str, frozenset[str]] = {
    "yaw.core": frozenset({"yaw.core"}),
    "yaw.datasets": frozenset({"yaw.datasets", "yaw.core"}),
    "yaw.retrieval": frozenset({"yaw.retrieval", "yaw.core"}),
    "yaw.pipeline": frozenset({"yaw.pipeline", "yaw.core", "yaw.retrieval"}),
    "yaw.qa": frozenset({"yaw.qa", "yaw.core", "yaw.retrieval"}),
    "yaw.runtime": frozenset({"yaw.runtime", "yaw.core", "yaw.retrieval", "yaw.pipeline", "yaw.qa"}),
    "yaw.mcp_server": frozenset({"yaw.mcp_server", "yaw.core", "yaw.runtime"}),
    "yaw.optimize": frozenset({"yaw.optimize", "yaw.core", "yaw.datasets", "yaw.pipeline", "yaw.qa", "yaw.retrieval"}),
    "yaw.cli": frozenset({"yaw.cli", "yaw.core", "yaw.mcp_server", "yaw.optimize", "yaw.runtime"}),
    "yaw.__main__": frozenset({"yaw.__main__", "yaw.cli"}),
}


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


def _resolve_relative_module(pkg_parts: list[str], level: int, module: str | None) -> str:
    drop = level - 1
    base_parts = [] if drop > len(pkg_parts) else pkg_parts[: len(pkg_parts) - drop]
    mod = ".".join(base_parts)
    if module is None:
        return mod
    return f"{mod}.{module}" if mod else module


def _extract_public_api_dict(tree: ast.Module) -> ast.Dict | None:
    for stmt in tree.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "_PUBLIC_API":
            if isinstance(stmt.value, ast.Dict):
                return stmt.value
        elif (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Dict)
            and any(isinstance(t, ast.Name) and t.id == "_PUBLIC_API" for t in stmt.targets)
        ):
            return stmt.value
    return None


def _collect_import_call_names(tree: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    importlib_mods = {"importlib"}
    import_module_fns: set[str] = set()
    builtin_import_fns = {"__import__"}
    builtins_mods = {"builtins"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name == "importlib":
                    importlib_mods.add(bound)
                elif alias.name == "builtins":
                    builtins_mods.add(bound)
        elif isinstance(node, ast.ImportFrom) and node.module in {"importlib", "builtins"}:
            target_set = import_module_fns if node.module == "importlib" else builtin_import_fns
            expected_name = "import_module" if node.module == "importlib" else "__import__"
            for alias in node.names:
                if alias.name == expected_name:
                    target_set.add(alias.asname or alias.name)
    return importlib_mods, import_module_fns, builtin_import_fns, builtins_mods


def _is_dynamic_import_call(
    node: ast.Call,
    importlib_mods: set[str],
    import_module_fns: set[str],
    builtin_import_fns: set[str],
    builtins_mods: set[str],
) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in import_module_fns or func.id in builtin_import_fns
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        is_importlib = func.attr == "import_module" and func.value.id in importlib_mods
        is_builtin = func.attr == "__import__" and func.value.id in builtins_mods
        return is_importlib or is_builtin
    return False


def _resolve_dynamic_call_targets(
    call: ast.Call,
    mod_parts: list[str],
    pkg_parts: list[str],
    is_init: bool,
) -> list[str] | None:
    arg0 = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "name"), None)
    if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
        return None

    raw_mod = arg0.value
    if not raw_mod.startswith("."):
        targets = [raw_mod]
        fromlist_arg = (
            call.args[3]
            if len(call.args) >= 4
            else next((kw.value for kw in call.keywords if kw.arg == "fromlist"), None)
        )
        if isinstance(fromlist_arg, ast.List | ast.Tuple):
            targets.extend(
                f"{raw_mod}.{item.value}"
                for item in fromlist_arg.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value != "*"
            )
        return targets

    pkg_arg = (
        call.args[1] if len(call.args) >= 2 else next((kw.value for kw in call.keywords if kw.arg == "package"), None)
    )
    if isinstance(pkg_arg, ast.Constant) and isinstance(pkg_arg.value, str):
        base_parts = pkg_arg.value.split(".")
    elif isinstance(pkg_arg, ast.Name) and pkg_arg.id == "__package__":
        base_parts = pkg_parts
    elif isinstance(pkg_arg, ast.Name) and pkg_arg.id == "__name__":
        base_parts = pkg_parts if is_init else mod_parts
    else:
        return None

    level = len(raw_mod) - len(raw_mod.lstrip("."))
    suffix = raw_mod.lstrip(".") or None
    return [_resolve_relative_module(base_parts, level, suffix)]


def _public_api_getattr_call_ids(tree: ast.Module) -> set[int]:
    if _extract_public_api_dict(tree) is None:
        return set()
    allowed_ids: set[int] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__getattr__":
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    allowed_ids.add(id(node))
    return allowed_ids


def _file_module_context(file_path: Path) -> tuple[list[str], list[str], bool]:
    is_init = file_path.name == "__init__.py"
    if not file_path.is_relative_to(PROJECT_ROOT):
        return [], [], is_init
    rel_parts = list(file_path.relative_to(PROJECT_ROOT).with_suffix("").parts)
    pkg_parts = rel_parts[:-1]
    mod_parts = pkg_parts if is_init else rel_parts
    return mod_parts, pkg_parts, is_init


def _parse_static_node_imports(
    node: ast.AST,
    mod_parts: list[str],
    pkg_parts: list[str],
    is_init: bool,
    call_names: tuple[set[str], set[str], set[str], set[str]],
) -> list[tuple[int, str]]:
    if isinstance(node, ast.Import):
        return [(node.lineno, alias.name) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return [(node.lineno, _resolve_relative_module(pkg_parts, node.level, node.module))]
        return [(node.lineno, node.module)] if node.module is not None else []
    if isinstance(node, ast.Call) and _is_dynamic_import_call(node, *call_names):
        targets = _resolve_dynamic_call_targets(node, mod_parts, pkg_parts, is_init)
        if targets is not None:
            return [(node.lineno, target) for target in targets]
    return []


def parse_module_imports(file_path: Path) -> list[tuple[int, str]]:
    """Parse static and string-literal dynamic imports in a Python file as (lineno, imported_module)."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    mod_parts, pkg_parts, is_init = _file_module_context(file_path)
    call_names = _collect_import_call_names(tree)
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        imports.extend(_parse_static_node_imports(node, mod_parts, pkg_parts, is_init, call_names))

    if is_init and (public_api := _extract_public_api_dict(tree)) is not None:
        for val in public_api.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                raw_mod = val.value
                if raw_mod.startswith("."):
                    level = len(raw_mod) - len(raw_mod.lstrip("."))
                    resolved = _resolve_relative_module(pkg_parts, level, raw_mod.lstrip(".") or None)
                    imports.append((val.lineno, resolved))
                else:
                    imports.append((val.lineno, raw_mod))
    return imports


def find_unresolvable_dynamic_imports(
    file_path: Path, *, allow_public_api_getattr: bool = False
) -> list[tuple[int, str]]:
    """Detect dynamic importlib.import_module/import_module/__import__ calls that cannot be resolved statically."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    mod_parts, pkg_parts, is_init = _file_module_context(file_path)
    call_names = _collect_import_call_names(tree)
    allowed_getattr_ids = _public_api_getattr_call_ids(tree) if allow_public_api_getattr and is_init else set()
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_dynamic_import_call(node, *call_names):
            if id(node) in allowed_getattr_ids:
                continue
            if _resolve_dynamic_call_targets(node, mod_parts, pkg_parts, is_init) is None:
                violations.append((node.lineno, ast.unparse(node)))
    return violations


def find_suppression_comments(source: str) -> list[tuple[int, str]]:
    """Return (lineno, comment) for # noqa, # ruff: noqa, or C901 suppression comments outside string literals."""
    return [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT and _SUPPRESSION_COMMENT_RE.search(tok.string)
    ]


def get_canonical_package(module_path: Path) -> str:
    """Return the dotted canonical domain/package name for a Python file inside yaw/."""
    rel = module_path.relative_to(PROJECT_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts == ["yaw", "__init__"]:
        return "yaw"
    if len(parts) == 2:
        return ".".join(parts)
    return ".".join(parts[:-1])


def _yaw_domain(dotted_name: str) -> str:
    parts = dotted_name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else dotted_name


def is_allowed_internal_import(importer_pkg: str, imported_mod: str) -> bool:
    """Validate monotonic L0-L6 layer DAG containment and downward intra-package imports."""
    if imported_mod != "yaw" and not imported_mod.startswith("yaw."):
        return True
    if imported_mod == "yaw":
        return importer_pkg == "yaw"

    importer_domain = _yaw_domain(importer_pkg)
    imported_domain = _yaw_domain(imported_mod)
    allowed_domains = _ALLOWED_DOMAIN_IMPORTS.get(importer_domain)
    if allowed_domains is None:
        if imported_domain == "yaw.core":
            return True
        return imported_domain == importer_domain and (
            imported_mod == importer_pkg or imported_mod.startswith(f"{importer_pkg}.")
        )

    if imported_domain not in allowed_domains:
        return False
    if imported_domain == importer_domain:
        return imported_mod == importer_pkg or imported_mod.startswith(f"{importer_pkg}.")
    return True


def is_leaf_package(package_dir: Path) -> bool:
    """Return True if package_dir contains no subdirectories with an __init__.py file."""
    return all(not (child.is_dir() and (child / "__init__.py").exists()) for child in package_dir.iterdir())


def _resolve_graph_target(name: str, module_map: dict[str, Path]) -> str | None:
    cand: str | None = name
    while cand is not None and cand not in module_map:
        cand = cand.rsplit(".", 1)[0] if "." in cand else None
    return cand


def _module_internal_edges(mod: str, file_path: Path, module_map: dict[str, Path]) -> set[str]:
    edges: set[str] = set()
    for _, imported_mod in parse_module_imports(file_path):
        if imported_mod == "yaw" or imported_mod.startswith("yaw."):
            cand = _resolve_graph_target(imported_mod, module_map)
            if cand is not None and cand != mod:
                edges.add(cand)

    _, pkg_parts, _ = _file_module_context(file_path)
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            full = (
                _resolve_relative_module(pkg_parts, node.level, node.module) if node.level > 0 else (node.module or "")
            )
            if full == "yaw" or full.startswith("yaw."):
                for alias in node.names:
                    alias_full = f"{full}.{alias.name}"
                    if alias_full in module_map and alias_full != mod:
                        edges.add(alias_full)
    return edges


def build_internal_import_graph() -> dict[str, set[str]]:
    """Build the directed module-level import graph for all Python modules in yaw/."""
    module_map: dict[str, Path] = {}
    for p in YAW_DIR.rglob("*.py"):
        rel = p.relative_to(PROJECT_ROOT).with_suffix("")
        parts = list(rel.parts)
        mod_name = ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)
        module_map[mod_name] = p
    return {mod: _module_internal_edges(mod, file_path, module_map) for mod, file_path in module_map.items()}


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


def _validate_yaw_vocabulary_and_bases(py_files: list[Path]) -> list[str]:
    legacy_terms_re = re.compile(
        r"manifold|hmm|hipporag|DatabaseSink|ReflectionLM|YawQA|"
        r"DSPyEmbeddingAdapter|RLMInstructionProposer|YawProgram|YawGBNFAdapter",
        re.IGNORECASE,
    )
    forbidden_bases = {"Module", "BaseLM", "Adapter", "dspy.Module", "dspy.BaseLM", "dspy.Adapter"}
    violations: list[str] = []
    for py_file in py_files:
        rel = py_file.relative_to(PROJECT_ROOT)
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if m := legacy_terms_re.search(line):
                violations.append(f"{rel}:{lineno} contains forbidden legacy symbol/term {m.group(0)!r}")
        tree = ast.parse(text, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_txt = ast.unparse(base)
                    if base_txt in forbidden_bases or base_txt.endswith((".Module", ".BaseLM", ".Adapter")):
                        violations.append(f"{rel}:{node.lineno}: class {node.name} must not subclass {base_txt}")
    return violations


def _validate_import_boundary(
    mod_path: Path,
    allowed_top: set[str],
    allowed_yaw: set[str],
    forbidden: set[str],
    *,
    allow_public_api_getattr: bool = False,
) -> list[str]:
    rel = mod_path.relative_to(PROJECT_ROOT).as_posix()
    violations = [
        f"{rel}:{lineno}: unresolvable dynamic import '{call_txt}'"
        for lineno, call_txt in find_unresolvable_dynamic_imports(
            mod_path, allow_public_api_getattr=allow_public_api_getattr
        )
    ]
    for lineno, mod in parse_module_imports(mod_path):
        top = mod.split(".")[0]
        if top not in allowed_top:
            violations.append(f"{rel}:{lineno} imports forbidden package '{mod}'")
        if top in forbidden:
            violations.append(f"{rel}:{lineno} explicitly forbidden import '{mod}'")
        if top == "yaw" and not any(mod == a or mod.startswith(f"{a}.") for a in allowed_yaw):
            violations.append(f"{rel}:{lineno} violates DAG by importing '{mod}' (allowed: {sorted(allowed_yaw)})")
    return violations


def _validate_cli_lazy_imports(cli_files: list[Path]) -> list[str]:
    violations: list[str] = []
    lazy_prefixes = ("yaw.optimize", "yaw.datasets")
    for cli_file in cli_files:
        rel = cli_file.relative_to(PROJECT_ROOT)
        mod_parts, pkg_parts, is_init = _file_module_context(cli_file)
        tree = ast.parse(cli_file.read_text(encoding="utf-8"), filename=str(cli_file))
        call_names = _collect_import_call_names(tree)
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            for node in ast.walk(stmt):
                for lineno, mod in _parse_static_node_imports(node, mod_parts, pkg_parts, is_init, call_names):
                    if mod.startswith(lazy_prefixes):
                        violations.append(f"{rel}:{lineno} must lazy-import {mod}")
    return violations


def _build_rule12_specs(
    stdlib: set[str],
) -> tuple[dict[str, tuple[set[str], set[str], set[str]]], dict[str, tuple[set[str], set[str], set[str]]]]:
    http_mods = {"httpx", "httpx2", "openai", "urllib", "http", "socket", "requests"}
    yaw_rules: dict[str, tuple[set[str], set[str], set[str]]] = {
        "core": (stdlib | {"yaw"}, {"yaw.core"}, {"scripts"}),
        "datasets": (
            stdlib | {"filelock", "numpy", "yaw"},
            {"yaw.core", "yaw.datasets"},
            {"dspy", "cocoindex", "sqlite3", "scripts"} | http_mods,
        ),
        "retrieval": (
            stdlib | {"numpy", "scipy", "sqlite3", "yaw"},
            {"yaw.core", "yaw.retrieval"},
            {"dspy", "cocoindex", "scripts"} | http_mods,
        ),
        "pipeline": (
            stdlib | {"cocoindex", "dspy", "numpy", "openai", "sqlite3", "yaw"},
            {"yaw.core", "yaw.retrieval", "yaw.pipeline"},
            {"scripts"},
        ),
        "qa": (
            stdlib | {"dspy", "numpy", "sqlite3", "yaw"},
            {"yaw.core", "yaw.retrieval", "yaw.qa"},
            {"cocoindex", "scripts"} | http_mods,
        ),
        "runtime": (
            stdlib | {"dspy", "filelock", "numpy", "sqlite3", "yaw"},
            {"yaw.core", "yaw.pipeline", "yaw.qa", "yaw.retrieval", "yaw.runtime"},
            {"cocoindex", "fastmcp", "starlette", "uvicorn", "scripts"},
        ),
        "mcp_server": (
            stdlib | {"fastmcp", "pydantic", "starlette", "uvicorn", "yaw"},
            {"yaw.core", "yaw.runtime", "yaw.mcp_server"},
            {"cocoindex", "dspy", "numpy", "sqlite3", "scripts"},
        ),
        "optimize": (
            stdlib | {"dspy", "filelock", "numpy", "sqlite3", "yaw"},
            {"yaw.core", "yaw.datasets", "yaw.pipeline", "yaw.qa", "yaw.retrieval", "yaw.optimize"},
            {"scripts"},
        ),
        "cli": (
            stdlib | {"filelock", "yaw"},
            {"yaw.core", "yaw.runtime", "yaw.mcp_server", "yaw.optimize", "yaw.cli"},
            {"dspy", "sqlite3", "cocoindex", "fastmcp", "starlette", "uvicorn", "numpy", "scripts"},
        ),
        "__main__": (stdlib | {"yaw"}, {"yaw.cli"}, {"scripts"}),
    }
    script_rules: dict[str, tuple[set[str], set[str], set[str]]] = {
        "watchdog.py": (
            stdlib | {"filelock", "matplotlib", "psutil", "sqlite3", "yaw"},
            {"yaw.core", "yaw.retrieval"},
            {"cocoindex", "fastmcp", "dspy"},
        ),
        "evaluation.py": (stdlib | {"numpy", "yaw"}, {"yaw"}, {"cocoindex", "fastmcp"}),
        "download_datasets.py": (stdlib | {"yaw"}, {"yaw"}, {"cocoindex", "fastmcp"}),
        "benchmark.py": (stdlib | {"dspy", "numpy", "scripts", "yaw"}, {"yaw"}, {"cocoindex", "fastmcp"}),
        "verify_paper_parity_gate.py": (
            stdlib | {"dspy", "numpy", "scripts", "yaw"},
            {"yaw"},
            {"cocoindex", "fastmcp"},
        ),
    }
    return yaw_rules, script_rules


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
        """Rule 1b: yaw.core exports its slim public contract and retired legacy files stay absent."""
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

    def test_domain_containment_and_zero_cross_domain_imports(self):
        """Rule 2 & 3: Monotonic L0-L6 DAG containment with zero upward or horizontal cross-domain edges."""
        py_files = sorted(YAW_DIR.rglob("*.py"))
        self.assertGreater(len(py_files), 0, "No .py files found in yaw/")

        violations = []
        for py_file in py_files:
            importer_pkg = get_canonical_package(py_file)
            if importer_pkg != "yaw":
                self.assertIn(
                    _yaw_domain(importer_pkg),
                    _ALLOWED_DOMAIN_IMPORTS,
                    f"{py_file.relative_to(PROJECT_ROOT)} belongs to undeclared domain '{importer_pkg}'",
                )
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
        """Rule: yaw/core/** must strictly import standard library only (zero third-party, zero runtime)."""
        core_files = sorted((YAW_DIR / "core").rglob("*.py"))
        self.assertGreater(len(core_files), 0, "No .py files found in yaw/core/")

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
        """Verify monotonic L0-L6 DAG checks reject L0->L1, same-layer sibling, and upward edges."""
        # L0 -> higher layers forbidden; own descendants allowed
        self.assertFalse(is_allowed_internal_import("yaw.core", "yaw.datasets"))
        self.assertFalse(is_allowed_internal_import("yaw.core", "yaw.retrieval"))
        self.assertFalse(is_allowed_internal_import("yaw.core", "yaw.runtime"))
        self.assertTrue(is_allowed_internal_import("yaw.core", "yaw.core.types"))

        # Same-layer horizontal siblings forbidden across L1, L2, L4
        self.assertFalse(is_allowed_internal_import("yaw.datasets", "yaw.retrieval"))
        self.assertFalse(is_allowed_internal_import("yaw.retrieval", "yaw.datasets"))
        self.assertFalse(is_allowed_internal_import("yaw.pipeline", "yaw.qa"))
        self.assertFalse(is_allowed_internal_import("yaw.qa", "yaw.pipeline"))
        self.assertFalse(is_allowed_internal_import("yaw.mcp_server", "yaw.optimize"))
        self.assertFalse(is_allowed_internal_import("yaw.optimize", "yaw.mcp_server"))

        # Upward cross-layer and non-declared skip edges forbidden
        self.assertFalse(is_allowed_internal_import("yaw.retrieval", "yaw.pipeline"))
        self.assertFalse(is_allowed_internal_import("yaw.pipeline", "yaw.runtime"))
        self.assertFalse(is_allowed_internal_import("yaw.qa", "yaw.runtime"))
        self.assertFalse(is_allowed_internal_import("yaw.runtime", "yaw.mcp_server"))
        self.assertFalse(is_allowed_internal_import("yaw.mcp_server", "yaw.cli"))
        self.assertFalse(is_allowed_internal_import("yaw.cli", "yaw.__main__"))
        self.assertFalse(is_allowed_internal_import("yaw.mcp_server", "yaw.retrieval"))
        self.assertFalse(is_allowed_internal_import("yaw.__main__", "yaw.core"))

        # Intra-package upward edges forbidden; downward edges and L0 core imports allowed
        self.assertFalse(is_allowed_internal_import("yaw.pkg_a", "yaw.pkg_b"))
        self.assertFalse(is_allowed_internal_import("yaw.pkg_a.sub", "yaw.pkg_a"))
        self.assertFalse(is_allowed_internal_import("yaw.retrieval.sub", "yaw.retrieval"))
        self.assertTrue(is_allowed_internal_import("yaw.pkg_a", "yaw.pkg_a.sub"))
        self.assertTrue(is_allowed_internal_import("yaw.pkg_a", "yaw.core.types"))
        self.assertTrue(is_allowed_internal_import("yaw.retrieval", "yaw.retrieval.graph"))
        self.assertTrue(is_allowed_internal_import("yaw.retrieval.sub", "yaw.core.types"))
        self.assertTrue(is_allowed_internal_import("yaw.pipeline", "yaw.retrieval.graph"))
        self.assertTrue(is_allowed_internal_import("yaw.qa", "yaw.retrieval"))
        self.assertTrue(is_allowed_internal_import("yaw.runtime", "yaw.pipeline"))
        self.assertTrue(is_allowed_internal_import("yaw.optimize", "yaw.datasets"))
        self.assertTrue(is_allowed_internal_import("yaw.mcp_server", "yaw.runtime"))
        self.assertTrue(is_allowed_internal_import("yaw.cli", "yaw.mcp_server"))
        self.assertTrue(is_allowed_internal_import("yaw.__main__", "yaw.cli"))

    def test_zero_hipporag_imports_in_yaw(self):
        """Rule 4: Zero hipporag imports across the entire repository (yaw/, scripts/, tests/)."""
        py_files = (
            list(YAW_DIR.rglob("*.py"))
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
            if (public_api := _extract_public_api_dict(tree)) is not None:
                for key in public_api.keys:
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
        """Rule 5b: Zero cross-object private/protected attribute access across all Python files in yaw/.

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
        self.assertGreater(len(target_files), 0, "No Python files found in yaw/")

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
        backticked_links: list[str] = []
        broken_links: list[str] = []
        target_cache: dict[Path, tuple[set[str], int]] = {}
        for md_file in checked_files:
            rel_source = md_file.relative_to(PROJECT_ROOT)
            raw_text = md_file.read_text(encoding="utf-8")
            for line in _strip_fenced_lines(raw_text):
                backticked_links.extend(
                    f"{rel_source}: backticked doc link {match.group(0)}"
                    for match in _BACKTICKED_LINK_RE.finditer(line)
                )
            prose = _strip_fenced_code_blocks(raw_text)
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
        self.assertEqual(backticked_links, [], "Backticked doc links found:\n" + "\n".join(backticked_links))
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

        domain_rules = ("core.md", "retrieval.md", "pipeline.md", "runtime.md", "qa.md", "tests.md")
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
        self.assertGreater(len(yaw_files), 0, "No Python files found in yaw/")
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
        """Rule 12: Monotonic L0-L6 DAG, zero wrappers/legacy terms, recursive yaw/ & scripts/ allowlists."""
        stdlib = set(sys.stdlib_module_names)
        py_files = sorted(YAW_DIR.rglob("*.py"))
        self.assertGreater(len(py_files), 0, "No Python files found in yaw/")
        self.assertFalse((YAW_DIR / "daemon").exists(), "Retired yaw/daemon/ directory must not exist")

        vocab_violations = _validate_yaw_vocabulary_and_bases(py_files)
        self.assertEqual(vocab_violations, [], "Legacy vocabulary/wrapper violations:\n" + "\n".join(vocab_violations))

        yaw_rules, script_rules = _build_rule12_specs(stdlib)
        violations: list[str] = []
        for py_file in py_files:
            rel = py_file.relative_to(YAW_DIR)
            if rel == Path("__init__.py"):
                self.assertEqual(parse_module_imports(py_file), [])
                self.assertEqual(find_unresolvable_dynamic_imports(py_file), [])
                continue
            domain = rel.parts[0][:-3] if len(rel.parts) == 1 else rel.parts[0]
            self.assertIn(domain, yaw_rules, f"Unmapped file in yaw/: {rel.as_posix()}")
            allowed_top, allowed_yaw, forbidden = yaw_rules[domain]
            allow_getattr = py_file.name == "__init__.py" and is_leaf_package(py_file.parent)
            violations.extend(
                _validate_import_boundary(
                    py_file, allowed_top, allowed_yaw, forbidden, allow_public_api_getattr=allow_getattr
                )
            )

        script_files = sorted((PROJECT_ROOT / "scripts").rglob("*.py"))
        self.assertGreater(len(script_files), 0, "No Python files found in scripts/")
        for script_file in script_files:
            rel_script = script_file.relative_to(PROJECT_ROOT / "scripts").as_posix()
            self.assertIn(rel_script, script_rules, f"Unmapped file in scripts/: {rel_script}")
            allowed_top, allowed_yaw, forbidden = script_rules[rel_script]
            violations.extend(_validate_import_boundary(script_file, allowed_top, allowed_yaw, forbidden))

        cli_files = [p for p in py_files if p.relative_to(YAW_DIR).parts[0] in {"cli.py", "cli"}]
        violations.extend(_validate_cli_lazy_imports(cli_files))
        self.assertEqual(violations, [], "Rule 12 boundary violations detected:\n" + "\n".join(violations))

    def test_zero_suppression_comments_across_repo(self):
        """Rule 12b: Zero # noqa, # ruff: noqa, or C901 suppression comments across yaw/, scripts/, and tests/."""
        py_files = sorted(
            list(YAW_DIR.rglob("*.py"))
            + list((PROJECT_ROOT / "scripts").rglob("*.py"))
            + list((PROJECT_ROOT / "tests").rglob("*.py"))
        )
        self.assertGreater(len(py_files), 0)
        violations = [
            f"{py_file.relative_to(PROJECT_ROOT)}:{lineno}: forbidden suppression comment {comment!r}"
            for py_file in py_files
            for lineno, comment in find_suppression_comments(py_file.read_text(encoding="utf-8"))
        ]
        self.assertEqual(violations, [], "Suppression comments detected:\n" + "\n".join(violations))

    def test_synthetic_dynamic_imports_and_suppression_comment_detection(self):
        """Verify static/dynamic importlib & __import__ detection and tokenize suppression comment scanner."""
        noqa_tok = "noqa"
        c901_tok = "C901"
        sample_src = "\n".join(
            [
                f"x = 1  # {noqa_tok}",
                f"y = 2  # {noqa_tok}: {c901_tok}",
                f"# ruff: {noqa_tok}",
                f'literal_ok = "# {noqa_tok}: {c901_tok}\\n# ruff: {noqa_tok}"',
            ]
        )
        hits = find_suppression_comments(sample_src)
        self.assertEqual([ln for ln, _ in hits], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            static_mod = tmp_path / "static_dyn.py"
            static_mod.write_text(
                "\n".join(
                    [
                        "import importlib",
                        "from importlib import import_module",
                        'm1 = importlib.import_module("yaw.pipeline")',
                        'm2 = import_module("dspy")',
                        'm3 = __import__("sqlite3")',
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_module_imports(static_mod)
            imported_names = {mod for _, mod in parsed}
            self.assertTrue({"yaw.pipeline", "dspy", "sqlite3"}.issubset(imported_names))
            self.assertEqual(find_unresolvable_dynamic_imports(static_mod), [])

            unresolvable_mod = tmp_path / "dyn_var.py"
            unresolvable_mod.write_text(
                "\n".join(
                    [
                        "import importlib",
                        "def load(name: str):",
                        "    return importlib.import_module(name)",
                        "def load2(name: str):",
                        "    return __import__(name)",
                    ]
                ),
                encoding="utf-8",
            )
            unresolvable = find_unresolvable_dynamic_imports(unresolvable_mod)
            self.assertEqual(len(unresolvable), 2)
            self.assertEqual([ln for ln, _ in unresolvable], [3, 5])

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

    def test_wheel_packaging_hygiene_and_bundled_catalog(self):
        """Rule 14: Wheel ships only yaw/ + yaw.core/ with py.typed & baselines.json and lean runtime deps."""
        import json
        import tomllib
        from importlib import resources

        import yaw.datasets as yaw_datasets

        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = data["project"]
        self.assertEqual(set(project["scripts"].keys()), {"yaw"})

        setuptools_cfg = data["tool"]["setuptools"]
        expected_packages = sorted(
            ".".join(p.parent.relative_to(PROJECT_ROOT).parts) for p in YAW_DIR.rglob("__init__.py")
        )
        self.assertEqual(sorted(setuptools_cfg["packages"]), expected_packages)
        self.assertNotIn("py-modules", setuptools_cfg)
        self.assertNotIn("find", setuptools_cfg)
        self.assertLessEqual({"py.typed", "baselines.json"}, set(setuptools_cfg["package-data"]["yaw"]))
        self.assertTrue((YAW_DIR / "py.typed").is_file())
        self.assertTrue((YAW_DIR / "baselines.json").is_file())

        def _req_names(reqs: list[str]) -> set[str]:
            names: set[str] = set()
            for req in reqs:
                match = re.match(r"^\s*([A-Za-z0-9._-]+)", req)
                if match:
                    names.add(match.group(1).lower().replace("_", "-"))
            return names

        runtime_deps = list(project["dependencies"])
        dev_deps = list(data["dependency-groups"]["dev"])
        runtime_names = _req_names(runtime_deps)
        dev_names = _req_names(dev_deps)
        self.assertNotIn("matplotlib", runtime_names)
        self.assertNotIn("psutil", runtime_names)
        self.assertIn("matplotlib", dev_names)
        self.assertIn("psutil", dev_names)
        coco_deps = [d for d in runtime_deps if d.lower().startswith("cocoindex")]
        self.assertTrue(
            any(d.startswith("cocoindex>=1.2.0") for d in coco_deps),
            f"Expected cocoindex>=1.2.0 in project.dependencies, got {coco_deps}",
        )

        catalog_res = resources.files("yaw").joinpath("baselines.json")
        self.assertTrue(catalog_res.is_file())
        catalog = json.loads(catalog_res.read_text(encoding="utf-8"))
        self.assertEqual(tuple(catalog["datasets"].keys()), yaw_datasets.DATASETS)
        self.assertEqual(dict(catalog["baselines"]), yaw_datasets.BASELINES)

        self.assertFalse((PROJECT_ROOT / "benchmark.py").exists())
        self.assertTrue((PROJECT_ROOT / "scripts" / "benchmark.py").is_file())
        self.assertTrue((PROJECT_ROOT / "scripts" / "download_datasets.py").is_file())
        self.assertFalse((PROJECT_ROOT / "scripts" / "preflight.py").exists())
        self.assertFalse((PROJECT_ROOT / "referencces").exists())
