"""Check YAW's local documentation without importing the application."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TypedDict
from urllib.parse import unquote, urlsplit

ROOT_DOCS = (
    "AGENTS.md",
    "FOUNDATIONAL.md",
    "README.md",
    "TEST_INFRA.md",
    "TEST_READY.md",
)
REFERENCES = {
    "alignment-audit-12-10-2026.md",
    "alignment-sprint-for-e5s2.md",
    "cocoindex.md",
    "cocoindex-neo4j.md",
    "efficient-ppr-algorithms.md",
    "karpathy-llm-wiki.md",
    "spacy-llm-api.md",
    "spacy-llm-docs.md",
}
EXPECTED_CASES = {f"E78-{number:02}" for number in range(1, 21)}
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
LINE_ANCHOR = re.compile(r"L\d+(?:-L\d+)?")


class Report(TypedDict):
    documents: int
    local_links: int
    acceptance_cases: int
    mapped_cases: int
    packages: int
    errors: list[str]


def unfenced(text: str) -> str:
    """Keep line positions while excluding fenced code examples."""
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    return "\n".join(lines)


def heading_ids(text: str) -> set[str]:
    result = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", unfenced(text), re.MULTILINE):
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def check(root: Path) -> Report:
    root = root.resolve()
    paths = [root / name for name in ROOT_DOCS]
    paths += sorted((root / "docs").rglob("*.md"))
    errors: list[str] = []
    texts: dict[Path, str] = {}
    local_links = 0
    for path in paths:
        if not path.is_file():
            errors.append(f"Missing document: {path.relative_to(root)}")
            continue
        texts[path] = path.read_text(encoding="utf-8-sig")
    anchors = {path: heading_ids(text) for path, text in texts.items()}
    for path, text in texts.items():
        rel = path.relative_to(root)
        plain = unfenced(text)
        for match in LINK.finditer(plain):
            destination = match[1].strip().strip("<>")
            line = plain.count("\n", 0, match.start()) + 1
            location = f"{rel.as_posix()}:{line}"
            scheme = urlsplit(destination).scheme
            if scheme:
                if scheme == "file" or len(scheme) == 1:
                    errors.append(f"{location}: nonportable link {destination}")
                continue
            filename, _, fragment = unquote(destination).partition("#")
            target = (path.parent / filename).resolve() if filename else path
            local_links += 1
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f"{location}: missing/local escape {destination}")
                continue
            if (
                fragment
                and target.suffix == ".md"
                and not LINE_ANCHOR.fullmatch(fragment)
            ):
                if target not in anchors:
                    anchors[target] = heading_ids(
                        target.read_text(encoding="utf-8-sig")
                    )
                if fragment not in anchors[target]:
                    errors.append(f"{location}: unknown heading {destination}")
        if path.name in REFERENCES or "reviews" in rel.parts:
            continue
        width = None
        for number, line in enumerate(
            unfenced(text.split("<details>")[0]).splitlines(), 1
        ):
            if line.startswith("|") and line.endswith("|"):
                cells = len(re.split(r"(?<!\\)\|", line)) - 2
                if width is not None and cells != width:
                    errors.append(f"{rel.as_posix()}:{number}: table column mismatch")
                width = cells
            else:
                width = None
    stories = texts.get(root / "docs/USER_STORIES.md", "")
    cases = re.findall(r"^### (E78-\d{2}) ", stories, re.MULTILINE)
    if len(cases) != len(EXPECTED_CASES) or set(cases) != EXPECTED_CASES:
        errors.append("USER_STORIES must declare E78-01 through E78-20 exactly once")
    test_plan = texts.get(root / "TEST_INFRA.md", "")
    target_map = test_plan.partition("## Test-target map\n")[2].split("\n## ")[0]
    mapped = set(re.findall(r"E78-\d{2}", target_map))
    if mapped != EXPECTED_CASES:
        errors.append("TEST_INFRA test-target map must cover every declared E78 case")
    epic = texts.get(root / "docs/epic7.8-open-ie-ontology-alignment.md", "")
    packages = re.findall(r"^- \[[ x]\] \*\*(7\.8\.\d)\b", epic, re.MULTILINE)
    expected_packages = {f"7.8.{number}" for number in range(8)}
    if len(packages) != 8 or set(packages) != expected_packages:
        errors.append("Epic checklist must declare 7.8.0 through 7.8.7 exactly once")
    return {
        "documents": len(texts),
        "local_links": local_links,
        "acceptance_cases": len(cases),
        "mapped_cases": len(mapped),
        "packages": len(packages),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the check report as JSON"
    )
    args = parser.parse_args()
    report = check(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        state = "FAIL" if report["errors"] else "OK"
        print(
            f"{state}: {report['documents']} documents, "
            f"{report['local_links']} local links, "
            f"{report['mapped_cases']}/{report['acceptance_cases']} mapped E78 cases, "
            f"{report['packages']} epic packages"
        )
        for error in report["errors"]:
            print(error)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
