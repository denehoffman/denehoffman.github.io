#!/usr/bin/env python3
"""Validate the site's structured content and inexpensive build invariants."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
PROJECT_DATA = ROOT / "data" / "projects.json"
MAX_ASSET_BYTES = 8 * 1024 * 1024
REQUIRED_PROJECT_FIELDS = {"title", "summary", "card_summary", "tags", "repository"}
ALLOWED_PROJECT_STATUSES = {"Experimental", "Archived"}
SCRIPT_PATTERN = re.compile(r"scripts\s*=\s*\[(?P<scripts>[^]]*)]", re.DOTALL)
SCRIPT_NAME_PATTERN = re.compile(r'"([^"]+\.js)"')


def error(message: str, errors: list[str]) -> None:
    errors.append(message)


def validate_projects(errors: list[str]) -> None:
    try:
        data = json.loads(PROJECT_DATA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        error(f"{PROJECT_DATA.relative_to(ROOT)}: {exc}", errors)
        return

    records = data.get("projects", {})
    order = data.get("order", [])
    featured = data.get("featured", [])
    if len(order) != len(set(order)):
        error("data/projects.json: order contains duplicate slugs", errors)
    if set(order) != set(records):
        error("data/projects.json: order must contain every project exactly once", errors)
    if not set(featured).issubset(records):
        error("data/projects.json: featured contains an unknown project", errors)

    for slug, project in records.items():
        missing = REQUIRED_PROJECT_FIELDS - project.keys()
        if missing:
            error(f"project {slug!r}: missing {', '.join(sorted(missing))}", errors)
        if not isinstance(project.get("tags"), list) or not project.get("tags"):
            error(f"project {slug!r}: tags must be a non-empty list", errors)
        status = project.get("status")
        if status is not None and status not in ALLOWED_PROJECT_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_PROJECT_STATUSES))
            error(f"project {slug!r}: status must be omitted or one of {allowed}", errors)
        page = CONTENT / "projects" / slug / "index.md"
        if not page.exists():
            error(f"project {slug!r}: missing {page.relative_to(ROOT)}", errors)
            continue
        expected = f'project_header(project="{slug}")'
        if expected not in page.read_text(encoding="utf-8"):
            error(f"{page.relative_to(ROOT)}: expected {{{{ {expected} }}}}", errors)


def validate_scripts(errors: list[str]) -> None:
    paths = [ROOT / "config.toml", *CONTENT.rglob("*.md")]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in SCRIPT_PATTERN.finditer(text):
            for script in SCRIPT_NAME_PATTERN.findall(match.group("scripts")):
                if not (ROOT / "static" / script).is_file():
                    error(f"{path.relative_to(ROOT)}: static/{script} does not exist", errors)


def validate_assets(errors: list[str]) -> None:
    oversized = []
    for base in (CONTENT, ROOT / "static"):
        for path in base.rglob("*"):
            if path.is_file() and path.stat().st_size > MAX_ASSET_BYTES:
                oversized.append((path.stat().st_size, path))
    for size, path in sorted(oversized, reverse=True):
        error(
            f"{path.relative_to(ROOT)}: {size / 1024 / 1024:.1f} MiB exceeds the 8 MiB asset budget",
            errors,
        )


def main() -> int:
    errors: list[str] = []
    validate_projects(errors)
    validate_scripts(errors)
    validate_assets(errors)
    if errors:
        print("Site validation failed:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print("Site structure and asset budgets are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
