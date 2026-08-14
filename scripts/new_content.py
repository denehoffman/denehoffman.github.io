#!/usr/bin/env python3
"""Create a blog post or project using this site's conventions."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA = ROOT / "data" / "projects.json"


def valid_slug(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise argparse.ArgumentTypeError("use lowercase letters, numbers, and single hyphens")
    return value


def create_post(slug: str, title: str) -> Path:
    today = date.today().isoformat()
    directory = ROOT / "content" / "blog" / f"{today}-{slug}"
    path = directory / "index.md"
    if path.exists():
        raise FileExistsError(path.relative_to(ROOT))
    directory.mkdir(parents=True, exist_ok=False)
    path.write_text(
        f'''+++
title = {json.dumps(title)}
description = "TODO: describe this post"
date = {today}
draft = true
[taxonomies]
tags = []
[extra]
scripts = ["article-reading.js"]
+++

Start writing here.
''',
        encoding="utf-8",
    )
    return path


def create_project(slug: str, title: str, summary: str) -> Path:
    directory = ROOT / "content" / "projects" / slug
    path = directory / "index.md"
    if path.exists():
        raise FileExistsError(path.relative_to(ROOT))

    data = json.loads(PROJECT_DATA.read_text(encoding="utf-8"))
    if slug in data["projects"]:
        raise ValueError(f"project {slug!r} already exists in data/projects.json")
    data["order"].append(slug)
    data["projects"][slug] = {
        "title": title,
        "summary": summary,
        "card_summary": "TODO: short project summary",
        "tags": ["TODO"],
        "repository": f"https://github.com/denehoffman/{slug}",
    }

    directory.mkdir(parents=True, exist_ok=False)
    path.write_text(
        f'''+++
title = {json.dumps(title)}
+++

{{{{ project_header(project={json.dumps(slug)}) }}}}

Start writing here.
''',
        encoding="utf-8",
    )
    PROJECT_DATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    post = subparsers.add_parser("post")
    post.add_argument("slug", type=valid_slug)
    post.add_argument("--title", required=True)

    project = subparsers.add_parser("project")
    project.add_argument("slug", type=valid_slug)
    project.add_argument("--title", required=True)
    project.add_argument("--summary", required=True)

    args = parser.parse_args()
    try:
        if args.kind == "post":
            path = create_post(args.slug, args.title)
        else:
            path = create_project(args.slug, args.title, args.summary)
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Created {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
