#!/usr/bin/env python3
"""Refresh the site's Google Scholar citation snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROFILE_ID = "39-XmFUAAAAJ"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "scholar.json"


class ScholarProfileParser(HTMLParser):
    """Collect the stable metric and graph elements from a Scholar profile."""

    targets = {
        "gsc_rsb_std": "metrics",
        "gsc_g_t": "years",
        "gsc_g_a": "citations",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: dict[str, list[str]] = {
            "metrics": [],
            "years": [],
            "citations": [],
        }
        self.capture: tuple[str, str, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "") or ""
        for class_name, target in self.targets.items():
            if class_name in classes.split():
                self.capture = (target, tag, [])
                return

    def handle_data(self, data: str) -> None:
        if self.capture is not None:
            self.capture[2].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.capture is None or self.capture[1] != tag:
            return
        target, _, chunks = self.capture
        value = "".join(chunks).strip()
        if value:
            self.values[target].append(value)
        self.capture = None


def fetch_profile(profile_id: str) -> str:
    url = (
        "https://scholar.google.com/citations"
        f"?user={profile_id}&hl=en&pagesize=100"
    )
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def parse_profile(html: str, profile_id: str) -> dict[str, object]:
    parser = ScholarProfileParser()
    parser.feed(html)

    try:
        metrics = [int(value.replace(",", "")) for value in parser.values["metrics"]]
        years = [int(value) for value in parser.values["years"]]
        yearly_citations = [
            int(value.replace(",", "")) for value in parser.values["citations"]
        ]
    except ValueError as error:
        raise ValueError("Scholar returned malformed citation data") from error

    if len(metrics) < 6 or not years or len(years) != len(yearly_citations):
        raise ValueError(
            "Scholar metrics were not found. Google may have returned a CAPTCHA; "
            "the existing snapshot was left untouched."
        )

    maximum = max(yearly_citations) or 1
    history = [
        {
            "year": year,
            "citations": citations,
            "height": f"{citations / maximum * 100:.2f}%",
        }
        for year, citations in zip(years, yearly_citations, strict=True)
    ]
    today = date.today()

    return {
        "profile_id": profile_id,
        "profile_url": (
            "https://scholar.google.com/citations"
            f"?user={profile_id}&hl=en"
        ),
        "updated": f"{today.day} {today.strftime('%B %Y')}",
        "citations": metrics[0],
        "h_index": metrics[2],
        "i10_index": metrics[4],
        "history": history,
    }


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    argument_parser.add_argument(
        "--html",
        type=Path,
        help="Parse a saved Scholar profile instead of fetching it",
    )
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the refreshed data without writing it",
    )
    args = argument_parser.parse_args()

    try:
        html = args.html.read_text(encoding="utf-8") if args.html else fetch_profile(args.profile_id)
        data = parse_profile(html, args.profile_id)
    except (HTTPError, URLError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.dry_run:
        print(rendered, end="")
    else:
        OUTPUT_PATH.write_text(rendered, encoding="utf-8")
        print(f"Updated {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
