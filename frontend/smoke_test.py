#!/usr/bin/env python3
"""
Frontend smoke test: verifies every nav item has a matching page div
and every loader entry references a page div that exists.

Run:  python frontend/smoke_test.py
Exit: 0 = pass, 1 = failures found
"""
import re
import sys
from pathlib import Path

HTML = Path(__file__).parent / "index.html"


def main() -> int:
    src = HTML.read_text()
    errors: list[str] = []

    # 1. Extract nav targets: showPage('xxx') from onclick attributes
    nav_targets = set(re.findall(r"""onclick="showPage\('(\w+)'\)""", src))

    # 2. Extract page divs: id="page-xxx"
    page_divs = set(re.findall(r'id="page-(\w+)"', src))

    # 3. Extract loader keys from the loaders = { ... } block
    loader_match = re.search(
        r"const loaders\s*=\s*\{(.*?)\};", src, re.DOTALL
    )
    loader_keys: set[str] = set()
    if loader_match:
        block = loader_match.group(1)
        loader_keys = set(re.findall(r"(\w+)\s*:", block))

    # ── Checks ────────────────────────────────────────────────────────────────

    # Every nav target must have a page div
    for name in sorted(nav_targets):
        if name not in page_divs:
            errors.append(f"NAV→PAGE MISSING: showPage('{name}') has no <div id=\"page-{name}\">")

    # Every loader key must have a page div
    for name in sorted(loader_keys):
        if name not in page_divs:
            errors.append(f"LOADER→PAGE MISSING: loaders['{name}'] has no <div id=\"page-{name}\">")

    # Every page div that has a nav entry should ideally have a loader
    # (advisory, not fatal — some pages are static)
    no_loader = sorted(nav_targets - loader_keys)
    if no_loader:
        print(f"  INFO: nav targets without loaders (static pages): {', '.join(no_loader)}")

    # ── Report ────────────────────────────────────────────────────────────────
    if errors:
        print(f"FAIL — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print(f"PASS — {len(nav_targets)} nav targets, {len(page_divs)} page divs, {len(loader_keys)} loaders. All wired up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
