"""Compute and display metrics from eval_api.py output JSONL files."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict


def _load(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "samples": n,
        "em": sum(r["exact_match"] for r in rows) / n * 100,
        "fm": sum(r["functional_match"] for r in rows) / n * 100,
        "avg_lat": sum(r["latency_s"] for r in rows) / n,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def _print_table(headers: list[str], rows: list[list[str]], col_sep: str = "  ") -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = col_sep.join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(col_sep.join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def report(paths: list[pathlib.Path], by_pattern: bool) -> None:
    overall_headers = ["model", "samples", "EM%", "FM%", "avg_lat", "errors"]
    overall_rows: list[list[str]] = []

    per_pattern_data: dict[str, dict[str, dict]] = {}  # model -> pattern -> summary

    for path in paths:
        model = path.stem
        rows = _load(path)
        if not rows:
            print(f"Warning: {path} is empty, skipping.", file=sys.stderr)
            continue

        s = _summarise(rows)
        overall_rows.append([
            model,
            str(s["samples"]),
            f"{s['em']:.1f}",
            f"{s['fm']:.1f}",
            f"{s['avg_lat']:.2f}s",
            str(s["errors"]),
        ])

        if by_pattern:
            by_pname: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_pname[r.get("pattern_name", "unknown")].append(r)
            per_pattern_data[model] = {
                pname: _summarise(prows) for pname, prows in sorted(by_pname.items())
            }

    print("\n=== Overall ===\n")
    _print_table(overall_headers, overall_rows)

    if by_pattern and per_pattern_data:
        all_patterns = sorted({p for m in per_pattern_data.values() for p in m})
        models = [p.stem for p in paths if p.stem in per_pattern_data]

        for metric, label in [("em", "EM%"), ("fm", "FM%")]:
            print(f"\n=== By pattern — {label} ===\n")
            headers = ["pattern"] + models
            table_rows: list[list[str]] = []
            for pname in all_patterns:
                row = [pname]
                for model in models:
                    s = per_pattern_data[model].get(pname)
                    row.append(f"{s[metric]:.1f}" if s else "-")
                table_rows.append(row)
            _print_table(headers, table_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from eval_api.py output JSONL files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more eval output JSONL files (filename stem is used as model name)",
    )
    parser.add_argument(
        "--by-pattern",
        action="store_true",
        help="Also break down EM%% and FM%% by pattern_name",
    )
    args = parser.parse_args(argv)

    paths = [pathlib.Path(f) for f in args.files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    report(paths, by_pattern=args.by_pattern)


if __name__ == "__main__":
    main()
