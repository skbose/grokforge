"""Parse logstash-patterns-core into usable (name, pattern) pairs."""

import pathlib
import re

_NAME_RE = re.compile(r"^[A-Z0-9_]+$")
_DEFAULT_PATTERNS_DIR = (
    pathlib.Path(__file__).parents[2]
    / "data"
    / "raw"
    / "logstash-patterns-core"
    / "patterns"
    / "legacy"
)


def parse_pattern_file(path: pathlib.Path) -> dict[str, str]:
    patterns: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, body = parts
        if not _NAME_RE.match(name):
            continue
        patterns[name] = body
    return patterns


def load_all_patterns(
    patterns_dir: pathlib.Path | str | None = None,
) -> dict[str, str]:
    directory = pathlib.Path(patterns_dir) if patterns_dir else _DEFAULT_PATTERNS_DIR
    merged: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            for name, body in parse_pattern_file(path).items():
                merged.setdefault(name, body)
    return merged


if __name__ == "__main__":
    patterns = load_all_patterns()
    print(f"Loaded {len(patterns)} patterns")
    for name, body in list(patterns.items())[:5]:
        print(f"  {name}: {body[:60]}")
