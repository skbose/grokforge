"""Synthetic log generation using Faker + known Grok patterns."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import sys
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from faker import Faker

from grokforge.dataset.parse_patterns import load_all_patterns

fake = Faker()
_rng = random.Random()

# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

@dataclass
class Literal:
    text: str

@dataclass
class Ref:
    pattern_name: str
    field_name: str | None

@dataclass
class Alternation:
    branches: list[list[Token]]

@dataclass
class Optional:
    tokens: list[Token]

Token = Literal | Ref | Alternation | Optional

# ---------------------------------------------------------------------------
# Primitive generators — leaf Grok pattern names mapped to Faker calls
# ---------------------------------------------------------------------------

PRIMITIVE_GENERATORS: dict[str, Callable[[], str]] = {
    "IPV4":             lambda: fake.ipv4_public(),
    "IPV6":             lambda: fake.ipv6(),
    "IP":               lambda: _rng.choice([fake.ipv4_public(), fake.ipv6()]),
    "HOSTNAME":         lambda: fake.hostname(),
    "IPORHOST":         lambda: _rng.choice([fake.ipv4_public(), fake.hostname()]),
    "HOSTPORT":         lambda: f"{fake.ipv4_public()}:{_rng.randint(1, 65535)}",
    "USERNAME":         lambda: fake.user_name(),
    "USER":             lambda: fake.user_name(),
    "EMAILLOCALPART":   lambda: fake.user_name(),
    "EMAILADDRESS":     lambda: fake.email(),
    "INT":              lambda: str(_rng.randint(-9999, 9999)),
    "BASE10NUM":        lambda: str(_rng.randint(0, 9999)),
    "NUMBER":           lambda: str(_rng.randint(0, 9999)),
    "POSINT":           lambda: str(_rng.randint(1, 9999)),
    "NONNEGINT":        lambda: str(_rng.randint(0, 9999)),
    "BASE16NUM":        lambda: hex(_rng.randint(0, 0xFFFF)),
    "WORD":             lambda: fake.word(),
    "NOTSPACE":         lambda: fake.slug(),
    "SPACE":            lambda: " ",
    "DATA":             lambda: fake.sentence(nb_words=5).rstrip("."),
    "GREEDYDATA":       lambda: fake.sentence().rstrip("."),
    "QUOTEDSTRING":     lambda: f'"{fake.sentence().rstrip(".")}"',
    "QS":               lambda: f'"{fake.sentence().rstrip(".")}"',
    "UUID":             lambda: str(uuid.uuid4()),
    "MAC":              lambda: fake.mac_address(),
    "CISCOMAC":         lambda: ".".join(f"{_rng.randint(0, 0xFFFF):04x}" for _ in range(3)),
    "WINDOWSMAC":       lambda: "-".join(f"{_rng.randint(0, 0xFF):02X}" for _ in range(6)),
    "COMMONMAC":        lambda: fake.mac_address(),
    "MONTH":            lambda: fake.date_time_this_year().strftime("%b"),
    "MONTHNUM":         lambda: f"{_rng.randint(1, 12):02d}",
    "MONTHNUM2":        lambda: f"{_rng.randint(1, 12):02d}",
    "MONTHDAY":         lambda: f"{_rng.randint(1, 28):02d}",
    "DAY":              lambda: fake.date_time_this_year().strftime("%a"),
    "YEAR":             lambda: str(_rng.randint(2020, 2026)),
    "HOUR":             lambda: f"{_rng.randint(0, 23):02d}",
    "MINUTE":           lambda: f"{_rng.randint(0, 59):02d}",
    "SECOND":           lambda: f"{_rng.randint(0, 59):02d}",
    "TIME":             lambda: fake.time(),
    "DATE_US":          lambda: fake.date_time_this_year().strftime("%m/%d/%y"),
    "DATE_EU":          lambda: fake.date_time_this_year().strftime("%d.%m.%y"),
    "DATESTAMP":        lambda: fake.date_time_this_year().strftime("%m/%d/%y %H:%M:%S"),
    "TZ":               lambda: _rng.choice(["UTC", "EST", "PST", "CST", "GMT"]),
    "ISO8601_TIMEZONE": lambda: _rng.choice(["+00:00", "-05:00", "+05:30", "Z"]),
    "TIMESTAMP_ISO8601":lambda: fake.date_time_this_year().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    "HTTPDATE":         lambda: fake.date_time_this_year().strftime("%d/%b/%Y:%H:%M:%S +0000"),
    "SYSLOGTIMESTAMP":  lambda: fake.date_time_this_year().strftime("%b %e %H:%M:%S"),
    "LOGLEVEL":         lambda: _rng.choice(["INFO", "WARN", "ERROR", "DEBUG", "TRACE", "FATAL"]),
    "CRON_ACTION":      lambda: _rng.choice(["CMD", "BEGIN EDIT", "END EDIT", "MAIL", "STARTUP", "REPLACE"]),
    "PROG":             lambda: _rng.choice(["sshd", "cron", "kernel", "systemd", "sudo", "dbus-daemon", "NetworkManager"]),
    "URIPROTO":         lambda: _rng.choice(["http", "https", "ftp"]),
    "URIPATH":          lambda: "/" + fake.uri_path(),
    "URIPARAM":         lambda: "?" + "&".join(f"{fake.word()}={fake.word()}" for _ in range(_rng.randint(1, 3))),
    "URIPATHPARAM":     lambda: "/" + fake.uri_path() + "?" + fake.word() + "=" + fake.word(),
    "URI":              lambda: fake.uri(),
    "PATH":             lambda: fake.file_path(),
    "UNIXPATH":         lambda: fake.file_path(),
    "WINPATH":          lambda: "C:\\" + "\\".join(fake.words(nb=_rng.randint(1, 3))),
    "TTY":              lambda: f"/dev/pts/{_rng.randint(0, 9)}",
}

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _find_group_end(s: str, start: int) -> int:
    """Return index of the `)` that closes the `(` at s[start]."""
    depth = 0
    i = start
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s) - 1


def _split_alternation(s: str) -> list[str] | None:
    """Split s on top-level `|`. Returns list of branches, or None if no top-level `|`."""
    branches: list[str] = []
    depth = 0
    current_start = 0
    i = 0
    found = False
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c in ("(", "["):
            depth += 1
        elif c in (")", "]"):
            depth -= 1
        elif c == "|" and depth == 0:
            branches.append(s[current_start:i])
            current_start = i + 1
            found = True
        i += 1
    if not found:
        return None
    branches.append(s[current_start:])
    return branches


def tokenize(s: str) -> list[Token]:
    # Handle top-level alternation (e.g. HTTPDUSER body: %{EMAILADDRESS}|%{USER})
    branches = _split_alternation(s)
    if branches is not None:
        return [Alternation([_tokenize_inner(b) for b in branches])]
    return _tokenize_inner(s)


def _tokenize_inner(s: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    literal_buf = ""

    def flush_literal() -> None:
        nonlocal literal_buf
        if literal_buf:
            tokens.append(Literal(literal_buf))
            literal_buf = ""

    while i < len(s):
        # Pattern reference %{NAME} or %{NAME:field}
        if s[i] == "%" and i + 1 < len(s) and s[i + 1] == "{":
            flush_literal()
            try:
                end = s.index("}", i + 2)
            except ValueError:
                # Unclosed %{ — treat rest as literal
                literal_buf += s[i:]
                break
            inner = s[i + 2:end]
            if ":" in inner:
                name, field = inner.split(":", 1)
            else:
                name, field = inner, None
            tokens.append(Ref(name, field))
            i = end + 1

        # Non-capturing group (?:...)
        elif s[i:i+3] == "(?:":
            flush_literal()
            group_end = _find_group_end(s, i)
            inner = s[i + 3:group_end]
            is_optional = group_end + 1 < len(s) and s[group_end + 1] == "?"
            branches = _split_alternation(inner)
            if branches is not None:
                branch_tokens = [tokenize(b) for b in branches]
                if is_optional:
                    tokens.append(Optional([Alternation(branch_tokens)]))
                else:
                    tokens.append(Alternation(branch_tokens))
            else:
                inner_tokens = tokenize(inner)
                if is_optional:
                    tokens.append(Optional(inner_tokens))
                else:
                    tokens.extend(inner_tokens)
            i = group_end + 1 + (1 if is_optional else 0)

        # Character class [...] — treat as generic WORD
        elif s[i] == "[":
            flush_literal()
            try:
                end = s.index("]", i + 1)
            except ValueError:
                # Unclosed [ — skip to end
                break
            tokens.append(Ref("WORD", None))
            i = end + 1

        # Backslash escape — unescape to literal
        elif s[i] == "\\" and i + 1 < len(s):
            literal_buf += s[i + 1]
            i += 2

        # Quantifiers and anchors — drop
        elif s[i] in ("+", "*", "?", "^", "$"):
            i += 1

        # Everything else — accumulate as literal
        else:
            literal_buf += s[i]
            i += 1

    flush_literal()
    return tokens


# ---------------------------------------------------------------------------
# Pattern expansion
# ---------------------------------------------------------------------------

def _is_composite(pattern_body: str) -> bool:
    return "%{" in pattern_body


def _expand_tokens(
    tokens: list[Token],
    all_patterns: dict[str, str],
    depth: int = 0,
) -> list[Token]:
    if depth > 20:
        return [Literal("")]
    result: list[Token] = []
    for token in tokens:
        if isinstance(token, Ref):
            name = token.pattern_name
            if name in PRIMITIVE_GENERATORS:
                result.append(token)
            elif name in all_patterns and _is_composite(all_patterns[name]):
                expanded = tokenize(all_patterns[name])
                result.extend(_expand_tokens(expanded, all_patterns, depth + 1))
            elif name in all_patterns:
                result.append(Ref("WORD", token.field_name))
            else:
                result.append(Literal(""))
        elif isinstance(token, Alternation):
            result.append(Alternation([
                _expand_tokens(branch, all_patterns, depth)
                for branch in token.branches
            ]))
        elif isinstance(token, Optional):
            result.append(Optional(
                _expand_tokens(token.tokens, all_patterns, depth)
            ))
        else:
            result.append(token)
    return result


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_tokens(tokens: list[Token]) -> str:
    parts: list[str] = []
    for token in tokens:
        if isinstance(token, Literal):
            parts.append(token.text)
        elif isinstance(token, Ref):
            gen = PRIMITIVE_GENERATORS.get(token.pattern_name)
            parts.append(gen() if gen else "")
        elif isinstance(token, Alternation):
            branch = _rng.choice(token.branches)
            parts.append(_generate_tokens(branch))
        elif isinstance(token, Optional):
            if _rng.random() < 0.5:
                parts.append(_generate_tokens(token.tokens))
    return "".join(parts)


def _generate_tokens_with_pattern(tokens: list[Token]) -> tuple[str, str]:
    """Generate (log_fragment, pattern_fragment) in lockstep.

    The pattern fragment uses only primitive %{NAME:field} refs — no composite
    pattern names — and reflects exactly the branches/optionals that were chosen
    during generation.
    """
    log_parts: list[str] = []
    pat_parts: list[str] = []
    for token in tokens:
        if isinstance(token, Literal):
            log_parts.append(token.text)
            pat_parts.append(token.text)
        elif isinstance(token, Ref):
            gen = PRIMITIVE_GENERATORS.get(token.pattern_name)
            log_parts.append(gen() if gen else "")
            field = f":{token.field_name}" if token.field_name else ""
            pat_parts.append(f"%{{{token.pattern_name}{field}}}")
        elif isinstance(token, Alternation):
            branch = _rng.choice(token.branches)
            log_frag, pat_frag = _generate_tokens_with_pattern(branch)
            log_parts.append(log_frag)
            pat_parts.append(pat_frag)
        elif isinstance(token, Optional):
            if _rng.random() < 0.5:
                log_frag, pat_frag = _generate_tokens_with_pattern(token.tokens)
                log_parts.append(log_frag)
                pat_parts.append(pat_frag)
    return "".join(log_parts), "".join(pat_parts)


def generate_log(pattern_name: str, all_patterns: dict[str, str]) -> str:
    if pattern_name in PRIMITIVE_GENERATORS:
        return PRIMITIVE_GENERATORS[pattern_name]()
    raw = all_patterns[pattern_name]
    tokens = tokenize(raw)
    expanded = _expand_tokens(tokens, all_patterns)
    return _generate_tokens(expanded)


def generate_log_and_pattern(
    pattern_name: str, all_patterns: dict[str, str]
) -> tuple[str, str]:
    """Return (log_line, primitive_pattern) for the given composite pattern name."""
    if pattern_name in PRIMITIVE_GENERATORS:
        value = PRIMITIVE_GENERATORS[pattern_name]()
        return value, f"%{{{pattern_name}}}"
    raw = all_patterns[pattern_name]
    tokens = tokenize(raw)
    expanded = _expand_tokens(tokens, all_patterns)
    return _generate_tokens_with_pattern(expanded)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_PATTERNS = ["HTTPD_COMMONLOG", "HTTPD_COMBINEDLOG", "SYSLOGLINE", "CRONLOG"]


@dataclasses.dataclass(frozen=True)
class LogSample:
    log: str
    pattern_name: str  # composite source pattern (e.g. HTTPD_COMBINEDLOG)
    pattern: str       # primitive-expanded grok pattern string


def generate_samples(
    count: int,
    all_patterns: dict[str, str],
    pattern_names: list[str] | None = None,
    seed: int | None = None,
) -> Iterator[LogSample]:
    names = pattern_names or _DEFAULT_PATTERNS
    missing = [n for n in names if n not in all_patterns and n not in PRIMITIVE_GENERATORS]
    if missing:
        raise ValueError(f"Unknown pattern names: {missing}")
    if seed is not None:
        _rng.seed(seed)
        fake.seed_instance(seed)
    for i in range(count):
        name = names[i % len(names)]
        log, pattern = generate_log_and_pattern(name, all_patterns)
        yield LogSample(log=log, pattern_name=name, pattern=pattern)


def write_jsonl(samples: Iterable[LogSample], out_path: pathlib.Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(dataclasses.asdict(sample)) + "\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic log samples")
    parser.add_argument("--out", default="data/generated/logs.jsonl", help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--patterns", nargs="*", default=None, help="Pattern names (default: 4 starter patterns)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--list-patterns", action="store_true", help="List available pattern names and exit")
    args = parser.parse_args(argv)

    all_patterns = load_all_patterns()

    if args.list_patterns:
        composite = sorted(k for k in all_patterns if k not in PRIMITIVE_GENERATORS)
        primitive = sorted(PRIMITIVE_GENERATORS)
        print(f"Composite patterns ({len(composite)}) — use with --patterns:")
        for name in composite:
            print(f"  {name}")
        print(f"\nPrimitive patterns ({len(primitive)}) — use with --patterns:")
        for name in primitive:
            print(f"  {name}")
        return

    pattern_names = args.patterns or _DEFAULT_PATTERNS
    missing = [n for n in pattern_names if n not in all_patterns and n not in PRIMITIVE_GENERATORS]
    if missing:
        print(f"Error: unknown patterns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Warn about any expanded primitive refs not covered by PRIMITIVE_GENERATORS
    uncovered: set[str] = set()
    for name in pattern_names:
        if name in all_patterns:
            expanded = _expand_tokens(tokenize(all_patterns[name]), all_patterns)
            def _collect_refs(tokens: list[Token]) -> None:
                for t in tokens:
                    if isinstance(t, Ref) and t.pattern_name not in PRIMITIVE_GENERATORS:
                        uncovered.add(t.pattern_name)
                    elif isinstance(t, Alternation):
                        for b in t.branches:
                            _collect_refs(b)
                    elif isinstance(t, Optional):
                        _collect_refs(t.tokens)
            _collect_refs(expanded)
    if uncovered:
        print(f"Warning: unmapped primitives (will emit empty string): {uncovered}", file=sys.stderr)

    out_path = pathlib.Path(args.out)

    def _samples_with_progress() -> Iterator[LogSample]:
        for i, sample in enumerate(generate_samples(args.count, all_patterns, pattern_names, args.seed)):
            if (i + 1) % 100 == 0:
                print(f"\r{i + 1}/{args.count}", end="", file=sys.stderr)
            yield sample
        print(f"\r{args.count}/{args.count}", file=sys.stderr)

    written = write_jsonl(_samples_with_progress(), out_path)
    print(f"Wrote {written} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
