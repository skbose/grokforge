"""Evaluate foundational models on grok pattern generation via LangChain API."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

load_dotenv()

from pygrok import Grok

SYSTEM_PROMPT = """\
You are an expert in Logstash Grok pattern syntax.
Given a raw log line, output the single Grok pattern that matches it exactly.

Rules:
- Use %{PATTERN_NAME:field_name} for named captures and %{PATTERN_NAME} for unnamed ones.
- Preserve every literal character (spaces, brackets, quotes, dashes) exactly as they appear.
- Output ONLY the Grok pattern string — no explanation, no markdown fences, no extra text.

Common pattern names: IPV4, IPV6, IPORHOST, HOSTNAME, USERNAME, USER, INT, NUMBER,
NONNEGINT, POSINT, WORD, DATA, GREEDYDATA, QS, QUOTEDSTRING, HTTPDATE,
SYSLOGTIMESTAMP, LOGLEVEL, URI, URIPATH, URIPARAM, MAC, PATH, UUID."""

USER_TEMPLATE = "Log line:\n{log}\n\nGrok pattern:"


def _make_model(model_id: str, temperature: float = 0.0):
    """Return a LangChain chat model for the given model identifier."""
    if model_id.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, temperature=temperature)
    if model_id.startswith(("gpt-", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=temperature)
    if model_id.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature)
    raise ValueError(
        f"Cannot infer provider for model '{model_id}'. "
        "Prefix must start with 'claude', 'gpt-'/'o1'/'o3'/'o4', or 'gemini'."
    )


@dataclass
class EvalResult:
    log: str
    pattern_name: str
    reference: str
    predicted: str
    exact_match: bool
    functional_match: bool
    latency_s: float
    error: str = ""


def _functional_match(log: str, pattern: str) -> bool:
    """Return True if pattern matches the full log line via pygrok."""
    try:
        grok = Grok(f"^{pattern}$")
        return grok.match(log) is not None
    except Exception:
        return False


def _run_sample(chain, log: str) -> tuple[str, float]:
    """Invoke the chain and return (predicted_pattern, latency_seconds)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    t0 = time.perf_counter()
    response = chain.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=USER_TEMPLATE.format(log=log))]
    )
    latency = time.perf_counter() - t0
    predicted = response.content.strip()
    # Strip accidental markdown fences if present
    if predicted.startswith("```"):
        lines = predicted.splitlines()
        predicted = "\n".join(l for l in lines if not l.startswith("```")).strip()
    return predicted, latency


def evaluate(
    model_id: str,
    data_path: pathlib.Path,
    limit: int | None,
    output_path: pathlib.Path | None,
    delay: float,
) -> None:
    model = _make_model(model_id)

    samples = []
    with data_path.open() as f:
        for line in f:
            samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]

    results: list[EvalResult] = []
    n = len(samples)
    exact = functional = 0

    for i, sample in enumerate(samples, 1):
        log = sample["log"]
        reference = sample["pattern"]
        pattern_name = sample.get("pattern_name", "")

        try:
            predicted, latency = _run_sample(model, log)
            em = predicted == reference
            fm = _functional_match(log, predicted)
            error = ""
        except Exception as exc:
            predicted = ""
            latency = 0.0
            em = fm = False
            error = str(exc)

        result = EvalResult(
            log=log,
            pattern_name=pattern_name,
            reference=reference,
            predicted=predicted,
            exact_match=em,
            functional_match=fm,
            latency_s=round(latency, 3),
            error=error,
        )
        results.append(result)
        if em:
            exact += 1
        if fm:
            functional += 1

        status = f"EM={em!s:5} FM={fm!s:5}"
        if error:
            status = f"ERROR: {error[:60]}"
        print(f"[{i:4}/{n}] {status}  {pattern_name}", flush=True)

        if delay > 0 and i < n:
            time.sleep(delay)

    print(f"\n--- Results for {model_id} on {n} samples ---")
    print(f"Exact match:      {exact:4d} / {n}  ({100*exact/n:.1f}%)")
    print(f"Functional match: {functional:4d} / {n}  ({100*functional/n:.1f}%)")
    avg_lat = sum(r.latency_s for r in results) / len(results)
    print(f"Avg latency:      {avg_lat:.2f}s")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"\nDetailed results saved to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a foundational model on grok pattern generation"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID, e.g. claude-opus-4-7, gpt-4o, gemini-2.0-flash",
    )
    parser.add_argument(
        "--data",
        default="data/generated/eval.jsonl",
        help="Path to evaluation JSONL (default: data/generated/eval.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write per-sample result JSONL (optional)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls to avoid rate limits (default: 0)",
    )
    args = parser.parse_args(argv)

    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    output_path = pathlib.Path(args.output) if args.output else None

    evaluate(
        model_id=args.model,
        data_path=data_path,
        limit=args.limit,
        output_path=output_path,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
