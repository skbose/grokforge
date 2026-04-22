"""Split a JSONL dataset into train/val/eval files."""

import argparse
import json
import pathlib
import random


def split(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    val_size: int,
    eval_size: int,
    seed: int,
) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]

    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    if val_size + eval_size >= n:
        raise ValueError(
            f"val_size ({val_size}) + eval_size ({eval_size}) must be less than total samples ({n})"
        )

    eval_lines = lines[:eval_size]
    val_lines  = lines[eval_size:eval_size + val_size]
    train_lines = lines[eval_size + val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_lines, "val": val_lines, "eval": eval_lines}
    for name, split_lines in splits.items():
        out = output_dir / f"{name}.jsonl"
        out.write_text("\n".join(split_lines) + "\n", encoding="utf-8")
        print(f"{name:5s}: {len(split_lines):>5} samples → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSONL dataset into train/val/eval")
    parser.add_argument("--input",    default="data/generated/logs.jsonl", help="Source JSONL file")
    parser.add_argument("--out-dir",  default="data/generated",            help="Output directory")
    parser.add_argument("--val-size", type=int, default=100,               help="Number of val samples")
    parser.add_argument("--eval-size",type=int, default=100,               help="Number of eval samples")
    parser.add_argument("--seed",     type=int, default=42,                help="Random seed")
    args = parser.parse_args()

    split(
        input_path=pathlib.Path(args.input),
        output_dir=pathlib.Path(args.out_dir),
        val_size=args.val_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
