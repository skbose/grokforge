"""Push eval result JSONL files to a HuggingFace dataset repo."""

import argparse
import pathlib

from dotenv import load_dotenv
from datasets import DatasetDict, load_dataset

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Push eval results to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        default="skbose/grokforge-eval-results",
        help="HF dataset repo id (default: skbose/grokforge-eval-results)",
    )
    parser.add_argument(
        "--eval-dir",
        default="data/eval",
        help="Directory containing per-model eval JSONL files (default: data/eval)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private",
    )
    args = parser.parse_args()

    eval_dir = pathlib.Path(args.eval_dir)
    files = sorted(eval_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {eval_dir}")

    splits = {}
    for path in files:
        split_name = path.stem.replace("-", "_").replace(".", "_")
        splits[split_name] = load_dataset("json", data_files=str(path), split="train")
        print(f"  {split_name}: {len(splits[split_name])} rows")

    ds = DatasetDict(splits)
    print(f"\n{ds}\n")

    ds.push_to_hub(args.repo, private=args.private)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
