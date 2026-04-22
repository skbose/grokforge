"""Push generated train/val/eval splits to a HuggingFace dataset repo."""

import argparse
import pathlib

from datasets import DatasetDict, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Push dataset splits to HuggingFace Hub")
    parser.add_argument("--repo",     required=True,              help="HF repo id, e.g. skbose/grokforge-v1")
    parser.add_argument("--data-dir", default="data/generated",   help="Directory containing train/val/eval.jsonl")
    parser.add_argument("--private",  action="store_true",        help="Create repo as private")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    splits = {}
    for split in ("train", "val", "eval"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split] = load_dataset("json", data_files=str(path), split="train")

    ds = DatasetDict(splits)
    print(ds)

    ds.push_to_hub(args.repo, private=args.private)
    print(f"\nPushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
