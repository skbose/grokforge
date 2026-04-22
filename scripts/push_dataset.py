"""Push generated train/val/eval splits to a HuggingFace dataset repo."""

import argparse
import pathlib

import yaml
from datasets import DatasetDict, load_dataset


def _load_base_config() -> dict:
    base = pathlib.Path(__file__).parent.parent / "configs" / "base.yaml"
    if base.exists():
        with base.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def main() -> None:
    config = _load_base_config()
    default_repo = config.get("data", {}).get("dataset_id")

    parser = argparse.ArgumentParser(description="Push dataset splits to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        default=default_repo,
        required=default_repo is None,
        help=f"HF repo id (default from configs/base.yaml: {default_repo})",
    )
    parser.add_argument("--data-dir", default="data/generated", help="Directory containing train/val/eval.jsonl")
    parser.add_argument("--private",  action="store_true",      help="Create repo as private")
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
