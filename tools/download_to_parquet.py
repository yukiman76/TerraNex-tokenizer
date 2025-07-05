#!/usr/bin/env python3
"""
Download HuggingFace datasets and save as Parquet files in ./datasets.

- Supports streaming and batch writing for memory efficiency.
- Can be used for any dataset, not just those in DATA_SETS.
- Usage:
    python tools/download_to_parquet.py --dataset bigcode/the-stack-march-sample-special-tokens-stripped --split train --field content
    python tools/download_to_parquet.py --all-datasets  # Download all in DATA_SETS

Requirements:
    pip install datasets pyarrow tqdm
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the 'datasets' library: pip install datasets")
    sys.exit(1)

# DATA_SETS from sonny_tokenizer_training.py
DATA_SETS = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": {"field": "content", "extra": []},
    "strombergnlp_nordic_langid_50k": {"field": "text", "extra": []},
    "codeparrot/github-code": {"field": "code", "extra": []},
    "bigcode/the-stack-github-issues": {"field": "content", "extra": []},
    "iohadrubin/wikitext-103-raw-v1": {"field": "text", "extra": []},
    "oscar-corpus/mOSCAR": {"field": "text", "extra": [
        "swe_Latn", "eng_Latn", "spa_Latn", "deu_Latn", "cym_Latn",
        "dan_Latn", "fra_Latn", "fin_Latn", "ita_Latn", "nld_Latn",
        "nno_Latn", "nob_Latn", "pol_Latn",
    ]},
    "allenai/c4": {"field": "text", "extra": ["sv", "en", "es", "de", "da", "fr", "it", "nl", "no", "pl"]},
    "togethercomputer/RedPajama-Data-1T": {"field": "text", "extra": []},
    "HuggingFaceH4/ultrachat_200k": {"field": "messages", "extra": []},
    "gutenberg": {"field": "text", "extra": []},
    "arxiv": {"field": "text", "extra": []},
    "wikipedia": {"field": "text", "extra": [
        "20220301.sv", "20220301.en", "20220301.es", "20220301.de",
        "20220301.da", "20220301.fr", "20220301.it", "20220301.nl",
        "20220301.no", "20220301.pl",
    ]},
    "cc_news": {"field": "text", "extra": []},
}


def save_to_parquet(dataset, out_path: Path, field: str, batch_size: int = 10000, max_rows: Optional[int] = None):
    """Stream dataset and save to Parquet file."""
    schema = pa.schema([(field, pa.string())])
    writer = None
    n = 0
    try:
        for batch in tqdm(dataset.iter(batch_size=batch_size), desc=f"Writing {out_path.name}"):
            rows = [x[field] for x in batch if field in x and isinstance(x[field], str)]
            if not rows:
                continue
            arr = pa.array(rows, type=pa.string())
            table = pa.Table.from_arrays([arr], names=[field])
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), schema)
            writer.write_table(table)
            n += len(rows)
            if max_rows and n >= max_rows:
                break
    finally:
        if writer:
            writer.close()
    print(f"Wrote {n} rows to {out_path}")


def download_one(dataset_name, split, field, subset=None, out_dir="./datasets", max_rows=None):
    """Download a single dataset split and save as Parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = dataset_name.replace('/', '__')
    if subset:
        out_name += f"__{subset}"
    if split:
        out_name += f"__{split}"
    out_path = out_dir / f"{out_name}.parquet"
    if out_path.exists():
        print(f"File exists: {out_path}, skipping.")
        return
    print(f"Downloading {dataset_name} split={split} subset={subset or '-'} field={field}")
    ds_kwargs = {"split": split}
    if subset:
        ds_kwargs["name"] = subset
    ds = load_dataset(dataset_name, **ds_kwargs, streaming=True)
    save_to_parquet(ds, out_path, field, max_rows=max_rows)


def download_all_from_config():
    for ds_name, cfg in DATA_SETS.items():
        field = cfg["field"]
        extras = cfg["extra"]
        if extras:
            for extra in extras:
                try:
                    download_one(ds_name, split="train", field=field, subset=extra)
                except Exception as e:
                    print(f"Failed: {ds_name} {extra}: {e}")
        else:
            try:
                download_one(ds_name, split="train", field=field)
            except Exception as e:
                print(f"Failed: {ds_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets as Parquet for tokenizer training.")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g. bigcode/the-stack-march-sample-special-tokens-stripped)")
    parser.add_argument("--split", type=str, default="train", help="Split to download (default: train)")
    parser.add_argument("--field", type=str, help="Field to extract (e.g. text, content, code)")
    parser.add_argument("--subset", type=str, default=None, help="Subset/language (if needed)")
    parser.add_argument("--out-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to download (for testing)")
    parser.add_argument("--all-datasets", action="store_true", help="Download all datasets in DATA_SETS")
    args = parser.parse_args()

    if args.all_datasets:
        download_all_from_config()
    elif args.dataset and args.field:
        download_one(args.dataset, args.split, args.field, args.subset, args.out_dir, args.max_rows)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
