<u>**Short Explanation:**</u>


Use analyze.py to analyze the dataset and get a undrestanding about the languages included.
you must create the manifest file (CSV)

The manifest file is needed because the scanner does not discover files by itself. It expects an explicit list of input files in a CSV with a `file_path` column.

Purpose in your code:

- `list_manifest_files(manifest_csv)` reads the CSV and extracts all `file_path` entries.

- That list defines the full universe of Parquet files to scan.

- Sharding (`shard_list_deterministic`) depends on that list to split files across workers or jobs in a deterministic way.

- Without the manifest, the scanner cannot know which files exist, cannot shard correctly, and cannot guarantee reproducibility or resume logic.

In short: **the manifest is the contract telling the scanner exactly which Parquet files to process, in what order, and consistently across shards and retries.**



Run it like this:

```bash
python ./sharding/analyze_v2.py --datasets-dir ./datasets --manifest ./out/manifest.csv
```


