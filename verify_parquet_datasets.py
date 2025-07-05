import os
import pyarrow.parquet as pq

DATASET_DIR = "datasets"
SAMPLE_SIZE = 10

def verify_parquet_file(filepath):
    print(f"\n=== Verifying: {filepath} ===")
    try:
        pf = pq.ParquetFile(filepath)
        print("Schema:")
        print(pf.schema)
        # Read a small batch
        batch = pf.read_row_group(0, columns=None) if pf.num_row_groups > 0 else None
        if batch is not None:
            df = batch.to_pandas().head(SAMPLE_SIZE)
            print(f"Sample rows (up to {SAMPLE_SIZE}):")
            print(df)
            print("Null counts:")
            print(df.isnull().sum())
        else:
            print("[INFO] No row groups found in this file.")
        print(f"Total rows (from metadata): {pf.metadata.num_rows}")
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")

def main():
    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".parquet"):
            verify_parquet_file(os.path.join(DATASET_DIR, fname))

if __name__ == "__main__":
    main()
