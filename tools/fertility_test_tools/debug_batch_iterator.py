import pandas as pd
from tqdm import tqdm
import numpy as np
import unicodedata

# Load the parquet file
df = pd.read_parquet('datasets/oscar-corpus_mOSCAR_swe_Latn.parquet')

# Mock DSLoader-like object
class DSLoaderMock:
    def __init__(self, dataset, affected_field, dataset_name):
        self.dataset = dataset
        self.affected_field = affected_field
        self.dataset_name = dataset_name

def batch_iterator(my_datasets, batch_size=10000):
    i_ds = 1
    for d in tqdm(my_datasets, desc="Processing Datasets"):
        for record in tqdm(d.dataset, desc=f"Processing dataset {d.dataset_name} ({i_ds})"):
            k = record.get(d.affected_field, "")
            s = ""
            if isinstance(k, list):
                if len(k) == 0:
                    continue
                if isinstance(k[0], dict) and 'text' in k[0]:
                    # Special handling for list of dicts with 'text' key
                    s = " ".join([item['text'] for item in k if 'text' in item])
                elif isinstance(k[0], list):
                    for sublist in k:
                        s = " ".join(sublist) if isinstance(sublist[0], str) else ""
                elif isinstance(k[0], str):
                    s = " ".join(k)
            elif isinstance(k, str):
                s = k
            for p in range(0, len(s), batch_size):
                yield s[p : p + batch_size]
        i_ds += 1

# Prepare the mock dataset
mock_dataset = [
    {"text": row["text"]} for _, row in df.head(10).iterrows()
]
loader = DSLoaderMock(mock_dataset, "text", "oscar-corpus_mOSCAR_swe_Latn")

# Print the first 10 samples from batch_iterator
for i, sample in enumerate(batch_iterator([loader])):
    print(f"Sample {i+1}:\n{sample}\n{'-'*40}")
    if not sample.strip():
        print("[WARNING] Empty sample!")
    if i >= 9:
        break

# Print diagnostic info for each record
print("\n--- Diagnostic: Raw extracted strings from each record ---")
for record in mock_dataset:
    k = record.get("text", "")
    s = ""
    if isinstance(k, list):
        # Always join all 'text' values from list of dicts
        s = " ".join([unicodedata.normalize("NFC", item['text']) for item in k if isinstance(item, dict) and 'text' in item])
    elif isinstance(k, str):
        s = unicodedata.normalize("NFC", k)
    print(f"Extracted: {s[:200]}{'...' if len(s) > 200 else ''}")
