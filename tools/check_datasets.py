from datasets import load_dataset
import sys

def check_dataset(dataset_name, config=None):
    try:
        print(f"Checking {dataset_name}" + (f" ({config})" if config else "") + "...", end=" ")
        load_dataset(dataset_name, config, split="train", streaming=True)
        print("✓ Exists")
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

print("Checking mOSCAR datasets:")
moscar_configs = ["swe_Latn", "nob_Latn", "nno_Latn", "dan_Latn"]
for config in moscar_configs:
    check_dataset("oscar-corpus/mOSCAR", config)

print("\nChecking CC-100 datasets:")
cc100_configs = ["sv", "no", "da"]
for config in cc100_configs:
    check_dataset("statmt/cc100", config) 