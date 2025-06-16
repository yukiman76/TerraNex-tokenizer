import os
import glob

# Find and remove lock files in your cache directory
cache_dirs = ["~/.cache/huggingface/datasets/", "./datasets"]
for cache_dir in cache_dirs:
    lock_files = glob.glob(os.path.expanduser(f"{cache_dir}/**/*.lock"), recursive=True)

    for lock_file in lock_files:
        os.remove(lock_file)
        print(f"Removed: {lock_file}")
