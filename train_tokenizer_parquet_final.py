import os
import time  # Sonny ---> Added for time tracking

os.environ["HF_DATASETS_CACHE"] = "./datasets"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5000"  # 5 minutes
import json
import logging
import sys

import click
import numpy as np
import psutil
import torch
from safetensors.torch import save_file
from tokenizers import ByteLevelBPETokenizer, normalizers
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from datasets import Dataset, load_dataset

# Move logging configuration to after click options
logger = logging.getLogger(__name__)


class DSLoader:
    dataset: Dataset = None
    affected_field: str = None
    dataset_name: str = None


SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}


def get_parquet_metadata(file_paths):
    """Get row counts and sizes from local parquet files using PyArrow"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("PyArrow not available, falling back to file size estimation")
        from pathlib import Path
        total_size_gb = sum(Path(f).stat().st_size for f in file_paths) / (1024**3)
        return None, total_size_gb

    total_rows = 0
    total_size_gb = 0

    for file_path in file_paths:
        try:
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            total_rows += metadata.num_rows
            # Fix: Use actual file size instead of tiny metadata header size
            from pathlib import Path
            total_size_gb += Path(file_path).stat().st_size / (1024**3)
        except Exception as e:
            logger.warning(f"Could not read metadata for {file_path}: {e}")
            # Fallback to file size
            from pathlib import Path
            total_size_gb += Path(file_path).stat().st_size / (1024**3)

    return total_rows, total_size_gb


def auto_detect_field(file_path):
    """Auto-detect the text field in a parquet file"""
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)

        # Get column names
        schema = parquet_file.schema_arrow
        column_names = [field.name for field in schema]

        # Try common field names in order of preference
        preferred_fields = ["text", "content", "code", "messages"]

        for field in preferred_fields:
            if field in column_names:
                logger.info(f"Auto-detected field '{field}' in {file_path}")
                return field

        # If no preferred field found, try to find any string-like field
        for field_name in column_names:
            if any(keyword in field_name.lower() for keyword in ["text", "content", "message", "code", "body"]):
                logger.info(f"Auto-detected field '{field_name}' in {file_path}")
                return field_name

        # Default to first string field
        if column_names:
            first_field = column_names[0]
            logger.warning(f"Using first field '{first_field}' for {file_path}")
            return first_field

    except Exception as e:
        logger.error(f"Failed to auto-detect field in {file_path}: {e}")

    # Last resort
    return "text"


def discover_local_parquet_files():
    """Discover and group local parquet files by dataset name and config"""
    from pathlib import Path

    datasets_dir = Path("./datasets")
    if not datasets_dir.exists():
        logger.info("No ./datasets directory found")
        return {}

    parquet_files = list(datasets_dir.glob("*.parquet"))
    if not parquet_files:
        logger.info("No parquet files found in ./datasets directory")
        return {}

    logger.info(f"Found {len(parquet_files)} parquet files in ./datasets directory")

    # Group files by dataset - simple generic parsing
    grouped_files = {}

    for file_path in parquet_files:
        filename = file_path.name

        # Generic parsing: org_dataset_xxxx.parquet -> org/dataset
        parts = filename.split('_')
        if len(parts) >= 3:
            dataset_name = f"{parts[0]}/{parts[1]}"

            if dataset_name not in grouped_files:
                grouped_files[dataset_name] = {"main": []}

            grouped_files[dataset_name]["main"].append(str(file_path))
        else:
            logger.warning(f"Skipping file with unexpected name format: {filename}")

    # Log discovered datasets with metadata
    for dataset_name, configs in grouped_files.items():
        for config, files in configs.items():
            config_str = f".{config}" if config != "main" else ""
            rows, size_gb = get_parquet_metadata(files)
            if rows:
                logger.info(f"Found {len(files)} files for {dataset_name}{config_str}: {rows:,} rows, {size_gb:.2f} GB")
            else:
                logger.info(f"Found {len(files)} files for {dataset_name}{config_str}: {size_gb:.2f} GB")

    return grouped_files


def get_local_file_size_gb(file_paths):
    """Calculate total size of local files in GB"""
    from pathlib import Path
    return sum(Path(f).stat().st_size for f in file_paths) / (1024**3)


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    logger.info(f"memory usage: {mem:.2f} MB | {mem/1024:.2f} GB")


def log_dataset_progress(dataset_id, current_dataset, total_datasets, rows=None, size_gb=None):
    """Log progress for dataset loading with local metadata"""
    progress_pct = (current_dataset / total_datasets) * 100

    if rows and size_gb:
        logger.info(f"Loaded {dataset_id} ({current_dataset}/{total_datasets}): {rows:,} rows, {size_gb:.2f} GB [{progress_pct:.1f}%]")
    elif size_gb:
        logger.info(f"Loaded {dataset_id} ({current_dataset}/{total_datasets}): {size_gb:.2f} GB [{progress_pct:.1f}%]")
    else:
        logger.info(f"Loaded {dataset_id} ({current_dataset}/{total_datasets}) [{progress_pct:.1f}%]")


def update_dataset_timing(
    dataset_id,
    dataset_start,
    start_time,
    processed_size,
    total_size,
    dataset_times,
    lang=None,
):
    # Sonny ---> Calculate and store dataset loading time
    dataset_time = time.time() - dataset_start
    dataset_times[dataset_id] = dataset_time
    elapsed_time = time.time() - start_time
    progress = (processed_size / total_size) * 100
    if progress > 0:
        estimated_total = elapsed_time / (progress / 100)
        remaining = estimated_total - elapsed_time
        logger.info(
            f"Dataset loaded in {dataset_time:.1f}s | Est. remaining: {remaining/60:.1f}min"
        )
    return dataset_time


def load_all_datasets(
    max_workers=4,
    streaming=True,
    slurm_logging=False,
):
    dataset_count = 0
    total_size_gb = 0
    total_rows = 0
    start_time = time.time()
    dataset_times = {}
    last_hourly_log = start_time

    # Discover local parquet files (local-only mode)
    logger.info("Discovering local parquet files...")
    local_files = discover_local_parquet_files()

    if not local_files:
        logger.error("No local parquet files found!")
        logger.error("Please run: python dataset_downloader.py")
        logger.error("Then retry training with the downloaded datasets")
        sys.exit(1)

    logger.info(f"Found local parquet files for {len(local_files)} datasets")

    # Calculate total datasets to process for progress tracking
    total_dataset_configs = 0
    for configs in local_files.values():
        total_dataset_configs += len(configs)

    current_dataset = 0

    # Process each dataset
    for dataset_name, configs in local_files.items():
        for config, local_parquet_files in configs.items():
            current_dataset += 1

            if config == "main":
                dataset_id = dataset_name
            else:
                dataset_id = f"{dataset_name}.{config}"

            dataset_start = time.time()
            logger.info(f"Processing {dataset_id} ({current_dataset}/{total_dataset_configs})")

            # Get metadata for progress tracking
            rows, size_gb = get_parquet_metadata(local_parquet_files)
            total_size_gb += size_gb
            if rows:
                total_rows += rows

            d = DSLoader()
            try:
                logger.info(f"Using {len(local_parquet_files)} local parquet files for {dataset_id}")

                # Load from local parquet files...
                d.dataset = load_dataset(
                    "parquet",
                    data_files=local_parquet_files,
                    split="train",
                    streaming=streaming,
                )

                # Auto-detect field name from first file.... instead of hardcoding it or overcomplicating it with multiple files!
                d.affected_field = auto_detect_field(local_parquet_files[0])
                d.dataset_name = dataset_id

                dataset_count += 1
                log_dataset_progress(dataset_id, current_dataset, total_dataset_configs, rows, size_gb)

                # Calculate dataset timing first
                dataset_time = time.time() - dataset_start
                dataset_times[dataset_id] = dataset_time

                # Add time estimation with correct parameters
                update_dataset_timing(
                    dataset_id,
                    dataset_start,
                    start_time,
                    current_dataset,  # processed datasets
                    total_dataset_configs,  # total datasets
                    dataset_times
                )

                # Hourly dataset completion logging for Slurm
                if slurm_logging:
                    current_time = time.time()
                    if current_time - last_hourly_log >= 3600:  # 1 hour
                        runtime_hours = (current_time - start_time) / 3600
                        tqdm.write(f"Datasets completed: {current_dataset}/{total_dataset_configs} - Current: processing {dataset_id} - runtime: {runtime_hours:.1f} hours")
                        last_hourly_log = current_time

                yield d

            except Exception as e:
                logger.error(f"Failed to load local dataset {dataset_id}: {e}")
                continue

    if dataset_count == 0:
        logger.error("No datasets were successfully loaded!")
        logger.error("Check your parquet files in ./datasets directory")
        sys.exit(1)

    total_time = time.time() - start_time
    avg_load_time = sum(dataset_times.values()) / len(dataset_times) if dataset_times else 0

    logger.info(f"Successfully loaded {dataset_count} datasets for training")
    if total_rows:
        logger.info(f"Total data: {total_rows:,} rows, {total_size_gb:.2f} GB")
    else:
        logger.info(f"Total data: {total_size_gb:.2f} GB")

    # Fix time display - show seconds if under 1 minute, otherwise minutes
    if total_time < 60:
        logger.info(f"Total time: {total_time:.1f} seconds")
    else:
        logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average dataset load time: {avg_load_time:.1f} seconds")

    # Save dataset loading statistics
    stats_path = os.path.join("stats", "dataset_load_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_datasets_loaded": dataset_count,
                "total_time_sec": total_time,
                "average_dataset_load_time_sec": avg_load_time,
                "dataset_times_sec": dataset_times,
                "total_size_gb": total_size_gb,
                "total_rows": total_rows,
            },
            f,
            indent=2,
        )
    logger.info(f"Dataset stats written to {stats_path}")




def initialize_embedding_matrix(tokenizer, embedding_dim=1024):
    vocab_size = tokenizer.get_vocab_size()
    logger.info(
        f"Initializing embedding matrix for vocabulary_size={vocab_size}, embedding_dim={embedding_dim}"
    )
    try:
        weights = torch.empty((vocab_size, embedding_dim))
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        logger.info(f"Embedding matrix shape: {weights.shape}")
        return weights
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}", exc_info=True)
        raise


def batch_iterator(my_datasets, batch_size=500, slurm_logging=False):
    """Iterate over datasets yielding batches of text samples (not character chunks)
    Smaller batch size provides more diverse samples for better subword learning"""
    i_ds = 1
    record_count = 0
    batch = []
    batch_count = 0
    start_time = time.time()
    last_progress_log = start_time

    try:
        for d in tqdm(my_datasets, desc="Processing Datasets"):
            for record in tqdm(
                d.dataset, desc=f"Processing dataset {d.dataset_name} ({i_ds})"
            ):
                record_count += 1
                # Log memory usage every 50,000 records only... spam protection!
                if record_count % 50000 == 0:
                    log_memory_usage()

                try:
                    k = record.get(d.affected_field, "")
                except AttributeError:
                    continue  # skip malformed record

                # Extract text from various field types
                text = ""
                if isinstance(k, list):
                    if len(k) == 0:
                        continue
                    if isinstance(k[0], list):  # e.g., list of lists
                        for sublist in k:
                            text = " ".join(sublist) if isinstance(sublist[0], str) else ""
                    elif isinstance(k[0], str):  # list of strings
                        text = " ".join(k)
                elif isinstance(k, str):  # single string
                    text = k

                # Only add non-empty text with sufficient length for good subword learning
                if text and text.strip():
                    # Filter out very short texts - they don't help BPE learn good subwords
                    stripped_text = text.strip()
                    if len(stripped_text) >= 50:  # Minimum 50 characters for meaningful subword patterns
                        # Additional quality filters to improve subword learning
                        # Skip texts that are mostly numbers, special chars, or very repetitive
                        alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in stripped_text) / len(stripped_text)
                        if alphanumeric_ratio >= 0.7:  # At least 70% alphanumeric + spaces
                            batch.append(stripped_text)

                            # Yield batch when it reaches desired size
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                                batch_count += 1
                                
                                # Slurm-compatible progress logging every 30 minutes
                                if slurm_logging:
                                    current_time = time.time()
                                    if current_time - last_progress_log >= 1800:  # 30 minutes
                                        runtime_hours = (current_time - start_time) / 3600
                                        tqdm.write(f"Training progress: processed {batch_count} batches, {record_count:,} records - runtime: {runtime_hours:.1f} hours")
                                        last_progress_log = current_time

            i_ds += 1

        # Yield remaining batch if any
        if batch:
            yield batch

    except Exception as e:
        logger.error(f"Error in batch_iterator: {e}")
        raise





def train_tokenizer(
    vocab_size,
    output_dir,
    max_workers,
    streaming=True,
    slurm_logging=False,
):
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            slurm_logging=slurm_logging,
        )

        logger.info("Starting tokenizer training...")
        log_memory_usage()  # Only log once at start of training instead of spamming the blody terminal!
        logger.info(
            "Step 2: Train ByteLevelBPE tokenizer using datasets library multithreading"
        )
        tokenizer = ByteLevelBPETokenizer()

        norm_sequence = [normalizers.NFC()]
        # norm_sequence.append(normalizers.Lowercase())
        norm_sequence.append(normalizers.Replace("\t", " "))
        norm_sequence.append(normalizers.Replace(r"\s+", " "))
        norm_sequence.append(normalizers.Replace("\u00a0", " "))
        # norm_sequence.append(normalizers.Replace(r"[\x00-\x09\x0B-\x1F\x7F]", ""))
        norm_sequence.append(normalizers.Strip())

        tokenizer.normalizer = normalizers.Sequence(norm_sequence)
        # The datasets library handles multithreading internally when we iterate through the datasets
        tokenizer.train_from_iterator(
            batch_iterator(my_datasets, slurm_logging=slurm_logging),
            vocab_size=vocab_size,
            min_frequency=3,  # Lowered to improve fertility (reduce over-segmentation)
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=True,
        )

        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to train or save tokenizer: {e}", exc_info=True)
        sys.exit(1)


def validate_tokenizer(tokenizer_dir):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        specials = dict(SPECIAL_TOKENS)
        patched = False
        tokenizer_vocab = set(tokenizer.get_vocab().keys())
        for name, token in specials.items():
            attr = getattr(tokenizer, name, None)
            if attr is None or token not in tokenizer_vocab:
                setattr(tokenizer, name, token)
                patched = True
        if patched:
            tokenizer.add_special_tokens(specials)
            tokenizer.save_pretrained(tokenizer_dir)
            logger.warning("Special tokens patched and tokenizer saved.")
        else:
            logger.info("All required special tokens present in attributes and vocab.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to validate or save tokenizer: {e}", exc_info=True)
        sys.exit(1)


def save_tokenizer_config(tokenizer_dir, vocab_size, embedding_dim):
    """Save tokenizer configuration to config.json"""
    config = {
        "model_type": "byte_level_bpe",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "special_tokens": SPECIAL_TOKENS,
        "max_position_embeddings": 2048,  #  default value??? different models have different values read a lot about this and i dont really know for sure what it should be ???
        "pad_token": SPECIAL_TOKENS["pad_token"],
        "bos_token": SPECIAL_TOKENS["bos_token"],
        "eos_token": SPECIAL_TOKENS["eos_token"],
        "unk_token": SPECIAL_TOKENS["unk_token"],
        "mask_token": SPECIAL_TOKENS["mask_token"],
    }

    config_path = os.path.join(tokenizer_dir, "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved tokenizer config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer config: {e}", exc_info=True)
        raise


@click.command()
@click.option(
    "--tokenizer-out-dir",
    default="custom_tokenizer",
    show_default=True,
    help="Directory to save the tokenizer",
)
@click.option(
    "--vocab-size",
    default=128000,
    show_default=True,
    help="Vocabulary size for tokenizer",
)
@click.option(
    "--embedding-dim",
    default=1024,
    show_default=True,
    help="Embedding dimension for initialization",
)

@click.option(
    "--max_workers",
    default=4,
    show_default=True,
    help="Maximum parallel dataset loaders (used for datasets library multiprocessing)",
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    show_default=True,
    help="Enable/disable streaming mode for datasets (streaming=True uses less memory)",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Run in offline mode using cached datasets only",
)
@click.option(
    "--local-data-dir",
    default=None,
    help="Directory containing local text files to use as fallback/additional data source",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--slurm-logging",
    is_flag=True,
    default=False,
    help="Enable periodic time-based logging for Slurm job monitoring",
)
def main(
    tokenizer_out_dir,
    vocab_size,
    embedding_dim,
    max_workers,
    streaming,
    offline,
    local_data_dir,
    log_level,
    slurm_logging,
):

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Handle offline mode setup
    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        # Disable HF Hub connectivity
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"



    logger.info(f"Using max_workers={max_workers} for datasets multiprocessing")
    logger.info(f"Streaming mode: {'enabled' if streaming else 'disabled'}")
    logger.info(f"tokenizer_out_dir {tokenizer_out_dir}")
    logger.info(f"vocab_size {vocab_size}")
    logger.info(f"embedding_dim {embedding_dim}")
    logger.info(f"max_workers {max_workers}")
    logger.info(f"streaming {streaming}")
    logger.info(f"offline {offline}")
    logger.info(f"local_data_dir {local_data_dir}")
    logger.info(f"log_level {log_level}")
    if not streaming:
        logger.info(
            "Non-streaming mode will use more memory but enables better multiprocessing"
        )

    try:
        logger.info("Step 1: Train tokenizer")
        tokenizer = train_tokenizer(
            vocab_size,
            tokenizer_out_dir,
            max_workers,
            streaming=streaming,
            slurm_logging=slurm_logging,
        )

        logger.info("Step 2: Validate tokenizer")
        validate_tokenizer(tokenizer_out_dir)

        logger.info("Step 3: Initialize embedding matrix")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)

        # Save embedding matrix with safetensors as primary format.. We are not using npy format. we use  safetensors format and fallback to pt format.
        try:
            safetensors_path = os.path.join(tokenizer_out_dir, "embedding_matrix.safetensors")
            save_file({"embedding_matrix": weights}, safetensors_path)
            logger.info(f"Saved embedding matrix to {safetensors_path}")
        except Exception as e:
            logger.warning(f"Failed to save safetensors format: {e}")
            # Fallback to PyTorch format
            try:
                pt_path = os.path.join(tokenizer_out_dir, "embedding_matrix.pt")
                torch.save(weights, pt_path)
                logger.info(f"Saved embedding matrix to {pt_path} (PyTorch fallback)")
            except Exception as pt_e:
                logger.error(f"Failed to save PyTorch format: {pt_e}")
                raise

        # Optional: Also save numpy version if possible.. we can get numpy version issues wich requires a downgrade
        """
        A module that was compiled using NumPy 1.x cannot be run in
        NumPy 2.3.2 as it may crash. To support both 1.x and 2.x
        versions of NumPy, modules must be compiled with NumPy 2.0.
        Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
        If you are a user of the module, the easiest solution will be to
        downgrade to 'numpy<2' or try to upgrade the affected module.
        We expect that some modules will need time to support NumPy 2.
        """
        try:
            np.save(
                os.path.join(tokenizer_out_dir, "embedding_matrix.npy"),
                weights.cpu().numpy(),
            )
            logger.info("Also saved numpy version (embedding_matrix.npy)")
        except Exception as np_e:
            logger.warning(f"Could not save numpy version: {np_e}")

        logger.info("Step 4: Save tokenizer configuration")
        save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)

        logger.info("All steps completed successfully.")
        logger.info(f"Tokenizer saved to: {tokenizer_out_dir}")
        logger.info("Files created:")
        logger.info("  - vocab.json")
        logger.info("  - merges.txt")
        logger.info("  - config.json")
        logger.info("  - embedding_matrix.safetensors (or .pt as fallback)")
        logger.info("  - embedding_matrix.npy (optional, if numpy compatible)")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
