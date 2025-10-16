import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Any

import click
import numpy as np
import psutil
import torch
from datasets import Dataset, load_dataset, load_dataset_builder
from tokenizers import Tokenizer, models, normalizers, trainers
from transformers import GPT2TokenizerFast

# Environment setup
os.environ["HF_DATASETS_CACHE"] = "./datasets"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5000"

# Logger setup
logger = logging.getLogger(__name__)

# Constants
SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

DEFAULT_CONFIG = {
    "timeout": 5000,
    "retry_total": 10,
    "retry_backoff_factor": 2,
    "max_retries": 10,
    "batch_size": 10_000,
    "min_frequency": 10,
    "sample_size": 22_500_000,
}

DATASETS = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": {
        "field": "content",
        "extra": [],
        "priority": 1,
    },
    "codeparrot/github-code": {
        "field": "code",
        "extra": [],
        "priority": 1,
    },
    "bigcode/the-stack-github-issues": {
        "field": "content",
        "extra": [],
        "priority": 2,
    },
    "iohadrubin/wikitext-103-raw-v1": {
        "field": "text",
        "extra": [],
        "priority": 1,
    },
    "oscar-corpus/mOSCAR": {
        "field": "text",
        "extra": [
            "swe_Latn", "eng_Latn", "spa_Latn", "deu_Latn", "cym_Latn",
            "dan_Latn", "fra_Latn", "fin_Latn", "ita_Latn", "nld_Latn",
            "nno_Latn", "nob_Latn", "pol_Latn",
        ],
        "priority": 2,
    },
    "allenai/c4": {
        "field": "text",
        "extra": ["sv", "en", "es", "de", "da", "fr", "it", "nl", "no", "pl"],
        "priority": 1,
    },
    "togethercomputer/RedPajama-Data-1T": {
        "field": "text",
        "extra": [],
        "priority": 1,
    },
    "HuggingFaceH4/ultrachat_200k": {
        "field": "messages",
        "extra": [],
        "priority": 2,
    },
    "gutenberg": {
        "field": "text",
        "extra": [],
        "priority": 3,
    },
    "arxiv": {
        "field": "text",
        "extra": [],
        "priority": 2,
    },
    "wikipedia": {
        "field": "text",
        "extra": [
            "20220301.sv", "20220301.en", "20220301.es", "20220301.de",
            "20220301.da", "20220301.fr", "20220301.it", "20220301.nl",
            "20220301.no", "20220301.pl",
        ],
        "priority": 1,
    },
    "cc_news": {
        "field": "text",
        "extra": [],
        "priority": 3,
    },
}


@dataclass
class DSLoader:
    """Enhanced container for dataset information"""
    dataset: Optional[Dataset] = None
    affected_field: Optional[str] = None
    dataset_name: Optional[str] = None
    size_gb: float = 0.0
    load_time: float = 0.0
    priority: int = 999


@dataclass
class TrainingState:
    """State management for training process"""
    checkpoint_dir: str = "checkpoints"
    last_checkpoint: Optional[str] = None
    datasets_processed: List[str] = field(default_factory=list)
    total_tokens_processed: int = 0
    training_start_time: float = field(default_factory=time.time)


def log_memory_usage() -> float:
    """Log current memory usage and return value in MB"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Current memory usage: {mem_mb:.2f} MB")
    return mem_mb


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay


def download_single_dataset(dataset_name: str, lang: Optional[str] = None) -> bool:
    """Download a single dataset with exponential backoff"""
    dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name

    for attempt in range(DEFAULT_CONFIG["max_retries"]):
        try:
            logger.info(f"Downloading {dataset_id} (attempt {attempt + 1})")

            kwargs = {
                "split": "train",
                "cache_dir": "./datasets",
                "trust_remote_code": True,
                "storage_options": DEFAULT_CONFIG,
            }

            if lang:
                kwargs["name"] = lang

            load_dataset(dataset_name, **kwargs)
            logger.info(f"✓ Downloaded {dataset_id}")
            return True

        except Exception as e:
            delay = exponential_backoff(attempt)
            logger.warning(f"Attempt {attempt + 1} failed for {dataset_id}: {e}")

            if attempt < DEFAULT_CONFIG["max_retries"] - 1:
                logger.info(f"Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            else:
                logger.error(f"✗ Failed to download {dataset_id} after all retries")
                return False

    return False


def download_all_datasets(dataset_filter: Optional[List[str]] = None, max_concurrent: int = 3) -> List[str]:
    """Download all datasets with concurrent downloads"""
    logger.info("Downloading datasets for offline use...")
    failed_list = []
    download_tasks = []

    # Prepare download tasks
    for dataset_name, config in DATASETS.items():
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        if config["extra"]:
            for lang in config["extra"]:
                download_tasks.append((dataset_name, lang))
        else:
            download_tasks.append((dataset_name, None))

    # Execute downloads concurrently
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_dataset = {
            executor.submit(download_single_dataset, name, lang): (name, lang)
            for name, lang in download_tasks
        }

        for future in as_completed(future_to_dataset):
            name, lang = future_to_dataset[future]
            dataset_id = f"{name}.{lang}" if lang else name

            try:
                if not future.result():
                    failed_list.append(dataset_id)
            except Exception as e:
                logger.error(f"Download failed for {dataset_id}: {e}")
                failed_list.append(dataset_id)

    logger.info("Dataset download process completed!")
    if failed_list:
        logger.error(f"Failed datasets: {failed_list}")

    return failed_list


def estimate_dataset_sizes(dataset_filter: Optional[List[str]] = None) -> Dict[str, float]:
    """Estimate sizes for all datasets and return mapping"""
    sizes = {}

    for dataset_name, config in DATASETS.items():
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        if config["extra"]:
            for lang in config["extra"]:
                dataset_id = f"{dataset_name}.{lang}"
                try:
                    dataset_info = load_dataset_builder(dataset_name, name=lang)
                    size_gb = dataset_info.info.size_in_bytes / (1024**3)
                    sizes[dataset_id] = size_gb
                    logger.debug(f"Size of {dataset_id}: {size_gb:.2f} GB")
                except Exception as e:
                    logger.warning(f"Could not estimate size for {dataset_id}: {e}")
                    sizes[dataset_id] = 0.0
        else:
            try:
                dataset_info = load_dataset_builder(dataset_name)
                size_gb = dataset_info.info.size_in_bytes / (1024**3)
                sizes[dataset_name] = size_gb
                logger.debug(f"Size of {dataset_name}: {size_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not estimate size for {dataset_name}: {e}")
                sizes[dataset_name] = 0.0

    total_size = sum(sizes.values())
    logger.info(f"Total estimated dataset size: {total_size:.2f} GB")

    return sizes


def load_single_dataset(dataset_name: str, lang: Optional[str], config: Dict[str, Any],
                       max_workers: int, streaming: bool, sample_size: Optional[int],
                       offline_mode: bool, size_gb: float = 0.0) -> Optional[DSLoader]:
    """Load a single dataset with enhanced error handling"""
    dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
    start_time = time.time()

    try:
        logger.info(f"Loading {dataset_id} ({size_gb:.2f} GB)")

        kwargs = {
            "split": "train",
            "streaming": streaming,
            "cache_dir": "./datasets",
        }

        if lang:
            kwargs["name"] = lang

        if not streaming:
            kwargs["num_proc"] = max_workers

        if offline_mode:
            kwargs["download_mode"] = "reuse_cache_if_exists"

        d = DSLoader()
        d.dataset = load_dataset(dataset_name, **kwargs)
        d.affected_field = config["field"]
        d.dataset_name = dataset_id
        d.size_gb = size_gb
        d.priority = config.get("priority", 999)
        d.load_time = time.time() - start_time

        if sample_size and not streaming:
            d.dataset = d.dataset.shuffle(seed=42).take(sample_size)

        logger.info(f"✓ Loaded {dataset_id} in {d.load_time:.1f}s")
        return d

    except Exception as e:
        if offline_mode:
            logger.warning(f"Skipping {dataset_id} - not available offline: {e}")
        else:
            logger.error(f"Failed to load {dataset_id}: {e}")
        return None


def load_all_datasets(max_workers: int = 4, streaming: bool = True,
                     sample_size: Optional[int] = None,
                     offline_mode: bool = False,
                     local_data_dir: Optional[str] = None,
                     dataset_filter: Optional[List[str]] = None,
                     priority_threshold: Optional[int] = None) -> Generator[DSLoader, None, None]:
    """Load all datasets with improved logic"""
    dataset_count = 0
    start_time = time.time()
    dataset_times = {}

    # Estimate sizes once
    sizes = estimate_dataset_sizes(dataset_filter)
    total_size = sum(sizes.values())

    # Sort datasets by priority
    sorted_datasets = sorted(
        DATASETS.items(),
        key=lambda x: x[1].get("priority", 999)
    )

    # Load datasets
    processed_size = 0.0

    for dataset_name, config in sorted_datasets:
        # Apply filters
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        if priority_threshold and config.get("priority", 999) > priority_threshold:
            logger.info(f"Skipping {dataset_name} due to priority threshold")
            continue

        langs = config["extra"] if config["extra"] else [None]

        for lang in langs:
            dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
            size_gb = sizes.get(dataset_id, 0.0)

            d = load_single_dataset(
                dataset_name, lang, config, max_workers,
                streaming, sample_size, offline_mode, size_gb
            )

            if d:
                dataset_count += 1
                processed_size += size_gb

                # Progress tracking
                if total_size > 0:
                    progress = (processed_size / total_size) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / progress * 100) - elapsed if progress > 0 else 0
                    logger.info(
                        f"Progress: {progress:.1f}% | "
                        f"Processed: {processed_size:.2f}/{total_size:.2f} GB | "
                        f"ETA: {eta/60:.1f} min"
                    )

                dataset_times[dataset_id] = d.load_time
                yield d

    if dataset_count == 0:
        logger.error("No datasets available for training!")
        if not offline_mode:
            logger.error("Consider running with --download-only first")
        sys.exit(1)

    # Save statistics
    save_training_stats(dataset_count, dataset_times, processed_size, start_time)


def save_training_stats(dataset_count: int, dataset_times: Dict[str, float],
                       processed_size: float, start_time: float):
    """Save training statistics to file"""
    total_time = time.time() - start_time
    avg_load_time = sum(dataset_times.values()) / len(dataset_times) if dataset_times else 0

    logger.info(f"Successfully loaded {dataset_count} datasets")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average dataset load time: {avg_load_time:.1f} seconds")

    stats_path = os.path.join("stats", "dataset_load_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    with open(stats_path, "w") as f:
        json.dump({
            "total_datasets_loaded": dataset_count,
            "total_time_sec": total_time,
            "average_dataset_load_time_sec": avg_load_time,
            "dataset_times_sec": dataset_times,
            "processed_size_gb": processed_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    logger.info(f"Dataset stats written to {stats_path}")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with improved regex"""
    # Handle common abbreviations and edge cases
    text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return sentences


def batch_iterator(my_datasets: Generator[DSLoader, None, None],
                  batch_size: int = 10_000,
                  memory_limit_mb: float = 4096) -> Generator[str, None, None]:
    """Create batches with memory management"""
    buffer = ""
    buffer_count = 0
    total_yielded = 0

    for d in my_datasets:
        logger.info(f"Processing dataset: {d.dataset_name}")

        for record_idx, record in enumerate(d.dataset):
            # Memory check every 1000 records
            if record_idx % 1000 == 0:
                current_mem = log_memory_usage()
                if current_mem > memory_limit_mb:
                    logger.warning(f"Memory limit approaching ({current_mem:.2f} MB), yielding buffer")
                    if buffer:
                        yield buffer.strip()
                        total_yielded += 1
                        buffer = ""

            try:
                val = record.get(d.affected_field, "")
            except Exception as e:
                logger.debug(f"Failed to get field {d.affected_field}: {e}")
                continue

            # Normalize value
            if isinstance(val, list):
                if all(isinstance(item, dict) for item in val):
                    # Handle message format (e.g., chat data)
                    val = " ".join(
                        msg.get("content", "") for msg in val
                        if isinstance(msg, dict)
                    )
                else:
                    val = " ".join(
                        " ".join(sub) if isinstance(sub, list) else str(sub)
                        for sub in val if sub
                    )
            elif not isinstance(val, str):
                val = str(val) if val is not None else ""

            if not val:
                continue

            for sentence in split_sentences(val):
                if not sentence or len(sentence.strip()) < 3:
                    continue

                if len(buffer) + len(sentence) + 1 > batch_size:
                    if buffer:
                        yield buffer.strip()
                        total_yielded += 1
                    buffer = sentence
                    buffer_count = 1
                else:
                    buffer = f"{buffer} {sentence}" if buffer else sentence
                    buffer_count += 1

    if buffer:
        yield buffer.strip()
        total_yielded += 1

    logger.info(f"Total batches yielded: {total_yielded}")


def save_checkpoint(tokenizer: Tokenizer, checkpoint_dir: str,
                   iteration: int, state: TrainingState):
    """Save training checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
    os.makedirs(checkpoint_path, exist_ok=True)

    tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"))

    state_path = os.path.join(checkpoint_path, "state.json")
    with open(state_path, "w") as f:
        json.dump({
            "iteration": iteration,
            "datasets_processed": state.datasets_processed,
            "total_tokens_processed": state.total_tokens_processed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    logger.info(f"Checkpoint saved to {checkpoint_path}")
    state.last_checkpoint = checkpoint_path


def train_tokenizer(vocab_size: int, output_dir: str, max_workers: int,
                   streaming: bool = True, offline_mode: bool = False,
                   local_data_dir: Optional[str] = None,
                   dataset_filter: Optional[List[str]] = None,
                   checkpoint_interval: int = 10000) -> Tokenizer:
    """Train the BPE tokenizer with checkpointing"""
    state = TrainingState()

    try:
        logger.info("Loading datasets...")
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            sample_size=DEFAULT_CONFIG.get("sample_size"),
            offline_mode=offline_mode,
            local_data_dir=local_data_dir,
            dataset_filter=dataset_filter,
        )

        log_memory_usage()
        logger.info("Training ByteLevelBPE tokenizer...")

        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Set normalizers
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace("\t", " "),
            normalizers.Replace(r"\s+", " "),
            normalizers.Replace("\u00a0", " "),
            normalizers.Strip()
        ])

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=DEFAULT_CONFIG["min_frequency"],
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=True,
        )

        # Train tokenizer with memory management
        tokenizer.train_from_iterator(
            batch_iterator(my_datasets,
                         batch_size=DEFAULT_CONFIG["batch_size"],
                         memory_limit_mb=4096),
            trainer=trainer,
        )

        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to train tokenizer: {e}", exc_info=True)

        # Save partial state on failure
        if state.last_checkpoint:
            logger.info(f"Partial training saved at: {state.last_checkpoint}")

        sys.exit(1)


def validate_tokenizer(tokenizer_dir: str) -> GPT2TokenizerFast:
    """Validate and patch tokenizer special tokens"""
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
            logger.info("All required special tokens present.")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to validate tokenizer: {e}", exc_info=True)
        sys.exit(1)


def initialize_embedding_matrix(tokenizer: Tokenizer, embedding_dim: int = 1024) -> torch.Tensor:
    """Initialize embedding matrix for the tokenizer"""
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Initializing embedding matrix: vocab_size={vocab_size}, embedding_dim={embedding_dim}")

    try:
        weights = torch.empty((vocab_size, embedding_dim))
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        logger.info(f"Embedding matrix shape: {weights.shape}")
        return weights
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}", exc_info=True)
        raise


def save_tokenizer_config(tokenizer_dir: str, vocab_size: int, embedding_dim: int):
    """Save tokenizer configuration"""
    config = {
        "model_type": "byte_level_bpe",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "special_tokens": SPECIAL_TOKENS,
        "max_position_embeddings": 2048,
        "pad_token": SPECIAL_TOKENS["pad_token"],
        "bos_token": SPECIAL_TOKENS["bos_token"],
        "eos_token": SPECIAL_TOKENS["eos_token"],
        "unk_token": SPECIAL_TOKENS["unk_token"],
        "mask_token": SPECIAL_TOKENS["mask_token"],
        "training_config": DEFAULT_CONFIG,
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
@click.option("--tokenizer-out-dir", default="custom_tokenizer", show_default=True,
              help="Directory to save the tokenizer")
@click.option("--vocab-size", default=128000, show_default=True,
              help="Vocabulary size for tokenizer")
@click.option("--embedding-dim", default=1024, show_default=True,
              help="Embedding dimension for initialization")
@click.option("--max-workers", default=4, show_default=True,
              help="Maximum parallel dataset loaders")
@click.option("--streaming/--no-streaming", default=True, show_default=True,
              help="Enable/disable streaming mode for datasets")
@click.option("--offline", is_flag=True, default=False,
              help="Run in offline mode using cached datasets only")
@click.option("--download-only", is_flag=True, default=False,
              help="Download datasets only (for preparing offline cache)")
@click.option("--local-data-dir", default=None,
              help="Directory containing local text files")
@click.option("--dataset-filter", multiple=True, default=None,
              help="Only use specified datasets (can be used multiple times)")
@click.option("--priority-threshold", type=int, default=None,
              help="Only use datasets with priority <= threshold")
@click.option("--max-concurrent-downloads", default=3, show_default=True,
              help="Maximum concurrent dataset downloads")
@click.option("--memory-limit-mb", default=4096, show_default=True,
              help="Memory limit in MB for batch processing")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Set the logging level")
def main(tokenizer_out_dir, vocab_size, embedding_dim, max_workers, streaming,
         offline, download_only, local_data_dir, dataset_filter,
         priority_threshold, max_concurrent_downloads, memory_limit_mb, log_level):
    """Train a custom BPE tokenizer on multiple datasets with improved logic"""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Update global config with CLI options
    DEFAULT_CONFIG["memory_limit_mb"] = memory_limit_mb

    # Handle download-only mode
    if download_only:
        logger.info("Running in download-only mode")
        failed = download_all_datasets(
            dataset_filter=list(dataset_filter) if dataset_filter else None,
            max_concurrent=max_concurrent_downloads
        )
        if not failed:
            logger.info("All downloads completed successfully!")
        else:
            logger.warning(f"Some downloads failed: {failed}")
        return

    # Handle offline mode
    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Output directory: {tokenizer_out_dir}")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Max workers: {max_workers}")
    logger.info(f"  Streaming: {streaming}")
    logger.info(f"  Offline mode: {offline}")
    logger.info(f"  Memory limit: {memory_limit_mb} MB")

    if dataset_filter:
        logger.info(f"  Dataset filter: {list(dataset_filter)}")
    if priority_threshold:
        logger.info(f"  Priority threshold: {priority_threshold}")

    try:
        # Step 1: Train tokenizer
        logger.info("Step 1: Training tokenizer")
        tokenizer = train_tokenizer(
            vocab_size, tokenizer_out_dir, max_workers,
            streaming=streaming, offline_mode=offline,
            local_data_dir=local_data_dir,
            dataset_filter=list(dataset_filter) if dataset_filter else None
        )

        # Step 2: Validate tokenizer
        logger.info("Step 2: Validating tokenizer")
        validate_tokenizer(tokenizer_out_dir)

        # Step 3: Initialize embeddings
        logger.info("Step 3: Initializing embedding matrix")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)
        np.save(
            os.path.join(tokenizer_out_dir, "embedding_matrix.npy"),
            weights.cpu().numpy()
        )

        # Step 4: Save configuration
        logger.info("Step 4: Saving tokenizer configuration")
        save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)

        # Summary
        logger.info("Training completed successfully!")
        logger.info(f"Tokenizer saved to: {tokenizer_out_dir}")
        logger.info("Files created:")
        logger.info("  - vocab.json")
        logger.info("  - merges.txt")
        logger.info("  - config.json")
        logger.info("  - embedding_matrix.npy")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()