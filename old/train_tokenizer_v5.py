import json
import logging
import os
import re
import sys
import time

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
}

DATASETS = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": {
        "field": "content",
        "extra": [],
    },
    "codeparrot/github-code": {"field": "code", "extra": []},
    "bigcode/the-stack-github-issues": {"field": "content", "extra": []},
    "iohadrubin/wikitext-103-raw-v1": {"field": "text", "extra": []},
    "oscar-corpus/mOSCAR": {
        "field": "text",
        "extra": [
            "swe_Latn", "eng_Latn", "spa_Latn", "deu_Latn", "cym_Latn",
            "dan_Latn", "fra_Latn", "fin_Latn", "ita_Latn", "nld_Latn",
            "nno_Latn", "nob_Latn", "pol_Latn",
        ],
    },
    "allenai/c4": {
        "field": "text",
        "extra": ["sv", "en", "es", "de", "da", "fr", "it", "nl", "no", "pl"],
    },
    "togethercomputer/RedPajama-Data-1T": {"field": "text", "extra": []},
    "HuggingFaceH4/ultrachat_200k": {"field": "messages", "extra": []},
    "gutenberg": {"field": "text", "extra": []},
    "arxiv": {"field": "text", "extra": []},
    "wikipedia": {
        "field": "text",
        "extra": [
            "20220301.sv", "20220301.en", "20220301.es", "20220301.de",
            "20220301.da", "20220301.fr", "20220301.it", "20220301.nl",
            "20220301.no", "20220301.pl",
        ],
    },
    "cc_news": {"field": "text", "extra": []},
}


class DSLoader:
    """Container for dataset information"""
    def __init__(self):
        self.dataset = None
        self.affected_field = None
        self.dataset_name = None


def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Current memory usage: {mem:.2f} MB")


def download_single_dataset(dataset_name, lang=None):
    """Download a single dataset with retries"""
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
            logger.warning(f"Attempt {attempt + 1} failed for {dataset_id}: {e}")
            if attempt == DEFAULT_CONFIG["max_retries"] - 1:
                logger.error(f"✗ Failed to download {dataset_id} after all retries")
                return False

    return False


def download_all_datasets():
    """Download all datasets for offline use"""
    logger.info("Downloading all datasets for offline use...")
    failed_list = []

    for dataset_name, config in DATASETS.items():
        if config["extra"]:
            for lang in config["extra"]:
                if not download_single_dataset(dataset_name, lang):
                    failed_list.append(f"{dataset_name}.{lang}")
        else:
            if not download_single_dataset(dataset_name):
                failed_list.append(dataset_name)

    logger.info("Dataset download process completed!")
    if failed_list:
        logger.error(f"Failed datasets: {failed_list}")

    return failed_list


def estimate_dataset_size(dataset_name, lang=None):
    """Estimate size of a single dataset"""
    try:
        dataset_info = load_dataset_builder(dataset_name, name=lang)
        size_gb = dataset_info.info.size_in_bytes / (1024**3)
        dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
        logger.info(f"Estimated size of {dataset_id}: {size_gb:.2f} GB")
        return size_gb
    except Exception as e:
        dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
        logger.warning(f"Could not estimate size for {dataset_id}: {e}")
        return 0


def load_single_dataset(dataset_name, lang, config, max_workers, streaming,
                       sample_size, offline_mode):
    """Load a single dataset"""
    dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name

    try:
        logger.info(f"Loading {dataset_id}")

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

        if sample_size:
            d.dataset = d.dataset.shuffle(seed=42).take(sample_size)

        return d

    except Exception as e:
        if offline_mode:
            logger.warning(f"Skipping {dataset_id} - not available offline: {e}")
        else:
            logger.error(f"Failed to load {dataset_id}: {e}")
        return None


def load_all_datasets(max_workers=4, streaming=True, sample_size=None,
                     offline_mode=False, local_data_dir=None):
    """Load all datasets with progress tracking"""
    dataset_count = 0
    start_time = time.time()
    dataset_times = {}

    # Estimate total size
    logger.info("Estimating dataset sizes...")
    total_size = 0

    for dataset_name, config in DATASETS.items():
        if config["extra"]:
            for lang in config["extra"]:
                total_size += estimate_dataset_size(dataset_name, lang)
        else:
            total_size += estimate_dataset_size(dataset_name)

    logger.info(f"Total estimated dataset size: {total_size:.2f} GB")

    # Load datasets
    processed_size = 0

    for dataset_name, config in DATASETS.items():
        langs = config["extra"] if config["extra"] else [None]

        for lang in langs:
            dataset_start = time.time()

            d = load_single_dataset(
                dataset_name, lang, config, max_workers,
                streaming, sample_size, offline_mode
            )

            if d:
                dataset_count += 1

                # Update progress
                dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
                size_gb = estimate_dataset_size(dataset_name, lang)
                processed_size += size_gb

                if total_size > 0:
                    progress = (processed_size / total_size) * 100
                    logger.info(f"Progress: {progress:.1f}% ({processed_size:.2f}/{total_size:.2f} GB)")

                # Track timing
                dataset_time = time.time() - dataset_start
                dataset_times[dataset_id] = dataset_time
                logger.info(f"Dataset loaded in {dataset_time:.1f}s")

                yield d

    if dataset_count == 0:
        logger.error("No datasets available for training! Consider running with --download-only first")
        sys.exit(1)

    # Save statistics
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
        }, f, indent=2)

    logger.info(f"Dataset stats written to {stats_path}")


def split_sentences(text):
    """Split text into sentences"""
    return re.split(r"(?<=[.!?])\s+", text.strip())


def batch_iterator(my_datasets, batch_size=10_000):
    """Create batches from dataset iterator"""
    buffer = ""

    for d in my_datasets:
        for record in d.dataset:
            try:
                val = record.get(d.affected_field, "")
            except Exception:
                continue

            # Normalize value
            if isinstance(val, list):
                val = " ".join(
                    " ".join(sub) if isinstance(sub, list) else sub
                    for sub in val if sub
                )
            elif not isinstance(val, str):
                continue

            for sentence in split_sentences(val):
                if not sentence:
                    continue

                if len(buffer) + len(sentence) + 1 > batch_size:
                    yield buffer.strip()
                    buffer = sentence
                else:
                    buffer += " " + sentence

    if buffer:
        yield buffer.strip()


def train_tokenizer(vocab_size, output_dir, max_workers, streaming=True,
                   offline_mode=False, local_data_dir=None):
    """Train the BPE tokenizer"""
    try:
        logger.info("Loading datasets...")
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            sample_size=22_500_000,
            offline_mode=offline_mode,
            local_data_dir=local_data_dir,
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
            min_frequency=10,
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=True,
        )

        # Train tokenizer
        tokenizer.train_from_iterator(
            batch_iterator(my_datasets),
            trainer=trainer,
        )

        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to train tokenizer: {e}", exc_info=True)
        sys.exit(1)


def validate_tokenizer(tokenizer_dir):
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


def initialize_embedding_matrix(tokenizer, embedding_dim=1024):
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


def save_tokenizer_config(tokenizer_dir, vocab_size, embedding_dim):
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
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Set the logging level")
def main(tokenizer_out_dir, vocab_size, embedding_dim, max_workers, streaming,
         offline, download_only, local_data_dir, log_level):
    """Train a custom BPE tokenizer on multiple datasets"""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Handle download-only mode
    if download_only:
        logger.info("Running in download-only mode")
        download_all_datasets()
        logger.info("Download completed. You can now run with --offline flag.")
        return

    # Handle offline mode
    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Output directory: {tokenizer_out_dir}")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Max workers: {max_workers}")
    logger.info(f"  Streaming: {streaming}")
    logger.info(f"  Offline mode: {offline}")

    try:
        # Step 1: Train tokenizer
        logger.info("Step 1: Training tokenizer")
        tokenizer = train_tokenizer(
            vocab_size, tokenizer_out_dir, max_workers,
            streaming=streaming, offline_mode=offline,
            local_data_dir=local_data_dir
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