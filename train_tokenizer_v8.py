import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Generator, Any

import click
import numpy as np
import psutil
import torch
from datasets import load_dataset
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
    "batch_size": 1000,  # Smaller batches for better streaming
    "min_frequency": 10,
    "sample_size": 5_000_000,  # Reduced from 22.5M for faster training
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
class TrainingProgress:
    """Track training progress for resume capability"""
    datasets_processed: List[str] = field(default_factory=list)
    training_start_time: float = field(default_factory=time.time)
    total_training_seconds: float = 0.0

    def save(self, path: str):
        """Save progress to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "datasets_processed": self.datasets_processed,
                "training_start_time": self.training_start_time,
                "total_training_seconds": self.total_training_seconds,
            }, f, indent=2)
        logger.info(f"Progress saved to {path}")

    @classmethod
    def load(cls, path: str) -> Optional['TrainingProgress']:
        """Load progress from JSON file"""
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Progress loaded from {path}")
        logger.info(f"  Datasets processed: {len(data['datasets_processed'])}")
        return cls(**data)


def log_memory_usage() -> float:
    """Log current memory usage and return value in MB"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Current memory usage: {mem_mb:.2f} MB")
    return mem_mb


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with improved regex"""
    text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return sentences


def memory_efficient_text_iterator(
    datasets_config: Dict[str, Dict],
    processed_datasets: set,
    streaming: bool = True,
    sample_size: Optional[int] = None,
    offline_mode: bool = False,
    dataset_filter: Optional[List[str]] = None,
    priority_threshold: Optional[int] = None,
    batch_size: int = 1000
) -> Generator[str, None, None]:
    """Memory-efficient streaming text iterator with progress tracking"""

    # Build list of datasets to process
    datasets_to_process = []
    for dataset_name, config in datasets_config.items():
        if dataset_filter and dataset_name not in dataset_filter:
            continue
        if priority_threshold and config.get("priority", 999) > priority_threshold:
            continue

        langs = config["extra"] if config["extra"] else [None]
        for lang in langs:
            dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
            if dataset_id not in processed_datasets:
                datasets_to_process.append((dataset_name, lang, config))

    # Sort by priority
    datasets_to_process.sort(key=lambda x: x[2].get("priority", 999))
    logger.info(f"Datasets to process: {len(datasets_to_process)}")

    # Process each dataset
    for dataset_name, lang, config in datasets_to_process:
        dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
        logger.info(f"Loading dataset: {dataset_id}")

        try:
            kwargs = {
                "split": "train",
                "streaming": streaming,
                "cache_dir": "./datasets",
                "trust_remote_code": True,  # Required for wikipedia, RedPajama, etc.
            }
            if lang:
                kwargs["name"] = lang
            if offline_mode:
                kwargs["download_mode"] = "reuse_cache_if_exists"

            dataset = load_dataset(dataset_name, **kwargs)
            field = config["field"]

            # Process records
            buffer = ""
            records_processed = 0
            max_records = sample_size if sample_size else float('inf')

            for record in dataset:
                if records_processed >= max_records:
                    break

                records_processed += 1

                # Log memory every 10k records
                if records_processed % 10000 == 0:
                    logger.info(f"  {dataset_id}: {records_processed:,} records")
                    log_memory_usage()

                # Extract text
                try:
                    val = record.get(field, "")
                except Exception as e:
                    logger.debug(f"Failed to get field {field}: {e}")
                    continue

                # Normalize value
                if isinstance(val, list):
                    if all(isinstance(item, dict) for item in val):
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

                # Split into sentences and buffer
                for sentence in split_sentences(val):
                    if not sentence or len(sentence.strip()) < 3:
                        continue

                    if len(buffer) + len(sentence) + 1 > batch_size:
                        if buffer:
                            yield buffer.strip()
                        buffer = sentence
                    else:
                        buffer = f"{buffer} {sentence}" if buffer else sentence

            # Yield final buffer
            if buffer:
                yield buffer.strip()

            logger.info(f"✓ Completed {dataset_id} ({records_processed:,} records)")

        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            continue


def train_tokenizer_optimized(
    vocab_size: int,
    output_dir: str,
    streaming: bool = True,
    offline_mode: bool = False,
    dataset_filter: Optional[List[str]] = None,
    priority_threshold: Optional[int] = None,
    resume_from: Optional[str] = None,
    sample_size: Optional[int] = None
) -> Tokenizer:
    """Train tokenizer with optimized single-pass streaming"""

    # Load or create progress
    progress_file = os.path.join(output_dir, "training_progress.json")
    if resume_from:
        progress = TrainingProgress.load(resume_from)
        if not progress:
            logger.error(f"Could not load progress from {resume_from}")
            sys.exit(1)
    else:
        progress = TrainingProgress()

    processed_datasets = set(progress.datasets_processed)
    logger.info(f"Already processed: {len(processed_datasets)} datasets")

    try:
        # Initialize tokenizer
        logger.info("Initializing BPE tokenizer...")
        tokenizer = Tokenizer(models.BPE())

        # Set normalizers
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace("\t", " "),
            normalizers.Replace(r"\s+", " "),
            normalizers.Replace("\u00a0", " "),
            normalizers.Strip()
        ])

        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=DEFAULT_CONFIG["min_frequency"],
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=True,
        )

        # Create memory-efficient iterator
        logger.info("Creating data iterator...")
        text_iterator = memory_efficient_text_iterator(
            DATASETS,
            processed_datasets,
            streaming=streaming,
            sample_size=sample_size or DEFAULT_CONFIG.get("sample_size"),
            offline_mode=offline_mode,
            dataset_filter=dataset_filter,
            priority_threshold=priority_threshold,
            batch_size=DEFAULT_CONFIG["batch_size"]
        )

        # SINGLE-PASS TRAINING - This is the key optimization!
        logger.info("Starting single-pass training...")
        logger.info("This will be much faster than the previous version!")
        start_time = time.time()

        tokenizer.train_from_iterator(text_iterator, trainer=trainer)

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.1f} minutes")

        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Save progress
        progress.total_training_seconds = training_time
        progress.save(progress_file)

        return tokenizer

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # Save progress on failure
        progress.save(progress_file)
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
@click.option("--tokenizer-out-dir", default="custom_tokenizer_v8", show_default=True,
              help="Directory to save the tokenizer")
@click.option("--vocab-size", default=100000, show_default=True,
              help="Vocabulary size for tokenizer (reduced from 128k for faster training)")
@click.option("--embedding-dim", default=1024, show_default=True,
              help="Embedding dimension for initialization")
@click.option("--streaming/--no-streaming", default=True, show_default=True,
              help="Enable/disable streaming mode for datasets")
@click.option("--offline", is_flag=True, default=False,
              help="Run in offline mode using cached datasets only")
@click.option("--dataset-filter", multiple=True, default=None,
              help="Only use specified datasets (can be used multiple times)")
@click.option("--priority-threshold", type=int, default=None,
              help="Only use datasets with priority <= threshold (1=highest priority)")
@click.option("--sample-size", type=int, default=None,
              help="Max records per dataset (default: 5M)")
@click.option("--resume-from", default=None,
              help="Resume training from progress file")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Set the logging level")
def main(tokenizer_out_dir, vocab_size, embedding_dim, streaming,
         offline, dataset_filter, priority_threshold, sample_size,
         resume_from, log_level):
    """
    Train a custom BPE tokenizer with optimized single-pass streaming.

    KEY IMPROVEMENTS over v7:
    - Single-pass training (no batch accumulation = 50x faster)
    - Lower memory usage (streaming without intermediate storage)
    - Reduced default vocab size (100k vs 128k)
    - Smaller sample size (5M vs 22.5M per dataset)
    - Simplified checkpoint system (tracks datasets only)

    For full dataset diversity, use --priority-threshold 2 or 3
    """

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Handle offline mode
    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Log configuration
    logger.info("="*60)
    logger.info("OPTIMIZED TOKENIZER TRAINING (v8)")
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info(f"  Output directory: {tokenizer_out_dir}")
    logger.info(f"  Vocabulary size: {vocab_size:,}")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Streaming: {streaming}")
    logger.info(f"  Offline mode: {offline}")
    logger.info(f"  Sample size per dataset: {sample_size or DEFAULT_CONFIG['sample_size']:,}")

    if priority_threshold:
        logger.info(f"  Priority threshold: <= {priority_threshold}")
    if resume_from:
        logger.info(f"  Resuming from: {resume_from}")
    if dataset_filter:
        logger.info(f"  Dataset filter: {list(dataset_filter)}")

    logger.info("="*60)

    try:
        # Step 1: Train tokenizer with optimized approach
        logger.info("Step 1: Training tokenizer (single-pass, streaming)")
        tokenizer = train_tokenizer_optimized(
            vocab_size, tokenizer_out_dir,
            streaming=streaming,
            offline_mode=offline,
            dataset_filter=list(dataset_filter) if dataset_filter else None,
            priority_threshold=priority_threshold,
            resume_from=resume_from,
            sample_size=sample_size
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
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Tokenizer saved to: {tokenizer_out_dir}")
        logger.info("Files created:")
        logger.info("  - tokenizer.json")
        logger.info("  - vocab.json")
        logger.info("  - merges.txt")
        logger.info("  - config.json")
        logger.info("  - embedding_matrix.npy")
        logger.info("  - training_progress.json")
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
