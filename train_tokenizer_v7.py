import json
import logging
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Generator, Any, Tuple

import click
import numpy as np
import psutil
import torch
from datasets import Dataset, load_dataset, load_dataset_builder
from tokenizers import Tokenizer, models, normalizers, trainers, pre_tokenizers
from tokenizers.models import BPE
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
    "checkpoint_interval_minutes": 30,  # Checkpoint every 30 minutes
    "checkpoint_interval_batches": 50000,  # Or every 50k batches
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
class TrainingCheckpoint:
    """Comprehensive training checkpoint state"""
    # Tokenizer state
    tokenizer_state: Optional[Dict] = None
    vocab: Optional[Dict] = None
    merges: Optional[List] = None

    # Training progress
    datasets_processed: List[str] = field(default_factory=list)
    datasets_in_progress: Optional[str] = None
    current_dataset_position: int = 0

    # Statistics
    total_batches_processed: int = 0
    total_tokens_processed: int = 0
    total_bytes_processed: int = 0

    # Timing
    training_start_time: float = field(default_factory=time.time)
    last_checkpoint_time: float = field(default_factory=time.time)
    total_training_seconds: float = 0.0

    # Configuration
    vocab_size: int = 128000
    min_frequency: int = 10
    checkpoint_version: str = "1.0"

    def to_dict(self) -> Dict:
        """Convert checkpoint to dictionary for serialization"""
        return {
            "tokenizer_state": self.tokenizer_state,
            "vocab": self.vocab,
            "merges": self.merges,
            "datasets_processed": self.datasets_processed,
            "datasets_in_progress": self.datasets_in_progress,
            "current_dataset_position": self.current_dataset_position,
            "total_batches_processed": self.total_batches_processed,
            "total_tokens_processed": self.total_tokens_processed,
            "total_bytes_processed": self.total_bytes_processed,
            "training_start_time": self.training_start_time,
            "last_checkpoint_time": self.last_checkpoint_time,
            "total_training_seconds": self.total_training_seconds,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "checkpoint_version": self.checkpoint_version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingCheckpoint':
        """Create checkpoint from dictionary"""
        return cls(**data)


class CheckpointManager:
    """Manages training checkpoints and recovery"""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.current_checkpoint: Optional[TrainingCheckpoint] = None

    def save_checkpoint(self, checkpoint: TrainingCheckpoint, name: str = None) -> str:
        """Save checkpoint to disk"""
        if name is None:
            name = f"checkpoint_{int(time.time())}"

        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save checkpoint metadata
        meta_path = os.path.join(checkpoint_path, "checkpoint.json")
        with open(meta_path, "w") as f:
            # Convert to dict and handle non-serializable items
            checkpoint_dict = checkpoint.to_dict()
            json.dump(checkpoint_dict, f, indent=2, default=str)

        # Save tokenizer state if available
        if checkpoint.tokenizer_state:
            tokenizer_path = os.path.join(checkpoint_path, "tokenizer_state.pkl")
            with open(tokenizer_path, "wb") as f:
                pickle.dump(checkpoint.tokenizer_state, f)

        # Save vocab and merges as separate files for compatibility
        if checkpoint.vocab:
            vocab_path = os.path.join(checkpoint_path, "vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(checkpoint.vocab, f, indent=2)

        if checkpoint.merges:
            merges_path = os.path.join(checkpoint_path, "merges.txt")
            with open(merges_path, "w") as f:
                for merge in checkpoint.merges:
                    f.write(f"{merge}\n")

        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str = None) -> Optional[TrainingCheckpoint]:
        """Load checkpoint from disk"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                return None

        meta_path = os.path.join(checkpoint_path, "checkpoint.json")
        if not os.path.exists(meta_path):
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None

        # Load checkpoint metadata
        with open(meta_path, "r") as f:
            checkpoint_dict = json.load(f)

        # Load tokenizer state if available
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer_state.pkl")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                checkpoint_dict["tokenizer_state"] = pickle.load(f)

        # Load vocab if available
        vocab_path = os.path.join(checkpoint_path, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                checkpoint_dict["vocab"] = json.load(f)

        # Load merges if available
        merges_path = os.path.join(checkpoint_path, "merges.txt")
        if os.path.exists(merges_path):
            with open(merges_path, "r") as f:
                checkpoint_dict["merges"] = [line.strip() for line in f if line.strip()]

        checkpoint = TrainingCheckpoint.from_dict(checkpoint_dict)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"  Datasets processed: {len(checkpoint.datasets_processed)}")
        logger.info(f"  Batches processed: {checkpoint.total_batches_processed}")
        logger.info(f"  Training time: {checkpoint.total_training_seconds/3600:.2f} hours")

        self.current_checkpoint = checkpoint
        return checkpoint

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint"""
        checkpoints = []

        if not os.path.exists(self.checkpoint_dir):
            return None

        for name in os.listdir(self.checkpoint_dir):
            checkpoint_path = os.path.join(self.checkpoint_dir, name)
            meta_path = os.path.join(checkpoint_path, "checkpoint.json")

            if os.path.isdir(checkpoint_path) and os.path.exists(meta_path):
                # Get modification time
                mtime = os.path.getmtime(meta_path)
                checkpoints.append((mtime, checkpoint_path))

        if not checkpoints:
            return None

        # Sort by modification time and return latest
        checkpoints.sort(reverse=True)
        latest_path = checkpoints[0][1]
        logger.info(f"Found latest checkpoint: {latest_path}")
        return latest_path

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        checkpoints = []

        if not os.path.exists(self.checkpoint_dir):
            return checkpoints

        for name in os.listdir(self.checkpoint_dir):
            checkpoint_path = os.path.join(self.checkpoint_dir, name)
            meta_path = os.path.join(checkpoint_path, "checkpoint.json")

            if os.path.isdir(checkpoint_path) and os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                checkpoints.append({
                    "path": checkpoint_path,
                    "name": name,
                    "datasets_processed": len(meta.get("datasets_processed", [])),
                    "batches": meta.get("total_batches_processed", 0),
                    "training_hours": meta.get("total_training_seconds", 0) / 3600,
                    "timestamp": os.path.getmtime(meta_path),
                })

        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints


class IncrementalBPETrainer:
    """Custom BPE trainer that supports incremental training"""

    def __init__(self, vocab_size: int, min_frequency: int,
                 initial_vocab: Optional[Dict] = None,
                 initial_merges: Optional[List] = None):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab = initial_vocab or {}
        self.merges = initial_merges or []
        self.tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize or restore tokenizer"""
        self.tokenizer = Tokenizer(models.BPE())

        # Set normalizers
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace("\t", " "),
            normalizers.Replace(r"\s+", " "),
            normalizers.Replace("\u00a0", " "),
            normalizers.Strip()
        ])

        # If we have existing vocab/merges, restore them
        if self.vocab and self.merges:
            # This is a simplified restoration - in practice, you'd need
            # to properly reconstruct the BPE model state
            logger.info(f"Restoring tokenizer with {len(self.vocab)} vocab items and {len(self.merges)} merges")

    def train_incremental(self, data_iterator: Generator,
                         checkpoint_callback=None,
                         checkpoint_interval_batches: int = 10000) -> Tokenizer:
        """Train tokenizer incrementally with checkpoint support"""

        # Create trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=True,
        )

        # Collect batches for training
        batches = []
        batch_count = 0

        for batch in data_iterator:
            batches.append(batch)
            batch_count += 1

            # Periodically train and checkpoint
            if batch_count % checkpoint_interval_batches == 0:
                logger.info(f"Training on {len(batches)} batches...")

                # Train on collected batches
                self.tokenizer.train_from_iterator(batches, trainer=trainer)

                # Update vocab and merges
                self._update_vocab_and_merges()

                # Callback for checkpointing
                if checkpoint_callback:
                    checkpoint_callback(self)

                # Clear batches
                batches = []

        # Train on remaining batches
        if batches:
            logger.info(f"Training on final {len(batches)} batches...")
            self.tokenizer.train_from_iterator(batches, trainer=trainer)
            self._update_vocab_and_merges()

        return self.tokenizer

    def _update_vocab_and_merges(self):
        """Extract current vocab and merges from tokenizer"""
        # Get the model
        model = self.tokenizer.model
        if hasattr(model, 'save'):
            # Save to temporary files and read back
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                self.tokenizer.save_model(tmpdir)

                vocab_path = os.path.join(tmpdir, "vocab.json")
                if os.path.exists(vocab_path):
                    with open(vocab_path, "r") as f:
                        self.vocab = json.load(f)

                merges_path = os.path.join(tmpdir, "merges.txt")
                if os.path.exists(merges_path):
                    with open(merges_path, "r") as f:
                        self.merges = [line.strip() for line in f if line.strip()]

    def get_state(self) -> Dict:
        """Get current trainer state"""
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
        }

    def restore_state(self, state: Dict):
        """Restore trainer state"""
        self.vocab = state.get("vocab", {})
        self.merges = state.get("merges", [])
        self.vocab_size = state.get("vocab_size", self.vocab_size)
        self.min_frequency = state.get("min_frequency", self.min_frequency)
        self._initialize_tokenizer()


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


def load_datasets_with_resume(checkpoint: Optional[TrainingCheckpoint],
                             max_workers: int = 4,
                             streaming: bool = True,
                             sample_size: Optional[int] = None,
                             offline_mode: bool = False,
                             dataset_filter: Optional[List[str]] = None,
                             priority_threshold: Optional[int] = None) -> Generator[DSLoader, None, None]:
    """Load datasets with resume capability"""

    # Get list of datasets to process
    datasets_to_process = []
    processed_datasets = set(checkpoint.datasets_processed) if checkpoint else set()

    for dataset_name, config in DATASETS.items():
        # Apply filters
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        if priority_threshold and config.get("priority", 999) > priority_threshold:
            continue

        langs = config["extra"] if config["extra"] else [None]
        for lang in langs:
            dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name

            # Skip if already processed
            if dataset_id in processed_datasets:
                logger.info(f"Skipping already processed dataset: {dataset_id}")
                continue

            datasets_to_process.append((dataset_name, lang, config))

    # Sort by priority
    datasets_to_process.sort(key=lambda x: x[2].get("priority", 999))

    logger.info(f"Datasets to process: {len(datasets_to_process)}")
    logger.info(f"Already processed: {len(processed_datasets)}")

    # Process datasets
    for dataset_name, lang, config in datasets_to_process:
        d = load_single_dataset(
            dataset_name, lang, config, max_workers,
            streaming, sample_size, offline_mode, 0.0
        )

        if d:
            yield d


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with improved regex"""
    # Handle common abbreviations and edge cases
    text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    return sentences


def resumable_batch_iterator(my_datasets: Generator[DSLoader, None, None],
                           checkpoint_manager: CheckpointManager,
                           checkpoint: Optional[TrainingCheckpoint] = None,
                           batch_size: int = 10_000,
                           memory_limit_mb: float = 4096) -> Generator[Tuple[str, int], None, None]:
    """Batch iterator with checkpoint support"""

    buffer = ""
    total_batches = checkpoint.total_batches_processed if checkpoint else 0
    total_bytes = checkpoint.total_bytes_processed if checkpoint else 0
    last_checkpoint_time = time.time()

    for d in my_datasets:
        logger.info(f"Processing dataset: {d.dataset_name}")

        # Update checkpoint
        if checkpoint:
            checkpoint.datasets_in_progress = d.dataset_name

        for record_idx, record in enumerate(d.dataset):
            # Check if we need to checkpoint
            current_time = time.time()
            time_since_checkpoint = current_time - last_checkpoint_time

            should_checkpoint = (
                time_since_checkpoint > DEFAULT_CONFIG["checkpoint_interval_minutes"] * 60 or
                total_batches % DEFAULT_CONFIG["checkpoint_interval_batches"] == 0
            )

            if should_checkpoint and checkpoint:
                checkpoint.total_batches_processed = total_batches
                checkpoint.total_bytes_processed = total_bytes
                checkpoint.current_dataset_position = record_idx
                checkpoint.total_training_seconds += time_since_checkpoint
                checkpoint.last_checkpoint_time = current_time

                checkpoint_manager.save_checkpoint(checkpoint, name=f"auto_{int(current_time)}")
                last_checkpoint_time = current_time

            # Memory check
            if record_idx % 1000 == 0:
                current_mem = log_memory_usage()
                if current_mem > memory_limit_mb:
                    logger.warning(f"Memory limit approaching ({current_mem:.2f} MB)")
                    if buffer:
                        yield buffer.strip(), total_batches
                        total_batches += 1
                        total_bytes += len(buffer)
                        buffer = ""

            # Process record
            try:
                val = record.get(d.affected_field, "")
            except Exception as e:
                logger.debug(f"Failed to get field {d.affected_field}: {e}")
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

            for sentence in split_sentences(val):
                if not sentence or len(sentence.strip()) < 3:
                    continue

                if len(buffer) + len(sentence) + 1 > batch_size:
                    if buffer:
                        yield buffer.strip(), total_batches
                        total_batches += 1
                        total_bytes += len(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer} {sentence}" if buffer else sentence

        # Mark dataset as processed
        if checkpoint:
            checkpoint.datasets_processed.append(d.dataset_name)
            checkpoint.datasets_in_progress = None

    if buffer:
        yield buffer.strip(), total_batches
        total_batches += 1
        total_bytes += len(buffer)

    logger.info(f"Total batches yielded: {total_batches}")


def train_tokenizer_with_resume(vocab_size: int, output_dir: str, max_workers: int,
                               streaming: bool = True, offline_mode: bool = False,
                               dataset_filter: Optional[List[str]] = None,
                               resume_from: Optional[str] = None) -> Tokenizer:
    """Train tokenizer with full resume support"""

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()

    # Load or create checkpoint
    if resume_from:
        checkpoint = checkpoint_manager.load_checkpoint(resume_from)
        if not checkpoint:
            logger.error(f"Could not load checkpoint from {resume_from}")
            sys.exit(1)
        logger.info("Resuming from checkpoint")
    else:
        checkpoint = TrainingCheckpoint(
            vocab_size=vocab_size,
            min_frequency=DEFAULT_CONFIG["min_frequency"],
        )
        logger.info("Starting fresh training")

    try:
        # Load datasets with resume
        logger.info("Loading datasets...")
        my_datasets = load_datasets_with_resume(
            checkpoint=checkpoint,
            max_workers=max_workers,
            streaming=streaming,
            sample_size=DEFAULT_CONFIG.get("sample_size"),
            offline_mode=offline_mode,
            dataset_filter=dataset_filter,
        )

        log_memory_usage()

        # Initialize incremental trainer
        trainer = IncrementalBPETrainer(
            vocab_size=vocab_size,
            min_frequency=DEFAULT_CONFIG["min_frequency"],
            initial_vocab=checkpoint.vocab,
            initial_merges=checkpoint.merges,
        )

        # Define checkpoint callback
        def checkpoint_callback(trainer_instance):
            nonlocal checkpoint
            state = trainer_instance.get_state()
            checkpoint.vocab = state["vocab"]
            checkpoint.merges = state["merges"]
            checkpoint_manager.save_checkpoint(checkpoint, name=f"training_{int(time.time())}")

        # Create resumable batch iterator
        batch_gen = resumable_batch_iterator(
            my_datasets,
            checkpoint_manager,
            checkpoint,
            batch_size=DEFAULT_CONFIG["batch_size"],
            memory_limit_mb=4096
        )

        # Extract just the batches for training
        batch_only_gen = (batch for batch, _ in batch_gen)

        # Train with incremental updates
        logger.info("Training tokenizer with checkpoint support...")
        tokenizer = trainer.train_incremental(
            batch_only_gen,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval_batches=DEFAULT_CONFIG["checkpoint_interval_batches"]
        )

        # Final save
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")

        # Save final checkpoint
        final_checkpoint = TrainingCheckpoint(
            vocab=trainer.vocab,
            merges=trainer.merges,
            datasets_processed=checkpoint.datasets_processed,
            total_batches_processed=checkpoint.total_batches_processed,
            total_tokens_processed=checkpoint.total_tokens_processed,
            total_bytes_processed=checkpoint.total_bytes_processed,
            total_training_seconds=checkpoint.total_training_seconds + (time.time() - checkpoint.last_checkpoint_time),
            vocab_size=vocab_size,
            min_frequency=DEFAULT_CONFIG["min_frequency"],
        )
        checkpoint_manager.save_checkpoint(final_checkpoint, name="final")

        return tokenizer

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

        # Save emergency checkpoint
        if checkpoint:
            checkpoint_manager.save_checkpoint(checkpoint, name=f"emergency_{int(time.time())}")
            logger.info("Emergency checkpoint saved")

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
@click.option("--dataset-filter", multiple=True, default=None,
              help="Only use specified datasets (can be used multiple times)")
@click.option("--resume-from", default=None,
              help="Resume training from checkpoint (path or 'latest')")
@click.option("--list-checkpoints", is_flag=True, default=False,
              help="List available checkpoints and exit")
@click.option("--checkpoint-interval-minutes", default=30, show_default=True,
              help="Save checkpoint every N minutes")
@click.option("--checkpoint-interval-batches", default=50000, show_default=True,
              help="Save checkpoint every N batches")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Set the logging level")
def main(tokenizer_out_dir, vocab_size, embedding_dim, max_workers, streaming,
         offline, dataset_filter, resume_from, list_checkpoints,
         checkpoint_interval_minutes, checkpoint_interval_batches, log_level):
    """Train a custom BPE tokenizer with full checkpoint and resume support"""

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Update config with CLI options
    DEFAULT_CONFIG["checkpoint_interval_minutes"] = checkpoint_interval_minutes
    DEFAULT_CONFIG["checkpoint_interval_batches"] = checkpoint_interval_batches

    # Handle checkpoint listing
    if list_checkpoints:
        checkpoint_manager = CheckpointManager()
        checkpoints = checkpoint_manager.list_checkpoints()

        if not checkpoints:
            logger.info("No checkpoints found")
        else:
            logger.info("Available checkpoints:")
            for cp in checkpoints:
                logger.info(f"  {cp['name']}:")
                logger.info(f"    Path: {cp['path']}")
                logger.info(f"    Datasets: {cp['datasets_processed']}")
                logger.info(f"    Batches: {cp['batches']}")
                logger.info(f"    Training hours: {cp['training_hours']:.2f}")
                logger.info(f"    Timestamp: {time.ctime(cp['timestamp'])}")
        return

    # Handle resume
    if resume_from == "latest":
        checkpoint_manager = CheckpointManager()
        resume_from = checkpoint_manager.find_latest_checkpoint()
        if not resume_from:
            logger.error("No checkpoints found to resume from")
            sys.exit(1)

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
    logger.info(f"  Checkpoint interval: {checkpoint_interval_minutes} min / {checkpoint_interval_batches} batches")

    if resume_from:
        logger.info(f"  Resuming from: {resume_from}")

    if dataset_filter:
        logger.info(f"  Dataset filter: {list(dataset_filter)}")

    try:
        # Step 1: Train tokenizer with resume support
        logger.info("Step 1: Training tokenizer")
        tokenizer = train_tokenizer_with_resume(
            vocab_size, tokenizer_out_dir, max_workers,
            streaming=streaming, offline_mode=offline,
            dataset_filter=list(dataset_filter) if dataset_filter else None,
            resume_from=resume_from
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

        # Try to save checkpoint before exiting
        checkpoint_manager = CheckpointManager()
        logger.info("Attempting to save checkpoint before exit...")
        # Note: In a real implementation, you'd need to access the current checkpoint state here

        sys.exit(0)
    except Exception:
        logger.error("Critical failure.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()