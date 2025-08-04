import os
import time
import gc
from dataclasses import dataclass
from typing import Optional, Dict

os.environ["HF_DATASETS_CACHE"] = "./datasets"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5000"
import json
import logging
import sys

import click
import numpy as np
import psutil
import torch
import mlflow
from datasets import Dataset, load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
from halo import Halo

from transformers import GPT2TokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    current_mb: float
    peak_mb: float
    available_mb: float
    total_mb: float
    swap_used_mb: float
    swap_total_mb: float
    gpu_used_mb: Optional[float] = None
    gpu_total_mb: Optional[float] = None



class MemoryManager:
    def __init__(self, warning_threshold_mb: float = 3500, critical_threshold_mb: float = 3000):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.peak_memory_mb = 0
        self.process = psutil.Process(os.getpid())
        self.has_gpu = torch.cuda.is_available()
        self.last_gc_time = time.time()
        self.gc_interval = 15
        self.gc_events = 0
        
    def get_memory_stats(self) -> MemoryStats:
        sys_mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        current_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory_mb = max(self.peak_memory_mb, current_mem)
        
        stats = MemoryStats(
            current_mb=current_mem,
            peak_mb=self.peak_memory_mb,
            available_mb=sys_mem.available / (1024 * 1024),
            total_mb=sys_mem.total / (1024 * 1024),
            swap_used_mb=swap.used / (1024 * 1024),
            swap_total_mb=swap.total / (1024 * 1024)
        )
        
        if self.has_gpu:
            stats.gpu_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            stats.gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            
        return stats
    
    def check_memory_pressure(self) -> bool:
        stats = self.get_memory_stats()
        return (stats.available_mb < self.warning_threshold_mb or 
                stats.current_mb > self.critical_threshold_mb)
    
    def get_recommended_batch_size(self, current_batch_size: int) -> int:
        stats = self.get_memory_stats()
        
        # Critical memory pressure - aggressive reduction
        if stats.current_mb > self.critical_threshold_mb:
            return max(1000, current_batch_size // 4)
        # Warning level - moderate reduction
        elif stats.current_mb > self.warning_threshold_mb:
            return max(1000, current_batch_size // 2)
        return current_batch_size
    
    def format_memory_stats(self) -> Dict[str, str]:
        stats = self.get_memory_stats()
        formatted = {
            'mem': f"{stats.current_mb:.0f}MB",
            'peak': f"{stats.peak_mb:.0f}MB",
            'avail': f"{stats.available_mb:.0f}MB",
            'gc': str(self.gc_events)
        }
        if self.has_gpu:
            formatted['gpu'] = f"{stats.gpu_used_mb:.0f}MB"
        return formatted
    
    def force_cleanup(self):
        current_time = time.time()
        
        if current_time - self.last_gc_time >= self.gc_interval:
            self.gc_events += 1
            
            # Get memory stats before cleanup
            stats = self.get_memory_stats()
            
            # Use Halo spinner for clean GC feedback
            with Halo(text=f'GC #{self.gc_events} | Memory: {stats.current_mb:.0f}MB', spinner='dots') as spinner:
                # Perform cleanup
                gc.collect()
                if self.has_gpu:
                    torch.cuda.empty_cache()
                    
                self.last_gc_time = current_time
                
                # Update with success message
                stats = self.get_memory_stats()
                spinner.succeed(f'GC #{self.gc_events} | Cleaned to: {stats.current_mb:.0f}MB')


memory_manager = MemoryManager()


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

data_sets = {
    "oscar-corpus_mOSCAR_swe_Latn": {
        "field": "text",
        "extra": [],
    },
    "strombergnlp_nordic_langid_50k": {
        "field": "text",
        "extra": [],
    }
}


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Current memory usage: {mem:.2f} MB")


def download_all_datasets():
    logger.info("Downloading all datasets for offline use...")
    failed_list = []
    for dataset_name in data_sets:
        i_trys = 10
        is_done = False
        while not is_done:
            try:
                if len(data_sets[dataset_name]["extra"]) > 0:
                    for lang in data_sets[dataset_name]["extra"]:
                        logger.info(f"Downloading {dataset_name}.{lang}")
                        try:
                            load_dataset(
                                dataset_name,
                                name=lang,
                                split="train",
                                cache_dir="./datasets",
                                trust_remote_code=True,
                                storage_options={
                                    "timeout": 5000,
                                    "retry_total": 10,
                                    "retry_backoff_factor": 2,
                                },
                            )
                            logger.info(f"✓ Downloaded {dataset_name}.{lang}")
                            is_done = True
                        except Exception as e:
                            logger.warning(
                                f"✗ Failed to download {dataset_name}.{lang}: {e}"
                            )
                            i_trys -= 1
                            if i_trys < 0:
                                failed_list.append(f"{dataset_name}.{lang}")
                                is_done = True
                else:
                    logger.info(f"Downloading {dataset_name}")
                    try:
                        load_dataset(
                            dataset_name,
                            split="train",
                            cache_dir="./datasets",
                            trust_remote_code=True,
                            storage_options={
                                "timeout": 5000,
                                "retry_total": 10,
                                "retry_backoff_factor": 2,
                            },
                        )
                        logger.info(f"✓ Downloaded {dataset_name}")
                        is_done = True
                    except Exception as e:
                        logger.warning(f"✗ Failed to download {dataset_name}: {e}")
                        i_trys -= 1
                        if i_trys < 0:
                            failed_list.append(f"{dataset_name}")
                            is_done = True

            except Exception as e:
                logger.error(f"Critical error downloading {dataset_name}: {e}")
                is_done = True

    import IPython

    IPython.embed()
    logger.info("Dataset download process completed!")
    logger.info("failed_list")
    logger.info(failed_list)


def update_progress(dataset_name, lang=None, processed_size=0, total_size=0):
    try:
        # Use local parquet file size for progress if available
        parquet_path = f"./datasets/{dataset_name}.parquet"
        if os.path.exists(parquet_path):
            size_gb = os.path.getsize(parquet_path) / (1024**3)
        else:
            size_gb = 0
        processed_size += size_gb
        # If total_size is 0, skip progress calculation
        if total_size > 0:
            progress = (processed_size / total_size) * 100
        else:
            progress = 0
        dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
        logger.info(
            f"Progress for {dataset_id}: {progress:.1f}% ({processed_size:.2f}/{total_size:.2f} GB)"
        )
        return processed_size
    except Exception as e:
        logger.warning(
            f"Could not update progress for {dataset_name}{f'.{lang}' if lang else ''}: {e}"
        )
        return processed_size


def update_dataset_timing(
    dataset_id,
    dataset_start,
    start_time,
    processed_size,
    total_size,
    dataset_times,
    lang=None,
):
    dataset_time = time.time() - dataset_start
    dataset_times[dataset_id] = dataset_time
    elapsed_time = time.time() - start_time
    # Only calculate progress if total_size > 0
    if total_size > 0:
        progress = (processed_size / total_size) * 100
    else:
        progress = 0
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
    sample_size=None,
    offline_mode=False,
    local_data_dir=None,
):
    dataset_count = 0
    total_size = 0
    start_time = time.time()  # Sonny ---> Track overall start time
    dataset_times = {}  # Sonny ---> Track individual dataset times

    logger.info("Loading local parquet datasets...")
    for dataset_name in data_sets:
        parquet_path = f"./datasets/{dataset_name}.parquet"
        if os.path.exists(parquet_path):
            total_size += os.path.getsize(parquet_path) / (1024**3)

    processed_size = 0
    for dataset_name in data_sets:
        if len(data_sets[dataset_name]["extra"]) > 0:
            for lang in data_sets[dataset_name]["extra"]:
                dataset_id = f"{dataset_name}.{lang}"
                dataset_start = time.time()
                logger.info(f"Processing {dataset_id}")
                d = DSLoader()
                try:
                    d.dataset = load_dataset(
                        "parquet",
                        data_files=f"./datasets/{dataset_name}.parquet",
                        split="train",
                        streaming=streaming
                    )

                    d.affected_field = data_sets[dataset_name]["field"]
                    d.dataset_name = dataset_id

                    if sample_size:
                        d.dataset =  d.dataset.shuffle(seed=42).take(sample_size)

                    dataset_count += 1
                    processed_size = update_progress(
                        dataset_name, lang, processed_size, total_size
                    )
                    update_dataset_timing(
                        dataset_id,
                        dataset_start,
                        start_time,
                        processed_size,
                        total_size,
                        dataset_times,
                        lang,
                    )
                    yield d

                except Exception as e:
                    if offline_mode:
                        logger.warning(
                            f"Skipping {dataset_id} - not available offline: {e}"
                        )
                        continue
                    else:
                        logger.error(f"Failed to load {dataset_id}: {e}")
                        continue
        else:
            dataset_id = dataset_name
            dataset_start = time.time()  # Sonny ---> Track dataset start time
            logger.info(f"Processing {dataset_id}")
            d = DSLoader()
            try:
                d.dataset = load_dataset(
                    "parquet",
                    data_files=f"./datasets/{dataset_name}.parquet",
                    split="train",
                    streaming=streaming
                )

                d.affected_field = data_sets[dataset_name]["field"]
                d.dataset_name = dataset_id

                if sample_size:
                    d.dataset =  d.dataset.shuffle(seed=42).take(sample_size)

                dataset_count += 1
                processed_size = update_progress(
                    dataset_name, processed_size=processed_size, total_size=total_size
                )
                update_dataset_timing(
                    dataset_id,
                    dataset_start,
                    start_time,
                    processed_size,
                    total_size,
                    dataset_times,
                )
                yield d

            except Exception as e:
                if offline_mode:
                    logger.warning(
                        f"Skipping {dataset_id} - not available offline: {e}"
                    )
                    continue
                else:
                    logger.error(f"Failed to load {dataset_id}: {e}")
                    continue

    if dataset_count == 0:
        logger.error("No datasets available for training! Consider:")
        logger.error("Running with --download-only first")
        sys.exit(1)

    total_time = time.time() - start_time
    avg_load_time = (
        sum(dataset_times.values()) / len(dataset_times) if dataset_times else 0
    )
    logger.info(f"Successfully loaded {dataset_count} datasets for training")
    logger.info(f"Total processed size: {processed_size:.2f} GB")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average dataset load time: {avg_load_time:.1f} seconds")
    stats_path = os.path.join("stats", "dataset_load_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_datasets_loaded": dataset_count,
                "total_time_sec": total_time,
                "average_dataset_load_time_sec": avg_load_time,
                "dataset_times_sec": dataset_times,
                "processed_size_gb": processed_size,
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


def get_total_records():
    from datasets import load_dataset
    total = 0
    for dataset_name in data_sets:
        parquet_path = f"./datasets/{dataset_name}.parquet"
        if os.path.exists(parquet_path):
            ds = load_dataset("parquet", data_files=parquet_path, split="train")
            total += len(ds)
    return total

def batch_iterator(my_datasets, batch_size=10_000, total_records=None):
    i_ds = 1
    record_count = 0
    current_batch_size = batch_size
    chunk_buffer = []  # Buffer to hold chunks before yielding
    buffer_size = 50  # Smaller buffer size for more frequent cleanup
    
    start_time = time.time()
    tokens_processed = 0
    
    try:
        with tqdm(total=total_records, desc="Tokenizing records") as pbar:
            for d in my_datasets:
                for record in d.dataset:
                    try:
                        k = record.get(d.affected_field, "")
                    except AttributeError:
                        continue

                    # Process text without creating large intermediate strings
                    text = ""
                    if isinstance(k, list):
                        if len(k) == 0:
                            continue
                        if isinstance(k[0], list):
                            text = " ".join(x for sublist in k 
                                         for x in sublist if isinstance(x, str))
                        elif isinstance(k[0], str):
                            text = " ".join(k)
                    elif isinstance(k, str):
                        text = k
                    else:
                        continue

                    # Split into chunks without storing the whole string
                    for p in range(0, len(text), current_batch_size):
                        chunk = text[p:p + current_batch_size]
                        chunk_buffer.append(chunk)
                        tokens_processed += len(chunk)
                        
                        # Yield accumulated chunks and clear buffer periodically
                        if len(chunk_buffer) >= buffer_size:
                            for c in chunk_buffer:
                                yield c
                            chunk_buffer.clear()
                            
                            # Force garbage collection after clearing buffer
                            memory_manager.force_cleanup()
                    
                    # Memory management and metrics logging every 1000 records
                    if record_count % 1000 == 0:
                        stats = memory_manager.get_memory_stats()
                        elapsed = time.time() - start_time
                        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                        
                        try:
                            mlflow.log_metric("memory_current_mb", stats.current_mb, step=record_count)
                            mlflow.log_metric("memory_peak_mb", stats.peak_mb, step=record_count)
                            mlflow.log_metric("memory_available_mb", stats.available_mb, step=record_count)
                            mlflow.log_metric("memory_gc_events", memory_manager.gc_events, step=record_count)
                            mlflow.log_metric("tokens_per_sec", tokens_per_sec, step=record_count)
                            mlflow.log_metric("batch_size", current_batch_size, step=record_count)
                        except Exception:
                            pass
                        
                        pbar.set_postfix(**memory_manager.format_memory_stats(), 
                                       tokens_per_sec=f"{tokens_per_sec:.0f}")
                        
                        # Check memory pressure and adjust batch size
                        new_batch_size = memory_manager.get_recommended_batch_size(current_batch_size)
                        if new_batch_size != current_batch_size:
                            logger.info(f"Adjusting batch size from {current_batch_size} to {new_batch_size} due to memory pressure")
                            current_batch_size = new_batch_size
                            
                            try:
                                mlflow.log_metric("batch_size_adjustment", current_batch_size, step=record_count)
                            except Exception:
                                pass
                            
                            # Force cleanup after batch size adjustment
                            chunk_buffer.clear()
                            memory_manager.force_cleanup()
                    
                    record_count += 1
                    pbar.update(1)
                i_ds += 1

            for c in chunk_buffer:
                yield c
            chunk_buffer.clear()
            memory_manager.force_cleanup()
            
    except Exception as e:
        logger.error(f"Error in batch_iterator: {e}")
        stats = memory_manager.get_memory_stats()
        logger.error(f"Memory stats at error: Current={stats.current_mb:.0f}MB, Peak={stats.peak_mb:.0f}MB, Available={stats.available_mb:.0f}MB")
        logger.error(f"GC events: {memory_manager.gc_events}")
        # Error stats logged to console only - no MLflow from threading context
        raise  # Re-raise the exception after logging memory stats


def train_tokenizer(
    vocab_size,
    output_dir,
    max_workers,
    streaming=True,
    offline_mode=False,
    local_data_dir=None,
):
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        stats = memory_manager.get_memory_stats()
        logger.info(f"Initial memory state: Using {stats.current_mb:.0f}MB, {stats.available_mb:.0f}MB available")
        
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            sample_size=1_500_000,
            offline_mode=offline_mode,
            local_data_dir=local_data_dir,
        )
        
        stats = memory_manager.get_memory_stats()
        logger.info(f"After dataset loading: Using {stats.current_mb:.0f}MB (peak: {stats.peak_mb:.0f}MB), {stats.available_mb:.0f}MB available")
        
        total_records = get_total_records()
        logger.info("Step 2: Train ByteLevelBPE tokenizer using datasets library multithreading")
        
        memory_manager.force_cleanup()
        
        tokenizer = ByteLevelBPETokenizer()
        
        text_iter = batch_iterator(my_datasets, batch_size=10000, total_records=total_records)
        
        tokenizer.train_from_iterator(
            text_iter,
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=False,
        )
        
        stats = memory_manager.get_memory_stats()
        logger.info(f"After tokenizer training: Using {stats.current_mb:.0f}MB (peak: {stats.peak_mb:.0f}MB), {stats.available_mb:.0f}MB available")
        
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        
        stats = memory_manager.get_memory_stats()
        logger.info(f"Final memory state: Using {stats.current_mb:.0f}MB (peak during run: {stats.peak_mb:.0f}MB)")
        if stats.gpu_used_mb is not None:
            logger.info(f"GPU memory: Using {stats.gpu_used_mb:.0f}MB of {stats.gpu_total_mb:.0f}MB")
        
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        return tokenizer
    except Exception as e:
        stats = memory_manager.get_memory_stats()
        logger.error(f"Error memory state: Using {stats.current_mb:.0f}MB (peak: {stats.peak_mb:.0f}MB)")
        logger.error(f"Failed to train or save tokenizer: {e}", exc_info=True)
        
        return None


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
    config = {
        "model_type": "byte_level_bpe",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "special_tokens": SPECIAL_TOKENS,
        "max_position_embeddings": 2048,  # Common default value
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


def log_memory_metrics(stats: Dict[str, float], step: int):
    try:
        # Only log if there's an active MLflow run
        if mlflow.active_run() is not None:
            for key, value in stats.items():
                mlflow.log_metric(f"memory_{key}", value, step=step)
        else:
            logger.debug("Skipping memory metrics logging - no active MLflow run")
    except Exception as e:
        logger.warning(f"Failed to log memory metrics: {e}")
        # Don't break training on MLflow errors


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
    "--download-only",
    is_flag=True,
    default=False,
    help="Download datasets only (for preparing offline cache)",
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
def main(
    tokenizer_out_dir,
    vocab_size,
    embedding_dim,
    max_workers,
    streaming,
    offline,
    download_only,
    local_data_dir,
    log_level,
):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if download_only:
        logger.info("Running in download-only mode")
        download_all_datasets()
        logger.info("Download completed. You can now run with --offline flag.")
        return

    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        # Disable HF Hub connectivity
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    logger.info(f"Using max_workers={max_workers} for datasets multiprocessing")
    logger.info(f"Streaming mode: {'enabled' if streaming else 'disabled'}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("tokenizer_training")

    run_name = f"tokenizer_training_{time.strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "max_workers": max_workers,
            "streaming": streaming,
            "offline": offline,
            "batch_size": 10000,
            "warning_threshold_mb": memory_manager.warning_threshold_mb,
            "critical_threshold_mb": memory_manager.critical_threshold_mb
        })
            
        try:
            logger.info("Step 1: Train tokenizer")
            start_time = time.time()
            tokenizer = train_tokenizer(
                vocab_size,
                tokenizer_out_dir,
                max_workers,
                streaming=streaming,
                offline_mode=offline,
                local_data_dir=local_data_dir,
            )
            training_time = time.time() - start_time
            
            mlflow.log_metric("training_time_seconds", training_time)
            
            logger.info("Step 2: Validate tokenizer")
            with Halo(text='Validating tokenizer', spinner='dots') as spinner:
                validate_tokenizer(tokenizer_out_dir)
                spinner.succeed('Tokenizer validation completed')

            logger.info("Step 3: Initialize embedding matrix")
            with Halo(text='Initializing embedding matrix', spinner='dots') as spinner:
                weights = initialize_embedding_matrix(tokenizer, embedding_dim)
                np.save(
                    os.path.join(tokenizer_out_dir, "embedding_matrix.npy"),
                    weights.cpu().numpy(),
                )
                spinner.succeed('Embedding matrix initialized and saved')

            logger.info("Step 4: Save tokenizer configuration")
            with Halo(text='Saving tokenizer configuration', spinner='dots') as spinner:
                save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)
                spinner.succeed('Tokenizer configuration saved')

            final_stats = memory_manager.get_memory_stats()
            mlflow.log_metric("memory_final_current_mb", final_stats.current_mb, step=0)
            mlflow.log_metric("memory_final_peak_mb", final_stats.peak_mb, step=0)
            mlflow.log_metric("memory_final_available_mb", final_stats.available_mb, step=0)
            
            with Halo(text='Logging artifacts to MLflow', spinner='dots') as spinner:
                mlflow.log_artifacts(tokenizer_out_dir, "tokenizer")
                spinner.succeed('Artifacts logged to MLflow')
                
            logger.info("All steps completed successfully.")
            logger.info(f"Tokenizer saved to: {tokenizer_out_dir}")
            logger.info("Files created:")
            logger.info("  - vocab.json")
            logger.info("  - merges.txt")
            logger.info("  - config.json")
            logger.info("  - embedding_matrix.npy")
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

        except KeyboardInterrupt:
            logger.warning("Process interrupted by user. Exiting gracefully.")
            sys.exit(0)
        except Exception:
            logger.error("Critical failure. Aborting.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
