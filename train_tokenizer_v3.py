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
from datasets import Dataset, load_dataset, load_dataset_builder
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

# from transformers import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast

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

data_sets = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": {
        "field": "content",
        "extra": [],
    },  # 1.1G
    "codeparrot/github-code": {"field": "code", "extra": []},  # 1.1 TB
    "bigcode/the-stack-github-issues": {"field": "content", "extra": []},  # 66.6 G
    "iohadrubin/wikitext-103-raw-v1": {"field": "text", "extra": []},  # 310M
    "oscar-corpus/mOSCAR": {
        "field": "text",
        "extra": [
            "swe_Latn",
            "eng_Latn",
            "spa_Latn",
            "deu_Latn",
            "cym_Latn",
            "dan_Latn",
            "fra_Latn",
            "fin_Latn",
            "ita_Latn",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
            "pol_Latn",
        ],
    },  # 689G
    "allenai/c4": {
        "field": "text",
        "extra": [
            "sv",
            "en",
            "es",
            "de",
            "da",
            "fr",
            "it",
            "nl",
            "no",
            "pl",
        ],  # Note: no Welsh or Latin
    },  # this takes over mc4
    "togethercomputer/RedPajama-Data-1T": {
        "field": "text",
        "extra": [],
    },  # 2.92G
    # Add conversational data
    "HuggingFaceH4/ultrachat_200k": {
        "field": "messages",
        "extra": [],
    },  # 200k conversations
    "gutenberg": {
        "field": "text",
        "extra": [],
    },  # Project Gutenberg - all public domain
    # Open access academic content
    "arxiv": {"field": "text", "extra": []},  # Academic papers, legally redistributable
    "wikipedia": {
        "field": "text",
        "extra": [
            "20220301.sv",
            "20220301.en",
            "20220301.es",
            "20220301.de",
            "20220301.da",
            "20220301.fr",
            "20220301.it",
            "20220301.nl",
            "20220301.no",
            "20220301.pl",
        ],
    },  # High-quality reference content
    # Legal news/journalism
    "cc_news": {"field": "text", "extra": []},  # News articles with proper licensing
}


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
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
        dataset_info = load_dataset_builder(dataset_name, name=lang)
        size_gb = dataset_info.info.size_in_bytes / (1024**3)
        processed_size += size_gb
        progress = (processed_size / total_size) * 100
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
    sample_size=None,
    offline_mode=False,
    local_data_dir=None,
):
    dataset_count = 0
    total_size = 0
    start_time = time.time()  # Sonny ---> Track overall start time
    dataset_times = {}  # Sonny ---> Track individual dataset times

    # First, estimate total size
    logger.info("Estimating dataset sizes...")
    for dataset_name in data_sets:
        if len(data_sets[dataset_name]["extra"]) > 0:
            for lang in data_sets[dataset_name]["extra"]:
                try:
                    dataset_info = load_dataset_builder(dataset_name, name=lang)
                    size_gb = dataset_info.info.size_in_bytes / (1024**3)
                    total_size += size_gb
                    logger.info(
                        f"Estimated size of {dataset_name}.{lang}: {size_gb:.2f} GB"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not estimate size for {dataset_name}.{lang}: {e}"
                    )
        else:
            try:
                dataset_info = load_dataset_builder(dataset_name)
                size_gb = dataset_info.info.size_in_bytes / (1024**3)
                total_size += size_gb
                logger.info(f"Estimated size of {dataset_name}: {size_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not estimate size for {dataset_name}: {e}")

    logger.info(f"Total estimated dataset size: {total_size:.2f} GB")

    # Sonny ---> adding progress tracking
    processed_size = 0
    for dataset_name in data_sets:
        if len(data_sets[dataset_name]["extra"]) > 0:
            for lang in data_sets[dataset_name]["extra"]:
                dataset_id = f"{dataset_name}.{lang}"
                dataset_start = time.time()  # Sonny ---> Track dataset start time
                logger.info(f"Processing {dataset_id}")
                d = DSLoader()
                try:
                    d.dataset = load_dataset(
                        dataset_name,
                        name=lang,
                        split="train",
                        streaming=streaming,
                        num_proc=max_workers if not streaming else None,
                        cache_dir="./datasets",
                        download_mode="reuse_cache_if_exists" if offline_mode else None,
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
                    dataset_name,
                    split="train",
                    streaming=streaming,
                    num_proc=max_workers if not streaming else None,
                    cache_dir="./datasets",
                    download_mode="reuse_cache_if_exists" if offline_mode else None,
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

    # Sonny ---> Print final statistics
    total_time = time.time() - start_time
    avg_load_time = (
        sum(dataset_times.values()) / len(dataset_times) if dataset_times else 0
    )
    logger.info(f"Successfully loaded {dataset_count} datasets for training")
    logger.info(f"Total processed size: {processed_size:.2f} GB")
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


def batch_iterator(my_datasets, batch_size=10_000):
    i_ds = 1
    try:
        for d in tqdm(my_datasets, desc="Processing Datasets"):
            for record in tqdm(
                d.dataset, desc=f"Processing dataset {d.dataset_name} ({i_ds})"
            ):
                log_memory_usage()
                try:
                    k = record.get(d.affected_field, "")
                except AttributeError:
                    continue  # skip malformed record

                s = ""
                if isinstance(k, list):
                    if len(k) == 0:
                        continue
                    if isinstance(k[0], list):  # e.g., list of lists
                        for sublist in k:
                            s = "".join(sublist) if isinstance(sublist[0], str) else ""
                    elif isinstance(k[0], str):  # list of strings
                        s = "".join(k)
                elif isinstance(k, str):  # single string
                    s = k

                for p in range(0, len(s), batch_size):
                    yield s[p : p + batch_size]
            i_ds += 1
    except Exception as e:
        print(f"Error: {e}")
        import IPython
        IPython.embed()


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
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            sample_size=1_500_000,
            offline_mode=offline_mode,
            local_data_dir=local_data_dir,
        )

        log_memory_usage()
        logger.info(
            "Step 2: Train ByteLevelBPE tokenizer using datasets library multithreading"
        )
        tokenizer = ByteLevelBPETokenizer()

        # The datasets library handles multithreading internally when we iterate through the datasets
        tokenizer.train_from_iterator(
            batch_iterator(my_datasets),
            vocab_size=vocab_size,
            min_frequency=2,  # TODO: we might need to change this
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
    # Configure logging based on command line option
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

    # Handle offline mode setup
    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        # Disable HF Hub connectivity
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Log multiprocessing settings
    logger.info(f"Using max_workers={max_workers} for datasets multiprocessing")
    logger.info(f"Streaming mode: {'enabled' if streaming else 'disabled'}")
    logger.info(f"tokenizer_out_dir {tokenizer_out_dir}")
    logger.info(f"vocab_size {vocab_size}")
    logger.info(f"embedding_dim {embedding_dim}")
    logger.info(f"max_workers {max_workers}")
    logger.info(f"tokenizer_out_dir {tokenizer_out_dir}")
    logger.info(f"streaming {streaming}")
    logger.info(f"offline {offline}")
    logger.info(f"download_only {download_only}")
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
            offline_mode=offline,
            local_data_dir=local_data_dir,
        )

        logger.info("Step 2: Validate tokenizer")
        validate_tokenizer(tokenizer_out_dir)

        logger.info("Step 3: Initialize embedding matrix")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)
        np.save(
            os.path.join(tokenizer_out_dir, "embedding_matrix.npy"),
            weights.cpu().numpy(),
        )

        logger.info("Step 4: Save tokenizer configuration")
        save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)

        logger.info("All steps completed successfully.")
        logger.info(f"Tokenizer saved to: {tokenizer_out_dir}")
        logger.info("Files created:")
        logger.info("  - vocab.json")
        logger.info("  - merges.txt")
        logger.info("  - config.json")
        logger.info("  - embedding_matrix.npy")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
