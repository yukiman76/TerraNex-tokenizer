import json
import logging
import os
import random
import sys
import tempfile
import time

import click
import huggingface_hub
import numpy as np
import requests
import torch
import yaml
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

# Move logging configuration to after click options
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>"
}

DATASET_FIELD_MAPPING = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": "content",
    "codeparrot/github-code": "content",
    "bigcode/the-stack-github-issues": "content",
    "iohadrubin/wikitext-103-raw-v1": "text"
}

def load_config(config_path):
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {str(e)}", exc_info=True)
        sys.exit(1)

def fetch_file_from_url(url):
    try:
        logger.info(f"Downloading file from {url}")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Downloading {url}", unit="chunk"):
                f.write(chunk)
            temp_path = f.name
        logger.info(f"Downloaded file to {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}", exc_info=True)
        return None

def yield_lines_from_file(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line_text = line.strip()
                if line_text:
                    yield line_text
    except Exception as e:
        logger.warning(f"Failed to read from file {file_path}: {str(e)}")

def extract_hf_dataset(dataset_name, config=None, split="train", field="auto", max_pct=1.0, streaming=True, min_line_length=32, max_samples=None):
    logger.info(f"Loading dataset {dataset_name} with config={config}, split={split}, field={field}, max_pct={max_pct}, max_samples={max_samples}")

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            dataset = load_dataset(
                dataset_name,
                config,
                split=split,
                streaming=streaming,
                download_config=DownloadConfig(resume_download=True)
            )
            logger.info(f"Dataset loaded successfully. Streaming mode: {streaming}")

            # Use the field mapping or default to 'content'
            field = DATASET_FIELD_MAPPING.get(dataset_name, "content")
            logger.info(f"Using field '{field}' for dataset {dataset_name}")

            total_lines = 0
            sampled_lines = 0
            for entry in tqdm(dataset, desc=f"Processing {dataset_name}", total=None):
                total_lines += 1

                # Skip this entry if we're sampling and random value is above max_pct
                if random.random() > max_pct:
                    continue

                text = str(entry.get(field, "")).strip().replace("\n", " ")

                if len(text) >= min_line_length:
                    sampled_lines += 1
                    yield text

                    # Stop if we've reached max_samples
                    if max_samples is not None and sampled_lines >= max_samples:
                        logger.info(f"Reached max_samples ({max_samples}) for {dataset_name}")
                        break

            logger.info(f"Finished processing {dataset_name}. Total lines: {total_lines:,}, Sampled lines: {sampled_lines:,}")
            break  # Success, exit the retry loop

        except requests.exceptions.ReadTimeout:
            attempt += 1
            if attempt < max_attempts:
                wait_time = 5 * attempt  # Progressive backoff: 5s, 10s, 15s
                logger.warning(f"Connection timed out (attempt {attempt}/{max_attempts}). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Failed to connect to Hugging Face after multiple attempts. Please check your internet connection.")
                raise
        except huggingface_hub.errors.HfHubHTTPError as e:
            if "403" in str(e):
                logger.error(f"Access denied to dataset {dataset_name}. Please check if you have the correct permissions.")
            else:
                logger.error(f"Error accessing dataset {dataset_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading dataset {dataset_name}: {str(e)}")
            raise

def expand_all_hf_configs(dataset_name):
    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
        logger.info(f"Found configs for {dataset_name}: {configs}")
        return configs
    except Exception as e:
        logger.warning(f"No configs found for {dataset_name}: {e}")
        return [None]

def parse_hf_source(source):
    # Format: hf::dataset[:config][:split][:field][:pct]
    parts = source.split("::", 1)[1].split(":")
    dataset = parts[0]
    config = None
    split = "train"
    field = "auto"
    pct = 1.0
    if len(parts) > 1 and parts[1]:
        config = parts[1]
    if len(parts) > 2 and parts[2]:
        split = parts[2]
    if len(parts) > 3 and parts[3]:
        field = parts[3]
    if len(parts) > 4 and parts[4]:
        try:
            pct = float(parts[4])
        except ValueError:
            logger.warning(f"Invalid pct value '{parts[4]}' in source {source}, defaulting to 1.0")
            pct = 1.0
    return dataset, config, split, field, pct

def train_tokenizer_streaming(sources, vocab_size, output_dir, embedding_dim, max_workers=4, max_samples=None):
    try:
        tokenizer = ByteLevelBPETokenizer()

        def text_iterator():
            for source in sources:
                if source.startswith("hf::"):
                    dataset, config, split, field, pct = parse_hf_source(source)
                    for text in extract_hf_dataset(dataset, config, split, field, pct, streaming=True, max_samples=max_samples):
                        yield text
                elif source.startswith(("http://", "https://")):
                    temp_file = fetch_file_from_url(source)
                    if temp_file:
                        for text in yield_lines_from_file(temp_file):
                            yield text
                        os.remove(temp_file)
                else:
                    if os.path.exists(source):
                        for text in yield_lines_from_file(source):
                            yield text

        # Train directly from iterator
        tokenizer.train_from_iterator(
            text_iterator(),
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS.values())
        )

        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to train tokenizer: {e}", exc_info=True)
        raise

def validate_tokenizer(tokenizer_dir):
    try:
        tokenizer = ByteLevelBPETokenizer.from_file(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt")
        )
        specials = dict(SPECIAL_TOKENS)
        for _, token in specials.items():
            if token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens([token])
        tokenizer.save_model(tokenizer_dir)
        logger.info("Tokenizer validated and saved with special tokens.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to validate or save tokenizer: {e}", exc_info=True)
        sys.exit(1)

def initialize_embedding_matrix(tokenizer, embedding_dim):
    vocab_size = len(tokenizer.get_vocab())
    logger.info(f"Initializing embedding matrix for vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    try:
        weights = torch.empty((vocab_size, embedding_dim))
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        logger.info(f"Embedding matrix shape: {weights.shape}")
        return weights
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}", exc_info=True)
        raise

def save_tokenizer_config(tokenizer_dir, vocab_size, embedding_dim, special_tokens):
    """Save tokenizer configuration to config.json"""
    config = {
        "model_type": "byte_level_bpe",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "special_tokens": special_tokens,
        "max_position_embeddings": 2048,  # Common default value
        "pad_token": special_tokens["pad_token"],
        "bos_token": special_tokens["bos_token"],
        "eos_token": special_tokens["eos_token"],
        "unk_token": special_tokens["unk_token"],
        "mask_token": special_tokens["mask_token"]
    }

    config_path = os.path.join(tokenizer_dir, "config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved tokenizer config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer config: {e}", exc_info=True)
        raise

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--source', multiple=True, help="Corpus source: local file, http(s) URL, or Hugging Face dataset using hf::dataset[:config][:split][:field][:pct] syntax. Use config 'all' for all languages. (Can be repeated)")
@click.option('--tokenizer-dir', help='Directory to save the tokenizer')
@click.option('--vocab-size', type=int, help='Vocabulary size for tokenizer')
@click.option('--min-line-length', type=int, help='Minimum length for a corpus line')
@click.option('--embedding-dim', type=int, help='Embedding dimension for initialization')
@click.option('--max_workers', type=int, help='Maximum parallel dataset loaders')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Set the logging level')
@click.option('--test-mode', is_flag=True, help='Run in test mode with 100 samples per dataset')
def main(config, source, tokenizer_dir, vocab_size, min_line_length, embedding_dim, max_workers, log_level, test_mode):
    # Load config if provided
    if config:
        config_data = load_config(config)
        # Override command line args with config values if not provided
        source = source or config_data.get('sources', [])
        tokenizer_dir = tokenizer_dir or config_data.get('tokenizer', {}).get('output_dir', 'custom_tokenizer')
        vocab_size = vocab_size or config_data.get('tokenizer', {}).get('vocab_size', 52000)
        min_line_length = min_line_length or config_data.get('processing', {}).get('min_line_length', 32)
        embedding_dim = embedding_dim or config_data.get('tokenizer', {}).get('embedding_dim', 1024)
        max_workers = max_workers or config_data.get('processing', {}).get('max_workers', 4)
        log_level = log_level or config_data.get('logging', {}).get('level', 'INFO')
        special_tokens = config_data.get('tokenizer', {}).get('special_tokens', SPECIAL_TOKENS)

    # Set max_samples if in test mode
    max_samples = 100 if test_mode else None

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    try:
        logger.info("Step 1: Train tokenizer with streaming")
        if test_mode:
            logger.info("Running in test mode with 100 samples per dataset")
        tokenizer = train_tokenizer_streaming(source, vocab_size, tokenizer_dir, embedding_dim, max_workers, max_samples)

        logger.info("Step 2: Validate tokenizer")
        tokenizer = validate_tokenizer(tokenizer_dir)

        logger.info("Step 3: Initialize embedding matrix")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)
        np.save(os.path.join(tokenizer_dir, "embedding_matrix.npy"), weights.cpu().numpy())

        logger.info("Step 4: Save tokenizer config")
        save_tokenizer_config(tokenizer_dir, vocab_size, embedding_dim, special_tokens)

        logger.info("All steps completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except huggingface_hub.errors.HfHubHTTPError as e:
        if "403" in str(e):
            logger.error("Access denied to dataset. Please check if you have the correct permissions.")
        else:
            logger.error(f"Error accessing dataset: {str(e)}")
        sys.exit(1)
    except requests.exceptions.ReadTimeout:
        logger.error("Connection timed out. Please check your internet connection and try again.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
