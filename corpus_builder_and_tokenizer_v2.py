import os
import sys
import logging
import click
import tempfile
import requests
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Move logging configuration to after click options
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>"
}

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
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_text = line.strip()
                if line_text:
                    yield line_text
    except Exception as e:
        logger.warning(f"Failed to read from file {file_path}: {e}")

def extract_hf_dataset(dataset_name, config=None, split="train", field="auto", max_pct=1.0, streaming=True, min_line_length=32):
    logger.info(f"Loading dataset {dataset_name} with config={config}, split={split}, field={field}, max_pct={max_pct}")
    try:
        dataset = load_dataset(dataset_name, config, split=split, streaming=streaming)
        logger.info(f"Dataset loaded successfully. Streaming mode: {streaming}")
        
        # First try to use the 'text' field
        if "text" in dataset.features:
            field = "text"
            logger.info(f"Using 'text' field for dataset {dataset_name}")
        else:
            # Look for any field that might contain text
            text_fields = [f for f in dataset.features.keys() 
                         if isinstance(dataset.features[f], (str, list)) 
                         and f not in SPECIAL_TOKENS.keys()]
            
            if text_fields:
                field = text_fields[0]
                logger.info(f"Using field '{field}' for {dataset_name}")
            else:
                logger.warning(f"Could not find any text fields in {dataset_name}")
                field = None
        
        total_lines = 0
        sampled_lines = 0
        for entry in tqdm(dataset, desc=f"Processing {dataset_name}"):
            total_lines += 1
            if total_lines % 10000 == 0:
                logger.info(f"Processed {total_lines:,} lines from {dataset_name}")
            
            if random.random() <= max_pct:
                value = entry.get(field, "")
                # Handle both string and list values
                if isinstance(value, list):
                    text = " ".join(str(item) for item in value if item)
                else:
                    text = str(value)
                
                text = text.strip().replace("\n", " ")
                if len(text) >= min_line_length:
                    sampled_lines += 1
                    if sampled_lines % 1000 == 0:
                        logger.info(f"Sampled {sampled_lines:,} valid lines from {dataset_name}")
                    yield text
        
        logger.info(f"Finished processing {dataset_name}. Total lines: {total_lines:,}, Sampled lines: {sampled_lines:,}")
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}", exc_info=True)
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

def process_source(source, min_line_length):
    logger.info(f"Processing source: {source}")
    corpus_lines = []
    if source.startswith("hf::"):
        dataset, config, split, field, pct = parse_hf_source(source)
        configs_to_use = [config]
        if (dataset in {"oscar-corpus/mOSCAR", "statmt/cc100"}) and (not config or config.lower() == "all"):
            configs_to_use = expand_all_hf_configs(dataset)
            logger.info(f"Expanded configs for {dataset}: {configs_to_use}")
        
        for c in configs_to_use:
            logger.info(f"Loading dataset: {dataset}, config: {c}")
            try:
                for s in extract_hf_dataset(dataset, c, split, field, pct, streaming=True, min_line_length=min_line_length):
                    corpus_lines.append(s)
                logger.info(f"Successfully processed {len(corpus_lines)} lines from {dataset}/{c}")
            except Exception as e:
                logger.error(f"Failed to process {dataset}/{c}: {str(e)}", exc_info=True)
                raise
    elif source.startswith("http://") or source.startswith("https://"):
        logger.info(f"Processing URL source: {source}")
        temp_file = fetch_file_from_url(source)
        if temp_file:
            for s in yield_lines_from_file(temp_file):
                if len(s) >= min_line_length:
                    corpus_lines.append(s)
            os.remove(temp_file)
            logger.info(f"Processed {len(corpus_lines)} lines from URL source")
    else:
        logger.info(f"Processing local file: {source}")
        if os.path.exists(source):
            for s in yield_lines_from_file(source):
                if len(s) >= min_line_length:
                    corpus_lines.append(s)
            logger.info(f"Processed {len(corpus_lines)} lines from local file")
        else:
            logger.warning(f"File not found: {source}")
    
    logger.info(f"Total lines collected from source {source}: {len(corpus_lines):,}")
    return corpus_lines

def build_corpus_parallel(sources, min_line_length, max_workers=4):
    logger.info(f"Starting parallel corpus building with {max_workers} workers")
    corpus_lines = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {executor.submit(process_source, source, min_line_length): source for source in sources}
        for future in tqdm(as_completed(future_to_source), total=len(sources), desc="Loading sources in parallel"):
            try:
                source = future_to_source[future]
                lines = future.result()
                logger.info(f"Processing results from {source}: {len(lines):,} lines")
                corpus_lines.extend(lines)
                logger.info(f"Current corpus size after {source}: {len(corpus_lines):,} lines")
            except Exception as e:
                logger.error(f"Failed while processing source: {e}", exc_info=True)
                raise
    
    logger.info("Final corpus statistics:")
    logger.info(f"- Total lines: {len(corpus_lines):,}")
    return corpus_lines

def save_corpus(corpus, output_file):
    try:
        abs_path = os.path.abspath(output_file)
        logger.info(f"Starting to save corpus to: {abs_path}")
        logger.info(f"Corpus contains {len(corpus):,} lines")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(corpus, desc="Saving corpus")):
                f.write(line + "\n")
                if (i + 1) % 100000 == 0:
                    logger.info(f"Saved {i + 1:,} lines to {abs_path}")
        
        # Verify the file was created and has content
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            lines = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))
            logger.info("Corpus saved successfully:")
            logger.info(f"- File: {abs_path}")
            logger.info(f"- Size: {size:,} bytes")
            logger.info(f"- Lines: {lines:,}")
        else:
            raise Exception(f"File {abs_path} was not created")
            
    except Exception as e:
        logger.error(f"Failed to save corpus: {e}", exc_info=True)
        sys.exit(1)

def train_tokenizer(corpus_file, vocab_size, output_dir):
    try:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS.values())
        )
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to train or save tokenizer: {e}", exc_info=True)
        sys.exit(1)

def validate_tokenizer(tokenizer_dir):
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
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

def preview_tokenization(tokenizer, corpus, num_samples=5, preview_chars=80):
    logger.info("Tokenization preview:")
    for sample in tqdm(corpus[:num_samples], desc="Tokenizing preview"):
        tokens = tokenizer.tokenize(sample)
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        logger.info(
            f"\nSample: {sample[:preview_chars]}{'...' if len(sample) > preview_chars else ''}\n"
            f"Tokens: {tokens}\nToken IDs: {ids}\nDecoded: {decoded}"
        )

def token_frequency_stats(tokenizer, corpus_file, stats_file):
    logger.info("Calculating token frequency statistics...")
    try:
        freq = defaultdict(int)
        total = 0
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Token freq"):
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line)
                for i in ids:
                    freq[i] += 1
                    total += 1
        freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        with open(stats_file, "w", encoding="utf-8") as fstat:
            for i, count in freq_sorted:
                try:
                    tok_str = tokenizer.decode([i])
                except Exception:
                    tok_str = f"<ID_{i}>"
                fstat.write(f"{tok_str}\t{i}\t{count}\n")
        logger.info(f"Token frequency stats saved to {stats_file}. Top 10 tokens:")
        for i, count in freq_sorted[:10]:
            try:
                tok_str = tokenizer.decode([i])
            except Exception:
                tok_str = f"<ID_{i}>"
            logger.info(f"Token: '{tok_str}' (id={i})  count={count}")
    except Exception as e:
        logger.error(f"Failed during token frequency analysis: {e}", exc_info=True)
        sys.exit(1)

def initialize_embedding_matrix(tokenizer, embedding_dim):
    logger.info(f"Initializing embedding matrix for vocab_size={getattr(tokenizer, 'vocab_size', len(tokenizer))}, embedding_dim={embedding_dim}")
    try:
        vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
        weights = torch.empty((vocab_size, embedding_dim))
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        logger.info(f"Embedding matrix shape: {weights.shape}")
        return weights
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}", exc_info=True)
        raise

@click.command()
@click.option('--source', multiple=True, required=True, help="Corpus source: local file, http(s) URL, or Hugging Face dataset using hf::dataset[:config][:split][:field][:pct] syntax. Use config 'all' for all languages. (Can be repeated)")
@click.option('--output-corpus', default="corpus.txt", show_default=True, help='Path to output corpus txt')
@click.option('--tokenizer-dir', default="custom_tokenizer", show_default=True, help='Directory to save the tokenizer')
@click.option('--vocab-size', default=52000, show_default=True, help='Vocabulary size for tokenizer')
@click.option('--min-line-length', default=32, show_default=True, help='Minimum length for a corpus line')
@click.option('--embedding-dim', default=1024, show_default=True, help='Embedding dimension for initialization')
@click.option('--preview-chars', default=80, show_default=True, help='How many chars of each sample to show in preview logs')
@click.option('--max_workers', default=4, show_default=True, help='Maximum parallel dataset loaders')
@click.option('--log-level', default="INFO", type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), help='Set the logging level')
def main(source, output_corpus, tokenizer_dir, vocab_size, min_line_length, embedding_dim, preview_chars, max_workers, log_level):
    # Configure logging based on command line option
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        corpus = build_corpus_parallel(source, min_line_length, max_workers=max_workers)
        save_corpus(corpus, output_corpus)
        logger.info("Step 2: Train tokenizer")
        train_tokenizer(output_corpus, vocab_size, tokenizer_dir)
        logger.info("Step 3: Validate tokenizer")
        tokenizer = validate_tokenizer(tokenizer_dir)
        logger.info("Step 4: Preview tokenization")
        preview_tokenization(tokenizer, corpus, preview_chars=preview_chars)
        logger.info("Step 5: Token frequency statistics")
        token_frequency_stats(tokenizer, output_corpus, os.path.join(tokenizer_dir, "token_stats.txt"))
        logger.info("Step 6: Embedding matrix initialization")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)
        np.save(os.path.join(tokenizer_dir, "embedding_matrix.npy"), weights.cpu().numpy())
        logger.info("All steps completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
