import os
import sys
import logging
import click
import hashlib
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
import yaml
from typing import List
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
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
                l = line.strip()
                if l:
                    yield l
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
    if len(parts) > 1 and parts[1]: config = parts[1]
    if len(parts) > 2 and parts[2]: split = parts[2]
    if len(parts) > 3 and parts[3]: field = parts[3]
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
    dedup_set = set()
    corpus_lines = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {executor.submit(process_source, source, min_line_length): source for source in sources}
        for future in tqdm(as_completed(future_to_source), total=len(sources), desc="Loading sources in parallel"):
            try:
                source = future_to_source[future]
                lines = future.result()
                logger.info(f"Processing results from {source}: {len(lines):,} lines")
                for line in lines:
                    key = hashlib.sha1(line.encode('utf-8')).hexdigest()
                    if key not in dedup_set:
                        dedup_set.add(key)
                        corpus_lines.append(line)
                logger.info(f"Current corpus size after {source}: {len(corpus_lines):,} lines, {len(dedup_set):,} unique lines")
            except Exception as e:
                logger.error(f"Failed while processing source: {e}", exc_info=True)
                raise
    
    logger.info(f"Final corpus statistics:")
    logger.info(f"- Total lines after deduplication: {len(corpus_lines):,}")
    logger.info(f"- Unique lines: {len(dedup_set):,}")
    logger.info(f"- Duplicate lines removed: {len(dedup_set) - len(corpus_lines):,}")
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
            logger.info(f"Corpus saved successfully:")
            logger.info(f"- File: {abs_path}")
            logger.info(f"- Size: {size:,} bytes")
            logger.info(f"- Lines: {lines:,}")
        else:
            raise Exception(f"File {abs_path} was not created")
            
    except Exception as e:
        logger.error(f"Failed to save corpus: {e}", exc_info=True)
        sys.exit(1)

def stream_text_from_dataset(dataset_name, config=None, split="train", field="text"):
    """Stream text directly from a dataset."""
    try:
        dataset = load_dataset(dataset_name, config, split=split, streaming=True)
        for item in dataset:
            if field in item and item[field]:
                text = str(item[field]).strip()
                if text:
                    yield text
    except Exception as e:
        logger.error(f"Error streaming from {dataset_name}: {e}")

def train_tokenizer_streaming(sources, vocab_size, output_dir, embedding_dim, max_workers=4):
    """Train tokenizer using streaming approach."""
    try:
        tokenizer = ByteLevelBPETokenizer()
        valid_sources = []
        
        # First validate all sources
        logger.info("Validating data sources...")
        for source in sources:
            if source.startswith("hf::"):
                parts = source[4:].split(":")
                dataset_name = parts[0]
                config = parts[1] if len(parts) > 1 else None
                split = parts[2] if len(parts) > 2 else "train"
                field = parts[3] if len(parts) > 3 else "text"
                
                try:
                    # Test if we can access the dataset
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                    next(iter(dataset))  # Try to get first item
                    valid_sources.append((source, dataset_name, config, split, field))
                    logger.info(f"Validated dataset: {dataset_name}")
                except Exception as e:
                    logger.error(f"Error validating dataset {dataset_name}: {e}")
            else:
                # For local files, check if they exist
                if os.path.exists(source):
                    valid_sources.append((source, None, None, None, None))
                    logger.info(f"Validated local file: {source}")
                else:
                    logger.error(f"Local file not found: {source}")
        
        if not valid_sources:
            raise ValueError("No valid data sources found")
        
        def text_iterator():
            for source_info in valid_sources:
                source, dataset_name, config, split, field = source_info
                if dataset_name:  # HuggingFace dataset
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                    for item in dataset:
                        if field in item and item[field]:
                            text = str(item[field]).strip()
                            if text:
                                yield text
                else:  # Local file
                    with open(source, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                yield line

        # Train the tokenizer
        logger.info("Training tokenizer...")
        tokenizer.train_from_iterator(
            text_iterator(),
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS.values())
        )
        
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        
        # Count frequencies
        logger.info("Counting token frequencies...")
        freq = defaultdict(int)
        total = 0
        
        def process_source(source_info):
            nonlocal total
            source, dataset_name, config, split, field = source_info
            try:
                if dataset_name:  # HuggingFace dataset
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                    for item in dataset:
                        if field in item and item[field]:
                            text = str(item[field]).strip()
                            if text:
                                ids = tokenizer.encode(text)
                                for i in ids:
                                    freq[i] += 1
                                    total += 1
                else:  # Local file
                    with open(source, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                ids = tokenizer.encode(line)
                                for i in ids:
                                    freq[i] += 1
                                    total += 1
            except Exception as e:
                logger.error(f"Error processing source {source}: {e}")
        
        # Process sources in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_source, valid_sources))
        
        if total == 0:
            logger.warning("No tokens were processed during frequency counting")
            return tokenizer
        
        # Save token frequency statistics
        stats_file = os.path.join(output_dir, "token_stats.txt")
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
        
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to train tokenizer: {e}", exc_info=True)
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
        vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
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
    """Initialize embedding matrix for the tokenizer."""
    try:
        # Get vocab size from the tokenizer's vocabulary
        vocab_size = len(tokenizer.get_vocab())
        logger.info(f"Initializing embedding matrix for vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
        weights = torch.empty((vocab_size, embedding_dim))
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        logger.info(f"Embedding matrix shape: {weights.shape}")
        return weights
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}", exc_info=True)
        raise

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return {}

def get_dataset_sources_for_language(language: str, config: dict) -> List[str]:
    """Generate dataset sources for a specific language."""
    sources = []
    fallback_datasets = config.get('fallback_datasets', [])
    
    # Map language codes to dataset configs
    lang_configs = {
        'sv': {'oscar': 'swe_Latn', 'cc100': 'sv'},  # Swedish
        'no': {'oscar': 'nor_Latn', 'cc100': 'no'},  # Norwegian
        'da': {'oscar': 'dan_Latn', 'cc100': 'da'},  # Danish
        'fi': {'oscar': 'fin_Latn', 'cc100': 'fi'},  # Finnish
        'de': {'oscar': 'deu_Latn', 'cc100': 'de'},  # German
        'fr': {'oscar': 'fra_Latn', 'cc100': 'fr'},  # French
        'es': {'oscar': 'spa_Latn', 'cc100': 'es'},  # Spanish
        'it': {'oscar': 'ita_Latn', 'cc100': 'it'},  # Italian
        'nl': {'oscar': 'nld_Latn', 'cc100': 'nl'},  # Dutch
        'pl': {'oscar': 'pol_Latn', 'cc100': 'pl'},  # Polish
        'pt': {'oscar': 'por_Latn', 'cc100': 'pt'},  # Portuguese
        'ru': {'oscar': 'rus_Cyrl', 'cc100': 'ru'},  # Russian
        'cs': {'oscar': 'ces_Latn', 'cc100': 'cs'},  # Czech
        'hu': {'oscar': 'hun_Latn', 'cc100': 'hu'},  # Hungarian
        'ro': {'oscar': 'ron_Latn', 'cc100': 'ro'},  # Romanian
        'bg': {'oscar': 'bul_Cyrl', 'cc100': 'bg'},  # Bulgarian
        'el': {'oscar': 'ell_Grek', 'cc100': 'el'},  # Greek
        'hr': {'oscar': 'hrv_Latn', 'cc100': 'hr'},  # Croatian
        'sk': {'oscar': 'slk_Latn', 'cc100': 'sk'},  # Slovak
        'sl': {'oscar': 'slv_Latn', 'cc100': 'sl'},  # Slovenian
    }
    
    if language not in lang_configs:
        logging.warning(f"No dataset configuration found for language: {language}")
        return sources
    
    lang_config = lang_configs[language]
    
    # Add mOSCAR dataset
    if 'oscar-corpus/mOSCAR' in fallback_datasets:
        sources.append(f"hf::oscar-corpus/mOSCAR:{lang_config['oscar']}:train:text:0.05")
    
    # Add CC100 dataset
    if 'statmt/cc100' in fallback_datasets:
        sources.append(f"hf::statmt/cc100:{lang_config['cc100']}:train:text:0.05")
    
    return sources

def main():
    parser = argparse.ArgumentParser(description='Build corpus and train tokenizer')
    parser.add_argument('--source', action='append', help='Data source in format hf::dataset:config:split:field:percentage')
    parser.add_argument('--output-corpus', default='corpus.txt', help='Output corpus file path')
    parser.add_argument('--output-tokenizer', default='custom_tokenizer', help='Output tokenizer directory')
    parser.add_argument('--vocab-size', type=int, default=128000, help='Vocabulary size')
    parser.add_argument('--embedding-dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker processes')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return
    
    # Get languages from config
    languages = config.get('languages', [])
    if not languages:
        logging.error("No languages specified in config file")
        return
    
    # Map language codes to dataset configs
    lang_configs = {
        'sv': {'oscar': 'swe_Latn', 'cc100': 'sv'},  # Swedish
        'no': {'oscar': 'nor_Latn', 'cc100': 'no'},  # Norwegian
        'da': {'oscar': 'dan_Latn', 'cc100': 'da'},  # Danish
        'fi': {'oscar': 'fin_Latn', 'cc100': 'fi'},  # Finnish
        'de': {'oscar': 'deu_Latn', 'cc100': 'de'},  # German
        'fr': {'oscar': 'fra_Latn', 'cc100': 'fr'},  # French
        'es': {'oscar': 'spa_Latn', 'cc100': 'es'},  # Spanish
        'it': {'oscar': 'ita_Latn', 'cc100': 'it'},  # Italian
        'nl': {'oscar': 'nld_Latn', 'cc100': 'nl'},  # Dutch
        'pl': {'oscar': 'pol_Latn', 'cc100': 'pl'},  # Polish
        'pt': {'oscar': 'por_Latn', 'cc100': 'pt'},  # Portuguese
        'ru': {'oscar': 'rus_Cyrl', 'cc100': 'ru'},  # Russian
        'cs': {'oscar': 'ces_Latn', 'cc100': 'cs'},  # Czech
        'hu': {'oscar': 'hun_Latn', 'cc100': 'hu'},  # Hungarian
        'ro': {'oscar': 'ron_Latn', 'cc100': 'ro'},  # Romanian
        'bg': {'oscar': 'bul_Cyrl', 'cc100': 'bg'},  # Bulgarian
        'el': {'oscar': 'ell_Grek', 'cc100': 'el'},  # Greek
        'hr': {'oscar': 'hrv_Latn', 'cc100': 'hr'},  # Croatian
        'sk': {'oscar': 'slk_Latn', 'cc100': 'sk'},  # Slovak
        'sl': {'oscar': 'slv_Latn', 'cc100': 'sl'},  # Slovenian
    }
    
    # Generate sources for all languages
    all_sources = []
    fallback_datasets = config.get('fallback_datasets', [])
    
    for lang in languages:
        if lang not in lang_configs:
            logging.warning(f"No dataset configuration found for language: {lang}")
            continue
            
        lang_config = lang_configs[lang]
        
        # Add mOSCAR dataset
        if 'oscar-corpus/mOSCAR' in fallback_datasets:
            all_sources.append(f"hf::oscar-corpus/mOSCAR:{lang_config['oscar']}:train:text:0.05")
        
        # Add CC100 dataset
        if 'statmt/cc100' in fallback_datasets:
            all_sources.append(f"hf::statmt/cc100:{lang_config['cc100']}:train:text:0.05")
    
    # Add any manually specified sources
    if args.source:
        all_sources.extend(args.source)
    
    if not all_sources:
        logging.error("No valid data sources found")
        return
    
    # Remove duplicates while preserving order
    all_sources = list(dict.fromkeys(all_sources))
    
    logging.info(f"Training tokenizer on {len(all_sources)} sources for {len(languages)} languages")
    for source in all_sources:
        logging.info(f"Source: {source}")
    
    try:
        # Train tokenizer using streaming approach
        train_tokenizer_streaming(
            sources=all_sources,
            output_dir=args.output_tokenizer,
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            max_workers=args.max_workers
        )
        
        # Validate tokenizer
        validate_tokenizer(
            tokenizer_dir=args.output_tokenizer,
            output_dir=args.output_tokenizer,
            sample_size=config.get('sample_size', 100),
            max_workers=args.max_workers
        )
        
        # Initialize embedding matrix
        initialize_embedding_matrix(
            tokenizer_dir=args.output_tokenizer,
            embedding_dim=args.embedding_dim
        )
        
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()
