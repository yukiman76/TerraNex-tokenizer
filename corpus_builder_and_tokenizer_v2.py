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
    try:
        ds_split = split if max_pct == 1.0 else f"{split}[:{max_pct}%]"
        ds = load_dataset(
            dataset_name, config, split=ds_split,
            streaming=streaming, trust_remote_code=True
        ) if config else load_dataset(
            dataset_name, split=ds_split, streaming=streaming, trust_remote_code=True
        )
        # Field auto-detection
        if field == "auto":
            first = next(iter(ds))
            candidates = [k for k in first.keys() if isinstance(first[k], (str, list))]
            field = candidates[0] if candidates else list(first.keys())[0]
            logger.info(f"[{dataset_name}:{config}] Auto-detected field: {field}")
        for entry in ds:
            value = entry.get(field, None)
            if value:
                if isinstance(value, list):
                    s = " ".join([str(ss) for ss in value if ss])
                else:
                    s = str(value)
                s = s.strip().replace("\n", " ")
                if len(s) >= min_line_length:
                    yield s
    except Exception as e:
        logger.error(f"Failed to load or parse {dataset_name}:{config}: {e}", exc_info=True)
        return

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
    corpus_lines = []
    if source.startswith("hf::"):
        dataset, config, split, field, pct = parse_hf_source(source)
        configs_to_use = [config]
        if (dataset in {"oscar-corpus/mOSCAR", "statmt/cc100"}) and (not config or config.lower() == "all"):
            configs_to_use = expand_all_hf_configs(dataset)
        for c in configs_to_use:
            logger.info(f"Loading dataset: {dataset}, config: {c}")
            for s in extract_hf_dataset(dataset, c, split, field, pct, streaming=True, min_line_length=min_line_length):
                corpus_lines.append(s)
    elif source.startswith("http://") or source.startswith("https://"):
        temp_file = fetch_file_from_url(source)
        if temp_file:
            for s in yield_lines_from_file(temp_file):
                if len(s) >= min_line_length:
                    corpus_lines.append(s)
            os.remove(temp_file)
    else:
        if os.path.exists(source):
            for s in yield_lines_from_file(source):
                if len(s) >= min_line_length:
                    corpus_lines.append(s)
        else:
            logger.warning(f"File not found: {source}")
    return corpus_lines

def build_corpus_parallel(sources, min_line_length, max_workers=4):
    dedup_set = set()
    corpus_lines = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {executor.submit(process_source, source, min_line_length): source for source in sources}
        for future in tqdm(as_completed(future_to_source), total=len(sources), desc="Loading sources in parallel"):
            try:
                lines = future.result()
                for line in lines:
                    key = hashlib.sha1(line.encode('utf-8')).hexdigest()
                    if key not in dedup_set:
                        dedup_set.add(key)
                        corpus_lines.append(line)
            except Exception as e:
                logger.error(f"Failed while processing source: {e}", exc_info=True)
    logger.info(f"Corpus lines after deduplication: {len(corpus_lines):,}")
    return corpus_lines

def save_corpus(corpus, output_file):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for line in tqdm(corpus, desc="Saving corpus"):
                f.write(line + "\n")
        logger.info(f"Saved corpus to {output_file}")
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
def main(source, output_corpus, tokenizer_dir, vocab_size, min_line_length, embedding_dim, preview_chars, max_workers):
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
    except Exception as e:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
