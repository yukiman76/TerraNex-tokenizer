import os
import sys
import logging
import click
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import torch
import numpy as np
import requests
import tempfile
from collections import defaultdict

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


def extract_hf_dataset(dataset_name, config=None, split="train", field="text", max_pct=1.0):
    logger.info(f"Loading HuggingFace dataset {dataset_name} config={config} split={split}")
    try:
        ds_split = f"{split}" if max_pct == 1.0 else f"{split}[:{max_pct}%]"
        ds = load_dataset(dataset_name, config, split=ds_split) if config else load_dataset(dataset_name, split=ds_split)
        # Infer field if not provided (try a few common)
        if field == "auto":
            sample = ds[0]
            for try_field in ["text", "code", "content", "data"]:
                if try_field in sample:
                    field = try_field
                    break
            else:
                raise Exception(f"No recognized text field found in sample: {sample}")
        output = []
        for entry in tqdm(ds, desc=f"{dataset_name}/{config or ''}"):
            if field in entry and entry[field]:
                text = str(entry[field]).strip().replace("\n", " ")
                if text:
                    output.append(text)
        logger.info(f"Loaded {len(output):,} lines from {dataset_name} ({config}) split={split} field={field}")
        return output
    except Exception as e:
        logger.error(f"Failed to load or parse {dataset_name}: {e}", exc_info=True)
        return []


def build_corpus(dynamic_sources, min_line_length):
    all_samples = []
    for source in dynamic_sources:
        if source.startswith("hf::"):
            # Format: hf::dataset_name[:config][:split][:field][:pct]
            parts = source.split("::", 1)[1].split(":")
            dataset = parts[0]
            config = parts[1] if len(parts) > 1 and parts[1] else None
            split = parts[2] if len(parts) > 2 and parts[2] else "train"
            field = parts[3] if len(parts) > 3 and parts[3] else "text"
            pct = float(parts[4]) if len(parts) > 4 and parts[4] else 1.0
            samples = extract_hf_dataset(dataset, config, split, field, pct)
            all_samples += samples
        elif source.startswith("http://") or source.startswith("https://"):
            temp_file = fetch_file_from_url(source)
            if temp_file:
                all_samples += list(yield_lines_from_file(temp_file))
                os.remove(temp_file)
        else:
            if os.path.exists(source):
                all_samples += list(yield_lines_from_file(source))
            else:
                logger.warning(f"File not found: {source}")
    before = len(all_samples)
    all_samples = [line for line in all_samples if len(line) >= min_line_length]
    after = len(all_samples)
    logger.info(f"Corpus before filtering: {before:,} lines. After filtering: {after:,} lines.")
    return all_samples


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
            # Only patch if missing or not in vocab
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
@click.option('--source', multiple=True, required=True, help="Corpus source: local file, http(s) URL, or Hugging Face dataset using hf::dataset[:config[:split[:field[:pct]]]] syntax. (Can be repeated)")
@click.option('--output-corpus', default="corpus.txt", show_default=True, help='Path to output corpus txt')
@click.option('--tokenizer-dir', default="custom_tokenizer", show_default=True, help='Directory to save the tokenizer')
@click.option('--vocab-size', default=52000, show_default=True, help='Vocabulary size for tokenizer')
@click.option('--min-line-length', default=32, show_default=True, help='Minimum length for a corpus line')
@click.option('--embedding-dim', default=1024, show_default=True, help='Embedding dimension for initialization')
@click.option('--preview-chars', default=80, show_default=True, help='How many chars of each sample to show in preview logs')
def main(source, output_corpus, tokenizer_dir, vocab_size, min_line_length, embedding_dim, preview_chars):
    try:
        logger.info("Step 1: Build and filter corpus from provided sources")
        corpus = build_corpus(source, min_line_length)
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
