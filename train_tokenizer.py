import os

os.environ["HF_DATASETS_CACHE"] = "./datasets"
import sys
import json
import click
import torch
import logging
import itertools
import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_dataset
from tokenizers import ByteLevelBPETokenizer

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
            "ita_Latn",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
        ],
    },  # 689G
    "statmt/cc100": {
        "field": "text",
        "extra": ["sv", "en", "es", "de", "cy", "da", "fr", "it", "la", "nl", "no"],
    },  # 713G
}


def load_all_datasets(max_workers=4, streaming=True, sample=None):
    for dataset_name in data_sets:
        if len(data_sets[dataset_name]["extra"]) > 0:
            for lang in data_sets[dataset_name]["extra"]:
                logger.info(f"Processing {dataset_name}.{lang}")
                d = DSLoader()
                d.dataset = load_dataset(
                    dataset_name,
                    name=lang,
                    split="train",
                    streaming=streaming,
                    #    num_proc=max_workers,
                    cache_dir="./datasets",
                )
                # lets set some helpers
                d.affected_field = data_sets[dataset_name]["field"]
                d.dataset_name = f"{dataset_name}.{lang}"

                if sample:
                    shuffled_dataset = d.dataset.shuffle(seed=42)
                    shuffled_dataset.affected_field = data_sets[dataset_name]
                    shuffled_dataset.dataset_name = dataset_name
                    sampled_list = list(itertools.islice(shuffled_dataset, sample))
                    d.dataset = Dataset.from_list(sampled_list)

                yield d
        else:
            d = DSLoader()
            d.dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=streaming,
                #    num_proc=max_workers,
                cache_dir="./datasets",
            )
            # lets set some helpers
            d.affected_field = data_sets[dataset_name]["field"]
            d.dataset_name = dataset_name

            if sample:
                shuffled_dataset = d.dataset.shuffle(seed=42)
                shuffled_dataset.affected_field = data_sets[dataset_name]
                shuffled_dataset.dataset_name = dataset_name
                sampled_list = list(itertools.islice(shuffled_dataset, sample))
                d.dataset = Dataset.from_list(sampled_list)

            yield d


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


def train_tokenizer(vocab_size, output_dir, max_workers):
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        my_datasets = load_all_datasets(max_workers=max_workers, sample=None)
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            batch_iterator(my_datasets),
            vocab_size=vocab_size,
            min_frequency=2,  # TODO: we might need to change this
            special_tokens=list(SPECIAL_TOKENS.values()),
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
        "mask_token": SPECIAL_TOKENS["mask_token"]
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
    help="Maximum parallel dataset loaders",
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
    log_level,
):
    # Configure logging based on command line option
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        logger.info("Step 1: Train tokenizer")
        tokenizer = train_tokenizer(vocab_size, tokenizer_out_dir, max_workers)
        logger.info("Step 3: Validate tokenizer")
        validate_tokenizer(tokenizer_out_dir)
        logger.info("Step 4: Embedding matrix initialization")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)
        np.save(
            os.path.join(tokenizer_out_dir, "embedding_matrix.npy"),
            weights.cpu().numpy(),
        )
        logger.info("Step 5: create tokenizer wrapper")
        save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)
        logger.info("All steps completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
