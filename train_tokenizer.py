import os

os.environ["HF_DATASETS_CACHE"] = "./datasets"
import sys
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

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped",  # 1.1G
    "codeparrot/github-code",  # 1.1 TB
    # "bigcode/the-stack-github-issues",  # 66.6 G
    # "iohadrubin/wikitext-103-raw-v1",  # 310M
]


def get_field(dataset):
    if "text" in dataset.features:
        field = "text"
    else:
        text_fields = []
        for f in dataset.features.keys():
            if dataset.features[f].dtype in ("string"):
                if f not in SPECIAL_TOKENS.keys():
                    text_fields.append(f)

        if text_fields:
            field = text_fields[0]
        else:
            field = None

    return field


def load_all_datasets(max_workers=4, streaming=True, sample=None):
    for dataset_name in data_sets:
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=streaming,
            #    num_proc=max_workers,
            cache_dir="./datasets",
        )
        if sample:
            shuffled_dataset = dataset.shuffle(seed=42)
            sampled_list = list(itertools.islice(shuffled_dataset, sample))
            sampled_dataset = Dataset.from_list(sampled_list)

        yield sampled_dataset


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
    for dataset in tqdm(my_datasets, desc="Processing"):
        field = get_field(dataset)
        for i in tqdm(dataset, desc=f"Processing dataset {i_ds} "):
            k = dataset[field]
            if isinstance(k, list):
                s = "".join(k)
            else:
                s = k
            for p in range(0, len(s), batch_size):
                # print(s[p: p+batch_size])
                yield s[p: p+batch_size]
        i_ds += 1


def train_tokenizer(vocab_size, output_dir, max_workers):
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        my_datasets = load_all_datasets(max_workers=max_workers, sample=100)
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


def save_tokenizer_config(tokenizer_dir):
    from transformers import PreTrainedTokenizerFast

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
        vocab_file=f"{tokenizer_dir}/vocab.json",
        merges_file=f"{tokenizer_dir}/merges.txt",
        unk_token=SPECIAL_TOKENS["unk_token"],
        pad_token=SPECIAL_TOKENS["pad_token"],
        bos_token=SPECIAL_TOKENS["bos_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
    )

    fast_tokenizer.save_pretrained(tokenizer_dir)


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
        save_tokenizer_config(tokenizer_out_dir)
        logger.info("All steps completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
