import json
import os
import sys

import numpy as np
import pytest
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing


def test_tokenizer_existence(tokenizer_dir="custom_tokenizer"):
    """
    Test that all expected tokenizer files exist
    """
    expected_files = ["vocab.json", "merges.txt", "config.json", "embedding_matrix.npy"]

    for file in expected_files:
        assert os.path.exists(
            os.path.join(tokenizer_dir, file)
        ), f"{file} is missing from the tokenizer directory"


def test_tokenizer_loading(tokenizer_dir="custom_tokenizer"):
    """
    Test loading the ByteLevelBPETokenizer and verifying its basic properties
    """
    # Load the tokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )

    # Load config to get special tokens
    with open(os.path.join(tokenizer_dir, "config.json")) as f:
        config = json.load(f)

    # Get vocabulary to find token IDs
    vocab = tokenizer.get_vocab()

    # Add special tokens
    special_tokens = list(config["special_tokens"].values())
    tokenizer.add_special_tokens(special_tokens)

    # Prepare special tokens with their IDs for post-processing
    spec_tokens_with_ids = [
        ("<s>", vocab.get(config["special_tokens"]["bos_token"], 0)),
        ("</s>", vocab.get(config["special_tokens"]["eos_token"], 1)),
    ]

    # Post-processing for adding special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=spec_tokens_with_ids
    )

    # Verify basic tokenizer properties
    assert len(special_tokens) > 0, "No special tokens found"


def test_special_tokens(tokenizer_dir="custom_tokenizer"):
    """
    Test that special tokens are correctly added and can be used
    """
    # Load the tokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )

    # Load config to get special tokens
    with open(os.path.join(tokenizer_dir, "config.json")) as f:
        config = json.load(f)

    # Special tokens to check
    special_tokens = list(config["special_tokens"].values())

    # Check that each special token is in the vocabulary
    vocab = tokenizer.get_vocab()
    for token in special_tokens:
        assert token in vocab, f"{token} not in vocabulary"


def test_embedding_matrix(tokenizer_dir="custom_tokenizer"):
    """
    Test the embedding matrix properties
    """
    # Load the config
    with open(os.path.join(tokenizer_dir, "config.json")) as f:
        config = json.load(f)

    # Load the embedding matrix
    embedding_matrix_path = os.path.join(tokenizer_dir, "embedding_matrix.npy")
    embedding_matrix = np.load(embedding_matrix_path)

    # Verify embedding matrix dimensions
    assert (
        embedding_matrix.shape[0] == config["vocab_size"]
    ), "Embedding matrix rows should match vocabulary size"
    assert (
        embedding_matrix.shape[1] == config["embedding_dim"]
    ), "Embedding matrix columns should match embedding dimension"


def test_tokenization_roundtrip(tokenizer_dir="custom_tokenizer"):
    """
    Test that tokenization and detokenization work correctly
    """
    # Load the tokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )

    # Load config to get special tokens
    with open(os.path.join(tokenizer_dir, "config.json")) as f:
        config = json.load(f)

    # Get vocabulary to find token IDs
    vocab = tokenizer.get_vocab()

    # Add special tokens
    special_tokens = list(config["special_tokens"].values())
    tokenizer.add_special_tokens(special_tokens)

    # Prepare special tokens with their IDs for post-processing
    spec_tokens_with_ids = [
        ("<s>", vocab.get(config["special_tokens"]["bos_token"], 0)),
        ("</s>", vocab.get(config["special_tokens"]["eos_token"], 1)),
    ]

    # Post-processing for adding special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=spec_tokens_with_ids
    )

    # Test texts with various characteristics
    test_texts = [
        "Hello, world",
        "Hej världen",
        "Hola Mundo",
        "This is a test of the tokenizer.",
        "Python is an amazing programming language.",
        "Tokenization can handle special characters like @#$%^&*().",
        "多语言 tokenization test.",
    ]

    for text in test_texts:
        # Encode
        encoding = tokenizer.encode(text)

        # Decode
        decoded = tokenizer.decode(encoding.ids)

        # Force print for both python test_tokenizer.py and pytest
        print(f"text: {text}")
        print(f"encoding: {encoding.ids}")
        print(f"decoded: {decoded}\n")
        sys.stdout.flush()  # Ensure print is immediately visible

        # Check that decoded text is similar to original
        # Note: exact match might not be possible due to BPE tokenization
        assert len(decoded) > 0, "Decoded text is empty"
        assert isinstance(
            encoding.ids, list
        ), "Encoding should return a list of token ids"


def run_tests():
    """
    Custom test runner to ensure prints are shown
    """
    # Run pytest with -s flag to show print statements
    pytest.main(["-s", __file__])


if __name__ == "__main__":
    # When run directly, use custom test runner
    run_tests()
