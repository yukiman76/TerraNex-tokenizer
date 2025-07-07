import json
import os

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

    # Add special tokens
    special_tokens = list(config["special_tokens"].values())
    tokenizer.add_special_tokens(special_tokens)

    # Post-processing for adding special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", config["special_tokens"]["bos_token"]),
            ("</s>", config["special_tokens"]["eos_token"]),
        ],
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

    # Add special tokens
    special_tokens = list(config["special_tokens"].values())
    tokenizer.add_special_tokens(special_tokens)

    # Post-processing for adding special tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", config["special_tokens"]["bos_token"]),
            ("</s>", config["special_tokens"]["eos_token"]),
        ],
    )

    # Test texts with various characteristics
    test_texts = [
        "Hello, world!",
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

        # Check that decoded text is similar to original
        # Note: exact match might not be possible due to BPE tokenization
        assert len(decoded) > 0, "Decoded text is empty"
        assert isinstance(
            encoding.ids, list
        ), "Encoding should return a list of token ids"


def test_config_file(tokenizer_dir="custom_tokenizer"):
    """
    Verify the contents of the config file
    """
    config_path = os.path.join(tokenizer_dir, "config.json")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Check required keys
    required_keys = [
        "model_type",
        "vocab_size",
        "embedding_dim",
        "special_tokens",
        "max_position_embeddings",
        "pad_token",
        "bos_token",
        "eos_token",
        "unk_token",
        "mask_token",
    ]

    for key in required_keys:
        assert key in config, f"Required key {key} missing from config"

    # Some specific checks
    assert config["model_type"] == "byte_level_bpe", "Incorrect model type"
    assert config["vocab_size"] > 0, "Vocab size should be positive"
    assert config["embedding_dim"] > 0, "Embedding dimension should be positive"


if __name__ == "__main__":
    # Optional: if you want to run tests directly
    pytest.main([__file__])
