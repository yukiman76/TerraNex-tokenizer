import os
import json

VOCAB_PATH = os.path.join('test_tokenizer_case', 'vocab.json')
MERGES_PATH = os.path.join('test_tokenizer_case', 'merges.txt')

# 1. Print the number of entries in vocab.json and show a few samples
def check_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"vocab.json entries: {len(vocab)}")
    print("Sample vocab entries:")
    for i, (token, idx) in enumerate(list(vocab.items())[:10]):
        print(f"  {i+1}: {repr(token)} -> {idx}")
    return set(vocab.keys())

# 2. Print the number of merges and show a few samples
def check_merges():
    with open(MERGES_PATH, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"merges.txt entries: {len(lines)}")
    print("Sample merges:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i+1}: {line}")
    return lines

# 3. Check that all tokens in merges are in vocab
def check_consistency(vocab_tokens, merges):
    missing = set()
    for line in merges:
        parts = line.split()
        for part in parts:
            if part not in vocab_tokens:
                missing.add(part)
    if missing:
        print(f"WARNING: {len(missing)} tokens in merges.txt not found in vocab.json:")
        print(list(missing)[:10])
    else:
        print("All tokens in merges.txt are present in vocab.json.")

# 4. Optionally, validate the tokenizer by loading and running encode/decode
def validate_tokenizer():
    try:
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer(VOCAB_PATH, MERGES_PATH)
        test_text = "This is a test sentence for tokenizer validation."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Encode/decode test: '{test_text}' -> {encoded.tokens} -> '{decoded}'")
        if test_text.strip() == decoded.strip():
            print("Tokenizer encode/decode test PASSED.")
        else:
            print("Tokenizer encode/decode test FAILED.")
    except Exception as e:
        print(f"Tokenizer validation failed: {e}")

def check_fertility():
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(VOCAB_PATH, MERGES_PATH)
    # Use a small set of test sentences
    test_texts = [
        "This is a test sentence for tokenizer validation.",
        "Another example sentence to check tokenization.",
        "Short text.",
        "A longer sentence with, perhaps, more punctuation and variety!",
        "Numbers 123456 and symbols #$%& are included as well."
    ]
    total_tokens = 0
    for text in test_texts:
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)
    fertility = total_tokens / len(test_texts)
    print(f"Fertility (tokens/sample) for known tokenizer: {fertility:.4f}")

def main():
    print("--- Checking vocab.json ---")
    vocab_tokens = check_vocab()
    print("\n--- Checking merges.txt ---")
    merges = check_merges()
    print("\n--- Checking consistency ---")
    check_consistency(vocab_tokens, merges)
    print("\n--- Validating tokenizer ---")
    validate_tokenizer()
    print("\n--- Fertility calculation ---")
    check_fertility()

if __name__ == '__main__':
    main()
