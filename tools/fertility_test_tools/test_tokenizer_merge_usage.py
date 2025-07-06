from tokenizers import ByteLevelBPETokenizer
import os

VOCAB_PATH = os.path.join('sonny_tokenizer', 'vocab.json')
MERGES_PATH = os.path.join('sonny_tokenizer', 'merges.txt')

sample_text = "Det här är ett test på om tokenizer använder merges korrekt."

def main():
    if not os.path.exists(VOCAB_PATH):
        print(f"ERROR: Vocab file not found: {VOCAB_PATH}")
        return
    if not os.path.exists(MERGES_PATH):
        print(f"ERROR: Merges file not found: {MERGES_PATH}")
        return
    try:
        tokenizer = ByteLevelBPETokenizer(VOCAB_PATH, MERGES_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}")
        return
    encoding = tokenizer.encode(sample_text)
    print(f"Input: {sample_text!r}")
    if not encoding.tokens:
        print("ERROR: No tokens produced by the tokenizer!")
        return
    print("Tokens:")
    for i, token in enumerate(encoding.tokens):
        print(f"  {i+1}: {repr(token)} (len={len(token)})")
    max_token_len = max(len(token) for token in encoding.tokens)
    print(f"\nMax token length: {max_token_len}")
    unique_tokens = set(encoding.tokens)
    unique_token_ratio = len(unique_tokens) / len(encoding.tokens) if encoding.tokens else 0
    print(f"Unique token ratio: {unique_token_ratio:.2f}")

if __name__ == '__main__':
    main()
