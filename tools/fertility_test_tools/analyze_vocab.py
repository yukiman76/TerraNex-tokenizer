import json
import os
from tokenizers import ByteLevelBPETokenizer

def is_byte_token(token):
    # Byte-level tokens are usually single bytes or \xNN escapes
    if len(token) == 1 and ord(token) < 256:
        return True
    if token.startswith("<") and token.endswith(">"):
        return False  # special tokens
    if token.startswith("\\x") and len(token) == 4:
        try:
            int(token[2:], 16)
            return True
        except ValueError:
            return False
    return False

def main():
    vocab_path = os.path.join("sonny_custom_tokenizer", "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"File not found: {vocab_path}")
        return
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokens = list(vocab.keys())
    total = len(tokens)
    byte_tokens = [t for t in tokens if is_byte_token(t)]
    num_byte = len(byte_tokens)
    print(f"Total tokens: {total}")
    print(f"Byte-level tokens: {num_byte}")
    print(f"Proportion byte-level: {num_byte/total:.2%}")
    if num_byte/total > 0.5:
        print("WARNING: Vocabulary is likely degenerate (mostly byte-level tokens)")
    else:
        print("Vocabulary is not degenerate (majority are not byte-level tokens)")
    # Optionally print some examples
    print("Sample byte-level tokens:", byte_tokens[:10])
    print("Sample non-byte tokens:", [t for t in tokens if t not in byte_tokens][:10])

if __name__ == "__main__":
    main()
