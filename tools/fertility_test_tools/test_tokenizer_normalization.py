from tokenizers import Tokenizer
import sys

# Path to the exported tokenizer.json
TOKENIZER_PATH = "sonny_tokenizer/tokenizer.json"

# Test string (should be normalized: lowercased, tabs/nbsp replaced, control chars removed, etc.)
TEST_STRING = "\t  Example\u00A0Text\nWith\x07Control\x0BChars  "

try:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    sys.exit(1)

# Try to access the normalizer directly (if possible)
try:
    normed = tokenizer.normalizer.normalize_str(TEST_STRING)
    print(f"[NORMALIZED] {normed!r}")
except Exception as e:
    print(f"Could not access normalizer directly: {e}")
    # Fallback: encode and decode to see the effect
    encoding = tokenizer.encode(TEST_STRING)
    decoded = tokenizer.decode(encoding.ids)
    print(f"[ENCODED] {encoding.tokens}")
    print(f"[DECODED] {decoded!r}")
