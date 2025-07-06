from tokenizers import ByteLevelBPETokenizer
import os


tokenizer_dir = "sonny_custom_tokenizer"
vocab_path = os.path.join(tokenizer_dir, "vocab.json")
merges_path = os.path.join(tokenizer_dir, "merges.txt")

# Load the tokenizer
if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
    print("Tokenizer files not found. Please train the tokenizer first.")
    exit(1)

tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

# Sample Swedish sentence for fertility test
sample_text = "Det här är en testmening för att kontrollera tokeniseringsfertiliteten."

# Tokenize
output = tokenizer.encode(sample_text)
print(f"Input: {sample_text}")
print(f"Number of tokens: {len(output.tokens)}")
print(f"Tokens: {output.tokens}")
