from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tokenizers import (
    pre_tokenizers,
    decoders,
    Tokenizer,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import itertools

# need to load the following DS's
# oscar, cc100, wiki, github_issues, github_code
# todo add more DS For non-US native languages
data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped",
    # "codeparrot/github-code",
    "bigcode/the-stack-github-issues",
    "iohadrubin/wikitext-103-raw-v1",
    # "oscar-corpus/mOSCAR",
    "statmt/cc100"
]

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<filename>",
    "<gh_stars>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
]

def global_batch_iterator(datasets, batch_size=1000):
    """
    Combines content from multiple datasets into a single iterator for tokenizer training.
    Handles datasets with 'content', 'text', or 'body' columns.
    """
    iterators = []
    for ds_name in datasets:
        print(f"Loading dataset: {ds_name}")
        try:
            # Attempt to load the train split. Some datasets might only have a default split.
            current_ds = load_dataset(ds_name, split="train", streaming=True)
        except ValueError:
            current_ds = load_dataset(ds_name, split="default", streaming=True) # Fallback for datasets like cc100
        except Exception as e:
            print(f"Could not load dataset {ds_name}: {e}. Skipping.")
            continue

        def get_content_iterator(dataset_stream):
            for item in dataset_stream:
                if 'content' in item:
                    yield item['content']
                elif 'text' in item: # For datasets like wikitext-103 and cc100, mOSCAR
                    yield item['text']
                elif 'body' in item: # For some other potential text datasets
                    yield item['body']
                else:
                    print(f"Warning: No 'content', 'text', or 'body' key found in an item from {ds_name}. Skipping this item.")
                    continue

        iterators.append(get_content_iterator(current_ds))

    # Chain all iterators together
    combined_iterator = itertools.chain(*iterators)

    # Yield batches from the combined iterator
    batch = []
    for item in combined_iterator:
        if item is not None: # Ensure we don't add None if a key wasn't found
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch: # Yield any remaining items in the last batch
        yield batch


VOCAB_SIZE = 49_152

# Pre-tokenizers
digits_pretokenizer = pre_tokenizers.Digits(individual_digits=True)
bytelevel_pretokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

# Decoder
bytelevel_decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)

# Initialize the tokenizer
tokenizer = Tokenizer(BPE())

# Set pre-tokenizer and decoder
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([digits_pretokenizer, bytelevel_pretokenizer])
tokenizer.decoder = bytelevel_decoder

# Initialize the trainer
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True, special_tokens=SPECIAL_TOKENS)

print("Starting tokenizer training...")
# Train the tokenizer using the global batch iterator
tokenizer.train_from_iterator(global_batch_iterator(data_sets), trainer=trainer)
print("Tokenizer training complete!")

# Wrap the trained tokenizer with GPT2TokenizerFast
tokenizer_wrapper = GPT2TokenizerFast(
    tokenizer_object=tokenizer,
    vocab_size=VOCAB_SIZE,
    additional_special_tokens=SPECIAL_TOKENS,
    bos_token=SPECIAL_TOKENS[0],
    eos_token=SPECIAL_TOKENS[0],
    unk_token=SPECIAL_TOKENS[0]
)

# Save the tokenizer
save_path = './TerraNex-tokenizer'
tokenizer_wrapper.save_pretrained(save_path)
print(f"Tokenizer saved to {save_path}")