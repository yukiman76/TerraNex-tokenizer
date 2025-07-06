from transformers import GPT2TokenizerFast
import logging
from pathlib import Path
from sonny_tokenizer_training import MemoryMappedDatasetProcessor

class DummyMemoryProfiler:
    def force_gc_if_needed(self):
        pass

def main():
    logging.basicConfig(level=logging.INFO)
    cache_dir = './datasets'
    processor = MemoryMappedDatasetProcessor(DummyMemoryProfiler(), cache_dir, lowercase=False)
    generator = processor.create_streaming_generator(max_samples_per_dataset=1000)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    print('Testing HuggingFace GPT2TokenizerFast on first 10 samples:')
    for i, sample in enumerate(generator):
        if not sample:
            continue
        enc = tokenizer.encode(sample)
        print(f'Sample {i+1}: {len(enc)} tokens, preview: {sample[:80]!r}')
        if i >= 9:
            break

if __name__ == '__main__':
    main()
