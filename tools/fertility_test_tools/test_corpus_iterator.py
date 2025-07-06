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
    print('First 10 samples from corpus:')
    for i, sample in enumerate(generator):
        print(f'Sample {i+1}:', repr(sample))
        if i >= 9:
            break

if __name__ == '__main__':
    main()
