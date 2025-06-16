#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=True

python train_tokenizer.py --max_workers 8 --tokenizer-out-dir "european_tokenizer" 