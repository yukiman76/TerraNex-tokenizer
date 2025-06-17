#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=True

# python train_tokenizer_v2.py --download-only
python train_tokenizer_v2.py --max_workers 20 --offline --streaming