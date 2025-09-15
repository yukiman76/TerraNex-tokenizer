#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=True

# we hould set the vocab size to around 400k for prod but not to exceed 500k
python train_tokenizer_v4.py --max_workers 1 --offline --streaming --vocab-size 128000
