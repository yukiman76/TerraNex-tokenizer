#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=True

python train_tokenizer_v3.py --max_workers 1 --offline --streaming --vocab-size 256000
