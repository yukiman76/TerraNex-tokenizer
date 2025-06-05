#!/bin/bash
set -e

# Optionally activate your conda environment
CONDA_ENV="tokenizer-env"
CONDA_ENV="TerraNex"
if command -v conda &> /dev/null; then
    echo "Activating conda environment: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

python corpus_builder_and_tokenizer_v2.py \
  --source "hf::oscar-corpus/mOSCAR:all:train:auto:1.0" \
  --source "hf::statmt/cc100:all:train:auto:1.0" \
  --source "hf::bigcode/the-stack-march-sample-special-tokens-stripped::train::content::1.0" \
  --source "hf::codeparrot/github-code::train::code::1.0" \
  --source "hf::bigcode/the-stack-github-issues::train::content::1.0" \
  --source "hf::iohadrubin/wikitext-103-raw-v1::train::text::1.0" \
  --vocab-size 52000 \
  --embedding-dim 1024
   --max_workers 8
  