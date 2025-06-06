#!/bin/bash
set -e

# Optionally activate your conda environment
CONDA_ENV="TerraNex"
if command -v conda &> /dev/null; then
    echo "Activating conda environment: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# European languages to include (ISO 639-1 codes)
LANGUAGES=("de" "fr" "es" "it" "nl" "pl" "pt" "ru" "sv" "da" "fi" "no" "cs" "hu" "ro" "bg" "el" "hr" "sk" "sl")

# Create a balanced dataset by sampling 5% from each language
# This ensures equal representation while keeping the corpus size manageable
python corpus_builder_and_tokenizer_v2.py \
  --source "hf::oscar-corpus/mOSCAR:de:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:fr:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:es:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:it:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:nl:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:pl:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:pt:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:ru:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:sv:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:da:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:fi:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:no:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:cs:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:hu:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:ro:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:bg:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:el:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:hr:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:sk:train:auto:0.05" \
  --source "hf::oscar-corpus/mOSCAR:sl:train:auto:0.05" \
  --source "hf::statmt/cc100:de:train:auto:0.05" \
  --source "hf::statmt/cc100:fr:train:auto:0.05" \
  --source "hf::statmt/cc100:es:train:auto:0.05" \
  --source "hf::statmt/cc100:it:train:auto:0.05" \
  --source "hf::statmt/cc100:nl:train:auto:0.05" \
  --source "hf::statmt/cc100:pl:train:auto:0.05" \
  --source "hf::statmt/cc100:pt:train:auto:0.05" \
  --source "hf::statmt/cc100:ru:train:auto:0.05" \
  --source "hf::statmt/cc100:sv:train:auto:0.05" \
  --source "hf::statmt/cc100:da:train:auto:0.05" \
  --source "hf::statmt/cc100:fi:train:auto:0.05" \
  --source "hf::statmt/cc100:no:train:auto:0.05" \
  --source "hf::statmt/cc100:cs:train:auto:0.05" \
  --source "hf::statmt/cc100:hu:train:auto:0.05" \
  --source "hf::statmt/cc100:ro:train:auto:0.05" \
  --source "hf::statmt/cc100:bg:train:auto:0.05" \
  --source "hf::statmt/cc100:el:train:auto:0.05" \
  --source "hf::statmt/cc100:hr:train:auto:0.05" \
  --source "hf::statmt/cc100:sk:train:auto:0.05" \
  --source "hf::statmt/cc100:sl:train:auto:0.05" \
  --vocab-size 100000 \
  --embedding-dim 1024 \
  --max_workers 8 \
  --output-corpus "european_corpus.txt" \
  --tokenizer-dir "european_tokenizer" 