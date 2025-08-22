# **Quality first:**

```bash
TMPDIR=/dev/shm exec 3< <(pigz -dc /home/sonny/Code/HPC_tokenizer/cache/counts_merged.jsonl.gz); \
python /home/sonny/Code/HPC_tokenizer/train.py \
  --counts /proc/self/fd/3 \
  --outdir /home/sonny/Code/HPC_tokenizer/artifacts/bpe_nordic_eu_64k_case_qfast \
  --vocab-size 64000 \
  --min-frequency 2 \
  --normalization NFKC \
  --pair-workers 8 \
  --auto-pair-mem-fraction 0.65 \
  --pair-mem-threshold 92 \
  --batch-merges 20000 \
  --embedding-dim 1024 \
  --special-tokens "[PAD],[UNK],[BOS],[EOS],[MASK],<|endoftext|>" \
  --log-level INFO

```



So,<mark> JSON+gzip costs in time.</mark> Train directly from TSV counts in near-one-pass.

# Do this:

1. Use TSV streaming + RAM spills. No pruning. No one-pass shortcuts.

```bash
# Re-merge to TSV (lossless vs JSON)
python /home/sonny/Code/HPC_tokenizer/merge.py \
  --counts-dir /home/sonny/Code/HPC_tokenizer/cache \
  --out-counts /home/sonny/Code/HPC_tokenizer/cache/counts_merged.tsv.gz \
  --out-format tsv --merge-fanin 1024 --resume-merge --log-level INFO

```

2. Train from TSV with parallel decompression and big batches:

```bash

# Train with full quality (min-frequency=2, batch-merges=1000, case-sensitive)
TMPDIR=/dev/shm exec 3< <(pigz -dc /home/sonny/Code/HPC_tokenizer/cache/counts_merged.tsv.gz); \
python /home/sonny/Code/HPC_tokenizer/sharding/train.py \
  --counts /proc/self/fd/3 \
  --outdir /home/sonny/Code/HPC_tokenizer/artifacts/bpe_nordic_eu_64k_case_fullq \
  --vocab-size 64000 \
  --min-frequency 2 \
  --normalization NFKC \
  --pair-workers 8 \
  --auto-pair-mem-fraction 0.7 \
  --pair-mem-threshold 90 \
  --batch-merges 1000 \
  --embedding-dim 1024 \
  --special-tokens "[PAD],[UNK],[BOS],[EOS],[MASK],<|endoftext|>" \
  --log-level INFO

```

Speedup comes only from TSV parse, parallel decompression, RAM tmp, and higher memory headroom.

**Notes:**

- Large `batch-merges` ≈ fewer passes. Slightly less “greedy” than classic multi-pass BPE. Typical impact: small (≈1–3% more tokens) if data is large and diverse.

- Raising `min-frequency` prunes rare but important pairs in morphologically rich Nordic languages. That hurts coverage more.
