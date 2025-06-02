data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped",
    # "codeparrot/github-code",
    "bigcode/the-stack-github-issues",
    "iohadrubin/wikitext-103-raw-v1",
    # "oscar-corpus/mOSCAR", ace_Latn
    # "statmt/cc100" trust_remote_code
]
# Open
huggingface-cli download "bigcode/the-stack-march-sample-special-tokens-stripped" --repo-type dataset
huggingface-cli download "bigcode/the-stack-github-issues" --repo-type dataset


# Gated
huggingface-cli download "iohadrubin/wikitext-103-raw-v1" --repo-type dataset





huggingface-cli download "codeparrot/github-code" --repo-type dataset

huggingface-cli download "oscar-corpus/mOSCAR" --repo-type dataset
huggingface-cli download "statmt/cc100" --repo-type dataset