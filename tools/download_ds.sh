data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped", #1.1G
    # "codeparrot/github-code", # 1.1 TB
    "bigcode/the-stack-github-issues", #66.6 G
    "iohadrubin/wikitext-103-raw-v1", # 310M
    # "oscar-corpus/mOSCAR", #190G
    # "statmt/cc100" trust_remote_cod# 200G
]
# Open
huggingface-cli download "bigcode/the-stack-march-sample-special-tokens-stripped" --repo-type dataset --local-dir datasets
huggingface-cli download "bigcode/the-stack-github-issues" --repo-type dataset --local-dir datasets

# Gated
huggingface-cli download "iohadrubin/wikitext-103-raw-v1" --repo-type dataset --local-dir datasets

huggingface-cli download "codeparrot/github-code" --repo-type dataset --local-dir datasets
huggingface-cli download "oscar-corpus/mOSCAR" --repo-type dataset --local-dir datasets
huggingface-cli download "statmt/cc100" --repo-type dataset  --local-dir datasets


# sv: Swedish (21G)
# en: English (82G)
# es: Spanish (14G)
# de: German (18G)
# cy: Welsh (179M)
# da: Danish (12G)
# fr: French (14G) ?
# it: Italian (7.8G) ?
# la: Latin (609M)
# nl: Dutch (7.9G)
# no: Norwegian (13G)