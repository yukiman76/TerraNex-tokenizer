from datasets import load_dataset


data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped",  # 1.1G
    "codeparrot/github-code", # 1.1 TB
    "bigcode/the-stack-github-issues",  # 66.6 G
    "iohadrubin/wikitext-103-raw-v1",  # 310M
    "oscar-corpus/mOSCAR", #190G
    # "statmt/cc100" trust_remote_cod# 200G
]

for ds in data_sets:
    _ = load_dataset(ds, trust_remote_code=True, num_proc=20)

langs = ["sv", "en", "es", "de", "cy", "da", "fr", "it", "la", "nl", "no"]

for lang in langs:
    _ = load_dataset(
        "SEACrowd/cc100", name=lang, trust_remote_code=True, num_proc=20
    )
