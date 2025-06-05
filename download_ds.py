from datasets import load_dataset, DownloadConfig

download_config = DownloadConfig(timeout=600.0) # Timeout in seconds

data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped",  # 1.1G
    "codeparrot/github-code", # 1.1 TB
    "bigcode/the-stack-github-issues",  # 66.6 G
    "iohadrubin/wikitext-103-raw-v1",  # 310M
    "oscar-corpus/mOSCAR", #190G
]

for ds in data_sets:
    _ = load_dataset(ds, trust_remote_code=True, num_proc=20)

langs = ["sv", "en", "es", "de", "cy", "da", "fr", "it", "la", "nl", "no"]

# "statmt/cc100" trust_remote_cod# 200G
for lang in langs:
    _ = load_dataset(
        "statmt/cc100", 
        name=lang, 
        trust_remote_code=True, 
        download_config=download_config,
        num_proc=1
    )
