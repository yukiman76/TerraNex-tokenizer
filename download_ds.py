from datasets import load_dataset, DownloadConfig

download_config = DownloadConfig(resume_download=True, max_retries=3)

data_sets = [
    "bigcode/the-stack-march-sample-special-tokens-stripped",  # 1.1G
    "codeparrot/github-code",  # 1.1 TB
    "bigcode/the-stack-github-issues",  # 66.6 G
    "iohadrubin/wikitext-103-raw-v1",  # 310M
]

# download entire DS
for ds in data_sets:
    print(f"Downloading  {ds}")
    _ = load_dataset(ds, trust_remote_code=True, num_proc=20)

langs = ['swe_Latn', 'eng_Latn', 'spa_Latn', 'deu_Latn', 'cym_Latn', 'dan_Latn', 
         'fra_Latn', 'ita_Latn', 'nld_Latn', 'nno_Latn', 'nob_Latn']

# "oscar-corpus/mOSCAR" trust_remote_cod# #190G
for lang in langs:
    print(f"oscar-corpus/mOSCAR  {lang}")
    _ = load_dataset(
       "oscar-corpus/mOSCAR",
        name=lang,
        trust_remote_code=True,
        download_config=download_config,
        num_proc=1,
        max_retries=3,
    )

langs = ["sv", "en", "es", "de", "cy", "da", "fr", "it", "la", "nl", "no"]

# "statmt/cc100" trust_remote_cod# 200G
for lang in langs:
    print(f"statmt/cc100  {lang}")
    _ = load_dataset(
        "statmt/cc100",
        name=lang,
        trust_remote_code=True,
        download_config=download_config,
        num_proc=1,
        max_retries=3,
    )
