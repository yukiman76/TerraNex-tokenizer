# Dataset Downloader

A robust Python script for downloading HuggingFace datasets using the Parquet API, designed to work without `trust_remote_code=True`.

## Features

- **Auto-resume downloads** - Automatically continues interrupted downloads from where they left off
- **Integrity verification** - Validates downloaded Parquet files using pyarrow
- **Smart skip** - Avoids re-downloading files that already exist and pass verification
- **Multi-language support** - Handles datasets with multiple language configurations
- **Progress tracking** - Shows download progress with detailed logging
- **Robust error handling** - Exponential backoff retry with comprehensive error reporting
- **Manifest tracking** - Maintains download state and metadata in JSON manifest

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `requests>=2.28.0`
- `click>=8.0.0`
- `tqdm>=4.64.0`
- `pyarrow>=12.0.0`

## Usage

### Basic Usage

```bash
# Download all datasets (default mode with auto-resume + verification)
python dataset_downloader.py

# Download specific dataset only
python dataset_downloader.py --dataset "bigcode/the-stack-github-issues"

# Only verify existing files (no downloads)
python dataset_downloader.py --verify-only

# Only resume partial downloads
python dataset_downloader.py --resume-only

# Set logging level
python dataset_downloader.py --log-level DEBUG
```

### Command Line Options

- `--dataset NAME` - Download specific dataset only
- `--verify-only` - Only verify existing files, do not download
- `--resume-only` - Only resume existing partial downloads
- `--log-level LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)

## How It Works

1. **Fetches Parquet URLs** - Uses HuggingFace datasets-server API to get Parquet file URLs
2. **Downloads with resume** - Downloads files in chunks with automatic resume capability
3. **Verifies integrity** - Validates file size and Parquet format after download
4. **Tracks progress** - Maintains manifest file with download state and metadata
5. **Handles errors** - Retries failed downloads with exponential backoff

## Included Datasets

The script downloads 13 datasets with 58+ configurations across 13 languages:

### Code Datasets
- `bigcode/the-stack-march-sample-special-tokens-stripped` (1.1G)
- `codeparrot/github-code` (1.1TB) 
- `bigcode/the-stack-github-issues` (66.6G)

### Text Datasets
- `iohadrubin/wikitext-103-raw-v1` (310M)
- `togethercomputer/RedPajama-Data-1T` (2.92G)
- `gutenberg` - Project Gutenberg public domain texts
- `arxiv` - Academic papers
- `cc_news` - News articles

### Conversational Data
- `HuggingFaceH4/ultrachat_200k` - 200k conversations

### Multi-Language Datasets
- **oscar-corpus/mOSCAR** (689G) - 13 languages
- **allenai/c4** - 10 languages  
- **wikipedia** - 10 language versions (March 2022)
- **HuggingFaceFW/fineweb-2** (20TB) - 13 languages

### Supported Languages
Swedish, English, Spanish, German, Danish, French, Italian, Dutch, Norwegian, Polish, Welsh, Finnish

## Adding New Datasets

To add new datasets, modify the `data_sets` dictionary in `dataset_downloader.py`:

### Single Language Dataset

```python
data_sets = {
    "your-org/dataset-name": {
        "field": "text",        # Field name containing the text data
        "extra": [],            # Empty for single-language datasets
    },
}
```

### Multi-Language Dataset

```python
data_sets = {
    "your-org/multilang-dataset": {
        "field": "content",     # Field name containing the text data
        "extra": [              # Language/config codes
            "en",
            "fr", 
            "de",
            "es",
        ],
    },
}
```

### Configuration Fields

- **`field`** - The column name in the Parquet file containing text data
  - Common values: `"text"`, `"content"`, `"code"`, `"messages"`
- **`extra`** - List of language/configuration codes for multi-config datasets
  - For mOSCAR: `"eng_Latn"`, `"fra_Latn"`, etc.
  - For C4: `"en"`, `"fr"`, etc.
  - For Wikipedia: `"20220301.en"`, `"20220301.fr"`, etc.
  - Empty list `[]` for single-config datasets

### Example: Adding a New Dataset

```python
# Add to data_sets dictionary
"microsoft/DialoGPT-medium": {
    "field": "text",
    "extra": [],
},

"facebook/flores": {
    "field": "sentence", 
    "extra": [
        "eng_Latn",
        "fra_Latn", 
        "deu_Latn",
    ],
},
```

## File Structure

```
./datasets/                           # Download directory
├── download_manifest.json            # Download tracking metadata
├── dataset_download.log              # Detailed logging
├── bigcode_the-stack-*.parquet       # Downloaded Parquet files
├── oscar-corpus_mOSCAR_eng_*.parquet # Multi-language files
└── *.partial                         # Partial downloads (auto-resumed)
```

## Output Files

- **Parquet files** - Downloaded dataset files in Parquet format
- **download_manifest.json** - Tracks download state, URLs, and metadata
- **dataset_download.log** - Detailed download logs
- **\*.partial** - Temporary files for interrupted downloads
- **\*.corrupted** - Quarantined corrupted files

## Error Handling

The script handles various error conditions:

- **Network interruptions** - Auto-resume from last downloaded byte
- **Corrupted files** - Move to `.corrupted` and re-download
- **API failures** - Retry with exponential backoff (max 10 attempts)
- **Missing datasets** - Log errors and continue with other datasets
- **Disk space** - Graceful handling of disk full conditions

## Resume Functionality

The script automatically resumes interrupted downloads:

1. **Detects partial files** - Finds `*.partial` files in datasets directory
2. **Looks up URLs** - Retrieves original URLs from manifest file
3. **Continues download** - Uses HTTP Range headers to resume from exact byte position
4. **Verifies completion** - Validates resumed files after completion

## Verification Process

Downloaded files are verified using multiple checks:

1. **File size validation** - Ensures downloaded size matches expected size
2. **Parquet format check** - Uses pyarrow to validate file structure
3. **Metadata verification** - Checks for non-zero row count
4. **Schema validation** - Ensures required columns exist

## Troubleshooting

### Common Issues

**Import Error: pyarrow not found**
```bash
pip install pyarrow>=12.0.0
```

**Permission denied writing to ./datasets**
```bash
mkdir -p ./datasets
chmod 755 ./datasets
```

**Network timeout errors**
- Script automatically retries with exponential backoff
- Check internet connection
- Some datasets are very large (1TB+) and may take hours

**Corrupted download files**
- Script automatically moves corrupted files to `.corrupted` suffix
- Re-run the script to re-download corrupted files
- Check available disk space

### Debug Mode

Run with debug logging for detailed information:
```bash
python dataset_downloader.py --log-level DEBUG
```

### Manual Cleanup

```bash
# Remove all partial downloads
rm ./datasets/*.partial

# Remove corrupted files  
rm ./datasets/*.corrupted

# Reset manifest (will re-download everything)
rm ./datasets/download_manifest.json
```

## Performance

- **Parallel downloads** - Downloads multiple chunks concurrently
- **Efficient resume** - Only downloads missing bytes, not entire files
- **Smart caching** - Skips files that already exist and pass verification
- **Progress tracking** - Real-time progress bars with ETA calculations

Expected download times (depending on connection):
- Small datasets (< 1GB): 5-30 minutes
- Medium datasets (1-10GB): 30 minutes - 2 hours  
- Large datasets (100GB+): 2-24+ hours
- Very large datasets (1TB+): Multiple days
