#!/usr/bin/env python3
"""
Wrote a Dataset Downloader for HuggingFace Datasets using Parquet API
Downloads all datasets defined in train_tokenizer_v3.py without trust_remote_code which was the cause of the issue.

Features:
- Auto-resume interrupted downloads
- Integrity verification with re-download of corrupted files
- Smart skip of already downloaded and verified files
- Progress tracking and detailed logging
- Support for multi-language dataset configurations

Usage:
    python dataset_downloader.py                    # Download + Resume + Verify (default)
    python dataset_downloader.py --resume-only      # Only resume partial downloads
    python dataset_downloader.py --verify-only      # Only verify existing files
    python dataset_downloader.py --dataset NAME     # Download specific dataset
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import click
import requests
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("Warning: pyarrow not found. Install with: pip install pyarrow")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_download.log')
    ]
)
logger = logging.getLogger(__name__)


data_sets = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": {
        "field": "content",
        "extra": [],
    },
    "codeparrot/github-code": {"field": "code", "extra": []},
    "bigcode/the-stack-github-issues": {"field": "content", "extra": []},
    "iohadrubin/wikitext-103-raw-v1": {"field": "text", "extra": []},
    "oscar-corpus/mOSCAR": {
        "field": "text",
        "extra": [
            "swe_Latn",
            "eng_Latn",
            "spa_Latn",
            "deu_Latn",
            "cym_Latn",
            "dan_Latn",
            "fra_Latn",
            "fin_Latn",
            "ita_Latn",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
            "pol_Latn",
        ],
    },
    "allenai/c4": {
        "field": "text",
        "extra": [
            "sv",
            "en",
            "es",
            "de",
            "da",
            "fr",
            "it",
            "nl",
            "no",
            "pl",
        ],
    },
    "togethercomputer/RedPajama-Data-1T": {
        "field": "text",
        "extra": [],
    },
    "HuggingFaceH4/ultrachat_200k": {
        "field": "messages",
        "extra": [],
    },
    "gutenberg": {
        "field": "text",
        "extra": [],
    },
    "arxiv": {"field": "text", "extra": []},
    "wikipedia": {
        "field": "text",
        "extra": [
            "20220301.sv",
            "20220301.en",
            "20220301.es",
            "20220301.de",
            "20220301.da",
            "20220301.fr",
            "20220301.it",
            "20220301.nl",
            "20220301.no",
            "20220301.pl",
        ],
    },
    "cc_news": {"field": "text", "extra": []},
    "HuggingFaceFW/fineweb-2": {
        "field": "text",
        "extra": [
            "swe_Latn",
            "eng_Latn",
            "spa_Latn",
            "deu_Latn",
            "cym_Latn",
            "dan_Latn",
            "fra_Latn",
            "fin_Latn",
            "ita_Latn",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
            "pol_Latn",
        ],
    },
}

DATASETS_DIR = Path("./datasets")
MANIFEST_FILE = DATASETS_DIR / "download_manifest.json"
CHUNK_SIZE = 8192
MAX_RETRIES = 10
TIMEOUT = 300


class DownloadStats:
    def __init__(self):
        self.downloaded = 0
        self.resumed = 0
        self.verified = 0
        self.failed = 0
        self.skipped = 0
        self.corrupted = 0
        self.total_bytes = 0
        self.start_time = time.time()


class DatasetDownloader:

    def __init__(self):
        self.stats = DownloadStats()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'dataset-downloader/1.0'
        })

        DATASETS_DIR.mkdir(exist_ok=True)

        self.manifest = self.load_manifest()

    def load_manifest(self) -> dict[str, Any]:
        if MANIFEST_FILE.exists():
            try:
                with open(MANIFEST_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")

        return {
            "version": "1.0",
            "datasets": {},
            "last_updated": None
        }

    def save_manifest(self):
        self.manifest["last_updated"] = time.time()
        try:
            with open(MANIFEST_FILE, 'w') as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def get_parquet_urls(self, dataset_name: str, config: Optional[str] = None) -> list[dict[str, Any]]:
        api_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
        if config:
            api_url += f"&config={config}"

        logger.info(f"Fetching Parquet URLs for {dataset_name}" + (f" (config: {config})" if config else ""))

        try:
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()

            data = response.json()
            if 'parquet_files' in data:
                parquet_files = data['parquet_files']
                logger.info(f"Found {len(parquet_files)} Parquet files for {dataset_name}")
                return parquet_files
            else:
                logger.warning(f"No parquet_files found in API response for {dataset_name}")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Parquet URLs for {dataset_name}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {dataset_name}: {e}")
            return []

    def get_file_size_from_url(self, url: str) -> Optional[int]:
        try:
            response = self.session.head(url, timeout=30)
            response.raise_for_status()

            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
        except Exception as e:
            logger.debug(f"Failed to get file size for {url}: {e}")

        return None

    def verify_parquet_integrity(self, filepath: Path) -> bool:
        if not filepath.exists():
            return False

        try:
            file_size = filepath.stat().st_size
            if file_size == 0:
                logger.warning(f"File {filepath.name} is empty")
                return False

            if HAS_PYARROW:
                try:
                    parquet_file = pq.ParquetFile(filepath)
                    metadata = parquet_file.metadata
                    if metadata.num_rows == 0:
                        logger.warning(f"Parquet file {filepath.name} has no rows")
                        return False

                    logger.debug(f"Verified Parquet file {filepath.name}: {metadata.num_rows} rows")
                    return True

                except Exception as e:
                    logger.error(f"Parquet validation failed for {filepath.name}: {e}")
                    return False
            else:
                logger.debug(f"Verified file {filepath.name} (pyarrow not available)")
                return True

        except Exception as e:
            logger.error(f"File integrity check failed for {filepath.name}: {e}")
            return False

    def download_with_resume(self, url: str, filepath: Path, expected_size: Optional[int] = None) -> bool:
        partial_path = filepath.with_suffix(filepath.suffix + '.partial')
        resume_byte = 0

        if partial_path.exists():
            resume_byte = partial_path.stat().st_size
            logger.info(f"Resuming download of {filepath.name} from byte {resume_byte}")
            self.stats.resumed += 1

        headers = {}
        if resume_byte > 0:
            headers['Range'] = f'bytes={resume_byte}-'

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Downloading {filepath.name} (attempt {attempt + 1}/{MAX_RETRIES})")

                response = self.session.get(url, headers=headers, stream=True, timeout=TIMEOUT)
                response.raise_for_status()

                total_size = expected_size
                if 'Content-Length' in response.headers:
                    content_length = int(response.headers['Content-Length'])
                    total_size = resume_byte + content_length if resume_byte > 0 else content_length

                mode = 'ab' if resume_byte > 0 else 'wb'
                with open(partial_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=resume_byte,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=filepath.name[:50]
                    ) as pbar:

                        downloaded_this_session = 0
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                chunk_size = len(chunk)
                                downloaded_this_session += chunk_size
                                pbar.update(chunk_size)

                                self.stats.total_bytes += chunk_size

                final_size = partial_path.stat().st_size
                if expected_size and abs(final_size - expected_size) > 1024:
                    raise ValueError(f"File size mismatch: expected {expected_size}, got {final_size}")

                if filepath.exists():
                    filepath.unlink()
                partial_path.rename(filepath)

                logger.info(f"Successfully downloaded {filepath.name} ({final_size:,} bytes)")
                self.stats.downloaded += 1
                return True

            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {filepath.name}: {e}")

                if attempt < MAX_RETRIES - 1:
                    wait_time = min(2 ** attempt, 32)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

                    if partial_path.exists():
                        resume_byte = partial_path.stat().st_size
                        headers = {'Range': f'bytes={resume_byte}-'} if resume_byte > 0 else {}

        logger.error(f"Failed to download {filepath.name} after {MAX_RETRIES} attempts")
        self.stats.failed += 1

        # Move partial file to corrupted if it exists
        if partial_path.exists():
            corrupted_path = filepath.with_suffix(filepath.suffix + '.corrupted')
            partial_path.rename(corrupted_path)
            logger.info(f"Moved partial download to {corrupted_path.name}")

        return False

    def process_dataset(self, dataset_name: str, config: Optional[str] = None) -> bool:
        """Process a single dataset (with optional config)"""
        dataset_id = f"{dataset_name}" + (f".{config}" if config else "")
        logger.info(f"Processing dataset: {dataset_id}")

        # Get Parquet URLs
        parquet_files = self.get_parquet_urls(dataset_name, config)
        if not parquet_files:
            logger.error(f"No Parquet files found for {dataset_id}")
            return False

        success_count = 0

        for i, file_info in enumerate(parquet_files):
            url = file_info.get('url')
            if not url:
                logger.warning(f"No URL found for file {i} in {dataset_id}")
                continue

            # Generate filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename.endswith('.parquet'):
                filename += '.parquet'

            # Create unique filename for multi-config datasets
            if config:
                filename = f"{dataset_name.replace('/', '_')}_{config}_{filename}"
            else:
                filename = f"{dataset_name.replace('/', '_')}_{filename}"

            filepath = DATASETS_DIR / filename

            # Check if file already exists and is valid
            if filepath.exists() and self.verify_parquet_integrity(filepath):
                logger.info(f"File {filename} already exists and is valid, skipping")
                self.stats.skipped += 1
                self.stats.verified += 1
                success_count += 1
                continue

            # Get expected file size
            expected_size = file_info.get('size') or self.get_file_size_from_url(url)

            # Download file
            if self.download_with_resume(url, filepath, expected_size):
                # Verify downloaded file
                if self.verify_parquet_integrity(filepath):
                    logger.info(f"Successfully downloaded and verified {filename}")
                    self.stats.verified += 1
                    success_count += 1
                else:
                    logger.error(f"Downloaded file {filename} failed verification")
                    # Move corrupted file
                    corrupted_path = filepath.with_suffix(filepath.suffix + '.corrupted')
                    filepath.rename(corrupted_path)
                    self.stats.corrupted += 1

            # Update manifest after each file
            if dataset_id not in self.manifest["datasets"]:
                self.manifest["datasets"][dataset_id] = {"files": {}}

            self.manifest["datasets"][dataset_id]["files"][filename] = {
                "url": url,
                "expected_size": expected_size,
                "downloaded": filepath.exists(),
                "verified": filepath.exists() and self.verify_parquet_integrity(filepath),
                "last_attempt": time.time()
            }

            self.save_manifest()

        logger.info(f"Completed {dataset_id}: {success_count}/{len(parquet_files)} files successful")
        return success_count > 0

    def download_all_datasets(self, specific_dataset: Optional[str] = None) -> bool:
        """Download all datasets (or specific dataset if provided)"""
        logger.info("Starting dataset download process...")

        datasets_to_process = {}
        if specific_dataset:
            if specific_dataset in data_sets:
                datasets_to_process = {specific_dataset: data_sets[specific_dataset]}
            else:
                logger.error(f"Dataset '{specific_dataset}' not found in configuration")
                return False
        else:
            datasets_to_process = data_sets

        success_count = 0
        total_datasets = sum(1 + len(config.get("extra", [])) for config in datasets_to_process.values())

        logger.info(f"Processing {total_datasets} dataset configurations...")

        for dataset_name, config in datasets_to_process.items():
            try:
                # Process main dataset
                if self.process_dataset(dataset_name):
                    success_count += 1

                # Process language/config variants
                for lang_config in config.get("extra", []):
                    if self.process_dataset(dataset_name, lang_config):
                        success_count += 1

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue


        elapsed_time = time.time() - self.stats.start_time
        logger.info("\n=== Download Summary ===")
        logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Datasets processed: {success_count}/{total_datasets}")
        logger.info(f"Files downloaded: {self.stats.downloaded}")
        logger.info(f"Files resumed: {self.stats.resumed}")
        logger.info(f"Files verified: {self.stats.verified}")
        logger.info(f"Files skipped: {self.stats.skipped}")
        logger.info(f"Files failed: {self.stats.failed}")
        logger.info(f"Files corrupted: {self.stats.corrupted}")
        logger.info(f"Total bytes downloaded: {self.stats.total_bytes/1024/1024/1024:.2f} GB")

        return success_count > 0

    def verify_existing_files(self) -> bool:
        """Verify integrity of existing downloaded files"""
        logger.info("Verifying existing downloaded files...")

        verified_count = 0
        corrupted_count = 0

        for filepath in DATASETS_DIR.glob("*.parquet"):
            if self.verify_parquet_integrity(filepath):
                logger.info(f"✓ {filepath.name} is valid")
                verified_count += 1
            else:
                logger.error(f"✗ {filepath.name} is corrupted")
                # Move to corrupted
                corrupted_path = filepath.with_suffix(filepath.suffix + '.corrupted')
                filepath.rename(corrupted_path)
                corrupted_count += 1

        logger.info(f"Verification complete: {verified_count} valid, {corrupted_count} corrupted")
        return corrupted_count == 0

    def find_url_in_manifest(self, filename: str) -> str | None:
        """Find URL for filename in manifest"""
        for _dataset_id, dataset_info in self.manifest.get("datasets", {}).items():
            for file_name, file_info in dataset_info.get("files", {}).items():
                if file_name == filename:
                    return file_info.get("url")
        return None

    def resume_partial_downloads(self) -> bool:
        """Resume any partial downloads found"""
        logger.info("Checking for partial downloads to resume...")

        partial_files = list(DATASETS_DIR.glob("*.partial"))
        if not partial_files:
            logger.info("No partial downloads found")
            return True

        logger.info(f"Found {len(partial_files)} partial downloads to resume")

        resumed_count = 0
        for partial_path in partial_files:
            original_path = partial_path.with_suffix('')
            original_filename = original_path.name
            logger.info(f"Attempting to resume {original_filename}...")

            url = self.find_url_in_manifest(original_filename)
            if url:
                if self.download_with_resume(url, original_path):
                    resumed_count += 1
                    logger.info(f"Successfully resumed {original_filename}")
                else:
                    logger.error(f"Failed to resume {original_filename}")
            else:
                logger.warning(f"Cannot resume {original_filename} - URL not found in manifest")

        logger.info(f"Resume complete: {resumed_count}/{len(partial_files)} files resumed")
        return resumed_count > 0


@click.command()
@click.option(
    '--resume-only',
    is_flag=True,
    help='Only resume existing partial downloads'
)
@click.option(
    '--verify-only',
    is_flag=True,
    help='Only verify existing files, do not download'
)
@click.option(
    '--dataset',
    help='Download specific dataset only'
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    help='Set logging level'
)
def main(resume_only: bool, verify_only: bool, dataset: Optional[str], log_level: str):
    """
    Robust Dataset Downloader for HuggingFace Datasets using Parquet API

    By default, downloads all datasets with auto-resume and verification.
    """

    logging.getLogger().setLevel(getattr(logging, log_level))

    downloader = DatasetDownloader()

    if verify_only:
        logger.info("Running in verify-only mode")
        success = downloader.verify_existing_files()
    elif resume_only:
        logger.info("Running in resume-only mode")
        success = downloader.resume_partial_downloads()
    else:
        logger.info("Running in full download mode (with auto-resume and verification)")
        success = downloader.download_all_datasets(dataset)

    if success:
        logger.info("Process completed successfully!")
        sys.exit(0)
    else:
        logger.error("Process completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
