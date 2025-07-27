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
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import click
import requests
from tqdm import tqdm

from datasets import load_dataset

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


# Universal language code mappings (ISO2 -> ISO3)
# This helps match common language codes to their ISO3 equivalents
UNIVERSAL_LANGUAGE_CODES = {
    "sv": ["sv", "swe", "swedish"],
    "en": ["en", "eng", "english"],
    "es": ["es", "spa", "spanish"],
    "de": ["de", "deu", "ger", "german"],
    "da": ["da", "dan", "danish"],
    "fr": ["fr", "fra", "fre", "french"],
    "it": ["it", "ita", "italian"],
    "nl": ["nl", "nld", "dut", "dutch"],
    "no": ["no", "nor", "nob", "nno", "norwegian"],
    "pl": ["pl", "pol", "polish"]
}

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
        self.tier1_parquet_api = 0
        self.tier2_fixed_parquet = 0
        self.tier3_repository_discovery = 0
        self.tier4_huggingface = 0
        self.parquet_api_downloads = 0
        self.huggingface_downloads = 0
        self.start_time = time.time()


class DatasetDownloader:

    def __init__(self, hf_token: Optional[str] = None):
        self.stats = DownloadStats()
        self.session = requests.Session()

        token = hf_token or os.getenv('HF_TOKEN')

        if token:
            logger.info("Using HuggingFace authentication token")

        self.session.headers.update({
            'User-Agent': 'dataset-downloader/1.0',
            'Authorization': f'Bearer {token}'
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

    def validate_parquet_url(self, url: str, dataset_name: str) -> tuple[bool, str]:
        """Validate if Parquet URL is accessible. Returns (is_valid, error_message)"""
        try:
            response = self.session.head(url, timeout=10)
            response.raise_for_status()
            return True, ""
        except Exception as e:
            error_msg = str(e)

            if "404" in error_msg:
                return False, f"Parquet URLs malformed/broken for {dataset_name} (404 Not Found). Using HuggingFace fallback."
            elif "401" in error_msg:
                return False, f"Dataset {dataset_name} requires authentication/gated access (401 Unauthorized). Trying HuggingFace with token."
            elif "403" in error_msg:
                return False, f"Access denied to Parquet files for {dataset_name} (403 Forbidden). Trying HuggingFace method."
            elif "501" in error_msg:
                return False, f"Parquet API not supported for {dataset_name} (501 Not Implemented). Using HuggingFace fallback."
            elif any(code in error_msg for code in ["500", "502", "503"]):
                return False, f"Server error for {dataset_name} ({error_msg}). Will try HuggingFace fallback."
            else:
                return False, f"Network/connection error for {dataset_name}: {error_msg}. Trying HuggingFace fallback."

    def download_via_huggingface(self, dataset_name: str, config: Optional[str] = None, parquet_files_exist: bool = False) -> list[dict[str, Any]]:
        """Download dataset using HuggingFace datasets library and convert to Parquet files"""
        dataset_id = f"{dataset_name}" + (f".{config}" if config else "")
        logger.info(f"Attempting HuggingFace fallback download for {dataset_id}")

        try:
            token = self.session.headers.get('Authorization', '').replace('Bearer ', '')
            if token:
                os.environ['HF_TOKEN'] = token

            logger.info(f"Downloading {dataset_id} via HuggingFace datasets library...")

            if config:
                dataset = load_dataset(
                    dataset_name,
                    name=config,
                    split="train",
                    cache_dir=str(DATASETS_DIR / "hf_cache"),
                    trust_remote_code=False
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split="train",
                    cache_dir=str(DATASETS_DIR / "hf_cache"),
                    trust_remote_code=False
                )

            parquet_files = []

            base_filename = dataset_name.replace('/', '_')
            if config:
                base_filename += f"_{config}"

            parquet_filename = f"{base_filename}_0000.parquet"
            parquet_path = DATASETS_DIR / parquet_filename

            logger.info(f"Converting {dataset_id} to Parquet format: {parquet_filename}")
            df = dataset.to_pandas()
            df.to_parquet(str(parquet_path), index=False)
            file_size = parquet_path.stat().st_size

            parquet_files.append({
                "url": f"huggingface://{dataset_name}" + (f"/{config}" if config else ""),
                "size": file_size,
                "filename": parquet_filename
            })

            logger.info(f"Successfully converted {dataset_id} to Parquet: {parquet_filename} ({file_size:,} bytes)")
            self.stats.huggingface_downloads += 1

            hf_cache_dir = DATASETS_DIR / "hf_cache"
            if hf_cache_dir.exists():
                shutil.rmtree(hf_cache_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary HuggingFace cache for {dataset_id}")

            return parquet_files

        except Exception as e:
            hf_cache_dir = DATASETS_DIR / "hf_cache"
            if hf_cache_dir.exists():
                shutil.rmtree(hf_cache_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary HuggingFace cache after failed download for {dataset_id}")

            if "Dataset scripts are no longer supported" in str(e) and parquet_files_exist:
                logger.warning(f"Traditional loading failed due to deprecated scripts for {dataset_id}: {e}")
                logger.info(f"Trying direct Parquet loading for {dataset_id}")

                original_parquet_files = self.get_parquet_urls(dataset_name, config)
                return self.download_parquet_directly(dataset_name, config, original_parquet_files)
            else:
                logger.error(f"HuggingFace fallback failed for {dataset_id}: {e}")
                return []

    def fix_parquet_url(self, api_url: str) -> str:
        """Convert broken Parquet API URLs to working HTTPS URLs"""

        if not api_url.startswith('https://'):
            return api_url

        fixed_url = api_url.replace('refs%2Fconvert%2Fparquet', 'refs/convert/parquet')
        fixed_url = fixed_url.replace('#', '%23')  # Properly encode # character

        return fixed_url

    def discover_repository_parquet_files(self, dataset_name: str) -> list[str]:
        """Discover all Parquet files in a HuggingFace repository using the API"""
        api_url = f"https://huggingface.co/api/datasets/{dataset_name}/tree/main"

        def explore_directory(path: str = "") -> list[str]:
            """Recursively explore repository directories for Parquet files"""
            parquet_files = []
            url = api_url if not path else f"{api_url}/{path}"

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                files_data = response.json()

                for item in files_data:
                    if item.get('type') == 'directory':
                        subdir_path = f"{path}/{item['path']}" if path else item['path']
                        parquet_files.extend(explore_directory(subdir_path))
                    elif item.get('type') == 'file' and item.get('path', '').endswith('.parquet'):
                        file_path = item['path']
                        file_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{file_path}"
                        file_size = item.get('size')  # Extract file size from API response

                        parquet_files.append({
                            'url': file_url,
                            'path': file_path,
                            'size': file_size
                        })

            except Exception as e:
                logger.debug(f"Failed to explore {url}: {e}")

            return parquet_files

        logger.info(f"Discovering Parquet files in repository {dataset_name}")
        all_files = explore_directory()
        logger.info(f"Found {len(all_files)} Parquet files in repository")

        return all_files

    def detect_language_subsets(self, dataset_name: str, language_codes: list[str]) -> list[str]:
        """Automatically detect and match language codes to dataset subsets using universal patterns"""
        logger.info(f"Auto-detecting language patterns for {dataset_name} with languages: {language_codes}")

        # Get all available configurations/subsets for this dataset
        try:
            api_url = f"https://datasets-server.huggingface.co/splits?dataset={dataset_name}"
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()

            available_configs = set()
            for split_info in data.get('splits', []):
                config_name = split_info.get('config')
                if config_name:
                    available_configs.add(config_name)

            if not available_configs:
                logger.warning(f"No configs found for {dataset_name}, falling back to direct language codes")
                return language_codes

            logger.info(f"Found {len(available_configs)} available configs for {dataset_name}")
            logger.debug(f"Available configs: {sorted(available_configs)}")

            matched_subsets = []

            for lang_code in language_codes:
                # Get all possible variations of this language code
                possible_codes = UNIVERSAL_LANGUAGE_CODES.get(lang_code, [lang_code])

                best_match = None

                for config in available_configs:
                    config_lower = config.lower()

                    # Strategy 1: Direct match
                    if config == lang_code or config_lower == lang_code.lower():
                        best_match = config
                        break

                    # Strategy 2: Check if any language variant appears in config
                    for variant in possible_codes:
                        if variant.lower() in config_lower:
                            best_match = config
                            break

                    if best_match:
                        break

                    # Strategy 3: Pattern matching for common formats
                    # Format: date.language (e.g., "20231101.sv")
                    if '.' in config and config.split('.')[-1].lower() in [c.lower() for c in possible_codes]:
                        best_match = config
                        break

                    # Format: language_script (e.g., "swe_Latn")
                    if '_' in config:
                        config_lang = config.split('_')[0].lower()
                        if config_lang in [c.lower() for c in possible_codes]:
                            best_match = config
                            break

                if best_match:
                    matched_subsets.append(best_match)
                    logger.info(f"Matched language '{lang_code}' to subset '{best_match}'")
                else:
                    logger.warning(f"Could not match language '{lang_code}' to any available subset")
                    logger.info(f"Available subsets for reference: {sorted(list(available_configs)[:10])}...")
                    # Fall back to using the language code as-is
                    matched_subsets.append(lang_code)

            return matched_subsets

        except Exception as e:
            logger.warning(f"Failed to auto-detect language patterns for {dataset_name}: {e}")
            logger.info("Falling back to using language codes directly")
            return language_codes

    def download_parquet_directly(self, dataset_name: str, config: Optional[str] = None, parquet_files_list: list[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Download Parquet files directly using modern HuggingFace approach for datasets with deprecated scripts"""
        dataset_id = f"{dataset_name}" + (f".{config}" if config else "")
        logger.info(f"Attempting direct Parquet loading for {dataset_id}")

        try:
            if not parquet_files_list:
                logger.error(f"No Parquet files list provided for {dataset_id}")
                return []

            logger.info(f"Using {len(parquet_files_list)} Parquet files from API for {dataset_id}")

            data_files = []
            for file_info in parquet_files_list:
                api_url = file_info.get('url', '')
                if not api_url:
                    continue

                fixed_url = self.fix_parquet_url(api_url)
                data_files.append(fixed_url)

            if data_files:
                logger.info(f"First URL example: {data_files[0]}")
                if len(data_files) > 1:
                    logger.info(f"Last URL example: {data_files[-1]}")

            if not data_files:
                logger.error(f"No valid Parquet URLs found for {dataset_id}")
                return []

            logger.info(f"Loading {len(data_files)} Parquet files for {dataset_id}")

            dataset = load_dataset(
                "parquet",
                data_files={"train": data_files},
                split="train"
            )

            safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
            if config:
                safe_dataset_name += f"_{config.replace('-', '_')}"

            parquet_filename = f"{safe_dataset_name}_consolidated.parquet"
            parquet_path = DATASETS_DIR / parquet_filename

            logger.info(f"Converting {dataset_id} to consolidated Parquet file: {parquet_filename}")

            df = dataset.to_pandas()
            df.to_parquet(str(parquet_path), index=False)

            file_size = parquet_path.stat().st_size

            parquet_files = [{
                "url": f"direct_parquet://{dataset_name}",
                "size": file_size,
                "filename": parquet_filename
            }]

            logger.info(f"Successfully loaded {dataset_id} via direct Parquet access: {parquet_filename} ({file_size:,} bytes)")
            self.stats.huggingface_downloads += 1

            return parquet_files

        except Exception as e:
            logger.warning(f"Fixed Parquet URLs failed for {dataset_id}: {e}")
            logger.info(f"Trying repository file discovery for {dataset_id}")

            try:
                repo_parquet_files = self.discover_repository_parquet_files(dataset_name)

                if not repo_parquet_files:
                    logger.error(f"No Parquet files found in repository {dataset_name}")
                    return []

                logger.info(f"Found {len(repo_parquet_files)} Parquet files via repository discovery")
                logger.info(f"First discovered file: {repo_parquet_files[0]}")
                if len(repo_parquet_files) > 1:
                    logger.info(f"Last discovered file: {repo_parquet_files[-1]}")

                dataset = load_dataset(
                    "parquet",
                    data_files={"train": repo_parquet_files},
                    split="train"
                )

                safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
                if config:
                    safe_dataset_name += f"_{config.replace('-', '_')}"

                parquet_filename = f"{safe_dataset_name}_discovered.parquet"
                parquet_path = DATASETS_DIR / parquet_filename

                logger.info(f"Converting {dataset_id} to consolidated Parquet file: {parquet_filename}")

                df = dataset.to_pandas()
                df.to_parquet(str(parquet_path), index=False)

                file_size = parquet_path.stat().st_size

                parquet_files = [{
                    "url": f"repository_discovery://{dataset_name}",
                    "size": file_size,
                    "filename": parquet_filename
                }]

                logger.info(f"Successfully loaded {dataset_id} via repository discovery: {parquet_filename} ({file_size:,} bytes)")
                self.stats.huggingface_downloads += 1

                return parquet_files

            except Exception as repo_error:
                logger.error(f"Repository discovery also failed for {dataset_id}: {repo_error}")
                return []

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
        if filepath.exists():
            existing_size = filepath.stat().st_size
            if not expected_size or expected_size <= 1000 or abs(existing_size - expected_size) <= max(1024, expected_size * 0.1):
                logger.info(f"File {filepath.name} already exists and appears complete ({existing_size} bytes)")
                return True

        partial_path = filepath.with_suffix(filepath.suffix + '.partial')
        resume_byte = 0

        if partial_path.exists():
            partial_size = partial_path.stat().st_size

            if expected_size and abs(partial_size - expected_size) <= max(1024, expected_size * 0.1):
                logger.info(f"Partial file {filepath.name} appears complete, moving to final location")
                if filepath.exists():
                    filepath.unlink()
                partial_path.rename(filepath)
                return True
            elif not expected_size or expected_size <= 1000:
                if partial_size > 1024*1024:  # File is reasonably large, likely complete
                    logger.info(f"Partial file {filepath.name} appears complete (no reliable expected size), moving to final location")
                    if filepath.exists():
                        filepath.unlink()
                    partial_path.rename(filepath)
                    return True

            resume_byte = partial_size
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
                if expected_size and expected_size > 1000 and abs(final_size - expected_size) > max(1024, expected_size * 0.1):
                    raise ValueError(f"File size mismatch: expected {expected_size}, got {final_size}")
                elif expected_size and expected_size <= 1000 and final_size > 1024*1024:
                    logger.warning(f"Ignoring unreliable expected_size {expected_size} for {filepath.name} (actual: {final_size})")
                elif expected_size and abs(final_size - expected_size) > 1024:
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

        if partial_path.exists():
            corrupted_path = filepath.with_suffix(filepath.suffix + '.corrupted')
            partial_path.rename(corrupted_path)
            logger.info(f"Moved partial download to {corrupted_path.name}")

        return False

    def process_dataset(self, dataset_name: str, config: Optional[str] = None) -> bool:
        """Process a single dataset using universal 4-tier fallback system"""
        dataset_id = f"{dataset_name}" + (f".{config}" if config else "")
        logger.info(f"Processing dataset: {dataset_id}")

        # Universal 4-Tier Fallback System
        parquet_files = None
        method_used = None

        # Tier 1: Standard Parquet API
        logger.info(f"Tier 1: Trying standard Parquet API for {dataset_id}")
        parquet_files = self.get_parquet_urls(dataset_name, config)

        if parquet_files:
            first_url = parquet_files[0].get('url')
            if first_url:
                is_valid, error_msg = self.validate_parquet_url(first_url, dataset_name)
                if is_valid:
                    logger.info(f"✓ Tier 1 successful: Standard Parquet API for {dataset_id}")
                    method_used = "tier1_parquet_api"
                    self.stats.tier1_parquet_api += 1
                    self.stats.parquet_api_downloads += 1  # Legacy compatibility
                else:
                    logger.warning(f"Tier 1 failed: {error_msg}")
                    parquet_files = None

        # Tier 2: Fixed Parquet API URLs
        if not parquet_files:
            logger.info(f"Tier 2: Trying fixed Parquet API URLs for {dataset_id}")
            original_files = self.get_parquet_urls(dataset_name, config)
            if original_files:
                fixed_files = []
                for file_info in original_files:
                    if 'url' in file_info:
                        fixed_url = self.fix_parquet_url(file_info['url'])
                        if fixed_url != file_info['url']:
                            fixed_info = file_info.copy()
                            fixed_info['url'] = fixed_url
                            fixed_files.append(fixed_info)

                if fixed_files:
                    is_valid, error_msg = self.validate_parquet_url(fixed_files[0]['url'], dataset_name)
                    if is_valid:
                        logger.info(f"✓ Tier 2 successful: Fixed Parquet URLs for {dataset_id}")
                        parquet_files = fixed_files
                        method_used = "tier2_fixed_parquet"
                        self.stats.tier2_fixed_parquet += 1
                    else:
                        logger.warning(f"Tier 2 failed: {error_msg}")

        # Tier 3: Repository File Discovery
        if not parquet_files:
            logger.info(f"Tier 3: Trying repository file discovery for {dataset_id}")
            try:
                repo_files = self.discover_repository_parquet_files(dataset_name)
                if repo_files:
                    parquet_files = []
                    for file_info in repo_files:
                        file_url = file_info['url']
                        file_path = file_info['path']
                        file_size = file_info.get('size')

                        filename = Path(file_path).name

                        parquet_files.append({
                            'url': file_url,
                            'filename': filename,
                            'path': file_path,
                            'size': file_size
                        })

                    logger.info(f"✓ Tier 3 successful: Repository discovery found {len(parquet_files)} files for {dataset_id}")
                    method_used = "tier3_repository_discovery"
                    self.stats.tier3_repository_discovery += 1
                else:
                    logger.warning("Tier 3 failed: No Parquet files discovered in repository")
            except Exception as e:
                logger.warning(f"Tier 3 failed: Repository discovery error: {e}")

        # Tier 4: Traditional HuggingFace Loading
        if not parquet_files:
            logger.info(f"Tier 4: Trying traditional HuggingFace loading for {dataset_id}")
            try:
                parquet_files = self.download_via_huggingface(dataset_name, config, parquet_files_exist=False)
                if parquet_files:
                    logger.info(f"✓ Tier 4 successful: Traditional HuggingFace loading for {dataset_id}")
                    method_used = "tier4_huggingface"
                    self.stats.tier4_huggingface += 1
                    self.stats.huggingface_downloads += 1  # Legacy compatibility
                else:
                    logger.warning("Tier 4 failed: Traditional loading returned no files")
            except Exception as e:
                logger.warning(f"Tier 4 failed: Traditional loading error: {e}")

        if not parquet_files:
            logger.error(f"All 4 tiers failed for {dataset_id}")
            return False

        logger.info(f"Using method: {method_used} for {dataset_id} ({len(parquet_files)} files)")

        success_count = 0

        for i, file_info in enumerate(parquet_files):
            url = file_info.get('url')
            if not url:
                logger.warning(f"No URL found for file {i} in {dataset_id}")
                continue

            if method_used == "tier4_huggingface":
                filename = file_info.get('filename')
                filepath = DATASETS_DIR / filename
                expected_size = file_info.get('size')

                if self.verify_parquet_integrity(filepath):
                    logger.info(f"Successfully verified HuggingFace-converted file {filename}")
                    self.stats.verified += 1
                    success_count += 1
                else:
                    logger.error(f"HuggingFace-converted file {filename} failed verification")
                    self.stats.corrupted += 1

            else:
                if 'filename' in file_info:
                    filename = file_info['filename']
                else:
                    parsed_url = urlparse(url)
                    filename = Path(parsed_url.path).name
                    if not filename.endswith('.parquet'):
                        filename += '.parquet'

                base_name = dataset_name.replace('/', '_')
                if config:
                    base_name += f"_{config}"

                if method_used == "tier3_repository_discovery" and 'path' in file_info:
                    original_path = file_info['path']
                    path_suffix = original_path.replace('/', '_').replace('\\', '_')
                    if not path_suffix.startswith(base_name):
                        filename = f"{base_name}_{path_suffix}"
                    else:
                        filename = path_suffix
                else:
                    if not filename.startswith(base_name):
                        filename = f"{base_name}_{filename}"

                filepath = DATASETS_DIR / filename
                expected_size = file_info.get('size') or self.get_file_size_from_url(url)

                if filepath.exists() and self.verify_parquet_integrity(filepath):
                    logger.info(f"File {filename} already exists and is valid, skipping")
                    self.stats.skipped += 1
                    self.stats.verified += 1
                    success_count += 1
                else:
                    if self.download_with_resume(url, filepath, expected_size):
                        if self.verify_parquet_integrity(filepath):
                            logger.info(f"Successfully downloaded and verified {filename}")
                            self.stats.verified += 1
                            success_count += 1
                        else:
                            logger.error(f"Downloaded file {filename} failed verification")
                            corrupted_path = filepath.with_suffix(filepath.suffix + '.corrupted')
                            filepath.rename(corrupted_path)
                            self.stats.corrupted += 1

            if dataset_id not in self.manifest["datasets"]:
                self.manifest["datasets"][dataset_id] = {"files": {}, "method": method_used}

            self.manifest["datasets"][dataset_id]["files"][filename] = {
                "url": url,
                "expected_size": expected_size,
                "downloaded": filepath.exists(),
                "verified": filepath.exists() and self.verify_parquet_integrity(filepath),
                "method": method_used,
                "last_attempt": time.time()
            }

            self.save_manifest()

        logger.info(f"Completed {dataset_id}: {success_count}/{len(parquet_files)} files successful")
        return success_count > 0

    def download_all_datasets(self, specific_datasets: tuple[str, ...] = (), subsets: Optional[str] = None, languages: Optional[str] = None) -> bool:
        """Download all datasets (or specific datasets if provided)"""
        logger.info("Starting dataset download process...")

        # Parse languages and subsets
        subset_list = []
        language_list = []

        if languages:
            language_list = [lang.strip() for lang in languages.split(',') if lang.strip()]
            logger.info(f"Using specified languages: {language_list}")

        if subsets:
            subset_list = [s.strip() for s in subsets.split(',') if s.strip()]
            logger.info(f"Using specified subsets: {subset_list}")

        if languages and subsets:
            logger.info("Both --languages and --subsets specified. --languages takes priority.")

        datasets_to_process = {}
        if specific_datasets:
            for dataset_name in specific_datasets:
                if dataset_name in data_sets:
                    dataset_config = data_sets[dataset_name].copy()

                    # Use language auto-detection if languages specified
                    if language_list:
                        detected_subsets = self.detect_language_subsets(dataset_name, language_list)
                        dataset_config["extra"] = detected_subsets
                        logger.info(f"Auto-detected subsets for {dataset_name}: {detected_subsets}")
                    elif subset_list:
                        # Fallback to manual subsets
                        dataset_config["extra"] = subset_list
                        logger.info(f"Using manual subsets for {dataset_name}: {subset_list}")

                    datasets_to_process[dataset_name] = dataset_config
                else:
                    # Use default configuration for arbitrary datasets
                    logger.info(f"Dataset '{dataset_name}' not in pre-configured list, using default settings")

                    final_subsets = []
                    if language_list:
                        # Try auto-detection for arbitrary datasets too
                        final_subsets = self.detect_language_subsets(dataset_name, language_list)
                        logger.info(f"Auto-detected subsets for arbitrary dataset {dataset_name}: {final_subsets}")
                    elif subset_list:
                        final_subsets = subset_list

                    datasets_to_process[dataset_name] = {
                        "field": "text",
                        "extra": final_subsets
                    }
        else:
            if languages or subsets:
                logger.warning("--languages or --subsets specified but no --dataset provided. They will be ignored.")
            datasets_to_process = data_sets

        success_count = 0
        total_datasets = sum(1 + len(config.get("extra", [])) for config in datasets_to_process.values())

        logger.info(f"Processing {total_datasets} dataset configurations...")

        for dataset_name, config in datasets_to_process.items():
            try:
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
        logger.info("\n=== Universal Tier Statistics ===")
        logger.info(f"Tier 1 (Standard Parquet API): {self.stats.tier1_parquet_api} datasets")
        logger.info(f"Tier 2 (Fixed Parquet URLs): {self.stats.tier2_fixed_parquet} datasets")
        logger.info(f"Tier 3 (Repository Discovery): {self.stats.tier3_repository_discovery} datasets")
        logger.info(f"Tier 4 (Traditional HuggingFace): {self.stats.tier4_huggingface} datasets")

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
    multiple=True,
    help='Download specific dataset(s) only. Can be used multiple times.'
)
@click.option(
    '--subsets',
    help='Comma-separated list of subsets/configs to download (use with --dataset)'
)
@click.option(
    '--languages',
    help='Comma-separated language codes (sv,en,es,de,da,fr,it,nl,no,pl) - automatically mapped to dataset format'
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    help='Set logging level'
)
@click.option(
    '--hf-token',
    help='HuggingFace token for authentication (overrides default)'
)
def main(resume_only: bool, verify_only: bool, dataset: tuple[str, ...], subsets: Optional[str], languages: Optional[str], log_level: str, hf_token: Optional[str]):
    """
    Robust Dataset Downloader for HuggingFace Datasets using Parquet API

    By default, downloads all datasets with auto-resume and verification.
    """

    logging.getLogger().setLevel(getattr(logging, log_level))

    downloader = DatasetDownloader(hf_token=hf_token)

    if verify_only:
        logger.info("Running in verify-only mode")
        success = downloader.verify_existing_files()
    elif resume_only:
        logger.info("Running in resume-only mode")
        success = downloader.resume_partial_downloads()
    else:
        logger.info("Running in full download mode (with auto-resume and verification)")
        success = downloader.download_all_datasets(dataset, subsets, languages)

    if success:
        logger.info("Process completed successfully!")
        sys.exit(0)
    else:
        logger.error("Process completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
