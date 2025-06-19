import fcntl
import json
import logging
import os
import re
import signal
import socket
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import click
import psutil

from datasets import Dataset, load_dataset


class ColoredFormatter(logging.Formatter):

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[37m',     # White
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Apply color to the level name
        record.levelname = f"{color}{record.levelname}{reset}"

        # Format the message
        return super().format(record)


logger = logging.getLogger("🖥️")
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
colored_formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(colored_formatter)
logger.addHandler(console_handler)
logger.propagate = False

CACHE_DIR = os.getenv("HF_CACHE_DIR", "./datasets")
MAX_RETRIES = max(1, min(50, int(os.getenv("HF_MAX_RETRIES", "10"))))
LOCK_EXPIRY_HOURS = max(1, min(168, int(os.getenv("HF_LOCK_EXPIRY_HOURS", "48"))))
RETRY_DELAY = max(1, min(60, int(os.getenv("HF_RETRY_DELAY", "1"))))

VALID_DATASET_PREFIXES = {
    "bigcode/",
    "codeparrot/",
    "iohadrubin/",
    "oscar-corpus/",
    "statmt/",
}

DATA_SETS = {
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
            "ita_Latn",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
            "pol_Latn",
        ],
    },
    "statmt/cc100": {
        "field": "text",
        "extra": [
            "sv",
            "en",
            "es",
            "de",
            "cy",
            "da",
            "fr",
            "it",
            "la",
            "nl",
            "no",
            "pl",
        ],
    },
}

STATUS_PATH = Path(CACHE_DIR) / "_status.json"
LOCKS_DIR = Path(CACHE_DIR) / "_locks"
LOCKS_DIR.mkdir(parents=True, exist_ok=True)

LOCK_TIMEOUT = 30
FILE_OPERATION_TIMEOUT = 60
VERIFICATION_SAMPLES = 5
DOWNLOAD_VERIFICATION_SAMPLES = 3
MAX_MEMORY_GB = 8.0
HF_TOKEN = os.getenv("HF_TOKEN")
SAFE_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")
_last_memory_check = {"time": 0, "value": 0.0}
MEMORY_CHECK_INTERVAL = 5.0  # seconds


class DSLoader:
    dataset: Dataset = None
    affected_field: str = None
    dataset_name: str = None


class DownloadError(Exception):
    pass


class ValidationError(Exception):
    pass


class LockError(Exception):
    pass


class MemoryError(Exception):
    pass


def validate_configuration() -> bool:
    required_fields = ["field", "extra"]
    for dataset_name, config in DATA_SETS.items():
        if not isinstance(config, dict):
            logger.error(f"Invalid configuration for {dataset_name}: not a dict")
            return False
        for field in required_fields:
            if field not in config:
                logger.error(
                    f"Missing field '{field}' in configuration for {dataset_name}"
                )
                return False
        if not isinstance(config["extra"], list):
            logger.error(f"Invalid 'extra' field for {dataset_name}: must be a list")
            return False

    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set - some datasets may require authentication")
        logger.info(
            "💡 Set HF_TOKEN environment variable or run 'huggingface-cli login'"
        )
    else:
        logger.info("✅ HF_TOKEN configured")

    return True


def check_network_connectivity() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False


def check_memory_usage(force_check: bool = False) -> float:
    global _last_memory_check
    current_time = time.time()

    if (
        not force_check
        and (current_time - _last_memory_check["time"]) < MEMORY_CHECK_INTERVAL
    ):
        return _last_memory_check["value"]

    try:
        memory_percent = psutil.virtual_memory().percent
        _last_memory_check = {"time": current_time, "value": memory_percent}
        return memory_percent
    except Exception:
        return 0.0


def validate_dataset_name(dataset_name: str) -> bool:
    return any(dataset_name.startswith(prefix) for prefix in VALID_DATASET_PREFIXES)


def generate_safe_dataset_id(dataset_name: str, lang: Optional[str] = None) -> str:
    if not validate_dataset_name(dataset_name):
        raise ValidationError(f"Invalid dataset name: {dataset_name}")

    safe_name = SAFE_CHAR_PATTERN.sub("_", dataset_name)
    if lang:
        safe_lang = SAFE_CHAR_PATTERN.sub("_", lang)
        return f"{safe_name}.{safe_lang}"
    return safe_name


@contextmanager
def status_file_lock():
    lock_file = STATUS_PATH.with_suffix(".lock")
    file_handle = None

    try:
        file_handle = open(lock_file, "w")
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if file_handle:
            try:
                file_handle.close()
                if lock_file.exists():
                    lock_file.unlink()
            except OSError:
                pass


def load_status() -> dict[str, Any]:
    with status_file_lock():
        try:
            if STATUS_PATH.exists():
                with open(STATUS_PATH, encoding="utf-8") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load status file: {e}")
        return {}


def save_status_optimized(status: dict[str, Any], force_sync: bool = False) -> None:
    try:
        with status_file_lock():
            temp_path = STATUS_PATH.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
                f.flush()
                if force_sync:
                    os.fsync(f.fileno())
            os.rename(temp_path, STATUS_PATH)
    except OSError as e:
        logger.warning(f"Failed to save status: {e}")
        raise


def save_status(status: dict[str, Any]) -> None:
    save_status_optimized(status, force_sync=True)


def lock_path(dataset_id: str) -> Path:
    safe_id = generate_safe_dataset_id(dataset_id)
    return LOCKS_DIR / f"{safe_id}.lock"


@contextmanager
def atomic_lock(dataset_id: str):
    lock_file = lock_path(dataset_id)
    lock_acquired = False
    file_handle = None

    try:
        file_handle = open(lock_file, "w")
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        timestamp = datetime.now(UTC).isoformat()
        file_handle.write(timestamp)
        file_handle.flush()
        os.fsync(file_handle.fileno())

        lock_acquired = True
        logger.debug(f"Acquired lock for {dataset_id}")
        yield True

    except OSError as e:
        logger.debug(f"Could not acquire lock for {dataset_id}: {e}")
        yield False
    except Exception as e:
        logger.error(f"Error in atomic_lock for {dataset_id}: {e}")
        yield False
    finally:
        if file_handle:
            try:
                file_handle.close()
            except OSError:
                pass
        if lock_acquired:
            try:
                if lock_file.exists():
                    lock_file.unlink()
                    logger.debug(f"Released lock for {dataset_id}")
            except OSError:
                pass


def lock_exists(dataset_id: str) -> bool:
    path = lock_path(dataset_id)
    if not path.exists():
        return False

    try:
        with open(path) as f:
            ts_str = f.read().strip()
            ts = datetime.fromisoformat(ts_str)

            if datetime.now(UTC) - ts > timedelta(hours=LOCK_EXPIRY_HOURS):
                logger.warning(f"Stale lock for {dataset_id} expired. Removing.")
                path.unlink()
                return False
            return True
    except (OSError, ValueError) as e:
        logger.warning(f"Invalid lock file for {dataset_id}: {e}")
        try:
            path.unlink()
        except OSError:
            pass
        return False


def cleanup_locks():
    try:
        for lock_file in LOCKS_DIR.glob("*.lock"):
            try:
                with open(lock_file) as f:
                    ts_str = f.read().strip()
                    ts = datetime.fromisoformat(ts_str)

                    if datetime.now(UTC) - ts > timedelta(hours=LOCK_EXPIRY_HOURS):
                        lock_file.unlink()
                        logger.info(f"Cleaned up expired lock: {lock_file.name}")
            except (OSError, ValueError):
                lock_file.unlink()
                logger.info(f"Cleaned up invalid lock: {lock_file.name}")
    except Exception as e:
        logger.error(f"Error during lock cleanup: {e}")


def check_cache_permissions() -> bool:
    try:
        test_file = LOCKS_DIR / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        return True
    except Exception as e:
        logger.error(f"Cache directory not writable: {e}")
        return False


def check_disk_space(required_gb: float = 1.0) -> bool:
    try:
        stat = os.statvfs(CACHE_DIR)
        free_gb = (stat.f_frsize * stat.f_bavail) / (1024**3)
        return free_gb >= required_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True


def download_dataset_with_retry(
    dataset_name: str, lang: Optional[str] = None, max_retries: int = MAX_RETRIES
) -> bool:
    dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name

    if not check_network_connectivity():
        logger.error(f"No network connectivity, skipping {dataset_id}")
        return False

    for attempt in range(max_retries):
        try:
            memory_usage = check_memory_usage()
            if memory_usage > 90.0:
                logger.error(
                    f"Memory usage too high ({memory_usage}%), skipping {dataset_id}"
                )
                return False

            logger.info(
                f"Downloading {dataset_id} (attempt {attempt + 1}/{max_retries}) - Memory: {memory_usage:.1f}%"
            )

            dataset = load_dataset(
                dataset_name,
                name=lang,
                split="train",
                cache_dir=CACHE_DIR,
                streaming=True,
                token=HF_TOKEN,
            )

            sample_count = 0
            try:
                for _ in dataset:
                    sample_count += 1
                    if sample_count >= DOWNLOAD_VERIFICATION_SAMPLES:
                        break

                    if sample_count % 10 == 0:
                        memory_usage = check_memory_usage()
                        if memory_usage > 95.0:
                            logger.warning(
                                f"Memory usage critical ({memory_usage}%), stopping verification"
                            )
                            break

            finally:
                if hasattr(dataset, "close"):
                    dataset.close()

            logger.info(
                f"✓ Downloaded and verified {dataset_id} ({sample_count} samples checked)"
            )
            return True

        except Exception as e:
            error_msg = str(e)
            logger.warning(
                f"✗ Failed to download {dataset_id} (attempt {attempt + 1}): {error_msg}"
            )

            if "rate limit" in error_msg.lower() or "429" in error_msg:
                wait_time = min(60 * (2**attempt), 300)
                logger.info(f"Rate limited, waiting {wait_time}s before retry")
                time.sleep(wait_time)
            elif "not found" in error_msg.lower() or "404" in error_msg:
                logger.error(f"Dataset {dataset_id} not found, skipping")
                return False
            elif "permission" in error_msg.lower() or "403" in error_msg:
                logger.error(f"Permission denied for {dataset_id}, skipping")
                return False
            else:
                time.sleep(RETRY_DELAY)

    logger.error(f"Failed to download {dataset_id} after {max_retries} attempts")
    return False


def download_all_datasets(force: bool = False) -> None:
    logger.info("Starting dataset download process...")

    if not validate_configuration():
        logger.error("Invalid configuration, aborting")
        return

    if not check_cache_permissions():
        logger.error("Cache directory not writable")
        return

    if not check_disk_space(10.0):
        logger.error("Insufficient disk space for downloads")
        return

    if not check_network_connectivity():
        logger.error("No network connectivity")
        return

    status = load_status()
    success_count = 0
    failure_count = 0
    pending_updates = []

    cleanup_locks()

    total_datasets = sum(
        len(config["extra"]) if config["extra"] else 1 for config in DATA_SETS.values()
    )
    processed_datasets = 0

    for dataset_name, config in DATA_SETS.items():
        if not validate_dataset_name(dataset_name):
            logger.error(f"Skipping invalid dataset name: {dataset_name}")
            continue

        items = config["extra"] if config["extra"] else [None]

        for lang in items:
            dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name
            processed_datasets += 1

            logger.info(
                f"Progress: {processed_datasets}/{total_datasets} - Processing {dataset_id}"
            )

            if status.get(dataset_id) == "done" and not force:
                logger.info(f"Skipping {dataset_id}, already marked done.")
                continue

            if lock_exists(dataset_id) and not force:
                logger.info(
                    f"Skipping {dataset_id}, already downloading in another job."
                )
                continue

            with atomic_lock(dataset_id) as lock_acquired:
                if not lock_acquired and not force:
                    logger.info(f"Could not acquire lock for {dataset_id}, skipping.")
                    continue

                if download_dataset_with_retry(dataset_name, lang):
                    status[dataset_id] = "done"
                    success_count += 1
                else:
                    status[dataset_id] = "failed"
                    failure_count += 1

                pending_updates.append(dataset_id)

                if len(pending_updates) >= 5:
                    try:
                        save_status_optimized(
                            status, force_sync=False
                        )  # No fsync for intermediate saves
                        pending_updates.clear()
                    except Exception as e:
                        logger.error(f"Failed to save status: {e}")

    if pending_updates:
        try:
            save_status(status)  # Force sync for final save
        except Exception as e:
            logger.error(f"Failed to save final status: {e}")

    logger.info(
        f"Download process completed. Success: {success_count}, Failures: {failure_count}"
    )


def verify_downloads() -> None:
    """
    🚀 OPTIMIZED: Use cached token and reduced memory checks
    """
    logger.info("Verifying downloaded datasets...")
    status = load_status()
    verified_count = 0
    failed_count = 0

    for dataset_name, config in DATA_SETS.items():
        if not validate_dataset_name(dataset_name):
            continue

        items = config["extra"] if config["extra"] else [None]

        for lang in items:
            dataset_id = f"{dataset_name}.{lang}" if lang else dataset_name

            if status.get(dataset_id) != "done":
                continue

            try:
                dataset = load_dataset(
                    dataset_name,
                    name=lang,
                    split="train",
                    cache_dir=CACHE_DIR,
                    streaming=True,
                    token=HF_TOKEN,
                )

                sample_count = 0
                try:
                    for _ in dataset:
                        sample_count += 1
                        if sample_count >= VERIFICATION_SAMPLES:
                            break
                finally:
                    if hasattr(dataset, "close"):
                        dataset.close()

                logger.info(
                    f"✅ Verified {dataset_id} ({sample_count} samples checked)"
                )
                verified_count += 1

            except Exception as e:
                logger.error(f"❌ Verification failed for {dataset_id}: {e}")
                status[dataset_id] = "corrupt"
                failed_count += 1

    try:
        save_status(status)
    except Exception as e:
        logger.error(f"Failed to save verification status: {e}")

    logger.info(
        f"Verification completed. Verified: {verified_count}, Failed: {failed_count}"
    )


def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_locks()
    sys.exit(0)


@click.command()
@click.option("--download-only", is_flag=True, help="Only download datasets")
@click.option("--verify", is_flag=True, help="Verify downloaded datasets")
@click.option("--force", is_flag=True, help="Force redownload even if done or locked")
@click.option("--cleanup", is_flag=True, help="Clean up expired locks")
def main(download_only: bool, verify: bool, force: bool, cleanup: bool):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if cleanup:
            cleanup_locks()
            return

        if download_only:
            download_all_datasets(force=force)
        if verify:
            verify_downloads()

        if not any([download_only, verify, cleanup]):
            click.echo("No action specified. Use --help for options.")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        cleanup_locks()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        cleanup_locks()
        sys.exit(1)


if __name__ == "__main__":
    main()
