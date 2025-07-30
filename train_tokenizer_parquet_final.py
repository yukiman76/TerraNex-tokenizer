import os
import time  # Sonny ---> Added for time tracking

os.environ["HF_DATASETS_CACHE"] = "./datasets"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5000"  # 5 minutes
import json
import logging
import sys

import click
import numpy as np
import psutil
import torch
from safetensors.torch import save_file
from tokenizers import ByteLevelBPETokenizer, normalizers
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from datasets import Dataset, load_dataset

# Move logging configuration to after click options
logger = logging.getLogger(__name__)


class DSLoader:
    dataset: Dataset = None
    affected_field: str = None
    dataset_name: str = None


SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}

# Dataset classification and sampling configuration
# Total available: Nordic=1226GB, English=28GB, European=2665GB, Code=647GB
# Realistic ratios for 1500GB: English=28GB(2%), Nordic=375GB(25%), European=872GB(58%), Code=225GB(15%)
DATASET_LANGUAGE_MAP = {
    # Nordic languages (target: 25% = 375GB from 1226GB total)
    ("four-two-labs/culturax-nord", None): ("nordic", 922293.667, 0.35),  # 35% sampling → 323GB
    ("OskarLiew/swedish-sentence-pairs", None): ("nordic", 397.752, 1.0),
    ("AI-Sweden-Models/BiaSWE", None): ("nordic", 0.103, 1.0),
    ("strombergnlp/nordic_langid", None): ("nordic", 25.043, 1.0),
    ("HuggingFaceFW/fineweb-2", "swe_Latn"): ("nordic", 86177.324, 1.0),
    ("HuggingFaceFW/fineweb-2", "dan_Latn"): ("nordic", 62740.632, 1.0),
    ("HuggingFaceFW/fineweb-2", "fin_Latn"): ("nordic", 58859.524, 1.0),
    ("HuggingFaceFW/fineweb-2", "nno_Latn"): ("nordic", 13.898, 1.0),
    ("HuggingFaceFW/fineweb-2", "nob_Latn"): ("nordic", 75598.415, 1.0),
    ("allenai/c4", "da"): ("nordic", 2786.974, 1.0),
    ("allenai/c4", "sv"): ("nordic", 2842.154, 1.0),
    ("allenai/c4", "no"): ("nordic", 2832.71, 1.0),
    ("oscar-corpus/mOSCAR", "dan_Latn"): ("nordic", 2949.088, 1.0),
    ("oscar-corpus/mOSCAR", "swe_Latn"): ("nordic", 2924.454, 1.0),
    ("oscar-corpus/mOSCAR", "fin_Latn"): ("nordic", 2900.636, 1.0),
    ("oscar-corpus/mOSCAR", "nno_Latn"): ("nordic", 250.215, 1.0),
    ("oscar-corpus/mOSCAR", "nob_Latn"): ("nordic", 2929.566, 1.0),
    
    # English foundation (target: 2% = 28GB - use ALL available English data)
    ("togethercomputer/RedPajama-Data-1T", None): ("english", 2626.851, 1.0),
    ("allenai/c4", None): ("english", 5000.0, 1.0),  # Base c4 dataset - primarily English
    ("allenai/c4", "en"): ("english", 2941.557, 1.0),
    ("HuggingFaceH4/ultrachat_200k", None): ("english", 698.116, 1.0),
    ("common-pile/arxiv_papers", None): ("english", 2254.236, 1.0),
    ("iohadrubin/wikitext-103-raw-v1", None): ("english", 145.704, 1.0),
    ("bjoernp/1-sentence-level-gutenberg-en_arxiv_pubmed_soda", None): ("english", 16065.094, 1.0),
    ("oscar-corpus/mOSCAR", "eng_Latn"): ("english", 2977.288, 1.0),
    
    # European languages (target: 58% = 872GB from 2665GB total)
    ("HuggingFaceFW/fineweb-2", "deu_Latn"): ("european", 731818.733, 0.33),  # 33% → 242GB
    ("HuggingFaceFW/fineweb-2", "fra_Latn"): ("european", 510328.865, 0.33),  # 33% → 168GB
    ("HuggingFaceFW/fineweb-2", "spa_Latn"): ("european", 603508.703, 0.33),  # 33% → 199GB
    ("HuggingFaceFW/fineweb-2", "ita_Latn"): ("european", 335889.676, 0.33),  # 33% → 111GB
    ("HuggingFaceFW/fineweb-2", "nld_Latn"): ("european", 176642.159, 0.33),  # 33% → 58GB
    ("HuggingFaceFW/fineweb-2", "pol_Latn"): ("european", 210828.198, 0.33),  # 33% → 70GB
    ("allenai/c4", "de"): ("european", 2878.298, 1.0),
    ("allenai/c4", "fr"): ("european", 2803.531, 1.0),
    ("allenai/c4", "es"): ("european", 2874.962, 1.0),
    ("allenai/c4", "it"): ("european", 2924.462, 1.0),
    ("allenai/c4", "nl"): ("european", 2838.958, 1.0),
    ("allenai/c4", "pl"): ("european", 3007.54, 1.0),
    ("oscar-corpus/mOSCAR", "deu_Latn"): ("european", 35641.371, 1.0),
    ("oscar-corpus/mOSCAR", "fra_Latn"): ("european", 2899.733, 1.0),
    ("oscar-corpus/mOSCAR", "spa_Latn"): ("european", 2799.59, 1.0),
    ("oscar-corpus/mOSCAR", "ita_Latn"): ("european", 2840.544, 1.0),
    ("oscar-corpus/mOSCAR", "nld_Latn"): ("european", 2902.906, 1.0),
    ("oscar-corpus/mOSCAR", "pol_Latn"): ("european", 2847.898, 1.0),
    ("sedthh/gutenberg_multilang", None): ("european", 1822.975, 1.0),
    ("wikimedia/wikipedia", None): ("european", 9576.832, 1.0),
    
    # Code/Technical (target: 15% = 225GB from 647GB total)
    ("codeparrot/github-code", None): ("code", 274481.649, 0.35),  # 35% → 96GB
    ("codeparrot/github-code-clean", None): ("code", 299274.559, 0.35),  # 35% → 105GB
    ("bigcode/the-stack-github-issues", None): ("code", 63472.385, 0.35),  # 35% → 22GB
    ("microsoft/rStar-Coder", None): ("code", 9304.05, 0.35),  # 35% → 3GB
    ("bigcode/the-stack-march-sample-special-tokens-stripped", None): ("code", 1056.049, 1.0),
    
    # Mixed language datasets (need classification based on content)
    ("oscar-corpus/mOSCAR", None): ("european", 50000.0, 0.3),  # Mixed European languages
    ("HuggingFaceFW/fineweb-2", None): ("european", 200000.0, 0.3),  # Mixed European languages
    
    # Other datasets (conservative sampling)
    ("Armindvd/persian-wiki-cleaned", None): ("other", 347.593, 0.1),
    ("HuggingFaceFW/fineweb-2", "cym_Latn"): ("other", 11.887, 1.0),
    ("oscar-corpus/mOSCAR", "cym_Latn"): ("other", 81.219, 1.0),
    ("manu/project_gutenberg", None): ("other", 11398.747, 0.2),
    ("matthh/gutenberg-poetry-corpus", None): ("other", 76.505, 1.0),
}

def validate_language_ratios(nordic_ratio, european_ratio, english_ratio, code_ratio, other_ratio):
    """Validate that language ratios are valid and sum to approximately 1.0"""
    ratios = [nordic_ratio, european_ratio, english_ratio, code_ratio, other_ratio]
    
    # Check all ratios are non-negative
    for ratio in ratios:
        if ratio < 0:
            raise ValueError(f"All language ratios must be >= 0, got: {ratio}")
    
    # Check ratios sum to approximately 1.0 (with small tolerance for floating point)
    total = sum(ratios)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Language ratios must sum to 1.0, got: {total:.3f}. "
                        f"Nordic: {nordic_ratio}, European: {european_ratio}, "
                        f"English: {english_ratio}, Code: {code_ratio}, Other: {other_ratio}")
    
    return True


def build_language_categories(nordic_ratio=0.25, european_ratio=0.58, english_ratio=0.02, 
                              code_ratio=0.15, other_ratio=0.0):
    """Build LANGUAGE_CATEGORIES dynamically from CLI arguments with fallback to defaults"""
    
    # Validate ratios first
    validate_language_ratios(nordic_ratio, european_ratio, english_ratio, code_ratio, other_ratio)
    
    # Build the categories dict with same structure as original
    return {
        "nordic": {"target_ratio": nordic_ratio, "priority": 1},      # Nordic focus
        "english": {"target_ratio": english_ratio, "priority": 2},    # English content
        "european": {"target_ratio": european_ratio, "priority": 3},  # European languages
        "code": {"target_ratio": code_ratio, "priority": 4},          # Code/technical
        "other": {"target_ratio": other_ratio, "priority": 5},        # Other languages
        "unknown": {"target_ratio": 0.0, "priority": 6}              # Fallback for new datasets
    }


# Language category configuration - Default ratios (used as fallback)
LANGUAGE_CATEGORIES = {
    "nordic": {"target_ratio": 0.25, "priority": 1},    # 375GB Nordic focus
    "english": {"target_ratio": 0.02, "priority": 2},   # 28GB all available English
    "european": {"target_ratio": 0.58, "priority": 3},  # 872GB major European languages
    "code": {"target_ratio": 0.15, "priority": 4},      # 225GB code/technical
    "other": {"target_ratio": 0.0, "priority": 5},      # Minimal inclusion
    "unknown": {"target_ratio": 0.0, "priority": 6}     # Fallback for new datasets
}


def parse_dataset_filename(filename):
    """Parse dataset filename to extract dataset name and language subset
    
    Handles multiple filename formats:
    - Simple: HuggingFaceFW_fineweb-2_swe_Latn_0001.parquet -> ('HuggingFaceFW/fineweb-2', 'swe_Latn')
    - Complex: codeparrot_github-code-clean_data_train-00012-of-00880.parquet -> ('codeparrot/github-code-clean', None)
    - No subset: oscar-corpus_mOSCAR_0025.parquet -> ('oscar-corpus/mOSCAR', None)
    - HF without subset: HuggingFaceFW_fineweb-2_0173.parquet -> ('HuggingFaceFW/fineweb-2', None)
    """
    import re
    
    # Remove extensions
    name = filename.replace('.parquet', '').replace('.partial', '')
    parts = name.split('_')
    
    if len(parts) < 2:
        logger.warning(f"Unexpected filename format: {filename}")
        return None, None
    
    # Known language codes for detection
    LANGUAGE_CODES = {
        # Nordic languages
        'da', 'sv', 'no', 'swe_Latn', 'dan_Latn', 'fin_Latn', 'nno_Latn', 'nob_Latn',
        # European languages  
        'de', 'fr', 'es', 'it', 'nl', 'pl', 'deu_Latn', 'fra_Latn', 'spa_Latn', 
        'ita_Latn', 'nld_Latn', 'pol_Latn',
        # English
        'en', 'eng_Latn',
        # Other
        'cym_Latn'
    }
    
    # Check for different number patterns at the end
    # Pattern 1: Simple number like _0001
    # Pattern 2: Complex like _data_train-00012-of-00880
    # Pattern 3: Complex like train-00012-of-00880
    
    # Find the ending number pattern
    last_part = parts[-1]
    number_found = False
    
    # Check for simple 4-digit number
    if re.match(r'^\d{4}$', last_part):
        number_found = True
        content_parts = parts[:-1]
    
    # Check for complex train-XXXXX-of-XXXXX pattern
    elif re.match(r'^train-\d+-of-\d+$', last_part):
        number_found = True
        # Remove both 'data' and train pattern parts
        if len(parts) >= 3 and parts[-2] == 'data':
            content_parts = parts[:-2]
        else:
            content_parts = parts[:-1]
    
    # Check if last part looks like a number (fallback)
    elif re.search(r'\d', last_part):
        number_found = True
        content_parts = parts[:-1]
    
    if not number_found:
        logger.warning(f"No number pattern found in filename: {filename}")
        return None, None
    
    if len(content_parts) < 2:
        logger.warning(f"Insufficient parts after number removal: {filename}")
        return None, None
    
    # Check for language codes at the end (before the number/data parts)
    language_subset = None
    dataset_parts = content_parts
    
    # Check for lang_Latn pattern (2 parts)
    if len(content_parts) >= 3:
        potential_lang = f"{content_parts[-2]}_{content_parts[-1]}"
        if potential_lang in LANGUAGE_CODES:
            language_subset = potential_lang
            dataset_parts = content_parts[:-2]
    
    # Check for single language code
    if not language_subset and len(content_parts) >= 2:
        potential_lang = content_parts[-1]
        if potential_lang in LANGUAGE_CODES:
            language_subset = potential_lang
            dataset_parts = content_parts[:-1]
    
    # Reconstruct dataset name
    if len(dataset_parts) < 2:
        logger.warning(f"Could not extract org/dataset from: {filename}")
        return None, None
    
    org = dataset_parts[0]
    dataset = '_'.join(dataset_parts[1:])  # Handle datasets with underscores
    dataset_name = f"{org}/{dataset}"
    
    return dataset_name, language_subset


def get_dataset_sampling_info(dataset_name, language_subset=None):
    """Get language category and sampling ratio for a dataset"""
    key = (dataset_name, language_subset)
    
    if key in DATASET_LANGUAGE_MAP:
        category, size_mb, sampling_ratio = DATASET_LANGUAGE_MAP[key]
        return category, sampling_ratio, size_mb
    
    # Fallback for unknown datasets
    logger.warning(f"Unknown dataset: {dataset_name} (subset: {language_subset}) - using conservative sampling")
    return "unknown", 0.2, 0.0  # Conservative 20% sampling for unknown datasets


def validate_dataset_completeness(discovered_datasets):
    """Validate that all expected datasets are present"""
    expected_datasets = set(dataset_name for (dataset_name, _), _ in DATASET_LANGUAGE_MAP.items())
    found_datasets = set(discovered_datasets.keys())
    
    missing_datasets = expected_datasets - found_datasets
    unexpected_datasets = found_datasets - expected_datasets
    
    if missing_datasets:
        logger.warning(f"⚠️  Missing expected datasets: {sorted(missing_datasets)}")
        for dataset in missing_datasets:
            # Find size from mapping
            for (ds_name, subset), (category, size_mb, ratio) in DATASET_LANGUAGE_MAP.items():
                if ds_name == dataset:
                    logger.warning(f"   - {dataset}: {size_mb/1024:.1f}GB {category} data will be missing")
                    break
    
    if unexpected_datasets:
        logger.info(f"📦 Found unexpected datasets: {sorted(unexpected_datasets)}")
        logger.info("   These will use fallback 20% sampling ratio")
    
    logger.info(f"✅ Dataset validation: {len(found_datasets)}/{len(expected_datasets)} expected datasets found")
    return len(missing_datasets) == 0


def validate_file_accessibility(file_paths):
    """Validate that files exist and are readable"""
    from pathlib import Path
    
    missing_files = []
    total_size = 0
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(file_path)
        elif not path.is_file():
            missing_files.append(f"{file_path} (not a file)")
        else:
            try:
                total_size += path.stat().st_size
            except (OSError, PermissionError) as e:
                missing_files.append(f"{file_path} (read error: {e})")
    
    if missing_files:
        logger.error(f"❌ File accessibility issues found:")
        for issue in missing_files[:10]:  # Show first 10 issues
            logger.error(f"   - {issue}")
        if len(missing_files) > 10:
            logger.error(f"   ... and {len(missing_files)-10} more issues")
        return False
    
    logger.info(f"✅ File validation: {len(file_paths)} files accessible, {total_size/1024**3:.1f}GB total")
    return True


def log_sampling_plan_summary(target_data_gb=1500):
    """Log the sampling plan summary based on DATASET_LANGUAGE_MAP"""
    category_totals = {cat: 0.0 for cat in LANGUAGE_CATEGORIES.keys()}
    
    # Calculate totals from the mapping
    for key, (category, size_mb, sampling_ratio) in DATASET_LANGUAGE_MAP.items():
        category_totals[category] += size_mb * sampling_ratio
    
    total_sampled_size = sum(category_totals.values())
    total_original_size = sum(size_mb for _, size_mb, _ in DATASET_LANGUAGE_MAP.values())
    
    logger.info(f"=== DATASET SAMPLING PLAN ===")
    logger.info(f"Target total data: {target_data_gb:.1f} GB")
    logger.info(f"Original total data: {total_original_size/1024:.1f} GB") 
    logger.info(f"Planned sampled data: {total_sampled_size/1024:.1f} GB")
    logger.info(f"Overall sampling ratio: {total_sampled_size/total_original_size:.3f}")
    
    for category, target_info in LANGUAGE_CATEGORIES.items():
        if category_totals[category] > 0:
            target_gb = target_data_gb * target_info['target_ratio']
            actual_gb = category_totals[category] / 1024
            percentage = (actual_gb / (total_sampled_size/1024)) * 100 if total_sampled_size > 0 else 0
            status = "✓" if abs(percentage - target_info['target_ratio']*100) < 3 else "⚠"
            
            logger.info(f"{category.upper()}: {actual_gb:.1f}GB ({percentage:.1f}%) target={target_info['target_ratio']*100:.1f}% {status}")


def get_parquet_metadata(file_paths):
    """Get row counts and sizes from local parquet files using PyArrow"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("PyArrow not available, falling back to file size estimation")
        from pathlib import Path
        total_size_gb = sum(Path(f).stat().st_size for f in file_paths) / (1024**3)
        return None, total_size_gb

    total_rows = 0
    total_size_gb = 0

    for file_path in file_paths:
        try:
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            total_rows += metadata.num_rows
            # Fix: Use actual file size instead of tiny metadata header size
            from pathlib import Path
            total_size_gb += Path(file_path).stat().st_size / (1024**3)
        except Exception as e:
            logger.warning(f"Could not read metadata for {file_path}: {e}")
            # Fallback to file size
            from pathlib import Path
            total_size_gb += Path(file_path).stat().st_size / (1024**3)

    return total_rows, total_size_gb


def auto_detect_field(file_path):
    """Auto-detect the text field in a parquet file"""
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)

        # Get column names
        schema = parquet_file.schema_arrow
        column_names = [field.name for field in schema]

        # Try common field names in order of preference
        preferred_fields = ["text", "content", "code", "messages"]

        for field in preferred_fields:
            if field in column_names:
                logger.info(f"Auto-detected field '{field}' in {file_path}")
                return field

        # If no preferred field found, try to find any string-like field
        for field_name in column_names:
            if any(keyword in field_name.lower() for keyword in ["text", "content", "message", "code", "body"]):
                logger.info(f"Auto-detected field '{field_name}' in {file_path}")
                return field_name

        # Default to first string field
        if column_names:
            first_field = column_names[0]
            logger.warning(f"Using first field '{first_field}' for {file_path}")
            return first_field

    except Exception as e:
        logger.error(f"Failed to auto-detect field in {file_path}: {e}")

    # Last resort
    return "text"


def discover_local_parquet_files(target_data_gb=1500, enable_sampling=True):
    """Discover and group local parquet files with intelligent sampling"""
    from pathlib import Path
    import random

    datasets_dir = Path("./datasets")
    if not datasets_dir.exists():
        logger.info("No ./datasets directory found")
        return {}

    parquet_files = list(datasets_dir.glob("*.parquet"))
    if not parquet_files:
        logger.info("No parquet files found in ./datasets directory")
        return {}

    logger.info(f"Found {len(parquet_files)} parquet files in ./datasets directory")

    # Validate file accessibility early
    all_file_paths = [str(f) for f in parquet_files]
    if not validate_file_accessibility(all_file_paths):
        logger.error("❌ File validation failed - some files are not accessible")
        return {}

    # Group files by dataset and language subset
    all_files = {}
    
    for file_path in parquet_files:
        filename = file_path.name
        dataset_name, language_subset = parse_dataset_filename(filename)
        
        if not dataset_name:
            continue
            
        # Create nested structure: dataset -> language_subset -> files
        if dataset_name not in all_files:
            all_files[dataset_name] = {}
        
        subset_key = language_subset if language_subset else "main"
        if subset_key not in all_files[dataset_name]:
            all_files[dataset_name][subset_key] = []
        
        all_files[dataset_name][subset_key].append(str(file_path))

    # Validate dataset completeness
    if not validate_dataset_completeness(all_files):
        logger.warning("⚠️  Some expected datasets are missing - proceeding with available data")

    if not enable_sampling:
        # Return all files without sampling
        logger.info("Sampling disabled - using all available files")
        return all_files

    # Log the sampling plan
    log_sampling_plan_summary(target_data_gb)
    
    # Apply sampling to files
    sampled_files = {}
    total_files_before = 0
    total_files_after = 0
    
    # Set random seed once for reproducible sampling across all datasets  
    random.seed(42)
    
    for dataset_name, subsets in all_files.items():
        for subset_key, file_paths in subsets.items():
            total_files_before += len(file_paths)
            
            # Find sampling ratio for this dataset/subset combination
            language_subset = None if subset_key == "main" else subset_key
            category, sampling_ratio, _ = get_dataset_sampling_info(dataset_name, language_subset)
            
            if sampling_ratio < 1.0:
                # Apply sampling (seed already set above)
                sampled_count = max(1, int(len(file_paths) * sampling_ratio))
                sampled_file_paths = random.sample(file_paths, sampled_count)
            else:
                sampled_file_paths = file_paths
            
            total_files_after += len(sampled_file_paths)
            
            # Store in output structure
            if dataset_name not in sampled_files:
                sampled_files[dataset_name] = {}
            sampled_files[dataset_name][subset_key] = sampled_file_paths
            
            # Log sampling decision
            if len(sampled_file_paths) != len(file_paths):
                subset_str = f" ({language_subset})" if language_subset else ""
                logger.info(f"Sampled {dataset_name}{subset_str}: {len(file_paths)} → {len(sampled_file_paths)} files ({sampling_ratio:.2f} ratio)")

    logger.info(f"=== SAMPLING SUMMARY ===")
    logger.info(f"Total files: {total_files_before} → {total_files_after} ({total_files_after/total_files_before:.3f} ratio)")
    
    # Global scaling stage to enforce target_data_gb limit
    # Calculate current total planned size from sampled files
    total_planned_size_gb = 0.0
    for dataset_name, subsets in sampled_files.items():
        for subset_files in subsets.values():
            if subset_files:  # Only calculate if files exist
                _, size_gb = get_parquet_metadata(subset_files)
                total_planned_size_gb += size_gb
    
    # Apply global scaling if planned size exceeds target
    if total_planned_size_gb > target_data_gb:
        global_scale_factor = target_data_gb / total_planned_size_gb
        logger.info(f"=== GLOBAL SCALING REQUIRED ===")
        logger.info(f"Planned size: {total_planned_size_gb:.1f}GB exceeds target: {target_data_gb:.1f}GB")
        logger.info(f"Applying global scale factor: {global_scale_factor:.3f}")
        
        # Apply scaling to all datasets proportionally
        scaled_files = {}
        for dataset_name, subsets in sampled_files.items():
            scaled_files[dataset_name] = {}
            for subset_key, file_paths in subsets.items():
                if file_paths:
                    # Calculate scaled file count, ensure at least 1 file if original had files
                    original_count = len(file_paths)
                    scaled_count = max(1, int(original_count * global_scale_factor))
                    
                    # Sample down to scaled count (reproducible with seed already set)
                    if scaled_count < original_count:
                        scaled_file_paths = random.sample(file_paths, scaled_count)
                    else:
                        scaled_file_paths = file_paths
                    
                    scaled_files[dataset_name][subset_key] = scaled_file_paths
                    
                    # Log significant scaling changes
                    if scaled_count != original_count:
                        language_subset = None if subset_key == "main" else subset_key
                        subset_str = f" ({language_subset})" if language_subset else ""
                        logger.info(f"Global scaled {dataset_name}{subset_str}: {original_count} → {scaled_count} files")
                else:
                    scaled_files[dataset_name][subset_key] = file_paths
        
        # Replace sampled_files with scaled version
        sampled_files = scaled_files
        
        # Recalculate final totals for verification
        final_total_gb = 0.0
        for dataset_name, subsets in sampled_files.items():
            for subset_files in subsets.values():
                if subset_files:
                    _, size_gb = get_parquet_metadata(subset_files)
                    final_total_gb += size_gb
        
        logger.info(f"Final scaled size: {final_total_gb:.1f}GB (target: {target_data_gb:.1f}GB)")
    else:
        logger.info(f"No global scaling needed: {total_planned_size_gb:.1f}GB ≤ {target_data_gb:.1f}GB target")
    
    # Convert back to the expected format (dataset -> {"main": files})
    grouped_files = {}
    for dataset_name, subsets in sampled_files.items():
        # Flatten all subsets into main config for backward compatibility
        all_subset_files = []
        for subset_files in subsets.values():
            all_subset_files.extend(subset_files)
        
        grouped_files[dataset_name] = {"main": all_subset_files}

    # Log final dataset summary
    for dataset_name, configs in grouped_files.items():
        for config, files in configs.items():
            rows, size_gb = get_parquet_metadata(files)
            if rows:
                logger.info(f"Final: {len(files)} files for {dataset_name}: {rows:,} rows, {size_gb:.2f} GB")
            else:
                logger.info(f"Final: {len(files)} files for {dataset_name}: {size_gb:.2f} GB")

    return grouped_files


def get_local_file_size_gb(file_paths):
    """Calculate total size of local files in GB"""
    from pathlib import Path
    return sum(Path(f).stat().st_size for f in file_paths) / (1024**3)


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    logger.info(f"memory usage: {mem:.2f} MB | {mem/1024:.2f} GB")


def log_dataset_progress(dataset_id, current_dataset, total_datasets, rows=None, size_gb=None):
    """Log progress for dataset loading with local metadata"""
    progress_pct = (current_dataset / total_datasets) * 100

    if rows and size_gb:
        logger.info(f"Loaded {dataset_id} ({current_dataset}/{total_datasets}): {rows:,} rows, {size_gb:.2f} GB [{progress_pct:.1f}%]")
    elif size_gb:
        logger.info(f"Loaded {dataset_id} ({current_dataset}/{total_datasets}): {size_gb:.2f} GB [{progress_pct:.1f}%]")
    else:
        logger.info(f"Loaded {dataset_id} ({current_dataset}/{total_datasets}) [{progress_pct:.1f}%]")


def update_dataset_timing(
    dataset_id,
    dataset_start,
    start_time,
    processed_size,
    total_size,
    dataset_times,
    lang=None,
):
    # Sonny ---> Calculate and store dataset loading time
    dataset_time = time.time() - dataset_start
    dataset_times[dataset_id] = dataset_time
    elapsed_time = time.time() - start_time
    progress = (processed_size / total_size) * 100
    if progress > 0:
        estimated_total = elapsed_time / (progress / 100)
        remaining = estimated_total - elapsed_time
        logger.info(
            f"Dataset loaded in {dataset_time:.1f}s | Est. remaining: {remaining/60:.1f}min"
        )
    return dataset_time


def load_all_datasets(
    max_workers=32,
    streaming=True,
    slurm_logging=False,
    target_data_gb=1500,
    disable_sampling=False,
):
    dataset_count = 0
    total_size_gb = 0
    total_rows = 0
    start_time = time.time()
    dataset_times = {}
    last_hourly_log = start_time

    # Discover local parquet files with intelligent sampling
    logger.info("Discovering local parquet files...")
    local_files = discover_local_parquet_files(
        target_data_gb=target_data_gb,
        enable_sampling=not disable_sampling
    )

    if not local_files:
        logger.error("No local parquet files found!")
        logger.error("Please run: python dataset_downloader.py")
        logger.error("Then retry training with the downloaded datasets")
        sys.exit(1)

    logger.info(f"Found local parquet files for {len(local_files)} datasets")

    # Calculate total datasets to process for progress tracking
    total_dataset_configs = 0
    for configs in local_files.values():
        total_dataset_configs += len(configs)

    current_dataset = 0

    # Process each dataset
    for dataset_name, configs in local_files.items():
        for config, local_parquet_files in configs.items():
            current_dataset += 1

            if config == "main":
                dataset_id = dataset_name
            else:
                dataset_id = f"{dataset_name}.{config}"

            dataset_start = time.time()
            logger.info(f"Processing {dataset_id} ({current_dataset}/{total_dataset_configs})")

            # Get metadata for progress tracking
            rows, size_gb = get_parquet_metadata(local_parquet_files)
            total_size_gb += size_gb
            if rows:
                total_rows += rows

            d = DSLoader()
            try:
                logger.info(f"Using {len(local_parquet_files)} local parquet files for {dataset_id}")

                # Load from local parquet files...
                d.dataset = load_dataset(
                    "parquet",
                    data_files=local_parquet_files,
                    split="train",
                    streaming=streaming,
                )

                # Auto-detect field name from first file.... instead of hardcoding it or overcomplicating it with multiple files!
                d.affected_field = auto_detect_field(local_parquet_files[0])
                d.dataset_name = dataset_id

                dataset_count += 1
                log_dataset_progress(dataset_id, current_dataset, total_dataset_configs, rows, size_gb)

                # Calculate dataset timing first
                dataset_time = time.time() - dataset_start
                dataset_times[dataset_id] = dataset_time

                # Add time estimation with correct parameters
                update_dataset_timing(
                    dataset_id,
                    dataset_start,
                    start_time,
                    current_dataset,  # processed datasets
                    total_dataset_configs,  # total datasets
                    dataset_times
                )

                # Hourly dataset completion logging for Slurm
                if slurm_logging:
                    current_time = time.time()
                    if current_time - last_hourly_log >= 3600:  # 1 hour
                        runtime_hours = (current_time - start_time) / 3600
                        tqdm.write(f"Datasets completed: {current_dataset}/{total_dataset_configs} - Current: processing {dataset_id} - runtime: {runtime_hours:.1f} hours")
                        last_hourly_log = current_time

                yield d

            except Exception as e:
                logger.error(f"Failed to load local dataset {dataset_id}: {e}")
                continue

    if dataset_count == 0:
        logger.error("No datasets were successfully loaded!")
        logger.error("Check your parquet files in ./datasets directory")
        sys.exit(1)

    total_time = time.time() - start_time
    avg_load_time = sum(dataset_times.values()) / len(dataset_times) if dataset_times else 0

    logger.info(f"Successfully loaded {dataset_count} datasets for training")
    if total_rows:
        logger.info(f"Total data: {total_rows:,} rows, {total_size_gb:.2f} GB")
    else:
        logger.info(f"Total data: {total_size_gb:.2f} GB")

    # Fix time display - show seconds if under 1 minute, otherwise minutes
    if total_time < 60:
        logger.info(f"Total time: {total_time:.1f} seconds")
    else:
        logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average dataset load time: {avg_load_time:.1f} seconds")

    # Save dataset loading statistics
    stats_path = os.path.join("stats", "dataset_load_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_datasets_loaded": dataset_count,
                "total_time_sec": total_time,
                "average_dataset_load_time_sec": avg_load_time,
                "dataset_times_sec": dataset_times,
                "total_size_gb": total_size_gb,
                "total_rows": total_rows,
            },
            f,
            indent=2,
        )
    logger.info(f"Dataset stats written to {stats_path}")




def initialize_embedding_matrix(tokenizer, embedding_dim=1024):
    vocab_size = tokenizer.get_vocab_size()
    logger.info(
        f"Initializing embedding matrix for vocabulary_size={vocab_size}, embedding_dim={embedding_dim}"
    )
    try:
        weights = torch.empty((vocab_size, embedding_dim))
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        logger.info(f"Embedding matrix shape: {weights.shape}")
        return weights
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}", exc_info=True)
        raise


def batch_iterator(my_datasets, batch_size=300, slurm_logging=False):
    """Iterate over datasets yielding batches of text samples (not character chunks)
    Smaller batch size provides more diverse samples for better subword learning"""
    i_ds = 1
    record_count = 0
    batch = []
    batch_count = 0
    start_time = time.time()
    last_progress_log = start_time

    try:
        for d in tqdm(my_datasets, desc="Processing Datasets"):
            for record in tqdm(
                d.dataset, desc=f"Processing dataset {d.dataset_name} ({i_ds})"
            ):
                record_count += 1
                # More frequent memory monitoring for early warning
                if record_count % 25000 == 0:
                    log_memory_usage()
                    # Trigger garbage collection to free memory
                    import gc
                    gc.collect()

                try:
                    k = record.get(d.affected_field, "")
                except AttributeError:
                    continue  # skip malformed record

                # Extract text from various field types
                text = ""
                if isinstance(k, list):
                    if len(k) == 0:
                        continue
                    if isinstance(k[0], list):  # e.g., list of lists
                        for sublist in k:
                            text = " ".join(sublist) if isinstance(sublist[0], str) else ""
                    elif isinstance(k[0], str):  # list of strings
                        text = " ".join(k)
                elif isinstance(k, str):  # single string
                    text = k

                # Only add non-empty text with sufficient length for good subword learning
                if text and text.strip():
                    # Filter out very short texts - they don't help BPE learn good subwords
                    stripped_text = text.strip()
                    if len(stripped_text) >= 50:  # Minimum 50 characters for meaningful subword patterns
                        # Additional quality filters to improve subword learning
                        # Skip texts that are mostly numbers, special chars, or very repetitive
                        alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in stripped_text) / len(stripped_text)
                        if alphanumeric_ratio >= 0.7:  # At least 70% alphanumeric + spaces
                            batch.append(stripped_text)

                            # Yield batch when it reaches desired size
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                                batch_count += 1
                                
                                # Slurm-compatible progress logging every 30 minutes
                                if slurm_logging:
                                    current_time = time.time()
                                    if current_time - last_progress_log >= 1800:  # 30 minutes
                                        runtime_hours = (current_time - start_time) / 3600
                                        tqdm.write(f"Training progress: processed {batch_count} batches, {record_count:,} records - runtime: {runtime_hours:.1f} hours")
                                        last_progress_log = current_time

            i_ds += 1

        # Yield remaining batch if any
        if batch:
            yield batch

    except Exception as e:
        logger.error(f"Error in batch_iterator: {e}")
        raise





def train_tokenizer(
    vocab_size,
    output_dir,
    max_workers,
    streaming=True,
    slurm_logging=False,
    target_data_gb=1500,
    disable_sampling=False,
):
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            slurm_logging=slurm_logging,
            target_data_gb=target_data_gb,
            disable_sampling=disable_sampling,
        )

        logger.info("Starting tokenizer training...")
        log_memory_usage()  # Only log once at start of training instead of spamming the blody terminal!
        logger.info(
            "Step 2: Train ByteLevelBPE tokenizer using datasets library multithreading"
        )
        tokenizer = ByteLevelBPETokenizer()

        norm_sequence = [normalizers.NFC()]
        # norm_sequence.append(normalizers.Lowercase())
        norm_sequence.append(normalizers.Replace("\t", " "))
        norm_sequence.append(normalizers.Replace(r"\s+", " "))
        norm_sequence.append(normalizers.Replace("\u00a0", " "))
        # norm_sequence.append(normalizers.Replace(r"[\x00-\x09\x0B-\x1F\x7F]", ""))
        norm_sequence.append(normalizers.Strip())

        tokenizer.normalizer = normalizers.Sequence(norm_sequence)
        # The datasets library handles multithreading internally when we iterate through the datasets
        tokenizer.train_from_iterator(
            batch_iterator(my_datasets, slurm_logging=slurm_logging),
            vocab_size=vocab_size,
            min_frequency=3,  # Lowered to improve fertility (reduce over-segmentation)
            special_tokens=list(SPECIAL_TOKENS.values()),
            show_progress=True,
        )

        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_model(output_dir)
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to train or save tokenizer: {e}", exc_info=True)
        sys.exit(1)


def validate_tokenizer(tokenizer_dir):
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        specials = dict(SPECIAL_TOKENS)
        patched = False
        tokenizer_vocab = set(tokenizer.get_vocab().keys())
        for name, token in specials.items():
            attr = getattr(tokenizer, name, None)
            if attr is None or token not in tokenizer_vocab:
                setattr(tokenizer, name, token)
                patched = True
        if patched:
            tokenizer.add_special_tokens(specials)
            tokenizer.save_pretrained(tokenizer_dir)
            logger.warning("Special tokens patched and tokenizer saved.")
        else:
            logger.info("All required special tokens present in attributes and vocab.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to validate or save tokenizer: {e}", exc_info=True)
        sys.exit(1)


def save_tokenizer_config(tokenizer_dir, vocab_size, embedding_dim):
    """Save tokenizer configuration to config.json"""
    config = {
        "model_type": "byte_level_bpe",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "special_tokens": SPECIAL_TOKENS,
        "max_position_embeddings": 2048,  #  default value??? different models have different values read a lot about this and i dont really know for sure what it should be ???
        "pad_token": SPECIAL_TOKENS["pad_token"],
        "bos_token": SPECIAL_TOKENS["bos_token"],
        "eos_token": SPECIAL_TOKENS["eos_token"],
        "unk_token": SPECIAL_TOKENS["unk_token"],
        "mask_token": SPECIAL_TOKENS["mask_token"],
    }

    config_path = os.path.join(tokenizer_dir, "config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved tokenizer config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer config: {e}", exc_info=True)
        raise


@click.command()
@click.option(
    "--tokenizer-out-dir",
    default="custom_tokenizer",
    show_default=True,
    help="Directory to save the tokenizer",
)
@click.option(
    "--vocab-size",
    default=128000,
    show_default=True,
    help="Vocabulary size for tokenizer",
)
@click.option(
    "--embedding-dim",
    default=1024,
    show_default=True,
    help="Embedding dimension for initialization",
)

@click.option(
    "--max_workers",
    default=4,
    show_default=True,
    help="Maximum parallel dataset loaders (used for datasets library multiprocessing)",
)
@click.option(
    "--streaming/--no-streaming",
    default=True,
    show_default=True,
    help="Enable/disable streaming mode for datasets (streaming=True uses less memory)",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Run in offline mode using cached datasets only",
)
@click.option(
    "--local-data-dir",
    default=None,
    help="Directory containing local text files to use as fallback/additional data source",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--slurm-logging",
    is_flag=True,
    default=False,
    help="Enable periodic time-based logging for Slurm job monitoring",
)
@click.option(
    "--target-data-gb",
    default=1500,
    show_default=True,
    help="Target total data size in GB (for memory management)",
)
@click.option(
    "--disable-sampling",
    is_flag=True,
    default=False,
    help="Disable intelligent sampling (use all data - may cause OOM)",
)
@click.option(
    "--nordic-ratio",
    default=0.25,
    type=float,
    show_default=True,
    help="Nordic languages ratio (0.0-1.0)",
)
@click.option(
    "--european-ratio",
    default=0.58,
    type=float,
    show_default=True,
    help="European languages ratio (0.0-1.0)",
)
@click.option(
    "--english-ratio",
    default=0.02,
    type=float,
    show_default=True,
    help="English language ratio (0.0-1.0)",
)
@click.option(
    "--code-ratio",
    default=0.15,
    type=float,
    show_default=True,
    help="Code/technical content ratio (0.0-1.0)",
)
@click.option(
    "--other-ratio",
    default=0.0,
    type=float,
    show_default=True,
    help="Other languages ratio (0.0-1.0)",
)
def main(
    tokenizer_out_dir,
    vocab_size,
    embedding_dim,
    max_workers,
    streaming,
    offline,
    local_data_dir,
    log_level,
    slurm_logging,
    target_data_gb,
    disable_sampling,
    nordic_ratio,
    european_ratio,
    english_ratio,
    code_ratio,
    other_ratio,
):

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Handle offline mode setup
    if offline:
        logger.info("Running in offline mode - using cached datasets only")
        # Disable HF Hub connectivity
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Build dynamic LANGUAGE_CATEGORIES from CLI arguments
    global LANGUAGE_CATEGORIES
    try:
        LANGUAGE_CATEGORIES = build_language_categories(
            nordic_ratio=nordic_ratio,
            european_ratio=european_ratio, 
            english_ratio=english_ratio,
            code_ratio=code_ratio,
            other_ratio=other_ratio
        )
        logger.info(f"Language distribution: Nordic={nordic_ratio:.1%}, European={european_ratio:.1%}, "
                   f"English={english_ratio:.1%}, Code={code_ratio:.1%}, Other={other_ratio:.1%}")
    except ValueError as e:
        logger.error(f"Invalid language ratios: {e}")
        sys.exit(1)

    logger.info(f"Using max_workers={max_workers} for datasets multiprocessing")
    logger.info(f"Streaming mode: {'enabled' if streaming else 'disabled'}")
    logger.info(f"tokenizer_out_dir {tokenizer_out_dir}")
    logger.info(f"vocab_size {vocab_size}")
    logger.info(f"embedding_dim {embedding_dim}")
    logger.info(f"max_workers {max_workers}")
    logger.info(f"streaming {streaming}")
    logger.info(f"offline {offline}")
    logger.info(f"local_data_dir {local_data_dir}")
    logger.info(f"log_level {log_level}")
    logger.info(f"target_data_gb {target_data_gb}")
    logger.info(f"disable_sampling {disable_sampling}")
    
    if disable_sampling:
        logger.warning("⚠️  SAMPLING DISABLED - May exceed memory limits with large datasets!")
    
    if not streaming:
        logger.info(
            "Non-streaming mode will use more memory but enables better multiprocessing"
        )

    try:
        logger.info("Step 1: Train tokenizer")
        tokenizer = train_tokenizer(
            vocab_size,
            tokenizer_out_dir,
            max_workers,
            streaming=streaming,
            slurm_logging=slurm_logging,
            target_data_gb=target_data_gb,
            disable_sampling=disable_sampling,
        )

        logger.info("Step 2: Validate tokenizer")
        validate_tokenizer(tokenizer_out_dir)

        logger.info("Step 3: Initialize embedding matrix")
        weights = initialize_embedding_matrix(tokenizer, embedding_dim)

        # Save embedding matrix with safetensors as primary format.. We are not using npy format. we use  safetensors format and fallback to pt format.
        try:
            safetensors_path = os.path.join(tokenizer_out_dir, "embedding_matrix.safetensors")
            save_file({"embedding_matrix": weights}, safetensors_path)
            logger.info(f"Saved embedding matrix to {safetensors_path}")
        except Exception as e:
            logger.warning(f"Failed to save safetensors format: {e}")
            # Fallback to PyTorch format
            try:
                pt_path = os.path.join(tokenizer_out_dir, "embedding_matrix.pt")
                torch.save(weights, pt_path)
                logger.info(f"Saved embedding matrix to {pt_path} (PyTorch fallback)")
            except Exception as pt_e:
                logger.error(f"Failed to save PyTorch format: {pt_e}")
                raise

        # Optional: Also save numpy version if possible.. we can get numpy version issues wich requires a downgrade
        """
        A module that was compiled using NumPy 1.x cannot be run in
        NumPy 2.3.2 as it may crash. To support both 1.x and 2.x
        versions of NumPy, modules must be compiled with NumPy 2.0.
        Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
        If you are a user of the module, the easiest solution will be to
        downgrade to 'numpy<2' or try to upgrade the affected module.
        We expect that some modules will need time to support NumPy 2.
        """
        try:
            np.save(
                os.path.join(tokenizer_out_dir, "embedding_matrix.npy"),
                weights.cpu().numpy(),
            )
            logger.info("Also saved numpy version (embedding_matrix.npy)")
        except Exception as np_e:
            logger.warning(f"Could not save numpy version: {np_e}")

        logger.info("Step 4: Save tokenizer configuration")
        save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)

        logger.info("All steps completed successfully.")
        logger.info(f"Tokenizer saved to: {tokenizer_out_dir}")
        logger.info("Files created:")
        logger.info("  - vocab.json")
        logger.info("  - merges.txt")
        logger.info("  - config.json")
        logger.info("  - embedding_matrix.safetensors (or .pt as fallback)")
        logger.info("  - embedding_matrix.npy (optional, if numpy compatible)")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception:
        logger.error("Critical failure. Aborting.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
