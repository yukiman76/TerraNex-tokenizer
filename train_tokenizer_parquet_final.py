import gc  # Added for memory management
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


def parse_data_size(size_str: str) -> float:
    """
    Parse data size string with unit into GB value.

    Args:
        size_str: String like "500MB", "1.5GB", "2TB", "2.5TB"

    Returns:
        float: Size in GB

    Raises:
        ValueError: If format is invalid

    Examples:
        parse_data_size("500MB") -> 0.5
        parse_data_size("1.5GB") -> 1.5
        parse_data_size("2TB") -> 2000.0
        parse_data_size("2.5TB") -> 2500.0
    """
    import re

    if not isinstance(size_str, str):
        raise ValueError(f"Size must be a string, got: {type(size_str)}")

    # Remove spaces and convert to uppercase for consistent parsing
    size_str = size_str.strip().upper()

    # Pattern to match number followed by unit (MB, GB, TB)
    pattern = r'^(\d+(?:\.\d+)?)(MB|GB|TB)$'
    match = re.match(pattern, size_str)

    if not match:
        raise ValueError(
            f"Invalid size format: '{size_str}'. "
            f"Expected format: number + unit (e.g., '500MB', '1.5GB', '2TB'). "
            f"Supported units: MB, GB, TB"
        )

    value_str, unit = match.groups()

    try:
        value = float(value_str)
    except ValueError as err:
        raise ValueError(f"Invalid number in size: '{value_str}'") from err

    if value <= 0:
        raise ValueError(f"Size must be positive, got: {value}")

    # Convert to GB
    if unit == 'MB':
        return value / 1000.0  # MB to GB
    elif unit == 'GB':
        return value
    elif unit == 'TB':
        return value * 1000.0  # TB to GB
    else:
        # This shouldn't happen due to regex, but just in case
        raise ValueError(f"Unsupported unit: {unit}")


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

# ================================================================================================
# DYNAMIC DATASET DISCOVERY AND LANGUAGE DETECTION SYSTEM
# ================================================================================================

class UniversalFilenameParser:
    """Universal parser for any parquet filename format to extract dataset, subset, and language"""

    def __init__(self):
        # Comprehensive language code mappings
        self.language_patterns_2 = {
            'nordic': ['sv', 'da', 'no', 'fi', 'is'],
            'european': [
                'de', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ro', 'hu', 'cs', 'sk', 'bg', 'hr', 'sl', 'et', 'lv', 'lt',
                'ca',  # Catalan
                'gl',  # Galician
                'eu',  # Basque
                'ga',  # Irish
                'mt',  # Maltese
                'el',  # Greek
                'sr',  # Serbian
                'mk',  # Macedonian
                'sq',  # Albanian
                'bs',  # Bosnian
                'be',  # Belarusian
                'uk',  # Ukrainian
                'ru',  # Russian (sometimes considered European)
                'tr',  # Turkish (sometimes considered European)
            ],
            'english': ['en'],
            'other': ['ar', 'zh', 'ja', 'ko', 'hi', 'fa', 'th', 'vi', 'he', 'ur', 'bn', 'ta', 'te', 'ml', 'kn']
        }

        # ISO 639-3 (3-letter codes)
        self.language_patterns_3 = {
            'nordic': ['swe', 'dan', 'nor', 'fin', 'isl', 'nno', 'nob', 'sme', 'fao'],
            'european': [
                'deu', 'fra', 'spa', 'ita', 'nld', 'pol', 'por', 'ron', 'hun', 'ces', 'slk', 'bul', 'hrv', 'slv', 'est', 'lav', 'lit',
                'cat',  # Catalan
                'glg',  # Galician
                'eus',  # Basque
                'gle',  # Irish
                'mlt',  # Maltese
                'ell',  # Greek
                'srp',  # Serbian
                'mkd',  # Macedonian
                'sqi',  # Albanian
                'bos',  # Bosnian
                'bel',  # Belarusian
                'ukr',  # Ukrainian
                'rus',  # Russian
                'tur',  # Turkish
            ],
            'english': ['eng'],
            'other': ['ara', 'zho', 'jpn', 'kor', 'hin', 'fas', 'tha', 'vie', 'heb', 'urd', 'ben', 'tam', 'tel', 'mal', 'kan']
        }

    def parse_filename(self, filename, file_path=None):
        """Parse filename to extract dataset name, subset, language category, and metadata"""
        # Remove extensions
        name = str(filename).replace('.parquet', '').replace('.partial', '')
        parts = name.split('_')

        if len(parts) < 2:
            return None, None, 'unknown', {}

        # Detect language category from filename parts
        language_category = 'unknown'
        subset = None
        metadata = {}

        # Check for language codes in parts
        for part in parts:
            # Check 2-letter codes
            for category, codes in self.language_patterns_2.items():
                if part.lower() in codes:
                    language_category = category
                    subset = part
                    break

            # Check 3-letter codes
            for category, codes in self.language_patterns_3.items():
                if part.lower() in codes:
                    language_category = category
                    subset = part
                    break

            # Check for file numbers
            if part.isdigit() or (len(part) == 4 and part.isdigit()):
                metadata['file_number'] = part

        # Extract dataset name (first part or first two parts for org/repo format)
        if len(parts) >= 2:
            # Check if first part looks like an organization
            known_orgs = ['HuggingFaceFW', 'HuggingFaceH4', 'togethercomputer', 'oscar-corpus', 'common-pile']
            if parts[0] in known_orgs:
                dataset_name = f"{parts[0]}/{parts[1]}"
            else:
                dataset_name = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else parts[0]
        else:
            dataset_name = parts[0]

        # Special handling for specific datasets
        name_lower = name.lower()
        if any(x in name_lower for x in ['github', 'code', 'arxiv', 'redpajama']):
            language_category = 'code'
        elif any(x in name_lower for x in ['swedish', 'gutenberg_multilang', 'biaswb']):
            language_category = 'nordic' if 'swedish' in name_lower else 'european'
        elif 'ultrachat' in name_lower or 'gutenberg-en' in name_lower:
            language_category = 'english'

        return dataset_name, subset, language_category, metadata

    def _detect_via_content_sampling(self, context: dict) -> str:
        """Memory-efficient language detection by sampling parquet file content"""
        try:
            import os

            import pyarrow as pa
            import pyarrow.parquet as pq
            from langdetect import LangDetectException, detect

            file_path = context.get('file_path')
            if not file_path:
                context['methods_tried'].append('content_sampling_failed: no file path provided')
                return 'unknown'

            # Step 1: File size pre-check (skip very large files)
            max_file_size_mb = 500  # Skip files larger than 500MB
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > max_file_size_mb:
                    context['methods_tried'].append(f'content_sampling_skipped: file too large ({file_size_mb:.1f}MB > {max_file_size_mb}MB)')
                    return 'unknown'
            except OSError:
                context['methods_tried'].append('content_sampling_failed: cannot access file')
                return 'unknown'

            # Step 2: Schema inspection to find text columns
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema

            # Identify likely text columns from schema
            text_columns = []
            priority_names = ['text', 'content', 'message', 'body', 'description', 'title', 'summary', 'sentence']

            for _i, field in enumerate(schema):
                field_name = field.name.lower()
                # Include string columns that are likely text
                logical_type = field.logical_type
                physical_type = field.physical_type

                # Check for string types using correct PyArrow API
                is_string_type = (
                    str(logical_type) == 'String' or
                    str(logical_type) == 'LargeString' or
                    (str(physical_type) == 'BYTE_ARRAY' and any(name in field_name for name in priority_names))
                )

                if is_string_type:
                    text_columns.append(field.name)

            # Prioritize columns with text-like names
            text_columns.sort(key=lambda x: (
                0 if any(name in x.lower() for name in priority_names) else 1,
                len(x)  # Shorter names first
            ))

            # Limit to maximum 5 text columns to control memory
            text_columns = text_columns[:5]

            if not text_columns:
                context['methods_tried'].append('content_sampling_failed: no text columns found in schema')
                return 'unknown'

            # Step 3: Memory-efficient limited reading
            max_rows = 100  # Strict row limit
            try:
                # Use read_table with strict limits
                table = pq.read_table(
                    file_path,
                    columns=text_columns,  # Only read text columns
                    # Note: PyArrow doesn't have nrows parameter, so we'll slice after reading
                )

                # Slice to limit rows (more memory efficient than reading all then slicing)
                if table.num_rows > max_rows:
                    table = table.slice(0, max_rows)

                # Convert to pandas with only the data we need
                df = table.to_pandas()

            except pa.ArrowMemoryError:
                context['methods_tried'].append('content_sampling_failed: Arrow memory error during read')
                return 'unknown'
            except MemoryError:
                context['methods_tried'].append('content_sampling_failed: system memory error during read')
                return 'unknown'
            except Exception as e:
                if 'memory' in str(e).lower() or 'allocation' in str(e).lower():
                    context['methods_tried'].append(f'content_sampling_failed: memory error - {str(e)[:100]}')
                    return 'unknown'
                raise  # Re-raise non-memory errors

            # Step 4: Memory-efficient text extraction
            text_content = ""
            max_chars_per_column = 500  # Limit characters from each column

            for col in df.columns:
                try:
                    # Get non-null values and convert to string efficiently
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Take only first 10 entries and limit character length
                        sample_texts = col_data.head(10).astype(str)
                        col_text = " ".join(sample_texts)[:max_chars_per_column]
                        text_content += col_text + " "

                        # Early exit if we have enough text
                        if len(text_content) >= 1000:
                            break
                except Exception:
                    continue  # Skip problematic columns

            # Capture row count before cleanup
            num_rows = len(df) if 'df' in locals() and df is not None else 0

            # Clean up large objects
            del df, table

            if len(text_content.strip()) < 50:
                context['methods_tried'].append('content_sampling_failed: insufficient text content extracted')
                return 'unknown'

            # Step 5: Language detection with memory efficiency
            # Limit text for detection to avoid memory issues in langdetect
            detection_text = text_content[:1000].strip()
            detected_lang = detect(detection_text)

            # Map detected language to categories
            lang_mapping = {
                'sv': 'nordic', 'da': 'nordic', 'no': 'nordic', 'fi': 'nordic', 'is': 'nordic',
                'en': 'english',
                'de': 'european', 'fr': 'european', 'es': 'european', 'it': 'european',
                'nl': 'european', 'pl': 'european', 'pt': 'european', 'ro': 'european',
                'hu': 'european', 'cs': 'european', 'sk': 'european', 'bg': 'european',
                'hr': 'european', 'sl': 'european', 'et': 'european', 'lv': 'european', 'lt': 'european'
            }

            category = lang_mapping.get(detected_lang, 'other')
            context['methods_tried'].append(f'content_sampling_success: {len(text_columns)} cols, {num_rows} rows, detected {detected_lang} -> {category}')
            return category

        except ImportError:
            context['methods_tried'].append('content_sampling_failed: missing dependencies (pyarrow, langdetect)')
            return 'unknown'
        except LangDetectException:
            context['methods_tried'].append('content_sampling_failed: language detection failed on extracted text')
            return 'unknown'
        except MemoryError:
            context['methods_tried'].append('content_sampling_failed: system memory exhausted')
            return 'unknown'
        except Exception as e:
            # Catch any remaining errors gracefully
            error_msg = str(e)[:100]  # Limit error message length
            context['methods_tried'].append(f'content_sampling_failed: {error_msg}')
            return 'unknown'

    def _detect_via_iso_codes(self, context: dict) -> str:
        """Extract and detect language from ISO codes in filename"""
        try:
            dataset_name = context.get('dataset_name', '')
            subset = context.get('subset', '')
            filename = context.get('original_filename', '')

            # Combine all sources for code detection
            search_text = f"{dataset_name} {subset} {filename}".lower()

            # Check for 2-letter codes
            for category, codes in self.language_patterns_2.items():
                for code in codes:
                    if code in search_text:
                        context['methods_tried'].append(f'iso_codes_success: found {code} -> {category}')
                        return category

            # Check for 3-letter codes
            for category, codes in self.language_patterns_3.items():
                for code in codes:
                    if code in search_text:
                        context['methods_tried'].append(f'iso_codes_success: found {code} -> {category}')
                        return category

            # Check for script codes (like Latn)
            script_patterns = {
                'latn': 'european',  # Latin script - assume European if not already categorized
                'cyrl': 'european',  # Cyrillic script
            }

            for script, category in script_patterns.items():
                if script in search_text:
                    context['methods_tried'].append(f'iso_codes_success: found script {script} -> {category}')
                    return category

            context['methods_tried'].append('iso_codes_failed: no ISO codes found')
            return 'unknown'

        except Exception as e:
            context['methods_tried'].append(f'iso_codes_failed: {str(e)}')
            return 'unknown'


class DynamicRatioCalculator:
    """Calculate optimal sampling ratios based on discovered data and user preferences"""

    def calculate_sampling_strategy(self, dataset_inventory: dict,
                                  target_total_gb: float,
                                  user_language_ratios: dict) -> dict:
        """
        Calculate exact sampling ratios to achieve user's language distribution within target size.

        Args:
            dataset_inventory: {(dataset, subset): {'language': str, 'total_size_gb': float, ...}}
            target_total_gb: User's --target-data-gb parameter
            user_language_ratios: {'nordic': 0.30, 'european': 0.45, ...}

        Returns:
            sampling_plan: {(dataset, subset): {'sampling_ratio': float, 'target_gb': float, ...}}
        """

        # Step 1: Analyze available data by language
        available_by_language = self._analyze_available_data(dataset_inventory)

        # Step 2: Calculate target sizes per language
        target_by_language = {
            lang: target_total_gb * ratio
            for lang, ratio in user_language_ratios.items() if ratio > 0
        }

        # Step 3: Calculate sampling ratios for each dataset
        sampling_plan = self._calculate_optimal_sampling(
            dataset_inventory,
            available_by_language,
            target_by_language
        )

        # Step 4: Validate and adjust if needed
        final_plan = self._validate_and_adjust(sampling_plan, target_total_gb)

        return final_plan

    def _analyze_available_data(self, dataset_inventory: dict) -> dict:
        """Analyze available data by language category"""
        available = {}

        for (_dataset, _subset), info in dataset_inventory.items():
            language = info['language']
            size_gb = info['total_size_gb']

            if language not in available:
                available[language] = {
                    'total_gb': 0,
                    'datasets': [],
                    'largest_dataset_gb': 0
                }

            available[language]['total_gb'] += size_gb
            available[language]['datasets'].append((_dataset, _subset, size_gb))
            available[language]['largest_dataset_gb'] = max(
                available[language]['largest_dataset_gb'],
                size_gb
            )

        return available

    def _calculate_optimal_sampling(self, dataset_inventory: dict,
                                  available_by_language: dict,
                                  target_by_language: dict) -> dict:
        """Calculate optimal per-dataset sampling ratios to enforce user-defined language ratios"""
        sampling_plan = {}
        warnings = []

        for language, target_gb in target_by_language.items():
            if language not in available_by_language:
                logger.warning(f"No data available for language category: {language}")
                warnings.append(f"No data available for {language} (target: {target_gb:.1f}GB)")
                continue

            available_gb = available_by_language[language]['total_gb']

            # ALWAYS calculate sampling ratio - no exceptions
            lang_sampling_ratio = target_gb / available_gb

            if lang_sampling_ratio <= 1.0:
                # Over-represented language: sample down to exact target
                logger.info(f"{language.upper()}: {available_gb:.1f}GB available -> {target_gb:.1f}GB target (sampling {lang_sampling_ratio:.3f}x)")
            else:
                # Under-represented language: use all available + warning
                lang_sampling_ratio = 1.0
                shortage_gb = target_gb - available_gb
                warning_msg = f"{language.upper()}: Only {available_gb:.1f}GB available but {target_gb:.1f}GB requested (shortage: {shortage_gb:.1f}GB)"
                logger.warning(warning_msg)
                warnings.append(f"{language}: {available_gb:.1f}GB available < {target_gb:.1f}GB requested")

            # Apply this ratio to all datasets in this language
            for (_dataset, _subset), info in dataset_inventory.items():
                if info['language'] == language:
                    sampling_plan[(_dataset, _subset)] = {
                        'language': language,
                        'available_gb': info['total_size_gb'],
                        'sampling_ratio': lang_sampling_ratio,
                        'target_gb': info['total_size_gb'] * lang_sampling_ratio,
                        'file_count': info['file_count'],
                        'target_files': max(1, int(info['file_count'] * lang_sampling_ratio)),
                        'files': info['files']
                    }

        # Store warnings in the sampling plan for later reporting
        sampling_plan['_warnings'] = warnings
        return sampling_plan

    def _validate_and_adjust(self, sampling_plan: dict, target_total_gb: float) -> dict:
        """Validate total doesn't exceed target, adjust if needed"""
        # Extract warnings before validation
        warnings = sampling_plan.pop('_warnings', [])

        # Calculate actual total from sampling plan (excluding warnings)
        actual_total_gb = sum(plan['target_gb'] for key, plan in sampling_plan.items() if isinstance(plan, dict) and 'target_gb' in plan)

        logger.info(f"Planned total: {actual_total_gb:.1f}GB, Target: {target_total_gb:.1f}GB")

        if actual_total_gb > target_total_gb:
            # Apply proportional scaling to fit exactly within target
            scale_factor = target_total_gb / actual_total_gb
            logger.info(f"Applying final scale factor: {scale_factor:.3f} to fit target")

            for _key, plan in sampling_plan.items():
                if isinstance(plan, dict) and 'target_gb' in plan:
                    plan['sampling_ratio'] *= scale_factor
                    plan['target_gb'] *= scale_factor
                    plan['target_files'] = max(1, int(plan['target_files'] * scale_factor))
                    plan['scaled'] = True
        else:
            logger.info("No additional scaling needed")

        # Re-add warnings to the final plan
        if warnings:
            sampling_plan['_warnings'] = warnings
            logger.info(f"Language ratio warnings: {len(warnings)} language(s) have insufficient data")

        return sampling_plan


class DatasetCompositionAnalyzer:
    """Analyze dataset composition and provide intelligent feedback for ratio validation"""

    def __init__(self):
        self.filename_parser = UniversalFilenameParser()

    def _detect_language_for_analysis(self, dataset_name: str, subset: str, original_filename: str = None, file_path: str = None) -> str:
        """Language detection for analysis mode - content sampling and ISO code extraction only (no API calls)"""

        # Initialize detection context
        detection_context = {
            'dataset_name': dataset_name,
            'subset': subset,
            'original_filename': original_filename,
            'file_path': file_path,
            'methods_tried': [],
            'detection_results': {}
        }

        # 1. Content Sampling + Language Detection (accurate)
        if file_path:
            try:
                result = self.filename_parser._detect_via_content_sampling(detection_context)
                if result != 'unknown':
                    return result
            except Exception as e:
                detection_context['methods_tried'].append(f'content_sampling_failed: {str(e)}')

        # 2. Dynamic ISO Code Extraction (fast fallback)
        try:
            result = self.filename_parser._detect_via_iso_codes(detection_context)
            if result != 'unknown':
                return result
        except Exception as e:
            detection_context['methods_tried'].append(f'iso_codes_failed: {str(e)}')

        # 3. LAST RESORT: Log what was tried and return unknown
        logger.debug(f"Language detection failed for {original_filename or dataset_name}. Tried: {detection_context['methods_tried']}")
        return 'unknown'

    def analyze_available_data(self, datasets_dir: str) -> dict:
        """
        Memory-efficient analysis of available data with intelligent sampling.
        Uses representative sampling to avoid OOM with huge datasets.

        Returns:
            dict: {
                'total_size_gb': float,
                'languages': {
                    'language_category': {
                        'size_gb': float,
                        'percentage': float,
                        'datasets': [list of dataset names],
                        'file_count': int
                    }
                },
                'datasets': [list of all datasets found],
                'warnings': [list of warnings/issues],
                'sampling_info': {
                    'total_files': int,
                    'analyzed_files': int,
                    'sampling_used': bool
                }
            }
        """
        from pathlib import Path

        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            return {'error': f"Datasets directory not found: {datasets_dir}"}

        parquet_files = list(datasets_path.glob("*.parquet"))
        if not parquet_files:
            return {'error': f"No parquet files found in: {datasets_dir}"}

        total_files = len(parquet_files)
        logger.info(f"Found {total_files} parquet files for analysis")

        # Memory-efficient sampling strategy
        max_files_to_analyze = 500  # Reasonable limit to prevent OOM
        sampling_used = total_files > max_files_to_analyze

        if sampling_used:
            # Intelligent sampling: take files from different size ranges
            file_sizes = [(f, f.stat().st_size) for f in parquet_files]
            file_sizes.sort(key=lambda x: x[1])  # Sort by size

            # Sample from different quartiles to get representative data
            step = len(file_sizes) // max_files_to_analyze
            sampled_files = [file_sizes[i][0] for i in range(0, len(file_sizes), max(1, step))]
            sampled_files = sampled_files[:max_files_to_analyze]

            logger.info(f"Using intelligent sampling: analyzing {len(sampled_files)}/{total_files} files")
            files_to_process = sampled_files
        else:
            files_to_process = parquet_files

        # Build inventory using memory-efficient processing
        total_size_gb = 0
        languages = {}
        datasets_found = set()
        warnings = []
        processed_count = 0
        memory_checks = 0

        for file_path in files_to_process:
            try:
                # Memory monitoring every 50 files
                if processed_count % 50 == 0 and processed_count > 0:
                    memory_usage = get_memory_usage_percent()
                    if memory_usage > 80:
                        logger.warning(f"High memory usage during analysis: {memory_usage:.1f}% (processed {processed_count}/{len(files_to_process)} files)")
                        gc.collect()  # Force garbage collection
                        memory_checks += 1

                        # Emergency exit if memory remains high
                        if memory_checks > 3 and memory_usage > 85:
                            warnings.append(f"Analysis stopped early due to memory pressure (processed {processed_count}/{len(files_to_process)} files)")
                            break

                # Use basic filename parsing first (fast, no file I/O)
                dataset_name, subset, _, metadata = self.filename_parser.parse_filename(file_path.name)

                # Use the full language detection (which includes filename + content sampling as fallback)
                try:
                    language_category = self._detect_language_for_analysis(
                        dataset_name, subset, file_path.name, str(file_path)
                    )
                except Exception as e:
                    language_category = 'other'
                    warnings.append(f"Language detection failed for {file_path.name}: {str(e)[:100]}")

                # Get file size
                file_size_gb = file_path.stat().st_size / (1024**3)

                # If sampling, scale up the size to estimate total
                if sampling_used:
                    scale_factor = total_files / len(files_to_process)
                    file_size_gb *= scale_factor

                total_size_gb += file_size_gb

                # Track dataset
                datasets_found.add(dataset_name)

                # Group by language category
                if language_category not in languages:
                    languages[language_category] = {
                        'size_gb': 0,
                        'percentage': 0,
                        'datasets': set(),
                        'file_count': 0
                    }

                languages[language_category]['size_gb'] += file_size_gb
                languages[language_category]['datasets'].add(dataset_name)
                languages[language_category]['file_count'] += (scale_factor if sampling_used else 1)

                processed_count += 1

                # Periodic cleanup and progress logging
                if processed_count % 100 == 0:
                    gc.collect()
                    logger.info(f"Analysis progress: {processed_count}/{len(files_to_process)} files processed")

            except Exception as e:
                warnings.append(f"Could not process {file_path.name}: {str(e)[:100]}")
                continue

        # Calculate percentages
        for lang_info in languages.values():
            lang_info['percentage'] = (lang_info['size_gb'] / total_size_gb) * 100 if total_size_gb > 0 else 0
            lang_info['datasets'] = list(lang_info['datasets'])  # Convert set to list
            lang_info['file_count'] = int(lang_info['file_count'])  # Ensure integer

        # Final cleanup
        gc.collect()

        return {
            'total_size_gb': total_size_gb,
            'languages': languages,
            'datasets': list(datasets_found),
            'warnings': warnings,
            'sampling_info': {
                'total_files': total_files,
                'analyzed_files': processed_count,
                'sampling_used': sampling_used
            }
        }

    def validate_user_ratios(self, analysis: dict, user_language_ratios: dict) -> dict:
        """
        Validate user ratios against available data and provide recommendations.

        Returns:
            dict: {
                'valid': bool,
                'issues': [list of validation issues],
                'recommendations': [list of suggested corrections],
                'adjusted_ratios': dict  # suggested adjusted ratios if needed
            }
        """
        if 'error' in analysis:
            return {'valid': False, 'issues': [analysis['error']], 'recommendations': [], 'adjusted_ratios': {}}

        available_languages = set(analysis['languages'].keys())
        requested_languages = {lang for lang, ratio in user_language_ratios.items() if ratio > 0}

        issues = []
        recommendations = []
        adjusted_ratios = user_language_ratios.copy()

        # Check for requested languages not available
        missing_languages = requested_languages - available_languages
        if missing_languages:
            issues.append(f"Requested languages not available in dataset: {', '.join(missing_languages)}")
            recommendations.append("Remove unavailable languages from your ratios or set them to 0.0")
            # Set missing languages to 0
            for lang in missing_languages:
                adjusted_ratios[lang] = 0.0

        # Check for available languages not requested
        unused_languages = available_languages - requested_languages
        if unused_languages:
            unused_data = sum(analysis['languages'][lang]['size_gb'] for lang in unused_languages)
            total_unused_pct = (unused_data / analysis['total_size_gb']) * 100
            recommendations.append(f"Available languages not used: {', '.join(unused_languages)} "
                                 f"({unused_data:.1f}GB, {total_unused_pct:.1f}% of total data)")

        # Normalize ratios for available languages only
        available_ratio_sum = sum(adjusted_ratios[lang] for lang in adjusted_ratios if lang in available_languages)
        if available_ratio_sum > 0:
            normalization_factor = 1.0 / available_ratio_sum
            for lang in adjusted_ratios:
                if lang in available_languages:
                    adjusted_ratios[lang] *= normalization_factor
                else:
                    adjusted_ratios[lang] = 0.0

        # Check if any language doesn't have enough data
        insufficient_data = []
        for lang, requested_ratio in adjusted_ratios.items():
            if requested_ratio > 0 and lang in available_languages:
                available_pct = analysis['languages'][lang]['percentage'] / 100
                if requested_ratio > available_pct * 2:  # More than 2x available percentage
                    insufficient_data.append(f"{lang}: requested {requested_ratio*100:.1f}% but only {available_pct*100:.1f}% available")

        if insufficient_data:
            issues.append("Some languages may not have sufficient data for requested ratios:")
            issues.extend(insufficient_data)
            recommendations.append("Consider reducing ratios for languages with limited data")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'adjusted_ratios': adjusted_ratios
        }

    def print_composition_report(self, analysis: dict, user_language_ratios: dict = None):
        """Print a detailed composition report"""

        if 'error' in analysis:
            logger.error(f"❌ {analysis['error']}")
            return

        logger.info("📊 DATASET COMPOSITION ANALYSIS")
        logger.info("=" * 80)

        # Show sampling info if available
        if 'sampling_info' in analysis:
            sampling_info = analysis['sampling_info']
            if sampling_info['sampling_used']:
                logger.info(f"🎯 Sampling: {sampling_info['analyzed_files']}/{sampling_info['total_files']} files analyzed (intelligent sampling used)")
            else:
                logger.info(f"📋 Analysis: {sampling_info['analyzed_files']} files analyzed (complete)")

        # Overall stats
        logger.info(f"📁 Total Data: {analysis['total_size_gb']:.2f}GB")
        logger.info(f"📦 Datasets Found: {len(analysis['datasets'])}")
        logger.info(f"🌍 Languages Available: {len(analysis['languages'])}")

        if analysis['warnings']:
            logger.warning(f"⚠️  Warnings: {len(analysis['warnings'])}")
            for warning in analysis['warnings']:
                logger.warning(f"   • {warning}")

        logger.info("\n📈 LANGUAGE BREAKDOWN:")
        logger.info("-" * 50)

        # Sort languages by size
        sorted_languages = sorted(analysis['languages'].items(),
                                key=lambda x: x[1]['size_gb'], reverse=True)

        for language, info in sorted_languages:
            logger.info(f"🔤 {language.upper()}:")
            logger.info(f"   Size: {info['size_gb']:.2f}GB ({info['percentage']:.1f}%)")
            logger.info(f"   Files: {info['file_count']}")
            logger.info(f"   Datasets: {', '.join(info['datasets'])}")

            # Show comparison with user ratios if provided
            if user_language_ratios and language in user_language_ratios:
                requested_pct = user_language_ratios[language] * 100
                if requested_pct > 0:
                    logger.info(f"   👤 Requested: {requested_pct:.1f}% ({'✅ Available' if info['percentage'] >= requested_pct else '⚠️  Limited'})")
            logger.info("")

        # Validation if user ratios provided
        if user_language_ratios:
            validation = self.validate_user_ratios(analysis, user_language_ratios)
            logger.info("🔍 RATIO VALIDATION:")
            logger.info("-" * 50)

            if validation['valid']:
                logger.info("✅ User ratios are valid and achievable!")
            else:
                logger.warning("⚠️  Issues found with user ratios:")
                for issue in validation['issues']:
                    logger.warning(f"   • {issue}")

                if validation['recommendations']:
                    logger.info("💡 RECOMMENDATIONS:")
                    for rec in validation['recommendations']:
                        logger.info(f"   • {rec}")

                if validation['adjusted_ratios'] != user_language_ratios:
                    logger.info("🔧 SUGGESTED ADJUSTED RATIOS:")
                    for lang, ratio in validation['adjusted_ratios'].items():
                        if ratio > 0:
                            logger.info(f"   {lang}: {ratio:.3f} ({ratio*100:.1f}%)")


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
    language_codes = {
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
        if potential_lang in language_codes:
            language_subset = potential_lang
            dataset_parts = content_parts[:-2]

    # Check for single language code
    if not language_subset and len(content_parts) >= 2:
        potential_lang = content_parts[-1]
        if potential_lang in language_codes:
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
        logger.error("❌ File accessibility issues found:")
        for issue in missing_files[:10]:  # Show first 10 issues
            logger.error(f"   - {issue}")
        if len(missing_files) > 10:
            logger.error(f"   ... and {len(missing_files)-10} more issues")
        return False

    logger.info(f"✅ File validation: {len(file_paths)} files accessible, {total_size/1024**3:.1f}GB total")
    return True



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


def discover_local_parquet_files(target_data_gb=1500, enable_sampling=True, user_language_ratios=None):
    """
    NEW DYNAMIC DISCOVERY SYSTEM: Discover parquet files and calculate optimal ratios from actual data

    Args:
        target_data_gb: Target total data size in GB
        enable_sampling: Whether to apply sampling
        user_language_ratios: {'nordic': 0.30, 'european': 0.45, 'english': 0.15, 'code': 0.10, 'other': 0.0}

    Returns:
        Dictionary compatible with existing system: {dataset_name: {"main": [file_paths]}}
    """
    import random
    from pathlib import Path

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

    logger.info(f"✅ File validation: {len(parquet_files)} files accessible, {get_local_file_size_gb(all_file_paths):.1f}GB total")

    # NOTE: Interactive ratio analysis is only done in --analyze-ratios mode
    # Training mode goes directly to dataset discovery for HPC compatibility

    # STEP 1: DYNAMIC DATASET DISCOVERY AND INVENTORY
    logger.info("=== DYNAMIC DATASET DISCOVERY ===")
    parser = UniversalFilenameParser()
    dataset_inventory = {}

    for file_path in parquet_files:
        filename = file_path.name

        # Parse filename with new universal parser and file path for intelligent detection
        dataset_name, subset, language_category, metadata = parser.parse_filename(filename, str(file_path))

        # Calculate file size
        file_size_gb = file_path.stat().st_size / (1024**3)

        # Build inventory
        key = (dataset_name, subset)
        if key not in dataset_inventory:
            dataset_inventory[key] = {
                'language': language_category,
                'total_size_gb': 0,
                'file_count': 0,
                'files': []
            }

        dataset_inventory[key]['total_size_gb'] += file_size_gb
        dataset_inventory[key]['file_count'] += 1
        dataset_inventory[key]['files'].append(str(file_path))

    # Log discovery summary
    total_discovered_gb = sum(info['total_size_gb'] for info in dataset_inventory.values())
    unique_datasets = len({dataset for dataset, subset in dataset_inventory.keys()})
    logger.info(f"Discovered: {unique_datasets} datasets, {len(dataset_inventory)} dataset/subset combinations, {total_discovered_gb:.1f}GB total")

    # Log by language category
    by_language = {}
    for (_dataset, _subset), info in dataset_inventory.items():
        lang = info['language']
        if lang not in by_language:
            by_language[lang] = {'count': 0, 'size_gb': 0.0}
        by_language[lang]['count'] += 1
        by_language[lang]['size_gb'] += info['total_size_gb']

    for lang, stats in sorted(by_language.items()):
        logger.info(f"  {lang.upper()}: {stats['count']} combinations, {stats['size_gb']:.1f}GB")

    if not enable_sampling:
        logger.info("Sampling disabled - using all available files")
        # Convert to expected format
        return _convert_inventory_to_legacy_format(dataset_inventory)

    # STEP 2: CALCULATE DYNAMIC RATIOS
    if user_language_ratios is None:
        user_language_ratios = {
            'nordic': 0.25, 'european': 0.58, 'english': 0.02, 'code': 0.15, 'other': 0.0
        }

    logger.info("=== DYNAMIC RATIO CALCULATION ===")
    # Check if we're using the total available data (when target equals discovered total)
    if abs(target_data_gb - total_discovered_gb) < 0.1:  # Within 0.1GB tolerance
        logger.info(f"Target: ALL available data ({total_discovered_gb:.1f}GB) with ratios {user_language_ratios}")
    else:
        logger.info(f"Target: {target_data_gb:.1f}GB with ratios {user_language_ratios}")

    calculator = DynamicRatioCalculator()
    sampling_plan = calculator.calculate_sampling_strategy(
        dataset_inventory,
        target_data_gb,
        user_language_ratios
    )

    # STEP 3: APPLY SAMPLING BASED ON CALCULATED RATIOS
    logger.info("=== APPLYING DYNAMIC SAMPLING ===")

    # Extract and display warnings first
    warnings = sampling_plan.pop('_warnings', [])
    if warnings:
        logger.warning("RATIO ENFORCEMENT WARNINGS:")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    random.seed(42)  # Reproducible sampling

    final_files = {}
    total_sampled_gb = 0.0

    for (dataset, subset), plan in sampling_plan.items():
        # Skip non-plan entries
        if not isinstance(plan, dict) or 'files' not in plan:
            continue

        available_files = plan['files']
        target_files = plan['target_files']

        # Apply sampling
        if target_files < len(available_files):
            sampled_files = random.sample(available_files, target_files)
            logger.info(f"Sampled {dataset}" + (f" ({subset})" if subset else "") + f": {len(available_files)} -> {target_files} files ({plan['sampling_ratio']:.3f} ratio)")
        else:
            sampled_files = available_files

        # Store in final structure
        if dataset not in final_files:
            final_files[dataset] = {}

        subset_key = subset if subset else "main"
        final_files[dataset][subset_key] = sampled_files

        total_sampled_gb += plan['target_gb']

    # Check if we're using all available data
    if abs(target_data_gb - total_discovered_gb) < 0.1:  # Within 0.1GB tolerance
        logger.info(f"Total sampled: {total_sampled_gb:.1f}GB (target: ALL available data)")
    else:
        logger.info(f"Total sampled: {total_sampled_gb:.1f}GB (target: {target_data_gb:.1f}GB)")

    # STEP 4: CONVERT TO LEGACY FORMAT
    grouped_files = {}
    for dataset_name, subsets in final_files.items():
        # Flatten all subsets into main config for backward compatibility
        all_subset_files = []
        for subset_files in subsets.values():
            all_subset_files.extend(subset_files)

        grouped_files[dataset_name] = {"main": all_subset_files}

    # Log final summary
    logger.info("=== FINAL DATASET SUMMARY ===")
    final_total_gb = 0.0
    for dataset_name, configs in grouped_files.items():
        for _config, files in configs.items():
            rows, size_gb = get_parquet_metadata(files)
            final_total_gb += size_gb
            if rows:
                logger.info(f"Final: {len(files)} files for {dataset_name}: {rows:,} rows, {size_gb:.2f} GB")
            else:
                logger.info(f"Final: {len(files)} files for {dataset_name}: {size_gb:.2f} GB")

    logger.info(f"Final total: {final_total_gb:.1f}GB")
    return grouped_files


def _convert_inventory_to_legacy_format(dataset_inventory):
    """Convert dataset inventory to legacy format when sampling is disabled"""
    grouped_files = {}

    for (dataset_name, _subset), info in dataset_inventory.items():
        if dataset_name not in grouped_files:
            grouped_files[dataset_name] = {"main": []}

        # Add all files to main
        grouped_files[dataset_name]["main"].extend(info['files'])

    return grouped_files


def get_local_file_size_gb(file_paths):
    """Calculate total size of local files in GB"""
    from pathlib import Path
    return sum(Path(f).stat().st_size for f in file_paths) / (1024**3)


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    logger.info(f"memory usage: {mem:.2f} MB | {mem/1024:.2f} GB")


def get_memory_usage_percent():
    """Get current system memory usage percentage"""
    return psutil.virtual_memory().percent




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
    target_data_gb=None,
    disable_sampling=False,
    user_language_ratios=None,
):
    dataset_count = 0
    total_size_gb = 0
    total_rows = 0
    start_time = time.time()
    dataset_times = {}
    last_hourly_log = start_time

    # Discover local parquet files with intelligent sampling
    logger.info("Discovering local parquet files...")

    # Handle unlimited data case
    if target_data_gb is None:
        logger.info("🚀 No data size limit - calculating total available data for ratio application")
        # Calculate actual total data size for ratio calculations
        from pathlib import Path
        datasets_path = Path("./datasets")
        if datasets_path.exists():
            parquet_files = list(datasets_path.glob("*.parquet"))
            total_available_gb = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            logger.info(f"📊 Total available data: {total_available_gb:.1f}GB - will apply ratios to this amount")
            effective_target_gb = total_available_gb  # Use actual total for ratio calculations
        else:
            logger.warning("No datasets directory found - using fallback ratio calculation")
            effective_target_gb = 1500  # Fallback if no data found
        effective_sampling = not disable_sampling  # Enable sampling for ratio application
    else:
        effective_sampling = not disable_sampling
        effective_target_gb = target_data_gb

    local_files = discover_local_parquet_files(
        target_data_gb=effective_target_gb,
        enable_sampling=effective_sampling,
        user_language_ratios=user_language_ratios
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

    # Process each dataset with proper streaming
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

                # Load from local parquet files with streaming enabled
                d.dataset = load_dataset(
                    "parquet",
                    data_files=local_parquet_files,
                    split="train",
                    streaming=streaming,
                )

                # Auto-detect field name from first file
                d.affected_field = auto_detect_field(local_parquet_files[0])
                d.dataset_name = dataset_id

                dataset_count += 1
                log_dataset_progress(dataset_id, current_dataset, total_dataset_configs, rows, size_gb)

                # Calculate dataset timing
                dataset_time = time.time() - dataset_start
                dataset_times[dataset_id] = dataset_time

                # Add time estimation
                update_dataset_timing(
                    dataset_id,
                    dataset_start,
                    start_time,
                    current_dataset,
                    total_dataset_configs,
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


def memory_efficient_batch_iterator(my_datasets, initial_batch_size=5000, slurm_logging=False):
    """
    Memory-efficient batch iterator with adaptive sizing based on memory pressure.
    Yields batches of text samples optimized for ByteLevelBPE training.
    """
    current_batch_size = initial_batch_size
    i_ds = 1
    record_count = 0
    batch = []
    batch_count = 0
    start_time = time.time()
    last_progress_log = start_time
    last_memory_check = start_time

    # Memory management thresholds
    memory_warning_threshold = 75.0   # Start reducing batch size
    memory_critical_threshold = 85.0  # Aggressive batch size reduction
    memory_recovery_threshold = 60.0  # Start increasing batch size again

    logger.info(f"Starting memory-efficient batch iterator with initial batch size: {current_batch_size}")
    logger.info(f"Memory thresholds - Warning: {memory_warning_threshold}%, Critical: {memory_critical_threshold}%, Recovery: {memory_recovery_threshold}%")

    try:
        for d in tqdm(my_datasets, desc="Processing Datasets"):
            logger.info(f"Processing dataset: {d.dataset_name} (Dataset {i_ds})")

            for record in tqdm(d.dataset, desc=f"Processing {d.dataset_name}"):
                record_count += 1

                # Memory monitoring and adaptive batch sizing
                current_time = time.time()
                if current_time - last_memory_check >= 30:  # Check every 30 seconds
                    memory_usage = get_memory_usage_percent()

                    # Adaptive batch size scaling
                    old_batch_size = current_batch_size
                    if memory_usage > memory_critical_threshold:
                        # Critical: reduce by 50%
                        current_batch_size = max(100, current_batch_size // 2)
                        if old_batch_size != current_batch_size:
                            logger.warning(f"Critical memory pressure: {memory_usage:.1f}% - reducing batch size: {old_batch_size} → {current_batch_size}")
                    elif memory_usage > memory_warning_threshold:
                        # Warning: reduce by 25%
                        current_batch_size = max(500, int(current_batch_size * 0.75))
                        if old_batch_size != current_batch_size:
                            logger.warning(f"Memory pressure detected: {memory_usage:.1f}% - reducing batch size: {old_batch_size} → {current_batch_size}")
                    elif memory_usage < memory_recovery_threshold and current_batch_size < initial_batch_size:
                        # Recovery: increase by 25%
                        current_batch_size = min(initial_batch_size, int(current_batch_size * 1.25))
                        if old_batch_size != current_batch_size:
                            logger.info(f"Memory recovered: {memory_usage:.1f}% - increasing batch size: {old_batch_size} → {current_batch_size}")

                    last_memory_check = current_time

                # Periodic garbage collection and memory logging
                if record_count % 50000 == 0:
                    memory_usage = get_memory_usage_percent()
                    logger.info(f"Processed {record_count:,} records | Memory: {memory_usage:.1f}% | Batch size: {current_batch_size}")
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

                # Quality filtering for better BPE training
                if text and text.strip():
                    stripped_text = text.strip()
                    if len(stripped_text) >= 50:  # Minimum length for meaningful patterns
                        # Filter out low-quality text
                        alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in stripped_text) / len(stripped_text)
                        if alphanumeric_ratio >= 0.7:  # At least 70% alphanumeric + spaces
                            batch.append(stripped_text)

                            # Yield batch when it reaches current size
                            if len(batch) >= current_batch_size:
                                yield batch
                                batch = []
                                batch_count += 1

                                # Slurm-compatible progress logging
                                if slurm_logging:
                                    current_time = time.time()
                                    if current_time - last_progress_log >= 1800:  # 30 minutes
                                        runtime_hours = (current_time - start_time) / 3600
                                        memory_usage = get_memory_usage_percent()
                                        tqdm.write(f"Training progress: processed {batch_count} batches, {record_count:,} records | Memory: {memory_usage:.1f}% | Batch size: {current_batch_size} - runtime: {runtime_hours:.1f} hours")
                                        last_progress_log = current_time

            # Clean up after each dataset to free memory
            logger.info(f"Completed dataset {d.dataset_name}. Running cleanup...")
            gc.collect()
            i_ds += 1

        # Yield remaining batch if any
        if batch:
            logger.info(f"Yielding final batch with {len(batch)} texts")
            yield batch

        # Final statistics
        total_time = time.time() - start_time
        final_memory = get_memory_usage_percent()
        logger.info("Memory-efficient batch iterator completed:")
        logger.info(f"  - Total batches: {batch_count}")
        logger.info(f"  - Total records processed: {record_count:,}")
        logger.info(f"  - Total time: {total_time/60:.1f} minutes")
        logger.info(f"  - Final memory usage: {final_memory:.1f}%")
        logger.info(f"  - Final batch size: {current_batch_size}")

    except Exception as e:
        logger.error(f"Error in memory_efficient_batch_iterator: {e}")
        raise





def train_tokenizer(
    vocab_size,
    output_dir,
    max_workers,
    streaming=True,
    slurm_logging=False,
    target_data_gb=None,
    disable_sampling=False,
    user_language_ratios=None,
):
    try:
        logger.info("Step 1: Build and deduplicate corpus from provided sources")
        my_datasets = load_all_datasets(
            max_workers=max_workers,
            streaming=streaming,
            slurm_logging=slurm_logging,
            target_data_gb=target_data_gb,
            disable_sampling=disable_sampling,
            user_language_ratios=user_language_ratios,
        )

        logger.info("Starting tokenizer training...")
        log_memory_usage()
        logger.info("Step 2: Train ByteLevelBPE tokenizer using datasets library multithreading")

        tokenizer = ByteLevelBPETokenizer()

        norm_sequence = [normalizers.NFC()]
        norm_sequence.append(normalizers.Replace("\t", " "))
        norm_sequence.append(normalizers.Replace(r"\s+", " "))
        norm_sequence.append(normalizers.Replace("\u00a0", " "))
        norm_sequence.append(normalizers.Strip())

        tokenizer.normalizer = normalizers.Sequence(norm_sequence)

        # Use memory-efficient batch iterator with adaptive sizing
        tokenizer.train_from_iterator(
            memory_efficient_batch_iterator(my_datasets, slurm_logging=slurm_logging),
            vocab_size=vocab_size,
            min_frequency=1,  # Reduced from 3 to 1 for lower fertility
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
    "--target-data-size",
    default=None,
    help="Target total data size (e.g., 500MB, 1.5GB, 2TB, 2.5TB). If not specified, uses ALL available data.",
)
@click.option(
    "--disable-sampling",
    is_flag=True,
    default=False,
    help="Disable intelligent sampling (use all data - may cause OOM)",
)
@click.option(
    "--analyze-ratios",
    is_flag=True,
    default=False,
    help="Analyze dataset composition and language ratios, then exit (no tokenizer training)",
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
    target_data_size,
    disable_sampling,
    analyze_ratios,
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

    # Parse target data size with unit support
    if target_data_size is None:
        target_data_gb = None
        logger.info("🚀 Using ALL available data (no size limit specified)")
    else:
        try:
            target_data_gb = parse_data_size(target_data_size)
            logger.info(f"🎯 Target data size: {target_data_size} ({target_data_gb}GB)")
        except ValueError as e:
            logger.error(f"Invalid target data size: {e}")
            sys.exit(1)

    # Handle analyze-ratios mode
    if analyze_ratios:
        logger.info("🔍 ANALYZE RATIOS MODE - Analyzing dataset composition")
        analyzer = DatasetCompositionAnalyzer()
        analysis = analyzer.analyze_available_data("./datasets")

        # Print comprehensive analysis report
        analyzer.print_composition_report(analysis)

        logger.info("✅ Analysis complete! Use this information to set optimal language ratios.")
        logger.info("💡 Example usage with ratios:")

        # Suggest example ratios based on available data
        if 'languages' in analysis:
            total_gb = analysis['total_size_gb']
            example_target = f"{total_gb:.0f}GB" if target_data_size is None else target_data_size
            logger.info(f"   python {sys.argv[0]} --target-data-size {example_target} \\")
            for lang, info in sorted(analysis['languages'].items(), key=lambda x: x[1]['size_gb'], reverse=True):
                suggested_ratio = min(0.40, info['percentage'] / 100 * 1.5)  # Cap at 40%, scale by 1.5x
                if suggested_ratio >= 0.05:  # Only show if >= 5%
                    logger.info(f"     --{lang.lower()}-ratio {suggested_ratio:.2f} \\")
            logger.info(f"     --vocab-size {max(8000, min(128000, int(total_gb * 1500)))} # ~1.5K vocab per GB (research-based optimal ratio)")

        sys.exit(0)  # Exit successfully after analysis

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
        logger.info("Step 1: Train tokenizer using memory-efficient ByteLevelBPE")
        tokenizer = train_tokenizer(
            vocab_size,
            tokenizer_out_dir,
            max_workers,
            streaming=streaming,
            slurm_logging=slurm_logging,
            target_data_gb=target_data_gb,
            disable_sampling=disable_sampling,
            user_language_ratios={
                'nordic': nordic_ratio,
                'european': european_ratio,
                'english': english_ratio,
                'code': code_ratio,
                'other': other_ratio
            },
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
