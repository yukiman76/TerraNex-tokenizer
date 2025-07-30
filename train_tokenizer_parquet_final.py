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
    except ValueError:
        raise ValueError(f"Invalid number in size: '{value_str}'")

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
            'european': ['de', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ro', 'hu', 'cs', 'sk', 'bg', 'hr', 'sl', 'et', 'lv', 'lt'],
            'english': ['en'],
            'other': ['ar', 'zh', 'ja', 'ko', 'ru', 'hi', 'tr', 'fa', 'th', 'vi', 'he', 'ur', 'bn', 'ta', 'te', 'ml', 'kn']
        }

        # ISO 639-3 (3-letter codes)
        self.language_patterns_3 = {
            'nordic': ['swe', 'dan', 'nor', 'fin', 'isl', 'nno', 'nob', 'sme', 'fao'],
            'european': ['deu', 'fra', 'spa', 'ita', 'nld', 'pol', 'por', 'ron', 'hun', 'ces', 'slk', 'bul', 'hrv', 'slv', 'est', 'lav', 'lit'],
            'english': ['eng'],
            'other': ['ara', 'zho', 'jpn', 'kor', 'rus', 'hin', 'tur', 'fas', 'tha', 'vie', 'heb', 'urd', 'ben', 'tam', 'tel', 'mal', 'kan']
        }

        # Script codes (for language_Script format)
        self.script_codes = ['Latn', 'Cyrl', 'Arab', 'Hans', 'Hant', 'Jpan', 'Kore', 'Deva', 'Thai', 'Hebr']

        # Dataset name patterns for fallback detection
        self.dataset_patterns = {
            'code': ['github', 'code', 'stack', 'codeparrot', 'bigcode', 'rstar'],
            'nordic': ['nordic', 'svensk', 'danish', 'norwegian', 'finnish', 'culturax-nord', 'sweden'],
            'english': ['redpajama', 'gutenberg', 'ultrachat', 'pile', 'wikitext'],
            'multilingual': ['oscar', 'fineweb', 'c4', 'moscar', 'wikipedia'],
            'european': ['gutenberg_multilang']
        }

    def parse_filename(self, filename: str, file_path: str = None) -> tuple:
        """
        Parse any parquet filename format and return (dataset_name, subset, language_category, metadata)
        
        Args:
            filename: The filename to parse  
            file_path: Optional full path to the file for content-based language detection
        
        Examples:
            'HuggingFaceFW_fineweb-2_deu_Latn_0159.parquet'
            → ('HuggingFaceFW/fineweb-2', 'deu_Latn', 'european', {'file_number': '0159'})
            
            'codeparrot_github-code-clean_data_train-00012-of-00880.parquet'
            → ('codeparrot/github-code-clean', None, 'code', {'file_number': '00012', 'total_files': '00880'})
        """

        # Remove .parquet extension
        base_name = filename.replace('.parquet', '')

        # Try multiple parsing strategies in order of specificity
        strategies = [
            self._parse_complex_train_format,  # train-XXXXX-of-XXXXX format
            self._parse_huggingface_org_format,  # org_repo_subset_lang_number format
            self._parse_underscore_format,  # standard underscore separation
            self._parse_dash_format,  # dash separation
            self._parse_generic_format  # fallback for any format
        ]

        for strategy in strategies:
            result = strategy(base_name)
            if result:
                dataset_name, subset, metadata = result
                # Pass original filename AND file path for intelligent language detection
                language_category = self._detect_language_category(dataset_name, subset, original_filename=base_name, file_path=file_path)
                return dataset_name, subset, language_category, metadata

        # Ultimate fallback - use original filename for language detection
        language_category = self._detect_language_category(base_name, None, original_filename=base_name, file_path=file_path)
        return base_name, None, language_category, {}

    def _parse_complex_train_format(self, name: str) -> tuple:
        """Parse: codeparrot_github-code-clean_data_train-00012-of-00880"""
        import re
        train_pattern = r'(.+)_data_train-(\d+)-of-(\d+)$'
        match = re.match(train_pattern, name)

        if match:
            dataset_part, file_num, total_files = match.groups()
            # Convert org_repo format
            dataset_name = dataset_part.replace('_', '/', 1) if '_' in dataset_part else dataset_part

            metadata = {
                'file_number': file_num,
                'total_files': total_files,
                'format': 'train_split'
            }

            return dataset_name, None, metadata

        return None

    def _parse_huggingface_org_format(self, name: str) -> tuple:
        """Parse: HuggingFaceFW_fineweb-2_deu_Latn_0159"""
        import re
        parts = name.split('_')
        if len(parts) < 3:
            return None

        # Common HuggingFace organizations
        hf_orgs = ['HuggingFaceFW', 'HuggingFaceH4', 'allenai', 'oscar-corpus', 'four-two-labs', 'togethercomputer']

        if parts[0] in hf_orgs and len(parts) >= 3:
            # Format: org_repo_subset_lang_number
            org = parts[0]
            repo = parts[1]
            remaining = parts[2:]

            dataset_name = f"{org}/{repo}"

            # Extract language codes, file numbers
            lang_parts = []
            file_number = None

            for part in remaining:
                if self._is_language_code(part) or part in self.script_codes:
                    lang_parts.append(part)
                elif re.match(r'^\d+[a-zA-Z]*$', part):  # number possibly with suffix
                    file_number = part

            subset = '_'.join(lang_parts) if lang_parts else None
            metadata = {'file_number': file_number} if file_number else {}

            return dataset_name, subset, metadata

        return None

    def _parse_underscore_format(self, name: str) -> tuple:
        """Parse standard underscore-separated format"""
        import re
        parts = name.split('_')

        if len(parts) < 2:
            return None

        # Look for file number (usually at the end)
        file_number = None
        content_parts = parts

        if re.match(r'^\d+[a-zA-Z]*$', parts[-1]):
            file_number = parts[-1]
            content_parts = parts[:-1]

        # Try to identify org/repo vs single name
        if len(content_parts) >= 2:
            # Check if first part looks like org name
            potential_org = content_parts[0]
            if any(org_name in potential_org.lower() for org_name in ['hugging', 'allen', 'oscar', 'microsoft', 'together']):
                dataset_name = f"{content_parts[0]}/{content_parts[1]}"
                remaining = content_parts[2:]
            else:
                # Treat first part as single dataset name
                dataset_name = content_parts[0]
                remaining = content_parts[1:]
        else:
            dataset_name = content_parts[0]
            remaining = []

        # Extract language codes from remaining parts
        lang_parts = [part for part in remaining if self._is_language_code(part) or part in self.script_codes]
        subset = '_'.join(lang_parts) if lang_parts else None

        metadata = {'file_number': file_number} if file_number else {}

        return dataset_name, subset, metadata

    def _parse_dash_format(self, name: str) -> tuple:
        """Parse dash-separated format"""
        # Convert dashes to underscores and reuse underscore parser
        underscore_name = name.replace('-', '_')
        return self._parse_underscore_format(underscore_name)

    def _parse_generic_format(self, name: str) -> tuple:
        """Fallback parser for any remaining format"""
        import re

        # Extract any numbers (likely file numbers)
        numbers = re.findall(r'\d+', name)
        file_number = numbers[-1] if numbers else None

        # Remove numbers to get base name
        base_name = re.sub(r'_?\d+[a-zA-Z]*$', '', name)

        # Look for language codes anywhere in the name
        all_parts = re.split(r'[_\-\.]', name)
        lang_parts = [part for part in all_parts if self._is_language_code(part) or part in self.script_codes]

        subset = '_'.join(lang_parts) if lang_parts else None
        metadata = {'file_number': file_number} if file_number else {}

        return base_name, subset, metadata

    def _is_language_code(self, code: str) -> bool:
        """Check if string is a valid language code"""
        code_lower = code.lower()

        # Check 2-letter codes
        for lang_list in self.language_patterns_2.values():
            if code_lower in lang_list:
                return True

        # Check 3-letter codes
        for lang_list in self.language_patterns_3.values():
            if code_lower in lang_list:
                return True

        return False

    def _detect_language_category(self, dataset_name: str, subset: str, original_filename: str = None, file_path: str = None) -> str:
        """Intelligent automatic language detection with fallback chain - optimized for training mode"""

        # For training mode, use only fast detection methods (no API calls or content sampling)
        # This keeps training HPC-friendly and fast

        # Initialize detection context
        detection_context = {
            'dataset_name': dataset_name,
            'subset': subset,
            'original_filename': original_filename,
            'file_path': file_path,
            'methods_tried': [],
            'detection_results': {}
        }

        # FAST PATH: Dynamic ISO Code Extraction (training-friendly)
        try:
            result = self._detect_via_iso_codes(detection_context)
            if result != 'unknown':
                return result
        except Exception as e:
            detection_context['methods_tried'].append(f'iso_codes_failed: {str(e)}')

        # Note: HuggingFace API and content sampling are only used in --analyze-ratios mode
        # This keeps training fast and HPC-compatible

        return 'unknown'

    def _detect_via_huggingface_api(self, context: dict) -> str:
        """Detect language via HuggingFace dataset metadata API"""
        try:
            # Try to import huggingface_hub
            import requests
            from huggingface_hub import DatasetCardData, list_datasets

            dataset_name = context['dataset_name']

            # Extract potential org/repo from dataset name
            if '/' in dataset_name:
                repo_id = dataset_name
            else:
                # Try common dataset patterns to construct repo_id
                potential_repos = [
                    dataset_name,  # Direct match
                    f"HuggingFace/{dataset_name}",
                    f"huggingface/{dataset_name}",
                    f"{dataset_name.lower()}/{dataset_name.lower()}"
                ]

                repo_id = None
                for candidate in potential_repos:
                    try:
                        # Quick check if dataset exists
                        response = requests.head(f"https://huggingface.co/datasets/{candidate}", timeout=2)
                        if response.status_code == 200:
                            repo_id = candidate
                            break
                    except:
                        continue

                if not repo_id:
                    context['methods_tried'].append('huggingface_api: no matching repo found')
                    return 'unknown'

            # Query dataset metadata
            try:
                from huggingface_hub import dataset_info
                info = dataset_info(repo_id, timeout=3)

                # Extract language information
                if hasattr(info, 'cardData') and info.cardData:
                    if hasattr(info.cardData, 'language') and info.cardData.language:
                        languages = info.cardData.language
                        if isinstance(languages, list) and languages:
                            # Map first language to our categories
                            detected_lang = self._map_language_to_category(languages[0])
                            if detected_lang != 'unknown':
                                context['methods_tried'].append(f'huggingface_api: found {languages[0]}')
                                return detected_lang

                context['methods_tried'].append('huggingface_api: no language metadata')
                return 'unknown'

            except Exception as e:
                context['methods_tried'].append(f'huggingface_api: query failed - {str(e)}')
                return 'unknown'

        except ImportError:
            context['methods_tried'].append('huggingface_api: huggingface_hub not installed')
            return 'unknown'
        except Exception as e:
            context['methods_tried'].append(f'huggingface_api: error - {str(e)}')
            return 'unknown'

    def _detect_via_content_sampling(self, context: dict) -> str:
        """Detect language by sampling actual parquet file content"""
        try:
            # Try to import langdetect
            import pandas as pd
            import pyarrow.parquet as pq
            from langdetect import LangDetectException, detect

            file_path = context['file_path']
            if not file_path:
                context['methods_tried'].append('content_sampling: no file path provided')
                return 'unknown'

            # Sample content from parquet file
            try:
                # Read small sample of the parquet file
                table = pq.read_table(file_path, columns=None)
                df = table.to_pandas()

                # Find text columns
                text_columns = []
                for col in df.columns:
                    if df[col].dtype == 'object':  # String columns
                        # Check if it looks like text content
                        sample_values = df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            avg_length = sample_values.astype(str).str.len().mean()
                            if avg_length > 20:  # Assume longer strings are text content
                                text_columns.append(col)

                if not text_columns:
                    context['methods_tried'].append('content_sampling: no text columns found')
                    return 'unknown'

                # Sample text content
                sample_texts = []
                for col in text_columns[:2]:  # Max 2 text columns
                    sample_data = df[col].dropna().head(100)  # Sample first 100 rows
                    for text in sample_data:
                        if isinstance(text, str) and len(text.strip()) > 50:
                            sample_texts.append(text.strip()[:500])  # Max 500 chars per sample

                if len(sample_texts) < 5:
                    context['methods_tried'].append('content_sampling: insufficient text content')
                    return 'unknown'

                # Combine samples for detection
                combined_text = ' '.join(sample_texts[:20])  # Max 20 samples

                # Detect language
                detected_lang_code = detect(combined_text)
                detected_category = self._map_language_to_category(detected_lang_code)

                context['methods_tried'].append(f'content_sampling: detected {detected_lang_code}')
                return detected_category

            except Exception as e:
                context['methods_tried'].append(f'content_sampling: file read error - {str(e)}')
                return 'unknown'

        except ImportError:
            context['methods_tried'].append('content_sampling: langdetect not installed')
            return 'unknown'
        except LangDetectException as e:
            context['methods_tried'].append(f'content_sampling: detection failed - {str(e)}')
            return 'unknown'
        except Exception as e:
            context['methods_tried'].append(f'content_sampling: error - {str(e)}')
            return 'unknown'

    def _detect_via_iso_codes(self, context: dict) -> str:
        """Extract ISO language codes dynamically from filenames"""
        import re

        # Combine all available text for analysis
        texts_to_analyze = [
            context.get('original_filename', ''),
            context.get('dataset_name', ''),
            context.get('subset', '') or ''
        ]

        full_text = ' '.join(filter(None, texts_to_analyze)).lower()

        # Enhanced regex patterns for language code extraction
        patterns = [
            r'([a-z]{2,3})_latn',           # swe_Latn, eng_Latn format
            r'([a-z]{2,3})_[a-z]{4}',       # any language_Script format
            r'(?<=[_-])([a-z]{2,3})(?=[_-]|$)',  # 2-3 letter codes after underscore/dash
            r'^([a-z]{2,3})(?=[_-])',       # 2-3 letter codes at start
        ]

        detected_codes = []
        for pattern in patterns:
            matches = re.findall(pattern, full_text)
            detected_codes.extend(matches)

        # Filter valid language codes and map to categories
        for code in detected_codes:
            if len(code) >= 2 and code not in ['iew', 'thh', 'erg', 'med', 'www', 'com']:  # Filter common false positives
                category = self._map_language_to_category(code)
                if category != 'unknown':
                    context['methods_tried'].append(f'iso_codes: found {code}')
                    return category

        context['methods_tried'].append('iso_codes: no valid codes found')
        return 'unknown'

    def _map_language_to_category(self, language_code: str) -> str:
        """Map ISO language codes to our categories dynamically"""
        if not language_code:
            return 'unknown'

        code = language_code.lower().strip()

        # ISO 639-1 and 639-3 mappings (no hardcoding of datasets, just standard language codes)
        language_mappings = {
            'nordic': {
                # ISO 639-1
                'sv', 'da', 'no', 'fi', 'is',
                # ISO 639-3
                'swe', 'dan', 'nor', 'fin', 'isl', 'nno', 'nob', 'sme', 'fao'
            },
            'english': {
                'en', 'eng'
            },
            'european': {
                # Major European languages
                'de', 'fr', 'es', 'it', 'nl', 'pl', 'pt', 'ro', 'hu', 'cs', 'sk', 'bg', 'hr', 'sl', 'et', 'lv', 'lt',
                'deu', 'fra', 'spa', 'ita', 'nld', 'pol', 'por', 'ron', 'hun', 'ces', 'slk', 'bul', 'hrv', 'slv', 'est', 'lav', 'lit'
            },
            'code': {
                # This would be detected via content analysis, not language codes
                # But we can leave this for potential filename patterns
            }
        }

        # Check mappings
        for category, codes in language_mappings.items():
            if code in codes:
                return category

        return 'unknown'

    def _extract_language_codes_from_text(self, text: str) -> str:
        """Extract and categorize language codes from text using regex"""
        import re

        if not text:
            return 'unknown'

        text_lower = text.lower()

        # Enhanced regex patterns for language code extraction
        patterns = [
            r'([a-z]{2,3})_latn',           # swe_Latn, eng_Latn format
            r'([a-z]{2,3})_[a-z]{4}',       # any language_Script format
            r'(?<=[_-])([a-z]{2,3})(?=[_-]|$)',  # 2-3 letter codes after underscore/dash
            r'^([a-z]{2,3})(?=[_-])',       # 2-3 letter codes at start
        ]

        detected_codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            detected_codes.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_codes = []
        for code in detected_codes:
            if code not in seen and code not in self.script_codes and len(code) >= 2:
                seen.add(code)
                unique_codes.append(code)

        # Categorize detected codes
        for code in unique_codes:
            # Check 2-letter codes first
            for category, codes in self.language_patterns_2.items():
                if code in codes:
                    return category

            # Check 3-letter codes
            for category, codes in self.language_patterns_3.items():
                if code in codes:
                    return category

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

        for (dataset, subset), info in dataset_inventory.items():
            language = info['language']
            size_gb = info['total_size_gb']

            if language not in available:
                available[language] = {
                    'total_gb': 0,
                    'datasets': [],
                    'largest_dataset_gb': 0
                }

            available[language]['total_gb'] += size_gb
            available[language]['datasets'].append((dataset, subset, size_gb))
            available[language]['largest_dataset_gb'] = max(
                available[language]['largest_dataset_gb'],
                size_gb
            )

        return available

    def _calculate_optimal_sampling(self, dataset_inventory: dict,
                                  available_by_language: dict,
                                  target_by_language: dict) -> dict:
        """Calculate optimal per-dataset sampling ratios"""
        sampling_plan = {}

        for language, target_gb in target_by_language.items():
            if language not in available_by_language:
                logger.warning(f"No data available for language category: {language}")
                continue

            available_gb = available_by_language[language]['total_gb']

            # Calculate language-level sampling ratio
            if available_gb <= target_gb:
                # Use all available data for this language
                lang_sampling_ratio = 1.0
                logger.info(f"Using all {available_gb:.1f}GB available for {language} (target: {target_gb:.1f}GB)")
            else:
                # Need to sample down
                lang_sampling_ratio = target_gb / available_gb
                logger.info(f"Sampling {lang_sampling_ratio:.3f} of {language} data ({target_gb:.1f}GB from {available_gb:.1f}GB)")

            # Apply this ratio to all datasets in this language
            for (dataset, subset), info in dataset_inventory.items():
                if info['language'] == language:
                    sampling_plan[(dataset, subset)] = {
                        'language': language,
                        'available_gb': info['total_size_gb'],
                        'sampling_ratio': lang_sampling_ratio,
                        'target_gb': info['total_size_gb'] * lang_sampling_ratio,
                        'file_count': info['file_count'],
                        'target_files': max(1, int(info['file_count'] * lang_sampling_ratio)),
                        'files': info['files']
                    }

        return sampling_plan

    def _validate_and_adjust(self, sampling_plan: dict, target_total_gb: float) -> dict:
        """Validate total doesn't exceed target, adjust if needed"""
        # Calculate actual total
        actual_total_gb = sum(plan['target_gb'] for plan in sampling_plan.values())

        logger.info(f"Planned total: {actual_total_gb:.1f}GB, Target: {target_total_gb:.1f}GB")

        if actual_total_gb > target_total_gb:
            # Apply proportional scaling to fit exactly within target
            scale_factor = target_total_gb / actual_total_gb
            logger.info(f"Applying final scale factor: {scale_factor:.3f} to fit target")

            for key, plan in sampling_plan.items():
                plan['sampling_ratio'] *= scale_factor
                plan['target_gb'] *= scale_factor
                plan['target_files'] = max(1, int(plan['target_files'] * scale_factor))
                plan['scaled'] = True
        else:
            logger.info("No additional scaling needed")

        return sampling_plan


class DatasetCompositionAnalyzer:
    """Analyze dataset composition and provide intelligent feedback for ratio validation"""

    def __init__(self):
        self.filename_parser = UniversalFilenameParser()

    def _detect_language_for_analysis(self, dataset_name: str, subset: str, original_filename: str = None, file_path: str = None) -> str:
        """Full intelligent language detection for analysis mode - includes API calls and content sampling"""

        # Initialize detection context
        detection_context = {
            'dataset_name': dataset_name,
            'subset': subset,
            'original_filename': original_filename,
            'file_path': file_path,
            'methods_tried': [],
            'detection_results': {}
        }

        # Full Automatic Fallback Chain (used only in analysis mode)

        # 1. PRIMARY: HuggingFace API Metadata Query (fast, authoritative)
        try:
            result = self.filename_parser._detect_via_huggingface_api(detection_context)
            if result != 'unknown':
                return result
        except Exception as e:
            detection_context['methods_tried'].append(f'huggingface_api_failed: {str(e)}')

        # 2. SECONDARY: Content Sampling + Language Detection (accurate)
        if file_path:
            try:
                result = self.filename_parser._detect_via_content_sampling(detection_context)
                if result != 'unknown':
                    return result
            except Exception as e:
                detection_context['methods_tried'].append(f'content_sampling_failed: {str(e)}')

        # 3. TERTIARY: Dynamic ISO Code Extraction (fast fallback)
        try:
            result = self.filename_parser._detect_via_iso_codes(detection_context)
            if result != 'unknown':
                return result
        except Exception as e:
            detection_context['methods_tried'].append(f'iso_codes_failed: {str(e)}')

        # 4. LAST RESORT: Log what was tried and return unknown
        logger.debug(f"Language detection failed for {original_filename or dataset_name}. Tried: {detection_context['methods_tried']}")
        return 'unknown'

    def analyze_available_data(self, datasets_dir: str) -> dict:
        """
        Analyze what data is actually available in the datasets directory.
        
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
                'warnings': [list of warnings/issues]
            }
        """
        from pathlib import Path

        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            return {'error': f"Datasets directory not found: {datasets_dir}"}

        parquet_files = list(datasets_path.glob("*.parquet"))
        if not parquet_files:
            return {'error': f"No parquet files found in: {datasets_dir}"}

        # Build inventory using existing logic
        dataset_inventory = {}
        total_size_gb = 0
        languages = {}
        datasets_found = set()
        warnings = []

        for file_path in parquet_files:
            try:
                # Use basic filename parsing first
                dataset_name, subset, _, metadata = self.filename_parser.parse_filename(file_path.name)

                # Then use full analysis detection (includes API calls and content sampling)
                language_category = self._detect_language_for_analysis(dataset_name, subset, file_path.name, str(file_path))

                # Get file size
                file_size_gb = file_path.stat().st_size / (1024**3)
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
                languages[language_category]['file_count'] += 1

            except Exception as e:
                warnings.append(f"Could not process {file_path.name}: {e}")

        # Calculate percentages
        for lang_info in languages.values():
            lang_info['percentage'] = (lang_info['size_gb'] / total_size_gb) * 100 if total_size_gb > 0 else 0
            lang_info['datasets'] = list(lang_info['datasets'])  # Convert set to list

        return {
            'total_size_gb': total_size_gb,
            'languages': languages,
            'datasets': list(datasets_found),
            'warnings': warnings
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
        requested_languages = set(lang for lang, ratio in user_language_ratios.items() if ratio > 0)

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
        logger.error("❌ File accessibility issues found:")
        for issue in missing_files[:10]:  # Show first 10 issues
            logger.error(f"   - {issue}")
        if len(missing_files) > 10:
            logger.error(f"   ... and {len(missing_files)-10} more issues")
        return False

    logger.info(f"✅ File validation: {len(file_paths)} files accessible, {total_size/1024**3:.1f}GB total")
    return True


def log_sampling_plan_summary(target_data_gb=1500):
    """Log the sampling plan summary based on DATASET_LANGUAGE_MAP"""
    category_totals = dict.fromkeys(LANGUAGE_CATEGORIES.keys(), 0.0)

    # Calculate totals from the mapping
    for key, (category, size_mb, sampling_ratio) in DATASET_LANGUAGE_MAP.items():
        category_totals[category] += size_mb * sampling_ratio

    total_sampled_size = sum(category_totals.values())
    total_original_size = sum(size_mb for _, size_mb, _ in DATASET_LANGUAGE_MAP.values())

    logger.info("=== DATASET SAMPLING PLAN ===")
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
    unique_datasets = len(set(dataset for dataset, subset in dataset_inventory.keys()))
    logger.info(f"Discovered: {unique_datasets} datasets, {len(dataset_inventory)} dataset/subset combinations, {total_discovered_gb:.1f}GB total")

    # Log by language category
    by_language = {}
    for (dataset, subset), info in dataset_inventory.items():
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
    logger.info(f"Target: {target_data_gb:.1f}GB with ratios {user_language_ratios}")

    calculator = DynamicRatioCalculator()
    sampling_plan = calculator.calculate_sampling_strategy(
        dataset_inventory,
        target_data_gb,
        user_language_ratios
    )

    # STEP 3: APPLY SAMPLING BASED ON CALCULATED RATIOS
    logger.info("=== APPLYING DYNAMIC SAMPLING ===")
    random.seed(42)  # Reproducible sampling

    final_files = {}
    total_sampled_gb = 0.0

    for (dataset, subset), plan in sampling_plan.items():
        available_files = plan['files']
        target_files = plan['target_files']

        # Apply sampling
        if target_files < len(available_files):
            sampled_files = random.sample(available_files, target_files)
            logger.info(f"Sampled {dataset}" + (f" ({subset})" if subset else "") + f": {len(available_files)} → {target_files} files ({plan['sampling_ratio']:.3f} ratio)")
        else:
            sampled_files = available_files

        # Store in final structure
        if dataset not in final_files:
            final_files[dataset] = {}

        subset_key = subset if subset else "main"
        final_files[dataset][subset_key] = sampled_files

        total_sampled_gb += plan['target_gb']

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
        for config, files in configs.items():
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

    for (dataset_name, subset), info in dataset_inventory.items():
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

    local_files = discover_local_parquet_files(
        target_data_gb=target_data_gb,
        enable_sampling=not disable_sampling,
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
    "--target-data-size",
    default="1500GB",
    show_default=True,
    help="Target total data size (e.g., 500MB, 1.5GB, 2TB, 2.5TB)",
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
    try:
        target_data_gb = parse_data_size(target_data_size)
        logger.info(f"Target data size: {target_data_size} ({target_data_gb}GB)")
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
            logger.info(f"   python {sys.argv[0]} --target-data-size {target_data_size} \\")
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
