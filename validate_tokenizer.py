import os
import json
import signal
import sys
import logging
import click
import yaml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import pandas as pd
from jinja2 import Template
import webbrowser
import requests
from requests.exceptions import RequestException
from contextlib import contextmanager
import tempfile
import shutil
import time
import psutil
import csv
from typing import Dict, List, Optional, Union, Any, Tuple
import logging.handlers
from PIL import Image
import threading
from datasets import load_dataset

# Dataset language code mappings
dataset_lang_codes = {
    "oscar-corpus/mOSCAR": "swe_Latn",  # Swedish
    "statmt/cc100": "sv",              # Swedish
    "wikipedia": "sv"                   # Swedish
}

# Important: Do remember if you set this to false, the tokenizer will not be able to parallelize.
# and if you set it to true, the tokenizer will be able to parallelize but you need to also set the max_workers and everything else.

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class TokenizerValidationError(Exception):
    """Custom exception for tokenizer validation errors"""
    pass

class ResourceLimitExceededError(Exception):
    """Exception raised when resource limits are exceeded"""
    pass

class GracefulExit(Exception):
    """Exception raised when a graceful shutdown is requested"""
    pass

def signal_handler(signum, frame):
    """Handle termination signals by initiating a graceful shutdown"""
    logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
    raise GracefulExit("Shutdown requested")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class TokenizerThreadSafeWrapper:
    """Thread-safe wrapper for the tokenizer."""
    
    def __init__(self, tokenizer_path: str):
        """Initialize the tokenizer wrapper."""
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None
        self._lock = threading.Lock()
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer instance, creating it if necessary."""
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer
    
    def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze text using the tokenizer."""
        try:
            tokenizer = self.get_tokenizer()
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            
            # Calculate character coverage
            unique_chars = set(text)
            vocab_chars = set(''.join(tokenizer.get_vocab().keys()))
            char_coverage = len(unique_chars.intersection(vocab_chars)) / len(unique_chars) if unique_chars else 0
            
            # Calculate token statistics
            token_lengths = [len(token) for token in tokens]
            avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
            std_token_length = np.std(token_lengths) if len(token_lengths) > 1 else 0
            
            # Calculate tokenization ratio
            tokens_per_char = len(tokens) / len(text) if len(text) > 0 else 0
            
            metrics = {
                'tokens': len(tokens),
                'chars': len(text),
                'tokens_per_char': tokens_per_char,
                'char_coverage': char_coverage,
                'avg_token_length': avg_token_length,
                'std_token_length': std_token_length,
                'unique_tokens': len(set(tokens)),
                'token_lengths': token_lengths
            }
            
            logging.debug(f"Text analysis metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            return None

@contextmanager
def temporary_directory():
    """Context manager for creating and cleaning up temporary directories"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class ResourceMonitor:
    """Monitor system resources and enforce limits"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process = psutil.Process()
        self._initialize_cpu_measurement()
    
    def _initialize_cpu_measurement(self):
        """Initialize CPU measurement by getting an initial reading"""
        self.process.cpu_percent()
        time.sleep(0.1)  # Wait for a short interval
        self.process.cpu_percent()  # Get the first real measurement
    
    def check_limits(self) -> None:
        """Check if current resource usage exceeds limits"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            if memory_mb > self.config['resource_limits']['max_memory_mb']:
                raise ResourceLimitExceededError(
                    f"Memory usage ({memory_mb:.1f}MB) exceeds limit "
                    f"({self.config['resource_limits']['max_memory_mb']}MB)"
                )
            
            if cpu_percent > self.config['resource_limits']['max_cpu_percent']:
                raise ResourceLimitExceededError(
                    f"CPU usage ({cpu_percent:.1f}%) exceeds limit "
                    f"({self.config['resource_limits']['max_cpu_percent']}%)"
                )
        except Exception as e:
            logging.warning(f"Failed to check resource limits: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file with enhanced error handling"""
    if not os.path.exists(config_path):
        raise TokenizerValidationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise TokenizerValidationError("Configuration file must contain a valid YAML dictionary")
        
        # Validate required configuration fields
        required_fields = {
            'tokenizer_path': str,
            'output_dir': str,
            'log_dir': str,
            'sample_size': int,
            'max_workers': int,
            'z_threshold': float,
            'languages': list,
            'logging': dict,
            'metrics': list,
            'anomaly_detection': dict,
            'resource_limits': dict
        }
        
        # Check required fields
        missing_fields = []
        type_errors = []
        for field, field_type in required_fields.items():
            if field not in config:
                missing_fields.append(field)
            elif not isinstance(config[field], field_type):
                type_errors.append(f"{field} (expected {field_type.__name__}, got {type(config[field]).__name__})")
        
        if missing_fields:
            raise TokenizerValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        if type_errors:
            raise TokenizerValidationError(f"Type errors in fields: {', '.join(type_errors)}")
        
        # Set default values for optional fields
        config.setdefault('temp_dir', 'temp')
        config.setdefault('retry_attempts', 3)
        config.setdefault('retry_delay', 5)
        config.setdefault('fallback_datasets', [])
        config.setdefault('output_formats', ['json', 'html'])
        
        # Validate nested configurations
        if 'logging' in config:
            logging_config = config['logging']
            logging_config.setdefault('max_size', 10485760)  # 10MB
            logging_config.setdefault('backup_count', 5)
            logging_config.setdefault('console_level', 'INFO')
            logging_config.setdefault('file_level', 'DEBUG')
        
        return config
    except yaml.YAMLError as e:
        raise TokenizerValidationError(f"Invalid YAML format in configuration file: {e}")
    except Exception as e:
        raise TokenizerValidationError(f"Failed to load config file: {e}")

def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on config"""
    try:
        log_dir = Path(config['log_dir'])
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / datetime.now().strftime(config['logging']['file_format'])

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config['logging']['max_size'],
            backupCount=config['logging']['backup_count']
        )
        file_handler.setLevel(getattr(logging, config['logging']['file_level']))
        file_handler.setFormatter(logging.Formatter(config['logging']['format']))
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config['logging']['console_level']))
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)
        
        # Log initial configuration
        logging.debug("Logging configured with the following settings:")
        logging.debug(f"File level: {config['logging']['file_level']}")
        logging.debug(f"Console level: {config['logging']['console_level']}")
        logging.debug(f"Log file: {log_file}")
        
    except Exception as e:
        raise TokenizerValidationError(f"Failed to setup logging: {e}")

def validate_tokenizer_path(tokenizer_path: str) -> bool:
    """Validate tokenizer path or model name"""
    try:
        # Check if it's a local path
        if os.path.exists(tokenizer_path):
            return True
        
        # Check if it's a valid model name on Hugging Face Hub
        try:
            response = requests.head(f"https://huggingface.co/{tokenizer_path}")
            if response.status_code == 200:
                return True
        except RequestException:
            pass
        
        return False
    except Exception as e:
        logging.error(f"Error validating tokenizer path: {e}")
        return False

def load_test_samples(language_code: str, config: Dict[str, Any]) -> List[str]:
    """Load test samples for a specific language."""
    sample_size = config['sample_size']
    failed_datasets = []
    
    # First we try the primary dataset
    try:
        primary_dataset_name = "oscar-corpus/mOSCAR"
        primary_dataset = load_dataset(
            primary_dataset_name,
            dataset_lang_codes[primary_dataset_name],
            split="train",
            trust_remote_code=True
        )
        # Extract just the text from each sample and ensure it's a string
        test_samples = []
        for item in primary_dataset.select(range(sample_size)):
            if isinstance(item, dict) and 'text' in item:
                text = item['text']
                if isinstance(text, str):
                    test_samples.append(text)
                elif isinstance(text, list):
                    # Handle case where text is a list of dictionaries
                    for text_item in text:
                        if isinstance(text_item, dict) and 'text' in text_item:
                            inner_text = text_item['text']
                            if isinstance(inner_text, str):
                                test_samples.append(inner_text)
        logging.info(f"Successfully loaded {len(test_samples)} samples for {language_code}")
        return test_samples
    except Exception as e:
        failed_datasets.append(("oscar-corpus/mOSCAR", str(e)))
        logging.warning(f"Failed to load primary dataset for {language_code}: {e}")
    
    # Then we try the fallback datasets
    for dataset_name in config['fallback_datasets']:
        try:
            dataset = load_dataset(
                dataset_name,
                dataset_lang_codes[dataset_name],
                split="train",
                trust_remote_code=True
            )
            # Extract just the text from each sample and ensure it's a string
            test_samples = []
            for item in dataset.select(range(sample_size)):
                if isinstance(item, dict) and 'text' in item:
                    text = item['text']
                    if isinstance(text, str):
                        test_samples.append(text)
                    elif isinstance(text, list):
                        # Handle case where text is a list of dictionaries
                        for text_item in text:
                            if isinstance(text_item, dict) and 'text' in text_item:
                                inner_text = text_item['text']
                                if isinstance(inner_text, str):
                                    test_samples.append(inner_text)
            logging.info(f"Successfully loaded {len(test_samples)} samples from {dataset_name} for {language_code}")
            return test_samples
        except Exception as e:
            failed_datasets.append((dataset_name, str(e)))
            logging.warning(f"Failed to load {dataset_name} for {language_code}: {e}")
    
    logging.error(f"Failed to load samples for {language_code} from any dataset: {failed_datasets}")
    return []

def verify_image_file(file_path: Path) -> bool:
    """Verify that a file is a valid image"""
    try:
        # First check if file exists and has content
        if not file_path.exists():
            logging.error(f"File does not exist: {file_path}")
            return False
            
        if file_path.stat().st_size == 0:
            logging.error(f"File is empty: {file_path}")
            return False
            
        # Open and verify the image in a single operation
        try:
            with Image.open(file_path) as img:
                # Verify immediately after opening
                img.verify()
                # Create a new image to load the data
                with Image.open(file_path) as img2:
                    img2.load()
                    # Check if it's a PNG
                    if img2.format != 'PNG':
                        logging.error(f"File is not a PNG: {file_path}")
                        return False
                    # Check if it has valid dimensions
                    if img2.size[0] == 0 or img2.size[1] == 0:
                        logging.error(f"Image has invalid dimensions: {file_path}")
                        return False
            return True
        except Exception as e:
            logging.error(f"Failed to verify image {file_path}: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error processing image file {file_path}: {e}")
        return False

def verify_visualization_files(output_dir: Path) -> None:
    """Verify that all required visualization files exist and are valid."""
    required_files = [
        'tokens_per_char.png',
        'metrics_heatmap.png',
        'consistency_scatter.png',
        'char_coverage.png'
    ]
    
    missing_files = []
    for filename in required_files:
        file_path = output_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        elif file_path.stat().st_size == 0:
            missing_files.append(filename)
        elif not verify_image_file(file_path):
            missing_files.append(filename)
    
    if missing_files:
        raise TokenizerValidationError(f"Missing files: {', '.join(missing_files)}")

def save_results(results: Dict[str, Any], config: Dict[str, Any], output_dir: Path) -> None:
    """Save results in multiple formats"""
    try:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for format in config['output_formats']:
            if format == 'json':
                with open(output_dir / "tokenizer_validation_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            elif format == 'csv':
                with open(output_dir / "tokenizer_validation_results.csv", "w", newline='', encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'language', 'status', 'avg_tokens', 'avg_chars', 'avg_tokens_per_char', 'sample_count'
                    ])
                    writer.writeheader()
                    for lang, result in results.items():
                        if result['status'] == 'success' and 'metrics' in result:
                            metrics = result['metrics']
                            writer.writerow({
                                'language': lang,
                                'status': result['status'],
                                'avg_tokens': metrics.get('avg_tokens', 0),
                                'avg_chars': metrics.get('avg_chars', 0),
                                'avg_tokens_per_char': metrics.get('avg_tokens_per_char', 0),
                                'sample_count': metrics.get('sample_count', 0)
                            })
                        else:
                            writer.writerow({
                                'language': lang,
                                'status': result['status'],
                                'avg_tokens': 0,
                                'avg_chars': 0,
                                'avg_tokens_per_char': 0,
                                'sample_count': 0
                            })
            
            elif format == 'md':
                with open(output_dir / "tokenizer_validation_results.md", "w", encoding="utf-8") as f:
                    f.write("# Tokenizer Validation Results\n\n")
                    f.write("## Summary Statistics\n\n")
                    f.write("| Language | Status | Avg Tokens | Avg Chars | Tokens/Char | Samples |\n")
                    f.write("|----------|--------|------------|-----------|-------------|----------|\n")
                    for lang, result in results.items():
                        if result['status'] == 'success' and 'metrics' in result:
                            metrics = result['metrics']
                            f.write(f"| {lang} | {result['status']} | "
                                   f"{metrics.get('avg_tokens', 0):.1f} | "
                                   f"{metrics.get('avg_chars', 0):.1f} | "
                                   f"{metrics.get('avg_tokens_per_char', 0):.3f} | "
                                   f"{metrics.get('sample_count', 0)} |\n")
                        else:
                            f.write(f"| {lang} | {result['status']} | 0 | 0 | 0 | 0 |\n")
    
    except Exception as e:
        raise TokenizerValidationError(f"Error saving results: {e}")

def analyze_tokenization(tokenizer: AutoTokenizer, text: str) -> Optional[Dict[str, Any]]:
    """Enhanced tokenization analysis with additional metrics"""
    try:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        unique_tokens = set(tokens)
        
        # Calculate token lengths
        token_lengths = [len(token) for token in tokens]
        
        # Calculate character coverage
        unique_chars = set(text)
        vocab_chars = set(''.join(tokenizer.get_vocab().keys()))
        char_coverage = len(unique_chars.intersection(vocab_chars)) / len(unique_chars) if unique_chars else 0
        
        return {
            "num_tokens": len(tokens),
            "num_chars": len(text),
            "tokens": tokens,
            "ids": ids,
            "unique_token_count": len(unique_tokens),
            "token_length_mean": np.mean(token_lengths),
            "token_length_std": np.std(token_lengths),
            "char_coverage": char_coverage,
            "ratio": len(tokens) / len(text) if len(text) > 0 else 0
        }
    except Exception as e:
        logging.error(f"Error analyzing tokenization: {e}", exc_info=True)
        return None

def detect_anomalies(ratios: List[float], z_threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalous tokenization ratios using Z-score"""
    try:
        z_scores = np.abs(stats.zscore(ratios))
        anomalies = z_scores > z_threshold
        return anomalies, z_scores
    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}", exc_info=True)
        return np.zeros_like(ratios, dtype=bool), np.zeros_like(ratios)

def validate_language(language_code: str, tokenizer_wrapper: TokenizerThreadSafeWrapper, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tokenizer performance for a specific language."""
    try:
        # Load test samples
        test_samples = load_test_samples(language_code, config)
        if not test_samples:
            return {
                'language': language_code,
                'status': 'error',
                'error': 'No test samples available'
            }
        
        # Process each sample
        results = []
        ratios = []
        char_coverages = []
        token_lengths = []
        unique_tokens = set()
        
        for sample in test_samples:
            # Skip empty samples or non-string samples
            if not sample or not isinstance(sample, str):
                logging.debug(f"Skipping invalid sample type: {type(sample)}")
                continue
                
            # Clean the sample
            sample = sample.strip()
            if not sample:
                continue
                
            # Get tokenization metrics
            metrics = tokenizer_wrapper.analyze_text(sample)
            if metrics:
                results.append(metrics)
                ratios.append(metrics['tokens_per_char'])
                char_coverages.append(metrics['char_coverage'])
                token_lengths.extend(metrics['token_lengths'])
                # Get the actual tokens from the tokenizer
                tokens = tokenizer_wrapper.get_tokenizer().tokenize(sample)
                unique_tokens.update(tokens)
        
        if not results:
            return {
                'language': language_code,
                'status': 'error',
                'error': 'No valid results were generated'
            }
        
        # Calculate aggregate metrics
        avg_tokens = sum(r['tokens'] for r in results) / len(results)
        avg_chars = sum(r['chars'] for r in results) / len(results)
        avg_tokens_per_char = sum(r['tokens_per_char'] for r in results) / len(results)
        std_tokens_per_char = np.std(ratios) if len(ratios) > 1 else 0
        avg_char_coverage = sum(char_coverages) / len(char_coverages) if char_coverages else 0
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        std_token_length = np.std(token_lengths) if len(token_lengths) > 1 else 0
        
        # Detect anomalies using tokenization ratio only
        anomalies, z_scores = detect_anomalies(ratios, config['z_threshold'])
        anomaly_count = int(np.sum(anomalies))
        
        metrics = {
            'avg_tokens': avg_tokens,
            'avg_chars': avg_chars,
            'avg_tokens_per_char': avg_tokens_per_char,
            'std_tokens_per_char': std_tokens_per_char,
            'anomaly_count': anomaly_count,
            'avg_char_coverage': avg_char_coverage,
            'avg_token_length': avg_token_length,
            'std_token_length': std_token_length,
            'unique_token_count': len(unique_tokens),
            'sample_count': len(results)
        }
        
        logging.info(f"Language {language_code} metrics: {metrics}")
        
        return {
            'language': language_code,
            'status': 'success',
            'metrics': metrics
        }
        
    except Exception as e:
        logging.error(f"Error processing {language_code}: {str(e)}")
        return {
            'language': language_code,
            'status': 'error',
            'error': str(e)
        }

def visualize_metrics(results, config):
    """Generate visualizations of the validation results."""
    try:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)

        # Extract metrics for visualization
        languages = []
        tokens_per_char = []
        std_tokens_per_char = []
        anomaly_counts = []
        char_coverage = []
        sample_counts = []
        avg_token_lengths = []
        std_token_lengths = []
        unique_token_counts = []

        for lang, result in results.items():
            if result['status'] == 'success' and 'metrics' in result:
                metrics = result['metrics']
                languages.append(lang)
                tokens_per_char.append(metrics.get('avg_tokens_per_char', 0))
                std_tokens_per_char.append(metrics.get('std_tokens_per_char', 0))
                anomaly_counts.append(metrics.get('anomaly_count', 0))
                char_coverage.append(metrics.get('avg_char_coverage', 0))
                sample_counts.append(metrics.get('sample_count', 0))
                avg_token_lengths.append(metrics.get('avg_token_length', 0))
                std_token_lengths.append(metrics.get('std_token_length', 0))
                unique_token_counts.append(metrics.get('unique_token_count', 0))

        if not languages:
            logging.warning("No valid results to visualize")
            return

        sns.set_theme(style="whitegrid")

        # 1. Bar plot for tokens per character
        fig1 = plt.figure(figsize=(12, 6))
        plt.bar(languages, tokens_per_char)
        plt.title('Average Tokens per Character by Language')
        plt.ylabel('Avg Tokens per Char')
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig1.savefig(str(output_dir / 'tokens_per_char.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # 2. Heatmap for all key metrics
        metrics_matrix = np.array([
            tokens_per_char,
            std_tokens_per_char,
            anomaly_counts,
            char_coverage,
            avg_token_lengths,
            std_token_lengths,
            unique_token_counts
        ])
        metric_labels = [
            'Avg Tokens/Char',
            'Std Dev',
            'Anomalies',
            'Char Coverage',
            'Avg Token Length',
            'Token Length Std',
            'Unique Tokens'
        ]
        fig2, ax2 = plt.subplots(figsize=(max(8, len(languages)), 8))
        sns.heatmap(metrics_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   yticklabels=metric_labels, xticklabels=languages, ax=ax2)
        plt.title('Validation Metrics Heatmap')
        plt.tight_layout()
        fig2.savefig(str(output_dir / 'metrics_heatmap.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # 3. Scatter plot: sample count vs tokens per char, color by anomaly count
        fig3 = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(sample_counts, tokens_per_char, c=anomaly_counts,
                            cmap='coolwarm', alpha=0.7, s=100)
        for i, lang in enumerate(languages):
            plt.annotate(lang, (sample_counts[i], tokens_per_char[i]), fontsize=9, ha='right')
        plt.xlabel('Number of Samples')
        plt.ylabel('Average Tokens per Character')
        plt.title('Sample Size vs Tokenization Ratio (color=anomalies)')
        plt.colorbar(scatter, label='Anomaly Count')
        plt.tight_layout()
        fig3.savefig(str(output_dir / 'consistency_scatter.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # 4. Bar plot for character coverage
        fig4 = plt.figure(figsize=(12, 6))
        plt.bar(languages, char_coverage)
        plt.title('Character Coverage by Language')
        plt.ylabel('Coverage Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig4.savefig(str(output_dir / 'char_coverage.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close(fig4)

        plt.close('all')

        # Verify the files were created and are valid
        for filename in ['tokens_per_char.png', 'metrics_heatmap.png', 'consistency_scatter.png', 'char_coverage.png']:
            file_path = output_dir / filename
            if not file_path.exists():
                logging.warning(f"Failed to create {filename}")
            elif file_path.stat().st_size == 0:
                logging.warning(f"Created empty file {filename}")
            elif not verify_image_file(file_path):
                logging.warning(f"Created invalid image file {filename}")

    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        # Don't raise the error, just log it and continue

def generate_html_report(results, config):
    """Generate an HTML report with results and visualizations"""
    try:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Verify visualization files exist
        verify_visualization_files(output_dir)
        
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tokenizer Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .warning { color: #ff6b6b; }
                .success { color: #51cf66; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Tokenizer Validation Report</h1>
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Language</th>
                    <th>Status</th>
                    <th>Avg Tokens/Char</th>
                    <th>Std Dev</th>
                    <th>Anomalies</th>
                    <th>Char Coverage</th>
                    <th>Samples</th>
                </tr>
                {% for lang, result in results.items() %}
                <tr>
                    <td>{{ lang }}</td>
                    <td class="{{ 'warning' if result.status == 'error' or (result.metrics.anomaly_count if result.status == 'success' and 'metrics' in result and 'anomaly_count' in result.metrics else 0) > 0 else 'success' }}">
                        {{ '⚠️' if result.status == 'error' or (result.metrics.anomaly_count if result.status == 'success' and 'metrics' in result and 'anomaly_count' in result.metrics else 0) > 0 else '✅' }}
                    </td>
                    {% if result.status == 'success' and 'metrics' in result %}
                    <td>{{ "%.3f"|format(result.metrics.avg_tokens_per_char) if 'avg_tokens_per_char' in result.metrics else 'N/A' }}</td>
                    <td>{{ "%.3f"|format(result.metrics.std_tokens_per_char) if 'std_tokens_per_char' in result.metrics else 'N/A' }}</td>
                    <td>{{ result.metrics.anomaly_count if 'anomaly_count' in result.metrics else 0 }}</td>
                    <td>{{ "%.2f"|format(result.metrics.avg_char_coverage) if 'avg_char_coverage' in result.metrics else 'N/A' }}</td>
                    <td>{{ result.metrics.sample_count if 'sample_count' in result.metrics else 0 }}</td>
                    {% else %}
                    <td colspan="6">{{ result.error if 'error' in result else 'N/A' }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </table>
            
            <h2>Visualizations</h2>
            <img src="tokens_per_char.png" alt="Tokens per Character">
            <img src="metrics_heatmap.png" alt="Metrics Heatmap">
            <img src="consistency_scatter.png" alt="Consistency Scatter">
            <img src="char_coverage.png" alt="Character Coverage">
        </body>
        </html>
        """
        
        report_path = output_dir / "report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(Template(template).render(results=results))
        
        return report_path
    except Exception as e:
        raise TokenizerValidationError(f"Error generating HTML report: {e}")

@click.command()
@click.option('--config', '-c', 
              default='config.yaml',
              help='Path to configuration file',
              type=click.Path(exists=True))
@click.option('--sample-size', '-s',
              help='Override sample size from config',
              type=int)
@click.option('--max-workers', '-w',
              help='Override max workers from config',
              type=int)
@click.option('--languages', '-l',
              help='Override languages from config (space-separated)',
              type=str)
@click.option('--no-browser', is_flag=True,
              help='Do not open browser with report')
@click.option('--output-format', '-f',
              help='Output format (json, csv, md, html)',
              type=click.Choice(['json', 'csv', 'md', 'html']),
              multiple=True)
@click.option('--log-level',
              help='Logging level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']))
def main(config, sample_size, max_workers, languages, no_browser, output_format, log_level):
    """Validate tokenizer performance across multiple languages."""
    config_data = None
    executor = None
    try:
        config_data = load_config(config)
        
        # Override config with command line arguments if provided
        if sample_size:
            config_data['sample_size'] = sample_size
        if max_workers:
            config_data['max_workers'] = max_workers
        if languages:
            config_data['languages'] = languages.split()
        if output_format:
            config_data['output_formats'] = list(output_format)
        if log_level:
            config_data['logging']['console_level'] = log_level
            config_data['logging']['file_level'] = log_level
        
        setup_logging(config_data)
        logging.info("Starting tokenizer validation")
        
        resource_monitor = ResourceMonitor(config_data)
        resource_monitor.check_limits()

        tokenizer_path = config_data['tokenizer_path']
        if not validate_tokenizer_path(tokenizer_path):
            raise TokenizerValidationError(
                f"Invalid tokenizer path or model name: {tokenizer_path}"
            )
        
        tokenizer_wrapper = TokenizerThreadSafeWrapper(tokenizer_path)
        logging.info("Successfully initialized tokenizer wrapper")

        validation_args = [(lang, tokenizer_wrapper, config_data) for lang in config_data['languages']]
        
        results = {}
        executor = ThreadPoolExecutor(max_workers=config_data['max_workers'])
        futures = {executor.submit(validate_language, *args): args[0] 
                  for args in validation_args}
        
        try:
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc="Validating languages"):
                lang = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[lang] = result
                    resource_monitor.check_limits()
                except Exception as e:
                    logging.error(f"Error processing {lang}: {e}", exc_info=True)
        except GracefulExit:
            logging.info("Graceful shutdown requested. Waiting for current tasks to complete...")
            # Cancel pending futures
            for future in futures:
                if not future.done():
                    future.cancel()
            # Wait for running tasks to complete
            for future in futures:
                if not future.cancelled():
                    try:
                        future.result(timeout=5)  # Give tasks 5 seconds to complete
                    except Exception:
                        pass
            raise
        
        if not results:
            raise TokenizerValidationError("No valid results were generated")
        
        # Generate visualizations and save results
        visualize_metrics(results, config_data)
        output_dir = Path(config_data['output_dir'])
        save_results(results, config_data, output_dir)
        
        if 'html' in config_data['output_formats']:
            report_path = generate_html_report(results, config_data)
            if not no_browser:
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
        
        # Print summary
        click.echo("\nTokenization Quality Summary:")
        click.echo("-" * 120)
        click.echo(f"{'Language':<8} {'Samples':<8} {'Avg Tokens/Char':<15} {'Std Dev':<10} {'Anomalies':<10} {'Char Coverage':<15} {'Status':<10}")
        click.echo("-" * 120)

        for lang, result in results.items():
            if result["status"] == "success" and "metrics" in result:
                metrics = result["metrics"]
                avg_tokens_per_char = metrics.get("avg_tokens_per_char", 0)
                std_tokens_per_char = metrics.get("std_tokens_per_char", 0)
                anomaly_count = metrics.get("anomaly_count", 0)
                avg_char_coverage = metrics.get("avg_char_coverage", 0)
                sample_count = metrics.get("sample_count", 0)
                status = "⚠️" if anomaly_count > 0 else "✅"
                click.echo(f"{lang:<8} {sample_count:<8} {avg_tokens_per_char:<15.3f} {std_tokens_per_char:<10.3f} {anomaly_count:<10} {avg_char_coverage:<15.2f} {status:<10}")
            else:
                status = "⚠️"
                click.echo(f"{lang:<8} {'N/A':<8} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'N/A':<15} {status:<10}")
        
        logging.info(f"Results saved to {output_dir}")
        logging.info("Validation completed successfully")
        
    except GracefulExit as e:
        logging.info(f"Graceful shutdown initiated: {e}")
        if config_data and results:
            try:
                output_dir = Path(config_data['output_dir'])
                save_results(results, config_data, output_dir)
                logging.info("Partial results saved before exit")
            except Exception as save_error:
                logging.error(f"Failed to save partial results: {save_error}")
        sys.exit(0)
    except TokenizerValidationError as e:
        logging.error(f"Validation error: {e}")
        sys.exit(1)
    except ResourceLimitExceededError as e:
        logging.error(f"Resource limit exceeded: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Critical error during validation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if executor:
            executor.shutdown(wait=False)
        if config_data and config_data.get('resource_limits', {}).get('cleanup_on_exit', True):
            temp_dir = Path(config_data.get('temp_dir', 'temp'))
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main() 