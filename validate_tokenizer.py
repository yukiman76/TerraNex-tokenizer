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
    """Thread-safe wrapper for tokenizer operations using thread-local storage"""
    def __init__(self, tokenizer_path: str):
        self._tokenizer_path = tokenizer_path
        self._local = threading.local()
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get or initialize tokenizer in a thread-safe manner using thread-local storage"""
        if not hasattr(self._local, 'tokenizer'):
            self._local.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        return self._local.tokenizer

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

def load_test_samples(language_code: str, sample_size: int, config: Dict[str, Any]) -> List[str]:
    """Load test samples for a specific language with retry mechanism"""
    logging.info(f"Loading {sample_size} test samples for language: {language_code}")
    test_samples = []
    failed_datasets = []
    
    for attempt in range(config['retry_attempts']):
        try:
            from datasets import load_dataset
            
            # First we try the primary dataset
            try:
                dataset = load_dataset("oscar-corpus/mOSCAR", language_code, split="test")
                test_samples.extend([item["text"] for item in dataset.select(range(sample_size))])
                logging.info(f"Successfully loaded {len(test_samples)} samples for {language_code}")
                return test_samples
            except Exception as e:
                failed_datasets.append(("oscar-corpus/mOSCAR", str(e)))
                logging.warning(f"Failed to load primary dataset for {language_code}: {e}")
            
            # Then we try the fallback datasets
            for dataset_name in config['fallback_datasets']:
                try:
                    dataset = load_dataset(dataset_name, language_code, split="test")
                    test_samples.extend([item["text"] for item in dataset.select(range(sample_size))])
                    logging.info(f"Successfully loaded {len(test_samples)} samples from {dataset_name} for {language_code}")
                    return test_samples
                except Exception as e:
                    failed_datasets.append((dataset_name, str(e)))
                    logging.warning(f"Failed to load {dataset_name} for {language_code}: {e}")
            
            if attempt < config['retry_attempts'] - 1:
                time.sleep(config['retry_delay'])
                
        except Exception as e:
            logging.error(f"Error loading samples for {language_code}: {e}")
            if attempt < config['retry_attempts'] - 1:
                time.sleep(config['retry_delay'])
                
    error_msg = f"Failed to load samples for {language_code} after {config['retry_attempts']} attempts.\n"
    error_msg += "Failed datasets and reasons:\n"
    for dataset_name, error in failed_datasets:
        error_msg += f"- {dataset_name}: {error}\n"
    logging.error(error_msg)
    return []

def verify_image_file(file_path: Path) -> bool:
    """Verify that a file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify it's an image
            img.load()    # Try to load it
        return True
    except Exception as e:
        logging.error(f"Invalid image file {file_path}: {e}")
        return False

def verify_visualization_files(output_dir: Path) -> None:
    """Verify that all visualization files exist and are valid"""
    required_files = [
        "boxplot_by_language.png",
        "metrics_heatmap.png",
        "consistency_scatter.png"
    ]
    
    missing_files = []
    invalid_files = []
    corrupted_files = []
    
    for file in required_files:
        file_path = output_dir / file
        if not file_path.exists():
            missing_files.append(file)
        elif file_path.stat().st_size == 0:
            invalid_files.append(file)
        elif not verify_image_file(file_path):
            corrupted_files.append(file)
    
    if missing_files or invalid_files or corrupted_files:
        error_msg = []
        if missing_files:
            error_msg.append(f"Missing files: {', '.join(missing_files)}")
        if invalid_files:
            error_msg.append(f"Invalid files: {', '.join(invalid_files)}")
        if corrupted_files:
            error_msg.append(f"Corrupted files: {', '.join(corrupted_files)}")
        raise TokenizerValidationError("; ".join(error_msg))

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
                        'language', 'num_samples', 'avg_tokens_per_char',
                        'std_tokens_per_char', 'anomaly_count', 'avg_char_coverage'
                    ])
                    writer.writeheader()
                    for lang, result in results.items():
                        writer.writerow({
                            'language': lang,
                            'num_samples': result['num_samples'],
                            'avg_tokens_per_char': result['avg_tokens_per_char'],
                            'std_tokens_per_char': result['std_tokens_per_char'],
                            'anomaly_count': result['anomaly_count'],
                            'avg_char_coverage': result['avg_char_coverage']
                        })
            
            elif format == 'md':
                with open(output_dir / "tokenizer_validation_results.md", "w", encoding="utf-8") as f:
                    f.write("# Tokenizer Validation Results\n\n")
                    f.write("## Summary Statistics\n\n")
                    f.write("| Language | Samples | Avg Tokens/Char | Std Dev | Anomalies | Char Coverage | Status |\n")
                    f.write("|----------|---------|----------------|---------|-----------|---------------|--------|\n")
                    for lang, result in results.items():
                        status = "⚠️" if result["anomaly_count"] > 0 else "✅"
                        f.write(f"| {lang} | {result['num_samples']} | "
                               f"{result['avg_tokens_per_char']:.3f} | "
                               f"{result['std_tokens_per_char']:.3f} | "
                               f"{result['anomaly_count']} | "
                               f"{result['avg_char_coverage']:.2f} | {status} |\n")
    
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

def validate_language(tokenizer_wrapper: TokenizerThreadSafeWrapper, 
                     language_code: str, 
                     sample_size: int, 
                     config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate tokenizer performance on a specific language"""
    logging.info(f"Starting validation for language: {language_code}")
    
    # Get thread-safe tokenizer instance. I think it works.
    tokenizer = tokenizer_wrapper.get_tokenizer()
    
    samples = load_test_samples(language_code, sample_size, config)
    if not samples:
        logging.warning(f"No samples available for {language_code}")
        return None
    
    results = []
    for i, sample in enumerate(samples):
        if sample.strip():
            result = analyze_tokenization(tokenizer, sample)
            if result:
                results.append(result)
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1} samples for {language_code}")
    
    if not results:
        logging.warning(f"No valid results for {language_code}")
        return None
    
    ratios = [r["ratio"] for r in results]
    anomalies, z_scores = detect_anomalies(ratios, config['z_threshold'])
    
    return {
        "language": language_code,
        "num_samples": len(results),
        "avg_tokens_per_char": np.mean(ratios),
        "std_tokens_per_char": np.std(ratios),
        "min_tokens_per_char": np.min(ratios),
        "max_tokens_per_char": np.max(ratios),
        "anomaly_count": np.sum(anomalies),
        "z_scores": z_scores.tolist(),
        "avg_token_length": np.mean([r["token_length_mean"] for r in results]),
        "token_length_std": np.mean([r["token_length_std"] for r in results]),
        "avg_char_coverage": np.mean([r["char_coverage"] for r in results]),
        "sample_analysis": results[:5]
    }

def visualize_metrics(results, config):
    """Enhanced visualizations with better styling and annotations"""
    try:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # here we set up Seaborn style and create the boxplot
        sns.set(style="whitegrid", context="notebook", font_scale=1.2)
        with sns.axes_style("whitegrid"):
            plt.figure(figsize=tuple(config['figure_size']['boxplot']))
            data = []
            labels = []
            for lang, result in results.items():
                if result["num_samples"] > 0:
                    data.append(result["z_scores"])
                    labels.append(lang)
            
            sns.boxplot(data=data)
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.title("Tokenization Ratio Distribution by Language", pad=20)
            plt.ylabel("Z-score of Tokens per Character")
            plt.tight_layout()
            plt.savefig(output_dir / "boxplot_by_language.png", 
                        dpi=config['dpi'], 
                        bbox_inches='tight')
            plt.close()
        
        # Heatmap
        with sns.axes_style("whitegrid"):
            plt.figure(figsize=tuple(config['figure_size']['heatmap']))
            metrics = np.array([[result["avg_tokens_per_char"], 
                                result["std_tokens_per_char"],
                                result["anomaly_count"],
                                result["avg_char_coverage"]] 
                               for result in results.values()])
            
            metrics_norm = (metrics - metrics.min(axis=0)) / (metrics.max(axis=0) - metrics.min(axis=0))
            
            sns.heatmap(metrics_norm, 
                        annot=metrics,
                        fmt='.3f',
                        xticklabels=['Avg Tokens/Char', 'Std Dev', 'Anomalies', 'Char Coverage'],
                        yticklabels=results.keys(),
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Normalized Value'})
            plt.title("Normalized Tokenization Metrics by Language", pad=20)
            plt.tight_layout()
            plt.savefig(output_dir / "metrics_heatmap.png", 
                        dpi=config['dpi'], 
                        bbox_inches='tight')
            plt.close()
        
        # Scatter plot
        with sns.axes_style("whitegrid"):
            plt.figure(figsize=tuple(config['figure_size']['scatter']))
            for lang, result in results.items():
                plt.scatter(result["avg_tokens_per_char"], 
                           result["std_tokens_per_char"],
                           label=lang,
                           s=100)
                plt.annotate(lang, 
                            (result["avg_tokens_per_char"], result["std_tokens_per_char"]),
                            xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel("Average Tokens per Character")
            plt.ylabel("Standard Deviation")
            plt.title("Tokenization Consistency by Language", pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_dir / "consistency_scatter.png", 
                        dpi=config['dpi'], 
                        bbox_inches='tight')
            plt.close()
        
        # Verify that all visualization files were created
        verify_visualization_files(output_dir)
        
    except Exception as e:
        raise TokenizerValidationError(f"Error generating visualizations: {e}")

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
                    <th>Samples</th>
                    <th>Avg Tokens/Char</th>
                    <th>Std Dev</th>
                    <th>Anomalies</th>
                    <th>Char Coverage</th>
                    <th>Status</th>
                </tr>
                {% for lang, result in results.items() %}
                <tr>
                    <td>{{ lang }}</td>
                    <td>{{ result.num_samples }}</td>
                    <td>{{ "%.3f"|format(result.avg_tokens_per_char) }}</td>
                    <td>{{ "%.3f"|format(result.std_tokens_per_char) }}</td>
                    <td>{{ result.anomaly_count }}</td>
                    <td>{{ "%.2f"|format(result.avg_char_coverage) }}</td>
                    <td class="{{ 'warning' if result.anomaly_count > 0 else 'success' }}">
                        {{ '⚠️' if result.anomaly_count > 0 else '✅' }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Visualizations</h2>
            <img src="boxplot_by_language.png" alt="Box Plot">
            <img src="metrics_heatmap.png" alt="Metrics Heatmap">
            <img src="consistency_scatter.png" alt="Consistency Scatter">
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

        validation_args = [(tokenizer_wrapper, lang, config_data['sample_size'], config_data) 
                         for lang in config_data['languages']]
        
        results = {}
        executor = ThreadPoolExecutor(max_workers=config_data['max_workers'])
        futures = {executor.submit(validate_language, *args): args[1] 
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
        click.echo(f"{'Language':<6} {'Samples':<8} {'Avg Tokens/Char':<15} {'Std Dev':<10} {'Anomalies':<10} {'Char Coverage':<15} {'Status':<10}")
        click.echo("-" * 120)
        
        for lang, result in results.items():
            status = "⚠️" if result["anomaly_count"] > 0 else "✅"
            click.echo(f"{lang:<6} {result['num_samples']:<8} {result['avg_tokens_per_char']:<15.3f} "
                      f"{result['std_tokens_per_char']:<10.3f} {result['anomaly_count']:<10} "
                      f"{result['avg_char_coverage']:<15.2f} {status:<10}")
        
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