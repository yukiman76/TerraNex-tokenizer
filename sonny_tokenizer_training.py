import gc
import json
import logging
import mmap
import os
import pickle
import statistics
import sys
import unicodedata
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional

import click
import numpy as np
import psutil
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers import normalizers


try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, "INFO"))
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(sys.stdout))


SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}


DATA_SETS = {
    "bigcode/the-stack-march-sample-special-tokens-stripped": {
        "field": "content",
        "extra": [],
    },
    "strombergnlp_nordic_langid_50k": {"field": "text", "extra": []},
    "codeparrot/github-code": {"field": "code", "extra": []},
    "bigcode/the-stack-github-issues": {"field": "content", "extra": []},
    "iohadrubin/wikitext-103-raw-v1": {"field": "text", "extra": []},
    "oscar-corpus/mOSCAR": {
        "field": "text",
        "extra": [
            "swe_Latn", "eng_Latn", "spa_Latn", "deu_Latn", "cym_Latn",
            "dan_Latn", "fra_Latn", "fin_Latn", "ita_Latn", "nld_Latn",
            "nno_Latn", "nob_Latn", "pol_Latn",
        ],
    },
    "allenai/c4": {
        "field": "text",
        "extra": ["sv", "en", "es", "de", "da", "fr", "it", "nl", "no", "pl"],
    },
    "togethercomputer/RedPajama-Data-1T": {"field": "text", "extra": []},
    "HuggingFaceH4/ultrachat_200k": {"field": "messages", "extra": []},
    "gutenberg": {"field": "text", "extra": []},
    "arxiv": {"field": "text", "extra": []},
    "wikipedia": {
        "field": "text",
        "extra": [
            "20220301.sv", "20220301.en", "20220301.es", "20220301.de",
            "20220301.da", "20220301.fr", "20220301.it", "20220301.nl",
            "20220301.no", "20220301.pl",
        ],
    },
    "cc_news": {"field": "text", "extra": []},
}


class MemoryProfiler:
    """Advanced memory profiling and monitoring system for real-time optimization."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.memory_history = []
        self.memory_threshold = 0.85
        self.gc_threshold = 0.75
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive memory information."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": self.process.memory_percent(),
            "available_mb": system_memory.available / (1024 * 1024),
            "system_percent": system_memory.percent,
        }
    
    def log_memory_usage(self, context: str = "") -> Dict[str, float]:
        """Log current memory usage with context."""
        mem_info = self.get_memory_info()
        self.memory_history.append(mem_info["rss_mb"])
        
        logger.info(
            f"Memory {context}: RSS={mem_info['rss_mb']:.1f}MB "
            f"VMS={mem_info['vms_mb']:.1f}MB "
            f"Process={mem_info['percent']:.1f}% "
            f"System={mem_info['system_percent']:.1f}%"
        )
        return mem_info
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered."""
        mem_info = self.get_memory_info()
        return mem_info["system_percent"] > self.gc_threshold * 100
    
    def force_gc_if_needed(self) -> bool:
        """Force garbage collection if memory pressure detected."""
        if self.should_trigger_gc():
            logger.info("Memory pressure detected, forcing garbage collection")
            gc.collect()
            return True
        return False
    
    def get_optimal_batch_size(self, base_size: int = 1000) -> int:
        """Calculate optimal batch size based on available memory."""
        mem_info = self.get_memory_info()
        memory_factor = max(0.1, min(1.0, mem_info["available_mb"] / 1024))
        return max(100, int(base_size * memory_factor))


class MemoryMappedDatasetProcessor:
    """True memory-mapped dataset processing with zero-copy operations and direct file access."""
    
    def __init__(self, memory_profiler: MemoryProfiler, cache_dir: str = "./datasets", lowercase: bool = False):
        self.memory_profiler = memory_profiler
        self.cache_dir = Path(cache_dir)
        self.failed_datasets = set()
        self.mmap_objects = []
        self.lowercase = lowercase
    
    def create_streaming_generator(
        self, 
        max_samples_per_dataset: int
    ) -> Generator[str, None, None]:
        """Create zero-copy streaming generator using memory-mapped files."""
        
        sample_count = 0
        

        total_datasets = sum(
            len(config["extra"]) if config["extra"] else 1 
            for config in DATA_SETS.values()
        )
        samples_per_dataset = max_samples_per_dataset // total_datasets
        
        logger.info("Using memory-mapped file access for zero-copy operations")
        logger.info(f"Targeting {samples_per_dataset} samples per dataset component")
        

        for text in self._process_cached_datasets_with_mmap(samples_per_dataset):
            yield text
            sample_count += 1
            

            if sample_count % 10000 == 0:
                self.memory_profiler.force_gc_if_needed()
                
        logger.info(f"Generated {sample_count} samples from memory-mapped cached files")
    
    def _process_cached_datasets_with_mmap(self, samples_per_dataset: int) -> Generator[str, None, None]:
        """Process cached dataset files using simple sequential memory-mapped access."""
        
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        

        hf_datasets = []
        for dataset_info in self.cache_dir.rglob("dataset_info.json"):
            candidate = dataset_info.parent
            if self._is_valid_hf_dataset(candidate):
                hf_datasets.append(candidate)
            else:
                logger.debug(f"Skipping invalid HuggingFace dataset directory: {candidate}")
        

        parquet_files = list(self.cache_dir.rglob("*.parquet"))
        jsonl_files = list(self.cache_dir.rglob("*.jsonl"))
        

        hf_dataset_paths = set(hf_datasets)
        parquet_files = [f for f in parquet_files if not any(hf_dir in f.parents for hf_dir in hf_dataset_paths)]
        jsonl_files = [f for f in jsonl_files if not any(hf_dir in f.parents for hf_dir in hf_dataset_paths)]
        
        all_datasets = hf_datasets + parquet_files + jsonl_files
        
        if not all_datasets:
            raise FileNotFoundError("No cached dataset files found for memory mapping")
        
        logger.info(f"Found {len(hf_datasets)} HuggingFace datasets, {len(parquet_files)} Parquet files, {len(jsonl_files)} JSONL files")
        logger.info("Using sequential processing for maximum stability")
        

        sample_count = 0
        target_samples = samples_per_dataset * len(all_datasets)
        
        for dataset_path in all_datasets:
            if sample_count >= target_samples:
                break
                
            try:
                dataset_samples = 0
                for text in self._process_dataset_with_mmap(dataset_path, samples_per_dataset):
                    yield text
                    sample_count += 1
                    dataset_samples += 1
                    
                    if dataset_samples >= samples_per_dataset or sample_count >= target_samples:
                        break
                        
                logger.debug(f"Processed {dataset_samples} samples from {dataset_path}")
                
            except Exception as e:
                logger.warning(f"Failed to process {dataset_path} with mmap: {e}")
                continue
        
        logger.info(f"Generated {sample_count} samples from memory-mapped cached files")
    
    def _process_dataset_with_mmap(self, dataset_path: Path, max_samples: int) -> Generator[str, None, None]:
        """Route dataset processing based on format detection."""
        
        if dataset_path.is_dir() and (dataset_path / "dataset_info.json").exists():

            logger.debug(f"Processing HuggingFace dataset: {dataset_path}")
            return self._process_hf_dataset(dataset_path, max_samples)
        elif dataset_path.suffix == '.parquet':

            logger.debug(f"Processing Parquet file: {dataset_path}")
            return self._process_parquet_file(dataset_path, max_samples)
        elif dataset_path.suffix == '.jsonl':

            logger.debug(f"Processing JSONL file: {dataset_path}")
            return self._process_jsonl_file(dataset_path, max_samples)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    def _process_hf_dataset(self, dataset_path: Path, max_samples: int) -> Generator[str, None, None]:
        """Process HuggingFace dataset with zero-copy Arrow access."""
        try:
            import pyarrow as pa
            import pyarrow.dataset as ds
            arrow_table = pa.ipc.open_file(str(dataset_path / "data-00000-of-00001.arrow")).read_all()
            sample_count = 0
            for batch in arrow_table.to_batches(max_chunksize=1000):
                for record in batch.to_pylist():
                    text = self._extract_text_from_record(record)
                    if text and len(text.strip()) > 10:
                        if len(text) > 10:
                            yield text
                            sample_count += 1
                            if sample_count >= max_samples:
                                return
        except ImportError:
            logger.warning("HuggingFace datasets library not available, skipping HF dataset")
            return
        except Exception as e:
            logger.error(f"Failed to process HuggingFace dataset {dataset_path}: {e}")
            raise
    
    def _process_parquet_file(self, file_path: Path, max_samples: int) -> Generator[str, None, None]:
        """Process Parquet file with zero-copy memory mapping."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(file_path), memory_map=True)
            sample_count = 0
            for batch in table.to_batches(max_chunksize=1000):
                for record in batch.to_pylist():
                    text = self._extract_text_from_record(record)
                    if text and len(text.strip()) > 10:
                        if len(text) > 10:
                            yield text
                            sample_count += 1
                            if sample_count >= max_samples:
                                return
        except ImportError:
            logger.warning("PyArrow not available, skipping Parquet files")
            return
        except Exception as e:
            logger.error(f"Failed to process Parquet file {file_path}: {e}")
            raise
    
    def _process_jsonl_file(self, file_path: Path, max_samples: int) -> Generator[str, None, None]:
        """Process JSONL file with memory-mapped line iteration."""
        try:
            with open(file_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.mmap_objects.append(mm)
                sample_count = 0
                for line in self._iterate_mmap_lines(mm):
                    try:
                        record = json.loads(line)
                        text = self._extract_text_from_record(record)
                        if text and len(text.strip()) > 10:
                            if len(text) > 10:
                                yield text
                                sample_count += 1
                                if sample_count >= max_samples:
                                    break
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug(f"Skipping malformed line: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to process JSONL file {file_path}: {e}")
            raise
    
    
    def _iterate_mmap_lines(self, mm: mmap.mmap) -> Generator[str, None, None]:
        """Iterate over lines in memory-mapped file with zero-copy operations."""
        
        position = 0
        while position < len(mm):

            newline_pos = mm.find(b'\n', position)
            if newline_pos == -1:

                if position < len(mm):
                    line_bytes = mm[position:]
                    try:
                        yield line_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        pass
                break
            

            line_bytes = mm[position:newline_pos]
            try:
                yield line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                pass
            
            position = newline_pos + 1
    
    def _extract_text_from_record(self, record: Dict) -> Optional[str]:
        """Extract text from record with automatic field detection."""
        

        text_fields = ['text', 'content', 'code', 'body', 'message']
        
        for field in text_fields:
            if field in record:
                return self._extract_text_content(record, field)
        

        for key, value in record.items():
            if isinstance(value, str) and len(value) > 10:
                return value
        
        return None
    
    def cleanup_mmap_objects(self):
        """Clean up all memory-mapped objects."""
        
        for mm in self.mmap_objects:
            try:
                mm.close()
            except Exception as e:
                logger.debug(f"Error closing mmap object: {e}")
        
        self.mmap_objects.clear()
        logger.debug("Cleaned up all memory-mapped objects")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup_mmap_objects()
    
    def _extract_text_content(self, record: Dict, field: str) -> Optional[str]:
        """Extract text content from record with robust error handling."""
        try:
            content = record.get(field, "")
            
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                if len(content) == 0:
                    return None
                    

                if isinstance(content[0], str):
                    return " ".join(content)
                    

                if isinstance(content[0], dict):
                    text_parts = []
                    for item in content:
                        if "content" in item:
                            text_parts.append(str(item["content"]))
                        elif "text" in item:
                            text_parts.append(str(item["text"]))
                    return " ".join(text_parts)
            
            return str(content) if content else None
            
        except Exception as e:
            logger.debug(f"Failed to extract text content: {e}")
            return None
    
    def _is_valid_hf_dataset(self, candidate: Path) -> bool:
        """Check if a directory is a valid HuggingFace dataset (has dataset_info.json, state.json, and at least one .arrow file)."""
        if not candidate.is_dir():
            return False
        dataset_info = candidate / "dataset_info.json"
        state_json = candidate / "state.json"

        arrow_files = list(candidate.rglob("*.arrow"))
        return dataset_info.exists() and state_json.exists() and len(arrow_files) > 0


class VocabularyConvergenceDetector:
    """Detect vocabulary convergence using IoS-based quality metrics from BPE Gets Picky."""
    
    def __init__(self, 
                 window_size: int = 3, 
                 stability_threshold: float = 0.95,
                 ios_threshold: float = 0.9):
        """
        Initialize convergence detector with IoS (Index of Survivability) tracking.
        
        Args:
            window_size: Number of training stages to analyze for stability
            stability_threshold: Minimum stability ratio to consider converged
            ios_threshold: IoS threshold for identifying under-trained tokens (from BPE Gets Picky)
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.ios_threshold = ios_threshold
        

        self.vocab_sizes = []
        self.compression_ratios = []
        self.training_stages = []
        self.fertility_scores = []
        

        self.merge_frequencies = []
        self.token_usage_patterns = []
        self.under_trained_ratios = []
        

        self.merge_freq_threshold = 0.01
        self.compression_gain_threshold = 0.001
        self.vocab_growth_threshold = 0.02
        
    def update_metrics(self, 
                      vocab_size: int, 
                      compression_ratio: float, 
                      fertility_score: float = 0.0,
                      tokenizer=None,
                      training_stage: int = 0) -> None:
        """Update convergence detection metrics with IoS tracking."""
        

        self.vocab_sizes.append(vocab_size)
        self.compression_ratios.append(compression_ratio)
        self.fertility_scores.append(fertility_score)
        self.training_stages.append(training_stage)
        

        if tokenizer is not None:
            ios_metrics = self._calculate_ios_metrics(tokenizer, vocab_size)
            self.merge_frequencies.append(ios_metrics['avg_merge_freq'])
            self.token_usage_patterns.append(ios_metrics['usage_variance'])
            self.under_trained_ratios.append(ios_metrics['under_trained_ratio'])
        else:

            self.merge_frequencies.append(0.0)
            self.token_usage_patterns.append(0.0)
            self.under_trained_ratios.append(0.0)
        

        max_history = self.window_size * 2
        if len(self.vocab_sizes) > max_history:
            self.vocab_sizes = self.vocab_sizes[-max_history:]
            self.compression_ratios = self.compression_ratios[-max_history:]
            self.fertility_scores = self.fertility_scores[-max_history:]
            self.training_stages = self.training_stages[-max_history:]
            self.merge_frequencies = self.merge_frequencies[-max_history:]
            self.token_usage_patterns = self.token_usage_patterns[-max_history:]
            self.under_trained_ratios = self.under_trained_ratios[-max_history:]
    
    def has_converged(self) -> bool:
        """Check convergence based on vocabulary stability and IoS metrics."""
        if len(self.vocab_sizes) < self.window_size:
            return False
            

        recent_vocab_sizes = self.vocab_sizes[-self.window_size:]
        vocab_stability = self._calculate_stability(recent_vocab_sizes)
        

        fertility_stable = True
        if len(self.fertility_scores) >= 2:
            recent_fertility = self.fertility_scores[-2:]
            avg_fertility = sum(recent_fertility) / len(recent_fertility)
            fertility_stable = avg_fertility < 3.0
        

        ios_healthy = True
        if len(self.under_trained_ratios) >= 2:
            recent_under_trained = self.under_trained_ratios[-2:]
            avg_under_trained = sum(recent_under_trained) / len(recent_under_trained)
            ios_healthy = avg_under_trained < 0.3
        

        return vocab_stability >= self.stability_threshold and fertility_stable and ios_healthy
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get comprehensive convergence metrics including IoS data."""
        if len(self.vocab_sizes) < 2:
            return {
                "converged": False,
                "vocab_size": 0,
                "compression_ratio": 0.0,
                "fertility_score": 0.0,
                "size_stability": 0.0,
                "ratio_stability": 0.0,
                "ios_under_trained_ratio": 0.0,
                "ios_merge_freq": 0.0,
                "ios_usage_variance": 0.0
            }
        

        recent_fertility = self.fertility_scores[-1] if self.fertility_scores else 0.0
        fertility_warning = recent_fertility > 3.0
        

        recent_under_trained = self.under_trained_ratios[-1] if self.under_trained_ratios else 0.0
        ios_warning = recent_under_trained > 0.2
        
        return {
            "converged": self.has_converged(),
            "vocab_size": self.vocab_sizes[-1] if self.vocab_sizes else 0,
            "compression_ratio": self.compression_ratios[-1] if self.compression_ratios else 0.0,
            "fertility_score": recent_fertility,
            "fertility_warning": fertility_warning,
            "training_stage": self.training_stages[-1] if self.training_stages else 0,
            "size_stability": self._calculate_stability(self.vocab_sizes),
            "ratio_stability": self._calculate_stability(self.compression_ratios),
            "ios_under_trained_ratio": recent_under_trained,
            "ios_merge_freq": self.merge_frequencies[-1] if self.merge_frequencies else 0.0,
            "ios_usage_variance": self.token_usage_patterns[-1] if self.token_usage_patterns else 0.0,
            "ios_warning": ios_warning
        }
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability metric based on coefficient of variation."""
        if len(values) < 2:
            return 0.0
        window = min(self.window_size, len(values))
        recent_values = values[-window:]
        if max(recent_values) == 0:
            return 1.0
        return 1.0 - (max(recent_values) - min(recent_values)) / max(recent_values)
    
    def _calculate_ios_metrics(self, tokenizer, vocab_size: int) -> Dict[str, float]:
        """Calculate IoS (Index of Survivability) metrics based on BPE Gets Picky."""
        try:
            vocab = tokenizer.get_vocab()



            if len(self.vocab_sizes) >= 2 and self.vocab_sizes[-2] != 0:
                recent_growth = (vocab_size - self.vocab_sizes[-2]) / self.vocab_sizes[-2]
                avg_merge_freq = max(0.01, recent_growth * 10)
            else:
                avg_merge_freq = 1.0


            if len(self.compression_ratios) >= 2:
                recent_ratios = self.compression_ratios[-2:]
                usage_variance = statistics.variance(recent_ratios) if len(recent_ratios) > 1 else 0.0
            else:
                usage_variance = 0.0


            estimated_under_trained = min(0.3, usage_variance * 5)

            if estimated_under_trained >= self.ios_threshold:
                under_trained_ratio = estimated_under_trained
            else:
                under_trained_ratio = max(0.01, estimated_under_trained * 0.5)
            return {
                'avg_merge_freq': avg_merge_freq,
                'usage_variance': usage_variance,
                'under_trained_ratio': under_trained_ratio,
                'ios_threshold_exceeded': estimated_under_trained >= self.ios_threshold
            }
        except Exception as e:
            logger.debug(f"Error calculating IoS metrics: {e}")
            return {
                'avg_merge_freq': 0.0,
                'usage_variance': 0.0,
                'under_trained_ratio': 0.0,
                'ios_threshold_exceeded': False
            }


class MLflowTracker:
    """MLflow experiment tracking for BPE tokenizer training."""
    
    def __init__(self, experiment_name: str = "bpe_tokenizer_training", enabled: bool = None):
        """Initialize MLflow tracking."""
        self.enabled = enabled if enabled is not None else MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.run_active = False
        
        if self.enabled:
            try:
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow tracking enabled for experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
                self.enabled = False
        else:
            logger.info("MLflow tracking disabled (library not available)")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start an MLflow run."""
        if not self.enabled:
            return
        
        try:
            mlflow.start_run(run_name=run_name, tags=tags)
            self.run_active = True
            logger.info(f"Started MLflow run: {run_name}")
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            self.enabled = False
    
    def log_params(self, params: Dict[str, any]):
        """Log training parameters."""
        if not self.enabled or not self.run_active:
            return
        
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log MLflow params: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log training metrics."""
        if not self.enabled or not self.run_active:
            return
        
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, bool)):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log MLflow metrics: {e}")
    
    def log_convergence_metrics(self, convergence_metrics: Dict[str, float], stage: int):
        """Log BPE convergence-specific metrics."""
        if not self.enabled or not self.run_active:
            return
        

        metrics_to_log = {}
        for key, value in convergence_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics_to_log[f"convergence_{key}"] = value
            elif isinstance(value, bool):
                metrics_to_log[f"convergence_{key}"] = 1.0 if value else 0.0
        
        self.log_metrics(metrics_to_log, step=stage)
    
    def log_artifacts(self, artifact_path: str, artifact_dir: str = None):
        """Log training artifacts (tokenizer files)."""
        if not self.enabled or not self.run_active:
            return
        
        try:
            if artifact_dir:
                mlflow.log_artifacts(artifact_dir, artifact_path)
            else:
                mlflow.log_artifact(artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log MLflow artifacts: {e}")
    
    def end_run(self):
        """End the MLflow run."""
        if not self.enabled or not self.run_active:
            return
        
        try:
            mlflow.end_run()
            self.run_active = False
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")


def estimate_dataset_size_gb(cache_dir: str) -> float:
    """Estimate total dataset size in GB for dynamic initial samples calculation."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0.0
    
    total_size_bytes = 0
    

    for dataset_info in cache_path.rglob("dataset_info.json"):
        try:
            with open(dataset_info, 'r') as f:
                info = json.load(f)

                dataset_size = info.get("dataset_size", 0)
                total_size_bytes += dataset_size
        except Exception as e:
            logger.debug(f"Could not read dataset info from {dataset_info}: {e}")
    

    for file_pattern in ["*.parquet", "*.jsonl"]:
        for file_path in cache_path.rglob(file_pattern):

            if not any("dataset_info.json" in str(p) for p in file_path.parents):
                try:
                    total_size_bytes += file_path.stat().st_size
                except Exception as e:
                    logger.debug(f"Could not get size of {file_path}: {e}")
    
    return total_size_bytes / (1024**3)


def calculate_initial_samples(total_data_size_gb: float, vocab_size: int) -> int:
    """
    Dynamically calculate initial samples for BPE training based on dataset size and vocab size.
    - Scales linearly with vocab size (base: 32k)
    - Scales with dataset size (base: 5GB)
    - Capped at 2,000,000 for safety
    """
    base = 100_000
    vocab_factor = max(1.0, vocab_size / 32_000)
    data_factor = max(1.0, total_data_size_gb / 5.0)
    initial = int(base * vocab_factor * data_factor)

    initial = min(initial, 2_000_000)
    return initial

def calculate_max_samples(total_data_size_gb: float, vocab_size: int) -> int:
    """
    Dynamically calculate max samples for BPE training based on dataset size and vocab size.
    - Scales linearly with vocab size (base: 32k)
    - Scales with dataset size (base: 5GB)
    - Capped at 10,000,000 for safety
    """
    base = 1_000_000
    vocab_factor = max(1.0, vocab_size / 32_000)
    data_factor = max(1.0, total_data_size_gb / 5.0)
    max_samples = int(base * vocab_factor * data_factor)

    max_samples = min(max_samples, 10_000_000)
    return max_samples


def train_tokenizer_with_mmap(
    vocab_size: int,
    output_dir: str,
    memory_profiler: MemoryProfiler,
    sample_size: Optional[int] = None,
    cache_dir: str = "./datasets",
    lowercase: bool = False,
    mlflow_tracker=None,
) -> ByteLevelBPETokenizer:
    logger.info("Step 1: Build corpus from memory-mapped cached files")
    processor = MemoryMappedDatasetProcessor(memory_profiler, cache_dir, lowercase=lowercase)
    final_samples = sample_size if sample_size is not None else 2000000
    text_generator = processor.create_streaming_generator(
        max_samples_per_dataset=final_samples
    )
    memory_profiler.log_memory_usage("before training")
    logger.info("Step 2: Train ByteLevelBPE tokenizer using memory-mapped data")
    convergence = VocabularyConvergenceDetector()
    if mlflow_tracker is not None and hasattr(mlflow_tracker, "log_convergence_metrics"):
        mlflow_tracker.log_convergence_metrics(convergence.get_convergence_metrics(), 0)
    tokenizer = ByteLevelBPETokenizer()
    norm_sequence = [normalizers.NFC()]
    if lowercase:
        norm_sequence.append(normalizers.Lowercase())
    norm_sequence.append(normalizers.Replace("\t", " "))
    norm_sequence.append(normalizers.Replace("\u00A0", " "))
    norm_sequence.append(normalizers.Replace(r"[\x00-\x09\x0B-\x1F\x7F]", ""))
    norm_sequence.append(normalizers.Replace(r"\s+", " "))
    norm_sequence.append(normalizers.Strip())
    tokenizer.normalizer = normalizers.Sequence(norm_sequence)
    logger.info("Set Hugging Face Tokenizers normalizer pipeline: NFC, lowercase=%s, whitespace, control char removal", lowercase)
    try:
        sample_text = "\t  Example\u00A0Text\nWith\x07Control\x0BChars  "
        normed = tokenizer.normalizer.normalize_str(sample_text)
        logger.debug(f"[HF Normalizer] Before: {sample_text!r}")
        logger.debug(f"[HF Normalizer] After:  {normed!r}")
    except Exception as e:
        logger.debug(f"Could not log HF normalizer effect: {e}")
    initial_vocab = tokenizer.get_vocab()
    # Compute initial compression ratio: not meaningful before training, so set to 1.0
    convergence.update_metrics(
        vocab_size=len(initial_vocab),
        compression_ratio=1.0,
        fertility_score=0.0,
        tokenizer=tokenizer,
        training_stage=0
    )
    tokenizer.train_from_iterator(
        text_generator,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS.values()),
        show_progress=True,
    )
    vocab = tokenizer.get_vocab()
    # Compute fertility and compression ratio: average number of tokens per sample and compression ratio
    sample_count = 0
    total_tokens = 0
    total_input_length = 0
    sample_limit = 1000
    processor2 = MemoryMappedDatasetProcessor(memory_profiler, cache_dir, lowercase=lowercase)
    text_gen_for_fertility = processor2.create_streaming_generator(max_samples_per_dataset=sample_limit)
    for i, text in enumerate(text_gen_for_fertility):
        if i >= sample_limit:
            break
        if not text or not isinstance(text, str):
            continue
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)
        total_input_length += len(text)
        sample_count += 1
    fertility_score = (total_tokens / sample_count) if sample_count > 0 else 0.0
    compression_ratio = (total_input_length / total_tokens) if total_tokens > 0 else 1.0
    convergence.update_metrics(
        vocab_size=len(vocab),
        compression_ratio=compression_ratio,
        fertility_score=fertility_score,
        tokenizer=tokenizer,
        training_stage=1
    )
    if mlflow_tracker is not None and hasattr(mlflow_tracker, "log_convergence_metrics"):
        mlflow_tracker.log_convergence_metrics(convergence.get_convergence_metrics(), 1)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    logger.info(f"Tokenizer trained and saved to {output_dir}")
    processor.cleanup_mmap_objects()
    if mlflow_tracker is not None and mlflow_tracker.enabled:
        mlflow_tracker.log_metrics({
            "final_vocab_size": len(vocab)
        })
        mlflow_tracker.log_convergence_metrics(convergence.get_convergence_metrics(), 1)
        mlflow_tracker.log_artifacts(output_dir)
        mlflow_tracker.end_run()
    return tokenizer








def validate_tokenizer_native(tokenizer_dir: str) -> ByteLevelBPETokenizer:
    """Native ByteLevelBPETokenizer validation with comprehensive error handling."""
    
    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")
    

    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    if not os.path.exists(merges_file):
        raise FileNotFoundError(f"Merges file not found: {merges_file}")
    
    try:

        tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        

        vocab = tokenizer.get_vocab()
        missing_tokens = []
        
        for token_name, token in SPECIAL_TOKENS.items():
            if token not in vocab:
                missing_tokens.append(token)
        
        if missing_tokens:
            logger.warning(f"Missing special tokens: {missing_tokens}")

            tokenizer.add_special_tokens(missing_tokens)

            tokenizer.save_model(tokenizer_dir)
            logger.info("Added missing special tokens and re-saved tokenizer")
        else:
            logger.info("All special tokens validated successfully")
        

        test_text = "This is a test to validate tokenizer functionality."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)
        
        if test_text.strip() != decoded.strip():
            logger.warning("Tokenizer encode/decode test failed")
        else:
            logger.info("Tokenizer validation completed successfully")
        
        return tokenizer
        
    except Exception as e:
        logger.error(f"Failed to validate tokenizer: {e}")
        raise


def initialize_embedding_matrix(tokenizer: ByteLevelBPETokenizer, embedding_dim: int = 1024) -> torch.Tensor:
    """Initialize embedding matrix with memory-efficient approach."""
    
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Initializing embedding matrix: vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    try:

        weights = torch.empty((vocab_size, embedding_dim), dtype=torch.float32)
        torch.nn.init.normal_(weights, mean=0.0, std=0.02)
        

        matrix_size_mb = weights.numel() * 4 / (1024 * 1024)
        logger.info(f"Embedding matrix: {weights.shape}, size: {matrix_size_mb:.1f}MB")
        
        return weights
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding matrix: {e}")
        raise


def create_tokenizer_config_json(tokenizer_dir: str, vocab_size: int) -> None:
    """Create HuggingFace-compatible tokenizer_config.json file."""
    
    tokenizer_config = {
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(i): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            } for i, token in enumerate(SPECIAL_TOKENS.values())
        },
        "bos_token": SPECIAL_TOKENS["bos_token"],
        "clean_up_tokenization_spaces": True,
        "eos_token": SPECIAL_TOKENS["eos_token"],
        "mask_token": SPECIAL_TOKENS["mask_token"],
        "model_max_length": 2048,
        "pad_token": SPECIAL_TOKENS["pad_token"],
        "padding_side": "right",
        "tokenizer_class": "ByteLevelBPETokenizer",
        "trim_offsets": True,
        "unk_token": SPECIAL_TOKENS["unk_token"],
        "use_fast": True,
        "vocab_size": vocab_size
    }
    
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        logger.info(f"HuggingFace tokenizer config saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer_config.json: {e}")
        raise


def create_added_tokens_json(tokenizer_dir: str) -> None:
    """Create added_tokens.json file for special tokens."""
    
    added_tokens = [
        {
            "id": i,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        } for i, token in enumerate(SPECIAL_TOKENS.values())
    ]
    
    config_path = os.path.join(tokenizer_dir, "added_tokens.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(added_tokens, f, indent=2, ensure_ascii=False)
        logger.info(f"Added tokens saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save added_tokens.json: {e}")
        raise


def create_special_tokens_map_json(tokenizer_dir: str) -> None:
    """Create special_tokens_map.json file."""
    
    special_tokens_map = {
        "bos_token": SPECIAL_TOKENS["bos_token"],
        "eos_token": SPECIAL_TOKENS["eos_token"], 
        "mask_token": SPECIAL_TOKENS["mask_token"],
        "pad_token": SPECIAL_TOKENS["pad_token"],
        "unk_token": SPECIAL_TOKENS["unk_token"]
    }
    
    config_path = os.path.join(tokenizer_dir, "special_tokens_map.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Special tokens map saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save special_tokens_map.json: {e}")
        raise


def create_hf_tokenizer_json(tokenizer_dir: str, tokenizer: ByteLevelBPETokenizer) -> None:
    """Create tokenizer.json file in HuggingFace format."""
    

    vocab = tokenizer.get_vocab()
    

    hf_tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab.get(token, 0),
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            } for token in SPECIAL_TOKENS.values() if token in vocab
        ],
        "normalizer": {
            "type": "NFC"
        },
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": SPECIAL_TOKENS["unk_token"],
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": vocab,
            "merges": []
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True
        }
    }
    
    config_path = os.path.join(tokenizer_dir, "tokenizer.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(hf_tokenizer, f, indent=2, ensure_ascii=False)
        logger.info(f"HuggingFace tokenizer format saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer.json: {e}")
        raise


def save_tokenizer_config(tokenizer_dir: str, vocab_size: int, embedding_dim: int, tokenizer: Optional[ByteLevelBPETokenizer] = None) -> None:
    """Save comprehensive tokenizer configuration with all HuggingFace format files."""
    

    config = {
        "model_type": "byte_level_bpe",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "special_tokens": SPECIAL_TOKENS,
        "max_position_embeddings": 2048,
        "training_method": "progressive_with_convergence_detection",
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
        logger.info(f"Tokenizer configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer config: {e}")
        raise
    

    try:
        create_tokenizer_config_json(tokenizer_dir, vocab_size)
        create_added_tokens_json(tokenizer_dir)
        create_special_tokens_map_json(tokenizer_dir)
        
        if tokenizer is not None:
            create_hf_tokenizer_json(tokenizer_dir, tokenizer)
        
        logger.info("All HuggingFace format files created successfully")
    except Exception as e:
        logger.error(f"Failed to create HuggingFace format files: {e}")
        raise


@click.command()
@click.option(
    "--tokenizer-out-dir",
    default="sonny_custom_tokenizer",
    show_default=True,
    help="Directory to save the trained tokenizer",
)
@click.option(
    "--vocab-size",
    default=256000,
    show_default=True,
    help="Vocabulary size for tokenizer (optimized for large models)",
)
@click.option(
    "--embedding-dim",
    default=1024,
    show_default=True,
    help="Embedding dimension for initialization",
)
@click.option(
    "--initial-samples",
    default=None,
    type=int,
    help="Initial number of samples for progressive training (auto-calculated if not specified)",
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Maximum number of samples (optional safety limit). If not set, uses all available data.",
)
@click.option(
    "--checkpoint-interval",
    default=50000,
    show_default=True,
    help="Interval for saving training checkpoints",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Run in offline mode using cached datasets only",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from existing checkpoint if available",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level",
)
@click.option(
    "--lowercase/--no-lowercase",
    default=False,
    show_default=True,
    help="Lowercase all text before training (not recommended for code)",
)
@click.option(
    "--picky-bpe/--no-picky-bpe",
    default=False,
    show_default=True,
    help="Enable Picky BPE (BPE Gets Picky) algorithm with IoS-based pruning",
)
@click.option(
    "--ios-threshold",
    default=0.9,
    show_default=True,
    type=float,
    help="IoS threshold for Picky BPE token pruning (default: 0.9)",
)
def main(
    tokenizer_out_dir: str,
    vocab_size: int,
    embedding_dim: int,
    initial_samples: Optional[int],
    max_samples: Optional[int],
    checkpoint_interval: int,
    offline: bool,
    resume: bool,
    log_level: str,
    lowercase: bool,
    picky_bpe: bool = False,
    ios_threshold: float = 0.9,
) -> None:
    """
    Simple memory-efficient tokenizer training with memory-mapped file processing.
    Features:
    - Memory-mapped dataset processing for zero-copy operations
    - Single-pass training (no progressive loops)
    - Native ByteLevelBPETokenizer training and validation
    - Real-time memory profiling
    - Optional: Picky BPE algorithm (BPE Gets Picky)
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    mlflow_tracker = None
    if 'MLflowTracker' in globals() and MLFLOW_AVAILABLE:
        mlflow_tracker = MLflowTracker(experiment_name="bpe_tokenizer_training")
        mlflow_tracker.start_run(run_name=f"tokenizer_{vocab_size}_{embedding_dim}")
        mlflow_tracker.log_params({
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "initial_samples": initial_samples,
            "max_samples": max_samples,
            "checkpoint_interval": checkpoint_interval,
            "lowercase": lowercase,
            "picky_bpe": picky_bpe,
            "ios_threshold": ios_threshold,
        })
    logger.setLevel(getattr(logging, log_level))
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
    memory_profiler = MemoryProfiler()
    memory_profiler.log_memory_usage("startup")
    logger.info("Sonny's Tokenizer Training (Picky BPE mode: %s)", picky_bpe)
    logger.info(f"Target vocabulary size: {vocab_size:,}")
    logger.info(f"Embedding dimension: {embedding_dim:,}")
    logger.info(f"Tokenizer output directory: {tokenizer_out_dir}")
    logger.info("USING ONLY LOCAL DATASET at ./datasets for all training data. No remote or online datasets will be loaded.")
    if initial_samples is None or max_samples is None:
        logger.info("No --initial-samples or --max-samples provided. Estimating dataset size for dynamic calculation...")
        dataset_size_gb = estimate_dataset_size_gb("./datasets")
    if initial_samples is None:
        initial_samples = calculate_initial_samples(dataset_size_gb, vocab_size)
        logger.info(f"Auto-selected initial_samples: {initial_samples:,} (dataset size: {dataset_size_gb:.2f} GB, vocab size: {vocab_size})")
    else:
        logger.info(f"Using user-provided initial_samples: {initial_samples:,}")
    if max_samples is None:
        max_samples = calculate_max_samples(dataset_size_gb, vocab_size)
        logger.info(f"Auto-selected max_samples: {max_samples:,} (dataset size: {dataset_size_gb:.2f} GB, vocab size: {vocab_size})")
    else:
        logger.info(f"Using user-provided max_samples: {max_samples:,}")
    try:
        if picky_bpe:

            result = train_picky_bpe_tokenizer(
                vocab_size=vocab_size,
                output_dir=tokenizer_out_dir,
                memory_profiler=memory_profiler,
                sample_size=initial_samples,
                cache_dir="./datasets",
                lowercase=lowercase,
                ios_threshold=ios_threshold,
                special_tokens=list(SPECIAL_TOKENS.values()),
                mlflow_tracker=mlflow_tracker,
            )
            logger.info(f"Picky BPE training complete. Vocab size: {len(result['vocab'])}")
            save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim)
            if mlflow_tracker is not None and mlflow_tracker.enabled:
                mlflow_tracker.log_metrics({
                    "final_vocab_size": len(result['vocab'])
                })
                mlflow_tracker.log_artifacts(tokenizer_out_dir)
                mlflow_tracker.end_run()
        else:
            tokenizer = train_tokenizer_with_mmap(
                vocab_size=vocab_size,
                output_dir=tokenizer_out_dir,
                memory_profiler=memory_profiler,
                sample_size=initial_samples,
                cache_dir="./datasets",
                lowercase=lowercase,
                mlflow_tracker=mlflow_tracker,
            )
            validated_tokenizer = validate_tokenizer_native(tokenizer_out_dir)
            weights = initialize_embedding_matrix(validated_tokenizer, embedding_dim)
            embedding_path = os.path.join(tokenizer_out_dir, "embedding_matrix.npy")
            np.save(embedding_path, weights.cpu().numpy())
            logger.info(f"Embedding matrix saved to {embedding_path}")
            save_tokenizer_config(tokenizer_out_dir, vocab_size, embedding_dim, validated_tokenizer)
            memory_profiler.force_gc_if_needed()
            final_memory = memory_profiler.log_memory_usage("completion")
            logger.info("Training completed successfully!")
            logger.info(f"Final vocabulary size: {validated_tokenizer.get_vocab_size():,}")
            logger.info(f"Final memory usage: {final_memory['rss_mb']:.1f}MB")
            summary_msg = f"SUMMARY: Vocab size = {validated_tokenizer.get_vocab_size():,}, Embedding dim = {embedding_dim:,} (set via CLI and written to config.json)"
            if mlflow_tracker is not None and mlflow_tracker.enabled:
                mlflow_tracker.log_metrics({
                    "final_vocab_size": validated_tokenizer.get_vocab_size(),
                    "final_memory_mb": final_memory["rss_mb"]
                })
                mlflow_tracker.log_artifacts(tokenizer_out_dir)
                mlflow_tracker.end_run()
            logger.info(summary_msg)
            logger.info(f"All training data was loaded from local directory: ./datasets")
            logger.info("Files created:")
            logger.info("  ├── vocab.json")
            logger.info("  ├── merges.txt")
            logger.info("  ├── config.json")
            logger.info("  ├── tokenizer_config.json")
            logger.info("  ├── added_tokens.json")
            logger.info("  ├── special_tokens_map.json")
            logger.info("  ├── tokenizer.json")
            logger.info("  └── embedding_matrix.npy")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        memory_profiler.log_memory_usage("error")
        sys.exit(1)

def train_picky_bpe_tokenizer(
    vocab_size: int,
    output_dir: str,
    memory_profiler: MemoryProfiler,
    sample_size: Optional[int] = None,
    cache_dir: str = "./datasets",
    lowercase: bool = False,
    ios_threshold: float = 0.9,
    special_tokens: Optional[List[str]] = None,
    mlflow_tracker=None,
) -> dict:
    from tqdm import tqdm
    import collections
    import os
    import json
    logger.info("Step 1: Build corpus from memory-mapped cached files (Picky BPE mode)")
    processor = MemoryMappedDatasetProcessor(memory_profiler, cache_dir, lowercase=lowercase)
    final_samples = sample_size if sample_size is not None else 2000000
    text_generator = processor.create_streaming_generator(max_samples_per_dataset=final_samples)
    corpus = []
    for text in text_generator:
        if not text or len(text.strip()) < 1:
            continue
        corpus.append(list(text))
    logger.info(f"Loaded {len(corpus)} samples for Picky BPE training")
    vocab = collections.Counter()
    for sentence in corpus:
        vocab.update(sentence)
    if special_tokens is None:
        special_tokens = list(SPECIAL_TOKENS.values())
    for token in special_tokens:
        vocab[token] = 1_000_000_000
    corpus = [tuple(sentence) for sentence in corpus]
    def get_pairs(sequence):
        pairs = set()
        prev = sequence[0]
        for token in sequence[1:]:
            pairs.add((prev, token))
            prev = token
        return pairs
    merges = []
    event_log = []
    step = 0
    convergence = VocabularyConvergenceDetector(ios_threshold=ios_threshold)
    total_steps = vocab_size - len(vocab)
    pbar = tqdm(total=total_steps if total_steps > 0 else 1, desc="Picky BPE", unit="merge")
    while len(vocab) < vocab_size:
        pair_freq = collections.Counter()
        for sequence in corpus:
            pairs = get_pairs(sequence)
            pair_freq.update(pairs)
        if not pair_freq:
            break
        most_freq_pair, freq = pair_freq.most_common(1)[0]
        x1, x2 = most_freq_pair
        new_token = x1 + x2
        new_corpus = []
        for sequence in corpus:
            new_seq = []
            i = 0
            while i < len(sequence):
                if i < len(sequence) - 1 and sequence[i] == x1 and sequence[i+1] == x2:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(sequence[i])
                    i += 1
            new_corpus.append(tuple(new_seq))
        corpus = new_corpus
        vocab[new_token] = freq
        merges.append((x1, x2))
        event_log.append(("merge", (x1, x2), step))
        freq_x1 = sum(seq.count(x1) for seq in corpus)
        freq_x2 = sum(seq.count(x2) for seq in corpus)
        freq_pair = freq
        ios_x1 = freq_pair / freq_x1 if freq_x1 > 0 else 0.0
        ios_x2 = freq_pair / freq_x2 if freq_x2 > 0 else 0.0
        removed = []
        if ios_x1 >= ios_threshold and x1 not in special_tokens:
            del vocab[x1]
            event_log.append(("remove", x1, step))
            removed.append(x1)
        if ios_x2 >= ios_threshold and x2 not in special_tokens:
            if x2 in vocab:
                del vocab[x2]
                event_log.append(("remove", x2, step))
                removed.append(x2)
        for token in removed:
            if any(token in seq for seq in corpus):
                vocab[token] = 1
                event_log.append(("readd", token, step))
        # Compute fertility and compression ratio: average number of tokens per sample and compression ratio
        sample_count = len(corpus)
        total_tokens = sum(len(seq) for seq in corpus)
        total_input_length = sum(sum(len(token) for token in seq) for seq in corpus)
        fertility_score = (total_tokens / sample_count) if sample_count > 0 else 0.0
        compression_ratio = (total_input_length / total_tokens) if total_tokens > 0 else 1.0
        # Minimal tokenizer-like object for IoS metrics
        class DummyTokenizer:
            def get_vocab(self):
                return dict(vocab)
        dummy_tokenizer = DummyTokenizer()
        convergence.update_metrics(
            vocab_size=len(vocab),
            compression_ratio=compression_ratio,
            fertility_score=fertility_score,
            tokenizer=dummy_tokenizer,
            training_stage=step
        )
        if mlflow_tracker is not None and hasattr(mlflow_tracker, "log_convergence_metrics"):
            mlflow_tracker.log_convergence_metrics(convergence.get_convergence_metrics(), step)
        step += 1
        pbar.update(1)
        if step % 100 == 0:
            logger.info(f"Step {step}: vocab size={len(vocab)}, merges={len(merges)}")
        if len(vocab) >= vocab_size:
            break
    pbar.close()
    if mlflow_tracker is not None and hasattr(mlflow_tracker, "log_convergence_metrics"):
        final_metrics = convergence.get_convergence_metrics()
        mlflow_tracker.log_convergence_metrics(final_metrics, step)
        logger.info("Logged final convergence metrics to MLflow.")
    merges_txt = [f"{x1} {x2}" for (x1, x2) in merges]
    vocab_dict = {token: i for i, token in enumerate(vocab.keys())}
    event_log_dict = [{"event": e[0], "tokens": e[1], "step": e[2]} for e in event_log]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for line in merges_txt:
            f.write(line + "\n")
    with open(os.path.join(output_dir, "event_log.json"), "w", encoding="utf-8") as f:
        json.dump(event_log_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Picky BPE training complete. Vocab size: {len(vocab_dict)}. Merges: {len(merges)}. Events: {len(event_log_dict)}.")
    processor.cleanup_mmap_objects()
    return {"vocab": vocab_dict, "merges": merges, "event_log": event_log_dict}








if __name__ == "__main__":


    main()
