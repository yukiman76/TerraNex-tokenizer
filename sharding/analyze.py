"""
analyze.py — Portable streaming analyzer for language ratios in Parquet datasets.

Usage:
    python analyze.py
    python analyze.py --datasets-dir ./datasets --target-data-size 500GB
    python analyze.py --datasets-dir ./datasets --manifest ./out/manifest.csv
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_data_size(size_str: str) -> float:
    import re
    if not isinstance(size_str, str):
        raise ValueError(f"Size must be a string, got: {type(size_str)}")
    text = size_str.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)(MB|GB|TB)$", text)
    if not match:
        raise ValueError("Invalid size format. Use number + unit (MB|GB|TB).")
    value = float(match.group(1))
    if value <= 0:
        raise ValueError(f"Size must be positive, got: {value}")
    unit = match.group(2)
    if unit == "MB":
        return value / 1000.0
    if unit == "GB":
        return value
    return value * 1000.0


def get_memory_usage_percent() -> float:
    return psutil.virtual_memory().percent


def get_total_system_memory_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def get_used_memory_gb(percentage: float) -> float:
    total_gb = get_total_system_memory_gb()
    return (percentage / 100.0) * total_gb


def format_memory_log(percentage: float) -> str:
    used_gb = get_used_memory_gb(percentage)
    total_gb = get_total_system_memory_gb()
    return f"{percentage:.1f}% ({used_gb:.1f}GB/{total_gb:.1f}GB)"


def aggressive_memory_cleanup(context: str = "general") -> None:
    gc.collect()
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    if context in {"analysis_phase", "training_complete"}:
        try:
            from datasets import Dataset
            Dataset.cleanup_cache_files()
        except Exception:
            pass
    logger.debug("Memory cleanup completed: %s", context)


def get_cache_path() -> str:
    try:
        cache_dir = "/scratch/project_465002167/dataset/cache"
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "analysis_cache.json")
        logger.info("Using cluster cache: %s", path)
        return path
    except Exception as exc:
        cache_dir = "./cache"
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "analysis_cache.json")
        logger.info("Cluster cache unavailable (%s), using local cache: %s", exc, path)
        return path


def load_analysis_cache(cache_path: str) -> Dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded analysis cache with %d file mappings", len(data.get("file_mappings", {})))
    return data


class UniversalFilenameParser:
    @staticmethod
    def parse_filename(filename: str, file_path: Optional[str] = None) -> Tuple[str, Optional[str], str, Dict]:
        import re
        try:
            name = str(filename)
            if name.endswith(".parquet"):
                name = name[:-8]
            if name.endswith(".partial"):
                name = name[:-8]
            tokens = [t for t in re.split(r"[^A-Za-z0-9]+", name) if t]
            dataset_name = tokens[0] if tokens else "unknown"
            if len(tokens) >= 2:
                dataset_name = f"{tokens[0]}/{tokens[1]}"
            subset = None
            language_category = "unknown"
            metadata: Dict[str, str] = {}
            digits = [t for t in tokens if t.isdigit()]
            if digits:
                metadata["file_number"] = digits[-1]
            return dataset_name, subset, language_category, metadata
        except Exception as exc:
            logger.error("parse_filename failed: %s", exc)
            return "unknown", None, "unknown", {}


class _EnsembleLanguageDetector:
    def __init__(self) -> None:
        self._cld3_identifier = None
        self._langid = None
        self._ft_model = None
        self._langdetect = None
        try:
            import pycld3  # type: ignore
            self._cld3_identifier = pycld3.NNetLanguageIdentifier(min_num_bytes=30, max_num_bytes=2000)
            logger.info("pycld3 enabled")
        except Exception:
            logger.info("pycld3 not available")
        try:
            import langid  # type: ignore
            self._langid = langid
            self._langid.set_languages(self._langid.langid.DEFAULT_LANGS)
            logger.info("langid enabled")
        except Exception:
            logger.info("langid not available")
        try:
            import fasttext  # type: ignore
            model_path = os.environ.get("FASTTEXT_LID_PATH", "").strip()
            if model_path and os.path.isfile(model_path):
                self._ft_model = fasttext.load_model(model_path)
                logger.info("fastText LID model loaded")
            elif model_path:
                logger.warning("FASTTEXT_LID_PATH set but file not found: %s", model_path)
        except Exception:
            logger.info("fastText not available")
        try:
            from langdetect import detect, DetectorFactory  # type: ignore  # noqa: F401
            self._langdetect = True
            from langdetect import DetectorFactory as _DF  # type: ignore
            _DF.seed = 0
            logger.info("langdetect enabled")
        except Exception:
            logger.info("langdetect not available")
        if not any([self._cld3_identifier, self._langid, self._ft_model, self._langdetect]):
            logger.warning("No language detectors available. Results may degrade.")

    @staticmethod
    def _normalize(code: Optional[str]) -> str:
        if not code:
            return "und"
        c = code.lower()
        remap = {"iw": "he", "ji": "yi", "in": "id", "cmn": "zh", "zho": "zh", "eng": "en"}
        if c.startswith("__label__"):
            c = c.replace("__label__", "")
        c = remap.get(c, c)
        if len(c) > 2:
            c = c[:2]
        return c

    def _detect_cld3(self, text: str) -> Optional[str]:
        try:
            if not self._cld3_identifier:
                return None
            res = self._cld3_identifier.FindLanguage(text)
            if res and getattr(res, "is_reliable", False) and getattr(res, "language", None):
                return self._normalize(res.language)
        except Exception:
            return None
        return None

    def _detect_langid(self, text: str) -> Optional[str]:
        try:
            if not self._langid:
                return None
            code, _ = self._langid.classify(text)
            return self._normalize(code)
        except Exception:
            return None

    def _detect_fasttext(self, text: str) -> Optional[str]:
        try:
            if not self._ft_model:
                return None
            preds = self._ft_model.predict(text.replace("\n", " ")[:2000], k=1)
            if preds and preds[0]:
                return self._normalize(preds[0][0])
        except Exception:
            return None
        return None

    def _detect_langdetect(self, text: str) -> Optional[str]:
        try:
            if not self._langdetect:
                return None
            from langdetect import detect  # type: ignore
            return self._normalize(detect(text))
        except Exception:
            return None

    def detect(self, text: str) -> str:
        votes: List[str] = []
        for fn in (self._detect_cld3, self._detect_langid, self._detect_fasttext, self._detect_langdetect):
            code = fn(text)
            if code and code != "und":
                votes.append(code)
        if not votes:
            return "und"
        return Counter(votes).most_common(1)[0][0]


class StreamingDatasetAnalyzer:
    def __init__(self) -> None:
        self.filename_parser = UniversalFilenameParser()
        self.file_mappings: Dict[str, str] = {}
        self.language_stats: Dict[str, int] = defaultdict(int)
        self.datasets_found: set[str] = set()
        self.processed_count = 0
        self.warnings: List[str] = []
        self.file_usability_estimates: Dict[str, Dict] = {}
        self._detector = _EnsembleLanguageDetector()

    def _reset_analysis_state(self) -> None:
        self.file_mappings = {}
        self.language_stats = defaultdict(int)
        self.datasets_found = set()
        self.processed_count = 0
        self.warnings = []
        self.file_usability_estimates = {}

    @staticmethod
    def _first_text_column(schema) -> Optional[str]:
        for field in schema:
            name = field.name.lower()
            logical = str(getattr(field, "logical_type", ""))
            if "string" in logical.lower() or any(k in name for k in ("text", "content", "message", "body", "document")):
                return field.name
        return None

    def micro_sample_content(self, parquet_file, max_bytes: int = 8192) -> str:
        try:
            col = self._first_text_column(parquet_file.schema)
            if not col:
                return ""
            total_rows = parquet_file.metadata.num_rows or 0
            rows_to_read = min(32, total_rows) if total_rows else 16
            row_groups = min(parquet_file.num_row_groups, 4)
            parts: List[str] = []
            acc = 0
            for rg in range(row_groups):
                try:
                    table = parquet_file.read_row_group(rg, columns=[col])
                except Exception:
                    continue
                if table.num_rows == 0:
                    continue
                arr = table.column(0)
                for val in arr.to_pylist()[: rows_to_read]:
                    if val is None:
                        continue
                    s = str(val).strip()
                    if not s:
                        continue
                    parts.append(s)
                    acc += len(s)
                    if acc >= max_bytes:
                        break
                if acc >= max_bytes:
                    break
            text = "\n".join(parts)
            if len(text) > max_bytes:
                text = text[:max_bytes]
            return text
        except Exception:
            return ""
        finally:
            try:
                import pyarrow as pa
                pa.default_memory_pool().release_unused()
            except Exception:
                pass

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        if not text or len(text) < 40:
            return False
        patterns = (
            "def ",
            "class ",
            "import ",
            "#include",
            "public ",
            "private ",
            "protected ",
            "const ",
            "var ",
            "let ",
            "=>",
            "console.log",
            "print(",
            "println(",
            "system.out",
            "printf(",
            "cout <<",
            "cin >>",
            "<?php",
            "#!/",
            "using ",
            "namespace ",
        )
        t = text.lower()
        matches = sum(1 for p in patterns if p in t)
        braces = t.count("{") + t.count("}")
        semis = t.count(";")
        return matches >= 3 or (braces >= 4 and semis >= 4)

    def detect_language(self, sample_text: str) -> str:
        if sample_text and self._looks_like_code(sample_text):
            return "code"
        if sample_text and len(sample_text.strip()) >= 30:
            return self._detector.detect(sample_text[:2000])
        return "und"

    def process_single_file_streaming(self, file_path: Path) -> None:
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            self.warnings.append(f"PyArrow unavailable: {exc}")
            return
        parquet_file = None
        try:
            dataset_name, _, _, _ = self.filename_parser.parse_filename(file_path.name)
            if dataset_name:
                self.datasets_found.add(dataset_name)
            parquet_file = pq.ParquetFile(str(file_path))
            file_size = file_path.stat().st_size
            sample_text = self.micro_sample_content(parquet_file, max_bytes=8192)
            language = self.detect_language(sample_text)
            self.file_mappings[str(file_path.resolve())] = language
            self.language_stats[language] += file_size
            self.processed_count += 1
            try:
                col = self._first_text_column(parquet_file.schema)
                total_rows = int(parquet_file.metadata.num_rows or 0)
                if col and total_rows > 0 and parquet_file.num_row_groups > 0:
                    table0 = parquet_file.read_row_group(0, columns=[col])
                    vals = table0.column(0).to_pylist()
                    non_empty = sum(1 for x in vals if x and str(x).strip())
                    usable_ratio = non_empty / max(len(vals), 1)
                    est_usable = int(usable_ratio * total_rows)
                    self.file_usability_estimates[str(file_path.resolve())] = {
                        "usable_ratio": round(usable_ratio, 3),
                        "estimated_usable": est_usable,
                        "total_records_estimate": total_rows,
                    }
            except Exception:
                pass
        except Exception as exc:
            self.warnings.append(f"Failed to analyze {file_path.name}: {str(exc)[:200]}")
            self.file_mappings[str(file_path.resolve())] = "und"
        finally:
            if parquet_file is not None:
                try:
                    parquet_file.close()
                except Exception:
                    pass
                del parquet_file

    def _sample_files_default(
        self,
        files_by_dataset: Dict[str, List[Path]],
        total_files_count: Optional[int] = None,
        total_dataset_gb: Optional[float] = None,
    ) -> List[Path]:
        import math
        if total_files_count is None:
            total_files_count = sum(len(files) for files in files_by_dataset.values())
        if total_dataset_gb:
            discovery_percentage = 0.25 + (math.log10(max(total_dataset_gb, 1e-6)) * 0.08)
            discovery_percentage = max(0.25, min(discovery_percentage, 0.60))
        else:
            discovery_percentage = 0.40
        target_sample_size = int(total_files_count * discovery_percentage)
        sampling_ratio = min(1.0, target_sample_size / max(total_files_count, 1))
        logger.info(
            "Dataset-aware discovery: %.0fGB -> %.1f%% = %d files",
            total_dataset_gb or 0.0,
            discovery_percentage * 100.0,
            target_sample_size,
        )
        sampled_files: List[Path] = []
        for dataset_name, dataset_files in files_by_dataset.items():
            sample_count = max(1, int(len(dataset_files) * sampling_ratio))
            picks = dataset_files.copy()
            random.shuffle(picks)
            dataset_sample = picks[:sample_count]
            sampled_files.extend(dataset_sample)
            logger.info(
                "%s: %d/%d files sampled (%.1f%%)",
                dataset_name,
                len(dataset_sample),
                len(dataset_files),
                (len(dataset_sample) / max(len(dataset_files), 1)) * 100.0,
            )
        random.shuffle(sampled_files)
        logger.info(
            "Total discovery sample: %d files (%.1f%%)",
            len(sampled_files),
            (len(sampled_files) / max(total_files_count, 1)) * 100.0,
        )
        return sampled_files

    def _sample_files_by_language_proportion(
        self,
        all_files: List[Path],
        full_file_mappings: Dict[str, str],
        file_sizes_cache: Dict[str, int],
        full_language_stats: Dict[str, int],
        target_data_gb: float,
    ) -> List[Path]:
        target_bytes = int(target_data_gb * (1024**3))
        total_language_bytes = sum(full_language_stats.values())
        if total_language_bytes <= 0:
            return []
        language_targets: Dict[str, int] = {}
        logger.info("Language targets for proportional sampling:")
        for lang, bytes_count in full_language_stats.items():
            if bytes_count <= 0:
                continue
            proportion = bytes_count / total_language_bytes
            language_targets[lang] = int(target_bytes * proportion)
            logger.info("%s: %.3fGB (%.1f%%)", lang, language_targets[lang] / (1024**3), proportion * 100.0)
        idx = {str(f.resolve()): f for f in all_files}
        files_by_lang: Dict[str, List[Path]] = defaultdict(list)
        for fpath, lang in full_file_mappings.items():
            if fpath in idx:
                files_by_lang[lang].append(idx[fpath])
        selected: List[Path] = []
        current_bytes: Dict[str, int] = {k: 0 for k in language_targets.keys()}
        for lang, target in language_targets.items():
            picks = files_by_lang.get(lang, []).copy()
            random.shuffle(picks)
            chosen = 0
            for f in picks:
                if current_bytes[lang] >= target:
                    break
                fsize = file_sizes_cache.get(str(f.resolve()), 0)
                selected.append(f)
                current_bytes[lang] += fsize
                chosen += 1
            logger.info("%s: selected %d files (%.3fGB)", lang, chosen, current_bytes[lang] / (1024**3))
        total_selected_gb = sum(current_bytes.values()) / (1024**3)
        logger.info("Proportional sampling complete: %d files, %.3fGB", len(selected), total_selected_gb)
        return selected

    def analyze_available_data_streaming(self, datasets_dir: str, target_data_gb: Optional[float] = None) -> Dict:
        start_mem = get_memory_usage_percent()
        logger.info("STREAMING ANALYSIS START - Memory: %s", format_memory_log(start_mem))
        base = Path(datasets_dir)
        if not base.exists():
            return {"error": f"Datasets directory not found: {datasets_dir}"}
        parquet_files = list(base.glob("*.parquet"))
        if not parquet_files:
            return {"error": f"No parquet files found in: {datasets_dir}"}
        total_files = len(parquet_files)
        if target_data_gb:
            logger.info("Found %d parquet files, targeting %.1fGB", total_files, target_data_gb)
        else:
            logger.info("Found %d parquet files (full dataset mode)", total_files)
        files_by_dataset: Dict[str, List[Path]] = defaultdict(list)
        file_sizes_cache: Dict[str, int] = {}
        for f in parquet_files:
            dataset_name, _, _, _ = self.filename_parser.parse_filename(f.name)
            files_by_dataset[dataset_name].append(f)
            try:
                file_sizes_cache[str(f.resolve())] = f.stat().st_size
            except Exception:
                file_sizes_cache[str(f.resolve())] = 0
        logger.info("Phase 1: Full dataset discovery")
        total_full_size_gb = sum(file_sizes_cache.values()) / (1024**3)
        discovery_files = self._sample_files_default(files_by_dataset, total_files, total_full_size_gb)
        logger.info("Discovery sampling: %d/%d files", len(discovery_files), total_files)
        self._reset_analysis_state()
        for i, path in enumerate(discovery_files, start=1):
            self.process_single_file_streaming(path)
            if i % 50 == 0:
                aggressive_memory_cleanup("discovery_milestone")
        full_lang_stats = dict(self.language_stats)
        full_file_map = dict(self.file_mappings)
        logger.info("Discovery complete. Total size: %.2fGB across %d files", total_full_size_gb, total_files)
        denom = float(sum(full_lang_stats.values()) or 1)
        for lang, bytes_count in sorted(full_lang_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info("%s: %.2fGB (%.1f%%)", lang, bytes_count / (1024**3), (bytes_count / denom) * 100.0)
        if target_data_gb:
            logger.info("Phase 2: Proportional sampling for %.1fGB target", target_data_gb)
            self._reset_analysis_state()
            target_files = self._sample_files_by_language_proportion(
                parquet_files, full_file_map, file_sizes_cache, full_lang_stats, target_data_gb
            )
            final_files = target_files
            total_sampled_size_gb = sum(file_sizes_cache.get(str(f.resolve()), 0) for f in final_files) / (1024**3)
            logger.info("Target sampling: %d/%d files (%.2fGB/%.1fGB)", len(final_files), total_files, total_sampled_size_gb, target_data_gb)
        else:
            final_files = parquet_files
            total_sampled_size_gb = sum(file_sizes_cache.get(str(f.resolve()), 0) for f in final_files) / (1024**3)
            logger.info("Using ALL files: %d/%d files (%.2fGB)", len(final_files), total_files, total_sampled_size_gb)
        for i, path in enumerate(final_files, start=1):
            self.process_single_file_streaming(path)
            if i % 50 == 0:
                cur_mem = get_memory_usage_percent()
                logger.info("PROGRESS: %d/%d files, Memory: %s", i, len(final_files), format_memory_log(cur_mem))
                aggressive_memory_cleanup("streaming_milestone")
        peak_mem = get_memory_usage_percent()
        logger.info("STREAMING ANALYSIS PEAK - Memory: %s", format_memory_log(peak_mem))
        results = {
            "full_dataset_stats": {
                "total_size_gb": total_full_size_gb,
                "total_files": total_files,
                "language_statistics": full_lang_stats,
                "file_mappings": full_file_map,
                "datasets_found": list(self.datasets_found),
                "analysis_method": "two_phase_streaming_detection",
            },
            "target_analysis": {
                "target_data_gb": target_data_gb,
                "actual_analyzed_gb": total_sampled_size_gb,
                "files_processed": self.processed_count,
                "selected_file_mappings": dict(self.file_mappings),
                "sampled_language_statistics": dict(self.language_stats),
                "maintains_proportions": target_data_gb is not None,
                "sampling_ratio": (total_sampled_size_gb / total_full_size_gb) if total_full_size_gb > 0 else 0.0,
            },
            "total_size_gb": total_full_size_gb,
            "file_mappings": dict(self.file_mappings),
            "language_statistics": full_lang_stats,
            "datasets_found": list(self.datasets_found),
            "warnings": list(self.warnings),
            "analysis_method": "two_phase_streaming_detection",
            "target_data_gb": target_data_gb,
            "actual_analyzed_gb": total_sampled_size_gb,
            "sampling_ratio": (total_sampled_size_gb / total_full_size_gb) if total_full_size_gb > 0 else 0.0,
            "maintains_proportions": bool(target_data_gb),
            "resampled_from_cache": False,
            "file_usability_estimates": dict(self.file_usability_estimates),
        }
        aggressive_memory_cleanup("analysis_phase")
        return results

    @staticmethod
    def export_to_cache(results: Dict, cache_path: str) -> str:
        from datetime import datetime
        if "full_dataset_stats" in results:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "full_dataset_stats": results["full_dataset_stats"],
                "target_analysis": results["target_analysis"],
                "total_files_processed": results["target_analysis"]["files_processed"],
                "file_mappings": results["target_analysis"]["selected_file_mappings"],
                "language_statistics": results["full_dataset_stats"]["language_statistics"],
                "datasets_found": results["full_dataset_stats"]["datasets_found"],
                "warnings": results.get("warnings", []),
                "analysis_method": results["full_dataset_stats"]["analysis_method"],
                "total_size_gb": results["full_dataset_stats"]["total_size_gb"],
                "target_data_gb": results["target_analysis"].get("target_data_gb"),
                "actual_analyzed_gb": results["target_analysis"].get("actual_analyzed_gb"),
                "sampling_ratio": results.get("sampling_ratio"),
                "maintains_proportions": results.get("maintains_proportions", False),
                "resampled_from_cache": results.get("resampled_from_cache", False),
                "file_usability_estimates": results.get("file_usability_estimates", {}),
            }
        else:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "total_files_processed": results.get("total_files_processed", 0),
                "file_mappings": results.get("file_mappings", {}),
                "language_statistics": results.get("language_statistics", {}),
                "datasets_found": results.get("datasets_found", []),
                "warnings": results.get("warnings", []),
                "analysis_method": results.get("analysis_method", "streaming"),
                "total_size_gb": results.get("total_size_gb", 0.0),
                "target_data_gb": results.get("target_data_gb"),
                "actual_analyzed_gb": results.get("actual_analyzed_gb"),
                "sampling_ratio": results.get("sampling_ratio"),
                "maintains_proportions": results.get("maintains_proportions", False),
                "resampled_from_cache": results.get("resampled_from_cache", False),
                "file_usability_estimates": results.get("file_usability_estimates", {}),
            }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Analysis cache exported to %s", cache_path)
        return cache_path


def run_streaming_analysis_phase(cache_path: str, datasets_dir: str, target_data_gb: Optional[float] = None) -> Optional[str]:
    start_mem = get_memory_usage_percent()
    logger.info("STREAMING ANALYSIS PHASE START - Memory: %s", format_memory_log(start_mem))
    analyzer = StreamingDatasetAnalyzer()
    results = analyzer.analyze_available_data_streaming(datasets_dir, target_data_gb=target_data_gb)
    if "error" in results:
        logger.error("Analysis failed: %s", results["error"])
        return None
    analyzer.export_to_cache(results, cache_path)
    peak_mem = get_memory_usage_percent()
    logger.info("ANALYSIS PEAK MEMORY - Memory: %s", format_memory_log(peak_mem))
    logger.info("Releasing analysis memory")
    del analyzer
    del results
    aggressive_memory_cleanup("analysis_complete")
    final_mem = get_memory_usage_percent()
    freed = get_used_memory_gb(peak_mem) - get_used_memory_gb(final_mem)
    logger.info("ANALYSIS MEMORY RELEASED - Memory: %s", format_memory_log(final_mem))
    logger.info("TOTAL MEMORY FREED: %.1fGB", freed)
    increase = final_mem - start_mem
    if increase > 5:
        logger.warning("Potential memory leak in analysis")
        logger.warning("Started: %.1f%%, Now: %.1f%% (Δ %.1f%%, ≈ %.1fGB)", start_mem, final_mem, increase, get_used_memory_gb(increase))
    else:
        logger.info("Memory clean - increase: %.1f%%", increase)
    return cache_path


def emit_manifest_from_cache(cache: Dict, manifest_path: str) -> Tuple[int, int]:
    mappings: Dict[str, str] = {}
    top = cache.get("file_mappings") or {}
    if isinstance(top, dict):
        mappings.update(top)
    ta = cache.get("target_analysis") or {}
    sel = ta.get("selected_file_mappings") or {}
    if isinstance(sel, dict):
        for k, v in sel.items():
            if k not in mappings:
                mappings[k] = v
    if not mappings:
        logger.error("No file_mappings found in cache. Cannot emit manifest.")
        return 0, 0
    rows: List[Tuple[str, str]] = []
    skipped = 0
    for abs_path, lang in sorted(mappings.items()):
        try:
            p = Path(abs_path).resolve()
        except Exception:
            p = Path(abs_path)
        if not p.is_file():
            logger.warning("Missing file skipped: %s", str(p))
            skipped += 1
            continue
        rows.append((str(p), str(lang or "und")))
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    import csv
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "language"])
        w.writerows(rows)
    import hashlib
    digest = hashlib.blake2b("".join(f"{p}:{l}" for p, l in rows).encode("utf-8"), digest_size=16).hexdigest()
    logger.info("Manifest written: %s | rows=%d | skipped=%d | blake2b16=%s", manifest_path, len(rows), skipped, digest)
    return len(rows), skipped


@click.command()
@click.option(
    "--datasets-dir",
    default="./datasets",
    show_default=True,
    help="Directory containing parquet files for analysis.",
)
@click.option(
    "--target-data-size",
    default=None,
    help="Target total data size (e.g., 500MB, 1.5GB, 2TB). If not set, all data is analyzed.",
)
@click.option(
    "--manifest",
    default=None,
    help="Optional path to write a manifest CSV (file_path,language) directly after analysis.",
)
def main(datasets_dir: str, target_data_size: Optional[str], manifest: Optional[str]) -> None:
    logger.info("=" * 47)
    logger.info("ANALYSIS MODE - Dataset composition analysis")
    logger.info("=" * 47)
    target_data_gb: Optional[float] = None
    if target_data_size:
        try:
            target_data_gb = parse_data_size(target_data_size)
            logger.info("Target data size: %s (%.3fGB)", target_data_size, target_data_gb)
        except ValueError as exc:
            logger.error("Invalid target data size: %s", exc)
            sys.exit(1)
    cache_path = get_cache_path()
    try:
        result_path = run_streaming_analysis_phase(cache_path, datasets_dir, target_data_gb)
    except KeyboardInterrupt:
        logger.error("Interrupted by user. Exiting gracefully.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error during analysis: %s", exc)
        sys.exit(1)
    if not result_path:
        logger.error("Streaming analysis failed")
        sys.exit(1)
    try:
        cached = load_analysis_cache(cache_path)
    except KeyboardInterrupt:
        logger.error("Interrupted by user during cache load. Exiting gracefully.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Failed to load analysis cache: %s", exc)
        sys.exit(1)
    logger.info("DEBUG: target_data_gb in cache = %s", cached.get("target_data_gb"))
    logger.info("DEBUG: actual_analyzed_gb in cache = %s", cached.get("actual_analyzed_gb"))
    analysis = {
        "total_size_gb": cached.get("total_size_gb", 0.0),
        "languages": {},
        "datasets": cached.get("datasets_found", []),
        "warnings": cached.get("warnings", []),
    }
    full_stats = cached.get("full_dataset_stats", {})
    lang_stats: Dict[str, int] = full_stats.get("language_statistics", {}) or cached.get("language_statistics", {})
    total_bytes = float(sum(lang_stats.values()))
    total_gb = float(cached.get("total_size_gb", 0.0))
    for lang, bytes_count in lang_stats.items():
        if bytes_count <= 0:
            continue
        pct = (bytes_count / total_bytes) * 100.0 if total_bytes > 0 else 0.0
        display_size_gb = (total_gb * pct) / 100.0
        file_count = sum(1 for _, l in cached.get("file_mappings", {}).items() if l == lang)
        analysis["languages"][lang] = {
            "size_gb": display_size_gb,
            "percentage": pct,
            "datasets": [],
            "file_count": file_count,
        }
    logger.info("DATASET COMPOSITION ANALYSIS")
    logger.info("=" * 80)
    tgt_gb = cached.get("target_data_gb")
    if tgt_gb:
        actual_gb = cached.get("actual_analyzed_gb", analysis["total_size_gb"])
        sampling_ratio = (cached.get("sampling_ratio", 0.0) or 0.0) * 100.0
        logger.info("Target Size: %.1fGB", tgt_gb)
        logger.info("Analyzed: %.2fGB (%.1f%% of total dataset)", actual_gb, sampling_ratio)
    else:
        logger.info("Analyzed Data: %.2fGB (full dataset mode)", analysis["total_size_gb"])
    logger.info("Datasets Found: %d", len(analysis["datasets"]))
    logger.info("Languages Detected: %d", len(analysis["languages"]))
    logger.info("")
    logger.info("LANGUAGE BREAKDOWN")
    logger.info("-" * 50)
    for language, info in sorted(analysis["languages"].items(), key=lambda x: x[1]["size_gb"], reverse=True):
        logger.info("%s:", language.upper())
        logger.info("   Size: %.2fGB (%.1f%%)", info["size_gb"], info["percentage"])
        logger.info("   Files: %d", info["file_count"])
        logger.info("")
    if analysis["warnings"]:
        logger.info("WARNINGS:")
        for w in analysis["warnings"]:
            logger.info(" - %s", w)
    if manifest:
        try:
            rows_written, skipped = emit_manifest_from_cache(cached, manifest)
            if rows_written <= 0:
                logger.error("No rows written to manifest. Ensure cache contains absolute paths.")
                sys.exit(1)
            logger.info("Manifest emitted successfully: %s (rows=%d, skipped=%d)", manifest, rows_written, skipped)
        except KeyboardInterrupt:
            logger.error("Interrupted by user during manifest emission. Exiting gracefully.")
            sys.exit(130)
        except Exception as exc:
            logger.error("Failed to emit manifest: %s", exc)
            sys.exit(1)
    logger.info("Analysis complete. Use these ratios as input for downstream processes.")


if __name__ == "__main__":
    main()
