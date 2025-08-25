"""
Offline character-level multilingual BPE training optimized for European, Nordic, and English languages.

Features:
- Comprehensive European/Nordic Unicode coverage
- Optimized initial vocabulary for target languages
- Adaptive batch sizing and memory management
- Pair selection for optimal compression
- High-quality merge selection for excellent fertility
"""

import os
import sys
import gc
import io
import gzip
import json
import time
import heapq
import ctypes
import psutil
import signal
import shutil
import logging
import argparse
import tempfile
import traceback
import resource
import subprocess
import unicodedata
from typing import Dict, List, Tuple, Optional, Iterable, Any, Set
import threading
import queue
import hashlib


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with proper formatting and handlers."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.root.handlers = []
    logging.root.setLevel(lvl)
    
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logging.root.addHandler(ch)
    
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        except Exception:
            pass
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logging.root.addHandler(fh)
    
    return logging.getLogger("train")


def human_int(n: int) -> str:
    """Format integer with thousands separators."""
    return f"{n:,}"


def memory_percent() -> float:
    """Get current memory usage percentage."""
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def gated_cleanup():
    """Perform garbage collection and memory trimming."""
    try:
        gc.collect()
    except Exception:
        pass
    
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def safe_fanin(requested: int) -> int:
    """Calculate safe file descriptor limit for merging."""
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        cap = max(64, int(soft * 0.6))
        return max(2, min(requested, cap))
    except Exception:
        return min(512, max(2, requested))


def blake2b16(s: str) -> str:
    """Generate 16-char blake2b hash for string."""
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def install_signal_handler(stop_event: threading.Event, logger: logging.Logger):
    """Install signal handlers for graceful shutdown."""
    def handler(signum, frame):
        logger.warning(f"Received signal {signum}, requesting shutdown...")
        stop_event.set()
    
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


def safe_concatenate(str_a: str, str_b: str) -> Optional[str]:
    """Safely concatenate two Unicode strings with validation."""
    try:
        result = str_a + str_b
        # Validate the result is proper Unicode
        result.encode('utf-8').decode('utf-8')
        # Additional validation: check for combining characters at boundaries
        if len(result) > 0 and unicodedata.category(result[0]) == 'Mn':
            return None  # Don't start with combining mark
        return result
    except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
        return None


def get_optimal_batch_size(available_memory_gb: float, target_vocab_size: int) -> int:
    """Calculate optimal batch size based on available memory and target vocabulary."""
    base_batch = min(16000, target_vocab_size // 8)  # Don't exceed 1/8 of target vocab
    
    if available_memory_gb >= 128:
        return min(20000, base_batch * 2)
    elif available_memory_gb >= 64:
        return min(16000, base_batch)
    elif available_memory_gb >= 32:
        return min(12000, int(base_batch * 0.8))
    elif available_memory_gb >= 16:
        return min(8000, int(base_batch * 0.6))
    else:
        return min(4000, int(base_batch * 0.4))


def calculate_adaptive_min_frequency(total_sequences: int, base_min_frequency: int) -> int:
    """Calculate adaptive minimum frequency based on corpus size."""
    if total_sequences > 100_000_000:  # 100M+ sequences
        return max(10, base_min_frequency * 2)
    elif total_sequences > 10_000_000:  # 10M+ sequences
        return max(5, int(base_min_frequency * 1.5))
    elif total_sequences > 1_000_000:  # 1M+ sequences
        return max(3, base_min_frequency)
    else:
        return max(2, base_min_frequency)


def init_european_nordic_vocab() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Initialize vocabulary optimized for European, Nordic, and English languages."""
    token_str: Dict[int, str] = {}
    token2id: Dict[str, int] = {}
    
    # Comprehensive European/Nordic Unicode ranges
    char_ranges = [
        # Core Latin and European extensions
        (32, 127),       # Basic Latin (English)
        (160, 256),      # Latin-1 Supplement (Western European)
        (256, 384),      # Latin Extended-A (Central/Eastern European)
        (384, 592),      # Latin Extended-B (more European languages)
        (7680, 7936),    # Latin Extended Additional (Vietnamese, etc.)
        
        # European scripts
        (880, 1024),     # Greek and Coptic
        (1024, 1280),    # Cyrillic (Russian, Bulgarian, Serbian, etc.)
        (1328, 1424),    # Armenian
        
        # Punctuation and symbols
        (8192, 8304),    # General Punctuation (em-dash, quotes, etc.)
        (8352, 8400),    # Currency Symbols (€, £, ¥, etc.)
        (8448, 8528),    # Letterlike Symbols
        (8704, 8960),    # Mathematical Operators
        (9216, 9280),    # Control Pictures
        (9312, 9472),    # Enclosed Alphanumerics
        (9472, 9600),    # Box Drawing
        (9600, 9632),    # Block Elements
        
        # Additional European ranges
        (11360, 11392),  # Latin Extended-D
    ]
    
    next_id = 0
    
    # Add characters from European ranges
    for start, end in char_ranges:
        for cp in range(start, end):
            try:
                char = chr(cp)
                cat = unicodedata.category(char)
                # Include printable characters, skip control characters
                if cat[0] not in ['C'] or cat in ['Cf']:  # Include format chars like soft hyphen
                    token_str[next_id] = char
                    token2id[char] = next_id
                    next_id += 1
                    if next_id >= 12000:  # Generous limit for European languages
                        break
            except ValueError:
                continue
        if next_id >= 12000:
            break
    
    # Ensure critical European/Nordic characters are included
    critical_chars = [
        # Nordic/Scandinavian essentials
        'æ', 'ø', 'å', 'Æ', 'Ø', 'Å',           # Norwegian, Danish
        'ä', 'ö', 'Ä', 'Ö',                     # Swedish, Finnish, German
        'ß',                                     # German eszett
        'þ', 'ð', 'Þ', 'Ð',                    # Icelandic
        
        # Central/Eastern European
        'ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż',  # Polish
        'Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż',
        'č', 'ď', 'ě', 'ň', 'ř', 'š', 'ť', 'ů', 'ý', 'ž',  # Czech
        'Č', 'Ď', 'Ě', 'Ň', 'Ř', 'Š', 'Ť', 'Ů', 'Ý', 'Ž',
        'á', 'é', 'í', 'ó', 'ú', 'ý', 'À', 'È', 'Ì', 'Ò', 'Ù',  # Accented vowels
        'Á', 'É', 'Í', 'Ó', 'Ú', 'Ý',
        
        # Western European
        'ñ', 'Ñ',                               # Spanish
        'ç', 'Ç',                               # French, Portuguese
        'ü', 'Ü',                               # German, etc.
        'ë', 'ï', 'Ë', 'Ï',                    # French, Dutch
        'ğ', 'ı', 'ş', 'Ğ', 'İ', 'Ş',          # Turkish
        
        # Cyrillic essentials
        'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й',
        'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
        'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
        'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й',
        'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
        'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
        
        # Greek essentials
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ',
        'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ',
        'φ', 'χ', 'ψ', 'ω', 'ς',
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ',
        'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ',
        'Φ', 'Χ', 'Ψ', 'Ω',
        
        # European punctuation and symbols
        '€', '£', '¥', '§', '©', '®', '°', '±', '²', '³', '¼', '½', '¾',
        '×', '÷', '‚', '„', '"', '"', ''', ''', '‹', '›', '«', '»',
        '–', '—', '•', '…', '‰', '‱', '′', '″', '‴', '‵', '‶', '‷',
        '⁰', '¹', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁺', '⁻', '⁼', '⁽', '⁾',
        
        # Mathematical and technical
        '∀', '∂', '∃', '∅', '∇', '∈', '∉', '∋', '∏', '∑',
        '−', '∓', '∗', '∘', '√', '∝', '∞', '∟', '∠', '∡',
        '∢', '∣', '∤', '∥', '∦', '∧', '∨', '∩', '∪', '∫',
        '∴', '∵', '∶', '∷', '∸', '∹', '∺', '∻', '∼', '∽',
        '∾', '∿', '≀', '≁', '≂', '≃', '≄', '≅', '≆', '≇',
        '≈', '≉', '≊', '≋', '≌', '≍', '≎', '≏', '≐', '≑',
        '≒', '≓', '≔', '≕', '≖', '≗', '≘', '≙', '≚', '≛',
        '≜', '≝', '≞', '≟', '≠', '≡', '≢', '≣', '≤', '≥',
    ]
    
    # Add critical characters if not already present
    for char in critical_chars:
        if char not in token2id and next_id < 12000:
            token_str[next_id] = char
            token2id[char] = next_id
            next_id += 1
    
    return token_str, token2id


class CountsReader:
    """Robust counts file reader supporting multiple formats."""
    
    def __init__(self, path: str, counts_format: str, logger: logging.Logger):
        self.path = path
        self.format = counts_format.lower().strip() if counts_format else "auto"
        self._proc: Optional[subprocess.Popen] = None
        self._fh: Optional[io.TextIOBase] = None
        self.logger = logger

    def _detect_format(self, path: str) -> str:
        """Auto-detect file format from extension."""
        if self.format != "auto":
            return self.format
        
        p = path.lower()
        if p.endswith((".jsonl", ".json")):
            return "jsonl"
        if p.endswith((".jsonl.gz", ".json.gz")):
            return "jsonl.gz"
        if p.endswith((".tsv", ".csv")):
            return "tsv"
        if p.endswith((".tsv.gz", ".csv.gz")):
            return "tsv.gz"
        
        return "jsonl"  # Default fallback

    def __enter__(self):
        fmt = self._detect_format(self.path)
        gz = fmt.endswith(".gz")
        
        if self.path in ("-", "/dev/stdin") or self.path.startswith("/proc/self/fd/"):
            self._fh = open(self.path, "rt", encoding="utf-8", errors="strict")
            self.logger.info(f"Counts reader: format={'tsv' if 'tsv' in fmt else 'jsonl'} | source={self.path}")
            return self
        
        if gz:
            # Try pigz for multi-core decompression
            pigz = shutil.which("pigz")
            if pigz:
                try:
                    self._proc = subprocess.Popen(
                        [pigz, "-dc", self.path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=False
                    )
                    if self._proc.stdout is None:
                        raise RuntimeError("Failed to open pigz stdout")
                    self._fh = io.TextIOWrapper(self._proc.stdout, encoding="utf-8", errors="strict")
                    self.logger.info(f"Counts reader: format={'tsv' if 'tsv' in fmt else 'jsonl'} | source={self.path} | via pigz")
                except Exception as e:
                    self.logger.warning(f"pigz failed: {e}, falling back to gzip")
                    if self._proc:
                        try:
                            self._proc.kill()
                        except Exception:
                            pass
                        self._proc = None
                    self._fh = io.TextIOWrapper(gzip.open(self.path, "rb"), encoding="utf-8", errors="strict")
                    self.logger.info(f"Counts reader: format={'tsv' if 'tsv' in fmt else 'jsonl'} | source={self.path} | via gzip")
            else:
                self._fh = io.TextIOWrapper(gzip.open(self.path, "rb"), encoding="utf-8", errors="strict")
                self.logger.info(f"Counts reader: format={'tsv' if 'tsv' in fmt else 'jsonl'} | source={self.path} | via gzip")
        else:
            self._fh = open(self.path, "rt", encoding="utf-8", errors="strict")
            self.logger.info(f"Counts reader: format={'tsv' if 'tsv' in fmt else 'jsonl'} | source={self.path}")
        
        self.format = "tsv" if "tsv" in fmt else "jsonl"
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
        finally:
            if self._proc:
                try:
                    if self._proc.stdout:
                        self._proc.stdout.close()
                except Exception:
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=1.0)
                except Exception:
                    try:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)
                    except Exception:
                        pass

    def iter_records(self, stop: threading.Event):
        """Yield (List[int] codepoint_seq, int count) tuples."""
        assert self._fh is not None
        fh = self._fh
        malformed = 0
        line_no = 0
        
        if self.format == "tsv":
            for line in fh:
                if stop.is_set():
                    break
                line_no += 1
                if not line or line == "\n":
                    continue
                try:
                    k, c = line.rstrip("\n").split("\t", 1)
                    seq = [int(x) for x in k.split() if x.strip()]
                    cnt = int(c)
                    if seq and cnt > 0:
                        yield seq, cnt
                except Exception as e:
                    malformed += 1
                    if malformed == 1:
                        self.logger.warning(f"First malformed TSV at line {line_no}: {e}")
                    if malformed <= 10:  # Show first few errors
                        self.logger.debug(f"Malformed TSV line {line_no}: {line.strip()[:100]}")
        else:
            for line in fh:
                if stop.is_set():
                    break
                line_no += 1
                if not line or line == "\n":
                    continue
                try:
                    obj = json.loads(line)
                    keystr = obj.get("seq", "")
                    seq = [int(x) for x in keystr.split() if x.strip()]
                    cnt = int(obj.get("count", 0))
                    if seq and cnt > 0:
                        yield seq, cnt
                except json.JSONDecodeError as e:
                    malformed += 1
                    if malformed == 1:
                        self.logger.warning(f"First malformed JSON at line {line_no}: {e}")
                    if malformed <= 10:
                        self.logger.debug(f"Malformed JSON line {line_no}: {line.strip()[:100]}")
                except Exception as e:
                    malformed += 1
                    if malformed == 1:
                        self.logger.warning(f"First parsing error at line {line_no}: {e}")
        
        if malformed > 0:
            self.logger.warning(f"Total malformed lines skipped: {malformed:,}")


def select_merges_optimal(candidates: List[Tuple[Tuple[int, int], int]], 
                         max_merges: int) -> List[Tuple[Tuple[int, int], int]]:
    """Select merges using optimal greedy algorithm that maximizes total frequency."""
    if not candidates:
        return []
    
    # Sort by frequency descending
    candidates_sorted = sorted(candidates, key=lambda x: -x[1])
    
    selected = []
    used_symbols: Set[int] = set()
    
    for (a, b), freq in candidates_sorted:
        if a not in used_symbols and b not in used_symbols:
            selected.append(((a, b), freq))
            used_symbols.add(a)
            used_symbols.add(b)
            if len(selected) >= max_merges:
                break
    
    return selected


class OfflineBPE:
    """Optimized offline BPE trainer for European/Nordic languages."""
    
    def __init__(self, logger: logging.Logger, auto_pair_mem_fraction: Optional[float] = None, 
                 pair_mem_threshold: int = 85, gzip_level: int = 1, bytes_per_pair_entry: int = 64):
        self.logger = logger
        self.PAIR_CHUNK_LIMIT = 10_000_000  # Increased for better performance
        self.MEM_THRESHOLD = int(pair_mem_threshold)
        self.GZIP_LEVEL = int(gzip_level)
        self.BYTES_PER_PAIR_ENTRY = int(bytes_per_pair_entry)
        self.auto_pair_mem_fraction = (auto_pair_mem_fraction 
                                     if (auto_pair_mem_fraction and auto_pair_mem_fraction > 0.0) 
                                     else None)

    def _auto_size_pair_chunk_limit(self):
        """Automatically size pair chunk limit based on available memory."""
        if self.auto_pair_mem_fraction is None:
            return
        
        try:
            vm = psutil.virtual_memory()
            total = float(vm.total)
            current_pct = float(vm.percent)
        except Exception:
            return
        
        safety = 3.0  # More conservative for stability
        target_bytes = total * float(self.auto_pair_mem_fraction)
        near_threshold = current_pct >= (self.MEM_THRESHOLD - safety - 10)
        
        if near_threshold:
            allowed_extra_pct = (self.MEM_THRESHOLD - safety) - current_pct
            if allowed_extra_pct <= 1.0:
                return
            budget_bytes = total * (allowed_extra_pct / 100.0)
        else:
            budget_bytes = target_bytes
        
        est_limit = max(2_000_000, int(budget_bytes / self.BYTES_PER_PAIR_ENTRY))
        
        if est_limit > self.PAIR_CHUNK_LIMIT * 1.2:
            old = self.PAIR_CHUNK_LIMIT
            self.PAIR_CHUNK_LIMIT = est_limit
            self.logger.info(
                f"Auto-sized PAIR_CHUNK_LIMIT: {old:,} -> {self.PAIR_CHUNK_LIMIT:,} | "
                f"mem_now={current_pct:.1f}% mode={'full-target' if not near_threshold else 'clamped'}"
            )
        elif self.PAIR_CHUNK_LIMIT > est_limit * 2 and current_pct > (self.MEM_THRESHOLD - 5):
            old = self.PAIR_CHUNK_LIMIT
            self.PAIR_CHUNK_LIMIT = max(2_000_000, est_limit)
            self.logger.info(f"Reduced PAIR_CHUNK_LIMIT: {old:,} -> {self.PAIR_CHUNK_LIMIT:,}")

    @staticmethod
    def pairs_in_seq(seq: List[int]) -> Iterable[Tuple[int, int]]:
        """Generate adjacent pairs from sequence."""
        for i in range(len(seq) - 1):
            yield (seq[i], seq[i+1])

    @staticmethod
    def _apply_merges_greedy(seq: List[int], pair2newid: Dict[Tuple[int, int], int]) -> List[int]:
        """Apply learned merges greedily to a sequence."""
        if not seq or not pair2newid:
            return seq
        
        out: List[int] = []
        i = 0
        L = len(seq)
        
        while i < L:
            if i + 1 < L:
                nid = pair2newid.get((seq[i], seq[i+1]))
                if nid is not None:
                    out.append(nid)
                    i += 2
                    continue
            out.append(seq[i])
            i += 1
        
        return out

    def _spill_pair_chunk(self, tmpdir: str, idx: int, pc: Dict[Tuple[int, int], int]) -> str:
        """Spill pair counts to a compressed TSV file."""
        os.makedirs(tmpdir, exist_ok=True)
        path_final = os.path.join(tmpdir, f"pc_{idx:06d}.tsv.gz")
        fd, tmppath = tempfile.mkstemp(prefix=".pc_", suffix=".tsv.gz", dir=tmpdir)
        os.close(fd)
        
        try:
            with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=self.GZIP_LEVEL) as f:
                for (a, b), cnt in sorted(pc.items()):
                    f.write(f"{a} {b}\t{cnt}\n")
            
            try:
                with open(tmppath, "rb") as _f:
                    os.fsync(_f.fileno())
            except Exception:
                pass
            
            os.replace(tmppath, path_final)
            pc.clear()
            return path_final
            
        except Exception as e:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            raise e

    @staticmethod
    def _read_pc_line(fh) -> Optional[Tuple[Tuple[int, int], int]]:
        """Read one line from pair count file."""
        try:
            line = fh.readline()
        except Exception:
            return None
        
        if not line:
            return None
        
        try:
            kv, c = line.rstrip("\n").split("\t", 1)
            parts = kv.split(" ")
            if len(parts) != 2:
                return None
            a_str, b_str = parts
            return (int(a_str), int(b_str)), int(c)
        except (ValueError, IndexError):
            return None

    def _merge_pair_chunks_topk(self, chunk_paths: List[str], min_frequency: int, topk: int, 
                               exclude_pairs: Set[Tuple[int, int]], logger: logging.Logger) -> List[Tuple[Tuple[int, int], int]]:
        """K-way merge pair chunks and select top-K pairs by frequency."""
        iters = []
        for p in chunk_paths:
            try:
                iters.append(gzip.open(p, "rt", encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not open pair chunk {p}: {e}")
        
        if not iters:
            return []
        
        heads = []
        heap = []
        
        # Initialize heap with first record from each file
        for idx, fh in enumerate(iters):
            rec = self._read_pc_line(fh)
            heads.append(rec)
            if rec is not None:
                heapq.heappush(heap, (rec[0], idx))
        
        top: List[Tuple[int, Tuple[int, int]]] = []  # min-heap of (count, pair)
        skipped_dups = 0
        processed_pairs = 0
        
        while heap:
            pair, src = heapq.heappop(heap)
            total = 0
            
            # Accumulate counts for this pair across all sources
            while True:
                rec = heads[src]
                if rec is not None and rec[0] == pair:
                    total += rec[1]
                    heads[src] = self._read_pc_line(iters[src])
                    if heads[src] is not None:
                        heapq.heappush(heap, (heads[src][0], src))
                
                # Check if next pair in heap is the same
                if heap and heap[0][0] == pair:
                    _, src2 = heapq.heappop(heap)
                    src = src2
                    continue
                break
            
            processed_pairs += 1
            
            # Filter by frequency and exclusion set
            if total >= min_frequency:
                if pair in exclude_pairs:
                    skipped_dups += 1
                else:
                    if len(top) < topk:
                        heapq.heappush(top, (total, pair))
                    else:
                        if total > top[0][0]:
                            heapq.heapreplace(top, (total, pair))
            
            # Log progress for large merges
            if processed_pairs % 1_000_000 == 0:
                logger.info(f"Processed {human_int(processed_pairs)} pairs, top heap size: {len(top)}")
        
        # Close all file handles
        for fh in iters:
            try:
                fh.close()
            except Exception:
                pass
        
        if skipped_dups > 0:
            logger.info(f"Top-K selection: skipped {human_int(skipped_dups)} pairs already merged.")
        
        logger.info(f"Processed {human_int(processed_pairs)} total pairs, selected {len(top)} candidates")
        
        # Convert to list and sort by frequency descending
        top_sorted = sorted([(p, c) for (c, p) in top], key=lambda x: -x[1])
        return top_sorted

    def _count_pairs_one_pass(self, counts_reader: CountsReader, pair2newid: Dict[Tuple[int, int], int], 
                             tmpdir: str, logger: logging.Logger, pair_workers: int, 
                             stop: threading.Event) -> Tuple[List[str], int, int, int]:
        """Streaming pair counting with RAM-bounded spill and multi-threading support."""
        pc: Dict[Tuple[int, int], int] = {}
        pc_lock = threading.Lock()
        chunk_paths: List[str] = []
        chunk_idx = 0
        unique_types = 0
        expanded_tokens = 0
        max_pc_size = 0

        def maybe_spill_locked():
            nonlocal chunk_idx, max_pc_size
            if len(pc) > max_pc_size:
                max_pc_size = len(pc)
            
            should_spill = (
                len(pc) >= self.PAIR_CHUNK_LIMIT or 
                memory_percent() >= float(self.MEM_THRESHOLD)
            )
            
            if should_spill:
                path = self._spill_pair_chunk(tmpdir, chunk_idx, pc)
                chunk_paths.append(path)
                chunk_idx += 1
                logger.info(f"Spilled chunk {chunk_idx-1}: {len(pc):,} pairs, mem={memory_percent():.1f}%")
                gated_cleanup()

        # Threading constants
        PRODUCER_BATCH_LINES = 10000  # Larger batches for efficiency
        WORKER_LOCAL_PAIR_CAP = 500_000  # Increased capacity
        WORKER_LOCAL_ITEM_CAP = 50_000

        def worker(wid: int, q: "queue.Queue[Optional[List[Tuple[List[int], int]]]]"):
            nonlocal expanded_tokens
            local_pairs: Dict[Tuple[int, int], int] = {}
            local_expanded = 0
            processed_items = 0
            
            while not stop.is_set():
                try:
                    batch = q.get(timeout=1.0)
                    if batch is None:
                        break
                except queue.Empty:
                    continue
                
                for seq, cnt in batch:
                    if stop.is_set():
                        break
                    
                    # Apply existing merges
                    s2 = self._apply_merges_greedy(seq, pair2newid)
                    local_expanded += len(s2) * cnt
                    
                    # Count adjacent pairs
                    for a, b in self.pairs_in_seq(s2):
                        local_pairs[(a, b)] = local_pairs.get((a, b), 0) + cnt
                    
                    processed_items += 1
                    
                    # Flush local data periodically
                    should_flush = (
                        len(local_pairs) >= WORKER_LOCAL_PAIR_CAP or 
                        processed_items >= WORKER_LOCAL_ITEM_CAP
                    )
                    
                    if should_flush:
                        with pc_lock:
                            for k, v in local_pairs.items():
                                pc[k] = pc.get(k, 0) + v
                            expanded_tokens += local_expanded
                            local_pairs.clear()
                            local_expanded = 0
                            maybe_spill_locked()
                        processed_items = 0
                        
                        if stop.is_set():
                            break
                
                q.task_done()
            
            # Final flush
            if local_pairs:
                with pc_lock:
                    for k, v in local_pairs.items():
                        pc[k] = pc.get(k, 0) + v
                    expanded_tokens += local_expanded
                    maybe_spill_locked()

        pair_workers = max(1, int(pair_workers))
        
        # Single-threaded path for small workloads or single worker
        if pair_workers == 1:
            with counts_reader as cr:
                for i, (seq, cnt) in enumerate(cr.iter_records(stop), 1):
                    if stop.is_set():
                        break
                    
                    unique_types += 1
                    s2 = self._apply_merges_greedy(seq, pair2newid)
                    expanded_tokens += len(s2) * cnt
                    
                    for a, b in self.pairs_in_seq(s2):
                        pc[(a, b)] = pc.get((a, b), 0) + cnt
                    
                    # Periodic logging and spilling
                    if (i % 1_000_000) == 0:
                        est_mem_bytes = len(pc) * self.BYTES_PER_PAIR_ENTRY
                        mem_disp = self._format_bytes(est_mem_bytes)
                        
                        logger.info(
                            f"Pair pass: seqs={human_int(i)} distinct_pairs={human_int(len(pc))} "
                            f"est_pair_mem={mem_disp} limit={human_int(self.PAIR_CHUNK_LIMIT)} "
                            f"chunks={len(chunk_paths)} mem={memory_percent():.1f}%"
                        )
                        
                        if len(pc) >= self.PAIR_CHUNK_LIMIT or memory_percent() >= float(self.MEM_THRESHOLD):
                            with pc_lock:
                                path = self._spill_pair_chunk(tmpdir, chunk_idx, pc)
                                chunk_paths.append(path)
                                chunk_idx += 1
            
            # Final spill
            if pc:
                with pc_lock:
                    path = self._spill_pair_chunk(tmpdir, chunk_idx, pc)
                    chunk_paths.append(path)
            
            logger.info(
                f"Pair pass summary: unique_types={human_int(unique_types)} "
                f"peak_pairs={human_int(len(pc)) if pc else human_int(max_pc_size)} "
                f"chunks={len(chunk_paths)} mem={memory_percent():.1f}%"
            )
            
            return chunk_paths, unique_types, expanded_tokens, max_pc_size

        # Multi-threaded path
        q: "queue.Queue[Optional[List[Tuple[List[int], int]]]]" = queue.Queue(maxsize=pair_workers * 6)
        threads: List[threading.Thread] = []
        
        for wid in range(pair_workers):
            t = threading.Thread(target=worker, args=(wid, q), daemon=False, name=f"pair-{wid}")
            t.start()
            threads.append(t)

        last_log_time = time.time()
        last_log_items = 0
        
        try:
            with counts_reader as cr:
                batch: List[Tuple[List[int], int]] = []
                for i, (seq, cnt) in enumerate(cr.iter_records(stop), 1):
                    if stop.is_set():
                        break
                    
                    unique_types += 1
                    batch.append((seq, cnt))
                    
                    if len(batch) >= PRODUCER_BATCH_LINES:
                        q.put(batch)
                        batch = []
                    
                    # Periodic logging
                    if (i % 1_000_000) == 0:
                        now = time.time()
                        rate = (i - last_log_items) / max(1e-6, (now - last_log_time))
                        last_log_items = i
                        last_log_time = now
                        
                        with pc_lock:
                            est_mem_bytes = len(pc) * self.BYTES_PER_PAIR_ENTRY
                            mem_disp = self._format_bytes(est_mem_bytes)
                            
                            logger.info(
                                f"Pair pass (mt): seqs={human_int(i)} ({rate:,.0f}/s) "
                                f"distinct_pairs={human_int(len(pc))} est_pair_mem={mem_disp} "
                                f"chunks={len(chunk_paths)} mem={memory_percent():.1f}%"
                            )
                
                # Send final batch
                if batch:
                    q.put(batch)
        
        finally:
            # Signal workers to stop
            for _ in threads:
                q.put(None)
            
            # Wait for all workers to complete
            for t in threads:
                t.join(timeout=30.0)
                if t.is_alive():
                    logger.warning(f"Worker thread {t.name} did not finish in time")

        # Final spill
        with pc_lock:
            if pc:
                path = self._spill_pair_chunk(tmpdir, chunk_idx, pc)
                chunk_paths.append(path)
        
        logger.info(
            f"Pair pass summary: unique_types={human_int(unique_types)} "
            f"peak_pairs={human_int(max(len(pc), max_pc_size))} "
            f"chunks={len(chunk_paths)} mem={memory_percent():.1f}%"
        )
        
        return chunk_paths, unique_types, expanded_tokens, max_pc_size

    def _format_bytes(self, byte_count: int) -> str:
        """Format byte count as human readable string."""
        if byte_count < 1024:
            return f"{byte_count}B"
        elif byte_count < 1024**2:
            return f"{byte_count/1024:.1f}K"
        elif byte_count < 1024**3:
            return f"{byte_count/(1024**2):.1f}M"
        else:
            return f"{byte_count/(1024**3):.2f}G"

    def train(self, counts_path: str, counts_format: str, outdir: str, vocab_size: int, 
              min_frequency: int, batch_merges: int, pair_workers: int, 
              stop: threading.Event) -> Tuple[List[Tuple[int, int]], Dict[int, str], Dict[str, Any]]:
        """Train BPE model from counts file."""
        
        logger = self.logger
        os.makedirs(outdir, exist_ok=True)
        
        # Create temp directory under outdir
        tmp_root = os.path.join(outdir, f".bpe_train_tmp.{blake2b16(counts_path)}.{int(time.time())}")
        os.makedirs(tmp_root, exist_ok=True)
        
        # Initialize European/Nordic vocabulary
        token_str, token2id = init_european_nordic_vocab()
        next_id = len(token_str)
        initial_vocab_size = len(token_str)
        
        merges: List[Tuple[int, int]] = []
        pair2newid: Dict[Tuple[int, int], int] = {}
        target_vocab = max(initial_vocab_size, int(vocab_size))
        
        logger.info(f"Starting European/Nordic BPE training")
        logger.info(f"Initial vocabulary: {human_int(initial_vocab_size)} characters")
        logger.info(f"Target vocabulary: {human_int(target_vocab)}")
        logger.info(f"Target merges: {human_int(target_vocab - initial_vocab_size)}")
        logger.info(f"Min frequency: {min_frequency}, Batch size: {batch_merges}")
        
        # Get optimal batch size based on system resources
        try:
            available_gb = psutil.virtual_memory().available / (1024**3)
            optimal_batch = get_optimal_batch_size(available_gb, target_vocab)
            if optimal_batch != batch_merges:
                logger.info(f"Suggested optimal batch size: {optimal_batch} (you specified: {batch_merges})")
        except Exception:
            pass
        
        t0_total = time.time()
        rounds = 0
        total_sequences_processed = 0
        
        while (initial_vocab_size + len(merges)) < target_vocab and not stop.is_set():
            rounds += 1
            self._auto_size_pair_chunk_limit()
            
            pc_tmpdir = os.path.join(tmp_root, f"pairs_round_{rounds:03d}")
            try:
                os.makedirs(pc_tmpdir, exist_ok=True)
            except Exception:
                pass
            
            logger.info(f"=== Round {rounds} ===")
            logger.info(f"Current vocabulary size: {human_int(initial_vocab_size + len(merges))}")
            logger.info(f"Merges learned so far: {human_int(len(merges))}")
            
            counts_reader = CountsReader(counts_path, counts_format, logger)
            t1 = time.time()
            
            # Count pairs for this round
            chunk_paths, uniq_types, expanded_tokens, max_pairs = self._count_pairs_one_pass(
                counts_reader=counts_reader,
                pair2newid=pair2newid,
                tmpdir=pc_tmpdir,
                logger=logger,
                pair_workers=pair_workers,
                stop=stop,
            )
            
            if stop.is_set():
                logger.info("Training interrupted by signal")
                break
            
            if not chunk_paths:
                logger.info("No pair chunks produced; stopping training.")
                break
            
            # Track total sequences for adaptive frequency calculation
            if rounds == 1:
                total_sequences_processed = uniq_types
                adaptive_min_freq = calculate_adaptive_min_frequency(total_sequences_processed, min_frequency)
                if adaptive_min_freq != min_frequency:
                    logger.info(f"Adaptive min frequency: {adaptive_min_freq} (base: {min_frequency})")
                    min_frequency = adaptive_min_freq
            
            # Select top candidate pairs
            logger.info(f"Merging {len(chunk_paths)} pair chunks to select top candidates...")
            raw_top = self._merge_pair_chunks_topk(
                chunk_paths=chunk_paths,
                min_frequency=min_frequency,
                topk=batch_merges * 3,  # Get more candidates for better selection
                exclude_pairs=set(pair2newid.keys()),
                logger=logger,
            )
            
            # Clean up chunk files early to save space
            for p in chunk_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            try:
                os.rmdir(pc_tmpdir)
            except Exception:
                pass
            
            if not raw_top:
                logger.info("No pairs meet minimum frequency threshold; stopping training.")
                break
            
            logger.info(f"Found {len(raw_top)} candidate pairs for merging")
            
            # Optimal conflict-free selection
            accepted = select_merges_optimal(raw_top, batch_merges)
            
            if not accepted:
                logger.warning("No conflict-free pairs selected; stopping.")
                break
            
            # Apply selected merges
            merges_before = len(merges)
            new_tokens_created = 0
            merge_frequencies = []
            
            for (a, b), cnt in accepted:
                if (a, b) in pair2newid:
                    continue  # Already processed
                
                # Safely concatenate tokens
                s_new = safe_concatenate(token_str[a], token_str[b])
                if s_new is None:
                    logger.debug(f"Skipping invalid merge: {repr(token_str[a])} + {repr(token_str[b])}")
                    continue
                
                # Check if token already exists
                existing_id = token2id.get(s_new)
                
                if existing_id is not None:
                    # Reuse existing token
                    pair2newid[(a, b)] = existing_id
                else:
                    # Create new token
                    nid = next_id
                    next_id += 1
                    token_str[nid] = s_new
                    token2id[s_new] = nid
                    pair2newid[(a, b)] = nid
                    new_tokens_created += 1
                
                merges.append((a, b))
                merge_frequencies.append(cnt)
                
                # Stop if we've reached target vocabulary
                if (initial_vocab_size + len(merges)) >= target_vocab:
                    break
            
            accepted_pairs = len(merges) - merges_before
            dt = time.time() - t1
            
            if merge_frequencies:
                min_freq_this_round = min(merge_frequencies)
                max_freq_this_round = max(merge_frequencies)
                avg_freq_this_round = sum(merge_frequencies) / len(merge_frequencies)
                
                logger.info(
                    f"Round {rounds} completed in {dt:.1f}s"
                )
                logger.info(
                    f"  Merges: {human_int(len(merges))}/{human_int(target_vocab - initial_vocab_size)} "
                    f"(+{accepted_pairs} this round)"
                )
                logger.info(
                    f"  New tokens created: {new_tokens_created}"
                )
                logger.info(
                    f"  Unique sequences: ~{human_int(uniq_types)}"
                )
                logger.info(
                    f"  Expanded tokens: ~{human_int(expanded_tokens)}"
                )
                logger.info(
                    f"  Merge frequencies: min={human_int(min_freq_this_round)} "
                    f"avg={human_int(int(avg_freq_this_round))} max={human_int(max_freq_this_round)}"
                )
                logger.info(
                    f"  Peak pair count: {human_int(max_pairs)}"
                )
                logger.info(
                    f"  Memory usage: {memory_percent():.1f}%"
                )
            
            gated_cleanup()
            
            # Check for stagnation
            if accepted_pairs == 0 or new_tokens_created == 0:
                logger.warning("Training stagnated (no new merges accepted); stopping.")
                break
            
            # Adaptive batch size adjustment
            if rounds > 1 and len(merge_frequencies) < batch_merges * 0.5:
                logger.info(f"Few merges found ({len(merge_frequencies)} < {batch_merges * 0.5}), "
                           f"consider reducing batch size for remaining rounds")
        
        # Training summary
        final_vocab_size = initial_vocab_size + len(merges)
        total_time_hours = (time.time() - t0_total) / 3600.0
        
        stats = {
            "initial_vocab_size": initial_vocab_size,
            "target_vocab_size": int(vocab_size),
            "final_vocab_size": final_vocab_size,
            "merges_learned": len(merges),
            "training_rounds": rounds,
            "total_sequences_processed": total_sequences_processed,
            "training_time_hours": round(total_time_hours, 3),
            "merges_per_hour": round(len(merges) / max(total_time_hours, 0.001), 1),
        }
        
        logger.info("=== Training Complete ===")
        logger.info(f"Final vocabulary size: {human_int(final_vocab_size)}")
        logger.info(f"Total merges learned: {human_int(len(merges))}")
        logger.info(f"Training rounds: {rounds}")
        logger.info(f"Total training time: {total_time_hours:.2f} hours")
        logger.info(f"Training rate: {len(merges) / max(total_time_hours, 0.001):.1f} merges/hour")
        
        # Cleanup temporary directory
        try:
            for n in os.listdir(tmp_root):
                p = os.path.join(tmp_root, n)
                try:
                    if os.path.isdir(p):
                        for nn in os.listdir(p):
                            try:
                                os.remove(os.path.join(p, nn))
                            except Exception:
                                pass
                        os.rmdir(p)
                    else:
                        os.remove(p)
                except Exception:
                    pass
            os.rmdir(tmp_root)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {tmp_root}: {e}")
        
        return merges, token_str, stats


def save_artifacts(outdir: str, merges: List[Tuple[int, int]], token_str: Dict[int, str], 
                  vocab_size_target: int, special_tokens: List[str], normalization: str, 
                  strip_accents: bool, strip: bool, lowercase: bool, logger: logging.Logger):
    """Save BPE model artifacts with atomic writes."""
    
    os.makedirs(outdir, exist_ok=True)
    
    # Build vocabulary: special tokens first, then base chars, then merged tokens
    ordered_tokens: List[str] = []
    seen = set()
    
    # Add special tokens first
    for sp in special_tokens:
        if sp and sp not in seen:
            ordered_tokens.append(sp)
            seen.add(sp)
    
    # Add character and merge tokens sorted by ID
    for i in sorted(token_str.keys()):
        s = token_str[i]
        if s not in seen:
            ordered_tokens.append(s)
            seen.add(s)
    
    # Create final vocabulary mapping
    vocab = {tok: idx for idx, tok in enumerate(ordered_tokens)}
    
    # Prepare file paths
    merges_txt = os.path.join(outdir, "merges.txt")
    vocab_json = os.path.join(outdir, "vocab.json")
    meta_json = os.path.join(outdir, "tokenizer_info.json")
    
    # Create temporary files for atomic writes
    fd1, tmp1 = tempfile.mkstemp(prefix=".merges_", suffix=".txt", dir=outdir)
    os.close(fd1)
    fd2, tmp2 = tempfile.mkstemp(prefix=".vocab_", suffix=".json", dir=outdir)
    os.close(fd2)
    fd3, tmp3 = tempfile.mkstemp(prefix=".meta_", suffix=".json", dir=outdir)
    os.close(fd3)
    
    try:
        # Write merges.txt
        with open(tmp1, "w", encoding="utf-8", newline='\n') as f:
            for a, b in merges:
                f.write(f"{token_str[a]} {token_str[b]}\n")
        
        # Write vocab.json
        with open(tmp2, "w", encoding="utf-8", newline='\n') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        # Write metadata
        info = {
            "model_type": "BPE",
            "language_focus": "European/Nordic/English",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_vocab_size": int(vocab_size_target),
            "actual_vocab_entries": len(vocab),
            "special_tokens": special_tokens,
            "character_level": True,
            "multilingual": True,
            "byte_level": False,
            "normalization": normalization,
            "strip_accents": bool(strip_accents),
            "strip_whitespace": bool(strip),
            "lowercase": bool(lowercase),
            "vocab_file": "vocab.json",
            "merges_file": "merges.txt",
            "unk_token": "[UNK]" if "[UNK]" in special_tokens else None,
        }
        
        with open(tmp3, "w", encoding="utf-8", newline='\n') as f:
            json.dump(info, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        # Sync files to disk
        for tmp in (tmp1, tmp2, tmp3):
            try:
                with open(tmp, "rb") as _f:
                    os.fsync(_f.fileno())
            except Exception:
                pass
        
        # Atomically replace target files
        os.replace(tmp1, merges_txt)
        os.replace(tmp2, vocab_json)
        os.replace(tmp3, meta_json)
        
        logger.info(f"Successfully wrote tokenizer artifacts:")
        logger.info(f"  - {merges_txt}")
        logger.info(f"  - {vocab_json}")
        logger.info(f"  - {meta_json}")
        
    except Exception as e:
        # Clean up temporary files on error
        for tmp in (tmp1, tmp2, tmp3):
            try:
                os.remove(tmp)
            except Exception:
                pass
        raise e


def main():
    """Main entry point for BPE training."""
    parser = argparse.ArgumentParser(
        description="European/Nordic/English multilingual BPE tokenizer training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument("--counts", type=str, required=True, 
                       help="Path to merged counts file (.jsonl[.gz] or .tsv[.gz])")
    parser.add_argument("--counts-format", type=str, default="auto", 
                       choices=["auto", "jsonl", "tsv"], 
                       help="Force input format (auto-detect if not specified)")
    parser.add_argument("--outdir", type=str, required=True, 
                       help="Output directory for tokenizer artifacts")
    
    # Model parameters
    parser.add_argument("--vocab-size", type=int, default=64000,
                       help="Target vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2,
                       help="Minimum frequency for pair merges")
    parser.add_argument("--batch-merges", type=int, default=12000,
                       help="Number of merges to apply per training round")
    
    # Performance parameters
    parser.add_argument("--pair-workers", type=int, default=8,
                       help="Number of worker threads for pair counting")
    parser.add_argument("--auto-pair-mem-fraction", type=float, default=0.6,
                       help="Fraction of system RAM to use for pair counting")
    parser.add_argument("--pair-mem-threshold", type=int, default=85,
                       help="Memory usage percentage to trigger spilling")
    
    # Text normalization (metadata only - normalization done in scan phase)
    parser.add_argument("--normalization", type=str, default="NFKC",
                       choices=["NONE", "NFC", "NFD", "NFKC", "NFKD"],
                       help="Unicode normalization form (for metadata)")
    parser.add_argument("--strip-accents", action="store_true", default=False,
                       help="Whether accents were stripped (for metadata)")
    parser.add_argument("--strip-whitespace", action="store_true", default=False,
                       help="Whether whitespace was stripped (for metadata)")
    parser.add_argument("--lowercase", action="store_true", default=False,
                       help="Whether text was lowercased (for metadata)")
    
    # Special tokens
    parser.add_argument("--special-tokens", type=str, 
                       default="[PAD],[UNK],[BOS],[EOS],[MASK],<|endoftext|>",
                       help="Comma-separated list of special tokens")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Optional log file path")
    
    args = parser.parse_args()
    
    # Configure logging
    logger = configure_logging(args.log_level, args.log_file)
    stop = threading.Event()
    install_signal_handler(stop, logger)
    
    try:
        # Parse special tokens
        special_tokens = [s.strip() for s in (args.special_tokens or "").split(",") if s.strip()]
        
        # Ensure [UNK] is present
        if "[UNK]" not in special_tokens:
            special_tokens.insert(0, "[UNK]")
        
        # Deduplicate while preserving order
        seen = set()
        special_tokens = [s for s in special_tokens if not (s in seen or seen.add(s))]
        
        logger.info(f"Special tokens: {special_tokens}")
        
        # Validate arguments
        if args.vocab_size < 1000:
            raise ValueError("vocab-size must be at least 1000")
        if args.min_frequency < 1:
            raise ValueError("min-frequency must be at least 1")
        if args.batch_merges < 100:
            raise ValueError("batch-merges must be at least 100")
        if not (0.1 <= args.auto_pair_mem_fraction <= 0.9):
            raise ValueError("auto-pair-mem-fraction must be between 0.1 and 0.9")
        if not (50 <= args.pair_mem_threshold <= 95):
            raise ValueError("pair-mem-threshold must be between 50 and 95")
        
        # Log system info
        try:
            mem_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = os.cpu_count() or 1
            logger.info(f"System: {mem_gb:.1f}GB RAM, {cpu_count} CPUs")
            
            optimal_batch = get_optimal_batch_size(mem_gb, args.vocab_size)
            if optimal_batch != args.batch_merges:
                logger.info(f"Recommended batch size for your system: {optimal_batch}")
        except Exception:
            pass
        
        # Initialize BPE trainer
        bpe = OfflineBPE(
            logger=logger,
            auto_pair_mem_fraction=args.auto_pair_mem_fraction,
            pair_mem_threshold=args.pair_mem_threshold,
            gzip_level=1,
            bytes_per_pair_entry=64,
        )
        
        # Train the BPE model
        logger.info("Starting BPE training...")
        merges, token_str, stats = bpe.train(
            counts_path=args.counts,
            counts_format=args.counts_format,
            outdir=args.outdir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            batch_merges=args.batch_merges,
            pair_workers=args.pair_workers,
            stop=stop,
        )
        
        if stop.is_set():
            logger.warning("Training was interrupted")
            return
        
        # Save model artifacts
        logger.info("Saving tokenizer artifacts...")
        save_artifacts(
            outdir=args.outdir,
            merges=merges,
            token_str=token_str,
            vocab_size_target=args.vocab_size,
            special_tokens=special_tokens,
            normalization=args.normalization,
            strip_accents=args.strip_accents,
            strip=args.strip_whitespace,
            lowercase=args.lowercase,
            logger=logger,
        )
        
        # Print final statistics
        logger.info("=== Final Statistics ===")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if isinstance(value, int) and value > 1000:
                    logger.info(f"{key.replace('_', ' ').title()}: {human_int(value)}")
                else:
                    logger.info(f"{key.replace('_', ' ').title()}: {value}")
            else:
                logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        # Calculate compression ratio estimate
        if stats.get("total_sequences_processed", 0) > 0:
            initial_chars = stats["initial_vocab_size"]
            final_tokens = stats["final_vocab_size"]
            compression_estimate = max(1.0, initial_chars / final_tokens * 100)
            logger.info(f"Estimated compression ratio: {compression_estimate:.1f}%")
        
        logger.info("European/Nordic BPE training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        gated_cleanup()


if __name__ == "__main__":
    main()