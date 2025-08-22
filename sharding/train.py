
"""
Offline, multilingual-ready BPE training from pre-scanned counts.

Fixes and features:
- Supports ALL counts formats: JSONL, JSONL.GZ, TSV, TSV.GZ, and streaming FDs/FIFOs.
- Stream-safe format detection (no double-open; works with /proc/self/fd/*).
- parsing for TSV ("<byte_ids_space_separated>\\t<count>") and JSONL {"seq": "...", "count": N}.
- Hard fail if zero merges learned (prevents empty merges.txt).
- Mem reclame on Linux via malloc_trim.
- Configurable Unicode normalization (NFKC default) and other options.
- pair-chunk spilling with atomic writes and fsync.
- Special tokens reserved first to avoid collisions.
"""

import os
import sys
import gc
import gzip
import json
import time
import psutil
import ctypes
import signal
import logging
import argparse
import tempfile
import traceback
import resource
import hashlib
import threading
import queue
from typing import Dict, List, Tuple, Optional, Iterable, Any

try:
    import numpy as np
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, decoders
    from transformers import PreTrainedTokenizerFast
    from safetensors.numpy import save_file as safetensors_save_file
except Exception as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)



# Utilis

def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.root.handlers = []
    logging.root.setLevel(lvl)
    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    return f"{n:,}"


def memory_percent() -> float:
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def gated_cleanup():
    try:
        gc.collect()
    except Exception:
        pass
    # mem reclaime on Linux
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


def install_signal_handler(stop_event: threading.Event, logger: logging.Logger):
    """
    Set SIGINT/SIGTERM handlers. SIGINT raises KeyboardInterrupt for fast exit.
    SIGTERM sets the event and raises SystemExit to allow cleanup.
    """
    def handler(signum, frame):
        logger.warning(f"Received signal {signum}, requesting shutdown...")
        stop_event.set()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
        if signum == signal.SIGTERM:
            raise SystemExit(0)
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


def blake2b16(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def build_normalizer(unicode_form: str = "NFKC",
                     strip_accents: bool = False,
                     strip: bool = False,
                     lowercase: bool = False):
    """
    Compose a normalization pipeline. NFKC default is essential for multilingual consistency as they say.
    """
    steps: List[Any] = []
    uf = (unicode_form or "NFKC").upper()
    if uf != "NONE":
        if uf in {"NFC", "NFD", "NFKC", "NFKD"}:
            steps.append(getattr(normalizers, uf)())
        else:
            steps.append(normalizers.NFKC())
    if strip_accents:
        steps.append(normalizers.StripAccents())
    if strip:
        steps.append(normalizers.Strip())
    if lowercase:
        steps.append(normalizers.Lowercase())
    if not steps:
        return None
    return normalizers.Sequence(steps)


def keystr_to_bytes(s: str) -> bytes:
    return bytes(int(x) for x in s.split(" ") if x)


# Offline BPE


class OfflineBPE:
    def __init__(self, logger: logging.Logger, *,
                 auto_pair_mem_fraction: Optional[float] = None,
                 pair_chunk_limit: Optional[int] = None,
                 mem_threshold: int = 85,
                 batch_merges: int = 1000,
                 counts_format: str = "auto"):
        self.logger = logger
        self.byte_to_uni = self._bytes_to_unicode()
        self.PAIR_CHUNK_LIMIT = int(pair_chunk_limit) if pair_chunk_limit else 8_000_000
        self.BATCH_MERGES = max(1, int(batch_merges))
        self.GZIP_LEVEL = 1
        self.MEM_THRESHOLD = int(mem_threshold)
        self.BYTES_PER_PAIR_ENTRY = 56
        self.auto_pair_mem_fraction = auto_pair_mem_fraction if (auto_pair_mem_fraction and auto_pair_mem_fraction > 0) else None
        self.counts_format = counts_format.lower().strip() if counts_format else "auto"

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    @staticmethod
    def parse_key_to_bytes(seq_key: str) -> List[int]:
        return list(keystr_to_bytes(seq_key))

    @staticmethod
    def pairs_in_seq(seq: List[int]) -> List[Tuple[int, int]]:
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    @staticmethod
    def _apply_merges_greedy(seq: List[int], pair2newid: Dict[Tuple[int, int], int]) -> List[int]:
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

    def _auto_size_pair_chunk_limit(self):
        if self.auto_pair_mem_fraction is None:
            return
        try:
            vm = psutil.virtual_memory()
            total = float(vm.total)
            current_pct = float(vm.percent)
        except Exception:
            return
        safety = 2.0
        target_bytes = total * float(self.auto_pair_mem_fraction)
        near_threshold = current_pct >= (self.MEM_THRESHOLD - safety - 5)
        if near_threshold:
            allowed_extra_pct = (self.MEM_THRESHOLD - safety) - current_pct
            if allowed_extra_pct <= 0.5:
                return
            budget_bytes = total * (allowed_extra_pct / 100.0)
            limited = True
        else:
            budget_bytes = target_bytes
            limited = False
        est_limit = int(budget_bytes / self.BYTES_PER_PAIR_ENTRY)
        est_limit = max(1_000_000, est_limit)
        if est_limit > self.PAIR_CHUNK_LIMIT * 1.1:
            old = self.PAIR_CHUNK_LIMIT
            self.PAIR_CHUNK_LIMIT = est_limit
            reason = "clamped-near-threshold" if limited else "full-target"
            self.logger.info(f"Auto-sized PAIR_CHUNK_LIMIT: {old:,} -> {self.PAIR_CHUNK_LIMIT:,} | mem_now={current_pct:.1f}% mode={reason}")
        elif self.PAIR_CHUNK_LIMIT > est_limit * 2 and current_pct > (self.MEM_THRESHOLD - 3):
            old = self.PAIR_CHUNK_LIMIT
            self.PAIR_CHUNK_LIMIT = est_limit
            self.logger.info(f"Reduced PAIR_CHUNK_LIMIT: {old:,} -> {self.PAIR_CHUNK_LIMIT:,} (mem={current_pct:.1f}%)")

    def _spill_pair_chunk(self, tmpdir: str, idx: int, pc: Dict[Tuple[int, int], int]) -> str:
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
        except Exception:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            raise

    @staticmethod
    def _read_pc_line(fh) -> Optional[Tuple[Tuple[int, int], int]]:
        try:
            line = fh.readline()
        except Exception:
            return None
        if not line:
            return None
        try:
            kv, c = line.rstrip("\n").split("\t")
            a_str, b_str = kv.split(" ")
            return (int(a_str), int(b_str)), int(c)
        except Exception:
            return None

    def _merge_pair_chunks_topk(self,
                                chunk_paths,
                                min_frequency,
                                topk,
                                pair2newid,
                                logger):
        import heapq
        iters = []
        for p in chunk_paths:
            try:
                iters.append(gzip.open(p, "rt", encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not open pair chunk {p}: {e}")
        heads = []
        keyheap = []
        for idx, fh in enumerate(iters):
            rec = self._read_pc_line(fh)
            heads.append(rec)
            if rec is not None:
                keyheap.append((rec[0], idx))
        heapq.heapify(keyheap)
        top = []  # (count, (a,b))
        skipped_dups = 0
        while keyheap:
            pair, src = heapq.heappop(keyheap)
            total = 0
            while True:
                rec = heads[src]
                if rec is not None and rec[0] == pair:
                    total += rec[1]
                    heads[src] = self._read_pc_line(iters[src])
                    if heads[src] is not None:
                        heapq.heappush(keyheap, (heads[src][0], src))
                if keyheap and keyheap[0][0] == pair:
                    _, src2 = heapq.heappop(keyheap)
                    src = src2
                    continue
                break
            if total >= min_frequency:
                if pair in pair2newid:
                    skipped_dups += 1
                else:
                    if len(top) < topk:
                        heapq.heappush(top, (total, pair))
                    else:
                        if total > top[0][0]:
                            heapq.heapreplace(top, (total, pair))
        for fh in iters:
            try:
                fh.close()
            except Exception:
                pass
        if skipped_dups:
            logger.info(f"Top-K selection: skipped {skipped_dups:,} duplicate pairs already merged.")
        top_sorted = sorted([(pair, cnt) for (cnt, pair) in top], key=lambda x: -x[1])
        return top_sorted

    # Counts reader with stream-safe format detection
    def _open_reader(self, path: str):
        if str(path).endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "rt", encoding="utf-8")

    def _detect_or_use_counts_format(self, sample_line: str) -> str:
        if self.counts_format in {"json", "tsv"}:
            return self.counts_format
        # auto: detect from first non-empty line
        if "\t" in sample_line:
            return "tsv"
        try:
            _ = json.loads(sample_line)
            return "json"
        except Exception:
            # Default to TSV if it cannot determine
            return "tsv"

    def _parse_line(self, line: str, fmt: str) -> Optional[Tuple[List[int], int]]:
        if not line or not line.strip():
            return None
        try:
            if fmt == "json":
                obj = json.loads(line)
                seq_key = obj.get("seq", "")
                cnt = int(obj.get("count", 0))
            else:
                k, c = line.rstrip("\n").split("\t")
                seq_key, cnt = k, int(c)
            s = self.parse_key_to_bytes(seq_key)
            if not s or cnt < 1:
                return None
            return s, cnt
        except Exception:
            return None

    def _count_pairs_one_pass(self, counts_path: str,
                              pair2newid: Dict[Tuple[int, int], int],
                              tmpdir: str,
                              logger: logging.Logger,
                              pair_workers: int = 1) -> Tuple[List[str], int, int, int]:
        pair_workers = max(1, int(pair_workers))
        pc: Dict[Tuple[int, int], int] = {}
        pc_lock = threading.Lock()
        chunk_paths: List[str] = []
        chunk_idx = 0
        unique_types = 0
        expanded_tokens = 0
        max_pc_size = 0
        malformed_lines_total = 0

        def maybe_spill_locked() -> None:
            nonlocal chunk_idx, max_pc_size
            if len(pc) > max_pc_size:
                max_pc_size = len(pc)
            if len(pc) >= self.PAIR_CHUNK_LIMIT or memory_percent() >= float(self.MEM_THRESHOLD):
                path = self._spill_pair_chunk(tmpdir, chunk_idx, pc)
                chunk_paths.append(path)
                chunk_idx += 1

        # Open once, detect format from first nonempty line, then process including the peeked lines.
        with self._open_reader(counts_path) as f:
            # Gather a small set of initial lines to see the format without reopening
            first_lines: List[str] = []
            fmt: Optional[str] = None
            # Read until we find a non empty line for detection; keep up to 100 lines cached
            for _ in range(100):
                l = f.readline()
                if not l:
                    break
                first_lines.append(l)
                if l.strip() and fmt is None:
                    fmt = self._detect_or_use_counts_format(l)

            if fmt is None:
                fmt = "tsv" if counts_path.endswith(".tsv") or counts_path.endswith(".tsv.gz") else "json"
            logger.info(f"Counts reader: format={fmt} | source={counts_path}")

            # Single-threaded
            if pair_workers == 1:
                i = 0
                # Iterate cached lines then the rest of the file
                def line_iter():
                    for x in first_lines:
                        yield x
                    while True:
                        x = f.readline()
                        if not x:
                            break
                        yield x

                for line in line_iter():
                    i += 1
                    parsed = self._parse_line(line, fmt)
                    if parsed is None:
                        malformed_lines_total += 1
                        continue
                    s, c = parsed
                    unique_types += 1
                    s2 = self._apply_merges_greedy(s, pair2newid)
                    expanded_tokens += len(s2) * c
                    for a, b in self.pairs_in_seq(s2):
                        pc[(a, b)] = pc.get((a, b), 0) + c
                    if len(pc) >= self.PAIR_CHUNK_LIMIT or memory_percent() >= float(self.MEM_THRESHOLD):
                        maybe_spill_locked()
                    if (i % 1_000_000) == 0:
                        est_mem_bytes = len(pc) * self.BYTES_PER_PAIR_ENTRY
                        logger.info(
                            f"Pair pass: seqs={human_int(i)} distinct_pairs={human_int(len(pc))} "
                            f"est_pair_mem={est_mem_bytes/1e6:.2f}M limit={human_int(self.PAIR_CHUNK_LIMIT)} "
                            f"chunks={len(chunk_paths)} mem={memory_percent():.1f}% fmt={fmt}"
                        )
                if pc:
                    maybe_spill_locked()
                if malformed_lines_total > 0:
                    logger.warning(f"Skipped {malformed_lines_total} malformed lines")
                logger.info(f"Pair pass summary: peak_distinct_pairs={human_int(max_pc_size)} chunks={len(chunk_paths)} mem={memory_percent():.1f}% fmt={fmt}")
                return chunk_paths, unique_types, expanded_tokens, max_pc_size

            # Multi threaded
            qlines: "queue.Queue[Optional[List[str]]]" = queue.Queue(maxsize=pair_workers * 4)
            worker_flush_counts = [0] * pair_workers
            PRODUCER_BATCH_LINES = 5000
            WORKER_LOCAL_PAIR_CAP = 200_000
            WORKER_LOCAL_ITEM_CAP = 20_000
            malformed_lock = threading.Lock()

            def worker(wid: int):
                nonlocal expanded_tokens
                local_pairs: Dict[Tuple[int, int], int] = {}
                local_expanded = 0
                processed_items = 0
                local_malformed = 0
                while True:
                    batch = qlines.get()
                    if batch is None:
                        break
                    for raw in batch:
                        parsed = self._parse_line(raw, fmt)
                        if parsed is None:
                            local_malformed += 1
                            continue
                        s, c = parsed
                        s2 = self._apply_merges_greedy(s, pair2newid)
                        local_expanded += len(s2) * c
                        for a, b in self.pairs_in_seq(s2):
                            local_pairs[(a, b)] = local_pairs.get((a, b), 0) + c
                        processed_items += 1
                        if len(local_pairs) >= WORKER_LOCAL_PAIR_CAP or processed_items >= WORKER_LOCAL_ITEM_CAP:
                            with pc_lock:
                                for k, v in local_pairs.items():
                                    pc[k] = pc.get(k, 0) + v
                                expanded_tokens += local_expanded
                                local_pairs.clear()
                                local_expanded = 0
                                maybe_spill_locked()
                                worker_flush_counts[wid] += 1
                            processed_items = 0
                if local_pairs:
                    with pc_lock:
                        for k, v in local_pairs.items():
                            pc[k] = pc.get(k, 0) + v
                        expanded_tokens += local_expanded
                        maybe_spill_locked()
                        worker_flush_counts[wid] += 1
                if local_malformed:
                    with malformed_lock:
                        nonlocal malformed_lines_total
                        malformed_lines_total += local_malformed

            threads: List[threading.Thread] = []
            for wid in range(pair_workers):
                t = threading.Thread(target=worker, args=(wid,), daemon=True)
                t.start()
                threads.append(t)

            # Producer: feed cached lines first, then stream from file
            i = 0
            batch: List[str] = []
            for l in first_lines:
                if not l:
                    continue
                batch.append(l)
                i += 1
                if len(batch) >= PRODUCER_BATCH_LINES:
                    qlines.put(batch)
                    batch = []

            while True:
                l = f.readline()
                if not l:
                    break
                batch.append(l)
                i += 1
                if len(batch) >= PRODUCER_BATCH_LINES:
                    qlines.put(batch)
                    batch = []

                if (i % 1_000_000) == 0:
                    with pc_lock:
                        est_mem_bytes = len(pc) * self.BYTES_PER_PAIR_ENTRY
                        logger.info(
                            f"Pair pass (mt): seqs={human_int(i)} distinct_pairs={human_int(len(pc))} "
                            f"est_pair_mem={est_mem_bytes/1e6:.2f}M limit={human_int(self.PAIR_CHUNK_LIMIT)} "
                            f"chunks={len(chunk_paths)} mem={memory_percent():.1f}% "
                            f"worker_flushes={sum(worker_flush_counts)} qsize={qlines.qsize()} fmt={fmt}"
                        )
            if batch:
                qlines.put(batch)
            for _ in threads:
                qlines.put(None)
            for t in threads:
                t.join()

            with pc_lock:
                if pc:
                    path = self._spill_pair_chunk(tmpdir, chunk_idx, pc)
                    chunk_paths.append(path)

            if malformed_lines_total > 0:
                logger.warning(f"Skipped {malformed_lines_total} malformed lines")
            logger.info(f"Pair pass summary: peak_distinct_pairs={human_int(max_pc_size)} chunks={len(chunk_paths)} mem={memory_percent():.1f}% fmt={fmt}")
            return chunk_paths, unique_types, expanded_tokens, max_pc_size

    def train(self,
              counts_path: str,
              vocab_size: int,
              min_frequency: int,
              pair_workers: int = 1) -> Tuple[List[Tuple[int, int]], Dict[int, str], Dict[str, Any]]:
        logger = self.logger
        tmp_root = tempfile.mkdtemp(
            prefix=f".bpe_train_tmp.{blake2b16(counts_path)}.{os.getpid()}.",
            dir=os.getenv("TMPDIR", None)
        )

        token_str: Dict[int, str] = {i: self.byte_to_uni.get(i, chr(i)) for i in range(256)}
        token2id: Dict[str, int] = {token_str[i]: i for i in range(256)}

        next_id = 256
        merges: List[Tuple[int, int]] = []
        pair2newid: Dict[Tuple[int, int], int] = {}
        target_vocab_size = max(256, int(vocab_size))
        logger.info(f"Starting offline BPE | target merges: {human_int(target_vocab_size - 256)} | min_frequency={min_frequency}")

        total_unique_seen = 0
        total_expanded_tokens = 0
        batch_num = 0
        t_train_start = time.time()

        def current_vocab_size_no_specials() -> int:
            return len(token2id)

        while current_vocab_size_no_specials() < target_vocab_size:
            batch_num += 1
            merges_needed = max(0, target_vocab_size - current_vocab_size_no_specials())
            this_batch = min(self.BATCH_MERGES, merges_needed) if merges_needed > 0 else 0
            if this_batch <= 0:
                break

            self._auto_size_pair_chunk_limit()
            pc_tmpdir = os.path.join(tmp_root, f"pairs_pass_{len(merges):06d}")
            try:
                if os.path.isdir(pc_tmpdir):
                    for n in os.listdir(pc_tmpdir):
                        try:
                            os.remove(os.path.join(pc_tmpdir, n))
                        except Exception:
                            pass
                os.makedirs(pc_tmpdir, exist_ok=True)
            except Exception:
                pass

            t0 = time.time()
            chunk_paths, uniq, expanded, _peak = self._count_pairs_one_pass(
                counts_path, pair2newid, pc_tmpdir, logger, pair_workers=pair_workers
            )
            total_unique_seen = uniq
            total_expanded_tokens = expanded

            if not chunk_paths:
                logger.error("No pair chunks produced; check counts format and content.")
                break

            top_pairs = self._merge_pair_chunks_topk(
                chunk_paths=chunk_paths,
                min_frequency=min_frequency,
                topk=this_batch,
                pair2newid=pair2newid,
                logger=logger,
            )

            for p in chunk_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            try:
                os.rmdir(pc_tmpdir)
            except Exception:
                pass

            if not top_pairs:
                logger.info("No pairs meet min_frequency; stopping.")
                break

            merges_before = len(merges)
            new_tokens_created = 0
            for (a, b), cnt in top_pairs:
                if (a, b) in pair2newid:
                    continue
                s_new = token_str.get(a, "") + token_str.get(b, "")
                existing_id = token2id.get(s_new)
                if existing_id is not None:
                    pair2newid[(a, b)] = existing_id
                else:
                    nid = next_id
                    next_id += 1
                    pair2newid[(a, b)] = nid
                    token_str[nid] = s_new
                    token2id[s_new] = nid
                    new_tokens_created += 1
                merges.append((a, b))
                if len(merges) % 10000 == 0 or current_vocab_size_no_specials() >= target_vocab_size:
                    elapsed_total = time.time() - t_train_start
                    avg_rate = (len(merges) / max(1e-6, elapsed_total)) if elapsed_total > 0 else 0.0
                    merges_left_cum = max(0, (target_vocab_size - 256) - len(merges))
                    eta_min_cum = (merges_left_cum / max(1e-6, avg_rate)) / 60.0 if avg_rate > 0 else float('inf')
                    logger.info(f"Merged {human_int(len(merges))}/{human_int(target_vocab_size - 256)} | last=({a},{b})~{cnt} | mem={memory_percent():.1f}% | ETA~{eta_min_cum:.1f} min")

            accepted_pairs = len(merges) - merges_before
            duplicate_pairs = len(top_pairs) - accepted_pairs
            dt = time.time() - t0

            if accepted_pairs == 0 or new_tokens_created == 0:
                logger.warning(
                    f"Stagnation: new_tokens_created={new_tokens_created} accepted_pairs={accepted_pairs} "
                    f"(duplicates={duplicate_pairs}). Stopping early."
                )
                break

            logger.info(
                f"Batch {batch_num}: {dt:.1f}s | merges {human_int(len(merges))}/{human_int(target_vocab_size - 256)} | "
                f"accepted={accepted_pairs} dup={duplicate_pairs} | new_tokens={new_tokens_created} | "
                f"unique_types~{human_int(total_unique_seen)} | expanded_tokens~{human_int(total_expanded_tokens)} | mem={memory_percent():.1f}%"
            )
            gated_cleanup()

        # Hard fail if no merges learned
        if len(merges) == 0:
            try:
                # try to remove tmp_root
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
            except Exception:
                pass
            raise RuntimeError("No merges learned. Check counts format/content and training parameters.")

        final_vocab_no_specials = len(token2id)
        stats = {
            "target_vocab_size": int(vocab_size),
            "final_vocab_size": int(final_vocab_no_specials),
            "merges_learned": int(len(merges)),
            "min_frequency": int(min_frequency),
        }
        logger.info(f"Finished merges: {human_int(len(merges))} | Final vocab (no specials) ≈ {human_int(final_vocab_no_specials)}")

        # Cleanup tmp root
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
        except Exception:
            pass

        return merges, token_str, stats



# Saving artifacts

def save_tokenizer_package(outdir: str,
                           merges: List[Tuple[int, int]],
                           token_str: Dict[int, str],
                           vocab_size_target: int,
                           embedding_dim: int,
                           special_tokens: List[str],
                           seed: int,
                           logger: logging.Logger,
                           normalization: str = "NFKC",
                           strip_accents: bool = False,
                           strip: bool = False,
                           lowercase: bool = False) -> Tuple[Tokenizer, PreTrainedTokenizerFast]:
    os.makedirs(outdir, exist_ok=True)

    # Reserve specials first
    specials = list(dict.fromkeys(special_tokens))
    ordered_tokens: List[str] = specials.copy()
    seen = set(specials)
    for i in sorted(token_str.keys()):
        s = token_str[i]
        if s not in seen:
            ordered_tokens.append(s)
            seen.add(s)

    base_vocab = {tok: idx for idx, tok in enumerate(ordered_tokens)}

    # Write merges.txt and vocab.json atomically with fsync
    merges_txt = os.path.join(outdir, "merges.txt")
    vocab_json = os.path.join(outdir, "vocab.json")
    merges_str: List[Tuple[str, str]] = [(token_str[a], token_str[b]) for a, b in merges]

    m_fd, m_tmp = tempfile.mkstemp(prefix=".merges_", suffix=".txt", dir=outdir)
    os.close(m_fd)
    try:
        with open(m_tmp, "w", encoding="utf-8") as f:
            for a, b in merges_str:
                f.write(f"{a} {b}\n")
        try:
            with open(m_tmp, "rb") as _f:
                os.fsync(_f.fileno())
        except Exception:
            pass
        os.replace(m_tmp, merges_txt)
    except Exception:
        try:
            if os.path.exists(m_tmp):
                os.remove(m_tmp)
        except Exception:
            pass
        raise

    v_fd, v_tmp = tempfile.mkstemp(prefix=".vocab_", suffix=".json", dir=outdir)
    os.close(v_fd)
    try:
        with open(v_tmp, "w", encoding="utf-8") as f:
            json.dump(base_vocab, f, ensure_ascii=False, indent=2)
        try:
            with open(v_tmp, "rb") as _f:
                os.fsync(_f.fileno())
        except Exception:
            pass
        os.replace(v_tmp, vocab_json)
    except Exception:
        try:
            if os.path.exists(v_tmp):
                os.remove(v_tmp)
        except Exception:
            pass
        raise

    # Build the Tnex Tokenizer
    model = models.BPE(vocab=base_vocab,
                       merges=merges_str,
                       unk_token="[UNK]")
    tok = Tokenizer(model)
    norm = build_normalizer(normalization, bool(strip_accents), bool(strip), bool(lowercase))
    if norm:
        tok.normalizer = norm
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()  # Proper decoding of byte-level tokens

    try:
        tok.save(os.path.join(outdir, "tokenizer.json"))
    except Exception as e:
        logger.warning(f"Could not save tokenizer.json: {e}")

    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
        additional_special_tokens=[t for t in specials if t not in {"[PAD]","[UNK]","[BOS]","[EOS]","[MASK]"}],
    )
    hf.save_pretrained(outdir)

    info = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_vocab_size": int(vocab_size_target),
        "actual_vocab_entries": int(len(base_vocab)),
        "embedding_dim": int(embedding_dim),
        "special_tokens": specials,
        "byte_level": True,
        "add_prefix_space": False,
        "trim_offsets": False,
        "normalization": normalization,
        "strip_accents": bool(strip_accents),
        "strip": bool(strip),
        "lowercase": bool(lowercase),
    }
    with open(os.path.join(outdir, "tokenizer_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    rng = np.random.default_rng(seed)
    final_vs = int(len(base_vocab))
    emb = rng.standard_normal((final_vs, embedding_dim)).astype(np.float32)
    np.save(os.path.join(outdir, "embedding_matrix.npy"), emb)
    safetensors_save_file({"embeddings": emb}, os.path.join(outdir, "embedding_matrix.safetensors"))

    logger.info(f"Saved tokenizer artifacts to: {outdir}")
    return tok, hf



# Orchestrator

def train_from_counts(counts_path: str,
                      outdir: str,
                      vocab_size: int,
                      min_frequency: int,
                      embedding_dim: int,
                      seed: int,
                      log_level: str,
                      log_file: Optional[str],
                      normalization: str = "NFKC",
                      strip_accents: bool = False,
                      strip: bool = False,
                      lowercase: bool = False,
                      pair_chunk_limit: Optional[int] = None,
                      auto_pair_mem_fraction: Optional[float] = None,
                      pair_mem_threshold: int = 85,
                      batch_merges: int = 1000,
                      pair_workers: int = 1,
                      special_tokens: Optional[str] = None,
                      counts_format: str = "auto") -> None:
    logger = configure_logging(log_level, log_file)
    stop_event = threading.Event()
    install_signal_handler(stop_event, logger)

    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    try:
        if special_tokens:
            specials = [s.strip() for s in special_tokens.split(",") if s.strip()]
            if "[UNK]" not in specials:
                specials.insert(0, "[UNK]")
            specials = list(dict.fromkeys(specials))
        else:
            specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]

        bpe = OfflineBPE(
            logger,
            auto_pair_mem_fraction=auto_pair_mem_fraction,
            pair_chunk_limit=pair_chunk_limit,
            mem_threshold=pair_mem_threshold,
            batch_merges=batch_merges,
            counts_format=counts_format,
        )
        merges, token_str, _ = bpe.train(
            counts_path=counts_path,
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            pair_workers=int(pair_workers),
        )
        _tok, _hf = save_tokenizer_package(
            outdir=outdir,
            merges=merges,
            token_str=token_str,
            vocab_size_target=vocab_size,
            embedding_dim=embedding_dim,
            special_tokens=specials,
            seed=seed,
            logger=logger,
            normalization=normalization,
            strip_accents=strip_accents,
            strip=strip,
            lowercase=lowercase,
        )
        logger.info("Training pipeline complete.")
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Exiting.")
        sys.exit(130)
    except SystemExit as e:
        logger.warning(f"SystemExit requested: {e}. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        gated_cleanup()

# CLI

def main():
    p = argparse.ArgumentParser(
        description="Offline BPE training from counts (JSONL/TSV; compressed, plain, or streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--counts", type=str, required=True, help="Counts file path or FD (JSONL/TSV; .gz or plain; supports /proc/self/fd/*).")
    p.add_argument("--counts-format", type=str, default="auto", choices=["auto","json","tsv"], help="Override auto-detection if needed (useful for streams).")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--embedding-dim", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--log-file", type=str, default=None)

    # Normalization for multilingual
    p.add_argument("--normalization", type=str, default="NFKC", choices=["NONE","NFC","NFD","NFKC","NFKD"])
    p.add_argument("--strip-accents", action="store_true", default=False)
    p.add_argument("--strip", action="store_true", default=False)
    p.add_argument("--lowercase", action="store_true", default=False)

    # Offline BPE controls
    p.add_argument("--pair-chunk-limit", type=int, default=None, help="Max distinct pairs before spill.")
    p.add_argument("--auto-pair-mem-fraction", type=float, default=None, help="Fraction of RAM for pair map.")
    p.add_argument("--pair-mem-threshold", type=int, default=85, help="Percent RAM to trigger spill.")
    p.add_argument("--batch-merges", type=int, default=1000, help="Merges per pass.")
    p.add_argument("--pair-workers", type=int, default=1, help="Threads for pair counting.")

    # Specials
    p.add_argument("--special-tokens", type=str, default=None, help="Comma-separated specials; default [PAD,UNK,BOS,EOS,MASK]")

    args = p.parse_args()

    train_from_counts(
        counts_path=args.counts,
        outdir=args.outdir,
        vocab_size=int(args.vocab_size),
        min_frequency=int(args.min_frequency),
        embedding_dim=int(args.embedding_dim),
        seed=int(args.seed),
        log_level=args.log_level,
        log_file=args.log_file,
        normalization=args.normalization,
        strip_accents=bool(args.strip_accents),
        strip=bool(args.strip),
        lowercase=bool(args.lowercase),
        pair_chunk_limit=args.pair_chunk_limit,
        auto_pair_mem_fraction=args.auto_pair_mem_fraction,
        pair_mem_threshold=int(args.pair_mem_threshold),
        batch_merges=int(args.batch_merges),
        pair_workers=int(args.pair_workers),
        special_tokens=args.special_tokens,
        counts_format=args.counts_format,
    )


if __name__ == "__main__":
    main()
