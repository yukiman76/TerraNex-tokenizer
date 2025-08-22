
import os
import sys
import gc
import csv
import gzip
import json
import time
import heapq
import psutil
import signal
import logging
import argparse
import threading
import queue
import hashlib
import tempfile
import traceback
from typing import Dict, List, Tuple, Optional, Iterable, Any
from contextlib import ExitStack
from collections import deque

# Module-level constants
DEFAULT_SPILL_BYTES_LIMIT = 256 * 1024 * 1024  # 256 MiB per worker safety cap
MAX_FANIN_OPEN_FILES = 512                     # cap to avoid too many FDs
MAX_FLUSH_INDEX = 10_000_000                   # hard cap on number of partial flushes
FAILED_FILES_MAX_KEEP = 10_000                 # keep last N failed file paths

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)


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
    return logging.getLogger("scan")


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


def install_signal_handler(stop_event: threading.Event, logger: logging.Logger):
    def handler(signum, frame):
        stop_event.set()
        try:
            logger.warning(f"Received signal {signum}, requesting shutdown...")
        except Exception:
            pass
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


def blake2b16(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def bytes_to_keystr(b: bytes) -> str:
    return " ".join(str(x) for x in b)


def keystr_to_bytes(s: str) -> bytes:
    return bytes(int(x) for x in s.split(" ") if x)


def list_manifest_files(manifest_csv: str) -> List[str]:
    out: List[str] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames or "file_path" not in rdr.fieldnames:
            raise ValueError("manifest CSV must have 'file_path' column")
        for row in rdr:
            p = (row.get("file_path") or "").strip()
            if p:
                out.append(p)
    return out


def shard_list_deterministic(items: List[str], shard_index: int, shard_count: int) -> List[str]:
    if shard_count <= 1:
        return sorted(items)
    items_sorted = sorted(items)
    return [p for i, p in enumerate(items_sorted) if (i % shard_count) == shard_index]


def pick_text_column(pf: pq.ParquetFile, preferred: List[str]) -> Optional[str]:
    schema = pf.schema_arrow
    for name in preferred:
        idx = schema.get_field_index(name)
        if idx != -1 and (pa.types.is_string(schema[idx].type) or pa.types.is_large_string(schema[idx].type)):
            return name
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            return field.name
    return None


def iter_parquet_texts(file_path: str,
                       text_columns_priority: List[str],
                       parquet_use_threads: bool,
                       stop_event: threading.Event,
                       logger: logging.Logger,
                       batch_size: int = 65536) -> Iterable[str]:
    try:
        with pq.ParquetFile(file_path) as pf:
            text_col = pick_text_column(pf, text_columns_priority)
            if text_col is None:
                logger.warning(f"No string column in: {file_path}")
                return
            for batch in pf.iter_batches(columns=[text_col],
                                         use_threads=parquet_use_threads,
                                         batch_size=batch_size):
                if stop_event.is_set():
                    return
                col = batch.column(0)
                if not (pa.types.is_string(col.type) or pa.types.is_large_string(col.type)):
                    try:
                        col = col.cast(pa.string())
                    except Exception:
                        continue
                for i in range(len(col)):
                    if stop_event.is_set():
                        return
                    v = col[i].as_py()
                    if isinstance(v, str) and v:
                        yield v
    except FileNotFoundError as e:
        logger.warning(f"File not found: {file_path} -> {e}")
    except Exception as e:
        logger.error(f"Open/read error: {file_path} -> {e}")


def _shard_tmpdir_for(out_counts_path: str) -> str:
    base = os.path.dirname(out_counts_path) or "."
    name = os.path.basename(out_counts_path)
    tag = blake2b16(out_counts_path)
    return os.path.join(base, f".scan_tmp_{name}.{tag}")


def _append_flush_chunk(tmpdir: str, chunk_idx: int, local_map: Dict[bytes, int], logger: logging.Logger) -> str:
    os.makedirs(tmpdir, exist_ok=True)
    final_path = os.path.join(tmpdir, f"partial_{chunk_idx:05d}.tsv.gz")
    fd, tmppath = tempfile.mkstemp(prefix=".partial_", suffix=".tsv.gz", dir=tmpdir)
    try:
        try:
            os.close(fd)
        except Exception:
            pass
        sorted_keys = sorted(local_map.keys(), key=bytes_to_keystr)
        with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=1) as f:
            for k in sorted_keys:
                f.write(f"{bytes_to_keystr(k)}\t{local_map[k]}\n")
        try:
            with open(tmppath, "rb") as _f:
                os.fsync(_f.fileno())
        except Exception:
            pass
        os.replace(tmppath, final_path)
        logger.info(f"Shard flush → {final_path} (records: {human_int(len(local_map))}) | Memory: {memory_percent():.1f}%")
        return final_path
    except Exception as e:
        try:
            if os.path.exists(tmppath):
                os.remove(tmppath)
        except Exception:
            pass
        logger.warning(f"Flush failed for chunk {chunk_idx}: {e}")
        raise


def _read_tsv_line(fh) -> Tuple[Optional[Tuple[str, int]], bool]:
    try:
        line = fh.readline()
    except Exception:
        return None, True  # IO error → treat as EOF
    if not line:
        return None, True
    try:
        k, c = line.rstrip("\n").split("\t")
        keystr_to_bytes(k)  # validate
        return (k, int(c)), False
    except Exception:
        return None, False  # malformed


def _kmerge_tsv_to_json(in_paths: List[str], out_path: str, logger: logging.Logger) -> int:
    def open_tsv(p):
        return gzip.open(p, "rt", encoding="utf-8")
    out_total = 0
    with ExitStack() as stack:
        iters: List[Any] = []
        for p in in_paths:
            try:
                iters.append(stack.enter_context(open_tsv(p)))
            except Exception as e:
                logger.warning(f"Could not open partial {p}: {e}")
        if not iters:
            logger.warning("No readable partials; nothing to merge.")
            return 0
        heads: List[Optional[Tuple[str, int]]] = []
        heap: List[Tuple[str, int]] = []
        malformed = 0
        for idx, fh in enumerate(iters):
            rec, eof = _read_tsv_line(fh)
            while rec is None and not eof:
                malformed += 1
                rec, eof = _read_tsv_line(fh)
            heads.append(rec)
            if rec is not None:
                heap.append((rec[0], idx))
        heapq.heapify(heap)
        fd, tmppath = tempfile.mkstemp(prefix=".final_", suffix=".jsonl.gz", dir=os.path.dirname(out_path) or ".")
        try:
            try:
                os.close(fd)
            except Exception:
                pass
            with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=1) as out:
                while heap:
                    k, src = heapq.heappop(heap)
                    total = 0
                    while True:
                        rec = heads[src]
                        if rec is not None and rec[0] == k:
                            total += rec[1]
                            nxt, eof = _read_tsv_line(iters[src])
                            while nxt is None and not eof:
                                malformed += 1
                                nxt, eof = _read_tsv_line(iters[src])
                            heads[src] = nxt
                            if nxt is not None:
                                heapq.heappush(heap, (nxt[0], src))
                        if heap and heap[0][0] == k:
                            _, src2 = heapq.heappop(heap)
                            src = src2
                            continue
                        break
                    out.write(json.dumps({"seq": k, "count": int(total)}, ensure_ascii=False) + "\n")
                    out_total += 1
            try:
                with open(tmppath, "rb") as _f:
                    os.fsync(_f.fileno())
            except Exception:
                pass
            os.replace(tmppath, out_path)
        except Exception as e:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            raise e
        if malformed > 0:
            logger.warning(f"Merge encountered {malformed} malformed lines that were skipped.")
    return out_total


def _kmerge_tsv_to_tsv(in_paths: List[str], out_path: str, logger: logging.Logger) -> int:
    def open_tsv(p):
        return gzip.open(p, "rt", encoding="utf-8")
    out_total = 0
    with ExitStack() as stack:
        iters: List[Any] = []
        for p in in_paths:
            try:
                iters.append(stack.enter_context(open_tsv(p)))
            except Exception as e:
                logger.warning(f"Could not open partial {p}: {e}")
        if not iters:
            logger.warning("No readable partials for intermediate merge.")
            return 0
        heads: List[Optional[Tuple[str, int]]] = []
        heap: List[Tuple[str, int]] = []
        malformed = 0
        for idx, fh in enumerate(iters):
            rec, eof = _read_tsv_line(fh)
            while rec is None and not eof:
                malformed += 1
                rec, eof = _read_tsv_line(fh)
            heads.append(rec)
            if rec is not None:
                heap.append((rec[0], idx))
        heapq.heapify(heap)
        fd, tmppath = tempfile.mkstemp(prefix=".merge_", suffix=".tsv.gz", dir=os.path.dirname(out_path) or ".")
        try:
            try:
                os.close(fd)
            except Exception:
                pass
            with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=1) as out:
                while heap:
                    k, src = heapq.heappop(heap)
                    total = 0
                    while True:
                        rec = heads[src]
                        if rec is not None and rec[0] == k:
                            total += rec[1]
                            nxt, eof = _read_tsv_line(iters[src])
                            while nxt is None and not eof:
                                malformed += 1
                                nxt, eof = _read_tsv_line(iters[src])
                            heads[src] = nxt
                            if nxt is not None:
                                heapq.heappush(heap, (nxt[0], src))
                        if heap and heap[0][0] == k:
                            _, src2 = heapq.heappop(heap)
                            src = src2
                            continue
                        break
                    out.write(f"{k}\t{int(total)}\n")
                    out_total += 1
            try:
                with open(tmppath, "rb") as _f:
                    os.fsync(_f.fileno())
            except Exception:
                pass
            os.replace(tmppath, out_path)
        except Exception as e:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            raise e
        if malformed > 0:
            logger.warning(f"Intermediate merge encountered {malformed} malformed lines that were skipped.")
    return out_total


def _finalize_shard_partials(tmpdir: str, out_counts_path: str, logger: logging.Logger, finalize_fanin: int = 1024):
    if not out_counts_path.endswith(".jsonl.gz"):
        logger.warning(f"Output path does not end with .jsonl.gz: {out_counts_path}. Proceeding anyway.")
    if os.path.exists(out_counts_path):
        if os.path.isdir(tmpdir):
            try:
                parts = [
                    os.path.join(tmpdir, n) for n in os.listdir(tmpdir)
                    if (n.endswith(".tsv.gz") and (n.startswith("partial_") or n.startswith("merge_r")))
                ]
                removed = 0
                for p in parts:
                    try:
                        os.remove(p)
                        removed += 1
                    except Exception:
                        pass
                try:
                    os.rmdir(tmpdir)
                except Exception:
                    pass
                logger.info(f"Shard finalize: {out_counts_path} already exists; cleaned {removed} stale partials in {tmpdir}.")
            except Exception as e:
                logger.warning(f"Shard finalize: cleanup of {tmpdir} failed: {e}")
        else:
            logger.info(f"Shard finalize: {out_counts_path} already exists; skipping finalize.")
        return
    if not os.path.isdir(tmpdir):
        logger.warning(f"Shard finalize: tmpdir missing: {tmpdir}")
        return
    parts_all = sorted([
        os.path.join(tmpdir, n) for n in os.listdir(tmpdir)
        if n.startswith("partial_") and n.endswith(".tsv.gz")
    ])
    parts: List[str] = []
    for p in parts_all:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                _ = f.readline()
            parts.append(p)
        except Exception as e:
            logger.warning(f"Shard finalize: skipping corrupt partial {p}: {e}")
    if not parts:
        logger.warning(f"Shard finalize: no valid partials in {tmpdir}.")
        return
    fanin = min(finalize_fanin, MAX_FANIN_OPEN_FILES)
    if finalize_fanin > MAX_FANIN_OPEN_FILES:
        logger.info(f"Capping finalize fan-in to {MAX_FANIN_OPEN_FILES} open files per round for safety.")
    worklist = parts[:]
    intermediates: List[str] = []
    round_idx = 0
    while len(worklist) > fanin:
        new_worklist: List[str] = []
        for i in range(0, len(worklist), fanin):
            batch = worklist[i:i + fanin]
            outp = os.path.join(tmpdir, f"merge_r{round_idx}_{i//fanin:05d}.tsv.gz")
            try:
                _kmerge_tsv_to_tsv(batch, outp, logger)
                new_worklist.append(outp)
                intermediates.append(outp)
            except Exception as e:
                logger.error(f"Intermediate merge failed: {e}")
                raise
        worklist = new_worklist
        round_idx += 1
    try:
        _ = _kmerge_tsv_to_json(worklist, out_counts_path, logger)
    except Exception as e:
        logger.error(f"Finalize failed: {e}")
        raise
    removed = 0
    for p in parts + intermediates:
        try:
            os.remove(p)
            removed += 1
        except Exception:
            pass
    try:
        os.rmdir(tmpdir)
    except Exception:
        pass
    logger.info(f"Shard finalize: wrote {out_counts_path}. Cleaned {removed} partials.")


def resolve_shard_index(auto_index: bool, shard_index: int) -> int:
    if auto_index:
        v = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("SHARD_INDEX"))
        if v is not None and v.isdigit():
            return int(v)
    return shard_index


def validate_args(args: argparse.Namespace):
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be > 0")
    if not args.auto_index:
        if args.shard_index < 0 or args.shard_index >= args.shard_count:
            raise ValueError("--shard-index must be in [0, shard-count)")
    for name in ["num_workers", "parquet_batch_size", "scan_unique_cap", "finalize_fanin"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be > 0")
    if not (0 <= args.scan_mem_threshold <= 100):
        raise ValueError("--scan-mem-threshold must be between 0 and 100")
    if args.spill_bytes_limit <= 0:
        raise ValueError("--spill-bytes-limit must be > 0")
    if not args.text_columns.strip():
        raise ValueError("--text-columns must be non-empty")
    if not args.out_counts.strip():
        raise ValueError("--out-counts must be non-empty")
    if not args.manifest_csv.strip():
        raise ValueError("--manifest-csv must be non-empty")


def scan_shard_to_counts(manifest_csv: str,
                         text_columns: List[str],
                         out_counts_path: str,
                         shard_index: int,
                         shard_count: int,
                         num_workers: int,
                         parquet_use_threads: bool,
                         parquet_batch_size: int,
                         resume: bool,
                         scan_flush_every_files: int,  # deprecated
                         scan_flush_every_seconds: int,
                         scan_unique_cap: int,
                         scan_mem_threshold: int,
                         spill_bytes_limit: int,
                         finalize_fanin: int,
                         log_level: str,
                         log_file: Optional[str]) -> None:
    logger = configure_logging(log_level, log_file)
    stop_event = threading.Event()
    install_signal_handler(stop_event, logger)
    try:
        os.makedirs(os.path.dirname(out_counts_path) or ".", exist_ok=True)
    except Exception:
        pass
    tmpdir = _shard_tmpdir_for(out_counts_path)
    if resume and os.path.exists(out_counts_path):
        try:
            opener = gzip.open if out_counts_path.endswith(".gz") else open
            with opener(out_counts_path, "rt", encoding="utf-8") as f:
                _ = f.readline()
            logger.info(f"Resume on: counts exists and readable → {out_counts_path}. Skipping shard {shard_index}.")
            if os.path.isdir(tmpdir):
                try:
                    parts = [os.path.join(tmpdir, n) for n in os.listdir(tmpdir)
                             if n.endswith(".tsv.gz") and (n.startswith("partial_") or n.startswith("merge_r"))]
                    removed = 0
                    for p in parts:
                        try:
                            os.remove(p)
                            removed += 1
                        except Exception:
                            pass
                    try:
                        os.rmdir(tmpdir)
                    except Exception:
                        pass
                    logger.info(f"Resume cleanup: removed {removed} partials in {tmpdir}.")
                except Exception as e:
                    logger.warning(f"Resume cleanup failed for {tmpdir}: {e}")
            return
        except Exception as e:
            logger.warning(f"Resume probe failed on {out_counts_path}: {e}. Re-finalizing if needed.")
    files_all = list_manifest_files(manifest_csv)
    files = shard_list_deterministic(files_all, shard_index, shard_count)
    logger.info(f"Scanning shard {shard_index}/{shard_count-1}: {len(files)} files | tmpdir={tmpdir}")
    q: "queue.Queue[str]" = queue.Queue()
    for fp in files:
        q.put(fp)

    total_wp = 0
    total_wp_bytes = 0
    processed_files = 0
    failed_files = 0
    failed_file_names: deque[str] = deque(maxlen=FAILED_FILES_MAX_KEEP)

    counters_lock = threading.Lock()
    progress_lock = threading.Lock()
    flush_lock = threading.Lock()
    flush_index_lock = threading.Lock()

    with flush_index_lock:
        flush_idx = 0
        if resume and os.path.isdir(tmpdir):
            existing = len([n for n in os.listdir(tmpdir) if n.startswith("partial_")])
            if existing >= MAX_FLUSH_INDEX:
                raise RuntimeError(f"Existing partials {existing} exceed MAX_FLUSH_INDEX={MAX_FLUSH_INDEX}")
            flush_idx = existing
            logger.info(f"Resume: found {existing} partial chunks in {tmpdir}; next flush index = {flush_idx:05d}")

    progress_stop = threading.Event()

    def progress_reporter():
        last_logged = 0.0
        while not progress_stop.is_set():
            time.sleep(5.0)
            with progress_lock:
                pf = processed_files
                ff = failed_files
                mem = memory_percent()  # read under same lock for consistency
            if pf != 0 or (time.time() - last_logged) >= 10.0:
                logger.info(f"Progress: processed {pf}/{len(files)} files, failed {ff} | Memory: {mem:.1f}%")
                last_logged = time.time()

    reporter = threading.Thread(target=progress_reporter, name="progress", daemon=False)
    reporter.start()

    def spill_local_map(local_map: Dict[bytes, int]):
        nonlocal flush_idx
        if not local_map:
            return
        with flush_lock:
            with flush_index_lock:
                if flush_idx >= MAX_FLUSH_INDEX:
                    raise RuntimeError(f"flush_idx reached MAX_FLUSH_INDEX={MAX_FLUSH_INDEX}. Aborting to avoid unbounded partials.")
                idx = flush_idx
                flush_idx += 1
            _append_flush_chunk(tmpdir, idx, local_map, logger)

    def worker_loop(wid: int):
        nonlocal total_wp, total_wp_bytes, processed_files, failed_files
        local_map: Dict[bytes, int] = {}
        local_rows = 0
        local_wp = 0
        local_wp_bytes = 0
        local_bad_encodes = 0
        local_cap = max(100_000, scan_unique_cap // max(1, num_workers))
        last_spill = time.time()
        while not stop_event.is_set():
            file_retrieved = False
            fp = None
            try:
                fp = q.get_nowait()
                file_retrieved = True
            except queue.Empty:
                break
            try:
                for text in iter_parquet_texts(fp, text_columns, parquet_use_threads, stop_event, logger, batch_size=parquet_batch_size):
                    try:
                        tb = text.encode("utf-8")
                    except Exception:
                        local_bad_encodes += 1
                        continue
                    if not tb:
                        continue
                    local_map[tb] = local_map.get(tb, 0) + 1
                    local_wp += 1
                    local_wp_bytes += len(tb)
                    local_rows += 1
                    if (local_rows % 2048) == 0:
                        if (len(local_map) >= local_cap) or \
                           (scan_flush_every_seconds > 0 and (time.time() - last_spill) >= scan_flush_every_seconds) or \
                           (memory_percent() >= float(scan_mem_threshold)) or \
                           (local_wp_bytes >= spill_bytes_limit):
                            spill_local_map(local_map)
                            local_map = {}
                            last_spill = time.time()
                            with counters_lock:
                                total_wp += local_wp
                                total_wp_bytes += local_wp_bytes
                            local_wp = 0
                            local_wp_bytes = 0
                            gated_cleanup()
            except Exception as e:
                logger.warning(f"Worker {wid} file error on {fp}: {e}")
                with progress_lock:
                    failed_files += 1
                    if fp is not None:
                        failed_file_names.append(str(fp))
            finally:
                if local_bad_encodes > 0:
                    logger.warning(f"Worker {wid}: skipped {local_bad_encodes} records with UTF-8 encode errors.")
                    local_bad_encodes = 0
                if local_map:
                    spill_local_map(local_map)
                    local_map = {}
                    with counters_lock:
                        total_wp += local_wp
                        total_wp_bytes += local_wp_bytes
                    local_wp = 0
                    local_wp_bytes = 0
                with progress_lock:
                    processed_files += 1
                if file_retrieved:
                    try:
                        q.task_done()
                    except Exception:
                        pass
                gated_cleanup()
        if local_map:
            spill_local_map(local_map)
            with counters_lock:
                total_wp += local_wp
                total_wp_bytes += local_wp_bytes

    threads: List[threading.Thread] = []
    for i in range(max(1, num_workers)):
        t = threading.Thread(target=worker_loop, args=(i,), name=f"scan-{i}", daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    progress_stop.set()
    reporter.join(timeout=2.0)

    # Persist failed file list (bounded)
    if failed_files > 0 and len(failed_file_names) > 0:
        try:
            fail_list_path = f"{out_counts_path}.failed.txt"
            with open(fail_list_path, "w", encoding="utf-8") as fh:
                for pth in failed_file_names:
                    fh.write(pth + "\n")
            logger.warning(f"Wrote list of {len(failed_file_names)} failed files to: {fail_list_path} (total failures counted: {failed_files})")
        except Exception as e:
            logger.warning(f"Could not write failed files list: {e}")

    logger.info(f"Scan complete. Counted {human_int(total_wp)} sequences, bytes={human_int(total_wp_bytes)}. Partials in: {tmpdir}")
    _finalize_shard_partials(tmpdir, out_counts_path, logger, finalize_fanin=finalize_fanin)


def main():
    p = argparse.ArgumentParser(
        description="Shard scanner: Parquet → raw byte sequence counts with per-worker spill, resume, and finalize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest-csv", type=str, required=True, help="CSV with 'file_path' column.")
    p.add_argument("--text-columns", type=str, required=True, help="Comma-separated list, e.g. 'content,sentence,text'")
    p.add_argument("--out-counts", type=str, required=True, help="Output counts JSONL.GZ for this shard.")
    p.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based). Ignored if --auto-index.")
    p.add_argument("--shard-count", type=int, default=1, help="Total shards.")
    p.add_argument("--auto-index", action="store_true", default=False, help="Derive shard index from SLURM_ARRAY_TASK_ID or $SHARD_INDEX.")
    p.add_argument("--resume", action="store_true", default=False, help="Skip if --out-counts exists; also reuse shard partials.")
    p.add_argument("--num-workers", type=int, default=8, help="Scanner threads.")
    p.add_argument("--parquet-use-threads", action="store_true", default=False, help="Enable Arrow internal threads.")
    p.add_argument("--parquet-batch-size", type=int, default=65536, help="Parquet iter_batches size.")
    p.add_argument("--scan-flush-every-files", type=int, default=1, help="[deprecated] No effect; spills occur at file end.")
    p.add_argument("--scan-flush-every-seconds", type=int, default=60, help="Spill if seconds elapsed since last spill (per worker).")
    p.add_argument("--scan-unique-cap", type=int, default=2_000_000, help="Approx max distinct keys in a worker map before spill.")
    p.add_argument("--scan-mem-threshold", type=int, default=85, help="Percent RAM usage to trigger immediate spill.")
    p.add_argument("--spill-bytes-limit", type=int, default=DEFAULT_SPILL_BYTES_LIMIT, help="Per-worker bytes before forced spill.")
    p.add_argument("--finalize-fanin", type=int, default=1024, help="Max files merged per round; may be capped for safety.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Logging level.")
    p.add_argument("--log-file", type=str, default=None, help="Optional log file.")
    args = p.parse_args()
    try:
        validate_args(args)
        si = resolve_shard_index(bool(args.auto_index), int(args.shard_index))
        scan_shard_to_counts(
            manifest_csv=args.manifest_csv,
            text_columns=[c.strip() for c in args.text_columns.split(",") if c.strip()],
            out_counts_path=args.out_counts,
            shard_index=si,
            shard_count=int(args.shard_count),
            num_workers=int(args.num_workers),
            parquet_use_threads=bool(args.parquet_use_threads),
            parquet_batch_size=int(args.parquet_batch_size),
            resume=bool(args.resume),
            scan_flush_every_files=int(args.scan_flush_every_files),  # deprecated
            scan_flush_every_seconds=int(args.scan_flush_every_seconds),
            scan_unique_cap=int(args.scan_unique_cap),
            scan_mem_threshold=int(args.scan_mem_threshold),
            spill_bytes_limit=int(args.spill_bytes_limit),
            finalize_fanin=int(args.finalize_fanin),
            log_level=args.log_level,
            log_file=args.log_file,
        )
    except KeyboardInterrupt:
        logging.getLogger("scan").warning("KeyboardInterrupt received. Exiting.")
        sys.exit(130)
    except Exception as e:
        logging.getLogger("scan").error(f"Fatal: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        gated_cleanup()


if __name__ == "__main__":
    main()
