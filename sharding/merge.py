import os
import sys
import gc
import gzip
import json
import psutil
import signal
import logging
import argparse
import tempfile
import traceback
import resource
import heapq
import time
from contextlib import ExitStack
from typing import Dict, List, Tuple, Optional, Any


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
    return logging.getLogger("merge")


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


def install_signal_handler(stop_event: Dict[str, bool], logger: logging.Logger):
    def handler(signum, frame):
        logger.warning(f"Received signal {signum}, requesting shutdown...")
        stop_event["stop"] = True
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


def safe_fanin(requested: int) -> int:
    """
    Cap fan-in by RLIMIT_NOFILE with a conservative fallback.
    """
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        cap = max(64, int(soft * 0.6))
        return max(2, min(requested, cap))
    except Exception:
        return min(512, max(2, requested))  # conservative fallback


def cleanup_tmpdir(tmpdir: str, logger: logging.Logger):
    if not os.path.isdir(tmpdir):
        return
    try:
        for n in os.listdir(tmpdir):
            p = os.path.join(tmpdir, n)
            try:
                os.remove(p)
            except Exception:
                pass
        os.rmdir(tmpdir)
    except Exception as e:
        logger.warning(f"Cleanup failed for tmpdir {tmpdir}: {e}")


def _create_empty_output(out_path: str, out_format: str, logger: logging.Logger):
    """
    Produce a valid empty output artifact for empty inputs.
    """
    try:
        opener = gzip.open if out_path.endswith(".gz") else open
        mode = "wt"
        with opener(out_path, mode, encoding="utf-8") as f:
            pass  # empty file is valid
        logger.info(f"No data to merge; created empty {out_format.upper()} output: {out_path}")
    except Exception as e:
        logger.error(f"Failed to create empty output {out_path}: {e}")
        raise


# Inputs

def merge_counts_dir(counts_dir: str) -> List[str]:
    if not os.path.isdir(counts_dir):
        raise ValueError(f"Not a directory: {counts_dir}")
    files: List[str] = []
    for name in sorted(os.listdir(counts_dir)):
        if name.startswith("counts_shard") and (
            name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".tsv.gz") or name.endswith(".tsv")
        ):
            files.append(os.path.join(counts_dir, name))
    return files


# K-way merge helpers

def _kmerge_tsv_to_tsv(in_paths: List[str],
                       out_path: str,
                       logger: logging.Logger,
                       stop_event: Dict[str, bool]) -> int:
    """
    Merge multiple sorted TSV(.gz) {key\tcount} into a single TSV.GZ with aggregated counts.
    """
    def read_line(fh) -> Tuple[Optional[Tuple[str, int]], bool]:
        try:
            line = fh.readline()
        except Exception as e:
            logger.error(f"Read error during k-merge TSV: {e}")
            return None, True
        if not line:
            return None, False
        try:
            k, c = line.rstrip("\n").split("\t")
            return (k, int(c)), False
        except Exception as e:
            logger.warning(f"Bad TSV line skipped: {e}")
            return None, False

    out_total = 0
    fd, tmppath = tempfile.mkstemp(prefix=".kmerge_", suffix=".tsv.gz", dir=os.path.dirname(out_path) or ".")
    os.close(fd)

    with ExitStack() as stack:
        iters: List[Any] = []
        for p in in_paths:
            try:
                iters.append(stack.enter_context(gzip.open(p, "rt", encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Could not open partial {p}: {e}")

        heads: List[Optional[Tuple[str, int]]] = []
        heap: List[Tuple[str, int]] = []

        for idx, fh in enumerate(iters):
            rec, err = read_line(fh)
            if err:
                raise IOError("I/O error in TSV input during priming")
            heads.append(rec)
            if rec is not None:
                heap.append((rec[0], idx))
        heapq.heapify(heap)

        try:
            with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=1) as out:
                while heap:
                    if stop_event.get("stop", False):
                        logger.info("Merge interrupted by signal (TSV->TSV).")
                        raise KeyboardInterrupt()
                    k, src = heapq.heappop(heap)
                    total = 0
                    while True:
                        rec = heads[src]
                        if rec is not None and rec[0] == k:
                            total += rec[1]
                            nxt, err = read_line(iters[src])
                            if err:
                                raise IOError("I/O error in TSV input during merge")
                            heads[src] = nxt
                            if heads[src] is not None:
                                heapq.heappush(heap, (heads[src][0], src))
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
            except Exception as e:
                logger.warning(f"fsync failed (data may not be durable): {e}")
            os.replace(tmppath, out_path)
        except Exception:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            raise

    return out_total


def _kmerge_tsv_to_json(in_paths: List[str],
                        out_path: str,
                        logger: logging.Logger,
                        stop_event: Dict[str, bool]) -> int:
    """
    Merge multiple sorted TSV(.gz) {key\tcount} into a JSONL.GZ with {"seq": key, "count": int}.
    """
    def read_line(fh) -> Tuple[Optional[Tuple[str, int]], bool]:
        try:
            line = fh.readline()
        except Exception as e:
            logger.error(f"Read error during k-merge TSV: {e}")
            return None, True
        if not line:
            return None, False
        try:
            k, c = line.rstrip("\n").split("\t")
            return (k, int(c)), False
        except Exception as e:
            logger.warning(f"Bad TSV line skipped: {e}")
            return None, False

    out_total = 0
    fd, tmppath = tempfile.mkstemp(prefix=".kmerge_", suffix=".jsonl.gz", dir=os.path.dirname(out_path) or ".")
    os.close(fd)

    with ExitStack() as stack:
        iters: List[Any] = []
        for p in in_paths:
            try:
                iters.append(stack.enter_context(gzip.open(p, "rt", encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Could not open partial {p}: {e}")

        heads: List[Optional[Tuple[str, int]]] = []
        heap: List[Tuple[str, int]] = []

        for idx, fh in enumerate(iters):
            rec, err = read_line(fh)
            if err:
                raise IOError("I/O error in TSV input during priming")
            heads.append(rec)
            if rec is not None:
                heap.append((rec[0], idx))
        heapq.heapify(heap)

        try:
            with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=1) as out:
                while heap:
                    if stop_event.get("stop", False):
                        logger.info("Merge interrupted by signal (TSV->JSON).")
                        raise KeyboardInterrupt()
                    k, src = heapq.heappop(heap)
                    total = 0
                    while True:
                        rec = heads[src]
                        if rec is not None and rec[0] == k:
                            total += rec[1]
                            nxt, err = read_line(iters[src])
                            if err:
                                raise IOError("I/O error in TSV input during merge")
                            heads[src] = nxt
                            if heads[src] is not None:
                                heapq.heappush(heap, (heads[src][0], src))
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
            except Exception as e:
                logger.warning(f"fsync failed (data may not be durable): {e}")
            os.replace(tmppath, out_path)
        except Exception:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            raise

    return out_total


# Merge driver

def merge_counts(in_paths: List[str],
                 out_path: str,
                 count_chunk_limit: int,
                 resume_merge: bool,
                 merge_fanin: int,
                 out_format: str,
                 log_level: str,
                 log_file: Optional[str],
                 stop_event: Optional[Dict[str, bool]] = None) -> None:
    """
    Merge multiple counts files (JSONL[.gz] or TSV[.gz]) into a single output.
    Exit codes: 0=success, 130=interrupted, 2=usage/data error, 1=unexpected error.
    """
    logger = configure_logging(log_level, log_file)
    stop = stop_event or {"stop": False}

    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    except Exception:
        pass

    if not in_paths:
        logger.error("No input counts provided to merge.")
        sys.exit(2)

    # Resume check on final artifact
    if resume_merge and os.path.exists(out_path):
        try:
            opener = gzip.open if out_path.endswith(".gz") else open
            with opener(out_path, "rt", encoding="utf-8") as f:
                _ = f.readline()
            logger.info(f"Resume-merge: final output exists and is readable ({out_path}); skipping merge.")
            # Clean any stale tmpdir from previous attempts
            tmpdir_stale = os.path.join(os.path.dirname(out_path) or ".", f".merge_tmp_{os.path.basename(out_path)}")
            cleanup_tmpdir(tmpdir_stale, logger)
            return
        except Exception:
            logger.warning("Existing output is unreadable; redoing merge.")

    # Unique tmpdir per run to avoid races
    base_tmp = f".merge_tmp_{os.path.basename(out_path)}"
    tmpdir = os.path.join(os.path.dirname(out_path) or ".", f"{base_tmp}.{os.getpid()}.{int(time.time())}")
    os.makedirs(tmpdir, exist_ok=True)

    chunk_files: List[str] = []
    cur: Dict[str, int] = {}
    chunk_idx = 0
    total_lines = 0

    def spill_chunk(idx: int):
        if not cur:
            return
        path_final = os.path.join(tmpdir, f"chunk_{idx:06d}.tsv.gz")
        fd, tmppath = tempfile.mkstemp(prefix=".chunk_", suffix=".tsv.gz", dir=tmpdir)
        os.close(fd)
        try:
            with gzip.open(tmppath, "wt", encoding="utf-8", compresslevel=1) as f:
                for k, v in sorted(cur.items()):
                    f.write(f"{k}\t{v}\n")
            try:
                with open(tmppath, "rb") as _f:
                    os.fsync(_f.fileno())
            except Exception as e:
                logger.warning(f"fsync failed (data may not be durable): {e}")
            os.replace(tmppath, path_final)
            chunk_files.append(path_final)
            cur.clear()
            logger.info(f"Spilled chunk #{idx} -> {path_final} | Memory: {memory_percent():.1f}%")
        except Exception as e:
            try:
                if os.path.exists(tmppath):
                    os.remove(tmppath)
            except Exception:
                pass
            logger.error(f"Failed spilling chunk {idx}: {e}")
            raise

    # Phase 1: read inputs, aggregate into sorted TSV chunks with RAM cap
    for ip in in_paths:
        if stop.get("stop", False):
            logger.info("Merge interrupted by signal during input scan.")
            cleanup_tmpdir(tmpdir, logger)
            sys.exit(130)

        is_gz = ip.endswith(".gz")
        opener = gzip.open if is_gz else open
        is_tsv = ip.endswith(".tsv") or ip.endswith(".tsv.gz")

        malformed_lines = 0

        try:
            with opener(ip, "rt", encoding="utf-8") as f:
                if is_tsv:
                    for line in f:
                        if stop.get("stop", False):
                            logger.info("Merge interrupted by signal during TSV read.")
                            cleanup_tmpdir(tmpdir, logger)
                            sys.exit(130)
                        if not line.strip():
                            continue
                        try:
                            k, c = line.rstrip("\n").split("\t")
                            cur[k] = cur.get(k, 0) + int(c)
                            total_lines += 1
                        except Exception:
                            malformed_lines += 1
                            continue
                        if memory_percent() >= 85.0 or len(cur) >= count_chunk_limit:
                            spill_chunk(chunk_idx)
                            chunk_idx += 1
                else:
                    for line in f:
                        if stop.get("stop", False):
                            logger.info("Merge interrupted by signal during JSON read.")
                            cleanup_tmpdir(tmpdir, logger)
                            sys.exit(130)
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            k = obj.get("seq")
                            if k is None:
                                malformed_lines += 1
                                continue
                            c = int(obj.get("count", 0))
                        except Exception:
                            malformed_lines += 1
                            continue
                        cur[k] = cur.get(k, 0) + c
                        total_lines += 1
                        if memory_percent() >= 85.0 or len(cur) >= count_chunk_limit:
                            spill_chunk(chunk_idx)
                            chunk_idx += 1
        except Exception as e:
            logger.warning(f"Read error {ip}: {e}")

        if malformed_lines > 0:
            logger.warning(f"Skipped {malformed_lines} malformed lines in {ip}")

        if total_lines and (total_lines % 1_000_000) == 0:
            logger.info(f"Merging... read {human_int(total_lines)} lines | distinct in-memory: {human_int(len(cur))}")

    spill_chunk(chunk_idx)

    # create empty output and exit success
    if not chunk_files:
        _create_empty_output(out_path, out_format, logger)
        cleanup_tmpdir(tmpdir, logger)
        return

    # Phase 2: bounded fan-in multi-round k-way merge of chunk_files to final
    fanin = safe_fanin(merge_fanin)
    round_idx = 0
    curr = chunk_files
    while len(curr) > fanin:
        if stop.get("stop", False):
            logger.info("Merge interrupted by signal before intermediate round.")
            cleanup_tmpdir(tmpdir, logger)
            sys.exit(130)
        round_idx += 1
        logger.info(f"Merge round {round_idx}: inputs={len(curr)} fanin={fanin}")
        next_paths: List[str] = []
        for i in range(0, len(curr), fanin):
            group = curr[i:i+fanin]
            interm = os.path.join(tmpdir, f"merge_round{round_idx:02d}_{i//fanin:06d}.tsv.gz")
            _ = _kmerge_tsv_to_tsv(group, interm, logger, stop)
            next_paths.append(interm)
        curr = next_paths

    if stop.get("stop", False):
        logger.info("Merge interrupted by signal before final round.")
        cleanup_tmpdir(tmpdir, logger)
        sys.exit(130)

    if out_format == "json":
        out_total = _kmerge_tsv_to_json(curr, out_path, logger, stop)
    else:
        out_total = _kmerge_tsv_to_tsv(curr, out_path, logger, stop)

    logger.info(f"Merge complete. Wrote {out_path} with {human_int(out_total)} unique sequences.")
    cleanup_tmpdir(tmpdir, logger)


# CLI

def main():
    p = argparse.ArgumentParser(
        description="Merge counts JSONL(.gz)/TSV(.gz) with RAM-bounded spills and bounded fan-in k-way merges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--counts-dir", type=str, help="Directory containing counts_shard*.jsonl(.gz|.tsv[.gz]).")
    group.add_argument("--in-counts", type=str, nargs="+", help="Explicit list of counts files.")
    p.add_argument("--out-counts", type=str, required=True, help="Output merged counts .jsonl.gz or .tsv.gz.")
    p.add_argument("--count-chunk-limit", type=int, default=2_000_000, help="Max unique keys in RAM before spilling a chunk.")
    p.add_argument("--resume-merge", action="store_true", default=True)
    p.add_argument("--merge-fanin", type=int, default=1024, help="Max open files per round.")
    p.add_argument("--out-format", type=str, default="json", choices=["json", "tsv"], help="Output format.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=str, default=None)
    args = p.parse_args()

    stop = {"stop": False}
    logger = logging.getLogger("merge")
    install_signal_handler(stop, logger)

    try:
        if args.counts_dir:
            in_paths = merge_counts_dir(args.counts_dir)
        else:
            in_paths = args.in_counts
        merge_counts(
            in_paths=in_paths,
            out_path=args.out_counts,
            count_chunk_limit=int(args.count_chunk_limit),
            resume_merge=bool(getattr(args, "resume_merge", True)),
            merge_fanin=int(args.merge_fanin),
            out_format=args.out_format,
            log_level=args.log_level,
            log_file=args.log_file,
            stop_event=stop,
        )
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logging.getLogger("merge").warning("KeyboardInterrupt received. Exiting.")
        sys.exit(130)
    except Exception as e:
        logging.getLogger("merge").error(f"Fatal: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        gated_cleanup()


if __name__ == "__main__":
    main()
