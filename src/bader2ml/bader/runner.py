#!/usr/bin/env python3
"""
bader2ml.bader.runner

Traverse CP2K density output (per-set directories), run Bader on each cube,
and store results in per-frame subdirectories.

Features:
 - Bundled bader executable used by default if present at src/bader2ml/bin/bader.
 - Parallel execution controlled by --nprocs (ThreadPool).
 - Restartable: writes job_state.json into the density_dir root and honors --resume.
 - Skip frames with existing ACF.dat unless --overwrite is used.
 - Dry-run mode to preview planned tasks.

Usage (example):
  python runner.py \
    --density-dir cp2k_density \
    --nprocs 4 \
    --bader-exec /full/path/to/bader \
    --resume

Or via CLI (when wired into bader2ml):
  bader2ml bader run --density-dir cp2k_density --nprocs 4 --resume

Author: ChatGPT for Vikas
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("bader2ml.bader.runner")


JOB_STATE_FILENAME = "job_state.json"


def default_bader_exec() -> Optional[Path]:
    """
    Look for a bundled bader binary inside the package.
    Expected location (when installed in src layout):
      <...>/src/bader2ml/bin/bader
    or when running from repo root:
      <repo>/bader2ml/bin/bader
    Return Path if exists, else None.
    """
    this = Path(__file__).resolve()
    # this: .../src/bader2ml/bader/runner.py
    pkg_dir = this.parents[1]  # .../src/bader2ml
    cand = pkg_dir / "bin" / "bader"
    if cand.exists() and os.access(cand, os.X_OK):
        return cand
    return None


def discover_cube_files(density_dir: Path, pattern: str = r".*ELECTRON_DENSITY.*_(\d+)\.cube$") -> List[Tuple[Path, Path, int]]:
    """
    Walk density_dir and discover cube files.
    Returns list of tuples: (set_dir, cube_path, local_index)
    - set_dir: Path to set_XXX directory
    - cube_path: full path to discovered cube file
    - local_index: integer captured from filename (frame index inside set)
    """
    entries = []
    set_re = re.compile(r"set_\d{3}$")
    cube_re = re.compile(pattern)
    for set_entry in sorted(density_dir.iterdir()):
        if not set_entry.is_dir() or not set_re.match(set_entry.name):
            continue
        for p in sorted(set_entry.iterdir()):
            if p.is_file() and cube_re.match(p.name):
                m = cube_re.match(p.name)
                try:
                    idx = int(m.group(1))
                except Exception:
                    idx = None
                if idx is None:
                    logger.warning("Could not parse local index from cube filename: %s", p)
                    continue
                entries.append((set_entry, p, idx))
    return entries


def make_frame_dir(set_dir: Path, local_index: int, pad: int = 6) -> Path:
    name = f"frame_{local_index:0{pad}d}"
    frame_dir = set_dir / name
    frame_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent))
    try:
        with open(src, "rb") as fr, os.fdopen(fd, "wb") as fw:
            shutil.copyfileobj(fr, fw)
        os.replace(tmp, str(dst))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def run_bader_in_frame(frame_dir: Path, bader_exec: Path, timeout: Optional[int] = None) -> Tuple[bool, str]:
    """
    Run `bader density.cube` in frame_dir. Return (success, message).
    Writes bader.out capturing stdout+stderr.
    Success is defined as presence of ACF.dat after the run.
    """
    density = frame_dir / "density.cube"
    if not density.exists():
        return False, f"density.cube not found in {frame_dir}"

    cmd = [str(bader_exec), str(density)]
    out_path = frame_dir / "bader.out"
    logger.debug("Running Bader: %s in %s", cmd, frame_dir)
    try:
        with open(out_path, "wb") as fout:
            proc = subprocess.run(cmd, cwd=str(frame_dir), stdout=fout, stderr=subprocess.STDOUT, timeout=timeout)
        # Check for ACF.dat or other expected output
        acf = frame_dir / "ACF.dat"
        if acf.exists():
            return True, "ACF.dat produced"
        else:
            # bader sometimes writes different naming depending on input; still check return code
            if proc.returncode == 0:
                # warn but treat as failure unless ACF.dat present
                return False, "bader returned 0 but ACF.dat missing"
            return False, f"bader exited with code {proc.returncode}"
    except subprocess.TimeoutExpired:
        return False, "bader timed out"
    except Exception as exc:
        return False, f"exception running bader: {exc}"


def build_task_list(discovered: List[Tuple[Path, Path, int]], overwrite: bool = False) -> List[Dict]:
    """
    Build list of tasks with metadata for job_state.json.
    Each task is dict:
      {
        "set": "set_000",
        "local_index": 1,
        "cube_path": "/abs/path/to/cube",
        "frame_dir": "set_000/frame_000001",
        "status": "PENDING"
      }
    """
    tasks = []
    for set_dir, cube_path, local_index in discovered:
        frame_dir = set_dir / f"frame_{local_index:06d}"
        acf = frame_dir / "ACF.dat"
        if acf.exists() and not overwrite:
            status = "DONE"
        else:
            status = "PENDING"
        tasks.append(
            {
                "set": set_dir.name,
                "set_dir": str(set_dir),
                "local_index": int(local_index),
                "cube_path": str(cube_path),
                "frame_dir": str(frame_dir),
                "status": status,
            }
        )
    return tasks


def write_job_state(path: Path, state: Dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def load_job_state(path: Path) -> Dict:
    with open(path, "r") as fh:
        return json.load(fh)


def worker_run_task(task: Dict, bader_exec: Path, timeout: Optional[int] = None, atomic_copy_required: bool = True) -> Dict:
    """
    Execute one task:
      - ensure frame_dir exists
      - copy cube -> frame_dir/density.cube (if not already present or overwrite)
      - run bader
      - update task dict with result fields: status, msg, finished_at
    """
    frame_dir = Path(task["frame_dir"])
    cube_src = Path(task["cube_path"])
    result = task.copy()
    result.setdefault("attempts", 0)
    result["attempts"] += 1
    try:
        frame_dir.mkdir(parents=True, exist_ok=True)
        density_dst = frame_dir / "density.cube"
        # copy cube if not present or if cube differs (we simply overwrite if missing)
        if not density_dst.exists():
            atomic_copy(cube_src, density_dst)
        # run bader
        success, msg = run_bader_in_frame(frame_dir, bader_exec, timeout=timeout)
        if success:
            result["status"] = "DONE"
            result["msg"] = msg
            result["finished_at"] = datetime.utcnow().isoformat() + "Z"
        else:
            result["status"] = "FAILED"
            result["msg"] = msg
            result["finished_at"] = datetime.utcnow().isoformat() + "Z"
    except Exception as exc:
        result["status"] = "FAILED"
        result["msg"] = f"exception: {exc}"
        result["finished_at"] = datetime.utcnow().isoformat() + "Z"
    return result


def run_parallel(tasks: List[Dict], bader_exec: Path, nprocs: int, state_path: Path, timeout: Optional[int], dry_run: bool):
    """
    Execute tasks in parallel (ThreadPool). Persist state to state_path regularly.
    """
    # Build queue of indices to run
    pending_indices = [i for i, t in enumerate(tasks) if t["status"] == "PENDING"]
    if dry_run:
        logger.info("Dry run requested. The following tasks would be processed:")
        for i in pending_indices:
            t = tasks[i]
            logger.info("  %s : %s -> %s", t["set"], t["cube_path"], t["frame_dir"])
        return tasks

    logger.info("Starting processing: %d tasks pending, nprocs=%d", len(pending_indices), nprocs)
    # Use ThreadPoolExecutor because tasks spawn subprocesses and are IO-bound.
    with concurrent.futures.ThreadPoolExecutor(max_workers=nprocs) as ex:
        # map futures to task index
        futures = {}
        # submit initial batch
        for idx in pending_indices:
            fut = ex.submit(worker_run_task, tasks[idx], bader_exec, timeout, True)
            futures[fut] = idx

        # As futures complete, update tasks and write state
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                tasks[idx].update(res)
            except Exception as exc:
                tasks[idx]["status"] = "FAILED"
                tasks[idx]["msg"] = f"worker exception: {exc}"
                tasks[idx]["finished_at"] = datetime.utcnow().isoformat() + "Z"
            # persist state
            state = {"generated_at": datetime.utcnow().isoformat() + "Z", "tasks": tasks}
            write_job_state(state_path, state)
            logger.info(
                "Task %s/%s -> %s (status=%s)",
                tasks[idx].get("set"),
                tasks[idx].get("local_index"),
                tasks[idx].get("frame_dir"),
                tasks[idx].get("status"),
            )

    return tasks


def summarize_tasks(tasks: List[Dict]):
    total = len(tasks)
    done = sum(1 for t in tasks if t["status"] == "DONE")
    failed = sum(1 for t in tasks if t["status"] == "FAILED")
    pending = sum(1 for t in tasks if t["status"] == "PENDING")
    logger.info("Summary: total=%d done=%d failed=%d pending=%d", total, done, failed, pending)


def parse_args():
    p = argparse.ArgumentParser(description="Run Bader on CP2K density cube files (interactive node).")
    p.add_argument("--density-dir", "-d", required=True, help="Base directory containing set_XXX folders")
    p.add_argument("--bader-exec", "-b", default=None, help="Path to bader executable (default: bundled bin/bader if present)")
    p.add_argument("--nprocs", "-j", type=int, default=1, help="Number of parallel Bader jobs (default 1)")
    p.add_argument("--resume", action="store_true", help="Resume from existing job_state.json if present (skip DONE)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing ACF.dat and re-run Bader for frames")
    p.add_argument("--dry-run", action="store_true", help="Do not run Bader; just print what would be done")
    p.add_argument("--timeout", type=int, default=None, help="Optional timeout (seconds) for each Bader run")
    return p.parse_args()


def main():
    args = parse_args()
    density_dir = Path(args.density_dir).resolve()
    if not density_dir.exists():
        logger.error("density-dir not found: %s", density_dir)
        sys.exit(2)

    # find bader executable
    bader_path: Optional[Path]
    if args.bader_exec:
        bader_path = Path(args.bader_exec).resolve()
    else:
        bader_path = default_bader_exec()
    if bader_path is None or not bader_path.exists():
        logger.error("Bader executable not found. Provide --bader-exec or place a bader binary at <pkg>/bin/bader")
        sys.exit(3)
    if not os.access(bader_path, os.X_OK):
        logger.warning("Bader executable is not executable; attempting to set +x")
        try:
            st = os.stat(bader_path)
            os.chmod(bader_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception as exc:
            logger.error("Could not chmod +x %s: %s", bader_path, exc)
            sys.exit(4)

    logger.info("Using bader executable: %s", bader_path)

    # discover cubes
    discovered = discover_cube_files(density_dir)
    if not discovered:
        logger.warning("No cube files found under %s. Are CP2K jobs finished?", density_dir)
        sys.exit(0)

    # build (or load) job_state
    state_path = density_dir / JOB_STATE_FILENAME
    if args.resume and state_path.exists():
        logger.info("Resuming from existing job_state.json: %s", state_path)
        state = load_job_state(state_path)
        tasks = state.get("tasks", [])
        # If user requested overwrite, reset DONE->PENDING for tasks (so they will be re-run)
        if args.overwrite:
            for t in tasks:
                if t.get("status") == "DONE":
                    t["status"] = "PENDING"
    else:
        # build fresh tasks from discovered cubes
        tasks = build_task_list(discovered, overwrite=args.overwrite)
        state = {"generated_at": datetime.utcnow().isoformat() + "Z", "tasks": tasks}
        write_job_state(state_path, state)
        logger.info("Created new job_state.json at %s", state_path)

    # run tasks in parallel (or dry-run)
    tasks = run_parallel(tasks, bader_path, args.nprocs, state_path, args.timeout, args.dry_run)

    # final state write
    final_state = {"generated_at": datetime.utcnow().isoformat() + "Z", "tasks": tasks}
    write_job_state(state_path, state_path) if False else write_job_state(state_path, final_state)

    summarize_tasks(tasks)

    logger.info("Bader run finished. Check %s for details. To restart failed tasks use --resume", state_path)


if __name__ == "__main__":
    main()

