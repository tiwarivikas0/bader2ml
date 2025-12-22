#!/usr/bin/env python3
"""
bader2ml.bader.runner

Run Bader charge analysis on CP2K electron density cube files,
using frame_XXXXXX.xyz as the source of truth for frame numbering.

Directory structure expected:
cp2k_density/
├── set_000/
│   ├── frame_000001.xyz
│   ├── frame_000002.xyz
│   ├── ...
│   ├── SPE-valence_density-ELECTRON_DENSITY-1_1.cube
│   ├── SPE-valence_density-ELECTRON_DENSITY-1_2.cube
│   └── ...
│
├── set_001/
│   └── ...

Each cube file is mapped by order (1..N) to frame_XXXXXX.xyz.
Output is written to:
  set_XXX/frame_XXXXXX/ACF.dat

Features:
 - Parallel execution (--nprocs)
 - Restartable via job_state.json (--resume)
 - Safe overwrite support (--overwrite)
 - Dry-run mode (--dry-run)

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
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("bader2ml.bader.runner")

# ----------------------------------------------------------------------
# Constants / regex
# ----------------------------------------------------------------------
FRAME_XYZ_RE = re.compile(r"frame_(\d+)\.xyz$")
CUBE_RE = re.compile(r".*ELECTRON_DENSITY.*_(\d+)\.cube$")
JOB_STATE_FILENAME = "job_state.json"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def default_bader_exec() -> Optional[Path]:
    this = Path(__file__).resolve()
    pkg_dir = this.parents[1]  # .../src/bader2ml
    cand = pkg_dir / "bin" / "bader"
    if cand.exists() and os.access(cand, os.X_OK):
        return cand
    return None


def atomic_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent))
    try:
        with open(src, "rb") as fr, os.fdopen(fd, "wb") as fw:
            shutil.copyfileobj(fr, fw)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_bader(frame_dir: Path, bader_exec: Path, timeout: Optional[int]) -> bool:
    density = frame_dir / "density.cube"
    out = frame_dir / "bader.out"

    with open(out, "wb") as fout:
        proc = subprocess.run(
            [str(bader_exec), str(density)],
            cwd=frame_dir,
            stdout=fout,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

    return (frame_dir / "ACF.dat").exists() and proc.returncode == 0


# ----------------------------------------------------------------------
# Task discovery (XYZ-driven)
# ----------------------------------------------------------------------
def discover_tasks(density_dir: Path, overwrite: bool) -> List[Dict]:
    tasks = []
    set_re = re.compile(r"set_\d{3}$")

    for set_dir in sorted(density_dir.iterdir()):
        if not set_dir.is_dir() or not set_re.match(set_dir.name):
            continue

        xyz_files = sorted(set_dir.glob("frame_*.xyz"))
        cube_files = sorted(p for p in set_dir.iterdir() if p.is_file() and CUBE_RE.match(p.name))

        if not xyz_files:
            logger.warning("No frame_*.xyz found in %s", set_dir)
            continue

        if len(cube_files) < len(xyz_files):
            raise RuntimeError(f"{set_dir}: fewer cubes than xyz frames")

        for local_idx, xyz in enumerate(xyz_files, start=1):
            m = FRAME_XYZ_RE.match(xyz.name)
            global_frame = int(m.group(1))

            cube = cube_files[local_idx - 1]
            frame_dir = set_dir / f"frame_{global_frame:06d}"
            acf = frame_dir / "ACF.dat"

            tasks.append({
                "set": set_dir.name,
                "global_frame": global_frame,
                "cube": str(cube),
                "frame_dir": str(frame_dir),
                "status": "DONE" if acf.exists() and not overwrite else "PENDING",
            })

    return tasks


# ----------------------------------------------------------------------
# Job state I/O
# ----------------------------------------------------------------------
def write_state(path: Path, tasks: List[Dict]):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(
            {"generated_at": datetime.utcnow().isoformat() + "Z", "tasks": tasks},
            f,
            indent=2,
        )
    os.replace(tmp, path)


def load_state(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)["tasks"]


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
def worker(task: Dict, bader_exec: Path, timeout: Optional[int]) -> Dict:
    frame_dir = Path(task["frame_dir"])
    cube = Path(task["cube"])

    try:
        frame_dir.mkdir(parents=True, exist_ok=True)
        density = frame_dir / "density.cube"

        if not density.exists():
            atomic_copy(cube, density)

        ok = run_bader(frame_dir, bader_exec, timeout)
        task["status"] = "DONE" if ok else "FAILED"
        task["finished_at"] = datetime.utcnow().isoformat() + "Z"

    except Exception as e:
        task["status"] = "FAILED"
        task["error"] = str(e)

    return task


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--density-dir", "-d", required=True)
    ap.add_argument("--bader-exec", default=None)
    ap.add_argument("--nprocs", "-j", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=None)
    args = ap.parse_args(argv)

    density_dir = Path(args.density_dir).resolve()
    if not density_dir.exists():
        sys.exit("density-dir not found")

    state_path = density_dir / JOB_STATE_FILENAME

    # Bader executable
    bader_exec = Path(args.bader_exec).resolve() if args.bader_exec else default_bader_exec()
    if not bader_exec or not bader_exec.exists():
        sys.exit("Bader executable not found")

    if not os.access(bader_exec, os.X_OK):
        os.chmod(bader_exec, os.stat(bader_exec).st_mode | stat.S_IXUSR)

    # Tasks
    if args.resume and state_path.exists():
        logger.info("Resuming from %s", state_path)
        tasks = load_state(state_path)
        if args.overwrite:
            for t in tasks:
                if t["status"] == "DONE":
                    t["status"] = "PENDING"
    else:
        tasks = discover_tasks(density_dir, args.overwrite)
        write_state(state_path, tasks)

    pending = [t for t in tasks if t["status"] == "PENDING"]
    logger.info("Tasks: %d total, %d pending", len(tasks), len(pending))

    if args.dry_run:
        for t in pending:
            logger.info("DRY: %s -> %s", t["cube"], t["frame_dir"])
        return

    # Run
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.nprocs) as ex:
        futures = {ex.submit(worker, t, bader_exec, args.timeout): t for t in pending}
        for fut in concurrent.futures.as_completed(futures):
            write_state(state_path, tasks)

    write_state(state_path, tasks)
    logger.info("Bader analysis complete")


if __name__ == "__main__":
    main()
