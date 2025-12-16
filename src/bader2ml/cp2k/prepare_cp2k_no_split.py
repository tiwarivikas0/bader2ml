#!/usr/bin/env python3
"""
prepare_cp2k_split.py

Split a multi-frame XYZ trajectory into sets and per-set coords (coords.xyz),
single-frame files and frame directories suitable for the Bader pipeline.

For each set_XXX:
 - writes coords.xyz (multi-frame file containing frames for that set)
 - writes ff_input.xyz (first frame of coords.xyz) and also writes single-frame
   files frame_000001.xyz ... (kept for backward compatibility)
 - creates empty frame_000001/ directories where ACF.dat will be placed
 - copies cp2k.inp into the set and **modifies REFTRAJ/COORD_FILE_NAME**:
     TRAJ_FILE_NAME coords.xyz
     FIRST_SNAPSHOT 1
     STRIDE 1
     LAST_SNAPSHOT <n_frames_in_set>
     COORD_FILE_NAME ff_input.xyz
 - copies job.pbs into set and ensures "#PBS -N set_XXX"

Usage:
  python -m bader2ml.cp2k.prepare_cp2k_split \
      --xyz trajectory.xyz --cp2k-inp cp2k.inp --pbs-template job.pbs \
      --frames-per-set 100 --output-dir cp2k_density [--overwrite]
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("bader2ml.cp2k.prepare_split")

XYZ_COUNT_RE = re.compile(r"^\s*(\d+)\s*$")
PBS_NAME_RE = re.compile(r"^\s*#PBS\s+-N\s+.+$", flags=re.IGNORECASE)

# REFTRAJ / COORD patterns (case-insensitive)
REFTRAJ_TRAJ_RE = re.compile(r"^\s*TRAJ_FILE_NAME\s+(.+)$", flags=re.IGNORECASE)
REFTRAJ_FIRST_RE = re.compile(r"^\s*FIRST_SNAPSHOT\s+(\d+)", flags=re.IGNORECASE)
REFTRAJ_STRIDE_RE = re.compile(r"^\s*STRIDE\s+(\d+)", flags=re.IGNORECASE)
REFTRAJ_LAST_RE = re.compile(r"^\s*LAST_SNAPSHOT\s+(\d+)", flags=re.IGNORECASE)
COORD_FILE_RE = re.compile(r"^\s*COORD_FILE_NAME\s+(.+)$", flags=re.IGNORECASE)


def read_xyz_frames(path: Path) -> List[Tuple[int, str, List[str]]]:
    """Read all frames from a multi-frame xyz."""
    frames: List[Tuple[int, str, List[str]]] = []
    with open(path, "r") as fh:
        while True:
            # find next non-empty line for atom count
            atom_line = None
            for raw in fh:
                if raw.strip() == "":
                    continue
                atom_line = raw.rstrip("\n")
                break
            if atom_line is None:
                break
            m = XYZ_COUNT_RE.match(atom_line)
            if not m:
                raise ValueError(f"Bad atom count line in {path}: {atom_line!r}")
            n_atoms = int(m.group(1))
            comment = fh.readline()
            if comment is None:
                raise EOFError("Unexpected EOF while reading XYZ comment")
            comment = comment.rstrip("\n")
            atom_lines = []
            for _ in range(n_atoms):
                ln = fh.readline()
                if ln is None:
                    raise EOFError("Unexpected EOF while reading atoms")
                if ln.strip() == "":
                    raise ValueError("Empty atom line inside frame")
                atom_lines.append(ln.rstrip("\n"))
            frames.append((n_atoms, comment, atom_lines))
    return frames


def atomic_write_text(dst: Path, text: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, str(dst))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


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


def modify_pbs_job_name(template_text: str, job_name: str) -> str:
    """
    Replace or insert '#PBS -N <job_name>' in PBS script.
    """
    lines = template_text.splitlines()
    found = False
    new_lines = []
    for ln in lines:
        if PBS_NAME_RE.match(ln):
            new_lines.append(f"#PBS -N {job_name}")
            found = True
        else:
            new_lines.append(ln)
    if not found:
        new_lines.insert(0, f"#PBS -N {job_name}")
    return "\n".join(new_lines) + "\n"


def modify_cp2k_input_for_set(template_text: str, coords_name: str, n_frames: int, coord_file_name: str = "ff_input.xyz", stride: int = 1) -> str:
    """
    Modify or insert REFTRAJ directives and COORD_FILE_NAME so that the cp2k input in the set uses:
       TRAJ_FILE_NAME <coords_name>
       FIRST_SNAPSHOT 1
       STRIDE <stride>
       LAST_SNAPSHOT <n_frames>
    and
       COORD_FILE_NAME <coord_file_name>
    If existing REFTRAJ lines are present they are replaced; otherwise a &REFTRAJ block is appended.
    COORD_FILE_NAME is replaced/inserted at top-level if present or appended near REFTRAJ block.
    """
    lines = template_text.splitlines()
    found_traj = False
    found_first = False
    found_stride = False
    found_last = False
    found_coord = False

    new_lines = []
    # Replace existing lines if present
    for ln in lines:
        if REFTRAJ_TRAJ_RE.match(ln):
            new_lines.append(f"   TRAJ_FILE_NAME {coords_name}")
            found_traj = True
            continue
        if REFTRAJ_FIRST_RE.match(ln):
            new_lines.append(f"   FIRST_SNAPSHOT 1")
            found_first = True
            continue
        if REFTRAJ_STRIDE_RE.match(ln):
            new_lines.append(f"   STRIDE {stride}")
            found_stride = True
            continue
        if REFTRAJ_LAST_RE.match(ln):
            new_lines.append(f"   LAST_SNAPSHOT {n_frames}")
            found_last = True
            continue
        if COORD_FILE_RE.match(ln):
            new_lines.append(f"COORD_FILE_NAME {coord_file_name}")
            found_coord = True
            continue
        new_lines.append(ln)

    # If none of the REFTRAJ lines were present, append a REFTRAJ block
    if not any([found_traj, found_first, found_stride, found_last]):
        block = [
            "",
            "&REFTRAJ",
            f"   TRAJ_FILE_NAME {coords_name}",
            "   FIRST_SNAPSHOT 1",
            f"   STRIDE {stride}",
            f"   LAST_SNAPSHOT {n_frames}",
            "&END",
        ]
        new_lines.extend(block)
    else:
        # If some REFTRAJ lines were present but some missing, try to insert missing ones near first occurrence
        missing_parts = []
        if not found_traj:
            missing_parts.append(f"   TRAJ_FILE_NAME {coords_name}")
        if not found_first:
            missing_parts.append("   FIRST_SNAPSHOT 1")
        if not found_stride:
            missing_parts.append(f"   STRIDE {stride}")
        if not found_last:
            missing_parts.append(f"   LAST_SNAPSHOT {n_frames}")
        if missing_parts:
            # insert after first REFTRAJ-like line
            insert_idx = 0
            for i, ln in enumerate(new_lines):
                if REFTRAJ_TRAJ_RE.match(ln) or REFTRAJ_FIRST_RE.match(ln) or REFTRAJ_STRIDE_RE.match(ln) or REFTRAJ_LAST_RE.match(ln):
                    insert_idx = i
                    break
            for j, ins in enumerate(missing_parts):
                new_lines.insert(insert_idx + 1 + j, ins)

    # Ensure COORD_FILE_NAME exists; if not, append after REFTRAJ block or at end
    if not found_coord:
        # try to insert after REFTRAJ block end (&END)
        inserted = False
        for i, ln in enumerate(new_lines):
            if ln.strip().upper() == "&END":
                # insert COORD_FILE_NAME after this line
                new_lines.insert(i + 1, f"COORD_FILE_NAME {coord_file_name}")
                inserted = True
                break
        if not inserted:
            # append at end
            new_lines.append(f"COORD_FILE_NAME {coord_file_name}")

    return "\n".join(new_lines) + "\n"


def build_frame_map(n_frames: int, frames_per_set: int) -> Dict:
    n_sets = math.ceil(n_frames / frames_per_set)
    sets = {}
    for s in range(n_sets):
        start = s * frames_per_set + 1
        end = min((s + 1) * frames_per_set, n_frames)
        sets[f"set_{s:03d}"] = {"first_snapshot": start, "last_snapshot": end}
    return {"frames_per_set": frames_per_set, "total_frames": n_frames, "sets": sets, "created": datetime.utcnow().isoformat() + "Z"}


def write_frame_xyz(n_atoms: int, comment: str, atom_lines: List[str], path: Path):
    text = f"{n_atoms}\n{comment}\n" + "\n".join(atom_lines) + "\n"
    atomic_write_text(path, text)


def write_coords_xyz_for_set(frames: List[Tuple[int, str, List[str]]], path: Path):
    """
    frames: list of (n_atoms, comment, atom_lines) for the set in order.
    Writes a multi-frame xyz file (concatenated frames) to path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            for n_atoms, comment, atom_lines in frames:
                fh.write(f"{n_atoms}\n{comment}\n")
                fh.write("\n".join(atom_lines) + "\n")
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def prepare_split(
    trajectory: Path,
    cp2k_inp_template: Path,
    pbs_template: Path,
    frames_per_set: int,
    output_dir: Path,
    overwrite: bool = False,
):
    trajectory = trajectory.resolve()
    cp2k_inp_template = cp2k_inp_template.resolve()
    pbs_template = pbs_template.resolve()
    output_dir = output_dir.resolve()

    if not trajectory.exists():
        raise FileNotFoundError(f"Trajectory not found: {trajectory}")
    if not cp2k_inp_template.exists():
        raise FileNotFoundError(f"CP2K input template not found: {cp2k_inp_template}")
    if not pbs_template.exists():
        raise FileNotFoundError(f"PBS template not found: {pbs_template}")

    frames = read_xyz_frames(trajectory)
    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("No frames found in trajectory")

    logger.info("Trajectory %s contains %d frames", trajectory, total_frames)

    frame_map = build_frame_map(total_frames, frames_per_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    fm_path = output_dir / "frame_map.json"
    atomic_write_text(fm_path, json.dumps(frame_map, indent=2))
    logger.info("Wrote frame_map.json to %s", fm_path)

    cp2k_text = cp2k_inp_template.read_text()
    pbs_text = pbs_template.read_text()

    # iterate sets and write files
    for set_name, meta in frame_map["sets"].items():
        set_dir = output_dir / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        first = int(meta["first_snapshot"])
        last = int(meta["last_snapshot"])
        n_frames_in_set = last - first + 1
        logger.info("Preparing %s: frames %d..%d (n=%d)", set_name, first, last, n_frames_in_set)

        # prepare list of frames for this set (1-based indexing)
        set_frames = [frames[i - 1] for i in range(first, last + 1)]

        # write coords.xyz (multi-frame)
        coords_path = set_dir / "coords.xyz"
        if overwrite or not coords_path.exists():
            write_coords_xyz_for_set(set_frames, coords_path)

        # write ff_input.xyz (first frame of coords)
        ff_dst = set_dir / "ff_input.xyz"
        if overwrite or not ff_dst.exists():
            n_atoms, comment, atom_lines = set_frames[0]
            write_frame_xyz(n_atoms, comment, atom_lines, ff_dst)

        # also write individual single-frame files (backwards compatible)
        for local_idx, (n_atoms, comment, atom_lines) in enumerate(set_frames, start=1):
            global_idx = first + local_idx - 1
            frame_idx6 = f"frame_{global_idx:06d}"
            frame_xyz_path = set_dir / f"{frame_idx6}.xyz"
            frame_dir = set_dir / frame_idx6
            # create frame_dir if missing
            if not frame_dir.exists():
                frame_dir.mkdir(parents=True, exist_ok=True)
            if overwrite or not frame_xyz_path.exists():
                write_frame_xyz(n_atoms, comment, atom_lines, frame_xyz_path)

        # modify cp2k input for this set: point to coords.xyz and set REFTRAJ FIRST/LAST
        modified_cp2k = modify_cp2k_input_for_set(cp2k_text, coords_name="coords.xyz", n_frames=n_frames_in_set, coord_file_name="ff_input.xyz", stride=1)
        cp2k_dst = set_dir / cp2k_inp_template.name
        if overwrite or not cp2k_dst.exists():
            atomic_write_text(cp2k_dst, modified_cp2k)

        # modify and write PBS with job name
        pbs_mod = modify_pbs_job_name(pbs_text, set_name)
        pbs_dst = set_dir / pbs_template.name
        if overwrite or not pbs_dst.exists():
            atomic_write_text(pbs_dst, pbs_mod)

    logger.info("Prepared %d sets under %s (frames_per_set=%d)", len(frame_map["sets"]), output_dir, frames_per_set)
    logger.info("Done. You can now edit CP2K inputs in each set or submit jobs.")

# CLI
def parse_cli(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Prepare CP2K jobs by splitting trajectory into per-set frames.")
    p.add_argument("--xyz", "-x", required=True, help="Path to multi-frame xyz trajectory")
    p.add_argument("--cp2k-inp", "-i", required=True, help="CP2K input template to copy into each set")
    p.add_argument("--pbs-template", "-p", required=True, help="PBS job template to copy into each set")
    p.add_argument("--frames-per-set", "-n", type=int, required=True, help="Frames per set (e.g. 100)")
    p.add_argument("--output-dir", "-o", default="cp2k_density", help="Output base directory for sets")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output dir")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_cli(argv)
    try:
        prepare_split(
            trajectory=Path(args.xyz),
            cp2k_inp_template=Path(args.cp2k_inp),
            pbs_template=Path(args.pbs_template),
            frames_per_set=int(args.frames_per_set),
            output_dir=Path(args.output_dir),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:
        logger.exception("Failed to prepare cp2k split: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()

