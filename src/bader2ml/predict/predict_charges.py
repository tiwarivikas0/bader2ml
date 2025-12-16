#!/usr/bin/env python3
"""
Predict Bader charges using a SchNet model and a LAMMPS type->element mapping file.

Type map file format (space separated):
# comments allowed
1 Au
2 O
3 Mg

The script will map LAMMPS type IDs -> element symbol -> Z -> valence electrons (from embedded VALENCE_MAP).
Outputs an EXTXYZ with 'net_charges' and 'bader_charges' arrays and optional per-element CSV summary.

Example:
  python predict_charges_type_map.py -i dump.lammps -m model.pt --type-map types.map -o traj-charges.extxyz
"""

from pathlib import Path
import argparse, logging, json, sys
import numpy as np, pandas as pd, torch
import schnetpack as spk
from ase.io import read, write
from ase.data import atomic_numbers, atomic_masses, chemical_symbols
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ------------------ embedded valence map (symbol -> valence electrons) ------------------
VALENCE_MAP = {
 "H":1,"Li":3,"Na":9,"K":9,"Rb":9,"Cs":9,"Fr":9,
 "Be":4,"Mg":10,"Ca":10,"Sr":10,"Ba":10,"Ra":10,
 "Sc":11,"Y":11,"La":11,"Ac":11,
 "Ti":12,"Zr":12,"Hf":12,"Th":12,
 "V":13,"Nb":13,"Ta":13,"Pa":13,
 "Cr":14,"Mo":14,"W":14,"U":14,
 "Mn":15,"Tc":15,"Re":15,"Np":15,
 "Fe":16,"Ru":16,"Os":16,"Pu":16,
 "Co":17,"Rh":17,"Ir":17,"Am":17,
 "Ni":18,"Pd":18,"Pt":18,"Cm":18,
 "Cu":11,"Ag":11,"Au":11,"Bk":19,
 "Zn":12,"Cd":12,"Hg":12,"Cf":20,
 "B":3,"Al":3,"Ga":13,"In":13,"Tl":13,"Es":21,
 "C":4,"Si":4,"Ge":4,"Sn":4,"Pb":4,"Fm":22,
 "N":5,"P":5,"As":5,"Sb":5,"Bi":5,
 "O":6,"S":6,"Se":6,"Te":6,"Po":6,
 "F":7,"Cl":7,"Br":7,"I":7,"At":7,
 "He":2,"Ne":8,"Ar":8,"Kr":8,"Xe":8,"Rn":8,
 "Ce":12,"Pr":13,"Nd":14,"Pm":15,"Sm":16,"Eu":17,"Gd":18,
 "Tb":19,"Dy":20,"Ho":21,"Er":22,"Tm":23,"Yb":24,"Lu":25
}
# -----------------------------------------------------------------------------------------

def parse_type_map_file(path: Path):
    """
    Parse a two-column (space separated) file: <type_id> <symbol_or_Z>
    Returns dict: int(type_id) -> (symbol:str, Z:int, valence:float)
    """
    mp = {}
    txt = path.read_text().splitlines()
    for ln in txt:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2:
            logging.warning("Skipping malformed line in type-map: %r", ln)
            continue
        try:
            tid = int(parts[0])
        except ValueError:
            logging.warning("Skipping line with non-integer type id: %r", ln); continue
        token = parts[1]
        # token may be element symbol or atomic number string
        Z = None; sym = None
        # try int atomic number
        try:
            zcand = int(token)
            if 1 <= zcand < len(chemical_symbols):
                Z = zcand; sym = chemical_symbols[Z]
        except Exception:
            pass
        if sym is None:
            # try as element symbol (case-insensitive)
            tok_up = token.capitalize()
            if tok_up in VALENCE_MAP or tok_up in atomic_numbers:
                sym = tok_up
                Z = atomic_numbers.get(sym, None)
            else:
                logging.warning("Unknown element symbol '%s' in type-map line: %r", token, ln)
                sym = token  # keep raw; Z remains None
        val = 0.0
        if sym in VALENCE_MAP:
            val = float(VALENCE_MAP[sym])
        elif Z is not None and chemical_symbols[Z] in VALENCE_MAP:
            val = float(VALENCE_MAP[chemical_symbols[Z]])
        else:
            logging.warning("Valence not found for symbol=%s (type %d); using 0.0", sym, tid)
            val = 0.0
        mp[int(tid)] = (sym, int(Z) if Z is not None else 0, float(val))
    return mp

def load_model(model_file: str, device: torch.device):
    """
    Robust loader that tries:
      1) torch.load(model_file, map_location=device, weights_only=False)  (if supported)
      2) torch.load(model_file, map_location=device)
      3) schnetpack.utils.load_model(model_file)  (no device kw)
    Returns a model on `device` in eval() mode or raises a RuntimeError.
    """
    logging.info("Loading model from %s", model_file)

    # 1) try torch.load with weights_only=False (PyTorch >=2.6)
    try:
        try:
            model = torch.load(model_file, map_location=device, weights_only=False)
        except TypeError:
            # weights_only not accepted by this torch version — try without it
            model = torch.load(model_file, map_location=device)
        model.to(device)
        model.eval()
        logging.info("Loaded model via torch.load()")
        return model
    except Exception as e_torch:
        logging.warning("torch.load() failed: %s", e_torch)

    # 2) fallback: schnetpack.utils.load_model (do not pass device kw)
    try:
        from schnetpack.utils import load_model as spk_load
        model = spk_load(str(model_file))
        model.to(device)
        model.eval()
        logging.info("Loaded model via schnetpack.utils.load_model()")
        return model
    except Exception as e_spk:
        logging.warning("schnetpack.utils.load_model() failed: %s", e_spk)

    raise RuntimeError(
        f"Failed to load model from {model_file}.\n"
        f"torch.load error: {e_torch!r}\n"
        f"schnetpack.utils.load_model error: {e_spk!r}\n"
        "If you control training, recommended: save model.state_dict() and load it into a freshly constructed model."
    )

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict Bader charges using SchNet + type-map file")
    p.add_argument("--input","-i",required=True,help="Input traj (.extxyz/.xyz) or LAMMPS dump (.lammps)")
    p.add_argument("--model","-m",required=True,help="Trained model file (torch save)")
    p.add_argument("--type-map","-t",required=True,help="Type map file (space-separated two columns: id symbol)")
    p.add_argument("--output","-o",default="traj-charges.extxyz",help="Output EXTXYZ (overwrites)")
    p.add_argument("--summary","-s",default=None,help="Optional CSV summary file with per-element sums")
    p.add_argument("--every","-e",type=int,default=1,help="Process every N-th frame")
    p.add_argument("--from","-f",dest="fr",type=int,default=0,help="Start frame index")
    p.add_argument("--upto","-u",dest="upto",default="",help="End frame index (ASE slice end)")
    p.add_argument("--cutoff",type=float,default=3.5,help="Environment cutoff (Å)")
    p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"],help="Device")
    p.add_argument("--max-z",type=int,default=100,help="Max Z for valence table")
    p.add_argument("--no-write",dest="write_traj",action="store_false",help="Do not write output trajectory")
    p.add_argument("--verbose","-v",action="store_true",help="Verbose logging")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    inp = Path(args.input)
    model_file = Path(args.model)
    type_map_file = Path(args.type_map)
    out_traj = Path(args.output)
    summary_out = Path(args.summary) if args.summary else None

    # device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info("Device: %s", device)

    # parse type-map file
    if not type_map_file.exists():
        raise FileNotFoundError(f"type-map file not found: {type_map_file}")
    type_map = parse_type_map_file(type_map_file)
    logging.info("Parsed %d type-map entries (sample): %s", len(type_map), str(list(type_map.items())[:6]))

    # build valence_by_Z for fallback checks
    valence_by_Z = np.zeros((args.max_z+1,1),dtype=float)
    for sym,val in VALENCE_MAP.items():
        Z = atomic_numbers.get(sym)
        if Z and Z <= args.max_z:
            valence_by_Z[Z,0] = float(val)

    # read trajectory (ASE indexing)
    idx = f"{args.fr}:{args.upto}:{args.every}" if args.upto != "" else f"{args.fr}::{args.every}"
    logging.info("Reading input %s index=%s", inp, idx)
    if inp.suffix in (".xyz", ".extxyz"):
        traj = list(read(str(inp), format="extxyz", index=idx))
    elif inp.suffix in (".lammps", ".dump") or "lammps" in inp.suffix:
        traj = list(read(str(inp), format="lammps-dump-text", index=idx))
    else:
        raise RuntimeError("Unsupported input extension (expected .xyz/.extxyz or lammps dump)")

    if len(traj) == 0:
        raise RuntimeError("No frames read from input")

    # remove frames with NaNs
    traj = [a for a in traj if not np.isnan(a.positions).any()]
    if len(traj) == 0:
        raise RuntimeError("No valid frames (positions contain NaN)")

    # load model
    model = load_model(model_file, device)

    # converter
    env_provider = spk.environment.AseEnvironmentProvider(args.cutoff)
    converter = spk.data.AtomsConverter(device=device, environment_provider=env_provider)

    out_frames = []
    logging.info("Starting prediction loop for %d frames", len(traj))

    # iterate frames
    for i, atoms in enumerate(tqdm(traj)):
        # atoms.get_atomic_numbers() in LAMMPS dumps are typically type IDs; we'll map them
        types_or_Z = np.array(atoms.get_atomic_numbers(), dtype=int)
        # map each entry: if value is a key in type_map -> use mapped Z; else assume it's already Z
        Zs = np.zeros_like(types_or_Z)
        for j, v in enumerate(types_or_Z):
            if int(v) in type_map:
                sym, Z, val = type_map[int(v)]
                Zs[j] = int(Z)
            else:
                # assume it's already an atomic number if plausible
                if 1 <= int(v) < len(chemical_symbols) and chemical_symbols[int(v)] != "X":
                    Zs[j] = int(v)
                else:
                    logging.warning("Frame %d atom %d: type/id %s not in type_map and not a valid Z; setting Z=0", i, j, v)
                    Zs[j] = 0

        # set atomic numbers on atoms object
        atoms.set_atomic_numbers(list(map(int, Zs)))

        # convert and predict
        inputs = converter(atoms)
        with torch.no_grad():
            pred = model(inputs)

        # get prediction key (prefer "charge")
        key = "charge" if "charge" in pred else next((k for k in pred.keys() if "charge" in k), None)
        if key is None:
            raise RuntimeError("Model output does not contain 'charge' key")

        y = pred[key].detach().cpu().numpy()[0].reshape(-1)  # (natoms,)
        atoms.set_array("net_charges", y.astype(float))

        # compute base valence per atom using type_map if possible else valence_by_Z
        base = np.zeros_like(y, dtype=float)
        for j, Z in enumerate(Zs):
            # try to find valence from type_map entry (preferred)
            t = int(types_or_Z[j])
            if int(t) in type_map:
                _, Zm, val = type_map[int(t)]
                base[j] = float(val)
            else:
                if 0 <= int(Z) < valence_by_Z.shape[0]:
                    base[j] = float(valence_by_Z[int(Z), 0])
                else:
                    base[j] = 0.0

        bader_q = base - y
        atoms.set_array("bader_charges", bader_q.astype(float))

        out_frames.append(atoms)

    # write output extxyz
    if args.write_traj:
        cols = ["symbols", "positions", "bader_charges", "net_charges"]
        logging.info("Writing %d frames to %s", len(out_frames), out_traj)
        write(str(out_traj), out_frames, format="extxyz", columns=cols)
        logging.info("Wrote: %s", out_traj)

    # write summary CSV
    if summary_out is not None:
        logging.info("Writing summary CSV to %s", summary_out)
        symbols_all = np.unique(out_frames[0].get_chemical_symbols())
        summary = {}
        for sym in symbols_all:
            sums = []
            for atoms in out_frames:
                mask = np.array(atoms.get_chemical_symbols()) == sym
                sums.append(float(np.sum(atoms.get_array("net_charges")[mask])) if mask.sum() else 0.0)
            summary[sym] = sums
        df = pd.DataFrame(summary)
        df.to_csv(str(summary_out), index=False)
        logging.info("Wrote summary: %s", summary_out)

    logging.info("Done. Processed %d frames.", len(out_frames))

if __name__ == "__main__":
    main()

