#!/usr/bin/env python3
"""Convert Bader ACF.dat + XYZ frames to multi-frame EXTXYZ"""

from pathlib import Path
import argparse,logging,re,sys

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s")

VALENCE_ELECTRONS = {
    # Group 1 (Alkali Metals) - Includes semi-core (s,p) for Na+
    # Li includes 1s (q3)
    "H": 1, "Li": 3, "Na": 9, "K": 9, "Rb": 9, "Cs": 9, "Fr": 9,
    # Group 2 (Alkaline Earth Metals) - Includes semi-core (s,p) for Mg+
    # Be includes 1s (q4)
    "Be": 4, "Mg": 10, "Ca": 10, "Sr": 10, "Ba": 10, "Ra": 10,
    # Group 3 (Scandium Family) - q11 (s2 p6 d1 s2)
    "Sc": 11, "Y": 11, "La": 11, "Ac": 11,
    # Group 4 (Titanium Family) - q12 (s2 p6 d2 s2)
    "Ti": 12, "Zr": 12, "Hf": 12, "Th": 12,
    # Group 5 (Vanadium Family) - q13
    "V": 13, "Nb": 13, "Ta": 13, "Pa": 13,
    # Group 6 (Chromium Family) - q14
    "Cr": 14, "Mo": 14, "W": 14, "U": 14,
    # Group 7 (Manganese Family) - q15
    "Mn": 15, "Tc": 15, "Re": 15, "Np": 15,
    # Group 8 (Iron Family) - q16
    "Fe": 16, "Ru": 16, "Os": 16, "Pu": 16,
    # Group 9 (Cobalt Family) - q17
    "Co": 17, "Rh": 17, "Ir": 17, "Am": 17,
    # Group 10 (Nickel Family) - q18
    "Ni": 18, "Pd": 18, "Pt": 18, "Cm": 18,
    # Group 11 (Copper Family) - q11 (sometimes q19, but q11 is common GTH-PBE default)
    "Cu": 11, "Ag": 11, "Au": 11, "Bk": 19, # Bk is Actinide, follows f-trend
    # Group 12 (Zinc Family) - q12 (Standard GTH-PBE)
    "Zn": 12, "Cd": 12, "Hg": 12, "Cf": 20, # Cf is Actinide
    # Group 13 (Boron Family) - Ga/In/Tl include d-shell (q13)
    "B": 3, "Al": 3, "Ga": 13, "In": 13, "Tl": 13, "Es": 21, # Es is Actinide
    # Group 14 (Carbon Family) - Usually d is core (q4)
    "C": 4, "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4, "Fm": 22, # Fm is Actinide
    # Group 15 (Nitrogen Family)
    "N": 5, "P": 5, "As": 5, "Sb": 5, "Bi": 5,
    # Group 16 (Oxygen Family)
    "O": 6, "S": 6, "Se": 6, "Te": 6, "Po": 6,
    # Group 17 (Halogens)
    "F": 7, "Cl": 7, "Br": 7, "I": 7, "At": 7,
    # Group 18 (Noble Gases)
    "He": 2, "Ne": 8, "Ar": 8, "Kr": 8, "Xe": 8, "Rn": 8,
    # Remaining Lanthanides (Ce-Lu) - Linear increase q12 -> q25
    "Ce": 12, "Pr": 13, "Nd": 14, "Pm": 15, "Sm": 16, "Eu": 17, "Gd": 18,
    "Tb": 19, "Dy": 20, "Ho": 21, "Er": 22, "Tm": 23, "Yb": 24, "Lu": 25
}

SET_RE=re.compile(r"set_\d{3}")
FRAME_RE=re.compile(r"frame_(\d+)\.xyz")

def parse_acf(p):
    d={}
    for ln in open(p):
        if ln.strip() and ln.lstrip()[0].isdigit():
            s=ln.split(); d[int(s[0])]=float(s[4])
    return d

def read_xyz(p):
    with open(p) as f:
        n=int(f.readline()); f.readline()
        a=[f.readline().split() for _ in range(n)]
    return n,a

# regex to match floats (incl. scientific)
_NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_lattice_from_cp2k(inp_path: Path) -> str:
    """
    Robustly parse lattice vectors from a CP2K input file.

    Supported forms (case-insensitive):
      &CELL
        A ax ay az
        B bx by bz
        C cx cy cz
      &END

    or

      &CELL
        ABC a b c
      &END

    Also accepts unit tokens like '[angstrom]' anywhere on the line.
    Returns a string "Ax Ay Az Bx By Bz Cx Cy Cz" (9 floats, 6 decimal places).
    Raises RuntimeError if lattice cannot be determined.
    """
    text = inp_path.read_text()
    # try to extract &CELL ... &END block first (if present)
    m = re.search(r'&CELL\b(.*?)&END\b', text, flags=re.IGNORECASE | re.DOTALL)
    block = m.group(1) if m else text

    A = B = C = None

    for raw_ln in block.splitlines():
        ln = raw_ln.split('!')[0].strip()           # remove CP2K inline comments '!'
        if not ln:
            continue
        up = ln.upper()
        # lines like "A 16.845 0.0 0.0 [angstrom]" or "A  1.0 2.0 3.0"
        if up.startswith('A '):
            nums = _NUM_RE.findall(ln)
            if len(nums) >= 3:
                A = [float(x) for x in nums[:3]]
        elif up.startswith('B '):
            nums = _NUM_RE.findall(ln)
            if len(nums) >= 3:
                B = [float(x) for x in nums[:3]]
        elif up.startswith('C '):
            nums = _NUM_RE.findall(ln)
            if len(nums) >= 3:
                C = [float(x) for x in nums[:3]]
        # lines like "ABC 16.845 16.845 25.000 [angstrom]"
        elif up.startswith('ABC'):
            nums = _NUM_RE.findall(ln)
            if len(nums) >= 3:
                a, b, c = [float(x) for x in nums[:3]]
                A = [a, 0.0, 0.0]; B = [0.0, b, 0.0]; C = [0.0, 0.0, c]

        # if we have all three, stop early
        if A is not None and B is not None and C is not None:
            break

    # final fallback: try to find first occurrence of three floats on any single non-empty line
    if not (A and B and C):
        for raw_ln in block.splitlines():
            ln = raw_ln.split('!')[0].strip()
            if not ln:
                continue
            nums = _NUM_RE.findall(ln)
            if len(nums) >= 3:
                # treat as ABC if the line looks short (no leading letter)
                first_tok = ln.split()[0].upper()
                if first_tok not in ('A','B','C','ABC','&CELL'):
                    # assume this is ABC-style numbers somewhere — take as ABC
                    a,b,c = [float(x) for x in nums[:3]]
                    A = [a,0.0,0.0]; B = [0.0,b,0.0]; C = [0.0,0.0,c]
                    break

    if not (A and B and C):
        raise RuntimeError(f"Could not parse lattice vectors from CP2K input: {inp_path}")

    return " ".join(f"{v[i]:.6f}" for v in (A,B,C) for i in range(3))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--density-dir",default="cp2k_density")
    ap.add_argument("--out",default="charges.extxyz")
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()

    mode="w" if args.overwrite else "a"
    frames=0

    with open(args.out,mode) as fout:
        for s in sorted(Path(args.density_dir).iterdir()):
            if not SET_RE.match(s.name): continue
            cp2k_inp=next(s.glob("*.inp"),None)
            if not cp2k_inp: raise FileNotFoundError(f"No cp2k input in {s}")
            lattice=parse_lattice_from_cp2k(cp2k_inp)

            for xyz in sorted(s.glob("frame_*.xyz")):
                idx=int(FRAME_RE.match(xyz.name).group(1))
                acf=s/f"frame_{idx:06d}"/"ACF.dat"
                if not acf.exists():
                    logging.warning("Missing ACF.dat for %s",xyz); continue

                n,atoms=read_xyz(xyz); bader=parse_acf(acf)

                fout.write(
                    f'{n}\nLattice="{lattice}" '
                    'Properties=species:S:1:pos:R:3:bader_charges:R:1:net_charges:R:1 '
                    'pbc="T T T"\n'
                )

                for i,a in enumerate(atoms,1):
                    el,x,y,z=a[0],float(a[1]),float(a[2]),float(a[3])
                    if el not in VALENCE_ELECTRONS: raise KeyError(f"No valence for {el}")
                    net=VALENCE_ELECTRONS[el]-bader[i]
                    fout.write(f"{el:2s} {x:10.6f} {y:10.6f} {z:10.6f} {bader[i]:10.6f} {net:10.6f}\n")

                frames+=1

    logging.info("Wrote %d frames to %s",frames,args.out)

if __name__=="__main__":
    main()

