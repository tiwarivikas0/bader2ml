#!/usr/bin/env python3
"""Convert multi-frame EXTXYZ -> ASE sqlite DB (always overwrite)"""

import argparse,logging,sys
from pathlib import Path
import numpy as np
from ase.io import iread
from ase.db import connect

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s")

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--extxyz","-x",default="charges.extxyz",help="Input EXTXYZ")
    p.add_argument("--db","-d",default="database.db",help="Output ASE DB (always overwritten)")
    p.add_argument("--index","-i",default=":",help="Frame slice (ASE notation)")
    p.add_argument("--fallback","-f",default="bader_charges",help="Fallback array if net_charges missing")
    p.add_argument("--verbose","-v",action="store_true",help="Verbose logging")
    args=p.parse_args(argv)

    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    extxyz=Path(args.extxyz); dbp=Path(args.db)
    if not extxyz.exists(): logging.error("EXTXYZ not found: %s",extxyz); sys.exit(2)

    if dbp.exists():
        logging.info("Overwriting existing DB: %s",dbp)
        dbp.unlink()

    written=0
    logging.info("Reading %s -> writing %s (index=%s)",extxyz,dbp,args.index)

    with connect(str(dbp)) as db:
        for i,atoms in enumerate(iread(str(extxyz),format="extxyz",index=args.index)):
            logging.debug("Frame %d: natoms=%d",i,len(atoms))

            if "net_charges" in atoms.arrays:
                charges=atoms.arrays["net_charges"]
            elif args.fallback in atoms.arrays:
                charges=atoms.arrays[args.fallback]
            else:
                logging.warning("Frame %d: no charge array found -> skipped",i)
                continue

            charges=np.asarray(charges,dtype=float)
            data={"charges":charges.tolist(),"sum_charge":float(np.sum(charges))}
            kv={"bd_charge":"yes"}

            db.write(atoms,data=data,key_value_pairs=kv)
            written+=1

    logging.info("Done. Wrote %d frames to %s",written,dbp)

if __name__=="__main__":
    main()

