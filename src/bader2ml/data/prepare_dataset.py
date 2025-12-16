#!/usr/bin/env python3
"""Prepare SchNet dataset DB and train/val split from ASE .db files (minimal user input)"""

import sys,os,glob,argparse,logging
from pathlib import Path
import numpy as np
from ase.db import connect
import schnetpack as spk
from schnetpack import AtomsData

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s")

def collect_frames_from_dbs(source_dir,selection="bd_charge=yes"):
    source=Path(source_dir)
    dbs=sorted(glob.glob(str(source/"*.db")))
    if not dbs: raise RuntimeError(f"No .db files found in {source}")
    traj_list,idxs_list,data_list=[],[],[]
    logging.info("Found %d db files under %s",len(dbs),source)
    for database in dbs:
        traj,idxs,data=[],[],[]
        with connect(database) as db:
            sel=db.select(selection)
            num=db.count(selection)
            logging.info("Scanning %s -> %d selected rows",Path(database).name,num)
            if num==0: continue
            for row in sel:
                atoms=row.toatoms(); traj.append(atoms); idxs.append(row.id); data.append(row.data)
        traj_list.append(traj); idxs_list.append(idxs); data_list.append(data)
    return traj_list,idxs_list,data_list

def make_property_list_from_rowdata(traj,data_list,ase_property_name="charges",target="charge"):
    prop_list=[]
    sum_target="sum_"+target
    for atoms,row_data in zip(traj,data_list):
        if ase_property_name not in row_data: logging.debug("Skipping row without %s",ase_property_name); continue
        arr=np.asarray(row_data[ase_property_name],dtype=np.float32)
        arr=arr.reshape(-1,1)            # shape (natoms,1)
        sumv=np.asarray([float(np.sum(arr))],dtype=np.float32)
        prop_list.append({target:arr,sum_target:sumv})
    return prop_list

def parse_cli(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir","-s",default=".",help="Folder to search for source ASE .db files (default: current dir)")
    p.add_argument("--out-dir","-o",default="./schnet_db",help="Output folder for schnet.db and split.npz (default: current dir)")
    p.add_argument("--cutoff","-c",type=float,default=3.5,help="Environment cutoff (default 3.5 Å)")
    p.add_argument("--val-fraction","-v",type=float,default=0.10,help="Validation fraction (default 0.10)")
    p.add_argument("--seed","-r",type=int,default=42,help="Random seed for split (default 42)")
    p.add_argument("--selection",default="bd_charge=yes",help="DB selection string (default: bd_charge=yes)")
    p.add_argument("--ase-property",default="charges",help="Property name in DB holding per-atom charges (default: charges)")
    p.add_argument("--verbose","-V",action="store_true",help="Verbose logging")
    return p.parse_args(argv)

def main(argv=None):
    args=parse_cli(argv)
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    src=Path(args.source_dir).expanduser().resolve(); out=Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    db_file=str(out/"schnet.db"); split_file=str(out/"split.npz")
    logging.info("Source dir: %s",src); logging.info("Output DB: %s",db_file)

    # collect frames
    traj_list,idxs_list,data_list = collect_frames_from_dbs(src,selection=args.selection)

    # create/overwrite ASE schnet.db
    if Path(db_file).exists():
        logging.info("Overwriting existing DB: %s",db_file); Path(db_file).unlink()

    environment_provider = spk.environment.AseEnvironmentProvider(args.cutoff)
    new_dataset = AtomsData(db_file, available_properties=["charge","sum_charge"], environment_provider=environment_provider)

    # Add systems
    total_added=0
    for traj,data in zip(traj_list,data_list):
        prop_list = make_property_list_from_rowdata(traj,data,ase_property_name=args.ase_property,target="charge")
        if not prop_list:
            logging.debug("No valid rows in one DB chunk -> skipping")
            continue
        new_dataset.add_systems(traj,property_list=prop_list)
        total_added += len(prop_list)
        logging.info("Added %d systems (cumulative %d)",len(prop_list),total_added)

    if total_added==0: raise RuntimeError("No frames were added to the dataset (check selection/property names)")

    logging.info("Total systems in dataset: %d",len(new_dataset))
    # create split: compute n_val as fraction of total (rounded)
    total=len(new_dataset); n_val=max(1,int(round(total*args.val_fraction))); n_train=max(0,total-n_val)
    logging.info("Split: total=%d -> train=%d val=%d (seed=%d)",total,n_train,n_val,args.seed)
    
    np.random.seed(args.seed)
    spk.train_test_split(data=new_dataset,num_train=n_train,num_val=n_val,split_file=split_file)
    logging.info("Wrote split to %s",split_file)
    logging.info("Done. Dataset ready for training.")

if __name__=="__main__":
    main()

