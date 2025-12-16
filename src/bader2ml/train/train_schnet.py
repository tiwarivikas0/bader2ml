#!/usr/bin/env python3
"""Train SchNet for partial charges (embedded atomrefs library, default)"""

import os,sys,argparse,logging
from pathlib import Path
import numpy as np,torch
import schnetpack as spk
from schnetpack import AtomsData
import schnetpack.train as trn
import schnetpack as spk

try:
    from ase.data import atomic_numbers
except Exception:
    atomic_numbers = None

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- embedded atom reference map (symbol -> valence electrons) ----------------
VALENCE_MAP = {
    # Group 1 (Alkali Metals)
    "H":1,"Li":3,"Na":9,"K":9,"Rb":9,"Cs":9,"Fr":9,
    # Group 2 (Alkaline Earth Metals)
    "Be":4,"Mg":10,"Ca":10,"Sr":10,"Ba":10,"Ra":10,
    # Group 3 (Scandium Family)
    "Sc":11,"Y":11,"La":11,"Ac":11,
    # Group 4 (Titanium Family)
    "Ti":12,"Zr":12,"Hf":12,"Th":12,
    # Group 5 (Vanadium Family)
    "V":13,"Nb":13,"Ta":13,"Pa":13,
    # Group 6 (Chromium Family)
    "Cr":14,"Mo":14,"W":14,"U":14,
    # Group 7 (Manganese Family)
    "Mn":15,"Tc":15,"Re":15,"Np":15,
    # Group 8 (Iron Family)
    "Fe":16,"Ru":16,"Os":16,"Pu":16,
    # Group 9 (Cobalt Family)
    "Co":17,"Rh":17,"Ir":17,"Am":17,
    # Group 10 (Nickel Family)
    "Ni":18,"Pd":18,"Pt":18,"Cm":18,
    # Group 11 (Copper Family)
    "Cu":11,"Ag":11,"Au":11,"Bk":19,
    # Group 12 (Zinc Family)
    "Zn":12,"Cd":12,"Hg":12,"Cf":20,
    # Group 13 (Boron Family)
    "B":3,"Al":3,"Ga":13,"In":13,"Tl":13,"Es":21,
    # Group 14 (Carbon Family)
    "C":4,"Si":4,"Ge":4,"Sn":4,"Pb":4,"Fm":22,
    # Group 15 (Nitrogen Family)
    "N":5,"P":5,"As":5,"Sb":5,"Bi":5,
    # Group 16 (Oxygen Family)
    "O":6,"S":6,"Se":6,"Te":6,"Po":6,
    # Group 17 (Halogens)
    "F":7,"Cl":7,"Br":7,"I":7,"At":7,
    # Group 18 (Noble Gases)
    "He":2,"Ne":8,"Ar":8,"Kr":8,"Xe":8,"Rn":8,
    # Lanthanides (Ce-Lu)
    "Ce":12,"Pr":13,"Nd":14,"Pm":15,"Sm":16,"Eu":17,"Gd":18,
    "Tb":19,"Dy":20,"Ho":21,"Er":22,"Tm":23,"Yb":24,"Lu":25
}
# ------------------------------------------------------------------------------------------

def build_atomrefs_from_map(max_z=100):
    """Create atomrefs array (max_z+1,1) from VALENCE_MAP using ASE atomic_numbers mapping where available."""
    arr = np.zeros((max_z+1,1),dtype=float)
    if atomic_numbers is None:
        # fallback: try simple periodic table mapping for common elements (minimal)
        logging.warning("ASE not available; attempting best-effort mapping for atomrefs")
        sym_to_z = {sym: i for i,sym in enumerate([
            "X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
            "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
            "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
            "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
            "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"
        ])}
        for sym,val in VALENCE_MAP.items():
            Z = sym_to_z.get(sym)
            if Z is None or Z>max_z: continue
            arr[Z,0] = float(val)
    else:
        for sym,val in VALENCE_MAP.items():
            Z = atomic_numbers.get(sym)
            if Z is None or Z>max_z: continue
            arr[Z,0] = float(val)
    return arr

def parse_cli(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--db","-d",required=True,help="Database folder (contains schnet.db and split.npz)")
    p.add_argument("--atomrefs","-a",default=None,help="Path to atomrefs .npy (optional override)")
    p.add_argument("--max-z","-m",type=int,default=100,help="Maximum atomic number for atomrefs (default 100)")
    p.add_argument("--atom_basis","-b",type=int,default=64)
    p.add_argument("--filters","-f",type=int,default=64)
    p.add_argument("--gaussians","-g",type=int,default=30)
    p.add_argument("--interactions","-i",type=int,default=5)
    p.add_argument("--cutoff","-c",type=float,default=3.5)
    p.add_argument("--layers","-l",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--epochs","-e",type=int,default=1000)
    p.add_argument("--earlystop",type=int,default=10)
    p.add_argument("--reducelr",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--verbose","-v",action="store_true")
    return p.parse_args(argv)

def load_atomrefs(path):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"atomrefs not found: {p}")
    arr=np.load(str(p))
    if arr.ndim==1: arr=arr.reshape(-1,1)
    if arr.shape[1]!=1: raise RuntimeError("atomrefs must be shape (max_z+1,1)")
    return arr.astype(float)

def main(argv=None):
    args=parse_cli(argv)
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)

    db_folder = args.db if args.db.endswith(os.sep) else args.db+os.sep
    db_file = db_folder + 'schnet.db'
    split_file = db_folder + 'split.npz'
    logging.info("DB: %s",db_file); logging.info("Split: %s",split_file)

    # atomrefs: load if provided, else build from embedded map
    if args.atomrefs:
        atomrefs = load_atomrefs(args.atomrefs)
        logging.info("Loaded atomrefs from %s shape=%s",args.atomrefs,atomrefs.shape)
    else:
        atomrefs = build_atomrefs_from_map(max_z=args.max_z)
        logging.info("Built embedded atomrefs shape=%s (max_z=%d)",atomrefs.shape,args.max_z)

    max_z = atomrefs.shape[0]-1
    logging.info("Using atomrefs up to Z=%d (atomrefs[8,0]=%s)",max_z,atomrefs[8,0] if atomrefs.shape[0]>8 else "N/A")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: %s",device)

    tr_folder = f"train-cut_{args.cutoff}-sch_{args.atom_basis}_{args.filters}_{args.gaussians}_{args.interactions}-nn_{args.atom_basis}_{args.layers}-opt_{args.lr}-es_{args.earlystop}-rlr_{args.reducelr}_{args.epochs}/"
    Path(tr_folder).mkdir(parents=True,exist_ok=True)
    logging.info("Training folder: %s",tr_folder)

    target='charge'; sum_target='sum_'+target

    # dataset
    environment_provider = spk.environment.AseEnvironmentProvider(args.cutoff)
    ds = AtomsData(db_file, available_properties=[target,sum_target],environment_provider=environment_provider)
    logging.info("Dataset size: %d", len(ds)); logging.info("Available properties: %s", ds.available_properties)

    # load split
    train,val,test = spk.train_test_split(data=ds, split_file=split_file)
    train_loader = spk.AtomsLoader(train,batch_size=args.batch_size,shuffle=True)
    val_loader = spk.AtomsLoader(val,batch_size=args.batch_size)

    # model
    schnet = spk.representation.SchNet(n_atom_basis=args.atom_basis,n_filters=args.filters,
        n_gaussians=args.gaussians,n_interactions=args.interactions,cutoff=args.cutoff,
        cutoff_network=spk.nn.cutoff.CosineCutoff)
    output_nn = spk.atomistic.Atomwise(n_in=args.atom_basis,n_layers=args.layers,
        property=sum_target,contributions=target,atomref=atomrefs)
    model = spk.AtomisticModel(representation=schnet,output_modules=output_nn)

    # loss functions
    alpha_sum=1.0
    def mse_loss_atom(batch,result):
        diff=batch[target]-result[target]; return torch.mean(diff**2)
    def mse_loss_atom_and_sum(batch,result):
        nat=batch[target].shape[1]; diff=batch[target]-result[target]; err=torch.mean(diff**2)
        diff_sum=batch[sum_target]-result[sum_target]; err_sum=torch.mean(diff_sum**2)/(nat**2)
        return err + alpha_sum*err_sum

    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

    metrics=[spk.metrics.MeanAbsoluteError(target),spk.metrics.MeanAbsoluteError(sum_target)]
    hooks = [ trn.CSVHook(log_path=tr_folder+'.', metrics=metrics) ]
    if args.earlystop>0: hooks.append(trn.EarlyStoppingHook(patience=args.earlystop))
    if args.reducelr>0: hooks.append(trn.ReduceLROnPlateauHook(optimizer,patience=args.reducelr,factor=0.9,min_lr=1e-6,stop_after_min=True))

    trainer=trn.Trainer(model_path=tr_folder,model=model,hooks=hooks,loss_fn=mse_loss_atom,
        optimizer=optimizer,train_loader=train_loader,validation_loader=val_loader,)

    trainer.train(device=device,n_epochs=args.epochs)
    logging.info("Training finished")

if __name__=="__main__":
    main()

