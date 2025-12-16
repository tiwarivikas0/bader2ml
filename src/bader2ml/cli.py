#!/usr/bin/env python3
# src/bader2ml/cli.py
import argparse,sys
from pathlib import Path

from bader2ml.cp2k.prepare_cp2k_no_split import main as cp2k_prepare_main
from bader2ml.bader.runner import main as bader_run_main
from bader2ml.data.acf_to_exyz import main as acf2exyz_main
from bader2ml.data.exyz_to_db import main as exyz2db_main
from bader2ml.data.prepare_dataset import main as prepare_dataset_main
from bader2ml.train.train_schnet import main as train_charges_main
from bader2ml.predict.predict_charges import main as predict_charges_main

def main():
    p = argparse.ArgumentParser(prog="bader2ml")
    sp = p.add_subparsers(dest="cmd", required=True)

    # ----------------- DATA group -----------------
    data_parser = sp.add_parser("data", help="Data preparation utilities")
    data_sub = data_parser.add_subparsers(dest="data_cmd", required=True)

    acf2exyz = data_sub.add_parser("acf2exyz", help="Combine Bader ACF.dat and XYZ into EXTXYZ")
    acf2exyz.add_argument("--density-dir", default="cp2k_density", help="Directory containing set_XXX folders")
    acf2exyz.add_argument("--out", default="charges.extxyz", help="Output EXTXYZ file")
    acf2exyz.add_argument("--overwrite", action="store_true", help="Overwrite output instead of appending")

    exyz2db = data_sub.add_parser("exyz2db", help="Convert EXTXYZ -> ASE sqlite DB")
    exyz2db.add_argument("--extxyz", "-x", default="charges.extxyz", help="Input EXTXYZ")
    exyz2db.add_argument("--db", "-d", default="database.db", help="Output DB")
    exyz2db.add_argument("--index", "-i", default=":", help="Frame slice (':' default)")
    exyz2db.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    prepds = data_sub.add_parser("prepare-dataset", help="Prepare SchNet dataset and split")
    prepds.add_argument("--source-dir", "-s", default=".", help="Source ASE .db directory (default: .)")
    prepds.add_argument("--out-dir", "-o", default="schnet_db", help="Output directory (default: ./schnet_db)")
    prepds.add_argument("--cutoff", "-c", type=float, default=3.5, help="Environment cutoff (Å)")
    prepds.add_argument("--val-fraction", "-v", type=float, default=0.10, help="Validation fraction (default 0.10)")
    prepds.add_argument("--seed", "-r", type=int, default=42, help="Random seed")
    prepds.add_argument("--selection", default="bd_charge=yes", help="DB selection string")
    prepds.add_argument("--ase-property", default="charges", help="Charge property name in DB")
    prepds.add_argument("--verbose", "-V", action="store_true", help="Verbose logging")

    # ----------------- CP2K group -----------------
    cp2k = sp.add_parser("cp2k", help="CP2K workflows")
    cp2k_sp = cp2k.add_subparsers(dest="cp2k_cmd", required=True)

    prep = cp2k_sp.add_parser("prepare", help="Prepare CP2K jobs (no submission)")
    prep.add_argument("--xyz", "-x", default="input.xyz", help="Multi-frame xyz trajectory (default: input.xyz)")
    prep.add_argument("--cp2k-inp", "-i", required=True, help="CP2K input template")
    prep.add_argument("--pbs-template", "-p", required=True, help="PBS job template")
    prep.add_argument("--frames-per-set", "-n", type=int, required=True, help="Frames per CP2K set")
    prep.add_argument("--output-dir", "-o", default="cp2k_density", help="Output directory (default: cp2k_density)")
    prep.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # ----------------- BADER group -----------------
    bader_parser = sp.add_parser("bader", help="Bader charge analysis")
    bader_sub = bader_parser.add_subparsers(dest="bader_cmd", required=True)

    bader_run = bader_sub.add_parser("run", help="Run Bader analysis on CP2K density cubes (interactive node)")
    bader_run.add_argument("--density-dir", "-d", required=True, help="Directory with set_XXX folders (required)")
    bader_run.add_argument("--nprocs", "-j", type=int, default=1, help="Number of parallel Bader jobs (default: 1)")
    bader_run.add_argument("--bader-exec", default=None, help="Path to bader executable (default: bundled)")
    bader_run.add_argument("--resume", action="store_true", help="Resume from previous run")
    bader_run.add_argument("--overwrite", action="store_true", help="Overwrite existing ACF.dat and re-run Bader")
    bader_run.add_argument("--dry-run", action="store_true", help="Show what would run without executing Bader")

    # ----------------- TRAIN group -----------------
    train = sp.add_parser("train", help="Model training")
    train_sp = train.add_subparsers(dest="train_cmd", required=True)
    train_chg = train_sp.add_parser("charges", help="Train SchNet for partial charges")

    # use underscore names to match training script argument names (avoid hyphen/underscore mismatch)
    train_chg.add_argument("--db", "-d", default="schnet_db", help="SchNet database folder (default: schnet_db)")
    train_chg.add_argument("--epochs", type=int, default=1000)
    train_chg.add_argument("--lr", type=float, default=1e-3)
    train_chg.add_argument("--cutoff", type=float, default=3.5)

    train_chg.add_argument("--atom_basis", type=int, default=64)
    train_chg.add_argument("--filters", type=int, default=64)
    train_chg.add_argument("--gaussians", type=int, default=30)
    train_chg.add_argument("--interactions", type=int, default=5)
    train_chg.add_argument("--layers", type=int, default=2)

    train_chg.add_argument("--earlystop", type=int, default=10)
    train_chg.add_argument("--reducelr", type=int, default=0)
    train_chg.add_argument("--batch-size", type=int, default=32)
    train_chg.add_argument("--clean", action="store_true", help="Remove existing training folder before starting")
    train_chg.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
   # ================= PREDICT =================
    predict = sp.add_parser("predict", help="Model prediction")
    predict_sp = predict.add_subparsers(dest="predict_cmd", required=True)

    predict_chg = predict_sp.add_parser("charges",help="Predict Bader / net charges using a trained SchNet model",)

    predict_chg.add_argument("--model", "-m",required=True,help="Trained model file (e.g. best_model)",)
    predict_chg.add_argument("--input", "-i",default="traj.xyz",help="Input trajectory (.xyz or .lammpstrj)")
    predict_chg.add_argument("--output", "-o",default="traj-charges.xyz",help="Output trajectory with charges (extxyz)")
    predict_chg.add_argument("--type-map",required=True,help="Type-map file: two columns (type_id atom_symbol)")
    predict_chg.add_argument("--cutoff",type=float,default=3.5,help="Neighbor cutoff (Å)")
    predict_chg.add_argument("--every",type=int,default=1,help="Read every N frames")
    predict_chg.add_argument("--from-frame",type=int,default=0,help="Start frame index")
    predict_chg.add_argument("--upto",default="",help="End frame index")
    predict_chg.add_argument("--device",default="auto",help="Device: auto | cpu | cuda")
    predict_chg.add_argument("--summary",default=None,help="Write per-element charge summary CSV")
    predict_chg.add_argument("--no-write-traj",dest="write_traj",action="store_false",help="Do not write output trajectory")
    predict_chg.set_defaults(write_traj=True)

    # ----------------- parse args (only once) -----------------
    args = p.parse_args()

    # ----------------- DISPATCH -----------------
    if args.cmd == "cp2k" and args.cp2k_cmd == "prepare":
        print("[bader2ml] STEP 1: Preparing CP2K jobs (no submission)")
        argv = [
            "--xyz", args.xyz,
            "--cp2k-inp", args.cp2k_inp,
            "--pbs-template", args.pbs_template,
            "--frames-per-set", str(args.frames_per_set),
            "--output-dir", args.output_dir,
        ]
        if args.overwrite: argv.append("--overwrite")
        cp2k_prepare_main(argv)

        print("\n[bader2ml] CP2K preparation complete. Submit jobs from each set directory.")

    elif args.cmd == "bader" and args.bader_cmd == "run":
        print("[bader2ml] STEP 2: Running Bader charge analysis")
        argv = ["--density-dir", args.density_dir, "--nprocs", str(args.nprocs)]
        if args.bader_exec: argv += ["--bader-exec", args.bader_exec]
        if args.resume: argv += ["--resume"]
        if args.overwrite: argv += ["--overwrite"]
        if args.dry_run: argv += ["--dry-run"]
        bader_run_main(argv)

    elif args.cmd == "data" and args.data_cmd == "acf2exyz":
        print("[bader2ml] STEP 3: Converting ACF.dat to EXTXYZ")
        argv = ["--density-dir", args.density_dir, "--out", args.out]
        if args.overwrite: argv += ["--overwrite"]
        acf2exyz_main(argv)

    elif args.cmd == "data" and args.data_cmd == "exyz2db":
        argv = ["--extxyz", args.extxyz, "--db", args.db, "--index", args.index]
        if args.verbose: argv.append("--verbose")
        exyz2db_main(argv)

    elif args.cmd == "data" and args.data_cmd == "prepare-dataset":
        print("[bader2ml] STEP 4: Preparing SchNet dataset and split")
        argv = [
            "--source-dir", args.source_dir,
            "--out-dir", args.out_dir,
            "--cutoff", str(args.cutoff),
            "--val-fraction", str(args.val_fraction),
            "--seed", str(args.seed),
            "--selection", args.selection,
            "--ase-property", args.ase_property,
        ]
        if args.verbose: argv.append("--verbose")
        prepare_dataset_main(argv)

    elif args.cmd == "train" and args.train_cmd == "charges":
        print("[bader2ml] STEP 5: Training charge model (SchNet)")
        argv = [
            "--db", args.db,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--cutoff", str(args.cutoff),
            "--atom_basis", str(args.atom_basis),
            "--filters", str(args.filters),
            "--gaussians", str(args.gaussians),
            "--interactions", str(args.interactions),
            "--layers", str(args.layers),
            "--earlystop", str(args.earlystop),
            "--reducelr", str(args.reducelr),
            "--batch-size", str(args.batch_size),
        ]
        if args.clean: argv.append("--clean")
        if args.verbose: argv.append("--verbose")
        train_charges_main(argv)

    elif args.cmd == "predict" and args.predict_cmd == "charges":
        print("[bader2ml] STEP 6: Predicting charges")

        sys.argv = [
            "predict-charges",
            "--model", args.model,
            "--input", args.input,
            "--output", args.output,
            "--type-map", args.type_map,
            "--cutoff", str(args.cutoff),
            "--every", str(args.every),
            "--from", str(args.from_frame),
            "--upto", str(args.upto),
            "--device", args.device,
        ]

        if args.summary:
            sys.argv += ["--summary", args.summary]

        if not args.write_traj:
            sys.argv.append("--no-write-traj")

        predict_charges_main()

    else:
        p.error("Unknown command")

if __name__ == "__main__":
    main()

