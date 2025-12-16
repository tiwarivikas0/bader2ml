# bader2ml
**End-to-End Machine-Learned Bader Charges from CP2K Electron Density**

`bader2ml` is a Python workflow that automates the complete pipeline for learning the Bader charges using Neural network:

**CP2K → Electron Density → Bader Charges → Dataset → SchNet Training → Charge Prediction**

---

## Installation

### Requirements

- Python ≥ 3.5
- CP2K (external)
- SchNetPack v1.0 (https://github.com/atomistic-machine-learning/SchNet)

### SchNetPack installation tip
Install in in a conda enviroment and version 1.0 from the link given above. (This will not work with other higher versions)

### Install

```bash
git clone https://github.com/tiwarivikas0/bader2ml.git
cd bader2ml
pip install -e .
```

### Verify installation

```bash
bader2ml --help
```
## STEP 1 — CP2K Electron Density Preparation
Prepare CP2K input files from a multi-frame XYZ trajectory.

```bash
bader2ml cp2k prepare --xyz input.xyz --cp2k-inp template.inp --pbs-template cp2k.pbs --frames-per-set 100
```
### What this does
- Splits the trajectory into sets
- Creates CP2K inputs with REFTRAJ blocks
- Extracts the first frame for COORD_FILE_NAME
- Names PBS jobs according to set ID

Make sure that your cp2k input file has this block:
&PRINT <br>
  &E_DENSITY_CUBE <br>
       FILENAME valence_density <br>
       STRIDE 1 <br>
  &END E_DENSITY_CUBE <br>
&END PRINT <br>

### Output
cp2k_density/ <br>
├── set_000/ <br>
│   ├── coords.xyz <br>
│   ├── ff_input.xyz <br>
│   ├── cp2k.inp <br>
│   └── job.pbs <br>

CP2K jobs are not submitted automatically.
Submit manually from each set_XXX directory. Once the cp2k jobs have complete you can move to the second step and verify that everything is fine.

## STEP 2 — Bader Charge Analysis
Run Bader analysis on CP2K electron density cube files.

```bash
bader2ml bader run --density-dir cp2k_density --nprocs 8
```
### Features
- Parallel execution
- Restart support
- Interactive-node friendly
- Optional overwrite

Suggestion: Use interactive node to do the Bader charge analysis. Set the value of nprocs to high number depending upon the number of CPUs you have on your interactive node. (number of nprocs = no of parrallel Bader charge calculations) Lets say you have total 100 frames and you set nprocs to 10 then this script will run 10 jobs at a time. And once all the calculation have completed go to the next step.

## STEP 3 — Combine ACF.dat → EXTXYZ
Combine Bader charges (ACF.dat) and atomic coordinates into a single multi-frame EXTXYZ file.

```bash
bader2ml data acf2exyz --density-dir cp2k_density --out charges.extxyz --overwrite
```
This step will combine you coordinates, lattice constants, Bader and net charges to create a extended xyz trajectory file. The lattice constants will be read from your cp2k input file.

## STEP 4 — EXTXYZ → SchNet Dataset
Convert EXTXYZ into an ASE SQLite database.

```bash
bader2ml data exyz2db --extxyz charges.extxyz --db database.db
```
## STEP 5 — Prepare Training Dataset & Split

```bash
bader2ml data prepare-dataset
```
### Defaults
- Output directory: schnet_db/
- Validation fraction: 10%
- Random seed: 42

## STEP 6 — Train SchNet (Charges)
Train a SchNet model to predict atomic charges.

```bash
bader2ml train charges --db schnet_db --epochs 1000
```

### Features
- Embedded atom reference library
- Early stopping
- Reduce-LR-on-plateau

## STEP 7 — Predict Charges on New Trajectories

```bash
bader2ml predict charges --model best_model --input traj.lammpstrj --type-map typemap.dat --output traj-charges.xyz
```
The typemap.dat files should contain the atom id and atom name mapping for example:

1 Mg <br>
2 O <br>
3 Au <br>

### Supported format
LAMMPS Trjectory. (Do not use trajectory saved using VMD)

### Output
EXTXYZ trajectory with net_charges and bader_charges. The dynamic atomic charges can we visualized in OVITO visualization software.

## Citations

If you use this code, please cite the following works:

- Journal of Chemical Physics, **163**, 214715 (2025)  
  https://doi.org/10.1063/5.0287822

- Proceedings of the National Academy of Sciences, **121**, e2313023120 (2024)  
  https://doi.org/10.1073/pnas.2313023120

- ACS Catalysis, **14**, 14652–14664 (2024)
  https://doi.org/10.1021/acscatal.4c01920












