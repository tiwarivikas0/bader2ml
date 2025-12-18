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

### SchNetPack installation using conda and pip
⚠️ Note: This package is **not compatible with versions higher than 1.0**.

```bash
conda activate
conda create -n schnetv1 python=3.8
conda activate schnetv1
pip install schnetpack==1.0.0
```
Verify installation:

```bash
python -c "import importlib.metadata as m; print(m.version('schnetpack'))"
```

### Bader2ml Installation

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
Prepare CP2K input files from a multi-frame XYZ trajectory. (**Note:** Keep number of frames around 500-1200)

```bash
bader2ml cp2k prepare --xyz input.xyz --cp2k-inp template.inp --pbs-template cp2k.pbs --frames-per-set 100
```
### What this does
- Splits the trajectory into sets
- Creates CP2K inputs with REFTRAJ blocks
- Extracts the first frame for COORD_FILE_NAME
- Names PBS jobs according to set ID

Make sure that your cp2k input file has this block inside the **&DFT block**:
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

CP2K jobs are **not submitted automatically**.  
Please submit the jobs **manually from each `set_XXX` directory**.

One-liner to submit all jobs (from cp2k_density):

```bash
for d in set_*; do (cd "$d" && qsub pbs.sh); done
```
Change pbs.sh to your pbs script name.

After all CP2K jobs have completed successfully, verify that everything is correct and then proceed to the second step.

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

**Suggestion:** Run the Bader charge analysis on an interactive node.  
Choose `nprocs` based on the available CPUs, as it controls the number of parallel Bader calculations.  
For instance, with 100 frames and `nprocs = 10`, the script executes 10 jobs concurrently.  
Proceed to the next step after all calculations finish.

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

- ACS Catalysis, **14**, 14652–14664 (2024) <br>
  https://doi.org/10.1021/acscatal.4c01920












