# Statistical Downscaling Repository (Temperature)

**Author:** M. El Aabaribaoune (@um6p) & S. El Mahboub

This repository contains the experiments and environments for statistical downscaling focusing on the **Temperature** variable. The different subdirectories correspond to the project's configurations (M0, M1, M2), aiming to harmonize and test different sources of predictors mapped to high-resolution grids (MSWT, LMDZ35).

## Subdirectory Contents

### 1. `era5_to_mswt` (M0 Experiment)
- **Description**: Main pipeline (baseline) for the downscaling of global reanalysis observations (ERA5) to the target reference data (MSWT).
- **Contents**:
  - `main/`: The core of the training and evaluation pipeline, including configurations for the CNN architecture (and Physics-Informed CNN).
  - `postproc/`: Post-processing scripts to generate maps, metrics (RMSE, Bias, etc.), and visual validations of the M0 experiment.
  - Metadata scratch and utility scripts.

### 2. `lmdz250_to_lmdz35` (M1 Experiment)
- **Description**: M1 experiment dedicated to learning the downscaling from the low-resolution global climate model LMDZ (250 km) to its higher-resolution regional version (35 km).
- **Contents**: 
  - `main/`: Training scripts and configurations (CNN) specifically adapted to use LMDZ 250km as input, based on the harmonized structure.
  - `postproc/`: Post-processing scripts and evaluation for the M1 experiment.

### 3. `lmdz35Upscaled2degre_to_lmdz35` (M2 Experiment)
- **Description**: M2 experiment ("Perfect Prog" or idealized test). Uses LMDZ 35km data that has been previously upscaled (coarsened to a common 2-degree grid) to attempt to reconstruct the original LMDZ 35km target. This serves to validate the intrinsic capacity of the network (CNN) to reconstruct spatial scales without the bias induced by changing the physical model (unlike ERA5 or LMDZ250).
- **Contents**:
  - `main/` and `postproc/` sharing the same architecture as the other folders, but specifically configured for the geometric tensors of the M2 experiment.



## Standard Experiment Architecture (`main/`)

Within the `era5_to_mswt`, `lmdz250_to_lmdz35`, and `lmdz35Upscaled2degre_to_lmdz35` folders, the internal structure of the `main/` directory is standardized to facilitate comparison between models:
- `configs/`: YAML configuration files managing hyperparameters, experiments (Xiong, Serifi), and physical constraints (`phy_ai`).
- `src/`: Modular source code for AI models (PyTorch), data loading (DataLoaders), and continuous/directional gradient loss functions.
- `scripts/`: Slurm submission scripts for the HPC cluster (`job_cpu.sh`, `job_gpu.sh`, `submit_cnn_jobs.sh`).
- `train.py` / `eval.py`: Model training and evaluation entry points.
