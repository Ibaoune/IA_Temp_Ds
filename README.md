Voici un **README complet prêt à copier-coller dans un fichier `.txt`** :

---

# Downscaling Project: Deep Learning for Temperature and Precipitation (DS Intern)

This project implements a modular and extensible deep learning pipeline for climate data downscaling over Morocco. It supports multiple model architectures (U-Net, CNN, and GLM) to generate high-resolution precipitation (MSWEP) and temperature (MSWT) from coarse-resolution datasets such as ERA5 and LMDZ.

---

## 1. Project Overview

Climate downscaling aims to transform coarse-resolution climate data into finer spatial resolution outputs. This project provides:

* Deep learning models for spatial downscaling
* Statistical baseline (GLM)
* Support for multiple climate scenarios (ERA5, HIST, SSP245, SSP585)
* End-to-end pipeline: data loading → preprocessing → training → evaluation

---

## 2. Project Structure

```
.
├── train.py              # Main training entry point
├── eval.py               # Evaluation and inference script
├── configs/              # Configuration files (organized by model)
├── src/                  # Source code (models, data, core logic)
├── scripts/              # SLURM job scripts
├── results/              # Outputs (models, logs, predictions)
└── README.md
```

---

## 3. Configuration System

Configurations are stored in the `configs/` directory and organized by model type:

```
configs/
├── cnn/
├── glm/
└── unet/
    ├── config.yaml
    └── test.yaml
```

### Key Parameters

Each YAML configuration file typically includes:

* `variable`: `temp` or `precip`
* `model_type`: `unet`, `cnn`,  `glm`
* `scenario`: ERA5, HIST, SSP245, SSP585
* `training parameters`: batch size, learning rate, epochs, etc.
* `data paths`: input/output dataset locations

Configuration loading is handled by:

```
src/core/config.py
```

---

## 4. Source Code Details

### 4.1 Models (`src/models/`)

* `unet_arch.py`: U-Net architecture for spatial prediction
* `cnn.py`: Convolutional neural network baseline
* `glm.py`: Generalized Linear Model (statistical baseline)

---

### 4.2 Data Processing (`src/data/`)

* `data_loading.py`: Loads NetCDF datasets (ERA5, LMDZ, MSWEP/MSWT)
* `preprocessing.py`: Data normalization and tensor preparation
* `interpolation.py`: Spatial interpolation between grids

---

### 4.3 Core Logic (`src/core/`)

* `training.py`: Training loop and validation logic
* `evaluation.py`: Metrics computation and result generation
* `losses.py`: Custom loss functions:

  * Gaussian loss (temperature)
  * Bernoulli-Gamma loss (precipitation)
* `utils.py`: Utilities (logging, saving, directory management)
* `config.py`: YAML configuration parser

---

## 5. Data Requirements

This project requires external datasets (not included in the repository):

* ERA5 reanalysis data
* MSWEP (precipitation) / MSWT (temperature)
* LMDZ model outputs
* Morocco shapefile (for spatial masking)

Ensure all dataset paths are correctly specified in the configuration files.

---

## 6. Installation & Environment

Activate your Python environment:

```
conda activate clean_env_Pytorch
```

Make sure required libraries are installed (PyTorch, NumPy, xarray, etc.).

---

## 7. Usage

### 7.1 Training

Run training with a selected configuration:

```
python train.py configs/unet/config.yaml
```

---

### 7.2 Evaluation / Inference

```
python eval.py configs/unet/test.yaml
```

---

### 7.3 Running on SLURM Cluster

CPU job:

```
sbatch scripts/job_cpu.sh
```

GPU job:

```
sbatch scripts/job_gpu.sh
```

Make sure scripts are properly configured (paths, environment, flags).

---

## 8. Outputs

All outputs are stored in the `results/` directory:

* Trained models (checkpoints)
* Logs (training progress)
* Predictions
* Evaluation metrics (possibly NetCDF format)

---

## 9. Extending the Project

### Adding a New Model

1. Add model implementation in:

```
src/models/
```

2. Register or integrate it in the training pipeline

3. Create a new configuration:

```
configs/<your_model>/
```

---

### Adding New Data

* Update `data_loading.py` if needed
* Modify preprocessing steps in `preprocessing.py`
* Adjust configuration paths

---

## 10. Features

* Modular architecture
* Multi-model support (DL + statistical)
* Flexible configuration system
* Custom loss functions for climate variables
* HPC-ready (SLURM support)
* NetCDF-compatible outputs

---

## 11. Notes

* The pipeline is designed to be model-agnostic and easily extendable.
* Evaluation outputs can be used directly with climate analysis tools.
* Proper preprocessing and normalization are critical for model performance.

---

