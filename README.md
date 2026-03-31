# Downscaling Project: Deep Learning for Temperature and Precipitation

This project implements a multi-model downscaling pipeline for climate data (ERA5 and LMDZ datasets) over Morocco, targeting fine-resolution precipitation (MSWEP) and temperature (MSWT).

## 🚀 Key Features
- **Multi-Model Support**: CNN, UNet, and GLM (Generalized Linear Model) implementations.
- **Unified Pipeline**: Single configuration for both training and prediction.
- **Bias Correction**: Includes Statistical Downscaling and Bias Correction (SDM) for LMDZ scenarios.
- **ERA5 & LMDZ Integration**: Capability to evaluate models on reanalysis data and GCM historical/future scenarios (SSP245, SSP585).

## 📂 Project Structure

### 📁 `general/` (Core Logic)
- **`config.py`**: Configuration class to load parameters from YAML files.
- **`config.yaml`**: Primary configuration file for all experiments.
- **`data_loading.py`**: Robust data loader for ERA5 and LMDZ, including variable mapping and masking.
- **`preprocessing.py`**: Data normalization, unit conversion, and PyTorch tensor preparation.
- **`interpolation.py`**: Grid interpolation utilities to match predictor and target resolutions.
- **`bias_correction.py`**: Implementation of SDM for LMDZ data bias correction.

### 📁 `training/` (Model Implementation)
- **`train.py`**: Main script for training models with early stopping.
- **`predict.py`**: Main script for generating predictions across various scenarios.
- **`cnn.py` / `unet_arch.py`**: Neural network architectures tailored for spatial downscaling.
- **`glm.py`**: Statistical baseline (Generalized Linear Model) implementation.
- **`losses.py`**: Loss functions (Gaussian for Temperature, Bernoulli-Gamma for Precipitation).

### 📁 `running/` (Launch Scripts)
- **`predict.sh`**: Bash script to launch the prediction pipeline with SLURM/Conda support.
- **`training.sh`**: Bash script for starting model training.

### 📁 `prediction/` (Output - Ignored by Git)
Root folder for generated `.nc` (NetCDF) prediction files, organized by experiment.

### 📁 `results/` (In Git)
Contains experiment metadata and models:
- **`models/`**: Saved `.pth` (Torch) or `.pkl` (GLM) model weights.
- **`logs/`**: Execution logs and loss history.
- **`config.txt`**: Snapshot of the configuration used for the experiment.

---

## 🛠️ Installation & Setup

1. **Environment**: Recommended environment `env_torch` (Conda).
   ```bash
   conda activate env_torch
   ```
2. **Paths**: Update the `general/config.yaml` with your local `Dataset/` paths.
3. **Run Prediction**:
   ```bash
   bash running/predict.sh general/config.yaml
   ```

## 📋 Configuration

The `general/config.yaml` controls everything:
- **Experiment**: Change `experiment` and `model_type` to switch between models.
- **Variable**: Change `variable` between `temp` and `precip`.
- **Target**: Set your target dataset resolution and path.
- **Scenarios**: Define which years/datasets to project (ERA5, LMDZ HIST, SSPs).

---
