# Downscaling Project: Deep Learning for Temperature and Precipitation (DS Intern)

This project implements a multi-model downscaling pipeline for climate data (ERA5 and LMDZ datasets) over Morocco, targeting fine-resolution precipitation (MSWEP) and temperature (MSWT).

## 📂 Project Structure

This repository follows a flat structure for easier access and management:

- **`train.py`**: Main entry point for training the model.
- **`predict.py`**: Main entry point for generating predictions (multi-scenario).
- **`eval.py`**: Script for model evaluation, inference, and diagnostics.

### 🛠 Core Logic
- **`config.yaml`**: Hierarchical configuration file containing all training/evaluation parameters.
- **`config.py`**: Python wrapper to load and manage YAML configuration.
- **`data_loading.py`**: Handle loading NetCDF files for ERA5, MSWEP, and masking logic.
- **`preprocessing.py`**: Data normalization, unit conversion, and PyTorch tensor preparation.
- **`interpolation.py`**: Grid interpolation utilities to match predictor and target resolutions.
- **`utils.py`**: Common utility functions (printing, model saving, directory management).

### 🧠 Model Architectures & Logic
- **`unet_arch.py`**: U-Net architecture tailored for spatial downscaling.
- **`cnn.py`**: CNN-based architectural implementation.
- **`glm.py`**: Statistical baseline (Generalized Linear Model) implementation.
- **`losses.py`**: Custom loss functions (Gaussian for Temperature, Bernoulli-Gamma for Precipitation).
- **`training.py`**: Generic training loop logic (train/validation iteration).
- **`evaluation.py`**: Implementation of evaluation metrics and NetCDF results generation.

### 🐚 Run Scripts (SLURM)
- **`job_cpu.sh`**: Batch script to run the pipeline on SLURM-based CPU nodes.
- **`job_gpu.sh`**: Batch script to run the pipeline on SLURM-based GPU nodes.

### 📂 External Resources
- **`results/`**: Root folder for generated results (models, logs, outputs).
- **`Dataset/`**: Local storage for climate datasets (NetCDF).
- **`morocco_shapefile/`**: Region masking and geometry definitions.

---

## 🚀 Getting Started

1. **Environment Setup**: Activate the recommended environment:
   ```bash
   conda activate env_torch
   ```

2. **Configuration**: Edit `config.yaml` to specify:
   - `variable`: `temp` or `precip`
   - `model_type`: `unet`, `cnn`, or `glm`
   - `scenarios`: ERA5, HIST, SSP245, SSP585

3. **Running the Pipeline**:
   - For interactive training: `python train.py config.yaml`
   - For cluster execution: `sbatch job_cpu.sh` (ensure `train="yes"` in the script)

4. **Monitoring**: Logs are stored in `log.txt` (via SLURM) and experiment-specific directories in `results/`.
