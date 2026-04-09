# Configuration Guide (config.yaml)

This document explains the parameters used in the YAML configuration files for the downscaling project.

## 1. General Settings (`general`)
*   **`experiment`**: Name of the experiment. Results will be saved in a folder with this name.
*   **`variable`**: The target variable to predict (e.g., `precip` for precipitation, `temp` for temperature).
*   **`target`**: The reference dataset name (e.g., `mswep` or `mswt`).
*   **`src`**: The predictor source (`era5` or `lmdz`).
*   **`model_type`**: Architecture to use (`unet`, `cnn`, `vit`, or `glm`).
*   **`interpolation_type`**: Method to use for initial spatial matching (`linear`, `nearest`, `bilinear`).
*   **`variables`**: List of predictor variables (e.g., `["z", "q", "t", "u", "v"]`).
*   **`levels`**: Vertical pressure levels for predictors (e.g., `[500, 700, 850]`).
*   **`resolution`**: The spatial resolution of the input data (e.g., `2.0` degrees).

## 2. Training Parameters (`training`)
*   **`learning_rate`**: Step size for the optimizer (e.g., `1e-3`).
*   **`epochs`**: Number of full passes through the training data.
*   **`batch_size`**: Number of samples processed before updating model weights.
*   **`loss_type`**: The loss function (`bernoulli_gamma` is specialized for precipitation).
*   **`norm_mode`**: Data normalization strategy (`gridbox` or `standard`).
*   **`early_stopping_max`**: Number of epochs to wait for improvement before stopping early.

## 3. Architecture Specifics (`vit`)
*   **`emb_size`**: Embedding dimension for Vision Transformers.
*   **`patch_size`**: Dimension of the square patches extracted from images.
*   **`num_layers`**: Number of Transformer layers.
*   **`num_heads`**: Number of attention heads.
*   **`dropout`**: Dropout rate for regularization.

## 4. Region Selection (`region`)
*   **`lon_min`, `lon_max`, `lat_min`, `lat_max`**: The geographic bounding box for the study area (e.g., Morocco).

## 5. Dates (`dates`)
*   **`train`**: Start and end dates for training the model.
*   **`test`**: Start and end dates for evaluating the model on independent data.

## 6. Paths & Patterns (`paths`)
*   **`root_dir`**: The base absolute path where datasets are stored.
*   **`results_dir`**: Folder where models and logs will be saved (relative to root).
*   **`shapefile_path`**: Path to the Morocco shapefile used for masking (relative to root).
*   **`era5_predictor_pattern`**: naming convention for ERA5 files (e.g., `{var}_1979-2020_levels.nc`).
*   **`mswep_path`**: Path to the target MSWEP file.

## 7. Mappings & Logic
*   **`lmdz_var_map`**: Maps standard variable names to specific LMDZ internal names.

## 8. Prediction Scenarios (`prediction`)
*   **`scenarios`**: A list of specialized runs (e.g., predicting SSP future scenarios or present-day LMDZ) with specific dates and bias-correction flags.
