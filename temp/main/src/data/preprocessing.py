"""
==========================================================
 Script: preprocessing.py
 Author: M. El Aabaribaoune
 Description:
     Data preprocessing utilities:
     - Normalization of predictors
     - Unit conversion (Precipitation and Temperature)
     - Conversion to PyTorch tensors

 Notes:
     - CPU-only preprocessing (GPU handled in training loop)
     - Model-agnostic
==========================================================
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.core.utils import vprint

def _print_stats(label, arr, units=None):
    if arr is None:
        vprint(f"{label}: None")
        return

    u = f" [{units}]" if units else ""
    vprint(
        f"{label}{u} → min={np.nanmin(arr):.4g}, "
        f"max={np.nanmax(arr):.4g}, mean={np.nanmean(arr):.4g}"
    )

def _detect_units_from_source(cfg):
    """
    Detect target units from variable + target name.
    """
    target = cfg.target.lower()
    variable = cfg.variable.lower()

    if variable == "precip":
        if target in ["lmdz", "era5", "lmdz35"]:
            return "kg/m2/s"
        elif target in ["mswep", "ter"]:
            return "mm/day"
        elif target in ["imerg"]:
            return "mm/hr"
        else:
            vprint(f"Warning: unknown precipitation target '{cfg.target}', units unclear")
            return None

    elif variable == "temp":
        if target in ["mswt"]:
            return "degree_Celsius"
        elif target in ["era5", "lmdz", "tas"]:
            return "K"
        else:
            vprint(f"Warning: unknown temperature target '{cfg.target}', units unclear")
            return None

    else:
        vprint(f"Warning: unknown variable '{cfg.variable}', units unclear")
        return None


def _convert_units(arr, units, variable, label):
    vprint(f"--- Processing {variable} ({label}) ---")
    _print_stats(f"{label} raw", arr, units)

    if arr is None:
        return None, units

    if units is None:
        vprint("  Units unknown → skipping conversion")
        return arr, None

    if variable == "precip":
        if units in ["kg/m2/s", "kg/m^2/s", "kg/m²/s"]:
            vprint("  Converting kg/m²/s → mm/day")
            arr = arr * 86400.0
            units = "mm/day"
        elif units == "mm/hr":
            vprint("  Converting mm/hr → mm/day")
            arr = arr * 24.0
            units = "mm/day"
    
    elif variable == "temp":
        if units in ["K", "kelvin", "Kelvin"]:
            vprint("  Converting Kelvin → Celsius")
            arr = arr - 273.15
            units = "C"
        elif units in ["degree_Celsius", "Celsius", "C"]:
            vprint("  Units already in Celsius → skipping conversion")
            units = "C"

    _print_stats(f"{label} converted", arr, units)
    return arr, units

def preprocess_data(cfg, X, y_train, y_test):
    vprint("=== Preprocessing data ===")

    if X is None:
        raise ValueError("Input dataset X is missing.")

    # Extract predictor arrays
    x_train_np = X.sel(
        time=slice(cfg.start_date_train, cfg.end_date_train)
    ).values

    x_test_np = X.sel(
        time=slice(cfg.start_date_test, cfg.end_date_test)
    ).values

    # Normalization
    vprint(f"Applying normalization mode: {cfg.norm_mode}")

    if cfg.norm_mode == "global":
        mean = x_train_np.mean()
        std  = x_train_np.std()
        x_train_np = (x_train_np - mean) / (std + 1e-8)
        x_test_np  = (x_test_np  - mean) / (std + 1e-8)

    elif cfg.norm_mode == "channel":
        mean = x_train_np.mean(axis=(0, 2, 3), keepdims=True)
        std  = x_train_np.std (axis=(0, 2, 3), keepdims=True)
        x_train_np = (x_train_np - mean) / (std + 1e-8)
        x_test_np  = (x_test_np  - mean) / (std + 1e-8)

    elif cfg.norm_mode == "gridbox":
        t, c, h, w = x_train_np.shape
        x_train_std = np.zeros_like(x_train_np)
        x_test_std  = np.zeros_like(x_test_np)

        for lvl in range(c):
            for i in range(h):
                for j in range(w):
                    scaler = StandardScaler()
                    x_train_std[:, lvl, i, j] = scaler.fit_transform(
                        x_train_np[:, lvl, i, j].reshape(-1, 1)
                    ).ravel()
                    x_test_std[:, lvl, i, j] = scaler.transform(
                        x_test_np[:, lvl, i, j].reshape(-1, 1)
                    ).ravel()

        x_train_np = x_train_std
        x_test_np  = x_test_std

    elif cfg.norm_mode == "flatten":
        scaler = StandardScaler()
        x_train_flat = x_train_np.reshape(x_train_np.shape[0], -1)
        x_test_flat  = x_test_np.reshape(x_test_np.shape[0], -1)
        x_train_np = scaler.fit_transform(x_train_flat).reshape(x_train_np.shape)
        x_test_np  = scaler.transform(x_test_flat).reshape(x_test_np.shape)

    else:
        raise ValueError(f"Unknown norm_mode '{cfg.norm_mode}'")

    vprint("Normalization done.")

    # Target preprocessing (numpy → convert → tensor)
    y_train_np = y_train.numpy()
    y_test_np  = y_test.numpy()

    units = _detect_units_from_source(cfg)
    y_train_np, _ = _convert_units(y_train_np, units, cfg.variable, "train")
    y_test_np , _ = _convert_units(y_test_np , units, cfg.variable, "test")

    # DeepESD (Bano et al. 2022) methodology for precipitation:
    # Subtract 0.99 and clamp to 0 to better fit Bernoulli-Gamma and handle trace rain.
    if cfg.target.lower() == "mswep" and cfg.loss_type == "bernoulli_gamma":
        vprint("DeepESD Methodology: Subtracting 0.99 threshold from MSWEP precipitation")
        y_train_np = np.maximum(y_train_np - 0.99, 0)
        y_test_np  = np.maximum(y_test_np  - 0.99, 0)

    # Convert to CPU tensors
    x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
    x_test_tensor  = torch.tensor(x_test_np , dtype=torch.float32)
    
    # Ensure y has channel dimension (N, 1, H, W) for models (CNN, UNet, GLM)
    y_train_tensor = torch.tensor(y_train_np[:, None, :, :], dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test_np[:, None, :, :], dtype=torch.float32)

    vprint("=== Preprocessing complete ===")

    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor

