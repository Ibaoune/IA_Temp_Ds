"""
==========================================================
 Script: predict.py
 Author: Hassan
 Description:
     Unified prediction script for the downscaling project.
     - Loads configuration and data
     - Loads trained models (UNet, CNN, ViT, GLM)
     - Handles probabilistic outputs
     - Saves results as NetCDF in results directory
==========================================================
"""

import torch
import sys
import os
import xarray as xr
import numpy as np
from config import load_config
from data_loading import load_datasets
from preprocessing import preprocess_data
from utils import vprint, load_model, set_verbose, build_experiment_path
from training import _build_model

def perform_single_prediction(cfg, scenario=None):
    """
    Runs prediction for a single scenario.
    """
    if scenario:
        vprint(f"\n--- Starting Scenario: {scenario['name']} ---")
        cfg.src = scenario["src"]
        cfg.start_date_test = scenario["start"]
        cfg.end_date_test = scenario["end"]
        cfg.folder = scenario.get("folder")
        apply_bc = scenario.get("bias_correction", False)
    else:
        apply_bc = False

    # 1. Load prediction data
    datasets = load_datasets(cfg)
    X = datasets[0] # xarray.Dataset (predictors)
    y_train = datasets[1]
    y_test = datasets[2]
    lon_out, lat_out, time_test = datasets[5], datasets[6], datasets[8]

    # 2. Apply Bias Correction if requested
    if apply_bc and cfg.src == "lmdz":
        vprint("Applying Bias Correction (SDM) for LMDZ...")
        from bias_correction import scaling_delta_mapping
        
        # We need historical LMDZ and ERA5 for the same reference period
        # Let's say 1979-2020 as reference
        orig_start = cfg.start_date_test
        orig_end = cfg.end_date_test
        orig_src = cfg.src

        # Load ERA5 hist (reference) - use reference period from config
        ref_start, ref_end = "1979-01-01", "1983-12-31" # Standard training period
        cfg.src = "era5"
        cfg.start_date_test = ref_start
        cfg.end_date_test = ref_end
        ds_era5_ref = load_datasets(cfg)[0]

        # Load LMDZ hist (GCM hist) for the same reference period
        cfg.src = "lmdz"
        cfg.folder = cfg.bc_reference_folder
        cfg.start_date_test = ref_start
        cfg.end_date_test = ref_end
        ds_lmdz_hist = load_datasets(cfg)[0]

        # Apply correction to the full/target LMDZ data
        vprint(f"Correcting predictors for period {orig_start} to {orig_end} using reference {ref_start}-{ref_end}...")
        X_corrected = scaling_delta_mapping(X, ds_lmdz_hist, ds_era5_ref.reindex_like(X, method="nearest"))
        X = X_corrected
        
        # Store reference stats for normalization
        ref_mean = ds_era5_ref.mean(dim=["time", "lat", "lon"])
        ref_std = ds_era5_ref.std(dim=["time", "lat", "lon"])

        # Restore original config
        cfg.start_date_test = orig_start
        cfg.end_date_test = orig_end
        cfg.src = orig_src

    # 3. Normalization (Handle LMDZ scenarios by referencing ERA5 stats if necessary)
    if apply_bc and cfg.src == "lmdz":
        vprint("Using Reference ERA5 stats for LMDZ normalization...")
        # Custom normalization for LMDZ target data using stored ERA5 baseline
        x_values = X.values.astype("float32")
        mean_vals = ref_mean.values.reshape(1, -1, 1, 1)
        std_vals = ref_std.values.reshape(1, -1, 1, 1)
        x_test_np = (x_values - mean_vals) / (std_vals + 1e-8)
        x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
    else:
        _, x_test_tensor, _, _ = preprocess_data(cfg, X, y_train, y_test)
    
    # 4. Load Model
    if not hasattr(perform_single_prediction, "cached_model"):
        vprint("Loading model...")
        if cfg.model_type == "glm":
            model, _, _ = load_model(cfg, model=None)
        else:
            model = _build_model(cfg, x_test_tensor, y_test.unsqueeze(1)).to(cfg.device)
            model, _, _ = load_model(cfg, model=model)
            model.eval()
        perform_single_prediction.cached_model = model
    
    model = perform_single_prediction.cached_model
    
    # 5. Perform Prediction
    if cfg.model_type == "glm":
        preds_np = model.predict(x_test_tensor.cpu().numpy())
    else:
        x_test_tensor = x_test_tensor.to(cfg.device)
        with torch.no_grad():
            outputs = model(x_test_tensor)
            if outputs.shape[1] == 3: # BG
                prob = torch.sigmoid(outputs[:, 0, :, :])
                shape = torch.exp(outputs[:, 1, :, :])
                scale = torch.exp(outputs[:, 2, :, :])
                preds_tensor = prob * (shape * scale)
            elif outputs.shape[1] == 2: # Gaussian
                preds_tensor = outputs[:, 0, :, :]
            else:
                preds_tensor = outputs.squeeze(1)
            preds_np = preds_tensor.cpu().numpy()
    
    if cfg.model_type == "glm":
        # Ensure GLM data is shaped correctly (T, H, W)
        preds_np = preds_np.reshape(-1, preds_np.shape[-2], preds_np.shape[-1])
        # Interpolate to target resolution if shapes mismatch
        if preds_np.shape[1] != len(lat_out) or preds_np.shape[2] != len(lon_out):
            vprint(f"Interpolating GLM output from {preds_np.shape[1:]} to {(len(lat_out), len(lon_out))}...")
            preds_tensor = torch.tensor(preds_np).unsqueeze(1) # (N, 1, H, W)
            preds_tensor = torch.nn.functional.interpolate(preds_tensor, size=(len(lat_out), len(lon_out)), mode='bilinear')
            preds_np = preds_tensor.squeeze(1).numpy()

    # 6. Save NetCDF with Step-2 naming convention
    # Use the root 'prediction' folder for saving NetCDF output as requested
    pred_dir = os.path.join("prediction", cfg.experiment)
    os.makedirs(pred_dir, exist_ok=True)
    
    target_name = "mswep" if cfg.variable == "precip" else "mswt"
    var_name = "precipitation" if cfg.variable == "precip" else "air_temperature"
    
    if scenario:
        if scenario["src"] == "era5":
            name = f"{cfg.model_type}_{cfg.variable}_era5.nc"
        else:
            name = f"{scenario['name']}_predictions.nc"
    else:
        name = f"pred_{cfg.model_type}_{cfg.variable}.nc"

    ds_pred = xr.Dataset(
        {var_name: (["time", "lat", "lon"], preds_np)},
        coords={"time": time_test, "lat": lat_out, "lon": lon_out}
    )
    units = "mm/day" if cfg.variable == "precip" else "degree_Celsius"
    ds_pred[var_name].attrs["units"] = units
    
    save_path = os.path.join(pred_dir, name)
    ds_pred.to_netcdf(save_path)
    vprint(f"Results saved to: {save_path}")


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(root_dir, "general", "config.yaml")
    
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    if not os.path.exists(cfg_path):
        vprint(f"Error: Config file {cfg_path} not found.")
        sys.exit(1)

    cfg = load_config(train_mode=False, path=cfg_path)
    set_verbose(cfg.verbose)
    
    vprint(f"=== Starting Prediction for {cfg.variable} (Model: {cfg.model_type}) ===")
    
    if cfg.scenarios:
        vprint(f"Running predictions for {len(cfg.scenarios)} scenarios...")
        for scenario in cfg.scenarios:
            try:
                # Fresh config for each scenario
                cfg_scenario = load_config(train_mode=False, path=cfg_path)
                perform_single_prediction(cfg_scenario, scenario=scenario)
                vprint(f"--> [SUCCESS] Scenario '{scenario['name']}' completed.")
            except Exception as e:
                vprint(f"--> [ERROR] Scenario '{scenario['name']}' failed: {e}")
                import traceback
                vprint(traceback.format_exc())
    else:
        perform_single_prediction(cfg)

    vprint(f"=== All Selected Predictions Completed ===")


if __name__ == "__main__":
    main()
