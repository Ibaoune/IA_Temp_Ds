"""
==========================================================
 Script: data_loading.py
 Author: M. El Aabaribaoune
 Description:
     Unified data loader for downscaling experiments.

     - Loads predictors (ERA5 or LMDZ)
     - Loads precipitation targets (MSWEP or LMDZ35)
     - Applies spatial masking
     - Handles daily / sub-daily temporal resolution
     - Returns tensors and coordinate metadata

 Notes:
     - Model-agnostic (ViT, CNN, UNet, RF...)
     - GPU-ready but does NOT force GPU allocation
     - Logging via vprint only
==========================================================
"""

import os
import numpy as np
import xarray as xr
import torch

import utils as use
from utils import vprint

# -------------------------------------
#              Helpers
# -------------------------------------

def _is_daily_time_index(time_index):
    """
    Returns True if the time_index spacing is (approximately) 1 day.
    Uses median timestep for robustness to missing values.
    """
    if len(time_index) < 2:
        return True

    deltas = np.diff(time_index.values.astype("datetime64[ns]"))
    median_days = np.median(deltas).astype("timedelta64[ns]") / np.timedelta64(1, "D")
    return np.isclose(median_days, 1.0, rtol=1e-3, atol=1e-6)


def _print_basic_stats(arr, name, vprint_fn):
    """
    Print mean / min / max statistics for an xarray DataArray.
    """
    try:
        vals = arr.values
        if not np.issubdtype(vals.dtype, np.number):
            vals = vals.astype("float32")

        flat = vals.ravel()
        if flat.size == 0:
            vprint_fn(f"   [STATS] {name}: EMPTY array")
            return

        mean = np.nanmean(flat)
        vmin = np.nanmin(flat)
        vmax = np.nanmax(flat)
        units = arr.attrs.get("units", "")
        unit_str = f" {units}" if units else ""

        vprint_fn(
            f"   [STATS] {name}: mean={mean:.6g}{unit_str}, "
            f"min={vmin:.6g}, max={vmax:.6g}"
        )

    except Exception as e:
        vprint_fn(f"   [STATS] {name}: unable to compute stats ({e})")

# -------------------------------------
#              Main loader
# -------------------------------------

def load_datasets(cfg):
    """
    Unified loader for ERA5 → MSWEP and LMDZ → LMDZ35.

    Returns:
        X           : xarray.Dataset (full predictors)
        y_train     : torch.Tensor
        y_test      : torch.Tensor
        lon_in      : np.ndarray
        lat_in      : np.ndarray
        lon_out     : np.ndarray
        lat_out     : np.ndarray
        time_train  : np.ndarray
        time_test   : np.ndarray
    """

    vprint(f"=== Loading datasets (src={cfg.src}, target={cfg.target}) ===")

    # 1) Load target variable
    if cfg.variable == "precip":
        pr_file = cfg.target_path
        if cfg.target == "lmdz35":
            precip_var = "precip"
            time_dim = "time_counter"
        elif cfg.target == "mswep":
            precip_var = "precipitation"
            time_dim = "time"
        else:
            raise ValueError(f"Unsupported precip target: {cfg.target}")
    elif cfg.variable == "temp":
        pr_file = cfg.target_path
        ds_temp_check = xr.open_dataset(pr_file)
        if "air_temperature" in ds_temp_check.data_vars:
            precip_var = "air_temperature"
        elif "t2m" in ds_temp_check.data_vars:
            precip_var = "t2m"
        else:
            precip_var = "tas"
        
        time_dim = "time" if "time" in ds_temp_check.dims else "time_counter"
        ds_temp_check.close()
    else:
        raise ValueError(f"Unsupported variable: {cfg.variable}")

    vprint(f"Loading precipitation file: {pr_file}")
    if not os.path.exists(pr_file):
        if not cfg.train_mode:
            vprint(f"  WARNING: Target file {pr_file} NOT FOUND. Creating dummy target for prediction.")
            ds_pr = None
        else:
            raise FileNotFoundError(f"Target file {pr_file} not found and train_mode is True.")
    else:
        ds_pr = xr.open_dataset(pr_file)

    if ds_pr is not None:
        ds_pr = use.mask_dataset(
            ds_pr,
            slice(cfg.lon_min, cfg.lon_max),
            slice(cfg.lat_min, cfg.lat_max),
        )

        if precip_var not in ds_pr:
            raise KeyError(f"Precip variable '{precip_var}' not found")

        y_train_x = ds_pr[precip_var].sel(
            {time_dim: slice(cfg.start_date_train, cfg.end_date_train)}
        )
        y_test_x = ds_pr[precip_var].sel(
            {time_dim: slice(cfg.start_date_test, cfg.end_date_test)}
        )

        vprint(f"Precip shapes → train={y_train_x.shape}, test={y_test_x.shape}")

        if cfg.src == "era5":
                vprint("Checking target time resolution...")
                if not _is_daily_time_index(y_train_x):
                    y_train_x = y_train_x.resample(time="1D").mean()
                if not _is_daily_time_index(y_test_x):
                    y_test_x = y_test_x.resample(time="1D").mean()

        _print_basic_stats(y_train_x, "precip (train)", vprint)
        _print_basic_stats(y_test_x, "precip (test)", vprint)

        # Keep tensors on CPU (GPU later in training loop)
        y_train = torch.tensor(y_train_x.values.astype("float32"))
        y_test  = torch.tensor(y_test_x.values.astype("float32"))
        
        lon_out = ds_pr.lon.values
        lat_out = ds_pr.lat.values
        time_dim_found = next((d for d in [time_dim, "time", "time_counter"] if d in y_train_x.coords or d in y_train_x.dims), time_dim)
        time_train_out = y_train_x[time_dim_found].values
        time_test_out = y_test_x[time_dim_found].values
    else:
        # Dummy values to be filled after X is loaded
        y_train = None
        y_test = None
        lon_out = None
        lat_out = None
        time_train_out = None
        time_test_out = None

    # 2) Load predictor variables
    variables = cfg.variables
    levels = cfg.levels
    data_arrays = []

    # Map our generic config variables to LMDZ specific names from config.yaml
    lmdz_var_map = cfg.lmdz_var_map

    if cfg.src == "lmdz":
        base_dir = getattr(cfg, "folder", cfg.bc_reference_folder)
        if not base_dir.endswith("/"):
            base_dir += "/"
        
        # Decide the suffix (hist, ssp245, ssp585) based on the current scenario or folder
        suffix = "hist" 
        for s in ["ssp245", "ssp585"]:
            if s in base_dir.lower() or (hasattr(cfg, "scenario_name") and s in cfg.scenario_name.lower()):
                suffix = s
                break
        
        file_pattern = cfg.lmdz_predictor_pattern
    else:
        file_pattern = cfg.era5_predictor_pattern

    for var in variables:
        lmdz_var = lmdz_var_map.get(var.lower(), var.lower())
        
        if cfg.src == "lmdz":
            filename = file_pattern.format(folder=base_dir.rstrip("/"), lmdz_var=lmdz_var, suffix=suffix)
        else:
            filename = file_pattern.format(var=var.lower())

        vprint(f"Loading {filename}...")

        if not os.path.exists(filename) and cfg.src == "lmdz" and "present" in base_dir:
            # Fallback for LMDZ present folder structure
            alt_filename = os.path.join(base_dir, "all_Mor", f"{lmdz_var}-{suffix}.nc")
            if os.path.exists(alt_filename):
                filename = alt_filename
                vprint(f"  → Found in: {filename}")

        if not os.path.exists(filename):
            vprint(f"  → FILE NOT FOUND: {filename}")
            continue

        try:
            ds = xr.open_dataset(filename).squeeze()
            
            # Detect dimensions dynamically for this file
            curr_time_dim = next((d for d in ["time_counter", "time"] if d in ds.dims), "time")
            curr_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds.dims), "level")

            ds = use.mask_dataset(
                ds,
                slice(cfg.lon_min, cfg.lon_max),
                slice(cfg.lat_min, cfg.lat_max),
            )

            for lev in levels:
                vprint(f"  → Extracting level {lev} for {var} (via {curr_lev_dim})")
                try:
                    actual_var = next(
                        (v for v in ds.data_vars if v.lower() == lmdz_var.lower()),
                        next((v for v in ds.data_vars if curr_lev_dim in ds[v].dims and not v.endswith("_bnds")), list(ds.data_vars)[0])
                    )
                    arr = ds[actual_var].sel({curr_lev_dim: lev}, method="nearest").squeeze()
                except (ValueError, KeyError):
                    vprint(f"    WARNING: Level {lev} selection failed for {var}")
                    continue

                # ERA5 cleanup
                if cfg.src == "era5":
                    rename = {}
                    if "latitude" in arr.dims:
                        rename["latitude"] = "lat"
                    if "longitude" in arr.dims:
                        rename["longitude"] = "lon"
                    if rename:
                        arr = arr.rename(rename)

                    if "level" in arr.dims:
                        arr = arr.drop_vars("level", errors="ignore")
                    if "level" in arr.coords:
                        arr = arr.drop_vars("level", errors="ignore")

                    for c in ["level", "number", "expver", "surface", "valid_time"]:
                        if c in arr.coords:
                            arr = arr.drop_vars(c, errors="ignore")

                # Coordinate names
                lat_name = 'lat' if 'lat' in arr.coords or 'lat' in arr.dims else 'latitude'
                lon_name = 'lon' if 'lon' in arr.coords or 'lon' in arr.dims else 'longitude'
                
                rename_dict = {curr_time_dim: "time"}
                if lat_name != "lat": rename_dict[lat_name] = "lat"
                if lon_name != "lon": rename_dict[lon_name] = "lon"
                arr = arr.rename(rename_dict)

                # Explicitly drop the level coordinate to avoid MergeError during concat
                if curr_lev_dim in arr.coords:
                    arr = arr.drop_vars(curr_lev_dim)

                arr = arr.transpose("time", "lat", "lon")
                
                # Subset dates explicitly using the detected dim for this file
                time_slice = slice(
                    min(cfg.start_date_train, cfg.start_date_test),
                    max(cfg.end_date_train, cfg.end_date_test)
                )
                arr_sub = arr.sel({"time": time_slice})
                
                _print_basic_stats(arr_sub, f"{var}_{lev}", vprint)

                arr_sub = arr_sub.expand_dims({"level": [f"{var}_{lev}"]})
                data_arrays.append(arr_sub)

        except Exception as e:
            vprint(f"  ERROR: {e}")

    if not data_arrays:
        raise ValueError("No predictor files loaded")

    X = xr.concat(data_arrays, dim="level", coords="minimal")
    X = X.transpose("time", "level", "lat", "lon")

    # Interpolate predictors to target resolution
    from interpolation import interpolate_to_target_resolution
    vprint(f"Interpolating predictors to {cfg.resolution} degree resolution using {cfg.interpolation_type} interpolation...")
    X = interpolate_to_target_resolution(
        X, 
        resolution=cfg.resolution, 
        method=cfg.interpolation_type,
        bounds=(cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max) # Ensure consistent grid size
    )

    if cfg.src == "era5":
        vprint("Checking time resolution...")
        if not _is_daily_time_index(X.time):
            X = X.resample(time="1D").mean()

    x_train = X.sel(time=slice(cfg.start_date_train, cfg.end_date_train))
    x_test  = X.sel(time=slice(cfg.start_date_test, cfg.end_date_test))

    # Fill metadata from X if it was missing from target
    if lon_out is None:
        lon_out = X.lon.values
    if lat_out is None:
        lat_out = X.lat.values
    if time_test_out is None:
        time_test_out = x_test.time.values
    if time_train_out is None:
        time_train_out = x_train.time.values
    
    # Fill dummy tensors if missing
    if y_train is None:
        y_train = torch.zeros((len(time_train_out), len(lat_out), len(lon_out)))
    if y_test is None:
        y_test = torch.zeros((len(time_test_out), len(lat_out), len(lon_out)))

    vprint("=== Finished loading datasets ===")

    return (
        X,
        y_train,
        y_test,
        X.lon.values,
        X.lat.values,
        lon_out,
        lat_out,
        time_train_out,
        time_test_out,
    )

