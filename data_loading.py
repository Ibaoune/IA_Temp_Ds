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
        # Avoid loading everything just for stats if it's a large array
        if arr.size > 1000000:
            # Take a small sample lazily if possible, or just compute mean/min/max lazily
            mean = float(arr.mean().values)
            vmin = float(arr.min().values)
            vmax = float(arr.max().values)
            vprint_fn(f"   [STATS] {name}: mean={mean:.6g}, min={vmin:.6g}, max={vmax:.6g}")
            return

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

def _process_level_array(cfg, arr, var, lev, curr_time_dim, curr_lev_dim):
    """
    Common processing for a single level array (renaming, transposition, interpolation).
    """
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
    
    # Subset dates
    time_slice = slice(
        min(cfg.start_date_train, cfg.start_date_test),
        max(cfg.end_date_train, cfg.end_date_test)
    )
    arr_sub = arr.sel({"time": time_slice})
    
    # Interpolate
    from interpolation import interpolate_to_target_resolution
    arr_sub = interpolate_to_target_resolution(
        arr_sub, 
        resolution=cfg.resolution, 
        method=cfg.interpolation_type,
        bounds=(cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max)
    )
    
    _print_basic_stats(arr_sub, f"{var}_{lev}", vprint)
    return arr_sub.expand_dims({"level": [f"{var}_{lev}"]})

# -------------------------------------
#              Main loader
# -------------------------------------

def load_datasets(cfg):
    """
    Unified loader for ERA5 → MSWEP and LMDZ → LMDZ35.
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

        lon_out = ds_pr.lon.values
        lat_out = ds_pr.lat.values
        time_dim_found = next((d for d in [time_dim, "time", "time_counter"] if d in y_train_x.coords or d in y_train_x.dims), time_dim)
        time_train_out = y_train_x[time_dim_found].values
        time_test_out = y_test_x[time_dim_found].values

        if cfg.train_mode:
            y_train = torch.tensor(y_train_x.values.astype("float32"))
        else:
            y_train = torch.zeros((len(time_train_out), len(lat_out), len(lon_out)))
            
        y_test  = torch.tensor(y_test_x.values.astype("float32"))
    else:
        y_train, y_test, lon_out, lat_out, time_train_out, time_test_out = [None]*6

    # 2) Load predictor variables
    variables = cfg.variables
    levels = cfg.levels
    data_arrays = []
    lmdz_var_map = cfg.lmdz_var_map

    for var in variables:
        lmdz_var = lmdz_var_map.get(var.lower(), var.lower())
        
        if cfg.src == "lmdz":
            base_dir = getattr(cfg, "folder", cfg.bc_reference_folder).rstrip("/") + "/"
            suffix = "hist" 
            for s in ["ssp245", "ssp585"]:
                if s in base_dir.lower(): suffix = s; break
            file_pattern = cfg.lmdz_predictor_pattern
        else:
            base_dir = None
            suffix = None
            file_pattern = cfg.era5_predictor_pattern

        is_level_specific = "{level}" in file_pattern or "{lev}" in file_pattern

        if not is_level_specific:
            # Merged variable files
            if cfg.src == "lmdz":
                filename = file_pattern.format(folder=base_dir.rstrip("/"), lmdz_var=lmdz_var, suffix=suffix)
            else:
                filename = file_pattern.format(var=var.lower())

            vprint(f"Loading merged variable file: {filename}...")
            if not os.path.exists(filename):
                vprint(f"  → FILE NOT FOUND: {filename}")
                continue

            try:
                ds_full = xr.open_dataset(filename)
                curr_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds_full.dims), "level")
                curr_time_dim = next((d for d in ["time_counter", "time"] if d in ds_full.dims), "time")

                for lev in levels:
                    vprint(f"  → Extracting level {lev} from {var}")
                    ds = use.mask_dataset(ds_full, slice(cfg.lon_min, cfg.lon_max), slice(cfg.lat_min, cfg.lat_max))
                    actual_var = next((v for v in ds.data_vars if v.lower() == lmdz_var.lower()), list(ds.data_vars)[0])
                    arr = ds[actual_var].sel({curr_lev_dim: lev}, method="nearest").squeeze()
                    
                    p_arr = _process_level_array(cfg, arr, var, lev, curr_time_dim, curr_lev_dim)
                    if p_arr is not None: data_arrays.append(p_arr)
                ds_full.close()
            except Exception as e:
                vprint(f"  ERROR: {e}")

        else:
            # Level-specific files
            for lev in levels:
                if cfg.src == "lmdz":
                    filename = file_pattern.format(folder=base_dir.rstrip("/"), lmdz_var=lmdz_var, suffix=suffix, level=lev, lev=lev)
                else:
                    filename = file_pattern.format(var=var.lower(), level=lev, lev=lev)

                vprint(f"Loading level file: {filename}...")
                if not os.path.exists(filename):
                    vprint(f"  → FILE NOT FOUND: {filename}")
                    continue

                try:
                    ds = xr.open_dataset(filename).squeeze()
                    curr_time_dim = next((d for d in ["time_counter", "time"] if d in ds.dims), "time")
                    curr_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds.dims), "level")
                    
                    ds = use.mask_dataset(ds, slice(cfg.lon_min, cfg.lon_max), slice(cfg.lat_min, cfg.lat_max))
                    actual_var = next((v for v in ds.data_vars if v.lower() == lmdz_var.lower()), list(ds.data_vars)[0])
                    arr = ds[actual_var].squeeze()

                    p_arr = _process_level_array(cfg, arr, var, lev, curr_time_dim, curr_lev_dim)
                    if p_arr is not None: data_arrays.append(p_arr)
                    ds.close()
                except Exception as e:
                    vprint(f"  ERROR: {e}")



    if not data_arrays:
        raise ValueError("No predictor files loaded")

    X = xr.concat(data_arrays, dim="level", coords="minimal")
    X = X.transpose("time", "level", "lat", "lon")

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

