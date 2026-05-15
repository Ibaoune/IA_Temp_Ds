"""
==========================================================
 Script: data_loading.py
 Author: M. El Aabaribaoune
 Description:
     Unified data loader for downscaling experiments.

     - Loads predictors (ERA5 or LMDZ)
     - Loads target variables (Precipitation or Temperature)
     - Supports MSWEP, LMDZ35 for precip; ERA5, LMDZ for temperature
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

from src.core import utils as use
from src.core.utils import vprint

# -------------------------------------
#              Helpers
# -------------------------------------

def _get_temperature_target_var(ds, target):
    """
    Choose the target variable name from cfg.target.
    No target_var is needed in YAML.
    """
    target = target.lower()

    if target == "mswt":
        candidates = ["air_temperature", "t2m", "temp", "tas"]
    elif target == "lmdz35":
        candidates = ["temp", "tas", "air_temperature", "t2m"]
    else:
        candidates = ["air_temperature", "t2m", "temp", "tas"]

    for name in candidates:
        if name in ds.data_vars:
            return name

    raise KeyError(
        f"No known temperature variable found for target='{target}'. "
        f"Available variables: {list(ds.data_vars)}"
    )

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

def _pressure_to_hpa(level_value):
    """
    Convert pressure level to hPa if needed.
    If already in hPa (500, 700, 850, 1000), keep as is.
    If in Pa (50000, 70000, ...), convert to hPa.
    """
    p = float(level_value)
    return p / 100.0 if p > 2000 else p


def _relative_to_specific_humidity(rh, temp_k, pressure_level):
    """
    Convert relative humidity to specific humidity.

    Parameters
    ----------
    rh : xarray.DataArray
        Relative humidity, either in fraction [0,1] or in % [0,100].
    temp_k : xarray.DataArray
        Temperature in Kelvin.
    pressure_level : float
        Pressure level (usually 500, 700, 850, 1000 hPa).

    Returns
    -------
    q : xarray.DataArray
        Specific humidity in kg/kg.
    """
    # Kelvin -> Celsius
    temp_c = temp_k - 273.15

    # Detect RH convention
    # if values are > 1.5, assume RH in %
    rh_frac = xr.where(rh > 1.5, rh / 100.0, rh)
    rh_frac = rh_frac.clip(min=0.0, max=1.0)

    # Pressure in hPa
    p_hpa = _pressure_to_hpa(pressure_level)

    # Saturation vapor pressure (Tetens), in hPa
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

    # Actual vapor pressure, in hPa
    e = rh_frac * es

    # Specific humidity, kg/kg
    q = 0.622 * e / (p_hpa - 0.378 * e)

    # Physical constraint: specific humidity cannot be negative
    q = q.clip(min=0.0)

    q = q.astype("float32")
    q.attrs["units"] = "kg kg-1"
    q.attrs["long_name"] = "specific humidity"
    q.name = "q"
    return q


def _load_lmdz_q_from_rh(cfg, levels, base_dir, suffix, file_pattern=None):
    """
    Build LMDZ specific humidity q from relative humidity (rhum) + temperature (temp).
    Returns a list of processed level arrays ready for concat.
    """
    data_arrays_q = []

    rh_name = cfg.lmdz_var_map.get("q", "rhum")
    t_name = cfg.lmdz_var_map.get("t", "temp")

    if file_pattern is None:
        file_pattern = getattr(cfg, "predictor_pattern", cfg.lmdz_predictor_pattern)

    rh_file = file_pattern.format(
        folder=base_dir.rstrip("/"),
        lmdz_var=rh_name,
        suffix=suffix
    )

    t_file = file_pattern.format(
        folder=base_dir.rstrip("/"),
        lmdz_var=t_name,
        suffix=suffix
    )

    vprint(f"Loading RH file for q conversion: {rh_file}")
    vprint(f"Loading T  file for q conversion: {t_file}")

    if not os.path.exists(rh_file):
        raise FileNotFoundError(f"RH file not found: {rh_file}")
    if not os.path.exists(t_file):
        raise FileNotFoundError(f"Temperature file not found: {t_file}")

    ds_rh_full = xr.open_dataset(rh_file)
    ds_t_full = xr.open_dataset(t_file)

    try:
        ds_rh = use.mask_dataset(
            ds_rh_full,
            slice(cfg.lon_min, cfg.lon_max),
            slice(cfg.lat_min, cfg.lat_max),
        )
        ds_t = use.mask_dataset(
            ds_t_full,
            slice(cfg.lon_min, cfg.lon_max),
            slice(cfg.lat_min, cfg.lat_max),
        )

        rh_var = next((v for v in ds_rh.data_vars if v.lower() == rh_name.lower()), list(ds_rh.data_vars)[0])
        t_var  = next((v for v in ds_t.data_vars if v.lower() == t_name.lower()), list(ds_t.data_vars)[0])

        rh_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds_rh[rh_var].dims), "level")
        rh_time_dim = next((d for d in ["time_counter", "time"] if d in ds_rh[rh_var].dims), "time")

        t_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds_t[t_var].dims), "level")
        t_time_dim = next((d for d in ["time_counter", "time"] if d in ds_t[t_var].dims), "time")

        for lev in levels:
            vprint(f"  → Building q at level {lev} from rhum + temp")

            rh_arr = ds_rh[rh_var].sel({rh_lev_dim: lev}, method="nearest").squeeze()
            t_arr  = ds_t[t_var].sel({t_lev_dim: lev}, method="nearest").squeeze()

            q_arr = _relative_to_specific_humidity(rh_arr, t_arr, lev)

            # process like any other predictor level
            p_arr = _process_level_array(cfg, q_arr, "q", lev, rh_time_dim, rh_lev_dim)
            if p_arr is not None:
                data_arrays_q.append(p_arr)

    finally:
        ds_rh_full.close()
        ds_t_full.close()

    return data_arrays_q

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
    if curr_lev_dim is not None:
        if curr_lev_dim in arr.coords:
            arr = arr.drop_vars(curr_lev_dim, errors="ignore")
        if curr_lev_dim in arr.dims:
            arr = arr.squeeze(drop=True)

    arr = arr.transpose("time", "lat", "lon")
    
    # Subset dates
    time_slice = slice(
        min(cfg.start_date_train, cfg.start_date_test),
        max(cfg.end_date_train, cfg.end_date_test)
    )
    arr_sub = arr.sel({"time": time_slice})
    
    # Interpolate
    from src.data.interpolation import interpolate_to_target_resolution
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
    Unified loader for ERA5→MSWEX and LMDZ→LMDZ35
    """

    vprint(f"=== Loading datasets (src={cfg.src}, target={cfg.target}) ===")

    # 1) Load target variable
    if cfg.variable == "precip":
        pr_file = cfg.target_path
        if cfg.target == "lmdz35":
            target_var = "precip"
            time_dim = "time_counter"
        elif cfg.target == "mswep":
            target_var = "precipitation"
            time_dim = "time"
        else:
            raise ValueError(f"Unsupported precip target: {cfg.target}")
    elif cfg.variable == "temp":
        pr_file = cfg.target_path
        ds_temp_check = xr.open_dataset(pr_file)

        target_var = _get_temperature_target_var(ds_temp_check, cfg.target)

        time_dim = "time" if "time" in ds_temp_check.dims else "time_counter"
        ds_temp_check.close()
    else:
        raise ValueError(f"Unsupported variable: {cfg.variable}")

    vprint(f"Loading {cfg.variable} file: {pr_file}")
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

        if target_var not in ds_pr:
            raise KeyError(f"Target variable '{target_var}' (type: {cfg.variable}) not found")

        y_train_x = ds_pr[target_var].sel(
            {time_dim: slice(cfg.start_date_train, cfg.end_date_train)}
        )
        y_test_x = ds_pr[target_var].sel(
            {time_dim: slice(cfg.start_date_test, cfg.end_date_test)}
        )

        vprint(f"{cfg.variable.upper()} shapes → train={y_train_x.shape}, test={y_test_x.shape}")

        if cfg.src == "era5":
                vprint("Checking target time resolution...")
                if not _is_daily_time_index(y_train_x):
                    y_train_x = y_train_x.resample(time="1D").mean()
                if not _is_daily_time_index(y_test_x):
                    y_test_x = y_test_x.resample(time="1D").mean()

        _print_basic_stats(y_train_x, f"{cfg.variable} (train)", vprint)
        _print_basic_stats(y_test_x, f"{cfg.variable} (test)", vprint)

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
    lmdz_var_map = getattr(cfg, "lmdz_var_map", {})

    # Default mapping for raw LMDZ files
    # z/q/t/u/v are the generic names used in the config,
    # while geop/rhum/temp/vitu/vitv are the real LMDZ filenames.
    if not lmdz_var_map:
        lmdz_var_map = {
            "z": "geop",
            "q": "rhum",
            "t": "temp",
            "u": "vitu",
            "v": "vitv",
        }

    for var in variables:
        var_key = var.lower()

        # --------------------------------------------------
        # Choose predictor pattern
        # --------------------------------------------------
        file_pattern = getattr(cfg, "predictor_pattern", "")

        if not file_pattern:
            raise ValueError(
                f"No predictor pattern found for src='{cfg.src}'. "
                "Check config.py and YAML paths."
            )

        # --------------------------------------------------
        # Detect raw LMDZ format
        # Raw LMDZ files use {lmdz_var}, for example:
        # geop-hist.nc, rhum-hist.nc, temp-hist.nc, ...
        # --------------------------------------------------
        is_raw_lmdz = (
            cfg.src in ["lmdz", "lmdz250", "lmdz35"]
            and "{lmdz_var}" in file_pattern
        )

        if is_raw_lmdz:
            lmdz_var = lmdz_var_map.get(var_key, var_key)

            folder_value = getattr(
                cfg,
                "folder",
                getattr(cfg, "bc_reference_folder", "")
            )

            base_dir = folder_value.rstrip("/")
            suffix = "hist"

            # Keep this for historical/future LMDZ cases
            search_text = f"{base_dir} {file_pattern}".lower()
            for s in ["ssp245", "ssp585"]:
                if s in search_text:
                    suffix = s
                    break

            # q is not directly stored as q in raw LMDZ.
            # It must be reconstructed from rhum + temp.
            if var_key == "q":
                q_arrays = _load_lmdz_q_from_rh(
                    cfg,
                    levels,
                    base_dir,
                    suffix,
                    file_pattern=file_pattern
                )
                data_arrays.extend(q_arrays)
                continue

        else:
            # Harmonized format:
            # z_1979-2020_levels.nc, q_1979-2020_levels.nc, ...
            lmdz_var = var_key
            base_dir = ""
            suffix = "hist"

        is_level_specific = "{level}" in file_pattern or "{lev}" in file_pattern

        if not is_level_specific:
            # Merged variable files:
            # Example raw LMDZ:
            #   geop-hist.nc, rhum-hist.nc, temp-hist.nc
            # Example harmonized:
            #   z_1979-2020_levels.nc, q_1979-2020_levels.nc

            filename = file_pattern.format(
                folder=base_dir,
                lmdz_var=lmdz_var,
                var=var_key,
                suffix=suffix
            )

            vprint(f"Loading merged variable file: {filename}...")

            if not os.path.exists(filename):
                vprint(f"  → FILE NOT FOUND: {filename}")
                continue

            try:
                ds_full = xr.open_dataset(filename)

                curr_lev_dim = next(
                    (d for d in ["presnivs", "plev", "level", "lev"] if d in ds_full.dims),
                    None
                )

                curr_time_dim = next(
                    (d for d in ["time_counter", "time", "valid_time"] if d in ds_full.dims),
                    "time"
                )

                ds = use.mask_dataset(
                    ds_full,
                    slice(cfg.lon_min, cfg.lon_max),
                    slice(cfg.lat_min, cfg.lat_max)
                )

                # Find variable inside NetCDF.
                # For raw LMDZ, we expect geop/rhum/temp/vitu/vitv.
                # For harmonized files, we expect z/q/t/u/v.
                # If not found, we take the first data variable.
                actual_var = next(
                    (v for v in ds.data_vars if v.lower() == lmdz_var.lower()),
                    list(ds.data_vars)[0]
                )

                if curr_lev_dim is not None:
                    for lev in levels:
                        vprint(f"  → Extracting level {lev} from {var}")

                        arr = ds[actual_var].sel(
                            {curr_lev_dim: lev},
                            method="nearest"
                        ).squeeze()

                        # Convert temperature predictor to Celsius if stored in Kelvin
                        if var_key == "t":
                            arr_units = arr.attrs.get("units", "")
                            if arr_units in ["K", "kelvin", "Kelvin"]:
                                arr = arr - 273.15
                                arr.attrs["units"] = "degree_Celsius"

                        p_arr = _process_level_array(
                            cfg,
                            arr,
                            var,
                            lev,
                            curr_time_dim,
                            curr_lev_dim
                        )

                        if p_arr is not None:
                            data_arrays.append(p_arr)

                else:
                    # If no vertical level exists, treat variable as a single channel
                    vprint(f"  → Processing {var} without vertical level")

                    arr = ds[actual_var].squeeze()

                    p_arr = _process_level_array(
                        cfg,
                        arr,
                        var,
                        "surface",
                        curr_time_dim,
                        None
                    )

                    if p_arr is not None:
                        data_arrays.append(p_arr)

                ds_full.close()

            except Exception as e:
                vprint(f"  ERROR: {e}")

        else:
            # Level-specific files:
            # Example:
            #   z_500.nc, z_700.nc, ...
            # or:
            #   geop_500-hist.nc, geop_700-hist.nc, ...

            for lev in levels:
                filename = file_pattern.format(
                    folder=base_dir,
                    lmdz_var=lmdz_var,
                    var=var_key,
                    suffix=suffix,
                    level=lev,
                    lev=lev
                )

                vprint(f"Loading level file: {filename}...")

                if not os.path.exists(filename):
                    vprint(f"  → FILE NOT FOUND: {filename}")
                    continue

                try:
                    ds = xr.open_dataset(filename).squeeze()

                    curr_time_dim = next(
                        (d for d in ["time_counter", "time", "valid_time"] if d in ds.dims),
                        "time"
                    )

                    curr_lev_dim = next(
                        (d for d in ["presnivs", "plev", "level", "lev"] if d in ds.dims),
                        None
                    )

                    ds = use.mask_dataset(
                        ds,
                        slice(cfg.lon_min, cfg.lon_max),
                        slice(cfg.lat_min, cfg.lat_max)
                    )

                    actual_var = next(
                        (v for v in ds.data_vars if v.lower() == lmdz_var.lower()),
                        list(ds.data_vars)[0]
                    )

                    arr = ds[actual_var].squeeze()

                    p_arr = _process_level_array(
                        cfg,
                        arr,
                        var,
                        lev,
                        curr_time_dim,
                        curr_lev_dim
                    )

                    if p_arr is not None:
                        data_arrays.append(p_arr)

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

     # Final shape verification
    vprint(f"Predictor Shape (X): {X.shape}")
    vprint(f"Target Train Shape (y_train): {y_train.shape if y_train is not None else 'None'}")
    vprint(f"Target Test Shape (y_test): {y_test.shape if y_test is not None else 'None'}")

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