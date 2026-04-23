from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


SEASONS = {
    "Annual": list(range(1, 13)),
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

TEMP_CANDIDATES = ("air_temperature", "t2m", "tas", "temperature", "temp")


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize coordinate names to (time, lat, lon).
    """
    rename_map = {}

    if "latitude" in ds.coords or "latitude" in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords or "longitude" in ds.dims:
        rename_map["longitude"] = "lon"
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        if "time" not in ds.coords and "time" not in ds.dims:
            rename_map["valid_time"] = "time"
    if "time_counter" in ds.coords or "time_counter" in ds.dims:
        if "time" not in ds.coords and "time" not in ds.dims:
            rename_map["time_counter"] = "time"

    if rename_map:
        ds = ds.rename(rename_map)

    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")
    if "time" in ds.coords:
        ds = ds.sortby("time")

    return ds


def infer_temperature_var(ds: xr.Dataset, preferred: Optional[str] = None) -> str:
    """
    Infer the temperature variable name.
    """
    if preferred is not None:
        if preferred not in ds.data_vars:
            raise KeyError(
                f"Variable '{preferred}' not found. Available: {list(ds.data_vars)}"
            )
        return preferred

    for name in TEMP_CANDIDATES:
        if name in ds.data_vars:
            return name

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise KeyError(
        "Unable to infer temperature variable automatically. "
        f"Available variables: {list(ds.data_vars)}"
    )


def open_temperature_dataarray(path: str | Path, var_name: Optional[str] = None) -> xr.DataArray:
    """
    Open a temperature dataset and return one standardized DataArray.
    """
    ds = xr.open_dataset(path)
    ds = standardize_coords(ds)

    chosen_var = infer_temperature_var(ds, preferred=var_name)
    da = ds[chosen_var].squeeze(drop=True)

    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(set(da.dims)):
        raise ValueError(
            f"Selected variable '{chosen_var}' must contain dims {required_dims}, "
            f"but found {da.dims}"
        )

    return da.transpose("time", "lat", "lon")


def subset_test_period(
    da: xr.DataArray,
    start_date: Optional[str],
    end_date: Optional[str],
) -> xr.DataArray:
    """
    Restrict to test period.
    """
    if start_date is None and end_date is None:
        return da
    return da.sel(time=slice(start_date, end_date))


def align_prediction_and_observation(
    pred: xr.DataArray,
    obs: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Align on common time/lat/lon coordinates.
    """
    pred_aligned, obs_aligned = xr.align(pred, obs, join="inner")

    if pred_aligned.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time coordinates between prediction and observation.")
    if pred_aligned.sizes.get("lat", 0) == 0 or pred_aligned.sizes.get("lon", 0) == 0:
        raise ValueError("No overlapping spatial coordinates between prediction and observation.")

    return pred_aligned, obs_aligned


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    return unit.strip().lower().replace("°", "").replace("_", "").replace(" ", "")


def convert_temperature_to_celsius(
    da: xr.DataArray,
    forced_unit: Optional[str] = None,
) -> tuple[xr.DataArray, str]:
    """
    Convert temperature DataArray to Celsius when possible.

    If unit is missing and not forced, no conversion is applied.
    """
    unit = forced_unit if forced_unit is not None else da.attrs.get("units")
    unit_norm = _normalize_unit(unit)

    kelvin_units = {"k", "kelvin"}
    celsius_units = {"c", "degc", "celsius", "degreecelsius", "degreescelsius"}

    if unit_norm is None:
        return da, "Unit unknown -> no automatic conversion applied."

    if unit_norm in kelvin_units:
        da = da - 273.15
        da.attrs["units"] = "C"
        return da, "Converted from Kelvin to Celsius."

    if unit_norm in celsius_units:
        da.attrs["units"] = "C"
        return da, "Already in Celsius."

    return da, f"Unrecognized unit '{unit}' -> kept unchanged."


def get_months(time_array) -> np.ndarray:
    return np.array(pd.to_datetime(time_array).month)


def get_years(time_array) -> np.ndarray:
    return np.array(pd.to_datetime(time_array).year)


def get_season_years(time_array, season_name: str) -> np.ndarray:
    """
    For DJF, December is assigned to the following year.
    Example:
      Dec 2005 + Jan 2006 + Feb 2006 -> DJF 2006
    """
    dates = pd.to_datetime(time_array)
    years = np.array(dates.year)
    months = np.array(dates.month)

    if season_name == "DJF":
        years = years + (months == 12).astype(int)

    return years

def spatial_summary(da: xr.DataArray) -> dict:
    vals = da.values
    valid = np.isfinite(vals)

    if valid.sum() == 0:
        return {
            "n_valid": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    arr = vals[valid]
    return {
        "n_valid": int(valid.sum()),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def ensure_metric_dirs(exp_path: str | Path, metric_name: str) -> tuple[Path, Path]:
    """
    Create:
    results/<experiment>/metrics/<metric_name>/data
    results/<experiment>/metrics/<metric_name>/plots
    """
    exp_path = Path(exp_path)

    metric_root = exp_path / "metrics" / metric_name
    data_dir = metric_root / "data"
    plot_dir = metric_root / "plots"

    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, plot_dir

def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def save_summary_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)