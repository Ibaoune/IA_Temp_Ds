from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import (
    align_prediction_and_observation,
    convert_temperature_to_celsius,
    ensure_metric_dirs,
    open_temperature_dataarray,
    save_json,
    save_summary_csv,
    spatial_summary,
    subset_test_period,
)


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_METRIC_CONFIG = THIS_FILE.with_name("config.yaml")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RSTD (ratio of standard deviations of deseasonalized daily anomalies) using a metric-specific YAML config."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to rstd metric config.yaml",
    )
    return parser.parse_args()


# =========================================================
# YAML helpers
# =========================================================
def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def resolve_from_project_root(path_value: str | None) -> Path | None:
    if path_value in (None, "", "null"):
        return None

    p = Path(path_value)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


# =========================================================
# Validation helpers
# =========================================================
def validate_metric_config(metric_cfg: dict):
    if "project" not in metric_cfg:
        raise ValueError("Missing 'project' section in metric config.")
    if "metric" not in metric_cfg:
        raise ValueError("Missing 'metric' section in metric config.")
    if "data" not in metric_cfg:
        raise ValueError("Missing 'data' section in metric config.")
    if "time" not in metric_cfg:
        raise ValueError("Missing 'time' section in metric config.")
    if "output" not in metric_cfg:
        raise ValueError("Missing 'output' section in metric config.")

    metric_name = metric_cfg["metric"].get("name", None)
    if metric_name != "rstd":
        raise ValueError(f"metric.name must be 'rstd', got {metric_name!r}")

    window = int(metric_cfg["metric"].get("window", 31))
    min_valid = int(metric_cfg["metric"].get("min_valid", 10))

    if window < 1:
        raise ValueError("metric.window must be >= 1")
    if window % 2 == 0:
        raise ValueError("metric.window must be odd (e.g. 31)")
    if min_valid < 1:
        raise ValueError("metric.min_valid must be >= 1")

    use_main_period = metric_cfg["time"].get("use_test_period_from_main_config", True)
    if not use_main_period:
        start_date = metric_cfg["time"].get("start_date", None)
        end_date = metric_cfg["time"].get("end_date", None)
        if not start_date or not end_date:
            raise ValueError(
                "time.start_date and time.end_date must be provided when "
                "use_test_period_from_main_config = false"
            )


# =========================================================
# Paths / config-driven resolvers
# =========================================================
def build_prediction_path(cfg) -> Path:
    """
    Default prediction file path from the main project config.
    Compatible with ERA5->MSWT, LMDZ250->LMDZ35, and LMDZ35_2deg->LMDZ35.
    """
    exp_path = Path(build_experiment_path(cfg))
    out_dir = exp_path / "output_data"

    candidates = [
        out_dir / f"{cfg.model_type}_predictions_{cfg.experiment}.nc",
        out_dir / f"{cfg.model_type}_predictions_{cfg.src}_to_{cfg.target}.nc",
        out_dir / f"{cfg.model_type}_predictions_{cfg.target}.nc",
        out_dir / f"{cfg.model_type}_predictions_era5_to_{cfg.target}.nc",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No prediction file found. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def get_time_window(metric_cfg: dict, cfg):
    use_main_period = metric_cfg["time"].get("use_test_period_from_main_config", True)

    if use_main_period:
        return cfg.start_date_test, cfg.end_date_test

    return (
        metric_cfg["time"]["start_date"],
        metric_cfg["time"]["end_date"],
    )


def get_prediction_and_reference_paths(metric_cfg: dict, cfg):
    pred_override = resolve_from_project_root(metric_cfg["data"].get("prediction_path"))
    ref_override = resolve_from_project_root(metric_cfg["data"].get("reference_path"))

    pred_path = pred_override if pred_override is not None else build_prediction_path(cfg)
    obs_path = ref_override if ref_override is not None else Path(cfg.target_path)

    return pred_path, obs_path


# =========================================================
# Deseasonalization helpers
# =========================================================
def drop_feb29(da: xr.DataArray) -> xr.DataArray:
    month = da["time"].dt.month
    day = da["time"].dt.day
    return da.sel(time=~((month == 2) & (day == 29)))


def build_climatological_day_index(time_values) -> np.ndarray:
    dates = pd.to_datetime(time_values)

    clim_day = []
    for dt in dates:
        ref = pd.Timestamp(year=2001, month=dt.month, day=dt.day)
        clim_day.append(ref.dayofyear)

    return np.asarray(clim_day, dtype=int)


def smooth_daily_climatology(clim: xr.DataArray, window: int = 31) -> xr.DataArray:
    if window < 1:
        raise ValueError("window must be >= 1")
    if window % 2 == 0:
        raise ValueError("window must be odd for centered smoothing (e.g. 31)")

    pad = window // 2

    left = clim.isel(clim_day=slice(-pad, None)).copy()
    left = left.assign_coords(clim_day=np.arange(1 - pad, 1))

    right = clim.isel(clim_day=slice(0, pad)).copy()
    right = right.assign_coords(clim_day=np.arange(366, 366 + pad))

    extended = xr.concat([left, clim, right], dim="clim_day")
    smoothed = extended.rolling(clim_day=window, center=True, min_periods=1).mean(skipna=True)

    return smoothed.sel(clim_day=slice(1, 365))


def deseasonalize_daily(da: xr.DataArray, window: int = 31) -> xr.DataArray:
    da = drop_feb29(da)

    clim_day = build_climatological_day_index(da["time"].values)
    da = da.assign_coords(clim_day=("time", clim_day))

    daily_clim = da.groupby("clim_day").mean("time", skipna=True)
    daily_clim = daily_clim.reindex(clim_day=np.arange(1, 366))

    daily_clim_smooth = smooth_daily_climatology(daily_clim, window=window)

    anomalies = da.groupby("clim_day") - daily_clim_smooth
    return anomalies


# =========================================================
# Core computation
# =========================================================
def compute_rstd_fields(
    pred: xr.DataArray,
    obs: xr.DataArray,
    window: int = 31,
    min_valid: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute:
        std_pred = standard deviation of deseasonalized prediction anomalies
        std_obs  = standard deviation of deseasonalized observation anomalies
        rstd     = std_pred / std_obs
    """
    pred_anom = deseasonalize_daily(pred, window=window)
    obs_anom = deseasonalize_daily(obs, window=window)

    pred_anom, obs_anom = xr.align(pred_anom, obs_anom, join="inner")

    valid = xr.where(np.isfinite(pred_anom) & np.isfinite(obs_anom), 1, 0)
    n_valid = valid.sum("time")

    pred_valid = pred_anom.where(valid == 1)
    obs_valid = obs_anom.where(valid == 1)

    std_pred = pred_valid.std("time", skipna=True)
    std_obs = obs_valid.std("time", skipna=True)

    rstd = std_pred / std_obs

    valid_mask = (n_valid >= min_valid) & (std_obs > 0) & np.isfinite(std_obs)
    std_pred = std_pred.where(valid_mask)
    std_obs = std_obs.where(valid_mask)
    rstd = rstd.where(valid_mask)

    std_pred.name = "std_pred"
    std_obs.name = "std_obs"
    rstd.name = "rstd"

    return std_pred, std_obs, rstd


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    metric_cfg_path = Path(args.metric_config).resolve()
    metric_cfg = load_yaml(metric_cfg_path)
    validate_metric_config(metric_cfg)

    main_cfg_path = resolve_from_project_root(
        metric_cfg["project"].get("main_config_path")
    )
    if main_cfg_path is None:
        raise ValueError("project.main_config_path must be provided in metric config.")
    if not main_cfg_path.exists():
        raise FileNotFoundError(f"Main config not found: {main_cfg_path}")

    cfg = load_config(train_mode=False, path=str(main_cfg_path))

    if str(cfg.variable).lower() != "temp":
        raise ValueError(
            f"This compute script is only for temperature, but cfg.variable={cfg.variable!r}"
        )

    exp_path = Path(build_experiment_path(cfg))
    metric_name = metric_cfg["metric"]["name"]
    window = int(metric_cfg["metric"].get("window", 31))
    min_valid = int(metric_cfg["metric"].get("min_valid", 10))

    pred_var = metric_cfg["data"].get("prediction_var", "air_temperature")
    obs_var = metric_cfg["data"].get("reference_var", None)
    pred_units = metric_cfg["data"].get("prediction_units", None)
    obs_units = metric_cfg["data"].get("reference_units", None)

    save_netcdf = bool(metric_cfg["output"].get("save_netcdf", True))
    save_summary_json = bool(metric_cfg["output"].get("save_summary_json", True))
    save_summary_csv_flag = bool(metric_cfg["output"].get("save_summary_csv", True))

    pred_path, obs_path = get_prediction_and_reference_paths(metric_cfg, cfg)
    start_date, end_date = get_time_window(metric_cfg, cfg)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    data_dir, plot_dir = ensure_metric_dirs(exp_path, metric_name)

    print("=== Temperature RSTD postprocessing ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Experiment root : {exp_path}")
    print(f"Prediction file : {pred_path}")
    print(f"Observation file: {obs_path}")
    print(f"Time window     : {start_date} -> {end_date}")
    print(f"Window          : {window}")
    print(f"Min valid       : {min_valid}")
    print(f"Data output dir : {data_dir}")
    print(f"Plot output dir : {plot_dir}")

    pred = open_temperature_dataarray(pred_path, var_name=pred_var)
    obs = open_temperature_dataarray(obs_path, var_name=obs_var)

    pred = subset_test_period(pred, start_date, end_date)
    obs = subset_test_period(obs, start_date, end_date)

    pred, pred_unit_note = convert_temperature_to_celsius(pred, forced_unit=pred_units)
    obs, obs_unit_note = convert_temperature_to_celsius(obs, forced_unit=obs_units)

    print(f"Prediction units : {pred_unit_note}")
    print(f"Observation units: {obs_unit_note}")

    pred, obs = align_prediction_and_observation(pred, obs)
    print(f"Aligned shape    : pred={pred.shape}, obs={obs.shape}")

    std_pred, std_obs, rstd = compute_rstd_fields(
        pred=pred,
        obs=obs,
        window=window,
        min_valid=min_valid,
    )

    ds_out = xr.Dataset(
        {
            "std_pred": std_pred,
            "std_obs": std_obs,
            "rstd": rstd,
        }
    )
    ds_out.attrs["metric"] = "rstd"
    ds_out.attrs["variable"] = "temperature"
    ds_out.attrs["prediction_file"] = str(pred_path)
    ds_out.attrs["observation_file"] = str(obs_path)
    ds_out.attrs["time_start"] = str(start_date)
    ds_out.attrs["time_end"] = str(end_date)
    ds_out.attrs["window_days"] = window
    ds_out.attrs["min_valid"] = min_valid

    ds_out["std_pred"].attrs["long_name"] = "Standard deviation of deseasonalized prediction anomalies"
    ds_out["std_pred"].attrs["units"] = "C"

    ds_out["std_obs"].attrs["long_name"] = "Standard deviation of deseasonalized observation anomalies"
    ds_out["std_obs"].attrs["units"] = "C"

    ds_out["rstd"].attrs["long_name"] = "Ratio of standard deviations"
    ds_out["rstd"].attrs["units"] = "dimensionless"
    ds_out["rstd"].attrs["description"] = "rstd = std_pred / std_obs"

    out_nc = data_dir / "rstd_annual_mean_period.nc"

    if save_netcdf:
        ds_out.to_netcdf(out_nc)

    rstd_stats = spatial_summary(ds_out["rstd"])

    if save_summary_json:
        save_json(
            {
                "metric": "rstd",
                "window_days": window,
                "min_valid": min_valid,
                "file": str(out_nc) if save_netcdf else "",
                "rstd": rstd_stats,
            },
            data_dir / "summary_rstd_annual_mean_period.json",
        )

    if save_summary_csv_flag:
        save_summary_csv(
            [
                {
                    "metric": "rstd",
                    "window_days": window,
                    "min_valid": min_valid,
                    **rstd_stats,
                    "file": str(out_nc) if save_netcdf else "",
                }
            ],
            data_dir / "rstd_summary.csv",
        )
        print(f"Saved summary CSV: {data_dir / 'rstd_summary.csv'}")

    if save_netcdf:
        print(f"Saved: {out_nc}")
    print(
        f"  RSTD mean={rstd_stats['mean']:.4f}, "
        f"min={rstd_stats['min']:.4f}, "
        f"max={rstd_stats['max']:.4f}"
    )

    print("\nRSTD products saved successfully.")


if __name__ == "__main__":
    main()