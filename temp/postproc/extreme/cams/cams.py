from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
        description="Compute CAMS (Cold Annual Max Spell) using a metric-specific YAML config."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to cams metric config.yaml",
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
    if metric_name != "cams":
        raise ValueError(f"metric.name must be 'cams', got {metric_name!r}")

    threshold_quantile = float(metric_cfg["metric"].get("threshold_quantile", 0.10))
    if not (0.0 < threshold_quantile < 1.0):
        raise ValueError("metric.threshold_quantile must be between 0 and 1.")

    min_spell_length = int(metric_cfg["metric"].get("min_spell_length", 2))
    min_valid_days = int(metric_cfg["metric"].get("min_valid_days", 10))
    min_valid_years = int(metric_cfg["metric"].get("min_valid_years", 1))

    if min_spell_length < 1:
        raise ValueError("metric.min_spell_length must be >= 1")
    if min_valid_days < 1:
        raise ValueError("metric.min_valid_days must be >= 1")
    if min_valid_years < 1:
        raise ValueError("metric.min_valid_years must be >= 1")

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
# Core helpers
# =========================================================
def _drop_quantile_coord(da: xr.DataArray) -> xr.DataArray:
    if "quantile" in da.coords:
        da = da.drop_vars("quantile")
    if "quantile" in da.dims:
        da = da.squeeze("quantile", drop=True)
    return da


def longest_spell_below_threshold(
    below: np.ndarray,
    min_spell_length: int = 2,
) -> np.ndarray:
    if below.ndim != 3:
        raise ValueError(f"'below' must have shape (time, lat, lon), got {below.shape}")

    _, n_lat, n_lon = below.shape
    run = np.zeros((n_lat, n_lon), dtype=np.int16)
    max_run = np.zeros((n_lat, n_lon), dtype=np.int16)

    for t in range(below.shape[0]):
        run = (run + 1) * below[t]
        max_run = np.maximum(max_run, run)

    max_run = np.where(max_run >= min_spell_length, max_run, 0)
    return max_run.astype(np.float32)


def compute_annual_max_cold_spell_lengths(
    da: xr.DataArray,
    threshold: xr.DataArray,
    min_spell_length: int = 2,
    min_valid_days: int = 10,
) -> xr.DataArray:
    years = np.unique(da["time"].dt.year.values)
    annual_max_list = []

    threshold_2d = threshold.values

    for year in years:
        sub = da.sel(time=str(int(year)))
        data = sub.values

        valid = np.isfinite(data)
        valid_count = valid.sum(axis=0)

        below = valid & (data < threshold_2d[None, :, :])

        annual_max = longest_spell_below_threshold(
            below=below,
            min_spell_length=min_spell_length,
        )

        annual_max = np.where(valid_count >= min_valid_days, annual_max, np.nan)
        annual_max_list.append(annual_max)

    annual_max_arr = np.stack(annual_max_list, axis=0)

    return xr.DataArray(
        annual_max_arr,
        coords={"year": years, "lat": da["lat"], "lon": da["lon"]},
        dims=("year", "lat", "lon"),
        name="annual_max_cold_spell",
    )


def compute_cams_fields(
    pred: xr.DataArray,
    obs: xr.DataArray,
    threshold_quantile: float = 0.10,
    min_spell_length: int = 2,
    min_valid_days: int = 10,
    min_valid_years: int = 1,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    if not (0.0 < threshold_quantile < 1.0):
        raise ValueError("threshold_quantile must be between 0 and 1.")

    pred_p10 = pred.quantile(threshold_quantile, dim="time", skipna=True)
    obs_p10 = obs.quantile(threshold_quantile, dim="time", skipna=True)

    pred_p10 = _drop_quantile_coord(pred_p10)
    obs_p10 = _drop_quantile_coord(obs_p10)

    pred_yearly = compute_annual_max_cold_spell_lengths(
        da=pred,
        threshold=pred_p10,
        min_spell_length=min_spell_length,
        min_valid_days=min_valid_days,
    )
    obs_yearly = compute_annual_max_cold_spell_lengths(
        da=obs,
        threshold=obs_p10,
        min_spell_length=min_spell_length,
        min_valid_days=min_valid_days,
    )

    cams_pred = pred_yearly.median(dim="year", skipna=True)
    cams_obs = obs_yearly.median(dim="year", skipna=True)

    pred_valid_years = pred_yearly.count(dim="year")
    obs_valid_years = obs_yearly.count(dim="year")
    valid_year_mask = (pred_valid_years >= min_valid_years) & (obs_valid_years >= min_valid_years)

    cams_pred = cams_pred.where(valid_year_mask)
    cams_obs = cams_obs.where(valid_year_mask)

    bcams = cams_pred - cams_obs

    cams_pred.name = "cams_pred"
    cams_obs.name = "cams_obs"
    bcams.name = "bcams"

    return cams_pred, cams_obs, bcams


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

    threshold_quantile = float(metric_cfg["metric"].get("threshold_quantile", 0.10))
    min_spell_length = int(metric_cfg["metric"].get("min_spell_length", 2))
    min_valid_days = int(metric_cfg["metric"].get("min_valid_days", 10))
    min_valid_years = int(metric_cfg["metric"].get("min_valid_years", 1))

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

    print("=== Temperature CAMS postprocessing ===")
    print(f"Metric config     : {metric_cfg_path}")
    print(f"Main config       : {main_cfg_path}")
    print(f"Experiment root   : {exp_path}")
    print(f"Prediction file   : {pred_path}")
    print(f"Observation file  : {obs_path}")
    print(f"Time window       : {start_date} -> {end_date}")
    print(f"Threshold quantile: {threshold_quantile}")
    print(f"Min spell length  : {min_spell_length}")
    print(f"Min valid days    : {min_valid_days}")
    print(f"Min valid years   : {min_valid_years}")
    print(f"Data output dir   : {data_dir}")
    print(f"Plot output dir   : {plot_dir}")

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

    cams_pred, cams_obs, bcams = compute_cams_fields(
        pred=pred,
        obs=obs,
        threshold_quantile=threshold_quantile,
        min_spell_length=min_spell_length,
        min_valid_days=min_valid_days,
        min_valid_years=min_valid_years,
    )

    ds_out = xr.Dataset(
        {
            "cams_pred": cams_pred,
            "cams_obs": cams_obs,
            "bcams": bcams,
        }
    )
    ds_out.attrs["metric"] = "cams"
    ds_out.attrs["variable"] = "temperature"
    ds_out.attrs["prediction_file"] = str(pred_path)
    ds_out.attrs["observation_file"] = str(obs_path)
    ds_out.attrs["time_start"] = str(start_date)
    ds_out.attrs["time_end"] = str(end_date)
    ds_out.attrs["threshold_quantile"] = threshold_quantile
    ds_out.attrs["min_spell_length"] = min_spell_length
    ds_out.attrs["min_valid_days"] = min_valid_days
    ds_out.attrs["min_valid_years"] = min_valid_years

    ds_out["cams_pred"].attrs["long_name"] = "Cold Annual Max Spell - prediction"
    ds_out["cams_pred"].attrs["units"] = "day_count"

    ds_out["cams_obs"].attrs["long_name"] = "Cold Annual Max Spell - observation"
    ds_out["cams_obs"].attrs["units"] = "day_count"

    ds_out["bcams"].attrs["long_name"] = "Bias of Cold Annual Max Spell"
    ds_out["bcams"].attrs["units"] = "day_count"
    ds_out["bcams"].attrs["description"] = "bcams = cams_pred - cams_obs"

    out_nc = data_dir / "cams_annual_mean_period.nc"

    if save_netcdf:
        ds_out.to_netcdf(out_nc)

    bcams_stats = spatial_summary(ds_out["bcams"])

    if save_summary_json:
        save_json(
            {
                "metric": "cams",
                "threshold_quantile": threshold_quantile,
                "min_spell_length": min_spell_length,
                "min_valid_days": min_valid_days,
                "min_valid_years": min_valid_years,
                "file": str(out_nc) if save_netcdf else "",
                "bcams": bcams_stats,
            },
            data_dir / "summary_cams_annual_mean_period.json",
        )

    if save_summary_csv_flag:
        save_summary_csv(
            [
                {
                    "metric": "bcams",
                    "threshold_quantile": threshold_quantile,
                    "min_spell_length": min_spell_length,
                    "min_valid_days": min_valid_days,
                    "min_valid_years": min_valid_years,
                    **bcams_stats,
                    "file": str(out_nc) if save_netcdf else "",
                }
            ],
            data_dir / "cams_summary.csv",
        )
        print(f"Saved summary CSV: {data_dir / 'cams_summary.csv'}")

    if save_netcdf:
        print(f"Saved: {out_nc}")
    print(
        f"  bCAMS mean={bcams_stats['mean']:.4f}, "
        f"min={bcams_stats['min']:.4f}, "
        f"max={bcams_stats['max']:.4f}"
    )

    print("\nCAMS products saved successfully.")


if __name__ == "__main__":
    main()