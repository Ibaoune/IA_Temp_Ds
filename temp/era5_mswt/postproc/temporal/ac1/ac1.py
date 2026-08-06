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
    apply_spatial_context_to_inputs,
    convert_temperature_to_celsius,
    ensure_spatial_metric_dirs,
    get_spatial_context,
    open_temperature_dataarray,
    restrict_to_evaluation_pixels,
    restore_points_to_grid,
    save_json,
    save_summary_csv,
    spatial_summary,
    subset_test_period,
    validate_spatial_config,
)


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_METRIC_CONFIG = THIS_FILE.with_name("config.yaml")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute AC1 (lag-1 autocorrelation) using a metric-specific YAML config."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to ac1 metric config.yaml",
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
    if metric_name != "ac1":
        raise ValueError(f"metric.name must be 'ac1', got {metric_name!r}")

    min_valid = int(metric_cfg["metric"].get("min_valid", 10))
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

    validate_spatial_config(metric_cfg)
        
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
# Core computation
# =========================================================
def compute_lag1_autocorrelation(
    da: xr.DataArray,
    min_valid: int = 10,
) -> xr.DataArray:
    """
    Compute lag-1 autocorrelation (AC1) per grid point.
    """
    if "points" in da.dims:
        data = da.transpose("time", "points").values
        out_coords = {"points": da["points"]}
        out_dims = ("points",)
    else:
        data = da.transpose("time", "lat", "lon").values
        out_coords = {"lat": da["lat"], "lon": da["lon"]}
        out_dims = ("lat", "lon")

    if data.ndim not in {2, 3}:
        raise ValueError(
            f"Expected (time, points) or (time, lat, lon), got shape={data.shape}"
        )

    x0 = data[:-1, ...]
    x1 = data[1:, ...]

    valid = np.isfinite(x0) & np.isfinite(x1)
    n_valid = valid.sum(axis=0)

    x0v = np.where(valid, x0, np.nan)
    x1v = np.where(valid, x1, np.nan)

    mean0 = np.nanmean(x0v, axis=0)
    mean1 = np.nanmean(x1v, axis=0)

    cov = np.nanmean((x0v - mean0) * (x1v - mean1), axis=0)
    std0 = np.nanstd(x0v, axis=0)
    std1 = np.nanstd(x1v, axis=0)

    ac1 = cov / (std0 * std1)

    valid_mask = (n_valid >= min_valid) & (std0 > 0) & (std1 > 0)
    ac1 = np.where(valid_mask, ac1, np.nan)

    return xr.DataArray(
        ac1,
        coords=out_coords,
        dims=out_dims,
        name="ac1",
    )


def compute_ac1_fields(
    pred: xr.DataArray,
    obs: xr.DataArray,
    min_valid: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    ac1_pred = compute_lag1_autocorrelation(pred, min_valid=min_valid)
    ac1_obs = compute_lag1_autocorrelation(obs, min_valid=min_valid)

    bac1 = ac1_pred - ac1_obs

    ac1_pred.name = "ac1_pred"
    ac1_obs.name = "ac1_obs"
    bac1.name = "bac1"

    return ac1_pred, ac1_obs, bac1


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
    min_valid = int(metric_cfg["metric"].get("min_valid", 10))

    spatial_ctx = get_spatial_context(
        metric_cfg=metric_cfg,
        cfg=cfg,
        project_root=PROJECT_ROOT,
    )

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

    data_dir, plot_dir = ensure_spatial_metric_dirs(
        exp_path=exp_path,
        metric_name=metric_name,
        eval_domain=spatial_ctx.eval_domain,
    )

    print("=== Temperature AC1 postprocessing ===")
    print(f"Spatial domain   : {spatial_ctx.eval_domain}")
    print(f"Save mask        : {spatial_ctx.save_mask}")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Experiment root : {exp_path}")
    print(f"Prediction file : {pred_path}")
    print(f"Observation file: {obs_path}")
    print(f"Time window     : {start_date} -> {end_date}")
    print(f"Min valid pairs : {min_valid}")
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

    pred, obs, spatial_mask = apply_spatial_context_to_inputs(
        pred=pred,
        obs=obs,
        spatial_ctx=spatial_ctx,
    )

    print(f"Spatial evaluation domain: {spatial_ctx.eval_domain}")
    print(f"Valid spatial pixels      : {int(spatial_mask.sum().values)}")

    pred_eval, obs_eval = restrict_to_evaluation_pixels(
        pred=pred,
        obs=obs,
        spatial_mask=spatial_mask,
        min_valid=min_valid,
    )
    print(f"Computation pixels        : {pred_eval.sizes['points']}")

    ac1_pred, ac1_obs, bac1 = compute_ac1_fields(
        pred=pred_eval,
        obs=obs_eval,
        min_valid=min_valid,
    )
    ac1_pred = restore_points_to_grid(ac1_pred, spatial_mask)
    ac1_obs = restore_points_to_grid(ac1_obs, spatial_mask)
    bac1 = restore_points_to_grid(bac1, spatial_mask)

    ds_out = xr.Dataset(
        {
            "ac1_pred": ac1_pred,
            "ac1_obs": ac1_obs,
            "bac1": bac1,
        }
    )
    ds_out.attrs["metric"] = "ac1"
    ds_out.attrs["variable"] = "temperature"
    ds_out.attrs["prediction_file"] = str(pred_path)
    ds_out.attrs["observation_file"] = str(obs_path)
    ds_out.attrs["time_start"] = str(start_date)
    ds_out.attrs["time_end"] = str(end_date)
    ds_out.attrs["min_valid_pairs"] = min_valid
    ds_out.attrs["spatial_eval_domain"] = spatial_ctx.eval_domain
    ds_out.attrs["mask_applied_before_compute"] = "true"

    ds_out["ac1_pred"].attrs["long_name"] = "Lag-1 autocorrelation - prediction"
    ds_out["ac1_pred"].attrs["units"] = "dimensionless"

    ds_out["ac1_obs"].attrs["long_name"] = "Lag-1 autocorrelation - observation"
    ds_out["ac1_obs"].attrs["units"] = "dimensionless"

    ds_out["bac1"].attrs["long_name"] = "Bias of lag-1 autocorrelation"
    ds_out["bac1"].attrs["units"] = "dimensionless"
    ds_out["bac1"].attrs["description"] = "bac1 = ac1_pred - ac1_obs"

    for var_name in ["ac1_pred", "ac1_obs", "bac1"]:
        ds_out[var_name].attrs["spatial_eval_domain"] = spatial_ctx.eval_domain
        ds_out[var_name].attrs["mask_applied_before_compute"] = "true"

    if spatial_ctx.save_mask:
        ds_out["spatial_mask"] = spatial_mask.astype("int8")

    out_nc = data_dir / "ac1_annual_mean_period.nc"

    if save_netcdf:
        ds_out.to_netcdf(out_nc)

    bac1_stats = spatial_summary(ds_out["bac1"])
    summary_row = {
        "metric": "bac1",
        "min_valid_pairs": min_valid,
        "spatial_eval_domain": spatial_ctx.eval_domain,
        **bac1_stats,
        "file": str(out_nc) if save_netcdf else "",
    }

    if save_summary_json:
        save_json(
            summary_row,
            data_dir / "summary_ac1_annual_mean_period.json",
        )

    if save_summary_csv_flag:
        save_summary_csv(
            [summary_row],
            data_dir / "ac1_summary.csv",
        )
        print(f"Saved summary CSV: {data_dir / 'ac1_summary.csv'}")

    if save_netcdf:
        print(f"Saved: {out_nc}")
    print(
        f"  bAC1 mean={bac1_stats['mean']:.4f}, "
        f"min={bac1_stats['min']:.4f}, "
        f"max={bac1_stats['max']:.4f}"
    )

    print("\nAC1 products saved successfully.")


if __name__ == "__main__":
    main()
