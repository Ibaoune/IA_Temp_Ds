from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.stats import t as student_t

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import (
    SEASONS,
    align_prediction_and_observation,
    convert_temperature_to_celsius,
    ensure_metric_dirs,
    get_months,
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
        description="Compute temperature correlations using a metric-specific YAML config."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to correlation metric config.yaml",
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
    if metric_name != "correlation":
        raise ValueError(f"metric.name must be 'correlation', got {metric_name!r}")

    seasons = metric_cfg["metric"].get("seasons", list(SEASONS.keys()))
    invalid = [s for s in seasons if s not in SEASONS]
    if invalid:
        raise ValueError(
            f"Unknown seasons in metric.seasons: {invalid}. Valid keys: {list(SEASONS.keys())}"
        )

    min_valid = int(metric_cfg["metric"].get("min_valid", 10))
    if min_valid < 1:
        raise ValueError("metric.min_valid must be >= 1")

    window = int(metric_cfg["metric"].get("window", 31))
    if window < 1:
        raise ValueError("metric.window must be >= 1")
    if window % 2 == 0:
        raise ValueError("metric.window must be odd (e.g. 31)")

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


def get_selected_seasons(metric_cfg: dict):
    selected = metric_cfg["metric"].get("seasons", list(SEASONS.keys()))
    return {season_name: SEASONS[season_name] for season_name in selected}


def get_prediction_and_reference_paths(metric_cfg: dict, cfg):
    pred_override = resolve_from_project_root(metric_cfg["data"].get("prediction_path"))
    ref_override = resolve_from_project_root(metric_cfg["data"].get("reference_path"))

    pred_path = pred_override if pred_override is not None else build_prediction_path(cfg)
    obs_path = ref_override if ref_override is not None else Path(cfg.target_path)

    return pred_path, obs_path


# =========================================================
# Correlation helpers
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
        raise ValueError("window must be odd for centered smoothing.")

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


def pearson_corr_pvalue_sig_maps(
    x: xr.DataArray,
    y: xr.DataArray,
    min_valid: int = 10,
    alpha: float = 0.05,
):
    valid = xr.where(np.isfinite(x) & np.isfinite(y), 1, 0)
    n_valid = valid.sum("time")

    x_valid = x.where(valid == 1)
    y_valid = y.where(valid == 1)

    x_mean = x_valid.mean("time", skipna=True)
    y_mean = y_valid.mean("time", skipna=True)

    cov = ((x_valid - x_mean) * (y_valid - y_mean)).mean("time", skipna=True)
    x_std = x_valid.std("time", skipna=True)
    y_std = y_valid.std("time", skipna=True)

    corr = cov / (x_std * y_std)
    corr = corr.where((n_valid >= min_valid) & (x_std > 0) & (y_std > 0))

    dof = n_valid - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = corr * np.sqrt(dof / (1.0 - corr**2))

    pval = xr.apply_ufunc(
        lambda t, df: 2.0 * student_t.sf(np.abs(t), df),
        t_stat,
        dof,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    pval = pval.where((n_valid >= min_valid) & (dof > 0) & np.isfinite(corr))

    sig = (pval < alpha).astype(np.int8)
    sig = sig.where(np.isfinite(corr))

    return corr, pval, sig, n_valid

def compute_corr_d(
    pred: xr.DataArray,
    obs: xr.DataArray,
    min_valid: int = 10,
    window: int = 31,
    alpha: float = 0.05,
):
    pred_anom = deseasonalize_daily(pred, window=window)
    obs_anom = deseasonalize_daily(obs, window=window)

    pred_anom, obs_anom = xr.align(pred_anom, obs_anom, join="inner")

    corr_d, pval_d, sig_d, n_valid_d = pearson_corr_pvalue_sig_maps(
        pred_anom, obs_anom, min_valid=min_valid, alpha=alpha
    )

    corr_d.name = "corr_d"
    corr_d.attrs["long_name"] = "Daily deseasonalized Pearson correlation"

    pval_d.name = "corr_d_pval"
    pval_d.attrs["long_name"] = "P-value of CORR D"

    sig_d.name = "corr_d_sig"
    sig_d.attrs["long_name"] = "Significance mask of CORR D (1: significant, 0: not significant)"
    sig_d.attrs["alpha"] = alpha

    n_valid_d.name = "corr_d_n_valid"
    n_valid_d.attrs["long_name"] = "Number of valid time steps used for CORR D"

    return corr_d, pval_d, sig_d, n_valid_d


def compute_corr_m(
    pred: xr.DataArray,
    obs: xr.DataArray,
    min_valid: int = 10,
    alpha: float = 0.05,
):
    pred_month = pred.resample(time="MS").mean(skipna=True)
    obs_month = obs.resample(time="MS").mean(skipna=True)

    pred_month, obs_month = xr.align(pred_month, obs_month, join="inner")

    corr_m, pval_m, sig_m, n_valid_m = pearson_corr_pvalue_sig_maps(
        pred_month, obs_month, min_valid=min_valid, alpha=alpha
    )

    corr_m.name = "corr_m"
    corr_m.attrs["long_name"] = "Monthly Pearson correlation"

    pval_m.name = "corr_m_pval"
    pval_m.attrs["long_name"] = "P-value of CORR M"

    sig_m.name = "corr_m_sig"
    sig_m.attrs["long_name"] = "Significance mask of CORR M (1: significant, 0: not significant)"
    sig_m.attrs["alpha"] = alpha

    n_valid_m.name = "corr_m_n_valid"
    n_valid_m.attrs["long_name"] = "Number of valid time steps used for CORR M"

    return corr_m, pval_m, sig_m, n_valid_m


def compute_one_tag(
    pred: xr.DataArray,
    obs: xr.DataArray,
    season_name: str,
    months: list[int],
    min_valid: int,
    window: int,
    alpha: float,
):
    month_mask = get_months(pred["time"].values)
    idx = [m in months for m in month_mask]

    pred_sub = pred.isel(time=idx)
    obs_sub = obs.isel(time=idx)

    if pred_sub.sizes["time"] == 0:
        raise ValueError(f"No time steps found for season/tag '{season_name}'.")

    corr_d, corr_d_pval, corr_d_sig, corr_d_n_valid = compute_corr_d(
        pred_sub,
        obs_sub,
        min_valid=min_valid,
        window=window,
        alpha=alpha,
    )

    corr_m, corr_m_pval, corr_m_sig, corr_m_n_valid = compute_corr_m(
        pred_sub,
        obs_sub,
        min_valid=min_valid,
        alpha=alpha,
    )

    for da in [corr_d, corr_d_pval, corr_d_sig, corr_d_n_valid]:
        da.attrs["season"] = season_name

    for da in [corr_m, corr_m_pval, corr_m_sig, corr_m_n_valid]:
        da.attrs["season"] = season_name

    return (
        corr_d, corr_d_pval, corr_d_sig, corr_d_n_valid,
        corr_m, corr_m_pval, corr_m_sig, corr_m_n_valid,
    )


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
    selected_seasons = get_selected_seasons(metric_cfg)

    if not selected_seasons:
        raise ValueError("metric.seasons cannot be empty.")

    min_valid = int(metric_cfg["metric"].get("min_valid", 10))
    window = int(metric_cfg["metric"].get("window", 31))
    alpha = float(metric_cfg["metric"].get("alpha", 0.05))

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

    corr_d_data_dir, corr_d_plots_dir = ensure_metric_dirs(exp_path, "corr_d")
    corr_m_data_dir, corr_m_plots_dir = ensure_metric_dirs(exp_path, "corr_m")

    print("=== Temperature correlation postprocessing ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Experiment root : {exp_path}")
    print(f"Prediction file : {pred_path}")
    print(f"Observation file: {obs_path}")
    print(f"Time window     : {start_date} -> {end_date}")
    print(f"Window (CORR D) : {window}")
    print(f"Min valid       : {min_valid}")
    print(f"Seasons         : {list(selected_seasons.keys())}")
    print(f"corr_d data dir : {corr_d_data_dir}")
    print(f"corr_m data dir : {corr_m_data_dir}")
    print(f"corr_d plot dir : {corr_d_plots_dir}")
    print(f"corr_m plot dir : {corr_m_plots_dir}")

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

    summary_rows_corr_d = []
    summary_rows_corr_m = []

    for season_name, months in selected_seasons.items():
        print(f"\n--- Computing correlations for {season_name} ---")

        (
            corr_d, corr_d_pval, corr_d_sig, corr_d_n,
            corr_m, corr_m_pval, corr_m_sig, corr_m_n,
        ) = compute_one_tag(
            pred=pred,
            obs=obs,
            season_name=season_name,
            months=months,
            min_valid=min_valid,
            window=window,
            alpha=alpha,
        )

        ds_corr_d = xr.Dataset({
            "corr_d": corr_d,
            "corr_d_pval": corr_d_pval,
            "corr_d_sig": corr_d_sig,
            "corr_d_n_valid": corr_d_n,
        })

        ds_corr_m = xr.Dataset({
            "corr_m": corr_m,
            "corr_m_pval": corr_m_pval,
            "corr_m_sig": corr_m_sig,
            "corr_m_n_valid": corr_m_n,
        })

        for ds_out, metric_name in [(ds_corr_d, "corr_d"), (ds_corr_m, "corr_m")]:
            ds_out.attrs["metric"] = metric_name
            ds_out.attrs["metric_family"] = "correlation"
            ds_out.attrs["variable"] = "temperature"
            ds_out.attrs["season"] = season_name
            ds_out.attrs["prediction_file"] = str(pred_path)
            ds_out.attrs["observation_file"] = str(obs_path)
            ds_out.attrs["time_start"] = str(start_date)
            ds_out.attrs["time_end"] = str(end_date)
            ds_out.attrs["min_valid"] = min_valid
            if metric_name == "corr_d":
                ds_out.attrs["window_days"] = window

        out_nc_corr_d = corr_d_data_dir / f"corr_d_{season_name.lower()}_mean_period.nc"
        out_nc_corr_m = corr_m_data_dir / f"corr_m_{season_name.lower()}_mean_period.nc"

        if save_netcdf:
            ds_corr_d.to_netcdf(out_nc_corr_d)
            ds_corr_m.to_netcdf(out_nc_corr_m)

        corr_d_stats = spatial_summary(corr_d)
        corr_m_stats = spatial_summary(corr_m)

        if save_summary_json:
            save_json(
                {
                    "season": season_name,
                    "metric": "corr_d",
                    "min_valid": min_valid,
                    "window_days": window,
                    **corr_d_stats,
                    "file": str(out_nc_corr_d) if save_netcdf else "",
                },
                corr_d_data_dir / f"summary_corr_d_{season_name.lower()}_mean_period.json",
            )

            save_json(
                {
                    "season": season_name,
                    "metric": "corr_m",
                    "min_valid": min_valid,
                    **corr_m_stats,
                    "file": str(out_nc_corr_m) if save_netcdf else "",
                },
                corr_m_data_dir / f"summary_corr_m_{season_name.lower()}_mean_period.json",
            )

        summary_rows_corr_d.append(
            {
                "season": season_name,
                "metric": "corr_d",
                "min_valid": min_valid,
                "window_days": window,
                **corr_d_stats,
                "file": str(out_nc_corr_d) if save_netcdf else "",
            }
        )

        summary_rows_corr_m.append(
            {
                "season": season_name,
                "metric": "corr_m",
                "min_valid": min_valid,
                **corr_m_stats,
                "file": str(out_nc_corr_m) if save_netcdf else "",
            }
        )

        if save_netcdf:
            print(f"Saved: {out_nc_corr_d}")
            print(f"Saved: {out_nc_corr_m}")

        print(
            f"  CORR D mean={corr_d_stats['mean']:.4f}, "
            f"min={corr_d_stats['min']:.4f}, "
            f"max={corr_d_stats['max']:.4f}"
        )
        print(
            f"  CORR M mean={corr_m_stats['mean']:.4f}, "
            f"min={corr_m_stats['min']:.4f}, "
            f"max={corr_m_stats['max']:.4f}"
        )

    if save_summary_csv_flag:
        save_summary_csv(
            summary_rows_corr_d,
            corr_d_data_dir / "corr_d_summary.csv",
        )
        save_summary_csv(
            summary_rows_corr_m,
            corr_m_data_dir / "corr_m_summary.csv",
        )
        print(f"\nSaved summary CSV: {corr_d_data_dir / 'corr_d_summary.csv'}")
        print(f"Saved summary CSV: {corr_m_data_dir / 'corr_m_summary.csv'}")

    print("\nAll correlation products saved successfully.")


if __name__ == "__main__":
    main()