from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


# =========================================================
# Path setup
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]   
DEFAULT_METRIC_CONFIG = THIS_FILE.with_name("config.yaml")

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import (
    SEASONS,
    align_prediction_and_observation,
    convert_temperature_to_celsius,
    ensure_metric_dirs,
    get_months,
    get_season_years,
    open_temperature_dataarray,
    save_json,
    save_summary_csv,
    spatial_summary,
    subset_test_period,
)


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute temperature bias maps using a metric-specific YAML config."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to bias metric config.yaml",
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
    if data is None:
        data = {}
    return data


def resolve_from_project_root(path_value: str | None) -> Path | None:
    """
    Resolve a path written in YAML.
    Relative paths are interpreted from the project root.
    """
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
    if metric_name != "bias":
        raise ValueError(
            f"metric.name must be 'bias', got {metric_name!r}"
        )

    strategy = metric_cfg["metric"].get("strategy", "daily_first")
    if strategy not in {"daily_first", "mean_first"}:
        raise ValueError(
            f"metric.strategy must be 'daily_first' or 'mean_first', got {strategy!r}"
        )

    seasons = metric_cfg["metric"].get("seasons", list(SEASONS.keys()))
    invalid = [s for s in seasons if s not in SEASONS]
    if invalid:
        raise ValueError(
            f"Unknown seasons in metric.seasons: {invalid}. Valid keys: {list(SEASONS.keys())}"
        )

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
# Paths / output
# =========================================================
def build_prediction_path(cfg) -> Path:
    """
    Default prediction file path from the main project config.
    """
    exp_path = Path(build_experiment_path(cfg))
    return exp_path / "output_data" / f"{cfg.model_type}_predictions_era5_to_{cfg.target}.nc"


# =========================================================
# Core bias computation
# =========================================================
def compute_bias_flagged(
    model: xr.DataArray,
    obs: xr.DataArray,
    strategy: str = "mean_first",
    return_by_year: bool = False,
    min_valid: int = 10,
    group_labels: np.ndarray | None = None,
    group_dim_name: str = "year",
) -> xr.DataArray:
    """
    Vectorized bias computation.

    Parameters
    ----------
    model, obs : xr.DataArray
        Dims: (time, lat, lon)

    strategy : str
        - "daily_first":
            compute (model - obs) every day, then average inside each group
        - "mean_first":
            compute mean(model) and mean(obs) inside each group, then subtract

    return_by_year : bool
        If True:
            output dims -> (group_dim_name, lat, lon)
        If False:
            average over groups and return -> (lat, lon)

    min_valid : int
        Minimum number of valid paired values inside each group.

    group_labels : array-like
        Labels used to group time steps, e.g. year or season-year.

    group_dim_name : str
        Output name of the grouping dimension.
    """
    if strategy not in {"mean_first", "daily_first"}:
        raise ValueError("strategy must be 'mean_first' or 'daily_first'")

    if group_labels is None:
        group_labels = np.array(model["time"].dt.year)

    model = model.assign_coords(group=("time", group_labels))
    obs = obs.assign_coords(group=("time", group_labels))

    valid = xr.where(np.isfinite(model) & np.isfinite(obs), 1, 0)
    valid_count = valid.groupby("group").sum("time")

    if strategy == "daily_first":
        diff = (model - obs).where(valid == 1)
        grouped_bias = diff.groupby("group").mean("time", skipna=True)
    else:
        grouped_model_mean = model.groupby("group").mean("time", skipna=True)
        grouped_obs_mean = obs.groupby("group").mean("time", skipna=True)
        grouped_bias = grouped_model_mean - grouped_obs_mean

    grouped_bias = grouped_bias.where(valid_count >= min_valid)
    grouped_bias = grouped_bias.rename({"group": group_dim_name})
    grouped_bias.name = "bias"

    if return_by_year:
        return grouped_bias

    return grouped_bias.mean(dim=group_dim_name, skipna=True)


def compute_one_tag(
    pred: xr.DataArray,
    obs: xr.DataArray,
    season_name: str,
    months: list[int],
    strategy: str,
    return_by_year: bool,
    min_valid: int,
) -> xr.DataArray:
    """
    Compute one bias product for one tag:
    Annual / DJF / MAM / JJA / SON
    """
    month_mask = get_months(pred["time"].values)
    idx = [m in months for m in month_mask]

    pred_sub = pred.isel(time=idx)
    obs_sub = obs.isel(time=idx)

    if pred_sub.sizes["time"] == 0:
        raise ValueError(f"No time steps found for season/tag '{season_name}'.")

    season_years = get_season_years(pred_sub["time"].values, season_name=season_name)

    bias_da = compute_bias_flagged(
        model=pred_sub,
        obs=obs_sub,
        strategy=strategy,
        return_by_year=return_by_year,
        min_valid=min_valid,
        group_labels=season_years,
        group_dim_name="year",
    )

    bias_da.attrs["long_name"] = f"Temperature bias - {season_name}"
    bias_da.attrs["description"] = (
        f"Bias computed with strategy='{strategy}' for season '{season_name}'"
    )
    bias_da.attrs["units"] = "C"
    bias_da.attrs["season"] = season_name
    bias_da.attrs["strategy"] = strategy

    return bias_da


# =========================================================
# Config-driven resolvers
# =========================================================
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

    cfg = load_config(train_mode=False, path=str(main_cfg_path))

    if str(cfg.variable).lower() != "temp":
        raise ValueError(
            f"This compute script is only for temperature, but cfg.variable={cfg.variable!r}"
        )

    exp_path = Path(build_experiment_path(cfg))
    metric_name = metric_cfg["metric"]["name"]
    strategy = metric_cfg["metric"].get("strategy", "daily_first")
    return_by_year = bool(metric_cfg["metric"].get("return_by_year", False))
    min_valid = int(metric_cfg["metric"].get("min_valid", 10))
    selected_seasons = get_selected_seasons(metric_cfg)

    pred_var = metric_cfg["data"].get("prediction_var", "air_temperature")
    obs_var = metric_cfg["data"].get("reference_var", None)
    pred_units = metric_cfg["data"].get("prediction_units", None)
    obs_units = metric_cfg["data"].get("reference_units", None)

    save_netcdf = bool(metric_cfg["output"].get("save_netcdf", True))
    save_summary_json = bool(metric_cfg["output"].get("save_summary_json", True))
    save_summary_csv_flag = bool(metric_cfg["output"].get("save_summary_csv", True))

    pred_path, obs_path = get_prediction_and_reference_paths(metric_cfg, cfg)
    start_date, end_date = get_time_window(metric_cfg, cfg)

    data_dir, plots_dir = ensure_metric_dirs(exp_path, metric_name)

    print("=== Temperature bias postprocessing ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Experiment root : {exp_path}")
    print(f"Prediction file : {pred_path}")
    print(f"Observation file: {obs_path}")
    print(f"Time window     : {start_date} -> {end_date}")
    print(f"Strategy        : {strategy}")
    print(f"Return by year  : {return_by_year}")
    print(f"Min valid       : {min_valid}")
    print(f"Seasons         : {list(selected_seasons.keys())}")
    print(f"Data output dir : {data_dir}")
    print(f"Plot output dir : {plots_dir}")

    # -------------------------
    # Load prediction / observation
    # -------------------------
    pred = open_temperature_dataarray(pred_path, var_name=pred_var)
    obs = open_temperature_dataarray(obs_path, var_name=obs_var)

    # -------------------------
    # Restrict to chosen period
    # -------------------------
    pred = subset_test_period(pred, start_date, end_date)
    obs = subset_test_period(obs, start_date, end_date)

    # -------------------------
    # Convert to Celsius if needed
    # -------------------------
    pred, pred_unit_note = convert_temperature_to_celsius(
        pred, forced_unit=pred_units
    )
    obs, obs_unit_note = convert_temperature_to_celsius(
        obs, forced_unit=obs_units
    )

    print(f"Prediction units : {pred_unit_note}")
    print(f"Observation units: {obs_unit_note}")

    # -------------------------
    # Align time/lat/lon
    # -------------------------
    pred, obs = align_prediction_and_observation(pred, obs)
    print(f"Aligned shape: pred={pred.shape}, obs={obs.shape}")

    summary_rows = []

    for season_name, months in selected_seasons.items():
        print(f"\n--- Computing bias for {season_name} ---")

        bias_da = compute_one_tag(
            pred=pred,
            obs=obs,
            season_name=season_name,
            months=months,
            strategy=strategy,
            return_by_year=return_by_year,
            min_valid=min_valid,
        )

        ds_out = xr.Dataset({"bias": bias_da})
        ds_out.attrs["metric"] = "bias"
        ds_out.attrs["variable"] = "temperature"
        ds_out.attrs["strategy"] = strategy
        ds_out.attrs["season"] = season_name
        ds_out.attrs["prediction_file"] = str(pred_path)
        ds_out.attrs["observation_file"] = str(obs_path)
        ds_out.attrs["time_start"] = str(start_date)
        ds_out.attrs["time_end"] = str(end_date)

        mode_suffix = "by_year" if return_by_year else "mean_period"
        nc_name = f"bias_{season_name.lower()}_{strategy}_{mode_suffix}.nc"
        json_name = f"summary_{season_name.lower()}_{strategy}_{mode_suffix}.json"

        if save_netcdf:
            out_nc = data_dir / nc_name
            ds_out.to_netcdf(out_nc)
        else:
            out_nc = None

        if return_by_year:
            bias_for_summary = bias_da.mean(dim="year", skipna=True)
        else:
            bias_for_summary = bias_da

        stats = spatial_summary(bias_for_summary)
        stats_row = {
            "season": season_name,
            "strategy": strategy,
            "mode": mode_suffix,
            "min_valid": min_valid,
            "mean": stats.get("mean"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "std": stats.get("std"),
            "file": str(out_nc) if out_nc is not None else "",
        }
        summary_rows.append(stats_row)

        if save_summary_json:
            save_json(stats_row, data_dir / json_name)

        if out_nc is not None:
            print(f"Saved NetCDF: {out_nc}")
        print(
            f"  mean={stats_row['mean']:.4f}, "
            f"min={stats_row['min']:.4f}, "
            f"max={stats_row['max']:.4f}, "
            f"std={stats_row['std']:.4f}"
        )

    if save_summary_csv_flag:
        csv_name = f"bias_summary_{strategy}.csv"
        save_summary_csv(summary_rows, data_dir / csv_name)
        print(f"\nSaved summary CSV: {data_dir / csv_name}")

    print("\nAll bias products saved successfully.")


if __name__ == "__main__":
    main()