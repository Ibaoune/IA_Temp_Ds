from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr
import yaml

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
        description="Compute B-98 (bias of the 98th percentile) using a metric-specific YAML config."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to b98 metric config.yaml",
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
    if metric_name != "b98":
        raise ValueError(f"metric.name must be 'b98', got {metric_name!r}")

    q = float(metric_cfg["metric"].get("quantile", 0.98))
    if not (0.0 < q < 1.0):
        raise ValueError("metric.quantile must be between 0 and 1.")

    min_valid = int(metric_cfg["metric"].get("min_valid", 10))
    if min_valid < 1:
        raise ValueError("metric.min_valid must be >= 1")

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
# Core computation
# =========================================================
def compute_quantile_fields(
    pred: xr.DataArray,
    obs: xr.DataArray,
    q: float = 0.98,
    min_valid: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute:
        p98_pred = Q_q(pred)
        p98_obs  = Q_q(obs)
        b98      = Q_q(pred) - Q_q(obs)
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be between 0 and 1.")

    pred_valid_count = pred.count(dim="time")
    obs_valid_count = obs.count(dim="time")

    pred_q = pred.quantile(q, dim="time", skipna=True)
    obs_q = obs.quantile(q, dim="time", skipna=True)

    if "quantile" in pred_q.dims:
        pred_q = pred_q.squeeze("quantile", drop=True)
    if "quantile" in pred_q.coords:
        pred_q = pred_q.drop_vars("quantile")

    if "quantile" in obs_q.dims:
        obs_q = obs_q.squeeze("quantile", drop=True)
    if "quantile" in obs_q.coords:
        obs_q = obs_q.drop_vars("quantile")

    valid_mask = (pred_valid_count >= min_valid) & (obs_valid_count >= min_valid)

    pred_q = pred_q.where(valid_mask)
    obs_q = obs_q.where(valid_mask)

    bias_q = pred_q - obs_q
    pred_q.name = "p98_pred"
    obs_q.name = "p98_obs"
    bias_q.name = "b98"

    return pred_q, obs_q, bias_q


def compute_one_tag(
    pred: xr.DataArray,
    obs: xr.DataArray,
    season_name: str,
    months: list[int],
    q: float,
    min_valid: int,
) -> xr.Dataset:
    month_mask = get_months(pred["time"].values)
    idx = [m in months for m in month_mask]

    pred_sub = pred.isel(time=idx)
    obs_sub = obs.isel(time=idx)

    if pred_sub.sizes["time"] == 0:
        raise ValueError(f"No time steps found for season/tag '{season_name}'.")

    p98_pred, p98_obs, b98_da = compute_quantile_fields(
        pred=pred_sub,
        obs=obs_sub,
        q=q,
        min_valid=min_valid,
    )

    p98_pred.attrs["long_name"] = f"Predicted P98 temperature - {season_name}"
    p98_pred.attrs["units"] = "C"
    p98_pred.attrs["season"] = season_name
    p98_pred.attrs["quantile"] = q

    p98_obs.attrs["long_name"] = f"Observed P98 temperature - {season_name}"
    p98_obs.attrs["units"] = "C"
    p98_obs.attrs["season"] = season_name
    p98_obs.attrs["quantile"] = q

    b98_da.attrs["long_name"] = f"Bias of P98 temperature - {season_name}"
    b98_da.attrs["description"] = (
        f"B-98 computed as Q{q:.2f}(prediction) - Q{q:.2f}(observation) "
        f"for season '{season_name}'."
    )
    b98_da.attrs["units"] = "C"
    b98_da.attrs["season"] = season_name
    b98_da.attrs["quantile"] = q

    ds_out = xr.Dataset(
        {
            "p98_pred": p98_pred,
            "p98_obs": p98_obs,
            "b98": b98_da,
        }
    )
    ds_out.attrs["season"] = season_name
    ds_out.attrs["quantile"] = q

    return ds_out


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
    q = float(metric_cfg["metric"].get("quantile", 0.98))
    min_valid = int(metric_cfg["metric"].get("min_valid", 10))
    selected_seasons = get_selected_seasons(metric_cfg)

    if not selected_seasons:
        raise ValueError("metric.seasons cannot be empty.")

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

    print("=== Temperature B-98 postprocessing ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Experiment root : {exp_path}")
    print(f"Prediction file : {pred_path}")
    print(f"Observation file: {obs_path}")
    print(f"Time window     : {start_date} -> {end_date}")
    print(f"Quantile        : {q}")
    print(f"Min valid       : {min_valid}")
    print(f"Seasons         : {list(selected_seasons.keys())}")
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

    summary_rows = []

    for season_name, months in selected_seasons.items():
        print(f"\n--- Computing B-98 for {season_name} ---")

        ds_out = compute_one_tag(
            pred=pred,
            obs=obs,
            season_name=season_name,
            months=months,
            q=q,
            min_valid=min_valid,
        )

        ds_out.attrs["metric"] = "b98"
        ds_out.attrs["variable"] = "temperature"
        ds_out.attrs["prediction_file"] = str(pred_path)
        ds_out.attrs["observation_file"] = str(obs_path)
        ds_out.attrs["time_start"] = str(start_date)
        ds_out.attrs["time_end"] = str(end_date)

        out_nc = data_dir / f"b98_{season_name.lower()}_mean_period.nc"

        if save_netcdf:
            ds_out.to_netcdf(out_nc)

        stats = spatial_summary(ds_out["b98"])
        stats_row = {
            "season": season_name,
            "metric": "b98",
            "quantile": q,
            "min_valid": min_valid,
            **stats,
            "file": str(out_nc) if save_netcdf else "",
        }
        summary_rows.append(stats_row)

        if save_summary_json:
            save_json(
                stats_row,
                data_dir / f"summary_b98_{season_name.lower()}_mean_period.json",
            )

        if save_netcdf:
            print(f"Saved: {out_nc}")
        print(
            f"  mean={stats_row['mean']:.4f}, "
            f"min={stats_row['min']:.4f}, "
            f"max={stats_row['max']:.4f}"
        )

    if save_summary_csv_flag:
        save_summary_csv(
            summary_rows,
            data_dir / "b98_summary.csv",
        )
        print(f"\nSaved summary CSV: {data_dir / 'b98_summary.csv'}")

    print("\nAll B-98 products saved successfully.")


if __name__ == "__main__":
    main()