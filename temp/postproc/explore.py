from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import yaml

from ..main.src.core.config import load_config
from ..main.src.core.utils import build_experiment_path
from .common import open_temperature_dataarray


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_CONFIG = THIS_FILE.with_name("explore_config.yaml")

DISPLAY_NAMES = {
    "z": "Géopotentiel",
    "t": "Température de l’air",
    "q": "Humidité spécifique",
    "u": "Vitesse du vent zonal",
    "v": "Vitesse du vent méridional",
    "target": "Température cible (MSWT)",
    "prediction": "Prédiction du modèle",
}


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore temperature predictors, predictand, and model output."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to exploration config.yaml",
    )
    return parser.parse_args()


# =========================================================
# YAML / config helpers
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


def build_prediction_path(cfg) -> Path:
    exp_path = Path(build_experiment_path(cfg))
    return exp_path / "output_data" / f"{cfg.model_type}_predictions_era5_to_{cfg.target}.nc"


def ensure_exploration_dirs(exp_path: Path) -> tuple[Path, Path]:
    data_dir = exp_path / "exploration" / "data"
    plot_dir = exp_path / "exploration" / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir


def get_time_window(explore_cfg: dict, cfg):
    use_main_period = explore_cfg["time"].get("use_test_period_from_main_config", True)
    if use_main_period:
        return cfg.start_date_test, cfg.end_date_test
    return (
        explore_cfg["time"]["start_date"],
        explore_cfg["time"]["end_date"],
    )


def get_predictor_pattern(explore_cfg: dict, cfg) -> str:
    override = explore_cfg["data"].get("predictor_pattern", None)
    if override not in (None, "", "null"):
        return str(resolve_from_project_root(override))
    pattern = getattr(cfg, "era5_predictor_pattern", None)
    if pattern is None:
        raise ValueError(
            "No predictor pattern found. Set data.predictor_pattern in exploration config "
            "or ensure the main config exposes era5_predictor_pattern."
        )
    return str(pattern)


def get_prediction_and_reference_paths(explore_cfg: dict, cfg) -> tuple[Path, Path]:
    pred_override = resolve_from_project_root(explore_cfg["data"].get("prediction_path"))
    ref_override = resolve_from_project_root(explore_cfg["data"].get("reference_path"))

    pred_path = pred_override if pred_override is not None else build_prediction_path(cfg)
    obs_path = ref_override if ref_override is not None else Path(cfg.target_path)

    return pred_path, obs_path


# =========================================================
# Metadata helpers
# =========================================================
def pick_first_data_var(ds: xr.Dataset, preferred: str | None = None) -> xr.DataArray:
    if preferred is not None and preferred in ds.data_vars:
        return ds[preferred]

    for name, da in ds.data_vars.items():
        if name not in ds.coords:
            return da

    raise ValueError("No non-coordinate data variable found in dataset.")


def maybe_get_coord_name(obj, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in obj.coords:
            return name
        if hasattr(obj, "dims") and name in obj.dims:
            return name
    return None


def get_lat_lon_values(obj) -> tuple[pd.Index | None, pd.Index | None]:
    lat_name = maybe_get_coord_name(obj, ["lat", "latitude"])
    lon_name = maybe_get_coord_name(obj, ["lon", "longitude"])

    lat_vals = obj[lat_name].values if lat_name is not None else None
    lon_vals = obj[lon_name].values if lon_name is not None else None
    return lat_vals, lon_vals


def estimate_resolution(values) -> float | None:
    if values is None or len(values) < 2:
        return None
    diffs = pd.Series(values).diff().dropna().abs()
    if len(diffs) == 0:
        return None
    return float(diffs.median())


def format_resolution(lat_vals, lon_vals) -> str:
    dlat = estimate_resolution(lat_vals)
    dlon = estimate_resolution(lon_vals)
    if dlat is None or dlon is None:
        return "-"
    return f"{dlat:.3f}° × {dlon:.3f}°"


def format_time_range(obj) -> str:
    time_name = maybe_get_coord_name(obj, ["time", "valid_time"])
    if time_name is None:
        return "-"
    vals = pd.to_datetime(obj[time_name].values)
    if len(vals) == 0:
        return "-"
    return f"{vals.min().date()} → {vals.max().date()}"


def format_dims(obj) -> str:
    parts = []
    for k, v in obj.sizes.items():
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def get_levels_from_obj(obj) -> list[int]:
    level_name = maybe_get_coord_name(obj, ["level", "pressure_level", "plev"])
    if level_name is None:
        return []
    vals = pd.Series(obj[level_name].values).dropna().tolist()
    out = []
    for v in vals:
        try:
            out.append(int(round(float(v))))
        except Exception:
            continue
    return out


def get_unit(da: xr.DataArray, forced_unit: str | None = None) -> str:
    if forced_unit not in (None, "", "null"):
        return str(forced_unit)
    return str(da.attrs.get("units", "-"))


def level_mark(available_levels: list[int], target_level: int) -> str:
    return "✓" if target_level in available_levels else "-"


def compact_row(
    role: str,
    variable_label: str,
    unit: str,
    available_levels: list[int],
    resolution: str,
) -> dict:
    return {
        "Role": role,
        "Variable": variable_label,
        "Unité": unit,
        "1000": level_mark(available_levels, 1000) if available_levels else "-",
        "850": level_mark(available_levels, 850) if available_levels else "-",
        "700": level_mark(available_levels, 700) if available_levels else "-",
        "500": level_mark(available_levels, 500) if available_levels else "-",
        "Résolution": resolution,
    }


def detailed_row(
    role: str,
    variable_key: str,
    unit: str,
    available_levels: list[int],
    resolution: str,
    time_range: str,
    dims_str: str,
    file_path: str,
    analysis_window: str,
) -> dict:
    row = compact_row(
        role=role,
        variable_label=DISPLAY_NAMES.get(variable_key, variable_key),
        unit=unit,
        available_levels=available_levels,
        resolution=resolution,
    )
    row["Time range"] = time_range
    row["Dimensions"] = dims_str
    row["Analysis window"] = analysis_window
    row["File path"] = file_path
    return row


def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_table_figure(df: pd.DataFrame, fig_path: Path, title: str):
    nrows, ncols = df.shape
    fig_w = max(12, ncols * 1.6)
    fig_h = max(3.5, 0.65 * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EFEFEF")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    explore_cfg_path = Path(args.config).resolve()
    explore_cfg = load_yaml(explore_cfg_path)

    main_cfg_path = resolve_from_project_root(
        explore_cfg["project"].get("main_config_path")
    )
    if main_cfg_path is None:
        raise ValueError("project.main_config_path must be provided in config.yaml")
    if not main_cfg_path.exists():
        raise FileNotFoundError(f"Main config not found: {main_cfg_path}")

    cfg = load_config(train_mode=False, path=str(main_cfg_path))
    if str(cfg.variable).lower() != "temp":
        raise ValueError("This exploration is only for temperature experiments.")

    exp_path = Path(build_experiment_path(cfg))
    data_dir, plot_dir = ensure_exploration_dirs(exp_path)

    predictor_vars = explore_cfg["exploration"].get("predictor_variables", ["z", "t", "q", "u", "v"])
    configured_levels = explore_cfg["exploration"].get("predictor_levels", [1000, 850, 700, 500])
    include_prediction = bool(explore_cfg["exploration"].get("include_prediction", True))

    predictor_pattern = get_predictor_pattern(explore_cfg, cfg)
    pred_path, ref_path = get_prediction_and_reference_paths(explore_cfg, cfg)
    analysis_start, analysis_end = get_time_window(explore_cfg, cfg)
    analysis_window = f"{analysis_start} → {analysis_end}"

    compact_rows = []
    detailed_rows = []
    channel_rows = []

    print("=== Temperature exploration ===")
    print(f"Main config      : {main_cfg_path}")
    print(f"Experiment root  : {exp_path}")
    print(f"Exploration data : {data_dir}")
    print(f"Exploration plots: {plot_dir}")
    print(f"Analysis window  : {analysis_window}")

    # -----------------------------------------------------
    # 1) Predictors
    # -----------------------------------------------------
    for var in predictor_vars:
        file_path = Path(predictor_pattern.format(var=var))
        if not file_path.exists():
            detailed_rows.append(
                detailed_row(
                    role="Predictor",
                    variable_key=var,
                    unit="-",
                    available_levels=[],
                    resolution="-",
                    time_range="-",
                    dims_str="MISSING",
                    file_path=str(file_path),
                    analysis_window=analysis_window,
                )
            )
            compact_rows.append(
                compact_row(
                    role="Predictor",
                    variable_label=DISPLAY_NAMES.get(var, var),
                    unit="-",
                    available_levels=[],
                    resolution="-",
                )
            )
            continue

        with xr.open_dataset(file_path) as ds:
            da = pick_first_data_var(ds, preferred=var)
            lat_vals, lon_vals = get_lat_lon_values(da)
            available_levels = get_levels_from_obj(da)
            unit = get_unit(da)
            resolution = format_resolution(lat_vals, lon_vals)
            time_range = format_time_range(da)
            dims_str = format_dims(da)

        compact_rows.append(
            compact_row(
                role="Predictor",
                variable_label=DISPLAY_NAMES.get(var, var),
                unit=unit,
                available_levels=available_levels,
                resolution=resolution,
            )
        )
        detailed_rows.append(
            detailed_row(
                role="Predictor",
                variable_key=var,
                unit=unit,
                available_levels=available_levels,
                resolution=resolution,
                time_range=time_range,
                dims_str=dims_str,
                file_path=str(file_path),
                analysis_window=analysis_window,
            )
        )

        for level in configured_levels:
            channel_rows.append(
                {
                    "variable_key": var,
                    "variable_label": DISPLAY_NAMES.get(var, var),
                    "level_hpa": level,
                    "available": level in available_levels,
                    "unit": unit,
                    "time_range": time_range,
                    "resolution": resolution,
                    "file_path": str(file_path),
                }
            )

    # -----------------------------------------------------
    # 2) Reference target
    # -----------------------------------------------------
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference target not found: {ref_path}")

    ref_var = explore_cfg["data"].get("reference_var", None)
    ref_units_forced = explore_cfg["data"].get("reference_units", None)

    ref_da = open_temperature_dataarray(ref_path, var_name=ref_var)
    ref_lat, ref_lon = get_lat_lon_values(ref_da)

    ref_unit = get_unit(ref_da, forced_unit=ref_units_forced)
    ref_resolution = format_resolution(ref_lat, ref_lon)
    ref_time_range = format_time_range(ref_da)
    ref_dims = format_dims(ref_da)

    compact_rows.append(
        compact_row(
            role="Predictand",
            variable_label=DISPLAY_NAMES["target"],
            unit=ref_unit,
            available_levels=[],
            resolution=ref_resolution,
        )
    )
    detailed_rows.append(
        detailed_row(
            role="Predictand",
            variable_key="target",
            unit=ref_unit,
            available_levels=[],
            resolution=ref_resolution,
            time_range=ref_time_range,
            dims_str=ref_dims,
            file_path=str(ref_path),
            analysis_window=analysis_window,
        )
    )

    # -----------------------------------------------------
    # 3) Model prediction
    # -----------------------------------------------------
    if include_prediction:
        pred_var = explore_cfg["data"].get("prediction_var", "air_temperature")
        pred_units_forced = explore_cfg["data"].get("prediction_units", None)

        if pred_path.exists():
            pred_da = open_temperature_dataarray(pred_path, var_name=pred_var)
            pred_lat, pred_lon = get_lat_lon_values(pred_da)

            pred_unit = get_unit(pred_da, forced_unit=pred_units_forced)
            pred_resolution = format_resolution(pred_lat, pred_lon)
            pred_time_range = format_time_range(pred_da)
            pred_dims = format_dims(pred_da)
        else:
            pred_unit = "-"
            pred_resolution = "-"
            pred_time_range = "-"
            pred_dims = "MISSING"

        compact_rows.append(
            compact_row(
                role="Model output",
                variable_label=DISPLAY_NAMES["prediction"],
                unit=pred_unit,
                available_levels=[],
                resolution=pred_resolution,
            )
        )
        detailed_rows.append(
            detailed_row(
                role="Model output",
                variable_key="prediction",
                unit=pred_unit,
                available_levels=[],
                resolution=pred_resolution,
                time_range=pred_time_range,
                dims_str=pred_dims,
                file_path=str(pred_path),
                analysis_window=analysis_window,
            )
        )

    compact_df = pd.DataFrame(compact_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    channel_df = pd.DataFrame(channel_rows)

    if bool(explore_cfg["exploration"].get("save_compact_table_csv", True)):
        compact_df.to_csv(data_dir / "temperature_overview_compact.csv", index=False)

    if bool(explore_cfg["exploration"].get("save_detailed_table_csv", True)):
        detailed_df.to_csv(data_dir / "temperature_overview_detailed.csv", index=False)

    if bool(explore_cfg["exploration"].get("save_channel_csv", True)):
        channel_df.to_csv(data_dir / "predictor_channels_summary.csv", index=False)

    if bool(explore_cfg["exploration"].get("save_json", True)):
        save_json(
            {
                "analysis_window": analysis_window,
                "compact_rows": compact_rows,
                "detailed_rows": detailed_rows,
                "predictor_channels": channel_rows,
            },
            data_dir / "temperature_overview.json",
        )

    if bool(explore_cfg["exploration"].get("save_table_png", True)):
        save_table_figure(
            compact_df,
            plot_dir / "temperature_overview_table.png",
            title="Temperature experiment: predictors and predictand overview",
        )

    print(f"[SUCCESS] Exploration files saved to: {data_dir} and {plot_dir}")


if __name__ == "__main__":
    main()