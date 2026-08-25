from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import yaml

from ..main.src.core.config import load_config
from ..main.src.core.utils import build_experiment_path
from .common import (
    align_prediction_and_observation,
    convert_temperature_to_celsius,
    open_temperature_dataarray,
    subset_test_period,
)
from .map_utils import (
    apply_shape_mask,
    MapStyle,
    compute_sequential_levels,
    get_temperature_cmap,
    load_project_shape,
    draw_project_boundaries,
    _get_lon_lat_2d,
    add_stats_box,
)


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
DEFAULT_CONFIG = THIS_FILE.with_name("explore_config.yaml")

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_TITLES = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot seasonal temperature comparison: observation vs model."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to exploration config.yaml",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure interactively.",
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


def get_prediction_and_reference_paths(explore_cfg: dict, cfg) -> tuple[Path, Path]:
    pred_override = resolve_from_project_root(explore_cfg["data"].get("prediction_path"))
    ref_override = resolve_from_project_root(explore_cfg["data"].get("reference_path"))

    pred_path = pred_override if pred_override is not None else build_prediction_path(cfg)
    obs_path = ref_override if ref_override is not None else Path(cfg.target_path)

    return pred_path, obs_path


# =========================================================
# Seasonal climatology helpers
# =========================================================
def compute_one_season_mean(da: xr.DataArray, months: list[int]) -> xr.DataArray:
    month_mask = da["time"].dt.month.isin(months)
    sub = da.sel(time=month_mask)

    if sub.sizes["time"] == 0:
        raise ValueError(f"No time steps found for months={months}")

    return sub.mean(dim="time", skipna=True)


def compute_seasonal_climatologies(
    pred: xr.DataArray,
    obs: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    season_months = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    obs_list = []
    pred_list = []

    for season in SEASON_ORDER:
        months = season_months[season]
        obs_mean = compute_one_season_mean(obs, months)
        pred_mean = compute_one_season_mean(pred, months)

        obs_list.append(obs_mean.values)
        pred_list.append(pred_mean.values)

    obs_seasonal = np.stack(obs_list, axis=0)
    pred_seasonal = np.stack(pred_list, axis=0)

    lons = obs["lon"].values
    lats = obs["lat"].values

    return obs_seasonal, pred_seasonal, lons, lats


# =========================================================
# Main plot
# =========================================================
def plot_main_comparison(
    obs_seasonal: np.ndarray,
    pred_seasonal: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    cfg,
    fig_path: Path,
    robust: bool = True,
    save_hd: bool = True,
    show: bool = False,
) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm

    style = MapStyle()

    obs_masked = np.stack(
        [apply_shape_mask(arr, lons, lats, cfg.shapefile_path) for arr in obs_seasonal],
        axis=0,
    )
    pred_masked = np.stack(
        [apply_shape_mask(arr, lons, lats, cfg.shapefile_path) for arr in pred_seasonal],
        axis=0,
    )

    merged = np.concatenate([
        obs_masked[np.isfinite(obs_masked)],
        pred_masked[np.isfinite(pred_masked)],
    ])

    if merged.size == 0:
        merged = np.array([0.0, 1.0])

    levels = compute_sequential_levels(
        merged,
        n_bins=9,
        robust=robust,
        force_zero_min=False,
    )
    cmap = get_temperature_cmap(len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    shape_gdf = load_project_shape(cfg.shapefile_path)
    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20, 11),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )

    for season_idx, season_name in enumerate(SEASON_TITLES):
        ax = axes[0, season_idx]
        im = ax.pcolormesh(
            lon2d,
            lat2d,
            obs_masked[season_idx],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_extent([cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=style.coast_linewidth, zorder=4)
        draw_project_boundaries(ax, shape_gdf)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=style.grid_linewidth,
            linestyle="--",
            alpha=style.grid_alpha,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": style.tick_labelsize}
        gl.ylabel_style = {"size": style.tick_labelsize}

        ax.set_title(season_name, fontsize=style.panel_title_fontsize, fontweight="bold")
        add_stats_box(ax, obs_masked[season_idx], unit="°C")

        ax = axes[1, season_idx]
        im = ax.pcolormesh(
            lon2d,
            lat2d,
            pred_masked[season_idx],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )
        ax.set_extent([cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=style.coast_linewidth, zorder=4)
        draw_project_boundaries(ax, shape_gdf)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=style.grid_linewidth,
            linestyle="--",
            alpha=style.grid_alpha,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": style.tick_labelsize}
        gl.ylabel_style = {"size": style.tick_labelsize}

        add_stats_box(ax, pred_masked[season_idx], unit="°C")

    fig.text(0.02, 0.70, "OBS", fontsize=16, fontweight="bold", rotation=90, va="center", ha="center")
    fig.text(
        0.02,
        0.30,
        str(cfg.model_type).upper(),
        fontsize=16,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Temperature (°C)", fontsize=style.cbar_labelsize, labelpad=10)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(
        f"Seasonal Temperature Patterns: Observations vs {str(cfg.model_type).upper()}",
        fontsize=style.title_fontsize + 4,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.05, wspace=0.10, hspace=0.20)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if save_hd:
        fig_path_hd = fig_path.with_name(fig_path.stem + "_hd.png")
        fig.savefig(fig_path_hd, dpi=300, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


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
        raise ValueError("This script is only for temperature experiments.")

    pred_path, obs_path = get_prediction_and_reference_paths(explore_cfg, cfg)
    _, plot_dir = ensure_exploration_dirs(Path(build_experiment_path(cfg)))

    pred_var = explore_cfg["data"].get("prediction_var", "air_temperature")
    obs_var = explore_cfg["data"].get("reference_var", None)
    pred_units = explore_cfg["data"].get("prediction_units", None)
    obs_units = explore_cfg["data"].get("reference_units", None)

    start_date, end_date = get_time_window(explore_cfg, cfg)

    robust = bool(explore_cfg.get("plots", {}).get("seasonal_compare", {}).get("robust", True))
    save_hd = bool(explore_cfg.get("plots", {}).get("seasonal_compare", {}).get("save_hd", True))
    figure_name = str(
        explore_cfg.get("plots", {}).get("seasonal_compare", {}).get(
            "figure_name", "seasonal_temperature_comparison.png"
        )
    )

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    print("=== Seasonal temperature comparison plotting ===")
    print(f"Prediction file : {pred_path}")
    print(f"Observation file: {obs_path}")
    print(f"Time window     : {start_date} -> {end_date}")
    print(f"Output plot dir : {plot_dir}")

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

    print("[STEP] Computing seasonal climatologies")
    obs_seasonal, pred_seasonal, lons, lats = compute_seasonal_climatologies(pred=pred, obs=obs)

    print("[STEP] Plotting seasonal comparison figure")
    plot_main_comparison(
        obs_seasonal=obs_seasonal,
        pred_seasonal=pred_seasonal,
        lons=lons,
        lats=lats,
        cfg=cfg,
        fig_path=plot_dir / figure_name,
        robust=robust,
        save_hd=save_hd,
        show=args.show,
    )

    print(f"[SUCCESS] Seasonal comparison figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()