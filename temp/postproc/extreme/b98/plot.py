from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import yaml

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import ensure_metric_dirs
from ...map_utils import (
    apply_shape_mask,
    plot_metric_map,
    plot_seasonal_bias_panel,
    plot_seasonal_bias_boxplot,
    plot_annual_bias_boxplot,
    flatten_valid,
    MapStyle,
    compute_sequential_levels,
    get_temperature_cmap,
    load_project_shape,
    draw_project_boundaries,
    _get_lon_lat_2d,
    add_stats_box,
)


SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_TITLES = {
    "DJF": "Winter (DJF)",
    "MAM": "Spring (MAM)",
    "JJA": "Summer (JJA)",
    "SON": "Autumn (SON)",
}

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_METRIC_CONFIG = THIS_FILE.with_name("config.yaml")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot B-98 diagnostics for temperature."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to b98 metric config.yaml",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use robust percentile-based color scaling.",
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

def get_target_display_name(cfg):
    target = str(cfg.target).lower()

    if target == "mswt":
        return "MSWT"
    elif target == "lmdz35":
        return "LMDZ35"
    elif target == "lmdz":
        return "LMDZ"
    else:
        return target.upper()

# =========================================================
# File helpers
# =========================================================
def resolve_b98_file(data_dir: Path, season: str) -> Path | None:
    path = data_dir / f"b98_{season.lower()}_mean_period.nc"
    return path if path.exists() else None


def load_b98_fields(
    nc_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path)

    required = ["p98_pred", "p98_obs", "b98"]
    for var in required:
        if var not in ds.data_vars:
            raise KeyError(f"'{var}' variable not found in {nc_path}")

    return (
        ds["p98_pred"].values,
        ds["p98_obs"].values,
        ds["b98"].values,
        ds["lon"].values,
        ds["lat"].values,
    )


# =========================================================
# Annual comparison panel
# =========================================================
def plot_p98_comparison_panel(
    p98_obs: np.ndarray,
    p98_pred: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    shapefile_path: str | Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    model_label: str,
    reference_label: str,
    fig_path: Path,
    robust: bool = True,
    show: bool = False,
) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm

    style = MapStyle()

    obs_masked = apply_shape_mask(p98_obs, lons, lats, shapefile_path)
    pred_masked = apply_shape_mask(p98_pred, lons, lats, shapefile_path)

    merged = np.concatenate(
        [
            flatten_valid(obs_masked),
            flatten_valid(pred_masked),
        ]
    )

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

    shape_gdf = load_project_shape(shapefile_path)
    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )

    arrays = [obs_masked, pred_masked]
    titles = [f"{reference_label} P98", f"{model_label} P98"]

    for i, ax in enumerate(axes):
        arr = arrays[i]

        im = ax.pcolormesh(
            lon2d,
            lat2d,
            arr,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
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

        ax.set_title(titles[i], fontsize=style.panel_title_fontsize, fontweight="bold")
        add_stats_box(ax, arr, unit="°C")

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Temperature (°C)", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(
        f"Annual P98 temperature: {reference_label} vs {model_label}",
        fontsize=style.title_fontsize + 3,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.08, wspace=0.08)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    metric_cfg_path = Path(args.metric_config).resolve()
    metric_cfg = load_yaml(metric_cfg_path)

    main_cfg_path = resolve_from_project_root(
        metric_cfg["project"].get("main_config_path")
    )
    if main_cfg_path is None:
        raise ValueError("project.main_config_path must be provided in metric config.")

    cfg = load_config(train_mode=False, path=str(main_cfg_path))

    if str(cfg.variable).lower() != "temp":
        raise ValueError("This plot script is only for temperature experiments.")

    selected_seasons = metric_cfg["metric"].get(
        "seasons", ["Annual", "DJF", "MAM", "JJA", "SON"]
    )

    exp_path = Path(build_experiment_path(cfg))
    data_dir, plot_dir = ensure_metric_dirs(exp_path, "b98")

    print("=== Temperature B-98 plotting ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")

    # -------------------------------------------------
    # 1) Annual comparison + annual B-98 map
    # -------------------------------------------------
    annual_b98 = None
    annual_lons = None
    annual_lats = None

    if "Annual" in selected_seasons:
        annual_path = resolve_b98_file(data_dir, "Annual")

        if annual_path is not None:
            print("[STEP] Plotting annual P98 comparison")
            p98_pred, p98_obs, annual_b98, annual_lons, annual_lats = load_b98_fields(annual_path)
            reference_label = get_target_display_name(cfg)

            plot_p98_comparison_panel(
                p98_obs=p98_obs,
                p98_pred=p98_pred,
                lons=annual_lons,
                lats=annual_lats,
                shapefile_path=cfg.shapefile_path,
                lon_min=cfg.lon_min,
                lon_max=cfg.lon_max,
                lat_min=cfg.lat_min,
                lat_max=cfg.lat_max,
                model_label=str(cfg.model_type).upper(),
                reference_label=reference_label,
                fig_path=plot_dir / "annual_p98_comparison.png",
                robust=args.robust,
                show=args.show,
            )

            print("[STEP] Plotting annual B-98 map")
            plot_metric_map(
                arr=annual_b98,
                lons=annual_lons,
                lats=annual_lats,
                title="Annual B-98 of temperature",
                fig_path=plot_dir / "annual_b98_map.png",
                shapefile_path=cfg.shapefile_path,
                lon_min=cfg.lon_min,
                lon_max=cfg.lon_max,
                lat_min=cfg.lat_min,
                lat_max=cfg.lat_max,
                unit="°C",
                metric_type="bias",
                n_bins=11,
                robust=args.robust,
                show=args.show,
            )
        else:
            print("[WARNING] Annual B-98 file not found.")

    # -------------------------------------------------
    # 2) Individual seasonal B-98 maps + grouped panel
    # -------------------------------------------------
    seasonal_b98 = []
    seasonal_labels = []
    seasonal_lons = None
    seasonal_lats = None

    for season in SEASON_ORDER:
        if season not in selected_seasons:
            continue

        nc_path = resolve_b98_file(data_dir, season)
        if nc_path is None:
            print(f"[WARNING] Missing seasonal file for {season}")
            continue

        _, _, b98, lons, lats = load_b98_fields(nc_path)
        seasonal_b98.append(b98)
        seasonal_labels.append(season)

        if seasonal_lons is None:
            seasonal_lons = lons
            seasonal_lats = lats

        print(f"[STEP] Plotting {season} B-98 map")
        plot_metric_map(
            arr=b98,
            lons=lons,
            lats=lats,
            title=f"{SEASON_TITLES[season]} B-98 of temperature",
            fig_path=plot_dir / f"b98_{season.lower()}_map.png",
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            unit="°C",
            metric_type="bias",
            n_bins=11,
            robust=args.robust,
            show=args.show,
        )

    if len(seasonal_b98) == 4:
        print("[STEP] Plotting grouped seasonal B-98 panel")
        plot_seasonal_bias_panel(
            seasonal_arrays=seasonal_b98,
            lons=seasonal_lons,
            lats=seasonal_lats,
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            fig_path=plot_dir / "seasonal_b98_panel.png",
            season_titles=[SEASON_TITLES[s] for s in seasonal_labels],
            title="Seasonal B-98 of temperature",
            unit="°C",
            n_bins=11,
            robust=args.robust,
            show=args.show,
        )
    else:
        print("[WARNING] Seasonal B-98 panel not created: one or more seasonal files are missing.")

    # -------------------------------------------------
    # 3) Seasonal boxplot
    # -------------------------------------------------
    if len(seasonal_b98) == 4:
        print("[STEP] Plotting seasonal B-98 boxplot")

        seasonal_b98_masked = [
            apply_shape_mask(arr, seasonal_lons, seasonal_lats, cfg.shapefile_path)
            for arr in seasonal_b98
        ]

        plot_seasonal_bias_boxplot(
            seasonal_arrays=seasonal_b98_masked,
            labels=seasonal_labels,
            fig_path=plot_dir / "seasonal_b98_boxplot.png",
            title="Seasonal B-98 distribution",
            ylabel="B-98 (°C)",
            show=args.show,
        )
    else:
        print("[WARNING] Seasonal B-98 boxplot skipped: seasonal arrays incomplete.")

    # -------------------------------------------------
    # 4) Annual boxplot
    # -------------------------------------------------
    if annual_b98 is not None:
        print("[STEP] Plotting annual B-98 boxplot")

        annual_b98_masked = apply_shape_mask(
            annual_b98,
            annual_lons,
            annual_lats,
            cfg.shapefile_path,
        )

        plot_annual_bias_boxplot(
            annual_array=annual_b98_masked,
            fig_path=plot_dir / "annual_b98_boxplot.png",
            title="Annual B-98 distribution",
            ylabel="B-98 (°C)",
            show=args.show,
        )
    else:
        print("[WARNING] Annual B-98 boxplot skipped: annual array missing.")

    print(f"[SUCCESS] B-98 figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()