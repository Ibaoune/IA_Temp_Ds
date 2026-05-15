from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import yaml

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import ensure_metric_dirs
from ...map_utils import (
    apply_shape_mask,
    plot_metric_map,
    flatten_valid,
    MapStyle,
    compute_sequential_levels,
    load_project_shape,
    draw_project_boundaries,
    _get_lon_lat_2d,
    add_stats_box,
    plot_annual_bias_boxplot,
)


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_METRIC_CONFIG = THIS_FILE.with_name("config.yaml")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot WAMS diagnostics for temperature."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to wams metric config.yaml",
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


# =========================================================
# Colormap
# =========================================================
def get_warm_spell_cmap(n_colors: int) -> mcolors.Colormap:
    colors = [
        "#fff5eb",
        "#fee6ce",
        "#fdd0a2",
        "#fdae6b",
        "#fd8d3c",
        "#f16913",
        "#d94801",
        "#a63603",
        "#7f2704",
    ]
    return mcolors.LinearSegmentedColormap.from_list("warm_spell", colors, N=n_colors)

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
def resolve_wams_file(data_dir: Path) -> Path | None:
    p = data_dir / "wams_annual_mean_period.nc"
    return p if p.exists() else None


def load_wams_fields(
    nc_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path, decode_timedelta=False)

    required = ["wams_pred", "wams_obs", "bwams"]
    for var in required:
        if var not in ds.data_vars:
            raise KeyError(f"'{var}' variable not found in {nc_path}")

    return (
        ds["wams_pred"].values,
        ds["wams_obs"].values,
        ds["bwams"].values,
        ds["lon"].values,
        ds["lat"].values,
    )


# =========================================================
# Comparison panel
# =========================================================
def plot_wams_comparison_panel(
    wams_obs: np.ndarray,
    wams_pred: np.ndarray,
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

    obs_masked = apply_shape_mask(wams_obs, lons, lats, shapefile_path)
    pred_masked = apply_shape_mask(wams_pred, lons, lats, shapefile_path)

    merged = np.concatenate([
        flatten_valid(obs_masked),
        flatten_valid(pred_masked),
    ])

    if merged.size == 0:
        merged = np.array([0.0, 1.0])

    levels = compute_sequential_levels(
        merged,
        n_bins=8,
        robust=robust,
        force_zero_min=True,
    )
    cmap = get_warm_spell_cmap(len(levels) - 1)
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
    titles = [f"{reference_label} WAMS", f"{model_label} WAMS"]

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
        add_stats_box(ax, arr, unit="days")

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Days", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(
        f"Annual WAMS: {reference_label} vs {model_label}",
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

    exp_path = Path(build_experiment_path(cfg))
    data_dir, plot_dir = ensure_metric_dirs(exp_path, "wams")

    print("=== Temperature WAMS plotting ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")

    annual_path = resolve_wams_file(data_dir)
    if annual_path is None:
        raise FileNotFoundError(
            f"WAMS file not found in {data_dir}. Run the WAMS compute first."
        )

    print("[STEP] Loading WAMS fields")
    wams_pred, wams_obs, bwams, lons, lats = load_wams_fields(annual_path)

    print("[STEP] Plotting annual WAMS comparison")
    reference_label = get_target_display_name(cfg)
    plot_wams_comparison_panel(
        wams_obs=wams_obs,
        wams_pred=wams_pred,
        lons=lons,
        lats=lats,
        shapefile_path=cfg.shapefile_path,
        lon_min=cfg.lon_min,
        lon_max=cfg.lon_max,
        lat_min=cfg.lat_min,
        lat_max=cfg.lat_max,
        model_label=str(cfg.model_type).upper(),
        reference_label=reference_label,
        fig_path=plot_dir / "annual_wams_comparison.png",
        robust=args.robust,
        show=args.show,
    )

    print("[STEP] Plotting annual bWAMS map")
    plot_metric_map(
        arr=bwams,
        lons=lons,
        lats=lats,
        title="Annual bWAMS",
        fig_path=plot_dir / "annual_bwams_map.png",
        shapefile_path=cfg.shapefile_path,
        lon_min=cfg.lon_min,
        lon_max=cfg.lon_max,
        lat_min=cfg.lat_min,
        lat_max=cfg.lat_max,
        unit="days",
        metric_type="bias",
        n_bins=11,
        robust=args.robust,
        show=args.show,
    )

    print("[STEP] Plotting annual bWAMS boxplot")
    bwams_masked = apply_shape_mask(
        bwams,
        lons,
        lats,
        cfg.shapefile_path,
    )

    plot_annual_bias_boxplot(
        annual_array=bwams_masked,
        fig_path=plot_dir / "annual_bwams_boxplot.png",
        title="Annual bWAMS distribution",
        ylabel="bWAMS (days)",
        show=args.show,
    )

    print(f"[SUCCESS] WAMS figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()