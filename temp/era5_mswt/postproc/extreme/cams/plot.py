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
from ...common import ensure_spatial_metric_dirs, get_spatial_context, get_plot_display_domain
from ...map_utils import (
    get_display_array,
    get_plot_extent,
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
        description="Plot CAMS diagnostics for temperature."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to cams metric config.yaml",
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
def get_cold_spell_cmap(n_colors: int) -> mcolors.Colormap:
    colors = [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ]
    return mcolors.LinearSegmentedColormap.from_list("cold_spell", colors, N=n_colors)


# =========================================================
# File helpers
# =========================================================
def resolve_cams_file(data_dir: Path) -> Path | None:
    p = data_dir / "cams_annual_mean_period.nc"
    return p if p.exists() else None


def load_cams_fields(
    nc_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path, decode_timedelta=False)

    required = ["cams_pred", "cams_obs", "bcams"]
    for var in required:
        if var not in ds.data_vars:
            raise KeyError(f"'{var}' variable not found in {nc_path}")

    return (
        ds["cams_pred"].values,
        ds["cams_obs"].values,
        ds["bcams"].values,
        ds["lon"].values,
        ds["lat"].values,
    )

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
# Comparison panel
# =========================================================
def plot_cams_comparison_panel(
    cams_obs: np.ndarray,
    cams_pred: np.ndarray,
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
    display_domain: str = "morocco_shape",
) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm

    style = MapStyle()

    obs_display = get_display_array(cams_obs, lons, lats, shapefile_path, display_domain)
    pred_display = get_display_array(cams_pred, lons, lats, shapefile_path, display_domain)

    merged = np.concatenate([
        flatten_valid(cams_obs),
        flatten_valid(cams_pred),
    ])

    if merged.size == 0:
        merged = np.array([0.0, 1.0])

    levels = compute_sequential_levels(
        merged,
        n_bins=8,
        robust=robust,
        force_zero_min=True,
    )
    cmap = get_cold_spell_cmap(len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    shape_gdf = load_project_shape(shapefile_path)
    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))
    extent = get_plot_extent(
        lons, lats, lon_min, lon_max, lat_min, lat_max, display_domain
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )

    display_arrays = [obs_display, pred_display]
    stats_arrays = [cams_obs, cams_pred]
    titles = [f"{reference_label} CAMS", f"{model_label} CAMS"]

    for i, ax in enumerate(axes):
        arr_display = display_arrays[i]
        arr_stats = stats_arrays[i]

        im = ax.pcolormesh(
            lon2d,
            lat2d,
            arr_display,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
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
        add_stats_box(ax, arr_stats, unit="days")

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Days", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(
        f"Annual CAMS: {reference_label} vs {model_label}",
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
    spatial_ctx = get_spatial_context(
        metric_cfg=metric_cfg,
        cfg=cfg,
        project_root=PROJECT_ROOT,
    )
    display_domain = get_plot_display_domain(metric_cfg)
    data_dir, plot_dir = ensure_spatial_metric_dirs(
        exp_path=exp_path,
        metric_name="cams",
        eval_domain=spatial_ctx.eval_domain,
    )

    print("=== Temperature CAMS plotting ===")
    print(f"Spatial domain  : {spatial_ctx.eval_domain}")
    print(f"Display domain  : {display_domain}")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")

    annual_path = resolve_cams_file(data_dir)
    if annual_path is None:
        raise FileNotFoundError(
            f"CAMS file not found in {data_dir}. Run the CAMS compute first."
        )

    print("[STEP] Loading CAMS fields")
    cams_pred, cams_obs, bcams, lons, lats = load_cams_fields(annual_path)
    reference_label = get_target_display_name(cfg)
    
    print("[STEP] Plotting annual CAMS comparison")
    plot_cams_comparison_panel(
        cams_obs=cams_obs,
        cams_pred=cams_pred,
        lons=lons,
        lats=lats,
        shapefile_path=cfg.shapefile_path,
        lon_min=cfg.lon_min,
        lon_max=cfg.lon_max,
        lat_min=cfg.lat_min,
        lat_max=cfg.lat_max,
        reference_label=reference_label,
        model_label=str(cfg.model_type).upper(),
        fig_path=plot_dir / "annual_cams_comparison.png",
        robust=args.robust,
        show=args.show,
        display_domain=display_domain,
    )

    print("[STEP] Plotting annual bCAMS map")
    plot_metric_map(
        arr=bcams,
        lons=lons,
        lats=lats,
        title="Annual bCAMS",
        fig_path=plot_dir / "annual_bcams_map.png",
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
        apply_mask_in_plot=(display_domain == "morocco_shape"),
        display_domain=display_domain,
        stats_arr=bcams,
    )

    print("[STEP] Plotting annual bCAMS boxplot")
    plot_annual_bias_boxplot(
        annual_array=bcams,
        fig_path=plot_dir / "annual_bcams_boxplot.png",
        title="Annual bCAMS distribution",
        ylabel="bCAMS (days)",
        show=args.show,
    )

    print(f"[SUCCESS] CAMS figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()
