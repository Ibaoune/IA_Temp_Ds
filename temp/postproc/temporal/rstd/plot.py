from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import yaml

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import ensure_spatial_metric_dirs, get_spatial_context
from ...map_utils import (
    apply_shape_mask,
    flatten_valid,
    MapStyle,
    compute_sequential_levels,
    get_temperature_cmap,
    load_project_shape,
    draw_project_boundaries,
    _get_lon_lat_2d,
    add_stats_box,
)


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_METRIC_CONFIG = THIS_FILE.with_name("config.yaml")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot RSTD diagnostics for temperature."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to rstd metric config.yaml",
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
def resolve_rstd_file(data_dir: Path) -> Path | None:
    p = data_dir / "rstd_annual_mean_period.nc"
    return p if p.exists() else None


def load_rstd_fields(
    nc_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(nc_path)

    required = ["std_pred", "std_obs", "rstd"]
    for var in required:
        if var not in ds.data_vars:
            raise KeyError(f"'{var}' variable not found in {nc_path}")

    return (
        ds["std_pred"].values,
        ds["std_obs"].values,
        ds["rstd"].values,
        ds["lon"].values,
        ds["lat"].values,
    )


# =========================================================
# Comparison panel
# =========================================================
def plot_std_comparison_panel(
    std_obs: np.ndarray,
    std_pred: np.ndarray,
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

    obs_display = apply_shape_mask(std_obs, lons, lats, shapefile_path)
    pred_display = apply_shape_mask(std_pred, lons, lats, shapefile_path)

    merged = np.concatenate([
        flatten_valid(std_obs),
        flatten_valid(std_pred),
    ])

    if merged.size == 0:
        merged = np.array([0.0, 1.0])

    levels = compute_sequential_levels(
        merged,
        n_bins=9,
        robust=robust,
        force_zero_min=True,
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

    display_arrays = [obs_display, pred_display]
    stats_arrays = [std_obs, std_pred]
    titles = [f"{reference_label} std", f"{model_label} std"]

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
        add_stats_box(ax, arr_stats, unit="°C")

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Std of anomalies (°C)", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(
        f"Annual variability amplitude: {reference_label} vs {model_label}",
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


def plot_single_rstd_map(
    rstd: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    shapefile_path: str | Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    fig_path: Path,
    robust: bool = True,
    show: bool = False,
) -> Path:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm

    style = MapStyle()

    arr_stats = np.asarray(rstd, dtype=float)
    arr_display = apply_shape_mask(rstd, lons, lats, shapefile_path)

    levels = compute_sequential_levels(
        arr_stats,
        n_bins=9,
        robust=robust,
        force_zero_min=True,
    )
    cmap = get_temperature_cmap(len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    shape_gdf = load_project_shape(shapefile_path)
    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))

    fig, ax = plt.subplots(
        figsize=style.figsize_single,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )

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

    cbar = plt.colorbar(
        im,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing="proportional",
        shrink=0.88,
        pad=0.03,
    )
    cbar.set_label("RSTD", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    ax.set_title("Annual RSTD", fontsize=style.title_fontsize, pad=10)
    add_stats_box(ax, arr_stats, unit=None)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


def plot_annual_rstd_boxplot(
    annual_array: np.ndarray,
    fig_path: Path,
    title: str = "Annual RSTD distribution",
    ylabel: str = "RSTD",
    show: bool = False,
    style: MapStyle = MapStyle(),
) -> Path | None:
    data = flatten_valid(annual_array)

    if data.size == 0:
        print("[WARNING] No valid data available for annual RSTD boxplot.")
        return None

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=style.dpi)

    box_color = "#4C78A8"
    edge_color = "#4F4F4F"

    ax.boxplot(
        [data],
        labels=["Annual"],
        patch_artist=True,
        widths=0.32,
        showmeans=False,
        showfliers=False,
        whis=(0, 100),
        boxprops=dict(
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=1.6,
            alpha=0.95,
        ),
        whiskerprops=dict(color=edge_color, linewidth=1.4),
        capprops=dict(color=edge_color, linewidth=1.4),
        medianprops=dict(color="black", linewidth=1.2),
    )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.4, alpha=0.8)

    ax.set_title(title, fontsize=style.title_fontsize + 4, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.28)

    fig_path = Path(fig_path)
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
    data_dir, plot_dir = ensure_spatial_metric_dirs(
        exp_path=exp_path,
        metric_name="rstd",
        eval_domain=spatial_ctx.eval_domain,
    )

    print("=== Temperature RSTD plotting ===")
    print(f"Spatial domain  : {spatial_ctx.eval_domain}")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")

    annual_path = resolve_rstd_file(data_dir)
    if annual_path is None:
        raise FileNotFoundError(
            f"RSTD file not found in {data_dir}. Run the RSTD compute first."
        )

    print("[STEP] Loading RSTD fields")
    std_pred, std_obs, rstd, lons, lats = load_rstd_fields(annual_path)

    print("[STEP] Plotting annual standard deviation comparison")
    reference_label = get_target_display_name(cfg)
    plot_std_comparison_panel(
        std_obs=std_obs,
        std_pred=std_pred,
        lons=lons,
        lats=lats,
        reference_label=reference_label,
        shapefile_path=cfg.shapefile_path,
        lon_min=cfg.lon_min,
        lon_max=cfg.lon_max,
        lat_min=cfg.lat_min,
        lat_max=cfg.lat_max,
        model_label=str(cfg.model_type).upper(),
        fig_path=plot_dir / "annual_std_comparison.png",
        robust=args.robust,
        show=args.show,
    )

    print("[STEP] Plotting annual RSTD map")
    plot_single_rstd_map(
        rstd=rstd,
        lons=lons,
        lats=lats,
        shapefile_path=cfg.shapefile_path,
        lon_min=cfg.lon_min,
        lon_max=cfg.lon_max,
        lat_min=cfg.lat_min,
        lat_max=cfg.lat_max,
        fig_path=plot_dir / "annual_rstd_map.png",
        robust=args.robust,
        show=args.show,
    )

    print("[STEP] Plotting annual RSTD boxplot")
    plot_annual_rstd_boxplot(
        annual_array=rstd,
        fig_path=plot_dir / "annual_rstd_boxplot.png",
        title="Annual RSTD distribution",
        ylabel="RSTD",
        show=args.show,
    )

    print(f"[SUCCESS] RSTD figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()
