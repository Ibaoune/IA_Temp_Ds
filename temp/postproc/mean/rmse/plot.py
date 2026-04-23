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
    plot_seasonal_bias_boxplot as plot_seasonal_boxplot,
    plot_annual_bias_boxplot as plot_annual_boxplot,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot temperature RMSE diagnostics from computed RMSE NetCDF files."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to rmse metric config.yaml",
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


def resolve_rmse_file(
    data_dir: Path,
    season: str,
    strategy: str,
    return_by_year: bool,
) -> Path | None:
    mode_suffix = "by_year" if return_by_year else "mean_period"
    path = data_dir / f"rmse_{season.lower()}_{strategy}_{mode_suffix}.nc"
    return path if path.exists() else None


def load_rmse_field(nc_path: Path):
    ds = xr.open_dataset(nc_path)

    if "rmse" not in ds.data_vars:
        raise KeyError(f"'rmse' variable not found in {nc_path}")

    da = ds["rmse"]

    if "year" in da.dims:
        da = da.mean(dim="year", skipna=True)

    return da.values, da["lon"].values, da["lat"].values


def plot_seasonal_rmse_panel(
    seasonal_arrays: list[np.ndarray],
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

    masked_arrays = [
        apply_shape_mask(arr, lons, lats, shapefile_path)
        for arr in seasonal_arrays
    ]

    merged_valid = []
    for arr in masked_arrays:
        vals = flatten_valid(arr)
        if vals.size > 0:
            merged_valid.append(vals)

    if not merged_valid:
        merged = np.array([0.0, 1.0])
    else:
        merged = np.concatenate(merged_valid)

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
        4,
        figsize=style.figsize_panel_1x4,
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )

    for i, ax in enumerate(axes):
        arr = masked_arrays[i]

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

        ax.set_title(SEASON_TITLES[SEASON_ORDER[i]], fontsize=style.panel_title_fontsize, fontweight="bold")
        add_stats_box(ax, arr, unit="°C")

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("RMSE (°C)", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(
        "Seasonal temperature RMSE",
        fontsize=style.title_fontsize + 3,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.08, wspace=0.10)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


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

    metric_name = metric_cfg["metric"].get("name", "rmse")
    strategy = metric_cfg["metric"].get("strategy", "daily_first")
    return_by_year = bool(metric_cfg["metric"].get("return_by_year", False))
    selected_seasons = metric_cfg["metric"].get(
        "seasons", ["Annual", "DJF", "MAM", "JJA", "SON"]
    )

    exp_path = Path(build_experiment_path(cfg))
    data_dir, plot_dir = ensure_metric_dirs(exp_path, metric_name)

    print("=== Temperature RMSE plotting ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")
    print(f"Strategy        : {strategy}")
    print(f"Return by year  : {return_by_year}")

    annual_arr = None
    annual_lons = None
    annual_lats = None

    if "Annual" in selected_seasons:
        annual_path = resolve_rmse_file(
            data_dir=data_dir,
            season="Annual",
            strategy=strategy,
            return_by_year=return_by_year,
        )

        if annual_path is not None:
            print("[STEP] Plotting annual RMSE map")
            annual_arr, annual_lons, annual_lats = load_rmse_field(annual_path)

            plot_metric_map(
                arr=annual_arr,
                lons=annual_lons,
                lats=annual_lats,
                title="Annual temperature RMSE",
                fig_path=plot_dir / f"annual_rmse_map_{strategy}.png",
                shapefile_path=cfg.shapefile_path,
                lon_min=cfg.lon_min,
                lon_max=cfg.lon_max,
                lat_min=cfg.lat_min,
                lat_max=cfg.lat_max,
                unit="°C",
                metric_type="rmse",
                n_bins=9,
                robust=args.robust,
                show=args.show,
            )
        else:
            print("[WARNING] Annual RMSE file not found.")

    seasonal_arrays = []
    seasonal_labels = []
    seasonal_lons = None
    seasonal_lats = None

    for season in SEASON_ORDER:
        if season not in selected_seasons:
            continue

        nc_path = resolve_rmse_file(
            data_dir=data_dir,
            season=season,
            strategy=strategy,
            return_by_year=return_by_year,
        )

        if nc_path is None:
            print(f"[WARNING] Missing seasonal file for {season}")
            continue

        arr, lons, lats = load_rmse_field(nc_path)

        seasonal_arrays.append(arr)
        seasonal_labels.append(season)

        if seasonal_lons is None:
            seasonal_lons = lons
            seasonal_lats = lats

        print(f"[STEP] Plotting {season} RMSE map")
        plot_metric_map(
            arr=arr,
            lons=lons,
            lats=lats,
            title=f"{SEASON_TITLES[season]} temperature RMSE",
            fig_path=plot_dir / f"rmse_{season.lower()}_map_{strategy}.png",
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            unit="°C",
            metric_type="rmse",
            n_bins=9,
            robust=args.robust,
            show=args.show,
        )

    if len(seasonal_arrays) == 4:
        print("[STEP] Plotting grouped seasonal RMSE panel")
        plot_seasonal_rmse_panel(
            seasonal_arrays=seasonal_arrays,
            lons=seasonal_lons,
            lats=seasonal_lats,
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            fig_path=plot_dir / f"seasonal_rmse_panel_{strategy}.png",
            robust=args.robust,
            show=args.show,
        )
    else:
        print("[WARNING] Seasonal RMSE panel not created: one or more seasonal files are missing.")

    if len(seasonal_arrays) == 4:
        print("[STEP] Plotting seasonal RMSE boxplot")

        seasonal_arrays_masked = [
            apply_shape_mask(arr, seasonal_lons, seasonal_lats, cfg.shapefile_path)
            for arr in seasonal_arrays
        ]

        plot_seasonal_boxplot(
            seasonal_arrays=seasonal_arrays_masked,
            labels=seasonal_labels,
            fig_path=plot_dir / f"seasonal_rmse_boxplot_{strategy}.png",
            title="Seasonal RMSE distribution",
            ylabel="RMSE (°C)",
            show=args.show,
        )
    else:
        print("[WARNING] Seasonal RMSE boxplot skipped: seasonal arrays incomplete.")

    if annual_arr is not None:
        print("[STEP] Plotting annual RMSE boxplot")

        annual_arr_masked = apply_shape_mask(
            annual_arr,
            annual_lons,
            annual_lats,
            cfg.shapefile_path,
        )

        plot_annual_boxplot(
            annual_array=annual_arr_masked,
            fig_path=plot_dir / f"annual_rmse_boxplot_{strategy}.png",
            title="Annual RMSE distribution",
            ylabel="RMSE (°C)",
            show=args.show,
        )
    else:
        print("[WARNING] Annual RMSE boxplot skipped: annual array missing.")

    print(f"[SUCCESS] RMSE figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()