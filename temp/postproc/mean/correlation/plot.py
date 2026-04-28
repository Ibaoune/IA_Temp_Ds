from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import ensure_metric_dirs
from ...map_utils import (
    apply_shape_mask,
    plot_metric_map,
    plot_annual_bias_boxplot,
    plot_seasonal_bias_boxplot,
    plot_seasonal_bias_panel,
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
        description="Plot temperature correlation diagnostics from computed NetCDF files."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to correlation metric config.yaml",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--robust", action="store_true")
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


def resolve_corr_file(data_dir: Path, corr_type: str, season: str) -> Path | None:
    path = data_dir / f"{corr_type}_{season.lower()}_mean_period.nc"
    return path if path.exists() else None


def load_corr_field(nc_path: Path, corr_type: str):
    ds = xr.open_dataset(nc_path)

    corr_var = corr_type
    sig_var = f"{corr_type}_sig"

    if corr_var not in ds.data_vars:
        raise KeyError(f"{corr_var!r} not found in {nc_path}")

    da = ds[corr_var]

    if "year" in da.dims:
        da = da.mean(dim="year", skipna=True)

    sig = None
    if sig_var in ds.data_vars:
        sig = ds[sig_var]
        if "year" in sig.dims:
            sig = sig.mean(dim="year", skipna=True)

    return da, sig


def get_lat_lon(da: xr.DataArray):
    if "lat" in da.coords:
        lat = da["lat"].values
    elif "latitude" in da.coords:
        lat = da["latitude"].values
    else:
        raise KeyError("Latitude coordinate not found.")

    if "lon" in da.coords:
        lon = da["lon"].values
    elif "longitude" in da.coords:
        lon = da["longitude"].values
    else:
        raise KeyError("Longitude coordinate not found.")

    return lon, lat


def plot_corr_with_significance(
    corr_da: xr.DataArray,
    sig_da: xr.DataArray | None,
    title: str,
    fig_path: Path,
    cfg,
    show: bool = False,
):
    arr = corr_da.values
    lons, lats = get_lat_lon(corr_da)

    masked_arr = apply_shape_mask(
        arr,
        lons,
        lats,
        cfg.shapefile_path,
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.pcolormesh(
        lons,
        lats,
        masked_arr,
        shading="auto",
        vmin=0,
        vmax=1,
        cmap="Blues",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    # Significativité : points noirs
    if sig_da is not None:
        sig_arr = sig_da.values
        sig_masked = apply_shape_mask(
            sig_arr,
            lons,
            lats,
            cfg.shapefile_path,
        )

        lon2d, lat2d = np.meshgrid(lons, lats)

        sig_points = np.isfinite(sig_masked) & (sig_masked == 1)

        ax.scatter(
            lon2d[sig_points],
            lat2d[sig_points],
            s=2,
            c="black",
            marker=".",
            alpha=0.8,
        )

    # Limites
    ax.set_xlim(cfg.lon_min, cfg.lon_max)
    ax.set_ylim(cfg.lat_min, cfg.lat_max)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Stats
    finite = masked_arr[np.isfinite(masked_arr)]
    if finite.size > 0:
        stats_text = (
            f"min: {np.nanmin(finite):.3f}\n"
            f"mean: {np.nanmean(finite):.3f}\n"
            f"max: {np.nanmax(finite):.3f}"
        )
        ax.text(
            0.98,
            0.05,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
        )

    if sig_da is not None:
        ax.text(
            0.02,
            0.05,
            "Black dots: significant correlation",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
        )

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_one_corr_type(
    corr_type: str,
    cfg,
    exp_path: Path,
    selected_seasons: list[str],
    show: bool = False,
    robust: bool = False,
):
    data_dir, plot_dir = ensure_metric_dirs(exp_path, corr_type)

    print(f"\n=== Plotting {corr_type} ===")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")

    if corr_type == "corr_d":
        full_title = "daily deseasonalized correlation"
        map_title = "Daily deseasonalized correlation (CORR D)"
    else:
        full_title = "monthly correlation"
        map_title = "Monthly correlation (CORR M)"

    annual_arr = None
    annual_lons = None
    annual_lats = None

    # Annual map
    if "Annual" in selected_seasons:
        annual_path = resolve_corr_file(data_dir, corr_type, "Annual")

        if annual_path is not None:
            print(f"[STEP] Plotting annual {corr_type} map")

            corr_da, sig_da = load_corr_field(annual_path, corr_type)
            annual_lons, annual_lats = get_lat_lon(corr_da)
            annual_arr = corr_da.values

            plot_metric_map(
                arr=annual_arr,
                lons=annual_lons,
                lats=annual_lats,
                title=f"Annual {map_title}",
                fig_path=plot_dir / f"annual_{corr_type}_map.png",
                shapefile_path=cfg.shapefile_path,
                lon_min=cfg.lon_min,
                lon_max=cfg.lon_max,
                lat_min=cfg.lat_min,
                lat_max=cfg.lat_max,
                unit="Correlation",
                metric_type="correlation",
                n_bins=10,
                robust=robust,
                show=show,
            )

            plot_corr_with_significance(
                corr_da=corr_da,
                sig_da=sig_da,
                title=f"Annual {map_title}",
                fig_path=plot_dir / f"annual_{corr_type}_significance_map.png",
                cfg=cfg,
                show=show,
            )
        else:
            print(f"[WARNING] Annual {corr_type} file not found.")

    # Seasonal maps
    seasonal_arrays = []
    seasonal_labels = []
    seasonal_lons = None
    seasonal_lats = None

    for season in SEASON_ORDER:
        if season not in selected_seasons:
            continue

        nc_path = resolve_corr_file(data_dir, corr_type, season)

        if nc_path is None:
            print(f"[WARNING] Missing {corr_type} file for {season}")
            continue

        corr_da, sig_da = load_corr_field(nc_path, corr_type)
        lons, lats = get_lat_lon(corr_da)
        arr = corr_da.values

        seasonal_arrays.append(arr)
        seasonal_labels.append(season)

        if seasonal_lons is None:
            seasonal_lons = lons
            seasonal_lats = lats

        print(f"[STEP] Plotting {season} {corr_type} map")

        plot_metric_map(
            arr=arr,
            lons=lons,
            lats=lats,
            title=f"{SEASON_TITLES[season]} {full_title}",
            fig_path=plot_dir / f"{corr_type}_{season.lower()}_map.png",
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            unit="Correlation",
            metric_type="correlation",
            n_bins=10,
            robust=robust,
            show=show,
        )

        plot_corr_with_significance(
            corr_da=corr_da,
            sig_da=sig_da,
            title=f"{SEASON_TITLES[season]} {full_title}",
            fig_path=plot_dir / f"{corr_type}_{season.lower()}_significance_map.png",
            cfg=cfg,
            show=show,
        )

    # Seasonal panel
    if len(seasonal_arrays) == 4:
        print(f"[STEP] Plotting seasonal {corr_type} panel")

        plot_seasonal_bias_panel(
            seasonal_arrays=seasonal_arrays,
            lons=seasonal_lons,
            lats=seasonal_lats,
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            fig_path=plot_dir / f"seasonal_{corr_type}_panel.png",
            season_titles=[SEASON_TITLES[s] for s in seasonal_labels],
            title=f"Seasonal {full_title}",
            unit="Correlation",
            n_bins=10,
            robust=robust,
            show=show,
        )
    else:
        print(f"[WARNING] Seasonal {corr_type} panel skipped.")

    # Seasonal boxplot
    if len(seasonal_arrays) == 4:
        print(f"[STEP] Plotting seasonal {corr_type} boxplot")

        seasonal_arrays_masked = [
            apply_shape_mask(arr, seasonal_lons, seasonal_lats, cfg.shapefile_path)
            for arr in seasonal_arrays
        ]

        plot_seasonal_bias_boxplot(
            seasonal_arrays=seasonal_arrays_masked,
            labels=seasonal_labels,
            fig_path=plot_dir / f"seasonal_{corr_type}_boxplot.png",
            title=f"Seasonal {corr_type.upper()} distribution",
            ylabel="Correlation",
            show=show,
        )

    # Annual boxplot
    if annual_arr is not None:
        print(f"[STEP] Plotting annual {corr_type} boxplot")

        annual_arr_masked = apply_shape_mask(
            annual_arr,
            annual_lons,
            annual_lats,
            cfg.shapefile_path,
        )

        plot_annual_bias_boxplot(
            annual_array=annual_arr_masked,
            fig_path=plot_dir / f"annual_{corr_type}_boxplot.png",
            title=f"Annual {corr_type.upper()} distribution",
            ylabel="Correlation",
            show=show,
        )

    print(f"[SUCCESS] {corr_type} figures saved to: {plot_dir}")


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
        "seasons",
        ["Annual", "DJF", "MAM", "JJA", "SON"],
    )

    exp_path = Path(build_experiment_path(cfg))

    print("=== Temperature correlation plotting ===")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Experiment root : {exp_path}")
    print(f"Selected seasons: {selected_seasons}")

    plot_one_corr_type(
        corr_type="corr_d",
        cfg=cfg,
        exp_path=exp_path,
        selected_seasons=selected_seasons,
        show=args.show,
        robust=args.robust,
    )

    plot_one_corr_type(
        corr_type="corr_m",
        cfg=cfg,
        exp_path=exp_path,
        selected_seasons=selected_seasons,
        show=args.show,
        robust=args.robust,
    )

    print("\nAll correlation figures saved successfully.")


if __name__ == "__main__":
    main()