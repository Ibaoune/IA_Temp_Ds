from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr
import yaml

from ....main.src.core.config import load_config
from ....main.src.core.utils import build_experiment_path
from ...common import (
    ensure_spatial_metric_dirs,
    get_spatial_context,
)
from ...map_utils import (
    plot_metric_map,
    plot_seasonal_bias_panel,
    plot_seasonal_bias_boxplot,
    plot_annual_bias_boxplot,
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
        description="Plot temperature bias diagnostics from computed bias NetCDF files."
    )
    parser.add_argument(
        "metric_config",
        nargs="?",
        default=str(DEFAULT_METRIC_CONFIG),
        help="Path to bias metric config.yaml",
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
# File resolvers
# =========================================================
def resolve_bias_file(
    data_dir: Path,
    season: str,
    strategy: str,
    return_by_year: bool,
) -> Path | None:
    mode_suffix = "by_year" if return_by_year else "mean_period"
    path = data_dir / f"bias_{season.lower()}_{strategy}_{mode_suffix}.nc"
    return path if path.exists() else None


def load_bias_field(nc_path: Path):
    ds = xr.open_dataset(nc_path)

    if "bias" not in ds.data_vars:
        raise KeyError(f"'bias' variable not found in {nc_path}")

    da = ds["bias"]

    if "year" in da.dims:
        da = da.mean(dim="year", skipna=True)

    return da.values, da["lon"].values, da["lat"].values


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

    metric_name = metric_cfg["metric"].get("name", "bias")
    strategy = metric_cfg["metric"].get("strategy", "daily_first")
    return_by_year = bool(metric_cfg["metric"].get("return_by_year", False))
    selected_seasons = metric_cfg["metric"].get(
        "seasons", ["Annual", "DJF", "MAM", "JJA", "SON"]
    )

    exp_path = Path(build_experiment_path(cfg))
    spatial_ctx = get_spatial_context(
        metric_cfg=metric_cfg,
        cfg=cfg,
        project_root=PROJECT_ROOT,
    )
    data_dir, plot_dir = ensure_spatial_metric_dirs(
        exp_path=exp_path,
        metric_name=metric_name,
        eval_domain=spatial_ctx.eval_domain,
    )

    print("=== Temperature bias plotting ===")
    print(f"Spatial domain  : {spatial_ctx.eval_domain}")
    print(f"Metric config   : {metric_cfg_path}")
    print(f"Main config     : {main_cfg_path}")
    print(f"Input data dir  : {data_dir}")
    print(f"Output plot dir : {plot_dir}")
    print(f"Strategy        : {strategy}")
    print(f"Return by year  : {return_by_year}")

    # -------------------------------------------------
    # 1) Annual map
    # -------------------------------------------------
    annual_arr = None
    annual_lons = None
    annual_lats = None

    if "Annual" in selected_seasons:
        annual_path = resolve_bias_file(
            data_dir=data_dir,
            season="Annual",
            strategy=strategy,
            return_by_year=return_by_year,
        )

        if annual_path is not None:
            print("[STEP] Plotting annual bias map")
            annual_arr, annual_lons, annual_lats = load_bias_field(annual_path)

            plot_metric_map(
                arr=annual_arr,
                lons=annual_lons,
                lats=annual_lats,
                title=f"Annual temperature bias ({cfg.src.upper()} → {cfg.target.upper()}, {cfg.model_type.upper()})",
                fig_path=plot_dir / f"annual_bias_map_{strategy}.png",
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
                apply_mask_in_plot=True,
                stats_arr=annual_arr,
            )
        else:
            print("[WARNING] Annual bias file not found.")

    # -------------------------------------------------
    # 2) Individual seasonal maps
    # -------------------------------------------------
    seasonal_arrays = []
    seasonal_labels = []
    seasonal_lons = None
    seasonal_lats = None

    for season in SEASON_ORDER:
        if season not in selected_seasons:
            continue

        nc_path = resolve_bias_file(
            data_dir=data_dir,
            season=season,
            strategy=strategy,
            return_by_year=return_by_year,
        )

        if nc_path is None:
            print(f"[WARNING] Missing seasonal file for {season}")
            continue

        arr, lons, lats = load_bias_field(nc_path)

        seasonal_arrays.append(arr)
        seasonal_labels.append(season)

        if seasonal_lons is None:
            seasonal_lons = lons
            seasonal_lats = lats

        print(f"[STEP] Plotting {season} bias map")
        plot_metric_map(
            arr=arr,
            lons=lons,
            lats=lats,
            title=f"{SEASON_TITLES[season]} temperature bias ({cfg.src.upper()} → {cfg.target.upper()}, {cfg.model_type.upper()})",
            fig_path=plot_dir / f"bias_{season.lower()}_map_{strategy}.png",
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
            apply_mask_in_plot=True,
            stats_arr=arr,
        )

    # -------------------------------------------------
    # 3) Grouped seasonal panel (1x4)
    # -------------------------------------------------
    if len(seasonal_arrays) == 4:
        print("[STEP] Plotting grouped seasonal bias panel")

        plot_seasonal_bias_panel(
            seasonal_arrays=seasonal_arrays,
            lons=seasonal_lons,
            lats=seasonal_lats,
            shapefile_path=cfg.shapefile_path,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            fig_path=plot_dir / f"seasonal_bias_panel_{strategy}.png",
            season_titles=[SEASON_TITLES[s] for s in seasonal_labels],
            title=f"Seasonal temperature bias ({cfg.src.upper()} → {cfg.target.upper()}, {cfg.model_type.upper()})",
            unit="°C",
            n_bins=11,
            robust=args.robust,
            show=args.show,
            apply_mask_in_plot=True,   
        )
    else:
        print("[WARNING] Seasonal panel not created: one or more seasonal files are missing.")

    # -------------------------------------------------
    # 4) Seasonal boxplot (normal boxplot only)
    # -------------------------------------------------
    if len(seasonal_arrays) == 4:
        print("[STEP] Plotting seasonal bias boxplot")

        seasonal_arrays_masked = seasonal_arrays

        plot_seasonal_bias_boxplot(
            seasonal_arrays=seasonal_arrays_masked,
            labels=seasonal_labels,
            fig_path=plot_dir / f"seasonal_bias_boxplot_{strategy}.png",
            title="Seasonal bias distribution",
            ylabel="Bias (°C)",
            show=args.show,
        )
    else:
        print("[WARNING] Seasonal boxplot skipped: seasonal arrays incomplete.")

    # -------------------------------------------------
    # 5) Annual boxplot (normal boxplot only)
    # -------------------------------------------------
    if annual_arr is not None:
        print("[STEP] Plotting annual bias boxplot")

        annual_arr_masked = annual_arr

        plot_annual_bias_boxplot(
            annual_array=annual_arr_masked,
            fig_path=plot_dir / f"annual_bias_boxplot_{strategy}.png",
            title="Annual bias distribution",
            ylabel="Bias (°C)",
            show=args.show,
        )
    else:
        print("[WARNING] Annual boxplot skipped: annual array missing.")

    print(f"[SUCCESS] Bias figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()