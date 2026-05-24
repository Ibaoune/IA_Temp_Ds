from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ...main.src.core.utils import build_experiment_path

from .common_bano import (
    DEFAULT_CONFIG,
    DEG_C,
    add_project_colorbar,
    get_main_cfg_and_bano_cfg,
    ensure_bano_output_dir,
    get_lon_lat,
    plot_map_bano,
    open_metric_field,
    get_bano_display_domain,
)


# ==========================================================
# Baño-style fixed limits
# ==========================================================

CORR_VMIN, CORR_VMAX = 0.86, 1.00
BIAS_VMIN, BIAS_VMAX = -2.0, 2.0

CMAP_CORR = "Reds"
CMAP_BIAS = "RdBu_r"


FIG3_SPECS = [
    {
        "title": "Cor. deseasonal",
        "metric_name": "corr_d",
        "patterns": ["corr_d_annual_mean_period.nc"],
        "vars": ["corr_d"],
        "vmin": CORR_VMIN,
        "vmax": CORR_VMAX,
        "cmap": CMAP_CORR,
        "color_kind": "correlation",
        "unit": None,
    },
    {
        "title": "Bias",
        "metric_name": "bias",
        "patterns": [
            "bias_annual_daily_first_mean_period.nc",
            "bias_annual_mean_first_mean_period.nc",
            "bias_annual_mean_period.nc",
        ],
        "vars": ["bias"],
        "vmin": BIAS_VMIN,
        "vmax": BIAS_VMAX,
        "cmap": CMAP_BIAS,
        "color_kind": "bias",
        "unit": "°C",
    },
    {
        "title": "Bias P02",
        "metric_name": "b02",
        "patterns": ["b02_annual_mean_period.nc"],
        "vars": ["b02", "bias_p02", "bp02"],
        "vmin": BIAS_VMIN,
        "vmax": BIAS_VMAX,
        "cmap": CMAP_BIAS,
        "color_kind": "bias",
        "unit": "°C",
    },
    {
        "title": "Bias P98",
        "metric_name": "b98",
        "patterns": ["b98_annual_mean_period.nc"],
        "vars": ["b98", "bias_p98", "bp98"],
        "vmin": BIAS_VMIN,
        "vmax": BIAS_VMAX,
        "cmap": CMAP_BIAS,
        "color_kind": "bias",
        "unit": "°C",
    },
]

for spec in FIG3_SPECS[1:]:
    spec["unit"] = DEG_C


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce Baño-style Figure 3: UNet maps for CORR_D, Bias, Bias P02, Bias P98."
    )
    parser.add_argument(
        "bano_config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to bano_compare config.yaml",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    bano_cfg, cfg, main_cfg_path = get_main_cfg_and_bano_cfg(args.bano_config)

    exp_path = Path(build_experiment_path(cfg))
    out_dir = ensure_bano_output_dir(
        cfg,
        folder_name=bano_cfg.get("output", {}).get("folder_name", "bano_compare"),
    )
    dpi = int(bano_cfg.get("output", {}).get("dpi", 300))

    spatial_domain = bano_cfg.get("spatial", {}).get("eval_domain", "land")
    display_domain = get_bano_display_domain(bano_cfg)
    model_label = bano_cfg.get("experiment", {}).get("model_label", "UNet1")

    shapefile_path = (
        cfg.shapefile_path
        if bano_cfg.get("spatial", {}).get("shapefile_path") in (None, "", "null")
        else bano_cfg["spatial"]["shapefile_path"]
    )

    print("=== Baño-style Figure 3: UNet maps ===")
    print(f"Main config    : {main_cfg_path}")
    print(f"Experiment root: {exp_path}")
    print(f"Spatial domain : {spatial_domain}")
    print(f"Display domain : {display_domain}")
    print(f"Shapefile      : {shapefile_path}")
    print(f"Output dir     : {out_dir}")

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.8, 3.4),
        dpi=dpi,
    )

    images = []

    for ax, spec in zip(axes, FIG3_SPECS):
        da, path, var_name = open_metric_field(
            exp_path=exp_path,
            metric_name=spec["metric_name"],
            eval_domain=spatial_domain,
            patterns=spec["patterns"],
            preferred_vars=spec["vars"],
        )

        print(
            f"[OK] {spec['metric_name']}: {path.name} | variable={var_name}"
        )

        lons, lats = get_lon_lat(da)
        arr = da.values

        im = plot_map_bano(
            ax=ax,
            arr=arr,
            stats_arr=arr,
            lons=lons,
            lats=lats,
            shapefile_path=shapefile_path,
            title=spec["title"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
            cmap=spec["cmap"],
            unit=spec["unit"],
            color_kind=spec["color_kind"],
            display_domain=display_domain,
        )

        images.append((im, ax, spec))

    # Row label: UNet1
    axes[0].text(
        -0.25,
        0.50,
        model_label,
        transform=axes[0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="normal",
    )

    # Colorbars: one for each map, Baño-like compact vertical bars
    for im, ax, spec in images:
        cbar = add_project_colorbar(fig, im, ax, unit=spec["unit"])

        if spec["unit"] is not None:
            cbar.ax.set_title(
                spec["unit"],
                fontsize=10,
                fontweight="semibold",
                pad=5,
            )


    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.84,
        bottom=0.12,
        wspace=0.22,
    )

    out_path = out_dir / "fig3_unet_maps.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
