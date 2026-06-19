from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...main.src.core.utils import build_experiment_path

from .common_bano import (
    DEFAULT_CONFIG,
    DEG_C,
    get_main_cfg_and_bano_cfg,
    ensure_bano_output_dir,
    get_lon_lat,
    plot_map_bano,
    open_metric_field,
    get_bano_display_domain,
)


BANO_RDBU_9_R = [
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7",
    "#F4A582",
    "#D6604D",
    "#B2182B",
]

BANO_ORRD_9 = [
    "#FFF7EC",
    "#FEE8C8",
    "#FDD49E",
    "#FDBB84",
    "#FC8D59",
    "#EF6548",
    "#D7301F",
    "#B30000",
    "#7F0000",
]


def make_bano_colormaps():
    cmap_corr = LinearSegmentedColormap.from_list(
        "bano_OrRd",
        BANO_ORRD_9,
        N=256,
    )
    cmap_bias = LinearSegmentedColormap.from_list(
        "bano_RdBu_r",
        BANO_RDBU_9_R,
        N=256,
    )
    return cmap_corr, cmap_bias


CMAP_CORR, CMAP_BIAS = make_bano_colormaps()


def get_bano_norm_and_ticks(color_kind):
    if color_kind == "correlation":
        bounds = np.arange(0.85, 1.0001, 0.005)
        ticks = np.arange(0.86, 1.0001, 0.02)
        norm = BoundaryNorm(bounds, CMAP_CORR.N, clip=True)
        return CMAP_CORR, norm, ticks

    bounds = np.arange(-2.0, 2.0001, 0.1)
    ticks = np.arange(-2.0, 2.0001, 0.5)
    norm = BoundaryNorm(bounds, CMAP_BIAS.N, clip=True)
    return CMAP_BIAS, norm, ticks


def add_bano_colorbar(fig, im, ax, color_kind):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")

    _, _, ticks = get_bano_norm_and_ticks(color_kind)
    cbar.set_ticks(ticks)

    if color_kind == "correlation":
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        cbar.ax.set_title("")
    else:
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        cbar.ax.set_title(DEG_C, fontsize=9, fontweight="semibold", pad=3)

    cbar.ax.tick_params(labelsize=8, length=2, width=0.5)
    return cbar


# ==========================================================
# Baño-style fixed limits
# ==========================================================

CORR_VMIN, CORR_VMAX = 0.86, 1.00
BIAS_VMIN, BIAS_VMAX = -2.0, 2.0

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

        _, norm, _ = get_bano_norm_and_ticks(spec["color_kind"])

        im = plot_map_bano(
            ax=ax,
            arr=arr,
            stats_arr=arr,
            lons=lons,
            lats=lats,
            shapefile_path=shapefile_path,
            title=spec["title"],
            cmap=spec["cmap"],
            norm=norm,
            unit=spec["unit"],
            color_kind=spec["color_kind"],
            display_domain=display_domain,
        )

        ax.set_title(spec["title"], fontsize=11, fontweight="bold", pad=5)

        images.append((im, ax, spec))

    # Row label: UNet1
    axes[0].text(
        -0.18,
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
        add_bano_colorbar(fig, im, ax, spec["color_kind"])

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
