from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .common_bano import (
    DEFAULT_CONFIG,
    DEG_C,
    add_project_colorbar,
    get_main_cfg_and_bano_cfg,
    ensure_bano_output_dir,
    load_masked_observation_train_test,
    get_lon_lat,
    plot_map_bano,
    get_bano_display_domain,
)


# Fixed limits, close to Baño figure
CLIM_VMIN, CLIM_VMAX = -2.0, 14.0
P02_VMIN, P02_VMAX = -20.0, 10.0
P98_VMIN, P98_VMAX = 10.0, 30.0
DIFF_VMIN, DIFF_VMAX = -2.0, 2.0

CMAP_TEMP = "OrRd"
CMAP_DIFF = "RdBu_r"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce Baño-style Figure 1: climatology / P02 / P98 for TRAIN and TEST-TRAIN."
    )
    parser.add_argument(
        "bano_config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to bano_compare config.yaml",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure interactively.",
    )
    return parser.parse_args()


def compute_fields(train_obs, test_obs):
    train_clim = train_obs.mean(dim="time", skipna=True)
    train_p02 = train_obs.quantile(0.02, dim="time", skipna=True)
    train_p98 = train_obs.quantile(0.98, dim="time", skipna=True)

    test_clim = test_obs.mean(dim="time", skipna=True)
    test_p02 = test_obs.quantile(0.02, dim="time", skipna=True)
    test_p98 = test_obs.quantile(0.98, dim="time", skipna=True)

    diff_clim = test_clim - train_clim
    diff_p02 = test_p02 - train_p02
    diff_p98 = test_p98 - train_p98

    return {
        "train_clim": train_clim,
        "train_p02": train_p02,
        "train_p98": train_p98,
        "diff_clim": diff_clim,
        "diff_p02": diff_p02,
        "diff_p98": diff_p98,
    }


def main():
    args = parse_args()

    bano_cfg, cfg, main_cfg_path = get_main_cfg_and_bano_cfg(args.bano_config)
    out_dir = ensure_bano_output_dir(
        cfg,
        folder_name=bano_cfg.get("output", {}).get("folder_name", "bano_compare"),
    )
    dpi = int(bano_cfg.get("output", {}).get("dpi", 300))
    display_domain = get_bano_display_domain(bano_cfg)

    train_obs, test_obs, spatial_ctx, spatial_mask, obs_path, unit_note = \
        load_masked_observation_train_test(bano_cfg, cfg)

    fields = compute_fields(train_obs, test_obs)

    lons, lats = get_lon_lat(fields["train_clim"])
    shapefile_path = spatial_ctx.shapefile_path
    map_extent = (cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max)

    print("=== Baño-style Figure 1 ===")
    print(f"Main config       : {main_cfg_path}")
    print(f"Observation file  : {obs_path}")
    print(f"Spatial domain    : {spatial_ctx.eval_domain}")
    print(f"Display domain    : {display_domain}")
    print(f"Map extent        : lon=({cfg.lon_min}, {cfg.lon_max}), lat=({cfg.lat_min}, {cfg.lat_max})")
    print(f"Shapefile         : {shapefile_path}")
    print(f"Observation units : {unit_note}")
    print(f"Output dir        : {out_dir}")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4), dpi=dpi)

    # ---------------------------
    # Row 1: TRAIN
    # ---------------------------
    im00 = plot_map_bano(
        ax=axes[0, 0],
        arr=fields["train_clim"].values,
        stats_arr=fields["train_clim"].values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="Climatology",
        vmin=CLIM_VMIN,
        vmax=CLIM_VMAX,
        cmap=CMAP_TEMP,
        unit=DEG_C,
        color_kind="temperature",
        zoom_to_shape=True,
        pad_frac=0.02,
        display_domain=display_domain,
        extent=map_extent,
    )

    im01 = plot_map_bano(
        ax=axes[0, 1],
        arr=fields["train_p02"].values,
        stats_arr=fields["train_p02"].values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="P02",
        vmin=P02_VMIN,
        vmax=P02_VMAX,
        cmap=CMAP_TEMP,
        unit=DEG_C,
        color_kind="temperature",
        zoom_to_shape=True,
        pad_frac=0.02,
        display_domain=display_domain,
        extent=map_extent,
    )

    im02 = plot_map_bano(
        ax=axes[0, 2],
        arr=fields["train_p98"].values,
        stats_arr=fields["train_p98"].values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="P98",
        vmin=P98_VMIN,
        vmax=P98_VMAX,
        cmap=CMAP_TEMP,
        unit=DEG_C,
        color_kind="temperature",
        zoom_to_shape=True,
        pad_frac=0.02,
        display_domain=display_domain,
        extent=map_extent,
    )

    # ---------------------------
    # Row 2: TEST - TRAIN
    # ---------------------------
    im10 = plot_map_bano(
        ax=axes[1, 0],
        arr=fields["diff_clim"].values,
        stats_arr=fields["diff_clim"].values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="",
        vmin=DIFF_VMIN,
        vmax=DIFF_VMAX,
        cmap=CMAP_DIFF,
        unit=DEG_C,
        color_kind="bias",
        zoom_to_shape=True,
        pad_frac=0.02,
        display_domain=display_domain,
        extent=map_extent,
    )

    im11 = plot_map_bano(
        ax=axes[1, 1],
        arr=fields["diff_p02"].values,
        stats_arr=fields["diff_p02"].values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="",
        vmin=DIFF_VMIN,
        vmax=DIFF_VMAX,
        cmap=CMAP_DIFF,
        unit=DEG_C,
        color_kind="bias",
        zoom_to_shape=True,
        pad_frac=0.02,
        display_domain=display_domain,
        extent=map_extent,
    )

    im12 = plot_map_bano(
        ax=axes[1, 2],
        arr=fields["diff_p98"].values,
        stats_arr=fields["diff_p98"].values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="",
        vmin=DIFF_VMIN,
        vmax=DIFF_VMAX,
        cmap=CMAP_DIFF,
        unit=DEG_C,
        color_kind="bias",
        zoom_to_shape=True,
        pad_frac=0.02,
        display_domain=display_domain,
        extent=map_extent,
    )

    # ---------------------------
    # Colorbars
    # ---------------------------
    cbar00 = add_project_colorbar(fig, im00, axes[0, 0], unit=DEG_C)
    cbar00.ax.set_title("°C", fontsize=11, pad=6)
    cbar01 = add_project_colorbar(fig, im01, axes[0, 1], unit=DEG_C)
    cbar01.ax.set_title("°C", fontsize=11, pad=6)
    cbar02 = add_project_colorbar(fig, im02, axes[0, 2], unit=DEG_C)
    cbar02.ax.set_title("°C", fontsize=11, pad=6)

    cbar10 = add_project_colorbar(fig, im10, axes[1, 0], unit=DEG_C)
    cbar10.ax.set_title("°C", fontsize=11, pad=6)
    cbar11 = add_project_colorbar(fig, im11, axes[1, 1], unit=DEG_C)
    cbar11.ax.set_title("°C", fontsize=11, pad=6)
    cbar12 = add_project_colorbar(fig, im12, axes[1, 2], unit=DEG_C)
    cbar12.ax.set_title("°C", fontsize=11, pad=6)

    for cbar in [cbar00, cbar01, cbar02, cbar10, cbar11, cbar12]:
        cbar.ax.set_title(DEG_C, fontsize=10, fontweight="semibold", pad=5)

    # ---------------------------
    # Left-side labels
    # ---------------------------
    fig.text(
        0.015,
        0.52,
        "Mean temperature climatology",
        rotation=90,
        va="center",
        ha="center",
        fontsize=18,
    )
    fig.text(
        0.065,
        0.73,
        "TRAIN",
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
    )
    fig.text(
        0.065,
        0.28,
        "TEST-TRAIN",
        rotation=90,
        va="center",
        ha="center",
        fontsize=16,
    )

    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.07, wspace=0.18, hspace=0.12)

    out_path = out_dir / "fig1_climatology_test_train.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
