from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from .common_bano import (
    DEFAULT_CONFIG,
    add_project_colorbar,
    get_main_cfg_and_bano_cfg,
    ensure_bano_output_dir,
    load_masked_observation_train_test,
    load_prediction_test,
    get_lon_lat,
    plot_map_bano,
    get_bano_display_domain,
)


# ==========================================================
# Baño-style fixed limits
# ==========================================================

FREQ_VMIN, FREQ_VMAX = 0.0, 3.0
CMAP_FREQ = "OrRd"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce Baño-style Figure 4: frequency of exceedance of train P99."
    )
    parser.add_argument(
        "bano_config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to bano_compare config.yaml",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def exceedance_frequency(
    data_test: xr.DataArray,
    threshold_train: xr.DataArray,
) -> xr.DataArray:
    """
    Compute frequency (%) of days where data_test exceeds threshold_train.

    threshold_train is a 2D field: (lat, lon)
    data_test is a 3D field: (time, lat, lon)
    """
    valid = np.isfinite(data_test) & np.isfinite(threshold_train)

    exceed = xr.where(
        valid,
        data_test > threshold_train,
        np.nan,
    )

    freq = 100.0 * exceed.mean(dim="time", skipna=True)
    freq.name = "p99_exceedance_frequency"
    freq.attrs["units"] = "%"
    freq.attrs["description"] = (
        "Frequency of test-period days exceeding the train-period observed P99 threshold"
    )

    return freq


def main():
    args = parse_args()

    bano_cfg, cfg, main_cfg_path = get_main_cfg_and_bano_cfg(args.bano_config)

    out_dir = ensure_bano_output_dir(
        cfg,
        folder_name=bano_cfg.get("output", {}).get("folder_name", "bano_compare"),
    )
    dpi = int(bano_cfg.get("output", {}).get("dpi", 300))
    model_label = bano_cfg.get("experiment", {}).get("model_label", "UNet1")
    display_domain = get_bano_display_domain(bano_cfg)

    # -------------------------
    # Load obs train/test with spatial mask
    # -------------------------
    train_obs, test_obs, spatial_ctx, spatial_mask, obs_path, obs_unit_note = (
        load_masked_observation_train_test(bano_cfg, cfg)
    )

    # -------------------------
    # Load prediction test with the same mask
    # -------------------------
    pred_test, pred_path, pred_unit_note = load_prediction_test(
        bano_cfg=bano_cfg,
        cfg=cfg,
        test_obs=test_obs,
        spatial_mask=spatial_mask,
    )

    # Align obs test with prediction test one more time for safety
    pred_test, test_obs = xr.align(pred_test, test_obs, join="inner")

    # -------------------------
    # Compute P99 threshold from TRAIN observation
    # -------------------------
    p99_train = train_obs.quantile(
        0.99,
        dim="time",
        skipna=True,
    )

    if "quantile" in p99_train.dims:
        p99_train = p99_train.squeeze("quantile", drop=True)

    p99_train.name = "p99_train_observation"
    p99_train.attrs["units"] = "C"

    # -------------------------
    # Compute frequencies
    # -------------------------
    freq_test_obs = exceedance_frequency(
        data_test=test_obs,
        threshold_train=p99_train,
    )

    freq_unet = exceedance_frequency(
        data_test=pred_test,
        threshold_train=p99_train,
    )

    lons, lats = get_lon_lat(freq_test_obs)
    shapefile_path = spatial_ctx.shapefile_path

    print("=== Baño-style Figure 4: P99 exceedance frequency ===")
    print(f"Main config       : {main_cfg_path}")
    print(f"Observation file  : {obs_path}")
    print(f"Prediction file   : {pred_path}")
    print(f"Spatial domain    : {spatial_ctx.eval_domain}")
    print(f"Display domain    : {display_domain}")
    print(f"Shapefile         : {shapefile_path}")
    print(f"Observation units : {obs_unit_note}")
    print(f"Prediction units  : {pred_unit_note}")
    print(f"Output dir        : {out_dir}")
    print(f"Mean freq Test    : {float(freq_test_obs.mean(skipna=True).values):.3f} %")
    print(f"Mean freq {model_label}: {float(freq_unet.mean(skipna=True).values):.3f} %")

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.4),
        dpi=dpi,
    )

    im0 = plot_map_bano(
        ax=axes[0],
        arr=freq_test_obs.values,
        stats_arr=freq_test_obs.values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title="Test",
        vmin=FREQ_VMIN,
        vmax=FREQ_VMAX,
        cmap=CMAP_FREQ,
        unit="%",
        color_kind="temperature",
        display_domain=display_domain,
    )

    im1 = plot_map_bano(
        ax=axes[1],
        arr=freq_unet.values,
        stats_arr=freq_unet.values,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        title=model_label,
        vmin=FREQ_VMIN,
        vmax=FREQ_VMAX,
        cmap=CMAP_FREQ,
        unit="%",
        color_kind="temperature",
        display_domain=display_domain,
    )

    # Baño-style compact vertical colorbars
    cbar_ax = fig.add_axes([0.90, 0.20, 0.025, 0.56])
    cbar = add_project_colorbar(fig, im1, axes, unit="%", cax=cbar_ax)
    cbar.ax.set_title("%", fontsize=10, fontweight="semibold", pad=5)
    cbar.ax.tick_params(labelsize=8)

    fig.text(
        0.02,
        0.50,
        "Frequency of exceedance of TRAIN P99",
        rotation=90,
        va="center",
        ha="center",
        fontsize=13,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        top=0.84,
        bottom=0.12,
        wspace=0.22,
    )

    out_png = out_dir / "fig4_p99_frequency.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {out_png}")

    # Optional NetCDF output for traceability
    out_nc = out_dir / "fig4_p99_frequency_fields.nc"

    ds_out = xr.Dataset(
        {
            "p99_train_observation": p99_train,
            "freq_test_observation": freq_test_obs,
            "freq_unet": freq_unet,
        }
    )

    ds_out.attrs["description"] = (
        "Baño-style P99 exceedance frequency figure fields. "
        "Threshold is P99 of train-period observation."
    )
    ds_out.attrs["spatial_eval_domain"] = spatial_ctx.eval_domain
    ds_out.attrs["observation_file"] = str(obs_path)
    ds_out.attrs["prediction_file"] = str(pred_path)
    ds_out.attrs["frequency_units"] = "%"

    ds_out.to_netcdf(out_nc)
    print(f"[SAVED] {out_nc}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
