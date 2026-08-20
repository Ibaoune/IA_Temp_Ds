from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ..common import (
    PRED_CONFIG,
    SELECTED_LABELS,
    apply_display_domain,
    get_configured_plot_extent,
    get_metric_dirs,
    load_metric_bundles,
    load_prediction_settings,
    make_plot_name,
)
from ...backend_paths import import_postproc_module

_map_utils = import_postproc_module("map_utils")
MapStyle = _map_utils.MapStyle
add_stats_box = _map_utils.add_stats_box
draw_project_boundaries = _map_utils.draw_project_boundaries
flatten_valid = _map_utils.flatten_valid
load_project_shape = _map_utils.load_project_shape
_get_lon_lat_2d = _map_utils._get_lon_lat_2d


def plot_corr_map_panel(
    bundles,
    fig_path: Path,
    title_tag: str,
    display_domain: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    shapefile_path: Path,
    land_shapefile_path: Path | None = None,
    show_significance: bool = False,
    show: bool = False,
) -> None:
    style = MapStyle()
    shape_gdf = load_project_shape(shapefile_path)
    mask_cache = {}

    arrays = []
    sig_arrays = []
    labels = []
    lons = None
    lats = None

    for label, ds in bundles:
        corr = ds["corr_d"].values
        sig = ds["corr_d_sig"].values if "corr_d_sig" in ds else None
        lons = ds["corr_d"]["lon"].values
        lats = ds["corr_d"]["lat"].values

        arrays.append(
            apply_display_domain(
                corr,
                lons,
                lats,
                display_domain,
                shapefile_path,
                land_shapefile_path=land_shapefile_path,
                mask_cache=mask_cache,
            )
        )

        if sig is None:
            sig_arrays.append(None)
        else:
            sig_arrays.append(
                apply_display_domain(
                    sig,
                    lons,
                    lats,
                    display_domain,
                    shapefile_path,
                    land_shapefile_path=land_shapefile_path,
                    mask_cache=mask_cache,
                )
            )

        labels.append(label)

    levels = np.linspace(-1.0, 1.0, 11)
    cmap = plt.get_cmap("RdBu_r", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))
    extent = get_configured_plot_extent(
        lons=lons,
        lats=lats,
        display_domain=display_domain,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )

    ncols = len(arrays)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(7.0 * ncols, 6.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )
    if ncols == 1:
        axes = [axes]

    for ax, arr, sig_arr, title in zip(axes, arrays, sig_arrays, labels):
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

        if show_significance and sig_arr is not None:
            sig_points = np.isfinite(sig_arr) & (sig_arr == 1)
            ax.scatter(
                lon2d[sig_points],
                lat2d[sig_points],
                s=2,
                c="black",
                marker=".",
                alpha=0.65,
                transform=ccrs.PlateCarree(),
                zorder=3,
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

        ax.set_title(title, fontsize=style.panel_title_fontsize, fontweight="bold")
        add_stats_box(ax, arr, unit="")

    cbar_ax = fig.add_axes([0.92, 0.17, 0.02, 0.66])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Correlation", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    if show_significance:
        fig.text(
            0.04,
            0.03,
            "Black dots: statistically significant correlation (p < 0.05)",
            fontsize=11,
        )

    title = "Bano-style daily anomaly correlation"
    if show_significance:
        title = "Bano-style daily anomaly correlation with significance"
    plt.suptitle(
        f"{title} ({title_tag})",
        fontsize=style.title_fontsize + 3,
        fontweight="bold",
        y=0.98,
    )

    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.08, wspace=0.08)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)


def plot_corr_boxplot(
    bundles,
    fig_path: Path,
    title_tag: str,
    display_domain: str,
    shapefile_path: Path,
    land_shapefile_path: Path | None = None,
    show: bool = False,
) -> None:
    labels = []
    values = []
    means = []
    mask_cache = {}

    for label, ds in bundles:
        arr = ds["corr_d"].values
        lons = ds["corr_d"]["lon"].values
        lats = ds["corr_d"]["lat"].values
        arr_display = apply_display_domain(
            arr,
            lons,
            lats,
            display_domain,
            shapefile_path,
            land_shapefile_path=land_shapefile_path,
            mask_cache=mask_cache,
        )
        flat = flatten_valid(arr_display)

        labels.append(label)
        values.append(flat)
        means.append(float(np.nanmean(flat)) if flat.size > 0 else np.nan)

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)
    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=False, widths=0.5)

    colors = ["#d94b59", "#4f7e9d", "#6a994e", "#9368B7"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("#3a3a3a")
        median.set_linewidth(1.8)

    x = np.arange(1, len(labels) + 1)
    ax.scatter(x, means, marker="D", s=90, color="#8B0000", zorder=5, label="Mean")
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_title(
        f"Distribution analysis: daily anomaly correlation ({title_tag})",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Model configuration", fontsize=16)
    ax.set_ylabel("Correlation", fontsize=16)
    ax.set_ylim(-1, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=13)
    ax.legend()

    fig.savefig(fig_path, dpi=160, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    settings = load_prediction_settings()
    data_dir, plot_dir = get_metric_dirs(settings, "correlation")
    bundles = load_metric_bundles(data_dir, settings, "corr_d_bundle.nc", SELECTED_LABELS)

    print("=== Plotting Bano-style anomaly correlation diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Main config       : {settings.main_config_path}")
    print(f"Plot extent       : {settings.configured_extent}")
    print(f"Input data dir    : {data_dir}")
    print(f"Output plot dir   : {plot_dir}")
    print("Models found      :", [label for label, _ in bundles])

    plot_corr_map_panel(
        bundles=bundles,
        fig_path=plot_dir
        / make_plot_name(
            "annual_corr_d_anomaly_distribution_map",
            settings.selected_start,
            settings.selected_end,
            settings.output_tag,
        ),
        title_tag=settings.output_tag,
        display_domain=settings.display_domain,
        lon_min=settings.lon_min,
        lon_max=settings.lon_max,
        lat_min=settings.lat_min,
        lat_max=settings.lat_max,
        shapefile_path=settings.shapefile_path,
        land_shapefile_path=settings.land_shapefile_path,
        show_significance=False,
        show=False,
    )

    plot_corr_map_panel(
        bundles=bundles,
        fig_path=plot_dir
        / make_plot_name(
            "annual_corr_d_anomaly_map",
            settings.selected_start,
            settings.selected_end,
            settings.output_tag,
        ),
        title_tag=settings.output_tag,
        display_domain=settings.display_domain,
        lon_min=settings.lon_min,
        lon_max=settings.lon_max,
        lat_min=settings.lat_min,
        lat_max=settings.lat_max,
        shapefile_path=settings.shapefile_path,
        land_shapefile_path=settings.land_shapefile_path,
        show_significance=True,
        show=False,
    )

    plot_corr_boxplot(
        bundles=bundles,
        fig_path=plot_dir
        / make_plot_name(
            "annual_corr_d_anomaly_boxplot",
            settings.selected_start,
            settings.selected_end,
            settings.output_tag,
        ),
        title_tag=settings.output_tag,
        display_domain=settings.display_domain,
        shapefile_path=settings.shapefile_path,
        land_shapefile_path=settings.land_shapefile_path,
        show=False,
    )

    print(f"[SUCCESS] Correlation figures saved to: {plot_dir}")


if __name__ == "__main__":
    main()
