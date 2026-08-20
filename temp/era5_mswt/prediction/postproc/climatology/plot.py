from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm

from ..common import (
    PRED_CONFIG,
    apply_display_domain,
    get_configured_plot_extent,
    get_metric_dirs,
    load_prediction_settings,
    source_var_name,
)
from ...backend_paths import import_postproc_module

_map_utils = import_postproc_module("map_utils")
MapStyle = _map_utils.MapStyle
add_stats_box = _map_utils.add_stats_box
compute_sequential_levels = _map_utils.compute_sequential_levels
compute_symmetric_levels = _map_utils.compute_symmetric_levels
draw_project_boundaries = _map_utils.draw_project_boundaries
flatten_valid = _map_utils.flatten_valid
get_bias_cmap = _map_utils.get_bias_cmap
get_temperature_cmap = _map_utils.get_temperature_cmap
load_project_shape = _map_utils.load_project_shape
_get_lon_lat_2d = _map_utils._get_lon_lat_2d


MODEL_ORDER = ["MSWT", "ERA5_downscaled", "LMDZ_35", "LMDZ_250"]


def _display_arrays(data_arrays, settings):
    mask_cache = {}
    arrays = []
    labels = []
    lons = None
    lats = None

    for label, da in data_arrays:
        lons = da["lon"].values
        lats = da["lat"].values
        arr = apply_display_domain(
            da.values,
            lons,
            lats,
            settings.display_domain,
            settings.shapefile_path,
            land_shapefile_path=settings.land_shapefile_path,
            mask_cache=mask_cache,
        )
        arrays.append(arr)
        labels.append(label)

    return labels, arrays, lons, lats


def _plot_map_panel(
    data_arrays,
    fig_path: Path,
    settings,
    title: str,
    cbar_label: str,
    metric_type: str,
) -> None:
    if not data_arrays:
        print(f"[WARNING] No fields available for {fig_path.name}; skipping.")
        return

    style = MapStyle()
    shape_gdf = load_project_shape(settings.shapefile_path)
    labels, arrays, lons, lats = _display_arrays(data_arrays, settings)

    valid_arrays = [flatten_valid(arr) for arr in arrays]
    valid_arrays = [arr for arr in valid_arrays if arr.size > 0]
    if valid_arrays:
        merged = np.concatenate(valid_arrays)
    else:
        merged = np.array([-1.0, 0.0, 1.0]) if metric_type == "bias" else np.array([0.0, 1.0])

    if metric_type == "bias":
        levels = compute_symmetric_levels(merged, n_bins=11, robust=True)
        cmap = get_bias_cmap(len(levels) - 1)
    else:
        levels = compute_sequential_levels(merged, n_bins=9, robust=True, force_zero_min=False)
        cmap = get_temperature_cmap(len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N, clip=False)

    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))
    extent = get_configured_plot_extent(
        lons=lons,
        lats=lats,
        display_domain=settings.display_domain,
        lon_min=settings.lon_min,
        lon_max=settings.lon_max,
        lat_min=settings.lat_min,
        lat_max=settings.lat_max,
    )

    ncols = len(arrays)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6.3 * ncols, 6.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=style.dpi,
    )
    if ncols == 1:
        axes = [axes]

    for ax, label, arr in zip(axes, labels, arrays):
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

        ax.set_title(label, fontsize=style.panel_title_fontsize, fontweight="bold")
        add_stats_box(ax, arr, unit="C")

    cbar_ax = fig.add_axes([0.92, 0.17, 0.02, 0.66])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(cbar_label, fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    plt.suptitle(title, fontsize=style.title_fontsize + 3, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.08, wspace=0.08)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {fig_path}")


def main() -> None:
    settings = load_prediction_settings()
    data_dir, plot_dir = get_metric_dirs(settings, "climatology")
    bundle_path = data_dir / (
        f"climatology_bundle_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.nc"
    )

    print("=== Plotting climatology diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Input bundle      : {bundle_path}")
    print(f"Output plot dir   : {plot_dir}")

    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing climatology bundle: {bundle_path}")

    ds = xr.open_dataset(bundle_path)
    mean_fields = []
    bias_fields = []
    for label in MODEL_ORDER:
        key = source_var_name(label)
        mean_var = f"{key}_mean_period"
        bias_var = f"{key}_bias_mean_period"
        if mean_var in ds:
            mean_fields.append((label, ds[mean_var]))
        if bias_var in ds:
            bias_fields.append((f"{label} - MSWT", ds[bias_var]))

    print("Models found      :", [label for label, _ in mean_fields])

    _plot_map_panel(
        data_arrays=mean_fields,
        fig_path=plot_dir / "annual_mean_temperature_comparison_all_models.png",
        settings=settings,
        title=f"Annual mean temperature comparison ({settings.output_tag})",
        cbar_label="Temperature (C)",
        metric_type="temperature",
    )

    _plot_map_panel(
        data_arrays=bias_fields,
        fig_path=plot_dir / "annual_bias_comparison_all_models.png",
        settings=settings,
        title=f"Annual mean bias against MSWT ({settings.output_tag})",
        cbar_label="Bias (C)",
        metric_type="bias",
    )

    ds.close()


if __name__ == "__main__":
    main()
