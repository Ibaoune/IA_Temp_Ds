from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import regionmask
import cartopy.io.shapereader as shpreader

# ==========================================================
# Style
# ==========================================================
@dataclass
class MapStyle:
    dpi: int = 220
    figsize_single: tuple = (6.8, 6.2)
    figsize_panel_1x4: tuple = (20, 5.6)
    figsize_boxplot: tuple = (10, 5.6)
    coast_linewidth: float = 0.8
    grid_linewidth: float = 0.35
    grid_alpha: float = 0.5
    title_fontsize: int = 12
    panel_title_fontsize: int = 11
    tick_labelsize: int = 9
    cbar_labelsize: int = 10
    stats_fontsize: int = 8


# ==========================================================
# Basic stats
# ==========================================================
def compute_field_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return {
            "n_valid": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    vals = arr[valid]
    return {
        "n_valid": int(valid.sum()),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
    }


def build_stats_box_text(arr: np.ndarray, unit: str | None = None) -> str:
    s = compute_field_stats(arr)
    suffix = f" {unit}" if unit else ""
    return (
        f"min:  {s['min']:.3f}{suffix}\n"
        f"mean: {s['mean']:.3f}{suffix}\n"
        f"max:  {s['max']:.3f}{suffix}"
    )


def build_distribution_text(arr: np.ndarray, unit: str | None = None) -> str:
    s = compute_field_stats(arr)
    suffix = f" {unit}" if unit else ""
    iqr = s["q75"] - s["q25"]
    return (
        f"N = {s['n_valid']}\n"
        f"median = {s['median']:.3f}{suffix}\n"
        f"IQR = {iqr:.3f}{suffix}\n"
        f"mean = {s['mean']:.3f}{suffix}"
    )


def flatten_valid(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr, dtype=float).ravel()
    return vals[np.isfinite(vals)]


# ==========================================================
# Levels
# ==========================================================
def _nice_step(value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        return 1.0

    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def compute_sequential_levels(
    data: np.ndarray,
    n_bins: int = 9,
    robust: bool = True,
    force_zero_min: bool = False,
) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.array([0.0, 1.0])

    if robust:
        data_min = float(np.nanpercentile(arr, 1))
        data_max = float(np.nanpercentile(arr, 99))
    else:
        data_min = float(np.nanmin(arr))
        data_max = float(np.nanmax(arr))

    if force_zero_min:
        data_min = 0.0

    if data_max <= data_min:
        data_max = data_min + 1.0

    raw_step = (data_max - data_min) / n_bins
    step = _nice_step(raw_step)

    lower = 0.0 if force_zero_min else math.floor(data_min / step) * step
    upper = math.ceil(data_max / step) * step

    levels = np.arange(lower, upper + step, step)

    if len(levels) < 2:
        levels = np.array([lower, upper + 1.0])

    return levels


def compute_symmetric_levels(
    data: np.ndarray,
    n_bins: int = 11,
    robust: bool = True,
) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.array([-1.0, 0.0, 1.0])

    if robust:
        vmax = float(np.nanpercentile(np.abs(arr), 99))
    else:
        vmax = float(np.nanmax(np.abs(arr)))

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    raw_step = (2.0 * vmax) / n_bins
    step = _nice_step(raw_step)
    bound = math.ceil(vmax / step) * step

    levels = np.arange(-bound, bound + step, step)

    if len(levels) < 3:
        levels = np.array([-bound, 0.0, bound])

    return levels


def compute_correlation_levels(step: float = 0.1, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    levels = np.arange(vmin, vmax + step, step)
    if levels[-1] < vmax:
        levels = np.append(levels, vmax)
    return levels


# ==========================================================
# Colormaps
# ==========================================================
def get_temperature_cmap(n_colors: int) -> mcolors.Colormap:
    colors = [
        "#313695",
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#e0f3f8",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#d73027",
    ]
    return mcolors.LinearSegmentedColormap.from_list("temp_metric", colors, N=n_colors)


def get_bias_cmap(n_colors: int) -> mcolors.Colormap:
    colors = [
        "#3b4cc0",
        "#6f92f3",
        "#aac7fd",
        "#dbe7f9",
        "#f7f7f7",
        "#f2d9c9",
        "#e7b89c",
        "#d67f6a",
        "#b40426",
    ]
    return mcolors.LinearSegmentedColormap.from_list("bias_metric", colors, N=n_colors)


def get_correlation_cmap(n_colors: int) -> mcolors.Colormap:
    colors = [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ]
    return mcolors.LinearSegmentedColormap.from_list("corr_metric", colors, N=n_colors)


# ==========================================================
# Shape / masking
# ==========================================================
def load_project_shape(shapefile_path: str | Path) -> gpd.GeoDataFrame:
    shp = gpd.read_file(shapefile_path)
    if shp.crs is None:
        shp = shp.set_crs("EPSG:4326")
    else:
        shp = shp.to_crs("EPSG:4326")
    return shp


def draw_project_boundaries(ax, shape_gdf: gpd.GeoDataFrame) -> None:
    shape_gdf.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=1.0,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )


def _get_lon_lat_2d(lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lons.ndim == 1 and lats.ndim == 1:
        return np.meshgrid(lons, lats)
    return lons, lats


def apply_shape_mask(
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    shapefile_path: str | Path,
) -> np.ndarray:
    """
    Apply a true shapefile mask to a 2D field.
    Expected:
    - arr shape = (lat, lon)
    - lons, lats are 1D coordinates
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"apply_shape_mask expects a 2D array, got shape={arr.shape}")

    shape_gdf = load_project_shape(shapefile_path).dissolve().reset_index(drop=True)
    shape_gdf["name"] = ["morocco"]

    da = xr.DataArray(
        arr,
        coords={"lat": np.asarray(lats), "lon": np.asarray(lons)},
        dims=("lat", "lon"),
    )

    mask_regions = regionmask.from_geopandas(
        shape_gdf,
        names="name",
        name="morocco",
    )

    mask = mask_regions.mask(da)
    da_masked = da.where(~mask.isnull())

    return da_masked.values

# ==========================================================
# Spatial evaluation masks
# ==========================================================

VALID_SPATIAL_DOMAINS = {"full_domain", "morocco_shape", "land"}
VALID_DISPLAY_DOMAINS = {"morocco_shape", "land"}


def normalize_display_domain(
    display_domain: str | None,
    default: str = "land",
) -> str:
    if display_domain in (None, "", "null"):
        display_domain = default

    display_domain = str(display_domain).strip()
    if display_domain not in VALID_DISPLAY_DOMAINS:
        raise ValueError(
            f"plot.display_domain must be one of {sorted(VALID_DISPLAY_DOMAINS)}, "
            f"got {display_domain!r}"
        )

    return display_domain


def get_display_array(
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    shapefile_path: str | Path,
    display_domain: str,
) -> np.ndarray:
    display_domain = normalize_display_domain(display_domain)

    if display_domain == "morocco_shape":
        return apply_shape_mask(
            arr=arr,
            lons=lons,
            lats=lats,
            shapefile_path=shapefile_path,
        )

    return np.asarray(arr, dtype=float)


def get_plot_extent(
    lons: np.ndarray,
    lats: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    display_domain: str,
) -> list[float]:
    display_domain = normalize_display_domain(display_domain)

    if display_domain == "land":
        return [
            float(np.nanmin(lons)),
            float(np.nanmax(lons)),
            float(np.nanmin(lats)),
            float(np.nanmax(lats)),
        ]

    return [lon_min, lon_max, lat_min, lat_max]


def _template_2d_from_dataarray(da: xr.DataArray) -> xr.DataArray:
    """
    Convert a DataArray to a 2D lat/lon template.

    Works for:
      - (lat, lon)
      - (time, lat, lon)
      - (year, lat, lon)
    """
    out = da

    for dim in list(out.dims):
        if dim not in {"lat", "lon"}:
            out = out.isel({dim: 0})

    if "lat" not in out.dims or "lon" not in out.dims:
        raise ValueError(f"Expected lat/lon dimensions, got dims={out.dims}")

    return out


def build_spatial_mask(
    da: xr.DataArray,
    eval_domain: str,
    shapefile_path: str | Path | None = None,
    land_shapefile_path: str | Path | None = None,
) -> xr.DataArray:
    """
    Build a 2D boolean spatial mask.

    eval_domain:
      - full_domain   : all grid cells
      - morocco_shape : cells inside Morocco shapefile
      - land          : land cells inside the current lat/lon domain
    """
    if eval_domain not in VALID_SPATIAL_DOMAINS:
        raise ValueError(
            f"Unknown eval_domain={eval_domain!r}. "
            f"Choose one of {sorted(VALID_SPATIAL_DOMAINS)}"
        )

    template = _template_2d_from_dataarray(da)

    if eval_domain == "full_domain":
        mask = xr.full_like(template, True, dtype=bool)
        mask.name = "spatial_mask"
        mask.attrs["eval_domain"] = "full_domain"
        return mask

    if eval_domain == "morocco_shape":
        if shapefile_path is None:
            raise ValueError(
                "shapefile_path is required when eval_domain='morocco_shape'"
            )

        shape_gdf = load_project_shape(shapefile_path).dissolve().reset_index(drop=True)
        shape_gdf["name"] = ["morocco"]

        regions = regionmask.from_geopandas(
            shape_gdf,
            names="name",
            name="morocco",
        )

        mask = regions.mask(template)
        mask_bool = xr.where(~mask.isnull(), True, False)
        mask_bool.name = "spatial_mask"
        mask_bool.attrs["eval_domain"] = "morocco_shape"
        return mask_bool

    if eval_domain == "land":
        if land_shapefile_path in (None, "", "null"):
            land_path = shpreader.natural_earth(
                resolution="10m",
                category="physical",
                name="land",
            )
        else:
            land_path = land_shapefile_path

        land_gdf = gpd.read_file(land_path)

        if land_gdf.crs is None:
            land_gdf = land_gdf.set_crs("EPSG:4326")
        else:
            land_gdf = land_gdf.to_crs("EPSG:4326")

        land_gdf = land_gdf.dissolve().reset_index(drop=True)
        land_gdf["name"] = ["land"]

        regions = regionmask.from_geopandas(
            land_gdf,
            names="name",
            name="land",
        )

        mask = regions.mask(template)
        mask_bool = xr.where(~mask.isnull(), True, False)
        mask_bool.name = "spatial_mask"
        mask_bool.attrs["eval_domain"] = "land"
        return mask_bool

    raise RuntimeError("Unreachable spatial mask case.")

# ==========================================================
# Annotation helpers
# ==========================================================
def add_stats_box(ax, arr: np.ndarray, unit: str | None = None) -> None:
    txt = build_stats_box_text(arr, unit=unit if unit else None)
    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.6",
            alpha=0.82,
        ),
        zorder=6,
    )


def add_distribution_box(ax, arr: np.ndarray, unit: str | None = None) -> None:
    txt = build_distribution_text(arr, unit=unit if unit else None)
    ax.text(
        0.98,
        0.97,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.6",
            alpha=0.82,
        ),
    )


# ==========================================================
# Map plots
# ==========================================================
def plot_metric_map(
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    title: str,
    fig_path: str | Path,
    shapefile_path: str | Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    unit: str = "",
    metric_type: str = "bias",
    n_bins: int = 11,
    robust: bool = True,
    show: bool = False,
    apply_mask_in_plot: bool = True,
    display_domain: str | None = None,
    stats_arr: np.ndarray | None = None,
    stats_label: str | None = None,
    style: MapStyle = MapStyle(),
) -> Path:
    """
    Generic temperature metric map plotter.

    metric_type:
        - "bias"
        - "temperature"
        - "rmse"
        - "correlation"
    """
    arr = np.asarray(arr, dtype=float)

    # Array used for statistics and color levels.
    # This must remain the compute-domain field, e.g. land/full_domain/morocco_shape.
    if stats_arr is None:
        arr_stats = arr
    else:
        arr_stats = np.asarray(stats_arr, dtype=float)

    # Array used only for visual display.
    # Statistics still come from arr_stats.
    if display_domain is None:
        display_domain = "morocco_shape" if apply_mask_in_plot else "land"
    display_domain = normalize_display_domain(display_domain)
    arr_display = get_display_array(
        arr=arr,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        display_domain=display_domain,
    )
    extent = get_plot_extent(
        lons=lons,
        lats=lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        display_domain=display_domain,
    )

    lon2d, lat2d = _get_lon_lat_2d(np.asarray(lons), np.asarray(lats))

    if metric_type == "bias":
        levels = compute_symmetric_levels(arr_stats, n_bins=n_bins, robust=robust)
        cmap = get_bias_cmap(len(levels) - 1)
    elif metric_type in {"temperature", "rmse"}:
        force_zero_min = (metric_type == "rmse")
        levels = compute_sequential_levels(
            arr_stats,
            n_bins=n_bins,
            robust=robust,
            force_zero_min=force_zero_min,
        )
        cmap = get_temperature_cmap(len(levels) - 1)
    elif metric_type == "correlation":
        levels = compute_correlation_levels()
        cmap = get_correlation_cmap(len(levels) - 1)
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    norm = BoundaryNorm(levels, cmap.N, clip=True)
    shape_gdf = load_project_shape(shapefile_path)

    fig, ax = plt.subplots(
        figsize=style.figsize_single,
        subplot_kw={"projection": ccrs.PlateCarree()},
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

    cbar = plt.colorbar(
        im,
        ax=ax,
        boundaries=levels,
        ticks=levels,
        spacing="proportional",
        shrink=0.88,
        pad=0.03,
    )
    cbar.set_label(unit, fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    ax.set_title(title, fontsize=style.title_fontsize, pad=10)
    add_stats_box(ax, arr_stats, unit=unit if unit else None)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


def plot_seasonal_bias_panel(
    seasonal_arrays: list[np.ndarray],
    lons: np.ndarray,
    lats: np.ndarray,
    shapefile_path: str | Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    fig_path: str | Path,
    season_titles: list[str] | None = None,
    title: str = "Seasonal temperature bias",
    unit: str = "°C",
    n_bins: int = 11,
    robust: bool = True,
    show: bool = False,
    apply_mask_in_plot: bool = True,
    display_domain: str | None = None,
    style: MapStyle = MapStyle(),
) -> Path:
    """
    Plot a single 1x4 seasonal bias figure:
    DJF / MAM / JJA / SON
    """
    if len(seasonal_arrays) != 4:
        raise ValueError("plot_seasonal_bias_panel expects exactly 4 seasonal arrays.")

    if season_titles is None:
        season_titles = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]

    stats_arrays = [
        np.asarray(arr, dtype=float)
        for arr in seasonal_arrays
    ]

    if display_domain is None:
        display_domain = "morocco_shape" if apply_mask_in_plot else "land"
    display_domain = normalize_display_domain(display_domain)
    display_arrays = [
        get_display_array(arr, lons, lats, shapefile_path, display_domain)
        for arr in stats_arrays
    ]
    extent = get_plot_extent(
        lons=lons,
        lats=lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        display_domain=display_domain,
    )
    merged_valid = []

    for arr in stats_arrays:
        vals = flatten_valid(arr)
        if vals.size > 0:
            merged_valid.append(vals)

    if not merged_valid:
        merged = np.array([-1.0, 0.0, 1.0])
    else:
        merged = np.concatenate(merged_valid)

    levels = compute_symmetric_levels(merged, n_bins=n_bins, robust=robust)
    cmap = get_bias_cmap(len(levels) - 1)
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

        ax.set_title(season_titles[i], fontsize=style.panel_title_fontsize, fontweight="bold")
        add_stats_box(ax, arr_stats, unit=unit)

    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label(f"Bias ({unit})", fontsize=style.cbar_labelsize)
    cbar.ax.tick_params(labelsize=style.tick_labelsize)

    tick_positions = levels if len(levels) <= 12 else levels[::2]
    cbar.set_ticks(tick_positions)

    plt.suptitle(title, fontsize=style.title_fontsize + 3, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.08, wspace=0.10)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


# ==========================================================
# Boxplots
# ==========================================================

def plot_seasonal_bias_boxplot(
    seasonal_arrays: list[np.ndarray],
    labels: list[str],
    fig_path: str | Path,
    title: str = "Seasonal bias distribution",
    ylabel: str = "Bias (°C)",
    show: bool = False,
    style: MapStyle = MapStyle(),
) -> Path:
    data = [flatten_valid(arr) for arr in seasonal_arrays]

    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"]
    edge_color = "#4F4F4F"

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=style.dpi)

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.45,
        showmeans=False,
        showfliers=False,
        whis=(0, 100),  # full range
        whiskerprops=dict(color=edge_color, linewidth=1.3),
        capprops=dict(color=edge_color, linewidth=1.3),
        medianprops=dict(color=edge_color, linewidth=1.4),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set(
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=0.95,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title(title, fontsize=style.title_fontsize + 4, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)

    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.28)

    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=style.dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    plt.close(fig)

    return fig_path


def plot_annual_bias_boxplot(
    annual_array: np.ndarray,
    fig_path: str | Path,
    title: str = "Annual bias distribution",
    ylabel: str = "Bias (°C)",
    show: bool = False,
    style: MapStyle = MapStyle(),
) -> Path:
    data = flatten_valid(annual_array)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=style.dpi)

    box_color = "#D68C78"
    edge_color = "#4F4F4F"

    ax.boxplot(
        [data],
        labels=["Annual"],
        patch_artist=True,
        widths=0.32,
        showmeans=False,
        showfliers=False,
        whis=(0, 100),  # full range
        boxprops=dict(
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=1.6,
            alpha=0.95,
        ),
        whiskerprops=dict(color=edge_color, linewidth=1.4),
        capprops=dict(color=edge_color, linewidth=1.4),
        medianprops=dict(color=edge_color, linewidth=1.4),
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.6, alpha=0.8)

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
