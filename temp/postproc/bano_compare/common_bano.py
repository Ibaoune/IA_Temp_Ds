from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import geopandas as gpd

from ...main.src.core.config import load_config
from ...main.src.core.utils import build_experiment_path

from ..common import (
    open_temperature_dataarray,
    subset_test_period,
    align_prediction_and_observation,
    convert_temperature_to_celsius,
    get_spatial_context,
    apply_spatial_context_to_inputs,
)
from ..map_utils import (
    get_bias_cmap,
    get_correlation_cmap,
    get_display_array,
    get_temperature_cmap,
    normalize_display_domain,
)


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
DEFAULT_CONFIG = THIS_FILE.with_name("config.yaml")
DEG_C = "\N{DEGREE SIGN}C"


# ==========================================================
# YAML / paths
# ==========================================================
def load_yaml(path: str | Path) -> dict:
    path = Path(path)
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


def get_main_cfg_and_bano_cfg(bano_cfg_path: str | Path):
    bano_cfg_path = Path(bano_cfg_path).resolve()
    bano_cfg = load_yaml(bano_cfg_path)

    main_cfg_path = resolve_from_project_root(
        bano_cfg["project"].get("main_config_path")
    )
    if main_cfg_path is None:
        raise ValueError("project.main_config_path must be provided in bano config.")

    cfg = load_config(train_mode=False, path=str(main_cfg_path))
    return bano_cfg, cfg, main_cfg_path


# ==========================================================
# Output folders
# ==========================================================
def ensure_bano_output_dir(cfg, folder_name: str = "bano_compare") -> Path:
    exp_path = Path(build_experiment_path(cfg))
    out_dir = exp_path / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ==========================================================
# Helpers
# ==========================================================
def flatten_valid(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def spatial_mean(arr) -> float:
    vals = flatten_valid(arr)
    return float(np.mean(vals)) if vals.size > 0 else np.nan


def get_lon_lat(da: xr.DataArray):
    if "lon" in da.coords:
        lons = da["lon"].values
    elif "longitude" in da.coords:
        lons = da["longitude"].values
    else:
        raise KeyError("Longitude coordinate not found.")

    if "lat" in da.coords:
        lats = da["lat"].values
    elif "latitude" in da.coords:
        lats = da["latitude"].values
    else:
        raise KeyError("Latitude coordinate not found.")

    return lons, lats


def load_project_shape(shapefile_path: str | Path) -> gpd.GeoDataFrame:
    shp = gpd.read_file(shapefile_path)
    if shp.crs is None:
        shp = shp.set_crs("EPSG:4326")
    else:
        shp = shp.to_crs("EPSG:4326")
    return shp

def get_shape_bounds(shapefile_path: str | Path):
    shp = load_project_shape(shapefile_path)
    xmin, ymin, xmax, ymax = shp.total_bounds
    return xmin, ymin, xmax, ymax


def draw_project_boundaries(ax, shapefile_path: str | Path) -> None:
    shp = load_project_shape(shapefile_path)
    shp.boundary.plot(ax=ax, color="0.5", linewidth=0.8, zorder=5)


def get_bano_display_domain(bano_cfg: dict) -> str:
    plot_cfg = bano_cfg.get("plot", {})
    return normalize_display_domain(
        plot_cfg.get("display_domain", "land"),
        default="land",
    )


def _colorbar_ticks(levels: np.ndarray, max_ticks: int = 7) -> np.ndarray:
    levels = np.asarray(levels, dtype=float)
    if levels.size <= max_ticks:
        return levels

    stride = int(np.ceil((levels.size - 1) / (max_ticks - 1)))
    ticks = list(levels[::stride])
    if not np.isclose(ticks[-1], levels[-1]):
        ticks.append(levels[-1])

    if levels[0] < 0 < levels[-1] and not any(np.isclose(t, 0.0) for t in ticks):
        ticks.append(0.0)
        ticks = sorted(ticks)

    return np.asarray(ticks, dtype=float)


def build_project_cmap_norm(
    vmin: float,
    vmax: float,
    color_kind: str = "temperature",
    n_bins: int = 9,
) -> tuple[np.ndarray, object, BoundaryNorm]:
    if vmax <= vmin:
        vmax = vmin + 1.0

    color_kind = (color_kind or "temperature").lower()
    n_bins = max(3, int(n_bins))

    if color_kind in {"bias", "difference", "diff"}:
        bound = max(abs(float(vmin)), abs(float(vmax)))
        if bound == 0:
            bound = 1.0
        if n_bins % 2:
            n_bins += 1
        levels = np.linspace(-bound, bound, n_bins + 1)
        cmap = get_bias_cmap(len(levels) - 1)
    elif color_kind == "correlation":
        levels = np.linspace(float(vmin), float(vmax), n_bins + 1)
        cmap = get_correlation_cmap(len(levels) - 1)
    else:
        levels = np.linspace(float(vmin), float(vmax), n_bins + 1)
        cmap = get_temperature_cmap(len(levels) - 1)

    norm = BoundaryNorm(levels, cmap.N, clip=True)
    return levels, cmap, norm


def add_project_colorbar(
    fig,
    im,
    ax,
    unit: str | None = None,
    label: str | None = None,
    max_ticks: int = 7,
    fraction: float = 0.046,
    pad: float = 0.02,
    cax=None,
):
    levels = getattr(im, "bano_levels", None)
    ticks = _colorbar_ticks(levels, max_ticks=max_ticks) if levels is not None else None

    colorbar_kwargs = dict(
        boundaries=levels,
        ticks=ticks,
        spacing="proportional",
    )
    if cax is None:
        colorbar_kwargs.update(ax=ax, fraction=fraction, pad=pad)
    else:
        colorbar_kwargs.update(cax=cax)

    cbar = fig.colorbar(im, **colorbar_kwargs)

    if label:
        cbar.set_label(label, fontsize=9, fontweight="semibold")
    elif unit:
        cbar.ax.set_title(unit, fontsize=10, fontweight="semibold", pad=5)

    cbar.ax.tick_params(labelsize=8, width=0.7, length=3)
    cbar.outline.set_linewidth(0.7)
    if hasattr(cbar, "solids") and cbar.solids is not None:
        cbar.solids.set_edgecolor("face")

    return cbar


def add_mean_text(ax, arr_stats, unit: str | None = None, fontsize: int = 13):
    mean_val = spatial_mean(arr_stats)
    if np.isfinite(mean_val):
        suffix = f" {unit}" if unit else ""
        ax.text(
            0.03,
            0.95,
            f"{mean_val:.2f}{suffix}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color="0.15",
            bbox=dict(
                facecolor="white",
                edgecolor="0.82",
                linewidth=0.45,
                alpha=0.82,
                boxstyle="round,pad=0.22",
            ),
            zorder=10,
        )

def plot_map_bano(
    ax,
    arr,
    lons,
    lats,
    shapefile_path: str | Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    stats_arr=None,
    unit: str | None = None,
    color_kind: str = "temperature",
    n_color_bins: int = 9,
    zoom_to_shape: bool = False,
    pad_frac: float = 0.04,
    display_domain: str = "morocco_shape",
    extent: tuple[float, float, float, float] | None = None,
):
    """
    Display either Morocco only or the already-computed land field, while computing
    the mean annotation from stats_arr (or from arr if stats_arr is None).
    """
    arr = np.asarray(arr, dtype=float)
    arr_stats = arr if stats_arr is None else np.asarray(stats_arr, dtype=float)
    display_domain = normalize_display_domain(display_domain)

    arr_display = get_display_array(
        arr=arr,
        lons=lons,
        lats=lats,
        shapefile_path=shapefile_path,
        display_domain=display_domain,
    )
    levels, project_cmap, norm = build_project_cmap_norm(
        vmin=vmin,
        vmax=vmax,
        color_kind=color_kind,
        n_bins=n_color_bins,
    )

    im = ax.pcolormesh(
        lons,
        lats,
        arr_display,
        shading="auto",
        cmap=project_cmap,
        norm=norm,
        zorder=1,
    )
    im.bano_levels = levels
    im.bano_unit = unit

    draw_project_boundaries(ax, shapefile_path)

    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    elif display_domain == "land":
        ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
        ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    elif zoom_to_shape:
        xmin, ymin, xmax, ymax = get_shape_bounds(shapefile_path)

        dx = xmax - xmin
        dy = ymax - ymin

        padx = pad_frac * dx
        pady = pad_frac * dy

        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.set_aspect("equal")

    ax.set_title(title, fontsize=17, fontweight="normal", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    add_mean_text(ax, arr_stats, unit=unit, fontsize=12)

    return im

# ==========================================================
# Observation loader for train/test
# ==========================================================
def get_observation_path(bano_cfg: dict, cfg) -> Path:
    obs_override = resolve_from_project_root(
        bano_cfg.get("data", {}).get("observation_path")
    )
    return obs_override if obs_override is not None else Path(cfg.target_path)


def load_masked_observation_train_test(
    bano_cfg: dict,
    cfg,
):
    data_cfg = bano_cfg.get("data", {})
    time_cfg = bano_cfg.get("time", {})

    obs_path = get_observation_path(bano_cfg, cfg)
    obs_var = data_cfg.get("observation_var", None)
    obs_units = data_cfg.get("observation_units", None)

    obs = open_temperature_dataarray(obs_path, var_name=obs_var)
    obs, unit_note = convert_temperature_to_celsius(obs, forced_unit=obs_units)

    spatial_ctx = get_spatial_context(
        metric_cfg=bano_cfg,
        cfg=cfg,
        project_root=PROJECT_ROOT,
    )

    # We apply the spatial domain before statistics/calculations
    dummy_pred = obs.copy()
    dummy_pred, obs_masked, spatial_mask = apply_spatial_context_to_inputs(
        pred=dummy_pred,
        obs=obs,
        spatial_ctx=spatial_ctx,
    )

    train_obs = subset_test_period(
        obs_masked,
        time_cfg.get("train_start"),
        time_cfg.get("train_end"),
    )

    test_obs = subset_test_period(
        obs_masked,
        time_cfg.get("test_start"),
        time_cfg.get("test_end"),
    )

    if train_obs.sizes.get("time", 0) == 0:
        raise ValueError("Train observation period is empty.")
    if test_obs.sizes.get("time", 0) == 0:
        raise ValueError("Test observation period is empty.")

    return train_obs, test_obs, spatial_ctx, spatial_mask, obs_path, unit_note

# ==========================================================
# Metric NetCDF readers
# ==========================================================

def find_metric_file(
    exp_path: str | Path,
    metric_name: str,
    eval_domain: str,
    patterns: list[str],
) -> Path:
    """
    Find a metric NetCDF file in:
      results/<experiment>/metrics/<metric_name>/data/<eval_domain>/

    Falls back to:
      results/<experiment>/metrics/<metric_name>/data/
    """
    exp_path = Path(exp_path)

    data_dir = exp_path / "metrics" / metric_name / "data" / eval_domain
    fallback_dir = exp_path / "metrics" / metric_name / "data"

    searched = []

    for base_dir in [data_dir, fallback_dir]:
        for pattern in patterns:
            candidate = base_dir / pattern
            searched.append(candidate)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Metric file not found. Tried:\n"
        + "\n".join(str(p) for p in searched)
    )


def infer_metric_variable(
    ds: xr.Dataset,
    preferred_names: list[str],
) -> str:
    """
    Infer the main variable in a metric NetCDF file.
    """
    for name in preferred_names:
        if name in ds.data_vars:
            return name

    candidates = [
        name for name, da in ds.data_vars.items()
        if "lat" in da.dims and "lon" in da.dims
        and name != "spatial_mask"
        and not name.endswith("_sig")
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        return candidates[0]

    raise KeyError(
        f"Unable to infer metric variable. Available variables: {list(ds.data_vars)}"
    )


def open_metric_field(
    exp_path: str | Path,
    metric_name: str,
    eval_domain: str,
    patterns: list[str],
    preferred_vars: list[str],
) -> tuple[xr.DataArray, Path, str]:
    """
    Open a metric field and return:
      DataArray, path, variable name

    If the field has a 'year' dimension, average over year.
    """
    path = find_metric_file(
        exp_path=exp_path,
        metric_name=metric_name,
        eval_domain=eval_domain,
        patterns=patterns,
    )

    ds = xr.open_dataset(path)
    var_name = infer_metric_variable(ds, preferred_vars)

    da = ds[var_name]

    if "year" in da.dims:
        da = da.mean(dim="year", skipna=True)

    return da, path, var_name

# ==========================================================
# Prediction loader
# ==========================================================

def resolve_prediction_path(bano_cfg: dict, cfg) -> Path:
    """
    Resolve prediction file path.

    If data.prediction_path is null, try standard output_data filenames.
    """
    pred_override = resolve_from_project_root(
        bano_cfg.get("data", {}).get("prediction_path")
    )

    if pred_override is not None:
        if not pred_override.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_override}")
        return pred_override

    exp_path = Path(build_experiment_path(cfg))
    out_dir = exp_path / "output_data"

    candidates = [
        out_dir / f"{cfg.model_type}_predictions_{cfg.experiment}.nc",
        out_dir / f"{cfg.model_type}_predictions_{cfg.src}_to_{cfg.target}.nc",
        out_dir / f"{cfg.model_type}_predictions_{cfg.target}.nc",
        out_dir / f"{cfg.model_type}_predictions_era5_to_{cfg.target}.nc",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No prediction file found. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_prediction_test(
    bano_cfg: dict,
    cfg,
    test_obs: xr.DataArray,
    spatial_mask: xr.DataArray,
) -> tuple[xr.DataArray, Path, str]:
    """
    Load prediction, convert units, subset test period, align with obs_test,
    and apply the same spatial mask.
    """
    data_cfg = bano_cfg.get("data", {})
    time_cfg = bano_cfg.get("time", {})

    pred_path = resolve_prediction_path(bano_cfg, cfg)

    pred_var = data_cfg.get("prediction_var", "air_temperature")
    pred_units = data_cfg.get("prediction_units", None)

    pred = open_temperature_dataarray(pred_path, var_name=pred_var)
    pred, unit_note = convert_temperature_to_celsius(
        pred,
        forced_unit=pred_units,
    )

    pred_test = subset_test_period(
        pred,
        time_cfg.get("test_start"),
        time_cfg.get("test_end"),
    )

    pred_test, test_obs_aligned = align_prediction_and_observation(
        pred_test,
        test_obs,
    )

    pred_test = pred_test.where(spatial_mask)
    test_obs_aligned = test_obs_aligned.where(spatial_mask)

    if pred_test.sizes.get("time", 0) == 0:
        raise ValueError("Prediction test period is empty after alignment.")

    return pred_test, pred_path, unit_note

