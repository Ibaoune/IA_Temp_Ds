from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ..backend_paths import CONFIGS_DIR, ERA5_MSWT_ROOT, PREDICTION_DIR, WORKSPACE_ROOT, import_main_module, import_postproc_module

_map_utils = import_postproc_module("map_utils")
_postproc_common = import_postproc_module("common")
_core_config = import_main_module("src.core.config")

VALID_SPATIAL_DOMAINS = _map_utils.VALID_SPATIAL_DOMAINS
build_spatial_mask = _map_utils.build_spatial_mask
_get_plot_extent = _map_utils.get_plot_extent
normalize_display_domain = _map_utils.normalize_display_domain
_get_plot_display_domain = _postproc_common.get_plot_display_domain
resolve_model_config_path = _core_config.resolve_model_config_path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = WORKSPACE_ROOT
TEMP_ROOT = ERA5_MSWT_ROOT
PRED_CONFIG = PREDICTION_DIR / "config.yaml"
DEFAULT_SHAPEFILE_PATH = PROJECT_ROOT / "morocco_shapefile" / "morocco_unified_fixed_v2.shp"

SELECTED_LABELS = ["LMDZ_35", "LMDZ_250"]


@dataclass(frozen=True)
class PredictionSettings:
    experiment: str
    selected_start: str
    selected_end: str
    prediction_tag: str
    output_tag: str
    reference_path: Path
    models_dir: Path
    output_dir: Path
    eval_domain: str
    display_domain: str
    shapefile_path: Path
    main_config_path: Path
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    land_shapefile_path: Path | None = None

    @property
    def prediction_dir(self) -> Path:
        return self.output_dir

    @property
    def configured_extent(self) -> list[float]:
        return [self.lon_min, self.lon_max, self.lat_min, self.lat_max]


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return {} if data is None else data


def resolve_path(base_dir: str | Path | None, path_value: str | Path | None) -> Path | None:
    if path_value in (None, "", "null"):
        return None

    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()

    if base_dir is None:
        base_dir = PROJECT_ROOT

    return (Path(base_dir) / p).resolve()


def normalize_eval_domain(eval_domain: str | None, default: str = "land") -> str:
    if eval_domain in (None, "", "null"):
        eval_domain = default

    eval_domain = str(eval_domain).strip()
    if eval_domain not in VALID_SPATIAL_DOMAINS:
        raise ValueError(
            f"spatial.eval_domain must be one of {sorted(VALID_SPATIAL_DOMAINS)}, "
            f"got {eval_domain!r}"
        )

    return eval_domain


def resolve_main_config_path(cfg: dict) -> Path:
    general = cfg.get("general", {})
    paths = cfg.get("paths", {})

    configured_path = paths.get("main_config_path") or cfg.get("main_config_path")
    if configured_path not in (None, "", "null"):
        resolved = resolve_path(PROJECT_ROOT, configured_path)
        if resolved is None:
            raise ValueError("main_config_path could not be resolved")
        return resolved

    model_type = general.get("model_type", "unet1")
    config_name = (
        general.get("config_name")
        or general.get("architecture_config")
        or cfg.get("config_name")
        or cfg.get("architecture_config")
    )
    return resolve_model_config_path(
        CONFIGS_DIR,
        model_type,
        config_name,
    )


def load_main_region_bounds(main_config_path: str | Path) -> tuple[float, float, float, float]:
    main_cfg = load_yaml(main_config_path)
    region = main_cfg.get("region", {})

    required = ("lon_min", "lon_max", "lat_min", "lat_max")
    missing = [name for name in required if region.get(name) in (None, "", "null")]
    if missing:
        raise ValueError(
            f"Missing region bounds in main config {main_config_path}: {missing}"
        )

    return (
        float(region["lon_min"]),
        float(region["lon_max"]),
        float(region["lat_min"]),
        float(region["lat_max"]),
    )


def get_configured_plot_extent(
    lons: np.ndarray,
    lats: np.ndarray,
    display_domain: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> list[float]:
    display_domain = normalize_display_domain(display_domain, default="land")

    if display_domain == "land":
        return [float(lon_min), float(lon_max), float(lat_min), float(lat_max)]

    return _get_plot_extent(
        lons=lons,
        lats=lats,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        display_domain=display_domain,
    )


def load_prediction_settings(config_path: str | Path = PRED_CONFIG) -> PredictionSettings:
    cfg = load_yaml(config_path)

    general = cfg.get("general", {})
    prediction = cfg.get("prediction", {})
    dates_test = cfg.get("dates", {}).get("test", {})
    paths = cfg.get("paths", {})
    spatial = cfg.get("spatial", {})

    experiment = general.get("experiment", "unet_temperature")
    selected_start = dates_test.get("start")
    selected_end = dates_test.get("end")

    if selected_start is None or selected_end is None:
        raise ValueError("dates.test.start and dates.test.end must exist in prediction config")

    bias_correction = bool(prediction.get("bias_correction", False))
    prediction_tag = "bc" if bias_correction else "nobc"
    output_tag = "bc" if bias_correction else "nbc"

    root_dir = resolve_path(PREDICTION_DIR, paths.get("root_dir"))
    if root_dir is None:
        root_dir = PROJECT_ROOT

    reference_path = resolve_path(root_dir, paths.get("mswt_path"))
    if reference_path is None:
        raise ValueError("paths.mswt_path must be defined in prediction config")

    models_dir = resolve_path(
        TEMP_ROOT / "prediction",
        prediction.get("models_dir", "../results/"),
    )
    output_dir = resolve_path(
        TEMP_ROOT / "prediction",
        prediction.get("output_dir", f"../results/{experiment}/prediction"),
    )

    if models_dir is None:
        raise ValueError("prediction.models_dir could not be resolved")
    if output_dir is None:
        raise ValueError("prediction.output_dir could not be resolved")

    shapefile_path = resolve_path(PREDICTION_DIR, paths.get("shapefile_path"))
    if shapefile_path is None:
        shapefile_path = DEFAULT_SHAPEFILE_PATH

    land_shapefile_path = resolve_path(PROJECT_ROOT, spatial.get("land_shapefile_path"))
    main_config_path = resolve_main_config_path(cfg)
    lon_min, lon_max, lat_min, lat_max = load_main_region_bounds(main_config_path)

    return PredictionSettings(
        experiment=experiment,
        selected_start=str(selected_start),
        selected_end=str(selected_end),
        prediction_tag=prediction_tag,
        output_tag=output_tag,
        reference_path=reference_path,
        models_dir=models_dir,
        output_dir=output_dir,
        eval_domain=normalize_eval_domain(spatial.get("eval_domain", "land"), default="land"),
        display_domain=_get_plot_display_domain(cfg),
        shapefile_path=shapefile_path,
        main_config_path=main_config_path,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        land_shapefile_path=land_shapefile_path,
    )


def get_metric_dirs(
    settings: PredictionSettings,
    metric_name: str,
    create: bool = True,
) -> tuple[Path, Path]:
    metric_root = (
        settings.models_dir
        / settings.experiment
        / "prediction_eval"
        / settings.output_tag
        / metric_name
    )
    data_dir = metric_root / "data"
    plot_dir = metric_root / "plots"

    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, plot_dir


def get_metric_data_dir(settings: PredictionSettings, metric_name: str, create: bool = True) -> Path:
    data_dir, _ = get_metric_dirs(settings, metric_name, create=create)
    return data_dir


def get_metric_plot_dir(settings: PredictionSettings, metric_name: str, create: bool = True) -> Path:
    _, plot_dir = get_metric_dirs(settings, metric_name, create=create)
    return plot_dir


def make_plot_name(stem: str, start: str, end: str, output_tag: str) -> str:
    return f"{stem}_{start}_{end}_{output_tag}.png"


def parse_prediction_filename(path: Path, label_lower: str, prediction_tag: str):
    pattern = (
        rf"^{re.escape(label_lower)}_"
        rf"(\d{{4}}-\d{{2}}-\d{{2}})_"
        rf"(\d{{4}}-\d{{2}}-\d{{2}})_"
        rf"{re.escape(prediction_tag)}\.nc$"
    )

    m = re.match(pattern, path.name)
    if m is None:
        return None

    return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))


def choose_prediction_file(
    prediction_dir: Path,
    label: str,
    selected_start: str,
    selected_end: str,
    prediction_tag: str,
) -> Path | None:
    label_lower = label.lower()
    candidates = sorted(prediction_dir.glob(f"{label_lower}_*_{prediction_tag}.nc"))

    if not candidates:
        return None

    selected_start_ts = pd.Timestamp(selected_start)
    selected_end_ts = pd.Timestamp(selected_end)

    valid = []
    for path in candidates:
        parsed = parse_prediction_filename(path, label_lower, prediction_tag)
        if parsed is None:
            continue

        file_start, file_end = parsed
        if file_start <= selected_start_ts and file_end >= selected_end_ts:
            exact_match = int(file_start == selected_start_ts and file_end == selected_end_ts)
            span_days = (file_end - file_start).days
            valid.append((exact_match, -span_days, path))

    if not valid:
        return None

    valid.sort(reverse=True)
    return valid[0][2]


def parse_era5_downscaled_filename(path: Path):
    matches = re.findall(r"(\d{4}-\d{2}-\d{2})", path.name)
    if len(matches) < 2:
        return None

    name_lower = path.name.lower()
    if not name_lower.startswith("era5") or path.suffix.lower() != ".nc":
        return None

    tag = None
    if name_lower.endswith("_nobc.nc"):
        tag = "nobc"
    elif name_lower.endswith("_bc.nc"):
        tag = "bc"

    return pd.Timestamp(matches[0]), pd.Timestamp(matches[1]), tag


def find_era5_downscaled_file(
    settings: PredictionSettings,
    preferred_tags: Iterable[str] = ("nobc", "bc"),
) -> Path | None:
    selected_start_ts = pd.Timestamp(settings.selected_start)
    selected_end_ts = pd.Timestamp(settings.selected_end)
    prediction_dir = settings.prediction_dir

    for tag in preferred_tags:
        exact = prediction_dir / f"era5_{settings.selected_start}_{settings.selected_end}_{tag}.nc"
        if exact.exists():
            return exact

    valid = []
    tag_rank = {tag: idx for idx, tag in enumerate(preferred_tags)}
    for path in sorted(prediction_dir.glob("era5*.nc")):
        parsed = parse_era5_downscaled_filename(path)
        if parsed is None:
            continue

        file_start, file_end, tag = parsed
        if file_start <= selected_start_ts and file_end >= selected_end_ts:
            exact_match = int(file_start == selected_start_ts and file_end == selected_end_ts)
            rank = tag_rank.get(tag, len(tag_rank))
            span_days = (file_end - file_start).days
            valid.append((exact_match, -rank, -span_days, path))

    if not valid:
        return None

    valid.sort(reverse=True)
    return valid[0][3]


def load_comparison_sources(settings: PredictionSettings) -> list[dict]:
    sources: list[dict] = [
        {
            "label": "MSWT",
            "kind": "obs",
            "path": settings.reference_path,
            "data": open_reference_var(settings.reference_path, preferred_var="air_temperature"),
        }
    ]

    era5_path = find_era5_downscaled_file(settings)
    if era5_path is None:
        print(
            "[WARNING] ERA5_downscaled file not found in "
            f"{settings.prediction_dir}; continuing without ERA5_downscaled."
        )
    else:
        sources.append(
            {
                "label": "ERA5_downscaled",
                "kind": "prediction",
                "path": era5_path,
            }
        )

    for label in SELECTED_LABELS:
        pred_path = choose_prediction_file(
            prediction_dir=settings.prediction_dir,
            label=label,
            selected_start=settings.selected_start,
            selected_end=settings.selected_end,
            prediction_tag=settings.prediction_tag,
        )
        if pred_path is None:
            print(
                f"[WARNING] {label} prediction file not found covering "
                f"{settings.selected_start} -> {settings.selected_end} "
                f"with mode={settings.prediction_tag}; skipping."
            )
            continue

        sources.append(
            {
                "label": label,
                "kind": "prediction",
                "path": pred_path,
            }
        )

    return sources


def source_var_name(label: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower()).strip("_")
    if not key:
        raise ValueError(f"Could not derive a variable-safe name from label={label!r}")
    return key


def load_aligned_comparison_data(settings: PredictionSettings) -> list[dict]:
    sources = load_comparison_sources(settings)
    obs_source = next(source for source in sources if source["kind"] == "obs")
    obs_full = obs_source["data"]
    obs = subset_time(obs_full, settings.selected_start, settings.selected_end)

    obs_land, _, spatial_mask = apply_eval_domain_to_inputs(
        pred=obs,
        obs=obs,
        eval_domain=settings.eval_domain,
        shapefile_path=settings.shapefile_path,
        land_shapefile_path=settings.land_shapefile_path,
    )

    aligned_sources: list[dict] = [
        {
            "label": "MSWT",
            "key": source_var_name("MSWT"),
            "kind": "obs",
            "path": settings.reference_path,
            "data": obs_land,
            "reference": obs_land,
            "spatial_mask": spatial_mask,
        }
    ]

    for source in sources:
        if source["kind"] != "prediction":
            continue

        label = source["label"]
        path = source["path"]
        pred_full = open_prediction_var(path, preferred_var="air_temperature")
        pred = subset_time(pred_full, settings.selected_start, settings.selected_end)
        obs_for_pred = subset_time(obs_full, settings.selected_start, settings.selected_end)

        try:
            pred, obs_for_pred = align_prediction_to_obs(pred, obs_for_pred)
        except ValueError as exc:
            print(f"[WARNING] {label}: {exc}; skipping.")
            continue

        pred, obs_for_pred, spatial_mask = apply_eval_domain_to_inputs(
            pred=pred,
            obs=obs_for_pred,
            eval_domain=settings.eval_domain,
            shapefile_path=settings.shapefile_path,
            land_shapefile_path=settings.land_shapefile_path,
        )

        aligned_sources.append(
            {
                "label": label,
                "key": source_var_name(label),
                "kind": "prediction",
                "path": path,
                "data": pred,
                "reference": obs_for_pred,
                "spatial_mask": spatial_mask,
            }
        )

    if aligned_sources:
        global_mask = aligned_sources[0]["spatial_mask"]
        for source in aligned_sources:
            data = source["data"]
            valid_pixels = np.isfinite(data).any(dim="time") if "time" in data.dims else np.isfinite(data)
            global_mask = global_mask & valid_pixels

        global_mask.name = "spatial_mask"
        global_mask.attrs["eval_domain"] = settings.eval_domain
        global_mask.attrs["mask_type"] = "global_common_valid"

        for source in aligned_sources:
            source["initial_spatial_mask"] = source["spatial_mask"]
            source["data"] = source["data"].where(global_mask)
            source["reference"] = source["reference"].where(global_mask)
            source["spatial_mask"] = global_mask

    return aligned_sources


def standardize_coords(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    rename = {}
    for old, new in (
        ("longitude", "lon"),
        ("latitude", "lat"),
        ("valid_time", "time"),
        ("time_counter", "time"),
    ):
        if (old in obj.dims or old in obj.coords) and new not in obj.dims and new not in obj.coords:
            rename[old] = new

    if rename:
        obj = obj.rename(rename)

    for coord in ("lat", "lon", "time"):
        if coord in obj.coords:
            obj = obj.sortby(coord)

    return obj


def normalize_time_to_date(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.coords:
        raise ValueError("Expected a time coordinate.")

    values = da["time"].values

    try:
        time_index = pd.to_datetime(values)
        time_index = pd.DatetimeIndex(time_index).normalize()
    except Exception:
        clean_dates = []
        for t in values:
            clean_dates.append(
                pd.Timestamp(
                    year=int(t.year),
                    month=int(t.month),
                    day=int(t.day),
                )
            )
        time_index = pd.DatetimeIndex(clean_dates)

    da = da.assign_coords(time=time_index)

    if da.indexes["time"].has_duplicates:
        da = da.groupby("time").mean(dim="time", skipna=True)

    return da


def open_main_var(path: str | Path, preferred_var: str | None = None) -> xr.DataArray:
    ds = xr.open_dataset(path)
    ds = standardize_coords(ds)

    if preferred_var and preferred_var in ds.data_vars:
        da = ds[preferred_var]
    else:
        if len(ds.data_vars) == 0:
            raise ValueError(f"No data variables found in {path}")
        da = ds[list(ds.data_vars)[0]]

    da = standardize_coords(da).squeeze(drop=True)

    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(set(da.dims)):
        raise ValueError(
            f"Selected variable from {path} must contain dims {required_dims}, "
            f"got {da.dims}"
        )

    return da.transpose("time", "lat", "lon")


def open_prediction_var(path: str | Path, preferred_var: str | None = "air_temperature") -> xr.DataArray:
    return open_main_var(path, preferred_var=preferred_var)


def open_reference_var(path: str | Path, preferred_var: str | None = "air_temperature") -> xr.DataArray:
    return open_main_var(path, preferred_var=preferred_var)


def subset_time(da: xr.DataArray, start: str, end: str) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError("Expected a time dimension.")

    da = normalize_time_to_date(da)
    return da.sel(time=slice(pd.Timestamp(start), pd.Timestamp(end)))


def align_prediction_to_obs(
    pred: xr.DataArray,
    obs: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    if "lat" not in pred.dims or "lon" not in pred.dims:
        raise ValueError("Prediction must have lat/lon dims.")
    if "lat" not in obs.dims or "lon" not in obs.dims:
        raise ValueError("Observation must have lat/lon dims.")

    pred = normalize_time_to_date(pred)
    obs = normalize_time_to_date(obs)

    same_lat = (
        pred.sizes["lat"] == obs.sizes["lat"]
        and np.allclose(pred["lat"].values, obs["lat"].values)
    )
    same_lon = (
        pred.sizes["lon"] == obs.sizes["lon"]
        and np.allclose(pred["lon"].values, obs["lon"].values)
    )

    if not (same_lat and same_lon):
        pred = pred.interp(lat=obs["lat"], lon=obs["lon"], method="linear")

    obs, pred = xr.align(obs, pred, join="inner")

    if pred.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time coordinates between prediction and observation.")
    if pred.sizes.get("lat", 0) == 0 or pred.sizes.get("lon", 0) == 0:
        raise ValueError("No overlapping spatial coordinates between prediction and observation.")

    return pred, obs


def apply_eval_domain_to_inputs(
    pred: xr.DataArray,
    obs: xr.DataArray,
    eval_domain: str,
    shapefile_path: str | Path | None,
    land_shapefile_path: str | Path | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    eval_domain = normalize_eval_domain(eval_domain, default="land")

    template = obs.isel(time=0) if "time" in obs.dims else obs
    spatial_mask = build_spatial_mask(
        da=template,
        eval_domain=eval_domain,
        shapefile_path=shapefile_path,
        land_shapefile_path=land_shapefile_path,
    )

    return pred.where(spatial_mask), obs.where(spatial_mask), spatial_mask


def build_common_valid_mask(
    obs: xr.DataArray,
    pred: xr.DataArray,
    spatial_mask: xr.DataArray,
) -> xr.DataArray:
    obs_valid = np.isfinite(obs).any(dim="time") if "time" in obs.dims else np.isfinite(obs)
    pred_valid = np.isfinite(pred).any(dim="time") if "time" in pred.dims else np.isfinite(pred)
    spatial_mask, obs_valid, pred_valid = xr.align(spatial_mask, obs_valid, pred_valid, join="inner")

    common_mask = spatial_mask.astype(bool) & obs_valid & pred_valid
    common_mask.name = "common_valid_mask"
    common_mask.attrs.update(spatial_mask.attrs)
    common_mask.attrs["mask_type"] = "spatial_and_common_finite"
    return common_mask


def apply_common_valid_mask(
    obs: xr.DataArray,
    pred: xr.DataArray,
    spatial_mask: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    common_mask = build_common_valid_mask(obs=obs, pred=pred, spatial_mask=spatial_mask)
    return obs.where(common_mask), pred.where(common_mask), common_mask


def _coords_signature(values: np.ndarray) -> tuple[tuple[int, ...], str]:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha1(arr.view(np.uint8)).hexdigest()
    return arr.shape, digest


def _mask_cache_key(
    lons: np.ndarray,
    lats: np.ndarray,
    domain: str,
    shapefile_path: str | Path | None,
    land_shapefile_path: str | Path | None,
) -> tuple:
    return (
        domain,
        str(Path(shapefile_path).resolve()) if shapefile_path not in (None, "", "null") else None,
        str(Path(land_shapefile_path).resolve()) if land_shapefile_path not in (None, "", "null") else None,
        _coords_signature(lons),
        _coords_signature(lats),
    )


def _template_from_grid(lons: np.ndarray, lats: np.ndarray) -> xr.DataArray:
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if lons.ndim == 1 and lats.ndim == 1:
        data = np.ones((lats.size, lons.size), dtype=float)
        return xr.DataArray(data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))

    if lons.shape != lats.shape:
        raise ValueError(f"2D lons/lats must have identical shapes, got {lons.shape} and {lats.shape}")

    data = np.ones(lons.shape, dtype=float)
    return xr.DataArray(
        data,
        coords={"lat": (("lat", "lon"), lats), "lon": (("lat", "lon"), lons)},
        dims=("lat", "lon"),
    )


_SPATIAL_MASK_CACHE: dict[tuple, xr.DataArray] = {}


def build_spatial_mask_for_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    domain: str,
    shapefile_path: str | Path | None,
    land_shapefile_path: str | Path | None = None,
    mask_cache: dict | None = None,
) -> xr.DataArray:
    domain = normalize_eval_domain(domain, default="land")
    cache = _SPATIAL_MASK_CACHE if mask_cache is None else mask_cache
    key = _mask_cache_key(lons, lats, domain, shapefile_path, land_shapefile_path)

    if key not in cache:
        template = _template_from_grid(lons, lats)
        cache[key] = build_spatial_mask(
            da=template,
            eval_domain=domain,
            shapefile_path=shapefile_path,
            land_shapefile_path=land_shapefile_path,
        )

    return cache[key]


def apply_display_domain(
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    display_domain: str,
    shapefile_path: str | Path | None,
    land_shapefile_path: str | Path | None = None,
    mask_cache: dict | None = None,
) -> np.ndarray:
    display_domain = normalize_display_domain(display_domain, default="land")
    arr = np.asarray(arr, dtype=float)

    da = xr.DataArray(
        arr,
        coords={"lat": np.asarray(lats), "lon": np.asarray(lons)},
        dims=("lat", "lon"),
    )
    spatial_mask = build_spatial_mask_for_grid(
        lons=lons,
        lats=lats,
        domain=display_domain,
        shapefile_path=shapefile_path,
        land_shapefile_path=land_shapefile_path,
        mask_cache=mask_cache,
    )

    return da.where(spatial_mask).values


def valid_pixel_count(mask: xr.DataArray) -> int:
    return int(mask.sum(skipna=True).item())


def finite_stats(arr: np.ndarray | xr.DataArray) -> dict:
    values = arr.values if isinstance(arr, xr.DataArray) else np.asarray(arr)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if valid.sum() == 0:
        return {
            "n_valid": 0,
            "min": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "std": np.nan,
            "median": np.nan,
        }

    vals = values[valid]
    return {
        "n_valid": int(valid.sum()),
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "max": float(np.nanmax(vals)),
        "std": float(np.nanstd(vals)),
        "median": float(np.nanmedian(vals)),
    }


def add_stat_attrs(ds: xr.Dataset, prefix: str, arr: np.ndarray | xr.DataArray) -> None:
    for key, value in finite_stats(arr).items():
        ds.attrs[f"{prefix}_{key}"] = value


def infer_bundle_mode(ds: xr.Dataset, path: Path, bundle_suffix: str) -> str:
    bc_mode = ds.attrs.get("bc_mode", None)
    if bc_mode in ("bc", "nbc"):
        return bc_mode

    name = path.name.lower()
    if "_nbc_" in name or name.endswith(f"_nbc_{bundle_suffix}"):
        return "nbc"
    if "_bc_" in name or name.endswith(f"_bc_{bundle_suffix}"):
        return "bc"

    return "unknown"


def load_metric_bundles(
    data_dir: Path,
    settings: PredictionSettings,
    bundle_suffix: str,
    selected_labels: Iterable[str] = SELECTED_LABELS,
) -> list[tuple[str, xr.Dataset]]:
    files = sorted(data_dir.glob(f"*_{bundle_suffix}"))
    if not files:
        raise FileNotFoundError(f"No bundles matching *_{bundle_suffix} found in {data_dir}")

    selected_labels = list(selected_labels)
    bundles = []

    for path in files:
        ds = xr.open_dataset(path)
        label = ds.attrs.get(
            "model_label",
            path.stem.replace(f"_{bundle_suffix.replace('.nc', '')}", "").upper(),
        )
        bundle_start = str(ds.attrs.get("start_date", ""))
        bundle_end = str(ds.attrs.get("end_date", ""))
        bundle_mode = infer_bundle_mode(ds, path, bundle_suffix)

        if label not in selected_labels:
            ds.close()
            continue
        if bundle_start != settings.selected_start or bundle_end != settings.selected_end:
            ds.close()
            continue
        if bundle_mode != settings.output_tag:
            ds.close()
            continue

        bundles.append((label, ds))

    ordered = []
    for wanted in selected_labels:
        for label, ds in bundles:
            if label == wanted:
                ordered.append((label, ds))
                break

    if not ordered:
        raise ValueError(
            f"No bundles match selected period={settings.selected_start}->{settings.selected_end} "
            f"and mode={settings.output_tag} in {data_dir}"
        )

    missing = [label for label in selected_labels if label not in [item[0] for item in ordered]]
    if missing:
        print(f"[WARNING] Missing bundles for: {missing}")

    return ordered
