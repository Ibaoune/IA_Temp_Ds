#fichiers bruts → data_loading.py → conversions/unités/masque/reconstruction de q → 
# raw predictors → bias correction éventuelle → corrected predictors → 
# extraction par variable/niveau → flatten → KDE/hist = estimation de la densité → figure

from __future__ import annotations

from pathlib import Path
import copy
import math
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------------------------------------------------------
# project imports
# ---------------------------------------------------------
try:
    from .backend_paths import CONFIGS_DIR, ERA5_MSWT_ROOT, PREDICTION_DIR, import_main_module
except ImportError:
    from backend_paths import CONFIGS_DIR, ERA5_MSWT_ROOT, PREDICTION_DIR, import_main_module

_core_config = import_main_module("src.core.config")
_data_loading = import_main_module("src.data.data_loading")
_core_utils = import_main_module("src.core.utils")

Config = _core_config.Config
resolve_model_config_path = _core_config.resolve_model_config_path
load_datasets = _data_loading.load_datasets
set_verbose = _core_utils.set_verbose
vprint = _core_utils.vprint

try:
    from .bias_correction import scaling_delta_mapping
except ImportError:
    from bias_correction import scaling_delta_mapping


# =========================================================
# PATHS
# =========================================================
THIS_FILE = Path(__file__).resolve()
TEMP_ROOT = ERA5_MSWT_ROOT
PRED_CONFIG = PREDICTION_DIR / "config.yaml"

RNG = np.random.default_rng(42)
MAX_POINTS_PER_CURVE = 50000


# =========================================================
# CONFIG / SETTINGS
# =========================================================
def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def merge_prediction_and_main_config() -> Config:
    pred_cfg_dict = load_yaml(PRED_CONFIG)

    general_cfg = pred_cfg_dict.get("general", {})
    model_type = general_cfg.get("model_type", "unet")
    config_name = (
        general_cfg.get("config_name")
        or general_cfg.get("architecture_config")
        or pred_cfg_dict.get("config_name")
        or pred_cfg_dict.get("architecture_config")
    )
    base_config_path = resolve_model_config_path(
        CONFIGS_DIR,
        model_type,
        config_name,
    )

    if base_config_path.exists():
        full_cfg_dict = load_yaml(base_config_path)
        for key in pred_cfg_dict:
            if key in full_cfg_dict and isinstance(full_cfg_dict[key], dict):
                full_cfg_dict[key].update(pred_cfg_dict[key])
            else:
                full_cfg_dict[key] = pred_cfg_dict[key]
    else:
        full_cfg_dict = pred_cfg_dict

    cfg = Config(full_cfg_dict, train_mode=False)

    general = pred_cfg_dict.get("general", {})
    paths = pred_cfg_dict.get("paths", {})
    dates_test = pred_cfg_dict.get("dates", {}).get("test", {})
    prediction = pred_cfg_dict.get("prediction", {})

    cfg.src = general.get("src", getattr(cfg, "src", "lmdz"))
    cfg.folder = paths.get("folder", getattr(cfg, "folder", ""))
    cfg.start_date_test = dates_test.get("start", getattr(cfg, "start_date_test", None))
    cfg.end_date_test = dates_test.get("end", getattr(cfg, "end_date_test", None))
    cfg.bc_mode = prediction.get("bc_mode", "historical")

    set_verbose(getattr(cfg, "verbose", True))
    return cfg


def load_runtime_settings(cfg: Config) -> dict:
    pred_cfg_dict = load_yaml(PRED_CONFIG)

    general = pred_cfg_dict.get("general", {})
    paths = pred_cfg_dict.get("paths", {})
    dates_test = pred_cfg_dict.get("dates", {}).get("test", {})
    prediction = pred_cfg_dict.get("prediction", {})

    experiment = general.get("experiment", "unet_temperature")
    raw_root_dir = Path(paths["root_dir"]) if paths.get("root_dir") else None
    root_dir = (
        raw_root_dir.resolve() if raw_root_dir and raw_root_dir.is_absolute()
        else (PREDICTION_DIR / raw_root_dir).resolve() if raw_root_dir
        else None
    )

    selected_start = dates_test.get("start")
    selected_end = dates_test.get("end")
    if selected_start is None or selected_end is None:
        raise ValueError("dates.test.start and dates.test.end must be defined in config.yaml")

    # For historical mode, reference period = selected period by default.
    # For future mode, you can optionally define reference_start/reference_end in prediction.
    bc_mode = prediction.get("bc_mode", "historical")
    reference_start = prediction.get("reference_start", selected_start)
    reference_end = prediction.get("reference_end", selected_end)

    out_data_dir = TEMP_ROOT / "results" / experiment / "prediction_dist" / "data"
    out_plot_dir = TEMP_ROOT / "results" / experiment / "prediction_dist" / "plot"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    return {
        "experiment": experiment,
        "root_dir": root_dir,
        "selected_start": selected_start,
        "selected_end": selected_end,
        "bc_mode": bc_mode,
        "reference_start": reference_start,
        "reference_end": reference_end,
        "output_data_dir": out_data_dir,
        "output_plot_dir": out_plot_dir,
    }


# =========================================================
# MODEL FOLDER DISCOVERY
# =========================================================
def discover_model_folder(root_dir: Path, label: str, start: str, end: str) -> str:
    """
    Return a folder path RELATIVE to root_dir, because data_loading.py
    already rebuilds the full path from root_dir + folder.
    """
    start_year = start[:4]
    end_year = end[:4]

    if label == "LMDZ_35":
        candidates = [
            Path("lmdz_35"),
            Path("lmdz_35") / f"all_Mor" / f"present_{start_year}_{end_year}",
        ]
    elif label == "LMDZ_250":
        candidates = [
            Path("lmdz_250"),
            Path("lmdz_250") / f"all_Mor" / f"present_{start_year}_{end_year}",
        ]
    else:
        raise ValueError(f"Unknown label: {label}")

    for rel_path in candidates:
        full_path = root_dir / rel_path
        if full_path.exists():
            return str(rel_path).replace("\\", "/")

    raise FileNotFoundError(
        f"Could not find folder for {label}. Tried:\n" +
        "\n".join(str(root_dir / c) for c in candidates)
    )


# =========================================================
# DATA LOADING HELPERS
# =========================================================
def clone_cfg(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def make_cfg_for_source(
    cfg_base: Config,
    src: str,
    folder: str | Path | None,
    start: str,
    end: str,
) -> Config:
    cfg = clone_cfg(cfg_base)
    cfg.src = src
    src_key = str(src).lower()
    if src_key == "era5":
        cfg.predictor_pattern = cfg.era5_predictor_pattern
    elif src_key == "lmdz250":
        cfg.predictor_pattern = cfg.lmdz250_predictor_pattern
    elif src_key == "lmdz35":
        cfg.predictor_pattern = cfg.lmdz35_predictor_pattern
    elif src_key == "lmdz":
        cfg.predictor_pattern = cfg.lmdz_predictor_pattern
    if folder is not None:
        cfg.folder = str(folder)
    cfg.start_date_test = start
    cfg.end_date_test = end
    return cfg


def load_predictor_stack(cfg: Config) -> xr.DataArray:
    """
    Returns X from load_datasets, restricted to cfg.start_date_test:end_date_test.
    """
    datasets = load_datasets(cfg)
    X = datasets[0].sel(time=slice(cfg.start_date_test, cfg.end_date_test))
    return X


def get_channel_dim_name(X: xr.DataArray) -> str:
    if len(X.dims) < 4:
        raise ValueError(f"Expected predictor stack with at least 4 dims, got {X.dims}")
    return X.dims[1]


def get_channel(X: xr.DataArray, idx: int) -> xr.DataArray:
    channel_dim = get_channel_dim_name(X)
    return X.isel({channel_dim: idx})


def build_channel_index(variables, levels):
    mapping = {}
    n_levels = len(levels)
    for iv, var in enumerate(variables):
        for il, lev in enumerate(levels):
            mapping[(str(var).lower(), int(lev))] = iv * n_levels + il
    return mapping


def sample_valid_values(arr: xr.DataArray | np.ndarray, max_points: int = MAX_POINTS_PER_CURVE) -> np.ndarray:
    vals = np.asarray(arr).astype("float64").ravel()
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return vals

    if vals.size > max_points:
        idx = RNG.choice(vals.size, size=max_points, replace=False)
        vals = vals[idx]

    return vals


# =========================================================
# DISTRIBUTION COMPUTE
# =========================================================
def compute_all_stacks(cfg_base: Config, settings: dict):
    root_dir = settings["root_dir"]
    selected_start = settings["selected_start"]
    selected_end = settings["selected_end"]
    bc_mode = settings["bc_mode"]
    ref_start = settings["reference_start"]
    ref_end = settings["reference_end"]

    if root_dir is None:
        raise ValueError("paths.root_dir must be defined in prediction config")

    # ERA5 reference for selected period (used for display)
    vprint("\n=== Loading ERA5 selected-period reference ===")
    cfg_era_selected = make_cfg_for_source(
        cfg_base, src="era5", folder=None, start=selected_start, end=selected_end
    )
    X_era_selected = load_predictor_stack(cfg_era_selected)

    # ERA5 reference for BC if needed
    if bc_mode == "historical":
        X_era_bc_ref = X_era_selected
    else:
        vprint(f"\n=== Loading ERA5 reference period for BC: {ref_start} -> {ref_end} ===")
        cfg_era_ref = make_cfg_for_source(
            cfg_base, src="era5", folder=None, start=ref_start, end=ref_end
        )
        X_era_bc_ref = load_predictor_stack(cfg_era_ref)

    out = {
        "ERA5": {
            "reference": X_era_selected,
        }
    }

    for label in ["LMDZ_35", "LMDZ_250"]:
        folder = discover_model_folder(root_dir, label, selected_start, selected_end)

        vprint(f"\n=== Loading {label} raw predictors ===")
        cfg_raw = make_cfg_for_source(
            cfg_base, src="lmdz", folder=folder, start=selected_start, end=selected_end
        )
        X_raw = load_predictor_stack(cfg_raw)

        if bc_mode == "historical":
            vprint(f"=== Applying historical BC for {label} on selected period ===")
            X_gcm_hist = X_raw
            X_corrected = scaling_delta_mapping(X_raw, X_gcm_hist, X_era_bc_ref)
        else:
            vprint(f"=== Loading GCM historical reference for {label}: {ref_start} -> {ref_end} ===")
            cfg_hist = make_cfg_for_source(
                cfg_base, src="lmdz", folder=folder, start=ref_start, end=ref_end
            )
            X_gcm_hist = load_predictor_stack(cfg_hist)
            X_corrected = scaling_delta_mapping(X_raw, X_gcm_hist, X_era_bc_ref)

        out[label] = {
            "raw": X_raw,
            "corrected": X_corrected,
        }

    return out


def make_distribution_dictionary(stacks: dict, variables, levels):
    ch_map = build_channel_index(variables, levels)

    distributions = {}

    for var in variables:
        var_key = str(var).lower()
        distributions[var_key] = {}

        for lev in levels:
            lev_int = int(lev)
            idx = ch_map[(var_key, lev_int)]

            distributions[var_key][lev_int] = {
                "ERA5_Reference": sample_valid_values(get_channel(stacks["ERA5"]["reference"], idx)),
                "LMDZ_35_Raw": sample_valid_values(get_channel(stacks["LMDZ_35"]["raw"], idx)),
                "LMDZ_35_Corrected": sample_valid_values(get_channel(stacks["LMDZ_35"]["corrected"], idx)),
                "LMDZ_250_Raw": sample_valid_values(get_channel(stacks["LMDZ_250"]["raw"], idx)),
                "LMDZ_250_Corrected": sample_valid_values(get_channel(stacks["LMDZ_250"]["corrected"], idx)),
            }

    return distributions


def save_distribution_data(distributions: dict, variables, levels, settings: dict):
    out_dir = settings["output_data_dir"]
    start = settings["selected_start"]
    end = settings["selected_end"]
    bc_mode = settings["bc_mode"]

    summary_rows = []

    for var in variables:
        var_key = str(var).lower()
        payload = {"levels": np.array(levels, dtype=int)}

        for lev in levels:
            lev_int = int(lev)
            series = distributions[var_key][lev_int]

            for name, arr in series.items():
                payload[f"{lev_int}__{name}"] = arr

                summary_rows.append(
                    {
                        "variable": var_key,
                        "level_hpa": lev_int,
                        "series": name,
                        "n": int(arr.size),
                        "mean": float(np.nanmean(arr)) if arr.size else np.nan,
                        "std": float(np.nanstd(arr)) if arr.size else np.nan,
                        "min": float(np.nanmin(arr)) if arr.size else np.nan,
                        "max": float(np.nanmax(arr)) if arr.size else np.nan,
                        "selected_start": start,
                        "selected_end": end,
                        "bc_mode": bc_mode,
                    }
                )

        out_npz = out_dir / f"{var_key}_dist_{start}_{end}_{bc_mode}.npz"
        np.savez_compressed(out_npz, **payload)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"distribution_summary_{start}_{end}_{bc_mode}.csv"
    summary_df.to_csv(summary_csv, index=False)

    return summary_csv


# =========================================================
# PLOTTING
# =========================================================
UNIT_MAP = {
    "z": "m²/s²",
    "q": "kg/kg",
    "t": "°C",
    "u": "m/s",
    "v": "m/s",
}

SERIES_STYLE = {
    "ERA5_Reference":    {"color": "black",    "linestyle": "-",  "linewidth": 2.2, "label": "ERA5 (Reference)"},
    "LMDZ_35_Raw":       {"color": "red",      "linestyle": "--", "linewidth": 1.5, "label": "LMDZ-35 Raw"},
    "LMDZ_35_Corrected": {"color": "orange",   "linestyle": "-",  "linewidth": 1.5, "label": "LMDZ-35 Corrected"},
    "LMDZ_250_Raw":      {"color": "blue",     "linestyle": "--", "linewidth": 1.5, "label": "LMDZ-250 Raw"},
    "LMDZ_250_Corrected":{"color": "cyan",     "linestyle": "-",  "linewidth": 1.5, "label": "LMDZ-250 Corrected"},
}


def plot_density_curve(ax, values: np.ndarray, style: dict):
    if values.size < 2:
        return

    values = values[np.isfinite(values)]
    if values.size < 2:
        return

    if HAVE_SCIPY and np.nanstd(values) > 0:
        xs = np.linspace(np.nanmin(values), np.nanmax(values), 400)
        kde = gaussian_kde(values)
        ys = kde(xs)
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
    else:
        bins = min(60, max(20, int(np.sqrt(values.size))))
        hist, edges = np.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            hist,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )


def plot_variable_distribution(
    var_key: str,
    var_data: dict,
    levels,
    settings: dict,
    cfg: Config,
):
    unit = UNIT_MAP.get(var_key, "")
    start = settings["selected_start"]
    end = settings["selected_end"]
    bc_mode = settings["bc_mode"]
    out_plot_dir = settings["output_plot_dir"]

    n_levels = len(levels)
    ncols = 2 if n_levels > 1 else 1
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, 4.8 * nrows),
        dpi=160,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, lev in enumerate(levels):
        lev_int = int(lev)
        ax = axes[ax_idx]
        series = var_data[lev_int]

        for series_name, style in SERIES_STYLE.items():
            plot_density_curve(ax, series[series_name], style)

        ax.set_title(f"Level: {lev_int} hPa", fontsize=11)
        ax.set_xlabel(f"{var_key} ({unit})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=9)

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right", frameon=True)

    for k in range(n_levels, len(axes)):
        axes[k].axis("off")

    region_str = (
        f"Region: [{cfg.lon_min:.1f},{cfg.lon_max:.1f}] Lon, "
        f"[{cfg.lat_min:.1f},{cfg.lat_max:.1f}] Lat  |  Unit: {unit}"
    )

    fig.suptitle(
        f"Distribution Comparison: {var_key} ({start} to {end})\n{region_str}",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_plot_dir / f"dist_{var_key}_{start}_{end}_{bc_mode}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = merge_prediction_and_main_config()
    settings = load_runtime_settings(cfg)

    variables = [str(v).lower() for v in cfg.variables]
    levels = [int(lv) for lv in cfg.levels]

    print("=== Predictor distribution diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Experiment        : {settings['experiment']}")
    print(f"Selected period   : {settings['selected_start']} -> {settings['selected_end']}")
    print(f"BC mode           : {settings['bc_mode']}")
    print(f"Output data dir   : {settings['output_data_dir']}")
    print(f"Output plot dir   : {settings['output_plot_dir']}")
    print(f"Variables         : {variables}")
    print(f"Levels            : {levels}")

    stacks = compute_all_stacks(cfg, settings)
    distributions = make_distribution_dictionary(stacks, variables, levels)
    summary_csv = save_distribution_data(distributions, variables, levels, settings)

    print(f"[SUCCESS] Summary CSV saved to: {summary_csv}")

    for var in variables:
        out_fig = plot_variable_distribution(
            var_key=var,
            var_data=distributions[var],
            levels=levels,
            settings=settings,
            cfg=cfg,
        )
        print(f"[SUCCESS] Figure saved: {out_fig}")

    print("[DONE] Predictor distribution diagnostics completed.")


if __name__ == "__main__":
    main()
