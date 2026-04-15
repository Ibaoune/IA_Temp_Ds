"""
==========================================================
 Script: utils.py
 Author: M. El Aabaribaoune
 Description:
     Collection of utility and helper functions used across
     the project, including:
     - Verbose printing utilities
     - Dataset spatial masking
     - Visualization and comparison plots
     - Loss curve plotting
     - Experiment metadata formatting
     - Intelligent experiment path construction
     - Model saving and loading utilities

==========================================================
"""

# Standard & Scientific Imports
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import regionmask
import xarray as xr
import platform
import textwrap
try:
    import psutil
except ImportError:
    psutil = None
import time


# Verbose Printing Utility
_VERBOSE = True
def set_verbose(flag: bool):
    global _VERBOSE
    _VERBOSE = flag

def vprint(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)

# Dataset Utilities
def mask_dataset(ds: xr.Dataset, lon_range: slice, lat_range: slice) -> xr.Dataset:
    """
    Apply spatial mask and print detailed debug information.
    """

    vprint("→ Applying spatial mask")

    # -----------------------------
    # Detect lon / lat names
    # -----------------------------
    lon_name = next(v for v in ds.coords if v.startswith("lon"))
    lat_name = next(v for v in ds.coords if v.startswith("lat"))

    # -----------------------------
    # DEBUG BEFORE MASK
    # -----------------------------
    vprint("  [BEFORE MASK]")
    vprint(f"   dims        : {ds.dims}")
    vprint(f"   shape       : {tuple(ds.sizes[d] for d in ds.dims)}")
    vprint(f"   lon range   : {float(ds[lon_name].min())} → {float(ds[lon_name].max())}")
    vprint(f"   lat range   : {float(ds[lat_name].min())} → {float(ds[lat_name].max())}")

    # -----------------------------
    # Handle lat orientation safely
    # -----------------------------
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_range.stop, lat_range.start)
    else:
        lat_slice = lat_range

    # -----------------------------
    # Apply mask
    # -----------------------------
    ds_masked = ds.sel(
        **{
            lon_name: lon_range,
            lat_name: lat_slice,
        }
    )

    # -----------------------------
    # DEBUG AFTER MASK
    # -----------------------------
    vprint("  [AFTER MASK]")
    vprint(f"   dims        : {ds_masked.dims}")
    vprint(f"   shape       : {tuple(ds_masked.sizes[d] for d in ds_masked.dims)}")

    if ds_masked.sizes[lat_name] == 0 or ds_masked.sizes[lon_name] == 0:
        vprint(">!!< WARNING: EMPTY spatial dimension after mask")

    vprint(f"  Mask applied: lon={lon_range}, lat={lat_range}")

    return ds_masked


def spatial_comparaison_plot(
    y_test_ds,
    output_ds,
    lon_values,
    lat_values,
    model_name,
    filename,
    title_suffix="",
    y_name="MSWT",
    y_var="air_temperature",
    model_var="air_temperature",
):
    """
    Improved spatial comparison plot for temperature.

    Keeps the old layout:
    - dedicated top area for title + metadata
    - two map panels
    - horizontal colorbar below
    """
    vprint("→ Generating improved spatial temperature plot")

    obs = y_test_ds[y_var]
    pred = output_ds[model_var]

    obs_mean = obs.mean(dim="time")
    pred_mean = pred.mean(dim="time")

    vmin = float(min(obs_mean.min().item(), pred_mean.min().item()))
    vmax = float(max(obs_mean.max().item(), pred_mean.max().item()))

    levels = np.arange(np.floor(vmin), np.ceil(vmax) + 2, 2)
    cmap = plt.get_cmap("RdYlBu_r", len(levels) - 1)
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N)

    fig = plt.figure(figsize=(14, 7), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.22, 1.0, 0.05]
    )

    meta_ax = fig.add_subplot(gs[0, :])
    meta_ax.axis("off")

    ax1 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())

    cax = fig.add_subplot(gs[2, :])

    datasets = [
        (obs_mean, y_name, ax1),
        (pred_mean, model_name, ax2),
    ]

    img = None
    for data, title, ax in datasets:
        img = ax.pcolormesh(
            lon_values,
            lat_values,
            data,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines(linewidth=0.8)
        ax.set_extent([
            float(np.min(lon_values)), float(np.max(lon_values)),
            float(np.min(lat_values)), float(np.max(lat_values))
        ])

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            alpha=0.4,
            linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

        ax.set_title(title, fontsize=12, weight="bold", pad=8)

        txt = (
            f"Mean: {float(data.mean()):.2f} °C\n"
            f"Min: {float(data.min()):.2f} °C\n"
            f"Max: {float(data.max()):.2f} °C"
        )
        ax.text(
            0.02, 0.02,
            txt,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.8,
                edgecolor="0.6"
            )
        )

    meta_ax.text(
        0.5, 0.88,
        "Spatial Distribution of Mean Temperature",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold"
    )

    if title_suffix:
        wrapped = "\n".join(textwrap.wrap(title_suffix, width=95))
        meta_ax.text(
            0.5, 0.52,
            wrapped,
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            color="black"
        )

    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.set_label("Temperature (°C)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(filename, dpi=300)
    plt.close(fig)

    vprint(f"  Improved spatial temperature plot saved to: {filename}")

def monthly_temp_comparaison_plot(
    output_model,
    y_test_ds,
    model_name,
    filename,
    y_name="MSWT",
    y_var="air_temperature",
    model_var="air_temperature",
    title="Monthly Average Temperature (°C)",
    title_suffix="",
):
    """
    Monthly comparison plot for temperature only.
    Keeps professor-style monthly comparison layout.
    """
    vprint("→ Generating monthly temperature comparison plot")

    region = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = region.mask(output_model)

    ds_masked_model = output_model.where(mask.notnull())
    ds_masked_y = y_test_ds.where(mask.notnull())

    dataset1 = ds_masked_model.resample(time="1ME").mean()
    dataset2 = ds_masked_y.resample(time="1ME").mean()

    temp_model = (
        dataset1.groupby("time.month")
        .mean(dim="time")
        .mean(dim=["lon", "lat"])[model_var]
        .values
    )
    temp_y = (
        dataset2.groupby("time.month")
        .mean(dim="time")
        .mean(dim=["lon", "lat"])[y_var]
        .values
    )

    months = np.arange(1, 13)
    bias_model = (temp_y - temp_model).mean()
    rmse_model = np.sqrt(((temp_y - temp_model) ** 2).mean())

    vprint(f"  {model_name} Bias: {bias_model:.4f}, RMSE: {rmse_model:.4f}")

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(
        months,
        temp_model,
        marker="o",
        label=f"{model_name}: Bias={bias_model:.2g}, RMSE={rmse_model:.2g}",
    )
    plt.plot(months, temp_y, marker="o", label=y_name)

    if title.strip():
        plt.title(f"{title}\n{y_name} vs {model_name}", fontsize=6, pad=10)
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=6, style="italic", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.xticks(
        months,
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    vprint(f"  Monthly plot saved to: {filename}")

# Training Curves
def plot_losses(
    train_losses,
    val_losses,
    model_name,
    epochs,
    batch_size,
    filename,
    title="Training and Validation Losses",
    title_suffix="",
):
    """
    Plot training and validation loss curves.
    """
    vprint("→ Plotting training / validation losses")

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_losses, label="Train Loss")
    if val_losses and len(val_losses) > 0:
        plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if title.strip():
        plt.title(f"{title} - {model_name}", weight="bold")
    if title_suffix:
        plt.suptitle(title_suffix, fontsize=7, style="italic", y=0.96)

    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    vprint(f"  Loss plot saved to: {filename}")


# Experiment Metadata Formatting
def format_components_for_title(**kwargs):
    """
    Compact plot title formatter (max 4 lines)
    Designed for evaluation plots.
    """
    def pick(*keys):
        return {k: kwargs[k] for k in keys if k in kwargs and kwargs[k] is not None}

    def fmt(d):
        return " | ".join(f"{k}: {v}" for k, v in d.items()) if d else None

    line1 = fmt(pick(
        "src", "target", "variable",
        "variables", "levels", "resolution"
    ))

    line2 = fmt(pick(
        "experiment", "model_type", "interpolation_type"
    ))

    line3 = fmt(pick(
        "norm_mode", "loss_type",
        "learning_rate", "batch_size", "epochs",
        "early_stopping_max"
    ))

    line4 = None
    if all(k in kwargs for k in [
        "train_start", "train_end",
        "test_start", "test_end"
    ]):
        line4 = (
            f"Train: {kwargs['train_start']} → {kwargs['train_end']} | "
            f"Test: {kwargs['test_start']} → {kwargs['test_end']}"
        )

    lines = [line1, line2, line3, line4]
    lines = [l for l in lines if l]
    return "\n".join(lines[:4])

# Experiment Path & Model IO
def build_experiment_path(cfg):
    """
    Build a structured experiment path. 
    Simplified to results/<experiment> as per user request.
    """
    base_dir = getattr(cfg, "results_dir", "results")
    return os.path.join(base_dir, cfg.experiment)


def save_model(cfg, model, train_losses=None, val_losses=None, tag="last", best_score=None):
    exp_path = build_experiment_path(cfg)
    model_dir = os.path.join(exp_path, "models")
    os.makedirs(model_dir, exist_ok=True)

    if isinstance(model, nn.Module):
        model_name = f"{cfg.model_type}_{cfg.variable}_{tag}.pth"
        model_path = os.path.join(model_dir, model_name)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_score": best_score,
                "tag": tag,
                "config": vars(cfg),
            },
            model_path,
        )
    else:
        import pickle
        model_name = f"{cfg.model_type}_{cfg.variable}_{tag}.pkl"
        model_path = os.path.join(model_dir, model_name)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    config_txt_path = os.path.join(exp_path, "config.txt")
    with open(config_txt_path, "w") as f:
        f.write(f"Experiment: {cfg.experiment}\n")
        f.write(f"Variable: {cfg.variable}\n")
        f.write(f"Model Type: {cfg.model_type}\n")
        f.write("-" * 30 + "\n")
        for k, v in vars(cfg).items():
            f.write(f"{k}: {v}\n")

    vprint(f" {tag.upper()} model saved at: {model_path}")
    vprint(f" Config saved at: {config_txt_path}")
    return model_path


def load_model(cfg, model=None, tag="best"):
    """
    Load trained checkpoint.
    By default, load BEST model if it exists.
    Fallback order for torch models:
    1) tagged file requested
    2) last file
    3) old legacy unsuffixed file
    """
    exp_path = build_experiment_path(cfg)
    model_dir = os.path.join(exp_path, "models")

    tagged_pth = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}_{tag}.pth")
    last_pth = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}_last.pth")
    legacy_pth = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}.pth")

    tagged_pkl = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}_{tag}.pkl")
    last_pkl = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}_last.pkl")
    legacy_pkl = os.path.join(model_dir, f"{cfg.model_type}_{cfg.variable}.pkl")

    # Torch models
    for pth_path in [tagged_pth, last_pth, legacy_pth]:
        if os.path.exists(pth_path):
            vprint(f" Loading PyTorch model from: {pth_path}")
            checkpoint = torch.load(pth_path, map_location="cpu")
            if model is not None:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            return (
                model,
                checkpoint.get("train_losses"),
                checkpoint.get("val_losses"),
            )

    # Non-torch models
    for pkl_path in [tagged_pkl, last_pkl, legacy_pkl]:
        if os.path.exists(pkl_path):
            vprint(f" Loading model from: {pkl_path}")
            import pickle
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            return model, None, None

    raise FileNotFoundError(
        f"No model found for {cfg.model_type}_{cfg.variable} in {model_dir}"
    )

# Metric Functions
def calculate_rmse(y_true, y_pred):
    """
    Compute RMSE along the time axis (axis 0).
    Accepts 1-D (single grid point) or N-D arrays.
    """
    return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=0))


def calculate_r2(y_true, y_pred):
    """
    Compute R² (coefficient of determination) along the time axis.
    """
    ss_res = np.nansum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.nansum((y_true - np.nanmean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(np.isfinite(r2), r2, np.nan)
    return r2


def calculate_pearson(y_true, y_pred):
    """
    Compute Pearson correlation coefficient along the time axis
    for each spatial grid point.
    """
    from scipy.stats import pearsonr as _pearsonr

    if y_true.ndim == 1:
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 3:
            return np.nan
        r, _ = _pearsonr(y_true[mask], y_pred[mask])
        return r

    # Spatial arrays: iterate over grid points
    nlat, nlon = y_true.shape[1], y_true.shape[2]
    result = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            t = y_true[:, i, j]
            p = y_pred[:, i, j]
            mask = ~(np.isnan(t) | np.isnan(p))
            if mask.sum() < 3:
                continue
            r, _ = _pearsonr(t[mask], p[mask])
            result[i, j] = r
    return result


def compute_spatial_metric(y_true, y_pred, metric_fn):
    """
    Apply a metric function to compute a 2-D spatial map.
    metric_fn should accept (y_true, y_pred) with shape (time, lat, lon)
    and return a (lat, lon) map.
    """
    return metric_fn(y_true, y_pred)


def calculate_monthly_metrics(pred_da, obs_da):
    """
    Compute monthly Pearson correlation, RMSE, and R² between
    two xarray DataArrays (model predictions and observations).

    Returns:
        months       : array of month indices (1..12)
        monthly_corr : Pearson correlation per month
        monthly_rmse : RMSE per month
        monthly_r2   : R² per month
    """
    from scipy.stats import pearsonr as _pearsonr

    months = np.arange(1, 13)
    monthly_corr = []
    monthly_rmse = []
    monthly_r2 = []

    for m in months:
        pred_m = pred_da.sel(time=pred_da["time.month"] == m).values.flatten()
        obs_m = obs_da.sel(time=obs_da["time.month"] == m).values.flatten()

        mask = ~(np.isnan(pred_m) | np.isnan(obs_m))
        pred_m = pred_m[mask]
        obs_m = obs_m[mask]

        if len(pred_m) < 3:
            monthly_corr.append(np.nan)
            monthly_rmse.append(np.nan)
            monthly_r2.append(np.nan)
            continue

        r, _ = _pearsonr(pred_m, obs_m)
        monthly_corr.append(r)

        rmse = np.sqrt(np.mean((pred_m - obs_m) ** 2))
        monthly_rmse.append(rmse)

        ss_res = np.sum((obs_m - pred_m) ** 2)
        ss_tot = np.sum((obs_m - np.mean(obs_m)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        monthly_r2.append(r2)

    return months, np.array(monthly_corr), np.array(monthly_rmse), np.array(monthly_r2)


def get_machine_features():
    """
    Returns a dictionary of relevant machine features (CPU, RAM, GPU).
    """
    features = {
        "os": f"{platform.system()} {platform.release()}",
    }

    if psutil:
        features["cpu"] = platform.processor()
        features["cores"] = psutil.cpu_count(logical=True)
        features["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    else:
        features["cpu"] = platform.processor()
        features["cores"] = "Unknown (psutil missing)"
        features["ram_total_gb"] = "Unknown (psutil missing)"

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        features["gpus"] = []
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            features["gpus"].append({
                "name": props.name,
                "memory_total_gb": round(props.total_memory / (1024**3), 2)
            })
    else:
        features["gpus"] = None

    return features


def estimate_total_time(epoch_duration, num_epochs):
    """
    Given the duration of one epoch, estimate the total time for all epochs.
    Returns a formatted string (HH:MM:SS or MM:SS).
    """
    total_seconds = epoch_duration * num_epochs
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    
    if hours > 0:
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    else:
        return f"{int(minutes):02}:{int(seconds):02}"

