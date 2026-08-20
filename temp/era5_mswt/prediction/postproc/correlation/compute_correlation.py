from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as student_t

from ..common import (
    PRED_CONFIG,
    SELECTED_LABELS,
    add_stat_attrs,
    align_prediction_to_obs,
    apply_common_valid_mask,
    apply_eval_domain_to_inputs,
    choose_prediction_file,
    finite_stats,
    get_metric_data_dir,
    load_prediction_settings,
    open_prediction_var,
    open_reference_var,
    subset_time,
    valid_pixel_count,
)


WINDOW_DAYS = 31
MIN_VALID = 10
ALPHA = 0.05


def drop_feb29(da: xr.DataArray) -> xr.DataArray:
    month = da["time"].dt.month
    day = da["time"].dt.day
    return da.sel(time=~((month == 2) & (day == 29)))


def build_climatological_day_index(time_values) -> np.ndarray:
    dates = pd.to_datetime(time_values)
    return np.asarray(
        [pd.Timestamp(year=2001, month=dt.month, day=dt.day).dayofyear for dt in dates],
        dtype=int,
    )


def smooth_daily_climatology(clim: xr.DataArray, window: int = 31) -> xr.DataArray:
    if window % 2 == 0:
        raise ValueError("window must be odd, e.g. 31")

    pad = window // 2

    left = clim.isel(clim_day=slice(-pad, None)).copy()
    left = left.assign_coords(clim_day=np.arange(1 - pad, 1))

    right = clim.isel(clim_day=slice(0, pad)).copy()
    right = right.assign_coords(clim_day=np.arange(366, 366 + pad))

    extended = xr.concat([left, clim, right], dim="clim_day")
    smoothed = extended.rolling(
        clim_day=window,
        center=True,
        min_periods=1,
    ).mean(skipna=True)

    return smoothed.sel(clim_day=slice(1, 365))


def deseasonalize_daily(da: xr.DataArray, window: int = 31) -> xr.DataArray:
    da = drop_feb29(da)
    clim_day = build_climatological_day_index(da["time"].values)
    da = da.assign_coords(clim_day=("time", clim_day))

    daily_clim = da.groupby("clim_day").mean("time", skipna=True)
    daily_clim = daily_clim.reindex(clim_day=np.arange(1, 366))
    daily_clim_smooth = smooth_daily_climatology(daily_clim, window=window)

    return da.groupby("clim_day") - daily_clim_smooth


def pearson_corr_pvalue_sig_maps(
    x: xr.DataArray,
    y: xr.DataArray,
    min_valid: int = 10,
    alpha: float = 0.05,
):
    valid = xr.where(np.isfinite(x) & np.isfinite(y), 1, 0)
    n_valid = valid.sum("time")

    x_valid = x.where(valid == 1)
    y_valid = y.where(valid == 1)

    x_mean = x_valid.mean("time", skipna=True)
    y_mean = y_valid.mean("time", skipna=True)

    cov = ((x_valid - x_mean) * (y_valid - y_mean)).mean("time", skipna=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice.",
            category=RuntimeWarning,
        )
        x_std = x_valid.std("time", skipna=True)
        y_std = y_valid.std("time", skipna=True)

    corr = cov / (x_std * y_std)
    corr = corr.where((n_valid >= min_valid) & (x_std > 0) & (y_std > 0))

    dof = n_valid - 2

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = corr * np.sqrt(dof / (1.0 - corr**2))

    pval = xr.apply_ufunc(
        lambda t, df: 2.0 * student_t.sf(np.abs(t), df),
        t_stat,
        dof,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    pval = pval.where((n_valid >= min_valid) & (dof > 0) & np.isfinite(corr))

    sig = (pval < alpha).astype(np.int8)
    sig = sig.where(np.isfinite(corr))

    return corr, pval, sig, n_valid


def compute_bundle(
    pred: xr.DataArray,
    obs: xr.DataArray,
    label: str,
    start: str,
    end: str,
    output_tag: str,
    source_prediction_file: str,
    eval_domain: str,
    display_domain: str,
    spatial_mask: xr.DataArray,
    window: int,
    min_valid: int,
    alpha: float,
) -> xr.Dataset:
    pred_anom = deseasonalize_daily(pred, window=window)
    obs_anom = deseasonalize_daily(obs, window=window)
    pred_anom, obs_anom = xr.align(pred_anom, obs_anom, join="inner")

    corr, pval, sig, n_valid = pearson_corr_pvalue_sig_maps(
        pred_anom,
        obs_anom,
        min_valid=min_valid,
        alpha=alpha,
    )

    corr.name = "corr_d"
    pval.name = "corr_d_pval"
    sig.name = "corr_d_sig"
    n_valid.name = "corr_d_n_valid"

    ds_out = xr.Dataset(
        {
            "corr_d": corr,
            "corr_d_pval": pval,
            "corr_d_sig": sig,
            "corr_d_n_valid": n_valid,
        }
    )

    sig_values = sig.values
    sig_fraction = float(np.nanmean(sig_values)) if np.isfinite(sig_values).any() else np.nan

    ds_out.attrs["model_label"] = label
    ds_out.attrs["start_date"] = start
    ds_out.attrs["end_date"] = end
    ds_out.attrs["bc_mode"] = output_tag
    ds_out.attrs["eval_domain"] = eval_domain
    ds_out.attrs["display_domain"] = display_domain
    ds_out.attrs["metric_name"] = "correlation"
    ds_out.attrs["correlation_definition"] = "Pearson correlation over time"
    ds_out.attrs["definition"] = "Daily Pearson correlation computed on deseasonalized anomalies"
    ds_out.attrs["window_days"] = window
    ds_out.attrs["min_valid"] = min_valid
    ds_out.attrs["alpha"] = alpha
    ds_out.attrs["source_prediction_file"] = source_prediction_file
    ds_out.attrs["valid_spatial_pixels"] = valid_pixel_count(spatial_mask)
    ds_out.attrs["significant_fraction"] = sig_fraction
    add_stat_attrs(ds_out, "correlation", corr)

    return ds_out


def summarize_bundle(ds: xr.Dataset) -> dict:
    corr_stats = finite_stats(ds["corr_d"])
    sig_values = ds["corr_d_sig"].values
    sig_fraction = float(np.nanmean(sig_values)) if np.isfinite(sig_values).any() else np.nan

    return {
        "model": ds.attrs.get("model_label", "UNKNOWN"),
        "start_date": ds.attrs.get("start_date", ""),
        "end_date": ds.attrs.get("end_date", ""),
        "bc_mode": ds.attrs.get("bc_mode", ""),
        "eval_domain": ds.attrs.get("eval_domain", ""),
        "display_domain": ds.attrs.get("display_domain", ""),
        "valid_spatial_pixels": ds.attrs.get("valid_spatial_pixels", np.nan),
        "source_prediction_file": ds.attrs.get("source_prediction_file", ""),
        "corr_min": corr_stats["min"],
        "corr_mean": corr_stats["mean"],
        "corr_max": corr_stats["max"],
        "significant_fraction": sig_fraction,
    }


def main() -> None:
    settings = load_prediction_settings()
    output_dir = get_metric_data_dir(settings, "correlation")

    print("=== Computing Bano-style anomaly correlation diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Reference file    : {settings.reference_path}")
    print(f"Prediction dir    : {settings.prediction_dir}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Prediction mode   : {settings.prediction_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Window days       : {WINDOW_DAYS}")
    print(f"Output dir        : {output_dir}")

    if not settings.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {settings.reference_path}")

    obs_full = open_reference_var(settings.reference_path, preferred_var="air_temperature")
    rows = []

    for label in SELECTED_LABELS:
        print(f"\n[MODEL] {label}")
        pred_path = choose_prediction_file(
            prediction_dir=settings.prediction_dir,
            label=label,
            selected_start=settings.selected_start,
            selected_end=settings.selected_end,
            prediction_tag=settings.prediction_tag,
        )

        if pred_path is None:
            print(
                f"  -> no prediction file found covering "
                f"{settings.selected_start} -> {settings.selected_end} with mode={settings.prediction_tag}"
            )
            continue

        print(f"  Prediction file      : {pred_path}")
        print(f"  Selected study period: {settings.selected_start} -> {settings.selected_end}")

        pred_full = open_prediction_var(pred_path, preferred_var="air_temperature")
        pred = subset_time(pred_full, settings.selected_start, settings.selected_end)
        obs = subset_time(obs_full, settings.selected_start, settings.selected_end)

        try:
            pred, obs = align_prediction_to_obs(pred, obs)
        except ValueError as exc:
            print(f"  -> {exc}, skipping")
            continue

        pred, obs, spatial_mask = apply_eval_domain_to_inputs(
            pred=pred,
            obs=obs,
            eval_domain=settings.eval_domain,
            shapefile_path=settings.shapefile_path,
            land_shapefile_path=settings.land_shapefile_path,
        )
        initial_spatial_pixels = valid_pixel_count(spatial_mask)
        obs, pred, spatial_mask = apply_common_valid_mask(
            obs=obs,
            pred=pred,
            spatial_mask=spatial_mask,
        )

        print(f"  Common time steps after align: {pred.sizes['time']}")
        print(f"  Initial spatial pixels: {initial_spatial_pixels}")
        print(f"  Common valid pixels  : {valid_pixel_count(spatial_mask)}")

        bundle = compute_bundle(
            pred=pred,
            obs=obs,
            label=label,
            start=settings.selected_start,
            end=settings.selected_end,
            output_tag=settings.output_tag,
            source_prediction_file=pred_path.name,
            eval_domain=settings.eval_domain,
            display_domain=settings.display_domain,
            spatial_mask=spatial_mask,
            window=WINDOW_DAYS,
            min_valid=MIN_VALID,
            alpha=ALPHA,
        )

        out_nc = (
            output_dir
            / f"{label.lower()}_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}_corr_d_bundle.nc"
        )
        bundle.to_netcdf(out_nc)
        print(f"  -> saved bundle: {out_nc}")

        rows.append(summarize_bundle(bundle))

    if rows:
        df = pd.DataFrame(rows)
        out_csv = output_dir / f"corr_d_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Summary CSV saved to: {out_csv}")
        print(df.to_string(index=False))
    else:
        print("\n[WARNING] No bundles were produced.")


if __name__ == "__main__":
    main()
