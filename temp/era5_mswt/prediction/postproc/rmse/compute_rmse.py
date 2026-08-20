from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

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
) -> xr.Dataset:
    error = pred - obs
    squared_error = error ** 2

    rmse_mean_period = np.sqrt(squared_error.mean(dim="time", skipna=True))
    obs_mean_period = obs.mean(dim="time", skipna=True)
    pred_mean_period = pred.mean(dim="time", skipna=True)

    rmse_annual = np.sqrt(squared_error.groupby("time.year").mean(dim="time", skipna=True))
    annual_rmse_series = rmse_annual.mean(dim=("lat", "lon"), skipna=True)

    ds_out = xr.Dataset(
        {
            "obs_mean_period": obs_mean_period,
            "pred_mean_period": pred_mean_period,
            "rmse_mean_period": rmse_mean_period,
            "annual_rmse_series": annual_rmse_series,
        }
    )

    ds_out.attrs["model_label"] = label
    ds_out.attrs["start_date"] = start
    ds_out.attrs["end_date"] = end
    ds_out.attrs["bc_mode"] = output_tag
    ds_out.attrs["eval_domain"] = eval_domain
    ds_out.attrs["display_domain"] = display_domain
    ds_out.attrs["metric_name"] = "rmse"
    ds_out.attrs["rmse_definition"] = "sqrt(mean((prediction - observation)^2))"
    ds_out.attrs["units"] = str(obs.attrs.get("units", "degree_Celsius"))
    ds_out.attrs["source_prediction_file"] = source_prediction_file
    ds_out.attrs["valid_spatial_pixels"] = valid_pixel_count(spatial_mask)
    add_stat_attrs(ds_out, "rmse", rmse_mean_period)
    add_stat_attrs(ds_out, "annual_rmse", annual_rmse_series)

    return ds_out


def summarize_bundle(ds: xr.Dataset) -> dict:
    rmse_stats = finite_stats(ds["rmse_mean_period"])
    annual_stats = finite_stats(ds["annual_rmse_series"])

    return {
        "model": ds.attrs.get("model_label", "UNKNOWN"),
        "start_date": ds.attrs.get("start_date", ""),
        "end_date": ds.attrs.get("end_date", ""),
        "bc_mode": ds.attrs.get("bc_mode", ""),
        "eval_domain": ds.attrs.get("eval_domain", ""),
        "display_domain": ds.attrs.get("display_domain", ""),
        "valid_spatial_pixels": ds.attrs.get("valid_spatial_pixels", np.nan),
        "source_prediction_file": ds.attrs.get("source_prediction_file", ""),
        "rmse_min": rmse_stats["min"],
        "rmse_mean": rmse_stats["mean"],
        "rmse_max": rmse_stats["max"],
        "annual_rmse_mean": annual_stats["mean"],
        "annual_rmse_min": annual_stats["min"],
        "annual_rmse_max": annual_stats["max"],
    }


def main() -> None:
    settings = load_prediction_settings()
    output_dir = get_metric_data_dir(settings, "rmse")

    print("=== Computing temperature RMSE diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Reference file    : {settings.reference_path}")
    print(f"Prediction dir    : {settings.prediction_dir}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Prediction mode   : {settings.prediction_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
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
        )

        out_nc = (
            output_dir
            / f"{label.lower()}_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}_rmse_bundle.nc"
        )
        bundle.to_netcdf(out_nc)
        print(f"  -> saved bundle: {out_nc}")

        rows.append(summarize_bundle(bundle))

    if rows:
        df = pd.DataFrame(rows)
        out_csv = output_dir / f"rmse_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Summary CSV saved to: {out_csv}")
        print(df.to_string(index=False))
    else:
        print("\n[WARNING] No bundles were produced.")


if __name__ == "__main__":
    main()
