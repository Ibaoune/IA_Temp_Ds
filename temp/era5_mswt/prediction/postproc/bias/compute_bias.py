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
    obs_mean_period = obs.mean(dim="time", skipna=True)
    pred_mean_period = pred.mean(dim="time", skipna=True)
    bias_mean_period = pred_mean_period - obs_mean_period

    obs_annual = obs.groupby("time.year").mean(dim="time", skipna=True)
    pred_annual = pred.groupby("time.year").mean(dim="time", skipna=True)
    bias_annual = pred_annual - obs_annual

    annual_obs_series = obs_annual.mean(dim=("lat", "lon"), skipna=True)
    annual_pred_series = pred_annual.mean(dim=("lat", "lon"), skipna=True)
    annual_bias_series = bias_annual.mean(dim=("lat", "lon"), skipna=True)

    ds_out = xr.Dataset(
        {
            "obs_mean_period": obs_mean_period,
            "pred_mean_period": pred_mean_period,
            "bias_mean_period": bias_mean_period,
            "annual_obs_series": annual_obs_series,
            "annual_pred_series": annual_pred_series,
            "annual_bias_series": annual_bias_series,
        }
    )

    ds_out.attrs["model_label"] = label
    ds_out.attrs["start_date"] = start
    ds_out.attrs["end_date"] = end
    ds_out.attrs["bc_mode"] = output_tag
    ds_out.attrs["eval_domain"] = eval_domain
    ds_out.attrs["display_domain"] = display_domain
    ds_out.attrs["metric_name"] = "bias"
    ds_out.attrs["bias_definition"] = "prediction - observation"
    ds_out.attrs["units"] = str(obs.attrs.get("units", "degree_Celsius"))
    ds_out.attrs["source_prediction_file"] = source_prediction_file
    ds_out.attrs["valid_spatial_pixels"] = valid_pixel_count(spatial_mask)
    add_stat_attrs(ds_out, "bias", bias_mean_period)
    add_stat_attrs(ds_out, "annual_bias", annual_bias_series)

    return ds_out


def summarize_bundle(ds: xr.Dataset) -> dict:
    bias_stats = finite_stats(ds["bias_mean_period"])
    annual_stats = finite_stats(ds["annual_bias_series"])

    return {
        "model": ds.attrs.get("model_label", "UNKNOWN"),
        "start_date": ds.attrs.get("start_date", ""),
        "end_date": ds.attrs.get("end_date", ""),
        "bc_mode": ds.attrs.get("bc_mode", ""),
        "eval_domain": ds.attrs.get("eval_domain", ""),
        "display_domain": ds.attrs.get("display_domain", ""),
        "valid_spatial_pixels": ds.attrs.get("valid_spatial_pixels", np.nan),
        "source_prediction_file": ds.attrs.get("source_prediction_file", ""),
        "bias_min": bias_stats["min"],
        "bias_mean": bias_stats["mean"],
        "bias_max": bias_stats["max"],
        "annual_bias_mean": annual_stats["mean"],
        "annual_bias_min": annual_stats["min"],
        "annual_bias_max": annual_stats["max"],
    }


def main() -> None:
    settings = load_prediction_settings()
    output_dir = get_metric_data_dir(settings, "bias")

    print("=== Computing temperature bias diagnostics ===")
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

        print(f"  Prediction time before align: {pred.time.values[0]} -> {pred.time.values[-1]}")
        print(f"  Observation time before align: {obs.time.values[0]} -> {obs.time.values[-1]}")

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

        obs_mean_period = obs.mean(dim="time", skipna=True)
        pred_mean_period = pred.mean(dim="time", skipna=True)
        bias_mean_period = pred_mean_period - obs_mean_period
        mean_delta = finite_stats(pred_mean_period)["mean"] - finite_stats(obs_mean_period)["mean"]
        mean_bias_map = finite_stats(bias_mean_period)["mean"]
        consistency_diff = mean_delta - mean_bias_map

        print(f"  Common time steps after align: {pred.sizes['time']}")
        print(f"  Initial spatial pixels: {initial_spatial_pixels}")
        print(f"  Common valid pixels  : {valid_pixel_count(spatial_mask)}")
        print(f"  mean_pred - mean_obs : {mean_delta:.12g}")
        print(f"  mean_bias_map        : {mean_bias_map:.12g}")
        print(f"  consistency diff     : {consistency_diff:.12g}")

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
            / f"{label.lower()}_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}_bias_bundle.nc"
        )
        bundle.to_netcdf(out_nc)
        print(f"  -> saved bundle: {out_nc}")

        rows.append(summarize_bundle(bundle))

    if rows:
        df = pd.DataFrame(rows)
        out_csv = output_dir / f"bias_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Summary CSV saved to: {out_csv}")
        print(df.to_string(index=False))
    else:
        print("\n[WARNING] No bundles were produced.")


if __name__ == "__main__":
    main()
