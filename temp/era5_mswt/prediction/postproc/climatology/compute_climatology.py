from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..common import (
    PRED_CONFIG,
    add_stat_attrs,
    finite_stats,
    get_metric_data_dir,
    load_aligned_comparison_data,
    load_prediction_settings,
    valid_pixel_count,
)


def _source_stats(mean_period: xr.DataArray) -> dict:
    stats = finite_stats(mean_period)
    return {
        "mean_temp": stats["mean"],
        "min_temp": stats["min"],
        "max_temp": stats["max"],
    }


def main() -> None:
    settings = load_prediction_settings()
    data_dir = get_metric_data_dir(settings, "climatology")

    print("=== Computing climatology diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Reference file    : {settings.reference_path}")
    print(f"Prediction dir    : {settings.prediction_dir}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Prediction mode   : {settings.prediction_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Output dir        : {data_dir}")

    if not settings.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {settings.reference_path}")

    sources = load_aligned_comparison_data(settings)
    print("Models found      :", [source["label"] for source in sources])

    data_vars = {}
    rows = []

    for source in sources:
        label = source["label"]
        key = source["key"]
        da = source["data"]
        ref = source["reference"]
        initial_pixels = valid_pixel_count(source.get("initial_spatial_mask", source["spatial_mask"]))
        common_pixels = valid_pixel_count(source["spatial_mask"])
        mean_period = da.mean(dim="time", skipna=True)
        ref_mean_period = ref.mean(dim="time", skipna=True)

        data_vars[f"{key}_mean_period"] = mean_period
        data_vars[f"{key}_spatial_mean_period"] = mean_period.mean(dim=("lat", "lon"), skipna=True)
        data_vars[f"{key}_spatial_min_period"] = mean_period.min(dim=("lat", "lon"), skipna=True)
        data_vars[f"{key}_spatial_max_period"] = mean_period.max(dim=("lat", "lon"), skipna=True)

        bias_mean = 0.0
        abs_bias_mean = 0.0
        if source["kind"] == "prediction":
            bias_mean_period = mean_period - ref_mean_period
            abs_bias_mean_period = np.abs(bias_mean_period)
            data_vars[f"{key}_bias_mean_period"] = bias_mean_period
            data_vars[f"{key}_abs_bias_mean_period"] = abs_bias_mean_period
            bias_mean = finite_stats(bias_mean_period)["mean"]
            abs_bias_mean = finite_stats(abs_bias_mean_period)["mean"]

        row = {
            "model": label,
            "start_date": settings.selected_start,
            "end_date": settings.selected_end,
            "eval_domain": settings.eval_domain,
            "display_domain": settings.display_domain,
            "initial_spatial_pixels": initial_pixels,
            "common_valid_pixels": common_pixels,
            **_source_stats(mean_period),
            "bias_mean": bias_mean,
            "abs_bias_mean": abs_bias_mean,
        }
        rows.append(row)

        print(
            f"  {label}: time={da.sizes.get('time', 0)}, "
            f"initial_pixels={initial_pixels}, common_pixels={common_pixels}"
        )

    ds_out = xr.Dataset(data_vars)
    ds_out.attrs["start_date"] = settings.selected_start
    ds_out.attrs["end_date"] = settings.selected_end
    ds_out.attrs["bc_mode"] = settings.output_tag
    ds_out.attrs["prediction_mode"] = settings.prediction_tag
    ds_out.attrs["eval_domain"] = settings.eval_domain
    ds_out.attrs["display_domain"] = settings.display_domain
    ds_out.attrs["metric_name"] = "climatology"
    ds_out.attrs["reference_label"] = "MSWT"
    ds_out.attrs["units"] = str(sources[0]["data"].attrs.get("units", "degree_Celsius"))
    ds_out.attrs["models_found"] = ",".join(source["label"] for source in sources)
    ds_out.attrs["global_common_pixels"] = valid_pixel_count(sources[0]["spatial_mask"]) if sources else 0
    for source in sources:
        path = source.get("path")
        if path is not None:
            ds_out.attrs[f"source_file_{source['key']}"] = str(path)
        add_stat_attrs(ds_out, f"{source['key']}_mean_period", data_vars[f"{source['key']}_mean_period"])

    out_nc = data_dir / (
        f"climatology_bundle_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.nc"
    )
    ds_out.to_netcdf(out_nc)
    print(f"[SUCCESS] Bundle saved to: {out_nc}")

    df = pd.DataFrame(rows)
    out_csv = data_dir / (
        f"climatology_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Summary CSV saved to: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
