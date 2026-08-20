from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..common import (
    PRED_CONFIG,
    get_metric_data_dir,
    load_aligned_comparison_data,
    load_prediction_settings,
    valid_pixel_count,
)


MONTHS = list(range(1, 13))


def _monthly_mean_2d(da: xr.DataArray) -> xr.DataArray:
    maps = []
    template = da.isel(time=0, drop=True)

    for month in MONTHS:
        month_da = da.where(da["time"].dt.month == month, drop=True)
        if month_da.sizes.get("time", 0) == 0:
            mean_map = template * np.nan
        else:
            mean_map = month_da.mean(dim="time", skipna=True).load()
        maps.append(mean_map)

    return xr.concat(maps, dim=xr.IndexVariable("month", MONTHS))


def _as_float(da: xr.DataArray) -> float:
    value = da.item()
    return float(value) if value is not None else np.nan


def main() -> None:
    settings = load_prediction_settings()
    data_dir = get_metric_data_dir(settings, "monthly_cycle")

    print("=== Computing monthly cycle diagnostics ===")
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

        monthly_mean = _monthly_mean_2d(da)
        ref_monthly_mean = _monthly_mean_2d(ref)
        spatial_monthly_mean = monthly_mean.mean(dim=("lat", "lon"), skipna=True)

        if source["kind"] == "prediction":
            monthly_bias = spatial_monthly_mean - ref_monthly_mean.mean(dim=("lat", "lon"), skipna=True)
            monthly_rmse_clim = np.sqrt(
                ((monthly_mean - ref_monthly_mean) ** 2).mean(dim=("lat", "lon"), skipna=True)
            )
            data_vars[f"{key}_monthly_bias"] = monthly_bias
            data_vars[f"{key}_monthly_rmse_clim"] = monthly_rmse_clim
        else:
            monthly_bias = xr.zeros_like(spatial_monthly_mean)
            monthly_rmse_clim = xr.zeros_like(spatial_monthly_mean)

        data_vars[f"{key}_monthly_mean"] = monthly_mean
        data_vars[f"{key}_spatial_monthly_mean"] = spatial_monthly_mean

        for month in MONTHS:
            rows.append(
                {
                    "model": label,
                    "month": month,
                    "mean_temperature": _as_float(spatial_monthly_mean.sel(month=month)),
                    "bias": _as_float(monthly_bias.sel(month=month)),
                    "rmse_clim": _as_float(monthly_rmse_clim.sel(month=month)),
                    "eval_domain": settings.eval_domain,
                    "display_domain": settings.display_domain,
                    "initial_spatial_pixels": initial_pixels,
                    "common_valid_pixels": common_pixels,
                }
            )

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
    ds_out.attrs["metric_name"] = "monthly_cycle"
    ds_out.attrs["reference_label"] = "MSWT"
    ds_out.attrs["units"] = str(sources[0]["data"].attrs.get("units", "degree_Celsius"))
    ds_out.attrs["models_found"] = ",".join(source["label"] for source in sources)
    ds_out.attrs["global_common_pixels"] = valid_pixel_count(sources[0]["spatial_mask"]) if sources else 0
    for source in sources:
        path = source.get("path")
        if path is not None:
            ds_out.attrs[f"source_file_{source['key']}"] = str(path)

    out_nc = data_dir / (
        f"monthly_cycle_bundle_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.nc"
    )
    ds_out.to_netcdf(out_nc)
    print(f"[SUCCESS] Bundle saved to: {out_nc}")

    df = pd.DataFrame(rows)
    out_csv = data_dir / (
        f"monthly_cycle_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Summary CSV saved to: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
