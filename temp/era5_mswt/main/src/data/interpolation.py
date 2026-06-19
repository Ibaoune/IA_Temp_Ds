"""
==========================================================
 Script: interpolation.py
 Description:
     Contains functions to interpolate predictor datasets
     (e.g., ERA5 or LMDZ) to a 2-degree grid.
==========================================================
"""

import numpy as np
import xarray as xr

def interpolate_to_target_resolution(ds, resolution=2.0, lon_name="lon", lat_name="lat", method="linear", bounds=None):
    """
    Interpolates an xarray Dataset or DataArray to a target degree resolution.

    Args:
        ds: xarray.Dataset or xarray.DataArray to be interpolated.
        resolution: Target grid resolution in degrees.
        lon_name: Name of the longitude coordinate.
        lat_name: Name of the latitude coordinate.
        method: Interpolation method ('linear', 'nearest').
        bounds: Optional tuple (min_lon, max_lon, min_lat, max_lat). If provided,
                the target grid is anchored to the floor/ceil of these bounds.
    """
    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise ValueError(f"Coordinates '{lon_name}' and/or '{lat_name}' not found in the dataset.")

    if bounds:
        min_lon, max_lon, min_lat, max_lat = bounds
    else:
        # Fallback to current dataset extent
        min_lon = float(ds[lon_name].min().values)
        max_lon = float(ds[lon_name].max().values)
        min_lat = float(ds[lat_name].min().values)
        max_lat = float(ds[lat_name].max().values)

    # Create the new coordinates covering the bounding box
    # Using floor/ceil ensures a consistent grid across different source datasets.
    new_lon = np.arange(np.floor(min_lon), np.ceil(max_lon) + (resolution / 10.0), resolution)
    new_lat = np.arange(np.floor(min_lat), np.ceil(max_lat) + (resolution / 10.0), resolution)

    # Interpolate using xarray select/interp
    if method == "nearest":
        ds_interpolated = ds.sel({lon_name: new_lon, lat_name: new_lat}, method="nearest")
        # Ensure coordinates explicitly match target resolution points exactly
        ds_interpolated = ds_interpolated.assign_coords({lon_name: new_lon, lat_name: new_lat})
    else:
        ds_interpolated = ds.interp(
            {lon_name: new_lon, lat_name: new_lat},
            method=method,
            kwargs={"fill_value": "extrapolate"}
        )

    return ds_interpolated
