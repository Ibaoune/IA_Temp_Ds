import xarray as xr
try:
    ds = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5/z_1979-2020_levels.nc")
    print(f"Dims: {ds.dims}")
    if 'longitude' in ds:
        print(f"lon min/max: {ds.longitude.min().values}, {ds.longitude.max().values}")
        print(f"lon diff: {ds.longitude.diff('longitude').mean().values}")
    if 'latitude' in ds:
        print(f"lat min/max: {ds.latitude.min().values}, {ds.latitude.max().values}")
        print(f"lat diff: {ds.latitude.diff('latitude').mean().values}")
except Exception as e:
    print(f"Failed to load ERA5: {e}")
