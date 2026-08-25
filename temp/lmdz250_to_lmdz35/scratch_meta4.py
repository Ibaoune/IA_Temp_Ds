import xarray as xr
import sys
try:
    ds = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswt_1979_2020.nc")
    print(f"Dims: {ds.dims}")
    print(f"Vars: {list(ds.data_vars)}")
    if 'lon' in ds:
        print(f"lon min/max: {ds.lon.min().values}, {ds.lon.max().values}")
        print(f"lon diff: {ds.lon.diff('lon').mean().values}")
    if 'lat' in ds:
        print(f"lat min/max: {ds.lat.min().values}, {ds.lat.max().values}")
        print(f"lat diff: {ds.lat.diff('lat').mean().values}")
except Exception as e:
    print(f"Failed to load MSWT: {e}")
