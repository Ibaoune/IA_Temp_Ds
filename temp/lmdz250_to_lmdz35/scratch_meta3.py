import xarray as xr
import numpy as np

def check_grid(path):
    ds = xr.open_dataset(path)
    lon = ds.lon.values
    lat = ds.lat.values
    
    lon_diff = np.diff(lon)
    lat_diff = np.diff(lat)
    
    print(f"File: {path}")
    print(f"Lon diff min/max: {lon_diff.min():.5f} / {lon_diff.max():.5f}")
    print(f"Lat diff min/max: {lat_diff.min():.5f} / {lat_diff.max():.5f}")

check_grid("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r250/temp-hist.nc")
check_grid("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r35/temp-hist.nc")
