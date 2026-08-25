import xarray as xr
import sys

def print_meta(path, name):
    try:
        ds = xr.open_dataset(path)
        print(f"=== {name} ===")
        print(f"Coords: {list(ds.coords)}")
        for c in ds.coords:
            print(f"  {c}: shape={ds[c].shape}, type={ds[c].dtype}, vals={ds[c].values[:2]}...")
        print(f"Dims: {ds.dims}")
        print(f"Vars: {list(ds.data_vars)}")
        for v in ds.data_vars:
            print(f"  {v}: shape={ds[v].shape}, units={ds[v].attrs.get('units')}")
        ds.close()
    except Exception as e:
        print(f"Error reading {path}: {e}")

print_meta("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r250/temp-hist.nc", "LMDZ250 Temp")
print_meta("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r35/temp-hist.nc", "LMDZ35 Temp")
