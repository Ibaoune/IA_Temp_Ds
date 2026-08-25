import xarray as xr
import glob
import os

def check_lmdz35_vars():
    path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r35/"
    files = glob.glob(path + "*.nc")
    print(f"Found {len(files)} netcdf files in LMDZ35 path.")
    for f in sorted(files):
        try:
            ds = xr.open_dataset(f)
            vars = list(ds.data_vars)
            
            # Print target-related details for temp/tas/t2m
            for v in vars:
                if v.lower() in ['temp', 'tas', 't2m', 'temp2m', 't2']:
                    da = ds[v]
                    dims = da.dims
                    units = da.attrs.get('units', 'None')
                    long_name = da.attrs.get('long_name', 'None')
                    std_name = da.attrs.get('standard_name', 'None')
                    print(f"[{os.path.basename(f)}] Var: {v} | dims: {dims} | units: {units} | long_name: {long_name} | standard_name: {std_name}")
            
            print(f"[{os.path.basename(f)}] All vars: {vars}")
            ds.close()
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    check_lmdz35_vars()
