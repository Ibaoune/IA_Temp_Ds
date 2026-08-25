import xarray as xr

def check_lmdz35_precip():
    path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r35/precip-hist.nc"
    ds = xr.open_dataset(path)
    var = 'precip'
    if var in ds.data_vars:
        da = ds[var]
        
        print(f"full file path: {path}")
        print(f"variable name: {var}")
        print(f"units: {da.attrs.get('units', 'N/A')}")
        print(f"dimensions: {da.dims}")
        t_dim = [d for d in da.dims if 'time' in d][0]
        print(f"period: {da[t_dim].min().values} to {da[t_dim].max().values}")
        print(f"lat/lon grid: {len(da.lat)}x{len(da.lon)}")
    else:
        print(f"Variable {var} not found in {path}")

if __name__ == "__main__":
    check_lmdz35_precip()
