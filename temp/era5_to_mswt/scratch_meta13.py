import xarray as xr

def check_lmdz250_t2m():
    path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_250/LMDZOR_250/t2m-hist1.nc"
    ds = xr.open_dataset(path)
    var = 't2m'
    da = ds[var]
    
    print(f"full file path: {path}")
    print(f"variable name: {var}")
    print(f"long_name: {da.attrs.get('long_name', 'N/A')}")
    print(f"units: {da.attrs.get('units', 'N/A')}")
    print(f"dimensions: {da.dims}")
    t_dim = [d for d in da.dims if 'time' in d][0]
    print(f"period: {da[t_dim].min().values} to {da[t_dim].max().values}")
    print(f"lat/lon dimensions: {len(da.lat)}x{len(da.lon)}")
    print(f"source simulation/configuration: {ds.attrs.get('title', 'N/A')} / {ds.attrs.get('source', 'N/A')}")

if __name__ == "__main__":
    check_lmdz250_t2m()
