import xarray as xr
import os

def check_lmdz_et():
    paths = {
        "LMDZ250_evap": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_250/LMDZOR_250/evap-hist1.nc",
        "LMDZ35_evap_MA": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/evap_1979_2014_MA.nc",
        "LMDZ35_evap_hist": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/evap-hist1.nc",
    }
    
    for name, path in paths.items():
        print(f"\n--- {name} ---")
        if not os.path.exists(path):
            print("File not found:", path)
            continue
        try:
            ds = xr.open_dataset(path)
            # Find evap variable
            varname = [v for v in ds.data_vars if 'evap' in v.lower()]
            if not varname:
                print("No 'evap' variable found, variables are:", list(ds.data_vars.keys()))
                continue
            v = varname[0]
            da = ds[v]
            
            print(f"file: {path.split('/')[-1]}")
            print(f"variable: {v}")
            print(f"long_name: {da.attrs.get('long_name', 'N/A')}")
            print(f"units: {da.attrs.get('units', 'N/A')}")
            print(f"dimensions: {da.dims}")
            
            t_dim = [d for d in da.dims if 'time' in d][0]
            print(f"period: {da[t_dim].min().values} to {da[t_dim].max().values}")
            
            print(f"grid: {len(da.lat)}x{len(da.lon)} (lat: {da.lat.min().values:.2f} to {da.lat.max().values:.2f}, lon: {da.lon.min().values:.2f} to {da.lon.max().values:.2f})")
            
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    check_lmdz_et()
