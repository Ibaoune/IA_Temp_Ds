import xarray as xr

def check_targets():
    paths = {
        "MSWT": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswt_1979_2020.nc",
        "MSWEP": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc"
    }
    
    for name, path in paths.items():
        print(f"\n--- {name} ---")
        try:
            ds = xr.open_dataset(path)
            print("Dims:", ds.dims)
            for c in ['lat', 'lon', 'latitude', 'longitude']:
                if c in ds.coords:
                    print(f"{c} min/max: {ds[c].min().values:.3f}, {ds[c].max().values:.3f} | diff: {ds[c].diff(c).mean().values:.4f}")
            
            t_dim = [d for d in ['time', 'time_counter'] if d in ds.coords][0]
            print(f"Time start/end: {ds[t_dim].min().values} to {ds[t_dim].max().values}")
            print("Variables:", list(ds.data_vars.keys()))
        except Exception as e:
            print(f"Error reading {name}: {e}")

if __name__ == "__main__":
    check_targets()
