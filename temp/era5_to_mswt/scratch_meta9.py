import xarray as xr
import os

def check_targets():
    paths = {
        "MSWT": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/observations/mswt_1979_2020.nc",
        "MSWEP": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/observations/mswep/mswep_1979_2020_daily.nc"
    }
    
    # Try alternate name for mswep if the first doesn't exist
    if not os.path.exists(paths["MSWEP"]):
        alt = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/observations/mswep_1979_2020.nc"
        if os.path.exists(alt):
            paths["MSWEP"] = alt
            
    # Try one more alternate
    if not os.path.exists(paths["MSWEP"]):
        paths["MSWEP"] = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/observations/mswep/mswep_1979_2020.nc"
        
    for name, path in paths.items():
        print(f"\n--- {name} ---")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            # Do a quick directory list
            dir_name = os.path.dirname(path)
            if os.path.exists(dir_name):
                print(f"Contents of {dir_name}:")
                for f in os.listdir(dir_name):
                    if 'mswep' in f.lower() or 'mswt' in f.lower():
                        print("  ", f)
            continue
            
        ds = xr.open_dataset(path)
        print("Dims:", ds.dims)
        for c in ['lat', 'lon', 'latitude', 'longitude']:
            if c in ds.coords:
                print(f"{c} min/max: {ds[c].min().values:.3f}, {ds[c].max().values:.3f} | diff: {ds[c].diff(c).mean().values:.4f}")
        
        t_dim = [d for d in ['time', 'time_counter'] if d in ds.coords][0]
        print(f"Time start/end: {ds[t_dim].min().values} to {ds[t_dim].max().values}")
        print("Variables:", list(ds.data_vars.keys()))

if __name__ == "__main__":
    check_targets()
