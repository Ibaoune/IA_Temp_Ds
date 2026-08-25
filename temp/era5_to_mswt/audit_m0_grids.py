import xarray as xr
import numpy as np
import os
import sys

# Append paths so we can import the configs
sys.path.append("/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling/univariate/temperature/M0_era5_to_mswt/main")
sys.path.append("/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling/univariate/precipitation/M0_era5_to_mswep")

def get_grid_from_cfg(cfg_path, project):
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    gen = cfg.get("general", {})
    reg = cfg.get("region", cfg.get("general", {}))
    min_lon = float(reg.get("lon_min", cfg.get("general", {}).get("lon_min")))
    max_lon = float(reg.get("lon_max", cfg.get("general", {}).get("lon_max")))
    min_lat = float(reg.get("lat_min", cfg.get("general", {}).get("lat_min")))
    max_lat = float(reg.get("lat_max", cfg.get("general", {}).get("lat_max")))
    res = float(gen.get("resolution", 2.0))
    
    # Check interpolation logic in the actual project
    # Most interpolation uses:
    # new_lon = np.arange(np.floor(min_lon), np.ceil(max_lon) + (resolution / 10.0), resolution)
    # new_lat = np.arange(np.floor(min_lat), np.ceil(max_lat) + (resolution / 10.0), resolution)
    # Let's read the exact interpolation script
    if project == "temp":
        interp_file = "/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling/univariate/temperature/M0_era5_to_mswt/main/src/data/interpolation.py"
    else:
        interp_file = "/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling/univariate/precipitation/M0_era5_to_mswep/src/data/interpolation.py"
        
    with open(interp_file, "r") as f:
        interp_code = f.read()
    
    new_lon = np.arange(np.floor(min_lon), np.ceil(max_lon) + (res / 10.0), res)
    new_lat = np.arange(np.floor(min_lat), np.ceil(max_lat) + (res / 10.0), res)
    
    return new_lat, new_lon, min_lat, max_lat, min_lon, max_lon, res, interp_code

def run():
    print("=== TEMPERATURE M0 GRID ===")
    t_cfg = "/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling/univariate/temperature/M0_era5_to_mswt/main/configs/cnn/config.yaml"
    t_lat, t_lon, t_min_lat, t_max_lat, t_min_lon, t_max_lon, t_res, t_code = get_grid_from_cfg(t_cfg, "temp")
    print(f"lat array: {t_lat}")
    print(f"lon array: {t_lon}")
    print(f"shape: ({len(t_lat)}, {len(t_lon)})")
    print(f"cfg bounds: lat {t_min_lat}-{t_max_lat}, lon {t_min_lon}-{t_max_lon}, res {t_res}")
    
    print("\n=== PRECIPITATION M0 GRID ===")
    p_cfg = "/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling/univariate/precipitation/M0_era5_to_mswep/configs/cnn/config_cnn.yaml"
    p_lat, p_lon, p_min_lat, p_max_lat, p_min_lon, p_max_lon, p_res, p_code = get_grid_from_cfg(p_cfg, "precip")
    print(f"lat array: {p_lat}")
    print(f"lon array: {p_lon}")
    print(f"shape: ({len(p_lat)}, {len(p_lon)})")
    print(f"cfg bounds: lat {p_min_lat}-{p_max_lat}, lon {p_min_lon}-{p_max_lon}, res {p_res}")

    print(f"\nDo they match exactly? {np.array_equal(t_lat, p_lat) and np.array_equal(t_lon, p_lon)}")

    # Check native grids
    print("\n=== NATIVE GRIDS ===")
    lmdz250_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_250/LMDZOR_250/temp-hist1.nc"
    if os.path.exists(lmdz250_path):
        ds = xr.open_dataset(lmdz250_path)
        print(f"LMDZ250 shape: {ds.temp.shape}")
        print(f"LMDZ250 lat: {ds.lat.values}")
        print(f"LMDZ250 lon: {ds.lon.values}")
        print(f"LMDZ250 min/max lat: {ds.lat.min().values} to {ds.lat.max().values}, lon: {ds.lon.min().values} to {ds.lon.max().values}")
        
    lmdz35_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r35/precip-hist.nc"
    if os.path.exists(lmdz35_path):
        ds = xr.open_dataset(lmdz35_path)
        print(f"\nLMDZ35 PRECIP target shape: {ds.precip.shape}")
        print(f"LMDZ35 PRECIP lat range: {ds.lat.min().values} to {ds.lat.max().values}")
        print(f"LMDZ35 PRECIP lon range: {ds.lon.min().values} to {ds.lon.max().values}")
        p35_lat = ds.lat.values
        p35_lon = ds.lon.values
        
    lmdz35_atm = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/albedo_glob-hist1.nc"
    if os.path.exists(lmdz35_atm):
        ds_atm = xr.open_dataset(lmdz35_atm)
        print(f"\nLMDZ35 ATM shape: {ds_atm.albedo_glob.shape}")
        print(f"LMDZ35 ATM lat range: {ds_atm.lat.min().values} to {ds_atm.lat.max().values}")
        print(f"LMDZ35 ATM lon range: {ds_atm.lon.min().values} to {ds_atm.lon.max().values}")
        atm_lat = ds_atm.lat.values
        atm_lon = ds_atm.lon.values
        
        if os.path.exists(lmdz35_path):
            lat_match = np.in1d(p35_lat, atm_lat)
            lon_match = np.in1d(p35_lon, atm_lon)
            if np.all(lat_match) and np.all(lon_match):
                print(f"\nIS PRECIP35 EXACT SUBSET OF LMDZ35 ATM? YES")
                lat_start_idx = np.where(atm_lat == p35_lat[0])[0][0]
                lon_start_idx = np.where(atm_lon == p35_lon[0])[0][0]
                print(f"Indices: ATM[lat={lat_start_idx}:{lat_start_idx+len(p35_lat)}, lon={lon_start_idx}:{lon_start_idx+len(p35_lon)}]")
            else:
                print(f"\nIS PRECIP35 EXACT SUBSET OF LMDZ35 ATM? NO")

if __name__ == "__main__":
    run()
