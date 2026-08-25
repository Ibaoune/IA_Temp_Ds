import xarray as xr
import numpy as np

def run_diagnostics():
    out = []
    
    # 1. Target Grid Comparison
    lmdz35_atm = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/precip-hist.nc"
    lmdz35_et = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/evap_1979_2014_MA.nc"
    
    ds_atm = xr.open_dataset(lmdz35_atm)
    ds_et = xr.open_dataset(lmdz35_et)
    
    out.append("--- Grid Comparison ---")
    out.append(f"ATM: {ds_atm.dims}, lat: {ds_atm.lat.min().values:.4f} to {ds_atm.lat.max().values:.4f}, lon: {ds_atm.lon.min().values:.4f} to {ds_atm.lon.max().values:.4f}")
    out.append(f"ET:  {ds_et.dims}, lat: {ds_et.lat.min().values:.4f} to {ds_et.lat.max().values:.4f}, lon: {ds_et.lon.min().values:.4f} to {ds_et.lon.max().values:.4f}")
    
    # Check if ET is exact subset
    lat_atm = ds_atm.lat.values
    lon_atm = ds_atm.lon.values
    lat_et = ds_et.lat.values
    lon_et = ds_et.lon.values
    
    lat_match = np.in1d(lat_et, lat_atm)
    lon_match = np.in1d(lon_et, lon_atm)
    
    if np.all(lat_match) and np.all(lon_match):
        out.append("ET grid is an EXACT coordinate subset of ATM.")
        lat_start_idx = np.where(lat_atm == lat_et[0])[0][0]
        lon_start_idx = np.where(lon_atm == lon_et[0])[0][0]
        out.append(f"Index mapping: ATM[lat={lat_start_idx}:{lat_start_idx+len(lat_et)}, lon={lon_start_idx}:{lon_start_idx+len(lon_et)}] matches ET grid.")
    else:
        out.append("ET grid is NOT an exact coordinate subset of ATM.")
        out.append(f"ET lat: {lat_et[:3]}...{lat_et[-3:]}, spacing: {lat_et[1]-lat_et[0]:.4f}")
        out.append(f"ATM lat: {lat_atm[:3]}...{lat_atm[-3:]}, spacing: {lat_atm[1]-lat_atm[0]:.4f}")
        
    # 2. Static Variables Audit (using already extracted files in evap_staging or LMDZOR_35)
    out.append("\n--- Static Fields Audit ---")
    albedo = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/albedo_glob-hist1.nc")
    lai = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/LAImean-hist1.nc")
    out.append(f"Albedo dims: {albedo.dims}")
    out.append(f"LAI dims: {lai.dims}")
    
    if 'time_counter' in albedo.dims:
        out.append(f"Albedo is DYNAMIC: {albedo.time_counter.size} timesteps")
    if 'time_counter' in lai.dims:
        out.append(f"LAI is DYNAMIC: {lai.time_counter.size} timesteps")

    # 3. ET Distribution (convert to mm/day first)
    out.append("\n--- ET Target Distribution ---")
    # Training period: 1979
    et_train = ds_et['evap'].sel(time_counter=slice('1979-01-01', '1979-12-31')).values
    et_train_mm = et_train * 86400.0  # Convert to mm/day
    
    out.append(f"Raw mean (kg/m2/s): {np.nanmean(et_train)}")
    out.append(f"Raw min (kg/m2/s): {np.nanmin(et_train)}")
    out.append(f"Raw max (kg/m2/s): {np.nanmax(et_train)}")
    
    valid_et = et_train_mm[~np.isnan(et_train_mm)]
    total = len(valid_et)
    
    out.append(f"Samples (1979): {total}")
    out.append(f"Mean (mm/day): {np.mean(valid_et):.4f}")
    out.append(f"Median: {np.median(valid_et):.4f}")
    out.append(f"StdDev: {np.std(valid_et):.4f}")
    out.append(f"Exactly 0.0: {(valid_et == 0).sum() / total * 100:.2f}%")
    out.append(f"< 0: {(valid_et < 0).sum() / total * 100:.2f}%")
    out.append(f"< 0.01: {(valid_et < 0.01).sum() / total * 100:.2f}%")
    out.append(f"< 0.1: {(valid_et < 0.1).sum() / total * 100:.2f}%")
    
    perc = [1, 5, 25, 75, 95, 99]
    pv = np.percentile(valid_et, perc)
    for p, v in zip(perc, pv):
        out.append(f"P{p:02d}: {v:.4f}")
    out.append(f"Max: {np.max(valid_et):.4f}")

    with open("/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/qc_report.txt", "w") as f:
        f.write("\n".join(out))
    
    print("\n".join(out))

if __name__ == "__main__":
    run_diagnostics()
