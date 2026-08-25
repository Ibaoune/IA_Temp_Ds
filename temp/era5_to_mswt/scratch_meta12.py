import xarray as xr
import os
import glob

def search_lmdz35_vars():
    search_dir = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/Intern_Hamza/LMDZOR_35/LMDZOR_35/"
    nc_files = glob.glob(os.path.join(search_dir, "*.nc"))
    
    found = False
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            vars = list(ds.data_vars.keys())
            for v in vars:
                vl = v.lower()
                if 't2m' in vl or 'tas' in vl or 'tair' in vl or 'temp' in vl:
                    print(f"File: {os.path.basename(f)} -> Variable: {v} (long_name: {ds[v].attrs.get('long_name', '')})")
                    found = True
        except:
            pass
            
    if not found:
        print("No temperature-related variables found in LMDZOR_35.")

if __name__ == "__main__":
    search_lmdz35_vars()
