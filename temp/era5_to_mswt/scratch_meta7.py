import os
import glob
import xarray as xr

base_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/"

print("Searching for tas, t2m, t2 in:", base_path)
for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith('.nc'):
            lower_f = f.lower()
            if 'tas' in lower_f or 't2m' in lower_f or 'temp2' in lower_f or 't2' in lower_f:
                print(f"Found match by filename: {os.path.join(root, f)}")
                
            # Quick check inside file if it's small or just check headers
            # (We won't open every single file to save time, only print filenames first)
