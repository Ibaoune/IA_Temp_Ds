import xarray as xr

print("LMDZ250 presnivs:")
ds250 = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r250/temp-hist.nc")
print(ds250.presnivs.values)
print("Units:", ds250.presnivs.attrs.get("units"))

print("\nLMDZ35 presnivs:")
ds35 = xr.open_dataset("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/lmdz/r35/temp-hist.nc")
print(ds35.presnivs.values)
print("Units:", ds35.presnivs.attrs.get("units"))
