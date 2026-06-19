import os
import glob
import re

src_dir = "/srv/data/mohammad.elaabaribao/work/interns/y2026/precip/ds_precip_intern/precip/era5_mswep/configs/unet/"
dst_dir = "/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/main/configs/unet/"

os.makedirs(dst_dir, exist_ok=True)

files = glob.glob(os.path.join(src_dir, "*.yaml"))

dates_block_new = """dates:
  train:
    start: '1979-01-01'
    end: '2005-12-31'
  test:
    start: '2006-01-01'
    end: '2020-12-31'"""

paths_block_new = """paths:
  # Paths for Mohammad's environment
  root_dir: "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5"
  results_dir: "/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/results/"
  shapefile_path: "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/shapefiles/Morocco_shpfile/DA_REGIONS_12R.shp"
  era5_predictor_pattern: "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5/{var}_1979-2020_levels.nc"
  mswt_path: "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswt_1979_2020.nc"
"""

for fpath in files:
    fname = os.path.basename(fpath)
    if fname == "test.yaml" or fname == "config.yaml":
        continue
    
    with open(fpath, "r") as f:
        content = f.read()

    # Replacements
    content = re.sub(r'variable:\s*["\']?precip["\']?', 'variable: "temp"', content)
    content = re.sub(r'target:\s*["\']?mswep["\']?', 'target: "mswt"', content)
    content = re.sub(r'loss_type:\s*["\']?bernoulli_gamma["\']?', 'loss_type: "gaussian"', content)
    
    # Replace dates block entirely
    content = re.sub(r'dates:.*?(?=paths:|plots:)', dates_block_new + '\n\n', content, flags=re.DOTALL)
    
    # Replace plots block if exists (precip has plots block)
    # We can just leave it or remove it, wait, `plots: ...` might exist. If so, leave it, but just replace paths block
    content = re.sub(r'paths:.*', paths_block_new, content, flags=re.DOTALL)
    
    # Make sure we didn't mangle if there is no dates block (though they all should have it)
    
    dst_fpath = os.path.join(dst_dir, fname)
    with open(dst_fpath, "w") as f:
        f.write(content)
    print(f"Adapted {fname} -> {dst_fpath}")
