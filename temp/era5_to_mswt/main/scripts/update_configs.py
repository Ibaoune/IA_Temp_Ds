import os
import yaml
import glob

# Paths
CONFIG_DIR = "/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/main/configs/cnn"

NEW_PATHS = {
    "root_dir": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5",
    "results_dir": "/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/main/results/cnn",
    "shapefile_path": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/shapefiles/Morocco_shpfile/DA_REGIONS_12R.shp",
    "era5_predictor_pattern": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/reanalysis/era5/{var}_1979-2020_levels.nc",
    "mswt_path": "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswt_1979_2020.nc"
}

def update_configs():
    config_files = glob.glob(os.path.join(CONFIG_DIR, "**/*.yaml"), recursive=True)
    if not config_files:
        print(f"No config files found in {CONFIG_DIR}")
        return

    for filepath in config_files:
        print(f"Updating {filepath}...")
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'paths' not in config:
                config['paths'] = {}
                
            for key, value in NEW_PATHS.items():
                config['paths'][key] = value
                
            # Remove any old mswep_path if it exists to avoid confusion
            if 'mswep_path' in config['paths']:
                del config['paths']['mswep_path']
                
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            print(f"  Successfully updated {filepath}")
        except Exception as e:
            print(f"  Error updating {filepath}: {e}")

if __name__ == "__main__":
    update_configs()
