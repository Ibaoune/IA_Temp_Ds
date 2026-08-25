import os
import yaml
import glob
import subprocess

CONFIG_DIR = "/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/main/configs/cnn"

def check_data(config):
    paths = config.get("paths", {})
    variables = config.get("general", {}).get("variables", ["z", "q", "t", "u", "v"])
    
    mswt_path = paths.get("mswt_path", "")
    era5_pattern = paths.get("era5_predictor_pattern", "")
    shapefile = paths.get("shapefile_path", "")
    
    missing = []
    
    if not os.path.exists(mswt_path):
        missing.append(f"MSWT Predictand: {mswt_path}")
        
    if not os.path.exists(shapefile):
        missing.append(f"Shapefile: {shapefile}")
        
    for var in variables:
        era5_file = era5_pattern.replace("{var}", var)
        if not os.path.exists(era5_file):
            missing.append(f"ERA5 Predictor ({var}): {era5_file}")
            
    return missing

def main():
    config_files = glob.glob(os.path.join(CONFIG_DIR, "**/*.yaml"), recursive=True)
    if not config_files:
        print(f"No config files found in {CONFIG_DIR}")
        return

    configs_to_submit = []

    for filepath in config_files:
        print(f"--- Checking {filepath} ---")
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            
            missing_files = check_data(config)
            
            if missing_files:
                print("  [FAILED] Missing data:")
                for m in missing_files:
                    print(f"    - {m}")
            else:
                print("  [OK] All data available.")
                configs_to_submit.append(filepath)
                
        except Exception as e:
            print(f"  [ERROR] Failed to read {filepath}: {e}")

    print("\n==========================================")
    print(f"Submitting {len(configs_to_submit)} jobs to CPU...")
    print("==========================================\n")
    
    for cfg in configs_to_submit:
        print(f"Submitting: {cfg}")
        # Make the config path relative to main/ directory since job runs there, 
        # or just pass the absolute path (which we have).
        cmd = ["sbatch", "/srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/main/scripts/job_train_cpu.sh", cfg]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  Success: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e.stderr.strip()}")

if __name__ == "__main__":
    main()
