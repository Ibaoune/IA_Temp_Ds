import os
import subprocess
from pathlib import Path
import yaml

POSTPROC_DIR = Path(__file__).resolve().parent
MAIN_DIR = POSTPROC_DIR.parent / "main"
CONFIGS_DIR = MAIN_DIR / "configs"

# Find all metric configs
metric_configs = []
for root, _, files in os.walk(POSTPROC_DIR):
    if "explore_config.yaml" in files:
        files.remove("explore_config.yaml") # skip this one
    for file in files:
        if file.endswith("config.yaml"):
            metric_configs.append(Path(root) / file)

print(f"Found {len(metric_configs)} metric configurations.")

# Find all model configs
model_configs = []
for arch in ["cnn", "unet"]:
    arch_dir = CONFIGS_DIR / arch
    if arch_dir.exists():
        for file in arch_dir.glob("*.yaml"):
            model_configs.append(file)

print(f"Found {len(model_configs)} model configurations to process.")

for model_cfg in model_configs:
    print(f"\n{'='*50}")
    print(f"Processing model: {model_cfg.name}")
    print(f"{'='*50}")
    
    # Update all metric configs to point to this model config
    for mcfg in metric_configs:
        with open(mcfg, "r") as f:
            lines = f.readlines()
            
        with open(mcfg, "w") as f:
            for line in lines:
                if "main_config_path:" in line:
                    # Replace with absolute path to avoid PROJECT_ROOT mismatch
                    f.write(f'  main_config_path: "{model_cfg.resolve()}"\n')
                else:
                    f.write(line)
                    
    # Run the bash script
    run_script = POSTPROC_DIR / "run_postproc.sh"
    subprocess.run(["bash", str(run_script), "config.yaml", "all"], cwd=POSTPROC_DIR)

print("\nFinished processing all models.")
