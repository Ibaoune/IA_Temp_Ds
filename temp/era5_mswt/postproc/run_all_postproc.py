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

# Find all model configs
model_configs = []
hardcoded_configs = [
    "cnn/config.yaml",
    "cnn/config_cnn1.yaml",
    "cnn/config_cnn1_mse.yaml",
    "cnn/config_cnn10_mse.yaml",
    "phy_ai/cnn/config_cnn1_serifi_gradient.yaml",
    "phy_ai/cnn/config_cnn1_xiong_continuity.yaml",
    "phy_ai/cnn/config_cnn1_xiong_directional.yaml",
    "phy_ai/cnn/config_cnn10_serifi_gradient.yaml",
    "phy_ai/cnn/config_cnn10_xiong_continuity.yaml",
    "phy_ai/cnn/config_cnn10_xiong_directional.yaml"
]

for cfg_rel_path in hardcoded_configs:
    cfg_path = CONFIGS_DIR / cfg_rel_path
    if cfg_path.exists():
        model_configs.append(cfg_path)
    else:
        print(f"Warning: Config not found: {cfg_path}")

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
