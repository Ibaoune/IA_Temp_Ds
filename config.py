"""
==========================================================
 Script: config.py
 Author: M. El Aabaribaoune
 Description:
     Loads configuration from YAML and exposes attributes
     as a simple namespace with robust type casting
     (int / float / bool / string safe).
==========================================================
"""

import yaml
import torch
import os


# Helper: safe casting
def _to_int(x):
    return int(x) if x is not None else None


def _to_float(x):
    return float(x) if x is not None else None


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y"]
    return bool(x)


# Config class
class Config:
    def __init__(self, cfg_dict, train_mode=True):
        self.train_mode = train_mode

        # ----------------------
        # Device
        # ----------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ----------------------
        # General
        # ----------------------
        self.verbose = _to_bool(cfg_dict["general"].get("verbose", True))
        self.experiment = str(cfg_dict["general"].get("experiment", "default_exp"))
        self.variable = str(cfg_dict["general"].get("variable", "precip"))
        self.target = str(cfg_dict["general"].get("target"))
        self.src = str(cfg_dict["general"].get("src"))
        self.model_type = str(cfg_dict["general"].get("model_type", "vit"))
        self.interpolation_type = str(cfg_dict["general"].get("interpolation_type", "nearest"))
        self.variables = cfg_dict["general"].get("variables", ["z", "q", "u", "v", "t"])
        self.levels = [int(l) for l in cfg_dict["general"].get("levels", [850, 700, 500])]
        self.resolution = _to_float(cfg_dict["general"].get("resolution", 2.0))

        # ----------------------
        # Training
        # ----------------------
        tr = cfg_dict["training"]
        self.learning_rate = _to_float(tr.get("learning_rate"))
        self.epochs = _to_int(tr.get("epochs"))
        self.batch_size = _to_int(tr.get("batch_size"))
        self.loss_type = str(tr.get("loss_type"))
        self.norm_mode = str(tr.get("norm_mode"))
        self.early_stopping_max = _to_int(tr.get("early_stopping_max"))

        # ----------------------
        # ViT parameters
        # ----------------------
        vit = cfg_dict["vit"]
        self.emb_size = _to_int(vit.get("emb_size"))
        self.patch_size = _to_int(vit.get("patch_size"))
        self.num_layers = _to_int(vit.get("num_layers"))
        self.num_heads = _to_int(vit.get("num_heads"))
        self.dropout = _to_float(vit.get("dropout"))

        # ----------------------
        # Region
        # ----------------------
        reg = cfg_dict["region"]
        self.lon_min = _to_float(reg.get("lon_min"))
        self.lon_max = _to_float(reg.get("lon_max"))
        self.lat_min = _to_float(reg.get("lat_min"))
        self.lat_max = _to_float(reg.get("lat_max"))

        # ----------------------
        # Dates
        # ----------------------
        self.start_date_train = str(cfg_dict["dates"]["train"]["start"])
        self.end_date_train = str(cfg_dict["dates"]["train"]["end"])
        self.start_date_test = str(cfg_dict["dates"]["test"]["start"])
        self.end_date_test = str(cfg_dict["dates"]["test"]["end"])

        # ----------------------
        # Paths & Patterns
        # ----------------------
        paths = cfg_dict["paths"]
        self.root_dir = str(paths.get("root_dir", ""))
        
        # Build absolute paths using root_dir if provided
        self.data_path = os.path.join(self.root_dir, str(paths.get("data_path", "")))
        self.results_dir = os.path.join(self.root_dir, str(paths.get("results_dir", "results/")))
        self.shapefile_path = os.path.join(self.root_dir, str(paths.get("shapefile_path", "")))
        
        # Patterns
        self.era5_predictor_pattern = str(paths.get("era5_predictor_pattern", ""))
        if not os.path.isabs(self.era5_predictor_pattern) and self.root_dir:
            self.era5_predictor_pattern = os.path.join(self.root_dir, self.era5_predictor_pattern)
            
        self.lmdz_predictor_pattern = str(paths.get("lmdz_predictor_pattern", ""))

        # ----------------------
        # Mappings
        # ----------------------
        self.lmdz_var_map = cfg_dict.get("mappings", {}).get("lmdz_var_map", {})

        # Target Path (Support both old and new YAML structure)
        if self.variable == "precip":
            # Check new structure first
            self.target_path = str(paths.get(f"{self.target}_path", ""))
            # Fallback to old structure
            if not self.target_path:
                self.target_path = str(cfg_dict.get("data",{}).get("precip", {}).get(f"{self.target}_path", ""))
        elif self.variable == "temp":
            var_target = "era5" if self.target == "mswt" else "lmdz"
            self.target_path = str(cfg_dict.get("data",{}).get("temp", {}).get(f"{var_target}_path", ""))
        
        if self.target_path and not os.path.isabs(self.target_path) and self.root_dir:
            self.target_path = os.path.join(self.root_dir, self.target_path)

        # ----------------------
        # Prediction
        # ----------------------
        pred_cfg = cfg_dict.get("prediction", {})
        self.scenarios = pred_cfg.get("scenarios", [])
        self.bc_reference_folder = pred_cfg.get("bc_reference_folder", "")
        
        # Final model save directory
        self.exp_dir = os.path.join(self.results_dir, self.experiment)
        self.model_save_dir = os.path.join(self.exp_dir, "models")
        
        os.makedirs(self.model_save_dir, exist_ok=True)

        # ----------------------
        # Evaluation
        # ----------------------
        self.evaluation = cfg_dict.get("evaluation", {})
        # Support both structures for MSWEP/MSWT paths
        self.mswep_path = str(paths.get("mswep_path", ""))
        if not self.mswep_path:
             self.mswep_path = cfg_dict.get("data", {}).get("precip", {}).get("mswep_path", "")
        
        if self.mswep_path and not os.path.isabs(self.mswep_path) and self.root_dir:
            self.mswep_path = os.path.join(self.root_dir, self.mswep_path)

        self.mswt_path = cfg_dict.get("data", {}).get("temp", {}).get("era5_path", "")


# YAML loader
def load_config(train_mode=True, path="config.yaml"):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return Config(cfg_dict, train_mode=train_mode)

