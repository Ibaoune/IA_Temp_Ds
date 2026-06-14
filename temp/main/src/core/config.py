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
from pathlib import Path


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


def resolve_model_config_path(configs_root, model_type, config_name=None):
    configs_root = Path(configs_root)
    model_key = str(model_type or "unet").lower()
    config_dir_key = "cnn" if model_key in {"cnn1", "cnn10"} else model_key
    unified_unet_dir = configs_root / "unet"

    if config_name not in (None, "", "null"):
        requested = Path(str(config_name))
        if requested.is_absolute():
            return requested

        if len(requested.parts) > 1:
            return configs_root / requested

        if model_key in {"unet", "unet1"}:
            candidate = unified_unet_dir / requested
            legacy_candidate = configs_root / model_key / requested
            if not candidate.exists() and legacy_candidate.exists():
                return legacy_candidate
            return candidate

        return configs_root / config_dir_key / requested

    default_names = {
        "unet": "config_arch.yaml",
        "unet1": "config_arch1.yaml",
    }
    if model_key in default_names:
        candidate = unified_unet_dir / default_names[model_key]
        if candidate.exists():
            return candidate

    legacy_candidate = configs_root / config_dir_key / "config.yaml"
    if legacy_candidate.exists():
        return legacy_candidate

    if model_key in default_names:
        return unified_unet_dir / default_names[model_key]

    return legacy_candidate


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
        self.model_type = str(cfg_dict["general"]["model_type"]).lower()
        allowed_models = {"cnn", "cnn1", "cnn10", "glm", "unet", "unet1"}
        if self.model_type not in allowed_models:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Expected one of: {sorted(allowed_models)}"
            )

        cnn_cfg = cfg_dict.get("cnn", {})
        raw_cnn_mode = cnn_cfg.get("mode", "")
        yaml_cnn_mode = "" if raw_cnn_mode is None else str(raw_cnn_mode).strip().lower()
        if yaml_cnn_mode == "null":
            yaml_cnn_mode = ""
        if self.model_type in {"cnn1", "cnn10"}:
            self.cnn_mode = self.model_type
            if yaml_cnn_mode and yaml_cnn_mode != self.cnn_mode:
                raise ValueError(
                    f"Inconsistent CNN config: model_type='{self.model_type}' "
                    f"but cnn.mode='{yaml_cnn_mode}'."
                )
        else:
            self.cnn_mode = yaml_cnn_mode or "cnn10"

        if self.model_type in {"cnn", "cnn1", "cnn10"} and self.cnn_mode not in {"cnn1", "cnn10"}:
            raise ValueError(
                f"Unsupported cnn.mode '{self.cnn_mode}'. "
                "Expected one of: ['cnn1', 'cnn10']"
            )
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

        es = tr.get("early_stopping", {})
        self.early_stopping_enable = _to_bool(es.get("enable", False))
        self.early_stopping_max = _to_int(es.get("max", 15))

        self.optimizer = str(tr.get("optimizer", "adam")).lower()

        wd = tr.get("weight_decay", {})
        self.weight_decay_enable = _to_bool(wd.get("enable", False))
        self.weight_decay_value = _to_float(wd.get("value", 0.0))

        gc = tr.get("gradient_clipping", {})
        self.gradient_clipping_enable = _to_bool(gc.get("enable", False))
        self.gradient_clipping_value = _to_float(gc.get("value", 1.0))

        drop = tr.get("dropout", {})
        self.dropout_enable = _to_bool(drop.get("enable", False))
        self.dropout_value = _to_float(drop.get("value", 0.0))

        gn = tr.get("group_norm", {})
        self.group_norm_enable = _to_bool(gn.get("enable", False))
        self.group_norm_num_groups = _to_int(gn.get("num_groups", 32))

        val = tr.get("validation", {})
        self.validation_enable = _to_bool(val.get("enable", False))
        self.validation_percentage = _to_float(val.get("percentage", val.get("pecentage", 0.2)))

        sched = tr.get("scheduler", {})
        self.scheduler_enable = _to_bool(sched.get("enable", False))
        self.scheduler_type = str(sched.get("type", "plateau")).lower()
        self.scheduler_patience = _to_int(sched.get("patience", 5))
        self.scheduler_factor = _to_float(sched.get("factor", 0.5))
        self.scheduler_min_lr = _to_float(sched.get("min_lr", 1e-6))

        glm_cfg = cfg_dict.get("glm", {})
        self.glm_n_neighbors = _to_int(glm_cfg.get("n_neighbors", 4))


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

        data_path_raw = str(paths.get("data_path", ""))
        self.data_path = os.path.join(self.root_dir, data_path_raw) if self.root_dir and data_path_raw else data_path_raw
        
        self.results_dir = str(paths.get("results_dir", "results/"))
        
        shapefile_path_raw = str(paths.get("shapefile_path", ""))
        self.shapefile_path = os.path.join(self.root_dir, shapefile_path_raw) if self.root_dir and shapefile_path_raw else shapefile_path_raw

        def _resolve_path(path_value):
            path_value = str(path_value)
            if path_value and not os.path.isabs(path_value) and self.root_dir:
                return os.path.join(self.root_dir, path_value)
            return path_value


        # ----------------------
        # Predictor patterns
        # ----------------------

        # ERA5 predictors
        self.era5_predictor_pattern = _resolve_path(
            paths.get("era5_predictor_pattern", "")
        )

        # Ancien format LMDZ brut, utile surtout pour prediction/
        # Exemple :
        # {folder}/all_Mor/{lmdz_var}-{suffix}.nc
        self.lmdz_predictor_pattern = _resolve_path(
            paths.get("lmdz_predictor_pattern", "")
        )

        # Nouveau format harmonisé pour main/
        # Exemple :
        # ./DATA1/predictors_lmdz250/{var}_1979-2020_levels.nc
        self.lmdz250_predictor_pattern = _resolve_path(
            paths.get("lmdz250_predictor_pattern", "")
        )

        self.lmdz35_predictor_pattern = _resolve_path(
            paths.get("lmdz35_predictor_pattern", "")
        )

        # Choisir automatiquement le pattern des prédicteurs selon general.src
        if self.src == "era5":
            self.predictor_pattern = self.era5_predictor_pattern

        elif self.src == "lmdz250":
            self.predictor_pattern = self.lmdz250_predictor_pattern

        elif self.src == "lmdz35":
            self.predictor_pattern = self.lmdz35_predictor_pattern

        elif self.src == "lmdz":
            # On garde l'ancien cas pour ne pas casser prediction/
            self.predictor_pattern = self.lmdz_predictor_pattern

        else:
            raise ValueError(
                f"Unsupported src '{self.src}'. "
                "Expected one of: era5, lmdz, lmdz250, lmdz35."
            )

        if not self.predictor_pattern:
            raise ValueError(
                f"Missing predictor pattern for src='{self.src}'. "
                "Check paths in YAML."
            )


        # ----------------------
        # Mappings
        # ----------------------
        # Gardé pour l'ancien format LMDZ brut :
        # z -> geop, q -> shum, t -> temp, u -> vitu, v -> vitv
        self.lmdz_var_map = cfg_dict.get("mappings", {}).get("lmdz_var_map", {})


        # ----------------------
        # Target path
        # ----------------------
        if self.variable == "precip":
            self.target_path = _resolve_path(paths.get(f"{self.target}_path", ""))

            if not self.target_path:
                self.target_path = _resolve_path(
                    cfg_dict.get("data", {}).get("precip", {}).get(f"{self.target}_path", "")
                )

        elif self.variable == "temp":
            temp_cfg = cfg_dict.get("data", {}).get("temp", {})

            if self.target == "mswt":
                self.target_path = _resolve_path(
                    paths.get(
                        "mswt_path",
                        temp_cfg.get("mswt_path", temp_cfg.get("mswet_path", ""))
                    )
                )

            elif self.target == "lmdz35":
                self.target_path = _resolve_path(
                    paths.get(
                        "lmdz35_path",
                        temp_cfg.get("lmdz35_path", "")
                    )
                )

            elif self.target == "lmdz":
                # Ancien cas gardé pour compatibilité
                self.target_path = _resolve_path(
                    paths.get(
                        "lmdz_path",
                        temp_cfg.get("lmdz_path", "")
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported temperature target '{self.target}'. "
                    "Expected one of: mswt, lmdz, lmdz35."
                )

        else:
            raise ValueError(f"Unsupported variable: {self.variable}")

        if not self.target_path:
            raise ValueError(
                f"Missing target path for target='{self.target}'. "
                "Check paths in YAML."
            )

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
        # Support both paths section (test.yaml style) and data section (config.yaml style)
        self.mswep_path = str(paths.get("mswep_path", cfg_dict.get("data", {}).get("precip", {}).get("mswep_path", "")))
        self.mswt_path = str(paths.get("mswt_path", cfg_dict.get("data", {}).get("temp", {}).get("mswet_path", "")))

        #----------------------
        # Plots
        #----------------------
        plots = cfg_dict.get("plots", {})
        eval_plots = plots.get("eval", {})

        self.show_suffix_components_in_title = _to_bool(
            eval_plots.get("show_suffix_components_in_title", False)
        )


# YAML loader
def load_config(train_mode=True, path="config.yaml"):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return Config(cfg_dict, train_mode=train_mode)

