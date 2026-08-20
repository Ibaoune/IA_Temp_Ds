from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

try:
    from .backend_paths import CONFIGS_DIR, import_main_module
except ImportError:
    from backend_paths import CONFIGS_DIR, import_main_module

_core_config = import_main_module("src.core.config")
_core_utils = import_main_module("src.core.utils")

Config = _core_config.Config
resolve_model_config_path = _core_config.resolve_model_config_path
set_verbose = _core_utils.set_verbose
vprint = _core_utils.vprint

try:
    from .predict import perform_single_prediction
except ImportError:
    from predict import perform_single_prediction


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _load_prediction_config() -> tuple[dict, Path]:
    current_dir = Path(__file__).resolve().parent
    pred_config_file = current_dir / "config.yaml"

    with open(pred_config_file, "r", encoding="utf-8") as f:
        pred_cfg_dict = yaml.safe_load(f)

    return pred_cfg_dict, pred_config_file


def _merge_with_training_config(pred_cfg_dict: dict) -> dict:
    general_cfg = pred_cfg_dict.get("general", {})
    model_type = general_cfg.get("model_type", "unet")
    config_name = (
        general_cfg.get("config_name")
        or general_cfg.get("architecture_config")
        or pred_cfg_dict.get("config_name")
        or pred_cfg_dict.get("architecture_config")
    )
    base_config_path = resolve_model_config_path(
        CONFIGS_DIR,
        model_type,
        config_name,
    )

    if not base_config_path.exists():
        return pred_cfg_dict

    with open(base_config_path, "r", encoding="utf-8") as f:
        full_cfg_dict = yaml.safe_load(f)

    for key in pred_cfg_dict:
        if key in full_cfg_dict and isinstance(full_cfg_dict[key], dict):
            full_cfg_dict[key].update(pred_cfg_dict[key])
        else:
            full_cfg_dict[key] = pred_cfg_dict[key]

    return full_cfg_dict


def _resolve_prediction_dirs(cfg, pred_cfg_dict: dict) -> str:
    current_dir = Path(__file__).resolve().parent
    pred_section = pred_cfg_dict.get("prediction", {})

    raw_models_dir = pred_section.get("models_dir", "../results/")
    if not os.path.isabs(raw_models_dir):
        abs_results_dir = os.path.abspath(os.path.join(current_dir, raw_models_dir))
    else:
        abs_results_dir = raw_models_dir

    cfg.results_dir = abs_results_dir
    cfg.exp_dir = os.path.join(abs_results_dir, cfg.experiment)
    cfg.model_save_dir = os.path.join(cfg.exp_dir, "models")

    raw_output_dir = pred_section.get(
        "output_dir",
        os.path.join("..", "results", cfg.experiment, "prediction"),
    )
    if not os.path.isabs(raw_output_dir):
        cfg.output_dir = os.path.abspath(os.path.join(current_dir, raw_output_dir))
    else:
        cfg.output_dir = raw_output_dir

    return abs_results_dir


def build_era5_prediction_config(
    pred_cfg_dict: dict,
    start: str | None = None,
    end: str | None = None,
    bias_correction: bool = False,
):
    full_cfg_dict = _merge_with_training_config(pred_cfg_dict)
    cfg = Config(full_cfg_dict, train_mode=False)

    general_section = pred_cfg_dict.get("general", {})
    paths_section = pred_cfg_dict.get("paths", {})
    dates_section = pred_cfg_dict.get("dates", {})
    pred_section = pred_cfg_dict.get("prediction", {})

    dates_test_section = dates_section.get("test", {})
    dates_reference_section = dates_section.get("reference", {})

    selected_start = start or dates_test_section.get("start")
    selected_end = end or dates_test_section.get("end")
    reference_start = dates_reference_section.get("start")
    reference_end = dates_reference_section.get("end")

    required_dates = [
        ("dates.test.start", selected_start),
        ("dates.test.end", selected_end),
        ("dates.reference.start", reference_start),
        ("dates.reference.end", reference_end),
    ]
    missing_dates = [name for name, value in required_dates if value is None]
    if missing_dates:
        raise ValueError(
            "Missing required date(s) in temp/prediction/config.yaml: "
            + ", ".join(missing_dates)
        )

    cfg.src = "era5"
    cfg.folder = ""
    raw_root_dir = paths_section.get("root_dir", getattr(cfg, "root_dir", ""))
    current_dir = Path(__file__).resolve().parent
    cfg.root_dir = str((current_dir / raw_root_dir).resolve()) if raw_root_dir and not os.path.isabs(raw_root_dir) else raw_root_dir
    cfg.predictor_pattern = cfg.era5_predictor_pattern
    cfg.start_date_test = str(selected_start)
    cfg.end_date_test = str(selected_end)
    cfg.start_date_reference = str(reference_start)
    cfg.end_date_reference = str(reference_end)
    cfg.bc_reference_folder = ""
    cfg.bias_correction = bool(bias_correction)
    cfg.output_units = pred_section.get("output_units", "degree_Celsius")

    # Keep user-configured metadata that may be used by Config/load_datasets.
    cfg.experiment = general_section.get("experiment", cfg.experiment)
    cfg.target_path = cfg.target_path
    if "era5_predictor_pattern" in paths_section:
        cfg.era5_predictor_pattern = cfg.era5_predictor_pattern
        cfg.predictor_pattern = cfg.era5_predictor_pattern

    abs_results_dir = _resolve_prediction_dirs(cfg, pred_cfg_dict)
    set_verbose(cfg.verbose)

    return cfg, abs_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the trained UNet weights to ERA5 predictors and save an ERA5 downscaled NetCDF."
    )
    parser.add_argument("--start", default=None, help="Prediction start date. Defaults to dates.test.start.")
    parser.add_argument("--end", default=None, help="Prediction end date. Defaults to dates.test.end.")
    parser.add_argument(
        "--bc",
        action="store_true",
        help="Also apply SDM bias correction to ERA5 predictors before UNet inference. Default is nobc.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_console_encoding()
    args = parse_args()
    pred_cfg_dict, pred_config_file = _load_prediction_config()
    cfg, abs_results_dir = build_era5_prediction_config(
        pred_cfg_dict=pred_cfg_dict,
        start=args.start,
        end=args.end,
        bias_correction=args.bc,
    )

    mode = "bc" if cfg.bias_correction else "nobc"
    expected_name = f"era5_{cfg.start_date_test}_{cfg.end_date_test}_{mode}.nc"
    expected_path = Path(cfg.output_dir) / expected_name

    print("=== Starting ERA5 downscaled prediction with UNet ===")
    print(f"Prediction config : {pred_config_file}")
    print(f"Models dir        : {cfg.model_save_dir}")
    print(f"ERA5 predictors   : {cfg.era5_predictor_pattern}")
    print(f"Selected period   : {cfg.start_date_test} -> {cfg.end_date_test}")
    print(f"Reference period  : {cfg.start_date_reference} -> {cfg.end_date_reference}")
    print(f"Mode              : {mode}")
    print(f"Output file       : {expected_path}")

    perform_single_prediction(cfg, abs_results_dir=abs_results_dir)

    if expected_path.exists():
        print(f"[SUCCESS] ERA5 downscaled file saved to: {expected_path}")
    else:
        print(f"[WARNING] Expected output was not found after prediction: {expected_path}")

    vprint("=== ERA5 downscaled prediction completed ===")


if __name__ == "__main__":
    main()
