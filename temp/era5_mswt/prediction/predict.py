from pathlib import Path
import os
import yaml
import torch
import xarray as xr

try:
    from .backend_paths import CONFIGS_DIR, import_main_module
except ImportError:
    from backend_paths import CONFIGS_DIR, import_main_module

_core_config = import_main_module("src.core.config")
_core_utils = import_main_module("src.core.utils")
_core_evaluation = import_main_module("src.core.evaluation")
_data_loading = import_main_module("src.data.data_loading")

Config = _core_config.Config
resolve_model_config_path = _core_config.resolve_model_config_path
vprint = _core_utils.vprint
load_model = _core_utils.load_model
set_verbose = _core_utils.set_verbose
get_model = _core_evaluation._build_model
load_datasets = _data_loading.load_datasets

try:
    from .bias_correction import scaling_delta_mapping, standardize_predictors
except ImportError:
    from bias_correction import scaling_delta_mapping, standardize_predictors



#def perform_single_prediction(cfg, scenario=None, abs_results_dir=None): (i modified this function)
def perform_single_prediction(cfg, abs_results_dir=None):
    """
    Load -> Interpolate -> Bias Correction -> Standardize -> Predict -> Save
    """
    vprint("\n--- Starting prediction ---")

    apply_bc = getattr(cfg, "bias_correction", False)

    reference_start = cfg.start_date_reference
    reference_end = cfg.end_date_reference

    # Snapshot cfg fields that will be temporarily mutated
    orig_src    = cfg.src
    orig_folder = cfg.folder
    orig_start  = cfg.start_date_test
    orig_end    = cfg.end_date_test
    orig_predictor_pattern = cfg.predictor_pattern

    def set_source(source):
        cfg.src = source
        if source == "era5":
            cfg.predictor_pattern = cfg.era5_predictor_pattern
        elif source == "lmdz":
            cfg.predictor_pattern = cfg.lmdz_predictor_pattern
        elif source == "lmdz250":
            cfg.predictor_pattern = cfg.lmdz250_predictor_pattern
        elif source == "lmdz35":
            cfg.predictor_pattern = cfg.lmdz35_predictor_pattern

    def restore_cfg():
        cfg.src = orig_src
        cfg.folder = orig_folder
        cfg.start_date_test = orig_start
        cfg.end_date_test   = orig_end
        cfg.predictor_pattern = orig_predictor_pattern

    # 1 & 2. Load + Interpolate
    vprint(f"Flow Step 1 & 2: Loading and Interpolating {cfg.src} predictors...")
    datasets = load_datasets(cfg)
    X = datasets[0].sel(time=slice(cfg.start_date_test, cfg.end_date_test))

    lon_out, lat_out = datasets[5], datasets[6]
    time_test = X.time.values

    # 3. Bias Correction + Standardization reference
    if apply_bc or cfg.norm_mode == "gridbox":
        vprint(f"Loading historical references (ERA5/GCM {reference_start} to {reference_end})...")

        # Use the absolute results_dir that was fixed in main() — cfg.results_dir may
        # have been reset to the relative training-config value by Config.__setattr__
        results_dir = abs_results_dir if abs_results_dir else cfg.results_dir
        cache_dir   = os.path.join(results_dir, cfg.experiment, "stats")
        os.makedirs(cache_dir, exist_ok=True)
        mean_path = os.path.join(cache_dir, f"mean_era5_{reference_start}_{reference_end}_{cfg.norm_mode}.nc")
        std_path  = os.path.join(cache_dir, f"std_era5_{reference_start}_{reference_end}_{cfg.norm_mode}.nc")
        # Load ERA5 historical whenever stats are not cached or BC is needed
        if not (os.path.exists(mean_path) and os.path.exists(std_path)) or apply_bc:
            vprint(f"Loading ERA5 historical ({reference_start} to {reference_end})...")
            set_source("era5")
            cfg.start_date_test, cfg.end_date_test = reference_start, reference_end
            X_era5_hist = load_datasets(cfg)[0].sel(time=slice(reference_start, reference_end))
            restore_cfg()

        if os.path.exists(mean_path) and os.path.exists(std_path):
            vprint(f"Loading cached ERA5 stats...")
            mean_ref = xr.open_dataarray(mean_path)
            std_ref  = xr.open_dataarray(std_path)
        else:
            vprint("Computing and caching ERA5 stats...")
            mean_ref = X_era5_hist.mean(dim="time") if cfg.norm_mode == "gridbox" else X_era5_hist.mean()
            std_ref  = X_era5_hist.std(dim="time")  if cfg.norm_mode == "gridbox" else X_era5_hist.std()
            mean_ref.to_netcdf(mean_path)
            std_ref.to_netcdf(std_path)
            vprint(f"Stats cached to {cache_dir}")

        if apply_bc:
            vprint("Flow Step 3: Applying Bias Correction (SDM)...")

            bc_reference_folder = getattr(cfg, "bc_reference_folder", "") or ""
            gcm_base_folder = bc_reference_folder if str(bc_reference_folder).strip() else orig_folder

            vprint("  SDM setup:")
            vprint(f"    gcm_full = {orig_src}, folder = {orig_folder}, period = {orig_start} -> {orig_end}")
            vprint(f"    gcm_hist = {orig_src}, folder = {gcm_base_folder}, period = {reference_start} -> {reference_end}")
            vprint(f"    obs_hist = ERA5, period = {reference_start} -> {reference_end}")

            # Charger la période de base du GCM/RCM
            set_source(orig_src)
            cfg.folder = gcm_base_folder
            cfg.start_date_test = reference_start
            cfg.end_date_test = reference_end

            X_gcm_hist = load_datasets(cfg)[0].sel(time=slice(reference_start, reference_end))

            restore_cfg()

            # Vérifications pour éviter une correction silencieuse avec données vides
            if X.time.size == 0:
                raise ValueError(
                    f"gcm_full is empty for period {orig_start} -> {orig_end} "
                    f"in folder {orig_folder}"
                )

            if X_gcm_hist.time.size == 0:
                raise ValueError(
                    f"gcm_hist is empty for period {reference_start} -> {reference_end} "
                    f"in folder {gcm_base_folder}"
                )

            if X_era5_hist.time.size == 0:
                raise ValueError(
                    f"obs_hist ERA5 is empty for period {reference_start} -> {reference_end}"
                )
            
            # gcm_full = période à corriger
            # gcm_hist = base GCM/RCM
            # obs_hist = base ERA5
            X = scaling_delta_mapping(X, X_gcm_hist, X_era5_hist)

        else:
            vprint("Flow Step 3: Skipping Bias Correction.")
    else:
        vprint("Flow Step 3: Skipping Bias Correction.")
        mean_ref, std_ref = 0.0, 1.0

    # 4. Standardization
    vprint("Flow Step 4: Standardizing predictors (Baseline: ERA5 Hist)...")
    X_std = standardize_predictors(X, mean_ref, std_ref)

    # 5. Load model (cached across scenarios) + Inference
    # Restore absolute results_dir before load_model — cfg may have been reset
    if abs_results_dir:
        cfg.results_dir    = abs_results_dir
        cfg.exp_dir        = os.path.join(abs_results_dir, cfg.experiment)
        cfg.model_save_dir = os.path.join(cfg.exp_dir, "models")

    if not hasattr(perform_single_prediction, "cached_model"):
        vprint("Flow Step 5: Preparing Model...")
        n_channels = X_std.shape[1]
        h_in, w_in = X_std.shape[2], X_std.shape[3]
        out_shape  = (len(lat_out), len(lon_out))
        dummy_x = torch.empty(1, n_channels, h_in, w_in)
        dummy_y = torch.empty(1, 1, *out_shape)
        model_arch = get_model(cfg, dummy_x, dummy_y).to(cfg.device)
        model, _, _ = load_model(cfg, model_arch)
        model.eval()
        perform_single_prediction.cached_model = model

    model = perform_single_prediction.cached_model
    vprint("Flow Step 5: Running Inference...")

    preds = []
    chunk_size = 512

    with torch.no_grad():
        for i in range(0, X_std.shape[0], chunk_size):
            xb_np = X_std.isel(time=slice(i, i + chunk_size)).values.astype("float32")
            xb = torch.tensor(xb_np, dtype=torch.float32).to(cfg.device)

            out = model(xb)

            # température : on garde le canal 0 = moyenne prédite
            pred = out[:, 0, :, :]

            preds.append(pred.cpu())

    preds_np = torch.cat(preds, dim=0).numpy()

    # 6. Save
    out_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    folder_tag = Path(str(cfg.folder)).parts[0] if getattr(cfg, "folder", "") else cfg.src
    bc_suffix = "_bc" if apply_bc else "_nobc"
    folder_str = str(getattr(cfg, "folder", "")).lower()

    if "lmdz_35" in folder_str:
        model_tag = "lmdz_35"
    elif "lmdz_250" in folder_str:
        model_tag = "lmdz_250"
    else:
        model_tag = str(cfg.src)

    save_name = f"{model_tag}_{cfg.start_date_test}_{cfg.end_date_test}{bc_suffix}.nc"
    out_nc = os.path.join(out_dir, save_name)

    var_name = "air_temperature"
    var_units = getattr(cfg, "output_units", "degree_Celsius")

    if var_units in ["C", "celsius", "Celsius", "degC", "°C", "degree_Celsius"]:
        var_units = "degree_Celsius"
    else:
        var_units = "degree_Celsius"

    ds_pred = xr.Dataset(
        {var_name: (["time", "lat", "lon"], preds_np)},
        coords={"time": time_test, "lat": lat_out, "lon": lon_out}
    )
    ds_pred.attrs.update(
        {
            "source_type": str(orig_src),
            "source_folder": str(orig_folder),
            "model_tag": str(model_tag),
            "bias_correction": str(bool(apply_bc)).lower(),
            "prediction_start": str(cfg.start_date_test),
            "prediction_end": str(cfg.end_date_test),
            "bc_reference_start": str(reference_start),
            "bc_reference_end": str(reference_end),
        }
    )
    ds_pred[var_name].attrs["units"] = var_units
    ds_pred.to_netcdf(out_nc)

    vprint(f"Flow Step 6: Results saved to {out_nc}")


def main():
    current_dir = Path(__file__).resolve().parent
    pred_config_file = current_dir / "config.yaml"

    with open(pred_config_file, "r") as f:
        pred_cfg_dict = yaml.safe_load(f)

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

    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            full_cfg_dict = yaml.safe_load(f)
        for key in pred_cfg_dict:
            if key in full_cfg_dict and isinstance(full_cfg_dict[key], dict):
                full_cfg_dict[key].update(pred_cfg_dict[key])
            else:
                full_cfg_dict[key] = pred_cfg_dict[key]
    else:
        full_cfg_dict = pred_cfg_dict

    cfg = Config(full_cfg_dict, train_mode=False)

    # Resolve absolute results_dir from prediction config — do NOT rely on cfg.models_dir
    # since Config may not expose nested prediction.models_dir as a flat attribute.
    raw_models_dir = pred_cfg_dict.get("prediction", {}).get("models_dir", "../results/")
    if not os.path.isabs(raw_models_dir):
        abs_results_dir = os.path.abspath(os.path.join(current_dir, raw_models_dir))
    else:
        abs_results_dir = raw_models_dir

    cfg.results_dir    = abs_results_dir
    cfg.exp_dir        = os.path.join(abs_results_dir, cfg.experiment)
    cfg.model_save_dir = os.path.join(cfg.exp_dir, "models")

    # Flatten remaining prediction-section paths that Config doesn't expose directly
    pred_section = pred_cfg_dict.get("prediction", {})
    general_section = pred_cfg_dict.get("general", {})
    paths_section = pred_cfg_dict.get("paths", {})
    dates_section = pred_cfg_dict.get("dates", {})

    dates_test_section = dates_section.get("test", {})
    dates_reference_section = dates_section.get("reference", {})

    required_dates = [
        ("dates.test.start", dates_test_section.get("start")),
        ("dates.test.end", dates_test_section.get("end")),
        ("dates.reference.start", dates_reference_section.get("start")),
        ("dates.reference.end", dates_reference_section.get("end")),
    ]

    missing_dates = [name for name, value in required_dates if value is None]

    if missing_dates:
        raise ValueError(
            "Missing required date(s) in temp/prediction/config.yaml: "
            + ", ".join(missing_dates)
        )

    cfg.src = general_section.get("src", cfg.src)
    cfg.folder = paths_section.get("folder", getattr(cfg, "folder", ""))
    raw_root_dir = paths_section.get("root_dir", getattr(cfg, "root_dir", ""))
    cfg.root_dir = str((current_dir / raw_root_dir).resolve()) if raw_root_dir and not os.path.isabs(raw_root_dir) else raw_root_dir

    #gcm_full
    cfg.start_date_test = dates_test_section["start"]
    cfg.end_date_test = dates_test_section["end"]

    # gcm_hist + obs_hist
    cfg.start_date_reference = dates_reference_section["start"]
    cfg.end_date_reference = dates_reference_section["end"]

    raw_output_dir = pred_section.get(
        "output_dir",
        os.path.join("..", "results", cfg.experiment, "prediction")
    )

    if not os.path.isabs(raw_output_dir):
        cfg.output_dir = os.path.abspath(os.path.join(current_dir, raw_output_dir))
    else:
        cfg.output_dir = raw_output_dir
    cfg.bc_reference_folder = pred_section.get("bc_reference_folder", "")
    cfg.bias_correction = pred_section.get("bias_correction", False)
    cfg.output_units = pred_section.get("output_units", "K")
    set_verbose(cfg.verbose)
    vprint(f"=== Starting Prediction Pipeline for {cfg.experiment} ===")
    vprint(f"Models dir: {cfg.model_save_dir}")

    perform_single_prediction(cfg, abs_results_dir=abs_results_dir)

    vprint("=== Prediction Task Completed ===")


if __name__ == "__main__":
    main()
