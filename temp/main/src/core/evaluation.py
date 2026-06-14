"""
==========================================================
 Script: evaluation.py
 Author: M. El Aabaribaoune
 Description:
     Model evaluation and diagnostics.

     - Loads trained model
     - Runs inference on test data
     - Saves predictions to NetCDF

 Notes:
     - GPU-safe (chunked inference)
     - Model-agnostic
==========================================================
"""

from fileinput import filename
import os
import torch
import xarray as xr

import src.core.utils as use
from src.core.utils import vprint, load_model


def _build_model(cfg, x_test, y_test):
    if cfg.loss_type == "bernoulli_gamma":
        out_channels = 3
    elif cfg.loss_type == "gaussian":
        out_channels = 2
    else:
        out_channels = 1

    if cfg.model_type == "unet":
        from src.models.unet_arch import UNet
        import torch.nn as nn
        import torch.nn.functional as F

        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet(
                    in_channels=x_test.shape[1],
                    out_channels=out_channels,
                    base_filters=64,
                    upscale_factor=1,
                )
                self.out_shape = (y_test.shape[-2], y_test.shape[-1])

            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape,  mode="bilinear", align_corners=False)
                return out

        return WrappedUNet()

    elif cfg.model_type == "unet1":
        from src.models.unet_arch1 import UNet1
        import torch.nn as nn

        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()

                self.out_shape = (
                    y_test.shape[-2],
                    y_test.shape[-1],
                )

                self.unet = UNet1(
                    in_channels=x_test.shape[1],   
                    out_channels=out_channels,     
                    base_filters=64,
                    use_gaussian=(out_channels == 2),
                    norm_type="group" if cfg.group_norm_enable else "batch",
                    num_groups=cfg.group_norm_num_groups,
                    dropout=cfg.dropout_value if cfg.dropout_enable else 0.0,
                )

            def forward(self, x):
                return self.unet(
                    x,
                    target_size=self.out_shape,
                )

        return WrappedUNet()

    elif cfg.model_type in {"cnn", "cnn1", "cnn10"}:
        from src.models.cnn import CNN

        cnn_mode = getattr(cfg, "cnn_mode", "cnn10")

        return CNN(
            input_shape=(x_test.shape[1], x_test.shape[2], x_test.shape[3]),
            out_channels=out_channels,
            output_shape=(y_test.shape[-2], y_test.shape[-1]),
            mode=cnn_mode,
        )

    elif cfg.model_type == "glm":
        return None

    else:
        raise NotImplementedError(f"Model {cfg.model_type} not supported")

def _get_target_display_name(cfg):
    """
    Return a clean display name for the target dataset.
    Used only for plots and labels.
    """
    target = str(cfg.target).lower()

    if target == "mswt":
        return "MSWT"
    elif target == "lmdz35":
        return "LMDZ35"
    elif target == "lmdz":
        return "LMDZ"
    elif target == "mswep":
        return "MSWEP"
    else:
        return target.upper()

def evaluate_and_save(cfg, x_test, y_test, lon, lat, time):
    vprint("=== Starting evaluation ===")

    device = cfg.device
    vprint(f"Using device: {device}")

    # Load trained model
    if cfg.model_type == "glm":
        # GLM models are non-torch objects, loaded directly
        model, train_losses, val_losses = load_model(cfg, model=None)
    else:
        model_arch = _build_model(cfg, x_test, y_test).to(device)
        model, train_losses, val_losses = load_model(cfg, model_arch)
        model.eval()

    vprint("Model loaded successfully.")

    # Experiment output path
    exp_path = use.build_experiment_path(cfg)
    os.makedirs(exp_path, exist_ok=True)

    preds = []
    pred_log_vars = []
    chunk_size_val = 512

    vprint("Running inference...")

    if cfg.model_type == "glm":
        # GLM uses numpy arrays and CPU only
        x_test_np = x_test.cpu().numpy() if hasattr(x_test, "cpu") else x_test
        preds_np = model.predict(x_test_np)  # Returns (N, H, W)
    else:
        # Torch models
        with torch.no_grad():
            for i in range(0, x_test.shape[0], chunk_size_val):
                xb = x_test[i:i+chunk_size_val].to(device)
                out = model(xb)

                if cfg.loss_type == "bernoulli_gamma":
                    occurrence = torch.sigmoid(out[:, 0, :, :])
                    shape = torch.exp(out[:, 1, :, :].clamp(-5, 5))
                    scale = torch.exp(out[:, 2, :, :].clamp(-5, 5))
                    pred_main = occurrence * shape * scale

                elif cfg.loss_type == "gaussian":
                    pred_main = out[:, 0, :, :]
                    pred_log_var = out[:, 1, :, :]
                    pred_log_vars.append(pred_log_var.cpu())

                else:
                    pred_main = out[:, 0, :, :]

                preds.append(pred_main.cpu())

        preds_np = torch.cat(preds, dim=0).numpy()

    if cfg.loss_type == "gaussian" and len(pred_log_vars) > 0:
        pred_log_vars_np = torch.cat(pred_log_vars, dim=0).numpy()
    else:
        pred_log_vars_np = None

    # Extract actual dimensions from predictions
    n_time, n_lat, n_lon = preds_np.shape

    # Use subset of coordinates that match prediction shape
    lat_pred = lat[:n_lat] if len(lat) >= n_lat else lat
    lon_pred = lon[:n_lon] if len(lon) >= n_lon else lon

    # Save predictions to NetCDF
    if cfg.variable == "temp":
        if cfg.loss_type == "gaussian" and pred_log_vars_np is not None:
            ds_pred = xr.Dataset(
                {
                    "air_temperature": (["time", "lat", "lon"], preds_np),
                    "log_variance": (["time", "lat", "lon"], pred_log_vars_np),
                },
                coords={"time": time, "lat": lat_pred, "lon": lon_pred},
            )
        else:
            ds_pred = xr.Dataset(
                {
                    "air_temperature": (["time", "lat", "lon"], preds_np),
                },
                coords={"time": time, "lat": lat_pred, "lon": lon_pred},
            )
    else:
        ds_pred = xr.Dataset(
            {
                "precipitation": (["time", "lat", "lon"], preds_np),
            },
            coords={"time": time, "lat": lat_pred, "lon": lon_pred},
        )

    path_out_data = os.path.join(exp_path, "output_data")
    os.makedirs(path_out_data, exist_ok=True)

    out_nc = os.path.join(
        path_out_data,
        f"{cfg.model_type}_predictions_{cfg.target}.nc"
    )
    ds_pred.to_netcdf(out_nc)

# ----------------------------------
# Ground truth dataset
# ----------------------------------
    y_test_np = y_test.squeeze().cpu().numpy()
    y_test_ds = xr.Dataset(
        {
            "air_temperature": (["time", "lat", "lon"], y_test_np),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )

    # For GLM, interpolate ONLY for visualization / monthly diagnostics
    if cfg.model_type == "glm":
        ds_pred_plot = ds_pred.interp(lat=lat, lon=lon, method="linear")
    else:
        ds_pred_plot = ds_pred

    # Diagnostics / plots
    path_out_figs = os.path.join(exp_path, "output_figs")
    os.makedirs(path_out_figs, exist_ok=True)

    vprint("Generating plots...")

    target_display_name = _get_target_display_name(cfg)

    plot_title_suffix = use.format_components_for_title(
        src=cfg.src,
        target=cfg.target,
        variable=cfg.variable,

        experiment=cfg.experiment,
        model_type=cfg.model_type,
        interpolation_type=cfg.interpolation_type,

        norm_mode=cfg.norm_mode,
        loss_type=cfg.loss_type,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,

        early_stopping_max=cfg.early_stopping_max,

        variables=cfg.variables,
        levels=cfg.levels,
        resolution=cfg.resolution,

        train_start=cfg.start_date_train,
        train_end=cfg.end_date_train,
        test_start=cfg.start_date_test,
        test_end=cfg.end_date_test,
    )

    if train_losses is not None and len(train_losses) > 0:
        use.plot_losses(
            train_losses,
            val_losses,
            model_name=cfg.model_type.upper(),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            filename=os.path.join(path_out_figs, "losses.png"),
        )
    else:
        vprint(f"Skipping loss plot for {cfg.model_type.upper()} (no training losses available).")

    use.spatial_comparaison_plot(
        y_test_ds,
        ds_pred_plot,
        lon,
        lat,
        model_name=cfg.model_type.upper(),
        filename=os.path.join(path_out_figs, "spatial_distribution.png"),
        title_suffix=plot_title_suffix,
        y_name=target_display_name,
        y_var="air_temperature",
        model_var="air_temperature",
    )

    use.monthly_temp_comparaison_plot(
        ds_pred_plot,
        y_test_ds,
        model_name=cfg.model_type.upper(),
        filename=os.path.join(path_out_figs, "monthly_means.png"),
        y_name=target_display_name,
        y_var="air_temperature",
        model_var="air_temperature",
        title="Monthly Average Temperature (°C)",
        title_suffix=plot_title_suffix,
    )

    vprint(f"Predictions saved at: {out_nc}")
    vprint(f"All evaluation outputs saved in: {exp_path}")
    vprint("=== Evaluation complete ===")
