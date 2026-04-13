"""
==========================================================
 Script: evaluation.py
 Author: M. El Aabaribaoune
 Description:
     Model evaluation and diagnostics.

     - Loads trained model
     - Runs inference on test data
     - Saves predictions to NetCDF
     - Generates diagnostic plots

 Notes:
     - GPU-safe (chunked inference)
     - Model-agnostic (ViT by default)
==========================================================
"""

import os
import torch
import xarray as xr
import numpy as np

import utils as use
from utils import vprint, load_model

def _build_model(cfg, x_test, y_test):
    if cfg.model_type == "vit":
        from vit_arch import DownscalingViT
        return DownscalingViT(
            in_channels=x_test.shape[1],
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            output_channels=1,
            n_lat_out=y_test.shape[-2],
            n_lon_out=y_test.shape[-1],
        )
    elif cfg.model_type == "unet":
        from unet_arch import UNet
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                out_channels = 3 if cfg.loss_type == "bernoulli_gamma" else 1
                self.unet = UNet(in_channels=x_test.shape[1], out_channels=out_channels)
                self.out_shape = (y_test.shape[-2], y_test.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet()
    elif cfg.model_type == "unet1":
        from unet_arch1 import UNet as UNet1
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet1(nn.Module):
            def __init__(self):
                super().__init__()
                out_channels = 3 if cfg.loss_type == "bernoulli_gamma" else 1
                self.unet = UNet1(in_channels=x_test.shape[1], out_channels=out_channels)
                self.out_shape = (y_test.shape[-2], y_test.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet1()
    else:
        raise NotImplementedError(f"Model {cfg.model_type} not supported")


def evaluate_and_save(cfg, x_test, y_test, lon, lat, time):
    vprint("=== Starting evaluation ===")

    device = cfg.device
    vprint(f"Using device: {device}")

    # Load trained model
    model_arch = _build_model(cfg, x_test, y_test).to(device)
    model, train_losses, val_losses = load_model(cfg, model_arch)
    model.eval()

    vprint("Model loaded successfully.")

    # Experiment output path
    exp_path = use.build_experiment_path(cfg)
    os.makedirs(exp_path, exist_ok=True)

    # Inference (chunked for memory safety)
    preds = []
    chunk_size_val = 512

    vprint("Running inference...")
    with torch.no_grad():
        for i in range(0, x_test.shape[0], chunk_size_val):
            xb = x_test[i:i+chunk_size_val].to(device)
            out = model(xb)
            
            if cfg.loss_type == "bernoulli_gamma":
                # Compute expected value: E[X] = P(X>0) * shape * scale
                occurrence = torch.sigmoid(out[:, 0, :, :])
                shape = torch.exp(out[:, 1, :, :].clamp(-5, 5))
                scale = torch.exp(out[:, 2, :, :].clamp(-5, 5))
                precip = occurrence * shape * scale
            else:
                precip = out[:, 0, :, :]
                
            preds.append(precip.cpu())

    preds_np = torch.cat(preds, dim=0).numpy()

    # Save predictions to NetCDF
    ds_pred = xr.Dataset(
        {"precipitation": (["time", "lat", "lon"], preds_np)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    path_out_data = os.path.join(exp_path, "output_data")
    os.makedirs(path_out_data, exist_ok=True)

    out_nc = os.path.join(
        path_out_data, f"{cfg.model_type}_predictions_era5_to_{cfg.target}.nc"
    )
    ds_pred.to_netcdf(out_nc)
    vprint(f"Predictions saved at: {out_nc}")

    # Ground truth dataset (squeezing channel dimension if present)
    y_test_np = y_test.squeeze().numpy()
    y_test_ds = xr.Dataset(
        {"precip": (["time", "lat", "lon"], y_test_np)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    # Diagnostics / plots
    path_out_figs = os.path.join(exp_path, "output_figs")
    os.makedirs(path_out_figs, exist_ok=True)

    vprint("Generating plots...")

    use.plot_losses(
        train_losses,
        val_losses,
        model_name=cfg.model_type.upper(),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        filename=os.path.join(path_out_figs, "losses.png"),
    )

    use.spatial_comparaison_plot(
        y_test_ds,
        ds_pred,
        lon,
        lat,
        model_name=cfg.model_type.upper(),
        filename=os.path.join(path_out_figs, "spatial_distribution.png"),
        title_suffix=" (ERA5→MSWEP)",
        y_name="MSWEP",
        y_var="precip",
        model_var="precipitation",
    )

    use.monthly_precip_comparaison_plot(
        ds_pred,
        y_test_ds,
        model_name=cfg.model_type.upper(),
        filename=os.path.join(path_out_figs, "monthly_means.png"),
        y_name="MSWEP",
        y_var="precip",
        model_var="precipitation",
        title="Monthly Average Precipitation (mm/day)",
        title_suffix=" (ERA5→MSWEP)",
    )

    vprint(f"All evaluation outputs saved in: {exp_path}")
    vprint("=== Evaluation complete ===")
