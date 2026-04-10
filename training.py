"""
==========================================================
 Script: training.py
 Author: M. El Aabaribaoune
 Description:
     Defines the training loop, loss computation, and
     early stopping for the downscaling model.

 Design:
     - Model-agnostic (ViT, CNN, UNet, etc.)
     - GPU / CPU compatible
==========================================================
"""

import torch
import sys
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from utils import vprint, get_machine_features, estimate_total_time
import time


def _build_model(cfg, x_train, y_train):
    """
    Factory function for model instantiation.
    """
    if cfg.loss_type == "bernoulli_gamma":
        out_channels = 3
    elif cfg.loss_type == "gaussian":
        out_channels = 2
    else:
        out_channels = 1

    if cfg.model_type == "vit":
        from vit_arch import DownscalingViT
        return DownscalingViT(
            in_channels=x_train.shape[1],
            emb_size=cfg.emb_size,
            patch_size=cfg.patch_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            output_channels=out_channels,
            n_lat_out=y_train.shape[-2],
            n_lon_out=y_train.shape[-1],
        )
    elif cfg.model_type == "unet":
        from unet_arch import UNet
        import torch.nn as nn
        import torch.nn.functional as F
        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet(in_channels=x_train.shape[1], out_channels=out_channels)
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])
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
                self.unet = UNet1(in_channels=x_train.shape[1], out_channels=out_channels)
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])
            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode='nearest')
                return out
        return WrappedUNet1()
    elif cfg.model_type == "cnn":
        from cnn import CNN
        return CNN(
            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            out_channels=out_channels,
            output_shape=(y_train.shape[-2], y_train.shape[-1])
        )
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not supported yet")


def train_model(cfg, x_train, y_train):
    vprint("Initializing model for training...")

    if cfg.model_type == "glm":
        from glm import train_glm
        return train_glm(cfg, x_train, y_train)

    model = _build_model(cfg, x_train, y_train).to(cfg.device)

    # ----------------
    # Loss & optimizer
    # ----------------
    if cfg.loss_type == "mse":
        criterion = nn.MSELoss()
    elif cfg.loss_type == "bernoulli_gamma":
        from losses import BernoulliGammaLoss
        criterion = BernoulliGammaLoss()
    elif cfg.loss_type == "gaussian":
        from losses import GaussianLoss
        criterion = GaussianLoss()
    else:
        raise ValueError(f"Unsupported loss type: {cfg.loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    if x_train.shape[0] != y_train.shape[0]:
        print("❌ Mismatch in number of samples!")
        print("x_train samples:", x_train.shape[0])
        print("y_train samples:", y_train.shape[0])
    else:
        print("✅ Same number of samples")
    # ----------------
    # DataLoader
    # ----------------
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=True,
    )

    train_losses = []
    val_losses = []  # kept for compatibility
    best_loss = float("inf")
    patience = 0

    # Training loop
    # ----------------
    for epoch in range(cfg.epochs):
        if epoch == 0:
            features = get_machine_features()
            vprint("\n----------- Machine Features -----------")
            vprint(f" OS: {features['os']}")
            vprint(f" CPU: {features['cpu']} ({features['cores']} cores)")
            vprint(f" RAM: {features['ram_total_gb']} GB")
            if features.get('gpus'):
                for idx, gpu in enumerate(features['gpus']):
                    vprint(f" GPU {idx}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
            else:
                vprint(" GPU: Not available")
            vprint("----------------------------------------")

        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        for xb, yb in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(xb)
            
            if cfg.loss_type == "mse":
                loss = criterion(outputs[:, 0, :, :], yb.squeeze(1))
            else:
                # BernoulliGammaLoss / GaussianLoss expect (B, C, H, W) and (B, 1, H, W)
                loss = criterion(outputs, yb)
                
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_duration = time.time() - epoch_start_time
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        vprint(f"Epoch {epoch+1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s - Loss: {epoch_loss:.4f}")

        if epoch == 0:
            estimated_time = estimate_total_time(epoch_duration, cfg.epochs)
            vprint(f"Estimated training process duration: {estimated_time}\n")

        # ----------------
        # Early stopping
        # ----------------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_max:
                vprint("Early stopping triggered.")
                break

    return model, train_losses, val_losses
