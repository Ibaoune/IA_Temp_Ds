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
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from src.core.utils import vprint, get_machine_features, estimate_total_time
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

    if cfg.model_type == "unet":
        from src.models.unet_arch import UNet
        import torch.nn as nn
        import torch.nn.functional as F

        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = UNet(
                    in_channels=x_train.shape[1],
                    out_channels=out_channels
                )
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])

            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(out, size=self.out_shape, mode="nearest")
                return out

        return WrappedUNet()
    elif cfg.model_type == "cnn":
        from src.models.cnn import CNN

        cnn_mode = getattr(cfg, "cnn_mode", "cnn10")

        return CNN(
            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            out_channels=out_channels,
            output_shape=(y_train.shape[-2], y_train.shape[-1]),
            mode=cnn_mode,
        )
    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not supported yet")


def train_model(cfg, x_train, y_train, lat_in=None, lon_in=None, lat_out=None, lon_out=None):
    vprint("Initializing model for training...")

    if cfg.model_type == "glm":
        from src.models.glm import train_glm
        return train_glm(
            cfg,
            x_train,
            y_train,
            lat_in=lat_in,
            lon_in=lon_in,
            lat_out=lat_out,
            lon_out=lon_out,
            n_neighbors=4,   # GLM4 par défaut
        )

    model = _build_model(cfg, x_train, y_train).to(cfg.device)

    # ----------------
    # Loss & optimizer
    # ----------------
    if cfg.loss_type == "mse":
        criterion = nn.MSELoss()
    elif cfg.loss_type == "bernoulli_gamma":
        from src.core.losses import BernoulliGammaLoss
        criterion = BernoulliGammaLoss()
    elif cfg.loss_type == "gaussian":
        from src.core.losses import GaussianLoss
        criterion = GaussianLoss()
    else:
        raise ValueError(f"Unsupported loss type: {cfg.loss_type}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=cfg.scheduler_factor,
    patience=cfg.scheduler_patience,
    min_lr=cfg.scheduler_min_lr,
    )

    # ----------------
    # DataLoader
    # ----------------
    dataset = TensorDataset(x_train, y_train)

    n_total = len(dataset)
    n_val = max(1, int(cfg.validation_split * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
    )

    train_losses = []
    val_losses = []  # kept for compatibility
    best_loss = float("inf")
    patience = 0
    best_state_dict = None

    # -----------------
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
        # Best model tracking + early stopping
        # ----------------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            vprint(f" New best model at epoch {epoch+1} with loss={best_loss:.6f}")
        else:
            patience += 1
            if patience >= cfg.early_stopping_max:
                vprint("Early stopping triggered.")
                break

    # Build best model copy
    best_model = _build_model(cfg, x_train, y_train).to(cfg.device)
    if best_state_dict is not None:
        best_model.load_state_dict(best_state_dict)
    else:
        best_model.load_state_dict(model.state_dict())

    # model = last model
    # best_model = best model according to current criterion
    return model, best_model, train_losses, val_losses, best_loss