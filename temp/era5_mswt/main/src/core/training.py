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

import copy
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils import clip_grad_norm_
from src.core.utils import vprint, get_machine_features, estimate_total_time


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
                    out_channels=out_channels,
                    base_filters=64,
                    upscale_factor=1,
                )
                self.out_shape = (y_train.shape[-2], y_train.shape[-1])

            def forward(self, x):
                out = self.unet(x)
                if out.shape[-2:] != self.out_shape:
                    out = F.interpolate(
                        out,
                        size=self.out_shape,
                        mode="bilinear",
                        align_corners=False
                    )
                return out

        return WrappedUNet()
    
    elif cfg.model_type == "unet1":
        from src.models.unet_arch1 import UNet1
        import torch.nn as nn

        class WrappedUNet(nn.Module):
            def __init__(self):
                super().__init__()

                self.out_shape = (
                    y_train.shape[-2],
                    y_train.shape[-1],
                )

                self.unet = UNet1(
                    in_channels=x_train.shape[1],   
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

        model, best_model, train_losses, val_losses, best_loss = train_glm(
            cfg,
            x_train,
            y_train,
            lat_in=lat_in,
            lon_in=lon_in,
            lat_out=lat_out,
            lon_out=lon_out,
            n_neighbors=cfg.glm_n_neighbors,
        )

        best_epoch = None
        return model, best_model, train_losses, val_losses, best_loss, best_epoch

    model = _build_model(cfg, x_train, y_train).to(cfg.device)

    # ----------------
    # Loss
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

    # ----------------
    # Optimizer
    # ----------------
    optimizer_name = getattr(cfg, "optimizer", "adam").lower()
    weight_decay_enable = getattr(cfg, "weight_decay_enable", False)
    weight_decay_value = getattr(cfg, "weight_decay_value", 0.0) if weight_decay_enable else 0.0

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay_value)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay_value)

    # ----------------
    # Dataset / Validation split
    # ----------------
    dataset = TensorDataset(x_train, y_train)

    validation_enable = getattr(cfg, "validation_enable", False)
    validation_percentage = getattr(cfg, "validation_percentage", 0.2)

    if validation_enable and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * validation_percentage))
        if val_size >= len(dataset):
            val_size = len(dataset) - 1
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    # ----------------
    # Scheduler
    # ----------------
    scheduler = None
    scheduler_enable = getattr(cfg, "scheduler_enable", False)
    scheduler_type = getattr(cfg, "scheduler_type", "cosine").lower()
    scheduler_patience = getattr(cfg, "scheduler_patience", 5)
    scheduler_factor = getattr(cfg, "scheduler_factor", 0.5)
    scheduler_min_lr = getattr(cfg, "scheduler_min_lr", 1e-6)

    if scheduler_enable:
        if scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=scheduler_patience,
                factor=scheduler_factor,
                min_lr=scheduler_min_lr,
            )
        elif scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs,
                eta_min=scheduler_min_lr,
            )

    train_losses = []
    val_losses = []
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    best_state_dict = None

    # -----------------
    # Training loop
    # -----------------
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

            if getattr(cfg, "gradient_clipping_enable", False):
                clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, "gradient_clipping_value", 1.0))

            optimizer.step()
            total_loss += loss.item()

        epoch_duration = time.time() - epoch_start_time
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # ----------------
        # Validation
        # ----------------
        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(cfg.device, non_blocking=True)
                    yb = yb.to(cfg.device, non_blocking=True)

                    outputs = model(xb)

                    if cfg.loss_type == "mse":
                        loss = criterion(outputs[:, 0, :, :], yb.squeeze(1))
                    else:
                        loss = criterion(outputs, yb)

                    total_val_loss += loss.item()

            val_loss = total_val_loss / len(val_loader)
            val_losses.append(val_loss)
            vprint(
                f"Epoch {epoch+1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s "
                f"- Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
        else:
            vprint(
                f"Epoch {epoch+1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s "
                f"- Loss: {epoch_loss:.4f}"
            )

        if epoch == 0:
            estimated_time = estimate_total_time(epoch_duration, cfg.epochs)
            vprint(f"Estimated training process duration: {estimated_time}\n")

        monitored_loss = val_loss if val_loss is not None else epoch_loss

        # ----------------
        # Scheduler step
        # ----------------
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]

            if scheduler_type == "plateau":
                scheduler.step(monitored_loss)
            else:
                scheduler.step()

            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != old_lr:
                vprint(f"Learning rate changed: {old_lr:.6e} -> {new_lr:.6e}")
            else:
                vprint(f"Learning rate unchanged: {new_lr:.6e}")

        # ----------------
        # Best model tracking + early stopping
        # ----------------
        if monitored_loss < best_loss:
            best_loss = monitored_loss
            best_epoch = epoch + 1
            patience = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            if val_loss is not None:
                vprint(
                    f"Best model updated at epoch {best_epoch} "
                )
            else:
                vprint(
                    f"Best model updated at epoch {best_epoch} "
                )
        else:
            if getattr(cfg, "early_stopping_enable", False):
                patience += 1
                vprint(
                    f"No improvement at epoch {epoch+1} "
                    f"(patience: {patience}/{getattr(cfg, 'early_stopping_max', 15)})"
                )
                if patience >= getattr(cfg, "early_stopping_max", 15):
                    vprint(
                        f"Early stopping triggered at epoch {epoch+1}/{cfg.epochs}. "
                        f"Best model was found at epoch {best_epoch} "
                        f"with monitored loss = {best_loss:.4f}."
                    )
                    break

    # Build best model copy
    best_model = _build_model(cfg, x_train, y_train).to(cfg.device)
    if best_state_dict is not None:
        best_model.load_state_dict(best_state_dict)
    else:
        best_model.load_state_dict(model.state_dict())

    # model = last model
    # best_model = best model according to current criterion
    return model, best_model, train_losses, val_losses, best_loss, best_epoch