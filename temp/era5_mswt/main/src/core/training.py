"""
==========================================================
 Script: training.py
 Author: M. El Aabaribaoune
 Description:
     Defines the training loop, loss computation, and
     early stopping for the downscaling model.

 Design:
     - Model-agnostic (CNN, UNet, UNet1, GLM, etc.)
     - GPU / CPU compatible
     - DataLoader-safe: tensors are kept on CPU, then batches
       are moved to GPU inside the training loop.
==========================================================
"""

import copy
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils import clip_grad_norm_
from src.core.utils import vprint, get_machine_features, estimate_total_time


# ==========================================================
# Helpers
# ==========================================================

def _get_device(cfg):
    """
    Returns torch.device safely, whether cfg.device is a string or torch.device.
    """
    if isinstance(cfg.device, torch.device):
        return cfg.device
    return torch.device(cfg.device)


def _device_is_cuda(device):
    """
    Checks if the selected device is CUDA.
    """
    return isinstance(device, torch.device) and device.type == "cuda"


def _to_cpu_float_tensor(x, name="tensor"):
    """
    Ensures tensors used by TensorDataset/DataLoader are CPU tensors.

    Important:
    DataLoader pin_memory works only with CPU tensors.
    If x is already on CUDA, we move it back to CPU here.
    The batch is moved to CUDA later inside the training loop.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.is_cuda:
        vprint(f"{name} is on CUDA -> moving it back to CPU for DataLoader safety.")

    x = x.detach().to("cpu", dtype=torch.float32).contiguous()
    return x


def _prepare_mse_loss(outputs, yb):
    """
    MSE case:
    outputs can be (B, 1, H, W) or (B, H, W)
    yb can be (B, H, W) or (B, 1, H, W)
    """
    if outputs.ndim == 4:
        pred = outputs[:, 0, :, :]
    else:
        pred = outputs

    if yb.ndim == 4 and yb.shape[1] == 1:
        target = yb[:, 0, :, :]
    else:
        target = yb

    return pred, target


def _prepare_probabilistic_target(yb):
    """
    GaussianLoss / BernoulliGammaLoss generally expect target shape:
    (B, 1, H, W)
    """
    if yb.ndim == 3:
        yb = yb.unsqueeze(1)
    return yb


def _copy_state_dict_to_cpu(model):
    """
    Stores the best weights on CPU to avoid unnecessary GPU memory usage.
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def _make_loader(dataset, batch_size, shuffle, drop_last, use_cuda, cfg):
    """
    Creates a DataLoader safely.

    Since tensors are kept on CPU, pin_memory can be used with CUDA.
    If you still get RAM pressure in Colab, set cfg.pin_memory_enable = False.
    """
    pin_memory_enable = getattr(cfg, "pin_memory_enable", True)
    num_workers = getattr(cfg, "num_workers", 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=(use_cuda and pin_memory_enable),
        num_workers=num_workers,
    )


# ==========================================================
# Model builder
# ==========================================================

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
                        align_corners=False,
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
                    norm_type="group" if getattr(cfg, "group_norm_enable", False) else "batch",
                    num_groups=getattr(cfg, "group_norm_num_groups", 8),
                    dropout=getattr(cfg, "dropout_value", 0.0)
                    if getattr(cfg, "dropout_enable", False)
                    else 0.0,
                )

            def forward(self, x):
                return self.unet(
                    x,
                    target_size=self.out_shape,
                )

        return WrappedUNet()

    elif cfg.model_type in {"cnn", "cnn1", "cnn10"}:
        from src.models.cnn import CNN

        if cfg.model_type in {"cnn1", "cnn10"}:
            cnn_mode = cfg.model_type
        else:
            cnn_mode = getattr(cfg, "cnn_mode", "cnn10")

        return CNN(
            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            out_channels=out_channels,
            output_shape=(y_train.shape[-2], y_train.shape[-1]),
            mode=cnn_mode,
        )

    else:
        raise NotImplementedError(f"Model type {cfg.model_type} not supported yet")


# ==========================================================
# Training function
# ==========================================================

def train_model(cfg, x_train, y_train, lat_in=None, lon_in=None, lat_out=None, lon_out=None):
    vprint("Initializing model for training...")

    device = _get_device(cfg)
    use_cuda = _device_is_cuda(device)

    # ----------------
    # GLM case
    # ----------------
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
            n_neighbors=getattr(cfg, "glm_n_neighbors", 4),
        )

        best_epoch = None
        return model, best_model, train_losses, val_losses, best_loss, best_epoch

    # ======================================================
    # IMPORTANT FIX
    # ======================================================
    # Keep full dataset on CPU.
    # DataLoader works with CPU tensors.
    # Each batch is moved to GPU inside the loop.
    x_train = _to_cpu_float_tensor(x_train, name="x_train")
    y_train = _to_cpu_float_tensor(y_train, name="y_train")

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "x_train and y_train do not have the same number of samples: "
            f"x_train={x_train.shape[0]}, y_train={y_train.shape[0]}"
        )

    # ----------------
    # Model
    # ----------------
    model = _build_model(cfg, x_train, y_train).to(device)

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
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=weight_decay_value,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=weight_decay_value,
        )

    # ----------------
    # Dataset / validation split
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
            generator=torch.Generator().manual_seed(getattr(cfg, "seed", 42)),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = _make_loader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        use_cuda=use_cuda,
        cfg=cfg,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = _make_loader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
            use_cuda=use_cuda,
            cfg=cfg,
        )

    if len(train_loader) == 0:
        raise RuntimeError(
            "train_loader is empty. Reduce batch_size or check dataset length."
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
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    # ----------------
    # Tracking
    # ----------------
    train_losses = []
    val_losses = []

    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    best_state_dict = None

    # ----------------
    # Training loop
    # ----------------
    for epoch in range(cfg.epochs):

        if epoch == 0:
            features = get_machine_features()
            vprint("\n----------- Machine Features -----------")
            vprint(f" OS: {features['os']}")
            vprint(f" CPU: {features['cpu']} ({features['cores']} cores)")
            vprint(f" RAM: {features['ram_total_gb']} GB")

            if features.get("gpus"):
                for idx, gpu in enumerate(features["gpus"]):
                    vprint(f" GPU {idx}: {gpu['name']} ({gpu['memory_total_gb']} GB)")
            else:
                vprint(" GPU: Not available")

            vprint("----------------------------------------")
            vprint(f"Training samples   : {len(train_dataset)}")
            if val_dataset is not None:
                vprint(f"Validation samples : {len(val_dataset)}")
            vprint(f"Batch size         : {cfg.batch_size}")
            vprint(f"Train batches      : {len(train_loader)}")
            vprint(f"Device             : {device}")
            vprint("----------------------------------------")

        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(xb)

            if cfg.loss_type == "mse":
                pred, target = _prepare_mse_loss(outputs, yb)
                loss = criterion(pred, target)
            else:
                target = _prepare_probabilistic_target(yb)
                loss = criterion(outputs, target)

            loss.backward()

            if getattr(cfg, "gradient_clipping_enable", False):
                clip_grad_norm_(
                    model.parameters(),
                    max_norm=getattr(cfg, "gradient_clipping_value", 1.0),
                )

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
                    xb = xb.to(device, non_blocking=use_cuda)
                    yb = yb.to(device, non_blocking=use_cuda)

                    outputs = model(xb)

                    if cfg.loss_type == "mse":
                        pred, target = _prepare_mse_loss(outputs, yb)
                        loss = criterion(pred, target)
                    else:
                        target = _prepare_probabilistic_target(yb)
                        loss = criterion(outputs, target)

                    total_val_loss += loss.item()

            val_loss = total_val_loss / len(val_loader)
            val_losses.append(val_loss)

            vprint(
                f"Epoch {epoch + 1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s "
                f"- Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

        else:
            vprint(
                f"Epoch {epoch + 1}/{cfg.epochs} - Duration: {epoch_duration:.2f}s "
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

            # Store best weights on CPU, safer for GPU memory
            best_state_dict = _copy_state_dict_to_cpu(model)

            if val_loss is not None:
                vprint(
                    f"Best model updated at epoch {best_epoch} "
                    f"(Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f})"
                )
            else:
                vprint(
                    f"Best model updated at epoch {best_epoch} "
                    f"(Loss: {epoch_loss:.4f})"
                )

        else:
            if getattr(cfg, "early_stopping_enable", False):
                patience += 1
                vprint(
                    f"No improvement at epoch {epoch + 1} "
                    f"(patience: {patience}/{getattr(cfg, 'early_stopping_max', 15)})"
                )

                if patience >= getattr(cfg, "early_stopping_max", 15):
                    vprint(
                        f"Early stopping triggered at epoch {epoch + 1}/{cfg.epochs}. "
                        f"Best model was found at epoch {best_epoch} "
                        f"with monitored loss = {best_loss:.4f}."
                    )
                    break

    # ----------------
    # Build best model copy
    # ----------------
    best_model = _build_model(cfg, x_train, y_train).to(device)

    if best_state_dict is not None:
        best_model.load_state_dict(best_state_dict)
    else:
        best_model.load_state_dict(model.state_dict())

    # model      = last model
    # best_model = best model according to monitored criterion
    return model, best_model, train_losses, val_losses, best_loss, best_epoch