"""
==========================================================
 Script: train.py
 Author: M. El Aabaribaoune
 Description:
     Main entry point for training the downscaling model.
     - Loads configuration
     - Loads and preprocesses data
     - Trains the model
     - Saves trained weights and losses

 Notes:
     - Compatible CPU / GPU
     - Model-agnostic (ViT by default)
==========================================================
"""
import sys

from src.core.config import load_config
from src.data.data_loading import load_datasets
from src.data.preprocessing import preprocess_data
from src.core.training import train_model
from src.core.utils import vprint, save_model, set_verbose

def main():
    # ----------------
    # Load configuration
    # ----------------
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    cfg = load_config(train_mode=True, path=cfg_path)
    set_verbose(cfg.verbose)
    
    vprint(f"Using device: {cfg.device}")
    vprint(f"=== Starting training process for {cfg.variable} ===")

    # ----------------
    # Load and preprocess data
    # ----------------
    # Load datasets returns: X, y_train, y_test, lon_in, lat_in, lon_out, lat_out, time_train, time_test
    datasets = load_datasets(cfg)
    X, y_train, y_test, lon_in, lat_in, lon_out, lat_out = (
        datasets[0], datasets[1], datasets[2],
        datasets[3], datasets[4], datasets[5], datasets[6]
    )

    # Preprocess data returns: x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor
    x_train_tensor, _, y_train_tensor, _ = preprocess_data(
        cfg, X, y_train, y_test
    )

    # Ensure tensors are on correct device
    x_train_tensor = x_train_tensor.to(cfg.device)
    y_train_tensor = y_train_tensor.to(cfg.device)

    # ----------------
    # Train model
    # ----------------
    last_model, best_model, train_losses, val_losses, best_loss, best_epoch = train_model(
        cfg,
        x_train_tensor,
        y_train_tensor,
        lat_in=lat_in,
        lon_in=lon_in,
        lat_out=lat_out,
        lon_out=lon_out,
    )

    # ----------------
    # Save model(s)
    # ----------------
    if cfg.model_type == "glm":
        glm_model_path = save_model(
            cfg,
            last_model,
            train_losses=train_losses,
            val_losses=val_losses,
            tag="last",
            best_score=best_loss,
        )
        vprint(f"GLM model saved at: {glm_model_path}")

    else:
        last_model_path = save_model(
            cfg,
            last_model,
            train_losses=train_losses,
            val_losses=val_losses,
            tag="last",
            best_score=best_loss,
        )

        best_model_path = save_model(
            cfg,
            best_model,
            train_losses=train_losses,
            val_losses=val_losses,
            tag="best",
            best_score=best_loss,
        )

    vprint("\n=========== Training Summary ===========")
    if best_epoch is not None:
        vprint(f"Best epoch         : {best_epoch}")

    if cfg.model_type == "glm":
        vprint(f"GLM model saved    : {glm_model_path}")
    else:
        if best_loss is not None:
            vprint(f"Best monitored loss: {best_loss:.4f}")
        vprint(f"LAST model saved   : {last_model_path}")
        vprint(f"BEST model saved   : {best_model_path}")

    vprint(f"Results directory  : {cfg.exp_dir}")
    vprint("========================================")


if __name__ == "__main__":
    main()
