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

import torch
import sys
import os

from  src.core.config import load_config 
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
    X, y_train, y_test = datasets[0], datasets[1], datasets[2]

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
    model, train_losses, val_losses = train_model(
        cfg, x_train_tensor, y_train_tensor
    )

    # ----------------
    # Save model
    # ----------------
    # save_model handled by utils.py, saves to results/<exp>/models/ and saves config as well
    model_path = save_model(cfg, model, train_losses, val_losses)
    vprint(f"Train execution over. Results saved in {cfg.exp_dir}")

    vprint("=== Training completed ===")


if __name__ == "__main__":
    main()
