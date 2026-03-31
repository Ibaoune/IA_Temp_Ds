"""
==========================================================
 Script: eval.py
 Author: M. El Aabaribaoune
 Description:
     Main evaluation entry point.

     - Loads configuration
     - Loads test data
     - Preprocesses inputs
     - Runs model evaluation and saves outputs
==========================================================
"""

from config import load_config
from data_loading import load_datasets
from preprocessing import preprocess_data
from evaluation import evaluate_and_save
from utils import vprint

from utils import set_verbose

cfg = load_config("config.yaml")
set_verbose(cfg.verbose)

def main():
    cfg = load_config(train_mode=False)
    vprint("=== Starting evaluation process ===")

     
    # Load and preprocess data
     
    (
        X,
        y_train,
        y_test,
        lon_in,
        lat_in,
        lon_out,
        lat_out,
        time_train,
        time_test,
    ) = load_datasets(cfg)

    _, x_test_tensor, _, y_test_tensor = preprocess_data(
        cfg, X, y_train, y_test
    )

     
    # Run evaluation
     
    evaluate_and_save(
        cfg,
        x_test_tensor,
        y_test_tensor,
        lon_out,
        lat_out,
        time_test,
    )

    vprint("Test data saved successfully")
    vprint("=== Evaluation completed ===")


if __name__ == "__main__":
    main()
