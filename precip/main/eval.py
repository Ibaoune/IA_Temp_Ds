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

from  src.core.config import load_config 
from src.data.data_loading import load_datasets
from src.data.preprocessing import preprocess_data
from src.core.evaluation import evaluate_and_save
from src.core.utils import vprint, set_verbose


def main():
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(train_mode=False, path=cfg_path)
    set_verbose(cfg.verbose)
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
