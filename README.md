# Statistical Downscaling over Morocco

This repository contains an ERA5-to-MSWT statistical downscaling workflow for
near-surface temperature over Morocco. The current experiment configurations
use CNN models.

## Repository layout

```text
temp/era5_mswt/
├── main/                  # training, evaluation, models, and configurations
│   ├── configs/
│   │   ├── cnn/           # classical CNN experiments
│   │   └── phy_ai/cnn/    # physics/structure-informed CNN experiments
│   ├── scripts/           # SLURM launchers
│   ├── src/               # pipeline implementation
│   ├── train.py
│   └── eval.py
└── postproc/              # metrics and figures
```

See [the main pipeline documentation](temp/era5_mswt/main/README.md),
[the configuration reference](temp/era5_mswt/main/configs/README.md), and
[the postprocessing guide](temp/era5_mswt/postproc/README.md).

## Quick start

Run commands from the repository root:

```bash
python temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/cnn/test.yaml
python temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/cnn/test.yaml
```

Dataset and output paths are controlled by the selected YAML configuration and
must be adapted to the local environment before a full run.
