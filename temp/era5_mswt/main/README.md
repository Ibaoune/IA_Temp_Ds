## Structure

```text
main/
├── configs/
│   ├── cnn/               # classical CNN1 and CNN10 configurations
│   └── phy_ai/
│       └── cnn/           # Xiong and Serifi loss experiments
├── scripts/
│   ├── job_cpu.sh
│   └── job_gpu.sh
├── src/
│   ├── core/              # configuration, data, training, evaluation, losses
│   └── models/            # model definitions
├── tests/
├── train.py
└── eval.py
```

The complete configuration inventory and the mathematical definitions of the
physics/structure-informed losses are in [configs/README.md](configs/README.md).

## Data and models

The configured mapping is:

```text
ERA5 predictors -> MSWT temperature
```

The active CNN configurations use ERA5 `z`, `q`, `t`, `u`, and `v` at 500,
700, 850, and 1000 hPa. `general.model_type` selects `cnn1` or `cnn10`, while
the `cnn.mode` field selects the matching CNN architecture.

A typical configuration begins with:

```yaml
general:
  src: "era5"
  target: "mswt"
  variable: "temp"
  model_type: "cnn1"
  variables: ["z", "q", "t", "u", "v"]
  levels: [500, 700, 850, 1000]

cnn:
  mode: "cnn1"
```

Update the paths under `paths` in the selected YAML file before running an
experiment.

## Classical configurations

`configs/cnn/` contains Gaussian negative-log-likelihood and deterministic MSE
experiments for CNN1 and CNN10, batch-size variants, and `test.yaml` for a
reduced run. The former UNet and GLM experiments are intentionally not listed
because their configuration folders were removed.

## Physics/structure-informed configurations

`configs/phy_ai/cnn/` contains three independent loss experiments for each of
CNN1 and CNN10:

- `xiong_continuity`: matches the spatial continuity energy of the prediction
  and target.
- `xiong_directional`: matches their aggregate neighbour-to-neighbour
  directional quantity.
- `serifi_gradient`: matches field values and local horizontal/vertical
  gradients.

Each full experiment also has a `_test.yaml` variant. The three constraints are
tested separately and are never combined in the supplied configurations.

## Training and evaluation

Run from the repository root. Replace `<config>` with a filename in
`configs/cnn/` or `configs/phy_ai/cnn/`.

Classical example:

```bash
python temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/cnn/test.yaml
python temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/cnn/test.yaml
```

Physics-informed example:

```bash
python temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn1_xiong_continuity_test.yaml
python temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn1_xiong_continuity_test.yaml
```

On Windows, the same commands can be run with the project interpreter, for
example `./climate_env/Scripts/python.exe`, if that environment exists.

## SLURM jobs

The CPU and GPU launchers accept the configuration through `MAIN_CONFIG`.
Always set it explicitly because historical defaults inside the scripts may
refer to removed configuration paths:

```bash
MAIN_CONFIG=temp/era5_mswt/main/configs/cnn/test.yaml \
  bash temp/era5_mswt/main/scripts/job_cpu.sh

MAIN_CONFIG=temp/era5_mswt/main/configs/phy_ai/cnn/config_cnn10_serifi_gradient.yaml \
  bash temp/era5_mswt/main/scripts/job_gpu.sh
```

## Outputs

Training and evaluation write under `paths.results_dir` using the configured
experiment name. Typical outputs include saved models, prediction data,
training histories, metrics, and plots.
