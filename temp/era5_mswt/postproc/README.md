# Postprocessing

This directory contains metrics and figures for ERA5-to-MSWT temperature
predictions. All documentation below reflects the current directory tree; no
UNet or GLM configuration is required by the active CNN workflow.

## Structure

```text
postproc/
├── mean/
│   ├── bias/
│   ├── correlation/
│   └── rmse/
├── extreme/
│   ├── b02/
│   ├── b98/
│   ├── cams/
│   └── wams/
├── temporal/
│   ├── ac1/
│   └── rstd/
├── bano_compare/
├── explore.py
├── run_all_postproc.py
├── run_all.sh
└── run_postproc.sh
```

Each metric directory contains a `config.yaml`. Set its `main_config_path` to
the CNN experiment configuration whose predictions must be analysed, for
example:

```yaml
main_config_path: "temp/era5_mswt/main/configs/cnn/config_cnn1.yaml"
```

Do not reuse historical paths under `configs/unet/` or `configs/glm/`; those
configuration families have been removed.

## Run the metric groups

From the repository root:

```bash
bash temp/era5_mswt/postproc/run_postproc.sh
```

The optional second argument selects a group:

```bash
bash temp/era5_mswt/postproc/run_postproc.sh config mean
bash temp/era5_mswt/postproc/run_postproc.sh config extreme
bash temp/era5_mswt/postproc/run_postproc.sh config temporal
```

Supported groups are `all` (default), `mean`, `extreme`, and `temporal`. The
runner currently accepts only the basename `config.yaml` for metric
configuration files.

## Metrics

- Mean metrics: bias, RMSE, and spatial correlation.
- Extreme metrics: B02, B98, CAMS, and WAMS.
- Temporal metrics: lag-one autocorrelation (AC1) and the relative standard
  deviation (RSTD).
- `bano_compare/`: scripts for the Baño-style climatology, metric boxplot,
  spatial-map, and percentile-frequency figures.

Individual modules can also be run from the `temp` directory, which is the
working directory used by `run_postproc.sh`. For example:

```bash
cd temp
python -m era5_mswt.postproc.mean.bias.bias era5_mswt/postproc/mean/bias/config.yaml
python -m era5_mswt.postproc.mean.bias.plot era5_mswt/postproc/mean/bias/config.yaml
```

## Outputs

Modules write their results to the experiment output directories resolved from
the selected main YAML configuration. Depending on the metric, outputs include
JSON or CSV summaries, NetCDF metric fields, and PNG figures.
