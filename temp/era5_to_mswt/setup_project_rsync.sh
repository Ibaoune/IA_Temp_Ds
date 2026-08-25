#!/bin/bash
set -e

ROOT="/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling"

# 1. Clean up partial copies
rm -rf $ROOT/univariate/temperature/* || true
rm -rf $ROOT/univariate/precipitation/* || true
rm -rf $ROOT/univariate/evapotranspiration/* || true

# 1. Create structure
mkdir -p $ROOT/univariate/temperature/M0_era5_to_mswt
mkdir -p $ROOT/univariate/temperature/M1_lmdz250_to_lmdz35
mkdir -p $ROOT/univariate/temperature/M2_lmdz35coarse_to_lmdz35

mkdir -p $ROOT/univariate/precipitation/M0_era5_to_mswep
mkdir -p $ROOT/univariate/precipitation/M1_lmdz250_to_lmdz35
mkdir -p $ROOT/univariate/precipitation/M2_lmdz35coarse_to_lmdz35

mkdir -p $ROOT/univariate/evapotranspiration/legacy
mkdir -p $ROOT/univariate/evapotranspiration/M0_reference
mkdir -p $ROOT/univariate/evapotranspiration/M1_lmdz250_to_lmdz35
mkdir -p $ROOT/univariate/evapotranspiration/M2_lmdz35coarse_to_lmdz35

mkdir -p $ROOT/multivariate/M0
mkdir -p $ROOT/multivariate/M1
mkdir -p $ROOT/multivariate/M2

# M0 and legacy (Copy all)
rsync -a --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/ $ROOT/univariate/temperature/M0_era5_to_mswt/
rsync -a --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/precip/ds_precip_intern/precip/era5_mswep/ $ROOT/univariate/precipitation/M0_era5_to_mswep/
rsync -a --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/evap/ds_evap_intern/ $ROOT/univariate/evapotranspiration/legacy/

# Remove diffusion from precip
rm -rf $ROOT/univariate/precipitation/M0_era5_to_mswep/configs/diffusion || true
rm -rf $ROOT/univariate/precipitation/M0_era5_to_mswep/models/diffusion || true
rm -f $ROOT/univariate/precipitation/M0_era5_to_mswep/train_diffusion.py || true
rm -f $ROOT/univariate/precipitation/M0_era5_to_mswep/src/core/diffusion_training.py || true

# M1/M2 skeletons (Exclude large files: .pth, .nc, .pkl)
rsync -a --exclude '*.pth' --exclude '*.nc' --exclude '*.pkl' --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/ $ROOT/univariate/temperature/M1_lmdz250_to_lmdz35/
rsync -a --exclude '*.pth' --exclude '*.nc' --exclude '*.pkl' --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/ $ROOT/univariate/temperature/M2_lmdz35coarse_to_lmdz35/

rsync -a --exclude '*.pth' --exclude '*.nc' --exclude '*.pkl' --exclude '__pycache__' $ROOT/univariate/precipitation/M0_era5_to_mswep/ $ROOT/univariate/precipitation/M1_lmdz250_to_lmdz35/
rsync -a --exclude '*.pth' --exclude '*.nc' --exclude '*.pkl' --exclude '__pycache__' $ROOT/univariate/precipitation/M0_era5_to_mswep/ $ROOT/univariate/precipitation/M2_lmdz35coarse_to_lmdz35/

rsync -a --exclude '*.pth' --exclude '*.nc' --exclude '*.pkl' --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/evap/ds_evap_intern/ $ROOT/univariate/evapotranspiration/M1_lmdz250_to_lmdz35/
rsync -a --exclude '*.pth' --exclude '*.nc' --exclude '*.pkl' --exclude '__pycache__' /srv/data/mohammad.elaabaribao/work/interns/y2026/evap/ds_evap_intern/ $ROOT/univariate/evapotranspiration/M2_lmdz35coarse_to_lmdz35/

echo "Project structure and files synced."
