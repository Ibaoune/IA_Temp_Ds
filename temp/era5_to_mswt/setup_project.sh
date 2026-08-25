#!/bin/bash
set -e

ROOT="/srv/data/mohammad.elaabaribao/work/interns/y2026/hydroclimate_downscaling"

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

# 2. Temperature
cp -r /srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/* $ROOT/univariate/temperature/M0_era5_to_mswt/
cp -r $ROOT/univariate/temperature/M0_era5_to_mswt/* $ROOT/univariate/temperature/M1_lmdz250_to_lmdz35/
cp -r $ROOT/univariate/temperature/M0_era5_to_mswt/* $ROOT/univariate/temperature/M2_lmdz35coarse_to_lmdz35/

# 3. Precipitation
cp -r /srv/data/mohammad.elaabaribao/work/interns/y2026/precip/ds_precip_intern/precip/era5_mswep/* $ROOT/univariate/precipitation/M0_era5_to_mswep/
# Exclude diffusion
rm -rf $ROOT/univariate/precipitation/M0_era5_to_mswep/configs/diffusion || true
rm -rf $ROOT/univariate/precipitation/M0_era5_to_mswep/models/diffusion || true
rm -f $ROOT/univariate/precipitation/M0_era5_to_mswep/train_diffusion.py || true
rm -f $ROOT/univariate/precipitation/M0_era5_to_mswep/src/core/diffusion_training.py || true

cp -r $ROOT/univariate/precipitation/M0_era5_to_mswep/* $ROOT/univariate/precipitation/M1_lmdz250_to_lmdz35/
cp -r $ROOT/univariate/precipitation/M0_era5_to_mswep/* $ROOT/univariate/precipitation/M2_lmdz35coarse_to_lmdz35/

# 4. ET legacy
cp -r /srv/data/mohammad.elaabaribao/work/interns/y2026/evap/ds_evap_intern/* $ROOT/univariate/evapotranspiration/legacy/
cp -r $ROOT/univariate/evapotranspiration/legacy/* $ROOT/univariate/evapotranspiration/M1_lmdz250_to_lmdz35/
cp -r $ROOT/univariate/evapotranspiration/legacy/* $ROOT/univariate/evapotranspiration/M2_lmdz35coarse_to_lmdz35/

# 5. Patch ET unit conversion in legacy (and inherited M1/M2)
# In evap repository, data_loading.py line ~45 handles Y_all = evap_lmdz35.values
# The user said: "Implement this correction only in the NEW copied ET project."
# Let's apply a perl one-liner or sed, but wait, the instructions forbid `sed`.
# I will use multi_replace_file_content or a python script to patch it safely.

echo "Setup script finished successfully."
