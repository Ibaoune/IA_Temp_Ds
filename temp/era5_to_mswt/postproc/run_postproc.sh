#!/bin/bash

set -e

# ==========================================================
# General postprocessing runner
# Usage:
#   bash era5_mswt/postproc/run_postproc.sh
#   bash era5_mswt/postproc/run_postproc.sh config
#
# Optional:
#   bash era5_mswt/postproc/run_postproc.sh config mean
#   bash era5_mswt/postproc/run_postproc.sh config extreme
#   bash era5_mswt/postproc/run_postproc.sh config temporal
# ==========================================================

echo "======================================"
echo " Running postprocessing metrics"
echo "======================================"

# ----------------------------------------------------------
# Go to project root
# This script is assumed to be in: era5_mswt/postproc/run_postproc.sh
# ----------------------------------------------------------
cd "$(dirname "$0")/../.."

echo "[INFO] Current directory:"
pwd

# ----------------------------------------------------------
# Read arguments
# ----------------------------------------------------------
CONFIG_NAME="$1"
GROUP="$2"

if [ -z "$CONFIG_NAME" ]; then
    CONFIG_NAME="config"
fi

# Add .yaml if user did not provide it
if [[ "$CONFIG_NAME" != *.yaml ]]; then
    CONFIG_NAME="${CONFIG_NAME}.yaml"
fi

if [[ "$CONFIG_NAME" != "config.yaml" ]]; then
    echo "[WARNING] Config '$CONFIG_NAME' is not supported by the current project."
    echo "[WARNING] Ignoring it and using config.yaml."
    CONFIG_NAME="config.yaml"
fi

# Default group = all
if [ -z "$GROUP" ]; then
    GROUP="all"
fi

echo ""
echo "[INFO] Selected config : $CONFIG_NAME"
echo "[INFO] Selected group  : $GROUP"
echo ""

# ----------------------------------------------------------
# Helper function
# ----------------------------------------------------------
run_metric () {
    METRIC_LABEL="$1"
    COMPUTE_MODULE="$2"
    PLOT_MODULE="$3"
    CONFIG_PATH="$4"

    echo ""
    echo "--------------------------------------"
    echo "[$METRIC_LABEL]"
    echo "Config: $CONFIG_PATH"
    echo "--------------------------------------"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "[WARNING] Config not found: $CONFIG_PATH"
        echo "[WARNING] Skipping $METRIC_LABEL"
        return
    fi

    echo "[$METRIC_LABEL] compute"
    python -m "$COMPUTE_MODULE" "$CONFIG_PATH"

    echo "[$METRIC_LABEL] plot"
    python -m "$PLOT_MODULE" "$CONFIG_PATH"

    echo "[$METRIC_LABEL] done"
}

# ==========================================================
# MEAN METRICS
# ==========================================================
run_mean_metrics () {
    echo ""
    echo "========== MEAN METRICS =========="

    run_metric \
        "BIAS" \
        "era5_mswt.postproc.mean.bias.bias" \
        "era5_mswt.postproc.mean.bias.plot" \
        "era5_mswt/postproc/mean/bias/$CONFIG_NAME"

    run_metric \
        "RMSE" \
        "era5_mswt.postproc.mean.rmse.rmse" \
        "era5_mswt.postproc.mean.rmse.plot" \
        "era5_mswt/postproc/mean/rmse/$CONFIG_NAME"

    run_metric \
        "CORRELATION" \
        "era5_mswt.postproc.mean.correlation.corr" \
        "era5_mswt.postproc.mean.correlation.plot" \
        "era5_mswt/postproc/mean/correlation/$CONFIG_NAME"
}

# ==========================================================
# EXTREME METRICS
# ==========================================================
run_extreme_metrics () {
    echo ""
    echo "========== EXTREME METRICS =========="

    run_metric \
        "B02" \
        "era5_mswt.postproc.extreme.b02.b02" \
        "era5_mswt.postproc.extreme.b02.plot" \
        "era5_mswt/postproc/extreme/b02/$CONFIG_NAME"

    run_metric \
        "B98" \
        "era5_mswt.postproc.extreme.b98.b98" \
        "era5_mswt.postproc.extreme.b98.plot" \
        "era5_mswt/postproc/extreme/b98/$CONFIG_NAME"

    run_metric \
        "CAMS" \
        "era5_mswt.postproc.extreme.cams.cams" \
        "era5_mswt.postproc.extreme.cams.plot" \
        "era5_mswt/postproc/extreme/cams/$CONFIG_NAME"

    run_metric \
        "WAMS" \
        "era5_mswt.postproc.extreme.wams.wams" \
        "era5_mswt.postproc.extreme.wams.plot" \
        "era5_mswt/postproc/extreme/wams/$CONFIG_NAME"
}

# ==========================================================
# TEMPORAL METRICS
# ==========================================================
run_temporal_metrics () {
    echo ""
    echo "========== TEMPORAL METRICS =========="

    run_metric \
        "AC1" \
        "era5_mswt.postproc.temporal.ac1.ac1" \
        "era5_mswt.postproc.temporal.ac1.plot" \
        "era5_mswt/postproc/temporal/ac1/$CONFIG_NAME"

    run_metric \
        "RSTD" \
        "era5_mswt.postproc.temporal.rstd.rstd" \
        "era5_mswt.postproc.temporal.rstd.plot" \
        "era5_mswt/postproc/temporal/rstd/$CONFIG_NAME"
}

# ----------------------------------------------------------
# Execute selected group
# ----------------------------------------------------------
case "$GROUP" in
    all)
        run_mean_metrics
        run_extreme_metrics
        run_temporal_metrics
        ;;
    mean)
        run_mean_metrics
        ;;
    extreme)
        run_extreme_metrics
        ;;
    temporal)
        run_temporal_metrics
        ;;
    *)
        echo "[ERROR] Unknown group: $GROUP"
        echo "Allowed groups: all, mean, extreme, temporal"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo " Postprocessing finished successfully"
echo "======================================"
