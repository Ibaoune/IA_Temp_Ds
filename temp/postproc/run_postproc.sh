#!/bin/bash

set -e

# ==========================================================
# General postprocessing runner
# Usage:
#   bash temp/postproc/run_postproc.sh lmdz250_to_lmdz35
#   bash temp/postproc/run_postproc.sh lmdz35_2deg_to_lmdz35
#
# Optional:
#   bash temp/postproc/run_postproc.sh lmdz250_to_lmdz35 mean
#   bash temp/postproc/run_postproc.sh lmdz250_to_lmdz35 extreme
#   bash temp/postproc/run_postproc.sh lmdz250_to_lmdz35 temporal
# ==========================================================

echo "======================================"
echo " Running postprocessing metrics"
echo "======================================"

# ----------------------------------------------------------
# Go to project root
# This script is assumed to be in: temp/postproc/run_postproc.sh
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
    echo ""
    echo "Choose a postprocessing config:"
    echo "1) lmdz250_to_lmdz35"
    echo "2) lmdz35_2deg_to_lmdz35"
    echo "3) era5_to_mswt"
    echo ""
    read -p "Enter choice [1-3]: " choice

    case "$choice" in
        1)
            CONFIG_NAME="lmdz250_to_lmdz35"
            ;;
        2)
            CONFIG_NAME="lmdz35_2deg_to_lmdz35"
            ;;
        3)
            CONFIG_NAME="era5_to_mswt"
            ;;
        *)
            echo "[ERROR] Invalid choice."
            exit 1
            ;;
    esac
fi

# Add .yaml if user did not provide it
if [[ "$CONFIG_NAME" != *.yaml ]]; then
    CONFIG_NAME="${CONFIG_NAME}.yaml"
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
        "temp.postproc.mean.bias.bias" \
        "temp.postproc.mean.bias.plot" \
        "temp/postproc/mean/bias/$CONFIG_NAME"

    run_metric \
        "RMSE" \
        "temp.postproc.mean.rmse.rmse" \
        "temp.postproc.mean.rmse.plot" \
        "temp/postproc/mean/rmse/$CONFIG_NAME"

    run_metric \
        "CORRELATION" \
        "temp.postproc.mean.correlation.corr" \
        "temp.postproc.mean.correlation.plot" \
        "temp/postproc/mean/correlation/$CONFIG_NAME"
}

# ==========================================================
# EXTREME METRICS
# ==========================================================
run_extreme_metrics () {
    echo ""
    echo "========== EXTREME METRICS =========="

    run_metric \
        "B02" \
        "temp.postproc.extreme.b02.b02" \
        "temp.postproc.extreme.b02.plot" \
        "temp/postproc/extreme/b02/$CONFIG_NAME"

    run_metric \
        "B98" \
        "temp.postproc.extreme.b98.b98" \
        "temp.postproc.extreme.b98.plot" \
        "temp/postproc/extreme/b98/$CONFIG_NAME"

    run_metric \
        "CAMS" \
        "temp.postproc.extreme.cams.cams" \
        "temp.postproc.extreme.cams.plot" \
        "temp/postproc/extreme/cams/$CONFIG_NAME"

    run_metric \
        "WAMS" \
        "temp.postproc.extreme.wams.wams" \
        "temp.postproc.extreme.wams.plot" \
        "temp/postproc/extreme/wams/$CONFIG_NAME"
}

# ==========================================================
# TEMPORAL METRICS
# ==========================================================
run_temporal_metrics () {
    echo ""
    echo "========== TEMPORAL METRICS =========="

    run_metric \
        "AC1" \
        "temp.postproc.temporal.ac1.ac1" \
        "temp.postproc.temporal.ac1.plot" \
        "temp/postproc/temporal/ac1/$CONFIG_NAME"

    run_metric \
        "RSTD" \
        "temp.postproc.temporal.rstd.rstd" \
        "temp.postproc.temporal.rstd.plot" \
        "temp/postproc/temporal/rstd/$CONFIG_NAME"
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