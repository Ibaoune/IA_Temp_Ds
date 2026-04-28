#!/bin/bash

set -e

echo "======================================"
echo " Running all postprocessing metrics"
echo "======================================"


cd "$(dirname "$0")/../.."

echo "[INFO] Current directory:"
pwd

echo ""
echo "========== MEAN METRICS =========="

echo "[BIAS] compute"
python -m temp.postproc.mean.bias.bias
echo "[BIAS] plot"
python -m temp.postproc.mean.bias.plot

echo "[RMSE] compute"
python -m temp.postproc.mean.rmse.rmse
echo "[RMSE] plot"
python -m temp.postproc.mean.rmse.plot

echo "[CORRELATION] compute"
python -m temp.postproc.mean.correlation.corr
echo "[CORRELATION] plot"
python -m temp.postproc.mean.correlation.plot

echo "[QQ] compute"
python -m temp.postproc.mean.QQ.compute_qq_data
echo "[QQ] plot"
python -m temp.postproc.mean.QQ.plot_qq


echo ""
echo "========== EXTREME METRICS =========="

echo "[B02] compute"
python -m temp.postproc.extreme.b02.b02
echo "[B02] plot"
python -m temp.postproc.extreme.b02.plot

echo "[B98] compute"
python -m temp.postproc.extreme.b98.b98
echo "[B98] plot"
python -m temp.postproc.extreme.b98.plot

echo "[CAMS] compute"
python -m temp.postproc.extreme.cams.cams
echo "[CAMS] plot"
python -m temp.postproc.extreme.cams.plot

echo "[WAMS] compute"
python -m temp.postproc.extreme.wams.wams
echo "[WAMS] plot"
python -m temp.postproc.extreme.wams.plot


echo ""
echo "========== TEMPORAL METRICS =========="

echo "[AC1] compute"
python -m temp.postproc.temporal.ac1.ac1
echo "[AC1] plot"
python -m temp.postproc.temporal.ac1.plot

echo "[RSTD] compute"
python -m temp.postproc.temporal.rstd.rstd
echo "[RSTD] plot"
python -m temp.postproc.temporal.rstd.plot


echo ""
echo "======================================"
echo " Postprocessing finished successfully"
echo "======================================"