#!/bin/bash
#SBATCH --job-name=postproc_multi
#SBATCH --output=run_all_postproc_%j.out
#SBATCH --error=run_all_postproc_%j.err
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh || true
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

# Ensure we're in the postproc directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
else
    cd "$(dirname "$0")" || exit 1
fi
POSTPROC_DIR=$(pwd)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Starting multi-model postprocessing"
echo "======================================"

# 1. Calculate individual metrics
echo ""
echo "[1/3] Calculating individual metrics for all models..."
python3 -u run_all_postproc.py

# 2. Go to parent directory to run Python modules properly (package structure)
cd ..

# 3. Boxplots comparison
echo ""
echo "[2/3] Generating multi-model Boxplots..."
python3 -u -m era5_mswt.postproc.bano_compare.multi_model_boxplots

# 4. Maps comparison
echo ""
echo "[3/3] Generating multi-model Maps..."
python3 -u -m era5_mswt.postproc.bano_compare.multi_model_maps

echo ""
echo "======================================"
echo "Done: $(date)"
echo "======================================"
