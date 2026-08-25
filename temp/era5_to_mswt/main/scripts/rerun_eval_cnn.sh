#!/bin/bash

# ==============================================================================
# Script: rerun_eval_cnn.sh
# Description: Relance eval.py uniquement pour les 4 configs CNN qui ont des
#              modeles .pth mais pas encore de predictions/figures.
# ==============================================================================

cd "$(dirname "$0")/.." || exit 1

CONDA_ENV="clean_env_Pytorch"
LOGS_DIR="logs/cnn"
MAIN_DIR=$(pwd)

mkdir -p $LOGS_DIR

CONFIGS=(
    "configs/cnn/config.yaml"
    "configs/cnn/config_cnn1.yaml"
    "configs/cnn/config_cnn1_mse.yaml"
    "configs/cnn/config_cnn10_mse.yaml"
)

echo "Re-running eval.py for CNN configs (models already trained)..."
echo "========================================="

for config_file in "${CONFIGS[@]}"; do
    filename=$(basename -- "$config_file")
    job_name="eval_cnn_${filename%.*}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$MAIN_DIR/$LOGS_DIR/%x_%j.out
#SBATCH --error=$MAIN_DIR/$LOGS_DIR/%x_%j.err
#SBATCH --account=climat-7ksifkvwkuy-default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh || true
conda activate $CONDA_ENV

export PYTHONUNBUFFERED=1
cd $MAIN_DIR

echo "======================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Start: \$(date)"
echo "Config: $config_file"
echo "======================================"

echo "[INFO] Running eval only (model already trained)..."
python3 -u eval.py $config_file

echo "======================================"
echo "Done: \$(date)"
echo "======================================"
EOT

    echo "  [OK] Submitted: $job_name"
done

echo "========================================="
echo "Eval re-run jobs submitted."
echo "Outputs -> results/cnn/<exp>/output_data/ & output_figs/"
