#!/bin/bash

# ==============================================================================
# Script: submit_phy_ai_jobs.sh
# Description: Soumet les configs phy_ai/cnn *_test (légères) sur CPU pour test.
#              Une fois validées, utiliser submit_phy_ai_gpu_jobs.sh pour les vraies simu.
# Environment: clean_env_Pytorch
# ==============================================================================

cd "$(dirname "$0")/.." || exit 1

CONDA_ENV="clean_env_Pytorch"
CONFIGS_DIR="configs/phy_ai/cnn"
LOGS_DIR="logs/phy_ai"
TRAIN_SCRIPT="train.py"
MAIN_DIR=$(pwd)

mkdir -p $LOGS_DIR

echo "Submitting phy_ai TEST configs on CPU..."
echo "========================================="

for config_file in $CONFIGS_DIR/*_test.yaml; do
    filename=$(basename -- "$config_file")
    job_name="phyai_test_${filename%.*}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$MAIN_DIR/$LOGS_DIR/%x_%j.out
#SBATCH --error=$MAIN_DIR/$LOGS_DIR/%x_%j.err
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU
#SBATCH --partition=compute
#SBATCH --nodes=1
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

echo "[INFO] Training (test)..."
python3 -u $TRAIN_SCRIPT $config_file

echo "[INFO] Evaluation (test)..."
python3 -u eval.py $config_file

echo "======================================"
echo "Done: \$(date)"
echo "======================================"
EOT

    echo "  [OK] Submitted: $job_name"
done

echo "========================================="
echo "All phy_ai TEST jobs submitted."
echo "Check logs in: $LOGS_DIR/"
echo "Once validated -> run: scripts/submit_phy_ai_gpu_jobs.sh"
