#!/bin/bash

# ==============================================================================
# Script: submit_phy_ai_gpu_jobs.sh
# Description: Soumet les VRAIES configs phy_ai/cnn (sans _test) sur GPU.
#              A lancer UNIQUEMENT après validation des configs _test sur CPU.
# Environment: clean_env_Pytorch
# ==============================================================================

cd "$(dirname "$0")/.." || exit 1

CONDA_ENV="clean_env_Pytorch"
CONFIGS_DIR="configs/phy_ai/cnn"
LOGS_DIR="logs/phy_ai"
TRAIN_SCRIPT="train.py"
MAIN_DIR=$(pwd)

mkdir -p $LOGS_DIR

echo "Submitting phy_ai FULL configs on GPU..."
echo "========================================="

for config_file in $CONFIGS_DIR/*.yaml; do
    filename=$(basename -- "$config_file")

    # Skip les configs test (légères)
    if [[ "$filename" == *"_test.yaml" ]]; then
        echo "  [SKIP] $filename"
        continue
    fi

    job_name="phyai_gpu_${filename%.*}"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$MAIN_DIR/$LOGS_DIR/%x_%j.out
#SBATCH --error=$MAIN_DIR/$LOGS_DIR/%x_%j.err
#SBATCH --account=climat-7ksifkvwkuy-default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh || true
conda activate $CONDA_ENV

export PYTHONUNBUFFERED=1
cd $MAIN_DIR

echo "======================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Start: \$(date)"
echo "Config: $config_file"
echo "======================================"

echo "[INFO] Training (full)..."
python3 -u $TRAIN_SCRIPT $config_file

echo "[INFO] Evaluation (full)..."
python3 -u eval.py $config_file

echo "======================================"
echo "Done: \$(date)"
echo "======================================"
EOT

    echo "  [OK] Submitted: $job_name"
done

echo "========================================="
echo "All phy_ai GPU jobs submitted."
echo "Check logs in: $LOGS_DIR/"
