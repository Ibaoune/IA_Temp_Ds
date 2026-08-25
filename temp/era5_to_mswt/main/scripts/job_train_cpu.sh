#!/bin/bash
#SBATCH --job-name=cnn_cpu
#SBATCH --output=logs/out_cpu_%j.log
#SBATCH --error=logs/err_cpu_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Ensure a config file is provided
if [ -z "$1" ]; then
    echo "Usage: sbatch job_train_cpu.sh <path_to_config.yaml>"
    exit 1
fi

MAIN_CONFIG="$1"

# Go to the main directory
cd /srv/data/mohammad.elaabaribao/work/interns/y2026/temp/ds_temp_intern/temp/era5_mswt/main || exit 1
mkdir -p logs

# Activate Conda environment
source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID (CPU)"
echo "Start time: $(date)"
echo "Config: $MAIN_CONFIG"
echo "======================================"

python3 -u train.py "$MAIN_CONFIG"

echo "======================================"
echo "Job completed."
echo "End time: $(date)"
echo "======================================"
