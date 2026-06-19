#!/bin/bash

# ==============================================================================
# Script: submit_eval_jobs.sh
# Description: Submits ONLY evaluation for 16 specific pre-trained configurations
# ==============================================================================

cd "$(dirname "$0")" || exit 1

CONDA_ENV="clean_env_Pytorch"

# List of completed configs formatted as: directory/filename
CONFIGS=(
    "cnn/config.yaml"
    "cnn/config_bs32.yaml"
    "cnn/config_cnn1.yaml"
    "cnn/config_mse.yaml"
    "unet/unet1.yaml"
    "unet/unet1_loss_gaussian.yaml"
    "unet/unet1_loss_mse.yaml"
    "unet/config_unet1_bs32.yaml"
    "unet/config_unet1_mse.yaml"
    "unet/config_unet1_reg.yaml"
    "unet/unet_bs_64.yaml"
    "unet/unet_dropout_02.yaml"
    "unet/unet_lr_1e4.yaml"
    "unet/unet_lr_5e4.yaml"
    "unet/unet_no_scheduler.yaml"
    "unet/unet_weight_decay_1e3.yaml"
)

echo "Preparing to submit evaluation jobs..."
echo "========================================="

for config_rel_path in "${CONFIGS[@]}"; do
    model_type=$(dirname "$config_rel_path")
    filename=$(basename "$config_rel_path")
    job_name="eval_${filename%.*}"
    LOGS_DIR="../logs/${model_type}"
    mkdir -p "$LOGS_DIR"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$LOGS_DIR/%x_%j.out
#SBATCH --error=$LOGS_DIR/%x_%j.err
#SBATCH --account=climat-7ksifkvwkuy-default-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh || true
conda activate $CONDA_ENV

cd ../

echo "Starting evaluation for configs/$config_rel_path"
python eval.py configs/$config_rel_path
echo "Finished evaluation for configs/$config_rel_path"
EOT
    
    echo "Submitted job: $job_name (Config: $config_rel_path)"
done

echo "========================================="
echo "All evaluation jobs submitted successfully."
