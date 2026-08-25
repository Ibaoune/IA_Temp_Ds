#!/bin/bash

# ==============================================================================
# Script: submit_unet_jobs.sh
# Description: Submits all UNet configuration files as SLURM jobs.
# Environment: clean_env_Pytorch
# ==============================================================================

# Ensure we are running this from the scripts directory or the main directory
cd "$(dirname "$0")" || exit 1

CONDA_ENV="clean_env_Pytorch"
CONFIGS_DIR="../configs/unet"
LOGS_DIR="../logs/unet"
TRAIN_SCRIPT="../train.py"

# Create logs directory if it doesn't exist
mkdir -p $LOGS_DIR

echo "Preparing to submit UNet jobs..."
echo "========================================="

# Find all config files
for config_file in $CONFIGS_DIR/*.yaml; do
    filename=$(basename -- "$config_file")
    
    job_name="unet_${filename%.*}"
    
    # Submit job using sbatch with heredoc
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
#SBATCH --time=24:00:00

# Initialize Conda (update path if your conda base differs)
source /srv/software/easybuild/software/Anaconda3/2020.11/etc/profile.d/conda.sh || true
conda activate $CONDA_ENV

# Change directory to the main project folder where train.py is located
cd ../

# Run training
echo "Starting training for $config_file"
python train.py configs/unet/$filename
echo "Finished training for $config_file"

# Run evaluation
echo "Starting evaluation for $config_file"
python eval.py configs/unet/$filename
echo "Finished evaluation for $config_file"
EOT
    
    echo "Submitted job: $job_name (Config: $filename)"
done

echo "========================================="
echo "All UNet jobs submitted successfully."
