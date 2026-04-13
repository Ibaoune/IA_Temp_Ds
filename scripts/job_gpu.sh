#!/bin/bash

#SBATCH --job-name=emul_gpu
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

#SBATCH --output=out_%j.log
#SBATCH --error=out_%j.log

########################################
# USER SHOULD SET THESE TO yes OR no
########################################
train="yes"        # yes = activate training
validation="yes"   # yes = activate validation
########################################

# Activer l'environnement Conda
source ~/.bashrc
conda activate clean_env_Pytorch

# Afficher les infos GPU (utile pour debug)
echo "Allocated GPU(s):"
nvidia-smi

# Enregistrer le début
start_time=$(date +%s)

# (Optionnel) monitoring GPU en arrière-plan
gpu_log="gpu_usage_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
           --format=csv,nounits,noheader \
           --loop=60 > "$gpu_log" &

########################################
# Run scripts
########################################

if [[ "$train" == "yes" ]]; then
    echo "Running training on GPU..."
    python3 -u train.py
fi

if [[ "$validation" == "yes" ]]; then
    echo "Running validation on GPU..."
    python3 -u eval.py
fi

if [[ "$train" != "yes" && "$validation" != "yes" ]]; then
    echo "Neither training nor validation selected."
fi

########################################
# Fin du job
########################################
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."

