#!/bin/bash

#SBATCH --job-name=emul_cpu
#SBATCH --output=out_%j.log
#SBATCH --error=out_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

########################################
# USER SHOULD SET THESE TO yes OR no
########################################
train="yes"
validation="yes"
########################################

# Activate Conda environment
conda activate clean_env_Pytorch

# Ensure Python logs are flushed immediately
export PYTHONUNBUFFERED=1

# Record start time
start_time=$(date +%s)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Train: $train | Validation: $validation"
echo "======================================"

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u ../train.py ../configs/unet/test.yaml
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u ../eval.py ../configs/unet/test.yaml
fi

if [[ "$train" != "yes" && "$validation" != "yes" ]]; then
    echo "[WARNING] Neither training nor validation selected."
    echo "Set train=\"yes\" and/or validation=\"yes\"."
fi

# Record end time
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"