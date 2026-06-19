#!/bin/bash
module load Anaconda3/2020.11
source activate clean_env_Pytorch
python run_all_postproc.py > run_all_postproc.log 2>&1
