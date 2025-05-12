#!/bin/bash

# Hyperparameter Sweep for Mamba DNA Model Training
# Usage: ./hp_sweep.sh

# Define sweep values
LEARNING_RATES=("8e-6" "1e-5" "3e-5" "5e-5")
#LEARNING_RATES=("8e-6" "1e-5" "3e-5")
#LEARNING_RATES=("8e-6" "5e-5")
BATCH_SIZES=(32 64 128)
#BATCH_SIZES=(256 512)
#LEARNING_RATES=("8e-3" "5e-3" "1e-3" "1e-2" "3e-2")
#BATCH_SIZES=(128 256)

script_dir="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/slurm_scripts"
echo "Starting hyperparameter sweep at $(date)" echo 
echo "Testing Learning Rates: ${LEARNING_RATES[*]}"
echo "Testing Batch Sizes: ${BATCH_SIZES[*]}"


for lr in "${LEARNING_RATES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
	    echo "Running $script_dir/run_pretrain_mamba.sh with lr=$lr batch_size=$batch_size"
	    sbatch $script_dir/run_pretrain_mamba.sh $lr $batch_size
    done
done

