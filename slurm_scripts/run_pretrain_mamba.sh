#!/bin/bash
#SBATCH --get-user-env                      # Retrieve the users login environment
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --qos=soc-gpu-np
#SBATCH -t 12:00:00                         # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:a6000:4                        # Number of GPUs
#SBATCH --ntasks-per-node=4                 # Should correspond to num devices (at least 1-1 task to GPU)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --exclude=notch501
#SBATCH --mem=0
#SBATCH -N 1                                # Number of nodes
#SBATCH --requeue                           # Requeue job if it fails
#SBATCH --job-name=char_mamba               # Job name
#SBATCH --output=../watch_folder/%x_%j.log  # Log file

echo "TIME: Start: = `date +"%Y-%m-%d %T"`"
module load cuda
nvidia-smi
source activate CADUCEUS_3

cd /uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN_REPEATED/caduceus

NUM_DEVICES=4

# Run script
SEQLEN=4096
MAX_STEPS=20000
D_MODEL=256
N_LAYER=4
LR=$1
RC_AUG="true"

BATCH_SIZE=$2
SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
WANDB_NAME="mamba_ntp_rc_aug_char_${SEQLEN_DIS}_d-${D_MODEL}_n-${N_LAYER}_lr-${LR}_bs-${BATCH_SIZE}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/pre_${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
srun python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.mlm=false \
  dataset.mlm_probability=0.0 \
  dataset.rc_aug="${RC_AUG}" \
  model=mamba \
  model.config.d_model=${D_MODEL} \
  model.config.n_layer=${N_LAYER} \
  optimizer.lr="${LR}" \
  train.global_batch_size=${BATCH_SIZE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  +trainer.val_check_interval=$(( MAX_STEPS / 5 )) \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}"
