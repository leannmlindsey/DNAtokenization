#!/bin/bash

# Choose one from below

# Hyena
#LOG_DIR="../watch_folder/ntv2_cv10/hyena"
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN/caduceus/outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-4_lr-6e-4/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN/caduceus/outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-4_lr-6e-4/checkpoints/last.ckpt"
#DISPLAY_NAME="hyena"
#MODEL="hyena"
#MODEL_NAME="dna_embedding"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "6e-3" "6e-4" "6e-5" )
#CSV_FILE=${CSV_FILE:-"best_hp_hyena_NTv2.csv"}

## Mamba NTP
#LOG_DIR="../watch_folder/ntv2_cv5/mamba"
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN_REPEATED/caduceus/outputs/pretrain/hg38/pre_mamba_ntp_rc_aug_char_4k_d-256_n-4_lr-1e-2_bs-256/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN_REPEATED/caduceus/outputs/pretrain/hg38/pre_mamba_ntp_rc_aug_char_4k_d-256_n-4_lr-1e-2_bs-256/checkpoints/last.ckpt"
#DISPLAY_NAME="mamba_uni_char"
#MODEL="mamba"
#MODEL_NAME="dna_embedding_mamba"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=("1e-4" "2e-4" "5e-5")
#CSV_FILE=${CSV_FILE:-"best_hp_mamba_char_NTv2.csv"}

## Caduceus Parameter Sharing
LOG_DIR="../watch_folder/ntv2_cv10/caduceus"
CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN_REPEATED/caduceus/outputs/pretrain/hg38/caduceus-ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-3/model_config.json"
PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN_REPEATED/caduceus/outputs/pretrain/hg38/caduceus-ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-3/checkpoints/last.ckpt"
DISPLAY_NAME="caduceus_ps"
MODEL="caduceus"
MODEL_NAME="dna_embedding_caduceus"
CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
CONJOIN_TEST="false"
RC_AUGS=( "false" )
LRS=("2e-3" "1e-3")
LRS=("1e-4" "5e-4" "1e-5")
CSV_FILE=${CSV_FILE:-"hp_round3/best_hp_caduceus_NTv2.csv"}

mkdir -p "${LOG_DIR}"
export_str="ALL,CONFIG_PATH=${CONFIG_PATH},PRETRAINED_PATH=${PRETRAINED_PATH},DISPLAY_NAME=${DISPLAY_NAME},MODEL=${MODEL},MODEL_NAME=${MODEL_NAME},CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER},CONJOIN_TEST=${CONJOIN_TEST}"

# Set to "true" for dry run (no actual job submission) or "false" for actual submission
DRY_RUN=${DRY_RUN:-"false"}

echo "== Job Submission Configuration =="
echo "CSV File: $CSV_FILE"
echo "Dry Run Mode: $DRY_RUN"
echo "==============================="

if [ ! -f "$CSV_FILE" ]; then
  echo "Error: CSV file $CSV_FILE not found!"
  exit 1
fi

# Count total number of jobs that will be submitted
TOTAL_JOBS=$(grep -v "Task" "$CSV_FILE" | wc -l)
echo "Total jobs to be submitted: $TOTAL_JOBS"

# Function to convert scientific notation from formats like 6.0e-04 to 6e-4
convert_lr_format() {
  local lr=$1
  # First check if it matches the 6.0e-03 pattern (with leading zeros in exponent)
  if [[ $lr =~ ([0-9]+)\.0e([-+])0*([1-9][0-9]*) ]]; then
    # Convert to the desired format without the .0 and removing leading zeros in exponent
    echo "${BASH_REMATCH[1]}e${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
  # Check if it matches the 6.0e-4 pattern (no leading zeros in exponent)
  elif [[ $lr =~ ([0-9]+)\.0e([-+][0-9]+) ]]; then
    # Convert to the desired format without the .0
    echo "${BASH_REMATCH[1]}e${BASH_REMATCH[2]}"
  else
    # Return the original value if it doesn't match any pattern
    echo "$lr"
  fi
}

# Read from the CSV file containing best hyperparameters
line_num=0
while IFS=, read -r col1 col2 col3 col4 col5 remainder || [ -n "$col1" ]; do
  line_num=$((line_num + 1))

  # Skip header if present
  if [[ "$col1" == "Task" || "$col1" == "TASK" || "$col1" == "task" ]]; then
    continue
  fi

  # Extract the required columns
  TASK=$(echo "$col1" | xargs)     # 1st column: Task
  MODEL=${MODEL}                    # Use the MODEL from environment
  # Get learning rate and convert format if needed
  LR_RAW=$(echo "$col4" | xargs)
  LR=$(convert_lr_format "$LR_RAW")
  BATCH_SIZE=$(echo "$col5" | xargs) # 5th column: Batch Size

  # Set default RC_AUG value
  RC_AUG=${RC_AUG:-"false"}

  # Create export string with the best hyperparameters
  current_export_str="${export_str},TASK=${TASK},LR=${LR},BATCH_SIZE=${BATCH_SIZE},RC_AUG=${RC_AUG}"

  # Create job name
  job_name="ntv2_${TASK}_${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"

  echo "Processing line $line_num: TASK=$TASK, LR=$LR, BATCH_SIZE=$BATCH_SIZE"

    if [ "$DRY_RUN" = "true" ]; then
      echo "[DRY RUN] Would submit job: ${job_name}"
      echo "[DRY RUN] Command: sbatch --job-name=\"${job_name}\" "
      echo "                          --output=\"${LOG_DIR}/%x_%j.log\" "
      echo "                          --export=\"${current_export_str}\" "
      echo "                          \"run_nucleotide_transformer_v2.sh\" "
    else
      echo "Submitting job: ${job_name}"
      sbatch \
        --job-name="${job_name}" \
        --output="${LOG_DIR}/%x_%j.log" \
        --export="${current_export_str}" \
        "run_nucleotide_transformer_v2.sh"
    fi

done < "$CSV_FILE"

echo "=== Completed ==="
if [ "$DRY_RUN" = "true" ]; then
  echo "This was a dry run. No jobs were actually submitted."
  echo "To submit jobs for real, run with: DRY_RUN=false"
fi
