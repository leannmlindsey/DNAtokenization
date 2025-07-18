#!/bin/bash

# Choose one from below
## Hyena
#LOG_DIR="../watch_folder/ntv2_cv10/hyena"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-4_lr-6e-4/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-4_lr-6e-4/checkpoints/last.ckpt")
#DISPLAY_NAME="hyena"
#MODEL="hyena"
#MODEL_NAME="dna_embedding"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "6e-3" "6e-4" "6e-5" )

## Mamba NTP
#LOG_DIR="../watch_folder/ntv2_cv5/mamba"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/pre_mamba_ntp_rc_aug_char_4k_d-256_n-4_lr-1e-2_bs-256/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/pre_mamba_ntp_rc_aug_char_4k_d-256_n-4_lr-1e-2_bs-256/checkpoints/last.ckpt")
#DISPLAY_NAME="mamba_char"
#MODEL="mamba"
#MODEL_NAME="dna_embedding_mamba"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=("1e-4" "3e-4" "5e-4")

## Caduceus Parameter Sharing
#LOG_DIR="../watch_folder/ntv2_cv10/caduceus"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-3/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-3/checkpoints/last.ckpt")
#DISPLAY_NAME="caduceus_ps"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )
#LRS=("8e-3" "1e-3" "1e-4" "1e-5" )
#LRS=("1e-3" "2e-3")

BATCH=$2
TASK_LIST=$3
mkdir -p "${LOG_DIR}"
export_str="ALL,CONFIG_PATH=${CONFIG_PATH},PRETRAINED_PATH=${PRETRAINED_PATH},DISPLAY_NAME=${DISPLAY_NAME},MODEL=${MODEL},MODEL_NAME=${MODEL_NAME},CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER},CONJOIN_TEST=${CONJOIN_TEST}"
for TASK in "H2AFZ" "H3K27ac", "splice_sites_donors", "splice_sites_acceptors", "H3K27me3", "H3K36me3", "H3K4me1", "splice_sites_all", "enhancers" "H3K4me2", "H3K4me3", "enhancers_types", "promoter_no_tata", "H3K9ac", "H3K9me3", "promoter_tata", "H4K20me1", "promoter_all"; do  
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 128 256; do
    for RC_AUG in "${RC_AUGS[@]}"; do
        export_str="${export_str},TASK=${TASK},LR=${LR},BATCH_SIZE=${BATCH_SIZE},RC_AUG=${RC_AUG}"
        job_name="nt_${TASK}_${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
        sbatch \
          --job-name="${job_name}" \
          --output="${LOG_DIR}/%x_%j.log" \
          --export="${export_str}" \
          "run_nucleotide_transformer_v2.sh"
      done
    done
  done
done
