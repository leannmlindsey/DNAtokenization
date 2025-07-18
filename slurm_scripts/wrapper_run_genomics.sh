#!/bin/bash

# Choose one from below

## Hyena
## TODO: Download HF model from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen to ../outputs/hyena_hf/hyenadna-tiny-1k-seqlen
#LOG_DIR="../watch_folder/gb_cv5/hyena"
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
#LOG_DIR="../watch_folder/gb_cv10/mamba_char"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/pre_mamba_ntp_rc_aug_char_4k_d-256_n-4_lr-1e-2_bs-256/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/pre_mamba_ntp_rc_aug_char_4k_d-256_n-4_lr-1e-2_bs-256/checkpoints/last.ckpt")
#DISPLAY_NAME="mamba_char"
#LOG_DIR="../watch_folder/gb_cv5/mamba"
#MODEL="mamba"
#MODEL_NAME="dna_embedding_mamba"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "1e-3" "2e-3" "1e-2")

## Caduceus Parameter Sharing
#LOG_DIR="../watch_folder/gb_cv10/caduceus"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-3/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-3/checkpoints/last.ckpt")
#DISPLAY_NAME="caduceus_ps"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )
#LRS=("1e-4" "5e-4" "1e-5")

mkdir -p "${LOG_DIR}"
export_str="ALL,CONFIG_PATH=${CONFIG_PATH},PRETRAINED_PATH=${PRETRAINED_PATH},DISPLAY_NAME=${DISPLAY_NAME},MODEL=${MODEL},MODEL_NAME=${MODEL_NAME},CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER},CONJOIN_TEST=${CONJOIN_TEST}"
for TASK in "dummy_mouse_enhancers_ensembl" "drosophilia_enhancers" "demo_coding_vs_intergenomic_seqs" "demo_human_or_worm" "human_enhancers_cohn" "human_enhancers_ensembl" "human_ensembl_regulatory" "human_nontata_promoters" "human_ocr_ensembl"; do
#for TASK in "drosophilia_enhancers"; do
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 128 256; do
      for RC_AUG in "${RC_AUGS[@]}"; do
        export_str="${export_str},TASK=${TASK},LR=${LR},BATCH_SIZE=${BATCH_SIZE},RC_AUG=${RC_AUG}"
        job_name="gb_${TASK}_${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
        sbatch \
          --job-name="${job_name}" \
          --output="${LOG_DIR}/%x_%j.log" \
          --export="${export_str}" \
          "run_genomics_benchmark.sh"
      done
    done
  done
done
