#!/bin/bash

LOG_DIR="../watch_folder/ntv2_cv10/cnn_baseline"
mkdir -p "${LOG_DIR}"
export_str="ALL"
for TASK in "H2AFZ" "H3K27ac", "splice_sites_donors", "splice_sites_acceptors", "H3K27me3", "H3K36me3", "H3K4me1", "splice_sites_all", "H3K4me2", "H3K4me3", "enhancers_types", "promoter_no_tata", "H3K9ac", "H3K9me3", "promoter_tata", "H4K20me1", "promoter_all"; do    
#for TASK in "enhancers"; do
  for RC_AUG in "false"; do
    export_str="${export_str},TASK=${TASK},RC_AUG=${RC_AUG}"
    job_name="ntv2_${TASK}_CNN_RC_AUG-${RC_AUG}"
    sbatch \
      --job-name="${job_name}" \
      --output="${LOG_DIR}/%x_%j.log" \
      --export="${export_str}" \
      "run_ntv2_cnn.sh"
  done
done
