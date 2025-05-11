#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 12:00:00
#SBATCH --mail-user=leann.lindsey@utah.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name=gpt_ntv2
#SBATCH --gpus=v100-32:1
#SBATCH -o /ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/outerror/%x%j.outerror

# Load Modules 
module load anaconda3/2024.10-1
nvidia-smi

echo "starting DNABERT env on conda"
source activate dna_sandbox
conda list
script_dir="/ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/MODELS/DNABERT_2/finetune"
output_path="/ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/RESULTS/GPT"

#data_path=$1
#lr=3e-5
lr=$1
seed=$2
kmer=6
data_path="/ocean/projects/bio230026p/lindseyl/TOKENIZATION_FINAL_PAPER/DATA"
echo "The provided data_path is $data_path"
echo "The learning rate is $lr"
echo "The seed is $seed"
echo "The output path is $output_path"

# Define a function to get the appropriate max_length for each dataset
get_max_length() {
    local dataset=$1

    case $dataset in
        # Promoter datasets - 300bp
        promoter_all|promoter_tata|promoter_no_tata)
            echo 300
            ;;
        # Enhancer datasets - 200bp
        enhancers|enhancers_types)
            echo 200
            ;;
        # Splice sites datasets
        splice_sites_all)
            echo 400
            ;;
        splice_sites_acceptors|splice_sites_donors)
            echo 600
            ;;
        # Histone modification datasets - 500bp
        H3K*|H4K*|H2AFZ)
            echo 500
            ;;
        # Default case if dataset not recognized
        *)
            echo 128  # Default to original value
            ;;
    esac
}
cd $script_dir

echo "The provided kmer is: $kmer, data_path is $data_path"

for seed in $seed
do
     for data in H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1 enhancers enhancers_types promoter_all promoter_no_tata promoter_tata splice_sites_acceptors splice_sites_all splice_sites_donors    
     do
	# Get the appropriate max_length for this dataset
        max_length=$(get_max_length $data)

        echo "Training on dataset: $data with max_length: $max_length"

    	python train_gpt.py \
          --model_name_or_path EleutherAI/gpt-neo-125m \
          --data_path ${data_path}/NTv2/$data \
          --kmer -1 \
          --run_name GTP_${lr}_${data}_seed${seed} \
          --model_max_length 500 \
          --per_device_train_batch_size 8 \
          --per_device_eval_batch_size 16 \
          --gradient_accumulation_steps 1 \
          --learning_rate ${lr} \
          --num_train_epochs 3 \
          --fp16 \
          --save_steps 200 \
          --output_dir ${output_path}/GTP_${lr}_${data}_seed${seed} \
          --evaluation_strategy steps \
          --eval_steps 200 \
          --warmup_steps 50 \
          --logging_steps 100000 \
          --overwrite_output_dir True \
          --log_level info \
          --find_unused_parameters False
                     
     done
done

